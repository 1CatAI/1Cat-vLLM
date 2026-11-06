// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Two-rank FP16 all-reduce through pinned host memory, for GPU pairs without
// peer access (PCIe boxes with P2P disabled, KVM guests, mismatched boards).
// NCCL's LL protocol costs tens of microseconds for the small decode
// all-reduces there, and the custom all-reduce needs P2P/IPC. One kernel does
// the whole exchange: each block writes its chunk to a host slot, fences,
// raises a per-block flag, spins on the peer's flag, then reads the peer's
// chunk straight from host memory and adds it. No CPU work and no stream
// synchronization, so it can be captured in CUDA graphs (the call counter
// lives in device memory and the last block advances it).
//
// Numerics: out = mine + peer in FP16, one rounding and commutative, so both
// ranks produce identical bits, the same as NCCL's two-rank FP16 sum.
//
// Slot reuse: double buffering by call parity. Before call n+2 overwrites
// slot (n & 1), this rank has observed the peer's flag for call n+1, which
// the peer raises only after it finished call n on its own stream.
//
// Prefill-sized payloads take the copy-engine mode instead: the copy engines
// move this rank's tensor into its host slot and the peer's back, chunk by
// chunk, and stream memory operations (cuStreamWriteValue32 after a chunk
// landed, cuStreamWaitValue32 before reading the peer's) order the exchange.
// No SM spins on host memory, so computation on other streams runs at full
// speed while the transfer is in flight, which SM copies through host memory
// (this kernel, NCCL's LL protocol) prevent. Same sum, same bits. The call
// sequence lives on the host, so this mode is for eager calls only.
//
// The exchange kernel is adapted from kernels/skinny_ar.cu of
// mzen17/v100-skinny-unify (https://github.com/mzen17/v100-skinny-unify),
// which carries this notice:
//
//   MIT License. Copyright (c) 2026 v100-skinny contributors
//
//   Permission is hereby granted, free of charge, to any person obtaining a
//   copy of this software and associated documentation files (the
//   "Software"), to deal in the Software without restriction, including
//   without limitation the rights to use, copy, modify, merge, publish,
//   distribute, sublicense, and/or sell copies of the Software, and to permit
//   persons to whom the Software is furnished to do so, subject to the
//   following conditions:
//
//   The above copyright notice and this permission notice shall be included
//   in all copies or substantial portions of the Software.
//
//   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
//   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
//   NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
//   DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
//   OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
//   USE OR OTHER DEALINGS IN THE SOFTWARE.
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <torch/all.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace {

constexpr int kMaxBlocks = 16;
constexpr int kMaxChunks = 64;
constexpr int kThreads = 256;
constexpr int kFlagStride = 32;  // one 128-byte cache line per flag
// flags: [parity 2][rank 2][kMaxChunks] * kFlagStride uints
constexpr size_t kFlagsBytes =
    2ull * 2 * kMaxChunks * kFlagStride * sizeof(unsigned);
// Copy-engine mode: up to kCopyChunks pieces of at least kMinCopyChunkBytes,
// so the peer copies a piece back while the next one is still going out.
constexpr int kCopyChunks = 16;
constexpr int64_t kMinCopyChunkBytes = 256 * 1024;
// copy flags: [rank 2][kCopyChunks piece flags, 1 read acknowledgement]
constexpr int kCopyAck = kCopyChunks;
constexpr size_t kCopyFlagsBytes =
    2ull * (kCopyChunks + 1) * kFlagStride * sizeof(unsigned);

struct Channel {
  void* host = nullptr;
  void* dev = nullptr;
  size_t map_bytes = 0;
  size_t slot_bytes = 0;
  unsigned* counters = nullptr;  // [0] call sequence, [1] finished blocks
  int rank = -1;
  int device = -1;
  // Copy-engine mode, behind the kernel mode's flags and slots in the mapping.
  size_t copy_offset = 0;
  size_t copy_slot_bytes = 0;  // 0: mode off
  unsigned copy_seq = 0;
  cudaStream_t copy_stream = nullptr;
  cudaEvent_t input_ready = nullptr;
  cudaEvent_t input_sent = nullptr;
};

std::mutex channels_mutex;
std::vector<std::unique_ptr<Channel>> channels;

Channel& channel(int64_t handle) {
  std::lock_guard<std::mutex> lock(channels_mutex);
  TORCH_CHECK(handle >= 0 && handle < static_cast<int64_t>(channels.size()) &&
                  channels[handle],
              "host_reduce: unknown handle ", handle);
  return *channels[handle];
}

__device__ __forceinline__ int4 load_volatile(const int4* p) {
  int4 v;
  asm volatile("ld.volatile.global.v4.s32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

__device__ __forceinline__ int4 add_half8(int4 a, int4 b) {
  const half2* pa = reinterpret_cast<const half2*>(&a);
  const half2* pb = reinterpret_cast<const half2*>(&b);
  int4 o;
  half2* po = reinterpret_cast<half2*>(&o);
#pragma unroll
  for (int i = 0; i < 4; i++) po[i] = __hadd2(pa[i], pb[i]);
  return o;
}

// x -> out over n8 groups of eight halves.
__global__ void __launch_bounds__(kThreads)
    host_reduce_kernel(const int4* __restrict__ x, int4* __restrict__ out,
                       int n8, unsigned* flags, int4* data, int rank,
                       int slot_int4, unsigned* counters) {
  __shared__ unsigned s_seq;
  if (threadIdx.x == 0) {
    s_seq = *reinterpret_cast<volatile unsigned*>(counters) + 1u;
  }
  __syncthreads();
  const unsigned seq = s_seq;
  const int parity = seq & 1u;
  const int peer = rank ^ 1;
  int4* my_slot = data + static_cast<size_t>(parity * 2 + rank) * slot_int4;
  const int4* peer_slot =
      data + static_cast<size_t>(parity * 2 + peer) * slot_int4;
  volatile unsigned* my_flag =
      flags +
      (static_cast<size_t>(parity * 2 + rank) * kMaxChunks + blockIdx.x) *
          kFlagStride;
  volatile unsigned* peer_flag =
      flags +
      (static_cast<size_t>(parity * 2 + peer) * kMaxChunks + blockIdx.x) *
          kFlagStride;

  const int per_block = (n8 + gridDim.x - 1) / gridDim.x;
  const int begin = blockIdx.x * per_block;
  const int end = min(n8, begin + per_block);

  // 1. Publish this rank's chunk (posted PCIe writes), fence, raise the flag.
  for (int i = begin + threadIdx.x; i < end; i += kThreads) my_slot[i] = x[i];
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    *my_flag = seq;
    // 2. Wait for the peer's chunk.
    while (*peer_flag != seq) {
    }
  }
  __syncthreads();
  // 3. Read the peer's chunk from host memory and add.
  for (int i = begin + threadIdx.x; i < end; i += kThreads) {
    out[i] = add_half8(x[i], load_volatile(peer_slot + i));
  }
  // 4. The last block to finish advances the call counter.
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned finished = atomicAdd(counters + 1, 1u);
    if (finished == gridDim.x - 1) {
      counters[1] = 0;
      __threadfence();
      *reinterpret_cast<volatile unsigned*>(counters) = seq;
    }
  }
}

__global__ void __launch_bounds__(kThreads)
    add_half8_kernel(const int4* __restrict__ a, const int4* __restrict__ b,
                     int4* __restrict__ out, int n8) {
  for (int i = blockIdx.x * kThreads + threadIdx.x; i < n8;
       i += gridDim.x * kThreads) {
    out[i] = add_half8(a[i], b[i]);
  }
}

void check_cu(CUresult result, const char* what) {
  if (result != CUDA_SUCCESS) {
    const char* message = nullptr;
    cuGetErrorString(result, &message);
    TORCH_CHECK(false, "host_reduce: ", what,
                " failed: ", message ? message : "unknown error");
  }
}

// Map the shared file both ranks opened, clear this rank's flags and slots so
// a stale mapping cannot satisfy a wait, and register it with CUDA. The caller
// barriers on both ranks before the first all-reduce.
int64_t open_channel(const std::string& path, int64_t rank, int64_t max_bytes,
                     int64_t copy_engine_max_bytes) {
  TORCH_CHECK(rank == 0 || rank == 1, "host_reduce supports exactly two ranks");
  TORCH_CHECK(max_bytes > 0 && max_bytes % 16 == 0 && max_bytes <= (1 << 22),
              "host_reduce: invalid max_bytes ", max_bytes);
  TORCH_CHECK(copy_engine_max_bytes >= 0 && copy_engine_max_bytes % 16 == 0,
              "host_reduce: invalid copy_engine_max_bytes ",
              copy_engine_max_bytes);
  auto ch = std::make_unique<Channel>();
  ch->rank = static_cast<int>(rank);
  ch->slot_bytes = static_cast<size_t>(max_bytes);
  ch->copy_slot_bytes = static_cast<size_t>(copy_engine_max_bytes);
  ch->copy_offset = kFlagsBytes + 4 * ch->slot_bytes;
  const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  const size_t copy_bytes =
      ch->copy_slot_bytes ? kCopyFlagsBytes + 4 * ch->copy_slot_bytes : 0;
  ch->map_bytes = (ch->copy_offset + copy_bytes + page - 1) / page * page;
  const int fd = open(path.c_str(), O_RDWR | O_CREAT, 0600);
  TORCH_CHECK(fd >= 0, "host_reduce: cannot open ", path, ": ",
              std::strerror(errno));
  const bool sized = ftruncate(fd, static_cast<off_t>(ch->map_bytes)) == 0;
  void* host = sized ? mmap(nullptr, ch->map_bytes, PROT_READ | PROT_WRITE,
                            MAP_SHARED, fd, 0)
                     : MAP_FAILED;
  const int error = errno;
  close(fd);
  TORCH_CHECK(sized, "host_reduce: ftruncate failed: ", std::strerror(error));
  TORCH_CHECK(host != MAP_FAILED,
              "host_reduce: mmap failed: ", std::strerror(error));
  ch->host = host;
  unsigned* flags = static_cast<unsigned*>(host);
  for (int parity = 0; parity < 2; parity++) {
    std::memset(
        flags +
            static_cast<size_t>(parity * 2 + rank) * kMaxChunks * kFlagStride,
        0, static_cast<size_t>(kMaxChunks) * kFlagStride * sizeof(unsigned));
    std::memset(static_cast<char*>(host) + kFlagsBytes +
                    static_cast<size_t>(parity * 2 + rank) * ch->slot_bytes,
                0, ch->slot_bytes);
  }
  if (ch->copy_slot_bytes) {
    std::memset(static_cast<char*>(host) + ch->copy_offset +
                    static_cast<size_t>(rank) * (kCopyChunks + 1) *
                        kFlagStride * sizeof(unsigned),
                0, (kCopyChunks + 1) * kFlagStride * sizeof(unsigned));
  }
  C10_CUDA_CHECK(cudaGetDevice(&ch->device));
  C10_CUDA_CHECK(cudaHostRegister(host, ch->map_bytes, cudaHostRegisterMapped));
  C10_CUDA_CHECK(cudaHostGetDevicePointer(&ch->dev, host, 0));
  C10_CUDA_CHECK(cudaMalloc(&ch->counters, 2 * sizeof(unsigned)));
  C10_CUDA_CHECK(cudaMemset(ch->counters, 0, 2 * sizeof(unsigned)));
  if (ch->copy_slot_bytes) {
    C10_CUDA_CHECK(
        cudaStreamCreateWithFlags(&ch->copy_stream, cudaStreamNonBlocking));
    C10_CUDA_CHECK(
        cudaEventCreateWithFlags(&ch->input_ready, cudaEventDisableTiming));
    C10_CUDA_CHECK(
        cudaEventCreateWithFlags(&ch->input_sent, cudaEventDisableTiming));
  }
  std::lock_guard<std::mutex> lock(channels_mutex);
  channels.push_back(std::move(ch));
  return static_cast<int64_t>(channels.size()) - 1;
}

void close_channel(int64_t handle) {
  std::unique_ptr<Channel> ch;
  {
    std::lock_guard<std::mutex> lock(channels_mutex);
    TORCH_CHECK(handle >= 0 && handle < static_cast<int64_t>(channels.size()),
                "host_reduce: unknown handle ", handle);
    ch = std::move(channels[handle]);
  }
  if (!ch) return;
  const c10::cuda::CUDAGuard guard(ch->device);
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  if (ch->copy_stream) {
    C10_CUDA_CHECK(cudaEventDestroy(ch->input_sent));
    C10_CUDA_CHECK(cudaEventDestroy(ch->input_ready));
    C10_CUDA_CHECK(cudaStreamDestroy(ch->copy_stream));
  }
  C10_CUDA_CHECK(cudaHostUnregister(ch->host));
  munmap(ch->host, ch->map_bytes);
  C10_CUDA_CHECK(cudaFree(ch->counters));
}

void host_reduce_out(torch::Tensor& output, const torch::Tensor& input,
                     int64_t handle) {
  Channel& ch = channel(handle);
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kHalf &&
                  input.is_contiguous(),
              "host_reduce: contiguous FP16 CUDA input required");
  TORCH_CHECK(output.is_contiguous() && output.sizes() == input.sizes() &&
                  output.scalar_type() == at::kHalf &&
                  output.device() == input.device(),
              "host_reduce: output must match input");
  const int64_t bytes = input.numel() * 2;
  TORCH_CHECK(bytes > 0 && bytes % 16 == 0 &&
                  bytes <= static_cast<int64_t>(ch.slot_bytes),
              "host_reduce: payload of ", bytes, " bytes outside the channel");
  TORCH_CHECK((reinterpret_cast<uintptr_t>(input.data_ptr()) & 15) == 0 &&
                  (reinterpret_cast<uintptr_t>(output.data_ptr()) & 15) == 0,
              "host_reduce: 16-byte alignment required");
  const c10::cuda::CUDAGuard guard(input.device());
  const int n8 = static_cast<int>(bytes / 16);
  const int blocks =
      std::max(1, std::min(kMaxBlocks, (n8 + kThreads - 1) / kThreads));
  unsigned* flags = static_cast<unsigned*>(ch.dev);
  int4* data =
      reinterpret_cast<int4*>(static_cast<char*>(ch.dev) + kFlagsBytes);
  host_reduce_kernel<<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const int4*>(input.data_ptr()),
      reinterpret_cast<int4*>(output.data_ptr()), n8, flags, data, ch.rank,
      static_cast<int>(ch.slot_bytes / 16), ch.counters);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void host_reduce_copy_engine_out(torch::Tensor& output,
                                 const torch::Tensor& input, int64_t handle) {
  Channel& ch = channel(handle);
  TORCH_CHECK(ch.copy_slot_bytes > 0,
              "host_reduce: the channel has no copy-engine mode");
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kHalf &&
                  input.is_contiguous(),
              "host_reduce: contiguous FP16 CUDA input required");
  TORCH_CHECK(output.is_contiguous() && output.sizes() == input.sizes() &&
                  output.scalar_type() == at::kHalf &&
                  output.device() == input.device(),
              "host_reduce: output must match input");
  const int64_t bytes = input.numel() * 2;
  TORCH_CHECK(bytes > 0 && bytes % 16 == 0 &&
                  bytes <= static_cast<int64_t>(ch.copy_slot_bytes),
              "host_reduce: payload of ", bytes,
              " bytes outside the copy-engine channel");
  TORCH_CHECK((reinterpret_cast<uintptr_t>(input.data_ptr()) & 15) == 0 &&
                  (reinterpret_cast<uintptr_t>(output.data_ptr()) & 15) == 0,
              "host_reduce: 16-byte alignment required");
  // The call sequence lives on the host; a graph replay would reuse it.
  TORCH_CHECK(at::cuda::currentStreamCaptureStatusMayInitCtx() ==
                  at::cuda::CaptureStatus::None,
              "host_reduce: the copy-engine mode cannot be captured");
  const c10::cuda::CUDAGuard guard(input.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const unsigned seq = ++ch.copy_seq;
  const int parity = seq & 1u;
  const int peer = ch.rank ^ 1;
  char* host_copy = static_cast<char*>(ch.host) + ch.copy_offset;
  const CUdeviceptr flags_dev = reinterpret_cast<CUdeviceptr>(
      static_cast<char*>(ch.dev) + ch.copy_offset);
  const auto flag = [&](int rank, int index) {
    return flags_dev + (static_cast<size_t>(rank) * (kCopyChunks + 1) + index) *
                           kFlagStride * sizeof(unsigned);
  };
  char* slots = host_copy + kCopyFlagsBytes;
  char* my_slot =
      slots + static_cast<size_t>(parity * 2 + ch.rank) * ch.copy_slot_bytes;
  const char* peer_slot =
      slots + static_cast<size_t>(parity * 2 + peer) * ch.copy_slot_bytes;
  const int chunks = static_cast<int>(
      std::clamp<int64_t>(bytes / kMinCopyChunkBytes, 1, kCopyChunks));
  // Pieces of at least kMinCopyChunkBytes, rounded to 16 bytes: none is empty.
  const int64_t chunk_bytes = (bytes / chunks + 15) / 16 * 16;
  const auto chunk_size = [&](int c) {
    return std::min(chunk_bytes, bytes - c * chunk_bytes);
  };
  const CUstream send = reinterpret_cast<CUstream>(ch.copy_stream);
  const CUstream receive = reinterpret_cast<CUstream>(stream);
  const char* in = static_cast<const char*>(input.data_ptr());

  // 1. Send on the channel's copy stream once the input is ready. The peer
  //    must have read this slot's previous content (two calls ago) first.
  C10_CUDA_CHECK(cudaEventRecord(ch.input_ready, stream));
  C10_CUDA_CHECK(cudaStreamWaitEvent(ch.copy_stream, ch.input_ready, 0));
  if (seq > 2) {
    check_cu(cuStreamWaitValue32(send, flag(peer, kCopyAck), seq - 2,
                                 CU_STREAM_WAIT_VALUE_GEQ),
             "wait for the peer's read acknowledgement");
  }
  for (int c = 0; c < chunks; c++) {
    C10_CUDA_CHECK(cudaMemcpyAsync(my_slot + c * chunk_bytes,
                                   in + c * chunk_bytes, chunk_size(c),
                                   cudaMemcpyDeviceToHost, ch.copy_stream));
    // Default write: ordered after the copy, so the piece is in host memory.
    check_cu(cuStreamWriteValue32(send, flag(ch.rank, c), seq,
                                  CU_STREAM_WRITE_VALUE_DEFAULT),
             "raise a piece flag");
  }
  C10_CUDA_CHECK(cudaEventRecord(ch.input_sent, ch.copy_stream));

  // 2. Receive on the caller's stream: each piece once the peer flagged it,
  //    then acknowledge that the peer's slot may be reused.
  torch::Tensor peer_input = torch::empty_like(input);
  char* peer_dev = static_cast<char*>(peer_input.data_ptr());
  for (int c = 0; c < chunks; c++) {
    check_cu(cuStreamWaitValue32(receive, flag(peer, c), seq,
                                 CU_STREAM_WAIT_VALUE_GEQ),
             "wait for a peer piece");
    C10_CUDA_CHECK(cudaMemcpyAsync(peer_dev + c * chunk_bytes,
                                   peer_slot + c * chunk_bytes, chunk_size(c),
                                   cudaMemcpyHostToDevice, stream));
  }
  check_cu(cuStreamWriteValue32(receive, flag(ch.rank, kCopyAck), seq,
                                CU_STREAM_WRITE_VALUE_DEFAULT),
           "acknowledge the read");

  // 3. Sum in FP16, one rounding: the same bits as the kernel mode and NCCL.
  const int n8 = static_cast<int>(bytes / 16);
  const int blocks =
      std::max(1, std::min(1024, (n8 + kThreads - 1) / kThreads));
  add_half8_kernel<<<blocks, kThreads, 0, stream>>>(
      reinterpret_cast<const int4*>(input.data_ptr()),
      reinterpret_cast<const int4*>(peer_input.data_ptr()),
      reinterpret_cast<int4*>(output.data_ptr()), n8);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // The caller may reuse the input once its stream passes this point.
  C10_CUDA_CHECK(cudaStreamWaitEvent(stream, ch.input_sent, 0));
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "host_reduce_open(str path, int rank, int max_bytes, int "
      "copy_engine_max_bytes) -> int",
      &open_channel);
  m.def("host_reduce_close(int handle) -> ()", &close_channel);
  m.def(
      "host_reduce_out(Tensor(a!) output, Tensor input, int handle) -> "
      "()");
  m.def(
      "host_reduce_copy_engine_out(Tensor(a!) output, Tensor input, int "
      "handle) -> ()");
}
TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("host_reduce_out", &host_reduce_out);
  m.impl("host_reduce_copy_engine_out", &host_reduce_copy_engine_out);
}
