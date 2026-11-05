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
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
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

struct Channel {
  void* host = nullptr;
  void* dev = nullptr;
  size_t map_bytes = 0;
  size_t slot_bytes = 0;
  unsigned* counters = nullptr;  // [0] call sequence, [1] finished blocks
  int rank = -1;
  int device = -1;
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

// Map the shared file both ranks opened, clear this rank's flags and slots so
// a stale mapping cannot satisfy a wait, and register it with CUDA. The caller
// barriers on both ranks before the first all-reduce.
int64_t open_channel(const std::string& path, int64_t rank, int64_t max_bytes) {
  TORCH_CHECK(rank == 0 || rank == 1, "host_reduce supports exactly two ranks");
  TORCH_CHECK(max_bytes > 0 && max_bytes % 16 == 0 && max_bytes <= (1 << 22),
              "host_reduce: invalid max_bytes ", max_bytes);
  auto ch = std::make_unique<Channel>();
  ch->rank = static_cast<int>(rank);
  ch->slot_bytes = static_cast<size_t>(max_bytes);
  const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  ch->map_bytes = (kFlagsBytes + 4 * ch->slot_bytes + page - 1) / page * page;
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
  C10_CUDA_CHECK(cudaGetDevice(&ch->device));
  C10_CUDA_CHECK(cudaHostRegister(host, ch->map_bytes, cudaHostRegisterMapped));
  C10_CUDA_CHECK(cudaHostGetDevicePointer(&ch->dev, host, 0));
  C10_CUDA_CHECK(cudaMalloc(&ch->counters, 2 * sizeof(unsigned)));
  C10_CUDA_CHECK(cudaMemset(ch->counters, 0, 2 * sizeof(unsigned)));
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

}  // namespace

TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def("host_reduce_open(str path, int rank, int max_bytes) -> int",
        &open_channel);
  m.def("host_reduce_close(int handle) -> ()", &close_channel);
  m.def(
      "host_reduce_out(Tensor(a!) output, Tensor input, int handle) -> "
      "()");
}
TORCH_LIBRARY_IMPL(_C, CUDA, m) { m.impl("host_reduce_out", &host_reduce_out); }
