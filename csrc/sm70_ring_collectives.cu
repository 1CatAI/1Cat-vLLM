// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Research-only verified-edge transport; no production dispatch changes.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/all.h>

#include <algorithm>
#include <climits>

namespace {
struct RankOrder {
  int ranks[8];
};
struct PeerOrder {
  int first, second, third;
};

// FP16 inputs and sums of at most eight FP16 values are zero, nonfinite,
// or have FP32 exponents in [-24,18]. Renumber that exponent without changing
// sign or mantissa. Two exact 30-bit values and a two-bit generation fit in
// one naturally aligned 64-bit atomic object. No sum narrows to FP16.
__device__ __forceinline__ uint32_t encode_partial(float x) {
  const uint32_t bits = __float_as_uint(x);
  const uint32_t exponent = (bits >> 23) & 255u;
  const uint32_t code = exponent == 0u     ? 0u
                        : exponent == 255u ? 63u
                                           : exponent - 102u;
  return (bits & 0x7fffffu) | ((bits >> 31) << 23) | (code << 24);
}

__device__ __forceinline__ float decode_partial(uint32_t bits) {
  const uint32_t code = (bits >> 24) & 63u;
  const uint32_t exponent = code == 0u ? 0u : code == 63u ? 255u : code + 102u;
  return __uint_as_float((bits & 0x7fffffu) | (((bits >> 23) & 1u) << 31) |
                         (exponent << 23));
}

__device__ __forceinline__ void publish_packet(uint64_t* ptr, float x, float y,
                                               uint32_t tag) {
  const uint64_t packet = uint64_t(encode_partial(x)) |
                          (uint64_t(encode_partial(y)) << 30) |
                          (uint64_t(tag) << 60);
  asm volatile("st.relaxed.sys.global.u64 [%0], %1;"
               :
               : "l"(ptr), "l"(packet)
               : "memory");
}

__device__ __forceinline__ uint64_t receive_packet(uint64_t* ptr,
                                                   uint32_t tag) {
  uint64_t packet;
  do {
    asm volatile("ld.relaxed.sys.global.u64 %0, [%1];"
                 : "=l"(packet)
                 : "l"(ptr)
                 : "memory");
  } while (uint32_t(packet >> 60) != tag);
  return packet;
}

// Every stage exchanges with its direct NVLink neighbor. Double buffering
// separates consecutive generations; each lane must consume a peer packet
// before it can advance to the following generation. Intermediate sums never
// narrow to FP16. Inactive pairs retain independent generation counters.
template <int Stages>
__global__ __launch_bounds__(128, 4) void cube_allreduce_kernel(
    half* output, const half* input, const int64_t* addresses,
    uint32_t* counters, PeerOrder peers, int logical_rank, int rank,
    int capacity, int size) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int packs = (size + 1) / 2;
  if (index < packs) {
    const uint32_t epoch = counters[index], tag = (epoch + 1u) & 3u;
    float x = __half2float(input[index * 2]);
    float y = index * 2 + 1 < size ? __half2float(input[index * 2 + 1]) : 0.0f;
#pragma unroll
    for (int stage = 0; stage < Stages; ++stage) {
      const int peer = stage == 0   ? peers.first
                       : stage == 1 ? peers.second
                                    : peers.third;
      const size_t offset =
          ((epoch & 1u) * Stages + stage) * static_cast<size_t>(capacity) +
          index;
      auto* destination = reinterpret_cast<uint64_t*>(addresses[peer]) + offset;
      auto* source = reinterpret_cast<uint64_t*>(addresses[rank]) + offset;
      publish_packet(destination, x, y, tag);
      const uint64_t received = receive_packet(source, tag);
      const float rx = decode_partial(uint32_t(received));
      const float ry = decode_partial(uint32_t(received >> 30));
      // The same logical-rank order produces the same association everywhere.
      if (logical_rank & (1 << stage)) {
        x = __fadd_rn(rx, x);
        y = __fadd_rn(ry, y);
      } else {
        x = __fadd_rn(x, rx);
        y = __fadd_rn(y, ry);
      }
    }
    output[index * 2] = __float2half_rn(x);
    if (index * 2 + 1 < size) output[index * 2 + 1] = __float2half_rn(y);
    counters[index] = epoch + 1u;
  }
}

void cube_allreduce(torch::Tensor output, torch::Tensor input,
                    torch::Tensor addresses, torch::Tensor counters,
                    const std::vector<int64_t>& rank_order, int64_t rank,
                    int64_t capacity, bool block_packets) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kHalf &&
                  input.is_contiguous() && input.numel() > 0 &&
                  input.numel() <= INT_MAX - 1,
              "Cube allreduce requires a nonempty contiguous CUDA FP16 input");
  const c10::cuda::CUDAGuard guard(input.device());
  const auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props->major == 7 && props->minor == 0, "SM70 required");
  const int world = rank_order.size();
  TORCH_CHECK((world == 2 || world == 4 || world == 8) && rank >= 0 &&
                  rank < world && capacity >= (input.numel() + 1) / 2 &&
                  capacity <= INT_MAX,
              "Cube allreduce requires 2/4/8 peers and sufficient capacity");
  RankOrder order{};
  int mask = 0, logical_rank = -1;
  for (int i = 0; i < world; ++i) {
    TORCH_CHECK(rank_order[i] >= 0 && rank_order[i] < world,
                "Rank order must be a permutation");
    mask |= 1 << rank_order[i];
    order.ranks[i] = rank_order[i];
    if (rank_order[i] == rank) logical_rank = i;
  }
  TORCH_CHECK(mask == (1 << world) - 1, "Rank order must be a permutation");
  TORCH_CHECK(output.is_cuda() && output.device() == input.device() &&
                  output.scalar_type() == at::kHalf && output.is_contiguous() &&
                  output.sizes() == input.sizes(),
              "Invalid cube allreduce output");
  TORCH_CHECK(addresses.is_cuda() && addresses.device() == input.device() &&
                  addresses.scalar_type() == at::kLong &&
                  addresses.is_contiguous() && addresses.numel() == world,
              "Invalid cube allreduce peer-address table");
  const int blocks = (input.numel() + (block_packets ? 511 : 255)) /
                     (block_packets ? 512 : 256);
  TORCH_CHECK(counters.is_cuda() && counters.device() == input.device() &&
                  counters.scalar_type() == at::kInt &&
                  counters.is_contiguous() &&
                  counters.numel() >= (input.numel() + 1) / 2,
              "Invalid cube allreduce epoch storage");
  int active_blocks = 0;
  TORCH_CHECK(!block_packets,
              "Only scalar atomic packet protocol is supported");
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, cube_allreduce_kernel<3>, 128, 0));
  TORCH_CHECK(blocks <= active_blocks * props->multiProcessorCount,
              "Cube allreduce requires a fully resident grid");
  const PeerOrder peers{order.ranks[logical_rank ^ 1],
                        world >= 4 ? order.ranks[logical_rank ^ 2] : int(rank),
                        world == 8 ? order.ranks[logical_rank ^ 4] : int(rank)};
  if (world == 2) {
    cube_allreduce_kernel<1>
        <<<blocks, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<half*>(output.data_ptr()),
            reinterpret_cast<const half*>(input.data_ptr()),
            addresses.data_ptr<int64_t>(),
            reinterpret_cast<uint32_t*>(counters.data_ptr()), peers,
            logical_rank, rank, capacity, input.numel());
  } else if (world == 4) {
    cube_allreduce_kernel<2>
        <<<blocks, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<half*>(output.data_ptr()),
            reinterpret_cast<const half*>(input.data_ptr()),
            addresses.data_ptr<int64_t>(),
            reinterpret_cast<uint32_t*>(counters.data_ptr()), peers,
            logical_rank, rank, capacity, input.numel());
  } else if (world == 8) {
    cube_allreduce_kernel<3>
        <<<blocks, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<half*>(output.data_ptr()),
            reinterpret_cast<const half*>(input.data_ptr()),
            addresses.data_ptr<int64_t>(),
            reinterpret_cast<uint32_t*>(counters.data_ptr()), peers,
            logical_rank, rank, capacity, input.numel());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
bool native_peer_atomics(int64_t source, int64_t destination) {
  int supported = 0;
  C10_CUDA_CHECK(cudaDeviceGetP2PAttribute(
      &supported, cudaDevP2PAttrNativeAtomicSupported, source, destination));
  return supported == 1;
}

void close_mem_handle(int64_t pointer) {
  C10_CUDA_CHECK(cudaIpcCloseMemHandle(reinterpret_cast<void*>(pointer)));
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def("sm70_ring_native_peer_atomics(int source, int destination) -> bool",
        &native_peer_atomics);
  m.def("sm70_ring_close_mem_handle(int pointer) -> ()", &close_mem_handle);
  m.def(
      "sm70_ring_atomic_allreduce_out(Tensor(a!) output, Tensor input, "
      "Tensor addresses, Tensor(b!) counters, int[] rank_order, int rank, "
      "int capacity, bool block_packets=False) -> ()");
}
TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("sm70_ring_atomic_allreduce_out", &cube_allreduce);
}
