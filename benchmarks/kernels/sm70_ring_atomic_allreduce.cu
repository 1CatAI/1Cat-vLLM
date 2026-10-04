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

// Each naturally aligned scalar packet contains one FP32 value and its
// generation. System-scoped atomic loads/stores synchronize the same 64-bit
// object. There are no overlapping accesses with another width.
__device__ __forceinline__ void publish_packet(uint64_t* ptr, float value,
                                               uint32_t tag) {
  const uint64_t packet = (uint64_t(tag) << 32) | __float_as_uint(value);
  __nv_atomic_store_n(ptr, packet, __NV_ATOMIC_RELEASE,
                      __NV_THREAD_SCOPE_SYSTEM);
}

__device__ __forceinline__ float receive_packet(uint64_t* ptr, uint32_t tag) {
  uint64_t packet;
  do {
    packet =
        __nv_atomic_load_n(ptr, __NV_ATOMIC_ACQUIRE, __NV_THREAD_SCOPE_SYSTEM);
  } while (uint32_t(packet >> 32) != tag);
  return __uint_as_float(uint32_t(packet));
}

// Every stage exchanges with its direct NVLink neighbor. Double buffering
// separates consecutive generations; each lane must consume a peer packet
// before it can advance to the following generation. Intermediate sums never
// narrow to FP16. Inactive blocks retain independent generation counters.
__global__ __launch_bounds__(128, 4) void cube_allreduce_kernel(
    half* output, const half* input, const int64_t* addresses,
    uint32_t* counters, RankOrder order, int logical_rank, int rank, int stages,
    int capacity, int size) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t epoch = counters[blockIdx.x], tag = epoch + 1u;
  const int packs = (size + 1) / 2;
  if (index < packs) {
    float x = __half2float(input[index * 2]);
    float y = index * 2 + 1 < size ? __half2float(input[index * 2 + 1]) : 0.0f;
    for (int stage = 0; stage < stages; ++stage) {
      const int peer = order.ranks[logical_rank ^ (1 << stage)];
      const size_t offset =
          ((epoch & 1u) * stages + stage) * static_cast<size_t>(capacity) +
          index;
      auto* destination =
          reinterpret_cast<uint64_t*>(addresses[peer]) + 2 * offset;
      auto* source = reinterpret_cast<uint64_t*>(addresses[rank]) + 2 * offset;
      publish_packet(destination, x, tag);
      publish_packet(destination + 1, y, tag);
      const float rx = receive_packet(source, tag);
      const float ry = receive_packet(source + 1, tag);
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
  }
  __syncthreads();
  if (threadIdx.x == 0) counters[blockIdx.x] = epoch + 1u;
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
                  counters.is_contiguous() && counters.numel() >= blocks,
              "Invalid cube allreduce epoch storage");
  int active_blocks = 0;
  TORCH_CHECK(!block_packets,
              "Only scalar atomic packet protocol is supported");
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, cube_allreduce_kernel, 128, 0));
  TORCH_CHECK(blocks <= active_blocks * props->multiProcessorCount,
              "Cube allreduce requires a fully resident grid");
  const int stages = world == 8 ? 3 : world == 4 ? 2 : 1;
  cube_allreduce_kernel<<<blocks, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<half*>(output.data_ptr()),
      reinterpret_cast<const half*>(input.data_ptr()),
      addresses.data_ptr<int64_t>(),
      reinterpret_cast<uint32_t*>(counters.data_ptr()), order, logical_rank,
      rank, stages, capacity, input.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "sm70_ring_atomic_allreduce_out(Tensor(a!) output, Tensor input, "
      "Tensor addresses, Tensor(b!) counters, int[] rank_order, int rank, "
      "int capacity, bool block_packets=False) -> ()");
}
TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("sm70_ring_atomic_allreduce_out", &cube_allreduce);
}
