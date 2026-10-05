// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Resident producer/consumer selection, sparse attention and ordered merge.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda/atomic>
#include <cuda_fp16.h>
#include "../../csrc/qsa_lexicographic_topk.cuh"

namespace {
constexpr int Partitions = 64, Heads = 6, Dimension = 256, TopK = 512;
constexpr int Width = TopK * 4 + 3,
              Tile = (Width + Partitions - 1) / Partitions;
constexpr int Blocks = Partitions + Heads + 1, Threads = 1024;
struct CoreStorage {
  float score[Tile], weight[Tile];
  int physical[Tile];
  float maximum, sum;
};
struct MergeStorage {
  float lse[Partitions], weight[Partitions], maximum, sum;
};
union Storage {
  vllm::qsa::LexicographicDecodeTopKShared<TopK> topk;
  CoreStorage core;
  MergeStorage merge;
};
__device__ float warp_sum(float x) {
  for (int d = 16; d; d >>= 1) x += __shfl_xor_sync(0xffffffff, x, d);
  return x;
}
__device__ void wait(uint32_t* flag, uint32_t generation) {
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> atom(*flag);
  while (atom.load(cuda::memory_order_acquire) != generation) {
  }
}
__device__ void publish(uint32_t* flag, uint32_t generation) {
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> atom(*flag);
  atom.store(generation, cuda::memory_order_release);
}
__global__ __launch_bounds__(Threads, 1) void qsa_selection_attention(
    const float* scores, const int* lengths, const half* queries,
    const half* cache, const int* table, const int64_t* positions,
    const half* gate, int* selected, float* partial, float* lse,
    uint32_t* flags, uint32_t* epochs, half* output, int m, int columns,
    int page_size) {
  __shared__ Storage storage;
  const int t = threadIdx.x, block = blockIdx.x;
  const uint32_t generation = epochs[block] + 1;
  if (block == Partitions + Heads) {
    for (int row = 0; row < m; ++row) {
      vllm::qsa::qsa_lexicographic_decode_topk_body<TopK>(
          scores + row * columns, lengths + row, selected + row * TopK, columns,
          storage.topk);
      __threadfence();
      __syncthreads();
      if (t == 0) publish(flags + row, generation);
      __syncthreads();
    }
  } else if (block < Partitions) {
    const int lane = t & 31, warp = t / 32, split = block;
    for (int row = 0; row < m; ++row) {
      // The query loads can precede the selection dependency.
      const int64_t position = positions[row];
      const int tail_count = (position + 1) % 4;
      const int tail_start = (position + 1) / 4 * 4;
      wait(flags + row, generation);
      for (int head = 0; head < Heads; ++head) {
        float q[8];
#pragma unroll
        for (int d = 0; d < 8; ++d)
          q[d] = __half2float(
              queries[(row * Heads + head) * Dimension + d * 32 + lane]);
        for (int item = warp; item < Tile; item += 32) {
          const int column = split * Tile + item;
          int logical = -1;
          if (column < TopK * 4)
            logical = selected[row * TopK + column / 4] * 4 + column % 4;
          else if (column < Width && column - TopK * 4 < tail_count)
            logical = tail_start + column - TopK * 4;
          int physical = -1;
          float dot = 0;
          if (logical >= 0 && logical <= position) {
            physical =
                table[logical / page_size] * page_size + logical % page_size;
            const int base =
                (physical / page_size * 2 * page_size + physical % page_size) *
                Dimension;
#pragma unroll
            for (int d = 0; d < 8; ++d)
              dot = fmaf(q[d], __half2float(cache[base + d * 32 + lane]), dot);
          }
          dot = warp_sum(dot);
          if (lane == 0) {
            storage.core.score[item] =
                physical >= 0 ? dot * 0.0625f : -INFINITY;
            storage.core.physical[item] = physical;
          }
        }
        __syncthreads();
        if (t == 0) {
          float maximum = -INFINITY;
#pragma unroll
          for (int item = 0; item < Tile; ++item)
            maximum = fmaxf(maximum, storage.core.score[item]);
          float sum = 0;
#pragma unroll
          for (int item = 0; item < Tile; ++item) {
            const float weight = storage.core.physical[item] >= 0
                                     ? expf(storage.core.score[item] - maximum)
                                     : 0.f;
            storage.core.weight[item] = weight;
            sum += weight;
          }
          storage.core.maximum = maximum;
          storage.core.sum = sum;
        }
        __syncthreads();
        const int job = (row * Heads + head) * Partitions + split;
        if (t < Dimension) {
          float accumulator = 0;
#pragma unroll
          for (int item = 0; item < Tile; ++item) {
            const int physical = storage.core.physical[item];
            if (physical >= 0) {
              const int index = ((physical / page_size * 2 + 1) * page_size +
                                 physical % page_size) *
                                    Dimension +
                                t;
              accumulator = fmaf(storage.core.weight[item],
                                 __half2float(cache[index]), accumulator);
            }
          }
          partial[job * Dimension + t] =
              storage.core.sum > 0 ? accumulator / storage.core.sum : 0.f;
        }
        if (t == 0)
          lse[job] = storage.core.sum > 0
                         ? storage.core.maximum + logf(storage.core.sum)
                         : -INFINITY;
        __threadfence();
        __syncthreads();
        if (t == 0) publish(flags + m + job, generation);
        __syncthreads();
      }
    }
  } else {
    const int head = block - Partitions;
    for (int row = 0; row < m; ++row) {
      const int base = (row * Heads + head) * Partitions;
      if (t < Partitions) {
        wait(flags + m + base + t, generation);
        storage.merge.lse[t] = lse[base + t];
      }
      __syncthreads();
      if (t == 0) {
        float maximum = -INFINITY;
#pragma unroll
        for (int i = 0; i < Partitions; ++i)
          maximum = fmaxf(maximum, storage.merge.lse[i]);
        float sum = 0;
#pragma unroll
        for (int i = 0; i < Partitions; ++i) {
          const float weight = isfinite(storage.merge.lse[i])
                                   ? expf(storage.merge.lse[i] - maximum)
                                   : 0.f;
          storage.merge.weight[i] = weight;
          sum += weight;
        }
        storage.merge.sum = sum;
      }
      __syncthreads();
      if (t < Dimension) {
        float result = 0;
#pragma unroll
        for (int i = 0; i < Partitions; ++i)
          result = fmaf(storage.merge.weight[i],
                        partial[(base + i) * Dimension + t], result);
        result = storage.merge.sum > 0 ? result / storage.merge.sum : 0.f;
        result = __half2float(__float2half_rn(result));
        const int index = (row * Heads + head) * Dimension + t;
        const float g = __half2float(gate[index]);
        output[index] = __float2half_rn(result / (1.f + expf(-g)));
      }
      __syncthreads();
    }
  }
  if (t == 0) epochs[block] = generation;
}
}  // namespace
void run(torch::Tensor scores, torch::Tensor lengths, torch::Tensor queries,
         torch::Tensor cache, torch::Tensor table, torch::Tensor positions,
         torch::Tensor gate, torch::Tensor selected, torch::Tensor partial,
         torch::Tensor lse, torch::Tensor flags, torch::Tensor epochs,
         torch::Tensor output) {
  const int m = queries.size(0), columns = scores.size(1),
            page_size = cache.size(2);
  TORCH_CHECK((m == 1 || m == 5) && columns <= 2304);
  TORCH_CHECK(queries.sizes() == at::IntArrayRef({m, 6, 256}));
  TORCH_CHECK(cache.dim() == 4 && cache.size(1) == 2 && cache.size(3) == 256);
  const auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props->major == 7 && props->minor == 0 &&
              props->multiProcessorCount >= Blocks);
  const auto stream = c10::cuda::getCurrentCUDAStream().stream();
  qsa_selection_attention<<<Blocks, Threads, 0, stream>>>(
      scores.data_ptr<float>(), lengths.data_ptr<int>(),
      (half*)queries.data_ptr(), (half*)cache.data_ptr(), table.data_ptr<int>(),
      positions.data_ptr<int64_t>(), (half*)gate.data_ptr(),
      selected.data_ptr<int>(), partial.data_ptr<float>(),
      lse.data_ptr<float>(), (uint32_t*)flags.data_ptr(),
      (uint32_t*)epochs.data_ptr(), (half*)output.data_ptr(), m, columns,
      page_size);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) { module.def("run", &run); }
