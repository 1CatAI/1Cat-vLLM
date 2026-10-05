// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Resident QSA keeps Tensor Core QK/PV and the native exact selector.
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
constexpr int MatrixTokens = 64;
struct CoreStorage {
  __align__(16) half packed[MatrixTokens * Dimension];
  float score[Heads * MatrixTokens];
  __align__(16) half probability[Heads * MatrixTokens];
  int64_t key_base[MatrixTokens];
  float maximum[Heads], sum[Heads];
};
struct MergeStorage {
  float lse[Partitions], weight[Partitions], maximum, sum;
};
union Storage {
  vllm::qsa::LexicographicDecodeTopKShared<TopK> topk;
  CoreStorage core;
  MergeStorage merge;
};
__device__ void mma(float (&acc)[8], uint32_t a0, uint32_t a1, uint32_t b0,
                    uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};"
      : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3]), "+f"(acc[4]),
        "+f"(acc[5]), "+f"(acc[6]), "+f"(acc[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
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
    const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int col = ((lane >> 2) & 3) * 8 + r;
    for (int row = 0; row < m; ++row) {
      wait(flags + row, generation);
      if (t < MatrixTokens) {
        const int column = split * Tile + t;
        const int64_t position = positions[row];
        const int tail_count = (position + 1) % 4;
        const int tail_start = (position + 1) / 4 * 4;
        int logical = -1;
        if (t < Tile && column < TopK * 4)
          logical = selected[row * TopK + column / 4] * 4 + column % 4;
        else if (t < Tile && column < Width && column - TopK * 4 < tail_count)
          logical = tail_start + column - TopK * 4;
        storage.core.key_base[t] = logical >= 0 && logical <= position
                                       ? int64_t(table[logical / page_size]) *
                                                 2 * page_size * Dimension +
                                             (logical % page_size) * Dimension
                                       : -1;
      }
      __syncthreads();
      // Transpose only the shared-memory layout, keeping global reads
      // coalesced.
      for (int index = t; index < MatrixTokens * Dimension; index += Threads) {
        const int token = index / Dimension, d = index % Dimension;
        const int64_t base = storage.core.key_base[token];
        const int packed_index =
            (token / 32 * (Dimension / 16) + d / 16) * 512 +
            (d % 16 / 8) * 256 + token % 32 * 8 + d % 8;
        storage.core.packed[packed_index] =
            base >= 0 ? cache[base + d] : __float2half(0.f);
      }
      __syncthreads();
      if (warp < MatrixTokens / 32) {
        float accum[8] = {};
        for (int g = 0; g < Dimension / 16; ++g) {
          const half* weight = storage.core.packed +
                               (warp * (Dimension / 16) + g) * 512 + col * 8;
          const uint4 lo = *(const uint4*)weight,
                      hi = *(const uint4*)(weight + 256);
          uint4 a = {}, b = {};
          if (r < Heads) {
            const half* q = queries + (row * Heads + r) * Dimension + g * 16;
            a = *(const uint4*)q;
            b = *(const uint4*)(q + 8);
          }
          mma(accum, a.x, a.y, lo.x, lo.y);
          mma(accum, a.z, a.w, lo.z, lo.w);
          mma(accum, b.x, b.y, hi.x, hi.y);
          mma(accum, b.z, b.w, hi.z, hi.w);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int head = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
          const int token =
              warp * 32 + ((lane >> 2) & 3) * 8 +
              ((i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2));
          if (head < Heads)
            storage.core.score[head * MatrixTokens + token] =
                accum[i] * 0.0625f;
        }
      }
      __syncthreads();
      if (t < Heads) {
        float maximum = -INFINITY;
        for (int token = 0; token < MatrixTokens; ++token)
          if (storage.core.key_base[token] >= 0)
            maximum =
                fmaxf(maximum, storage.core.score[t * MatrixTokens + token]);
        float sum = 0;
        for (int token = 0; token < MatrixTokens; ++token) {
          const float p =
              storage.core.key_base[token] >= 0
                  ? expf(storage.core.score[t * MatrixTokens + token] - maximum)
                  : 0.f;
          sum += p;
          storage.core.probability[t * MatrixTokens + token] =
              __float2half_rn(p);
        }
        storage.core.maximum[t] = maximum;
        storage.core.sum[t] = sum;
      }
      __syncthreads();
      for (int index = t; index < MatrixTokens * Dimension; index += Threads) {
        const int token = index / Dimension, d = index % Dimension;
        const int64_t base = storage.core.key_base[token];
        const int packed_index =
            (d / 32 * (MatrixTokens / 16) + token / 16) * 512 +
            (token % 16 / 8) * 256 + d % 32 * 8 + token % 8;
        storage.core.packed[packed_index] =
            base >= 0 ? cache[base + page_size * Dimension + d]
                      : __float2half(0.f);
      }
      __syncthreads();
      if (warp < Dimension / 32) {
        float accum[8] = {};
        for (int g = 0; g < MatrixTokens / 16; ++g) {
          const half* weight = storage.core.packed +
                               (warp * (MatrixTokens / 16) + g) * 512 + col * 8;
          const uint4 lo = *(const uint4*)weight,
                      hi = *(const uint4*)(weight + 256);
          uint4 a = {}, b = {};
          if (r < Heads) {
            const half* p =
                storage.core.probability + r * MatrixTokens + g * 16;
            a = *(const uint4*)p;
            b = *(const uint4*)(p + 8);
          }
          mma(accum, a.x, a.y, lo.x, lo.y);
          mma(accum, a.z, a.w, lo.z, lo.w);
          mma(accum, b.x, b.y, hi.x, hi.y);
          mma(accum, b.z, b.w, hi.z, hi.w);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int head = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
          const int d = warp * 32 + ((lane >> 2) & 3) * 8 +
                        ((i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2));
          if (head < Heads) {
            const int job = (row * Heads + head) * Partitions + split;
            partial[job * Dimension + d] =
                storage.core.sum[head] > 0 ? accum[i] / storage.core.sum[head]
                                           : 0.f;
          }
        }
      }
      if (t < Heads) {
        const int job = (row * Heads + t) * Partitions + split;
        lse[job] = storage.core.sum[t] > 0
                       ? storage.core.maximum[t] + logf(storage.core.sum[t])
                       : -INFINITY;
      }
      __threadfence();
      __syncthreads();
      if (t < Heads)
        publish(flags + m + (row * Heads + t) * Partitions + split, generation);
      __syncthreads();
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
  int active = 0;
  TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                  &active, qsa_selection_attention, Threads, 0) == cudaSuccess);
  TORCH_CHECK(props->major == 7 && props->minor == 0 &&
              props->cooperativeLaunch &&
              active * props->multiProcessorCount >= Blocks);
  auto* sp = scores.data_ptr<float>();
  auto* lp = lengths.data_ptr<int>();
  auto* qp = (half*)queries.data_ptr();
  auto* cp = (half*)cache.data_ptr();
  auto* tp = table.data_ptr<int>();
  auto* pp = positions.data_ptr<int64_t>();
  auto* gp = (half*)gate.data_ptr();
  auto* sel = selected.data_ptr<int>();
  auto* part = partial.data_ptr<float>();
  auto* ls = lse.data_ptr<float>();
  auto* fl = (uint32_t*)flags.data_ptr();
  auto* ep = (uint32_t*)epochs.data_ptr();
  auto* out = (half*)output.data_ptr();
  void* arguments[] = {&sp,
                       &lp,
                       &qp,
                       &cp,
                       &tp,
                       &pp,
                       &gp,
                       &sel,
                       &part,
                       &ls,
                       &fl,
                       &ep,
                       &out,
                       (void*)&m,
                       (void*)&columns,
                       (void*)&page_size};
  TORCH_CHECK(cudaLaunchCooperativeKernel(
                  (void*)qsa_selection_attention, dim3(Blocks), dim3(Threads),
                  arguments, 0,
                  c10::cuda::getCurrentCUDAStream().stream()) == cudaSuccess);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) { module.def("run", &run); }
