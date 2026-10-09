// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Reuse each compressed key across a single request's speculative queries.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <torch/library.h>
#include <torch/types.h>
#include <pybind11/pybind11.h>

namespace {
#define QSA_MMA(C, A0, A1, B0, B1)                                  \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

template <typename Position>
__global__ void mqa_multiquery(const half* q, const half* cache,
                               const int* table, const int* requests,
                               const Position* positions, const int* lengths,
                               float* scores, int* visible, int m, int columns,
                               int pages, int page_size, int table_width,
                               int sq0, int sq1, int sq2, int sc0, int sc1,
                               int sc3, int st1, float divisor, int compress) {
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int quad = (lane >> 2) & 3;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int column_base = blockIdx.x * 32 + warp * 8;
  const int col = column_base + r;
  int end = 0;
  for (int row = 0; row < m; ++row) {
    const long long prefix = positions[row] + 1;
    const long long length = requests[row] == 0 ? lengths[0] : 0;
    const long long causal = prefix < length ? prefix : length;
    const int n = causal >= 0 ? causal / compress
                              : -((-causal + compress - 1) / compress);
    end = max(end, min(n, columns));
    if (blockIdx.x == 0 && threadIdx.x == row) visible[row] = n;
  }
  if (blockIdx.x * 32 >= end) return;
  int page = col < end && col / page_size < table_width
                 ? table[(col / page_size) * st1]
                 : -1;
  const bool key_valid = col < end && page >= 0 && page < pages;
  const int qhead = quad * 8 + r, qrow = qhead / 4;
  float acc[8] = {};
#pragma unroll
  for (int g = 0; g < 8; ++g) {
    // General strides include transposed paged storage and sliced Q views.
    uint4 a = {}, b = {}, lo = {}, hi = {};
    half* av = reinterpret_cast<half*>(&a);
    half* bv = reinterpret_cast<half*>(&b);
    half* lv = reinterpret_cast<half*>(&lo);
    half* hv = reinterpret_cast<half*>(&hi);
    if (sc3 == 1 && sq2 == 1 && sc0 % 8 == 0 && sc1 % 8 == 0 && sq0 % 8 == 0 &&
        sq1 % 8 == 0 &&
        (reinterpret_cast<uintptr_t>(q) | reinterpret_cast<uintptr_t>(cache)) %
                16 ==
            0) {
      if (key_valid) {
        const half* k = cache + static_cast<int64_t>(page) * sc0 +
                        (col % page_size) * sc1 + g * 16;
        a = *reinterpret_cast<const uint4*>(k);
        b = *reinterpret_cast<const uint4*>(k + 8);
      }
      if (qrow < m) {
        const half* query = q + qrow * sq0 + (qhead % 4) * sq1 + g * 16;
        lo = *reinterpret_cast<const uint4*>(query);
        hi = *reinterpret_cast<const uint4*>(query + 8);
      }
    } else {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if (key_valid) {
          const half* k = cache + static_cast<int64_t>(page) * sc0 +
                          (col % page_size) * sc1;
          av[j] = k[(g * 16 + j) * sc3];
          bv[j] = k[(g * 16 + 8 + j) * sc3];
        }
        if (qrow < m) {
          const half* query = q + qrow * sq0 + (qhead % 4) * sq1;
          lv[j] = query[(g * 16 + j) * sq2];
          hv[j] = query[(g * 16 + 8 + j) * sq2];
        }
      }
    }
    QSA_MMA(acc, a.x, a.y, lo.x, lo.y);
    QSA_MMA(acc, a.z, a.w, lo.z, lo.w);
    QSA_MMA(acc, b.x, b.y, hi.x, hi.y);
    QSA_MMA(acc, b.z, b.w, hi.z, hi.w);
  }
#pragma unroll
  for (int p = 0; p < 2; ++p) {
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const int i = p * 4 + k * 2;
      const float h0 = fmaxf(acc[i], 0.f), h1 = fmaxf(acc[i + 1], 0.f);
      const float h2 = __shfl_xor_sync(0xffffffff, h0, 2);
      const float h3 = __shfl_xor_sync(0xffffffff, h1, 2);
      const float sum = __fadd_rn(__fadd_rn(__fadd_rn(h0, h1), h2), h3);
      const int row = quad * 2 + p;
      const int column =
          column_base + k * 2 + ((lane & 16) ? 4 : 0) + (lane & 1);
      // Keys for this output row reside in the lane with r==key-column.
      const int key_lane = (lane & ~3) | (k * 2 + (lane & 1));
      const bool valid =
          __shfl_sync(0xffffffff, static_cast<int>(key_valid), key_lane);
      if ((lane & 2) == 0 && row < m && requests[row] == 0 &&
          column < columns && column < (positions[row] + 1) / compress &&
          column < lengths[0] / compress) {
        scores[row * columns + column] = valid ? sum / divisor : -CUDART_INF_F;
      }
    }
  }
}

#undef QSA_MMA

void mqa_out(torch::Tensor scores, torch::Tensor visible, torch::Tensor q,
             torch::Tensor cache, torch::Tensor table, torch::Tensor requests,
             torch::Tensor positions, torch::Tensor lengths, double divisor,
             int64_t compress) {
  const c10::cuda::CUDAGuard guard(q.device());
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kHalf && q.dim() == 3 &&
                  q.size(0) > 0 && q.size(0) <= 8 && q.size(1) == 4 &&
                  q.size(2) == 128 && table.dim() == 2 && table.size(0) == 1,
              "Shared-key MQA requires FP16 M1..8/H4/D128 and one request");
  const int m = q.size(0), columns = scores.size(1);
  TORCH_CHECK(cache.dim() == 4 && cache.size(2) == 1 && cache.size(3) == 128 &&
                  cache.scalar_type() == at::kHalf &&
                  scores.scalar_type() == at::kFloat &&
                  scores.is_contiguous() && scores.size(0) == m &&
                  divisor > 0 && compress > 0,
              "Invalid MQA cache, scores, or scale");
  for (const auto& t : {table, requests, lengths, visible})
    TORCH_CHECK(t.device() == q.device() && t.scalar_type() == at::kInt,
                "MQA metadata must use same-device int32");
  TORCH_CHECK(scores.device() == q.device() && cache.device() == q.device() &&
                  requests.numel() >= m && positions.numel() >= m &&
                  visible.numel() >= m && lengths.numel() >= 1,
              "MQA tensor extent mismatch");
  TORCH_CHECK(positions.device() == q.device() && positions.is_contiguous() &&
                  (positions.scalar_type() == at::kLong ||
                   positions.scalar_type() == at::kInt),
              "Positions must use contiguous same-device int32/int64");
  TORCH_CHECK(requests.is_contiguous() && lengths.is_contiguous() &&
                  visible.is_contiguous(),
              "MQA row metadata must be contiguous");
  const auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props->major == 7 && props->minor == 0, "SM70 required");
  if (!columns) return;
  if (positions.scalar_type() == at::kLong) {
    mqa_multiquery<int64_t>
        <<<(columns + 31) / 32, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const half*>(q.data_ptr()),
            reinterpret_cast<const half*>(cache.data_ptr()),
            table.data_ptr<int>(), requests.data_ptr<int>(),
            positions.data_ptr<int64_t>(), lengths.data_ptr<int>(),
            scores.data_ptr<float>(), visible.data_ptr<int>(), m, columns,
            cache.size(0), cache.size(1), table.size(1), q.stride(0),
            q.stride(1), q.stride(2), cache.stride(0), cache.stride(1),
            cache.stride(3), table.stride(1), divisor, compress);
  } else {
    mqa_multiquery<int32_t>
        <<<(columns + 31) / 32, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const half*>(q.data_ptr()),
            reinterpret_cast<const half*>(cache.data_ptr()),
            table.data_ptr<int>(), requests.data_ptr<int>(),
            positions.data_ptr<int>(), lengths.data_ptr<int>(),
            scores.data_ptr<float>(), visible.data_ptr<int>(), m, columns,
            cache.size(0), cache.size(1), table.size(1), q.stride(0),
            q.stride(1), q.stride(2), cache.stride(0), cache.stride(1),
            cache.stride(3), table.stride(1), divisor, compress);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace
TORCH_LIBRARY(vllm_sm70_qsa_indexer, m) {
  m.def(
      "run(Tensor(a!) scores, Tensor(b!) visible, Tensor q, Tensor cache, "
      "Tensor table, Tensor requests, Tensor positions, Tensor lengths, "
      "float divisor, int compress) -> ()");
  m.impl("run", torch::kCUDA, &mqa_out);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
