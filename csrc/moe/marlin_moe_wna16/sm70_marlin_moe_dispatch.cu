/*
 * Modified by Neural Magic
 * Copyright (C) Marlin.2024 Elias Frantar
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Adapted from https://github.com/IST-DASLab/marlin
 */

#ifndef MARLIN_NAMESPACE_NAME
  #define MARLIN_NAMESPACE_NAME marlin_moe_wna16
#endif

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>

#include "quantization/marlin/marlin.cuh"
#include "core/registration.h"
#include "core/scalar_type.hpp"

namespace {

__device__ __forceinline__ int sm70_inverse_zp_interleave(int idx,
                                                          int num_bits) {
  if (num_bits == 4) {
    constexpr int inv[8] = {0, 4, 1, 5, 2, 6, 3, 7};
    return inv[idx];
  }
  constexpr int inv[4] = {0, 2, 1, 3};
  return inv[idx];
}

__device__ __forceinline__ int sm70_inverse_scale_perm(int idx,
                                                       int perm_len) {
  if (perm_len == 64) {
    return (idx % 8) * 8 + idx / 8;
  }
  constexpr int inv[32] = {0,  1,  8,  9,  16, 17, 24, 25,
                           2,  3,  10, 11, 18, 19, 26, 27,
                           4,  5,  12, 13, 20, 21, 28, 29,
                           6,  7,  14, 15, 22, 23, 30, 31};
  return inv[idx];
}

__global__ void sm70_unpermute_half_metadata_kernel(
    const half* __restrict__ src, half* __restrict__ dst, int64_t total,
    int perm_len) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int64_t block = idx / perm_len;
  const int pos = static_cast<int>(idx % perm_len);
  dst[idx] = src[block * perm_len + sm70_inverse_scale_perm(pos, perm_len)];
}

__global__ void sm70_unpack_moe_zp_to_half_kernel(
    const int32_t* __restrict__ packed_zp,
    const half* __restrict__ scales, half* __restrict__ out,
    int64_t total, int64_t num_groups, int64_t size_n, int64_t packed_n,
    int num_bits, bool is_a_8bit) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  const int pack_factor = 32 / num_bits;
  const int64_t n = idx % size_n;
  const int64_t group = (idx / size_n) % num_groups;
  const int64_t expert = idx / (num_groups * size_n);
  const int64_t block64 = n / 64;
  int packed_col_in_block =
      sm70_inverse_scale_perm(static_cast<int>(n % 64), 64);
  if (!is_a_8bit) {
    packed_col_in_block =
        (packed_col_in_block / pack_factor) * pack_factor +
        sm70_inverse_zp_interleave(packed_col_in_block % pack_factor,
                                   num_bits);
  }
  const int64_t packed_col = block64 * 64 + packed_col_in_block;
  const uint32_t word = reinterpret_cast<const uint32_t*>(packed_zp)
      [(expert * num_groups + group) * packed_n + packed_col / pack_factor];
  const uint32_t zp =
      (word >> ((packed_col % pack_factor) * num_bits)) &
      ((1u << num_bits) - 1u);
  out[idx] = __float2half(static_cast<float>(zp) * __half2float(scales[idx]));
}

torch::Tensor sm70_moe_scales_as_logical(torch::Tensor const& b_scales,
                                         int64_t size_k, int64_t group_size,
                                         bool is_a_8bit) {
  TORCH_CHECK(b_scales.scalar_type() == at::ScalarType::Half,
              "SM70 Marlin MoE adapter expects fp16 scales.");
  const bool uses_group_perm =
      group_size < size_k && group_size != -1 && !is_a_8bit;
  const int perm_len = uses_group_perm ? 64 : 32;
  TORCH_CHECK(b_scales.numel() % perm_len == 0,
              "SM70 Marlin MoE scale metadata size is not divisible by ",
              perm_len);

  auto out = torch::empty_like(b_scales);
  const int64_t total = out.numel();
  constexpr int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  sm70_unpermute_half_metadata_kernel<<<
      blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(b_scales.data_ptr<at::Half>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()), total, perm_len);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor sm70_moe_zp_as_half(torch::Tensor const& b_zeros,
                                  torch::Tensor const& b_scales,
                                  int64_t size_n, int num_bits,
                                  bool is_a_8bit) {
  if (b_zeros.scalar_type() == at::ScalarType::Half) {
    return b_zeros;
  }
  TORCH_CHECK(b_zeros.scalar_type() == at::ScalarType::Int,
              "SM70 Marlin MoE zero points must be fp16 or packed int32.");
  const int pack_factor = 32 / num_bits;
  TORCH_CHECK(b_zeros.dim() == 3, "b_zeros rank = ", b_zeros.dim(),
              " is not 3");
  TORCH_CHECK(b_zeros.size(0) == b_scales.size(0),
              "b_zeros dim 0 = ", b_zeros.size(0),
              " is not num_experts = ", b_scales.size(0));
  TORCH_CHECK(b_zeros.size(1) == b_scales.size(1),
              "b_zeros dim 1 = ", b_zeros.size(1),
              " is not num_groups = ", b_scales.size(1));
  TORCH_CHECK(b_zeros.size(2) * pack_factor == size_n,
              "packed b_zeros dim 2 = ", b_zeros.size(2),
              " does not match size_n = ", size_n);
  TORCH_CHECK(b_scales.scalar_type() == at::ScalarType::Half,
              "SM70 Marlin MoE packed zero-point adapter expects fp16 scales.");

  auto out =
      torch::empty({b_scales.size(0), b_scales.size(1), size_n},
                   b_scales.options());
  const int64_t total = out.numel();
  constexpr int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  sm70_unpack_moe_zp_to_half_kernel<<<
      blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      b_zeros.data_ptr<int32_t>(),
      reinterpret_cast<const half*>(b_scales.data_ptr<at::Half>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()), total,
      b_scales.size(1), size_n, b_zeros.size(2), num_bits, is_a_8bit);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace

namespace MARLIN_NAMESPACE_NAME {

torch::Tensor sm70_marlin_u4_gemm(
    torch::Tensor& a, torch::Tensor& c, torch::Tensor& b_q_weight,
    torch::Tensor& b_scales, torch::Tensor& b_zeros,
    torch::Tensor& sorted_token_ids, torch::Tensor& expert_ids,
    torch::Tensor& num_tokens_past_padded, torch::Tensor& topk_weights,
    int64_t moe_block_size, int64_t top_k, bool mul_topk_weights,
    int64_t size_m, int64_t size_n, int64_t size_k, int64_t group_size);

torch::Tensor sm70_marlin_u4b8_gemm(
    torch::Tensor& a, torch::Tensor& c, torch::Tensor& b_q_weight,
    torch::Tensor& b_scales, torch::Tensor& sorted_token_ids,
    torch::Tensor& expert_ids, torch::Tensor& num_tokens_past_padded,
    torch::Tensor& topk_weights, int64_t moe_block_size, int64_t top_k,
    bool mul_topk_weights, int64_t size_m, int64_t size_n, int64_t size_k,
    int64_t group_size);

torch::Tensor sm70_marlin_u8_gemm(
    torch::Tensor& a, torch::Tensor& c, torch::Tensor& b_q_weight,
    torch::Tensor& b_scales, torch::Tensor& b_zeros,
    torch::Tensor& sorted_token_ids, torch::Tensor& expert_ids,
    torch::Tensor& num_tokens_past_padded, torch::Tensor& topk_weights,
    int64_t moe_block_size, int64_t top_k, bool mul_topk_weights,
    int64_t size_m, int64_t size_n, int64_t size_k, int64_t group_size);

torch::Tensor sm70_marlin_u8b128_gemm(
    torch::Tensor& a, torch::Tensor& c, torch::Tensor& b_q_weight,
    torch::Tensor& b_scales, torch::Tensor& sorted_token_ids,
    torch::Tensor& expert_ids, torch::Tensor& num_tokens_past_padded,
    torch::Tensor& topk_weights, int64_t moe_block_size, int64_t top_k,
    bool mul_topk_weights, int64_t size_m, int64_t size_n, int64_t size_k,
    int64_t group_size);

torch::Tensor sm70_marlin_fp8_gemm(
    torch::Tensor& a, torch::Tensor& c, torch::Tensor& b_q_weight,
    torch::Tensor& b_scales, torch::Tensor& sorted_token_ids,
    torch::Tensor& expert_ids, torch::Tensor& num_tokens_past_padded,
    torch::Tensor& topk_weights, int64_t moe_block_size, int64_t top_k,
    bool mul_topk_weights, int64_t size_m, int64_t size_n, int64_t size_k,
    int64_t group_size);

torch::Tensor sm70_marlin_nvfp4_gemm(
    torch::Tensor& a, torch::Tensor& c, torch::Tensor& b_q_weight,
    torch::Tensor& b_scales, torch::Tensor& global_scale,
    torch::Tensor& sorted_token_ids, torch::Tensor& expert_ids,
    torch::Tensor& num_tokens_past_padded, torch::Tensor& topk_weights,
    int64_t moe_block_size, int64_t top_k, bool mul_topk_weights,
    int64_t size_m, int64_t size_n, int64_t size_k, int64_t group_size);

torch::Tensor sm70_marlin_mxfp4_gemm(
    torch::Tensor& a, torch::Tensor& c, torch::Tensor& b_q_weight,
    torch::Tensor& b_scales, torch::Tensor& sorted_token_ids,
    torch::Tensor& expert_ids, torch::Tensor& num_tokens_past_padded,
    torch::Tensor& topk_weights, int64_t moe_block_size, int64_t top_k,
    bool mul_topk_weights, int64_t size_m, int64_t size_n, int64_t size_k,
    int64_t group_size);

}  // namespace MARLIN_NAMESPACE_NAME

torch::Tensor moe_wna16_marlin_gemm(
    torch::Tensor& a, std::optional<torch::Tensor> c_or_none,
    torch::Tensor& b_q_weight,
    std::optional<torch::Tensor> const& b_bias_or_none, torch::Tensor& b_scales,
    std::optional<torch::Tensor> const& a_scales_or_none,
    std::optional<torch::Tensor> const& global_scale_or_none,
    std::optional<torch::Tensor> const& b_zeros_or_none,
    std::optional<torch::Tensor> const& g_idx_or_none,
    std::optional<torch::Tensor> const& perm_or_none,
    torch::Tensor& workspace,
    torch::Tensor& sorted_token_ids, torch::Tensor& expert_ids,
    torch::Tensor& num_tokens_past_padded, torch::Tensor& topk_weights,
    int64_t moe_block_size, int64_t top_k, bool mul_topk_weights,
    vllm::ScalarTypeId const& b_type_id, int64_t size_m, int64_t size_n,
    int64_t size_k, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce,
    bool is_zp_float, int64_t thread_k, int64_t thread_n,
    int64_t blocks_per_sm) {
  (void)workspace;
  (void)thread_k;
  (void)thread_n;
  (void)blocks_per_sm;

  vllm::ScalarTypeId a_type_id, c_type_id, s_type_id;

  auto c_dtype = a.dtype();
  if (a.scalar_type() == at::ScalarType::Half) {
    a_type_id = vllm::kFloat16.id();
    c_type_id = vllm::kFloat16.id();
  } else if (a.scalar_type() == at::ScalarType::BFloat16) {
    a_type_id = vllm::kBFloat16.id();
    c_type_id = vllm::kBFloat16.id();
  } else {
    c_dtype = b_scales.dtype();
    if (b_scales.scalar_type() == at::ScalarType::Half) {
      c_type_id = vllm::kFloat16.id();
    } else if (b_scales.scalar_type() == at::ScalarType::BFloat16) {
      c_type_id = vllm::kBFloat16.id();
    } else {
      c_type_id = vllm::kBFloat16.id();

      TORCH_CHECK(c_or_none.has_value(), "c must be passed for W4A8-FP4");
      torch::Tensor c = c_or_none.value();
      c_dtype = c.dtype();

      if (c.scalar_type() == at::ScalarType::Half) {
        c_type_id = vllm::kFloat16.id();
      } else if (c.scalar_type() == at::ScalarType::BFloat16) {
        c_type_id = vllm::kBFloat16.id();
      } else {
        TORCH_CHECK(false, "unsupported c dtype");
      }
    }

    if (a.scalar_type() == at::ScalarType::Float8_e4m3fn) {
      a_type_id = vllm::kFE4M3fn.id();
    } else if (a.scalar_type() == at::ScalarType::Char) {
      a_type_id = vllm::kS8.id();
    } else {
      TORCH_CHECK(false, "unsupported `a` scalar_type");
    }
  }

  s_type_id = c_type_id;
  if (b_type_id == vllm::kFE2M1f.id()) {
    if (b_scales.scalar_type() == at::ScalarType::Float8_e4m3fn) {
      s_type_id = vllm::kFE4M3fn.id();
    } else if (b_scales.scalar_type() == at::ScalarType::Float8_e8m0fnu) {
      s_type_id = vllm::kFE8M0fnu.id();
    } else {
      TORCH_CHECK(false,
                  "When b_type = float4_e2m1f, b_scale scalar type must be",
                  "float8_e4m3fn (for NVFP4) or float8_e8m0fnu (for MXFP4).");
    }
  }

  vllm::ScalarType a_type = vllm::ScalarType::from_id(a_type_id);
  vllm::ScalarType b_type = vllm::ScalarType::from_id(b_type_id);
  vllm::ScalarType c_type = vllm::ScalarType::from_id(c_type_id);
  vllm::ScalarType s_type = vllm::ScalarType::from_id(s_type_id);

  TORCH_CHECK(a_type == vllm::kFloat16,
              "SM70 Marlin MoE supports only float16 "
              "activations.");
  TORCH_CHECK(c_type == vllm::kFloat16,
              "SM70 Marlin MoE supports only float16 outputs.");
  TORCH_CHECK(s_type == vllm::kFloat16 ||
                  (b_type == vllm::kFE2M1f &&
                   (s_type == vllm::kFE4M3fn ||
                    s_type == vllm::kFE8M0fnu)),
              "SM70 Marlin MoE supports only float16 scales, "
              "except FP4 uses float8_e4m3fn scales for NVFP4 or "
              "float8_e8m0fnu scales for MXFP4.");
  TORCH_CHECK(b_type == vllm::kU4 || b_type == vllm::kU4B8 ||
                  b_type == vllm::kU8 || b_type == vllm::kU8B128 ||
                  b_type == vllm::kFE4M3fn || b_type == vllm::kFE2M1f,
              "SM70 Marlin MoE supports uint4, uint4b8, uint8, "
              "uint8b128, fp8_e4m3fn, nvfp4, and mxfp4 weights.");
  (void)use_fp32_reduce;

  int pack_factor = 32 / b_type.size_bits();
  int num_experts = b_q_weight.size(0);

  if (moe_block_size != 8) {
    TORCH_CHECK(moe_block_size % 16 == 0,
                "unsupported moe_block_size=", moe_block_size);
    TORCH_CHECK(moe_block_size >= 16 && moe_block_size <= 64,
                "unsupported moe_block_size=", moe_block_size);
  }

  // Verify A
  TORCH_CHECK(a.size(0) == size_m, "Shape mismatch: a.size(0) = ", a.size(0),
              ", size_m = ", size_m);
  TORCH_CHECK(a.size(1) == size_k, "Shape mismatch: a.size(1) = ", a.size(1),
              ", size_k = ", size_k);

  // Verify B
  TORCH_CHECK(
      size_k % MARLIN_NAMESPACE_NAME::tile_size == 0, "size_k = ", size_k,
      " is not divisible by tile_size = ", MARLIN_NAMESPACE_NAME::tile_size);
  TORCH_CHECK((size_k / MARLIN_NAMESPACE_NAME::tile_size) == b_q_weight.size(1),
              "Shape mismatch: b_q_weight.size(1) = ", b_q_weight.size(1),
              ", size_k = ", size_k,
              ", tile_size = ", MARLIN_NAMESPACE_NAME::tile_size);
  TORCH_CHECK(
      b_q_weight.size(2) % MARLIN_NAMESPACE_NAME::tile_size == 0,
      "b_q_weight.size(2) = ", b_q_weight.size(2),
      " is not divisible by tile_size = ", MARLIN_NAMESPACE_NAME::tile_size);
  int actual_size_n =
      (b_q_weight.size(2) / MARLIN_NAMESPACE_NAME::tile_size) * pack_factor;
  TORCH_CHECK(size_n == actual_size_n, "size_n = ", size_n,
              ", actual_size_n = ", actual_size_n);

  // Verify device and strides
  TORCH_CHECK(a.device().is_cuda(), "A is not on GPU");
  TORCH_CHECK(a.is_contiguous(), "A is not contiguous");

  TORCH_CHECK(b_q_weight.device().is_cuda(), "b_q_weight is not on GPU");
  TORCH_CHECK(b_q_weight.is_contiguous(), "b_q_weight is not contiguous");

  TORCH_CHECK(b_scales.device().is_cuda(), "b_scales is not on GPU");
  TORCH_CHECK(b_scales.is_contiguous(), "b_scales is not contiguous");
  TORCH_CHECK(b_scales.scalar_type() == at::ScalarType::Half ||
                  (b_type == vllm::kFE2M1f &&
                   (b_scales.scalar_type() ==
                        at::ScalarType::Float8_e4m3fn ||
                    b_scales.scalar_type() ==
                        at::ScalarType::Float8_e8m0fnu)),
              "SM70 Marlin MoE supports float16 scales, except FP4 uses "
              "float8_e4m3fn scales for NVFP4 or float8_e8m0fnu scales for "
              "MXFP4.");

  torch::Tensor a_scales;
  auto options = torch::TensorOptions().dtype(c_dtype).device(a.device());
  auto options_fp32 =
      torch::TensorOptions().dtype(at::kFloat).device(a.device());

  if (a_scales_or_none.has_value()) {
    a_scales = a_scales_or_none.value();
    TORCH_CHECK(a_type.size_bits() == 8,
                "a_scales can only be used for 8bit activation.");
  } else {
    a_scales = torch::empty({0}, options_fp32);
    TORCH_CHECK(a_type.size_bits() != 8,
                "the a_scales parameter must be passed for 8bit activation.");
  }

  // Alloc buffers
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  torch::Tensor c;
  if (c_or_none.has_value()) {
    c = c_or_none.value();
    TORCH_CHECK(c.device().is_cuda(), "c is not on GPU");
    TORCH_CHECK(c.is_contiguous(), "c is not contiguous");
    TORCH_CHECK(c.size(0) == size_m * top_k,
                "Shape mismatch: c.size(0) = ", c.size(0),
                ", size_m * topk = ", size_m * top_k);
    TORCH_CHECK(c.size(1) == size_n, "Shape mismatch: c.size(1) = ", c.size(1),
                ", size_n = ", size_n);
  } else {
    c = torch::empty({size_m * top_k, size_n}, options);
  }
  // Detect groupsize and act_order
  int num_groups = -1;
  int group_size = -1;

  int rank = b_scales.sizes().size();
  TORCH_CHECK(rank == 3, "b_scales rank = ", rank, " is not 3");
  TORCH_CHECK(b_scales.size(2) == size_n, "b_scales dim 2 = ", b_scales.size(2),
              " is not size_n = ", size_n);
  num_groups = b_scales.size(1);

  torch::Tensor g_idx, perm;
  if (g_idx_or_none.has_value() && perm_or_none.has_value()) {
    g_idx = g_idx_or_none.value();
    perm = perm_or_none.value();

    TORCH_CHECK(g_idx.device().is_cuda(), "g_idx is not on GPU");
    TORCH_CHECK(g_idx.is_contiguous(), "g_idx is not contiguous");
    TORCH_CHECK(perm.device().is_cuda(), "perm is not on GPU");
    TORCH_CHECK(perm.is_contiguous(), "perm is not contiguous");

    // Verify g_idx and perm
    TORCH_CHECK((g_idx.size(-1) == 0 && perm.size(-1) == 0) ||
                    (g_idx.size(-1) == size_k && perm.size(-1) == size_k),
                "Unexpected g_idx.size(-1) = ", g_idx.size(-1),
                " and perm.size(-1) = ", perm.size(-1),
                ", where size_k = ", size_k);
  } else {
    g_idx = torch::empty({0}, options);
    perm = torch::empty({0}, options);
  }
  bool has_act_order = g_idx.size(-1) > 0 && perm.size(-1) > 0;

  if (has_act_order) {
    if (is_k_full) {
      TORCH_CHECK(num_groups > 1, "For act_order, num_groups must be > 1");
      TORCH_CHECK(size_k % num_groups == 0, "size_k = ", size_k,
                  ", is not divisible by num_groups = ", num_groups);
      group_size = size_k / num_groups;
    } else {
      group_size = 0;
    }

  } else {
    if (num_groups > 1) {
      TORCH_CHECK(
          size_k % num_groups == 0, "size_k = ", size_k,
          ", is not divisible by b_scales.size(1) = ", b_scales.size(1));
      group_size = size_k / num_groups;
    } else {
      group_size = -1;
    }
  }

  torch::Tensor b_scales_kernel = b_scales;
  if (b_scales.scalar_type() == at::ScalarType::Half) {
    b_scales_kernel =
        sm70_moe_scales_as_logical(b_scales, size_k, group_size,
                                   a_type.size_bits() == 8);
  }

  torch::Tensor global_scale;
  if (global_scale_or_none.has_value()) {
    global_scale = global_scale_or_none.value();
    TORCH_CHECK(
        b_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn,
        "SM70 Marlin MoE supports global_scale only for nvfp4 format.");
    TORCH_CHECK(global_scale.device().is_cuda(), "global_scale is not on GPU");
    TORCH_CHECK(global_scale.is_contiguous(), "global_scale is not contiguous");
    TORCH_CHECK(global_scale.scalar_type() == at::ScalarType::Float,
                "SM70 Marlin MoE NVFP4 expects fp32 global_scale.");
    TORCH_CHECK(global_scale.numel() == num_experts,
                "SM70 Marlin MoE NVFP4 expects global_scale numel = "
                "num_experts. Got global_scale.numel() = ",
                global_scale.numel(), ", num_experts = ", num_experts);
  } else {
    global_scale = torch::empty({0}, options_fp32);
    TORCH_CHECK(!(b_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn),
                "the global_scale parameter must be passed for nvfp4 format.");
  }

  bool has_bias = b_bias_or_none.has_value();
  torch::Tensor b_bias;
  if (has_bias) {
    b_bias = b_bias_or_none.value();
    TORCH_CHECK(b_bias.device().is_cuda(), "b_bias is not on GPU");
    TORCH_CHECK(b_bias.is_contiguous(), "b_bias is not contiguous");
    TORCH_CHECK(b_bias.size(1) == size_n, "b_bias.size(1) != size_n");
    TORCH_CHECK(b_bias.stride(1) == 1, "b_bias.stride(1) != 1");
  } else {
    b_bias = torch::empty({0}, options);
  }

  torch::Tensor b_zeros;
  if (b_zeros_or_none.has_value()) {
    b_zeros = b_zeros_or_none.value();
    TORCH_CHECK(b_zeros.device().is_cuda(), "b_zeros is not on GPU");
    TORCH_CHECK(b_zeros.is_contiguous(), "b_zeros is not contiguous");
  } else {
    b_zeros = torch::empty({0}, options);
  }
  bool has_zp = b_zeros.size(-1) > 0;
  bool zp_is_float = is_zp_float;
  if (has_zp) {
    TORCH_CHECK(b_type == vllm::kU4 || b_type == vllm::kU8,
                "SM70 Marlin MoE supports only uint4 or uint8 "
                "weights when zero-points are enabled. Got = ",
                b_type.str());
    if (!is_zp_float && b_zeros.scalar_type() == at::ScalarType::Int) {
      b_zeros = sm70_moe_zp_as_half(
          b_zeros, b_scales_kernel, size_n, b_type == vllm::kU4 ? 4 : 8,
          a_type.size_bits() == 8);
      zp_is_float = true;
    }
  } else {
    TORCH_CHECK(!(b_type == vllm::kU4 || b_type == vllm::kU8),
                "SM70 Marlin MoE uint4/uint8 paths require fp16 zero points "
                "with is_zp_float=true.");
  }

  // Verify b_zeros
  if (has_zp) {
    TORCH_CHECK(zp_is_float,
                "SM70 Marlin MoE received unsupported zero-point metadata.");
    int rank = b_zeros.sizes().size();
    TORCH_CHECK(rank == 3, "b_zeros rank = ", rank, " is not 3");
    TORCH_CHECK(b_zeros.scalar_type() == at::ScalarType::Half,
                "SM70 Marlin MoE uint4/uint8 paths expect fp16 zero points.");
    TORCH_CHECK(b_zeros.size(0) == num_experts,
                "b_zeros dim 0 = ", b_zeros.size(0),
                " is not num_experts = ", num_experts);
    TORCH_CHECK(num_groups == b_zeros.size(1),
                "b_zeros dim 1 = ", b_zeros.size(1),
                " is not num_groups = ", num_groups);
    TORCH_CHECK(b_zeros.size(2) == size_n,
                "b_zeros dim 2 = ", b_zeros.size(2),
                " is not size_n = ", size_n);
  } else {
    TORCH_CHECK(!zp_is_float,
                "is_zp_float is true but b_zeros was not provided.");
  }

  TORCH_CHECK(size_n % MARLIN_NAMESPACE_NAME::min_thread_n == 0,
              "size_n = ", size_n, ", is not divisible by min_thread_n = ",
              MARLIN_NAMESPACE_NAME::min_thread_n);

  TORCH_CHECK(a_scales.scalar_type() == at::ScalarType::Float,
              "scalar type of a_scales must be float");
  TORCH_CHECK(global_scale.scalar_type() == at::ScalarType::Float,
              "scalar type of global_scale must be float");
  if (a_type.size_bits() == 16) {
    TORCH_CHECK(
        a.scalar_type() == c.scalar_type(),
        "scalar type of a must be the same with c for 16 bit activation");
  }

  TORCH_CHECK(!has_act_order,
              "act_order is not supported for the SM70 Marlin MoE path.");
  TORCH_CHECK(!has_bias,
              "SM70 Marlin MoE does not support bias.");
  TORCH_CHECK(!use_atomic_add,
              "SM70 Marlin MoE does not support atomic-add "
              "output.");
  TORCH_CHECK(is_k_full,
              "SM70 Marlin MoE requires full-K inputs.");

  if (b_type == vllm::kU4) {
    TORCH_CHECK(has_zp && zp_is_float,
                "SM70 Marlin MoE uint4 path requires fp16 zero points.");
    return MARLIN_NAMESPACE_NAME::sm70_marlin_u4_gemm(
        a, c, b_q_weight, b_scales_kernel, b_zeros, sorted_token_ids, expert_ids,
        num_tokens_past_padded, topk_weights, moe_block_size, top_k,
        mul_topk_weights, size_m, size_n, size_k, group_size);
  }

  if (b_type == vllm::kU8) {
    TORCH_CHECK(has_zp && zp_is_float,
                "SM70 Marlin MoE uint8 path requires fp16 zero points.");
    return MARLIN_NAMESPACE_NAME::sm70_marlin_u8_gemm(
        a, c, b_q_weight, b_scales_kernel, b_zeros, sorted_token_ids, expert_ids,
        num_tokens_past_padded, topk_weights, moe_block_size, top_k,
        mul_topk_weights, size_m, size_n, size_k, group_size);
  }

  if (b_type == vllm::kU4B8) {
    TORCH_CHECK(!has_zp && !is_zp_float,
                "SM70 Marlin MoE uint4b8 does not support zero-point "
                "metadata.");
    return MARLIN_NAMESPACE_NAME::sm70_marlin_u4b8_gemm(
        a, c, b_q_weight, b_scales_kernel, sorted_token_ids, expert_ids,
        num_tokens_past_padded, topk_weights, moe_block_size, top_k,
        mul_topk_weights, size_m, size_n, size_k, group_size);
  }

  if (b_type == vllm::kU8B128) {
    TORCH_CHECK(!has_zp && !is_zp_float,
                "SM70 Marlin MoE uint8b128 does not support zero-point "
                "metadata.");
    return MARLIN_NAMESPACE_NAME::sm70_marlin_u8b128_gemm(
        a, c, b_q_weight, b_scales_kernel, sorted_token_ids, expert_ids,
        num_tokens_past_padded, topk_weights, moe_block_size, top_k,
        mul_topk_weights, size_m, size_n, size_k, group_size);
  }

  if (b_type == vllm::kFE4M3fn) {
    TORCH_CHECK(group_size == -1 || group_size == 128,
                "SM70 Marlin MoE FP8 supports only group_size -1 or "
                "128. Got ",
                group_size);
    TORCH_CHECK(!has_zp && !is_zp_float,
                "SM70 Marlin MoE fp8 does not support zero-point metadata.");
    return MARLIN_NAMESPACE_NAME::sm70_marlin_fp8_gemm(
        a, c, b_q_weight, b_scales_kernel, sorted_token_ids, expert_ids,
        num_tokens_past_padded, topk_weights, moe_block_size, top_k,
        mul_topk_weights, size_m, size_n, size_k, group_size);
  }

  if (b_type == vllm::kFE2M1f) {
    TORCH_CHECK(!has_zp && !is_zp_float,
                "SM70 Marlin MoE fp4 does not support zero-point metadata.");
    if (s_type == vllm::kFE4M3fn) {
      TORCH_CHECK(global_scale.numel() == num_experts,
                  "the global_scale parameter must be passed for nvfp4 format.");
      TORCH_CHECK(group_size == 16,
                  "SM70 Marlin MoE NVFP4 supports only group_size 16. "
                  "Got ",
                  group_size);
      return MARLIN_NAMESPACE_NAME::sm70_marlin_nvfp4_gemm(
          a, c, b_q_weight, b_scales_kernel, global_scale, sorted_token_ids,
          expert_ids, num_tokens_past_padded, topk_weights, moe_block_size,
          top_k, mul_topk_weights, size_m, size_n, size_k, group_size);
    }

    TORCH_CHECK(s_type == vllm::kFE8M0fnu,
                "SM70 Marlin MoE MXFP4 expects float8_e8m0fnu scales.");
    TORCH_CHECK(group_size == 32,
                "SM70 Marlin MoE MXFP4 supports only group_size 32. "
                "Got ",
                group_size);
    return MARLIN_NAMESPACE_NAME::sm70_marlin_mxfp4_gemm(
        a, c, b_q_weight, b_scales_kernel, sorted_token_ids, expert_ids,
        num_tokens_past_padded, topk_weights, moe_block_size, top_k,
        mul_topk_weights, size_m, size_n, size_k, group_size);
  }

  TORCH_CHECK(false, "Unsupported SM70 Marlin MoE weight type.");
  return c;
}

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("moe_wna16_marlin_gemm", &moe_wna16_marlin_gemm);
}
