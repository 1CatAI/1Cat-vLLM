// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <torch/library.h>
#include <cmath>
#include <initializer_list>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

// Exact E4M3 storage expansion with the validated eight-byte feed.
namespace {
template <int FORMAT>
__device__ __forceinline__ __half decode_byte(uint8_t raw) {
  if constexpr (FORMAT == 2) {
    return __ushort_as_half(static_cast<uint16_t>(raw) << 8);
  } else {
    const uint16_t sign = static_cast<uint16_t>(raw & 0x80u) << 8;
    const uint16_t magnitude = raw & 0x7fu;
    uint16_t bits;
    if (magnitude == 0x7fu) {
      bits = 0x7e00u;
    } else if (magnitude < 8) {
      bits = __half_as_ushort(__float2half_rn(float(magnitude) * 0x1p-9f));
    } else {
      bits = (magnitude << 7) + 0x2000u;
    }
    return __ushort_as_half(sign | bits);
  }
}

template <int FORMAT>
__device__ __forceinline__ __half2 decode_pair(const uint16_t raw) {
  if constexpr (FORMAT == 2) {
    union {
      uint32_t u;
      __half2 h;
    } pair;
    pair.u = (uint32_t(raw & 0xffu) << 8) | (uint32_t(raw & 0xff00u) << 16);
    return pair.h;
  } else if constexpr (FORMAT >= 3) {
    // Exact E4M3 conversion: the intermediate half2 equals value / 256.
    // This is a power-of-two format conversion, never an FP16 accumulator.
    if ((raw & 0x7fu) == 0x7fu || ((raw >> 8) & 0x7fu) == 0x7fu) {
      return __halves2half2(decode_byte<1>(raw & 0xffu),
                            decode_byte<1>(raw >> 8));
    }
    union {
      uint32_t u;
      __half2 h;
    } pair;
    pair.u = (uint32_t(raw & 0x0080u) << 8) | (uint32_t(raw & 0x007fu) << 7) |
             (uint32_t(raw & 0x8000u) << 16) | (uint32_t(raw & 0x7f00u) << 15);
    return __hmul2(pair.h, __float2half2_rn(256.f));
  } else {
    return __halves2half2(decode_byte<FORMAT>(raw & 0xffu),
                          decode_byte<FORMAT>(raw >> 8));
  }
}
template <int FORMAT>
__device__ __forceinline__ __half2 load_pair(const void* cache,
                                             int64_t offset) {
  return decode_pair<FORMAT>(
      reinterpret_cast<const uint16_t*>(cache)[offset >> 1]);
}
}  // namespace

namespace {

constexpr int kThreads = 256;

template <int FORMAT, bool UNIT_SCALE>
__global__ void fp8_paged_kv_to_fp16_kernel(
    const void* __restrict__ key_cache, const void* __restrict__ value_cache,
    const int* __restrict__ block_table, const int* __restrict__ seq_lens,
    __half* __restrict__ key_out, __half* __restrict__ value_out,
    int batch_size, int max_num_blocks, int input_block_size,
    int output_blocks_per_seq, int output_block_size, int num_heads,
    int head_dim, int64_t key_block_stride, int64_t key_token_stride,
    int64_t key_head_stride, int64_t value_block_stride,
    int64_t value_token_stride, int64_t value_head_stride,
    int64_t key_out_block_stride, int64_t key_out_token_stride,
    int64_t key_out_head_stride, int64_t value_out_block_stride,
    int64_t value_out_token_stride, int64_t value_out_head_stride,
    float key_scale, float value_scale) {
  const int batch_idx = blockIdx.y;
  if (batch_idx >= batch_size) {
    return;
  }

  constexpr int kPairs = FORMAT == 4 ? 4 : 1;
  const int pairs_per_head = head_dim / (2 * kPairs);
  const int64_t pairs_per_token =
      static_cast<int64_t>(num_heads) * pairs_per_head;
  const int max_tokens = output_blocks_per_seq * output_block_size;
  const int64_t total_pairs =
      static_cast<int64_t>(max_tokens) * pairs_per_token;
  const int64_t pair_idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair_idx >= total_pairs) {
    return;
  }

  const int token_idx = static_cast<int>(pair_idx / pairs_per_token);
  const int seq_len = seq_lens[batch_idx];
  // Paged prefill loads complete 16-row WMMA tiles. Zero the partial tail so
  // masked probabilities cannot multiply uninitialized workspace values.
  const int padded_seq_len = min(max_tokens, (seq_len + 15) & ~15);
  if (token_idx >= padded_seq_len) {
    return;
  }
  const int pair_in_token = static_cast<int>(pair_idx % pairs_per_token);
  const int head_idx = pair_in_token / pairs_per_head;
  const int element_idx = (pair_in_token % pairs_per_head) * 2 * kPairs;

  const __half2 zero = __float2half2_rn(0.0f);
  __half2 key_pair[kPairs], value_pair[kPairs];
#pragma unroll
  for (int p = 0; p < kPairs; ++p) {
    key_pair[p] = zero;
    value_pair[p] = zero;
  }
  if (token_idx < seq_len) {
    const int logical_block = token_idx / input_block_size;
    const int input_block_offset = token_idx - logical_block * input_block_size;
    const int physical_block =
        __ldg(&block_table[batch_idx * max_num_blocks + logical_block]);
    const int64_t key_input_offset =
        static_cast<int64_t>(physical_block) * key_block_stride +
        static_cast<int64_t>(input_block_offset) * key_token_stride +
        static_cast<int64_t>(head_idx) * key_head_stride + element_idx;
    const int64_t value_input_offset =
        static_cast<int64_t>(physical_block) * value_block_stride +
        static_cast<int64_t>(input_block_offset) * value_token_stride +
        static_cast<int64_t>(head_idx) * value_head_stride + element_idx;

    if constexpr (kPairs == 4) {
      const auto kr = *reinterpret_cast<const uint64_t*>(
          static_cast<const uint8_t*>(key_cache) + key_input_offset);
      const auto vr = *reinterpret_cast<const uint64_t*>(
          static_cast<const uint8_t*>(value_cache) + value_input_offset);
#pragma unroll
      for (int p = 0; p < kPairs; ++p) {
        key_pair[p] =
            decode_pair<FORMAT>(static_cast<uint16_t>(kr >> (16 * p)));
        value_pair[p] =
            decode_pair<FORMAT>(static_cast<uint16_t>(vr >> (16 * p)));
      }
    } else {
      key_pair[0] = load_pair<FORMAT>(key_cache, key_input_offset);
      value_pair[0] = load_pair<FORMAT>(value_cache, value_input_offset);
    }
    if constexpr (!UNIT_SCALE) {
#pragma unroll
      for (int p = 0; p < kPairs; ++p) {
        const float2 key_values = __half22float2(key_pair[p]);
        const float2 value_values = __half22float2(value_pair[p]);
        key_pair[p] = __float22half2_rn(
            make_float2(key_values.x * key_scale, key_values.y * key_scale));
        value_pair[p] = __float22half2_rn(make_float2(
            value_values.x * value_scale, value_values.y * value_scale));
      }
    }
  }

  const int output_block_offset = token_idx / output_block_size;
  const int output_token_offset =
      token_idx - output_block_offset * output_block_size;
  const int output_block =
      batch_idx * output_blocks_per_seq + output_block_offset;
  const int64_t key_output_offset =
      static_cast<int64_t>(output_block) * key_out_block_stride +
      static_cast<int64_t>(output_token_offset) * key_out_token_stride +
      static_cast<int64_t>(head_idx) * key_out_head_stride + element_idx;
  const int64_t value_output_offset =
      static_cast<int64_t>(output_block) * value_out_block_stride +
      static_cast<int64_t>(output_token_offset) * value_out_token_stride +
      static_cast<int64_t>(head_idx) * value_out_head_stride + element_idx;
  if constexpr (kPairs == 4) {
    union {
      uint4 u;
      __half2 h[4];
    } kr, vr;
#pragma unroll
    for (int p = 0; p < kPairs; ++p) {
      kr.h[p] = key_pair[p];
      vr.h[p] = value_pair[p];
    }
    *reinterpret_cast<uint4*>(key_out + key_output_offset) = kr.u;
    *reinterpret_cast<uint4*>(value_out + value_output_offset) = vr.u;
  } else {
    *reinterpret_cast<__half2*>(key_out + key_output_offset) = key_pair[0];
    *reinterpret_cast<__half2*>(value_out + value_output_offset) =
        value_pair[0];
  }
}

}  // namespace

void sm70_v37_e4m3_paged_kv_to_fp16(const at::Tensor& key_cache,
                                    const at::Tensor& value_cache,
                                    const at::Tensor& block_table,
                                    const at::Tensor& seq_lens,
                                    at::Tensor& key_out, at::Tensor& value_out,
                                    const double key_scale_arg,
                                    const double value_scale_arg) {
  constexpr int format = 4;
  const float key_scale = static_cast<float>(key_scale_arg);
  const float value_scale = static_cast<float>(value_scale_arg);
  TORCH_CHECK(key_cache.is_cuda() && value_cache.is_cuda() &&
                  key_out.is_cuda() && value_out.is_cuda(),
              "FP8 KV bridge tensors must be CUDA tensors");
  TORCH_CHECK(block_table.is_cuda() && seq_lens.is_cuda(),
              "FP8 KV bridge metadata must be CUDA tensors");
  TORCH_CHECK(key_cache.scalar_type() == at::kByte &&
                  value_cache.scalar_type() == at::kByte,
              "FP8 input caches must be stored as uint8");
  TORCH_CHECK(key_out.scalar_type() == at::kHalf &&
                  value_out.scalar_type() == at::kHalf,
              "FP8 KV bridge output caches must be fp16");
  TORCH_CHECK(block_table.scalar_type() == at::kInt &&
                  seq_lens.scalar_type() == at::kInt,
              "FP8 KV bridge block_table and seq_lens must be int32");
  TORCH_CHECK(key_cache.dim() == 4 && value_cache.dim() == 4 &&
                  key_out.dim() == 4 && value_out.dim() == 4,
              "FP8 KV bridge expects paged [blocks,tokens,heads,dim] caches");
  TORCH_CHECK(key_cache.sizes() == value_cache.sizes(),
              "FP8 K/V input cache shapes must match");
  TORCH_CHECK(key_out.sizes() == value_out.sizes(),
              "FP16 K/V output cache shapes must match");
  TORCH_CHECK(key_cache.size(2) == key_out.size(2) &&
                  key_cache.size(3) == key_out.size(3),
              "FP8 KV bridge head shape must not change");
  TORCH_CHECK(key_cache.stride(3) == 1 && value_cache.stride(3) == 1 &&
                  key_out.stride(3) == 1 && value_out.stride(3) == 1,
              "FP8 KV bridge requires contiguous head dimensions");
  TORCH_CHECK(key_cache.size(3) % 2 == 0,
              "FP8 KV bridge requires an even head dimension");
  TORCH_CHECK(block_table.dim() == 2 && seq_lens.dim() == 1 &&
                  block_table.size(0) == seq_lens.size(0),
              "FP8 KV bridge metadata shape mismatch");
  TORCH_CHECK(key_scale > 0.f && value_scale > 0.f,
              "FP8 KV bridge scales must be positive");

  const int batch_size = block_table.size(0);
  TORCH_CHECK(batch_size > 0, "FP8 KV bridge batch must be non-empty");
  TORCH_CHECK(key_out.size(0) % batch_size == 0,
              "FP16 output blocks must divide evenly across the batch");
  const int output_blocks_per_seq = key_out.size(0) / batch_size;
  const int input_capacity = block_table.size(1) * key_cache.size(1);
  const int output_capacity = output_blocks_per_seq * key_out.size(1);
  TORCH_CHECK(output_capacity >= input_capacity,
              "FP16 output cache capacity must cover the input block table");

  TORCH_CHECK(format >= 1 && format <= 4,
              "format: 1 E4M3 scalar, 2 E5M2, 3 E4M3 packed, 4 E4M3 wide");
  TORCH_CHECK(std::isfinite(key_scale) && std::isfinite(value_scale),
              "finite scales required");
  TORCH_CHECK(block_table.is_contiguous() && seq_lens.is_contiguous(),
              "contiguous metadata required");
  for (const at::Tensor* tensor : std::initializer_list<const at::Tensor*>{
           &value_cache, &block_table, &seq_lens, &key_out, &value_out}) {
    TORCH_CHECK(tensor->device() == key_cache.device(),
                "all inputs must share a CUDA device");
  }
  for (const at::Tensor* tensor : std::initializer_list<const at::Tensor*>{
           &key_cache, &value_cache, &key_out, &value_out}) {
    TORCH_CHECK(
        tensor->size(1) > 0 && tensor->size(2) > 0 && tensor->size(3) > 0,
        "empty head/page unsupported");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor->data_ptr()) % 2 == 0,
                "paired loads require aligned pointers");
    for (int axis = 0; axis < 3; ++axis) {
      TORCH_CHECK(tensor->stride(axis) % 2 == 0,
                  "paired loads require even strides");
    }
    if (format == 4) {
      TORCH_CHECK(tensor->size(3) % 8 == 0 &&
                      reinterpret_cast<uintptr_t>(tensor->data_ptr()) % 16 == 0,
                  "wide conversion requires D divisible by 8 and 16-byte "
                  "aligned pointers");
      for (int axis = 0; axis < 3; ++axis)
        TORCH_CHECK(tensor->stride(axis) % 8 == 0,
                    "wide conversion requires 8-element aligned strides");
    }
  }
  c10::cuda::CUDAGuard device_guard(key_cache.device());
  const int64_t total_pairs = static_cast<int64_t>(output_capacity) *
                              key_cache.size(2) *
                              (key_cache.size(3) / (format == 4 ? 8 : 2));
  const dim3 grid(
      static_cast<unsigned int>((total_pairs + kThreads - 1) / kThreads),
      batch_size);
  const auto stream = at::cuda::getCurrentCUDAStream().stream();

#define LAUNCH_FP8_BRIDGE(FORMAT, UNIT_SCALE)                                  \
  fp8_paged_kv_to_fp16_kernel<FORMAT, UNIT_SCALE>                              \
      <<<grid, kThreads, 0, stream>>>(                                         \
          key_cache.data_ptr(), value_cache.data_ptr(),                        \
          block_table.data_ptr<int>(), seq_lens.data_ptr<int>(),               \
          reinterpret_cast<__half*>(key_out.data_ptr()),                       \
          reinterpret_cast<__half*>(value_out.data_ptr()), batch_size,         \
          block_table.size(1), key_cache.size(1), output_blocks_per_seq,       \
          key_out.size(1), key_cache.size(2), key_cache.size(3),               \
          key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),       \
          value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), \
          key_out.stride(0), key_out.stride(1), key_out.stride(2),             \
          value_out.stride(0), value_out.stride(1), value_out.stride(2),       \
          key_scale, value_scale)

  if (format == 4) {
    if (key_scale == 1.f && value_scale == 1.f) {
      LAUNCH_FP8_BRIDGE(4, true);
    } else {
      LAUNCH_FP8_BRIDGE(4, false);
    }
  }
#undef LAUNCH_FP8_BRIDGE

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace {

__device__ __forceinline__ __half2 bridge_int8_pair_unscaled(uint32_t word,
                                                             uint32_t sel) {
  const uint32_t biased = __byte_perm(word, 0x64646464u, sel) ^ 0x00800080u;
  const uint32_t offset = 0x64806480u;
  return __hsub2(*reinterpret_cast<const __half2*>(&biased),
                 *reinterpret_cast<const __half2*>(&offset));
}

__device__ __forceinline__ uint8_t quant_int8_rni_byte(float x) {
  int q;
  asm("cvt.rni.s32.f32 %0, %1;" : "=r"(q) : "f"(x));
  q = max(-127, min(127, q));
  return static_cast<uint8_t>(static_cast<int8_t>(q));
}

__device__ __forceinline__ uint64_t pack_int8_8(const float vals[8],
                                                float inv_scale) {
  uint32_t lo =
      static_cast<uint32_t>(quant_int8_rni_byte(vals[0] * inv_scale)) |
      (static_cast<uint32_t>(quant_int8_rni_byte(vals[1] * inv_scale)) << 8) |
      (static_cast<uint32_t>(quant_int8_rni_byte(vals[2] * inv_scale)) << 16) |
      (static_cast<uint32_t>(quant_int8_rni_byte(vals[3] * inv_scale)) << 24);
  uint32_t hi =
      static_cast<uint32_t>(quant_int8_rni_byte(vals[4] * inv_scale)) |
      (static_cast<uint32_t>(quant_int8_rni_byte(vals[5] * inv_scale)) << 8) |
      (static_cast<uint32_t>(quant_int8_rni_byte(vals[6] * inv_scale)) << 16) |
      (static_cast<uint32_t>(quant_int8_rni_byte(vals[7] * inv_scale)) << 24);
  return static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
}

template <typename SlotT>
__global__ __launch_bounds__(128) void sm70_reshape_and_cache_int8_pth_kernel(
    const __half* __restrict__ key, const __half* __restrict__ value,
    int8_t* __restrict__ key_cache, int8_t* __restrict__ value_cache,
    float* __restrict__ k_scales, float* __restrict__ v_scales,
    const SlotT* __restrict__ slot_mapping, int64_t total_items, int num_heads,
    int head_dim, int block_size, int num_blocks, int64_t key_tok_stride,
    int64_t key_head_stride, int64_t val_tok_stride, int64_t val_head_stride,
    int64_t kc_blk_stride, int64_t kc_tok_stride, int64_t kc_head_stride,
    int64_t vc_blk_stride, int64_t vc_tok_stride, int64_t vc_head_stride,
    int64_t ks_blk_stride, int64_t ks_tok_stride, int64_t ks_head_stride,
    int64_t vs_blk_stride, int64_t vs_tok_stride, int64_t vs_head_stride) {
  const int warp_in_cta = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int64_t item_idx = static_cast<int64_t>(blockIdx.x) * 4 + warp_in_cta;
  if (item_idx >= total_items) {
    return;
  }
  const int64_t tok_idx = item_idx / num_heads;
  const int head_idx = static_cast<int>(item_idx % num_heads);
  const int64_t slot = static_cast<int64_t>(slot_mapping[tok_idx]);
  if (slot < 0) {
    return;
  }
  const int64_t block_idx = slot / block_size;
  if (block_idx >= num_blocks) {
    return;
  }
  const int64_t block_offset = slot - block_idx * block_size;

  const __half* k_src =
      key + tok_idx * key_tok_stride + head_idx * key_head_stride;
  const __half* v_src =
      value + tok_idx * val_tok_stride + head_idx * val_head_stride;
  int8_t* k_dst = key_cache + block_idx * kc_blk_stride +
                  block_offset * kc_tok_stride + head_idx * kc_head_stride;
  int8_t* v_dst = value_cache + block_idx * vc_blk_stride +
                  block_offset * vc_tok_stride + head_idx * vc_head_stride;

  const int num_vecs = head_dim >> 3;  // head_dim / 8, <= 64 (max 2 iters)
  float k_vals[2][8];
  float v_vals[2][8];
  float k_amax = 0.0f;
  float v_amax = 0.0f;

#pragma unroll
  for (int iter = 0; iter < 2; ++iter) {
    const int vec_idx = lane + iter * 32;
    if (vec_idx < num_vecs) {
      const int col = vec_idx << 3;
      if ((reinterpret_cast<uintptr_t>(k_src + col) & 15u) == 0u) {
        union {
          uint4 u;
          __half2 h[4];
        } kv_u;
        kv_u.u = *reinterpret_cast<const uint4*>(k_src + col);
#pragma unroll
        for (int p = 0; p < 4; ++p) {
          const float2 f2 = __half22float2(kv_u.h[p]);
          k_vals[iter][2 * p + 0] = f2.x;
          k_vals[iter][2 * p + 1] = f2.y;
          k_amax = fmaxf(k_amax, fmaxf(fabsf(f2.x), fabsf(f2.y)));
        }
      } else {
#pragma unroll
        for (int c = 0; c < 8; ++c) {
          const float fk = __half2float(k_src[col + c]);
          k_vals[iter][c] = fk;
          k_amax = fmaxf(k_amax, fabsf(fk));
        }
      }
      if ((reinterpret_cast<uintptr_t>(v_src + col) & 15u) == 0u) {
        union {
          uint4 u;
          __half2 h[4];
        } vv_u;
        vv_u.u = *reinterpret_cast<const uint4*>(v_src + col);
#pragma unroll
        for (int p = 0; p < 4; ++p) {
          const float2 f2 = __half22float2(vv_u.h[p]);
          v_vals[iter][2 * p + 0] = f2.x;
          v_vals[iter][2 * p + 1] = f2.y;
          v_amax = fmaxf(v_amax, fmaxf(fabsf(f2.x), fabsf(f2.y)));
        }
      } else {
#pragma unroll
        for (int c = 0; c < 8; ++c) {
          const float fv = __half2float(v_src[col + c]);
          v_vals[iter][c] = fv;
          v_amax = fmaxf(v_amax, fabsf(fv));
        }
      }
    }
  }

#pragma unroll
  for (int mask = 16; mask >= 1; mask >>= 1) {
    k_amax = fmaxf(k_amax, __shfl_xor_sync(0xffffffffu, k_amax, mask));
    v_amax = fmaxf(v_amax, __shfl_xor_sync(0xffffffffu, v_amax, mask));
  }

  const float k_sc = k_amax > 0.0f ? k_amax * (1.0f / 127.0f) : 1.0f;
  const float v_sc = v_amax > 0.0f ? v_amax * (1.0f / 127.0f) : 1.0f;
  const float k_inv = k_amax > 0.0f ? 127.0f / k_amax : 0.0f;
  const float v_inv = v_amax > 0.0f ? 127.0f / v_amax : 0.0f;

#pragma unroll
  for (int iter = 0; iter < 2; ++iter) {
    const int vec_idx = lane + iter * 32;
    if (vec_idx < num_vecs) {
      reinterpret_cast<uint64_t*>(k_dst)[vec_idx] =
          pack_int8_8(k_vals[iter], k_inv);
      reinterpret_cast<uint64_t*>(v_dst)[vec_idx] =
          pack_int8_8(v_vals[iter], v_inv);
    }
  }
  if (lane == 0) {
    k_scales[block_idx * ks_blk_stride + block_offset * ks_tok_stride +
             head_idx * ks_head_stride] = k_sc;
    v_scales[block_idx * vs_blk_stride + block_offset * vs_tok_stride +
             head_idx * vs_head_stride] = v_sc;
  }
}

__global__ void int8_pth_paged_kv_to_fp16_kernel(
    const int8_t* __restrict__ key_cache,
    const int8_t* __restrict__ value_cache, const float* __restrict__ k_scales,
    const float* __restrict__ v_scales, const int* __restrict__ block_table,
    const int* __restrict__ seq_lens, __half* __restrict__ key_out,
    __half* __restrict__ value_out, int batch_size, int max_num_blocks,
    int input_block_size, int output_blocks_per_seq, int output_block_size,
    int num_heads, int head_dim, int64_t key_block_stride,
    int64_t key_token_stride, int64_t key_head_stride,
    int64_t value_block_stride, int64_t value_token_stride,
    int64_t value_head_stride, int64_t ks_block_stride, int64_t ks_token_stride,
    int64_t ks_head_stride, int64_t vs_block_stride, int64_t vs_token_stride,
    int64_t vs_head_stride, int64_t key_out_block_stride,
    int64_t key_out_token_stride, int64_t key_out_head_stride,
    int64_t value_out_block_stride, int64_t value_out_token_stride,
    int64_t value_out_head_stride) {
  const int batch_idx = blockIdx.y;
  if (batch_idx >= batch_size) {
    return;
  }

  const int vecs_per_head = head_dim >> 3;  // 8 elements per thread
  const int64_t vecs_per_token =
      static_cast<int64_t>(num_heads) * vecs_per_head;
  const int max_tokens = output_blocks_per_seq * output_block_size;
  const int64_t total_vecs = static_cast<int64_t>(max_tokens) * vecs_per_token;
  const int64_t vec_idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (vec_idx >= total_vecs) {
    return;
  }

  const int token_idx = static_cast<int>(vec_idx / vecs_per_token);
  const int seq_len = seq_lens[batch_idx];
  const int padded_seq_len = min(max_tokens, (seq_len + 15) & ~15);
  if (token_idx >= padded_seq_len) {
    return;
  }
  const int vec_in_token = static_cast<int>(vec_idx % vecs_per_token);
  const int head_idx = vec_in_token / vecs_per_head;
  const int element_idx = (vec_in_token % vecs_per_head) << 3;

  union {
    uint4 u;
    __half2 h[4];
  } kr, vr;
  kr.u = make_uint4(0, 0, 0, 0);
  vr.u = make_uint4(0, 0, 0, 0);

  if (token_idx < seq_len) {
    const int logical_block = token_idx / input_block_size;
    const int input_block_offset = token_idx - logical_block * input_block_size;
    const int physical_block =
        __ldg(&block_table[batch_idx * max_num_blocks + logical_block]);
    const int64_t key_input_offset =
        static_cast<int64_t>(physical_block) * key_block_stride +
        static_cast<int64_t>(input_block_offset) * key_token_stride +
        static_cast<int64_t>(head_idx) * key_head_stride + element_idx;
    const int64_t value_input_offset =
        static_cast<int64_t>(physical_block) * value_block_stride +
        static_cast<int64_t>(input_block_offset) * value_token_stride +
        static_cast<int64_t>(head_idx) * value_head_stride + element_idx;
    const float k_sc = __ldg(
        &k_scales[static_cast<int64_t>(physical_block) * ks_block_stride +
                  static_cast<int64_t>(input_block_offset) * ks_token_stride +
                  static_cast<int64_t>(head_idx) * ks_head_stride]);
    const float v_sc = __ldg(
        &v_scales[static_cast<int64_t>(physical_block) * vs_block_stride +
                  static_cast<int64_t>(input_block_offset) * vs_token_stride +
                  static_cast<int64_t>(head_idx) * vs_head_stride]);
    const uint64_t kraw =
        *reinterpret_cast<const uint64_t*>(key_cache + key_input_offset);
    const uint64_t vraw =
        *reinterpret_cast<const uint64_t*>(value_cache + value_input_offset);
    const __half2 k_sc2 = __float2half2_rn(k_sc);
    const __half2 v_sc2 = __float2half2_rn(v_sc);
    const uint32_t klo = static_cast<uint32_t>(kraw);
    const uint32_t khi = static_cast<uint32_t>(kraw >> 32);
    const uint32_t vlo = static_cast<uint32_t>(vraw);
    const uint32_t vhi = static_cast<uint32_t>(vraw >> 32);
    kr.h[0] = __hmul2(bridge_int8_pair_unscaled(klo, 0x5140u), k_sc2);
    kr.h[1] = __hmul2(bridge_int8_pair_unscaled(klo, 0x5342u), k_sc2);
    kr.h[2] = __hmul2(bridge_int8_pair_unscaled(khi, 0x5140u), k_sc2);
    kr.h[3] = __hmul2(bridge_int8_pair_unscaled(khi, 0x5342u), k_sc2);
    vr.h[0] = __hmul2(bridge_int8_pair_unscaled(vlo, 0x5140u), v_sc2);
    vr.h[1] = __hmul2(bridge_int8_pair_unscaled(vlo, 0x5342u), v_sc2);
    vr.h[2] = __hmul2(bridge_int8_pair_unscaled(vhi, 0x5140u), v_sc2);
    vr.h[3] = __hmul2(bridge_int8_pair_unscaled(vhi, 0x5342u), v_sc2);
  }

  const int output_block_offset = token_idx / output_block_size;
  const int output_token_offset =
      token_idx - output_block_offset * output_block_size;
  const int output_block =
      batch_idx * output_blocks_per_seq + output_block_offset;
  const int64_t key_output_offset =
      static_cast<int64_t>(output_block) * key_out_block_stride +
      static_cast<int64_t>(output_token_offset) * key_out_token_stride +
      static_cast<int64_t>(head_idx) * key_out_head_stride + element_idx;
  const int64_t value_output_offset =
      static_cast<int64_t>(output_block) * value_out_block_stride +
      static_cast<int64_t>(output_token_offset) * value_out_token_stride +
      static_cast<int64_t>(head_idx) * value_out_head_stride + element_idx;
  *reinterpret_cast<uint4*>(key_out + key_output_offset) = kr.u;
  *reinterpret_cast<uint4*>(value_out + value_output_offset) = vr.u;
}

}  // namespace

void sm70_reshape_and_cache_int8_pth(const at::Tensor& key,
                                     const at::Tensor& value,
                                     at::Tensor& key_cache,
                                     at::Tensor& value_cache,
                                     at::Tensor& k_scales, at::Tensor& v_scales,
                                     const at::Tensor& slot_mapping) {
  TORCH_CHECK(key.is_cuda() && value.is_cuda() && key_cache.is_cuda() &&
                  value_cache.is_cuda() && k_scales.is_cuda() &&
                  v_scales.is_cuda() && slot_mapping.is_cuda(),
              "INT8 PTH cache writer requires CUDA tensors");
  TORCH_CHECK(
      key.scalar_type() == at::kHalf && value.scalar_type() == at::kHalf,
      "INT8 PTH cache writer requires FP16 key/value");
  TORCH_CHECK(key_cache.scalar_type() == at::kChar &&
                  value_cache.scalar_type() == at::kChar,
              "INT8 PTH caches must be int8");
  TORCH_CHECK(k_scales.scalar_type() == at::kFloat &&
                  v_scales.scalar_type() == at::kFloat,
              "INT8 PTH scale caches must be float32");
  TORCH_CHECK(key.dim() == 3 && value.sizes() == key.sizes(),
              "INT8 PTH key/value shape mismatch");
  TORCH_CHECK(key_cache.dim() == 4 && value_cache.sizes() == key_cache.sizes(),
              "INT8 PTH key_cache/value_cache shape mismatch");
  TORCH_CHECK(k_scales.dim() == 3 && v_scales.sizes() == k_scales.sizes(),
              "INT8 PTH scale cache shape mismatch");
  const int64_t num_tokens = key.size(0);
  if (num_tokens == 0) {
    return;
  }
  const int num_heads = static_cast<int>(key.size(1));
  const int head_dim = static_cast<int>(key.size(2));
  TORCH_CHECK(head_dim > 0 && head_dim % 8 == 0 && head_dim <= 512,
              "INT8 PTH writer requires head_dim divisible by 8 and <= 512");
  TORCH_CHECK(key.stride(2) == 1 && value.stride(2) == 1 &&
                  key_cache.stride(3) == 1 && value_cache.stride(3) == 1,
              "INT8 PTH writer requires contiguous head dimension");
  TORCH_CHECK(slot_mapping.dim() == 1 && slot_mapping.size(0) >= num_tokens &&
                  slot_mapping.is_contiguous(),
              "INT8 PTH writer requires contiguous 1D slot_mapping");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(key_cache.data_ptr()) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(value_cache.data_ptr()) % 8 == 0,
              "INT8 PTH cache pointers must be 8-byte aligned");
  for (int axis = 0; axis < 3; ++axis) {
    TORCH_CHECK(
        key_cache.stride(axis) % 8 == 0 && value_cache.stride(axis) % 8 == 0,
        "INT8 PTH cache strides must be multiples of 8");
  }

  c10::cuda::CUDAGuard device_guard(key.device());
  const int64_t total_items = num_tokens * num_heads;
  const dim3 grid(static_cast<unsigned>((total_items + 3) / 4));
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  if (slot_mapping.scalar_type() == at::kLong) {
    sm70_reshape_and_cache_int8_pth_kernel<int64_t><<<grid, 128, 0, stream>>>(
        reinterpret_cast<const __half*>(key.data_ptr()),
        reinterpret_cast<const __half*>(value.data_ptr()),
        key_cache.data_ptr<int8_t>(), value_cache.data_ptr<int8_t>(),
        k_scales.data_ptr<float>(), v_scales.data_ptr<float>(),
        slot_mapping.data_ptr<int64_t>(), total_items, num_heads, head_dim,
        static_cast<int>(key_cache.size(1)),
        static_cast<int>(key_cache.size(0)), key.stride(0), key.stride(1),
        value.stride(0), value.stride(1), key_cache.stride(0),
        key_cache.stride(1), key_cache.stride(2), value_cache.stride(0),
        value_cache.stride(1), value_cache.stride(2), k_scales.stride(0),
        k_scales.stride(1), k_scales.stride(2), v_scales.stride(0),
        v_scales.stride(1), v_scales.stride(2));
  } else {
    TORCH_CHECK(slot_mapping.scalar_type() == at::kInt,
                "slot_mapping must be int64 or int32");
    sm70_reshape_and_cache_int8_pth_kernel<int32_t><<<grid, 128, 0, stream>>>(
        reinterpret_cast<const __half*>(key.data_ptr()),
        reinterpret_cast<const __half*>(value.data_ptr()),
        key_cache.data_ptr<int8_t>(), value_cache.data_ptr<int8_t>(),
        k_scales.data_ptr<float>(), v_scales.data_ptr<float>(),
        slot_mapping.data_ptr<int32_t>(), total_items, num_heads, head_dim,
        static_cast<int>(key_cache.size(1)),
        static_cast<int>(key_cache.size(0)), key.stride(0), key.stride(1),
        value.stride(0), value.stride(1), key_cache.stride(0),
        key_cache.stride(1), key_cache.stride(2), value_cache.stride(0),
        value_cache.stride(1), value_cache.stride(2), k_scales.stride(0),
        k_scales.stride(1), k_scales.stride(2), v_scales.stride(0),
        v_scales.stride(1), v_scales.stride(2));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sm70_v37_int8_pth_paged_kv_to_fp16(
    const at::Tensor& key_cache, const at::Tensor& value_cache,
    const at::Tensor& k_scales, const at::Tensor& v_scales,
    const at::Tensor& block_table, const at::Tensor& seq_lens,
    at::Tensor& key_out, at::Tensor& value_out) {
  TORCH_CHECK(key_cache.is_cuda() && value_cache.is_cuda() &&
                  k_scales.is_cuda() && v_scales.is_cuda() &&
                  key_out.is_cuda() && value_out.is_cuda() &&
                  block_table.is_cuda() && seq_lens.is_cuda(),
              "INT8 PTH KV bridge requires CUDA tensors");
  TORCH_CHECK(key_cache.scalar_type() == at::kChar &&
                  value_cache.scalar_type() == at::kChar,
              "INT8 PTH input caches must be int8");
  TORCH_CHECK(k_scales.scalar_type() == at::kFloat &&
                  v_scales.scalar_type() == at::kFloat,
              "INT8 PTH scale caches must be float32");
  TORCH_CHECK(key_out.scalar_type() == at::kHalf &&
                  value_out.scalar_type() == at::kHalf,
              "INT8 PTH bridge output caches must be fp16");
  TORCH_CHECK(block_table.scalar_type() == at::kInt &&
                  seq_lens.scalar_type() == at::kInt &&
                  block_table.is_contiguous() && seq_lens.is_contiguous(),
              "INT8 PTH bridge metadata must be contiguous int32");
  TORCH_CHECK(key_cache.dim() == 4 &&
                  value_cache.sizes() == key_cache.sizes() &&
                  key_out.dim() == 4 && value_out.sizes() == key_out.sizes() &&
                  k_scales.dim() == 3 && v_scales.sizes() == k_scales.sizes(),
              "INT8 PTH bridge shape mismatch");
  TORCH_CHECK(key_cache.size(2) == key_out.size(2) &&
                  key_cache.size(3) == key_out.size(3) &&
                  key_cache.size(3) % 8 == 0,
              "INT8 PTH bridge requires matching head shape with D % 8 == 0");
  const int batch_size = static_cast<int>(block_table.size(0));
  TORCH_CHECK(batch_size > 0 && key_out.size(0) % batch_size == 0,
              "INT8 PTH bridge invalid batch geometry");
  const int output_blocks_per_seq =
      static_cast<int>(key_out.size(0)) / batch_size;
  const int input_capacity =
      static_cast<int>(block_table.size(1) * key_cache.size(1));
  const int output_capacity =
      output_blocks_per_seq * static_cast<int>(key_out.size(1));
  TORCH_CHECK(output_capacity >= input_capacity,
              "FP16 output cache capacity must cover the input block table");
  for (const at::Tensor* tensor : std::initializer_list<const at::Tensor*>{
           &key_cache, &value_cache, &key_out, &value_out}) {
    TORCH_CHECK(tensor->stride(3) == 1 &&
                    reinterpret_cast<uintptr_t>(tensor->data_ptr()) % 16 == 0,
                "INT8 PTH bridge requires 16-byte aligned pointers");
    for (int axis = 0; axis < 3; ++axis) {
      TORCH_CHECK(tensor->stride(axis) % 8 == 0,
                  "INT8 PTH bridge requires 8-element aligned strides");
    }
  }

  c10::cuda::CUDAGuard device_guard(key_cache.device());
  const int64_t total_vecs = static_cast<int64_t>(output_capacity) *
                             key_cache.size(2) * (key_cache.size(3) / 8);
  const dim3 grid(
      static_cast<unsigned int>((total_vecs + kThreads - 1) / kThreads),
      batch_size);
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  int8_pth_paged_kv_to_fp16_kernel<<<grid, kThreads, 0, stream>>>(
      key_cache.data_ptr<int8_t>(), value_cache.data_ptr<int8_t>(),
      k_scales.data_ptr<float>(), v_scales.data_ptr<float>(),
      block_table.data_ptr<int>(), seq_lens.data_ptr<int>(),
      reinterpret_cast<__half*>(key_out.data_ptr()),
      reinterpret_cast<__half*>(value_out.data_ptr()), batch_size,
      static_cast<int>(block_table.size(1)),
      static_cast<int>(key_cache.size(1)), output_blocks_per_seq,
      static_cast<int>(key_out.size(1)), static_cast<int>(key_cache.size(2)),
      static_cast<int>(key_cache.size(3)), key_cache.stride(0),
      key_cache.stride(1), key_cache.stride(2), value_cache.stride(0),
      value_cache.stride(1), value_cache.stride(2), k_scales.stride(0),
      k_scales.stride(1), k_scales.stride(2), v_scales.stride(0),
      v_scales.stride(1), v_scales.stride(2), key_out.stride(0),
      key_out.stride(1), key_out.stride(2), value_out.stride(0),
      value_out.stride(1), value_out.stride(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

TORCH_LIBRARY_FRAGMENT(_vllm_fa2_C, m) {
  m.def(
      "sm70_v37_e4m3_bridge(Tensor key, Tensor value, Tensor table, Tensor "
      "lengths, Tensor(a!) key_out, Tensor(b!) value_out, float key_scale, "
      "float value_scale) -> ()");
  m.def(
      "sm70_v37_int8_pth_bridge(Tensor key, Tensor value, Tensor k_scales, "
      "Tensor v_scales, Tensor table, Tensor lengths, Tensor(a!) key_out, "
      "Tensor(b!) value_out) -> ()");
  m.def(
      "sm70_reshape_and_cache_int8_pth(Tensor key, Tensor value, "
      "Tensor(a!) key_cache, Tensor(b!) value_cache, Tensor(c!) k_scales, "
      "Tensor(d!) v_scales, Tensor slot_mapping) -> ()");
}
TORCH_LIBRARY_IMPL(_vllm_fa2_C, CUDA, m) {
  m.impl("sm70_v37_e4m3_bridge", &sm70_v37_e4m3_paged_kv_to_fp16);
  m.impl("sm70_v37_int8_pth_bridge", &sm70_v37_int8_pth_paged_kv_to_fp16);
  m.impl("sm70_reshape_and_cache_int8_pth", &sm70_reshape_and_cache_int8_pth);
}
