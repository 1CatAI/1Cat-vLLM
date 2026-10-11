#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include "kv_cache_view.h"

namespace {

constexpr int kThreads = 256;

template <int KV_DTYPE>
__device__ __forceinline__ __half2 load_bridge_half2(const void* cache,
                                                     int64_t offset) {
  if constexpr (KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E5M2) {
    return flash_v100::load_fp8_e5m2_half2_unscaled(cache, offset);
  } else if constexpr (KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP16) {
    return *reinterpret_cast<const __half2*>(static_cast<const __half*>(cache) +
                                             offset);
  } else if constexpr (KV_DTYPE == flash_v100::KV_CACHE_DTYPE_INT8) {
    const auto* bytes = static_cast<const int8_t*>(cache) + offset;
    return __floats2half2_rn(float(bytes[0]), float(bytes[1]));
  } else {
    const uint16_t pair = reinterpret_cast<const uint16_t*>(cache)[offset >> 1];
    return __float22half2_rn(make_float2(
        flash_v100::fp8_e4m3fn_to_float(static_cast<uint8_t>(pair)),
        flash_v100::fp8_e4m3fn_to_float(static_cast<uint8_t>(pair >> 8))));
  }
}

template <int KV_DTYPE, bool UNIT_SCALE, int PAIRS = 1>
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
    float key_scale, float value_scale, flash_v100::KVScaleView scales) {
  const int batch_idx = blockIdx.y;
  if (batch_idx >= batch_size) {
    return;
  }

  const int pairs_per_head = head_dim / (2 * PAIRS);
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
  const int element_idx = (pair_in_token % pairs_per_head) * 2 * PAIRS;

  const __half2 zero = __float2half2_rn(0.0f);
  __half2 key_pair[PAIRS], value_pair[PAIRS];
#pragma unroll
  for (int p = 0; p < PAIRS; ++p) key_pair[p] = value_pair[p] = zero;
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

    if constexpr (flash_v100::KVCodecTraits<KV_DTYPE>::per_token_head) {
      key_scale = scales.key.load(physical_block, input_block_offset, head_idx);
      value_scale =
          scales.value.load(physical_block, input_block_offset, head_idx);
    }
    if constexpr (PAIRS == 4) {
      union Packed {
        uint4 u;
        __half2 h[4];
      } k, v;
      k.u = flash_v100::KVCodecTraits<KV_DTYPE>::load_half8(
          key_cache, key_input_offset, 0);
      v.u = flash_v100::KVCodecTraits<KV_DTYPE>::load_half8(
          value_cache, value_input_offset, 0);
#pragma unroll
      for (int p = 0; p < PAIRS; ++p) {
        key_pair[p] = k.h[p];
        value_pair[p] = v.h[p];
      }
    } else {
      key_pair[0] = load_bridge_half2<KV_DTYPE>(key_cache, key_input_offset);
      value_pair[0] =
          load_bridge_half2<KV_DTYPE>(value_cache, value_input_offset);
    }
    if constexpr (!UNIT_SCALE) {
#pragma unroll
      for (int p = 0; p < PAIRS; ++p) {
        float2 k = __half22float2(key_pair[p]),
               v = __half22float2(value_pair[p]);
        k.x *= key_scale;
        k.y *= key_scale;
        v.x *= value_scale;
        v.y *= value_scale;
        // FP32 row scales must not overflow a finite FP16 destination. NaN
        // scale markers remain NaN; do not use fminf/fmaxf on invalid rows.
        if constexpr (flash_v100::KVCodecTraits<KV_DTYPE>::per_token_head) {
          if (isfinite(key_scale)) {
            k.x = fminf(65504.f, fmaxf(-65504.f, k.x));
            k.y = fminf(65504.f, fmaxf(-65504.f, k.y));
          }
          if (isfinite(value_scale)) {
            v.x = fminf(65504.f, fmaxf(-65504.f, v.x));
            v.y = fminf(65504.f, fmaxf(-65504.f, v.y));
          }
        }
        key_pair[p] = __float22half2_rn(k);
        value_pair[p] = __float22half2_rn(v);
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
  if constexpr (PAIRS == 4) {
    union Packed {
      uint4 u;
      __half2 h[4];
    } k, v;
#pragma unroll
    for (int p = 0; p < PAIRS; ++p) {
      k.h[p] = key_pair[p];
      v.h[p] = value_pair[p];
    }
    *reinterpret_cast<uint4*>(key_out + key_output_offset) = k.u;
    *reinterpret_cast<uint4*>(value_out + value_output_offset) = v.u;
  } else {
    *reinterpret_cast<__half2*>(key_out + key_output_offset) = key_pair[0];
    *reinterpret_cast<__half2*>(value_out + value_output_offset) =
        value_pair[0];
  }
}

}  // namespace

template <int KV_DTYPE, int PAIRS = 1>
void flash_attention_fp8_paged_kv_to_fp16(
    const at::Tensor& key_cache, const at::Tensor& value_cache,
    const at::Tensor& block_table, const at::Tensor& seq_lens,
    at::Tensor& key_out, at::Tensor& value_out, const float key_scale,
    const float value_scale, flash_v100::KVScaleView scales = {}) {
  TORCH_CHECK(key_cache.is_cuda() && value_cache.is_cuda() &&
                  key_out.is_cuda() && value_out.is_cuda(),
              "FP8 KV bridge tensors must be CUDA tensors");
  TORCH_CHECK(block_table.is_cuda() && seq_lens.is_cuda(),
              "FP8 KV bridge metadata must be CUDA tensors");
  TORCH_CHECK(
      key_cache.scalar_type() == flash_v100::kv_storage_type<KV_DTYPE>() &&
          value_cache.scalar_type() == flash_v100::kv_storage_type<KV_DTYPE>(),
      "KV input dtype does not match its codec");
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
  TORCH_CHECK(key_cache.size(3) % (2 * PAIRS) == 0,
              "FP8 KV bridge requires an even head dimension");
  TORCH_CHECK(block_table.dim() == 2 && seq_lens.dim() == 1 &&
                  block_table.size(0) == seq_lens.size(0),
              "FP8 KV bridge metadata shape mismatch");
  TORCH_CHECK(std::isfinite(key_scale) && std::isfinite(value_scale) &&
                  key_scale > 0.f && value_scale > 0.f,
              "FP8 KV bridge scales must be finite and positive");
  TORCH_CHECK(block_table.is_contiguous() && seq_lens.is_contiguous(),
              "FP8 KV bridge metadata must be contiguous");
  for (const auto* tensor : {&value_cache, &block_table, &seq_lens,
                             static_cast<const at::Tensor*>(&key_out),
                             static_cast<const at::Tensor*>(&value_out)}) {
    TORCH_CHECK(tensor->device() == key_cache.device(),
                "FP8 KV bridge tensors must share a device");
  }
  for (const auto* tensor :
       {&key_cache, &value_cache, static_cast<const at::Tensor*>(&key_out),
        static_cast<const at::Tensor*>(&value_out)}) {
    TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor->data_ptr()) %
                        (2 * tensor->element_size()) ==
                    0,
                "FP8 KV bridge requires pair-aligned storage");
    for (int i = 0; i < 3; ++i) {
      TORCH_CHECK(tensor->stride(i) % 2 == 0,
                  "FP8 KV bridge requires pair-aligned strides");
    }
  }
  TORCH_CHECK(key_cache.size(1) > 0 && key_cache.size(2) > 0 &&
                  key_cache.size(3) > 0 && key_out.size(1) > 0 &&
                  block_table.size(1) > 0,
              "FP8 KV bridge dimensions must be non-empty");

  if constexpr (PAIRS == 4) {
    for (const auto* tensor : {static_cast<const at::Tensor*>(&key_out),
                               static_cast<const at::Tensor*>(&value_out)}) {
      TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor->data_ptr()) % 16 == 0,
                  "Wide bridge output requires 16-byte alignment");
      for (int axis = 0; axis < 3; ++axis)
        TORCH_CHECK(tensor->stride(axis) % 8 == 0, "Wide bridge output stride");
    }
    for (const auto* tensor : {&key_cache, &value_cache}) {
      constexpr int alignment =
          flash_v100::KVStorageTraits<KV_DTYPE>::vector_alignment;
      TORCH_CHECK(
          reinterpret_cast<uintptr_t>(tensor->data_ptr()) % alignment == 0,
          "Wide bridge input alignment");
      for (int axis = 0; axis < 3; ++axis)
        TORCH_CHECK(
            (tensor->stride(axis) * tensor->element_size()) % alignment == 0,
            "Wide bridge input stride");
    }
  }
  const int batch_size = block_table.size(0);
  TORCH_CHECK(batch_size > 0, "FP8 KV bridge batch must be non-empty");
  TORCH_CHECK(key_out.size(0) % batch_size == 0,
              "FP16 output blocks must divide evenly across the batch");
  const int output_blocks_per_seq = key_out.size(0) / batch_size;
  const int input_capacity = block_table.size(1) * key_cache.size(1);
  const int output_capacity = output_blocks_per_seq * key_out.size(1);
  TORCH_CHECK(output_capacity >= input_capacity,
              "FP16 output cache capacity must cover the input block table");

  c10::cuda::CUDAGuard device_guard(key_cache.device());
  const int64_t total_pairs = static_cast<int64_t>(output_capacity) *
                              key_cache.size(2) *
                              (key_cache.size(3) / (2 * PAIRS));
  const dim3 grid(
      static_cast<unsigned int>((total_pairs + kThreads - 1) / kThreads),
      batch_size);
  const auto stream = at::cuda::getCurrentCUDAStream().stream();

#define LAUNCH_FP8_BRIDGE(UNIT_SCALE)                                          \
  fp8_paged_kv_to_fp16_kernel<KV_DTYPE, UNIT_SCALE, PAIRS>                     \
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
          key_scale, value_scale, scales)

  if (!flash_v100::KVCodecTraits<KV_DTYPE>::per_token_head &&
      key_scale == 1.f && value_scale == 1.f) {
    LAUNCH_FP8_BRIDGE(true);
  } else {
    LAUNCH_FP8_BRIDGE(false);
  }
#undef LAUNCH_FP8_BRIDGE

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void flash_attention_fp8_e5m2_paged_kv_to_fp16(
    const at::Tensor& key_cache, const at::Tensor& value_cache,
    const at::Tensor& block_table, const at::Tensor& seq_lens,
    at::Tensor& key_out, at::Tensor& value_out, const float key_scale,
    const float value_scale) {
  flash_attention_fp8_paged_kv_to_fp16<flash_v100::KV_CACHE_DTYPE_FP8_E5M2>(
      key_cache, value_cache, block_table, seq_lens, key_out, value_out,
      key_scale, value_scale);
}

void flash_attention_fp8_e4m3_paged_kv_to_fp16(
    const at::Tensor& key_cache, const at::Tensor& value_cache,
    const at::Tensor& block_table, const at::Tensor& seq_lens,
    at::Tensor& key_out, at::Tensor& value_out, const float key_scale,
    const float value_scale) {
  flash_attention_fp8_paged_kv_to_fp16<flash_v100::KV_CACHE_DTYPE_FP8_E4M3>(
      key_cache, value_cache, block_table, seq_lens, key_out, value_out,
      key_scale, value_scale);
}

// All compatible codecs share page traversal, tail clearing and output layout.
void flash_attention_paged_kv_to_fp16(
    const at::Tensor& key_cache, const at::Tensor& value_cache,
    const at::Tensor& block_table, const at::Tensor& seq_lens,
    at::Tensor& key_out, at::Tensor& value_out, int64_t codec, float key_scale,
    float value_scale, const std::optional<at::Tensor>& key_scale_cache,
    const std::optional<at::Tensor>& value_scale_cache) {
  const auto scales = flash_v100::bind_scale_view(
      codec, key_cache, value_cache, key_scale_cache, value_scale_cache);
  flash_v100::visit_kv_codec(codec, [&](auto tag) {
    flash_attention_fp8_paged_kv_to_fp16<decltype(tag)::value, 4>(
        key_cache, value_cache, block_table, seq_lens, key_out, value_out,
        key_scale, value_scale, scales);
  });
}
