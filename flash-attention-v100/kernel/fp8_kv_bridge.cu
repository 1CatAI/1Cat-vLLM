#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include "fp8_kv_utils.cuh"

namespace {

constexpr int kThreads = 256;

template <bool UNIT_SCALE>
__global__ void fp8_e5m2_paged_kv_to_fp16_kernel(
    const void* __restrict__ key_cache,
    const void* __restrict__ value_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ seq_lens,
    __half* __restrict__ key_out,
    __half* __restrict__ value_out,
    int batch_size,
    int max_num_blocks,
    int input_block_size,
    int output_blocks_per_seq,
    int output_block_size,
    int num_heads,
    int head_dim,
    int64_t key_block_stride,
    int64_t key_token_stride,
    int64_t key_head_stride,
    int64_t value_block_stride,
    int64_t value_token_stride,
    int64_t value_head_stride,
    int64_t key_out_block_stride,
    int64_t key_out_token_stride,
    int64_t key_out_head_stride,
    int64_t value_out_block_stride,
    int64_t value_out_token_stride,
    int64_t value_out_head_stride,
    float key_scale,
    float value_scale) {
  const int batch_idx = blockIdx.y;
  if (batch_idx >= batch_size) {
    return;
  }

  const int pairs_per_head = head_dim / 2;
  const int64_t pairs_per_token =
      static_cast<int64_t>(num_heads) * pairs_per_head;
  const int max_tokens = output_blocks_per_seq * output_block_size;
  const int64_t total_pairs = static_cast<int64_t>(max_tokens) * pairs_per_token;
  const int64_t pair_idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair_idx >= total_pairs) {
    return;
  }

  const int token_idx = static_cast<int>(pair_idx / pairs_per_token);
  if (token_idx >= seq_lens[batch_idx]) {
    return;
  }
  const int pair_in_token = static_cast<int>(pair_idx % pairs_per_token);
  const int head_idx = pair_in_token / pairs_per_head;
  const int element_idx = (pair_in_token % pairs_per_head) * 2;

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

  __half2 key_pair = flash_v100::load_fp8_e5m2_half2_unscaled(
      key_cache, key_input_offset);
  __half2 value_pair = flash_v100::load_fp8_e5m2_half2_unscaled(
      value_cache, value_input_offset);
  if constexpr (!UNIT_SCALE) {
    const float2 key_values = __half22float2(key_pair);
    const float2 value_values = __half22float2(value_pair);
    key_pair = __float22half2_rn(
        make_float2(key_values.x * key_scale, key_values.y * key_scale));
    value_pair = __float22half2_rn(make_float2(
        value_values.x * value_scale, value_values.y * value_scale));
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
  *reinterpret_cast<__half2*>(key_out + key_output_offset) = key_pair;
  *reinterpret_cast<__half2*>(value_out + value_output_offset) = value_pair;
}

}  // namespace

void flash_attention_fp8_e5m2_paged_kv_to_fp16(
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const at::Tensor& block_table,
    const at::Tensor& seq_lens,
    at::Tensor& key_out,
    at::Tensor& value_out,
    const float key_scale,
    const float value_scale) {
  TORCH_CHECK(
      key_cache.is_cuda() && value_cache.is_cuda() && key_out.is_cuda() &&
          value_out.is_cuda(),
      "FP8 KV bridge tensors must be CUDA tensors");
  TORCH_CHECK(
      block_table.is_cuda() && seq_lens.is_cuda(),
      "FP8 KV bridge metadata must be CUDA tensors");
  TORCH_CHECK(
      key_cache.scalar_type() == at::kByte &&
          value_cache.scalar_type() == at::kByte,
      "FP8 E5M2 input caches must be stored as uint8");
  TORCH_CHECK(
      key_out.scalar_type() == at::kHalf &&
          value_out.scalar_type() == at::kHalf,
      "FP8 KV bridge output caches must be fp16");
  TORCH_CHECK(
      block_table.scalar_type() == at::kInt &&
          seq_lens.scalar_type() == at::kInt,
      "FP8 KV bridge block_table and seq_lens must be int32");
  TORCH_CHECK(
      key_cache.dim() == 4 && value_cache.dim() == 4 && key_out.dim() == 4 &&
          value_out.dim() == 4,
      "FP8 KV bridge expects paged [blocks,tokens,heads,dim] caches");
  TORCH_CHECK(
      key_cache.sizes() == value_cache.sizes(),
      "FP8 K/V input cache shapes must match");
  TORCH_CHECK(
      key_out.sizes() == value_out.sizes(),
      "FP16 K/V output cache shapes must match");
  TORCH_CHECK(
      key_cache.size(2) == key_out.size(2) &&
          key_cache.size(3) == key_out.size(3),
      "FP8 KV bridge head shape must not change");
  TORCH_CHECK(
      key_cache.stride(3) == 1 && value_cache.stride(3) == 1 &&
          key_out.stride(3) == 1 && value_out.stride(3) == 1,
      "FP8 KV bridge requires contiguous head dimensions");
  TORCH_CHECK(
      key_cache.size(3) % 2 == 0,
      "FP8 KV bridge requires an even head dimension");
  TORCH_CHECK(
      block_table.dim() == 2 && seq_lens.dim() == 1 &&
          block_table.size(0) == seq_lens.size(0),
      "FP8 KV bridge metadata shape mismatch");
  TORCH_CHECK(
      key_scale > 0.f && value_scale > 0.f,
      "FP8 KV bridge scales must be positive");

  const int batch_size = block_table.size(0);
  TORCH_CHECK(batch_size > 0, "FP8 KV bridge batch must be non-empty");
  TORCH_CHECK(
      key_out.size(0) % batch_size == 0,
      "FP16 output blocks must divide evenly across the batch");
  const int output_blocks_per_seq = key_out.size(0) / batch_size;
  const int input_capacity = block_table.size(1) * key_cache.size(1);
  const int output_capacity = output_blocks_per_seq * key_out.size(1);
  TORCH_CHECK(
      output_capacity >= input_capacity,
      "FP16 output cache capacity must cover the input block table");

  c10::cuda::CUDAGuard device_guard(key_cache.device());
  const int64_t total_pairs =
      static_cast<int64_t>(output_capacity) * key_cache.size(2) *
      (key_cache.size(3) / 2);
  const dim3 grid(
      static_cast<unsigned int>((total_pairs + kThreads - 1) / kThreads),
      batch_size);
  const auto stream = at::cuda::getCurrentCUDAStream().stream();

#define LAUNCH_FP8_BRIDGE(UNIT_SCALE)                                           \
  fp8_e5m2_paged_kv_to_fp16_kernel<UNIT_SCALE><<<grid, kThreads, 0, stream>>>(  \
      key_cache.data_ptr(), value_cache.data_ptr(), block_table.data_ptr<int>(), \
      seq_lens.data_ptr<int>(), reinterpret_cast<__half*>(key_out.data_ptr()),   \
      reinterpret_cast<__half*>(value_out.data_ptr()), batch_size,               \
      block_table.size(1), key_cache.size(1), output_blocks_per_seq,             \
      key_out.size(1), key_cache.size(2), key_cache.size(3),                     \
      key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),             \
      value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),       \
      key_out.stride(0), key_out.stride(1), key_out.stride(2),                   \
      value_out.stride(0), value_out.stride(1), value_out.stride(2), key_scale,  \
      value_scale)

  if (key_scale == 1.f && value_scale == 1.f) {
    LAUNCH_FP8_BRIDGE(true);
  } else {
    LAUNCH_FP8_BRIDGE(false);
  }
#undef LAUNCH_FP8_BRIDGE

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
