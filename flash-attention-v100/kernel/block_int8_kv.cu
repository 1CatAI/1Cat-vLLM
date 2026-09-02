#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace {

constexpr int kChannelBlockSize = 32;
constexpr int kThreads = 256;

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
  }
  return value;
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__device__ __forceinline__ int8_t quantize_symmetric_int8(float value,
                                                            float scale) {
  if (scale == 0.0f) {
    return 0;
  }
  const float code = fminf(127.0f, fmaxf(-127.0f, nearbyintf(value / scale)));
  return static_cast<int8_t>(code);
}

__global__ void reset_int8_block32_page_owners_kernel(
    int* __restrict__ page_owners, const int num_blocks,
    const int64_t owner_stride) {
  const int page = blockIdx.x * blockDim.x + threadIdx.x;
  if (page < num_blocks) {
    page_owners[static_cast<int64_t>(page) * owner_stride] = 0x7fffffff;
  }
}

__global__ void mark_int8_block32_page_owners_kernel(
    const int64_t* __restrict__ slot_mapping, int* __restrict__ page_owners,
    const int num_tokens, const int num_blocks, const int block_size,
    const int64_t owner_stride) {
  const int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (token_idx >= num_tokens) {
    return;
  }
  const int64_t slot = slot_mapping[token_idx];
  if (slot < 0) {
    return;
  }
  const int64_t physical_block = slot / block_size;
  if (physical_block < num_blocks) {
    atomicMin(page_owners + physical_block * owner_stride, token_idx);
  }
}

__global__ void int8_block32_reshape_and_cache_kernel(
    const __half* __restrict__ key, const __half* __restrict__ value,
    int8_t* __restrict__ key_cache, int8_t* __restrict__ value_cache,
    __half* __restrict__ key_scales, __half* __restrict__ value_scales,
    const int* __restrict__ page_owners,
    const int64_t* __restrict__ slot_mapping, const int num_tokens,
    const int num_blocks, const int block_size, const int num_heads,
    const int head_dim, const int channel_blocks,
    const int64_t owner_stride, const int64_t key_stride0,
    const int64_t key_stride1, const int64_t value_stride0,
    const int64_t value_stride1, const int64_t key_block_stride,
    const int64_t key_token_stride, const int64_t key_head_stride,
    const int64_t value_block_stride, const int64_t value_token_stride,
    const int64_t value_head_stride, const int64_t scale_block_stride,
    const int64_t scale_head_stride) {
  const int head_idx = blockIdx.x;
  const int channel_block = blockIdx.y;
  const int owner_token = blockIdx.z;
  const int lane = threadIdx.x;
  if (head_idx >= num_heads || channel_block >= channel_blocks || lane >= 32 ||
      owner_token >= num_tokens) {
    return;
  }

  const int64_t owner_slot = slot_mapping[owner_token];
  if (owner_slot < 0) {
    return;
  }
  const int physical_block = static_cast<int>(owner_slot / block_size);
  if (physical_block < 0 || physical_block >= num_blocks ||
      page_owners[static_cast<int64_t>(physical_block) * owner_stride] !=
          owner_token) {
    return;
  }

  const int d = channel_block * kChannelBlockSize + lane;
  const bool valid = d < head_dim;
  const int64_t page_slot_base =
      static_cast<int64_t>(physical_block) * block_size;
  const int64_t scale_index =
      static_cast<int64_t>(physical_block) * scale_block_stride +
      static_cast<int64_t>(head_idx) * scale_head_stride + channel_block;

  bool reset_page = false;
  for (int token_idx = 0; token_idx < num_tokens; ++token_idx) {
    reset_page |= slot_mapping[token_idx] == page_slot_base;
  }
  if (reset_page) {
    for (int token = 0; token < block_size; ++token) {
      if (valid) {
        key_cache[static_cast<int64_t>(physical_block) * key_block_stride +
                  static_cast<int64_t>(token) * key_token_stride +
                  static_cast<int64_t>(head_idx) * key_head_stride + d] = 0;
        value_cache[static_cast<int64_t>(physical_block) * value_block_stride +
                    static_cast<int64_t>(token) * value_token_stride +
                    static_cast<int64_t>(head_idx) * value_head_stride + d] = 0;
      }
    }
    if (lane == 0) {
      key_scales[scale_index] = __float2half(0.0f);
      value_scales[scale_index] = __float2half(0.0f);
    }
  }
  __syncwarp();

  // Exactly one CTA owns each physical page, so noncontiguous or reordered
  // slot mappings cannot race page-local scale growth. First determine the
  // batch-final scale, then re-quantize historical values at most once.
  const float old_key_scale = __half2float(key_scales[scale_index]);
  const float old_value_scale = __half2float(value_scales[scale_index]);
  float batch_key_max = 0.0f;
  float batch_value_max = 0.0f;
  for (int token_idx = 0; token_idx < num_tokens; ++token_idx) {
    const int64_t slot = slot_mapping[token_idx];
    if (slot < page_slot_base || slot >= page_slot_base + block_size) {
      continue;
    }
    const float key_input =
        valid ? __half2float(key[static_cast<int64_t>(token_idx) * key_stride0 +
                                 static_cast<int64_t>(head_idx) * key_stride1 + d])
              : 0.0f;
    const float value_input =
        valid ? __half2float(
                    value[static_cast<int64_t>(token_idx) * value_stride0 +
                          static_cast<int64_t>(head_idx) * value_stride1 + d])
              : 0.0f;
    batch_key_max = fmaxf(
        batch_key_max,
        __shfl_sync(0xffffffff, warp_max(fabsf(key_input)), 0));
    batch_value_max = fmaxf(
        batch_value_max,
        __shfl_sync(0xffffffff, warp_max(fabsf(value_input)), 0));
  }

  const float next_key_scale = __half2float(
      __float2half_ru(fmaxf(old_key_scale, batch_key_max / 127.0f)));
  const float next_value_scale = __half2float(
      __float2half_ru(fmaxf(old_value_scale, batch_value_max / 127.0f)));
  if (next_key_scale > old_key_scale && old_key_scale > 0.0f && valid) {
    for (int token = 0; token < block_size; ++token) {
      const int64_t index =
          static_cast<int64_t>(physical_block) * key_block_stride +
          static_cast<int64_t>(token) * key_token_stride +
          static_cast<int64_t>(head_idx) * key_head_stride + d;
      key_cache[index] = quantize_symmetric_int8(
          static_cast<float>(key_cache[index]) * old_key_scale,
          next_key_scale);
    }
  }
  if (next_value_scale > old_value_scale && old_value_scale > 0.0f && valid) {
    for (int token = 0; token < block_size; ++token) {
      const int64_t index =
          static_cast<int64_t>(physical_block) * value_block_stride +
          static_cast<int64_t>(token) * value_token_stride +
          static_cast<int64_t>(head_idx) * value_head_stride + d;
      value_cache[index] = quantize_symmetric_int8(
          static_cast<float>(value_cache[index]) * old_value_scale,
          next_value_scale);
    }
  }
  if (lane == 0) {
    key_scales[scale_index] = __float2half(next_key_scale);
    value_scales[scale_index] = __float2half(next_value_scale);
  }
  __syncwarp();

  for (int token_idx = 0; token_idx < num_tokens; ++token_idx) {
    const int64_t slot = slot_mapping[token_idx];
    if (slot < page_slot_base || slot >= page_slot_base + block_size) {
      continue;
    }
    const int block_offset = static_cast<int>(slot - page_slot_base);
    if (valid) {
      const float key_input =
          __half2float(key[static_cast<int64_t>(token_idx) * key_stride0 +
                           static_cast<int64_t>(head_idx) * key_stride1 + d]);
      const float value_input =
          __half2float(value[static_cast<int64_t>(token_idx) * value_stride0 +
                             static_cast<int64_t>(head_idx) * value_stride1 + d]);
      const int64_t key_index =
          static_cast<int64_t>(physical_block) * key_block_stride +
          static_cast<int64_t>(block_offset) * key_token_stride +
          static_cast<int64_t>(head_idx) * key_head_stride + d;
      const int64_t value_index =
          static_cast<int64_t>(physical_block) * value_block_stride +
          static_cast<int64_t>(block_offset) * value_token_stride +
          static_cast<int64_t>(head_idx) * value_head_stride + d;
      key_cache[key_index] = quantize_symmetric_int8(key_input, next_key_scale);
      value_cache[value_index] =
          quantize_symmetric_int8(value_input, next_value_scale);
    }
  }
}

__global__ void int8_block32_decode_kernel(
    const __half* __restrict__ query, const int8_t* __restrict__ key_cache,
    const int8_t* __restrict__ value_cache, const __half* __restrict__ key_scales,
    const __half* __restrict__ value_scales,
    const int* __restrict__ query_start_loc,
    const int* __restrict__ block_table, const int* __restrict__ seq_lens,
    __half* __restrict__ output, const int num_queries, const int num_requests,
    const int max_num_blocks, const int num_heads_q, const int num_heads_kv,
    const int block_size, const int head_dim,
    const int channel_blocks, const int64_t query_stride0,
    const int64_t query_stride1, const int64_t key_block_stride,
    const int64_t key_token_stride, const int64_t key_head_stride,
    const int64_t value_block_stride, const int64_t value_token_stride,
    const int64_t value_head_stride, const int64_t scale_block_stride,
    const int64_t scale_head_stride, const int64_t output_stride0,
    const int64_t output_stride1, const float softmax_scale) {
  const int query_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int tid = threadIdx.x;
  if (query_idx >= num_queries || head_idx >= num_heads_q) {
    return;
  }

  int request_idx;
  int seq_len;
  if (query_start_loc != nullptr) {
    int low = 0;
    int high = num_requests;
    while (low + 1 < high) {
      const int middle = (low + high) / 2;
      if (query_idx >= query_start_loc[middle]) {
        low = middle;
      } else {
        high = middle;
      }
    }
    request_idx = low;
    const int query_start = query_start_loc[request_idx];
    const int query_len = query_start_loc[request_idx + 1] - query_start;
    seq_len = seq_lens[request_idx] - query_len + query_idx - query_start + 1;
  } else {
    request_idx = query_idx;
    seq_len = seq_lens[request_idx];
  }

  const int q_per_kv = num_heads_q / num_heads_kv;
  const int kv_head_idx = head_idx / q_per_kv;
  const __half* q = query + static_cast<int64_t>(query_idx) * query_stride0 +
                    static_cast<int64_t>(head_idx) * query_stride1;
  __half* out = output + static_cast<int64_t>(query_idx) * output_stride0 +
                static_cast<int64_t>(head_idx) * output_stride1;

  constexpr int kWarps = kThreads / 32;
  constexpr int kTileTokens = 256;
  const int lane = tid % 32;
  const int warp_idx = tid / 32;
  const int output_dim = tid;

  __shared__ float scores[kTileTokens];
  __shared__ float warp_stats[kWarps];
  __shared__ float tile_max_shared;
  __shared__ float tile_sum_shared;
  __shared__ float accumulator_alpha;
  __shared__ float accumulator_beta;

  float running_max = -CUDART_INF_F;
  float running_sum = 0.0f;
  float output_accumulator = 0.0f;

  for (int tile_start = 0; tile_start < seq_len;
       tile_start += kTileTokens) {
    const int tile_tokens = min(kTileTokens, seq_len - tile_start);
    float warp_local_max = -CUDART_INF_F;
    for (int token_local = warp_idx; token_local < tile_tokens;
         token_local += kWarps) {
      const int token_idx = tile_start + token_local;
      const int logical_block = token_idx / block_size;
      const int physical_block =
          block_table[request_idx * max_num_blocks + logical_block];
      const int block_offset = token_idx % block_size;
      float dot_part = 0.0f;
      for (int d = lane; d < head_dim; d += 32) {
        const int channel_block = d / kChannelBlockSize;
        const float k_scale = __half2float(
            key_scales[static_cast<int64_t>(physical_block) *
                           scale_block_stride +
                       static_cast<int64_t>(kv_head_idx) * scale_head_stride +
                       channel_block]);
        const int64_t key_index =
            static_cast<int64_t>(physical_block) * key_block_stride +
            static_cast<int64_t>(block_offset) * key_token_stride +
            static_cast<int64_t>(kv_head_idx) * key_head_stride + d;
        const __half key_value = __float2half_rn(
            static_cast<float>(key_cache[key_index]) * k_scale);
        dot_part =
            fmaf(__half2float(q[d]), __half2float(key_value), dot_part);
      }
      const float dot = warp_sum(dot_part);
      if (lane == 0) {
        const float score = dot * softmax_scale;
        scores[token_local] = score;
        warp_local_max = fmaxf(warp_local_max, score);
      }
    }

    const float warp_tile_max = warp_max(warp_local_max);
    if (lane == 0) {
      warp_stats[warp_idx] = warp_tile_max;
    }
    __syncthreads();
    if (warp_idx == 0) {
      const float value = lane < kWarps ? warp_stats[lane] : -CUDART_INF_F;
      const float tile_max = warp_max(value);
      if (lane == 0) {
        tile_max_shared = tile_max;
      }
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (int token_local = tid; token_local < tile_tokens;
         token_local += kThreads) {
      const float probability = expf(scores[token_local] - tile_max_shared);
      scores[token_local] = probability;
      local_sum += probability;
    }
    const float warp_tile_sum = warp_sum(local_sum);
    if (lane == 0) {
      warp_stats[warp_idx] = warp_tile_sum;
    }
    __syncthreads();
    if (warp_idx == 0) {
      const float value = lane < kWarps ? warp_stats[lane] : 0.0f;
      const float tile_sum = warp_sum(value);
      if (lane == 0) {
        tile_sum_shared = tile_sum;
        const float next_max = fmaxf(running_max, tile_max_shared);
        accumulator_alpha = expf(running_max - next_max);
        accumulator_beta = expf(tile_max_shared - next_max);
      }
    }
    __syncthreads();

    if (output_dim < head_dim) {
      float tile_accumulator = 0.0f;
      const int channel_block = output_dim / kChannelBlockSize;
      for (int token_local = 0; token_local < tile_tokens; ++token_local) {
        const int token_idx = tile_start + token_local;
        const int logical_block = token_idx / block_size;
        const int physical_block =
            block_table[request_idx * max_num_blocks + logical_block];
        const int block_offset = token_idx % block_size;
        const float v_scale = __half2float(
            value_scales[static_cast<int64_t>(physical_block) *
                             scale_block_stride +
                         static_cast<int64_t>(kv_head_idx) * scale_head_stride +
                         channel_block]);
        const int64_t value_index =
            static_cast<int64_t>(physical_block) * value_block_stride +
            static_cast<int64_t>(block_offset) * value_token_stride +
            static_cast<int64_t>(kv_head_idx) * value_head_stride + output_dim;
        const __half value = __float2half_rn(
            static_cast<float>(value_cache[value_index]) * v_scale);
        tile_accumulator = fmaf(
            scores[token_local], __half2float(value), tile_accumulator);
      }
      output_accumulator = output_accumulator * accumulator_alpha +
                           tile_accumulator * accumulator_beta;
    }
    running_sum = running_sum * accumulator_alpha +
                  tile_sum_shared * accumulator_beta;
    running_max = fmaxf(running_max, tile_max_shared);
    __syncthreads();
  }

  if (output_dim < head_dim) {
    const float inverse_sum = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
    out[output_dim] = __float2half(output_accumulator * inverse_sum);
  }
}

void check_int8_block32_cache(const at::Tensor& key_cache,
                              const at::Tensor& value_cache,
                              const at::Tensor& key_scales,
                              const at::Tensor& value_scales) {
  TORCH_CHECK(key_cache.is_cuda() && value_cache.is_cuda(),
              "INT8 block cache tensors must be CUDA tensors");
  TORCH_CHECK(key_scales.is_cuda() && value_scales.is_cuda(),
              "INT8 block scale tensors must be CUDA tensors");
  TORCH_CHECK(key_cache.scalar_type() == at::kChar &&
                  value_cache.scalar_type() == at::kChar,
              "INT8 block cache payloads must use int8 storage");
  TORCH_CHECK(key_scales.scalar_type() == at::kHalf &&
                  value_scales.scalar_type() == at::kHalf,
              "INT8 block scales must use fp16 storage");
  TORCH_CHECK(key_cache.dim() == 4 && value_cache.dim() == 4,
              "INT8 block cache payloads require [blocks,tokens,heads,dim]");
  TORCH_CHECK(key_scales.dim() == 3 && value_scales.dim() == 3,
              "INT8 block scales require [blocks,heads,channel_blocks]");
  TORCH_CHECK(key_scales.stride(2) == 1 && value_scales.stride(2) == 1,
              "INT8 block scale channel dimensions must be contiguous");
  TORCH_CHECK(key_cache.sizes() == value_cache.sizes(),
              "INT8 K/V cache payload shapes must match");
  TORCH_CHECK(key_scales.sizes() == value_scales.sizes(),
              "INT8 K/V scale shapes must match");
  TORCH_CHECK(key_cache.size(0) == key_scales.size(0) &&
                  key_cache.size(2) == key_scales.size(1),
              "INT8 block scale shape must match cache pages and heads");
  TORCH_CHECK(key_cache.size(3) % kChannelBlockSize == 0 &&
                  key_scales.size(2) == key_cache.size(3) / kChannelBlockSize,
              "INT8 block cache requires one scale per 32-channel block");
  TORCH_CHECK(key_cache.stride(3) == 1 && value_cache.stride(3) == 1,
              "INT8 block cache head dimensions must be contiguous");
}

}  // namespace

void flash_attention_int8_block32_reshape_and_cache(
    const at::Tensor& key, const at::Tensor& value, at::Tensor& key_cache,
    at::Tensor& value_cache, at::Tensor& key_scales, at::Tensor& value_scales,
    at::Tensor& page_owners, const at::Tensor& slot_mapping) {
  check_int8_block32_cache(key_cache, value_cache, key_scales, value_scales);
  TORCH_CHECK(key.is_cuda() && value.is_cuda() && page_owners.is_cuda() &&
                  slot_mapping.is_cuda(),
              "INT8 block cache writer tensors must be CUDA tensors");
  TORCH_CHECK(key.scalar_type() == at::kHalf && value.scalar_type() == at::kHalf,
              "INT8 block cache writer requires fp16 K/V inputs");
  TORCH_CHECK(key.dim() == 3 && value.dim() == 3,
              "INT8 block cache writer requires [tokens,heads,dim] K/V");
  TORCH_CHECK(key.sizes() == value.sizes() && key.size(0) == slot_mapping.numel(),
              "INT8 block cache writer input shapes must match slot_mapping");
  TORCH_CHECK(key.size(1) == key_cache.size(2) &&
                  key.size(2) == key_cache.size(3),
              "INT8 block cache writer K/V shape mismatch");
  TORCH_CHECK(key.stride(2) == 1 && value.stride(2) == 1,
              "INT8 block cache writer K/V head dimensions must be contiguous");
  TORCH_CHECK(slot_mapping.scalar_type() == at::kLong &&
                  slot_mapping.dim() == 1 && slot_mapping.is_contiguous(),
              "INT8 block cache slot_mapping must be contiguous int64");
  TORCH_CHECK(page_owners.scalar_type() == at::kInt &&
                  page_owners.dim() == 1 &&
                  page_owners.numel() == key_cache.size(0),
              "INT8 block cache requires one int32 publication owner per page");

  if (key.size(0) == 0) {
    return;
  }
  TORCH_CHECK(key.size(0) <= 65535,
              "INT8 block cache writer supports at most 65535 input tokens");
  TORCH_CHECK(key.device() == key_cache.device() &&
                  value.device() == key_cache.device() &&
                  page_owners.device() == key_cache.device() &&
                  slot_mapping.device() == key_cache.device(),
              "INT8 block cache writer tensors must share one CUDA device");

  c10::cuda::CUDAGuard device_guard(key.device());
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  constexpr int kOwnerThreads = 256;
  const int owner_blocks =
      (key_cache.size(0) + kOwnerThreads - 1) / kOwnerThreads;
  reset_int8_block32_page_owners_kernel<<<owner_blocks, kOwnerThreads, 0,
                                          stream>>>(
      page_owners.data_ptr<int>(), key_cache.size(0), page_owners.stride(0));
  const int token_blocks =
      (key.size(0) + kOwnerThreads - 1) / kOwnerThreads;
  mark_int8_block32_page_owners_kernel<<<token_blocks, kOwnerThreads, 0,
                                         stream>>>(
      slot_mapping.data_ptr<int64_t>(), page_owners.data_ptr<int>(),
      key.size(0), key_cache.size(0), key_cache.size(1),
      page_owners.stride(0));

  const dim3 grid(key_cache.size(2), key_scales.size(2), key.size(0));
  int8_block32_reshape_and_cache_kernel<<<grid, 32, 0, stream>>>(
      reinterpret_cast<const __half*>(key.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(value.data_ptr<at::Half>()),
      key_cache.data_ptr<int8_t>(), value_cache.data_ptr<int8_t>(),
      reinterpret_cast<__half*>(key_scales.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(value_scales.data_ptr<at::Half>()),
      page_owners.data_ptr<int>(), slot_mapping.data_ptr<int64_t>(),
      key.size(0), key_cache.size(0), key_cache.size(1), key_cache.size(2),
      key_cache.size(3), key_scales.size(2), page_owners.stride(0),
      key.stride(0), key.stride(1), value.stride(0), value.stride(1),
      key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
      value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
      key_scales.stride(0), key_scales.stride(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void flash_attention_int8_block32_decode_paged(
    const at::Tensor& query, const at::Tensor& key_cache,
    const at::Tensor& value_cache, const at::Tensor& key_scales,
    const at::Tensor& value_scales, const at::Tensor& block_table,
    const at::Tensor& seq_lens, at::Tensor& output, const float softmax_scale) {
  check_int8_block32_cache(key_cache, value_cache, key_scales, value_scales);
  TORCH_CHECK(query.is_cuda() && block_table.is_cuda() && seq_lens.is_cuda() &&
                  output.is_cuda(),
              "INT8 block decode tensors must be CUDA tensors");
  TORCH_CHECK(query.scalar_type() == at::kHalf && output.scalar_type() == at::kHalf,
              "INT8 block decode requires fp16 query and output");
  TORCH_CHECK(block_table.scalar_type() == at::kInt &&
                  seq_lens.scalar_type() == at::kInt,
              "INT8 block decode metadata must use int32");
  TORCH_CHECK(query.dim() == 3 && output.sizes() == query.sizes(),
              "INT8 block decode requires matching [batch,heads,dim] tensors");
  TORCH_CHECK(query.stride(2) == 1 && output.stride(2) == 1,
              "INT8 block decode query and output head dimensions must be contiguous");
  TORCH_CHECK(block_table.dim() == 2 && block_table.is_contiguous() &&
                  seq_lens.dim() == 1 && seq_lens.is_contiguous(),
              "INT8 block decode metadata must be contiguous");
  TORCH_CHECK(query.size(1) % key_cache.size(2) == 0 &&
                  query.size(2) == key_cache.size(3) &&
                  query.size(2) <= kThreads,
              "INT8 block decode head shape mismatch or dimension exceeds 256");
  TORCH_CHECK(query.size(0) <= block_table.size(0) &&
                  query.size(0) <= seq_lens.size(0),
              "INT8 block decode metadata batch size mismatch");
  TORCH_CHECK(query.device() == key_cache.device() &&
                  output.device() == key_cache.device() &&
                  block_table.device() == key_cache.device() &&
                  seq_lens.device() == key_cache.device(),
              "INT8 block decode tensors must share one CUDA device");
  TORCH_CHECK(softmax_scale > 0.0f,
              "INT8 block decode softmax_scale must be positive");

  c10::cuda::CUDAGuard device_guard(query.device());
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const dim3 grid(query.size(0), query.size(1));
  int8_block32_decode_kernel<<<grid, kThreads, 0, stream>>>(
      reinterpret_cast<const __half*>(query.data_ptr<at::Half>()),
      key_cache.data_ptr<int8_t>(), value_cache.data_ptr<int8_t>(),
      reinterpret_cast<const __half*>(key_scales.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(value_scales.data_ptr<at::Half>()),
      nullptr, block_table.data_ptr<int>(), seq_lens.data_ptr<int>(),
      reinterpret_cast<__half*>(output.data_ptr<at::Half>()), query.size(0),
      query.size(0), block_table.size(1), query.size(1), key_cache.size(2),
      key_cache.size(1),
      query.size(2), key_scales.size(2), query.stride(0), query.stride(1),
      key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
      value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
      key_scales.stride(0), key_scales.stride(1), output.stride(0),
      output.stride(1), softmax_scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void flash_attention_int8_block32_prefill_paged(
    const at::Tensor& query, const at::Tensor& key_cache,
    const at::Tensor& value_cache, const at::Tensor& key_scales,
    const at::Tensor& value_scales, const at::Tensor& block_table,
    const at::Tensor& seq_lens, const at::Tensor& query_start_loc,
    at::Tensor& output, const float softmax_scale) {
  check_int8_block32_cache(key_cache, value_cache, key_scales, value_scales);
  TORCH_CHECK(query.is_cuda() && block_table.is_cuda() && seq_lens.is_cuda() &&
                  query_start_loc.is_cuda() && output.is_cuda(),
              "INT8 block prefill tensors must be CUDA tensors");
  TORCH_CHECK(query.scalar_type() == at::kHalf &&
                  output.scalar_type() == at::kHalf,
              "INT8 block prefill requires fp16 query and output");
  TORCH_CHECK(block_table.scalar_type() == at::kInt &&
                  seq_lens.scalar_type() == at::kInt &&
                  query_start_loc.scalar_type() == at::kInt,
              "INT8 block prefill metadata must use int32");
  TORCH_CHECK(query.dim() == 3 && output.sizes() == query.sizes(),
              "INT8 block prefill requires matching [tokens,heads,dim] tensors");
  TORCH_CHECK(query.stride(2) == 1 && output.stride(2) == 1,
              "INT8 block prefill query and output head dimensions must be contiguous");
  TORCH_CHECK(block_table.dim() == 2 && block_table.is_contiguous() &&
                  seq_lens.dim() == 1 && seq_lens.is_contiguous() &&
                  query_start_loc.dim() == 1 && query_start_loc.is_contiguous() &&
                  query_start_loc.numel() == seq_lens.numel() + 1 &&
                  block_table.size(0) == seq_lens.size(0),
              "INT8 block prefill metadata shapes do not match");
  TORCH_CHECK(query.size(1) % key_cache.size(2) == 0 &&
                  query.size(2) == key_cache.size(3) &&
                  query.size(2) <= kThreads,
              "INT8 block prefill head shape mismatch or dimension exceeds 256");
  TORCH_CHECK(query.device() == key_cache.device() &&
                  output.device() == key_cache.device() &&
                  block_table.device() == key_cache.device() &&
                  seq_lens.device() == key_cache.device() &&
                  query_start_loc.device() == key_cache.device(),
              "INT8 block prefill tensors must share one CUDA device");
  TORCH_CHECK(softmax_scale > 0.0f,
              "INT8 block prefill softmax_scale must be positive");
  if (query.size(0) == 0) {
    return;
  }

  c10::cuda::CUDAGuard device_guard(query.device());
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const dim3 grid(query.size(0), query.size(1));
  int8_block32_decode_kernel<<<grid, kThreads, 0, stream>>>(
      reinterpret_cast<const __half*>(query.data_ptr<at::Half>()),
      key_cache.data_ptr<int8_t>(), value_cache.data_ptr<int8_t>(),
      reinterpret_cast<const __half*>(key_scales.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(value_scales.data_ptr<at::Half>()),
      query_start_loc.data_ptr<int>(), block_table.data_ptr<int>(),
      seq_lens.data_ptr<int>(),
      reinterpret_cast<__half*>(output.data_ptr<at::Half>()), query.size(0),
      seq_lens.size(0), block_table.size(1), query.size(1), key_cache.size(2),
      key_cache.size(1), query.size(2), key_scales.size(2), query.stride(0),
      query.stride(1), key_cache.stride(0), key_cache.stride(1),
      key_cache.stride(2), value_cache.stride(0), value_cache.stride(1),
      value_cache.stride(2), key_scales.stride(0), key_scales.stride(1),
      output.stride(0), output.stride(1), softmax_scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
