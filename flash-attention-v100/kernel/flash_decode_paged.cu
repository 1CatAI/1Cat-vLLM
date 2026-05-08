#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <string>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "fp8_kv_utils.cuh"

namespace {

int kv_cache_dtype_code_from_string(const std::string& kv_cache_dtype) {
  if (kv_cache_dtype == "auto" || kv_cache_dtype == "bfloat16") {
    return flash_v100::KV_CACHE_DTYPE_FP16;
  }
  if (kv_cache_dtype == "fp8" || kv_cache_dtype == "fp8_e4m3") {
    return flash_v100::KV_CACHE_DTYPE_FP8_E4M3;
  }
  if (kv_cache_dtype == "fp8_e5m2") {
    return flash_v100::KV_CACHE_DTYPE_FP8_E5M2;
  }
  return -1;
}

constexpr int kWarpSize = 32;
constexpr int kThreadsPerBlock = 256;
constexpr int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;

__device__ __forceinline__ float warp_reduce_sum(float val) {
  #pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
  #pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

template<int NUM_WARPS>
__device__ __forceinline__ float block_reduce_sum(float val) {
  __shared__ float shared[NUM_WARPS];
  __shared__ float result;
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;

  val = warp_reduce_sum(val);
  if (lane == 0) {
    shared[warp] = val;
  }
  __syncthreads();

  val = threadIdx.x < NUM_WARPS ? shared[lane] : 0.f;
  if (warp == 0) {
    val = warp_reduce_sum(val);
    if (lane == 0) {
      result = val;
    }
  }
  __syncthreads();
  return result;
}

template<int NUM_WARPS>
__device__ __forceinline__ float block_reduce_max(float val) {
  __shared__ float shared[NUM_WARPS];
  __shared__ float result;
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;

  val = warp_reduce_max(val);
  if (lane == 0) {
    shared[warp] = val;
  }
  __syncthreads();

  val = threadIdx.x < NUM_WARPS ? shared[lane] : -1.0e20f;
  if (warp == 0) {
    val = warp_reduce_max(val);
    if (lane == 0) {
      result = val;
    }
  }
  __syncthreads();
  return result;
}

template<int D>
__device__ __forceinline__ float dot_qk_half2(
    const __half* __restrict__ q_ptr,
    const __half* __restrict__ k_ptr,
    const int lane) {
  static_assert(D % 2 == 0, "Head dim must be even for half2 dot");
  const __half2* q_ptr2 = reinterpret_cast<const __half2*>(q_ptr);
  const __half2* k_ptr2 = reinterpret_cast<const __half2*>(k_ptr);

  float acc = 0.f;
  #pragma unroll
  for (int i = lane; i < D / 2; i += kWarpSize) {
    const float2 qv = __half22float2(q_ptr2[i]);
    const float2 kv = __half22float2(k_ptr2[i]);
    acc = fmaf(qv.x, kv.x, acc);
    acc = fmaf(qv.y, kv.y, acc);
  }
  return warp_reduce_sum(acc);
}

template<int D, int KV_DTYPE>
__device__ __forceinline__ float dot_qk_cache(
    const __half* __restrict__ q_ptr,
    const void* __restrict__ k_cache,
    const int64_t k_index_base,
    const int lane) {
  if constexpr (KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP16) {
    const __half* k_ptr = reinterpret_cast<const __half*>(k_cache) + k_index_base;
    return dot_qk_half2<D>(q_ptr, k_ptr, lane);
  } else if constexpr (KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E5M2) {
    static_assert(D % 2 == 0, "Head dim must be even for e5m2 half2 dot");
    const __half2* q_ptr2 = reinterpret_cast<const __half2*>(q_ptr);
    float acc = 0.f;
    #pragma unroll
    for (int i = lane; i < D / 2; i += kWarpSize) {
      const float2 qv = __half22float2(q_ptr2[i]);
      const __half2 k_h2 = flash_v100::load_fp8_e5m2_half2_unscaled(
          k_cache, k_index_base + static_cast<int64_t>(i) * 2);
      const float2 kv = __half22float2(k_h2);
      acc = fmaf(qv.x, kv.x, acc);
      acc = fmaf(qv.y, kv.y, acc);
    }
    return warp_reduce_sum(acc);
  } else {
    float acc = 0.f;
    #pragma unroll
    for (int d = lane; d < D; d += kWarpSize) {
      const float qv = __half2float(q_ptr[d]);
      const float kv = flash_v100::load_kv_cache_float_unscaled<KV_DTYPE>(
          k_cache, k_index_base + d);
      acc = fmaf(qv, kv, acc);
    }
    return warp_reduce_sum(acc);
  }
}

template<int D, int PARTITION_SIZE, int KV_DTYPE>
__global__ void flash_attention_decode_partition_kernel(
    const __half* __restrict__ q,
    const void* __restrict__ k_cache,
    const void* __restrict__ v_cache,
    __half* __restrict__ tmp_out,
    float* __restrict__ max_logits,
    float* __restrict__ exp_sums,
    const int* __restrict__ block_table,
    const int* __restrict__ seq_lens,
    const int batch_size,
    const int max_num_blocks,
    const int max_num_partitions,
    const int num_heads_q,
    const int num_heads_kv,
    const int block_size,
    const int64_t q_stride0,
    const int64_t q_stride1,
    const int64_t tmp_out_stride0,
    const int64_t tmp_out_stride1,
    const int64_t tmp_out_stride2,
    const int64_t stats_stride0,
    const int64_t stats_stride1,
    const int64_t k_block_stride,
    const int64_t k_token_stride,
    const int64_t k_head_stride,
    const int64_t v_block_stride,
    const int64_t v_token_stride,
    const int64_t v_head_stride,
    const float softmax_scale,
    const float k_scale,
    const float v_scale) {
  const int batch_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;

  if (batch_idx >= batch_size || head_idx >= num_heads_q ||
      partition_idx >= max_num_partitions) {
    return;
  }

  const int seq_len = seq_lens[batch_idx];
  const int start_token_idx = partition_idx * PARTITION_SIZE;
  if (seq_len <= 0 || start_token_idx >= seq_len) {
    return;
  }

  const int part_tokens = min(PARTITION_SIZE, seq_len - start_token_idx);
  const int q_per_kv = num_heads_q / num_heads_kv;
  const int kv_head_idx = head_idx / q_per_kv;
  const int lane = threadIdx.x % kWarpSize;
  const int warp_idx = threadIdx.x / kWarpSize;
  const float score_scale =
      KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP16 ? softmax_scale
                                                  : softmax_scale * k_scale;

  __shared__ __half q_shared[D];
  __shared__ float scores_shared[PARTITION_SIZE];
  __shared__ int block_idx_shared[PARTITION_SIZE];
  __shared__ int block_offset_shared[PARTITION_SIZE];

  const __half* q_ptr = q + static_cast<int64_t>(batch_idx) * q_stride0 +
                        static_cast<int64_t>(head_idx) * q_stride1;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    q_shared[d] = q_ptr[d];
  }
  for (int i = threadIdx.x; i < part_tokens; i += blockDim.x) {
    const int token_idx = start_token_idx + i;
    const int logical_block = token_idx / block_size;
    block_idx_shared[i] =
        block_table[batch_idx * max_num_blocks + logical_block];
    block_offset_shared[i] = token_idx - logical_block * block_size;
  }
  __syncthreads();

  float local_max = -1.0e20f;
  for (int token_local = warp_idx; token_local < part_tokens;
       token_local += kWarpsPerBlock) {
    const int physical_block = block_idx_shared[token_local];
    const int block_offset = block_offset_shared[token_local];
    const int64_t k_index =
        static_cast<int64_t>(physical_block) * k_block_stride +
        static_cast<int64_t>(block_offset) * k_token_stride +
        static_cast<int64_t>(kv_head_idx) * k_head_stride;

    float score =
        dot_qk_cache<D, KV_DTYPE>(q_shared, k_cache, k_index, lane);
    if (lane == 0) {
      score *= score_scale;
      scores_shared[token_local] = score;
      local_max = fmaxf(local_max, score);
    }
  }

  const float part_max = block_reduce_max<kWarpsPerBlock>(local_max);

  float local_sum = 0.f;
  for (int i = threadIdx.x; i < part_tokens; i += blockDim.x) {
    const float p = __expf(scores_shared[i] - part_max);
    scores_shared[i] = p;
    local_sum += p;
  }
  const float part_sum = block_reduce_sum<kWarpsPerBlock>(local_sum);
  const float inv_part_sum = part_sum > 0.f ? 1.f / part_sum : 0.f;
  __syncthreads();

  const int64_t tmp_out_base =
      static_cast<int64_t>(batch_idx) * tmp_out_stride0 +
      static_cast<int64_t>(head_idx) * tmp_out_stride1 +
      static_cast<int64_t>(partition_idx) * tmp_out_stride2;

  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float acc = 0.f;
    for (int i = 0; i < part_tokens; ++i) {
      const int physical_block = block_idx_shared[i];
      const int block_offset = block_offset_shared[i];
      const int64_t v_index =
          static_cast<int64_t>(physical_block) * v_block_stride +
          static_cast<int64_t>(block_offset) * v_token_stride +
          static_cast<int64_t>(kv_head_idx) * v_head_stride + d;
      const float vv = flash_v100::load_kv_cache_float_unscaled<KV_DTYPE>(
          v_cache, v_index);
      acc = fmaf(scores_shared[i], vv, acc);
    }
    const float out_scale =
        KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP16 ? inv_part_sum
                                                    : inv_part_sum * v_scale;
    tmp_out[tmp_out_base + d] = __float2half(acc * out_scale);
  }

  if (threadIdx.x == 0) {
    const int64_t stats_index =
        static_cast<int64_t>(batch_idx) * stats_stride0 +
        static_cast<int64_t>(head_idx) * stats_stride1 + partition_idx;
    max_logits[stats_index] = part_max;
    exp_sums[stats_index] = part_sum;
  }
}

template<int D, int PARTITION_SIZE>
__global__ void flash_attention_decode_reduce_kernel(
    const __half* __restrict__ tmp_out,
    const float* __restrict__ max_logits,
    const float* __restrict__ exp_sums,
    const int* __restrict__ seq_lens,
    __half* __restrict__ out,
    const int batch_size,
    const int max_num_partitions,
    const int num_heads_q,
    const int64_t tmp_out_stride0,
    const int64_t tmp_out_stride1,
    const int64_t tmp_out_stride2,
    const int64_t stats_stride0,
    const int64_t stats_stride1,
    const int64_t out_stride0,
    const int64_t out_stride1) {
  const int batch_idx = blockIdx.x;
  const int head_idx = blockIdx.y;

  if (batch_idx >= batch_size || head_idx >= num_heads_q) {
    return;
  }

  const int seq_len = seq_lens[batch_idx];
  const int num_partitions =
      (seq_len + PARTITION_SIZE - 1) / PARTITION_SIZE;

  if (seq_len <= 0 || num_partitions <= 0) {
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
      out[static_cast<int64_t>(batch_idx) * out_stride0 +
          static_cast<int64_t>(head_idx) * out_stride1 + d] =
          __float2half(0.f);
    }
    return;
  }

  extern __shared__ float shared_mem[];
  float* max_shared = shared_mem;
  float* weight_shared = shared_mem + max_num_partitions;

  float local_max = -1.0e20f;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    const int64_t stats_index =
        static_cast<int64_t>(batch_idx) * stats_stride0 +
        static_cast<int64_t>(head_idx) * stats_stride1 + i;
    const float m = max_logits[stats_index];
    max_shared[i] = m;
    local_max = fmaxf(local_max, m);
  }
  const float global_max = block_reduce_max<kWarpsPerBlock>(local_max);

  float local_sum = 0.f;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    const int64_t stats_index =
        static_cast<int64_t>(batch_idx) * stats_stride0 +
        static_cast<int64_t>(head_idx) * stats_stride1 + i;
    const float weight = exp_sums[stats_index] * __expf(max_shared[i] - global_max);
    weight_shared[i] = weight;
    local_sum += weight;
  }
  const float global_sum = block_reduce_sum<kWarpsPerBlock>(local_sum);
  const float inv_global_sum = global_sum > 0.f ? 1.f / global_sum : 0.f;
  __syncthreads();

  const int64_t out_base =
      static_cast<int64_t>(batch_idx) * out_stride0 +
      static_cast<int64_t>(head_idx) * out_stride1;
  const int64_t tmp_out_base =
      static_cast<int64_t>(batch_idx) * tmp_out_stride0 +
      static_cast<int64_t>(head_idx) * tmp_out_stride1;

  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float acc = 0.f;
    for (int i = 0; i < num_partitions; ++i) {
      acc = fmaf(
          weight_shared[i],
          __half2float(tmp_out[tmp_out_base + static_cast<int64_t>(i) * tmp_out_stride2 + d]),
          acc);
    }
    out[out_base + d] = __float2half(acc * inv_global_sum);
  }
}

template<int D, int PARTITION_SIZE, int KV_DTYPE>
void launch_flash_attention_decode_paged(
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    at::Tensor& out,
    const at::Tensor& block_table,
    const at::Tensor& seq_lens,
    at::Tensor& tmp_out,
    at::Tensor& max_logits,
    at::Tensor& exp_sums,
    const float softmax_scale,
    const float k_scale,
    const float v_scale,
    cudaStream_t stream) {
  const int batch_size = q.size(0);
  const int num_heads_q = q.size(1);
  const int num_heads_kv = k_cache.size(2);
  const int block_size = k_cache.size(1);
  const int max_num_blocks = block_table.size(1);
  const int max_num_partitions = tmp_out.size(2);

  const dim3 partition_grid(batch_size, num_heads_q, max_num_partitions);
  const dim3 reduce_grid(batch_size, num_heads_q, 1);
  const dim3 block(kThreadsPerBlock);
  const size_t reduce_shared_mem =
      static_cast<size_t>(2 * max_num_partitions) * sizeof(float);

  flash_attention_decode_partition_kernel<D, PARTITION_SIZE, KV_DTYPE>
      <<<partition_grid, block, 0, stream>>>(
      reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
      k_cache.data_ptr(),
      v_cache.data_ptr(),
      reinterpret_cast<__half*>(tmp_out.data_ptr<at::Half>()),
      max_logits.data_ptr<float>(),
      exp_sums.data_ptr<float>(),
      block_table.data_ptr<int>(),
      seq_lens.data_ptr<int>(),
      batch_size,
      max_num_blocks,
      max_num_partitions,
      num_heads_q,
      num_heads_kv,
      block_size,
      q.stride(0),
      q.stride(1),
      tmp_out.stride(0),
      tmp_out.stride(1),
      tmp_out.stride(2),
      max_logits.stride(0),
      max_logits.stride(1),
      k_cache.stride(0),
      k_cache.stride(1),
      k_cache.stride(2),
      v_cache.stride(0),
      v_cache.stride(1),
      v_cache.stride(2),
      softmax_scale,
      k_scale,
      v_scale);

  flash_attention_decode_reduce_kernel<D, PARTITION_SIZE><<<reduce_grid, block, reduce_shared_mem, stream>>>(
      reinterpret_cast<const __half*>(tmp_out.data_ptr<at::Half>()),
      max_logits.data_ptr<float>(),
      exp_sums.data_ptr<float>(),
      seq_lens.data_ptr<int>(),
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
      batch_size,
      max_num_partitions,
      num_heads_q,
      tmp_out.stride(0),
      tmp_out.stride(1),
      tmp_out.stride(2),
      max_logits.stride(0),
      max_logits.stride(1),
      out.stride(0),
      out.stride(1));
}

}

at::Tensor flash_attention_decode_paged(
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    std::optional<at::Tensor>& out_,
    const at::Tensor& block_table,
    const at::Tensor& seq_lens,
    at::Tensor& tmp_out,
    at::Tensor& max_logits,
    at::Tensor& exp_sums,
    const float softmax_scale,
    const int partition_size,
    const std::string& kv_cache_dtype,
    const float k_scale,
    const float v_scale) {
  TORCH_CHECK(q.is_cuda(), "q must be on CUDA");
  TORCH_CHECK(k_cache.is_cuda() && v_cache.is_cuda(), "k/v cache must be on CUDA");
  TORCH_CHECK(block_table.is_cuda() && seq_lens.is_cuda(), "block_table and seq_lens must be on CUDA");
  TORCH_CHECK(tmp_out.is_cuda() && max_logits.is_cuda() && exp_sums.is_cuda(),
              "workspace tensors must be on CUDA");
  TORCH_CHECK(q.dtype() == torch::kFloat16, "q must be fp16");
  const int kv_dtype_code = kv_cache_dtype_code_from_string(kv_cache_dtype);
  TORCH_CHECK(kv_dtype_code >= 0, "Unsupported kv_cache_dtype: ", kv_cache_dtype);
  if (kv_dtype_code == flash_v100::KV_CACHE_DTYPE_FP16) {
    TORCH_CHECK(k_cache.dtype() == torch::kFloat16, "k_cache must be fp16");
    TORCH_CHECK(v_cache.dtype() == torch::kFloat16, "v_cache must be fp16");
  } else {
    TORCH_CHECK(k_cache.dtype() == torch::kUInt8,
                "fp8 k_cache must be stored as uint8");
    TORCH_CHECK(v_cache.dtype() == torch::kUInt8,
                "fp8 v_cache must be stored as uint8");
    TORCH_CHECK(k_scale > 0.f && v_scale > 0.f,
                "fp8 k/v scales must be positive");
  }
  TORCH_CHECK(tmp_out.dtype() == torch::kFloat16, "tmp_out must be fp16");
  TORCH_CHECK(max_logits.dtype() == torch::kFloat32, "max_logits must be fp32");
  TORCH_CHECK(exp_sums.dtype() == torch::kFloat32, "exp_sums must be fp32");
  TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must be int32");
  TORCH_CHECK(seq_lens.dtype() == torch::kInt32, "seq_lens must be int32");
  TORCH_CHECK(q.dim() == 3, "q must have shape [B, H, D]");
  TORCH_CHECK(k_cache.dim() == 4, "k_cache must have shape [num_blocks, block_size, H_kv, D]");
  TORCH_CHECK(v_cache.dim() == 4, "v_cache must have shape [num_blocks, block_size, H_kv, D]");
  TORCH_CHECK(block_table.dim() == 2, "block_table must have shape [B, max_num_blocks]");
  TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must have shape [B]");
  TORCH_CHECK(tmp_out.dim() == 4, "tmp_out must have shape [B_cap, H, P, D]");
  TORCH_CHECK(max_logits.dim() == 3, "max_logits must have shape [B_cap, H, P]");
  TORCH_CHECK(exp_sums.dim() == 3, "exp_sums must have shape [B_cap, H, P]");
  TORCH_CHECK(q.stride(-1) == 1, "q last dim must be contiguous");
  TORCH_CHECK(k_cache.stride(-1) == 1, "k_cache last dim must be contiguous");
  TORCH_CHECK(v_cache.stride(-1) == 1, "v_cache last dim must be contiguous");
  TORCH_CHECK(tmp_out.stride(-1) == 1, "tmp_out last dim must be contiguous");

  const int batch_size = q.size(0);
  const int num_heads_q = q.size(1);
  const int head_dim = q.size(2);
  const int num_heads_kv = k_cache.size(2);

  TORCH_CHECK(q.size(0) <= block_table.size(0), "block_table batch size must cover q batch size");
  TORCH_CHECK(q.size(0) <= seq_lens.size(0), "seq_lens batch size must cover q batch size");
  TORCH_CHECK(q.size(0) <= tmp_out.size(0), "tmp_out batch size must cover q batch size");
  TORCH_CHECK(num_heads_q == tmp_out.size(1), "tmp_out head dimension mismatch");
  TORCH_CHECK(head_dim == tmp_out.size(3), "tmp_out head_dim mismatch");
  TORCH_CHECK(max_logits.size(0) == tmp_out.size(0) && max_logits.size(1) == tmp_out.size(1) &&
              max_logits.size(2) == tmp_out.size(2), "max_logits shape mismatch");
  TORCH_CHECK(exp_sums.sizes() == max_logits.sizes(), "exp_sums shape mismatch");
  TORCH_CHECK(num_heads_q % num_heads_kv == 0, "num_heads_q must be divisible by num_heads_kv");
  TORCH_CHECK(k_cache.size(3) == head_dim, "k_cache head_dim mismatch");
  TORCH_CHECK(v_cache.size(3) == head_dim, "v_cache head_dim mismatch");
  TORCH_CHECK(partition_size == 256 || partition_size == 512 || partition_size == 1024,
              "Unsupported decode partition_size: ", partition_size);

  at::Tensor out = out_.has_value() ? out_.value() : torch::empty_like(q);
  TORCH_CHECK(out.is_cuda(), "out must be on CUDA");
  TORCH_CHECK(out.dtype() == torch::kFloat16, "out must be fp16");
  TORCH_CHECK(out.sizes() == q.sizes(), "out must have same shape as q");
  TORCH_CHECK(out.stride(-1) == 1, "out last dim must be contiguous");

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  c10::cuda::CUDAGuard device_guard(q.device());

  #define LAUNCH_TYPED(HDIM, PARTITION, KV_DTYPE_CODE)                          \
    launch_flash_attention_decode_paged<HDIM, PARTITION, KV_DTYPE_CODE>(        \
        q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,   \
        exp_sums, softmax_scale, k_scale, v_scale, stream)

  #define LAUNCH_BY_KV_DTYPE(HDIM, PARTITION)                                   \
    do {                                                                        \
      switch (kv_dtype_code) {                                                  \
        case flash_v100::KV_CACHE_DTYPE_FP16:                                   \
          LAUNCH_TYPED(HDIM, PARTITION, flash_v100::KV_CACHE_DTYPE_FP16);       \
          break;                                                                \
        case flash_v100::KV_CACHE_DTYPE_FP8_E4M3:                               \
          LAUNCH_TYPED(HDIM, PARTITION, flash_v100::KV_CACHE_DTYPE_FP8_E4M3);   \
          break;                                                                \
        case flash_v100::KV_CACHE_DTYPE_FP8_E5M2:                               \
          LAUNCH_TYPED(HDIM, PARTITION, flash_v100::KV_CACHE_DTYPE_FP8_E5M2);   \
          break;                                                                \
        default:                                                                \
          TORCH_CHECK(false, "Unsupported kv_cache_dtype: ", kv_cache_dtype);  \
      }                                                                         \
    } while (0)

  #define LAUNCH_BY_PARTITION(HDIM)                                             \
    do {                                                                         \
      switch (partition_size) {                                                  \
        case 256:                                                                \
          LAUNCH_BY_KV_DTYPE(HDIM, 256);                                         \
          break;                                                                 \
        case 512:                                                                \
          LAUNCH_BY_KV_DTYPE(HDIM, 512);                                         \
          break;                                                                 \
        case 1024:                                                               \
          LAUNCH_BY_KV_DTYPE(HDIM, 1024);                                        \
          break;                                                                 \
        default:                                                                 \
          TORCH_CHECK(false, "Unsupported decode partition_size: ", partition_size); \
      }                                                                          \
    } while (0)

  switch (head_dim) {
    case 64:
      LAUNCH_BY_PARTITION(64);
      break;
    case 80:
      LAUNCH_BY_PARTITION(80);
      break;
    case 96:
      LAUNCH_BY_PARTITION(96);
      break;
    case 112:
      LAUNCH_BY_PARTITION(112);
      break;
    case 128:
      LAUNCH_BY_PARTITION(128);
      break;
    case 256:
      LAUNCH_BY_PARTITION(256);
      break;
    default:
      TORCH_CHECK(false, "Unsupported head_dim for paged decode: ", head_dim);
  }

  #undef LAUNCH_BY_PARTITION
  #undef LAUNCH_BY_KV_DTYPE
  #undef LAUNCH_TYPED

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
