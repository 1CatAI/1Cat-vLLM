#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <string>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "fp8_kv_utils.cuh"
#include "fused_mma.h"

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
constexpr int kXQATCBlockN = 128;
constexpr int kXQATCStride = 128;
constexpr int kXQATCPageIdsCapacity = kXQATCBlockN / 16;
constexpr int kXQATC256WideWarpCount = 8;
constexpr int kXQATC256WideThreads = kXQATC256WideWarpCount * kWarpSize;
constexpr int kXQATC256WideBlockM = 8;
constexpr int kXQATC256WideThreadsPerRow = kWarpSize;
constexpr float kXQANegInf = -1.0e30f;

struct alignas(256) XQATCSmem256Wide {
  alignas(16) __half q[kXQATC256WideBlockM * 256];
  union {
    alignas(16) __half k[kXQATCBlockN * kXQATCStride];
    alignas(16) __half v[kXQATCBlockN * kXQATCStride];
  } reuse_kv;
  union {
    alignas(16) float s[kXQATC256WideBlockM * kXQATCBlockN];
    alignas(16) __half p[kXQATC256WideBlockM * kXQATCBlockN];
  } reuse_sp;
  alignas(16) float row_max[kXQATC256WideBlockM];
  alignas(16) float row_sum[kXQATC256WideBlockM];
  alignas(16) int page_ids[kXQATCPageIdsCapacity];
};

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

// ---------------------------------------------------------------------------
// Multi-query (small-q) paged decode: one CTA processes all q_len query
// tokens of a request against each K/V tile, so the KV cache is read once per
// request instead of once per query token. query_start_loc gives per-request
// token offsets into q/out; seq_lens holds per-request total lengths
// (including the new tokens). Query token j of a request attends to KV
// positions [0, seq_len - q_len + j] inclusive (causal within the verify
// block); the sliding window applies relative to each query token's own
// position. The partition kernel is instantiated for MQ_QLEN in {1,2,4,8}
// (MTP1..MTP7 verify shapes); callers pick the smallest MQ_QLEN covering the
// batch's max q_len, and requests with smaller q_len take zero-filled tail
// rows. kDecodeMqQMax caps q_len_max at 8 (smem budget at head_dim=256).
// ---------------------------------------------------------------------------

constexpr int kDecodeMqQMax = 8;

__device__ __forceinline__ int decode_mq_query_limit(
    const int seq_len, const int q_len, const int j) {
  return max(0, min(seq_len, seq_len - q_len + j + 1));
}

template<int D, int KV_DTYPE, int MQ_QLEN>
__device__ __forceinline__ void dot_qk_cache_mq(
    const __half* __restrict__ q_shared,
    const void* __restrict__ k_cache,
    const int64_t k_index_base,
    const int lane,
    float* __restrict__ acc) {
  if constexpr (KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP16) {
    static_assert(D % 2 == 0, "Head dim must be even for half2 dot");
    const __half2* q_ptr2 = reinterpret_cast<const __half2*>(q_shared);
    const __half2* k_ptr2 = reinterpret_cast<const __half2*>(
        reinterpret_cast<const __half*>(k_cache) + k_index_base);
    #pragma unroll
    for (int i = lane; i < D / 2; i += kWarpSize) {
      const float2 kv = __half22float2(k_ptr2[i]);
      #pragma unroll
      for (int j = 0; j < MQ_QLEN; ++j) {
        const float2 qv = __half22float2(q_ptr2[j * (D / 2) + i]);
        acc[j] = fmaf(qv.x, kv.x, acc[j]);
        acc[j] = fmaf(qv.y, kv.y, acc[j]);
      }
    }
  } else if constexpr (KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E5M2) {
    static_assert(D % 2 == 0, "Head dim must be even for e5m2 half2 dot");
    const __half2* q_ptr2 = reinterpret_cast<const __half2*>(q_shared);
    #pragma unroll
    for (int i = lane; i < D / 2; i += kWarpSize) {
      const __half2 k_h2 = flash_v100::load_fp8_e5m2_half2_unscaled(
          k_cache, k_index_base + static_cast<int64_t>(i) * 2);
      const float2 kv = __half22float2(k_h2);
      #pragma unroll
      for (int j = 0; j < MQ_QLEN; ++j) {
        const float2 qv = __half22float2(q_ptr2[j * (D / 2) + i]);
        acc[j] = fmaf(qv.x, kv.x, acc[j]);
        acc[j] = fmaf(qv.y, kv.y, acc[j]);
      }
    }
  } else {
    #pragma unroll
    for (int d = lane; d < D; d += kWarpSize) {
      const float kv = flash_v100::load_kv_cache_float_unscaled<KV_DTYPE>(
          k_cache, k_index_base + d);
      #pragma unroll
      for (int j = 0; j < MQ_QLEN; ++j) {
        const float qv = __half2float(q_shared[j * D + d]);
        acc[j] = fmaf(qv, kv, acc[j]);
      }
    }
  }
}

template<int NUM_WARPS, int COUNT>
__device__ __forceinline__ void block_reduce_max_mq(
    float* __restrict__ vals) {
  __shared__ float shared[NUM_WARPS][COUNT];
  __shared__ float result[COUNT];
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;

  #pragma unroll
  for (int j = 0; j < COUNT; ++j) {
    vals[j] = warp_reduce_max(vals[j]);
  }
  if (lane == 0) {
    #pragma unroll
    for (int j = 0; j < COUNT; ++j) {
      shared[warp][j] = vals[j];
    }
  }
  __syncthreads();

  if (warp == 0) {
    #pragma unroll
    for (int j = 0; j < COUNT; ++j) {
      float val = lane < NUM_WARPS ? shared[lane][j] : -1.0e20f;
      val = warp_reduce_max(val);
      if (lane == 0) {
        result[j] = val;
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int j = 0; j < COUNT; ++j) {
    vals[j] = result[j];
  }
  __syncthreads();
}

template<int NUM_WARPS, int COUNT>
__device__ __forceinline__ void block_reduce_sum_mq(
    float* __restrict__ vals) {
  __shared__ float shared[NUM_WARPS][COUNT];
  __shared__ float result[COUNT];
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;

  #pragma unroll
  for (int j = 0; j < COUNT; ++j) {
    vals[j] = warp_reduce_sum(vals[j]);
  }
  if (lane == 0) {
    #pragma unroll
    for (int j = 0; j < COUNT; ++j) {
      shared[warp][j] = vals[j];
    }
  }
  __syncthreads();

  if (warp == 0) {
    #pragma unroll
    for (int j = 0; j < COUNT; ++j) {
      float val = lane < NUM_WARPS ? shared[lane][j] : 0.f;
      val = warp_reduce_sum(val);
      if (lane == 0) {
        result[j] = val;
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int j = 0; j < COUNT; ++j) {
    vals[j] = result[j];
  }
  __syncthreads();
}

template<int D, int PARTITION_SIZE, int KV_DTYPE, int MQ_QLEN>
__global__ void flash_attention_decode_partition_mq_kernel(
    const __half* __restrict__ q,
    const void* __restrict__ k_cache,
    const void* __restrict__ v_cache,
    __half* __restrict__ tmp_out,
    float* __restrict__ max_logits,
    float* __restrict__ exp_sums,
    const int* __restrict__ block_table,
    const int* __restrict__ seq_lens,
    const int* __restrict__ query_start_loc,
    const int* __restrict__ active_num_partitions,
    const int num_reqs,
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
    const int64_t tmp_out_stride3,
    const int64_t stats_stride0,
    const int64_t stats_stride1,
    const int64_t stats_stride2,
    const int64_t k_block_stride,
    const int64_t k_token_stride,
    const int64_t k_head_stride,
    const int64_t v_block_stride,
    const int64_t v_token_stride,
    const int64_t v_head_stride,
    const float softmax_scale,
    const float k_scale,
    const float v_scale,
    const int window_size_left,
    const int window_size_right) {
  const int req_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;

  if (req_idx >= num_reqs || head_idx >= num_heads_q ||
      partition_idx >= max_num_partitions) {
    return;
  }

  const int seq_len = seq_lens[req_idx];
  const int q_start = query_start_loc[req_idx];
  const int q_len = min(query_start_loc[req_idx + 1] - q_start, MQ_QLEN);
  const int start_token_idx = partition_idx * PARTITION_SIZE;
  if (seq_len <= 0 || q_len <= 0 || start_token_idx >= seq_len) {
    return;
  }
  const int runtime_num_partitions = active_num_partitions[0];
  const int seq_num_partitions =
      (seq_len + PARTITION_SIZE - 1) / PARTITION_SIZE;
  const int effective_num_partitions =
      min(max_num_partitions, max(runtime_num_partitions, seq_num_partitions));
  if (partition_idx >= effective_num_partitions) {
    return;
  }

  // Token range shared by all queries of this request: the first query owns
  // the leftmost window edge, the last query the rightmost causal edge.
  const int query_pos_first = seq_len - q_len;
  const int query_pos_last = seq_len - 1;
  const int min_token_idx = window_size_left >= 0
                                ? max(0, query_pos_first - window_size_left)
                                : 0;
  const int max_token_idx = window_size_right >= 0
                                ? min(seq_len - 1,
                                      query_pos_last + window_size_right)
                                : seq_len - 1;
  const int part_start = max(start_token_idx, min_token_idx);
  const int part_end = min(start_token_idx + PARTITION_SIZE, max_token_idx + 1);
  const int q_per_kv = num_heads_q / num_heads_kv;
  const int kv_head_idx = head_idx / q_per_kv;
  const int lane = threadIdx.x % kWarpSize;
  const int warp_idx = threadIdx.x / kWarpSize;
  const float score_scale =
      KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP16 ? softmax_scale
                                                  : softmax_scale * k_scale;

  const int64_t tmp_out_base =
      static_cast<int64_t>(req_idx) * tmp_out_stride0 +
      static_cast<int64_t>(head_idx) * tmp_out_stride1 +
      static_cast<int64_t>(partition_idx) * tmp_out_stride3;
  const int64_t stats_base =
      static_cast<int64_t>(req_idx) * stats_stride0 +
      static_cast<int64_t>(head_idx) * stats_stride1;
  if (part_start >= part_end) {
    for (int j = 0; j < q_len; ++j) {
      for (int d = threadIdx.x; d < D; d += blockDim.x) {
        tmp_out[tmp_out_base + static_cast<int64_t>(j) * tmp_out_stride2 + d] =
            __float2half(0.f);
      }
      if (threadIdx.x == 0) {
        const int64_t stats_index =
            stats_base + static_cast<int64_t>(j) * stats_stride2 +
            partition_idx;
        max_logits[stats_index] = -1.0e20f;
        exp_sums[stats_index] = 0.f;
      }
    }
    return;
  }

  const int part_tokens = part_end - part_start;

  __shared__ __half q_shared[MQ_QLEN * D];
  __shared__ float scores_shared[MQ_QLEN * PARTITION_SIZE];
  // Per-token KV slot in units of tokens: physical_block * tokens_per_block
  // (block_stride / token_stride) + block_offset. Works both for contiguous
  // [block, token, head, dim] caches (tokens_per_block == block_size) and the
  // interleaved [block, 2, token, head, dim] vLLM layout (== 2 * block_size);
  // the launcher requires K and V to share the same ratio, so the flat K/V
  // index is slot * token_stride + head * head_stride (+ d).
  __shared__ int slot_shared[PARTITION_SIZE];
  const int64_t tokens_per_block = k_block_stride / k_token_stride;

  const __half* q_ptr = q + static_cast<int64_t>(q_start) * q_stride0 +
                        static_cast<int64_t>(head_idx) * q_stride1;
  for (int j = 0; j < q_len; ++j) {
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
      q_shared[j * D + d] = q_ptr[static_cast<int64_t>(j) * q_stride0 + d];
    }
  }
  // Queries beyond this request's q_len get zeroed Q rows and empty windows
  // so their dots/scores are defined values that never become valid.
  #pragma unroll
  for (int j = q_len; j < MQ_QLEN; ++j) {
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
      q_shared[j * D + d] = __float2half(0.f);
    }
  }
  for (int i = threadIdx.x; i < part_tokens; i += blockDim.x) {
    const int token_idx = part_start + i;
    const int logical_block = token_idx / block_size;
    slot_shared[i] =
        block_table[req_idx * max_num_blocks + logical_block] *
            tokens_per_block +
        (token_idx - logical_block * block_size);
  }
  __syncthreads();

  int qlimit[MQ_QLEN];
  int qmin[MQ_QLEN];
  int qmax[MQ_QLEN];
  #pragma unroll
  for (int j = 0; j < MQ_QLEN; ++j) {
    if (j < q_len) {
      qlimit[j] = decode_mq_query_limit(seq_len, q_len, j);
      const int query_pos_j = qlimit[j] - 1;
      qmin[j] = window_size_left >= 0
                    ? max(0, query_pos_j - window_size_left)
                    : 0;
      qmax[j] = window_size_right >= 0
                    ? min(seq_len - 1, query_pos_j + window_size_right)
                    : seq_len - 1;
    } else {
      qlimit[j] = 0;
      qmin[j] = 0;
      qmax[j] = -1;
    }
  }

  float part_max[MQ_QLEN];
  #pragma unroll
  for (int j = 0; j < MQ_QLEN; ++j) {
    part_max[j] = -1.0e20f;
  }
  for (int token_local = warp_idx; token_local < part_tokens;
       token_local += kWarpsPerBlock) {
    const int token_abs = part_start + token_local;
    const int64_t k_index =
        static_cast<int64_t>(slot_shared[token_local]) * k_token_stride +
        static_cast<int64_t>(kv_head_idx) * k_head_stride;

    float acc[MQ_QLEN];
    #pragma unroll
    for (int j = 0; j < MQ_QLEN; ++j) {
      acc[j] = 0.f;
    }
    dot_qk_cache_mq<D, KV_DTYPE, MQ_QLEN>(q_shared, k_cache, k_index, lane,
                                          acc);
    #pragma unroll
    for (int j = 0; j < MQ_QLEN; ++j) {
      acc[j] = warp_reduce_sum(acc[j]);
      if (lane == 0) {
        const float score = acc[j] * score_scale;
        scores_shared[j * PARTITION_SIZE + token_local] = score;
        if (token_abs < qlimit[j] && token_abs >= qmin[j] &&
            token_abs <= qmax[j]) {
          part_max[j] = fmaxf(part_max[j], score);
        }
      }
    }
  }
  block_reduce_max_mq<kWarpsPerBlock, MQ_QLEN>(part_max);

  float part_sum[MQ_QLEN];
  #pragma unroll
  for (int j = 0; j < MQ_QLEN; ++j) {
    part_sum[j] = 0.f;
  }
  for (int i = threadIdx.x; i < part_tokens; i += blockDim.x) {
    const int token_abs = part_start + i;
    #pragma unroll
    for (int j = 0; j < MQ_QLEN; ++j) {
      const bool valid =
          token_abs < qlimit[j] && token_abs >= qmin[j] &&
          token_abs <= qmax[j];
      const float p =
          valid ? __expf(scores_shared[j * PARTITION_SIZE + i] - part_max[j])
                : 0.f;
      scores_shared[j * PARTITION_SIZE + i] = p;
      part_sum[j] += p;
    }
  }
  block_reduce_sum_mq<kWarpsPerBlock, MQ_QLEN>(part_sum);

  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float out_acc[MQ_QLEN];
    #pragma unroll
    for (int j = 0; j < MQ_QLEN; ++j) {
      out_acc[j] = 0.f;
    }
    for (int i = 0; i < part_tokens; ++i) {
      const int64_t v_index =
          static_cast<int64_t>(slot_shared[i]) * v_token_stride +
          static_cast<int64_t>(kv_head_idx) * v_head_stride + d;
      const float vv = flash_v100::load_kv_cache_float_unscaled<KV_DTYPE>(
          v_cache, v_index);
      #pragma unroll
      for (int j = 0; j < MQ_QLEN; ++j) {
        out_acc[j] =
            fmaf(scores_shared[j * PARTITION_SIZE + i], vv, out_acc[j]);
      }
    }
    #pragma unroll
    for (int j = 0; j < MQ_QLEN; ++j) {
      const float inv_part_sum = part_sum[j] > 0.f ? 1.f / part_sum[j] : 0.f;
      const float out_scale =
          KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP16
              ? inv_part_sum
              : inv_part_sum * v_scale;
      tmp_out[tmp_out_base + static_cast<int64_t>(j) * tmp_out_stride2 + d] =
          __float2half(out_acc[j] * out_scale);
    }
  }

  if (threadIdx.x == 0) {
    #pragma unroll
    for (int j = 0; j < MQ_QLEN; ++j) {
      const int64_t stats_index =
          stats_base + static_cast<int64_t>(j) * stats_stride2 + partition_idx;
      max_logits[stats_index] = part_max[j];
      exp_sums[stats_index] = part_sum[j];
    }
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
    const int* __restrict__ active_num_partitions,
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
    const float v_scale,
    const int window_size_left,
    const int window_size_right) {
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
  const int runtime_num_partitions = active_num_partitions[0];
  const int seq_num_partitions =
      (seq_len + PARTITION_SIZE - 1) / PARTITION_SIZE;
  const int effective_num_partitions =
      min(max_num_partitions, max(runtime_num_partitions, seq_num_partitions));
  if (partition_idx >= effective_num_partitions) {
    return;
  }

  const int query_pos = seq_len - 1;
  const int min_token_idx = window_size_left >= 0
                                ? max(0, query_pos - window_size_left)
                                : 0;
  const int max_token_idx = window_size_right >= 0
                                ? min(seq_len - 1, query_pos + window_size_right)
                                : seq_len - 1;
  const int part_start = max(start_token_idx, min_token_idx);
  const int part_end = min(start_token_idx + PARTITION_SIZE, max_token_idx + 1);
  const int q_per_kv = num_heads_q / num_heads_kv;
  const int kv_head_idx = head_idx / q_per_kv;
  const int lane = threadIdx.x % kWarpSize;
  const int warp_idx = threadIdx.x / kWarpSize;
  const float score_scale =
      KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP16 ? softmax_scale
                                                  : softmax_scale * k_scale;

  const int64_t tmp_out_base =
      static_cast<int64_t>(batch_idx) * tmp_out_stride0 +
      static_cast<int64_t>(head_idx) * tmp_out_stride1 +
      static_cast<int64_t>(partition_idx) * tmp_out_stride2;
  if (part_start >= part_end) {
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
      tmp_out[tmp_out_base + d] = __float2half(0.f);
    }
    if (threadIdx.x == 0) {
      const int64_t stats_index =
          static_cast<int64_t>(batch_idx) * stats_stride0 +
          static_cast<int64_t>(head_idx) * stats_stride1 + partition_idx;
      max_logits[stats_index] = -1.0e20f;
      exp_sums[stats_index] = 0.f;
    }
    return;
  }

  const int part_tokens = part_end - part_start;

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
    const int token_idx = part_start + i;
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

template<int PARTITION_SIZE, int GROUP_SIZE>
__global__ void __launch_bounds__(kXQATC256WideThreads, 1)
flash_attention_decode_xqa_tc_partition_kernel_256_wide(
    const __half* __restrict__ q,
    const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache,
    __half* __restrict__ tmp_out,
    float* __restrict__ max_logits,
    float* __restrict__ exp_sums,
    const int* __restrict__ block_table,
    const int* __restrict__ seq_lens,
    const int* __restrict__ active_num_partitions,
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
    const float softmax_scale) {
  constexpr int D = 256;
  constexpr int WMMA_M = 8;
  constexpr int WMMA_N = 32;
  constexpr int WMMA_K = 16;
  constexpr int kPanelDim = kXQATCStride;
  constexpr int kNumPanels = D / kPanelDim;
  constexpr int q_stride_uint4 = D / 8;
  constexpr int kv_stride_uint4 = kPanelDim / 8;
  constexpr int panel_d_stride_uint4 = kPanelDim / 8;
  constexpr int kAccumsPerThread = D / kWarpSize;
  static_assert(GROUP_SIZE == 4 || GROUP_SIZE == 6 || GROUP_SIZE == 8,
                "Wide D=256 TC XQA kernel supports q_per_kv in {4, 6, 8}");

  const int batch_idx = blockIdx.x;
  const int kv_head_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;

  if (batch_idx >= batch_size || kv_head_idx >= num_heads_kv ||
      partition_idx >= max_num_partitions) {
    return;
  }

  const int seq_len = seq_lens[batch_idx];
  const int start_token_idx = partition_idx * PARTITION_SIZE;
  if (seq_len <= 0 || start_token_idx >= seq_len) {
    return;
  }
  const int runtime_num_partitions = active_num_partitions[0];
  const int seq_num_partitions =
      (seq_len + PARTITION_SIZE - 1) / PARTITION_SIZE;
  const int effective_num_partitions =
      min(max_num_partitions, max(runtime_num_partitions, seq_num_partitions));
  if (partition_idx >= effective_num_partitions) {
    return;
  }

  const int q_head_base = kv_head_idx * GROUP_SIZE;
  if (q_head_base + GROUP_SIZE > num_heads_q) {
    return;
  }

  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int lane_id = tid % kWarpSize;
  const int part_tokens = min(PARTITION_SIZE, seq_len - start_token_idx);
  const int num_k_tiles = (part_tokens + kXQATCBlockN - 1) / kXQATCBlockN;
  const int* block_table_seq = block_table + batch_idx * max_num_blocks;

  extern __shared__ char smem_raw[];
  auto& smem = *reinterpret_cast<XQATCSmem256Wide*>(smem_raw);
  __half* sQ = smem.q;
  __half* sK = smem.reuse_kv.k;
  __half* sV = smem.reuse_kv.v;
  float* sS = smem.reuse_sp.s;
  __half* sP = smem.reuse_sp.p;
  float row_max_reg = kXQANegInf;
  float row_sum_reg = 0.f;
  float out_acc[kAccumsPerThread];
  #pragma unroll
  for (int i = 0; i < kAccumsPerThread; ++i) {
    out_acc[i] = 0.f;
  }

  const uint4* q_vec = reinterpret_cast<const uint4*>(q);
  uint4* sQ_vec = reinterpret_cast<uint4*>(sQ);
  for (int idx = tid; idx < GROUP_SIZE * q_stride_uint4;
       idx += kXQATC256WideThreads) {
    const int row = idx / q_stride_uint4;
    const int vec_col = idx % q_stride_uint4;
    const int64_t q_offset =
        static_cast<int64_t>(batch_idx) * q_stride0 +
        static_cast<int64_t>(q_head_base + row) * q_stride1;
    sQ_vec[row * q_stride_uint4 + vec_col] =
        __ldg(&q_vec[q_offset / 8 + vec_col]);
  }
  for (int idx = tid;
       idx < (kXQATC256WideBlockM - GROUP_SIZE) * q_stride_uint4;
       idx += kXQATC256WideThreads) {
    const int row = GROUP_SIZE + idx / q_stride_uint4;
    const int vec_col = idx % q_stride_uint4;
    sQ_vec[row * q_stride_uint4 + vec_col] = make_uint4(0, 0, 0, 0);
  }
  __syncthreads();

  volta::fragment<volta::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                  volta::row_major>
      qk_a_frag;
  volta::fragment<volta::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                  volta::col_major>
      qk_b_frag;
  volta::fragment<volta::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
      qk_acc_frag;

  for (int block_n = 0; block_n < num_k_tiles; ++block_n) {
    const int tile_token_start = start_token_idx + block_n * kXQATCBlockN;
    const int valid_k_rows =
        min(kXQATCBlockN, part_tokens - block_n * kXQATCBlockN);
    const int start_page = tile_token_start / block_size;
    const int tile_page_offset = tile_token_start - start_page * block_size;
    const int page_count =
        (tile_page_offset + valid_k_rows + block_size - 1) / block_size;

    for (int idx = tid; idx < page_count; idx += kXQATC256WideThreads) {
      smem.page_ids[idx] = __ldg(&block_table_seq[start_page + idx]);
    }
    __syncthreads();

    if (warp_id < (kXQATCBlockN / WMMA_N)) {
      volta::fill_fragment(qk_acc_frag, 0.0f);
    }
    for (int panel_idx = 0; panel_idx < kNumPanels; ++panel_idx) {
      const int panel_offset = panel_idx * kPanelDim;
      for (int idx = tid; idx < valid_k_rows * panel_d_stride_uint4;
           idx += kXQATC256WideThreads) {
        const int row = idx / panel_d_stride_uint4;
        const int vec_col = idx % panel_d_stride_uint4;
        const int token_offset = tile_page_offset + row;
        const int physical_block = smem.page_ids[token_offset / block_size];
        const int block_offset = token_offset % block_size;
        const int64_t physical_offset_halfs =
            static_cast<int64_t>(physical_block) * k_block_stride +
            static_cast<int64_t>(block_offset) * k_token_stride +
            static_cast<int64_t>(kv_head_idx) * k_head_stride +
            panel_offset;
        const uint4* k_vec = reinterpret_cast<const uint4*>(k_cache);
        reinterpret_cast<uint4*>(sK)[row * kv_stride_uint4 + vec_col] =
            __ldg(&k_vec[physical_offset_halfs / 8 + vec_col]);
      }
      for (int idx = tid + valid_k_rows * panel_d_stride_uint4;
           idx < kXQATCBlockN * panel_d_stride_uint4;
           idx += kXQATC256WideThreads) {
        reinterpret_cast<uint4*>(sK)[idx] = make_uint4(0, 0, 0, 0);
      }
      __syncthreads();

      if (warp_id < (kXQATCBlockN / WMMA_N)) {
        const int tile_n = warp_id * WMMA_N;
        #pragma unroll
        for (int k_tile = 0; k_tile < (kPanelDim / WMMA_K); ++k_tile) {
          const int k_offset = k_tile * WMMA_K;
          volta::load_matrix_sync(qk_a_frag, sQ + panel_offset + k_offset, D);
          volta::load_matrix_sync(qk_b_frag,
                                  sK + tile_n * kPanelDim + k_offset,
                                  kPanelDim);
          volta::mma_sync(qk_acc_frag, qk_a_frag, qk_b_frag, qk_acc_frag);
        }
      }
      __syncthreads();
    }

    if (warp_id < (kXQATCBlockN / WMMA_N)) {
      #pragma unroll
      for (int i = 0; i < qk_acc_frag.num_elements; ++i) {
        qk_acc_frag.x[i] *= softmax_scale;
      }
      volta::store_matrix_sync(sS + warp_id * WMMA_N, qk_acc_frag,
                               kXQATCBlockN, volta::mem_row_major);
    }
    __syncthreads();

    if (tid < GROUP_SIZE * kXQATC256WideThreadsPerRow) {
      const int row = tid / kXQATC256WideThreadsPerRow;
      const int thread_in_row = tid % kXQATC256WideThreadsPerRow;
      const unsigned mask = 0xffffffffu;
      float* sS_row_f = sS + row * kXQATCBlockN;
      __half* sP_row_h = sP + row * kXQATCBlockN;
      const int vec_cols = valid_k_rows >> 2;
      const int tail_start = vec_cols << 2;
      const int vec_col = thread_in_row;

      float thread_max = kXQANegInf;
      __half2 packed_exp0 = __float22half2_rn(make_float2(0.f, 0.f));
      __half2 packed_exp1 = __float22half2_rn(make_float2(0.f, 0.f));
      if (vec_col < vec_cols) {
        const float4 v4 = reinterpret_cast<float4*>(sS_row_f)[vec_col];
        thread_max = fmaxf(thread_max,
                           fmaxf(fmaxf(v4.x, v4.y), fmaxf(v4.z, v4.w)));
      }
      #pragma unroll
      for (int c = tail_start + thread_in_row; c < valid_k_rows;
           c += kXQATC256WideThreadsPerRow) {
        thread_max = fmaxf(thread_max, sS_row_f[c]);
      }
      #pragma unroll
      for (int o = kXQATC256WideThreadsPerRow / 2; o > 0; o >>= 1) {
        thread_max = fmaxf(
            thread_max, __shfl_down_sync(mask, thread_max, o, kWarpSize));
      }

      const float row_max = __shfl_sync(mask, thread_max, 0, kWarpSize);
      const float old_max = __shfl_sync(mask, row_max_reg, 0, kWarpSize);
      const float new_max = fmaxf(old_max, row_max);
      const float exp_diff = __expf(old_max - new_max);

      float thread_sum = 0.f;
      if (vec_col < vec_cols) {
        const float4 v4 = reinterpret_cast<float4*>(sS_row_f)[vec_col];
        const float e0 = __expf(fmaxf(v4.x - new_max, -80.0f));
        const float e1 = __expf(fmaxf(v4.y - new_max, -80.0f));
        const float e2 = __expf(fmaxf(v4.z - new_max, -80.0f));
        const float e3 = __expf(fmaxf(v4.w - new_max, -80.0f));
        thread_sum += (e0 + e1) + (e2 + e3);
        packed_exp0 = __float22half2_rn(make_float2(e0, e1));
        packed_exp1 = __float22half2_rn(make_float2(e2, e3));
      }

      #pragma unroll
      for (int c = tail_start + thread_in_row; c < kXQATCBlockN;
           c += kXQATC256WideThreadsPerRow) {
        const float v = (c < valid_k_rows) ? sS_row_f[c] : kXQANegInf;
        const float e = __expf(fmaxf(v - new_max, -80.0f));
        thread_sum += (c < valid_k_rows) ? e : 0.0f;
        sP_row_h[c] = (c < valid_k_rows) ? __float2half_rn(e)
                                         : __float2half(0.f);
      }

      #pragma unroll
      for (int o = kXQATC256WideThreadsPerRow / 2; o > 0; o >>= 1) {
        thread_sum += __shfl_down_sync(mask, thread_sum, o, kWarpSize);
      }

      const float row_sum = __shfl_sync(mask, thread_sum, 0, kWarpSize);
      const float old_sum = __shfl_sync(mask, row_sum_reg, 0, kWarpSize);

      if (thread_in_row == 0) {
        row_sum_reg = exp_diff * old_sum + row_sum;
        row_max_reg = new_max;
      }

      __half2* sP_half2 = reinterpret_cast<__half2*>(sP_row_h);
      if (vec_col < vec_cols) {
        const int base_offset = vec_col * 2;
        sP_half2[base_offset] = packed_exp0;
        sP_half2[base_offset + 1] = packed_exp1;
      }

      if (block_n > 0) {
        #pragma unroll
        for (int i = 0; i < kAccumsPerThread; ++i) {
          out_acc[i] *= exp_diff;
        }
      }
    }
    __syncthreads();

    for (int panel_idx = 0; panel_idx < kNumPanels; ++panel_idx) {
      const int panel_offset = panel_idx * kPanelDim;
      for (int idx = tid; idx < valid_k_rows * panel_d_stride_uint4;
           idx += kXQATC256WideThreads) {
        const int row = idx / panel_d_stride_uint4;
        const int vec_col = idx % panel_d_stride_uint4;
        const int token_offset = tile_page_offset + row;
        const int physical_block = smem.page_ids[token_offset / block_size];
        const int block_offset = token_offset % block_size;
        const int64_t physical_offset_halfs =
            static_cast<int64_t>(physical_block) * v_block_stride +
            static_cast<int64_t>(block_offset) * v_token_stride +
            static_cast<int64_t>(kv_head_idx) * v_head_stride +
            panel_offset;
        const uint4* v_vec = reinterpret_cast<const uint4*>(v_cache);
        reinterpret_cast<uint4*>(sV)[row * kv_stride_uint4 + vec_col] =
            __ldg(&v_vec[physical_offset_halfs / 8 + vec_col]);
      }
      for (int idx = tid + valid_k_rows * panel_d_stride_uint4;
           idx < kXQATCBlockN * panel_d_stride_uint4;
           idx += kXQATC256WideThreads) {
        reinterpret_cast<uint4*>(sV)[idx] = make_uint4(0, 0, 0, 0);
      }
      __syncthreads();

      if (tid < GROUP_SIZE * kXQATC256WideThreadsPerRow) {
        const int row = tid / kXQATC256WideThreadsPerRow;
        const __half* sP_row = sP + row * kXQATCBlockN;
        #pragma unroll
        for (int token = 0; token < kXQATCBlockN; ++token) {
          if (token >= valid_k_rows) {
            break;
          }
          const float prob = __half2float(sP_row[token]);
          const __half* sV_row = sV + token * kPanelDim;
          #pragma unroll
          for (int d_iter = 0; d_iter < (kPanelDim / kWarpSize); ++d_iter) {
            const int local_d = lane_id + d_iter * kWarpSize;
            const int acc_idx = panel_idx * (kPanelDim / kWarpSize) + d_iter;
            out_acc[acc_idx] =
                fmaf(prob, __half2float(sV_row[local_d]), out_acc[acc_idx]);
          }
        }
      }
      __syncthreads();
    }
  }

  if (tid < GROUP_SIZE * kXQATC256WideThreadsPerRow) {
    const int row = tid / kXQATC256WideThreadsPerRow;
    const int thread_in_row = tid % kXQATC256WideThreadsPerRow;
    if (thread_in_row == 0) {
      smem.row_max[row] = row_max_reg;
      smem.row_sum[row] = row_sum_reg;
    }
  }
  __syncthreads();

  if (tid < GROUP_SIZE * kXQATC256WideThreadsPerRow) {
    const int row = tid / kXQATC256WideThreadsPerRow;
    const int thread_in_row = tid % kXQATC256WideThreadsPerRow;
    const int head_idx = q_head_base + row;
    const float row_sum = smem.row_sum[row];
    const float inv_row_sum = row_sum > 0.f ? 1.f / row_sum : 0.f;
    __half* tmp_out_ptr =
        tmp_out + static_cast<int64_t>(batch_idx) * tmp_out_stride0 +
        static_cast<int64_t>(head_idx) * tmp_out_stride1 +
        static_cast<int64_t>(partition_idx) * tmp_out_stride2;
    for (int d = thread_in_row; d < D; d += kXQATC256WideThreadsPerRow) {
      tmp_out_ptr[d] = __float2half(out_acc[d / kWarpSize] * inv_row_sum);
    }
    if (thread_in_row == 0) {
      const int64_t stats_index =
          static_cast<int64_t>(batch_idx) * stats_stride0 +
          static_cast<int64_t>(head_idx) * stats_stride1 + partition_idx;
      max_logits[stats_index] = smem.row_max[row];
      exp_sums[stats_index] = row_sum;
    }
  }
}

template<int D, int PARTITION_SIZE>
__global__ void flash_attention_decode_reduce_kernel(
    const __half* __restrict__ tmp_out,
    const float* __restrict__ max_logits,
    const float* __restrict__ exp_sums,
    const int* __restrict__ seq_lens,
    const int* __restrict__ active_num_partitions,
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
      min(max_num_partitions,
          (seq_len + PARTITION_SIZE - 1) / PARTITION_SIZE);
  (void)active_num_partitions;

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

template<int D, int PARTITION_SIZE>
__global__ void flash_attention_decode_reduce_mq_kernel(
    const __half* __restrict__ tmp_out,
    const float* __restrict__ max_logits,
    const float* __restrict__ exp_sums,
    const int* __restrict__ seq_lens,
    const int* __restrict__ query_start_loc,
    const int* __restrict__ active_num_partitions,
    __half* __restrict__ out,
    const int num_reqs,
    const int max_num_partitions,
    const int num_heads_q,
    const int64_t tmp_out_stride0,
    const int64_t tmp_out_stride1,
    const int64_t tmp_out_stride2,
    const int64_t tmp_out_stride3,
    const int64_t stats_stride0,
    const int64_t stats_stride1,
    const int64_t stats_stride2,
    const int64_t out_stride0,
    const int64_t out_stride1) {
  const int req_idx = blockIdx.x;
  const int head_idx = blockIdx.y;

  if (req_idx >= num_reqs || head_idx >= num_heads_q) {
    return;
  }

  const int seq_len = seq_lens[req_idx];
  const int q_start = query_start_loc[req_idx];
  const int q_len =
      min(query_start_loc[req_idx + 1] - q_start, kDecodeMqQMax);
  if (q_len <= 0) {
    return;
  }
  __half* out_ptr = out + static_cast<int64_t>(q_start) * out_stride0 +
                    static_cast<int64_t>(head_idx) * out_stride1;
  if (seq_len <= 0) {
    for (int j = 0; j < q_len; ++j) {
      for (int d = threadIdx.x; d < D; d += blockDim.x) {
        out_ptr[static_cast<int64_t>(j) * out_stride0 + d] =
            __float2half(0.f);
      }
    }
    return;
  }
  const int num_partitions =
      min(max_num_partitions,
          (seq_len + PARTITION_SIZE - 1) / PARTITION_SIZE);
  (void)active_num_partitions;

  extern __shared__ float shared_mem[];
  float* max_shared = shared_mem;
  float* weight_shared = shared_mem + max_num_partitions;

  const int64_t tmp_base_bh =
      static_cast<int64_t>(req_idx) * tmp_out_stride0 +
      static_cast<int64_t>(head_idx) * tmp_out_stride1;
  const int64_t stats_base_bh =
      static_cast<int64_t>(req_idx) * stats_stride0 +
      static_cast<int64_t>(head_idx) * stats_stride1;

  for (int j = 0; j < q_len; ++j) {
    const int64_t stats_base =
        stats_base_bh + static_cast<int64_t>(j) * stats_stride2;
    float local_max = -1.0e20f;
    for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
      const float m = max_logits[stats_base + i];
      max_shared[i] = m;
      local_max = fmaxf(local_max, m);
    }
    const float global_max = block_reduce_max<kWarpsPerBlock>(local_max);

    float local_sum = 0.f;
    for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
      const float weight =
          exp_sums[stats_base + i] * __expf(max_shared[i] - global_max);
      weight_shared[i] = weight;
      local_sum += weight;
    }
    const float global_sum = block_reduce_sum<kWarpsPerBlock>(local_sum);
    const float inv_global_sum = global_sum > 0.f ? 1.f / global_sum : 0.f;
    __syncthreads();

    const int64_t tmp_base =
        tmp_base_bh + static_cast<int64_t>(j) * tmp_out_stride2;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
      float acc = 0.f;
      for (int i = 0; i < num_partitions; ++i) {
        acc = fmaf(
            weight_shared[i],
            __half2float(tmp_out[tmp_base +
                                 static_cast<int64_t>(i) * tmp_out_stride3 +
                                 d]),
            acc);
      }
      out_ptr[static_cast<int64_t>(j) * out_stride0 + d] =
          __float2half(acc * inv_global_sum);
    }
    __syncthreads();
  }
}

template<int D, int PARTITION_SIZE, int KV_DTYPE>
__global__ void flash_attention_decode_qk_scores_kernel(
    const __half* __restrict__ q,
    const void* __restrict__ k_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ seq_lens,
    float* __restrict__ scores,
    const int batch_size,
    const int max_num_blocks,
    const int max_num_partitions,
    const int num_heads_q,
    const int num_heads_kv,
    const int block_size,
    const int64_t q_stride0,
    const int64_t q_stride1,
    const int64_t scores_stride0,
    const int64_t scores_stride1,
    const int64_t scores_stride2,
    const int64_t k_block_stride,
    const int64_t k_token_stride,
    const int64_t k_head_stride,
    const float softmax_scale,
    const float k_scale) {
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

  const int64_t score_base =
      static_cast<int64_t>(batch_idx) * scores_stride0 +
      static_cast<int64_t>(head_idx) * scores_stride1 +
      static_cast<int64_t>(partition_idx) * scores_stride2;
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
      scores[score_base + token_local] = score * score_scale;
    }
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
    const at::Tensor& active_num_partitions,
    const float softmax_scale,
    const int launch_num_partitions,
    const float k_scale,
    const float v_scale,
    const int window_size_left,
    const int window_size_right,
    cudaStream_t stream) {
  const int batch_size = q.size(0);
  const int num_heads_q = q.size(1);
  const int num_heads_kv = k_cache.size(2);
  const int block_size = k_cache.size(1);
  const int max_num_blocks = block_table.size(1);
  const int max_num_partitions = launch_num_partitions;

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
      active_num_partitions.data_ptr<int>(),
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
      v_scale,
      window_size_left,
      window_size_right);

  flash_attention_decode_reduce_kernel<D, PARTITION_SIZE><<<reduce_grid, block, reduce_shared_mem, stream>>>(
      reinterpret_cast<const __half*>(tmp_out.data_ptr<at::Half>()),
      max_logits.data_ptr<float>(),
      exp_sums.data_ptr<float>(),
      seq_lens.data_ptr<int>(),
      active_num_partitions.data_ptr<int>(),
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

template<int D, int PARTITION_SIZE, int KV_DTYPE, int MQ_QLEN>
void launch_flash_attention_decode_paged_mq(
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    at::Tensor& out,
    const at::Tensor& block_table,
    const at::Tensor& seq_lens,
    const at::Tensor& query_start_loc,
    at::Tensor& tmp_out,
    at::Tensor& max_logits,
    at::Tensor& exp_sums,
    const at::Tensor& active_num_partitions,
    const float softmax_scale,
    const int launch_num_partitions,
    const float k_scale,
    const float v_scale,
    const int window_size_left,
    const int window_size_right,
    cudaStream_t stream) {
  const int num_reqs = query_start_loc.size(0) - 1;
  const int num_heads_q = q.size(1);
  const int num_heads_kv = k_cache.size(2);
  const int block_size = k_cache.size(1);
  const int max_num_blocks = block_table.size(1);
  const int max_num_partitions = launch_num_partitions;

  const dim3 partition_grid(num_reqs, num_heads_q, max_num_partitions);
  const dim3 reduce_grid(num_reqs, num_heads_q, 1);
  const dim3 block(kThreadsPerBlock);
  const size_t reduce_shared_mem =
      static_cast<size_t>(2 * max_num_partitions) * sizeof(float);

  flash_attention_decode_partition_mq_kernel<D, PARTITION_SIZE, KV_DTYPE,
                                             MQ_QLEN>
      <<<partition_grid, block, 0, stream>>>(
      reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
      k_cache.data_ptr(),
      v_cache.data_ptr(),
      reinterpret_cast<__half*>(tmp_out.data_ptr<at::Half>()),
      max_logits.data_ptr<float>(),
      exp_sums.data_ptr<float>(),
      block_table.data_ptr<int>(),
      seq_lens.data_ptr<int>(),
      query_start_loc.data_ptr<int>(),
      active_num_partitions.data_ptr<int>(),
      num_reqs,
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
      tmp_out.stride(3),
      max_logits.stride(0),
      max_logits.stride(1),
      max_logits.stride(2),
      k_cache.stride(0),
      k_cache.stride(1),
      k_cache.stride(2),
      v_cache.stride(0),
      v_cache.stride(1),
      v_cache.stride(2),
      softmax_scale,
      k_scale,
      v_scale,
      window_size_left,
      window_size_right);

  flash_attention_decode_reduce_mq_kernel<D, PARTITION_SIZE>
      <<<reduce_grid, block, reduce_shared_mem, stream>>>(
      reinterpret_cast<const __half*>(tmp_out.data_ptr<at::Half>()),
      max_logits.data_ptr<float>(),
      exp_sums.data_ptr<float>(),
      seq_lens.data_ptr<int>(),
      query_start_loc.data_ptr<int>(),
      active_num_partitions.data_ptr<int>(),
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
      num_reqs,
      max_num_partitions,
      num_heads_q,
      tmp_out.stride(0),
      tmp_out.stride(1),
      tmp_out.stride(2),
      tmp_out.stride(3),
      max_logits.stride(0),
      max_logits.stride(1),
      max_logits.stride(2),
      out.stride(0),
      out.stride(1));
}

template<int PARTITION_SIZE, int GROUP_SIZE>
void launch_flash_attention_decode_paged_xqa_tc_256_wide(
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    at::Tensor& out,
    const at::Tensor& block_table,
    const at::Tensor& seq_lens,
    at::Tensor& tmp_out,
    at::Tensor& max_logits,
    at::Tensor& exp_sums,
    const at::Tensor& active_num_partitions,
    const float softmax_scale,
    const int launch_num_partitions,
    cudaStream_t stream) {
  const int batch_size = q.size(0);
  const int num_heads_q = q.size(1);
  const int num_heads_kv = k_cache.size(2);
  const int max_num_blocks = block_table.size(1);
  const dim3 partition_grid(batch_size, num_heads_kv, launch_num_partitions);
  const size_t shared_mem = sizeof(XQATCSmem256Wide);
  auto partition_kernel =
      (void*)flash_attention_decode_xqa_tc_partition_kernel_256_wide<
          PARTITION_SIZE, GROUP_SIZE>;

  cudaFuncSetAttribute(partition_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       shared_mem);

  flash_attention_decode_xqa_tc_partition_kernel_256_wide<
      PARTITION_SIZE, GROUP_SIZE>
      <<<partition_grid, kXQATC256WideThreads, shared_mem, stream>>>(
          reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
          reinterpret_cast<const __half*>(k_cache.data_ptr<at::Half>()),
          reinterpret_cast<const __half*>(v_cache.data_ptr<at::Half>()),
          reinterpret_cast<__half*>(tmp_out.data_ptr<at::Half>()),
          max_logits.data_ptr<float>(),
          exp_sums.data_ptr<float>(),
          block_table.data_ptr<int>(),
          seq_lens.data_ptr<int>(),
          active_num_partitions.data_ptr<int>(),
          batch_size,
          max_num_blocks,
          launch_num_partitions,
          num_heads_q,
          num_heads_kv,
          k_cache.size(1),
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
          softmax_scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const dim3 reduce_grid(batch_size, num_heads_q, 1);
  const dim3 block(kThreadsPerBlock);
  const size_t reduce_shared_mem =
      static_cast<size_t>(2 * launch_num_partitions) * sizeof(float);
  flash_attention_decode_reduce_kernel<256, PARTITION_SIZE>
      <<<reduce_grid, block, reduce_shared_mem, stream>>>(
          reinterpret_cast<const __half*>(tmp_out.data_ptr<at::Half>()),
          max_logits.data_ptr<float>(),
          exp_sums.data_ptr<float>(),
          seq_lens.data_ptr<int>(),
          active_num_partitions.data_ptr<int>(),
          reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
          batch_size,
          launch_num_partitions,
          num_heads_q,
          tmp_out.stride(0),
          tmp_out.stride(1),
          tmp_out.stride(2),
          max_logits.stride(0),
          max_logits.stride(1),
          out.stride(0),
          out.stride(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int D, int PARTITION_SIZE, int KV_DTYPE>
void launch_flash_attention_decode_qk_scores(
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& block_table,
    const at::Tensor& seq_lens,
    at::Tensor& scores,
    const float softmax_scale,
    const float k_scale,
    cudaStream_t stream) {
  const int batch_size = q.size(0);
  const int num_heads_q = q.size(1);
  const int num_heads_kv = k_cache.size(2);
  const int block_size = k_cache.size(1);
  const int max_num_blocks = block_table.size(1);
  const int max_num_partitions = scores.size(2);

  const dim3 grid(batch_size, num_heads_q, max_num_partitions);
  const dim3 block(kThreadsPerBlock);

  flash_attention_decode_qk_scores_kernel<D, PARTITION_SIZE, KV_DTYPE>
      <<<grid, block, 0, stream>>>(
      reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
      k_cache.data_ptr(),
      block_table.data_ptr<int>(),
      seq_lens.data_ptr<int>(),
      scores.data_ptr<float>(),
      batch_size,
      max_num_blocks,
      max_num_partitions,
      num_heads_q,
      num_heads_kv,
      block_size,
      q.stride(0),
      q.stride(1),
      scores.stride(0),
      scores.stride(1),
      scores.stride(2),
      k_cache.stride(0),
      k_cache.stride(1),
      k_cache.stride(2),
      softmax_scale,
      k_scale);
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
    const at::Tensor& active_num_partitions,
    const float softmax_scale,
    const int partition_size,
    const int launch_num_partitions,
    const std::string& kv_cache_dtype,
    const float k_scale,
    const float v_scale,
    const int window_size_left,
    const int window_size_right) {
  TORCH_CHECK(q.is_cuda(), "q must be on CUDA");
  TORCH_CHECK(k_cache.is_cuda() && v_cache.is_cuda(), "k/v cache must be on CUDA");
  TORCH_CHECK(block_table.is_cuda() && seq_lens.is_cuda(), "block_table and seq_lens must be on CUDA");
  TORCH_CHECK(tmp_out.is_cuda() && max_logits.is_cuda() && exp_sums.is_cuda(),
              "workspace tensors must be on CUDA");
  TORCH_CHECK(active_num_partitions.is_cuda(),
              "active_num_partitions must be on CUDA");
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
  TORCH_CHECK(active_num_partitions.dtype() == torch::kInt32,
              "active_num_partitions must be int32");
  TORCH_CHECK(q.dim() == 3, "q must have shape [B, H, D]");
  TORCH_CHECK(k_cache.dim() == 4, "k_cache must have shape [num_blocks, block_size, H_kv, D]");
  TORCH_CHECK(v_cache.dim() == 4, "v_cache must have shape [num_blocks, block_size, H_kv, D]");
  TORCH_CHECK(block_table.dim() == 2, "block_table must have shape [B, max_num_blocks]");
  TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must have shape [B]");
  TORCH_CHECK(tmp_out.dim() == 4, "tmp_out must have shape [B_cap, H, P, D]");
  TORCH_CHECK(max_logits.dim() == 3, "max_logits must have shape [B_cap, H, P]");
  TORCH_CHECK(exp_sums.dim() == 3, "exp_sums must have shape [B_cap, H, P]");
  TORCH_CHECK(active_num_partitions.dim() == 1 &&
                  active_num_partitions.numel() == 1,
              "active_num_partitions must have shape [1]");
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
  TORCH_CHECK(launch_num_partitions > 0 &&
                  launch_num_partitions <= tmp_out.size(2),
              "launch_num_partitions must be in (0, tmp_out.size(2)]");
  TORCH_CHECK(window_size_left >= -1 && window_size_right >= -1,
              "window sizes must be >= -1");

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
        exp_sums, active_num_partitions, softmax_scale,                         \
        launch_num_partitions, k_scale, v_scale, window_size_left,              \
        window_size_right, stream)

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
    case 512:
      LAUNCH_BY_PARTITION(512);
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

at::Tensor flash_attention_decode_paged_mq(
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    std::optional<at::Tensor>& out_,
    const at::Tensor& block_table,
    const at::Tensor& seq_lens,
    const at::Tensor& query_start_loc,
    at::Tensor& tmp_out,
    at::Tensor& max_logits,
    at::Tensor& exp_sums,
    const at::Tensor& active_num_partitions,
    const float softmax_scale,
    const int partition_size,
    const int launch_num_partitions,
    const int q_len_max,
    const std::string& kv_cache_dtype,
    const float k_scale,
    const float v_scale,
    const int window_size_left,
    const int window_size_right) {
  TORCH_CHECK(q.is_cuda(), "q must be on CUDA");
  TORCH_CHECK(k_cache.is_cuda() && v_cache.is_cuda(), "k/v cache must be on CUDA");
  TORCH_CHECK(block_table.is_cuda() && seq_lens.is_cuda(),
              "block_table and seq_lens must be on CUDA");
  TORCH_CHECK(query_start_loc.is_cuda(), "query_start_loc must be on CUDA");
  TORCH_CHECK(tmp_out.is_cuda() && max_logits.is_cuda() && exp_sums.is_cuda(),
              "workspace tensors must be on CUDA");
  TORCH_CHECK(active_num_partitions.is_cuda(),
              "active_num_partitions must be on CUDA");
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
  TORCH_CHECK(query_start_loc.dtype() == torch::kInt32,
              "query_start_loc must be int32");
  TORCH_CHECK(active_num_partitions.dtype() == torch::kInt32,
              "active_num_partitions must be int32");
  TORCH_CHECK(q.dim() == 3, "q must have shape [total_tokens, H, D]");
  TORCH_CHECK(k_cache.dim() == 4, "k_cache must have shape [num_blocks, block_size, H_kv, D]");
  TORCH_CHECK(v_cache.dim() == 4, "v_cache must have shape [num_blocks, block_size, H_kv, D]");
  TORCH_CHECK(block_table.dim() == 2, "block_table must have shape [B, max_num_blocks]");
  TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must have shape [B]");
  TORCH_CHECK(query_start_loc.dim() == 1 && query_start_loc.size(0) >= 2,
              "query_start_loc must have shape [B+1]");
  TORCH_CHECK(tmp_out.dim() == 5,
              "tmp_out must have shape [B, H, q_len_max, P, D]");
  TORCH_CHECK(max_logits.dim() == 4,
              "max_logits must have shape [B, H, q_len_max, P]");
  TORCH_CHECK(exp_sums.dim() == 4,
              "exp_sums must have shape [B, H, q_len_max, P]");
  TORCH_CHECK(active_num_partitions.dim() == 1 &&
                  active_num_partitions.numel() == 1,
              "active_num_partitions must have shape [1]");
  TORCH_CHECK(q.stride(-1) == 1, "q last dim must be contiguous");
  TORCH_CHECK(k_cache.stride(-1) == 1, "k_cache last dim must be contiguous");
  TORCH_CHECK(v_cache.stride(-1) == 1, "v_cache last dim must be contiguous");
  TORCH_CHECK(tmp_out.stride(-1) == 1, "tmp_out last dim must be contiguous");
  TORCH_CHECK(k_cache.stride(1) > 0 &&
                  k_cache.stride(0) % k_cache.stride(1) == 0,
              "multi-query decode requires the K block stride to be a "
              "multiple of the token stride");
  TORCH_CHECK(v_cache.stride(1) > 0 &&
                  v_cache.stride(0) % v_cache.stride(1) == 0,
              "multi-query decode requires the V block stride to be a "
              "multiple of the token stride");
  TORCH_CHECK(k_cache.stride(0) / k_cache.stride(1) ==
                  v_cache.stride(0) / v_cache.stride(1),
              "multi-query decode requires matching K/V block-to-token "
              "stride ratios");
  TORCH_CHECK(
      k_cache.size(0) * (k_cache.stride(0) / k_cache.stride(1)) < (1LL << 31),
      "multi-query decode KV slot overflow");

  const int num_reqs = query_start_loc.size(0) - 1;
  const int num_heads_q = q.size(1);
  const int head_dim = q.size(2);
  const int num_heads_kv = k_cache.size(2);

  TORCH_CHECK(num_reqs <= block_table.size(0),
              "block_table batch size must cover num_reqs");
  TORCH_CHECK(num_reqs <= seq_lens.size(0),
              "seq_lens batch size must cover num_reqs");
  TORCH_CHECK(num_reqs <= tmp_out.size(0),
              "tmp_out batch size must cover num_reqs");
  TORCH_CHECK(num_heads_q == tmp_out.size(1), "tmp_out head dimension mismatch");
  TORCH_CHECK(q_len_max >= 1 && q_len_max <= kDecodeMqQMax,
              "q_len_max must be in [1, ", kDecodeMqQMax, "], got ", q_len_max);
  TORCH_CHECK(tmp_out.size(2) >= q_len_max,
              "tmp_out q_len dimension must cover q_len_max");
  TORCH_CHECK(head_dim == tmp_out.size(4), "tmp_out head_dim mismatch");
  TORCH_CHECK(max_logits.size(0) == tmp_out.size(0) &&
                  max_logits.size(1) == tmp_out.size(1) &&
                  max_logits.size(2) == tmp_out.size(2) &&
                  max_logits.size(3) == tmp_out.size(3),
              "max_logits shape mismatch");
  TORCH_CHECK(exp_sums.sizes() == max_logits.sizes(),
              "exp_sums shape mismatch");
  TORCH_CHECK(num_heads_q % num_heads_kv == 0,
              "num_heads_q must be divisible by num_heads_kv");
  TORCH_CHECK(k_cache.size(3) == head_dim, "k_cache head_dim mismatch");
  TORCH_CHECK(v_cache.size(3) == head_dim, "v_cache head_dim mismatch");
  TORCH_CHECK(partition_size == 256 || partition_size == 512 || partition_size == 1024,
              "Unsupported decode partition_size: ", partition_size);
  TORCH_CHECK(launch_num_partitions > 0 &&
                  launch_num_partitions <= tmp_out.size(3),
              "launch_num_partitions must be in (0, tmp_out.size(3)]");
  TORCH_CHECK(window_size_left >= -1 && window_size_right >= -1,
              "window sizes must be >= -1");

  at::Tensor out = out_.has_value() ? out_.value() : torch::empty_like(q);
  TORCH_CHECK(out.is_cuda(), "out must be on CUDA");
  TORCH_CHECK(out.dtype() == torch::kFloat16, "out must be fp16");
  TORCH_CHECK(out.sizes() == q.sizes(), "out must have same shape as q");
  TORCH_CHECK(out.stride(-1) == 1, "out last dim must be contiguous");

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  c10::cuda::CUDAGuard device_guard(q.device());

  #define LAUNCH_MQ_TYPED(HDIM, PARTITION, KV_DTYPE_CODE, QLEN)                 \
    launch_flash_attention_decode_paged_mq<HDIM, PARTITION, KV_DTYPE_CODE,      \
                                           QLEN>(                               \
        q, k_cache, v_cache, out, block_table, seq_lens, query_start_loc,       \
        tmp_out, max_logits, exp_sums, active_num_partitions, softmax_scale,    \
        launch_num_partitions, k_scale, v_scale, window_size_left,              \
        window_size_right, stream)

  #define LAUNCH_MQ_BY_KV_DTYPE(HDIM, PARTITION, QLEN)                          \
    do {                                                                        \
      switch (kv_dtype_code) {                                                  \
        case flash_v100::KV_CACHE_DTYPE_FP16:                                   \
          LAUNCH_MQ_TYPED(HDIM, PARTITION, flash_v100::KV_CACHE_DTYPE_FP16,     \
                          QLEN);                                                \
          break;                                                                \
        case flash_v100::KV_CACHE_DTYPE_FP8_E4M3:                               \
          LAUNCH_MQ_TYPED(HDIM, PARTITION, flash_v100::KV_CACHE_DTYPE_FP8_E4M3, \
                          QLEN);                                                \
          break;                                                                \
        case flash_v100::KV_CACHE_DTYPE_FP8_E5M2:                               \
          LAUNCH_MQ_TYPED(HDIM, PARTITION, flash_v100::KV_CACHE_DTYPE_FP8_E5M2, \
                          QLEN);                                                \
          break;                                                                \
        default:                                                                \
          TORCH_CHECK(false, "Unsupported kv_cache_dtype: ", kv_cache_dtype);  \
      }                                                                         \
    } while (0)

  #define LAUNCH_MQ_BY_PARTITION(HDIM, QLEN)                                    \
    do {                                                                        \
      switch (partition_size) {                                                 \
        case 256:                                                               \
          LAUNCH_MQ_BY_KV_DTYPE(HDIM, 256, QLEN);                               \
          break;                                                                \
        case 512:                                                               \
          LAUNCH_MQ_BY_KV_DTYPE(HDIM, 512, QLEN);                               \
          break;                                                                \
        case 1024:                                                              \
          LAUNCH_MQ_BY_KV_DTYPE(HDIM, 1024, QLEN);                              \
          break;                                                                \
        default:                                                                \
          TORCH_CHECK(false, "Unsupported decode partition_size: ",             \
                      partition_size);                                          \
      }                                                                         \
    } while (0)

  #define LAUNCH_MQ_BY_PARTITION_FP16_ONLY(HDIM, QLEN)                          \
    do {                                                                        \
      switch (partition_size) {                                                 \
        case 256:                                                               \
          LAUNCH_MQ_TYPED(HDIM, 256, flash_v100::KV_CACHE_DTYPE_FP16, QLEN);   \
          break;                                                                \
        case 512:                                                               \
          LAUNCH_MQ_TYPED(HDIM, 512, flash_v100::KV_CACHE_DTYPE_FP16, QLEN);   \
          break;                                                                \
        case 1024:                                                              \
          LAUNCH_MQ_TYPED(HDIM, 1024, flash_v100::KV_CACHE_DTYPE_FP16, QLEN);  \
          break;                                                                \
        default:                                                                \
          TORCH_CHECK(false, "Unsupported decode partition_size: ",             \
                      partition_size);                                          \
      }                                                                         \
    } while (0)

  TORCH_CHECK(head_dim == 256 || head_dim == 512,
              "multi-query paged decode supports head_dim in {256, 512}, got ",
              head_dim);
  if (head_dim == 512) {
    TORCH_CHECK(kv_dtype_code == flash_v100::KV_CACHE_DTYPE_FP16,
                "multi-query paged decode supports fp16 KV only at "
                "head_dim=512");
    if (q_len_max <= 1) {
      LAUNCH_MQ_BY_PARTITION_FP16_ONLY(512, 1);
    } else if (q_len_max <= 2) {
      LAUNCH_MQ_BY_PARTITION_FP16_ONLY(512, 2);
    } else if (q_len_max <= 4) {
      LAUNCH_MQ_BY_PARTITION_FP16_ONLY(512, 4);
    } else {
      LAUNCH_MQ_BY_PARTITION_FP16_ONLY(512, 8);
    }
  } else if (q_len_max <= 1) {
    LAUNCH_MQ_BY_PARTITION(256, 1);
  } else if (q_len_max <= 2) {
    LAUNCH_MQ_BY_PARTITION(256, 2);
  } else if (q_len_max <= 4) {
    LAUNCH_MQ_BY_PARTITION(256, 4);
  } else {
    LAUNCH_MQ_BY_PARTITION(256, 8);
  }

  #undef LAUNCH_MQ_BY_PARTITION_FP16_ONLY
  #undef LAUNCH_MQ_BY_PARTITION
  #undef LAUNCH_MQ_BY_KV_DTYPE
  #undef LAUNCH_MQ_TYPED

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor flash_attention_decode_paged_xqa(
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    std::optional<at::Tensor>& out_,
    const at::Tensor& block_table,
    const at::Tensor& seq_lens,
    at::Tensor& tmp_out,
    at::Tensor& max_logits,
    at::Tensor& exp_sums,
    const at::Tensor& active_num_partitions,
    const float softmax_scale,
    const int partition_size,
    const int launch_num_partitions,
    const std::string& kv_cache_dtype,
    const float k_scale,
    const float v_scale,
    const int window_size_left,
    const int window_size_right) {
  (void)k_scale;
  (void)v_scale;
  TORCH_CHECK(q.is_cuda(), "q must be on CUDA");
  TORCH_CHECK(k_cache.is_cuda() && v_cache.is_cuda(),
              "k_cache and v_cache must be on CUDA");
  TORCH_CHECK(block_table.is_cuda() && seq_lens.is_cuda(),
              "block_table and seq_lens must be on CUDA");
  TORCH_CHECK(tmp_out.is_cuda() && max_logits.is_cuda() && exp_sums.is_cuda(),
              "decode workspaces must be on CUDA");
  TORCH_CHECK(active_num_partitions.is_cuda(),
              "active_num_partitions must be on CUDA");
  TORCH_CHECK(q.dtype() == torch::kFloat16, "q must be fp16");
  TORCH_CHECK(k_cache.dtype() == torch::kFloat16 &&
                  v_cache.dtype() == torch::kFloat16,
              "XQA decode currently supports fp16 KV cache only");
  TORCH_CHECK(kv_cache_dtype == "auto" || kv_cache_dtype == "bfloat16",
              "XQA decode currently supports fp16 KV cache only");
  TORCH_CHECK(block_table.dtype() == torch::kInt32,
              "block_table must be int32");
  TORCH_CHECK(seq_lens.dtype() == torch::kInt32, "seq_lens must be int32");
  TORCH_CHECK(active_num_partitions.dtype() == torch::kInt32,
              "active_num_partitions must be int32");
  TORCH_CHECK(window_size_left == -1 && window_size_right == -1,
              "XQA decode does not support sliding-window attention");
  TORCH_CHECK(q.dim() == 3, "q must have shape [B, H, D]");
  TORCH_CHECK(k_cache.dim() == 4 && v_cache.dim() == 4,
              "KV cache must have shape [num_blocks, block_size, H_kv, D]");
  TORCH_CHECK(block_table.dim() == 2,
              "block_table must have shape [B, max_num_blocks]");
  TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must have shape [B]");
  TORCH_CHECK(active_num_partitions.dim() == 1 &&
                  active_num_partitions.numel() == 1,
              "active_num_partitions must have shape [1]");
  TORCH_CHECK(q.stride(-1) == 1, "q last dim must be contiguous");
  TORCH_CHECK(k_cache.stride(-1) == 1 && v_cache.stride(-1) == 1,
              "KV cache last dim must be contiguous");
  TORCH_CHECK(q.size(0) <= block_table.size(0),
              "block_table batch size must cover q batch size");
  TORCH_CHECK(q.size(0) <= seq_lens.size(0),
              "seq_lens batch size must cover q batch size");
  TORCH_CHECK(k_cache.sizes() == v_cache.sizes(), "K/V cache shape mismatch");
  TORCH_CHECK(k_cache.size(3) == q.size(2), "KV head_dim mismatch");
  TORCH_CHECK(q.size(2) == 256, "XQA decode supports head_dim=256 only");
  const int num_heads_q = q.size(1);
  const int num_heads_kv = k_cache.size(2);
  TORCH_CHECK(num_heads_kv > 0 && num_heads_q % num_heads_kv == 0,
              "num_heads_q must be divisible by num_heads_kv");
  const int q_per_kv = num_heads_q / num_heads_kv;
  TORCH_CHECK(q_per_kv == 4 || q_per_kv == 6 || q_per_kv == 8,
              "XQA decode supports q_per_kv in {4, 6, 8}, got ", q_per_kv);
  TORCH_CHECK(partition_size == 256 || partition_size == 512 ||
                  partition_size == 1024,
              "Unsupported XQA decode partition_size: ", partition_size);
  TORCH_CHECK(launch_num_partitions > 0,
              "launch_num_partitions must be positive");
  TORCH_CHECK(tmp_out.dtype() == torch::kFloat16,
              "XQA decode tmp_out must be fp16");
  TORCH_CHECK(tmp_out.size(0) >= q.size(0) && tmp_out.size(1) >= q.size(1) &&
                  tmp_out.size(2) >= launch_num_partitions &&
                  tmp_out.size(3) == q.size(2),
              "tmp_out shape does not cover XQA launch");
  TORCH_CHECK(max_logits.size(0) >= q.size(0) &&
                  max_logits.size(1) >= q.size(1) &&
                  max_logits.size(2) >= launch_num_partitions,
              "max_logits shape does not cover XQA launch");
  TORCH_CHECK(exp_sums.size(0) >= q.size(0) &&
                  exp_sums.size(1) >= q.size(1) &&
                  exp_sums.size(2) >= launch_num_partitions,
              "exp_sums shape does not cover XQA launch");

  c10::cuda::CUDAGuard device_guard(q.device());
  at::Tensor out = out_.has_value() ? out_.value() : torch::empty_like(q);
  TORCH_CHECK(out.is_cuda(), "out must be on CUDA");
  TORCH_CHECK(out.dtype() == torch::kFloat16, "out must be fp16");
  TORCH_CHECK(out.sizes() == q.sizes(), "out must have same shape as q");
  TORCH_CHECK(out.stride(-1) == 1, "out last dim must be contiguous");
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  #define LAUNCH_XQA_WIDE(GROUP_SIZE, PARTITION)                               \
    launch_flash_attention_decode_paged_xqa_tc_256_wide<PARTITION, GROUP_SIZE>(\
        q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,  \
        exp_sums, active_num_partitions, softmax_scale, launch_num_partitions, \
        stream)

  #define DISPATCH_PARTITION(GROUP_SIZE)                                       \
    do {                                                                       \
      switch (partition_size) {                                                \
        case 256:                                                              \
          LAUNCH_XQA_WIDE(GROUP_SIZE, 256);                                    \
          break;                                                               \
        case 512:                                                              \
          LAUNCH_XQA_WIDE(GROUP_SIZE, 512);                                    \
          break;                                                               \
        case 1024:                                                             \
          LAUNCH_XQA_WIDE(GROUP_SIZE, 1024);                                   \
          break;                                                               \
        default:                                                               \
          TORCH_CHECK(false, "Unsupported XQA partition_size: ", partition_size); \
      }                                                                        \
    } while (0)

  if (q_per_kv == 4) {
    DISPATCH_PARTITION(4);
  } else if (q_per_kv == 6) {
    DISPATCH_PARTITION(6);
  } else {
    DISPATCH_PARTITION(8);
  }

  #undef DISPATCH_PARTITION
  #undef LAUNCH_XQA_WIDE

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor flash_attention_decode_qk_scores(
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& block_table,
    const at::Tensor& seq_lens,
    const float softmax_scale,
    const int partition_size,
    const std::string& kv_cache_dtype,
    const float k_scale) {
  TORCH_CHECK(q.is_cuda(), "q must be on CUDA");
  TORCH_CHECK(k_cache.is_cuda(), "k_cache must be on CUDA");
  TORCH_CHECK(block_table.is_cuda() && seq_lens.is_cuda(),
              "block_table and seq_lens must be on CUDA");
  TORCH_CHECK(q.dtype() == torch::kFloat16, "q must be fp16");
  const int kv_dtype_code = kv_cache_dtype_code_from_string(kv_cache_dtype);
  TORCH_CHECK(kv_dtype_code >= 0, "Unsupported kv_cache_dtype: ", kv_cache_dtype);
  if (kv_dtype_code == flash_v100::KV_CACHE_DTYPE_FP16) {
    TORCH_CHECK(k_cache.dtype() == torch::kFloat16, "k_cache must be fp16");
  } else {
    TORCH_CHECK(k_cache.dtype() == torch::kUInt8,
                "fp8 k_cache must be stored as uint8");
    TORCH_CHECK(k_scale > 0.f, "fp8 k scale must be positive");
  }
  TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must be int32");
  TORCH_CHECK(seq_lens.dtype() == torch::kInt32, "seq_lens must be int32");
  TORCH_CHECK(q.dim() == 3, "q must have shape [B, H, D]");
  TORCH_CHECK(k_cache.dim() == 4,
              "k_cache must have shape [num_blocks, block_size, H_kv, D]");
  TORCH_CHECK(block_table.dim() == 2,
              "block_table must have shape [B, max_num_blocks]");
  TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must have shape [B]");
  TORCH_CHECK(q.stride(-1) == 1, "q last dim must be contiguous");
  TORCH_CHECK(k_cache.stride(-1) == 1, "k_cache last dim must be contiguous");

  const int batch_size = q.size(0);
  const int num_heads_q = q.size(1);
  const int head_dim = q.size(2);
  const int num_heads_kv = k_cache.size(2);
  const int block_size = k_cache.size(1);
  const int max_num_blocks = block_table.size(1);
  const int max_num_partitions =
      (max_num_blocks * block_size + partition_size - 1) / partition_size;

  TORCH_CHECK(q.size(0) <= block_table.size(0),
              "block_table batch size must cover q batch size");
  TORCH_CHECK(q.size(0) <= seq_lens.size(0),
              "seq_lens batch size must cover q batch size");
  TORCH_CHECK(num_heads_q % num_heads_kv == 0,
              "num_heads_q must be divisible by num_heads_kv");
  TORCH_CHECK(k_cache.size(3) == head_dim, "k_cache head_dim mismatch");
  TORCH_CHECK(partition_size == 256 || partition_size == 512 ||
                  partition_size == 1024,
              "Unsupported decode partition_size: ", partition_size);

  c10::cuda::CUDAGuard device_guard(q.device());
  auto scores = torch::full(
      {batch_size, num_heads_q, max_num_partitions, partition_size},
      -1.0e30f,
      q.options().dtype(torch::kFloat32));

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  #define LAUNCH_TYPED(HDIM, PARTITION, KV_DTYPE_CODE)                          \
    launch_flash_attention_decode_qk_scores<HDIM, PARTITION, KV_DTYPE_CODE>(    \
        q, k_cache, block_table, seq_lens, scores, softmax_scale, k_scale, stream)

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
  return scores;
}
