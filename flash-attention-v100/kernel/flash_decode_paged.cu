#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <cstdlib>
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
constexpr int kXQATCG6DualCtaThreads = 6 * kWarpSize;
// The dense 256/128-half strides alias Volta shared-memory banks in the
// WMMA A/B fragment loads. Both padded strides remain 16-byte aligned.
constexpr int kXQATC256WidePaddedQStride = 264;
constexpr int kXQATC256WidePaddedKVStride = 136;
constexpr int kXQATC256WideAlignedPaddedQStride = 272;
constexpr int kXQATC256WideAlignedPaddedKVStride = 144;
constexpr float kXQANegInf = -1.0e30f;

template <bool PADDED_SMEM, bool ALIGNED_PADDED_SMEM = false>
struct alignas(256) XQATCSmem256WideLayout {
  static_assert(!ALIGNED_PADDED_SMEM || PADDED_SMEM,
                "Aligned padding requires padded shared memory");
  static constexpr int kQStride =
      PADDED_SMEM ? (ALIGNED_PADDED_SMEM ? kXQATC256WideAlignedPaddedQStride
                                         : kXQATC256WidePaddedQStride)
                  : 256;
  static constexpr int kKVStride =
      PADDED_SMEM ? (ALIGNED_PADDED_SMEM ? kXQATC256WideAlignedPaddedKVStride
                                         : kXQATC256WidePaddedKVStride)
                  : kXQATCStride;
  alignas(16) __half q[kXQATC256WideBlockM * kQStride];
  union {
    alignas(16) __half k[kXQATCBlockN * kKVStride];
    alignas(16) __half v[kXQATCBlockN * kKVStride];
  } reuse_kv;
  struct {
    alignas(16) float s[kXQATC256WideBlockM * kXQATCBlockN];
    alignas(16) __half p[kXQATC256WideBlockM * kXQATCBlockN];
  } reuse_sp;
  alignas(16) float row_max[kXQATC256WideBlockM];
  alignas(16) float row_sum[kXQATC256WideBlockM];
  alignas(16) int page_ids[kXQATCPageIdsCapacity];
};

constexpr int kXQATCStagedPVTileRows = 64;

template <bool PADDED_SMEM>
struct alignas(256) XQATCStagedPVSmem256Wide {
  static constexpr int kKVStride =
      PADDED_SMEM ? kXQATC256WidePaddedKVStride : kXQATCStride;
  alignas(16) __half v[kXQATCStagedPVTileRows * kKVStride];
  alignas(16) int page_ids[kXQATCPageIdsCapacity];
};

bool xqa_padded_smem_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_PADDED_SMEM");
  return value == nullptr || value[0] != '0';
}

bool xqa_g6_dual_cta_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_G6_DUAL_CTA");
  return value != nullptr && value[0] == '1';
}

bool xqa_mtp5_dual_cta_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_MTP5_DUAL_CTA");
  return value == nullptr || value[0] == '1';
}

bool xqa_g6_dual_cta_dense_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_G6_DUAL_CTA_DENSE");
  return value != nullptr && value[0] == '1';
}

bool xqa_split_reduce_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_SPLIT_REDUCE");
  return value != nullptr && value[0] == '1';
}

int xqa_block16_layout_mode() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_BLOCK16_LAYOUT");
  if (value == nullptr) {
    return 0;
  }
  const int mode = std::atoi(value);
  return mode == 1 || mode == 2 ? mode : 0;
}

bool xqa_block16_layout_required() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_BLOCK16_LAYOUT_REQUIRE");
  return value != nullptr && value[0] == '1';
}

bool xqa_block16_layout_trace_enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("VLLM_FLASH_V100_XQA_BLOCK16_LAYOUT_TRACE");
    return value != nullptr && value[0] == '1';
  }();
  return enabled;
}

bool xqa_block784_index_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_BLOCK784_INDEX");
  return value == nullptr || value[0] != '0';
}

bool xqa_block784_index_trace_enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("VLLM_FLASH_V100_XQA_BLOCK784_INDEX_TRACE");
    return value != nullptr && value[0] == '1';
  }();
  return enabled;
}

bool xqa_aligned_padded_smem_enabled() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_ALIGNED_PADDED_SMEM");
  return value != nullptr && value[0] == '1';
}

bool xqa_aligned_padded_smem_trace_enabled() {
  static const bool enabled = [] {
    const char* value =
        std::getenv("VLLM_FLASH_V100_XQA_ALIGNED_PADDED_SMEM_TRACE");
    return value != nullptr && value[0] == '1';
  }();
  return enabled;
}

int xqa_split_reduce_dim_tile() {
  const char* value = std::getenv("VLLM_FLASH_V100_XQA_SPLIT_REDUCE_D_TILE");
  if (value == nullptr) {
    return 8;
  }
  const int dim_tile = std::atoi(value);
  return dim_tile == 8 || dim_tile == 16 || dim_tile == 32 ? dim_tile : 8;
}

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

template <int NUM_WARPS>
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

template <int NUM_WARPS>
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

__device__ __forceinline__ uint32_t
fp8_e5m2_pair_to_half2_bits(const uint16_t raw_pair) {
  return (static_cast<uint32_t>(raw_pair & 0x00ffu) << 8) |
         (static_cast<uint32_t>(raw_pair & 0xff00u) << 16);
}

__device__ __forceinline__ uint4 fp8_e5m2_vector_to_half8(const uint64_t raw) {
  return make_uint4(
      fp8_e5m2_pair_to_half2_bits(static_cast<uint16_t>(raw)),
      fp8_e5m2_pair_to_half2_bits(static_cast<uint16_t>(raw >> 16)),
      fp8_e5m2_pair_to_half2_bits(static_cast<uint16_t>(raw >> 32)),
      fp8_e5m2_pair_to_half2_bits(static_cast<uint16_t>(raw >> 48)));
}

template <int BLOCK_SIZE, bool CONTIGUOUS_HKV1_LAYOUT,
          int KV_DTYPE = flash_v100::KV_CACHE_DTYPE_FP16>
__device__ __forceinline__ uint4 load_xqa_tc_kv_vector(
    const void* __restrict__ kv_cache, const int* __restrict__ page_ids,
    const int copy_idx, const int panel_d_stride_uint4,
    const int tile_page_offset, const int kv_tile_start, const int block_size,
    const int kv_head_idx, const int64_t block_stride,
    const int64_t token_stride, const int64_t head_stride,
    const int panel_offset) {
  const int row = copy_idx / panel_d_stride_uint4;
  const int vec_col = copy_idx % panel_d_stride_uint4;
  const int token_offset = tile_page_offset + kv_tile_start + row;
  static_assert(BLOCK_SIZE == 0 || BLOCK_SIZE == 16 || BLOCK_SIZE == 784,
                "Unsupported paged-KV block-size specialization");
  static_assert(!CONTIGUOUS_HKV1_LAYOUT || BLOCK_SIZE == 16,
                "The contiguous Hkv=1 layout requires 16-token pages");
  int logical_block;
  int block_offset;
  if constexpr (BLOCK_SIZE == 16) {
    logical_block = token_offset >> 4;
    block_offset = token_offset & 15;
  } else if constexpr (BLOCK_SIZE == 784) {
    logical_block = token_offset / 784;
    block_offset = token_offset - logical_block * 784;
  } else {
    logical_block = token_offset / block_size;
    block_offset = token_offset % block_size;
  }
  const int physical_block = page_ids[logical_block];
  int64_t physical_offset;
  if constexpr (CONTIGUOUS_HKV1_LAYOUT) {
    constexpr int kHeadDim = 256;
    constexpr int kBlockStride = 16 * kHeadDim;
    physical_offset = static_cast<int64_t>(physical_block) * kBlockStride +
                      static_cast<int64_t>(block_offset) * kHeadDim +
                      panel_offset;
  } else {
    physical_offset = static_cast<int64_t>(physical_block) * block_stride +
                      static_cast<int64_t>(block_offset) * token_stride +
                      static_cast<int64_t>(kv_head_idx) * head_stride +
                      panel_offset;
  }
  if constexpr (KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP16) {
    const uint4* cache_vec = reinterpret_cast<const uint4*>(kv_cache);
    return __ldg(&cache_vec[physical_offset / 8 + vec_col]);
  } else {
    static_assert(KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP8_E5M2,
                  "XQA only supports fp16 and FP8 E5M2 KV");
    const uint64_t* cache_vec = reinterpret_cast<const uint64_t*>(kv_cache);
    return fp8_e5m2_vector_to_half8(
        __ldg(&cache_vec[physical_offset / 8 + vec_col]));
  }
}

template <int BLOCK_SIZE, bool CONTIGUOUS_HKV1_LAYOUT, int NUM_THREADS,
          int KV_DTYPE = flash_v100::KV_CACHE_DTYPE_FP16>
__device__ __forceinline__ void load_xqa_tc_kv_panel(
    __half* __restrict__ shared_kv, const void* __restrict__ kv_cache,
    const int* __restrict__ page_ids, const int valid_kv_tile_rows,
    const int panel_d_stride_uint4, const int kv_smem_stride_uint4,
    const int tile_page_offset, const int kv_tile_start, const int block_size,
    const int kv_head_idx, const int64_t block_stride,
    const int64_t token_stride, const int64_t head_stride,
    const int panel_offset) {
  const int copy_count = valid_kv_tile_rows * panel_d_stride_uint4;
  uint4* shared_vec = reinterpret_cast<uint4*>(shared_kv);
  for (int copy_idx = threadIdx.x; copy_idx < copy_count;
       copy_idx += NUM_THREADS) {
    const int row = copy_idx / panel_d_stride_uint4;
    const int vec_col = copy_idx % panel_d_stride_uint4;
    shared_vec[row * kv_smem_stride_uint4 + vec_col] =
        load_xqa_tc_kv_vector<BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT, KV_DTYPE>(
            kv_cache, page_ids, copy_idx, panel_d_stride_uint4,
            tile_page_offset, kv_tile_start, block_size, kv_head_idx,
            block_stride, token_stride, head_stride, panel_offset);
  }
}

template <int D>
__device__ __forceinline__ float dot_qk_half2(const __half* __restrict__ q_ptr,
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

template <int D, int KV_DTYPE>
__device__ __forceinline__ float dot_qk_cache(const __half* __restrict__ q_ptr,
                                              const void* __restrict__ k_cache,
                                              const int64_t k_index_base,
                                              const int lane) {
  if constexpr (KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP16) {
    const __half* k_ptr =
        reinterpret_cast<const __half*>(k_cache) + k_index_base;
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

template <int D, int PARTITION_SIZE, int KV_DTYPE>
__global__ void flash_attention_decode_partition_kernel(
    const __half* __restrict__ q, const void* __restrict__ k_cache,
    const void* __restrict__ v_cache, __half* __restrict__ tmp_out,
    float* __restrict__ max_logits, float* __restrict__ exp_sums,
    const int* __restrict__ block_table, const int* __restrict__ seq_lens,
    const int* __restrict__ active_num_partitions, const int batch_size,
    const int max_num_blocks, const int max_num_partitions,
    const int num_heads_q, const int num_heads_kv, const int block_size,
    const int64_t q_stride0, const int64_t q_stride1,
    const int64_t tmp_out_stride0, const int64_t tmp_out_stride1,
    const int64_t tmp_out_stride2, const int64_t stats_stride0,
    const int64_t stats_stride1, const int64_t k_block_stride,
    const int64_t k_token_stride, const int64_t k_head_stride,
    const int64_t v_block_stride, const int64_t v_token_stride,
    const int64_t v_head_stride, const float softmax_scale, const float k_scale,
    const float v_scale, const int window_size_left,
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
  const int min_token_idx =
      window_size_left >= 0 ? max(0, query_pos - window_size_left) : 0;
  const int max_token_idx =
      window_size_right >= 0 ? min(seq_len - 1, query_pos + window_size_right)
                             : seq_len - 1;
  const int part_start = max(start_token_idx, min_token_idx);
  const int part_end = min(start_token_idx + PARTITION_SIZE, max_token_idx + 1);
  const int q_per_kv = num_heads_q / num_heads_kv;
  const int kv_head_idx = head_idx / q_per_kv;
  const int lane = threadIdx.x % kWarpSize;
  const int warp_idx = threadIdx.x / kWarpSize;
  const float score_scale = KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP16
                                ? softmax_scale
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

    float score = dot_qk_cache<D, KV_DTYPE>(q_shared, k_cache, k_index, lane);
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
      const float vv =
          flash_v100::load_kv_cache_float_unscaled<KV_DTYPE>(v_cache, v_index);
      acc = fmaf(scores_shared[i], vv, acc);
    }
    const float out_scale = KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP16
                                ? inv_part_sum
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

template <int PARTITION_SIZE, int GROUP_SIZE, bool PADDED_SMEM, int NUM_THREADS,
          int MIN_BLOCKS_PER_SM, int BLOCK_SIZE, bool CONTIGUOUS_HKV1_LAYOUT,
          bool ALIGNED_PADDED_SMEM, int KV_DTYPE>
__global__ void __launch_bounds__(NUM_THREADS, MIN_BLOCKS_PER_SM)
    flash_attention_decode_xqa_tc_partition_kernel_256_wide(
        const __half* __restrict__ q, const void* __restrict__ k_cache,
        const void* __restrict__ v_cache, __half* __restrict__ tmp_out,
        float* __restrict__ max_logits, float* __restrict__ exp_sums,
        const int* __restrict__ block_table, const int* __restrict__ seq_lens,
        const int* __restrict__ active_num_partitions, const int batch_size,
        const int max_num_blocks, const int max_num_partitions,
        const int num_heads_q, const int num_heads_kv, const int block_size,
        const int64_t q_stride0, const int64_t q_stride1,
        const int64_t tmp_out_stride0, const int64_t tmp_out_stride1,
        const int64_t tmp_out_stride2, const int64_t stats_stride0,
        const int64_t stats_stride1, const int64_t k_block_stride,
        const int64_t k_token_stride, const int64_t k_head_stride,
        const int64_t v_block_stride, const int64_t v_token_stride,
        const int64_t v_head_stride, const float softmax_scale,
        const float k_scale, const float v_scale) {
  constexpr int D = 256;
  constexpr int WMMA_M = 8;
  constexpr int WMMA_N = 32;
  constexpr int WMMA_K = 16;
  constexpr int kPanelDim = kXQATCStride;
  constexpr int kNumPanels = D / kPanelDim;
  using SmemLayout = XQATCSmem256WideLayout<PADDED_SMEM, ALIGNED_PADDED_SMEM>;
  constexpr int q_global_stride_uint4 = D / 8;
  constexpr int q_smem_stride_uint4 = SmemLayout::kQStride / 8;
  constexpr int kv_smem_stride_uint4 = SmemLayout::kKVStride / 8;
  constexpr int panel_d_stride_uint4 = kPanelDim / 8;
  constexpr int kAccumsPerThread = D / kWarpSize;
  static_assert(GROUP_SIZE == 4 || GROUP_SIZE == 6 || GROUP_SIZE == 8,
                "Wide D=256 TC XQA kernel supports q_per_kv in {4, 6, 8}");
  static_assert(NUM_THREADS >= GROUP_SIZE * kWarpSize,
                "Each XQA query head requires one full softmax/PV warp");

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
  auto& smem = *reinterpret_cast<SmemLayout*>(smem_raw);
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
  for (int idx = tid; idx < GROUP_SIZE * q_global_stride_uint4;
       idx += NUM_THREADS) {
    const int row = idx / q_global_stride_uint4;
    const int vec_col = idx % q_global_stride_uint4;
    const int64_t q_offset =
        static_cast<int64_t>(batch_idx) * q_stride0 +
        static_cast<int64_t>(q_head_base + row) * q_stride1;
    sQ_vec[row * q_smem_stride_uint4 + vec_col] =
        __ldg(&q_vec[q_offset / 8 + vec_col]);
  }
  for (int idx = tid;
       idx < (kXQATC256WideBlockM - GROUP_SIZE) * q_global_stride_uint4;
       idx += NUM_THREADS) {
    const int row = GROUP_SIZE + idx / q_global_stride_uint4;
    const int vec_col = idx % q_global_stride_uint4;
    sQ_vec[row * q_smem_stride_uint4 + vec_col] = make_uint4(0, 0, 0, 0);
  }
  __syncthreads();

  for (int block_n = 0; block_n < num_k_tiles; ++block_n) {
    const int tile_token_start = start_token_idx + block_n * kXQATCBlockN;
    const int valid_k_rows =
        min(kXQATCBlockN, part_tokens - block_n * kXQATCBlockN);
    int start_page;
    int tile_page_offset;
    int page_count;
    if constexpr (BLOCK_SIZE == 16) {
      start_page = tile_token_start >> 4;
      tile_page_offset = tile_token_start & 15;
      page_count = (tile_page_offset + valid_k_rows + 15) >> 4;
    } else if constexpr (BLOCK_SIZE == 784) {
      start_page = tile_token_start / 784;
      tile_page_offset = tile_token_start - start_page * 784;
      page_count = (tile_page_offset + valid_k_rows + 783) / 784;
    } else {
      start_page = tile_token_start / block_size;
      tile_page_offset = tile_token_start - start_page * block_size;
      page_count =
          (tile_page_offset + valid_k_rows + block_size - 1) / block_size;
    }

    for (int idx = tid; idx < page_count; idx += NUM_THREADS) {
      smem.page_ids[idx] = __ldg(&block_table_seq[start_page + idx]);
    }
    __syncthreads();

    for (int kv_tile_start = 0; kv_tile_start < valid_k_rows;
         kv_tile_start += kXQATCBlockN) {
      const int valid_kv_tile_rows =
          min(kXQATCBlockN, valid_k_rows - kv_tile_start);
      // The QK fragments die before softmax/PV. This permits the compiler to
      // reuse their registers for the memory-latency-sensitive PV phase.
      volta::fragment<volta::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                      volta::row_major>
          qk_a_frag;
      volta::fragment<volta::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                      volta::col_major>
          qk_b_frag;
      volta::fragment<volta::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
          qk_acc_frag;
      if (warp_id < (kXQATCBlockN / WMMA_N)) {
        volta::fill_fragment(qk_acc_frag, 0.0f);
      }

      for (int panel_idx = 0; panel_idx < kNumPanels; ++panel_idx) {
        const int panel_offset = panel_idx * kPanelDim;
        load_xqa_tc_kv_panel<BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT, NUM_THREADS,
                             KV_DTYPE>(
            sK, k_cache, smem.page_ids, valid_kv_tile_rows,
            panel_d_stride_uint4, kv_smem_stride_uint4, tile_page_offset,
            kv_tile_start, block_size, kv_head_idx, k_block_stride,
            k_token_stride, k_head_stride, panel_offset);
        for (int idx = tid + valid_kv_tile_rows * panel_d_stride_uint4;
             idx < kXQATCBlockN * panel_d_stride_uint4; idx += NUM_THREADS) {
          const int row = idx / panel_d_stride_uint4;
          const int vec_col = idx % panel_d_stride_uint4;
          reinterpret_cast<uint4*>(sK)[row * kv_smem_stride_uint4 + vec_col] =
              make_uint4(0, 0, 0, 0);
        }
        __syncthreads();

        if (warp_id < (kXQATCBlockN / WMMA_N)) {
          const int tile_n = warp_id * WMMA_N;
#pragma unroll
          for (int k_tile = 0; k_tile < (kPanelDim / WMMA_K); ++k_tile) {
            const int k_offset = k_tile * WMMA_K;
            volta::load_matrix_sync(qk_a_frag, sQ + panel_offset + k_offset,
                                    SmemLayout::kQStride);
            volta::load_matrix_sync(
                qk_b_frag, sK + tile_n * SmemLayout::kKVStride + k_offset,
                SmemLayout::kKVStride);
            volta::mma_sync(qk_acc_frag, qk_a_frag, qk_b_frag, qk_acc_frag);
          }
        }
        __syncthreads();
      }

      if (warp_id < (kXQATCBlockN / WMMA_N)) {
#pragma unroll
        for (int i = 0; i < qk_acc_frag.num_elements; ++i) {
          qk_acc_frag.x[i] *= KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP16
                                  ? softmax_scale
                                  : softmax_scale * k_scale;
        }
        volta::store_matrix_sync(sS + kv_tile_start + warp_id * WMMA_N,
                                 qk_acc_frag, kXQATCBlockN,
                                 volta::mem_row_major);
      }
      __syncthreads();
    }

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
        thread_max =
            fmaxf(thread_max, fmaxf(fmaxf(v4.x, v4.y), fmaxf(v4.z, v4.w)));
      }
#pragma unroll
      for (int c = tail_start + thread_in_row; c < valid_k_rows;
           c += kXQATC256WideThreadsPerRow) {
        thread_max = fmaxf(thread_max, sS_row_f[c]);
      }
#pragma unroll
      for (int o = kXQATC256WideThreadsPerRow / 2; o > 0; o >>= 1) {
        thread_max =
            fmaxf(thread_max, __shfl_down_sync(mask, thread_max, o, kWarpSize));
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
        sP_row_h[c] =
            (c < valid_k_rows) ? __float2half_rn(e) : __float2half(0.f);
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
      for (int kv_tile_start = 0; kv_tile_start < valid_k_rows;
           kv_tile_start += kXQATCBlockN) {
        const int valid_kv_tile_rows =
            min(kXQATCBlockN, valid_k_rows - kv_tile_start);
        load_xqa_tc_kv_panel<BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT, NUM_THREADS,
                             KV_DTYPE>(
            sV, v_cache, smem.page_ids, valid_kv_tile_rows,
            panel_d_stride_uint4, kv_smem_stride_uint4, tile_page_offset,
            kv_tile_start, block_size, kv_head_idx, v_block_stride,
            v_token_stride, v_head_stride, panel_offset);
        for (int idx = tid + valid_kv_tile_rows * panel_d_stride_uint4;
             idx < kXQATCBlockN * panel_d_stride_uint4; idx += NUM_THREADS) {
          const int row = idx / panel_d_stride_uint4;
          const int vec_col = idx % panel_d_stride_uint4;
          reinterpret_cast<uint4*>(sV)[row * kv_smem_stride_uint4 + vec_col] =
              make_uint4(0, 0, 0, 0);
        }
        __syncthreads();

        if (tid < GROUP_SIZE * kXQATC256WideThreadsPerRow) {
          const int row = tid / kXQATC256WideThreadsPerRow;
          const __half* sP_row = sP + row * kXQATCBlockN + kv_tile_start;
#pragma unroll
          for (int token = 0; token < kXQATCBlockN; ++token) {
            if (token >= valid_kv_tile_rows) {
              break;
            }
            const float prob = __half2float(sP_row[token]);
            const __half* sV_row = sV + token * SmemLayout::kKVStride;
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
    __half* tmp_out_ptr = tmp_out +
                          static_cast<int64_t>(batch_idx) * tmp_out_stride0 +
                          static_cast<int64_t>(head_idx) * tmp_out_stride1 +
                          static_cast<int64_t>(partition_idx) * tmp_out_stride2;
    for (int d = thread_in_row; d < D; d += kXQATC256WideThreadsPerRow) {
      const float output_scale = KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP16
                                     ? inv_row_sum
                                     : inv_row_sum * v_scale;
      tmp_out_ptr[d] = __float2half(out_acc[d / kWarpSize] * output_scale);
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

template <int D, int PARTITION_SIZE>
__global__ void flash_attention_decode_reduce_kernel(
    const __half* __restrict__ tmp_out, const float* __restrict__ max_logits,
    const float* __restrict__ exp_sums, const int* __restrict__ seq_lens,
    const int* __restrict__ active_num_partitions, __half* __restrict__ out,
    const int batch_size, const int max_num_partitions, const int num_heads_q,
    const int64_t tmp_out_stride0, const int64_t tmp_out_stride1,
    const int64_t tmp_out_stride2, const int64_t stats_stride0,
    const int64_t stats_stride1, const int64_t out_stride0,
    const int64_t out_stride1) {
  const int batch_idx = blockIdx.x;
  const int head_idx = blockIdx.y;

  if (batch_idx >= batch_size || head_idx >= num_heads_q) {
    return;
  }

  const int seq_len = seq_lens[batch_idx];
  const int num_partitions =
      min(max_num_partitions, (seq_len + PARTITION_SIZE - 1) / PARTITION_SIZE);
  (void)active_num_partitions;

  if (seq_len <= 0 || num_partitions <= 0) {
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
      out[static_cast<int64_t>(batch_idx) * out_stride0 +
          static_cast<int64_t>(head_idx) * out_stride1 + d] = __float2half(0.f);
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
    const float weight =
        exp_sums[stats_index] * __expf(max_shared[i] - global_max);
    weight_shared[i] = weight;
    local_sum += weight;
  }
  const float global_sum = block_reduce_sum<kWarpsPerBlock>(local_sum);
  const float inv_global_sum = global_sum > 0.f ? 1.f / global_sum : 0.f;
  __syncthreads();

  const int64_t out_base = static_cast<int64_t>(batch_idx) * out_stride0 +
                           static_cast<int64_t>(head_idx) * out_stride1;
  const int64_t tmp_out_base =
      static_cast<int64_t>(batch_idx) * tmp_out_stride0 +
      static_cast<int64_t>(head_idx) * tmp_out_stride1;

  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float acc = 0.f;
    for (int i = 0; i < num_partitions; ++i) {
      acc = fmaf(
          weight_shared[i],
          __half2float(tmp_out[tmp_out_base +
                               static_cast<int64_t>(i) * tmp_out_stride2 + d]),
          acc);
    }
    out[out_base + d] = __float2half(acc * inv_global_sum);
  }
}

// This separates only the cross-partition reducer. The stats kernel preserves
// the original block-wide max/sum tree, and each output dimension retains its
// original ascending-partition FMA order.
template <int PARTITION_SIZE, int GROUP_SIZE, bool PADDED_SMEM, int NUM_THREADS,
          int MIN_BLOCKS_PER_SM>
__global__ void __launch_bounds__(NUM_THREADS, MIN_BLOCKS_PER_SM)
    flash_attention_decode_xqa_tc_qk_softmax_staged_kernel_256_wide(
        const __half* __restrict__ q, const __half* __restrict__ k_cache,
        __half* __restrict__ probabilities, float* __restrict__ max_logits,
        float* __restrict__ exp_sums, float* __restrict__ online_rescales,
        const int* __restrict__ block_table, const int* __restrict__ seq_lens,
        const int* __restrict__ active_num_partitions, const int batch_size,
        const int max_num_blocks, const int max_num_partitions,
        const int num_heads_q, const int num_heads_kv, const int block_size,
        const int64_t q_stride0, const int64_t q_stride1,
        const int64_t probability_stride0, const int64_t probability_stride1,
        const int64_t probability_stride2, const int64_t stats_stride0,
        const int64_t stats_stride1, const int64_t online_rescale_stride0,
        const int64_t online_rescale_stride1, const int64_t k_block_stride,
        const int64_t k_token_stride, const int64_t k_head_stride,
        const float softmax_scale) {
  constexpr int D = 256;
  constexpr int WMMA_M = 8;
  constexpr int WMMA_N = 32;
  constexpr int WMMA_K = 16;
  constexpr int kPanelDim = kXQATCStride;
  constexpr int kNumPanels = D / kPanelDim;
  using SmemLayout = XQATCSmem256WideLayout<PADDED_SMEM>;
  constexpr int q_global_stride_uint4 = D / 8;
  constexpr int q_smem_stride_uint4 = SmemLayout::kQStride / 8;
  constexpr int kv_smem_stride_uint4 = SmemLayout::kKVStride / 8;
  constexpr int panel_d_stride_uint4 = kPanelDim / 8;
  static_assert(GROUP_SIZE == 6,
                "Staged D=256 TC XQA is specialized for q_per_kv=6");
  static_assert(NUM_THREADS >= GROUP_SIZE * kWarpSize,
                "Each XQA query head requires one softmax warp");

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
  auto& smem = *reinterpret_cast<SmemLayout*>(smem_raw);
  __half* sQ = smem.q;
  __half* sK = smem.reuse_kv.k;
  float* sS = smem.reuse_sp.s;
  __half* sP = smem.reuse_sp.p;
  float row_max_reg = kXQANegInf;
  float row_sum_reg = 0.f;
  float row_rescale_reg = 1.f;

  const uint4* q_vec = reinterpret_cast<const uint4*>(q);
  uint4* sQ_vec = reinterpret_cast<uint4*>(sQ);
  for (int idx = tid; idx < GROUP_SIZE * q_global_stride_uint4;
       idx += NUM_THREADS) {
    const int row = idx / q_global_stride_uint4;
    const int vec_col = idx % q_global_stride_uint4;
    const int64_t q_offset =
        static_cast<int64_t>(batch_idx) * q_stride0 +
        static_cast<int64_t>(q_head_base + row) * q_stride1;
    sQ_vec[row * q_smem_stride_uint4 + vec_col] =
        __ldg(&q_vec[q_offset / 8 + vec_col]);
  }
  for (int idx = tid;
       idx < (kXQATC256WideBlockM - GROUP_SIZE) * q_global_stride_uint4;
       idx += NUM_THREADS) {
    const int row = GROUP_SIZE + idx / q_global_stride_uint4;
    const int vec_col = idx % q_global_stride_uint4;
    sQ_vec[row * q_smem_stride_uint4 + vec_col] = make_uint4(0, 0, 0, 0);
  }
  __syncthreads();

  for (int block_n = 0; block_n < num_k_tiles; ++block_n) {
    const int tile_token_start = start_token_idx + block_n * kXQATCBlockN;
    const int valid_k_rows =
        min(kXQATCBlockN, part_tokens - block_n * kXQATCBlockN);
    const int start_page = tile_token_start / block_size;
    const int tile_page_offset = tile_token_start - start_page * block_size;
    const int page_count =
        (tile_page_offset + valid_k_rows + block_size - 1) / block_size;

    for (int idx = tid; idx < page_count; idx += NUM_THREADS) {
      smem.page_ids[idx] = __ldg(&block_table_seq[start_page + idx]);
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
    if (warp_id < (kXQATCBlockN / WMMA_N)) {
      volta::fill_fragment(qk_acc_frag, 0.0f);
    }

    for (int panel_idx = 0; panel_idx < kNumPanels; ++panel_idx) {
      const int panel_offset = panel_idx * kPanelDim;
      for (int idx = tid; idx < valid_k_rows * panel_d_stride_uint4;
           idx += NUM_THREADS) {
        const int row = idx / panel_d_stride_uint4;
        const int vec_col = idx % panel_d_stride_uint4;
        const int token_offset = tile_page_offset + row;
        const int physical_block = smem.page_ids[token_offset / block_size];
        const int block_offset = token_offset % block_size;
        const int64_t physical_offset_half_elements =
            static_cast<int64_t>(physical_block) * k_block_stride +
            static_cast<int64_t>(block_offset) * k_token_stride +
            static_cast<int64_t>(kv_head_idx) * k_head_stride + panel_offset;
        const uint4* k_vec = reinterpret_cast<const uint4*>(k_cache);
        reinterpret_cast<uint4*>(sK)[row * kv_smem_stride_uint4 + vec_col] =
            __ldg(&k_vec[physical_offset_half_elements / 8 + vec_col]);
      }
      for (int idx = tid + valid_k_rows * panel_d_stride_uint4;
           idx < kXQATCBlockN * panel_d_stride_uint4; idx += NUM_THREADS) {
        reinterpret_cast<uint4*>(sK)[idx] = make_uint4(0, 0, 0, 0);
      }
      __syncthreads();

      if (warp_id < (kXQATCBlockN / WMMA_N)) {
        const int tile_n = warp_id * WMMA_N;
#pragma unroll
        for (int k_tile = 0; k_tile < (kPanelDim / WMMA_K); ++k_tile) {
          const int k_offset = k_tile * WMMA_K;
          volta::load_matrix_sync(qk_a_frag, sQ + panel_offset + k_offset,
                                  SmemLayout::kQStride);
          volta::load_matrix_sync(
              qk_b_frag, sK + tile_n * SmemLayout::kKVStride + k_offset,
              SmemLayout::kKVStride);
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
      volta::store_matrix_sync(sS + warp_id * WMMA_N, qk_acc_frag, kXQATCBlockN,
                               volta::mem_row_major);
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
        thread_max =
            fmaxf(thread_max, fmaxf(fmaxf(v4.x, v4.y), fmaxf(v4.z, v4.w)));
      }
#pragma unroll
      for (int c = tail_start + thread_in_row; c < valid_k_rows;
           c += kXQATC256WideThreadsPerRow) {
        thread_max = fmaxf(thread_max, sS_row_f[c]);
      }
#pragma unroll
      for (int o = kXQATC256WideThreadsPerRow / 2; o > 0; o >>= 1) {
        thread_max =
            fmaxf(thread_max, __shfl_down_sync(mask, thread_max, o, kWarpSize));
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
        sP_row_h[c] =
            (c < valid_k_rows) ? __float2half_rn(e) : __float2half(0.f);
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
      if (block_n > 0) {
        row_rescale_reg = exp_diff;
      }

      __half2* sP_half2 = reinterpret_cast<__half2*>(sP_row_h);
      if (vec_col < vec_cols) {
        const int base_offset = vec_col * 2;
        sP_half2[base_offset] = packed_exp0;
        sP_half2[base_offset + 1] = packed_exp1;
      }
    }
    __syncthreads();

    for (int idx = tid; idx < GROUP_SIZE * valid_k_rows; idx += NUM_THREADS) {
      const int row = idx / valid_k_rows;
      const int token = idx % valid_k_rows;
      const int head_idx = q_head_base + row;
      __half* probability_ptr =
          probabilities +
          static_cast<int64_t>(batch_idx) * probability_stride0 +
          static_cast<int64_t>(head_idx) * probability_stride1 +
          static_cast<int64_t>(partition_idx) * probability_stride2 +
          block_n * kXQATCBlockN + token;
      *probability_ptr = sP[row * kXQATCBlockN + token];
    }
    __syncthreads();
  }

  if (tid < GROUP_SIZE * kXQATC256WideThreadsPerRow) {
    const int row = tid / kXQATC256WideThreadsPerRow;
    const int thread_in_row = tid % kXQATC256WideThreadsPerRow;
    if (thread_in_row == 0) {
      const int head_idx = q_head_base + row;
      const int64_t stats_index =
          static_cast<int64_t>(batch_idx) * stats_stride0 +
          static_cast<int64_t>(head_idx) * stats_stride1 + partition_idx;
      max_logits[stats_index] = row_max_reg;
      exp_sums[stats_index] = row_sum_reg;
      online_rescales[static_cast<int64_t>(batch_idx) * online_rescale_stride0 +
                      static_cast<int64_t>(head_idx) * online_rescale_stride1 +
                      partition_idx] = row_rescale_reg;
    }
  }
}

template <int PARTITION_SIZE, int GROUP_SIZE, bool PADDED_SMEM, int NUM_THREADS,
          int MIN_BLOCKS_PER_SM>
__global__ void __launch_bounds__(NUM_THREADS, MIN_BLOCKS_PER_SM)
    flash_attention_decode_xqa_tc_pv_staged_kernel_256_wide(
        const __half* __restrict__ v_cache, const __half* probabilities,
        __half* tmp_out, const float* __restrict__ exp_sums,
        const float* __restrict__ online_rescales,
        const int* __restrict__ block_table, const int* __restrict__ seq_lens,
        const int* __restrict__ active_num_partitions, const int batch_size,
        const int max_num_blocks, const int max_num_partitions,
        const int num_heads_q, const int num_heads_kv, const int block_size,
        const int64_t tmp_out_stride0, const int64_t tmp_out_stride1,
        const int64_t tmp_out_stride2, const int64_t stats_stride0,
        const int64_t stats_stride1, const int64_t online_rescale_stride0,
        const int64_t online_rescale_stride1, const int64_t v_block_stride,
        const int64_t v_token_stride, const int64_t v_head_stride) {
  constexpr int D = 256;
  constexpr int kPanelDim = kXQATCStride;
  constexpr int kNumPanels = D / kPanelDim;
  constexpr int kAccumsPerThread = D / kWarpSize;
  using SmemLayout = XQATCStagedPVSmem256Wide<PADDED_SMEM>;
  constexpr int kv_smem_stride_uint4 = SmemLayout::kKVStride / 8;
  constexpr int panel_d_stride_uint4 = kPanelDim / 8;
  static_assert(GROUP_SIZE == 6,
                "Staged D=256 TC XQA is specialized for q_per_kv=6");
  static_assert(NUM_THREADS >= GROUP_SIZE * kWarpSize,
                "Each XQA query head requires one PV warp");

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
  const int lane_id = tid % kWarpSize;
  const int part_tokens = min(PARTITION_SIZE, seq_len - start_token_idx);
  const int num_k_tiles = (part_tokens + kXQATCBlockN - 1) / kXQATCBlockN;
  const int* block_table_seq = block_table + batch_idx * max_num_blocks;
  extern __shared__ char smem_raw[];
  auto& smem = *reinterpret_cast<SmemLayout*>(smem_raw);
  __half* sV = smem.v;
  float out_acc[kAccumsPerThread];
#pragma unroll
  for (int i = 0; i < kAccumsPerThread; ++i) {
    out_acc[i] = 0.f;
  }

  for (int block_n = 0; block_n < num_k_tiles; ++block_n) {
    const int tile_token_start = start_token_idx + block_n * kXQATCBlockN;
    const int valid_k_rows =
        min(kXQATCBlockN, part_tokens - block_n * kXQATCBlockN);
    const int start_page = tile_token_start / block_size;
    const int tile_page_offset = tile_token_start - start_page * block_size;
    const int page_count =
        (tile_page_offset + valid_k_rows + block_size - 1) / block_size;
    for (int idx = tid; idx < page_count; idx += NUM_THREADS) {
      smem.page_ids[idx] = __ldg(&block_table_seq[start_page + idx]);
    }
    __syncthreads();

    if (block_n > 0 && tid < GROUP_SIZE * kXQATC256WideThreadsPerRow) {
      const int row = tid / kXQATC256WideThreadsPerRow;
      const int thread_in_row = tid % kXQATC256WideThreadsPerRow;
      float rescale = 1.f;
      if (thread_in_row == 0) {
        const int head_idx = q_head_base + row;
        rescale = online_rescales[static_cast<int64_t>(batch_idx) *
                                      online_rescale_stride0 +
                                  static_cast<int64_t>(head_idx) *
                                      online_rescale_stride1 +
                                  partition_idx];
      }
      rescale = __shfl_sync(0xffffffffu, rescale, 0, kWarpSize);
#pragma unroll
      for (int i = 0; i < kAccumsPerThread; ++i) {
        out_acc[i] *= rescale;
      }
    }

    for (int panel_idx = 0; panel_idx < kNumPanels; ++panel_idx) {
      const int panel_offset = panel_idx * kPanelDim;
      for (int v_tile_start = 0; v_tile_start < valid_k_rows;
           v_tile_start += kXQATCStagedPVTileRows) {
        const int valid_v_rows =
            min(kXQATCStagedPVTileRows, valid_k_rows - v_tile_start);
        for (int idx = tid; idx < valid_v_rows * panel_d_stride_uint4;
             idx += NUM_THREADS) {
          const int row = idx / panel_d_stride_uint4;
          const int vec_col = idx % panel_d_stride_uint4;
          const int token_offset = tile_page_offset + v_tile_start + row;
          const int physical_block = smem.page_ids[token_offset / block_size];
          const int block_offset = token_offset % block_size;
          const int64_t physical_offset_half_elements =
              static_cast<int64_t>(physical_block) * v_block_stride +
              static_cast<int64_t>(block_offset) * v_token_stride +
              static_cast<int64_t>(kv_head_idx) * v_head_stride + panel_offset;
          const uint4* v_vec = reinterpret_cast<const uint4*>(v_cache);
          reinterpret_cast<uint4*>(sV)[row * kv_smem_stride_uint4 + vec_col] =
              __ldg(&v_vec[physical_offset_half_elements / 8 + vec_col]);
        }
        for (int idx = tid + valid_v_rows * panel_d_stride_uint4;
             idx < kXQATCStagedPVTileRows * panel_d_stride_uint4;
             idx += NUM_THREADS) {
          reinterpret_cast<uint4*>(sV)[idx] = make_uint4(0, 0, 0, 0);
        }
        __syncthreads();

        if (tid < GROUP_SIZE * kXQATC256WideThreadsPerRow) {
          const int row = tid / kXQATC256WideThreadsPerRow;
          const int head_idx = q_head_base + row;
          const __half* probability_ptr =
              probabilities +
              static_cast<int64_t>(batch_idx) * tmp_out_stride0 +
              static_cast<int64_t>(head_idx) * tmp_out_stride1 +
              static_cast<int64_t>(partition_idx) * tmp_out_stride2 +
              block_n * kXQATCBlockN + v_tile_start;
#pragma unroll
          for (int token = 0; token < kXQATCStagedPVTileRows; ++token) {
            if (token >= valid_v_rows) {
              break;
            }
            float prob = 0.f;
            if (lane_id == 0) {
              prob = __half2float(probability_ptr[token]);
            }
            prob = __shfl_sync(0xffffffffu, prob, 0, kWarpSize);
            const __half* sV_row = sV + token * SmemLayout::kKVStride;
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
  }

  if (tid < GROUP_SIZE * kXQATC256WideThreadsPerRow) {
    const int row = tid / kXQATC256WideThreadsPerRow;
    const int thread_in_row = tid % kXQATC256WideThreadsPerRow;
    const int head_idx = q_head_base + row;
    const float row_sum =
        exp_sums[static_cast<int64_t>(batch_idx) * stats_stride0 +
                 static_cast<int64_t>(head_idx) * stats_stride1 +
                 partition_idx];
    const float inv_row_sum = row_sum > 0.f ? 1.f / row_sum : 0.f;
    __half* tmp_out_ptr = tmp_out +
                          static_cast<int64_t>(batch_idx) * tmp_out_stride0 +
                          static_cast<int64_t>(head_idx) * tmp_out_stride1 +
                          static_cast<int64_t>(partition_idx) * tmp_out_stride2;
    for (int d = thread_in_row; d < D; d += kXQATC256WideThreadsPerRow) {
      tmp_out_ptr[d] = __float2half(out_acc[d / kWarpSize] * inv_row_sum);
    }
  }
}

template <int PARTITION_SIZE>
__global__ void flash_attention_decode_xqa_reduce_stats_kernel(
    float* __restrict__ max_logits, float* __restrict__ exp_sums,
    const int* __restrict__ seq_lens, const int batch_size,
    const int max_num_partitions, const int num_heads_q,
    const int64_t stats_stride0, const int64_t stats_stride1) {
  const int batch_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  if (batch_idx >= batch_size || head_idx >= num_heads_q) {
    return;
  }

  const int seq_len = seq_lens[batch_idx];
  const int num_partitions =
      min(max_num_partitions, (seq_len + PARTITION_SIZE - 1) / PARTITION_SIZE);
  if (seq_len <= 0 || num_partitions <= 0) {
    return;
  }

  extern __shared__ float max_shared[];
  const int64_t stats_base = static_cast<int64_t>(batch_idx) * stats_stride0 +
                             static_cast<int64_t>(head_idx) * stats_stride1;

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
    max_logits[stats_base + i] = weight;
    local_sum += weight;
  }
  const float global_sum = block_reduce_sum<kWarpsPerBlock>(local_sum);
  if (threadIdx.x == 0) {
    // The partition sum is dead after weights have been materialized.
    exp_sums[stats_base] = global_sum;
  }
}

template <int D, int PARTITION_SIZE, int D_TILE>
__global__ void flash_attention_decode_xqa_reduce_output_kernel(
    const __half* __restrict__ tmp_out, const float* __restrict__ weights,
    const float* __restrict__ global_sums, const int* __restrict__ seq_lens,
    __half* __restrict__ out, const int batch_size,
    const int max_num_partitions, const int num_heads_q,
    const int64_t tmp_out_stride0, const int64_t tmp_out_stride1,
    const int64_t tmp_out_stride2, const int64_t stats_stride0,
    const int64_t stats_stride1, const int64_t out_stride0,
    const int64_t out_stride1) {
  static_assert(D_TILE > 0 && D_TILE <= kWarpSize,
                "Split-reduce dimension tile must fit in one warp");
  const int batch_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int d = blockIdx.z * D_TILE + threadIdx.x;
  if (batch_idx >= batch_size || head_idx >= num_heads_q || d >= D) {
    return;
  }

  const int64_t out_index = static_cast<int64_t>(batch_idx) * out_stride0 +
                            static_cast<int64_t>(head_idx) * out_stride1 + d;
  const int seq_len = seq_lens[batch_idx];
  const int num_partitions =
      min(max_num_partitions, (seq_len + PARTITION_SIZE - 1) / PARTITION_SIZE);
  if (seq_len <= 0 || num_partitions <= 0) {
    out[out_index] = __float2half(0.f);
    return;
  }

  const int64_t stats_base = static_cast<int64_t>(batch_idx) * stats_stride0 +
                             static_cast<int64_t>(head_idx) * stats_stride1;
  const int64_t tmp_out_base =
      static_cast<int64_t>(batch_idx) * tmp_out_stride0 +
      static_cast<int64_t>(head_idx) * tmp_out_stride1;
  const float global_sum = global_sums[stats_base];
  const float inv_global_sum = global_sum > 0.f ? 1.f / global_sum : 0.f;

  float acc = 0.f;
  for (int i = 0; i < num_partitions; ++i) {
    acc = fmaf(
        weights[stats_base + i],
        __half2float(tmp_out[tmp_out_base +
                             static_cast<int64_t>(i) * tmp_out_stride2 + d]),
        acc);
  }
  out[out_index] = __float2half(acc * inv_global_sum);
}

template <int D, int PARTITION_SIZE, int KV_DTYPE>
__global__ void flash_attention_decode_qk_scores_kernel(
    const __half* __restrict__ q, const void* __restrict__ k_cache,
    const int* __restrict__ block_table, const int* __restrict__ seq_lens,
    float* __restrict__ scores, const int batch_size, const int max_num_blocks,
    const int max_num_partitions, const int num_heads_q, const int num_heads_kv,
    const int block_size, const int64_t q_stride0, const int64_t q_stride1,
    const int64_t scores_stride0, const int64_t scores_stride1,
    const int64_t scores_stride2, const int64_t k_block_stride,
    const int64_t k_token_stride, const int64_t k_head_stride,
    const float softmax_scale, const float k_scale) {
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
  const float score_scale = KV_DTYPE == flash_v100::KV_CACHE_DTYPE_FP16
                                ? softmax_scale
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

    float score = dot_qk_cache<D, KV_DTYPE>(q_shared, k_cache, k_index, lane);
    if (lane == 0) {
      scores[score_base + token_local] = score * score_scale;
    }
  }
}

template <int D, int PARTITION_SIZE, int KV_DTYPE>
void launch_flash_attention_decode_paged(
    const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache,
    at::Tensor& out, const at::Tensor& block_table, const at::Tensor& seq_lens,
    at::Tensor& tmp_out, at::Tensor& max_logits, at::Tensor& exp_sums,
    const at::Tensor& active_num_partitions, const float softmax_scale,
    const int launch_num_partitions, const float k_scale, const float v_scale,
    const int window_size_left, const int window_size_right,
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
          k_cache.data_ptr(), v_cache.data_ptr(),
          reinterpret_cast<__half*>(tmp_out.data_ptr<at::Half>()),
          max_logits.data_ptr<float>(), exp_sums.data_ptr<float>(),
          block_table.data_ptr<int>(), seq_lens.data_ptr<int>(),
          active_num_partitions.data_ptr<int>(), batch_size, max_num_blocks,
          max_num_partitions, num_heads_q, num_heads_kv, block_size,
          q.stride(0), q.stride(1), tmp_out.stride(0), tmp_out.stride(1),
          tmp_out.stride(2), max_logits.stride(0), max_logits.stride(1),
          k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
          v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
          softmax_scale, k_scale, v_scale, window_size_left, window_size_right);

  flash_attention_decode_reduce_kernel<D, PARTITION_SIZE>
      <<<reduce_grid, block, reduce_shared_mem, stream>>>(
          reinterpret_cast<const __half*>(tmp_out.data_ptr<at::Half>()),
          max_logits.data_ptr<float>(), exp_sums.data_ptr<float>(),
          seq_lens.data_ptr<int>(), active_num_partitions.data_ptr<int>(),
          reinterpret_cast<__half*>(out.data_ptr<at::Half>()), batch_size,
          max_num_partitions, num_heads_q, tmp_out.stride(0), tmp_out.stride(1),
          tmp_out.stride(2), max_logits.stride(0), max_logits.stride(1),
          out.stride(0), out.stride(1));
}

template <int PARTITION_SIZE>
void launch_flash_attention_decode_xqa_split_reduce(
    at::Tensor& out, const at::Tensor& seq_lens, const at::Tensor& tmp_out,
    at::Tensor& max_logits, at::Tensor& exp_sums,
    const int launch_num_partitions, const int dim_tile, cudaStream_t stream) {
  const int batch_size = out.size(0);
  const int num_heads_q = out.size(1);
  const dim3 stats_grid(batch_size, num_heads_q, 1);
  const dim3 stats_block(kThreadsPerBlock);
  const size_t stats_shared_mem =
      static_cast<size_t>(launch_num_partitions) * sizeof(float);
  flash_attention_decode_xqa_reduce_stats_kernel<PARTITION_SIZE>
      <<<stats_grid, stats_block, stats_shared_mem, stream>>>(
          max_logits.data_ptr<float>(), exp_sums.data_ptr<float>(),
          seq_lens.data_ptr<int>(), batch_size, launch_num_partitions,
          num_heads_q, max_logits.stride(0), max_logits.stride(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

#define LAUNCH_SPLIT_REDUCE_OUTPUT(D_TILE)                                   \
  do {                                                                       \
    const dim3 output_grid(batch_size, num_heads_q,                          \
                           (256 + D_TILE - 1) / D_TILE);                     \
    const dim3 output_block(D_TILE);                                         \
    flash_attention_decode_xqa_reduce_output_kernel<256, PARTITION_SIZE,     \
                                                    D_TILE>                  \
        <<<output_grid, output_block, 0, stream>>>(                          \
            reinterpret_cast<const __half*>(tmp_out.data_ptr<at::Half>()),   \
            max_logits.data_ptr<float>(), exp_sums.data_ptr<float>(),        \
            seq_lens.data_ptr<int>(),                                        \
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()), batch_size, \
            launch_num_partitions, num_heads_q, tmp_out.stride(0),           \
            tmp_out.stride(1), tmp_out.stride(2), max_logits.stride(0),      \
            max_logits.stride(1), out.stride(0), out.stride(1));             \
  } while (0)

  switch (dim_tile) {
    case 16:
      LAUNCH_SPLIT_REDUCE_OUTPUT(16);
      break;
    case 32:
      LAUNCH_SPLIT_REDUCE_OUTPUT(32);
      break;
    default:
      LAUNCH_SPLIT_REDUCE_OUTPUT(8);
      break;
  }
#undef LAUNCH_SPLIT_REDUCE_OUTPUT
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int PARTITION_SIZE, int GROUP_SIZE, bool PADDED_SMEM,
          int NUM_THREADS = kXQATC256WideThreads, int MIN_BLOCKS_PER_SM = 1,
          int BLOCK_SIZE = 0, bool CONTIGUOUS_HKV1_LAYOUT = false,
          bool ALIGNED_PADDED_SMEM = false>
void launch_flash_attention_decode_paged_xqa_tc_256_wide(
    const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache,
    at::Tensor& out, const at::Tensor& block_table, const at::Tensor& seq_lens,
    at::Tensor& tmp_out, at::Tensor& max_logits, at::Tensor& exp_sums,
    const at::Tensor& active_num_partitions, const float softmax_scale,
    const float k_scale, const float v_scale, const int launch_num_partitions,
    const bool use_split_reduce, const int split_reduce_dim_tile,
    cudaStream_t stream) {
  const int batch_size = q.size(0);
  const int num_heads_q = q.size(1);
  const int num_heads_kv = k_cache.size(2);
  const int max_num_blocks = block_table.size(1);
  const dim3 partition_grid(batch_size, num_heads_kv, launch_num_partitions);
  const size_t shared_mem =
      sizeof(XQATCSmem256WideLayout<PADDED_SMEM, ALIGNED_PADDED_SMEM>);
#define LAUNCH_XQA_PARTITION(KV_DTYPE)                                         \
  do {                                                                         \
    auto partition_kernel =                                                    \
        (void*)flash_attention_decode_xqa_tc_partition_kernel_256_wide<        \
            PARTITION_SIZE, GROUP_SIZE, PADDED_SMEM, NUM_THREADS,              \
            MIN_BLOCKS_PER_SM, BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT,             \
            ALIGNED_PADDED_SMEM, KV_DTYPE>;                                    \
    cudaFuncSetAttribute(partition_kernel,                                     \
                         cudaFuncAttributeMaxDynamicSharedMemorySize,          \
                         shared_mem);                                          \
    if constexpr (MIN_BLOCKS_PER_SM > 1) {                                     \
      cudaFuncSetAttribute(partition_kernel,                                   \
                           cudaFuncAttributePreferredSharedMemoryCarveout,     \
                           100);                                               \
    }                                                                          \
    flash_attention_decode_xqa_tc_partition_kernel_256_wide<                   \
        PARTITION_SIZE, GROUP_SIZE, PADDED_SMEM, NUM_THREADS,                  \
        MIN_BLOCKS_PER_SM, BLOCK_SIZE, CONTIGUOUS_HKV1_LAYOUT,                 \
        ALIGNED_PADDED_SMEM, KV_DTYPE>                                         \
        <<<partition_grid, NUM_THREADS, shared_mem, stream>>>(                 \
            reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),           \
            k_cache.data_ptr(), v_cache.data_ptr(),                            \
            reinterpret_cast<__half*>(tmp_out.data_ptr<at::Half>()),           \
            max_logits.data_ptr<float>(), exp_sums.data_ptr<float>(),          \
            block_table.data_ptr<int>(), seq_lens.data_ptr<int>(),             \
            active_num_partitions.data_ptr<int>(), batch_size, max_num_blocks, \
            launch_num_partitions, num_heads_q, num_heads_kv, k_cache.size(1), \
            q.stride(0), q.stride(1), tmp_out.stride(0), tmp_out.stride(1),    \
            tmp_out.stride(2), max_logits.stride(0), max_logits.stride(1),     \
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),           \
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),           \
            softmax_scale, k_scale, v_scale);                                  \
  } while (0)

  if (k_cache.scalar_type() == at::kByte) {
    LAUNCH_XQA_PARTITION(flash_v100::KV_CACHE_DTYPE_FP8_E5M2);
  } else {
    LAUNCH_XQA_PARTITION(flash_v100::KV_CACHE_DTYPE_FP16);
  }
#undef LAUNCH_XQA_PARTITION
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (use_split_reduce) {
    launch_flash_attention_decode_xqa_split_reduce<PARTITION_SIZE>(
        out, seq_lens, tmp_out, max_logits, exp_sums, launch_num_partitions,
        split_reduce_dim_tile, stream);
  } else {
    const dim3 reduce_grid(batch_size, num_heads_q, 1);
    const dim3 block(kThreadsPerBlock);
    const size_t reduce_shared_mem =
        static_cast<size_t>(2 * launch_num_partitions) * sizeof(float);
    flash_attention_decode_reduce_kernel<256, PARTITION_SIZE>
        <<<reduce_grid, block, reduce_shared_mem, stream>>>(
            reinterpret_cast<const __half*>(tmp_out.data_ptr<at::Half>()),
            max_logits.data_ptr<float>(), exp_sums.data_ptr<float>(),
            seq_lens.data_ptr<int>(), active_num_partitions.data_ptr<int>(),
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()), batch_size,
            launch_num_partitions, num_heads_q, tmp_out.stride(0),
            tmp_out.stride(1), tmp_out.stride(2), max_logits.stride(0),
            max_logits.stride(1), out.stride(0), out.stride(1));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_flash_attention_decode_paged_xqa_tc_256_staged(
    const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache,
    at::Tensor& out, const at::Tensor& block_table, const at::Tensor& seq_lens,
    at::Tensor& tmp_out, at::Tensor& max_logits, at::Tensor& exp_sums,
    at::Tensor& online_rescales, const at::Tensor& active_num_partitions,
    const float softmax_scale, const int launch_num_partitions,
    const bool use_split_reduce, const int split_reduce_dim_tile,
    cudaStream_t stream) {
  constexpr int kGroupSize = 6;
  constexpr int kQKThreads = kXQATCG6DualCtaThreads;
  constexpr int kPVThreads = kXQATCG6DualCtaThreads;
  constexpr int kQKMinBlocksPerSM = 2;
  constexpr int kPVMinBlocksPerSM = 4;
  using QKSmem = XQATCSmem256WideLayout<true>;
  using PVSmem = XQATCStagedPVSmem256Wide<true>;

  const int batch_size = q.size(0);
  const int num_heads_q = q.size(1);
  const int num_heads_kv = k_cache.size(2);
  const int max_num_blocks = block_table.size(1);
  const dim3 partition_grid(batch_size, num_heads_kv, launch_num_partitions);
  const size_t qk_shared_mem = sizeof(QKSmem);
  const size_t pv_shared_mem = sizeof(PVSmem);
  auto qk_kernel =
      (void*)flash_attention_decode_xqa_tc_qk_softmax_staged_kernel_256_wide<
          256, kGroupSize, true, kQKThreads, kQKMinBlocksPerSM>;
  auto pv_kernel =
      (void*)flash_attention_decode_xqa_tc_pv_staged_kernel_256_wide<
          256, kGroupSize, true, kPVThreads, kPVMinBlocksPerSM>;
  const cudaError_t qk_smem_status = cudaFuncSetAttribute(
      qk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, qk_shared_mem);
  TORCH_CHECK(qk_smem_status == cudaSuccess,
              "Failed to set staged XQA QK shared memory: ",
              cudaGetErrorString(qk_smem_status));
  const cudaError_t qk_carveout_status = cudaFuncSetAttribute(
      qk_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  TORCH_CHECK(qk_carveout_status == cudaSuccess,
              "Failed to set staged XQA QK shared-memory carveout: ",
              cudaGetErrorString(qk_carveout_status));
  const cudaError_t pv_smem_status = cudaFuncSetAttribute(
      pv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, pv_shared_mem);
  TORCH_CHECK(pv_smem_status == cudaSuccess,
              "Failed to set staged XQA PV shared memory: ",
              cudaGetErrorString(pv_smem_status));
  const cudaError_t pv_carveout_status = cudaFuncSetAttribute(
      pv_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  TORCH_CHECK(pv_carveout_status == cudaSuccess,
              "Failed to set staged XQA PV shared-memory carveout: ",
              cudaGetErrorString(pv_carveout_status));

  flash_attention_decode_xqa_tc_qk_softmax_staged_kernel_256_wide<
      256, kGroupSize, true, kQKThreads, kQKMinBlocksPerSM>
      <<<partition_grid, kQKThreads, qk_shared_mem, stream>>>(
          reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
          reinterpret_cast<const __half*>(k_cache.data_ptr<at::Half>()),
          reinterpret_cast<__half*>(tmp_out.data_ptr<at::Half>()),
          max_logits.data_ptr<float>(), exp_sums.data_ptr<float>(),
          online_rescales.data_ptr<float>(), block_table.data_ptr<int>(),
          seq_lens.data_ptr<int>(), active_num_partitions.data_ptr<int>(),
          batch_size, max_num_blocks, launch_num_partitions, num_heads_q,
          num_heads_kv, k_cache.size(1), q.stride(0), q.stride(1),
          tmp_out.stride(0), tmp_out.stride(1), tmp_out.stride(2),
          max_logits.stride(0), max_logits.stride(1), online_rescales.stride(0),
          online_rescales.stride(1), k_cache.stride(0), k_cache.stride(1),
          k_cache.stride(2), softmax_scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  flash_attention_decode_xqa_tc_pv_staged_kernel_256_wide<
      256, kGroupSize, true, kPVThreads, kPVMinBlocksPerSM>
      <<<partition_grid, kPVThreads, pv_shared_mem, stream>>>(
          reinterpret_cast<const __half*>(v_cache.data_ptr<at::Half>()),
          reinterpret_cast<const __half*>(tmp_out.data_ptr<at::Half>()),
          reinterpret_cast<__half*>(tmp_out.data_ptr<at::Half>()),
          exp_sums.data_ptr<float>(), online_rescales.data_ptr<float>(),
          block_table.data_ptr<int>(), seq_lens.data_ptr<int>(),
          active_num_partitions.data_ptr<int>(), batch_size, max_num_blocks,
          launch_num_partitions, num_heads_q, num_heads_kv, v_cache.size(1),
          tmp_out.stride(0), tmp_out.stride(1), tmp_out.stride(2),
          max_logits.stride(0), max_logits.stride(1), online_rescales.stride(0),
          online_rescales.stride(1), v_cache.stride(0), v_cache.stride(1),
          v_cache.stride(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (use_split_reduce) {
    launch_flash_attention_decode_xqa_split_reduce<256>(
        out, seq_lens, tmp_out, max_logits, exp_sums, launch_num_partitions,
        split_reduce_dim_tile, stream);
  } else {
    const dim3 reduce_grid(batch_size, num_heads_q, 1);
    const dim3 reduce_block(kThreadsPerBlock);
    const size_t reduce_shared_mem =
        static_cast<size_t>(2 * launch_num_partitions) * sizeof(float);
    flash_attention_decode_reduce_kernel<256, 256>
        <<<reduce_grid, reduce_block, reduce_shared_mem, stream>>>(
            reinterpret_cast<const __half*>(tmp_out.data_ptr<at::Half>()),
            max_logits.data_ptr<float>(), exp_sums.data_ptr<float>(),
            seq_lens.data_ptr<int>(), active_num_partitions.data_ptr<int>(),
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()), batch_size,
            launch_num_partitions, num_heads_q, tmp_out.stride(0),
            tmp_out.stride(1), tmp_out.stride(2), max_logits.stride(0),
            max_logits.stride(1), out.stride(0), out.stride(1));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int D, int PARTITION_SIZE, int KV_DTYPE>
void launch_flash_attention_decode_qk_scores(
    const at::Tensor& q, const at::Tensor& k_cache,
    const at::Tensor& block_table, const at::Tensor& seq_lens,
    at::Tensor& scores, const float softmax_scale, const float k_scale,
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
          k_cache.data_ptr(), block_table.data_ptr<int>(),
          seq_lens.data_ptr<int>(), scores.data_ptr<float>(), batch_size,
          max_num_blocks, max_num_partitions, num_heads_q, num_heads_kv,
          block_size, q.stride(0), q.stride(1), scores.stride(0),
          scores.stride(1), scores.stride(2), k_cache.stride(0),
          k_cache.stride(1), k_cache.stride(2), softmax_scale, k_scale);
}

}  // namespace

at::Tensor flash_attention_decode_paged(
    const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache,
    std::optional<at::Tensor>& out_, const at::Tensor& block_table,
    const at::Tensor& seq_lens, at::Tensor& tmp_out, at::Tensor& max_logits,
    at::Tensor& exp_sums, const at::Tensor& active_num_partitions,
    const float softmax_scale, const int partition_size,
    const int launch_num_partitions, const std::string& kv_cache_dtype,
    const float k_scale, const float v_scale, const int window_size_left,
    const int window_size_right) {
  TORCH_CHECK(q.is_cuda(), "q must be on CUDA");
  TORCH_CHECK(k_cache.is_cuda() && v_cache.is_cuda(),
              "k/v cache must be on CUDA");
  TORCH_CHECK(block_table.is_cuda() && seq_lens.is_cuda(),
              "block_table and seq_lens must be on CUDA");
  TORCH_CHECK(tmp_out.is_cuda() && max_logits.is_cuda() && exp_sums.is_cuda(),
              "workspace tensors must be on CUDA");
  TORCH_CHECK(active_num_partitions.is_cuda(),
              "active_num_partitions must be on CUDA");
  TORCH_CHECK(q.dtype() == torch::kFloat16, "q must be fp16");
  const int kv_dtype_code = kv_cache_dtype_code_from_string(kv_cache_dtype);
  TORCH_CHECK(kv_dtype_code >= 0,
              "Unsupported kv_cache_dtype: ", kv_cache_dtype);
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
  TORCH_CHECK(block_table.dtype() == torch::kInt32,
              "block_table must be int32");
  TORCH_CHECK(seq_lens.dtype() == torch::kInt32, "seq_lens must be int32");
  TORCH_CHECK(active_num_partitions.dtype() == torch::kInt32,
              "active_num_partitions must be int32");
  TORCH_CHECK(q.dim() == 3, "q must have shape [B, H, D]");
  TORCH_CHECK(k_cache.dim() == 4,
              "k_cache must have shape [num_blocks, block_size, H_kv, D]");
  TORCH_CHECK(v_cache.dim() == 4,
              "v_cache must have shape [num_blocks, block_size, H_kv, D]");
  TORCH_CHECK(block_table.dim() == 2,
              "block_table must have shape [B, max_num_blocks]");
  TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must have shape [B]");
  TORCH_CHECK(tmp_out.dim() == 4, "tmp_out must have shape [B_cap, H, P, D]");
  TORCH_CHECK(max_logits.dim() == 3,
              "max_logits must have shape [B_cap, H, P]");
  TORCH_CHECK(exp_sums.dim() == 3, "exp_sums must have shape [B_cap, H, P]");
  TORCH_CHECK(
      active_num_partitions.dim() == 1 && active_num_partitions.numel() == 1,
      "active_num_partitions must have shape [1]");
  TORCH_CHECK(q.stride(-1) == 1, "q last dim must be contiguous");
  TORCH_CHECK(k_cache.stride(-1) == 1, "k_cache last dim must be contiguous");
  TORCH_CHECK(v_cache.stride(-1) == 1, "v_cache last dim must be contiguous");
  TORCH_CHECK(tmp_out.stride(-1) == 1, "tmp_out last dim must be contiguous");

  const int batch_size = q.size(0);
  const int num_heads_q = q.size(1);
  const int head_dim = q.size(2);
  const int num_heads_kv = k_cache.size(2);

  TORCH_CHECK(q.size(0) <= block_table.size(0),
              "block_table batch size must cover q batch size");
  TORCH_CHECK(q.size(0) <= seq_lens.size(0),
              "seq_lens batch size must cover q batch size");
  TORCH_CHECK(q.size(0) <= tmp_out.size(0),
              "tmp_out batch size must cover q batch size");
  TORCH_CHECK(num_heads_q == tmp_out.size(1),
              "tmp_out head dimension mismatch");
  TORCH_CHECK(head_dim == tmp_out.size(3), "tmp_out head_dim mismatch");
  TORCH_CHECK(max_logits.size(0) == tmp_out.size(0) &&
                  max_logits.size(1) == tmp_out.size(1) &&
                  max_logits.size(2) == tmp_out.size(2),
              "max_logits shape mismatch");
  TORCH_CHECK(exp_sums.sizes() == max_logits.sizes(),
              "exp_sums shape mismatch");
  TORCH_CHECK(num_heads_q % num_heads_kv == 0,
              "num_heads_q must be divisible by num_heads_kv");
  TORCH_CHECK(k_cache.size(3) == head_dim, "k_cache head_dim mismatch");
  TORCH_CHECK(v_cache.size(3) == head_dim, "v_cache head_dim mismatch");
  TORCH_CHECK(
      partition_size == 256 || partition_size == 512 || partition_size == 1024,
      "Unsupported decode partition_size: ", partition_size);
  TORCH_CHECK(
      launch_num_partitions > 0 && launch_num_partitions <= tmp_out.size(2),
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

#define LAUNCH_TYPED(HDIM, PARTITION, KV_DTYPE_CODE)                         \
  launch_flash_attention_decode_paged<HDIM, PARTITION, KV_DTYPE_CODE>(       \
      q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,  \
      exp_sums, active_num_partitions, softmax_scale, launch_num_partitions, \
      k_scale, v_scale, window_size_left, window_size_right, stream)

#define LAUNCH_BY_KV_DTYPE(HDIM, PARTITION)                                 \
  do {                                                                      \
    switch (kv_dtype_code) {                                                \
      case flash_v100::KV_CACHE_DTYPE_FP16:                                 \
        LAUNCH_TYPED(HDIM, PARTITION, flash_v100::KV_CACHE_DTYPE_FP16);     \
        break;                                                              \
      case flash_v100::KV_CACHE_DTYPE_FP8_E4M3:                             \
        LAUNCH_TYPED(HDIM, PARTITION, flash_v100::KV_CACHE_DTYPE_FP8_E4M3); \
        break;                                                              \
      case flash_v100::KV_CACHE_DTYPE_FP8_E5M2:                             \
        LAUNCH_TYPED(HDIM, PARTITION, flash_v100::KV_CACHE_DTYPE_FP8_E5M2); \
        break;                                                              \
      default:                                                              \
        TORCH_CHECK(false, "Unsupported kv_cache_dtype: ", kv_cache_dtype); \
    }                                                                       \
  } while (0)

#define LAUNCH_BY_PARTITION(HDIM)                                           \
  do {                                                                      \
    switch (partition_size) {                                               \
      case 256:                                                             \
        LAUNCH_BY_KV_DTYPE(HDIM, 256);                                      \
        break;                                                              \
      case 512:                                                             \
        LAUNCH_BY_KV_DTYPE(HDIM, 512);                                      \
        break;                                                              \
      case 1024:                                                            \
        LAUNCH_BY_KV_DTYPE(HDIM, 1024);                                     \
        break;                                                              \
      default:                                                              \
        TORCH_CHECK(false,                                                  \
                    "Unsupported decode partition_size: ", partition_size); \
    }                                                                       \
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

at::Tensor flash_attention_decode_paged_xqa(
    const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache,
    std::optional<at::Tensor>& out_, const at::Tensor& block_table,
    const at::Tensor& seq_lens, at::Tensor& tmp_out, at::Tensor& max_logits,
    at::Tensor& exp_sums, const at::Tensor& active_num_partitions,
    const float softmax_scale, const int partition_size,
    const int launch_num_partitions, const std::string& kv_cache_dtype,
    const float k_scale, const float v_scale, const int window_size_left,
    const int window_size_right) {
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
  const int kv_dtype_code = kv_cache_dtype_code_from_string(kv_cache_dtype);
  TORCH_CHECK(kv_dtype_code == flash_v100::KV_CACHE_DTYPE_FP16 ||
                  kv_dtype_code == flash_v100::KV_CACHE_DTYPE_FP8_E5M2,
              "XQA decode supports fp16 and fp8_e5m2 KV cache only");
  if (kv_dtype_code == flash_v100::KV_CACHE_DTYPE_FP16) {
    TORCH_CHECK(k_cache.dtype() == torch::kFloat16 &&
                    v_cache.dtype() == torch::kFloat16,
                "fp16 XQA requires fp16 K/V tensors");
  } else {
    TORCH_CHECK(
        k_cache.dtype() == torch::kUInt8 && v_cache.dtype() == torch::kUInt8,
        "fp8_e5m2 XQA requires uint8 K/V tensors");
  }
  TORCH_CHECK(k_scale > 0.f && v_scale > 0.f,
              "XQA K/V scales must be positive");
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
  TORCH_CHECK(
      active_num_partitions.dim() == 1 && active_num_partitions.numel() == 1,
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
  TORCH_CHECK(
      partition_size == 256 || partition_size == 512 || partition_size == 1024,
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
  TORCH_CHECK(exp_sums.size(0) >= q.size(0) && exp_sums.size(1) >= q.size(1) &&
                  exp_sums.size(2) >= launch_num_partitions,
              "exp_sums shape does not cover XQA launch");

  c10::cuda::CUDAGuard device_guard(q.device());
  at::Tensor out = out_.has_value() ? out_.value() : torch::empty_like(q);
  TORCH_CHECK(out.is_cuda(), "out must be on CUDA");
  TORCH_CHECK(out.dtype() == torch::kFloat16, "out must be fp16");
  TORCH_CHECK(out.sizes() == q.sizes(), "out must have same shape as q");
  TORCH_CHECK(out.stride(-1) == 1, "out last dim must be contiguous");
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  // Default only the shape with exact direct and model-level evidence.
  const bool padded_smem_enabled = xqa_padded_smem_enabled();
  const bool use_padded_smem =
      padded_smem_enabled && q_per_kv == 6 && partition_size == 256;
  const bool use_g6_dual_cta_dense = !use_padded_smem &&
                                     k_cache.size(1) == 784 &&
                                     xqa_g6_dual_cta_dense_enabled();
  const bool use_mtp5_dual_cta =
      q.size(0) == 5 && q_per_kv == 6 && k_cache.size(1) == 1616 &&
      k_cache.size(2) > 0 &&
      kv_dtype_code == flash_v100::KV_CACHE_DTYPE_FP8_E5M2 &&
      xqa_mtp5_dual_cta_enabled();
  const bool use_g6_dual_cta =
      use_mtp5_dual_cta ||
      (xqa_g6_dual_cta_enabled() && (use_padded_smem || use_g6_dual_cta_dense));
  const bool use_split_reduce = use_g6_dual_cta && xqa_split_reduce_enabled();
  const bool supports_block16_index = use_g6_dual_cta && k_cache.size(1) == 16;
  const bool supports_block16_contiguous_layout =
      supports_block16_index && k_cache.size(2) == 1 &&
      k_cache.stride(0) == 4096 && k_cache.stride(1) == 256 &&
      k_cache.stride(2) == 256 && k_cache.stride(3) == 1 &&
      v_cache.stride(0) == 4096 && v_cache.stride(1) == 256 &&
      v_cache.stride(2) == 256 && v_cache.stride(3) == 1;
  const bool use_block784_index =
      use_g6_dual_cta && k_cache.size(1) == 784 && xqa_block784_index_enabled();
  const bool use_aligned_padded_smem = use_padded_smem && use_block784_index &&
                                       xqa_aligned_padded_smem_enabled();
  const int requested_block16_layout_mode = xqa_block16_layout_mode();
  const int block16_layout_mode =
      requested_block16_layout_mode == 1 && supports_block16_index ? 1
      : requested_block16_layout_mode == 2 && supports_block16_contiguous_layout
          ? 2
          : 0;
  if (xqa_block16_layout_required() && requested_block16_layout_mode != 0) {
    TORCH_CHECK(
        block16_layout_mode == requested_block16_layout_mode,
        "Requested block16 XQA mode ", requested_block16_layout_mode,
        " but the live KV cache does not satisfy its layout gate: block_size=",
        k_cache.size(1), ", num_kv_heads=", k_cache.size(2), ", k_strides=[",
        k_cache.stride(0), ",", k_cache.stride(1), ",", k_cache.stride(2), ",",
        k_cache.stride(3), "]");
  }
  static bool traced_block16_layout = false;
  if (xqa_block16_layout_trace_enabled() && block16_layout_mode != 0 &&
      !traced_block16_layout) {
    TORCH_WARN("Flash-V100 XQA block16 mode ", block16_layout_mode,
               " active for KV shape [blocks,", k_cache.size(1), ",",
               k_cache.size(2), ",", k_cache.size(3), "]");
    traced_block16_layout = true;
  }
  static bool traced_block784_index = false;
  if (xqa_block784_index_trace_enabled() && use_block784_index &&
      !traced_block784_index) {
    TORCH_WARN(
        "Flash-V100 XQA block784 index specialization active for KV "
        "shape [blocks,",
        k_cache.size(1), ",", k_cache.size(2), ",", k_cache.size(3), "]");
    traced_block784_index = true;
  }
  static bool traced_aligned_padded_smem = false;
  if (xqa_aligned_padded_smem_trace_enabled() && use_aligned_padded_smem &&
      !traced_aligned_padded_smem) {
    TORCH_WARN("Flash-V100 XQA aligned padded shared layout active");
    traced_aligned_padded_smem = true;
  }
  const int split_reduce_dim_tile = xqa_split_reduce_dim_tile();

#define LAUNCH_XQA_WIDE(GROUP_SIZE, PARTITION)                                 \
  do {                                                                         \
    if (use_padded_smem) {                                                     \
      launch_flash_attention_decode_paged_xqa_tc_256_wide<PARTITION,           \
                                                          GROUP_SIZE, true>(   \
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,            \
          max_logits, exp_sums, active_num_partitions, softmax_scale, k_scale, \
          v_scale, launch_num_partitions, false, 8, stream);                   \
    } else {                                                                   \
      launch_flash_attention_decode_paged_xqa_tc_256_wide<PARTITION,           \
                                                          GROUP_SIZE, false>(  \
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,            \
          max_logits, exp_sums, active_num_partitions, softmax_scale, k_scale, \
          v_scale, launch_num_partitions, false, 8, stream);                   \
    }                                                                          \
  } while (0)

#define DISPATCH_PARTITION(GROUP_SIZE)                                   \
  do {                                                                   \
    switch (partition_size) {                                            \
      case 256:                                                          \
        LAUNCH_XQA_WIDE(GROUP_SIZE, 256);                                \
        break;                                                           \
      case 512:                                                          \
        LAUNCH_XQA_WIDE(GROUP_SIZE, 512);                                \
        break;                                                           \
      case 1024:                                                         \
        LAUNCH_XQA_WIDE(GROUP_SIZE, 1024);                               \
        break;                                                           \
      default:                                                           \
        TORCH_CHECK(false,                                               \
                    "Unsupported XQA partition_size: ", partition_size); \
    }                                                                    \
  } while (0)

  if (use_g6_dual_cta) {
    if (use_block784_index) {
      if (use_aligned_padded_smem) {
        launch_flash_attention_decode_paged_xqa_tc_256_wide<
            256, 6, true, kXQATCG6DualCtaThreads, 2, 784, false, true>(
            q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,
            max_logits, exp_sums, active_num_partitions, softmax_scale, k_scale,
            v_scale, launch_num_partitions, use_split_reduce,
            split_reduce_dim_tile, stream);
      } else if (use_padded_smem) {
        launch_flash_attention_decode_paged_xqa_tc_256_wide<
            256, 6, true, kXQATCG6DualCtaThreads, 2, 784, false>(
            q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,
            max_logits, exp_sums, active_num_partitions, softmax_scale, k_scale,
            v_scale, launch_num_partitions, use_split_reduce,
            split_reduce_dim_tile, stream);
      } else {
        launch_flash_attention_decode_paged_xqa_tc_256_wide<
            256, 6, false, kXQATCG6DualCtaThreads, 2, 784, false>(
            q, k_cache, v_cache, out, block_table, seq_lens, tmp_out,
            max_logits, exp_sums, active_num_partitions, softmax_scale, k_scale,
            v_scale, launch_num_partitions, use_split_reduce,
            split_reduce_dim_tile, stream);
      }
    } else if (block16_layout_mode == 2) {
      launch_flash_attention_decode_paged_xqa_tc_256_wide<
          256, 6, true, kXQATCG6DualCtaThreads, 2, 16, true>(
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
          exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
          launch_num_partitions, use_split_reduce, split_reduce_dim_tile,
          stream);
    } else if (block16_layout_mode == 1) {
      launch_flash_attention_decode_paged_xqa_tc_256_wide<
          256, 6, true, kXQATCG6DualCtaThreads, 2, 16, false>(
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
          exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
          launch_num_partitions, use_split_reduce, split_reduce_dim_tile,
          stream);
    } else if (partition_size == 256) {
      launch_flash_attention_decode_paged_xqa_tc_256_wide<
          256, 6, true, kXQATCG6DualCtaThreads, 2, 0, false>(
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
          exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
          launch_num_partitions, use_split_reduce, split_reduce_dim_tile,
          stream);
    } else if (partition_size == 512) {
      launch_flash_attention_decode_paged_xqa_tc_256_wide<
          512, 6, false, kXQATCG6DualCtaThreads, 2, 0, false>(
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
          exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
          launch_num_partitions, use_split_reduce, split_reduce_dim_tile,
          stream);
    } else {
      launch_flash_attention_decode_paged_xqa_tc_256_wide<
          1024, 6, false, kXQATCG6DualCtaThreads, 2, 0, false>(
          q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
          exp_sums, active_num_partitions, softmax_scale, k_scale, v_scale,
          launch_num_partitions, use_split_reduce, split_reduce_dim_tile,
          stream);
    }
  } else if (q_per_kv == 4) {
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

at::Tensor flash_attention_decode_paged_xqa_staged(
    const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache,
    std::optional<at::Tensor>& out_, const at::Tensor& block_table,
    const at::Tensor& seq_lens, at::Tensor& tmp_out, at::Tensor& max_logits,
    at::Tensor& exp_sums, at::Tensor& online_rescales,
    const at::Tensor& active_num_partitions, const float softmax_scale,
    const int partition_size, const int launch_num_partitions,
    const std::string& kv_cache_dtype, const float k_scale, const float v_scale,
    const int window_size_left, const int window_size_right) {
  (void)k_scale;
  (void)v_scale;
  TORCH_CHECK(q.is_cuda(), "q must be on CUDA");
  TORCH_CHECK(k_cache.is_cuda() && v_cache.is_cuda(),
              "k_cache and v_cache must be on CUDA");
  TORCH_CHECK(block_table.is_cuda() && seq_lens.is_cuda(),
              "block_table and seq_lens must be on CUDA");
  TORCH_CHECK(tmp_out.is_cuda() && max_logits.is_cuda() && exp_sums.is_cuda() &&
                  online_rescales.is_cuda(),
              "staged XQA workspaces must be on CUDA");
  TORCH_CHECK(active_num_partitions.is_cuda(),
              "active_num_partitions must be on CUDA");
  TORCH_CHECK(q.dtype() == torch::kFloat16, "q must be fp16");
  TORCH_CHECK(
      k_cache.dtype() == torch::kFloat16 && v_cache.dtype() == torch::kFloat16,
      "staged XQA supports fp16 KV cache only");
  TORCH_CHECK(kv_cache_dtype == "auto" || kv_cache_dtype == "bfloat16",
              "staged XQA supports fp16 KV cache only");
  TORCH_CHECK(block_table.dtype() == torch::kInt32,
              "block_table must be int32");
  TORCH_CHECK(seq_lens.dtype() == torch::kInt32, "seq_lens must be int32");
  TORCH_CHECK(active_num_partitions.dtype() == torch::kInt32,
              "active_num_partitions must be int32");
  TORCH_CHECK(window_size_left == -1 && window_size_right == -1,
              "staged XQA does not support sliding-window attention");
  TORCH_CHECK(q.dim() == 3 && k_cache.dim() == 4 && v_cache.dim() == 4,
              "staged XQA expects q [B,H,D] and paged KV [blocks,T,H,D]");
  TORCH_CHECK(block_table.dim() == 2 && seq_lens.dim() == 1,
              "staged XQA block_table/seq_lens rank mismatch");
  TORCH_CHECK(
      active_num_partitions.dim() == 1 && active_num_partitions.numel() == 1,
      "active_num_partitions must have shape [1]");
  TORCH_CHECK(
      q.stride(-1) == 1 && k_cache.stride(-1) == 1 && v_cache.stride(-1) == 1,
      "staged XQA requires contiguous head dimensions");
  TORCH_CHECK(q.size(0) <= block_table.size(0) && q.size(0) <= seq_lens.size(0),
              "staged XQA metadata batch capacity is too small");
  TORCH_CHECK(k_cache.sizes() == v_cache.sizes(), "K/V cache shape mismatch");
  TORCH_CHECK(q.size(2) == 256 && k_cache.size(3) == 256,
              "staged XQA supports D=256 only");
  const int num_heads_q = q.size(1);
  const int num_heads_kv = k_cache.size(2);
  TORCH_CHECK(num_heads_kv > 0 && num_heads_q == 6 * num_heads_kv,
              "staged XQA supports q_per_kv=6 only");
  TORCH_CHECK(partition_size == 256, "staged XQA supports p256 only");
  TORCH_CHECK(launch_num_partitions > 0,
              "launch_num_partitions must be positive");
  TORCH_CHECK(tmp_out.dtype() == torch::kFloat16 &&
                  tmp_out.size(0) >= q.size(0) &&
                  tmp_out.size(1) >= q.size(1) &&
                  tmp_out.size(2) >= launch_num_partitions &&
                  tmp_out.size(3) == q.size(2),
              "tmp_out shape does not cover staged XQA launch");
  TORCH_CHECK(max_logits.dtype() == torch::kFloat32 &&
                  max_logits.size(0) >= q.size(0) &&
                  max_logits.size(1) >= q.size(1) &&
                  max_logits.size(2) >= launch_num_partitions,
              "max_logits shape does not cover staged XQA launch");
  TORCH_CHECK(exp_sums.sizes() == max_logits.sizes(),
              "exp_sums shape mismatch");
  TORCH_CHECK(online_rescales.dtype() == torch::kFloat32 &&
                  online_rescales.size(0) >= q.size(0) &&
                  online_rescales.size(1) >= q.size(1) &&
                  online_rescales.size(2) >= launch_num_partitions,
              "online_rescales shape does not cover staged XQA launch");

  c10::cuda::CUDAGuard device_guard(q.device());
  at::Tensor out = out_.has_value() ? out_.value() : torch::empty_like(q);
  TORCH_CHECK(out.is_cuda() && out.dtype() == torch::kFloat16 &&
                  out.sizes() == q.sizes() && out.stride(-1) == 1,
              "staged XQA output must be contiguous fp16 q-shaped CUDA tensor");
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  launch_flash_attention_decode_paged_xqa_tc_256_staged(
      q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits,
      exp_sums, online_rescales, active_num_partitions, softmax_scale,
      launch_num_partitions, xqa_split_reduce_enabled(),
      xqa_split_reduce_dim_tile(), stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor flash_attention_decode_qk_scores(
    const at::Tensor& q, const at::Tensor& k_cache,
    const at::Tensor& block_table, const at::Tensor& seq_lens,
    const float softmax_scale, const int partition_size,
    const std::string& kv_cache_dtype, const float k_scale) {
  TORCH_CHECK(q.is_cuda(), "q must be on CUDA");
  TORCH_CHECK(k_cache.is_cuda(), "k_cache must be on CUDA");
  TORCH_CHECK(block_table.is_cuda() && seq_lens.is_cuda(),
              "block_table and seq_lens must be on CUDA");
  TORCH_CHECK(q.dtype() == torch::kFloat16, "q must be fp16");
  const int kv_dtype_code = kv_cache_dtype_code_from_string(kv_cache_dtype);
  TORCH_CHECK(kv_dtype_code >= 0,
              "Unsupported kv_cache_dtype: ", kv_cache_dtype);
  if (kv_dtype_code == flash_v100::KV_CACHE_DTYPE_FP16) {
    TORCH_CHECK(k_cache.dtype() == torch::kFloat16, "k_cache must be fp16");
  } else {
    TORCH_CHECK(k_cache.dtype() == torch::kUInt8,
                "fp8 k_cache must be stored as uint8");
    TORCH_CHECK(k_scale > 0.f, "fp8 k scale must be positive");
  }
  TORCH_CHECK(block_table.dtype() == torch::kInt32,
              "block_table must be int32");
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
  TORCH_CHECK(
      partition_size == 256 || partition_size == 512 || partition_size == 1024,
      "Unsupported decode partition_size: ", partition_size);

  c10::cuda::CUDAGuard device_guard(q.device());
  auto scores =
      torch::full({batch_size, num_heads_q, max_num_partitions, partition_size},
                  -1.0e30f, q.options().dtype(torch::kFloat32));

  auto stream = at::cuda::getCurrentCUDAStream().stream();

#define LAUNCH_TYPED(HDIM, PARTITION, KV_DTYPE_CODE)                       \
  launch_flash_attention_decode_qk_scores<HDIM, PARTITION, KV_DTYPE_CODE>( \
      q, k_cache, block_table, seq_lens, scores, softmax_scale, k_scale,   \
      stream)

#define LAUNCH_BY_KV_DTYPE(HDIM, PARTITION)                                 \
  do {                                                                      \
    switch (kv_dtype_code) {                                                \
      case flash_v100::KV_CACHE_DTYPE_FP16:                                 \
        LAUNCH_TYPED(HDIM, PARTITION, flash_v100::KV_CACHE_DTYPE_FP16);     \
        break;                                                              \
      case flash_v100::KV_CACHE_DTYPE_FP8_E4M3:                             \
        LAUNCH_TYPED(HDIM, PARTITION, flash_v100::KV_CACHE_DTYPE_FP8_E4M3); \
        break;                                                              \
      case flash_v100::KV_CACHE_DTYPE_FP8_E5M2:                             \
        LAUNCH_TYPED(HDIM, PARTITION, flash_v100::KV_CACHE_DTYPE_FP8_E5M2); \
        break;                                                              \
      default:                                                              \
        TORCH_CHECK(false, "Unsupported kv_cache_dtype: ", kv_cache_dtype); \
    }                                                                       \
  } while (0)

#define LAUNCH_BY_PARTITION(HDIM)                                           \
  do {                                                                      \
    switch (partition_size) {                                               \
      case 256:                                                             \
        LAUNCH_BY_KV_DTYPE(HDIM, 256);                                      \
        break;                                                              \
      case 512:                                                             \
        LAUNCH_BY_KV_DTYPE(HDIM, 512);                                      \
        break;                                                              \
      case 1024:                                                            \
        LAUNCH_BY_KV_DTYPE(HDIM, 1024);                                     \
        break;                                                              \
      default:                                                              \
        TORCH_CHECK(false,                                                  \
                    "Unsupported decode partition_size: ", partition_size); \
    }                                                                       \
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
