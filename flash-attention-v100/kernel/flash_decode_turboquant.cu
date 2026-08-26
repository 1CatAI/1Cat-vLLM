#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

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

__device__ __forceinline__ uint16_t load_u16_le(
    const uint8_t* __restrict__ data,
    const int64_t byte_index) {
  return static_cast<uint16_t>(data[byte_index]) |
         (static_cast<uint16_t>(data[byte_index + 1]) << 8);
}

__device__ __forceinline__ float load_f16_le_float(
    const uint8_t* __restrict__ data,
    const int64_t byte_index) {
  return __half2float(__ushort_as_half(load_u16_le(data, byte_index)));
}

template<int D, int MSE_BITS, bool NORM_CORRECTION>
__device__ __forceinline__ float dot_qk_turboquant_mse(
    const float* __restrict__ q_rot,
    const uint8_t* __restrict__ kv_cache,
    const int64_t slot_base,
    const float* __restrict__ centroids,
    const int lane) {
  constexpr int kMseBytes = (D * MSE_BITS + 7) / 8;
  constexpr int kMseMask = (1 << MSE_BITS) - 1;

  float acc = 0.f;
  float centroid_norm_sq = 0.f;
  #pragma unroll
  for (int d = lane; d < D; d += kWarpSize) {
    const int bit_offset = d * MSE_BITS;
    const int byte_offset = bit_offset >> 3;
    const int bit_shift = bit_offset & 7;
    const uint16_t raw = load_u16_le(kv_cache, slot_base + byte_offset);
    const int centroid_idx = (raw >> bit_shift) & kMseMask;
    const float centroid = centroids[centroid_idx];
    acc = fmaf(q_rot[d], centroid, acc);
    if constexpr (NORM_CORRECTION) {
      centroid_norm_sq = fmaf(centroid, centroid, centroid_norm_sq);
    }
  }

  acc = warp_reduce_sum(acc);
  if constexpr (NORM_CORRECTION) {
    centroid_norm_sq = warp_reduce_sum(centroid_norm_sq);
    acc *= rsqrtf(centroid_norm_sq + 1.0e-16f);
  }

  const float vec_norm = load_f16_le_float(kv_cache, slot_base + kMseBytes);
  return acc * vec_norm;
}

template<int D, int MSE_BITS, int VQB>
__device__ __forceinline__ float load_turboquant_value_float(
    const uint8_t* __restrict__ kv_cache,
    const int64_t slot_base,
    const int d) {
  constexpr int kMseBytes = (D * MSE_BITS + 7) / 8;
  constexpr int kKeyPackedSize = kMseBytes + 2;
  constexpr int kValueDataBytes = (D * VQB + 7) / 8;
  const int64_t value_base = slot_base + kKeyPackedSize;

  int q_value;
  if constexpr (VQB == 4) {
    const uint8_t raw = kv_cache[value_base + (d >> 1)];
    q_value = (raw >> ((d & 1) * 4)) & 0x0f;
  } else {
    const int bit_offset = d * 3;
    const int byte_offset = bit_offset >> 3;
    const int bit_shift = bit_offset & 7;
    const uint16_t raw = load_u16_le(kv_cache, value_base + byte_offset);
    q_value = (raw >> bit_shift) & 0x07;
  }

  const int64_t scale_base = value_base + kValueDataBytes;
  const float scale = load_f16_le_float(kv_cache, scale_base);
  const float zero = load_f16_le_float(kv_cache, scale_base + 2);
  return static_cast<float>(q_value) * scale + zero;
}

template<int D, int PARTITION_SIZE, int MSE_BITS, int VQB, bool NORM_CORRECTION>
__global__ void flash_attention_turboquant_decode_partition_kernel(
    const float* __restrict__ q_rot,
    const uint8_t* __restrict__ kv_cache,
    float* __restrict__ tmp_out,
    float* __restrict__ max_logits,
    float* __restrict__ exp_sums,
    const int* __restrict__ block_table,
    const int* __restrict__ seq_lens,
    const float* __restrict__ centroids,
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
    const int64_t cache_block_stride,
    const int64_t cache_token_stride,
    const int64_t cache_head_stride,
    const float softmax_scale) {
  const int batch_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;

  if (batch_idx >= batch_size || head_idx >= num_heads_q ||
      partition_idx >= max_num_partitions) {
    return;
  }

  const int seq_len = seq_lens[batch_idx];
  const int split_len =
      (seq_len + max_num_partitions - 1) / max_num_partitions;
  const int start_token_idx = partition_idx * split_len;
  if (seq_len <= 0 || start_token_idx >= seq_len) {
    return;
  }

  const int part_tokens = min(split_len, seq_len - start_token_idx);
  const int q_per_kv = num_heads_q / num_heads_kv;
  const int kv_head_idx = head_idx / q_per_kv;
  const int lane = threadIdx.x % kWarpSize;
  const int warp_idx = threadIdx.x / kWarpSize;

  __shared__ float q_shared[D];
  __shared__ float scores_shared[PARTITION_SIZE];
  __shared__ int block_idx_shared[PARTITION_SIZE];
  __shared__ int block_offset_shared[PARTITION_SIZE];

  const float* q_ptr = q_rot + static_cast<int64_t>(batch_idx) * q_stride0 +
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
    const int64_t slot_base =
        static_cast<int64_t>(physical_block) * cache_block_stride +
        static_cast<int64_t>(block_offset) * cache_token_stride +
        static_cast<int64_t>(kv_head_idx) * cache_head_stride;

    float score = dot_qk_turboquant_mse<D, MSE_BITS, NORM_CORRECTION>(
        q_shared, kv_cache, slot_base, centroids, lane);
    if (lane == 0) {
      score *= softmax_scale;
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
      const int64_t slot_base =
          static_cast<int64_t>(physical_block) * cache_block_stride +
          static_cast<int64_t>(block_offset) * cache_token_stride +
          static_cast<int64_t>(kv_head_idx) * cache_head_stride;
      const float value = load_turboquant_value_float<D, MSE_BITS, VQB>(
          kv_cache, slot_base, d);
      acc = fmaf(scores_shared[i], value, acc);
    }
    tmp_out[tmp_out_base + d] = acc * inv_part_sum;
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
__global__ void flash_attention_turboquant_decode_reduce_kernel(
    const float* __restrict__ tmp_out,
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

  if (seq_len <= 0) {
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
      out[static_cast<int64_t>(batch_idx) * out_stride0 +
          static_cast<int64_t>(head_idx) * out_stride1 + d] =
          __float2half(0.f);
    }
    return;
  }
  const int split_len =
      (seq_len + max_num_partitions - 1) / max_num_partitions;
  const int num_partitions = (seq_len + split_len - 1) / split_len;

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
          tmp_out[tmp_out_base + static_cast<int64_t>(i) * tmp_out_stride2 + d],
          acc);
    }
    out[out_base + d] = __float2half(acc * inv_global_sum);
  }
}

template<int D, int PARTITION_SIZE, int MSE_BITS, int VQB, bool NORM_CORRECTION>
void launch_flash_attention_turboquant_decode_paged(
    const at::Tensor& q_rot,
    const at::Tensor& kv_cache,
    at::Tensor& out,
    const at::Tensor& block_table,
    const at::Tensor& seq_lens,
    at::Tensor& tmp_out,
    at::Tensor& max_logits,
    at::Tensor& exp_sums,
    const at::Tensor& centroids,
    const float softmax_scale,
    cudaStream_t stream) {
  const int batch_size = q_rot.size(0);
  const int num_heads_q = q_rot.size(1);
  const int num_heads_kv = kv_cache.size(2);
  const int block_size = kv_cache.size(1);
  const int max_num_blocks = block_table.size(1);
  const int max_num_partitions = tmp_out.size(2);

  const dim3 partition_grid(batch_size, num_heads_q, max_num_partitions);
  const dim3 reduce_grid(batch_size, num_heads_q, 1);
  const dim3 block(kThreadsPerBlock);
  const size_t reduce_shared_mem =
      static_cast<size_t>(2 * max_num_partitions) * sizeof(float);

  flash_attention_turboquant_decode_partition_kernel<
      D, PARTITION_SIZE, MSE_BITS, VQB, NORM_CORRECTION>
      <<<partition_grid, block, 0, stream>>>(
      q_rot.data_ptr<float>(),
      kv_cache.data_ptr<uint8_t>(),
      tmp_out.data_ptr<float>(),
      max_logits.data_ptr<float>(),
      exp_sums.data_ptr<float>(),
      block_table.data_ptr<int>(),
      seq_lens.data_ptr<int>(),
      centroids.data_ptr<float>(),
      batch_size,
      max_num_blocks,
      max_num_partitions,
      num_heads_q,
      num_heads_kv,
      block_size,
      q_rot.stride(0),
      q_rot.stride(1),
      tmp_out.stride(0),
      tmp_out.stride(1),
      tmp_out.stride(2),
      max_logits.stride(0),
      max_logits.stride(1),
      kv_cache.stride(0),
      kv_cache.stride(1),
      kv_cache.stride(2),
      softmax_scale);

  flash_attention_turboquant_decode_reduce_kernel<D, PARTITION_SIZE>
      <<<reduce_grid, block, reduce_shared_mem, stream>>>(
      tmp_out.data_ptr<float>(),
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

}  // namespace

at::Tensor flash_attention_turboquant_decode_paged(
    const at::Tensor& q_rot,
    const at::Tensor& kv_cache,
    std::optional<at::Tensor>& out_,
    const at::Tensor& block_table,
    const at::Tensor& seq_lens,
    at::Tensor& tmp_out,
    at::Tensor& max_logits,
    at::Tensor& exp_sums,
    const at::Tensor& centroids,
    const float softmax_scale,
    const int partition_size,
    const int mse_bits,
    const int value_quant_bits,
    const bool norm_correction) {
  TORCH_CHECK(q_rot.is_cuda(), "q_rot must be on CUDA");
  TORCH_CHECK(kv_cache.is_cuda(), "kv_cache must be on CUDA");
  TORCH_CHECK(block_table.is_cuda() && seq_lens.is_cuda(),
              "block_table and seq_lens must be on CUDA");
  TORCH_CHECK(tmp_out.is_cuda() && max_logits.is_cuda() && exp_sums.is_cuda(),
              "workspace tensors must be on CUDA");
  TORCH_CHECK(centroids.is_cuda(), "centroids must be on CUDA");
  TORCH_CHECK(q_rot.dtype() == torch::kFloat32, "q_rot must be fp32");
  TORCH_CHECK(kv_cache.dtype() == torch::kUInt8,
              "TurboQuant kv_cache must be stored as uint8");
  TORCH_CHECK(centroids.dtype() == torch::kFloat32, "centroids must be fp32");
  TORCH_CHECK(tmp_out.dtype() == torch::kFloat32, "tmp_out must be fp32");
  TORCH_CHECK(max_logits.dtype() == torch::kFloat32, "max_logits must be fp32");
  TORCH_CHECK(exp_sums.dtype() == torch::kFloat32, "exp_sums must be fp32");
  TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must be int32");
  TORCH_CHECK(seq_lens.dtype() == torch::kInt32, "seq_lens must be int32");
  TORCH_CHECK(q_rot.dim() == 3, "q_rot must have shape [B, H, D]");
  TORCH_CHECK(kv_cache.dim() == 4,
              "kv_cache must have shape [num_blocks, block_size, H_kv, slot]");
  TORCH_CHECK(block_table.dim() == 2,
              "block_table must have shape [B, max_num_blocks]");
  TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must have shape [B]");
  TORCH_CHECK(tmp_out.dim() == 4, "tmp_out must have shape [B_cap, H, P, D]");
  TORCH_CHECK(max_logits.dim() == 3, "max_logits must have shape [B_cap, H, P]");
  TORCH_CHECK(exp_sums.dim() == 3, "exp_sums must have shape [B_cap, H, P]");
  TORCH_CHECK(q_rot.stride(-1) == 1, "q_rot last dim must be contiguous");
  TORCH_CHECK(kv_cache.stride(-1) == 1, "kv_cache last dim must be contiguous");
  TORCH_CHECK(tmp_out.stride(-1) == 1, "tmp_out last dim must be contiguous");

  const int batch_size = q_rot.size(0);
  const int num_heads_q = q_rot.size(1);
  const int head_dim = q_rot.size(2);
  const int num_heads_kv = kv_cache.size(2);

  TORCH_CHECK(q_rot.size(0) <= block_table.size(0),
              "block_table batch size must cover q_rot batch size");
  TORCH_CHECK(q_rot.size(0) <= seq_lens.size(0),
              "seq_lens batch size must cover q_rot batch size");
  TORCH_CHECK(q_rot.size(0) <= tmp_out.size(0),
              "tmp_out batch size must cover q_rot batch size");
  TORCH_CHECK(num_heads_q == tmp_out.size(1), "tmp_out head dimension mismatch");
  TORCH_CHECK(head_dim == tmp_out.size(3), "tmp_out head_dim mismatch");
  TORCH_CHECK(max_logits.size(0) == tmp_out.size(0) &&
                  max_logits.size(1) == tmp_out.size(1) &&
                  max_logits.size(2) == tmp_out.size(2),
              "max_logits shape mismatch");
  TORCH_CHECK(exp_sums.sizes() == max_logits.sizes(), "exp_sums shape mismatch");
  TORCH_CHECK(num_heads_q % num_heads_kv == 0,
              "num_heads_q must be divisible by num_heads_kv");
  TORCH_CHECK(partition_size == 256 || partition_size == 512 ||
                  partition_size == 1024,
              "Unsupported decode partition_size: ", partition_size);
  TORCH_CHECK(
      block_table.size(1) * kv_cache.size(1) <= partition_size * tmp_out.size(2),
      "TurboQuant decode workspace cannot cover block_table capacity");
  TORCH_CHECK(mse_bits == 3 || mse_bits == 4,
              "TurboQuant Flash-V100 decode currently supports 3/4-bit MSE keys");
  TORCH_CHECK(value_quant_bits == 3 || value_quant_bits == 4,
              "TurboQuant Flash-V100 decode currently supports 3/4-bit values");
  TORCH_CHECK(centroids.numel() >= (1 << mse_bits),
              "centroids tensor is too small for mse_bits");

  at::Tensor out = out_.has_value()
                       ? out_.value()
                       : torch::empty({batch_size, num_heads_q, head_dim},
                                      q_rot.options().dtype(torch::kFloat16));
  TORCH_CHECK(out.is_cuda(), "out must be on CUDA");
  TORCH_CHECK(out.dtype() == torch::kFloat16, "out must be fp16");
  TORCH_CHECK(out.dim() == 3, "out must have shape [B, H, D]");
  TORCH_CHECK(out.size(0) == batch_size && out.size(1) == num_heads_q &&
                  out.size(2) == head_dim,
              "out must have same shape as q_rot");
  TORCH_CHECK(out.stride(-1) == 1, "out last dim must be contiguous");

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  c10::cuda::CUDAGuard device_guard(q_rot.device());

  #define LAUNCH_TYPED(HDIM, PARTITION, MSE, VBITS, NC)                       \
    launch_flash_attention_turboquant_decode_paged<HDIM, PARTITION, MSE,      \
                                                    VBITS, NC>(               \
        q_rot, kv_cache, out, block_table, seq_lens, tmp_out, max_logits,      \
        exp_sums, centroids, softmax_scale, stream)

  #define LAUNCH_BY_NORM(HDIM, PARTITION, MSE, VBITS)                         \
    do {                                                                      \
      if (norm_correction) {                                                  \
        LAUNCH_TYPED(HDIM, PARTITION, MSE, VBITS, true);                      \
      } else {                                                                \
        LAUNCH_TYPED(HDIM, PARTITION, MSE, VBITS, false);                     \
      }                                                                       \
    } while (0)

  #define LAUNCH_BY_VALUE(HDIM, PARTITION, MSE)                               \
    do {                                                                      \
      switch (value_quant_bits) {                                             \
        case 3:                                                               \
          LAUNCH_BY_NORM(HDIM, PARTITION, MSE, 3);                            \
          break;                                                              \
        case 4:                                                               \
          LAUNCH_BY_NORM(HDIM, PARTITION, MSE, 4);                            \
          break;                                                              \
        default:                                                              \
          TORCH_CHECK(false, "Unsupported value_quant_bits: ", value_quant_bits); \
      }                                                                       \
    } while (0)

  #define LAUNCH_BY_MSE(HDIM, PARTITION)                                      \
    do {                                                                      \
      switch (mse_bits) {                                                     \
        case 3:                                                               \
          LAUNCH_BY_VALUE(HDIM, PARTITION, 3);                                \
          break;                                                              \
        case 4:                                                               \
          LAUNCH_BY_VALUE(HDIM, PARTITION, 4);                                \
          break;                                                              \
        default:                                                              \
          TORCH_CHECK(false, "Unsupported mse_bits: ", mse_bits);            \
      }                                                                       \
    } while (0)

  #define LAUNCH_BY_PARTITION(HDIM)                                           \
    do {                                                                      \
      switch (partition_size) {                                               \
        case 256:                                                             \
          LAUNCH_BY_MSE(HDIM, 256);                                           \
          break;                                                              \
        case 512:                                                             \
          LAUNCH_BY_MSE(HDIM, 512);                                           \
          break;                                                              \
        case 1024:                                                            \
          LAUNCH_BY_MSE(HDIM, 1024);                                          \
          break;                                                              \
        default:                                                              \
          TORCH_CHECK(false, "Unsupported decode partition_size: ", partition_size); \
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
      TORCH_CHECK(false, "Unsupported head_dim for TurboQuant paged decode: ", head_dim);
  }

  #undef LAUNCH_BY_PARTITION
  #undef LAUNCH_BY_MSE
  #undef LAUNCH_BY_VALUE
  #undef LAUNCH_BY_NORM
  #undef LAUNCH_TYPED

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
