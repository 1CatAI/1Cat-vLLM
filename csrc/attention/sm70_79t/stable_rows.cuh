// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026, 1CatAI.

// The raw endpoint assumes zero-shift exponentials and FP16 numerators fit.
// Model scores violate both assumptions. Keep block masses and the online
// accumulator in FP32, and bound each FP16 PV operation by scaling V.
__device__ float const* g_79t_tail_row_max = nullptr;
constexpr float kStableScoreMargin = 4.0f;
constexpr float kStableValueCenterThreshold = 0.05f;
constexpr float kStableMaxExpInput = 10.0f;
constexpr float kStableValueHeadroom = 64.0f;

#if defined(PREFIX_TORCH_PREFIX_FP32_OUTPUT)
using StablePrefixPartial = float;
__device__ __forceinline__ float stable_partial_to_float(float value) {
  return value;
}
#else
using StablePrefixPartial = __half;
__device__ __forceinline__ float stable_partial_to_float(__half value) {
  return __half2float(value);
}
#endif

__device__ __forceinline__ float stable_exp(float value, float maximum) {
  return exp2f(fminf(value - maximum, kStableMaxExpInput) *
               1.4426950408889634f);
}

__device__ __forceinline__ int stable_tail_query_tile() {
  constexpr int kTailTiles =
      PREFIX_TORCH_QUERY_TOKENS / PREFIX_BATCHED_TAIL_TILE_TOKENS;
  constexpr int kGroupTiles = PREFIX_TAIL_FINE_PV_GROUP_TILES;
  int task = pv_task_index();
  int first = 0;
  int tasks = kTailTiles;
  while (task >= tasks) {
    task -= tasks;
    first += kGroupTiles;
    tasks -= kGroupTiles;
  }
  return first + task;
}

__device__ __forceinline__ float stable_value_scale(float maximum) {
  maximum = fmaxf(maximum, 1.0f);
  int exponent;
  float mantissa = frexpf(maximum, &exponent);
  return ldexpf(1.0f, exponent - (mantissa == 0.5f)) * kStableValueHeadroom;
}

__global__ void stable_value_center(__half const* values, float* center,
                                    int total_kv) {
  int d = threadIdx.x;
  if (d >= 256) return;
  int samples = min(total_kv, 4096);
  float sum = 0.0f;
  for (int token = 0; token < samples; ++token)
    sum += __half2float(values[int64_t(token) * 256 + d]);
  float mean = sum / samples;
  center[d] = fabsf(mean) >= kStableValueCenterThreshold ? mean : 0.0f;
}

__global__ void stable_value_amax(__half const* values, float const* center,
                                  float* maximum, int elements) {
  float local = 0.0f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements;
       i += blockDim.x * gridDim.x) {
    local = fmaxf(local, fabsf(__half2float(values[i]) - center[i & 255]));
  }
  for (int offset = 16; offset; offset >>= 1)
    local = fmaxf(local, __shfl_down_sync(0xffffffffu, local, offset));
  __shared__ float warp_max[8];
  if ((threadIdx.x & 31) == 0) warp_max[threadIdx.x >> 5] = local;
  __syncthreads();
  if (threadIdx.x == 0) {
    float value = 0.0f;
    for (int i = 0; i < 8; ++i) value = fmaxf(value, warp_max[i]);
    atomicMax(reinterpret_cast<unsigned int*>(maximum), __float_as_uint(value));
  }
}

__global__ void stable_scale_values(__half const* input, __half* output,
                                    float const* center, float const* maximum,
                                    int elements) {
  __shared__ float inverse;
  if (threadIdx.x == 0) inverse = 1.0f / stable_value_scale(*maximum);
  __syncthreads();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements;
       i += blockDim.x * gridDim.x)
    output[i] =
        __float2half_rn((__half2float(input[i]) - center[i & 255]) * inverse);
}

// Each lane reads a pair of adjacent query rows. K tiles stay independent,
// preserving coalesced loads from the transposed cuBLAS score workspace.
template <bool Tail>
__global__ void stable_row_max_partials(__half const* scores, float* partials,
                                        int rows, int width) {
  int row = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= rows) return;
  int stride = rows;
  int local_row = row;
  int64_t base = 0;
  if constexpr (Tail) {
    constexpr int tile_tokens = PREFIX_BATCHED_TAIL_TILE_TOKENS;
    constexpr int tile_rows = tile_tokens * 6;
    int tile = row / tile_rows;
    local_row = row % tile_rows;
    stride = tile_rows;
    width = (tile + 1) * tile_tokens;
    base = int64_t(tile_rows) * tile_tokens * tile * (tile + 1) / 2;
  }
  float2 maximum = {-CUDART_INF_F, -CUDART_INF_F};
  int end = min(width, int(blockIdx.y + 1) * 8192);
#pragma unroll 4
  // Every score participates: subsampling can miss isolated large logits and
  // make the exponent guard change the attention distribution.
  for (int col = int(blockIdx.y) * 8192; col < end; ++col) {
    float2 value = __half22float2(*reinterpret_cast<__half2 const*>(
        scores + base + int64_t(col) * stride + local_row));
    maximum.x = fmaxf(maximum.x, value.x);
    maximum.y = fmaxf(maximum.y, value.y);
  }
  int64_t offset = int64_t(blockIdx.y) * rows + row;
  partials[offset] = maximum.x;
  partials[offset + 1] = maximum.y;
}

__global__ void stable_finish_max(float const* partials, float* maxima,
                                  int rows, int tiles) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  float value = -CUDART_INF_F;
  for (int tile = 0; tile < tiles; ++tile)
    value = fmaxf(value, partials[int64_t(tile) * rows + row]);
  maxima[row] = value + kStableScoreMargin;
}

__global__ void stable_merge_prefix(StablePrefixPartial const* partial,
                                    float* block_sum, float const* block_max,
                                    float* accumulator, float* sum,
                                    float* maximum, bool first) {
  constexpr int kRowsPerBlock = 4;
  constexpr int kThreadsPerRow = 64;
  int group = threadIdx.x / kThreadsPerRow;
  int lane = threadIdx.x % kThreadsPerRow;
  int row = blockIdx.x * kRowsPerBlock + group;
  bool valid = row < g_rows;
  __shared__ float scales[kRowsPerBlock][2];
  if (lane == 0 && valid) {
    float old_max = first ? -CUDART_INF_F : maximum[row];
    float next = fmaxf(old_max, block_max[row]);
    scales[group][0] = first ? 0.0f : expf(old_max - next);
    scales[group][1] = expf(block_max[row] - next);
    sum[row] = (first ? 0.0f : sum[row] * scales[group][0]) +
               block_sum[row] * scales[group][1];
    maximum[row] = next;
    block_sum[row] = 0.0f;
  }
  __syncthreads();
  if (!valid) return;
#pragma unroll
  for (int d = lane; d < 256; d += kThreadsPerRow) {
    int64_t index = int64_t(row) * 256 + d;
    accumulator[index] =
        (first ? 0.0f : accumulator[index] * scales[group][0]) +
        stable_partial_to_float(partial[index]) * scales[group][1];
  }
}

__global__ void stable_merge_final(float const* prefix, float const* prefix_sum,
                                   float const* prefix_max, __half const* tail,
                                   float const* tail_sum, float const* tail_max,
                                   float const* value_center,
                                   float const* value_max, __half* output,
                                   int repaired_rows, bool has_prefix) {
  int row = blockIdx.x;
  int d = threadIdx.x;
  __shared__ float coefficients[3];
  if (d == 0) {
    float pm = has_prefix ? prefix_max[row] : -CUDART_INF_F;
    float tm = tail_max[row];
    float m = fmaxf(pm, tm);
    float ps = has_prefix ? expf(pm - m) : 0.0f;
    float ts = expf(tm - m);
    float mass =
        (has_prefix ? prefix_sum[row] * ps : 0.0f) + tail_sum[row] * ts;
    coefficients[0] = ps;
    coefficients[1] = row < repaired_rows ? ts * tail_sum[row] : ts;
    coefficients[2] = stable_value_scale(*value_max) / mass;
  }
  __syncthreads();
  int64_t index = int64_t(row) * 256 + d;
  float p = has_prefix ? prefix[index] * coefficients[0] : 0.0f;
  output[index] = __float2half_rn(
      (p + __half2float(tail[index]) * coefficients[1]) * coefficients[2] +
      value_center[d]);
}
