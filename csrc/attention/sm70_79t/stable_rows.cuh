// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026, 1CatAI.

// The raw endpoint assumes zero-shift exponentials and FP16 numerators fit.
// Model scores violate both assumptions. Keep block masses and the online
// accumulator in FP32, and bound each FP16 PV operation by scaling V.
__device__ float const* g_79t_tail_row_max = nullptr;

__device__ __forceinline__ int stable_tail_query_tile() {
  int task = pv_task_index();
  int first = 0;
  int tasks = 25;
  while (task >= tasks) {
    task -= tasks;
    first += 4;
    tasks -= 4;
  }
  return first + task;
}

__device__ __forceinline__ float stable_value_scale(float maximum) {
  maximum = fmaxf(maximum, 1.0f);
  int exponent;
  float mantissa = frexpf(maximum, &exponent);
  return ldexpf(1.0f, exponent - (mantissa == 0.5f));
}

__global__ void stable_value_amax(__half const* values, float* maximum,
                                  int elements) {
  float local = 0.0f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements;
       i += blockDim.x * gridDim.x) {
    local = fmaxf(local, fabsf(__half2float(values[i])));
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
                                    float const* maximum, int elements) {
  __shared__ float inverse;
  if (threadIdx.x == 0) inverse = 1.0f / stable_value_scale(*maximum);
  __syncthreads();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements;
       i += blockDim.x * gridDim.x)
    output[i] = __float2half_rn(__half2float(input[i]) * inverse);
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
    constexpr int tile_rows = 320 * 6;
    int tile = row / tile_rows;
    local_row = row % tile_rows;
    stride = tile_rows;
    width = (tile + 1) * 320;
    base = int64_t(tile_rows) * 320 * tile * (tile + 1) / 2;
  }
  float2 maximum = {-CUDART_INF_F, -CUDART_INF_F};
  int end = min(width, int(blockIdx.y + 1) * 512);
#pragma unroll 4
  for (int col = int(blockIdx.y) * 512; col < end; ++col) {
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
  maxima[row] = value;
}

__global__ void stable_merge_prefix(__half const* partial, float* block_sum,
                                    float const* block_max, float* accumulator,
                                    float* sum, float* maximum, bool first) {
  int row = blockIdx.x;
  int d = threadIdx.x;
  __shared__ float scales[2];
  if (d == 0) {
    float old_max = first ? -CUDART_INF_F : maximum[row];
    float next = fmaxf(old_max, block_max[row]);
    scales[0] = first ? 0.0f : expf(old_max - next);
    scales[1] = expf(block_max[row] - next);
    sum[row] =
        (first ? 0.0f : sum[row] * scales[0]) + block_sum[row] * scales[1];
    maximum[row] = next;
    block_sum[row] = 0.0f;
  }
  __syncthreads();
  int64_t index = int64_t(row) * 256 + d;
  accumulator[index] = (first ? 0.0f : accumulator[index] * scales[0]) +
                       __half2float(partial[index]) * scales[1];
}

__global__ void stable_merge_final(float const* prefix, float const* prefix_sum,
                                   float const* prefix_max, __half const* tail,
                                   float const* tail_sum, float const* tail_max,
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
      (p + __half2float(tail[index]) * coefficients[1]) * coefficients[2]);
}
