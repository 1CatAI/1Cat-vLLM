#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char* context) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(context) + ": " +
                             cudaGetErrorString(status));
  }
}

template <typename T>
__device__ __forceinline__ float load_as_float(const T* ptr, int64_t index) {
  return static_cast<float>(ptr[index]);
}

template <>
__device__ __forceinline__ float load_as_float<at::Half>(const at::Half* ptr,
                                                         int64_t index) {
  return __half2float(reinterpret_cast<const __half*>(ptr)[index]);
}

template <>
__device__ __forceinline__ float load_as_float<at::BFloat16>(
    const at::BFloat16* ptr, int64_t index) {
  return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(ptr)[index]);
}

template <typename T>
__device__ __forceinline__ void store_from_float(T* ptr,
                                                 int64_t index,
                                                 float value) {
  ptr[index] = static_cast<T>(value);
}

template <>
__device__ __forceinline__ void store_from_float<at::Half>(at::Half* ptr,
                                                           int64_t index,
                                                           float value) {
  reinterpret_cast<__half*>(ptr)[index] = __float2half(value);
}

template <>
__device__ __forceinline__ void store_from_float<at::BFloat16>(
    at::BFloat16* ptr,
    int64_t index,
    float value) {
  reinterpret_cast<__nv_bfloat16*>(ptr)[index] = __float2bfloat16(value);
}

__device__ __forceinline__ float subgroup_sum(float value, int width) {
  constexpr unsigned mask = 0xffffffffU;
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(mask, value, offset, width);
  }
  return value;
}

__device__ __forceinline__ float subgroup_broadcast(float value, int width) {
  return __shfl_sync(0xffffffffU, value, 0, width);
}

template <typename scalar_t,
          typename gate_t,
          typename beta_t,
          typename state_t,
          int K,
          int V,
          int COLS,
          int WIDTH,
          bool GateIsExp>
__global__ void gdn_forward_kernel(const scalar_t* __restrict__ q,
                                   const scalar_t* __restrict__ k,
                                   const scalar_t* __restrict__ v,
                                   const gate_t* __restrict__ gate,
                                   const beta_t* __restrict__ beta,
                                   const state_t* __restrict__ initial_state,
                                   scalar_t* __restrict__ output,
                                   float* __restrict__ final_state,
                                   int batch,
                                   int tokens,
                                   int q_heads,
                                   int v_heads,
                                   float scale,
                                   bool gate_is_exp) {
  static_assert(K % WIDTH == 0);
  constexpr int subgroups_per_warp = 32 / WIDTH;
  constexpr int rows_per_lane = K / WIDTH;

  const int hv = blockIdx.x;
  const int b = blockIdx.y;
  const int subgroup = threadIdx.x / WIDTH;
  const int lane = threadIdx.x % WIDTH;
  const int group_base =
      (blockIdx.z * blockDim.y + threadIdx.y) * subgroups_per_warp + subgroup;
  const int col_base = group_base * COLS;
  const int hq = hv / (v_heads / q_heads);

  float state_shard[COLS][rows_per_lane];

#pragma unroll
  for (int c = 0; c < COLS; ++c) {
    const int col = col_base + c;
#pragma unroll
    for (int r = 0; r < rows_per_lane; ++r) {
      const int row = r * WIDTH + lane;
      float value = 0.0F;
      if (col < V) {
        const int64_t state_index =
            (((static_cast<int64_t>(b) * v_heads + hv) * K + row) * V) + col;
        value = initial_state == nullptr ? 0.0F
                                         : load_as_float(initial_state, state_index);
      }
      state_shard[c][r] = value;
    }
  }

  for (int t = 0; t < tokens; ++t) {
    const int64_t gate_index =
        ((static_cast<int64_t>(b) * tokens + t) * v_heads + hv);
    float gate_value = 0.0F;
    float beta_value = 0.0F;
    if (threadIdx.x == 0) {
      const float gate_raw = load_as_float(gate, gate_index);
      gate_value = GateIsExp ? gate_raw : __expf(gate_raw);
      beta_value = load_as_float(beta, gate_index);
    }
    gate_value = __shfl_sync(0xffffffffU, gate_value, 0);
    beta_value = __shfl_sync(0xffffffffU, beta_value, 0);

    float k_reg[rows_per_lane];
    float q_reg[rows_per_lane];
    float kv_partial[COLS];
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      kv_partial[c] = 0.0F;
    }

#pragma unroll
    for (int r = 0; r < rows_per_lane; ++r) {
      const int row = r * WIDTH + lane;
      const int64_t qk_index =
          (((static_cast<int64_t>(b) * tokens + t) * q_heads + hq) * K) + row;
      const float q_value = load_as_float(q, qk_index);
      const float k_value = load_as_float(k, qk_index);
      q_reg[r] = q_value;
      k_reg[r] = k_value;
#pragma unroll
      for (int c = 0; c < COLS; ++c) {
        kv_partial[c] += state_shard[c][r] * k_value;
      }
    }

    float delta[COLS];
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      const int col = col_base + c;
      const float kv_col = subgroup_sum(kv_partial[c], WIDTH);
      float delta_value = 0.0F;
      if (lane == 0 && col < V) {
        const int64_t v_index =
            (((static_cast<int64_t>(b) * tokens + t) * v_heads + hv) * V) + col;
        delta_value =
            (load_as_float(v, v_index) - gate_value * kv_col) * beta_value;
      }
      delta[c] = subgroup_broadcast(delta_value, WIDTH);
    }

    float attn_partial[COLS];
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      attn_partial[c] = 0.0F;
    }

#pragma unroll
    for (int r = 0; r < rows_per_lane; ++r) {
#pragma unroll
      for (int c = 0; c < COLS; ++c) {
        const float new_state =
            fmaf(k_reg[r], delta[c], gate_value * state_shard[c][r]);
        state_shard[c][r] = new_state;
        attn_partial[c] += new_state * q_reg[r];
      }
    }

#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      attn_partial[c] = subgroup_sum(attn_partial[c], WIDTH);
    }

    if (lane == 0) {
      const int64_t out_base =
          (((static_cast<int64_t>(b) * tokens + t) * v_heads + hv) * V);
#pragma unroll
      for (int c = 0; c < COLS; ++c) {
        const int col = col_base + c;
        if (col < V) {
          store_from_float(output, out_base + col, attn_partial[c] * scale);
        }
      }
    }
  }

  if (final_state != nullptr) {
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      const int col = col_base + c;
#pragma unroll
      for (int r = 0; r < rows_per_lane; ++r) {
        const int row = r * WIDTH + lane;
        if (col < V) {
          const int64_t state_index =
              (((static_cast<int64_t>(b) * v_heads + hv) * K + row) * V) + col;
          final_state[state_index] = state_shard[c][r];
        }
      }
    }
  }
}

template <typename scalar_t,
          typename gate_t,
          typename beta_t,
          typename state_t,
          int K,
          int V,
          int COLS,
          int WIDTH,
          bool GateIsExp>
__global__ void gdn_forward_vlk_varlen_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const gate_t* __restrict__ gate,
    const beta_t* __restrict__ beta,
    const state_t* __restrict__ initial_state,
    const int32_t* __restrict__ cu_seqlens,
    scalar_t* __restrict__ output,
    float* __restrict__ final_state,
    int q_heads,
    int v_heads,
    float scale,
    bool gate_is_exp) {
  static_assert(K % WIDTH == 0);
  constexpr int subgroups_per_warp = 32 / WIDTH;
  constexpr int rows_per_lane = K / WIDTH;

  const int hv = blockIdx.x;
  const int n = blockIdx.y;
  const int subgroup = threadIdx.x / WIDTH;
  const int lane = threadIdx.x % WIDTH;
  const int group_base =
      (blockIdx.z * blockDim.y + threadIdx.y) * subgroups_per_warp + subgroup;
  const int col_base = group_base * COLS;
  const int hq = hv / (v_heads / q_heads);
  const int seq_start = cu_seqlens[n];
  const int seq_end = cu_seqlens[n + 1];

  float state_shard[COLS][rows_per_lane];

#pragma unroll
  for (int c = 0; c < COLS; ++c) {
    const int col = col_base + c;
#pragma unroll
    for (int r = 0; r < rows_per_lane; ++r) {
      const int row = r * WIDTH + lane;
      float value = 0.0F;
      if (col < V) {
        const int64_t state_index =
            (((static_cast<int64_t>(n) * v_heads + hv) * V + col) * K) + row;
        value = initial_state == nullptr ? 0.0F
                                         : load_as_float(initial_state, state_index);
      }
      state_shard[c][r] = value;
    }
  }

  for (int t = seq_start; t < seq_end; ++t) {
    const int64_t gate_index = (static_cast<int64_t>(t) * v_heads + hv);
    float gate_value = 0.0F;
    float beta_value = 0.0F;
    if (threadIdx.x == 0) {
      const float gate_raw = load_as_float(gate, gate_index);
      gate_value = GateIsExp ? gate_raw : __expf(gate_raw);
      beta_value = load_as_float(beta, gate_index);
    }
    gate_value = __shfl_sync(0xffffffffU, gate_value, 0);
    beta_value = __shfl_sync(0xffffffffU, beta_value, 0);

    float k_reg[rows_per_lane];
    float q_reg[rows_per_lane];
    float kv_partial[COLS];
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      kv_partial[c] = 0.0F;
    }

#pragma unroll
    for (int r = 0; r < rows_per_lane; ++r) {
      const int row = r * WIDTH + lane;
      const int64_t qk_index =
          ((static_cast<int64_t>(t) * q_heads + hq) * K) + row;
      const float q_value = load_as_float(q, qk_index);
      const float k_value = load_as_float(k, qk_index);
      q_reg[r] = q_value;
      k_reg[r] = k_value;
#pragma unroll
      for (int c = 0; c < COLS; ++c) {
        kv_partial[c] += state_shard[c][r] * k_value;
      }
    }

    float delta[COLS];
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      const int col = col_base + c;
      const float kv_col = subgroup_sum(kv_partial[c], WIDTH);
      float delta_value = 0.0F;
      if (lane == 0 && col < V) {
        const int64_t v_index =
            ((static_cast<int64_t>(t) * v_heads + hv) * V) + col;
        delta_value =
            (load_as_float(v, v_index) - gate_value * kv_col) * beta_value;
      }
      delta[c] = subgroup_broadcast(delta_value, WIDTH);
    }

    float attn_partial[COLS];
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      attn_partial[c] = 0.0F;
    }

#pragma unroll
    for (int r = 0; r < rows_per_lane; ++r) {
#pragma unroll
      for (int c = 0; c < COLS; ++c) {
        const float new_state =
            fmaf(k_reg[r], delta[c], gate_value * state_shard[c][r]);
        state_shard[c][r] = new_state;
        attn_partial[c] += new_state * q_reg[r];
      }
    }

#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      attn_partial[c] = subgroup_sum(attn_partial[c], WIDTH);
    }

    if (lane == 0) {
      const int64_t out_base = (static_cast<int64_t>(t) * v_heads + hv) * V;
#pragma unroll
      for (int c = 0; c < COLS; ++c) {
        const int col = col_base + c;
        if (col < V) {
          store_from_float(output, out_base + col, attn_partial[c] * scale);
        }
      }
    }
  }

  if (final_state != nullptr) {
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      const int col = col_base + c;
#pragma unroll
      for (int r = 0; r < rows_per_lane; ++r) {
        const int row = r * WIDTH + lane;
        if (col < V) {
          const int64_t state_index =
              (((static_cast<int64_t>(n) * v_heads + hv) * V + col) * K) + row;
          final_state[state_index] = state_shard[c][r];
        }
      }
    }
  }
}

template <typename scalar_t,
          typename bias_t,
          typename state_t,
          int K,
          int V,
          int COLS,
          int WIDTH>
__global__ void gdn_decode_mixed_qkv_global_state_kernel(
    const scalar_t* __restrict__ mixed_qkv,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const float* __restrict__ A_log,
    const bias_t* __restrict__ dt_bias,
    state_t* __restrict__ state,
    const int32_t* __restrict__ state_indices,
    scalar_t* __restrict__ output,
    int tokens,
    int slots,
    int64_t state_slot_stride,
    int q_heads,
    int v_heads,
    int qkv_stride,
    float scale,
    bool use_qk_l2norm) {
  static_assert(K % WIDTH == 0);
  constexpr int subgroups_per_warp = 32 / WIDTH;
  constexpr int rows_per_lane = K / WIDTH;

  const int hv = blockIdx.x;
  const int t = blockIdx.y;
  const int subgroup = threadIdx.x / WIDTH;
  const int lane = threadIdx.x % WIDTH;
  const int group_base =
      (blockIdx.z * blockDim.y + threadIdx.y) * subgroups_per_warp + subgroup;
  const int col_base = group_base * COLS;
  const int hq = hv / (v_heads / q_heads);
  const int32_t slot = state_indices[t];

  if (slot < 0 || slot >= slots) {
    if (lane == 0) {
      const int64_t out_base = (static_cast<int64_t>(t) * v_heads + hv) * V;
#pragma unroll
      for (int c = 0; c < COLS; ++c) {
        const int col = col_base + c;
        if (col < V) {
          store_from_float(output, out_base + col, 0.0F);
        }
      }
    }
    return;
  }

  float state_shard[COLS][rows_per_lane];
#pragma unroll
  for (int c = 0; c < COLS; ++c) {
    const int col = col_base + c;
#pragma unroll
    for (int r = 0; r < rows_per_lane; ++r) {
      const int row = r * WIDTH + lane;
      float value = 0.0F;
      if (col < V) {
        const int64_t state_index =
            static_cast<int64_t>(slot) * state_slot_stride +
            ((static_cast<int64_t>(hv) * V + col) * K) + row;
        value = load_as_float(state, state_index);
      }
      state_shard[c][r] = value;
    }
  }

  const int64_t mixed_base = static_cast<int64_t>(t) * qkv_stride;
  float k_reg[rows_per_lane];
  float q_reg[rows_per_lane];
  float q_norm = 0.0F;
  float k_norm = 0.0F;
#pragma unroll
  for (int r = 0; r < rows_per_lane; ++r) {
    const int row = r * WIDTH + lane;
    const int64_t q_index = mixed_base + hq * K + row;
    const int64_t k_index = mixed_base + q_heads * K + hq * K + row;
    const float q_value = load_as_float(mixed_qkv, q_index);
    const float k_value = load_as_float(mixed_qkv, k_index);
    q_reg[r] = q_value;
    k_reg[r] = k_value;
    q_norm += q_value * q_value;
    k_norm += k_value * k_value;
  }

  if (use_qk_l2norm) {
    q_norm = subgroup_broadcast(subgroup_sum(q_norm, WIDTH), WIDTH);
    k_norm = subgroup_broadcast(subgroup_sum(k_norm, WIDTH), WIDTH);
    const float q_inv = rsqrtf(q_norm + 1.0e-6F);
    const float k_inv = rsqrtf(k_norm + 1.0e-6F);
#pragma unroll
    for (int r = 0; r < rows_per_lane; ++r) {
      q_reg[r] *= q_inv;
      k_reg[r] *= k_inv;
    }
  }

  const int64_t gate_index = static_cast<int64_t>(t) * v_heads + hv;
  float gate_value = 0.0F;
  float beta_value = 0.0F;
  if (threadIdx.x == 0) {
    const float x = load_as_float(a, gate_index) + load_as_float(dt_bias, hv);
    const float softplus_x =
        x <= 20.0F ? log1pf(__expf(x)) : x;
    const float g_value = -__expf(A_log[hv]) * softplus_x;
    gate_value = __expf(g_value);
    const float b_value = load_as_float(b, gate_index);
    beta_value = 1.0F / (1.0F + __expf(-b_value));
  }
  gate_value = __shfl_sync(0xffffffffU, gate_value, 0);
  beta_value = __shfl_sync(0xffffffffU, beta_value, 0);

  float kv_partial[COLS];
#pragma unroll
  for (int c = 0; c < COLS; ++c) {
    kv_partial[c] = 0.0F;
  }
#pragma unroll
  for (int r = 0; r < rows_per_lane; ++r) {
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      kv_partial[c] += state_shard[c][r] * k_reg[r];
    }
  }

  float delta[COLS];
#pragma unroll
  for (int c = 0; c < COLS; ++c) {
    const int col = col_base + c;
    const float kv_col = subgroup_sum(kv_partial[c], WIDTH);
    float delta_value = 0.0F;
    if (lane == 0 && col < V) {
      const int64_t v_index =
          mixed_base + 2 * q_heads * K + hv * V + col;
      delta_value =
          (load_as_float(mixed_qkv, v_index) - gate_value * kv_col) * beta_value;
    }
    delta[c] = subgroup_broadcast(delta_value, WIDTH);
  }

  float attn_partial[COLS];
#pragma unroll
  for (int c = 0; c < COLS; ++c) {
    attn_partial[c] = 0.0F;
  }
#pragma unroll
  for (int r = 0; r < rows_per_lane; ++r) {
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      const float new_state =
          fmaf(k_reg[r], delta[c], gate_value * state_shard[c][r]);
      state_shard[c][r] = new_state;
      attn_partial[c] += new_state * q_reg[r];
    }
  }
#pragma unroll
  for (int c = 0; c < COLS; ++c) {
    attn_partial[c] = subgroup_sum(attn_partial[c], WIDTH);
  }

  if (lane == 0) {
    const int64_t out_base = (static_cast<int64_t>(t) * v_heads + hv) * V;
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      const int col = col_base + c;
      if (col < V) {
        store_from_float(output, out_base + col, attn_partial[c] * scale);
      }
    }
  }

#pragma unroll
  for (int c = 0; c < COLS; ++c) {
    const int col = col_base + c;
#pragma unroll
    for (int r = 0; r < rows_per_lane; ++r) {
      const int row = r * WIDTH + lane;
      if (col < V) {
        const int64_t state_index =
            static_cast<int64_t>(slot) * state_slot_stride +
            ((static_cast<int64_t>(hv) * V + col) * K) + row;
        store_from_float(state, state_index, state_shard[c][r]);
      }
    }
  }
}

template <typename scalar_t,
          typename bias_t,
          typename state_t,
          int K,
          int V,
          int COLS,
          int WIDTH>
__global__ void gdn_decode_mixed_qkv_ddtree_state_kernel(
    const scalar_t* __restrict__ mixed_qkv,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const float* __restrict__ A_log,
    const bias_t* __restrict__ dt_bias,
    state_t* __restrict__ state,
    const int32_t* __restrict__ state_indices,
    const int32_t* __restrict__ parent_ids,
    const int32_t* __restrict__ num_accepted_tokens,
    const int32_t* __restrict__ cu_seqlens,
    scalar_t* __restrict__ output,
    int num_sequences,
    int max_state_tokens,
    int tokens,
    int slots,
    int64_t state_slot_stride,
    int q_heads,
    int v_heads,
    int qkv_stride,
    float scale,
    bool use_qk_l2norm) {
  static_assert(K % WIDTH == 0);
  constexpr int subgroups_per_warp = 32 / WIDTH;
  constexpr int rows_per_lane = K / WIDTH;

  const int hv = blockIdx.x;
  const int n = blockIdx.y;
  const int subgroup = threadIdx.x / WIDTH;
  const int lane = threadIdx.x % WIDTH;
  const int group_base =
      (blockIdx.z * blockDim.y + threadIdx.y) * subgroups_per_warp + subgroup;
  const int col_base = group_base * COLS;
  const int hq = hv / (v_heads / q_heads);
  if (n >= num_sequences) {
    return;
  }

  const int seq_start = cu_seqlens[n];
  const int seq_end = cu_seqlens[n + 1];
  const int seq_tokens = seq_end - seq_start;
  if (seq_tokens <= 0 || seq_start < 0 || seq_end > tokens) {
    return;
  }

  int selector = num_accepted_tokens[n] - 1;
  selector = selector < 0 ? 0 : selector;
  selector = selector >= max_state_tokens ? max_state_tokens - 1 : selector;
  int32_t state_idx = state_indices[n * max_state_tokens + selector];

  float state_shard[COLS][rows_per_lane];
#pragma unroll
  for (int c = 0; c < COLS; ++c) {
    const int col = col_base + c;
#pragma unroll
    for (int r = 0; r < rows_per_lane; ++r) {
      const int row = r * WIDTH + lane;
      float value = 0.0F;
      if (col < V && state_idx >= 0 && state_idx < slots) {
        const int64_t state_index =
            static_cast<int64_t>(state_idx) * state_slot_stride +
            ((static_cast<int64_t>(hv) * V + col) * K) + row;
        value = load_as_float(state, state_index);
      }
      state_shard[c][r] = value;
    }
  }

  for (int local_t = 0; local_t < seq_tokens && local_t < max_state_tokens;
       ++local_t) {
    if (local_t > 0) {
      int parent_t = parent_ids[n * max_state_tokens + local_t];
      parent_t = parent_t < 0 ? 0 : parent_t;
      parent_t =
          parent_t >= max_state_tokens ? max_state_tokens - 1 : parent_t;
      const int32_t parent_state_idx =
          state_indices[n * max_state_tokens + parent_t];
#pragma unroll
      for (int c = 0; c < COLS; ++c) {
        const int col = col_base + c;
#pragma unroll
        for (int r = 0; r < rows_per_lane; ++r) {
          const int row = r * WIDTH + lane;
          float value = 0.0F;
          if (col < V && parent_state_idx >= 0 && parent_state_idx < slots) {
            const int64_t state_index =
                static_cast<int64_t>(parent_state_idx) * state_slot_stride +
                ((static_cast<int64_t>(hv) * V + col) * K) + row;
            value = load_as_float(state, state_index);
          }
          state_shard[c][r] = value;
        }
      }
    }

    const int t = seq_start + local_t;
    const int64_t mixed_base = static_cast<int64_t>(t) * qkv_stride;
    float k_reg[rows_per_lane];
    float q_reg[rows_per_lane];
    float q_norm = 0.0F;
    float k_norm = 0.0F;
#pragma unroll
    for (int r = 0; r < rows_per_lane; ++r) {
      const int row = r * WIDTH + lane;
      const int64_t q_index = mixed_base + hq * K + row;
      const int64_t k_index = mixed_base + q_heads * K + hq * K + row;
      const float q_value = load_as_float(mixed_qkv, q_index);
      const float k_value = load_as_float(mixed_qkv, k_index);
      q_reg[r] = q_value;
      k_reg[r] = k_value;
      q_norm += q_value * q_value;
      k_norm += k_value * k_value;
    }

    if (use_qk_l2norm) {
      q_norm = subgroup_broadcast(subgroup_sum(q_norm, WIDTH), WIDTH);
      k_norm = subgroup_broadcast(subgroup_sum(k_norm, WIDTH), WIDTH);
      const float q_inv = rsqrtf(q_norm + 1.0e-6F);
      const float k_inv = rsqrtf(k_norm + 1.0e-6F);
#pragma unroll
      for (int r = 0; r < rows_per_lane; ++r) {
        q_reg[r] *= q_inv;
        k_reg[r] *= k_inv;
      }
    }

    const int64_t gate_index = static_cast<int64_t>(t) * v_heads + hv;
    float gate_value = 0.0F;
    float beta_value = 0.0F;
    if (threadIdx.x == 0) {
      const float x = load_as_float(a, gate_index) + load_as_float(dt_bias, hv);
      const float softplus_x = x <= 20.0F ? log1pf(__expf(x)) : x;
      const float g_value = -__expf(A_log[hv]) * softplus_x;
      gate_value = __expf(g_value);
      const float b_value = load_as_float(b, gate_index);
      beta_value = 1.0F / (1.0F + __expf(-b_value));
    }
    gate_value = __shfl_sync(0xffffffffU, gate_value, 0);
    beta_value = __shfl_sync(0xffffffffU, beta_value, 0);

    float kv_partial[COLS];
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      kv_partial[c] = 0.0F;
    }
#pragma unroll
    for (int r = 0; r < rows_per_lane; ++r) {
#pragma unroll
      for (int c = 0; c < COLS; ++c) {
        kv_partial[c] += state_shard[c][r] * k_reg[r];
      }
    }

    float delta[COLS];
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      const int col = col_base + c;
      const float kv_col = subgroup_sum(kv_partial[c], WIDTH);
      float delta_value = 0.0F;
      if (lane == 0 && col < V) {
        const int64_t v_index = mixed_base + 2 * q_heads * K + hv * V + col;
        delta_value =
            (load_as_float(mixed_qkv, v_index) - gate_value * kv_col) *
            beta_value;
      }
      delta[c] = subgroup_broadcast(delta_value, WIDTH);
    }

    float attn_partial[COLS];
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      attn_partial[c] = 0.0F;
    }
#pragma unroll
    for (int r = 0; r < rows_per_lane; ++r) {
#pragma unroll
      for (int c = 0; c < COLS; ++c) {
        const float new_state =
            fmaf(k_reg[r], delta[c], gate_value * state_shard[c][r]);
        state_shard[c][r] = new_state;
        attn_partial[c] += new_state * q_reg[r];
      }
    }
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      attn_partial[c] = subgroup_sum(attn_partial[c], WIDTH);
    }

    if (lane == 0) {
      const int64_t out_base = (static_cast<int64_t>(t) * v_heads + hv) * V;
#pragma unroll
      for (int c = 0; c < COLS; ++c) {
        const int col = col_base + c;
        if (col < V) {
          store_from_float(output, out_base + col, attn_partial[c] * scale);
        }
      }
    }

    const int32_t dst_state_idx =
        state_indices[n * max_state_tokens + local_t];
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      const int col = col_base + c;
#pragma unroll
      for (int r = 0; r < rows_per_lane; ++r) {
        const int row = r * WIDTH + lane;
        if (col < V && dst_state_idx >= 0 && dst_state_idx < slots) {
          const int64_t state_index =
              static_cast<int64_t>(dst_state_idx) * state_slot_stride +
              ((static_cast<int64_t>(hv) * V + col) * K) + row;
          store_from_float(state, state_index, state_shard[c][r]);
        }
      }
    }
  }
}

template <typename scalar_t, typename gate_t, typename beta_t, typename state_t, int K, int V>
void launch_gdn_forward_typed(const scalar_t* q,
                              const scalar_t* k,
                              const scalar_t* v,
                              const gate_t* gate,
                              const beta_t* beta,
                              const state_t* initial_state,
                              scalar_t* output,
                              float* final_state,
                              int batch,
                              int tokens,
                              int q_heads,
                              int v_heads,
                              float scale,
                              bool gate_is_exp,
                              int column_groups_per_block,
                              cudaStream_t stream) {
  constexpr int cols = V == 128 ? 4 : 1;
  constexpr int width = K == 128 ? 16 : 32;
  constexpr int groups_per_warp = 32 / width;
  const dim3 block(32, column_groups_per_block);
  const int groups = (V + cols - 1) / cols;
  const int z = (groups + column_groups_per_block * groups_per_warp - 1) /
                (column_groups_per_block * groups_per_warp);
  const dim3 grid(v_heads, batch, z);
  if (gate_is_exp) {
    gdn_forward_kernel<scalar_t, gate_t, beta_t, state_t, K, V, cols, width, true>
        <<<grid, block, 0, stream>>>(q,
                                     k,
                                     v,
                                     gate,
                                     beta,
                                     initial_state,
                                     output,
                                     final_state,
                                     batch,
                                     tokens,
                                     q_heads,
                                     v_heads,
                                     scale,
                                     gate_is_exp);
  } else {
    gdn_forward_kernel<scalar_t, gate_t, beta_t, state_t, K, V, cols, width, false>
        <<<grid, block, 0, stream>>>(q,
                                     k,
                                     v,
                                     gate,
                                     beta,
                                     initial_state,
                                     output,
                                     final_state,
                                     batch,
                                     tokens,
                                     q_heads,
                                     v_heads,
                                     scale,
                                     gate_is_exp);
  }
}

template <typename scalar_t, typename gate_t, typename beta_t, typename state_t>
void launch_gdn_forward_kv(const scalar_t* q,
                           const scalar_t* k,
                           const scalar_t* v,
                           const gate_t* gate,
                           const beta_t* beta,
                           const state_t* initial_state,
                           scalar_t* output,
                           float* final_state,
                           int batch,
                           int tokens,
                           int q_heads,
                           int v_heads,
                           int k_dim,
                           int v_dim,
                           float scale,
                           bool gate_is_exp,
                           int column_groups_per_block,
                           cudaStream_t stream) {
  TORCH_CHECK(k_dim == 128 && v_dim == 128,
              "SM70 FlashQLA backend currently supports K=V=128");
  launch_gdn_forward_typed<scalar_t, gate_t, beta_t, state_t, 128, 128>(
      q, k, v, gate, beta, initial_state, output, final_state, batch, tokens,
      q_heads, v_heads, scale, gate_is_exp, column_groups_per_block, stream);
}

template <typename scalar_t, typename gate_t, typename beta_t, typename state_t, int K, int V>
void launch_gdn_forward_vlk_varlen_typed(const scalar_t* q,
                                         const scalar_t* k,
                                         const scalar_t* v,
                                         const gate_t* gate,
                                         const beta_t* beta,
                                         const state_t* initial_state,
                                         const int32_t* cu_seqlens,
                                         scalar_t* output,
                                         float* final_state,
                                         int num_sequences,
                                         int tokens,
                                         int q_heads,
                                         int v_heads,
                                         float scale,
                                         bool gate_is_exp,
                                         int column_groups_per_block,
                                         cudaStream_t stream) {
  constexpr int cols = V == 128 ? 4 : 1;
  constexpr int width = K == 128 ? 16 : 32;
  constexpr int groups_per_warp = 32 / width;
  const dim3 block(32, column_groups_per_block);
  const int groups = (V + cols - 1) / cols;
  const int z = (groups + column_groups_per_block * groups_per_warp - 1) /
                (column_groups_per_block * groups_per_warp);
  const dim3 grid(v_heads, num_sequences, z);
  if (gate_is_exp) {
    gdn_forward_vlk_varlen_kernel<scalar_t, gate_t, beta_t, state_t, K, V, cols,
                                  width, true>
        <<<grid, block, 0, stream>>>(q,
                                     k,
                                     v,
                                     gate,
                                     beta,
                                     initial_state,
                                     cu_seqlens,
                                     output,
                                     final_state,
                                     q_heads,
                                     v_heads,
                                     scale,
                                     gate_is_exp);
  } else {
    gdn_forward_vlk_varlen_kernel<scalar_t, gate_t, beta_t, state_t, K, V, cols,
                                  width, false>
        <<<grid, block, 0, stream>>>(q,
                                     k,
                                     v,
                                     gate,
                                     beta,
                                     initial_state,
                                     cu_seqlens,
                                     output,
                                     final_state,
                                     q_heads,
                                     v_heads,
                                     scale,
                                     gate_is_exp);
  }
}

template <typename scalar_t, typename gate_t, typename beta_t, typename state_t>
void launch_gdn_forward_vlk_varlen_kv(const scalar_t* q,
                                      const scalar_t* k,
                                      const scalar_t* v,
                                      const gate_t* gate,
                                      const beta_t* beta,
                                      const state_t* initial_state,
                                      const int32_t* cu_seqlens,
                                      scalar_t* output,
                                      float* final_state,
                                      int num_sequences,
                                      int tokens,
                                      int q_heads,
                                      int v_heads,
                                      int k_dim,
                                      int v_dim,
                                      float scale,
                                      bool gate_is_exp,
                                      int column_groups_per_block,
                                      cudaStream_t stream) {
  TORCH_CHECK(k_dim == 128 && v_dim == 128,
              "SM70 FlashQLA backend currently supports K=V=128");
  launch_gdn_forward_vlk_varlen_typed<scalar_t, gate_t, beta_t, state_t, 128, 128>(
      q, k, v, gate, beta, initial_state, cu_seqlens, output, final_state,
      num_sequences, tokens, q_heads, v_heads, scale, gate_is_exp,
      column_groups_per_block, stream);
}

template <typename scalar_t, typename bias_t, typename state_t, int K, int V>
void launch_gdn_decode_mixed_qkv_global_state_typed(
    const scalar_t* mixed_qkv,
    const scalar_t* a,
    const scalar_t* b,
    const float* A_log,
    const bias_t* dt_bias,
    state_t* state,
    const int32_t* state_indices,
    scalar_t* output,
    int tokens,
    int slots,
    int64_t state_slot_stride,
    int q_heads,
    int v_heads,
    int qkv_stride,
    float scale,
    bool use_qk_l2norm,
    int column_groups_per_block,
    cudaStream_t stream) {
  constexpr int cols = V == 128 ? 4 : 1;
  constexpr int width = K == 128 ? 16 : 32;
  constexpr int groups_per_warp = 32 / width;
  const dim3 block(32, column_groups_per_block);
  const int groups = (V + cols - 1) / cols;
  const int z = (groups + column_groups_per_block * groups_per_warp - 1) /
                (column_groups_per_block * groups_per_warp);
  const dim3 grid(v_heads, tokens, z);
  gdn_decode_mixed_qkv_global_state_kernel<scalar_t, bias_t, state_t, K, V, cols, width>
      <<<grid, block, 0, stream>>>(mixed_qkv,
                                   a,
                                   b,
                                   A_log,
                                   dt_bias,
                                   state,
                                   state_indices,
                                   output,
                                   tokens,
                                   slots,
                                   state_slot_stride,
                                   q_heads,
                                   v_heads,
                                   qkv_stride,
                                   scale,
                                   use_qk_l2norm);
}

template <typename scalar_t, typename bias_t, typename state_t>
void launch_gdn_decode_mixed_qkv_global_state_kv(
    const scalar_t* mixed_qkv,
    const scalar_t* a,
    const scalar_t* b,
    const float* A_log,
    const bias_t* dt_bias,
    state_t* state,
    const int32_t* state_indices,
    scalar_t* output,
    int tokens,
    int slots,
    int64_t state_slot_stride,
    int q_heads,
    int v_heads,
    int k_dim,
    int v_dim,
    int qkv_stride,
    float scale,
    bool use_qk_l2norm,
    int column_groups_per_block,
    cudaStream_t stream) {
  TORCH_CHECK(k_dim == 128 && v_dim == 128,
              "SM70 FlashQLA decode currently supports K=V=128");
  launch_gdn_decode_mixed_qkv_global_state_typed<scalar_t, bias_t, state_t, 128, 128>(
      mixed_qkv, a, b, A_log, dt_bias, state, state_indices, output, tokens,
      slots, state_slot_stride, q_heads, v_heads, qkv_stride, scale, use_qk_l2norm,
      column_groups_per_block, stream);
}

template <typename scalar_t, typename bias_t, typename state_t, int K, int V>
void launch_gdn_decode_mixed_qkv_ddtree_state_typed(
    const scalar_t* mixed_qkv,
    const scalar_t* a,
    const scalar_t* b,
    const float* A_log,
    const bias_t* dt_bias,
    state_t* state,
    const int32_t* state_indices,
    const int32_t* parent_ids,
    const int32_t* num_accepted_tokens,
    const int32_t* cu_seqlens,
    scalar_t* output,
    int num_sequences,
    int max_state_tokens,
    int tokens,
    int slots,
    int64_t state_slot_stride,
    int q_heads,
    int v_heads,
    int qkv_stride,
    float scale,
    bool use_qk_l2norm,
    int column_groups_per_block,
    cudaStream_t stream) {
  constexpr int cols = V == 128 ? 4 : 1;
  constexpr int width = K == 128 ? 16 : 32;
  constexpr int groups_per_warp = 32 / width;
  const dim3 block(32, column_groups_per_block);
  const int groups = (V + cols - 1) / cols;
  const int z = (groups + column_groups_per_block * groups_per_warp - 1) /
                (column_groups_per_block * groups_per_warp);
  const dim3 grid(v_heads, num_sequences, z);
  gdn_decode_mixed_qkv_ddtree_state_kernel<scalar_t,
                                           bias_t,
                                           state_t,
                                           K,
                                           V,
                                           cols,
                                           width>
      <<<grid, block, 0, stream>>>(mixed_qkv,
                                   a,
                                   b,
                                   A_log,
                                   dt_bias,
                                   state,
                                   state_indices,
                                   parent_ids,
                                   num_accepted_tokens,
                                   cu_seqlens,
                                   output,
                                   num_sequences,
                                   max_state_tokens,
                                   tokens,
                                   slots,
                                   state_slot_stride,
                                   q_heads,
                                   v_heads,
                                   qkv_stride,
                                   scale,
                                   use_qk_l2norm);
}

template <typename scalar_t, typename bias_t, typename state_t>
void launch_gdn_decode_mixed_qkv_ddtree_state_kv(
    const scalar_t* mixed_qkv,
    const scalar_t* a,
    const scalar_t* b,
    const float* A_log,
    const bias_t* dt_bias,
    state_t* state,
    const int32_t* state_indices,
    const int32_t* parent_ids,
    const int32_t* num_accepted_tokens,
    const int32_t* cu_seqlens,
    scalar_t* output,
    int num_sequences,
    int max_state_tokens,
    int tokens,
    int slots,
    int64_t state_slot_stride,
    int q_heads,
    int v_heads,
    int k_dim,
    int v_dim,
    int qkv_stride,
    float scale,
    bool use_qk_l2norm,
    int column_groups_per_block,
    cudaStream_t stream) {
  TORCH_CHECK(k_dim == 128 && v_dim == 128,
              "SM70 FlashQLA DDTree decode currently supports K=V=128");
  launch_gdn_decode_mixed_qkv_ddtree_state_typed<scalar_t,
                                                 bias_t,
                                                 state_t,
                                                 128,
                                                 128>(
      mixed_qkv, a, b, A_log, dt_bias, state, state_indices, parent_ids,
      num_accepted_tokens, cu_seqlens, output, num_sequences, max_state_tokens,
      tokens, slots, state_slot_stride, q_heads, v_heads, qkv_stride, scale,
      use_qk_l2norm, column_groups_per_block, stream);
}

void validate_tensor(const torch::Tensor& tensor,
                     const char* name,
                     int64_t dims) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.dim() == dims, name, " has wrong rank");
}

void validate_mixed_qkv_tensor(const torch::Tensor& tensor) {
  TORCH_CHECK(tensor.is_cuda(), "mixed_qkv must be a CUDA tensor");
  TORCH_CHECK(tensor.dim() == 2, "mixed_qkv has wrong rank");
  TORCH_CHECK(tensor.stride(1) == 1,
              "mixed_qkv must have dense columns");
  TORCH_CHECK(tensor.stride(0) >= tensor.size(1),
              "mixed_qkv row stride must be >= logical width");
}

void validate_cuda_rank(const torch::Tensor& tensor,
                        const char* name,
                        int64_t dims) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.dim() == dims, name, " has wrong rank");
}

void validate_activation_dtype(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16 ||
                  tensor.scalar_type() == torch::kBFloat16 ||
                  tensor.scalar_type() == torch::kFloat32,
              name,
              " must be fp16, bf16, or fp32");
}

void validate_same_device(const torch::Tensor& tensor,
                          const torch::Tensor& reference,
                          const char* name) {
  TORCH_CHECK(tensor.device() == reference.device(),
              name,
              " must be on the same CUDA device as q");
}

int get_column_groups_per_block(int tokens,
                                int q_heads,
                                int v_heads) {
  const char* raw = std::getenv("FLASH_QLA_SM70_COLUMN_GROUPS_PER_BLOCK");
  if (raw == nullptr || raw[0] == '\0') {
    // Real Qwen3.5/Qwen3.6 SM70 shapes are dominated by Hv=8/12/16/24/32
    // after TP. Keep this as a shape heuristic; the env above remains the
    // escape hatch for benchmarks and future model-specific tuning. V100 has
    // no native bf16, so the default is derived from the fp16 production path.
    if (v_heads == 24 || v_heads == 48) {
      return 2;
    }
    if (v_heads == 32) {
      if (q_heads == 16) {
        return tokens <= 1024 ? 4 : 2;
      }
      if (q_heads == 8) {
        return tokens <= 1024 ? 2 : 1;
      }
      return 1;
    }
    if (v_heads >= 64) {
      return 1;
    }
    if (v_heads == 16) {
      if (q_heads == 8) {
        return 2;
      }
      return tokens <= 1024 ? 2 : 1;
    }
    if (v_heads == 12) {
      return tokens >= 1024 ? 2 : 1;
    }
    if (v_heads == 8) {
      return tokens <= 1024 ? 1 : 4;
    }
    return 2;
  }
  const int value = std::atoi(raw);
  TORCH_CHECK(value == 1 || value == 2 || value == 4 || value == 8,
              "FLASH_QLA_SM70_COLUMN_GROUPS_PER_BLOCK must be one of 1, 2, 4, 8");
  return value;
}

int get_ddtree_column_groups_per_block(int tokens,
                                       int q_heads,
                                       int v_heads) {
  const char* raw = std::getenv("FLASH_QLA_SM70_DDTREE_COLUMN_GROUPS_PER_BLOCK");
  if (raw != nullptr && raw[0] != '\0') {
    const int value = std::atoi(raw);
    TORCH_CHECK(value == 1 || value == 2 || value == 4 || value == 8,
                "FLASH_QLA_SM70_DDTREE_COLUMN_GROUPS_PER_BLOCK must be one of "
                "1, 2, 4, 8");
    return value;
  }
  return get_column_groups_per_block(tokens, q_heads, v_heads);
}

}  // namespace

int resolve_column_groups_per_block(int tokens, int q_heads, int v_heads) {
  return get_column_groups_per_block(tokens, q_heads, v_heads);
}

std::vector<torch::Tensor> gdn_forward(torch::Tensor q,
                                       torch::Tensor k,
                                       torch::Tensor v,
                                       torch::Tensor gate,
                                       torch::Tensor beta,
                                       c10::optional<torch::Tensor> initial_state,
                                       double scale,
                                       bool output_final_state,
                                       bool gate_is_exp) {
  validate_tensor(q, "q", 4);
  validate_tensor(k, "k", 4);
  validate_tensor(v, "v", 4);
  validate_tensor(gate, "gate", 3);
  validate_tensor(beta, "beta", 3);
  validate_activation_dtype(q, "q");
  validate_same_device(k, q, "k");
  validate_same_device(v, q, "v");
  validate_same_device(gate, q, "gate");
  validate_same_device(beta, q, "beta");

  TORCH_CHECK(k.scalar_type() == q.scalar_type(), "k must match q dtype");
  TORCH_CHECK(v.scalar_type() == q.scalar_type(), "v must match q dtype");
  validate_activation_dtype(gate, "gate");
  validate_activation_dtype(beta, "beta");
  TORCH_CHECK(q.sizes() == k.sizes(), "q and k must have the same shape");

  const int batch = static_cast<int>(q.size(0));
  const int tokens = static_cast<int>(q.size(1));
  const int q_heads = static_cast<int>(q.size(2));
  const int k_dim = static_cast<int>(q.size(3));
  const int v_heads = static_cast<int>(v.size(2));
  const int v_dim = static_cast<int>(v.size(3));
  TORCH_CHECK(v.size(0) == batch && v.size(1) == tokens,
              "v must have shape [B, T, Hv, V] matching q/k");
  TORCH_CHECK(gate.size(0) == batch && gate.size(1) == tokens &&
                  gate.size(2) == v_heads,
              "gate must have shape [B, T, Hv]");
  TORCH_CHECK(beta.sizes() == gate.sizes(),
              "beta must have the same shape as gate");
  TORCH_CHECK(v_heads % q_heads == 0, "Hv must be divisible by Hq");
  TORCH_CHECK(k_dim == 128 && v_dim == 128,
              "SM70 FlashQLA backend currently supports K=V=128");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));

  const void* initial_ptr = nullptr;
  at::ScalarType state_dtype = torch::kFloat32;
  if (initial_state.has_value() && initial_state.value().defined()) {
    const auto& h0 = initial_state.value();
    validate_tensor(h0, "initial_state", 4);
    validate_same_device(h0, q, "initial_state");
    TORCH_CHECK(h0.scalar_type() == torch::kFloat16 ||
                    h0.scalar_type() == torch::kBFloat16 ||
                    h0.scalar_type() == torch::kFloat32,
                "initial_state must be fp16, bf16, or fp32");
    TORCH_CHECK(h0.size(0) == batch && h0.size(1) == v_heads &&
                    h0.size(2) == k_dim && h0.size(3) == v_dim,
                "initial_state must have shape [B, Hv, K, V]");
    initial_ptr = h0.data_ptr();
    state_dtype = h0.scalar_type();
  }

  auto output = torch::empty_like(v);
  auto final_state = output_final_state
                         ? torch::empty({batch, v_heads, k_dim, v_dim},
                                        q.options().dtype(torch::kFloat32))
                         : torch::Tensor();
  float* final_state_ptr =
      output_final_state ? final_state.data_ptr<float>() : nullptr;
  const int column_groups_per_block =
      get_column_groups_per_block(tokens, q_heads, v_heads);
  const auto stream = at::cuda::getCurrentCUDAStream(q.device().index()).stream();

  auto dispatch_state = [&](auto q_ptr, auto k_ptr, auto v_ptr, auto gate_ptr,
                            auto beta_ptr, auto out_ptr) {
    using scalar_t = std::remove_pointer_t<decltype(q_ptr)>;
    using gate_t = std::remove_pointer_t<decltype(gate_ptr)>;
    using beta_t = std::remove_pointer_t<decltype(beta_ptr)>;
    if (initial_ptr == nullptr || state_dtype == torch::kFloat32) {
      launch_gdn_forward_kv<scalar_t, gate_t, beta_t, float>(
          q_ptr, k_ptr, v_ptr, gate_ptr, beta_ptr,
          reinterpret_cast<const float*>(initial_ptr), out_ptr,
          final_state_ptr, batch, tokens, q_heads, v_heads, k_dim,
          v_dim, static_cast<float>(scale), gate_is_exp,
          column_groups_per_block, stream);
    } else if (state_dtype == torch::kFloat16) {
      launch_gdn_forward_kv<scalar_t, gate_t, beta_t, at::Half>(
          q_ptr, k_ptr, v_ptr, gate_ptr, beta_ptr,
          reinterpret_cast<const at::Half*>(initial_ptr), out_ptr,
          final_state_ptr, batch, tokens, q_heads, v_heads, k_dim,
          v_dim, static_cast<float>(scale), gate_is_exp,
          column_groups_per_block, stream);
    } else {
      launch_gdn_forward_kv<scalar_t, gate_t, beta_t, at::BFloat16>(
          q_ptr, k_ptr, v_ptr, gate_ptr, beta_ptr,
          reinterpret_cast<const at::BFloat16*>(initial_ptr), out_ptr,
          final_state_ptr, batch, tokens, q_heads, v_heads, k_dim,
          v_dim, static_cast<float>(scale), gate_is_exp,
          column_groups_per_block, stream);
    }
  };

  auto dispatch_beta = [&](auto q_ptr, auto k_ptr, auto v_ptr, auto gate_ptr,
                           auto out_ptr) {
    if (beta.scalar_type() == torch::kFloat16) {
      dispatch_state(q_ptr, k_ptr, v_ptr, gate_ptr, beta.data_ptr<at::Half>(),
                     out_ptr);
    } else if (beta.scalar_type() == torch::kBFloat16) {
      dispatch_state(q_ptr, k_ptr, v_ptr, gate_ptr,
                     beta.data_ptr<at::BFloat16>(), out_ptr);
    } else {
      dispatch_state(q_ptr, k_ptr, v_ptr, gate_ptr, beta.data_ptr<float>(),
                     out_ptr);
    }
  };

  auto dispatch_gate = [&](auto q_ptr, auto k_ptr, auto v_ptr, auto out_ptr) {
    if (gate.scalar_type() == torch::kFloat16) {
      dispatch_beta(q_ptr, k_ptr, v_ptr, gate.data_ptr<at::Half>(), out_ptr);
    } else if (gate.scalar_type() == torch::kBFloat16) {
      dispatch_beta(q_ptr, k_ptr, v_ptr, gate.data_ptr<at::BFloat16>(), out_ptr);
    } else {
      dispatch_beta(q_ptr, k_ptr, v_ptr, gate.data_ptr<float>(), out_ptr);
    }
  };

  if (q.scalar_type() == torch::kFloat16) {
    dispatch_gate(q.data_ptr<at::Half>(), k.data_ptr<at::Half>(),
                  v.data_ptr<at::Half>(), output.data_ptr<at::Half>());
  } else if (q.scalar_type() == torch::kBFloat16) {
    dispatch_gate(q.data_ptr<at::BFloat16>(), k.data_ptr<at::BFloat16>(),
                  v.data_ptr<at::BFloat16>(),
                  output.data_ptr<at::BFloat16>());
  } else {
    dispatch_gate(q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
                  output.data_ptr<float>());
  }

  check_cuda(cudaGetLastError(), "gdn_forward launch");
  return {output, final_state};
}

std::vector<torch::Tensor> gdn_forward_vlk_varlen(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor gate,
    torch::Tensor beta,
    c10::optional<torch::Tensor> initial_state,
    torch::Tensor cu_seqlens,
    double scale,
    bool output_final_state,
    bool validate_cu_seqlens,
    bool gate_is_exp,
    c10::optional<torch::Tensor> output_arg) {
  validate_tensor(q, "q", 4);
  validate_tensor(k, "k", 4);
  validate_tensor(v, "v", 4);
  validate_tensor(gate, "gate", 3);
  validate_tensor(beta, "beta", 3);
  validate_tensor(cu_seqlens, "cu_seqlens", 1);
  validate_activation_dtype(q, "q");
  validate_same_device(k, q, "k");
  validate_same_device(v, q, "v");
  validate_same_device(gate, q, "gate");
  validate_same_device(beta, q, "beta");
  validate_same_device(cu_seqlens, q, "cu_seqlens");

  TORCH_CHECK(k.scalar_type() == q.scalar_type(), "k must match q dtype");
  TORCH_CHECK(v.scalar_type() == q.scalar_type(), "v must match q dtype");
  validate_activation_dtype(gate, "gate");
  validate_activation_dtype(beta, "beta");
  TORCH_CHECK(cu_seqlens.scalar_type() == torch::kInt32,
              "cu_seqlens must be int32");
  if (validate_cu_seqlens) {
    TORCH_CHECK(cu_seqlens[0].item<int32_t>() == 0,
                "cu_seqlens must start at 0");
  }
  TORCH_CHECK(q.sizes() == k.sizes(), "q and k must have the same shape");

  const int batch = static_cast<int>(q.size(0));
  const int tokens = static_cast<int>(q.size(1));
  const int q_heads = static_cast<int>(q.size(2));
  const int k_dim = static_cast<int>(q.size(3));
  const int v_heads = static_cast<int>(v.size(2));
  const int v_dim = static_cast<int>(v.size(3));
  const int num_sequences = static_cast<int>(cu_seqlens.size(0) - 1);
  if (validate_cu_seqlens) {
    TORCH_CHECK(cu_seqlens[num_sequences].item<int32_t>() == tokens,
                "cu_seqlens must end at the flattened token count");
  }
  TORCH_CHECK(batch == 1,
              "gdn_forward_vlk_varlen expects flattened q/k/v with batch=1");
  TORCH_CHECK(num_sequences > 0, "cu_seqlens must contain at least one sequence");
  TORCH_CHECK(v.size(0) == batch && v.size(1) == tokens,
              "v must have shape [1, T, Hv, V] matching q/k");
  TORCH_CHECK(gate.size(0) == batch && gate.size(1) == tokens &&
                  gate.size(2) == v_heads,
              "gate must have shape [1, T, Hv]");
  TORCH_CHECK(beta.sizes() == gate.sizes(),
              "beta must have the same shape as gate");
  TORCH_CHECK(v_heads % q_heads == 0, "Hv must be divisible by Hq");
  TORCH_CHECK(k_dim == 128 && v_dim == 128,
              "SM70 FlashQLA backend currently supports K=V=128");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));

  const void* initial_ptr = nullptr;
  at::ScalarType state_dtype = torch::kFloat32;
  if (initial_state.has_value() && initial_state.value().defined()) {
    const auto& h0 = initial_state.value();
    validate_tensor(h0, "initial_state", 4);
    validate_same_device(h0, q, "initial_state");
    TORCH_CHECK(h0.scalar_type() == torch::kFloat16 ||
                    h0.scalar_type() == torch::kBFloat16 ||
                    h0.scalar_type() == torch::kFloat32,
                "initial_state must be fp16, bf16, or fp32");
    TORCH_CHECK(h0.size(0) == num_sequences && h0.size(1) == v_heads &&
                    h0.size(2) == v_dim && h0.size(3) == k_dim,
                "initial_state must have shape [N, Hv, V, K]");
    initial_ptr = h0.data_ptr();
    state_dtype = h0.scalar_type();
  }

  torch::Tensor output;
  if (output_arg.has_value() && output_arg.value().defined()) {
    output = output_arg.value();
    validate_tensor(output, "output", 4);
    validate_same_device(output, q, "output");
    TORCH_CHECK(output.scalar_type() == v.scalar_type(),
                "output must match v dtype");
    TORCH_CHECK(output.size(0) == batch && output.size(1) == tokens &&
                    output.size(2) == v_heads && output.size(3) == v_dim,
                "output must have shape [1, T, Hv, V]");
  } else {
    output = torch::empty_like(v);
  }
  auto final_state = output_final_state
                         ? torch::empty({num_sequences, v_heads, v_dim, k_dim},
                                        q.options().dtype(torch::kFloat32))
                         : torch::Tensor();
  float* final_state_ptr =
      output_final_state ? final_state.data_ptr<float>() : nullptr;
  const int column_groups_per_block =
      get_column_groups_per_block(tokens, q_heads, v_heads);
  const auto stream = at::cuda::getCurrentCUDAStream(q.device().index()).stream();

  auto dispatch_state = [&](auto q_ptr, auto k_ptr, auto v_ptr, auto gate_ptr,
                            auto beta_ptr, auto out_ptr) {
    using scalar_t = std::remove_pointer_t<decltype(q_ptr)>;
    using gate_t = std::remove_pointer_t<decltype(gate_ptr)>;
    using beta_t = std::remove_pointer_t<decltype(beta_ptr)>;
    if (initial_ptr == nullptr || state_dtype == torch::kFloat32) {
      launch_gdn_forward_vlk_varlen_kv<scalar_t, gate_t, beta_t, float>(
          q_ptr, k_ptr, v_ptr, gate_ptr, beta_ptr,
          reinterpret_cast<const float*>(initial_ptr),
          cu_seqlens.data_ptr<int32_t>(), out_ptr, final_state_ptr,
          num_sequences, tokens, q_heads, v_heads, k_dim, v_dim,
          static_cast<float>(scale), gate_is_exp, column_groups_per_block,
          stream);
    } else if (state_dtype == torch::kFloat16) {
      launch_gdn_forward_vlk_varlen_kv<scalar_t, gate_t, beta_t, at::Half>(
          q_ptr, k_ptr, v_ptr, gate_ptr, beta_ptr,
          reinterpret_cast<const at::Half*>(initial_ptr),
          cu_seqlens.data_ptr<int32_t>(), out_ptr, final_state_ptr,
          num_sequences, tokens, q_heads, v_heads, k_dim, v_dim,
          static_cast<float>(scale), gate_is_exp, column_groups_per_block,
          stream);
    } else {
      launch_gdn_forward_vlk_varlen_kv<scalar_t, gate_t, beta_t, at::BFloat16>(
          q_ptr, k_ptr, v_ptr, gate_ptr, beta_ptr,
          reinterpret_cast<const at::BFloat16*>(initial_ptr),
          cu_seqlens.data_ptr<int32_t>(), out_ptr, final_state_ptr,
          num_sequences, tokens, q_heads, v_heads, k_dim, v_dim,
          static_cast<float>(scale), gate_is_exp, column_groups_per_block,
          stream);
    }
  };

  auto dispatch_beta = [&](auto q_ptr, auto k_ptr, auto v_ptr, auto gate_ptr,
                           auto out_ptr) {
    if (beta.scalar_type() == torch::kFloat16) {
      dispatch_state(q_ptr, k_ptr, v_ptr, gate_ptr, beta.data_ptr<at::Half>(),
                     out_ptr);
    } else if (beta.scalar_type() == torch::kBFloat16) {
      dispatch_state(q_ptr, k_ptr, v_ptr, gate_ptr,
                     beta.data_ptr<at::BFloat16>(), out_ptr);
    } else {
      dispatch_state(q_ptr, k_ptr, v_ptr, gate_ptr, beta.data_ptr<float>(),
                     out_ptr);
    }
  };

  auto dispatch_gate = [&](auto q_ptr, auto k_ptr, auto v_ptr, auto out_ptr) {
    if (gate.scalar_type() == torch::kFloat16) {
      dispatch_beta(q_ptr, k_ptr, v_ptr, gate.data_ptr<at::Half>(), out_ptr);
    } else if (gate.scalar_type() == torch::kBFloat16) {
      dispatch_beta(q_ptr, k_ptr, v_ptr, gate.data_ptr<at::BFloat16>(), out_ptr);
    } else {
      dispatch_beta(q_ptr, k_ptr, v_ptr, gate.data_ptr<float>(), out_ptr);
    }
  };

  if (q.scalar_type() == torch::kFloat16) {
    dispatch_gate(q.data_ptr<at::Half>(), k.data_ptr<at::Half>(),
                  v.data_ptr<at::Half>(), output.data_ptr<at::Half>());
  } else if (q.scalar_type() == torch::kBFloat16) {
    dispatch_gate(q.data_ptr<at::BFloat16>(), k.data_ptr<at::BFloat16>(),
                  v.data_ptr<at::BFloat16>(),
                  output.data_ptr<at::BFloat16>());
  } else {
    dispatch_gate(q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
                  output.data_ptr<float>());
  }

  check_cuda(cudaGetLastError(), "gdn_forward_vlk_varlen launch");
  return {output, final_state};
}

void gdn_decode_mixed_qkv_global_state(torch::Tensor mixed_qkv,
                                       torch::Tensor a,
                                       torch::Tensor b,
                                       torch::Tensor A_log,
                                       torch::Tensor dt_bias,
                                       torch::Tensor state,
                                       torch::Tensor state_indices,
                                       torch::Tensor output,
                                       double scale,
                                       bool use_qk_l2norm) {
  validate_mixed_qkv_tensor(mixed_qkv);
  validate_tensor(a, "a", 2);
  validate_tensor(b, "b", 2);
  validate_tensor(A_log, "A_log", 1);
  validate_tensor(dt_bias, "dt_bias", 1);
  validate_cuda_rank(state, "state", 4);
  validate_tensor(state_indices, "state_indices", 1);
  validate_tensor(output, "output", 3);
  validate_activation_dtype(mixed_qkv, "mixed_qkv");
  validate_activation_dtype(a, "a");
  validate_activation_dtype(b, "b");
  validate_activation_dtype(state, "state");
  validate_activation_dtype(output, "output");
  validate_same_device(a, mixed_qkv, "a");
  validate_same_device(b, mixed_qkv, "b");
  validate_same_device(A_log, mixed_qkv, "A_log");
  validate_same_device(dt_bias, mixed_qkv, "dt_bias");
  validate_same_device(state, mixed_qkv, "state");
  validate_same_device(state_indices, mixed_qkv, "state_indices");
  validate_same_device(output, mixed_qkv, "output");

  TORCH_CHECK(mixed_qkv.scalar_type() == a.scalar_type(),
              "a must match mixed_qkv dtype");
  TORCH_CHECK(mixed_qkv.scalar_type() == b.scalar_type(),
              "b must match mixed_qkv dtype");
  TORCH_CHECK(mixed_qkv.scalar_type() == output.scalar_type(),
              "output must match mixed_qkv dtype");
  TORCH_CHECK(A_log.scalar_type() == torch::kFloat32,
              "A_log must be float32");
  validate_activation_dtype(dt_bias, "dt_bias");
  TORCH_CHECK(state_indices.scalar_type() == torch::kInt32,
              "state_indices must be int32");
  const int tokens = static_cast<int>(mixed_qkv.size(0));
  const int slots = static_cast<int>(state.size(0));
  const int v_heads = static_cast<int>(state.size(1));
  const int v_dim = static_cast<int>(state.size(2));
  const int k_dim = static_cast<int>(state.size(3));
  const int64_t state_slot_stride = state.stride(0);
  TORCH_CHECK(tokens > 0, "decode tokens must be positive");
  TORCH_CHECK(v_dim == 128 && k_dim == 128,
              "SM70 FlashQLA decode currently supports K=V=128");
  TORCH_CHECK(state.stride(1) == v_dim * k_dim && state.stride(2) == k_dim &&
                  state.stride(3) == 1,
              "state inner layout must be [slots,Hv,V,K] with contiguous "
              "[Hv,V,K] pages");
  TORCH_CHECK(a.size(0) == tokens && b.size(0) == tokens,
              "a/b must match mixed_qkv token count");
  TORCH_CHECK(a.size(1) == v_heads && b.size(1) == v_heads,
              "a/b must have HV columns matching state");
  TORCH_CHECK(A_log.size(0) == v_heads && dt_bias.size(0) == v_heads,
              "A_log/dt_bias must have HV elements");
  TORCH_CHECK(state_indices.size(0) == tokens,
              "state_indices must have one entry per decode token");
  TORCH_CHECK(output.size(0) == tokens && output.size(1) == v_heads &&
                  output.size(2) == v_dim,
              "output must have shape [tokens, Hv, V]");

  const int qkv_dim = static_cast<int>(mixed_qkv.size(1));
  const int qk_dim = qkv_dim - v_heads * v_dim;
  TORCH_CHECK(qk_dim > 0 && qk_dim % 2 == 0,
              "invalid packed mixed_qkv last dimension");
  const int q_dim = qk_dim / 2;
  TORCH_CHECK(q_dim % k_dim == 0,
              "packed Q dimension must be divisible by K");
  const int q_heads = q_dim / k_dim;
  TORCH_CHECK(q_heads > 0 && v_heads % q_heads == 0,
              "invalid H/HV inferred from mixed_qkv and state");
  const int qkv_stride = static_cast<int>(mixed_qkv.stride(0));
  const int column_groups_per_block =
      get_column_groups_per_block(tokens, q_heads, v_heads);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mixed_qkv));
  const auto stream =
      at::cuda::getCurrentCUDAStream(mixed_qkv.device().index()).stream();

  auto dispatch_state = [&](auto mixed_ptr, auto a_ptr, auto b_ptr,
                            auto dt_bias_ptr, auto out_ptr) {
    using scalar_t = std::remove_pointer_t<decltype(mixed_ptr)>;
    using bias_t = std::remove_pointer_t<decltype(dt_bias_ptr)>;
    if (state.scalar_type() == torch::kFloat32) {
      launch_gdn_decode_mixed_qkv_global_state_kv<scalar_t, bias_t, float>(
          mixed_ptr, a_ptr, b_ptr, A_log.data_ptr<float>(),
          dt_bias_ptr, state.data_ptr<float>(),
          state_indices.data_ptr<int32_t>(), out_ptr, tokens, slots,
          state_slot_stride, q_heads, v_heads, k_dim, v_dim, qkv_stride,
          static_cast<float>(scale), use_qk_l2norm, column_groups_per_block,
          stream);
    } else if (state.scalar_type() == torch::kFloat16) {
      launch_gdn_decode_mixed_qkv_global_state_kv<scalar_t, bias_t, at::Half>(
          mixed_ptr, a_ptr, b_ptr, A_log.data_ptr<float>(),
          dt_bias_ptr, state.data_ptr<at::Half>(),
          state_indices.data_ptr<int32_t>(), out_ptr, tokens, slots,
          state_slot_stride, q_heads, v_heads, k_dim, v_dim, qkv_stride,
          static_cast<float>(scale), use_qk_l2norm, column_groups_per_block,
          stream);
    } else {
      launch_gdn_decode_mixed_qkv_global_state_kv<scalar_t, bias_t, at::BFloat16>(
          mixed_ptr, a_ptr, b_ptr, A_log.data_ptr<float>(),
          dt_bias_ptr, state.data_ptr<at::BFloat16>(),
          state_indices.data_ptr<int32_t>(), out_ptr, tokens, slots,
          state_slot_stride, q_heads, v_heads, k_dim, v_dim, qkv_stride,
          static_cast<float>(scale), use_qk_l2norm, column_groups_per_block,
          stream);
    }
  };

  auto dispatch_dt_bias = [&](auto mixed_ptr, auto a_ptr, auto b_ptr,
                              auto out_ptr) {
    if (dt_bias.scalar_type() == torch::kFloat16) {
      dispatch_state(mixed_ptr, a_ptr, b_ptr, dt_bias.data_ptr<at::Half>(),
                     out_ptr);
    } else if (dt_bias.scalar_type() == torch::kBFloat16) {
      dispatch_state(mixed_ptr, a_ptr, b_ptr, dt_bias.data_ptr<at::BFloat16>(),
                     out_ptr);
    } else {
      dispatch_state(mixed_ptr, a_ptr, b_ptr, dt_bias.data_ptr<float>(),
                     out_ptr);
    }
  };

  if (mixed_qkv.scalar_type() == torch::kFloat16) {
    dispatch_dt_bias(mixed_qkv.data_ptr<at::Half>(), a.data_ptr<at::Half>(),
                     b.data_ptr<at::Half>(), output.data_ptr<at::Half>());
  } else if (mixed_qkv.scalar_type() == torch::kBFloat16) {
    dispatch_dt_bias(mixed_qkv.data_ptr<at::BFloat16>(),
                     a.data_ptr<at::BFloat16>(),
                     b.data_ptr<at::BFloat16>(),
                     output.data_ptr<at::BFloat16>());
  } else {
    dispatch_dt_bias(mixed_qkv.data_ptr<float>(), a.data_ptr<float>(),
                     b.data_ptr<float>(), output.data_ptr<float>());
  }

  check_cuda(cudaGetLastError(), "gdn_decode_mixed_qkv_global_state launch");
}

void gdn_decode_mixed_qkv_ddtree_state(torch::Tensor mixed_qkv,
                                       torch::Tensor a,
                                       torch::Tensor b,
                                       torch::Tensor A_log,
                                       torch::Tensor dt_bias,
                                       torch::Tensor state,
                                       torch::Tensor state_indices,
                                       torch::Tensor parent_ids,
                                       torch::Tensor num_accepted_tokens,
                                       torch::Tensor cu_seqlens,
                                       torch::Tensor output,
                                       double scale,
                                       bool use_qk_l2norm) {
  validate_mixed_qkv_tensor(mixed_qkv);
  validate_tensor(a, "a", 2);
  validate_tensor(b, "b", 2);
  validate_tensor(A_log, "A_log", 1);
  validate_tensor(dt_bias, "dt_bias", 1);
  validate_cuda_rank(state, "state", 4);
  validate_tensor(state_indices, "state_indices", 2);
  validate_tensor(parent_ids, "parent_ids", 2);
  validate_tensor(num_accepted_tokens, "num_accepted_tokens", 1);
  validate_tensor(cu_seqlens, "cu_seqlens", 1);
  validate_tensor(output, "output", 3);
  validate_activation_dtype(mixed_qkv, "mixed_qkv");
  validate_activation_dtype(a, "a");
  validate_activation_dtype(b, "b");
  validate_activation_dtype(state, "state");
  validate_activation_dtype(output, "output");
  validate_same_device(a, mixed_qkv, "a");
  validate_same_device(b, mixed_qkv, "b");
  validate_same_device(A_log, mixed_qkv, "A_log");
  validate_same_device(dt_bias, mixed_qkv, "dt_bias");
  validate_same_device(state, mixed_qkv, "state");
  validate_same_device(state_indices, mixed_qkv, "state_indices");
  validate_same_device(parent_ids, mixed_qkv, "parent_ids");
  validate_same_device(num_accepted_tokens, mixed_qkv, "num_accepted_tokens");
  validate_same_device(cu_seqlens, mixed_qkv, "cu_seqlens");
  validate_same_device(output, mixed_qkv, "output");

  TORCH_CHECK(mixed_qkv.scalar_type() == a.scalar_type(),
              "a must match mixed_qkv dtype");
  TORCH_CHECK(mixed_qkv.scalar_type() == b.scalar_type(),
              "b must match mixed_qkv dtype");
  TORCH_CHECK(mixed_qkv.scalar_type() == output.scalar_type(),
              "output must match mixed_qkv dtype");
  TORCH_CHECK(A_log.scalar_type() == torch::kFloat32,
              "A_log must be float32");
  validate_activation_dtype(dt_bias, "dt_bias");
  TORCH_CHECK(state_indices.scalar_type() == torch::kInt32,
              "state_indices must be int32");
  TORCH_CHECK(parent_ids.scalar_type() == torch::kInt32,
              "parent_ids must be int32");
  TORCH_CHECK(num_accepted_tokens.scalar_type() == torch::kInt32,
              "num_accepted_tokens must be int32");
  TORCH_CHECK(cu_seqlens.scalar_type() == torch::kInt32,
              "cu_seqlens must be int32");

  const int tokens = static_cast<int>(mixed_qkv.size(0));
  const int num_sequences = static_cast<int>(state_indices.size(0));
  const int max_state_tokens = static_cast<int>(state_indices.size(1));
  const int slots = static_cast<int>(state.size(0));
  const int v_heads = static_cast<int>(state.size(1));
  const int v_dim = static_cast<int>(state.size(2));
  const int k_dim = static_cast<int>(state.size(3));
  const int64_t state_slot_stride = state.stride(0);

  TORCH_CHECK(tokens > 0, "decode tokens must be positive");
  TORCH_CHECK(num_sequences > 0, "state_indices must have at least one row");
  TORCH_CHECK(max_state_tokens > 0,
              "state_indices must have at least one state token column");
  TORCH_CHECK(v_dim == 128 && k_dim == 128,
              "SM70 FlashQLA DDTree decode currently supports K=V=128");
  TORCH_CHECK(state.stride(1) == v_dim * k_dim && state.stride(2) == k_dim &&
                  state.stride(3) == 1,
              "state inner layout must be [slots,Hv,V,K] with contiguous "
              "[Hv,V,K] pages");
  TORCH_CHECK(parent_ids.sizes() == state_indices.sizes(),
              "parent_ids must match state_indices shape");
  TORCH_CHECK(num_accepted_tokens.size(0) == num_sequences,
              "num_accepted_tokens must have one entry per sequence");
  TORCH_CHECK(cu_seqlens.size(0) == num_sequences + 1,
              "cu_seqlens must have N + 1 entries");
  TORCH_CHECK(a.size(0) == tokens && b.size(0) == tokens,
              "a/b must match mixed_qkv token count");
  TORCH_CHECK(a.size(1) == v_heads && b.size(1) == v_heads,
              "a/b must have HV columns matching state");
  TORCH_CHECK(A_log.size(0) == v_heads && dt_bias.size(0) == v_heads,
              "A_log/dt_bias must have HV elements");
  TORCH_CHECK(output.size(0) == tokens && output.size(1) == v_heads &&
                  output.size(2) == v_dim,
              "output must have shape [tokens, Hv, V]");

  const int qkv_dim = static_cast<int>(mixed_qkv.size(1));
  const int qk_dim = qkv_dim - v_heads * v_dim;
  TORCH_CHECK(qk_dim > 0 && qk_dim % 2 == 0,
              "mixed_qkv width must be q + k + v");
  TORCH_CHECK(qk_dim % (2 * k_dim) == 0,
              "q/k packed width must be divisible by K");
  const int q_heads = qk_dim / (2 * k_dim);
  TORCH_CHECK(q_heads > 0, "q_heads must be positive");
  TORCH_CHECK(v_heads % q_heads == 0, "Hv must be divisible by Hq");
  const int qkv_stride = static_cast<int>(mixed_qkv.stride(0));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mixed_qkv));
  const int column_groups_per_block =
      get_ddtree_column_groups_per_block(tokens, q_heads, v_heads);
  const auto stream =
      at::cuda::getCurrentCUDAStream(mixed_qkv.device().index()).stream();

  auto dispatch_state = [&](auto mixed_ptr, auto a_ptr, auto b_ptr,
                            auto dt_bias_ptr, auto out_ptr) {
    using scalar_t = std::remove_pointer_t<decltype(mixed_ptr)>;
    using bias_t = std::remove_pointer_t<decltype(dt_bias_ptr)>;
    if (state.scalar_type() == torch::kFloat32) {
      launch_gdn_decode_mixed_qkv_ddtree_state_kv<scalar_t, bias_t, float>(
          mixed_ptr, a_ptr, b_ptr, A_log.data_ptr<float>(), dt_bias_ptr,
          state.data_ptr<float>(), state_indices.data_ptr<int32_t>(),
          parent_ids.data_ptr<int32_t>(),
          num_accepted_tokens.data_ptr<int32_t>(),
          cu_seqlens.data_ptr<int32_t>(), out_ptr, num_sequences,
          max_state_tokens, tokens, slots, state_slot_stride, q_heads, v_heads,
          k_dim, v_dim, qkv_stride, static_cast<float>(scale), use_qk_l2norm,
          column_groups_per_block, stream);
    } else if (state.scalar_type() == torch::kFloat16) {
      launch_gdn_decode_mixed_qkv_ddtree_state_kv<scalar_t, bias_t, at::Half>(
          mixed_ptr, a_ptr, b_ptr, A_log.data_ptr<float>(), dt_bias_ptr,
          state.data_ptr<at::Half>(), state_indices.data_ptr<int32_t>(),
          parent_ids.data_ptr<int32_t>(),
          num_accepted_tokens.data_ptr<int32_t>(),
          cu_seqlens.data_ptr<int32_t>(), out_ptr, num_sequences,
          max_state_tokens, tokens, slots, state_slot_stride, q_heads, v_heads,
          k_dim, v_dim, qkv_stride, static_cast<float>(scale), use_qk_l2norm,
          column_groups_per_block, stream);
    } else {
      launch_gdn_decode_mixed_qkv_ddtree_state_kv<scalar_t,
                                                 bias_t,
                                                 at::BFloat16>(
          mixed_ptr, a_ptr, b_ptr, A_log.data_ptr<float>(), dt_bias_ptr,
          state.data_ptr<at::BFloat16>(), state_indices.data_ptr<int32_t>(),
          parent_ids.data_ptr<int32_t>(),
          num_accepted_tokens.data_ptr<int32_t>(),
          cu_seqlens.data_ptr<int32_t>(), out_ptr, num_sequences,
          max_state_tokens, tokens, slots, state_slot_stride, q_heads, v_heads,
          k_dim, v_dim, qkv_stride, static_cast<float>(scale), use_qk_l2norm,
          column_groups_per_block, stream);
    }
  };

  auto dispatch_dt_bias = [&](auto mixed_ptr, auto a_ptr, auto b_ptr,
                              auto out_ptr) {
    if (dt_bias.scalar_type() == torch::kFloat16) {
      dispatch_state(mixed_ptr, a_ptr, b_ptr, dt_bias.data_ptr<at::Half>(),
                     out_ptr);
    } else if (dt_bias.scalar_type() == torch::kBFloat16) {
      dispatch_state(mixed_ptr, a_ptr, b_ptr,
                     dt_bias.data_ptr<at::BFloat16>(), out_ptr);
    } else {
      dispatch_state(mixed_ptr, a_ptr, b_ptr, dt_bias.data_ptr<float>(),
                     out_ptr);
    }
  };

  if (mixed_qkv.scalar_type() == torch::kFloat16) {
    dispatch_dt_bias(mixed_qkv.data_ptr<at::Half>(), a.data_ptr<at::Half>(),
                     b.data_ptr<at::Half>(), output.data_ptr<at::Half>());
  } else if (mixed_qkv.scalar_type() == torch::kBFloat16) {
    dispatch_dt_bias(mixed_qkv.data_ptr<at::BFloat16>(),
                     a.data_ptr<at::BFloat16>(), b.data_ptr<at::BFloat16>(),
                     output.data_ptr<at::BFloat16>());
  } else {
    dispatch_dt_bias(mixed_qkv.data_ptr<float>(), a.data_ptr<float>(),
                     b.data_ptr<float>(), output.data_ptr<float>());
  }

  check_cuda(cudaGetLastError(), "gdn_decode_mixed_qkv_ddtree_state launch");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gdn_forward", &gdn_forward, "SM70/SM75 FlashQLA GDN forward");
  m.def("gdn_forward_vlk_varlen",
        &gdn_forward_vlk_varlen,
        "SM70/SM75 FlashQLA GDN forward for vLLM [N,Hv,V,K] state");
  m.def("gdn_decode_mixed_qkv_global_state",
        &gdn_decode_mixed_qkv_global_state,
        "SM70/SM75 FlashQLA fused mixed-QKV decode for vLLM global state");
  m.def("gdn_decode_mixed_qkv_ddtree_state",
        &gdn_decode_mixed_qkv_ddtree_state,
        "SM70/SM75 FlashQLA mixed-QKV DDTree decode for vLLM global state");
  m.def("resolve_column_groups_per_block",
        &resolve_column_groups_per_block,
        "Resolve SM70/SM75 FlashQLA GDN column groups per block");
}
