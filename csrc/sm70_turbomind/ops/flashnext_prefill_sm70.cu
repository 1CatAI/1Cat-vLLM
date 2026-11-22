// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// High-throughput SM70 (Volta V100) prefill kernels for Qwen3.8-Flash-Next v6:
// 1. Counting-sort MoE permute + weighted scatter + CTA_M={32,64} scheduler
// 2. Bank-conflict-free (smem_idx17) NVFP4 W13+SwiGLU (CTA_M=64/160 & 32/320)
// 3. L2-resident coalesced atomicAdd NVFP4 W2 (CTA_N=128, 24 warps/SM) +
// finalize
// 4. Fast in-place shared-expert sigmoid gate multiply
// 5. Zero-spill native H=12 QSA staged INT8/E4M3 sparse attention
// 6. Fused QSA staged K/V encoder + hot-cache update + GPU HBM page cache copy
// 7. Tiled NVLink P2P DCP=2 Q-gather & FP16/FP32 LSE-combine+gate for any M
// 8. 3-Link Parallel NVLink P2P TP4 HC prefill + static unpacked IPC weight
// bank

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <pybind11/pybind11.h>
#include <torch/library.h>
#include <torch/types.h>

namespace vllm_fn_sm70 {

// ============================================================================
// Common SM70 MMA, Conflict-Free Shared Memory & Fast Math Helpers
// ============================================================================

#define FN_MMA_884_RC(C, A0, A1, B0, B1)                            \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

#define FN_MMA_884_RR(C, A0, A1, B0, B1)                            \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

#define FN_MMA_884_CR(C, A0, A1, B0, B1)                            \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

// Algebraically conflict-free 128-bit shared memory indexing (+1 uint4 per 4
// rows): Zero bank conflicts on A stores, B stores, A MMA multicast loads, and
// B MMA loads!
__device__ __forceinline__ int smem_idx17(int row, int slot4) {
  const int swz = slot4 ^ (row & 2) ^ ((row >> 3) & 1);
  return (row >> 2) * 17 + (row & 3) * 4 + swz;
}

__device__ __forceinline__ float fast_exp2(float x) {
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

__device__ __forceinline__ uint4 decode_e2m1x8_u4(unsigned packed,
                                                  half2 scale) {
  constexpr unsigned sign = 0x80008000u, em = 0x0e000e00u;
  const unsigned v0 = ((packed << 12) & sign) | ((packed << 9) & em);
  const unsigned v1 = ((packed << 8) & sign) | ((packed << 5) & em);
  const unsigned v2 = ((packed << 4) & sign) | ((packed << 1) & em);
  const unsigned v3 = (packed & sign) | ((packed >> 3) & em);
  const half2 h0 = __hmul2(*reinterpret_cast<const half2*>(&v0), scale);
  const half2 h1 = __hmul2(*reinterpret_cast<const half2*>(&v1), scale);
  const half2 h2 = __hmul2(*reinterpret_cast<const half2*>(&v2), scale);
  const half2 h3 = __hmul2(*reinterpret_cast<const half2*>(&v3), scale);
  return make_uint4(*reinterpret_cast<const unsigned*>(&h0),
                    *reinterpret_cast<const unsigned*>(&h1),
                    *reinterpret_cast<const unsigned*>(&h2),
                    *reinterpret_cast<const unsigned*>(&h3));
}

__device__ __forceinline__ void decode_e2m1x8(unsigned packed, half2 scale,
                                              half2* out) {
  *reinterpret_cast<uint4*>(out) = decode_e2m1x8_u4(packed, scale);
}

__device__ __forceinline__ half nvfp4_fast_hmma_scale(uint8_t scale_code,
                                                      float gs_x256) {
  const unsigned code = scale_code;
  const unsigned short bits = static_cast<unsigned short>(
      ((code & 0x80U) << 8U) | ((code & 0x7fU) << 7U));
  const half prepared =
      __float2half_rn(__half2float(__ushort_as_half(bits)) * gs_x256);
  return __hmul(prepared, __ushort_as_half(0x7400U));
}

// ============================================================================
// Part 1: Counting-Sort MoE Permute + Weighted Scatter + Tile Scheduler
// ============================================================================

constexpr int kMoEExperts = 512;
constexpr int kMoETopK = 10;

struct alignas(16) MoETileDesc {
  int32_t expert;
  int32_t start;
  int32_t count;
  int32_t pad;
};

template <typename IdT>
__global__ __launch_bounds__(256) void moe_hist_kernel(
    const IdT* __restrict__ topk_ids, int32_t* __restrict__ counts,
    int total_slots) {
  __shared__ int32_t s_counts[kMoEExperts];
  const int tx = threadIdx.x;
  s_counts[tx] = 0;
  s_counts[tx + 256] = 0;
  __syncthreads();

  for (int idx = blockIdx.x * 256 + tx; idx < total_slots;
       idx += gridDim.x * 256) {
    const int e = static_cast<int>(topk_ids[idx]);
    if (static_cast<unsigned>(e) < kMoEExperts) {
      atomicAdd(&s_counts[e], 1);
    }
  }
  __syncthreads();

  const int c0 = s_counts[tx];
  const int c1 = s_counts[tx + 256];
  if (c0) atomicAdd(&counts[tx], c0);
  if (c1) atomicAdd(&counts[tx + 256], c1);
}

__global__ __launch_bounds__(512) void moe_scan_schedule_kernel(
    const int32_t* __restrict__ counts, int32_t* __restrict__ offsets,
    int32_t* __restrict__ cursors, MoETileDesc* __restrict__ tiles,
    int32_t* __restrict__ num_tiles_out, int cta_m) {
  __shared__ int32_t s_warp_sum[16];
  __shared__ int32_t s_twarp_sum[16];

  const int tx = threadIdx.x;
  const int lane = tx & 31;
  const int warp = tx >> 5;

  const int c = counts[tx];
  const int tc = (c + cta_m - 1) / cta_m;

  int scan_c = c;
  int scan_tc = tc;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    const int n_c = __shfl_up_sync(0xffffffffu, scan_c, d);
    const int n_tc = __shfl_up_sync(0xffffffffu, scan_tc, d);
    if (lane >= d) {
      scan_c += n_c;
      scan_tc += n_tc;
    }
  }
  if (lane == 31) {
    s_warp_sum[warp] = scan_c;
    s_twarp_sum[warp] = scan_tc;
  }
  __syncthreads();

  if (warp == 0) {
    int wc = (lane < 16) ? s_warp_sum[lane] : 0;
    int wtc = (lane < 16) ? s_twarp_sum[lane] : 0;
#pragma unroll
    for (int d = 1; d < 16; d <<= 1) {
      const int nc = __shfl_up_sync(0xffffffffu, wc, d);
      const int ntc = __shfl_up_sync(0xffffffffu, wtc, d);
      if (lane >= d) {
        wc += nc;
        wtc += ntc;
      }
    }
    if (lane < 16) {
      s_warp_sum[lane] = wc;
      s_twarp_sum[lane] = wtc;
    }
  }
  __syncthreads();

  const int base_c = (warp > 0) ? s_warp_sum[warp - 1] : 0;
  const int base_tc = (warp > 0) ? s_twarp_sum[warp - 1] : 0;
  const int excl_c = base_c + scan_c - c;
  const int excl_tc = base_tc + scan_tc - tc;

  offsets[tx] = excl_c;
  cursors[tx] = excl_c;
  if (tx == kMoEExperts - 1) {
    offsets[kMoEExperts] = excl_c + c;
    *num_tiles_out = excl_tc + tc;
  }

  for (int t = 0; t < tc; ++t) {
    const int rem = c - t * cta_m;
    MoETileDesc desc;
    desc.expert = tx;
    desc.start = excl_c + t * cta_m;
    desc.count = rem < cta_m ? rem : cta_m;
    desc.pad = 0;
    tiles[excl_tc + t] = desc;
  }
}

__global__ __launch_bounds__(512) void moe_scan_schedule_dual_kernel(
    const int32_t* __restrict__ counts, int32_t* __restrict__ cursors,
    MoETileDesc* __restrict__ tiles_a, int32_t* __restrict__ num_tiles_a,
    int cta_m_a, MoETileDesc* __restrict__ tiles_b,
    int32_t* __restrict__ num_tiles_b, int cta_m_b) {
  __shared__ int32_t s_warp_c[16];
  __shared__ int32_t s_warp_ta[16];
  __shared__ int32_t s_warp_tb[16];

  const int tx = threadIdx.x;
  const int lane = tx & 31;
  const int warp = tx >> 5;

  const int c = counts[tx];
  const int ta = (c + cta_m_a - 1) / cta_m_a;
  const int tb = (c + cta_m_b - 1) / cta_m_b;

  int sc = c, sta = ta, stb = tb;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    const int nc = __shfl_up_sync(0xffffffffu, sc, d);
    const int nta = __shfl_up_sync(0xffffffffu, sta, d);
    const int ntb = __shfl_up_sync(0xffffffffu, stb, d);
    if (lane >= d) {
      sc += nc;
      sta += nta;
      stb += ntb;
    }
  }
  if (lane == 31) {
    s_warp_c[warp] = sc;
    s_warp_ta[warp] = sta;
    s_warp_tb[warp] = stb;
  }
  __syncthreads();

  if (warp == 0) {
    int wc = (lane < 16) ? s_warp_c[lane] : 0;
    int wta = (lane < 16) ? s_warp_ta[lane] : 0;
    int wtb = (lane < 16) ? s_warp_tb[lane] : 0;
#pragma unroll
    for (int d = 1; d < 16; d <<= 1) {
      const int nc = __shfl_up_sync(0xffffffffu, wc, d);
      const int nta = __shfl_up_sync(0xffffffffu, wta, d);
      const int ntb = __shfl_up_sync(0xffffffffu, wtb, d);
      if (lane >= d) {
        wc += nc;
        wta += nta;
        wtb += ntb;
      }
    }
    if (lane < 16) {
      s_warp_c[lane] = wc;
      s_warp_ta[lane] = wta;
      s_warp_tb[lane] = wtb;
    }
  }
  __syncthreads();

  const int excl_c = ((warp > 0) ? s_warp_c[warp - 1] : 0) + sc - c;
  const int excl_ta = ((warp > 0) ? s_warp_ta[warp - 1] : 0) + sta - ta;
  const int excl_tb = ((warp > 0) ? s_warp_tb[warp - 1] : 0) + stb - tb;

  cursors[tx] = excl_c;
  if (tx == kMoEExperts - 1) {
    *num_tiles_a = excl_ta + ta;
    *num_tiles_b = excl_tb + tb;
  }

  for (int t = 0; t < ta; ++t) {
    const int rem = c - t * cta_m_a;
    MoETileDesc d{tx, excl_c + t * cta_m_a, rem < cta_m_a ? rem : cta_m_a, 0};
    tiles_a[excl_ta + t] = d;
  }
  for (int t = 0; t < tb; ++t) {
    const int rem = c - t * cta_m_b;
    MoETileDesc d{tx, excl_c + t * cta_m_b, rem < cta_m_b ? rem : cta_m_b, 0};
    tiles_b[excl_tb + t] = d;
  }
}

template <typename IdT>
__global__ __launch_bounds__(256) void moe_scatter_kernel(
    const IdT* __restrict__ topk_ids, const float* __restrict__ topk_weights,
    int32_t* __restrict__ cursors, int32_t* __restrict__ sorted_token_idx,
    float* __restrict__ sorted_weights, int total_slots) {
  __shared__ int32_t s_cnt[kMoEExperts];
  __shared__ int32_t s_base[kMoEExperts];
  const int tx = threadIdx.x;
  const int start = blockIdx.x * 1024;
  const int end = min(start + 1024, total_slots);
  if (start >= end) return;

  s_cnt[tx] = 0;
  s_cnt[tx + 256] = 0;
  __syncthreads();

  int my_e[4];
  int my_local[4];
#pragma unroll
  for (int u = 0; u < 4; ++u) {
    const int idx = start + u * 256 + tx;
    int e = -1;
    int loc = 0;
    if (idx < end) {
      e = static_cast<int>(topk_ids[idx]);
      if (static_cast<unsigned>(e) < kMoEExperts) {
        loc = atomicAdd(&s_cnt[e], 1);
      } else {
        e = -1;
      }
    }
    my_e[u] = e;
    my_local[u] = loc;
  }
  __syncthreads();

  if (s_cnt[tx] > 0) {
    s_base[tx] = atomicAdd(&cursors[tx], s_cnt[tx]);
  }
  if (s_cnt[tx + 256] > 0) {
    s_base[tx + 256] = atomicAdd(&cursors[tx + 256], s_cnt[tx + 256]);
  }
  __syncthreads();

#pragma unroll
  for (int u = 0; u < 4; ++u) {
    const int e = my_e[u];
    if (e >= 0) {
      const int idx = start + u * 256 + tx;
      const int dst = s_base[e] + my_local[u];
      sorted_token_idx[dst] = idx / kMoETopK;
      sorted_weights[dst] = topk_weights[idx];
    }
  }
}

// Deterministic zero-global-atomic token-bucketed histogram + Group-Wave-Major
// scheduler: 1) Eliminates cudaMemsetAsync + global atomicAdd contention in
// hist/scatter 2) Keeps sorted_token_idx monotonically bucketed in ~256-token
// bands per expert 3) Emits independent L2-optimal Wave-Major tile schedules
// for W13 and W2!
template <typename IdT>
__global__ __launch_bounds__(256) void moe_block_hist_kernel(
    const IdT* __restrict__ topk_ids, int32_t* __restrict__ block_counts,
    int slots_per_blk, int total_slots) {
  __shared__ int32_t s_cnt[kMoEExperts];
  const int tx = threadIdx.x;
  s_cnt[tx] = 0;
  s_cnt[tx + 256] = 0;
  __syncthreads();

  const int start = blockIdx.x * slots_per_blk;
  const int end = min(start + slots_per_blk, total_slots);
  for (int idx = start + tx; idx < end; idx += 256) {
    const int e = static_cast<int>(topk_ids[idx]);
    if (static_cast<unsigned>(e) < kMoEExperts) {
      atomicAdd(&s_cnt[e], 1);
    }
  }
  __syncthreads();

  int32_t* dst = block_counts + static_cast<int>(blockIdx.x) * kMoEExperts;
  dst[tx] = s_cnt[tx];
  dst[tx + 256] = s_cnt[tx + 256];
}

__global__ __launch_bounds__(512) void moe_block_scan_wave_schedule_kernel(
    int32_t* __restrict__ block_counts, int num_blks,
    int32_t* __restrict__ offsets, MoETileDesc* __restrict__ tiles_w13,
    MoETileDesc* __restrict__ tiles_w2, int32_t* __restrict__ num_tiles_out,
    int32_t* __restrict__ num_tiles_w2_out, int cta_m_w13, int cta_m_w2,
    int group_w13, int group_w2) {
  __shared__ int32_t s_warp_c[16];
  __shared__ int32_t s_warp_tc[16];
  __shared__ int32_t s_excl_tc[kMoEExperts + 1];
  __shared__ int32_t s_wave_cnt[16];
  __shared__ int32_t s_max_tc;

  const int tx = threadIdx.x;  // 0..511
  const int lane = tx & 31;
  const int warp = tx >> 5;

  int c = 0;
  for (int b = 0; b < num_blks; ++b) {
    int32_t* row = block_counts + b * kMoEExperts;
    const int cnt = row[tx];
    row[tx] = c;
    c += cnt;
  }

  int scan_c = c;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    const int nc = __shfl_up_sync(0xffffffffu, scan_c, d);
    if (lane >= d) scan_c += nc;
  }
  if (lane == 31) s_warp_c[warp] = scan_c;
  __syncthreads();
  if (warp == 0) {
    int wc = (lane < 16) ? s_warp_c[lane] : 0;
#pragma unroll
    for (int d = 1; d < 16; d <<= 1) {
      const int nc = __shfl_up_sync(0xffffffffu, wc, d);
      if (lane >= d) wc += nc;
    }
    if (lane < 16) s_warp_c[lane] = wc;
  }
  __syncthreads();

  const int excl_c = ((warp > 0) ? s_warp_c[warp - 1] : 0) + scan_c - c;
  offsets[tx] = excl_c;
  if (tx == kMoEExperts - 1) {
    offsets[kMoEExperts] = excl_c + c;
  }

  auto build_and_emit = [&](MoETileDesc* out_tiles, int32_t* ntiles_ptr,
                            int cta_m, int group_sz) {
    if (out_tiles == nullptr) return;
    const int tc = (c + cta_m - 1) / cta_m;
    int scan_tc = tc;
    int w_max_tc = tc;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const int ntc = __shfl_up_sync(0xffffffffu, scan_tc, d);
      const int m_tc = __shfl_xor_sync(0xffffffffu, w_max_tc, d);
      if (lane >= d) scan_tc += ntc;
      w_max_tc = max(w_max_tc, m_tc);
    }
    if (lane == 31) {
      s_warp_tc[warp] = scan_tc;
      s_wave_cnt[warp] = w_max_tc;
    }
    __syncthreads();
    if (warp == 0) {
      int wtc = (lane < 16) ? s_warp_tc[lane] : 0;
      int m_tc = (lane < 16) ? s_wave_cnt[lane] : 0;
#pragma unroll
      for (int d = 1; d < 16; d <<= 1) {
        const int ntc = __shfl_up_sync(0xffffffffu, wtc, d);
        const int om = __shfl_xor_sync(0xffffffffu, m_tc, d);
        if (lane >= d) wtc += ntc;
        m_tc = max(m_tc, om);
      }
      if (lane < 16) s_warp_tc[lane] = wtc;
      if (lane == 0) s_max_tc = m_tc;
    }
    __syncthreads();

    const int excl_tc = ((warp > 0) ? s_warp_tc[warp - 1] : 0) + scan_tc - tc;
    s_excl_tc[tx] = excl_tc;
    if (tx == kMoEExperts - 1) {
      const int tot_tiles = excl_tc + tc;
      s_excl_tc[kMoEExperts] = tot_tiles;
      if (ntiles_ptr != nullptr) *ntiles_ptr = tot_tiles;
    }
    __syncthreads();

    const int global_max_tc = s_max_tc;
    if (group_sz <= 1) {
      for (int t = 0; t < tc; ++t) {
        const int rem = c - t * cta_m;
        out_tiles[excl_tc + t] =
            MoETileDesc{tx, excl_c + t * cta_m, rem < cta_m ? rem : cta_m, 0};
      }
      __syncthreads();
      return;
    }
    if (group_sz < 32) {
      const int sub_id = lane / group_sz;
      const int sub_lane = lane % group_sz;
      const uint32_t sub_mask = ((1u << group_sz) - 1u) << (sub_id * group_sz);
      int cursor = s_excl_tc[warp * 32 + sub_id * group_sz];
      for (int k = 0; k < w_max_tc; ++k) {
        const bool act = (k < tc);
        const uint32_t bal = __ballot_sync(0xffffffffu, act) & sub_mask;
        const int pos =
            __popc(bal & ((1u << (sub_id * group_sz + sub_lane)) - 1u));
        if (act) {
          const int rem = c - k * cta_m;
          out_tiles[cursor + pos] =
              MoETileDesc{tx, excl_c + k * cta_m, rem < cta_m ? rem : cta_m, 0};
        }
        cursor += __popc(bal);
      }
      __syncthreads();
      return;
    }
    if (group_sz == 32) {
      int cursor = s_excl_tc[warp * 32];
      for (int k = 0; k < w_max_tc; ++k) {
        const bool act = (k < tc);
        const uint32_t bal = __ballot_sync(0xffffffffu, act);
        const int pos = __popc(bal & ((1u << lane) - 1u));
        if (act) {
          const int rem = c - k * cta_m;
          out_tiles[cursor + pos] =
              MoETileDesc{tx, excl_c + k * cta_m, rem < cta_m ? rem : cta_m, 0};
        }
        cursor += __popc(bal);
      }
      __syncthreads();
      return;
    }
    const int warps_per_grp = group_sz >> 5;
    const int grp_idx = warp / warps_per_grp;
    const int w0 = grp_idx * warps_per_grp;
    int cursor = s_excl_tc[grp_idx * group_sz];
    for (int k = 0; k < global_max_tc; ++k) {
      const bool act = (k < tc);
      const uint32_t bal = __ballot_sync(0xffffffffu, act);
      if (lane == 0) {
        s_wave_cnt[warp] = __popc(bal);
      }
      __syncthreads();
      int prefix = 0;
      int tot = 0;
      for (int w = 0; w < warps_per_grp; ++w) {
        const int wc = s_wave_cnt[w0 + w];
        if (w0 + w < warp) prefix += wc;
        tot += wc;
      }
      if (act) {
        const int pos = prefix + __popc(bal & ((1u << lane) - 1u));
        const int rem = c - k * cta_m;
        out_tiles[cursor + pos] =
            MoETileDesc{tx, excl_c + k * cta_m, rem < cta_m ? rem : cta_m, 0};
      }
      cursor += tot;
      __syncthreads();
    }
  };

  build_and_emit(tiles_w13, num_tiles_out, cta_m_w13, group_w13);
  if (tiles_w2 != nullptr && tiles_w2 != tiles_w13) {
    build_and_emit(tiles_w2, num_tiles_w2_out, cta_m_w2, group_w2);
  } else if (num_tiles_w2_out != nullptr && tx == 0) {
    *num_tiles_w2_out = *num_tiles_out;
  }
}

template <typename IdT>
__global__ __launch_bounds__(256) void moe_block_scatter_kernel(
    const IdT* __restrict__ topk_ids, const float* __restrict__ topk_weights,
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ block_counts,
    int32_t* __restrict__ sorted_token_idx, float* __restrict__ sorted_weights,
    int slots_per_blk, int total_slots) {
  __shared__ int32_t s_cnt[kMoEExperts];
  __shared__ int32_t s_base[kMoEExperts];
  const int tx = threadIdx.x;
  const int32_t* blk_row =
      block_counts + static_cast<int>(blockIdx.x) * kMoEExperts;
  s_cnt[tx] = 0;
  s_cnt[tx + 256] = 0;
  s_base[tx] = offsets[tx] + blk_row[tx];
  s_base[tx + 256] = offsets[tx + 256] + blk_row[tx + 256];
  __syncthreads();

  const int start = blockIdx.x * slots_per_blk;
  const int end = min(start + slots_per_blk, total_slots);
  for (int idx = start + tx; idx < end; idx += 256) {
    const int e = static_cast<int>(topk_ids[idx]);
    if (static_cast<unsigned>(e) < kMoEExperts) {
      const int loc = atomicAdd(&s_cnt[e], 1);
      const int dst = s_base[e] + loc;
      sorted_token_idx[dst] = idx / kMoETopK;
      sorted_weights[dst] = topk_weights[idx];
    }
  }
}

// ============================================================================
// Part 2: Conflict-Free NVFP4 W13 + SwiGLU (CTA_M=64/160 & CTA_M=32/320)
// ============================================================================

// L2-paired CTA_M=128, CTA_N=160 (grid = dim3(2, max_tiles), 640 threads = 20
// warps/SM): 4 M-warps split B-dequant so each thread decodes ONLY 1 uint32_t
// per K=32 stage!
__global__ __launch_bounds__(640, 1) void fn_nvfp4_w13_m128_n160_kernel(
    const half* __restrict__ x,               // [M, 2560]
    const uint32_t* __restrict__ weights,     // [512, 2560, 40]
    const uint8_t* __restrict__ scale_codes,  // [512, 160, 320]
    const float* __restrict__ global_scales,  // [512, 2]
    const int32_t* __restrict__ sorted_token_idx,
    const MoETileDesc* __restrict__ tiles,
    const int32_t* __restrict__ num_tiles_ptr,
    half* __restrict__ out)  // [M * 10, 160]
{
  const int tile_id = blockIdx.y;
  if (tile_id >= *num_tiles_ptr) return;

  const int n_blk = blockIdx.x;  // 0..1 -> 160 raw cols (80 SwiGLU cols) each
  const MoETileDesc desc = tiles[tile_id];
  const int expert = desc.expert;
  const int m_start = desc.start;
  const int m_count = desc.count;

  constexpr int kHidden = 2560;
  constexpr int kRawN = 320;
  constexpr int kInter = 160;
  constexpr int kStepK = 32;
  constexpr int kNumStages = kHidden / kStepK;  // 80

  union SharedW13M128 {
    struct {
      uint4 s_A[2][32 * 17];
      uint4 s_B[2][40 * 17];
    } mainloop;
    half s_out[128][88];
  };
  __shared__ __align__(16) SharedW13M128 smem;
  __shared__ int32_t s_tok[128];

  const int tx = threadIdx.x;   // 0..639
  const int lane = tx & 31;     // 0..31
  const int warp = tx >> 5;     // 0..19
  const int warp_m = warp / 5;  // 0..3 -> 32 rows each = 128 rows
  const int warp_n = warp % 5;  // 0..4 -> 32 raw cols (16 SwiGLU cols) each

  if (tx < 128) {
    s_tok[tx] = (tx < m_count) ? sorted_token_idx[m_start + tx] : -1;
  }
  __syncthreads();

  const int a_r0 = tx >> 2;
  const int a_s0 = tx & 3;
  const int a_smem_idx = (tx < 512) ? smem_idx17(a_r0, a_s0) : 0;
  const half* x_ptr0 = nullptr;
  if (tx < 512) {
    const int tok0 = s_tok[a_r0];
    if (tok0 >= 0) {
      x_ptr0 = x + static_cast<size_t>(tok0) * kHidden + a_s0 * 8;
    }
  }

  const int sub_b = warp_m;  // 0..3 -> 1 uint32_t per thread!
  const int col160_b = warp_n * 32 + lane;
  const int b_store_idx = smem_idx17(col160_b, sub_b);
  const int t32_b = n_blk * 5 + warp_n;

  const uint32_t* exp_w =
      weights + static_cast<size_t>(expert) * kHidden * (kRawN / 8);
  const uint8_t* exp_sc =
      scale_codes + static_cast<size_t>(expert) * (kHidden / 16) * kRawN;
  const float gs_b_x256 =
      __ldg(global_scales + static_cast<size_t>(expert) * 2 + (t32_b & 1)) *
      256.0f;

  {
    const uint4 av0 = x_ptr0 ? __ldg(reinterpret_cast<const uint4*>(x_ptr0))
                             : make_uint4(0, 0, 0, 0);
    const int g = sub_b >> 1;
    const size_t w_off = (static_cast<size_t>(t32_b) * 320 + sub_b) * 32 + lane;
    const size_t s_idx = (static_cast<size_t>(g) * 10 + t32_b) * 32 + lane;
    const uint32_t w0 = __ldg(exp_w + w_off);
    const uint8_t sc = __ldg(exp_sc + s_idx);

    if (tx < 512) {
      smem.mainloop.s_A[0][a_smem_idx] = av0;
    }
    const half s_h = nvfp4_fast_hmma_scale(sc, gs_b_x256);
    alignas(16) half2 dec0[4];
    decode_e2m1x8(w0, __halves2half2(s_h, s_h), dec0);
    smem.mainloop.s_B[0][b_store_idx] = *reinterpret_cast<const uint4*>(dec0);
  }
  __syncthreads();

  const int mma_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int quad = (lane >> 2) & 3;
  const int b_col = warp_n * 32 + quad * 8 + mma_row;
  const int m_base = warp_m * 32;
  const int r0 = m_base + mma_row;
  const int r1 = m_base + 8 + mma_row;
  const int r2 = m_base + 16 + mma_row;
  const int r3 = m_base + 24 + mma_row;

  float acc0[8] = {};
  float acc1[8] = {};
  float acc2[8] = {};
  float acc3[8] = {};

  for (int st = 0; st < kNumStages; ++st) {
    const int cur = st & 1;
    const int nxt = cur ^ 1;

    uint4 av0 = make_uint4(0, 0, 0, 0);
    uint32_t w0 = 0;
    uint8_t sc = 0;
    if (st + 1 < kNumStages) {
      const int k_next = (st + 1) * kStepK;
      if (x_ptr0) av0 = __ldg(reinterpret_cast<const uint4*>(x_ptr0 + k_next));
      const int g = (st + 1) * 2 + (sub_b >> 1);
      const size_t w_off =
          (static_cast<size_t>(t32_b) * 320 + (st + 1) * 4 + sub_b) * 32 + lane;
      const size_t s_idx = (static_cast<size_t>(g) * 10 + t32_b) * 32 + lane;
      w0 = __ldg(exp_w + w_off);
      sc = __ldg(exp_sc + s_idx);
    }

    if (m_base < m_count) {
#pragma unroll
      for (int sub = 0; sub < 2; ++sub) {
        const int sl0 = sub * 2;
        const int sl1 = sub * 2 + 1;
        const uint4 b0 = smem.mainloop.s_B[cur][smem_idx17(b_col, sl0)];
        const uint4 b1 = smem.mainloop.s_B[cur][smem_idx17(b_col, sl1)];
        uint4 a_lo = smem.mainloop.s_A[cur][smem_idx17(r0, sl0)];
        uint4 a_hi = smem.mainloop.s_A[cur][smem_idx17(r0, sl1)];
        uint4 an_lo = smem.mainloop.s_A[cur][smem_idx17(r1, sl0)];
        uint4 an_hi = smem.mainloop.s_A[cur][smem_idx17(r1, sl1)];
        FN_MMA_884_RC(acc0, a_lo.x, a_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc0, a_lo.z, a_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc0, a_hi.x, a_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc0, a_hi.z, a_hi.w, b1.z, b1.w);
        a_lo = smem.mainloop.s_A[cur][smem_idx17(r2, sl0)];
        a_hi = smem.mainloop.s_A[cur][smem_idx17(r2, sl1)];
        FN_MMA_884_RC(acc1, an_lo.x, an_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc1, an_lo.z, an_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc1, an_hi.x, an_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc1, an_hi.z, an_hi.w, b1.z, b1.w);
        an_lo = smem.mainloop.s_A[cur][smem_idx17(r3, sl0)];
        an_hi = smem.mainloop.s_A[cur][smem_idx17(r3, sl1)];
        FN_MMA_884_RC(acc2, a_lo.x, a_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc2, a_lo.z, a_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc2, a_hi.x, a_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc2, a_hi.z, a_hi.w, b1.z, b1.w);
        FN_MMA_884_RC(acc3, an_lo.x, an_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc3, an_lo.z, an_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc3, an_hi.x, an_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc3, an_hi.z, an_hi.w, b1.z, b1.w);
      }
    }

    if (st + 1 < kNumStages) {
      if (tx < 512) {
        smem.mainloop.s_A[nxt][a_smem_idx] = av0;
      }
      const half s_h = nvfp4_fast_hmma_scale(sc, gs_b_x256);
      alignas(16) half2 dec0[4];
      decode_e2m1x8(w0, __halves2half2(s_h, s_h), dec0);
      smem.mainloop.s_B[nxt][b_store_idx] =
          *reinterpret_cast<const uint4*>(dec0);
    }
    __syncthreads();
  }

#define FN_STAGE_SWIGLU_8(ACC, ROW_BASE)                             \
  do {                                                               \
    if ((ROW_BASE) < m_count) {                                      \
      const int r_lane = (lane & 1) + ((lane & 16) ? 4 : 0);         \
      const int c_lane = quad * 4 + ((lane >> 1) & 1);               \
      _Pragma("unroll") for (int p = 0; p < 4; ++p) {                \
        const int r = (ROW_BASE) + r_lane + ((p & 1) ? 2 : 0);       \
        const int c = warp_n * 16 + c_lane + ((p & 2) ? 2 : 0);      \
        const half gate_h = __float2half_rn((ACC)[p * 2]);           \
        const half up_h = __float2half_rn((ACC)[p * 2 + 1]);         \
        const float gf = __half2float(gate_h);                       \
        const half act_h = __float2half_rn(gf / (1.0f + expf(-gf))); \
        smem.s_out[r][c] = __hmul(act_h, up_h);                      \
      }                                                              \
    }                                                                \
  } while (0)
  FN_STAGE_SWIGLU_8(acc0, m_base + 0);
  FN_STAGE_SWIGLU_8(acc1, m_base + 8);
  FN_STAGE_SWIGLU_8(acc2, m_base + 16);
  FN_STAGE_SWIGLU_8(acc3, m_base + 24);
#undef FN_STAGE_SWIGLU_8
  __syncthreads();

#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int vec_id = tx + i * 640;
    const int out_r = vec_id / 10;
    const int out_c8 = (vec_id % 10) * 8;
    if (out_r < m_count) {
      *reinterpret_cast<uint4*>(
          out + static_cast<size_t>(m_start + out_r) * kInter + n_blk * 80 +
          out_c8) = *reinterpret_cast<const uint4*>(&smem.s_out[out_r][out_c8]);
    }
  }
}

// L2-paired CTA_M=64, CTA_N=160 (grid = dim3(2, max_tiles), 2 CTAs/SM = 20
// warps/SM): Inter-CTA barrier latency hiding + zero-spill ping-pong LDS.128
// prefetch!
__global__ __launch_bounds__(320, 2) void fn_nvfp4_w13_m64_n160_kernel(
    const half* __restrict__ x,               // [M, 2560]
    const uint32_t* __restrict__ weights,     // [512, 2560, 40]
    const uint8_t* __restrict__ scale_codes,  // [512, 160, 320]
    const float* __restrict__ global_scales,  // [512, 2]
    const int32_t* __restrict__ sorted_token_idx,
    const MoETileDesc* __restrict__ tiles,
    const int32_t* __restrict__ num_tiles_ptr,
    half* __restrict__ out)  // [M * 10, 160]
{
  const int tile_id = blockIdx.y;
  if (tile_id >= *num_tiles_ptr) return;

  const int n_blk = blockIdx.x;  // 0..1 -> 160 raw cols (80 SwiGLU cols) each
  const MoETileDesc desc = tiles[tile_id];
  const int expert = desc.expert;
  const int m_start = desc.start;
  const int m_count = desc.count;

  constexpr int kHidden = 2560;
  constexpr int kRawN = 320;
  constexpr int kInter = 160;
  constexpr int kStepK = 32;
  constexpr int kNumStages = kHidden / kStepK;  // 80

  union SharedW13M64 {
    struct {
      uint4 s_A[2][16 * 17];
      uint4 s_B[2][40 * 17];
    } mainloop;
    half s_out[64][88];
  };
  __shared__ __align__(16) SharedW13M64 smem;
  __shared__ int32_t s_tok[64];

  const int tx = threadIdx.x;   // 0..319
  const int lane = tx & 31;     // 0..31
  const int warp = tx >> 5;     // 0..9
  const int warp_m = warp / 5;  // 0..1 -> 32 rows each
  const int warp_n = warp % 5;  // 0..4 -> 32 raw cols (16 SwiGLU cols) each

  if (tx < 64) {
    s_tok[tx] = (tx < m_count) ? sorted_token_idx[m_start + tx] : -1;
  }
  __syncthreads();

  const int a_r0 = tx >> 2;
  const int a_s0 = tx & 3;
  const int a_smem_idx = (tx < 256) ? smem_idx17(a_r0, a_s0) : 0;
  const half* x_ptr0 = nullptr;
  if (tx < 256) {
    const int tok0 = s_tok[a_r0];
    if (tok0 >= 0) {
      x_ptr0 = x + static_cast<size_t>(tok0) * kHidden + a_s0 * 8;
    }
  }

  const int sub_b = warp_m;
  const int col160_b = warp_n * 32 + lane;
  const int b_store_idx0 = smem_idx17(col160_b, sub_b * 2);
  const int b_store_idx1 = smem_idx17(col160_b, sub_b * 2 + 1);
  const int t32_b = n_blk * 5 + warp_n;

  const uint32_t* exp_w =
      weights + static_cast<size_t>(expert) * kHidden * (kRawN / 8);
  const uint8_t* exp_sc =
      scale_codes + static_cast<size_t>(expert) * (kHidden / 16) * kRawN;
  const float gs_b_x256 =
      __ldg(global_scales + static_cast<size_t>(expert) * 2 + (t32_b & 1)) *
      256.0f;

  {
    const uint4 av0 = x_ptr0 ? __ldg(reinterpret_cast<const uint4*>(x_ptr0))
                             : make_uint4(0, 0, 0, 0);
    const int g = sub_b;
    const size_t w_off = (static_cast<size_t>(t32_b) * 320 + g * 2) * 32 + lane;
    const size_t s_idx = (static_cast<size_t>(g) * 10 + t32_b) * 32 + lane;
    const uint32_t w0 = __ldg(exp_w + w_off);
    const uint32_t w1 = __ldg(exp_w + w_off + 32);
    const uint8_t sc = __ldg(exp_sc + s_idx);

    if (tx < 256) {
      smem.mainloop.s_A[0][a_smem_idx] = av0;
    }
    const half s_h = nvfp4_fast_hmma_scale(sc, gs_b_x256);
    const half2 scale2 = __halves2half2(s_h, s_h);
    alignas(16) half2 dec0[4], dec1[4];
    decode_e2m1x8(w0, scale2, dec0);
    decode_e2m1x8(w1, scale2, dec1);
    smem.mainloop.s_B[0][b_store_idx0] = *reinterpret_cast<const uint4*>(dec0);
    smem.mainloop.s_B[0][b_store_idx1] = *reinterpret_cast<const uint4*>(dec1);
  }
  __syncthreads();

  const int mma_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int quad = (lane >> 2) & 3;
  const int b_col = warp_n * 32 + quad * 8 + mma_row;
  const int m_base = warp_m * 32;
  const int r0 = m_base + mma_row;

  float acc0[8] = {};
  float acc1[8] = {};
  float acc2[8] = {};
  float acc3[8] = {};

#pragma unroll 1
  for (int st = 0; st < kNumStages; ++st) {
    const int cur = st & 1;
    const int nxt = cur ^ 1;

    uint4 av0 = make_uint4(0, 0, 0, 0);
    uint32_t w0 = 0, w1 = 0;
    uint8_t sc = 0;
    if (st + 1 < kNumStages) {
      const int k_next = (st + 1) * kStepK;
      if (x_ptr0) av0 = __ldg(reinterpret_cast<const uint4*>(x_ptr0 + k_next));
      const int g = (st + 1) * 2 + sub_b;
      const size_t w_off =
          (static_cast<size_t>(t32_b) * 320 + g * 2) * 32 + lane;
      const size_t s_idx = (static_cast<size_t>(g) * 10 + t32_b) * 32 + lane;
      w0 = __ldg(exp_w + w_off);
      w1 = __ldg(exp_w + w_off + 32);
      sc = __ldg(exp_sc + s_idx);
    }

    if (m_base < m_count) {
#pragma unroll
      for (int sub = 0; sub < 2; ++sub) {
        const int sl0 = sub * 2;
        const int sl1 = sub * 2 + 1;
        const uint4 b0 = smem.mainloop.s_B[cur][smem_idx17(b_col, sl0)];
        const uint4 b1 = smem.mainloop.s_B[cur][smem_idx17(b_col, sl1)];
        uint4 a_lo = smem.mainloop.s_A[cur][smem_idx17(r0, sl0)];
        uint4 a_hi = smem.mainloop.s_A[cur][smem_idx17(r0, sl1)];
        FN_MMA_884_RC(acc0, a_lo.x, a_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc0, a_lo.z, a_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc0, a_hi.x, a_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc0, a_hi.z, a_hi.w, b1.z, b1.w);
        a_lo = smem.mainloop.s_A[cur][smem_idx17(r0 + 8, sl0)];
        a_hi = smem.mainloop.s_A[cur][smem_idx17(r0 + 8, sl1)];
        FN_MMA_884_RC(acc1, a_lo.x, a_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc1, a_lo.z, a_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc1, a_hi.x, a_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc1, a_hi.z, a_hi.w, b1.z, b1.w);
        a_lo = smem.mainloop.s_A[cur][smem_idx17(r0 + 16, sl0)];
        a_hi = smem.mainloop.s_A[cur][smem_idx17(r0 + 16, sl1)];
        FN_MMA_884_RC(acc2, a_lo.x, a_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc2, a_lo.z, a_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc2, a_hi.x, a_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc2, a_hi.z, a_hi.w, b1.z, b1.w);
        a_lo = smem.mainloop.s_A[cur][smem_idx17(r0 + 24, sl0)];
        a_hi = smem.mainloop.s_A[cur][smem_idx17(r0 + 24, sl1)];
        FN_MMA_884_RC(acc3, a_lo.x, a_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc3, a_lo.z, a_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc3, a_hi.x, a_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc3, a_hi.z, a_hi.w, b1.z, b1.w);
      }
    }

    if (st + 1 < kNumStages) {
      if (tx < 256) {
        smem.mainloop.s_A[nxt][a_smem_idx] = av0;
      }
      const half s_h = nvfp4_fast_hmma_scale(sc, gs_b_x256);
      const half2 scale2 = __halves2half2(s_h, s_h);
      alignas(16) half2 dec0[4], dec1[4];
      decode_e2m1x8(w0, scale2, dec0);
      decode_e2m1x8(w1, scale2, dec1);
      smem.mainloop.s_B[nxt][b_store_idx0] =
          *reinterpret_cast<const uint4*>(dec0);
      smem.mainloop.s_B[nxt][b_store_idx1] =
          *reinterpret_cast<const uint4*>(dec1);
    }
    __syncthreads();
  }

#define FN_SWIGLU_ELEM_OUT(G_VAL, U_VAL, R_IDX, C_IDX)           \
  do {                                                           \
    const half gate_h = __float2half_rn(G_VAL);                  \
    const half up_h = __float2half_rn(U_VAL);                    \
    const float gf = __half2float(gate_h);                       \
    const half act_h = __float2half_rn(gf / (1.0f + expf(-gf))); \
    smem.s_out[R_IDX][C_IDX] = __hmul(act_h, up_h);              \
  } while (0)
#define FN_STAGE_SWIGLU_8_LIT(ACC, ROW_BASE, WN)                       \
  do {                                                                 \
    if ((ROW_BASE) < m_count) {                                        \
      const int r_b = (ROW_BASE) + (lane & 1) + ((lane & 16) ? 4 : 0); \
      const int c_b = (WN) * 16 + quad * 4 + ((lane >> 1) & 1);        \
      FN_SWIGLU_ELEM_OUT((ACC)[0], (ACC)[1], r_b, c_b);                \
      FN_SWIGLU_ELEM_OUT((ACC)[2], (ACC)[3], r_b + 2, c_b);            \
      FN_SWIGLU_ELEM_OUT((ACC)[4], (ACC)[5], r_b, c_b + 2);            \
      FN_SWIGLU_ELEM_OUT((ACC)[6], (ACC)[7], r_b + 2, c_b + 2);        \
    }                                                                  \
  } while (0)
  FN_STAGE_SWIGLU_8_LIT(acc0, m_base + 0, warp_n);
  FN_STAGE_SWIGLU_8_LIT(acc1, m_base + 8, warp_n);
  FN_STAGE_SWIGLU_8_LIT(acc2, m_base + 16, warp_n);
  FN_STAGE_SWIGLU_8_LIT(acc3, m_base + 24, warp_n);
  __syncthreads();

#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int vec_id = tx + i * 320;
    const int out_r = vec_id / 10;
    const int out_c8 = (vec_id % 10) * 8;
    if (out_r < m_count) {
      *reinterpret_cast<uint4*>(
          out + static_cast<size_t>(m_start + out_r) * kInter + n_blk * 80 +
          out_c8) = *reinterpret_cast<const uint4*>(&smem.s_out[out_r][out_c8]);
    }
  }
}

// Single-gather CTA_M=64, CTA_N=320 W13+SwiGLU (640 threads = 20 warps/SM,
// 52,480 B smem: gathers scattered x[s_tok[r]] ONLY ONCE instead of twice!)
__global__ __launch_bounds__(640, 1) void fn_nvfp4_w13_m64_n320_kernel(
    const half* __restrict__ x,               // [M, 2560]
    const uint32_t* __restrict__ weights,     // [512, 2560, 40]
    const uint8_t* __restrict__ scale_codes,  // [512, 160, 320]
    const float* __restrict__ global_scales,  // [512, 2]
    const int32_t* __restrict__ sorted_token_idx,
    const MoETileDesc* __restrict__ tiles,
    const int32_t* __restrict__ num_tiles_ptr,
    half* __restrict__ out)  // [M * 10, 160]
{
  const int tile_id = blockIdx.x;
  if (tile_id >= *num_tiles_ptr) return;

  const MoETileDesc desc = tiles[tile_id];
  const int expert = desc.expert;
  const int m_start = desc.start;
  const int m_count = desc.count;

  constexpr int kHidden = 2560;
  constexpr int kRawN = 320;
  constexpr int kInter = 160;
  constexpr int kStepK = 32;
  constexpr int kNumStages = kHidden / kStepK;  // 80

  struct SharedW13M64N320 {
    union {
      struct {
        uint4 s_A[2][32 * 17];  // [2][64 rows, 64 halves] = 17,408 B (2
                                // K-stages/buf!)
        uint4 s_B[2][80 * 17];  // [2][320 cols, 32 halves] = 43,520 B
      } mainloop;               // 60,928 B + 256 B = 61,184 B <= 96 KiB!
      half s_out[64][168];
    };
    int32_t s_tok[64];
  };
  extern __shared__ __align__(16) unsigned char w13_n320_raw[];
  SharedW13M64N320& smem = *reinterpret_cast<SharedW13M64N320*>(w13_n320_raw);
  int32_t* s_tok = smem.s_tok;

  const int tx = threadIdx.x;    // 0..639
  const int lane = tx & 31;      // 0..31
  const int warp = tx >> 5;      // 0..19
  const int warp_m = warp / 10;  // 0..1 -> 32 rows each = 64 rows
  const int warp_n = warp % 10;  // 0..9 -> 32 raw cols (16 SwiGLU cols) each

  if (tx < 64) {
    smem.s_tok[tx] = (tx < m_count) ? sorted_token_idx[m_start + tx] : -1;
  }
  __syncthreads();

  // All 512 threads (16 warps!) gather a full 128-byte cache line (64 halves =
  // 8 uint4s) per row of x ONCE every 2 stages (512+ HMMA cycles of HBM latency
  // hiding)!
  const int a_r0 = tx >> 3;      // 0..63 for tx < 512
  const int a_c8 = tx & 7;       // 0..7 -> 64 halves (128 bytes)!
  const int a_half = a_c8 >> 2;  // 0 for cols 0..31, 1 for cols 32..63
  const int a_s0 = a_c8 & 3;     // 0..3 within 32-half stage
  const int a_smem_idx = (tx < 512) ? smem_idx17(a_half * 64 + a_r0, a_s0) : 0;
  const half* x_ptr0 = nullptr;
  if (tx < 512) {
    const int tok0 = s_tok[a_r0];
    if (tok0 >= 0) {
      x_ptr0 = x + static_cast<size_t>(tok0) * kHidden + a_c8 * 8;
    }
  }

  const int sub_b = warp_m;  // 0..1
  const int col320_b = warp_n * 32 + lane;
  const int b_store_idx0 = smem_idx17(col320_b, sub_b * 2);
  const int b_store_idx1 = smem_idx17(col320_b, sub_b * 2 + 1);
  const int t32_b = warp_n;  // 0..9

  const uint32_t* exp_w =
      weights + static_cast<size_t>(expert) * kHidden * (kRawN / 8);
  const uint8_t* exp_sc =
      scale_codes + static_cast<size_t>(expert) * (kHidden / 16) * kRawN;
  const float gs_b_x256 =
      __ldg(global_scales + static_cast<size_t>(expert) * 2 + (t32_b & 1)) *
      256.0f;

  {
    const uint4 av0 = x_ptr0 ? __ldg(reinterpret_cast<const uint4*>(x_ptr0))
                             : make_uint4(0, 0, 0, 0);
    const int g = sub_b;
    const size_t w_off = (static_cast<size_t>(t32_b) * 320 + g * 2) * 32 + lane;
    const size_t s_idx = (static_cast<size_t>(g) * 10 + t32_b) * 32 + lane;
    const uint32_t w0 = __ldg(exp_w + w_off);
    const uint32_t w1 = __ldg(exp_w + w_off + 32);
    const uint8_t sc = __ldg(exp_sc + s_idx);

    if (tx < 512) {
      smem.mainloop.s_A[0][a_smem_idx] = av0;
    }
    const half s_h = nvfp4_fast_hmma_scale(sc, gs_b_x256);
    const half2 scale2 = __halves2half2(s_h, s_h);
    alignas(16) half2 dec0[4], dec1[4];
    decode_e2m1x8(w0, scale2, dec0);
    decode_e2m1x8(w1, scale2, dec1);
    smem.mainloop.s_B[0][b_store_idx0] = *reinterpret_cast<const uint4*>(dec0);
    smem.mainloop.s_B[0][b_store_idx1] = *reinterpret_cast<const uint4*>(dec1);
  }
  __syncthreads();

  const int mma_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int quad = (lane >> 2) & 3;
  const int b_col = warp_n * 32 + quad * 8 + mma_row;
  const int m_base = warp_m * 32;
  const int r0 = m_base + mma_row;

  float acc0[8] = {};
  float acc1[8] = {};
  float acc2[8] = {};
  float acc3[8] = {};

#pragma unroll 2
  for (int pair_st = 0; pair_st < (kNumStages / 2); ++pair_st) {
    const int cur_a = pair_st & 1;
    const int nxt_a = cur_a ^ 1;

    uint4 av_next = make_uint4(0, 0, 0, 0);
    if (pair_st + 1 < (kNumStages / 2) && x_ptr0) {
      av_next =
          __ldg(reinterpret_cast<const uint4*>(x_ptr0 + (pair_st + 1) * 64));
    }

#pragma unroll
    for (int sub_st = 0; sub_st < 2; ++sub_st) {
      const int st = pair_st * 2 + sub_st;
      const int cur_b = sub_st;
      const int nxt_b = sub_st ^ 1;
      const int a_row_off = sub_st * 64;

      uint32_t w0 = 0, w1 = 0;
      uint8_t sc = 0;
      if (st + 1 < kNumStages) {
        const int g = (st + 1) * 2 + sub_b;
        const size_t w_off =
            (static_cast<size_t>(t32_b) * 320 + g * 2) * 32 + lane;
        const size_t s_idx = (static_cast<size_t>(g) * 10 + t32_b) * 32 + lane;
        w0 = __ldg(exp_w + w_off);
        w1 = __ldg(exp_w + w_off + 32);
        sc = __ldg(exp_sc + s_idx);
      }

      if (m_base < m_count) {
#pragma unroll
        for (int sub = 0; sub < 2; ++sub) {
          const int sl0 = sub * 2;
          const int sl1 = sub * 2 + 1;
          const uint4 b0 = smem.mainloop.s_B[cur_b][smem_idx17(b_col, sl0)];
          const uint4 b1 = smem.mainloop.s_B[cur_b][smem_idx17(b_col, sl1)];
          uint4 a_lo =
              smem.mainloop.s_A[cur_a][smem_idx17(a_row_off + r0, sl0)];
          uint4 a_hi =
              smem.mainloop.s_A[cur_a][smem_idx17(a_row_off + r0, sl1)];
          FN_MMA_884_RC(acc0, a_lo.x, a_lo.y, b0.x, b0.y);
          FN_MMA_884_RC(acc0, a_lo.z, a_lo.w, b0.z, b0.w);
          FN_MMA_884_RC(acc0, a_hi.x, a_hi.y, b1.x, b1.y);
          FN_MMA_884_RC(acc0, a_hi.z, a_hi.w, b1.z, b1.w);
          a_lo = smem.mainloop.s_A[cur_a][smem_idx17(a_row_off + r0 + 8, sl0)];
          a_hi = smem.mainloop.s_A[cur_a][smem_idx17(a_row_off + r0 + 8, sl1)];
          FN_MMA_884_RC(acc1, a_lo.x, a_lo.y, b0.x, b0.y);
          FN_MMA_884_RC(acc1, a_lo.z, a_lo.w, b0.z, b0.w);
          FN_MMA_884_RC(acc1, a_hi.x, a_hi.y, b1.x, b1.y);
          FN_MMA_884_RC(acc1, a_hi.z, a_hi.w, b1.z, b1.w);
          a_lo = smem.mainloop.s_A[cur_a][smem_idx17(a_row_off + r0 + 16, sl0)];
          a_hi = smem.mainloop.s_A[cur_a][smem_idx17(a_row_off + r0 + 16, sl1)];
          FN_MMA_884_RC(acc2, a_lo.x, a_lo.y, b0.x, b0.y);
          FN_MMA_884_RC(acc2, a_lo.z, a_lo.w, b0.z, b0.w);
          FN_MMA_884_RC(acc2, a_hi.x, a_hi.y, b1.x, b1.y);
          FN_MMA_884_RC(acc2, a_hi.z, a_hi.w, b1.z, b1.w);
          a_lo = smem.mainloop.s_A[cur_a][smem_idx17(a_row_off + r0 + 24, sl0)];
          a_hi = smem.mainloop.s_A[cur_a][smem_idx17(a_row_off + r0 + 24, sl1)];
          FN_MMA_884_RC(acc3, a_lo.x, a_lo.y, b0.x, b0.y);
          FN_MMA_884_RC(acc3, a_lo.z, a_lo.w, b0.z, b0.w);
          FN_MMA_884_RC(acc3, a_hi.x, a_hi.y, b1.x, b1.y);
          FN_MMA_884_RC(acc3, a_hi.z, a_hi.w, b1.z, b1.w);
        }
      }

      if (sub_st == 1 && pair_st + 1 < (kNumStages / 2) && tx < 512) {
        smem.mainloop.s_A[nxt_a][a_smem_idx] = av_next;
      }
      if (st + 1 < kNumStages) {
        const half s_h = nvfp4_fast_hmma_scale(sc, gs_b_x256);
        const half2 scale2 = __halves2half2(s_h, s_h);
        alignas(16) half2 dec0[4], dec1[4];
        decode_e2m1x8(w0, scale2, dec0);
        decode_e2m1x8(w1, scale2, dec1);
        smem.mainloop.s_B[nxt_b][b_store_idx0] =
            *reinterpret_cast<const uint4*>(dec0);
        smem.mainloop.s_B[nxt_b][b_store_idx1] =
            *reinterpret_cast<const uint4*>(dec1);
      }
      __syncthreads();
    }
  }

  FN_STAGE_SWIGLU_8_LIT(acc0, m_base + 0, warp_n);
  FN_STAGE_SWIGLU_8_LIT(acc1, m_base + 8, warp_n);
  FN_STAGE_SWIGLU_8_LIT(acc2, m_base + 16, warp_n);
  FN_STAGE_SWIGLU_8_LIT(acc3, m_base + 24, warp_n);
  __syncthreads();

#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int vec_id = tx + i * 640;
    const int out_r = vec_id / 20;
    const int out_c8 = (vec_id % 20) * 8;
    if (out_r < m_count) {
      *reinterpret_cast<uint4*>(
          out + static_cast<size_t>(m_start + out_r) * kInter + out_c8) =
          *reinterpret_cast<const uint4*>(&smem.s_out[out_r][out_c8]);
    }
  }
}

// Small-M CTA_M=64, CTA_N=320 W13+SwiGLU (s_A[2][16*17] = 52,480 B smem:
// optimal for 1536 <= M <= 4096 where avg m_count <= 40)
__global__ __launch_bounds__(640, 1) void fn_nvfp4_w13_m64_n320_small_kernel(
    const half* __restrict__ x, const uint32_t* __restrict__ weights,
    const uint8_t* __restrict__ scale_codes,
    const float* __restrict__ global_scales,
    const int32_t* __restrict__ sorted_token_idx,
    const MoETileDesc* __restrict__ tiles,
    const int32_t* __restrict__ num_tiles_ptr, half* __restrict__ out) {
  const int tile_id = blockIdx.x;
  if (tile_id >= *num_tiles_ptr) return;

  const MoETileDesc desc = tiles[tile_id];
  const int expert = desc.expert;
  const int m_start = desc.start;
  const int m_count = desc.count;

  constexpr int kHidden = 2560;
  constexpr int kRawN = 320;
  constexpr int kInter = 160;
  constexpr int kStepK = 32;
  constexpr int kNumStages = kHidden / kStepK;  // 80

  struct SharedW13M64N320Small {
    union {
      struct {
        uint4 s_A[2][16 * 17];  //  8,704 B
        uint4 s_B[2][80 * 17];  // 43,520 B
      } mainloop;               // 52,224 B + 256 B = 52,480 B
      half s_out[64][168];
    };
    int32_t s_tok[64];
  };
  extern __shared__ __align__(16) unsigned char w13_n320_small_raw[];
  SharedW13M64N320Small& smem =
      *reinterpret_cast<SharedW13M64N320Small*>(w13_n320_small_raw);

  const int tx = threadIdx.x;
  const int lane = tx & 31;
  const int warp = tx >> 5;
  const int warp_m = warp / 10;
  const int warp_n = warp % 10;

  if (tx < 64) {
    smem.s_tok[tx] = (tx < m_count) ? sorted_token_idx[m_start + tx] : -1;
  }
  __syncthreads();

  const int a_r0 = tx >> 2;
  const int a_s0 = tx & 3;
  const int a_smem_idx = (tx < 256) ? smem_idx17(a_r0, a_s0) : 0;
  const half* x_ptr0 = nullptr;
  if (tx < 256) {
    const int tok0 = smem.s_tok[a_r0];
    if (tok0 >= 0) {
      x_ptr0 = x + static_cast<size_t>(tok0) * kHidden + a_s0 * 8;
    }
  }

  const int sub_b = warp_m;
  const int col320_b = warp_n * 32 + lane;
  const int b_store_idx0 = smem_idx17(col320_b, sub_b * 2);
  const int b_store_idx1 = smem_idx17(col320_b, sub_b * 2 + 1);
  const int t32_b = warp_n;

  const uint32_t* exp_w =
      weights + static_cast<size_t>(expert) * kHidden * (kRawN / 8);
  const uint8_t* exp_sc =
      scale_codes + static_cast<size_t>(expert) * (kHidden / 16) * kRawN;
  const float gs_b_x256 =
      __ldg(global_scales + static_cast<size_t>(expert) * 2 + (t32_b & 1)) *
      256.0f;

  {
    const uint4 av0 = x_ptr0 ? __ldg(reinterpret_cast<const uint4*>(x_ptr0))
                             : make_uint4(0, 0, 0, 0);
    const int g = sub_b;
    const size_t w_off = (static_cast<size_t>(t32_b) * 320 + g * 2) * 32 + lane;
    const size_t s_idx = (static_cast<size_t>(g) * 10 + t32_b) * 32 + lane;
    const uint32_t w0 = __ldg(exp_w + w_off);
    const uint32_t w1 = __ldg(exp_w + w_off + 32);
    const uint8_t sc = __ldg(exp_sc + s_idx);
    if (tx < 256) smem.mainloop.s_A[0][a_smem_idx] = av0;
    const half s_h = nvfp4_fast_hmma_scale(sc, gs_b_x256);
    const half2 scale2 = __halves2half2(s_h, s_h);
    alignas(16) half2 dec0[4], dec1[4];
    decode_e2m1x8(w0, scale2, dec0);
    decode_e2m1x8(w1, scale2, dec1);
    smem.mainloop.s_B[0][b_store_idx0] = *reinterpret_cast<const uint4*>(dec0);
    smem.mainloop.s_B[0][b_store_idx1] = *reinterpret_cast<const uint4*>(dec1);
  }
  __syncthreads();

  const int mma_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int quad = (lane >> 2) & 3;
  const int b_col = warp_n * 32 + quad * 8 + mma_row;
  const int m_base = warp_m * 32;
  const int r0 = m_base + mma_row;

  float acc0[8] = {}, acc1[8] = {}, acc2[8] = {}, acc3[8] = {};

#pragma unroll 1
  for (int st = 0; st < kNumStages; ++st) {
    const int cur = st & 1;
    const int nxt = cur ^ 1;
    uint4 av0 = make_uint4(0, 0, 0, 0);
    uint32_t w0 = 0, w1 = 0;
    uint8_t sc = 0;
    if (st + 1 < kNumStages) {
      const int k_next = (st + 1) * kStepK;
      if (x_ptr0) av0 = __ldg(reinterpret_cast<const uint4*>(x_ptr0 + k_next));
      const int g = (st + 1) * 2 + sub_b;
      const size_t w_off =
          (static_cast<size_t>(t32_b) * 320 + g * 2) * 32 + lane;
      const size_t s_idx = (static_cast<size_t>(g) * 10 + t32_b) * 32 + lane;
      w0 = __ldg(exp_w + w_off);
      w1 = __ldg(exp_w + w_off + 32);
      sc = __ldg(exp_sc + s_idx);
    }

    if (m_base < m_count) {
#pragma unroll
      for (int sub = 0; sub < 2; ++sub) {
        const int sl0 = sub * 2;
        const int sl1 = sub * 2 + 1;
        const uint4 b0 = smem.mainloop.s_B[cur][smem_idx17(b_col, sl0)];
        const uint4 b1 = smem.mainloop.s_B[cur][smem_idx17(b_col, sl1)];
        uint4 a_lo = smem.mainloop.s_A[cur][smem_idx17(r0, sl0)];
        uint4 a_hi = smem.mainloop.s_A[cur][smem_idx17(r0, sl1)];
        FN_MMA_884_RC(acc0, a_lo.x, a_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc0, a_lo.z, a_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc0, a_hi.x, a_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc0, a_hi.z, a_hi.w, b1.z, b1.w);
        a_lo = smem.mainloop.s_A[cur][smem_idx17(r0 + 8, sl0)];
        a_hi = smem.mainloop.s_A[cur][smem_idx17(r0 + 8, sl1)];
        FN_MMA_884_RC(acc1, a_lo.x, a_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc1, a_lo.z, a_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc1, a_hi.x, a_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc1, a_hi.z, a_hi.w, b1.z, b1.w);
        a_lo = smem.mainloop.s_A[cur][smem_idx17(r0 + 16, sl0)];
        a_hi = smem.mainloop.s_A[cur][smem_idx17(r0 + 16, sl1)];
        FN_MMA_884_RC(acc2, a_lo.x, a_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc2, a_lo.z, a_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc2, a_hi.x, a_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc2, a_hi.z, a_hi.w, b1.z, b1.w);
        a_lo = smem.mainloop.s_A[cur][smem_idx17(r0 + 24, sl0)];
        a_hi = smem.mainloop.s_A[cur][smem_idx17(r0 + 24, sl1)];
        FN_MMA_884_RC(acc3, a_lo.x, a_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc3, a_lo.z, a_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc3, a_hi.x, a_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc3, a_hi.z, a_hi.w, b1.z, b1.w);
      }
    }

    if (st + 1 < kNumStages) {
      if (tx < 256) smem.mainloop.s_A[nxt][a_smem_idx] = av0;
      const half s_h = nvfp4_fast_hmma_scale(sc, gs_b_x256);
      const half2 scale2 = __halves2half2(s_h, s_h);
      alignas(16) half2 dec0[4], dec1[4];
      decode_e2m1x8(w0, scale2, dec0);
      decode_e2m1x8(w1, scale2, dec1);
      smem.mainloop.s_B[nxt][b_store_idx0] =
          *reinterpret_cast<const uint4*>(dec0);
      smem.mainloop.s_B[nxt][b_store_idx1] =
          *reinterpret_cast<const uint4*>(dec1);
    }
    __syncthreads();
  }

  FN_STAGE_SWIGLU_8_LIT(acc0, m_base + 0, warp_n);
  FN_STAGE_SWIGLU_8_LIT(acc1, m_base + 8, warp_n);
  FN_STAGE_SWIGLU_8_LIT(acc2, m_base + 16, warp_n);
  FN_STAGE_SWIGLU_8_LIT(acc3, m_base + 24, warp_n);
  __syncthreads();

#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int vec_id = tx + i * 640;
    const int out_r = vec_id / 20;
    const int out_c8 = (vec_id % 20) * 8;
    if (out_r < m_count) {
      *reinterpret_cast<uint4*>(
          out + static_cast<size_t>(m_start + out_r) * kInter + out_c8) =
          *reinterpret_cast<const uint4*>(&smem.s_out[out_r][out_c8]);
    }
  }
}

// Conflict-free CTA_M=32, CTA_N=320 W13+SwiGLU (48,000 B smem = 2 CTAs = 20
// warps/SM!)
__global__ __launch_bounds__(320, 2) void fn_nvfp4_w13_swiglu_pipelined_kernel(
    const half* __restrict__ x,               // [M, 2560]
    const uint32_t* __restrict__ weights,     // [512, 2560, 40]
    const uint8_t* __restrict__ scale_codes,  // [512, 160, 320]
    const float* __restrict__ global_scales,  // [512, 2]
    const int32_t* __restrict__ sorted_token_idx,
    const MoETileDesc* __restrict__ tiles,
    const int32_t* __restrict__ num_tiles_ptr,
    half* __restrict__ out)  // [M * 10, 160]
{
  const int tile_id = blockIdx.x;
  if (tile_id >= *num_tiles_ptr) return;

  const MoETileDesc desc = tiles[tile_id];
  const int expert = desc.expert;
  const int m_start = desc.start;
  const int m_count = desc.count;

  constexpr int kHidden = 2560;
  constexpr int kRawN = 320;
  constexpr int kInter = 160;
  constexpr int kStepK = 32;
  constexpr int kNumStages = kHidden / kStepK;  // 80

  union SharedW13 {
    struct {
      uint4 s_A[2][8 * 17];
      uint4 s_B[2][80 * 17];
    } mainloop;
    half s_out[32][168];
  };
  __shared__ __align__(16) SharedW13 smem;
  __shared__ int32_t s_tok[32];

  const int tx = threadIdx.x;
  const int lane = tx & 31;
  const int warp = tx >> 5;

  if (tx < 32) {
    s_tok[tx] = (tx < m_count) ? sorted_token_idx[m_start + tx] : -1;
  }
  __syncthreads();

  const int a_r0 = tx >> 2;
  const int a_s0 = tx & 3;
  const int a_smem_idx = smem_idx17(a_r0, a_s0);
  const half* x_ptr0 = nullptr;
  if (tx < 128) {
    const int tok0 = s_tok[a_r0];
    if (tok0 >= 0) {
      x_ptr0 = x + static_cast<size_t>(tok0) * kHidden + a_s0 * 8;
    }
  }

  const int b_store_idx0 = smem_idx17(tx, 0);
  const int b_store_idx1 = smem_idx17(tx, 1);
  const int b_store_idx2 = smem_idx17(tx, 2);
  const int b_store_idx3 = smem_idx17(tx, 3);

  const uint32_t* exp_w =
      weights + static_cast<size_t>(expert) * kHidden * (kRawN / 8);
  const uint8_t* exp_sc =
      scale_codes + static_cast<size_t>(expert) * (kHidden / 16) * kRawN;
  const float gs_b_x256 =
      __ldg(global_scales + static_cast<size_t>(expert) * 2 + (warp & 1)) *
      256.0f;

  const int mma_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int quad = (lane >> 2) & 3;
  const int b_col = warp * 32 + quad * 8 + mma_row;

  {
    const uint4 av0 = x_ptr0 ? __ldg(reinterpret_cast<const uint4*>(x_ptr0))
                             : make_uint4(0, 0, 0, 0);
    const size_t w_off = (static_cast<size_t>(warp) * 320) * 32 + lane;
    const size_t s_idx = static_cast<size_t>(warp) * 32 + lane;
    const uint32_t w0 = __ldg(exp_w + w_off);
    const uint32_t w1 = __ldg(exp_w + w_off + 32);
    const uint32_t w2 = __ldg(exp_w + w_off + 64);
    const uint32_t w3 = __ldg(exp_w + w_off + 96);
    const uint8_t sc0 = __ldg(exp_sc + s_idx);
    const uint8_t sc1 = __ldg(exp_sc + s_idx + 320);

    if (tx < 128) {
      smem.mainloop.s_A[0][a_smem_idx] = av0;
    }
    const half s0 = nvfp4_fast_hmma_scale(sc0, gs_b_x256);
    const half s1 = nvfp4_fast_hmma_scale(sc1, gs_b_x256);
    const half2 scale2_0 = __halves2half2(s0, s0);
    const half2 scale2_1 = __halves2half2(s1, s1);
    alignas(16) half2 dec[4];
    decode_e2m1x8(w0, scale2_0, dec);
    smem.mainloop.s_B[0][b_store_idx0] = *reinterpret_cast<const uint4*>(dec);
    decode_e2m1x8(w1, scale2_0, dec);
    smem.mainloop.s_B[0][b_store_idx1] = *reinterpret_cast<const uint4*>(dec);
    decode_e2m1x8(w2, scale2_1, dec);
    smem.mainloop.s_B[0][b_store_idx2] = *reinterpret_cast<const uint4*>(dec);
    decode_e2m1x8(w3, scale2_1, dec);
    smem.mainloop.s_B[0][b_store_idx3] = *reinterpret_cast<const uint4*>(dec);
  }
  __syncthreads();

  float acc0[8] = {};
  float acc1[8] = {};
  float acc2[8] = {};
  float acc3[8] = {};
  const bool do_m0 = (m_count > 0);
  const bool do_m1 = (m_count > 8);
  const bool do_m2 = (m_count > 16);
  const bool do_m3 = (m_count > 24);

  for (int st = 0; st < kNumStages; ++st) {
    const int cur = st & 1;
    const int nxt = cur ^ 1;

    uint4 av0 = make_uint4(0, 0, 0, 0);
    uint32_t w0 = 0, w1 = 0, w2 = 0, w3 = 0;
    uint8_t sc0 = 0, sc1 = 0;
    if (st + 1 < kNumStages) {
      const int k_next = (st + 1) * kStepK;
      if (x_ptr0) av0 = __ldg(reinterpret_cast<const uint4*>(x_ptr0 + k_next));
      const int g0 = (st + 1) * 2;
      const size_t w_off =
          (static_cast<size_t>(warp) * 320 + g0 * 2) * 32 + lane;
      const size_t s_idx = (static_cast<size_t>(g0) * 10 + warp) * 32 + lane;
      w0 = __ldg(exp_w + w_off);
      w1 = __ldg(exp_w + w_off + 32);
      w2 = __ldg(exp_w + w_off + 64);
      w3 = __ldg(exp_w + w_off + 96);
      sc0 = __ldg(exp_sc + s_idx);
      sc1 = __ldg(exp_sc + s_idx + 320);
    }

#pragma unroll
    for (int sub = 0; sub < 2; ++sub) {
      const int sl0 = sub * 2;
      const int sl1 = sub * 2 + 1;
      const uint4 b0 = smem.mainloop.s_B[cur][smem_idx17(b_col, sl0)];
      const uint4 b1 = smem.mainloop.s_B[cur][smem_idx17(b_col, sl1)];
      if (do_m0) {
        const uint4 a_lo = smem.mainloop.s_A[cur][smem_idx17(mma_row, sl0)];
        const uint4 a_hi = smem.mainloop.s_A[cur][smem_idx17(mma_row, sl1)];
        FN_MMA_884_RC(acc0, a_lo.x, a_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc0, a_lo.z, a_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc0, a_hi.x, a_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc0, a_hi.z, a_hi.w, b1.z, b1.w);
      }
      if (do_m1) {
        const uint4 a_lo = smem.mainloop.s_A[cur][smem_idx17(8 + mma_row, sl0)];
        const uint4 a_hi = smem.mainloop.s_A[cur][smem_idx17(8 + mma_row, sl1)];
        FN_MMA_884_RC(acc1, a_lo.x, a_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc1, a_lo.z, a_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc1, a_hi.x, a_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc1, a_hi.z, a_hi.w, b1.z, b1.w);
      }
      if (do_m2) {
        const uint4 a_lo =
            smem.mainloop.s_A[cur][smem_idx17(16 + mma_row, sl0)];
        const uint4 a_hi =
            smem.mainloop.s_A[cur][smem_idx17(16 + mma_row, sl1)];
        FN_MMA_884_RC(acc2, a_lo.x, a_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc2, a_lo.z, a_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc2, a_hi.x, a_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc2, a_hi.z, a_hi.w, b1.z, b1.w);
      }
      if (do_m3) {
        const uint4 a_lo =
            smem.mainloop.s_A[cur][smem_idx17(24 + mma_row, sl0)];
        const uint4 a_hi =
            smem.mainloop.s_A[cur][smem_idx17(24 + mma_row, sl1)];
        FN_MMA_884_RC(acc3, a_lo.x, a_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc3, a_lo.z, a_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc3, a_hi.x, a_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc3, a_hi.z, a_hi.w, b1.z, b1.w);
      }
    }

    if (st + 1 < kNumStages) {
      if (tx < 128) {
        smem.mainloop.s_A[nxt][a_smem_idx] = av0;
      }
      const half s0 = nvfp4_fast_hmma_scale(sc0, gs_b_x256);
      const half s1 = nvfp4_fast_hmma_scale(sc1, gs_b_x256);
      const half2 scale2_0 = __halves2half2(s0, s0);
      const half2 scale2_1 = __halves2half2(s1, s1);
      alignas(16) half2 dec[4];
      decode_e2m1x8(w0, scale2_0, dec);
      smem.mainloop.s_B[nxt][b_store_idx0] =
          *reinterpret_cast<const uint4*>(dec);
      decode_e2m1x8(w1, scale2_0, dec);
      smem.mainloop.s_B[nxt][b_store_idx1] =
          *reinterpret_cast<const uint4*>(dec);
      decode_e2m1x8(w2, scale2_1, dec);
      smem.mainloop.s_B[nxt][b_store_idx2] =
          *reinterpret_cast<const uint4*>(dec);
      decode_e2m1x8(w3, scale2_1, dec);
      smem.mainloop.s_B[nxt][b_store_idx3] =
          *reinterpret_cast<const uint4*>(dec);
    }
    __syncthreads();
  }

  FN_STAGE_SWIGLU_8_LIT(acc0, 0, warp);
  FN_STAGE_SWIGLU_8_LIT(acc1, 8, warp);
  FN_STAGE_SWIGLU_8_LIT(acc2, 16, warp);
  FN_STAGE_SWIGLU_8_LIT(acc3, 24, warp);
  __syncthreads();

#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int vec_id = tx + i * 320;
    const int out_r = vec_id / 20;
    const int out_c8 = (vec_id % 20) * 8;
    if (out_r < m_count) {
      *reinterpret_cast<uint4*>(
          out + static_cast<size_t>(m_start + out_r) * kInter + out_c8) =
          *reinterpret_cast<const uint4*>(&smem.s_out[out_r][out_c8]);
    }
  }
}

// ============================================================================
// Part 3: Conflict-Free L2-Resident Coalesced atomicAdd NVFP4 W2 + Finalize
// ============================================================================

// grid = dim3(max_tiles, 20): all active SMs work on the SAME n_blk's [M, 128]
// FP32 slice (2 MiB at M=4096, 4 MiB at M=8192 -> 100% resident in V100 6 MiB
// L2)!
__global__ __launch_bounds__(256, 3) void fn_nvfp4_w2_m32_n128_atomic_kernel(
    const half* __restrict__ inter,           // [M * 10, 160]
    const uint32_t* __restrict__ weights,     // [512, 160, 320]
    const uint8_t* __restrict__ scale_codes,  // [512, 10, 2560]
    const float* __restrict__ global_scales,  // [512, 1]
    const int32_t* __restrict__ sorted_token_idx,
    const float* __restrict__ sorted_weights,
    const MoETileDesc* __restrict__ tiles,
    const int32_t* __restrict__ num_tiles_ptr,
    float* __restrict__ accum_f32,  // [20, M, 128]
    int M) {
  const int tile_id = blockIdx.x;
  if (tile_id >= *num_tiles_ptr) return;

  const MoETileDesc desc = tiles[tile_id];
  const int expert = desc.expert;
  const int m_start = desc.start;
  const int m_count = desc.count;
  const int n_blk = blockIdx.y;  // 0..19 -> 128 cols each

  constexpr int kInter = 160;
  constexpr int kHidden = 2560;
  constexpr int kNumStages = 5;

  union SharedW2M32N128 {
    struct {
      uint4 s_A[5][8 * 17];
      uint4 s_B[2][32 * 17];
    } mainloop;
    float s_out[32][132];
  };
  __shared__ __align__(16) SharedW2M32N128 smem;
  __shared__ int32_t s_tok[32];
  __shared__ float s_wt[32];

  const int tx = threadIdx.x;    // 0..255
  const int lane = tx & 31;      // 0..31
  const int warp = tx >> 5;      // 0..7
  const int warp_m = warp >> 2;  // 0..1 -> 16 rows each
  const int warp_n = warp & 3;   // 0..3 -> 32 cols each = 128 cols

  if (tx < 32) {
    if (tx < m_count) {
      s_tok[tx] = sorted_token_idx[m_start + tx];
      s_wt[tx] = sorted_weights[m_start + tx];
    } else {
      s_tok[tx] = -1;
      s_wt[tx] = 0.0f;
    }
  }

  // Load entire [32, 160] A tile (640 uint4s) once into conflict-free s_A[5]
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    const int vec_id = tx + i * 256;
    if (vec_id < 640) {
      const int r = vec_id / 20;
      const int c_vec = vec_id % 20;
      const int st_a = c_vec >> 2;
      const int sl_a = c_vec & 3;
      uint4 val = make_uint4(0, 0, 0, 0);
      if (r < m_count) {
        val = __ldg(reinterpret_cast<const uint4*>(
            inter + static_cast<size_t>(m_start + r) * kInter + c_vec * 8));
      }
      smem.mainloop.s_A[st_a][smem_idx17(r, sl_a)] = val;
    }
  }

  const int sub_b = tx >> 7;       // 0..1
  const int col128_b = tx & 127;   // 0..127
  const int wn_b = col128_b >> 5;  // 0..3
  const int c32_b = col128_b & 31;
  const int t32_b = n_blk * 4 + wn_b;
  const int b_store_idx0 = smem_idx17(col128_b, sub_b * 2);
  const int b_store_idx1 = smem_idx17(col128_b, sub_b * 2 + 1);

  const uint32_t* exp_w =
      weights + static_cast<size_t>(expert) * kInter * (kHidden / 8);
  const uint8_t* exp_sc =
      scale_codes + static_cast<size_t>(expert) * (kInter / 16) * kHidden;
  const float gs_x256 = __ldg(global_scales + expert) * 256.0f;

  {
    const int g = sub_b;
    const size_t w_off = (static_cast<size_t>(t32_b) * 20 + g * 2) * 32 + c32_b;
    const size_t s_idx = (static_cast<size_t>(g) * 80 + t32_b) * 32 + c32_b;
    const uint32_t w0 = __ldg(exp_w + w_off);
    const uint32_t w1 = __ldg(exp_w + w_off + 32);
    const uint8_t sc = __ldg(exp_sc + s_idx);
    const half scalar = nvfp4_fast_hmma_scale(sc, gs_x256);
    const half2 scale2 = __halves2half2(scalar, scalar);
    alignas(16) half2 dec0[4], dec1[4];
    decode_e2m1x8(w0, scale2, dec0);
    decode_e2m1x8(w1, scale2, dec1);
    smem.mainloop.s_B[0][b_store_idx0] = *reinterpret_cast<const uint4*>(dec0);
    smem.mainloop.s_B[0][b_store_idx1] = *reinterpret_cast<const uint4*>(dec1);
  }
  __syncthreads();

  const int mma_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int quad = (lane >> 2) & 3;
  const int b_col128 = warp_n * 32 + quad * 8 + mma_row;
  const int rb0 = warp_m * 16 + mma_row;
  const int rb1 = rb0 + 8;
  const bool do_m0 = (warp_m * 16 < m_count);
  const bool do_m1 = (warp_m * 16 + 8 < m_count);

  float acc0[8] = {};
  float acc1[8] = {};

#pragma unroll
  for (int st = 0; st < kNumStages; ++st) {
    const int cur = st & 1;
    const int nxt = cur ^ 1;
    uint32_t w0 = 0, w1 = 0;
    uint8_t sc = 0;
    if (st + 1 < kNumStages) {
      const int g = (st + 1) * 2 + sub_b;
      const size_t w_off =
          (static_cast<size_t>(t32_b) * 20 + g * 2) * 32 + c32_b;
      const size_t s_idx = (static_cast<size_t>(g) * 80 + t32_b) * 32 + c32_b;
      w0 = __ldg(exp_w + w_off);
      w1 = __ldg(exp_w + w_off + 32);
      sc = __ldg(exp_sc + s_idx);
    }

#pragma unroll
    for (int sub = 0; sub < 2; ++sub) {
      const int sl0 = sub * 2;
      const int sl1 = sub * 2 + 1;
      const uint4 b0 = smem.mainloop.s_B[cur][smem_idx17(b_col128, sl0)];
      const uint4 b1 = smem.mainloop.s_B[cur][smem_idx17(b_col128, sl1)];
      if (do_m0) {
        const uint4 a0_lo = smem.mainloop.s_A[st][smem_idx17(rb0, sl0)];
        const uint4 a0_hi = smem.mainloop.s_A[st][smem_idx17(rb0, sl1)];
        FN_MMA_884_RC(acc0, a0_lo.x, a0_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc0, a0_lo.z, a0_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc0, a0_hi.x, a0_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc0, a0_hi.z, a0_hi.w, b1.z, b1.w);
      }
      if (do_m1) {
        const uint4 a1_lo = smem.mainloop.s_A[st][smem_idx17(rb1, sl0)];
        const uint4 a1_hi = smem.mainloop.s_A[st][smem_idx17(rb1, sl1)];
        FN_MMA_884_RC(acc1, a1_lo.x, a1_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc1, a1_lo.z, a1_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc1, a1_hi.x, a1_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc1, a1_hi.z, a1_hi.w, b1.z, b1.w);
      }
    }

    if (st + 1 < kNumStages) {
      const half scalar = nvfp4_fast_hmma_scale(sc, gs_x256);
      const half2 scale2 = __halves2half2(scalar, scalar);
      alignas(16) half2 dec0[4], dec1[4];
      decode_e2m1x8(w0, scale2, dec0);
      decode_e2m1x8(w1, scale2, dec1);
      smem.mainloop.s_B[nxt][b_store_idx0] =
          *reinterpret_cast<const uint4*>(dec0);
      smem.mainloop.s_B[nxt][b_store_idx1] =
          *reinterpret_cast<const uint4*>(dec1);
    }
    __syncthreads();
  }

  auto stage_w2 = [&](const float (&acc)[8], int row_base) {
    if (row_base >= m_count) return;
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
      const int r = row_base + ((i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1));
      const int c =
          warp_n * 32 + quad * 8 + ((((lane >> 1) & 1) << 1) | ((i >> 2) << 2));
      const float w = s_wt[r];
      const float v0 = __half2float(__float2half_rn(acc[i])) * w;
      const float v1 = __half2float(__float2half_rn(acc[i + 1])) * w;
      smem.s_out[r][c] = v0;
      smem.s_out[r][c + 1] = v1;
    }
  };
  stage_w2(acc0, warp_m * 16);
  stage_w2(acc1, warp_m * 16 + 8);
  __syncthreads();

  // 100% coalesced 32-wide warp atomicAdd into L2-resident [M, 128] slice!
  float* slice_base = accum_f32 + static_cast<size_t>(n_blk) * M * 128;
  const int row_grp = tx >> 7;  // 0..1
  const int col = tx & 127;     // 0..127
  for (int r = row_grp; r < m_count; r += 2) {
    const int tok = s_tok[r];
    if (tok >= 0) {
      atomicAdd(slice_base + static_cast<size_t>(tok) * 128 + col,
                smem.s_out[r][col]);
    }
  }
}

// CTA_M=64, CTA_N=128 W2 atomicAdd (256 threads x 2 CTAs = 16 warps/SM, 0
// spills, full [64, 160] s_A[5] preloaded once, ping-pong LDS.128 prefetch, 4.0
// MiB L2-resident!)
__global__ __launch_bounds__(256, 2) void fn_nvfp4_w2_m64_n128_atomic_kernel(
    const half* __restrict__ inter,           // [M * 10, 160]
    const uint32_t* __restrict__ weights,     // [512, 160, 320]
    const uint8_t* __restrict__ scale_codes,  // [512, 10, 2560]
    const float* __restrict__ global_scales,  // [512, 1]
    const int32_t* __restrict__ sorted_token_idx,
    const float* __restrict__ sorted_weights,
    const MoETileDesc* __restrict__ tiles,
    const int32_t* __restrict__ num_tiles_ptr,
    float* __restrict__ accum_f32,  // [20, M, 128]
    int M) {
  const int tile_id = blockIdx.x;
  if (tile_id >= *num_tiles_ptr) return;

  const MoETileDesc desc = tiles[tile_id];
  const int expert = desc.expert;
  const int m_start = desc.start;
  const int m_count = desc.count;
  const int n_blk = blockIdx.y;  // 0..19 -> 128 cols each

  constexpr int kInter = 160;
  constexpr int kHidden = 2560;
  constexpr int kNumStages = 5;

  union SharedW2M64N128 {
    struct {
      uint4 s_A[5][16 * 17];
      uint4 s_B[2][32 * 17];
    } mainloop;
    float s_out[64][132];
  };
  __shared__ __align__(16) SharedW2M64N128 smem;
  __shared__ int32_t s_tok[64];
  __shared__ float s_wt[64];

  const int tx = threadIdx.x;    // 0..255
  const int lane = tx & 31;      // 0..31
  const int warp = tx >> 5;      // 0..7
  const int warp_m = warp >> 2;  // 0..1 -> 32 rows each
  const int warp_n = warp & 3;   // 0..3 -> 32 cols each = 128 cols

  if (tx < 64) {
    if (tx < m_count) {
      s_tok[tx] = sorted_token_idx[m_start + tx];
      s_wt[tx] = sorted_weights[m_start + tx];
    } else {
      s_tok[tx] = -1;
      s_wt[tx] = 0.0f;
    }
  }

  // Issue all 5 inter A-tile loads AND Stage-0 B-weight loads in parallel
  // before STS!
  uint4 av_pre[5];
#pragma unroll
  for (int i = 0; i < 5; ++i) {
    const int vec_id = tx + i * 256;
    const int r = vec_id / 20;
    const int c_vec = vec_id % 20;
    av_pre[i] = (r < m_count)
                    ? __ldg(reinterpret_cast<const uint4*>(
                          inter + static_cast<size_t>(m_start + r) * kInter +
                          c_vec * 8))
                    : make_uint4(0, 0, 0, 0);
  }

  const int sub_b = tx >> 7;
  const int col128_b = tx & 127;
  const int wn_b = col128_b >> 5;
  const int c32_b = col128_b & 31;
  const int t32_b = n_blk * 4 + wn_b;
  const int b_store_idx0 = smem_idx17(col128_b, sub_b * 2);
  const int b_store_idx1 = smem_idx17(col128_b, sub_b * 2 + 1);

  const uint32_t* exp_w =
      weights + static_cast<size_t>(expert) * kInter * (kHidden / 8);
  const uint8_t* exp_sc =
      scale_codes + static_cast<size_t>(expert) * (kInter / 16) * kHidden;
  const float gs_x256 = __ldg(global_scales + expert) * 256.0f;

  {
    const int g = sub_b;
    const size_t w_off = (static_cast<size_t>(t32_b) * 20 + g * 2) * 32 + c32_b;
    const size_t s_idx = (static_cast<size_t>(g) * 80 + t32_b) * 32 + c32_b;
    const uint32_t w0 = __ldg(exp_w + w_off);
    const uint32_t w1 = __ldg(exp_w + w_off + 32);
    const uint8_t sc = __ldg(exp_sc + s_idx);
#pragma unroll
    for (int i = 0; i < 5; ++i) {
      const int vec_id = tx + i * 256;
      const int r = vec_id / 20;
      const int c_vec = vec_id % 20;
      smem.mainloop.s_A[c_vec >> 2][smem_idx17(r, c_vec & 3)] = av_pre[i];
    }
    const half scalar = nvfp4_fast_hmma_scale(sc, gs_x256);
    const half2 scale2 = __halves2half2(scalar, scalar);
    alignas(16) half2 dec0[4], dec1[4];
    decode_e2m1x8(w0, scale2, dec0);
    decode_e2m1x8(w1, scale2, dec1);
    smem.mainloop.s_B[0][b_store_idx0] = *reinterpret_cast<const uint4*>(dec0);
    smem.mainloop.s_B[0][b_store_idx1] = *reinterpret_cast<const uint4*>(dec1);
  }
  __syncthreads();

  const int mma_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int quad = (lane >> 2) & 3;
  const int b_col128 = warp_n * 32 + quad * 8 + mma_row;
  const int m_base = warp_m * 32;
  const int r0 = m_base + mma_row;
  const int r1 = m_base + 8 + mma_row;
  const int r2 = m_base + 16 + mma_row;
  const int r3 = m_base + 24 + mma_row;

  float acc0[8] = {};
  float acc1[8] = {};
  float acc2[8] = {};
  float acc3[8] = {};

#pragma unroll
  for (int st = 0; st < kNumStages; ++st) {
    const int cur = st & 1;
    const int nxt = cur ^ 1;
    uint32_t w0 = 0, w1 = 0;
    uint8_t sc = 0;
    if (st + 1 < kNumStages) {
      const int g = (st + 1) * 2 + sub_b;
      const size_t w_off =
          (static_cast<size_t>(t32_b) * 20 + g * 2) * 32 + c32_b;
      const size_t s_idx = (static_cast<size_t>(g) * 80 + t32_b) * 32 + c32_b;
      w0 = __ldg(exp_w + w_off);
      w1 = __ldg(exp_w + w_off + 32);
      sc = __ldg(exp_sc + s_idx);
    }

    if (m_base < m_count) {
#pragma unroll
      for (int sub = 0; sub < 2; ++sub) {
        const int sl0 = sub * 2;
        const int sl1 = sub * 2 + 1;
        const uint4 b0 = smem.mainloop.s_B[cur][smem_idx17(b_col128, sl0)];
        const uint4 b1 = smem.mainloop.s_B[cur][smem_idx17(b_col128, sl1)];
        uint4 a_lo = smem.mainloop.s_A[st][smem_idx17(r0, sl0)];
        uint4 a_hi = smem.mainloop.s_A[st][smem_idx17(r0, sl1)];
        uint4 an_lo = smem.mainloop.s_A[st][smem_idx17(r1, sl0)];
        uint4 an_hi = smem.mainloop.s_A[st][smem_idx17(r1, sl1)];
        FN_MMA_884_RC(acc0, a_lo.x, a_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc0, a_lo.z, a_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc0, a_hi.x, a_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc0, a_hi.z, a_hi.w, b1.z, b1.w);
        a_lo = smem.mainloop.s_A[st][smem_idx17(r2, sl0)];
        a_hi = smem.mainloop.s_A[st][smem_idx17(r2, sl1)];
        FN_MMA_884_RC(acc1, an_lo.x, an_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc1, an_lo.z, an_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc1, an_hi.x, an_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc1, an_hi.z, an_hi.w, b1.z, b1.w);
        an_lo = smem.mainloop.s_A[st][smem_idx17(r3, sl0)];
        an_hi = smem.mainloop.s_A[st][smem_idx17(r3, sl1)];
        FN_MMA_884_RC(acc2, a_lo.x, a_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc2, a_lo.z, a_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc2, a_hi.x, a_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc2, a_hi.z, a_hi.w, b1.z, b1.w);
        FN_MMA_884_RC(acc3, an_lo.x, an_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc3, an_lo.z, an_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc3, an_hi.x, an_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc3, an_hi.z, an_hi.w, b1.z, b1.w);
      }
    }

    if (st + 1 < kNumStages) {
      const half scalar = nvfp4_fast_hmma_scale(sc, gs_x256);
      const half2 scale2 = __halves2half2(scalar, scalar);
      alignas(16) half2 dec0[4], dec1[4];
      decode_e2m1x8(w0, scale2, dec0);
      decode_e2m1x8(w1, scale2, dec1);
      smem.mainloop.s_B[nxt][b_store_idx0] =
          *reinterpret_cast<const uint4*>(dec0);
      smem.mainloop.s_B[nxt][b_store_idx1] =
          *reinterpret_cast<const uint4*>(dec1);
    }
    __syncthreads();
  }

  float* slice_base = accum_f32 + static_cast<size_t>(n_blk) * M * 128;
  const int row_grp = tx >> 7;
  const int col = tx & 127;

  auto stage_all = [&](const float (&acc)[8], int global_r) {
    if (global_r >= m_count) return;
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
      const int r_off = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
      const int r = global_r + r_off;
      const int c =
          warp_n * 32 + quad * 8 + ((((lane >> 1) & 1) << 1) | ((i >> 2) << 2));
      const float w = s_wt[r];
      *reinterpret_cast<float2*>(&smem.s_out[r][c]) =
          make_float2(__half2float(__float2half_rn(acc[i])) * w,
                      __half2float(__float2half_rn(acc[i + 1])) * w);
    }
  };
  stage_all(acc0, m_base + 0);
  stage_all(acc1, m_base + 8);
  stage_all(acc2, m_base + 16);
  stage_all(acc3, m_base + 24);
  __syncthreads();

  for (int r = row_grp; r < m_count; r += 2) {
    const int tok = s_tok[r];
    if (tok >= 0) {
      atomicAdd(slice_base + static_cast<size_t>(tok) * 128 + col,
                smem.s_out[r][col]);
    }
  }
}

// Zero-spill CTA_M=96, CTA_N=320 W13+SwiGLU with staggered 48+48 128B
// full-cache-line pair_st gather (640 threads = 20 warps/SM, <=96 regs/thread,
// 70,016 B smem, 0 spills: 6 acc tiles/warp = 1.5x HMMA/barrier, 100% 128B
// cache lines on x, -30% W13 HBM!)
__global__ __launch_bounds__(640, 1) void fn_nvfp4_w13_m96_n320_kernel(
    const half* __restrict__ x,               // [M, 2560]
    const uint32_t* __restrict__ weights,     // [512, 2560, 40]
    const uint8_t* __restrict__ scale_codes,  // [512, 160, 320]
    const float* __restrict__ global_scales,  // [512, 2]
    const int32_t* __restrict__ sorted_token_idx,
    const MoETileDesc* __restrict__ tiles,
    const int32_t* __restrict__ num_tiles_ptr,
    half* __restrict__ out)  // [M * 10, 160]
{
  const int tile_id = blockIdx.x;
  if (tile_id >= *num_tiles_ptr) return;

  const MoETileDesc desc = tiles[tile_id];
  const int expert = desc.expert;
  const int m_start = desc.start;
  const int m_count = desc.count;
  if (m_count <= 0) return;

  constexpr int kHidden = 2560;
  constexpr int kRawN = 320;
  constexpr int kInter = 160;
  constexpr int kStepK = 32;
  constexpr int kNumStages = kHidden / kStepK;  // 80

  struct SharedW13M96N320 {
    union {
      struct {
        uint4 s_A[2][48 * 17];  // [2][192 logical rows, 4 uint4s] = 26,112 B
        uint4 s_B[2][80 * 17];  // [2][320 cols, 4 uint4s] = 43,520 B
      } mainloop;               // 69,632 B + 384 B = 70,016 B <= 96 KiB!
      half s_out[96][168];
    };
    int32_t s_tok[96];
  };
  extern __shared__ __align__(16) unsigned char w13_m96_raw[];
  SharedW13M96N320& smem = *reinterpret_cast<SharedW13M96N320*>(w13_m96_raw);

  const int tx = threadIdx.x;    // 0..639
  const int lane = tx & 31;      // 0..31
  const int warp = tx >> 5;      // 0..19
  const int warp_m = warp / 10;  // 0..1 -> 48 rows each = 96 rows!
  const int warp_n = warp % 10;  // 0..9 -> 32 raw cols (16 SwiGLU cols) each

  if (tx < 96) {
    smem.s_tok[tx] = (tx < m_count) ? sorted_token_idx[m_start + tx] : -1;
  }
  __syncthreads();

  // 384 threads (12 warps) gather a FULL 128-byte cache line (64 halves = 8
  // uint4s) per row: rows 0..47 on sub_st=0, rows 48..95 on sub_st=1!
  const int a_r_lo = tx >> 3;    // 0..47 for tx < 384
  const int a_c8 = tx & 7;       // 0..7 -> 64 halves (128B full cache line!)
  const int a_half = a_c8 >> 2;  // 0 for cols 0..31, 1 for cols 32..63
  const int a_s0 = a_c8 & 3;     // 0..3 within 32-half stage
  const int a_smem_idx0 =
      (tx < 384) ? smem_idx17(a_half * 96 + a_r_lo, a_s0) : 0;
  const int a_smem_idx1 =
      (tx < 384) ? smem_idx17(a_half * 96 + 48 + a_r_lo, a_s0) : 0;
  const half* x_ptr0 = nullptr;
  const half* x_ptr1 = nullptr;
  if (tx < 384) {
    const int tok0 = smem.s_tok[a_r_lo];
    if (tok0 >= 0) {
      x_ptr0 = x + static_cast<size_t>(tok0) * kHidden + a_c8 * 8;
    }
    const int tok1 = smem.s_tok[48 + a_r_lo];
    if (tok1 >= 0) {
      x_ptr1 = x + static_cast<size_t>(tok1) * kHidden + a_c8 * 8;
    }
  }

  const int sub_b = warp_m;  // 0..1 across all 640 threads!
  const int col320_b = warp_n * 32 + lane;
  const int b_store_idx0 = smem_idx17(col320_b, sub_b * 2);
  const int b_store_idx1 = smem_idx17(col320_b, sub_b * 2 + 1);
  const int t32_b = warp_n;

  const uint32_t* exp_w =
      weights + static_cast<size_t>(expert) * kHidden * (kRawN / 8);
  const uint8_t* exp_sc =
      scale_codes + static_cast<size_t>(expert) * (kHidden / 16) * kRawN;
  const float gs_b_x256 =
      __ldg(global_scales + static_cast<size_t>(expert) * 2 + (t32_b & 1)) *
      256.0f;

  {
    const uint4 av0 = x_ptr0 ? __ldg(reinterpret_cast<const uint4*>(x_ptr0))
                             : make_uint4(0, 0, 0, 0);
    const uint4 av1 = x_ptr1 ? __ldg(reinterpret_cast<const uint4*>(x_ptr1))
                             : make_uint4(0, 0, 0, 0);
    const int g = sub_b;
    const size_t w_off = (static_cast<size_t>(t32_b) * 320 + g * 2) * 32 + lane;
    const size_t s_idx = (static_cast<size_t>(g) * 10 + t32_b) * 32 + lane;
    const uint32_t w0 = __ldg(exp_w + w_off);
    const uint32_t w1 = __ldg(exp_w + w_off + 32);
    const uint8_t sc = __ldg(exp_sc + s_idx);
    if (tx < 384) {
      smem.mainloop.s_A[0][a_smem_idx0] = av0;
      smem.mainloop.s_A[0][a_smem_idx1] = av1;
    }
    const half s_h = nvfp4_fast_hmma_scale(sc, gs_b_x256);
    const half2 scale2 = __halves2half2(s_h, s_h);
    alignas(16) half2 dec0[4], dec1[4];
    decode_e2m1x8(w0, scale2, dec0);
    decode_e2m1x8(w1, scale2, dec1);
    smem.mainloop.s_B[0][b_store_idx0] = *reinterpret_cast<const uint4*>(dec0);
    smem.mainloop.s_B[0][b_store_idx1] = *reinterpret_cast<const uint4*>(dec1);
  }
  __syncthreads();

  const int mma_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int quad = (lane >> 2) & 3;
  const int b_col = warp_n * 32 + quad * 8 + mma_row;
  const int m_base = warp_m * 48;

  float acc0[8] = {}, acc1[8] = {}, acc2[8] = {};
  float acc3[8] = {}, acc4[8] = {}, acc5[8] = {};

#pragma unroll 1
  for (int pair_st = 0; pair_st < (kNumStages / 2); ++pair_st) {
    const int cur_a = pair_st & 1;
    const int nxt_a = cur_a ^ 1;

#pragma unroll
    for (int sub_st = 0; sub_st < 2; ++sub_st) {
      const int st = pair_st * 2 + sub_st;
      const int cur_b = sub_st;
      const int nxt_b = sub_st ^ 1;
      const int a_row_off = sub_st * 96 + m_base + mma_row;

      uint4 av_next = make_uint4(0, 0, 0, 0);
      if (pair_st + 1 < (kNumStages / 2)) {
        const half* xp = (sub_st == 0) ? x_ptr0 : x_ptr1;
        if (xp) {
          av_next =
              __ldg(reinterpret_cast<const uint4*>(xp + (pair_st + 1) * 64));
        }
      }

      uint32_t w0 = 0, w1 = 0;
      uint8_t sc = 0;
      if (st + 1 < kNumStages) {
        const int g = (st + 1) * 2 + sub_b;
        const size_t w_off =
            (static_cast<size_t>(t32_b) * 320 + g * 2) * 32 + lane;
        const size_t s_idx = (static_cast<size_t>(g) * 10 + t32_b) * 32 + lane;
        w0 = __ldg(exp_w + w_off);
        w1 = __ldg(exp_w + w_off + 32);
        sc = __ldg(exp_sc + s_idx);
      }

      if (m_base < m_count) {
#pragma unroll
        for (int sub = 0; sub < 2; ++sub) {
          const int sl0 = sub * 2;
          const int sl1 = sub * 2 + 1;
          const uint4 b0 = smem.mainloop.s_B[cur_b][smem_idx17(b_col, sl0)];
          const uint4 b1 = smem.mainloop.s_B[cur_b][smem_idx17(b_col, sl1)];
          uint4 a_lo = smem.mainloop.s_A[cur_a][smem_idx17(a_row_off, sl0)];
          uint4 a_hi = smem.mainloop.s_A[cur_a][smem_idx17(a_row_off, sl1)];
          FN_MMA_884_RC(acc0, a_lo.x, a_lo.y, b0.x, b0.y);
          FN_MMA_884_RC(acc0, a_lo.z, a_lo.w, b0.z, b0.w);
          FN_MMA_884_RC(acc0, a_hi.x, a_hi.y, b1.x, b1.y);
          FN_MMA_884_RC(acc0, a_hi.z, a_hi.w, b1.z, b1.w);
          a_lo = smem.mainloop.s_A[cur_a][smem_idx17(a_row_off + 8, sl0)];
          a_hi = smem.mainloop.s_A[cur_a][smem_idx17(a_row_off + 8, sl1)];
          FN_MMA_884_RC(acc1, a_lo.x, a_lo.y, b0.x, b0.y);
          FN_MMA_884_RC(acc1, a_lo.z, a_lo.w, b0.z, b0.w);
          FN_MMA_884_RC(acc1, a_hi.x, a_hi.y, b1.x, b1.y);
          FN_MMA_884_RC(acc1, a_hi.z, a_hi.w, b1.z, b1.w);
          a_lo = smem.mainloop.s_A[cur_a][smem_idx17(a_row_off + 16, sl0)];
          a_hi = smem.mainloop.s_A[cur_a][smem_idx17(a_row_off + 16, sl1)];
          FN_MMA_884_RC(acc2, a_lo.x, a_lo.y, b0.x, b0.y);
          FN_MMA_884_RC(acc2, a_lo.z, a_lo.w, b0.z, b0.w);
          FN_MMA_884_RC(acc2, a_hi.x, a_hi.y, b1.x, b1.y);
          FN_MMA_884_RC(acc2, a_hi.z, a_hi.w, b1.z, b1.w);
          a_lo = smem.mainloop.s_A[cur_a][smem_idx17(a_row_off + 24, sl0)];
          a_hi = smem.mainloop.s_A[cur_a][smem_idx17(a_row_off + 24, sl1)];
          FN_MMA_884_RC(acc3, a_lo.x, a_lo.y, b0.x, b0.y);
          FN_MMA_884_RC(acc3, a_lo.z, a_lo.w, b0.z, b0.w);
          FN_MMA_884_RC(acc3, a_hi.x, a_hi.y, b1.x, b1.y);
          FN_MMA_884_RC(acc3, a_hi.z, a_hi.w, b1.z, b1.w);
          a_lo = smem.mainloop.s_A[cur_a][smem_idx17(a_row_off + 32, sl0)];
          a_hi = smem.mainloop.s_A[cur_a][smem_idx17(a_row_off + 32, sl1)];
          FN_MMA_884_RC(acc4, a_lo.x, a_lo.y, b0.x, b0.y);
          FN_MMA_884_RC(acc4, a_lo.z, a_lo.w, b0.z, b0.w);
          FN_MMA_884_RC(acc4, a_hi.x, a_hi.y, b1.x, b1.y);
          FN_MMA_884_RC(acc4, a_hi.z, a_hi.w, b1.z, b1.w);
          a_lo = smem.mainloop.s_A[cur_a][smem_idx17(a_row_off + 40, sl0)];
          a_hi = smem.mainloop.s_A[cur_a][smem_idx17(a_row_off + 40, sl1)];
          FN_MMA_884_RC(acc5, a_lo.x, a_lo.y, b0.x, b0.y);
          FN_MMA_884_RC(acc5, a_lo.z, a_lo.w, b0.z, b0.w);
          FN_MMA_884_RC(acc5, a_hi.x, a_hi.y, b1.x, b1.y);
          FN_MMA_884_RC(acc5, a_hi.z, a_hi.w, b1.z, b1.w);
        }
      }

      if (pair_st + 1 < (kNumStages / 2) && tx < 384) {
        smem.mainloop.s_A[nxt_a][(sub_st == 0) ? a_smem_idx0 : a_smem_idx1] =
            av_next;
      }
      if (st + 1 < kNumStages) {
        const half s_h = nvfp4_fast_hmma_scale(sc, gs_b_x256);
        const half2 scale2 = __halves2half2(s_h, s_h);
        alignas(16) half2 dec0[4], dec1[4];
        decode_e2m1x8(w0, scale2, dec0);
        decode_e2m1x8(w1, scale2, dec1);
        smem.mainloop.s_B[nxt_b][b_store_idx0] =
            *reinterpret_cast<const uint4*>(dec0);
        smem.mainloop.s_B[nxt_b][b_store_idx1] =
            *reinterpret_cast<const uint4*>(dec1);
      }
      __syncthreads();
    }
  }

  FN_STAGE_SWIGLU_8_LIT(acc0, m_base + 0, warp_n);
  FN_STAGE_SWIGLU_8_LIT(acc1, m_base + 8, warp_n);
  FN_STAGE_SWIGLU_8_LIT(acc2, m_base + 16, warp_n);
  FN_STAGE_SWIGLU_8_LIT(acc3, m_base + 24, warp_n);
  FN_STAGE_SWIGLU_8_LIT(acc4, m_base + 32, warp_n);
  FN_STAGE_SWIGLU_8_LIT(acc5, m_base + 40, warp_n);
#undef FN_STAGE_SWIGLU_8_LIT
#undef FN_SWIGLU_ELEM_OUT
  __syncthreads();

#pragma unroll
  for (int i = 0; i < 3; ++i) {
    const int vec_id = tx + i * 640;
    const int out_r = vec_id / 20;
    const int out_c8 = (vec_id % 20) * 8;
    if (out_r < m_count) {
      *reinterpret_cast<uint4*>(
          out + static_cast<size_t>(m_start + out_r) * kInter + out_c8) =
          *reinterpret_cast<const uint4*>(&smem.s_out[out_r][out_c8]);
    }
  }
}

// High-occupancy CTA_M=96, CTA_N=128 W2 atomicAdd (384 threads x 2 CTAs = 24
// warps/SM, 31,232 B smem/CTA -> 62.5 KiB <= 96 KiB, 0 spills, -30% weight HBM
// & dequant!)
__global__ __launch_bounds__(384, 2) void fn_nvfp4_w2_m96_n128_atomic_kernel(
    const half* __restrict__ inter, const uint32_t* __restrict__ weights,
    const uint8_t* __restrict__ scale_codes,
    const float* __restrict__ global_scales,
    const int32_t* __restrict__ sorted_token_idx,
    const float* __restrict__ sorted_weights,
    const MoETileDesc* __restrict__ tiles,
    const int32_t* __restrict__ num_tiles_ptr,
    float* __restrict__ accum_f32,  // [20, M, 128]
    int M) {
  const int tile_id = blockIdx.x;
  if (tile_id >= *num_tiles_ptr) return;
  const int n_blk = blockIdx.y;  // 0..19 -> 128 cols each

  const MoETileDesc desc = tiles[tile_id];
  const int expert = desc.expert;
  const int m_start = desc.start;
  const int m_count = desc.count;
  if (m_count <= 0) return;

  constexpr int kInter = 160;
  constexpr int kHidden = 2560;
  constexpr int kNumStages = 5;

  union SharedW2M96N128 {
    struct {
      uint4 s_A[2][24 * 17];  // 13,056 B
      uint4 s_B[2][32 * 17];  // 17,408 B
    } mainloop;               // 30,464 B + 768 B = 31,232 B -> 2 CTAs/SM!
    float s_out[48][132];     // 25,344 B
  };
  __shared__ __align__(16) SharedW2M96N128 smem;
  __shared__ int32_t s_tok[96];
  __shared__ float s_wt[96];

  const int tx = threadIdx.x;    // 0..383
  const int lane = tx & 31;      // 0..31
  const int warp = tx >> 5;      // 0..11
  const int warp_m = warp >> 2;  // 0..2 -> 32 rows each = 96 rows
  const int warp_n = warp & 3;   // 0..3 -> 32 cols each = 128 cols

  if (tx < 96) {
    if (tx < m_count) {
      s_tok[tx] = sorted_token_idx[m_start + tx];
      s_wt[tx] = sorted_weights[m_start + tx];
    } else {
      s_tok[tx] = -1;
      s_wt[tx] = 0.0f;
    }
  }

  // Exact 1 uint4 per thread per stage ([96, 32] = 384 uint4s)!
  const int a_r = tx >> 2;  // 0..95
  const int a_s = tx & 3;   // 0..3
  const int a_smem_idx = smem_idx17(a_r, a_s);
  const half* inter_ptr =
      (a_r < m_count)
          ? (inter + static_cast<size_t>(m_start + a_r) * kInter + a_s * 8)
          : nullptr;

  const bool do_b = (tx < 256);
  const int sub_b = tx >> 7;      // 0..1 when do_b
  const int col128_b = tx & 127;  // 0..127
  const int wn_b = col128_b >> 5;
  const int c32_b = col128_b & 31;
  const int t32_b = n_blk * 4 + wn_b;
  const int b_store_idx0 = do_b ? smem_idx17(col128_b, sub_b * 2) : 0;
  const int b_store_idx1 = do_b ? smem_idx17(col128_b, sub_b * 2 + 1) : 0;

  const uint32_t* exp_w =
      weights + static_cast<size_t>(expert) * kInter * (kHidden / 8);
  const uint8_t* exp_sc =
      scale_codes + static_cast<size_t>(expert) * (kInter / 16) * kHidden;
  const float gs_x256 = __ldg(global_scales + expert) * 256.0f;

  {
    const uint4 av = inter_ptr
                         ? __ldg(reinterpret_cast<const uint4*>(inter_ptr))
                         : make_uint4(0, 0, 0, 0);
    smem.mainloop.s_A[0][a_smem_idx] = av;
    if (do_b) {
      const int g = sub_b;
      const size_t w_off =
          (static_cast<size_t>(t32_b) * 20 + g * 2) * 32 + c32_b;
      const size_t s_idx = (static_cast<size_t>(g) * 80 + t32_b) * 32 + c32_b;
      const uint32_t w0 = __ldg(exp_w + w_off);
      const uint32_t w1 = __ldg(exp_w + w_off + 32);
      const uint8_t sc = __ldg(exp_sc + s_idx);
      const half scalar = nvfp4_fast_hmma_scale(sc, gs_x256);
      const half2 scale2 = __halves2half2(scalar, scalar);
      alignas(16) half2 dec0[4], dec1[4];
      decode_e2m1x8(w0, scale2, dec0);
      decode_e2m1x8(w1, scale2, dec1);
      smem.mainloop.s_B[0][b_store_idx0] =
          *reinterpret_cast<const uint4*>(dec0);
      smem.mainloop.s_B[0][b_store_idx1] =
          *reinterpret_cast<const uint4*>(dec1);
    }
  }
  __syncthreads();

  const int mma_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int quad = (lane >> 2) & 3;
  const int b_col128 = warp_n * 32 + quad * 8 + mma_row;
  const int m_base = warp_m * 32;
  const int r0 = m_base + mma_row;
  const int r1 = m_base + 8 + mma_row;
  const int r2 = m_base + 16 + mma_row;
  const int r3 = m_base + 24 + mma_row;

  float acc0[8] = {}, acc1[8] = {}, acc2[8] = {}, acc3[8] = {};

#pragma unroll 1
  for (int st = 0; st < kNumStages; ++st) {
    const int cur = st & 1;
    const int nxt = cur ^ 1;
    uint4 av = make_uint4(0, 0, 0, 0);
    uint32_t w0 = 0, w1 = 0;
    uint8_t sc = 0;
    if (st + 1 < kNumStages) {
      if (inter_ptr) {
        av = __ldg(reinterpret_cast<const uint4*>(inter_ptr + (st + 1) * 32));
      }
      if (do_b) {
        const int g = (st + 1) * 2 + sub_b;
        const size_t w_off =
            (static_cast<size_t>(t32_b) * 20 + g * 2) * 32 + c32_b;
        const size_t s_idx = (static_cast<size_t>(g) * 80 + t32_b) * 32 + c32_b;
        w0 = __ldg(exp_w + w_off);
        w1 = __ldg(exp_w + w_off + 32);
        sc = __ldg(exp_sc + s_idx);
      }
    }

    if (m_base < m_count) {
#pragma unroll
      for (int sub = 0; sub < 2; ++sub) {
        const int sl0 = sub * 2;
        const int sl1 = sub * 2 + 1;
        const uint4 b0 = smem.mainloop.s_B[cur][smem_idx17(b_col128, sl0)];
        const uint4 b1 = smem.mainloop.s_B[cur][smem_idx17(b_col128, sl1)];
        uint4 a_lo = smem.mainloop.s_A[cur][smem_idx17(r0, sl0)];
        uint4 a_hi = smem.mainloop.s_A[cur][smem_idx17(r0, sl1)];
        FN_MMA_884_RC(acc0, a_lo.x, a_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc0, a_lo.z, a_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc0, a_hi.x, a_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc0, a_hi.z, a_hi.w, b1.z, b1.w);
        a_lo = smem.mainloop.s_A[cur][smem_idx17(r1, sl0)];
        a_hi = smem.mainloop.s_A[cur][smem_idx17(r1, sl1)];
        FN_MMA_884_RC(acc1, a_lo.x, a_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc1, a_lo.z, a_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc1, a_hi.x, a_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc1, a_hi.z, a_hi.w, b1.z, b1.w);
        a_lo = smem.mainloop.s_A[cur][smem_idx17(r2, sl0)];
        a_hi = smem.mainloop.s_A[cur][smem_idx17(r2, sl1)];
        FN_MMA_884_RC(acc2, a_lo.x, a_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc2, a_lo.z, a_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc2, a_hi.x, a_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc2, a_hi.z, a_hi.w, b1.z, b1.w);
        a_lo = smem.mainloop.s_A[cur][smem_idx17(r3, sl0)];
        a_hi = smem.mainloop.s_A[cur][smem_idx17(r3, sl1)];
        FN_MMA_884_RC(acc3, a_lo.x, a_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc3, a_lo.z, a_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc3, a_hi.x, a_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc3, a_hi.z, a_hi.w, b1.z, b1.w);
      }
    }

    if (st + 1 < kNumStages) {
      smem.mainloop.s_A[nxt][a_smem_idx] = av;
      if (do_b) {
        const half scalar = nvfp4_fast_hmma_scale(sc, gs_x256);
        const half2 scale2 = __halves2half2(scalar, scalar);
        alignas(16) half2 dec0[4], dec1[4];
        decode_e2m1x8(w0, scale2, dec0);
        decode_e2m1x8(w1, scale2, dec1);
        smem.mainloop.s_B[nxt][b_store_idx0] =
            *reinterpret_cast<const uint4*>(dec0);
        smem.mainloop.s_B[nxt][b_store_idx1] =
            *reinterpret_cast<const uint4*>(dec1);
      }
    }
    __syncthreads();
  }

  float* slice_base = accum_f32 + static_cast<size_t>(n_blk) * M * 128;
  const int row_grp =
      tx >> 7;               // 0..2 (3 rows per iteration across 384 threads!)
  const int col = tx & 127;  // 0..127

  auto stage_sub = [&](const float (&acc)[8], int local_r_base, int global_r) {
    if (global_r >= m_count) return;
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
      const int r_off = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
      const int local_r = local_r_base + r_off;
      const int c =
          warp_n * 32 + quad * 8 + ((((lane >> 1) & 1) << 1) | ((i >> 2) << 2));
      const float w = s_wt[global_r + r_off];
      smem.s_out[local_r][c] = __half2float(__float2half_rn(acc[i])) * w;
      smem.s_out[local_r][c + 1] =
          __half2float(__float2half_rn(acc[i + 1])) * w;
    }
  };

  // Pass 0: rows 0..47 (warp_m == 0 writes 0..31, warp_m == 1 writes 32..47)
  if (warp_m == 0) {
    stage_sub(acc0, 0, 0);
    stage_sub(acc1, 8, 8);
    stage_sub(acc2, 16, 16);
    stage_sub(acc3, 24, 24);
  } else if (warp_m == 1) {
    stage_sub(acc0, 32, 32);
    stage_sub(acc1, 40, 40);
  }
  __syncthreads();
  const int pass0_lim = min(48, m_count);
  for (int r = row_grp; r < pass0_lim; r += 3) {
    const int tok = s_tok[r];
    if (tok >= 0) {
      atomicAdd(slice_base + static_cast<size_t>(tok) * 128 + col,
                smem.s_out[r][col]);
    }
  }

  if (m_count > 48) {
    __syncthreads();
    if (warp_m == 1) {
      stage_sub(acc2, 0, 48);
      stage_sub(acc3, 8, 56);
    } else if (warp_m == 2) {
      stage_sub(acc0, 16, 64);
      stage_sub(acc1, 24, 72);
      stage_sub(acc2, 32, 80);
      stage_sub(acc3, 40, 88);
    }
    __syncthreads();
    const int pass1_lim = m_count - 48;
    for (int r = row_grp; r < pass1_lim; r += 3) {
      const int tok = s_tok[48 + r];
      if (tok >= 0) {
        atomicAdd(slice_base + static_cast<size_t>(tok) * 128 + col,
                  smem.s_out[r][col]);
      }
    }
  }
}

// Zero-spill L2-resident CTA_M=128, CTA_N=64 W2 atomicAdd (256x2 = 512 threads,
// 128 regs -> 0 spills, 4.0 MiB <= 6.0 MiB L2 slice at M=16384, -40% W2 HBM!)
__global__ __launch_bounds__(256, 2) void fn_nvfp4_w2_m128_n64_atomic_kernel(
    const half* __restrict__ inter, const uint32_t* __restrict__ weights,
    const uint8_t* __restrict__ scale_codes,
    const float* __restrict__ global_scales,
    const int32_t* __restrict__ sorted_token_idx,
    const float* __restrict__ sorted_weights,
    const MoETileDesc* __restrict__ tiles,
    const int32_t* __restrict__ num_tiles_ptr,
    float* __restrict__ accum_f32,  // [40, M, 64] -> 4.0 MiB/slice at M=16384!
    int M) {
  const int tile_id = blockIdx.x;
  if (tile_id >= *num_tiles_ptr) return;
  const int n_blk = blockIdx.y;  // 0..39 -> 64 cols each = 2560 cols

  const MoETileDesc desc = tiles[tile_id];
  const int expert = desc.expert;
  const int m_start = desc.start;
  const int m_count = desc.count;
  if (m_count <= 0) return;

  constexpr int kInter = 160;
  constexpr int kHidden = 2560;
  constexpr int kNumStages = 5;

  union SharedW2M128N64 {
    struct {
      uint4 s_A[2][32 * 17];  // 17,408 B
      uint4 s_B[2][16 * 17];  //  8,704 B
    } mainloop;               // 26,112 B + 1,024 B = 27,136 B -> 2 CTAs/SM!
    float s_out[64][68];      // 17,408 B
  };
  __shared__ __align__(16) SharedW2M128N64 smem;
  __shared__ int32_t s_tok[128];
  __shared__ float s_wt[128];

  const int tx = threadIdx.x;    // 0..255
  const int lane = tx & 31;      // 0..31
  const int warp = tx >> 5;      // 0..7
  const int warp_m = warp >> 1;  // 0..3 -> 32 rows each = 128 rows
  const int warp_n = warp & 1;   // 0..1 -> 32 cols each = 64 cols

  if (tx < 128) {
    if (tx < m_count) {
      s_tok[tx] = sorted_token_idx[m_start + tx];
      s_wt[tx] = sorted_weights[m_start + tx];
    } else {
      s_tok[tx] = -1;
      s_wt[tx] = 0.0f;
    }
  }

  const int a_r0 = tx >> 2;    // 0..63
  const int a_r1 = a_r0 + 64;  // 64..127
  const int a_s = tx & 3;      // 0..3
  const int a_smem0 = smem_idx17(a_r0, a_s);
  const int a_smem1 = smem_idx17(a_r1, a_s);
  const half* inter_ptr0 =
      (a_r0 < m_count)
          ? (inter + static_cast<size_t>(m_start + a_r0) * kInter + a_s * 8)
          : nullptr;
  const half* inter_ptr1 =
      (a_r1 < m_count)
          ? (inter + static_cast<size_t>(m_start + a_r1) * kInter + a_s * 8)
          : nullptr;

  const int sub4 = tx >> 6;       // 0..3
  const int col64_b = tx & 63;    // 0..63
  const int wn_b = col64_b >> 5;  // 0..1
  const int c32_b = col64_b & 31;
  const int t32_b = n_blk * 2 + wn_b;
  const int b_store_idx = smem_idx17(col64_b, sub4);

  const uint32_t* exp_w =
      weights + static_cast<size_t>(expert) * kInter * (kHidden / 8);
  const uint8_t* exp_sc =
      scale_codes + static_cast<size_t>(expert) * (kInter / 16) * kHidden;
  const float gs_x256 = __ldg(global_scales + expert) * 256.0f;

  {
    const uint4 av0 = inter_ptr0
                          ? __ldg(reinterpret_cast<const uint4*>(inter_ptr0))
                          : make_uint4(0, 0, 0, 0);
    const uint4 av1 = inter_ptr1
                          ? __ldg(reinterpret_cast<const uint4*>(inter_ptr1))
                          : make_uint4(0, 0, 0, 0);
    smem.mainloop.s_A[0][a_smem0] = av0;
    smem.mainloop.s_A[0][a_smem1] = av1;
    const int g = sub4 >> 1;
    const size_t w_off = (static_cast<size_t>(t32_b) * 20 + sub4) * 32 + c32_b;
    const size_t s_idx = (static_cast<size_t>(g) * 80 + t32_b) * 32 + c32_b;
    const uint32_t w0 = __ldg(exp_w + w_off);
    const uint8_t sc = __ldg(exp_sc + s_idx);
    const half scalar = nvfp4_fast_hmma_scale(sc, gs_x256);
    alignas(16) half2 dec0[4];
    decode_e2m1x8(w0, __halves2half2(scalar, scalar), dec0);
    smem.mainloop.s_B[0][b_store_idx] = *reinterpret_cast<const uint4*>(dec0);
  }
  __syncthreads();

  const int mma_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int quad = (lane >> 2) & 3;
  const int b_col64 = warp_n * 32 + quad * 8 + mma_row;
  const int m_base = warp_m * 32;
  const int r0 = m_base + mma_row;
  const int r1 = m_base + 8 + mma_row;
  const int r2 = m_base + 16 + mma_row;
  const int r3 = m_base + 24 + mma_row;

  float acc0[8] = {}, acc1[8] = {}, acc2[8] = {}, acc3[8] = {};

#pragma unroll 1
  for (int st = 0; st < kNumStages; ++st) {
    const int cur = st & 1;
    const int nxt = cur ^ 1;
    uint4 av0 = make_uint4(0, 0, 0, 0);
    uint4 av1 = make_uint4(0, 0, 0, 0);
    uint32_t w0 = 0;
    uint8_t sc = 0;
    if (st + 1 < kNumStages) {
      const int k_next = (st + 1) * 32;
      if (inter_ptr0)
        av0 = __ldg(reinterpret_cast<const uint4*>(inter_ptr0 + k_next));
      if (inter_ptr1)
        av1 = __ldg(reinterpret_cast<const uint4*>(inter_ptr1 + k_next));
      const int g = (st + 1) * 2 + (sub4 >> 1);
      const size_t w_off =
          (static_cast<size_t>(t32_b) * 20 + (st + 1) * 4 + sub4) * 32 + c32_b;
      const size_t s_idx = (static_cast<size_t>(g) * 80 + t32_b) * 32 + c32_b;
      w0 = __ldg(exp_w + w_off);
      sc = __ldg(exp_sc + s_idx);
    }

    if (m_base < m_count) {
#pragma unroll
      for (int sub = 0; sub < 2; ++sub) {
        const int sl0 = sub * 2;
        const int sl1 = sub * 2 + 1;
        const uint4 b0 = smem.mainloop.s_B[cur][smem_idx17(b_col64, sl0)];
        const uint4 b1 = smem.mainloop.s_B[cur][smem_idx17(b_col64, sl1)];
        uint4 a_lo = smem.mainloop.s_A[cur][smem_idx17(r0, sl0)];
        uint4 a_hi = smem.mainloop.s_A[cur][smem_idx17(r0, sl1)];
        uint4 an_lo = smem.mainloop.s_A[cur][smem_idx17(r1, sl0)];
        uint4 an_hi = smem.mainloop.s_A[cur][smem_idx17(r1, sl1)];
        FN_MMA_884_RC(acc0, a_lo.x, a_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc0, a_lo.z, a_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc0, a_hi.x, a_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc0, a_hi.z, a_hi.w, b1.z, b1.w);
        a_lo = smem.mainloop.s_A[cur][smem_idx17(r2, sl0)];
        a_hi = smem.mainloop.s_A[cur][smem_idx17(r2, sl1)];
        FN_MMA_884_RC(acc1, an_lo.x, an_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc1, an_lo.z, an_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc1, an_hi.x, an_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc1, an_hi.z, an_hi.w, b1.z, b1.w);
        an_lo = smem.mainloop.s_A[cur][smem_idx17(r3, sl0)];
        an_hi = smem.mainloop.s_A[cur][smem_idx17(r3, sl1)];
        FN_MMA_884_RC(acc2, a_lo.x, a_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc2, a_lo.z, a_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc2, a_hi.x, a_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc2, a_hi.z, a_hi.w, b1.z, b1.w);
        FN_MMA_884_RC(acc3, an_lo.x, an_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc3, an_lo.z, an_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc3, an_hi.x, an_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc3, an_hi.z, an_hi.w, b1.z, b1.w);
      }
    }

    if (st + 1 < kNumStages) {
      smem.mainloop.s_A[nxt][a_smem0] = av0;
      smem.mainloop.s_A[nxt][a_smem1] = av1;
      const half scalar = nvfp4_fast_hmma_scale(sc, gs_x256);
      alignas(16) half2 dec0[4];
      decode_e2m1x8(w0, __halves2half2(scalar, scalar), dec0);
      smem.mainloop.s_B[nxt][b_store_idx] =
          *reinterpret_cast<const uint4*>(dec0);
    }
    __syncthreads();
  }

  float* slice_base = accum_f32 + static_cast<size_t>(n_blk) * M * 64;
  const int row_grp =
      tx >> 6;              // 0..3 (4 rows per iteration across 256 threads!)
  const int col = tx & 63;  // 0..63

  auto stage_sub = [&](const float (&acc)[8], int local_r_base, int global_r) {
    if (global_r >= m_count) return;
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
      const int r_off = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
      const int local_r = local_r_base + r_off;
      const int c =
          warp_n * 32 + quad * 8 + ((((lane >> 1) & 1) << 1) | ((i >> 2) << 2));
      const float w = s_wt[global_r + r_off];
      smem.s_out[local_r][c] = __half2float(__float2half_rn(acc[i])) * w;
      smem.s_out[local_r][c + 1] =
          __half2float(__float2half_rn(acc[i + 1])) * w;
    }
  };

  if (warp_m < 2) {
    const int base = warp_m * 32;
    stage_sub(acc0, base + 0, base + 0);
    stage_sub(acc1, base + 8, base + 8);
    stage_sub(acc2, base + 16, base + 16);
    stage_sub(acc3, base + 24, base + 24);
  }
  __syncthreads();
  const int pass0_lim = min(64, m_count);
  for (int r = row_grp; r < pass0_lim; r += 4) {
    const int tok = s_tok[r];
    if (tok >= 0) {
      atomicAdd(slice_base + static_cast<size_t>(tok) * 64 + col,
                smem.s_out[r][col]);
    }
  }
  if (m_count > 64) {
    __syncthreads();
    if (warp_m >= 2) {
      const int loc = (warp_m - 2) * 32;
      const int glb = warp_m * 32;
      stage_sub(acc0, loc + 0, glb + 0);
      stage_sub(acc1, loc + 8, glb + 8);
      stage_sub(acc2, loc + 16, glb + 16);
      stage_sub(acc3, loc + 24, glb + 24);
    }
    __syncthreads();
    const int pass1_lim = m_count - 64;
    for (int r = row_grp; r < pass1_lim; r += 4) {
      const int tok = s_tok[64 + r];
      if (tok >= 0) {
        atomicAdd(slice_base + static_cast<size_t>(tok) * 64 + col,
                  smem.s_out[r][col]);
      }
    }
  }
}

// 3-CTA/SM CTA_M=64, CTA_N=128 W2 atomicAdd (256x3 = 768 threads = 24 warps/SM,
// 26,624 B smem/CTA -> 78.0 KiB <= 96 KiB, <=80 regs, 0 stack, 0 spills!)
template <bool USE_F16X2 = false>
__global__ __launch_bounds__(256, 3) void fn_nvfp4_w2_m64_n128_3cta_kernel(
    const half* __restrict__ inter, const uint32_t* __restrict__ weights,
    const uint8_t* __restrict__ scale_codes,
    const float* __restrict__ global_scales,
    const int32_t* __restrict__ sorted_token_idx,
    const float* __restrict__ sorted_weights,
    const MoETileDesc* __restrict__ tiles,
    const int32_t* __restrict__ num_tiles_ptr, void* __restrict__ accum_buf,
    int m_stride) {
  const int tile_id = blockIdx.x;
  if (tile_id >= *num_tiles_ptr) return;

  const MoETileDesc desc = tiles[tile_id];
  const int expert = desc.expert;
  const int m_start = desc.start;
  const int m_count = desc.count;
  if (m_count <= 0) return;
  const int n_blk = blockIdx.y;

  constexpr int kInter = 160;
  constexpr int kHidden = 2560;
  constexpr int kNumStages = 5;

  union SharedW2M64N128_3CTA {
    struct {
      uint4 s_A[2][16 * 17];  //  8,704 B
      uint4 s_B[2][32 * 17];  // 17,408 B
    } mainloop;
    float s_out[48][132];     // 25,344 B <= 26,112 B (1-pass for mc_ep <= 48!)
    half s_out_f16[64][136];  // 17,408 B <= 26,112 B!
  };
  __shared__ __align__(16) SharedW2M64N128_3CTA smem;
  __shared__ int32_t s_tok[66];
  __shared__ float s_wt[65];

  const int tx = threadIdx.x;

  if (tx < 64) {
    if (tx < m_count) {
      s_tok[tx] = sorted_token_idx[m_start + tx];
      s_wt[tx] = sorted_weights[m_start + tx];
    } else {
      s_tok[tx] = -1;
      s_wt[tx] = 0.0f;
    }
    if (tx == 0) {
      s_tok[64] = m_count;
      s_tok[65] = m_start;
      s_wt[64] = __ldg(global_scales + expert) * 256.0f;
    }
  }
  __syncthreads();

  const uint32_t inter_off =
      static_cast<uint32_t>(s_tok[65] + (tx >> 2)) * kInter + (tx & 3) * 8;
  const bool valid_a = ((tx >> 2) < s_tok[64]);
  const uint32_t w_off =
      static_cast<uint32_t>(expert) * (kInter * (kHidden / 8)) +
      static_cast<uint32_t>((blockIdx.y * 4 + ((tx & 127) >> 5)) * 20 +
                            (tx >> 7) * 2) *
          32u +
      static_cast<uint32_t>(tx & 31);
  const uint32_t sc_off =
      static_cast<uint32_t>(expert) * ((kInter / 16) * kHidden) +
      static_cast<uint32_t>((tx >> 7) * 80 + blockIdx.y * 4 +
                            ((tx & 127) >> 5)) *
          32u +
      static_cast<uint32_t>(tx & 31);

  {
    const uint4 av =
        valid_a ? __ldg(reinterpret_cast<const uint4*>(inter + inter_off))
                : make_uint4(0, 0, 0, 0);
    const uint32_t w0 = __ldg(weights + w_off);
    const uint32_t w1 = __ldg(weights + w_off + 32u);
    const uint8_t sc = __ldg(scale_codes + sc_off);
    smem.mainloop.s_A[0][smem_idx17(tx >> 2, tx & 3)] = av;
    const half scalar = nvfp4_fast_hmma_scale(sc, s_wt[64]);
    const half2 scale2 = __halves2half2(scalar, scalar);
    smem.mainloop.s_B[0][smem_idx17(tx & 127, (tx >> 7) * 2)] =
        decode_e2m1x8_u4(w0, scale2);
    smem.mainloop.s_B[0][smem_idx17(tx & 127, (tx >> 7) * 2 + 1)] =
        decode_e2m1x8_u4(w1, scale2);
  }
  __syncthreads();

  float acc0[8] = {}, acc1[8] = {}, acc2[8] = {}, acc3[8] = {};

#pragma unroll 1
  for (int st = 0; st < kNumStages; ++st) {
    const int cur = st & 1;
    const int nxt = cur ^ 1;
    uint4 av = make_uint4(0, 0, 0, 0);
    uint32_t w0 = 0, w1 = 0;
    uint8_t sc = 0;
    if (st + 1 < kNumStages) {
      const uint32_t nxt_st = static_cast<uint32_t>(st + 1);
      if (valid_a) {
        av = __ldg(
            reinterpret_cast<const uint4*>(inter + inter_off + nxt_st * 32u));
      }
      w0 = __ldg(weights + w_off + nxt_st * 128u);
      w1 = __ldg(weights + w_off + nxt_st * 128u + 32u);
      sc = __ldg(scale_codes + sc_off + nxt_st * 5120u);
    }

    if ((tx >> 7) * 32 < s_tok[64]) {
      const int lane_i = tx & 31;
      const int mma_row_i = (lane_i & 3) + ((lane_i & 16) ? 4 : 0);
      const int b_col128 =
          ((tx >> 5) & 3) * 32 + ((lane_i >> 2) & 3) * 8 + mma_row_i;
      const int r0 = (tx >> 7) * 32 + mma_row_i;
#pragma unroll
      for (int sl = 0; sl < 4; ++sl) {
        const uint4 b = smem.mainloop.s_B[cur][smem_idx17(b_col128, sl)];
        uint4 a = smem.mainloop.s_A[cur][smem_idx17(r0, sl)];
        FN_MMA_884_RC(acc0, a.x, a.y, b.x, b.y);
        FN_MMA_884_RC(acc0, a.z, a.w, b.z, b.w);
        a = smem.mainloop.s_A[cur][smem_idx17(r0 + 8, sl)];
        FN_MMA_884_RC(acc1, a.x, a.y, b.x, b.y);
        FN_MMA_884_RC(acc1, a.z, a.w, b.z, b.w);
        a = smem.mainloop.s_A[cur][smem_idx17(r0 + 16, sl)];
        FN_MMA_884_RC(acc2, a.x, a.y, b.x, b.y);
        FN_MMA_884_RC(acc2, a.z, a.w, b.z, b.w);
        a = smem.mainloop.s_A[cur][smem_idx17(r0 + 24, sl)];
        FN_MMA_884_RC(acc3, a.x, a.y, b.x, b.y);
        FN_MMA_884_RC(acc3, a.z, a.w, b.z, b.w);
      }
    }

    if (st + 1 < kNumStages) {
      smem.mainloop.s_A[nxt][smem_idx17(tx >> 2, tx & 3)] = av;
      const half scalar = nvfp4_fast_hmma_scale(sc, s_wt[64]);
      const half2 scale2 = __halves2half2(scalar, scalar);
      smem.mainloop.s_B[nxt][smem_idx17(tx & 127, (tx >> 7) * 2)] =
          decode_e2m1x8_u4(w0, scale2);
      smem.mainloop.s_B[nxt][smem_idx17(tx & 127, (tx >> 7) * 2 + 1)] =
          decode_e2m1x8_u4(w1, scale2);
    }
    __syncthreads();
  }

  const int mc_ep = s_tok[64];
  const int lane = tx & 31;
  const int warp = tx >> 5;
  const int warp_m = warp >> 2;
  const int warp_n = warp & 3;
  const int quad = (lane >> 2) & 3;

  if constexpr (USE_F16X2) {
    const int m_base = warp_m * 32;
#define FN_W2_STAGE_F16_8(ACC, GLB_R)                                       \
  do {                                                                      \
    if ((GLB_R) < mc_ep) {                                                  \
      const int r_b = (GLB_R) + (((lane & 16) ? 4 : 0) | (lane & 1));       \
      const int c_b = warp_n * 32 + quad * 8 + (((lane >> 1) & 1) << 1);    \
      const float wt0 = s_wt[r_b];                                          \
      const float wt2 = s_wt[r_b + 2];                                      \
      *reinterpret_cast<half2*>(&smem.s_out_f16[r_b][c_b]) =                \
          __floats2half2_rn(__half2float(__float2half_rn((ACC)[0])) * wt0,  \
                            __half2float(__float2half_rn((ACC)[1])) * wt0); \
      *reinterpret_cast<half2*>(&smem.s_out_f16[r_b + 2][c_b]) =            \
          __floats2half2_rn(__half2float(__float2half_rn((ACC)[2])) * wt2,  \
                            __half2float(__float2half_rn((ACC)[3])) * wt2); \
      *reinterpret_cast<half2*>(&smem.s_out_f16[r_b][c_b + 4]) =            \
          __floats2half2_rn(__half2float(__float2half_rn((ACC)[4])) * wt0,  \
                            __half2float(__float2half_rn((ACC)[5])) * wt0); \
      *reinterpret_cast<half2*>(&smem.s_out_f16[r_b + 2][c_b + 4]) =        \
          __floats2half2_rn(__half2float(__float2half_rn((ACC)[6])) * wt2,  \
                            __half2float(__float2half_rn((ACC)[7])) * wt2); \
    }                                                                       \
  } while (0)
    FN_W2_STAGE_F16_8(acc0, m_base + 0);
    FN_W2_STAGE_F16_8(acc1, m_base + 8);
    FN_W2_STAGE_F16_8(acc2, m_base + 16);
    FN_W2_STAGE_F16_8(acc3, m_base + 24);
#undef FN_W2_STAGE_F16_8
    __syncthreads();

    half* slice_base = reinterpret_cast<half*>(accum_buf) +
                       static_cast<size_t>(blockIdx.y) * m_stride * 128;
    const int row_grp = tx >> 6;  // 0..3 -> 4 rows per step!
    const int col2 = tx & 63;     // 0..63 -> 128 halves!
    for (int r = row_grp; r < mc_ep; r += 4) {
      const int tok = s_tok[r];
      if (tok >= 0) {
        atomicAdd(
            reinterpret_cast<half2*>(slice_base +
                                     static_cast<size_t>(tok) * 128) +
                col2,
            *reinterpret_cast<const half2*>(&smem.s_out_f16[r][col2 * 2]));
      }
    }
  } else {
    float* slice_base = reinterpret_cast<float*>(accum_buf) +
                        static_cast<size_t>(blockIdx.y) * m_stride * 128;
    const int row_grp = tx >> 7;
    const int col = tx & 127;

#define FN_W2_STAGE_F32_8(ACC, LOC_R, GLB_R)                             \
  do {                                                                   \
    if ((GLB_R) < mc_ep) {                                               \
      const int r_off0 = ((lane & 16) ? 4 : 0) | (lane & 1);             \
      const int lr0 = (LOC_R) + r_off0;                                  \
      const int gr0 = (GLB_R) + r_off0;                                  \
      const int c_b = warp_n * 32 + quad * 8 + (((lane >> 1) & 1) << 1); \
      const float wt0 = s_wt[gr0];                                       \
      const float wt2 = s_wt[gr0 + 2];                                   \
      *reinterpret_cast<float2*>(&smem.s_out[lr0][c_b]) =                \
          make_float2(__half2float(__float2half_rn((ACC)[0])) * wt0,     \
                      __half2float(__float2half_rn((ACC)[1])) * wt0);    \
      *reinterpret_cast<float2*>(&smem.s_out[lr0 + 2][c_b]) =            \
          make_float2(__half2float(__float2half_rn((ACC)[2])) * wt2,     \
                      __half2float(__float2half_rn((ACC)[3])) * wt2);    \
      *reinterpret_cast<float2*>(&smem.s_out[lr0][c_b + 4]) =            \
          make_float2(__half2float(__float2half_rn((ACC)[4])) * wt0,     \
                      __half2float(__float2half_rn((ACC)[5])) * wt0);    \
      *reinterpret_cast<float2*>(&smem.s_out[lr0 + 2][c_b + 4]) =        \
          make_float2(__half2float(__float2half_rn((ACC)[6])) * wt2,     \
                      __half2float(__float2half_rn((ACC)[7])) * wt2);    \
    }                                                                    \
  } while (0)

    if (warp_m == 0) {
      FN_W2_STAGE_F32_8(acc0, 0, 0);
      FN_W2_STAGE_F32_8(acc1, 8, 8);
      FN_W2_STAGE_F32_8(acc2, 16, 16);
      FN_W2_STAGE_F32_8(acc3, 24, 24);
    } else {
      FN_W2_STAGE_F32_8(acc0, 32, 32);
      FN_W2_STAGE_F32_8(acc1, 40, 40);
    }
    __syncthreads();
    const int pass0_lim = min(48, mc_ep);
    for (int r = row_grp; r < pass0_lim; r += 2) {
      const int tok = s_tok[r];
      if (tok >= 0) {
        atomicAdd(slice_base + static_cast<size_t>(tok) * 128 + col,
                  smem.s_out[r][col]);
      }
    }
    if (mc_ep > 48) {
      __syncthreads();
      if (warp_m == 1) {
        FN_W2_STAGE_F32_8(acc2, 0, 48);
        FN_W2_STAGE_F32_8(acc3, 8, 56);
      }
      __syncthreads();
      const int pass1_lim = mc_ep - 48;
      for (int r = row_grp; r < pass1_lim; r += 2) {
        const int tok = s_tok[48 + r];
        if (tok >= 0) {
          atomicAdd(slice_base + static_cast<size_t>(tok) * 128 + col,
                    smem.s_out[r][col]);
        }
      }
    }
#undef FN_W2_STAGE_F32_8
  }
}

// 3-CTA/SM L2-resident CTA_M=64, CTA_N=64 W2 preload atomicAdd:
// - 128 threads x 3 CTAs/SM = 384 threads/SM (max 168 regs/thread -> 0 spills!)
// - Preloads entire [64, 160] inter into s_A[5][16*17] ONCE with #pragma unroll
//   (100% compile-time constant LDS.128 immediate offsets!)
// - 30,976 B smem/CTA x 3 = 92,928 B <= 96 KiB
// - [16384, 64] FP32 slice = 4.00 MiB <= 6.00 MiB V100 L2 cache (100% L2 hit!)
__global__ __launch_bounds__(128, 3) void fn_nvfp4_w2_m64_n64_preload_kernel(
    const half* __restrict__ inter, const uint32_t* __restrict__ weights,
    const uint8_t* __restrict__ scale_codes,
    const float* __restrict__ global_scales,
    const int32_t* __restrict__ sorted_token_idx,
    const float* __restrict__ sorted_weights,
    const MoETileDesc* __restrict__ tiles,
    const int32_t* __restrict__ num_tiles_ptr,
    float* __restrict__ accum_f32,  // [40, M, 64] -> 4.00 MiB/slice at M=16384!
    int M) {
  const int tile_id = blockIdx.x;
  if (tile_id >= *num_tiles_ptr) return;
  const int n_blk = blockIdx.y;  // 0..39 -> 64 cols each = 2560 cols

  const MoETileDesc desc = tiles[tile_id];
  const int expert = desc.expert;
  const int m_start = desc.start;
  const int m_count = desc.count;
  if (m_count <= 0) return;

  constexpr int kInter = 160;
  constexpr int kHidden = 2560;
  constexpr int kNumStages = 5;

  union SharedW2M64N64 {
    struct {
      uint4 s_A[5][16 * 17];  // 21,760 B
      uint4 s_B[2][16 * 17];  //  8,704 B
    } mainloop;               // 30,464 B + 512 B = 30,976 B -> 3 CTAs/SM!
    float s_out[64][68];      // 17,408 B (single-pass 64-row stage!)
  };
  __shared__ __align__(16) SharedW2M64N64 smem;
  __shared__ int32_t s_tok[64];
  __shared__ float s_wt[64];

  const int tx = threadIdx.x;    // 0..127
  const int lane = tx & 31;      // 0..31
  const int warp = tx >> 5;      // 0..3
  const int warp_m = warp >> 1;  // 0..1 -> 32 rows each = 64 rows
  const int warp_n = warp & 1;   // 0..1 -> 32 cols each = 64 cols

  if (tx < 64) {
    if (tx < m_count) {
      s_tok[tx] = sorted_token_idx[m_start + tx];
      s_wt[tx] = sorted_weights[m_start + tx];
    } else {
      s_tok[tx] = -1;
      s_wt[tx] = 0.0f;
    }
  }

  // Preload entire [64, 160] A tile (1280 uint4s = exact 10 uint4s/thread!)
#pragma unroll
  for (int i = 0; i < 10; ++i) {
    const int vec_id = tx + i * 128;
    const int r = vec_id / 20;
    const int c_vec = vec_id % 20;
    const int st_a = c_vec >> 2;
    const int sl_a = c_vec & 3;
    uint4 val = make_uint4(0, 0, 0, 0);
    if (r < m_count) {
      val = __ldg(reinterpret_cast<const uint4*>(
          inter + static_cast<size_t>(m_start + r) * kInter + c_vec * 8));
    }
    smem.mainloop.s_A[st_a][smem_idx17(r, sl_a)] = val;
  }

  const int sub_b = tx >> 6;       // 0..1
  const int col64_b = tx & 63;     // 0..63
  const int wn_b = col64_b >> 5;   // 0..1
  const int c32_b = col64_b & 31;  // 0..31
  const int t32_b = n_blk * 2 + wn_b;
  const int b_store_idx0 = smem_idx17(col64_b, sub_b * 2);
  const int b_store_idx1 = smem_idx17(col64_b, sub_b * 2 + 1);

  const uint32_t* exp_w =
      weights + static_cast<size_t>(expert) * kInter * (kHidden / 8);
  const uint8_t* exp_sc =
      scale_codes + static_cast<size_t>(expert) * (kInter / 16) * kHidden;
  const float gs_x256 = __ldg(global_scales + expert) * 256.0f;

  {
    const int g = sub_b;
    const size_t w_off = (static_cast<size_t>(t32_b) * 20 + g * 2) * 32 + c32_b;
    const size_t s_idx = (static_cast<size_t>(g) * 80 + t32_b) * 32 + c32_b;
    const uint32_t w0 = __ldg(exp_w + w_off);
    const uint32_t w1 = __ldg(exp_w + w_off + 32);
    const uint8_t sc = __ldg(exp_sc + s_idx);
    const half scalar = nvfp4_fast_hmma_scale(sc, gs_x256);
    const half2 scale2 = __halves2half2(scalar, scalar);
    alignas(16) half2 dec0[4], dec1[4];
    decode_e2m1x8(w0, scale2, dec0);
    decode_e2m1x8(w1, scale2, dec1);
    smem.mainloop.s_B[0][b_store_idx0] = *reinterpret_cast<const uint4*>(dec0);
    smem.mainloop.s_B[0][b_store_idx1] = *reinterpret_cast<const uint4*>(dec1);
  }
  __syncthreads();

  const int mma_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int quad = (lane >> 2) & 3;
  const int b_col64 = warp_n * 32 + quad * 8 + mma_row;
  const int m_base = warp_m * 32;
  const int r0 = m_base + mma_row;
  const int r1 = m_base + 8 + mma_row;
  const int r2 = m_base + 16 + mma_row;
  const int r3 = m_base + 24 + mma_row;

  float acc0[8] = {}, acc1[8] = {}, acc2[8] = {}, acc3[8] = {};

#pragma unroll
  for (int st = 0; st < kNumStages; ++st) {
    const int cur = st & 1;
    const int nxt = cur ^ 1;
    uint32_t w0 = 0, w1 = 0;
    uint8_t sc = 0;
    if (st + 1 < kNumStages) {
      const int g = (st + 1) * 2 + sub_b;
      const size_t w_off =
          (static_cast<size_t>(t32_b) * 20 + g * 2) * 32 + c32_b;
      const size_t s_idx = (static_cast<size_t>(g) * 80 + t32_b) * 32 + c32_b;
      w0 = __ldg(exp_w + w_off);
      w1 = __ldg(exp_w + w_off + 32);
      sc = __ldg(exp_sc + s_idx);
    }

    if (m_base < m_count) {
#pragma unroll
      for (int sub = 0; sub < 2; ++sub) {
        const int sl0 = sub * 2;
        const int sl1 = sub * 2 + 1;
        const uint4 b0 = smem.mainloop.s_B[cur][smem_idx17(b_col64, sl0)];
        const uint4 b1 = smem.mainloop.s_B[cur][smem_idx17(b_col64, sl1)];
        uint4 a_lo = smem.mainloop.s_A[st][smem_idx17(r0, sl0)];
        uint4 a_hi = smem.mainloop.s_A[st][smem_idx17(r0, sl1)];
        uint4 an_lo = smem.mainloop.s_A[st][smem_idx17(r1, sl0)];
        uint4 an_hi = smem.mainloop.s_A[st][smem_idx17(r1, sl1)];
        FN_MMA_884_RC(acc0, a_lo.x, a_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc0, a_lo.z, a_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc0, a_hi.x, a_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc0, a_hi.z, a_hi.w, b1.z, b1.w);
        a_lo = smem.mainloop.s_A[st][smem_idx17(r2, sl0)];
        a_hi = smem.mainloop.s_A[st][smem_idx17(r2, sl1)];
        FN_MMA_884_RC(acc1, an_lo.x, an_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc1, an_lo.z, an_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc1, an_hi.x, an_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc1, an_hi.z, an_hi.w, b1.z, b1.w);
        an_lo = smem.mainloop.s_A[st][smem_idx17(r3, sl0)];
        an_hi = smem.mainloop.s_A[st][smem_idx17(r3, sl1)];
        FN_MMA_884_RC(acc2, a_lo.x, a_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc2, a_lo.z, a_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc2, a_hi.x, a_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc2, a_hi.z, a_hi.w, b1.z, b1.w);
        FN_MMA_884_RC(acc3, an_lo.x, an_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc3, an_lo.z, an_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc3, an_hi.x, an_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc3, an_hi.z, an_hi.w, b1.z, b1.w);
      }
    }

    if (st + 1 < kNumStages) {
      const half scalar = nvfp4_fast_hmma_scale(sc, gs_x256);
      const half2 scale2 = __halves2half2(scalar, scalar);
      alignas(16) half2 dec0[4], dec1[4];
      decode_e2m1x8(w0, scale2, dec0);
      decode_e2m1x8(w1, scale2, dec1);
      smem.mainloop.s_B[nxt][b_store_idx0] =
          *reinterpret_cast<const uint4*>(dec0);
      smem.mainloop.s_B[nxt][b_store_idx1] =
          *reinterpret_cast<const uint4*>(dec1);
    }
    __syncthreads();
  }

  auto stage_all = [&](const float (&acc)[8], int global_r) {
    if (global_r >= m_count) return;
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
      const int r_off = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
      const int r = global_r + r_off;
      const int c =
          warp_n * 32 + quad * 8 + ((((lane >> 1) & 1) << 1) | ((i >> 2) << 2));
      const float w = s_wt[r];
      smem.s_out[r][c] = __half2float(__float2half_rn(acc[i])) * w;
      smem.s_out[r][c + 1] = __half2float(__float2half_rn(acc[i + 1])) * w;
    }
  };
  stage_all(acc0, m_base + 0);
  stage_all(acc1, m_base + 8);
  stage_all(acc2, m_base + 16);
  stage_all(acc3, m_base + 24);
  __syncthreads();

  float* slice_base = accum_f32 + static_cast<size_t>(n_blk) * M * 64;
  const int row_grp = tx >> 6;  // 0..1
  const int col = tx & 63;      // 0..63
  for (int r = row_grp; r < m_count; r += 2) {
    const int tok = s_tok[r];
    if (tok >= 0) {
      atomicAdd(slice_base + static_cast<size_t>(tok) * 64 + col,
                smem.s_out[r][col]);
    }
  }
}

__global__ __launch_bounds__(320, 2) void fn_nvfp4_w2_m64_n160_atomic_kernel(
    const half* __restrict__ inter, const uint32_t* __restrict__ weights,
    const uint8_t* __restrict__ scale_codes,
    const float* __restrict__ global_scales,
    const int32_t* __restrict__ sorted_token_idx,
    const float* __restrict__ sorted_weights,
    const MoETileDesc* __restrict__ tiles,
    const int32_t* __restrict__ num_tiles_ptr,
    float* __restrict__ accum_f32,  // [16, M, 160]
    int M) {
  const int tile_idx = blockIdx.x;
  if (tile_idx >= *num_tiles_ptr) return;
  const int n_blk = blockIdx.y;  // 0..15 -> 160 cols each = 2560 cols

  const MoETileDesc desc = tiles[tile_idx];
  const int expert = desc.expert;
  const int m_start = desc.start;
  const int m_count = desc.count;
  if (m_count <= 0) return;

  constexpr int kInter = 160;
  constexpr int kHidden = 2560;
  constexpr int kNumStages = 5;

  union SharedW2M64N160 {
    struct {
      uint4 s_A[5][16 * 17];  // 21,760 B
      uint4 s_B[2][40 * 17];  // 21,760 B
    } mainloop;            // 43,520 B + 512 B = 44,032 B <= 48 KiB (2 CTAs/SM!)
    float s_out[32][164];  // 20,992 B
  };
  __shared__ __align__(16) SharedW2M64N160 smem;
  __shared__ int32_t s_tok[64];
  __shared__ float s_wt[64];

  const int tx = threadIdx.x;   // 0..319
  const int lane = tx & 31;     // 0..31
  const int warp = tx >> 5;     // 0..9
  const int warp_m = warp / 5;  // 0..1 -> 32 rows each = 64 rows
  const int warp_n = warp % 5;  // 0..4 -> 32 cols each = 160 cols

  if (tx < 64) {
    if (tx < m_count) {
      s_tok[tx] = sorted_token_idx[m_start + tx];
      s_wt[tx] = sorted_weights[m_start + tx];
    } else {
      s_tok[tx] = -1;
      s_wt[tx] = 0.0f;
    }
  }

  // Issue all 4 inter A-tile loads AND Stage-0 B-weight loads in parallel
  // before STS!
  uint4 av_pre[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int vec_id = tx + i * 320;
    const int r = vec_id / 20;
    const int c_vec = vec_id % 20;
    av_pre[i] = (r < m_count)
                    ? __ldg(reinterpret_cast<const uint4*>(
                          inter + static_cast<size_t>(m_start + r) * kInter +
                          c_vec * 8))
                    : make_uint4(0, 0, 0, 0);
  }

  const int sub_b = tx / 160;       // 0..1
  const int col160_b = tx % 160;    // 0..159
  const int wn_b = col160_b >> 5;   // 0..4
  const int c32_b = col160_b & 31;  // 0..31
  const int t32_b = n_blk * 5 + wn_b;
  const int b_store_idx0 = smem_idx17(col160_b, sub_b * 2);
  const int b_store_idx1 = smem_idx17(col160_b, sub_b * 2 + 1);

  const uint32_t* exp_w =
      weights + static_cast<size_t>(expert) * kInter * (kHidden / 8);
  const uint8_t* exp_sc =
      scale_codes + static_cast<size_t>(expert) * (kInter / 16) * kHidden;
  const float gs_x256 = __ldg(global_scales + expert) * 256.0f;

  {
    const int g = sub_b;
    const size_t w_off = (static_cast<size_t>(t32_b) * 20 + g * 2) * 32 + c32_b;
    const size_t s_idx = (static_cast<size_t>(g) * 80 + t32_b) * 32 + c32_b;
    const uint32_t w0 = __ldg(exp_w + w_off);
    const uint32_t w1 = __ldg(exp_w + w_off + 32);
    const uint8_t sc = __ldg(exp_sc + s_idx);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int vec_id = tx + i * 320;
      const int r = vec_id / 20;
      const int c_vec = vec_id % 20;
      smem.mainloop.s_A[c_vec >> 2][smem_idx17(r, c_vec & 3)] = av_pre[i];
    }
    const half scalar = nvfp4_fast_hmma_scale(sc, gs_x256);
    const half2 scale2 = __halves2half2(scalar, scalar);
    alignas(16) half2 dec0[4], dec1[4];
    decode_e2m1x8(w0, scale2, dec0);
    decode_e2m1x8(w1, scale2, dec1);
    smem.mainloop.s_B[0][b_store_idx0] = *reinterpret_cast<const uint4*>(dec0);
    smem.mainloop.s_B[0][b_store_idx1] = *reinterpret_cast<const uint4*>(dec1);
  }
  __syncthreads();

  const int mma_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int quad = (lane >> 2) & 3;
  const int b_col160 = warp_n * 32 + quad * 8 + mma_row;
  const int m_base = warp_m * 32;
  const int r0 = m_base + mma_row;
  const int r1 = m_base + 8 + mma_row;
  const int r2 = m_base + 16 + mma_row;
  const int r3 = m_base + 24 + mma_row;

  float acc0[8] = {}, acc1[8] = {}, acc2[8] = {}, acc3[8] = {};

#pragma unroll
  for (int st = 0; st < kNumStages; ++st) {
    const int cur = st & 1;
    const int nxt = cur ^ 1;
    uint32_t w0 = 0, w1 = 0;
    uint8_t sc = 0;
    if (st + 1 < kNumStages) {
      const int g = (st + 1) * 2 + sub_b;
      const size_t w_off =
          (static_cast<size_t>(t32_b) * 20 + g * 2) * 32 + c32_b;
      const size_t s_idx = (static_cast<size_t>(g) * 80 + t32_b) * 32 + c32_b;
      w0 = __ldg(exp_w + w_off);
      w1 = __ldg(exp_w + w_off + 32);
      sc = __ldg(exp_sc + s_idx);
    }

    if (m_base < m_count) {
#pragma unroll
      for (int sub = 0; sub < 2; ++sub) {
        const int sl0 = sub * 2;
        const int sl1 = sub * 2 + 1;
        const uint4 b0 = smem.mainloop.s_B[cur][smem_idx17(b_col160, sl0)];
        const uint4 b1 = smem.mainloop.s_B[cur][smem_idx17(b_col160, sl1)];
        uint4 a_lo = smem.mainloop.s_A[st][smem_idx17(r0, sl0)];
        uint4 a_hi = smem.mainloop.s_A[st][smem_idx17(r0, sl1)];
        uint4 an_lo = smem.mainloop.s_A[st][smem_idx17(r1, sl0)];
        uint4 an_hi = smem.mainloop.s_A[st][smem_idx17(r1, sl1)];
        FN_MMA_884_RC(acc0, a_lo.x, a_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc0, a_lo.z, a_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc0, a_hi.x, a_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc0, a_hi.z, a_hi.w, b1.z, b1.w);
        a_lo = smem.mainloop.s_A[st][smem_idx17(r2, sl0)];
        a_hi = smem.mainloop.s_A[st][smem_idx17(r2, sl1)];
        FN_MMA_884_RC(acc1, an_lo.x, an_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc1, an_lo.z, an_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc1, an_hi.x, an_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc1, an_hi.z, an_hi.w, b1.z, b1.w);
        an_lo = smem.mainloop.s_A[st][smem_idx17(r3, sl0)];
        an_hi = smem.mainloop.s_A[st][smem_idx17(r3, sl1)];
        FN_MMA_884_RC(acc2, a_lo.x, a_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc2, a_lo.z, a_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc2, a_hi.x, a_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc2, a_hi.z, a_hi.w, b1.z, b1.w);
        FN_MMA_884_RC(acc3, an_lo.x, an_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc3, an_lo.z, an_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc3, an_hi.x, an_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc3, an_hi.z, an_hi.w, b1.z, b1.w);
      }
    }

    if (st + 1 < kNumStages) {
      const half scalar = nvfp4_fast_hmma_scale(sc, gs_x256);
      const half2 scale2 = __halves2half2(scalar, scalar);
      alignas(16) half2 dec0[4], dec1[4];
      decode_e2m1x8(w0, scale2, dec0);
      decode_e2m1x8(w1, scale2, dec1);
      smem.mainloop.s_B[nxt][b_store_idx0] =
          *reinterpret_cast<const uint4*>(dec0);
      smem.mainloop.s_B[nxt][b_store_idx1] =
          *reinterpret_cast<const uint4*>(dec1);
    }
    __syncthreads();
  }

  float* slice_base = accum_f32 + static_cast<size_t>(n_blk) * M * 160;
  const int row_grp = tx / 160;  // 0..1
  const int col = tx % 160;      // 0..159

  auto stage_sub = [&](const float (&acc)[8], int local_r_base, int global_r) {
    if (global_r >= m_count) return;
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
      const int r_off = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
      const int local_r = local_r_base + r_off;
      const int c =
          warp_n * 32 + quad * 8 + ((((lane >> 1) & 1) << 1) | ((i >> 2) << 2));
      const float w = s_wt[global_r + r_off];
      *reinterpret_cast<float2*>(&smem.s_out[local_r][c]) =
          make_float2(__half2float(__float2half_rn(acc[i])) * w,
                      __half2float(__float2half_rn(acc[i + 1])) * w);
    }
  };

  if (warp_m == 0) {
    stage_sub(acc0, 0, 0);
    stage_sub(acc1, 8, 8);
    stage_sub(acc2, 16, 16);
    stage_sub(acc3, 24, 24);
  }
  __syncthreads();
  const int pass0_lim = min(32, m_count);
  for (int r = row_grp; r < pass0_lim; r += 2) {
    const int tok = s_tok[r];
    if (tok >= 0) {
      atomicAdd(slice_base + static_cast<size_t>(tok) * 160 + col,
                smem.s_out[r][col]);
    }
  }

  if (m_count > 32) {
    __syncthreads();
    if (warp_m == 1) {
      stage_sub(acc0, 0, 32);
      stage_sub(acc1, 8, 40);
      stage_sub(acc2, 16, 48);
      stage_sub(acc3, 24, 56);
    }
    __syncthreads();
    const int pass1_lim = m_count - 32;
    for (int r = row_grp; r < pass1_lim; r += 2) {
      const int tok = s_tok[32 + r];
      if (tok >= 0) {
        atomicAdd(slice_base + static_cast<size_t>(tok) * 160 + col,
                  smem.s_out[r][col]);
      }
    }
  }
}

__global__
__launch_bounds__(256, 2) void fn_nvfp4_w2_m64_n128_f16x2_atomic_kernel(
    const half* __restrict__ inter, const uint32_t* __restrict__ weights,
    const uint8_t* __restrict__ scale_codes,
    const float* __restrict__ global_scales,
    const int32_t* __restrict__ sorted_token_idx,
    const float* __restrict__ sorted_weights,
    const MoETileDesc* __restrict__ tiles,
    const int32_t* __restrict__ num_tiles_ptr,
    half* __restrict__ accum_f16,  // [20, M, 128] -> 4.0 MiB at M=16384 (100%
                                   // L2!)
    int M) {
  const int tile_idx = blockIdx.x;
  if (tile_idx >= *num_tiles_ptr) return;
  const int n_blk = blockIdx.y;  // 0..19 -> 128 cols each

  const MoETileDesc desc = tiles[tile_idx];
  const int expert = desc.expert;
  const int m_start = desc.start;
  const int m_count = desc.count;
  if (m_count <= 0) return;

  constexpr int kInter = 160;
  constexpr int kHidden = 2560;
  constexpr int kNumStages = 5;

  union SharedW2M64N128F16 {
    struct {
      uint4 s_A[5][16 * 17];
      uint4 s_B[2][32 * 17];
    } mainloop;
    half s_out[64][136];
  };
  __shared__ __align__(16) SharedW2M64N128F16 smem;
  __shared__ int32_t s_tok[64];
  __shared__ float s_wt[64];

  const int tx = threadIdx.x;
  const int lane = tx & 31;
  const int warp = tx >> 5;
  const int warp_m = warp >> 2;
  const int warp_n = warp & 3;

  if (tx < 64) {
    if (tx < m_count) {
      s_tok[tx] = sorted_token_idx[m_start + tx];
      s_wt[tx] = sorted_weights[m_start + tx];
    } else {
      s_tok[tx] = -1;
      s_wt[tx] = 0.0f;
    }
  }

#pragma unroll
  for (int i = 0; i < 5; ++i) {
    const int vec_id = tx + i * 256;
    const int r = vec_id / 20;
    const int c_vec = vec_id % 20;
    const int st_a = c_vec >> 2;
    const int sl_a = c_vec & 3;
    uint4 val = make_uint4(0, 0, 0, 0);
    if (r < m_count) {
      val = __ldg(reinterpret_cast<const uint4*>(
          inter + static_cast<size_t>(m_start + r) * kInter + c_vec * 8));
    }
    smem.mainloop.s_A[st_a][smem_idx17(r, sl_a)] = val;
  }

  const int sub_b = tx >> 7;
  const int col128_b = tx & 127;
  const int wn_b = col128_b >> 5;
  const int c32_b = col128_b & 31;
  const int t32_b = n_blk * 4 + wn_b;
  const int b_store_idx0 = smem_idx17(col128_b, sub_b * 2);
  const int b_store_idx1 = smem_idx17(col128_b, sub_b * 2 + 1);

  const uint32_t* exp_w =
      weights + static_cast<size_t>(expert) * kInter * (kHidden / 8);
  const uint8_t* exp_sc =
      scale_codes + static_cast<size_t>(expert) * (kInter / 16) * kHidden;
  const float gs_x256 = __ldg(global_scales + expert) * 256.0f;

  {
    const int g = sub_b;
    const size_t w_off = (static_cast<size_t>(t32_b) * 20 + g * 2) * 32 + c32_b;
    const size_t s_idx = (static_cast<size_t>(g) * 80 + t32_b) * 32 + c32_b;
    const uint32_t w0 = __ldg(exp_w + w_off);
    const uint32_t w1 = __ldg(exp_w + w_off + 32);
    const uint8_t sc = __ldg(exp_sc + s_idx);
    const half scalar = nvfp4_fast_hmma_scale(sc, gs_x256);
    const half2 scale2 = __halves2half2(scalar, scalar);
    alignas(16) half2 dec0[4], dec1[4];
    decode_e2m1x8(w0, scale2, dec0);
    decode_e2m1x8(w1, scale2, dec1);
    smem.mainloop.s_B[0][b_store_idx0] = *reinterpret_cast<const uint4*>(dec0);
    smem.mainloop.s_B[0][b_store_idx1] = *reinterpret_cast<const uint4*>(dec1);
  }
  __syncthreads();

  const int mma_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int quad = (lane >> 2) & 3;
  const int b_col128 = warp_n * 32 + quad * 8 + mma_row;
  const int m_base = warp_m * 32;
  const int r0 = m_base + mma_row;
  const int r1 = m_base + 8 + mma_row;
  const int r2 = m_base + 16 + mma_row;
  const int r3 = m_base + 24 + mma_row;

  float acc0[8] = {};
  float acc1[8] = {};
  float acc2[8] = {};
  float acc3[8] = {};

#pragma unroll
  for (int st = 0; st < kNumStages; ++st) {
    const int cur = st & 1;
    const int nxt = cur ^ 1;
    uint32_t w0 = 0, w1 = 0;
    uint8_t sc = 0;
    if (st + 1 < kNumStages) {
      const int g = (st + 1) * 2 + sub_b;
      const size_t w_off =
          (static_cast<size_t>(t32_b) * 20 + g * 2) * 32 + c32_b;
      const size_t s_idx = (static_cast<size_t>(g) * 80 + t32_b) * 32 + c32_b;
      w0 = __ldg(exp_w + w_off);
      w1 = __ldg(exp_w + w_off + 32);
      sc = __ldg(exp_sc + s_idx);
    }

    if (m_base < m_count) {
#pragma unroll
      for (int sub = 0; sub < 2; ++sub) {
        const int sl0 = sub * 2;
        const int sl1 = sub * 2 + 1;
        const uint4 b0 = smem.mainloop.s_B[cur][smem_idx17(b_col128, sl0)];
        const uint4 b1 = smem.mainloop.s_B[cur][smem_idx17(b_col128, sl1)];
        uint4 a_lo = smem.mainloop.s_A[st][smem_idx17(r0, sl0)];
        uint4 a_hi = smem.mainloop.s_A[st][smem_idx17(r0, sl1)];
        uint4 an_lo = smem.mainloop.s_A[st][smem_idx17(r1, sl0)];
        uint4 an_hi = smem.mainloop.s_A[st][smem_idx17(r1, sl1)];
        FN_MMA_884_RC(acc0, a_lo.x, a_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc0, a_lo.z, a_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc0, a_hi.x, a_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc0, a_hi.z, a_hi.w, b1.z, b1.w);
        a_lo = smem.mainloop.s_A[st][smem_idx17(r2, sl0)];
        a_hi = smem.mainloop.s_A[st][smem_idx17(r2, sl1)];
        FN_MMA_884_RC(acc1, an_lo.x, an_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc1, an_lo.z, an_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc1, an_hi.x, an_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc1, an_hi.z, an_hi.w, b1.z, b1.w);
        an_lo = smem.mainloop.s_A[st][smem_idx17(r3, sl0)];
        an_hi = smem.mainloop.s_A[st][smem_idx17(r3, sl1)];
        FN_MMA_884_RC(acc2, a_lo.x, a_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc2, a_lo.z, a_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc2, a_hi.x, a_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc2, a_hi.z, a_hi.w, b1.z, b1.w);
        FN_MMA_884_RC(acc3, an_lo.x, an_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc3, an_lo.z, an_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc3, an_hi.x, an_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc3, an_hi.z, an_hi.w, b1.z, b1.w);
      }
    }

    if (st + 1 < kNumStages) {
      const half scalar = nvfp4_fast_hmma_scale(sc, gs_x256);
      const half2 scale2 = __halves2half2(scalar, scalar);
      alignas(16) half2 dec0[4], dec1[4];
      decode_e2m1x8(w0, scale2, dec0);
      decode_e2m1x8(w1, scale2, dec1);
      smem.mainloop.s_B[nxt][b_store_idx0] =
          *reinterpret_cast<const uint4*>(dec0);
      smem.mainloop.s_B[nxt][b_store_idx1] =
          *reinterpret_cast<const uint4*>(dec1);
    }
    __syncthreads();
  }

  auto stage_f16 = [&](const float (&acc)[8], int global_r) {
    if (global_r >= m_count) return;
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
      const int r_off = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
      const int r = global_r + r_off;
      const int c =
          warp_n * 32 + quad * 8 + ((((lane >> 1) & 1) << 1) | ((i >> 2) << 2));
      const float w = s_wt[r];
      const float v0 = __half2float(__float2half_rn(acc[i])) * w;
      const float v1 = __half2float(__float2half_rn(acc[i + 1])) * w;
      *reinterpret_cast<half2*>(&smem.s_out[r][c]) = __floats2half2_rn(v0, v1);
    }
  };
  stage_f16(acc0, m_base + 0);
  stage_f16(acc1, m_base + 8);
  stage_f16(acc2, m_base + 16);
  stage_f16(acc3, m_base + 24);
  __syncthreads();

  half* slice_base = accum_f16 + static_cast<size_t>(n_blk) * M * 128;
  const int row_grp = tx >> 6;  // 0..3
  const int col2 = tx & 63;     // 0..63 -> 128 halves!
  for (int r = row_grp; r < m_count; r += 4) {
    const int tok = s_tok[r];
    if (tok >= 0) {
      atomicAdd(reinterpret_cast<half2*>(slice_base +
                                         static_cast<size_t>(tok) * 128) +
                    col2,
                *reinterpret_cast<const half2*>(&smem.s_out[r][col2 * 2]));
    }
  }
}

struct SharedW2M64N256 {
  union {
    struct {
      uint4 s_A[5][16 * 17];
      uint4 s_B[2][64 * 17];
    } mainloop;
    float s_out[32][260];
    half s_out_f16[64][264];
  };
  int32_t s_tok[64];
  float s_wt[64];
};

template <bool USE_F16X2>
__global__ __launch_bounds__(512, 1) void fn_nvfp4_w2_m64_n256_atomic_kernel(
    const half* __restrict__ inter, const uint32_t* __restrict__ weights,
    const uint8_t* __restrict__ scale_codes,
    const float* __restrict__ global_scales,
    const int32_t* __restrict__ sorted_token_idx,
    const float* __restrict__ sorted_weights,
    const MoETileDesc* __restrict__ tiles,
    const int32_t* __restrict__ num_tiles_ptr,
    void* __restrict__ accum_buf,  // [10, M, 256] float or half
    int M) {
  const int tile_idx = blockIdx.x;
  if (tile_idx >= *num_tiles_ptr) return;
  const int n_blk = blockIdx.y;  // 0..9 -> 256 cols each

  const MoETileDesc desc = tiles[tile_idx];
  const int expert = desc.expert;
  const int m_start = desc.start;
  const int m_count = desc.count;
  if (m_count <= 0) return;

  constexpr int kInter = 160;
  constexpr int kHidden = 2560;
  constexpr int kNumStages = 5;

  extern __shared__ __align__(16) unsigned char w2_n256_raw[];
  SharedW2M64N256& smem = *reinterpret_cast<SharedW2M64N256*>(w2_n256_raw);

  const int tx = threadIdx.x;    // 0..511
  const int lane = tx & 31;      // 0..31
  const int warp = tx >> 5;      // 0..15
  const int warp_m = warp >> 3;  // 0..1 -> 32 rows each = 64 rows
  const int warp_n = warp & 7;   // 0..7 -> 32 cols each = 256 cols

  if (tx < 64) {
    if (tx < m_count) {
      smem.s_tok[tx] = sorted_token_idx[m_start + tx];
      smem.s_wt[tx] = sorted_weights[m_start + tx];
    } else {
      smem.s_tok[tx] = -1;
      smem.s_wt[tx] = 0.0f;
    }
  }

  uint4 av_pre[3];
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    const int vec_id = tx + i * 512;
    av_pre[i] = make_uint4(0, 0, 0, 0);
    if (vec_id < 1280) {
      const int r = vec_id / 20;
      const int c_vec = vec_id % 20;
      if (r < m_count) {
        av_pre[i] = __ldg(reinterpret_cast<const uint4*>(
            inter + static_cast<size_t>(m_start + r) * kInter + c_vec * 8));
      }
    }
  }

  const int sub_b = tx >> 8;        // 0..1
  const int col256_b = tx & 255;    // 0..255
  const int wn_b = col256_b >> 5;   // 0..7
  const int c32_b = col256_b & 31;  // 0..31
  const int t32_b = n_blk * 8 + wn_b;
  const int b_store_idx0 = smem_idx17(col256_b, sub_b * 2);
  const int b_store_idx1 = smem_idx17(col256_b, sub_b * 2 + 1);

  const uint32_t* exp_w =
      weights + static_cast<size_t>(expert) * kInter * (kHidden / 8);
  const uint8_t* exp_sc =
      scale_codes + static_cast<size_t>(expert) * (kInter / 16) * kHidden;
  const float gs_x256 = __ldg(global_scales + expert) * 256.0f;

  {
    const int g = sub_b;
    const size_t w_off = (static_cast<size_t>(t32_b) * 20 + g * 2) * 32 + c32_b;
    const size_t s_idx = (static_cast<size_t>(g) * 80 + t32_b) * 32 + c32_b;
    const uint32_t w0 = __ldg(exp_w + w_off);
    const uint32_t w1 = __ldg(exp_w + w_off + 32);
    const uint8_t sc = __ldg(exp_sc + s_idx);
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      const int vec_id = tx + i * 512;
      if (vec_id < 1280) {
        const int r = vec_id / 20;
        const int c_vec = vec_id % 20;
        smem.mainloop.s_A[c_vec >> 2][smem_idx17(r, c_vec & 3)] = av_pre[i];
      }
    }
    const half scalar = nvfp4_fast_hmma_scale(sc, gs_x256);
    const half2 scale2 = __halves2half2(scalar, scalar);
    alignas(16) half2 dec0[4], dec1[4];
    decode_e2m1x8(w0, scale2, dec0);
    decode_e2m1x8(w1, scale2, dec1);
    smem.mainloop.s_B[0][b_store_idx0] = *reinterpret_cast<const uint4*>(dec0);
    smem.mainloop.s_B[0][b_store_idx1] = *reinterpret_cast<const uint4*>(dec1);
  }
  __syncthreads();

  const int mma_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int quad = (lane >> 2) & 3;
  const int b_col256 = warp_n * 32 + quad * 8 + mma_row;
  const int m_base = warp_m * 32;
  const int r0 = m_base + mma_row;
  const int r1 = m_base + 8 + mma_row;
  const int r2 = m_base + 16 + mma_row;
  const int r3 = m_base + 24 + mma_row;

  float acc0[8] = {};
  float acc1[8] = {};
  float acc2[8] = {};
  float acc3[8] = {};

#pragma unroll
  for (int st = 0; st < kNumStages; ++st) {
    const int cur = st & 1;
    const int nxt = cur ^ 1;
    uint32_t w0 = 0, w1 = 0;
    uint8_t sc = 0;
    if (st + 1 < kNumStages) {
      const int g = (st + 1) * 2 + sub_b;
      const size_t w_off =
          (static_cast<size_t>(t32_b) * 20 + g * 2) * 32 + c32_b;
      const size_t s_idx = (static_cast<size_t>(g) * 80 + t32_b) * 32 + c32_b;
      w0 = __ldg(exp_w + w_off);
      w1 = __ldg(exp_w + w_off + 32);
      sc = __ldg(exp_sc + s_idx);
    }

    if (m_base < m_count) {
#pragma unroll
      for (int sub = 0; sub < 2; ++sub) {
        const int sl0 = sub * 2;
        const int sl1 = sub * 2 + 1;
        const uint4 b0 = smem.mainloop.s_B[cur][smem_idx17(b_col256, sl0)];
        const uint4 b1 = smem.mainloop.s_B[cur][smem_idx17(b_col256, sl1)];
        uint4 a_lo = smem.mainloop.s_A[st][smem_idx17(r0, sl0)];
        uint4 a_hi = smem.mainloop.s_A[st][smem_idx17(r0, sl1)];
        uint4 an_lo = smem.mainloop.s_A[st][smem_idx17(r1, sl0)];
        uint4 an_hi = smem.mainloop.s_A[st][smem_idx17(r1, sl1)];
        FN_MMA_884_RC(acc0, a_lo.x, a_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc0, a_lo.z, a_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc0, a_hi.x, a_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc0, a_hi.z, a_hi.w, b1.z, b1.w);
        a_lo = smem.mainloop.s_A[st][smem_idx17(r2, sl0)];
        a_hi = smem.mainloop.s_A[st][smem_idx17(r2, sl1)];
        FN_MMA_884_RC(acc1, an_lo.x, an_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc1, an_lo.z, an_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc1, an_hi.x, an_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc1, an_hi.z, an_hi.w, b1.z, b1.w);
        an_lo = smem.mainloop.s_A[st][smem_idx17(r3, sl0)];
        an_hi = smem.mainloop.s_A[st][smem_idx17(r3, sl1)];
        FN_MMA_884_RC(acc2, a_lo.x, a_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc2, a_lo.z, a_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc2, a_hi.x, a_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc2, a_hi.z, a_hi.w, b1.z, b1.w);
        FN_MMA_884_RC(acc3, an_lo.x, an_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc3, an_lo.z, an_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc3, an_hi.x, an_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc3, an_hi.z, an_hi.w, b1.z, b1.w);
      }
    }

    if (st + 1 < kNumStages) {
      const half scalar = nvfp4_fast_hmma_scale(sc, gs_x256);
      const half2 scale2 = __halves2half2(scalar, scalar);
      alignas(16) half2 dec0[4], dec1[4];
      decode_e2m1x8(w0, scale2, dec0);
      decode_e2m1x8(w1, scale2, dec1);
      smem.mainloop.s_B[nxt][b_store_idx0] =
          *reinterpret_cast<const uint4*>(dec0);
      smem.mainloop.s_B[nxt][b_store_idx1] =
          *reinterpret_cast<const uint4*>(dec1);
    }
    __syncthreads();
  }

  if constexpr (USE_F16X2) {
    auto stage_f16 = [&](const float (&acc)[8], int global_r) {
      if (global_r >= m_count) return;
#pragma unroll
      for (int i = 0; i < 8; i += 2) {
        const int r_off = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
        const int r = global_r + r_off;
        const int c = warp_n * 32 + quad * 8 +
                      ((((lane >> 1) & 1) << 1) | ((i >> 2) << 2));
        const float w = smem.s_wt[r];
        const float v0 = __half2float(__float2half_rn(acc[i])) * w;
        const float v1 = __half2float(__float2half_rn(acc[i + 1])) * w;
        *reinterpret_cast<half2*>(&smem.s_out_f16[r][c]) =
            __floats2half2_rn(v0, v1);
      }
    };
    stage_f16(acc0, m_base + 0);
    stage_f16(acc1, m_base + 8);
    stage_f16(acc2, m_base + 16);
    stage_f16(acc3, m_base + 24);
    __syncthreads();

    half* slice_base = reinterpret_cast<half*>(accum_buf) +
                       static_cast<size_t>(n_blk) * M * 256;
    const int row_grp = tx >> 7;  // 0..3
    const int col2 = tx & 127;    // 0..127 -> 256 halves!
    for (int r = row_grp; r < m_count; r += 4) {
      const int tok = smem.s_tok[r];
      if (tok >= 0) {
        atomicAdd(
            reinterpret_cast<half2*>(slice_base +
                                     static_cast<size_t>(tok) * 256) +
                col2,
            *reinterpret_cast<const half2*>(&smem.s_out_f16[r][col2 * 2]));
      }
    }
  } else {
    float* slice_base = reinterpret_cast<float*>(accum_buf) +
                        static_cast<size_t>(n_blk) * M * 256;
    const int row_grp = tx >> 8;  // 0..1
    const int col = tx & 255;     // 0..255
    auto stage_sub = [&](const float (&acc)[8], int local_r_base,
                         int global_r) {
      if (global_r >= m_count) return;
#pragma unroll
      for (int i = 0; i < 8; i += 2) {
        const int r_off = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
        const int local_r = local_r_base + r_off;
        const int c = warp_n * 32 + quad * 8 +
                      ((((lane >> 1) & 1) << 1) | ((i >> 2) << 2));
        const float w = smem.s_wt[global_r + r_off];
        smem.s_out[local_r][c] = __half2float(__float2half_rn(acc[i])) * w;
        smem.s_out[local_r][c + 1] =
            __half2float(__float2half_rn(acc[i + 1])) * w;
      }
    };
    if (warp_m == 0) {
      stage_sub(acc0, 0, 0);
      stage_sub(acc1, 8, 8);
      stage_sub(acc2, 16, 16);
      stage_sub(acc3, 24, 24);
    }
    __syncthreads();
    const int pass0_lim = min(32, m_count);
    for (int r = row_grp; r < pass0_lim; r += 2) {
      const int tok = smem.s_tok[r];
      if (tok >= 0) {
        atomicAdd(slice_base + static_cast<size_t>(tok) * 256 + col,
                  smem.s_out[r][col]);
      }
    }
    if (m_count > 32) {
      __syncthreads();
      if (warp_m == 1) {
        stage_sub(acc0, 0, 32);
        stage_sub(acc1, 8, 40);
        stage_sub(acc2, 16, 48);
        stage_sub(acc3, 24, 56);
      }
      __syncthreads();
      const int pass1_lim = m_count - 32;
      for (int r = row_grp; r < pass1_lim; r += 2) {
        const int tok = smem.s_tok[32 + r];
        if (tok >= 0) {
          atomicAdd(slice_base + static_cast<size_t>(tok) * 256 + col,
                    smem.s_out[r][col]);
        }
      }
    }
  }
}

// CTA_M=128, CTA_N=128 W2 atomicAdd (512 threads = 16 warps/SM, 1 uint32_t
// B-decode/thread!)
__global__ __launch_bounds__(512, 1) void fn_nvfp4_w2_m128_n128_atomic_kernel(
    const half* __restrict__ inter,           // [M * 10, 160]
    const uint32_t* __restrict__ weights,     // [512, 160, 320]
    const uint8_t* __restrict__ scale_codes,  // [512, 10, 2560]
    const float* __restrict__ global_scales,  // [512, 1]
    const int32_t* __restrict__ sorted_token_idx,
    const float* __restrict__ sorted_weights,
    const MoETileDesc* __restrict__ tiles,
    const int32_t* __restrict__ num_tiles_ptr,
    float* __restrict__ accum_f32,  // [20, M, 128]
    int M) {
  const int tile_id = blockIdx.x;
  if (tile_id >= *num_tiles_ptr) return;

  const MoETileDesc desc = tiles[tile_id];
  const int expert = desc.expert;
  const int m_start = desc.start;
  const int m_count = desc.count;
  const int n_blk = blockIdx.y;  // 0..19 -> 128 cols each

  constexpr int kInter = 160;
  constexpr int kHidden = 2560;
  constexpr int kNumStages = 5;

  union SharedW2M128N128 {
    struct {
      uint4 s_A[2][32 * 17];
      uint4 s_B[2][32 * 17];
    } mainloop;
    float s_out[32][132];
  };
  __shared__ __align__(16) SharedW2M128N128 smem;
  __shared__ int32_t s_tok[128];
  __shared__ float s_wt[128];

  const int tx = threadIdx.x;    // 0..511
  const int lane = tx & 31;      // 0..31
  const int warp = tx >> 5;      // 0..15
  const int warp_m = warp >> 2;  // 0..3 -> 32 rows each = 128 rows
  const int warp_n = warp & 3;   // 0..3 -> 32 cols each = 128 cols

  if (tx < 128) {
    if (tx < m_count) {
      s_tok[tx] = sorted_token_idx[m_start + tx];
      s_wt[tx] = sorted_weights[m_start + tx];
    } else {
      s_tok[tx] = -1;
      s_wt[tx] = 0.0f;
    }
  }

  const int a_r0 = tx >> 2;  // 0..127
  const int a_s0 = tx & 3;   // 0..3
  const int a_smem_idx = smem_idx17(a_r0, a_s0);
  const half* a_ptr0 = nullptr;
  if (a_r0 < m_count) {
    a_ptr0 = inter + static_cast<size_t>(m_start + a_r0) * kInter + a_s0 * 8;
  }

  const int sub_b = tx >> 7;      // 0..3
  const int col128_b = tx & 127;  // 0..127
  const int wn_b = col128_b >> 5;
  const int c32_b = col128_b & 31;
  const int t32_b = n_blk * 4 + wn_b;
  const int b_store_idx = smem_idx17(col128_b, sub_b);

  const uint32_t* exp_w =
      weights + static_cast<size_t>(expert) * kInter * (kHidden / 8);
  const uint8_t* exp_sc =
      scale_codes + static_cast<size_t>(expert) * (kInter / 16) * kHidden;
  const float gs_x256 = __ldg(global_scales + expert) * 256.0f;

  {
    const uint4 av0 = a_ptr0 ? __ldg(reinterpret_cast<const uint4*>(a_ptr0))
                             : make_uint4(0, 0, 0, 0);
    const int g = sub_b >> 1;
    const size_t w_off = (static_cast<size_t>(t32_b) * 20 + sub_b) * 32 + c32_b;
    const size_t s_idx = (static_cast<size_t>(g) * 80 + t32_b) * 32 + c32_b;
    const uint32_t w0 = __ldg(exp_w + w_off);
    const uint8_t sc = __ldg(exp_sc + s_idx);
    smem.mainloop.s_A[0][a_smem_idx] = av0;
    const half scalar = nvfp4_fast_hmma_scale(sc, gs_x256);
    alignas(16) half2 dec0[4];
    decode_e2m1x8(w0, __halves2half2(scalar, scalar), dec0);
    smem.mainloop.s_B[0][b_store_idx] = *reinterpret_cast<const uint4*>(dec0);
  }
  __syncthreads();

  const int mma_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int quad = (lane >> 2) & 3;
  const int b_col128 = warp_n * 32 + quad * 8 + mma_row;
  const int m_base = warp_m * 32;
  const int r0 = m_base + mma_row;
  const int r1 = m_base + 8 + mma_row;
  const int r2 = m_base + 16 + mma_row;
  const int r3 = m_base + 24 + mma_row;

  float acc0[8] = {};
  float acc1[8] = {};
  float acc2[8] = {};
  float acc3[8] = {};

#pragma unroll
  for (int st = 0; st < kNumStages; ++st) {
    const int cur = st & 1;
    const int nxt = cur ^ 1;
    uint4 av0 = make_uint4(0, 0, 0, 0);
    uint32_t w0 = 0;
    uint8_t sc = 0;
    if (st + 1 < kNumStages) {
      if (a_ptr0) {
        av0 = __ldg(reinterpret_cast<const uint4*>(a_ptr0 + (st + 1) * 32));
      }
      const int g = (st + 1) * 2 + (sub_b >> 1);
      const size_t w_off =
          (static_cast<size_t>(t32_b) * 20 + (st + 1) * 4 + sub_b) * 32 + c32_b;
      const size_t s_idx = (static_cast<size_t>(g) * 80 + t32_b) * 32 + c32_b;
      w0 = __ldg(exp_w + w_off);
      sc = __ldg(exp_sc + s_idx);
    }

    if (m_base < m_count) {
#pragma unroll
      for (int sub = 0; sub < 2; ++sub) {
        const int sl0 = sub * 2;
        const int sl1 = sub * 2 + 1;
        const uint4 b0 = smem.mainloop.s_B[cur][smem_idx17(b_col128, sl0)];
        const uint4 b1 = smem.mainloop.s_B[cur][smem_idx17(b_col128, sl1)];
        uint4 a_lo = smem.mainloop.s_A[cur][smem_idx17(r0, sl0)];
        uint4 a_hi = smem.mainloop.s_A[cur][smem_idx17(r0, sl1)];
        uint4 an_lo = smem.mainloop.s_A[cur][smem_idx17(r1, sl0)];
        uint4 an_hi = smem.mainloop.s_A[cur][smem_idx17(r1, sl1)];
        FN_MMA_884_RC(acc0, a_lo.x, a_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc0, a_lo.z, a_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc0, a_hi.x, a_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc0, a_hi.z, a_hi.w, b1.z, b1.w);
        a_lo = smem.mainloop.s_A[cur][smem_idx17(r2, sl0)];
        a_hi = smem.mainloop.s_A[cur][smem_idx17(r2, sl1)];
        FN_MMA_884_RC(acc1, an_lo.x, an_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc1, an_lo.z, an_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc1, an_hi.x, an_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc1, an_hi.z, an_hi.w, b1.z, b1.w);
        an_lo = smem.mainloop.s_A[cur][smem_idx17(r3, sl0)];
        an_hi = smem.mainloop.s_A[cur][smem_idx17(r3, sl1)];
        FN_MMA_884_RC(acc2, a_lo.x, a_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc2, a_lo.z, a_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc2, a_hi.x, a_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc2, a_hi.z, a_hi.w, b1.z, b1.w);
        FN_MMA_884_RC(acc3, an_lo.x, an_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc3, an_lo.z, an_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc3, an_hi.x, an_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc3, an_hi.z, an_hi.w, b1.z, b1.w);
      }
    }

    if (st + 1 < kNumStages) {
      smem.mainloop.s_A[nxt][a_smem_idx] = av0;
      const half scalar = nvfp4_fast_hmma_scale(sc, gs_x256);
      alignas(16) half2 dec0[4];
      decode_e2m1x8(w0, __halves2half2(scalar, scalar), dec0);
      smem.mainloop.s_B[nxt][b_store_idx] =
          *reinterpret_cast<const uint4*>(dec0);
    }
    __syncthreads();
  }

  float* slice_base = accum_f32 + static_cast<size_t>(n_blk) * M * 128;
  const int row_grp = tx >> 7;  // 0..3
  const int col = tx & 127;     // 0..127

  auto stage_sub = [&](const float (&acc)[8], int local_r_base, int global_r) {
    if (global_r >= m_count) return;
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
      const int r_off = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
      const int local_r = local_r_base + r_off;
      const int c =
          warp_n * 32 + quad * 8 + ((((lane >> 1) & 1) << 1) | ((i >> 2) << 2));
      const float w = s_wt[global_r + r_off];
      smem.s_out[local_r][c] = __half2float(__float2half_rn(acc[i])) * w;
      smem.s_out[local_r][c + 1] =
          __half2float(__float2half_rn(acc[i + 1])) * w;
    }
  };

#pragma unroll
  for (int pass = 0; pass < 4; ++pass) {
    const int base_r = pass * 32;
    if (base_r >= m_count) break;
    if (pass > 0) __syncthreads();
    if (warp_m == pass) {
      stage_sub(acc0, 0, base_r);
      stage_sub(acc1, 8, base_r + 8);
      stage_sub(acc2, 16, base_r + 16);
      stage_sub(acc3, 24, base_r + 24);
    }
    __syncthreads();
    const int pass_lim = min(32, m_count - base_r);
    for (int r = row_grp; r < pass_lim; r += 4) {
      const int tok = s_tok[base_r + r];
      if (tok >= 0) {
        atomicAdd(slice_base + static_cast<size_t>(tok) * 128 + col,
                  smem.s_out[r][col]);
      }
    }
  }
}

struct PeerPtrs4 {
  uintptr_t p[4];
};

__device__ __forceinline__ void p2p_st_u32(unsigned* addr, unsigned val) {
  asm volatile("st.volatile.global.u32 [%0], %1;" ::"l"(addr), "r"(val)
               : "memory");
}

__device__ __forceinline__ unsigned p2p_ld_u32(const unsigned* addr) {
  unsigned v;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(addr));
  return v;
}

__device__ __forceinline__ uint4 p2p_ld_cv_u4(const uint4* addr) {
  uint4 v;
  asm volatile("ld.global.cv.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(addr));
  return v;
}

__device__ __forceinline__ float p2p_ld_cv_f32(const float* addr) {
  float v;
  asm volatile("ld.global.cv.f32 %0, [%1];" : "=f"(v) : "l"(addr));
  return v;
}

__device__ __forceinline__ void hc_p2p_signal_last_cta(
    PeerPtrs4 sync_ptrs, unsigned* __restrict__ local_ep_ptr, int my_rank) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();
    unsigned* done_cnt = reinterpret_cast<unsigned*>(sync_ptrs.p[my_rank]) + 4;
    const unsigned total_ctas = gridDim.x * gridDim.y * gridDim.z;
    if (atomicAdd(done_cnt, 1u) + 1u == total_ctas) {
      *done_cnt = 0u;
      const unsigned ep = *local_ep_ptr + 1u;
      *local_ep_ptr = ep;
      __threadfence_system();
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        p2p_st_u32(reinterpret_cast<unsigned*>(sync_ptrs.p[p]) + my_rank, ep);
      }
    }
  }
}

__device__ __forceinline__ void hc_p2p_wait_peers(uintptr_t my_sync_ptr,
                                                  unsigned ep) {
  if (threadIdx.x < 4) {
    const unsigned* my_flag =
        reinterpret_cast<const unsigned*>(my_sync_ptr) + threadIdx.x;
    const long long t0 = clock64();
    while (p2p_ld_u32(my_flag) < ep) {
      const long long w0 = clock64();
      while (clock64() - w0 < 64) {
      }
      if (w0 - t0 > 8000000000LL) __trap();
    }
  }
  __syncthreads();
}

template <int SLICE_COLS, bool SCALE_0125 = false>
__global__ __launch_bounds__(320, 4) void fn_moe_finalize_shared_kernel(
    float* __restrict__ accum_f32,  // [2560/SLICE_COLS, m_stride, SLICE_COLS]
    const half* __restrict__ shared_out,          // [M, 2560] or nullptr
    const half* __restrict__ shared_gate_logits,  // [M, 1] or nullptr
    half* __restrict__ out,                       // [>=M, 2560]
    int M, int pad_rows = 0, int m_stride = 0, PeerPtrs4 sync_ptrs = {},
    unsigned* __restrict__ ep_ptr = nullptr, int my_rank = 0) {
  const int tx = threadIdx.x;  // 0..319
  const int c8 = tx * 8;
  const int eff_m_stride = (m_stride > 0) ? m_stride : M;
  constexpr int kVecsPerSlice = SLICE_COLS / 8;
  const int n_blk = tx / kVecsPerSlice;
  const int col_in_slice = (tx % kVecsPerSlice) * 8;
  const int row_limit = max(M, pad_rows);
  for (int m = blockIdx.x; m < row_limit; m += gridDim.x) {
    if (m >= M) {
      *reinterpret_cast<uint4*>(out + static_cast<size_t>(m) * 2560 + c8) =
          make_uint4(0, 0, 0, 0);
    } else {
      float* src =
          accum_f32 +
          (static_cast<size_t>(n_blk) * eff_m_stride + m) * SLICE_COLS +
          col_in_slice;
      const float4 f_lo = __ldg(reinterpret_cast<const float4*>(src));
      const float4 f_hi = __ldg(reinterpret_cast<const float4*>(src + 4));
      const uint4 sv =
          (shared_out != nullptr)
              ? __ldg(reinterpret_cast<const uint4*>(
                    shared_out + static_cast<size_t>(m) * 2560 + c8))
              : make_uint4(0, 0, 0, 0);
      const half gl_h = (shared_out != nullptr && shared_gate_logits != nullptr)
                            ? __ldg(shared_gate_logits + m)
                            : __float2half(0.0f);
      asm volatile("st.global.cs.v4.f32 [%0], {%1, %2, %3, %4};"
                   :
                   : "l"(src), "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f)
                   : "memory");
      asm volatile("st.global.cs.v4.f32 [%0], {%1, %2, %3, %4};"
                   :
                   : "l"(src + 4), "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f)
                   : "memory");
      const float sum[8] = {f_lo.x, f_lo.y, f_lo.z, f_lo.w,
                            f_hi.x, f_hi.y, f_hi.z, f_hi.w};

      uint4 out_v;
      half* oh = reinterpret_cast<half*>(&out_v);
      if (shared_out != nullptr) {
        const half* sh = reinterpret_cast<const half*>(&sv);
        if (shared_gate_logits != nullptr) {
          const half g_sig =
              __float2half_rn(1.0f / (1.0f + expf(-__half2float(gl_h))));
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            const half r_h = __float2half_rn(sum[j]);
            const half s_h = __hmul(g_sig, sh[j]);
            oh[j] = SCALE_0125
                        ? __float2half_rn(
                              (__half2float(r_h) + __half2float(s_h)) * 0.125f)
                        : __hadd(r_h, s_h);
          }
        } else {
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            const half r_h = __float2half_rn(sum[j]);
            oh[j] =
                SCALE_0125
                    ? __float2half_rn(
                          (__half2float(r_h) + __half2float(sh[j])) * 0.125f)
                    : __hadd(r_h, sh[j]);
          }
        }
      } else {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const half r_h = __float2half_rn(sum[j]);
          oh[j] =
              SCALE_0125 ? __float2half_rn(__half2float(r_h) * 0.125f) : r_h;
        }
      }
      *reinterpret_cast<uint4*>(out + static_cast<size_t>(m) * 2560 + c8) =
          out_v;
    }
  }
  if (ep_ptr != nullptr) {
    hc_p2p_signal_last_cta(sync_ptrs, ep_ptr, my_rank);
  }
}

template <int SLICE_COLS, bool SCALE_0125 = false>
__global__ __launch_bounds__(320, 4) void fn_moe_finalize_shared_f16_kernel(
    half* __restrict__ accum_f16,  // [2560/SLICE_COLS, m_stride, SLICE_COLS]
    const half* __restrict__ shared_out,          // [M, 2560] or nullptr
    const half* __restrict__ shared_gate_logits,  // [M, 1] or nullptr
    half* __restrict__ out,                       // [>=M, 2560]
    int M, int pad_rows = 0, int m_stride = 0, PeerPtrs4 sync_ptrs = {},
    unsigned* __restrict__ ep_ptr = nullptr, int my_rank = 0) {
  const int tx = threadIdx.x;  // 0..319
  const int c8 = tx * 8;
  const int eff_m_stride = (m_stride > 0) ? m_stride : M;
  constexpr int kVecsPerSlice = SLICE_COLS / 8;
  const int n_blk = tx / kVecsPerSlice;
  const int col_in_slice = (tx % kVecsPerSlice) * 8;
  const int row_limit = max(M, pad_rows);
  for (int m = blockIdx.x; m < row_limit; m += gridDim.x) {
    if (m >= M) {
      *reinterpret_cast<uint4*>(out + static_cast<size_t>(m) * 2560 + c8) =
          make_uint4(0, 0, 0, 0);
    } else {
      half* src = accum_f16 +
                  (static_cast<size_t>(n_blk) * eff_m_stride + m) * SLICE_COLS +
                  col_in_slice;
      const uint4 rv = __ldg(reinterpret_cast<const uint4*>(src));
      const uint4 sv =
          (shared_out != nullptr)
              ? __ldg(reinterpret_cast<const uint4*>(
                    shared_out + static_cast<size_t>(m) * 2560 + c8))
              : make_uint4(0, 0, 0, 0);
      const half gl_h = (shared_out != nullptr && shared_gate_logits != nullptr)
                            ? __ldg(shared_gate_logits + m)
                            : __float2half(0.0f);
      *reinterpret_cast<uint4*>(src) = make_uint4(0, 0, 0, 0);
      const half* rh = reinterpret_cast<const half*>(&rv);

      uint4 out_v;
      half* oh = reinterpret_cast<half*>(&out_v);
      if (shared_out != nullptr) {
        const half* sh = reinterpret_cast<const half*>(&sv);
        if (shared_gate_logits != nullptr) {
          const half g_sig =
              __float2half_rn(1.0f / (1.0f + expf(-__half2float(gl_h))));
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            const half s_h = __hmul(g_sig, sh[j]);
            oh[j] =
                SCALE_0125
                    ? __float2half_rn(
                          (__half2float(rh[j]) + __half2float(s_h)) * 0.125f)
                    : __hadd(rh[j], s_h);
          }
        } else {
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            oh[j] =
                SCALE_0125
                    ? __float2half_rn(
                          (__half2float(rh[j]) + __half2float(sh[j])) * 0.125f)
                    : __hadd(rh[j], sh[j]);
          }
        }
      } else {
        if constexpr (SCALE_0125) {
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            oh[j] = __float2half_rn(__half2float(rh[j]) * 0.125f);
          }
        } else {
          out_v = rv;
        }
      }
      *reinterpret_cast<uint4*>(out + static_cast<size_t>(m) * 2560 + c8) =
          out_v;
    }
  }
  if (ep_ptr != nullptr) {
    hc_p2p_signal_last_cta(sync_ptrs, ep_ptr, my_rank);
  }
}

__global__ __launch_bounds__(320, 3) void shared_gate_mul_inplace_kernel(
    half* __restrict__ out, const half* __restrict__ gate_logits) {
  __shared__ half s_sig;
  const int m = blockIdx.x;
  const int tx = threadIdx.x;
  if (tx == 0) {
    const float gf = __half2float(gate_logits[m]);
    s_sig = __float2half_rn(1.0f / (1.0f + expf(-gf)));
  }
  __syncthreads();
  const half2 sig2 = __halves2half2(s_sig, s_sig);
  uint4* row_ptr =
      reinterpret_cast<uint4*>(out + static_cast<size_t>(m) * 2560 + tx * 8);
  uint4 v = __ldg(row_ptr);
  half2* v2 = reinterpret_cast<half2*>(&v);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    v2[i] = __hmul2(v2[i], sig2);
  }
  *row_ptr = v;
}

void shared_gate_mul_inplace_out(torch::Tensor out, torch::Tensor gate_logits) {
  const c10::cuda::CUDAGuard guard(out.device());
  const int M = static_cast<int>(out.size(0));
  if (M == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream();
  shared_gate_mul_inplace_kernel<<<M, 320, 0, stream>>>(
      reinterpret_cast<half*>(out.data_ptr()),
      reinterpret_cast<const half*>(gate_logits.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ __launch_bounds__(320, 4) void shared_gate_dot_mul_inplace_kernel(
    half* __restrict__ out, const half* __restrict__ x,
    const half* __restrict__ gate_weight) {
  __shared__ float s_warp_dot[10];
  __shared__ half s_sig;
  const int m = blockIdx.x;
  const int tx = threadIdx.x;
  const int lane = tx & 31;
  const int warp = tx >> 5;
  const int c8 = tx * 8;

  const uint4 xv = __ldg(
      reinterpret_cast<const uint4*>(x + static_cast<size_t>(m) * 2560 + c8));
  const uint4 wv = __ldg(reinterpret_cast<const uint4*>(gate_weight + c8));
  const half* xh = reinterpret_cast<const half*>(&xv);
  const half* wh = reinterpret_cast<const half*>(&wv);
  float dot = 0.0f;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    dot = __fmaf_rn(__half2float(xh[j]), __half2float(wh[j]), dot);
  }
#pragma unroll
  for (int sh = 16; sh > 0; sh >>= 1) {
    dot += __shfl_xor_sync(0xffffffffu, dot, sh);
  }
  if (lane == 0) s_warp_dot[warp] = dot;
  __syncthreads();
  if (tx == 0) {
    float total = 0.0f;
#pragma unroll
    for (int w = 0; w < 10; ++w) total += s_warp_dot[w];
    const float gf = __half2float(__float2half_rn(total));
    s_sig = __float2half_rn(1.0f / (1.0f + expf(-gf)));
  }
  __syncthreads();

  const half2 sig2 = __halves2half2(s_sig, s_sig);
  uint4* row_ptr =
      reinterpret_cast<uint4*>(out + static_cast<size_t>(m) * 2560 + c8);
  uint4 v = __ldg(row_ptr);
  half2* v2 = reinterpret_cast<half2*>(&v);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    v2[i] = __hmul2(v2[i], sig2);
  }
  *row_ptr = v;
}

void shared_gate_dot_mul_inplace_out(torch::Tensor out, torch::Tensor x,
                                     torch::Tensor gate_weight) {
  const c10::cuda::CUDAGuard guard(out.device());
  const int M = static_cast<int>(out.size(0));
  if (M == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream();
  shared_gate_dot_mul_inplace_kernel<<<M, 320, 0, stream>>>(
      reinterpret_cast<half*>(out.data_ptr()),
      reinterpret_cast<const half*>(x.data_ptr()),
      reinterpret_cast<const half*>(gate_weight.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ __launch_bounds__(256, 4) void shared_gate_dot_only_kernel(
    half* __restrict__ gate_logits, const half* __restrict__ x,
    const half* __restrict__ gate_weight) {
  __shared__ float s_warp_dot[8];
  const int m = blockIdx.x;
  const int tx = threadIdx.x;
  const int lane = tx & 31;
  const int warp = tx >> 5;

  float dot = 0.0f;
  for (int c8 = tx * 8; c8 < 2560; c8 += 256 * 8) {
    const uint4 xv = __ldg(
        reinterpret_cast<const uint4*>(x + static_cast<size_t>(m) * 2560 + c8));
    const uint4 wv = __ldg(reinterpret_cast<const uint4*>(gate_weight + c8));
    const half* xh = reinterpret_cast<const half*>(&xv);
    const half* wh = reinterpret_cast<const half*>(&wv);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      dot = __fmaf_rn(__half2float(xh[j]), __half2float(wh[j]), dot);
    }
  }
#pragma unroll
  for (int sh = 16; sh > 0; sh >>= 1) {
    dot += __shfl_xor_sync(0xffffffffu, dot, sh);
  }
  if (lane == 0) s_warp_dot[warp] = dot;
  __syncthreads();
  if (tx == 0) {
    float total = 0.0f;
#pragma unroll
    for (int w = 0; w < 8; ++w) total += s_warp_dot[w];
    gate_logits[m] = __float2half_rn(total);
  }
}

void shared_gate_dot_out(torch::Tensor gate_logits, torch::Tensor x,
                         torch::Tensor gate_weight) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int M = static_cast<int>(x.size(0));
  if (M == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream();
  shared_gate_dot_only_kernel<<<M, 256, 0, stream>>>(
      reinterpret_cast<half*>(gate_logits.data_ptr()),
      reinterpret_cast<const half*>(x.data_ptr()),
      reinterpret_cast<const half*>(gate_weight.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Fused SiLU*Mul + Shared-Expert Gate Extraction from [M, 328] packed GEMM
// output: cols 0..159 = gate_proj, cols 160..319 = up_proj, col 320 =
// shared_expert_gate!
__global__ __launch_bounds__(128, 8) void silu_mul_split_gate_kernel(
    half* __restrict__ act_out, half* __restrict__ gate_logits,
    const half* __restrict__ raw, int M) {
  const int warp = threadIdx.x >> 5;  // 0..3
  const int lane = threadIdx.x & 31;  // 0..31
  const int m = blockIdx.x * 4 + warp;
  if (m >= M) return;
  const half* row = raw + static_cast<size_t>(m) * 328;
  if (lane < 20) {
    const uint4 g8 = __ldg(reinterpret_cast<const uint4*>(row + lane * 8));
    const uint4 u8 =
        __ldg(reinterpret_cast<const uint4*>(row + 160 + lane * 8));
    const half* gh = reinterpret_cast<const half*>(&g8);
    const half* uh = reinterpret_cast<const half*>(&u8);
    uint4 out8;
    half* oh = reinterpret_cast<half*>(&out8);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float gf = __half2float(gh[j]);
      const half act_h = __float2half_rn(gf / (1.0f + expf(-gf)));
      oh[j] = __hmul(act_h, uh[j]);
    }
    *reinterpret_cast<uint4*>(act_out + static_cast<size_t>(m) * 160 +
                              lane * 8) = out8;
  } else if (lane == 20) {
    gate_logits[m] = __ldg(row + 320);
  }
}

void silu_mul_split_gate_out(torch::Tensor act_out, torch::Tensor gate_logits,
                             torch::Tensor raw) {
  const c10::cuda::CUDAGuard guard(raw.device());
  const int M = static_cast<int>(raw.size(0));
  if (M == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream();
  silu_mul_split_gate_kernel<<<(M + 3) / 4, 128, 0, stream>>>(
      reinterpret_cast<half*>(act_out.data_ptr()),
      reinterpret_cast<half*>(gate_logits.data_ptr()),
      reinterpret_cast<const half*>(raw.data_ptr()), M);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Fused split + SwiGLU for [M, 840] = [router(512); shared_gate(160);
// shared_up(160); shared_eg(1); pad(7)] 256 threads/CTA, 3 rows/CTA (255 active
// threads = 99.6% utilization, 0 barriers, 0 smem!)
__global__ __launch_bounds__(256, 6) void split_router_silu_mul_gate_kernel(
    half* __restrict__ router_out, half* __restrict__ act_out,
    half* __restrict__ gate_logits, const half* __restrict__ raw, int M) {
  const int tx = threadIdx.x;
  if (tx >= 255) return;
  const int sub = tx / 85;
  const int tid = tx - sub * 85;  // 0..84
  const int m = blockIdx.x * 3 + sub;
  if (m >= M) return;

  const half* row = raw + static_cast<size_t>(m) * 840;
  if (tid < 64) {
    *reinterpret_cast<uint4*>(router_out + static_cast<size_t>(m) * 512 +
                              tid * 8) =
        __ldg(reinterpret_cast<const uint4*>(row + tid * 8));
  } else if (tid < 84) {
    const int k = tid - 64;  // 0..19
    const uint4 g8 = __ldg(reinterpret_cast<const uint4*>(row + 512 + k * 8));
    const uint4 u8 = __ldg(reinterpret_cast<const uint4*>(row + 672 + k * 8));
    const half* gh = reinterpret_cast<const half*>(&g8);
    const half* uh = reinterpret_cast<const half*>(&u8);
    uint4 out8;
    half* oh = reinterpret_cast<half*>(&out8);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float gf = __half2float(gh[j]);
      const half act_h = __float2half_rn(gf / (1.0f + expf(-gf)));
      oh[j] = __hmul(act_h, uh[j]);
    }
    *reinterpret_cast<uint4*>(act_out + static_cast<size_t>(m) * 160 + k * 8) =
        out8;
  } else {
    gate_logits[m] = __ldg(row + 832);
  }
}

void split_router_silu_mul_gate_out(torch::Tensor router_out,
                                    torch::Tensor act_out,
                                    torch::Tensor gate_logits,
                                    torch::Tensor raw) {
  const c10::cuda::CUDAGuard guard(raw.device());
  const int M = static_cast<int>(raw.size(0));
  if (M == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream();
  split_router_silu_mul_gate_kernel<<<(M + 2) / 3, 256, 0, stream>>>(
      reinterpret_cast<half*>(router_out.data_ptr()),
      reinterpret_cast<half*>(act_out.data_ptr()),
      reinterpret_cast<half*>(gate_logits.data_ptr()),
      reinterpret_cast<const half*>(raw.data_ptr()), M);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Fused tail copy for [M, 4120] = [qkv(2560); z(1536); b(12); a(12)]
// Keeps qkv as zero-copy view raw[:, :2560]; copies z (uint4) and b, a (uint2,
// 8B-aligned for odd M!)
__global__ __launch_bounds__(256, 6) void split_gdn_4120_tails_kernel(
    half* __restrict__ z_out, half* __restrict__ b_out,
    half* __restrict__ a_out, const half* __restrict__ raw, int M) {
  const int m = blockIdx.x;
  if (m >= M) return;
  const int tx = threadIdx.x;
  const half* row = raw + static_cast<size_t>(m) * 4120;
  if (tx < 192) {
    *reinterpret_cast<uint4*>(z_out + static_cast<size_t>(m) * 1536 + tx * 8) =
        __ldg(reinterpret_cast<const uint4*>(row + 2560 + tx * 8));
  } else if (tx < 195) {
    const int k = tx - 192;  // 0..2 -> 3 uint2s = 12 halves
    *reinterpret_cast<uint2*>(b_out + static_cast<size_t>(m) * 12 + k * 4) =
        __ldg(reinterpret_cast<const uint2*>(row + 4096 + k * 4));
  } else if (tx < 198) {
    const int k = tx - 195;  // 0..2 -> 3 uint2s = 12 halves
    *reinterpret_cast<uint2*>(a_out + static_cast<size_t>(m) * 12 + k * 4) =
        __ldg(reinterpret_cast<const uint2*>(row + 4108 + k * 4));
  }
}

void split_gdn_4120_tails_out(torch::Tensor z_out, torch::Tensor b_out,
                              torch::Tensor a_out, torch::Tensor raw) {
  const c10::cuda::CUDAGuard guard(raw.device());
  const int M = static_cast<int>(raw.size(0));
  if (M == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream();
  split_gdn_4120_tails_kernel<<<M, 256, 0, stream>>>(
      reinterpret_cast<half*>(z_out.data_ptr()),
      reinterpret_cast<half*>(b_out.data_ptr()),
      reinterpret_cast<half*>(a_out.data_ptr()),
      reinterpret_cast<const half*>(raw.data_ptr()), M);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Fast C++ tail copy for separate GDN [M, 4096] + [M, 24] (preserves Kernel2
// 109 TFLOPs/s!)
__global__ __launch_bounds__(256, 6) void split_gdn_qkvz_ba_tails_kernel(
    half* __restrict__ z_out, half* __restrict__ b_out,
    half* __restrict__ a_out, const half* __restrict__ qkvz,
    const half* __restrict__ ba, int M) {
  const int m = blockIdx.x;
  if (m >= M) return;
  const int tx = threadIdx.x;
  if (tx < 192) {
    *reinterpret_cast<uint4*>(z_out + static_cast<size_t>(m) * 1536 + tx * 8) =
        __ldg(reinterpret_cast<const uint4*>(
            qkvz + static_cast<size_t>(m) * 4096 + 2560 + tx * 8));
  } else if (tx < 195) {
    const int k = tx - 192;
    *reinterpret_cast<uint2*>(b_out + static_cast<size_t>(m) * 12 + k * 4) =
        __ldg(reinterpret_cast<const uint2*>(ba + static_cast<size_t>(m) * 24 +
                                             k * 4));
  } else if (tx < 198) {
    const int k = tx - 195;
    *reinterpret_cast<uint2*>(a_out + static_cast<size_t>(m) * 12 + k * 4) =
        __ldg(reinterpret_cast<const uint2*>(ba + static_cast<size_t>(m) * 24 +
                                             12 + k * 4));
  }
}

void split_gdn_qkvz_ba_tails_out(torch::Tensor z_out, torch::Tensor b_out,
                                 torch::Tensor a_out, torch::Tensor qkvz,
                                 torch::Tensor ba) {
  const c10::cuda::CUDAGuard guard(qkvz.device());
  const int M = static_cast<int>(qkvz.size(0));
  if (M == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream();
  split_gdn_qkvz_ba_tails_kernel<<<M, 256, 0, stream>>>(
      reinterpret_cast<half*>(z_out.data_ptr()),
      reinterpret_cast<half*>(b_out.data_ptr()),
      reinterpret_cast<half*>(a_out.data_ptr()),
      reinterpret_cast<const half*>(qkvz.data_ptr()),
      reinterpret_cast<const half*>(ba.data_ptr()), M);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void moe_prefill_raw_out(
    torch::Tensor out, torch::Tensor x, torch::Tensor topk_ids,
    torch::Tensor topk_weights, torch::Tensor w13_weight,
    torch::Tensor w13_scale_codes, torch::Tensor w13_global_scales,
    torch::Tensor w2_weight, torch::Tensor w2_scale_codes,
    torch::Tensor w2_global_scales, std::optional<torch::Tensor> shared_out,
    std::optional<torch::Tensor> shared_gate_logits, torch::Tensor ws_counts,
    torch::Tensor ws_offsets, torch::Tensor ws_cursors,
    torch::Tensor ws_sorted_token_idx, torch::Tensor ws_sorted_weights,
    torch::Tensor ws_tiles, torch::Tensor ws_num_tiles, torch::Tensor ws_inter,
    torch::Tensor ws_accum_f32, bool force_scale_0125 = false, int64_t s0 = 0,
    int64_t s1 = 0, int64_t s2 = 0, int64_t s3 = 0,
    std::optional<torch::Tensor> ep_state = std::nullopt, int64_t my_rank = 0) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int M = static_cast<int>(x.size(0));
  if (M == 0) return;
  const int total_slots = M * kMoETopK;
  const char* mode_env = std::getenv("FN_MOE_MODE");
  const int moe_mode = mode_env ? std::atoi(mode_env) : 0;
  const char* w2_env = std::getenv("FN_MOE_W2_MODE");
  const int w2_mode = w2_env ? std::atoi(w2_env) : 5;
  const bool use_m96_w13_m64_w2 = (moe_mode == 7);
  const bool use_m128_w13_m64_w2 = (moe_mode == 8);
  const int cta_m_w13 =
      (moe_mode == 3 || use_m128_w13_m64_w2)
          ? 128
          : ((moe_mode == 4 || use_m96_w13_m64_w2)
                 ? 96
                 : ((moe_mode == 5)
                        ? 32
                        : ((moe_mode == 1 || moe_mode == 2 || moe_mode == 6)
                               ? 64
                               : ((M >= 1536) ? 64 : 32))));
  const int cta_m_w2 =
      (use_m96_w13_m64_w2 || use_m128_w13_m64_w2) ? 64 : cta_m_w13;
  const int cta_m = cta_m_w13;
  const bool use_dual_m128_w2 = (cta_m_w13 == 64) && (w2_mode == 6);
  const int max_tiles = (total_slots + cta_m_w13 - 1) / cta_m_w13 + kMoEExperts;
  const int max_tiles_w2 =
      (total_slots + cta_m_w2 - 1) / cta_m_w2 + kMoEExperts;
  const int max_tiles_w2_128 = (total_slots + 127) / 128 + kMoEExperts;
  auto stream = at::cuda::getCurrentCUDAStream();

  const char* gw13_env = std::getenv("FN_MOE_GRP_W13");
  const char* gw2_env = std::getenv("FN_MOE_GRP_W2");
  const int grp_w13 = gw13_env ? std::atoi(gw13_env) : 1;
  const int default_grp_w2 =
      (w2_mode == 1 || w2_mode == 2) ? 32 : ((M >= 10240) ? 512 : 1);
  const int grp_w2 = gw2_env ? std::atoi(gw2_env) : default_grp_w2;

  const int tok_per_blk = std::max(256, ((M + 63) / 64 + 31) & ~31);
  const int num_blks = (M + tok_per_blk - 1) / tok_per_blk;
  const int slots_per_blk = tok_per_blk * kMoETopK;

  MoETileDesc* tiles_w13_ptr =
      reinterpret_cast<MoETileDesc*>(ws_tiles.data_ptr());
  const bool need_separate_w2_tiles =
      (cta_m_w13 != cta_m_w2) || (grp_w13 != grp_w2);
  const bool can_dual_wave_tiles =
      !use_dual_m128_w2 && (ws_tiles.size(0) >= max_tiles + max_tiles_w2) &&
      need_separate_w2_tiles;
  MoETileDesc* tiles_w2_ptr =
      can_dual_wave_tiles ? (tiles_w13_ptr + max_tiles) : tiles_w13_ptr;
  int32_t* num_tiles_w2_ptr = (can_dual_wave_tiles && cta_m_w13 != cta_m_w2)
                                  ? ws_cursors.data_ptr<int32_t>()
                                  : ws_num_tiles.data_ptr<int32_t>();
  MoETileDesc* tiles_w2_128_ptr = tiles_w13_ptr + max_tiles;
  int32_t* num_tiles_w2_128_ptr = ws_offsets.data_ptr<int32_t>();

  if (!use_dual_m128_w2 &&
      ws_counts.numel() >= static_cast<int64_t>(num_blks) * kMoEExperts) {
    if (topk_ids.scalar_type() == at::kInt) {
      moe_block_hist_kernel<int32_t><<<num_blks, 256, 0, stream>>>(
          topk_ids.data_ptr<int32_t>(), ws_counts.data_ptr<int32_t>(),
          slots_per_blk, total_slots);
    } else {
      moe_block_hist_kernel<int64_t><<<num_blks, 256, 0, stream>>>(
          topk_ids.data_ptr<int64_t>(), ws_counts.data_ptr<int32_t>(),
          slots_per_blk, total_slots);
    }
    moe_block_scan_wave_schedule_kernel<<<1, kMoEExperts, 0, stream>>>(
        ws_counts.data_ptr<int32_t>(), num_blks, ws_offsets.data_ptr<int32_t>(),
        tiles_w13_ptr, tiles_w2_ptr, ws_num_tiles.data_ptr<int32_t>(),
        num_tiles_w2_ptr, cta_m_w13, cta_m_w2, grp_w13,
        can_dual_wave_tiles ? grp_w2 : grp_w13);
    if (topk_ids.scalar_type() == at::kInt) {
      moe_block_scatter_kernel<int32_t><<<num_blks, 256, 0, stream>>>(
          topk_ids.data_ptr<int32_t>(), topk_weights.data_ptr<float>(),
          ws_offsets.data_ptr<int32_t>(), ws_counts.data_ptr<int32_t>(),
          ws_sorted_token_idx.data_ptr<int32_t>(),
          ws_sorted_weights.data_ptr<float>(), slots_per_blk, total_slots);
    } else {
      moe_block_scatter_kernel<int64_t><<<num_blks, 256, 0, stream>>>(
          topk_ids.data_ptr<int64_t>(), topk_weights.data_ptr<float>(),
          ws_offsets.data_ptr<int32_t>(), ws_counts.data_ptr<int32_t>(),
          ws_sorted_token_idx.data_ptr<int32_t>(),
          ws_sorted_weights.data_ptr<float>(), slots_per_blk, total_slots);
    }
  } else {
    C10_CUDA_CHECK(cudaMemsetAsync(ws_counts.data_ptr(), 0,
                                   kMoEExperts * sizeof(int32_t), stream));
    const int hist_blocks = std::min((total_slots + 255) / 256, 160);
    if (topk_ids.scalar_type() == at::kInt) {
      moe_hist_kernel<int32_t><<<hist_blocks, 256, 0, stream>>>(
          topk_ids.data_ptr<int32_t>(), ws_counts.data_ptr<int32_t>(),
          total_slots);
    } else {
      moe_hist_kernel<int64_t><<<hist_blocks, 256, 0, stream>>>(
          topk_ids.data_ptr<int64_t>(), ws_counts.data_ptr<int32_t>(),
          total_slots);
    }
    if (use_dual_m128_w2) {
      moe_scan_schedule_dual_kernel<<<1, kMoEExperts, 0, stream>>>(
          ws_counts.data_ptr<int32_t>(), ws_cursors.data_ptr<int32_t>(),
          tiles_w13_ptr, ws_num_tiles.data_ptr<int32_t>(), 64, tiles_w2_128_ptr,
          num_tiles_w2_128_ptr, 128);
    } else {
      moe_scan_schedule_kernel<<<1, kMoEExperts, 0, stream>>>(
          ws_counts.data_ptr<int32_t>(), ws_offsets.data_ptr<int32_t>(),
          ws_cursors.data_ptr<int32_t>(), tiles_w13_ptr,
          ws_num_tiles.data_ptr<int32_t>(), cta_m);
    }
    const int scatter_blocks = (total_slots + 1023) / 1024;
    if (topk_ids.scalar_type() == at::kInt) {
      moe_scatter_kernel<int32_t><<<scatter_blocks, 256, 0, stream>>>(
          topk_ids.data_ptr<int32_t>(), topk_weights.data_ptr<float>(),
          ws_cursors.data_ptr<int32_t>(),
          ws_sorted_token_idx.data_ptr<int32_t>(),
          ws_sorted_weights.data_ptr<float>(), total_slots);
    } else {
      moe_scatter_kernel<int64_t><<<scatter_blocks, 256, 0, stream>>>(
          topk_ids.data_ptr<int64_t>(), topk_weights.data_ptr<float>(),
          ws_cursors.data_ptr<int32_t>(),
          ws_sorted_token_idx.data_ptr<int32_t>(),
          ws_sorted_weights.data_ptr<float>(), total_slots);
    }
  }

  const int cur_dev = x.device().index();
  static bool configured_moe_devs[16] = {};
  if (cur_dev >= 0 && cur_dev < 16 && !configured_moe_devs[cur_dev]) {
    constexpr size_t kSmemW13M64N320 = 61184;
    constexpr size_t kSmemW13M64N320Small = 52480;
    constexpr size_t kSmemW13M96N320 = 70016;
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fn_nvfp4_w13_m128_n160_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fn_nvfp4_w2_m128_n128_atomic_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fn_nvfp4_w2_m128_n64_atomic_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fn_nvfp4_w13_m96_n320_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fn_nvfp4_w13_m96_n320_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemW13M96N320));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fn_nvfp4_w2_m96_n128_atomic_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fn_nvfp4_w13_m64_n160_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fn_nvfp4_w13_m64_n320_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fn_nvfp4_w13_m64_n320_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemW13M64N320));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fn_nvfp4_w13_m64_n320_small_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fn_nvfp4_w13_m64_n320_small_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemW13M64N320Small));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fn_nvfp4_w2_m64_n128_atomic_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fn_nvfp4_w2_m64_n128_3cta_kernel<false>,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fn_nvfp4_w2_m64_n128_3cta_kernel<true>,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fn_nvfp4_w2_m64_n64_preload_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fn_nvfp4_w2_m64_n160_atomic_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fn_nvfp4_w2_m64_n128_f16x2_atomic_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    constexpr size_t kSmemW2M64N256 = sizeof(SharedW2M64N256);
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fn_nvfp4_w2_m64_n256_atomic_kernel<false>,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fn_nvfp4_w2_m64_n256_atomic_kernel<false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemW2M64N256));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fn_nvfp4_w2_m64_n256_atomic_kernel<true>,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fn_nvfp4_w2_m64_n256_atomic_kernel<true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemW2M64N256));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fn_nvfp4_w13_swiglu_pipelined_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fn_nvfp4_w2_m32_n128_atomic_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    configured_moe_devs[cur_dev] = true;
  }

  const half* shared_ptr =
      (shared_out.has_value() && shared_out->defined())
          ? reinterpret_cast<const half*>(shared_out->data_ptr())
          : nullptr;
  const half* gate_ptr =
      (shared_gate_logits.has_value() && shared_gate_logits->defined())
          ? reinterpret_cast<const half*>(shared_gate_logits->data_ptr())
          : nullptr;

  int slice_cols = 128;
  bool use_f16_accum = false;
  const int m_stride_pad =
      (((M & 511) == 0) &&
       ws_accum_f32.numel() >= static_cast<int64_t>(M + 16) * 2560)
          ? (M + 16)
          : M;
  int eff_m_stride = M;

  if (cta_m == 128) {
    fn_nvfp4_w13_m128_n160_kernel<<<dim3(2, max_tiles), 640, 0, stream>>>(
        reinterpret_cast<const half*>(x.data_ptr()),
        reinterpret_cast<const uint32_t*>(w13_weight.data_ptr()),
        w13_scale_codes.data_ptr<uint8_t>(),
        w13_global_scales.data_ptr<float>(),
        ws_sorted_token_idx.data_ptr<int32_t>(), tiles_w13_ptr,
        ws_num_tiles.data_ptr<int32_t>(),
        reinterpret_cast<half*>(ws_inter.data_ptr()));

    if (use_m128_w13_m64_w2) {
      eff_m_stride = m_stride_pad;
      use_f16_accum = (w2_mode != 5);
      if (use_f16_accum) {
        fn_nvfp4_w2_m64_n128_3cta_kernel<true>
            <<<dim3(max_tiles_w2, 20), 256, 0, stream>>>(
                reinterpret_cast<const half*>(ws_inter.data_ptr()),
                reinterpret_cast<const uint32_t*>(w2_weight.data_ptr()),
                w2_scale_codes.data_ptr<uint8_t>(),
                w2_global_scales.data_ptr<float>(),
                ws_sorted_token_idx.data_ptr<int32_t>(),
                ws_sorted_weights.data_ptr<float>(), tiles_w2_ptr,
                num_tiles_w2_ptr, ws_accum_f32.data_ptr(), eff_m_stride);
      } else {
        fn_nvfp4_w2_m64_n128_3cta_kernel<false>
            <<<dim3(max_tiles_w2, 20), 256, 0, stream>>>(
                reinterpret_cast<const half*>(ws_inter.data_ptr()),
                reinterpret_cast<const uint32_t*>(w2_weight.data_ptr()),
                w2_scale_codes.data_ptr<uint8_t>(),
                w2_global_scales.data_ptr<float>(),
                ws_sorted_token_idx.data_ptr<int32_t>(),
                ws_sorted_weights.data_ptr<float>(), tiles_w2_ptr,
                num_tiles_w2_ptr, ws_accum_f32.data_ptr(), eff_m_stride);
      }
    } else {
      fn_nvfp4_w2_m128_n128_atomic_kernel<<<dim3(max_tiles, 20), 512, 0,
                                            stream>>>(
          reinterpret_cast<const half*>(ws_inter.data_ptr()),
          reinterpret_cast<const uint32_t*>(w2_weight.data_ptr()),
          w2_scale_codes.data_ptr<uint8_t>(),
          w2_global_scales.data_ptr<float>(),
          ws_sorted_token_idx.data_ptr<int32_t>(),
          ws_sorted_weights.data_ptr<float>(), tiles_w2_ptr, num_tiles_w2_ptr,
          ws_accum_f32.data_ptr<float>(), M);
    }
  } else if (cta_m == 96) {
    fn_nvfp4_w13_m96_n320_kernel<<<max_tiles, 640, 70016, stream>>>(
        reinterpret_cast<const half*>(x.data_ptr()),
        reinterpret_cast<const uint32_t*>(w13_weight.data_ptr()),
        w13_scale_codes.data_ptr<uint8_t>(),
        w13_global_scales.data_ptr<float>(),
        ws_sorted_token_idx.data_ptr<int32_t>(), tiles_w13_ptr,
        ws_num_tiles.data_ptr<int32_t>(),
        reinterpret_cast<half*>(ws_inter.data_ptr()));

    if (use_m96_w13_m64_w2) {
      if (w2_mode == 4) {
        slice_cols = 160;
        fn_nvfp4_w2_m64_n160_atomic_kernel<<<dim3(max_tiles_w2, 16), 320, 0,
                                             stream>>>(
            reinterpret_cast<const half*>(ws_inter.data_ptr()),
            reinterpret_cast<const uint32_t*>(w2_weight.data_ptr()),
            w2_scale_codes.data_ptr<uint8_t>(),
            w2_global_scales.data_ptr<float>(),
            ws_sorted_token_idx.data_ptr<int32_t>(),
            ws_sorted_weights.data_ptr<float>(), tiles_w2_ptr, num_tiles_w2_ptr,
            ws_accum_f32.data_ptr<float>(), M);
      } else if (w2_mode == 0) {
        eff_m_stride = m_stride_pad;
        use_f16_accum = true;
        fn_nvfp4_w2_m64_n128_3cta_kernel<true>
            <<<dim3(max_tiles_w2, 20), 256, 0, stream>>>(
                reinterpret_cast<const half*>(ws_inter.data_ptr()),
                reinterpret_cast<const uint32_t*>(w2_weight.data_ptr()),
                w2_scale_codes.data_ptr<uint8_t>(),
                w2_global_scales.data_ptr<float>(),
                ws_sorted_token_idx.data_ptr<int32_t>(),
                ws_sorted_weights.data_ptr<float>(), tiles_w2_ptr,
                num_tiles_w2_ptr, ws_accum_f32.data_ptr(), eff_m_stride);
      } else if (w2_mode == 5) {
        eff_m_stride = m_stride_pad;
        fn_nvfp4_w2_m64_n128_3cta_kernel<false>
            <<<dim3(max_tiles_w2, 20), 256, 0, stream>>>(
                reinterpret_cast<const half*>(ws_inter.data_ptr()),
                reinterpret_cast<const uint32_t*>(w2_weight.data_ptr()),
                w2_scale_codes.data_ptr<uint8_t>(),
                w2_global_scales.data_ptr<float>(),
                ws_sorted_token_idx.data_ptr<int32_t>(),
                ws_sorted_weights.data_ptr<float>(), tiles_w2_ptr,
                num_tiles_w2_ptr, ws_accum_f32.data_ptr(), eff_m_stride);
      } else {
        eff_m_stride = m_stride_pad;
        fn_nvfp4_w2_m64_n128_atomic_kernel<<<dim3(max_tiles_w2, 20), 256, 0,
                                             stream>>>(
            reinterpret_cast<const half*>(ws_inter.data_ptr()),
            reinterpret_cast<const uint32_t*>(w2_weight.data_ptr()),
            w2_scale_codes.data_ptr<uint8_t>(),
            w2_global_scales.data_ptr<float>(),
            ws_sorted_token_idx.data_ptr<int32_t>(),
            ws_sorted_weights.data_ptr<float>(), tiles_w2_ptr, num_tiles_w2_ptr,
            ws_accum_f32.data_ptr<float>(), eff_m_stride);
      }
    } else {
      fn_nvfp4_w2_m96_n128_atomic_kernel<<<dim3(max_tiles, 20), 384, 0,
                                           stream>>>(
          reinterpret_cast<const half*>(ws_inter.data_ptr()),
          reinterpret_cast<const uint32_t*>(w2_weight.data_ptr()),
          w2_scale_codes.data_ptr<uint8_t>(),
          w2_global_scales.data_ptr<float>(),
          ws_sorted_token_idx.data_ptr<int32_t>(),
          ws_sorted_weights.data_ptr<float>(), tiles_w2_ptr, num_tiles_w2_ptr,
          ws_accum_f32.data_ptr<float>(), M);
    }
  } else if (cta_m == 64) {
    if (moe_mode == 0 || moe_mode == 2) {
      fn_nvfp4_w13_m64_n320_kernel<<<max_tiles, 640, 61184, stream>>>(
          reinterpret_cast<const half*>(x.data_ptr()),
          reinterpret_cast<const uint32_t*>(w13_weight.data_ptr()),
          w13_scale_codes.data_ptr<uint8_t>(),
          w13_global_scales.data_ptr<float>(),
          ws_sorted_token_idx.data_ptr<int32_t>(),
          reinterpret_cast<const MoETileDesc*>(ws_tiles.data_ptr()),
          ws_num_tiles.data_ptr<int32_t>(),
          reinterpret_cast<half*>(ws_inter.data_ptr()));
    } else if (moe_mode == 6) {
      fn_nvfp4_w13_m64_n320_small_kernel<<<max_tiles, 640, 52480, stream>>>(
          reinterpret_cast<const half*>(x.data_ptr()),
          reinterpret_cast<const uint32_t*>(w13_weight.data_ptr()),
          w13_scale_codes.data_ptr<uint8_t>(),
          w13_global_scales.data_ptr<float>(),
          ws_sorted_token_idx.data_ptr<int32_t>(),
          reinterpret_cast<const MoETileDesc*>(ws_tiles.data_ptr()),
          ws_num_tiles.data_ptr<int32_t>(),
          reinterpret_cast<half*>(ws_inter.data_ptr()));
    } else {
      fn_nvfp4_w13_m64_n160_kernel<<<dim3(2, max_tiles), 320, 0, stream>>>(
          reinterpret_cast<const half*>(x.data_ptr()),
          reinterpret_cast<const uint32_t*>(w13_weight.data_ptr()),
          w13_scale_codes.data_ptr<uint8_t>(),
          w13_global_scales.data_ptr<float>(),
          ws_sorted_token_idx.data_ptr<int32_t>(),
          reinterpret_cast<const MoETileDesc*>(ws_tiles.data_ptr()),
          ws_num_tiles.data_ptr<int32_t>(),
          reinterpret_cast<half*>(ws_inter.data_ptr()));
    }

    constexpr size_t kSmemW2N256 = sizeof(SharedW2M64N256);
    if (use_dual_m128_w2) {
      slice_cols = 64;
      fn_nvfp4_w2_m128_n64_atomic_kernel<<<dim3(max_tiles_w2_128, 40), 256, 0,
                                           stream>>>(
          reinterpret_cast<const half*>(ws_inter.data_ptr()),
          reinterpret_cast<const uint32_t*>(w2_weight.data_ptr()),
          w2_scale_codes.data_ptr<uint8_t>(),
          w2_global_scales.data_ptr<float>(),
          ws_sorted_token_idx.data_ptr<int32_t>(),
          ws_sorted_weights.data_ptr<float>(), tiles_w2_128_ptr,
          num_tiles_w2_128_ptr, ws_accum_f32.data_ptr<float>(), M);
    } else if (w2_mode == 7) {
      slice_cols = 64;
      fn_nvfp4_w2_m64_n64_preload_kernel<<<dim3(max_tiles, 40), 128, 0,
                                           stream>>>(
          reinterpret_cast<const half*>(ws_inter.data_ptr()),
          reinterpret_cast<const uint32_t*>(w2_weight.data_ptr()),
          w2_scale_codes.data_ptr<uint8_t>(),
          w2_global_scales.data_ptr<float>(),
          ws_sorted_token_idx.data_ptr<int32_t>(),
          ws_sorted_weights.data_ptr<float>(), tiles_w2_ptr,
          ws_num_tiles.data_ptr<int32_t>(), ws_accum_f32.data_ptr<float>(), M);
    } else if (w2_mode == 0) {
      eff_m_stride = m_stride_pad;
      use_f16_accum = true;
      fn_nvfp4_w2_m64_n128_3cta_kernel<true>
          <<<dim3(max_tiles, 20), 256, 0, stream>>>(
              reinterpret_cast<const half*>(ws_inter.data_ptr()),
              reinterpret_cast<const uint32_t*>(w2_weight.data_ptr()),
              w2_scale_codes.data_ptr<uint8_t>(),
              w2_global_scales.data_ptr<float>(),
              ws_sorted_token_idx.data_ptr<int32_t>(),
              ws_sorted_weights.data_ptr<float>(), tiles_w2_ptr,
              ws_num_tiles.data_ptr<int32_t>(), ws_accum_f32.data_ptr(),
              eff_m_stride);
    } else if (w2_mode == 5) {
      eff_m_stride = m_stride_pad;
      fn_nvfp4_w2_m64_n128_3cta_kernel<false>
          <<<dim3(max_tiles, 20), 256, 0, stream>>>(
              reinterpret_cast<const half*>(ws_inter.data_ptr()),
              reinterpret_cast<const uint32_t*>(w2_weight.data_ptr()),
              w2_scale_codes.data_ptr<uint8_t>(),
              w2_global_scales.data_ptr<float>(),
              ws_sorted_token_idx.data_ptr<int32_t>(),
              ws_sorted_weights.data_ptr<float>(), tiles_w2_ptr,
              ws_num_tiles.data_ptr<int32_t>(), ws_accum_f32.data_ptr(),
              eff_m_stride);
    } else if (w2_mode == 1) {
      slice_cols = 256;
      fn_nvfp4_w2_m64_n256_atomic_kernel<false>
          <<<dim3(max_tiles, 10), 512, kSmemW2N256, stream>>>(
              reinterpret_cast<const half*>(ws_inter.data_ptr()),
              reinterpret_cast<const uint32_t*>(w2_weight.data_ptr()),
              w2_scale_codes.data_ptr<uint8_t>(),
              w2_global_scales.data_ptr<float>(),
              ws_sorted_token_idx.data_ptr<int32_t>(),
              ws_sorted_weights.data_ptr<float>(), tiles_w2_ptr,
              ws_num_tiles.data_ptr<int32_t>(), ws_accum_f32.data_ptr(), M);
    } else if (w2_mode == 2) {
      slice_cols = 256;
      use_f16_accum = true;
      fn_nvfp4_w2_m64_n256_atomic_kernel<true>
          <<<dim3(max_tiles, 10), 512, kSmemW2N256, stream>>>(
              reinterpret_cast<const half*>(ws_inter.data_ptr()),
              reinterpret_cast<const uint32_t*>(w2_weight.data_ptr()),
              w2_scale_codes.data_ptr<uint8_t>(),
              w2_global_scales.data_ptr<float>(),
              ws_sorted_token_idx.data_ptr<int32_t>(),
              ws_sorted_weights.data_ptr<float>(), tiles_w2_ptr,
              ws_num_tiles.data_ptr<int32_t>(), ws_accum_f32.data_ptr(), M);
    } else if (w2_mode == 3) {
      use_f16_accum = true;
      fn_nvfp4_w2_m64_n128_f16x2_atomic_kernel<<<dim3(max_tiles, 20), 256, 0,
                                                 stream>>>(
          reinterpret_cast<const half*>(ws_inter.data_ptr()),
          reinterpret_cast<const uint32_t*>(w2_weight.data_ptr()),
          w2_scale_codes.data_ptr<uint8_t>(),
          w2_global_scales.data_ptr<float>(),
          ws_sorted_token_idx.data_ptr<int32_t>(),
          ws_sorted_weights.data_ptr<float>(), tiles_w2_ptr,
          ws_num_tiles.data_ptr<int32_t>(),
          reinterpret_cast<half*>(ws_accum_f32.data_ptr()), M);
    } else if (w2_mode == 4) {
      slice_cols = 160;
      fn_nvfp4_w2_m64_n160_atomic_kernel<<<dim3(max_tiles, 16), 320, 0,
                                           stream>>>(
          reinterpret_cast<const half*>(ws_inter.data_ptr()),
          reinterpret_cast<const uint32_t*>(w2_weight.data_ptr()),
          w2_scale_codes.data_ptr<uint8_t>(),
          w2_global_scales.data_ptr<float>(),
          ws_sorted_token_idx.data_ptr<int32_t>(),
          ws_sorted_weights.data_ptr<float>(), tiles_w2_ptr,
          ws_num_tiles.data_ptr<int32_t>(), ws_accum_f32.data_ptr<float>(), M);
    } else {
      eff_m_stride = m_stride_pad;
      fn_nvfp4_w2_m64_n128_atomic_kernel<<<dim3(max_tiles, 20), 256, 0,
                                           stream>>>(
          reinterpret_cast<const half*>(ws_inter.data_ptr()),
          reinterpret_cast<const uint32_t*>(w2_weight.data_ptr()),
          w2_scale_codes.data_ptr<uint8_t>(),
          w2_global_scales.data_ptr<float>(),
          ws_sorted_token_idx.data_ptr<int32_t>(),
          ws_sorted_weights.data_ptr<float>(), tiles_w2_ptr,
          ws_num_tiles.data_ptr<int32_t>(), ws_accum_f32.data_ptr<float>(),
          eff_m_stride);
    }
  } else {
    fn_nvfp4_w13_swiglu_pipelined_kernel<<<max_tiles, 320, 0, stream>>>(
        reinterpret_cast<const half*>(x.data_ptr()),
        reinterpret_cast<const uint32_t*>(w13_weight.data_ptr()),
        w13_scale_codes.data_ptr<uint8_t>(),
        w13_global_scales.data_ptr<float>(),
        ws_sorted_token_idx.data_ptr<int32_t>(),
        reinterpret_cast<const MoETileDesc*>(ws_tiles.data_ptr()),
        ws_num_tiles.data_ptr<int32_t>(),
        reinterpret_cast<half*>(ws_inter.data_ptr()));

    fn_nvfp4_w2_m32_n128_atomic_kernel<<<dim3(max_tiles, 20), 256, 0, stream>>>(
        reinterpret_cast<const half*>(ws_inter.data_ptr()),
        reinterpret_cast<const uint32_t*>(w2_weight.data_ptr()),
        w2_scale_codes.data_ptr<uint8_t>(), w2_global_scales.data_ptr<float>(),
        ws_sorted_token_idx.data_ptr<int32_t>(),
        ws_sorted_weights.data_ptr<float>(),
        reinterpret_cast<const MoETileDesc*>(ws_tiles.data_ptr()),
        ws_num_tiles.data_ptr<int32_t>(), ws_accum_f32.data_ptr<float>(), M);
  }

  const int out_rows = static_cast<int>(out.size(0));
  const bool scale_0125 = force_scale_0125 || (out_rows > M) ||
                          (out.stride(0) == 2560 && out_rows < 0);
  const int launch_m = std::max(M, out_rows);
  PeerPtrs4 sync_ptrs{{static_cast<uintptr_t>(s0), static_cast<uintptr_t>(s1),
                       static_cast<uintptr_t>(s2), static_cast<uintptr_t>(s3)}};
  unsigned* ep_ptr =
      (ep_state.has_value() && ep_state->defined())
          ? reinterpret_cast<unsigned*>(ep_state->data_ptr<int32_t>())
          : nullptr;
  const int rk = static_cast<int>(my_rank);
  if (use_f16_accum) {
    const int fin_ctas = std::min(scale_0125 ? launch_m : M, 320);
    if (slice_cols == 256) {
      if (scale_0125) {
        fn_moe_finalize_shared_f16_kernel<256, true>
            <<<fin_ctas, 320, 0, stream>>>(
                reinterpret_cast<half*>(ws_accum_f32.data_ptr()), shared_ptr,
                gate_ptr, reinterpret_cast<half*>(out.data_ptr()), M, out_rows,
                0, sync_ptrs, ep_ptr, rk);
      } else {
        fn_moe_finalize_shared_f16_kernel<256, false>
            <<<fin_ctas, 320, 0, stream>>>(
                reinterpret_cast<half*>(ws_accum_f32.data_ptr()), shared_ptr,
                gate_ptr, reinterpret_cast<half*>(out.data_ptr()), M, M, 0,
                sync_ptrs, ep_ptr, rk);
      }
    } else {
      if (scale_0125) {
        fn_moe_finalize_shared_f16_kernel<128, true>
            <<<fin_ctas, 320, 0, stream>>>(
                reinterpret_cast<half*>(ws_accum_f32.data_ptr()), shared_ptr,
                gate_ptr, reinterpret_cast<half*>(out.data_ptr()), M, out_rows,
                eff_m_stride, sync_ptrs, ep_ptr, rk);
      } else {
        fn_moe_finalize_shared_f16_kernel<128, false>
            <<<fin_ctas, 320, 0, stream>>>(
                reinterpret_cast<half*>(ws_accum_f32.data_ptr()), shared_ptr,
                gate_ptr, reinterpret_cast<half*>(out.data_ptr()), M, M,
                eff_m_stride, sync_ptrs, ep_ptr, rk);
      }
    }
  } else if (slice_cols == 256) {
    const int fin_ctas = std::min(scale_0125 ? launch_m : M, 320);
    if (scale_0125) {
      fn_moe_finalize_shared_kernel<256, true><<<fin_ctas, 320, 0, stream>>>(
          ws_accum_f32.data_ptr<float>(), shared_ptr, gate_ptr,
          reinterpret_cast<half*>(out.data_ptr()), M, out_rows, 0, sync_ptrs,
          ep_ptr, rk);
    } else {
      fn_moe_finalize_shared_kernel<256, false><<<fin_ctas, 320, 0, stream>>>(
          ws_accum_f32.data_ptr<float>(), shared_ptr, gate_ptr,
          reinterpret_cast<half*>(out.data_ptr()), M, M, 0, sync_ptrs, ep_ptr,
          rk);
    }
  } else if (slice_cols == 160) {
    const int fin_ctas = std::min(scale_0125 ? launch_m : M, 320);
    if (scale_0125) {
      fn_moe_finalize_shared_kernel<160, true><<<fin_ctas, 320, 0, stream>>>(
          ws_accum_f32.data_ptr<float>(), shared_ptr, gate_ptr,
          reinterpret_cast<half*>(out.data_ptr()), M, out_rows, 0, sync_ptrs,
          ep_ptr, rk);
    } else {
      fn_moe_finalize_shared_kernel<160, false><<<fin_ctas, 320, 0, stream>>>(
          ws_accum_f32.data_ptr<float>(), shared_ptr, gate_ptr,
          reinterpret_cast<half*>(out.data_ptr()), M, M, 0, sync_ptrs, ep_ptr,
          rk);
    }
  } else if (slice_cols == 64) {
    const int fin_ctas = std::min(scale_0125 ? launch_m : M, 320);
    if (scale_0125) {
      fn_moe_finalize_shared_kernel<64, true><<<fin_ctas, 320, 0, stream>>>(
          ws_accum_f32.data_ptr<float>(), shared_ptr, gate_ptr,
          reinterpret_cast<half*>(out.data_ptr()), M, out_rows, 0, sync_ptrs,
          ep_ptr, rk);
    } else {
      fn_moe_finalize_shared_kernel<64, false><<<fin_ctas, 320, 0, stream>>>(
          ws_accum_f32.data_ptr<float>(), shared_ptr, gate_ptr,
          reinterpret_cast<half*>(out.data_ptr()), M, M, 0, sync_ptrs, ep_ptr,
          rk);
    }
  } else {
    const int fin_ctas = std::min(scale_0125 ? launch_m : M, 320);
    if (scale_0125) {
      fn_moe_finalize_shared_kernel<128, true><<<fin_ctas, 320, 0, stream>>>(
          ws_accum_f32.data_ptr<float>(), shared_ptr, gate_ptr,
          reinterpret_cast<half*>(out.data_ptr()), M, out_rows, eff_m_stride,
          sync_ptrs, ep_ptr, rk);
    } else {
      fn_moe_finalize_shared_kernel<128, false><<<fin_ctas, 320, 0, stream>>>(
          ws_accum_f32.data_ptr<float>(), shared_ptr, gate_ptr,
          reinterpret_cast<half*>(out.data_ptr()), M, M, eff_m_stride,
          sync_ptrs, ep_ptr, rk);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ============================================================================
// Part 4: Zero-Spill Native H=12 QSA Staged INT8/E4M3 Attention
// ============================================================================

namespace qsa_h12 {

constexpr int D = 256;
constexpr int H6 = 6;
constexpr int H12 = 12;
constexpr int C = 64;
constexpr int PAD = 264;
constexpr int RKP = 256;
constexpr int SP = C + 4;

__device__ __forceinline__ int tile_swz_col(int token, int col) {
  return col ^ ((token & 3) << 5) ^ (((token >> 2) & 1) << 4);
}

__device__ __forceinline__ uint32_t e4m3x2_half(uint32_t word, uint32_t sel) {
  const uint32_t x = __byte_perm(word, 0u, sel);
  return ((x << 8) & 0x80008000u) | ((x << 7) & 0x3f803f80u);
}

__device__ __forceinline__ uint4 decode8_unscaled(const uint2 packed) {
  return make_uint4(
      e4m3x2_half(packed.x, 0x4140u), e4m3x2_half(packed.x, 0x4342u),
      e4m3x2_half(packed.y, 0x4140u), e4m3x2_half(packed.y, 0x4342u));
}

__device__ __forceinline__ uint32_t int8x2_half(uint32_t word, uint32_t sel) {
  const uint32_t biased = __byte_perm(word, 0x64646464u, sel) ^ 0x00800080u;
  const uint32_t bias_bits = 0x64806480u;
  const half2 h2 = __hsub2(*reinterpret_cast<const half2*>(&biased),
                           *reinterpret_cast<const half2*>(&bias_bits));
  return *reinterpret_cast<const uint32_t*>(&h2);
}

__device__ __forceinline__ uint4 decode8_int8_unscaled(const uint2 packed) {
  return make_uint4(
      int8x2_half(packed.x, 0x4140u), int8x2_half(packed.x, 0x4342u),
      int8x2_half(packed.y, 0x4140u), int8x2_half(packed.y, 0x4342u));
}

struct H12Smem {
  alignas(16) half query[2][8][PAD];       //  8,448 B
  alignas(16) unsigned char tile[C][RKP];  // 16,384 B
  alignas(16) float scores[4][H12][SP];    // 13,056 B
  alignas(16) uint2 sp_f16[2][8][C / 4];   //  2,048 B
  int64_t slots[3][C];                     //  1,536 B
  alignas(16) float kscale[3][C];          //    768 B
  alignas(16) float vscale[3][C];          //    768 B
  float alpha[H12];                        //     48 B
  float denom[H12];                        //     48 B
};  // Total = 43,104 B (<= 48 KiB -> 2 CTAs/SM!)

template <bool IS_INT8>
__global__ __launch_bounds__(256, 2) void rows_fp8_h12_kernel(
    const half* __restrict__ q, int64_t qs0, int64_t qs1,
    const unsigned char* __restrict__ history, const float* __restrict__ scales,
    const int32_t* __restrict__ indices, int64_t is0,
    const int32_t* __restrict__ table, int64_t ts0,
    const int32_t* __restrict__ requests, const int64_t* __restrict__ positions,
    const int32_t* __restrict__ lengths, float* __restrict__ out_f32,
    half* __restrict__ out_f16, int64_t os0, int64_t os1,
    const half* __restrict__ gate, int64_t gs0, int64_t gs1,
    float* __restrict__ lse, int64_t ls0, int blocks, int page, int width,
    int table_width, int num_requests) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  H12Smem& s = *reinterpret_cast<H12Smem*>(smem_raw);

  const int row = blockIdx.x;
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
  const int req = requests[row];
  const int64_t position = positions[row];
  const bool live = req >= 0 && req < num_requests;
  const int length = live ? lengths[req] : 0;
  const int ntiles = (width + C - 1) / C;

  const int hlane = lane & 15;
  const int c4 = hlane * 4;
  {
    const int head = warp, d8 = lane * 8;
    const uint4 qv0 = head < H6 ? __ldg(reinterpret_cast<const uint4*>(
                                      q + row * qs0 + head * qs1 + d8))
                                : make_uint4(0, 0, 0, 0);
    const uint4 qv1 = head < H6 ? __ldg(reinterpret_cast<const uint4*>(
                                      q + row * qs0 + (H6 + head) * qs1 + d8))
                                : make_uint4(0, 0, 0, 0);
    *reinterpret_cast<uint4*>(&s.query[0][head][d8]) = qv0;
    *reinterpret_cast<uint4*>(&s.query[1][head][d8]) = qv1;
    if (warp >= 6) {
      s.sp_f16[warp - 6][6 + (lane >> 4)][hlane] = make_uint2(0, 0);
    }
  }

  int logical = -1;
  int64_t found = -1;
  float found_k = 0.f, found_v = 0.f;
  auto stage_index = [&](int j) {
    const int col = j * C + t;
    logical =
        (t < C && j < ntiles && col < width) ? indices[row * is0 + col] : -1;
  };
  auto stage_table = [&]() {
    found = -1;
    if (live && logical >= 0 && logical <= position && logical < length &&
        logical / page < table_width) {
      const int physical = table[req * ts0 + logical / page];
      if (physical >= 0 && physical < blocks)
        found = int64_t(physical) * page + logical % page;
    }
  };
  auto stage_scales = [&]() {
    constexpr float kDecodeFactor = IS_INT8 ? 1.f : 256.f;
    found_k = found >= 0 ? __ldg(scales + found * 2) * kDecodeFactor : 0.f;
    found_v = found >= 0 ? __ldg(scales + found * 2 + 1) * kDecodeFactor : 0.f;
  };
  auto stage_store = [&](int b) {
    if (t < C) {
      s.slots[b][t] = found;
      s.kscale[b][t] = found_k;
      s.vscale[b][t] = found_v;
    }
    return t < C && found >= 0;
  };

  constexpr int kChunks16 = C * 16 / 256;  // 4 uint4 (16B) loads per thread!
  uint4 raw[kChunks16];
  auto load = [&](int b, int kv) {
#pragma unroll
    for (int i = 0; i < kChunks16; ++i) {
      const int v = t + i * 256, token = v >> 4, dim = (v & 15) * 16;
      const int64_t slot = s.slots[b][token];
      raw[i] = make_uint4(0, 0, 0, 0);
      if (slot >= 0) {
        const int blk = slot / page, off = slot % page;
        raw[i] = __ldg(reinterpret_cast<const uint4*>(
            history + (int64_t(blk) * 2 * page + off) * D + dim +
            (kv ? int64_t(page) * D : 0)));
      }
    }
  };
  auto store = [&]() {
#pragma unroll
    for (int i = 0; i < kChunks16; ++i) {
      const int v = t + i * 256, token = v >> 4, dim = (v & 15) * 16;
      *reinterpret_cast<uint4*>(&s.tile[token][tile_swz_col(token, dim)]) =
          raw[i];
    }
  };

  stage_index(0);
  stage_table();
  stage_scales();
  int count0 = __syncthreads_count(stage_store(0));
  stage_index(1);
  stage_table();
  stage_scales();
  int count1 = __syncthreads_count(stage_store(1));
  if (count0) {
    load(0, 0);
  }

  const int nt = warp & 1, kw = warp >> 1;
  const int r8 = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int nr =
      nt * 32 + ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
  const int pv_base = warp * 32 + ((lane >> 2) & 3) * 8;
  auto pv_head = [&](int e) {
    return (e & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
  };
  auto pv_dim = [&](int e) {
    return pv_base + ((e & 1) | (((lane >> 1) & 1) << 1) | ((e >> 2) << 2));
  };

  float rmax_h = -1e20f;
  float rsum_h = 0.f;
  float pv0[8] = {};
  float pv1[8] = {};

  for (int j = 0; j < ntiles; ++j) {
    const int b = j % 3, b1 = (j + 1) % 3, b2 = (j + 2) % 3;
    const bool current = count0 > 0;
    if (!current && count1 == 0) break;
    const int tile_w = min(C, width - j * C);
    stage_index(j + 2);
    if (current) {
      store();
      load(b, 1);
    }
    __syncthreads();

    if (current && (nt == 0 || tile_w > 32)) {
      float acc0[8] = {};
      float acc1[8] = {};
#pragma unroll
      for (int sub = 0; sub < 4; ++sub) {
        const int k = kw * 64 + sub * 16;
        const uint4 raw16 =
            *reinterpret_cast<const uint4*>(&s.tile[nr][tile_swz_col(nr, k)]);
        const uint4 k_lo =
            IS_INT8 ? decode8_int8_unscaled(make_uint2(raw16.x, raw16.y))
                    : decode8_unscaled(make_uint2(raw16.x, raw16.y));
        const uint4 k_hi =
            IS_INT8 ? decode8_int8_unscaled(make_uint2(raw16.z, raw16.w))
                    : decode8_unscaled(make_uint2(raw16.z, raw16.w));
        const uint4 q0_0 = *reinterpret_cast<const uint4*>(&s.query[0][r8][k]);
        const uint4 q0_1 =
            *reinterpret_cast<const uint4*>(&s.query[0][r8][k + 8]);
        FN_MMA_884_RC(acc0, q0_0.x, q0_0.y, k_lo.x, k_lo.y);
        FN_MMA_884_RC(acc0, q0_0.z, q0_0.w, k_lo.z, k_lo.w);
        FN_MMA_884_RC(acc0, q0_1.x, q0_1.y, k_hi.x, k_hi.y);
        FN_MMA_884_RC(acc0, q0_1.z, q0_1.w, k_hi.z, k_hi.w);

        const uint4 q1_0 = *reinterpret_cast<const uint4*>(&s.query[1][r8][k]);
        const uint4 q1_1 =
            *reinterpret_cast<const uint4*>(&s.query[1][r8][k + 8]);
        FN_MMA_884_RC(acc1, q1_0.x, q1_0.y, k_lo.x, k_lo.y);
        FN_MMA_884_RC(acc1, q1_0.z, q1_0.w, k_lo.z, k_lo.w);
        FN_MMA_884_RC(acc1, q1_1.x, q1_1.y, k_hi.x, k_hi.y);
        FN_MMA_884_RC(acc1, q1_1.z, q1_1.w, k_hi.z, k_hi.w);
      }
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int h = (e & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
        if (h < H6) {
          const int col =
              nt * 32 + ((lane >> 2) & 3) * 8 +
              ((e & 1) | (((lane >> 1) & 1) << 1) | ((e >> 2) << 2));
          s.scores[kw][h][col] = acc0[e];
          s.scores[kw][H6 + h][col] = acc1[e];
        }
      }
    }
    stage_table();
    __syncthreads();

    // Concurrently store V(b) to s.tile, prefetch K(b1), and run Softmax ->
    // sp_f16!
    if (current) {
      store();
      if (warp < 6) {
        const int h = warp * 2 + (lane >> 4);
        float s0 = -1e20f, s1 = -1e20f, s2 = -1e20f, s3 = -1e20f;
        bool val0 = false, val1 = false, val2 = false, val3 = false;
        float4 vs = make_float4(0.f, 0.f, 0.f, 0.f);
        if (c4 < tile_w) {
          val0 = (c4 + 0 < tile_w) && (s.slots[b][c4 + 0] >= 0);
          val1 = (c4 + 1 < tile_w) && (s.slots[b][c4 + 1] >= 0);
          val2 = (c4 + 2 < tile_w) && (s.slots[b][c4 + 2] >= 0);
          val3 = (c4 + 3 < tile_w) && (s.slots[b][c4 + 3] >= 0);
          const float4 ks = *reinterpret_cast<const float4*>(&s.kscale[b][c4]);
          vs = *reinterpret_cast<const float4*>(&s.vscale[b][c4]);
          const float4 sc0 =
              *reinterpret_cast<const float4*>(&s.scores[0][h][c4]);
          const float4 sc1 =
              *reinterpret_cast<const float4*>(&s.scores[1][h][c4]);
          const float4 sc2 =
              *reinterpret_cast<const float4*>(&s.scores[2][h][c4]);
          const float4 sc3 =
              *reinterpret_cast<const float4*>(&s.scores[3][h][c4]);
          constexpr float kLog2Scale = 0.0625f * 1.4426950408889634f;
          if (val0)
            s0 = (((sc0.x + sc1.x) + sc2.x) + sc3.x) * (ks.x * kLog2Scale);
          if (val1)
            s1 = (((sc0.y + sc1.y) + sc2.y) + sc3.y) * (ks.y * kLog2Scale);
          if (val2)
            s2 = (((sc0.z + sc1.z) + sc2.z) + sc3.z) * (ks.z * kLog2Scale);
          if (val3)
            s3 = (((sc0.w + sc1.w) + sc2.w) + sc3.w) * (ks.w * kLog2Scale);
        }
        float maximum = fmaxf(fmaxf(s0, s1), fmaxf(s2, s3));
#pragma unroll
        for (int sh = 8; sh > 0; sh >>= 1)
          maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, sh));
        maximum = fmaxf(maximum, rmax_h);
        const float p0 = val0 ? fast_exp2(s0 - maximum) : 0.f;
        const float p1 = val1 ? fast_exp2(s1 - maximum) : 0.f;
        const float p2 = val2 ? fast_exp2(s2 - maximum) : 0.f;
        const float p3 = val3 ? fast_exp2(s3 - maximum) : 0.f;
        const half2 p_lo = __floats2half2_rn(p0 * vs.x, p1 * vs.y);
        const half2 p_hi = __floats2half2_rn(p2 * vs.z, p3 * vs.w);
        const int hg = (warp >= 3) ? 1 : 0;
        const int hl = (warp >= 3 ? (warp - 3) : warp) * 2 + (lane >> 4);
        s.sp_f16[hg][hl][hlane] =
            make_uint2(*reinterpret_cast<const uint32_t*>(&p_lo),
                       *reinterpret_cast<const uint32_t*>(&p_hi));
        float sum = (p0 + p1) + (p2 + p3);
#pragma unroll
        for (int sh = 8; sh > 0; sh >>= 1)
          sum += __shfl_xor_sync(0xffffffffu, sum, sh);
        const float scale = fast_exp2(rmax_h - maximum);
        rsum_h = rsum_h * scale + sum;
        rmax_h = maximum;
        if (hlane == 0) s.alpha[h] = scale;
      }
    }
    if (count1) load(b1, 0);
    stage_scales();
    __syncthreads();

    if (current) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int ph = pv_head(e);
        if (ph < H6) {
          pv0[e] *= s.alpha[ph];
          pv1[e] *= s.alpha[H6 + ph];
        }
      }
      const int vk = lane & 3, vn = pv_base + ((lane & 16) ? 4 : 0);
#pragma unroll 4
      for (int k = 0; k < tile_w; k += 4) {
        const int tok_v = k + vk;
        const int kh4 = k >> 2;
        const uint32_t codes = *reinterpret_cast<const uint32_t*>(
            &s.tile[tok_v][tile_swz_col(tok_v, vn)]);
        const uint32_t v01 =
            IS_INT8 ? int8x2_half(codes, 0x4140u) : e4m3x2_half(codes, 0x4140u);
        const uint32_t v23 =
            IS_INT8 ? int8x2_half(codes, 0x4342u) : e4m3x2_half(codes, 0x4342u);

        const uint2 p0_u2 = s.sp_f16[0][r8][kh4];
        FN_MMA_884_RR(pv0, p0_u2.x, p0_u2.y, v01, v23);

        const uint2 p1_u2 = s.sp_f16[1][r8][kh4];
        FN_MMA_884_RR(pv1, p1_u2.x, p1_u2.y, v01, v23);
      }
    }

    const int count2 = __syncthreads_count(stage_store(b2));
    count0 = count1;
    count1 = count2;
  }

  if (warp < 6 && hlane == 0) {
    const int h = warp * 2 + (lane >> 4);
    s.denom[h] = rsum_h;
    if (lse != nullptr) {
      lse[row * ls0 + h] =
          rsum_h > 0.f ? (rmax_h + log2f(rsum_h)) : -CUDART_INF_F;
    }
  }
  __syncthreads();

  auto emit = [&](int grp, int h6, int dim, float sum) {
    const int h = grp * H6 + h6;
    const float den = s.denom[h];
    float val = den > 0.f ? __fdiv_rn(sum, den) : 0.f;
    if (gate != nullptr) {
      val = __half2float(__float2half_rn(val));
      const float g = __half2float(gate[row * gs0 + h * gs1 + dim]);
      val *= __fdiv_rn(1.f, 1.f + fast_exp2(-g * 1.4426950408889634f));
    }
    if (out_f32 != nullptr) {
      out_f32[row * os0 + h * os1 + dim] = val;
    } else {
      out_f16[row * os0 + h * os1 + dim] = __float2half_rn(val);
    }
  };

#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const int ph = pv_head(e);
    if (ph < H6) {
      const int dm = pv_dim(e);
      emit(0, ph, dm, pv0[e]);
      emit(1, ph, dm, pv1[e]);
    }
  }
}

struct QuadH12Smem {
  alignas(16) half query[4][H12][PAD];     // 25,344 B
  alignas(16) unsigned char tile[C][RKP];  // 16,384 B
  alignas(16) float scores[4][H12][SP];    // 13,056 B
  int64_t slots[2][C];                     //  1,024 B
  alignas(16) float kscale[2][C];          //    512 B
  alignas(16) float vscale[2][C];          //    512 B
  uint8_t qmask[2][C];                     //    128 B
  int32_t in_idx[4][C];                    //  1,024 B
  int32_t u01_idx[C];                      //    256 B
  int32_t u23_idx[C];                      //    256 B
  int32_t u_out_idx[C];                    //    256 B
  uint8_t u01_m[C];                        //     64 B
  uint8_t u23_m[C];                        //     64 B
  float alpha[4][H12];                     //    192 B
  float denom[4][H12];                     //    192 B
  int32_t cur[4];                          //     16 B
  int32_t req[4];                          //     16 B
  int64_t pos[4];                          //     32 B
  int32_t len[4];                          //     16 B
  int32_t live[4];                         //     16 B
  int32_t tile_cnt[2];                     //      8 B
  int32_t last_val;                        //      4 B
  int32_t active_q;                        //      4 B
  int32_t same_req;                        //      4 B
};

template <bool IS_INT8>
__global__ __launch_bounds__(512, 1) void rows_fp8_h12_quad_kernel(
    const half* __restrict__ q, int64_t qs0, int64_t qs1,
    const unsigned char* __restrict__ history, const float* __restrict__ scales,
    const int32_t* __restrict__ indices, int64_t is0,
    const int32_t* __restrict__ table, int64_t ts0,
    const int32_t* __restrict__ requests, const int64_t* __restrict__ positions,
    const int32_t* __restrict__ lengths, float* __restrict__ out_f32,
    half* __restrict__ out_f16, int64_t os0, int64_t os1,
    const half* __restrict__ gate, int64_t gs0, int64_t gs1,
    float* __restrict__ lse, int64_t ls0, int total_rows, int blocks, int page,
    int width, int table_width, int num_requests) {
  extern __shared__ __align__(16) unsigned char smem_quad_raw[];
  QuadH12Smem& s = *reinterpret_cast<QuadH12Smem*>(smem_quad_raw);

  const int r_base = blockIdx.x * 4;
  const int q_cnt = min(4, total_rows - r_base);
  if (q_cnt <= 0) return;

  const int t = threadIdx.x;  // 0..511
  const int warp = t >> 5;    // 0..15
  const int lane = t & 31;    // 0..31

  // Load 4 * 12 = 48 query heads x 32 uint4s = 1536 uint4s (3 per thread)
#pragma unroll
  for (int u = 0; u < 3; ++u) {
    const int vid = t + u * 512;  // 0..1535
    const int qh = vid >> 5;      // 0..47
    const int d8 = (vid & 31) * 8;
    const int qi = qh / H12;
    const int hi = qh - qi * H12;
    uint4 qv = make_uint4(0, 0, 0, 0);
    if (qi < q_cnt) {
      qv = __ldg(reinterpret_cast<const uint4*>(
          q + static_cast<int64_t>(r_base + qi) * qs0 + hi * qs1 + d8));
    }
    *reinterpret_cast<uint4*>(&s.query[qi][hi][d8]) = qv;
  }

  if (t < 4) {
    s.cur[t] = 0;
    if (t < q_cnt) {
      const int rq = requests[r_base + t];
      const bool lv = (rq >= 0 && rq < num_requests);
      s.req[t] = rq;
      s.pos[t] = positions[r_base + t];
      s.len[t] = lv ? lengths[rq] : 0;
      s.live[t] = lv ? 1 : 0;
    } else {
      s.req[t] = -1;
      s.pos[t] = -1;
      s.len[t] = 0;
      s.live[t] = 0;
    }
  }
  __syncthreads();
  if (t == 0) {
    s.active_q = 0;
    bool sr = (s.live[0] != 0);
    for (int qi = 1; qi < q_cnt; ++qi) {
      if (!s.live[qi] || s.req[qi] != s.req[0]) sr = false;
    }
    s.same_req = sr ? 1 : 0;
  }
  __syncthreads();

  constexpr float kDecodeFactor = IS_INT8 ? 1.f : 256.f;

  auto build_union_tile = [&](int buf) -> int {
    if (s.same_req) {
      if (t < 256) {
        const int qi = t >> 6;
        const int off = t & 63;
        int32_t val = 0x7fffffff;
        if (qi < q_cnt) {
          const int col = s.cur[qi] + off;
          if (col < width) {
            const int32_t raw_i =
                indices[static_cast<int64_t>(r_base + qi) * is0 + col];
            if (raw_i >= 0 && raw_i <= s.pos[qi] && raw_i < s.len[0]) {
              val = raw_i;
            }
          }
        }
        s.in_idx[qi][off] = val;
      }
      __syncthreads();

      if (t == 0) {
        int i0 = 0, i1 = 0, out_k = 0;
        while (out_k < C) {
          const int32_t v0 = (i0 < C) ? s.in_idx[0][i0] : 0x7fffffff;
          const int32_t v1 = (i1 < C) ? s.in_idx[1][i1] : 0x7fffffff;
          const int32_t mn = min(v0, v1);
          if (mn == 0x7fffffff) break;
          uint8_t m = 0;
          if (v0 == mn) {
            m |= 1u;
            ++i0;
          }
          if (v1 == mn) {
            m |= 2u;
            ++i1;
          }
          s.u01_idx[out_k] = mn;
          s.u01_m[out_k] = m;
          ++out_k;
        }
        for (; out_k < C; ++out_k) {
          s.u01_idx[out_k] = 0x7fffffff;
          s.u01_m[out_k] = 0;
        }
      } else if (t == 32) {
        int i2 = 0, i3 = 0, out_k = 0;
        while (out_k < C) {
          const int32_t v2 = (i2 < C) ? s.in_idx[2][i2] : 0x7fffffff;
          const int32_t v3 = (i3 < C) ? s.in_idx[3][i3] : 0x7fffffff;
          const int32_t mn = min(v2, v3);
          if (mn == 0x7fffffff) break;
          uint8_t m = 0;
          if (v2 == mn) {
            m |= 4u;
            ++i2;
          }
          if (v3 == mn) {
            m |= 8u;
            ++i3;
          }
          s.u23_idx[out_k] = mn;
          s.u23_m[out_k] = m;
          ++out_k;
        }
        for (; out_k < C; ++out_k) {
          s.u23_idx[out_k] = 0x7fffffff;
          s.u23_m[out_k] = 0;
        }
      }
      __syncthreads();

      if (t == 0) {
        int ia = 0, ib = 0, cnt = 0;
        while (cnt < C) {
          const int32_t va = (ia < C) ? s.u01_idx[ia] : 0x7fffffff;
          const int32_t vb = (ib < C) ? s.u23_idx[ib] : 0x7fffffff;
          const int32_t mn = min(va, vb);
          if (mn == 0x7fffffff) break;
          uint8_t m = 0;
          if (va == mn) {
            m |= s.u01_m[ia];
            ++ia;
          }
          if (vb == mn) {
            m |= s.u23_m[ib];
            ++ib;
          }
          s.u_out_idx[cnt] = mn;
          s.qmask[buf][cnt] = m;
          ++cnt;
        }
        s.tile_cnt[buf] = cnt;
        s.last_val = (cnt > 0) ? s.u_out_idx[cnt - 1] : -1;
        for (int k = cnt; k < C; ++k) {
          s.u_out_idx[k] = -1;
          s.qmask[buf][k] = 0;
        }
      }
      __syncthreads();

      const int cnt = s.tile_cnt[buf];
      if (cnt > 0) {
        if (warp < 4) {
          const int qi = warp;
          const int32_t lv = s.last_val;
          const bool b0 = (s.in_idx[qi][lane] <= lv);
          const bool b1 = (s.in_idx[qi][lane + 32] <= lv);
          const int adv = __popc(__ballot_sync(0xffffffffu, b0)) +
                          __popc(__ballot_sync(0xffffffffu, b1));
          if (lane == 0) s.cur[qi] += adv;
        }
        if (t < C) {
          const int32_t logical = (t < cnt) ? s.u_out_idx[t] : -1;
          int64_t found = -1;
          float fk = 0.f, fv = 0.f;
          if (logical >= 0 && logical / page < table_width) {
            const int phys = table[s.req[0] * ts0 + logical / page];
            if (phys >= 0 && phys < blocks) {
              found = int64_t(phys) * page + logical % page;
              fk = __ldg(scales + found * 2) * kDecodeFactor;
              fv = __ldg(scales + found * 2 + 1) * kDecodeFactor;
            }
          }
          if (found < 0) s.qmask[buf][t] = 0;
          s.slots[buf][t] = found;
          s.kscale[buf][t] = fk;
          s.vscale[buf][t] = fv;
        }
      }
      __syncthreads();
      return s.tile_cnt[buf];
    } else {
      if (t == 0) {
        int aq = s.active_q;
        while (aq < q_cnt && (!s.live[aq] || s.cur[aq] >= width)) ++aq;
        s.active_q = aq;
        s.tile_cnt[buf] = (aq < q_cnt) ? min(C, width - s.cur[aq]) : 0;
      }
      __syncthreads();
      const int cnt = s.tile_cnt[buf];
      const int aq = s.active_q;
      if (cnt > 0 && t < C) {
        const int col = s.cur[aq] + t;
        int64_t found = -1;
        float fk = 0.f, fv = 0.f;
        uint8_t m = 0;
        if (t < cnt && col < width) {
          const int32_t logical =
              indices[static_cast<int64_t>(r_base + aq) * is0 + col];
          if (logical >= 0 && logical <= s.pos[aq] && logical < s.len[aq] &&
              logical / page < table_width) {
            const int phys = table[s.req[aq] * ts0 + logical / page];
            if (phys >= 0 && phys < blocks) {
              found = int64_t(phys) * page + logical % page;
              fk = __ldg(scales + found * 2) * kDecodeFactor;
              fv = __ldg(scales + found * 2 + 1) * kDecodeFactor;
              m = static_cast<uint8_t>(1u << aq);
            }
          }
        }
        s.slots[buf][t] = found;
        s.kscale[buf][t] = fk;
        s.vscale[buf][t] = fv;
        s.qmask[buf][t] = m;
      }
      __syncthreads();
      if (cnt > 0 && t == 0) {
        s.cur[aq] += cnt;
      }
      __syncthreads();
      return cnt;
    }
  };

  constexpr int kChunks = C * 32 / 512;  // 4 uint2 loads per thread!
  uint2 raw[kChunks];
  auto load = [&](int b, int kv) {
#pragma unroll
    for (int i = 0; i < kChunks; ++i) {
      const int v = t + i * 512, token = v >> 5, dim = (v & 31) * 8;
      const int64_t slot = s.slots[b][token];
      raw[i] = make_uint2(0, 0);
      if (slot >= 0) {
        const int blk = slot / page, off = slot % page;
        raw[i] = __ldg(reinterpret_cast<const uint2*>(
            history + (int64_t(blk) * 2 * page + off) * D + dim +
            (kv ? int64_t(page) * D : 0)));
      }
    }
  };
  auto store = [&]() {
#pragma unroll
    for (int i = 0; i < kChunks; ++i) {
      const int v = t + i * 512, token = v >> 5, dim = (v & 31) * 8;
      *reinterpret_cast<uint2*>(&s.tile[token][tile_swz_col(token, dim)]) =
          raw[i];
    }
  };

  int count0 = build_union_tile(0);
  if (count0 > 0) load(0, 0);
  int count1 = (count0 > 0) ? build_union_tile(1) : 0;

  // QK warp roles: qi = warp >> 2 (0..3), nt = (warp >> 1) & 1, kh = warp & 1
  const int qk_qi = warp >> 2;
  const int qk_nt = (warp >> 1) & 1;
  const int qk_kh = warp & 1;
  const int r8 = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int nr = qk_nt * 32 + ((lane >> 2) & 3) * 8 + r8;

  // Softmax half-warp roles: 32 half-warps cover 48 heads in 2 passes
  const int hw = warp * 2 + (lane >> 4);
  const int hlane = lane & 15;
  const int c4 = hlane * 4;
  float rmax_h[2] = {-1e20f, -1e20f};
  float rsum_h[2] = {0.f, 0.f};

  // PV warp roles: q_grp = warp >> 3 (0 -> q0,q1; 1 -> q2,q3), dw = warp & 7
  const int q_grp = warp >> 3;
  const int dw = warp & 7;
  const int pv_base = dw * 32 + ((lane >> 2) & 3) * 8;
  auto pv_head = [&](int e) {
    return (e & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
  };
  auto pv_dim = [&](int e) {
    return pv_base + ((e & 1) | (((lane >> 1) & 1) << 1) | ((e >> 2) << 2));
  };
  float pv00[8] = {}, pv01[8] = {}, pv10[8] = {}, pv11[8] = {};

  for (int j = 0; count0 > 0; ++j) {
    const int b = j & 1, b1 = b ^ 1;
    const int tile_w = count0;
    store();
    load(b, 1);
    __syncthreads();

    float acc0[8] = {};
    float acc1[8] = {};
    const bool qk_active = (qk_qi < q_cnt) && (qk_nt == 0 || tile_w > 32);
    if (qk_active) {
#pragma unroll
      for (int sub = 0; sub < 8; ++sub) {
        const int k = qk_kh * 128 + sub * 16;
        const uint2 raw_lo =
            *reinterpret_cast<const uint2*>(&s.tile[nr][tile_swz_col(nr, k)]);
        const uint2 raw_hi = *reinterpret_cast<const uint2*>(
            &s.tile[nr][tile_swz_col(nr, k + 8)]);
        const uint4 k_lo =
            IS_INT8 ? decode8_int8_unscaled(raw_lo) : decode8_unscaled(raw_lo);
        const uint4 k_hi =
            IS_INT8 ? decode8_int8_unscaled(raw_hi) : decode8_unscaled(raw_hi);
        const uint4 q0_0 =
            (r8 < H6) ? *reinterpret_cast<const uint4*>(&s.query[qk_qi][r8][k])
                      : make_uint4(0, 0, 0, 0);
        const uint4 q0_1 =
            (r8 < H6)
                ? *reinterpret_cast<const uint4*>(&s.query[qk_qi][r8][k + 8])
                : make_uint4(0, 0, 0, 0);
        FN_MMA_884_RC(acc0, q0_0.x, q0_0.y, k_lo.x, k_lo.y);
        FN_MMA_884_RC(acc0, q0_0.z, q0_0.w, k_lo.z, k_lo.w);
        FN_MMA_884_RC(acc0, q0_1.x, q0_1.y, k_hi.x, k_hi.y);
        FN_MMA_884_RC(acc0, q0_1.z, q0_1.w, k_hi.z, k_hi.w);

        const uint4 q1_0 =
            (r8 < H6)
                ? *reinterpret_cast<const uint4*>(&s.query[qk_qi][H6 + r8][k])
                : make_uint4(0, 0, 0, 0);
        const uint4 q1_1 = (r8 < H6) ? *reinterpret_cast<const uint4*>(
                                           &s.query[qk_qi][H6 + r8][k + 8])
                                     : make_uint4(0, 0, 0, 0);
        FN_MMA_884_RC(acc1, q1_0.x, q1_0.y, k_lo.x, k_lo.y);
        FN_MMA_884_RC(acc1, q1_0.z, q1_0.w, k_lo.z, k_lo.w);
        FN_MMA_884_RC(acc1, q1_1.x, q1_1.y, k_hi.x, k_hi.y);
        FN_MMA_884_RC(acc1, q1_1.z, q1_1.w, k_hi.z, k_hi.w);
      }
    }

    if (qk_active && qk_kh == 0) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int h = (e & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
        if (h < H6) {
          const int col =
              qk_nt * 32 + ((lane >> 2) & 3) * 8 +
              ((e & 1) | (((lane >> 1) & 1) << 1) | ((e >> 2) << 2));
          s.scores[qk_qi][h][col] = acc0[e];
          s.scores[qk_qi][H6 + h][col] = acc1[e];
        }
      }
    }
    __syncthreads();
    if (qk_active && qk_kh == 1) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int h = (e & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
        if (h < H6) {
          const int col =
              qk_nt * 32 + ((lane >> 2) & 3) * 8 +
              ((e & 1) | (((lane >> 1) & 1) << 1) | ((e >> 2) << 2));
          s.scores[qk_qi][h][col] += acc0[e];
          s.scores[qk_qi][H6 + h][col] += acc1[e];
        }
      }
    }
    __syncthreads();

    // Single-pass float4 online softmax across 48 heads (2 passes of 32
    // half-warps)
#pragma unroll
    for (int hp = 0; hp < 2; ++hp) {
      const int h_tot = hw + hp * 32;
      if (h_tot < q_cnt * H12) {
        const int qi = h_tot / H12;
        const int hi = h_tot - qi * H12;
        const uint8_t qbit = static_cast<uint8_t>(1u << qi);
        float s0 = -1e20f, s1 = -1e20f, s2 = -1e20f, s3 = -1e20f;
        bool val0 = false, val1 = false, val2 = false, val3 = false;
        float4 vs = make_float4(0.f, 0.f, 0.f, 0.f);
        if (c4 < tile_w) {
          const uint32_t m4 =
              *reinterpret_cast<const uint32_t*>(&s.qmask[b][c4]);
          val0 = (c4 + 0 < tile_w) && (((m4 >> 0) & qbit) != 0);
          val1 = (c4 + 1 < tile_w) && (((m4 >> 8) & qbit) != 0);
          val2 = (c4 + 2 < tile_w) && (((m4 >> 16) & qbit) != 0);
          val3 = (c4 + 3 < tile_w) && (((m4 >> 24) & qbit) != 0);
          const float4 ks = *reinterpret_cast<const float4*>(&s.kscale[b][c4]);
          vs = *reinterpret_cast<const float4*>(&s.vscale[b][c4]);
          const float4 sc =
              *reinterpret_cast<const float4*>(&s.scores[qi][hi][c4]);
          constexpr float kLog2Scale = 0.0625f * 1.4426950408889634f;
          if (val0) s0 = sc.x * (ks.x * kLog2Scale);
          if (val1) s1 = sc.y * (ks.y * kLog2Scale);
          if (val2) s2 = sc.z * (ks.z * kLog2Scale);
          if (val3) s3 = sc.w * (ks.w * kLog2Scale);
        }
        float maximum = fmaxf(fmaxf(s0, s1), fmaxf(s2, s3));
#pragma unroll
        for (int sh = 8; sh > 0; sh >>= 1)
          maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, sh));
        maximum = fmaxf(maximum, rmax_h[hp]);
        const float p0 = val0 ? fast_exp2(s0 - maximum) : 0.f;
        const float p1 = val1 ? fast_exp2(s1 - maximum) : 0.f;
        const float p2 = val2 ? fast_exp2(s2 - maximum) : 0.f;
        const float p3 = val3 ? fast_exp2(s3 - maximum) : 0.f;
        *reinterpret_cast<float4*>(&s.scores[qi][hi][c4]) =
            make_float4(p0 * vs.x, p1 * vs.y, p2 * vs.z, p3 * vs.w);
        float sum = (p0 + p1) + (p2 + p3);
#pragma unroll
        for (int sh = 8; sh > 0; sh >>= 1)
          sum += __shfl_xor_sync(0xffffffffu, sum, sh);
        const float scale = fast_exp2(rmax_h[hp] - maximum);
        rsum_h[hp] = rsum_h[hp] * scale + sum;
        rmax_h[hp] = maximum;
        if (hlane == 0) s.alpha[qi][hi] = scale;
      }
    }
    __syncthreads();

    store();
    if (count1 > 0) load(b1, 0);
    __syncthreads();

    const int qi0 = q_grp * 2;
    const int qi1 = qi0 + 1;
    const bool do_q0 = (qi0 < q_cnt);
    const bool do_q1 = (qi1 < q_cnt);
    if (do_q0) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int ph = pv_head(e);
        if (ph < H6) {
          pv00[e] *= s.alpha[qi0][ph];
          pv01[e] *= s.alpha[qi0][H6 + ph];
        }
      }
    }
    if (do_q1) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int ph = pv_head(e);
        if (ph < H6) {
          pv10[e] *= s.alpha[qi1][ph];
          pv11[e] *= s.alpha[qi1][H6 + ph];
        }
      }
    }

    const int vk = lane & 3, vn = pv_base + ((lane & 16) ? 4 : 0);
    for (int k = 0; k < tile_w; k += 4) {
      const int tok_v = k + vk;
      const uint32_t codes = (tok_v < tile_w)
                                 ? *reinterpret_cast<const uint32_t*>(
                                       &s.tile[tok_v][tile_swz_col(tok_v, vn)])
                                 : 0u;
      const uint32_t v01 =
          IS_INT8 ? int8x2_half(codes, 0x4140u) : e4m3x2_half(codes, 0x4140u);
      const uint32_t v23 =
          IS_INT8 ? int8x2_half(codes, 0x4342u) : e4m3x2_half(codes, 0x4342u);

      if (do_q0) {
        const float4 p_g0 =
            (r8 < H6) ? *reinterpret_cast<const float4*>(&s.scores[qi0][r8][k])
                      : make_float4(0.f, 0.f, 0.f, 0.f);
        const half2 p0_01 = __floats2half2_rn(p_g0.x, p_g0.y);
        const half2 p0_23 = __floats2half2_rn(p_g0.z, p_g0.w);
        FN_MMA_884_RR(pv00, *reinterpret_cast<const uint32_t*>(&p0_01),
                      *reinterpret_cast<const uint32_t*>(&p0_23), v01, v23);

        const float4 p_g1 =
            (r8 < H6)
                ? *reinterpret_cast<const float4*>(&s.scores[qi0][H6 + r8][k])
                : make_float4(0.f, 0.f, 0.f, 0.f);
        const half2 p1_01 = __floats2half2_rn(p_g1.x, p_g1.y);
        const half2 p1_23 = __floats2half2_rn(p_g1.z, p_g1.w);
        FN_MMA_884_RR(pv01, *reinterpret_cast<const uint32_t*>(&p1_01),
                      *reinterpret_cast<const uint32_t*>(&p1_23), v01, v23);
      }
      if (do_q1) {
        const float4 p_g0 =
            (r8 < H6) ? *reinterpret_cast<const float4*>(&s.scores[qi1][r8][k])
                      : make_float4(0.f, 0.f, 0.f, 0.f);
        const half2 p0_01 = __floats2half2_rn(p_g0.x, p_g0.y);
        const half2 p0_23 = __floats2half2_rn(p_g0.z, p_g0.w);
        FN_MMA_884_RR(pv10, *reinterpret_cast<const uint32_t*>(&p0_01),
                      *reinterpret_cast<const uint32_t*>(&p0_23), v01, v23);

        const float4 p_g1 =
            (r8 < H6)
                ? *reinterpret_cast<const float4*>(&s.scores[qi1][H6 + r8][k])
                : make_float4(0.f, 0.f, 0.f, 0.f);
        const half2 p1_01 = __floats2half2_rn(p_g1.x, p_g1.y);
        const half2 p1_23 = __floats2half2_rn(p_g1.z, p_g1.w);
        FN_MMA_884_RR(pv11, *reinterpret_cast<const uint32_t*>(&p1_01),
                      *reinterpret_cast<const uint32_t*>(&p1_23), v01, v23);
      }
    }
    __syncthreads();

    const int count2 = (count1 > 0) ? build_union_tile(b) : 0;
    count0 = count1;
    count1 = count2;
  }

#pragma unroll
  for (int hp = 0; hp < 2; ++hp) {
    const int h_tot = hw + hp * 32;
    if (h_tot < q_cnt * H12 && hlane == 0) {
      const int qi = h_tot / H12;
      const int hi = h_tot - qi * H12;
      s.denom[qi][hi] = rsum_h[hp];
      if (lse != nullptr) {
        lse[static_cast<int64_t>(r_base + qi) * ls0 + hi] =
            rsum_h[hp] > 0.f ? (rmax_h[hp] + log2f(rsum_h[hp])) : -CUDART_INF_F;
      }
    }
  }
  __syncthreads();

  auto emit_q = [&](int qi, int grp, int h6, int dim, float sum) {
    const int row = r_base + qi;
    const int h = grp * H6 + h6;
    const float den = s.denom[qi][h];
    float val = den > 0.f ? __fdiv_rn(sum, den) : 0.f;
    if (gate != nullptr) {
      val = __half2float(__float2half_rn(val));
      const float g =
          __half2float(gate[static_cast<int64_t>(row) * gs0 + h * gs1 + dim]);
      val *= __fdiv_rn(1.f, 1.f + fast_exp2(-g * 1.4426950408889634f));
    }
    if (out_f32 != nullptr) {
      out_f32[static_cast<int64_t>(row) * os0 + h * os1 + dim] = val;
    } else {
      out_f16[static_cast<int64_t>(row) * os0 + h * os1 + dim] =
          __float2half_rn(val);
    }
  };

  const int qi0 = q_grp * 2;
  const int qi1 = qi0 + 1;
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const int ph = pv_head(e);
    if (ph < H6) {
      const int dm = pv_dim(e);
      if (qi0 < q_cnt) {
        emit_q(qi0, 0, ph, dm, pv00[e]);
        emit_q(qi0, 1, ph, dm, pv01[e]);
      }
      if (qi1 < q_cnt) {
        emit_q(qi1, 0, ph, dm, pv10[e]);
        emit_q(qi1, 1, ph, dm, pv11[e]);
      }
    }
  }
}

// Ultra-lightweight 4-warp/CTA (100% SM occupancy, zero d=32 binary search!)
// MergePath pre-merger:
__global__ __launch_bounds__(128, 16) void fn_qsa_pair_merge_indices_kernel(
    const int32_t* __restrict__ indices, int64_t is0,
    const int32_t* __restrict__ requests, const int64_t* __restrict__ positions,
    const int32_t* __restrict__ lengths, int32_t* __restrict__ merged_indices,
    uint8_t* __restrict__ merged_qmask, int32_t* __restrict__ pair_ntiles,
    int M, int width, int merge_stride, int num_requests, int num_pairs) {
  __shared__ int32_t s_a[4][32];
  __shared__ int32_t s_b[4][32];
  __shared__ int32_t s_mval[4][33];
  __shared__ uint8_t s_msrc[4][33];

  const int w = threadIdx.x >> 5;     // 0..3
  const int lane = threadIdx.x & 31;  // 0..31
  const int pair_id = blockIdx.x * 4 + w;
  if (pair_id >= num_pairs) return;

  const int r0 = pair_id * 2;
  const int r1 = r0 + 1;
  const bool has_r1 = (r1 < M);

  const int req0 = __ldg(requests + r0);
  const int64_t pos0 = __ldg(positions + r0);
  const bool live0 = (req0 >= 0 && req0 < num_requests);
  const int len0 = live0 ? __ldg(lengths + req0) : 0;

  const int req1 = has_r1 ? __ldg(requests + r1) : -1;
  const int64_t pos1 = has_r1 ? __ldg(positions + r1) : -1;
  const bool live1 = has_r1 && (req1 >= 0 && req1 < num_requests);
  const int len1 = live1 ? __ldg(lengths + req1) : 0;
  const bool same_req = has_r1 && live0 && live1 && (req0 == req1);

  int32_t* out_idx =
      merged_indices + static_cast<int64_t>(pair_id) * merge_stride;
  uint8_t* out_qm = merged_qmask + static_cast<int64_t>(pair_id) * merge_stride;
  int out_pos = 0;

  if (same_req) {
    const int32_t* row0 = indices + static_cast<int64_t>(r0) * is0;
    const int32_t* row1 = indices + static_cast<int64_t>(r1) * is0;
    int c0 = 0, c1 = 0;
    while (c0 < width || c1 < width) {
      int32_t va = 0x7fffffff;
      if (c0 + lane < width) {
        const int32_t v = __ldg(row0 + c0 + lane);
        if (v >= 0 && v <= pos0 && v < len0) va = v;
      }
      int32_t vb = 0x7fffffff;
      if (c1 + lane < width) {
        const int32_t v = __ldg(row1 + c1 + lane);
        if (v >= 0 && v <= pos1 && v < len1) vb = v;
      }
      s_a[w][lane] = va;
      s_b[w][lane] = vb;
      __syncwarp(0xffffffffu);
      if (s_a[w][0] == 0x7fffffff && s_b[w][0] == 0x7fffffff) break;

      int lo = 0;
      int hi = lane;
#pragma unroll
      for (int step = 0; step < 6; ++step) {
        const int mid = (lo + hi) >> 1;
        const int j_idx = lane - mid;
        if (mid > 0 && s_a[w][mid - 1] > s_b[w][j_idx]) {
          hi = mid - 1;
        } else if (j_idx > 0 && s_a[w][mid] <= s_b[w][j_idx - 1]) {
          lo = mid + 1;
        } else {
          lo = hi = mid;
        }
      }
      const int i_idx = lo;
      const int j_idx = lane - i_idx;
      const bool take_a = (s_a[w][i_idx] <= s_b[w][j_idx]);
      s_mval[w][lane] = take_a ? s_a[w][i_idx] : s_b[w][j_idx];
      s_msrc[w][lane] = take_a ? 1u : 2u;

      // Exact O(1) diagonal-32 partition via __popc(__ballot_sync(take_a)):
      const uint32_t bal_a = __ballot_sync(0xffffffffu, take_a);
      if (lane == 0) {
        const int i32 = __popc(bal_a);
        const int j32 = 32 - i32;
        const bool ta32 =
            (j32 >= 32) || (i32 < 32 && s_a[w][i32] <= s_b[w][j32]);
        s_mval[w][32] = ta32 ? s_a[w][i32] : s_b[w][j32];
        s_msrc[w][32] = ta32 ? 1u : 2u;
      }
      __syncwarp(0xffffffffu);

      const int32_t mv = s_mval[w][lane];
      const bool valid = (mv != 0x7fffffff);
      const bool is_u = valid && (lane == 0 || mv != s_mval[w][lane - 1]);
      const uint32_t u_bal = __ballot_sync(0xffffffffu, is_u);
      const int u_off = __popc(u_bal & ((1u << lane) - 1u));
      if (is_u) {
        uint8_t qm = s_msrc[w][lane];
        if (s_mval[w][lane + 1] == mv) qm |= s_msrc[w][lane + 1];
        out_idx[out_pos + u_off] = mv;
        out_qm[out_pos + u_off] = qm;
      }
      out_pos += __popc(u_bal);

      const uint8_t src = valid ? s_msrc[w][lane] : 0u;
      int adv_a = __popc(__ballot_sync(0xffffffffu, src == 1u));
      int adv_b = __popc(__ballot_sync(0xffffffffu, src == 2u));
      if (s_mval[w][31] != 0x7fffffff && s_mval[w][32] == s_mval[w][31]) {
        if (s_msrc[w][32] == 1u)
          ++adv_a;
        else
          ++adv_b;
      }
      if (__ballot_sync(0xffffffffu, !valid) != 0u) break;
      c0 += adv_a;
      c1 += adv_b;
      __syncwarp(0xffffffffu);
    }
  } else {
    if (live0) {
      const int32_t* row0 = indices + static_cast<int64_t>(r0) * is0;
      for (int c0 = 0; c0 < width; c0 += 32) {
        int32_t v = -1;
        if (c0 + lane < width) {
          const int32_t raw = __ldg(row0 + c0 + lane);
          if (raw >= 0 && raw <= pos0 && raw < len0) v = raw;
        }
        const bool ok = (v >= 0);
        const uint32_t bal = __ballot_sync(0xffffffffu, ok);
        if (bal == 0u) break;
        const int off = __popc(bal & ((1u << lane) - 1u));
        if (ok) {
          out_idx[out_pos + off] = v;
          out_qm[out_pos + off] = 1u;
        }
        out_pos += __popc(bal);
      }
    }
    if (live1) {
      const int32_t* row1 = indices + static_cast<int64_t>(r1) * is0;
      for (int c1 = 0; c1 < width; c1 += 32) {
        int32_t v = -1;
        if (c1 + lane < width) {
          const int32_t raw = __ldg(row1 + c1 + lane);
          if (raw >= 0 && raw <= pos1 && raw < len1) v = raw;
        }
        const bool ok = (v >= 0);
        const uint32_t bal = __ballot_sync(0xffffffffu, ok);
        if (bal == 0u) break;
        const int off = __popc(bal & ((1u << lane) - 1u));
        if (ok) {
          out_idx[out_pos + off] = v;
          out_qm[out_pos + off] = 2u;
        }
        out_pos += __popc(bal);
      }
    }
  }

  // Pad trailing elements of the final 64-key tile with -1 / 0
  const int ntiles = (out_pos + 63) >> 6;
  const int padded_end = ntiles * 64;
  for (int p = out_pos + lane; p < padded_end; p += 32) {
    out_idx[p] = -1;
    out_qm[p] = 0u;
  }
  if (lane == 0) {
    pair_ntiles[pair_id] = out_pos;
  }
}

// Stage-2 1-warp MergePath pre-merger: merges two pre-merged pairs (p0, p1)
// into a 4-row deduplicated key list (48 heads = 6x8, ~3.3x KV compression)!
__global__ __launch_bounds__(32, 16) void fn_qsa_merge_two_pairs_kernel(
    const int32_t* __restrict__ pair_idx, const uint8_t* __restrict__ pair_qm,
    const int32_t* __restrict__ pair_counts, int pair_stride,
    const int32_t* __restrict__ requests, int32_t* __restrict__ quad_idx,
    uint8_t* __restrict__ quad_qm, int32_t* __restrict__ quad_counts,
    int quad_stride, int M, int num_pairs, int num_requests) {
  __shared__ int32_t s_a[32];
  __shared__ int32_t s_b[32];
  __shared__ uint8_t s_qa[32];
  __shared__ uint8_t s_qb[32];
  __shared__ int32_t s_mval[33];
  __shared__ uint8_t s_msrc[33];
  __shared__ uint8_t s_mqm[33];

  const int quad_id = blockIdx.x;
  const int p0 = quad_id * 2;
  const int p1 = p0 + 1;
  const bool has_p1 = (p1 < num_pairs);
  const int lane = threadIdx.x;

  const int r_base = quad_id * 4;
  int ref_req = -1;
  bool all_same_req = true;
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const int r = r_base + q;
    if (r < M) {
      const int rq = requests[r];
      if (rq >= 0 && rq < num_requests) {
        if (ref_req < 0)
          ref_req = rq;
        else if (rq != ref_req)
          all_same_req = false;
      }
    }
  }

  const int len_a = (p0 < num_pairs) ? pair_counts[p0] : 0;
  const int len_b = has_p1 ? pair_counts[p1] : 0;
  const int32_t* in_a = pair_idx + static_cast<int64_t>(p0) * pair_stride;
  const uint8_t* qa_in = pair_qm + static_cast<int64_t>(p0) * pair_stride;
  const int32_t* in_b =
      has_p1 ? (pair_idx + static_cast<int64_t>(p1) * pair_stride) : nullptr;
  const uint8_t* qb_in =
      has_p1 ? (pair_qm + static_cast<int64_t>(p1) * pair_stride) : nullptr;

  int32_t* out_idx = quad_idx + static_cast<int64_t>(quad_id) * quad_stride;
  uint8_t* out_qm = quad_qm + static_cast<int64_t>(quad_id) * quad_stride;
  int out_pos = 0;

  if (all_same_req && len_a > 0 && len_b > 0) {
    int c0 = 0, c1 = 0;
    while (c0 < len_a || c1 < len_b) {
      int32_t va = 0x7fffffff;
      uint8_t qa = 0u;
      if (c0 + lane < len_a) {
        va = in_a[c0 + lane];
        qa = qa_in[c0 + lane];
      }
      int32_t vb = 0x7fffffff;
      uint8_t qb = 0u;
      if (c1 + lane < len_b) {
        vb = in_b[c1 + lane];
        qb = static_cast<uint8_t>(qb_in[c1 + lane] << 2);
      }
      s_a[lane] = va;
      s_b[lane] = vb;
      s_qa[lane] = qa;
      s_qb[lane] = qb;
      __syncwarp(0xffffffffu);
      if (s_a[0] == 0x7fffffff && s_b[0] == 0x7fffffff) break;

      auto merge_at = [&](int d) {
        int lo = max(0, d - 32);
        int hi = min(d, 32);
#pragma unroll
        for (int step = 0; step < 6; ++step) {
          const int mid = (lo + hi) >> 1;
          const int j_idx = d - mid;
          if (mid > 0 && j_idx < 32 && s_a[mid - 1] > s_b[j_idx]) {
            hi = mid - 1;
          } else if (j_idx > 0 && mid < 32 && s_a[mid] <= s_b[j_idx - 1]) {
            lo = mid + 1;
          } else {
            lo = hi = mid;
          }
        }
        const int i_idx = lo;
        const int j_idx = d - i_idx;
        const bool take_a =
            (j_idx >= 32) || (i_idx < 32 && s_a[i_idx] <= s_b[j_idx]);
        s_mval[d] = take_a ? s_a[i_idx] : s_b[j_idx];
        s_msrc[d] = take_a ? 1u : 2u;
        s_mqm[d] = take_a ? s_qa[i_idx] : s_qb[j_idx];
      };
      merge_at(lane);
      if (lane == 0) merge_at(32);
      __syncwarp(0xffffffffu);

      const int32_t mv = s_mval[lane];
      const bool valid = (mv != 0x7fffffff);
      const bool is_u = valid && (lane == 0 || mv != s_mval[lane - 1]);
      const uint32_t u_bal = __ballot_sync(0xffffffffu, is_u);
      const int u_off = __popc(u_bal & ((1u << lane) - 1u));
      if (is_u) {
        uint8_t qm = s_mqm[lane];
        if (s_mval[lane + 1] == mv) qm |= s_mqm[lane + 1];
        out_idx[out_pos + u_off] = mv;
        out_qm[out_pos + u_off] = qm;
      }
      out_pos += __popc(u_bal);

      const uint8_t src = valid ? s_msrc[lane] : 0u;
      int adv_a = __popc(__ballot_sync(0xffffffffu, src == 1u));
      int adv_b = __popc(__ballot_sync(0xffffffffu, src == 2u));
      if (s_mval[31] != 0x7fffffff && s_mval[32] == s_mval[31]) {
        if (s_msrc[32] == 1u)
          ++adv_a;
        else
          ++adv_b;
      }
      if (__ballot_sync(0xffffffffu, !valid) != 0u) break;
      c0 += adv_a;
      c1 += adv_b;
      __syncwarp(0xffffffffu);
    }
  } else {
    for (int i = lane; i < len_a; i += 32) {
      out_idx[i] = in_a[i];
      out_qm[i] = qa_in[i];
    }
    for (int i = lane; i < len_b; i += 32) {
      out_idx[len_a + i] = in_b[i];
      out_qm[len_a + i] = static_cast<uint8_t>(qb_in[i] << 2);
    }
    out_pos = len_a + len_b;
  }

  const int ntiles = (out_pos + 63) >> 6;
  const int padded_end = ntiles * 64;
  for (int p = out_pos + lane; p < padded_end; p += 32) {
    out_idx[p] = -1;
    out_qm[p] = 0u;
  }
  if (lane == 0) {
    quad_counts[quad_id] = out_pos;
  }
}

struct PairFast384Smem {
  alignas(16) half query[3][8][PAD];  // 12,672 B (24 heads = 3 x 8, 0 waste!)
  alignas(16) unsigned char tile[C][RKP];  // 16,384 B
  alignas(16) float scores[2][24][SP];     // 13,056 B
  alignas(16) uint2 sp_f16[3][8][C / 4];   //  3,072 B
  int32_t slots[3][C];                     //    768 B (full 3-stage pipeline!)
  alignas(16) float kscale[3][C];          //    768 B
  alignas(16) float vscale[3][C];          //    768 B
  uint8_t qmask[3][C];                     //    192 B
  float alpha[24];                         //     96 B
  float denom[24];                         //     96 B
};  // Total = 47,872 B <= 49,152 B -> 2 CTAs/SM (768 threads = 24 warps/SM, 0
    // spills!)

template <bool IS_INT8>
__global__ __launch_bounds__(384, 2) void rows_fp8_h12_pair_fast_kernel(
    const half* __restrict__ q, int64_t qs0, int64_t qs1,
    const unsigned char* __restrict__ history, const float* __restrict__ scales,
    const int32_t* __restrict__ merged_indices,
    const uint8_t* __restrict__ merged_qmask,
    const int32_t* __restrict__ pair_counts, int merge_stride,
    const int32_t* __restrict__ table, int64_t ts0,
    const int32_t* __restrict__ requests, float* __restrict__ out_f32,
    half* __restrict__ out_f16, int64_t os0, int64_t os1,
    const half* __restrict__ gate, int64_t gs0, int64_t gs1,
    float* __restrict__ lse, int64_t ls0, int M, int blocks, int page,
    int table_width, int num_requests) {
  extern __shared__ __align__(16) unsigned char pair_fast_smem_raw[];
  PairFast384Smem& s = *reinterpret_cast<PairFast384Smem*>(pair_fast_smem_raw);

  const int pair_id = blockIdx.x;
  const int r0 = pair_id * 2;
  const int r1 = r0 + 1;
  const bool has_r1 = (r1 < M);
  const int t = threadIdx.x;
  const int warp = t >> 5;  // 0..11
  const int lane = t & 31;

  const int req0 = requests[r0];
  const bool live0 = (req0 >= 0 && req0 < num_requests);
  const int req1 = has_r1 ? requests[r1] : -1;
  const bool live1 = has_r1 && (req1 >= 0 && req1 < num_requests);
  const int total_u = pair_counts[pair_id];
  const int ntiles = (total_u + 63) >> 6;
  const int64_t pair_base = static_cast<int64_t>(pair_id) * merge_stride;

  {
    const int d8 = lane * 8;
#pragma unroll
    for (int sub_h = 0; sub_h < 2; ++sub_h) {
      const int h24 = warp * 2 + sub_h;
      const int hg = h24 >> 3;
      const int hl = h24 & 7;
      uint4 qv = make_uint4(0, 0, 0, 0);
      if (h24 < 12) {
        if (live0) {
          qv = __ldg(
              reinterpret_cast<const uint4*>(q + r0 * qs0 + h24 * qs1 + d8));
        }
      } else if (live1) {
        qv = __ldg(reinterpret_cast<const uint4*>(q + r1 * qs0 +
                                                  (h24 - 12) * qs1 + d8));
      }
      *reinterpret_cast<uint4*>(&s.query[hg][hl][d8]) = qv;
    }
  }

  int logical = -1;
  uint8_t qm_reg = 0;
  int32_t found = -1;
  float found_k = 0.f, found_v = 0.f;
  // Offload stage_* onto warps 8..9 (t in [256, 320)), which only issue 2 KV
  // uint4 loads instead of 3, eliminating warp 0..1 scoreboard stragglers!
  const int st_t = t - 256;
  const bool is_st = (static_cast<unsigned>(st_t) < static_cast<unsigned>(C));
  auto stage_index = [&](int j) {
    if (is_st && j < ntiles) {
      const int col = j * C + st_t;
      logical = __ldg(merged_indices + pair_base + col);
      qm_reg = __ldg(merged_qmask + pair_base + col);
    } else {
      logical = -1;
      qm_reg = 0;
    }
  };
  auto stage_table = [&]() {
    found = -1;
    if (logical >= 0 && logical / page < table_width) {
      const int use_req = (qm_reg & 1u) ? req0 : req1;
      const int physical =
          __ldg(table + static_cast<int64_t>(use_req) * ts0 + logical / page);
      if (physical >= 0 && physical < blocks) {
        found = physical * page + (logical % page);
      }
    }
  };
  auto stage_scales = [&]() {
    constexpr float kDecodeFactor = IS_INT8 ? 1.f : 256.f;
    if (found >= 0) {
      const float2 kv_s = __ldg(reinterpret_cast<const float2*>(
          scales + static_cast<int64_t>(found) * 2));
      found_k = kv_s.x * kDecodeFactor;
      found_v = kv_s.y * kDecodeFactor;
    } else {
      found_k = 0.f;
      found_v = 0.f;
    }
  };
  auto stage_store = [&](int b) {
    if (is_st) {
      s.slots[b][st_t] = found;
      s.kscale[b][st_t] = found_k;
      s.vscale[b][st_t] = found_v;
      s.qmask[b][st_t] = (found >= 0) ? qm_reg : 0u;
    }
    return is_st && found >= 0;
  };

  uint4 raw[3];
  auto load = [&](int b, int kv) {
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      const int v = t + i * 384;
      raw[i] = make_uint4(0, 0, 0, 0);
      if (v < 1024) {
        const int token = v >> 4, dim = (v & 15) * 16;
        const int32_t slot = s.slots[b][token];
        if (slot >= 0) {
          const int blk = slot / page, off = slot % page;
          raw[i] = __ldg(reinterpret_cast<const uint4*>(
              history + (int64_t(blk) * 2 * page + off) * D + dim +
              (kv ? int64_t(page) * D : 0)));
        }
      }
    }
  };
  auto store = [&]() {
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      const int v = t + i * 384;
      if (v < 1024) {
        const int token = v >> 4, dim = (v & 15) * 16;
        *reinterpret_cast<uint4*>(&s.tile[token][tile_swz_col(token, dim)]) =
            raw[i];
      }
    }
  };

  stage_index(0);
  stage_table();
  stage_scales();
  int count0 = __syncthreads_count(stage_store(0));
  stage_index(1);
  stage_table();
  stage_scales();
  int count1 = __syncthreads_count(stage_store(1));
  if (count0) load(0, 0);

  const int hg_qk = warp >> 2;
  const int kw_qk = (warp >> 1) & 1;
  const int nt_qk = warp & 1;
  const int r8 = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int nr =
      nt_qk * 32 + ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);

  const int h24_sm = warp * 2 + (lane >> 4);
  const uint8_t row_bit_sm = (h24_sm < 12) ? 1u : 2u;
  const int hlane = lane & 15;
  const int c4 = hlane * 4;

  const int hg_pv = warp >> 2;
  const int d_sub = warp & 3;
  const int pv_base0 = d_sub * 64 + ((lane >> 2) & 3) * 8;
  const int pv_base1 = pv_base0 + 32;
  auto pv_head = [&](int e) {
    return (e & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
  };
  auto pv_dim_off = [&](int e) {
    return (e & 1) | (((lane >> 1) & 1) << 1) | ((e >> 2) << 2);
  };

  float rmax_h = -1e20f;
  float rsum_h = 0.f;
  float pv0[8] = {};
  float pv1[8] = {};

  for (int j = 0; j < ntiles; ++j) {
    const int b = j % 3, b1 = (j + 1) % 3, b2 = (j + 2) % 3;
    const bool current = (count0 > 0);
    if (!current && count1 == 0) break;
    const int tile_w = min(C, total_u - j * C);
    stage_index(j + 2);
    if (current) {
      store();
      load(b, 1);
    }
    __syncthreads();

    if (current && (nt_qk == 0 || tile_w > 32)) {
      float acc[8] = {};
#pragma unroll 4
      for (int sub = 0; sub < 8; ++sub) {
        const int k = kw_qk * 128 + sub * 16;
        const uint4 raw16 =
            *reinterpret_cast<const uint4*>(&s.tile[nr][tile_swz_col(nr, k)]);
        const uint4 k_lo =
            IS_INT8 ? decode8_int8_unscaled(make_uint2(raw16.x, raw16.y))
                    : decode8_unscaled(make_uint2(raw16.x, raw16.y));
        const uint4 k_hi =
            IS_INT8 ? decode8_int8_unscaled(make_uint2(raw16.z, raw16.w))
                    : decode8_unscaled(make_uint2(raw16.z, raw16.w));
        const uint4 q_0 =
            *reinterpret_cast<const uint4*>(&s.query[hg_qk][r8][k]);
        const uint4 q_1 =
            *reinterpret_cast<const uint4*>(&s.query[hg_qk][r8][k + 8]);
        FN_MMA_884_RC(acc, q_0.x, q_0.y, k_lo.x, k_lo.y);
        FN_MMA_884_RC(acc, q_0.z, q_0.w, k_lo.z, k_lo.w);
        FN_MMA_884_RC(acc, q_1.x, q_1.y, k_hi.x, k_hi.y);
        FN_MMA_884_RC(acc, q_1.z, q_1.w, k_hi.z, k_hi.w);
      }
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int h = (e & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
        const int col = nt_qk * 32 + ((lane >> 2) & 3) * 8 +
                        ((e & 1) | (((lane >> 1) & 1) << 1) | ((e >> 2) << 2));
        s.scores[kw_qk][hg_qk * 8 + h][col] = acc[e];
      }
    }
    stage_table();
    __syncthreads();

    if (current) {
      float s0 = -1e20f, s1 = -1e20f, s2 = -1e20f, s3 = -1e20f;
      bool v0 = false, v1 = false, v2 = false, v3 = false;
      float4 vs = make_float4(0.f, 0.f, 0.f, 0.f);
      if (c4 < tile_w) {
        const uint32_t m4 = *reinterpret_cast<const uint32_t*>(&s.qmask[b][c4]);
        v0 = (((m4 >> 0) & row_bit_sm) != 0);
        v1 = (((m4 >> 8) & row_bit_sm) != 0);
        v2 = (((m4 >> 16) & row_bit_sm) != 0);
        v3 = (((m4 >> 24) & row_bit_sm) != 0);
        const float4 ks = *reinterpret_cast<const float4*>(&s.kscale[b][c4]);
        vs = *reinterpret_cast<const float4*>(&s.vscale[b][c4]);
        const float4 sc0 =
            *reinterpret_cast<const float4*>(&s.scores[0][h24_sm][c4]);
        const float4 sc1 =
            *reinterpret_cast<const float4*>(&s.scores[1][h24_sm][c4]);
        constexpr float kLog2Scale = 0.0625f * 1.4426950408889634f;
        if (v0) s0 = (sc0.x + sc1.x) * (ks.x * kLog2Scale);
        if (v1) s1 = (sc0.y + sc1.y) * (ks.y * kLog2Scale);
        if (v2) s2 = (sc0.z + sc1.z) * (ks.z * kLog2Scale);
        if (v3) s3 = (sc0.w + sc1.w) * (ks.w * kLog2Scale);
      }

      float maximum = fmaxf(fmaxf(s0, s1), fmaxf(s2, s3));
#pragma unroll
      for (int sh = 8; sh > 0; sh >>= 1)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, sh));
      maximum = fmaxf(maximum, rmax_h);
      const float p0 = v0 ? fast_exp2(s0 - maximum) : 0.f;
      const float p1 = v1 ? fast_exp2(s1 - maximum) : 0.f;
      const float p2 = v2 ? fast_exp2(s2 - maximum) : 0.f;
      const float p3 = v3 ? fast_exp2(s3 - maximum) : 0.f;
      const half2 p_lo = __floats2half2_rn(p0 * vs.x, p1 * vs.y);
      const half2 p_hi = __floats2half2_rn(p2 * vs.z, p3 * vs.w);
      s.sp_f16[h24_sm >> 3][h24_sm & 7][hlane] =
          make_uint2(*reinterpret_cast<const uint32_t*>(&p_lo),
                     *reinterpret_cast<const uint32_t*>(&p_hi));
      float sum = (p0 + p1) + (p2 + p3);
#pragma unroll
      for (int sh = 8; sh > 0; sh >>= 1)
        sum += __shfl_xor_sync(0xffffffffu, sum, sh);
      const float scale = fast_exp2(rmax_h - maximum);
      rsum_h = rsum_h * scale + sum;
      rmax_h = maximum;
      if (hlane == 0) s.alpha[h24_sm] = scale;
      // Store V(b) AFTER Softmax so V(b) L2/HBM loads hide under both QK AND
      // Softmax!
      store();
    }
    if (count1) load(b1, 0);
    stage_scales();
    __syncthreads();

    if (current) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float a_s = s.alpha[hg_pv * 8 + pv_head(e)];
        pv0[e] *= a_s;
        pv1[e] *= a_s;
      }
      const int vk = lane & 3;
      const int vn0 = pv_base0 + ((lane & 16) ? 4 : 0);
      const int vn1 = pv_base1 + ((lane & 16) ? 4 : 0);
      auto pv_step = [&](int k) {
        const int tok_v = k + vk;
        const int kh4 = k >> 2;
        const uint2 p_u2 = s.sp_f16[hg_pv][r8][kh4];

        const uint32_t c0 = *reinterpret_cast<const uint32_t*>(
            &s.tile[tok_v][tile_swz_col(tok_v, vn0)]);
        const uint32_t v0_01 =
            IS_INT8 ? int8x2_half(c0, 0x4140u) : e4m3x2_half(c0, 0x4140u);
        const uint32_t v0_23 =
            IS_INT8 ? int8x2_half(c0, 0x4342u) : e4m3x2_half(c0, 0x4342u);
        FN_MMA_884_RR(pv0, p_u2.x, p_u2.y, v0_01, v0_23);

        const uint32_t c1 = *reinterpret_cast<const uint32_t*>(
            &s.tile[tok_v][tile_swz_col(tok_v, vn1)]);
        const uint32_t v1_01 =
            IS_INT8 ? int8x2_half(c1, 0x4140u) : e4m3x2_half(c1, 0x4140u);
        const uint32_t v1_23 =
            IS_INT8 ? int8x2_half(c1, 0x4342u) : e4m3x2_half(c1, 0x4342u);
        FN_MMA_884_RR(pv1, p_u2.x, p_u2.y, v1_01, v1_23);
      };
      if (tile_w == C) {
#pragma unroll 8
        for (int k = 0; k < C; k += 4) pv_step(k);
      } else {
#pragma unroll 4
        for (int k = 0; k < tile_w; k += 4) pv_step(k);
      }
    }
    const int count2 = __syncthreads_count(stage_store(b2));
    count0 = count1;
    count1 = count2;
  }

  if (hlane == 0) {
    s.denom[h24_sm] = rsum_h;
    if (lse != nullptr && (h24_sm < 12 || has_r1)) {
      const int out_r = (h24_sm < 12) ? r0 : r1;
      const int out_h = (h24_sm < 12) ? h24_sm : (h24_sm - 12);
      lse[out_r * ls0 + out_h] =
          rsum_h > 0.f ? (rmax_h + log2f(rsum_h)) : -CUDART_INF_F;
    }
  }
  __syncthreads();

  const bool io_even =
      ((os0 & 1) == 0) && ((os1 & 1) == 0) &&
      (gate == nullptr || (((gs0 & 1) == 0) && ((gs1 & 1) == 0)));
  auto emit_pair = [&](int h24, int dim0, float sum0, float sum1) {
    if (h24 >= 12 && !has_r1) return;
    const int out_r = (h24 < 12) ? r0 : r1;
    const int out_h = (h24 < 12) ? h24 : (h24 - 12);
    const float den = s.denom[h24];
    float v0 = den > 0.f ? __fdiv_rn(sum0, den) : 0.f;
    float v1 = den > 0.f ? __fdiv_rn(sum1, den) : 0.f;
    if (gate != nullptr) {
      v0 = __half2float(__float2half_rn(v0));
      v1 = __half2float(__float2half_rn(v1));
      float g0, g1;
      if (io_even) {
        const half2 g2 = __ldg(reinterpret_cast<const half2*>(
            gate + out_r * gs0 + out_h * gs1 + dim0));
        g0 = __half2float(__low2half(g2));
        g1 = __half2float(__high2half(g2));
      } else {
        g0 = __half2float(gate[out_r * gs0 + out_h * gs1 + dim0]);
        g1 = __half2float(gate[out_r * gs0 + out_h * gs1 + dim0 + 1]);
      }
      v0 *= __fdiv_rn(1.f, 1.f + fast_exp2(-g0 * 1.4426950408889634f));
      v1 *= __fdiv_rn(1.f, 1.f + fast_exp2(-g1 * 1.4426950408889634f));
    }
    const int64_t o_off = out_r * os0 + out_h * os1 + dim0;
    if (out_f32 != nullptr) {
      if (io_even) {
        *reinterpret_cast<float2*>(out_f32 + o_off) = make_float2(v0, v1);
      } else {
        out_f32[o_off] = v0;
        out_f32[o_off + 1] = v1;
      }
    } else if (io_even) {
      *reinterpret_cast<half2*>(out_f16 + o_off) = __floats2half2_rn(v0, v1);
    } else {
      out_f16[o_off] = __float2half_rn(v0);
      out_f16[o_off + 1] = __float2half_rn(v1);
    }
  };

#pragma unroll
  for (int e = 0; e < 8; e += 2) {
    const int h24 = hg_pv * 8 + pv_head(e);
    const int doff = pv_dim_off(e);
    emit_pair(h24, pv_base0 + doff, pv0[e], pv0[e + 1]);
    emit_pair(h24, pv_base1 + doff, pv1[e], pv1[e + 1]);
  }
}

// 256-thread (8 warps x 2 CTAs/SM = 128 regs/thread!) variant:
// Dequantizes K and V ONLY ONCE across all 24 heads!
template <bool IS_INT8>
__global__ __launch_bounds__(256, 2) void rows_fp8_h12_pair_fast256_kernel(
    const half* __restrict__ q, int64_t qs0, int64_t qs1,
    const unsigned char* __restrict__ history, const float* __restrict__ scales,
    const int32_t* __restrict__ merged_indices,
    const uint8_t* __restrict__ merged_qmask,
    const int32_t* __restrict__ pair_ntiles, int merge_stride,
    const int32_t* __restrict__ table, int64_t ts0,
    const int32_t* __restrict__ requests, float* __restrict__ out_f32,
    half* __restrict__ out_f16, int64_t os0, int64_t os1,
    const half* __restrict__ gate, int64_t gs0, int64_t gs1,
    float* __restrict__ lse, int64_t ls0, int M, int blocks, int page,
    int table_width, int num_requests) {
  extern __shared__ __align__(16) unsigned char pair_fast256_smem_raw[];
  PairFast384Smem& s =
      *reinterpret_cast<PairFast384Smem*>(pair_fast256_smem_raw);

  const int pair_id = blockIdx.x;
  const int r0 = pair_id * 2;
  const int r1 = r0 + 1;
  const bool has_r1 = (r1 < M);
  const int t = threadIdx.x;
  const int warp = t >> 5;  // 0..7
  const int lane = t & 31;

  const int req0 = requests[r0];
  const bool live0 = (req0 >= 0 && req0 < num_requests);
  const int req1 = has_r1 ? requests[r1] : -1;
  const bool live1 = has_r1 && (req1 >= 0 && req1 < num_requests);
  const int total_u = pair_ntiles[pair_id];
  const int ntiles = (total_u + 63) >> 6;
  const int64_t pair_base = static_cast<int64_t>(pair_id) * merge_stride;

  {
    const int d8 = lane * 8;
#pragma unroll
    for (int hg = 0; hg < 3; ++hg) {
      const int h24 = hg * 8 + warp;
      uint4 qv = make_uint4(0, 0, 0, 0);
      if (h24 < 12) {
        if (live0) {
          qv = __ldg(
              reinterpret_cast<const uint4*>(q + r0 * qs0 + h24 * qs1 + d8));
        }
      } else if (live1) {
        qv = __ldg(reinterpret_cast<const uint4*>(q + r1 * qs0 +
                                                  (h24 - 12) * qs1 + d8));
      }
      *reinterpret_cast<uint4*>(&s.query[hg][warp][d8]) = qv;
    }
  }

  int logical = -1;
  uint8_t qm_reg = 0;
  int32_t found = -1;
  float found_k = 0.f, found_v = 0.f;
  auto stage_index = [&](int j) {
    const int col = j * C + t;
    if (t < C && j < ntiles) {
      logical = merged_indices[pair_base + col];
      qm_reg = merged_qmask[pair_base + col];
    } else {
      logical = -1;
      qm_reg = 0;
    }
  };
  auto stage_table = [&]() {
    found = -1;
    if (logical >= 0 && logical / page < table_width) {
      const int use_req = (qm_reg & 1u) ? req0 : req1;
      const int physical =
          table[static_cast<int64_t>(use_req) * ts0 + logical / page];
      if (physical >= 0 && physical < blocks) {
        found = physical * page + (logical % page);
      }
    }
  };
  auto stage_scales = [&]() {
    constexpr float kDecodeFactor = IS_INT8 ? 1.f : 256.f;
    found_k = found >= 0 ? __ldg(scales + static_cast<int64_t>(found) * 2) *
                               kDecodeFactor
                         : 0.f;
    found_v = found >= 0 ? __ldg(scales + static_cast<int64_t>(found) * 2 + 1) *
                               kDecodeFactor
                         : 0.f;
  };
  auto stage_store = [&](int b) {
    if (t < C) {
      s.slots[b][t] = found;
      s.kscale[b][t] = found_k;
      s.vscale[b][t] = found_v;
      s.qmask[b][t] = (found >= 0) ? qm_reg : 0u;
    }
    return t < C && found >= 0;
  };

  uint4 raw[4];
  auto load = [&](int b, int kv) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int v = t + i * 256, token = v >> 4, dim = (v & 15) * 16;
      const int32_t slot = s.slots[b][token];
      raw[i] = make_uint4(0, 0, 0, 0);
      if (slot >= 0) {
        const int blk = slot / page, off = slot % page;
        raw[i] = __ldg(reinterpret_cast<const uint4*>(
            history + (int64_t(blk) * 2 * page + off) * D + dim +
            (kv ? int64_t(page) * D : 0)));
      }
    }
  };
  auto store = [&]() {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int v = t + i * 256, token = v >> 4, dim = (v & 15) * 16;
      *reinterpret_cast<uint4*>(&s.tile[token][tile_swz_col(token, dim)]) =
          raw[i];
    }
  };

  stage_index(0);
  stage_table();
  stage_scales();
  int count0 = __syncthreads_count(stage_store(0));
  stage_index(1);
  stage_table();
  stage_scales();
  int count1 = __syncthreads_count(stage_store(1));
  if (count0) load(0, 0);

  const int nt = warp & 1, kw = warp >> 1;  // kw = 0..3, nt = 0..1
  const int r8 = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int nr =
      nt * 32 + ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
  const int pv_base = warp * 32 + ((lane >> 2) & 3) * 8;
  auto pv_head = [&](int e) {
    return (e & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
  };
  auto pv_dim = [&](int e) {
    return pv_base + ((e & 1) | (((lane >> 1) & 1) << 1) | ((e >> 2) << 2));
  };

  const int hw = warp * 2 + (lane >> 4);  // 0..15
  const int hlane = lane & 15;
  const int c4 = hlane * 4;
  float rmax_h0 = -1e20f, rmax_h1 = -1e20f;
  float rsum_h0 = 0.f, rsum_h1 = 0.f;
  float pv0[8] = {}, pv1[8] = {}, pv2[8] = {};

  for (int j = 0; j < ntiles; ++j) {
    const int b = j % 3, b1 = (j + 1) % 3, b2 = (j + 2) % 3;
    const bool current = (count0 > 0);
    if (!current && count1 == 0) break;
    stage_index(j + 2);
    if (current) {
      store();
      load(b, 1);
    }
    __syncthreads();

    float acc0[8] = {}, acc1[8] = {}, acc2[8] = {};
    if (current) {
#pragma unroll
      for (int sub = 0; sub < 4; ++sub) {
        const int k = kw * 64 + sub * 16;
        const uint4 raw16 =
            *reinterpret_cast<const uint4*>(&s.tile[nr][tile_swz_col(nr, k)]);
        const uint4 k_lo =
            IS_INT8 ? decode8_int8_unscaled(make_uint2(raw16.x, raw16.y))
                    : decode8_unscaled(make_uint2(raw16.x, raw16.y));
        const uint4 k_hi =
            IS_INT8 ? decode8_int8_unscaled(make_uint2(raw16.z, raw16.w))
                    : decode8_unscaled(make_uint2(raw16.z, raw16.w));
        const uint4 q0_0 = *reinterpret_cast<const uint4*>(&s.query[0][r8][k]);
        const uint4 q0_1 =
            *reinterpret_cast<const uint4*>(&s.query[0][r8][k + 8]);
        FN_MMA_884_RC(acc0, q0_0.x, q0_0.y, k_lo.x, k_lo.y);
        FN_MMA_884_RC(acc0, q0_0.z, q0_0.w, k_lo.z, k_lo.w);
        FN_MMA_884_RC(acc0, q0_1.x, q0_1.y, k_hi.x, k_hi.y);
        FN_MMA_884_RC(acc0, q0_1.z, q0_1.w, k_hi.z, k_hi.w);

        const uint4 q1_0 = *reinterpret_cast<const uint4*>(&s.query[1][r8][k]);
        const uint4 q1_1 =
            *reinterpret_cast<const uint4*>(&s.query[1][r8][k + 8]);
        FN_MMA_884_RC(acc1, q1_0.x, q1_0.y, k_lo.x, k_lo.y);
        FN_MMA_884_RC(acc1, q1_0.z, q1_0.w, k_lo.z, k_lo.w);
        FN_MMA_884_RC(acc1, q1_1.x, q1_1.y, k_hi.x, k_hi.y);
        FN_MMA_884_RC(acc1, q1_1.z, q1_1.w, k_hi.z, k_hi.w);

        const uint4 q2_0 = *reinterpret_cast<const uint4*>(&s.query[2][r8][k]);
        const uint4 q2_1 =
            *reinterpret_cast<const uint4*>(&s.query[2][r8][k + 8]);
        FN_MMA_884_RC(acc2, q2_0.x, q2_0.y, k_lo.x, k_lo.y);
        FN_MMA_884_RC(acc2, q2_0.z, q2_0.w, k_lo.z, k_lo.w);
        FN_MMA_884_RC(acc2, q2_1.x, q2_1.y, k_hi.x, k_hi.y);
        FN_MMA_884_RC(acc2, q2_1.z, q2_1.w, k_hi.z, k_hi.w);
      }
      if (kw < 2) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int h = (e & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
          const int col =
              nt * 32 + ((lane >> 2) & 3) * 8 +
              ((e & 1) | (((lane >> 1) & 1) << 1) | ((e >> 2) << 2));
          s.scores[kw][h][col] = acc0[e];
          s.scores[kw][8 + h][col] = acc1[e];
          s.scores[kw][16 + h][col] = acc2[e];
        }
      }
    }
    __syncthreads();
    if (current && kw >= 2) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int h = (e & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
        const int col = nt * 32 + ((lane >> 2) & 3) * 8 +
                        ((e & 1) | (((lane >> 1) & 1) << 1) | ((e >> 2) << 2));
        s.scores[kw - 2][h][col] += acc0[e];
        s.scores[kw - 2][8 + h][col] += acc1[e];
        s.scores[kw - 2][16 + h][col] += acc2[e];
      }
    }
    stage_table();
    __syncthreads();

    if (current) {
      store();
      const uint32_t m4 = *reinterpret_cast<const uint32_t*>(&s.qmask[b][c4]);
      const float4 ks = *reinterpret_cast<const float4*>(&s.kscale[b][c4]);
      const float4 vs = *reinterpret_cast<const float4*>(&s.vscale[b][c4]);
      auto run_sm = [&](int h24, float& rmax, float& rsum) {
        const uint8_t rbit = (h24 < 12) ? 1u : 2u;
        const bool v0 = (((m4 >> 0) & rbit) != 0);
        const bool v1 = (((m4 >> 8) & rbit) != 0);
        const bool v2 = (((m4 >> 16) & rbit) != 0);
        const bool v3 = (((m4 >> 24) & rbit) != 0);
        const float4 sc0 =
            *reinterpret_cast<const float4*>(&s.scores[0][h24][c4]);
        const float4 sc1 =
            *reinterpret_cast<const float4*>(&s.scores[1][h24][c4]);
        constexpr float kLog2Scale = 0.0625f * 1.4426950408889634f;
        const float s0 = v0 ? (sc0.x + sc1.x) * (ks.x * kLog2Scale) : -1e20f;
        const float s1 = v1 ? (sc0.y + sc1.y) * (ks.y * kLog2Scale) : -1e20f;
        const float s2 = v2 ? (sc0.z + sc1.z) * (ks.z * kLog2Scale) : -1e20f;
        const float s3 = v3 ? (sc0.w + sc1.w) * (ks.w * kLog2Scale) : -1e20f;
        float maximum = fmaxf(fmaxf(s0, s1), fmaxf(s2, s3));
#pragma unroll
        for (int sh = 8; sh > 0; sh >>= 1)
          maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, sh));
        maximum = fmaxf(maximum, rmax);
        const float p0 = v0 ? fast_exp2(s0 - maximum) : 0.f;
        const float p1 = v1 ? fast_exp2(s1 - maximum) : 0.f;
        const float p2 = v2 ? fast_exp2(s2 - maximum) : 0.f;
        const float p3 = v3 ? fast_exp2(s3 - maximum) : 0.f;
        const half2 p_lo = __floats2half2_rn(p0 * vs.x, p1 * vs.y);
        const half2 p_hi = __floats2half2_rn(p2 * vs.z, p3 * vs.w);
        s.sp_f16[h24 >> 3][h24 & 7][hlane] =
            make_uint2(*reinterpret_cast<const uint32_t*>(&p_lo),
                       *reinterpret_cast<const uint32_t*>(&p_hi));
        float sum = (p0 + p1) + (p2 + p3);
#pragma unroll
        for (int sh = 8; sh > 0; sh >>= 1)
          sum += __shfl_xor_sync(0xffffffffu, sum, sh);
        const float scale = fast_exp2(rmax - maximum);
        rsum = rsum * scale + sum;
        rmax = maximum;
        if (hlane == 0) s.alpha[h24] = scale;
      };
      run_sm(hw, rmax_h0, rsum_h0);
      if (warp < 4) run_sm(16 + hw, rmax_h1, rsum_h1);
    }
    if (count1) load(b1, 0);
    stage_scales();
    __syncthreads();

    if (current) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int ph = pv_head(e);
        pv0[e] *= s.alpha[ph];
        pv1[e] *= s.alpha[8 + ph];
        pv2[e] *= s.alpha[16 + ph];
      }
      const int vk = lane & 3, vn = pv_base + ((lane & 16) ? 4 : 0);
#pragma unroll 4
      for (int k = 0; k < C; k += 4) {
        const int tok_v = k + vk;
        const int kh4 = k >> 2;
        const uint32_t codes = *reinterpret_cast<const uint32_t*>(
            &s.tile[tok_v][tile_swz_col(tok_v, vn)]);
        const uint32_t v01 =
            IS_INT8 ? int8x2_half(codes, 0x4140u) : e4m3x2_half(codes, 0x4140u);
        const uint32_t v23 =
            IS_INT8 ? int8x2_half(codes, 0x4342u) : e4m3x2_half(codes, 0x4342u);

        const uint2 p0_u2 = s.sp_f16[0][r8][kh4];
        FN_MMA_884_RR(pv0, p0_u2.x, p0_u2.y, v01, v23);
        const uint2 p1_u2 = s.sp_f16[1][r8][kh4];
        FN_MMA_884_RR(pv1, p1_u2.x, p1_u2.y, v01, v23);
        const uint2 p2_u2 = s.sp_f16[2][r8][kh4];
        FN_MMA_884_RR(pv2, p2_u2.x, p2_u2.y, v01, v23);
      }
    }
    const int count2 = __syncthreads_count(stage_store(b2));
    count0 = count1;
    count1 = count2;
  }

  if (hlane == 0) {
    s.denom[hw] = rsum_h0;
    if (lse != nullptr && (hw < 12 || has_r1)) {
      const int out_r = (hw < 12) ? r0 : r1;
      const int out_h = (hw < 12) ? hw : (hw - 12);
      lse[out_r * ls0 + out_h] =
          rsum_h0 > 0.f ? (rmax_h0 + log2f(rsum_h0)) : -CUDART_INF_F;
    }
    if (warp < 4) {
      const int h24 = 16 + hw;
      s.denom[h24] = rsum_h1;
      if (lse != nullptr && has_r1) {
        lse[r1 * ls0 + (h24 - 12)] =
            rsum_h1 > 0.f ? (rmax_h1 + log2f(rsum_h1)) : -CUDART_INF_F;
      }
    }
  }
  __syncthreads();

  auto emit = [&](int h24, int dim, float sum) {
    if (h24 >= 12 && !has_r1) return;
    const int out_r = (h24 < 12) ? r0 : r1;
    const int out_h = (h24 < 12) ? h24 : (h24 - 12);
    const float den = s.denom[h24];
    float val = den > 0.f ? __fdiv_rn(sum, den) : 0.f;
    if (gate != nullptr) {
      val = __half2float(__float2half_rn(val));
      const float g = __half2float(gate[out_r * gs0 + out_h * gs1 + dim]);
      val *= __fdiv_rn(1.f, 1.f + fast_exp2(-g * 1.4426950408889634f));
    }
    if (out_f32 != nullptr) {
      out_f32[out_r * os0 + out_h * os1 + dim] = val;
    } else {
      out_f16[out_r * os0 + out_h * os1 + dim] = __float2half_rn(val);
    }
  };

#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const int ph = pv_head(e);
    const int dm = pv_dim(e);
    emit(ph, dm, pv0[e]);
    emit(8 + ph, dm, pv1[e]);
    emit(16 + ph, dm, pv2[e]);
  }
}

struct QuadFast768Smem {
  alignas(16) half query[6][8][PAD];  // 25,344 B (48 heads = 6 x 8, 0 waste!)
  alignas(16) unsigned char tile[C][RKP];  // 16,384 B
  alignas(16) float scores[2][48][SP];     // 26,112 B
  alignas(16) uint2 sp_f16[6][8][C / 4];   //  6,144 B
  int32_t slots[3][C];                     //    768 B (full 3-stage pipeline!)
  alignas(16) float kscale[3][C];          //    768 B
  alignas(16) float vscale[3][C];          //    768 B
  uint8_t qmask[3][C];                     //    192 B
  float alpha[48];                         //    192 B
  float denom[48];                         //    192 B
};  // Total = 76,864 B <= 98,304 B -> 1 CTA x 768 threads = 24 warps/SM, <=80
    // regs, 0 spills!

template <bool IS_INT8>
__global__ __launch_bounds__(768, 1) void rows_fp8_h12_quad_fast_kernel(
    const half* __restrict__ q, int64_t qs0, int64_t qs1,
    const unsigned char* __restrict__ history, const float* __restrict__ scales,
    const int32_t* __restrict__ quad_indices,
    const uint8_t* __restrict__ quad_qmask,
    const int32_t* __restrict__ quad_counts, int quad_stride,
    const int32_t* __restrict__ table, int64_t ts0,
    const int32_t* __restrict__ requests, float* __restrict__ out_f32,
    half* __restrict__ out_f16, int64_t os0, int64_t os1,
    const half* __restrict__ gate, int64_t gs0, int64_t gs1,
    float* __restrict__ lse, int64_t ls0, int M, int blocks, int page,
    int table_width, int num_requests) {
  extern __shared__ __align__(16) unsigned char quad_fast_smem_raw[];
  QuadFast768Smem& s = *reinterpret_cast<QuadFast768Smem*>(quad_fast_smem_raw);

  const int quad_id = blockIdx.x;
  const int r_base = quad_id * 4;
  const int t = threadIdx.x;  // 0..767
  const int warp = t >> 5;    // 0..23
  const int lane = t & 31;    // 0..31

  int reqs[4];
  bool lives[4];
#pragma unroll
  for (int qi = 0; qi < 4; ++qi) {
    const int r = r_base + qi;
    const int rq = (r < M) ? requests[r] : -1;
    reqs[qi] = rq;
    lives[qi] = (r < M) && (rq >= 0 && rq < num_requests);
  }

  const int total_u = quad_counts[quad_id];
  const int ntiles = (total_u + 63) >> 6;
  const int64_t quad_base = static_cast<int64_t>(quad_id) * quad_stride;

  {
    const int d8 = lane * 8;
#pragma unroll
    for (int sub_h = 0; sub_h < 2; ++sub_h) {
      const int h48 = warp * 2 + sub_h;
      const int hg = h48 >> 3;
      const int hl = h48 & 7;
      const int qi = h48 / 12;
      const int hi = h48 - qi * 12;
      uint4 qv = make_uint4(0, 0, 0, 0);
      if (lives[qi]) {
        qv = __ldg(reinterpret_cast<const uint4*>(
            q + static_cast<int64_t>(r_base + qi) * qs0 + hi * qs1 + d8));
      }
      *reinterpret_cast<uint4*>(&s.query[hg][hl][d8]) = qv;
    }
  }

  int logical = -1;
  uint8_t qm_reg = 0;
  int32_t found = -1;
  float found_k = 0.f, found_v = 0.f;
  auto stage_index = [&](int j) {
    const int col = j * C + t;
    if (t < C && j < ntiles) {
      logical = quad_indices[quad_base + col];
      qm_reg = quad_qmask[quad_base + col];
    } else {
      logical = -1;
      qm_reg = 0;
    }
  };
  auto stage_table = [&]() {
    found = -1;
    if (logical >= 0 && logical / page < table_width && qm_reg != 0u) {
      const int owner_q = __ffs(static_cast<unsigned>(qm_reg)) - 1;
      const int use_req = reqs[owner_q];
      const int physical =
          table[static_cast<int64_t>(use_req) * ts0 + logical / page];
      if (physical >= 0 && physical < blocks) {
        found = physical * page + (logical % page);
      }
    }
  };
  auto stage_scales = [&]() {
    constexpr float kDecodeFactor = IS_INT8 ? 1.f : 256.f;
    found_k = found >= 0 ? __ldg(scales + static_cast<int64_t>(found) * 2) *
                               kDecodeFactor
                         : 0.f;
    found_v = found >= 0 ? __ldg(scales + static_cast<int64_t>(found) * 2 + 1) *
                               kDecodeFactor
                         : 0.f;
  };
  auto stage_store = [&](int b) {
    if (t < C) {
      s.slots[b][t] = found;
      s.kscale[b][t] = found_k;
      s.vscale[b][t] = found_v;
      s.qmask[b][t] = (found >= 0) ? qm_reg : 0u;
    }
    return t < C && found >= 0;
  };

  uint4 raw[2];
  auto load = [&](int b, int kv) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int v = t + i * 768;
      raw[i] = make_uint4(0, 0, 0, 0);
      if (v < 1024) {
        const int token = v >> 4, dim = (v & 15) * 16;
        const int32_t slot = s.slots[b][token];
        if (slot >= 0) {
          const int blk = slot / page, off = slot % page;
          raw[i] = __ldg(reinterpret_cast<const uint4*>(
              history + (int64_t(blk) * 2 * page + off) * D + dim +
              (kv ? int64_t(page) * D : 0)));
        }
      }
    }
  };
  auto store = [&]() {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int v = t + i * 768;
      if (v < 1024) {
        const int token = v >> 4, dim = (v & 15) * 16;
        *reinterpret_cast<uint4*>(&s.tile[token][tile_swz_col(token, dim)]) =
            raw[i];
      }
    }
  };

  stage_index(0);
  stage_table();
  stage_scales();
  int count0 = __syncthreads_count(stage_store(0));
  stage_index(1);
  stage_table();
  stage_scales();
  int count1 = __syncthreads_count(stage_store(1));
  if (count0) load(0, 0);

  const int hg_qk = warp >> 2;        // 0..5 (6 groups of 8 heads = 48 heads!)
  const int kw_qk = (warp >> 1) & 1;  // 0..1
  const int nt_qk = warp & 1;         // 0..1
  const int r8 = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int nr =
      nt_qk * 32 + ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);

  const int h48_sm = warp * 2 + (lane >> 4);  // 0..47
  const uint8_t row_bit_sm = static_cast<uint8_t>(1u << (h48_sm / 12));
  const int hlane = lane & 15;
  const int c4 = hlane * 4;

  const int hg_pv = warp >> 2;  // 0..5
  const int d_sub = warp & 3;   // 0..3 -> 64 dims/warp
  const int pv_base0 = d_sub * 64 + ((lane >> 2) & 3) * 8;
  const int pv_base1 = pv_base0 + 32;
  auto pv_head = [&](int e) {
    return (e & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
  };
  auto pv_dim_off = [&](int e) {
    return (e & 1) | (((lane >> 1) & 1) << 1) | ((e >> 2) << 2);
  };

  float rmax_h = -1e20f;
  float rsum_h = 0.f;
  float pv0[8] = {};
  float pv1[8] = {};

  for (int j = 0; j < ntiles; ++j) {
    const int b = j % 3, b1 = (j + 1) % 3, b2 = (j + 2) % 3;
    const bool current = (count0 > 0);
    if (!current && count1 == 0) break;
    const int tile_w = min(C, total_u - j * C);
    stage_index(j + 2);
    if (current) {
      store();
      load(b, 1);
    }
    __syncthreads();

    if (current && (nt_qk == 0 || tile_w > 32)) {
      float acc[8] = {};
#pragma unroll 4
      for (int sub = 0; sub < 8; ++sub) {
        const int k = kw_qk * 128 + sub * 16;
        const uint4 raw16 =
            *reinterpret_cast<const uint4*>(&s.tile[nr][tile_swz_col(nr, k)]);
        const uint4 k_lo =
            IS_INT8 ? decode8_int8_unscaled(make_uint2(raw16.x, raw16.y))
                    : decode8_unscaled(make_uint2(raw16.x, raw16.y));
        const uint4 k_hi =
            IS_INT8 ? decode8_int8_unscaled(make_uint2(raw16.z, raw16.w))
                    : decode8_unscaled(make_uint2(raw16.z, raw16.w));
        const uint4 q_0 =
            *reinterpret_cast<const uint4*>(&s.query[hg_qk][r8][k]);
        const uint4 q_1 =
            *reinterpret_cast<const uint4*>(&s.query[hg_qk][r8][k + 8]);
        FN_MMA_884_RC(acc, q_0.x, q_0.y, k_lo.x, k_lo.y);
        FN_MMA_884_RC(acc, q_0.z, q_0.w, k_lo.z, k_lo.w);
        FN_MMA_884_RC(acc, q_1.x, q_1.y, k_hi.x, k_hi.y);
        FN_MMA_884_RC(acc, q_1.z, q_1.w, k_hi.z, k_hi.w);
      }
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int h = (e & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
        const int col = nt_qk * 32 + ((lane >> 2) & 3) * 8 +
                        ((e & 1) | (((lane >> 1) & 1) << 1) | ((e >> 2) << 2));
        s.scores[kw_qk][hg_qk * 8 + h][col] = acc[e];
      }
    }
    stage_table();
    __syncthreads();

    if (current) {
      store();
      float s0 = -1e20f, s1 = -1e20f, s2 = -1e20f, s3 = -1e20f;
      bool v0 = false, v1 = false, v2 = false, v3 = false;
      float4 vs = make_float4(0.f, 0.f, 0.f, 0.f);
      if (c4 < tile_w) {
        const uint32_t m4 = *reinterpret_cast<const uint32_t*>(&s.qmask[b][c4]);
        v0 = (((m4 >> 0) & row_bit_sm) != 0);
        v1 = (((m4 >> 8) & row_bit_sm) != 0);
        v2 = (((m4 >> 16) & row_bit_sm) != 0);
        v3 = (((m4 >> 24) & row_bit_sm) != 0);
        const float4 ks = *reinterpret_cast<const float4*>(&s.kscale[b][c4]);
        vs = *reinterpret_cast<const float4*>(&s.vscale[b][c4]);
        const float4 sc0 =
            *reinterpret_cast<const float4*>(&s.scores[0][h48_sm][c4]);
        const float4 sc1 =
            *reinterpret_cast<const float4*>(&s.scores[1][h48_sm][c4]);
        constexpr float kLog2Scale = 0.0625f * 1.4426950408889634f;
        if (v0) s0 = (sc0.x + sc1.x) * (ks.x * kLog2Scale);
        if (v1) s1 = (sc0.y + sc1.y) * (ks.y * kLog2Scale);
        if (v2) s2 = (sc0.z + sc1.z) * (ks.z * kLog2Scale);
        if (v3) s3 = (sc0.w + sc1.w) * (ks.w * kLog2Scale);
      }

      float maximum = fmaxf(fmaxf(s0, s1), fmaxf(s2, s3));
#pragma unroll
      for (int sh = 8; sh > 0; sh >>= 1)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, sh));
      maximum = fmaxf(maximum, rmax_h);
      const float p0 = v0 ? fast_exp2(s0 - maximum) : 0.f;
      const float p1 = v1 ? fast_exp2(s1 - maximum) : 0.f;
      const float p2 = v2 ? fast_exp2(s2 - maximum) : 0.f;
      const float p3 = v3 ? fast_exp2(s3 - maximum) : 0.f;
      const half2 p_lo = __floats2half2_rn(p0 * vs.x, p1 * vs.y);
      const half2 p_hi = __floats2half2_rn(p2 * vs.z, p3 * vs.w);
      s.sp_f16[h48_sm >> 3][h48_sm & 7][hlane] =
          make_uint2(*reinterpret_cast<const uint32_t*>(&p_lo),
                     *reinterpret_cast<const uint32_t*>(&p_hi));
      float sum = (p0 + p1) + (p2 + p3);
#pragma unroll
      for (int sh = 8; sh > 0; sh >>= 1)
        sum += __shfl_xor_sync(0xffffffffu, sum, sh);
      const float scale = fast_exp2(rmax_h - maximum);
      rsum_h = rsum_h * scale + sum;
      rmax_h = maximum;
      if (hlane == 0) s.alpha[h48_sm] = scale;
    }
    if (count1) load(b1, 0);
    stage_scales();
    __syncthreads();

    if (current) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float a_s = s.alpha[hg_pv * 8 + pv_head(e)];
        pv0[e] *= a_s;
        pv1[e] *= a_s;
      }
      const int vk = lane & 3;
      const int vn0 = pv_base0 + ((lane & 16) ? 4 : 0);
      const int vn1 = pv_base1 + ((lane & 16) ? 4 : 0);
#pragma unroll 4
      for (int k = 0; k < tile_w; k += 4) {
        const int tok_v = k + vk;
        const int kh4 = k >> 2;
        const uint2 p_u2 = s.sp_f16[hg_pv][r8][kh4];

        const uint32_t c0 = *reinterpret_cast<const uint32_t*>(
            &s.tile[tok_v][tile_swz_col(tok_v, vn0)]);
        const uint32_t v0_01 =
            IS_INT8 ? int8x2_half(c0, 0x4140u) : e4m3x2_half(c0, 0x4140u);
        const uint32_t v0_23 =
            IS_INT8 ? int8x2_half(c0, 0x4342u) : e4m3x2_half(c0, 0x4342u);
        FN_MMA_884_RR(pv0, p_u2.x, p_u2.y, v0_01, v0_23);

        const uint32_t c1 = *reinterpret_cast<const uint32_t*>(
            &s.tile[tok_v][tile_swz_col(tok_v, vn1)]);
        const uint32_t v1_01 =
            IS_INT8 ? int8x2_half(c1, 0x4140u) : e4m3x2_half(c1, 0x4140u);
        const uint32_t v1_23 =
            IS_INT8 ? int8x2_half(c1, 0x4342u) : e4m3x2_half(c1, 0x4342u);
        FN_MMA_884_RR(pv1, p_u2.x, p_u2.y, v1_01, v1_23);
      }
    }
    const int count2 = __syncthreads_count(stage_store(b2));
    count0 = count1;
    count1 = count2;
  }

  if (hlane == 0) {
    s.denom[h48_sm] = rsum_h;
    const int qi = h48_sm / 12;
    const int out_r = r_base + qi;
    if (lse != nullptr && out_r < M) {
      const int out_h = h48_sm - qi * 12;
      lse[static_cast<int64_t>(out_r) * ls0 + out_h] =
          rsum_h > 0.f ? (rmax_h + log2f(rsum_h)) : -CUDART_INF_F;
    }
  }
  __syncthreads();

  auto emit = [&](int h48, int dim, float sum) {
    const int qi = h48 / 12;
    const int out_r = r_base + qi;
    if (out_r >= M) return;
    const int out_h = h48 - qi * 12;
    const float den = s.denom[h48];
    float val = den > 0.f ? __fdiv_rn(sum, den) : 0.f;
    if (gate != nullptr) {
      val = __half2float(__float2half_rn(val));
      const float g = __half2float(
          gate[static_cast<int64_t>(out_r) * gs0 + out_h * gs1 + dim]);
      val *= __fdiv_rn(1.f, 1.f + fast_exp2(-g * 1.4426950408889634f));
    }
    if (out_f32 != nullptr) {
      out_f32[static_cast<int64_t>(out_r) * os0 + out_h * os1 + dim] = val;
    } else {
      out_f16[static_cast<int64_t>(out_r) * os0 + out_h * os1 + dim] =
          __float2half_rn(val);
    }
  };

#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const int h48 = hg_pv * 8 + pv_head(e);
    const int doff = pv_dim_off(e);
    emit(h48, pv_base0 + doff, pv0[e]);
    emit(h48, pv_base1 + doff, pv1[e]);
  }
}

}  // namespace qsa_h12

namespace {
torch::Tensor g_pair_merged_idx[16];
torch::Tensor g_pair_merged_qm[16];
torch::Tensor g_pair_ntiles_buf[16];
}  // namespace

void init_qsa_merge_workspace_out(int64_t max_m, int64_t width,
                                  int64_t device_idx) {
  const int dev = static_cast<int>(device_idx);
  if (dev < 0 || dev >= 16 || max_m <= 0 || width <= 0) return;
  const c10::cuda::CUDAGuard guard(dev);
  const int max_p = std::max(static_cast<int>((max_m + 1) / 2), 8192);
  const int max_s =
      std::max(static_cast<int>(((2 * width + 127) / 64) * 64), 2176);
  const int64_t needed = static_cast<int64_t>(max_p) * max_s;
  auto opts_dev = torch::TensorOptions().device(torch::kCUDA, dev);
  if (!g_pair_merged_idx[dev].defined() ||
      g_pair_merged_idx[dev].numel() < needed) {
    g_pair_merged_idx[dev] =
        torch::empty({needed}, opts_dev.dtype(torch::kInt32));
    g_pair_merged_qm[dev] =
        torch::empty({needed}, opts_dev.dtype(torch::kUInt8));
    g_pair_ntiles_buf[dev] =
        torch::empty({max_p}, opts_dev.dtype(torch::kInt32));
  }
}

void run_rows_h12(torch::Tensor q, torch::Tensor history, torch::Tensor scales,
                  torch::Tensor indices, torch::Tensor table,
                  torch::Tensor requests, torch::Tensor positions,
                  torch::Tensor lengths, torch::Tensor out,
                  std::optional<torch::Tensor> gate,
                  std::optional<torch::Tensor> lse) {
  const c10::cuda::CUDAGuard guard(q.device());
  const int M = static_cast<int>(q.size(0));
  if (M == 0) return;
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kHalf && q.dim() == 3 &&
              q.size(1) == 12 && q.size(2) == 256 && q.stride(2) == 1);
  const bool is_int8 = history.scalar_type() == at::kChar;
  const bool is_e4m3 = history.scalar_type() == at::kByte;
  TORCH_CHECK(is_int8 || is_e4m3);
  const bool out_is_f32 = out.scalar_type() == at::kFloat;
  const int cur_dev = q.device().index();
  constexpr size_t kSmemBytes = sizeof(qsa_h12::H12Smem);
  constexpr size_t kPairFastSmemBytes = sizeof(qsa_h12::PairFast384Smem);
  constexpr size_t kQuadFastSmemBytes = sizeof(qsa_h12::QuadFast768Smem);
  constexpr size_t kQuadSmemBytes = sizeof(qsa_h12::QuadH12Smem);
  const char* qsa_env = std::getenv("FN_QSA_MODE");
  const int qsa_mode = qsa_env ? std::atoi(qsa_env) : 0;
  const bool use_quad_fast768 = (qsa_mode == 5) && (M >= 256);
  const bool use_pair_fast384 =
      (qsa_mode == 2 && M >= 64) || (qsa_mode == 0 && M >= 64);
  const bool use_pair_fast256 = (qsa_mode == 3 && M >= 64);
  const bool use_quad = (qsa_mode == 4) && (M >= 64);
  auto stream = at::cuda::getCurrentCUDAStream();

  torch::Tensor(&pair_merged_idx)[16] = g_pair_merged_idx;
  torch::Tensor(&pair_merged_qm)[16] = g_pair_merged_qm;
  torch::Tensor(&pair_ntiles_buf)[16] = g_pair_ntiles_buf;
  static torch::Tensor quad_merged_idx[16];
  static torch::Tensor quad_merged_qm[16];
  static torch::Tensor quad_counts_buf[16];
  int num_pairs = (M + 1) / 2;
  int num_quads = (M + 3) / 4;
  int width = static_cast<int>(indices.size(1));
  int merge_stride = ((2 * width + 127) / 64) * 64;
  int quad_stride = ((4 * width + 127) / 64) * 64;
  if ((use_quad_fast768 || use_pair_fast384 || use_pair_fast256) &&
      cur_dev >= 0 && cur_dev < 16) {
    const int64_t needed = static_cast<int64_t>(num_pairs) * merge_stride;
    if (!pair_merged_idx[cur_dev].defined() ||
        pair_merged_idx[cur_dev].numel() < needed) {
      const int max_p = std::max(num_pairs, 8192);
      const int max_s = std::max(merge_stride, 2176);
      pair_merged_idx[cur_dev] = torch::empty(
          {static_cast<int64_t>(max_p) * max_s},
          torch::TensorOptions().dtype(torch::kInt32).device(q.device()));
      pair_merged_qm[cur_dev] = torch::empty(
          {static_cast<int64_t>(max_p) * max_s},
          torch::TensorOptions().dtype(torch::kUInt8).device(q.device()));
      pair_ntiles_buf[cur_dev] = torch::empty(
          {max_p},
          torch::TensorOptions().dtype(torch::kInt32).device(q.device()));
    }
    qsa_h12::fn_qsa_pair_merge_indices_kernel<<<(num_pairs + 3) / 4, 128, 0,
                                                stream>>>(
        indices.data_ptr<int32_t>(), indices.stride(0),
        requests.data_ptr<int32_t>(), positions.data_ptr<int64_t>(),
        lengths.data_ptr<int32_t>(),
        pair_merged_idx[cur_dev].data_ptr<int32_t>(),
        pair_merged_qm[cur_dev].data_ptr<uint8_t>(),
        pair_ntiles_buf[cur_dev].data_ptr<int32_t>(), M, width, merge_stride,
        static_cast<int>(table.size(0)), num_pairs);
    if (use_quad_fast768) {
      const int64_t q_needed = static_cast<int64_t>(num_quads) * quad_stride;
      if (!quad_merged_idx[cur_dev].defined() ||
          quad_merged_idx[cur_dev].numel() < q_needed) {
        const int max_q = std::max(num_quads, 4096);
        const int max_qs = std::max(quad_stride, 4224);
        quad_merged_idx[cur_dev] = torch::empty(
            {static_cast<int64_t>(max_q) * max_qs},
            torch::TensorOptions().dtype(torch::kInt32).device(q.device()));
        quad_merged_qm[cur_dev] = torch::empty(
            {static_cast<int64_t>(max_q) * max_qs},
            torch::TensorOptions().dtype(torch::kUInt8).device(q.device()));
        quad_counts_buf[cur_dev] = torch::empty(
            {max_q},
            torch::TensorOptions().dtype(torch::kInt32).device(q.device()));
      }
      qsa_h12::fn_qsa_merge_two_pairs_kernel<<<num_quads, 32, 0, stream>>>(
          pair_merged_idx[cur_dev].data_ptr<int32_t>(),
          pair_merged_qm[cur_dev].data_ptr<uint8_t>(),
          pair_ntiles_buf[cur_dev].data_ptr<int32_t>(), merge_stride,
          requests.data_ptr<int32_t>(),
          quad_merged_idx[cur_dev].data_ptr<int32_t>(),
          quad_merged_qm[cur_dev].data_ptr<uint8_t>(),
          quad_counts_buf[cur_dev].data_ptr<int32_t>(), quad_stride, M,
          num_pairs, static_cast<int>(table.size(0)));
    }
  }

  if (is_int8) {
    static bool configured_devs_i8[16] = {};
    if (cur_dev >= 0 && cur_dev < 16 && !configured_devs_i8[cur_dev]) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          qsa_h12::rows_fp8_h12_kernel<true>,
          cudaFuncAttributePreferredSharedMemoryCarveout, 100));
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          qsa_h12::rows_fp8_h12_kernel<true>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          qsa_h12::rows_fp8_h12_pair_fast_kernel<true>,
          cudaFuncAttributePreferredSharedMemoryCarveout, 100));
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          qsa_h12::rows_fp8_h12_pair_fast_kernel<true>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, kPairFastSmemBytes));
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          qsa_h12::rows_fp8_h12_quad_fast_kernel<true>,
          cudaFuncAttributePreferredSharedMemoryCarveout, 100));
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          qsa_h12::rows_fp8_h12_quad_fast_kernel<true>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, kQuadFastSmemBytes));
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          qsa_h12::rows_fp8_h12_pair_fast256_kernel<true>,
          cudaFuncAttributePreferredSharedMemoryCarveout, 100));
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          qsa_h12::rows_fp8_h12_pair_fast256_kernel<true>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, kPairFastSmemBytes));
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          qsa_h12::rows_fp8_h12_quad_kernel<true>,
          cudaFuncAttributePreferredSharedMemoryCarveout, 100));
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          qsa_h12::rows_fp8_h12_quad_kernel<true>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, kQuadSmemBytes));
      configured_devs_i8[cur_dev] = true;
    }
    if (use_quad_fast768) {
      qsa_h12::rows_fp8_h12_quad_fast_kernel<true>
          <<<num_quads, 768, kQuadFastSmemBytes, stream>>>(
              reinterpret_cast<const half*>(q.data_ptr()), q.stride(0),
              q.stride(1),
              reinterpret_cast<const unsigned char*>(history.data_ptr()),
              scales.data_ptr<float>(),
              quad_merged_idx[cur_dev].data_ptr<int32_t>(),
              quad_merged_qm[cur_dev].data_ptr<uint8_t>(),
              quad_counts_buf[cur_dev].data_ptr<int32_t>(), quad_stride,
              table.data_ptr<int32_t>(), table.stride(0),
              requests.data_ptr<int32_t>(),
              out_is_f32 ? out.data_ptr<float>() : nullptr,
              out_is_f32 ? nullptr : reinterpret_cast<half*>(out.data_ptr()),
              out.stride(0), out.stride(1),
              gate ? reinterpret_cast<const half*>(gate->data_ptr()) : nullptr,
              gate ? gate->stride(0) : 0, gate ? gate->stride(1) : 0,
              lse ? lse->data_ptr<float>() : nullptr, lse ? lse->stride(0) : 0,
              M, static_cast<int>(history.size(0)),
              static_cast<int>(history.size(2)),
              static_cast<int>(table.size(1)), static_cast<int>(table.size(0)));
    } else if (use_pair_fast384) {
      qsa_h12::rows_fp8_h12_pair_fast_kernel<true>
          <<<num_pairs, 384, kPairFastSmemBytes, stream>>>(
              reinterpret_cast<const half*>(q.data_ptr()), q.stride(0),
              q.stride(1),
              reinterpret_cast<const unsigned char*>(history.data_ptr()),
              scales.data_ptr<float>(),
              pair_merged_idx[cur_dev].data_ptr<int32_t>(),
              pair_merged_qm[cur_dev].data_ptr<uint8_t>(),
              pair_ntiles_buf[cur_dev].data_ptr<int32_t>(), merge_stride,
              table.data_ptr<int32_t>(), table.stride(0),
              requests.data_ptr<int32_t>(),
              out_is_f32 ? out.data_ptr<float>() : nullptr,
              out_is_f32 ? nullptr : reinterpret_cast<half*>(out.data_ptr()),
              out.stride(0), out.stride(1),
              gate ? reinterpret_cast<const half*>(gate->data_ptr()) : nullptr,
              gate ? gate->stride(0) : 0, gate ? gate->stride(1) : 0,
              lse ? lse->data_ptr<float>() : nullptr, lse ? lse->stride(0) : 0,
              M, static_cast<int>(history.size(0)),
              static_cast<int>(history.size(2)),
              static_cast<int>(table.size(1)), static_cast<int>(table.size(0)));
    } else if (use_pair_fast256) {
      qsa_h12::rows_fp8_h12_pair_fast256_kernel<true>
          <<<num_pairs, 256, kPairFastSmemBytes, stream>>>(
              reinterpret_cast<const half*>(q.data_ptr()), q.stride(0),
              q.stride(1),
              reinterpret_cast<const unsigned char*>(history.data_ptr()),
              scales.data_ptr<float>(),
              pair_merged_idx[cur_dev].data_ptr<int32_t>(),
              pair_merged_qm[cur_dev].data_ptr<uint8_t>(),
              pair_ntiles_buf[cur_dev].data_ptr<int32_t>(), merge_stride,
              table.data_ptr<int32_t>(), table.stride(0),
              requests.data_ptr<int32_t>(),
              out_is_f32 ? out.data_ptr<float>() : nullptr,
              out_is_f32 ? nullptr : reinterpret_cast<half*>(out.data_ptr()),
              out.stride(0), out.stride(1),
              gate ? reinterpret_cast<const half*>(gate->data_ptr()) : nullptr,
              gate ? gate->stride(0) : 0, gate ? gate->stride(1) : 0,
              lse ? lse->data_ptr<float>() : nullptr, lse ? lse->stride(0) : 0,
              M, static_cast<int>(history.size(0)),
              static_cast<int>(history.size(2)),
              static_cast<int>(table.size(1)), static_cast<int>(table.size(0)));
    } else if (use_quad) {
      qsa_h12::rows_fp8_h12_quad_kernel<true>
          <<<(M + 3) / 4, 512, kQuadSmemBytes, stream>>>(
              reinterpret_cast<const half*>(q.data_ptr()), q.stride(0),
              q.stride(1),
              reinterpret_cast<const unsigned char*>(history.data_ptr()),
              scales.data_ptr<float>(), indices.data_ptr<int32_t>(),
              indices.stride(0), table.data_ptr<int32_t>(), table.stride(0),
              requests.data_ptr<int32_t>(), positions.data_ptr<int64_t>(),
              lengths.data_ptr<int32_t>(),
              out_is_f32 ? out.data_ptr<float>() : nullptr,
              out_is_f32 ? nullptr : reinterpret_cast<half*>(out.data_ptr()),
              out.stride(0), out.stride(1),
              gate ? reinterpret_cast<const half*>(gate->data_ptr()) : nullptr,
              gate ? gate->stride(0) : 0, gate ? gate->stride(1) : 0,
              lse ? lse->data_ptr<float>() : nullptr, lse ? lse->stride(0) : 0,
              M, static_cast<int>(history.size(0)),
              static_cast<int>(history.size(2)),
              static_cast<int>(indices.size(1)),
              static_cast<int>(table.size(1)), static_cast<int>(table.size(0)));
    } else {
      qsa_h12::rows_fp8_h12_kernel<true><<<M, 256, kSmemBytes, stream>>>(
          reinterpret_cast<const half*>(q.data_ptr()), q.stride(0), q.stride(1),
          reinterpret_cast<const unsigned char*>(history.data_ptr()),
          scales.data_ptr<float>(), indices.data_ptr<int32_t>(),
          indices.stride(0), table.data_ptr<int32_t>(), table.stride(0),
          requests.data_ptr<int32_t>(), positions.data_ptr<int64_t>(),
          lengths.data_ptr<int32_t>(),
          out_is_f32 ? out.data_ptr<float>() : nullptr,
          out_is_f32 ? nullptr : reinterpret_cast<half*>(out.data_ptr()),
          out.stride(0), out.stride(1),
          gate ? reinterpret_cast<const half*>(gate->data_ptr()) : nullptr,
          gate ? gate->stride(0) : 0, gate ? gate->stride(1) : 0,
          lse ? lse->data_ptr<float>() : nullptr, lse ? lse->stride(0) : 0,
          static_cast<int>(history.size(0)), static_cast<int>(history.size(2)),
          static_cast<int>(indices.size(1)), static_cast<int>(table.size(1)),
          static_cast<int>(table.size(0)));
    }
  } else {
    static bool configured_devs_e4[16] = {};
    if (cur_dev >= 0 && cur_dev < 16 && !configured_devs_e4[cur_dev]) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          qsa_h12::rows_fp8_h12_kernel<false>,
          cudaFuncAttributePreferredSharedMemoryCarveout, 100));
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          qsa_h12::rows_fp8_h12_kernel<false>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          qsa_h12::rows_fp8_h12_pair_fast_kernel<false>,
          cudaFuncAttributePreferredSharedMemoryCarveout, 100));
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          qsa_h12::rows_fp8_h12_pair_fast_kernel<false>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, kPairFastSmemBytes));
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          qsa_h12::rows_fp8_h12_quad_fast_kernel<false>,
          cudaFuncAttributePreferredSharedMemoryCarveout, 100));
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          qsa_h12::rows_fp8_h12_quad_fast_kernel<false>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, kQuadFastSmemBytes));
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          qsa_h12::rows_fp8_h12_pair_fast256_kernel<false>,
          cudaFuncAttributePreferredSharedMemoryCarveout, 100));
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          qsa_h12::rows_fp8_h12_pair_fast256_kernel<false>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, kPairFastSmemBytes));
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          qsa_h12::rows_fp8_h12_quad_kernel<false>,
          cudaFuncAttributePreferredSharedMemoryCarveout, 100));
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          qsa_h12::rows_fp8_h12_quad_kernel<false>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, kQuadSmemBytes));
      configured_devs_e4[cur_dev] = true;
    }
    if (use_quad_fast768) {
      qsa_h12::rows_fp8_h12_quad_fast_kernel<false>
          <<<num_quads, 768, kQuadFastSmemBytes, stream>>>(
              reinterpret_cast<const half*>(q.data_ptr()), q.stride(0),
              q.stride(1),
              reinterpret_cast<const unsigned char*>(history.data_ptr()),
              scales.data_ptr<float>(),
              quad_merged_idx[cur_dev].data_ptr<int32_t>(),
              quad_merged_qm[cur_dev].data_ptr<uint8_t>(),
              quad_counts_buf[cur_dev].data_ptr<int32_t>(), quad_stride,
              table.data_ptr<int32_t>(), table.stride(0),
              requests.data_ptr<int32_t>(),
              out_is_f32 ? out.data_ptr<float>() : nullptr,
              out_is_f32 ? nullptr : reinterpret_cast<half*>(out.data_ptr()),
              out.stride(0), out.stride(1),
              gate ? reinterpret_cast<const half*>(gate->data_ptr()) : nullptr,
              gate ? gate->stride(0) : 0, gate ? gate->stride(1) : 0,
              lse ? lse->data_ptr<float>() : nullptr, lse ? lse->stride(0) : 0,
              M, static_cast<int>(history.size(0)),
              static_cast<int>(history.size(2)),
              static_cast<int>(table.size(1)), static_cast<int>(table.size(0)));
    } else if (use_pair_fast384) {
      qsa_h12::rows_fp8_h12_pair_fast_kernel<false>
          <<<num_pairs, 384, kPairFastSmemBytes, stream>>>(
              reinterpret_cast<const half*>(q.data_ptr()), q.stride(0),
              q.stride(1),
              reinterpret_cast<const unsigned char*>(history.data_ptr()),
              scales.data_ptr<float>(),
              pair_merged_idx[cur_dev].data_ptr<int32_t>(),
              pair_merged_qm[cur_dev].data_ptr<uint8_t>(),
              pair_ntiles_buf[cur_dev].data_ptr<int32_t>(), merge_stride,
              table.data_ptr<int32_t>(), table.stride(0),
              requests.data_ptr<int32_t>(),
              out_is_f32 ? out.data_ptr<float>() : nullptr,
              out_is_f32 ? nullptr : reinterpret_cast<half*>(out.data_ptr()),
              out.stride(0), out.stride(1),
              gate ? reinterpret_cast<const half*>(gate->data_ptr()) : nullptr,
              gate ? gate->stride(0) : 0, gate ? gate->stride(1) : 0,
              lse ? lse->data_ptr<float>() : nullptr, lse ? lse->stride(0) : 0,
              M, static_cast<int>(history.size(0)),
              static_cast<int>(history.size(2)),
              static_cast<int>(table.size(1)), static_cast<int>(table.size(0)));
    } else if (use_pair_fast256) {
      qsa_h12::rows_fp8_h12_pair_fast256_kernel<false>
          <<<num_pairs, 256, kPairFastSmemBytes, stream>>>(
              reinterpret_cast<const half*>(q.data_ptr()), q.stride(0),
              q.stride(1),
              reinterpret_cast<const unsigned char*>(history.data_ptr()),
              scales.data_ptr<float>(),
              pair_merged_idx[cur_dev].data_ptr<int32_t>(),
              pair_merged_qm[cur_dev].data_ptr<uint8_t>(),
              pair_ntiles_buf[cur_dev].data_ptr<int32_t>(), merge_stride,
              table.data_ptr<int32_t>(), table.stride(0),
              requests.data_ptr<int32_t>(),
              out_is_f32 ? out.data_ptr<float>() : nullptr,
              out_is_f32 ? nullptr : reinterpret_cast<half*>(out.data_ptr()),
              out.stride(0), out.stride(1),
              gate ? reinterpret_cast<const half*>(gate->data_ptr()) : nullptr,
              gate ? gate->stride(0) : 0, gate ? gate->stride(1) : 0,
              lse ? lse->data_ptr<float>() : nullptr, lse ? lse->stride(0) : 0,
              M, static_cast<int>(history.size(0)),
              static_cast<int>(history.size(2)),
              static_cast<int>(table.size(1)), static_cast<int>(table.size(0)));
    } else if (use_quad) {
      qsa_h12::rows_fp8_h12_quad_kernel<false>
          <<<(M + 3) / 4, 512, kQuadSmemBytes, stream>>>(
              reinterpret_cast<const half*>(q.data_ptr()), q.stride(0),
              q.stride(1),
              reinterpret_cast<const unsigned char*>(history.data_ptr()),
              scales.data_ptr<float>(), indices.data_ptr<int32_t>(),
              indices.stride(0), table.data_ptr<int32_t>(), table.stride(0),
              requests.data_ptr<int32_t>(), positions.data_ptr<int64_t>(),
              lengths.data_ptr<int32_t>(),
              out_is_f32 ? out.data_ptr<float>() : nullptr,
              out_is_f32 ? nullptr : reinterpret_cast<half*>(out.data_ptr()),
              out.stride(0), out.stride(1),
              gate ? reinterpret_cast<const half*>(gate->data_ptr()) : nullptr,
              gate ? gate->stride(0) : 0, gate ? gate->stride(1) : 0,
              lse ? lse->data_ptr<float>() : nullptr, lse ? lse->stride(0) : 0,
              M, static_cast<int>(history.size(0)),
              static_cast<int>(history.size(2)),
              static_cast<int>(indices.size(1)),
              static_cast<int>(table.size(1)), static_cast<int>(table.size(0)));
    } else {
      qsa_h12::rows_fp8_h12_kernel<false><<<M, 256, kSmemBytes, stream>>>(
          reinterpret_cast<const half*>(q.data_ptr()), q.stride(0), q.stride(1),
          reinterpret_cast<const unsigned char*>(history.data_ptr()),
          scales.data_ptr<float>(), indices.data_ptr<int32_t>(),
          indices.stride(0), table.data_ptr<int32_t>(), table.stride(0),
          requests.data_ptr<int32_t>(), positions.data_ptr<int64_t>(),
          lengths.data_ptr<int32_t>(),
          out_is_f32 ? out.data_ptr<float>() : nullptr,
          out_is_f32 ? nullptr : reinterpret_cast<half*>(out.data_ptr()),
          out.stride(0), out.stride(1),
          gate ? reinterpret_cast<const half*>(gate->data_ptr()) : nullptr,
          gate ? gate->stride(0) : 0, gate ? gate->stride(1) : 0,
          lse ? lse->data_ptr<float>() : nullptr, lse ? lse->stride(0) : 0,
          static_cast<int>(history.size(0)), static_cast<int>(history.size(2)),
          static_cast<int>(indices.size(1)), static_cast<int>(table.size(1)),
          static_cast<int>(table.size(0)));
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ============================================================================
// Part 5: Bulk 16B NVLink P2P DCP=2 Q-Prep & FP16/FP32 Combine-Gate
// ============================================================================

namespace dcp2_any_m {

constexpr int kBulkM = 4096;
constexpr int kHLocal = 6;
constexpr int kDim = 256;
constexpr int64_t kQSlotHalves = static_cast<int64_t>(kBulkM) * kHLocal * kDim;
constexpr int64_t kQSyncOffsetBytes =
    2 * kQSlotHalves * static_cast<int64_t>(sizeof(half));

// FP32 partial_out layout
constexpr int64_t kOSlotOutFloats =
    static_cast<int64_t>(kBulkM) * kHLocal * kDim;
constexpr int64_t kOSlotLseFloats = static_cast<int64_t>(kBulkM) * kHLocal;
constexpr int64_t kOSlotTotalFloats = kOSlotOutFloats + kOSlotLseFloats;
constexpr int64_t kOSyncOffsetBytes =
    2 * kOSlotTotalFloats * static_cast<int64_t>(sizeof(float));

// FP16 partial_out layout (halves NVLink P2P push bytes from 24 MiB -> 12 MiB!)
constexpr int64_t kOSlotOutHalfBytes =
    static_cast<int64_t>(kBulkM) * kHLocal * kDim * sizeof(half);
constexpr int64_t kOSlotLseBytes =
    static_cast<int64_t>(kBulkM) * kHLocal * sizeof(float);
constexpr int64_t kOSlotTotalFp16Bytes = kOSlotOutHalfBytes + kOSlotLseBytes;
constexpr int64_t kOSyncOffsetFp16Bytes = 2 * kOSlotTotalFp16Bytes;

__device__ __forceinline__ unsigned ldv_u32(const unsigned* p) {
  unsigned v;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

__device__ __forceinline__ void stv_u32(unsigned* p, unsigned v) {
  asm volatile("st.volatile.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ void dcp2_signal_last_cta(
    unsigned* __restrict__ my_sync, unsigned* __restrict__ peer_sync,
    unsigned* __restrict__ seq_state) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();
    if (atomicAdd(my_sync + 1, 1u) + 1u == gridDim.x) {
      my_sync[1] = 0u;
      const unsigned ep = *seq_state + 1u;
      *seq_state = ep;
      __threadfence_system();
      stv_u32(peer_sync, ep);
    }
  }
}

__device__ __forceinline__ void dcp2_wait_peer(
    const unsigned* __restrict__ my_sync, unsigned ep) {
  if (threadIdx.x == 0) {
    const long long t0 = clock64();
    while (ldv_u32(my_sync) < ep) {
      const long long w0 = clock64();
      while (clock64() - w0 < 64) {
      }
      if (w0 - t0 > 8000000000LL) __trap();
    }
  }
  __syncthreads();
}

__global__ __launch_bounds__(256) void dcp2_o_push_kernel(
    const float* __restrict__ partial_out,
    const float* __restrict__ partial_lse, float* __restrict__ peer_buf,
    unsigned* __restrict__ my_sync, unsigned* __restrict__ peer_sync,
    unsigned* __restrict__ seq_state, int m0, int tm, int dcp_rank) {
  const int slot = static_cast<int>(*seq_state & 1u);
  float* peer_slot_base = peer_buf + slot * kOSlotTotalFloats;
  float4* peer_o4 = reinterpret_cast<float4*>(peer_slot_base);
  float* peer_lse = peer_slot_base + kOSlotOutFloats;

  const int peer_h0 = (1 - dcp_rank) * kHLocal;
  const int tx = threadIdx.x;
  const int pair_in_cta = tx >> 6;
  const int d_vec = tx & 63;
  const int total_pairs = tm * kHLocal;

  for (int pair = blockIdx.x * 4 + pair_in_cta; pair < total_pairs;
       pair += gridDim.x * 4) {
    const int lm = pair / kHLocal;
    const int h = pair % kHLocal;
    const int m = m0 + lm;
    const int ph = peer_h0 + h;
    const float4 v4 = __ldg(reinterpret_cast<const float4*>(
        partial_out + (static_cast<int64_t>(m) * (2 * kHLocal) + ph) * kDim +
        d_vec * 4));
    peer_o4[static_cast<int64_t>(pair) * 64 + d_vec] = v4;
    if (d_vec == 0) {
      peer_lse[pair] =
          partial_lse[static_cast<int64_t>(m) * (2 * kHLocal) + ph];
    }
  }
  dcp2_signal_last_cta(my_sync, peer_sync, seq_state);
}

__global__ __launch_bounds__(256) void dcp2_o_combine_gate_kernel(
    const float* __restrict__ partial_out,
    const float* __restrict__ partial_lse, const float* __restrict__ my_buf,
    const unsigned* __restrict__ my_sync,
    const unsigned* __restrict__ seq_state, const half* __restrict__ gate,
    int64_t gs0, int64_t gs1, half* __restrict__ out, int64_t os0, int64_t os1,
    int m0, int tm, int dcp_rank) {
  const unsigned ep = *seq_state;
  dcp2_wait_peer(my_sync, ep);
  const int slot = static_cast<int>((ep - 1u) & 1u);
  const float* my_slot_base = my_buf + slot * kOSlotTotalFloats;
  const float4* my_o4 = reinterpret_cast<const float4*>(my_slot_base);
  const float* my_lse = my_slot_base + kOSlotOutFloats;

  const int my_h0 = dcp_rank * kHLocal;
  const int tx = threadIdx.x;
  const int pair_in_cta = tx >> 6;
  const int d_vec = tx & 63;
  const int total_pairs = tm * kHLocal;
  const bool io_aligned = ((gs0 & 3) == 0) && ((gs1 & 3) == 0) &&
                          ((os0 & 3) == 0) && ((os1 & 3) == 0);

  for (int pair = blockIdx.x * 4 + pair_in_cta; pair < total_pairs;
       pair += gridDim.x * 4) {
    const int lm = pair / kHLocal;
    const int h = pair % kHLocal;
    const int m = m0 + lm;
    const int mh = my_h0 + h;

    const float4 loc4 = __ldg(reinterpret_cast<const float4*>(
        partial_out + (static_cast<int64_t>(m) * (2 * kHLocal) + mh) * kDim +
        d_vec * 4));
    const float4 peer4 = __ldg(my_o4 + static_cast<int64_t>(pair) * 64 + d_vec);
    const float lse_local =
        __ldg(partial_lse + static_cast<int64_t>(m) * (2 * kHLocal) + mh);
    const float lse_peer = __ldg(my_lse + pair);

    const bool v0 = (lse_local > -1.0e30f);
    const bool v1 = (lse_peer > -1.0e30f);
    float w0 = 0.0f, w1 = 0.0f, inv_w = 0.0f;
    if (v0 && v1) {
      const float mx = fmaxf(lse_local, lse_peer);
      w0 = exp2f(lse_local - mx);
      w1 = exp2f(lse_peer - mx);
      inv_w = w0 + w1;
    }

    const float loc_arr[4] = {loc4.x, loc4.y, loc4.z, loc4.w};
    const float peer_arr[4] = {peer4.x, peer4.y, peer4.z, peer4.w};
    const int64_t g_base = static_cast<int64_t>(m) * gs0 +
                           static_cast<int64_t>(h) * gs1 + d_vec * 4;
    const int64_t o_base = static_cast<int64_t>(m) * os0 +
                           static_cast<int64_t>(h) * os1 + d_vec * 4;

    half gh[4];
    if (io_aligned) {
      *reinterpret_cast<uint2*>(gh) =
          __ldg(reinterpret_cast<const uint2*>(gate + g_base));
    } else {
#pragma unroll
      for (int j = 0; j < 4; ++j) gh[j] = gate[g_base + j];
    }

    half oh[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float val = 0.0f;
      if (v0 && v1) {
        val = __fdiv_rn(loc_arr[j] * w0 + peer_arr[j] * w1, inv_w);
      } else if (v0) {
        val = loc_arr[j];
      } else if (v1) {
        val = peer_arr[j];
      }
      val = __half2float(__float2half_rn(val));
      const float g = __half2float(gh[j]);
      val *= __fdiv_rn(1.0f, 1.0f + fast_exp2(-g * 1.4426950408889634f));
      oh[j] = __float2half_rn(val);
    }

    if (io_aligned) {
      *reinterpret_cast<uint2*>(out + o_base) =
          *reinterpret_cast<const uint2*>(oh);
    } else {
#pragma unroll
      for (int j = 0; j < 4; ++j) out[o_base + j] = oh[j];
    }
  }
}

__device__ __forceinline__ void dcp2_signal_last_cta_chan(
    unsigned* __restrict__ done_counter, unsigned* __restrict__ peer_flag,
    unsigned* __restrict__ seq_word, unsigned ep) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();
    if (atomicAdd(done_counter, 1u) + 1u == gridDim.x) {
      *done_counter = 0u;
      *seq_word = ep;
      __threadfence_system();
      stv_u32(peer_flag, ep);
    }
  }
}

__global__ __launch_bounds__(256) void dcp2_q_push_local_pipelined_kernel(
    const half* __restrict__ query, int64_t qs0, int64_t qs1,
    half* __restrict__ gathered_q, const int64_t* __restrict__ positions,
    const int32_t* __restrict__ lengths, int64_t* __restrict__ local_positions,
    int32_t* __restrict__ local_lengths, half* __restrict__ my_buf,
    unsigned* __restrict__ my_sync, unsigned* __restrict__ peer_sync,
    unsigned* __restrict__ seq_state, int m0, int tm, int M, int num_reqs,
    int dcp_rank, int interleave, bool compute_meta) {
  const unsigned ep = seq_state[0] + 1u;
  if (ep > 2u) {
    dcp2_wait_peer(my_sync + 2, ep - 2u);
  }
  const int tx = threadIdx.x;
  if (compute_meta) {
    const int cycle = 2 * interleave;
    const int rank_off = dcp_rank * interleave;
    for (int i = blockIdx.x * 256 + tx; i < M; i += gridDim.x * 256) {
      int64_t p1 = positions[i] + 1;
      if (p1 < 0) p1 = 0;
      const int64_t rem = p1 % cycle;
      int64_t extra = rem - rank_off;
      if (extra < 0) extra = 0;
      if (extra > interleave) extra = interleave;
      local_positions[i] = (p1 / cycle) * interleave + extra - 1;
    }
    if (blockIdx.x == 0) {
      for (int r = tx; r < num_reqs; r += 256) {
        int32_t len = lengths[r];
        if (len < 0) len = 0;
        const int32_t rem = len % cycle;
        int32_t extra = rem - rank_off;
        if (extra < 0) extra = 0;
        if (extra > interleave) extra = interleave;
        local_lengths[r] = (len / cycle) * interleave + extra;
      }
    }
  }

  const int slot = static_cast<int>((ep - 1u) & 1u);
  uint4* my_slot = reinterpret_cast<uint4*>(my_buf + slot * kQSlotHalves);
  const int my_h0 = dcp_rank * kHLocal;
  const int pair_in_cta = tx >> 5;
  const int d_vec = tx & 31;
  const int total_pairs = tm * kHLocal;
  const bool q_aligned = ((qs0 & 7) == 0) && ((qs1 & 7) == 0);

  for (int pair = blockIdx.x * 8 + pair_in_cta; pair < total_pairs;
       pair += gridDim.x * 8) {
    const int lm = pair / kHLocal;
    const int h = pair % kHLocal;
    const int m = m0 + lm;
    const half* q_src = query + static_cast<int64_t>(m) * qs0 +
                        static_cast<int64_t>(h) * qs1 + d_vec * 8;
    uint4 v;
    if (q_aligned) {
      v = __ldg(reinterpret_cast<const uint4*>(q_src));
    } else {
      half* vh = reinterpret_cast<half*>(&v);
#pragma unroll
      for (int j = 0; j < 8; ++j) vh[j] = q_src[j];
    }
    *reinterpret_cast<uint4*>(
        gathered_q +
        (static_cast<int64_t>(m) * (2 * kHLocal) + my_h0 + h) * kDim +
        d_vec * 8) = v;
    my_slot[static_cast<int64_t>(pair) * 32 + d_vec] = v;
  }
  dcp2_signal_last_cta_chan(my_sync + 1, peer_sync + 0, seq_state + 0, ep);
}

__global__ __launch_bounds__(256) void dcp2_q_unpack_peer_pipelined_kernel(
    const half* __restrict__ peer_buf, unsigned* __restrict__ my_sync,
    unsigned* __restrict__ peer_sync, unsigned* __restrict__ seq_state,
    half* __restrict__ gathered_q, int m0, int tm, int dcp_rank) {
  const unsigned ep = seq_state[1] + 1u;
  dcp2_wait_peer(my_sync + 0, ep);
  const int slot = static_cast<int>((ep - 1u) & 1u);
  const uint4* peer_slot =
      reinterpret_cast<const uint4*>(peer_buf + slot * kQSlotHalves);
  const int peer_h0 = (1 - dcp_rank) * kHLocal;
  const int tx = threadIdx.x;
  const int pair_in_cta = tx >> 5;
  const int d_vec = tx & 31;
  const int total_pairs = tm * kHLocal;

  for (int pair = blockIdx.x * 8 + pair_in_cta; pair < total_pairs;
       pair += gridDim.x * 8) {
    const int lm = pair / kHLocal;
    const int h = pair % kHLocal;
    const int m = m0 + lm;
    const uint4 v =
        p2p_ld_cv_u4(peer_slot + static_cast<int64_t>(pair) * 32 + d_vec);
    *reinterpret_cast<uint4*>(
        gathered_q +
        (static_cast<int64_t>(m) * (2 * kHLocal) + peer_h0 + h) * kDim +
        d_vec * 8) = v;
  }
  dcp2_signal_last_cta_chan(my_sync + 3, peer_sync + 2, seq_state + 1, ep);
}

__global__ __launch_bounds__(256) void dcp2_o_push_fp16_pipelined_kernel(
    const half* __restrict__ partial_out, const float* __restrict__ partial_lse,
    char* __restrict__ my_buf, unsigned* __restrict__ my_sync,
    unsigned* __restrict__ peer_sync, unsigned* __restrict__ seq_state, int m0,
    int tm, int dcp_rank) {
  const unsigned ep = seq_state[0] + 1u;
  if (ep > 2u) {
    dcp2_wait_peer(my_sync + 2, ep - 2u);
  }
  const int slot = static_cast<int>((ep - 1u) & 1u);
  char* my_slot_base = my_buf + slot * kOSlotTotalFp16Bytes;
  uint4* my_o8 = reinterpret_cast<uint4*>(my_slot_base);
  float* my_lse = reinterpret_cast<float*>(my_slot_base + kOSlotOutHalfBytes);

  const int peer_h0 = (1 - dcp_rank) * kHLocal;
  const int tx = threadIdx.x;
  const int pair_in_cta = tx >> 5;
  const int d_vec = tx & 31;
  const int total_pairs = tm * kHLocal;

  for (int pair = blockIdx.x * 8 + pair_in_cta; pair < total_pairs;
       pair += gridDim.x * 8) {
    const int lm = pair / kHLocal;
    const int h = pair % kHLocal;
    const int m = m0 + lm;
    const int ph = peer_h0 + h;
    const uint4 v8 = __ldg(reinterpret_cast<const uint4*>(
        partial_out + (static_cast<int64_t>(m) * (2 * kHLocal) + ph) * kDim +
        d_vec * 8));
    my_o8[static_cast<int64_t>(pair) * 32 + d_vec] = v8;
    if (d_vec == 0) {
      my_lse[pair] = partial_lse[static_cast<int64_t>(m) * (2 * kHLocal) + ph];
    }
  }
  dcp2_signal_last_cta_chan(my_sync + 1, peer_sync + 0, seq_state + 0, ep);
}

__global__
__launch_bounds__(256) void dcp2_o_combine_gate_fp16_pipelined_kernel(
    const half* __restrict__ partial_out, const float* __restrict__ partial_lse,
    const char* __restrict__ peer_buf, unsigned* __restrict__ my_sync,
    unsigned* __restrict__ peer_sync, unsigned* __restrict__ seq_state,
    const half* __restrict__ gate, int64_t gs0, int64_t gs1,
    half* __restrict__ out, int64_t os0, int64_t os1, int m0, int tm,
    int dcp_rank) {
  const unsigned ep = seq_state[1] + 1u;
  dcp2_wait_peer(my_sync + 0, ep);
  const int slot = static_cast<int>((ep - 1u) & 1u);
  const char* peer_slot_base = peer_buf + slot * kOSlotTotalFp16Bytes;
  const uint4* peer_o8 = reinterpret_cast<const uint4*>(peer_slot_base);
  const float* peer_lse =
      reinterpret_cast<const float*>(peer_slot_base + kOSlotOutHalfBytes);

  const int my_h0 = dcp_rank * kHLocal;
  const int tx = threadIdx.x;
  const int pair_in_cta = tx >> 5;
  const int d_vec = tx & 31;
  const int total_pairs = tm * kHLocal;
  const bool io_aligned = ((gs0 & 7) == 0) && ((gs1 & 7) == 0) &&
                          ((os0 & 7) == 0) && ((os1 & 7) == 0);

  for (int pair = blockIdx.x * 8 + pair_in_cta; pair < total_pairs;
       pair += gridDim.x * 8) {
    const int lm = pair / kHLocal;
    const int h = pair % kHLocal;
    const int m = m0 + lm;
    const int mh = my_h0 + h;

    const uint4 loc8 = __ldg(reinterpret_cast<const uint4*>(
        partial_out + (static_cast<int64_t>(m) * (2 * kHLocal) + mh) * kDim +
        d_vec * 8));
    const uint4 peer8 =
        p2p_ld_cv_u4(peer_o8 + static_cast<int64_t>(pair) * 32 + d_vec);
    const float lse_local =
        __ldg(partial_lse + static_cast<int64_t>(m) * (2 * kHLocal) + mh);
    const float lse_peer = p2p_ld_cv_f32(peer_lse + pair);

    const bool v0 = (lse_local > -1.0e30f);
    const bool v1 = (lse_peer > -1.0e30f);
    float w0 = 0.0f, w1 = 0.0f, inv_w = 0.0f;
    if (v0 && v1) {
      const float mx = fmaxf(lse_local, lse_peer);
      w0 = exp2f(lse_local - mx);
      w1 = exp2f(lse_peer - mx);
      inv_w = w0 + w1;
    }

    const half* loc_h = reinterpret_cast<const half*>(&loc8);
    const half* peer_h = reinterpret_cast<const half*>(&peer8);
    const int64_t g_base = static_cast<int64_t>(m) * gs0 +
                           static_cast<int64_t>(h) * gs1 + d_vec * 8;
    const int64_t o_base = static_cast<int64_t>(m) * os0 +
                           static_cast<int64_t>(h) * os1 + d_vec * 8;

    uint4 gu4;
    half* gh = reinterpret_cast<half*>(&gu4);
    if (io_aligned) {
      gu4 = __ldg(reinterpret_cast<const uint4*>(gate + g_base));
    } else {
#pragma unroll
      for (int j = 0; j < 8; ++j) gh[j] = gate[g_base + j];
    }

    uint4 ou4;
    half* oh = reinterpret_cast<half*>(&ou4);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float val = 0.0f;
      if (v0 && v1) {
        val = __fdiv_rn(
            __half2float(loc_h[j]) * w0 + __half2float(peer_h[j]) * w1, inv_w);
      } else if (v0) {
        val = __half2float(loc_h[j]);
      } else if (v1) {
        val = __half2float(peer_h[j]);
      }
      val = __half2float(__float2half_rn(val));
      const float g = __half2float(gh[j]);
      val *= __fdiv_rn(1.0f, 1.0f + fast_exp2(-g * 1.4426950408889634f));
      oh[j] = __float2half_rn(val);
    }

    if (io_aligned) {
      *reinterpret_cast<uint4*>(out + o_base) = ou4;
    } else {
#pragma unroll
      for (int j = 0; j < 8; ++j) out[o_base + j] = oh[j];
    }
  }
  dcp2_signal_last_cta_chan(my_sync + 3, peer_sync + 2, seq_state + 1, ep);
}

namespace {
struct Dcp2AsyncResources {
  bool inited = false;
  cudaStream_t rx_stream = nullptr;
  cudaEvent_t ev_start = nullptr;
  cudaEvent_t ev_done = nullptr;
};
Dcp2AsyncResources g_dcp2_res[16];

Dcp2AsyncResources& get_dcp2_res(int dev) {
  auto& r = g_dcp2_res[dev];
  if (!r.inited) {
    int lo = 0, hi = 0;
    C10_CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
    C10_CUDA_CHECK(
        cudaStreamCreateWithPriority(&r.rx_stream, cudaStreamNonBlocking, hi));
    C10_CUDA_CHECK(
        cudaEventCreateWithFlags(&r.ev_start, cudaEventDisableTiming));
    C10_CUDA_CHECK(
        cudaEventCreateWithFlags(&r.ev_done, cudaEventDisableTiming));
    r.inited = true;
  }
  return r;
}
}  // namespace

void dcp2_q_prep_any_m_out(torch::Tensor query, torch::Tensor gathered_q,
                           torch::Tensor positions, torch::Tensor lengths,
                           torch::Tensor local_positions,
                           torch::Tensor local_lengths, int64_t my_buf_ptr,
                           int64_t peer_buf_ptr, torch::Tensor seq_state,
                           int64_t dcp_rank, int64_t interleave) {
  const c10::cuda::CUDAGuard guard(query.device());
  const int M = static_cast<int>(query.size(0));
  if (M == 0) return;
  const int num_reqs = static_cast<int>(lengths.size(0));
  auto stream = at::cuda::getCurrentCUDAStream();

  half* my_buf = reinterpret_cast<half*>(static_cast<uintptr_t>(my_buf_ptr));
  half* peer_buf =
      reinterpret_cast<half*>(static_cast<uintptr_t>(peer_buf_ptr));
  unsigned* my_sync = reinterpret_cast<unsigned*>(
      static_cast<uintptr_t>(my_buf_ptr) + kQSyncOffsetBytes);
  unsigned* peer_sync = reinterpret_cast<unsigned*>(
      static_cast<uintptr_t>(peer_buf_ptr) + kQSyncOffsetBytes);
  unsigned* seq_ptr =
      reinterpret_cast<unsigned*>(seq_state.data_ptr<int32_t>());

  cudaStreamCaptureStatus cap_status = cudaStreamCaptureStatusNone;
  C10_CUDA_CHECK(cudaStreamIsCapturing(stream.stream(), &cap_status));
  const char* sync_env = std::getenv("FN_DCP2_SYNC");
  const bool use_pipelined = !(sync_env && std::atoi(sync_env) != 0) &&
                             (M > kBulkM) && (seq_state.numel() >= 2) &&
                             (cap_status == cudaStreamCaptureStatusNone);

  if (use_pipelined) {
    auto& res = get_dcp2_res(query.device().index());
    C10_CUDA_CHECK(cudaEventRecord(res.ev_start, stream.stream()));
    C10_CUDA_CHECK(cudaStreamWaitEvent(res.rx_stream, res.ev_start, 0));
    for (int m0 = 0; m0 < M; m0 += kBulkM) {
      const int tm = std::min(kBulkM, M - m0);
      const int local_ctas = std::min((tm * kHLocal + 7) / 8, 80);
      const int pull_ctas = std::min((tm * kHLocal + 7) / 8, 32);
      dcp2_q_push_local_pipelined_kernel<<<local_ctas, 256, 0, stream>>>(
          reinterpret_cast<const half*>(query.data_ptr()), query.stride(0),
          query.stride(1), reinterpret_cast<half*>(gathered_q.data_ptr()),
          positions.data_ptr<int64_t>(), lengths.data_ptr<int32_t>(),
          local_positions.data_ptr<int64_t>(),
          local_lengths.data_ptr<int32_t>(), my_buf, my_sync, peer_sync,
          seq_ptr, m0, tm, M, num_reqs, static_cast<int>(dcp_rank),
          static_cast<int>(interleave), m0 == 0);
      dcp2_q_unpack_peer_pipelined_kernel<<<pull_ctas, 256, 0, res.rx_stream>>>(
          peer_buf, my_sync, peer_sync, seq_ptr,
          reinterpret_cast<half*>(gathered_q.data_ptr()), m0, tm,
          static_cast<int>(dcp_rank));
    }
    C10_CUDA_CHECK(cudaEventRecord(res.ev_done, res.rx_stream));
    C10_CUDA_CHECK(cudaStreamWaitEvent(stream.stream(), res.ev_done, 0));
  } else {
    for (int m0 = 0; m0 < M; m0 += kBulkM) {
      const int tm = std::min(kBulkM, M - m0);
      const int local_ctas = std::min((tm * kHLocal + 7) / 8, 80);
      const int pull_ctas = std::min((tm * kHLocal + 7) / 8, 32);
      dcp2_q_push_local_pipelined_kernel<<<local_ctas, 256, 0, stream>>>(
          reinterpret_cast<const half*>(query.data_ptr()), query.stride(0),
          query.stride(1), reinterpret_cast<half*>(gathered_q.data_ptr()),
          positions.data_ptr<int64_t>(), lengths.data_ptr<int32_t>(),
          local_positions.data_ptr<int64_t>(),
          local_lengths.data_ptr<int32_t>(), my_buf, my_sync, peer_sync,
          seq_ptr, m0, tm, M, num_reqs, static_cast<int>(dcp_rank),
          static_cast<int>(interleave), m0 == 0);
      dcp2_q_unpack_peer_pipelined_kernel<<<pull_ctas, 256, 0, stream>>>(
          peer_buf, my_sync, peer_sync, seq_ptr,
          reinterpret_cast<half*>(gathered_q.data_ptr()), m0, tm,
          static_cast<int>(dcp_rank));
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dcp2_combine_gate_any_m_out(torch::Tensor partial_out,
                                 torch::Tensor partial_lse, torch::Tensor gate,
                                 torch::Tensor out, int64_t my_buf_ptr,
                                 int64_t peer_buf_ptr, torch::Tensor seq_state,
                                 int64_t dcp_rank) {
  const c10::cuda::CUDAGuard guard(partial_out.device());
  const int M = static_cast<int>(partial_out.size(0));
  if (M == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream();
  unsigned* seq_ptr =
      reinterpret_cast<unsigned*>(seq_state.data_ptr<int32_t>());

  if (partial_out.scalar_type() == at::kHalf) {
    char* my_buf = reinterpret_cast<char*>(static_cast<uintptr_t>(my_buf_ptr));
    char* peer_buf =
        reinterpret_cast<char*>(static_cast<uintptr_t>(peer_buf_ptr));
    unsigned* my_sync = reinterpret_cast<unsigned*>(
        static_cast<uintptr_t>(my_buf_ptr) + kOSyncOffsetFp16Bytes);
    unsigned* peer_sync = reinterpret_cast<unsigned*>(
        static_cast<uintptr_t>(peer_buf_ptr) + kOSyncOffsetFp16Bytes);
    cudaStreamCaptureStatus cap_status = cudaStreamCaptureStatusNone;
    C10_CUDA_CHECK(cudaStreamIsCapturing(stream.stream(), &cap_status));
    const char* sync_env = std::getenv("FN_DCP2_SYNC");
    const bool use_pipelined = !(sync_env && std::atoi(sync_env) != 0) &&
                               (M > kBulkM) && (seq_state.numel() >= 2) &&
                               (cap_status == cudaStreamCaptureStatusNone);
    if (use_pipelined) {
      auto& res = get_dcp2_res(partial_out.device().index());
      C10_CUDA_CHECK(cudaEventRecord(res.ev_start, stream.stream()));
      C10_CUDA_CHECK(cudaStreamWaitEvent(res.rx_stream, res.ev_start, 0));
      for (int m0 = 0; m0 < M; m0 += kBulkM) {
        const int tm = std::min(kBulkM, M - m0);
        const int local_ctas = std::min((tm * kHLocal + 7) / 8, 80);
        const int pull_ctas = std::min((tm * kHLocal + 7) / 8, 32);
        dcp2_o_push_fp16_pipelined_kernel<<<local_ctas, 256, 0, stream>>>(
            reinterpret_cast<const half*>(partial_out.data_ptr()),
            partial_lse.data_ptr<float>(), my_buf, my_sync, peer_sync, seq_ptr,
            m0, tm, static_cast<int>(dcp_rank));
        dcp2_o_combine_gate_fp16_pipelined_kernel<<<pull_ctas, 256, 0,
                                                    res.rx_stream>>>(
            reinterpret_cast<const half*>(partial_out.data_ptr()),
            partial_lse.data_ptr<float>(), peer_buf, my_sync, peer_sync,
            seq_ptr, reinterpret_cast<const half*>(gate.data_ptr()),
            gate.stride(0), gate.stride(1),
            reinterpret_cast<half*>(out.data_ptr()), out.stride(0),
            out.stride(1), m0, tm, static_cast<int>(dcp_rank));
      }
      C10_CUDA_CHECK(cudaEventRecord(res.ev_done, res.rx_stream));
      C10_CUDA_CHECK(cudaStreamWaitEvent(stream.stream(), res.ev_done, 0));
    } else {
      for (int m0 = 0; m0 < M; m0 += kBulkM) {
        const int tm = std::min(kBulkM, M - m0);
        const int local_ctas = std::min((tm * kHLocal + 7) / 8, 80);
        const int pull_ctas = std::min((tm * kHLocal + 7) / 8, 32);
        dcp2_o_push_fp16_pipelined_kernel<<<local_ctas, 256, 0, stream>>>(
            reinterpret_cast<const half*>(partial_out.data_ptr()),
            partial_lse.data_ptr<float>(), my_buf, my_sync, peer_sync, seq_ptr,
            m0, tm, static_cast<int>(dcp_rank));
        dcp2_o_combine_gate_fp16_pipelined_kernel<<<pull_ctas, 256, 0,
                                                    stream>>>(
            reinterpret_cast<const half*>(partial_out.data_ptr()),
            partial_lse.data_ptr<float>(), peer_buf, my_sync, peer_sync,
            seq_ptr, reinterpret_cast<const half*>(gate.data_ptr()),
            gate.stride(0), gate.stride(1),
            reinterpret_cast<half*>(out.data_ptr()), out.stride(0),
            out.stride(1), m0, tm, static_cast<int>(dcp_rank));
      }
    }
  } else {
    float* my_buf =
        reinterpret_cast<float*>(static_cast<uintptr_t>(my_buf_ptr));
    float* peer_buf =
        reinterpret_cast<float*>(static_cast<uintptr_t>(peer_buf_ptr));
    unsigned* my_sync = reinterpret_cast<unsigned*>(
        static_cast<uintptr_t>(my_buf_ptr) + kOSyncOffsetBytes);
    unsigned* peer_sync = reinterpret_cast<unsigned*>(
        static_cast<uintptr_t>(peer_buf_ptr) + kOSyncOffsetBytes);
    for (int m0 = 0; m0 < M; m0 += kBulkM) {
      const int tm = std::min(kBulkM, M - m0);
      const int ctas = std::min((tm * kHLocal + 3) / 4, 160);
      dcp2_o_push_kernel<<<ctas, 256, 0, stream>>>(
          partial_out.data_ptr<float>(), partial_lse.data_ptr<float>(),
          peer_buf, my_sync, peer_sync, seq_ptr, m0, tm,
          static_cast<int>(dcp_rank));
      dcp2_o_combine_gate_kernel<<<ctas, 256, 0, stream>>>(
          partial_out.data_ptr<float>(), partial_lse.data_ptr<float>(), my_buf,
          my_sync, seq_ptr, reinterpret_cast<const half*>(gate.data_ptr()),
          gate.stride(0), gate.stride(1),
          reinterpret_cast<half*>(out.data_ptr()), out.stride(0), out.stride(1),
          m0, tm, static_cast<int>(dcp_rank));
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace dcp2_any_m

// ============================================================================
// Part 6: Fused QSA Staged K/V Writer + GPU HBM Multi-Page Cache Copy
// ============================================================================

namespace qsa_stage_write {

__device__ __forceinline__ uint8_t f32_to_e4m3fn_bits(float v) {
  const uint32_t bits = __float_as_uint(v);
  const uint32_t sign = (bits >> 24) & 0x80u;
  const float av = fabsf(v);
  if (!(av > 0.f)) return static_cast<uint8_t>(sign);
  if (av >= 448.f) return static_cast<uint8_t>(sign | 0x7eu);
  if (av < 0x1p-9f) {
    const float r = rintf(av * 512.f);
    const uint32_t m = min(static_cast<uint32_t>(r), 7u);
    return static_cast<uint8_t>(sign | m);
  }
  int exp_unbiased = (static_cast<int>((bits >> 23) & 0xffu)) - 127;
  uint32_t mant = bits & 0x7fffffu;
  if (exp_unbiased < -6) {
    const float r = rintf(av * 512.f);
    const uint32_t m = min(static_cast<uint32_t>(r), 8u);
    return static_cast<uint8_t>(sign | m);
  }
  uint32_t round_bit = (mant >> 19) & 1u;
  uint32_t sticky = (mant & 0x7ffffu) != 0u;
  uint32_t m3 = (mant >> 20) & 7u;
  if (round_bit && (sticky || (m3 & 1u))) {
    ++m3;
    if (m3 == 8u) {
      m3 = 0u;
      ++exp_unbiased;
    }
  }
  int e4 = exp_unbiased + 7;
  if (e4 > 15 || (e4 == 15 && m3 == 7u)) {
    return static_cast<uint8_t>(sign | 0x7eu);
  }
  return static_cast<uint8_t>(sign | (static_cast<uint32_t>(e4) << 3) | m3);
}

__device__ __forceinline__ float e4m3fn_bits_to_f32(uint8_t b) {
  const uint32_t sign = static_cast<uint32_t>(b & 0x80u) << 24;
  const uint32_t exp = (b >> 3) & 0x0fu;
  const uint32_t mant = b & 0x07u;
  if (exp == 0u) {
    if (mant == 0u) return __uint_as_float(sign);
    const float v = static_cast<float>(mant) * (1.0f / 512.0f);
    return (b & 0x80u) ? -v : v;
  }
  if (exp == 15u && mant == 7u) return __uint_as_float(sign | 0x7fc00000u);
  const uint32_t fbits = sign | ((exp + 120u) << 23) | (mant << 20);
  return __uint_as_float(fbits);
}

__global__ __launch_bounds__(256) void init_neg1_kernel(
    int32_t* __restrict__ arr, int n) {
  for (int i = blockIdx.x * 256 + threadIdx.x; i < n; i += gridDim.x * 256) {
    arr[i] = -1;
  }
}

__global__ __launch_bounds__(256) void scatter_staged_map_kernel(
    int32_t* __restrict__ staged_of, const int32_t* __restrict__ pages, int k,
    int num_blocks) {
  for (int i = blockIdx.x * 256 + threadIdx.x; i < k; i += gridDim.x * 256) {
    const int p = pages[i];
    if (p >= 0 && p < num_blocks) staged_of[p] = i;
  }
}

template <typename SlotT, int MODE>
__global__ __launch_bounds__(64) void write_staged_kv_kernel(
    const half* __restrict__ key, int64_t ks0, const half* __restrict__ val,
    int64_t vs0, const SlotT* __restrict__ slots,
    const int32_t* __restrict__ staged_of, void* __restrict__ history,
    float* __restrict__ scales, const int32_t* __restrict__ tags,
    const int32_t* __restrict__ page_slots, void* __restrict__ hot_values,
    float* __restrict__ hot_scales, bool hot_fp8, int num_blocks, int page_size,
    int k_pages) {
  const int row = blockIdx.x;
  const int64_t src_slot = static_cast<int64_t>(slots[row]);
  if (src_slot < 0) return;
  const int64_t src_blk = src_slot / page_size;
  if (src_blk >= num_blocks) return;
  const int32_t dst_blk = staged_of[src_blk];
  if (dst_blk < 0 || dst_blk >= k_pages) return;
  const int64_t token = src_slot % page_size;
  const int64_t dst_slot = static_cast<int64_t>(dst_blk) * page_size + token;

  const int kv = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const half* ptr = (kv == 0) ? (key + row * ks0) : (val + row * vs0);
  const uint4 raw = *reinterpret_cast<const uint4*>(ptr + lane * 8);
  const half* h = reinterpret_cast<const half*>(&raw);

  int32_t hot_res = -1;
  if (hot_values != nullptr && tags != nullptr && page_slots != nullptr) {
    const int32_t page4 = static_cast<int32_t>(src_slot >> 2);
    const int32_t res = page_slots[page4];
    if (res >= 0 && tags[res] == page4) {
      hot_res = res;
    }
  }
  const int64_t hot_tok_idx =
      (hot_res >= 0)
          ? ((static_cast<int64_t>(hot_res) * 4 + (src_slot & 3)) * 2 + kv)
          : -1;

  if constexpr (MODE == 0) {
    half* dst =
        static_cast<half*>(history) +
        ((static_cast<int64_t>(dst_blk) * 2 + kv) * page_size + token) * 256 +
        lane * 8;
    *reinterpret_cast<uint4*>(dst) = raw;
    if (hot_tok_idx >= 0) {
      *reinterpret_cast<uint4*>(static_cast<half*>(hot_values) +
                                hot_tok_idx * 256 + lane * 8) = raw;
    }
  } else {
    float vf[8];
    float amax = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      vf[j] = __half2float(h[j]);
      amax = fmaxf(amax, fabsf(vf[j]));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    }
    if constexpr (MODE == 1) {
      const float sc = fmaxf(amax * (1.0f / 127.0f), 1.0e-8f);
      if (lane == 0) scales[dst_slot * 2 + kv] = sc;
      const float inv = 1.0f / sc;
      int8_t codes[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float q = fminf(fmaxf(rintf(vf[j] * inv), -127.f), 127.f);
        codes[j] = static_cast<int8_t>(q);
      }
      int8_t* dst =
          static_cast<int8_t*>(history) +
          ((static_cast<int64_t>(dst_blk) * 2 + kv) * page_size + token) * 256 +
          lane * 8;
      *reinterpret_cast<uint2*>(dst) = *reinterpret_cast<const uint2*>(codes);
      if (hot_tok_idx >= 0) {
        if (hot_fp8) {
          if (lane == 0 && hot_scales != nullptr) {
            hot_scales[hot_tok_idx] = sc;
          }
          *reinterpret_cast<uint2*>(static_cast<int8_t*>(hot_values) +
                                    hot_tok_idx * 256 + lane * 8) =
              *reinterpret_cast<const uint2*>(codes);
        } else {
          half dec[8];
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            dec[j] = __float2half_rn(static_cast<float>(codes[j]) * sc);
          }
          *reinterpret_cast<uint4*>(static_cast<half*>(hot_values) +
                                    hot_tok_idx * 256 + lane * 8) =
              *reinterpret_cast<const uint4*>(dec);
        }
      }
    } else {
      const float sc = fmaxf(__fdiv_rn(amax, 448.0f), 0x1p-126f);
      if (lane == 0) scales[dst_slot * 2 + kv] = sc;
      uint8_t codes[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        codes[j] = f32_to_e4m3fn_bits(__fdiv_rn(vf[j], sc));
      }
      uint8_t* dst =
          static_cast<uint8_t*>(history) +
          ((static_cast<int64_t>(dst_blk) * 2 + kv) * page_size + token) * 256 +
          lane * 8;
      *reinterpret_cast<uint2*>(dst) = *reinterpret_cast<const uint2*>(codes);
      if (hot_tok_idx >= 0) {
        if (hot_fp8) {
          if (lane == 0 && hot_scales != nullptr) {
            hot_scales[hot_tok_idx] = sc;
          }
          *reinterpret_cast<uint2*>(static_cast<uint8_t*>(hot_values) +
                                    hot_tok_idx * 256 + lane * 8) =
              *reinterpret_cast<const uint2*>(codes);
        } else {
          half dec[8];
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            dec[j] = __float2half_rn(e4m3fn_bits_to_f32(codes[j]) * sc);
          }
          *reinterpret_cast<uint4*>(static_cast<half*>(hot_values) +
                                    hot_tok_idx * 256 + lane * 8) =
              *reinterpret_cast<const uint4*>(dec);
        }
      }
    }
  }
}

__global__ __launch_bounds__(256, 4) void copy_qsa_pages_kernel(
    uint4* __restrict__ dst_hist, float4* __restrict__ dst_scales,
    const uint4* __restrict__ src_hist, const float4* __restrict__ src_scales,
    const int32_t* __restrict__ dst_pages,
    const int32_t* __restrict__ src_pages, int hist_vec_per_page,
    int scale_vec_per_page, int num_pages) {
  for (int p = blockIdx.x; p < num_pages; p += gridDim.x) {
    const int dp = dst_pages[p];
    const int sp = src_pages[p];
    if (dp < 0 || sp < 0) continue;

    uint4* dh = dst_hist + static_cast<int64_t>(dp) * hist_vec_per_page;
    const uint4* sh = src_hist + static_cast<int64_t>(sp) * hist_vec_per_page;
    for (int i = threadIdx.x; i < hist_vec_per_page; i += 256) {
      dh[i] = __ldg(sh + i);
    }
    if (scale_vec_per_page > 0 && dst_scales != nullptr &&
        src_scales != nullptr) {
      float4* ds = dst_scales + static_cast<int64_t>(dp) * scale_vec_per_page;
      const float4* ss =
          src_scales + static_cast<int64_t>(sp) * scale_vec_per_page;
      for (int i = threadIdx.x; i < scale_vec_per_page; i += 256) {
        ds[i] = __ldg(ss + i);
      }
    }
  }
}

void build_staged_map_out(torch::Tensor staged_of, torch::Tensor pages) {
  const c10::cuda::CUDAGuard guard(staged_of.device());
  const int num_blocks = static_cast<int>(staged_of.numel());
  const int k = static_cast<int>(pages.numel());
  auto stream = at::cuda::getCurrentCUDAStream();
  init_neg1_kernel<<<std::min((num_blocks + 255) / 256, 160), 256, 0, stream>>>(
      staged_of.data_ptr<int32_t>(), num_blocks);
  if (k > 0) {
    scatter_staged_map_kernel<<<std::min((k + 255) / 256, 160), 256, 0,
                                stream>>>(staged_of.data_ptr<int32_t>(),
                                          pages.data_ptr<int32_t>(), k,
                                          num_blocks);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void write_staged_kv_out(torch::Tensor key, torch::Tensor val,
                         torch::Tensor slots, torch::Tensor staged_of,
                         torch::Tensor history, torch::Tensor scales,
                         std::optional<torch::Tensor> tags,
                         std::optional<torch::Tensor> page_slots,
                         std::optional<torch::Tensor> hot_values,
                         std::optional<torch::Tensor> hot_scales,
                         bool hot_fp8) {
  const c10::cuda::CUDAGuard guard(key.device());
  const int M = static_cast<int>(slots.numel());
  const int k_pages = static_cast<int>(history.size(0));
  if (M == 0 || k_pages == 0) return;
  const int page_size = static_cast<int>(history.size(2));
  const int num_blocks = static_cast<int>(staged_of.numel());
  const int mode = (history.scalar_type() == at::kChar)
                       ? 1
                       : ((history.scalar_type() == at::kByte) ? 2 : 0);
  auto stream = at::cuda::getCurrentCUDAStream();
  const half* kp = reinterpret_cast<const half*>(key.data_ptr());
  const half* vp = reinterpret_cast<const half*>(val.data_ptr());
  const int32_t* map_p = staged_of.data_ptr<int32_t>();
  float* sc_p = scales.data_ptr<float>();
  const int32_t* tags_p = (tags.has_value() && tags->defined())
                              ? tags->data_ptr<int32_t>()
                              : nullptr;
  const int32_t* ps_p = (page_slots.has_value() && page_slots->defined())
                            ? page_slots->data_ptr<int32_t>()
                            : nullptr;
  void* hv_p = (hot_values.has_value() && hot_values->defined())
                   ? hot_values->data_ptr()
                   : nullptr;
  float* hs_p = (hot_scales.has_value() && hot_scales->defined() &&
                 hot_scales->numel() > 0)
                    ? hot_scales->data_ptr<float>()
                    : nullptr;

#define LAUNCH_WS(SLOT_T, M_CONST)                                             \
  write_staged_kv_kernel<SLOT_T, M_CONST><<<M, 64, 0, stream>>>(               \
      kp, key.stride(0), vp, val.stride(0), slots.data_ptr<SLOT_T>(), map_p,   \
      history.data_ptr(), sc_p, tags_p, ps_p, hv_p, hs_p, hot_fp8, num_blocks, \
      page_size, k_pages)

  if (slots.scalar_type() == at::kLong) {
    if (mode == 1)
      LAUNCH_WS(int64_t, 1);
    else if (mode == 2)
      LAUNCH_WS(int64_t, 2);
    else
      LAUNCH_WS(int64_t, 0);
  } else {
    if (mode == 1)
      LAUNCH_WS(int32_t, 1);
    else if (mode == 2)
      LAUNCH_WS(int32_t, 2);
    else
      LAUNCH_WS(int32_t, 0);
  }
#undef LAUNCH_WS
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void copy_qsa_pages_out(torch::Tensor dst_hist,
                        std::optional<torch::Tensor> dst_scales,
                        torch::Tensor src_hist,
                        std::optional<torch::Tensor> src_scales,
                        torch::Tensor dst_pages, torch::Tensor src_pages,
                        int64_t max_ctas = 0) {
  const c10::cuda::CUDAGuard guard(dst_hist.device());
  const int n = static_cast<int>(dst_pages.numel());
  if (n == 0) return;
  const int page_size = static_cast<int>(dst_hist.size(2));
  const int64_t page_bytes =
      2LL * page_size * 256LL * static_cast<int64_t>(dst_hist.element_size());
  const int hist_vec_per_page = static_cast<int>(page_bytes / 16);
  const bool has_scales = dst_scales.has_value() && dst_scales->defined() &&
                          src_scales.has_value() && src_scales->defined() &&
                          dst_scales->numel() >= page_size * 2;
  const int scale_vec_per_page = has_scales ? (page_size * 2) / 4 : 0;
  const int grid = (max_ctas > 0) ? std::min(n, static_cast<int>(max_ctas))
                                  : std::min(n, 256);
  auto stream = at::cuda::getCurrentCUDAStream();
  copy_qsa_pages_kernel<<<grid, 256, 0, stream>>>(
      reinterpret_cast<uint4*>(dst_hist.data_ptr()),
      has_scales ? reinterpret_cast<float4*>(dst_scales->data_ptr()) : nullptr,
      reinterpret_cast<const uint4*>(src_hist.data_ptr()),
      has_scales ? reinterpret_cast<const float4*>(src_scales->data_ptr())
                 : nullptr,
      dst_pages.data_ptr<int32_t>(), src_pages.data_ptr<int32_t>(),
      hist_vec_per_page, scale_vec_per_page, n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void async_persist_qsa_page_out(torch::Tensor host_hist,
                                std::optional<torch::Tensor> host_scales,
                                torch::Tensor gpu_hist,
                                std::optional<torch::Tensor> gpu_scales,
                                int64_t phys_blk, int64_t cache_slot) {
  const c10::cuda::CUDAGuard guard(gpu_hist.device());
  const int64_t page_size = gpu_hist.size(2);
  const size_t page_bytes =
      static_cast<size_t>(2LL * page_size * 256LL * gpu_hist.element_size());
  auto stream = at::cuda::getCurrentCUDAStream();
  const char* src_h =
      static_cast<const char*>(gpu_hist.data_ptr()) + cache_slot * page_bytes;
  char* dst_h =
      static_cast<char*>(host_hist.data_ptr()) + phys_blk * page_bytes;
  C10_CUDA_CHECK(
      cudaMemcpyAsync(dst_h, src_h, page_bytes, cudaMemcpyDefault, stream));
  if (host_scales.has_value() && host_scales->defined() &&
      gpu_scales.has_value() && gpu_scales->defined() &&
      host_scales->numel() >= page_size * 2) {
    const size_t sc_bytes =
        static_cast<size_t>(page_size * 2LL * sizeof(float));
    const char* src_s = static_cast<const char*>(gpu_scales->data_ptr()) +
                        cache_slot * sc_bytes;
    char* dst_s =
        static_cast<char*>(host_scales->data_ptr()) + phys_blk * sc_bytes;
    C10_CUDA_CHECK(
        cudaMemcpyAsync(dst_s, src_s, sc_bytes, cudaMemcpyDefault, stream));
  }
}

void async_persist_qsa_pages_batch_out(torch::Tensor host_hist,
                                       std::optional<torch::Tensor> host_scales,
                                       torch::Tensor gpu_hist,
                                       std::optional<torch::Tensor> gpu_scales,
                                       std::vector<int64_t> phys_blks,
                                       std::vector<int64_t> cache_slots) {
  const c10::cuda::CUDAGuard guard(gpu_hist.device());
  const size_t n = phys_blks.size();
  if (n == 0 || cache_slots.size() != n) return;
  const int64_t page_size = gpu_hist.size(2);
  const size_t page_bytes =
      static_cast<size_t>(2LL * page_size * 256LL * gpu_hist.element_size());
  const bool has_scales = host_scales.has_value() && host_scales->defined() &&
                          gpu_scales.has_value() && gpu_scales->defined() &&
                          host_scales->numel() >= page_size * 2;
  const size_t sc_bytes = static_cast<size_t>(page_size * 2LL * sizeof(float));
  auto stream = at::cuda::getCurrentCUDAStream();
  const char* src_h_base = static_cast<const char*>(gpu_hist.data_ptr());
  char* dst_h_base = static_cast<char*>(host_hist.data_ptr());
  const char* src_s_base =
      has_scales ? static_cast<const char*>(gpu_scales->data_ptr()) : nullptr;
  char* dst_s_base =
      has_scales ? static_cast<char*>(host_scales->data_ptr()) : nullptr;

  size_t i = 0;
  while (i < n) {
    size_t run = 1;
    while (i + run < n &&
           phys_blks[i + run] == phys_blks[i] + static_cast<int64_t>(run) &&
           cache_slots[i + run] == cache_slots[i] + static_cast<int64_t>(run)) {
      ++run;
    }
    C10_CUDA_CHECK(cudaMemcpyAsync(dst_h_base + phys_blks[i] * page_bytes,
                                   src_h_base + cache_slots[i] * page_bytes,
                                   run * page_bytes, cudaMemcpyDefault,
                                   stream));
    if (has_scales) {
      C10_CUDA_CHECK(cudaMemcpyAsync(dst_s_base + phys_blks[i] * sc_bytes,
                                     src_s_base + cache_slots[i] * sc_bytes,
                                     run * sc_bytes, cudaMemcpyDefault,
                                     stream));
    }
    i += run;
  }
}

__global__ __launch_bounds__(32) void inc_hot_epoch_kernel(
    int32_t* __restrict__ epoch) {
  if (threadIdx.x == 0) {
    const int32_t v = *epoch;
    *epoch = (v <= 0) ? 2 : (v + 1);
  }
}

}  // namespace qsa_stage_write

// ============================================================================
// Part 7: 3-Link Parallel NVLink P2P TP4 HyperConnection Prefill Kernels
// ============================================================================

namespace hc_p2p {

__global__ __launch_bounds__(32) void hc_p2p_signal_only_kernel(
    PeerPtrs4 sync_ptrs, unsigned* __restrict__ local_ep_ptr, int my_rank) {
  const int tx = threadIdx.x;
  if (tx >= 4) return;
  const unsigned ep = *local_ep_ptr + 1u;
  __threadfence_system();
  if (tx == 0) {
    *local_ep_ptr = ep;
  }
  unsigned* peer_flag = reinterpret_cast<unsigned*>(sync_ptrs.p[tx]) + my_rank;
  p2p_st_u32(peer_flag, ep);
}

__global__ __launch_bounds__(32) void hc_p2p_barrier_kernel(
    PeerPtrs4 sync_ptrs, unsigned* __restrict__ local_ep_ptr, int my_rank) {
  const int tx = threadIdx.x;
  if (tx >= 4) return;
  const unsigned ep = *local_ep_ptr + 1u;
  __threadfence_system();
  unsigned* peer_flag = reinterpret_cast<unsigned*>(sync_ptrs.p[tx]) + my_rank;
  p2p_st_u32(peer_flag, ep);

  const unsigned* my_flag =
      reinterpret_cast<const unsigned*>(sync_ptrs.p[my_rank]) + tx;
  const long long t0 = clock64();
  while (p2p_ld_u32(my_flag) < ep) {
    if (clock64() - t0 > 8000000000LL) __trap();
  }
  __syncwarp(0xfu);
  if (tx == 0) {
    *local_ep_ptr = ep;
  }
}

template <bool SUM2>
__global__ __launch_bounds__(256) void hc_stage_partials_f16_v3_kernel(
    half* __restrict__ dst_base, PeerPtrs4 sync_ptrs,
    unsigned* __restrict__ ep_ptr, int my_rank, int64_t bank_elems,
    const half* __restrict__ first, const half* __restrict__ second,
    int valid_rows, int padded_rows) {
  const int slot = static_cast<int>(*ep_ptr & 1u);
  half* dst = dst_base + static_cast<int64_t>(slot) * bank_elems;
  const int64_t total_vec = static_cast<int64_t>(padded_rows) * 320;
  const int64_t valid_vec = static_cast<int64_t>(valid_rows) * 320;
  for (int64_t v = static_cast<int64_t>(blockIdx.x) * 256 + threadIdx.x;
       v < total_vec; v += static_cast<int64_t>(gridDim.x) * 256) {
    uint4 val = make_uint4(0, 0, 0, 0);
    if (v < valid_vec) {
      if constexpr (!SUM2) {
        val = __ldg(reinterpret_cast<const uint4*>(first + v * 8));
      } else {
        const uint4 a = __ldg(reinterpret_cast<const uint4*>(first + v * 8));
        const uint4 b = __ldg(reinterpret_cast<const uint4*>(second + v * 8));
        const half* ah = reinterpret_cast<const half*>(&a);
        const half* bh = reinterpret_cast<const half*>(&b);
        half* vh = reinterpret_cast<half*>(&val);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          vh[j] = __float2half_rn((__half2float(ah[j]) + __half2float(bh[j])) *
                                  0.125f);
        }
      }
    }
    *reinterpret_cast<uint4*>(dst + v * 8) = val;
  }
  hc_p2p_signal_last_cta(sync_ptrs, ep_ptr, my_rank);
}

template <bool SCALED8, bool NORMALIZE>
__global__ __launch_bounds__(320, 2) void hc_p2p_reduce_combine_norm_kernel(
    const half* __restrict__ ptr0, const half* __restrict__ ptr1,
    const half* __restrict__ ptr2, const half* __restrict__ ptr3,
    uintptr_t my_sync_ptr, const unsigned* __restrict__ ep_ptr,
    int64_t bank_elems, const half* __restrict__ hidden,
    const half* __restrict__ inj, const half* __restrict__ weight,
    half* __restrict__ updated, half* __restrict__ normalized, int my_rank,
    int local_rows, bool w_shared, float eps) {
  __shared__ uint4 s_w[4][320];
  __shared__ float s_warp_sq[4][10];
  __shared__ float s_rrms[4];

  const int tx = threadIdx.x;
  const int c8 = tx * 8;
  if constexpr (NORMALIZE) {
    if (w_shared) {
      const uint4 sw = __ldg(reinterpret_cast<const uint4*>(weight + c8));
      s_w[0][tx] = sw;
      s_w[1][tx] = sw;
      s_w[2][tx] = sw;
      s_w[3][tx] = sw;
    } else {
#pragma unroll
      for (int stream = 0; stream < 4; ++stream) {
        s_w[stream][tx] =
            __ldg(reinterpret_cast<const uint4*>(weight + stream * 2560 + c8));
      }
    }
  }

  const unsigned ep = *ep_ptr;
  hc_p2p_wait_peers(my_sync_ptr, ep);

  const int slot = static_cast<int>((ep - 1u) & 1u);
  int row = blockIdx.x;
  if (row >= local_rows) return;

  uint4 u0, u1, u2, u3;
  {
    const int global_row = my_rank * local_rows + row;
    const int64_t off = static_cast<int64_t>(slot) * bank_elems +
                        static_cast<int64_t>(global_row) * 2560 + c8;
    u0 = __ldg(reinterpret_cast<const uint4*>(ptr0 + off));
    u1 = p2p_ld_cv_u4(reinterpret_cast<const uint4*>(ptr1 + off));
    u2 = p2p_ld_cv_u4(reinterpret_cast<const uint4*>(ptr2 + off));
    u3 = p2p_ld_cv_u4(reinterpret_cast<const uint4*>(ptr3 + off));
  }

  for (; row < local_rows; row += gridDim.x) {
    const uint2 inj_u2 = __ldg(reinterpret_cast<const uint2*>(inj + row * 4));

    const half* h0 = reinterpret_cast<const half*>(&u0);
    const half* h1 = reinterpret_cast<const half*>(&u1);
    const half* h2 = reinterpret_cast<const half*>(&u2);
    const half* h3 = reinterpret_cast<const half*>(&u3);
    float blk[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float sum = (__half2float(h0[j]) + __half2float(h1[j])) +
                        (__half2float(h2[j]) + __half2float(h3[j]));
      const float s = SCALED8 ? (sum * 8.0f) : sum;
      blk[j] = __half2float(__float2half_rn(s));
    }

    const half* inj_h = reinterpret_cast<const half*>(&inj_u2);

    uint4 upd_u4[4];
    float sq[4] = {};
#pragma unroll
    for (int stream = 0; stream < 4; ++stream) {
      const float factor =
          2.0f / (1.0f + expf(-__half2float(inj_h[stream]) * 0.25f));
      const int64_t h_off =
          (static_cast<int64_t>(row) * 4 + stream) * 2560 + c8;
      const uint4 res_v = __ldg(reinterpret_cast<const uint4*>(hidden + h_off));
      const half* res_h = reinterpret_cast<const half*>(&res_v);
      half* uh = reinterpret_cast<half*>(&upd_u4[stream]);
      float s_sq = 0.f;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const half hv =
            __float2half_rn(__half2float(res_h[j]) + factor * blk[j]);
        uh[j] = hv;
        if constexpr (NORMALIZE) {
          const float fv = __half2float(hv);
          s_sq = __fmaf_rn(fv, fv, s_sq);
        }
      }
      *reinterpret_cast<uint4*>(updated + h_off) = upd_u4[stream];
      sq[stream] = s_sq;
    }

    // Issue next row's 3-link NVLink peer loads BEFORE __syncthreads() so
    // NVLink stays 100% saturated across row barriers and normalized HBM
    // stores!
    const int next_row = row + gridDim.x;
    if (next_row < local_rows) {
      const int next_glb = my_rank * local_rows + next_row;
      const int64_t n_off = static_cast<int64_t>(slot) * bank_elems +
                            static_cast<int64_t>(next_glb) * 2560 + c8;
      u0 = __ldg(reinterpret_cast<const uint4*>(ptr0 + n_off));
      u1 = p2p_ld_cv_u4(reinterpret_cast<const uint4*>(ptr1 + n_off));
      u2 = p2p_ld_cv_u4(reinterpret_cast<const uint4*>(ptr2 + n_off));
      u3 = p2p_ld_cv_u4(reinterpret_cast<const uint4*>(ptr3 + n_off));
    }

    if constexpr (NORMALIZE) {
      const int lane = tx & 31;
      const int warp = tx >> 5;
#pragma unroll
      for (int stream = 0; stream < 4; ++stream) {
        float v_sq = sq[stream];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
          v_sq += __shfl_xor_sync(0xffffffffu, v_sq, o);
        }
        if (lane == 0) s_warp_sq[stream][warp] = v_sq;
      }
      __syncthreads();

      if (tx < 4) {
        float tot = 0.f;
#pragma unroll
        for (int w = 0; w < 10; ++w) tot += s_warp_sq[tx][w];
        s_rrms[tx] = rsqrtf(tot * (1.0f / 2560.0f) + eps);
      }
      __syncthreads();

#pragma unroll
      for (int stream = 0; stream < 4; ++stream) {
        const float rrms = s_rrms[stream];
        const uint4 w_v = s_w[stream][tx];
        const half* wh = reinterpret_cast<const half*>(&w_v);
        const half* uh = reinterpret_cast<const half*>(&upd_u4[stream]);
        uint4 norm_u4;
        half* nh = reinterpret_cast<half*>(&norm_u4);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          float y = __half2float(uh[j]) * rrms;
          y = __fmaf_rn(y, __half2float(wh[j]), y);
          nh[j] = __float2half_rn(y);
        }
        const int64_t h_off =
            (static_cast<int64_t>(row) * 4 + stream) * 2560 + c8;
        *reinterpret_cast<uint4*>(normalized + h_off) = norm_u4;
      }
      if (next_row < local_rows) {
        __syncthreads();
      }
    }
  }
}

void hc_p2p_reduce_combine_norm_out(
    torch::Tensor updated, torch::Tensor normalized, torch::Tensor hidden,
    torch::Tensor first, std::optional<torch::Tensor> second,
    torch::Tensor injection, torch::Tensor weight, double eps, bool normalize,
    int64_t p0, int64_t p1, int64_t p2, int64_t p3, int64_t s0, int64_t s1,
    int64_t s2, int64_t s3, torch::Tensor ep_state, int64_t bank_bytes,
    int64_t my_rank, bool skip_stage = false, bool pre_scaled8 = false,
    bool already_signaled = false) {
  const c10::cuda::CUDAGuard guard(hidden.device());
  const int local_rows = static_cast<int>(hidden.size(0));
  if (local_rows == 0) return;
  const int valid_rows = static_cast<int>(first.size(0));
  const int padded_rows = 4 * local_rows;
  const bool has_second = second.has_value() && second->defined();
  auto stream = at::cuda::getCurrentCUDAStream();

  PeerPtrs4 rs_ptrs{{static_cast<uintptr_t>(p0), static_cast<uintptr_t>(p1),
                     static_cast<uintptr_t>(p2), static_cast<uintptr_t>(p3)}};
  PeerPtrs4 sync_ptrs{{static_cast<uintptr_t>(s0), static_cast<uintptr_t>(s1),
                       static_cast<uintptr_t>(s2), static_cast<uintptr_t>(s3)}};
  unsigned* ep_ptr = reinterpret_cast<unsigned*>(ep_state.data_ptr<int32_t>());
  const int64_t bank_elems = bank_bytes / sizeof(half);
  const int stage_ctas = std::min((padded_rows * 320 + 255) / 256, 160);
  const bool w_shared = (weight.numel() == 2560);
  const bool scaled8 = skip_stage ? pre_scaled8 : has_second;
  const int rk = static_cast<int>(my_rank);

  if (!skip_stage) {
    if (has_second) {
      hc_stage_partials_f16_v3_kernel<true><<<stage_ctas, 256, 0, stream>>>(
          reinterpret_cast<half*>(rs_ptrs.p[rk]), sync_ptrs, ep_ptr, rk,
          bank_elems, reinterpret_cast<const half*>(first.data_ptr()),
          reinterpret_cast<const half*>(second->data_ptr()), valid_rows,
          padded_rows);
    } else {
      hc_stage_partials_f16_v3_kernel<false><<<stage_ctas, 256, 0, stream>>>(
          reinterpret_cast<half*>(rs_ptrs.p[rk]), sync_ptrs, ep_ptr, rk,
          bank_elems, reinterpret_cast<const half*>(first.data_ptr()), nullptr,
          valid_rows, padded_rows);
    }
  } else if (!already_signaled) {
    hc_p2p_signal_only_kernel<<<1, 32, 0, stream>>>(sync_ptrs, ep_ptr, rk);
  }
  const uintptr_t my_sync = sync_ptrs.p[rk];
  const half* ptr0 = reinterpret_cast<const half*>(rs_ptrs.p[rk]);
  const half* ptr1 = reinterpret_cast<const half*>(rs_ptrs.p[(rk + 1) & 3]);
  const half* ptr2 = reinterpret_cast<const half*>(rs_ptrs.p[(rk + 2) & 3]);
  const half* ptr3 = reinterpret_cast<const half*>(rs_ptrs.p[(rk + 3) & 3]);
  const int cur_dev = hidden.device().index();
  static bool configured_hc_rcn[16] = {};
  if (cur_dev >= 0 && cur_dev < 16 && !configured_hc_rcn[cur_dev]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        hc_p2p_reduce_combine_norm_kernel<true, true>,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        hc_p2p_reduce_combine_norm_kernel<false, true>,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    configured_hc_rcn[cur_dev] = true;
  }
  const int norm_ctas = std::min(local_rows, 240);
  if (scaled8) {
    if (normalize) {
      hc_p2p_reduce_combine_norm_kernel<true, true>
          <<<norm_ctas, 320, 0, stream>>>(
              ptr0, ptr1, ptr2, ptr3, my_sync, ep_ptr, bank_elems,
              reinterpret_cast<const half*>(hidden.data_ptr()),
              reinterpret_cast<const half*>(injection.data_ptr()),
              reinterpret_cast<const half*>(weight.data_ptr()),
              reinterpret_cast<half*>(updated.data_ptr()),
              reinterpret_cast<half*>(normalized.data_ptr()), rk, local_rows,
              w_shared, static_cast<float>(eps));
    } else {
      hc_p2p_reduce_combine_norm_kernel<true, false>
          <<<norm_ctas, 320, 0, stream>>>(
              ptr0, ptr1, ptr2, ptr3, my_sync, ep_ptr, bank_elems,
              reinterpret_cast<const half*>(hidden.data_ptr()),
              reinterpret_cast<const half*>(injection.data_ptr()),
              reinterpret_cast<const half*>(weight.data_ptr()),
              reinterpret_cast<half*>(updated.data_ptr()), nullptr, rk,
              local_rows, w_shared, static_cast<float>(eps));
    }
  } else {
    if (normalize) {
      hc_p2p_reduce_combine_norm_kernel<false, true>
          <<<norm_ctas, 320, 0, stream>>>(
              ptr0, ptr1, ptr2, ptr3, my_sync, ep_ptr, bank_elems,
              reinterpret_cast<const half*>(hidden.data_ptr()),
              reinterpret_cast<const half*>(injection.data_ptr()),
              reinterpret_cast<const half*>(weight.data_ptr()),
              reinterpret_cast<half*>(updated.data_ptr()),
              reinterpret_cast<half*>(normalized.data_ptr()), rk, local_rows,
              w_shared, static_cast<float>(eps));
    } else {
      hc_p2p_reduce_combine_norm_kernel<false, false>
          <<<norm_ctas, 320, 0, stream>>>(
              ptr0, ptr1, ptr2, ptr3, my_sync, ep_ptr, bank_elems,
              reinterpret_cast<const half*>(hidden.data_ptr()),
              reinterpret_cast<const half*>(injection.data_ptr()),
              reinterpret_cast<const half*>(weight.data_ptr()),
              reinterpret_cast<half*>(updated.data_ptr()), nullptr, rk,
              local_rows, w_shared, static_cast<float>(eps));
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

constexpr int kLocalDownHalves = 88 * 10240;
constexpr int kLocalUpHalves = 2560 * 320;
constexpr int kWeightBankHalves = kLocalDownHalves + kLocalUpHalves;

__global__ __launch_bounds__(256) void hc_unpack_5d_to_slot_kernel(
    half* __restrict__ ipc_base, const unsigned* __restrict__ ep_ptr,
    int fixed_slot, const uint4* __restrict__ down5d,
    const uint4* __restrict__ up6d) {
  const int slot =
      (fixed_slot >= 0) ? fixed_slot : static_cast<int>(*ep_ptr & 3u);
  uint4* dst_down = reinterpret_cast<uint4*>(
      ipc_base + static_cast<int64_t>(slot) * kWeightBankHalves);
  uint4* dst_up = dst_down + (kLocalDownHalves / 8);

  constexpr int kDownVecs = kLocalDownHalves / 8;
  constexpr int kUpVecs = kLocalUpHalves / 8;
  constexpr int kTotVecs = kDownVecs + kUpVecs;

  for (int v = blockIdx.x * 256 + threadIdx.x; v < kTotVecs;
       v += gridDim.x * 256) {
    if (v < kDownVecs) {
      const int r96 = v / 1280;
      const int c_vec = v % 1280;
      const int d0 = r96 >> 5;
      const int d3 = r96 & 31;
      const int d1 = c_vec >> 1;
      const int d2 = c_vec & 1;
      const int src_idx = ((d0 * 640 + d1) * 2 + d2) * 32 + d3;
      dst_down[v] = __ldg(down5d + src_idx);
    } else {
      const int uv = v - kDownVecs;
      const int r2560 = uv / 40;
      const int c_vec = uv % 40;
      const int stream = r2560 / 640;
      const int c640 = r2560 % 640;
      const int u0 = c640 >> 3;
      const int u4 = c640 & 7;
      const int u1 = c_vec >> 1;
      const int u2 = c_vec & 1;
      const int src_idx = (((u0 * 20 + u1) * 2 + u2) * 4 + stream) * 8 + u4;
      dst_up[uv] = __ldg(up6d + src_idx);
    }
  }
}

__global__ __launch_bounds__(256) void hc_copy_unpacked_to_slot_kernel(
    half* __restrict__ ipc_base, const unsigned* __restrict__ ep_ptr,
    int fixed_slot, const uint4* __restrict__ local_down,
    const uint4* __restrict__ local_up) {
  const int slot =
      (fixed_slot >= 0) ? fixed_slot : static_cast<int>(*ep_ptr & 3u);
  uint4* dst_down = reinterpret_cast<uint4*>(
      ipc_base + static_cast<int64_t>(slot) * kWeightBankHalves);
  uint4* dst_up = dst_down + (kLocalDownHalves / 8);
  constexpr int kDownVecs = kLocalDownHalves / 8;
  constexpr int kUpVecs = kLocalUpHalves / 8;
  for (int v = blockIdx.x * 256 + threadIdx.x; v < kDownVecs + kUpVecs;
       v += gridDim.x * 256) {
    if (v < kDownVecs)
      dst_down[v] = __ldg(local_down + v);
    else
      dst_up[v - kDownVecs] = __ldg(local_up + (v - kDownVecs));
  }
}

__global__ __launch_bounds__(256, 3) void hc_p2p_gather_weights_striped_kernel(
    PeerPtrs4 w_ptrs, const unsigned* __restrict__ ep_ptr, int fixed_slot,
    int local_ctas, int my_rank, int r0, int r1, int r2, int r3,
    uint4* __restrict__ full_down, uint4* __restrict__ full_up) {
  const int slot =
      (fixed_slot >= 0) ? fixed_slot : static_cast<int>((*ep_ptr - 1u) & 3u);

  const int local_j =
      (r0 == my_rank) ? 0 : ((r1 == my_rank) ? 1 : ((r2 == my_rank) ? 2 : 3));
  auto find_j = [&](int target_phys) -> int {
    return (r0 == target_phys)
               ? 0
               : ((r1 == target_phys) ? 1 : ((r2 == target_phys) ? 2 : 3));
  };
  const int rem_j0 = find_j((my_rank + 1) & 3);
  const int rem_j1 = find_j((my_rank + 2) & 3);
  const int rem_j2 = find_j((my_rank + 3) & 3);

  int j = local_j;
  int cta_idx = blockIdx.x;
  int num_ctas = local_ctas;
  if (static_cast<int>(blockIdx.x) >= local_ctas) {
    const int rel = static_cast<int>(blockIdx.x) - local_ctas;
    const int p = rel % 3;
    j = (p == 0) ? rem_j0 : ((p == 1) ? rem_j1 : rem_j2);
    cta_idx = rel / 3;
    num_ctas = (static_cast<int>(gridDim.x) - local_ctas) / 3;
  }

  const int phys = (j == 0) ? r0 : ((j == 1) ? r1 : ((j == 2) ? r2 : r3));
  const uintptr_t phys_ptr =
      (phys == 0) ? w_ptrs.p[0]
                  : ((phys == 1) ? w_ptrs.p[1]
                                 : ((phys == 2) ? w_ptrs.p[2] : w_ptrs.p[3]));
  const uint4* peer_down = reinterpret_cast<const uint4*>(
      reinterpret_cast<const half*>(phys_ptr) +
      static_cast<int64_t>(slot) * kWeightBankHalves);
  const uint4* peer_up = peer_down + (kLocalDownHalves / 8);

  const int stride = num_ctas * 256;
  const int tid = cta_idx * 256 + threadIdx.x;

  const int down_vecs = ((j < 3) ? 80 : 84) * 1280;
  uint4* dst_d = full_down + j * 80 * 1280;
  int idx = tid;
  const int s3 = stride * 3;
  const int s4 = stride * 4;
  for (; idx + s3 < down_vecs; idx += s4) {
    const uint4 d0 = peer_down[idx];
    const uint4 d1 = peer_down[idx + stride];
    const uint4 d2 = peer_down[idx + stride * 2];
    const uint4 d3 = peer_down[idx + s3];
    dst_d[idx] = d0;
    dst_d[idx + stride] = d1;
    dst_d[idx + stride * 2] = d2;
    dst_d[idx + s3] = d3;
  }
  for (; idx < down_vecs; idx += stride) {
    dst_d[idx] = peer_down[idx];
  }

  constexpr int kUpVecsPerPeer = 4 * 640 * 40;
  int uidx = tid;
  for (; uidx + s3 < kUpVecsPerPeer; uidx += s4) {
    const uint4 u0 = peer_up[uidx];
    const uint4 u1 = peer_up[uidx + stride];
    const uint4 u2 = peer_up[uidx + stride * 2];
    const uint4 u3 = peer_up[uidx + s3];
    const int i0 = uidx, i1 = uidx + stride, i2 = uidx + stride * 2,
              i3 = uidx + s3;
    full_up[((i0 / 25600) * 2560 + j * 640) * 40 + (i0 % 25600)] = u0;
    full_up[((i1 / 25600) * 2560 + j * 640) * 40 + (i1 % 25600)] = u1;
    full_up[((i2 / 25600) * 2560 + j * 640) * 40 + (i2 % 25600)] = u2;
    full_up[((i3 / 25600) * 2560 + j * 640) * 40 + (i3 % 25600)] = u3;
  }
  for (; uidx < kUpVecsPerPeer; uidx += stride) {
    const int stream = uidx / (640 * 40);
    const int rem = uidx % (640 * 40);
    full_up[(stream * 2560 + j * 640) * 40 + rem] = peer_up[uidx];
  }

  if (static_cast<int>(blockIdx.x) < local_ctas) {
    for (int zidx = tid; zidx < 12 * 1280; zidx += stride) {
      full_down[324 * 1280 + zidx] = make_uint4(0, 0, 0, 0);
    }
  }
}

__global__ __launch_bounds__(88) void hc_split_silu_kernel(
    const half* __restrict__ projected, half* __restrict__ lora,
    half* __restrict__ inj) {
  const int row = blockIdx.x;
  const int tx = threadIdx.x;
  if (tx < 80) {
    const uint2 u = __ldg(reinterpret_cast<const uint2*>(
        projected + static_cast<int64_t>(row) * 336 + tx * 4));
    const half* h = reinterpret_cast<const half*>(&u);
    uint2 out_u;
    half* oh = reinterpret_cast<half*>(&out_u);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float x = __half2float(h[j]) * 0.25f;
      oh[j] = __float2half_rn(x / (1.0f + expf(-x)));
    }
    *reinterpret_cast<uint2*>(lora + static_cast<int64_t>(row) * 320 + tx * 4) =
        out_u;
  } else if (tx == 80) {
    *reinterpret_cast<uint2*>(inj + static_cast<int64_t>(row) * 4) =
        __ldg(reinterpret_cast<const uint2*>(
            projected + static_cast<int64_t>(row) * 336 + 320));
  }
}

__global__ __launch_bounds__(320, 3) void hc_gate_mix_to_ipc_kernel(
    half* __restrict__ ag_base, PeerPtrs4 sync_ptrs,
    unsigned* __restrict__ ep_ptr, int my_rank, int64_t bank_elems,
    const half* __restrict__ x, const half* __restrict__ gate, int local_rows) {
  const int slot = static_cast<int>(*ep_ptr & 1u);
  half* dst = ag_base + static_cast<int64_t>(slot) * bank_elems;
  const int row = blockIdx.x;
  const int c8 = threadIdx.x * 8;
  float acc[8] = {};
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    const int64_t off = (static_cast<int64_t>(row) * 4 + s) * 2560 + c8;
    const uint4 xu = __ldg(reinterpret_cast<const uint4*>(x + off));
    const uint4 gu = __ldg(reinterpret_cast<const uint4*>(gate + off));
    const half* xh = reinterpret_cast<const half*>(&xu);
    const half* gh = reinterpret_cast<const half*>(&gu);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float gf = __half2float(gh[j]);
      const float sig = 1.0f / (1.0f + expf(-gf));
      acc[j] = __fmaf_rn(sig, __half2float(xh[j]), acc[j]);
    }
  }
  uint4 out_u4;
  half* oh = reinterpret_cast<half*>(&out_u4);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    oh[j] = __float2half_rn(acc[j] * 0.25f);
  }
  *reinterpret_cast<uint4*>(dst + static_cast<int64_t>(row) * 2560 + c8) =
      out_u4;
  hc_p2p_signal_last_cta(sync_ptrs, ep_ptr, my_rank);
}

__global__ __launch_bounds__(256, 3) void hc_p2p_gather_rows_kernel(
    PeerPtrs4 ag_ptrs, uintptr_t my_sync_ptr,
    const unsigned* __restrict__ ep_ptr, int64_t bank_elems,
    half* __restrict__ out, int my_rank, int local_rows, int out_rows,
    int row_width, int local_ctas = 16) {
  const unsigned ep = *ep_ptr;
  if (my_sync_ptr != 0) {
    hc_p2p_wait_peers(my_sync_ptr, ep);
  }
  const int slot = static_cast<int>((ep - 1u) & 1u);
  const int vec_per_row = row_width / 8;

  int owner = my_rank;
  int cta_idx = blockIdx.x;
  int num_ctas = local_ctas;
  const bool is_peer = (static_cast<int>(blockIdx.x) >= local_ctas);
  if (is_peer) {
    const int rel = static_cast<int>(blockIdx.x) - local_ctas;
    const int p = rel % 3;
    owner = (my_rank + 1 + p) & 3;
    cta_idx = rel / 3;
    num_ctas = (static_cast<int>(gridDim.x) - local_ctas) / 3;
  }

  const int g_base = owner * local_rows;
  const int valid_rows = max(0, min(local_rows, out_rows - g_base));
  const int64_t peer_vecs = static_cast<int64_t>(valid_rows) * vec_per_row;
  const int64_t stride = static_cast<int64_t>(num_ctas) * 256;
  const int64_t tid = static_cast<int64_t>(cta_idx) * 256 + threadIdx.x;

  const uintptr_t owner_ptr =
      (owner == 0)
          ? ag_ptrs.p[0]
          : ((owner == 1) ? ag_ptrs.p[1]
                          : ((owner == 2) ? ag_ptrs.p[2] : ag_ptrs.p[3]));
  const uint4* src =
      reinterpret_cast<const uint4*>(reinterpret_cast<const half*>(owner_ptr) +
                                     static_cast<int64_t>(slot) * bank_elems);
  uint4* dst =
      reinterpret_cast<uint4*>(out + static_cast<int64_t>(g_base) * row_width);
  int64_t v = tid;
  const int64_t s3 = stride * 3;
  const int64_t s4 = stride * 4;
  if (is_peer) {
    for (; v + s3 < peer_vecs; v += s4) {
      const uint4 u0 = p2p_ld_cv_u4(src + v);
      const uint4 u1 = p2p_ld_cv_u4(src + v + stride);
      const uint4 u2 = p2p_ld_cv_u4(src + v + stride * 2);
      const uint4 u3 = p2p_ld_cv_u4(src + v + s3);
      dst[v] = u0;
      dst[v + stride] = u1;
      dst[v + stride * 2] = u2;
      dst[v + s3] = u3;
    }
    for (; v < peer_vecs; v += stride) {
      dst[v] = p2p_ld_cv_u4(src + v);
    }
  } else {
    for (; v + s3 < peer_vecs; v += s4) {
      const uint4 u0 = __ldg(src + v);
      const uint4 u1 = __ldg(src + v + stride);
      const uint4 u2 = __ldg(src + v + stride * 2);
      const uint4 u3 = __ldg(src + v + s3);
      dst[v] = u0;
      dst[v + stride] = u1;
      dst[v + stride * 2] = u2;
      dst[v + s3] = u3;
    }
    for (; v < peer_vecs; v += stride) {
      dst[v] = __ldg(src + v);
    }
  }
}

void hc_p2p_unpack_gather_weights_out(
    torch::Tensor full_down, torch::Tensor full_up, torch::Tensor down_shard,
    torch::Tensor up_shard, bool already_unpacked, int64_t w0, int64_t w1,
    int64_t w2, int64_t w3, int64_t s0, int64_t s1, int64_t s2, int64_t s3,
    torch::Tensor ep_state, int64_t my_rank, int64_t ord0, int64_t ord1,
    int64_t ord2, int64_t ord3, bool async_low_sm = false) {
  const c10::cuda::CUDAGuard guard(down_shard.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  PeerPtrs4 w_ptrs{{static_cast<uintptr_t>(w0), static_cast<uintptr_t>(w1),
                    static_cast<uintptr_t>(w2), static_cast<uintptr_t>(w3)}};
  PeerPtrs4 sync_ptrs{{static_cast<uintptr_t>(s0), static_cast<uintptr_t>(s1),
                       static_cast<uintptr_t>(s2), static_cast<uintptr_t>(s3)}};
  unsigned* ep_ptr = reinterpret_cast<unsigned*>(ep_state.data_ptr<int32_t>());
  const int unpack_ctas = async_low_sm ? 16 : 160;
  const int local_ctas = async_low_sm ? 4 : 40;
  const int gather_ctas = async_low_sm ? 28 : 256;

  if (already_unpacked) {
    hc_copy_unpacked_to_slot_kernel<<<unpack_ctas, 256, 0, stream>>>(
        reinterpret_cast<half*>(w_ptrs.p[my_rank]), ep_ptr, -1,
        reinterpret_cast<const uint4*>(down_shard.data_ptr()),
        reinterpret_cast<const uint4*>(up_shard.data_ptr()));
  } else {
    hc_unpack_5d_to_slot_kernel<<<unpack_ctas, 256, 0, stream>>>(
        reinterpret_cast<half*>(w_ptrs.p[my_rank]), ep_ptr, -1,
        reinterpret_cast<const uint4*>(down_shard.data_ptr()),
        reinterpret_cast<const uint4*>(up_shard.data_ptr()));
  }
  hc_p2p_barrier_kernel<<<1, 32, 0, stream>>>(sync_ptrs, ep_ptr,
                                              static_cast<int>(my_rank));
  hc_p2p_gather_weights_striped_kernel<<<gather_ctas, 256, 0, stream>>>(
      w_ptrs, ep_ptr, -1, local_ctas, static_cast<int>(my_rank),
      static_cast<int>(ord0), static_cast<int>(ord1), static_cast<int>(ord2),
      static_cast<int>(ord3), reinterpret_cast<uint4*>(full_down.data_ptr()),
      reinterpret_cast<uint4*>(full_up.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void hc_p2p_init_static_weight_slot_out(
    torch::Tensor down_shard, torch::Tensor up_shard, bool already_unpacked,
    int64_t static_slot, int64_t sw0, int64_t sw1, int64_t sw2, int64_t sw3,
    int64_t s0, int64_t s1, int64_t s2, int64_t s3, torch::Tensor ep_state,
    int64_t my_rank) {
  const c10::cuda::CUDAGuard guard(down_shard.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  PeerPtrs4 sw_ptrs{{static_cast<uintptr_t>(sw0), static_cast<uintptr_t>(sw1),
                     static_cast<uintptr_t>(sw2), static_cast<uintptr_t>(sw3)}};
  PeerPtrs4 sync_ptrs{{static_cast<uintptr_t>(s0), static_cast<uintptr_t>(s1),
                       static_cast<uintptr_t>(s2), static_cast<uintptr_t>(s3)}};
  unsigned* ep_ptr = reinterpret_cast<unsigned*>(ep_state.data_ptr<int32_t>());

  if (already_unpacked) {
    hc_copy_unpacked_to_slot_kernel<<<160, 256, 0, stream>>>(
        reinterpret_cast<half*>(sw_ptrs.p[my_rank]), nullptr,
        static_cast<int>(static_slot),
        reinterpret_cast<const uint4*>(down_shard.data_ptr()),
        reinterpret_cast<const uint4*>(up_shard.data_ptr()));
  } else {
    hc_unpack_5d_to_slot_kernel<<<160, 256, 0, stream>>>(
        reinterpret_cast<half*>(sw_ptrs.p[my_rank]), nullptr,
        static_cast<int>(static_slot),
        reinterpret_cast<const uint4*>(down_shard.data_ptr()),
        reinterpret_cast<const uint4*>(up_shard.data_ptr()));
  }
  hc_p2p_barrier_kernel<<<1, 32, 0, stream>>>(sync_ptrs, ep_ptr,
                                              static_cast<int>(my_rank));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void hc_p2p_gather_static_weights_out(
    torch::Tensor full_down, torch::Tensor full_up, int64_t static_slot,
    int64_t sw0, int64_t sw1, int64_t sw2, int64_t sw3, int64_t my_rank,
    int64_t ord0, int64_t ord1, int64_t ord2, int64_t ord3) {
  const c10::cuda::CUDAGuard guard(full_down.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  PeerPtrs4 sw_ptrs{{static_cast<uintptr_t>(sw0), static_cast<uintptr_t>(sw1),
                     static_cast<uintptr_t>(sw2), static_cast<uintptr_t>(sw3)}};
  hc_p2p_gather_weights_striped_kernel<<<256, 256, 0, stream>>>(
      sw_ptrs, nullptr, static_cast<int>(static_slot), 40,
      static_cast<int>(my_rank), static_cast<int>(ord0), static_cast<int>(ord1),
      static_cast<int>(ord2), static_cast<int>(ord3),
      reinterpret_cast<uint4*>(full_down.data_ptr()),
      reinterpret_cast<uint4*>(full_up.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Single-wave conflict-free SM70 Tensor Core HC down-projection + fused
// SiLU/inj: Replaces slow cuBLAS 64x64 split-K=2 + splitKreduce +
// hc_split_silu!
__global__ __launch_bounds__(256, 3) void hc_down_proj_split_silu_kernel(
    const half* __restrict__ x,          // [M, 10240]
    const half* __restrict__ full_down,  // [>=324, 10240]
    half* __restrict__ lora,             // [M, 320]
    half* __restrict__ inj,              // [M, 4]
    int M) {
  const int n_blk =
      blockIdx.x;  // 0..5 (0..4 -> 320 lora cols; 5 -> 4 inj cols)
  const int m_start = blockIdx.y * 64;
  if (m_start >= M) return;
  const int m_count = min(64, M - m_start);

  constexpr int kHidden = 10240;
  constexpr int kStepK = 32;
  constexpr int kNumStages = kHidden / kStepK;  // 320

  union SharedHcDown {
    struct {
      uint4 s_A[2][16 * 17];
      uint4 s_B[2][16 * 17];
    } mainloop;
    half s_out[64][72];
  };
  __shared__ __align__(16) SharedHcDown smem;

  const int tx = threadIdx.x;    // 0..255
  const int lane = tx & 31;      // 0..31
  const int warp = tx >> 5;      // 0..7
  const int warp_m = warp >> 1;  // 0..3 -> 16 rows each = 64 rows
  const int warp_n = warp & 1;   // 0..1 -> 32 cols each = 64 cols

  const int a_r = tx >> 2;  // 0..63
  const int a_s = tx & 3;   // 0..3
  const int a_smem_idx = smem_idx17(a_r, a_s);
  const half* a_ptr =
      (a_r < m_count)
          ? (x + static_cast<size_t>(m_start + a_r) * kHidden + a_s * 8)
          : nullptr;

  const int b_r = tx >> 2;  // 0..63
  const int b_s = tx & 3;   // 0..3
  const int global_n = n_blk * 64 + b_r;
  const int b_smem_idx = smem_idx17(b_r, b_s);
  const half* b_ptr =
      (global_n < 324)
          ? (full_down + static_cast<size_t>(global_n) * kHidden + b_s * 8)
          : nullptr;

  {
    const uint4 av = a_ptr ? __ldg(reinterpret_cast<const uint4*>(a_ptr))
                           : make_uint4(0, 0, 0, 0);
    const uint4 bv = b_ptr ? __ldg(reinterpret_cast<const uint4*>(b_ptr))
                           : make_uint4(0, 0, 0, 0);
    smem.mainloop.s_A[0][a_smem_idx] = av;
    smem.mainloop.s_B[0][b_smem_idx] = bv;
  }
  __syncthreads();

  const int mma_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int quad = (lane >> 2) & 3;
  const int b_col = warp_n * 32 + quad * 8 + mma_row;
  const int rb0 = warp_m * 16 + mma_row;
  const int rb1 = rb0 + 8;
  const bool active_w = (n_blk < 5) || (warp_n == 0);
  const bool do_m0 = active_w && (warp_m * 16 < m_count);
  const bool do_m1 = active_w && (warp_m * 16 + 8 < m_count);

  float acc0[8] = {};
  float acc1[8] = {};

  for (int st = 0; st < kNumStages; ++st) {
    const int cur = st & 1;
    const int nxt = cur ^ 1;
    uint4 av = make_uint4(0, 0, 0, 0);
    uint4 bv = make_uint4(0, 0, 0, 0);
    if (st + 1 < kNumStages) {
      const int k_next = (st + 1) * kStepK;
      if (a_ptr) av = __ldg(reinterpret_cast<const uint4*>(a_ptr + k_next));
      if (b_ptr) bv = __ldg(reinterpret_cast<const uint4*>(b_ptr + k_next));
    }

    if (active_w) {
#pragma unroll
      for (int sub = 0; sub < 2; ++sub) {
        const int sl0 = sub * 2;
        const int sl1 = sub * 2 + 1;
        const uint4 b0 = smem.mainloop.s_B[cur][smem_idx17(b_col, sl0)];
        const uint4 b1 = smem.mainloop.s_B[cur][smem_idx17(b_col, sl1)];
        if (do_m0) {
          const uint4 a_lo = smem.mainloop.s_A[cur][smem_idx17(rb0, sl0)];
          const uint4 a_hi = smem.mainloop.s_A[cur][smem_idx17(rb0, sl1)];
          FN_MMA_884_RC(acc0, a_lo.x, a_lo.y, b0.x, b0.y);
          FN_MMA_884_RC(acc0, a_lo.z, a_lo.w, b0.z, b0.w);
          FN_MMA_884_RC(acc0, a_hi.x, a_hi.y, b1.x, b1.y);
          FN_MMA_884_RC(acc0, a_hi.z, a_hi.w, b1.z, b1.w);
        }
        if (do_m1) {
          const uint4 a_lo = smem.mainloop.s_A[cur][smem_idx17(rb1, sl0)];
          const uint4 a_hi = smem.mainloop.s_A[cur][smem_idx17(rb1, sl1)];
          FN_MMA_884_RC(acc1, a_lo.x, a_lo.y, b0.x, b0.y);
          FN_MMA_884_RC(acc1, a_lo.z, a_lo.w, b0.z, b0.w);
          FN_MMA_884_RC(acc1, a_hi.x, a_hi.y, b1.x, b1.y);
          FN_MMA_884_RC(acc1, a_hi.z, a_hi.w, b1.z, b1.w);
        }
      }
    }

    if (st + 1 < kNumStages) {
      smem.mainloop.s_A[nxt][a_smem_idx] = av;
      smem.mainloop.s_B[nxt][b_smem_idx] = bv;
    }
    __syncthreads();
  }

  auto stage_epilogue = [&](const float (&acc)[8], int row_base) {
    if (row_base >= m_count) return;
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
      const int r = row_base + ((i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1));
      const int c =
          warp_n * 32 + quad * 8 + ((((lane >> 1) & 1) << 1) | ((i >> 2) << 2));
      const half h0 = __float2half_rn(acc[i]);
      const half h1 = __float2half_rn(acc[i + 1]);
      if (n_blk < 5) {
        const float x0 = __half2float(h0) * 0.25f;
        const float x1 = __half2float(h1) * 0.25f;
        smem.s_out[r][c] = __float2half_rn(x0 / (1.0f + expf(-x0)));
        smem.s_out[r][c + 1] = __float2half_rn(x1 / (1.0f + expf(-x1)));
      } else if (c < 4) {
        smem.s_out[r][c] = h0;
        smem.s_out[r][c + 1] = h1;
      }
    }
  };
  if (active_w) {
    stage_epilogue(acc0, warp_m * 16);
    stage_epilogue(acc1, warp_m * 16 + 8);
  }
  __syncthreads();

  if (n_blk < 5) {
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int vid = tx + u * 256;
      const int orow = vid >> 3;
      const int ocol8 = (vid & 7) * 8;
      if (orow < m_count) {
        *reinterpret_cast<uint4*>(
            lora + static_cast<size_t>(m_start + orow) * 320 + n_blk * 64 +
            ocol8) = *reinterpret_cast<const uint4*>(&smem.s_out[orow][ocol8]);
      }
    }
  } else if (tx < m_count) {
    *reinterpret_cast<uint2*>(inj + static_cast<size_t>(m_start + tx) * 4) =
        *reinterpret_cast<const uint2*>(&smem.s_out[tx][0]);
  }
}

// Zero-Spill Single-Wave 2-CTA/SM (20 warps/SM, <=96 regs, 0 spills!)
// Fused HC Down-Projection + SiLU(0.25 * x) + Injection Split!
// Grid = 2 * ceil(M/64) main CTAs (CTA_M=64, CTA_N=160) + ceil(M/320) inj CTAs
// (CTA_M=320, CTA_N=4) = 141 CTAs <= 160 slots at M=4096 and 124 CTAs <= 160
// slots at M=3584 -> ONE SINGLE WAVE across all 80 V100 SMs!
__global__ __launch_bounds__(320, 2) void hc_down_m64_n160_2cta_silu_kernel(
    const half* __restrict__ x,          // [M, 10240]
    const half* __restrict__ full_down,  // [>=324, 10240]
    half* __restrict__ lora,             // [M, 320]
    half* __restrict__ inj,              // [M, 4]
    int M, int m_tiles) {
  constexpr int kHiddenU4 = 10240 / 8;    // 1280 uint4s/row
  constexpr int kNumStages = 10240 / 32;  // 320

  union SharedHcDown64x160 {
    struct {
      uint4 s_A[2][16 * 17];  //  8,704 B
      uint4 s_B[2][40 * 17];  // 21,760 B
    } mainloop;               // 30,464 B x 2 = 60,928 B <= 96 KiB!
    struct {
      uint4 s_Ai[2][80 * 17];  // 43,520 B ([320, 32] for inj CTAs)
      uint4 s_Bi[2][2 * 17];   //  1,088 B
    } injloop;
    alignas(16) half s_out[64][168];  // 21,504 B
  };
  __shared__ __align__(16) SharedHcDown64x160 smem;

  const int tx = threadIdx.x;  // 0..319
  const int lane = tx & 31;    // 0..31
  const int warp = tx >> 5;    // 0..9
  const uint4* x_u4 = reinterpret_cast<const uint4*>(x);
  const uint4* fd_u4 = reinterpret_cast<const uint4*>(full_down);

  if (static_cast<int>(blockIdx.x) < 2 * m_tiles) {
    const int n_blk = blockIdx.x / m_tiles;  // 0 or 1
    const int m_blk = blockIdx.x - n_blk * m_tiles;
    const int m_start = m_blk * 64;
    const int m_count = min(64, M - m_start);
    const int warp_m = warp / 5;  // 0..1 -> 32 rows each = 64 rows
    const int warp_n = warp % 5;  // 0..4 -> 32 cols each = 160 cols

    const int a_r = tx >> 2;
    const int a_s = tx & 3;
    const int a_smem_idx = (tx < 256) ? smem_idx17(a_r, a_s) : 0;
    const bool valid_a = (tx < 256) && (a_r < m_count);
    const uint32_t x_u4_base =
        valid_a ? (static_cast<uint32_t>(m_start + a_r) * kHiddenU4 + a_s) : 0u;

    const int b_r0 = tx >> 2;    // 0..79
    const int b_r1 = b_r0 + 80;  // 80..159
    const int b_s = tx & 3;      // 0..3
    const int b_smem0 = smem_idx17(b_r0, b_s);
    const int b_smem1 = smem_idx17(b_r1, b_s);
    const uint32_t b0_u4_base =
        static_cast<uint32_t>(n_blk * 160 + b_r0) * kHiddenU4 + b_s;
    const uint32_t b1_u4_base =
        static_cast<uint32_t>(n_blk * 160 + b_r1) * kHiddenU4 + b_s;

    {
      if (tx < 256) {
        smem.mainloop.s_A[0][a_smem_idx] =
            valid_a ? __ldg(x_u4 + x_u4_base) : make_uint4(0, 0, 0, 0);
      }
      smem.mainloop.s_B[0][b_smem0] = __ldg(fd_u4 + b0_u4_base);
      smem.mainloop.s_B[0][b_smem1] = __ldg(fd_u4 + b1_u4_base);
    }
    __syncthreads();

    const int mma_row = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int quad = (lane >> 2) & 3;
    const int b_col = warp_n * 32 + quad * 8 + mma_row;
    const int m_base = warp_m * 32;
    const int r0 = m_base + mma_row;

    float acc0[8] = {}, acc1[8] = {}, acc2[8] = {}, acc3[8] = {};

#pragma unroll 1
    for (int st = 0; st < kNumStages; ++st) {
      const int cur = st & 1;
      const int nxt = cur ^ 1;
      uint4 av = make_uint4(0, 0, 0, 0);
      uint4 bv0 = make_uint4(0, 0, 0, 0);
      uint4 bv1 = make_uint4(0, 0, 0, 0);
      if (st + 1 < kNumStages) {
        const uint32_t k_u4 = static_cast<uint32_t>(st + 1) * 4u;
        if (valid_a) av = __ldg(x_u4 + x_u4_base + k_u4);
        bv0 = __ldg(fd_u4 + b0_u4_base + k_u4);
        bv1 = __ldg(fd_u4 + b1_u4_base + k_u4);
      }

      if (m_base < m_count) {
#pragma unroll
        for (int sl = 0; sl < 4; ++sl) {
          const uint4 b = smem.mainloop.s_B[cur][smem_idx17(b_col, sl)];
          uint4 a = smem.mainloop.s_A[cur][smem_idx17(r0, sl)];
          FN_MMA_884_RC(acc0, a.x, a.y, b.x, b.y);
          FN_MMA_884_RC(acc0, a.z, a.w, b.z, b.w);
          a = smem.mainloop.s_A[cur][smem_idx17(r0 + 8, sl)];
          FN_MMA_884_RC(acc1, a.x, a.y, b.x, b.y);
          FN_MMA_884_RC(acc1, a.z, a.w, b.z, b.w);
          a = smem.mainloop.s_A[cur][smem_idx17(r0 + 16, sl)];
          FN_MMA_884_RC(acc2, a.x, a.y, b.x, b.y);
          FN_MMA_884_RC(acc2, a.z, a.w, b.z, b.w);
          a = smem.mainloop.s_A[cur][smem_idx17(r0 + 24, sl)];
          FN_MMA_884_RC(acc3, a.x, a.y, b.x, b.y);
          FN_MMA_884_RC(acc3, a.z, a.w, b.z, b.w);
        }
      }

      if (st + 1 < kNumStages) {
        if (tx < 256) smem.mainloop.s_A[nxt][a_smem_idx] = av;
        smem.mainloop.s_B[nxt][b_smem0] = bv0;
        smem.mainloop.s_B[nxt][b_smem1] = bv1;
      }
      __syncthreads();
    }

    const int r_sub = ((lane & 16) ? 4 : 0) | (lane & 1);
    const int c_sub = warp_n * 32 + quad * 8 + (((lane >> 1) & 1) << 1);
    auto stage_lora = [&](const float (&acc)[8], int row_base) {
      if (row_base >= m_count) return;
      const int rb0 = row_base + r_sub;
      const int rb2 = rb0 + 2;
      auto silu025_pair = [](float v0, float v1) -> half2 {
        const float x0 = __half2float(__float2half_rn(v0)) * 0.25f;
        const float x1 = __half2float(__float2half_rn(v1)) * 0.25f;
        return __floats2half2_rn(x0 / (1.0f + expf(-x0)),
                                 x1 / (1.0f + expf(-x1)));
      };
      *reinterpret_cast<half2*>(&smem.s_out[rb0][c_sub]) =
          silu025_pair(acc[0], acc[1]);
      *reinterpret_cast<half2*>(&smem.s_out[rb2][c_sub]) =
          silu025_pair(acc[2], acc[3]);
      *reinterpret_cast<half2*>(&smem.s_out[rb0][c_sub + 4]) =
          silu025_pair(acc[4], acc[5]);
      *reinterpret_cast<half2*>(&smem.s_out[rb2][c_sub + 4]) =
          silu025_pair(acc[6], acc[7]);
    };
    stage_lora(acc0, m_base + 0);
    stage_lora(acc1, m_base + 8);
    stage_lora(acc2, m_base + 16);
    stage_lora(acc3, m_base + 24);
    __syncthreads();

#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int vid = tx + u * 320;
      const int orow = vid / 20;
      const int ocol8 = (vid % 20) * 8;
      if (orow < m_count) {
        *reinterpret_cast<uint4*>(
            lora + static_cast<size_t>(m_start + orow) * 320 + n_blk * 160 +
            ocol8) = *reinterpret_cast<const uint4*>(&smem.s_out[orow][ocol8]);
      }
    }
  } else {
    // Dedicated 320-row x 4-col Tensor Core injection CTAs (only 13 CTAs at
    // M=4096!)
    const int m_inj = static_cast<int>(blockIdx.x) - 2 * m_tiles;
    const int m_start = m_inj * 320;
    if (m_start >= M) return;
    const int m_count = min(320, M - m_start);

    const int a_r0 = tx >> 2;  // 0..79
    const int a_s = tx & 3;    // 0..3
    const uint32_t b_u4_base =
        (tx < 16) ? (static_cast<uint32_t>(320 + (tx >> 2)) * kHiddenU4 + a_s)
                  : 0u;

    {
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int r = a_r0 + u * 80;
        smem.injloop.s_Ai[0][smem_idx17(r, a_s)] =
            (r < m_count)
                ? __ldg(x_u4 + static_cast<uint32_t>(m_start + r) * kHiddenU4 +
                        a_s)
                : make_uint4(0, 0, 0, 0);
      }
      if (tx < 32) {
        smem.injloop.s_Bi[0][smem_idx17(tx >> 2, a_s)] =
            (tx < 16) ? __ldg(fd_u4 + b_u4_base) : make_uint4(0, 0, 0, 0);
      }
    }
    __syncthreads();

    const int mma_row = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int quad = (lane >> 2) & 3;
    const int m_base = warp * 32;
    const int r0 = m_base + mma_row;

    float acc0[8] = {}, acc1[8] = {}, acc2[8] = {}, acc3[8] = {};

#pragma unroll 1
    for (int st = 0; st < kNumStages; ++st) {
      const int cur = st & 1;
      const int nxt = cur ^ 1;
      uint4 av0 = make_uint4(0, 0, 0, 0);
      uint4 av1 = make_uint4(0, 0, 0, 0);
      uint4 av2 = make_uint4(0, 0, 0, 0);
      uint4 av3 = make_uint4(0, 0, 0, 0);
      uint4 bv = make_uint4(0, 0, 0, 0);
      if (st + 1 < kNumStages) {
        const uint32_t k_u4 = static_cast<uint32_t>(st + 1) * 4u + a_s;
        if (a_r0 < m_count)
          av0 = __ldg(x_u4 + static_cast<uint32_t>(m_start + a_r0) * kHiddenU4 +
                      k_u4);
        if (a_r0 + 80 < m_count)
          av1 = __ldg(x_u4 +
                      static_cast<uint32_t>(m_start + a_r0 + 80) * kHiddenU4 +
                      k_u4);
        if (a_r0 + 160 < m_count)
          av2 = __ldg(x_u4 +
                      static_cast<uint32_t>(m_start + a_r0 + 160) * kHiddenU4 +
                      k_u4);
        if (a_r0 + 240 < m_count)
          av3 = __ldg(x_u4 +
                      static_cast<uint32_t>(m_start + a_r0 + 240) * kHiddenU4 +
                      k_u4);
        if (tx < 16)
          bv = __ldg(fd_u4 + b_u4_base + static_cast<uint32_t>(st + 1) * 4u);
      }

      if (m_base < m_count) {
#pragma unroll
        for (int sl = 0; sl < 4; ++sl) {
          const uint4 b = smem.injloop.s_Bi[cur][smem_idx17(mma_row, sl)];
          uint4 a = smem.injloop.s_Ai[cur][smem_idx17(r0, sl)];
          FN_MMA_884_RC(acc0, a.x, a.y, b.x, b.y);
          FN_MMA_884_RC(acc0, a.z, a.w, b.z, b.w);
          a = smem.injloop.s_Ai[cur][smem_idx17(r0 + 8, sl)];
          FN_MMA_884_RC(acc1, a.x, a.y, b.x, b.y);
          FN_MMA_884_RC(acc1, a.z, a.w, b.z, b.w);
          a = smem.injloop.s_Ai[cur][smem_idx17(r0 + 16, sl)];
          FN_MMA_884_RC(acc2, a.x, a.y, b.x, b.y);
          FN_MMA_884_RC(acc2, a.z, a.w, b.z, b.w);
          a = smem.injloop.s_Ai[cur][smem_idx17(r0 + 24, sl)];
          FN_MMA_884_RC(acc3, a.x, a.y, b.x, b.y);
          FN_MMA_884_RC(acc3, a.z, a.w, b.z, b.w);
        }
      }

      if (st + 1 < kNumStages) {
        smem.injloop.s_Ai[nxt][smem_idx17(a_r0, a_s)] = av0;
        smem.injloop.s_Ai[nxt][smem_idx17(a_r0 + 80, a_s)] = av1;
        smem.injloop.s_Ai[nxt][smem_idx17(a_r0 + 160, a_s)] = av2;
        smem.injloop.s_Ai[nxt][smem_idx17(a_r0 + 240, a_s)] = av3;
        if (tx < 32) smem.injloop.s_Bi[nxt][smem_idx17(tx >> 2, a_s)] = bv;
      }
      __syncthreads();
    }

    if (quad == 0) {
      const int r_sub = ((lane & 16) ? 4 : 0) | (lane & 1);
      const int c = ((lane >> 1) & 1) << 1;  // 0 or 2 -> cols 0..3!
      auto write_inj = [&](const float (&acc)[8], int row_base) {
        const int rb0 = row_base + r_sub;
        const int rb2 = rb0 + 2;
        if (rb0 < m_count) {
          *reinterpret_cast<half2*>(
              inj + static_cast<size_t>(m_start + rb0) * 4 + c) =
              __floats2half2_rn(acc[0], acc[1]);
        }
        if (rb2 < m_count) {
          *reinterpret_cast<half2*>(
              inj + static_cast<size_t>(m_start + rb2) * 4 + c) =
              __floats2half2_rn(acc[2], acc[3]);
        }
      };
      write_inj(acc0, m_base + 0);
      write_inj(acc1, m_base + 8);
      write_inj(acc2, m_base + 16);
      write_inj(acc3, m_base + 24);
    }
  }
}

void hc_down_proj_split_silu_out(torch::Tensor lora, torch::Tensor inj,
                                 torch::Tensor x, torch::Tensor full_down) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int M = static_cast<int>(x.size(0));
  if (M == 0) return;
  const int cur_dev = x.device().index();
  static bool configured_hc_down[16] = {};
  if (cur_dev >= 0 && cur_dev < 16 && !configured_hc_down[cur_dev]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        hc_down_proj_split_silu_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        hc_down_m64_n160_2cta_silu_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    configured_hc_down[cur_dev] = true;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  if (M >= 1536) {
    const int m_tiles = (M + 63) / 64;
    const int inj_tiles = (M + 319) / 320;
    hc_down_m64_n160_2cta_silu_kernel<<<2 * m_tiles + inj_tiles, 320, 0,
                                        stream>>>(
        reinterpret_cast<const half*>(x.data_ptr()),
        reinterpret_cast<const half*>(full_down.data_ptr()),
        reinterpret_cast<half*>(lora.data_ptr()),
        reinterpret_cast<half*>(inj.data_ptr()), M, m_tiles);
  } else {
    hc_down_proj_split_silu_kernel<<<dim3(6, (M + 63) / 64), 256, 0, stream>>>(
        reinterpret_cast<const half*>(x.data_ptr()),
        reinterpret_cast<const half*>(full_down.data_ptr()),
        reinterpret_cast<half*>(lora.data_ptr()),
        reinterpret_cast<half*>(inj.data_ptr()), M);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Cold-HBM optimized Split-K=2 Tensor Core HC down-projection (CTA_M=128,
// CTA_N=160
// + fused inj CTA_M=256, CTA_N=4 -> exact 80 CTAs at M=2048 across 80 V100
// SMs): Cuts HBM reads 2.5x vs cuBLAS 64x64 split-K=2 (180 MiB vs 448 MiB)!
__global__ __launch_bounds__(640, 1) void hc_down_m128_n160_splitk2_kernel(
    const half* __restrict__ x,          // [M, 10240]
    const half* __restrict__ full_down,  // [>=324, 10240]
    float* __restrict__ partial_f32,     // [2, >=M, 324], stride0 = ps0
    int M, int m_tiles, int inj_tiles, int64_t ps0) {
  constexpr int kHidden = 10240;
  constexpr int kHalfK = 5120;
  constexpr int kStepK = 32;
  constexpr int kNumStages = kHalfK / kStepK;  // 160

  union SharedHcDownSplitK {
    struct {
      uint4 s_A[2][32 * 17];
      uint4 s_B[2][40 * 17];
    } main128;
    struct {
      uint4 s_A[2][64 * 17];
      uint4 s_B[2][2 * 17];
    } inj256;
  };
  __shared__ __align__(16) SharedHcDownSplitK smem;

  const int tx = threadIdx.x;  // 0..639
  const int lane = tx & 31;
  const int warp = tx >> 5;
  const int main_ctas = 4 * m_tiles;

  if (blockIdx.x < main_ctas) {
    const int k_split = blockIdx.x / (2 * m_tiles);
    const int rem = blockIdx.x % (2 * m_tiles);
    const int n_blk = rem / m_tiles;  // 0..1 -> 160 cols each = 320 cols
    const int m_blk = rem % m_tiles;
    const int m_start = m_blk * 128;
    const int m_count = min(128, M - m_start);
    const int k_base = k_split * kHalfK;

    const int warp_m = warp / 5;  // 0..3 -> 32 rows each = 128 rows
    const int warp_n = warp % 5;  // 0..4 -> 32 cols each = 160 cols

    const int a_r = tx >> 2;
    const int a_s = tx & 3;
    const int a_smem_idx = (tx < 512) ? smem_idx17(a_r, a_s) : 0;
    const half* a_ptr =
        (tx < 512 && a_r < m_count)
            ? (x + static_cast<size_t>(m_start + a_r) * kHidden + k_base +
               a_s * 8)
            : nullptr;

    const int b_r = tx >> 2;  // 0..159 (all 640 threads!)
    const int b_s = tx & 3;   // 0..3
    const int b_smem_idx = smem_idx17(b_r, b_s);
    const half* b_ptr = full_down +
                        static_cast<size_t>(n_blk * 160 + b_r) * kHidden +
                        k_base + b_s * 8;

    {
      const uint4 av = a_ptr ? __ldg(reinterpret_cast<const uint4*>(a_ptr))
                             : make_uint4(0, 0, 0, 0);
      const uint4 bv = __ldg(reinterpret_cast<const uint4*>(b_ptr));
      if (tx < 512) smem.main128.s_A[0][a_smem_idx] = av;
      smem.main128.s_B[0][b_smem_idx] = bv;
    }
    __syncthreads();

    const int mma_row = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int quad = (lane >> 2) & 3;
    const int b_col = warp_n * 32 + quad * 8 + mma_row;
    const int m_base = warp_m * 32;
    const int r0 = m_base + mma_row;
    const int r1 = m_base + 8 + mma_row;
    const int r2 = m_base + 16 + mma_row;
    const int r3 = m_base + 24 + mma_row;

    float acc0[8] = {};
    float acc1[8] = {};
    float acc2[8] = {};
    float acc3[8] = {};

#pragma unroll 2
    for (int st = 0; st < kNumStages; ++st) {
      const int cur = st & 1;
      const int nxt = cur ^ 1;
      uint4 av = make_uint4(0, 0, 0, 0);
      uint4 bv = make_uint4(0, 0, 0, 0);
      if (st + 1 < kNumStages) {
        const int k_next = (st + 1) * kStepK;
        if (a_ptr) av = __ldg(reinterpret_cast<const uint4*>(a_ptr + k_next));
        bv = __ldg(reinterpret_cast<const uint4*>(b_ptr + k_next));
      }

#pragma unroll
      for (int sub = 0; sub < 2; ++sub) {
        const int sl0 = sub * 2;
        const int sl1 = sub * 2 + 1;
        const uint4 b0 = smem.main128.s_B[cur][smem_idx17(b_col, sl0)];
        const uint4 b1 = smem.main128.s_B[cur][smem_idx17(b_col, sl1)];
        const uint4 a0_lo = smem.main128.s_A[cur][smem_idx17(r0, sl0)];
        const uint4 a0_hi = smem.main128.s_A[cur][smem_idx17(r0, sl1)];
        const uint4 a1_lo = smem.main128.s_A[cur][smem_idx17(r1, sl0)];
        const uint4 a1_hi = smem.main128.s_A[cur][smem_idx17(r1, sl1)];
        FN_MMA_884_RC(acc0, a0_lo.x, a0_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc0, a0_lo.z, a0_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc0, a0_hi.x, a0_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc0, a0_hi.z, a0_hi.w, b1.z, b1.w);
        const uint4 a2_lo = smem.main128.s_A[cur][smem_idx17(r2, sl0)];
        const uint4 a2_hi = smem.main128.s_A[cur][smem_idx17(r2, sl1)];
        FN_MMA_884_RC(acc1, a1_lo.x, a1_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc1, a1_lo.z, a1_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc1, a1_hi.x, a1_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc1, a1_hi.z, a1_hi.w, b1.z, b1.w);
        const uint4 a3_lo = smem.main128.s_A[cur][smem_idx17(r3, sl0)];
        const uint4 a3_hi = smem.main128.s_A[cur][smem_idx17(r3, sl1)];
        FN_MMA_884_RC(acc2, a2_lo.x, a2_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc2, a2_lo.z, a2_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc2, a2_hi.x, a2_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc2, a2_hi.z, a2_hi.w, b1.z, b1.w);
        FN_MMA_884_RC(acc3, a3_lo.x, a3_lo.y, b0.x, b0.y);
        FN_MMA_884_RC(acc3, a3_lo.z, a3_lo.w, b0.z, b0.w);
        FN_MMA_884_RC(acc3, a3_hi.x, a3_hi.y, b1.x, b1.y);
        FN_MMA_884_RC(acc3, a3_hi.z, a3_hi.w, b1.z, b1.w);
      }

      if (st + 1 < kNumStages) {
        if (tx < 512) smem.main128.s_A[nxt][a_smem_idx] = av;
        smem.main128.s_B[nxt][b_smem_idx] = bv;
      }
      __syncthreads();
    }

    float* dst_split = partial_f32 + static_cast<int64_t>(k_split) * ps0;
    const int col_base = n_blk * 160 + warp_n * 32 + quad * 8;
    auto write_acc = [&](const float (&acc)[8], int row_base) {
#pragma unroll
      for (int i = 0; i < 8; i += 2) {
        const int r = row_base + ((i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1));
        if (r < m_count) {
          const int c = col_base + ((((lane >> 1) & 1) << 1) | ((i >> 2) << 2));
          *reinterpret_cast<float2*>(
              dst_split + static_cast<int64_t>(m_start + r) * 324 + c) =
              make_float2(acc[i], acc[i + 1]);
        }
      }
    };
    write_acc(acc0, m_base + 0);
    write_acc(acc1, m_base + 8);
    write_acc(acc2, m_base + 16);
    write_acc(acc3, m_base + 24);
  } else {
    const int rel = blockIdx.x - main_ctas;
    const int k_split = rel / inj_tiles;
    const int m_inj = rel % inj_tiles;
    if (k_split >= 2) return;
    const int m_start = m_inj * 256;
    const int m_count = min(256, M - m_start);
    const int k_base = k_split * kHalfK;

    const int a_r0 = tx >> 2;
    const int a_s = tx & 3;
    const int a_r1 = a_r0 + 128;
    const int a_smem0 = (tx < 512) ? smem_idx17(a_r0, a_s) : 0;
    const int a_smem1 = (tx < 512) ? smem_idx17(a_r1, a_s) : 0;
    const half* a_ptr0 =
        (tx < 512 && a_r0 < m_count)
            ? (x + static_cast<size_t>(m_start + a_r0) * kHidden + k_base +
               a_s * 8)
            : nullptr;
    const half* a_ptr1 =
        (tx < 512 && a_r1 < m_count)
            ? (x + static_cast<size_t>(m_start + a_r1) * kHidden + k_base +
               a_s * 8)
            : nullptr;

    const int b_r = tx >> 2;  // 0..7 for tx < 32 (cols 0..3 valid, 4..7 zero)
    const int b_s = tx & 3;
    const int b_smem_idx = (tx < 32) ? smem_idx17(b_r, b_s) : 0;
    const half* b_ptr =
        (tx < 16) ? (full_down + static_cast<size_t>(320 + b_r) * kHidden +
                     k_base + b_s * 8)
                  : nullptr;

    {
      if (tx < 512) {
        smem.inj256.s_A[0][a_smem0] =
            a_ptr0 ? __ldg(reinterpret_cast<const uint4*>(a_ptr0))
                   : make_uint4(0, 0, 0, 0);
        smem.inj256.s_A[0][a_smem1] =
            a_ptr1 ? __ldg(reinterpret_cast<const uint4*>(a_ptr1))
                   : make_uint4(0, 0, 0, 0);
      }
      if (tx < 32) {
        smem.inj256.s_B[0][b_smem_idx] =
            b_ptr ? __ldg(reinterpret_cast<const uint4*>(b_ptr))
                  : make_uint4(0, 0, 0, 0);
      }
    }
    __syncthreads();

    const int mma_row = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int quad = (lane >> 2) & 3;
    const int m_base = warp * 16;
    const int r0 = m_base + mma_row;
    const int r1 = m_base + 8 + mma_row;

    float acc0[8] = {};
    float acc1[8] = {};

#pragma unroll 2
    for (int st = 0; st < kNumStages; ++st) {
      const int cur = st & 1;
      const int nxt = cur ^ 1;
      uint4 av0 = make_uint4(0, 0, 0, 0);
      uint4 av1 = make_uint4(0, 0, 0, 0);
      uint4 bv = make_uint4(0, 0, 0, 0);
      if (st + 1 < kNumStages) {
        const int k_next = (st + 1) * kStepK;
        if (a_ptr0)
          av0 = __ldg(reinterpret_cast<const uint4*>(a_ptr0 + k_next));
        if (a_ptr1)
          av1 = __ldg(reinterpret_cast<const uint4*>(a_ptr1 + k_next));
        if (b_ptr) bv = __ldg(reinterpret_cast<const uint4*>(b_ptr + k_next));
      }

      if (warp < 16) {
#pragma unroll
        for (int sub = 0; sub < 2; ++sub) {
          const int sl0 = sub * 2;
          const int sl1 = sub * 2 + 1;
          const uint4 b0 = smem.inj256.s_B[cur][smem_idx17(mma_row, sl0)];
          const uint4 b1 = smem.inj256.s_B[cur][smem_idx17(mma_row, sl1)];
          const uint4 a0_lo = smem.inj256.s_A[cur][smem_idx17(r0, sl0)];
          const uint4 a0_hi = smem.inj256.s_A[cur][smem_idx17(r0, sl1)];
          const uint4 a1_lo = smem.inj256.s_A[cur][smem_idx17(r1, sl0)];
          const uint4 a1_hi = smem.inj256.s_A[cur][smem_idx17(r1, sl1)];
          FN_MMA_884_RC(acc0, a0_lo.x, a0_lo.y, b0.x, b0.y);
          FN_MMA_884_RC(acc0, a0_lo.z, a0_lo.w, b0.z, b0.w);
          FN_MMA_884_RC(acc0, a0_hi.x, a0_hi.y, b1.x, b1.y);
          FN_MMA_884_RC(acc0, a0_hi.z, a0_hi.w, b1.z, b1.w);
          FN_MMA_884_RC(acc1, a1_lo.x, a1_lo.y, b0.x, b0.y);
          FN_MMA_884_RC(acc1, a1_lo.z, a1_lo.w, b0.z, b0.w);
          FN_MMA_884_RC(acc1, a1_hi.x, a1_hi.y, b1.x, b1.y);
          FN_MMA_884_RC(acc1, a1_hi.z, a1_hi.w, b1.z, b1.w);
        }
      }

      if (st + 1 < kNumStages) {
        if (tx < 512) {
          smem.inj256.s_A[nxt][a_smem0] = av0;
          smem.inj256.s_A[nxt][a_smem1] = av1;
        }
        if (tx < 32) {
          smem.inj256.s_B[nxt][b_smem_idx] = bv;
        }
      }
      __syncthreads();
    }

    if (warp < 16 && quad == 0) {
      float* dst_split = partial_f32 + static_cast<int64_t>(k_split) * ps0;
      const int c = (((lane >> 1) & 1) << 1);  // 0 or 2 -> cols 0..3!
      auto write_inj = [&](const float (&acc)[8], int row_base) {
#pragma unroll
        for (int i = 0; i < 4; i += 2) {
          const int r =
              row_base + ((i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1));
          if (r < m_count) {
            *reinterpret_cast<float2*>(
                dst_split + static_cast<int64_t>(m_start + r) * 324 + 320 + c) =
                make_float2(acc[i], acc[i + 1]);
          }
        }
      };
      write_inj(acc0, m_base + 0);
      write_inj(acc1, m_base + 8);
    }
  }
}

__global__ __launch_bounds__(192, 4) void hc_reduce2_silu_kernel(
    const float* __restrict__ partial_f32, int64_t ps0, half* __restrict__ lora,
    half* __restrict__ inj) {
  const int m = blockIdx.x;
  const int tx = threadIdx.x;
  const float* p0 = partial_f32 + static_cast<int64_t>(m) * 324;
  const float* p1 = partial_f32 + ps0 + static_cast<int64_t>(m) * 324;
  if (tx < 160) {
    const float2 v0 = __ldg(reinterpret_cast<const float2*>(p0 + tx * 2));
    const float2 v1 = __ldg(reinterpret_cast<const float2*>(p1 + tx * 2));
    const half h0 = __float2half_rn(v0.x + v1.x);
    const half h1 = __float2half_rn(v0.y + v1.y);
    const float x0 = __half2float(h0) * 0.25f;
    const float x1 = __half2float(h1) * 0.25f;
    const half2 out2 =
        __floats2half2_rn(x0 / (1.0f + expf(-x0)), x1 / (1.0f + expf(-x1)));
    *reinterpret_cast<half2*>(lora + static_cast<size_t>(m) * 320 + tx * 2) =
        out2;
  } else if (tx < 162) {
    const int ic = (tx - 160) * 2;
    const float2 v0 = __ldg(reinterpret_cast<const float2*>(p0 + 320 + ic));
    const float2 v1 = __ldg(reinterpret_cast<const float2*>(p1 + 320 + ic));
    const half2 out2 = __floats2half2_rn(v0.x + v1.x, v0.y + v1.y);
    *reinterpret_cast<half2*>(inj + static_cast<size_t>(m) * 4 + ic) = out2;
  }
}

void hc_down_splitk2_silu_out(torch::Tensor lora, torch::Tensor inj,
                              torch::Tensor x, torch::Tensor full_down,
                              torch::Tensor partial_f32) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int M = static_cast<int>(x.size(0));
  if (M == 0) return;
  const int cur_dev = x.device().index();
  static bool configured_splitk2[16] = {};
  if (cur_dev >= 0 && cur_dev < 16 && !configured_splitk2[cur_dev]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        hc_down_m128_n160_splitk2_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    configured_splitk2[cur_dev] = true;
  }
  const int m_tiles = (M + 127) / 128;
  const int inj_tiles = (M + 255) / 256;
  const int grid = 4 * m_tiles + 2 * inj_tiles;
  const int64_t ps0 = partial_f32.stride(0);
  auto stream = at::cuda::getCurrentCUDAStream();
  hc_down_m128_n160_splitk2_kernel<<<grid, 640, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr()),
      reinterpret_cast<const half*>(full_down.data_ptr()),
      partial_f32.data_ptr<float>(), M, m_tiles, inj_tiles, ps0);
  hc_reduce2_silu_kernel<<<M, 192, 0, stream>>>(
      partial_f32.data_ptr<float>(), ps0,
      reinterpret_cast<half*>(lora.data_ptr()),
      reinterpret_cast<half*>(inj.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ __launch_bounds__(256) void hc_copy_rows_to_ipc_kernel(
    half* __restrict__ ipc_base, PeerPtrs4 sync_ptrs,
    unsigned* __restrict__ ep_ptr, int my_rank, int64_t bank_elems,
    const uint4* __restrict__ src, int64_t total_vecs) {
  const int slot = static_cast<int>(*ep_ptr & 1u);
  uint4* dst = reinterpret_cast<uint4*>(ipc_base + static_cast<int64_t>(slot) *
                                                       bank_elems);
  for (int64_t v = static_cast<int64_t>(blockIdx.x) * 256 + threadIdx.x;
       v < total_vecs; v += static_cast<int64_t>(gridDim.x) * 256) {
    dst[v] = __ldg(src + v);
  }
  hc_p2p_signal_last_cta(sync_ptrs, ep_ptr, my_rank);
}

void hc_p2p_allgather_rows_out(torch::Tensor out, torch::Tensor src, int64_t p0,
                               int64_t p1, int64_t p2, int64_t p3, int64_t s0,
                               int64_t s1, int64_t s2, int64_t s3,
                               torch::Tensor ep_state, int64_t bank_bytes,
                               int64_t my_rank) {
  const c10::cuda::CUDAGuard guard(src.device());
  const int local_rows = static_cast<int>(src.size(0));
  const int out_rows = static_cast<int>(out.size(0));
  if (local_rows == 0 || out_rows == 0) return;
  const int64_t row_bytes =
      static_cast<int64_t>(src.size(1)) * src.element_size();
  TORCH_CHECK((row_bytes % 16) == 0 &&
              static_cast<int64_t>(local_rows) * row_bytes <= bank_bytes);
  const int half_width = static_cast<int>(row_bytes / sizeof(half));
  const int64_t bank_elems = bank_bytes / sizeof(half);
  const int64_t total_vecs =
      static_cast<int64_t>(local_rows) * (half_width / 8);
  auto stream = at::cuda::getCurrentCUDAStream();

  PeerPtrs4 ptrs{{static_cast<uintptr_t>(p0), static_cast<uintptr_t>(p1),
                  static_cast<uintptr_t>(p2), static_cast<uintptr_t>(p3)}};
  PeerPtrs4 sync_ptrs{{static_cast<uintptr_t>(s0), static_cast<uintptr_t>(s1),
                       static_cast<uintptr_t>(s2), static_cast<uintptr_t>(s3)}};
  unsigned* ep_ptr = reinterpret_cast<unsigned*>(ep_state.data_ptr<int32_t>());
  const int rk = static_cast<int>(my_rank);

  const int copy_ctas =
      static_cast<int>(std::min<int64_t>((total_vecs + 255) / 256, 160));
  const int local_g_ctas = (total_vecs <= 32768) ? 4 : 16;
  const int peer_g_ctas = static_cast<int>(
      std::max<int64_t>(4, std::min<int64_t>((total_vecs + 1023) / 1024, 80)));
  const int gather_ctas = local_g_ctas + 3 * peer_g_ctas;
  hc_copy_rows_to_ipc_kernel<<<copy_ctas, 256, 0, stream>>>(
      reinterpret_cast<half*>(ptrs.p[rk]), sync_ptrs, ep_ptr, rk, bank_elems,
      reinterpret_cast<const uint4*>(src.data_ptr()), total_vecs);
  hc_p2p_gather_rows_kernel<<<gather_ctas, 256, 0, stream>>>(
      ptrs, sync_ptrs.p[rk], ep_ptr, bank_elems,
      reinterpret_cast<half*>(out.data_ptr()), rk, local_rows, out_rows,
      half_width, local_g_ctas);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename IdT>
__global__ __launch_bounds__(256) void hc_stage_topk_to_ipc_kernel(
    half* __restrict__ ipc_base, PeerPtrs4 sync_ptrs,
    unsigned* __restrict__ ep_ptr, int my_rank, int64_t bank_elems,
    const IdT* __restrict__ loc_ids, const float* __restrict__ loc_w, int n_loc,
    int step, int k, int row_u4s) {
  const int slot = static_cast<int>(*ep_ptr & 1u);
  uint4* dst_base = reinterpret_cast<uint4*>(
      ipc_base + static_cast<int64_t>(slot) * bank_elems);
  const int total_u4s = step * row_u4s;
  for (int v = blockIdx.x * 256 + threadIdx.x; v < total_u4s;
       v += gridDim.x * 256) {
    const int r = v / row_u4s;
    const int c4 = (v - r * row_u4s) * 4;
    uint4 out_v = make_uint4(0, 0, 0, 0);
    if (r < n_loc) {
      int32_t vals[4] = {0, 0, 0, 0};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int col = c4 + j;
        if (col < k) {
          vals[j] =
              static_cast<int32_t>(loc_ids[static_cast<int64_t>(r) * k + col]);
        } else if (col < 2 * k) {
          vals[j] =
              __float_as_int(loc_w[static_cast<int64_t>(r) * k + (col - k)]);
        }
      }
      out_v = *reinterpret_cast<const uint4*>(vals);
    }
    dst_base[v] = out_v;
  }
  hc_p2p_signal_last_cta(sync_ptrs, ep_ptr, my_rank);
}

template <typename IdT>
__global__ __launch_bounds__(256) void hc_gather_split_topk_kernel(
    PeerPtrs4 ptrs, uintptr_t my_sync_ptr, const unsigned* __restrict__ ep_ptr,
    int64_t bank_elems, IdT* __restrict__ out_ids, float* __restrict__ out_w,
    int my_rank, int step, int out_rows, int k, int row_u4s) {
  const unsigned ep = *ep_ptr;
  hc_p2p_wait_peers(my_sync_ptr, ep);
  const int slot = static_cast<int>((ep - 1u) & 1u);
  const int total_u4s = out_rows * row_u4s;
  for (int v = blockIdx.x * 256 + threadIdx.x; v < total_u4s;
       v += gridDim.x * 256) {
    const int r = v / row_u4s;
    const int u = v - r * row_u4s;
    const int owner = r / step;
    const int loc_r = r - owner * step;
    const uint4* src_base = reinterpret_cast<const uint4*>(
        reinterpret_cast<const half*>(ptrs.p[owner]) +
        static_cast<int64_t>(slot) * bank_elems);
    const uint4* src_ptr = src_base + static_cast<int64_t>(loc_r) * row_u4s + u;
    const uint4 raw =
        (owner == my_rank) ? __ldg(src_ptr) : p2p_ld_cv_u4(src_ptr);
    const int32_t* vals = reinterpret_cast<const int32_t*>(&raw);
    const int c4 = u * 4;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int col = c4 + j;
      if (col < k) {
        out_ids[static_cast<int64_t>(r) * k + col] = static_cast<IdT>(vals[j]);
      } else if (col < 2 * k) {
        out_w[static_cast<int64_t>(r) * k + (col - k)] =
            __int_as_float(vals[j]);
      }
    }
  }
}

void hc_p2p_allgather_topk_out(torch::Tensor out_w, torch::Tensor out_ids,
                               torch::Tensor loc_w, torch::Tensor loc_ids,
                               int64_t step, int64_t p0, int64_t p1, int64_t p2,
                               int64_t p3, int64_t s0, int64_t s1, int64_t s2,
                               int64_t s3, torch::Tensor ep_state,
                               int64_t bank_bytes, int64_t my_rank) {
  const c10::cuda::CUDAGuard guard(loc_w.device());
  const int n_loc = static_cast<int>(loc_ids.size(0));
  const int out_rows = static_cast<int>(out_ids.size(0));
  const int k = static_cast<int>(loc_ids.size(1));
  if (out_rows <= 0 || step <= 0 || k <= 0) return;
  const int row_u4s = (2 * k + 3) / 4;
  TORCH_CHECK(step * row_u4s * 16LL <= bank_bytes);
  const int64_t bank_elems = bank_bytes / sizeof(half);
  auto stream = at::cuda::getCurrentCUDAStream();

  PeerPtrs4 ptrs{{static_cast<uintptr_t>(p0), static_cast<uintptr_t>(p1),
                  static_cast<uintptr_t>(p2), static_cast<uintptr_t>(p3)}};
  PeerPtrs4 sync_ptrs{{static_cast<uintptr_t>(s0), static_cast<uintptr_t>(s1),
                       static_cast<uintptr_t>(s2), static_cast<uintptr_t>(s3)}};
  unsigned* ep_ptr = reinterpret_cast<unsigned*>(ep_state.data_ptr<int32_t>());
  const int rk = static_cast<int>(my_rank);

  const int stage_u4s = static_cast<int>(step) * row_u4s;
  const int gather_u4s = out_rows * row_u4s;
  const int stage_ctas = std::max(1, std::min((stage_u4s + 255) / 256, 80));
  const int gather_ctas = std::max(1, std::min((gather_u4s + 255) / 256, 160));
  if (loc_ids.scalar_type() == at::kInt) {
    hc_stage_topk_to_ipc_kernel<int32_t><<<stage_ctas, 256, 0, stream>>>(
        reinterpret_cast<half*>(ptrs.p[rk]), sync_ptrs, ep_ptr, rk, bank_elems,
        loc_ids.data_ptr<int32_t>(), loc_w.data_ptr<float>(), n_loc,
        static_cast<int>(step), k, row_u4s);
    hc_gather_split_topk_kernel<int32_t><<<gather_ctas, 256, 0, stream>>>(
        ptrs, sync_ptrs.p[rk], ep_ptr, bank_elems, out_ids.data_ptr<int32_t>(),
        out_w.data_ptr<float>(), rk, static_cast<int>(step), out_rows, k,
        row_u4s);
  } else {
    hc_stage_topk_to_ipc_kernel<int64_t><<<stage_ctas, 256, 0, stream>>>(
        reinterpret_cast<half*>(ptrs.p[rk]), sync_ptrs, ep_ptr, rk, bank_elems,
        loc_ids.data_ptr<int64_t>(), loc_w.data_ptr<float>(), n_loc,
        static_cast<int>(step), k, row_u4s);
    hc_gather_split_topk_kernel<int64_t><<<gather_ctas, 256, 0, stream>>>(
        ptrs, sync_ptrs.p[rk], ep_ptr, bank_elems, out_ids.data_ptr<int64_t>(),
        out_w.data_ptr<float>(), rk, static_cast<int>(step), out_rows, k,
        row_u4s);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void hc_split_silu_out(torch::Tensor lora, torch::Tensor inj,
                       torch::Tensor projected) {
  const c10::cuda::CUDAGuard guard(projected.device());
  const int M = static_cast<int>(projected.size(0));
  if (M == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream();
  hc_split_silu_kernel<<<M, 88, 0, stream>>>(
      reinterpret_cast<const half*>(projected.data_ptr()),
      reinterpret_cast<half*>(lora.data_ptr()),
      reinterpret_cast<half*>(inj.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ __launch_bounds__(320, 3) void hc_gate_mix_push4_kernel(
    PeerPtrs4 ag_ptrs, const unsigned* __restrict__ ep_ptr, int64_t bank_elems,
    const half* __restrict__ x, const half* __restrict__ gate, int my_rank,
    int local_rows) {
  const int slot = static_cast<int>(*ep_ptr & 1u);
  const int row = blockIdx.x;
  const int c8 = threadIdx.x * 8;
  float acc[8] = {};
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    const int64_t off = (static_cast<int64_t>(row) * 4 + s) * 2560 + c8;
    const uint4 xu = __ldg(reinterpret_cast<const uint4*>(x + off));
    const uint4 gu = __ldg(reinterpret_cast<const uint4*>(gate + off));
    const half* xh = reinterpret_cast<const half*>(&xu);
    const half* gh = reinterpret_cast<const half*>(&gu);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float gf = __half2float(gh[j]);
      const float sig = 1.0f / (1.0f + expf(-gf));
      acc[j] = __fmaf_rn(sig, __half2float(xh[j]), acc[j]);
    }
  }
  uint4 out_u4;
  half* oh = reinterpret_cast<half*>(&out_u4);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    oh[j] = __float2half_rn(acc[j] * 0.25f);
  }
  const int64_t dst_off =
      static_cast<int64_t>(slot) * bank_elems +
      static_cast<int64_t>(my_rank * local_rows + row) * 2560 + c8;
  const int p0 = my_rank;
  const int p1 = (my_rank + 1) & 3;
  const int p2 = (my_rank + 2) & 3;
  const int p3 = (my_rank + 3) & 3;
  *reinterpret_cast<uint4*>(reinterpret_cast<half*>(ag_ptrs.p[p0]) + dst_off) =
      out_u4;
  *reinterpret_cast<uint4*>(reinterpret_cast<half*>(ag_ptrs.p[p1]) + dst_off) =
      out_u4;
  *reinterpret_cast<uint4*>(reinterpret_cast<half*>(ag_ptrs.p[p2]) + dst_off) =
      out_u4;
  *reinterpret_cast<uint4*>(reinterpret_cast<half*>(ag_ptrs.p[p3]) + dst_off) =
      out_u4;
}

void hc_p2p_gate_mix_push4_out(torch::Tensor x, torch::Tensor gate, int64_t a0,
                               int64_t a1, int64_t a2, int64_t a3, int64_t s0,
                               int64_t s1, int64_t s2, int64_t s3,
                               torch::Tensor ep_state, int64_t bank_bytes,
                               int64_t my_rank) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int local_rows = static_cast<int>(x.size(0));
  if (local_rows == 0) return;
  const int64_t bank_elems = bank_bytes / sizeof(half);
  auto stream = at::cuda::getCurrentCUDAStream();

  PeerPtrs4 ag_ptrs{{static_cast<uintptr_t>(a0), static_cast<uintptr_t>(a1),
                     static_cast<uintptr_t>(a2), static_cast<uintptr_t>(a3)}};
  PeerPtrs4 sync_ptrs{{static_cast<uintptr_t>(s0), static_cast<uintptr_t>(s1),
                       static_cast<uintptr_t>(s2), static_cast<uintptr_t>(s3)}};
  unsigned* ep_ptr = reinterpret_cast<unsigned*>(ep_state.data_ptr<int32_t>());

  hc_gate_mix_push4_kernel<<<local_rows, 320, 0, stream>>>(
      ag_ptrs, ep_ptr, bank_elems, reinterpret_cast<const half*>(x.data_ptr()),
      reinterpret_cast<const half*>(gate.data_ptr()), static_cast<int>(my_rank),
      local_rows);
  hc_p2p_barrier_kernel<<<1, 32, 0, stream>>>(sync_ptrs, ep_ptr,
                                              static_cast<int>(my_rank));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Writes my_rank's mixed rows directly into [my_rank * local_rows ..] inside
// my_ipc[slot]
template <int SLOT_SHIFT, bool INLINE_SIGNAL = false>
__global__ __launch_bounds__(320, 3) void hc_gate_mix_to_ipc_slot_kernel(
    half* __restrict__ ag_base, PeerPtrs4 sync_ptrs,
    unsigned* __restrict__ ep_ptr, int64_t bank_elems,
    const half* __restrict__ x, const half* __restrict__ gate, int my_rank,
    int local_rows) {
  const int slot = static_cast<int>((*ep_ptr >> SLOT_SHIFT) & 1u);
  half* dst = ag_base + static_cast<int64_t>(slot) * bank_elems +
              static_cast<int64_t>(my_rank * local_rows) * 2560;
  const int c8 = threadIdx.x * 8;
  for (int row = blockIdx.x; row < local_rows; row += gridDim.x) {
    float acc[8] = {};
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const int64_t off = (static_cast<int64_t>(row) * 4 + s) * 2560 + c8;
      const uint4 xu = __ldg(reinterpret_cast<const uint4*>(x + off));
      const uint4 gu = __ldg(reinterpret_cast<const uint4*>(gate + off));
      const half* xh = reinterpret_cast<const half*>(&xu);
      const half* gh = reinterpret_cast<const half*>(&gu);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float gf = __half2float(gh[j]);
        const float sig = 1.0f / (1.0f + expf(-gf));
        acc[j] = __fmaf_rn(sig, __half2float(xh[j]), acc[j]);
      }
    }
    uint4 out_u4;
    half* oh = reinterpret_cast<half*>(&out_u4);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      oh[j] = __float2half_rn(acc[j] * 0.25f);
    }
    *reinterpret_cast<uint4*>(dst + static_cast<int64_t>(row) * 2560 + c8) =
        out_u4;
  }
  if constexpr (INLINE_SIGNAL) {
    hc_p2p_signal_last_cta(sync_ptrs, ep_ptr, my_rank);
  }
}

// Zero-local-copy 3-peer parallel NVLink RX pull (240 CTAs = 80 CTAs/peer,
// inline wait!)
__global__ __launch_bounds__(256, 3) void hc_p2p_gather_peers_only_kernel(
    PeerPtrs4 ag_ptrs, uintptr_t my_sync_ptr,
    const unsigned* __restrict__ ep_ptr, int64_t bank_elems, int my_rank,
    int local_rows, int out_rows) {
  const unsigned ep = *ep_ptr;
  if (my_sync_ptr != 0) {
    hc_p2p_wait_peers(my_sync_ptr, ep);
  }
  const int slot = static_cast<int>((ep - 1u) & 1u);
  const int p = blockIdx.x % 3;
  const int owner = (my_rank + 1 + p) & 3;
  const int cta_idx = blockIdx.x / 3;
  const int num_ctas = gridDim.x / 3;

  const int g_base = owner * local_rows;
  const int valid_rows = max(0, min(local_rows, out_rows - g_base));
  const int64_t peer_vecs = static_cast<int64_t>(valid_rows) * 320;
  const int64_t stride = static_cast<int64_t>(num_ctas) * 256;
  const int64_t tid = static_cast<int64_t>(cta_idx) * 256 + threadIdx.x;

  const int64_t region_off = static_cast<int64_t>(slot) * bank_elems +
                             static_cast<int64_t>(g_base) * 2560;
  const uintptr_t owner_ptr =
      (owner == 0)
          ? ag_ptrs.p[0]
          : ((owner == 1) ? ag_ptrs.p[1]
                          : ((owner == 2) ? ag_ptrs.p[2] : ag_ptrs.p[3]));
  const uintptr_t my_ptr =
      (my_rank == 0)
          ? ag_ptrs.p[0]
          : ((my_rank == 1) ? ag_ptrs.p[1]
                            : ((my_rank == 2) ? ag_ptrs.p[2] : ag_ptrs.p[3]));
  const uint4* src = reinterpret_cast<const uint4*>(
      reinterpret_cast<const half*>(owner_ptr) + region_off);
  uint4* dst =
      reinterpret_cast<uint4*>(reinterpret_cast<half*>(my_ptr) + region_off);

  int64_t v = tid;
  const int64_t s3 = stride * 3;
  const int64_t s4 = stride * 4;
  for (; v + s3 < peer_vecs; v += s4) {
    const uint4 u0 = p2p_ld_cv_u4(src + v);
    const uint4 u1 = p2p_ld_cv_u4(src + v + stride);
    const uint4 u2 = p2p_ld_cv_u4(src + v + stride * 2);
    const uint4 u3 = p2p_ld_cv_u4(src + v + s3);
    dst[v] = u0;
    dst[v + stride] = u1;
    dst[v + stride * 2] = u2;
    dst[v + s3] = u3;
  }
  for (; v < peer_vecs; v += stride) {
    dst[v] = p2p_ld_cv_u4(src + v);
  }
}

// Full-Duplex Bi-Directional NVLink2 (150 GB/s TX + 150 GB/s RX = 300 GB/s!):
// First 120 CTAs (40/peer) PUSH my_rank's first half [0..H) over TX;
// Second 120 CTAs (40/peer) PULL each peer's second half [H..peer_vecs) over
// RX!
__global__ __launch_bounds__(256, 3) void hc_p2p_duplex_peers_kernel(
    PeerPtrs4 ag_ptrs, const unsigned* __restrict__ ep_ptr, int64_t bank_elems,
    int my_rank, int local_rows, int out_rows) {
  const int slot = static_cast<int>(((*ep_ptr - 1u) >> 1) & 1u);
  const int half_ctas = gridDim.x / 2;  // 120

  if (blockIdx.x < half_ctas) {
    // TX PUSH half [0 .. my_half_vecs) from my_rank to target peer
    const int p = blockIdx.x % 3;
    const int dst_peer = (my_rank + 1 + p) & 3;
    const int cta_idx = blockIdx.x / 3;
    const int num_ctas = half_ctas / 3;

    const int my_g_base = my_rank * local_rows;
    const int my_valid_rows = max(0, min(local_rows, out_rows - my_g_base));
    const int64_t my_vecs = static_cast<int64_t>(my_valid_rows) * 320;
    const int64_t push_vecs = my_vecs / 2;
    const int64_t stride = static_cast<int64_t>(num_ctas) * 256;
    const int64_t tid = static_cast<int64_t>(cta_idx) * 256 + threadIdx.x;

    const int64_t region_off = static_cast<int64_t>(slot) * bank_elems +
                               static_cast<int64_t>(my_g_base) * 2560;
    const uint4* src = reinterpret_cast<const uint4*>(
        reinterpret_cast<const half*>(ag_ptrs.p[my_rank]) + region_off);
    uint4* dst = reinterpret_cast<uint4*>(
        reinterpret_cast<half*>(ag_ptrs.p[dst_peer]) + region_off);

    int64_t v = tid;
    const int64_t s3 = stride * 3;
    const int64_t s4 = stride * 4;
    for (; v + s3 < push_vecs; v += s4) {
      const uint4 u0 = __ldg(src + v);
      const uint4 u1 = __ldg(src + v + stride);
      const uint4 u2 = __ldg(src + v + stride * 2);
      const uint4 u3 = __ldg(src + v + s3);
      dst[v] = u0;
      dst[v + stride] = u1;
      dst[v + stride * 2] = u2;
      dst[v + s3] = u3;
    }
    for (; v < push_vecs; v += stride) {
      dst[v] = __ldg(src + v);
    }
  } else {
    // RX PULL second half [peer_half_vecs .. peer_vecs) from src_peer to
    // my_rank
    const int rel = blockIdx.x - half_ctas;
    const int p = rel % 3;
    const int src_peer = (my_rank + 1 + p) & 3;
    const int cta_idx = rel / 3;
    const int num_ctas = half_ctas / 3;

    const int peer_g_base = src_peer * local_rows;
    const int peer_valid_rows = max(0, min(local_rows, out_rows - peer_g_base));
    const int64_t peer_vecs = static_cast<int64_t>(peer_valid_rows) * 320;
    const int64_t start_vec = peer_vecs / 2;
    const int64_t stride = static_cast<int64_t>(num_ctas) * 256;
    const int64_t tid =
        start_vec + static_cast<int64_t>(cta_idx) * 256 + threadIdx.x;

    const int64_t region_off = static_cast<int64_t>(slot) * bank_elems +
                               static_cast<int64_t>(peer_g_base) * 2560;
    const uint4* src = reinterpret_cast<const uint4*>(
        reinterpret_cast<const half*>(ag_ptrs.p[src_peer]) + region_off);
    uint4* dst = reinterpret_cast<uint4*>(
        reinterpret_cast<half*>(ag_ptrs.p[my_rank]) + region_off);

    int64_t v = tid;
    const int64_t s3 = stride * 3;
    const int64_t s4 = stride * 4;
    for (; v + s3 < peer_vecs; v += s4) {
      const uint4 u0 = src[v];
      const uint4 u1 = src[v + stride];
      const uint4 u2 = src[v + stride * 2];
      const uint4 u3 = src[v + s3];
      dst[v] = u0;
      dst[v + stride] = u1;
      dst[v + stride * 2] = u2;
      dst[v + s3] = u3;
    }
    for (; v < peer_vecs; v += stride) {
      dst[v] = src[v];
    }
  }
}

void hc_p2p_gate_mix_local_only_out(torch::Tensor x, torch::Tensor gate,
                                    int64_t a0, int64_t a1, int64_t a2,
                                    int64_t a3, int64_t s0, int64_t s1,
                                    int64_t s2, int64_t s3,
                                    torch::Tensor ep_state, int64_t bank_bytes,
                                    int64_t my_rank) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int local_rows = static_cast<int>(x.size(0));
  if (local_rows == 0) return;
  const int64_t bank_elems = bank_bytes / sizeof(half);
  auto stream = at::cuda::getCurrentCUDAStream();

  PeerPtrs4 ag_ptrs{{static_cast<uintptr_t>(a0), static_cast<uintptr_t>(a1),
                     static_cast<uintptr_t>(a2), static_cast<uintptr_t>(a3)}};
  PeerPtrs4 sync_ptrs{{static_cast<uintptr_t>(s0), static_cast<uintptr_t>(s1),
                       static_cast<uintptr_t>(s2), static_cast<uintptr_t>(s3)}};
  unsigned* ep_ptr = reinterpret_cast<unsigned*>(ep_state.data_ptr<int32_t>());

  const int rk = static_cast<int>(my_rank);
  const int mix_ctas = std::min(local_rows, 240);
  hc_gate_mix_to_ipc_slot_kernel<0, true><<<mix_ctas, 320, 0, stream>>>(
      reinterpret_cast<half*>(ag_ptrs.p[rk]), sync_ptrs, ep_ptr, bank_elems,
      reinterpret_cast<const half*>(x.data_ptr()),
      reinterpret_cast<const half*>(gate.data_ptr()), rk, local_rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void hc_p2p_gather_peers_async_out(int64_t out_rows, int64_t local_rows,
                                   int64_t a0, int64_t a1, int64_t a2,
                                   int64_t a3, int64_t my_sync,
                                   torch::Tensor ep_state, int64_t bank_bytes,
                                   int64_t my_rank, int64_t num_ctas) {
  const c10::cuda::CUDAGuard guard(ep_state.device());
  if (local_rows <= 0) return;
  const int64_t bank_elems = bank_bytes / sizeof(half);
  auto stream = at::cuda::getCurrentCUDAStream();

  PeerPtrs4 ag_ptrs{{static_cast<uintptr_t>(a0), static_cast<uintptr_t>(a1),
                     static_cast<uintptr_t>(a2), static_cast<uintptr_t>(a3)}};
  unsigned* ep_ptr = reinterpret_cast<unsigned*>(ep_state.data_ptr<int32_t>());
  const int rk = static_cast<int>(my_rank);
  const int ctas = std::max(3, ((static_cast<int>(num_ctas) + 2) / 3) * 3);
  hc_p2p_gather_peers_only_kernel<<<ctas, 256, 0, stream>>>(
      ag_ptrs, static_cast<uintptr_t>(my_sync), ep_ptr, bank_elems, rk,
      static_cast<int>(local_rows), static_cast<int>(out_rows));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void hc_p2p_gate_mix_peers_pull_out(torch::Tensor x, torch::Tensor gate,
                                    int64_t out_rows, int64_t a0, int64_t a1,
                                    int64_t a2, int64_t a3, int64_t s0,
                                    int64_t s1, int64_t s2, int64_t s3,
                                    torch::Tensor ep_state, int64_t bank_bytes,
                                    int64_t my_rank) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int local_rows = static_cast<int>(x.size(0));
  if (local_rows == 0) return;
  const int64_t bank_elems = bank_bytes / sizeof(half);
  auto stream = at::cuda::getCurrentCUDAStream();

  PeerPtrs4 ag_ptrs{{static_cast<uintptr_t>(a0), static_cast<uintptr_t>(a1),
                     static_cast<uintptr_t>(a2), static_cast<uintptr_t>(a3)}};
  PeerPtrs4 sync_ptrs{{static_cast<uintptr_t>(s0), static_cast<uintptr_t>(s1),
                       static_cast<uintptr_t>(s2), static_cast<uintptr_t>(s3)}};
  unsigned* ep_ptr = reinterpret_cast<unsigned*>(ep_state.data_ptr<int32_t>());

  const int rk = static_cast<int>(my_rank);
  const int mix_ctas = std::min(local_rows, 240);
  hc_gate_mix_to_ipc_slot_kernel<0, true><<<mix_ctas, 320, 0, stream>>>(
      reinterpret_cast<half*>(ag_ptrs.p[rk]), sync_ptrs, ep_ptr, bank_elems,
      reinterpret_cast<const half*>(x.data_ptr()),
      reinterpret_cast<const half*>(gate.data_ptr()), rk, local_rows);
  hc_p2p_gather_peers_only_kernel<<<240, 256, 0, stream>>>(
      ag_ptrs, sync_ptrs.p[rk], ep_ptr, bank_elems, rk, local_rows,
      static_cast<int>(out_rows));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void hc_p2p_gate_mix_duplex_out(torch::Tensor x, torch::Tensor gate,
                                int64_t out_rows, int64_t a0, int64_t a1,
                                int64_t a2, int64_t a3, int64_t s0, int64_t s1,
                                int64_t s2, int64_t s3, torch::Tensor ep_state,
                                int64_t bank_bytes, int64_t my_rank) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int local_rows = static_cast<int>(x.size(0));
  if (local_rows == 0) return;
  const int64_t bank_elems = bank_bytes / sizeof(half);
  auto stream = at::cuda::getCurrentCUDAStream();

  PeerPtrs4 ag_ptrs{{static_cast<uintptr_t>(a0), static_cast<uintptr_t>(a1),
                     static_cast<uintptr_t>(a2), static_cast<uintptr_t>(a3)}};
  PeerPtrs4 sync_ptrs{{static_cast<uintptr_t>(s0), static_cast<uintptr_t>(s1),
                       static_cast<uintptr_t>(s2), static_cast<uintptr_t>(s3)}};
  unsigned* ep_ptr = reinterpret_cast<unsigned*>(ep_state.data_ptr<int32_t>());

  const int mix_ctas = std::min(local_rows, 240);
  hc_gate_mix_to_ipc_slot_kernel<1, false><<<mix_ctas, 320, 0, stream>>>(
      reinterpret_cast<half*>(ag_ptrs.p[my_rank]), sync_ptrs, ep_ptr,
      bank_elems, reinterpret_cast<const half*>(x.data_ptr()),
      reinterpret_cast<const half*>(gate.data_ptr()), static_cast<int>(my_rank),
      local_rows);
  hc_p2p_barrier_kernel<<<1, 32, 0, stream>>>(sync_ptrs, ep_ptr,
                                              static_cast<int>(my_rank));
  hc_p2p_duplex_peers_kernel<<<240, 256, 0, stream>>>(
      ag_ptrs, ep_ptr, bank_elems, static_cast<int>(my_rank), local_rows,
      static_cast<int>(out_rows));
  hc_p2p_barrier_kernel<<<1, 32, 0, stream>>>(sync_ptrs, ep_ptr,
                                              static_cast<int>(my_rank));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void hc_p2p_gate_mix_allgather_out(torch::Tensor block_out, torch::Tensor x,
                                   torch::Tensor gate, int64_t a0, int64_t a1,
                                   int64_t a2, int64_t a3, int64_t s0,
                                   int64_t s1, int64_t s2, int64_t s3,
                                   torch::Tensor ep_state, int64_t bank_bytes,
                                   int64_t my_rank) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int local_rows = static_cast<int>(x.size(0));
  if (local_rows == 0) return;
  const int out_rows = static_cast<int>(block_out.size(0));
  const int64_t bank_elems = bank_bytes / sizeof(half);
  auto stream = at::cuda::getCurrentCUDAStream();

  PeerPtrs4 ag_ptrs{{static_cast<uintptr_t>(a0), static_cast<uintptr_t>(a1),
                     static_cast<uintptr_t>(a2), static_cast<uintptr_t>(a3)}};
  PeerPtrs4 sync_ptrs{{static_cast<uintptr_t>(s0), static_cast<uintptr_t>(s1),
                       static_cast<uintptr_t>(s2), static_cast<uintptr_t>(s3)}};
  unsigned* ep_ptr = reinterpret_cast<unsigned*>(ep_state.data_ptr<int32_t>());
  const int rk = static_cast<int>(my_rank);

  hc_gate_mix_to_ipc_kernel<<<local_rows, 320, 0, stream>>>(
      reinterpret_cast<half*>(ag_ptrs.p[rk]), sync_ptrs, ep_ptr, rk, bank_elems,
      reinterpret_cast<const half*>(x.data_ptr()),
      reinterpret_cast<const half*>(gate.data_ptr()), local_rows);
  hc_p2p_gather_rows_kernel<<<256, 256, 0, stream>>>(
      ag_ptrs, sync_ptrs.p[rk], ep_ptr, bank_elems,
      reinterpret_cast<half*>(block_out.data_ptr()), rk, local_rows, out_rows,
      2560, 16);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Zero-spill SM70 Tensor Core HC Up-Projection + 4-Stream Sigmoid Gate Mix ->
// IPC: CTA_M=32, CTA_C=128: preloads [32, 320] lora ONCE into s_A[10] (38.25
// KiB <= 48 KiB
// -> 2 CTAs/SM, 0 spills!) and eliminates 80 MiB [M_loc, 10240] gate HBM
// traffic!
__global__ __launch_bounds__(256, 2) void hc_up_gate_mix_to_ipc_kernel(
    half* __restrict__ ag_base, const unsigned* __restrict__ ep_ptr,
    int64_t bank_elems, const half* __restrict__ x,
    const half* __restrict__ lora, const half* __restrict__ full_up,
    int local_rows) {
  const int c_blk = blockIdx.x;  // 0..19 -> 128 cols each = 2560 cols
  const int m_start = blockIdx.y * 32;
  if (m_start >= local_rows) return;
  const int m_count = min(32, local_rows - m_start);

  constexpr int kLora = 320;
  constexpr int kStepK = 32;
  constexpr int kNumStages = kLora / kStepK;  // 10

  union SharedHcUpMixV2 {
    struct {
      uint4 s_A[10][8 * 17];  // 21,760 B: entire [32, 320] lora preloaded ONCE!
      uint4 s_B[2][32 * 17];  // 17,408 B: double-buffered [128, 32] up_buf
    } mainloop;               // 39,168 B <= 48 KiB -> 2 CTAs/SM!
    half s_out[32][136];      //  8,704 B
  };
  __shared__ __align__(16) SharedHcUpMixV2 smem;

  const int tx = threadIdx.x;    // 0..255
  const int lane = tx & 31;      // 0..31
  const int warp = tx >> 5;      // 0..7
  const int warp_m = warp >> 2;  // 0..1 -> 16 rows each = 32 rows
  const int warp_c = warp & 3;   // 0..3 -> 32 cols each = 128 cols

  // Preload entire [32, 320] lora tile = 1280 uint4s = exact 5 uint4s/thread!
#pragma unroll
  for (int i = 0; i < 5; ++i) {
    const int vid = tx + i * 256;
    const int r = vid / 40;
    const int c_vec = vid % 40;
    const int st_a = c_vec >> 2;
    const int sl_a = c_vec & 3;
    uint4 val = make_uint4(0, 0, 0, 0);
    if (r < m_count) {
      val = __ldg(reinterpret_cast<const uint4*>(
          lora + static_cast<size_t>(m_start + r) * kLora + c_vec * 8));
    }
    smem.mainloop.s_A[st_a][smem_idx17(r, sl_a)] = val;
  }

  const int b_r0 = tx >> 2;    // 0..63
  const int b_r1 = b_r0 + 64;  // 64..127
  const int b_s = tx & 3;      // 0..3
  const int b_smem_idx0 = smem_idx17(b_r0, b_s);
  const int b_smem_idx1 = smem_idx17(b_r1, b_s);

  const int mma_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int quad = (lane >> 2) & 3;
  const int b_col = warp_c * 32 + quad * 8 + mma_row;
  const int m_base = warp_m * 16;
  const int r0 = m_base + mma_row;
  const int r1 = m_base + 8 + mma_row;

  float mix0[8] = {}, mix1[8] = {};

#pragma unroll 1
  for (int s_idx = 0; s_idx < 4; ++s_idx) {
    const half* up_base =
        full_up + static_cast<size_t>(s_idx * 2560 + c_blk * 128) * kLora +
        b_s * 8;
    const half* b_ptr0 = up_base + static_cast<size_t>(b_r0) * kLora;
    const half* b_ptr1 = up_base + static_cast<size_t>(b_r1) * kLora;

    {
      const uint4 bv0 = __ldg(reinterpret_cast<const uint4*>(b_ptr0));
      const uint4 bv1 = __ldg(reinterpret_cast<const uint4*>(b_ptr1));
      smem.mainloop.s_B[0][b_smem_idx0] = bv0;
      smem.mainloop.s_B[0][b_smem_idx1] = bv1;
    }
    __syncthreads();

    float acc0[8] = {}, acc1[8] = {};

#pragma unroll 2
    for (int st = 0; st < kNumStages; ++st) {
      const int cur = st & 1;
      const int nxt = cur ^ 1;
      uint4 bv0 = make_uint4(0, 0, 0, 0);
      uint4 bv1 = make_uint4(0, 0, 0, 0);
      if (st + 1 < kNumStages) {
        const int k_next = (st + 1) * kStepK;
        bv0 = __ldg(reinterpret_cast<const uint4*>(b_ptr0 + k_next));
        bv1 = __ldg(reinterpret_cast<const uint4*>(b_ptr1 + k_next));
      }

      if (m_base < m_count) {
#pragma unroll
        for (int sub = 0; sub < 2; ++sub) {
          const int sl0 = sub * 2;
          const int sl1 = sub * 2 + 1;
          const uint4 b0 = smem.mainloop.s_B[cur][smem_idx17(b_col, sl0)];
          const uint4 b1 = smem.mainloop.s_B[cur][smem_idx17(b_col, sl1)];
          const uint4 a0_lo = smem.mainloop.s_A[st][smem_idx17(r0, sl0)];
          const uint4 a0_hi = smem.mainloop.s_A[st][smem_idx17(r0, sl1)];
          const uint4 a1_lo = smem.mainloop.s_A[st][smem_idx17(r1, sl0)];
          const uint4 a1_hi = smem.mainloop.s_A[st][smem_idx17(r1, sl1)];
          FN_MMA_884_RC(acc0, a0_lo.x, a0_lo.y, b0.x, b0.y);
          FN_MMA_884_RC(acc0, a0_lo.z, a0_lo.w, b0.z, b0.w);
          FN_MMA_884_RC(acc0, a0_hi.x, a0_hi.y, b1.x, b1.y);
          FN_MMA_884_RC(acc0, a0_hi.z, a0_hi.w, b1.z, b1.w);
          FN_MMA_884_RC(acc1, a1_lo.x, a1_lo.y, b0.x, b0.y);
          FN_MMA_884_RC(acc1, a1_lo.z, a1_lo.w, b0.z, b0.w);
          FN_MMA_884_RC(acc1, a1_hi.x, a1_hi.y, b1.x, b1.y);
          FN_MMA_884_RC(acc1, a1_hi.z, a1_hi.w, b1.z, b1.w);
        }
      }

      if (st + 1 < kNumStages) {
        smem.mainloop.s_B[nxt][b_smem_idx0] = bv0;
        smem.mainloop.s_B[nxt][b_smem_idx1] = bv1;
      }
      __syncthreads();
    }

    auto fold_stream = [&](const float (&acc)[8], float (&mix)[8],
                           int row_base) {
      if (row_base >= m_count) return;
#pragma unroll
      for (int i = 0; i < 8; i += 2) {
        const int r = row_base + ((i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1));
        if (r < m_count) {
          const int c = warp_c * 32 + quad * 8 +
                        ((((lane >> 1) & 1) << 1) | ((i >> 2) << 2));
          const half2 x2 = __ldg(reinterpret_cast<const half2*>(
              x + (static_cast<int64_t>(m_start + r) * 4 + s_idx) * 2560 +
              c_blk * 128 + c));
          const float g0 = __half2float(__float2half_rn(acc[i]));
          const float g1 = __half2float(__float2half_rn(acc[i + 1]));
          const float sig0 = 1.0f / (1.0f + expf(-g0));
          const float sig1 = 1.0f / (1.0f + expf(-g1));
          mix[i] = __fmaf_rn(sig0, __half2float(__low2half(x2)), mix[i]);
          mix[i + 1] =
              __fmaf_rn(sig1, __half2float(__high2half(x2)), mix[i + 1]);
        }
      }
    };
    fold_stream(acc0, mix0, m_base + 0);
    fold_stream(acc1, mix1, m_base + 8);
  }

  auto stage_out = [&](const float (&mix)[8], int row_base) {
    if (row_base >= m_count) return;
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
      const int r = row_base + ((i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1));
      if (r < m_count) {
        const int c = warp_c * 32 + quad * 8 +
                      ((((lane >> 1) & 1) << 1) | ((i >> 2) << 2));
        *reinterpret_cast<half2*>(&smem.s_out[r][c]) =
            __floats2half2_rn(mix[i] * 0.25f, mix[i + 1] * 0.25f);
      }
    }
  };
  stage_out(mix0, m_base + 0);
  stage_out(mix1, m_base + 8);
  __syncthreads();

  const int slot = static_cast<int>(*ep_ptr & 1u);
  half* dst = ag_base + static_cast<int64_t>(slot) * bank_elems;
#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int vid = tx + u * 256;  // 0..511
    const int orow = vid >> 4;     // 0..31
    const int ocol8 = (vid & 15) * 8;
    if (orow < m_count) {
      *reinterpret_cast<uint4*>(
          dst + static_cast<int64_t>(m_start + orow) * 2560 + c_blk * 128 +
          ocol8) = *reinterpret_cast<const uint4*>(&smem.s_out[orow][ocol8]);
    }
  }
}

// Zero-spill CTA_M=128, CTA_C=128 Tensor Core HC Up-Projection + 4-Stream Gate
// Mix -> IPC! 512 threads = 16 warps/SM, 34,816 B smem <= 96 KiB, 1 av + 1 bv
// per thread -> <=120 regs, 0 spills! Eliminates 168 MB [M_loc, 10240] gate HBM
// write+read AND 4x reduces up_buf HBM vs CTA_M=32!
__global__ __launch_bounds__(512, 1) void hc_up128_gate_mix_to_ipc_kernel(
    half* __restrict__ ag_base, const unsigned* __restrict__ ep_ptr,
    int64_t bank_elems, const half* __restrict__ x,
    const half* __restrict__ lora, const half* __restrict__ full_up,
    int local_rows) {
  const int c_blk = blockIdx.x;  // 0..19 -> 128 cols each = 2560 cols
  const int m_start = blockIdx.y * 128;
  if (m_start >= local_rows) return;
  const int m_count = min(128, local_rows - m_start);

  constexpr int kLora = 320;
  constexpr int kStepK = 32;
  constexpr int kNumStages = kLora / kStepK;  // 10

  union SharedHcUp128Mix {
    struct {
      uint4 s_A[2][32 * 17];  // 17,408 B: double-buffered [128, 32] lora
      uint4 s_B[2][32 * 17];  // 17,408 B: double-buffered [128, 32] up_buf
    } mainloop;               // 34,816 B <= 96 KiB -> 16 warps/SM, 0 spills!
    half s_out[128][136];     // 34,816 B
  };
  __shared__ __align__(16) SharedHcUp128Mix smem;

  const int tx = threadIdx.x;    // 0..511
  const int lane = tx & 31;      // 0..31
  const int warp = tx >> 5;      // 0..15
  const int warp_m = warp >> 2;  // 0..3 -> 32 rows each = 128 rows
  const int warp_c = warp & 3;   // 0..3 -> 32 cols each = 128 cols

  const int ab_r = tx >> 2;  // 0..127 (1 uint4 av and 1 uint4 bv per thread!)
  const int ab_s = tx & 3;   // 0..3
  const int ab_smem_idx = smem_idx17(ab_r, ab_s);
  const half* a_base =
      (ab_r < m_count)
          ? (lora + static_cast<size_t>(m_start + ab_r) * kLora + ab_s * 8)
          : nullptr;

  const int mma_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int quad = (lane >> 2) & 3;
  const int b_col = warp_c * 32 + quad * 8 + mma_row;
  const int m_base = warp_m * 32;
  const int r0 = m_base + mma_row;

  float mix0[8] = {}, mix1[8] = {}, mix2[8] = {}, mix3[8] = {};

#pragma unroll 1
  for (int s_idx = 0; s_idx < 4; ++s_idx) {
    const half* b_ptr =
        full_up +
        static_cast<size_t>(s_idx * 2560 + c_blk * 128 + ab_r) * kLora +
        ab_s * 8;

    {
      const uint4 av = a_base ? __ldg(reinterpret_cast<const uint4*>(a_base))
                              : make_uint4(0, 0, 0, 0);
      const uint4 bv = __ldg(reinterpret_cast<const uint4*>(b_ptr));
      smem.mainloop.s_A[0][ab_smem_idx] = av;
      smem.mainloop.s_B[0][ab_smem_idx] = bv;
    }
    __syncthreads();

    float acc0[8] = {}, acc1[8] = {}, acc2[8] = {}, acc3[8] = {};

#pragma unroll 1
    for (int st = 0; st < kNumStages; ++st) {
      const int cur = st & 1;
      const int nxt = cur ^ 1;
      uint4 av = make_uint4(0, 0, 0, 0);
      uint4 bv = make_uint4(0, 0, 0, 0);
      if (st + 1 < kNumStages) {
        const int k_next = (st + 1) * kStepK;
        if (a_base) av = __ldg(reinterpret_cast<const uint4*>(a_base + k_next));
        bv = __ldg(reinterpret_cast<const uint4*>(b_ptr + k_next));
      }

      if (m_base < m_count) {
#pragma unroll
        for (int sub = 0; sub < 2; ++sub) {
          const int sl0 = sub * 2;
          const int sl1 = sub * 2 + 1;
          const uint4 b0 = smem.mainloop.s_B[cur][smem_idx17(b_col, sl0)];
          const uint4 b1 = smem.mainloop.s_B[cur][smem_idx17(b_col, sl1)];
          uint4 a_lo = smem.mainloop.s_A[cur][smem_idx17(r0, sl0)];
          uint4 a_hi = smem.mainloop.s_A[cur][smem_idx17(r0, sl1)];
          FN_MMA_884_RC(acc0, a_lo.x, a_lo.y, b0.x, b0.y);
          FN_MMA_884_RC(acc0, a_lo.z, a_lo.w, b0.z, b0.w);
          FN_MMA_884_RC(acc0, a_hi.x, a_hi.y, b1.x, b1.y);
          FN_MMA_884_RC(acc0, a_hi.z, a_hi.w, b1.z, b1.w);
          a_lo = smem.mainloop.s_A[cur][smem_idx17(r0 + 8, sl0)];
          a_hi = smem.mainloop.s_A[cur][smem_idx17(r0 + 8, sl1)];
          FN_MMA_884_RC(acc1, a_lo.x, a_lo.y, b0.x, b0.y);
          FN_MMA_884_RC(acc1, a_lo.z, a_lo.w, b0.z, b0.w);
          FN_MMA_884_RC(acc1, a_hi.x, a_hi.y, b1.x, b1.y);
          FN_MMA_884_RC(acc1, a_hi.z, a_hi.w, b1.z, b1.w);
          a_lo = smem.mainloop.s_A[cur][smem_idx17(r0 + 16, sl0)];
          a_hi = smem.mainloop.s_A[cur][smem_idx17(r0 + 16, sl1)];
          FN_MMA_884_RC(acc2, a_lo.x, a_lo.y, b0.x, b0.y);
          FN_MMA_884_RC(acc2, a_lo.z, a_lo.w, b0.z, b0.w);
          FN_MMA_884_RC(acc2, a_hi.x, a_hi.y, b1.x, b1.y);
          FN_MMA_884_RC(acc2, a_hi.z, a_hi.w, b1.z, b1.w);
          a_lo = smem.mainloop.s_A[cur][smem_idx17(r0 + 24, sl0)];
          a_hi = smem.mainloop.s_A[cur][smem_idx17(r0 + 24, sl1)];
          FN_MMA_884_RC(acc3, a_lo.x, a_lo.y, b0.x, b0.y);
          FN_MMA_884_RC(acc3, a_lo.z, a_lo.w, b0.z, b0.w);
          FN_MMA_884_RC(acc3, a_hi.x, a_hi.y, b1.x, b1.y);
          FN_MMA_884_RC(acc3, a_hi.z, a_hi.w, b1.z, b1.w);
        }
      }

      if (st + 1 < kNumStages) {
        smem.mainloop.s_A[nxt][ab_smem_idx] = av;
        smem.mainloop.s_B[nxt][ab_smem_idx] = bv;
      }
      __syncthreads();
    }

    auto fold_stream = [&](const float (&acc)[8], float (&mix)[8],
                           int row_base) {
      if (row_base >= m_count) return;
#pragma unroll
      for (int i = 0; i < 8; i += 2) {
        const int r = row_base + ((i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1));
        if (r < m_count) {
          const int c = warp_c * 32 + quad * 8 +
                        ((((lane >> 1) & 1) << 1) | ((i >> 2) << 2));
          const half2 x2 = __ldg(reinterpret_cast<const half2*>(
              x + (static_cast<int64_t>(m_start + r) * 4 + s_idx) * 2560 +
              c_blk * 128 + c));
          const float g0 = __half2float(__float2half_rn(acc[i]));
          const float g1 = __half2float(__float2half_rn(acc[i + 1]));
          const float sig0 = 1.0f / (1.0f + expf(-g0));
          const float sig1 = 1.0f / (1.0f + expf(-g1));
          mix[i] = __fmaf_rn(sig0, __half2float(__low2half(x2)), mix[i]);
          mix[i + 1] =
              __fmaf_rn(sig1, __half2float(__high2half(x2)), mix[i + 1]);
        }
      }
    };
    fold_stream(acc0, mix0, m_base + 0);
    fold_stream(acc1, mix1, m_base + 8);
    fold_stream(acc2, mix2, m_base + 16);
    fold_stream(acc3, mix3, m_base + 24);
    if (s_idx < 3) __syncthreads();
  }

  auto stage_out = [&](const float (&mix)[8], int row_base) {
    if (row_base >= m_count) return;
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
      const int r = row_base + ((i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1));
      if (r < m_count) {
        const int c = warp_c * 32 + quad * 8 +
                      ((((lane >> 1) & 1) << 1) | ((i >> 2) << 2));
        *reinterpret_cast<half2*>(&smem.s_out[r][c]) =
            __floats2half2_rn(mix[i] * 0.25f, mix[i + 1] * 0.25f);
      }
    }
  };
  stage_out(mix0, m_base + 0);
  stage_out(mix1, m_base + 8);
  stage_out(mix2, m_base + 16);
  stage_out(mix3, m_base + 24);
  __syncthreads();

  const int slot = static_cast<int>(*ep_ptr & 1u);
  half* dst = ag_base + static_cast<int64_t>(slot) * bank_elems;
#pragma unroll
  for (int u = 0; u < 4; ++u) {
    const int vid = tx + u * 512;  // 0..2047
    const int orow = vid >> 4;     // 0..127
    const int ocol8 = (vid & 15) * 8;
    if (orow < m_count) {
      *reinterpret_cast<uint4*>(
          dst + static_cast<int64_t>(m_start + orow) * 2560 + c_blk * 128 +
          ocol8) = *reinterpret_cast<const uint4*>(&smem.s_out[orow][ocol8]);
    }
  }
}

void hc_p2p_up_gate_mix_allgather_out(torch::Tensor block_out, torch::Tensor x,
                                      torch::Tensor lora, torch::Tensor full_up,
                                      int64_t a0, int64_t a1, int64_t a2,
                                      int64_t a3, int64_t s0, int64_t s1,
                                      int64_t s2, int64_t s3,
                                      torch::Tensor ep_state,
                                      int64_t bank_bytes, int64_t my_rank) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int local_rows = static_cast<int>(x.size(0));
  if (local_rows == 0) return;
  const int out_rows = static_cast<int>(block_out.size(0));
  const int64_t bank_elems = bank_bytes / sizeof(half);
  const int cur_dev = x.device().index();
  static bool configured_hc_up[16] = {};
  if (cur_dev >= 0 && cur_dev < 16 && !configured_hc_up[cur_dev]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        hc_up_gate_mix_to_ipc_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        hc_up128_gate_mix_to_ipc_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    configured_hc_up[cur_dev] = true;
  }
  auto stream = at::cuda::getCurrentCUDAStream();

  PeerPtrs4 ag_ptrs{{static_cast<uintptr_t>(a0), static_cast<uintptr_t>(a1),
                     static_cast<uintptr_t>(a2), static_cast<uintptr_t>(a3)}};
  PeerPtrs4 sync_ptrs{{static_cast<uintptr_t>(s0), static_cast<uintptr_t>(s1),
                       static_cast<uintptr_t>(s2), static_cast<uintptr_t>(s3)}};
  unsigned* ep_ptr = reinterpret_cast<unsigned*>(ep_state.data_ptr<int32_t>());

  if (local_rows >= 1024) {
    hc_up128_gate_mix_to_ipc_kernel<<<dim3(20, (local_rows + 127) / 128), 512,
                                      0, stream>>>(
        reinterpret_cast<half*>(ag_ptrs.p[my_rank]), ep_ptr, bank_elems,
        reinterpret_cast<const half*>(x.data_ptr()),
        reinterpret_cast<const half*>(lora.data_ptr()),
        reinterpret_cast<const half*>(full_up.data_ptr()), local_rows);
  } else {
    hc_up_gate_mix_to_ipc_kernel<<<dim3(20, (local_rows + 31) / 32), 256, 0,
                                   stream>>>(
        reinterpret_cast<half*>(ag_ptrs.p[my_rank]), ep_ptr, bank_elems,
        reinterpret_cast<const half*>(x.data_ptr()),
        reinterpret_cast<const half*>(lora.data_ptr()),
        reinterpret_cast<const half*>(full_up.data_ptr()), local_rows);
  }
  hc_p2p_barrier_kernel<<<1, 32, 0, stream>>>(sync_ptrs, ep_ptr,
                                              static_cast<int>(my_rank));
  hc_p2p_gather_rows_kernel<<<256, 256, 0, stream>>>(
      ag_ptrs, 0, ep_ptr, bank_elems,
      reinterpret_cast<half*>(block_out.data_ptr()), static_cast<int>(my_rank),
      local_rows, out_rows, 2560);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// All-4-Stream Fused SM70 Tensor Core HC Up-Projection + Gate Mix -> IPC slot!
// grid = (80, (local_rows + 63) / 64), 256 threads, __launch_bounds__(256, 2),
// 26,112 B smem <= 48 KiB -> 2 CTAs/SM = 16 warps/SM across all 80 V100 SMs!
// Computes CTA_M=64 rows x CTA_C=32 cols across ALL 4 STREAMS in ONE K=320 pass
// (loads lora ONCE instead of 4x!), stages exact FP16 gate into 16B-aligned
// s_gate[64][4][40], reads x via 100% coalesced uint4, and writes ag_slot!
template <bool INLINE_SIGNAL>
__global__ __launch_bounds__(256, 2) void hc_up_all4_gate_mix_to_ipc_kernel(
    half* __restrict__ ag_base, PeerPtrs4 sync_ptrs,
    unsigned* __restrict__ ep_ptr, int64_t bank_elems,
    const half* __restrict__ x, const half* __restrict__ lora,
    const half* __restrict__ full_up, int my_rank, int local_rows) {
  const int c_blk = blockIdx.x;  // 0..79 -> 32 cols each = 2560 cols
  const int m_start = blockIdx.y * 64;
  const int m_count = max(0, min(64, local_rows - m_start));
  const int slot = static_cast<int>(*ep_ptr & 1u);

  if (m_count > 0) {
    constexpr int kLora = 320;
    constexpr int kStepK = 32;
    constexpr int kNumStages = kLora / kStepK;  // 10

    union SharedHcUpAll4 {
      struct {
        uint4 s_A[2][16 * 17];  //  8,704 B: [64, 32] lora
        uint4 s_B[2][32 * 17];  // 17,408 B: [4 * 32, 32] full_up
      } mainloop;               // 26,112 B <= 48 KiB -> 2 CTAs/SM!
      alignas(
          16) half s_gate[64][4][40];  // 20,480 B (80B stride = 16B aligned!)
    };
    __shared__ __align__(16) SharedHcUpAll4 smem;

    const int tx = threadIdx.x;    // 0..255
    const int lane = tx & 31;      // 0..31
    const int warp = tx >> 5;      // 0..7
    const int warp_m = warp >> 2;  // 0..1 -> 32 rows each = 64 rows
    const int warp_s = warp & 3;   // 0..3 -> stream 0..3 (32 cols each!)

    const int a_r = tx >> 2;  // 0..63
    const int a_s = tx & 3;   // 0..3
    const int a_smem_idx = smem_idx17(a_r, a_s);
    const half* a_base =
        (a_r < m_count)
            ? (lora + static_cast<size_t>(m_start + a_r) * kLora + a_s * 8)
            : nullptr;

    const int b_r0 = tx >> 2;    // 0..63  -> streams 0..1, cols 0..31
    const int b_r1 = b_r0 + 64;  // 64..127 -> streams 2..3, cols 0..31
    const int b_s = tx & 3;      // 0..3
    const int b_smem_idx0 = smem_idx17(b_r0, b_s);
    const int b_smem_idx1 = smem_idx17(b_r1, b_s);
    const half* b_ptr0 =
        full_up +
        static_cast<size_t>((b_r0 >> 5) * 2560 + c_blk * 32 + (b_r0 & 31)) *
            kLora +
        b_s * 8;
    const half* b_ptr1 =
        full_up +
        static_cast<size_t>((b_r1 >> 5) * 2560 + c_blk * 32 + (b_r1 & 31)) *
            kLora +
        b_s * 8;

    {
      const uint4 av = a_base ? __ldg(reinterpret_cast<const uint4*>(a_base))
                              : make_uint4(0, 0, 0, 0);
      const uint4 bv0 = __ldg(reinterpret_cast<const uint4*>(b_ptr0));
      const uint4 bv1 = __ldg(reinterpret_cast<const uint4*>(b_ptr1));
      smem.mainloop.s_A[0][a_smem_idx] = av;
      smem.mainloop.s_B[0][b_smem_idx0] = bv0;
      smem.mainloop.s_B[0][b_smem_idx1] = bv1;
    }
    __syncthreads();

    const int mma_row = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int quad = (lane >> 2) & 3;
    const int b_col = warp_s * 32 + quad * 8 + mma_row;
    const int m_base = warp_m * 32;
    const int r0 = m_base + mma_row;

    float acc0[8] = {}, acc1[8] = {}, acc2[8] = {}, acc3[8] = {};

#pragma unroll
    for (int st = 0; st < kNumStages; ++st) {
      const int cur = st & 1;
      const int nxt = cur ^ 1;
      uint4 av = make_uint4(0, 0, 0, 0);
      uint4 bv0 = make_uint4(0, 0, 0, 0);
      uint4 bv1 = make_uint4(0, 0, 0, 0);
      if (st + 1 < kNumStages) {
        const int k_next = (st + 1) * kStepK;
        if (a_base) av = __ldg(reinterpret_cast<const uint4*>(a_base + k_next));
        bv0 = __ldg(reinterpret_cast<const uint4*>(b_ptr0 + k_next));
        bv1 = __ldg(reinterpret_cast<const uint4*>(b_ptr1 + k_next));
      }

      if (m_base < m_count) {
#pragma unroll
        for (int sl = 0; sl < 4; ++sl) {
          const uint4 b = smem.mainloop.s_B[cur][smem_idx17(b_col, sl)];
          uint4 a = smem.mainloop.s_A[cur][smem_idx17(r0, sl)];
          FN_MMA_884_RC(acc0, a.x, a.y, b.x, b.y);
          FN_MMA_884_RC(acc0, a.z, a.w, b.z, b.w);
          a = smem.mainloop.s_A[cur][smem_idx17(r0 + 8, sl)];
          FN_MMA_884_RC(acc1, a.x, a.y, b.x, b.y);
          FN_MMA_884_RC(acc1, a.z, a.w, b.z, b.w);
          a = smem.mainloop.s_A[cur][smem_idx17(r0 + 16, sl)];
          FN_MMA_884_RC(acc2, a.x, a.y, b.x, b.y);
          FN_MMA_884_RC(acc2, a.z, a.w, b.z, b.w);
          a = smem.mainloop.s_A[cur][smem_idx17(r0 + 24, sl)];
          FN_MMA_884_RC(acc3, a.x, a.y, b.x, b.y);
          FN_MMA_884_RC(acc3, a.z, a.w, b.z, b.w);
        }
      }

      if (st + 1 < kNumStages) {
        smem.mainloop.s_A[nxt][a_smem_idx] = av;
        smem.mainloop.s_B[nxt][b_smem_idx0] = bv0;
        smem.mainloop.s_B[nxt][b_smem_idx1] = bv1;
      }
      __syncthreads();
    }

    const int r_sub = ((lane & 16) ? 4 : 0) | (lane & 1);
    const int c_sub = quad * 8 + (((lane >> 1) & 1) << 1);
    auto stage_gate8 = [&](const float (&acc)[8], int row_base) {
      if (row_base >= m_count) return;
      const int rb0 = row_base + r_sub;
      const int rb2 = rb0 + 2;
      *reinterpret_cast<half2*>(&smem.s_gate[rb0][warp_s][c_sub]) =
          __floats2half2_rn(acc[0], acc[1]);
      *reinterpret_cast<half2*>(&smem.s_gate[rb2][warp_s][c_sub]) =
          __floats2half2_rn(acc[2], acc[3]);
      *reinterpret_cast<half2*>(&smem.s_gate[rb0][warp_s][c_sub + 4]) =
          __floats2half2_rn(acc[4], acc[5]);
      *reinterpret_cast<half2*>(&smem.s_gate[rb2][warp_s][c_sub + 4]) =
          __floats2half2_rn(acc[6], acc[7]);
    };
    stage_gate8(acc0, m_base + 0);
    stage_gate8(acc1, m_base + 8);
    stage_gate8(acc2, m_base + 16);
    stage_gate8(acc3, m_base + 24);
    __syncthreads();

    const int orow = tx >> 2;        // 0..63
    const int ocol8 = (tx & 3) * 8;  // 0, 8, 16, 24
    if (orow < m_count) {
      const int col_glb = c_blk * 32 + ocol8;
      float mix[8] = {};
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const int64_t x_off =
            (static_cast<int64_t>(m_start + orow) * 4 + s) * 2560 + col_glb;
        const uint4 xu = __ldg(reinterpret_cast<const uint4*>(x + x_off));
        const uint4 gu =
            *reinterpret_cast<const uint4*>(&smem.s_gate[orow][s][ocol8]);
        const half* xh = reinterpret_cast<const half*>(&xu);
        const half* gh = reinterpret_cast<const half*>(&gu);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const float gf = __half2float(gh[j]);
          const float sig = 1.0f / (1.0f + expf(-gf));
          mix[j] = __fmaf_rn(sig, __half2float(xh[j]), mix[j]);
        }
      }
      uint4 out_u4;
      half* oh = reinterpret_cast<half*>(&out_u4);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        oh[j] = __float2half_rn(mix[j] * 0.25f);
      }
      half* dst = ag_base + static_cast<int64_t>(slot) * bank_elems +
                  static_cast<int64_t>(my_rank * local_rows) * 2560;
      *reinterpret_cast<uint4*>(
          dst + static_cast<int64_t>(m_start + orow) * 2560 + col_glb) = out_u4;
    }
  }
  if constexpr (INLINE_SIGNAL) {
    hc_p2p_signal_last_cta(sync_ptrs, ep_ptr, my_rank);
  }
}

static void ensure_hc_up_all4_configured(int cur_dev) {
  static bool configured_hc_up_all4[16] = {};
  if (cur_dev >= 0 && cur_dev < 16 && !configured_hc_up_all4[cur_dev]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        hc_up_all4_gate_mix_to_ipc_kernel<true>,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        hc_up_all4_gate_mix_to_ipc_kernel<false>,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    configured_hc_up_all4[cur_dev] = true;
  }
}

void hc_p2p_up_gate_mix_local_only_out(torch::Tensor x, torch::Tensor lora,
                                       torch::Tensor full_up, int64_t a0,
                                       int64_t a1, int64_t a2, int64_t a3,
                                       int64_t s0, int64_t s1, int64_t s2,
                                       int64_t s3, torch::Tensor ep_state,
                                       int64_t bank_bytes, int64_t my_rank) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int local_rows = static_cast<int>(x.size(0));
  if (local_rows == 0) return;
  const int64_t bank_elems = bank_bytes / sizeof(half);
  ensure_hc_up_all4_configured(x.device().index());
  auto stream = at::cuda::getCurrentCUDAStream();

  PeerPtrs4 ag_ptrs{{static_cast<uintptr_t>(a0), static_cast<uintptr_t>(a1),
                     static_cast<uintptr_t>(a2), static_cast<uintptr_t>(a3)}};
  PeerPtrs4 sync_ptrs{{static_cast<uintptr_t>(s0), static_cast<uintptr_t>(s1),
                       static_cast<uintptr_t>(s2), static_cast<uintptr_t>(s3)}};
  unsigned* ep_ptr = reinterpret_cast<unsigned*>(ep_state.data_ptr<int32_t>());
  const int rk = static_cast<int>(my_rank);

  hc_up_all4_gate_mix_to_ipc_kernel<true>
      <<<dim3(80, (local_rows + 63) / 64), 256, 0, stream>>>(
          reinterpret_cast<half*>(ag_ptrs.p[rk]), sync_ptrs, ep_ptr, bank_elems,
          reinterpret_cast<const half*>(x.data_ptr()),
          reinterpret_cast<const half*>(lora.data_ptr()),
          reinterpret_cast<const half*>(full_up.data_ptr()), rk, local_rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void hc_p2p_up_gate_mix_peers_pull_out(torch::Tensor x, torch::Tensor lora,
                                       torch::Tensor full_up, int64_t out_rows,
                                       int64_t a0, int64_t a1, int64_t a2,
                                       int64_t a3, int64_t s0, int64_t s1,
                                       int64_t s2, int64_t s3,
                                       torch::Tensor ep_state,
                                       int64_t bank_bytes, int64_t my_rank) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int local_rows = static_cast<int>(x.size(0));
  if (local_rows == 0) return;
  const int64_t bank_elems = bank_bytes / sizeof(half);
  ensure_hc_up_all4_configured(x.device().index());
  auto stream = at::cuda::getCurrentCUDAStream();

  PeerPtrs4 ag_ptrs{{static_cast<uintptr_t>(a0), static_cast<uintptr_t>(a1),
                     static_cast<uintptr_t>(a2), static_cast<uintptr_t>(a3)}};
  PeerPtrs4 sync_ptrs{{static_cast<uintptr_t>(s0), static_cast<uintptr_t>(s1),
                       static_cast<uintptr_t>(s2), static_cast<uintptr_t>(s3)}};
  unsigned* ep_ptr = reinterpret_cast<unsigned*>(ep_state.data_ptr<int32_t>());
  const int rk = static_cast<int>(my_rank);

  hc_up_all4_gate_mix_to_ipc_kernel<true>
      <<<dim3(80, (local_rows + 63) / 64), 256, 0, stream>>>(
          reinterpret_cast<half*>(ag_ptrs.p[rk]), sync_ptrs, ep_ptr, bank_elems,
          reinterpret_cast<const half*>(x.data_ptr()),
          reinterpret_cast<const half*>(lora.data_ptr()),
          reinterpret_cast<const half*>(full_up.data_ptr()), rk, local_rows);
  hc_p2p_gather_peers_only_kernel<<<240, 256, 0, stream>>>(
      ag_ptrs, sync_ptrs.p[rk], ep_ptr, bank_elems, rk, local_rows,
      static_cast<int>(out_rows));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename WeightT>
__global__ __launch_bounds__(256, 4) void gdn_rmsnorm_gated_h12d128_kernel(
    half* __restrict__ out, const half* __restrict__ x,
    const half* __restrict__ z, const WeightT* __restrict__ weight, int64_t zs0,
    int total_heads, float eps) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  float wf[4];
  if constexpr (std::is_same_v<WeightT, float>) {
    const float4 wf4 =
        __ldg(reinterpret_cast<const float4*>(weight + lane * 4));
    wf[0] = wf4.x;
    wf[1] = wf4.y;
    wf[2] = wf4.z;
    wf[3] = wf4.w;
  } else {
    const uint2 wu2 = __ldg(reinterpret_cast<const uint2*>(weight + lane * 4));
    const half* wh = reinterpret_cast<const half*>(&wu2);
#pragma unroll
    for (int j = 0; j < 4; ++j) wf[j] = __half2float(wh[j]);
  }

  for (int idx = blockIdx.x * 8 + warp; idx < total_heads;
       idx += gridDim.x * 8) {
    const int m = idx / 12;
    const int h = idx - m * 12;
    const uint2 xu2 = __ldg(reinterpret_cast<const uint2*>(
        x + static_cast<int64_t>(idx) * 128 + lane * 4));
    const uint2 zu2 = __ldg(reinterpret_cast<const uint2*>(
        z + static_cast<int64_t>(m) * zs0 + h * 128 + lane * 4));
    const half* xh = reinterpret_cast<const half*>(&xu2);
    const half* zh = reinterpret_cast<const half*>(&zu2);

    float xf[4];
    float sq = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      xf[j] = __half2float(xh[j]);
      sq = __fmaf_rn(xf[j], xf[j], sq);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      sq += __shfl_xor_sync(0xffffffffu, sq, o);
    }
    const float rstd = rsqrtf(sq * (1.0f / 128.0f) + eps);

    uint2 yu2;
    half* yh = reinterpret_cast<half*>(&yu2);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float zf = __half2float(zh[j]);
      const float silu_z = zf / (1.0f + expf(-zf));
      yh[j] = __float2half_rn((xf[j] * rstd * wf[j]) * silu_z);
    }
    *reinterpret_cast<uint2*>(out + static_cast<int64_t>(idx) * 128 +
                              lane * 4) = yu2;
  }
}

void gdn_rmsnorm_gated_h12d128_out(torch::Tensor out, torch::Tensor x,
                                   torch::Tensor z, torch::Tensor weight,
                                   double eps) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int M = static_cast<int>(x.size(0));
  if (M == 0) return;
  const int total_heads = M * 12;
  const int ctas = std::min((total_heads + 7) / 8, 240);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (weight.scalar_type() == at::kFloat) {
    gdn_rmsnorm_gated_h12d128_kernel<float><<<ctas, 256, 0, stream>>>(
        reinterpret_cast<half*>(out.data_ptr()),
        reinterpret_cast<const half*>(x.data_ptr()),
        reinterpret_cast<const half*>(z.data_ptr()), weight.data_ptr<float>(),
        z.stride(0), total_heads, static_cast<float>(eps));
  } else {
    gdn_rmsnorm_gated_h12d128_kernel<half><<<ctas, 256, 0, stream>>>(
        reinterpret_cast<half*>(out.data_ptr()),
        reinterpret_cast<const half*>(x.data_ptr()),
        reinterpret_cast<const half*>(z.data_ptr()),
        reinterpret_cast<const half*>(weight.data_ptr()), z.stride(0),
        total_heads, static_cast<float>(eps));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor wrap_ipc_half_2d(int64_t ptr, int64_t rows, int64_t cols,
                               int64_t device_idx) {
  auto opts = torch::TensorOptions()
                  .dtype(torch::kFloat16)
                  .device(torch::kCUDA, static_cast<int>(device_idx));
  return torch::from_blob(reinterpret_cast<void*>(static_cast<uintptr_t>(ptr)),
                          {rows, cols}, opts);
}

}  // namespace hc_p2p

// ============================================================================
// Part 8: Zero-Spill Native SM70 80-CTA/96-CTA GDN Chunk Forward (Volta HMMA)
// ============================================================================

namespace gdn_sm70 {

constexpr int kGdnHg = 4;
constexpr int kGdnH = 12;
constexpr int kGdnK = 128;
constexpr int kGdnV = 128;
constexpr int kGdnS = 64;
constexpr float kLog2E = 1.4426950408889634f;

// Chunk-parallel (grid = (num_chunks, 12)) prep kernel:
// 1. Scales A[s, t] in-place by exp2((g[s] - g[t]) * log2(e)) * b[t] (s >= t)
// 2. Computes P[s, t] = fp16(scale * exp2((g[s] - g[t]) * log2(e)) * (Q @ K^T))
//    ONCE per (chunk, head) across all 80 SMs instead of 4x-8x per V-slice!
template <typename BetaT, typename IdxT>
__global__ __launch_bounds__(256, 2) void fn_gdn_prep_ap_kernel(
    const half* __restrict__ q,   // [1, T, 4, 128]
    const half* __restrict__ k,   // [1, T, 4, 128]
    half* __restrict__ a,         // [1, T, 12, 64] (in-place -> A_bar)
    half* __restrict__ p_out,     // [1, T, 12, 64]
    const float* __restrict__ g,  // [1, T, 12] (already chunk-cumsum'd)
    const BetaT* __restrict__ b,  // [1, T, 12]
    const int32_t* __restrict__ cu_seqlens,
    const IdxT* __restrict__ chunk_indices,  // [num_chunks, 2]
    float scale) {
  const int bc = blockIdx.x;
  const int bh = blockIdx.y;  // 0..11
  const int bhg = bh / (kGdnH / kGdnHg);
  const int batch_idx = static_cast<int>(chunk_indices[bc * 2 + 0]);
  const int chunk_idx = static_cast<int>(chunk_indices[bc * 2 + 1]);
  const int seq_start = cu_seqlens[batch_idx];
  const int seq_end = cu_seqlens[batch_idx + 1];
  const int left = seq_start + chunk_idx * kGdnS;
  const int valid_len = min(kGdnS, max(0, seq_end - left));
  if (valid_len <= 0) return;

  union PrepSmem {
    struct {
      half s_q[kGdnS][136];
      half s_k[kGdnS][136];
    } qk;
    half s_p[kGdnS][72];
  };
  __shared__ __align__(16) PrepSmem smem;
  __shared__ float s_g[kGdnS];
  __shared__ float s_b[kGdnS];

  const int tx = threadIdx.x;  // 0..255
  const int lane = tx & 31;
  const int warp = tx >> 5;  // 0..7

  if (tx < kGdnS) {
    const int tok_g = left + min(tx, valid_len - 1);
    s_g[tx] = __ldg(g + static_cast<int64_t>(tok_g) * kGdnH + bh);
    if (tx < valid_len) {
      if constexpr (std::is_same_v<BetaT, float>) {
        s_b[tx] = __ldg(b + static_cast<int64_t>(left + tx) * kGdnH + bh);
      } else {
        s_b[tx] = __half2float(
            __ldg(b + static_cast<int64_t>(left + tx) * kGdnH + bh));
      }
    } else {
      s_b[tx] = 0.0f;
    }
  }

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int idx = tx + i * 256;  // 0..1023
    const int r = idx >> 4;        // 0..63
    const int c8 = (idx & 15) * 8;
    uint4 qv = make_uint4(0, 0, 0, 0);
    uint4 kv = make_uint4(0, 0, 0, 0);
    if (r < valid_len) {
      const int64_t base =
          (static_cast<int64_t>(left + r) * kGdnHg + bhg) * kGdnK + c8;
      qv = __ldg(reinterpret_cast<const uint4*>(q + base));
      kv = __ldg(reinterpret_cast<const uint4*>(k + base));
    }
    *reinterpret_cast<uint4*>(&smem.qk.s_q[r][c8]) = qv;
    *reinterpret_cast<uint4*>(&smem.qk.s_k[r][c8]) = kv;
  }
  __syncthreads();

  // In-place scale A -> A_bar using 100% coalesced 128-bit loads/stores
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int idx = tx + i * 256;  // 0..511
    const int r = idx >> 3;        // 0..63
    const int c0 = (idx & 7) * 8;
    if (r < valid_len) {
      half* a_ptr =
          a + (static_cast<int64_t>(left + r) * kGdnH + bh) * kGdnS + c0;
      uint4 av = __ldg(reinterpret_cast<const uint4*>(a_ptr));
      half* ah = reinterpret_cast<half*>(&av);
      const float gr = s_g[r];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int c = c0 + j;
        const float gf = (r >= c) ? exp2f((gr - s_g[c]) * kLog2E) : 0.0f;
        ah[j] = __float2half_rn(__half2float(ah[j]) * gf * s_b[c]);
      }
      *reinterpret_cast<uint4*>(a_ptr) = av;
    }
  }

  // Q @ K^T ([64, 128] @ [128, 64]) across 8 warps: each warp owns 8 rows!
  const int mma_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int quad = (lane >> 2) & 3;
  const int rq = warp * 8 + mma_row;
  const int ck0 = quad * 8 + mma_row;
  const int ck1 = 32 + ck0;
  float p0[8] = {}, p1[8] = {};

#pragma unroll
  for (int k_off = 0; k_off < kGdnK; k_off += 8) {
    const uint4 qa = *reinterpret_cast<const uint4*>(&smem.qk.s_q[rq][k_off]);
    const uint4 kb0 = *reinterpret_cast<const uint4*>(&smem.qk.s_k[ck0][k_off]);
    const uint4 kb1 = *reinterpret_cast<const uint4*>(&smem.qk.s_k[ck1][k_off]);
    FN_MMA_884_RC(p0, qa.x, qa.y, kb0.x, kb0.y);
    FN_MMA_884_RC(p0, qa.z, qa.w, kb0.z, kb0.w);
    FN_MMA_884_RC(p1, qa.x, qa.y, kb1.x, kb1.y);
    FN_MMA_884_RC(p1, qa.z, qa.w, kb1.z, kb1.w);
  }
  __syncthreads();

  const int r_off = ((lane & 16) ? 4 : 0) | (lane & 1);
  const int r0 = warp * 8 + r_off;
  const int r2 = r0 + 2;
  const int c_sub = quad * 8 + (((lane >> 1) & 1) << 1);
  const float gr0 = s_g[r0];
  const float gr2 = s_g[r2];

  auto emit_p_pair = [&](const float (&acc)[8], int c_half) {
    const int cb = c_half + c_sub;
    const float gc0 = s_g[cb];
    const float gc1 = s_g[cb + 1];
    const float gc4 = s_g[cb + 4];
    const float gc5 = s_g[cb + 5];
    *reinterpret_cast<half2*>(&smem.s_p[r0][cb]) = __floats2half2_rn(
        (r0 >= cb) ? acc[0] * (scale * exp2f((gr0 - gc0) * kLog2E)) : 0.0f,
        (r0 >= cb + 1) ? acc[1] * (scale * exp2f((gr0 - gc1) * kLog2E)) : 0.0f);
    *reinterpret_cast<half2*>(&smem.s_p[r2][cb]) = __floats2half2_rn(
        (r2 >= cb) ? acc[2] * (scale * exp2f((gr2 - gc0) * kLog2E)) : 0.0f,
        (r2 >= cb + 1) ? acc[3] * (scale * exp2f((gr2 - gc1) * kLog2E)) : 0.0f);
    *reinterpret_cast<half2*>(&smem.s_p[r0][cb + 4]) = __floats2half2_rn(
        (r0 >= cb + 4) ? acc[4] * (scale * exp2f((gr0 - gc4) * kLog2E)) : 0.0f,
        (r0 >= cb + 5) ? acc[5] * (scale * exp2f((gr0 - gc5) * kLog2E)) : 0.0f);
    *reinterpret_cast<half2*>(&smem.s_p[r2][cb + 4]) = __floats2half2_rn(
        (r2 >= cb + 4) ? acc[6] * (scale * exp2f((gr2 - gc4) * kLog2E)) : 0.0f,
        (r2 >= cb + 5) ? acc[7] * (scale * exp2f((gr2 - gc5) * kLog2E)) : 0.0f);
  };
  emit_p_pair(p0, 0);
  emit_p_pair(p1, 32);
  __syncthreads();

#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int idx = tx + i * 256;
    const int r = idx >> 3;
    const int c0 = (idx & 7) * 8;
    if (r < valid_len) {
      *reinterpret_cast<uint4*>(
          p_out + (static_cast<int64_t>(left + r) * kGdnH + bh) * kGdnS + c0) =
          *reinterpret_cast<const uint4*>(&smem.s_p[r][c0]);
    }
  }
}

// Single-launch Fused (chunk_local_cumsum + HMMA K@K^T + FP32 64x64 Block-16/32
// Unit Lower-Triangular Inverse + A_bar + HMMA Q@K^T P) across all 80 V100 SMs!
// Eliminates chunk_local_cumsum + tilelang_kkt_solve_kernel + 5.4 GB
// intermediate a_mat HBM read/write + duplicate K HBM read! 35,328 B static
// smem <= 48 KiB -> 2 CTAs/SM (16 warps/SM), 0 spills!
template <typename BetaT, typename IdxT>
__global__ __launch_bounds__(256, 2) void fn_gdn_cumsum_kkt_prep_ap_kernel(
    const half* __restrict__ q,      // [1, T, 4, 128]
    const half* __restrict__ k,      // [1, T, 4, 128]
    half* __restrict__ a_bar_out,    // [1, T, 12, 64]
    half* __restrict__ p_out,        // [1, T, 12, 64]
    float* __restrict__ g_out,       // [1, T, 12]
    const float* __restrict__ g_in,  // [1, T, 12] (raw un-cumsum'd g)
    const BetaT* __restrict__ b,     // [1, T, 12]
    const int32_t* __restrict__ cu_seqlens,
    const IdxT* __restrict__ chunk_indices,  // [num_chunks, 2]
    float scale) {
  const int bc = blockIdx.x;
  const int bh = blockIdx.y;  // 0..11
  const int bhg = bh / (kGdnH / kGdnHg);
  const int batch_idx = static_cast<int>(chunk_indices[bc * 2 + 0]);
  const int chunk_idx = static_cast<int>(chunk_indices[bc * 2 + 1]);
  const int seq_start = cu_seqlens[batch_idx];
  const int seq_end = cu_seqlens[batch_idx + 1];
  const int left = seq_start + chunk_idx * kGdnS;
  const int valid_len = min(kGdnS, max(0, seq_end - left));
  if (valid_len <= 0) return;

  union FusedPrepSmem {
    struct {
      half s_q[kGdnS][136];  // 17,408 B
      half s_k[kGdnS][136];  // 17,408 B -> 34,816 B
    } qk;
    struct {
      float s_A[kGdnS][65];  // 16,640 B (odd stride 65 = 0 bank conflicts!)
      half s_p[kGdnS][72];   //  9,216 B -> 25,856 B
    } inv;
  };
  __shared__ __align__(16) FusedPrepSmem smem;
  __shared__ float s_g[kGdnS];
  __shared__ float s_b[kGdnS];

  const int tx = threadIdx.x;  // 0..255
  const int lane = tx & 31;
  const int warp = tx >> 5;  // 0..7

  // 1. Warp 0 computes inclusive chunk_local_cumsum(g) + stores g_out & s_g;
  //    Warp 1 loads beta into s_b; all 8 warps load Q and K ONCE into smem!
  if (warp == 0) {
    const int r0 = lane;
    const int r1 = lane + 32;
    float gv0 = (r0 < valid_len)
                    ? __ldg(g_in + static_cast<int64_t>(left + r0) * kGdnH + bh)
                    : 0.0f;
    float gv1 = (r1 < valid_len)
                    ? __ldg(g_in + static_cast<int64_t>(left + r1) * kGdnH + bh)
                    : 0.0f;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const float n0 = __shfl_up_sync(0xffffffffu, gv0, d);
      const float n1 = __shfl_up_sync(0xffffffffu, gv1, d);
      if (lane >= d) {
        gv0 += n0;
        gv1 += n1;
      }
    }
    const float sum0 = __shfl_sync(0xffffffffu, gv0, 31);
    gv1 += sum0;
    s_g[r0] = gv0;
    s_g[r1] = gv1;
    if (r0 < valid_len) {
      g_out[static_cast<int64_t>(left + r0) * kGdnH + bh] = gv0;
    }
    if (r1 < valid_len) {
      g_out[static_cast<int64_t>(left + r1) * kGdnH + bh] = gv1;
    }
  } else if (warp == 1) {
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int r = lane + u * 32;
      float bv = 0.0f;
      if (r < valid_len) {
        if constexpr (std::is_same_v<BetaT, float>) {
          bv = __ldg(b + static_cast<int64_t>(left + r) * kGdnH + bh);
        } else {
          bv = __half2float(
              __ldg(b + static_cast<int64_t>(left + r) * kGdnH + bh));
        }
      }
      s_b[r] = bv;
    }
  }

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int idx = tx + i * 256;
    const int r = idx >> 4;
    const int c8 = (idx & 15) * 8;
    uint4 qv = make_uint4(0, 0, 0, 0);
    uint4 kv = make_uint4(0, 0, 0, 0);
    if (r < valid_len) {
      const int64_t base =
          (static_cast<int64_t>(left + r) * kGdnHg + bhg) * kGdnK + c8;
      qv = __ldg(reinterpret_cast<const uint4*>(q + base));
      kv = __ldg(reinterpret_cast<const uint4*>(k + base));
    }
    *reinterpret_cast<uint4*>(&smem.qk.s_q[r][c8]) = qv;
    *reinterpret_cast<uint4*>(&smem.qk.s_k[r][c8]) = kv;
  }
  __syncthreads();

  // 2. Simultaneously compute Q @ K^T AND K @ K^T on Volta Tensor Cores while
  //    s_q and s_k are resident in shared memory (reusing kb0, kb1 loads)!
  const int mma_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int quad = (lane >> 2) & 3;
  const int rq = warp * 8 + mma_row;
  const int ck0 = quad * 8 + mma_row;
  const int ck1 = 32 + ck0;
  float p0[8] = {}, p1[8] = {};
  float k0[8] = {}, k1[8] = {};

#pragma unroll
  for (int k_off = 0; k_off < kGdnK; k_off += 8) {
    const uint4 qa = *reinterpret_cast<const uint4*>(&smem.qk.s_q[rq][k_off]);
    const uint4 ka = *reinterpret_cast<const uint4*>(&smem.qk.s_k[rq][k_off]);
    const uint4 kb0 = *reinterpret_cast<const uint4*>(&smem.qk.s_k[ck0][k_off]);
    const uint4 kb1 = *reinterpret_cast<const uint4*>(&smem.qk.s_k[ck1][k_off]);
    FN_MMA_884_RC(p0, qa.x, qa.y, kb0.x, kb0.y);
    FN_MMA_884_RC(p0, qa.z, qa.w, kb0.z, kb0.w);
    FN_MMA_884_RC(p1, qa.x, qa.y, kb1.x, kb1.y);
    FN_MMA_884_RC(p1, qa.z, qa.w, kb1.z, kb1.w);
    FN_MMA_884_RC(k0, ka.x, ka.y, kb0.x, kb0.y);
    FN_MMA_884_RC(k0, ka.z, ka.w, kb0.z, kb0.w);
    FN_MMA_884_RC(k1, ka.x, ka.y, kb1.x, kb1.y);
    FN_MMA_884_RC(k1, ka.z, ka.w, kb1.z, kb1.w);
  }
  __syncthreads();  // Done with s_q and s_k!

  // 3. Stage BOTH P -> smem.inv.s_p AND initial (I - strictLower(b * K@K^T))
  //    -> smem.inv.s_A BEFORE the next barrier!
  const int r_off = ((lane & 16) ? 4 : 0) | (lane & 1);
  const int r0 = warp * 8 + r_off;
  const int r2 = r0 + 2;
  const int c_sub = quad * 8 + (((lane >> 1) & 1) << 1);
  const float gr0 = s_g[r0];
  const float gr2 = s_g[r2];
  const float br0 = s_b[r0];
  const float br2 = s_b[r2];

  auto stage_pk_half = [&](const float (&pa)[8], const float (&ka)[8],
                           int c_half) {
    const int cb = c_half + c_sub;
    const float gc0 = s_g[cb];
    const float gc1 = s_g[cb + 1];
    const float gc4 = s_g[cb + 4];
    const float gc5 = s_g[cb + 5];
    *reinterpret_cast<half2*>(&smem.inv.s_p[r0][cb]) = __floats2half2_rn(
        (r0 >= cb) ? pa[0] * (scale * exp2f((gr0 - gc0) * kLog2E)) : 0.0f,
        (r0 >= cb + 1) ? pa[1] * (scale * exp2f((gr0 - gc1) * kLog2E)) : 0.0f);
    *reinterpret_cast<half2*>(&smem.inv.s_p[r2][cb]) = __floats2half2_rn(
        (r2 >= cb) ? pa[2] * (scale * exp2f((gr2 - gc0) * kLog2E)) : 0.0f,
        (r2 >= cb + 1) ? pa[3] * (scale * exp2f((gr2 - gc1) * kLog2E)) : 0.0f);
    *reinterpret_cast<half2*>(&smem.inv.s_p[r0][cb + 4]) = __floats2half2_rn(
        (r0 >= cb + 4) ? pa[4] * (scale * exp2f((gr0 - gc4) * kLog2E)) : 0.0f,
        (r0 >= cb + 5) ? pa[5] * (scale * exp2f((gr0 - gc5) * kLog2E)) : 0.0f);
    *reinterpret_cast<half2*>(&smem.inv.s_p[r2][cb + 4]) = __floats2half2_rn(
        (r2 >= cb + 4) ? pa[6] * (scale * exp2f((gr2 - gc4) * kLog2E)) : 0.0f,
        (r2 >= cb + 5) ? pa[7] * (scale * exp2f((gr2 - gc5) * kLog2E)) : 0.0f);

    auto init_a = [&](int r, int c, float br, float kv) -> float {
      if (r == c) return 1.0f;
      if (r > c && r < valid_len && c < valid_len) return -br * kv;
      return 0.0f;
    };
    smem.inv.s_A[r0][cb + 0] = init_a(r0, cb + 0, br0, ka[0]);
    smem.inv.s_A[r0][cb + 1] = init_a(r0, cb + 1, br0, ka[1]);
    smem.inv.s_A[r2][cb + 0] = init_a(r2, cb + 0, br2, ka[2]);
    smem.inv.s_A[r2][cb + 1] = init_a(r2, cb + 1, br2, ka[3]);
    smem.inv.s_A[r0][cb + 4] = init_a(r0, cb + 4, br0, ka[4]);
    smem.inv.s_A[r0][cb + 5] = init_a(r0, cb + 5, br0, ka[5]);
    smem.inv.s_A[r2][cb + 4] = init_a(r2, cb + 4, br2, ka[6]);
    smem.inv.s_A[r2][cb + 5] = init_a(r2, cb + 5, br2, ka[7]);
  };
  stage_pk_half(p0, k0, 0);
  stage_pk_half(p1, k1, 32);
  __syncthreads();

  // Flush P to global memory using coalesced 128-bit stores
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int idx = tx + i * 256;
    const int r = idx >> 3;
    const int c0 = (idx & 7) * 8;
    if (r < valid_len) {
      *reinterpret_cast<uint4*>(
          p_out + (static_cast<int64_t>(left + r) * kGdnH + bh) * kGdnS + c0) =
          *reinterpret_cast<const uint4*>(&smem.inv.s_p[r][c0]);
    }
  }

  // 4. Exact FP32 64x64 Block-16/32 Unit Lower-Triangular Inverse in
  // smem.inv.s_A: Step 4a: Warps 0..3 invert the four 16x16 diagonal blocks in
  // parallel!
  if (warp < 4) {
    const int base = warp * 16;
    for (int i = 1; i < 16; ++i) {
      const int r = base + i;
      const float a_rc = (lane < i) ? smem.inv.s_A[r][base + lane] : 0.0f;
      float sum = 0.0f;
      for (int m = 1; m < i; ++m) {
        const float neg_l_rm = __shfl_sync(0xffffffffu, a_rc, m);
        if (lane < m) {
          sum = __fmaf_rn(neg_l_rm, smem.inv.s_A[base + m][base + lane], sum);
        }
      }
      if (lane < i) {
        smem.inv.s_A[r][base + lane] = a_rc + sum;
      }
      __syncwarp(0xffffffffu);
    }
  }
  __syncthreads();

  // Step 4b: Off-diagonal 16x16 blocks A_10 = A_11 * (-L_10) * A_00 and
  //          A_32 = A_33 * (-L_32) * A_22 across all 256 threads!
  {
    const int lr = tx >> 4;  // 0..15
    const int lc = tx & 15;  // 0..15
    float t10 = 0.0f, t32 = 0.0f;
#pragma unroll
    for (int m = 0; m < 16; ++m) {
      t10 = __fmaf_rn(smem.inv.s_A[16 + lr][m], smem.inv.s_A[m][lc], t10);
      t32 = __fmaf_rn(smem.inv.s_A[48 + lr][32 + m],
                      smem.inv.s_A[32 + m][32 + lc], t32);
    }
    __syncthreads();
    smem.inv.s_A[16 + lr][lc] = t10;
    smem.inv.s_A[48 + lr][32 + lc] = t32;
    __syncthreads();

    float a10 = 0.0f, a32 = 0.0f;
#pragma unroll
    for (int m = 0; m < 16; ++m) {
      a10 = __fmaf_rn(smem.inv.s_A[16 + lr][16 + m], smem.inv.s_A[16 + m][lc],
                      a10);
      a32 = __fmaf_rn(smem.inv.s_A[48 + lr][48 + m],
                      smem.inv.s_A[48 + m][32 + lc], a32);
    }
    __syncthreads();
    smem.inv.s_A[16 + lr][lc] = a10;
    smem.inv.s_A[48 + lr][32 + lc] = a32;
    __syncthreads();
  }

  // Step 4c: Bottom-left 32x32 quadrant M_10 = M_1 * (-L_bottom_left) * M_0!
  {
    const int lr = tx >> 3;           // 0..31
    const int c_base = (tx & 7) * 4;  // 0, 4, ..., 28
    float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
#pragma unroll
    for (int m = 0; m < 32; ++m) {
      const float lhs = smem.inv.s_A[32 + lr][m];
      t0 = __fmaf_rn(lhs, smem.inv.s_A[m][c_base + 0], t0);
      t1 = __fmaf_rn(lhs, smem.inv.s_A[m][c_base + 1], t1);
      t2 = __fmaf_rn(lhs, smem.inv.s_A[m][c_base + 2], t2);
      t3 = __fmaf_rn(lhs, smem.inv.s_A[m][c_base + 3], t3);
    }
    __syncthreads();
    smem.inv.s_A[32 + lr][c_base + 0] = t0;
    smem.inv.s_A[32 + lr][c_base + 1] = t1;
    smem.inv.s_A[32 + lr][c_base + 2] = t2;
    smem.inv.s_A[32 + lr][c_base + 3] = t3;
    __syncthreads();

    float m0 = 0.0f, m1 = 0.0f, m2 = 0.0f, m3 = 0.0f;
#pragma unroll
    for (int m = 0; m < 32; ++m) {
      const float lhs = smem.inv.s_A[32 + lr][32 + m];
      m0 = __fmaf_rn(lhs, smem.inv.s_A[32 + m][c_base + 0], m0);
      m1 = __fmaf_rn(lhs, smem.inv.s_A[32 + m][c_base + 1], m1);
      m2 = __fmaf_rn(lhs, smem.inv.s_A[32 + m][c_base + 2], m2);
      m3 = __fmaf_rn(lhs, smem.inv.s_A[32 + m][c_base + 3], m3);
    }
    __syncthreads();
    smem.inv.s_A[32 + lr][c_base + 0] = m0;
    smem.inv.s_A[32 + lr][c_base + 1] = m1;
    smem.inv.s_A[32 + lr][c_base + 2] = m2;
    smem.inv.s_A[32 + lr][c_base + 3] = m3;
    __syncthreads();
  }

  // 5. Scale A -> A_bar and write coalesced 128-bit uint4 directly to HBM!
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int idx = tx + i * 256;
    const int r = idx >> 3;
    const int c0 = (idx & 7) * 8;
    if (r < valid_len) {
      const float gr = s_g[r];
      uint4 av;
      half* ah = reinterpret_cast<half*>(&av);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int c = c0 + j;
        const float gf =
            (r >= c && c < valid_len) ? exp2f((gr - s_g[c]) * kLog2E) : 0.0f;
        ah[j] = __float2half_rn(smem.inv.s_A[r][c] * gf * s_b[c]);
      }
      *reinterpret_cast<uint4*>(
          a_bar_out + (static_cast<int64_t>(left + r) * kGdnH + bh) * kGdnS +
          c0) = av;
    }
  }
}

struct GdnRecurrentSmem {
  half s_k[kGdnS][136];  // 17,408 B
  half s_q[kGdnS][136];  // 17,408 B
  half s_ap[kGdnS][72];  //  9,216 B (holds A_bar, then P)
  half s_h[kGdnK][36];   //  9,216 B
  half s_v[kGdnS][36];   //  4,608 B (holds V -> u -> vn)
  half s_vd[kGdnS][36];  //  4,608 B
  float s_gexp[kGdnS];   //    256 B
  float s_grev[kGdnS];   //    256 B -> Total = 62,976 B (<= 64 KB on 80 SMs!)
};

template <bool MODE_96CTA>
__global__ __launch_bounds__(256, 1) void fn_gdn_recurrent_kernel(
    const half* __restrict__ q,      // [1, T, 4, 128]
    const half* __restrict__ k,      // [1, T, 4, 128]
    const half* __restrict__ v,      // [1, T, 12, 128]
    const half* __restrict__ a_bar,  // [1, T, 12, 64]
    const half* __restrict__ p_mat,  // [1, T, 12, 64]
    const float* __restrict__ g,     // [1, T, 12]
    float* __restrict__ h0_state,    // [slots, 12, 128, 128]
    const int32_t* __restrict__ cu_seqlens,
    const int32_t* __restrict__ state_indices,
    const bool* __restrict__ has_initial_state,
    half* __restrict__ o_out,    // [1, T, 12, 128]
    float* __restrict__ ht_out,  // [B, 12, 128, 128] or nullptr
    float scale, bool use_initial_state, bool state_layout_vlk,
    bool inplace_final_state, bool store_final_state) {
  extern __shared__ __align__(16) char raw_smem[];
  GdnRecurrentSmem& smem = *reinterpret_cast<GdnRecurrentSmem*>(raw_smem);

  const int cta_id = blockIdx.x;
  const int bb = blockIdx.y;
  int bh = 0, col_start = 0, bv = 16;
  if constexpr (MODE_96CTA) {
    bh = cta_id >> 3;  // 0..11
    col_start = (cta_id & 7) * 16;
    bv = 16;
  } else {
    // Exact 80-CTA decomposition across all 80 V100 SMs (zero idle SMs!):
    // Heads 0..7: 7 CTAs each (24, 24, 16, 16, 16, 16, 16) = 56 CTAs
    // Heads 8..11: 6 CTAs each (24, 24, 24, 24, 16, 16)    = 24 CTAs
    if (cta_id < 56) {
      bh = cta_id / 7;
      const int s = cta_id % 7;
      if (s < 2) {
        col_start = s * 24;
        bv = 24;
      } else {
        col_start = 48 + (s - 2) * 16;
        bv = 16;
      }
    } else {
      const int rem = cta_id - 56;
      bh = 8 + rem / 6;
      const int s = rem % 6;
      if (s < 4) {
        col_start = s * 24;
        bv = 24;
      } else {
        col_start = 96 + (s - 4) * 16;
        bv = 16;
      }
    }
  }
  const int bhg = bh / (kGdnH / kGdnHg);
  const int seq_start = cu_seqlens[bb];
  const int seq_end = cu_seqlens[bb + 1];
  const int seq_len = seq_end - seq_start;
  const int num_iters = (seq_len + kGdnS - 1) / kGdnS;
  const int state_slot = (state_indices != nullptr) ? state_indices[bb] : bb;
  const bool init_valid = use_initial_state && (has_initial_state == nullptr ||
                                                has_initial_state[bb]);

  const int tx = threadIdx.x;
  const int lane = tx & 31;
  const int warp = tx >> 5;  // 0..7
  const int quad = (lane >> 2) & 3;
  const int mma_row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int r_off = ((lane & 16) ? 4 : 0) | (lane & 1);
  const int c_b = quad * 8 + (((lane >> 1) & 1) << 1);
  const int vn_col = quad * 8 + ((lane & 16) ? 4 : 0);
  const int vk = lane & 3;

  float h0[8] = {}, h1[8] = {};
  if (init_valid) {
    const float* h_src =
        h0_state +
        (static_cast<int64_t>(state_slot) * kGdnH + bh) * (kGdnK * kGdnV);
    auto load_h_tile = [&](float (&acc)[8], int kb) {
      const int kr0 = kb + r_off;
      const int kr2 = kr0 + 2;
      auto rd = [&](int kr, int vc) -> float {
        if (vc >= bv) return 0.0f;
        const int gc = col_start + vc;
        return state_layout_vlk ? __ldg(h_src + gc * kGdnK + kr)
                                : __ldg(h_src + kr * kGdnV + gc);
      };
      acc[0] = rd(kr0, c_b);
      acc[1] = rd(kr0, c_b + 1);
      acc[2] = rd(kr2, c_b);
      acc[3] = rd(kr2, c_b + 1);
      acc[4] = rd(kr0, c_b + 4);
      acc[5] = rd(kr0, c_b + 5);
      acc[6] = rd(kr2, c_b + 4);
      acc[7] = rd(kr2, c_b + 5);
    };
    load_h_tile(h0, warp * 16);
    load_h_tile(h1, warp * 16 + 8);
  }

  if (num_iters <= 0) return;

  // Software-pipelined chunk prefetch registers (~46 regs out of 127 free regs
  // at 1 CTA/SM -> <= 185 regs <= 255 max, ZERO spills, 0.0 bit-exact diff!):
  // Overlaps chunk (i_s + 1)'s global K, Q, A_bar, V, g loads with Step 4's
  // 80 Tensor Core HMMA instructions!
  uint4 kv_pre[4], qv_pre[4], av_pre[2];
  uint4 vv_pre = make_uint4(0, 0, 0, 0);
  float g0_pre = 0.0f, g1_pre = 0.0f;

  auto prefetch_chunk_inputs = [&](int chunk_left, int chunk_valid) {
    if (warp == 0) {
      const int r0 = lane;
      const int r1 = lane + 32;
      g0_pre = __ldg(
          g +
          static_cast<int64_t>(chunk_left + min(r0, chunk_valid - 1)) * kGdnH +
          bh);
      g1_pre = __ldg(
          g +
          static_cast<int64_t>(chunk_left + min(r1, chunk_valid - 1)) * kGdnH +
          bh);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int idx = tx + i * 256;
      const int r = idx >> 4;
      const int c8 = (idx & 15) * 8;
      kv_pre[i] = make_uint4(0, 0, 0, 0);
      qv_pre[i] = make_uint4(0, 0, 0, 0);
      if (r < chunk_valid) {
        const int64_t base =
            (static_cast<int64_t>(chunk_left + r) * kGdnHg + bhg) * kGdnK + c8;
        kv_pre[i] = __ldg(reinterpret_cast<const uint4*>(k + base));
        qv_pre[i] = __ldg(reinterpret_cast<const uint4*>(q + base));
      }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int idx = tx + i * 256;
      const int r = idx >> 3;
      const int c8 = (idx & 7) * 8;
      av_pre[i] = make_uint4(0, 0, 0, 0);
      if (r < chunk_valid) {
        av_pre[i] = __ldg(reinterpret_cast<const uint4*>(
            a_bar +
            (static_cast<int64_t>(chunk_left + r) * kGdnH + bh) * kGdnS + c8));
      }
    }
    {
      const int r = tx >> 2;
      const int c8 = (tx & 3) * 8;
      vv_pre = make_uint4(0, 0, 0, 0);
      if (r < chunk_valid && c8 < bv) {
        vv_pre = __ldg(reinterpret_cast<const uint4*>(
            v + (static_cast<int64_t>(chunk_left + r) * kGdnH + bh) * kGdnV +
            col_start + c8));
      }
    }
  };

  prefetch_chunk_inputs(seq_start, min(kGdnS, seq_end - seq_start));

  for (int i_s = 0; i_s < num_iters; ++i_s) {
    const int left = seq_start + i_s * kGdnS;
    const int valid_len = min(kGdnS, seq_end - left);

    // 1. Stage prefetched K, Q, A_bar, V, g + H_old into smem in ONE barrier!
    if (warp == 0) {
      const int r0 = lane;
      const int r1 = lane + 32;
      const float g_last = __shfl_sync(0xffffffffu, g1_pre, 31);
      smem.s_gexp[r0] = exp2f(g0_pre * kLog2E);
      smem.s_gexp[r1] = exp2f(g1_pre * kLog2E);
      smem.s_grev[r0] =
          (r0 < valid_len) ? exp2f((g_last - g0_pre) * kLog2E) : 0.0f;
      smem.s_grev[r1] =
          (r1 < valid_len) ? exp2f((g_last - g1_pre) * kLog2E) : 0.0f;
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int idx = tx + i * 256;
      const int r = idx >> 4;
      const int c8 = (idx & 15) * 8;
      *reinterpret_cast<uint4*>(&smem.s_k[r][c8]) = kv_pre[i];
      *reinterpret_cast<uint4*>(&smem.s_q[r][c8]) = qv_pre[i];
    }

#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int idx = tx + i * 256;
      const int r = idx >> 3;
      const int c8 = (idx & 7) * 8;
      *reinterpret_cast<uint4*>(&smem.s_ap[r][c8]) = av_pre[i];
    }

    {
      const int r = tx >> 2;
      const int c8 = (tx & 3) * 8;
      const uint2* vu2 = reinterpret_cast<const uint2*>(&vv_pre);
      *reinterpret_cast<uint2*>(&smem.s_v[r][c8]) = vu2[0];
      *reinterpret_cast<uint2*>(&smem.s_v[r][c8 + 4]) = vu2[1];
    }

    // Stage H_old ([128, bv]) into s_h
    auto stage_h = [&](const float (&acc)[8], int kb) {
      const int kr0 = kb + r_off;
      const int kr2 = kr0 + 2;
      *reinterpret_cast<half2*>(&smem.s_h[kr0][c_b]) =
          __floats2half2_rn(acc[0], acc[1]);
      *reinterpret_cast<half2*>(&smem.s_h[kr2][c_b]) =
          __floats2half2_rn(acc[2], acc[3]);
      *reinterpret_cast<half2*>(&smem.s_h[kr0][c_b + 4]) =
          __floats2half2_rn(acc[4], acc[5]);
      *reinterpret_cast<half2*>(&smem.s_h[kr2][c_b + 4]) =
          __floats2half2_rn(acc[6], acc[7]);
    };
    stage_h(h0, warp * 16);
    stage_h(h1, warp * 16 + 8);
    __syncthreads();  // Barrier #1

    // 2. u = V - g_exp * (K @ H_old), while prefetching P in registers!
    uint4 p_pre0 = make_uint4(0, 0, 0, 0);
    uint4 p_pre1 = make_uint4(0, 0, 0, 0);
    {
      const int idx0 = tx;
      const int idx1 = tx + 256;
      const int pr0 = idx0 >> 3, pc0 = (idx0 & 7) * 8;
      const int pr1 = idx1 >> 3, pc1 = (idx1 & 7) * 8;
      if (pr0 < valid_len) {
        p_pre0 = __ldg(reinterpret_cast<const uint4*>(
            p_mat + (static_cast<int64_t>(left + pr0) * kGdnH + bh) * kGdnS +
            pc0));
      }
      if (pr1 < valid_len) {
        p_pre1 = __ldg(reinterpret_cast<const uint4*>(
            p_mat + (static_cast<int64_t>(left + pr1) * kGdnH + bh) * kGdnS +
            pc1));
      }
    }

    const int sb = warp * 8;
    const int sr0 = sb + r_off;
    const int sr2 = sr0 + 2;
    float u_acc[8] = {};
#pragma unroll
    for (int k_off = 0; k_off < kGdnK; k_off += 8) {
      const uint4 ka =
          *reinterpret_cast<const uint4*>(&smem.s_k[sb + mma_row][k_off]);
      const uint2 hb0 =
          *reinterpret_cast<const uint2*>(&smem.s_h[k_off + vk][vn_col]);
      const uint2 hb1 =
          *reinterpret_cast<const uint2*>(&smem.s_h[k_off + 4 + vk][vn_col]);
      FN_MMA_884_RR(u_acc, ka.x, ka.y, hb0.x, hb0.y);
      FN_MMA_884_RR(u_acc, ka.z, ka.w, hb1.x, hb1.y);
    }
    {
      const float ge0 = smem.s_gexp[sr0];
      const float ge2 = smem.s_gexp[sr2];
      const half2 v01 = *reinterpret_cast<const half2*>(&smem.s_v[sr0][c_b]);
      const half2 v23 = *reinterpret_cast<const half2*>(&smem.s_v[sr2][c_b]);
      const half2 v45 =
          *reinterpret_cast<const half2*>(&smem.s_v[sr0][c_b + 4]);
      const half2 v67 =
          *reinterpret_cast<const half2*>(&smem.s_v[sr2][c_b + 4]);
      *reinterpret_cast<half2*>(&smem.s_v[sr0][c_b]) =
          __floats2half2_rn(__half2float(v01.x) - ge0 * u_acc[0],
                            __half2float(v01.y) - ge0 * u_acc[1]);
      *reinterpret_cast<half2*>(&smem.s_v[sr2][c_b]) =
          __floats2half2_rn(__half2float(v23.x) - ge2 * u_acc[2],
                            __half2float(v23.y) - ge2 * u_acc[3]);
      *reinterpret_cast<half2*>(&smem.s_v[sr0][c_b + 4]) =
          __floats2half2_rn(__half2float(v45.x) - ge0 * u_acc[4],
                            __half2float(v45.y) - ge0 * u_acc[5]);
      *reinterpret_cast<half2*>(&smem.s_v[sr2][c_b + 4]) =
          __floats2half2_rn(__half2float(v67.x) - ge2 * u_acc[6],
                            __half2float(v67.y) - ge2 * u_acc[7]);
    }
    __syncthreads();  // Barrier #2

    // 3. vd = A_bar @ u, then store vd -> s_vd, vn -> s_v, and P -> s_ap!
    float vd_acc[8] = {};
#pragma unroll
    for (int k_off = 0; k_off < kGdnS; k_off += 8) {
      const uint4 aa =
          *reinterpret_cast<const uint4*>(&smem.s_ap[sb + mma_row][k_off]);
      const uint2 ub0 =
          *reinterpret_cast<const uint2*>(&smem.s_v[k_off + vk][vn_col]);
      const uint2 ub1 =
          *reinterpret_cast<const uint2*>(&smem.s_v[k_off + 4 + vk][vn_col]);
      FN_MMA_884_RR(vd_acc, aa.x, aa.y, ub0.x, ub0.y);
      FN_MMA_884_RR(vd_acc, aa.z, aa.w, ub1.x, ub1.y);
    }
    __syncthreads();  // Ensure all warps finished reading s_ap and s_v!
    {
      const float grv0 = smem.s_grev[sr0];
      const float grv2 = smem.s_grev[sr2];
      *reinterpret_cast<half2*>(&smem.s_vd[sr0][c_b]) =
          __floats2half2_rn(vd_acc[0], vd_acc[1]);
      *reinterpret_cast<half2*>(&smem.s_vd[sr2][c_b]) =
          __floats2half2_rn(vd_acc[2], vd_acc[3]);
      *reinterpret_cast<half2*>(&smem.s_vd[sr0][c_b + 4]) =
          __floats2half2_rn(vd_acc[4], vd_acc[5]);
      *reinterpret_cast<half2*>(&smem.s_vd[sr2][c_b + 4]) =
          __floats2half2_rn(vd_acc[6], vd_acc[7]);

      *reinterpret_cast<half2*>(&smem.s_v[sr0][c_b]) =
          __floats2half2_rn(vd_acc[0] * grv0, vd_acc[1] * grv0);
      *reinterpret_cast<half2*>(&smem.s_v[sr2][c_b]) =
          __floats2half2_rn(vd_acc[2] * grv2, vd_acc[3] * grv2);
      *reinterpret_cast<half2*>(&smem.s_v[sr0][c_b + 4]) =
          __floats2half2_rn(vd_acc[4] * grv0, vd_acc[5] * grv0);
      *reinterpret_cast<half2*>(&smem.s_v[sr2][c_b + 4]) =
          __floats2half2_rn(vd_acc[6] * grv2, vd_acc[7] * grv2);

      const int idx0 = tx, idx1 = tx + 256;
      *reinterpret_cast<uint4*>(&smem.s_ap[idx0 >> 3][(idx0 & 7) * 8]) = p_pre0;
      *reinterpret_cast<uint4*>(&smem.s_ap[idx1 >> 3][(idx1 & 7) * 8]) = p_pre1;
    }
    __syncthreads();  // Barrier #3

    // Prefetch chunk (i_s + 1)'s K, Q, A_bar, V, g BEFORE Step 4's 80 HMMAs!
    if (i_s + 1 < num_iters) {
      const int next_left = left + kGdnS;
      prefetch_chunk_inputs(next_left, min(kGdnS, seq_end - next_left));
    }

    // 4. O = scale * g_exp * (Q @ H_old) + P @ vd, AND H_new = g_last * H_old +
    // K^T @ vn!
    float o_acc[8] = {};
#pragma unroll
    for (int k_off = 0; k_off < kGdnK; k_off += 8) {
      const uint4 qa =
          *reinterpret_cast<const uint4*>(&smem.s_q[sb + mma_row][k_off]);
      const uint2 hb0 =
          *reinterpret_cast<const uint2*>(&smem.s_h[k_off + vk][vn_col]);
      const uint2 hb1 =
          *reinterpret_cast<const uint2*>(&smem.s_h[k_off + 4 + vk][vn_col]);
      FN_MMA_884_RR(o_acc, qa.x, qa.y, hb0.x, hb0.y);
      FN_MMA_884_RR(o_acc, qa.z, qa.w, hb1.x, hb1.y);
    }
    {
      const float sge0 = scale * smem.s_gexp[sr0];
      const float sge2 = scale * smem.s_gexp[sr2];
      o_acc[0] *= sge0;
      o_acc[1] *= sge0;
      o_acc[4] *= sge0;
      o_acc[5] *= sge0;
      o_acc[2] *= sge2;
      o_acc[3] *= sge2;
      o_acc[6] *= sge2;
      o_acc[7] *= sge2;
    }
#pragma unroll
    for (int k_off = 0; k_off < kGdnS; k_off += 8) {
      const uint4 pa =
          *reinterpret_cast<const uint4*>(&smem.s_ap[sb + mma_row][k_off]);
      const uint2 vdb0 =
          *reinterpret_cast<const uint2*>(&smem.s_vd[k_off + vk][vn_col]);
      const uint2 vdb1 =
          *reinterpret_cast<const uint2*>(&smem.s_vd[k_off + 4 + vk][vn_col]);
      FN_MMA_884_RR(o_acc, pa.x, pa.y, vdb0.x, vdb0.y);
      FN_MMA_884_RR(o_acc, pa.z, pa.w, vdb1.x, vdb1.y);
    }
    // Emit O directly to global HBM
    {
      auto emit_o2 = [&](int sr, int cb_off, float v0, float v1) {
        if (sr < valid_len && cb_off < bv) {
          *reinterpret_cast<half2*>(
              o_out + (static_cast<int64_t>(left + sr) * kGdnH + bh) * kGdnV +
              col_start + cb_off) = __floats2half2_rn(v0, v1);
        }
      };
      emit_o2(sr0, c_b, o_acc[0], o_acc[1]);
      emit_o2(sr2, c_b, o_acc[2], o_acc[3]);
      emit_o2(sr0, c_b + 4, o_acc[4], o_acc[5]);
      emit_o2(sr2, c_b + 4, o_acc[6], o_acc[7]);
    }

    // Update H_new = g_exp[63] * H_old + K^T @ vn
    {
      const float g_last = smem.s_gexp[kGdnS - 1];
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        h0[e] *= g_last;
        h1[e] *= g_last;
      }
      const int am = (lane & 16) ? 4 : 0;
      const int kb0 = warp * 16 + am;
      const int kb1 = warp * 16 + 8 + am;
#pragma unroll
      for (int k_off = 0; k_off < kGdnS; k_off += 4) {
        const uint2 vnb =
            *reinterpret_cast<const uint2*>(&smem.s_v[k_off + vk][vn_col]);
        const uint2 ka0 =
            *reinterpret_cast<const uint2*>(&smem.s_k[k_off + vk][kb0]);
        const uint2 ka1 =
            *reinterpret_cast<const uint2*>(&smem.s_k[k_off + vk][kb1]);
        FN_MMA_884_CR(h0, ka0.x, ka0.y, vnb.x, vnb.y);
        FN_MMA_884_CR(h1, ka1.x, ka1.y, vnb.x, vnb.y);
      }
    }
    __syncthreads();  // Barrier #4 before next chunk overwrites smem
  }

  if (store_final_state) {
    float* dst =
        inplace_final_state
            ? (h0_state + (static_cast<int64_t>(state_slot) * kGdnH + bh) *
                              (kGdnK * kGdnV))
            : (ht_out +
               (static_cast<int64_t>(bb) * kGdnH + bh) * (kGdnK * kGdnV));
    auto store_h_tile = [&](const float (&acc)[8], int kb) {
      const int kr0 = kb + r_off;
      const int kr2 = kr0 + 2;
      auto wr = [&](int kr, int vc, float val) {
        if (vc < bv) {
          const int gc = col_start + vc;
          if (state_layout_vlk) {
            dst[gc * kGdnK + kr] = val;
          } else {
            dst[kr * kGdnV + gc] = val;
          }
        }
      };
      wr(kr0, c_b, acc[0]);
      wr(kr0, c_b + 1, acc[1]);
      wr(kr2, c_b, acc[2]);
      wr(kr2, c_b + 1, acc[3]);
      wr(kr0, c_b + 4, acc[4]);
      wr(kr0, c_b + 5, acc[5]);
      wr(kr2, c_b + 4, acc[6]);
      wr(kr2, c_b + 5, acc[7]);
    };
    store_h_tile(h0, warp * 16);
    store_h_tile(h1, warp * 16 + 8);
  }
}

void gdn_chunk_fwd_sm70_out(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor a,
    torch::Tensor p_buf, torch::Tensor g, torch::Tensor b,
    torch::Tensor initial_state, torch::Tensor cu_seqlens,
    torch::Tensor chunk_indices, std::optional<torch::Tensor> state_indices,
    std::optional<torch::Tensor> has_initial_state, torch::Tensor out,
    std::optional<torch::Tensor> final_state, double scale,
    bool use_initial_state, bool state_layout_vlk, bool inplace_final_state,
    bool store_final_state) {
  const c10::cuda::CUDAGuard guard(q.device());
  const int cur_dev = q.device().index();
  const int num_chunks = static_cast<int>(chunk_indices.size(0));
  const int real_batch = static_cast<int>(cu_seqlens.size(0)) - 1;
  if (real_batch <= 0) return;
  auto stream = at::cuda::getCurrentCUDAStream();

  static bool configured_gdn_devs[16] = {};
  constexpr size_t kRecSmem = sizeof(GdnRecurrentSmem);
  if (cur_dev >= 0 && cur_dev < 16 && !configured_gdn_devs[cur_dev]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fn_gdn_recurrent_kernel<false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kRecSmem));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fn_gdn_recurrent_kernel<true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kRecSmem));
    configured_gdn_devs[cur_dev] = true;
  }

  if (num_chunks > 0) {
    dim3 prep_grid(num_chunks, kGdnH);
    if (b.scalar_type() == at::kFloat) {
      if (chunk_indices.scalar_type() == at::kInt) {
        fn_gdn_prep_ap_kernel<float, int32_t><<<prep_grid, 256, 0, stream>>>(
            reinterpret_cast<const half*>(q.data_ptr()),
            reinterpret_cast<const half*>(k.data_ptr()),
            reinterpret_cast<half*>(a.data_ptr()),
            reinterpret_cast<half*>(p_buf.data_ptr()), g.data_ptr<float>(),
            b.data_ptr<float>(), cu_seqlens.data_ptr<int32_t>(),
            chunk_indices.data_ptr<int32_t>(), static_cast<float>(scale));
      } else {
        fn_gdn_prep_ap_kernel<float, int64_t><<<prep_grid, 256, 0, stream>>>(
            reinterpret_cast<const half*>(q.data_ptr()),
            reinterpret_cast<const half*>(k.data_ptr()),
            reinterpret_cast<half*>(a.data_ptr()),
            reinterpret_cast<half*>(p_buf.data_ptr()), g.data_ptr<float>(),
            b.data_ptr<float>(), cu_seqlens.data_ptr<int32_t>(),
            chunk_indices.data_ptr<int64_t>(), static_cast<float>(scale));
      }
    } else {
      if (chunk_indices.scalar_type() == at::kInt) {
        fn_gdn_prep_ap_kernel<half, int32_t><<<prep_grid, 256, 0, stream>>>(
            reinterpret_cast<const half*>(q.data_ptr()),
            reinterpret_cast<const half*>(k.data_ptr()),
            reinterpret_cast<half*>(a.data_ptr()),
            reinterpret_cast<half*>(p_buf.data_ptr()), g.data_ptr<float>(),
            reinterpret_cast<const half*>(b.data_ptr()),
            cu_seqlens.data_ptr<int32_t>(), chunk_indices.data_ptr<int32_t>(),
            static_cast<float>(scale));
      } else {
        fn_gdn_prep_ap_kernel<half, int64_t><<<prep_grid, 256, 0, stream>>>(
            reinterpret_cast<const half*>(q.data_ptr()),
            reinterpret_cast<const half*>(k.data_ptr()),
            reinterpret_cast<half*>(a.data_ptr()),
            reinterpret_cast<half*>(p_buf.data_ptr()), g.data_ptr<float>(),
            reinterpret_cast<const half*>(b.data_ptr()),
            cu_seqlens.data_ptr<int32_t>(), chunk_indices.data_ptr<int64_t>(),
            static_cast<float>(scale));
      }
    }
  }

  const int32_t* st_idx_ptr =
      (state_indices.has_value() && state_indices->defined())
          ? state_indices->data_ptr<int32_t>()
          : nullptr;
  const bool* has_init_ptr =
      (has_initial_state.has_value() && has_initial_state->defined())
          ? has_initial_state->data_ptr<bool>()
          : nullptr;
  float* ht_ptr = (final_state.has_value() && final_state->defined())
                      ? final_state->data_ptr<float>()
                      : nullptr;

  const char* mode_env = std::getenv("FN_GDN_MODE");
  const int gdn_mode = mode_env ? std::atoi(mode_env) : 0;
  if (gdn_mode == 1) {
    fn_gdn_recurrent_kernel<true>
        <<<dim3(96, real_batch), 256, kRecSmem, stream>>>(
            reinterpret_cast<const half*>(q.data_ptr()),
            reinterpret_cast<const half*>(k.data_ptr()),
            reinterpret_cast<const half*>(v.data_ptr()),
            reinterpret_cast<const half*>(a.data_ptr()),
            reinterpret_cast<const half*>(p_buf.data_ptr()),
            g.data_ptr<float>(), initial_state.data_ptr<float>(),
            cu_seqlens.data_ptr<int32_t>(), st_idx_ptr, has_init_ptr,
            reinterpret_cast<half*>(out.data_ptr()), ht_ptr,
            static_cast<float>(scale), use_initial_state, state_layout_vlk,
            inplace_final_state, store_final_state);
  } else {
    fn_gdn_recurrent_kernel<false>
        <<<dim3(80, real_batch), 256, kRecSmem, stream>>>(
            reinterpret_cast<const half*>(q.data_ptr()),
            reinterpret_cast<const half*>(k.data_ptr()),
            reinterpret_cast<const half*>(v.data_ptr()),
            reinterpret_cast<const half*>(a.data_ptr()),
            reinterpret_cast<const half*>(p_buf.data_ptr()),
            g.data_ptr<float>(), initial_state.data_ptr<float>(),
            cu_seqlens.data_ptr<int32_t>(), st_idx_ptr, has_init_ptr,
            reinterpret_cast<half*>(out.data_ptr()), ht_ptr,
            static_cast<float>(scale), use_initial_state, state_layout_vlk,
            inplace_final_state, store_final_state);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gdn_chunk_fwd_fused_sm70_out(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor a_bar_buf,
    torch::Tensor p_buf, torch::Tensor g_cs_buf, torch::Tensor g_raw,
    torch::Tensor b, torch::Tensor initial_state, torch::Tensor cu_seqlens,
    torch::Tensor chunk_indices, std::optional<torch::Tensor> state_indices,
    std::optional<torch::Tensor> has_initial_state, torch::Tensor out,
    std::optional<torch::Tensor> final_state, double scale,
    bool use_initial_state, bool state_layout_vlk, bool inplace_final_state,
    bool store_final_state) {
  const c10::cuda::CUDAGuard guard(q.device());
  const int cur_dev = q.device().index();
  const int num_chunks = static_cast<int>(chunk_indices.size(0));
  const int real_batch = static_cast<int>(cu_seqlens.size(0)) - 1;
  if (real_batch <= 0) return;
  auto stream = at::cuda::getCurrentCUDAStream();

  static bool configured_gdn_fused_devs[16] = {};
  constexpr size_t kRecSmem = sizeof(GdnRecurrentSmem);
  if (cur_dev >= 0 && cur_dev < 16 && !configured_gdn_fused_devs[cur_dev]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fn_gdn_cumsum_kkt_prep_ap_kernel<float, int32_t>,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fn_gdn_cumsum_kkt_prep_ap_kernel<float, int64_t>,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fn_gdn_cumsum_kkt_prep_ap_kernel<half, int32_t>,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fn_gdn_cumsum_kkt_prep_ap_kernel<half, int64_t>,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fn_gdn_recurrent_kernel<false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kRecSmem));
    configured_gdn_fused_devs[cur_dev] = true;
  }

  if (num_chunks > 0) {
    dim3 prep_grid(num_chunks, kGdnH);
    if (b.scalar_type() == at::kFloat) {
      if (chunk_indices.scalar_type() == at::kInt) {
        fn_gdn_cumsum_kkt_prep_ap_kernel<float, int32_t>
            <<<prep_grid, 256, 0, stream>>>(
                reinterpret_cast<const half*>(q.data_ptr()),
                reinterpret_cast<const half*>(k.data_ptr()),
                reinterpret_cast<half*>(a_bar_buf.data_ptr()),
                reinterpret_cast<half*>(p_buf.data_ptr()),
                g_cs_buf.data_ptr<float>(), g_raw.data_ptr<float>(),
                b.data_ptr<float>(), cu_seqlens.data_ptr<int32_t>(),
                chunk_indices.data_ptr<int32_t>(), static_cast<float>(scale));
      } else {
        fn_gdn_cumsum_kkt_prep_ap_kernel<float, int64_t>
            <<<prep_grid, 256, 0, stream>>>(
                reinterpret_cast<const half*>(q.data_ptr()),
                reinterpret_cast<const half*>(k.data_ptr()),
                reinterpret_cast<half*>(a_bar_buf.data_ptr()),
                reinterpret_cast<half*>(p_buf.data_ptr()),
                g_cs_buf.data_ptr<float>(), g_raw.data_ptr<float>(),
                b.data_ptr<float>(), cu_seqlens.data_ptr<int32_t>(),
                chunk_indices.data_ptr<int64_t>(), static_cast<float>(scale));
      }
    } else {
      if (chunk_indices.scalar_type() == at::kInt) {
        fn_gdn_cumsum_kkt_prep_ap_kernel<half, int32_t>
            <<<prep_grid, 256, 0, stream>>>(
                reinterpret_cast<const half*>(q.data_ptr()),
                reinterpret_cast<const half*>(k.data_ptr()),
                reinterpret_cast<half*>(a_bar_buf.data_ptr()),
                reinterpret_cast<half*>(p_buf.data_ptr()),
                g_cs_buf.data_ptr<float>(), g_raw.data_ptr<float>(),
                reinterpret_cast<const half*>(b.data_ptr()),
                cu_seqlens.data_ptr<int32_t>(),
                chunk_indices.data_ptr<int32_t>(), static_cast<float>(scale));
      } else {
        fn_gdn_cumsum_kkt_prep_ap_kernel<half, int64_t>
            <<<prep_grid, 256, 0, stream>>>(
                reinterpret_cast<const half*>(q.data_ptr()),
                reinterpret_cast<const half*>(k.data_ptr()),
                reinterpret_cast<half*>(a_bar_buf.data_ptr()),
                reinterpret_cast<half*>(p_buf.data_ptr()),
                g_cs_buf.data_ptr<float>(), g_raw.data_ptr<float>(),
                reinterpret_cast<const half*>(b.data_ptr()),
                cu_seqlens.data_ptr<int32_t>(),
                chunk_indices.data_ptr<int64_t>(), static_cast<float>(scale));
      }
    }
  }

  const int32_t* st_idx_ptr =
      (state_indices.has_value() && state_indices->defined())
          ? state_indices->data_ptr<int32_t>()
          : nullptr;
  const bool* has_init_ptr =
      (has_initial_state.has_value() && has_initial_state->defined())
          ? has_initial_state->data_ptr<bool>()
          : nullptr;
  float* ht_ptr = (final_state.has_value() && final_state->defined())
                      ? final_state->data_ptr<float>()
                      : nullptr;

  fn_gdn_recurrent_kernel<false>
      <<<dim3(80, real_batch), 256, kRecSmem, stream>>>(
          reinterpret_cast<const half*>(q.data_ptr()),
          reinterpret_cast<const half*>(k.data_ptr()),
          reinterpret_cast<const half*>(v.data_ptr()),
          reinterpret_cast<const half*>(a_bar_buf.data_ptr()),
          reinterpret_cast<const half*>(p_buf.data_ptr()),
          g_cs_buf.data_ptr<float>(), initial_state.data_ptr<float>(),
          cu_seqlens.data_ptr<int32_t>(), st_idx_ptr, has_init_ptr,
          reinterpret_cast<half*>(out.data_ptr()), ht_ptr,
          static_cast<float>(scale), use_initial_state, state_layout_vlk,
          inplace_final_state, store_final_state);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace gdn_sm70

}  // namespace vllm_fn_sm70

TORCH_LIBRARY(vllm_sm70_fn_prefill, m) {
  m.def(
      "moe_prefill_raw(Tensor(a!) out, Tensor x, Tensor topk_ids, "
      "Tensor topk_weights, Tensor w13_weight, Tensor w13_scale_codes, "
      "Tensor w13_global_scales, Tensor w2_weight, Tensor w2_scale_codes, "
      "Tensor w2_global_scales, Tensor? shared_out, Tensor? shared_gate_logits,"
      " Tensor(b!) ws_counts, Tensor(c!) ws_offsets, Tensor(d!) ws_cursors, "
      "Tensor(e!) ws_sorted_token_idx, Tensor(f!) ws_sorted_weights, "
      "Tensor(g!) ws_tiles, Tensor(h!) ws_num_tiles, Tensor(i!) ws_inter, "
      "Tensor(j!) ws_accum_f32, bool scale_0125=False, int s0=0, int s1=0, "
      "int s2=0, int s3=0, Tensor(k!)? ep_state=None, int my_rank=0) -> ()");
  m.impl("moe_prefill_raw", torch::kCUDA, &vllm_fn_sm70::moe_prefill_raw_out);

  m.def("shared_gate_mul_inplace(Tensor(a!) out, Tensor gate_logits) -> ()");
  m.impl("shared_gate_mul_inplace", torch::kCUDA,
         &vllm_fn_sm70::shared_gate_mul_inplace_out);

  m.def(
      "shared_gate_dot_mul_inplace(Tensor(a!) out, Tensor x, "
      "Tensor gate_weight) -> ()");
  m.impl("shared_gate_dot_mul_inplace", torch::kCUDA,
         &vllm_fn_sm70::shared_gate_dot_mul_inplace_out);

  m.def(
      "shared_gate_dot(Tensor(a!) gate_logits, Tensor x, "
      "Tensor gate_weight) -> ()");
  m.impl("shared_gate_dot", torch::kCUDA, &vllm_fn_sm70::shared_gate_dot_out);

  m.def(
      "silu_mul_split_gate(Tensor(a!) act_out, Tensor(b!) gate_logits, "
      "Tensor raw) -> ()");
  m.impl("silu_mul_split_gate", torch::kCUDA,
         &vllm_fn_sm70::silu_mul_split_gate_out);

  m.def(
      "split_router_silu_mul_gate(Tensor(a!) router_out, Tensor(b!) act_out, "
      "Tensor(c!) gate_logits, Tensor raw) -> ()");
  m.impl("split_router_silu_mul_gate", torch::kCUDA,
         &vllm_fn_sm70::split_router_silu_mul_gate_out);

  m.def(
      "split_gdn_4120_tails(Tensor(a!) z_out, Tensor(b!) b_out, "
      "Tensor(c!) a_out, Tensor raw) -> ()");
  m.impl("split_gdn_4120_tails", torch::kCUDA,
         &vllm_fn_sm70::split_gdn_4120_tails_out);

  m.def(
      "split_gdn_qkvz_ba_tails(Tensor(a!) z_out, Tensor(b!) b_out, "
      "Tensor(c!) a_out, Tensor qkvz, Tensor ba) -> ()");
  m.impl("split_gdn_qkvz_ba_tails", torch::kCUDA,
         &vllm_fn_sm70::split_gdn_qkvz_ba_tails_out);

  m.def("init_qsa_merge_workspace(int max_m, int width, int device_idx) -> ()");
  m.impl("init_qsa_merge_workspace",
         &vllm_fn_sm70::init_qsa_merge_workspace_out);

  m.def(
      "run_rows_h12(Tensor q, Tensor history, Tensor scales, Tensor indices, "
      "Tensor table, Tensor requests, Tensor positions, Tensor lengths, "
      "Tensor(a!) out, Tensor? gate, Tensor(b!)? lse=None) -> ()");
  m.impl("run_rows_h12", torch::kCUDA, &vllm_fn_sm70::run_rows_h12);

  m.def(
      "dcp2_q_prep_any_m(Tensor query, Tensor(a!) gathered_q, Tensor positions,"
      " Tensor lengths, Tensor(b!) local_positions, Tensor(c!) local_lengths, "
      "int my_buf_ptr, int peer_buf_ptr, Tensor(d!) seq_state, int dcp_rank, "
      "int interleave) -> ()");
  m.impl("dcp2_q_prep_any_m", torch::kCUDA,
         &vllm_fn_sm70::dcp2_any_m::dcp2_q_prep_any_m_out);

  m.def(
      "dcp2_combine_gate_any_m(Tensor partial_out, Tensor partial_lse, "
      "Tensor gate, Tensor(a!) out, int my_buf_ptr, int peer_buf_ptr, "
      "Tensor(b!) seq_state, int dcp_rank) -> ()");
  m.impl("dcp2_combine_gate_any_m", torch::kCUDA,
         &vllm_fn_sm70::dcp2_any_m::dcp2_combine_gate_any_m_out);

  m.def("build_staged_map(Tensor(a!) staged_of, Tensor pages) -> ()");
  m.impl("build_staged_map", torch::kCUDA,
         &vllm_fn_sm70::qsa_stage_write::build_staged_map_out);

  m.def(
      "write_staged_kv(Tensor key, Tensor val, Tensor slots, Tensor staged_of, "
      "Tensor(a!) history, Tensor(b!) scales, Tensor? tags=None, "
      "Tensor? page_slots=None, Tensor(c!)? hot_values=None, "
      "Tensor(d!)? hot_scales=None, bool hot_fp8=False) -> ()");
  m.impl("write_staged_kv", torch::kCUDA,
         &vllm_fn_sm70::qsa_stage_write::write_staged_kv_out);

  m.def(
      "copy_qsa_pages(Tensor(a!) dst_hist, Tensor(b!)? dst_scales, "
      "Tensor src_hist, Tensor? src_scales, Tensor dst_pages, "
      "Tensor src_pages, int max_ctas=0) -> ()");
  m.impl("copy_qsa_pages", torch::kCUDA,
         &vllm_fn_sm70::qsa_stage_write::copy_qsa_pages_out);

  m.def(
      "async_persist_qsa_page(Tensor(a!) host_hist, Tensor(b!)? host_scales, "
      "Tensor gpu_hist, Tensor? gpu_scales, int phys_blk, int cache_slot) -> "
      "()");
  m.impl("async_persist_qsa_page", torch::kCUDA,
         &vllm_fn_sm70::qsa_stage_write::async_persist_qsa_page_out);

  m.def(
      "async_persist_qsa_pages_batch(Tensor(a!) host_hist, "
      "Tensor(b!)? host_scales, Tensor gpu_hist, Tensor? gpu_scales, "
      "int[] phys_blks, int[] cache_slots) -> ()");
  m.impl("async_persist_qsa_pages_batch", torch::kCUDA,
         &vllm_fn_sm70::qsa_stage_write::async_persist_qsa_pages_batch_out);

  m.def("inc_hot_epoch(Tensor(a!) epoch) -> ()");
  m.impl("inc_hot_epoch", torch::kCUDA, [](torch::Tensor epoch) {
    const c10::cuda::CUDAGuard guard(epoch.device());
    vllm_fn_sm70::qsa_stage_write::
        inc_hot_epoch_kernel<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
            epoch.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });

  m.def(
      "hc_p2p_reduce_combine_norm(Tensor(a!) updated, Tensor(b!) normalized, "
      "Tensor hidden, Tensor first, Tensor? second, Tensor injection, "
      "Tensor weight, float eps, bool normalize, int p0, int p1, int p2, "
      "int p3, int s0, int s1, int s2, int s3, Tensor(c!) ep_state, "
      "int bank_bytes, int my_rank, bool skip_stage=False, "
      "bool pre_scaled8=False, bool already_signaled=False) -> ()");
  m.impl("hc_p2p_reduce_combine_norm", torch::kCUDA,
         &vllm_fn_sm70::hc_p2p::hc_p2p_reduce_combine_norm_out);

  m.def(
      "hc_p2p_unpack_gather_weights(Tensor(a!) full_down, Tensor(b!) full_up, "
      "Tensor down_shard, Tensor up_shard, bool already_unpacked, int w0, "
      "int w1, int w2, int w3, int s0, int s1, int s2, int s3, "
      "Tensor(c!) ep_state, int my_rank, int ord0, int ord1, int ord2, "
      "int ord3, bool async_low_sm=False) -> ()");
  m.impl("hc_p2p_unpack_gather_weights", torch::kCUDA,
         &vllm_fn_sm70::hc_p2p::hc_p2p_unpack_gather_weights_out);

  m.def(
      "hc_p2p_init_static_weight_slot(Tensor down_shard, Tensor up_shard, "
      "bool already_unpacked, int static_slot, int sw0, int sw1, int sw2, "
      "int sw3, int s0, int s1, int s2, int s3, Tensor(a!) ep_state, "
      "int my_rank) -> ()");
  m.impl("hc_p2p_init_static_weight_slot", torch::kCUDA,
         &vllm_fn_sm70::hc_p2p::hc_p2p_init_static_weight_slot_out);

  m.def(
      "hc_p2p_gather_static_weights(Tensor(a!) full_down, Tensor(b!) full_up, "
      "int static_slot, int sw0, int sw1, int sw2, int sw3, int my_rank, "
      "int ord0, int ord1, int ord2, int ord3) -> ()");
  m.impl("hc_p2p_gather_static_weights", torch::kCUDA,
         &vllm_fn_sm70::hc_p2p::hc_p2p_gather_static_weights_out);

  m.def(
      "hc_split_silu(Tensor(a!) lora, Tensor(b!) inj, Tensor projected) -> ()");
  m.impl("hc_split_silu", torch::kCUDA,
         &vllm_fn_sm70::hc_p2p::hc_split_silu_out);

  m.def(
      "hc_down_proj_split_silu(Tensor(a!) lora, Tensor(b!) inj, Tensor x, "
      "Tensor full_down) -> ()");
  m.impl("hc_down_proj_split_silu", torch::kCUDA,
         &vllm_fn_sm70::hc_p2p::hc_down_proj_split_silu_out);

  m.def(
      "hc_down_splitk2_silu(Tensor(a!) lora, Tensor(b!) inj, Tensor x, "
      "Tensor full_down, Tensor(c!) partial_f32) -> ()");
  m.impl("hc_down_splitk2_silu", torch::kCUDA,
         &vllm_fn_sm70::hc_p2p::hc_down_splitk2_silu_out);

  m.def(
      "hc_p2p_allgather_rows(Tensor(a!) out, Tensor src, int p0, int p1, "
      "int p2, int p3, int s0, int s1, int s2, int s3, Tensor(b!) ep_state, "
      "int bank_bytes, int my_rank) -> ()");
  m.impl("hc_p2p_allgather_rows", torch::kCUDA,
         &vllm_fn_sm70::hc_p2p::hc_p2p_allgather_rows_out);

  m.def(
      "hc_p2p_allgather_topk(Tensor(a!) out_w, Tensor(b!) out_ids, "
      "Tensor loc_w, Tensor loc_ids, int step, int p0, int p1, int p2, int p3, "
      "int s0, int s1, int s2, int s3, Tensor(c!) ep_state, int bank_bytes, "
      "int my_rank) -> ()");
  m.impl("hc_p2p_allgather_topk", torch::kCUDA,
         &vllm_fn_sm70::hc_p2p::hc_p2p_allgather_topk_out);

  m.def(
      "hc_p2p_gate_mix_push4(Tensor x, Tensor gate, int a0, int a1, int a2, "
      "int a3, int s0, int s1, int s2, int s3, Tensor(a!) ep_state, "
      "int bank_bytes, int my_rank) -> ()");
  m.impl("hc_p2p_gate_mix_push4", torch::kCUDA,
         &vllm_fn_sm70::hc_p2p::hc_p2p_gate_mix_push4_out);

  m.def(
      "hc_p2p_gate_mix_local_only(Tensor x, Tensor gate, int a0, int a1, "
      "int a2, int a3, int s0, int s1, int s2, int s3, Tensor(a!) ep_state, "
      "int bank_bytes, int my_rank) -> ()");
  m.impl("hc_p2p_gate_mix_local_only", torch::kCUDA,
         &vllm_fn_sm70::hc_p2p::hc_p2p_gate_mix_local_only_out);

  m.def(
      "hc_p2p_gather_peers_async(int out_rows, int local_rows, int a0, int a1, "
      "int a2, int a3, int my_sync, Tensor ep_state, int bank_bytes, "
      "int my_rank, int num_ctas) -> ()");
  m.impl("hc_p2p_gather_peers_async", torch::kCUDA,
         &vllm_fn_sm70::hc_p2p::hc_p2p_gather_peers_async_out);

  m.def(
      "hc_p2p_gate_mix_peers_pull(Tensor x, Tensor gate, int out_rows, int a0, "
      "int a1, int a2, int a3, int s0, int s1, int s2, int s3, "
      "Tensor(a!) ep_state, int bank_bytes, int my_rank) -> ()");
  m.impl("hc_p2p_gate_mix_peers_pull", torch::kCUDA,
         &vllm_fn_sm70::hc_p2p::hc_p2p_gate_mix_peers_pull_out);

  m.def(
      "hc_p2p_gate_mix_duplex(Tensor x, Tensor gate, int out_rows, int a0, "
      "int a1, int a2, int a3, int s0, int s1, int s2, int s3, "
      "Tensor(a!) ep_state, int bank_bytes, int my_rank) -> ()");
  m.impl("hc_p2p_gate_mix_duplex", torch::kCUDA,
         &vllm_fn_sm70::hc_p2p::hc_p2p_gate_mix_duplex_out);

  m.def(
      "hc_p2p_gate_mix_allgather(Tensor(a!) block_out, Tensor x, Tensor gate, "
      "int a0, int a1, int a2, int a3, int s0, int s1, int s2, int s3, "
      "Tensor(b!) ep_state, int bank_bytes, int my_rank) -> ()");
  m.impl("hc_p2p_gate_mix_allgather", torch::kCUDA,
         &vllm_fn_sm70::hc_p2p::hc_p2p_gate_mix_allgather_out);

  m.def(
      "hc_p2p_up_gate_mix_allgather(Tensor(a!) block_out, Tensor x, "
      "Tensor lora, Tensor full_up, int a0, int a1, int a2, int a3, int s0, "
      "int s1, int s2, int s3, Tensor(b!) ep_state, int bank_bytes, "
      "int my_rank) -> ()");
  m.impl("hc_p2p_up_gate_mix_allgather", torch::kCUDA,
         &vllm_fn_sm70::hc_p2p::hc_p2p_up_gate_mix_allgather_out);

  m.def(
      "hc_p2p_up_gate_mix_local_only(Tensor x, Tensor lora, Tensor full_up, "
      "int a0, int a1, int a2, int a3, int s0, int s1, int s2, int s3, "
      "Tensor(a!) ep_state, int bank_bytes, int my_rank) -> ()");
  m.impl("hc_p2p_up_gate_mix_local_only", torch::kCUDA,
         &vllm_fn_sm70::hc_p2p::hc_p2p_up_gate_mix_local_only_out);

  m.def(
      "hc_p2p_up_gate_mix_peers_pull(Tensor x, Tensor lora, Tensor full_up, "
      "int out_rows, int a0, int a1, int a2, int a3, int s0, int s1, int s2, "
      "int s3, Tensor(a!) ep_state, int bank_bytes, int my_rank) -> ()");
  m.impl("hc_p2p_up_gate_mix_peers_pull", torch::kCUDA,
         &vllm_fn_sm70::hc_p2p::hc_p2p_up_gate_mix_peers_pull_out);

  m.def(
      "gdn_rmsnorm_gated_h12d128(Tensor(a!) out, Tensor x, Tensor z, "
      "Tensor weight, float eps) -> ()");
  m.impl("gdn_rmsnorm_gated_h12d128", torch::kCUDA,
         &vllm_fn_sm70::hc_p2p::gdn_rmsnorm_gated_h12d128_out);

  m.def(
      "wrap_ipc_half_2d(int ptr, int rows, int cols, int device_idx) -> "
      "Tensor");
  m.impl("wrap_ipc_half_2d", &vllm_fn_sm70::hc_p2p::wrap_ipc_half_2d);

  m.def(
      "gdn_chunk_fwd_sm70(Tensor q, Tensor k, Tensor v, Tensor(a!) a, "
      "Tensor(b!) p_buf, Tensor g, Tensor b, Tensor(c!) initial_state, "
      "Tensor cu_seqlens, Tensor chunk_indices, Tensor? state_indices, "
      "Tensor? has_initial_state, Tensor(d!) out, Tensor(e!)? final_state, "
      "float scale, bool use_initial_state, bool state_layout_vlk, "
      "bool inplace_final_state, bool store_final_state) -> ()");
  m.impl("gdn_chunk_fwd_sm70", torch::kCUDA,
         &vllm_fn_sm70::gdn_sm70::gdn_chunk_fwd_sm70_out);

  m.def(
      "gdn_chunk_fwd_fused_sm70(Tensor q, Tensor k, Tensor v, "
      "Tensor(a!) a_bar_buf, Tensor(b!) p_buf, Tensor(c!) g_cs_buf, "
      "Tensor g_raw, Tensor b, Tensor(d!) initial_state, Tensor cu_seqlens, "
      "Tensor chunk_indices, Tensor? state_indices, Tensor? has_initial_state, "
      "Tensor(e!) out, Tensor(f!)? final_state, float scale, "
      "bool use_initial_state, bool state_layout_vlk, "
      "bool inplace_final_state, bool store_final_state) -> ()");
  m.impl("gdn_chunk_fwd_fused_sm70", torch::kCUDA,
         &vllm_fn_sm70::gdn_sm70::gdn_chunk_fwd_fused_sm70_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "SM70 Qwen3.8-Flash-Next high-throughput prefill operators v6";
}
