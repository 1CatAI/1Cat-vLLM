// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 FlashInfer team
// SM70 adaptation by 1Cat-vLLM contributors. Source: FlashInfer
// 6c14bbd5ff34210404d5d4b5f6ff3b4b2527f59f, gdn_fused_decode_sm120.cu.
#pragma once
/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <math.h>

namespace flashinfer::sm70::gdn {

// Fused GDN decode step for one layer geometry: a single persistent kernel
// covering the in_proj_ba GEMV, the depthwise causal conv1d update (width 4,
// silu), the q/k/v head split, and the gated delta-rule state update with
// qk-L2-norm, replacing the multi-launch serving chain.
//
// The layer geometry is a compile-time parameter of this translation unit,
// supplied by benchmarks/kernels/flashinfer_sm70_gdn_conv.py (one JIT
// module per geometry; a serving process runs one model, hence one module).
// Only the sizes change: the block shape, warp->row mapping and reduction
// trees below are geometry-independent, and the static_asserts state exactly
// which divisibility relations the code relies on.
#if !defined(FI_GDN_HIDDEN) || !defined(FI_GDN_N_BA) ||                        \
    !defined(FI_GDN_QKV_DIM) || !defined(FI_GDN_H_Q) || !defined(FI_GDN_HV) || \
    !defined(FI_GDN_D) || !defined(FI_GDN_CONV_WIDTH) ||                       \
    !defined(FI_GDN_CONV_STATE_LEN)
  #error "gdn_fused_decode.cuh requires the FI_GDN_* geometry defines"
#endif
constexpr int HIDDEN = FI_GDN_HIDDEN;
constexpr int N_BA = FI_GDN_N_BA;
constexpr int QKV_DIM = FI_GDN_QKV_DIM;
constexpr int H_Q = FI_GDN_H_Q;
constexpr int HV = FI_GDN_HV;
constexpr int D = FI_GDN_D;
constexpr int CONV_WIDTH = FI_GDN_CONV_WIDTH;
constexpr int CONV_STATE_LEN = FI_GDN_CONV_STATE_LEN;
// v-heads per qk-head: the delta phase maps v-head h to qk-head h/HEADS_PER_QK.
constexpr int HEADS_PER_QK = HV / H_Q;
constexpr int ROWS_PER_WARP = 8;
constexpr int GEMV_NSPLIT = 160;
// The gate reduction below unrolls the GEMV partials as 5 warp-wide loads.
static_assert(GEMV_NSPLIT == 5 * 32,
              "gate reduction assumes 5 warp-strided loads");
// The b/a projection produces HV gate values and HV decay values, stored as
// the low and high halves of the N_BA columns (see the delta phase's
// base_b/base_a offsets).
static_assert(N_BA == 2 * HV, "w_ba columns are [b gates | a decays], HV each");
static_assert(HV % H_Q == 0,
              "each qk-head must serve a whole number of v-heads");
// The delta phase gives each lane 4 consecutive channels of a D-wide row and
// reduces across the warp; the B=1 fast path indexes rows with shifts/masks.
static_assert(D == 4 * 32,
              "delta phase maps one D-wide row onto a warp, 4 per lane");
static_assert((D & (D - 1)) == 0,
              "B=1 row index math uses D as a power of two");
static_assert(D % ROWS_PER_WARP == 0, "warps own whole groups of state rows");
// mixed_qkv is [q | k | v] with H_Q q-heads, H_Q k-heads and HV v-heads.
static_assert(QKV_DIM == (2 * H_Q + HV) * D,
              "qkv_dim must match the head split");
// The conv phase is unrolled over the width-4 / 3-step shift register below.
static_assert(CONV_WIDTH == 4 && CONV_STATE_LEN == 3,
              "conv taps are unrolled as width 4");

// log2(D), for the B=1 fast path's shift/mask row indexing.
constexpr int ilog2_ce(int v) { return v <= 1 ? 0 : 1 + ilog2_ce(v >> 1); }
constexpr int D_LOG2 = ilog2_ce(D);

typedef half f16;

__device__ __forceinline__ float siluf(float x) {
  return x / (1.0f + __expf(-x));
}
__device__ __forceinline__ float sigmoidf(float x) {
  return 1.0f / (1.0f + __expf(-x));
}
__device__ __forceinline__ float softplusf(float x) {
  return x > 20.0f ? x : log1pf(__expf(x));
}
__device__ __forceinline__ float warp_reduce(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
  return v;
}

// Cooperative launch guarantees grid residency even when vLLM has auxiliary
// streams. Use CUDA's specified grid synchronization instead of an occupancy-
// capped regular launch with a software spin barrier.
__device__ __forceinline__ void grid_barrier() {
  cooperative_groups::this_grid().sync();
}

// Single persistent kernel: gemv+conv -> [barrier] -> delta. Cooperative
// launch. The kB1 instantiation specializes the serving-hot B=1 case:
// batch/split index math collapses to compile-time constants and the fp32 state
// rows for the (single) delta task of each warp are prefetched before the
// barrier so their long-scoreboard latency overlaps the gemv/conv phases (the
// state pool is only written after the barrier, each row by the warp that
// prefetched it).
//
// Padded batch rows: a NEGATIVE state index (vLLM's PAD_SLOT_ID = -1) marks a
// batch row that owns no pool slot -- what a CUDA-graph replay carries in the
// rows between the live request count and the captured batch size. Such a row
// is skipped in every phase that touches a pool (no read and no write of
// conv_state / ssm_state) and its output rows are written as zero, matching
// the fp32 path of gated_delta_rule_decode_pretranspose. The check has to be
// here rather than on the host: reading index VALUES host-side costs a
// device-to-host sync per layer per decode step and is impossible under graph
// capture. Each guard is uniform over the threads that share a batch row
// (warp-uniform in the delta phase, where the warp-wide shuffles below make
// divergence unacceptable), so it costs one predicated branch, not divergence.
// Indices >= P are NOT padding and are not checked -- see the note on
// state_indices in the FFI entry point below.
//
// Aliasing: the op updates both state pools IN PLACE, so the launcher passes
// the same pointer for (conv_state, updated_conv) and for (ssm_state,
// ssm_out).  Those four parameters therefore carry no __restrict__ -- the
// pools are read and written through two different parameters, which is
// exactly what restrict promises does not happen, and promising it would let
// the compiler reorder a pool load across a pool store.  The remaining
// pointers are genuinely disjoint buffers and keep the qualifier.  The read
// path pays for this: loads from the pools can no longer be promoted to
// ld.global.nc.  Only this impl is affected -- the registry prefers the
// CuTe-DSL kernel for every shipped row -- and correctness is not a thing to
// trade for a read-only-cache hint.
template <bool kB1>
__global__ void gdn_fused_decode_kernel(
    const f16* __restrict__ hidden, const f16* __restrict__ w_ba,
    const f16* __restrict__ mixed_qkv, const f16* __restrict__ conv_weight,
    const f16* __restrict__ conv_bias, const f16* conv_state,
    const float* __restrict__ A_log, const f16* __restrict__ dt_bias,
    const float* ssm_state, const int* __restrict__ state_indices, float scale,
    long state_stride_0, long qkv_stride, long conv_stride_p,
    long conv_stride_c, long conv_stride_t, f16* __restrict__ output,
    f16* updated_conv, float* ssm_out, float* __restrict__ ba_part,
    f16* __restrict__ conv_out, int B) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;
  const int Beff = kB1 ? 1 : B;

  // ---- Phase A1: GEMV partials ----
  // tasks: (split in 0..GEMV_NSPLIT-1) x (b) x (col in 0..N_BA-1). Partials
  // are stored split-major per (col, b) so the gate reduction reads them with
  // warp-coalesced loads.
  long gemv_tasks = (long)GEMV_NSPLIT * Beff * N_BA;
  for (long t = tid; t < gemv_tasks; t += nthreads) {
    int col = t % N_BA;
    long r = t / N_BA;
    int b;
    int split;
    if constexpr (kB1) {
      b = 0;
      split = (int)r;
    } else {
      b = r % B;
      split = r / B;
    }
    const f16* hrow = hidden + (long)b * HIDDEN;
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    int k = split;
    for (; k + 3 * GEMV_NSPLIT < HIDDEN; k += 4 * GEMV_NSPLIT) {
      a0 += __half2float(hrow[k]) * __half2float(w_ba[(long)k * N_BA + col]);
      a1 += __half2float(hrow[k + GEMV_NSPLIT]) *
            __half2float(w_ba[(long)(k + GEMV_NSPLIT) * N_BA + col]);
      a2 += __half2float(hrow[k + 2 * GEMV_NSPLIT]) *
            __half2float(w_ba[(long)(k + 2 * GEMV_NSPLIT) * N_BA + col]);
      a3 += __half2float(hrow[k + 3 * GEMV_NSPLIT]) *
            __half2float(w_ba[(long)(k + 3 * GEMV_NSPLIT) * N_BA + col]);
    }
    for (; k < HIDDEN; k += GEMV_NSPLIT)
      a0 += __half2float(hrow[k]) * __half2float(w_ba[(long)k * N_BA + col]);
    ba_part[((long)col * Beff + b) * GEMV_NSPLIT + split] =
        (a0 + a1) + (a2 + a3);
  }

  // ---- Phase A2: conv (independent of gemv) ----
  long conv_tasks = (long)Beff * QKV_DIM;
  for (long t = tid; t < conv_tasks; t += nthreads) {
    int b;
    int c;
    if constexpr (kB1) {
      b = 0;
      c = (int)t;
    } else {
      b = t / QKV_DIM;
      c = t % QKV_DIM;
    }
    int idx = state_indices[b];
    // Padded row: owns no conv-state slot, so neither shift it nor append to
    // it. conv_out for this row stays whatever the scratch held -- the delta
    // phase skips the same row, so nothing reads it.
    if (idx < 0) continue;
    // conv_state addressing is stride-parameterized: the pool arrives as a
    // logical [P, QKV_DIM, CONV_STATE_LEN] view of either a DS-dense pool
    // (strides p,3,1 -> per-thread 3-element rows) or a transposed SD pool
    // (strides p,1,QKV_DIM -> fully coalesced across channels, the vLLM
    // default). Pure index arithmetic; the update math is identical.
    const f16* st =
        conv_state + (long)idx * conv_stride_p + (long)c * conv_stride_c;
    f16 s0 = st[0], s1 = st[conv_stride_t], s2 = st[2 * conv_stride_t];
    // mixed_qkv rows may be strided (e.g. a view into a wider projection).
    f16 xr = mixed_qkv[(long)b * qkv_stride + c];
    const f16* w = conv_weight + (long)c * CONV_WIDTH;
    // vLLM's FP16 conv actually emits mul.f16 then cvt.f32.f16, with
    // FP32 accumulation. Preserve those load-bearing product roundings;
    // widening before the multiply changes this model's recurrent inputs.
    float y = conv_bias ? __half2float(conv_bias[c]) : 0.f;
    y += __half2float(__hmul(s0, w[0]));
    y += __half2float(__hmul(s1, w[1]));
    y += __half2float(__hmul(s2, w[2]));
    y += __half2float(__hmul(xr, w[3]));
    conv_out[(long)b * QKV_DIM + c] = __float2half_rn(siluf(y));
    f16* uc =
        updated_conv + (long)idx * conv_stride_p + (long)c * conv_stride_c;
    uc[0] = s1;
    uc[conv_stride_t] = s2;
    uc[2 * conv_stride_t] = xr;
  }

  // ---- Pre-barrier prefetch of this warp's first delta task's state rows.
  // The state pool is read-only until phase C, and each row is written only
  // by the warp that owns (and prefetched) it, so this is race-free.
  int gwarp = tid >> 5;
  int lane = threadIdx.x & 31;
  int nwarps = nthreads >> 5;
  long total_rows = (long)Beff * HV * D;
  long warps_needed = (total_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;

  float4 s_pre[ROWS_PER_WARP];
  long pre_row_base = -1;
  if (gwarp < warps_needed) {
    long first_row = (long)gwarp * ROWS_PER_WARP;
    int v0;
    int h;
    int b;
    if constexpr (kB1) {
      v0 = (int)(first_row & (D - 1));
      h = (int)(first_row >> D_LOG2);
      b = 0;
    } else {
      v0 = first_row % D;
      long tmp = first_row / D;
      h = tmp % HV;
      b = tmp / HV;
    }
    int idx = state_indices[b];
    // A padded row has no state row to prefetch. pre_row_base stays -1, which
    // no live row_base can equal, so the delta phase never mistakes the
    // (unwritten) s_pre registers for this warp's prefetched rows.
    if (idx >= 0) {
      pre_row_base = (long)idx * state_stride_0 + (long)h * (D * D);
      const float4* base_srow =
          (const float4*)(ssm_state + pre_row_base + (long)v0 * D);
#pragma unroll
      for (int r = 0; r < ROWS_PER_WARP; ++r)
        s_pre[r] = base_srow[r * (D / 4) + lane];
    }
  }

  grid_barrier();

  // ---- Phase C: delta (gate reduced inline from ba_part) ----
  for (long w = gwarp; w < warps_needed; w += nwarps) {
    long first_row = w * ROWS_PER_WARP;
    int v0;
    int h;
    int b;
    if constexpr (kB1) {
      v0 = (int)(first_row & (D - 1));
      h = (int)(first_row >> D_LOG2);
      b = 0;
    } else {
      v0 = first_row % D;
      long tmp = first_row / D;
      h = tmp % HV;
      b = tmp / HV;
    }
    int j = h / HEADS_PER_QK;
    const f16* co = conv_out + (long)b * QKV_DIM;
    const f16* qb = co + j * D;
    const f16* kb = co + H_Q * D + j * D;
    int k0 = lane * 4;
    int idx = state_indices[b];
    if (idx < 0) {
      // Padded row: no state row to read or write; its output rows are zero.
      // b (hence idx) is warp-uniform -- first_row is a multiple of
      // ROWS_PER_WARP and the whole warp shares w -- so the whole warp takes
      // this branch together and the warp-wide shuffles below are never
      // reached with a partial mask.
      if (lane == 0) {
#pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r)
          output[((long)b * HV + h) * D + v0 + r] = __float2half_rn(0.0f);
      }
      continue;
    }
    long row_base = (long)idx * state_stride_0 + (long)h * (D * D);
    float4 s4[ROWS_PER_WARP];
    if (w == gwarp && row_base == pre_row_base) {
      // First iteration (the only one for B=1): use the prefetched rows.
#pragma unroll
      for (int r = 0; r < ROWS_PER_WARP; ++r) s4[r] = s_pre[r];
    } else {
      const float4* base_srow =
          (const float4*)(ssm_state + row_base + (long)v0 * D);
#pragma unroll
      for (int r = 0; r < ROWS_PER_WARP; ++r)
        s4[r] = base_srow[r * (D / 4) + lane];
    }
    // Issue the gate-partial loads early (10 concurrent warp-wide loads) so
    // they overlap the qk-norm compute below.
    const float* base_b = ba_part + ((long)h * Beff + b) * GEMV_NSPLIT;
    const float* base_a = ba_part + ((long)(HV + h) * Beff + b) * GEMV_NSPLIT;
    float b0 = base_b[lane + 0];
    float a0v = base_a[lane + 0];
    float b1 = base_b[lane + 32];
    float a1v = base_a[lane + 32];
    float b2 = base_b[lane + 64];
    float a2v = base_a[lane + 64];
    float b3 = base_b[lane + 96];
    float a3v = base_a[lane + 96];
    float b4 = base_b[lane + 128];
    float a4v = base_a[lane + 128];
    float qraw[4], kraw[4];
    float qss = 0.f, kss = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      qraw[i] = __half2float(qb[k0 + i]);
      kraw[i] = __half2float(kb[k0 + i]);
      qss += qraw[i] * qraw[i];
      kss += kraw[i] * kraw[i];
    }
    qss = warp_reduce(qss);
    kss = warp_reduce(kss);
    qss = __shfl_sync(0xffffffff, qss, 0);
    kss = __shfl_sync(0xffffffff, kss, 0);
    float qn = rsqrtf(qss + 1e-6f), kn = rsqrtf(kss + 1e-6f);
    float qh[4], kh[4], QKp = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      qh[i] = qraw[i] * qn;
      kh[i] = kraw[i] * kn;
      QKp += qh[i] * kh[i];
    }
    QKp = warp_reduce(QKp);
    float QK = __shfl_sync(0xffffffff, QKp, 0);
    // gate g,beta reduced from the split-major partials (values now arrived)
    float accb = ((b0 + b1) + (b2 + b3)) + b4;
    float acca = ((a0v + a1v) + (a2v + a3v)) + a4v;
    accb = warp_reduce(accb);
    acca = warp_reduce(acca);
    accb = __shfl_sync(0xffffffff, accb, 0);
    acca = __shfl_sync(0xffffffff, acca, 0);
    // The f16 round-trip of the two gate sums is load-bearing, not a leftover:
    // the composable path materializes `ba = (hidden @ w_ba)` as a f16 tensor
    // and only then widens it for the gates, so the values it feeds
    // sigmoid/softplus are f16-rounded. Keeping the fp32 accumulator here
    // would make this kernel *more* precise than the operation it implements
    // and move the gates off the reference by up to one f16 ulp -- amplified
    // by exp() in the decay gate. Track the composable path's `ba` dtype, not
    // the accumulator's.
    float beta = sigmoidf(__half2float(__float2half_rn(accb)));
    float xg = __half2float(__float2half_rn(acca)) + __half2float(dt_bias[h]);
    float g = __expf(-__expf(A_log[h]) * softplusf(xg));
#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
      int v = v0 + r;
      float s[4] = {s4[r].x, s4[r].y, s4[r].z, s4[r].w};
      float vv = __half2float(co[2 * H_Q * D + h * D + v]);
      float kSp = 0.f, qSp = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        kSp += kh[i] * s[i];
        qSp += qh[i] * s[i];
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        kSp += __shfl_down_sync(0xffffffff, kSp, o);
        qSp += __shfl_down_sync(0xffffffff, qSp, o);
      }
      float kS = __shfl_sync(0xffffffff, kSp, 0);
      float qS = __shfl_sync(0xffffffff, qSp, 0);
      float old_v = g * kS;
      float delta = beta * (vv - old_v);
      float out_v = scale * (g * qS + delta * QK);
      float4* Sorow = (float4*)(ssm_out + row_base + (long)v * D);
      float4 o4;
      o4.x = g * s[0] + kh[0] * delta;
      o4.y = g * s[1] + kh[1] * delta;
      o4.z = g * s[2] + kh[2] * delta;
      o4.w = g * s[3] + kh[3] * delta;
      Sorow[lane] = o4;
      if (lane == 0)
        output[((long)b * HV + h) * D + v] = __float2half_rn(out_v);
    }
  }
}

}  // namespace flashinfer::sm70::gdn
