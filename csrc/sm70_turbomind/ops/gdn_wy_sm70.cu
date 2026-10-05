// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Snapshot-free chain-WY GDN verify step for SM70 (27B TP4 shapes: H=4, HV=12,
// K=V=128, T=8).
//
// Per step and per value head:   S_t = e^{G_t} S0 + sum_{i<=t} e^{G_t-G_i} u_i
// k_i^T
//   u_t = beta_t (v_t - e^{G_t} S0 k_t - sum_{i<t} e^{G_t-G_i} (k_i.k_t) u_i)
//   o_t = e^{G_t} S0 q_t + sum_{i<=t} e^{G_t-G_i} (k_i.q_t) u_i
// Rows of S are independent given the shared 8x8 coefficients.  The persistent
// state slot holds the committed start state; at the start of a step the
// previous step's accepted prefix (a tokens) is committed from the (u, k, G)
// that step saved:
//   S0 = e^{Gp_a} S_prev + sum_{i<a} e^{Gp_a-Gp_i} up_i kp_i^T
// One state read and one state write per step; FP32 throughout.
#include <torch/types.h>
#include <torch/library.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

namespace {
constexpr int T = 8, H = 4, HV = 12, K = 128, V = 128,
              KS = 132;  // KS: padded smem stride

// Transposed butterfly over the low log2(LANES) lane bits: reduce N partial
// sums held by each lane. Afterwards the lane bit pattern selects which
// complete sums it holds.
template <int N, int LANES>
__device__ __forceinline__ void transpose_reduce(float (&x)[N], int lane) {
  int count = N;
#pragma unroll
  for (int off = LANES / 2; off >= 1; off >>= 1) {
    if (count > 1) {
      const int half = count / 2;
      const bool up = lane & off;
#pragma unroll
      for (int j = 0; j < N / 2; ++j) {
        if (j < half) {
          const float send = up ? x[j] : x[j + half];
          const float keep = up ? x[j + half] : x[j];
          x[j] = keep + __shfl_xor_sync(0xffffffffu, send, off);
        }
      }
      count = half;
    } else {
      x[0] += __shfl_xor_sync(0xffffffffu, x[0], off);
    }
  }
}

struct Smem {
  float kn[T * KS], qn[T * KS], pk[T * KS];
  float sq[4][2 * T], G[T];
  float C[T][T], M[T][T], eG[T], bt[T], wp[T], eGa;
};

// Shared per-head prologue (all 128 threads): normalize k/q, previous k,
// gating, 8x8 coefficient matrices.
__device__ __forceinline__ void prologue(
    Smem& sm, const half* __restrict__ q, const half* __restrict__ k,
    const half* __restrict__ a_in, const half* __restrict__ b_in,
    const float* A_log, const void* dt_bias, bool bias_fp32, int q_stride,
    int k_stride, int ab_stride, int nt, float scale,
    const float* __restrict__ prev_k, const float* __restrict__ prev_G, int a,
    int hv, float* __restrict__ new_k, float* __restrict__ new_G) {
  const int ih = hv / (HV / H), tid = threadIdx.x, warp = tid >> 5,
            lane = tid & 31;
  float gl = 0.f, bl = 0.f, pg = 0.f, pga = 0.f;
  if (tid < T) {
    if (tid < nt) {
      const float z =
          __half2float(a_in[tid * ab_stride + hv]) +
          (bias_fp32
               ? reinterpret_cast<const float*>(dt_bias)[hv]
               : __half2float(reinterpret_cast<const half*>(dt_bias)[hv]));
      const float softplus = z > 20.f ? z : log1pf(__expf(z));
      gl = -__expf(A_log[hv]) * softplus;
      bl = 1.f / (1.f + __expf(-__half2float(b_in[tid * ab_stride + hv])));
    }
    if (a > 0) {
      pg = prev_G[tid * HV + hv];
      pga = prev_G[(a - 1) * HV + hv];
    }
  }
  float kr[T], qr[T], ss[2 * T];
#pragma unroll
  for (int t = 0; t < T; ++t) {
    kr[t] = (t < nt ? __half2float(k[t * k_stride + ih * K + tid]) : 0.f);
    qr[t] = (t < nt ? __half2float(q[t * q_stride + ih * K + tid]) : 0.f);
    ss[t] = kr[t] * kr[t];
    ss[T + t] = qr[t] * qr[t];
    sm.pk[t * KS + tid] = t < a ? prev_k[(t * H + ih) * K + tid] : 0.f;
  }
  transpose_reduce<2 * T, 32>(ss, lane);  // lane holds sum (lane>>1)&15
  if ((lane & 1) == 0) sm.sq[warp][(lane >> 1) & 15] = ss[0];
  if (warp == 0) {
    float G = gl;  // inclusive scan over T tokens
#pragma unroll
    for (int off = 1; off < T; off <<= 1) {
      const float y = __shfl_up_sync(0xffffffffu, G, off);
      if (lane >= off) G += y;
    }
    if (lane < T) {
      sm.eG[lane] = __expf(G);
      sm.bt[lane] = bl;
      sm.G[lane] = G;
      sm.wp[lane] = (a > 0 && lane < a) ? __expf(pga - pg) : 0.f;
      if (lane == 0) sm.eGa = a > 0 ? __expf(pga) : 1.f;
    }
  }
  __syncthreads();
#pragma unroll
  for (int t = 0; t < T; ++t) {
    const float nk =
        rsqrtf(sm.sq[0][t] + sm.sq[1][t] + sm.sq[2][t] + sm.sq[3][t] + 1e-6f);
    const float nq = rsqrtf(sm.sq[0][T + t] + sm.sq[1][T + t] +
                            sm.sq[2][T + t] + sm.sq[3][T + t] + 1e-6f);
    sm.kn[t * KS + tid] = kr[t] * nk;
    sm.qn[t * KS + tid] = qr[t] * nq * scale;
  }
  __syncthreads();
  {  // warp w: which = w>>1 (KK / KQ), i in [(w&1)*4, +4), t in [0,T)
    const int which = warp >> 1, i0 = (warp & 1) * 4;
    const float* yb = which ? sm.qn : sm.kn;
    float4 xi[4], yt[T];
#pragma unroll
    for (int ii = 0; ii < 4; ++ii)
      xi[ii] = reinterpret_cast<const float4*>(sm.kn + (i0 + ii) * KS)[lane];
#pragma unroll
    for (int t = 0; t < T; ++t)
      yt[t] = reinterpret_cast<const float4*>(yb + t * KS)[lane];
    float p[32];
#pragma unroll
    for (int ii = 0; ii < 4; ++ii)
#pragma unroll
      for (int t = 0; t < T; ++t)
        p[ii * T + t] = xi[ii].x * yt[t].x + xi[ii].y * yt[t].y +
                        xi[ii].z * yt[t].z + xi[ii].w * yt[t].w;
    transpose_reduce<32, 32>(p, lane);  // lane holds dot index lane
    const int i = i0 + lane / T, t = lane % T;
    const float d = __expf(sm.G[t] - sm.G[i]);
    if (which == 0)
      sm.C[t][i] = i < t ? sm.bt[t] * d * p[0] : 0.f;
    else
      sm.M[t][i] = i <= t ? d * p[0] : 0.f;
  }
  if (blockIdx.x ==
      0) {  // save this step's normalized k and G for the next commit
    if (hv % (HV / H) == 0)
      for (int t = 0; t < T; ++t)
        new_k[(t * H + ih) * K + tid] = sm.kn[t * KS + tid];
    if (tid < T) new_G[tid * HV + hv] = sm.G[tid];
  }
  __syncthreads();
}

// LPR lanes per row (32: one row per warp-pass; 8: four rows per warp-pass),
// ITER passes.
template <int LPR, int ITER>
__global__ void __launch_bounds__(128)
    gdn_wy_kernel(const half* __restrict__ q, const half* __restrict__ k,
                  const half* __restrict__ v, const half* __restrict__ a_in,
                  const half* __restrict__ b_in, const float* A_log,
                  const void* dt_bias, bool bias_fp32, float scale,
                  const int* indices, int index_stride,
                  const int* factor_indices, int factor_stride,
                  int factor_slots, int* pending, const int* cu_seqlens,
                  int q_stride, int k_stride, int v_stride, int ab_stride,
                  int out_stride, int n_slots, int64_t state_stride,
                  int64_t u_stride, int64_t k_cache_stride, int64_t G_stride,
                  float* __restrict__ state, const float* __restrict__ prev_u,
                  const float* __restrict__ prev_k,
                  const float* __restrict__ prev_G,
                  const int* __restrict__ n_accept_prev, half* __restrict__ o,
                  float* __restrict__ new_u, float* __restrict__ new_k,
                  float* __restrict__ new_G) {
  constexpr int GROUPS = 32 / LPR,
                CH = 32 / LPR;  // rows per pass, float4 chunks per lane
  constexpr int ROWS_PER_WARP = GROUPS * ITER;
  __shared__ Smem sm;
  const int seq = blockIdx.z;
  const int slot = indices[seq * index_stride];
  const int bos = cu_seqlens[seq], nt = cu_seqlens[seq + 1] - bos;
  if (slot < 0 || slot >= n_slots || nt <= 0 || nt > T) return;
  int factor_slot = -1;
  if (factor_stride < 0) {
    const bool* mask = reinterpret_cast<const bool*>(factor_indices);
    int ordinal = 0;
    for (int req = 0; req < -factor_stride; ++req) {
      if (mask[req]) {
        if (ordinal == seq) {
          factor_slot = req;
          break;
        }
        ++ordinal;
      }
    }
  } else {
    factor_slot = factor_indices[seq * factor_stride];
  }
  if (factor_slot < 0 || factor_slot >= factor_slots) return;
  if (pending && blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
    pending[factor_slot] = slot;
  q += static_cast<int64_t>(bos) * q_stride;
  k += static_cast<int64_t>(bos) * k_stride;
  v += static_cast<int64_t>(bos) * v_stride;
  a_in += static_cast<int64_t>(bos) * ab_stride;
  b_in += static_cast<int64_t>(bos) * ab_stride;
  o += static_cast<int64_t>(bos) * out_stride;
  state += static_cast<int64_t>(slot) * state_stride;
  prev_u += static_cast<int64_t>(factor_slot) * u_stride;
  new_u += static_cast<int64_t>(factor_slot) * u_stride;
  prev_k += static_cast<int64_t>(factor_slot) * k_cache_stride;
  new_k += static_cast<int64_t>(factor_slot) * k_cache_stride;
  prev_G += static_cast<int64_t>(factor_slot) * G_stride;
  new_G += static_cast<int64_t>(factor_slot) * G_stride;
  const int hv = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int grp = lane / LPR, gl = lane % LPR, base = grp * LPR;
  const int row0 = (blockIdx.x * 4 + warp) * ROWS_PER_WARP;
  const int a = min(T, max(0, n_accept_prev[seq]));

  // issue all global loads for owned rows before the prologue
  float4 s[ITER][CH];
  float pu[ITER], vvr[ITER];
#pragma unroll
  for (int it = 0; it < ITER; ++it) {
    const int vr = row0 + it * GROUPS + grp;
#pragma unroll
    for (int j = 0; j < CH; ++j)
      s[it][j] = reinterpret_cast<const float4*>(state + ((size_t)hv * V + vr) *
                                                             K)[gl + LPR * j];
    pu[it] = (gl < T && gl < a) ? prev_u[(gl * HV + hv) * V + vr] : 0.f;
    vvr[it] = gl < nt ? __half2float(v[gl * v_stride + hv * V + vr]) : 0.f;
  }
  prologue(sm, q, k, a_in, b_in, A_log, dt_bias, bias_fp32, q_stride, k_stride,
           ab_stride, nt, scale, prev_k, prev_G, a, hv, new_k, new_G);

#pragma unroll
  for (int it = 0; it < ITER; ++it) {
    const int vr = row0 + it * GROUPS + grp;
    float4 x[CH];
#pragma unroll
    for (int j = 0; j < CH; ++j) x[j] = s[it][j];
    if (a > 0) {  // commit previous step's accepted prefix
#pragma unroll
      for (int j = 0; j < CH; ++j) {
        x[j].x *= sm.eGa;
        x[j].y *= sm.eGa;
        x[j].z *= sm.eGa;
        x[j].w *= sm.eGa;
      }
      for (int i = 0; i < a; ++i) {
        const float c = sm.wp[i] * __shfl_sync(0xffffffffu, pu[it], base + i);
#pragma unroll
        for (int j = 0; j < CH; ++j) {
          const float4 kk =
              reinterpret_cast<const float4*>(sm.pk + i * KS)[gl + LPR * j];
          x[j].x = fmaf(c, kk.x, x[j].x);
          x[j].y = fmaf(c, kk.y, x[j].y);
          x[j].z = fmaf(c, kk.z, x[j].z);
          x[j].w = fmaf(c, kk.w, x[j].w);
        }
      }
#pragma unroll
      for (int j = 0; j < CH; ++j)
        reinterpret_cast<float4*>(state + ((size_t)hv * V + vr) *
                                              K)[gl + LPR * j] = x[j];
    }
    float d[2 * T];
#pragma unroll
    for (int t = 0; t < 2 * T; ++t) d[t] = 0.f;
#pragma unroll
    for (int j = 0; j < CH; ++j) {
#pragma unroll
      for (int t = 0; t < T; ++t) {
        const float4 kk =
            reinterpret_cast<const float4*>(sm.kn + t * KS)[gl + LPR * j];
        const float4 qq =
            reinterpret_cast<const float4*>(sm.qn + t * KS)[gl + LPR * j];
        d[t] += x[j].x * kk.x + x[j].y * kk.y + x[j].z * kk.z + x[j].w * kk.w;
        d[T + t] +=
            x[j].x * qq.x + x[j].y * qq.y + x[j].z * qq.z + x[j].w * qq.w;
      }
    }
    transpose_reduce<2 * T, LPR>(d, lane);
    // LPR=32: lane holds sum (lane>>1)&15 in d[0].  LPR=16: sums 2*... ; LPR=8:
    // lane holds sums 2*gl, 2*gl+1 in d[0], d[1].  Generic: sums held =
    // 2*T/LPR*... handled below.
    constexpr int HELD = (2 * T >= LPR) ? (2 * T / LPR) : 1;
    float r[HELD];
#pragma unroll
    for (int h = 0; h < HELD; ++h) r[h] = d[h];
#pragma unroll
    for (int jj = 0; jj < 2 * T; ++jj) {
      if (LPR == 32)
        d[jj] = __shfl_sync(0xffffffffu, r[0], 2 * jj);
      else
        d[jj] = __shfl_sync(0xffffffffu, r[jj % HELD], base + jj / HELD);
    }
    float u[T], my_u = 0.f, my_o = 0.f;
#pragma unroll
    for (int t = 0; t < T; ++t) {
      float acc = sm.bt[t] * (__shfl_sync(0xffffffffu, vvr[it], base + t) -
                              sm.eG[t] * d[t]);
#pragma unroll
      for (int i = 0; i < t; ++i) acc = fmaf(-sm.C[t][i], u[i], acc);
      u[t] = acc;
      float output_value = sm.eG[t] * d[T + t];
#pragma unroll
      for (int i = 0; i <= t; ++i)
        output_value = fmaf(sm.M[t][i], u[i], output_value);
      if (gl == t) {
        my_u = acc;
        my_o = output_value;
      }
    }
    if (gl < nt) {
      o[gl * out_stride + hv * V + vr] = __float2half_rn(my_o);
      new_u[(gl * HV + hv) * V + vr] = my_u;
    }
  }
}

// Materialize an arbitrary accepted prefix. This kernel never changes metadata:
// all CTAs must see the same source bank until the launch has completed.
__global__ void __launch_bounds__(128)
    commit_kernel(float* state, int64_t state_stride, int slots, const float* u,
                  int64_t u_stride, const float* k, int64_t k_stride,
                  const float* G, int64_t G_stride, const int* src,
                  const int* dst, const int* accepted, int src_stride,
                  int dst_stride, const int* factors, int factor_stride,
                  int factor_slots) {
  const int seq = blockIdx.z;
  const int si = src[seq * src_stride], di = dst[seq * dst_stride];
  const int fi = factors[seq * factor_stride];
  if (si < 0 || si >= slots || di < 0 || di >= slots || fi < 0 ||
      fi >= factor_slots)
    return;
  const int a = min(T, max(0, accepted[seq]));
  if (a == 0 && si == di) return;
  const int hv = blockIdx.y, ih = hv / (HV / H);
  // Four rows per warp, four adjacent float4 columns per lane.
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int vr = blockIdx.x * 16 + warp * 4 + lane / 8;
  const float ga = a > 0 ? G[fi * G_stride + (a - 1) * HV + hv] : 0.f;
  const float decay = __expf(ga);
#pragma unroll
  for (int chunk = 0; chunk < 4; ++chunk) {
    const int kc = ((lane & 7) + chunk * 8) * 4;
    float4 x = *reinterpret_cast<const float4*>(state + si * state_stride +
                                                (hv * V + vr) * K + kc);
    x.x *= decay;
    x.y *= decay;
    x.z *= decay;
    x.w *= decay;
    for (int i = 0; i < a; ++i) {
      const float c = __expf(ga - G[fi * G_stride + i * HV + hv]) *
                      u[fi * u_stride + (i * HV + hv) * V + vr];
      const float4 kk = *reinterpret_cast<const float4*>(k + fi * k_stride +
                                                         (i * H + ih) * K + kc);
      x.x = fmaf(c, kk.x, x.x);
      x.y = fmaf(c, kk.y, x.y);
      x.z = fmaf(c, kk.z, x.z);
      x.w = fmaf(c, kk.w, x.w);
    }
    *reinterpret_cast<float4*>(state + di * state_stride + (hv * V + vr) * K +
                               kc) = x;
  }
}

void check_tensor(const torch::Tensor& t, const torch::Tensor& reference,
                  at::ScalarType dtype) {
  TORCH_CHECK(
      t.is_cuda() && t.device() == reference.device() &&
          t.scalar_type() == dtype,
      "WY tensors must share a CUDA device and have the declared dtype");
}
void check_state(const torch::Tensor& state) {
  TORCH_CHECK(state.is_cuda() && state.scalar_type() == at::kFloat &&
                  state.dim() == 4 && state.size(1) == HV &&
                  state.size(2) == V && state.size(3) == K &&
                  state.stride(3) == 1 && state.stride(2) == K &&
                  state.stride(1) == V * K && state.stride(0) % 4 == 0,
              "WY requires FP32 [slots,12,128,128] state with contiguous inner "
              "dimensions");
}
void check_cache(const torch::Tensor& u, const torch::Tensor& k,
                 const torch::Tensor& G, const torch::Tensor& state) {
  for (const auto& t : {u, k, G}) check_tensor(t, state, at::kFloat);
  TORCH_CHECK(u.dim() == 4 && u.size(0) > 0 && u.size(1) == T &&
                  u.size(2) == HV && u.size(3) == V && u.stride(1) == HV * V &&
                  u.stride(2) == V && u.stride(3) == 1 && k.dim() == 4 &&
                  k.size(0) == u.size(0) && k.size(1) == T && k.size(2) == H &&
                  k.size(3) == K && k.stride(1) == H * K && k.stride(2) == K &&
                  k.stride(3) == 1 && G.dim() == 3 && G.size(0) == u.size(0) &&
                  G.size(1) == T && G.size(2) == HV && G.stride(1) == HV &&
                  G.stride(2) == 1,
              "invalid WY factor-cache dimensions or strides");
}
int check_indices(const torch::Tensor& ids, const torch::Tensor& state) {
  check_tensor(ids, state, at::kInt);
  TORCH_CHECK(
      (ids.dim() == 1 || ids.dim() == 2) &&
          (ids.stride(ids.dim() - 1) == 1 || ids.stride(ids.dim() - 1) == 0),
      "state indices must be an int32 vector or table with contiguous or "
      "repeated columns");
  return ids.stride(0);
}
void wy_verify(torch::Tensor out, torch::Tensor state, torch::Tensor new_u,
               torch::Tensor new_k, torch::Tensor new_G, torch::Tensor q,
               torch::Tensor k, torch::Tensor v, torch::Tensor a,
               torch::Tensor b, torch::Tensor A_log, torch::Tensor dt_bias,
               torch::Tensor indices, torch::Tensor factor_indices,
               std::optional<torch::Tensor> pending, torch::Tensor cu_seqlens,
               torch::Tensor accepted, torch::Tensor prev_u,
               torch::Tensor prev_k, torch::Tensor prev_G, double scale,
               int64_t variant) {
  const c10::cuda::CUDAGuard guard(state.device());
  check_state(state);
  check_cache(prev_u, prev_k, prev_G, state);
  check_cache(new_u, new_k, new_G, state);
  TORCH_CHECK(new_u.size(0) == prev_u.size(0) &&
                  new_u.stride(0) == prev_u.stride(0) &&
                  new_k.stride(0) == prev_k.stride(0) &&
                  new_G.stride(0) == prev_G.stride(0),
              "WY banks must have identical strides");
  for (const auto& t : {q, k, v, a, b, out}) {
    check_tensor(t, state, at::kHalf);
    TORCH_CHECK(t.dim() == 2 && t.stride(1) == 1 && t.size(0) == q.size(0),
                "WY token inputs must be row-major matrices");
  }
  TORCH_CHECK(q.size(1) == H * K && k.size(1) == H * K && v.size(1) == HV * V &&
                  out.size(1) == HV * V && a.size(1) == HV && b.size(1) == HV &&
                  a.stride(0) == b.stride(0),
              "invalid WY projection dimensions");
  for (const auto& t : {A_log, dt_bias}) {
    TORCH_CHECK(t.scalar_type() == at::kFloat ||
                    (t.is_same(dt_bias) && t.scalar_type() == at::kHalf),
                "unsupported WY gate dtype");
    check_tensor(t, state, t.scalar_type());
    TORCH_CHECK(t.numel() == HV && t.is_contiguous(),
                "WY gate parameters must be FP32 vectors");
  }
  const int is = check_indices(indices, state), n = indices.size(0);
  int fs;
  if (factor_indices.scalar_type() == at::kBool) {
    check_tensor(factor_indices, state, at::kBool);
    TORCH_CHECK(factor_indices.dim() == 1 && factor_indices.is_contiguous(),
                "invalid WY request mask");
    fs = -factor_indices.numel();
  } else {
    fs = check_indices(factor_indices, state);
    TORCH_CHECK(factor_indices.size(0) == n, "inconsistent WY factor count");
  }
  if (pending) {
    check_tensor(*pending, state, at::kInt);
    TORCH_CHECK(pending->dim() == 1 && pending->numel() == prev_u.size(0) &&
                    pending->is_contiguous(),
                "invalid WY pending-state vector");
  }
  for (const auto& t : {cu_seqlens, accepted}) {
    check_tensor(t, state, at::kInt);
    TORCH_CHECK(t.dim() == 1 && t.is_contiguous(),
                "WY sequence metadata must be int32 vectors");
  }
  TORCH_CHECK(cu_seqlens.numel() == n + 1 && accepted.numel() == n,
              "inconsistent WY sequence counts");
  TORCH_CHECK(variant >= 0 && variant <= 4, "invalid WY row variant");
  if (n == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream();
#define LAUNCH(LPR, IT)                                                      \
  gdn_wy_kernel<LPR, IT>                                                     \
      <<<dim3(V / (4 * (32 / LPR) * IT), HV, n), 128, 0, stream>>>(          \
          (const half*)q.data_ptr(), (const half*)k.data_ptr(),              \
          (const half*)v.data_ptr(), (const half*)a.data_ptr(),              \
          (const half*)b.data_ptr(), A_log.data_ptr<float>(),                \
          dt_bias.data_ptr(), dt_bias.scalar_type() == at::kFloat,           \
          (float)scale, indices.data_ptr<int>(), is,                         \
          reinterpret_cast<const int*>(factor_indices.data_ptr()), fs,       \
          prev_u.size(0), pending ? pending->data_ptr<int>() : nullptr,      \
          cu_seqlens.data_ptr<int>(), q.stride(0), k.stride(0), v.stride(0), \
          a.stride(0), out.stride(0), state.size(0), state.stride(0),        \
          prev_u.stride(0), prev_k.stride(0), prev_G.stride(0),              \
          state.data_ptr<float>(), prev_u.data_ptr<float>(),                 \
          prev_k.data_ptr<float>(), prev_G.data_ptr<float>(),                \
          accepted.data_ptr<int>(), (half*)out.data_ptr(),                   \
          new_u.data_ptr<float>(), new_k.data_ptr<float>(),                  \
          new_G.data_ptr<float>())
  switch (variant) {
    case 0:
      LAUNCH(32, 1);
      break;
    case 1:
      LAUNCH(32, 2);
      break;
    case 2:
      LAUNCH(16, 1);
      break;
    case 3:
      LAUNCH(8, 1);
      break;
    case 4:
      LAUNCH(8, 2);
      break;
  }
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void wy_commit(torch::Tensor state, torch::Tensor u, torch::Tensor k,
               torch::Tensor G, torch::Tensor src, torch::Tensor dst,
               torch::Tensor accepted, torch::Tensor factor_indices) {
  const c10::cuda::CUDAGuard guard(state.device());
  check_state(state);
  check_cache(u, k, G, state);
  const int ss = check_indices(src, state), ds = check_indices(dst, state),
            fs = check_indices(factor_indices, state), n = src.size(0);
  check_tensor(accepted, state, at::kInt);
  TORCH_CHECK(dst.size(0) == n && factor_indices.size(0) == n &&
                  accepted.dim() == 1 && accepted.numel() == n &&
                  accepted.is_contiguous(),
              "inconsistent WY commit sequence counts");
  if (n == 0) return;
  commit_kernel<<<dim3(V / 16, HV, n), 128, 0,
                  at::cuda::getCurrentCUDAStream()>>>(
      state.data_ptr<float>(), state.stride(0), state.size(0),
      u.data_ptr<float>(), u.stride(0), k.data_ptr<float>(), k.stride(0),
      G.data_ptr<float>(), G.stride(0), src.data_ptr<int>(),
      dst.data_ptr<int>(), accepted.data_ptr<int>(), ss, ds,
      factor_indices.data_ptr<int>(), fs, u.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Descriptor columns: state address/stride, u/k/G addresses, pending address,
// block-table group, physical slot count, factor capacity. All strides are in
// FP32 elements; factor buffers are contiguous [request,T,...].
__global__ void __launch_bounds__(128)
    group_commit_kernel(const int64_t* descriptors, const int64_t* block_tables,
                        int64_t table_stride, const int* accepted,
                        const int* scheduled, const int* computed,
                        const int* draft, int block_size,
                        const int* idx_mapping, bool mrv2) {
  const int req = blockIdx.y, layer = blockIdx.z;
  const int64_t* d = descriptors + layer * 9;
  auto state = reinterpret_cast<float*>(d[0]);
  const int64_t state_stride = d[1];
  const auto u = reinterpret_cast<const float*>(d[2]);
  const auto k = reinterpret_cast<const float*>(d[3]);
  const auto G = reinterpret_cast<const float*>(d[4]);
  const auto pending = reinterpret_cast<const int*>(d[5]);
  if (req >= d[8]) return;
  const int si = pending[req];
  if (si < 0 || si >= d[7]) return;
  const int ri = mrv2 ? idx_mapping[req] : req;
  if (ri < 0) return;
  const int a = min(T, max(0, accepted[ri]));
  const int running =
      mrv2 ? computed[ri] - a + 1 : computed[req] + scheduled[req] - draft[req];
  const int end = running + a - 1;
  const int boundary = (end / block_size) * block_size;
  const bool save = a > 0 && boundary >= running;
  int di = -1, prefix = 0;
  if (save) {
    const auto table = reinterpret_cast<const int*>(block_tables[d[6]]);
    di = table[req * table_stride + boundary / block_size - 1];
    prefix = boundary - running + 1;
  }
  const int hv = blockIdx.x / (V / 16), ih = hv / (HV / H);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int vr = (blockIdx.x % (V / 16)) * 16 + warp * 4 + lane / 8;
  const int64_t ub = static_cast<int64_t>(req) * T * HV * V;
  const int64_t kb = static_cast<int64_t>(req) * T * H * K;
  const int64_t gb = static_cast<int64_t>(req) * T * HV;
  const float ga = a > 0 ? G[gb + (a - 1) * HV + hv] : 0.f;
  const float gp = prefix > 0 ? G[gb + (prefix - 1) * HV + hv] : 0.f;
#pragma unroll
  for (int chunk = 0; chunk < 4; ++chunk) {
    const int kc = ((lane & 7) + chunk * 8) * 4;
    const int64_t offset = (hv * V + vr) * K + kc;
    const float4 base =
        *reinterpret_cast<const float4*>(state + si * state_stride + offset);
    float4 x = base, xp = base;
    const float decay = __expf(ga), dp = __expf(gp);
    x.x *= decay;
    x.y *= decay;
    x.z *= decay;
    x.w *= decay;
    xp.x *= dp;
    xp.y *= dp;
    xp.z *= dp;
    xp.w *= dp;
    for (int i = 0; i < a; ++i) {
      const float gi = G[gb + i * HV + hv], ui = u[ub + (i * HV + hv) * V + vr];
      const float4 kk =
          *reinterpret_cast<const float4*>(k + kb + (i * H + ih) * K + kc);
      const float c = __expf(ga - gi) * ui;
      x.x = fmaf(c, kk.x, x.x);
      x.y = fmaf(c, kk.y, x.y);
      x.z = fmaf(c, kk.z, x.z);
      x.w = fmaf(c, kk.w, x.w);
      if (save && i < prefix && di != si) {
        const float cp = __expf(gp - gi) * ui;
        xp.x = fmaf(cp, kk.x, xp.x);
        xp.y = fmaf(cp, kk.y, xp.y);
        xp.z = fmaf(cp, kk.z, xp.z);
        xp.w = fmaf(cp, kk.w, xp.w);
      }
    }
    if (save && di >= 0 && di < d[7] && di != si)
      *reinterpret_cast<float4*>(state + di * state_stride + offset) = xp;
    *reinterpret_cast<float4*>(state + si * state_stride + offset) = x;
  }
}
__global__ void clear_pending_kernel(const int64_t* descriptors, int layers,
                                     int reqs) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= layers * reqs) return;
  const int layer = i / reqs, req = i % reqs;
  const int64_t* d = descriptors + layer * 9;
  if (req < d[8]) reinterpret_cast<int*>(d[5])[req] = -1;
}
void wy_group_commit(std::vector<torch::Tensor> states,
                     std::vector<torch::Tensor> pending,
                     torch::Tensor descriptors, torch::Tensor tables,
                     int64_t table_stride, torch::Tensor accepted,
                     torch::Tensor scheduled, torch::Tensor computed,
                     torch::Tensor draft, int64_t block_size,
                     bool clear_metadata) {
  TORCH_CHECK(!states.empty() && pending.size() == states.size(),
              "invalid WY layer group");
  const c10::cuda::CUDAGuard guard(states[0].device());
  for (const auto& state : states) check_state(state);
  for (const auto& t : {accepted, scheduled, computed, draft}) {
    check_tensor(t, states[0], at::kInt);
    TORCH_CHECK(
        t.dim() == 1 && t.is_contiguous() && t.numel() >= accepted.numel(),
        "invalid WY commit metadata");
  }
  for (const auto& t : {descriptors, tables}) {
    check_tensor(t, states[0], at::kLong);
    TORCH_CHECK(t.is_contiguous(), "WY pointer tables must be contiguous");
  }
  TORCH_CHECK(descriptors.dim() == 2 && descriptors.size(0) == states.size() &&
                  descriptors.size(1) == 9 && tables.dim() == 1 &&
                  tables.numel() > 0 && table_stride > 0 && block_size > 0,
              "invalid WY group layout");
  const int reqs = accepted.numel(), layers = states.size();
  for (const auto& t : pending) {
    check_tensor(t, states[0], at::kInt);
    TORCH_CHECK(t.is_contiguous() && t.numel() >= reqs,
                "invalid WY pending storage");
  }
  if (!reqs) return;
  auto stream = at::cuda::getCurrentCUDAStream();
  group_commit_kernel<<<dim3(HV * V / 16, reqs, layers), 128, 0, stream>>>(
      descriptors.data_ptr<int64_t>(), tables.data_ptr<int64_t>(), table_stride,
      accepted.data_ptr<int>(), scheduled.data_ptr<int>(),
      computed.data_ptr<int>(), draft.data_ptr<int>(), block_size, nullptr,
      false);
  if (clear_metadata)
    clear_pending_kernel<<<(layers * reqs + 127) / 128, 128, 0, stream>>>(
        descriptors.data_ptr<int64_t>(), layers, reqs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Publish accepted convolution history only after boundary-state copies have
// consumed the speculative window. One CTA owns a request/layer and its pending
// flag, so clearing that flag cannot race a reader in another CTA.
__global__ void finish_group_kernel(
    const int64_t* desc, const int64_t* conv_desc, const int64_t* tables,
    int64_t table_stride, const int* accepted, const int* scheduled,
    const int* computed, const int* draft, int block_size, int* next_counts,
    int* next_selectors, const int* idx_mapping, bool mrv2) {
  const int layer = blockIdx.x, req = blockIdx.y;
  const int64_t* d = desc + layer * 9;
  const int64_t* cd = conv_desc + layer * 3;
  if (req >= d[8]) return;
  auto pending = reinterpret_cast<int*>(d[5]);
  __shared__ int si_shared;
  if (threadIdx.x == 0) si_shared = pending[req];
  __syncthreads();
  const int si = si_shared;
  if (si < 0 || si >= d[7]) return;
  const int ri = mrv2 ? idx_mapping[req] : req;
  if (ri < 0) return;
  const int a = min(T, max(0, accepted[ri]));
  const int running =
      mrv2 ? computed[ri] - a + 1 : computed[req] + scheduled[req] - draft[req];
  const int boundary = ((running + a - 1) / block_size) * block_size;
  int offset = max(0, a - 1);
  if (a > 0 && boundary >= running) {
    const auto table = reinterpret_cast<const int*>(tables[d[6]]);
    const int di = table[req * table_stride + boundary / block_size - 1];
    if (di == si) offset = 0;  // generic postprocess already shifted this block
  }
  auto conv = reinterpret_cast<half*>(cd[0]) + si * cd[1];
  const int channels = cd[2];
  for (int channel = threadIdx.x; channel < channels; channel += blockDim.x) {
    const half h0 = conv[offset * channels + channel];
    const half h1 = conv[(offset + 1) * channels + channel];
    const half h2 = conv[(offset + 2) * channels + channel];
    conv[channel] = h0;
    conv[channels + channel] = h1;
    conv[2 * channels + channel] = h2;
  }
  if (threadIdx.x == 0) {
    pending[req] = -1;
    if (layer == 0) {
      next_counts[ri] = 1;
      next_selectors[ri] = 1;
    }
  }
}
void wy_finish_group(std::vector<torch::Tensor> pending,
                     std::vector<torch::Tensor> conv, torch::Tensor desc,
                     torch::Tensor conv_desc, torch::Tensor tables,
                     int64_t table_stride, torch::Tensor accepted,
                     torch::Tensor scheduled, torch::Tensor computed,
                     torch::Tensor draft, int64_t block_size,
                     torch::Tensor next_counts, torch::Tensor next_selectors) {
  TORCH_CHECK(!conv.empty() && pending.size() == conv.size(),
              "invalid WY finish group");
  const c10::cuda::CUDAGuard guard(conv[0].device());
  for (const auto& t : conv) {
    check_tensor(t, conv[0], at::kHalf);
    TORCH_CHECK(t.dim() == 3 && t.size(1) == 10 && t.size(2) == 2560 &&
                    t.stride(2) == 1 && t.stride(1) == 2560,
                "WY publication requires width-first FP16 convolution history");
  }
  for (const auto& t : pending) check_tensor(t, conv[0], at::kInt);
  for (const auto& t :
       {accepted, scheduled, computed, draft, next_counts, next_selectors}) {
    check_tensor(t, conv[0], at::kInt);
    TORCH_CHECK(
        t.dim() == 1 && t.is_contiguous() && t.numel() >= accepted.numel(),
        "invalid WY finish metadata");
  }
  for (const auto& t : {desc, conv_desc, tables}) {
    check_tensor(t, conv[0], at::kLong);
    TORCH_CHECK(t.is_contiguous(), "invalid WY finish pointer table");
  }
  TORCH_CHECK(desc.dim() == 2 && desc.size(0) == conv.size() &&
                  desc.size(1) == 9 && conv_desc.dim() == 2 &&
                  conv_desc.size(0) == conv.size() && conv_desc.size(1) == 3 &&
                  tables.dim() == 1 && tables.numel() > 0 && block_size > 0 &&
                  table_stride > 0,
              "invalid WY finish descriptors");
  if (!accepted.numel()) return;
  finish_group_kernel<<<dim3(conv.size(), accepted.numel()), 128, 0,
                        at::cuda::getCurrentCUDAStream()>>>(
      desc.data_ptr<int64_t>(), conv_desc.data_ptr<int64_t>(),
      tables.data_ptr<int64_t>(), table_stride, accepted.data_ptr<int>(),
      scheduled.data_ptr<int>(), computed.data_ptr<int>(),
      draft.data_ptr<int>(), block_size, next_counts.data_ptr<int>(),
      next_selectors.data_ptr<int>(), nullptr, false);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void wy_commit_group_v2(std::vector<torch::Tensor> states,
                        std::vector<torch::Tensor> pending, torch::Tensor desc,
                        torch::Tensor tables, int64_t table_stride,
                        torch::Tensor accepted, torch::Tensor computed,
                        torch::Tensor mapping, int64_t block_size) {
  TORCH_CHECK(!states.empty() && states.size() == pending.size(),
              "invalid WY MRV2 group");
  const c10::cuda::CUDAGuard guard(states[0].device());
  for (const auto& t : states) check_state(t);
  for (const auto& t : pending) check_tensor(t, states[0], at::kInt);
  for (const auto& t : {accepted, computed, mapping}) {
    check_tensor(t, states[0], at::kInt);
    TORCH_CHECK(t.dim() == 1 && t.is_contiguous(),
                "invalid WY MRV2 sequence metadata");
  }
  for (const auto& t : {desc, tables}) {
    check_tensor(t, states[0], at::kLong);
    TORCH_CHECK(t.is_contiguous(), "invalid WY MRV2 pointer table");
  }
  TORCH_CHECK(desc.dim() == 2 && desc.size(0) == states.size() &&
                  desc.size(1) == 9 && tables.dim() == 1 &&
                  tables.numel() > 0 && table_stride > 0 && block_size > 0 &&
                  computed.numel() == accepted.numel(),
              "invalid WY MRV2 layout");
  if (!mapping.numel()) return;
  group_commit_kernel<<<dim3(HV * V / 16, mapping.numel(), states.size()), 128,
                        0, at::cuda::getCurrentCUDAStream()>>>(
      desc.data_ptr<int64_t>(), tables.data_ptr<int64_t>(), table_stride,
      accepted.data_ptr<int>(), nullptr, computed.data_ptr<int>(), nullptr,
      block_size, mapping.data_ptr<int>(), true);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void wy_finish_group_v2(std::vector<torch::Tensor> pending,
                        std::vector<torch::Tensor> conv, torch::Tensor desc,
                        torch::Tensor conv_desc, torch::Tensor tables,
                        int64_t table_stride, torch::Tensor accepted_snapshot,
                        torch::Tensor computed, torch::Tensor mapping,
                        int64_t block_size, torch::Tensor next_counts) {
  TORCH_CHECK(!conv.empty() && conv.size() == pending.size(),
              "invalid WY MRV2 finish group");
  const c10::cuda::CUDAGuard guard(conv[0].device());
  for (const auto& t : conv) {
    check_tensor(t, conv[0], at::kHalf);
    TORCH_CHECK(t.dim() == 3 && t.size(1) == 10 && t.size(2) == 2560 &&
                    t.stride(2) == 1 && t.stride(1) == 2560,
                "WY requires width-first FP16 convolution history");
  }
  for (const auto& t : pending) check_tensor(t, conv[0], at::kInt);
  for (const auto& t : {accepted_snapshot, computed, mapping, next_counts}) {
    check_tensor(t, conv[0], at::kInt);
    TORCH_CHECK(t.dim() == 1 && t.is_contiguous(),
                "invalid WY MRV2 finish metadata");
  }
  for (const auto& t : {desc, conv_desc, tables}) {
    check_tensor(t, conv[0], at::kLong);
    TORCH_CHECK(t.is_contiguous(), "invalid WY MRV2 finish pointers");
  }
  TORCH_CHECK(desc.dim() == 2 && desc.size(0) == conv.size() &&
                  desc.size(1) == 9 && conv_desc.dim() == 2 &&
                  conv_desc.size(0) == conv.size() && conv_desc.size(1) == 3 &&
                  tables.dim() == 1 && tables.numel() > 0 && table_stride > 0 &&
                  block_size > 0 &&
                  computed.numel() == accepted_snapshot.numel() &&
                  next_counts.numel() == computed.numel(),
              "invalid WY MRV2 finish layout");
  if (!mapping.numel()) return;
  finish_group_kernel<<<dim3(conv.size(), mapping.numel()), 128, 0,
                        at::cuda::getCurrentCUDAStream()>>>(
      desc.data_ptr<int64_t>(), conv_desc.data_ptr<int64_t>(),
      tables.data_ptr<int64_t>(), table_stride,
      accepted_snapshot.data_ptr<int>(), nullptr, computed.data_ptr<int>(),
      nullptr, block_size, next_counts.data_ptr<int>(),
      next_counts.data_ptr<int>(), mapping.data_ptr<int>(), true);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace
TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "gdn_wy_commit_group_v2_sm70(Tensor(a!)[] states, Tensor(b!)[] pending, "
      "Tensor desc, Tensor tables, int table_stride, Tensor accepted, Tensor "
      "computed, Tensor mapping, int block_size) -> ()");
  m.def(
      "gdn_wy_finish_group_v2_sm70(Tensor(a!)[] pending, Tensor(b!)[] conv, "
      "Tensor desc, Tensor conv_desc, Tensor tables, int table_stride, Tensor "
      "accepted_snapshot, Tensor computed, Tensor mapping, int block_size, "
      "Tensor(c!) next_counts) -> ()");
  m.def(
      "gdn_wy_finish_group_sm70(Tensor(a!)[] pending, Tensor(b!)[] conv, "
      "Tensor desc, Tensor conv_desc, Tensor tables, int table_stride, Tensor "
      "accepted, Tensor scheduled, Tensor computed, Tensor draft, int "
      "block_size, Tensor(c!) next_counts, Tensor(d!) next_selectors) -> ()");
  m.def(
      "gdn_wy_commit_group_sm70(Tensor(a!)[] states, Tensor(b!)[] pending, "
      "Tensor descriptors, Tensor tables, int table_stride, Tensor accepted, "
      "Tensor scheduled, Tensor computed, Tensor draft, int block_size, bool "
      "clear_metadata=True) -> ()");
  m.def(
      "gdn_wy_verify_sm70_out(Tensor(a!) out, Tensor(b!) state, Tensor(c!) "
      "new_u, Tensor(d!) new_k, Tensor(e!) new_G, Tensor q, Tensor k, Tensor "
      "v, Tensor a, Tensor b, Tensor A_log, Tensor dt_bias, Tensor indices, "
      "Tensor factor_indices, Tensor(f!)? pending, Tensor cu_seqlens, Tensor "
      "accepted, Tensor prev_u, Tensor prev_k, Tensor prev_G, float scale, int "
      "variant) -> ()");
  m.def(
      "gdn_wy_commit_sm70(Tensor(a!) state, Tensor u, Tensor k, Tensor G, "
      "Tensor src, Tensor dst, Tensor accepted, Tensor factor_indices) -> ()");
}
TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("gdn_wy_commit_group_v2_sm70", &wy_commit_group_v2);
  m.impl("gdn_wy_finish_group_v2_sm70", &wy_finish_group_v2);
  m.impl("gdn_wy_finish_group_sm70", &wy_finish_group);
  m.impl("gdn_wy_commit_group_sm70", &wy_group_commit);
  m.impl("gdn_wy_verify_sm70_out", &wy_verify);
  m.impl("gdn_wy_commit_sm70", &wy_commit);
}
