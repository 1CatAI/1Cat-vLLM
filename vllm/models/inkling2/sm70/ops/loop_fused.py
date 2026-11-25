# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Fused loop message/blend kernels for decode-size batches on SM70.

Replaces, for `T <= MAX_TOKENS` tokens, the per-visit chain

    attn_res (message) -> pair_rms_norm -> bmm(down) -> bmm(gain/forget)
    -> blend_rms_norm

by

* `all_messages`: ONE launch per loop pass. The messages of a visit depend
  only on the pass's `bank` and the layer's queries, never on the evolving
  residual state, so the messages of all core layers x 4 streams are mixed
  together (same Triton row kernel as `attn_res`, so bit-identical).
* `blend`: two CUDA kernels per blend (was ~5 launches, two of them cuBLAS
  GEMMs that use 1-3 CTAs at M = 1..8):
    A  (`pair_down`): RMS of dest and message, fp16 pair, down projection
       (rank 64) as a GEMV over the whole GPU -> fp16 hidden.
    B  (`gf_blend`): gain/forget projection (fp16 weights, fp32 accumulate),
       `(1 - tanh f) * dest + tanh g * msg` unfused in fp32, RMSNorm over the
       row and the weight scale, fp16 out. The RMS needs the whole row, so
       CTAs publish fp32 values and per-CTA partial sums of squares and the
       last CTA of a stream (atomic ticket, self-resetting counter) finishes
       the row in a fixed order (deterministic).

Precision matches the unfused path (fp16 pair and hidden, fp32 elsewhere);
only the summation order inside the GEMV/reductions differs.
"""

import os

import torch

from vllm.triton_utils import triton

from .attn_res import _attn_res_row_kernel

MAX_TOKENS = 16
RANK = 64

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define R_ 64

__device__ __forceinline__ float rsqrt_approx(float x) {
  float y;
  asm("rsqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ float ldf(const float* p) { return *p; }
__device__ __forceinline__ float ldf(const __half* p) { return __half2float(*p); }

__device__ __forceinline__ float2 h2f2(unsigned u) {
  return __half22float2(*reinterpret_cast<__half2*>(&u));
}

// ---------------------------------------------------------------- kernel A
// grid (16, T, S), block 128. CTA = (stream s, token t, 4 down rows); warp = row.
// K8 = 2D/8 = 576 uint4 per row = 18 per lane (D = 2304): D must be 2304.
#define K8_ 576
__global__ void __launch_bounds__(128) pair_down_kernel(
    const float* __restrict__ dest, long dest_stride_s,
    const float* __restrict__ msg,
    const __half* __restrict__ wd, long wd_stride_s,
    __half* __restrict__ hidden,
    int T, int D, float eps) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  __half* pair = reinterpret_cast<__half*>(smem_raw);  // 2D halves
  __shared__ float red[2][4];
  const int s = blockIdx.z, t = blockIdx.y, nb = blockIdx.x;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  // Prefetch this warp's weight row (independent of the activations).
  const int r = nb * 4 + warp;
  const uint4* w8 = reinterpret_cast<const uint4*>(wd + s * wd_stride_s + (long)r * 2 * D);
  uint4 wr[K8_ / 32];
#pragma unroll
  for (int i = 0; i < K8_ / 32; i++) wr[i] = __ldg(w8 + lane + 32 * i);
  const float4* d4 = reinterpret_cast<const float4*>(dest + s * dest_stride_s + (long)t * D);
  const float4* m4 = reinterpret_cast<const float4*>(msg + ((long)s * T + t) * D);
  const int n4 = D >> 2;
  float4 da[5], ma[5];
#pragma unroll
  for (int i = 0; i < 5; i++) {
    int q = tid + 128 * i;
    if (q < n4) { da[i] = d4[q]; ma[i] = m4[q]; }
    else { da[i] = make_float4(0.f, 0.f, 0.f, 0.f); ma[i] = da[i]; }
  }
  float sd = 0.f, sm = 0.f;
#pragma unroll
  for (int i = 0; i < 5; i++) {
    float4 a = da[i], b = ma[i];
    sd += __fmul_rn(a.x, a.x); sd += __fmul_rn(a.y, a.y); sd += __fmul_rn(a.z, a.z); sd += __fmul_rn(a.w, a.w);
    sm += __fmul_rn(b.x, b.x); sm += __fmul_rn(b.y, b.y); sm += __fmul_rn(b.z, b.z); sm += __fmul_rn(b.w, b.w);
  }
  sd = warp_sum(sd); sm = warp_sum(sm);
  if (lane == 0) { red[0][warp] = sd; red[1][warp] = sm; }
  __syncthreads();
  sd = (red[0][0] + red[0][1]) + (red[0][2] + red[0][3]);
  sm = (red[1][0] + red[1][1]) + (red[1][2] + red[1][3]);
  const float rd = rsqrt_approx(__fadd_rn(__fdiv_rn(sd, (float)D), eps));
  const float rm = rsqrt_approx(__fadd_rn(__fdiv_rn(sm, (float)D), eps));
  __half2* p2 = reinterpret_cast<__half2*>(pair);
#pragma unroll
  for (int i = 0; i < 5; i++) {
    int q = tid + 128 * i;
    if (q < n4) {
      float4 a = da[i], b = ma[i];
      p2[2 * q] = __floats2half2_rn(a.x * rd, a.y * rd);
      p2[2 * q + 1] = __floats2half2_rn(a.z * rd, a.w * rd);
      p2[(D >> 1) + 2 * q] = __floats2half2_rn(b.x * rm, b.y * rm);
      p2[(D >> 1) + 2 * q + 1] = __floats2half2_rn(b.z * rm, b.w * rm);
    }
  }
  __syncthreads();
  const uint4* pair8 = reinterpret_cast<const uint4*>(pair);
  float a0 = 0.f, a1 = 0.f;
#pragma unroll
  for (int i = 0; i < K8_ / 32; i++) {
    uint4 w = wr[i];
    uint4 p = pair8[lane + 32 * i];
    float2 w0 = h2f2(w.x), p0 = h2f2(p.x), w1 = h2f2(w.y), p1 = h2f2(p.y);
    float2 w2 = h2f2(w.z), p2_ = h2f2(p.z), w3 = h2f2(w.w), p3 = h2f2(p.w);
    a0 = fmaf(w0.x, p0.x, a0); a1 = fmaf(w0.y, p0.y, a1);
    a0 = fmaf(w1.x, p1.x, a0); a1 = fmaf(w1.y, p1.y, a1);
    a0 = fmaf(w2.x, p2_.x, a0); a1 = fmaf(w2.y, p2_.y, a1);
    a0 = fmaf(w3.x, p3.x, a0); a1 = fmaf(w3.y, p3.y, a1);
  }
  float acc = warp_sum(a0 + a1);
  if (lane == 0) hidden[((long)s * T + t) * R_ + r] = __float2half_rn(acc);
}

// ---------------------------------------------------------------- kernel B
// grid (D/NT, S), block NT. Thread = output column j of stream s, all TT tokens.
template <int NT, int TT, typename WT, int MODE>
__global__ void __launch_bounds__(NT) gf_blend_kernel(
    const float* __restrict__ dest, long dest_stride_s,
    const float* __restrict__ msg,
    const __half* __restrict__ hidden,
    const __half* __restrict__ wgf, long wgf_stride_s,
    const WT* __restrict__ nw, long nw_stride_s,
    __half* __restrict__ out,
    float* __restrict__ xs,
    float* __restrict__ partial,
    unsigned* __restrict__ counters,
    int T, int D, float eps) {
  __shared__ float hs[TT][R_];
  __shared__ float red[NT / 32][TT];
  __shared__ int is_last;
  const int s = blockIdx.y, cb = blockIdx.x, CB = gridDim.x;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int j = cb * NT + tid;

  float wg[R_], wf[R_];
  {
    const uint4* g8 = reinterpret_cast<const uint4*>(wgf + s * wgf_stride_s + (long)j * R_);
    const uint4* f8 = reinterpret_cast<const uint4*>(wgf + s * wgf_stride_s + (long)(D + j) * R_);
#pragma unroll
    for (int i = 0; i < R_ / 8; i++) {
      uint4 g = __ldg(g8 + i), f = __ldg(f8 + i);
      float2 v;
      v = h2f2(g.x); wg[8 * i + 0] = v.x; wg[8 * i + 1] = v.y;
      v = h2f2(g.y); wg[8 * i + 2] = v.x; wg[8 * i + 3] = v.y;
      v = h2f2(g.z); wg[8 * i + 4] = v.x; wg[8 * i + 5] = v.y;
      v = h2f2(g.w); wg[8 * i + 6] = v.x; wg[8 * i + 7] = v.y;
      v = h2f2(f.x); wf[8 * i + 0] = v.x; wf[8 * i + 1] = v.y;
      v = h2f2(f.y); wf[8 * i + 2] = v.x; wf[8 * i + 3] = v.y;
      v = h2f2(f.z); wf[8 * i + 4] = v.x; wf[8 * i + 5] = v.y;
      v = h2f2(f.w); wf[8 * i + 6] = v.x; wf[8 * i + 7] = v.y;
    }
  }
  for (int e = tid; e < T * R_; e += NT) hs[e / R_][e % R_] = __half2float(hidden[(long)s * T * R_ + e]);
  __syncthreads();

  float xv[TT];
#pragma unroll
  for (int t = 0; t < TT; t++) {
    float g0 = 0.f, g1 = 0.f, g2 = 0.f, g3 = 0.f, f0 = 0.f, f1 = 0.f, f2 = 0.f, f3 = 0.f;
    const float4* h4 = reinterpret_cast<const float4*>(hs[t]);
#pragma unroll
    for (int i = 0; i < R_ / 4; i++) {
      float4 h = h4[i];
      g0 = fmaf(wg[4 * i + 0], h.x, g0); g1 = fmaf(wg[4 * i + 1], h.y, g1);
      g2 = fmaf(wg[4 * i + 2], h.z, g2); g3 = fmaf(wg[4 * i + 3], h.w, g3);
      f0 = fmaf(wf[4 * i + 0], h.x, f0); f1 = fmaf(wf[4 * i + 1], h.y, f1);
      f2 = fmaf(wf[4 * i + 2], h.z, f2); f3 = fmaf(wf[4 * i + 3], h.w, f3);
    }
    float gain = tanhf((g0 + g1) + (g2 + g3));
    float forget = tanhf((f0 + f1) + (f2 + f3));
    float x = 0.f;
    if (t < T) {
      float d = dest[s * dest_stride_s + (long)t * D + j];
      float m = msg[((long)s * T + t) * D + j];
      x = __fadd_rn(__fmul_rn(__fsub_rn(1.f, forget), d), __fmul_rn(gain, m));
      xs[((long)s * T + t) * D + j] = x;
    }
    xv[t] = x;
  }
#pragma unroll
  for (int t = 0; t < TT; t++) {
    float v = warp_sum(__fmul_rn(xv[t], xv[t]));
    if (lane == 0) red[warp][t] = v;
  }
  __syncthreads();
  if (tid < T) {
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < NT / 32; w++) v += red[w][tid < TT ? tid : 0];
    partial[((long)s * T + tid) * CB + cb] = v;
  }
  if (MODE == 2) return;  // partials published by the kernel boundary; B2 finishes
  __threadfence();
  __syncthreads();
  if (MODE == 1) {
    // Grid-wide barrier over the CB CTAs of this stream (all co-resident: grid <= SM count).
    // Counter cycles 0..2CB-1 (atomicInc wraps): phase 0 waits for v >= CB, phase 1 for v < CB.
    if (tid == 0) {
      unsigned old = atomicInc(counters + s, 2 * CB - 1);
      const bool ph1 = old >= (unsigned)CB;
      volatile unsigned* c = counters + s;
      while (true) {
        unsigned v = *c;
        if (ph1 ? (v < (unsigned)CB) : (v >= (unsigned)CB)) break;
      }
      __threadfence();
    }
    __syncthreads();
    __shared__ float rs1[TT];
    for (int t = warp; t < T; t += NT / 32) {
      float p = 0.f;
      for (int c = lane; c < CB; c += 32) p += __ldcg(partial + ((long)s * T + t) * CB + c);
      p = warp_sum(p);
      if (lane == 0) rs1[t] = rsqrt_approx(__fadd_rn(__fdiv_rn(p, (float)D), eps));
    }
    __syncthreads();
    const WT* w = nw + s * nw_stride_s;
    const float wj = ldf(w + j);
#pragma unroll
    for (int t = 0; t < TT; t++)
      if (t < T) out[((long)s * T + t) * D + j] = __float2half_rn(__fmul_rn(__fmul_rn(xv[t], rs1[t]), wj));
    return;
  }
  if (tid == 0) is_last = (atomicInc(counters + s, CB - 1) == (unsigned)(CB - 1));
  __syncthreads();
  if (!is_last) return;
  __threadfence();
  __shared__ float rs[TT];
  for (int t = warp; t < T; t += NT / 32) {
    float p = 0.f;
    for (int c = lane; c < CB; c += 32) p += __ldcg(partial + ((long)s * T + t) * CB + c);
    p = warp_sum(p);
    if (lane == 0) rs[t] = rsqrt_approx(__fadd_rn(__fdiv_rn(p, (float)D), eps));
  }
  __syncthreads();
  const int n4 = D >> 2;
  const WT* w = nw + s * nw_stride_s;
#pragma unroll 4
  for (int e = tid; e < T * n4; e += NT) {
    const int t = e / n4, q = e - t * n4;
    const float rstd = rs[t];
    float4 x = __ldcg(reinterpret_cast<const float4*>(xs + ((long)s * T + t) * D) + q);
    float w0 = ldf(w + 4 * q), w1 = ldf(w + 4 * q + 1), w2 = ldf(w + 4 * q + 2), w3 = ldf(w + 4 * q + 3);
    __half2* o2 = reinterpret_cast<__half2*>(out + ((long)s * T + t) * D);
    o2[2 * q] = __floats2half2_rn(__fmul_rn(__fmul_rn(x.x, rstd), w0), __fmul_rn(__fmul_rn(x.y, rstd), w1));
    o2[2 * q + 1] = __floats2half2_rn(__fmul_rn(__fmul_rn(x.z, rstd), w2), __fmul_rn(__fmul_rn(x.w, rstd), w3));
  }
}

// MODE 2 second kernel: grid (T, S), block 256.
template <typename WT>
__global__ void __launch_bounds__(256) norm_finish_kernel(
    const float* __restrict__ xs, const float* __restrict__ partial,
    const WT* __restrict__ nw, long nw_stride_s, __half* __restrict__ out,
    int T, int D, int CB, float eps) {
  const int t = blockIdx.x, s = blockIdx.y, tid = threadIdx.x;
  float p = 0.f;
  for (int c = 0; c < CB; c++) p += partial[((long)s * T + t) * CB + c];
  const float rstd = rsqrt_approx(__fadd_rn(__fdiv_rn(p, (float)D), eps));
  const int n4 = D >> 2;
  const WT* w = nw + s * nw_stride_s;
  for (int q = tid; q < n4; q += 256) {
    float4 x = reinterpret_cast<const float4*>(xs + ((long)s * T + t) * D)[q];
    float w0 = ldf(w + 4 * q), w1 = ldf(w + 4 * q + 1), w2 = ldf(w + 4 * q + 2), w3 = ldf(w + 4 * q + 3);
    __half2* o2 = reinterpret_cast<__half2*>(out + ((long)s * T + t) * D);
    o2[2 * q] = __floats2half2_rn(__fmul_rn(__fmul_rn(x.x, rstd), w0), __fmul_rn(__fmul_rn(x.y, rstd), w1));
    o2[2 * q + 1] = __floats2half2_rn(__fmul_rn(__fmul_rn(x.z, rstd), w2), __fmul_rn(__fmul_rn(x.w, rstd), w3));
  }
}

template <int NT, int TT, typename WT, int MODE>
static void launch_b(const at::Tensor& dest, const at::Tensor& msg, const at::Tensor& hidden,
                     const at::Tensor& wgf, const at::Tensor& nw, const at::Tensor& out,
                     const at::Tensor& xs, const at::Tensor& partial, const at::Tensor& counters,
                     int64_t S, int64_t T, int64_t D, double eps) {
  dim3 grid(D / NT, S);
  auto stream = at::cuda::getCurrentCUDAStream();
  gf_blend_kernel<NT, TT, WT, MODE><<<grid, NT, 0, stream>>>(
      dest.data_ptr<float>(), dest.stride(0), msg.data_ptr<float>(),
      reinterpret_cast<const __half*>(hidden.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(wgf.data_ptr<at::Half>()), wgf.stride(0),
      reinterpret_cast<const WT*>(nw.data_ptr()), nw.stride(0),
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()), xs.data_ptr<float>(),
      partial.data_ptr<float>(), reinterpret_cast<unsigned*>(counters.data_ptr<int>()),
      (int)T, (int)D, (float)eps);
  if (MODE == 2) {
    norm_finish_kernel<WT><<<dim3(T, S), 256, 0, stream>>>(
        xs.data_ptr<float>(), partial.data_ptr<float>(), reinterpret_cast<const WT*>(nw.data_ptr()), nw.stride(0),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()), (int)T, (int)D, (int)(D / NT), (float)eps);
  }
}

template <int NT, typename WT, int MODE>
static void dispatch_tt(int64_t T, const at::Tensor& dest, const at::Tensor& msg, const at::Tensor& hidden,
                        const at::Tensor& wgf, const at::Tensor& nw, const at::Tensor& out,
                        const at::Tensor& xs, const at::Tensor& partial, const at::Tensor& counters,
                        int64_t S, int64_t D, double eps) {
#define CASE(TT_) if (T <= TT_) { launch_b<NT, TT_, WT, MODE>(dest, msg, hidden, wgf, nw, out, xs, partial, counters, S, T, D, eps); return; }
  CASE(1) CASE(2) CASE(4) CASE(8) CASE(16)
#undef CASE
  TORCH_CHECK(false, "T too large");
}

void blend(const at::Tensor& dest, const at::Tensor& msg, const at::Tensor& wd, const at::Tensor& wgf,
           const at::Tensor& nw, const at::Tensor& hidden, const at::Tensor& out, const at::Tensor& xs,
           const at::Tensor& partial, const at::Tensor& counters, int64_t nt, int64_t nb, int64_t mode, double eps) {
  const int64_t S = msg.size(0), T = msg.size(1), D = msg.size(2);
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(D == 2304 && nb == 16, "pair_down needs D=2304, nb=16");
  dim3 ga(nb, T, S);
  size_t smem = 2 * D * sizeof(__half);
  pair_down_kernel<<<ga, 128, smem, stream>>>(
      dest.data_ptr<float>(), dest.stride(0), msg.data_ptr<float>(),
      reinterpret_cast<const __half*>(wd.data_ptr<at::Half>()), wd.stride(0),
      reinterpret_cast<__half*>(hidden.data_ptr<at::Half>()), (int)T, (int)D, 1e-6f);
  const bool wf32 = nw.scalar_type() == at::kFloat;
#define GO(NT_, WT_, M_) dispatch_tt<NT_, WT_, M_>(T, dest, msg, hidden, wgf, nw, out, xs, partial, counters, S, D, eps)
#define GOM(NT_, WT_) if (mode == 0) GO(NT_, WT_, 0); else if (mode == 1) GO(NT_, WT_, 1); else GO(NT_, WT_, 2)
  // Only the shipped variants are instantiated (JIT build time): 128 threads/CTA,
  // mode 0 (last-CTA finish), mode 1 (grid barrier), mode 2 (second finishing kernel).
  TORCH_CHECK(nt == 128 && mode >= 0 && mode <= 2, "unsupported nt/mode");
  if (wf32) { GOM(128, float); } else { GOM(128, __half); }
#undef GO
#undef GOM
}
"""

_CPP_SRC = (
    "void blend(const at::Tensor& dest, const at::Tensor& msg, const at::Tensor& wd, const at::Tensor& wgf,"
    " const at::Tensor& nw, const at::Tensor& hidden, const at::Tensor& out, const at::Tensor& xs,"
    " const at::Tensor& partial, const at::Tensor& counters, int64_t nt, int64_t nb, int64_t mode, double eps);"
)

_ext = None


def _load():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load_inline

        build_dir = os.path.join(
            os.environ.get(
                "VLLM_INKLING2_SM70_EXT_DIR",
                os.path.expanduser("~/.cache/vllm/inkling2_sm70_ext"),
            ),
            "loop_fused",
        )
        os.makedirs(build_dir, exist_ok=True)
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
        _ext = load_inline(
            name="inkling2_sm70_loop_fused",
            cpp_sources=[_CPP_SRC],
            cuda_sources=[_CUDA_SRC],
            functions=["blend"],
            extra_cuda_cflags=["-O3", "--fmad=false"]
            + (["-Xptxas=-v"] if os.environ.get("EXT_VERBOSE") else []),
            build_directory=build_dir,
            verbose=bool(os.environ.get("EXT_VERBOSE")),
        )
    return _ext


def enabled() -> bool:
    return os.environ.get("VLLM_INKLING2_SM70_FUSED_BLEND", "1") != "0"


def make_state(device) -> torch.Tensor:
    """Ticket counters of the last-CTA row finish (self-resetting): [stream]."""
    _load()
    return torch.zeros(8, dtype=torch.int32, device=device)


def supports(num_tokens: int, d: int, rank: int) -> bool:
    return (
        enabled()
        and 1 <= num_tokens <= MAX_TOKENS
        and rank == RANK
        and d == 2304  # kernel A is specialised to D = 2304
    )


_NT = int(os.environ.get("VLLM_INKLING2_SM70_BLEND_NT", "128"))
_NB = 16
_MODE = int(os.environ.get("VLLM_INKLING2_SM70_BLEND_MODE", "1"))


def blend(
    dest: torch.Tensor,
    message: torch.Tensor,
    controller_down: torch.Tensor,
    controller_gain_forget: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    counters: torch.Tensor,
    mode: int | None = None,
    nt: int | None = None,
) -> torch.Tensor:
    """Same contract as `Inkling2LoopDecoderLayer.blend` for T <= MAX_TOKENS:
    fp32 `(S, T, D)` `dest` (may broadcast over streams) and `message`,
    `controller_down` fp16 `(S, R, 2D)`, `controller_gain_forget` fp16
    `(S, 2D, R)` of the same streams, `norm_weight` `(S, D)` (may broadcast).
    Returns fp16 `(S, T, D)`."""
    ext = _load()
    mode = _MODE if mode is None else mode
    nt = _NT if nt is None else nt
    s, t, d = message.shape
    if mode == 1 and s * (d // nt) > 80:
        mode = 0  # grid-wide barrier needs every CTA co-resident
    assert message.is_contiguous() and message.dtype == torch.float32
    assert dest.dtype == torch.float32 and dest.stride(1) == d and dest.stride(2) == 1
    assert controller_down.is_contiguous() and controller_gain_forget.is_contiguous()
    assert norm_weight.shape == (s, d) and norm_weight.stride(1) == 1
    assert norm_weight.dtype in (torch.float32, torch.float16)
    dev = message.device
    hidden = torch.empty((s, t, RANK), dtype=torch.float16, device=dev)
    out = torch.empty((s, t, d), dtype=torch.float16, device=dev)
    xs = torch.empty((s, t, d), dtype=torch.float32, device=dev)
    partial = torch.empty((s, t, d // nt), dtype=torch.float32, device=dev)
    ext.blend(
        dest,
        message,
        controller_down,
        controller_gain_forget,
        norm_weight,
        hidden,
        out,
        xs,
        partial,
        counters,
        nt,
        _NB,
        mode,
        norm_eps,
    )
    return out


def all_messages(bank: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
    """fp32 `(L, Q, T, D)` messages of every core layer and stream for the
    pass's `bank` `(N+1, T, D)`: `queries` `(L, Q, D)` fp32 pre-scaled.
    One launch of the `attn_res` row kernel (normalized values, no norm)."""
    num_layers, num_q, d = queries.shape
    stack, last = bank[:-1], bank[-1]
    num_tokens = last.shape[0]
    assert (
        last.is_contiguous()
        and queries.is_contiguous()
        and queries.dtype == torch.float32
    )
    assert stack.stride(1) == d and stack.stride(2) == 1
    out = torch.empty(
        (num_layers * num_q, num_tokens, d), dtype=torch.float32, device=last.device
    )
    num_stack = stack.shape[0]
    block_a = triton.next_power_of_2(d + 1) // 2
    block_b = max(triton.next_power_of_2(d - block_a), 16)
    _attn_res_row_kernel[(num_layers * num_q, num_tokens)](
        stack,
        last,
        None,
        queries,
        queries,
        out,
        stack.stride(0),
        num_tokens,
        1e-6,
        0.0,
        D=d,
        N_STACK=num_stack,
        HAS_ADDEND=False,
        NORMALIZE=True,
        APPLY_NORM=False,
        BLOCK_N=triton.next_power_of_2(num_stack + 1),
        enable_fp_fusion=False,
        BLOCK_A=block_a,
        BLOCK_B=block_b,
        num_warps=8,
    )
    return out.view(num_layers, num_q, num_tokens, d)
