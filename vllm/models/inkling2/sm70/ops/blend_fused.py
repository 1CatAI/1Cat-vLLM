# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""One-launch, token-parallel loop blend for decode-size batches on SM70.

`ops/loop_fused.py` runs a blend as two launches: `pair_down_kernel` and
`gf_blend_kernel<128, TT>`, the second fully unrolled over TT = 1/2/4/8/16
tokens per thread (5 tokens run the 8-token code), so its time grows with
the token count. This kernel does the same arithmetic in ONE launch with one
CTA per (128 columns, stream, token): grid (D / 128, S * T) x 128 threads, so
the latency is that of one token while the CTAs are co-resident.

  A  CTA cb of row (s, t) owns down rows 4 cb .. 4 cb + 3 (one per warp): RMS
     of dest and of the message, the fp16 pair in shared memory, the rank-64
     down dot -> hidden.
  -- grid barrier among the 18 CTAs of the row (hidden complete)
  B  thread = output column j: gain / forget dots against hidden,
     `(1 - tanh f) dest + tanh g msg` in fp32, per-CTA partial sum of squares
  -- grid barrier (row RMS), then the weight scale and the fp16 store.

Every expression, reduction order and compile flag (`--fmad=false`) is that
of `loop_fused.py` (pair_down + gf_blend mode 1), so the output is
bit-identical to it; only the launch count and the work split differ.
"""

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

MAX_TOKENS = 16
RANK = 64
_NT = 128
# The two barriers need every CTA resident at once. A CTA is 128 threads, 9.5 KB of
# shared memory and 64 registers per thread (ptxas), so an SM holds 8 of them (the
# register file is the limit): 80 SMs x 7 leaves a margin. 3 streams x 8 tokens = 432.
MAX_CTAS = int(os.environ.get("VLLM_INKLING2_SM70_BLEND_ONE_MAX_CTAS", "560"))

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define R_ 64
#define K8_ 576          // 2D / 8 uint4 per down row (D = 2304)
#define NT_ 128

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

__device__ __forceinline__ void prefetch_l2(const void* p) {
  asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}

__device__ __forceinline__ float ldf(const float* p) { return *p; }
__device__ __forceinline__ float ldf(const __half* p) { return __half2float(*p); }

__device__ __forceinline__ float2 h2f2(unsigned u) {
  return __half22float2(*reinterpret_cast<__half2*>(&u));
}

// Barrier over the CB co-resident CTAs of one (stream, token) row. The counter
// cycles 0..2CB-1 (atomicInc wraps), so it never needs a reset.
__device__ __forceinline__ void row_barrier(unsigned* ctr, int CB) {
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned old = atomicInc(ctr, 2u * CB - 1u);
    const bool ph1 = old >= (unsigned)CB;
    volatile unsigned* c = ctr;
    while (true) {
      const unsigned v = *c;
      if (ph1 ? (v < (unsigned)CB) : (v >= (unsigned)CB)) break;
    }
    __threadfence();
  }
  __syncthreads();
}

// grid (D / NT_, S * T), block NT_: CTA = (128 columns, stream s, token t).
template <typename WT>
__global__ void __launch_bounds__(NT_) blend_fused_kernel(
    const float* __restrict__ dest, long dest_stride_s,
    const float* __restrict__ msg,
    const __half* __restrict__ wd, long wd_stride_s,
    const __half* __restrict__ wgf, long wgf_stride_s,
    const WT* __restrict__ nw, long nw_stride_s,
    __half* hidden, float* partial, unsigned* counters,
    __half* __restrict__ out,
    int T, int D, float pair_eps, float eps) {
  __shared__ __align__(16) __half pair[2 * 2304];
  __shared__ __align__(16) float hs[R_];
  __shared__ float red[2][NT_ / 32];
  __shared__ float red2[NT_ / 32];
  __shared__ float rs1;
  const int y = blockIdx.y, s = y / T, t = y - s * T;
  const int cb = blockIdx.x, CB = gridDim.x;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const float* drow = dest + s * dest_stride_s + (long)t * D;
  const float* mrow = msg + (long)y * D;
#ifdef BLEND_PREFETCH
  // Weights of both phases do not depend on anything computed here: pull them toward L2 now so
  // the loads after the RMS and after the barrier hit L2 instead of DRAM.
  {
    const __half* gwp = wgf + s * wgf_stride_s + (long)(cb * NT_ + tid) * R_;
    prefetch_l2(gwp);
    prefetch_l2(gwp + (long)D * R_);
    if (cb * 4 < R_) {
      const char* wp = reinterpret_cast<const char*>(wd + s * wd_stride_s + (long)(cb * 4 + warp) * 2 * D);
      prefetch_l2(wp + lane * 128);
      prefetch_l2(wp + (lane + 32) * 128);
      if (lane < 8) prefetch_l2(wp + (lane + 64) * 128);
    }
  }
#endif

  // ------------- A: pair RMS + rank-64 down projection (rows 4 cb .. 4 cb + 3) -------------
  if (cb * 4 < R_) {
    const int r = cb * 4 + warp;
    const uint4* w8 = reinterpret_cast<const uint4*>(wd + s * wd_stride_s + (long)r * 2 * D);
    const float4* d4 = reinterpret_cast<const float4*>(drow);
    const float4* m4 = reinterpret_cast<const float4*>(mrow);
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
    const float rd = rsqrt_approx(__fadd_rn(__fdiv_rn(sd, (float)D), pair_eps));
    const float rm = rsqrt_approx(__fadd_rn(__fdiv_rn(sm, (float)D), pair_eps));
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
#pragma unroll 6
    for (int i = 0; i < K8_ / 32; i++) {
      uint4 w = __ldg(w8 + lane + 32 * i);
      uint4 p = pair8[lane + 32 * i];
      float2 w0 = h2f2(w.x), p0 = h2f2(p.x), w1 = h2f2(w.y), p1 = h2f2(p.y);
      float2 w2 = h2f2(w.z), p2_ = h2f2(p.z), w3 = h2f2(w.w), p3 = h2f2(p.w);
      a0 = fmaf(w0.x, p0.x, a0); a1 = fmaf(w0.y, p0.y, a1);
      a0 = fmaf(w1.x, p1.x, a0); a1 = fmaf(w1.y, p1.y, a1);
      a0 = fmaf(w2.x, p2_.x, a0); a1 = fmaf(w2.y, p2_.y, a1);
      a0 = fmaf(w3.x, p3.x, a0); a1 = fmaf(w3.y, p3.y, a1);
    }
    float acc = warp_sum(a0 + a1);
    if (lane == 0) hidden[(long)y * R_ + r] = __float2half_rn(acc);
  }
  row_barrier(counters + 2 * y, CB);

  // ------------- B: gain / forget, blend, row RMS, weight scale -------------
  const int j = cb * NT_ + tid;
  // hidden was written by the other CTAs of the row in this launch: ld.cg.
  if (tid < R_ / 8) {
    const uint4 h = __ldcg(reinterpret_cast<const uint4*>(hidden + (long)y * R_) + tid);
    float* dst = hs + 8 * tid;
    float2 v;
    v = h2f2(h.x); dst[0] = v.x; dst[1] = v.y;
    v = h2f2(h.y); dst[2] = v.x; dst[3] = v.y;
    v = h2f2(h.z); dst[4] = v.x; dst[5] = v.y;
    v = h2f2(h.w); dst[6] = v.x; dst[7] = v.y;
  }
  const uint4* g8 = reinterpret_cast<const uint4*>(wgf + s * wgf_stride_s + (long)j * R_);
  const uint4* f8 = reinterpret_cast<const uint4*>(wgf + s * wgf_stride_s + (long)(D + j) * R_);
  uint4 gw[R_ / 8], fw[R_ / 8];
#pragma unroll
  for (int i = 0; i < R_ / 8; i++) { gw[i] = __ldg(g8 + i); fw[i] = __ldg(f8 + i); }
  const float d = drow[j];
  const float m = mrow[j];
  __syncthreads();
  float g0 = 0.f, g1 = 0.f, g2 = 0.f, g3 = 0.f, f0 = 0.f, f1 = 0.f, f2 = 0.f, f3 = 0.f;
  const float4* h4 = reinterpret_cast<const float4*>(hs);
#pragma unroll
  for (int i = 0; i < R_ / 8; i++) {
    const float4 ha = h4[2 * i], hb = h4[2 * i + 1];
    const float2 ga = h2f2(gw[i].x), gb = h2f2(gw[i].y), gc = h2f2(gw[i].z), gd = h2f2(gw[i].w);
    const float2 fa = h2f2(fw[i].x), fb = h2f2(fw[i].y), fc = h2f2(fw[i].z), fd = h2f2(fw[i].w);
    g0 = fmaf(ga.x, ha.x, g0); g1 = fmaf(ga.y, ha.y, g1);
    g2 = fmaf(gb.x, ha.z, g2); g3 = fmaf(gb.y, ha.w, g3);
    f0 = fmaf(fa.x, ha.x, f0); f1 = fmaf(fa.y, ha.y, f1);
    f2 = fmaf(fb.x, ha.z, f2); f3 = fmaf(fb.y, ha.w, f3);
    g0 = fmaf(gc.x, hb.x, g0); g1 = fmaf(gc.y, hb.y, g1);
    g2 = fmaf(gd.x, hb.z, g2); g3 = fmaf(gd.y, hb.w, g3);
    f0 = fmaf(fc.x, hb.x, f0); f1 = fmaf(fc.y, hb.y, f1);
    f2 = fmaf(fd.x, hb.z, f2); f3 = fmaf(fd.y, hb.w, f3);
  }
  const float gain = tanhf((g0 + g1) + (g2 + g3));
  const float forget = tanhf((f0 + f1) + (f2 + f3));
  const float x = __fadd_rn(__fmul_rn(__fsub_rn(1.f, forget), d), __fmul_rn(gain, m));
  {
    float v = warp_sum(__fmul_rn(x, x));
    if (lane == 0) red2[warp] = v;
  }
  __syncthreads();
  if (tid == 0) {
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < NT_ / 32; w++) v += red2[w];
    partial[(long)y * CB + cb] = v;
  }
  row_barrier(counters + 2 * y + 1, CB);
  if (warp == 0) {
    float p = 0.f;
    for (int c = lane; c < CB; c += 32) p += __ldcg(partial + (long)y * CB + c);
    p = warp_sum(p);
    if (lane == 0) rs1 = rsqrt_approx(__fadd_rn(__fdiv_rn(p, (float)D), eps));
  }
  __syncthreads();
  const float wj = ldf(nw + s * nw_stride_s + j);
  out[(long)y * D + j] = __float2half_rn(__fmul_rn(__fmul_rn(x, rs1), wj));
}

void blend_one(const at::Tensor& dest, const at::Tensor& msg, const at::Tensor& wd, const at::Tensor& wgf,
               const at::Tensor& nw, const at::Tensor& hidden, const at::Tensor& partial,
               const at::Tensor& counters, const at::Tensor& out, int64_t max_ctas, double eps) {
  const int64_t S = msg.size(0), T = msg.size(1), D = msg.size(2);
  TORCH_CHECK(D == 2304 && T >= 1 && S >= 1 && S * T * (D / NT_) <= max_ctas,
              "blend_one: needs D=2304 and all CTAs co-resident");
  TORCH_CHECK(dest.scalar_type() == at::kFloat && msg.scalar_type() == at::kFloat && msg.is_contiguous() &&
              wd.scalar_type() == at::kHalf && wgf.scalar_type() == at::kHalf &&
              wd.size(1) == R_ && wd.size(2) == 2 * D && wgf.size(1) == 2 * D && wgf.size(2) == R_,
              "blend_one: inputs");
  TORCH_CHECK(hidden.scalar_type() == at::kHalf && hidden.numel() >= S * T * R_ &&
              partial.scalar_type() == at::kFloat && partial.numel() >= S * T * (D / NT_) &&
              counters.scalar_type() == at::kInt && counters.numel() >= 2 * S * T &&
              out.scalar_type() == at::kHalf && out.is_contiguous() && out.numel() == S * T * D,
              "blend_one: scratch / out");
  dim3 grid(D / NT_, S * T);
  auto stream = at::cuda::getCurrentCUDAStream();
#define GO(WT_)                                                                                   \
  blend_fused_kernel<WT_><<<grid, NT_, 0, stream>>>(                                              \
      dest.data_ptr<float>(), dest.stride(0), msg.data_ptr<float>(),                              \
      reinterpret_cast<const __half*>(wd.data_ptr<at::Half>()), wd.stride(0),                     \
      reinterpret_cast<const __half*>(wgf.data_ptr<at::Half>()), wgf.stride(0),                   \
      reinterpret_cast<const WT_*>(nw.data_ptr()), nw.stride(0),                                  \
      reinterpret_cast<__half*>(hidden.data_ptr<at::Half>()), partial.data_ptr<float>(),          \
      reinterpret_cast<unsigned*>(counters.data_ptr<int>()),                                      \
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()), (int)T, (int)D, 1e-6f, (float)eps)
  if (nw.scalar_type() == at::kFloat) { GO(float); } else { GO(__half); }
#undef GO
}
"""

_CPP_SRC = (
    "void blend_one(const at::Tensor& dest, const at::Tensor& msg, const at::Tensor& wd, const at::Tensor& wgf,"
    " const at::Tensor& nw, const at::Tensor& hidden, const at::Tensor& partial,"
    " const at::Tensor& counters, const at::Tensor& out, int64_t max_ctas, double eps);"
)

ENABLED = os.environ.get("VLLM_INKLING2_SM70_BLEND_ONE", "1") == "1"
# VLLM_INKLING2_SM70_BLEND_PREFETCH: weights of both phases prefetched to L2 at the start (same bits).
_PREFETCH = os.environ.get("VLLM_INKLING2_SM70_BLEND_PREFETCH", "1") == "1"
_MAX_ROWS = 64  # (stream, token) rows with their own barrier counters

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
            "blend_fused" + ("_pf" if _PREFETCH else ""),
        )
        os.makedirs(build_dir, exist_ok=True)
        verbose = bool(os.environ.get("VLLM_INKLING2_SM70_EXT_VERBOSE"))
        _ext = load_inline(
            name="inkling2_sm70_blend_fused" + ("_pf" if _PREFETCH else ""),
            cpp_sources=[_CPP_SRC],
            cuda_sources=[_CUDA_SRC],
            functions=["blend_one"],
            # Same flags as loop_fused.py: the arithmetic is written with explicit fma / rn ops.
            extra_cuda_cflags=[
                "-O3",
                "--fmad=false",
                "-gencode=arch=compute_70,code=sm_70",
            ]
            + (["-DBLEND_PREFETCH"] if _PREFETCH else [])
            + (["-Xptxas=-v"] if verbose else []),
            build_directory=build_dir,
            verbose=verbose,
        )
    return _ext


class BlendState:
    """Per-layer scratch of the one-launch blend: barrier counters and the
    cross-CTA buffers (hidden, partial sums)."""

    def __init__(self, device: torch.device, d: int) -> None:
        self.counters = torch.zeros(2 * _MAX_ROWS, dtype=torch.int32, device=device)
        self.hidden = torch.zeros((_MAX_ROWS, RANK), dtype=torch.float16, device=device)
        self.partial = torch.zeros(
            (_MAX_ROWS, d // _NT), dtype=torch.float32, device=device
        )


def make_state(device: torch.device, d: int, rank: int) -> BlendState | None:
    """Scratch of one loop layer (None = path off). Loads the extension: call at
    load time, never inside CUDA-graph capture."""
    if not ENABLED or d != 2304 or rank != RANK:
        return None
    _load()
    return BlendState(device, d)


# Which blends take the one-launch kernel: "all" = whatever is admitted (in the real
# TP4 step the 3-stream attn blend at 5 tokens goes 36.8 -> 22.7 us, the mlp blend
# 35.6 -> 17.5), "auto" = the single-stream mlp blend at any token count and the attn
# blend up to 2 tokens. The output is the same bits either way.
_POLICY = os.environ.get("VLLM_INKLING2_SM70_BLEND_ONE_POLICY", "all")


def supports(num_streams: int, num_tokens: int, d: int, rank: int) -> bool:
    rows = num_streams * num_tokens
    return (
        (_POLICY == "all" or num_streams == 1 or num_tokens <= 2)
        and 1 <= num_tokens <= MAX_TOKENS
        and rank == RANK
        and d == 2304
        and 1 <= rows <= _MAX_ROWS
        and rows * (d // _NT) <= MAX_CTAS  # the barriers need every CTA co-resident
    )


def blend(
    dest: torch.Tensor,
    message: torch.Tensor,
    controller_down: torch.Tensor,
    controller_gain_forget: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    state: BlendState,
) -> torch.Tensor:
    """Same contract as `loop_fused.blend`: fp32 `(S, T, D)` `dest` (may
    broadcast over streams) and `message`, fp16 `controller_down (S, R, 2D)`
    and `controller_gain_forget (S, 2D, R)`, `norm_weight (S, D)` (may
    broadcast). Returns fp16 `(S, T, D)`, bit-identical to `loop_fused.blend`."""
    s, t, d = message.shape
    assert message.is_contiguous() and message.dtype == torch.float32
    assert dest.dtype == torch.float32 and dest.stride(1) == d and dest.stride(2) == 1
    assert controller_down.is_contiguous() and controller_gain_forget.is_contiguous()
    assert norm_weight.shape == (s, d) and norm_weight.stride(1) == 1
    assert norm_weight.dtype in (torch.float32, torch.float16)
    out = torch.empty((s, t, d), dtype=torch.float16, device=message.device)
    logger.info_once(
        "Inkling2 SM70 one-launch loop blend in use (streams=%d, policy %s).",
        s,
        _POLICY,
    )
    _load().blend_one(
        dest,
        message,
        controller_down,
        controller_gain_forget,
        norm_weight,
        state.hidden,
        state.partial,
        state.counters,
        out,
        MAX_CTAS,
        norm_eps,
    )
    return out
