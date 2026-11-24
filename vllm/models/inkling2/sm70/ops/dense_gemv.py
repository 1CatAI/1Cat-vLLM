# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Small-M (decode / spec-verify, M <= 16) fp16 dense projections for SM70.

`out[m, n] = sum_k x[m, k] * W[n, k]` with fp16 inputs/outputs and fp32
accumulation (exact fp16 products, fp32 FMA, fixed reduction order: deterministic).
cuBLAS runs these projections at 200-400 GB/s through gemv2T / cutlass split-K +
splitKreduce kernels; this CUDA kernel puts one warp on each weight row (16 B loads,
the whole row in flight), and a job table lets several projections that share K
(qkvr + the global-layer K/V dynamic taps) run in ONE launch.

Flag `VLLM_INKLING2_SM70_DENSE_GEMV` (default 1); 0 keeps cuBLAS everywhere.
CUDA-graph safe: grid depends only on shapes, no host syncs or data-dependent sizes.
JIT extension, build dir `$VLLM_INKLING2_TORCH_EXT` (default work dir torch_ext).
"""

import os
import threading

import torch

ENABLED = os.environ.get("VLLM_INKLING2_SM70_DENSE_GEMV", "1") != "0"
# VLLM_INKLING2_SM70_GEMV_V2=1: ops/dense_gemv2.py (exact M, rows share x, reduce-scatter): bit-identical output, fewer
# instructions; also takes M 5..8 of the small-K weights (wo) away from cuBLAS. 0 = this file's kernel everywhere.
GEMV_V2 = os.environ.get("VLLM_INKLING2_SM70_GEMV_V2", "1") == "1"
MAX_M = 8
MAX_JOBS = 5

_CPP = r"""
#include <torch/extension.h>
#include <vector>
void dense_gemv_jobs(std::vector<torch::Tensor> w, std::vector<torch::Tensor> x,
                     std::vector<torch::Tensor> out, int64_t wpr, int64_t rpw);
"""

_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

constexpr int kMaxJobs = 5;
struct Job {
  const __half* w; const __half* x; __half* out;
  int n; int ldw; int ldx; int ldo; int blk0;
};
struct Jobs { Job j[kMaxJobs]; int njobs; };

__device__ __forceinline__ void fma8(float& acc, const uint4& wv, const uint4& xv) {
  const __half2* wh = reinterpret_cast<const __half2*>(&wv);
  const __half2* xh = reinterpret_cast<const __half2*>(&xv);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    float2 a = __half22float2(wh[i]);
    float2 b = __half22float2(xh[i]);
    acc = fmaf(a.x, b.x, acc);
    acc = fmaf(a.y, b.y, acc);
  }
}

// One block = 8 warps. WPR warps share a row (split along K, smem reduce), each
// warp handles RPW rows. MT = padded batch (1,2,4,8,16).
template <int MT, int WPR, int RPW, int ITERS>
__global__ void __launch_bounds__(256) gemv_kernel(Jobs jobs, int M, int K) {
  constexpr int WARPS = 8;
  constexpr int ROWS_PER_BLOCK = (WARPS / WPR) * RPW;
  const int bid = blockIdx.x;
  int ji = 0;
#pragma unroll
  for (int i = 1; i < kMaxJobs; i++)
    if (i < jobs.njobs && bid >= jobs.j[i].blk0) ji = i;
  const Job job = jobs.j[ji];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int wrow = warp / WPR;          // row slot inside block
  const int wk = warp % WPR;            // K slice of this warp
  const int row0 = (bid - job.blk0) * ROWS_PER_BLOCK + wrow * RPW;
  const int chunks = K >> 3;            // 16 B chunks per row
  float acc[RPW][MT];
#pragma unroll
  for (int r = 0; r < RPW; r++)
#pragma unroll
    for (int m = 0; m < MT; m++) acc[r][m] = 0.f;
  uint4 wreg[RPW][ITERS];
#pragma unroll
  for (int it = 0; it < ITERS; it++) {
    const int c = (it * WPR + wk) * 32 + lane;
#pragma unroll
    for (int r = 0; r < RPW; r++) {
      const int row = row0 + r;
      if (c < chunks && row < job.n)
        wreg[r][it] = __ldg(reinterpret_cast<const uint4*>(job.w + (size_t)row * job.ldw) + c);
      else
        wreg[r][it] = make_uint4(0, 0, 0, 0);
    }
  }
#pragma unroll
  for (int it = 0; it < ITERS; it++) {
    const int c = (it * WPR + wk) * 32 + lane;
    if (c < chunks) {
      float wf[RPW][8];
#pragma unroll
      for (int r = 0; r < RPW; r++) {
        const __half2* wh = reinterpret_cast<const __half2*>(&wreg[r][it]);
#pragma unroll
        for (int i = 0; i < 4; i++) {
          float2 a = __half22float2(wh[i]);
          wf[r][2 * i] = a.x; wf[r][2 * i + 1] = a.y;
        }
      }
#pragma unroll
      for (int m = 0; m < MT; m++) {
        const int mm = m < M ? m : M - 1;
        uint4 xv = __ldg(reinterpret_cast<const uint4*>(job.x + (size_t)mm * job.ldx) + c);
        const __half2* xh = reinterpret_cast<const __half2*>(&xv);
        float xf[8];
#pragma unroll
        for (int i = 0; i < 4; i++) {
          float2 b = __half22float2(xh[i]);
          xf[2 * i] = b.x; xf[2 * i + 1] = b.y;
        }
#pragma unroll
        for (int r = 0; r < RPW; r++)
#pragma unroll
          for (int i = 0; i < 8; i++) acc[r][m] = fmaf(wf[r][i], xf[i], acc[r][m]);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < RPW; r++)
#pragma unroll
    for (int m = 0; m < MT; m++) {
      float v = acc[r][m];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      acc[r][m] = v;
    }
  if (WPR == 1) {
    if (lane < MT * RPW) {
      const int r = lane / MT, m = lane % MT;
      float v = 0.f;
#pragma unroll
      for (int rr = 0; rr < RPW; rr++)
#pragma unroll
        for (int mm = 0; mm < MT; mm++)
          if (rr == r && mm == m) v = acc[rr][mm];
      const int row = row0 + r;
      if (m < M && row < job.n) job.out[(size_t)m * job.ldo + row] = __float2half_rn(v);
    }
  } else {
    __shared__ float red[WARPS][RPW * MT];
    if (lane == 0) {
#pragma unroll
      for (int r = 0; r < RPW; r++)
#pragma unroll
        for (int m = 0; m < MT; m++) red[warp][r * MT + m] = acc[r][m];
    }
    __syncthreads();
    if (wk == 0 && lane < MT * RPW) {
      float v = 0.f;
      for (int k = 0; k < WPR; k++) v += red[wrow * WPR + k][lane];
      const int r = lane / MT, m = lane % MT;
      const int row = row0 + r;
      if (m < M && row < job.n) job.out[(size_t)m * job.ldo + row] = __float2half_rn(v);
    }
  }
}

template <int MT, int WPR, int RPW>
void launch(const Jobs& jobs, int M, int K, int blocks, cudaStream_t s) {
  const int chunks = K / 8;
  const int iters = (chunks + WPR * 32 - 1) / (WPR * 32);
  switch (iters) {
#define CASE(I) case I: gemv_kernel<MT, WPR, RPW, I><<<blocks, 256, 0, s>>>(jobs, M, K); break;
    CASE(1) CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8) CASE(9)
    CASE(10) CASE(11) CASE(12) CASE(13) CASE(14) CASE(15) CASE(16)
#undef CASE
    default: TORCH_CHECK(false, "K too large for dense_gemv: ", K);
  }
}

template <int MT>
void launch_cfg(const Jobs& jobs, int M, int K, int blocks, int wpr, int rpw, cudaStream_t s) {
  if (wpr == 1 && rpw == 1) launch<MT, 1, 1>(jobs, M, K, blocks, s);
  else if (wpr == 1 && rpw == 2) launch<MT, 1, 2>(jobs, M, K, blocks, s);
  else if (wpr == 1 && rpw == 4) launch<MT, 1, 4>(jobs, M, K, blocks, s);
  else if (wpr == 2 && rpw == 1) launch<MT, 2, 1>(jobs, M, K, blocks, s);
  else if (wpr == 4 && rpw == 1) launch<MT, 4, 1>(jobs, M, K, blocks, s);
  else if (wpr == 8 && rpw == 1) launch<MT, 8, 1>(jobs, M, K, blocks, s);
  else TORCH_CHECK(false, "bad wpr/rpw");
}

void dense_gemv_jobs(std::vector<torch::Tensor> w, std::vector<torch::Tensor> x,
                     std::vector<torch::Tensor> out, int64_t wpr, int64_t rpw) {
  const int nj = (int)w.size();
  TORCH_CHECK(nj >= 1 && nj <= kMaxJobs && x.size() == w.size() && out.size() == w.size());
  const c10::cuda::OptionalCUDAGuard guard(w[0].device());
  Jobs jobs; jobs.njobs = nj;
  const int M = (int)x[0].size(0), K = (int)w[0].size(1);
  const int rows_per_block = (8 / (int)wpr) * (int)rpw;
  int blk = 0;
  for (int i = 0; i < nj; i++) {
    TORCH_CHECK(w[i].size(1) == K && x[i].size(1) == K && x[i].size(0) == M && out[i].size(0) == M);
    TORCH_CHECK(w[i].stride(1) == 1 && x[i].stride(1) == 1 && out[i].stride(1) == 1);
    TORCH_CHECK(w[i].scalar_type() == at::kHalf && x[i].scalar_type() == at::kHalf && out[i].scalar_type() == at::kHalf);
    TORCH_CHECK(K % 8 == 0 && w[i].stride(0) % 8 == 0 && x[i].stride(0) % 8 == 0);
    TORCH_CHECK(((uintptr_t)w[i].data_ptr() | (uintptr_t)x[i].data_ptr()) % 16 == 0);
    jobs.j[i] = Job{(const __half*)w[i].data_ptr(), (const __half*)x[i].data_ptr(),
                    (__half*)out[i].data_ptr(), (int)w[i].size(0), (int)w[i].stride(0),
                    (int)x[i].stride(0), (int)out[i].stride(0), blk};
    blk += ((int)w[i].size(0) + rows_per_block - 1) / rows_per_block;
  }
  cudaStream_t s = at::cuda::getCurrentCUDAStream();
  if (M <= 1) launch_cfg<1>(jobs, M, K, blk, (int)wpr, (int)rpw, s);
  else if (M <= 2) launch_cfg<2>(jobs, M, K, blk, (int)wpr, (int)rpw, s);
  else if (M <= 4) launch_cfg<4>(jobs, M, K, blk, (int)wpr, (int)rpw, s);
  else if (M <= 8) launch_cfg<8>(jobs, M, K, blk, (int)wpr, (int)rpw, s);
  else if (M <= 16) launch_cfg<16>(jobs, M, K, blk, (int)wpr, (int)rpw, s);
  else TORCH_CHECK(false, "M > 16");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

_mod = None
_lock = threading.Lock()


def _ext():
    global _mod
    if _mod is None:
        with _lock:
            if _mod is None:
                from torch.utils.cpp_extension import load_inline

                build = os.environ.get("VLLM_INKLING2_TORCH_EXT") or os.path.join(
                    os.environ.get(
                        "VLLM_INKLING2_SM70_EXT_DIR",
                        os.path.expanduser("~/.cache/vllm/inkling2_sm70_ext"),
                    ),
                    "dense_gemv",
                )
                os.makedirs(build, exist_ok=True)
                _mod = load_inline(
                    "inkling2_dense_gemv",
                    cpp_sources=_CPP,
                    cuda_sources=_CUDA,
                    functions=["dense_gemv_jobs"],
                    extra_cuda_cflags=["-O3", "-arch=sm_70", "--use_fast_math"],
                    build_directory=build,
                    verbose=False,
                )
    return _mod


# Tuned on V100 (bench_dense.py tune, cold weights): warps-per-row 8 (one row per
# 8-warp block) wins for the K=2304 projections at M<=8; the K<=768 wo wants 4
# warps/row and only beats cuBLAS at M<=4.
def _cfg(n: int, k: int, m: int = 1) -> tuple[int, int]:
    if n < 512 or k >= 2048:
        return (8, 1)
    return (4, 1)


def _wins(n: int, k: int, m: int) -> bool:
    """Whether the new kernel beats cuBLAS (cold weights, V100): always at M<=4;
    at M 5..8 only for small K=2304-class weights (<= 12 MB: TP4 qkvr, taps);
    the CUDA-core FMA rate loses to cuBLAS for bigger/wider cases."""
    if m <= 4:
        return True
    if GEMV_V2 and m <= 8:
        from . import dense_gemv2 as _g2

        return _g2.config(n, k, m) is not None
    return m <= 8 and (n * k * 2 <= 12_000_000) and k >= 2048


def usable(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        ENABLED
        and x.dtype == torch.float16
        and weight.dtype == torch.float16
        and x.ndim == 2
        and x.shape[0] <= MAX_M
        and _wins(weight.shape[0], weight.shape[1], x.shape[0])
        and x.is_cuda
        and x.stride(1) == 1
        and weight.stride(1) == 1
        and weight.shape[1] % 8 == 0
    )


def aligned(x: torch.Tensor) -> bool:
    return x.data_ptr() % 16 == 0 and x.stride(0) % 8 == 0


def dense_gemv_multi(
    jobs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    cfg: tuple[int, int] | None = None,
) -> None:
    """Run several `out = x @ w.T` (same K and M) in one launch. `out` is written
    in place; it may be a column slice of a wider buffer (stride(1) == 1)."""
    ws, xs, outs = zip(*jobs)
    if GEMV_V2 and cfg is None and xs[0].shape[0] <= MAX_M:
        from . import dense_gemv2 as _g2

        c2 = _g2.config(sum(w.shape[0] for w in ws), ws[0].shape[1], xs[0].shape[0])
        if c2 is not None and _g2.gemv2(ws, xs, outs, *c2):
            return
    wpr, rpw = cfg or _cfg(sum(w.shape[0] for w in ws), ws[0].shape[1], xs[0].shape[0])
    _ext().dense_gemv_jobs(list(ws), list(xs), list(outs), wpr, rpw)


def dense_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None = None,
    cfg: tuple[int, int] | None = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty(
            (x.shape[0], weight.shape[0]), dtype=torch.float16, device=x.device
        )
    dense_gemv_multi([(weight, x, out)], cfg)
    return out
