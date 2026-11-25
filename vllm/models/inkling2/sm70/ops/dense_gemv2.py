# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Second-generation small-M fp16 GEMV (M 1..8 exact, no padding), bit-identical to ops/dense_gemv.py.

Same mapping of work to lanes and the same fp32 accumulation order as `gemv_kernel` (lane chunks
in `it` order, then the xor-16/8/4/2/1 lane tree, then the cross-warp sum in warp order), so the
output equals the first generation bit for bit. What changes is the instruction count per row:
* MT is the real M (1..8): no padded rows, no padded x loads;
* RPW rows per warp share every x load / fp16->fp32 conversion (x costs MT x (1 LDG + 8 cvt)
  per chunk, as much as the FMAs when RPW is 1);
* the 32-lane sum of the N = RPW x MT accumulators is a butterfly reduce-scatter (N - 1 + 5
  shuffles instead of 5 N), which pairs the same values at every level.

Flag `VLLM_INKLING2_SM70_GEMV_V2` (default off until gated): 0 = first generation everywhere.
JIT extension `dense_gemv2` in `$VLLM_INKLING2_SM70_EXT_DIR` (sources split per M so ninja builds
them in parallel).
"""

import os
import threading
from pathlib import Path

_HDR = r"""
#pragma once
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <stdint.h>

constexpr int kMaxJobs = 5;
struct Job {
  const __half* w; const __half* x; __half* out;
  int n; int ldw; int ldx; int ldo; int blk0;
};
struct Jobs { Job j[kMaxJobs]; int njobs; };

template <int N> struct PadN { static constexpr int P = N <= 1 ? 1 : N <= 2 ? 2 : N <= 4 ? 4 : N <= 8 ? 8 : N <= 16 ? 16 : 32; };
template <int P> struct Log2 { static constexpr int V = P <= 1 ? 0 : P <= 2 ? 1 : P <= 4 ? 2 : P <= 8 ? 3 : P <= 16 ? 4 : 5; };

// one reduce-scatter level: CNT values -> CNT / 2 (pair j with j + CNT / 2), or a plain butterfly add at CNT 1
template <int LV, int P>
__device__ __forceinline__ void rscatter(float (&a)[P], int lane) {
  if constexpr (LV < 5) {
    constexpr int o = 16 >> LV;
    constexpr int CNT = (P >> LV) > 0 ? (P >> LV) : 1;
    if constexpr (CNT > 1) {
      const bool up = (lane & o) != 0;
#pragma unroll
      for (int j = 0; j < CNT / 2; j++) {
        const float send = up ? a[j] : a[j + CNT / 2];
        const float keep = up ? a[j + CNT / 2] : a[j];
        a[j] = keep + __shfl_xor_sync(0xffffffffu, send, o);
      }
    } else {
      a[0] = a[0] + __shfl_xor_sync(0xffffffffu, a[0], o);
    }
    rscatter<LV + 1, P>(a, lane);
  }
}

__device__ __forceinline__ void unpack8(const uint4& v, float (&f)[8]) {
  const __half2* h = reinterpret_cast<const __half2*>(&v);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const float2 t = __half22float2(h[i]);
    f[2 * i] = t.x; f[2 * i + 1] = t.y;
  }
}

template <int MT, int WPR, int RPW, int ITERS>
__global__ void __launch_bounds__(256) gemv2_kernel(Jobs jobs, int K) {
  constexpr int WARPS = 8;
  constexpr int RPB = (WARPS / WPR) * RPW;   // rows per block
  constexpr int N = RPW * MT;
  constexpr int P = PadN<N>::P;
  constexpr int LP = Log2<P>::V;
  const int bid = blockIdx.x;
  int ji = 0;
#pragma unroll
  for (int i = 1; i < kMaxJobs; i++)
    if (i < jobs.njobs && bid >= jobs.j[i].blk0) ji = i;
  const Job job = jobs.j[ji];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int wrow = warp / WPR;
  const int wk = warp % WPR;
  const int row0 = (bid - job.blk0) * RPB + wrow * RPW;
  const int chunks = K >> 3;
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
  float acc[N];
#pragma unroll
  for (int i = 0; i < N; i++) acc[i] = 0.f;
#pragma unroll
  for (int it = 0; it < ITERS; it++) {
    const int c = (it * WPR + wk) * 32 + lane;
    if (c < chunks) {
      float wf[RPW][8];
#pragma unroll
      for (int r = 0; r < RPW; r++) unpack8(wreg[r][it], wf[r]);
#pragma unroll
      for (int m = 0; m < MT; m++) {
        const uint4 xv = __ldg(reinterpret_cast<const uint4*>(job.x + (size_t)m * job.ldx) + c);
        float xf[8];
        unpack8(xv, xf);
#pragma unroll
        for (int r = 0; r < RPW; r++)
#pragma unroll
          for (int i = 0; i < 8; i++) acc[r * MT + m] = fmaf(wf[r][i], xf[i], acc[r * MT + m]);
      }
    }
  }
  float a[P];
#pragma unroll
  for (int i = 0; i < P; i++) a[i] = i < N ? acc[i] : 0.f;
  rscatter<0, P>(a, lane);
  int n = 0;
#pragma unroll
  for (int lv = 0; lv < LP; lv++) n += ((lane >> (4 - lv)) & 1) * (P >> (lv + 1));
  const bool writer = (lane & ((1 << (5 - LP)) - 1)) == 0 && n < N;
  const int r = n / MT, m = n % MT;
  if (WPR == 1) {
    const int row = row0 + r;
    if (writer && row < job.n) job.out[(size_t)m * job.ldo + row] = __float2half_rn(a[0]);
  } else {
    __shared__ float red[WARPS][N];
    if (writer) red[warp][n] = a[0];
    __syncthreads();
    if (wk == 0 && lane < N) {
      float v = 0.f;
      for (int k = 0; k < WPR; k++) v += red[wrow * WPR + k][lane];
      const int rr = lane / MT, mm = lane % MT;
      const int row = row0 + rr;
      if (row < job.n) job.out[(size_t)mm * job.ldo + row] = __float2half_rn(v);
    }
  }
}

template <int MT>
bool launch_mt(const Jobs& jobs, int K, int blocks, int wpr, int rpw, cudaStream_t s) {
  const int iters = (K / 8 + wpr * 32 - 1) / (wpr * 32);
#define G2(W, R, I) if (wpr == W && rpw == R && iters == I) { gemv2_kernel<MT, W, R, I><<<blocks, 256, 0, s>>>(jobs, K); return true; }
  G2(8, 1, 2) G2(8, 2, 2) G2(8, 4, 2) G2(4, 1, 3) G2(4, 2, 3)
  G2(8, 1, 1) G2(8, 2, 1) G2(8, 4, 1) G2(4, 1, 1) G2(4, 2, 1) G2(4, 4, 1)
  G2(8, 1, 3) G2(4, 1, 2) G2(4, 2, 2)
#undef G2
  return false;
}
"""

_MAX_N = 16  # accumulators per thread (RPW x MT) allowed in a configuration


def _unit(mt: int) -> str:
    # configurations with RPW x MT > 16 are not instantiated (register pressure)
    return (
        '#include "dg2.cuh"\n'
        f"bool launch_m{mt}(const Jobs& j, int K, int b, int w, int r, cudaStream_t s) {{\n"
        f"  if (r * {mt} > {_MAX_N}) return false;\n"
        f"  return launch_mt<{mt}>(j, K, b, w, r, s);\n}}\n"
    )


_MAIN = (
    r"""
#include "dg2.cuh"
#include <vector>
"""
    + "".join(
        f"bool launch_m{m}(const Jobs&, int, int, int, int, cudaStream_t);\n"
        for m in range(1, 9)
    )
    + r"""
bool dense_gemv2_jobs(std::vector<torch::Tensor> w, std::vector<torch::Tensor> x,
                      std::vector<torch::Tensor> out, int64_t wpr, int64_t rpw) {
  const int nj = (int)w.size();
  TORCH_CHECK(nj >= 1 && nj <= kMaxJobs && x.size() == w.size() && out.size() == w.size());
  const c10::cuda::OptionalCUDAGuard guard(w[0].device());
  Jobs jobs; jobs.njobs = nj;
  const int M = (int)x[0].size(0), K = (int)w[0].size(1);
  TORCH_CHECK(M >= 1 && M <= 8);
  const int rpb = (8 / (int)wpr) * (int)rpw;
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
    blk += ((int)w[i].size(0) + rpb - 1) / rpb;
  }
  cudaStream_t s = at::cuda::getCurrentCUDAStream();
  bool ok = false;
  switch (M) {
    case 1: ok = launch_m1(jobs, K, blk, (int)wpr, (int)rpw, s); break;
    case 2: ok = launch_m2(jobs, K, blk, (int)wpr, (int)rpw, s); break;
    case 3: ok = launch_m3(jobs, K, blk, (int)wpr, (int)rpw, s); break;
    case 4: ok = launch_m4(jobs, K, blk, (int)wpr, (int)rpw, s); break;
    case 5: ok = launch_m5(jobs, K, blk, (int)wpr, (int)rpw, s); break;
    case 6: ok = launch_m6(jobs, K, blk, (int)wpr, (int)rpw, s); break;
    case 7: ok = launch_m7(jobs, K, blk, (int)wpr, (int)rpw, s); break;
    default: ok = launch_m8(jobs, K, blk, (int)wpr, (int)rpw, s); break;
  }
  if (ok) C10_CUDA_KERNEL_LAUNCH_CHECK();
  return ok;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("dense_gemv2_jobs", &dense_gemv2_jobs); }
"""
)

_mod = None
_lock = threading.Lock()


def _ext():
    global _mod
    if _mod is None:
        with _lock:
            if _mod is None:
                from torch.utils.cpp_extension import load

                build = os.path.join(
                    os.environ.get(
                        "VLLM_INKLING2_SM70_EXT_DIR",
                        os.path.expanduser("~/.cache/vllm/inkling2_sm70_ext"),
                    ),
                    "dense_gemv2",
                )
                os.makedirs(build, exist_ok=True)
                files = {"dg2.cuh": _HDR, "dg2_main.cu": _MAIN}
                for m in range(1, 9):
                    files[f"dg2_m{m}.cu"] = _unit(m)
                for name, text in files.items():
                    path = os.path.join(build, name)
                    if not os.path.exists(path) or Path(path).read_text() != text:
                        with open(path, "w") as f:
                            f.write(text)
                srcs = [os.path.join(build, n) for n in files if n.endswith(".cu")]
                _mod = load(
                    "inkling2_dense_gemv2",
                    sources=srcs,
                    extra_cuda_cflags=[
                        "-O3",
                        "-arch=sm_70",
                        "--use_fast_math",
                        "-std=c++17",
                    ],
                    extra_include_paths=[build],
                    build_directory=build,
                    verbose=False,
                )
    return _mod


def gemv2(ws, xs, outs, wpr: int, rpw: int) -> bool:
    """One launch over the jobs; False when this (M, wpr, rpw, K) has no instantiation (caller keeps gen 1)."""
    return _ext().dense_gemv2_jobs(list(ws), list(xs), list(outs), wpr, rpw)


# (n, k, m) -> (warps per row, rows per warp), from `bench_gemv.py` (V100, cold weights). None = keep gen 1 / cuBLAS.
_FORCE = os.environ.get("INK_GEMV2_CFG")  # tuning aid: "wpr,rpw" for every shape


def config(n: int, k: int, m: int):
    if _FORCE:
        w, r = (int(x) for x in _FORCE.split(","))
        return (w, r)
    if k % 8 or m < 1 or m > 8:
        return None
    # bench_gemv.py (V100, cold weights, bit-equal to gen 1 for these): K=2304 weights (qkvr, taps): 8 warps per row, 2 rows per warp
    # from 3 tokens on; K<=768 (wo): 4 warps per row, 2 rows per warp.
    if k >= 2048:
        return (8, 1) if m <= 2 else (8, 2)
    return (4, 2)
