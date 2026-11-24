# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Fused Inkling2 MoE router for SM70 (JIT CUDA extension).

Replaces the ~12-kernel chain of `moe_ref.route` (x.float(), fp32 GEMV, sigmoid,
+bias, topk, gather, sum, div, mul, ids.to(int32)) by ONE kernel for decode
(M <= 16): 128 CTAs of 288 threads, one expert row each (2304 = 288 x 8), fp16 x
and fp32 weight with fp32 accumulation; the last CTA to finish (atomic counter,
self-resetting, CUDA-graph safe) runs the select: sigmoid, +bias top-8 by key
(ties: lower expert id), weights = score / (sum + 1e-20) * route_scale.
Larger M: cuBLAS fp32 GEMM for the logits + the same select as its own kernel.
"""

import os

import torch

_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

#define E 128
#define TOPK 8
#define H 2304
#define NT 288          // H / 8
#define MAXM 16

// Select for one token by one warp: logits row `lg` (E floats).
__device__ __forceinline__ void select_row(const float* lg, const float* __restrict__ bias,
                                           float route_scale, float* w_out, int* id_out, int lane) {
  float s[4], key[4];
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    int e = lane + 32 * i;
    float l = __ldcg(lg + e);
    s[i] = 1.0f / (1.0f + expf(-l));
    key[i] = s[i] + bias[e];
  }
  float sel_s[TOPK];
  int sel_e[TOPK];
  #pragma unroll
  for (int k = 0; k < TOPK; k++) {
    float bk = key[0]; int be = lane; int bi = 0;
    #pragma unroll
    for (int i = 1; i < 4; i++) {
      if (key[i] > bk) { bk = key[i]; be = lane + 32 * i; bi = i; }
    }
    float mk = bk; int me = be;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      float ok = __shfl_xor_sync(0xffffffffu, mk, o);
      int oe = __shfl_xor_sync(0xffffffffu, me, o);
      if (ok > mk || (ok == mk && oe < me)) { mk = ok; me = oe; }
    }
    // owner removes it
    float score = 0.f;
    if (me % 32 == lane) {
      int i = me / 32;
      #pragma unroll
      for (int j = 0; j < 4; j++) if (j == i) { score = s[j]; key[j] = -INFINITY; }
    }
    score = __shfl_sync(0xffffffffu, score, me % 32);
    sel_s[k] = score; sel_e[k] = me;
  }
  float sum = 0.f;
  #pragma unroll
  for (int k = 0; k < TOPK; k++) sum += sel_s[k];
  float den = sum + 1e-20f;
  if (lane < TOPK) {
    float ws = 0.f; int ie = 0;
    #pragma unroll
    for (int k = 0; k < TOPK; k++) if (k == lane) { ws = sel_s[k]; ie = sel_e[k]; }
    w_out[lane] = (ws / den) * route_scale;
    id_out[lane] = ie;
  }
}

__global__ void __launch_bounds__(NT) router_fused_kernel(
    const __half* __restrict__ x, const float* __restrict__ W, const float* __restrict__ bias,
    float* logits, int* counter, float* w_out, int* id_out, int M, float route_scale) {
  const int e = blockIdx.x, t = threadIdx.x;
  const float4* wp = reinterpret_cast<const float4*>(W + (size_t)e * H) + 2 * t;
  float4 w0 = __ldg(wp), w1 = __ldg(wp + 1);
  __shared__ float red[NT / 32][MAXM];
  __shared__ int is_last;
  const int lane = t & 31, warp = t >> 5;
  #pragma unroll
  for (int m = 0; m < MAXM; m++) {
    float acc = 0.f;
    if (m < M) {
      uint4 xv = __ldg(reinterpret_cast<const uint4*>(x + (size_t)m * H) + t);
      const __half2* xh = reinterpret_cast<const __half2*>(&xv);
      float2 a = __half22float2(xh[0]), b = __half22float2(xh[1]);
      float2 c = __half22float2(xh[2]), d = __half22float2(xh[3]);
      acc = w0.x * a.x + w0.y * a.y + w0.z * b.x + w0.w * b.y +
            w1.x * c.x + w1.y * c.y + w1.z * d.x + w1.w * d.y;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) red[warp][m] = acc;
  }
  __syncthreads();
  if (t < M) {
    float v = 0.f;
    #pragma unroll
    for (int w = 0; w < NT / 32; w++) v += red[w][t];
    logits[(size_t)t * E + e] = v;
  }
  __threadfence();
  __syncthreads();
  if (t == 0) {
    int prev = atomicAdd(counter, 1);
    is_last = (prev == (int)gridDim.x - 1);
  }
  __syncthreads();
  if (!is_last) return;
  __threadfence();
  for (int m = warp; m < M; m += NT / 32)
    select_row(logits + (size_t)m * E, bias, route_scale, w_out + m * TOPK, id_out + m * TOPK, lane);
  if (t == 0) *counter = 0;
}

__global__ void select_kernel(const float* logits, const float* __restrict__ bias, float* w_out,
                              int* id_out, int M, float route_scale) {
  int m = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
  if (m >= M) return;
  select_row(logits + (size_t)m * E, bias, route_scale, w_out + m * TOPK, id_out + m * TOPK,
             threadIdx.x & 31);
}

void router_fused(torch::Tensor x, torch::Tensor W, torch::Tensor bias, torch::Tensor logits,
                  torch::Tensor counter, torch::Tensor w_out, torch::Tensor id_out,
                  double route_scale) {
  int M = x.size(0);
  auto stream = at::cuda::getCurrentCUDAStream();
  router_fused_kernel<<<E, NT, 0, stream>>>(
      reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), W.data_ptr<float>(),
      bias.data_ptr<float>(), logits.data_ptr<float>(), counter.data_ptr<int>(),
      w_out.data_ptr<float>(), id_out.data_ptr<int>(), M, (float)route_scale);
}

void router_select(torch::Tensor logits, torch::Tensor bias, torch::Tensor w_out,
                   torch::Tensor id_out, double route_scale) {
  int M = logits.size(0);
  auto stream = at::cuda::getCurrentCUDAStream();
  select_kernel<<<(M + 7) / 8, 256, 0, stream>>>(logits.data_ptr<float>(), bias.data_ptr<float>(),
      w_out.data_ptr<float>(), id_out.data_ptr<int>(), M, (float)route_scale);
}
"""

_CPP = """
void router_fused(torch::Tensor x, torch::Tensor W, torch::Tensor bias, torch::Tensor logits,
                  torch::Tensor counter, torch::Tensor w_out, torch::Tensor id_out,
                  double route_scale);
void router_select(torch::Tensor logits, torch::Tensor bias, torch::Tensor w_out,
                   torch::Tensor id_out, double route_scale);
"""

ENABLED = os.environ.get("VLLM_INKLING2_SM70_FUSED_ROUTER", "1") == "1"
# M > 16: logits as an fp16 tensor-core GEMM with fp32 accumulate/output on an fp16 copy of the
# (bf16-origin) gate weight, instead of x.float() + an fp32 SGEMM.
ROUTER_HMMA = os.environ.get("VLLM_INKLING2_SM70_ROUTER_HMMA", "0") == "1"
MAX_FUSED_M = 16
_E, _H, _TOPK = 128, 2304, 8

_ext = None
_counters: dict[torch.device, torch.Tensor] = {}


def _load():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load_inline

        build = os.environ.get("VLLM_INKLING2_ROUTER_BUILD") or os.path.join(
            os.environ.get(
                "VLLM_INKLING2_SM70_EXT_DIR",
                os.path.expanduser("~/.cache/vllm/inkling2_sm70_ext"),
            ),
            "router",
        )
        os.makedirs(build, exist_ok=True)
        _ext = load_inline(
            name="inkling2_router",
            cpp_sources=[_CPP],
            cuda_sources=[_SRC],
            functions=["router_fused", "router_select"],
            build_directory=build,
            extra_cuda_cflags=["-O3", "-arch=sm_70"],
            verbose=False,
        )
    return _ext


def _logits_hmma(x: torch.Tensor, gate_weight: torch.Tensor) -> torch.Tensor:
    w16 = getattr(gate_weight, "_inkling_w16", None)
    if w16 is None:
        w16 = gate_weight._inkling_w16 = gate_weight.to(torch.float16)
    try:
        return torch.mm(x, w16.t(), out_dtype=torch.float32)
    except (TypeError, RuntimeError):
        return torch.mm(x.float(), w16.float().t())


def supported(x: torch.Tensor, gate_weight: torch.Tensor, top_k: int) -> bool:
    return (
        ENABLED
        and x.dtype == torch.float16
        and gate_weight.dtype == torch.float32
        and tuple(gate_weight.shape) == (_E, _H)
        and x.shape[1] == _H
        and top_k == _TOPK
        and x.is_contiguous()
    )


def fused_route(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: torch.Tensor,
    route_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(weights float32 [M, 8], ids int32 [M, 8])`, same semantics as `moe_ref.route`.
    The caller checks `supported`."""
    ext = _load()
    m = x.shape[0]
    dev = x.device
    w_out = torch.empty((m, _TOPK), dtype=torch.float32, device=dev)
    ids = torch.empty((m, _TOPK), dtype=torch.int32, device=dev)
    if m <= MAX_FUSED_M:
        counter = _counters.get(dev)
        if counter is None:
            counter = _counters[dev] = torch.zeros(1, dtype=torch.int32, device=dev)
        logits = torch.empty((m, _E), dtype=torch.float32, device=dev)
        ext.router_fused(
            x, gate_weight, gate_bias, logits, counter, w_out, ids, route_scale
        )
    else:
        if ROUTER_HMMA:
            logits = _logits_hmma(x, gate_weight)
        else:
            logits = torch.nn.functional.linear(x.float(), gate_weight)
        ext.router_select(logits, gate_bias, w_out, ids, route_scale)
    return w_out, ids
