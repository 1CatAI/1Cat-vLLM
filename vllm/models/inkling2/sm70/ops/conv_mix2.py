# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Fused (all-reduce +) short conv + AttnRes mix kernel, second generation (SM70,
CUDA JIT). Same arithmetic, thread layout and push all-reduce wire protocol as
`fused_ar_mix.py` (it uses that module's IPC buffers); see the header of the
CUDA source for what changed: push first, a code-warming pass during the wait
for the peers, and the all-reduce kept inside the kernel when a request brings
several new tokens (the speculative verify step). Generated from
fused_ar_mix.py by work/spd-attn/gen_conv_mix2.py."""

import os
from pathlib import Path

import torch

from .fused_ar_mix import HIDDEN, FusedAR

CUDA_SRC = r"""
// SPDX-License-Identifier: Apache-2.0
// Fused sublayer kernel for the SM70 Inkling2 port, second generation (session spd-attn).
// Arithmetic, thread layout (288 threads x 8 fp16) and the push all-reduce wire protocol are
// those of fused_ar_mix.py (outputs are bit-identical); the control flow differs:
//  * AR: this rank's partial sums are pushed to the 3 peers before anything else (the own
//    slot is neither written nor read), then the kernel runs the whole conv + mix once on
//    its own partial sums while the peers' data is on its way: that pass only warms the
//    SM's instruction cache (stores are off). The real pass after the poll runs warm code.
//  * AR with several new tokens per request (speculative verify, tiny prefills): taps that
//    lie inside the batch are read from the conv cache after a grid barrier among the
//    token CTAs (each CTA writes its reduced row to the cache first), so the all-reduce
//    stays inside this kernel for 1..16 tokens.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

constexpr int D = 2304;
constexpr int NTHR = 288;  // 8 halves per thread
constexpr int MAXT = 16;
constexpr int NRANK = 4;
constexpr int NEPOCH = 2;
constexpr int SIG_BYTES = 256;
constexpr uint16_t SENT = 0x7f7f;
constexpr size_t BUF_BYTES =
    SIG_BYTES + (size_t)NEPOCH * NRANK * MAXT * NTHR * 16;

struct Params {
  const __half* x;  // [T, D]; AR: local partial sums, else already reduced
  __half* cache;
  const __half* weight;  // [D, 4]
  __half* conv_out;      // [T, D]
  const __half* res;     // nullable, may alias conv_out
  const __half* addend;  // nullable
  const int64_t* pos;
  const int32_t* seq_idx;
  const int64_t* slot;
  const int32_t* block_table;
  const int32_t* qstart;
  const __half* stack;
  const __half* last;  // nullable
  const float* queries;
  const __half* normw;  // nullable (no norm)
  void* out;            // [NQ, T, D] half (norm) or float
  int64_t stride_blk, stride_h, stride_n, stride_s;
  int T, stride_bt, max_blocks, N, WS, OFF, H;
  float scale, eps, norm_eps;
  int normalize;
  char* peers[NRANK];
  int rank;
  int inbatch;  // AR: some token has a tap inside the batch (max query length > 1)
  int rehearse; // AR: run a code-warming pass on the own partial sums before the poll
  int nshift;   // log2(N)
  uint32_t ws_recip;  // ceil(2^32 / WS)
  int* ctr;     // AR + inbatch: barrier counter (self-resetting)
  int* fault;   // sticky barrier-timeout flag
};

constexpr int SPIN_MAX = 1 << 22;

__device__ __forceinline__ int ld_volatile32(const int* p) {
  int v;
  asm volatile("ld.volatile.global.s32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

__device__ __forceinline__ void u2f(const uint4 v, float f[8]) {
  const __half2* h = reinterpret_cast<const __half2*>(&v);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    float2 t = __half22float2(h[i]);
    f[2 * i] = t.x;
    f[2 * i + 1] = t.y;
  }
}

__device__ __forceinline__ uint4 f2u(const float f[8]) {
  uint4 v;
  __half2* h = reinterpret_cast<__half2*>(&v);
#pragma unroll
  for (int i = 0; i < 4; i++) h[i] = __floats2half2_rn(f[2 * i], f[2 * i + 1]);
  return v;
}

__device__ __forceinline__ float rsqrt_approx(float x) {
  float r;
  asm("rsqrt.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

__device__ __forceinline__ uint4 ld_volatile16(const void* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

__device__ __forceinline__ void st_volatile16(void* p, const uint4& v) {
  asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
}

__device__ __forceinline__ bool has_sentinel(const uint4& v) {
  const uint16_t* h = reinterpret_cast<const uint16_t*>(&v);
  bool s = false;
#pragma unroll
  for (int i = 0; i < 8; i++) s |= (h[i] == SENT);
  return s;
}

// Warp shuffles as one inline SHFL each (what Triton emits). nvcc lowers __shfl_*_sync on
// sm_70 to a call into a ~10-instruction helper per shuffle; every lane of the warp must
// execute these together (they do at every call site: the control flow around them is
// uniform per warp).
__device__ __forceinline__ float shx(float v, int o) {
  float r;
  asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(r) : "f"(v), "r"(o));
  return r;
}
// Sum K per-thread values over the 288-thread block (9 warps, fixed order).
template <int K>
__device__ __forceinline__ void block_sum(float (&v)[K], float* smem) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
    for (int k = 0; k < K; k++) v[k] += shx(v[k], off);
  }
  if (lane == 0) {
#pragma unroll
    for (int k = 0; k < K; k++) smem[warp * K + k] = v[k];
  }
  __syncthreads();
#pragma unroll
  for (int k = 0; k < K; k++) {
    float s = smem[k];
#pragma unroll
    for (int w = 1; w < NTHR / 32; w++) s += smem[w * K + k];
    v[k] = s;
  }
}

template <int NS, int NQ, bool AR>
__global__ void __launch_bounds__(NTHR) conv_mix2_kernel(const Params p) {
  constexpr int NSRC = NS + 1;
  constexpr int K1 = NSRC * (1 + NQ);
  __shared__ float red1[(NTHR / 32) * K1];
  __shared__ float red2[(NTHR / 32) * NQ];

  const int t = blockIdx.x, tid = threadIdx.x, c0 = tid * 8;
  const int64_t row = (int64_t)t * D;

  // ---- AR: push this rank's partial sums to the peers before anything else ----
  uint4 pv = make_uint4(0, 0, 0, 0);
  uint32_t epoch = 0;
  char* local = nullptr;
  if constexpr (AR) {
    pv = *reinterpret_cast<const uint4*>(p.x + row + c0);
    uint16_t* ph = reinterpret_cast<uint16_t*>(&pv);
#pragma unroll
    for (int i = 0; i < 8; i++)
      if (ph[i] == SENT) ph[i] = 0x7e00u;
    local = p.peers[p.rank];
    epoch = reinterpret_cast<const volatile uint32_t*>(local)[t];
    const size_t mine = SIG_BYTES +
                        (((size_t)epoch * NRANK + p.rank) * MAXT + t) * NTHR * 16 +
                        (size_t)tid * 16;
#pragma unroll
    for (int d = 0; d < NRANK; d++)
      if (d != p.rank) st_volatile16(p.peers[d] + mine, pv);
  }

  // ---- metadata + prefetch (independent of the all-reduce) ----
  const int64_t pos = p.pos[t];
  const int64_t slot = p.slot[t];
  const int req = p.seq_idx[t];
  const int qs = p.qstart[t];
  const bool valid = slot >= 0;
  // c0 / WS without a runtime division (exact for c0, WS < 2^16)
  const int head = min((int)(((uint64_t)(uint32_t)c0 * p.ws_recip) >> 32), p.H - 1);
  const int64_t cbase = (int64_t)head * p.stride_h + p.OFF + (c0 - head * p.WS);
  const int64_t nmask = ((int64_t)1 << p.nshift) - 1;

  uint4 tapv[3];
  const uint4* tapp[3];
#pragma unroll
  for (int l = 0; l < 3; l++) {
    const int lag = l + 1;
    const int64_t src = pos - lag;
    const bool in_win = valid && src >= 0;
    const bool cached = in_win && (t - lag < qs);
    uint4 v = make_uint4(0, 0, 0, 0);
    tapp[l] = nullptr;
    if (in_win && (AR || cached)) {
      const int lblk = (int)min((int64_t)(src >> p.nshift), (int64_t)(p.max_blocks - 1));
      const int64_t blk = p.block_table[(int64_t)req * p.stride_bt + lblk];
      const uint4* cp = reinterpret_cast<const uint4*>(p.cache + blk * p.stride_blk +
                                                       (src & nmask) * p.stride_n + cbase);
      if (cached) v = *cp;
      else tapp[l] = cp;  // AR: inside the batch, read after the barrier
    } else if (in_win) {
      v = *reinterpret_cast<const uint4*>(p.x + (int64_t)(t - lag) * D + c0);
    }
    tapv[l] = v;
  }
  uint4 wv[4];
#pragma unroll
  for (int i = 0; i < 4; i++)
    wv[i] = *reinterpret_cast<const uint4*>(p.weight + (int64_t)c0 * 4 + i * 8);
  uint4 sv[NS];
#pragma unroll
  for (int n = 0; n < NS; n++)
    sv[n] = *reinterpret_cast<const uint4*>(p.stack + n * p.stride_s + row + c0);
  uint4 lastv = make_uint4(0, 0, 0, 0), resv = lastv, addv = lastv;
  if (p.last) lastv = *reinterpret_cast<const uint4*>(p.last + row + c0);
  if (p.addend) addv = *reinterpret_cast<const uint4*>(p.addend + row + c0);
  if (p.res) resv = *reinterpret_cast<const uint4*>(p.res + row + c0);
  float qv[NQ][8];
#pragma unroll
  for (int q = 0; q < NQ; q++) {
    const float4 a = *reinterpret_cast<const float4*>(p.queries + (int64_t)q * D + c0);
    const float4 b = *reinterpret_cast<const float4*>(p.queries + (int64_t)q * D + c0 + 4);
    qv[q][0] = a.x; qv[q][1] = a.y; qv[q][2] = a.z; qv[q][3] = a.w;
    qv[q][4] = b.x; qv[q][5] = b.y; qv[q][6] = b.z; qv[q][7] = b.w;
  }
  uint4 nwv[NQ];
#pragma unroll
  for (int q = 0; q < NQ; q++) {
    nwv[q] = make_uint4(0, 0, 0, 0);
    if (p.normw) nwv[q] = *reinterpret_cast<const uint4*>(p.normw + (int64_t)q * D + c0);
  }

  // pass 0 (AR, optional): the whole kernel on this rank's own partial sums, stores off.
  // It overlaps the wait for the peers and leaves the code in the instruction cache: it
  // pays when the peers are late (rank skew), and costs a second warm pass when they are not.
#pragma unroll 1
  for (int pass = (AR && p.rehearse) ? 0 : 1; pass < 2; pass++) {
    // ---- current row: all-reduce or plain load ----
    uint4 curv = pv;
    if constexpr (AR) {
      if (pass) {
        uint4 pvs[NRANK];
        while (true) {
          bool empty = false;
#pragma unroll
          for (int s = 0; s < NRANK; s++) {
            const size_t off = SIG_BYTES +
                               (((size_t)epoch * NRANK + s) * MAXT + t) * NTHR * 16 +
                               (size_t)tid * 16;
            pvs[s] = ld_volatile16(local + off);
            if (s == p.rank) pvs[s] = pv;
            empty |= has_sentinel(pvs[s]);
          }
          if (!empty) break;
        }
        float a[8];
        u2f(pvs[0], a);
#pragma unroll
        for (int s = 1; s < NRANK; s++) {
          float b[8];
          u2f(pvs[s], b);
#pragma unroll
          for (int j = 0; j < 8; j++) a[j] += b[j];
        }
        curv = f2u(a);
        uint4 empty_v;
        empty_v.x = empty_v.y = empty_v.z = empty_v.w = 0x7f7f7f7fu;
#pragma unroll
        for (int s = 0; s < NRANK; s++) {
          const size_t off = SIG_BYTES +
                             (((size_t)epoch * NRANK + s) * MAXT + t) * NTHR * 16 +
                             (size_t)tid * 16;
          if (s != p.rank) st_volatile16(local + off, empty_v);
        }
      }
    } else {
      curv = *reinterpret_cast<const uint4*>(p.x + row + c0);
    }
    if (pass && valid) {
      *reinterpret_cast<uint4*>(p.cache + (slot >> p.nshift) * p.stride_blk +
                                (slot & nmask) * p.stride_n + cbase) = curv;
    }
    if constexpr (AR) {
      if (pass && p.inbatch) {
        // every token CTA has written its reduced row to the conv cache: read the taps
        // that lie inside the batch from there
        __threadfence();
        __syncthreads();
        if (tid == 0) {
          atomicInc(reinterpret_cast<unsigned int*>(p.ctr), (unsigned int)(p.T - 1));
          int it = 0;
          while (ld_volatile32(p.ctr) != 0 && ++it < SPIN_MAX) {
          }
          if (it >= SPIN_MAX) *p.fault = 1;
        }
        __syncthreads();
#pragma unroll
        for (int l = 0; l < 3; l++)
          if (tapp[l]) tapv[l] = __ldcg(tapp[l]);
      }
    }

    // ---- short conv (taps: lag 3, 2, 1 from cache/x, lag 0 = current) ----
    float cur[8], tap[4][8], w[8][4];
    u2f(curv, cur);
    u2f(tapv[2], tap[0]);
    u2f(tapv[1], tap[1]);
    u2f(tapv[0], tap[2]);
  #pragma unroll
    for (int j = 0; j < 8; j++) tap[3][j] = valid ? cur[j] : 0.f;
    {
      float wf[4][8];
  #pragma unroll
      for (int i = 0; i < 4; i++) u2f(wv[i], wf[i]);
  #pragma unroll
      for (int j = 0; j < 8; j++)
  #pragma unroll
        for (int k = 0; k < 4; k++) w[j][k] = wf[(j * 4 + k) / 8][(j * 4 + k) % 8];
    }
    float y[8];
  #pragma unroll
    for (int j = 0; j < 8; j++) {
      float acc = 0.f;
  #pragma unroll
      for (int k = 0; k < 4; k++) acc = __fmaf_rn(tap[k][j], w[j][k], acc);
      y[j] = __fmul_rn(__fadd_rn(acc, cur[j]), p.scale);
    }
    uint4 yv = f2u(y);
    u2f(yv, y);  // fp16-rounded conv output
    if (p.addend) {
      float a[8], r[8];
      u2f(addv, a);
      if (p.res) {
        u2f(resv, r);
  #pragma unroll
        for (int j = 0; j < 8; j++) a[j] = __fadd_rn(r[j], a[j]);
      }
  #pragma unroll
      for (int j = 0; j < 8; j++) y[j] = __fadd_rn(a[j], y[j]);
      yv = f2u(y);
      u2f(yv, y);
    }
    if (pass) *reinterpret_cast<uint4*>(p.conv_out + row + c0) = yv;

    // ---- AttnRes mix ----
    float sf[NSRC][8];
  #pragma unroll
    for (int n = 0; n < NS; n++) u2f(sv[n], sf[n]);
    if (p.last) {
      float l[8];
      u2f(lastv, l);
  #pragma unroll
      for (int j = 0; j < 8; j++) sf[NS][j] = __fadd_rn(l[j], y[j]);
    } else {
  #pragma unroll
      for (int j = 0; j < 8; j++) sf[NS][j] = y[j];
    }

    float red[K1];
  #pragma unroll
    for (int n = 0; n < NSRC; n++) {
      float ss = 0.f;
  #pragma unroll
      for (int j = 0; j < 8; j++) ss = __fmaf_rn(sf[n][j], sf[n][j], ss);
      red[n] = ss;
  #pragma unroll
      for (int q = 0; q < NQ; q++) {
        float dt = 0.f;
  #pragma unroll
        for (int j = 0; j < 8; j++) dt = __fmaf_rn(qv[q][j], sf[n][j], dt);
        red[NSRC * (1 + q) + n] = dt;
      }
    }
    block_sum<K1>(red, red1);

    float inv_rms[NSRC];
  #pragma unroll
    for (int n = 0; n < NSRC; n++)
      inv_rms[n] = rsqrt_approx(__fmaf_rn(red[n], 1.0f / D, p.eps));

    float acc[NQ][8];
  #pragma unroll
    for (int q = 0; q < NQ; q++) {
      float logit[NSRC], mx = -INFINITY;
  #pragma unroll
      for (int n = 0; n < NSRC; n++) {
        logit[n] = red[NSRC * (1 + q) + n] * inv_rms[n];
        mx = fmaxf(mx, logit[n]);
      }
      float pr[NSRC], sum = 0.f;
  #pragma unroll
      for (int n = 0; n < NSRC; n++) {
        pr[n] = expf(logit[n] - mx);
        sum += pr[n];
      }
      const float inv_sum = __fdiv_rn(1.0f, sum);
  #pragma unroll
      for (int n = 0; n < NSRC; n++)
        if (p.normalize) pr[n] *= inv_rms[n];
  #pragma unroll
      for (int j = 0; j < 8; j++) {
        float a = 0.f;
  #pragma unroll
        for (int n = 0; n < NSRC; n++) a = __fmaf_rn(pr[n], sf[n][j], a);
        acc[q][j] = a * inv_sum;
      }
    }

    if (p.normw) {
      float r2[NQ];
  #pragma unroll
      for (int q = 0; q < NQ; q++) {
        float s = 0.f;
  #pragma unroll
        for (int j = 0; j < 8; j++) s = __fmaf_rn(acc[q][j], acc[q][j], s);
        r2[q] = s;
      }
      block_sum<NQ>(r2, red2);
  #pragma unroll
      for (int q = 0; q < NQ; q++) {
        const float out_rstd = rsqrt_approx(__fdiv_rn(r2[q], (float)D) + p.norm_eps);
        float nw[8];
        u2f(nwv[q], nw);
  #pragma unroll
        for (int j = 0; j < 8; j++) acc[q][j] = (acc[q][j] * out_rstd) * nw[j];
        if (pass) *reinterpret_cast<uint4*>(reinterpret_cast<__half*>(p.out) +
                                            ((int64_t)q * p.T + t) * D + c0) = f2u(acc[q]);
      }
    } else {
  #pragma unroll
      for (int q = 0; q < NQ; q++) {
        float* o = reinterpret_cast<float*>(p.out) + ((int64_t)q * p.T + t) * D + c0;
        if (pass) {
          *reinterpret_cast<float4*>(o) = make_float4(acc[q][0], acc[q][1], acc[q][2], acc[q][3]);
          *reinterpret_cast<float4*>(o + 4) = make_float4(acc[q][4], acc[q][5], acc[q][6], acc[q][7]);
        }
      }
    }

  }

  if constexpr (AR) {
    __syncthreads();
    if (tid == 0) reinterpret_cast<volatile uint32_t*>(p.peers[p.rank])[t] = (epoch + 1) % NEPOCH;
  }
}

template <int NS, int NQ>
void launch(const Params& p, bool ar, cudaStream_t stream) {
  if (ar)
    conv_mix2_kernel<NS, NQ, true><<<p.T, NTHR, 0, stream>>>(p);
  else
    conv_mix2_kernel<NS, NQ, false><<<p.T, NTHR, 0, stream>>>(p);
}

template <int NS>
void launch_q(const Params& p, int nq, bool ar, cudaStream_t s) {
  if (nq == 1) launch<NS, 1>(p, ar, s);
  else if (nq == 3) launch<NS, 3>(p, ar, s);
  else TORCH_CHECK(false, "unsupported number of queries ", nq);
}

}  // namespace

#define CHECK_HALF(t) TORCH_CHECK((t).scalar_type() == at::kHalf, #t " must be fp16")

// Returns nothing; `out` is [NQ, T, D] fp16 (normw given) or fp32 (normw absent).
void conv_mix(torch::Tensor x, torch::Tensor cache, torch::Tensor weight,
              torch::Tensor conv_out, c10::optional<torch::Tensor> res,
              c10::optional<torch::Tensor> addend, torch::Tensor pos,
              torch::Tensor seq_idx, torch::Tensor slot, torch::Tensor block_table,
              torch::Tensor qstart, torch::Tensor stack,
              c10::optional<torch::Tensor> last, torch::Tensor queries,
              c10::optional<torch::Tensor> normw, torch::Tensor out, int64_t off,
              int64_t ws, double eps, double norm_eps, bool normalize,
              double scale, int64_t rank, std::vector<int64_t> peers, bool inbatch,
              bool rehearse, torch::Tensor ctr, torch::Tensor fault) {
  const at::cuda::CUDAGuard guard(x.device());
  CHECK_HALF(x); CHECK_HALF(cache); CHECK_HALF(weight); CHECK_HALF(conv_out); CHECK_HALF(stack);
  const int T = x.size(0);
  TORCH_CHECK(T >= 1 && T <= MAXT && x.size(1) == D, "bad x shape");
  TORCH_CHECK(x.is_contiguous() && conv_out.is_contiguous() && weight.is_contiguous());
  TORCH_CHECK(stack.stride(1) == D && stack.stride(2) == 1);
  TORCH_CHECK(cache.stride(3) == 1 && cache.stride(0) % 8 == 0 && cache.stride(1) % 8 == 0 &&
              cache.stride(2) % 8 == 0 && off % 8 == 0 && ws % 8 == 0 &&
              reinterpret_cast<uintptr_t>(cache.data_ptr()) % 16 == 0);
  TORCH_CHECK(pos.scalar_type() == at::kLong && slot.scalar_type() == at::kLong);
  TORCH_CHECK(seq_idx.scalar_type() == at::kInt && block_table.scalar_type() == at::kInt &&
              qstart.scalar_type() == at::kInt);
  TORCH_CHECK(queries.scalar_type() == at::kFloat && queries.is_contiguous());
  const int ns = stack.size(0), nq = queries.size(0);
  Params p;
  p.x = reinterpret_cast<const __half*>(x.data_ptr());
  p.cache = reinterpret_cast<__half*>(cache.data_ptr());
  p.weight = reinterpret_cast<const __half*>(weight.data_ptr());
  p.conv_out = reinterpret_cast<__half*>(conv_out.data_ptr());
  p.res = res ? reinterpret_cast<const __half*>(res->data_ptr()) : nullptr;
  p.addend = addend ? reinterpret_cast<const __half*>(addend->data_ptr()) : nullptr;
  TORCH_CHECK(!p.res || p.addend, "residual needs addend");
  p.pos = pos.data_ptr<int64_t>();
  p.seq_idx = seq_idx.data_ptr<int32_t>();
  p.slot = slot.data_ptr<int64_t>();
  p.block_table = block_table.data_ptr<int32_t>();
  p.qstart = qstart.data_ptr<int32_t>();
  p.stack = reinterpret_cast<const __half*>(stack.data_ptr());
  p.last = last ? reinterpret_cast<const __half*>(last->data_ptr()) : nullptr;
  p.queries = queries.data_ptr<float>();
  p.normw = normw ? reinterpret_cast<const __half*>(normw->data_ptr()) : nullptr;
  p.out = out.data_ptr();
  p.stride_blk = cache.stride(0);
  p.stride_h = cache.stride(1);
  p.stride_n = cache.stride(2);
  p.stride_s = stack.stride(0);
  p.T = T;
  p.stride_bt = block_table.stride(0);
  p.max_blocks = block_table.size(1);
  p.N = cache.size(2);
  p.WS = ws;
  p.OFF = off;
  p.H = cache.size(1);
  p.scale = (float)scale;
  p.eps = (float)eps;
  p.norm_eps = (float)norm_eps;
  p.normalize = normalize ? 1 : 0;
  p.rank = (int)rank;
  const bool ar = !peers.empty();
  p.inbatch = inbatch ? 1 : 0;
  p.rehearse = rehearse ? 1 : 0;
  int ns_ = 0;
  while ((1 << ns_) < cache.size(2)) ns_++;
  TORCH_CHECK((1 << ns_) == cache.size(2), "conv cache block size must be a power of 2");
  p.nshift = ns_;
  TORCH_CHECK(ws > 0 && ws < 65536 && x.size(1) < 65536);
  p.ws_recip = (uint32_t)((((uint64_t)1 << 32) + ws - 1) / ws);
  p.ctr = ctr.data_ptr<int>();
  p.fault = fault.data_ptr<int>();
  for (int i = 0; i < NRANK; i++) p.peers[i] = nullptr;
  if (ar) {
    TORCH_CHECK(peers.size() == NRANK && rank >= 0 && rank < NRANK);
    for (int i = 0; i < NRANK; i++) p.peers[i] = reinterpret_cast<char*>(peers[i]);
  }
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  switch (ns) {
    case 1: launch_q<1>(p, nq, ar, stream); break;
    case 2: launch_q<2>(p, nq, ar, stream); break;
    case 3: launch_q<3>(p, nq, ar, stream); break;
    case 4: launch_q<4>(p, nq, ar, stream); break;
    case 5: launch_q<5>(p, nq, ar, stream); break;
    case 6: launch_q<6>(p, nq, ar, stream); break;
    default: TORCH_CHECK(false, "unsupported stack size ", ns);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("conv_mix", &conv_mix, "fused (all-reduce +) conv + AttnRes mix, second generation");
}
"""

_EXT = None
_STATE: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
# 1: with the all-reduce inside, run a code-warming pass before polling the peers
_REHEARSE = os.environ.get("VLLM_INKLING2_SM70_CONV_MIX_REHEARSE", "0") == "1"


def _ext():
    """JIT-build (once) and return the CUDA extension (sm_70)."""
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load

        build_dir = os.path.join(
            os.environ.get(
                "VLLM_INKLING2_SM70_EXT_DIR",
                os.path.expanduser("~/.cache/vllm/inkling2_sm70_ext"),
            ),
            "conv_mix2",
        )
        os.makedirs(build_dir, exist_ok=True)
        src = os.path.join(build_dir, "conv_mix2.cu")
        if not os.path.exists(src) or Path(src).read_text() != CUDA_SRC:
            tmp = f"{src}.{os.getpid()}"
            with open(tmp, "w") as f:
                f.write(CUDA_SRC)
            os.replace(tmp, src)
        _EXT = load(
            name="inkling2_conv_mix2",
            sources=[src],
            build_directory=build_dir,
            extra_cuda_cflags=[
                "-O3",
                "-gencode=arch=compute_70,code=sm_70",
                "-fmad=false",
                "--expt-relaxed-constexpr",
            ],
            verbose=bool(os.environ.get("VLLM_INKLING2_SM70_EXT_VERBOSE")),
        )
    return _EXT


def _state(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    idx = (
        device.index
        if device.index is not None
        else torch.accelerator.current_device_index()
    )
    st = _STATE.get(idx)
    if st is None:
        st = (
            torch.zeros(1, dtype=torch.int32, device=device),  # barrier counter
            torch.zeros(1, dtype=torch.int32, device=device),  # barrier timeout flag
        )
        _STATE[idx] = st
    return st


def prepare(device: torch.device) -> None:
    """Build the extension and allocate the barrier state eagerly (never inside
    CUDA graph capture)."""
    _ext()
    _state(device)


def fault(device: torch.device) -> bool:
    return bool(_state(device)[1].item())


def conv_mix2_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    cache: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    seq_idx: torch.Tensor,
    slot_mapping: torch.Tensor,
    query_start: torch.Tensor,
    stream_offset: int,
    stream_width: int,
    conv_out: torch.Tensor,
    residual: torch.Tensor | None,
    addend: torch.Tensor | None,
    stack: torch.Tensor,
    last: torch.Tensor | None,
    queries: torch.Tensor,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
    normalize_values: bool = True,
    eps: float = 1e-6,
    ar: "FusedAR | None" = None,
    inbatch: bool = False,
) -> torch.Tensor:
    """As `fused_ar_mix.conv_mix_cuda`. With `ar`, `x` holds this rank's partial
    sums; `inbatch` must be set when a token of the batch can have a conv tap
    on another token of the batch (max query length > 1)."""
    num_tokens = x.shape[0]
    out = torch.empty(
        (queries.shape[0], num_tokens, HIDDEN),
        dtype=torch.float16 if norm_weight is not None else torch.float32,
        device=x.device,
    )
    ctr, fault_flag = _state(x.device)
    _ext().conv_mix(
        x,
        cache,
        weight,
        conv_out,
        residual,
        addend,
        positions,
        seq_idx,
        slot_mapping,
        block_table,
        query_start,
        stack,
        last,
        queries,
        norm_weight,
        out,
        stream_offset,
        stream_width,
        eps,
        norm_eps,
        normalize_values,
        (weight.shape[1] + 1) ** -0.5,
        0 if ar is None else ar.rank,
        [] if ar is None else ar.peers,
        inbatch and ar is not None,
        _REHEARSE,
        ctr,
        fault_flag,
    )
    return out
