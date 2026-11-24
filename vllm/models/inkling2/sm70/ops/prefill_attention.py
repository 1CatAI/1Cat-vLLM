# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Tensor-core prefill attention for Inkling2 on SM70.

Replaces the Triton `rel_attention` (scalar-FMA `tl.dot` on SM70) plus
`gate_sink` for every batch with more than 8 query tokens per request: a
CUDA flash-attention kernel on WMMA m16n16k16 (fp16 inputs, fp32 accumulate)
with the semantics of `_rel_attention_kernel` (see `rel_attention.py`):

- one CTA per (16 * NTT query tokens, KV head, request); a warp owns 16
  consecutive tokens of one of the 3 query heads of the KV head, so a K/V
  tile staged in shared memory serves 3 * NTT warps;
- keys stream through shared memory in tiles of BN, straight from the paged
  fp16 cache `(blocks, page, H_kv, 2 * D)` (K | V of a head side by side);
- `S = Q K^T` and `O += P V` run on tensor cores, the online softmax runs on
  CUDA cores out of a per-warp shared-memory copy of S (fp32);
- the relative bias `fp16(r . rel_proj[:, pos_q - pos_k])` is computed per tile
  on tensor cores: a 16 x 16 R tile times the 16 x 64 window of rel_proj
  columns covering every distance of the tile, then a diagonal gather from
  shared memory (no sheared fp16 table in global memory);
- causal and sliding-window masking, then the epilogue folds the output
  gate and the sink: `o / l * sigmoid(gate) * sigmoid(lse - sink)`, written
  as fp16 `(T, H * D)`. No LSE or fp32 output tensor is materialised.
"""

import os

import torch

from vllm.triton_utils import tl, triton

from .norm import _l2_clip

_EXT = None

_CPP_SRC = r"""
void prefill_attention_cuda(
    torch::Tensor q, torch::Tensor r, torch::Tensor proj, torch::Tensor kv, torch::Tensor block_table,
    torch::Tensor seq_lens, torch::Tensor cu_seqlens_q, torch::Tensor gate,
    torch::Tensor sinks, torch::Tensor out, int64_t num_reqs, int64_t max_q,
    int64_t window, int64_t extent, double scale, int64_t variant);
"""

_CUDA_SRC = r"""
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <cstdint>

using namespace nvcuda;

namespace {

constexpr int D = 128;
constexpr int G = 3;           // query heads per KV head
constexpr int LDK = D + 8;     // smem pitch (halves) of K / V tiles
constexpr float LOG2E = 1.4426950408889634f;
constexpr float LN2 = 0.6931471805599453f;

__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + ex2(-x * LOG2E)); }

// Row / column of accumulator element e (0..7) of lane `lane` in a 16x16 tile.
__device__ __forceinline__ int acc_row(int lane, int e) {
  return ACC_ROW_EXPR;
}
__device__ __forceinline__ int acc_col(int lane, int e) {
  return ACC_COL_EXPR;
}

template <int NTT, int BN, int MINB>
__global__ void __launch_bounds__(96 * NTT, MINB) prefill_attn_kernel(
    const __half* __restrict__ Q, const __half* __restrict__ KV,
    const __half* __restrict__ Rr, const __half* __restrict__ Proj, const int* __restrict__ BT,
    const int* __restrict__ SeqLens, const int* __restrict__ CuQ,
    const __half* __restrict__ Gate, const __half* __restrict__ Sink,
    __half* __restrict__ Out, int H, int64_t stride_kv_blk, int64_t stride_kv_slot,
    int64_t stride_kv_h, int64_t stride_bt, int64_t stride_gate,
    int64_t stride_sink, int page, int E, int window, float scale) {
  constexpr int NW = 3 * NTT;
  constexpr int NT = 32 * NW;
  constexpr int LDS = BN + 4;     // fp32 S pitch
  constexpr int LDP = BN + 8;     // half P pitch
  constexpr int NF = BN / 16;     // key fragments per tile
  constexpr int NLD = (BN * 32 + NT - 1) / NT;  // int4 loads per thread per tile
  constexpr int NBF = (BN + 45) / 16;   // 16-wide windows of rel_proj columns per tile
  constexpr int LDM = NBF * 16 + 4;     // fp32 pitch of the bias tile

  extern __shared__ __align__(128) unsigned char smem_raw[];
  __half* sK = reinterpret_cast<__half*>(smem_raw);
  __half* sV = sK + BN * LDK;
  float* sS = reinterpret_cast<float*>(sV + BN * LDK);
  __half* sP = reinterpret_cast<__half*>(sS + NW * 16 * LDS);
  float* sAux = reinterpret_cast<float*>(sP + NW * 16 * LDP);  // NW x 3 x 16
  float* sM = sAux + NW * 48;                                   // NW x 16 x LDM
  __half* sR = reinterpret_cast<__half*>(sM + NW * 16 * LDM);   // NW x 16 x 24

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int req = blockIdx.z;
  const int kvh = blockIdx.y;
  const int q_start = CuQ[req];
  const int q_len = CuQ[req + 1] - q_start;
  const int first_tok = blockIdx.x * (16 * NTT);
  if (first_tok >= q_len) return;
  const int seqlen_k = SeqLens[req];
  const int past = seqlen_k - q_len;

  const int g = warp % 3;
  const int wt = warp / 3;
  const int tok0 = first_tok + wt * 16;
  const int head = kvh * G + g;
  const int64_t qrow = (int64_t)H * D;  // Q / Out row pitch

  float* sSw = sS + warp * 16 * LDS;
  __half* sPw = sP + warp * 16 * LDP;
  float* sAlpha = sAux + warp * 48;
  float* sL = sAlpha + 16;
  float* sSink = sAlpha + 32;
  float* sMw = sM + warp * 16 * LDM;
  __half* sRw = sR + warp * 16 * 24;

  // ---- Q fragments (rows of this warp: 16 consecutive tokens of one head).
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> qa[D / 16];
#pragma unroll
  for (int kk = 0; kk < D / 16; ++kk) {
    const int row = lane >> 1, hf = lane & 1;
    const int tok = tok0 + row;
    int4 v = make_int4(0, 0, 0, 0);
    if (tok < q_len)
      v = *reinterpret_cast<const int4*>(Q + (int64_t)(q_start + tok) * qrow + head * D +
                                         kk * 16 + hf * 8);
    *reinterpret_cast<int4*>(sPw + row * 24 + hf * 8) = v;
    __syncwarp();
    wmma::load_matrix_sync(qa[kk], sPw, 24);
    __syncwarp();
  }

  // ---- R rows of this warp (clipped r, 16 halves per token and head), staged for the bias MMA.
  {
    const int row = lane >> 1, hf = lane & 1;
    const int tok = tok0 + row;
    int4 v = make_int4(0, 0, 0, 0);
    if (tok < q_len)
      v = *reinterpret_cast<const int4*>(Rr + (int64_t)(q_start + tok) * H * 16 + head * 16 + hf * 8);
    *reinterpret_cast<int4*>(sRw + row * 24 + hf * 8) = v;
    __syncwarp();
  }

  // ---- key range of the CTA.
  const int pos_lo = past + first_tok;
  const int pos_hi = past + min(first_tok + 16 * NTT, q_len) - 1;
  int n_lo = 0;
  if (window > 0) n_lo = max(pos_lo - (window - 1), 0) / BN;
  int n_hi = min(pos_hi + 1, seqlen_k);
  n_hi = (n_hi + BN - 1) / BN;
  // key range of this warp.
  const int wpos_lo = past + tok0;
  const int wpos_base = wpos_lo;
  const int wpos_hi = past + min(tok0 + 15, q_len - 1);
  const bool warp_live = tok0 < q_len;
  const float qk_scale = scale * LOG2E;

  // Softmax ownership: lane = (row = lane >> 1, half = lane & 1); the pair owns the 16 x BN
  // tile row, lane `half` taking the float4 chunks 2q + half.
  const int srow = lane >> 1, shalf = lane & 1;
  const int stok = tok0 + srow;
  const int spos = past + stok;
  const bool rvalid = stok < q_len;
  float m_reg = -INFINITY;  // running max of this lane's row (identical in both lanes of the pair)
  float l_reg = 0.f;        // partial running sum over this lane's columns
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[D / 16];
#pragma unroll
  for (int f = 0; f < D / 16; ++f) wmma::fill_fragment(acc[f], 0.f);

  const int* bt = BT + (int64_t)req * stride_bt;
  int4 buf[NLD];
  auto load_tile = [&](int n) {
#pragma unroll
    for (int it = 0; it < NLD; ++it) {
      const int idx = it * NT + tid;
      int4 v = make_int4(0, 0, 0, 0);
      if (idx < BN * 32) {
        const int row = idx >> 5, c = idx & 31;
        const int key = n * BN + row;
        bool ok = key < seqlen_k;
        if (window > 0) ok = ok && key > pos_lo - window;
        if (ok) {
          const int64_t pg = bt[key / page];
          const __half* src = KV + pg * stride_kv_blk + (int64_t)(key % page) * stride_kv_slot +
                              kvh * stride_kv_h + c * 8;
          v = *reinterpret_cast<const int4*>(src);
        }
      }
      buf[it] = v;
    }
  };
  auto store_tile = [&]() {
#pragma unroll
    for (int it = 0; it < NLD; ++it) {
      const int idx = it * NT + tid;
      if (idx < BN * 32) {
        const int row = idx >> 5, c = idx & 31;
        __half* dst = (c < 16) ? (sK + row * LDK + c * 8) : (sV + row * LDK + (c - 16) * 8);
        *reinterpret_cast<int4*>(dst) = buf[it];
      }
    }
  };

  if (n_hi > n_lo) load_tile(n_lo);
  for (int n = n_lo; n < n_hi; ++n) {
    __syncthreads();
    store_tile();
    __syncthreads();
    if (n + 1 < n_hi) load_tile(n + 1);
    const int kbase = n * BN;
    const bool live = warp_live && kbase <= wpos_hi &&
                      (window == 0 || kbase + BN - 1 > wpos_lo - window);
    if (live) {
      // S = Q K^T
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> sf[NF];
#pragma unroll
      for (int nf = 0; nf < NF; ++nf) wmma::fill_fragment(sf[nf], 0.f);
#pragma unroll
      for (int kk = 0; kk < D / 16; ++kk) {
#pragma unroll
        for (int nf = 0; nf < NF; ++nf) {
          wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> kb;
          wmma::load_matrix_sync(kb, sK + nf * 16 * LDK + kk * 16, LDK);
          wmma::mma_sync(sf[nf], qa[kk], kb, sf[nf]);
        }
      }
#pragma unroll
      for (int nf = 0; nf < NF; ++nf)
        wmma::store_matrix_sync(sSw + nf * 16, sf[nf], LDS, wmma::mem_row_major);
      __syncwarp();

      // Tile classes (warp-uniform): A = no mask, no bias; B = no mask, bias on every
      // element; C = general (mask + bias where distance < E).
      const bool interior = (kbase + BN - 1 <= wpos_lo) && (kbase + BN <= seqlen_k) &&
                            (window == 0 || wpos_hi - kbase < window);
      const bool no_bias = interior && (wpos_lo - (kbase + BN - 1) >= E);
      const bool all_bias = interior && (wpos_hi - kbase < E);

      float s[BN / 2];
#pragma unroll
      for (int q = 0; q < BN / 8; ++q) {
        const float4 v = *reinterpret_cast<const float4*>(sSw + srow * LDS + (2 * q + shalf) * 4);
        s[4 * q + 0] = v.x * qk_scale;
        s[4 * q + 1] = v.y * qk_scale;
        s[4 * q + 2] = v.z * qk_scale;
        s[4 * q + 3] = v.w * qk_scale;
      }
      if (!no_bias) {
        // M[i][c] = r_i . rel_proj[:, dstart + c] for the distances of this tile.
        const int dmin = wpos_base - (kbase + BN - 1);
        const int dstart = min(max((dmin >> 4) << 4, 0), E - NBF * 16);
        {
          wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> ra;
          wmma::load_matrix_sync(ra, sRw, 24);
#pragma unroll
          for (int nb = 0; nb < NBF; ++nb) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> pb;
            wmma::load_matrix_sync(pb, Proj + dstart + nb * 16, E);
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> mf;
            wmma::fill_fragment(mf, 0.f);
            wmma::mma_sync(mf, ra, pb, mf);
            wmma::store_matrix_sync(sMw + nb * 16, mf, LDM, wmma::mem_row_major);
          }
        }
        __syncwarp();
        const int base_idx = spos - kbase - dstart;  // M column of key `kbase`
        if (all_bias) {
          if (rvalid) {
#pragma unroll
            for (int q = 0; q < BN / 8; ++q) {
#pragma unroll
              for (int u = 0; u < 4; ++u) {
                const int j = (2 * q + shalf) * 4 + u;
                const float b = __half2float(__float2half(sMw[srow * LDM + base_idx - j]));
                s[4 * q + u] += b * LOG2E;
              }
            }
          }
        } else {
#pragma unroll
          for (int q = 0; q < BN / 8; ++q) {
#pragma unroll
            for (int u = 0; u < 4; ++u) {
              const int j = (2 * q + shalf) * 4 + u;
              const int key = kbase + j;
              const int dist = spos - key;
              bool vis = rvalid && dist >= 0 && key < seqlen_k;
              if (window > 0) vis = vis && dist < window;
              float v = s[4 * q + u];
              if (vis && dist < E) {
                const int idx = min(max(base_idx - j, 0), NBF * 16 - 1);
                v += __half2float(__float2half(sMw[srow * LDM + idx])) * LOG2E;
              }
              s[4 * q + u] = vis ? v : -INFINITY;
            }
          }
        }
      }
      float mx = s[0];
#pragma unroll
      for (int j = 1; j < BN / 2; ++j) mx = fmaxf(mx, s[j]);
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
      const float m_new = fmaxf(m_reg, mx);
      const float m_safe = (m_new == -INFINITY) ? 0.f : m_new;
      const float alpha = ex2(m_reg - m_safe);
      float lsum = 0.f;
#pragma unroll
      for (int q = 0; q < BN / 8; ++q) {
        const float p0 = ex2(s[4 * q + 0] - m_safe), p1 = ex2(s[4 * q + 1] - m_safe);
        const float p2 = ex2(s[4 * q + 2] - m_safe), p3 = ex2(s[4 * q + 3] - m_safe);
        lsum += (p0 + p1) + (p2 + p3);
        __half2 h01 = __floats2half2_rn(p0, p1), h23 = __floats2half2_rn(p2, p3);
        uint2 raw;
        raw.x = *reinterpret_cast<unsigned*>(&h01);
        raw.y = *reinterpret_cast<unsigned*>(&h23);
        *reinterpret_cast<uint2*>(sPw + srow * LDP + (2 * q + shalf) * 4) = raw;
      }
      l_reg = l_reg * alpha + lsum;
      m_reg = m_new;
      if (shalf == 0) sAlpha[srow] = alpha;
      __syncwarp();
      // rescale O
      float al[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) al[e] = sAlpha[acc_row(lane, e)];
#pragma unroll
      for (int f = 0; f < D / 16; ++f)
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[f].x[e] *= al[e];
      // O += P V
#pragma unroll
      for (int j = 0; j < NF; ++j) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> pa;
        wmma::load_matrix_sync(pa, sPw + j * 16, LDP);
#pragma unroll
        for (int f = 0; f < D / 16; ++f) {
          wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> vb;
          wmma::load_matrix_sync(vb, sV + j * 16 * LDK + f * 16, LDK);
          wmma::mma_sync(acc[f], pa, vb, acc[f]);
        }
      }
      __syncwarp();
    }
  }

  // ---- epilogue: o / l * sigmoid(gate) * sigmoid(lse - sink)
  if (warp_live) {
    const float l = l_reg + __shfl_xor_sync(0xffffffffu, l_reg, 1);
    if (shalf == 0) {
      float ss = 0.f;
      if (rvalid) {
        const float sink = __half2float(Sink[(int64_t)(q_start + stok) * stride_sink + head]);
        const float lse = (l == 0.f) ? -INFINITY : (m_reg + __log2f(l)) * LN2;
        ss = sigm(lse - sink);
      }
      sL[srow] = (l == 0.f) ? 1.f : l;
      sSink[srow] = ss;
    }
    __syncwarp();
#pragma unroll
    for (int e = 0; e < 8; e += 2) {
      const int r = acc_row(lane, e);
      const int c = acc_col(lane, e);
      const int tok = tok0 + r;
      if (tok < q_len) {
        const float inv_l = sL[r];
        const float ss = sSink[r];
        const int64_t base = (int64_t)(q_start + tok);
#pragma unroll
        for (int f = 0; f < D / 16; ++f) {
          const int col = f * 16 + c;
          const float2 gv = __half22float2(
              *reinterpret_cast<const __half2*>(Gate + base * stride_gate + head * D + col));
          float o0 = acc[f].x[e] / inv_l, o1 = acc[f].x[e + 1] / inv_l;
          o0 = o0 * sigm(gv.x);
          o1 = o1 * sigm(gv.y);
          o0 = o0 * ss;
          o1 = o1 * ss;
          *reinterpret_cast<__half2*>(Out + base * qrow + head * D + col) = __floats2half2_rn(o0, o1);
        }
      }
    }
  }
}

template <int NTT, int BN, int MINB>
void launch(const at::Tensor& q, const at::Tensor& r, const at::Tensor& proj, const at::Tensor& kv,
            const at::Tensor& bt, const at::Tensor& seq_lens, const at::Tensor& cuq,
            const at::Tensor& gate, const at::Tensor& sinks, const at::Tensor& out,
            int64_t num_reqs, int64_t max_q, int64_t window, int64_t extent, double scale) {
  constexpr int NW = 3 * NTT;
  constexpr int LDS = BN + 4, LDP = BN + 8;
  constexpr int LDM = ((BN + 45) / 16) * 16 + 4;
  const size_t smem = 2 * BN * LDK * sizeof(__half) + NW * 16 * LDS * sizeof(float) +
                      NW * 16 * LDP * sizeof(__half) + NW * 48 * sizeof(float) +
                      NW * 16 * LDM * sizeof(float) + NW * 16 * 24 * sizeof(__half);
  auto kern = prefill_attn_kernel<NTT, BN, MINB>;
  static bool attr_set = false;
  if (!attr_set && smem > 48 * 1024) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    attr_set = true;
  }
  const int H = (int)q.size(1);
  const int Hkv = (int)kv.size(2);
  dim3 grid((unsigned)((max_q + 16 * NTT - 1) / (16 * NTT)), Hkv, (unsigned)num_reqs);
  kern<<<grid, 96 * NTT, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(q.data_ptr()), reinterpret_cast<const __half*>(kv.data_ptr()),
      reinterpret_cast<const __half*>(r.data_ptr()), reinterpret_cast<const __half*>(proj.data_ptr()),
      bt.data_ptr<int>(), seq_lens.data_ptr<int>(),
      cuq.data_ptr<int>(), reinterpret_cast<const __half*>(gate.data_ptr()),
      reinterpret_cast<const __half*>(sinks.data_ptr()), reinterpret_cast<__half*>(out.data_ptr()),
      H, kv.stride(0), kv.stride(1), kv.stride(2), bt.stride(0), gate.stride(0),
      sinks.stride(0), (int)kv.size(1), (int)extent, (int)window, (float)scale);
}

}  // namespace

void prefill_attention_cuda(
    torch::Tensor q, torch::Tensor r, torch::Tensor proj, torch::Tensor kv, torch::Tensor block_table,
    torch::Tensor seq_lens, torch::Tensor cu_seqlens_q, torch::Tensor gate,
    torch::Tensor sinks, torch::Tensor out, int64_t num_reqs, int64_t max_q,
    int64_t window, int64_t extent, double scale, int64_t variant) {
  switch (variant) {
    case 0: launch<2, 32, 1>(q, r, proj, kv, block_table, seq_lens, cu_seqlens_q, gate, sinks, out, num_reqs, max_q, window, extent, scale); break;
    case 1: launch<1, 32, 2>(q, r, proj, kv, block_table, seq_lens, cu_seqlens_q, gate, sinks, out, num_reqs, max_q, window, extent, scale); break;
    case 2: launch<1, 64, 2>(q, r, proj, kv, block_table, seq_lens, cu_seqlens_q, gate, sinks, out, num_reqs, max_q, window, extent, scale); break;
    default: TORCH_CHECK(false, "bad variant");
  }
}
"""

# Accumulator element -> (row, col) of a 16x16 WMMA fp32 tile (measured with a probe).
_ACC_ROW_EXPR = (
    "(lane & 1) + 2 * ((e >> 1) & 1) + 4 * ((lane >> 4) & 1) + 8 * ((lane >> 2) & 1)"
)
_ACC_COL_EXPR = (
    "(e & 1) + 2 * ((lane >> 1) & 1) + 4 * ((e >> 2) & 1) + 8 * ((lane >> 3) & 1)"
)

_VARIANT = int(os.environ.get("VLLM_INKLING2_SM70_PREFILL_ATTN_VARIANT", "0"))


def _ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load_inline

        build_dir = os.environ.get(
            "VLLM_INKLING2_SM70_EXT_DIR",
            os.path.expanduser("~/.cache/vllm/inkling2_sm70_ext"),
        )
        build_dir = os.path.join(build_dir, "prefill_attention")
        os.makedirs(build_dir, exist_ok=True)
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
        src = _CUDA_SRC.replace("ACC_ROW_EXPR", _ACC_ROW_EXPR).replace(
            "ACC_COL_EXPR", _ACC_COL_EXPR
        )
        _EXT = load_inline(
            name="inkling2_sm70_prefill_attention",
            cpp_sources=[_CPP_SRC],
            cuda_sources=[src],
            functions=["prefill_attention_cuda"],
            build_directory=build_dir,
            extra_cuda_cflags=[
                "-O3",
                "-lineinfo",
                "-gencode=arch=compute_70,code=sm_70",
            ],
            verbose=False,
        )
    return _EXT


def prepare_prefill_attention() -> None:
    """Build the extension (at model load, never inside a forward)."""
    _ext()


def supports_prefill_attention(
    num_heads: int, num_kv_heads: int, head_dim: int, d_rel: int, dtype: torch.dtype
) -> bool:
    return (
        num_heads // num_kv_heads == 3
        and num_heads % num_kv_heads == 0
        and head_dim == 128
        and d_rel == 16
        and dtype == torch.float16
    )


@triton.jit
def _qr_clip_kernel(
    Q,
    R,
    QScale,
    RScale,
    Tau,
    QOut,
    ROut,
    stride_q,
    stride_r,
    q_threshold,
    r_threshold,
    eps,
    H: tl.constexpr,
    D: tl.constexpr,
    D_REL: tl.constexpr,
    HAS_TAU: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # Same arithmetic as `qr_clip_rel_tokens` (norm.py), minus the relative logits.
    t = tl.program_id(0).to(tl.int64)
    heads = tl.arange(0, BLOCK_H)[:, None]
    h_mask = heads < H
    q_cols = tl.arange(0, D)[None, :]
    r_cols = tl.arange(0, D_REL)[None, :]
    tau = tl.load(Tau + t) if HAS_TAU else 1.0
    q = _l2_clip(
        Q + t * stride_q, QScale, heads, q_cols, h_mask, q_threshold, eps, tau, HAS_TAU
    )
    tl.store(QOut + t * H * D + heads * D + q_cols, q, h_mask)
    r = _l2_clip(
        R + t * stride_r, RScale, heads, r_cols, h_mask, r_threshold, eps, tau, HAS_TAU
    )
    tl.store(ROut + t * H * D_REL + heads * D_REL + r_cols, r, h_mask)


def clip_qr(
    q: torch.Tensor,
    r: torch.Tensor,
    q_scale: torch.Tensor,
    r_scale: torch.Tensor,
    tau: torch.Tensor | None,
    *,
    clip_threshold: float,
    rel_threshold: float,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """L2-clipped `q * q_scale` `(T, H, D)` and `r * r_scale` `(T, H, d_rel)`
    (both rounded to fp16 and scaled by the fp32 per-token `tau` if given)."""
    num_tokens, num_heads, head_dim = q.shape
    d_rel = r.shape[-1]
    assert q.stride(2) == 1 and q.stride(1) == head_dim
    assert r.stride(-1) == 1 and r.stride(-2) == d_rel
    q_out = torch.empty_like(q)
    r_out = torch.empty((num_tokens, num_heads, d_rel), dtype=r.dtype, device=r.device)
    if num_tokens:
        _qr_clip_kernel[(num_tokens,)](
            q,
            r,
            q_scale,
            r_scale,
            tau,
            q_out,
            r_out,
            q.stride(0),
            r.stride(0),
            clip_threshold,
            rel_threshold,
            eps,
            H=num_heads,
            D=head_dim,
            D_REL=d_rel,
            HAS_TAU=tau is not None,
            BLOCK_H=triton.next_power_of_2(num_heads),
            num_warps=4,
        )
    return q_out, r_out


def prefill_attention(
    q: torch.Tensor,
    r: torch.Tensor,
    rel_proj: torch.Tensor,
    kv_cache: torch.Tensor,
    gate: torch.Tensor,
    sinks: torch.Tensor,
    out: torch.Tensor,
    *,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_query_len: int,
    num_actual_tokens: int,
    softmax_scale: float,
    window: int,
    variant: int | None = None,
) -> None:
    """Gated fp16 `(T, H * D)` attention output of a prefill / chunked-prefill batch.

    `q` `(T, H, D)` and `r` `(T, H, d_rel)` are the clipped projections
    (`clip_qr`), `rel_proj` the fp16 `(d_rel, extent)` relative projection,
    `window` the sliding window (keys up to a query's position inclusive;
    0 = full). Rows from `num_actual_tokens` on are zeroed.
    """
    num_tokens, num_heads, head_dim = q.shape
    assert (
        q.is_contiguous()
        and r.is_contiguous()
        and kv_cache.stride(3) == 1
        and out.is_contiguous()
    )
    assert (
        rel_proj.is_contiguous()
        and rel_proj.dtype == torch.float16
        and rel_proj.shape[0] == 16
    )
    assert gate.stride(1) == 1 and sinks.stride(1) == 1
    assert (
        gate.stride(0) % 2 == 0
        and gate.data_ptr() % 4 == 0
        and kv_cache.data_ptr() % 16 == 0
    )
    assert kv_cache.dtype == torch.float16 and q.dtype == torch.float16
    extent = rel_proj.shape[1]
    assert extent % 16 == 0 and extent >= 128
    if num_actual_tokens < num_tokens:
        out[num_actual_tokens:].zero_()
    if num_actual_tokens == 0:
        return
    _ext().prefill_attention_cuda(
        q,
        r,
        rel_proj,
        kv_cache,
        block_table,
        seq_lens,
        cu_seqlens_q,
        gate,
        sinks,
        out,
        seq_lens.shape[0],
        max_query_len,
        window,
        extent,
        softmax_scale,
        _VARIANT if variant is None else variant,
    )
