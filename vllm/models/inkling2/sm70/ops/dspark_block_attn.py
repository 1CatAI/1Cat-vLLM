# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""SM70 WMMA tensor-core split-KV bidirectional block attention for DSpark draft.

Replaces Triton `rel_attention` (scalar FP32 FMA `tl.dot` on SM70) in
`ops/dspark.py:block_attention`. Specialized for `head_dim == 128`,
`num_heads // num_kv_heads == 3`, `block <= 10` (M = block * 3 <= 32,
2 WMMA M=16 tiles), paged fp16 KV cache `(blocks, page, H_kv, 2 * D)`.

8 warps (256 threads) per CTA, `BN = 64` keys per tile:
  - QK^T: warp (wm=w>>2, wn=w&3) computes one 16x16 WMMA tile -> covers 32x64
    in 8 WMMA ops.
  - Online softmax: 8 warps each reduce 4 of the 32 rows (64 cols = 2/lane),
    writing fp16 P [32, 64] to shared memory and rescaling O accumulators.
  - P @ V: warp (wm=w>>2, wd=w&3) computes two 16x16 WMMA tiles (32 dims),
    accumulating into persistent fp32 fragments across all key tiles.
"""

import os

import torch

_EXT = None

_CPP_SRC = r"""
void dspark_block_attn_cuda(
    torch::Tensor q, torch::Tensor kv, torch::Tensor block_table,
    torch::Tensor seq_lens, torch::Tensor cu_seqlens_q,
    torch::Tensor o_part, torch::Tensor lse_part,
    int64_t num_reqs, int64_t block_size, int64_t window,
    int64_t num_splits, double scale);
"""

_CUDA_SRC = r"""
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <cstdint>

using namespace nvcuda;

namespace {

constexpr int HD = 128;
constexpr int GQ = 3;
constexpr int BM = 32;   // 2 WMMA M=16 tiles (covers block_size <= 10, rows <= 30)
constexpr int BN = 64;   // 4 WMMA N=16 tiles
constexpr int LDK = HD + 8;  // 136 halves (bank-conflict-free)
constexpr int LDS = BN + 4;  // 68 floats
constexpr int LDP = BN + 8;  // 72 halves
constexpr float LOG2E_F = 1.4426950408889634f;

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// grid (num_splits, H_kv, num_reqs), block 256 (8 warps)
__global__ void __launch_bounds__(256, 2) dspark_block_attn_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ KV,
    const int* __restrict__ block_table,
    const int* __restrict__ seq_lens,
    const int* __restrict__ cu_q,
    float* __restrict__ O_part,
    float* __restrict__ Lse_part,
    int T, int H, int Hkv, int page_size,
    int64_t kv_s_blk, int64_t kv_s_pos, int64_t kv_s_h,
    int64_t bt_stride,
    int block_size, int window, int num_splits, float scale) {
  const int split = blockIdx.x;
  const int kvh = blockIdx.y;
  const int req = blockIdx.z;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  const int q_start = cu_q[req];
  const int q_end = cu_q[req + 1];
  const int q_len = q_end - q_start;
  const int rows = q_len * GQ;  // <= 30 (for block_size <= 10)
  const int seqlen = seq_lens[req];

  if (q_len <= 0 || seqlen <= 0) {
    for (int r = warp; r < rows; r += 8) {
      const int tk = r / GQ, qh = kvh * GQ + (r - tk * GQ);
      const int64_t idx = ((int64_t)split * T + q_start + tk) * H + qh;
      *reinterpret_cast<float4*>(O_part + idx * HD + lane * 4) = make_float4(0.f, 0.f, 0.f, 0.f);
      if (lane == 0) Lse_part[idx] = -INFINITY;
    }
    return;
  }

  const int lo = (window > 0 && seqlen > window) ? (seqlen - window) : 0;
  const int hi = seqlen;
  const int tile0 = lo / BN;
  const int tile1 = (hi + BN - 1) / BN;
  const int ntiles = tile1 - tile0;

  if (split >= ntiles) {
    for (int r = warp; r < rows; r += 8) {
      const int tk = r / GQ, qh = kvh * GQ + (r - tk * GQ);
      const int64_t idx = ((int64_t)split * T + q_start + tk) * H + qh;
      *reinterpret_cast<float4*>(O_part + idx * HD + lane * 4) = make_float4(0.f, 0.f, 0.f, 0.f);
      if (lane == 0) Lse_part[idx] = -INFINITY;
    }
    return;
  }

  // Shared memory layout (~57 KB, fits in V100 96 KB smem)
  extern __shared__ __align__(16) unsigned char smem_raw[];
  __half* sQ = reinterpret_cast<__half*>(smem_raw);                    // 32 * 136 * 2 = 8,704 B
  __half* sK = sQ + BM * LDK;                                          // 64 * 136 * 2 = 17,408 B
  __half* sV = sK + BN * LDK;                                          // 64 * 136 * 2 = 17,408 B
  float* sS = reinterpret_cast<float*>(sV + BN * LDK);                 // 32 * 68 * 4  = 8,704 B
  __half* sP = reinterpret_cast<__half*>(sS + BM * LDS);               // 32 * 72 * 2  = 4,608 B
  float* sM = reinterpret_cast<float*>(sP + BM * LDP);                 // 32 * 4
  float* sL = sM + BM;                                                 // 32 * 4
  float* sAlpha = sL + BM;                                             // 32 * 4

  // Load Q [rows, 128] into sQ [32, LDK] (zero-pad rows..31)
  for (int idx = tid; idx < BM * (HD / 8); idx += 256) {
    const int r = idx >> 4;
    const int c8 = idx & 15;
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (r < rows) {
      const int tk = r / GQ, qh = kvh * GQ + (r - tk * GQ);
      v = *reinterpret_cast<const uint4*>(Q + ((int64_t)(q_start + tk) * H + qh) * HD + c8 * 8);
    }
    *reinterpret_cast<uint4*>(sQ + r * LDK + c8 * 8) = v;
  }
  if (tid < BM) {
    sM[tid] = -INFINITY;
    sL[tid] = 0.f;
    sAlpha[tid] = 1.f;
  }
  __syncthreads();

  const int wm = warp >> 2;  // 0 or 1 (which M=16 tile)
  const int wn = warp & 3;   // 0..3 (which N=16 key tile for QK, or which 32-dim slice for PV)
  const int page_shift = __ffs(page_size) - 1;
  const int page_mask = page_size - 1;
  const int* bt_row = block_table + (int64_t)req * bt_stride;

  // Persistent Q fragments for this warp M=16 tile (8 K-chunks of 16)
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> q_frag[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    wmma::load_matrix_sync(q_frag[k], sQ + wm * 16 * LDK + k * 16, LDK);
  }

  // Persistent O accumulators: 2 tiles of 16x16 = 32 output dims per warp
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_frag[2];
  wmma::fill_fragment(o_frag[0], 0.f);
  wmma::fill_fragment(o_frag[1], 0.f);

  const float scale_log2 = scale * LOG2E_F;

  for (int t = tile0 + split; t < tile1; t += num_splits) {
    const int kbase = t * BN;
    // Load K and V [64, 128] from paged KV cache (256 threads -> 4 uint4 loads/thread each)
#pragma unroll
    for (int it = 0; it < 4; ++it) {
      const int idx = tid + it * 256;
      const int row = idx >> 4;   // 0..63
      const int c8 = idx & 15;    // 0..15 (x8 halves = 128)
      const int pos = kbase + row;
      uint4 kv_k = make_uint4(0u, 0u, 0u, 0u);
      uint4 kv_v = make_uint4(0u, 0u, 0u, 0u);
      if (pos >= lo && pos < hi) {
        const int blk = bt_row[pos >> page_shift];
        const __half* p = KV + (int64_t)blk * kv_s_blk + (int64_t)(pos & page_mask) * kv_s_pos + (int64_t)kvh * kv_s_h;
        kv_k = __ldg(reinterpret_cast<const uint4*>(p + c8 * 8));
        kv_v = __ldg(reinterpret_cast<const uint4*>(p + HD + c8 * 8));
      }
      *reinterpret_cast<uint4*>(sK + row * LDK + c8 * 8) = kv_k;
      *reinterpret_cast<uint4*>(sV + row * LDK + c8 * 8) = kv_v;
    }
    __syncthreads();

    // 1. QK^T: warp (wm, wn) computes 16x16 tile of S [32, 64]
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag;
    wmma::fill_fragment(s_frag, 0.f);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> k_frag;
      wmma::load_matrix_sync(k_frag, sK + wn * 16 * LDK + k * 16, LDK);
      wmma::mma_sync(s_frag, q_frag[k], k_frag, s_frag);
    }
    wmma::store_matrix_sync(sS + wm * 16 * LDS + wn * 16, s_frag, LDS, wmma::mem_row_major);
    __syncthreads();

    // 2. Online softmax: each warp handles 4 rows (warp*4 .. warp*4+3), 64 cols = 2/lane
    for (int dr = 0; dr < 4; ++dr) {
      const int r = warp * 4 + dr;
      const int c0 = lane, c1 = lane + 32;
      const int p0 = kbase + c0, p1 = kbase + c1;
      float v0 = (p0 >= lo && p0 < hi) ? sS[r * LDS + c0] * scale_log2 : -INFINITY;
      float v1 = (p1 >= lo && p1 < hi) ? sS[r * LDS + c1] * scale_log2 : -INFINITY;
      const float mo = sM[r];
      const float rmax = warp_max(fmaxf(v0, v1));
      const float mn = fmaxf(mo, rmax);
      const float ms = (mn == -INFINITY) ? 0.f : mn;
      const float alpha = (mo == -INFINITY) ? 0.f : exp2f(mo - ms);
      const float e0 = (p0 >= lo && p0 < hi) ? exp2f(v0 - ms) : 0.f;
      const float e1 = (p1 >= lo && p1 < hi) ? exp2f(v1 - ms) : 0.f;
      const float lsum = warp_sum(e0 + e1);
      sP[r * LDP + c0] = __float2half_rn(e0);
      sP[r * LDP + c1] = __float2half_rn(e1);
      if (lane == 0) {
        sM[r] = mn;
        sL[r] = sL[r] * alpha + lsum;
        sAlpha[r] = alpha;
      }
    }
    __syncthreads();

    // Rescale persistent O fragments by per-row alpha
    const int base_r = wm * 16;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int r = base_r + (lane & 1) + 2 * ((e >> 1) & 1) + 4 * ((lane >> 4) & 1) + 8 * ((lane >> 2) & 1);
      const float a = sAlpha[r];
      o_frag[0].x[e] *= a;
      o_frag[1].x[e] *= a;
    }

    // 3. P @ V: warp (wm, wn) computes 16x32 dims [wn*32 .. wn*32+31]
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> p_frag;
      wmma::load_matrix_sync(p_frag, sP + wm * 16 * LDP + k * 16, LDP);
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> v0_frag, v1_frag;
      wmma::load_matrix_sync(v0_frag, sV + k * 16 * LDK + (wn * 32 + 0), LDK);
      wmma::load_matrix_sync(v1_frag, sV + k * 16 * LDK + (wn * 32 + 16), LDK);
      wmma::mma_sync(o_frag[0], p_frag, v0_frag, o_frag[0]);
      wmma::mma_sync(o_frag[1], p_frag, v1_frag, o_frag[1]);
    }
    __syncthreads();
  }

  // Store normalized O_part & Lse_part (reusing sK as float [32, 128] = 16 KB)
  float* sO = reinterpret_cast<float*>(sK);
  wmma::store_matrix_sync(sO + wm * 16 * HD + (wn * 32 + 0), o_frag[0], HD, wmma::mem_row_major);
  wmma::store_matrix_sync(sO + wm * 16 * HD + (wn * 32 + 16), o_frag[1], HD, wmma::mem_row_major);
  __syncthreads();

  for (int r = warp; r < rows; r += 8) {
    const int tk = r / GQ, qh = kvh * GQ + (r - tk * GQ);
    const int64_t idx = ((int64_t)split * T + q_start + tk) * H + qh;
    const float l = sL[r];
    const float inv_l = (l > 0.f) ? (1.f / l) : 0.f;
    float4 v = *reinterpret_cast<const float4*>(sO + r * HD + lane * 4);
    v.x *= inv_l; v.y *= inv_l; v.z *= inv_l; v.w *= inv_l;
    *reinterpret_cast<float4*>(O_part + idx * HD + lane * 4) = v;
    if (lane == 0) {
      Lse_part[idx] = (l > 0.f) ? (sM[r] + log2f(l)) : -INFINITY;
    }
  }
}

}  // namespace

void dspark_block_attn_cuda(
    torch::Tensor q, torch::Tensor kv, torch::Tensor block_table,
    torch::Tensor seq_lens, torch::Tensor cu_seqlens_q,
    torch::Tensor o_part, torch::Tensor lse_part,
    int64_t num_reqs, int64_t block_size, int64_t window,
    int64_t num_splits, double scale) {
  const int T = (int)q.size(0);
  const int H = (int)q.size(1);
  const int Hkv = (int)kv.size(2);
  const int page_size = (int)kv.size(1);
  const size_t smem =
      BM * LDK * sizeof(__half) +
      2 * BN * LDK * sizeof(__half) +
      BM * LDS * sizeof(float) +
      BM * LDP * sizeof(__half) +
      3 * BM * sizeof(float);
  static bool attr_set = false;
  if (!attr_set && smem > 48 * 1024) {
    cudaFuncSetAttribute(
        dspark_block_attn_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem);
    attr_set = true;
  }
  dim3 grid((unsigned)num_splits, (unsigned)Hkv, (unsigned)num_reqs);
  dspark_block_attn_kernel<<<grid, 256, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(q.data_ptr()),
      reinterpret_cast<const __half*>(kv.data_ptr()),
      block_table.data_ptr<int>(),
      seq_lens.data_ptr<int>(),
      cu_seqlens_q.data_ptr<int>(),
      o_part.data_ptr<float>(),
      lse_part.data_ptr<float>(),
      T, H, Hkv, page_size,
      kv.stride(0), kv.stride(1), kv.stride(2),
      block_table.stride(0),
      (int)block_size, (int)window, (int)num_splits, (float)scale);
}
"""


def _ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load_inline

        build_dir = os.path.join(
            os.environ.get(
                "VLLM_INKLING2_SM70_EXT_DIR",
                os.path.expanduser("~/.cache/vllm/inkling2_sm70_ext"),
            ),
            "dspark_block_attn",
        )
        os.makedirs(build_dir, exist_ok=True)
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
        _EXT = load_inline(
            name="inkling2_sm70_dspark_block_attn",
            cpp_sources=[_CPP_SRC],
            cuda_sources=[_CUDA_SRC],
            functions=["dspark_block_attn_cuda"],
            build_directory=build_dir,
            extra_cuda_cflags=[
                "-O3",
                "-gencode=arch=compute_70,code=sm_70",
            ],
            verbose=False,
        )
    return _EXT


def supported(q: torch.Tensor, kv_cache: torch.Tensor, block: int) -> bool:
    return (
        q.is_cuda
        and q.dtype == torch.float16
        and kv_cache.dtype == torch.float16
        and q.shape[2] == 128
        and q.shape[1] == kv_cache.shape[2] * 3
        and 1 <= block <= 10
        and q.is_contiguous()
    )


def run_partial(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    o_part: torch.Tensor,
    lse_part: torch.Tensor,
    *,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    block: int,
    softmax_scale: float,
    window: int,
    num_splits: int,
) -> None:
    _ext().dspark_block_attn_cuda(
        q,
        kv_cache,
        block_table,
        cache_seqlens,
        cu_seqlens_q,
        o_part,
        lse_part,
        block_table.shape[0],
        block,
        window,
        num_splits,
        float(softmax_scale),
    )
