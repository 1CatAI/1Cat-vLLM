# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Grouped tensor-core prefill for the MXFP8 (fp16-dequantized) Inkling2 experts, SM70.

Replaces the per-expert host loop of `Inkling2DenseExperts` (bincount().tolist(), ~1000
launches per layer visit) by four launches and no host sync:

1. `route_kernel`  (1 CTA): histogram of the expert ids, prefix sums, tile map (BM-row tiles
   per expert), scatter of the slots into expert-sorted order (`sorted_slot`, `inv_pos`).
2. `gemm1` (WMMA m16n16k16, fp16 in / fp32 acc): rows of `x` gathered by token, per-expert
   `W13[e]` ([gate; up] halves), epilogue = softcapped SwiGLU on the fp16-rounded
   gate/up accumulators, written as fp16 `act [S, I]` in sorted order.
3. `gemm2`: `act @ W2[e]^T` -> fp16 `down [S, H]` (sorted order).
4. `reduce_kernel`: `out[t] = sum_k w[t, k] * down[inv_pos[t * K + k]]` in fp32 -> fp16.

Tiles are scheduled from a device-side tile map (grid sized for the worst case, surplus
CTAs exit), so nothing depends on the routing on the host. Numerically it equals the old
path (fp16 GEMM outputs, fp32 weighted sum), only the fp32 summation order of the 8 expert
terms differs.
"""

import os

import torch

_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

#define BM 128
#define BK 32
#define PAD 8
#define NTHREADS 256
#define MAXE 256

// ---------------- routing: one CTA ----------------
__global__ void __launch_bounds__(1024) route_kernel(
    const int* __restrict__ ids, int S, int E, int* __restrict__ sorted_slot,
    int* __restrict__ inv_pos, int4* __restrict__ tiles, int* __restrict__ ntiles_out) {
  __shared__ int cnt[MAXE], off[MAXE], toff[MAXE], cur[MAXE];
  __shared__ int ntiles;
  int tid = threadIdx.x;
  if (tid < MAXE) { cnt[tid] = 0; cur[tid] = 0; }
  __syncthreads();
  for (int s = tid; s < S; s += blockDim.x) atomicAdd(&cnt[ids[s]], 1);
  __syncthreads();
  if (tid == 0) {
    int run = 0, tr = 0;
    for (int e = 0; e < E; e++) {
      off[e] = run; run += cnt[e];
      toff[e] = tr; tr += (cnt[e] + BM - 1) / BM;
    }
    ntiles = tr; *ntiles_out = tr;
  }
  __syncthreads();
  if (tid < E) {
    int n = (cnt[tid] + BM - 1) / BM;
    for (int t = 0; t < n; t++) {
      int rows = cnt[tid] - t * BM; if (rows > BM) rows = BM;
      tiles[toff[tid] + t] = make_int4(tid, off[tid] + t * BM, rows, 0);
    }
  }
  for (int s = tid; s < S; s += blockDim.x) {
    int e = ids[s];
    int p = off[e] + atomicAdd(&cur[e], 1);
    sorted_slot[p] = s; inv_pos[s] = p;
  }
}

__device__ __forceinline__ float2 h2f(__half2 v) { return __half22float2(v); }

// ---------------- grouped GEMM ----------------
// C[row, n] = sum_k A[row, k] * W[e][nrow(n), k].
// SWIGLU: W is [E, 2I, K] ([gate; up] halves), tile n-range BNW of both halves, out = act [S, I].
// else:   W is [E, N, K], tile n-range 2*BNW, out = [S, N].
// GATHER: A row = x[sorted_slot[row0 + r] / TOPK] else A row = a[row0 + r].
template <int BNW, bool SWIGLU, bool GATHER>
__global__ void __launch_bounds__(NTHREADS, 2) gemm_kernel(
    const __half* __restrict__ A, const __half* __restrict__ W, __half* __restrict__ C,
    const int* __restrict__ sorted_slot, const int4* __restrict__ tiles,
    const int* __restrict__ ntiles, int K, int N, int topk, float gcap, float ucap) {
  constexpr int BN = 2 * BNW;
  constexpr int FM = 2;                 // 32 rows per warp
  constexpr int FN = BNW / 16;
  constexpr int LDA = BK + PAD, LDB = BK + PAD;
  constexpr int A_ELEMS = BM * LDA, B_ELEMS = BN * LDB;
  constexpr int NAV = BM * (BK / 8), NBV = BN * (BK / 8);   // 16-byte vectors per tile
  constexpr int NAI = (NAV + NTHREADS - 1) / NTHREADS, NBI = (NBV + NTHREADS - 1) / NTHREADS;
  constexpr int LDS = BN + PAD;         // epilogue staging (half)

  extern __shared__ __align__(128) unsigned char smem_raw[];
  __half* As = reinterpret_cast<__half*>(smem_raw);
  __half* Bs = As + 2 * A_ELEMS;

  const int tile = blockIdx.y;
  if (tile >= *ntiles) return;
  const int4 td = tiles[tile];
  const int e = td.x, row0 = td.y, nrows = td.z;
  const int tid = threadIdx.x, warp = tid >> 5;
  const int wm = warp >> 1, wn = warp & 1;
  const int ntile = blockIdx.x;

  const __half* We = W + (size_t)e * (size_t)(SWIGLU ? 2 * N : N) * K;   // N = I (swiglu) or N

  // per-thread load descriptors
  const __half* ap[NAI]; int a_off[NAI]; bool a_ok[NAI];
  const __half* bp[NBI]; int b_off[NBI]; bool b_ok[NBI];
  #pragma unroll
  for (int i = 0; i < NAI; i++) {
    int v = tid + i * NTHREADS;
    int r = v >> 2, c = v & 3;
    a_ok[i] = (v < NAV);
    a_off[i] = r * LDA + c * 8;
    const __half* p = nullptr;
    if (a_ok[i] && r < nrows) {
      if (GATHER) p = A + (size_t)(sorted_slot[row0 + r] / topk) * K + c * 8;
      else p = A + (size_t)(row0 + r) * K + c * 8;
    }
    ap[i] = p;
  }
  #pragma unroll
  for (int i = 0; i < NBI; i++) {
    int v = tid + i * NTHREADS;
    int r = v >> 2, c = v & 3;
    b_ok[i] = (v < NBV);
    b_off[i] = r * LDB + c * 8;
    int grow;
    if (SWIGLU) grow = (r < BNW) ? (ntile * BNW + r) : (N + ntile * BNW + (r - BNW));
    else grow = ntile * BN + r;
    bp[i] = b_ok[i] ? We + (size_t)grow * K + c * 8 : nullptr;
  }

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[FM][FN];
  #pragma unroll
  for (int i = 0; i < FM; i++)
    #pragma unroll
    for (int j = 0; j < FN; j++) wmma::fill_fragment(acc[i][j], 0.f);

  uint4 ra[NAI], rb[NBI];
  const int KT = K / BK;
  auto gload = [&](int kt) {
    #pragma unroll
    for (int i = 0; i < NAI; i++)
      ra[i] = ap[i] ? __ldg(reinterpret_cast<const uint4*>(ap[i] + kt * BK)) : make_uint4(0, 0, 0, 0);
    #pragma unroll
    for (int i = 0; i < NBI; i++)
      rb[i] = bp[i] ? __ldg(reinterpret_cast<const uint4*>(bp[i] + kt * BK)) : make_uint4(0, 0, 0, 0);
  };
  auto sstore = [&](int buf) {
    #pragma unroll
    for (int i = 0; i < NAI; i++)
      if (a_ok[i]) *reinterpret_cast<uint4*>(As + buf * A_ELEMS + a_off[i]) = ra[i];
    #pragma unroll
    for (int i = 0; i < NBI; i++)
      if (b_ok[i]) *reinterpret_cast<uint4*>(Bs + buf * B_ELEMS + b_off[i]) = rb[i];
  };

  gload(0); sstore(0); __syncthreads();
  int cur = 0;
  for (int kt = 0; kt < KT; kt++) {
    if (kt + 1 < KT) gload(kt + 1);
    const __half* as = As + cur * A_ELEMS + (wm * 32) * LDA;
    const __half* bs = Bs + cur * B_ELEMS + (wn * BNW) * LDB;
    #pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa[FM];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb[FN];
      #pragma unroll
      for (int i = 0; i < FM; i++) wmma::load_matrix_sync(fa[i], as + i * 16 * LDA + kk, LDA);
      #pragma unroll
      for (int j = 0; j < FN; j++) wmma::load_matrix_sync(fb[j], bs + j * 16 * LDB + kk, LDB);
      #pragma unroll
      for (int i = 0; i < FM; i++)
        #pragma unroll
        for (int j = 0; j < FN; j++) wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
    }
    if (kt + 1 < KT) sstore(cur ^ 1);
    __syncthreads();
    cur ^= 1;
  }

  // epilogue: accumulators -> fp16 staging in smem (reuses the pipeline buffers). The f32 and f16
  // accumulator fragment layouts differ on Volta, so each warp bounces its 16x16 tiles through a
  // per-warp f32 scratch and converts there.
  constexpr int STAGE_BYTES = BM * LDS * 2;
  __half* st = reinterpret_cast<__half*>(smem_raw);
  float* scr = reinterpret_cast<float*>(smem_raw + STAGE_BYTES) + warp * 256;
  const int lane = tid & 31;
  #pragma unroll
  for (int i = 0; i < FM; i++)
    #pragma unroll
    for (int j = 0; j < FN; j++) {
      wmma::store_matrix_sync(scr, acc[i][j], 16, wmma::mem_row_major);
      __syncwarp();
      const int r = lane >> 1, h = lane & 1;
      float4 v0 = *reinterpret_cast<const float4*>(scr + r * 16 + h * 8);
      float4 v1 = *reinterpret_cast<const float4*>(scr + r * 16 + h * 8 + 4);
      uint4 pk; __half2* p2 = reinterpret_cast<__half2*>(&pk);
      p2[0] = __floats2half2_rn(v0.x, v0.y); p2[1] = __floats2half2_rn(v0.z, v0.w);
      p2[2] = __floats2half2_rn(v1.x, v1.y); p2[3] = __floats2half2_rn(v1.z, v1.w);
      *reinterpret_cast<uint4*>(st + (wm * 32 + i * 16 + r) * LDS + wn * BNW + j * 16 + h * 8) = pk;
      __syncwarp();
    }
  __syncthreads();
  if (SWIGLU) {
    constexpr int VPR = BNW / 8;
    for (int v = tid; v < BM * VPR; v += NTHREADS) {
      int r = v / VPR, c8 = (v % VPR) * 8;
      if (r >= nrows) continue;
      uint4 gv = *reinterpret_cast<const uint4*>(st + r * LDS + c8);
      uint4 uv = *reinterpret_cast<const uint4*>(st + r * LDS + BNW + c8);
      const __half2* g2 = reinterpret_cast<const __half2*>(&gv);
      const __half2* u2 = reinterpret_cast<const __half2*>(&uv);
      uint4 ov; __half2* o2 = reinterpret_cast<__half2*>(&ov);
      #pragma unroll
      for (int q = 0; q < 4; q++) {
        float2 g = h2f(g2[q]), u = h2f(u2[q]);
        float r2[2];
        #pragma unroll
        for (int z = 0; z < 2; z++) {
          float gg = z ? g.y : g.x, uu = z ? u.y : u.x;
          float gf = gg;
          if (gcap > 0.f) gf = gcap * tanhf(gg / gcap);
          if (ucap > 0.f) uu = ucap * tanhf(uu / ucap);
          float sig = 1.f / (1.f + expf(-gg));
          r2[z] = gf * sig * uu;
        }
        o2[q] = __floats2half2_rn(r2[0], r2[1]);
      }
      *reinterpret_cast<uint4*>(C + (size_t)(row0 + r) * N + ntile * BNW + c8) = ov;
    }
  } else {
    constexpr int VPR = BN / 8;
    for (int v = tid; v < BM * VPR; v += NTHREADS) {
      int r = v / VPR, c8 = (v % VPR) * 8;
      if (r >= nrows) continue;
      *reinterpret_cast<uint4*>(C + (size_t)(row0 + r) * N + ntile * BN + c8) =
          *reinterpret_cast<const uint4*>(st + r * LDS + c8);
    }
  }
}

// ---------------- weighted reduce ----------------
__global__ void reduce_kernel(const __half* __restrict__ down, const float* __restrict__ wts,
                              const int* __restrict__ inv_pos, __half* __restrict__ out, int H,
                              int topk) {
  int t = blockIdx.x;
  for (int c = threadIdx.x * 8; c < H; c += blockDim.x * 8) {
    float acc[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = 0.f;
    for (int k = 0; k < topk; k++) {
      int p = inv_pos[t * topk + k];
      float w = wts[t * topk + k];
      uint4 v = __ldg(reinterpret_cast<const uint4*>(down + (size_t)p * H + c));
      const __half2* h = reinterpret_cast<const __half2*>(&v);
      #pragma unroll
      for (int q = 0; q < 4; q++) {
        float2 f = h2f(h[q]);
        acc[2 * q] += w * f.x; acc[2 * q + 1] += w * f.y;
      }
    }
    uint4 ov; __half2* o2 = reinterpret_cast<__half2*>(&ov);
    #pragma unroll
    for (int q = 0; q < 4; q++) o2[q] = __floats2half2_rn(acc[2 * q], acc[2 * q + 1]);
    *reinterpret_cast<uint4*>(out + (size_t)t * H + c) = ov;
  }
}

template <int BNW, bool SWIGLU, bool GATHER>
static void launch_gemm(const __half* A, const __half* W, __half* C, const int* sorted_slot,
                        const int4* tiles, const int* ntiles, int K, int N, int topk, int max_tiles,
                        float gcap, float ucap, cudaStream_t stream) {
  constexpr int BN = 2 * BNW;
  constexpr int pipe = 2 * (BM + BN) * (BK + PAD) * 2;
  constexpr int stage = BM * (BN + PAD) * 2 + 8 * 1024;
  constexpr int smem = pipe > stage ? pipe : stage;
  auto kern = gemm_kernel<BNW, SWIGLU, GATHER>;
  static bool init = false;
  if (!init) { cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); init = true; }
  int ntn = SWIGLU ? N / BNW : N / BN;
  dim3 grid(ntn, max_tiles);
  kern<<<grid, NTHREADS, smem, stream>>>(A, W, C, sorted_slot, tiles, ntiles, K, N, topk, gcap, ucap);
}

void dense_moe_prefill(torch::Tensor x, torch::Tensor wts, torch::Tensor ids, torch::Tensor w13,
                       torch::Tensor w2, torch::Tensor out, torch::Tensor sorted_slot,
                       torch::Tensor inv_pos, torch::Tensor tiles, torch::Tensor ntiles,
                       torch::Tensor gate_up_act, torch::Tensor down, double gcap, double ucap,
                       int64_t bnw1) {
  const int T = x.size(0), H = x.size(1), topk = ids.size(1), E = w13.size(0);
  const int I = w2.size(2);
  const int S = T * topk;
  auto stream = at::cuda::getCurrentCUDAStream();
  const int max_tiles = tiles.size(0);
  route_kernel<<<1, 1024, 0, stream>>>(ids.data_ptr<int>(), S, E, sorted_slot.data_ptr<int>(),
                                       inv_pos.data_ptr<int>(),
                                       reinterpret_cast<int4*>(tiles.data_ptr<int>()),
                                       ntiles.data_ptr<int>());
  const __half* xp = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
  const __half* w13p = reinterpret_cast<const __half*>(w13.data_ptr<at::Half>());
  const __half* w2p = reinterpret_cast<const __half*>(w2.data_ptr<at::Half>());
  __half* act = reinterpret_cast<__half*>(gate_up_act.data_ptr<at::Half>());
  __half* dn = reinterpret_cast<__half*>(down.data_ptr<at::Half>());
  const int4* tl = reinterpret_cast<const int4*>(tiles.data_ptr<int>());
  if (bnw1 == 48)
    launch_gemm<48, true, true>(xp, w13p, act, sorted_slot.data_ptr<int>(), tl, ntiles.data_ptr<int>(),
                                H, I, topk, max_tiles, (float)gcap, (float)ucap, stream);
  else if (bnw1 == 64)
    launch_gemm<64, true, true>(xp, w13p, act, sorted_slot.data_ptr<int>(), tl, ntiles.data_ptr<int>(),
                                H, I, topk, max_tiles, (float)gcap, (float)ucap, stream);
  else
    launch_gemm<32, true, true>(xp, w13p, act, sorted_slot.data_ptr<int>(), tl, ntiles.data_ptr<int>(),
                                H, I, topk, max_tiles, (float)gcap, (float)ucap, stream);
  launch_gemm<64, false, false>(act, w2p, dn, sorted_slot.data_ptr<int>(), tl, ntiles.data_ptr<int>(),
                                I, H, topk, max_tiles, 0.f, 0.f, stream);
  reduce_kernel<<<T, 288, 0, stream>>>(dn, wts.data_ptr<float>(), inv_pos.data_ptr<int>(),
                                       reinterpret_cast<__half*>(out.data_ptr<at::Half>()), H, topk);
}
"""

_CPP = """
void dense_moe_prefill(torch::Tensor x, torch::Tensor wts, torch::Tensor ids, torch::Tensor w13,
                       torch::Tensor w2, torch::Tensor out, torch::Tensor sorted_slot,
                       torch::Tensor inv_pos, torch::Tensor tiles, torch::Tensor ntiles,
                       torch::Tensor gate_up_act, torch::Tensor down, double gcap, double ucap,
                       int64_t bnw1);
"""

ENABLED = os.environ.get("VLLM_INKLING2_SM70_DENSE_MOE_PREFILL", "1") == "1"
_BM = 128

_ext = None


def _load():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load_inline

        build = os.path.join(
            os.environ.get(
                "VLLM_INKLING2_SM70_EXT_DIR",
                os.path.expanduser("~/.cache/vllm/inkling2_sm70_ext"),
            ),
            "dense_moe_prefill",
        )
        os.makedirs(build, exist_ok=True)
        _ext = load_inline(
            name="inkling2_dense_moe_prefill",
            cpp_sources=[_CPP],
            cuda_sources=[_SRC],
            functions=["dense_moe_prefill"],
            build_directory=build,
            extra_cuda_cflags=["-O3", "-arch=sm_70"],
            verbose=False,
        )
    return _ext


def _pick_bnw(inter: int) -> int:
    for b in (64, 48, 32):
        if inter % b == 0:
            return b
    return 0


def supported(
    x: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor, ids: torch.Tensor
) -> bool:
    if not (ENABLED and x.is_cuda and x.dtype == torch.float16 and x.is_contiguous()):
        return False
    e, rows13, hidden = w13.shape
    inter = w2.shape[2]
    return (
        w13.dtype == torch.float16
        and w2.dtype == torch.float16
        and rows13 == 2 * inter
        and tuple(w2.shape) == (e, hidden, inter)
        and w13.is_contiguous()
        and w2.is_contiguous()
        and hidden % 128 == 0
        and hidden % 8 == 0
        and hidden % 64 == 0
        and inter % 32 == 0
        and _pick_bnw(inter) > 0
        and x.shape[1] == hidden
        and e <= 256
        and ids.shape[1] == 8
    )


def dense_moe_prefill(
    x: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    gate_cap: float,
    up_cap: float,
) -> torch.Tensor:
    """`out [T, H]` fp16 = sum_k weights[t, k] * W2[ids[t, k]] swiglu(W13[ids[t, k]] x[t]).
    `weights` fp32 `[T, K]`, `ids` int32 `[T, K]`; the caller checks `supported`."""
    ext = _load()
    t, hidden = x.shape
    topk = ids.shape[1]
    e = w13.shape[0]
    inter = w2.shape[2]
    s = t * topk
    dev = x.device
    max_tiles = s // _BM + e
    sorted_slot = torch.empty(s, dtype=torch.int32, device=dev)
    inv_pos = torch.empty(s, dtype=torch.int32, device=dev)
    tiles = torch.empty((max_tiles, 4), dtype=torch.int32, device=dev)
    ntiles = torch.empty(1, dtype=torch.int32, device=dev)
    act = torch.empty((s, inter), dtype=torch.float16, device=dev)
    down = torch.empty((s, hidden), dtype=torch.float16, device=dev)
    out = torch.empty((t, hidden), dtype=torch.float16, device=dev)
    ext.dense_moe_prefill(
        x,
        weights.contiguous(),
        ids.contiguous(),
        w13,
        w2,
        out,
        sorted_slot,
        inv_pos,
        tiles,
        ntiles,
        act,
        down,
        float(gate_cap),
        float(up_cap),
        _pick_bnw(inter),
    )
    return out
