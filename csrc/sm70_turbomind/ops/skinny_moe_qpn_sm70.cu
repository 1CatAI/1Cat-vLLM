// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Skinny NVFP4/MXFP4 GEMM and grouped MoE kernels for SM70/SM75, taken from
// dnv2003/v100-skinny (MIT) with the grouped MoE kernel and the MXFP4 scale
// mode added by its contributors. See LICENSE.v100-skinny in this directory
// for the retained MIT notice.
//
// Weights are prepacked once at load into the fragment order of the card's
// tensor-core MMA ([tile N/32][group K/16][lane 32] x 8B codes plus 32 scale
// bytes per tile and scale group): mma.m8n8k4 on SM70, mma.m16n8k8 from SM75
// on (mma_fragments_sm70.cuh). The permutation lives in
// vllm/model_executor/layers/fused_moe/experts/skinny_sm70_moe.py.

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

#include "mma_fragments_sm70.cuh"

namespace {

using vllm::sm70::fp4x8_to_half2x4;
using vllm::sm70::mma_m16n8k8;

// Register order the expert weights were packed in (QpnFragmentOrder in
// skinny_sm70_moe.py).
enum FragmentOrder { ORDER_M8N8K4 = 0, ORDER_M16N8K8 = 1 };

__device__ __forceinline__ half2 fp8e4m3_to_half2(unsigned char b) {
  const unsigned short hb =
      (((unsigned short)b & 0x80u) << 8) | (((unsigned short)b & 0x7Fu) << 7);
  const half hs =
      __hmul(__ushort_as_half(hb), __ushort_as_half(0x5C00));  // *256
  return __halves2half2(hs, hs);
}

// MXFP4 block scale: E8M0 is a bare exponent, value = 2^(b - 127). fp16
// stores exponent + 15, so the field is b - 112 and the mantissa stays 0 --
// one shift, no multiply. Callers keep the checkpoint's global factor in
// gm2, which is what holds b - 112 inside fp16's normal range (1..30).
__device__ __forceinline__ half2 e8m0_to_half2(unsigned char b) {
  const unsigned short hb = (unsigned short)(((int)b - 112) << 10);
  const half hs = __ushort_as_half(hb);
  return __halves2half2(hs, hs);
}

// Group-scale layouts a weight tile can carry. NVFP4 ships one fp8-e4m3
// scale per 16 codes; MXFP4 one E8M0 scale per 32. Everything else -- code
// packing, fragment order, the decoder, the MMA -- is identical, so the
// kernels take this as a template argument instead of duplicating them.
enum ScaleMode { SCALE_NVFP4_FP8_16 = 0, SCALE_MXFP4_E8M0_32 = 1 };

// Raw scale byte of group g and its decode, split so a caller can issue the
// load ahead of its use.
template <int MODE>
__device__ __forceinline__ uint8_t group_scale_byte(const uint8_t* sb, int g) {
  if (MODE == SCALE_MXFP4_E8M0_32) return __ldg(sb + (size_t)(g >> 1) * 32);
  return __ldg(sb + (size_t)g * 32);
}

template <int MODE>
__device__ __forceinline__ half2 decode_scale(uint8_t b) {
  if (MODE == SCALE_MXFP4_E8M0_32) return e8m0_to_half2(b);
  return fp8e4m3_to_half2(b);
}

template <int MODE>
__device__ __forceinline__ half2 group_scale(const uint8_t* sb, int g) {
  return decode_scale<MODE>(group_scale_byte<MODE>(sb, g));
}

// Eight e2m1 codes times one scale (the 2^14 decoder re-bias is folded into
// the caller's scale). out[i] holds codes (i, i+4) of the 8-code word.
__device__ __forceinline__ void dequant8_tm(unsigned q, half2 sc2p,
                                            half2 out[4]) {
  fp4x8_to_half2x4(q, out);
#pragma unroll
  for (int i = 0; i < 4; ++i) out[i] = __hmul2(out[i], sc2p);
}

// ---------------------------------------------------------------------------
// QPN kernel: Volta-native four-quadpair mma.m8n8k4, M 1..16.
//
// The quadpairs split the N dimension: one warp instruction = four
// independent 8x8x4 MMAs sharing a single 8x4 activation A tile (the A
// fragment map depends only on lane-position inside the quadpair, so
// QP-sibling lanes hold identical A registers). MT template = number of
// 8-row A tiles (MT=2 decodes B once for M 9..16). Weights arrive
// PREPACKED in fragment order ([tile N/32][group K/16][lane 32] x 8B,
// nibbles pre-interleaved so dequant8_tm's (i, i+4) output IS the
// adjacent-k B register pair), built once at weight load by qpn_prepack in
// vllm/model_executor/layers/fused_moe/experts/skinny_sm70_moe.py.
// No smem in the main loop, no barriers except the cross-warp K-reduce
// at output (CTA = 4 warps splitting K to keep the grid at N/32).
// ---------------------------------------------------------------------------

#define SKINNY_MMA_8N8K4(C, A0, A1, B0, B1)                         \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

template <int MT>
__global__ void skinny_nvfp4_qpn(const uint8_t* __restrict__ qcodes,
                                 const uint8_t* __restrict__ qscales,
                                 const half* __restrict__ x,
                                 half* __restrict__ y, int N, int K, int M,
                                 float gscale) {
  constexpr int WARPS = 4;
  __shared__ float cs[WARPS][MT * 256];

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  const int qp = (lane >> 2) & 3;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);  // A row & B local col
  const int G = K >> 4, Gq = G / WARPS;
  const int g0 = warp * Gq;
  const uint2* cb =
      reinterpret_cast<const uint2*>(qcodes) + (size_t)tile * G * 32 + lane;
  const uint8_t* sb = qscales + (size_t)tile * G * 32 + lane;

  const half2 gm2 = __float2half2_rn(gscale * 16384.f);
  float c[MT][8];
#pragma unroll
  for (int t = 0; t < MT; t++)
#pragma unroll
    for (int i = 0; i < 8; i++) c[t][i] = 0.f;

#pragma unroll 4
  for (int g = g0; g < g0 + Gq; g++) {
    const uint2 q2 = __ldcs(cb + (size_t)g * 32);
    const half2 sc2 =
        __hmul2(fp8e4m3_to_half2(__ldg(sb + (size_t)g * 32)), gm2);
    half2 b[8];
    dequant8_tm(q2.x, sc2, b + 0);  // slices 0,1 (k0..7 adjacent pairs)
    dequant8_tm(q2.y, sc2, b + 4);  // slices 2,3 (k8..15)
    const unsigned* B = reinterpret_cast<const unsigned*>(b);
#pragma unroll
    for (int t = 0; t < MT; t++) {
      const int ar = t * 8 + r;
      uint4 a01 = make_uint4(0, 0, 0, 0), a23 = make_uint4(0, 0, 0, 0);
      if (ar < M) {
        const half* xrow = x + (size_t)ar * K;
        a01 = *reinterpret_cast<const uint4*>(xrow + g * 16);
        a23 = *reinterpret_cast<const uint4*>(xrow + g * 16 + 8);
      }
      const unsigned* A0 = reinterpret_cast<const unsigned*>(&a01);
      const unsigned* A1 = reinterpret_cast<const unsigned*>(&a23);
      SKINNY_MMA_8N8K4(c[t], A0[0], A0[1], B[0], B[1]);
      SKINNY_MMA_8N8K4(c[t], A0[2], A0[3], B[2], B[3]);
      SKINNY_MMA_8N8K4(c[t], A1[0], A1[1], B[4], B[5]);
      SKINNY_MMA_8N8K4(c[t], A1[2], A1[3], B[6], B[7]);
    }
  }

  // C map (mma8_probe.cu, roles swapped): reg i of lane L ->
  //   A-row (i&2)|((L&16)?4:0)|(L&1); B-col (i&1)|(((L>>1)&1)<<1)|((i>>2)<<2)
#pragma unroll
  for (int t = 0; t < MT; t++)
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
      const int col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
      cs[warp][(t * 8 + row) * 32 + qp * 8 + col] = c[t][i];
    }
  __syncthreads();  // the only barrier: cross-warp K reduce
  for (int e = threadIdx.x; e < MT * 256; e += blockDim.x) {
    const float v = cs[0][e] + cs[1][e] + cs[2][e] + cs[3][e];
    const int row = e >> 5, col = e & 31;
    if (row < M) y[(size_t)row * N + (size_t)tile * 32 + col] = __float2half(v);
  }
}

// ---------------------------------------------------------------------------
// Grouped MoE QPN kernel: device-side routing (permutation and group
// offsets on the device, inactive experts exit before touching their
// weights, CUDA-graph safe) around the QPN tensor-core dataflow (mma.m8n8k4
// on fragment-order prepacked weights, SPLITK warps splitting K on one N=32
// tile, NACC independent accumulator fragments). Weights are prepacked per
// expert by qpn_prepack: [E][tile N/32][group K/16][lane 32] x 8B codes plus
// the scale bytes. A rows come through the slot indirection (token-major x
// for w13, slot-major for w2).
// Routing arrives COMPACT: grid.y spans min(slots, experts) group slots (a
// static bound, so it captures into CUDA graphs; no batch has more groups),
// `gids[grp]` names the group's expert and `goff[grp]..goff[grp+1]` its slot
// range; padding groups carry an empty range and exit. Either bound alone
// schedules a flood of empty blocks: per expert at decode with many experts
// (E=512, top-k 10, T=1), per slot in prefill (T=512, top-k 6: 3072 rows for
// at most 256 groups, 8-47 % of the GEMM time on V100 and Turing).
// ---------------------------------------------------------------------------
// RB row blocks share one weight load: each group's codes are fetched and
// dequantized once and feed the MMAs of up to RB x 8 rows, instead of one
// re-read per 8-row pass. Every row keeps its own accumulators and the same
// K order, so the result does not depend on RB.

template <int SPLITK, int NACC, int SCALE_MODE = SCALE_NVFP4_FP8_16, int RB = 1>
__global__ void __launch_bounds__(32 * SPLITK)
    skinny_nvfp4_moe_qpn(const uint8_t* __restrict__ qcodes,
                         const uint8_t* __restrict__ qscales,
                         const float* __restrict__ gscales,
                         const half* __restrict__ x, half* __restrict__ y_slots,
                         const int* __restrict__ perm,
                         const int* __restrict__ gids,
                         const int* __restrict__ goff, int N, int K, int topk,
                         int x_slot_major) {
  constexpr int MMAX = 8;
  constexpr int ROWS = RB * MMAX;
  // Activations are staged per warp, XCH groups at a time: the four
  // quadpairs of a warp read the same A rows, so loading them straight from
  // global memory issued every load four times. Rows are padded to 40 halves
  // (80 B) so the eight rows of a fragment fall into distinct banks.
  constexpr int XCH = 2;
  constexpr int XROW = XCH * 16 + 8;
  __shared__ const half* xrows[ROWS];
  __shared__ int slots[ROWS];
  // The activation stage (used while computing; decode, RB = 1, reads A
  // directly and needs none) and the split-K reduction buffer (used after)
  // share one allocation, which keeps two 512-thread blocks resident on a
  // 64 KiB-shared Turing SM.
  constexpr int XS_BYTES = (RB == 1 ? 1 : SPLITK) * RB * MMAX * XROW * 2;
  constexpr int CS_BYTES = SPLITK * 256 * 4;
  __shared__ __align__(
      16) unsigned char smem[XS_BYTES > CS_BYTES ? XS_BYTES : CS_BYTES];
  auto xs = reinterpret_cast<half(*)[RB][MMAX][XROW]>(smem);
  auto cs = reinterpret_cast<float (*)[256]>(smem);

  const int grp = blockIdx.y;
  const int beg = goff[grp];
  const int cnt = goff[grp + 1] - beg;
  if (cnt <= 0) return;  // block-uniform: padding group, no weight read
  const int e = gids[grp];

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  const int qp = (lane >> 2) & 3;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int G = K >> 4, Gq = G / SPLITK;
  const int g0 = warp * Gq;
  const uint2* cb = reinterpret_cast<const uint2*>(qcodes) +
                    ((size_t)e * (N >> 5) + tile) * G * 32 + lane;
  // MXFP4 keeps one scale per 32 codes, so its table is half as long.
  const int SG = (SCALE_MODE == SCALE_MXFP4_E8M0_32) ? (G >> 1) : G;
  const uint8_t* sb = qscales + ((size_t)e * (N >> 5) + tile) * SG * 32 + lane;
  const half2 gm2 = __float2half2_rn(gscales[e] * 16384.f);

  // An expert with more than ROWS slots (hash routing at decode, large
  // prefill chunks) takes one pass per ROWS rows, each re-reading this
  // tile's weights; within a pass the weights are read once.
  for (int base = 0; base < cnt; base += ROWS) {
    const int rows = min(ROWS, cnt - base);
    const int blocks = (rows + MMAX - 1) / MMAX;
    __syncthreads();  // previous pass done with slots/xrows/cs
    if (threadIdx.x < ROWS) {
      const int m = threadIdx.x;
      const int slot = m < rows ? perm[beg + base + m] : 0;
      slots[m] = slot;
      xrows[m] = x + (size_t)(x_slot_major ? slot : slot / topk) * K;
    }
    __syncthreads();
    float c[RB][NACC][8];
#pragma unroll
    for (int bl = 0; bl < RB; bl++)
#pragma unroll
      for (int a = 0; a < NACC; a++)
#pragma unroll
        for (int i = 0; i < 8; i++) c[bl][a][i] = 0.f;

    if constexpr (RB == 1) {
      // Decode (at most 8 rows per expert): staging costs more than the
      // repeated loads it saves, so A comes straight from global memory.
      const half* xr = r < rows ? xrows[r] : nullptr;
#pragma unroll 4
      for (int g = g0; g < g0 + Gq; g++) {
        const uint2 q2 = __ldcs(cb + (size_t)g * 32);
        const half2 sc2 = __hmul2(group_scale<SCALE_MODE>(sb, g), gm2);
        half2 bq[8];
        dequant8_tm(q2.x, sc2, bq + 0);
        dequant8_tm(q2.y, sc2, bq + 4);
        const unsigned* B = reinterpret_cast<const unsigned*>(bq);
        uint4 a01 = make_uint4(0, 0, 0, 0), a23 = make_uint4(0, 0, 0, 0);
        if (xr) {
          a01 = *reinterpret_cast<const uint4*>(xr + g * 16);
          a23 = *reinterpret_cast<const uint4*>(xr + g * 16 + 8);
        }
        const unsigned* A0 = reinterpret_cast<const unsigned*>(&a01);
        const unsigned* A1 = reinterpret_cast<const unsigned*>(&a23);
        SKINNY_MMA_8N8K4(c[0][0], A0[0], A0[1], B[0], B[1]);
        SKINNY_MMA_8N8K4(c[0][1 % NACC], A0[2], A0[3], B[2], B[3]);
        SKINNY_MMA_8N8K4(c[0][2 % NACC], A1[0], A1[1], B[4], B[5]);
        SKINNY_MMA_8N8K4(c[0][3 % NACC], A1[2], A1[3], B[6], B[7]);
      }
    } else {
      // Lane l stages 16 B of row (l >> 2) of each row block per chunk. The
      // next chunk's codes and scales -- streamed from DRAM, unlike the
      // activation rows every tile shares through L2 -- are loaded while the
      // current one computes, hiding the latency that dominated the prefill
      // stalls.
      const int srow = lane >> 2, scol = (lane & 3) * 8;
      const int gend = g0 + Gq;
      // A warp's slice may end on an odd group (K/16 = splitk, for instance),
      // so every read of a chunk's second group is bounded by gend. The guards
      // are uniform across the warp and fall away when Gq is even.
      uint2 qv[XCH];
      uint8_t sv[XCH];
      auto load_weights = [&](int gc) {
#pragma unroll
        for (int gi = 0; gi < XCH; gi++) {
          if (gi > 0 && gc + gi >= gend) break;
          qv[gi] = __ldcs(cb + (size_t)(gc + gi) * 32);
          sv[gi] = group_scale_byte<SCALE_MODE>(sb, gc + gi);
        }
      };
      load_weights(g0);
      for (int gc = g0; gc < gend; gc += XCH) {
        const int gcnt = min(XCH, gend - gc);
#pragma unroll
        for (int bl = 0; bl < RB; bl++) {
          if (bl > 0 && bl >= blocks) break;  // block-uniform
          const int row = bl * MMAX + srow;
          uint4 v = make_uint4(0, 0, 0, 0);
          // scol spans 8 halves inside the chunk, so it belongs to group
          // gc + scol / 16; skip it when that group is past the slice.
          if (row < rows && scol / 16 < gcnt)
            v = *reinterpret_cast<const uint4*>(xrows[row] + gc * 16 + scol);
          *reinterpret_cast<uint4*>(&xs[warp][bl][srow][scol]) = v;
        }
        uint2 q[XCH];
        uint8_t s[XCH];
#pragma unroll
        for (int gi = 0; gi < XCH; gi++) {
          if (gi > 0 && gi >= gcnt) break;
          q[gi] = qv[gi];
          s[gi] = sv[gi];
        }
        __syncwarp();
        if (gc + XCH < gend) load_weights(gc + XCH);
#pragma unroll
        for (int gi = 0; gi < XCH; gi++) {
          if (gi > 0 && gi >= gcnt) break;
          const uint2 q2 = q[gi];
          const half2 sc2 = __hmul2(decode_scale<SCALE_MODE>(s[gi]), gm2);
          half2 bq[8];
          dequant8_tm(q2.x, sc2, bq + 0);
          dequant8_tm(q2.y, sc2, bq + 4);
          const unsigned* B = reinterpret_cast<const unsigned*>(bq);
#pragma unroll
          for (int bl = 0; bl < RB; bl++) {
            // Block 0 always has rows; only later blocks can be empty (block-
            // uniform), and keeping block 0 unguarded leaves RB = 1
            // branch-free.
            if (bl > 0 && bl >= blocks) break;
            const uint4 a01 =
                *reinterpret_cast<const uint4*>(&xs[warp][bl][r][gi * 16]);
            const uint4 a23 =
                *reinterpret_cast<const uint4*>(&xs[warp][bl][r][gi * 16 + 8]);
            const unsigned* A0 = reinterpret_cast<const unsigned*>(&a01);
            const unsigned* A1 = reinterpret_cast<const unsigned*>(&a23);
            SKINNY_MMA_8N8K4(c[bl][0], A0[0], A0[1], B[0], B[1]);
            SKINNY_MMA_8N8K4(c[bl][1 % NACC], A0[2], A0[3], B[2], B[3]);
            SKINNY_MMA_8N8K4(c[bl][2 % NACC], A1[0], A1[1], B[4], B[5]);
            SKINNY_MMA_8N8K4(c[bl][3 % NACC], A1[2], A1[3], B[6], B[7]);
          }
        }
        __syncwarp();  // chunk consumed before the next one overwrites xs
      }
    }

    // Other warps may still read their stage, which the reduction buffer
    // overlaps.
    if constexpr (RB > 1) __syncthreads();
#pragma unroll
    for (int bl = 0; bl < RB; bl++) {
      if (bl > 0 && bl >= blocks) break;  // block-uniform: barriers match
#pragma unroll
      for (int a = 1; a < NACC; a++)
#pragma unroll
        for (int i = 0; i < 8; i++) c[bl][0][i] += c[bl][a][i];
      if (bl > 0) __syncthreads();  // previous block done reading cs
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
        const int col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
        cs[warp][row * 32 + qp * 8 + col] = c[bl][0][i];
      }
      __syncthreads();
      for (int t = threadIdx.x; t < 256; t += blockDim.x) {
        float v = 0.f;
#pragma unroll
        for (int w = 0; w < SPLITK; w++) v += cs[w][t];
        const int row = bl * MMAX + (t >> 5), col = t & 31;
        if (row < rows)
          y_slots[(size_t)slots[row] * N + (size_t)tile * 32 + col] =
              __float2half(v);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Grouped MoE kernel on m16n8k8 fragment-order weights (SM75 and later): the
// routing, grid rows, split-K warps and output of skinny_nvfp4_moe_qpn, with
// the weights as the 16-row A operand and the expert's rows as the 8-column B
// operand (mma_fragments_sm70.cuh). A lane's activation fragment is 8 bytes
// of its own row, so a warp reads each row slice exactly once and needs no
// shared-memory stage. The next group's codes, scales and activations load
// while the current group computes (deeper rings measured slower); RB tiles
// of 8 rows share each weight decode, and NT adjacent 32-column tiles share
// each activation load. The decoded weights equal the m8n8k4 path's bit for
// bit; only the tensor core's summation order differs.
// ---------------------------------------------------------------------------

template <int SPLITK, int SCALE_MODE, int RB, int NT>
__global__ void __launch_bounds__(32 * SPLITK) skinny_nvfp4_moe_m16n8k8(
    const uint8_t* __restrict__ qcodes, const uint8_t* __restrict__ qscales,
    const float* __restrict__ gscales, const half* __restrict__ x,
    half* __restrict__ y_slots, const int* __restrict__ perm,
    const int* __restrict__ gids, const int* __restrict__ goff, int N, int K,
    int topk, int x_slot_major) {
  constexpr int ROWS = RB * 8;
  // 36-float rows put the C fragment stores of a warp into 32 distinct banks.
  constexpr int CS_ROW = 36;
  __shared__ const half* xrows[ROWS];
  __shared__ int slots[ROWS];
  __shared__ float cs[SPLITK][NT][8 * CS_ROW];

  const int grp = blockIdx.y;
  const int beg = goff[grp];
  const int cnt = goff[grp + 1] - beg;
  if (cnt <= 0) return;  // block-uniform: padding group, no weight read
  const int e = gids[grp];

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int tile0 = blockIdx.x * NT;
  const int row_group = lane >> 2;
  const int k_offset = (lane & 3) * 4;
  const int G = K >> 4, Gq = G / SPLITK;
  const int g0 = warp * Gq, gend = g0 + Gq;
  const size_t tile_index = (size_t)e * (N >> 5) + tile0;
  const uint2* cb =
      reinterpret_cast<const uint2*>(qcodes) + tile_index * G * 32 + lane;
  // Four scale bytes per row group: columns row_group + 8j, j = 0..3.
  const int SG = (SCALE_MODE == SCALE_MXFP4_E8M0_32) ? (G >> 1) : G;
  const uint32_t* sb = reinterpret_cast<const uint32_t*>(qscales) +
                       tile_index * SG * 8 + row_group;
  const half2 gm2 = __float2half2_rn(gscales[e] * 16384.f);

  for (int base = 0; base < cnt; base += ROWS) {
    const int rows = min(ROWS, cnt - base);
    const int blocks = (rows + 7) / 8;
    __syncthreads();  // previous pass done with slots/xrows/cs
    if (threadIdx.x < ROWS) {
      const int m = threadIdx.x;
      const int slot = m < rows ? perm[beg + base + m] : 0;
      slots[m] = slot;
      xrows[m] = m < rows ? x + (size_t)(x_slot_major ? slot : slot / topk) * K
                          : nullptr;
    }
    __syncthreads();
    const half* xr[RB];
#pragma unroll
    for (int bl = 0; bl < RB; bl++) xr[bl] = xrows[bl * 8 + row_group];

    float c[NT][RB][2][4];
#pragma unroll
    for (int nt = 0; nt < NT; nt++)
#pragma unroll
      for (int bl = 0; bl < RB; bl++)
#pragma unroll
        for (int t = 0; t < 2; t++)
#pragma unroll
          for (int i = 0; i < 4; i++) c[nt][bl][t][i] = 0.f;

    uint2 q_next[NT], a_next[RB];
    uint32_t s_next[NT];
    auto load_group = [&](int g) {
#pragma unroll
      for (int nt = 0; nt < NT; nt++) {
        q_next[nt] = __ldcs(cb + ((size_t)nt * G + g) * 32);
        s_next[nt] =
            __ldg(sb + ((size_t)nt * SG +
                        (SCALE_MODE == SCALE_MXFP4_E8M0_32 ? g >> 1 : g)) *
                           8);
      }
#pragma unroll
      for (int bl = 0; bl < RB; bl++)
        a_next[bl] =
            xr[bl] ? *reinterpret_cast<const uint2*>(xr[bl] + g * 16 + k_offset)
                   : make_uint2(0, 0);
    };
    load_group(g0);
    for (int g = g0; g < gend; g++) {
      uint2 q[NT], a[RB];
      uint32_t s[NT];
#pragma unroll
      for (int nt = 0; nt < NT; nt++) {
        q[nt] = q_next[nt];
        s[nt] = s_next[nt];
      }
#pragma unroll
      for (int bl = 0; bl < RB; bl++) a[bl] = a_next[bl];
      if (g + 1 < gend) load_group(g + 1);

#pragma unroll
      for (int nt = 0; nt < NT; nt++) {
        // Word t holds A tile t: columns 16t + row_group (scale byte 2t) and
        // 16t + row_group + 8 (scale byte 2t + 1), interleaved per MMA step.
#pragma unroll
        for (int t = 0; t < 2; t++) {
          const half2 s_lo = __hmul2(
              decode_scale<SCALE_MODE>((uint8_t)(s[nt] >> (16 * t))), gm2);
          const half2 s_hi = __hmul2(
              decode_scale<SCALE_MODE>((uint8_t)(s[nt] >> (16 * t + 8))), gm2);
          half2 v[4];
          fp4x8_to_half2x4(t ? q[nt].y : q[nt].x, v);
          v[0] = __hmul2(v[0], s_lo);
          v[1] = __hmul2(v[1], s_hi);
          v[2] = __hmul2(v[2], s_lo);
          v[3] = __hmul2(v[3], s_hi);
          const uint32_t* w = reinterpret_cast<const uint32_t*>(v);
#pragma unroll
          for (int bl = 0; bl < RB; bl++) {
            if (bl > 0 && bl >= blocks) break;  // block-uniform
            mma_m16n8k8(c[nt][bl][t], w[0], w[1], a[bl].x);
            mma_m16n8k8(c[nt][bl][t], w[2], w[3], a[bl].y);
          }
        }
      }
    }

#pragma unroll
    for (int bl = 0; bl < RB; bl++) {
      if (bl > 0 && bl >= blocks) break;  // block-uniform: barriers match
      if (bl > 0) __syncthreads();        // previous block done reading cs
      // C of A tile t: column 16t + row_group (+8 for i >= 2), row
      // (lane & 3) * 2 + (i & 1) of this 8-row block.
#pragma unroll
      for (int nt = 0; nt < NT; nt++)
#pragma unroll
        for (int t = 0; t < 2; t++)
#pragma unroll
          for (int i = 0; i < 4; i++) {
            const int col = 16 * t + row_group + ((i & 2) ? 8 : 0);
            const int row = (lane & 3) * 2 + (i & 1);
            cs[warp][nt][row * CS_ROW + col] = c[nt][bl][t][i];
          }
      __syncthreads();
      for (int idx = threadIdx.x; idx < NT * 256; idx += blockDim.x) {
        const int nt = idx >> 8, row = (idx >> 5) & 7, col = idx & 31;
        float v = 0.f;
#pragma unroll
        for (int split = 0; split < SPLITK; split++)
          v += cs[split][nt][row * CS_ROW + col];
        const int out_row = bl * 8 + row;
        if (out_row < rows)
          y_slots[(size_t)slots[out_row] * N + (size_t)(tile0 + nt) * 32 +
                  col] = __float2half(v);
      }
    }
  }
}

// y_slots[S, N] (S = tokens*topk, slot-major) = x[row(slot)] @ W[expert(slot)]
// with W in per-expert QPN fragment order (see qpn_prepack).

#undef SKINNY_MMA_8N8K4

}  // namespace

torch::Tensor skinny_qpn_gemm_sm70(torch::Tensor x, torch::Tensor qcodes,
                                   torch::Tensor qscales, double gscale,
                                   int64_t n) {
  const int64_t m = x.size(0), k = x.size(1);
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.is_contiguous());
  TORCH_CHECK(qcodes.is_cuda() && qcodes.dtype() == torch::kUInt8 &&
              qcodes.is_contiguous());
  TORCH_CHECK(qscales.is_cuda() && qscales.dtype() == torch::kUInt8 &&
              qscales.is_contiguous());
  TORCH_CHECK(m >= 1 && m <= 16, "qpn supports M 1..16, got ", m);
  TORCH_CHECK(k % 64 == 0, "K % 64 (4-warp split of 16-k groups)");
  TORCH_CHECK(n % 32 == 0, "N % 32");
  TORCH_CHECK(qcodes.numel() == n * (k >> 1), "qpn codes size");
  TORCH_CHECK(qscales.numel() == n * (k >> 4), "qpn scales size");
  auto y = torch::empty({m, n}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  if (m <= 8)
    skinny_nvfp4_qpn<1><<<dim3((int)(n / 32)), dim3(128), 0, stream>>>(
        qcodes.data_ptr<uint8_t>(), qscales.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<half*>(y.data_ptr<at::Half>()), (int)n, (int)k, (int)m,
        (float)gscale);
  else
    skinny_nvfp4_qpn<2><<<dim3((int)(n / 32)), dim3(128), 0, stream>>>(
        qcodes.data_ptr<uint8_t>(), qscales.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<half*>(y.data_ptr<at::Half>()), (int)n, (int)k, (int)m,
        (float)gscale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

void skinny_moe_qpn_sm70(torch::Tensor x, torch::Tensor qcodes,
                         torch::Tensor qscales, torch::Tensor gscales,
                         torch::Tensor perm, torch::Tensor gids,
                         torch::Tensor goff, int64_t topk,
                         torch::Tensor y_slots, bool x_slot_major,
                         int64_t num_tokens, int64_t splitk, int64_t nacc,
                         int64_t scale_mode, int64_t fragment_order) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.is_contiguous());
  const c10::cuda::CUDAGuard device_guard(x.device());
  TORCH_CHECK(fragment_order == ORDER_M8N8K4 || fragment_order == ORDER_M16N8K8,
              "fragment_order 0 (m8n8k4) or 1 (m16n8k8)");
  if (fragment_order == ORDER_M16N8K8) {
    const cudaDeviceProp* props = at::cuda::getDeviceProperties(x.get_device());
    TORCH_CHECK(props->major * 10 + props->minor >= 75,
                "m16n8k8 fragment order needs SM75 or later");
    TORCH_CHECK(nacc == 1, "the m16n8k8 kernel keeps one accumulator set");
  }
  TORCH_CHECK(x.dim() == 2 && y_slots.dim() == 2,
              "x and output must be matrices");
  for (const auto& tensor :
       {qcodes, qscales, gscales, perm, gids, goff, y_slots}) {
    TORCH_CHECK(tensor.device() == x.device(),
                "all skinny MoE tensors must share a device");
  }
  TORCH_CHECK(num_tokens > 0 && topk > 0 && topk <= 65535,
              "positive token count and supported routing width are required");
  TORCH_CHECK(num_tokens <= 65535 / topk, "routing slots exceed CUDA grid-y");
  TORCH_CHECK(gscales.dim() == 1 && gscales.numel() > 0,
              "expert scales must be nonempty");
  TORCH_CHECK(qcodes.is_cuda() && qcodes.dtype() == torch::kUInt8 &&
              qcodes.is_contiguous());
  TORCH_CHECK(qscales.is_cuda() && qscales.dtype() == torch::kUInt8 &&
              qscales.is_contiguous());
  TORCH_CHECK(gscales.is_cuda() && gscales.dtype() == torch::kFloat &&
              gscales.is_contiguous());
  TORCH_CHECK(perm.is_cuda() && perm.dtype() == torch::kInt &&
              perm.is_contiguous());
  TORCH_CHECK(gids.is_cuda() && gids.dtype() == torch::kInt &&
              gids.is_contiguous());
  TORCH_CHECK(goff.is_cuda() && goff.dtype() == torch::kInt &&
              goff.is_contiguous());
  TORCH_CHECK(y_slots.is_cuda() && y_slots.dtype() == torch::kHalf &&
              y_slots.is_contiguous());
  const int64_t T = num_tokens, K = x.size(1);
  const int64_t E = gscales.size(0), N = y_slots.size(1);
  const int64_t S = T * topk;
  TORCH_CHECK(x.size(0) == (x_slot_major ? S : T), "x rows mismatch");
  TORCH_CHECK(qcodes.numel() == E * N * (K >> 1), "qpn codes size");
  TORCH_CHECK(
      scale_mode == SCALE_NVFP4_FP8_16 || scale_mode == SCALE_MXFP4_E8M0_32,
      "scale_mode 0 (NVFP4 fp8/16) or 1 (MXFP4 e8m0/32)");
  // One scale per 16 codes for NVFP4, per 32 for MXFP4.
  const int64_t sh = 4 + scale_mode;
  TORCH_CHECK(qscales.numel() == E * N * (K >> sh), "qpn scales size");
  TORCH_CHECK(gids.size(0) == S && goff.size(0) == S + 1,
              "compact routing size");
  TORCH_CHECK(perm.size(0) == S && y_slots.size(0) == S);
  TORCH_CHECK(S <= 65535, "grouped qpn MoE: tokens * topk = ", S,
              " exceeds the CUDA grid y limit");
  TORCH_CHECK(K % 64 == 0 && (K / 16) % splitk == 0,
              "K/16 must split into splitk equal slices");
  TORCH_CHECK(N % 32 == 0, "N % 32");
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t mean_rows = (S + E - 1) / E;

  if (fragment_order == ORDER_M16N8K8) {
    // Tuning sweep 21.10.2026 on the RTX 8000 (DSv4, GLM-5.3 and Flash-Next
    // expert shapes): two 32-column tiles per CTA sharing each activation
    // load, and four 8-row tiles per weight decode once experts average more
    // than 8 rows, cut the prefill GEMMs to 0.59-0.68 of the m8n8k4 kernel
    // on the same card; decode (one row tile) stays at parity. Four column
    // tiles gained at most 4 % in prefill and lost 3-5 % in decode; deeper
    // load rings measured slower.
    constexpr int kColumnTiles = 2;
    TORCH_CHECK(N % (32 * kColumnTiles) == 0, "m16n8k8 MoE needs N % 64");
    const dim3 grid((unsigned)(N / (32 * kColumnTiles)),
                    (unsigned)std::min(S, E));
    const bool prefill = mean_rows > 8;
#define LAUNCH_MOE_M16N8K8_R(SPv, SMv, RBv)                                 \
  skinny_nvfp4_moe_m16n8k8<SPv, SMv, RBv, kColumnTiles>                     \
      <<<grid, dim3(32 * SPv), 0, stream>>>(                                \
          qcodes.data_ptr<uint8_t>(), qscales.data_ptr<uint8_t>(),          \
          gscales.data_ptr<float>(),                                        \
          reinterpret_cast<const half*>(x.data_ptr<at::Half>()),            \
          reinterpret_cast<half*>(y_slots.data_ptr<at::Half>()),            \
          perm.data_ptr<int>(), gids.data_ptr<int>(), goff.data_ptr<int>(), \
          (int)N, (int)K, (int)topk, x_slot_major ? 1 : 0)

#define LAUNCH_MOE_M16N8K8_S(SPv, SMv)   \
  do {                                   \
    if (prefill)                         \
      LAUNCH_MOE_M16N8K8_R(SPv, SMv, 4); \
    else                                 \
      LAUNCH_MOE_M16N8K8_R(SPv, SMv, 1); \
  } while (0)

#define LAUNCH_MOE_M16N8K8(SPv)                       \
  do {                                                \
    if (scale_mode == SCALE_MXFP4_E8M0_32)            \
      LAUNCH_MOE_M16N8K8_S(SPv, SCALE_MXFP4_E8M0_32); \
    else                                              \
      LAUNCH_MOE_M16N8K8_S(SPv, SCALE_NVFP4_FP8_16);  \
  } while (0)

    switch (splitk) {
      // 4 serves short reductions (K < 2048), whose warps would otherwise get
      // fewer than 16 groups (grouped_splitk in skinny_sm70_moe.py).
      case 4:
        LAUNCH_MOE_M16N8K8(4);
        break;
      case 8:
        LAUNCH_MOE_M16N8K8(8);
        break;
      default:
        TORCH_CHECK(false, "m16n8k8 MoE splitk in {4, 8}");
    }
#undef LAUNCH_MOE_M16N8K8
#undef LAUNCH_MOE_M16N8K8_S
#undef LAUNCH_MOE_M16N8K8_R
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }

  const dim3 grid((unsigned)(N / 32), (unsigned)std::min(S, E));
  // Row blocks per weight load, from the mean rows per expert: one block
  // while experts see at most 8 rows (decode), two for prefill chunks. Four
  // need ~80 registers (halving the resident blocks per SM) and more shared
  // memory than the static 48 KiB for SPLITK 16.
  const int rb = mean_rows > 8 ? 2 : 1;

#define LAUNCH_MOE_QPN_R(SPv, NAv, SMv, RBv)                                \
  skinny_nvfp4_moe_qpn<SPv, NAv, SMv, RBv>                                  \
      <<<grid, dim3(32 * SPv), 0, stream>>>(                                \
          qcodes.data_ptr<uint8_t>(), qscales.data_ptr<uint8_t>(),          \
          gscales.data_ptr<float>(),                                        \
          reinterpret_cast<const half*>(x.data_ptr<at::Half>()),            \
          reinterpret_cast<half*>(y_slots.data_ptr<at::Half>()),            \
          perm.data_ptr<int>(), gids.data_ptr<int>(), goff.data_ptr<int>(), \
          (int)N, (int)K, (int)topk, x_slot_major ? 1 : 0)

#define LAUNCH_MOE_QPN_S(SPv, NAv, SMv)   \
  do {                                    \
    if (rb == 2)                          \
      LAUNCH_MOE_QPN_R(SPv, NAv, SMv, 2); \
    else                                  \
      LAUNCH_MOE_QPN_R(SPv, NAv, SMv, 1); \
  } while (0)

#define LAUNCH_MOE_QPN(SPv, NAv)                       \
  do {                                                 \
    if (scale_mode == SCALE_MXFP4_E8M0_32)             \
      LAUNCH_MOE_QPN_S(SPv, NAv, SCALE_MXFP4_E8M0_32); \
    else                                               \
      LAUNCH_MOE_QPN_S(SPv, NAv, SCALE_NVFP4_FP8_16);  \
  } while (0)

  const int key = (int)(splitk * 10 + nacc);
  switch (key) {
    case 81:
      LAUNCH_MOE_QPN(8, 1);
      break;
    case 82:
      LAUNCH_MOE_QPN(8, 2);
      break;
    // SPLITK 10 serves K = 320 (K/16 = 20 = 10 slices of one 2-group chunk),
    // e.g. a 640-wide expert intermediate split over TP2, which 8 and 16
    // cannot divide.
    case 101:
      LAUNCH_MOE_QPN(10, 1);
      break;
    case 161:
      LAUNCH_MOE_QPN(16, 1);
      break;
    case 162:
      LAUNCH_MOE_QPN(16, 2);
      break;
    // SPLITK 32 would need more than the 48 KiB of static shared memory for
    // the per-warp activation stage plus the split-K reduction.
    default:
      TORCH_CHECK(
          false,
          "moe_qpn (splitk, nacc) in {(8,1), (8,2), (10,1), (16,1), (16,2)}");
  }
#undef LAUNCH_MOE_QPN
#undef LAUNCH_MOE_QPN_S
#undef LAUNCH_MOE_QPN_R
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
