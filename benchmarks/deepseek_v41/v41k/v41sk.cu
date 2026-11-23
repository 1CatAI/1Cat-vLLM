// v41sk: the skinny MXFP4 grouped MoE kernels of
// csrc/sm70_turbomind/ops/skinny_moe_qpn_sm70.cu with the split-K 4/6/9/12/18
// specializations of PR #1030, built as a JIT extension (torch.ops.v41sk) so
// the V4.1 TP4 W2 shape (K = 576) runs without a wheel rebuild. The MIT notice
// of the original kernels is in csrc/sm70_turbomind/ops/LICENSE.v100-skinny.

#include <cstdlib>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>

namespace v41sk {

__device__ __forceinline__ float rp_ex2(float x);
__device__ __forceinline__ float rp_div(float a, float b);

namespace {

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

// XOR swizzle on the low 3 bits of a k-pair index; conflict-free for the

// Alternative e2m1 decoder derived from TurboMind's cvt_f16x8_e2m1
// (Apache-2.0; 1Cat-vLLM csrc/sm70_turbomind/lmdeploy/src/turbomind/
// kernels/attention/quantization.h). Shifts sign/EM bits into fp16
// positions; the 2^14 exponent re-bias is folded into the caller's
// scale, so no extra multiply. Output half2 pairing is INTERLEAVED:
// out[i] holds codes (i, i+4) of the 8-code word.
__device__ __forceinline__ void dequant8_tm(unsigned q, half2 sc2p,
                                            half2 out[4]) {
  constexpr unsigned S = 0x80008000u, EM = 0x0E000E00u;
  unsigned v0 = ((q << 12) & S) | ((q << 9) & EM);
  unsigned v1 = ((q << 8) & S) | ((q << 5) & EM);
  unsigned v2 = ((q << 4) & S) | ((q << 1) & EM);
  unsigned v3 = (q & S) | ((q >> 3) & EM);
  out[0] = __hmul2(*reinterpret_cast<half2*>(&v0), sc2p);
  out[1] = __hmul2(*reinterpret_cast<half2*>(&v1), sc2p);
  out[2] = __hmul2(*reinterpret_cast<half2*>(&v2), sc2p);
  out[3] = __hmul2(*reinterpret_cast<half2*>(&v3), sc2p);
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

// Code words of one N/32 tile for one lane. QPN order is
// [tile][K/16 group][lane][2 words]; built with -DV41SK_TM_CODES=1 the codes
// are instead TurboMind's SM70 E2M1 B/Pack1 order [tile][K/8][column][word]
// (what mxfp4_moe_dense_stage_sm70_out reads), so one copy serves both. The
// nibble order within a word is the same; only the word address moves: lane l
// owns TurboMind column ((l >> 2) & 3) * 8 + (l & 3) + (l & 16 ? 4 : 0) and
// words 2g and 2g + 1 of that column (tm_qpn_layout_check.py).
struct QpnCodes {
#if V41SK_TM_CODES
  const unsigned* p;
  __device__ __forceinline__ QpnCodes(const uint8_t* codes, size_t tile, int G,
                                      int lane)
      : p(reinterpret_cast<const unsigned*>(codes) + tile * G * 64 +
          ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0)) {}
  __device__ __forceinline__ uint2 ld(int g) const {
    const unsigned* w = p + (size_t)g * 64;
    return make_uint2(__ldcs(w), __ldcs(w + 32));
  }
#else
  const uint2* p;
  __device__ __forceinline__ QpnCodes(const uint8_t* codes, size_t tile, int G,
                                      int lane)
      : p(reinterpret_cast<const uint2*>(codes) + tile * G * 32 + lane) {}
  __device__ __forceinline__ uint2 ld(int g) const {
    return __ldcs(p + (size_t)g * 32);
  }
#endif
};

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
  const QpnCodes cb(qcodes, (size_t)tile, G, lane);
  const uint8_t* sb = qscales + (size_t)tile * G * 32 + lane;

  const half2 gm2 = __float2half2_rn(gscale * 16384.f);
  float c[MT][8];
#pragma unroll
  for (int t = 0; t < MT; t++)
#pragma unroll
    for (int i = 0; i < 8; i++) c[t][i] = 0.f;

#pragma unroll 4
  for (int g = g0; g < g0 + Gq; g++) {
    const uint2 q2 = cb.ld(g);
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
// Routing arrives COMPACT: grid.y spans slot-count-many group slots (a
// static bound, so it captures into CUDA graphs), `gids[grp]` names the
// group's expert and `goff[grp]..goff[grp+1]` its slot range; padding
// groups carry an empty range and exit. This keeps the launch independent
// of the expert count -- with a per-expert grid, many-expert models
// (E=512, top-k 10, T=1) schedule thousands of empty blocks that cost
// more than the actual work.
// ---------------------------------------------------------------------------
// RB row blocks share one weight load: each group's codes are fetched and
// dequantized once and feed the MMAs of up to RB x 8 rows, instead of one
// re-read per 8-row pass. Every row keeps its own accumulators and the same
// K order, so the result does not depend on RB.

template <int SPLITK, int NACC, int SCALE_MODE = SCALE_NVFP4_FP8_16, int RB = 1,
          int FIXED_K = 0>
__global__ void __launch_bounds__(32 * SPLITK)
    skinny_nvfp4_moe_qpn(const uint8_t* __restrict__ qcodes,
                         const uint8_t* __restrict__ qscales,
                         const float* __restrict__ gscales,
                         const half* __restrict__ x, half* __restrict__ y_slots,
                         const int* __restrict__ perm,
                         const int* __restrict__ gids,
                         const int* __restrict__ goff, int N, int runtime_k,
                         int topk, int x_slot_major) {
  const int K = FIXED_K ? FIXED_K : runtime_k;
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
  // x_slot_major bits 1..: skip groups above that many rows (VLLM_V41_MOE_PF
  // _GEMM hands them to moe_pf_gemm).
  const int row_cap = x_slot_major >> 1;
  if (cnt <= 0 || (row_cap && cnt > row_cap)) return;
  const int e = gids[grp];

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  const int qp = (lane >> 2) & 3;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int G = K >> 4, Gq = G / SPLITK;
  const int g0 = warp * Gq;
  const QpnCodes cb(qcodes, ((size_t)e * (N >> 5) + tile), G, lane);
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
      xrows[m] = x + (size_t)((x_slot_major & 1) ? slot : slot / topk) * K;
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
        const uint2 q2 = cb.ld(g);
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
          qv[gi] = cb.ld((gc + gi));
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

// ---- decode grouped MoE (every expert sees <= 8 rows per pass) ----
// Same grid, prepacked layout, split-K reduction and FP16 output as
// skinny_nvfp4_moe_qpn, rebuilt for V100 decode where the old RB = 1 loop
// stalled on L2 activation reloads (ncu: 54% long-scoreboard). Each warp
// streams its K slice in chunks of XCH groups through two shared-memory
// activation buffers: while chunk c computes, chunk c + 1's activations and
// weight words are already in flight in registers.
template <int SPLITK, int SCALE_MODE, int XCH, int FIXED_K = 0>
__global__ void __launch_bounds__(32 * SPLITK)
    moe_dec_kernel(const uint8_t* __restrict__ qcodes,
                   const uint8_t* __restrict__ qscales,
                   const float* __restrict__ gscales,
                   const half* __restrict__ x, half* __restrict__ y_slots,
                   const int* __restrict__ perm, const int* __restrict__ gids,
                   const int* __restrict__ goff, int N, int runtime_k, int topk,
                   int x_slot_major) {
  const int K = FIXED_K ? FIXED_K : runtime_k;
  constexpr int MMAX = 8;
  constexpr int XROW = XCH * 16 + 8;  // +8 halves: rows land in distinct banks
  constexpr int XV = XCH * 2;         // 16-byte activation pieces per row/chunk
  constexpr int XPL = XV * MMAX / 32;  // pieces per lane per chunk
  static_assert(XPL >= 1 && (XV * MMAX) % 32 == 0,
                "a chunk must split into whole 16-byte pieces per lane");
  constexpr int XS_BYTES = SPLITK * 2 * MMAX * XROW * 2;
  constexpr int CS_BYTES = SPLITK * 256 * 4;
  __shared__ const half* xrows[MMAX];
  __shared__ int slots[MMAX];
  __shared__ __align__(
      16) unsigned char smem[XS_BYTES > CS_BYTES ? XS_BYTES : CS_BYTES];
  auto xs = reinterpret_cast<half(*)[2][MMAX][XROW]>(smem);
  auto cs = reinterpret_cast<float (*)[256]>(smem);

  int grp = blockIdx.y;
  do {
    const int beg = goff[grp];
    const int cnt = goff[grp + 1] - beg;
    if (cnt <= 0) return;
    const int e = gids[grp];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int tile = blockIdx.x;
    const int qp = (lane >> 2) & 3;
    const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int G = K >> 4, Gq = G / SPLITK;
    const int g0 = warp * Gq, gend = g0 + Gq;
    const QpnCodes cb(qcodes, ((size_t)e * (N >> 5) + tile), G, lane);
    const int SG = (SCALE_MODE == SCALE_MXFP4_E8M0_32) ? (G >> 1) : G;
    const uint8_t* sb = qscales + ((size_t)e * (N >> 5) + tile) * SG * 32 + lane;
    const half2 gm2 = __float2half2_rn(gscales[e] * 16384.f);

    uint2 qv[XCH];
    uint8_t sv[XCH];
    auto load_weights = [&](int gc) {
#pragma unroll
      for (int gi = 0; gi < XCH; gi++) {
        if (gc + gi < gend) {
          qv[gi] = cb.ld((gc + gi));
          sv[gi] = group_scale_byte<SCALE_MODE>(sb, gc + gi);
        }
      }
    };
    // The first chunk's weights are requested before the row setup barrier.
    load_weights(g0);

    for (int base = 0; base < cnt; base += MMAX) {
      const int rows = min(MMAX, cnt - base);
      __syncthreads();  // previous pass done with slots/xrows/cs
      if (threadIdx.x < MMAX) {
        const int m = threadIdx.x;
        const int slot = m < rows ? perm[beg + base + m] : 0;
        slots[m] = slot;
        xrows[m] = x + (size_t)(x_slot_major ? slot : slot / topk) * K;
      }
      __syncthreads();
      if (base > 0) load_weights(g0);
      // lane -> (row, piece) of a chunk: rows 0..7, XV pieces of 8 halves
      uint4 xp[XPL];
      auto load_x = [&](int gc) {
#pragma unroll
        for (int i = 0; i < XPL; i++) {
          const int idx = lane + 32 * i;
          const int row = idx / XV, piece = idx % XV;
          const int k = gc * 16 + piece * 8;
          xp[i] = (row < rows && k < gend * 16)
                      ? *reinterpret_cast<const uint4*>(xrows[row] + k)
                      : make_uint4(0, 0, 0, 0);
        }
      };
      auto store_x = [&](int buf) {
#pragma unroll
        for (int i = 0; i < XPL; i++) {
          const int idx = lane + 32 * i;
          const int row = idx / XV, piece = idx % XV;
          *reinterpret_cast<uint4*>(&xs[warp][buf][row][piece * 8]) = xp[i];
        }
      };
      load_x(g0);
      store_x(0);
      __syncwarp();

      float c[8];
#pragma unroll
      for (int i = 0; i < 8; i++) c[i] = 0.f;
      int buf = 0;
      for (int gc = g0; gc < gend; gc += XCH) {
        const int gcnt = min(XCH, gend - gc);
        uint2 q[XCH];
        uint8_t sc[XCH];
#pragma unroll
        for (int gi = 0; gi < XCH; gi++) {
          q[gi] = qv[gi];
          sc[gi] = sv[gi];
        }
        const bool more = gc + XCH < gend;
        if (more) {
          load_weights(gc + XCH);
          load_x(gc + XCH);
        }
#pragma unroll
        for (int gi = 0; gi < XCH; gi++) {
          if (gi >= gcnt) break;
          const half2 sc2 = __hmul2(decode_scale<SCALE_MODE>(sc[gi]), gm2);
          half2 bq[8];
          dequant8_tm(q[gi].x, sc2, bq + 0);
          dequant8_tm(q[gi].y, sc2, bq + 4);
          const unsigned* B = reinterpret_cast<const unsigned*>(bq);
          const uint4 a01 =
              *reinterpret_cast<const uint4*>(&xs[warp][buf][r][gi * 16]);
          const uint4 a23 =
              *reinterpret_cast<const uint4*>(&xs[warp][buf][r][gi * 16 + 8]);
          const unsigned* A0 = reinterpret_cast<const unsigned*>(&a01);
          const unsigned* A1 = reinterpret_cast<const unsigned*>(&a23);
          SKINNY_MMA_8N8K4(c, A0[0], A0[1], B[0], B[1]);
          SKINNY_MMA_8N8K4(c, A0[2], A0[3], B[2], B[3]);
          SKINNY_MMA_8N8K4(c, A1[0], A1[1], B[4], B[5]);
          SKINNY_MMA_8N8K4(c, A1[2], A1[3], B[6], B[7]);
        }
        if (more) {
          store_x(buf ^ 1);  // the other buffer was last read one chunk ago
          __syncwarp();
          buf ^= 1;
        }
      }
      __syncthreads();  // all stages consumed: cs overlaps them
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
        const int col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
        cs[warp][row * 32 + qp * 8 + col] = c[i];
      }
      __syncthreads();
      for (int t = threadIdx.x; t < 256; t += blockDim.x) {
        float v = 0.f;
#pragma unroll
        for (int w = 0; w < SPLITK; w++) v += cs[w][t];
        const int row = t >> 5, col = t & 31;
        if (row < rows)
          y_slots[(size_t)slots[row] * N + (size_t)tile * 32 + col] =
              __float2half(v);
      }
    }
    if constexpr (FIXED_K == 576) __syncthreads();
  } while (FIXED_K == 576 && (grp += gridDim.y) < 36);
}

// Gate and up projections share one staged activation tile. Separate warp
// groups preserve both four-way K reductions; the epilogue keeps the FP16
// projection rounding before the existing clipped SwiGLU and route weight.
template <int XCH>
__global__ void __launch_bounds__(256) moe_w13_swiglu_kernel(
    const uint8_t* __restrict__ qcodes, const uint8_t* __restrict__ qscales,
    const float* __restrict__ gscales, const half* __restrict__ x,
    const float* __restrict__ probabilities, half* __restrict__ activated,
    const int* __restrict__ perm, const int* __restrict__ gids,
    const int* __restrict__ goff, int intermediate, int topk) {
  constexpr int K = 5120, SPLITK = 4;
  constexpr int SCALE_MODE = SCALE_MXFP4_E8M0_32;
  const int N = 2 * intermediate;
  constexpr int MMAX = 8;
  constexpr int XROW = XCH * 16 + 8;  // +8 halves: rows land in distinct banks
  constexpr int XV = XCH * 2;         // 16-byte activation pieces per row/chunk
  constexpr int XPL = XV * MMAX / 32;  // pieces per lane per chunk
  static_assert(XPL >= 1 && (XV * MMAX) % 32 == 0,
                "a chunk must split into whole 16-byte pieces per lane");
  constexpr int XS_BYTES = SPLITK * 2 * MMAX * XROW * 2;
  constexpr int CS_BYTES = 2 * SPLITK * 256 * 4;
  __shared__ const half* xrows[MMAX];
  __shared__ int slots[MMAX];
  __shared__ float probability[MMAX];
  __shared__ __align__(
      16) unsigned char smem[XS_BYTES > CS_BYTES ? XS_BYTES : CS_BYTES];
  auto xs = reinterpret_cast<half(*)[2][MMAX][XROW]>(smem);
  auto cs = reinterpret_cast<float (*)[256]>(smem);

  const int grp = blockIdx.y;
  const int beg = goff[grp];
  const int cnt = goff[grp + 1] - beg;
  if (cnt <= 0) return;
  const int e = gids[grp];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int projection = warp / SPLITK, local_warp = warp % SPLITK;
  const int tile = blockIdx.x + projection * (intermediate / 32);
  const int qp = (lane >> 2) & 3;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int G = K >> 4, Gq = G / SPLITK;
  const int g0 = local_warp * Gq, gend = g0 + Gq;
  const QpnCodes cb(qcodes, ((size_t)e * (N >> 5) + tile), G, lane);
  const int SG = (SCALE_MODE == SCALE_MXFP4_E8M0_32) ? (G >> 1) : G;
  const uint8_t* sb = qscales + ((size_t)e * (N >> 5) + tile) * SG * 32 + lane;
  const half2 gm2 = __float2half2_rn(gscales[e] * 16384.f);

  uint2 qv[XCH];
  uint8_t sv[XCH];
  auto load_weights = [&](int gc) {
#pragma unroll
    for (int gi = 0; gi < XCH; gi++) {
      if (gc + gi < gend) {
        qv[gi] = cb.ld((gc + gi));
        sv[gi] = group_scale_byte<SCALE_MODE>(sb, gc + gi);
      }
    }
  };
  // The first chunk's weights are requested before the row setup barrier.
  load_weights(g0);

  for (int base = 0; base < cnt; base += MMAX) {
    const int rows = min(MMAX, cnt - base);
    __syncthreads();  // previous pass done with slots/xrows/cs
    if (threadIdx.x < MMAX) {
      const int m = threadIdx.x;
      const int slot = m < rows ? perm[beg + base + m] : 0;
      slots[m] = slot;
      xrows[m] = x + (size_t)(slot / topk) * K;
      probability[m] = m < rows ? probabilities[slot] : 0.f;
    }
    __syncthreads();
    if (base > 0) load_weights(g0);
    // lane -> (row, piece) of a chunk: rows 0..7, XV pieces of 8 halves
    uint4 xp[XPL];
    auto load_x = [&](int gc) {
#pragma unroll
      for (int i = 0; i < XPL; i++) {
        const int idx = lane + 32 * i;
        const int row = idx / XV, piece = idx % XV;
        const int k = gc * 16 + piece * 8;
        xp[i] = (row < rows && k < gend * 16)
                    ? *reinterpret_cast<const uint4*>(xrows[row] + k)
                    : make_uint4(0, 0, 0, 0);
      }
    };
    auto store_x = [&](int buf) {
#pragma unroll
      for (int i = 0; i < XPL; i++) {
        const int idx = lane + 32 * i;
        const int row = idx / XV, piece = idx % XV;
        *reinterpret_cast<uint4*>(&xs[local_warp][buf][row][piece * 8]) = xp[i];
      }
    };
    if (projection == 0) {
      load_x(g0);
      store_x(0);
    }
    __syncthreads();

    float c[8];
#pragma unroll
    for (int i = 0; i < 8; i++) c[i] = 0.f;
    int buf = 0;
    for (int gc = g0; gc < gend; gc += XCH) {
      const int gcnt = min(XCH, gend - gc);
      uint2 q[XCH];
      uint8_t sc[XCH];
#pragma unroll
      for (int gi = 0; gi < XCH; gi++) {
        q[gi] = qv[gi];
        sc[gi] = sv[gi];
      }
      const bool more = gc + XCH < gend;
      if (more) {
        load_weights(gc + XCH);
        if (projection == 0) load_x(gc + XCH);
      }
#pragma unroll
      for (int gi = 0; gi < XCH; gi++) {
        if (gi >= gcnt) break;
        const half2 sc2 = __hmul2(decode_scale<SCALE_MODE>(sc[gi]), gm2);
        half2 bq[8];
        dequant8_tm(q[gi].x, sc2, bq + 0);
        dequant8_tm(q[gi].y, sc2, bq + 4);
        const unsigned* B = reinterpret_cast<const unsigned*>(bq);
        const uint4 a01 =
            *reinterpret_cast<const uint4*>(&xs[local_warp][buf][r][gi * 16]);
        const uint4 a23 = *reinterpret_cast<const uint4*>(
            &xs[local_warp][buf][r][gi * 16 + 8]);
        const unsigned* A0 = reinterpret_cast<const unsigned*>(&a01);
        const unsigned* A1 = reinterpret_cast<const unsigned*>(&a23);
        SKINNY_MMA_8N8K4(c, A0[0], A0[1], B[0], B[1]);
        SKINNY_MMA_8N8K4(c, A0[2], A0[3], B[2], B[3]);
        SKINNY_MMA_8N8K4(c, A1[0], A1[1], B[4], B[5]);
        SKINNY_MMA_8N8K4(c, A1[2], A1[3], B[6], B[7]);
      }
      if (more) {
        if (projection == 0) store_x(buf ^ 1);
        __syncthreads();
        buf ^= 1;
      }
    }
    __syncthreads();  // all stages consumed: cs overlaps them
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
      const int col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
      cs[warp][row * 32 + qp * 8 + col] = c[i];
    }
    __syncthreads();
    for (int t = threadIdx.x; t < 256; t += blockDim.x) {
      float gate_sum = 0.f, up_sum = 0.f;
#pragma unroll
      for (int w = 0; w < SPLITK; w++) {
        gate_sum += cs[w][t];
        up_sum += cs[w + SPLITK][t];
      }
      const int row = t >> 5, col = t & 31;
      if (row < rows) {
        const float gate = fminf(__half2float(__float2half_rn(gate_sum)), 10.f);
        const float up =
            fminf(fmaxf(__half2float(__float2half_rn(up_sum)), -10.f), 10.f);
        const float sigmoid = rp_div(gate, 1.f + rp_ex2(-gate * 1.44269502f));
        const float value = (sigmoid * up) * probability[row];
        activated[(size_t)slots[row] * intermediate + (size_t)blockIdx.x * 32 +
                  col] = __float2half_rn(value);
      }
    }
  }
}

template <int SPLITK, int SCALE_MODE, int XCH, int RB>
__global__ void __launch_bounds__(32 * SPLITK)
    moe_pf_kernel(const uint8_t* __restrict__ qcodes,
                  const uint8_t* __restrict__ qscales,
                  const float* __restrict__ gscales, const half* __restrict__ x,
                  half* __restrict__ y_slots, const int* __restrict__ perm,
                  const int* __restrict__ gids, const int* __restrict__ goff,
                  int N, int K, int topk, int x_slot_major) {
  constexpr int MMAX = 8 * RB;        // rows per pass (RB 8-row blocks)
  constexpr int XROW = XCH * 16 + 8;  // +8 halves: rows land in distinct banks
  constexpr int XV = XCH * 2;         // 16-byte activation pieces per row/chunk
  constexpr int XPL = XV * MMAX / 32;  // pieces per lane per chunk
  static_assert(XPL >= 1 && (XV * MMAX) % 32 == 0,
                "a chunk must split into whole 16-byte pieces per lane");
  constexpr int XS_BYTES = SPLITK * 2 * MMAX * XROW * 2;
  constexpr int CS_BYTES = SPLITK * MMAX * 32 * 4;
  __shared__ const half* xrows[MMAX];
  __shared__ int slots[MMAX];
  extern __shared__ __align__(16) unsigned char smem[];
  auto xs = reinterpret_cast<half(*)[2][MMAX][XROW]>(smem);
  auto cs = reinterpret_cast<float (*)[MMAX * 32]>(smem);

  const int grp = blockIdx.y;
  const int beg = goff[grp];
  const int cnt = goff[grp + 1] - beg;
  if (cnt <= 0) return;
  const int e = gids[grp];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  const int qp = (lane >> 2) & 3;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int G = K >> 4, Gq = G / SPLITK;
  const int g0 = warp * Gq, gend = g0 + Gq;
  const QpnCodes cb(qcodes, ((size_t)e * (N >> 5) + tile), G, lane);
  const int SG = (SCALE_MODE == SCALE_MXFP4_E8M0_32) ? (G >> 1) : G;
  const uint8_t* sb = qscales + ((size_t)e * (N >> 5) + tile) * SG * 32 + lane;
  const half2 gm2 = __float2half2_rn(gscales[e] * 16384.f);

  uint2 qv[XCH];
  uint8_t sv[XCH];
  auto load_weights = [&](int gc) {
#pragma unroll
    for (int gi = 0; gi < XCH; gi++) {
      if (gc + gi < gend) {
        qv[gi] = cb.ld((gc + gi));
        sv[gi] = group_scale_byte<SCALE_MODE>(sb, gc + gi);
      }
    }
  };
  // The first chunk's weights are requested before the row setup barrier.
  load_weights(g0);

  for (int base = 0; base < cnt; base += MMAX) {
    const int rows = min(MMAX, cnt - base);
    __syncthreads();  // previous pass done with slots/xrows/cs
    if (threadIdx.x < MMAX) {
      const int m = threadIdx.x;
      const int slot = m < rows ? perm[beg + base + m] : 0;
      slots[m] = slot;
      xrows[m] = x + (size_t)(x_slot_major ? slot : slot / topk) * K;
    }
    __syncthreads();
    if (base > 0) load_weights(g0);
    // lane -> (row, piece) of a chunk: rows 0..7, XV pieces of 8 halves
    uint4 xp[XPL];
    auto load_x = [&](int gc) {
#pragma unroll
      for (int i = 0; i < XPL; i++) {
        const int idx = lane + 32 * i;
        const int row = idx / XV, piece = idx % XV;
        const int k = gc * 16 + piece * 8;
        xp[i] = (row < rows && k < gend * 16)
                    ? *reinterpret_cast<const uint4*>(xrows[row] + k)
                    : make_uint4(0, 0, 0, 0);
      }
    };
    auto store_x = [&](int buf) {
#pragma unroll
      for (int i = 0; i < XPL; i++) {
        const int idx = lane + 32 * i;
        const int row = idx / XV, piece = idx % XV;
        *reinterpret_cast<uint4*>(&xs[warp][buf][row][piece * 8]) = xp[i];
      }
    };
    load_x(g0);
    store_x(0);
    __syncwarp();

    float c[RB][8];
#pragma unroll
    for (int bl = 0; bl < RB; bl++)
#pragma unroll
      for (int i = 0; i < 8; i++) c[bl][i] = 0.f;
    int buf = 0;
    for (int gc = g0; gc < gend; gc += XCH) {
      const int gcnt = min(XCH, gend - gc);
      uint2 q[XCH];
      uint8_t sc[XCH];
#pragma unroll
      for (int gi = 0; gi < XCH; gi++) {
        q[gi] = qv[gi];
        sc[gi] = sv[gi];
      }
      const bool more = gc + XCH < gend;
      if (more) {
        load_weights(gc + XCH);
        load_x(gc + XCH);
      }
#pragma unroll
      for (int gi = 0; gi < XCH; gi++) {
        if (gi >= gcnt) break;
        const half2 sc2 = __hmul2(decode_scale<SCALE_MODE>(sc[gi]), gm2);
        half2 bq[8];
        dequant8_tm(q[gi].x, sc2, bq + 0);
        dequant8_tm(q[gi].y, sc2, bq + 4);
        const unsigned* B = reinterpret_cast<const unsigned*>(bq);
#pragma unroll
        for (int bl = 0; bl < RB; bl++) {
          const uint4 a01 = *reinterpret_cast<const uint4*>(
              &xs[warp][buf][bl * 8 + r][gi * 16]);
          const uint4 a23 = *reinterpret_cast<const uint4*>(
              &xs[warp][buf][bl * 8 + r][gi * 16 + 8]);
          const unsigned* A0 = reinterpret_cast<const unsigned*>(&a01);
          const unsigned* A1 = reinterpret_cast<const unsigned*>(&a23);
          SKINNY_MMA_8N8K4(c[bl], A0[0], A0[1], B[0], B[1]);
          SKINNY_MMA_8N8K4(c[bl], A0[2], A0[3], B[2], B[3]);
          SKINNY_MMA_8N8K4(c[bl], A1[0], A1[1], B[4], B[5]);
          SKINNY_MMA_8N8K4(c[bl], A1[2], A1[3], B[6], B[7]);
        }
      }
      if (more) {
        store_x(buf ^ 1);  // the other buffer was last read one chunk ago
        __syncwarp();
        buf ^= 1;
      }
    }
    __syncthreads();  // all stages consumed: cs overlaps them
#pragma unroll
    for (int bl = 0; bl < RB; bl++)
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
        const int col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
        cs[warp][(bl * 8 + row) * 32 + qp * 8 + col] = c[bl][i];
      }
    __syncthreads();
    for (int t = threadIdx.x; t < MMAX * 32; t += blockDim.x) {
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < SPLITK; w++) v += cs[w][t];
      const int row = t >> 5, col = t & 31;
      if (row < rows)
        y_slots[(size_t)slots[row] * N + (size_t)tile * 32 + col] =
            __float2half(v);
    }
  }
}

// ---- decode w13 as a persistent stream of (expert, tile) units ----
// moe_dec gives each (tile, expert) its own CTA: ~720 short CTAs whose plan
// loads, row setup and split-K epilogue cost latency, in about two partial
// waves. Here gridDim.x CTAs (one per SM) walk the units u = blockIdx.x +
// i * gridDim.x; the whole activation x[T, K] sits in shared memory (staged
// once), and each warp's next weight chunk -- across unit boundaries too -- is
// in flight while the current chunk computes. Output as moe_dec: FP16 y_slots.
template <int WARPS, int XCH>
__global__ void __launch_bounds__(32 * WARPS)
    moe_w13_units_kernel(const uint8_t* __restrict__ qcodes,
                         const uint8_t* __restrict__ qscales,
                         const float* __restrict__ gscales,
                         const half* __restrict__ x, half* __restrict__ y_slots,
                         const int* __restrict__ perm,
                         const int* __restrict__ gids,
                         const int* __restrict__ goff, int N, int K, int topk,
                         int ngroups_max, int T) {
  constexpr int MMAX = 8;
  const int G = K >> 4, GW = G / WARPS, SG = G >> 1, NT = N >> 5;
  const int XROW = K + 8;
  extern __shared__ __align__(16) unsigned char w13sm[];
  half* xs = reinterpret_cast<half*>(w13sm);  // [T + 1][XROW], row T = zeros
  float* cs = reinterpret_cast<float*>(w13sm + (size_t)(T + 1) * XROW *
                                                   2);  // [WARPS][256]
  __shared__ int sgoff[65], sgid[64], s_ngroups, rowtok[MMAX], rowslot[MMAX];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int qp = (lane >> 2) & 3;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int g0 = warp * GW;
  for (int i = threadIdx.x; i <= ngroups_max; i += blockDim.x) {
    sgoff[i] = goff[i];
    if (i < ngroups_max) sgid[i] = gids[i];
  }
  {
    const int pieces = XROW / 8;  // uint4 per staged row (padding included)
    for (int i = threadIdx.x; i < (T + 1) * pieces; i += blockDim.x) {
      const int row = i / pieces, pc = i % pieces;
      uint4 v = make_uint4(0, 0, 0, 0);
      if (row < T && pc * 8 < K)
        v = *reinterpret_cast<const uint4*>(x + (size_t)row * K + pc * 8);
      *reinterpret_cast<uint4*>(xs + (size_t)row * XROW + pc * 8) = v;
    }
  }
  __syncthreads();
  if (warp == 0) {
    int n = 0;
    for (int b = 0; b < ngroups_max; b += 32) {
      const int i = b + lane;
      n += __popc(__ballot_sync(0xffffffffu,
                                i < ngroups_max && sgoff[i + 1] > sgoff[i]));
    }
    if (lane == 0) s_ngroups = n;
  }
  __syncthreads();
  const int units = s_ngroups * NT;
  const int nchunk = (GW + XCH - 1) / XCH;
  uint2 qv[XCH];
  uint8_t sv[XCH];
  half2 gmv;
  // weights of (unit, chunk) into registers
  auto load_w = [&](int u, int ch) {
    const int grp = u / NT, tile = u % NT, e = sgid[grp];
    const QpnCodes cb(qcodes, ((size_t)e * NT + tile), G, lane);
    const uint8_t* sb = qscales + ((size_t)e * NT + tile) * SG * 32 + lane;
    const int gc = g0 + ch * XCH;
#pragma unroll
    for (int i = 0; i < XCH; ++i) {
      if (gc + i < g0 + GW) {
        qv[i] = cb.ld((gc + i));
        sv[i] = __ldg(sb + (size_t)((gc + i) >> 1) * 32);
      }
    }
    gmv = __float2half2_rn(gscales[e] * 16384.f);
  };
  int u = blockIdx.x;
  if (u < units) load_w(u, 0);
  for (; u < units; u += gridDim.x) {
    const int grp = u / NT, tile = u % NT;
    const int beg = sgoff[grp], cnt = min(MMAX, sgoff[grp + 1] - beg);
    if (threadIdx.x < MMAX) {
      const int slot = threadIdx.x < cnt ? perm[beg + threadIdx.x] : -1;
      rowslot[threadIdx.x] = slot;
      rowtok[threadIdx.x] = slot >= 0 ? slot / topk : T;  // row T is zeros
    }
    __syncthreads();
    const half* xr = xs + (size_t)rowtok[r] * XROW;
    float c[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) c[i] = 0.f;
    for (int ch = 0; ch < nchunk; ++ch) {
      uint2 q[XCH];
      uint8_t s[XCH];
#pragma unroll
      for (int i = 0; i < XCH; ++i) {
        q[i] = qv[i];
        s[i] = sv[i];
      }
      const half2 gm2 = gmv;
      // prefetch: next chunk of this unit, or chunk 0 of the next unit
      if (ch + 1 < nchunk)
        load_w(u, ch + 1);
      else if (u + (int)gridDim.x < units)
        load_w(u + gridDim.x, 0);
      const int gc = g0 + ch * XCH;
#pragma unroll
      for (int i = 0; i < XCH; ++i) {
        if (gc + i >= g0 + GW) break;
        const int g = gc + i;
        const half2 sc2 = __hmul2(e8m0_to_half2(s[i]), gm2);
        half2 bq[8];
        dequant8_tm(q[i].x, sc2, bq + 0);
        dequant8_tm(q[i].y, sc2, bq + 4);
        const unsigned* B = reinterpret_cast<const unsigned*>(bq);
        const uint4 a01 = *reinterpret_cast<const uint4*>(xr + g * 16);
        const uint4 a23 = *reinterpret_cast<const uint4*>(xr + g * 16 + 8);
        const unsigned* A0 = reinterpret_cast<const unsigned*>(&a01);
        const unsigned* A1 = reinterpret_cast<const unsigned*>(&a23);
        SKINNY_MMA_8N8K4(c, A0[0], A0[1], B[0], B[1]);
        SKINNY_MMA_8N8K4(c, A0[2], A0[3], B[2], B[3]);
        SKINNY_MMA_8N8K4(c, A1[0], A1[1], B[4], B[5]);
        SKINNY_MMA_8N8K4(c, A1[2], A1[3], B[6], B[7]);
      }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
      const int col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
      cs[warp * 256 + row * 32 + qp * 8 + col] = c[i];
    }
    __syncthreads();
    for (int t = threadIdx.x; t < 256; t += blockDim.x) {
      const int row = t >> 5, slot = rowslot[row];
      if (slot >= 0) {
        float v = 0.f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) v += cs[w * 256 + t];
        y_slots[(size_t)slot * N + (size_t)tile * 32 + (t & 31)] =
            __float2half(v);
      }
    }
    __syncthreads();
  }
}

void moe_w13_units(torch::Tensor x, torch::Tensor qcodes, torch::Tensor qscales,
                   torch::Tensor gscales, torch::Tensor perm,
                   torch::Tensor gids, torch::Tensor goff, int64_t topk,
                   torch::Tensor y_slots, int64_t ctas, int64_t warps,
                   int64_t xch) {
  const int64_t T = x.size(0), K = x.size(1), N = y_slots.size(1),
                S = y_slots.size(0);
  TORCH_CHECK(x.scalar_type() == torch::kFloat16 && x.is_contiguous() &&
              T <= 8);
  TORCH_CHECK(y_slots.scalar_type() == torch::kFloat16 &&
              y_slots.is_contiguous());
  TORCH_CHECK(K % 16 == 0 && (K / 16) % warps == 0 && N % 32 == 0);
  const int ngroups_max = (int)std::min<int64_t>(S, qscales.size(0));
  TORCH_CHECK(ngroups_max <= 64, "moe_w13_units: at most 64 groups");
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int smem = (int)((T + 1) * (K + 8) * 2 + warps * 256 * 4);
  TORCH_CHECK(smem <= 96 * 1024,
              "moe_w13_units: activation does not fit shared memory");
#define W13U(Wv, Xv)                                                         \
  {                                                                          \
    cudaFuncSetAttribute(moe_w13_units_kernel<Wv, Xv>,                       \
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
    moe_w13_units_kernel<Wv, Xv><<<(unsigned)ctas, 32 * Wv, smem, stream>>>( \
        qcodes.data_ptr<uint8_t>(), qscales.data_ptr<uint8_t>(),             \
        gscales.data_ptr<float>(),                                           \
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),               \
        reinterpret_cast<half*>(y_slots.data_ptr<at::Half>()),               \
        perm.data_ptr<int>(), gids.data_ptr<int>(), goff.data_ptr<int>(),    \
        (int)N, (int)K, (int)topk, ngroups_max, (int)T);                     \
  }
  if (warps == 8 && xch == 8)
    W13U(8, 8)
  else if (warps == 8 && xch == 4)
    W13U(8, 4)
  else if (warps == 16 && xch == 4)
    W13U(16, 4)
  else if (warps == 16 && xch == 8)
    W13U(16, 8)
  else if (warps == 4 && xch == 8)
    W13U(4, 8)
  else
    TORCH_CHECK(false,
                "moe_w13_units: (warps, xch) in {8x8, 8x4, 16x4, 16x8, 4x8}");
#undef W13U
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---- decode w2 + route sum in one launch (every expert sees <= 8 rows) ----
// One CTA per 32-column output tile walks all active experts of the plan:
// W2SK warps split K, the next expert's weight words and activation rows are
// in flight while the current one computes, and the per-token sums over the
// routes accumulate in shared FP32 (plus the shared expert, added last), so
// out[T, N] FP32 is written once -- no [S, N] slot buffer and no route-sum
// kernel. Summation order over routes follows the plan's expert order (FP32;
// not bitwise equal to the slot-buffer path).
template <int W2SK>
__global__ void __launch_bounds__(32 * W2SK) moe_w2_route_kernel(
    const uint8_t* __restrict__ qcodes, const uint8_t* __restrict__ qscales,
    const float* __restrict__ gscales, const half* __restrict__ act,
    const int* __restrict__ perm, const int* __restrict__ gids,
    const int* __restrict__ goff, const half* __restrict__ shared,
    float* __restrict__ out, int N, int K, int topk, int ngroups_max,
    int tokens) {
  constexpr int MMAX = 8;
  const int G = K >> 4, GW = G / W2SK, SG = G >> 1;
  const int XROW = K + 8;  // halves; +8 keeps the eight rows on distinct banks
  extern __shared__ __align__(16) unsigned char w2sm[];
  half* xs = reinterpret_cast<half*>(w2sm);                      // [8][XROW]
  float* cs = reinterpret_cast<float*>(w2sm + MMAX * XROW * 2);  // [W2SK][256]
  float* acc = cs + W2SK * 256;                                  // [8][32]
  __shared__ int rows_tok[MMAX];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  const int qp = (lane >> 2) & 3;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int g0 = warp * GW;
  for (int i = threadIdx.x; i < MMAX * 32; i += blockDim.x) acc[i] = 0.f;
  // The plan in shared memory (one parallel load); active groups come first,
  // padding groups are empty.
  __shared__ int sgoff[65], sgid[64], s_ngroups;
  for (int i = threadIdx.x; i <= ngroups_max; i += blockDim.x) {
    sgoff[i] = goff[i];
    if (i < ngroups_max) sgid[i] = gids[i];
  }
  __syncthreads();
  if (warp == 0) {
    int n = 0;
    for (int b = 0; b < ngroups_max; b += 32) {
      const int i = b + lane;
      n += __popc(__ballot_sync(0xffffffffu,
                                i < ngroups_max && sgoff[i + 1] > sgoff[i]));
    }
    if (lane == 0) s_ngroups = n;
  }
  __syncthreads();
  const int ngroups = s_ngroups;

  constexpr int MAXGW = 12;
  uint2 qv[MAXGW];
  uint8_t sv[MAXGW / 2 + 1];
  half2 gm2v;
  auto load_w = [&](int grp) {
    const int e = sgid[grp];
    const QpnCodes cb(qcodes, ((size_t)e * (N >> 5) + tile), G, lane);
    const uint8_t* sb =
        qscales + ((size_t)e * (N >> 5) + tile) * SG * 32 + lane;
#pragma unroll
    for (int i = 0; i < MAXGW; ++i)
      if (i < GW) qv[i] = cb.ld((g0 + i));
#pragma unroll
    for (int i = 0; i < MAXGW / 2 + 1; ++i)
      if (2 * i < GW + 1 && ((g0 >> 1) + i) * 2 < G)
        sv[i] = __ldg(sb + (size_t)((g0 >> 1) + i) * 32);
    gm2v = __float2half2_rn(gscales[e] * 16384.f);
  };
  // activation rows of a group: lane-strided 16-byte pieces into registers
  const int pieces = K / 8;  // uint4 per row
  constexpr int XPT = 4;  // pieces per thread (8 rows x K/8 pieces / threads)
  uint4 xv[XPT];
  int xcnt = 0;
  auto load_x = [&](int grp) {
    const int beg = sgoff[grp];
    xcnt = sgoff[grp + 1] - beg;
#pragma unroll
    for (int i = 0; i < XPT; ++i) {
      const int idx = threadIdx.x + i * blockDim.x;
      const int row = idx / pieces, pc = idx % pieces;
      xv[i] = (row < xcnt && row < MMAX)
                  ? *reinterpret_cast<const uint4*>(
                        act + (size_t)perm[beg + row] * K + pc * 8)
                  : make_uint4(0, 0, 0, 0);
    }
  };
  if (ngroups > 0) {
    load_w(0);
    load_x(0);
  }
  for (int grp = 0; grp < ngroups; ++grp) {
    // stage this group's rows
#pragma unroll
    for (int i = 0; i < XPT; ++i) {
      const int idx = threadIdx.x + i * blockDim.x;
      const int row = idx / pieces, pc = idx % pieces;
      if (row < MMAX)
        *reinterpret_cast<uint4*>(xs + row * XROW + pc * 8) = xv[i];
    }
    const int beg = sgoff[grp], cnt = min(MMAX, sgoff[grp + 1] - beg);
    if (threadIdx.x < MMAX)
      rows_tok[threadIdx.x] =
          threadIdx.x < cnt ? perm[beg + threadIdx.x] / topk : -1;
    uint2 q[MAXGW];
    uint8_t s[MAXGW / 2 + 1];
#pragma unroll
    for (int i = 0; i < MAXGW; ++i) q[i] = qv[i];
#pragma unroll
    for (int i = 0; i < MAXGW / 2 + 1; ++i) s[i] = sv[i];
    const half2 gm2 = gm2v;
    __syncthreads();
    if (grp + 1 < ngroups) {
      load_w(grp + 1);
      load_x(grp + 1);
    }
    float c[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) c[i] = 0.f;
#pragma unroll
    for (int i = 0; i < MAXGW; ++i) {
      if (i >= GW) break;
      const int g = g0 + i;
      const half2 sc2 = __hmul2(e8m0_to_half2(s[(g >> 1) - (g0 >> 1)]), gm2);
      half2 bq[8];
      dequant8_tm(q[i].x, sc2, bq + 0);
      dequant8_tm(q[i].y, sc2, bq + 4);
      const unsigned* B = reinterpret_cast<const unsigned*>(bq);
      const uint4 a01 = *reinterpret_cast<const uint4*>(xs + r * XROW + g * 16);
      const uint4 a23 =
          *reinterpret_cast<const uint4*>(xs + r * XROW + g * 16 + 8);
      const unsigned* A0 = reinterpret_cast<const unsigned*>(&a01);
      const unsigned* A1 = reinterpret_cast<const unsigned*>(&a23);
      SKINNY_MMA_8N8K4(c, A0[0], A0[1], B[0], B[1]);
      SKINNY_MMA_8N8K4(c, A0[2], A0[3], B[2], B[3]);
      SKINNY_MMA_8N8K4(c, A1[0], A1[1], B[4], B[5]);
      SKINNY_MMA_8N8K4(c, A1[2], A1[3], B[6], B[7]);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
      const int col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
      cs[warp * 256 + row * 32 + qp * 8 + col] = c[i];
    }
    __syncthreads();
    for (int t = threadIdx.x; t < 256; t += blockDim.x) {
      const int row = t >> 5;
      const int tok = rows_tok[row];
      if (tok >= 0) {
        float v = 0.f;
#pragma unroll
        for (int w = 0; w < W2SK; ++w) v += cs[w * 256 + t];
        // FP16 slot value (as the slot buffer path stores it), summed in FP32
        acc[tok * 32 + (t & 31)] += __half2float(__float2half(v));
      }
    }
    __syncthreads();
  }
  for (int t = threadIdx.x; t < tokens * 32; t += blockDim.x) {
    const int tok = t >> 5, col = t & 31;
    float v = acc[t];
    if (shared) v += __half2float(shared[(size_t)tok * N + tile * 32 + col]);
    out[(size_t)tok * N + tile * 32 + col] = v;
  }
}

void moe_w2_route(torch::Tensor act, torch::Tensor qcodes,
                  torch::Tensor qscales, torch::Tensor gscales,
                  torch::Tensor perm, torch::Tensor gids, torch::Tensor goff,
                  int64_t topk, std::optional<torch::Tensor> shared,
                  torch::Tensor out, int64_t warps) {
  const int64_t S = act.size(0), K = act.size(1), N = out.size(1),
                T = out.size(0);
  TORCH_CHECK(act.scalar_type() == torch::kFloat16 && act.is_contiguous());
  TORCH_CHECK(out.scalar_type() == torch::kFloat32 && out.is_contiguous() &&
              T <= 8);
  TORCH_CHECK(K % 16 == 0 && (K / 16) % warps == 0 && (K / 16) / warps <= 12 &&
              N % 32 == 0);
  TORCH_CHECK(8 * (K / 8) <= 4 * 32 * warps,
              "moe_w2_route: rows x pieces > 4 per thread");
  TORCH_CHECK(!shared.has_value() ||
              (shared->scalar_type() == torch::kFloat16 &&
               shared->is_contiguous() && shared->size(1) == N));
  const at::cuda::OptionalCUDAGuard guard(device_of(act));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int smem = 8 * (K + 8) * 2 + warps * 256 * 4 + 8 * 32 * 4;
  const int ngroups_max = (int)std::min<int64_t>(S, qscales.size(0));
  TORCH_CHECK(ngroups_max <= 64, "moe_w2_route: at most 64 groups");
#define W2R(Wv)                                                             \
  {                                                                         \
    static bool set = false;                                                \
    if (!set) {                                                             \
      cudaFuncSetAttribute(moe_w2_route_kernel<Wv>,                         \
                           cudaFuncAttributeMaxDynamicSharedMemorySize,     \
                           64 * 1024);                                      \
      set = true;                                                           \
    }                                                                       \
    moe_w2_route_kernel<Wv><<<(unsigned)(N / 32), 32 * Wv, smem, stream>>>( \
        qcodes.data_ptr<uint8_t>(), qscales.data_ptr<uint8_t>(),            \
        gscales.data_ptr<float>(),                                          \
        reinterpret_cast<const half*>(act.data_ptr<at::Half>()),            \
        perm.data_ptr<int>(), gids.data_ptr<int>(), goff.data_ptr<int>(),   \
        shared.has_value()                                                  \
            ? reinterpret_cast<const half*>(shared->data_ptr<at::Half>())   \
            : nullptr,                                                      \
        out.data_ptr<float>(), (int)N, (int)K, (int)topk, ngroups_max,      \
        (int)T);                                                            \
  }
  if (warps == 3)
    W2R(3)
  else if (warps == 4)
    W2R(4)
  else if (warps == 6)
    W2R(6)
  else if (warps == 9)
    W2R(9)
  else
    TORCH_CHECK(false, "moe_w2_route: warps in {3, 4, 6, 9}");
#undef W2R
  C10_CUDA_KERNEL_LAUNCH_CHECK();
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

// Launch one (SPLITK, NACC, scale, RB) instance; combinations whose static
// activation stage exceeds 48 KiB are never instantiated (the host clamps rb).
template <int SP, int NA, int SM, int RB>
void launch_qpn(dim3 grid, cudaStream_t stream, const uint8_t* qc,
                const uint8_t* qs, const float* gs, const half* x, half* y,
                const int* perm, const int* gids, const int* goff, int N, int K,
                int topk, int xsm) {
  if constexpr ((RB == 1 ? 1 : SP) * RB * 8 * 40 * 2 <= 48 * 1024 &&
                (NA == 1 || RB <= 2)) {
    if constexpr (SP == 6 && NA == 1 && SM == SCALE_MXFP4_E8M0_32 && RB == 1) {
      if (K == 576) {
        skinny_nvfp4_moe_qpn<SP, NA, SM, RB, 576>
            <<<grid, dim3(32 * SP), 0, stream>>>(qc, qs, gs, x, y, perm, gids,
                                                 goff, N, K, topk, xsm);
        return;
      }
    }
    skinny_nvfp4_moe_qpn<SP, NA, SM, RB><<<grid, dim3(32 * SP), 0, stream>>>(
        qc, qs, gs, x, y, perm, gids, goff, N, K, topk, xsm);
  } else {
    TORCH_CHECK(false, "skinny qpn: row-block instance not built");
  }
}

void skinny_moe_qpn_sm70(torch::Tensor x, torch::Tensor qcodes,
                         torch::Tensor qscales, torch::Tensor gscales,
                         torch::Tensor perm, torch::Tensor gids,
                         torch::Tensor goff, int64_t topk,
                         torch::Tensor y_slots, bool x_slot_major,
                         int64_t num_tokens, int64_t splitk, int64_t nacc,
                         int64_t scale_mode, int64_t max_rows) {
  TORCH_CHECK(max_rows >= 0 && max_rows < (1 << 20), "max_rows");
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.is_contiguous());
  const c10::cuda::CUDAGuard device_guard(x.device());
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
  // 4, 6, 9, 12 and 18 serve K/16 = 36 and 18 (DeepSeek-V4.1 W2 at TP4,
  // K = 576, and TP8, K = 288), which 8, 10 and 16 cannot divide. They are
  // instantiated with one accumulator only.
  TORCH_CHECK(splitk == 4 || splitk == 6 || splitk == 8 || splitk == 9 ||
                  splitk == 10 || splitk == 12 || splitk == 16 || splitk == 18,
              "unsupported split-K specialization");
  TORCH_CHECK(nacc == 1 || (nacc == 2 && (splitk == 8 || splitk == 16)),
              "unsupported accumulator specialization");
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
  // The grouped kernel walks K in 16-code groups and reads MXFP4 scales per
  // pair of groups, so K % 32 is its real requirement (the dense QPN kernel
  // above keeps K % 64 for its fixed four-warp split).
  TORCH_CHECK(K % 32 == 0 && (K / 16) % splitk == 0,
              "K/16 must split into splitk equal slices");
  TORCH_CHECK(N % 32 == 0, "N % 32");
  // At most one group per expert: slots past E would only launch empty CTAs
  // (a 4K-token prefill chunk has S = 24.5K slots but <= 384 groups).
  const dim3 grid((unsigned)(N / 32), (unsigned)std::min<int64_t>(S, E));
  auto stream = at::cuda::getCurrentCUDAStream();
  // Row blocks per weight load, from the mean rows per expert: one block
  // while experts see at most 8 rows (decode), two for prefill chunks. Four
  // need ~80 registers (halving the resident blocks per SM) and more shared
  // memory than the static 48 KiB for SPLITK 16.
  const int64_t mean_rows = (S + E - 1) / E;
  int rb = mean_rows > 8 ? 2 : 1;
  // V41SK_RB_MAX (default 2): allow 4 or 8 row blocks per weight load for
  // long prefill chunks, bounded by the static 48 KiB activation stage.
  static const int rb_max = [] {
    const char* v = std::getenv("V41SK_RB_MAX");
    return v ? std::atoi(v) : 2;
  }();
  while (rb < rb_max && mean_rows > rb * 8 && rb < 8) rb *= 2;
  while (rb > 2 && splitk * rb * 8 * 40 * 2 > 48 * 1024) rb /= 2;

#define LAUNCH_MOE_QPN_R(SPv, NAv, SMv, RBv)                                 \
  launch_qpn<SPv, NAv, SMv, RBv>(                                            \
      grid, stream, qcodes.data_ptr<uint8_t>(), qscales.data_ptr<uint8_t>(), \
      gscales.data_ptr<float>(),                                             \
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),                 \
      reinterpret_cast<half*>(y_slots.data_ptr<at::Half>()),                 \
      perm.data_ptr<int>(), gids.data_ptr<int>(), goff.data_ptr<int>(),      \
      (int)N, (int)K, (int)topk, (x_slot_major ? 1 : 0) | (int)(max_rows << 1))

#define LAUNCH_MOE_QPN_S(SPv, NAv, SMv)   \
  do {                                    \
    if (rb == 8)                          \
      LAUNCH_MOE_QPN_R(SPv, NAv, SMv, 8); \
    else if (rb == 4)                     \
      LAUNCH_MOE_QPN_R(SPv, NAv, SMv, 4); \
    else if (rb == 2)                     \
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
    case 41:
      LAUNCH_MOE_QPN(4, 1);
      break;
    case 61:
      LAUNCH_MOE_QPN(6, 1);
      break;
    case 91:
      LAUNCH_MOE_QPN(9, 1);
      break;
    case 121:
      LAUNCH_MOE_QPN(12, 1);
      break;
    case 181:
      LAUNCH_MOE_QPN(18, 1);
      break;
    default:
      TORCH_CHECK(false,
                  "moe_qpn (splitk, nacc) in {(4,1), (6,1), (8,1), (8,2), "
                  "(9,1), (10,1), (12,1), (16,1), (16,2), (18,1)}");
  }
#undef LAUNCH_MOE_QPN
#undef LAUNCH_MOE_QPN_S
#undef LAUNCH_MOE_QPN_R
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
// Decode grouped MoE (moe_dec_kernel): MXFP4 E8M0-32 experts, at most 8
// rows per expert and pass. splitk warps per CTA, xch 16-code groups per
// activation chunk (2, 4 or 8; 8 needs splitk <= 8 for static shared memory).
void skinny_moe_dec_sm70(torch::Tensor x, torch::Tensor qcodes,
                         torch::Tensor qscales, torch::Tensor gscales,
                         torch::Tensor perm, torch::Tensor gids,
                         torch::Tensor goff, int64_t topk,
                         torch::Tensor y_slots, bool x_slot_major,
                         int64_t splitk, int64_t xch) {
  const int64_t K = x.size(1), N = y_slots.size(1), S = perm.size(0);
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.is_contiguous());
  TORCH_CHECK(y_slots.dtype() == torch::kHalf && y_slots.is_contiguous());
  TORCH_CHECK(qcodes.dtype() == torch::kUInt8 &&
              qscales.dtype() == torch::kUInt8);
  TORCH_CHECK(gids.size(0) == S && goff.size(0) == S + 1 &&
              y_slots.size(0) == S);
  TORCH_CHECK(K % 32 == 0 && (K / 16) % splitk == 0 && N % 32 == 0);
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  // Six DSpark verification rows have at most 36 routed slots. Each CTA
  // visits more compacted groups only when they exist. Unused groups trail
  // the live prefix in goff, and leave without reading expert weights.
  const bool short_w2 = x_slot_major && K == 576 && N == 5120 && S == 36 &&
                        topk == 6 && splitk == 4 && xch == 4;
  const dim3 grid((unsigned)(N / 32),
                  (unsigned)std::min<int64_t>(short_w2 ? 10 : S, qscales.size(0)));
  auto stream = at::cuda::getCurrentCUDAStream();
#define MOE_DEC(SPv, XCv, KV)                                               \
  moe_dec_kernel<SPv, SCALE_MXFP4_E8M0_32, XCv, KV>                         \
      <<<grid, dim3(32 * SPv), 0, stream>>>(                                \
          qcodes.data_ptr<uint8_t>(), qscales.data_ptr<uint8_t>(),          \
          gscales.data_ptr<float>(),                                        \
          reinterpret_cast<const half*>(x.data_ptr<at::Half>()),            \
          reinterpret_cast<half*>(y_slots.data_ptr<at::Half>()),            \
          perm.data_ptr<int>(), gids.data_ptr<int>(), goff.data_ptr<int>(), \
          (int)N, (int)K, (int)topk, x_slot_major ? 1 : 0)
  bool ok = true;
  if (short_w2)
    MOE_DEC(4, 4, 576);
  else if (splitk == 4 && xch == 8 && K == 5120)
    MOE_DEC(4, 8, 5120);
  else if (splitk == 4 && xch == 8)
    MOE_DEC(4, 8, 0);
  else if (splitk == 4 && xch == 4)
    MOE_DEC(4, 4, 0);
  else if (splitk == 4 && xch == 2)
    MOE_DEC(4, 2, 0);
  else if (splitk == 8 && xch == 8)
    MOE_DEC(8, 8, 0);
  else if (splitk == 8 && xch == 4)
    MOE_DEC(8, 4, 0);
  else if (splitk == 8 && xch == 2)
    MOE_DEC(8, 2, 0);
  else if (splitk == 16 && xch == 4)
    MOE_DEC(16, 4, 0);
  else if (splitk == 16 && xch == 2)
    MOE_DEC(16, 2, 0);
  else
    ok = false;
#undef MOE_DEC
  TORCH_CHECK(
      ok, "skinny_moe_dec: (splitk, xch) in {4, 8} x {2, 4, 8}, {16} x {2, 4}");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void moe_w13_swiglu(torch::Tensor x, torch::Tensor qcodes,
                    torch::Tensor qscales, torch::Tensor gscales,
                    torch::Tensor probabilities, torch::Tensor perm,
                    torch::Tensor gids, torch::Tensor goff, int64_t topk,
                    torch::Tensor activated, int64_t xch) {
  // Any token count: each group stages its rows 8 at a time (expert-parallel
  // ranks see every DP rank's gathered tokens).
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.is_contiguous() &&
              x.dim() == 2 && x.size(1) == 5120 && x.size(0) >= 1 &&
              x.size(0) * topk <= 65535);
  const c10::cuda::CUDAGuard guard(x.device());
  for (const auto& tensor :
       {qcodes, qscales, gscales, probabilities, perm, gids, goff, activated})
    TORCH_CHECK(tensor.device() == x.device() && tensor.is_contiguous());
  TORCH_CHECK(activated.dim() == 2 && activated.dtype() == torch::kHalf);
  TORCH_CHECK(gscales.dim() == 1 && gscales.dtype() == torch::kFloat &&
              gscales.numel() > 0 && probabilities.dtype() == torch::kFloat);
  TORCH_CHECK(qcodes.dtype() == torch::kUInt8 &&
              qscales.dtype() == torch::kUInt8);
  TORCH_CHECK(perm.dtype() == torch::kInt && gids.dtype() == torch::kInt &&
              goff.dtype() == torch::kInt);
  TORCH_CHECK(topk >= 1 && topk <= 8 && (xch == 2 || xch == 4 || xch == 8));
  const int64_t slots = x.size(0) * topk, intermediate = activated.size(1);
  TORCH_CHECK(intermediate > 0 && intermediate % 32 == 0 &&
              activated.size(0) == slots && perm.numel() == slots &&
              gids.numel() == slots && goff.numel() == slots + 1 &&
              probabilities.numel() == slots);
  TORCH_CHECK(qcodes.numel() == gscales.numel() * intermediate * 5120 &&
              qscales.numel() == gscales.numel() * 2 * intermediate * 160);
  const dim3 grid(intermediate / 32, std::min<int64_t>(slots, gscales.numel()));
  const auto stream = at::cuda::getCurrentCUDAStream();
#define W13_SWIGLU(CHUNK)                                               \
  moe_w13_swiglu_kernel<CHUNK><<<grid, 256, 0, stream>>>(               \
      qcodes.data_ptr<uint8_t>(), qscales.data_ptr<uint8_t>(),          \
      gscales.data_ptr<float>(),                                        \
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),            \
      probabilities.data_ptr<float>(),                                  \
      reinterpret_cast<half*>(activated.data_ptr<at::Half>()),          \
      perm.data_ptr<int>(), gids.data_ptr<int>(), goff.data_ptr<int>(), \
      (int)intermediate, (int)topk)
  if (xch == 2)
    W13_SWIGLU(2);
  else if (xch == 4)
    W13_SWIGLU(4);
  else
    W13_SWIGLU(8);
#undef W13_SWIGLU
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Prefill: moe_dec's double-buffered stream with RB 8-row blocks per weight
// load (16 or 32 rows per pass), dynamic shared memory.
void skinny_moe_pf_sm70(torch::Tensor x, torch::Tensor qcodes,
                        torch::Tensor qscales, torch::Tensor gscales,
                        torch::Tensor perm, torch::Tensor gids,
                        torch::Tensor goff, int64_t topk, torch::Tensor y_slots,
                        bool x_slot_major, int64_t splitk, int64_t xch,
                        int64_t rb) {
  const int64_t K = x.size(1), N = y_slots.size(1), S = perm.size(0);
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.is_contiguous());
  TORCH_CHECK(y_slots.dtype() == torch::kHalf && y_slots.is_contiguous());
  TORCH_CHECK(gids.size(0) == S && goff.size(0) == S + 1 &&
              y_slots.size(0) == S);
  TORCH_CHECK(K % 32 == 0 && (K / 16) % splitk == 0 && N % 32 == 0);
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  const dim3 grid((unsigned)(N / 32),
                  (unsigned)std::min<int64_t>(S, qscales.size(0)));
  auto stream = at::cuda::getCurrentCUDAStream();
#define MOE_PF(SPv, XCv, RBv)                                                 \
  {                                                                           \
    constexpr int XROW = XCv * 16 + 8;                                        \
    constexpr int XS = SPv * 2 * 8 * RBv * XROW * 2,                          \
                  CSB = SPv * 8 * RBv * 32 * 4;                               \
    constexpr int SM = XS > CSB ? XS : CSB;                                   \
    TORCH_CHECK(SM <= 96 * 1024,                                              \
                "skinny_moe_pf: stage exceeds shared memory");                \
    cudaFuncSetAttribute(moe_pf_kernel<SPv, SCALE_MXFP4_E8M0_32, XCv, RBv>,   \
                         cudaFuncAttributeMaxDynamicSharedMemorySize, SM);    \
    moe_pf_kernel<SPv, SCALE_MXFP4_E8M0_32, XCv, RBv>                         \
        <<<grid, 32 * SPv, SM, stream>>>(                                     \
            qcodes.data_ptr<uint8_t>(), qscales.data_ptr<uint8_t>(),          \
            gscales.data_ptr<float>(),                                        \
            reinterpret_cast<const half*>(x.data_ptr<at::Half>()),            \
            reinterpret_cast<half*>(y_slots.data_ptr<at::Half>()),            \
            perm.data_ptr<int>(), gids.data_ptr<int>(), goff.data_ptr<int>(), \
            (int)N, (int)K, (int)topk, x_slot_major ? 1 : 0);                 \
  }
  const int key = (int)(splitk * 100 + xch * 10 + rb);
  switch (key) {
    case 482:
      MOE_PF(4, 8, 2) break;
    case 484:
      MOE_PF(4, 8, 4) break;
    case 442:
      MOE_PF(4, 4, 2) break;
    case 444:
      MOE_PF(4, 4, 4) break;
    case 882:
      MOE_PF(8, 8, 2) break;
    case 842:
      MOE_PF(8, 4, 2) break;
    case 844:
      MOE_PF(8, 4, 4) break;
    case 642:
      MOE_PF(6, 4, 2) break;
    case 644:
      MOE_PF(6, 4, 4) break;
    case 622:
      MOE_PF(6, 2, 2) break;
    case 624:
      MOE_PF(6, 2, 4) break;
    default:
      TORCH_CHECK(false, "skinny_moe_pf: unsupported (splitk, xch, rb)");
  }
#undef MOE_PF
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---- V4.1 decode attention over gathered latent keys (MQA, V = K) ----
// out[t, h] = sum_j softmax_j(scale * q[t,h] . k[t,j]; sink[h]) k[t,j], with
// unused keys masked. Split over key chunks of KC: one CTA (8 warps) per
// (token, chunk) does QK (WMMA 16x16x16, FP32 accumulate), a masked partial
// softmax, and P.K, writing an FP32 partial and its (max, sum); a combine
// kernel merges the chunks and the sink. Keys are read once from DRAM.
namespace attn {
constexpr int H = 16, D = 512, KC = 32;
}
// Dynamic shared memory layout (halves unless noted), 16-element padding keeps
// WMMA tiles 32-byte aligned and rows off the same banks.
namespace attn {
constexpr int QLD = D + 16, KLD = D + 16, SLD = KC + 4, PLD = KC + 16;
constexpr size_t Q_OFF = 0;
constexpr size_t K_OFF = Q_OFF + sizeof(half) * H * QLD;
constexpr size_t S_OFF = K_OFF + sizeof(half) * KC * KLD;  // float [2][H][SLD]
constexpr size_t P_OFF = S_OFF + sizeof(float) * 2 * H * SLD;  // half [H][PLD]
constexpr size_t T_OFF = P_OFF + sizeof(half) * H * PLD;       // float [4][256]
constexpr size_t SMEM = T_OFF + sizeof(float) * 4 * 256;
constexpr int DEC_WARPS = 32;
constexpr size_t DEC_SMEM = T_OFF + sizeof(float) * DEC_WARPS * 256;
}  // namespace attn
// Ring gather (attention_ops._gather_attention_ring) done in the kernel: key
// j < window comes from this forward's latents or the window ring, later keys
// from the paged compressed cache through the top-k indices.
// Packed compressed rows (official DeepInfer V41_FP4_FORMAT, 288 bytes:
// 256 B E2M1 nibbles, element 2i in the low nibble of byte i, then 32 E4M3
// scales, one per 16 values). Gather sources point at such rows with bit 0
// set; v41_key8 decodes 8 values to FP16 (FP32 product, round to nearest),
// bit for bit the FP16 cache rows the QDQ wrote before.
constexpr int V41_C_ROW = 288, V41_I_ROW = 68;

// FP16 bits of an E2M1 code (sign << 3 | exponent << 1 | mantissa).
__device__ __forceinline__ unsigned v41_e2m1_h(unsigned code) {
  const unsigned e = (code >> 1) & 3u, m = code & 1u;
  const unsigned mag = e ? ((e + 14u) << 10) | (m << 9) : (m ? 0x3800u : 0u);
  return mag | ((code & 8u) << 12);
}

// An E4M3 scale (positive, finite) as FP16: exact, 2^-9 .. 448.
__device__ __forceinline__ __half v41_e4m3_h(unsigned b) {
  const unsigned e = (b >> 3) & 15u, m = b & 7u;
  if (e) return __ushort_as_half((unsigned short)(((e + 8u) << 10) | (m << 7)));
  return __float2half_rn((float)m * 0.001953125f);  // 2^-9 subnormals, exact
}

// 8 codes (one 32-bit word, element 2i in the low nibble of byte i) times
// a scale. An E2M1 code placed as FP16 bits (e << 10 | m << 9 | sign << 15)
// is its value times 2^-14 (subnormal m = 1 included), so times 2^14 it is
// exact; the scale multiply is then the one IEEE rounding of the exact
// product, as the FP32 product rounded to FP16. Two codes per half2.
__device__ __forceinline__ uint4 v41_fp4x8(unsigned nib, __half scale) {
  const __half2 s2 = __halves2half2(scale, scale);
  const __half2 k2 = __halves2half2(__float2half(16384.f), __float2half(16384.f));
  uint4 out;
  unsigned* w = reinterpret_cast<unsigned*>(&out);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const unsigned byte = (nib >> (8 * k)) & 0xFFu;
    const unsigned u = (byte & 0xFu) | ((byte & 0xF0u) << 12);
    const unsigned bits = ((u & 0x00070007u) << 9) | ((u & 0x00080008u) << 12);
    const __half2 h =
        __hmul2(__hmul2(*reinterpret_cast<const __half2*>(&bits), k2), s2);
    w[k] = *reinterpret_cast<const unsigned*>(&h);
  }
  return out;
}

__device__ __forceinline__ const half* v41_tag_packed(const void* row) {
  return reinterpret_cast<const half*>(reinterpret_cast<uintptr_t>(row) | 1u);
}

// Window rows in the official V41_FORMAT (VLLM_V41_SWA_FP8, 528 bytes: 512
// E4M3 values, then 16 UE8M0 exponents, one per 32 values). Tag bit 1.
constexpr int V41_W_ROW = 528;

__device__ __forceinline__ const half* v41_tag_fp8(const void* row) {
  return reinterpret_cast<const half*>(reinterpret_cast<uintptr_t>(row) | 2u);
}

// FP16 bits of an E4M3FN code (finite).
__device__ __forceinline__ unsigned v41_e4m3_bits_h(unsigned b) {
  const unsigned e = (b >> 3) & 15u, m = b & 7u, sign = (b & 0x80u) << 8;
  if (e) return sign | ((e + 8u) << 10) | (m << 7);
  return sign | __half_as_ushort(__float2half_rn((float)m * 0.001953125f));
}

// Values c .. c + 7 of an FP8 window row: 8 codes times 2^(E - 127). An
// E4M3 code's low 7 bits shifted left by 7 (sign to bit 15) are FP16 bits of
// its value times 2^-8, subnormals included; times 2^8 and the scale (both
// powers of two) the product is exact while it stays an FP16 normal or
// subnormal, which the window QDQ guarantees (the FP16 cache held it).
__device__ __forceinline__ uint4 v41_fp8x8_dec(uint2 codes, unsigned exponent) {
  uint4 out;
  if (exponent >= 95u && exponent <= 134u) {  // 2^8 * scale in 2^-24 .. 2^15
    const unsigned e = exponent + 8u;  // fold the 2^8
    const unsigned short sb =
        e >= 113u ? (unsigned short)((e - 112u) << 10)
                  : (unsigned short)(1u << (e - 103u));
    const __half2 s2 = __halves2half2(__ushort_as_half(sb), __ushort_as_half(sb));
    unsigned* w = reinterpret_cast<unsigned*>(&out);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const unsigned word = k < 2 ? codes.x : codes.y;
      const unsigned pair = (word >> (16 * (k & 1))) & 0xFFFFu;
      const unsigned u = (pair & 0xFFu) | ((pair & 0xFF00u) << 8);
      const unsigned bits = ((u & 0x007F007Fu) << 7) | ((u & 0x00800080u) << 8);
      const __half2 h = __hmul2(*reinterpret_cast<const __half2*>(&bits), s2);
      w[k] = *reinterpret_cast<const unsigned*>(&h);
    }
    return out;
  }
  const float scale = __uint_as_float(exponent << 23);
  __half* h = reinterpret_cast<__half*>(&out);
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const unsigned word = k < 4 ? codes.x : codes.y;
    const unsigned bits = v41_e4m3_bits_h((word >> (8 * (k & 3))) & 255u);
    h[k] = __float2half_rn(__half2float(__ushort_as_half((unsigned short)bits)) * scale);
  }
  return out;
}

__device__ __forceinline__ uint4 v41_fp8x8(const unsigned char* row, int c) {
  return v41_fp8x8_dec(__ldcg(reinterpret_cast<const uint2*>(row + c)),
                       __ldg(row + 512 + c / 32));
}

// v41_key8 split in two: the loads (issued a tile ahead) and the decode (at
// the shared-memory store), so the gather latency overlaps the math.
struct V41KeyRaw {
  uint4 r;     // FP16 values, or FP4 nibbles (x) / FP8 codes (x, y)
  unsigned s;  // scale byte | tag << 8 (0 FP16, 1 packed FP4, 2 FP8, 3 zero)
};
__device__ __forceinline__ V41KeyRaw v41_key_raw(const half* src, int c) {
  V41KeyRaw k;
  const uintptr_t p = reinterpret_cast<uintptr_t>(src);
  if (!(p & 3u)) {
    k.r = __ldcg(reinterpret_cast<const uint4*>(src + c));
    k.s = 0;
  } else if (p & 2u) {
    const unsigned char* row = reinterpret_cast<const unsigned char*>(p - 2u);
    const uint2 codes = __ldcg(reinterpret_cast<const uint2*>(row + c));
    k.r = make_uint4(codes.x, codes.y, 0, 0);
    k.s = __ldg(row + 512 + c / 32) | (2u << 8);
  } else {
    const unsigned char* row = reinterpret_cast<const unsigned char*>(p - 1u);
    k.r = make_uint4(__ldcg(reinterpret_cast<const unsigned*>(row + c / 2)), 0, 0, 0);
    k.s = __ldg(row + 256 + c / 16) | (1u << 8);
  }
  return k;
}
__device__ __forceinline__ uint4 v41_key_decode(const V41KeyRaw& k) {
  const unsigned tag = k.s >> 8;
  if (tag == 0) return k.r;
  if (tag == 1) return v41_fp4x8(k.r.x, v41_e4m3_h(k.s & 255u));
  if (tag == 2) return v41_fp8x8_dec(make_uint2(k.r.x, k.r.y), k.s & 255u);
  return make_uint4(0, 0, 0, 0);
}

// Values c .. c + 7 of a key row: FP16, a tagged packed compressed row
// (bit 0) or a tagged FP8 window row (bit 1).
__device__ __forceinline__ uint4 v41_key8(const half* src, int c) {
  const uintptr_t p = reinterpret_cast<uintptr_t>(src);
  if (!(p & 3u)) return __ldcg(reinterpret_cast<const uint4*>(src + c));
  if (p & 2u) return v41_fp8x8(reinterpret_cast<const unsigned char*>(p - 2u), c);
  const unsigned char* row = reinterpret_cast<const unsigned char*>(p - 1u);
  const unsigned nib = __ldcg(reinterpret_cast<const unsigned*>(row + c / 2));
  return v41_fp4x8(nib, v41_e4m3_h(__ldg(row + 256 + c / 16)));
}

// E4M3FN code of a value that is exactly one (|v| <= 448).
__device__ __forceinline__ unsigned v41_e4m3_code(float v) {
  const unsigned bits = __float_as_uint(v), sign = (bits >> 24) & 0x80u;
  const float a = fabsf(v);
  if (a < 0x1p-6f) return sign | (unsigned)__float2int_rn(a * 512.f);
  return sign | ((((bits >> 23) & 255u) - 120u) << 3) | ((bits >> 20) & 7u);
}

// UE8M0 exponent of the window QDQ scale for a group's |max| (quant_rope
// _qdq FP8 branch: ceil power of two of max(amax, 1e-4) / 448).
__device__ __forceinline__ unsigned v41_fp8_group_exponent(float amax) {
  const unsigned bits = __float_as_uint(fmaxf(amax, 1e-4f) * (1.0f / 448.0f));
  return ((bits >> 23) & 255u) + ((bits & 0x7FFFFFu) != 0u);
}

// Values c .. c + 7 of a packed index row (FP4_INDEX_KV_FORMAT, 68 bytes:
// 64 B nibbles, 4 UE8M0 exponents, one per 32 values). Scales below FP16's
// normal range take the FP32 product (one rounding of the exact value).
__device__ __forceinline__ uint4 v41_ikey8(const unsigned char* row, int c) {
  const unsigned nib = __ldcg(reinterpret_cast<const unsigned*>(row + c / 2));
  const unsigned exponent = __ldg(row + 64 + c / 32);
  if (exponent >= 113u && exponent <= 142u)  // 2^-14 .. 2^15
    return v41_fp4x8(nib, __ushort_as_half((unsigned short)((exponent - 112u) << 10)));
  const float scale = __uint_as_float(exponent << 23);
  uint4 out;
  __half* h = reinterpret_cast<__half*>(&out);
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const unsigned bits = v41_e2m1_h((nib >> (4 * k)) & 15u);
    h[k] = __float2half_rn(__half2float(__ushort_as_half((unsigned short)bits)) * scale);
  }
  return out;
}

struct RingGather {
  const half* ring;       // [RING, 512]
  const half* values;     // this forward's latents [*, 512]
  const int64_t* starts;  // positions[0] of this forward
  const half* cache;      // compressed pages
  const int* table;       // compressed block table
  const int64_t* positions;
  const bool* valid;
  const int* topk;  // [T, TOPK] (int32)
  int64_t stride;   // cache.stride(0)
  int block, ratio, window, ntopk, ring_rows;
  // Paged mode (attn_paged_dec, VLLM_V41_BATCH): the window keys before this
  // step come from the paged window cache, the step's own from values; every
  // token reads its request row of both block tables.
  bool paged = false;
  const half* wcache = nullptr;  // [pages, wblock, 512]
  const int* wtable = nullptr;   // [reqs, wtable_stride]
  const int* rows = nullptr;     // token -> request row
  const int* bases = nullptr;    // token -> first token of its row
  int64_t wstride = 0;
  int wtable_stride = 0, wblock = 1, table_stride = 0;
  // Compressed pages hold packed 288-byte rows (stride in bytes).
  bool packed = false;
  // Window pages hold FP8 528-byte rows (wstride in bytes).
  bool wpacked = false;
};

// Window row `slot` of `page`: FP16, or tagged FP8.
__device__ __forceinline__ const half* v41_window_src(const RingGather& rg,
                                                      int page, int64_t slot) {
  if (rg.wpacked)
    return v41_tag_fp8(reinterpret_cast<const unsigned char*>(rg.wcache) +
                       page * rg.wstride + slot * V41_W_ROW);
  return rg.wcache + page * rg.wstride + slot * attn::D;
}

// Compressed row `index % per` of `page`: FP16, or tagged packed.
__device__ __forceinline__ const half* v41_compressed_src(const RingGather& rg,
                                                          int page,
                                                          int64_t slot) {
  if (rg.packed)
    return v41_tag_packed(reinterpret_cast<const unsigned char*>(rg.cache) +
                          page * rg.stride + slot * V41_C_ROW);
  return rg.cache + page * rg.stride + slot * attn::D;
}

// rg.cache from a compressed cache: FP16 [pages, rows, 512] or packed uint8
// [pages, rows, 288] (rg.stride is then cache.stride(0) bytes).
static void v41_set_compressed(RingGather& rg, const torch::Tensor& cache,
                               int64_t ntopk) {
  rg.cache = nullptr;
  rg.packed = false;
  if (!ntopk) return;
  rg.packed = cache.scalar_type() == torch::kUInt8;
  TORCH_CHECK(cache.is_cuda() &&
                  (rg.packed ? cache.dim() == 3 && cache.size(2) == V41_C_ROW &&
                                   cache.stride(1) == V41_C_ROW
                             : cache.scalar_type() == torch::kFloat16),
              "v41sk attention: compressed cache must be FP16 rows or packed "
              "288-byte rows");
  rg.cache = reinterpret_cast<const half*>(cache.data_ptr());
}

// rg.wcache from a paged window cache: FP16 [pages, wblock, 512] or FP8
// uint8 [pages, wblock, 528] (VLLM_V41_SWA_FP8; rg.wstride in bytes).
static void v41_set_window(RingGather& rg, const torch::Tensor& wcache,
                           int64_t wblock) {
  rg.wpacked = wcache.scalar_type() == torch::kUInt8;
  TORCH_CHECK(wcache.is_cuda() && wcache.dim() == 3 && wcache.size(1) == wblock &&
                  (rg.wpacked ? wcache.size(2) == V41_W_ROW &&
                                    wcache.stride(1) == V41_W_ROW
                              : wcache.scalar_type() == torch::kFloat16 &&
                                    wcache.size(2) == attn::D &&
                                    wcache.stride(1) == attn::D),
              "v41sk attention: window cache must be FP16 rows or FP8 "
              "528-byte rows");
  rg.wcache = reinterpret_cast<const half*>(wcache.data_ptr());
  rg.wstride = wcache.stride(0);
}

template <bool GATHER>
__global__ void __launch_bounds__(32 * attn::DEC_WARPS)
    attn_dec_split(const half* __restrict__ q, const half* __restrict__ keys,
                   const bool* __restrict__ unused, half* __restrict__ opart,
                   float* __restrict__ mpart, float* __restrict__ lpart,
                   int width, float scale, RingGather rg) {
  using namespace nvcuda;
  using namespace attn;
  extern __shared__ __align__(128) unsigned char dsm[];
  half* qs = reinterpret_cast<half*>(dsm + Q_OFF);
  half* ks = reinterpret_cast<half*>(dsm + K_OFF);
  float* sc = reinterpret_cast<float*>(dsm + S_OFF);
  half* ps = reinterpret_cast<half*>(dsm + P_OFF);
  float* stage = reinterpret_cast<float*>(dsm + T_OFF);
  __shared__ unsigned live_mask;
  const int t = blockIdx.x, chunk = blockIdx.y, nchunk = gridDim.y;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int k0 = chunk * KC;
  const int kn = min(KC, width - k0);
  const half* qt = q + (size_t)t * H * D;
  const half* kt = keys + ((size_t)t * width + k0) * D;
  // Chunks without a live key (short contexts leave most of the 512 top-k
  // slots unused) read no keys: they report an empty softmax (l = 0).
  __shared__ const half* srcs[KC];
  if (warp == 0) {
    bool live;
    if constexpr (GATHER) {
      const int key = k0 + lane;
      const int64_t pos = rg.positions[t];
      live = lane < kn && rg.valid[t];
      const half* src = nullptr;
      const int row = rg.paged ? rg.rows[t] : 0;
      const int vbase = rg.paged ? rg.bases[t] : 0;
      if (key < rg.window) {
        const int64_t at = pos - rg.window + 1 + key;
        live &= at >= 0;
        const int64_t start = rg.paged ? rg.positions[vbase] : rg.starts[0];
        if (at >= start) {
          src = rg.values + (vbase + at - start) * D;
        } else if (rg.paged) {
          const int page =
              live ? rg.wtable[(size_t)row * rg.wtable_stride + at / rg.wblock]
                   : -1;
          live &= page >= 0;
          if (live) src = v41_window_src(rg, page, at % rg.wblock);
        } else {
          src =
              rg.ring + ((at % rg.ring_rows + rg.ring_rows) % rg.ring_rows) * D;
        }
      } else if (live) {
        const int64_t index = rg.topk[(size_t)t * rg.ntopk + key - rg.window];
        const int per = rg.block / rg.ratio;
        live &= index >= 0 && index < (pos + 1) / rg.ratio;
        const int page =
            live ? rg.table[(size_t)row * rg.table_stride + index / per] : -1;
        live &= page >= 0;
        if (live) src = v41_compressed_src(rg, page, index % per);
      }
      srcs[lane] = src;
    } else {
      live = lane < kn && !unused[(size_t)t * width + k0 + lane];
    }
    const unsigned m = __ballot_sync(0xffffffffu, live);
    if (lane == 0) live_mask = m;
  }
  __syncthreads();
  const unsigned lm = live_mask;
  if (lm == 0) {
    if (threadIdx.x < H) {
      mpart[((size_t)t * nchunk + chunk) * H + threadIdx.x] = -1e30f;
      lpart[((size_t)t * nchunk + chunk) * H + threadIdx.x] = 0.f;
    }
    return;
  }
  // stage q (16 x 512) and the live key rows (unused rows are zero)
  for (int i = threadIdx.x; i < H * D / 8; i += 32 * DEC_WARPS) {
    const int h = i / (D / 8), c = (i % (D / 8)) * 8;
    *reinterpret_cast<uint4*>(qs + h * QLD + c) =
        *reinterpret_cast<const uint4*>(qt + h * D + c);
  }
  {
    constexpr int PER =
        KC * D / 8 / (32 * DEC_WARPS);  // all in flight, then stored
    uint4 v[PER];
#pragma unroll
    for (int u = 0; u < PER; ++u) {
      const int i = threadIdx.x + u * (32 * DEC_WARPS);
      const int j = i / (D / 8), c = (i % (D / 8)) * 8;
      const half* row = GATHER ? srcs[j] : kt + (size_t)j * D;
      v[u] = (lm >> j) & 1u ? v41_key8(row, c)
                            : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int u = 0; u < PER; ++u) {
      const int i = threadIdx.x + u * (32 * DEC_WARPS);
      const int j = i / (D / 8), c = (i % (D / 8)) * 8;
      *reinterpret_cast<uint4*>(ks + j * KLD + c) = v[u];
    }
  }
  __syncthreads();
  // QK retains the four-warp reduction order. The other warps join for PV.
  if (warp < 4) {
    const int tile = warp & 1, half_d = warp >> 1;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.f);
#pragma unroll 1
    for (int d = half_d * (D / 2); d < (half_d + 1) * (D / 2); d += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
      wmma::load_matrix_sync(a, qs + d, QLD);
      wmma::load_matrix_sync(b, ks + tile * 16 * KLD + d, KLD);
      wmma::mma_sync(acc, a, b, acc);
    }
    wmma::store_matrix_sync(sc + half_d * H * SLD + tile * 16, acc, SLD,
                            wmma::mem_row_major);
  }
  __syncthreads();
  // Masked partial softmax per head, lanes map to keys.
  for (int h = warp; h < H; h += DEC_WARPS) {
    const bool live = (lm >> lane) & 1u;
    const float raw = sc[h * SLD + lane] + sc[H * SLD + h * SLD + lane];
    const float sv = live ? raw * scale : -INFINITY;
    float m = sv;
#pragma unroll
    for (int o = 16; o; o >>= 1)
      m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    const float e = live ? __expf(sv - m) : 0.f;
    float l = e;
#pragma unroll
    for (int o = 16; o; o >>= 1) l += __shfl_xor_sync(0xffffffffu, l, o);
    ps[h * PLD + lane] = __float2half(e);
    if (lane == 0) {
      mpart[((size_t)t * nchunk + chunk) * H + h] = m;
      lpart[((size_t)t * nchunk + chunk) * H + h] = l;
    }
  }
  __syncthreads();
  // PV distributes the 32 output tiles across all decode warps.
  half* o_tok = opart + ((size_t)t * nchunk + chunk) * H * D;
  float* st = stage + warp * 256;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> pa0, pa1;
  wmma::load_matrix_sync(pa0, ps, PLD);
  wmma::load_matrix_sync(pa1, ps + 16, PLD);
#pragma unroll 1
  for (int tile = warp; tile < D / 16; tile += DEC_WARPS) {
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.f);
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b0, b1;
    wmma::load_matrix_sync(b0, ks + tile * 16, KLD);
    wmma::load_matrix_sync(b1, ks + 16 * KLD + tile * 16, KLD);
    wmma::mma_sync(acc, pa0, b0, acc);
    wmma::mma_sync(acc, pa1, b1, acc);
    // FP32 tile -> FP16 partial (unnormalised sum of <= 32 keys)
    wmma::store_matrix_sync(st, acc, 16, wmma::mem_row_major);
    __syncwarp();
#pragma unroll
    for (int i = lane; i < 128; i += 32) {
      const int row = i >> 3, col = (i & 7) * 2;
      *reinterpret_cast<half2*>(o_tok + row * D + tile * 16 + col) =
          __floats2half2_rn(st[row * 16 + col], st[row * 16 + col + 1]);
    }
    __syncwarp();
  }
}

// attn_dec_split with TILES 32-key tiles per CTA (V41SK_ATTN_TILES / large
// batches): Q is staged once and an online softmax (running max / sum per head)
// carries the PV accumulator across tiles, so a token writes nchunk / TILES
// partials instead of nchunk (each of the 20 split CTAs of a 640-key row
// re-read the 16 KB query and wrote a 16 KB partial: 2.5x the key bytes).
// Partials keep attn_dec_combine's format (unnormalised FP16 sums with their
// max and sum); summation order differs from attn_dec_split (not bitwise).
template <bool GATHER>
__global__ void __launch_bounds__(32 * attn::DEC_WARPS)
    attn_dec_multi(const half* __restrict__ q, half* __restrict__ opart,
                   float* __restrict__ mpart, float* __restrict__ lpart,
                   int width, float scale, int tiles, RingGather rg) {
  using namespace nvcuda;
  using namespace attn;
  extern __shared__ __align__(128) unsigned char dsm[];
  half* qs = reinterpret_cast<half*>(dsm + Q_OFF);
  half* ks = reinterpret_cast<half*>(dsm + K_OFF);
  float* sc = reinterpret_cast<float*>(dsm + S_OFF);
  half* ps = reinterpret_cast<half*>(dsm + P_OFF);
  float* stage = reinterpret_cast<float*>(dsm + T_OFF);
  __shared__ unsigned live_masks[2];
  __shared__ const half* srcs[2][KC];
  __shared__ float m_run[H], l_run[H], alpha[H];
  __shared__ float rows_of[16 * 16];
  const int t = blockIdx.x, part = blockIdx.y, nparts = gridDim.y;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const half* qt = q + (size_t)t * H * D;
  for (int i = threadIdx.x; i < H * D / 8; i += 32 * DEC_WARPS) {
    const int h = i / (D / 8), c = (i % (D / 8)) * 8;
    *reinterpret_cast<uint4*>(qs + h * QLD + c) =
        *reinterpret_cast<const uint4*>(qt + h * D + c);
  }
  if (threadIdx.x < H) {
    m_run[threadIdx.x] = -1e30f;
    l_run[threadIdx.x] = 0.f;
  }
  if (threadIdx.x < 256) rows_of[threadIdx.x] = (float)(threadIdx.x >> 4);
  __syncthreads();
  // Each accumulator element's head row (the fragment layout is opaque).
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> row_frag, acc;
  wmma::load_matrix_sync(row_frag, rows_of, 16, wmma::mem_row_major);
  wmma::fill_fragment(acc, 0.f);
  const int tile_o = warp;  // DEC_WARPS == D / 16: one output tile per warp
  // Warp 0 resolves tile u's 32 key addresses into buffer u & 1.
  auto resolve = [&](int u) {
    const int k0 = (part * tiles + u) * KC;
    const int kn = min(KC, width - k0);
    const int key = k0 + lane;
    const int64_t pos = rg.positions[t];
    bool live = k0 < width && lane < kn && rg.valid[t];
    const half* src = nullptr;
    const int row = rg.paged ? rg.rows[t] : 0;
    const int vbase = rg.paged ? rg.bases[t] : 0;
    if (live && key < rg.window) {
      const int64_t at = pos - rg.window + 1 + key;
      live &= at >= 0;
      const int64_t start = rg.paged ? rg.positions[vbase] : rg.starts[0];
      if (at >= start) {
        src = rg.values + (vbase + at - start) * D;
      } else if (rg.paged) {
        const int page =
            live ? rg.wtable[(size_t)row * rg.wtable_stride + at / rg.wblock]
                 : -1;
        live &= page >= 0;
        if (live) src = v41_window_src(rg, page, at % rg.wblock);
      } else {
        src = rg.ring + ((at % rg.ring_rows + rg.ring_rows) % rg.ring_rows) * D;
      }
    } else if (live) {
      const int64_t index = rg.topk[(size_t)t * rg.ntopk + key - rg.window];
      const int per = rg.block / rg.ratio;
      live &= index >= 0 && index < (pos + 1) / rg.ratio;
      const int page =
          live ? rg.table[(size_t)row * rg.table_stride + index / per] : -1;
      live &= page >= 0;
      if (live) src = v41_compressed_src(rg, page, index % per);
    }
    srcs[u & 1][lane] = src;
    const unsigned m = __ballot_sync(0xffffffffu, live);
    if (lane == 0) live_masks[u & 1] = m;
  };
  constexpr int PER = KC * D / 8 / (32 * DEC_WARPS);
  uint4 v[PER];
  // Tile u's keys into registers (issued a tile ahead, stored after use).
  auto fetch = [&](int u) {
    const unsigned lm = live_masks[u & 1];
#pragma unroll
    for (int w = 0; w < PER; ++w) {
      const int i = threadIdx.x + w * (32 * DEC_WARPS);
      const int j = i / (D / 8), c = (i % (D / 8)) * 8;
      v[w] = (lm >> j) & 1u
                 ? v41_key8(srcs[u & 1][j], c)
                 : make_uint4(0, 0, 0, 0);
    }
  };
  const int last = min(tiles, (width - part * tiles * KC + KC - 1) / KC);
  if (warp == 0 && last > 0) resolve(0);
  __syncthreads();
  if (last > 0) fetch(0);
  for (int u = 0; u < last; ++u) {
    const unsigned lm = live_masks[u & 1];
#pragma unroll
    for (int w = 0; w < PER; ++w) {
      const int i = threadIdx.x + w * (32 * DEC_WARPS);
      const int j = i / (D / 8), c = (i % (D / 8)) * 8;
      *reinterpret_cast<uint4*>(ks + j * KLD + c) = v[w];
    }
    if (warp == 0 && u + 1 < last) resolve(u + 1);
    __syncthreads();
    if (u + 1 < last) fetch(u + 1);  // in flight during this tile's math
    if (lm == 0) {
      __syncthreads();
      continue;  // uniform: no live key in this tile
    }
    if (warp < 4) {
      const int tile = warp & 1, half_d = warp >> 1;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> s;
      wmma::fill_fragment(s, 0.f);
#pragma unroll 1
      for (int d = half_d * (D / 2); d < (half_d + 1) * (D / 2); d += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
        wmma::load_matrix_sync(a, qs + d, QLD);
        wmma::load_matrix_sync(b, ks + tile * 16 * KLD + d, KLD);
        wmma::mma_sync(s, a, b, s);
      }
      wmma::store_matrix_sync(sc + half_d * H * SLD + tile * 16, s, SLD,
                              wmma::mem_row_major);
    }
    __syncthreads();
    // Online softmax per head: rescale factor for what came before.
    for (int h = warp; h < H; h += DEC_WARPS) {
      const bool live = (lm >> lane) & 1u;
      const float raw = sc[h * SLD + lane] + sc[H * SLD + h * SLD + lane];
      const float sv = live ? raw * scale : -INFINITY;
      float m = sv;
#pragma unroll
      for (int o = 16; o; o >>= 1)
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
      const float m_new = fmaxf(m_run[h], m);
      const float e = live ? __expf(sv - m_new) : 0.f;
      float l = e;
#pragma unroll
      for (int o = 16; o; o >>= 1) l += __shfl_xor_sync(0xffffffffu, l, o);
      ps[h * PLD + lane] = __float2half(e);
      if (lane == 0) {
        const float a = __expf(m_run[h] - m_new);
        alpha[h] = a;
        l_run[h] = l_run[h] * a + l;
        m_run[h] = m_new;
      }
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < acc.num_elements; ++i)
      acc.x[i] *= alpha[(int)row_frag.x[i]];
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> pa0, pa1;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b0, b1;
    wmma::load_matrix_sync(pa0, ps, PLD);
    wmma::load_matrix_sync(pa1, ps + 16, PLD);
    wmma::load_matrix_sync(b0, ks + tile_o * 16, KLD);
    wmma::load_matrix_sync(b1, ks + 16 * KLD + tile_o * 16, KLD);
    wmma::mma_sync(acc, pa0, b0, acc);
    wmma::mma_sync(acc, pa1, b1, acc);
    __syncthreads();  // ks / ps / alpha are rewritten by the next tile
  }
  if (threadIdx.x < H) {
    mpart[((size_t)t * nparts + part) * H + threadIdx.x] = m_run[threadIdx.x];
    lpart[((size_t)t * nparts + part) * H + threadIdx.x] = l_run[threadIdx.x];
  }
  half* o_tok = opart + ((size_t)t * nparts + part) * H * D;
  float* st = stage + warp * 256;
  wmma::store_matrix_sync(st, acc, 16, wmma::mem_row_major);
  __syncwarp();
#pragma unroll
  for (int i = lane; i < 128; i += 32) {
    const int row = i >> 3, col = (i & 7) * 2;
    *reinterpret_cast<half2*>(o_tok + row * D + tile_o * 16 + col) =
        __floats2half2_rn(st[row * 16 + col], st[row * 16 + col + 1]);
  }
}

// ---- 64-head sparse prefill attention (VLLM_V41_ATTN_H64) --------------------
// FlashMLA's sparse prefill layout on sm70: one CTA per query token with ALL
// 64 heads (the TP group exchanged token blocks for head blocks), so the
// token's window + top-k latent rows are gathered and decoded once for 64
// heads (WMMA M = 64 instead of 16). Keys in tiles of 16, double-buffered
// through registers; 8 warps: QK^T as 4 head blocks x 2 halves of D (Q held
// in registers), online softmax over 64 heads, PV with each warp owning 64 of
// the 512 output columns for all heads. The attention sink joins the
// normalizer as in attn_dec_combine; the output is normalized in place.
// Phase cycles of attn_sparse_h64 (V41SK_ATTN64_ABLATE & 8: thread 0 of
// every CTA adds clock64 deltas: store+resolve, QK, softmax, PV, tail).
__device__ unsigned long long g_attn64_prof[8];
namespace sp64 {
constexpr int HQ = 64, D = 512, KT = 16, QLD = D + 8, KLD = D + 8, SLD = KT + 4,
              PLD = KT + 8;
constexpr size_t K_OFF = sizeof(half) * HQ * QLD;                 // 66,560
constexpr size_t S_OFF = K_OFF + sizeof(half) * KT * KLD;         // 83,200
constexpr size_t P_OFF = S_OFF + sizeof(float) * 2 * HQ * SLD;    // 93,440
constexpr size_t M_OFF = P_OFF + sizeof(half) * HQ * PLD;         // 96,512
constexpr size_t T_OFF = M_OFF + sizeof(float) * 3 * HQ;          // 97,280
constexpr size_t SMEM = 97664;  // <= 98,304 (96 KiB max on sm_70), 0 static
}  // namespace sp64

// WARPS 8 or 16: each warp owns D / WARPS output columns of all 64 heads;
// more warps keep more key gathers in flight (one CTA per SM either way).
template <int WARPS>
__global__ void __launch_bounds__(32 * WARPS, 1)
    attn_sparse_h64(half* __restrict__ out, const half* __restrict__ q,
                    int token0, int width, float scale,
                    const float* __restrict__ sink, RingGather rg, int ablate) {
  using namespace nvcuda;
  using namespace sp64;
  extern __shared__ __align__(128) unsigned char dsm[];
  half* qs = reinterpret_cast<half*>(dsm);
  half* ks = reinterpret_cast<half*>(dsm + K_OFF);
  float* sc = reinterpret_cast<float*>(dsm + S_OFF);
  half* ps = reinterpret_cast<half*>(dsm + P_OFF);
  float* m_run = reinterpret_cast<float*>(dsm + M_OFF);
  float* l_run = m_run + HQ;
  float* alpha = l_run + HQ;
  unsigned* live_masks = reinterpret_cast<unsigned*>(dsm + T_OFF);
  const half** srcs_flat = reinterpret_cast<const half**>(dsm + T_OFF + 16);
  const int tl = blockIdx.x, t = token0 + tl;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const half* qt = q + (size_t)tl * HQ * D;
  for (int i = threadIdx.x; i < HQ * D / 8; i += blockDim.x) {
    const int h = i / (D / 8), c = (i % (D / 8)) * 8;
    *reinterpret_cast<uint4*>(qs + h * QLD + c) =
        __ldg(reinterpret_cast<const uint4*>(qt + h * D + c));
  }
  if (threadIdx.x < HQ) {
    m_run[threadIdx.x] = -1e30f;
    l_run[threadIdx.x] = 0.f;
  }
  constexpr int DPW = D / WARPS, NJ = DPW / 16, HPW = HQ / WARPS,
                LPH = 32 / HPW, KPL = KT / LPH;
  if (threadIdx.x < 256) sc[threadIdx.x] = (float)(threadIdx.x >> 4);
  __syncthreads();
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4][NJ];
  uint32_t row_pack = 0;
  {
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> rf;
    wmma::load_matrix_sync(rf, sc, 16, wmma::mem_row_major);
#pragma unroll
    for (int x = 0; x < 8; ++x)
      row_pack |= (uint32_t)((int)rf.x[x] & 15) << (x * 4);
  }
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < NJ; ++j) wmma::fill_fragment(acc[i][j], 0.f);
  constexpr int Q_REG = (WARPS == 8) ? 2 : 0;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>
      a_reg[Q_REG > 0 ? Q_REG : 1][2];
  if constexpr (Q_REG > 0) {
    const int m_half = warp & 1, d_part = warp >> 1;
    const int k_base = d_part * (D / 64);
#pragma unroll
    for (int k = 0; k < Q_REG; ++k) {
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        wmma::load_matrix_sync(
            a_reg[k][m],
            qs + (m_half * 32 + m * 16) * QLD + (k_base + k) * 16, QLD);
      }
    }
  }
  // Warp 0 resolves tile u's 16 key addresses (attn_dec_multi's rules).
  auto resolve = [&](int u) {
    const int key = u * KT + lane;
    const int64_t pos = rg.positions[t];
    bool live = lane < KT && key < width && rg.valid[t];
    const half* src = nullptr;
    const int row = rg.rows[t];
    const int vbase = rg.bases[t];
    if (live && key < rg.window) {
      const int64_t at = pos - rg.window + 1 + key;
      live &= at >= 0;
      const int64_t start = rg.positions[vbase];
      if (at >= start) {
        src = rg.values + (vbase + at - start) * D;
      } else {
        const int page =
            live ? rg.wtable[(size_t)row * rg.wtable_stride + at / rg.wblock]
                 : -1;
        live &= page >= 0;
        if (live) src = v41_window_src(rg, page, at % rg.wblock);
      }
    } else if (live) {
      const int64_t index = rg.topk[(size_t)t * rg.ntopk + key - rg.window];
      const int per = rg.block / rg.ratio;
      live &= index >= 0 && index < (pos + 1) / rg.ratio;
      const int page =
          live ? rg.table[(size_t)row * rg.table_stride + index / per] : -1;
      live &= page >= 0;
      if (live) src = v41_compressed_src(rg, page, index % per);
    }
    if (lane < KT) srcs_flat[(u & 1) * KT + lane] = src;
    const unsigned m = __ballot_sync(0xffffffffu, live);
    if (lane == 0) live_masks[u & 1] = m;
  };
  constexpr int PER = KT * D / 8 / (32 * WARPS);  // uint4 per thread
  uint4 v[PER];
  auto fetch = [&](int u) {
    const unsigned lm = live_masks[u & 1];
#pragma unroll
    for (int w = 0; w < PER; ++w) {
      const int i = threadIdx.x + w * (32 * WARPS);
      const int j = i / (D / 8), c = (i % (D / 8)) * 8;
      v[w] = ((lm >> j) & 1u) && !(ablate & 1)
                 ? v41_key8(srcs_flat[(u & 1) * KT + j], c)
                 : make_uint4(0, 0, 0, 0);
    }
  };
  const int tiles = (width + KT - 1) / KT;
  if (warp == 0 && tiles > 0) resolve(0);
  __syncthreads();
  if (tiles > 0) fetch(0);
  const bool prof = (ablate & 8) && threadIdx.x == 0;
  long long c0 = prof ? clock64() : 0, c1;
  auto lap = [&](int slot) {
    if (prof) {
      c1 = clock64();
      atomicAdd(&g_attn64_prof[slot], (unsigned long long)(c1 - c0));
      c0 = c1;
    }
  };
  for (int u = 0; u < tiles; ++u) {
    const unsigned lm = live_masks[u & 1];
#pragma unroll
    for (int w = 0; w < PER; ++w) {
      const int i = threadIdx.x + w * (32 * WARPS);
      const int j = i / (D / 8), c = (i % (D / 8)) * 8;
      *reinterpret_cast<uint4*>(ks + j * KLD + c) = v[w];
    }
    if (warp == 0 && u + 1 < tiles) resolve(u + 1);
    __syncthreads();
    lap(0);
    if (u + 1 < tiles) fetch(u + 1);  // in flight during this tile's math
    if (lm == 0) {
      __syncthreads();  // the next store rewrites ks
      continue;
    }
    // QK^T: for WARPS==8, 4-way split-D x 2-way M (each warp reuses K across
    // 32 heads, cutting K shared-memory loads in half).
    if constexpr (WARPS == 8) {
      if (!(ablate & 2)) {
        const int m_half = warp & 1, d_part = warp >> 1;
        const int k_base = d_part * (D / 64);
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> s[2];
        wmma::fill_fragment(s[0], 0.f);
        wmma::fill_fragment(s[1], 0.f);
#pragma unroll
        for (int k = 0; k < Q_REG; ++k) {
          wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
          wmma::load_matrix_sync(b, ks + (k_base + k) * 16, KLD);
          wmma::mma_sync(s[0], a_reg[k][0], b, s[0]);
          wmma::mma_sync(s[1], a_reg[k][1], b, s[1]);
        }
#pragma unroll
        for (int k = Q_REG; k < D / 64; ++k) {
          wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
          wmma::load_matrix_sync(b, ks + (k_base + k) * 16, KLD);
#pragma unroll
          for (int m = 0; m < 2; ++m) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
            wmma::load_matrix_sync(
                a, qs + (m_half * 32 + m * 16) * QLD + (k_base + k) * 16, QLD);
            wmma::mma_sync(s[m], a, b, s[m]);
          }
        }
        if (d_part < 2) {
#pragma unroll
          for (int m = 0; m < 2; ++m) {
            wmma::store_matrix_sync(
                sc + (d_part * HQ + m_half * 32 + m * 16) * SLD, s[m], SLD,
                wmma::mem_row_major);
          }
        }
        __syncthreads();
        if (d_part >= 2) {
#pragma unroll
          for (int m = 0; m < 2; ++m) {
            float* dst = sc + ((d_part - 2) * HQ + m_half * 32 + m * 16) * SLD;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> p;
            wmma::load_matrix_sync(p, dst, SLD, wmma::mem_row_major);
#pragma unroll
            for (int x = 0; x < 8; ++x) s[m].x[x] += p.x[x];
            wmma::store_matrix_sync(dst, s[m], SLD, wmma::mem_row_major);
          }
        }
      }
    } else if (warp < 8 && !(ablate & 2)) {
      const int m_tile = warp >> 1, d_half = warp & 1;
      const int k_base = d_half * (D / 32);
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> s;
      wmma::fill_fragment(s, 0.f);
#pragma unroll 8
      for (int k = 0; k < D / 32; ++k) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
        wmma::load_matrix_sync(
            a, qs + m_tile * 16 * QLD + (k_base + k) * 16, QLD);
        wmma::load_matrix_sync(b, ks + (k_base + k) * 16, KLD);
        wmma::mma_sync(s, a, b, s);
      }
      wmma::store_matrix_sync(sc + (d_half * HQ + m_tile * 16) * SLD, s, SLD,
                              wmma::mem_row_major);
    }
    __syncthreads();
    lap(1);
    // Online softmax: LPH lanes per head, KPL keys per lane.
    {
      const int h = warp * HPW + lane / LPH, k0 = (lane % LPH) * KPL;
      float e[KPL], m = -INFINITY;
#pragma unroll
      for (int i = 0; i < KPL; ++i) {
        const int k = k0 + i;
        const bool live = (lm >> k) & 1u;
        e[i] = live ? (sc[h * SLD + k] + sc[(HQ + h) * SLD + k]) * scale
                    : -INFINITY;
        m = fmaxf(m, e[i]);
      }
#pragma unroll
      for (int o = 1; o < LPH; o <<= 1)
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
      const float m_new = fmaxf(m_run[h], m);
      float l = 0.f;
#pragma unroll
      for (int i = 0; i < KPL; ++i) {
        const int k = k0 + i;
        const float x = ((lm >> k) & 1u) ? __expf(e[i] - m_new) : 0.f;
        l += x;
        ps[h * PLD + k] = __float2half(x);
      }
#pragma unroll
      for (int o = 1; o < LPH; o <<= 1) l += __shfl_xor_sync(0xffffffffu, l, o);
      __syncwarp();
      if (lane % LPH == 0) {
        const float a = __expf(m_run[h] - m_new);
        alpha[h] = a;
        l_run[h] = l_run[h] * a + l;
        m_run[h] = m_new;
      }
    }
    __syncthreads();
    lap(2);
    // PV: this warp's output columns for all 64 heads (hoist V loads across i).
    if (!(ablate & 4)) {
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> vb[NJ];
#pragma unroll
      for (int j = 0; j < NJ; ++j) {
        wmma::load_matrix_sync(vb[j], ks + warp * DPW + j * 16, KLD);
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float ar_scale[8];
        bool any_scale = false;
#pragma unroll
        for (int x = 0; x < 8; ++x) {
          ar_scale[x] = alpha[i * 16 + ((row_pack >> (x * 4)) & 15)];
          any_scale |= (ar_scale[x] != 1.f);
        }
        if (__any_sync(0xffffffffu, any_scale)) {
#pragma unroll
          for (int j = 0; j < NJ; ++j)
#pragma unroll
            for (int x = 0; x < 8; ++x)
              acc[i][j].x[x] *= ar_scale[x];
        }
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> pa;
        wmma::load_matrix_sync(pa, ps + i * 16 * PLD, PLD);
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
          wmma::mma_sync(acc[i][j], pa, vb[j], acc[i][j]);
        }
      }
    }
    __syncthreads();  // ks / ps / alpha are rewritten by the next tile
    lap(3);
  }
  // Normalize with the sink (attn_dec_combine's single-chunk weights).
  if (threadIdx.x < HQ) {
    const int h = threadIdx.x;
    const float l = l_run[h];
    const float M = l > 0.f ? fmaxf(sink[h], m_run[h]) : sink[h];
    const float w = l > 0.f ? __expf(m_run[h] - M) : 0.f;
    alpha[h] = w / (l * w + __expf(sink[h] - M));
  }
  __syncthreads();
  float* st = reinterpret_cast<float*>(dsm) + warp * 256;  // reuse Q
  half* ot = out + (size_t)tl * HQ * D;
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
#pragma unroll
      for (int x = 0; x < 8; ++x)
        acc[i][j].x[x] *= alpha[i * 16 + ((row_pack >> (x * 4)) & 15)];
      wmma::store_matrix_sync(st, acc[i][j], 16, wmma::mem_row_major);
      __syncwarp();
#pragma unroll
      for (int e = lane; e < 128; e += 32) {
        const int r = e >> 3, c = (e & 7) * 2;
        *reinterpret_cast<half2*>(ot + (i * 16 + r) * D + warp * DPW + j * 16 + c) =
            __floats2half2_rn(st[r * 16 + c], st[r * 16 + c + 1]);
      }
      __syncwarp();
    }
}

namespace sp32 {
constexpr int HQ = 32, HQ_ALL = 64, D = 512, KT = 16, Q_REG = 2;
constexpr int D_REM = D - 4 * Q_REG * 16;  // 384 dims staged in qs
constexpr int QLD = D_REM + 8, KLD = D + 8, SLD = KT + 4, PLD = KT + 8;
constexpr size_t K_OFF = sizeof(half) * HQ * QLD;               // 25,088
constexpr size_t S_OFF = K_OFF + sizeof(half) * KT * KLD;       // 41,728
constexpr size_t P_OFF = S_OFF + sizeof(float) * 2 * HQ * SLD;  // 46,848
constexpr size_t M_OFF = P_OFF + sizeof(half) * HQ * PLD;       // 48,384
constexpr size_t T_OFF = M_OFF + sizeof(float) * 3 * HQ;        // 48,768
constexpr size_t SMEM = 49152;  // 48 KiB exact -> 2 CTAs/SM on Volta (96 KiB)
}  // namespace sp32

// 2 CTAs per token (HQ = 32 heads per CTA, grid = dim3(2, Tq)): halves PV
// accumulator registers (64 vs 128) and shared memory (48 KiB vs 96 KiB) to
// run 2 CTAs (16 warps = 512 threads) per Volta SM while sharing key pages in L2.
__global__ void __launch_bounds__(256, 2)
    attn_sparse_h32(half* __restrict__ out, const half* __restrict__ q,
                    int token0, int width, float scale,
                    const float* __restrict__ sink_all, RingGather rg) {
  using namespace nvcuda;
  using namespace sp32;
  constexpr int WARPS = 8;
  extern __shared__ __align__(128) unsigned char dsm[];
  half* qs = reinterpret_cast<half*>(dsm);
  half* ks = reinterpret_cast<half*>(dsm + K_OFF);
  float* sc = reinterpret_cast<float*>(dsm + S_OFF);
  half* ps = reinterpret_cast<half*>(dsm + P_OFF);
  float* m_run = reinterpret_cast<float*>(dsm + M_OFF);
  float* l_run = m_run + HQ;
  float* alpha = l_run + HQ;
  unsigned* live_masks = reinterpret_cast<unsigned*>(dsm + T_OFF);
  const half** srcs_flat = reinterpret_cast<const half**>(dsm + T_OFF + 16);

  const int h_cta = blockIdx.x, tl = blockIdx.y, t = token0 + tl;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const half* qt = q + ((size_t)tl * HQ_ALL + h_cta * HQ) * D;
  const float* sink = sink_all + h_cta * HQ;

  // Stage full [32, 512] Q slice in first 33,280 B of dsm, preload Q_REG=2
  // WMMA fragments per warp into registers, and compact remaining 384 dims.
  half* q_tmp = reinterpret_cast<half*>(dsm);
  for (int i = threadIdx.x; i < HQ * (D / 8); i += 256) {
    const int h = i / (D / 8), c = (i % (D / 8)) * 8;
    *reinterpret_cast<uint4*>(q_tmp + h * KLD + c) =
        __ldg(reinterpret_cast<const uint4*>(qt + h * D + c));
  }
  if (threadIdx.x < HQ) {
    m_run[threadIdx.x] = -1e30f;
    l_run[threadIdx.x] = 0.f;
  }
  __syncthreads();

  const int m_tile = warp & 1, d_part = warp >> 1;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>
      a_reg[Q_REG];
#pragma unroll
  for (int k = 0; k < Q_REG; ++k) {
    wmma::load_matrix_sync(
        a_reg[k], q_tmp + m_tile * 16 * KLD + (d_part * 8 + k) * 16, KLD);
  }
  uint4 q_rem[6];
#pragma unroll
  for (int r = 0; r < 6; ++r) {
    const int idx = threadIdx.x + r * 256;
    const int h = idx / 48, col384 = (idx % 48) * 8;
    const int src_col = (col384 / 96) * 128 + Q_REG * 16 + (col384 % 96);
    q_rem[r] = *reinterpret_cast<const uint4*>(q_tmp + h * KLD + src_col);
  }
  __syncthreads();

#pragma unroll
  for (int r = 0; r < 6; ++r) {
    const int idx = threadIdx.x + r * 256;
    const int h = idx / 48, col384 = (idx % 48) * 8;
    *reinterpret_cast<uint4*>(qs + h * QLD + col384) = q_rem[r];
  }
  sc[threadIdx.x] = (float)(threadIdx.x >> 4);
  __syncthreads();

  constexpr int DPW = D / WARPS, NJ = DPW / 16, HPW = HQ / WARPS,
                LPH = 32 / HPW, KPL = KT / LPH;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][NJ];
  uint32_t row_pack = 0;
  {
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> rf;
    wmma::load_matrix_sync(rf, sc, 16, wmma::mem_row_major);
#pragma unroll
    for (int x = 0; x < 8; ++x)
      row_pack |= (uint32_t)((int)rf.x[x] & 15) << (x * 4);
  }
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < NJ; ++j) wmma::fill_fragment(acc[i][j], 0.f);

  auto resolve = [&](int u) {
    const int key = u * KT + lane;
    const int64_t pos = rg.positions[t];
    bool live = lane < KT && key < width && rg.valid[t];
    const half* src = nullptr;
    const int row = rg.rows[t];
    const int vbase = rg.bases[t];
    if (live && key < rg.window) {
      const int64_t at = pos - rg.window + 1 + key;
      live &= at >= 0;
      const int64_t start = rg.positions[vbase];
      if (at >= start) {
        src = rg.values + (vbase + at - start) * D;
      } else {
        const int page =
            live ? rg.wtable[(size_t)row * rg.wtable_stride + at / rg.wblock]
                 : -1;
        live &= page >= 0;
        if (live) src = v41_window_src(rg, page, at % rg.wblock);
      }
    } else if (live) {
      const int64_t index = rg.topk[(size_t)t * rg.ntopk + key - rg.window];
      const int per = rg.block / rg.ratio;
      live &= index >= 0 && index < (pos + 1) / rg.ratio;
      const int page =
          live ? rg.table[(size_t)row * rg.table_stride + index / per] : -1;
      live &= page >= 0;
      if (live) src = v41_compressed_src(rg, page, index % per);
    }
    if (lane < KT) srcs_flat[(u & 1) * KT + lane] = src;
    const unsigned m = __ballot_sync(0xffffffffu, live);
    if (lane == 0) live_masks[u & 1] = m;
  };
  constexpr int PER = KT * D / 8 / 256;
  uint4 v[PER];
  auto fetch = [&](int u) {
    const unsigned lm = live_masks[u & 1];
#pragma unroll
    for (int w = 0; w < PER; ++w) {
      const int i = threadIdx.x + w * 256;
      const int j = i / (D / 8), c = (i % (D / 8)) * 8;
      v[w] = ((lm >> j) & 1u) ? v41_key8(srcs_flat[(u & 1) * KT + j], c)
                              : make_uint4(0, 0, 0, 0);
    }
  };

  const int tiles = (width + KT - 1) / KT;
  if (warp == 0 && tiles > 0) resolve(0);
  __syncthreads();
  if (tiles > 0) fetch(0);

  for (int u = 0; u < tiles; ++u) {
    const unsigned lm = live_masks[u & 1];
#pragma unroll
    for (int w = 0; w < PER; ++w) {
      const int i = threadIdx.x + w * 256;
      const int j = i / (D / 8), c = (i % (D / 8)) * 8;
      *reinterpret_cast<uint4*>(ks + j * KLD + c) = v[w];
    }
    if (warp == 0 && u + 1 < tiles) resolve(u + 1);
    __syncthreads();
    if (u + 1 < tiles) fetch(u + 1);
    if (lm == 0) {
      __syncthreads();
      continue;
    }

    // QK^T: 4-way split-D x 2-way M (16 heads per warp).
    {
      const int k_base = d_part * 8;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> s;
      wmma::fill_fragment(s, 0.f);
#pragma unroll
      for (int k = 0; k < Q_REG; ++k) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
        wmma::load_matrix_sync(b, ks + (k_base + k) * 16, KLD);
        wmma::mma_sync(s, a_reg[k], b, s);
      }
#pragma unroll
      for (int k = Q_REG; k < 8; ++k) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
        wmma::load_matrix_sync(
            a, qs + m_tile * 16 * QLD + (d_part * 6 + (k - Q_REG)) * 16, QLD);
        wmma::load_matrix_sync(b, ks + (k_base + k) * 16, KLD);
        wmma::mma_sync(s, a, b, s);
      }
      if (d_part < 2) {
        wmma::store_matrix_sync(sc + (d_part * HQ + m_tile * 16) * SLD, s, SLD,
                                wmma::mem_row_major);
      }
      __syncthreads();
      if (d_part >= 2) {
        float* dst = sc + ((d_part - 2) * HQ + m_tile * 16) * SLD;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> p;
        wmma::load_matrix_sync(p, dst, SLD, wmma::mem_row_major);
#pragma unroll
        for (int x = 0; x < 8; ++x) s.x[x] += p.x[x];
        wmma::store_matrix_sync(dst, s, SLD, wmma::mem_row_major);
      }
    }
    __syncthreads();

    // Online softmax: 4 heads/warp, 8 lanes/head, 2 keys/lane.
    {
      const int h = warp * HPW + lane / LPH, k0 = (lane % LPH) * KPL;
      float e[KPL], m = -INFINITY;
#pragma unroll
      for (int i = 0; i < KPL; ++i) {
        const int k = k0 + i;
        const bool live = (lm >> k) & 1u;
        e[i] = live ? (sc[h * SLD + k] + sc[(HQ + h) * SLD + k]) * scale
                    : -INFINITY;
        m = fmaxf(m, e[i]);
      }
#pragma unroll
      for (int o = 1; o < LPH; o <<= 1)
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
      const float m_new = fmaxf(m_run[h], m);
      float l = 0.f;
#pragma unroll
      for (int i = 0; i < KPL; ++i) {
        const int k = k0 + i;
        const float x = ((lm >> k) & 1u) ? __expf(e[i] - m_new) : 0.f;
        l += x;
        ps[h * PLD + k] = __float2half(x);
      }
#pragma unroll
      for (int o = 1; o < LPH; o <<= 1)
        l += __shfl_xor_sync(0xffffffffu, l, o);
      __syncwarp();
      if (lane % LPH == 0) {
        const float a = __expf(m_run[h] - m_new);
        alpha[h] = a;
        l_run[h] = l_run[h] * a + l;
        m_run[h] = m_new;
      }
    }
    __syncthreads();

    // PV: hoist vb[NJ] across the 2 head-tiles.
    {
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> vb[NJ];
#pragma unroll
      for (int j = 0; j < NJ; ++j)
        wmma::load_matrix_sync(vb[j], ks + warp * DPW + j * 16, KLD);
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        float ar_scale[8];
        bool any_scale = false;
#pragma unroll
        for (int x = 0; x < 8; ++x) {
          ar_scale[x] = alpha[i * 16 + ((row_pack >> (x * 4)) & 15)];
          any_scale |= (ar_scale[x] != 1.f);
        }
        if (__any_sync(0xffffffffu, any_scale)) {
#pragma unroll
          for (int j = 0; j < NJ; ++j)
#pragma unroll
            for (int x = 0; x < 8; ++x) acc[i][j].x[x] *= ar_scale[x];
        }
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> pa;
        wmma::load_matrix_sync(pa, ps + i * 16 * PLD, PLD);
#pragma unroll
        for (int j = 0; j < NJ; ++j)
          wmma::mma_sync(acc[i][j], pa, vb[j], acc[i][j]);
      }
    }
    __syncthreads();
  }

  if (threadIdx.x < HQ) {
    const int h = threadIdx.x;
    const float l = l_run[h];
    const float M = l > 0.f ? fmaxf(sink[h], m_run[h]) : sink[h];
    const float w = l > 0.f ? __expf(m_run[h] - M) : 0.f;
    alpha[h] = w / (l * w + __expf(sink[h] - M));
  }
  __syncthreads();
  float* st = reinterpret_cast<float*>(dsm) + warp * 256;
  half* ot = out + ((size_t)tl * HQ_ALL + h_cta * HQ) * D;
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
#pragma unroll
      for (int x = 0; x < 8; ++x)
        acc[i][j].x[x] *= alpha[i * 16 + ((row_pack >> (x * 4)) & 15)];
      wmma::store_matrix_sync(st, acc[i][j], 16, wmma::mem_row_major);
      __syncwarp();
#pragma unroll
      for (int e = lane; e < 128; e += 32) {
        const int r = e >> 3, c = (e & 7) * 2;
        *reinterpret_cast<half2*>(ot + (i * 16 + r) * D + warp * DPW +
                                  j * 16 + c) =
            __floats2half2_rn(st[r * 16 + c], st[r * 16 + c + 1]);
      }
      __syncwarp();
    }
}

torch::Tensor attn64_prof(bool reset) {
  unsigned long long host[8];
  cudaMemcpyFromSymbol(host, g_attn64_prof, sizeof(host));
  if (reset) {
    unsigned long long zero[8] = {};
    cudaMemcpyToSymbol(g_attn64_prof, zero, sizeof(zero));
  }
  auto out = torch::empty({8}, torch::kInt64);
  for (int i = 0; i < 8; ++i) out[i] = (int64_t)host[i];
  return out;
}

// out / q [Tq, 64, 512]: tokens token0 .. token0 + Tq - 1 of the step (rows,
// bases, positions, valid, topk index the whole step; values its window rows).
void attn_sparse64(torch::Tensor out, torch::Tensor q, int64_t token0,
                   torch::Tensor values, torch::Tensor wcache,
                   torch::Tensor wtable, torch::Tensor cache,
                   torch::Tensor table, torch::Tensor rows,
                   torch::Tensor bases, torch::Tensor positions,
                   torch::Tensor valid, torch::Tensor topk,
                   torch::Tensor sink, double scale, int64_t wblock,
                   int64_t block, int64_t ratio, int64_t window,
                   int64_t ntopk) {
  using namespace sp64;
  const int Tq = q.size(0);
  TORCH_CHECK(q.scalar_type() == torch::kFloat16 && q.is_contiguous() &&
              q.size(1) == HQ && q.size(2) == D);
  TORCH_CHECK(out.scalar_type() == torch::kFloat16 && out.is_contiguous() &&
              out.sizes() == q.sizes());
  TORCH_CHECK(sink.scalar_type() == torch::kFloat32 && sink.numel() == HQ);
  TORCH_CHECK(values.scalar_type() == torch::kFloat16 && values.is_contiguous() &&
              values.size(-1) == D);
  TORCH_CHECK(positions.scalar_type() == torch::kInt64 &&
              valid.scalar_type() == torch::kBool);
  TORCH_CHECK(token0 >= 0 && token0 + Tq <= positions.numel());
  TORCH_CHECK(ntopk == 0 ||
              (topk.scalar_type() == torch::kInt32 && topk.is_contiguous() &&
               table.scalar_type() == torch::kInt32));
  TORCH_CHECK(wtable.scalar_type() == torch::kInt32 && wtable.stride(1) == 1);
  TORCH_CHECK(rows.scalar_type() == torch::kInt32 &&
              bases.scalar_type() == torch::kInt32);
  if (Tq == 0) return;
  const at::cuda::OptionalCUDAGuard guard(device_of(q));
  auto stream = at::cuda::getCurrentCUDAStream();
  RingGather rg;
  rg.ring = nullptr;
  rg.values = reinterpret_cast<const half*>(values.data_ptr<at::Half>());
  rg.starts = positions.data_ptr<int64_t>();
  v41_set_compressed(rg, cache, ntopk);
  rg.table = ntopk ? table.data_ptr<int>() : nullptr;
  rg.positions = positions.data_ptr<int64_t>();
  rg.valid = valid.data_ptr<bool>();
  rg.topk = ntopk ? topk.data_ptr<int>() : nullptr;
  rg.stride = ntopk ? cache.stride(0) : 0;
  rg.block = block;
  rg.ratio = ratio;
  rg.window = window;
  rg.ntopk = ntopk;
  rg.ring_rows = 1;
  rg.paged = true;
  v41_set_window(rg, wcache, wblock);
  rg.wtable = wtable.data_ptr<int>();
  rg.rows = rows.data_ptr<int>();
  rg.bases = bases.data_ptr<int>();
  rg.wtable_stride = wtable.stride(0);
  rg.wblock = wblock;
  rg.table_stride = ntopk ? table.stride(0) : 0;
  // V41SK_ATTN64_ABLATE (timing only): 1 skip key decode, 2 skip QK^T, 4 skip PV.
  static const int ablate = [] {
    const char* v = std::getenv("V41SK_ATTN64_ABLATE");
    return v ? std::atoi(v) : 0;
  }();
  static const int warps = [] {
    const char* v = std::getenv("V41SK_ATTN64_WARPS");
    return v ? std::atoi(v) : 8;
  }();
  static const int split2 = [] {
    const char* v = std::getenv("V41SK_ATTN64_SPLIT");
    return v ? std::atoi(v) : 0;
  }();
  static bool smem_set = false;
  if (!smem_set) {
    cudaFuncSetAttribute(attn_sparse_h64<8>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM);
    cudaFuncSetAttribute(attn_sparse_h64<16>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM);
    cudaFuncSetAttribute(attn_sparse_h32,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)sp32::SMEM);
    smem_set = true;
  }
  if (split2 && ablate == 0) {
    attn_sparse_h32<<<dim3(2, Tq), 256, sp32::SMEM, stream>>>(
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(q.data_ptr<at::Half>()), (int)token0,
        (int)(window + ntopk), (float)scale, sink.data_ptr<float>(), rg);
  } else {
    auto* kernel = warps == 8 ? attn_sparse_h64<8> : attn_sparse_h64<16>;
    kernel<<<Tq, 32 * (warps == 8 ? 8 : 16), SMEM, stream>>>(
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(q.data_ptr<at::Half>()), (int)token0,
        (int)(window + ntopk), (float)scale, sink.data_ptr<float>(), rg,
        ablate);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---- Communication-free 4-token x 16-head (M=64) key-union prefill ---------
// Consecutive tokens in a prefill chunk share ~98% of their SWA window keys and
// ~80% of their top-k compressed keys (4-token union ~1.23x single token).
// Grouping G=4 consecutive query tokens x 16 local heads gives M=64 WMMA rows
// per CTA on each TP rank directly (zero TP all-to-all, zero transpose/pad),
// while all 8 warps split D=512 in half during QK^T for 100% warp utilization.
namespace spg4 {
constexpr int G = 4, H = 16, HQ = G * H, D = 512, KT = 16;
constexpr int QLD = D + 8, KLD = D + 8, SLD = KT + 4, PLD = KT + 8;
constexpr size_t K_OFF = sizeof(half) * HQ * QLD;                 // 66,560
constexpr size_t S_OFF = K_OFF + sizeof(half) * KT * KLD;         // 83,200
constexpr size_t P_OFF = S_OFF + sizeof(float) * 2 * HQ * SLD;    // 93,440
constexpr size_t M_OFF = P_OFF + sizeof(half) * HQ * PLD;         // 96,512
constexpr size_t T_OFF = M_OFF + sizeof(float) * 3 * HQ;          // 97,280
constexpr size_t SMEM = 97664;  // <= 98,304 (96 KiB max on sm_70), 0 static
constexpr int HT_CAP = 4096;
}  // namespace spg4

__global__ void __launch_bounds__(256)
    attn_union_g4_kernel(uint64_t* __restrict__ out_srcs,
                         uint64_t* __restrict__ out_tmasks,
                         int* __restrict__ out_ntiles, int token0, int Tq,
                         int width, int max_tiles, RingGather rg) {
  using namespace spg4;
  __shared__ unsigned long long ht[HT_CAP];
  __shared__ int warp_cnt[8];

  const int g = blockIdx.x;
  const int tl0 = g * G;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;

  for (int i = threadIdx.x; i < HT_CAP; i += 256) ht[i] = 0ULL;
  __syncthreads();

  const int total_cand = G * width;
  for (int idx = threadIdx.x; idx < total_cand; idx += 256) {
    const int m = idx / width, key = idx - m * width;
    const int tl = tl0 + m;
    if (tl >= Tq) continue;
    const int t = token0 + tl;
    if (!rg.valid[t]) continue;

    const int64_t pos = rg.positions[t];
    bool live = true;
    const half* src = nullptr;
    const int row = rg.rows[t];
    const int vbase = rg.bases[t];
    if (key < rg.window) {
      const int64_t at = pos - rg.window + 1 + key;
      live = at >= 0;
      const int64_t start = rg.positions[vbase];
      if (live && at >= start) {
        src = rg.values + (vbase + at - start) * D;
      } else if (live) {
        const int page =
            rg.wtable[(size_t)row * rg.wtable_stride + at / rg.wblock];
        live = page >= 0;
        if (live) src = v41_window_src(rg, page, at % rg.wblock);
      }
    } else {
      const int64_t index = rg.topk[(size_t)t * rg.ntopk + key - rg.window];
      const int per = rg.block / rg.ratio;
      live = index >= 0 && index < (pos + 1) / rg.ratio;
      const int page =
          live ? rg.table[(size_t)row * rg.table_stride + index / per] : -1;
      live = page >= 0;
      if (live) src = v41_compressed_src(rg, page, index % per);
    }
    if (!live || src == nullptr) continue;

    const uint64_t addr = reinterpret_cast<uintptr_t>(src);
    const unsigned long long bit = 1ULL << (56 + m);
    uint32_t h =
        (uint32_t)((addr ^ (addr >> 17) ^ (addr >> 33)) * 0x9e3779b1u) &
        (HT_CAP - 1);
    for (;; h = (h + 1u) & (HT_CAP - 1)) {
      unsigned long long cur = ht[h];
      bool done = false;
      for (;;) {
        if (cur == 0ULL) {
          const unsigned long long prev =
              atomicCAS(&ht[h], 0ULL, (unsigned long long)addr | bit);
          if (prev == 0ULL) {
            done = true;
            break;
          }
          cur = prev;
        } else if ((cur & ((1ULL << 56) - 1ULL)) == addr && !(cur & bit)) {
          const unsigned long long prev = atomicCAS(&ht[h], cur, cur | bit);
          if (prev == cur) {
            done = true;
            break;
          }
          cur = prev;
        } else {
          break;
        }
      }
      if (done) break;
    }
  }
  __syncthreads();

  unsigned long long vals[16];
  int my_cnt = 0;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const unsigned long long v = ht[threadIdx.x * 16 + j];
    vals[j] = v;
    if (v != 0ULL) ++my_cnt;
  }
  int pref = my_cnt;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int x = __shfl_up_sync(0xffffffffu, pref, o);
    if (lane >= o) pref += x;
  }
  if (lane == 31) warp_cnt[warp] = pref;
  __syncthreads();

  int wbase = 0, total_keys = 0;
#pragma unroll
  for (int w = 0; w < 8; ++w) {
    if (w < warp) wbase += warp_cnt[w];
    total_keys += warp_cnt[w];
  }
  int pos_out = wbase + pref - my_cnt;
  if (threadIdx.x < 16 && total_keys + (int)threadIdx.x < HT_CAP)
    ht[total_keys + threadIdx.x] = 0ULL;
  __syncthreads();

#pragma unroll
  for (int j = 0; j < 16; ++j) {
    if (vals[j] != 0ULL) ht[pos_out++] = vals[j];
  }
  __syncthreads();

  const int ntiles = (total_keys + KT - 1) / KT;
  if (threadIdx.x == 0) out_ntiles[g] = ntiles;

  uint64_t* g_srcs = out_srcs + (size_t)g * max_tiles * KT;
  uint64_t* g_tmasks = out_tmasks + (size_t)g * max_tiles;
  for (int u = warp; u < ntiles; u += 8) {
    const unsigned long long e =
        (lane < KT && u * KT + lane < total_keys) ? ht[u * KT + lane] : 0ULL;
    if (lane < KT) {
      g_srcs[u * KT + lane] = (uint64_t)(e & ((1ULL << 56) - 1ULL));
    }
    const unsigned m4 = (unsigned)(e >> 56);
    const unsigned b0 = __ballot_sync(0xffffffffu, lane < KT && (m4 & 1u));
    const unsigned b1 = __ballot_sync(0xffffffffu, lane < KT && (m4 & 2u));
    const unsigned b2 = __ballot_sync(0xffffffffu, lane < KT && (m4 & 4u));
    const unsigned b3 = __ballot_sync(0xffffffffu, lane < KT && (m4 & 8u));
    if (lane == 0) {
      g_tmasks[u] = (uint64_t)(b0 & 0xffffu) |
                    ((uint64_t)(b1 & 0xffffu) << 16) |
                    ((uint64_t)(b2 & 0xffffu) << 32) |
                    ((uint64_t)(b3 & 0xffffu) << 48);
    }
  }
}

__global__ void __launch_bounds__(256, 1)
    attn_sparse_g4_kernel(half* __restrict__ out, const half* __restrict__ q,
                          const uint64_t* __restrict__ union_srcs,
                          const uint64_t* __restrict__ union_tmasks,
                          const int* __restrict__ union_ntiles, int Tq,
                          int max_tiles, float scale,
                          const float* __restrict__ sink) {
  using namespace nvcuda;
  using namespace spg4;
  constexpr int WARPS = 8;
  extern __shared__ __align__(128) unsigned char dsm[];
  half* qs = reinterpret_cast<half*>(dsm);
  half* ks = reinterpret_cast<half*>(dsm + K_OFF);
  float* sc = reinterpret_cast<float*>(dsm + S_OFF);
  half* ps = reinterpret_cast<half*>(dsm + P_OFF);
  float* m_run = reinterpret_cast<float*>(dsm + M_OFF);
  float* l_run = m_run + HQ;
  float* alpha = l_run + HQ;
  uint64_t* tmasks = reinterpret_cast<uint64_t*>(dsm + T_OFF);
  const half** srcs = reinterpret_cast<const half**>(dsm + T_OFF + 16);

  const int g = blockIdx.x, tl0 = g * G;
  const int valid_rows = min(G, Tq - tl0) * H;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const half* qt = q + (size_t)tl0 * H * D;
  for (int i = threadIdx.x; i < HQ * D / 8; i += 256) {
    const int h = i / (D / 8), c = (i % (D / 8)) * 8;
    *reinterpret_cast<uint4*>(qs + h * QLD + c) =
        h < valid_rows ? __ldg(reinterpret_cast<const uint4*>(qt + h * D + c))
                       : make_uint4(0, 0, 0, 0);
  }
  if (threadIdx.x < HQ) {
    m_run[threadIdx.x] = -1e30f;
    l_run[threadIdx.x] = 0.f;
  }
  constexpr int DPW = D / WARPS, NJ = DPW / 16, HPW = HQ / WARPS,
                LPH = 32 / HPW, KPL = KT / LPH;
  if (threadIdx.x < 256) sc[threadIdx.x] = (float)(threadIdx.x >> 4);
  __syncthreads();
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4][NJ];
  uint32_t row_pack = 0;
  {
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> rf;
    wmma::load_matrix_sync(rf, sc, 16, wmma::mem_row_major);
#pragma unroll
    for (int x = 0; x < 8; ++x)
      row_pack |= (uint32_t)((int)rf.x[x] & 15) << (x * 4);
  }
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < NJ; ++j) wmma::fill_fragment(acc[i][j], 0.f);

  constexpr int Q_REG = 2;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>
      a_reg[Q_REG][2];
  {
    const int m_half = warp & 1, d_part = warp >> 1;
    const int k_base = d_part * (D / 64);
#pragma unroll
    for (int k = 0; k < Q_REG; ++k) {
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        wmma::load_matrix_sync(
            a_reg[k][m],
            qs + (m_half * 32 + m * 16) * QLD + (k_base + k) * 16, QLD);
      }
    }
  }

  const int tiles = union_ntiles[g];
  const uint64_t* g_srcs = union_srcs + (size_t)g * max_tiles * KT;
  const uint64_t* g_tmasks = union_tmasks + (size_t)g * max_tiles;

  auto resolve = [&](int u) {
    if (lane < KT) {
      srcs[(u & 1) * KT + lane] =
          reinterpret_cast<const half*>(__ldg(g_srcs + u * KT + lane));
    }
    if (lane == 0) tmasks[u & 1] = __ldg(g_tmasks + u);
  };
  constexpr int PER = KT * D / 8 / (32 * WARPS);
  uint4 v[PER];
  auto fetch = [&](int u) {
    const uint64_t tm = tmasks[u & 1];
    const unsigned lm =
        (unsigned)((tm | (tm >> 16) | (tm >> 32) | (tm >> 48)) & 0xffffu);
#pragma unroll
    for (int w = 0; w < PER; ++w) {
      const int i = threadIdx.x + w * (32 * WARPS);
      const int j = i / (D / 8), c = (i % (D / 8)) * 8;
      v[w] = ((lm >> j) & 1u) ? v41_key8(srcs[(u & 1) * KT + j], c)
                              : make_uint4(0, 0, 0, 0);
    }
  };

  if (warp == 0 && tiles > 0) resolve(0);
  __syncthreads();
  if (tiles > 0) fetch(0);

  for (int u = 0; u < tiles; ++u) {
    const uint64_t tm = tmasks[u & 1];
    const unsigned lm =
        (unsigned)((tm | (tm >> 16) | (tm >> 32) | (tm >> 48)) & 0xffffu);
#pragma unroll
    for (int w = 0; w < PER; ++w) {
      const int i = threadIdx.x + w * (32 * WARPS);
      const int j = i / (D / 8), c = (i % (D / 8)) * 8;
      *reinterpret_cast<uint4*>(ks + j * KLD + c) = v[w];
    }
    if (warp == 0 && u + 1 < tiles) resolve(u + 1);
    __syncthreads();
    if (u + 1 < tiles) fetch(u + 1);
    if (lm == 0) {
      __syncthreads();
      continue;
    }
    // QK^T: 4-way split-D x 2-way M (each warp reuses K across 32 rows).
    {
      const int m_half = warp & 1, d_part = warp >> 1;
      const int k_base = d_part * (D / 64);
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> s[2];
      wmma::fill_fragment(s[0], 0.f);
      wmma::fill_fragment(s[1], 0.f);
#pragma unroll
      for (int k = 0; k < Q_REG; ++k) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
        wmma::load_matrix_sync(b, ks + (k_base + k) * 16, KLD);
        wmma::mma_sync(s[0], a_reg[k][0], b, s[0]);
        wmma::mma_sync(s[1], a_reg[k][1], b, s[1]);
      }
#pragma unroll
      for (int k = Q_REG; k < D / 64; ++k) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
        wmma::load_matrix_sync(b, ks + (k_base + k) * 16, KLD);
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
          wmma::load_matrix_sync(
              a, qs + (m_half * 32 + m * 16) * QLD + (k_base + k) * 16, QLD);
          wmma::mma_sync(s[m], a, b, s[m]);
        }
      }
      if (d_part < 2) {
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          wmma::store_matrix_sync(
              sc + (d_part * HQ + m_half * 32 + m * 16) * SLD, s[m], SLD,
              wmma::mem_row_major);
        }
      }
      __syncthreads();
      if (d_part >= 2) {
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          float* dst = sc + ((d_part - 2) * HQ + m_half * 32 + m * 16) * SLD;
          wmma::fragment<wmma::accumulator, 16, 16, 16, float> p;
          wmma::load_matrix_sync(p, dst, SLD, wmma::mem_row_major);
#pragma unroll
          for (int x = 0; x < 8; ++x) s[m].x[x] += p.x[x];
          wmma::store_matrix_sync(dst, s[m], SLD, wmma::mem_row_major);
        }
      }
    }
    __syncthreads();
    // Online softmax with per-token 16-bit live mask.
    {
      const int h = warp * HPW + lane / LPH, k0 = (lane % LPH) * KPL;
      const unsigned lm_h = (unsigned)((tm >> (16 * (h >> 4))) & 0xffffu);
      float e[KPL], m = -INFINITY;
#pragma unroll
      for (int i = 0; i < KPL; ++i) {
        const int k = k0 + i;
        const bool live = (lm_h >> k) & 1u;
        e[i] = live ? (sc[h * SLD + k] + sc[(HQ + h) * SLD + k]) * scale
                    : -INFINITY;
        m = fmaxf(m, e[i]);
      }
#pragma unroll
      for (int o = 1; o < LPH; o <<= 1)
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
      const float m_new = fmaxf(m_run[h], m);
      float l = 0.f;
#pragma unroll
      for (int i = 0; i < KPL; ++i) {
        const int k = k0 + i;
        const float x = ((lm_h >> k) & 1u) ? __expf(e[i] - m_new) : 0.f;
        l += x;
        ps[h * PLD + k] = __float2half(x);
      }
#pragma unroll
      for (int o = 1; o < LPH; o <<= 1) l += __shfl_xor_sync(0xffffffffu, l, o);
      __syncwarp();
      if (lane % LPH == 0) {
        const float a = lm_h ? __expf(m_run[h] - m_new) : 1.f;
        alpha[h] = a;
        l_run[h] = l_run[h] * a + l;
        m_run[h] = m_new;
      }
    }
    __syncthreads();
    // PV: this warp's 64 output columns for all 64 (4 tokens x 16 heads) rows.
    {
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> vb[NJ];
#pragma unroll
      for (int j = 0; j < NJ; ++j)
        wmma::load_matrix_sync(vb[j], ks + warp * DPW + j * 16, KLD);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float ar_scale[8];
        bool any_scale = false;
#pragma unroll
        for (int x = 0; x < 8; ++x) {
          ar_scale[x] = alpha[i * 16 + ((row_pack >> (x * 4)) & 15)];
          any_scale |= (ar_scale[x] != 1.f);
        }
        if (__any_sync(0xffffffffu, any_scale)) {
#pragma unroll
          for (int j = 0; j < NJ; ++j)
#pragma unroll
            for (int x = 0; x < 8; ++x) acc[i][j].x[x] *= ar_scale[x];
        }
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> pa;
        wmma::load_matrix_sync(pa, ps + i * 16 * PLD, PLD);
#pragma unroll
        for (int j = 0; j < NJ; ++j)
          wmma::mma_sync(acc[i][j], pa, vb[j], acc[i][j]);
      }
    }
    __syncthreads();
  }
  if (threadIdx.x < HQ) {
    const int h = threadIdx.x;
    const float sk = sink[h & 15];
    const float l = l_run[h];
    const float M = l > 0.f ? fmaxf(sk, m_run[h]) : sk;
    const float w = l > 0.f ? __expf(m_run[h] - M) : 0.f;
    alpha[h] = w / (l * w + __expf(sk - M));
  }
  __syncthreads();
  float* st = reinterpret_cast<float*>(dsm) + warp * 256;
  half* ot = out + (size_t)tl0 * H * D;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    if (i * 16 >= valid_rows) break;
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
#pragma unroll
      for (int x = 0; x < 8; ++x)
        acc[i][j].x[x] *= alpha[i * 16 + ((row_pack >> (x * 4)) & 15)];
      wmma::store_matrix_sync(st, acc[i][j], 16, wmma::mem_row_major);
      __syncwarp();
#pragma unroll
      for (int e = lane; e < 128; e += 32) {
        const int r = e >> 3, c = (e & 7) * 2;
        if (i * 16 + r < valid_rows) {
          *reinterpret_cast<half2*>(ot + (i * 16 + r) * D + warp * DPW +
                                    j * 16 + c) =
              __floats2half2_rn(st[r * 16 + c], st[r * 16 + c + 1]);
        }
      }
      __syncwarp();
    }
  }
}

void attn_sparse_g4(torch::Tensor out, torch::Tensor q, int64_t token0,
                    torch::Tensor values, torch::Tensor wcache,
                    torch::Tensor wtable, torch::Tensor cache,
                    torch::Tensor table, torch::Tensor rows,
                    torch::Tensor bases, torch::Tensor positions,
                    torch::Tensor valid, torch::Tensor topk,
                    torch::Tensor sink, double scale, int64_t wblock,
                    int64_t block, int64_t ratio, int64_t window,
                    int64_t ntopk) {
  using namespace spg4;
  const int Tq = q.size(0);
  const int width = (int)(window + ntopk);
  TORCH_CHECK(q.scalar_type() == torch::kFloat16 && q.is_contiguous() &&
              q.size(1) == H && q.size(2) == D);
  TORCH_CHECK(out.scalar_type() == torch::kFloat16 && out.is_contiguous() &&
              out.sizes() == q.sizes());
  TORCH_CHECK(sink.scalar_type() == torch::kFloat32 && sink.numel() == H);
  TORCH_CHECK(values.scalar_type() == torch::kFloat16 && values.is_contiguous() &&
              values.size(-1) == D);
  TORCH_CHECK(positions.scalar_type() == torch::kInt64 &&
              valid.scalar_type() == torch::kBool);
  TORCH_CHECK(token0 >= 0 && token0 + Tq <= positions.numel());
  TORCH_CHECK(width > 0 && G * width <= 3584);
  TORCH_CHECK(ntopk == 0 ||
              (topk.scalar_type() == torch::kInt32 && topk.is_contiguous() &&
               table.scalar_type() == torch::kInt32));
  TORCH_CHECK(wtable.scalar_type() == torch::kInt32 && wtable.stride(1) == 1);
  TORCH_CHECK(rows.scalar_type() == torch::kInt32 &&
              bases.scalar_type() == torch::kInt32);
  if (Tq == 0) return;
  const at::cuda::OptionalCUDAGuard guard(device_of(q));
  auto stream = at::cuda::getCurrentCUDAStream();
  RingGather rg;
  rg.ring = nullptr;
  rg.values = reinterpret_cast<const half*>(values.data_ptr<at::Half>());
  rg.starts = positions.data_ptr<int64_t>();
  v41_set_compressed(rg, cache, ntopk);
  rg.table = ntopk ? table.data_ptr<int>() : nullptr;
  rg.positions = positions.data_ptr<int64_t>();
  rg.valid = valid.data_ptr<bool>();
  rg.topk = ntopk ? topk.data_ptr<int>() : nullptr;
  rg.stride = ntopk ? cache.stride(0) : 0;
  rg.block = block;
  rg.ratio = ratio;
  rg.window = window;
  rg.ntopk = ntopk;
  rg.ring_rows = 1;
  rg.paged = true;
  v41_set_window(rg, wcache, wblock);
  rg.wtable = wtable.data_ptr<int>();
  rg.rows = rows.data_ptr<int>();
  rg.bases = bases.data_ptr<int>();
  rg.wtable_stride = wtable.stride(0);
  rg.wblock = wblock;
  rg.table_stride = ntopk ? table.stride(0) : 0;

  const int ngroups = (Tq + G - 1) / G;
  const int max_tiles = (G * width + KT - 1) / KT;
  auto opts_i64 = q.options().dtype(torch::kInt64);
  auto opts_i32 = q.options().dtype(torch::kInt32);
  auto union_srcs = torch::empty({ngroups, max_tiles * KT}, opts_i64);
  auto union_tmasks = torch::empty({ngroups, max_tiles}, opts_i64);
  auto union_ntiles = torch::empty({ngroups}, opts_i32);

  attn_union_g4_kernel<<<ngroups, 256, 0, stream>>>(
      reinterpret_cast<uint64_t*>(union_srcs.data_ptr<int64_t>()),
      reinterpret_cast<uint64_t*>(union_tmasks.data_ptr<int64_t>()),
      union_ntiles.data_ptr<int>(), (int)token0, Tq, width, max_tiles, rg);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  static bool smem_set = false;
  if (!smem_set) {
    cudaFuncSetAttribute(attn_sparse_g4_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)SMEM);
    smem_set = true;
  }
  attn_sparse_g4_kernel<<<ngroups, 256, SMEM, stream>>>(
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<const uint64_t*>(union_srcs.data_ptr<int64_t>()),
      reinterpret_cast<const uint64_t*>(union_tmasks.data_ptr<int64_t>()),
      union_ntiles.data_ptr<int>(), Tq, max_tiles, (float)scale,
      sink.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---- Communication-free 16-token x 16-head (M=256) cuBLAS TensorCore path ---
// On real 64K prefill routes, 16 consecutive tokens share ~858 unique keys out
// of 16 x 640 = 10,240 candidates (11.1x overlap, 3.28x fewer key gathers than
// G=4). At G=16, q [Tq, 16, 512] is already a zero-copy [Tq/16, 256, 512]
// contiguous batch: pre-dequantize the group's union keys once into reusable
// scratch, multiply QK^T and PV via Volta HMMA cublasGemmStridedBatchedEx, and
// fuse the 16-bit per-token mask + sink softmax in place in FP32.
namespace spg16 {
constexpr int G = 16, H = 16, HQ = G * H, D = 512;
constexpr int HT_CAP = 11264;  // 88 KiB + 32 B <= 96 KiB dynamic smem on sm_70
constexpr size_t UNION_SMEM = HT_CAP * sizeof(unsigned long long) + 32;
}  // namespace spg16

__global__ void __launch_bounds__(256)
    attn_union_g16_kernel(uint64_t* __restrict__ out_ptrs,
                          uint16_t* __restrict__ out_masks,
                          int* __restrict__ out_cnts,
                          int* __restrict__ d_max_u, int token0, int Tq,
                          int width, int max_u_cap, RingGather rg) {
  using namespace spg16;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  unsigned long long* ht = reinterpret_cast<unsigned long long*>(smem_raw);
  int* warp_cnt = reinterpret_cast<int*>(ht + HT_CAP);

  const int g = blockIdx.x;
  const int tl0 = g * G;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;

  for (int i = threadIdx.x; i < HT_CAP; i += 256) ht[i] = 0ULL;
  __syncthreads();

  const int total_cand = G * width;
  for (int idx = threadIdx.x; idx < total_cand; idx += 256) {
    const int m = idx / width, key = idx - m * width;
    const int tl = tl0 + m;
    if (tl >= Tq) continue;
    const int t = token0 + tl;
    if (!rg.valid[t]) continue;

    const int64_t pos = rg.positions[t];
    bool live = true;
    const half* src = nullptr;
    const int row = rg.rows[t];
    const int vbase = rg.bases[t];
    if (key < rg.window) {
      const int64_t at = pos - rg.window + 1 + key;
      live = at >= 0;
      const int64_t start = rg.positions[vbase];
      if (live && at >= start) {
        src = rg.values + (vbase + at - start) * D;
      } else if (live) {
        const int page =
            rg.wtable[(size_t)row * rg.wtable_stride + at / rg.wblock];
        live = page >= 0;
        if (live) src = v41_window_src(rg, page, at % rg.wblock);
      }
    } else {
      const int64_t index = rg.topk[(size_t)t * rg.ntopk + key - rg.window];
      const int per = rg.block / rg.ratio;
      live = index >= 0 && index < (pos + 1) / rg.ratio;
      const int page =
          live ? rg.table[(size_t)row * rg.table_stride + index / per] : -1;
      live = page >= 0;
      if (live) src = v41_compressed_src(rg, page, index % per);
    }
    if (!live || src == nullptr) continue;

    const uint64_t addr =
        reinterpret_cast<uintptr_t>(src) & ((1ULL << 48) - 1ULL);
    const unsigned long long bit = 1ULL << (48 + m);
    uint32_t h = (uint32_t)(((addr ^ (addr >> 17) ^ (addr >> 33)) * 0x9e3779b1u) %
                            (uint32_t)HT_CAP);
    for (int probes = 0; probes < HT_CAP; ++probes) {
      unsigned long long cur = ht[h];
      bool done = false;
      for (;;) {
        if (cur == 0ULL) {
          const unsigned long long prev =
              atomicCAS(&ht[h], 0ULL, (unsigned long long)addr | bit);
          if (prev == 0ULL) {
            done = true;
            break;
          }
          cur = prev;
        } else if ((cur & ((1ULL << 48) - 1ULL)) == addr && !(cur & bit)) {
          const unsigned long long prev = atomicCAS(&ht[h], cur, cur | bit);
          if (prev == cur) {
            done = true;
            break;
          }
          cur = prev;
        } else {
          break;
        }
      }
      if (done) break;
      if (++h == (uint32_t)HT_CAP) h = 0;
    }
  }
  __syncthreads();

  int my_cnt = 0;
  for (int i = threadIdx.x; i < HT_CAP; i += 256) {
    if (ht[i] != 0ULL) ++my_cnt;
  }
  int pref = my_cnt;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int x = __shfl_up_sync(0xffffffffu, pref, o);
    if (lane >= o) pref += x;
  }
  if (lane == 31) warp_cnt[warp] = pref;
  __syncthreads();

  int wbase = 0, total_keys = 0;
#pragma unroll
  for (int w = 0; w < 8; ++w) {
    if (w < warp) wbase += warp_cnt[w];
    total_keys += warp_cnt[w];
  }
  int pos_out = wbase + pref - my_cnt;
  const int capped = min(total_keys, max_u_cap);
  uint64_t* g_ptrs = out_ptrs + (size_t)g * max_u_cap;
  uint16_t* g_masks = out_masks + (size_t)g * max_u_cap;

  for (int i = threadIdx.x; i < HT_CAP; i += 256) {
    const unsigned long long v = ht[i];
    if (v != 0ULL) {
      if (pos_out < capped) {
        g_ptrs[pos_out] = (uint64_t)(v & ((1ULL << 48) - 1ULL));
        g_masks[pos_out] = (uint16_t)(v >> 48);
      }
      ++pos_out;
    }
  }
  if (threadIdx.x == 0) {
    out_cnts[g] = capped;
    const int u_pad = ((capped + 31) / 32) * 32;
    if (u_pad > 0) atomicMax(d_max_u, u_pad);
  }
}

__global__ void __launch_bounds__(256)
    attn_dequant_g16_kernel(half* __restrict__ k_union,
                            const uint64_t* __restrict__ union_ptrs,
                            const int* __restrict__ union_cnts, int g0, int U,
                            int max_u_cap) {
  using namespace spg16;
  const int b = blockIdx.x;
  const int g = g0 + b;
  const int cnt = union_cnts[g];
  const int u = blockIdx.y * 4 + (threadIdx.x >> 6);
  const int c = (threadIdx.x & 63) * 8;
  uint4 val = make_uint4(0, 0, 0, 0);
  if (u < cnt) {
    const half* src =
        reinterpret_cast<const half*>(union_ptrs[(size_t)g * max_u_cap + u]);
    if (src != nullptr) val = v41_key8(src, c);
  }
  *reinterpret_cast<uint4*>(k_union + ((size_t)b * U + u) * D + c) = val;
}

__global__ void __launch_bounds__(256)
    attn_softmax_sink_g16_kernel(float* __restrict__ scores,
                                 const uint16_t* __restrict__ union_masks,
                                 const int* __restrict__ union_cnts,
                                 const float* __restrict__ sink,
                                 const bool* __restrict__ valid, int token0,
                                 int Tq, int g0, int U, int max_u_cap) {
  using namespace spg16;
  const int b = blockIdx.x;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int m = blockIdx.y * 8 + warp;
  const int sub_t = m >> 4, head = m & 15;
  const int g = g0 + b;
  const int tl = g * G + sub_t;
  float* s_row = scores + ((size_t)b * HQ + m) * U;
  half* p_row = reinterpret_cast<half*>(s_row);

  if (tl >= Tq || !valid[token0 + tl]) {
    for (int u = lane; u < U; u += 32) p_row[u] = __float2half_rn(0.f);
    return;
  }

  const int cnt = union_cnts[g];
  const uint16_t* g_masks = union_masks + (size_t)g * max_u_cap;
  const uint16_t bit = (uint16_t)(1u << sub_t);
  const float sk = sink[head];

  float rmax = -INFINITY;
  for (int u = lane; u < cnt; u += 32) {
    if (g_masks[u] & bit) rmax = fmaxf(rmax, s_row[u]);
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    rmax = fmaxf(rmax, __shfl_xor_sync(0xffffffffu, rmax, o));
  const float M_val = fmaxf(rmax, sk);

  float rsum = 0.f;
  for (int u = lane; u < cnt; u += 32) {
    if (g_masks[u] & bit) rsum += __expf(s_row[u] - M_val);
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    rsum += __shfl_xor_sync(0xffffffffu, rsum, o);
  const float inv = 1.f / (rsum + __expf(sk - M_val));
  __syncwarp();

  for (int k0 = 0; k0 < U; k0 += 32) {
    const int u = k0 + lane;
    const bool live = (u < cnt) && ((g_masks[u] & bit) != 0);
    const float sv = live ? s_row[u] : 0.f;
    if (k0 == 0) __syncwarp();
    p_row[u] =
        live ? __float2half_rn(__expf(sv - M_val) * inv) : __float2half_rn(0.f);
  }
}

void attn_sparse_g16(torch::Tensor out, torch::Tensor q, int64_t token0,
                     torch::Tensor values, torch::Tensor wcache,
                     torch::Tensor wtable, torch::Tensor cache,
                     torch::Tensor table, torch::Tensor rows,
                     torch::Tensor bases, torch::Tensor positions,
                     torch::Tensor valid, torch::Tensor topk,
                     torch::Tensor sink, double scale, int64_t wblock,
                     int64_t block, int64_t ratio, int64_t window,
                     int64_t ntopk, torch::Tensor scratch) {
  using namespace spg16;
  const int Tq = q.size(0);
  const int width = (int)(window + ntopk);
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kFloat16 &&
              q.is_contiguous() && q.size(1) == H && q.size(2) == D);
  TORCH_CHECK(out.scalar_type() == torch::kFloat16 && out.is_contiguous() &&
              out.sizes() == q.sizes());
  TORCH_CHECK(sink.scalar_type() == torch::kFloat32 && sink.numel() == H);
  TORCH_CHECK(values.scalar_type() == torch::kFloat16 &&
              values.is_contiguous() && values.size(-1) == D);
  TORCH_CHECK(positions.scalar_type() == torch::kInt64 &&
              valid.scalar_type() == torch::kBool && valid.is_contiguous());
  TORCH_CHECK(token0 >= 0 && token0 + Tq <= positions.numel());
  TORCH_CHECK(width > 0 && width <= 896);
  TORCH_CHECK(ntopk == 0 ||
              (topk.scalar_type() == torch::kInt32 && topk.is_contiguous() &&
               table.scalar_type() == torch::kInt32));
  TORCH_CHECK(wtable.scalar_type() == torch::kInt32 && wtable.stride(1) == 1);
  TORCH_CHECK(rows.scalar_type() == torch::kInt32 &&
              bases.scalar_type() == torch::kInt32);
  TORCH_CHECK(scratch.is_cuda() && scratch.is_contiguous());
  TORCH_CHECK((reinterpret_cast<uintptr_t>(q.data_ptr()) >> 48) == 0 &&
              (reinterpret_cast<uintptr_t>(wcache.data_ptr()) >> 48) == 0 &&
              (reinterpret_cast<uintptr_t>(cache.data_ptr()) >> 48) == 0);
  if (Tq == 0) return;

  const at::cuda::OptionalCUDAGuard guard(device_of(q));
  auto stream = at::cuda::getCurrentCUDAStream();
  RingGather rg;
  rg.ring = nullptr;
  rg.values = reinterpret_cast<const half*>(values.data_ptr<at::Half>());
  rg.starts = positions.data_ptr<int64_t>();
  v41_set_compressed(rg, cache, ntopk);
  rg.table = ntopk ? table.data_ptr<int>() : nullptr;
  rg.positions = positions.data_ptr<int64_t>();
  rg.valid = valid.data_ptr<bool>();
  rg.topk = ntopk ? topk.data_ptr<int>() : nullptr;
  rg.stride = ntopk ? cache.stride(0) : 0;
  rg.block = block;
  rg.ratio = ratio;
  rg.window = window;
  rg.ntopk = ntopk;
  rg.ring_rows = 1;
  rg.paged = true;
  v41_set_window(rg, wcache, wblock);
  rg.wtable = wtable.data_ptr<int>();
  rg.rows = rows.data_ptr<int>();
  rg.bases = bases.data_ptr<int>();
  rg.wtable_stride = wtable.stride(0);
  rg.wblock = wblock;
  rg.table_stride = ntopk ? table.stride(0) : 0;

  const int ngroups = (Tq + G - 1) / G;
  const int max_u_cap =
      std::min<int>(HT_CAP - 256, ((G * width + 31) / 32) * 32);

  auto align256 = [](size_t x) { return (x + 255) & ~size_t(255); };
  uint8_t* ws_base = reinterpret_cast<uint8_t*>(scratch.data_ptr());
  const size_t ws_bytes = (size_t)scratch.numel() * scratch.element_size();
  size_t cur_off = 0;
  auto alloc_ws = [&](size_t nbytes) -> uint8_t* {
    const size_t sz = align256(nbytes);
    TORCH_CHECK(cur_off + sz <= ws_bytes,
                "attn_sparse_g16: scratch workspace exhausted");
    uint8_t* p = ws_base + cur_off;
    cur_off += sz;
    return p;
  };

  auto* union_ptrs = reinterpret_cast<uint64_t*>(
      alloc_ws((size_t)ngroups * max_u_cap * sizeof(uint64_t)));
  auto* union_masks = reinterpret_cast<uint16_t*>(
      alloc_ws((size_t)ngroups * max_u_cap * sizeof(uint16_t)));
  auto* union_cnts =
      reinterpret_cast<int*>(alloc_ws((size_t)ngroups * sizeof(int)));
  auto* d_max_u = reinterpret_cast<int*>(alloc_ws(sizeof(int)));
  auto* q_tail = reinterpret_cast<half*>(alloc_ws(HQ * D * sizeof(half)));
  auto* out_tail = reinterpret_cast<half*>(alloc_ws(HQ * D * sizeof(half)));

  C10_CUDA_CHECK(cudaMemsetAsync(d_max_u, 0, sizeof(int), stream));

  static bool smem_set = false;
  if (!smem_set) {
    cudaFuncSetAttribute(attn_union_g16_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)UNION_SMEM);
    smem_set = true;
  }
  attn_union_g16_kernel<<<ngroups, 256, UNION_SMEM, stream>>>(
      union_ptrs, union_masks, union_cnts, d_max_u, (int)token0, Tq, width,
      max_u_cap, rg);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // Pinned host int per device for fast ~2 us D2H readback of max union width.
  static int* h_max_u_dev[16] = {};
  const int dev = q.get_device();
  TORCH_CHECK(dev >= 0 && dev < 16);
  if (!h_max_u_dev[dev]) {
    C10_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h_max_u_dev[dev]),
                                 sizeof(int), cudaHostAllocDefault));
  }
  C10_CUDA_CHECK(cudaMemcpyAsync(h_max_u_dev[dev], d_max_u, sizeof(int),
                                 cudaMemcpyDeviceToHost, stream));
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  const int raw_u = *h_max_u_dev[dev];
  if (raw_u <= 0) {
    out.zero_();
    return;
  }
  const int U = std::max(32, ((raw_u + 31) / 32) * 32);

  const size_t rem_bytes = ws_bytes - cur_off;
  const size_t k_per_b = (size_t)U * D * sizeof(half);
  const size_t s_per_b = (size_t)HQ * U * sizeof(float);
  const int max_b = (int)((rem_bytes - 512) / (k_per_b + s_per_b));
  TORCH_CHECK(max_b >= 1, "attn_sparse_g16: scratch too small for U");

  cublasHandle_t blas = at::cuda::getCurrentCUDABlasHandle();
  cublasSetStream(blas, stream);
  const float alpha_qk = (float)scale, alpha_pv = 1.0f, beta_zero = 0.0f;

  auto run_slice = [&](int g0, int B, const half* q_in, half* o_out) {
    half* k_union = reinterpret_cast<half*>(ws_base + cur_off);
    float* scores = reinterpret_cast<float*>(
        ws_base + cur_off + align256((size_t)B * k_per_b));
    half* p_half = reinterpret_cast<half*>(scores);

    attn_dequant_g16_kernel<<<dim3(B, U / 4), 256, 0, stream>>>(
        k_union, union_ptrs, union_cnts, g0, U, max_u_cap);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // QK^T: Scores_cm(U, 256) = scale * K_cm(512, U)^T * Q_cm(512, 256)
    cublasStatus_t st1 = cublasGemmStridedBatchedEx(
        blas, CUBLAS_OP_T, CUBLAS_OP_N, U, HQ, D, &alpha_qk, k_union,
        CUDA_R_16F, D, (long long)U * D, q_in, CUDA_R_16F, D, (long long)HQ * D,
        &beta_zero, scores, CUDA_R_32F, U, (long long)HQ * U, B, CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    TORCH_CHECK(st1 == CUBLAS_STATUS_SUCCESS,
                "attn_sparse_g16: cublas QK^T failed");

    attn_softmax_sink_g16_kernel<<<dim3(B, 32), 256, 0, stream>>>(
        scores, union_masks, union_cnts, sink.data_ptr<float>(),
        valid.data_ptr<bool>(), (int)token0, Tq, g0, U, max_u_cap);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // PV: Out_cm(512, 256) = K_cm(512, U) * P_cm(U, 256), ldb(P) = 2 * U
    cublasStatus_t st2 = cublasGemmStridedBatchedEx(
        blas, CUBLAS_OP_N, CUBLAS_OP_N, D, HQ, U, &alpha_pv, k_union,
        CUDA_R_16F, D, (long long)U * D, p_half, CUDA_R_16F, 2 * U,
        (long long)HQ * 2 * U, &beta_zero, o_out, CUDA_R_16F, D,
        (long long)HQ * D, B, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    TORCH_CHECK(st2 == CUBLAS_STATUS_SUCCESS,
                "attn_sparse_g16: cublas PV failed");
  };

  const half* q_ptr = reinterpret_cast<const half*>(q.data_ptr<at::Half>());
  half* out_ptr = reinterpret_cast<half*>(out.data_ptr<at::Half>());
  const int full_groups = Tq / G;
  for (int g0 = 0; g0 < full_groups; g0 += max_b) {
    const int B = std::min(max_b, full_groups - g0);
    run_slice(g0, B, q_ptr + (size_t)g0 * HQ * D,
              out_ptr + (size_t)g0 * HQ * D);
  }
  const int rem_toks = Tq - full_groups * G;
  if (rem_toks > 0) {
    const size_t tail_bytes = (size_t)rem_toks * H * D * sizeof(half);
    C10_CUDA_CHECK(cudaMemsetAsync(q_tail, 0, HQ * D * sizeof(half), stream));
    C10_CUDA_CHECK(cudaMemcpyAsync(q_tail,
                                   q_ptr + (size_t)full_groups * HQ * D,
                                   tail_bytes, cudaMemcpyDeviceToDevice,
                                   stream));
    run_slice(full_groups, 1, q_tail, out_tail);
    C10_CUDA_CHECK(cudaMemcpyAsync(out_ptr + (size_t)full_groups * HQ * D,
                                   out_tail, tail_bytes,
                                   cudaMemcpyDeviceToDevice, stream));
  }
}

// Exact E4M3FN round-trip for finite values. Power-of-two scales keep the
// normalization and dequantization exact in FP32.
__device__ __forceinline__ float attn_fp8_round(float value) {
  const float magnitude = fminf(fabsf(value), 448.f);
  float rounded;
  if (magnitude < 0x1p-6f) {
    rounded = (float)__float2int_rn(magnitude * 512.f) * 0x1p-9f;
  } else {
    const unsigned exponent = (__float_as_uint(magnitude) >> 23) & 255u;
    const float step = __uint_as_float((exponent - 3u) << 23);
    const float reciprocal = __uint_as_float((257u - exponent) << 23);
    rounded = (float)__float2int_rn(magnitude * reciprocal) * step;
  }
  return copysignf(rounded, value);
}

template <bool POST>
__global__ void __launch_bounds__(128)
    attn_dec_combine(const half* __restrict__ opart,
                     const float* __restrict__ mpart,
                     const float* __restrict__ lpart,
                     const float* __restrict__ sink, half* __restrict__ out,
                     int nchunk, RingGather rg, const float* rope,
                     int rope_length, half* ring_output) {
  // One CTA per (token, head): chunk weights exp(m_c - M) / L computed once
  // into shared memory, then 128 threads x 4 dims read the FP16 partials.
  using namespace attn;
  __shared__ float wsh[64];
  __shared__ float inv_l;
  const int t = blockIdx.x, h = blockIdx.y;
  if constexpr (POST) {
    if (h == H) {
      const int64_t position = rg.positions[t];
      if (rg.valid[t] && position >= 0) {
        const half* source;
        half* destination;
        if (rg.paged) {  // the step's window row into its page
          const int page = rg.wtable[(size_t)rg.rows[t] * rg.wtable_stride +
                                     position / rg.wblock];
          if (page < 0) return;
          source = rg.values + (size_t)t * D;
          if (rg.wpacked) {
            // FP8 row: 128 threads x 4 values; 8 threads share a 32-group.
            unsigned char* row =
                const_cast<unsigned char*>(
                    reinterpret_cast<const unsigned char*>(rg.wcache)) +
                page * rg.wstride + (position % rg.wblock) * V41_W_ROW;
            const int i = threadIdx.x * 4;
            const uint2 raw = *reinterpret_cast<const uint2*>(source + i);
            const half2* pair = reinterpret_cast<const half2*>(&raw);
            const float2 lo = __half22float2(pair[0]), hi = __half22float2(pair[1]);
            float amax = fmaxf(fmaxf(fabsf(lo.x), fabsf(lo.y)),
                               fmaxf(fabsf(hi.x), fabsf(hi.y)));
#pragma unroll
            for (int o = 1; o < 8; o <<= 1)
              amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
            const unsigned exponent = v41_fp8_group_exponent(amax);
            const float inv = __uint_as_float((254u - exponent) << 23);
            const unsigned codes = v41_e4m3_code(lo.x * inv) |
                                   (v41_e4m3_code(lo.y * inv) << 8) |
                                   (v41_e4m3_code(hi.x * inv) << 16) |
                                   (v41_e4m3_code(hi.y * inv) << 24);
            *reinterpret_cast<unsigned*>(row + i) = codes;
            if ((threadIdx.x & 7) == 0) row[512 + i / 32] = (unsigned char)exponent;
            return;
          }
          destination = const_cast<half*>(rg.wcache) + page * rg.wstride +
                        (position % rg.wblock) * D;
        } else {
          source = rg.values + (position - rg.starts[0]) * D;
          destination = ring_output + (position % rg.ring_rows) * D;
        }
        for (int i = threadIdx.x; i < D / 8; i += blockDim.x)
          reinterpret_cast<uint4*>(destination)[i] =
              reinterpret_cast<const uint4*>(source)[i];
      }
      return;
    }
  }
  const size_t base = (size_t)t * nchunk * H + h;
  if (threadIdx.x < 32) {
    float M = sink[h];
    for (int c = threadIdx.x; c < nchunk; c += 32)
      if (lpart[base + (size_t)c * H] > 0.f)
        M = fmaxf(M, mpart[base + (size_t)c * H]);
    for (int o = 16; o; o >>= 1)
      M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));
    float L = 0.f;
    for (int c = threadIdx.x; c < nchunk; c += 32) {
      const float l = lpart[base + (size_t)c * H];
      const float w = l > 0.f ? __expf(mpart[base + (size_t)c * H] - M) : 0.f;
      wsh[c] = w;
      L += w * l;
    }
    for (int o = 16; o; o >>= 1) L += __shfl_xor_sync(0xffffffffu, L, o);
    if (threadIdx.x == 0) inv_l = 1.f / (L + __expf(sink[h] - M));
  }
  __syncthreads();
  const float inv = inv_l;
  const int d = threadIdx.x * 4;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int c = 0; c < nchunk; ++c) {
    const float w = wsh[c];
    if (w == 0.f) continue;
    const uint2 v = *reinterpret_cast<const uint2*>(
        opart + ((size_t)(t * nchunk + c) * H + h) * D + d);
    const half2* hv = reinterpret_cast<const half2*>(&v);
    const float2 a = __half22float2(hv[0]), b = __half22float2(hv[1]);
    acc[0] += w * a.x;
    acc[1] += w * a.y;
    acc[2] += w * b.x;
    acc[3] += w * b.y;
  }
  if constexpr (POST) {
    float value[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
      value[i] = __half2float(__float2half_rn(acc[i] * inv));
    if (d >= D - 64) {
      const int64_t position = rg.positions[t];
      const int first_pair = (d - (D - 64)) / 2;
#pragma unroll
      for (int pair = 0; pair < 2; ++pair) {
        const int index = first_pair + pair;
        float cosine, sine;
        if (position < rope_length) {
          cosine = rope[position * 64 + index];
          sine = -rope[position * 64 + 32 + index];
        } else {
          // Full-range sincosf: __cosf/__sinf are only accurate near zero.
          const float angle = (float)position * rope[rope_length * 64 + index];
          sincosf(angle, &sine, &cosine);
          sine = -sine;
        }
        const float a = value[pair * 2], b = value[pair * 2 + 1];
        value[pair * 2] = __half2float(__float2half_rn(
            __fsub_rn(__fmul_rn(a, cosine), __fmul_rn(b, sine))));
        value[pair * 2 + 1] = __half2float(__float2half_rn(
            __fadd_rn(__fmul_rn(a, sine), __fmul_rn(b, cosine))));
      }
    }
    float maximum = fmaxf(fmaxf(fabsf(value[0]), fabsf(value[1])),
                          fmaxf(fabsf(value[2]), fabsf(value[3])));
#pragma unroll
    for (int delta = 16; delta; delta >>= 1)
      maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, delta));
    const unsigned bits =
        __float_as_uint(fmaxf(maximum, 1e-4f) * (1.f / 448.f));
    const unsigned exponent = (bits >> 23) + ((bits & 0x7fffffu) != 0);
    const float quant_scale = __uint_as_float(exponent << 23);
    const float reciprocal = __uint_as_float((254u - exponent) << 23);
#pragma unroll
    for (int i = 0; i < 4; ++i)
      value[i] = attn_fp8_round(value[i] * reciprocal) * quant_scale;
    half2* output = reinterpret_cast<half2*>(out + ((size_t)t * H + h) * D + d);
    output[0] = __floats2half2_rn(value[0], value[1]);
    output[1] = __floats2half2_rn(value[2], value[3]);
  } else {
    half2* output = reinterpret_cast<half2*>(out + ((size_t)t * H + h) * D + d);
    output[0] = __floats2half2_rn(acc[0] * inv, acc[1] * inv);
    output[1] = __floats2half2_rn(acc[2] * inv, acc[3] * inv);
  }
}

// Fused prefill/decode attention prep:
// 1) In-place RoPE on q[:, :, 448:512] (skips reading/writing 448/512 = 87.5%
//    of q entirely).
// 2) Strided kv[:, :512] RMSNorm + RoPE + group-{32,128} E4M3 QDQ -> window.
template <typename POS_T, int WGRP>
__global__ void __launch_bounds__(128)
    attn_prep_cuda_kernel(half* __restrict__ q, const half* __restrict__ kv,
                          int64_t kv_stride, const float* __restrict__ kv_weight,
                          const POS_T* __restrict__ positions,
                          const float* __restrict__ table,
                          half* __restrict__ window, int heads, int rope_len,
                          float eps) {
  __shared__ float s_sq[4];
  __shared__ float s_inv;
  const int t = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int64_t pos = static_cast<int64_t>(positions[t]);

  // Part 1: In-place rotary tail [448..511] across all query heads of token t.
  {
    float q_cos, q_sin;
    if (pos < rope_len) {
      q_cos = __ldg(table + pos * 64 + lane);
      q_sin = __ldg(table + pos * 64 + 32 + lane);
    } else {
      const float angle =
          static_cast<float>(pos) * __ldg(table + rope_len * 64 + lane);
      sincosf(angle, &q_sin, &q_cos);
    }
    for (int h = warp; h < heads; h += 4) {
      half2* qp =
          reinterpret_cast<half2*>(q + ((size_t)t * heads + h) * 512 + 448) +
          lane;
      const float2 ab = __half22float2(*qp);
      const float rx =
          __fsub_rn(__fmul_rn(ab.x, q_cos), __fmul_rn(ab.y, q_sin));
      const float ry =
          __fadd_rn(__fmul_rn(ab.x, q_sin), __fmul_rn(ab.y, q_cos));
      *qp = __floats2half2_rn(rx, ry);
    }
  }

  // Part 2: kv [t, 0..511] RMSNorm + RoPE + E4M3 QDQ -> window [t, 0..511].
  const int d = tid * 4;
  const uint2 raw =
      __ldg(reinterpret_cast<const uint2*>(kv + (size_t)t * kv_stride + d));
  const half2* hv = reinterpret_cast<const half2*>(&raw);
  const float2 f0 = __half22float2(hv[0]), f1 = __half22float2(hv[1]);
  const float src[4] = {f0.x, f0.y, f1.x, f1.y};
  float sq = src[0] * src[0] + src[1] * src[1] + src[2] * src[2] +
             src[3] * src[3];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
  if (lane == 0) s_sq[warp] = sq;
  __syncthreads();
  if (tid == 0) {
    const float tot = s_sq[0] + s_sq[1] + s_sq[2] + s_sq[3];
    s_inv = rsqrtf(tot * (1.0f / 512.0f) + eps);
  }
  __syncthreads();
  const float inv = s_inv;
  const float4 kw4 = __ldg(reinterpret_cast<const float4*>(kv_weight + d));
  const float kw[4] = {kw4.x, kw4.y, kw4.z, kw4.w};
  float val[4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
    val[i] = __half2float(__float2half_rn(src[i] * inv * kw[i]));

  if (d >= 448) {
    const int first_pair = (d - 448) >> 1;
#pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
      const int idx = first_pair + pair;
      float c, s;
      if (pos < rope_len) {
        c = __ldg(table + pos * 64 + idx);
        s = __ldg(table + pos * 64 + 32 + idx);
      } else {
        const float angle =
            static_cast<float>(pos) * __ldg(table + rope_len * 64 + idx);
        sincosf(angle, &s, &c);
      }
      const float a = val[pair * 2], b = val[pair * 2 + 1];
      val[pair * 2] =
          __half2float(__float2half_rn(__fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s))));
      val[pair * 2 + 1] =
          __half2float(__float2half_rn(__fadd_rn(__fmul_rn(a, s), __fmul_rn(b, c))));
    }
  }

  float maximum = fmaxf(fmaxf(fabsf(val[0]), fabsf(val[1])),
                        fmaxf(fabsf(val[2]), fabsf(val[3])));
  constexpr int LANES_PER_GRP = WGRP / 4;
#pragma unroll
  for (int delta = LANES_PER_GRP >> 1; delta > 0; delta >>= 1)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, delta));
  const unsigned bits =
      __float_as_uint(fmaxf(maximum, 1e-4f) * (1.0f / 448.0f));
  const unsigned exponent = (bits >> 23) + ((bits & 0x7fffffu) != 0);
  const float quant_scale = __uint_as_float(exponent << 23);
  const float reciprocal = __uint_as_float((254u - exponent) << 23);
#pragma unroll
  for (int i = 0; i < 4; ++i)
    val[i] = attn_fp8_round(val[i] * reciprocal) * quant_scale;
  half2* win_out = reinterpret_cast<half2*>(window + (size_t)t * 512 + d);
  win_out[0] = __floats2half2_rn(val[0], val[1]);
  win_out[1] = __floats2half2_rn(val[2], val[3]);
}

void attn_prep_cuda(torch::Tensor q, torch::Tensor kv, torch::Tensor kv_weight,
                    torch::Tensor positions, torch::Tensor table,
                    torch::Tensor window, double eps, int64_t window_group) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kFloat16 &&
              q.dim() == 3 && q.size(2) == 512 && q.is_contiguous());
  const int T = static_cast<int>(q.size(0));
  const int heads = static_cast<int>(q.size(1));
  TORCH_CHECK(kv.is_cuda() && kv.scalar_type() == torch::kFloat16 &&
              kv.dim() == 2 && kv.size(0) == T && kv.size(1) == 512 &&
              kv.stride(1) == 1 && (kv.stride(0) & 3) == 0);
  TORCH_CHECK(kv_weight.is_cuda() && kv_weight.scalar_type() == torch::kFloat32 &&
              kv_weight.numel() == 512 && kv_weight.is_contiguous());
  TORCH_CHECK(positions.is_cuda() && positions.is_contiguous() &&
              positions.numel() == T &&
              (positions.scalar_type() == torch::kInt64 ||
               positions.scalar_type() == torch::kInt32));
  TORCH_CHECK(table.is_cuda() && table.scalar_type() == torch::kFloat32 &&
              table.dim() == 2 && table.size(1) == 64 && table.is_contiguous());
  TORCH_CHECK(window.is_cuda() && window.scalar_type() == torch::kFloat16 &&
              window.dim() == 2 && window.size(0) == T &&
              window.size(1) == 512 && window.is_contiguous());
  TORCH_CHECK(window_group == 32 || window_group == 128);
  if (T == 0) return;

  const at::cuda::OptionalCUDAGuard guard(device_of(q));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int rope_len = static_cast<int>(table.size(0) - 1);
  auto* q_ptr = reinterpret_cast<half*>(q.data_ptr<at::Half>());
  const auto* kv_ptr = reinterpret_cast<const half*>(kv.data_ptr<at::Half>());
  auto* win_ptr = reinterpret_cast<half*>(window.data_ptr<at::Half>());
  const float* kw_ptr = kv_weight.data_ptr<float>();
  const float* tb_ptr = table.data_ptr<float>();
  const int64_t kv_st = kv.stride(0);

  if (positions.scalar_type() == torch::kInt64) {
    if (window_group == 32) {
      attn_prep_cuda_kernel<int64_t, 32><<<T, 128, 0, stream>>>(
          q_ptr, kv_ptr, kv_st, kw_ptr, positions.data_ptr<int64_t>(), tb_ptr,
          win_ptr, heads, rope_len, static_cast<float>(eps));
    } else {
      attn_prep_cuda_kernel<int64_t, 128><<<T, 128, 0, stream>>>(
          q_ptr, kv_ptr, kv_st, kw_ptr, positions.data_ptr<int64_t>(), tb_ptr,
          win_ptr, heads, rope_len, static_cast<float>(eps));
    }
  } else {
    if (window_group == 32) {
      attn_prep_cuda_kernel<int32_t, 32><<<T, 128, 0, stream>>>(
          q_ptr, kv_ptr, kv_st, kw_ptr, positions.data_ptr<int32_t>(), tb_ptr,
          win_ptr, heads, rope_len, static_cast<float>(eps));
    } else {
      attn_prep_cuda_kernel<int32_t, 128><<<T, 128, 0, stream>>>(
          q_ptr, kv_ptr, kv_st, kw_ptr, positions.data_ptr<int32_t>(), tb_ptr,
          win_ptr, heads, rope_len, static_cast<float>(eps));
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename POS_T, bool DO_RING>
__global__ void __launch_bounds__(256)
    attn_post_cuda_kernel(const half* __restrict__ x,
                          const POS_T* __restrict__ positions,
                          const float* __restrict__ table,
                          half* __restrict__ out, half* __restrict__ ring,
                          const half* __restrict__ values,
                          const bool* __restrict__ valid, int heads,
                          int h_blocks, int rope_len, int ring_rows) {
  const int hb = blockIdx.x;
  const int t = blockIdx.y;
  const int64_t pos = static_cast<int64_t>(positions[t]);

  if constexpr (DO_RING) {
    if (hb == h_blocks) {
      if (valid[t] && pos >= 0) {
        const uint4* src =
            reinterpret_cast<const uint4*>(values + (size_t)t * 512);
        uint4* dst =
            reinterpret_cast<uint4*>(ring + (size_t)(pos % ring_rows) * 512);
        if (threadIdx.x < 64) dst[threadIdx.x] = src[threadIdx.x];
      }
      return;
    }
  }

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int head = hb * 4 + (warp >> 1);
  if (head >= heads) return;
  const int half_col = (warp & 1) * 256;

#pragma unroll
  for (int sub = 0; sub < 2; ++sub) {
    const int d = half_col + sub * 128 + lane * 4;
    const size_t off = ((size_t)t * heads + head) * 512 + d;
    const uint2 raw = __ldg(reinterpret_cast<const uint2*>(x + off));
    const half2* hv = reinterpret_cast<const half2*>(&raw);
    const float2 f0 = __half22float2(hv[0]), f1 = __half22float2(hv[1]);
    float value[4] = {f0.x, f0.y, f1.x, f1.y};

    if (d >= 448) {
      const int first_pair = (d - 448) >> 1;
#pragma unroll
      for (int pair = 0; pair < 2; ++pair) {
        const int idx = first_pair + pair;
        float cosine, sine;
        if (pos < rope_len) {
          cosine = __ldg(table + pos * 64 + idx);
          sine = -__ldg(table + pos * 64 + 32 + idx);
        } else {
          const float angle =
              static_cast<float>(pos) * __ldg(table + rope_len * 64 + idx);
          sincosf(angle, &sine, &cosine);
          sine = -sine;
        }
        const float a = value[pair * 2], b = value[pair * 2 + 1];
        value[pair * 2] = __half2float(
            __float2half_rn(__fsub_rn(__fmul_rn(a, cosine), __fmul_rn(b, sine))));
        value[pair * 2 + 1] = __half2float(
            __float2half_rn(__fadd_rn(__fmul_rn(a, sine), __fmul_rn(b, cosine))));
      }
    }

    float maximum = fmaxf(fmaxf(fabsf(value[0]), fabsf(value[1])),
                          fmaxf(fabsf(value[2]), fabsf(value[3])));
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1)
      maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, delta));
    const unsigned bits =
        __float_as_uint(fmaxf(maximum, 1e-4f) * (1.0f / 448.0f));
    const unsigned exponent = (bits >> 23) + ((bits & 0x7fffffu) != 0);
    const float quant_scale = __uint_as_float(exponent << 23);
    const float reciprocal = __uint_as_float((254u - exponent) << 23);
#pragma unroll
    for (int i = 0; i < 4; ++i)
      value[i] = attn_fp8_round(value[i] * reciprocal) * quant_scale;
    half2* out_h2 = reinterpret_cast<half2*>(out + off);
    out_h2[0] = __floats2half2_rn(value[0], value[1]);
    out_h2[1] = __floats2half2_rn(value[2], value[3]);
  }
}

void attn_post_cuda(torch::Tensor x, torch::Tensor positions,
                    torch::Tensor table, torch::Tensor out,
                    std::optional<torch::Tensor> ring,
                    std::optional<torch::Tensor> values,
                    std::optional<torch::Tensor> valid) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16 &&
              x.dim() == 3 && x.size(2) == 512 && x.is_contiguous());
  const int T = static_cast<int>(x.size(0));
  const int heads = static_cast<int>(x.size(1));
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat16 &&
              out.is_contiguous() && out.numel() == x.numel());
  TORCH_CHECK(positions.is_cuda() && positions.is_contiguous() &&
              positions.numel() == T &&
              (positions.scalar_type() == torch::kInt64 ||
               positions.scalar_type() == torch::kInt32));
  TORCH_CHECK(table.is_cuda() && table.scalar_type() == torch::kFloat32 &&
              table.dim() == 2 && table.size(1) == 64 && table.is_contiguous());
  if (T == 0) return;

  const bool do_ring = ring.has_value();
  half* ring_ptr = nullptr;
  const half* val_ptr = nullptr;
  const bool* vld_ptr = nullptr;
  int ring_rows = 0;
  if (do_ring) {
    TORCH_CHECK(values.has_value() && valid.has_value());
    TORCH_CHECK(ring->is_cuda() && ring->scalar_type() == torch::kFloat16 &&
                ring->dim() == 2 && ring->size(1) == 512 &&
                ring->is_contiguous());
    TORCH_CHECK(values->is_cuda() && values->scalar_type() == torch::kFloat16 &&
                values->dim() == 2 && values->size(0) == T &&
                values->size(1) == 512 && values->is_contiguous());
    TORCH_CHECK(valid->is_cuda() && valid->scalar_type() == torch::kBool &&
                valid->numel() == T && valid->is_contiguous());
    ring_ptr = reinterpret_cast<half*>(ring->data_ptr<at::Half>());
    val_ptr = reinterpret_cast<const half*>(values->data_ptr<at::Half>());
    vld_ptr = valid->data_ptr<bool>();
    ring_rows = static_cast<int>(ring->size(0));
  }

  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int h_blocks = (heads + 3) / 4;
  const dim3 grid(h_blocks + (do_ring ? 1 : 0), T);
  const int rope_len = static_cast<int>(table.size(0) - 1);
  const auto* x_ptr = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
  auto* out_ptr = reinterpret_cast<half*>(out.data_ptr<at::Half>());
  const float* tb_ptr = table.data_ptr<float>();

  if (positions.scalar_type() == torch::kInt64) {
    if (do_ring) {
      attn_post_cuda_kernel<int64_t, true><<<grid, 256, 0, stream>>>(
          x_ptr, positions.data_ptr<int64_t>(), tb_ptr, out_ptr, ring_ptr,
          val_ptr, vld_ptr, heads, h_blocks, rope_len, ring_rows);
    } else {
      attn_post_cuda_kernel<int64_t, false><<<grid, 256, 0, stream>>>(
          x_ptr, positions.data_ptr<int64_t>(), tb_ptr, out_ptr, nullptr,
          nullptr, nullptr, heads, h_blocks, rope_len, 0);
    }
  } else {
    if (do_ring) {
      attn_post_cuda_kernel<int32_t, true><<<grid, 256, 0, stream>>>(
          x_ptr, positions.data_ptr<int32_t>(), tb_ptr, out_ptr, ring_ptr,
          val_ptr, vld_ptr, heads, h_blocks, rope_len, ring_rows);
    } else {
      attn_post_cuda_kernel<int32_t, false><<<grid, 256, 0, stream>>>(
          x_ptr, positions.data_ptr<int32_t>(), tb_ptr, out_ptr, nullptr,
          nullptr, nullptr, heads, h_blocks, rope_len, 0);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---- prefill: one CTA per token over all its keys (online softmax) ----
// Same gather and math as attn_dec_split + attn_dec_combine, but each CTA
// walks every 32-key chunk of its token with a running (max, sum) per head and
// the FP32 output accumulator in shared memory: no per-chunk partials, no
// combine launch. Built for long prefill chunks (many tokens).
namespace attn {
constexpr int TKW = 8;  // warps per token CTA
constexpr size_t TK_Q = 0;
constexpr size_t TK_K = TK_Q + sizeof(half) * H * QLD;
constexpr size_t TK_S = TK_K + sizeof(half) * KC * KLD;  // float [4][H][SLD]
constexpr size_t TK_P = TK_S + sizeof(float) * 4 * H * SLD;  // half [H][PLD]
constexpr size_t TK_O = TK_P + sizeof(half) * H * PLD;       // float [H][D]
constexpr size_t TK_SMEM = TK_O + sizeof(float) * H * D;
}  // namespace attn

// Key gather source of (token, key) or nullptr when the key is not live.
__device__ __forceinline__ const half* ring_key_src(const RingGather& rg, int t,
                                                    int key, int width,
                                                    int64_t pos, bool tvalid) {
  using attn::D;
  if (key >= width || !tvalid) return nullptr;
  // Paged mode (attn_paged_tok, VLLM_V41_BATCH prefill): the step's window
  // rows are already in the paged window cache; each token reads its
  // request's row of both block tables.
  const int row = rg.paged ? rg.rows[t] : 0;
  if (key < rg.window) {
    const int64_t at = pos - rg.window + 1 + key;
    if (at < 0) return nullptr;
    if (rg.paged) {
      const int wpage =
          rg.wtable[(size_t)row * rg.wtable_stride + at / rg.wblock];
      return wpage < 0 ? nullptr : v41_window_src(rg, wpage, at % rg.wblock);
    }
    const int64_t start = rg.starts[0];
    return at >= start
               ? rg.values + (at - start) * D
               : rg.ring +
                     ((at % rg.ring_rows + rg.ring_rows) % rg.ring_rows) * D;
  }
  const int64_t index = rg.topk[(size_t)t * rg.ntopk + key - rg.window];
  const int per = rg.block / rg.ratio;
  if (index < 0 || index >= (pos + 1) / rg.ratio) return nullptr;
  const int page = rg.table[(size_t)row * rg.table_stride + index / per];
  if (page < 0) return nullptr;
  return v41_compressed_src(rg, page, index % per);
}

__global__ void __launch_bounds__(32 * attn::TKW)
    attn_tok(const half* __restrict__ q, half* __restrict__ out,
             const float* __restrict__ sink, int width, float scale,
             RingGather rg) {
  using namespace nvcuda;
  using namespace attn;
  constexpr int NT = 32 * TKW;
  constexpr int PER =
      KC * D / 8 / NT;  // 16-byte key pieces per thread per chunk
  extern __shared__ __align__(128) unsigned char dsm[];
  half* qs = reinterpret_cast<half*>(dsm + TK_Q);
  half* ks = reinterpret_cast<half*>(dsm + TK_K);
  float* sc = reinterpret_cast<float*>(dsm + TK_S);
  half* ps = reinterpret_cast<half*>(dsm + TK_P);
  float* os = reinterpret_cast<float*>(dsm + TK_O);
  __shared__ const half* srcs[2][KC];
  __shared__ unsigned live_mask[2];
  __shared__ float run_m[H], run_l[H], alpha[H];
  const int t = blockIdx.x;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int nchunk = (width + KC - 1) / KC;
  const half* qt = q + (size_t)t * H * D;
  const int64_t pos = rg.positions[t];
  const bool tvalid = rg.valid[t];
  for (int i = threadIdx.x; i < H * D / 8; i += NT) {
    const int h = i / (D / 8), c = (i % (D / 8)) * 8;
    *reinterpret_cast<uint4*>(qs + h * QLD + c) =
        *reinterpret_cast<const uint4*>(qt + h * D + c);
  }
  for (int i = threadIdx.x; i < H * D; i += NT) os[i] = 0.f;
  if (threadIdx.x < H) {
    run_m[threadIdx.x] = sink[threadIdx.x];  // the sink joins as one more logit
    run_l[threadIdx.x] = 1.f;
  }
  auto plan = [&](int chunk, int slot) {  // warp 0
    const half* src =
        ring_key_src(rg, t, chunk * KC + lane, width, pos, tvalid);
    srcs[slot][lane] = src;
    const unsigned m = __ballot_sync(0xffffffffu, src != nullptr);
    if (lane == 0) live_mask[slot] = m;
  };
  uint4 v[PER];
  auto fetch = [&](int slot) {
    const unsigned lm = live_mask[slot];
#pragma unroll
    for (int u = 0; u < PER; ++u) {
      const int i = threadIdx.x + u * NT;
      const int j = i / (D / 8), c = (i % (D / 8)) * 8;
      v[u] = (lm >> j) & 1u
                 ? v41_key8(srcs[slot][j], c)
                 : make_uint4(0, 0, 0, 0);
    }
  };
  if (warp == 0) plan(0, 0);
  __syncthreads();
  fetch(0);
  for (int chunk = 0; chunk < nchunk; ++chunk) {
    const int slot = chunk & 1;
    const unsigned lm = live_mask[slot];
    // keys of this chunk (in registers since the previous iteration) -> smem
#pragma unroll
    for (int u = 0; u < PER; ++u) {
      const int i = threadIdx.x + u * NT;
      const int j = i / (D / 8), c = (i % (D / 8)) * 8;
      *reinterpret_cast<uint4*>(ks + j * KLD + c) = v[u];
    }
    if (warp == 0 && chunk + 1 < nchunk) plan(chunk + 1, slot ^ 1);
    __syncthreads();
    if (chunk + 1 < nchunk) fetch(slot ^ 1);  // in flight during this chunk
    if (lm != 0) {                            // block-uniform
      {
        // QK: 2 key tiles x 4 D quarters, one per warp
        const int tile = warp & 1, part = warp >> 1;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.f);
#pragma unroll
        for (int d = part * (D / 4); d < (part + 1) * (D / 4); d += 16) {
          wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
          wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
          wmma::load_matrix_sync(a, qs + d, QLD);
          wmma::load_matrix_sync(b, ks + tile * 16 * KLD + d, KLD);
          wmma::mma_sync(acc, a, b, acc);
        }
        wmma::store_matrix_sync(sc + part * H * SLD + tile * 16, acc, SLD,
                                wmma::mem_row_major);
      }
      __syncthreads();
      for (int h = warp; h < H; h += TKW) {
        const bool live = (lm >> lane) & 1u;
        const int o = h * SLD + lane;
        const float raw =
            sc[o] + sc[H * SLD + o] + sc[2 * H * SLD + o] + sc[3 * H * SLD + o];
        const float sv = live ? raw * scale : -INFINITY;
        float m = sv;
#pragma unroll
        for (int x = 16; x; x >>= 1)
          m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, x));
        const float m_old = run_m[h];
        const float m_new = fmaxf(m_old, m);
        const float e = live ? __expf(sv - m_new) : 0.f;
        float l = e;
#pragma unroll
        for (int x = 16; x; x >>= 1) l += __shfl_xor_sync(0xffffffffu, l, x);
        ps[h * PLD + lane] = __float2half(e);
        if (lane == 0) {
          const float a = __expf(m_old - m_new);
          alpha[h] = a;
          run_l[h] = run_l[h] * a + l;
          run_m[h] = m_new;
        }
      }
      __syncthreads();
      // O = O * alpha + P . K; 32 output tiles, 4 per warp (rows rescaled
      // first)
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> pa0,
          pa1;
      wmma::load_matrix_sync(pa0, ps, PLD);
      wmma::load_matrix_sync(pa1, ps + 16, PLD);
#pragma unroll
      for (int tile = warp; tile < D / 16; tile += TKW) {
        for (int i = lane; i < 256; i += 32)
          os[(i >> 4) * D + tile * 16 + (i & 15)] *= alpha[i >> 4];
        __syncwarp();
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::load_matrix_sync(acc, os + tile * 16, D, wmma::mem_row_major);
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b0,
            b1;
        wmma::load_matrix_sync(b0, ks + tile * 16, KLD);
        wmma::load_matrix_sync(b1, ks + 16 * KLD + tile * 16, KLD);
        wmma::mma_sync(acc, pa0, b0, acc);
        wmma::mma_sync(acc, pa1, b1, acc);
        wmma::store_matrix_sync(os + tile * 16, acc, D, wmma::mem_row_major);
      }
    }
    __syncthreads();  // ks / ps / sc reused next chunk
  }
  half* o_tok = out + (size_t)t * H * D;
  for (int i = threadIdx.x * 2; i < H * D; i += 2 * NT) {
    const float inv = 1.f / run_l[i / D];
    *reinterpret_cast<half2*>(o_tok + i) =
        __floats2half2_rn(os[i] * inv, os[i + 1] * inv);
  }
}

void attn_ring_tok(torch::Tensor out, torch::Tensor q, torch::Tensor ring,
                   torch::Tensor values, torch::Tensor positions_all,
                   torch::Tensor cache, torch::Tensor table,
                   torch::Tensor positions, torch::Tensor valid,
                   torch::Tensor topk, torch::Tensor sink, double scale,
                   int64_t block, int64_t ratio, int64_t window,
                   int64_t ntopk) {
  using namespace attn;
  const int T = q.size(0);
  const int width = window + ntopk;
  TORCH_CHECK(q.scalar_type() == torch::kFloat16 && q.is_contiguous() &&
              q.size(1) == H && q.size(2) == D && out.is_contiguous());
  TORCH_CHECK(positions.scalar_type() == torch::kInt64 &&
              positions_all.scalar_type() == torch::kInt64);
  TORCH_CHECK(valid.scalar_type() == torch::kBool);
  TORCH_CHECK(ntopk == 0 ||
              (topk.scalar_type() == torch::kInt32 && topk.is_contiguous() &&
               table.scalar_type() == torch::kInt32));
  const at::cuda::OptionalCUDAGuard guard(device_of(q));
  auto stream = at::cuda::getCurrentCUDAStream();
  RingGather rg;
  rg.ring = reinterpret_cast<const half*>(ring.data_ptr<at::Half>());
  rg.values = reinterpret_cast<const half*>(values.data_ptr<at::Half>());
  rg.starts = positions_all.data_ptr<int64_t>();
  v41_set_compressed(rg, cache, ntopk);
  rg.table = ntopk ? table.data_ptr<int>() : nullptr;
  rg.positions = positions.data_ptr<int64_t>();
  rg.valid = valid.data_ptr<bool>();
  rg.topk = ntopk ? topk.data_ptr<int>() : nullptr;
  rg.stride = ntopk ? cache.stride(0) : 0;
  rg.block = block;
  rg.ratio = ratio;
  rg.window = window;
  rg.ntopk = ntopk;
  rg.ring_rows = ring.size(0);
  static bool smem_set = false;
  if (!smem_set) {
    cudaFuncSetAttribute(attn_tok, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)TK_SMEM);
    smem_set = true;
  }
  attn_tok<<<T, 32 * TKW, TK_SMEM, stream>>>(
      reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()), sink.data_ptr<float>(),
      width, (float)scale, rg);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// attn_ring_tok over paged window caches (VLLM_V41_BATCH prefill): one CTA per
// token gathers its window rows and top-k compressed rows straight from the
// paged caches of its request (rows[t]), no [T, width, 512] key tensor.
void attn_paged_tok(torch::Tensor out, torch::Tensor q, torch::Tensor wcache,
                    torch::Tensor wtable, torch::Tensor cache,
                    torch::Tensor table, torch::Tensor rows,
                    torch::Tensor positions, torch::Tensor valid,
                    torch::Tensor topk, torch::Tensor sink, double scale,
                    int64_t wblock, int64_t block, int64_t ratio,
                    int64_t window, int64_t ntopk) {
  using namespace attn;
  const int T = q.size(0);
  const int width = window + ntopk;
  TORCH_CHECK(q.scalar_type() == torch::kFloat16 && q.is_contiguous() &&
              q.size(1) == H && q.size(2) == D && out.is_contiguous());
  TORCH_CHECK(positions.scalar_type() == torch::kInt64 &&
              valid.scalar_type() == torch::kBool);
  TORCH_CHECK(rows.scalar_type() == torch::kInt32 && rows.size(0) >= T);
  TORCH_CHECK(wtable.scalar_type() == torch::kInt32 && wtable.stride(1) == 1);
  TORCH_CHECK(ntopk == 0 ||
              (topk.scalar_type() == torch::kInt32 && topk.is_contiguous() &&
               table.scalar_type() == torch::kInt32 && table.stride(1) == 1));
  if (T == 0) return;
  const at::cuda::OptionalCUDAGuard guard(device_of(q));
  auto stream = at::cuda::getCurrentCUDAStream();
  RingGather rg;
  rg.positions = positions.data_ptr<int64_t>();
  rg.valid = valid.data_ptr<bool>();
  v41_set_compressed(rg, cache, ntopk);
  rg.table = ntopk ? table.data_ptr<int>() : nullptr;
  rg.topk = ntopk ? topk.data_ptr<int>() : nullptr;
  rg.stride = ntopk ? cache.stride(0) : 0;
  rg.table_stride = ntopk ? table.stride(0) : 0;
  rg.block = block;
  rg.ratio = ratio;
  rg.window = window;
  rg.ntopk = ntopk;
  rg.ring_rows = 1;
  rg.paged = true;
  v41_set_window(rg, wcache, wblock);
  rg.wtable = wtable.data_ptr<int>();
  rg.rows = rows.data_ptr<int>();
  rg.wtable_stride = wtable.stride(0);
  rg.wblock = wblock;
  static bool smem_set = false;
  if (!smem_set) {
    cudaFuncSetAttribute(attn_tok, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)TK_SMEM);
    smem_set = true;
  }
  attn_tok<<<T, 32 * TKW, TK_SMEM, stream>>>(
      reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()), sink.data_ptr<float>(),
      width, (float)scale, rg);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Decode attention gathering its keys from the window ring and the paged
// compressed cache (no [T, width, 512] key tensor).
void attn_ring_dec(torch::Tensor out, torch::Tensor q, torch::Tensor ring,
                   torch::Tensor values, torch::Tensor positions_all,
                   torch::Tensor cache, torch::Tensor table,
                   torch::Tensor positions, torch::Tensor valid,
                   torch::Tensor topk, torch::Tensor sink, double scale,
                   int64_t block, int64_t ratio, int64_t window, int64_t ntopk,
                   std::optional<torch::Tensor> rope) {
  using namespace attn;
  const int T = q.size(0);
  const int width = window + ntopk;
  TORCH_CHECK(q.scalar_type() == torch::kFloat16 && q.is_contiguous() &&
              q.size(1) == H && q.size(2) == D);
  TORCH_CHECK(ring.scalar_type() == torch::kFloat16 &&
              values.scalar_type() == torch::kFloat16);
  TORCH_CHECK(positions.scalar_type() == torch::kInt64 &&
              positions_all.scalar_type() == torch::kInt64);
  TORCH_CHECK(valid.scalar_type() == torch::kBool);
  TORCH_CHECK(ntopk == 0 ||
              (topk.scalar_type() == torch::kInt32 && topk.is_contiguous() &&
               table.scalar_type() == torch::kInt32));
  const at::cuda::OptionalCUDAGuard guard(device_of(q));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int nchunk = (width + KC - 1) / KC;
  TORCH_CHECK(nchunk <= 64, "attention width > 2048 keys");
  auto opts = q.options().dtype(torch::kFloat32);
  auto opart = torch::empty({(int64_t)T * nchunk * H * D}, q.options());
  auto ml = torch::empty({2 * (int64_t)T * nchunk * H}, opts);
  float* mp = ml.data_ptr<float>();
  float* lp = mp + (int64_t)T * nchunk * H;
  RingGather rg;
  rg.ring = reinterpret_cast<const half*>(ring.data_ptr<at::Half>());
  rg.values = reinterpret_cast<const half*>(values.data_ptr<at::Half>());
  rg.starts = positions_all.data_ptr<int64_t>();
  v41_set_compressed(rg, cache, ntopk);
  rg.table = ntopk ? table.data_ptr<int>() : nullptr;
  rg.positions = positions.data_ptr<int64_t>();
  rg.valid = valid.data_ptr<bool>();
  rg.topk = ntopk ? topk.data_ptr<int>() : nullptr;
  rg.stride = ntopk ? cache.stride(0) : 0;
  rg.block = block;
  rg.ratio = ratio;
  rg.window = window;
  rg.ntopk = ntopk;
  rg.ring_rows = ring.size(0);
  static bool smem_set = false;
  if (!smem_set) {
    cudaFuncSetAttribute(attn_dec_split<true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)DEC_SMEM);
    smem_set = true;
  }
  attn_dec_split<true><<<dim3(T, nchunk), 32 * DEC_WARPS, DEC_SMEM, stream>>>(
      reinterpret_cast<const half*>(q.data_ptr<at::Half>()), nullptr, nullptr,
      reinterpret_cast<half*>(opart.data_ptr<at::Half>()), mp, lp, width,
      (float)scale, rg);
  if (rope.has_value()) {
    TORCH_CHECK(rope->is_cuda() && rope->scalar_type() == torch::kFloat32 &&
                    rope->is_contiguous() && rope->dim() == 2 &&
                    rope->size(1) == 64 && rope->size(0) > 1 &&
                    rope->device() == q.device(),
                "attn_ring_dec: RoPE table must be CUDA FP32 [length + 1, 64]");
    attn_dec_combine<true><<<dim3(T, H + 1), 128, 0, stream>>>(
        reinterpret_cast<const half*>(opart.data_ptr<at::Half>()), mp, lp,
        sink.data_ptr<float>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()), nchunk, rg,
        rope->data_ptr<float>(), rope->size(0) - 1,
        reinterpret_cast<half*>(ring.data_ptr<at::Half>()));
  } else {
    attn_dec_combine<false><<<dim3(T, H), 128, 0, stream>>>(
        reinterpret_cast<const half*>(opart.data_ptr<at::Half>()), mp, lp,
        sink.data_ptr<float>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()), nchunk, RingGather{},
        nullptr, 0, nullptr);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Batched decode attention over the paged window cache (VLLM_V41_BATCH):
// attn_ring_dec's gather + split softmax + combine, every token on its own
// request rows; with rope, the combine also does the inverse RoPE + QDQ and
// inserts the step's window rows into their pages (no separate insert).
void attn_paged_dec(torch::Tensor out, torch::Tensor q, torch::Tensor values,
                    torch::Tensor wcache, torch::Tensor wtable,
                    torch::Tensor cache, torch::Tensor table,
                    torch::Tensor rows, torch::Tensor bases,
                    torch::Tensor positions, torch::Tensor valid,
                    torch::Tensor topk, torch::Tensor sink, double scale,
                    int64_t wblock, int64_t block, int64_t ratio,
                    int64_t window, int64_t ntopk,
                    std::optional<torch::Tensor> rope) {
  torch::Tensor ring = wcache, positions_all = positions;
  using namespace attn;
  const int T = q.size(0);
  const int width = window + ntopk;
  TORCH_CHECK(q.scalar_type() == torch::kFloat16 && q.is_contiguous() &&
              q.size(1) == H && q.size(2) == D);
  TORCH_CHECK(values.scalar_type() == torch::kFloat16);
  TORCH_CHECK(positions.scalar_type() == torch::kInt64 &&
              positions_all.scalar_type() == torch::kInt64);
  TORCH_CHECK(valid.scalar_type() == torch::kBool);
  TORCH_CHECK(ntopk == 0 ||
              (topk.scalar_type() == torch::kInt32 && topk.is_contiguous() &&
               table.scalar_type() == torch::kInt32));
  const at::cuda::OptionalCUDAGuard guard(device_of(q));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int nkeys = (width + KC - 1) / KC;
  TORCH_CHECK(nkeys <= 64, "attention width > 2048 keys");
  // Key tiles per CTA (attn_dec_multi): enough CTAs to fill the 80 SMs about
  // three times over, fewer per-token partials beyond that. V41SK_ATTN_TILES
  // fixes it (1: one tile per CTA, attn_dec_split).
  static const int forced_tiles = [] {
    const char* v = std::getenv("V41SK_ATTN_TILES");
    return v ? std::max(0, std::atoi(v)) : 0;
  }();
  const int tiles =
      forced_tiles ? std::min(forced_tiles, nkeys)
                   : std::max(1, std::min(nkeys, T * nkeys / 240));
  const int nchunk = (nkeys + tiles - 1) / tiles;
  auto opts = q.options().dtype(torch::kFloat32);
  auto opart = torch::empty({(int64_t)T * nchunk * H * D}, q.options());
  auto ml = torch::empty({2 * (int64_t)T * nchunk * H}, opts);
  float* mp = ml.data_ptr<float>();
  float* lp = mp + (int64_t)T * nchunk * H;
  RingGather rg;
  rg.ring = reinterpret_cast<const half*>(ring.data_ptr());  // unused (paged)
  rg.values = reinterpret_cast<const half*>(values.data_ptr<at::Half>());
  rg.starts = positions_all.data_ptr<int64_t>();
  v41_set_compressed(rg, cache, ntopk);
  rg.table = ntopk ? table.data_ptr<int>() : nullptr;
  rg.positions = positions.data_ptr<int64_t>();
  rg.valid = valid.data_ptr<bool>();
  rg.topk = ntopk ? topk.data_ptr<int>() : nullptr;
  rg.stride = ntopk ? cache.stride(0) : 0;
  rg.block = block;
  rg.ratio = ratio;
  rg.window = window;
  rg.ntopk = ntopk;
  rg.ring_rows = 1;
  TORCH_CHECK(wtable.scalar_type() == torch::kInt32 && wtable.stride(1) == 1);
  TORCH_CHECK(rows.scalar_type() == torch::kInt32 &&
              bases.scalar_type() == torch::kInt32);
  TORCH_CHECK(values.is_contiguous() && values.size(0) >= T &&
              values.size(-1) == D);
  rg.paged = true;
  v41_set_window(rg, wcache, wblock);
  rg.wtable = wtable.data_ptr<int>();
  rg.rows = rows.data_ptr<int>();
  rg.bases = bases.data_ptr<int>();
  rg.wtable_stride = wtable.stride(0);
  rg.wblock = wblock;
  rg.table_stride = ntopk ? table.stride(0) : 0;
  static bool smem_set_paged = false;
  if (!smem_set_paged) {
    cudaFuncSetAttribute(attn_dec_split<true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)DEC_SMEM);
    cudaFuncSetAttribute(attn_dec_multi<true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)DEC_SMEM);
    smem_set_paged = true;
  }
  if (tiles == 1) {
    attn_dec_split<true><<<dim3(T, nchunk), 32 * DEC_WARPS, DEC_SMEM, stream>>>(
        reinterpret_cast<const half*>(q.data_ptr<at::Half>()), nullptr,
        nullptr, reinterpret_cast<half*>(opart.data_ptr<at::Half>()), mp, lp,
        width, (float)scale, rg);
  } else {
    attn_dec_multi<true><<<dim3(T, nchunk), 32 * DEC_WARPS, DEC_SMEM, stream>>>(
        reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<half*>(opart.data_ptr<at::Half>()), mp, lp, width,
        (float)scale, tiles, rg);
  }
  if (rope.has_value()) {
    TORCH_CHECK(rope->is_cuda() && rope->scalar_type() == torch::kFloat32 &&
                    rope->is_contiguous() && rope->dim() == 2 &&
                    rope->size(1) == 64 && rope->size(0) > 1 &&
                    rope->device() == q.device(),
                "attn_ring_dec: RoPE table must be CUDA FP32 [length + 1, 64]");
    attn_dec_combine<true><<<dim3(T, H + 1), 128, 0, stream>>>(
        reinterpret_cast<const half*>(opart.data_ptr<at::Half>()), mp, lp,
        sink.data_ptr<float>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()), nchunk, rg,
        rope->data_ptr<float>(), rope->size(0) - 1,
        reinterpret_cast<half*>(ring.data_ptr()));  // ring_output: unused (paged)
  } else {
    attn_dec_combine<false><<<dim3(T, H), 128, 0, stream>>>(
        reinterpret_cast<const half*>(opart.data_ptr<at::Half>()), mp, lp,
        sink.data_ptr<float>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()), nchunk, RingGather{},
        nullptr, 0, nullptr);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void attn_dec(torch::Tensor out, torch::Tensor q, torch::Tensor keys,
              torch::Tensor unused, torch::Tensor sink, double scale) {
  using namespace attn;
  const int T = q.size(0), width = keys.size(1);
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kFloat16 &&
              q.is_contiguous() && q.size(1) == H && q.size(2) == D);
  TORCH_CHECK(keys.scalar_type() == torch::kFloat16 && keys.is_contiguous() &&
              keys.size(0) == T && keys.size(2) == D);
  TORCH_CHECK(unused.scalar_type() == torch::kBool && unused.is_contiguous() &&
              unused.size(0) == T && unused.size(1) == width);
  TORCH_CHECK(sink.scalar_type() == torch::kFloat32 && sink.numel() == H);
  TORCH_CHECK(out.scalar_type() == torch::kFloat16 && out.is_contiguous());
  const at::cuda::OptionalCUDAGuard guard(device_of(q));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int nchunk = (width + KC - 1) / KC;
  TORCH_CHECK(nchunk <= 64, "attention width > 2048 keys");
  auto opts = q.options().dtype(torch::kFloat32);
  auto opart = torch::empty({(int64_t)T * nchunk * H * D}, q.options());
  auto ml = torch::empty({2 * (int64_t)T * nchunk * H}, opts);
  float* mp = ml.data_ptr<float>();
  float* lp = mp + (int64_t)T * nchunk * H;
  static bool smem_set = false;
  if (!smem_set) {
    cudaFuncSetAttribute(attn_dec_split<false>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)DEC_SMEM);
    smem_set = true;
  }
  attn_dec_split<false><<<dim3(T, nchunk), 32 * DEC_WARPS, DEC_SMEM, stream>>>(
      reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(keys.data_ptr<at::Half>()),
      unused.data_ptr<bool>(),
      reinterpret_cast<half*>(opart.data_ptr<at::Half>()), mp, lp, width,
      (float)scale, RingGather{});
  attn_dec_combine<false><<<dim3(T, H), 128, 0, stream>>>(
      reinterpret_cast<const half*>(opart.data_ptr<at::Half>()), mp, lp,
      sink.data_ptr<float>(), reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      nchunk, RingGather{}, nullptr, 0, nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---- V4.1 router + slot plan in one launch (decode, T * K < 64) ----
// Same selection and FP32 values as native.py's _route_i32 (Triton): exp is
// ex2.approx of x * log2(e), log is libdevice logf, sqrt is sqrt.approx.ftz,
// the K-sum follows Triton's xor-4/2/1 butterfly, the division is div.full.
// Then warp 0 builds _plan_kernel's stable expert sort and groups.
__device__ __forceinline__ float rp_ex2(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float rp_sqrt(float x) {
  float y;
  asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float rp_div(float a, float b) {
  float y;
  asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}

__device__ __forceinline__ float route_score(float g) {
  const float soft =
      g > 20.0f ? g : logf(1.0f + rp_ex2(fminf(g, 20.0f) * 1.44269502f));
  return rp_sqrt(soft);
}

// Group by the first occurrence of each expert. Slot order within an
// expert stays stable. Existing callers retain sorted groups by default.
__device__ __forceinline__ void route_match_groups(const int* slot_ids,
                                                   int* perm, int* experts,
                                                   int* offsets, int slots) {
  const unsigned full = 0xffffffffu;
  const int lane = threadIdx.x & 31;
  const bool live0 = lane < slots, live1 = lane + 32 < slots;
  const int expert0 = live0 ? slot_ids[lane] : 0x7fffffff;
  const int expert1 = live1 ? slot_ids[lane + 32] : 0x7fffffff;
  const unsigned active0 = __ballot_sync(full, live0);
  const unsigned active1 = __ballot_sync(full, live1);
  const unsigned matches0 = __match_any_sync(full, expert0) & active0;
  const unsigned matches1 = __match_any_sync(full, expert1) & active1;
  const int leader0 = max(__ffs(matches0) - 1, 0);
  const int leader1 = max(__ffs(matches1) - 1, 0);
  const bool first0 = live0 && lane == leader0;
  const unsigned leaders0 = __ballot_sync(full, first0);
  unsigned tail_base = 0;
  int count0 = __popc(matches0);
#pragma unroll 1
  for (int tail = 0; tail < slots - 32; ++tail) {
    const int expert = __shfl_sync(full, expert1, tail);
    const unsigned matches = __ballot_sync(full, live0 && expert0 == expert);
    if (lane == tail) tail_base = matches;
    count0 += live0 && expert0 == expert;
  }
  const bool first1 = live1 && lane == leader1 && tail_base == 0;
  const unsigned leaders1 = __ballot_sync(full, first1);
  const int count1 = __popc(matches1);
  int prefix0 = first0 ? count0 : 0;
  int prefix1 = first1 ? count1 : 0;
#pragma unroll
  for (int step = 1; step < 32; step <<= 1) {
    const int before0 = __shfl_up_sync(full, prefix0, step);
    const int before1 = __shfl_up_sync(full, prefix1, step);
    if (lane >= step) {
      prefix0 += before0;
      prefix1 += before1;
    }
  }
  const int offset0 = __shfl_sync(full, prefix0, leader0) - count0;
  const int base_slots = __shfl_sync(full, prefix0, 31);
  const int offset1 = base_slots + __shfl_sync(full, prefix1, leader1) - count1;
  const int base_leader = max(__ffs(tail_base) - 1, 0);
  const int tail_base_offset = __shfl_sync(full, offset0, base_leader);
  const unsigned preceding = (1u << lane) - 1;
  const int groups0 = __popc(leaders0);
  const int groups = groups0 + __popc(leaders1);
  if (live0) perm[offset0 + __popc(matches0 & preceding)] = lane;
  if (live1) {
    const int start =
        tail_base ? tail_base_offset + __popc(tail_base) : offset1;
    perm[start + __popc(matches1 & preceding)] = lane + 32;
  }
  if (first0) {
    const int group = __popc(leaders0 & preceding);
    experts[group] = expert0;
    offsets[group] = offset0;
  }
  if (first1) {
    const int group = groups0 + __popc(leaders1 & preceding);
    experts[group] = expert1;
    offsets[group] = offset1;
  }
  for (int i = lane; i <= slots; i += 32) {
    if (i >= groups && i < slots) experts[i] = 0;
    if (i >= groups) offsets[i] = slots;
  }
}

// Shared by route_plan_kernel and gate_route_kernel's last CTA. Every thread
// of the block calls it (it has a block barrier); COHERENT reads the logits
// with ld.cg because other CTAs of the same launch wrote them.
template <int EPL, bool COHERENT, bool PREPARED = false,
          bool MATCH_GROUPS = false>
__device__ __forceinline__ void route_plan_body(
    const float* __restrict__ logits, const float* __restrict__ bias,
    float* __restrict__ probs, int* __restrict__ ids, int* __restrict__ perm,
    int* __restrict__ experts, int* __restrict__ offsets, int T, int K,
    float scale) {
  constexpr int E = EPL * 32;
  __shared__ int slot_ids[64];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const unsigned full = 0xffffffffu;
  // Scores and keys live in shared memory so every loop below can stay
  // rolled: this body runs once per layer right after the MoE weights flushed
  // L2, so its code is fetched cold from DRAM (unrolled it was 3.8K
  // instructions and the route cost ~25 us cold; the arithmetic is the same).
  __shared__ float sh_score[8][E];
  __shared__ unsigned sh_key[8][E];
  if (warp < T) {
    float* sc = sh_score[warp];
    unsigned* ky = sh_key[warp];
#pragma unroll
    for (int i = 0; i < EPL; ++i) {  // loads only: all in flight at once
      const int e = lane + 32 * i;
      sc[e] = COHERENT ? __ldcg(logits + warp * E + e) : logits[warp * E + e];
      ky[e] = __float_as_uint(bias[e]);
    }
#pragma unroll 1
    for (int i = 0; i < EPL; ++i) {
      const int e = lane + 32 * i;
      const float g = sc[e];
      const float score = PREPARED ? g : route_score(g);
      sc[e] = score;
      unsigned key = __float_as_uint(score + __uint_as_float(ky[e]));
      // Encode once before all K selections.
      ky[e] = (key & 0x80000000u) ? ~key : (key | 0x80000000u);
    }
    __syncwarp();
    // Lane j keeps the j-th pick.
    float my_v = 0.0f;
    int my_e = 0;
#pragma unroll 1
    for (int j = 0; j < K; ++j) {
      // Max of (key, -e) packed in 64 bits: the highest key, then the
      // lowest expert on ties (Triton's min index of the max).
      unsigned long long best = 0ull;
#pragma unroll 4
      for (int i = 0; i < EPL; ++i) {
        const unsigned u = ky[lane + 32 * i];
        const unsigned long long packed =
            ((unsigned long long)u << 32) | (0xffffffffu - (lane + 32 * i));
        best = packed > best ? packed : best;
      }
#pragma unroll
      for (int o = 16; o; o >>= 1) {
        const unsigned long long other = __shfl_xor_sync(full, best, o);
        best = other > best ? other : best;
      }
      const int best_e =
          (int)(0xffffffffu - (unsigned int)(best & 0xffffffffu));
      if (lane == j) {
        my_v = sc[best_e];
        my_e = best_e;
      }
      if (lane == (best_e & 31)) ky[best_e] = 0x007fffffu;  // ordered -infinity
      __syncwarp();
    }
    // ((v0 + v4) + (v2 + v6)) + ((v1 + v5) + (v3 + v7)): the xor-4/2/1
    // butterfly over lanes 0..7 (picks past K are 0), as before.
    float total = my_v + __shfl_xor_sync(full, my_v, 4);
    total += __shfl_xor_sync(full, total, 2);
    total += __shfl_xor_sync(full, total, 1);
    const float denom = __shfl_sync(full, total, 0) + 1e-20f;
    if (lane < K) {
      probs[warp * K + lane] = rp_div(my_v, denom) * scale;
      ids[warp * K + lane] = my_e;
      slot_ids[warp * K + lane] = my_e;
    }
  }
  __syncthreads();
  if (warp != 0) return;
  const int S = T * K;
  if constexpr (MATCH_GROUPS) {
    route_match_groups(slot_ids, perm, experts, offsets, S);
    return;
  }
  const int i0 = lane, i1 = lane + 32;
  // Sort (expert, original slot) pairs in registers. The slot bits preserve
  // the stable order required by the grouped expert kernels.
  int key0 = i0 < S ? (slot_ids[i0] << 6) | i0 : 0x7fffffff;
  int key1 = i1 < S ? (slot_ids[i1] << 6) | i1 : 0x7fffffff;
#pragma unroll 1
  for (int width = 2; width <= (1 << (32 - __clz(S - 1))); width <<= 1) {
#pragma unroll 1
    for (int stride = width >> 1; stride; stride >>= 1) {
      if (stride == 32) {
        const int low = min(key0, key1), high = max(key0, key1);
        key0 = low;
        key1 = high;
      } else {
        const int peer0 = __shfl_xor_sync(full, key0, stride);
        const int peer1 = __shfl_xor_sync(full, key1, stride);
        const bool low0 = ((i0 & stride) == 0) == ((i0 & width) == 0);
        const bool low1 = ((i1 & stride) == 0) == ((i1 & width) == 0);
        key0 = low0 ? min(key0, peer0) : max(key0, peer0);
        key1 = low1 ? min(key1, peer1) : max(key1, peer1);
      }
    }
  }
  const int expert0 = key0 >> 6, expert1 = key1 >> 6;
  const int previous0 = __shfl_up_sync(full, expert0, 1);
  const int previous1 = __shfl_up_sync(full, expert1, 1);
  const int last0 = __shfl_sync(full, expert0, 31);
  const bool live0 = i0 < S, live1 = i1 < S;
  const bool first0 = live0 && (lane == 0 || expert0 != previous0);
  const bool first1 = live1 && expert1 != (lane == 0 ? last0 : previous1);
  const unsigned flags0 = __ballot_sync(full, first0);
  const unsigned flags1 = __ballot_sync(full, first1);
  const unsigned preceding = (1u << lane) - 1;
  const int groups0 = __popc(flags0);
  const int groups = groups0 + __popc(flags1);
  if (live0) perm[i0] = key0 & 63;
  if (live1) perm[i1] = key1 & 63;
  if (first0) {
    const int group = __popc(flags0 & preceding);
    experts[group] = expert0;
    offsets[group] = i0;
  }
  if (first1) {
    const int group = groups0 + __popc(flags1 & preceding);
    experts[group] = expert1;
    offsets[group] = i1;
  }
  for (int j = lane; j <= S; j += 32) {
    if (j >= groups && j < S) experts[j] = 0;
    if (j >= groups) offsets[j] = S;
  }
}

template <int EPL, bool MATCH_GROUPS = false>
__global__ void __launch_bounds__(256)
    route_plan_kernel(const float* __restrict__ logits,
                      const float* __restrict__ bias, float* __restrict__ probs,
                      int* __restrict__ ids, int* __restrict__ perm,
                      int* __restrict__ experts, int* __restrict__ offsets,
                      int T, int K, float scale) {
  route_plan_body<EPL, false, false, MATCH_GROUPS>(
      logits, bias, probs, ids, perm, experts, offsets, T, K, scale);
}

void route_plan(torch::Tensor logits, torch::Tensor bias, torch::Tensor probs,
                torch::Tensor ids, torch::Tensor perm, torch::Tensor experts,
                torch::Tensor offsets, double scale, bool match_groups) {
  const int T = logits.size(0), E = logits.size(1), K = ids.size(1);
  TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == torch::kFloat32 &&
              logits.is_contiguous() && bias.scalar_type() == torch::kFloat32 &&
              bias.numel() == E && probs.scalar_type() == torch::kFloat32 &&
              ids.scalar_type() == torch::kInt32 &&
              perm.scalar_type() == torch::kInt32 &&
              experts.scalar_type() == torch::kInt32 &&
              offsets.scalar_type() == torch::kInt32);
  TORCH_CHECK(T >= 1 && T <= 8 && K >= 1 && K <= 8 && T * K < 64,
              "route_plan: T <= 8, K <= 8, T * K < 64");
  const at::cuda::OptionalCUDAGuard guard(device_of(logits));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#define V41SK_RP(EPL, MATCH)                                              \
  route_plan_kernel<EPL, MATCH><<<1, 32 * T, 0, stream>>>(                \
      logits.data_ptr<float>(), bias.data_ptr<float>(),                   \
      probs.data_ptr<float>(), ids.data_ptr<int>(), perm.data_ptr<int>(), \
      experts.data_ptr<int>(), offsets.data_ptr<int>(), T, K, (float)scale)
  if (E == 384) {
    if (match_groups) {
      V41SK_RP(12, true);
    } else {
      V41SK_RP(12, false);
    }
  } else if (E == 128) {
    if (match_groups) {
      V41SK_RP(4, true);
    } else {
      V41SK_RP(4, false);
    }
  } else if (E == 256) {
    if (match_groups) {
      V41SK_RP(8, true);
    } else {
      V41SK_RP(8, false);
    }
  } else {
    TORCH_CHECK(false, "route_plan: E in {128, 256, 384}");
  }
#undef V41SK_RP
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
#define H16_MMA(C, A0, A1, B0, B1)                                  \
  asm volatile(                                                     \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "            \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "             \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]), \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                          \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))
// ---- FP16-weight HMMA GEMV for M <= 8 (gate, lm_head) ----
// Weights prepacked per (32-column tile, 16-K group, lane): 16 consecutive
// K values of the lane's QPN column (the FP8 QPN8 order after decode), so a
// lane loads 32 bytes per group straight into the m8n8k4 B operands. grid.y
// CTAs share a tile along K (W warps each); per tile the last CTA (ticket)
// sums the FP32 partials in CTA order. ROUTE (gate): when every tile is done,
// the last tile's CTA routes (top-k + slot plan) in the same launch.
template <int W, int NACC, bool OUT_F32, bool ROUTE, int EPL,
          bool MATCH_GROUPS = false>
__global__ void __launch_bounds__(W * 32)
    h16_gemv_kernel(const uint4* __restrict__ packed,
                    const half* __restrict__ x, void* __restrict__ out,
                    float* __restrict__ workspace, int* __restrict__ tickets,
                    int n, int k, int m, const float* __restrict__ bias,
                    float* __restrict__ probs, int* __restrict__ ids,
                    int* __restrict__ perm, int* __restrict__ experts,
                    int* __restrict__ offsets, int topk, float scale,
                    float oscale) {
  __shared__ float partials[W][256];
  __shared__ bool last;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int tile = blockIdx.x, part = blockIdx.y, parts = gridDim.y;
  const int tiles = gridDim.x;
  const int quadpair = (lane >> 2) & 3;
  const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int G = k >> 4;
  const int gpw = G / (W * parts);
  const int g0 = (part * W + warp) * gpw;
  const uint4* wp = packed + ((size_t)tile * G * 32 + lane) * 2;
  const half* xr = x + (size_t)row * k;
  const bool live = row < m;
  float acc[NACC][8];
#pragma unroll
  for (int a = 0; a < NACC; a++)
#pragma unroll
    for (int i = 0; i < 8; i++) acc[a][i] = 0.f;
#pragma unroll 4
  for (int g = g0; g < g0 + gpw; ++g) {
    const uint4 w0 = __ldcs(wp + (size_t)g * 64);
    const uint4 w1 = __ldcs(wp + (size_t)g * 64 + 1);
    uint4 a01 = make_uint4(0, 0, 0, 0), a23 = make_uint4(0, 0, 0, 0);
    if (live) {
      a01 = *reinterpret_cast<const uint4*>(xr + g * 16);
      a23 = *reinterpret_cast<const uint4*>(xr + g * 16 + 8);
    }
    const unsigned* A0 = reinterpret_cast<const unsigned*>(&a01);
    const unsigned* A1 = reinterpret_cast<const unsigned*>(&a23);
    const unsigned* B0 = reinterpret_cast<const unsigned*>(&w0);
    const unsigned* B1 = reinterpret_cast<const unsigned*>(&w1);
    H16_MMA(acc[0], A0[0], A0[1], B0[0], B0[1]);
    H16_MMA(acc[1 % NACC], A0[2], A0[3], B0[2], B0[3]);
    H16_MMA(acc[2 % NACC], A1[0], A1[1], B1[0], B1[1]);
    H16_MMA(acc[3 % NACC], A1[2], A1[3], B1[2], B1[3]);
  }
#pragma unroll
  for (int a = 1; a < NACC; a++)
#pragma unroll
    for (int i = 0; i < 8; i++) acc[0][i] += acc[a][i];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const int orow = (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
    const int ocol = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
    partials[warp][orow * 32 + quadpair * 8 + ocol] = acc[0][i];
  }
  __syncthreads();
  const int live_n = m * 32;
  auto store = [&](int element, float v) {
    const size_t o = (size_t)(element >> 5) * n + tile * 32 + (element & 31);
    // oscale: the exact 2^-s of weights stored as FP16 * 2^s (gate)
    if constexpr (OUT_F32) {
      const float value = v * oscale;
      reinterpret_cast<float*>(out)[o] = value;
      // Compute the independent softplus/sqrt scores across every gate tile
      // before the final ticket. The routing CTA consumes the same FP32 values.
      if constexpr (ROUTE)
        workspace[(size_t)tiles * parts * 256 + o] = route_score(value);
    } else {
      reinterpret_cast<half*>(out)[o] = __float2half(v * oscale);
    }
  };
  if (parts == 1) {
    for (int e = threadIdx.x; e < live_n; e += blockDim.x) {
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < W; ++w) v += partials[w][e];
      store(e, v);
    }
  } else {
    float* mine = workspace + ((size_t)tile * parts + part) * 256;
    for (int e = threadIdx.x; e < live_n; e += blockDim.x) {
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < W; ++w) v += partials[w][e];
      mine[e] = v;
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(tickets + tile, 1) == parts - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    const float* all = workspace + (size_t)tile * parts * 256;
    for (int e = threadIdx.x; e < live_n; e += blockDim.x) {
      float v = 0.f;
      for (int p = 0; p < parts; ++p) v += __ldcg(all + p * 256 + e);
      store(e, v);
    }
    if (threadIdx.x == 0) tickets[tile] = 0;
  }
  if constexpr (ROUTE) {
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(tickets + tiles, 1) == tiles - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    route_plan_body<EPL, true, true, MATCH_GROUPS>(
        workspace + (size_t)tiles * parts * 256, bias, probs, ids, perm,
        experts, offsets, m, topk, scale);
    if (threadIdx.x == 0) tickets[tiles] = 0;
  }
}

// out[M, N] (FP32 or FP16) = x[M, K] @ W^T, W prepacked by h16 order.
// tickets: zeroed int32 [N/32 + 1], re-armed by every launch.
void h16_gemv(torch::Tensor out, torch::Tensor x, torch::Tensor packed,
              torch::Tensor tickets, int64_t warps, int64_t parts,
              double oscale) {
  const int m = x.size(0), k = x.size(1), n = out.size(1);
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16 &&
              x.is_contiguous());
  TORCH_CHECK(m >= 1 && m <= 8 && out.size(0) == m && out.is_contiguous());
  TORCH_CHECK(n % 32 == 0 && k % 16 == 0 && packed.numel() == (int64_t)n * k &&
              packed.scalar_type() == torch::kFloat16);
  TORCH_CHECK((k / 16) % (warps * parts) == 0,
              "h16_gemv: K/16 % (warps*parts)");
  TORCH_CHECK(tickets.numel() >= n / 32 + 1);
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream();
  auto ws = torch::empty({parts > 1 ? (int64_t)(n / 32) * parts * 256 : 1},
                         x.options().dtype(torch::kFloat32));
  const dim3 grid(n / 32, parts);
  const bool f32 = out.scalar_type() == torch::kFloat32;
  const auto* pk = reinterpret_cast<const uint4*>(packed.data_ptr<at::Half>());
  const auto* xp = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
#define H16(Wv, F32v)                                                        \
  h16_gemv_kernel<Wv, 2, F32v, false, 12><<<grid, Wv * 32, 0, stream>>>(     \
      pk, xp, out.data_ptr(), ws.data_ptr<float>(), tickets.data_ptr<int>(), \
      n, k, m, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0.f, \
      (float)oscale)
  if (warps == 4) {
    if (f32)
      H16(4, true);
    else
      H16(4, false);
  } else if (warps == 8) {
    if (f32)
      H16(8, true);
    else
      H16(8, false);
  } else if (warps == 16) {
    if (f32)
      H16(16, true);
    else
      H16(16, false);
  } else
    TORCH_CHECK(false, "h16_gemv: warps in {4, 8, 16}");
#undef H16
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Router gate + top-k + slot plan in one launch: logits[M, E] FP32 (also an
// output), then route_plan's probs / ids / perm / experts / offsets.
void gate_route(torch::Tensor logits, torch::Tensor x, torch::Tensor packed,
                torch::Tensor tickets, torch::Tensor bias, torch::Tensor probs,
                torch::Tensor ids, torch::Tensor perm, torch::Tensor experts,
                torch::Tensor offsets, int64_t parts, double scale,
                double oscale, bool match_groups) {
  const int m = x.size(0), k = x.size(1), n = logits.size(1), K = ids.size(1);
  TORCH_CHECK(x.scalar_type() == torch::kFloat16 && x.is_contiguous());
  TORCH_CHECK(logits.scalar_type() == torch::kFloat32 && logits.size(0) == m);
  TORCH_CHECK(m >= 1 && m <= 8 && K <= 8 && m * K < 64 &&
              (n == 384 || n == 128));
  TORCH_CHECK((k / 16) % (8 * parts) == 0, "gate_route: K/16 % (8*parts)");
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream();
  auto ws = torch::empty({(int64_t)(n / 32) * parts * 256 + (int64_t)m * n},
                         x.options().dtype(torch::kFloat32));
  const dim3 grid(n / 32, parts);
  const auto* pk = reinterpret_cast<const uint4*>(packed.data_ptr<at::Half>());
  const auto* xp = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
#define GR(EPLv, MATCH)                                                     \
  h16_gemv_kernel<8, 2, true, true, EPLv, MATCH><<<grid, 256, 0, stream>>>( \
      pk, xp, logits.data_ptr(), ws.data_ptr<float>(),                      \
      tickets.data_ptr<int>(), n, k, m, bias.data_ptr<float>(),             \
      probs.data_ptr<float>(), ids.data_ptr<int>(), perm.data_ptr<int>(),   \
      experts.data_ptr<int>(), offsets.data_ptr<int>(), K, (float)scale,    \
      (float)oscale)
  if (n == 384) {
    if (match_groups) {
      GR(12, true);
    } else {
      GR(12, false);
    }
  } else {
    if (match_groups) {
      GR(4, true);
    } else {
      GR(4, false);
    }
  }
#undef GR
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#undef H16_MMA

// ---- TP4 one-shot push all-reduce (low-latency, no barriers) ----------------
// Every rank stores its 16-byte packs straight into each peer's receive slot
// [parity][src rank]; the slot memory holds an all-ones sentinel (a NaN) while
// empty, so a reader spins on the data itself and needs no flag round trip.
// After summing it re-arms its own slots. Two parities alternate per call: a
// peer can run at most one call ahead (it needs this rank's packs to finish),
// so it never writes a slot this rank is still reading. Every call launches
// one block per counter entry (LL_BLOCKS or more, the same for every call on
// one set of slots), so all per-block counters advance together. The sum runs
// in rank order 0..3 on every rank: all ranks get bitwise-identical results.
constexpr int LL_BLOCKS = 16, LL_MAX_BLOCKS = 80, LL_THREADS = 512;
struct LLPeers {
  char* p[4];
};

__device__ __forceinline__ uint4 ll_load(const void* a) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(a));
  return v;
}
__device__ __forceinline__ void ll_store(void* a, uint4 v) {
  asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(a),
               "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
}
template <bool HALF>
__device__ __forceinline__ uint32_t ll_escape(uint32_t w) {
  if (HALF) {
    if ((w & 0xFFFFu) == 0xFFFFu) w = (w & 0xFFFF0000u) | 0x7E00u;
    if ((w >> 16) == 0xFFFFu) w = (w & 0xFFFFu) | 0x7E000000u;
    return w;
  }
  return w == 0xFFFFFFFFu ? 0x7FC00000u : w;
}
template <bool HALF>
__device__ __forceinline__ bool ll_ready(uint32_t w) {
  return HALF ? ((w & 0xFFFFu) != 0xFFFFu && (w >> 16) != 0xFFFFu)
              : w != 0xFFFFFFFFu;
}
template <bool HALF>
__device__ __forceinline__ bool ll_ready4(const uint4& v) {
  return ll_ready<HALF>(v.x) && ll_ready<HALF>(v.y) && ll_ready<HALF>(v.z) &&
         ll_ready<HALF>(v.w);
}
template <bool HALF>
__device__ __forceinline__ void ll_add(float (&acc)[8], const uint4& v) {
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
  if (HALF) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&w[i]));
      acc[2 * i] += f.x;
      acc[2 * i + 1] += f.y;
    }
  } else {
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[i] += __uint_as_float(w[i]);
  }
}

template <bool HALF>
__global__ void __launch_bounds__(LL_THREADS)
    ll_allreduce_kernel(uint4* __restrict__ out, const uint4* __restrict__ in,
                        LLPeers peers, int rank, int64_t npack, int64_t cap,
                        int* __restrict__ counter) {
  int parity = counter[blockIdx.x] & 1;
  const int stride = gridDim.x * blockDim.x;
  const int first = blockIdx.x * blockDim.x + threadIdx.x;
  const int b0 = blockIdx.x * blockDim.x;
  const int total_cap_packs = (int)(cap / 16);
  const int max_packs = total_cap_packs - LL_MAX_BLOCKS;
  const int sync_pack = total_cap_packs - 1 - blockIdx.x;
  const uint4 empty = make_uint4(~0u, ~0u, ~0u, ~0u);
  const uint4 zero4 = make_uint4(0u, 0u, 0u, 0u);
  const int nchunks = (int)((npack + max_packs - 1) / max_packs);

  for (int64_t off = 0; off < npack; off += max_packs) {
    const int cur =
        (npack - off) < max_packs ? (int)(npack - off) : max_packs;
    const int64_t base = (int64_t)parity * 4 * cap;
    for (int p = first; p < cur; p += stride) {
      uint4 v = in[off + p];
      v.x = ll_escape<HALF>(v.x);
      v.y = ll_escape<HALF>(v.y);
      v.z = ll_escape<HALF>(v.z);
      v.w = ll_escape<HALF>(v.w);
#pragma unroll
      for (int r = 1; r < 4; ++r)
        ll_store(peers.p[(rank + r) & 3] + base + rank * cap + (int64_t)p * 16,
                 v);
    }
    if (cur <= b0 && threadIdx.x == 0) {
#pragma unroll
      for (int r = 1; r < 4; ++r)
        ll_store(peers.p[(rank + r) & 3] + base + rank * cap +
                     (int64_t)sync_pack * 16,
                 zero4);
    }
    char* mine = peers.p[rank] + base;
    for (int p = first; p < cur; p += stride) {
      uint4 v[4];
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        if (r == rank) {
          v[r] = in[off + p];
          v[r].x = ll_escape<HALF>(v[r].x);
          v[r].y = ll_escape<HALF>(v[r].y);
          v[r].z = ll_escape<HALF>(v[r].z);
          v[r].w = ll_escape<HALF>(v[r].w);
        } else {
          const char* a = mine + r * cap + (int64_t)p * 16;
          do {
            v[r] = ll_load(a);
          } while (!ll_ready4<HALF>(v[r]));
        }
      }
#pragma unroll
      for (int r = 0; r < 4; ++r)
        if (r != rank) ll_store(mine + r * cap + (int64_t)p * 16, empty);
      float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int r = 0; r < 4; ++r) ll_add<HALF>(acc, v[r]);
      uint4 o;
      if (HALF) {
        __half2 h[4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
          h[i] = __floats2half2_rn(acc[2 * i], acc[2 * i + 1]);
        o = *reinterpret_cast<uint4*>(h);
      } else {
        o = make_uint4(__float_as_uint(acc[0]), __float_as_uint(acc[1]),
                       __float_as_uint(acc[2]), __float_as_uint(acc[3]));
      }
      out[off + p] = o;
    }
    if (cur <= b0 && threadIdx.x == 0) {
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        if (r == rank) continue;
        char* a = mine + r * cap + (int64_t)sync_pack * 16;
        uint4 v;
        do {
          v = ll_load(a);
        } while (!ll_ready4<HALF>(v));
        ll_store(a, empty);
      }
    }
    if (nchunks > 1) {
      __threadfence_system();
      __syncthreads();
    }
    parity ^= 1;
  }
  __syncthreads();
  if (threadIdx.x == 0) counter[blockIdx.x] += nchunks;
}

// All-gather along the last dim with the same slots and parity: rank r's
// [rows, n] lands in out[:, r*n:(r+1)*n]. row_packs = n * esz / 16.
template <bool HALF>
__global__ void __launch_bounds__(LL_THREADS)
    ll_allgather_kernel(uint4* __restrict__ out, const uint4* __restrict__ in,
                        LLPeers peers, int rank, int64_t npack,
                        int64_t row_packs, int64_t cap,
                        int* __restrict__ counter) {
  int parity = counter[blockIdx.x] & 1;
  const int stride = gridDim.x * blockDim.x;
  const int first = blockIdx.x * blockDim.x + threadIdx.x;
  const int b0 = blockIdx.x * blockDim.x;
  const int total_cap_packs = (int)(cap / 16);
  const int max_packs = total_cap_packs - LL_MAX_BLOCKS;
  const int sync_pack = total_cap_packs - 1 - blockIdx.x;
  const uint4 empty = make_uint4(~0u, ~0u, ~0u, ~0u);
  const uint4 zero4 = make_uint4(0u, 0u, 0u, 0u);
  const int nchunks = (int)((npack + max_packs - 1) / max_packs);

  for (int64_t off = 0; off < npack; off += max_packs) {
    const int cur =
        (npack - off) < max_packs ? (int)(npack - off) : max_packs;
    const int64_t base = (int64_t)parity * 4 * cap;
    for (int p = first; p < cur; p += stride) {
      const int64_t gp = off + p;
      const uint4 raw = in[gp];
      uint4 v = raw;
      v.x = ll_escape<HALF>(v.x);
      v.y = ll_escape<HALF>(v.y);
      v.z = ll_escape<HALF>(v.z);
      v.w = ll_escape<HALF>(v.w);
#pragma unroll
      for (int r = 1; r < 4; ++r)
        ll_store(peers.p[(rank + r) & 3] + base + rank * cap + (int64_t)p * 16,
                 v);
      out[(gp / row_packs) * 4 * row_packs + rank * row_packs +
          gp % row_packs] = raw;
    }
    if (cur <= b0 && threadIdx.x == 0) {
#pragma unroll
      for (int r = 1; r < 4; ++r)
        ll_store(peers.p[(rank + r) & 3] + base + rank * cap +
                     (int64_t)sync_pack * 16,
                 zero4);
    }
    char* mine = peers.p[rank] + base;
    for (int p = first; p < cur; p += stride) {
      const int64_t gp = off + p;
      const int64_t row = gp / row_packs, j = gp % row_packs;
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        if (r == rank) continue;
        char* a = mine + r * cap + (int64_t)p * 16;
        uint4 v;
        do {
          v = ll_load(a);
        } while (!ll_ready4<HALF>(v));
        ll_store(a, empty);
        out[row * 4 * row_packs + r * row_packs + j] = v;
      }
    }
    if (cur <= b0 && threadIdx.x == 0) {
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        if (r == rank) continue;
        char* a = mine + r * cap + (int64_t)sync_pack * 16;
        uint4 v;
        do {
          v = ll_load(a);
        } while (!ll_ready4<HALF>(v));
        ll_store(a, empty);
      }
    }
    if (nchunks > 1) {
      __threadfence_system();
      __syncthreads();
    }
    parity ^= 1;
  }
  __syncthreads();
  if (threadIdx.x == 0) counter[blockIdx.x] += nchunks;
}

// Variable-row dim-0 all-gather over 4 ranks (uneven prefill/decode batches or
// equal-row TP4-SP allgather): rank r holds inp [s_r, ...] FP16/FP32;
// out [s0+s1+s2+s3, ...] receives rank r's rows at row offset sum(s[:r]).
template <bool HALF>
__global__ void __launch_bounds__(LL_THREADS)
    ll_dp_allgather_var_kernel(uint4* __restrict__ out,
                               const uint4* __restrict__ in, LLPeers peers,
                               int rank, int s0, int s1, int s2, int s3,
                               int64_t row_packs, int64_t cap,
                               int* __restrict__ counter) {
  const int64_t s[4] = {s0, s1, s2, s3};
  const int64_t np[4] = {s[0] * row_packs, s[1] * row_packs, s[2] * row_packs,
                         s[3] * row_packs};
  const int64_t ooff[4] = {0, np[0], np[0] + np[1], np[0] + np[1] + np[2]};
  const int64_t max_np = max(max(np[0], np[1]), max(np[2], np[3]));
  const int64_t min_np = min(min(np[0], np[1]), min(np[2], np[3]));
  if (max_np == 0) return;

  int parity = counter[blockIdx.x] & 1;
  const int stride = gridDim.x * blockDim.x;
  const int first = blockIdx.x * blockDim.x + threadIdx.x;
  const int b0 = blockIdx.x * blockDim.x;
  const int total_cap_packs = (int)(cap / 16);
  const int max_packs = total_cap_packs - LL_MAX_BLOCKS;
  const int sync_pack = total_cap_packs - 1 - blockIdx.x;
  const uint4 empty = make_uint4(~0u, ~0u, ~0u, ~0u);
  const uint4 zero4 = make_uint4(0u, 0u, 0u, 0u);
  const int nchunks = (int)((max_np + max_packs - 1) / max_packs);
  const bool need_sync = (nchunks > 1) || (min_np <= b0);

  for (int64_t off = 0; off < max_np; off += max_packs) {
    const int64_t base = (int64_t)parity * 4 * cap;
    const int64_t my_rem = np[rank] - off;
    const int my_cur =
        my_rem <= 0 ? 0 : (my_rem < max_packs ? (int)my_rem : max_packs);
    for (int p = first; p < my_cur; p += stride) {
      const uint4 raw = in[off + p];
      uint4 v = raw;
      v.x = ll_escape<HALF>(v.x);
      v.y = ll_escape<HALF>(v.y);
      v.z = ll_escape<HALF>(v.z);
      v.w = ll_escape<HALF>(v.w);
#pragma unroll
      for (int r = 1; r < 4; ++r)
        ll_store(peers.p[(rank + r) & 3] + base + rank * cap + (int64_t)p * 16,
                 v);
      out[ooff[rank] + off + p] = raw;
    }
    char* mine = peers.p[rank] + base;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      if (r == rank) continue;
      const int64_t r_rem = np[r] - off;
      const int rcur =
          r_rem <= 0 ? 0 : (r_rem < max_packs ? (int)r_rem : max_packs);
      for (int p = first; p < rcur; p += stride) {
        char* a = mine + r * cap + (int64_t)p * 16;
        uint4 v;
        do {
          v = ll_load(a);
        } while (!ll_ready4<HALF>(v));
        ll_store(a, empty);
        out[ooff[r] + off + p] = v;
      }
    }
    if (need_sync) {
      // Post-read 4-peer per-CTA barrier ensuring all slot re-arms complete
      // before any peer reuses this parity on chunk + 2 or the next call.
      __threadfence_system();
      __syncthreads();
      if (threadIdx.x < 4 && (int)threadIdx.x != rank) {
        ll_store(peers.p[threadIdx.x] + base + rank * cap +
                     (int64_t)sync_pack * 16,
                 zero4);
        char* a = mine + (int64_t)threadIdx.x * cap + (int64_t)sync_pack * 16;
        uint4 v;
        do {
          v = ll_load(a);
        } while (!ll_ready4<HALF>(v));
        ll_store(a, empty);
      }
    }
    __syncthreads();
    parity ^= 1;
  }
  if (threadIdx.x == 0) counter[blockIdx.x] += nchunks;
}

// All-to-all over the 4-rank slots (VLLM_V41_ATTN_H64 head/token exchange):
// inp holds 4 equal chunks (chunk r for rank r); out receives chunk `rank`
// of every rank r at out chunk r. Same escape / parity protocol as
// ll_allgather (each source has its own `cap` region on the receiver).
template <bool HALF>
__global__ void __launch_bounds__(LL_THREADS)
    ll_alltoall_kernel(uint4* __restrict__ out, const uint4* __restrict__ in,
                       LLPeers peers, int rank, int npack, int64_t cap,
                       int* __restrict__ counter) {
  const int64_t base = (int64_t)(counter[blockIdx.x] & 1) * 4 * cap;
  const int stride = gridDim.x * blockDim.x;
  const int first = blockIdx.x * blockDim.x + threadIdx.x;
  for (int p = first; p < npack; p += stride) {
#pragma unroll
    for (int r = 1; r < 4; ++r) {
      const int dst = (rank + r) & 3;
      uint4 v = in[(int64_t)dst * npack + p];
      v.x = ll_escape<HALF>(v.x);
      v.y = ll_escape<HALF>(v.y);
      v.z = ll_escape<HALF>(v.z);
      v.w = ll_escape<HALF>(v.w);
      ll_store(peers.p[dst] + base + rank * cap + (int64_t)p * 16, v);
    }
    out[(int64_t)rank * npack + p] = in[(int64_t)rank * npack + p];
  }
  char* mine = peers.p[rank] + base;
  const uint4 empty = make_uint4(~0u, ~0u, ~0u, ~0u);
  for (int p = first; p < npack; p += stride) {
#pragma unroll
    for (int r = 1; r < 4; ++r) {
      const int src = (rank + r) & 3;
      char* a = mine + src * cap + (int64_t)p * 16;
      uint4 v;
      do {
        v = ll_load(a);
      } while (!ll_ready4<HALF>(v));
      ll_store(a, empty);
      out[(int64_t)src * npack + p] = v;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) counter[blockIdx.x] += 1;
}

void ll_alltoall(torch::Tensor out, torch::Tensor inp, int64_t p0, int64_t p1,
                 int64_t p2, int64_t p3, int64_t rank, int64_t cap,
                 torch::Tensor counter) {
  TORCH_CHECK(inp.scalar_type() == torch::kFloat16 &&
              out.scalar_type() == torch::kFloat16 && inp.is_contiguous() &&
              out.is_contiguous() && inp.numel() == out.numel());
  const int64_t bytes = inp.numel() * 2 / 4;  // per rank chunk
  TORCH_CHECK(bytes % 16 == 0 && bytes <= cap, "ll_alltoall: size");
  TORCH_CHECK(counter.scalar_type() == torch::kInt32);
  const int blocks = (int)counter.numel();
  const at::cuda::OptionalCUDAGuard guard(device_of(inp));
  auto stream = at::cuda::getCurrentCUDAStream();
  LLPeers peers{{reinterpret_cast<char*>(p0), reinterpret_cast<char*>(p1),
                 reinterpret_cast<char*>(p2), reinterpret_cast<char*>(p3)}};
  ll_alltoall_kernel<true><<<blocks, LL_THREADS, 0, stream>>>(
      reinterpret_cast<uint4*>(out.data_ptr()),
      reinterpret_cast<const uint4*>(inp.data_ptr()), peers, (int)rank,
      (int)(bytes / 16), cap, counter.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ll_allgather(torch::Tensor out, torch::Tensor inp, int64_t p0, int64_t p1,
                  int64_t p2, int64_t p3, int64_t rank, int64_t cap,
                  torch::Tensor counter) {
  const bool half_t = inp.scalar_type() == torch::kFloat16;
  TORCH_CHECK(half_t || inp.scalar_type() == torch::kFloat32);
  TORCH_CHECK(inp.dim() == 2 && out.dim() == 2 && inp.is_contiguous() &&
              out.is_contiguous());
  TORCH_CHECK(out.scalar_type() == inp.scalar_type() &&
              out.size(0) == inp.size(0) && out.size(1) == 4 * inp.size(1));
  const int64_t row_bytes = inp.size(1) * inp.element_size();
  const int64_t bytes = inp.size(0) * row_bytes;
  TORCH_CHECK(row_bytes % 16 == 0 && cap > (LL_MAX_BLOCKS + 16) * 16,
              "ll_allgather: size");
  if (bytes == 0) return;
  TORCH_CHECK(counter.numel() >= LL_BLOCKS &&
              counter.scalar_type() == torch::kInt32);
  // One block per counter: every call on these slots launches the same grid.
  const int blocks = (int)std::min<int64_t>(counter.numel(), LL_MAX_BLOCKS);
  const at::cuda::OptionalCUDAGuard guard(device_of(inp));
  auto stream = at::cuda::getCurrentCUDAStream();
  LLPeers peers{{reinterpret_cast<char*>(p0), reinterpret_cast<char*>(p1),
                 reinterpret_cast<char*>(p2), reinterpret_cast<char*>(p3)}};
  auto* o = reinterpret_cast<uint4*>(out.data_ptr());
  const auto* i = reinterpret_cast<const uint4*>(inp.data_ptr());
  const int64_t npack = bytes / 16, rp = row_bytes / 16;
  if (half_t)
    ll_allgather_kernel<true><<<blocks, LL_THREADS, 0, stream>>>(
        o, i, peers, (int)rank, npack, rp, cap, counter.data_ptr<int>());
  else
    ll_allgather_kernel<false><<<blocks, LL_THREADS, 0, stream>>>(
        o, i, peers, (int)rank, npack, rp, cap, counter.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ll_dp_allgather_var(torch::Tensor out, torch::Tensor inp, int64_t p0,
                         int64_t p1, int64_t p2, int64_t p3, int64_t rank,
                         int64_t s0, int64_t s1, int64_t s2, int64_t s3,
                         int64_t cap, torch::Tensor counter) {
  const bool half_t = out.scalar_type() == torch::kFloat16;
  TORCH_CHECK(half_t || out.scalar_type() == torch::kFloat32);
  TORCH_CHECK(inp.scalar_type() == out.scalar_type());
  TORCH_CHECK(inp.dim() >= 2 && out.dim() == inp.dim() && inp.is_contiguous() &&
              out.is_contiguous());
  const int64_t s[4] = {s0, s1, s2, s3};
  const int64_t out_rows = s0 + s1 + s2 + s3;
  TORCH_CHECK(rank >= 0 && rank < 4 && inp.size(0) == s[rank] &&
              out.size(0) == out_rows);
  if (out_rows == 0) return;
  const int64_t width_elems = out.numel() / out_rows;
  if (s[rank] > 0) {
    TORCH_CHECK(inp.numel() == s[rank] * width_elems);
  }
  const int64_t row_bytes = width_elems * out.element_size();
  TORCH_CHECK(row_bytes % 16 == 0 && cap > (LL_MAX_BLOCKS + 16) * 16,
              "ll_dp_allgather_var: size");
  TORCH_CHECK(counter.numel() >= LL_BLOCKS &&
              counter.scalar_type() == torch::kInt32);
  const int blocks = (int)std::min<int64_t>(counter.numel(), LL_MAX_BLOCKS);
  const at::cuda::OptionalCUDAGuard guard(device_of(out));
  auto stream = at::cuda::getCurrentCUDAStream();
  LLPeers peers{{reinterpret_cast<char*>(p0), reinterpret_cast<char*>(p1),
                 reinterpret_cast<char*>(p2), reinterpret_cast<char*>(p3)}};
  if (half_t) {
    ll_dp_allgather_var_kernel<true><<<blocks, LL_THREADS, 0, stream>>>(
        reinterpret_cast<uint4*>(out.data_ptr()),
        reinterpret_cast<const uint4*>(inp.data_ptr()), peers, (int)rank,
        (int)s0, (int)s1, (int)s2, (int)s3, row_bytes / 16, cap,
        counter.data_ptr<int>());
  } else {
    ll_dp_allgather_var_kernel<false><<<blocks, LL_THREADS, 0, stream>>>(
        reinterpret_cast<uint4*>(out.data_ptr()),
        reinterpret_cast<const uint4*>(inp.data_ptr()), peers, (int)rank,
        (int)s0, (int)s1, (int)s2, (int)s3, row_bytes / 16, cap,
        counter.data_ptr<int>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ll_allgather_rows(torch::Tensor out, torch::Tensor inp, int64_t p0,
                       int64_t p1, int64_t p2, int64_t p3, int64_t rank,
                       int64_t cap, torch::Tensor counter) {
  const int64_t rows = inp.size(0);
  ll_dp_allgather_var(out, inp, p0, p1, p2, p3, rank, rows, rows, rows, rows,
                      cap, counter);
}

// out may alias inp. peers: the four ranks' receive buffers (2 * 4 * cap bytes,
// all-ones when armed); counter: LL_BLOCKS int32 zeros, private to this rank.
void ll_allreduce(torch::Tensor out, torch::Tensor inp, int64_t p0, int64_t p1,
                  int64_t p2, int64_t p3, int64_t rank, int64_t cap,
                  torch::Tensor counter) {
  const bool half_t = inp.scalar_type() == torch::kFloat16;
  TORCH_CHECK(half_t || inp.scalar_type() == torch::kFloat32);
  TORCH_CHECK(out.scalar_type() == inp.scalar_type() &&
              out.numel() == inp.numel());
  TORCH_CHECK(inp.is_contiguous() && out.is_contiguous());
  const int64_t bytes = inp.numel() * inp.element_size();
  TORCH_CHECK(bytes % 16 == 0 && cap > (LL_MAX_BLOCKS + 16) * 16,
              "ll_allreduce: size");
  if (bytes == 0) return;
  TORCH_CHECK(counter.numel() >= LL_BLOCKS &&
              counter.scalar_type() == torch::kInt32);
  // One block per counter: every call on these slots launches the same grid.
  const int blocks = (int)std::min<int64_t>(counter.numel(), LL_MAX_BLOCKS);
  const at::cuda::OptionalCUDAGuard guard(device_of(inp));
  auto stream = at::cuda::getCurrentCUDAStream();
  LLPeers peers{{reinterpret_cast<char*>(p0), reinterpret_cast<char*>(p1),
                 reinterpret_cast<char*>(p2), reinterpret_cast<char*>(p3)}};
  auto* o = reinterpret_cast<uint4*>(out.data_ptr());
  const auto* i = reinterpret_cast<const uint4*>(inp.data_ptr());
  if (half_t)
    ll_allreduce_kernel<true><<<blocks, LL_THREADS, 0, stream>>>(
        o, i, peers, (int)rank, bytes / 16, cap, counter.data_ptr<int>());
  else
    ll_allreduce_kernel<false><<<blocks, LL_THREADS, 0, stream>>>(
        o, i, peers, (int)rank, bytes / 16, cap, counter.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Multi-chunk reduce-scatter along dim 0 on the same slots and parity: inp
// holds four equal row chunks, chunk r goes to rank r, and each rank sums the
// four chunks it receives in rank order 0..3 (TP4-SP or DP4 combine).
template <bool HALF>
__global__ void __launch_bounds__(LL_THREADS)
    ll_reduce_scatter_kernel(uint4* __restrict__ out,
                             const uint4* __restrict__ in, LLPeers peers,
                             int rank, int64_t npack, int64_t cap,
                             int* __restrict__ counter) {
  if (npack == 0) return;
  int parity = counter[blockIdx.x] & 1;
  const int stride = gridDim.x * blockDim.x;
  const int first = blockIdx.x * blockDim.x + threadIdx.x;
  const int b0 = blockIdx.x * blockDim.x;
  const int total_cap_packs = (int)(cap / 16);
  const int max_packs = total_cap_packs - LL_MAX_BLOCKS;
  const int sync_pack = total_cap_packs - 1 - blockIdx.x;
  const uint4 empty = make_uint4(~0u, ~0u, ~0u, ~0u);
  const uint4 zero4 = make_uint4(0u, 0u, 0u, ~0u ^ ~0u);
  const int nchunks = (int)((npack + max_packs - 1) / max_packs);

  for (int64_t off = 0; off < npack; off += max_packs) {
    const int cur =
        (npack - off) < max_packs ? (int)(npack - off) : max_packs;
    const int64_t base = (int64_t)parity * 4 * cap;
    for (int p = first; p < cur; p += stride) {
#pragma unroll
      for (int r = 1; r < 4; ++r) {
        const int dst = (rank + r) & 3;
        uint4 v = in[(int64_t)dst * npack + off + p];
        v.x = ll_escape<HALF>(v.x);
        v.y = ll_escape<HALF>(v.y);
        v.z = ll_escape<HALF>(v.z);
        v.w = ll_escape<HALF>(v.w);
        ll_store(peers.p[dst] + base + rank * cap + (int64_t)p * 16, v);
      }
    }
    char* mine = peers.p[rank] + base;
    for (int p = first; p < cur; p += stride) {
      uint4 v[4];
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        if (r == rank) {
          v[r] = in[(int64_t)rank * npack + off + p];
          v[r].x = ll_escape<HALF>(v[r].x);
          v[r].y = ll_escape<HALF>(v[r].y);
          v[r].z = ll_escape<HALF>(v[r].z);
          v[r].w = ll_escape<HALF>(v[r].w);
        } else {
          const char* a = mine + r * cap + (int64_t)p * 16;
          do {
            v[r] = ll_load(a);
          } while (!ll_ready4<HALF>(v[r]));
        }
      }
#pragma unroll
      for (int r = 0; r < 4; ++r)
        if (r != rank) ll_store(mine + r * cap + (int64_t)p * 16, empty);
      float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int r = 0; r < 4; ++r) ll_add<HALF>(acc, v[r]);
      uint4 o;
      if (HALF) {
        __half2 h[4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
          h[i] = __floats2half2_rn(acc[2 * i], acc[2 * i + 1]);
        o = *reinterpret_cast<uint4*>(h);
      } else {
        o = make_uint4(__float_as_uint(acc[0]), __float_as_uint(acc[1]),
                       __float_as_uint(acc[2]), __float_as_uint(acc[3]));
      }
      out[off + p] = o;
    }
    if (nchunks > 1 || cur <= b0) {
      __threadfence_system();
      __syncthreads();
      if (threadIdx.x < 4 && (int)threadIdx.x != rank) {
        ll_store(peers.p[threadIdx.x] + base + rank * cap +
                     (int64_t)sync_pack * 16,
                 zero4);
        char* a = mine + (int64_t)threadIdx.x * cap + (int64_t)sync_pack * 16;
        uint4 v;
        do {
          v = ll_load(a);
        } while (!ll_ready4<HALF>(v));
        ll_store(a, empty);
      }
    }
    __syncthreads();
    parity ^= 1;
  }
  if (threadIdx.x == 0) counter[blockIdx.x] += nchunks;
}

// inp [4 * rows, ...] contiguous -> out [rows, ...]; arbitrary size (multi-chunk).
void ll_reduce_scatter(torch::Tensor out, torch::Tensor inp, int64_t p0,
                       int64_t p1, int64_t p2, int64_t p3, int64_t rank,
                       int64_t cap, torch::Tensor counter) {
  const bool half_t = inp.scalar_type() == torch::kFloat16;
  TORCH_CHECK(half_t || inp.scalar_type() == torch::kFloat32);
  TORCH_CHECK(out.scalar_type() == inp.scalar_type() &&
              4 * out.numel() == inp.numel());
  TORCH_CHECK(inp.is_contiguous() && out.is_contiguous());
  const int64_t bytes = out.numel() * out.element_size();
  TORCH_CHECK(bytes % 16 == 0 && cap > (LL_MAX_BLOCKS + 16) * 16,
              "ll_reduce_scatter: size");
  if (bytes == 0) return;
  TORCH_CHECK(counter.numel() >= LL_BLOCKS &&
              counter.scalar_type() == torch::kInt32);
  const int blocks = (int)std::min<int64_t>(counter.numel(), LL_MAX_BLOCKS);
  const at::cuda::OptionalCUDAGuard guard(device_of(inp));
  auto stream = at::cuda::getCurrentCUDAStream();
  LLPeers peers{{reinterpret_cast<char*>(p0), reinterpret_cast<char*>(p1),
                 reinterpret_cast<char*>(p2), reinterpret_cast<char*>(p3)}};
  auto* o = reinterpret_cast<uint4*>(out.data_ptr());
  const auto* i = reinterpret_cast<const uint4*>(inp.data_ptr());
  if (half_t)
    ll_reduce_scatter_kernel<true><<<blocks, LL_THREADS, 0, stream>>>(
        o, i, peers, (int)rank, bytes / 16, cap, counter.data_ptr<int>());
  else
    ll_reduce_scatter_kernel<false><<<blocks, LL_THREADS, 0, stream>>>(
        o, i, peers, (int)rank, bytes / 16, cap, counter.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---- Expert-parallel combine over 16 ranks in one push ----------------------
// TP4 x DP4 x EP16: rank (g, t) holds FP32 routed partials for the gathered
// rows of all four DP groups ([4 * rows, H], group g' rows at g' * rows) and its
// TP shard's shared-expert partial for its own group's rows. It pushes group
// g' rows (+ the shared partial when g' == g) to the four ranks of group g'
// (itself included); every rank then sums its 16 received blocks in source
// order 0..15, so the four TP ranks of a group get bitwise-equal rows. This is
// the reduce-scatter over DP, the shared add and the TP all-reduce in one sync.
// Slots: [parity][source 0..15][cap]; same sentinel and parity scheme as the
// four-rank kernels above.
struct LLPeers16 {
  char* p[16];
};

__global__ void __launch_bounds__(LL_THREADS)
    ll_ep_combine16_kernel(float4* __restrict__ out,
                           const uint4* __restrict__ routed,
                           const uint2* __restrict__ shared, LLPeers16 peers,
                           int rank, int npack, int64_t cap,
                           int* __restrict__ counter) {
  const int64_t base = (int64_t)(counter[blockIdx.x] & 1) * 16 * cap;
  const int stride = gridDim.x * blockDim.x;
  const int first = blockIdx.x * blockDim.x + threadIdx.x;
  const int group = rank >> 2;
  for (int p = first; p < npack; p += stride) {
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      uint4 v = routed[(int64_t)g * npack + p];
      if (g == group) {
        // The shared expert's FP16 partial, widened here (4 values per pack).
        const uint2 s = shared[p];
        const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&s.x));
        const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&s.y));
        v.x = __float_as_uint(__uint_as_float(v.x) + lo.x);
        v.y = __float_as_uint(__uint_as_float(v.y) + lo.y);
        v.z = __float_as_uint(__uint_as_float(v.z) + hi.x);
        v.w = __float_as_uint(__uint_as_float(v.w) + hi.y);
      }
      v.x = ll_escape<false>(v.x);
      v.y = ll_escape<false>(v.y);
      v.z = ll_escape<false>(v.z);
      v.w = ll_escape<false>(v.w);
#pragma unroll
      for (int t = 0; t < 4; ++t)
        ll_store(peers.p[g * 4 + t] + base + rank * cap + (int64_t)p * 16, v);
    }
  }
  char* mine = peers.p[rank] + base;
  const uint4 empty = make_uint4(~0u, ~0u, ~0u, ~0u);
  for (int p = first; p < npack; p += stride) {
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 4
    for (int src = 0; src < 16; ++src) {
      char* a = mine + src * cap + (int64_t)p * 16;
      uint4 v;
      do {
        v = ll_load(a);
      } while (!ll_ready4<false>(v));
      ll_store(a, empty);
      acc[0] += __uint_as_float(v.x);
      acc[1] += __uint_as_float(v.y);
      acc[2] += __uint_as_float(v.z);
      acc[3] += __uint_as_float(v.w);
    }
    out[p] = make_float4(acc[0], acc[1], acc[2], acc[3]);
  }
  __syncthreads();
  if (threadIdx.x == 0) counter[blockIdx.x] += 1;
}

// routed [4 * rows, H] FP32, shared [rows, H] FP16 -> out [rows, H] FP32.
// peers: 16 receive buffers (2 * 16 * cap bytes, armed), indexed by EP rank
// (DP group * 4 + TP rank); rows * H * 4 <= cap.
void ll_ep_combine16(torch::Tensor out, torch::Tensor routed,
                     torch::Tensor shared, torch::Tensor peers, int64_t rank,
                     int64_t cap, torch::Tensor counter) {
  TORCH_CHECK(routed.scalar_type() == torch::kFloat32 &&
              shared.scalar_type() == torch::kFloat16 &&
              out.scalar_type() == torch::kFloat32);
  TORCH_CHECK(routed.is_contiguous() && shared.is_contiguous() &&
              out.is_contiguous());
  TORCH_CHECK(routed.numel() == 4 * out.numel() &&
              shared.numel() == out.numel());
  const int64_t bytes = out.numel() * 4;
  TORCH_CHECK(bytes % 16 == 0 && bytes <= cap, "ll_ep_combine16: size");
  TORCH_CHECK(peers.numel() == 16 && peers.scalar_type() == torch::kInt64 &&
              peers.device().is_cpu());
  TORCH_CHECK(rank >= 0 && rank < 16);
  TORCH_CHECK(counter.numel() >= LL_BLOCKS &&
              counter.scalar_type() == torch::kInt32);
  const int blocks = (int)std::min<int64_t>(counter.numel(), LL_MAX_BLOCKS);
  const at::cuda::OptionalCUDAGuard guard(device_of(routed));
  auto stream = at::cuda::getCurrentCUDAStream();
  LLPeers16 p16;
  const int64_t* ptr = peers.data_ptr<int64_t>();
  for (int i = 0; i < 16; ++i) p16.p[i] = reinterpret_cast<char*>(ptr[i]);
  ll_ep_combine16_kernel<<<blocks, LL_THREADS, 0, stream>>>(
      reinterpret_cast<float4*>(out.data_ptr()),
      reinterpret_cast<const uint4*>(routed.data_ptr()),
      reinterpret_cast<const uint2*>(shared.data_ptr()), p16, (int)rank,
      (int)(bytes / 16), cap, counter.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---- The same combine, TP-sliced -------------------------------------------
// ll_ep_combine16 sends every rank's full rows to all four TP ranks of the
// destination group, which then sum identical data: each rank receives
// 16 * rows * H * 4 bytes. Here TP rank t of a group receives only columns
// [t * H/4, (t + 1) * H/4) of its group's rows from the 16 sources (a quarter),
// sums them in source order 0..15 as before, and pushes that slice to the
// three other TP ranks of its group; every rank then holds the full rows, each
// slice summed once, so the four TP ranks agree bitwise and the values equal
// ll_ep_combine16's. Phase-1 blocks of a source use the first half of its slot
// (rows * H <= cap / 2 bytes), the slice exchange the second half.
//
// SPARSE: ids [4 * rows, topk] are the gathered routes every rank holds (the
// dispatch all-gather). A source's partial of a row is exactly zero unless one
// of the row's routes is an expert it owns (id / per_rank; padded rows route
// past rank 15), so the source skips those rows and the receiver, deriving the
// same mask, neither polls nor resets them: at top-6 of 16 ranks about two
// thirds of the push. Rows of the receiver's own group always come from its
// group's four ranks (the shared partials). Sums keep source order with the
// zeros left out, so the result is bitwise the dense kernel's.
__device__ __forceinline__ unsigned ll_route_owners(const int* __restrict__ ids,
                                                    int64_t row, int topk,
                                                    int per_rank) {
  unsigned owners = 0;
  for (int j = 0; j < topk; ++j) {
    const int e = __ldg(ids + row * topk + j);
    const int o = e / per_rank;
    if (e >= 0 && o < 16) owners |= 1u << o;
  }
  return owners;
}

template <bool SPARSE>
__global__ void __launch_bounds__(LL_THREADS)
    ll_ep_combine16_tp_kernel(float4* __restrict__ out,
                              const uint4* __restrict__ routed,
                              const uint2* __restrict__ shared, LLPeers16 peers,
                              int rank, int rows, int hpack, int64_t cap,
                              int* __restrict__ counter,
                              const int* __restrict__ ids, int topk,
                              int per_rank) {
  const int64_t base = (int64_t)(counter[blockIdx.x] & 1) * 16 * cap;
  const int stride = gridDim.x * blockDim.x;
  const int first = blockIdx.x * blockDim.x + threadIdx.x;
  const int group = rank >> 2, tp = rank & 3;
  const int qpack = hpack >> 2;  // float4 packs per row per TP slice
  const int npack = rows * qpack;
  // Phase 1: slice t of group g's rows goes to rank (g, t).
  for (int p = first; p < npack; p += stride) {
    const int r = p / qpack, c = p - r * qpack;
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      if (SPARSE && g != group &&
          !((ll_route_owners(ids, (int64_t)g * rows + r, topk, per_rank) >>
             rank) &
            1))
        continue;
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        const int64_t col = (int64_t)t * qpack + c;
        uint4 v = routed[((int64_t)g * rows + r) * hpack + col];
        if (g == group) {
          const uint2 s = shared[(int64_t)r * hpack + col];
          const float2 lo =
              __half22float2(*reinterpret_cast<const __half2*>(&s.x));
          const float2 hi =
              __half22float2(*reinterpret_cast<const __half2*>(&s.y));
          v.x = __float_as_uint(__uint_as_float(v.x) + lo.x);
          v.y = __float_as_uint(__uint_as_float(v.y) + lo.y);
          v.z = __float_as_uint(__uint_as_float(v.z) + hi.x);
          v.w = __float_as_uint(__uint_as_float(v.w) + hi.y);
        }
        v.x = ll_escape<false>(v.x);
        v.y = ll_escape<false>(v.y);
        v.z = ll_escape<false>(v.z);
        v.w = ll_escape<false>(v.w);
        ll_store(peers.p[g * 4 + t] + base + rank * cap + (int64_t)p * 16, v);
      }
    }
  }
  // Phase 2: sum this rank's slice, keep it and push it to the group.
  char* mine = peers.p[rank] + base;
  const uint4 empty = make_uint4(~0u, ~0u, ~0u, ~0u);
  const int64_t half = cap >> 1;
  for (int p = first; p < npack; p += stride) {
    const int r = p / qpack, c = p - r * qpack;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    const unsigned expect =
        SPARSE ? ll_route_owners(ids, (int64_t)group * rows + r, topk,
                                 per_rank) |
                     (0xfu << (group * 4))
               : 0xffffu;
#pragma unroll 4
    for (int src = 0; src < 16; ++src) {
      if (!((expect >> src) & 1)) continue;
      char* a = mine + src * cap + (int64_t)p * 16;
      uint4 v;
      do {
        v = ll_load(a);
      } while (!ll_ready4<false>(v));
      ll_store(a, empty);
      acc[0] += __uint_as_float(v.x);
      acc[1] += __uint_as_float(v.y);
      acc[2] += __uint_as_float(v.z);
      acc[3] += __uint_as_float(v.w);
    }
    const float4 sum = make_float4(acc[0], acc[1], acc[2], acc[3]);
    out[(int64_t)r * hpack + (int64_t)tp * qpack + c] = sum;
    uint4 w = make_uint4(ll_escape<false>(__float_as_uint(sum.x)),
                         ll_escape<false>(__float_as_uint(sum.y)),
                         ll_escape<false>(__float_as_uint(sum.z)),
                         ll_escape<false>(__float_as_uint(sum.w)));
#pragma unroll
    for (int t = 0; t < 4; ++t)
      if (t != tp)
        ll_store(peers.p[group * 4 + t] + base + rank * cap + half +
                     (int64_t)p * 16,
                 w);
  }
  // Phase 3: the group's other three slices.
  for (int p = first; p < npack; p += stride) {
    const int r = p / qpack, c = p - r * qpack;
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      if (t == tp) continue;
      char* a = mine + (group * 4 + t) * cap + half + (int64_t)p * 16;
      uint4 v;
      do {
        v = ll_load(a);
      } while (!ll_ready4<false>(v));
      ll_store(a, empty);
      out[(int64_t)r * hpack + (int64_t)t * qpack + c] =
          make_float4(__uint_as_float(v.x), __uint_as_float(v.y),
                      __uint_as_float(v.z), __uint_as_float(v.w));
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) counter[blockIdx.x] += 1;
}

// As ll_ep_combine16 (same arguments and result); H % 16 == 0 and
// rows * H <= cap / 2 bytes.
static void ll_ep_combine16_tp_impl(torch::Tensor out, torch::Tensor routed,
                                    torch::Tensor shared, torch::Tensor peers,
                                    int64_t rank, int64_t cap,
                                    torch::Tensor counter,
                                    std::optional<torch::Tensor> ids,
                                    int64_t per_rank) {
  TORCH_CHECK(routed.scalar_type() == torch::kFloat32 &&
              shared.scalar_type() == torch::kFloat16 &&
              out.scalar_type() == torch::kFloat32);
  TORCH_CHECK(routed.is_contiguous() && shared.is_contiguous() &&
              out.is_contiguous() && out.dim() == 2);
  TORCH_CHECK(routed.numel() == 4 * out.numel() &&
              shared.numel() == out.numel());
  const int64_t rows = out.size(0), h = out.size(1);
  TORCH_CHECK(h % 16 == 0 && rows * h <= cap / 2,
              "ll_ep_combine16_tp: size");
  TORCH_CHECK(peers.numel() == 16 && peers.scalar_type() == torch::kInt64 &&
              peers.device().is_cpu());
  TORCH_CHECK(rank >= 0 && rank < 16);
  TORCH_CHECK(counter.numel() >= LL_BLOCKS &&
              counter.scalar_type() == torch::kInt32);
  const int blocks = (int)std::min<int64_t>(counter.numel(), LL_MAX_BLOCKS);
  const at::cuda::OptionalCUDAGuard guard(device_of(routed));
  auto stream = at::cuda::getCurrentCUDAStream();
  LLPeers16 p16;
  const int64_t* ptr = peers.data_ptr<int64_t>();
  for (int i = 0; i < 16; ++i) p16.p[i] = reinterpret_cast<char*>(ptr[i]);
  if (ids.has_value())
    ll_ep_combine16_tp_kernel<true><<<blocks, LL_THREADS, 0, stream>>>(
        reinterpret_cast<float4*>(out.data_ptr()),
        reinterpret_cast<const uint4*>(routed.data_ptr()),
        reinterpret_cast<const uint2*>(shared.data_ptr()), p16, (int)rank,
        (int)rows, (int)(h / 4), cap, counter.data_ptr<int>(),
        ids->data_ptr<int>(), (int)ids->size(1), (int)per_rank);
  else
    ll_ep_combine16_tp_kernel<false><<<blocks, LL_THREADS, 0, stream>>>(
        reinterpret_cast<float4*>(out.data_ptr()),
        reinterpret_cast<const uint4*>(routed.data_ptr()),
        reinterpret_cast<const uint2*>(shared.data_ptr()), p16, (int)rank,
        (int)rows, (int)(h / 4), cap, counter.data_ptr<int>(), nullptr, 0, 1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ll_ep_combine16_tp(torch::Tensor out, torch::Tensor routed,
                        torch::Tensor shared, torch::Tensor peers, int64_t rank,
                        int64_t cap, torch::Tensor counter) {
  ll_ep_combine16_tp_impl(out, routed, shared, peers, rank, cap, counter,
                          std::nullopt, 1);
}

// The TP-sliced combine skipping rows with no route to the source (SPARSE):
// ids [4 * rows, topk] int32 gathered routes, per_rank experts (physical
// slots) per EP rank. Bitwise the result of ll_ep_combine16_tp.
void ll_ep_combine16_tp_sparse(torch::Tensor out, torch::Tensor routed,
                               torch::Tensor shared, torch::Tensor ids,
                               torch::Tensor peers, int64_t rank, int64_t cap,
                               torch::Tensor counter, int64_t per_rank) {
  TORCH_CHECK(ids.is_cuda() && ids.scalar_type() == torch::kInt32 &&
                  ids.dim() == 2 && ids.is_contiguous() &&
                  ids.size(0) == routed.size(0) && per_rank >= 1,
              "ll_ep_combine16_tp_sparse: ids [4 * rows, topk] int32");
  ll_ep_combine16_tp_impl(out, routed, shared, peers, rank, cap, counter, ids,
                          per_rank);
}

// Variable-size & multi-chunk TP-sliced EP16 combine (handles prefill chunks
// and uneven DP row counts [s0, s1, s2, s3] in one launch with 0 extra VRAM).
template <bool SPARSE>
__global__ void __launch_bounds__(LL_THREADS)
    ll_ep_combine16_tp_var_kernel(float4* __restrict__ out,
                                  const uint4* __restrict__ routed,
                                  const uint2* __restrict__ shared,
                                  LLPeers16 peers, int rank, int s0, int s1,
                                  int s2, int s3, int hpack, int64_t cap,
                                  int* __restrict__ counter,
                                  const int* __restrict__ ids, int topk,
                                  int per_rank) {
  const int s[4] = {s0, s1, s2, s3};
  const int64_t goff[4] = {0, s0, s0 + s1, s0 + s1 + s2};
  const int max_s = max(max(s0, s1), max(s2, s3));
  if (max_s == 0) return;

  int parity = counter[blockIdx.x] & 1;
  const int stride = gridDim.x * blockDim.x;
  const int first = blockIdx.x * blockDim.x + threadIdx.x;
  const int group = rank >> 2, tp = rank & 3;
  const int qpack = hpack >> 2;
  const int64_t half = cap >> 1;
  const int max_data_packs = (int)(half / 16) - LL_MAX_BLOCKS;
  const int max_rows = max_data_packs / qpack;
  const int64_t sync_off = ((half / 16) - 1 - blockIdx.x) * 16;
  const uint4 empty = make_uint4(~0u, ~0u, ~0u, ~0u);
  const uint4 zero4 = make_uint4(0u, 0u, 0u, 0u);
  const int nchunks = (max_s + max_rows - 1) / max_rows;

  for (int row0 = 0; row0 < max_s; row0 += max_rows) {
    const int64_t base = (int64_t)parity * 16 * cap;
    // Phase 1: push slice t of group g's chunk rows to rank (g, t).
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      const int cur_g = max(0, min(max_rows, s[g] - row0));
      const int npack_g = cur_g * qpack;
      for (int p = first; p < npack_g; p += stride) {
        const int r_loc = p / qpack, c = p - r_loc * qpack;
        const int r = row0 + r_loc;
        const int64_t r_glob = goff[g] + r;
        if (SPARSE && g != group &&
            !((ll_route_owners(ids, r_glob, topk, per_rank) >> rank) & 1))
          continue;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
          const int64_t col = (int64_t)t * qpack + c;
          uint4 v = routed[r_glob * hpack + col];
          if (g == group) {
            const uint2 sh = shared[(int64_t)r * hpack + col];
            const float2 lo =
                __half22float2(*reinterpret_cast<const __half2*>(&sh.x));
            const float2 hi =
                __half22float2(*reinterpret_cast<const __half2*>(&sh.y));
            v.x = __float_as_uint(__uint_as_float(v.x) + lo.x);
            v.y = __float_as_uint(__uint_as_float(v.y) + lo.y);
            v.z = __float_as_uint(__uint_as_float(v.z) + hi.x);
            v.w = __float_as_uint(__uint_as_float(v.w) + hi.y);
          }
          v.x = ll_escape<false>(v.x);
          v.y = ll_escape<false>(v.y);
          v.z = ll_escape<false>(v.z);
          v.w = ll_escape<false>(v.w);
          ll_store(peers.p[g * 4 + t] + base + rank * cap + (int64_t)p * 16, v);
        }
      }
    }

    // Phase 2: sum this rank's slice of its own group's chunk rows & share.
    char* mine = peers.p[rank] + base;
    const int cur_me = max(0, min(max_rows, s[group] - row0));
    const int npack_me = cur_me * qpack;
    for (int p = first; p < npack_me; p += stride) {
      const int r_loc = p / qpack, c = p - r_loc * qpack;
      const int r = row0 + r_loc;
      const int64_t r_glob = goff[group] + r;
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
      const unsigned expect =
          SPARSE ? ll_route_owners(ids, r_glob, topk, per_rank) |
                       (0xfu << (group * 4))
                 : 0xffffu;
#pragma unroll 4
      for (int src = 0; src < 16; ++src) {
        if (!((expect >> src) & 1)) continue;
        char* a = mine + src * cap + (int64_t)p * 16;
        uint4 v;
        do {
          v = ll_load(a);
        } while (!ll_ready4<false>(v));
        ll_store(a, empty);
        acc[0] += __uint_as_float(v.x);
        acc[1] += __uint_as_float(v.y);
        acc[2] += __uint_as_float(v.z);
        acc[3] += __uint_as_float(v.w);
      }
      const float4 sum = make_float4(acc[0], acc[1], acc[2], acc[3]);
      out[(int64_t)r * hpack + (int64_t)tp * qpack + c] = sum;
      uint4 w = make_uint4(ll_escape<false>(__float_as_uint(sum.x)),
                           ll_escape<false>(__float_as_uint(sum.y)),
                           ll_escape<false>(__float_as_uint(sum.z)),
                           ll_escape<false>(__float_as_uint(sum.w)));
#pragma unroll
      for (int t = 0; t < 4; ++t)
        if (t != tp)
          ll_store(peers.p[group * 4 + t] + base + rank * cap + half +
                       (int64_t)p * 16,
                   w);
    }

    // Phase 3: receive the group's other three slices.
    for (int p = first; p < npack_me; p += stride) {
      const int r_loc = p / qpack, c = p - r_loc * qpack;
      const int r = row0 + r_loc;
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        if (t == tp) continue;
        char* a = mine + (group * 4 + t) * cap + half + (int64_t)p * 16;
        uint4 v;
        do {
          v = ll_load(a);
        } while (!ll_ready4<false>(v));
        ll_store(a, empty);
        out[(int64_t)r * hpack + (int64_t)t * qpack + c] =
            make_float4(__uint_as_float(v.x), __uint_as_float(v.y),
                        __uint_as_float(v.z), __uint_as_float(v.w));
      }
    }

    // 16-peer per-CTA barrier guaranteeing all re-arms precede reuse of parity.
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x < 16 && (int)threadIdx.x != rank) {
      ll_store(peers.p[threadIdx.x] + base + rank * cap + sync_off, zero4);
      char* a = mine + (int64_t)threadIdx.x * cap + sync_off;
      uint4 v;
      do {
        v = ll_load(a);
      } while (!ll_ready4<false>(v));
      ll_store(a, empty);
    }
    __syncthreads();
    parity ^= 1;
  }
  if (threadIdx.x == 0) counter[blockIdx.x] += nchunks;
}

void ll_ep_combine16_tp_var(torch::Tensor out, torch::Tensor routed,
                            torch::Tensor shared,
                            std::optional<torch::Tensor> ids,
                            torch::Tensor peers, int64_t rank, int64_t s0,
                            int64_t s1, int64_t s2, int64_t s3, int64_t cap,
                            torch::Tensor counter, int64_t per_rank) {
  TORCH_CHECK(routed.scalar_type() == torch::kFloat32 &&
              shared.scalar_type() == torch::kFloat16 &&
              out.scalar_type() == torch::kFloat32);
  TORCH_CHECK(routed.is_contiguous() && shared.is_contiguous() &&
              out.is_contiguous() && out.dim() == 2);
  const int64_t s[4] = {s0, s1, s2, s3};
  const int64_t total_rows = s0 + s1 + s2 + s3;
  TORCH_CHECK(rank >= 0 && rank < 16);
  const int group = (int)(rank >> 2);
  const int64_t h = out.size(1);
  TORCH_CHECK(out.size(0) == s[group] && shared.size(0) == s[group] &&
              shared.size(1) == h && routed.size(0) == total_rows &&
              routed.size(1) == h);
  const int qpack = (int)(h / 16);
  TORCH_CHECK(h % 16 == 0 && (cap / 32) > LL_MAX_BLOCKS + qpack,
              "ll_ep_combine16_tp_var: size");
  if (total_rows == 0) return;
  TORCH_CHECK(peers.numel() == 16 && peers.scalar_type() == torch::kInt64 &&
              peers.device().is_cpu());
  TORCH_CHECK(counter.numel() >= LL_BLOCKS &&
              counter.scalar_type() == torch::kInt32);
  const int blocks = (int)std::min<int64_t>(counter.numel(), LL_MAX_BLOCKS);
  const at::cuda::OptionalCUDAGuard guard(device_of(routed));
  auto stream = at::cuda::getCurrentCUDAStream();
  LLPeers16 p16;
  const int64_t* ptr = peers.data_ptr<int64_t>();
  for (int i = 0; i < 16; ++i) p16.p[i] = reinterpret_cast<char*>(ptr[i]);
  if (ids.has_value()) {
    TORCH_CHECK(ids->is_cuda() && ids->scalar_type() == torch::kInt32 &&
                ids->dim() == 2 && ids->is_contiguous() &&
                ids->size(0) == total_rows && per_rank >= 1);
    ll_ep_combine16_tp_var_kernel<true><<<blocks, LL_THREADS, 0, stream>>>(
        reinterpret_cast<float4*>(out.data_ptr()),
        reinterpret_cast<const uint4*>(routed.data_ptr()),
        reinterpret_cast<const uint2*>(shared.data_ptr()), p16, (int)rank,
        (int)s0, (int)s1, (int)s2, (int)s3, (int)(h / 4), cap,
        counter.data_ptr<int>(), ids->data_ptr<int>(), (int)ids->size(1),
        (int)per_rank);
  } else {
    ll_ep_combine16_tp_var_kernel<false><<<blocks, LL_THREADS, 0, stream>>>(
        reinterpret_cast<float4*>(out.data_ptr()),
        reinterpret_cast<const uint4*>(routed.data_ptr()),
        reinterpret_cast<const uint2*>(shared.data_ptr()), p16, (int)rank,
        (int)s0, (int)s1, (int)s2, (int)s3, (int)(h / 4), cap,
        counter.data_ptr<int>(), nullptr, 0, 1);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Token-sliced EP16 combine for Megatron Intra-TP4 Sequence Parallelism:
// DP group g has s[g] = 4 * sp[g] rows; rank (g, t) owns rows
// [t * sp[g] .. (t + 1) * sp[g]) of group g across ALL H columns.
// Every source rank pushes each destination rank's owned token slice directly
// (+ shared expert partial when g_src == g), and each destination rank sums its
// 16 sources directly into out [sp[group], H] FP32 with ZERO Phase-2/3 TP
// broadcast. Uses 100% of `cap` per peer; WIRE_FP16 halves NVLink bytes so even
// 4 x 2048 = 8,192 tokens runs in 1 single zero-barrier pass at cap = 5.25 MiB.
template <bool SPARSE, bool WIRE_FP16>
__global__ void __launch_bounds__(LL_THREADS)
    ll_ep_combine16_sp_var_kernel(float4* __restrict__ out,
                                  const float4* __restrict__ routed,
                                  const uint2* __restrict__ shared,
                                  LLPeers16 peers, int rank, int s0, int s1,
                                  int s2, int s3, int hpack, int64_t cap,
                                  int* __restrict__ counter,
                                  const int* __restrict__ ids, int topk,
                                  int per_rank) {
  const int sp[4] = {s0 >> 2, s1 >> 2, s2 >> 2, s3 >> 2};
  const int64_t goff[4] = {0, s0, s0 + s1, s0 + s1 + s2};
  const int max_sp = max(max(sp[0], sp[1]), max(sp[2], sp[3]));
  const int min_sp = min(min(sp[0], sp[1]), min(sp[2], sp[3]));
  if (max_sp == 0) return;

  int parity = counter[blockIdx.x] & 1;
  const int stride = gridDim.x * blockDim.x;
  const int first = blockIdx.x * blockDim.x + threadIdx.x;
  const int b0 = blockIdx.x * blockDim.x;
  const int group = rank >> 2, tp = rank & 3;
  const int wpack = WIRE_FP16 ? (hpack >> 1) : hpack;
  const int total_packs = (int)(cap / 16);
  const int max_data_packs = total_packs - LL_MAX_BLOCKS;
  const int max_rows = max_data_packs / wpack;
  const int64_t sync_off = (int64_t)(total_packs - 1 - blockIdx.x) * 16;
  const uint4 empty = make_uint4(~0u, ~0u, ~0u, ~0u);
  const uint4 zero4 = make_uint4(0u, 0u, 0u, 0u);
  const int nchunks = (max_sp + max_rows - 1) / max_rows;
  const bool need_sync = (nchunks > 1) || SPARSE || ((int64_t)min_sp * wpack <= b0);
  const uint4* shared_u4 = reinterpret_cast<const uint4*>(shared);

  for (int sub0 = 0; sub0 < max_sp; sub0 += max_rows) {
    const int64_t base = (int64_t)parity * 16 * cap;
    // Phase 1: push each peer (g, t)'s owned token slice directly.
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      const int cur_g = max(0, min(max_rows, sp[g] - sub0));
      const int npack_g = cur_g * wpack;
      for (int p = first; p < npack_g; p += stride) {
        const int r_loc = p / wpack, c = p - r_loc * wpack;
        const int r_sub = sub0 + r_loc;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
          const int dst = g * 4 + t;
          if (dst == rank) continue;
          const int r_grp = t * sp[g] + r_sub;
          const int64_t r_glob = goff[g] + r_grp;
          if (SPARSE && g != group &&
              !((ll_route_owners(ids, r_glob, topk, per_rank) >> rank) & 1))
            continue;
          uint4 v;
          if constexpr (WIRE_FP16) {
            const float4 rt0 = routed[r_glob * hpack + 2 * c + 0];
            const float4 rt1 = routed[r_glob * hpack + 2 * c + 1];
            float f[8] = {rt0.x, rt0.y, rt0.z, rt0.w,
                          rt1.x, rt1.y, rt1.z, rt1.w};
            if (g == group) {
              const uint4 sh = shared_u4[(int64_t)r_grp * wpack + c];
              ll_add<true>(f, sh);
            }
            __half2 h2[4];
#pragma unroll
            for (int i = 0; i < 4; ++i)
              h2[i] = __floats2half2_rn(f[2 * i], f[2 * i + 1]);
            v = *reinterpret_cast<const uint4*>(h2);
            v.x = ll_escape<true>(v.x);
            v.y = ll_escape<true>(v.y);
            v.z = ll_escape<true>(v.z);
            v.w = ll_escape<true>(v.w);
          } else {
            const float4 rt = routed[r_glob * hpack + c];
            float f[4] = {rt.x, rt.y, rt.z, rt.w};
            if (g == group) {
              const uint2 sh = shared[(int64_t)r_grp * hpack + c];
              const float2 lo =
                  __half22float2(*reinterpret_cast<const __half2*>(&sh.x));
              const float2 hi =
                  __half22float2(*reinterpret_cast<const __half2*>(&sh.y));
              f[0] += lo.x;
              f[1] += lo.y;
              f[2] += hi.x;
              f[3] += hi.y;
            }
            v = make_uint4(ll_escape<false>(__float_as_uint(f[0])),
                           ll_escape<false>(__float_as_uint(f[1])),
                           ll_escape<false>(__float_as_uint(f[2])),
                           ll_escape<false>(__float_as_uint(f[3])));
          }
          ll_store(peers.p[dst] + base + rank * cap + (int64_t)p * 16, v);
        }
      }
    }

    // Phase 2: sum all 16 sources for this rank's sp[group] rows directly.
    char* mine = peers.p[rank] + base;
    const int cur_me = max(0, min(max_rows, sp[group] - sub0));
    const int npack_me = cur_me * wpack;
    for (int p = first; p < npack_me; p += stride) {
      const int r_loc = p / wpack, c = p - r_loc * wpack;
      const int r_sub = sub0 + r_loc;
      const int r_grp = tp * sp[group] + r_sub;
      const int64_t r_glob = goff[group] + r_grp;
      const unsigned expect =
          SPARSE ? ll_route_owners(ids, r_glob, topk, per_rank) |
                       (0xfu << (group * 4))
                 : 0xffffu;
      if constexpr (WIRE_FP16) {
        float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll 4
        for (int src = 0; src < 16; ++src) {
          if (src == rank) {
            const float4 rt0 = routed[r_glob * hpack + 2 * c + 0];
            const float4 rt1 = routed[r_glob * hpack + 2 * c + 1];
            acc[0] += rt0.x;
            acc[1] += rt0.y;
            acc[2] += rt0.z;
            acc[3] += rt0.w;
            acc[4] += rt1.x;
            acc[5] += rt1.y;
            acc[6] += rt1.z;
            acc[7] += rt1.w;
            const uint4 sh = shared_u4[(int64_t)r_grp * wpack + c];
            ll_add<true>(acc, sh);
            continue;
          }
          if (!((expect >> src) & 1)) continue;
          char* a = mine + src * cap + (int64_t)p * 16;
          uint4 v;
          do {
            v = ll_load(a);
          } while (!ll_ready4<true>(v));
          ll_store(a, empty);
          ll_add<true>(acc, v);
        }
        out[(int64_t)r_sub * hpack + 2 * c + 0] =
            make_float4(acc[0], acc[1], acc[2], acc[3]);
        out[(int64_t)r_sub * hpack + 2 * c + 1] =
            make_float4(acc[4], acc[5], acc[6], acc[7]);
      } else {
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 4
        for (int src = 0; src < 16; ++src) {
          if (src == rank) {
            const float4 rt = routed[r_glob * hpack + c];
            const uint2 sh = shared[(int64_t)r_grp * hpack + c];
            const float2 lo =
                __half22float2(*reinterpret_cast<const __half2*>(&sh.x));
            const float2 hi =
                __half22float2(*reinterpret_cast<const __half2*>(&sh.y));
            acc[0] += rt.x + lo.x;
            acc[1] += rt.y + lo.y;
            acc[2] += rt.z + hi.x;
            acc[3] += rt.w + hi.y;
            continue;
          }
          if (!((expect >> src) & 1)) continue;
          char* a = mine + src * cap + (int64_t)p * 16;
          uint4 v;
          do {
            v = ll_load(a);
          } while (!ll_ready4<false>(v));
          ll_store(a, empty);
          acc[0] += __uint_as_float(v.x);
          acc[1] += __uint_as_float(v.y);
          acc[2] += __uint_as_float(v.z);
          acc[3] += __uint_as_float(v.w);
        }
        out[(int64_t)r_sub * hpack + c] =
            make_float4(acc[0], acc[1], acc[2], acc[3]);
      }
    }

    if (need_sync) {
      __threadfence_system();
      __syncthreads();
      if (threadIdx.x < 16 && (int)threadIdx.x != rank) {
        ll_store(peers.p[threadIdx.x] + base + rank * cap + sync_off, zero4);
        char* a = mine + (int64_t)threadIdx.x * cap + sync_off;
        uint4 v;
        do {
          v = ll_load(a);
        } while (!ll_ready4<false>(v));
        ll_store(a, empty);
      }
    }
    __syncthreads();
    parity ^= 1;
  }
  if (threadIdx.x == 0) counter[blockIdx.x] += nchunks;
}

void ll_ep_combine16_sp_var(torch::Tensor out, torch::Tensor routed,
                            torch::Tensor shared,
                            std::optional<torch::Tensor> ids,
                            torch::Tensor peers, int64_t rank, int64_t s0,
                            int64_t s1, int64_t s2, int64_t s3, int64_t cap,
                            torch::Tensor counter, int64_t per_rank,
                            bool wire_fp16) {
  TORCH_CHECK(routed.scalar_type() == torch::kFloat32 &&
              shared.scalar_type() == torch::kFloat16 &&
              out.scalar_type() == torch::kFloat32);
  TORCH_CHECK(routed.is_contiguous() && shared.is_contiguous() &&
              out.is_contiguous() && out.dim() == 2);
  TORCH_CHECK((s0 & 3) == 0 && (s1 & 3) == 0 && (s2 & 3) == 0 && (s3 & 3) == 0,
              "ll_ep_combine16_sp_var: s0..s3 must be divisible by 4");
  const int64_t s[4] = {s0, s1, s2, s3};
  const int64_t total_rows = s0 + s1 + s2 + s3;
  TORCH_CHECK(rank >= 0 && rank < 16);
  const int group = (int)(rank >> 2);
  const int64_t h = out.size(1);
  TORCH_CHECK(out.size(0) == (s[group] >> 2) && shared.size(0) == s[group] &&
              shared.size(1) == h && routed.size(0) == total_rows &&
              routed.size(1) == h);
  const int wpack = wire_fp16 ? (int)(h / 8) : (int)(h / 4);
  TORCH_CHECK(h % 16 == 0 && (cap / 16) > LL_MAX_BLOCKS + wpack,
              "ll_ep_combine16_sp_var: size");
  if (total_rows == 0) return;
  TORCH_CHECK(peers.numel() == 16 && peers.scalar_type() == torch::kInt64 &&
              peers.device().is_cpu());
  TORCH_CHECK(counter.numel() >= LL_BLOCKS &&
              counter.scalar_type() == torch::kInt32);
  const int blocks = (int)std::min<int64_t>(counter.numel(), LL_MAX_BLOCKS);
  const at::cuda::OptionalCUDAGuard guard(device_of(routed));
  auto stream = at::cuda::getCurrentCUDAStream();
  LLPeers16 p16;
  const int64_t* ptr = peers.data_ptr<int64_t>();
  for (int i = 0; i < 16; ++i) p16.p[i] = reinterpret_cast<char*>(ptr[i]);
  auto* out_p = reinterpret_cast<float4*>(out.data_ptr());
  const auto* rt_p = reinterpret_cast<const float4*>(routed.data_ptr());
  const auto* sh_p = reinterpret_cast<const uint2*>(shared.data_ptr());
  int* cnt_p = counter.data_ptr<int>();
  const int hpack = (int)(h / 4);

  if (ids.has_value()) {
    TORCH_CHECK(ids->is_cuda() && ids->scalar_type() == torch::kInt32 &&
                ids->dim() == 2 && ids->is_contiguous() &&
                ids->size(0) == total_rows && per_rank >= 1);
    if (wire_fp16) {
      ll_ep_combine16_sp_var_kernel<true, true><<<blocks, LL_THREADS, 0, stream>>>(
          out_p, rt_p, sh_p, p16, (int)rank, (int)s0, (int)s1, (int)s2, (int)s3,
          hpack, cap, cnt_p, ids->data_ptr<int>(), (int)ids->size(1),
          (int)per_rank);
    } else {
      ll_ep_combine16_sp_var_kernel<true, false><<<blocks, LL_THREADS, 0, stream>>>(
          out_p, rt_p, sh_p, p16, (int)rank, (int)s0, (int)s1, (int)s2, (int)s3,
          hpack, cap, cnt_p, ids->data_ptr<int>(), (int)ids->size(1),
          (int)per_rank);
    }
  } else {
    if (wire_fp16) {
      ll_ep_combine16_sp_var_kernel<false, true><<<blocks, LL_THREADS, 0, stream>>>(
          out_p, rt_p, sh_p, p16, (int)rank, (int)s0, (int)s1, (int)s2, (int)s3,
          hpack, cap, cnt_p, nullptr, 0, 1);
    } else {
      ll_ep_combine16_sp_var_kernel<false, false><<<blocks, LL_THREADS, 0, stream>>>(
          out_p, rt_p, sh_p, p16, (int)rank, (int)s0, (int)s1, (int)s2, (int)s3,
          hpack, cap, cnt_p, nullptr, 0, 1);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---- V4.1 HC mixes off the critical path ------------------------------------
// The next sublayer's input needs only the collapse; the mixes (pre, post,
// comb) of a transition are consumed one sublayer later. hc_mix computes them
// from the already-updated residual on a side stream: 80 CTAs own 64 columns
// each (fn rows x four streams, all tokens) and write partial dot products and
// squares; the last CTA to finish (ticket, no spinning) reduces them and runs
// Sinkhorn per token, one thread per token holding the 4x4 matrix. FP32;
// reduction order differs from the Triton kernels (not bitwise equal).
constexpr int HC_H = 5120, HC_COLS = 64, HC_CTAS = HC_H / HC_COLS, HC_V = 25;
// Tokens per grid.y block: batches spread over more CTAs (each re-reads its
// 24 KB fn slice); a decode step of up to 16 tokens keeps one block.
constexpr int HC_TPB = 16;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__global__ void __launch_bounds__(256)
    hc_mix_kernel(const float* __restrict__ residual,
                  const float* __restrict__ fn, const float* __restrict__ scale,
                  const float* __restrict__ base, float* __restrict__ pre_o,
                  float* __restrict__ post_o, float* __restrict__ comb_o,
                  float* __restrict__ ws, int* __restrict__ ticket, int T,
                  float norm_eps, float hc_eps, int iters) {
  constexpr int H = HC_H;
  __shared__ float red[8][8][8];  // [warp][token][value]
  __shared__ bool last;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int q = warp >> 1;  // quarter: fn rows q*6 .. q*6+5
  const int col = blockIdx.x * HC_COLS + (warp & 1) * 32 + lane;
  float f[6][4];
#pragma unroll
  for (int i = 0; i < 6; ++i)
#pragma unroll
    for (int s = 0; s < 4; ++s)
      f[i][s] = __ldg(fn + (size_t)(q * 6 + i) * 4 * H + s * H + col);
  // grid.y token blocks of HC_TPB: CTA (x, y) covers its 64 columns for the
  // block's tokens, in chunks of 8 (fn coefficients stay in registers).
  const int first = blockIdx.y * HC_TPB, stop = min(T, first + HC_TPB);
  for (int c0 = first; c0 < stop; c0 += 8) {
    const int tn = min(8, stop - c0);
    for (int tt = 0; tt < tn; ++tt) {
      const int t = c0 + tt;
      const float* rb = residual + (size_t)t * 4 * H + col;
      const float n0 = rb[0], n1 = rb[H], n2 = rb[2 * H], n3 = rb[3 * H];
      float v[7];
#pragma unroll
      for (int i = 0; i < 6; ++i)
        v[i] = f[i][0] * n0 + f[i][1] * n1 + f[i][2] * n2 + f[i][3] * n3;
      v[6] = n0 * n0 + n1 * n1 + n2 * n2 + n3 * n3;
#pragma unroll
      for (int i = 0; i < 7; ++i) v[i] = warp_sum(v[i]);
      if (lane < 7) {
        float w = v[0];
#pragma unroll
        for (int i = 1; i < 7; ++i)
          if (lane == i) w = v[i];
        red[warp][tt][lane] = w;
      }
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < tn * HC_V; idx += blockDim.x) {
      const int tt = idx / HC_V, v = idx % HC_V;
      const float s =
          v < 24 ? red[2 * (v / 6)][tt][v % 6] + red[2 * (v / 6) + 1][tt][v % 6]
                 : red[0][tt][6] + red[1][tt][6];
      ws[((size_t)blockIdx.x * T + c0 + tt) * HC_V + v] = s;
    }
    __syncthreads();
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    // One ticket per token block: its last CTA finishes the block's tokens.
    last = atomicAdd(ticket + blockIdx.y, 1) == (int)gridDim.x - 1;
    if (last) ticket[blockIdx.y] = 0;  // re-armed (graph-safe)
  }
  __syncthreads();
  if (!last) return;
  __threadfence();
  // Warp w: the block's tokens w, w + 8, ...: 25 sums over the CTAs, then
  // Sinkhorn in lane 0.
  for (int t = first + warp; t < stop; t += 8) {
    // Lane v < 25 sums value v over the CTAs (independent loads, in flight).
    float mine = 0.f;
    if (lane < HC_V) {
#pragma unroll 8
      for (int c = 0; c < HC_CTAS; ++c)
        mine += __ldcg(ws + ((size_t)c * T + t) * HC_V + lane);
    }
    float sums[HC_V];
#pragma unroll
    for (int v = 0; v < HC_V; ++v) sums[v] = __shfl_sync(0xffffffffu, mine, v);
    if (lane == 0) {
      const float rs = rsqrtf(sums[24] / (4 * H) + norm_eps);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float p = sums[i] * rs * scale[0] + base[i];
        const float qv = sums[4 + i] * rs * scale[1] + base[4 + i];
        pre_o[t * 4 + i] = 1.f / (1.f + __expf(-p)) + hc_eps;
        post_o[t * 4 + i] = 2.f * (1.f / (1.f + __expf(-qv)));
      }
      float m[4][4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float mx = -INFINITY;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          m[i][j] = sums[8 + i * 4 + j] * rs * scale[2] + base[8 + i * 4 + j];
          mx = fmaxf(mx, m[i][j]);
        }
        float rsum = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          m[i][j] = __expf(m[i][j] - mx);
          rsum += m[i][j];
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) m[i][j] = m[i][j] / rsum + hc_eps;
      }
      for (int it = 0; it < iters; ++it) {
        if (it > 0) {
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const float r =
                1.f / (m[i][0] + m[i][1] + m[i][2] + m[i][3] + hc_eps);
#pragma unroll
            for (int j = 0; j < 4; ++j) m[i][j] *= r;
          }
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float c =
              1.f / (m[0][j] + m[1][j] + m[2][j] + m[3][j] + hc_eps);
#pragma unroll
          for (int i = 0; i < 4; ++i) m[i][j] *= c;
        }
      }
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) comb_o[t * 16 + i * 4 + j] = m[i][j];
    }
  }
}

void hc_mix(torch::Tensor residual, torch::Tensor fn, torch::Tensor scale,
            torch::Tensor base, torch::Tensor pre_o, torch::Tensor post_o,
            torch::Tensor comb_o, torch::Tensor ws, torch::Tensor ticket,
            double norm_eps, double hc_eps, int64_t iters) {
  const int T = residual.size(0);
  TORCH_CHECK(T >= 1 && T <= 256 && residual.size(1) == 4 &&
              residual.size(2) == HC_H);
  TORCH_CHECK(residual.scalar_type() == torch::kFloat32 &&
              residual.is_contiguous());
  TORCH_CHECK(fn.scalar_type() == torch::kFloat32 && fn.is_contiguous() &&
              fn.size(0) == 24 && fn.size(1) == 4 * HC_H);
  const int blocks = (T + HC_TPB - 1) / HC_TPB;
  TORCH_CHECK(ws.numel() >= (int64_t)HC_CTAS * T * HC_V &&
              ticket.numel() >= blocks && ticket.scalar_type() == torch::kInt32);
  const at::cuda::OptionalCUDAGuard guard(device_of(residual));
  auto stream = at::cuda::getCurrentCUDAStream();
  hc_mix_kernel<<<dim3(HC_CTAS, blocks), 256, 0, stream>>>(
      residual.data_ptr<float>(), fn.data_ptr<float>(), scale.data_ptr<float>(),
      base.data_ptr<float>(), pre_o.data_ptr<float>(), post_o.data_ptr<float>(),
      comb_o.data_ptr<float>(), ws.data_ptr<float>(), ticket.data_ptr<int>(), T,
      (float)norm_eps, (float)hc_eps, (int)iters);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---- prefill HC transition parts, 8 tokens per CTA (fn read once) ----
// Same outputs and per-(token, 256-col chunk) partial sums as the Triton
// _transition_parts (residual updated in place, cbuf, csquares, squares,
// dots [t][chunk][24]); the Triton _transition_finish consumes them. Each
// thread owns one column and keeps its 24 x 4 fn coefficients in registers.
constexpr int HCP_COLS = 256, HCP_CHUNKS = HC_H / HCP_COLS, HCP_TB = 8;

template <typename TY, bool MIX>
__global__ void __launch_bounds__(HCP_COLS)
    hc_parts_tb_kernel(float* __restrict__ residual, const TY* __restrict__ y,
                       const float* __restrict__ post,
                       const float* __restrict__ comb,
                       const float* __restrict__ fn,
                       const float* __restrict__ cpre, float* __restrict__ cbuf,
                       float* __restrict__ dots, float* __restrict__ squares,
                       float* __restrict__ csquares, int T) {
  constexpr int H = HC_H;
  __shared__ float red[8][26];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int chunk = blockIdx.y;
  const int col = chunk * HCP_COLS + threadIdx.x;
  float f[MIX ? 24 : 1][4];
  if (MIX) {
#pragma unroll
    for (int r = 0; r < 24; ++r)
#pragma unroll
      for (int s = 0; s < 4; ++s)
        f[r][s] = __ldg(fn + (size_t)r * 4 * H + s * H + col);
  }
  const int t0 = blockIdx.x * HCP_TB;
  for (int tt = 0; tt < HCP_TB; ++tt) {
    const int t = t0 + tt;
    if (t >= T) break;  // block-uniform
    float* rb = residual + (size_t)t * 4 * H + col;
    const float r0 = rb[0], r1 = rb[H], r2 = rb[2 * H], r3 = rb[3 * H];
    const float yv = static_cast<float>(y[(size_t)t * H + col]);
    const float* cb = comb + t * 16;
    const float* pb = post + t * 4;
    float n[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
      n[j] = cb[j] * r0 + cb[4 + j] * r1 + cb[8 + j] * r2 + cb[12 + j] * r3 +
             pb[j] * yv;
    rb[0] = n[0];
    rb[H] = n[1];
    rb[2 * H] = n[2];
    rb[3 * H] = n[3];
    const float* pc = cpre + t * 4;
    const float coll =
        n[0] * pc[0] + n[1] * pc[1] + n[2] * pc[2] + n[3] * pc[3];
    cbuf[(size_t)t * H + col] = coll;
    float v[26];
    v[24] = coll * coll;
    v[25] = n[0] * n[0] + n[1] * n[1] + n[2] * n[2] + n[3] * n[3];
#pragma unroll
    for (int r = 0; r < 24; ++r)
      v[r] = MIX ? f[r][0] * n[0] + f[r][1] * n[1] + f[r][2] * n[2] +
                       f[r][3] * n[3]
                 : 0.f;
#pragma unroll
    for (int i = 0; i < 26; ++i) {
      if (!MIX && i < 24) continue;
      float x = v[i];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
      if (lane == 0) red[warp][i] = x;
    }
    __syncthreads();
    if (threadIdx.x < 26) {
      const int i = threadIdx.x;
      if (MIX || i >= 24) {
        float x = 0.f;
#pragma unroll
        for (int w = 0; w < 8; ++w) x += red[w][i];
        const size_t tc = (size_t)t * HCP_CHUNKS + chunk;
        if (i < 24)
          dots[tc * 24 + i] = x;
        else if (i == 24)
          csquares[tc] = x;
        else
          squares[tc] = x;
      }
    }
    __syncthreads();
  }
}

void hc_parts_tb(torch::Tensor residual, torch::Tensor y, torch::Tensor post,
                 torch::Tensor comb, std::optional<torch::Tensor> fn,
                 torch::Tensor cpre, torch::Tensor cbuf, torch::Tensor dots,
                 torch::Tensor squares, torch::Tensor csquares) {
  const int T = residual.size(0);
  TORCH_CHECK(residual.scalar_type() == torch::kFloat32 &&
              residual.is_contiguous() && residual.size(1) == 4 &&
              residual.size(2) == HC_H);
  TORCH_CHECK(y.is_contiguous() && y.numel() == (int64_t)T * HC_H);
  TORCH_CHECK(post.is_contiguous() && comb.is_contiguous() &&
              cpre.is_contiguous());
  const bool mix = fn.has_value();
  const at::cuda::OptionalCUDAGuard guard(device_of(residual));
  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid((T + HCP_TB - 1) / HCP_TB, HCP_CHUNKS);
#define HCP(TYv, MIXv)                                                        \
  hc_parts_tb_kernel<TYv, MIXv><<<grid, HCP_COLS, 0, stream>>>(               \
      residual.data_ptr<float>(), reinterpret_cast<const TYv*>(y.data_ptr()), \
      post.data_ptr<float>(), comb.data_ptr<float>(),                         \
      mix ? fn->data_ptr<float>() : nullptr, cpre.data_ptr<float>(),          \
      cbuf.data_ptr<float>(), dots.data_ptr<float>(),                         \
      squares.data_ptr<float>(), csquares.data_ptr<float>(), T)
  if (y.scalar_type() == torch::kFloat16) {
    if (mix)
      HCP(__half, true);
    else
      HCP(__half, false);
  } else {
    TORCH_CHECK(y.scalar_type() == torch::kFloat32);
    if (mix)
      HCP(float, true);
    else
      HCP(float, false);
  }
#undef HCP
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---- Fused prefill HC expand + collapse + RMSNorm (0 cbuf HBM traffic) ----
// 1 CTA per token (256 threads, 20 columns = 5 float4s per thread). Updates
// residual [T, 4, 5120] in place, keeps collapsed values in 20 registers (no
// local-memory spilling, no [T, 5120] FP32 cbuf round-trip), normalizes and
// writes out [T, 5120] FP16 directly; when MIX, also emits sq_out [T] FP32 for
// hc_sinkhorn after cuBLAS SGEMM dots = residual.view(T, 4H) @ fn^T.
template <typename TY, bool MIX>
__global__ void __launch_bounds__(256)
    hc_expand_collapse_norm_kernel(float* __restrict__ residual,
                                   const TY* __restrict__ y,
                                   const float* __restrict__ post,
                                   const float* __restrict__ comb,
                                   const float* __restrict__ cpre,
                                   const float* __restrict__ norm_w,
                                   __half* __restrict__ out,
                                   float* __restrict__ sq_out,
                                   float norm_eps) {
  constexpr int H = HC_H;
  __shared__ float s_red[2][8];
  __shared__ float s_inv;
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int t = blockIdx.x;
  float* rb = residual + static_cast<size_t>(t) * 4 * H;
  const float* cb_ptr = comb + static_cast<size_t>(t) * 16;
  const float* pb_ptr = post + static_cast<size_t>(t) * 4;
  const float* pc_ptr = cpre + static_cast<size_t>(t) * 4;
  float cb[16], pb[4], pc[4];
#pragma unroll
  for (int i = 0; i < 16; ++i) cb[i] = __ldg(cb_ptr + i);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    pb[i] = __ldg(pb_ptr + i);
    pc[i] = __ldg(pc_ptr + i);
  }
  float4 coll_v[5];
  float csq_acc = 0.0f;
  float sq_acc = 0.0f;
#pragma unroll
  for (int g = 0; g < 5; ++g) {
    const int col = g * 1024 + tid * 4;
    const float4 r0 = *reinterpret_cast<const float4*>(rb + 0 * H + col);
    const float4 r1 = *reinterpret_cast<const float4*>(rb + 1 * H + col);
    const float4 r2 = *reinterpret_cast<const float4*>(rb + 2 * H + col);
    const float4 r3 = *reinterpret_cast<const float4*>(rb + 3 * H + col);
    float4 yv;
    if constexpr (std::is_same_v<TY, __half>) {
      const uint2 y_u2 = __ldg(
          reinterpret_cast<const uint2*>(y + static_cast<size_t>(t) * H + col));
      const float2 yf0 =
          __half22float2(*reinterpret_cast<const __half2*>(&y_u2.x));
      const float2 yf1 =
          __half22float2(*reinterpret_cast<const __half2*>(&y_u2.y));
      yv = make_float4(yf0.x, yf0.y, yf1.x, yf1.y);
    } else {
      yv = __ldg(reinterpret_cast<const float4*>(y + static_cast<size_t>(t) * H +
                                                 col));
    }
    const float r0a[4] = {r0.x, r0.y, r0.z, r0.w};
    const float r1a[4] = {r1.x, r1.y, r1.z, r1.w};
    const float r2a[4] = {r2.x, r2.y, r2.z, r2.w};
    const float r3a[4] = {r3.x, r3.y, r3.z, r3.w};
    const float ya[4] = {yv.x, yv.y, yv.z, yv.w};
    float na[4][4];
    float ca[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        na[j][k] = cb[j] * r0a[k] + cb[4 + j] * r1a[k] + cb[8 + j] * r2a[k] +
                   cb[12 + j] * r3a[k] + pb[j] * ya[k];
      }
      ca[k] = na[0][k] * pc[0] + na[1][k] * pc[1] + na[2][k] * pc[2] +
              na[3][k] * pc[3];
      csq_acc += ca[k] * ca[k];
      if constexpr (MIX) {
        sq_acc += na[0][k] * na[0][k] + na[1][k] * na[1][k] +
                  na[2][k] * na[2][k] + na[3][k] * na[3][k];
      }
    }
    *reinterpret_cast<float4*>(rb + 0 * H + col) =
        make_float4(na[0][0], na[0][1], na[0][2], na[0][3]);
    *reinterpret_cast<float4*>(rb + 1 * H + col) =
        make_float4(na[1][0], na[1][1], na[1][2], na[1][3]);
    *reinterpret_cast<float4*>(rb + 2 * H + col) =
        make_float4(na[2][0], na[2][1], na[2][2], na[2][3]);
    *reinterpret_cast<float4*>(rb + 3 * H + col) =
        make_float4(na[3][0], na[3][1], na[3][2], na[3][3]);
    coll_v[g] = make_float4(ca[0], ca[1], ca[2], ca[3]);
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    csq_acc += __shfl_xor_sync(0xffffffffu, csq_acc, o);
    if constexpr (MIX) sq_acc += __shfl_xor_sync(0xffffffffu, sq_acc, o);
  }
  if (lane == 0) {
    s_red[0][warp] = csq_acc;
    if constexpr (MIX) s_red[1][warp] = sq_acc;
  }
  __syncthreads();
  if (warp == 0) {
    float c_sum = (lane < 8) ? s_red[0][lane] : 0.0f;
    float s_sum = (MIX && lane < 8) ? s_red[1][lane] : 0.0f;
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) {
      c_sum += __shfl_xor_sync(0xffffffffu, c_sum, o);
      if constexpr (MIX) s_sum += __shfl_xor_sync(0xffffffffu, s_sum, o);
    }
    if (lane == 0) {
      s_inv = rsqrtf(c_sum * (1.0f / static_cast<float>(H)) + norm_eps);
      if constexpr (MIX) sq_out[t] = s_sum;
    }
  }
  __syncthreads();
  const float inv = s_inv;
  __half* out_row = out + static_cast<size_t>(t) * H;
#pragma unroll
  for (int g = 0; g < 5; ++g) {
    const int col = g * 1024 + tid * 4;
    const float4 nw = __ldg(reinterpret_cast<const float4*>(norm_w + col));
    const __half2 h0 =
        __floats2half2_rn(coll_v[g].x * inv * nw.x, coll_v[g].y * inv * nw.y);
    const __half2 h1 =
        __floats2half2_rn(coll_v[g].z * inv * nw.z, coll_v[g].w * inv * nw.w);
    uint2 out_u2;
    out_u2.x = *reinterpret_cast<const unsigned*>(&h0);
    out_u2.y = *reinterpret_cast<const unsigned*>(&h1);
    *reinterpret_cast<uint2*>(out_row + col) = out_u2;
  }
}

__global__ void __launch_bounds__(256)
    hc_sinkhorn_kernel(const float* __restrict__ dots,
                       const float* __restrict__ sq_in,
                       const float* __restrict__ scale,
                       const float* __restrict__ base,
                       float* __restrict__ pre_o, float* __restrict__ post_o,
                       float* __restrict__ comb_o, int T, float norm_eps,
                       float hc_eps, int iters) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int t = blockIdx.x * 8 + warp;
  if (t >= T) return;
  const float rs =
      rsqrtf(__ldg(sq_in + t) * (1.0f / (4.0f * static_cast<float>(HC_H))) +
             norm_eps);
  const float s0 = __ldg(scale + 0), s1 = __ldg(scale + 1),
              s2 = __ldg(scale + 2);
  const float* drow = dots + static_cast<size_t>(t) * 24;
  if (lane < 4) {
    const float p = __ldg(drow + lane) * rs * s0 + __ldg(base + lane);
    pre_o[static_cast<size_t>(t) * 4 + lane] =
        1.0f / (1.0f + expf(-p)) + hc_eps;
    const float q = __ldg(drow + 4 + lane) * rs * s1 + __ldg(base + 4 + lane);
    post_o[static_cast<size_t>(t) * 4 + lane] = 2.0f / (1.0f + expf(-q));
  }
  float val = (lane < 16)
                  ? (__ldg(drow + 8 + lane) * rs * s2 + __ldg(base + 8 + lane))
                  : 0.0f;
  float rmax = val;
  rmax = fmaxf(rmax, __shfl_xor_sync(0xffffffffu, rmax, 1));
  rmax = fmaxf(rmax, __shfl_xor_sync(0xffffffffu, rmax, 2));
  val = (lane < 16) ? expf(val - rmax) : 0.0f;
  float rsum = val;
  rsum += __shfl_xor_sync(0xffffffffu, rsum, 1);
  rsum += __shfl_xor_sync(0xffffffffu, rsum, 2);
  val = (lane < 16) ? (val / rsum + hc_eps) : 0.0f;
  float csum = val;
  csum += __shfl_xor_sync(0xffffffffu, csum, 4);
  csum += __shfl_xor_sync(0xffffffffu, csum, 8);
  val = (lane < 16) ? (val / (csum + hc_eps)) : 0.0f;
  for (int it = 1; it < iters; ++it) {
    rsum = val;
    rsum += __shfl_xor_sync(0xffffffffu, rsum, 1);
    rsum += __shfl_xor_sync(0xffffffffu, rsum, 2);
    val = (lane < 16) ? (val / (rsum + hc_eps)) : 0.0f;
    csum = val;
    csum += __shfl_xor_sync(0xffffffffu, csum, 4);
    csum += __shfl_xor_sync(0xffffffffu, csum, 8);
    val = (lane < 16) ? (val / (csum + hc_eps)) : 0.0f;
  }
  if (lane < 16) comb_o[static_cast<size_t>(t) * 16 + lane] = val;
}

void hc_expand_collapse_norm(torch::Tensor residual, torch::Tensor y,
                             torch::Tensor post, torch::Tensor comb,
                             torch::Tensor cpre, torch::Tensor norm_w,
                             torch::Tensor out, torch::Tensor sq_out, bool mix,
                             double norm_eps) {
  const int T = residual.size(0);
  TORCH_CHECK(residual.is_cuda() && residual.scalar_type() == torch::kFloat32 &&
              residual.is_contiguous() && residual.size(1) == 4 &&
              residual.size(2) == HC_H);
  TORCH_CHECK(y.is_contiguous() && y.numel() == static_cast<int64_t>(T) * HC_H);
  TORCH_CHECK(post.is_contiguous() && comb.is_contiguous() &&
              cpre.is_contiguous() && norm_w.is_contiguous() &&
              out.is_contiguous() && out.scalar_type() == torch::kFloat16);
  TORCH_CHECK(!mix ||
              (sq_out.is_contiguous() &&
               sq_out.scalar_type() == torch::kFloat32 && sq_out.numel() >= T));
  if (T == 0) return;
  const at::cuda::OptionalCUDAGuard guard(device_of(residual));
  auto stream = at::cuda::getCurrentCUDAStream();
  float* sq_ptr = mix ? sq_out.data_ptr<float>() : nullptr;
#define HCECN(TYv, MIXv)                                                       \
  hc_expand_collapse_norm_kernel<TYv, MIXv><<<T, 256, 0, stream>>>(            \
      residual.data_ptr<float>(), reinterpret_cast<const TYv*>(y.data_ptr()),  \
      post.data_ptr<float>(), comb.data_ptr<float>(), cpre.data_ptr<float>(),  \
      norm_w.data_ptr<float>(), reinterpret_cast<__half*>(out.data_ptr()),     \
      sq_ptr, static_cast<float>(norm_eps))
  if (y.scalar_type() == torch::kFloat16) {
    if (mix)
      HCECN(__half, true);
    else
      HCECN(__half, false);
  } else {
    TORCH_CHECK(y.scalar_type() == torch::kFloat32);
    if (mix)
      HCECN(float, true);
    else
      HCECN(float, false);
  }
#undef HCECN
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void hc_sinkhorn(torch::Tensor dots, torch::Tensor sq_in, torch::Tensor scale,
                 torch::Tensor base, torch::Tensor pre_o, torch::Tensor post_o,
                 torch::Tensor comb_o, double norm_eps, double hc_eps,
                 int64_t iters) {
  const int T = dots.size(0);
  TORCH_CHECK(dots.is_cuda() && dots.scalar_type() == torch::kFloat32 &&
              dots.is_contiguous() && dots.size(1) == 24);
  TORCH_CHECK(sq_in.is_contiguous() && scale.is_contiguous() &&
              base.is_contiguous() && pre_o.is_contiguous() &&
              post_o.is_contiguous() && comb_o.is_contiguous());
  if (T == 0) return;
  const at::cuda::OptionalCUDAGuard guard(device_of(dots));
  auto stream = at::cuda::getCurrentCUDAStream();
  hc_sinkhorn_kernel<<<(T + 7) / 8, 256, 0, stream>>>(
      dots.data_ptr<float>(), sq_in.data_ptr<float>(), scale.data_ptr<float>(),
      base.data_ptr<float>(), pre_o.data_ptr<float>(), post_o.data_ptr<float>(),
      comb_o.data_ptr<float>(), T, static_cast<float>(norm_eps),
      static_cast<float>(hc_eps), static_cast<int>(iters));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ll_arm(int64_t ptr, int64_t bytes) {
  C10_CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(ptr), 0xFF, bytes));
}

// ---- V4.1 batched index scores over the paged index-key cache ---------------
// out[t, n] = fp16(sum_h fp16(fp16(relu(fp16(q[t,h] . k[n]))) * w[t,h])) for
// every token t of request row r and key n < (last position + 1) / ratio. One
// CTA (4 warps) per (request, key-tile stride): the row's T x H query rows are
// staged once in shared memory and each 32-key tile is one WMMA pass (rows
// padded to 16, keys read once). Replaces the Triton _bindex_scores (same
// rounding points; FP32 accumulation order differs).
namespace bidx {
constexpr int D = 128, KT = 32, QLD = D + 8, KLD = D + 8, SLD = KT + 4;
template <int ROWS>
constexpr int smem_bytes() {
  return (ROWS * QLD + KT * KLD) * 2 + ROWS * SLD * 4 + ROWS * 4;
}
}  // namespace bidx

// ROWS: query rows (tokens x heads) one CTA stages. nshard > 1 (the
// key-sharded indexer): key tile t belongs to shard t % nshard; this CTA scores
// only its shard's tiles, all heads, and writes them as compact columns
// (t / nshard) * KT + j of out.
template <int ROWS>
__global__ void __launch_bounds__(128)
    index_scores_kernel(__half* __restrict__ out, const __half* __restrict__ q,
                        const __half* __restrict__ w,
                        const __half* __restrict__ cache,
                        const int* __restrict__ table,
                        const int64_t* __restrict__ positions,
                        const int* __restrict__ qsl, int table_stride,
                        int n_out, int64_t page_stride, int keys_per_page,
                        int ratio, int H, int shard, int nshard, bool packed) {
  using namespace bidx;
  using namespace nvcuda;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  __half* qs = reinterpret_cast<__half*>(smem_raw);
  __half* ks = qs + ROWS * QLD;
  float* sc = reinterpret_cast<float*>(ks + KT * KLD);
  float* ws = sc + ROWS * SLD;
  const int r = blockIdx.x;
  const int first = qsl[r], count = qsl[r + 1] - first;
  if (count <= 0) return;
  const int64_t end = (positions[first + count - 1] + 1) / ratio;
  if (((int64_t)blockIdx.y * nshard + shard) * KT >= end) return;
  const int rows = count * H, mtiles = (rows + 15) / 16;
  if (rows > ROWS) return;  // callers keep decode rows within ROWS / H
  // Stage the request's query rows (zero padding) and weights.
  for (int i = threadIdx.x; i < mtiles * 16 * (D / 8); i += blockDim.x) {
    const int row = i / (D / 8), c = (i % (D / 8)) * 8;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (row < rows)
      v = __ldg(reinterpret_cast<const uint4*>(
          q + ((size_t)first * H + row) * D + c));
    *reinterpret_cast<uint4*>(qs + row * QLD + c) = v;
  }
  for (int i = threadIdx.x; i < rows; i += blockDim.x)
    ws[i] = __half2float(w[(size_t)first * H + i]);
  const int* trow = table + (size_t)r * table_stride;
  const int warp = threadIdx.x >> 5;
  for (int64_t local = blockIdx.y;; local += gridDim.y) {
    const int64_t n0 = (local * nshard + shard) * KT;
    if (n0 >= end) break;
    __syncthreads();  // qs ready / previous tile's sc consumed
    for (int i = threadIdx.x; i < KT * (D / 8); i += blockDim.x) {
      const int j = i / (D / 8), c = (i % (D / 8)) * 8;
      const int64_t n = n0 + j;
      uint4 v = make_uint4(0, 0, 0, 0);
      if (n < end) {
        const int page = trow[n / keys_per_page];
        if (page >= 0 && packed)  // FP4 index rows, page_stride in bytes
          v = v41_ikey8(reinterpret_cast<const unsigned char*>(cache) +
                            (int64_t)page * page_stride +
                            (n % keys_per_page) * V41_I_ROW,
                        c);
        else if (page >= 0)
          v = __ldcg(reinterpret_cast<const uint4*>(
              cache + (int64_t)page * page_stride + (n % keys_per_page) * D +
              c));
      }
      *reinterpret_cast<uint4*>(ks + j * KLD + c) = v;
    }
    __syncthreads();
    for (int job = warp; job < mtiles * 2; job += 4) {
      const int mt = job >> 1, nt = job & 1;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
      wmma::fill_fragment(acc, 0.f);
#pragma unroll
      for (int d = 0; d < D; d += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
        wmma::load_matrix_sync(a, qs + mt * 16 * QLD + d, QLD);
        wmma::load_matrix_sync(b, ks + nt * 16 * KLD + d, KLD);
        wmma::mma_sync(acc, a, b, acc);
      }
      wmma::store_matrix_sync(sc + mt * 16 * SLD + nt * 16, acc, SLD,
                              wmma::mem_row_major);
    }
    __syncthreads();
    const int64_t column0 = local * KT;
    for (int i = threadIdx.x; i < count * KT; i += blockDim.x) {
      const int t = i / KT, j = i % KT;
      if (n0 + j >= end) continue;
      float total = 0.f;
      for (int h = 0; h < H; ++h) {
        float v = __half2float(__float2half(sc[(t * H + h) * SLD + j]));
        v = fmaxf(v, 0.f);
        total += __half2float(__float2half(v * ws[t * H + h]));
      }
      out[(size_t)(first + t) * n_out + column0 + j] = __float2half(total);
    }
  }
}

template <int ROWS>
static void index_scores_launch(torch::Tensor out, torch::Tensor q,
                                torch::Tensor w, torch::Tensor cache,
                                torch::Tensor table, torch::Tensor positions,
                                torch::Tensor qsl, int64_t num_reqs,
                                int64_t ratio, int64_t ctas, int64_t shard,
                                int64_t nshard) {
  using namespace bidx;
  const int H = q.size(1);
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kFloat16 &&
              q.is_contiguous() && q.size(2) == D);
  TORCH_CHECK(w.scalar_type() == torch::kFloat16 && w.is_contiguous() &&
              w.size(0) == q.size(0) && w.size(1) == H);
  const bool packed = cache.scalar_type() == torch::kUInt8;
  TORCH_CHECK(cache.dim() == 3 && cache.stride(2) == 1 &&
              (packed ? cache.size(2) == V41_I_ROW &&
                            cache.stride(1) == V41_I_ROW
                      : cache.scalar_type() == torch::kFloat16 &&
                            cache.size(2) == D && cache.stride(1) == D));
  TORCH_CHECK(table.scalar_type() == torch::kInt32 && table.stride(1) == 1);
  TORCH_CHECK(positions.scalar_type() == torch::kInt64 &&
              qsl.scalar_type() == torch::kInt32);
  TORCH_CHECK(out.scalar_type() == torch::kFloat16 && out.is_contiguous());
  TORCH_CHECK(0 <= shard && shard < nshard);
  if (num_reqs <= 0) return;
  const at::cuda::OptionalCUDAGuard guard(device_of(q));
  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int SMEM = smem_bytes<ROWS>();
  static bool smem_set = false;
  if (!smem_set) {
    cudaFuncSetAttribute(index_scores_kernel<ROWS>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    smem_set = true;
  }
  index_scores_kernel<ROWS><<<dim3(num_reqs, ctas), 128, SMEM, stream>>>(
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(w.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(cache.data_ptr()),
      table.data_ptr<int>(), positions.data_ptr<int64_t>(), qsl.data_ptr<int>(),
      (int)table.stride(0), (int)out.size(1), cache.stride(0),
      (int)cache.size(1), (int)ratio, H, (int)shard, (int)nshard, packed);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Every request may have at most 64 / H query rows (8 at H = 8).
void index_scores(torch::Tensor out, torch::Tensor q, torch::Tensor w,
                  torch::Tensor cache, torch::Tensor table,
                  torch::Tensor positions, torch::Tensor qsl, int64_t num_reqs,
                  int64_t ratio, int64_t ctas) {
  TORCH_CHECK(8 * q.size(1) <= 64);
  index_scores_launch<64>(out, q, w, cache, table, positions, qsl, num_reqs,
                          ratio, ctas, 0, 1);
}

// Key-sharded: all H heads (32), at most 192 / H query rows (6) per request.
void index_scores_shard(torch::Tensor out, torch::Tensor q, torch::Tensor w,
                        torch::Tensor cache, torch::Tensor table,
                        torch::Tensor positions, torch::Tensor qsl,
                        int64_t num_reqs, int64_t ratio, int64_t ctas,
                        int64_t shard, int64_t nshard) {
  index_scores_launch<192>(out, q, w, cache, table, positions, qsl, num_reqs,
                           ratio, ctas, shard, nshard);
}

// ---- Prefill expert GEMM (VLLM_V41_MOE_PF_GEMM) ------------------------------
// out[row] = x[row] @ W_e^T for the rows of every expert group (rows sorted by
// group: offsets[g] .. offsets[g + 1] use local expert experts[g]), with the
// weights decoded exactly as the skinny kernels decode them (dequant8_tm on
// the prepacked codes, block scale x global scale x 2^14 in FP16) into shared
// memory and multiplied on WMMA 16x16x16 tiles with FP32 accumulation, so only
// the summation order differs from the skinny path. A CTA (8 warps, 2 x 4 of
// 32 x 32) computes a 64-row x 128-column tile of one group, K in steps of 32
// (one scale block), double-buffered: the next step's activations and code
// words are loaded into registers while this step multiplies. Hot experts'
// rows spread over many CTAs (the skinny kernel runs all of an expert's rows
// in one CTA per column tile). Output rows are written at out_rows[row]
// (slot-major for the route sum) or in place when out_rows is null.
namespace pfg {
constexpr int BN = 128, BK = 32, ALD = BK + 8, BLD = BK + 8, CLD = BN + 4;
template <int BM>
constexpr int stage_bytes() { return (BM * ALD + BN * BLD) * 2; }
template <int BM>
constexpr int smem_bytes() {
  constexpr int s2 = 2 * stage_bytes<BM>();
  constexpr int cs = BM * CLD * 4;
  return s2 > cs ? s2 : cs;
}
}  // namespace pfg

__device__ __forceinline__ unsigned pf_code_word(const uint8_t* codes_e,
                                                 int tile, int K, int c,
                                                 int kc) {
  // Word holding k = 8 kc .. 8 kc + 7 of column c (0..31) of an N/32 tile.
#if V41SK_TM_CODES
  return __ldg(reinterpret_cast<const unsigned*>(codes_e) +
               ((size_t)tile * (K >> 3) + kc) * 32 + c);
#else
  const int lane = (((c >> 3) & 3) << 2) | (c & 3) | (((c >> 2) & 1) << 4);
  return __ldg(reinterpret_cast<const unsigned*>(codes_e) +
               (((size_t)tile * (K >> 4) + (kc >> 1)) * 32 + lane) * 2 +
               (kc & 1));
#endif
}

template <int BM, bool ACCUM>
__global__ void __launch_bounds__((BM / 32) * 128)
    moe_pf_gemm_t_kernel(void* __restrict__ out, const half* __restrict__ x,
                         const int* __restrict__ a_rows, int a_div,
                         const int* __restrict__ out_rows, int out_div,
                         const uint8_t* __restrict__ codes,
                         const uint8_t* __restrict__ qscales,
                         const float* __restrict__ gscales,
                         const int* __restrict__ offsets,
                         const int* __restrict__ experts, int groups, int N,
                         int K, int min_rows) {
  using namespace pfg;
  using namespace nvcuda;
  constexpr int STAGE = stage_bytes<BM>();
  extern __shared__ __align__(16) unsigned char smem_raw[];
  float* Cs = reinterpret_cast<float*>(smem_raw);
  const int n0 = blockIdx.x * BN;
  const int tid = threadIdx.x, warp = tid >> 5;
  const int wm = warp >> 2, wn = warp & 3;
  const int ar = tid >> 2, ac = (tid & 3) * 8;
  const int bn = (BM == 32) ? tid : (tid >> 1);
  const int bw = (BM == 32) ? 0 : ((tid & 1) * 2);
  const int ncol = n0 + bn, ntile = ncol >> 5, nc = ncol & 31;
  const int nlane = (((nc >> 3) & 3) << 2) | (nc & 3) | (((nc >> 2) & 1) << 4);
  for (int tile = blockIdx.y;; tile += gridDim.y) {
    int t = tile, g = 0, row0 = -1, rows = 0;
    for (; g < groups; ++g) {
      const int a = offsets[g], b = offsets[g + 1];
      const int tiles = b - a > min_rows ? (b - a + BM - 1) / BM : 0;
      if (t < tiles) {
        row0 = a + t * BM;
        rows = min(BM, b - row0);
        break;
      }
      t -= tiles;
    }
    if (row0 < 0) return;
    const int e = experts[g];
    const uint8_t* codes_e = codes + (size_t)e * N * (K / 2);
    const uint8_t* scales_e = qscales + (size_t)e * N * (K / 32);
    const half2 gm2 = __float2half2_rn(gscales[e] * 16384.f);
    const half* xrow =
        ar < rows
            ? x + (size_t)(a_rows ? a_rows[row0 + ar] / a_div : row0 + ar) * K + ac
            : nullptr;
    const uint8_t* sptr = scales_e + (size_t)ntile * (K >> 5) * 32 + nlane;
    uint4 ra;
    unsigned rw0, rw1, rw2 = 0, rw3 = 0;
    uint8_t rs;
    auto load = [&](int k0) {
      ra = xrow ? __ldg(reinterpret_cast<const uint4*>(xrow + k0))
                : make_uint4(0, 0, 0, 0);
      rw0 = pf_code_word(codes_e, ntile, K, nc, (k0 >> 3) + bw);
      rw1 = pf_code_word(codes_e, ntile, K, nc, (k0 >> 3) + bw + 1);
      if constexpr (BM == 32) {
        rw2 = pf_code_word(codes_e, ntile, K, nc, (k0 >> 3) + 2);
        rw3 = pf_code_word(codes_e, ntile, K, nc, (k0 >> 3) + 3);
      }
      rs = __ldg(sptr + (size_t)(k0 >> 5) * 32);
    };
    auto store = [&](int stage) {
      half* As = reinterpret_cast<half*>(smem_raw + stage * STAGE);
      half* Bs = As + BM * ALD;
      *reinterpret_cast<uint4*>(As + ar * ALD + ac) = ra;
      const half2 sc2 = __hmul2(e8m0_to_half2(rs), gm2);
      if constexpr (BM == 32) {
        half2 h[16];
        dequant8_tm(rw0, sc2, h);
        dequant8_tm(rw1, sc2, h + 4);
        dequant8_tm(rw2, sc2, h + 8);
        dequant8_tm(rw3, sc2, h + 12);
        *reinterpret_cast<uint4*>(Bs + bn * BLD) = *reinterpret_cast<uint4*>(h);
        *reinterpret_cast<uint4*>(Bs + bn * BLD + 8) =
            *reinterpret_cast<uint4*>(h + 4);
        *reinterpret_cast<uint4*>(Bs + bn * BLD + 16) =
            *reinterpret_cast<uint4*>(h + 8);
        *reinterpret_cast<uint4*>(Bs + bn * BLD + 24) =
            *reinterpret_cast<uint4*>(h + 12);
      } else {
        half2 h[8];
        dequant8_tm(rw0, sc2, h);
        dequant8_tm(rw1, sc2, h + 4);
        *reinterpret_cast<uint4*>(Bs + bn * BLD + bw * 8) =
            *reinterpret_cast<uint4*>(h);
        *reinterpret_cast<uint4*>(Bs + bn * BLD + bw * 8 + 8) =
            *reinterpret_cast<uint4*>(h + 4);
      }
    };
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.f);
    load(0);
    store(0);
    __syncthreads();
    int stage = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
      const bool more = k0 + BK < K;
      if (more) load(k0 + BK);
      const half* As = reinterpret_cast<const half*>(smem_raw + stage * STAGE);
      const half* Bs = As + BM * ALD;
#pragma unroll
      for (int kk = 0; kk < BK; kk += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[2];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b[2];
#pragma unroll
        for (int i = 0; i < 2; ++i)
          wmma::load_matrix_sync(a[i], As + (wm * 32 + i * 16) * ALD + kk, ALD);
#pragma unroll
        for (int j = 0; j < 2; ++j)
          wmma::load_matrix_sync(b[j], Bs + (wn * 32 + j * 16) * BLD + kk, BLD);
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
          for (int j = 0; j < 2; ++j)
            wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
      }
      if (more) store(stage ^ 1);
      __syncthreads();
      stage ^= 1;
    }
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 2; ++j)
        wmma::store_matrix_sync(Cs + (wm * 32 + i * 16) * CLD + wn * 32 + j * 16,
                                acc[i][j], CLD, wmma::mem_row_major);
    __syncthreads();
    if constexpr (ACCUM) {
      float* out_f = reinterpret_cast<float*>(out);
      for (int i = tid; i < rows * (BN / 4); i += blockDim.x) {
        const int r = i / (BN / 4), c = (i % (BN / 4)) * 4;
        const int dst = out_rows[row0 + r] / out_div;
        float* dp = out_f + (size_t)dst * N + n0 + c;
#pragma unroll
        for (int q = 0; q < 4; ++q)
          atomicAdd(dp + q, __half2float(__float2half(Cs[r * CLD + c + q])));
      }
    } else {
      half* out_h = reinterpret_cast<half*>(out);
      for (int i = tid; i < rows * (BN / 8); i += blockDim.x) {
        const int r = i / (BN / 8), c = (i % (BN / 8)) * 8;
        const int dst = out_rows ? out_rows[row0 + r] : row0 + r;
        half h[8];
#pragma unroll
        for (int q = 0; q < 8; ++q) h[q] = __float2half(Cs[r * CLD + c + q]);
        *reinterpret_cast<uint4*>(out_h + (size_t)dst * N + n0 + c) =
            *reinterpret_cast<uint4*>(h);
      }
    }
    __syncthreads();
  }
}

__global__ void __launch_bounds__(256)
    moe_pf_swiglu_kernel(half* __restrict__ activated,
                         const half* __restrict__ gate_up,
                         const float* __restrict__ probs,
                         const int* __restrict__ perm,
                         const int* __restrict__ offsets, int groups,
                         int inter) {
  const int total = offsets[groups];
  for (int r = blockIdx.x; r < total; r += gridDim.x) {
    const float prob = probs[perm[r]];
    const half* gu = gate_up + (size_t)r * (2 * inter);
    half* act = activated + (size_t)r * inter;
    for (int c = threadIdx.x * 2; c < inter; c += 512) {
      const half2 g2 = *reinterpret_cast<const half2*>(gu + c);
      const half2 u2 = *reinterpret_cast<const half2*>(gu + inter + c);
      const float2 gf = __half22float2(g2);
      const float2 uf = __half22float2(u2);
      const float g0 = fminf(gf.x, 10.f), g1 = fminf(gf.y, 10.f);
      const float u0 = fminf(fmaxf(uf.x, -10.f), 10.f);
      const float u1 = fminf(fmaxf(uf.y, -10.f), 10.f);
      const float v0 = (g0 / (1.f + expf(-g0))) * u0 * prob;
      const float v1 = (g1 / (1.f + expf(-g1))) * u1 * prob;
      *reinterpret_cast<half2*>(act + c) = __floats2half2_rn(v0, v1);
    }
  }
}

void moe_pf_swiglu(torch::Tensor activated, torch::Tensor gate_up,
                   torch::Tensor probs, torch::Tensor perm,
                   torch::Tensor offsets, int64_t groups) {
  const int inter = activated.size(1);
  TORCH_CHECK(activated.is_cuda() && activated.scalar_type() == torch::kFloat16 &&
              activated.is_contiguous() && inter % 2 == 0);
  TORCH_CHECK(gate_up.scalar_type() == torch::kFloat16 &&
              gate_up.is_contiguous() && gate_up.size(1) == 2 * inter);
  TORCH_CHECK(probs.scalar_type() == torch::kFloat32 && probs.is_contiguous());
  TORCH_CHECK(perm.scalar_type() == torch::kInt32 && perm.is_contiguous() &&
              offsets.scalar_type() == torch::kInt32 && offsets.is_contiguous());
  if (groups <= 0 || gate_up.size(0) == 0) return;
  const at::cuda::OptionalCUDAGuard guard(device_of(gate_up));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int ctas = (int)std::min<int64_t>(320, std::max<int64_t>(1, gate_up.size(0)));
  moe_pf_swiglu_kernel<<<ctas, 256, 0, stream>>>(
      reinterpret_cast<half*>(activated.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(gate_up.data_ptr<at::Half>()),
      probs.data_ptr<float>(), perm.data_ptr<int>(), offsets.data_ptr<int>(),
      (int)groups, inter);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void moe_pf_gemm(torch::Tensor out, torch::Tensor x, torch::Tensor a_rows,
                 int64_t a_div, torch::Tensor out_rows,
                 torch::Tensor codes, torch::Tensor qscales,
                 torch::Tensor gscales, torch::Tensor offsets,
                 torch::Tensor experts, int64_t groups, int64_t row_tiles,
                 int64_t min_rows, int64_t bm) {
  using namespace pfg;
  const int K = x.size(1), N = out.size(1);
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16 &&
              x.is_contiguous() && K % BK == 0);
  TORCH_CHECK(out.scalar_type() == torch::kFloat16 && out.is_contiguous() &&
              N % BN == 0);
  TORCH_CHECK(codes.scalar_type() == torch::kUInt8 &&
              qscales.scalar_type() == torch::kUInt8 &&
              gscales.scalar_type() == torch::kFloat32);
  TORCH_CHECK(offsets.scalar_type() == torch::kInt32 &&
              experts.scalar_type() == torch::kInt32);
  const bool rows_given = out_rows.numel() > 0;
  const bool a_given = a_rows.numel() > 0;
  TORCH_CHECK(!a_given || (a_rows.scalar_type() == torch::kInt32 &&
                           a_rows.is_contiguous() && a_div > 0));
  TORCH_CHECK(!rows_given || (out_rows.scalar_type() == torch::kInt32 &&
                              out_rows.is_contiguous()));
  TORCH_CHECK(bm == 32 || bm == 64);
  if (row_tiles <= 0 || groups <= 0) return;
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream();
  static bool smem_set = false;
  if (!smem_set) {
    cudaFuncSetAttribute(moe_pf_gemm_t_kernel<32, false>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_bytes<32>());
    cudaFuncSetAttribute(moe_pf_gemm_t_kernel<64, false>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_bytes<64>());
    smem_set = true;
  }
  if (bm == 32) {
    moe_pf_gemm_t_kernel<32, false>
        <<<dim3(N / BN, row_tiles), 128, smem_bytes<32>(), stream>>>(
            out.data_ptr(), reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
            a_given ? a_rows.data_ptr<int>() : nullptr, (int)a_div,
            rows_given ? out_rows.data_ptr<int>() : nullptr, 1,
            codes.data_ptr<uint8_t>(), qscales.data_ptr<uint8_t>(),
            gscales.data_ptr<float>(), offsets.data_ptr<int>(),
            experts.data_ptr<int>(), (int)groups, N, K, (int)min_rows);
  } else {
    moe_pf_gemm_t_kernel<64, false>
        <<<dim3(N / BN, row_tiles), 256, smem_bytes<64>(), stream>>>(
            out.data_ptr(), reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
            a_given ? a_rows.data_ptr<int>() : nullptr, (int)a_div,
            rows_given ? out_rows.data_ptr<int>() : nullptr, 1,
            codes.data_ptr<uint8_t>(), qscales.data_ptr<uint8_t>(),
            gscales.data_ptr<float>(), offsets.data_ptr<int>(),
            experts.data_ptr<int>(), (int)groups, N, K, (int)min_rows);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void moe_pf_gemm_accum(torch::Tensor out_f32, torch::Tensor x,
                       torch::Tensor perm, int64_t out_div,
                       torch::Tensor codes, torch::Tensor qscales,
                       torch::Tensor gscales, torch::Tensor offsets,
                       torch::Tensor experts, int64_t groups,
                       int64_t row_tiles, int64_t bm) {
  using namespace pfg;
  const int K = x.size(1), N = out_f32.size(1);
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16 &&
              x.is_contiguous() && K % BK == 0);
  TORCH_CHECK(out_f32.scalar_type() == torch::kFloat32 &&
              out_f32.is_contiguous() && N % BN == 0);
  TORCH_CHECK(perm.scalar_type() == torch::kInt32 && perm.is_contiguous() &&
              out_div > 0);
  TORCH_CHECK(codes.scalar_type() == torch::kUInt8 &&
              qscales.scalar_type() == torch::kUInt8 &&
              gscales.scalar_type() == torch::kFloat32);
  TORCH_CHECK(offsets.scalar_type() == torch::kInt32 &&
              experts.scalar_type() == torch::kInt32);
  TORCH_CHECK(bm == 32 || bm == 64);
  if (row_tiles <= 0 || groups <= 0) return;
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream();
  static bool smem_set = false;
  if (!smem_set) {
    cudaFuncSetAttribute(moe_pf_gemm_t_kernel<32, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_bytes<32>());
    cudaFuncSetAttribute(moe_pf_gemm_t_kernel<64, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_bytes<64>());
    smem_set = true;
  }
  if (bm == 32) {
    moe_pf_gemm_t_kernel<32, true>
        <<<dim3(N / BN, row_tiles), 128, smem_bytes<32>(), stream>>>(
            out_f32.data_ptr<float>(),
            reinterpret_cast<const half*>(x.data_ptr<at::Half>()), nullptr, 1,
            perm.data_ptr<int>(), (int)out_div, codes.data_ptr<uint8_t>(),
            qscales.data_ptr<uint8_t>(), gscales.data_ptr<float>(),
            offsets.data_ptr<int>(), experts.data_ptr<int>(), (int)groups, N, K,
            0);
  } else {
    moe_pf_gemm_t_kernel<64, true>
        <<<dim3(N / BN, row_tiles), 256, smem_bytes<64>(), stream>>>(
            out_f32.data_ptr<float>(),
            reinterpret_cast<const half*>(x.data_ptr<at::Half>()), nullptr, 1,
            perm.data_ptr<int>(), (int)out_div, codes.data_ptr<uint8_t>(),
            qscales.data_ptr<uint8_t>(), gscales.data_ptr<float>(),
            offsets.data_ptr<int>(), experts.data_ptr<int>(), (int)groups, N, K,
            0);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---- Prefill index scores (VLLM_V41_INDEX_PREFILL_CUDA) ----------------------
// index_scores_shard for one request's chunk of T tokens: the same scores and
// rounding points, but each CTA unpacks a 128-key tile (four of the shard's
// 32-key tiles) once into shared memory and scores it for a block of 64
// tokens. Each of the 8 warps takes one token at a time: its 32 head rows'
// WMMA A fragments are read from L2 once per key tile and kept in registers;
// per 64 keys the 16-head halves are stored and summed in turn (h ascending),
// so the WMMA chain (d = 0..127 by 16) and the head sum order match
// index_scores_kernel and the scores are bitwise equal.
namespace pidx {
constexpr int D = 128, H = 32, KT = 128, TB = 64, WARPS = 8, KLD = D + 8,
              SLD = 64 + 4;
constexpr int SMEM = KT * KLD * 2 + WARPS * 16 * SLD * 4;
}  // namespace pidx

__global__ void __launch_bounds__(256)
    index_prefill_kernel(__half* __restrict__ out, const __half* __restrict__ q,
                         const __half* __restrict__ w,
                         const __half* __restrict__ cache,
                         const int* __restrict__ trow,
                         const int64_t* __restrict__ positions, int T,
                         int n_out, int64_t page_stride, int keys_per_page,
                         int ratio, int shard, int nshard, bool packed) {
  using namespace pidx;
  using namespace nvcuda;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  __half* ks = reinterpret_cast<__half*>(smem_raw);
  float* sc = reinterpret_cast<float*>(ks + KT * KLD) +
              (threadIdx.x >> 5) * 16 * SLD;
  const int t0 = blockIdx.x * TB, count = min(TB, T - t0);
  if (count <= 0) return;
  // Keys up to the block's last token (later tokens mask by own length).
  const int64_t end = (positions[t0 + count - 1] + 1) / ratio;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (int64_t quad = blockIdx.y;; quad += gridDim.y) {
    const int64_t local0 = quad * 4;  // the shard's 32-key tiles local0..+3
    if ((local0 * nshard + shard) * 32 >= end) break;
    __syncthreads();  // previous tile consumed
    for (int i = threadIdx.x; i < KT * (D / 8); i += blockDim.x) {
      const int jj = i / (D / 8), c = (i % (D / 8)) * 8;
      const int64_t n = ((local0 + jj / 32) * nshard + shard) * 32 + jj % 32;
      uint4 v = make_uint4(0, 0, 0, 0);
      if (n < end) {
        const int page = trow[n / keys_per_page];
        if (page >= 0 && packed)
          v = v41_ikey8(reinterpret_cast<const unsigned char*>(cache) +
                            (int64_t)page * page_stride +
                            (n % keys_per_page) * V41_I_ROW,
                        c);
        else if (page >= 0)
          v = __ldcg(reinterpret_cast<const uint4*>(
              cache + (int64_t)page * page_stride + (n % keys_per_page) * D +
              c));
      }
      *reinterpret_cast<uint4*>(ks + jj * KLD + c) = v;
    }
    __syncthreads();
    for (int t = warp; t < count; t += WARPS) {
      const __half* qt = q + (size_t)(t0 + t) * H * D;
      const __half* wt = w + (size_t)(t0 + t) * H;
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>
          a[2][D / 16];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int k = 0; k < D / 16; ++k)
          wmma::load_matrix_sync(a[mt][k], qt + mt * 16 * D + k * 16, D);
#pragma unroll
      for (int half_ = 0; half_ < KT / 64; ++half_) {
        float total[2] = {0.f, 0.f};
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
          for (int nt = 0; nt < 4; ++nt) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.f);
#pragma unroll
            for (int k = 0; k < D / 16; ++k) {
              wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major>
                  b;
              wmma::load_matrix_sync(
                  b, ks + (half_ * 64 + nt * 16) * KLD + k * 16, KLD);
              wmma::mma_sync(acc, a[mt][k], b, acc);
            }
            wmma::store_matrix_sync(sc + nt * 16, acc, SLD,
                                    wmma::mem_row_major);
          }
          __syncwarp();
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int j = lane + 32 * e;
            for (int h = 0; h < 16; ++h) {
              float v = __half2float(__float2half(sc[h * SLD + j]));
              v = fmaxf(v, 0.f);
              total[e] += __half2float(
                  __float2half(v * __half2float(wt[mt * 16 + h])));
            }
          }
          __syncwarp();
        }
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int jj = half_ * 64 + lane + 32 * e;
          const int64_t n = ((local0 + jj / 32) * nshard + shard) * 32 + lane;
          if (n < end)
            out[(size_t)(t0 + t) * n_out + local0 * 32 + jj] =
                __float2half(total[e]);
        }
      }
    }
  }
}

// out [T, n_out] (this shard's compact columns), q [T, 32, 128], w [T, 32],
// one request's block table row and the chunk's positions.
void index_scores_prefill(torch::Tensor out, torch::Tensor q, torch::Tensor w,
                          torch::Tensor cache, torch::Tensor table,
                          torch::Tensor positions, int64_t ratio, int64_t ctas,
                          int64_t shard, int64_t nshard) {
  using namespace pidx;
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kFloat16 &&
              q.is_contiguous() && q.size(1) == H && q.size(2) == D);
  TORCH_CHECK(w.scalar_type() == torch::kFloat16 && w.is_contiguous() &&
              w.size(0) == q.size(0) && w.size(1) == H);
  const bool packed = cache.scalar_type() == torch::kUInt8;
  TORCH_CHECK(cache.dim() == 3 && cache.stride(2) == 1 &&
              (packed ? cache.size(2) == V41_I_ROW &&
                            cache.stride(1) == V41_I_ROW
                      : cache.scalar_type() == torch::kFloat16 &&
                            cache.size(2) == D && cache.stride(1) == D));
  TORCH_CHECK(table.scalar_type() == torch::kInt32 && table.is_contiguous());
  TORCH_CHECK(positions.scalar_type() == torch::kInt64 &&
              positions.numel() == q.size(0));
  TORCH_CHECK(out.scalar_type() == torch::kFloat16 && out.is_contiguous() &&
              out.size(0) == q.size(0) && out.size(1) % 32 == 0);
  TORCH_CHECK(0 <= shard && shard < nshard);
  const int T = q.size(0);
  if (T == 0) return;
  const at::cuda::OptionalCUDAGuard guard(device_of(q));
  auto stream = at::cuda::getCurrentCUDAStream();
  static bool smem_set = false;
  if (!smem_set) {
    cudaFuncSetAttribute(index_prefill_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    smem_set = true;
  }
  index_prefill_kernel<<<dim3((T + TB - 1) / TB, ctas), 32 * WARPS, SMEM,
                         stream>>>(
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(w.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(cache.data_ptr()), table.data_ptr<int>(),
      positions.data_ptr<int64_t>(), T, (int)out.size(1), cache.stride(0),
      (int)cache.size(1), (int)ratio, (int)shard, (int)nshard, packed);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---- Router top-k + slot plan for many tokens (VLLM_V41_BATCH) --------------
// route_plan_body's per-token selection, warps looping over T tokens, then a
// stable counting sort of the T*K slots by expert (groups in expert order,
// as the Triton _plan_batch_kernel). One CTA, T*K <= 640.
constexpr int RPW_MAX = 640;

template <int EPL>
__global__ void __launch_bounds__(256)
    route_plan_wide_kernel(const float* __restrict__ logits,
                           const float* __restrict__ bias,
                           float* __restrict__ probs, int* __restrict__ ids,
                           int* __restrict__ perm, int* __restrict__ experts,
                           int* __restrict__ offsets, int T, int K,
                           float scale) {
  constexpr int E = EPL * 32;
  __shared__ int slot_ids[RPW_MAX];
  __shared__ float sh_score[8][E];
  __shared__ unsigned sh_key[8][E];
  __shared__ int cnt[E], start[E], gid[E];
  __shared__ int groups_total;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const unsigned full = 0xffffffffu;
  for (int e = threadIdx.x; e < E; e += blockDim.x) cnt[e] = 0;
  for (int t = warp; t < T; t += 8) {
    float* sc = sh_score[warp];
    unsigned* ky = sh_key[warp];
#pragma unroll
    for (int i = 0; i < EPL; ++i) {
      const int e = lane + 32 * i;
      sc[e] = logits[(size_t)t * E + e];
      ky[e] = __float_as_uint(bias[e]);
    }
#pragma unroll 1
    for (int i = 0; i < EPL; ++i) {
      const int e = lane + 32 * i;
      const float score = route_score(sc[e]);
      sc[e] = score;
      unsigned key = __float_as_uint(score + __uint_as_float(ky[e]));
      ky[e] = (key & 0x80000000u) ? ~key : (key | 0x80000000u);
    }
    __syncwarp();
    float my_v = 0.0f;
    int my_e = 0;
#pragma unroll 1
    for (int j = 0; j < K; ++j) {
      unsigned long long best = 0ull;
#pragma unroll 4
      for (int i = 0; i < EPL; ++i) {
        const unsigned u = ky[lane + 32 * i];
        const unsigned long long packed =
            ((unsigned long long)u << 32) | (0xffffffffu - (lane + 32 * i));
        best = packed > best ? packed : best;
      }
#pragma unroll
      for (int o = 16; o; o >>= 1) {
        const unsigned long long other = __shfl_xor_sync(full, best, o);
        best = other > best ? other : best;
      }
      const int best_e =
          (int)(0xffffffffu - (unsigned int)(best & 0xffffffffu));
      if (lane == j) {
        my_v = sc[best_e];
        my_e = best_e;
      }
      if (lane == (best_e & 31)) ky[best_e] = 0x007fffffu;
      __syncwarp();
    }
    float total = my_v + __shfl_xor_sync(full, my_v, 4);
    total += __shfl_xor_sync(full, total, 2);
    total += __shfl_xor_sync(full, total, 1);
    const float denom = __shfl_sync(full, total, 0) + 1e-20f;
    if (lane < K) {
      probs[t * K + lane] = rp_div(my_v, denom) * scale;
      ids[t * K + lane] = my_e;
      slot_ids[t * K + lane] = my_e;
    }
    __syncwarp();
  }
  __syncthreads();
  const int S = T * K;
  for (int s = threadIdx.x; s < S; s += blockDim.x)
    atomicAdd(&cnt[slot_ids[s]], 1);
  __syncthreads();
  if (warp == 0) {  // exclusive scans over experts: slot starts and group ids
    int run = 0, grun = 0;
    for (int base = 0; base < E; base += 32) {
      const int c = cnt[base + lane];
      int inc = c, ginc = c > 0;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int a = __shfl_up_sync(full, inc, o),
                  b = __shfl_up_sync(full, ginc, o);
        if (lane >= o) {
          inc += a;
          ginc += b;
        }
      }
      start[base + lane] = run + inc - c;
      gid[base + lane] = grun + ginc - (c > 0);
      run += __shfl_sync(full, inc, 31);
      grun += __shfl_sync(full, ginc, 31);
    }
    if (lane == 0) groups_total = grun;
  }
  __syncthreads();
  for (int s = threadIdx.x; s < S; s += blockDim.x) {
    const int e = slot_ids[s];
    int rank = 0;
    for (int j = 0; j < s; ++j) rank += slot_ids[j] == e;
    perm[start[e] + rank] = s;
  }
  for (int e = threadIdx.x; e < E; e += blockDim.x) {
    if (cnt[e] > 0) {
      experts[gid[e]] = e;
      offsets[gid[e]] = start[e];
    }
  }
  const int groups = groups_total;
  for (int j = threadIdx.x; j <= S; j += blockDim.x) {
    if (j >= groups && j < S) experts[j] = 0;
    if (j >= groups) offsets[j] = S;
  }
}

void route_plan_wide(torch::Tensor logits, torch::Tensor bias,
                     torch::Tensor probs, torch::Tensor ids, torch::Tensor perm,
                     torch::Tensor experts, torch::Tensor offsets,
                     double scale) {
  const int T = logits.size(0), E = logits.size(1), K = ids.size(1);
  TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == torch::kFloat32 &&
              logits.is_contiguous() && bias.scalar_type() == torch::kFloat32 &&
              bias.numel() == E && probs.scalar_type() == torch::kFloat32 &&
              ids.scalar_type() == torch::kInt32 &&
              perm.scalar_type() == torch::kInt32 &&
              experts.scalar_type() == torch::kInt32 &&
              offsets.scalar_type() == torch::kInt32);
  TORCH_CHECK(T >= 1 && K >= 1 && K <= 8 && T * K <= RPW_MAX,
              "route_plan_wide: T*K <= 640");
  const at::cuda::OptionalCUDAGuard guard(device_of(logits));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#define V41SK_RPW(EPL)                                                    \
  route_plan_wide_kernel<EPL><<<1, 256, 0, stream>>>(                     \
      logits.data_ptr<float>(), bias.data_ptr<float>(),                   \
      probs.data_ptr<float>(), ids.data_ptr<int>(), perm.data_ptr<int>(), \
      experts.data_ptr<int>(), offsets.data_ptr<int>(), T, K, (float)scale)
  if (E == 384)
    V41SK_RPW(12);
  else if (E == 256)
    V41SK_RPW(8);
  else if (E == 128)
    V41SK_RPW(4);
  else
    TORCH_CHECK(false, "route_plan_wide: E in {128, 256, 384}");
#undef V41SK_RPW
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Arbitrary-S EP slot plan in one 256-thread CTA (shared-memory counting sort
// for local_experts <= 64, 0 scratch tensors, supports int32 & int64 ids).
template <typename ID_T>
__global__ void __launch_bounds__(256)
    ep_plan_cuda_kernel(const ID_T* __restrict__ ids,
                        const int* __restrict__ expert_map,
                        int* __restrict__ local_ids, int* __restrict__ perm,
                        int* __restrict__ experts, int* __restrict__ offsets,
                        int S, int E, int local_experts, int exp_len,
                        int off_len) {
  __shared__ int s_cnt[64];
  __shared__ int s_start[64];
  __shared__ int s_groups;
  __shared__ int s_total;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  if (tid < 64) s_cnt[tid] = 0;
  __syncthreads();

  for (int s = tid; s < S; s += 256) {
    const int64_t g = static_cast<int64_t>(ids[s]);
    int loc = local_experts;
    if (g >= 0 && g < E) {
      const int owned = __ldg(expert_map + g);
      if (owned >= 0 && owned < local_experts) loc = owned;
    }
    local_ids[s] = loc;
    if (loc < local_experts) atomicAdd(&s_cnt[loc], 1);
  }
  __syncthreads();

  if (tid < 32) {
    const unsigned full = 0xffffffffu;
    const int c0 = (lane < local_experts) ? s_cnt[lane] : 0;
    const int c1 = (lane + 32 < local_experts) ? s_cnt[lane + 32] : 0;
    int pc0 = c0, pc1 = c1;
    int ph0 = (c0 > 0) ? 1 : 0, ph1 = (c1 > 0) ? 1 : 0;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int ac0 = __shfl_up_sync(full, pc0, o);
      const int ac1 = __shfl_up_sync(full, pc1, o);
      const int ah0 = __shfl_up_sync(full, ph0, o);
      const int ah1 = __shfl_up_sync(full, ph1, o);
      if (lane >= o) {
        pc0 += ac0;
        pc1 += ac1;
        ph0 += ah0;
        ph1 += ah1;
      }
    }
    const int tot0 = __shfl_sync(full, pc0, 31);
    const int grp0 = __shfl_sync(full, ph0, 31);
    const int st0 = pc0 - c0;
    const int st1 = tot0 + pc1 - c1;
    const int gr0 = ph0 - 1;
    const int gr1 = grp0 + ph1 - 1;

    s_start[lane] = st0;
    s_start[lane + 32] = st1;
    if (c0 > 0 && gr0 < exp_len) experts[gr0] = lane;
    if (c0 > 0 && gr0 < off_len) offsets[gr0] = st0;
    if (c1 > 0 && gr1 < exp_len) experts[gr1] = lane + 32;
    if (c1 > 0 && gr1 < off_len) offsets[gr1] = st1;

    if (lane == 31) {
      s_total = tot0 + pc1;
      s_groups = grp0 + ph1;
    }
  }
  __syncthreads();

  const int groups = s_groups;
  const int total = s_total;
  for (int s = tid; s < S; s += 256) {
    const int loc = local_ids[s];
    if (loc < local_experts) {
      const int pos = atomicAdd(&s_start[loc], 1);
      perm[pos] = s;
    }
  }
  for (int i = groups + tid; i < exp_len; i += 256) experts[i] = 0;
  for (int i = groups + tid; i < off_len; i += 256) offsets[i] = total;
}

void ep_plan(torch::Tensor ids, torch::Tensor expert_map,
             torch::Tensor local_ids, torch::Tensor perm,
             torch::Tensor experts, torch::Tensor offsets,
             int64_t local_experts) {
  TORCH_CHECK(ids.is_cuda() && ids.is_contiguous() &&
              (ids.scalar_type() == torch::kInt32 ||
               ids.scalar_type() == torch::kInt64));
  TORCH_CHECK(expert_map.is_cuda() &&
              expert_map.scalar_type() == torch::kInt32 &&
              expert_map.is_contiguous());
  TORCH_CHECK(local_ids.is_cuda() && local_ids.scalar_type() == torch::kInt32 &&
              local_ids.is_contiguous());
  TORCH_CHECK(perm.is_cuda() && perm.scalar_type() == torch::kInt32 &&
              perm.is_contiguous());
  TORCH_CHECK(experts.is_cuda() && experts.scalar_type() == torch::kInt32 &&
              experts.is_contiguous());
  TORCH_CHECK(offsets.is_cuda() && offsets.scalar_type() == torch::kInt32 &&
              offsets.is_contiguous());
  TORCH_CHECK(local_experts >= 1 && local_experts <= 64);

  const int S = static_cast<int>(ids.numel());
  TORCH_CHECK(local_ids.numel() >= S && perm.numel() >= S);
  const int exp_len = static_cast<int>(experts.numel());
  const int off_len = static_cast<int>(offsets.numel());
  const at::cuda::OptionalCUDAGuard guard(device_of(ids));
  auto stream = at::cuda::getCurrentCUDAStream();
  if (S == 0) {
    if (exp_len > 0) experts.zero_();
    if (off_len > 0) offsets.zero_();
    return;
  }

  const int E = static_cast<int>(expert_map.numel());
  if (ids.scalar_type() == torch::kInt32) {
    ep_plan_cuda_kernel<int32_t><<<1, 256, 0, stream>>>(
        ids.data_ptr<int32_t>(), expert_map.data_ptr<int>(),
        local_ids.data_ptr<int>(), perm.data_ptr<int>(),
        experts.data_ptr<int>(), offsets.data_ptr<int>(), S, E,
        static_cast<int>(local_experts), exp_len, off_len);
  } else {
    ep_plan_cuda_kernel<int64_t><<<1, 256, 0, stream>>>(
        ids.data_ptr<int64_t>(), expert_map.data_ptr<int>(),
        local_ids.data_ptr<int>(), perm.data_ptr<int>(),
        experts.data_ptr<int>(), offsets.data_ptr<int>(), S, E,
        static_cast<int>(local_experts), exp_len, off_len);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace v41sk

TORCH_LIBRARY(v41sk, m) {
  m.def(
      "moe_w13_swiglu(Tensor x, Tensor qcodes, Tensor qscales, Tensor gscales, "
      "Tensor probabilities, Tensor perm, Tensor gids, Tensor goff, int topk, "
      "Tensor(a!) activated, int xch=4) -> ()");
  m.def(
      "skinny_qpn_gemm_sm70(Tensor x, Tensor qcodes, Tensor qscales, "
      "float gscale, int n) -> Tensor");
  m.def(
      "skinny_moe_qpn_sm70(Tensor x, Tensor qcodes, Tensor qscales, "
      "Tensor gscales, Tensor perm, Tensor gids, Tensor goff, int topk, "
      "Tensor(a!) y_slots, bool x_slot_major, int num_tokens, int splitk, "
      "int nacc, int scale_mode, int max_rows=0) -> ()");
  m.def(
      "route_plan(Tensor logits, Tensor bias, Tensor(a!) probs, Tensor(b!) "
      "ids, "
      "Tensor(c!) perm, Tensor(d!) experts, Tensor(e!) offsets, float scale, "
      "bool match_groups=False) "
      "-> ()");
  m.def(
      "skinny_moe_dec_sm70(Tensor x, Tensor qcodes, Tensor qscales, "
      "Tensor gscales, Tensor perm, Tensor gids, Tensor goff, int topk, "
      "Tensor(a!) y_slots, bool x_slot_major, int splitk, int xch) -> ()");
  m.def(
      "h16_gemv(Tensor(a!) out, Tensor x, Tensor packed, Tensor(b!) tickets, "
      "int warps, int parts, float oscale=1.0) -> ()");
  m.def(
      "gate_route(Tensor(a!) logits, Tensor x, Tensor packed, Tensor(b!) "
      "tickets, Tensor bias, "
      "Tensor(c!) probs, Tensor(d!) ids, Tensor(e!) perm, Tensor(f!) experts, "
      "Tensor(g!) offsets, "
      "int parts, float scale, float oscale, bool match_groups=False) -> ()");
  m.def(
      "attn_ring_dec(Tensor(a!) out, Tensor q, Tensor(b!) ring, Tensor values, "
      "Tensor positions_all, Tensor cache, Tensor table, Tensor positions, "
      "Tensor valid, "
      "Tensor topk, Tensor sink, float scale, int block, int ratio, int "
      "window, int ntopk, "
      "Tensor? rope=None) -> ()");
  m.def(
      "attn_paged_dec(Tensor(a!) out, Tensor q, Tensor values, Tensor(b!) "
      "wcache, "
      "Tensor wtable, Tensor cache, Tensor table, Tensor rows, Tensor bases, "
      "Tensor positions, Tensor valid, Tensor topk, Tensor sink, float scale, "
      "int wblock, int block, int ratio, int window, int ntopk, Tensor? "
      "rope=None) -> ()");
  m.def(
      "attn_dec(Tensor(a!) out, Tensor q, Tensor keys, Tensor unused, "
      "Tensor sink, float scale) -> ()");
  m.def(
      "ll_allreduce(Tensor(a!) out, Tensor inp, int p0, int p1, int p2, int "
      "p3, "
      "int rank, int cap, Tensor(b!) counter) -> ()");
  m.def("ll_arm(int ptr, int bytes) -> ()");
  m.def(
      "ll_reduce_scatter(Tensor(a!) out, Tensor inp, int p0, int p1, int p2, "
      "int p3, int rank, int cap, Tensor(b!) counter) -> ()");
  m.def(
      "ll_ep_combine16(Tensor(a!) out, Tensor routed, Tensor shared, "
      "Tensor peers, int rank, int cap, Tensor(b!) counter) -> ()");
  m.def(
      "ll_ep_combine16_tp(Tensor(a!) out, Tensor routed, Tensor shared, "
      "Tensor peers, int rank, int cap, Tensor(b!) counter) -> ()");
  m.def(
      "ll_ep_combine16_tp_sparse(Tensor(a!) out, Tensor routed, Tensor shared, "
      "Tensor ids, Tensor peers, int rank, int cap, Tensor(b!) counter, "
      "int per_rank) -> ()");
  m.def(
      "skinny_moe_pf_sm70(Tensor x, Tensor qcodes, Tensor qscales, Tensor "
      "gscales, Tensor perm, "
      "Tensor gids, Tensor goff, int topk, Tensor(a!) y_slots, bool "
      "x_slot_major, int splitk, "
      "int xch, int rb) -> ()");
  m.def(
      "hc_parts_tb(Tensor(a!) residual, Tensor y, Tensor post, Tensor comb, "
      "Tensor? fn, "
      "Tensor cpre, Tensor(b!) cbuf, Tensor(c!) dots, Tensor(d!) squares, "
      "Tensor(e!) csquares) -> ()");
  m.def(
      "attn_ring_tok(Tensor(a!) out, Tensor q, Tensor ring, Tensor values, "
      "Tensor positions_all, Tensor cache, Tensor table, Tensor positions, "
      "Tensor valid, "
      "Tensor topk, Tensor sink, float scale, int block, int ratio, int "
      "window, int ntopk) -> ()");
  m.def(
      "attn_paged_tok(Tensor(a!) out, Tensor q, Tensor wcache, Tensor wtable, "
      "Tensor cache, "
      "Tensor table, Tensor rows, Tensor positions, Tensor valid, Tensor topk, "
      "Tensor sink, "
      "float scale, int wblock, int block, int ratio, int window, int ntopk) "
      "-> ()");
  m.def(
      "moe_w13_units(Tensor x, Tensor qcodes, Tensor qscales, Tensor gscales, "
      "Tensor perm, "
      "Tensor gids, Tensor goff, int topk, Tensor(a!) y_slots, int ctas, int "
      "warps, int xch) -> ()");
  m.def(
      "moe_w2_route(Tensor act, Tensor qcodes, Tensor qscales, Tensor gscales, "
      "Tensor perm, "
      "Tensor gids, Tensor goff, int topk, Tensor? shared, Tensor(a!) out, int "
      "warps) -> ()");
  m.def(
      "ll_allgather(Tensor(a!) out, Tensor inp, int p0, int p1, int p2, int "
      "p3, "
      "int rank, int cap, Tensor(b!) counter) -> ()");
  m.def(
      "hc_mix(Tensor residual, Tensor fn, Tensor scale, Tensor base, "
      "Tensor(a!) pre_o, "
      "Tensor(b!) post_o, Tensor(c!) comb_o, Tensor(d!) ws, Tensor(e!) ticket, "
      "float norm_eps, float hc_eps, int iters) -> ()");
  m.def(
      "route_plan_wide(Tensor logits, Tensor bias, Tensor(a!) probs, "
      "Tensor(b!) ids, "
      "Tensor(c!) perm, Tensor(d!) experts, Tensor(e!) offsets, float scale) "
      "-> ()");
  m.def(
      "index_scores(Tensor(a!) out, Tensor q, Tensor w, Tensor cache, Tensor "
      "table, "
      "Tensor positions, Tensor qsl, int num_reqs, int ratio, int ctas) -> ()");
  m.def(
      "index_scores_shard(Tensor(a!) out, Tensor q, Tensor w, Tensor cache, "
      "Tensor table, "
      "Tensor positions, Tensor qsl, int num_reqs, int ratio, int ctas, int "
      "shard, "
      "int nshard) -> ()");
  m.def("attn64_prof(bool reset) -> Tensor");
  m.def(
      "ll_alltoall(Tensor(a!) out, Tensor inp, int p0, int p1, int p2, int p3, "
      "int rank, int cap, Tensor counter) -> ()");
  m.def(
      "attn_sparse64(Tensor(a!) out, Tensor q, int token0, Tensor values, "
      "Tensor wcache, Tensor wtable, Tensor cache, Tensor table, Tensor rows, "
      "Tensor bases, Tensor positions, Tensor valid, Tensor topk, Tensor sink, "
      "float scale, int wblock, int block, int ratio, int window, int ntopk) "
      "-> ()");
  m.def(
      "moe_pf_gemm(Tensor(a!) out, Tensor x, Tensor a_rows, int a_div, "
      "Tensor out_rows, Tensor codes, "
      "Tensor qscales, Tensor gscales, Tensor offsets, Tensor experts, "
      "int groups, int row_tiles, int min_rows=0, int bm=64) -> ()");
  m.def(
      "moe_pf_swiglu(Tensor(a!) activated, Tensor gate_up, Tensor probs, "
      "Tensor perm, Tensor offsets, int groups) -> ()");
  m.def(
      "moe_pf_gemm_accum(Tensor(a!) out_f32, Tensor x, Tensor perm, "
      "int out_div, Tensor codes, Tensor qscales, Tensor gscales, "
      "Tensor offsets, Tensor experts, int groups, int row_tiles, "
      "int bm=32) -> ()");
  m.def(
      "index_scores_prefill(Tensor(a!) out, Tensor q, Tensor w, Tensor cache, "
      "Tensor table, Tensor positions, int ratio, int ctas, int shard, "
      "int nshard) -> ()");
  m.def(
      "hc_expand_collapse_norm(Tensor(a!) residual, Tensor y, Tensor post, "
      "Tensor comb, Tensor cpre, Tensor norm_w, Tensor(b!) out, "
      "Tensor(c!) sq_out, bool mix, float norm_eps) -> ()");
  m.def(
      "hc_sinkhorn(Tensor dots, Tensor sq_in, Tensor scale, Tensor base, "
      "Tensor(a!) pre_o, Tensor(b!) post_o, Tensor(c!) comb_o, "
      "float norm_eps, float hc_eps, int iters) -> ()");
  m.def(
      "attn_sparse_g4(Tensor(a!) out, Tensor q, int token0, Tensor values, "
      "Tensor wcache, Tensor wtable, Tensor cache, Tensor table, Tensor rows, "
      "Tensor bases, Tensor positions, Tensor valid, Tensor topk, Tensor sink, "
      "float scale, int wblock, int block, int ratio, int window, int ntopk) "
      "-> ()");
  m.def(
      "attn_sparse_g16(Tensor(a!) out, Tensor q, int token0, Tensor values, "
      "Tensor wcache, Tensor wtable, Tensor cache, Tensor table, Tensor rows, "
      "Tensor bases, Tensor positions, Tensor valid, Tensor topk, Tensor sink, "
      "float scale, int wblock, int block, int ratio, int window, int ntopk, "
      "Tensor(b!) scratch) -> ()");
  m.def(
      "ll_dp_allgather_var(Tensor(a!) out, Tensor inp, int p0, int p1, int p2, "
      "int p3, int rank, int s0, int s1, int s2, int s3, int cap, "
      "Tensor(b!) counter) -> ()");
  m.def(
      "ll_ep_combine16_tp_var(Tensor(a!) out, Tensor routed, Tensor shared, "
      "Tensor? ids, Tensor peers, int rank, int s0, int s1, int s2, int s3, "
      "int cap, Tensor(b!) counter, int per_rank) -> ()");
  m.def(
      "ll_ep_combine16_sp_var(Tensor(a!) out, Tensor routed, Tensor shared, "
      "Tensor? ids, Tensor peers, int rank, int s0, int s1, int s2, int s3, "
      "int cap, Tensor(b!) counter, int per_rank, bool wire_fp16=True) -> ()");
  m.def(
      "ll_allgather_rows(Tensor(a!) out, Tensor inp, int p0, int p1, int p2, "
      "int p3, int rank, int cap, Tensor(b!) counter) -> ()");
  m.def(
      "ep_plan(Tensor ids, Tensor expert_map, Tensor(a!) local_ids, "
      "Tensor(b!) perm, Tensor(c!) experts, Tensor(d!) offsets, "
      "int local_experts) -> ()");
  m.def(
      "attn_prep_cuda(Tensor(a!) q, Tensor kv, Tensor kv_weight, "
      "Tensor positions, Tensor table, Tensor(b!) window, float eps, "
      "int window_group=32) -> ()");
  m.def(
      "attn_post_cuda(Tensor x, Tensor positions, Tensor table, "
      "Tensor(a!) out, Tensor? ring=None, Tensor? values=None, "
      "Tensor? valid=None) -> ()");
}

TORCH_LIBRARY_IMPL(v41sk, CUDA, m) {
  m.impl("moe_w13_swiglu", &v41sk::moe_w13_swiglu);
  m.impl("skinny_qpn_gemm_sm70", &v41sk::skinny_qpn_gemm_sm70);
  m.impl("skinny_moe_qpn_sm70", &v41sk::skinny_moe_qpn_sm70);
  m.impl("route_plan", &v41sk::route_plan);
  m.impl("skinny_moe_dec_sm70", &v41sk::skinny_moe_dec_sm70);
  m.impl("attn_dec", &v41sk::attn_dec);
  m.impl("attn_ring_dec", &v41sk::attn_ring_dec);
  m.impl("attn_paged_dec", &v41sk::attn_paged_dec);
  m.impl("attn_ring_tok", &v41sk::attn_ring_tok);
  m.impl("attn_paged_tok", &v41sk::attn_paged_tok);
  m.impl("h16_gemv", &v41sk::h16_gemv);
  m.impl("gate_route", &v41sk::gate_route);
  m.impl("ll_allreduce", &v41sk::ll_allreduce);
  m.impl("skinny_moe_pf_sm70", &v41sk::skinny_moe_pf_sm70);
  m.impl("hc_parts_tb", &v41sk::hc_parts_tb);
  m.impl("moe_w2_route", &v41sk::moe_w2_route);
  m.impl("moe_w13_units", &v41sk::moe_w13_units);
  m.impl("ll_allgather", &v41sk::ll_allgather);
  m.impl("ll_reduce_scatter", &v41sk::ll_reduce_scatter);
  m.impl("ll_ep_combine16", &v41sk::ll_ep_combine16);
  m.impl("ll_ep_combine16_tp", &v41sk::ll_ep_combine16_tp);
  m.impl("ll_ep_combine16_tp_sparse", &v41sk::ll_ep_combine16_tp_sparse);
  m.impl("hc_mix", &v41sk::hc_mix);
  m.impl("index_scores", &v41sk::index_scores);
  m.impl("index_scores_shard", &v41sk::index_scores_shard);
  m.impl("index_scores_prefill", &v41sk::index_scores_prefill);
  m.impl("moe_pf_gemm", &v41sk::moe_pf_gemm);
  m.impl("moe_pf_swiglu", &v41sk::moe_pf_swiglu);
  m.impl("moe_pf_gemm_accum", &v41sk::moe_pf_gemm_accum);
  m.impl("attn_sparse64", &v41sk::attn_sparse64);
  m.impl("ll_alltoall", &v41sk::ll_alltoall);
  m.impl("route_plan_wide", &v41sk::route_plan_wide);
  m.impl("hc_expand_collapse_norm", &v41sk::hc_expand_collapse_norm);
  m.impl("hc_sinkhorn", &v41sk::hc_sinkhorn);
  m.impl("attn_sparse_g4", &v41sk::attn_sparse_g4);
  m.impl("attn_sparse_g16", &v41sk::attn_sparse_g16);
  m.impl("ll_dp_allgather_var", &v41sk::ll_dp_allgather_var);
  m.impl("ll_ep_combine16_tp_var", &v41sk::ll_ep_combine16_tp_var);
  m.impl("ll_ep_combine16_sp_var", &v41sk::ll_ep_combine16_sp_var);
  m.impl("ll_allgather_rows", &v41sk::ll_allgather_rows);
  m.impl("ep_plan", &v41sk::ep_plan);
  m.impl("attn_prep_cuda", &v41sk::attn_prep_cuda);
  m.impl("attn_post_cuda", &v41sk::attn_post_cuda);
}

TORCH_LIBRARY_IMPL(v41sk, CompositeExplicitAutograd, m) {
  m.impl("attn64_prof", &v41sk::attn64_prof);
  m.impl("ll_arm", &v41sk::ll_arm);
}

#include "solo_routes.cuh"

#include "host_agreement.cuh"
