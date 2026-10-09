// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

// Included after the segment decoder and output helpers. All twenty
// verification rows share one weight load and decode; their independent
// eight-row MMA accumulators retain the original K order and FP32 reduction
// tree.
template <int FMT>
__device__ __forceinline__ void decode_weight_major(const Ld<FMT>& L, int st,
                                                    uint32_t (&hw)[16]) {
  if constexpr (FMT != LUT4) {
    decode<FMT>(L, st, hw, nullptr);
  } else {
    const half2 s2 = lo2(word(L.sc, st));
    const uint32_t cr[4] = {L.c[st][0].x, L.c[st][0].y, L.c[st][0].z,
                            L.c[st][0].w};
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      // kIQ4 + 128 as bytes; nibble bit 3 picks the upper table half.
      const uint32_t T0 = 0x3F2D1801u, T1 = 0x766A5D4Fu, T2 = 0xA6998D81u,
                     T3 = 0xF1D9C5B5u;
      const half2 b1152 = h2(0x64806480u);
      const uint32_t q = cr[c];
      const uint32_t qm = q & 0x77777777u;
      uint32_t sel;
      asm("lop3.b32 %0, %1, %2, %3, 0xEA;"
          : "=r"(sel)
          : "r"(q >> 1), "r"(0x44444444u), "r"(0x32103210u));
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        uint32_t lo, hi, v, a, b;
        asm("prmt.b32 %0, %1, %2, %3;"
            : "=r"(lo)
            : "r"(T0), "r"(T1), "r"(qm >> (16 * i)));
        asm("prmt.b32 %0, %1, %2, %3;"
            : "=r"(hi)
            : "r"(T2), "r"(T3), "r"(qm >> (16 * i)));
        asm("prmt.b32 %0, %1, %2, %3;"
            : "=r"(v)
            : "r"(lo), "r"(hi), "r"(sel >> (16 * i)));
        asm("prmt.b32 %0, %1, %2, 0x4140;"
            : "=r"(a)
            : "r"(v), "r"(0x64646464u));
        asm("prmt.b32 %0, %1, %2, 0x4342;"
            : "=r"(b)
            : "r"(v), "r"(0x64646464u));
        hw[4 * c + 2 * i] = u32(__hmul2(__hsub2(h2(a), b1152), s2));
        hw[4 * c + 2 * i + 1] = u32(__hmul2(__hsub2(h2(b), b1152), s2));
      }
    }
  }
}

template <int FMT, int W, int TM>
__device__ __forceinline__ void body_weight_major(const Seg& sg, int t, int g0,
                                                  int g1, int S, int G,
                                                  const half* x, int ldx, int M,
                                                  int K, uint4* xs, half2* lut,
                                                  float (&acc)[TM][8]) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  int g = g0 + warp;
  Ld<FMT> A;
  if (g < g1) load<FMT>(A, sg, t, g, S, G, lane);
  while (g < g1) {
    Ld<FMT> B;
    const int gn = g + W;
    if (gn < g1) load<FMT>(B, sg, t, gn, S, G, lane);
#pragma unroll
    for (int st = 0; st < 4; ++st) {
      if (g * 4 + st < S) {
        uint32_t hw[16];
        decode_weight_major<FMT>(A, st, hw);
#pragma unroll
        for (int b = 0; b < TM; ++b) {
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            uint4 xa = make_uint4(0, 0, 0, 0);
            if (b * 8 + r < M)
              xa = __ldg(reinterpret_cast<const uint4*>(
                  x + (b * 8 + r) * ldx + g * 128 + st * 32 + j * 8));
            mma(acc[b], xa.x, xa.y, hw[4 * j], hw[4 * j + 1]);
            mma(acc[b], xa.z, xa.w, hw[4 * j + 2], hw[4 * j + 3]);
          }
        }
      }
    }
    if (gn >= g1) break;
    A = B;
    g = gn;
  }
}

template <int W, int TM>
__global__ void __launch_bounds__(32 * W)
    dense_weight_major(Segs segs, const half* x, int ldx, int M, int K, int S,
                       int G, int split, float* ws, int* cnt) {
  extern __shared__ uint4 smem[];
  __shared__ int last;
  const int tile = blockIdx.x, sp = blockIdx.y;
  int si = 0;
#pragma unroll
  for (int i = 1; i < MAXSEG; ++i)
    if (i < segs.nseg && tile >= segs.s[i].tile0) si = i;
  const Seg& sg = segs.s[si];
  const int t = tile - sg.tile0;
  const int gps = (G + split - 1) / split;
  const int g0 = min(G, sp * gps), g1 = min(G, g0 + gps);
  float acc[TM][8] = {};
  half2* lut = reinterpret_cast<half2*>(smem);
  switch (sg.fmt) {
    case Q4K:
      body_weight_major<Q4K, W, TM>(sg, t, g0, g1, S, G, x, ldx, M, K, smem,
                                    lut, acc);
      break;
    case Q5K:
      body_weight_major<Q5K, W, TM>(sg, t, g0, g1, S, G, x, ldx, M, K, smem,
                                    lut, acc);
      break;
    case Q6K:
      body_weight_major<Q6K, W, TM>(sg, t, g0, g1, S, G, x, ldx, M, K, smem,
                                    lut, acc);
      break;
    case LUT4:
      body_weight_major<LUT4, W, TM>(sg, t, g0, g1, S, G, x, ldx, M, K, smem,
                                     lut, acc);
      break;
    default:
      body_weight_major<Q8, W, TM>(sg, t, g0, g1, S, G, x, ldx, M, K, smem, lut,
                                   acc);
      break;
  }
  float* red = reinterpret_cast<float*>(smem);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
#pragma unroll
  for (int b = 0; b < TM; ++b)
#pragma unroll
    for (int i = 0; i < 8; ++i)
      red[(b * W + warp) * 256 + lane * 8 + i] = acc[b][i];
  __syncthreads();
#pragma unroll
  for (int b = 0; b < TM; ++b) {
    for (int v = threadIdx.x; v < 256; v += 32 * W) {
      float sum = 0.f;
#pragma unroll
      for (int w = 0; w < W; ++w) sum += red[(b * W + w) * 256 + v];
      if (split == 1)
        write_out(sg, t, v, sum, min(8, M - b * 8), segs.sgate, b * 8);
      else
        ws[((static_cast<size_t>(b) * gridDim.x + tile) * split + sp) * 256 +
           v] = sum;
    }
  }
  if (split == 1) return;
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    last = atomicAdd(cnt + tile, 1) == split - 1;
    if (last) __threadfence();
  }
  __syncthreads();
  if (!last) return;
#pragma unroll
  for (int b = 0; b < TM; ++b)
    for (int v = threadIdx.x; v < 256; v += 32 * W) {
      float sum = 0.f;
      for (int sp2 = 0; sp2 < split; ++sp2)
        sum += __ldcg(
            ws +
            ((static_cast<size_t>(b) * gridDim.x + tile) * split + sp2) * 256 +
            v);
      write_out(sg, t, v, sum, min(8, M - b * 8), segs.sgate, b * 8);
    }
  if (threadIdx.x == 0) cnt[tile] = 0;
}
