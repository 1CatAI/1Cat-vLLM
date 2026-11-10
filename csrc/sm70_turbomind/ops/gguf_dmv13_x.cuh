// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once
// DMV_X: small-M (M <= 8) Q5_K / Q8_0 dense projections over dmv13 segment
// planes, for the DS-V4.1 decode path (built into v41k, VLLM_V41_DMV_X).
//
// Same planes, same per-warp math and same reduction order as dense_mv in
// gguf_dmv13_core.cuh (TN = 1, no pair/ab/argmax/gate options), so with
// ACC = 1 and the same (warps, split) its outputs are bit-identical. The
// differences are scheduling only:
//   - per-segment x column offsets: block-diagonal projections (V4.1 wo_a,
//     one 1024-row O group per 4096-column input slice) run in one launch;
//   - no shared-memory table area (Q5_K / Q8_0 need none) and a max-shared
//     carveout, so the block count per SM is not cut by the L1 split;
//   - ACC = 2 (opt-in, not bit-identical): steps 0/2 and 1/3 of every K group
//     accumulate into two independent mma chains, summed as acc0 + acc1 at the
//     end, halving the dependent HMMA chain per warp;
//   - PROBE (benchmark only): the same loads, x staging and barriers with no
//     decode or mma (output is garbage), to time the memory/structure floor.
//
// Include inside a namespace after <torch/all.h>, <ATen/cuda/CUDAContext.h>,
// <cuda_fp16.h> and <cstdint>. The decode and load bodies below mirror
// gguf_dmv13_core.cuh (Q5K / Q8 branches, DMV_FULLK = 1) instruction for
// instruction; tests/kernels/quantization/test_dmv13_x_contract.py keeps them
// in sync.

namespace {

constexpr int kXQ5K = 1;  // dmv13 Fmt::Q5K
constexpr int kXQ8 = 4;   // dmv13 Fmt::Q8
constexpr int kXMaxSeg = 4;

struct XSeg {
  const uint4* codes;
  const uint4* high;
  const uint4* scale;
  half* out;
  long long out_ld;
  long long x_off;  // first input column of this segment
  int n;
  int tile0;
};
struct XSegs {
  XSeg s[kXMaxSeg];
  int nseg;
};

__device__ __forceinline__ void x_mma(float (&d)[8], uint32_t a0, uint32_t a1,
                                      uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
        "+f"(d[6]), "+f"(d[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t x_lop_or(uint32_t a, uint32_t mask,
                                             uint32_t c) {
  uint32_t r;  // (a & mask) | c
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(a), "r"(mask), "r"(c));
  return r;
}
__device__ __forceinline__ half2 x_h2(uint32_t v) {
  return *reinterpret_cast<half2*>(&v);
}
__device__ __forceinline__ uint32_t x_u32(half2 v) {
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ half2 x_lo2(uint32_t v) {
  return x_h2((v & 0xffffu) | (v << 16));
}
__device__ __forceinline__ half2 x_hi2(uint32_t v) {
  return x_h2((v >> 16) | (v & 0xffff0000u));
}
__device__ __forceinline__ uint32_t x_word(const uint4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

template <int FMT>
struct XLd {
  uint4 c[4][FMT == kXQ8 ? 2 : 1];
  uint4 sc;
  uint4 hi;
};

template <int FMT>
__device__ __forceinline__ void x_load(XLd<FMT>& L, const XSeg& s, int t, int g,
                                       int G, int lane) {
  constexpr int CW = FMT == kXQ8 ? 2 : 1;
  const size_t tg = static_cast<size_t>(t) * G + g;
  L.sc = __ldcs(s.scale + tg * 32 + lane);
  if constexpr (FMT == kXQ5K) L.hi = __ldcs(s.high + tg * 32 + lane);
  const size_t Sp = static_cast<size_t>(G) * 4;
#pragma unroll
  for (int st = 0; st < 4; ++st) {
    const int sidx = g * 4 + st;
#pragma unroll
    for (int j = 0; j < CW; ++j)
      L.c[st][j] =
          __ldcs(s.codes +
                 ((static_cast<size_t>(t) * Sp + sidx) * CW + j) * 32 + lane);
  }
}

// Decode one 32-weight step into 16 half2 in k order (dmv13 decode<Q5K/Q8>).
template <int FMT>
__device__ __forceinline__ void x_decode(const XLd<FMT>& L, int st,
                                         uint32_t (&hw)[16]) {
  const uint32_t MAGIC = 0x64006400u;
  const uint32_t sc = x_word(L.sc, st);
  if constexpr (FMT == kXQ5K) {
    const half2 s2 = x_lo2(sc), m2 = x_hi2(sc);
    const uint32_t hb = x_word(L.hi, st);
    const uint32_t cr[4] = {L.c[st][0].x, L.c[st][0].y, L.c[st][0].z,
                            L.c[st][0].w};
#pragma unroll
    for (int c = 0; c < 4; ++c) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int s = 4 * c + j - 4;
        const uint32_t sh =
            s >= 0 ? (hb >> (s >= 0 ? s : 0)) : (hb << (s < 0 ? -s : 0));
        const uint32_t extra = x_lop_or(sh, 0x00100010u, MAGIC);
        const uint32_t h = x_lop_or(cr[c] >> (4 * j), 0x000f000fu, extra);
        hw[4 * c + j] = x_u32(__hfma2(__hsub2(x_h2(h), x_h2(MAGIC)), s2, m2));
      }
    }
  } else {  // Q8: int8 codes, (x ^ 0x80) + 1024 trick, minus 1152.
    const half2 s2 = x_lo2(sc);
    const half2 bias = x_h2(0x64806480u);  // 1152
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      const uint32_t v = x_word(L.c[st][c >> 2], c & 3) ^ 0x80808080u;
      uint32_t a, b;
      asm("prmt.b32 %0, %1, %2, 0x4140;" : "=r"(a) : "r"(v), "r"(0x64646464u));
      asm("prmt.b32 %0, %1, %2, 0x4342;" : "=r"(b) : "r"(v), "r"(0x64646464u));
      hw[2 * c] = x_u32(__hmul2(__hsub2(x_h2(a), bias), s2));
      hw[2 * c + 1] = x_u32(__hmul2(__hsub2(x_h2(b), bias), s2));
    }
  }
}

template <int FMT>
__device__ __forceinline__ uint32_t x_fold(const XLd<FMT>& L) {
  uint32_t f = L.sc.x ^ L.sc.y ^ L.sc.z ^ L.sc.w;
  if constexpr (FMT == kXQ5K) f ^= L.hi.x ^ L.hi.y ^ L.hi.z ^ L.hi.w;
#pragma unroll
  for (int st = 0; st < 4; ++st)
#pragma unroll
    for (int j = 0; j < (FMT == kXQ8 ? 2 : 1); ++j)
      f ^= L.c[st][j].x ^ L.c[st][j].y ^ L.c[st][j].z ^ L.c[st][j].w;
  return f;
}

// One warp: groups g0 + kslot, g0 + kslot + KW, ... of tile t (body6, TN = 1).
template <int FMT, int KW, int ACC, bool PROBE>
__device__ __forceinline__ void x_body(const XSeg& sg, int t, int kslot, int g0,
                                       int g1, int G,
                                       const half* __restrict__ x, int ldx,
                                       int M, uint4* xs, float (&acc)[ACC][8]) {
  const int lane = threadIdx.x % 32;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  XLd<FMT> A;
  int g = g0 + kslot;
  uint4* slot = xs + kslot * 256;
  uint4 X[4];
  auto xload = [&](int gg) {
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      const int idx = lane + 32 * jj, c = idx >> 3, row = idx & 7;
      X[jj] = make_uint4(0, 0, 0, 0);
      if (row < M)
        X[jj] = __ldg(
            reinterpret_cast<const uint4*>(x + row * ldx + gg * 128 + c * 8));
    }
  };
  auto xstore = [&](int b) {
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) slot[b * 128 + lane + 32 * jj] = X[jj];
  };
  if (g >= g1) return;
  x_load<FMT>(A, sg, t, g, G, lane);
  xload(g);
  xstore(0);
  __syncwarp();
  int cur = 0;
  while (true) {
    XLd<FMT> B;
    const int gn = g + KW;
    const bool more = gn < g1;
    if (more) {
      x_load<FMT>(B, sg, t, gn, G, lane);
      xload(gn);
    }
    if constexpr (PROBE) {
      const uint4 xv = slot[cur * 128 + r];
      acc[0][0] += __uint_as_float((x_fold<FMT>(A) ^ xv.x) & 0x3fffffffu);
    } else {
#pragma unroll
      for (int st = 0; st < 4; ++st) {
        const uint4* xp = slot + cur * 128 + (st * 4) * 8 + r;
        uint4 xa[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) xa[j] = xp[j * 8];
        uint32_t hw[16];
        x_decode<FMT>(A, st, hw);
        float (&a)[8] = acc[st % ACC];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          x_mma(a, xa[j].x, xa[j].y, hw[4 * j], hw[4 * j + 1]);
          x_mma(a, xa[j].z, xa[j].w, hw[4 * j + 2], hw[4 * j + 3]);
        }
      }
    }
    if (!more) break;
    cur ^= 1;
    xstore(cur);
    __syncwarp();
    A = B;
    g = gn;
  }
}

// dense_mv's output mapping (write_out) of mma accumulator v = lane * 8 + e.
__device__ __forceinline__ void x_write(const XSeg& sg, int t, int v, float val,
                                        int M) {
  const int lv = v >> 3, i = v & 7;
  const int token = (i & 2) | ((lv & 16) ? 4 : 0) | (lv & 1);
  const int col = (i & 1) | (((lv >> 1) & 1) << 1) | ((i >> 2) << 2);
  const int row = t * 32 + ((lv >> 2) & 3) * 8 + col;
  if (token < M && row < sg.n)
    sg.out[token * sg.out_ld + row] = __float2half_rn(val);
}

template <int KW, int FMT, int ACC, bool PROBE>
__global__ void __launch_bounds__(32 * KW)
    dmv_x(XSegs segs, const half* __restrict__ x, int ldx, int M, int G,
          int split, float* __restrict__ ws, int* __restrict__ cnt) {
  extern __shared__ uint4 smem[];
  __shared__ int last;
  const int tg = blockIdx.x, sp = blockIdx.y;
  int si = 0;
#pragma unroll
  for (int i = 1; i < kXMaxSeg; ++i)
    if (i < segs.nseg && tg >= segs.s[i].tile0) si = i;
  const XSeg& sg = segs.s[si];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int t = tg - sg.tile0;
  const int gps = (G + split - 1) / split;
  const int g0 = min(G, sp * gps), g1 = min(G, g0 + gps);
  float acc[ACC][8] = {};
  x_body<FMT, KW, ACC, PROBE>(sg, t, warp, g0, g1, G, x + sg.x_off, ldx, M,
                              smem, acc);
  __syncthreads();
  float* red = reinterpret_cast<float*>(smem);
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    float a = acc[0][e];
    if constexpr (ACC == 2) a += acc[1][e];
    red[warp * 256 + lane * 8 + e] = a;
  }
  __syncthreads();
  for (int v = threadIdx.x; v < 256; v += 32 * KW) {
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < KW; ++k) s += red[k * 256 + v];
    if (split == 1)
      x_write(sg, t, v, s, M);
    else
      __stcg(ws + (static_cast<size_t>(tg) * split + sp) * 256 + v, s);
  }
  if (split == 1) return;
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    last = atomicAdd(cnt + tg, 1) == split - 1;
    if (last) __threadfence();
  }
  __syncthreads();
  if (!last) return;
  for (int v = threadIdx.x; v < 256; v += 32 * KW) {
    float pv[8];
#pragma unroll
    for (int p = 0; p < 8; ++p)
      pv[p] = p < split
                  ? __ldcg(ws + (static_cast<size_t>(tg) * split + p) * 256 + v)
                  : 0.f;
    float s = 0.f;
#pragma unroll
    for (int p = 0; p < 8; ++p) s += pv[p];
    x_write(sg, t, v, s, M);
  }
  if (threadIdx.x == 0) cnt[tg] = 0;
}

template <int KW, int FMT, int ACC, bool PROBE>
void x_launch(const XSegs& segs, const half* x, int ldx, int M, int G,
              int split, float* ws, int* cnt, int tiles, cudaStream_t st) {
  // x staging (two 2 KiB slots per warp) and the K-slot reduction share it.
  const size_t smem = static_cast<size_t>(KW) * 256 * 16;
  static bool init = false;
  if (!init) {
    // Shared-memory-heavy carveout: 96 KiB on SM70, so registers (not the
    // L1 split) bound the blocks per SM. A hint; scheduling only.
    cudaFuncSetAttribute(dmv_x<KW, FMT, ACC, PROBE>,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    init = true;
  }
  dmv_x<KW, FMT, ACC, PROBE><<<dim3(tiles, split), 32 * KW, smem, st>>>(
      segs, x, ldx, M, G, split, ws, cnt);
  const cudaError_t e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "dmv13_x launch: ", cudaGetErrorString(e));
}

}  // namespace

// x: [M, *] FP16 rows (stride ldx); segment i reads columns [x_off[i],
// x_off[i] + K) and writes out[i] ([M, n_i] view, any row stride).
inline void dmv13_x_run(torch::Tensor x, std::vector<torch::Tensor> codes,
                        std::vector<torch::Tensor> high,
                        std::vector<torch::Tensor> scale,
                        std::vector<torch::Tensor> out,
                        std::vector<int64_t> x_off, int64_t fmt, int64_t K,
                        int64_t split, int64_t warps, int64_t accs,
                        torch::Tensor ws, torch::Tensor cnt, int64_t probe) {
  const int nseg = static_cast<int>(codes.size());
  TORCH_CHECK(nseg >= 1 && nseg <= kXMaxSeg && high.size() == codes.size() &&
                  scale.size() == codes.size() && out.size() == codes.size() &&
                  x_off.size() == codes.size(),
              "dmv13_x: 1..4 segments with codes/high/scale/out/x_off each");
  TORCH_CHECK(fmt == kXQ5K || fmt == kXQ8, "dmv13_x: Q5_K or Q8_0 only");
  TORCH_CHECK(K > 0 && K % 128 == 0, "dmv13_x: K must be a multiple of 128");
  TORCH_CHECK(split >= 1 && split <= 8, "dmv13_x: split 1..8");
  TORCH_CHECK(accs == 1 || accs == 2, "dmv13_x: acc 1 or 2");
  TORCH_CHECK(x.scalar_type() == torch::kHalf && x.dim() == 2 &&
                  x.stride(1) == 1 && x.size(0) >= 1 && x.size(0) <= 8,
              "dmv13_x: x must be FP16 [1..8, *] with unit column stride");
  const int M = static_cast<int>(x.size(0));
  const int ldx = static_cast<int>(x.stride(0));
  const int G = static_cast<int>(K / 128);
  TORCH_CHECK(
      ldx % 8 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
      "dmv13_x: x rows must be 16-byte aligned");
  const int cw = fmt == kXQ8 ? 2 : 1;
  XSegs segs{};
  segs.nseg = nseg;
  int tiles = 0;
  for (int i = 0; i < nseg; ++i) {
    XSeg& s = segs.s[i];
    const int64_t n = out[i].size(1);
    const int64_t t = (n + 31) / 32;
    TORCH_CHECK(out[i].scalar_type() == torch::kHalf && out[i].dim() == 2 &&
                    out[i].size(0) == M && out[i].stride(1) == 1,
                "dmv13_x: out views must be FP16 [M, n] with unit stride");
    TORCH_CHECK(x_off[i] >= 0 && x_off[i] % 8 == 0 && x_off[i] + K <= x.size(1),
                "dmv13_x: x_off out of range or misaligned");
    TORCH_CHECK(codes[i].nbytes() >= static_cast<size_t>(t) * G * 2048 * cw &&
                    scale[i].nbytes() >= static_cast<size_t>(t) * G * 512 &&
                    (fmt != kXQ5K ||
                     high[i].nbytes() >= static_cast<size_t>(t) * G * 512),
                "dmv13_x: planes too small for n and K");
    s.codes = reinterpret_cast<const uint4*>(codes[i].data_ptr());
    s.high = fmt == kXQ5K ? reinterpret_cast<const uint4*>(high[i].data_ptr())
                          : nullptr;
    s.scale = reinterpret_cast<const uint4*>(scale[i].data_ptr());
    s.out = reinterpret_cast<half*>(out[i].data_ptr());
    s.out_ld = out[i].stride(0);
    s.x_off = x_off[i];
    s.n = static_cast<int>(n);
    s.tile0 = tiles;
    tiles += static_cast<int>(t);
  }
  TORCH_CHECK(
      split == 1 || (ws.scalar_type() == torch::kFloat &&
                     ws.numel() >= static_cast<int64_t>(tiles) * split * 256 &&
                     cnt.scalar_type() == torch::kInt && cnt.numel() >= tiles),
      "dmv13_x: workspace/counters too small for tiles x split");
  auto st = at::cuda::getCurrentCUDAStream();
  const half* xp = reinterpret_cast<const half*>(x.data_ptr());
  float* wsp = split > 1 ? ws.data_ptr<float>() : nullptr;
  int* cp = split > 1 ? cnt.data_ptr<int>() : nullptr;
#define DMVX_GO(KWV, F, A, P)                                     \
  if (warps == KWV && fmt == F && accs == A && (probe != 0) == P) \
    return x_launch<KWV, F, A, P>(                                \
        segs, xp, ldx, M, G, static_cast<int>(split), wsp, cp, tiles, st);
#define DMVX_W(KWV)              \
  DMVX_GO(KWV, kXQ5K, 1, false); \
  DMVX_GO(KWV, kXQ5K, 2, false); \
  DMVX_GO(KWV, kXQ8, 1, false);  \
  DMVX_GO(KWV, kXQ8, 2, false);  \
  DMVX_GO(KWV, kXQ5K, 1, true);  \
  DMVX_GO(KWV, kXQ8, 1, true);
  DMVX_W(2);
  DMVX_W(4);
  DMVX_W(8);
#undef DMVX_W
#undef DMVX_GO
  TORCH_CHECK(false, "dmv13_x: warps must be 2, 4 or 8 (probe: acc 1)");
}
