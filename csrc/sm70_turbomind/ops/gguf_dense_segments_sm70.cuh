// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Based on the supplied dense_mv2/shexp2 SM70 projection implementation.
// Small-M (M<=8) GGUF dense projections on SM70.
// One launch serves several same-input projections ("segments"). Weights are
// reconstructed to FP16 (scale*code+min, as the canonical TurboMind path) and
// multiplied with mma.m8n8k4 FP32 accumulation. Split-K across CTAs reduces
// deterministically in the same launch (last CTA sums partials in order).
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cstdint>
#include <vector>

namespace {

enum Fmt { Q4K = 0, Q5K = 1, Q6K = 2, LUT4 = 3, Q8 = 4 };
constexpr int MAXSEG = 4;

struct Seg {
  const uint4* codes;
  const uint4* high;
  const uint4* scale;
  half* out;
  long long out_ld;
  int n;
  int fmt;
  int tile0;
  int pad;
};
struct Segs {
  Seg s[MAXSEG];
  int nseg;
  const half* sgate;  // optional: out = fp16(fp16(acc) * sigmoid(sgate[token +
                      // first_row]))
};

__device__ __forceinline__ void mma(float (&d)[8], uint32_t a0, uint32_t a1,
                                    uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
        "+f"(d[6]), "+f"(d[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t lop_or(uint32_t a, uint32_t mask,
                                           uint32_t c) {
  uint32_t r;  // (a & mask) | c
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(a), "r"(mask), "r"(c));
  return r;
}

__device__ __forceinline__ half2 h2(uint32_t v) {
  return *reinterpret_cast<half2*>(&v);
}
__device__ __forceinline__ uint32_t u32(half2 v) {
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ half2 lo2(uint32_t v) {
  return h2((v & 0xffffu) | (v << 16));
}
__device__ __forceinline__ half2 hi2(uint32_t v) {
  return h2((v >> 16) | (v & 0xffff0000u));
}
template <int S>
__device__ __forceinline__ uint32_t shf(uint32_t x) {
  if constexpr (S >= 0)
    return x >> S;
  else
    return x << (-S);
}

template <int FMT>
struct CW {
  static constexpr int v = FMT == Q8 ? 2 : 1;
};

template <int FMT>
struct Ld {
  uint4 c[4][CW<FMT>::v];
  uint4 sc;
  uint4 hi[FMT == Q6K ? 2 : 1];
};

template <int FMT>
__device__ __forceinline__ void load(Ld<FMT>& L, const Seg& s, int t, int g,
                                     int S, int G, int lane) {
  const size_t tg = static_cast<size_t>(t) * G + g;
  L.sc = __ldcs(s.scale + tg * 32 + lane);
  if constexpr (FMT == Q5K) L.hi[0] = __ldcs(s.high + tg * 32 + lane);
  if constexpr (FMT == Q6K) {
    L.hi[0] = __ldcs(s.high + (tg * 2) * 32 + lane);
    L.hi[1] = __ldcs(s.high + (tg * 2 + 1) * 32 + lane);
  }
  const size_t Sp = static_cast<size_t>(G) * 4;
#pragma unroll
  for (int st = 0; st < 4; ++st) {
    const int sidx = g * 4 + st;
    if (sidx < S) {
#pragma unroll
      for (int j = 0; j < CW<FMT>::v; ++j)
        L.c[st][j] = __ldcs(
            s.codes +
            ((static_cast<size_t>(t) * Sp + sidx) * CW<FMT>::v + j) * 32 +
            lane);
    }
  }
}

__device__ __forceinline__ uint32_t word(const uint4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

// Decode one 32-weight step into 16 half2 in k order.
template <int FMT>
__device__ __forceinline__ void decode(const Ld<FMT>& L, int st,
                                       uint32_t (&hw)[16], const half2* lut) {
  const uint32_t sc = word(L.sc, st);
  const uint32_t MAGIC = 0x64006400u;
  if constexpr (FMT == Q4K || FMT == Q5K) {
    const half2 s2 = lo2(sc), m2 = hi2(sc);
    uint32_t hb = 0;
    if constexpr (FMT == Q5K) hb = word(L.hi[0], st);
    const uint32_t cr[4] = {L.c[st][0].x, L.c[st][0].y, L.c[st][0].z,
                            L.c[st][0].w};
#pragma unroll
    for (int c = 0; c < 4; ++c) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        uint32_t extra = MAGIC;
        if constexpr (FMT == Q5K) {
          uint32_t sh;
          const int s = 4 * c + j - 4;
          sh = s >= 0 ? (hb >> (s >= 0 ? s : 0)) : (hb << (s < 0 ? -s : 0));
          extra = lop_or(sh, 0x00100010u, MAGIC);
        }
        const uint32_t h = lop_or(cr[c] >> (4 * j), 0x000f000fu, extra);
        hw[4 * c + j] = u32(__hfma2(__hsub2(h2(h), h2(MAGIC)), s2, m2));
      }
    }
  } else if constexpr (FMT == Q6K) {
    const half2 s0 = lo2(sc), s1 = hi2(sc);
    const uint4& hv = L.hi[st >> 1];
    const uint32_t hx = (st & 1) ? hv.z : hv.x;
    const uint32_t hy = (st & 1) ? hv.w : hv.y;
    const uint32_t cr[4] = {L.c[st][0].x, L.c[st][0].y, L.c[st][0].z,
                            L.c[st][0].w};
    const half2 bias = h2(0x64206420u);  // 1056 = 1024 + 32
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const uint32_t hb = c < 2 ? hx : hy;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int s = 2 * (4 * (c & 1) + j) - 4;
        const uint32_t sh =
            s >= 0 ? (hb >> (s >= 0 ? s : 0)) : (hb << (s < 0 ? -s : 0));
        const uint32_t extra = lop_or(sh, 0x00300030u, MAGIC);
        const uint32_t h = lop_or(cr[c] >> (4 * j), 0x000f000fu, extra);
        hw[4 * c + j] = u32(__hmul2(__hsub2(h2(h), bias), c < 2 ? s0 : s1));
      }
    }
  } else if constexpr (FMT == LUT4) {
    const half2 s2 = lo2(sc);
    const uint32_t cr[4] = {L.c[st][0].x, L.c[st][0].y, L.c[st][0].z,
                            L.c[st][0].w};
#pragma unroll
    for (int c = 0; c < 4; ++c) {
#pragma unroll
      for (int b = 0; b < 4; ++b)
        hw[4 * c + b] = u32(__hmul2(lut[(cr[c] >> (8 * b)) & 0xff], s2));
    }
  } else {  // Q8: int8 codes, (x ^ 0x80) + 1024 trick, minus 1152.
    const half2 s2 = lo2(sc);
    const half2 bias = h2(0x64806480u);  // 1152
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      const uint32_t v = word(L.c[st][c >> 2], c & 3) ^ 0x80808080u;
      uint32_t a, b;
      asm("prmt.b32 %0, %1, %2, 0x4140;" : "=r"(a) : "r"(v), "r"(0x64646464u));
      asm("prmt.b32 %0, %1, %2, 0x4342;" : "=r"(b) : "r"(v), "r"(0x64646464u));
      hw[2 * c] = u32(__hmul2(__hsub2(h2(a), bias), s2));
      hw[2 * c + 1] = u32(__hmul2(__hsub2(h2(b), bias), s2));
    }
  }
}

__device__ __constant__ int8_t kIQ4[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

template <int FMT, int W, bool FILL_LUT = true, bool COHERENT_X = false>
__device__ __forceinline__ void body(const Seg& sg, int t, int g0, int g1,
                                     int S, int G, const half* __restrict__ x,
                                     int ldx, int M, int K, uint4* xs,
                                     half2* lut, float (&acc)[8],
                                     int warp = -1) {
  if (warp < 0) warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  Ld<FMT> A;
  int g = g0 + warp;
  // Warp-private, double-buffered activation slots: slot[chunk8][row], 2 KiB
  // each.
  uint4* slot = xs + warp * 256;
  uint4 X[4];
  auto xload = [&](int gg) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int idx = lane + 32 * j, c = idx >> 3, row = idx & 7,
                k = gg * 128 + c * 8;
      X[j] = make_uint4(0, 0, 0, 0);
      if (row < M && k < K) {
        const uint4* p = reinterpret_cast<const uint4*>(x + row * ldx + k);
        X[j] = COHERENT_X ? __ldcg(p) : __ldg(p);
      }
    }
  };
  auto xstore = [&](int b) {
#pragma unroll
    for (int j = 0; j < 4; ++j) slot[b * 128 + lane + 32 * j] = X[j];
  };
  if (g < g1) {
    load<FMT>(A, sg, t, g, S, G, lane);
    xload(g);
  }
  if constexpr (FMT == LUT4 && FILL_LUT) {
    for (int i = threadIdx.x; i < 256; i += 32 * W)
      lut[i] = __halves2half2(__int2half_rn(kIQ4[i & 15]),
                              __int2half_rn(kIQ4[i >> 4]));
    __syncthreads();
  }
  if (g >= g1) return;  // body only; caller continues to the CTA reduction
  xstore(0);
  __syncwarp();
  int cur = 0;
  while (true) {
    Ld<FMT> B;
    const int gn = g + W;
    const bool more = gn < g1;
    if (more) {
      load<FMT>(B, sg, t, gn, S, G, lane);
      xload(gn);
    }
#pragma unroll
    for (int st = 0; st < 4; ++st) {
      const int sidx = g * 4 + st;
      if (sidx < S) {
        uint32_t hw[16];
        decode<FMT>(A, st, hw, lut);
        const uint4* xp = slot + cur * 128 + (st * 4) * 8 + r;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const uint4 xa = xp[j * 8];
          mma(acc, xa.x, xa.y, hw[4 * j], hw[4 * j + 1]);
          mma(acc, xa.z, xa.w, hw[4 * j + 2], hw[4 * j + 3]);
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

__device__ __forceinline__ void write_out(const Seg& sg, int t, int v,
                                          float val, int M,
                                          const half* sgate = nullptr,
                                          int first_row = 0) {
  const int lv = v >> 3, i = v & 7;
  const int token = (i & 2) | ((lv & 16) ? 4 : 0) | (lv & 1);
  const int col = (i & 1) | (((lv >> 1) & 1) << 1) | ((i >> 2) << 2);
  const int row = t * 32 + ((lv >> 2) & 3) * 8 + col;
  if (token < M && row < sg.n) {
    if (sgate != nullptr) {
      const float gate =
          1.0f /
          (1.0f + __expf(-__half2float(__ldcg(sgate + token + first_row))));
      val = __half2float(__float2half_rn(val)) * gate;
    }
    sg.out[(token + first_row) * sg.out_ld + row] = __float2half_rn(val);
  }
}

#include "gguf_dense_weight_major_sm70.cuh"

template <int W, bool Batch>
__global__ void __launch_bounds__(32 * W)
    dense_mv(Segs segs, const half* __restrict__ x, int ldx, int M, int K,
             int S, int G, int split, float* ws, int* cnt) {
  extern __shared__ uint4 smem[];
  __shared__ int last;
  const int tile = blockIdx.x, sp = blockIdx.y;
  const int scratch_tile = Batch ? blockIdx.z * gridDim.x + tile : tile;
  if constexpr (Batch) {
    const int first_row = blockIdx.z * 8;
    M = min(8, M - first_row);
    x += first_row * ldx;
  }
  int si = 0;
#pragma unroll
  for (int i = 1; i < MAXSEG; ++i)
    if (i < segs.nseg && tile >= segs.s[i].tile0) si = i;
  const Seg& sg = segs.s[si];
  const int t = tile - sg.tile0;
  const int gps = (G + split - 1) / split;
  const int g0 = min(G, sp * gps), g1 = min(G, g0 + gps);
  uint4* xs = smem;
  half2* lut = reinterpret_cast<half2*>(smem + W * 256);
  float acc[8] = {};
  switch (sg.fmt) {
    case Q4K:
      body<Q4K, W>(sg, t, g0, g1, S, G, x, ldx, M, K, xs, lut, acc);
      break;
    case Q5K:
      body<Q5K, W>(sg, t, g0, g1, S, G, x, ldx, M, K, xs, lut, acc);
      break;
    case Q6K:
      body<Q6K, W>(sg, t, g0, g1, S, G, x, ldx, M, K, xs, lut, acc);
      break;
    case LUT4:
      body<LUT4, W>(sg, t, g0, g1, S, G, x, ldx, M, K, xs, lut, acc);
      break;
    default:
      body<Q8, W>(sg, t, g0, g1, S, G, x, ldx, M, K, xs, lut, acc);
      break;
  }
  __syncthreads();
  float* red = reinterpret_cast<float*>(smem);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
#pragma unroll
  for (int i = 0; i < 8; ++i) red[warp * 256 + lane * 8 + i] = acc[i];
  __syncthreads();
  for (int v = threadIdx.x; v < 256; v += 32 * W) {
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < W; ++w) s += red[w * 256 + v];
    if (split == 1)
      write_out(sg, t, v, s, M, segs.sgate, Batch ? blockIdx.z * 8 : 0);
    else
      ws[(static_cast<size_t>(scratch_tile) * split + sp) * 256 + v] = s;
  }
  if (split == 1) return;
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    last = atomicAdd(cnt + scratch_tile, 1) == split - 1;
    if (last) __threadfence();
  }
  __syncthreads();
  if (!last) return;
  for (int v = threadIdx.x; v < 256; v += 32 * W) {
    float s = 0.f;
    for (int p = 0; p < split; ++p)
      s += __ldcg(ws + (static_cast<size_t>(scratch_tile) * split + p) * 256 +
                  v);
    write_out(sg, t, v, s, M, segs.sgate, Batch ? blockIdx.z * 8 : 0);
  }
  if (threadIdx.x == 0) cnt[scratch_tile] = 0;
}

template <int W>
void launch(const Segs& segs, const half* x, int ldx, int M, int K, int S,
            int G, int split, float* ws, int* cnt, int tiles, cudaStream_t st) {
  // Small-M verification needs three token tiles, but each uses the same
  // weights. Keep the original path for other shapes and split-K schedules.
  int total_n = 0;
  for (int i = 0; i < segs.nseg; ++i) total_n += segs.s[i].n;
  if constexpr (W == 4 || W == 8) {
    if (M == 20 && split == 1 &&
        ((K == 2560 && (total_n == 4096 || total_n == 3584)) ||
         (K == 1536 && total_n == 2560))) {
      dense_weight_major<W, 3>
          <<<dim3(tiles, split), 32 * W, 3 * W * 256 * 4, st>>>(
              segs, x, ldx, M, K, S, G, split, ws, cnt);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return;
    }
  }
  const size_t xs_bytes = static_cast<size_t>(W) * 256 * 16 + 1024;
  const size_t red_bytes = static_cast<size_t>(W) * 256 * 4;
  const size_t smem = xs_bytes > red_bytes ? xs_bytes : red_bytes;
  if (smem > 48 * 1024) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        dense_mv<W, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  }
  if (M <= 8) {
    dense_mv<W, false><<<dim3(tiles, split), 32 * W, smem, st>>>(
        segs, x, ldx, M, K, S, G, split, ws, cnt);
  } else {
    if (smem > 48 * 1024)
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          dense_mv<W, true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
          smem));
    dense_mv<W, true><<<dim3(tiles, split, (M + 7) / 8), 32 * W, smem, st>>>(
        segs, x, ldx, M, K, S, G, split, ws, cnt);
  }
  const cudaError_t e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "dense_mv launch: ", cudaGetErrorString(e),
              " tiles=", tiles, " split=", split, " W=", W, " smem=", smem);
}

}  // namespace

// codes/high/scale: per segment planes; out: per segment [M, n] views.
void gguf_dense_segments_sm70_out(
    torch::Tensor x, std::vector<torch::Tensor> codes,
    std::vector<torch::Tensor> high, std::vector<torch::Tensor> scale,
    std::vector<torch::Tensor> out, std::vector<int64_t> fmt,
    std::vector<int64_t> n, int64_t K, int64_t split, int64_t warps,
    torch::Tensor ws, torch::Tensor cnt, std::optional<torch::Tensor> sgate) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.dim() == 2 &&
                  x.stride(1) == 1 && x.size(0) > 0 && x.size(0) <= 32,
              "input must be CUDA FP16 [1..32, K] with contiguous rows");
  const c10::cuda::CUDAGuard guard(x.device());
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major == 7 && properties->minor == 0,
              "SM70 required");
  TORCH_CHECK(K == x.size(1) && K > 0 && K % 32 == 0 && split > 0 &&
                  split <= (K + 127) / 128,
              "invalid K or split count");
  TORCH_CHECK(
      warps == 1 || warps == 2 || warps == 4 || warps == 8 || warps == 16,
      "unsupported warp count");
  TORCH_CHECK(!codes.empty() && codes.size() <= MAXSEG &&
                  codes.size() == high.size() && codes.size() == scale.size() &&
                  codes.size() == out.size() && codes.size() == fmt.size() &&
                  codes.size() == n.size(),
              "segment arrays must have equal nonzero lengths");
  auto same = [&](const torch::Tensor& t, at::ScalarType dtype) {
    TORCH_CHECK(t.device() == x.device() && t.scalar_type() == dtype &&
                    t.is_contiguous(),
                "packed storage must be contiguous and on the input device");
  };
  same(ws, at::kFloat);
  same(cnt, at::kInt);
  if (sgate && sgate->numel()) {
    same(*sgate, at::kHalf);
    TORCH_CHECK(sgate->numel() >= x.size(0),
                "shared gate storage is too small");
  }
  Segs segs{};
  segs.sgate = sgate && sgate->numel()
                   ? reinterpret_cast<const half*>(sgate->data_ptr())
                   : nullptr;
  segs.nseg = static_cast<int>(codes.size());
  TORCH_CHECK(segs.nseg <= MAXSEG);
  int tiles = 0;
  for (int i = 0; i < segs.nseg; ++i) {
    TORCH_CHECK(fmt[i] >= Q4K && fmt[i] <= Q8 && n[i] > 0,
                "invalid segment format or N");
    same(codes[i], at::kByte);
    same(scale[i], at::kByte);
    if (high[i].numel()) same(high[i], at::kByte);
    const int64_t tiles_i = (n[i] + 31) / 32, groups_i = (K + 127) / 128;
    TORCH_CHECK(codes[i].numel() ==
                        tiles_i * groups_i * 4 * 512 * (fmt[i] == Q8 ? 2 : 1) &&
                    scale[i].numel() == tiles_i * groups_i * 512,
                "packed code or coefficient size mismatch");
    TORCH_CHECK(high[i].numel() == tiles_i * groups_i * 512 *
                                       (fmt[i] == Q5K   ? 1
                                        : fmt[i] == Q6K ? 2
                                                        : 0),
                "high plane size mismatch");
    TORCH_CHECK(out[i].device() == x.device() &&
                    out[i].scalar_type() == at::kHalf && out[i].dim() == 2 &&
                    out[i].size(0) == x.size(0) && out[i].size(1) == n[i] &&
                    out[i].stride(1) == 1,
                "invalid output segment");
    Seg& s = segs.s[i];
    s.codes = reinterpret_cast<const uint4*>(codes[i].data_ptr());
    s.high = high[i].numel()
                 ? reinterpret_cast<const uint4*>(high[i].data_ptr())
                 : nullptr;
    s.scale = reinterpret_cast<const uint4*>(scale[i].data_ptr());
    s.out = reinterpret_cast<half*>(out[i].data_ptr());
    s.out_ld = out[i].stride(0);
    s.n = static_cast<int>(n[i]);
    s.fmt = static_cast<int>(fmt[i]);
    s.tile0 = tiles;
    tiles += (s.n + 31) / 32;
  }
  TORCH_CHECK(
      ws.numel() >= int64_t{tiles} * ((x.size(0) + 7) / 8) * split * 256 &&
          cnt.numel() >= tiles * ((x.size(0) + 7) / 8),
      "split workspace or counter storage is too small");
  const int M = static_cast<int>(x.size(0));
  TORCH_CHECK(M <= 32);
  const int S = static_cast<int>((K + 31) / 32);
  const int G = (S + 3) / 4;
  auto st = at::cuda::getCurrentCUDAStream();
  const half* xp = reinterpret_cast<const half*>(x.data_ptr());
  const int ldx = static_cast<int>(x.stride(0));
  float* wsp = ws.data_ptr<float>();
  int* cp = cnt.data_ptr<int>();
  switch (warps) {
    case 1:
      launch<1>(segs, xp, ldx, M, K, S, G, split, wsp, cp, tiles, st);
      break;
    case 2:
      launch<2>(segs, xp, ldx, M, K, S, G, split, wsp, cp, tiles, st);
      break;
    case 4:
      launch<4>(segs, xp, ldx, M, K, S, G, split, wsp, cp, tiles, st);
      break;
    case 16:
      launch<16>(segs, xp, ldx, M, K, S, G, split, wsp, cp, tiles, st);
      break;
    case 5:
      launch<5>(segs, xp, ldx, M, K, S, G, split, wsp, cp, tiles, st);
      break;
    case 6:
      launch<6>(segs, xp, ldx, M, K, S, G, split, wsp, cp, tiles, st);
      break;
    case 10:
      launch<10>(segs, xp, ldx, M, K, S, G, split, wsp, cp, tiles, st);
      break;
    case 12:
      launch<12>(segs, xp, ldx, M, K, S, G, split, wsp, cp, tiles, st);
      break;
    default:
      launch<8>(segs, xp, ldx, M, K, S, G, split, wsp, cp, tiles, st);
      break;
  }
}
