// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// EXL3 (trellis-coded, mcg codebook, K = 2/3/4 bits) routed-expert MoE for
// SM70. Weights are decoded on the fly from the trellis and multiplied with
// mma.m8n8k4 (fp32 accumulate); nothing is dequantized to memory.
//
// One slot = one (token, routed expert) assignment, s = t * topk + j.
//   pre  : X'[s] = had128(x[t] * suh[e])                                fp16
//   [S, k] gemv : Y'[s] = X'[s] @ T[e] fp32 [S, n] mid  : H'[s] =
//   had128(silu(had128(Yg') * svh_g[e]) *
//                         (had128(Yu') * svh_u[e]) * suh_d[e])           fp16
//                         [S, n]
//   post : out[t] = sum_j w[s] * had128(O'[s]) * svh_d[e]              fp32 [T,
//   k]
// W = diag(suh) H T H diag(svh), H = blockwise Sylvester-128 / sqrt(128).
//
// Trellis layout (exllamav3): int16 [E, k/16, n/16, 16K]; each 16x16 tile is a
// tail-biting 256K-bit stream (uint16 pairs half-swapped, i.e. MSB-first when
// read as little-endian u32), position p's state = the 16 bits ending at bit
// (p + 1) K. The tile order maps positions to tile elements:
//   colmajor: p -> (k = p % 16, n = p / 16). A lane reads 8 consecutive k of
//             one column per run: two runs per tile.
//   sm80    : exllamav3's default (m16n8k16 B-fragment order). Column c's 16
//             values are four runs of 4 positions, 8K bits apart; one
//             m8n8k4 per run, so a lane decodes 4 short runs from 4-6 words.
// Measured on V100 SXM2 (per-rank GLM-5.3 expert shapes), sm80 vs colmajor:
// +4-13% at K = 2, +12-20% at K = 3 in decode and none in grouped prefill at
// K = 3. Both are exact.
//
// All launches are shape-static per slot count and read expert ids on the
// device, so the decode path is CUDA-graph capturable. The grouped variant
// (prefill) decodes each expert once per <= 8 slots routed to it.

#include <torch/all.h>
#include <torch/library.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace {

constexpr int kWarps = 4;      // gemv: warps per block, each a k-slice
constexpr int kGlueWarps = 4;  // glue kernels: one warp per 128-chunk
constexpr int kMaxGridY = 65535;

__device__ __forceinline__ void mma884(float (&d)[8], const uint32_t (&a)[2],
                                       const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
        "+f"(d[6]), "+f"(d[7])
      : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ half2 mcg2(uint32_t s0, uint32_t s1) {
  uint32_t x0 = s0 * 0xCBAC1FEDu, x1 = s1 * 0xCBAC1FEDu;
  asm("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x0));
  asm("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x1));
  half2 h0 = *reinterpret_cast<half2*>(&x0);
  half2 h1 = *reinterpret_cast<half2*>(&x1);
  return __hadd2(__lows2half2(h0, h1), __highs2half2(h0, h1));
}

// N consecutive positions from the window starting at bit o of (wa, wb).
template <int K, int N>
__device__ __forceinline__ void decode_run(uint32_t wa, uint32_t wb, int o,
                                           uint32_t (&b)[N / 2]) {
  uint64_t v = (static_cast<uint64_t>(wa) << 32) | wb;
  uint32_t st[N];
#pragma unroll
  for (int i = 0; i < N; ++i) {
    st[i] = static_cast<uint32_t>(v >> (48 - o - i * K)) & 0xffffu;
  }
#pragma unroll
  for (int j = 0; j < N / 2; ++j) {
    half2 r = mcg2(st[2 * j], st[2 * j + 1]);
    b[j] = *reinterpret_cast<uint32_t*>(&r);
  }
}

// A warp's two-tile block (16K words) is loaded once, coalesced; lanes fetch
// their window words by shuffle.
template <int K>
struct TileWords {
  uint32_t r0, r1;
  __device__ __forceinline__ void load(const uint32_t* __restrict__ blk,
                                       int lane) {
    if (K == 2) {
      r0 = blk[lane];
      r1 = 0;
    } else if (2 * lane < 16 * K) {
      uint2 v = reinterpret_cast<const uint2*>(blk)[lane];
      r0 = v.x;
      r1 = v.y;
    } else {
      r0 = 0;
      r1 = 0;
    }
  }
  __device__ __forceinline__ uint32_t word(int j) const {
    if (K == 2) return __shfl_sync(0xffffffffu, r0, j);
    uint32_t x0 = __shfl_sync(0xffffffffu, r0, j >> 1);
    uint32_t x1 = __shfl_sync(0xffffffffu, r1, j >> 1);
    return (j & 1) ? x1 : x0;
  }
};

// Lane-constant window geometry for tile t (0/1 within the strip), column c.
// colmajor: runs h = 0, 1 at positions 16c + 8h. sm80: run 0 only (the base);
// runs 1-3 follow at +8K bits each.
template <int K, bool kColMajor>
struct Window {
  static constexpr int kRuns = kColMajor ? 2 : 1;
  int ja[kRuns], jb[kRuns], o[kRuns];
  __device__ __forceinline__ Window(int t, int c) {
    constexpr int kTW = 8 * K, kLB = 256 * K;
#pragma unroll
    for (int r = 0; r < kRuns; ++r) {
      int pos0 = kColMajor ? 16 * c + 8 * r : 32 * (c & 7) + 4 * (c >> 3);
      int s = (((pos0 + 1) * K - 16) % kLB + kLB) % kLB;
      ja[r] = t * kTW + (s >> 5);
      jb[r] = t * kTW + ((s >> 5) + 1) % kTW;
      o[r] = s & 31;
    }
  }
};

// One 16-row k-tile of this lane's column. xr points at the tile's 16
// activations (row 0 of the m8n8k4 A operand; other rows pass on = false).
template <int K, bool kColMajor>
__device__ __forceinline__ void tile_mma(float (&d)[8], const TileWords<K>& tw,
                                         const Window<K, kColMajor>& win,
                                         const half* __restrict__ xr, bool on) {
  if constexpr (kColMajor) {
    uint32_t bb[2][4];
    decode_run<K, 8>(tw.word(win.ja[0]), tw.word(win.jb[0]), win.o[0], bb[0]);
    decode_run<K, 8>(tw.word(win.ja[1]), tw.word(win.jb[1]), win.o[1], bb[1]);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      uint32_t b0[2] = {bb[h][0], bb[h][1]}, b1[2] = {bb[h][2], bb[h][3]};
      uint4 xa = on ? *reinterpret_cast<const uint4*>(xr + 8 * h)
                    : make_uint4(0, 0, 0, 0);
      uint32_t a0[2] = {xa.x, xa.y}, a1[2] = {xa.z, xa.w};
      mma884(d, a0, b0);
      mma884(d, a1, b1);
    }
  } else {
    constexpr int kTW = 8 * K, kNW = ((24 * K) >> 5) + 3;
    uint4 lo =
        on ? *reinterpret_cast<const uint4*>(xr) : make_uint4(0, 0, 0, 0);
    uint4 hi =
        on ? *reinterpret_cast<const uint4*>(xr + 8) : make_uint4(0, 0, 0, 0);
    const uint32_t xw[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    int base = win.ja[0] % kTW, t0 = win.ja[0] - base, o0 = win.o[0] & 31;
    uint32_t w[kNW];
#pragma unroll
    for (int m = 0; m < kNW; ++m) w[m] = tw.word(t0 + (base + m) % kTW);
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      // Run u holds rows (2u, 2u+1, 2u+8, 2u+9): A = activation pairs u, u+4.
      constexpr int kStep = 8 * K;
      const int ml = (kStep * u) >> 5, ol = (kStep * u) & 31;
      const int ou = o0 + ol;
      const bool carry = ou >= 32;
      uint32_t b[2];
      decode_run<K, 4>(carry ? w[ml + 1] : w[ml], carry ? w[ml + 2] : w[ml + 1],
                       ou & 31, b);
      uint32_t a[2] = {xw[u], xw[u + 4]};
      mma884(d, a, b);
    }
  }
}

// grid (n / 32, S or 2S), block kWarps warps. Slot s multiplies expert eid[s].
// M = 1: row 0 of the m8n8k4 tile carries the activation.
template <int K, bool kColMajor>
__global__ void __launch_bounds__(kWarps * 32)
    exl3_gemv_kernel(const uint32_t* __restrict__ trellis,
                     const int* __restrict__ eid, const half* __restrict__ x,
                     float* __restrict__ y, int k, int n, int ktiles_per_warp,
                     int64_t expert_words, int S,
                     const uint32_t* __restrict__ trellis2,
                     const half* __restrict__ x2, float* __restrict__ y2) {
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int s = blockIdx.y;
  if (s >= S) {  // second projection of a fused gate + up launch
    s -= S;
    trellis = trellis2;
    x = x2;
    y = y2;
  }
  int n0 = blockIdx.x * 32;
  int q = (lane >> 2) & 3;
  int nw = 8 * q + (lane >> 4) * 4 + lane % 4;
  int t = nw >> 4, c = nw & 15;
  int row = (lane >> 4) * 4 + lane % 4;
  int tiles_n = n >> 4;
  int tk0 = warp * ktiles_per_warp;
  const uint32_t* tr = trellis + static_cast<int64_t>(eid[s]) * expert_words;
  const half* xr = x + static_cast<int64_t>(s) * k;
  constexpr int kTW = 8 * K;
  Window<K, kColMajor> win(t, c);

  float d[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  for (int it = 0; it < ktiles_per_warp; ++it) {
    int tk = tk0 + it;
    TileWords<K> tw;
    tw.load(tr + (static_cast<int64_t>(tk) * tiles_n + (n0 >> 4)) * kTW, lane);
    tile_mma<K, kColMajor>(d, tw, win, xr + tk * 16, row == 0);
  }

  // D fragment: lane L, reg i -> row (L&1) + ((i>>1)&1)*2 + (L>=16 ? 4 : 0),
  // col ((i>>2)&1)*4 + (L&2) + (i&1)
  __shared__ float red[kWarps][32];
  if (lane < 16 && !(lane & 1)) {
    red[warp][8 * q + (lane & 2)] = d[0];
    red[warp][8 * q + (lane & 2) + 1] = d[1];
    red[warp][8 * q + 4 + (lane & 2)] = d[4];
    red[warp][8 * q + 4 + (lane & 2) + 1] = d[5];
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    float acc = 0.f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) acc += red[w][threadIdx.x];
    y[static_cast<int64_t>(s) * n + n0 + threadIdx.x] = acc;
  }
}

// Grouped variant (prefill): a group is up to 8 slots routed to the same
// expert; its tiles are decoded once and multiplied against all 8 rows.
// rows[g * 8 + m] = slot or -1. Launches are sized for the worst case; groups
// with rows[g * 8] < 0 exit immediately.
template <int K, bool kColMajor>
__global__ void __launch_bounds__(kWarps * 32) exl3_gemv_grouped_kernel(
    const uint32_t* __restrict__ trellis, const int* __restrict__ gexp,
    const int* __restrict__ rows, const half* __restrict__ x,
    float* __restrict__ y, int k, int n, int ktiles_per_warp,
    int64_t expert_words, int G, const uint32_t* __restrict__ trellis2,
    const half* __restrict__ x2, float* __restrict__ y2) {
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int g = blockIdx.y;
  if (g >= G) {
    g -= G;
    trellis = trellis2;
    x = x2;
    y = y2;
  }
  const int* gr = rows + static_cast<int64_t>(g) * 8;
  if (gr[0] < 0) return;
  int n0 = blockIdx.x * 32;
  int q = (lane >> 2) & 3;
  int nw = 8 * q + (lane >> 4) * 4 + lane % 4;
  int t = nw >> 4, c = nw & 15;
  int row = (lane >> 4) * 4 + lane % 4;  // A-operand row held by this lane
  int slot = gr[row];
  int tiles_n = n >> 4;
  int tk0 = warp * ktiles_per_warp;
  const uint32_t* tr = trellis + static_cast<int64_t>(gexp[g]) * expert_words;
  const half* xr = x + static_cast<int64_t>(slot < 0 ? 0 : slot) * k;
  constexpr int kTW = 8 * K;
  Window<K, kColMajor> win(t, c);

  float d[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  for (int it = 0; it < ktiles_per_warp; ++it) {
    int tk = tk0 + it;
    TileWords<K> tw;
    tw.load(tr + (static_cast<int64_t>(tk) * tiles_n + (n0 >> 4)) * kTW, lane);
    tile_mma<K, kColMajor>(d, tw, win, xr + tk * 16, slot >= 0);
  }

  __shared__ float red[kWarps][8][32];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    red[warp][(lane & 1) + ((i >> 1) & 1) * 2 + (lane >= 16 ? 4 : 0)]
       [8 * q + ((i >> 2) & 1) * 4 + (lane & 2) + (i & 1)] = d[i];
  }
  __syncthreads();
  for (int e = threadIdx.x; e < 256; e += kWarps * 32) {
    int m = e >> 5, nn = e & 31;
    int so = gr[m];
    if (so < 0) continue;
    float acc = 0.f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) acc += red[w][m][nn];
    y[static_cast<int64_t>(so) * n + n0 + nn] = acc;
  }
}

// 128-point Walsh-Hadamard (Sylvester order) in registers: one warp per
// 128-chunk, lane l holds elements 4l .. 4l+3.
__device__ __forceinline__ void fwht128(float (&v)[4]) {
  int lane = threadIdx.x & 31;
  float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
  v[0] = a + c;
  v[1] = b + d;
  v[2] = a - c;
  v[3] = b - d;
#pragma unroll
  for (int dl = 1; dl < 32; dl <<= 1) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float o = __shfl_xor_sync(0xffffffffu, v[i], dl);
      v[i] = (lane & dl) ? o - v[i] : v[i] + o;
    }
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] *= 0.08838834764831845f;  // 1 / sqrt(128)
}

__device__ __forceinline__ void load4h(const half* p, float (&v)[4]) {
  uint2 r = *reinterpret_cast<const uint2*>(p);
  float2 f0 = __half22float2(*reinterpret_cast<half2*>(&r.x));
  float2 f1 = __half22float2(*reinterpret_cast<half2*>(&r.y));
  v[0] = f0.x;
  v[1] = f0.y;
  v[2] = f1.x;
  v[3] = f1.y;
}

__device__ __forceinline__ void store4h(half* p, const float (&v)[4]) {
  half2 h0 = __floats2half2_rn(v[0], v[1]), h1 = __floats2half2_rn(v[2], v[3]);
  uint2 r;
  r.x = *reinterpret_cast<uint32_t*>(&h0);
  r.y = *reinterpret_cast<uint32_t*>(&h1);
  *reinterpret_cast<uint2*>(p) = r;
}

__device__ __forceinline__ int glue_offset() {
  return (blockIdx.x * kGlueWarps + (threadIdx.x >> 5)) * 128 +
         (threadIdx.x & 31) * 4;
}

__global__ void exl3_pre_kernel(const half* __restrict__ x,
                                const int* __restrict__ eid,
                                const half* __restrict__ suh,
                                half* __restrict__ out, int k, int topk, int S,
                                const half* __restrict__ suh2,
                                half* __restrict__ out2) {
  int s = blockIdx.y;
  if (s >= S) {
    s -= S;
    suh = suh2;
    out = out2;
  }
  int o = glue_offset();
  float v[4], g[4];
  load4h(x + static_cast<int64_t>(s / topk) * k + o, v);
  load4h(suh + static_cast<int64_t>(eid[s]) * k + o, g);
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] *= g[i];
  fwht128(v);
  store4h(out + static_cast<int64_t>(s) * k + o, v);
}

__global__ void exl3_mid_kernel(const float* __restrict__ yg,
                                const float* __restrict__ yu,
                                const int* __restrict__ eid,
                                const half* __restrict__ svh_g,
                                const half* __restrict__ svh_u,
                                const half* __restrict__ suh_d,
                                half* __restrict__ out, int n) {
  int s = blockIdx.y;
  int o = glue_offset();
  int64_t e = static_cast<int64_t>(eid[s]) * n + o;
  float g[4], u[4], sg[4], su[4], sd[4];
  float4 a =
      *reinterpret_cast<const float4*>(yg + static_cast<int64_t>(s) * n + o);
  float4 b =
      *reinterpret_cast<const float4*>(yu + static_cast<int64_t>(s) * n + o);
  g[0] = a.x;
  g[1] = a.y;
  g[2] = a.z;
  g[3] = a.w;
  u[0] = b.x;
  u[1] = b.y;
  u[2] = b.z;
  u[3] = b.w;
  fwht128(g);
  fwht128(u);
  load4h(svh_g + e, sg);
  load4h(svh_u + e, su);
  load4h(suh_d + e, sd);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float gg = g[i] * sg[i];
    g[i] = gg / (1.f + __expf(-gg)) * (u[i] * su[i]) * sd[i];
  }
  fwht128(g);
  store4h(out + static_cast<int64_t>(s) * n + o, g);
}

__global__ void exl3_post_kernel(const float* __restrict__ yd,
                                 const int* __restrict__ eid,
                                 const float* __restrict__ w,
                                 const half* __restrict__ svh_d,
                                 float* __restrict__ out, int n, int topk) {
  int t = blockIdx.y;
  int o = glue_offset();
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int j = 0; j < topk; ++j) {
    int s = t * topk + j;
    float v[4], sv[4];
    float4 a =
        *reinterpret_cast<const float4*>(yd + static_cast<int64_t>(s) * n + o);
    v[0] = a.x;
    v[1] = a.y;
    v[2] = a.z;
    v[3] = a.w;
    fwht128(v);
    load4h(svh_d + static_cast<int64_t>(eid[s]) * n + o, sv);
    float ww = w[s];
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[i] += ww * v[i] * sv[i];
  }
  *reinterpret_cast<float4*>(out + static_cast<int64_t>(t) * n + o) =
      make_float4(acc[0], acc[1], acc[2], acc[3]);
}

void check_cuda(const torch::Tensor& t, const char* name, at::ScalarType dtype,
                int dim, const torch::Tensor& ref) {
  TORCH_CHECK(t.is_cuda() && t.device() == ref.device(), name,
              " must be on the same CUDA device as the input");
  TORCH_CHECK(t.scalar_type() == dtype, name, " has the wrong dtype");
  TORCH_CHECK(t.dim() == dim, name, " must have ", dim, " dimensions");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

// trellis [E, k/16, n/16, 16K] -> (k, n, K)
std::tuple<int, int, int> trellis_dims(const torch::Tensor& trellis,
                                       const torch::Tensor& ref) {
  check_cuda(trellis, "trellis", at::kShort, 4, ref);
  TORCH_CHECK(trellis.size(3) % 16 == 0, "trellis: last dim must be 16 * K");
  int K = static_cast<int>(trellis.size(3) / 16);
  TORCH_CHECK(K == 2 || K == 3 || K == 4, "EXL3 MoE: K must be 2, 3 or 4");
  int k = static_cast<int>(trellis.size(1) * 16);
  int n = static_cast<int>(trellis.size(2) * 16);
  TORCH_CHECK((k / 16) % kWarps == 0 && n % 32 == 0,
              "EXL3 MoE: k must be a multiple of 64 and n of 32");
  return {k, n, K};
}

template <bool kColMajor>
void launch_gemv(bool grouped, int K, dim3 grid, cudaStream_t stream,
                 const uint32_t* tr, const int* ids, const int* rows,
                 const half* x, float* y, int k, int n, int64_t ew, int count,
                 const uint32_t* tr2, const half* x2, float* y2) {
  const int kt = k / 16 / kWarps;
#define EXL3_LAUNCH(KK)                                                        \
  if (grouped) {                                                               \
    exl3_gemv_grouped_kernel<KK, kColMajor><<<grid, kWarps * 32, 0, stream>>>( \
        tr, ids, rows, x, y, k, n, kt, ew, count, tr2, x2, y2);                \
  } else {                                                                     \
    exl3_gemv_kernel<KK, kColMajor><<<grid, kWarps * 32, 0, stream>>>(         \
        tr, ids, x, y, k, n, kt, ew, count, tr2, x2, y2);                      \
  }
  if (K == 2) {
    EXL3_LAUNCH(2)
  } else if (K == 3) {
    EXL3_LAUNCH(3)
  } else {
    EXL3_LAUNCH(4)
  }
#undef EXL3_LAUNCH
}

}  // namespace

// xg = had128(x[t] * g_suh[e]) per slot; with out2/suh2 also the up input.
void exl3_moe_pre_out(torch::Tensor out, std::optional<torch::Tensor> out2,
                      torch::Tensor x, torch::Tensor topk_ids,
                      torch::Tensor suh, std::optional<torch::Tensor> suh2) {
  check_cuda(x, "x", at::kHalf, 2, x);
  check_cuda(topk_ids, "topk_ids", at::kInt, 2, x);
  check_cuda(out, "out", at::kHalf, 2, x);
  check_cuda(suh, "suh", at::kHalf, 2, x);
  TORCH_CHECK(out2.has_value() == suh2.has_value(),
              "out2 and suh2 go together");
  const int k = static_cast<int>(x.size(1));
  const int topk = static_cast<int>(topk_ids.size(1));
  const int S = static_cast<int>(topk_ids.numel());
  TORCH_CHECK(topk_ids.size(0) == x.size(0), "topk_ids rows != tokens");
  TORCH_CHECK(k % (kGlueWarps * 128) == 0,
              "EXL3 MoE: k must be a multiple of 512");
  TORCH_CHECK(out.size(0) >= S && out.size(1) == k && suh.size(1) == k,
              "pre: shape mismatch");
  const int fused = out2.has_value() ? 2 : 1;
  if (out2) {
    check_cuda(*out2, "out2", at::kHalf, 2, x);
    check_cuda(*suh2, "suh2", at::kHalf, 2, x);
    TORCH_CHECK(out2->size(0) >= S && out2->size(1) == k && suh2->size(1) == k,
                "pre: shape mismatch");
  }
  if (S == 0) return;
  TORCH_CHECK(fused * S <= kMaxGridY, "EXL3 MoE: too many slots per launch");
  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  exl3_pre_kernel<<<dim3(k / (kGlueWarps * 128), fused * S), kGlueWarps * 32, 0,
                    stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      topk_ids.data_ptr<int>(),
      reinterpret_cast<const half*>(suh.data_ptr<at::Half>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()), k, topk, S,
      out2 ? reinterpret_cast<const half*>(suh2->data_ptr<at::Half>())
           : nullptr,
      out2 ? reinterpret_cast<half*>(out2->data_ptr<at::Half>()) : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// y[s] = x[s] @ T[expert(s)] in the trellis basis. Plain: expert_ids [S] holds
// each slot's expert. Grouped: expert_ids [G] holds each group's expert and
// group_rows [G, 8] its slots (-1 = empty). With y2/x2/trellis2, the second
// projection (up) runs in the same launch.
void exl3_moe_gemv_out(torch::Tensor y, std::optional<torch::Tensor> y2,
                       torch::Tensor x, std::optional<torch::Tensor> x2,
                       torch::Tensor trellis,
                       std::optional<torch::Tensor> trellis2,
                       torch::Tensor expert_ids,
                       std::optional<torch::Tensor> group_rows, bool colmajor) {
  check_cuda(x, "x", at::kHalf, 2, x);
  check_cuda(y, "y", at::kFloat, 2, x);
  check_cuda(expert_ids, "expert_ids", at::kInt, 1, x);
  auto [k, n, K] = trellis_dims(trellis, x);
  TORCH_CHECK(x.size(1) == k && y.size(1) == n && y.size(0) >= x.size(0),
              "gemv: shape mismatch");
  const bool fused = y2.has_value();
  TORCH_CHECK(fused == x2.has_value() && fused == trellis2.has_value(),
              "y2, x2 and trellis2 go together");
  if (fused) {
    check_cuda(*y2, "y2", at::kFloat, 2, x);
    check_cuda(*x2, "x2", at::kHalf, 2, x);
    auto [k2, n2, K2] = trellis_dims(*trellis2, x);
    TORCH_CHECK(k2 == k && n2 == n && K2 == K &&
                    trellis2->size(0) == trellis.size(0) &&
                    x2->sizes() == x.sizes() && y2->sizes() == y.sizes(),
                "gemv: the fused projections must have the same shapes");
  }
  const bool grouped = group_rows.has_value();
  int count = static_cast<int>(expert_ids.size(0));
  if (grouped) {
    check_cuda(*group_rows, "group_rows", at::kInt, 2, x);
    TORCH_CHECK(group_rows->size(0) == count && group_rows->size(1) == 8,
                "group_rows must be [G, 8]");
  } else {
    TORCH_CHECK(count <= x.size(0), "expert_ids longer than x");
  }
  if (count == 0) return;
  TORCH_CHECK((fused ? 2 : 1) * count <= kMaxGridY,
              "EXL3 MoE: too many slots per launch");
  const int64_t ew = trellis[0].numel() / 2;  // u32 words per expert
  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(n / 32, (fused ? 2 : 1) * count);
  auto tr = reinterpret_cast<const uint32_t*>(trellis.data_ptr<int16_t>());
  auto tr2 =
      fused ? reinterpret_cast<const uint32_t*>(trellis2->data_ptr<int16_t>())
            : nullptr;
  auto xp = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
  auto x2p =
      fused ? reinterpret_cast<const half*>(x2->data_ptr<at::Half>()) : nullptr;
  float* y2p = fused ? y2->data_ptr<float>() : nullptr;
  const int* rows = grouped ? group_rows->data_ptr<int>() : nullptr;
  if (colmajor) {
    launch_gemv<true>(grouped, K, grid, stream, tr, expert_ids.data_ptr<int>(),
                      rows, xp, y.data_ptr<float>(), k, n, ew, count, tr2, x2p,
                      y2p);
  } else {
    launch_gemv<false>(grouped, K, grid, stream, tr, expert_ids.data_ptr<int>(),
                       rows, xp, y.data_ptr<float>(), k, n, ew, count, tr2, x2p,
                       y2p);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// out[s] = had128(silu(had128(yg) * g_svh) * had128(yu) * u_svh * d_suh)
void exl3_moe_mid_out(torch::Tensor out, torch::Tensor yg, torch::Tensor yu,
                      torch::Tensor topk_ids, torch::Tensor g_svh,
                      torch::Tensor u_svh, torch::Tensor d_suh) {
  check_cuda(yg, "yg", at::kFloat, 2, yg);
  check_cuda(yu, "yu", at::kFloat, 2, yg);
  check_cuda(out, "out", at::kHalf, 2, yg);
  check_cuda(topk_ids, "topk_ids", at::kInt, 2, yg);
  check_cuda(g_svh, "g_svh", at::kHalf, 2, yg);
  check_cuda(u_svh, "u_svh", at::kHalf, 2, yg);
  check_cuda(d_suh, "d_suh", at::kHalf, 2, yg);
  const int n = static_cast<int>(yg.size(1));
  const int S = static_cast<int>(topk_ids.numel());
  TORCH_CHECK(n % (kGlueWarps * 128) == 0,
              "EXL3 MoE: n must be a multiple of 512");
  TORCH_CHECK(yg.size(0) >= S && yu.sizes() == yg.sizes() && out.size(0) >= S &&
                  out.size(1) == n && g_svh.size(1) == n &&
                  u_svh.size(1) == n && d_suh.size(1) == n,
              "mid: shape mismatch");
  if (S == 0) return;
  TORCH_CHECK(S <= kMaxGridY, "EXL3 MoE: too many slots per launch");
  const at::cuda::OptionalCUDAGuard guard(yg.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  exl3_mid_kernel<<<dim3(n / (kGlueWarps * 128), S), kGlueWarps * 32, 0,
                    stream>>>(
      yg.data_ptr<float>(), yu.data_ptr<float>(), topk_ids.data_ptr<int>(),
      reinterpret_cast<const half*>(g_svh.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(u_svh.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(d_suh.data_ptr<at::Half>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// out[t] = sum_j w[t, j] * had128(yd[t * topk + j]) * d_svh[e]   (fp32 [T, k])
void exl3_moe_post_out(torch::Tensor out, torch::Tensor yd,
                       torch::Tensor topk_ids, torch::Tensor topk_weights,
                       torch::Tensor d_svh) {
  check_cuda(yd, "yd", at::kFloat, 2, yd);
  check_cuda(out, "out", at::kFloat, 2, yd);
  check_cuda(topk_ids, "topk_ids", at::kInt, 2, yd);
  check_cuda(topk_weights, "topk_weights", at::kFloat, 2, yd);
  check_cuda(d_svh, "d_svh", at::kHalf, 2, yd);
  const int k = static_cast<int>(yd.size(1));
  const int T = static_cast<int>(topk_ids.size(0));
  const int topk = static_cast<int>(topk_ids.size(1));
  TORCH_CHECK(k % (kGlueWarps * 128) == 0,
              "EXL3 MoE: k must be a multiple of 512");
  TORCH_CHECK(topk_weights.sizes() == topk_ids.sizes() &&
                  yd.size(0) >= T * topk && out.size(0) >= T &&
                  out.size(1) == k && d_svh.size(1) == k,
              "post: shape mismatch");
  if (T == 0) return;
  TORCH_CHECK(T <= kMaxGridY, "EXL3 MoE: too many tokens per launch");
  const at::cuda::OptionalCUDAGuard guard(yd.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  exl3_post_kernel<<<dim3(k / (kGlueWarps * 128), T), kGlueWarps * 32, 0,
                     stream>>>(
      yd.data_ptr<float>(), topk_ids.data_ptr<int>(),
      topk_weights.data_ptr<float>(),
      reinterpret_cast<const half*>(d_svh.data_ptr<at::Half>()),
      out.data_ptr<float>(), k, topk);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
