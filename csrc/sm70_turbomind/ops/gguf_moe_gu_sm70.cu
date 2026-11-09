// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Routed expert gate/up (+SiLU*up) for Flash-Next GGUF experts on SM70,
// expert-grouped mma. Expert weights are repacked once into dmv13's
// lane-interleaved tile planes (coalesced uint4 reads, same decode as the dense
// path). blockIdx.y = distinct-expert slot (first-occurrence order over the
// routes), blockIdx.x = 32-row tile of the rank's intermediate shard. The
// tokens routed to the expert form the mma M dimension (<= 8), so every expert
// tile is read and decoded once per layer regardless of how many verify tokens
// picked it. Output: fp16 hidden[route][n] = fp16(silu(fp16 g)) * fp16(u), the
// production FP16 boundary.
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <vector>
#define DMV_FULLK 0
namespace dmvns {
#include "gguf_dmv13_core.cuh"
}  // namespace dmvns

namespace moe {
using namespace dmvns;
constexpr int TN = 2, MAXT = 8, MAXR = 256, MAXC = 32;

// Q8_1 block as produced by the production quantizer (gguf_dp4a.cuh): half2 {d,
// sum} + 32 int8.
struct Q8Block {
  half2 ds;
  int8_t qs[32];
};
static_assert(sizeof(Q8Block) == 36);

__device__ __forceinline__ void quantize_q8_warp(Q8Block* out, float value) {
  const int lane = threadIdx.x % 32;
  float maximum = fabsf(value), sum = value;
#pragma unroll
  for (int offset = 16; offset; offset >>= 1) {
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
    sum += __shfl_xor_sync(0xffffffff, sum, offset);
  }
  const float d = maximum / 127.f;
  out->qs[lane] = maximum == 0.f ? 0 : int8_t(roundf(value / d));
  if (!lane) out->ds = __floats2half2_rn(d, sum);
}

struct EArgs {
  const uint4* gc;
  const uint4* gs;
  const uint4* uc;
  const uint4* us;
  long long cstride, sstride;  // per-expert plane strides in uint4
  const half* x;
  int ldx;
  const int* ids;
  int routes, top_k, n, K;
  half* hout;
  Q8Block* qout;  // when set, the activated intermediate is emitted as Q8_1
                  // blocks instead of fp16
  const uint4* tab;
};

template <int FMT, int KW>
__global__ void __launch_bounds__(32 * KW * TN) moe_gu(EArgs a) {
  extern __shared__ uint4 smem[];
  __shared__ int sid[MAXR], s_expert, s_cnt, s_route[MAXC], s_tok[MAXC];
  __shared__ half qint[MAXT][32];
  const int t = threadIdx.x, warp = t / 32, lane = t % 32;
  // Distinct experts in ascending id order: slot u = u-th set bit of a 512-bit
  // presence mask.
  __shared__ unsigned s_mask[16];
  if (t < 16) s_mask[t] = 0u;
  if (t == 0) s_expert = -1;
  __syncthreads();
  for (int r = t; r < a.routes; r += blockDim.x) {
    const int e = a.ids[r];
    sid[r] = e;
    atomicOr(&s_mask[e >> 5], 1u << (e & 31));
  }
  __syncthreads();
  if (t < 32) {
    const unsigned w = t < 16 ? s_mask[t] : 0u;
    int c = __popc(w), pre = c;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(0xffffffffu, pre, o);
      if (t >= o) pre += v;
    }
    const int before = pre - c, slot = blockIdx.y;
    if (t < 16 && slot >= before && slot < pre) {
      unsigned m = w;
      for (int q = 0; q < slot - before; ++q) m &= m - 1u;
      s_expert = t * 32 + (__ffs(m) - 1);
    }
  }
  __syncthreads();
  const int e = s_expert;
  if (e < 0) return;
  if (t < 32) {
    int cnt = 0;
    for (int base = 0; base < a.routes; base += 32) {
      const int r = base + t;
      const bool hit = r < a.routes && sid[r] == e;
      const unsigned bal = __ballot_sync(0xffffffffu, hit);
      if (hit) {
        const int j = cnt + __popc(bal & ((1u << t) - 1u));
        if (j < MAXC) {
          s_route[j] = r;
          s_tok[j] = r / a.top_k;
        }
      }
      cnt += __popc(bal);
    }
    if (t == 0) s_cnt = cnt < MAXC ? cnt : MAXC;
  }
  __syncthreads();
  const int cnt = s_cnt;
  const int tin = warp % TN, kslot = warp / TN;
  Seg sg{};
  sg.codes = (tin ? a.uc : a.gc) + e * a.cstride;
  sg.scale = (tin ? a.us : a.gs) + e * a.sstride;
  sg.high = nullptr;
  sg.n = a.n;
  sg.fmt = FMT;
  const int G = a.K / 128;
  uint4* xs = smem;
  half2* lut =
      reinterpret_cast<half2*>(smem + KW * 256);  // up to TAB_VECS_IQ2S uint4
  // Tokens that picked this expert are the mma M dimension, eight at a time.
  for (int c0 = 0; c0 < cnt; c0 += MAXT) {
    const int cc = cnt - c0 < MAXT ? cnt - c0 : MAXT;
    float acc[8] = {};
    body6<FMT, KW, TN>(sg, blockIdx.x, true, kslot, tin, 0, G, 4 * G, G, a.x,
                       a.ldx, cc, xs, lut, acc, a.tab, s_tok + c0);
    __syncthreads();
    float* red = reinterpret_cast<float*>(smem);
#pragma unroll
    for (int i = 0; i < 8; ++i) red[warp * 256 + lane * 8 + i] = acc[i];
    __syncthreads();
    for (int v = t; v < 256;
         v += blockDim.x) {  // 256 outputs (8 tokens x 32 rows) per tile
      float g = 0.f, u = 0.f;
#pragma unroll
      for (int k = 0; k < KW; ++k) {
        g += red[(k * TN + 0) * 256 + v];
        u += red[(k * TN + 1) * 256 + v];
      }
      const int lv = v >> 3, i = v & 7;
      const int token = (i & 2) | ((lv & 16) ? 4 : 0) | (lv & 1);
      const int col = (i & 1) | (((lv >> 1) & 1) << 1) | ((i >> 2) << 2);
      const int lrow = ((lv >> 2) & 3) * 8 + col;
      const int row = blockIdx.x * 32 + lrow;
      if (token < cc && row < a.n) {
        const float g16 = __half2float(__float2half_rn(g));
        const float u16 = __half2float(__float2half_rn(u));
        const half silu = __float2half_rn(g16 / (1.f + expf(-g16)));
        const half value = __hmul(silu, __float2half_rn(u16));
        if (a.qout)
          qint[token][lrow] = value;
        else
          a.hout[static_cast<int64_t>(s_route[c0 + token]) * a.n + row] = value;
      }
    }
    if (a.qout) {
      __syncthreads();
      if (warp < cc)  // one 32-row tile is exactly one Q8_1 block of this route
        quantize_q8_warp(
            a.qout + static_cast<int64_t>(s_route[c0 + warp]) * (a.n / 32) +
                blockIdx.x,
            __half2float(qint[warp][lane]));
    }
    __syncthreads();
  }
}

// Routed down projection + weighted unroute, expert-grouped. CTA = 32 output
// columns; warps take the distinct experts round-robin; mma rows are tokens
// (zero rows for tokens that did not pick the expert), so each expert tile is
// read once and every token's contribution lands in its own row. Per route:
// fp16(down) * route_weight accumulated in fp32 (production FP16 boundary);
// experts summed in ascending-id order per warp, warps summed in order.
template <int FMT, int NWD>
__global__ void __launch_bounds__(32 * NWD)
    moe_down(const uint4* dc, const uint4* ds, long long cstride,
             long long sstride, const half* h, const float* rw, const int* ids,
             int routes, int top_k, int M, int kin, int Kp, half* out, int n,
             float* ws, int* cnt) {
  constexpr int HS = 160 + 8;  // staged hidden row (halves); kin <= 160
  __shared__ __align__(16) half hs[MAXR / 2][HS];
  __shared__ float sw[MAXR / 2];
  __shared__ int unique_experts[MAXR / 2], rm[MAXR / 2][MAXT], s_nu;
  __shared__ unsigned s_mask[16], s_pre[16];
  __shared__ float red[NWD][256];
  const int t = threadIdx.x, warp = t / 32, lane = t % 32;
  if (t < 16) s_mask[t] = 0u;
  for (int i = t; i < (MAXR / 2) * MAXT; i += blockDim.x) (&rm[0][0])[i] = -1;
  __syncthreads();
  for (int r = t; r < routes; r += blockDim.x) {
    const int e = ids[r];
    atomicOr(&s_mask[e >> 5], 1u << (e & 31));
    sw[r] = rw[r];
  }
  // stage the routed hidden rows (k < kin), zero padding up to Kp
  for (int i = t; i < routes * (kin / 8); i += blockDim.x) {
    const int r = i / (kin / 8), c = (i % (kin / 8)) * 8;
    *reinterpret_cast<uint4*>(&hs[r][c]) =
        *reinterpret_cast<const uint4*>(h + static_cast<int64_t>(r) * kin + c);
  }
  __syncthreads();
  if (t < 32) {
    const unsigned w = t < 16 ? s_mask[t] : 0u;
    int pre = __popc(w);
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(0xffffffffu, pre, o);
      if (t >= o) pre += v;
    }
    if (t < 16) s_pre[t] = pre - __popc(w);
    if (t == 31) s_nu = pre;
  }
  __syncthreads();
  for (int r = t; r < routes; r += blockDim.x) {
    const int e = ids[r];
    const unsigned w = s_mask[e >> 5];
    const int u = s_pre[e >> 5] + __popc(w & ((1u << (e & 31)) - 1u));
    unique_experts[u] = e;
    rm[u][r / top_k] = r;
  }
  __syncthreads();
  const int nu = s_nu, G = Kp / 128;
  const int r8 = (lane & 3) + ((lane & 16) ? 4 : 0);
  float tot[8] = {};
  // Experts are split over gridDim.y CTAs per column tile (expert u -> CTA u %
  // S, warp (u / S) % 8).
  const int S = gridDim.y, sp = blockIdx.y;
  const int ksteps = kin / 32;
  for (int u = sp + S * warp; u < nu; u += S * NWD) {
    const int e = unique_experts[u];
    Seg sg{};
    sg.codes = dc + e * cstride;
    sg.scale = ds + e * sstride;
    sg.n = n;
    sg.fmt = FMT;
    const int myroute = r8 < M ? rm[u][r8] : -1;
    Ld<FMT> L0, L1;
    load<FMT>(L0, sg, blockIdx.x, 0, 4 * G, G, lane);
    if (G > 1) load<FMT>(L1, sg, blockIdx.x, 1, 4 * G, G, lane);
    float acc[8] = {};
#pragma unroll
    for (int g = 0; g < 2; ++g) {
      if (g >= G) break;
#pragma unroll
      for (int st = 0; st < 4; ++st) {
        if (g * 4 + st >= ksteps) break;
        uint4 xa[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
          xa[j] = myroute >= 0 ? *reinterpret_cast<const uint4*>(
                                     &hs[myroute][g * 128 + (st * 4 + j) * 8])
                               : make_uint4(0, 0, 0, 0);
        uint32_t hw[16];
        decode<FMT>(g ? L1 : L0, st, hw, nullptr);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          mma(acc, xa[j].x, xa[j].y, hw[4 * j], hw[4 * j + 1]);
          mma(acc, xa[j].z, xa[j].w, hw[4 * j + 2], hw[4 * j + 3]);
        }
      }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int token = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
      const int r = token < M ? rm[u][token] : -1;
      if (r >= 0) tot[i] += __half2float(__float2half_rn(acc[i])) * sw[r];
    }
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) red[warp][lane * 8 + i] = tot[i];
  __syncthreads();
  __shared__ bool s_last;
  float sum = 0.f;
  const bool outv = t < 256;
  if (outv)
#pragma unroll
    for (int w = 0; w < NWD; ++w) sum += red[w][t];
  if (S > 1) {
    if (outv)
      __stcg(ws + (static_cast<size_t>(blockIdx.x) * S + sp) * 256 + t, sum);
    __threadfence();
    __syncthreads();
    if (t == 0) s_last = atomicAdd(cnt + blockIdx.x, 1) == S - 1;
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    sum = 0.f;
    if (outv)
      for (int p = 0; p < S; ++p)
        sum += __ldcg(ws + (static_cast<size_t>(blockIdx.x) * S + p) * 256 + t);
    if (t == 0) cnt[blockIdx.x] = 0;
  }
  {
    const int v = t;
    const int lv = v >> 3, i = v & 7;
    const int token = (i & 2) | ((lv & 16) ? 4 : 0) | (lv & 1);
    const int col = blockIdx.x * 32 + ((lv >> 2) & 3) * 8 +
                    ((i & 1) | (((lv >> 1) & 1) << 1) | ((i >> 2) << 2));
    if (outv && token < M && col < n)
      out[static_cast<int64_t>(token) * n + col] = __float2half_rn(sum);
  }
}

// Down + unroute, v2: grid (column tile, expert chunk). Each warp owns one
// distinct expert of its CTA's chunk for the tile (no hidden-row staging: A
// rows come straight from L2), the CTA sums its warps, and the last CTA of the
// tile adds the chunk partials in chunk order -> ascending-expert order
// overall, deterministic.
template <int FMT, int W>
__global__ void __launch_bounds__(32 * W)
    moe_down2(const uint4* dc, const uint4* ds, long long cstride,
              long long sstride, const half* h, const float* rw, const int* ids,
              int routes, int top_k, int M, int kin, half* out, int n,
              float* ws, int* cnt) {
  __shared__ int unique_experts[W], rm[W][MAXT], s_nu;
  __shared__ unsigned s_mask[16], s_pre[16];
  __shared__ float red[W][256];
  __shared__ bool s_last;
  const int t = threadIdx.x, warp = t / 32, lane = t % 32, S = gridDim.y,
            sp = blockIdx.y;
  if (t < 16) s_mask[t] = 0u;
  for (int i = t; i < W * MAXT; i += blockDim.x) (&rm[0][0])[i] = -1;
  __syncthreads();
  for (int r = t; r < routes; r += blockDim.x) {
    const int e = ids[r];
    atomicOr(&s_mask[e >> 5], 1u << (e & 31));
  }
  __syncthreads();
  if (t < 32) {
    const unsigned w = t < 16 ? s_mask[t] : 0u;
    int pre = __popc(w);
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(0xffffffffu, pre, o);
      if (t >= o) pre += v;
    }
    if (t < 16) s_pre[t] = pre - __popc(w);
    if (t == 31) s_nu = pre;
  }
  __syncthreads();
  for (int r = t; r < routes; r += blockDim.x) {
    const int e = ids[r];
    const unsigned w = s_mask[e >> 5];
    const int u = s_pre[e >> 5] + __popc(w & ((1u << (e & 31)) - 1u)) - sp * W;
    if (u >= 0 && u < W) {
      unique_experts[u] = e;
      rm[u][r / top_k] = r;
    }
  }
  __syncthreads();
  const int nu = s_nu, ksteps = kin / 32, G = (ksteps + 3) / 4;
  const int r8 = (lane & 3) + ((lane & 16) ? 4 : 0);
  float tot[8] = {};
  const int u = sp * W + warp;
  if (u < nu) {
    Seg sg{};
    const int e = unique_experts[warp];
    sg.codes = dc + e * cstride;
    sg.scale = ds + e * sstride;
    sg.n = n;
    sg.fmt = FMT;
    const int myroute = r8 < M ? rm[warp][r8] : -1;
    const half* hr = h + static_cast<int64_t>(myroute < 0 ? 0 : myroute) * kin;
    Ld<FMT> L0, L1;
    load<FMT>(L0, sg, blockIdx.x, 0, ksteps, G, lane);
    if (G > 1) load<FMT>(L1, sg, blockIdx.x, 1, ksteps, G, lane);
    float acc[8] = {};
#pragma unroll
    for (int g = 0; g < 2; ++g) {
      if (g >= G) break;
#pragma unroll
      for (int st = 0; st < 4; ++st) {
        if (g * 4 + st >= ksteps) break;
        uint4 xa[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
          xa[j] = myroute >= 0 ? __ldg(reinterpret_cast<const uint4*>(
                                     hr + g * 128 + (st * 4 + j) * 8))
                               : make_uint4(0, 0, 0, 0);
        uint32_t hw[16];
        decode<FMT>(g ? L1 : L0, st, hw, nullptr);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          mma(acc, xa[j].x, xa[j].y, hw[4 * j], hw[4 * j + 1]);
          mma(acc, xa[j].z, xa[j].w, hw[4 * j + 2], hw[4 * j + 3]);
        }
      }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int token = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
      const int r = token < M ? rm[warp][token] : -1;
      if (r >= 0) tot[i] = __half2float(__float2half_rn(acc[i])) * rw[r];
    }
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) red[warp][lane * 8 + i] = tot[i];
  __syncthreads();
  for (int v = t; v < 256; v += blockDim.x) {
    float sum = 0.f;
#pragma unroll
    for (int w = 0; w < W; ++w) sum += red[w][v];
    if (S > 1)
      __stcg(ws + (static_cast<size_t>(blockIdx.x) * S + sp) * 256 + v, sum);
    else {
      const int lv = v >> 3, i = v & 7;
      const int token = (i & 2) | ((lv & 16) ? 4 : 0) | (lv & 1);
      const int col = blockIdx.x * 32 + ((lv >> 2) & 3) * 8 +
                      ((i & 1) | (((lv >> 1) & 1) << 1) | ((i >> 2) << 2));
      if (token < M && col < n)
        out[static_cast<int64_t>(token) * n + col] = __float2half_rn(sum);
    }
  }
  if (S == 1) return;
  __threadfence();
  __syncthreads();
  if (t == 0) s_last = atomicAdd(cnt + blockIdx.x, 1) == S - 1;
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  for (int v = t; v < 256; v += blockDim.x) {
    float sum = 0.f;
    for (int p = 0; p < S; ++p)
      sum += __ldcg(ws + (static_cast<size_t>(blockIdx.x) * S + p) * 256 + v);
    const int lv = v >> 3, i = v & 7;
    const int token = (i & 2) | ((lv & 16) ? 4 : 0) | (lv & 1);
    const int col = blockIdx.x * 32 + ((lv >> 2) & 3) * 8 +
                    ((i & 1) | (((lv >> 1) & 1) << 1) | ((i >> 2) << 2));
    if (token < M && col < n)
      out[static_cast<int64_t>(token) * n + col] = __float2half_rn(sum);
  }
  if (t == 0) cnt[blockIdx.x] = 0;
}
}  // namespace moe

void gguf_moe_gate_up_sm70_out(torch::Tensor hout, torch::Tensor x,
                               torch::Tensor ids, torch::Tensor gc,
                               torch::Tensor gs, torch::Tensor uc,
                               torch::Tensor us, int64_t fmt, torch::Tensor tab,
                               int64_t kw) {
  moe::EArgs a{};
  const int E = gc.size(0);
  a.gc = reinterpret_cast<const uint4*>(gc.data_ptr());
  a.gs = reinterpret_cast<const uint4*>(gs.data_ptr());
  a.uc = reinterpret_cast<const uint4*>(uc.data_ptr());
  a.us = reinterpret_cast<const uint4*>(us.data_ptr());
  a.cstride = gc.numel() / E / 16;
  a.sstride = gs.numel() / E / 16;
  a.x = reinterpret_cast<const half*>(x.data_ptr());
  a.ldx = x.stride(0);
  a.ids = ids.data_ptr<int>();
  a.routes = ids.numel();
  a.top_k = ids.size(1);
  const bool quantized = hout.scalar_type() == torch::kUInt8;
  a.n = quantized ? hout.size(2) * 32 : hout.size(-1);
  a.K = x.size(1);
  a.hout = quantized ? nullptr : reinterpret_cast<half*>(hout.data_ptr());
  a.qout =
      quantized ? reinterpret_cast<moe::Q8Block*>(hout.data_ptr()) : nullptr;
  a.tab =
      tab.numel() ? reinterpret_cast<const uint4*>(tab.data_ptr()) : nullptr;
  TORCH_CHECK(E <= 512, "expert ids must be < 512");
  TORCH_CHECK(a.routes <= moe::MAXR && x.size(0) <= moe::MAXC &&
              a.K % 128 == 0 && a.n % 32 == 0);
  TORCH_CHECK(gc.numel() % (E * 16) == 0 && gs.numel() % (E * 16) == 0,
              "planes must be uint4-aligned per expert");
  const dim3 grid(a.n / 32, a.routes);
  auto st = at::cuda::getCurrentCUDAStream();
  auto go = [&](auto kw) {
    constexpr int KWV = decltype(kw)::value;
    const size_t sm = KWV * 256 * 16 + dmvns::TAB_VECS_IQ2S * 16;
    if (fmt == dmvns::IQ3X)
      moe::moe_gu<dmvns::IQ3X, KWV><<<grid, 64 * KWV, sm, st>>>(a);
    else if (fmt == dmvns::IQ3S)
      moe::moe_gu<dmvns::IQ3S, KWV><<<grid, 64 * KWV, sm, st>>>(a);
    else if (fmt == dmvns::LUT4)
      moe::moe_gu<dmvns::LUT4, KWV><<<grid, 64 * KWV, sm, st>>>(a);
    else if (fmt == dmvns::IQ2S)
      moe::moe_gu<dmvns::IQ2S, KWV><<<grid, 64 * KWV, sm, st>>>(a);
    else
      TORCH_CHECK(false, "moe_gate_up: format");
  };
  if (kw == 2)
    go(std::integral_constant<int, 2>{});
  else if (kw == 8)
    go(std::integral_constant<int, 8>{});
  else if (kw == 5)
    go(std::integral_constant<int, 5>{});
  else if (kw == 10)
    go(std::integral_constant<int, 10>{});
  else
    go(std::integral_constant<int, 4>{});
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
