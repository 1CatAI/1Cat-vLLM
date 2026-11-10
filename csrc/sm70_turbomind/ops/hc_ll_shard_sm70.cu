// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Based on the supplied shard3 TP4 HC LL implementation.
// TP4-sharded HC v3 (LL protocol) for Qwen3.8 Flash-Next MTP verify on SM70.
// Each rank reads 1/4 of the HC weights. Cross-GPU data travels as 32-bit
// words (16-bit tag | FP16 value), so receivers poll the data itself and no
// system fence / global barrier is needed (same idea as NCCL LL and the
// existing 1Cat half+tag HC transport).  Only partners r^1 and r^2 are ever
// written (direct NVLink pairs on the 0-1/0-2/1-3/2-3 boards); the diagonal
// slice is forwarded by r^2.
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

namespace {
constexpr int KD = 10240, HD = 2560,
              LW = 336;  // LL lora row: 320 lora + 4 inj (+pad)

struct Peer {
  uint32_t* ll[4];  // per-rank LL buffers (down: [8][LW]; up: [8][HD])
  unsigned* seq;    // local per-op call counter
  int rank;
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
__device__ __forceinline__ uint4 ordered_weight_load(const uint4* p) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}
__device__ __forceinline__ float sigm(float x) {
  return 1.0f / (1.0f + __expf(-x));
}
__device__ __forceinline__ float div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0,%1,%2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ unsigned ld_vol(const unsigned* p) {
  unsigned v;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ void st_vol(unsigned* p, unsigned v) {
  asm volatile("st.volatile.global.u32 [%0], %1;" ::"l"(p), "r"(v));
}
__device__ __forceinline__ uint32_t pk(half v, uint32_t tag) {
  return (tag << 16) | __half_as_ushort(v);
}
__device__ __forceinline__ uint32_t poll1(const uint32_t* p, uint32_t tag) {
  uint32_t v;
  const long long t0 = clock64();
  do {
    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p));
    if (clock64() - t0 > 4000000000LL) __trap();
  } while ((v >> 16) != tag);
  return v;
}
__device__ __forceinline__ uint4 poll4(const uint32_t* p, uint32_t tag) {
  uint4 v;
  const long long t0 = clock64();
  do {
    asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    if (clock64() - t0 > 4000000000LL) __trap();
  } while ((v.x >> 16) != tag || (v.y >> 16) != tag || (v.z >> 16) != tag ||
           (v.w >> 16) != tag);
  return v;
}
__device__ __forceinline__ uint32_t tag_of(unsigned seq) {
  return seq % 65535u + 1u;
}

// ---------------------------------------------------------------- down
// grid (3 tiles, S splits). Tile-last CTA reduces split-K, pushes LL words to
// self/r^1/r^2, then forwards r^1's tile to r^2.
template <int S, int WARPS, bool OPTIMIZED = false>
__global__ void __launch_bounds__(32 * WARPS)
    down3(const half* __restrict__ x, const half* __restrict__ wd,
          float* __restrict__ part, unsigned* __restrict__ cnt, Peer pr,
          int M) {
  constexpr int KC = KD / S, KW = KC / WARPS, G = KW / 16, NP = 96;
  static_assert(KD % S == 0 && KC % WARPS == 0 && KW % 16 == 0, "");
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int packet = blockIdx.x, token_group = packet / 3;
  const int tile = packet % 3, s = blockIdx.y;
  const int first_row = token_group * 8;
  const int total_m = M;
  const int groups_m = (M + 7) / 8;
  M = min(8, M - first_row);
  x += first_row * KD;
  part += first_row * 96;
  for (int i = 0; i < 4; ++i)
    if (pr.ll[i])
      pr.ll[i] += first_row * LW + ((ld_vol(pr.seq) + 1) & 1u) * 24 * LW;
  // Inactive rows must not retain matching tags from an earlier modulo
  // cycle. Only this generation's page is touched: a faster peer may
  // already be writing the next page. All ranks have the same total_m.
  if (packet == 0 && s == 0)
    for (int i = threadIdx.x; i < (20 - total_m) * LW; i += blockDim.x)
      pr.ll[pr.rank][total_m * LW + i] = 0;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int quad = (lane >> 2) & 3;
  const int col = quad * 8 + r;
  const int kbase = s * KC + warp * KW;
  const half* w = wd + static_cast<size_t>(tile) * KD * 32;
  uint4 wlo[G], whi[G];
  if constexpr (OPTIMIZED) {
#pragma unroll
    for (int g = 0; g < G; ++g) {
      const int kg = (kbase + g * 16) >> 4;
      wlo[g] = ordered_weight_load(
          reinterpret_cast<const uint4*>(w + (kg * 64 + col) * 8));
      whi[g] = ordered_weight_load(
          reinterpret_cast<const uint4*>(w + (kg * 64 + 32 + col) * 8));
    }
    __syncwarp();
  }
  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int g = 0; g < G; ++g) {
    const int k = kbase + g * 16, kg = k >> 4;
    const uint4 lo =
        OPTIMIZED
            ? wlo[g]
            : __ldg(reinterpret_cast<const uint4*>(w + (kg * 64 + col) * 8));
    const uint4 hi = OPTIMIZED ? whi[g]
                               : __ldg(reinterpret_cast<const uint4*>(
                                     w + (kg * 64 + 32 + col) * 8));
    uint4 a = make_uint4(0, 0, 0, 0), b = a;
    if (r < M) {
      a = *reinterpret_cast<const uint4*>(x + r * KD + k);
      b = *reinterpret_cast<const uint4*>(x + r * KD + k + 8);
    }
    mma(acc, a.x, a.y, lo.x, lo.y);
    mma(acc, a.z, a.w, lo.z, lo.w);
    mma(acc, b.x, b.y, hi.x, hi.y);
    mma(acc, b.z, b.w, hi.z, hi.w);
  }
  __shared__ float red[WARPS][8][32];
  __shared__ bool last;
  if constexpr (WARPS > 1) {
#pragma unroll
    for (int i = 0; i < 8; ++i) red[warp][i][lane] = acc[i];
    __syncthreads();
    if (warp == 0) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        float v = red[0][i][lane];
#pragma unroll
        for (int q = 1; q < WARPS; ++q) v += red[q][i][lane];
        acc[i] = v;
      }
    }
  }
  if (warp == 0) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
      const int n = tile * 32 + quad * 8 +
                    ((i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2));
      if (row < M) part[(s * 24 + row) * NP + n] = acc[i];
    }
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    last = atomicAdd(cnt + packet, 1u) == S - 1;
  }
  __syncthreads();
  if (!last) return;
  __threadfence();
  const int rk = pr.rank, p1 = rk ^ 1, p2 = rk ^ 2;
  const uint32_t tag = tag_of(ld_vol(pr.seq) + 1);
  auto gcol = [](int q, int n) { return n < 80 ? q * 80 + n : 320 + n - 80; };
  auto valid = [](int q, int n) { return n < 80 || (q == 3 && n < 84); };
  for (int idx = threadIdx.x; idx < M * 32; idx += blockDim.x) {
    const int row = idx >> 5, n = tile * 32 + (idx & 31);
    if (!valid(rk, n)) continue;
    float tot = 0.f;
#pragma unroll
    for (int sp = 0; sp < S; ++sp)
      tot += __ldcg(part + (sp * 24 + row) * NP + n);
    half v = __float2half_rn(tot);
    if (n < 80) {
      const float xx = div_full(__half2float(v), 4.0f);
      v = __float2half_rn(xx * sigm(xx));
    }
    const uint32_t word = pk(v, tag);
    const int o = row * LW + gcol(rk, n);
    pr.ll[rk][o] = word;
    pr.ll[p1][o] = word;
    pr.ll[p2][o] = word;
  }
  // Forward partner r^1's tile to r^2 (it is r^2's diagonal).
  for (int idx = threadIdx.x; idx < M * 32; idx += blockDim.x) {
    const int row = idx >> 5, n = tile * 32 + (idx & 31);
    if (!valid(p1, n)) continue;
    const int o = row * LW + gcol(p1, n);
    pr.ll[p2][o] = poll1(pr.ll[rk] + o, tag);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    cnt[packet] = 0;
    if (atomicAdd(cnt + groups_m * 3, 1u) == gridDim.x - 1) {
      cnt[groups_m * 3] = 0;
      st_vol(pr.seq, ld_vol(pr.seq) + 1);
    }
  }
}

// ---------------------------------------------------------------- up
// 80 CTAs; CTA t owns hidden cols [640r + 8t, +8) for the 4 branches.
template <int WARPS, bool OPTIMIZED = false>
__global__ void __launch_bounds__(32 * WARPS)
    up3(const uint32_t* __restrict__ ll_lora, const half* __restrict__ wu,
        const half* __restrict__ x, unsigned* __restrict__ cnt, Peer pr,
        const unsigned* __restrict__ down_seq, half* __restrict__ out,
        half* __restrict__ lora_out, half* __restrict__ inj_out, int M) {
  constexpr int GW = 20 / WARPS, SW = 640;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int tile = blockIdx.x % 80, token_group = blockIdx.x / 80;
  const int first_row = token_group * 8;
  const int total_m = M;
  M = min(8, M - first_row);
  x += first_row * KD;
  ll_lora += first_row * LW + (ld_vol(down_seq) & 1u) * 24 * LW;
  out += first_row * HD;
  lora_out += first_row * 320;
  inj_out += first_row * 4;
  for (int i = 0; i < 4; ++i)
    if (pr.ll[i])
      pr.ll[i] += first_row * HD + ((ld_vol(pr.seq) + 1) & 1u) * 24 * HD;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int branch = (lane >> 2) & 3;
  const int col = branch * 8 + r;
  const int rk = pr.rank, p1 = rk ^ 1, p2 = rk ^ 2, p3 = rk ^ 3;
  if (token_group == 0)
    for (int i = threadIdx.x; i < (20 - total_m) * 32; i += blockDim.x) {
      const int row = total_m + i / 32, c = i % 32, src = c / 8;
      // The local slice is written directly to out and is never polled.
      if (src != rk) pr.ll[rk][row * HD + src * SW + tile * 8 + c % 8] = 0;
    }
  const half* w = wu + static_cast<size_t>(tile) * 320 * 32;
  // Issue this CTA's weight loads first; they do not depend on the exchange.
  uint4 wlo[GW], whi[GW];
#pragma unroll
  for (int gg = 0; gg < GW; ++gg) {
    const int g = warp * GW + gg;
    wlo[gg] = __ldg(reinterpret_cast<const uint4*>(w + (g * 64 + col) * 8));
    whi[gg] =
        __ldg(reinterpret_cast<const uint4*>(w + (g * 64 + 32 + col) * 8));
  }
  // Prologue: poll the full LL lora (written by down3 on all ranks).
  __shared__ __align__(16) half ls[8][320 + (OPTIMIZED ? 8 : 0)];
  const uint32_t dtag =
      tag_of(ld_vol(down_seq));  // down3 already advanced its seq
  for (int idx = threadIdx.x; idx < M * 80; idx += blockDim.x) {
    const int row = idx / 80, c4 = (idx % 80) * 4;
    const uint4 v = poll4(ll_lora + row * LW + c4, dtag);
    ls[row][c4 + 0] = __ushort_as_half(v.x & 0xffff);
    ls[row][c4 + 1] = __ushort_as_half(v.y & 0xffff);
    ls[row][c4 + 2] = __ushort_as_half(v.z & 0xffff);
    ls[row][c4 + 3] = __ushort_as_half(v.w & 0xffff);
  }
  if (tile == 0 && threadIdx.x < M * 4) {
    const int row = threadIdx.x >> 2, c = threadIdx.x & 3;
    inj_out[row * 4 + c] =
        __ushort_as_half(poll1(ll_lora + row * LW + 320 + c, dtag) & 0xffff);
  }
  __syncthreads();
  if (tile == 0)
    for (int idx = threadIdx.x; idx < M * 320; idx += blockDim.x)
      lora_out[idx] = ls[idx / 320][idx % 320];
  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int gg = 0; gg < GW; ++gg) {
    const int g = warp * GW + gg;
    uint4 a = make_uint4(0, 0, 0, 0), b = a;
    if (r < M) {
      a = *reinterpret_cast<const uint4*>(&ls[r][g * 16]);
      b = *reinterpret_cast<const uint4*>(&ls[r][g * 16 + 8]);
    }
    mma(acc, a.x, a.y, wlo[gg].x, wlo[gg].y);
    mma(acc, a.z, a.w, wlo[gg].z, wlo[gg].w);
    mma(acc, b.x, b.y, whi[gg].x, whi[gg].y);
    mma(acc, b.z, b.w, whi[gg].z, whi[gg].w);
  }
  __shared__ __align__(16) half tout[8][8];
  if constexpr (WARPS > 1) {
    __shared__ float red[WARPS][8][32];
#pragma unroll
    for (int i = 0; i < 8; ++i) red[warp][i][lane] = acc[i];
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float v = red[0][i][lane];
#pragma unroll
      for (int q = 1; q < WARPS; ++q) v += red[q][i][lane];
      acc[i] = v;
    }
  }
  if (warp == 0) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
      const int hh = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
      const int hl = tile * 8 + hh;
      const float sg = sigm(__half2float(__float2half_rn(acc[i])));
      float v = 0.f;
      if (row < M) v = __half2float(x[row * KD + branch * HD + rk * SW + hl]);
      float mixed = 0.f;
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        const int src = (lane & ~12) | (b << 2);
        mixed = fmaf(__shfl_sync(0xffffffff, sg, src),
                     __shfl_sync(0xffffffff, v, src), mixed);
      }
      if (branch == 0) tout[row][hh] = __float2half_rn(div_full(mixed, 4.0f));
    }
  }
  __syncthreads();
  const uint32_t tag = tag_of(ld_vol(pr.seq) + 1);
  // Threads [0, 2M): push own tile (two uint4 per row) to r^1 and r^2.
  if (threadIdx.x < 2 * M) {
    const int row = threadIdx.x >> 1, h4 = (threadIdx.x & 1) * 4;
    uint4 wv;
    wv.x = pk(tout[row][h4 + 0], tag);
    wv.y = pk(tout[row][h4 + 1], tag);
    wv.z = pk(tout[row][h4 + 2], tag);
    wv.w = pk(tout[row][h4 + 3], tag);
    const int o = row * HD + rk * SW + tile * 8 + h4;
    *reinterpret_cast<uint4*>(pr.ll[p1] + o) = wv;
    *reinterpret_cast<uint4*>(pr.ll[p2] + o) = wv;
    *reinterpret_cast<uint2*>(out + o) =
        *reinterpret_cast<const uint2*>(&tout[row][h4]);
  }
  // Forward r^1's tile to r^2, then collect r^1, r^2, r^3 tiles into `out`.
  if (threadIdx.x >= 32 && threadIdx.x < 32 + 2 * M) {
    const int t = threadIdx.x - 32, row = t >> 1, h4 = (t & 1) * 4;
    const int o = row * HD + p1 * SW + tile * 8 + h4;
    const uint4 v = poll4(pr.ll[rk] + o, tag);
    *reinterpret_cast<uint4*>(pr.ll[p2] + o) = v;
    half2 h01 = __halves2half2(__ushort_as_half(v.x & 0xffff),
                               __ushort_as_half(v.y & 0xffff));
    half2 h23 = __halves2half2(__ushort_as_half(v.z & 0xffff),
                               __ushort_as_half(v.w & 0xffff));
    *reinterpret_cast<half2*>(out + o) = h01;
    *reinterpret_cast<half2*>(out + o + 2) = h23;
  }
  if (threadIdx.x >= 64 && threadIdx.x < 64 + 4 * M) {
    const int t = threadIdx.x - 64, src = (t & 2) ? p3 : p2;
    const int row = t >> 2, h4 = (t & 1) * 4;
    const int o = row * HD + src * SW + tile * 8 + h4;
    const uint4 v = poll4(pr.ll[rk] + o, tag);
    half2 h01 = __halves2half2(__ushort_as_half(v.x & 0xffff),
                               __ushort_as_half(v.y & 0xffff));
    half2 h23 = __halves2half2(__ushort_as_half(v.z & 0xffff),
                               __ushort_as_half(v.w & 0xffff));
    *reinterpret_cast<half2*>(out + o) = h01;
    *reinterpret_cast<half2*>(out + o + 2) = h23;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    if (atomicAdd(cnt, 1u) == gridDim.x - 1) {
      *cnt = 0;
      st_vol(pr.seq, ld_vol(pr.seq) + 1);
    }
  }
}

Peer make_peer(const std::vector<int64_t>& ll, torch::Tensor seq,
               int64_t rank) {
  Peer p;
  for (int i = 0; i < 4; ++i) p.ll[i] = reinterpret_cast<uint32_t*>(ll[i]);
  p.seq = reinterpret_cast<unsigned*>(seq.data_ptr<int>());
  p.rank = static_cast<int>(rank);
  return p;
}
}  // namespace

void sm70_hc_ll_down_out(torch::Tensor x, torch::Tensor wd, torch::Tensor part,
                         torch::Tensor cnt, std::vector<int64_t> ll,
                         torch::Tensor seq, int64_t rank, int64_t variant,
                         bool optimized_loads) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.dim() == 2 &&
                  x.size(0) >= 1 && x.size(0) <= 20 && x.size(1) == 10240 &&
                  x.is_contiguous(),
              "HC LL requires CUDA FP16 [1..20,10240]");
  const c10::cuda::CUDAGuard guard(x.device());
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major == 7 && properties->minor == 0,
              "SM70 required");
  TORCH_CHECK(ll.size() == 4 && rank >= 0 && rank < 4 && ll[rank] &&
                  ll[rank ^ 1] && ll[rank ^ 2],
              "HC LL requires local and two direct NVLink peer pointers");
  auto local = [&](const torch::Tensor& t, at::ScalarType dtype) {
    TORCH_CHECK(t.device() == x.device() && t.scalar_type() == dtype &&
                    t.is_contiguous(),
                "HC LL storage dtype/device mismatch");
  };
  local(cnt, at::kInt);
  local(seq, at::kInt);
  TORCH_CHECK(seq.numel() == 1, "HC LL sequence must have one element");
  const int M = x.size(0);
  local(wd, at::kHalf);
  local(part, at::kFloat);
  TORCH_CHECK(wd.sizes() == at::IntArrayRef({3, 640, 2, 32, 8}) &&
                  part.numel() >= 40 * 24 * 96 && cnt.numel() == 10,
              "HC LL down workspace or weight shape mismatch");
  auto st = at::cuda::getCurrentCUDAStream().stream();
  Peer p = make_peer(ll, seq, rank);
  auto c = reinterpret_cast<unsigned*>(cnt.data_ptr<int>());
#define D(S, W, O)                                                             \
  down3<S, W, O><<<dim3(3 * ((M + 7) / 8), S), 32 * W, 0, st>>>(               \
      reinterpret_cast<const half*>(x.data_ptr()),                             \
      reinterpret_cast<const half*>(wd.data_ptr()), part.data_ptr<float>(), c, \
      p, M)
  switch (variant) {
    case 0:
      D(20, 4, false);
      break;
    case 1:
      if (optimized_loads) {
        D(20, 8, true);
      } else {
        D(20, 8, false);
      }
      break;
    case 2:
      D(40, 4, false);
      break;
    case 3:
      D(40, 8, false);
      break;
    case 4:
      D(10, 8, false);
      break;
    case 5:
      D(20, 16, false);
      break;
    case 6:
      D(10, 16, false);
      break;
    default:
      TORCH_CHECK(false);
  }
#undef D
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sm70_hc_ll_up_out(int64_t ll_lora, torch::Tensor wu, torch::Tensor x,
                       torch::Tensor cnt, std::vector<int64_t> ll,
                       torch::Tensor seq, torch::Tensor down_seq, int64_t rank,
                       torch::Tensor out, torch::Tensor lora_out,
                       torch::Tensor inj_out, int64_t warps,
                       bool optimized_loads) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.dim() == 2 &&
                  x.size(0) >= 1 && x.size(0) <= 20 && x.size(1) == 10240 &&
                  x.is_contiguous(),
              "HC LL requires CUDA FP16 [1..20,10240]");
  const c10::cuda::CUDAGuard guard(x.device());
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major == 7 && properties->minor == 0,
              "SM70 required");
  TORCH_CHECK(ll.size() == 4 && rank >= 0 && rank < 4 && ll[rank] &&
                  ll[rank ^ 1] && ll[rank ^ 2],
              "HC LL requires local and two direct NVLink peer pointers");
  auto local = [&](const torch::Tensor& t, at::ScalarType dtype) {
    TORCH_CHECK(t.device() == x.device() && t.scalar_type() == dtype &&
                    t.is_contiguous(),
                "HC LL storage dtype/device mismatch");
  };
  local(cnt, at::kInt);
  local(seq, at::kInt);
  TORCH_CHECK(seq.numel() == 1, "HC LL sequence must have one element");
  const int M = x.size(0);
  local(wu, at::kHalf);
  local(down_seq, at::kInt);
  local(out, at::kHalf);
  local(lora_out, at::kHalf);
  local(inj_out, at::kHalf);
  TORCH_CHECK(ll_lora && wu.sizes() == at::IntArrayRef({80, 20, 2, 4, 8, 8}) &&
                  cnt.numel() == 1 && down_seq.numel() == 1 &&
                  out.sizes() == at::IntArrayRef({M, 2560}) &&
                  lora_out.sizes() == at::IntArrayRef({M, 320}) &&
                  inj_out.sizes() == at::IntArrayRef({M, 4}),
              "HC LL up workspace or shape mismatch");
  auto st = at::cuda::getCurrentCUDAStream().stream();
  Peer p = make_peer(ll, seq, rank);
  auto c = reinterpret_cast<unsigned*>(cnt.data_ptr<int>());
#define U(W, O)                                                    \
  up3<W, O><<<80 * ((M + 7) / 8), 32 * W, 0, st>>>(                \
      reinterpret_cast<const uint32_t*>(ll_lora),                  \
      reinterpret_cast<const half*>(wu.data_ptr()),                \
      reinterpret_cast<const half*>(x.data_ptr()), c, p,           \
      reinterpret_cast<const unsigned*>(down_seq.data_ptr<int>()), \
      reinterpret_cast<half*>(out.data_ptr()),                     \
      reinterpret_cast<half*>(lora_out.data_ptr()),                \
      reinterpret_cast<half*>(inj_out.data_ptr()), M)
  switch (warps) {
    case 4:
      U(4, false);
      break;
    case 5:
      if (optimized_loads) {
        U(5, true);
      } else {
        U(5, false);
      }
      break;
    default:
      TORCH_CHECK(false);
  }
#undef U
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
