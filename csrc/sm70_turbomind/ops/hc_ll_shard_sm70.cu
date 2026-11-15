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
constexpr int kMaxRows = 64, kPageRows = 64;

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
// grid (3 tiles, S splits). Registers hold wd once across all 8-row groups;
// tile-last CTA reduces split-K across all M rows and pushes LL words.
template <int S, int WARPS, bool OPTIMIZED = false>
__global__ void __launch_bounds__(32 * WARPS)
    down3(const half* __restrict__ x, const half* __restrict__ wd,
          float* __restrict__ part, unsigned* __restrict__ cnt, Peer pr,
          int total_m) {
  constexpr int KC = KD / S, KW = KC / WARPS, G = KW / 16, NP = 96;
  static_assert(KD % S == 0 && KC % WARPS == 0 && KW % 16 == 0, "");
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int tile = blockIdx.x, s = blockIdx.y;
  const int groups_m = (total_m + 7) / 8;
  const unsigned page_off = ((ld_vol(pr.seq) + 1) & 1u) * kPageRows * LW;
  for (int i = 0; i < 4; ++i)
    if (pr.ll[i]) pr.ll[i] += page_off;
  // Inactive rows must not retain matching tags from an earlier modulo
  // cycle. Only this generation's page is touched: a faster peer may
  // already be writing the next page. All ranks have the same total_m.
  if (s == 0) {
    for (int i = tile * blockDim.x + threadIdx.x; i < (kMaxRows - total_m) * LW;
         i += 3 * blockDim.x) {
      pr.ll[pr.rank][total_m * LW + i] = 0;
    }
  }
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
  } else {
#pragma unroll
    for (int g = 0; g < G; ++g) {
      const int kg = (kbase + g * 16) >> 4;
      wlo[g] = __ldg(reinterpret_cast<const uint4*>(w + (kg * 64 + col) * 8));
      whi[g] =
          __ldg(reinterpret_cast<const uint4*>(w + (kg * 64 + 32 + col) * 8));
    }
  }
  __shared__ float red[WARPS][8][32];
  __shared__ bool last;
  for (int grp = 0; grp < groups_m; ++grp) {
    const int first_row = grp * 8;
    const int gm = min(8, total_m - first_row);
    const half* xg = x + first_row * KD;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int g = 0; g < G; ++g) {
      const int k = kbase + g * 16;
      uint4 a = make_uint4(0, 0, 0, 0), b = a;
      if (r < gm) {
        a = *reinterpret_cast<const uint4*>(xg + r * KD + k);
        b = *reinterpret_cast<const uint4*>(xg + r * KD + k + 8);
      }
      mma(acc, a.x, a.y, wlo[g].x, wlo[g].y);
      mma(acc, a.z, a.w, wlo[g].z, wlo[g].w);
      mma(acc, b.x, b.y, whi[g].x, whi[g].y);
      mma(acc, b.z, b.w, whi[g].z, whi[g].w);
    }
    if constexpr (WARPS > 1) {
      if (grp > 0) __syncthreads();
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
        if (row < gm) part[(s * kPageRows + first_row + row) * NP + n] = acc[i];
      }
    }
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    last = atomicAdd(cnt + tile, 1u) == S - 1;
  }
  __syncthreads();
  if (!last) return;
  __threadfence();
  const int rk = pr.rank, p1 = rk ^ 1, p2 = rk ^ 2, p3 = rk ^ 3;
  const bool full_mesh = (pr.ll[p3] != nullptr);
  const uint32_t tag = tag_of(ld_vol(pr.seq) + 1);
  auto gcol = [](int q, int n) { return n < 80 ? q * 80 + n : 320 + n - 80; };
  auto valid = [](int q, int n) { return n < 80 || (q == 3 && n < 84); };
  for (int idx = threadIdx.x; idx < total_m * 32; idx += blockDim.x) {
    const int row = idx >> 5, n = tile * 32 + (idx & 31);
    if (!valid(rk, n)) continue;
    float tot = 0.f;
#pragma unroll
    for (int sp = 0; sp < S; ++sp)
      tot += __ldcg(part + (sp * kPageRows + row) * NP + n);
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
    if (full_mesh) pr.ll[p3][o] = word;
  }
  if (!full_mesh) {
    // Forward partner r^1's tile to r^2 (it is r^2's diagonal).
    for (int idx = threadIdx.x; idx < total_m * 32; idx += blockDim.x) {
      const int row = idx >> 5, n = tile * 32 + (idx & 31);
      if (!valid(p1, n)) continue;
      const int o = row * LW + gcol(p1, n);
      pr.ll[p2][o] = poll1(pr.ll[rk] + o, tag);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    cnt[tile] = 0;
    if (atomicAdd(cnt + 3, 1u) == 2u) {
      cnt[3] = 0;
      st_vol(pr.seq, ld_vol(pr.seq) + 1);
    }
  }
}

// ---------------------------------------------------------------- up
// 80 CTAs; CTA t owns hidden cols [640r + 8t, +8) for the 4 branches across all
// 8-row groups, keeping wu in registers.
template <int WARPS, bool OPTIMIZED = false>
__global__ void __launch_bounds__(32 * WARPS)
    up3(const uint32_t* __restrict__ ll_lora, const half* __restrict__ wu,
        const half* __restrict__ x, unsigned* __restrict__ cnt, Peer pr,
        const unsigned* __restrict__ down_seq, half* __restrict__ out,
        half* __restrict__ lora_out, half* __restrict__ inj_out, int total_m) {
  constexpr int GW = 20 / WARPS, SW = 640;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  const int groups_m = (total_m + 7) / 8;
  const unsigned dseq = ld_vol(down_seq);
  const unsigned useq_next = ld_vol(pr.seq) + 1;
  ll_lora += (dseq & 1u) * kPageRows * LW;
  for (int i = 0; i < 4; ++i)
    if (pr.ll[i]) pr.ll[i] += (useq_next & 1u) * kPageRows * HD;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int branch = (lane >> 2) & 3;
  const int col = branch * 8 + r;
  const int rk = pr.rank, p1 = rk ^ 1, p2 = rk ^ 2, p3 = rk ^ 3;
  const bool full_mesh = (pr.ll[p3] != nullptr);
  for (int i = threadIdx.x; i < (kMaxRows - total_m) * 32; i += blockDim.x) {
    const int row = total_m + i / 32, c = i % 32, src = c / 8;
    // The local slice is written directly to out and is never polled.
    if (src != rk) pr.ll[rk][row * HD + src * SW + tile * 8 + c % 8] = 0;
  }
  const half* w = wu + static_cast<size_t>(tile) * 320 * 32;
  // Issue this CTA's weight loads once; they stay in registers across groups.
  uint4 wlo[GW], whi[GW];
#pragma unroll
  for (int gg = 0; gg < GW; ++gg) {
    const int g = warp * GW + gg;
    wlo[gg] = __ldg(reinterpret_cast<const uint4*>(w + (g * 64 + col) * 8));
    whi[gg] =
        __ldg(reinterpret_cast<const uint4*>(w + (g * 64 + 32 + col) * 8));
  }
  __shared__ __align__(16) half ls[8][320 + (OPTIMIZED ? 8 : 0)];
  __shared__ __align__(16) half tout[8][8];
  __shared__ float red[WARPS > 1 ? WARPS : 1][8][32];
  const uint32_t dtag = tag_of(dseq);
  const uint32_t tag = tag_of(useq_next);
  if (tile == 0) {
    for (int idx = threadIdx.x; idx < total_m * 4; idx += blockDim.x) {
      const int row = idx >> 2, c = idx & 3;
      inj_out[row * 4 + c] =
          __ushort_as_half(poll1(ll_lora + row * LW + 320 + c, dtag) & 0xffff);
    }
  }
  for (int grp = 0; grp < groups_m; ++grp) {
    const int first_row = grp * 8;
    const int gm = min(8, total_m - first_row);
    if (grp > 0) __syncthreads();
    for (int idx = threadIdx.x; idx < gm * 80; idx += blockDim.x) {
      const int row = idx / 80, c4 = (idx % 80) * 4;
      const uint4 v = poll4(ll_lora + (first_row + row) * LW + c4, dtag);
      ls[row][c4 + 0] = __ushort_as_half(v.x & 0xffff);
      ls[row][c4 + 1] = __ushort_as_half(v.y & 0xffff);
      ls[row][c4 + 2] = __ushort_as_half(v.z & 0xffff);
      ls[row][c4 + 3] = __ushort_as_half(v.w & 0xffff);
    }
    __syncthreads();
    if (tile == 0 && lora_out != nullptr) {
      for (int idx = threadIdx.x; idx < gm * 320; idx += blockDim.x)
        lora_out[first_row * 320 + idx] = ls[idx / 320][idx % 320];
    }
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int gg = 0; gg < GW; ++gg) {
      const int g = warp * GW + gg;
      uint4 a = make_uint4(0, 0, 0, 0), b = a;
      if (r < gm) {
        a = *reinterpret_cast<const uint4*>(&ls[r][g * 16]);
        b = *reinterpret_cast<const uint4*>(&ls[r][g * 16 + 8]);
      }
      mma(acc, a.x, a.y, wlo[gg].x, wlo[gg].y);
      mma(acc, a.z, a.w, wlo[gg].z, wlo[gg].w);
      mma(acc, b.x, b.y, whi[gg].x, whi[gg].y);
      mma(acc, b.z, b.w, whi[gg].z, whi[gg].w);
    }
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
      const half* xg = x + first_row * KD;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
        const int hh = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
        const int hl = tile * 8 + hh;
        const float sg = sigm(__half2float(__float2half_rn(acc[i])));
        float v = 0.f;
        if (row < gm)
          v = __half2float(xg[row * KD + branch * HD + rk * SW + hl]);
        float mixed = 0.f;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          const int src = (lane & ~12) | (b << 2);
          mixed = fmaf(__shfl_sync(0xffffffff, sg, src),
                       __shfl_sync(0xffffffff, v, src), mixed);
        }
        if (branch == 0 && row < gm)
          tout[row][hh] = __float2half_rn(div_full(mixed, 4.0f));
      }
    }
    __syncthreads();
    if (threadIdx.x < 2 * gm) {
      const int row = threadIdx.x >> 1, h4 = (threadIdx.x & 1) * 4;
      uint4 wv;
      wv.x = pk(tout[row][h4 + 0], tag);
      wv.y = pk(tout[row][h4 + 1], tag);
      wv.z = pk(tout[row][h4 + 2], tag);
      wv.w = pk(tout[row][h4 + 3], tag);
      const int o = (first_row + row) * HD + rk * SW + tile * 8 + h4;
      *reinterpret_cast<uint4*>(pr.ll[p1] + o) = wv;
      *reinterpret_cast<uint4*>(pr.ll[p2] + o) = wv;
      if (full_mesh) *reinterpret_cast<uint4*>(pr.ll[p3] + o) = wv;
      *reinterpret_cast<uint2*>(out + o) =
          *reinterpret_cast<const uint2*>(&tout[row][h4]);
    }
  }
  if (full_mesh) {
    for (int idx = threadIdx.x; idx < total_m * 6; idx += blockDim.x) {
      const int peer_sel = idx % 3;
      const int src = (peer_sel == 0) ? p1 : ((peer_sel == 1) ? p2 : p3);
      const int rh = idx / 3;
      const int row = rh >> 1, h4 = (rh & 1) * 4;
      const int o = row * HD + src * SW + tile * 8 + h4;
      const uint4 v = poll4(pr.ll[rk] + o, tag);
      half2 h01 = __halves2half2(__ushort_as_half(v.x & 0xffff),
                                 __ushort_as_half(v.y & 0xffff));
      half2 h23 = __halves2half2(__ushort_as_half(v.z & 0xffff),
                                 __ushort_as_half(v.w & 0xffff));
      *reinterpret_cast<half2*>(out + o) = h01;
      *reinterpret_cast<half2*>(out + o + 2) = h23;
    }
  } else {
    // Forward r^1's tile to r^2, then collect r^1, r^2, r^3 tiles into `out`.
    for (int t = threadIdx.x; t < 2 * total_m; t += blockDim.x) {
      const int row = t >> 1, h4 = (t & 1) * 4;
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
    for (int t = threadIdx.x; t < 4 * total_m; t += blockDim.x) {
      const int src = (t & 2) ? p3 : p2;
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

__device__ __forceinline__ void st_vol4(uint32_t* p, uint4 v) {
  asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p),
               "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
}

struct PhysOrder4 {
  int v[4];
};

// 80 CTAs x 128 threads: fuse el [M, 640] + hl [M, 4, 640] + TP4 LL AllGather
// -> out [M, 10240] with physical-rank column placement b*2560 + phys[l]*640 +
// c.
__global__ void __launch_bounds__(128) fc_add_all_gather_kernel(
    half* __restrict__ out, const half* __restrict__ el,
    const half* __restrict__ hl, unsigned* __restrict__ cnt,
    unsigned* __restrict__ dirty_m, Peer pr, PhysOrder4 phys, int total_m) {
  constexpr int SW = 640;
  const int tid = threadIdx.x;
  const int tile = blockIdx.x;  // 0..79, owns cols [tile*8, tile*8 + 8)
  const unsigned next_seq = ld_vol(pr.seq) + 1u;
  const unsigned page = next_seq & 1u;
  const size_t page_off = static_cast<size_t>(page) * kPageRows * KD;
  for (int i = 0; i < 4; ++i) {
    if (pr.ll[i]) pr.ll[i] += page_off;
  }
  const int rk = pr.rank, p1 = rk ^ 1, p2 = rk ^ 2, p3 = rk ^ 3;
  const bool full_mesh = (pr.ll[p3] != nullptr);
  const int prev_m = static_cast<int>(ld_vol(dirty_m + page));
  if (total_m < prev_m) {
    const uint4 zero4 = make_uint4(0u, 0u, 0u, 0u);
    const int clean_rows = prev_m - total_m;
    for (int idx = tid; idx < clean_rows * 24; idx += 128) {
      const int peer_sel = idx % 3;
      const int src = (peer_sel == 0) ? p1 : ((peer_sel == 1) ? p2 : p3);
      const int rem = idx / 3;
      const int row = total_m + (rem >> 3);
      const int b = (rem >> 1) & 3;
      const int h4 = (rem & 1) * 4;
      const size_t o = static_cast<size_t>(row) * KD + b * HD +
                       phys.v[src] * SW + tile * 8 + h4;
      st_vol4(pr.ll[rk] + o, zero4);
    }
  }

  const uint32_t tag = tag_of(next_seq);
  const int phys_self = phys.v[rk];
  for (int idx = tid; idx < total_m * 8; idx += 128) {
    const int row = idx >> 3;
    const int b = (idx >> 1) & 3;
    const int h4 = (idx & 1) * 4;
    const int c = tile * 8 + h4;
    const uint2 ev =
        *reinterpret_cast<const uint2*>(el + static_cast<size_t>(row) * SW + c);
    const uint2 hv = *reinterpret_cast<const uint2*>(
        hl + static_cast<size_t>(row * 4 + b) * SW + c);
    const half2 s01 = __hadd2(*reinterpret_cast<const half2*>(&ev.x),
                              *reinterpret_cast<const half2*>(&hv.x));
    const half2 s23 = __hadd2(*reinterpret_cast<const half2*>(&ev.y),
                              *reinterpret_cast<const half2*>(&hv.y));
    uint4 wv;
    wv.x = pk(__low2half(s01), tag);
    wv.y = pk(__high2half(s01), tag);
    wv.z = pk(__low2half(s23), tag);
    wv.w = pk(__high2half(s23), tag);
    const size_t o =
        static_cast<size_t>(row) * KD + b * HD + phys_self * SW + c;
    st_vol4(pr.ll[p1] + o, wv);
    st_vol4(pr.ll[p2] + o, wv);
    if (full_mesh) st_vol4(pr.ll[p3] + o, wv);
    *reinterpret_cast<uint2*>(out + o) =
        make_uint2(*reinterpret_cast<const uint32_t*>(&s01),
                   *reinterpret_cast<const uint32_t*>(&s23));
  }

  if (full_mesh) {
    for (int idx = tid; idx < total_m * 24; idx += 128) {
      const int peer_sel = idx % 3;
      const int src = (peer_sel == 0) ? p1 : ((peer_sel == 1) ? p2 : p3);
      const int rem = idx / 3;
      const int row = rem >> 3;
      const int b = (rem >> 1) & 3;
      const int h4 = (rem & 1) * 4;
      const size_t o = static_cast<size_t>(row) * KD + b * HD +
                       phys.v[src] * SW + tile * 8 + h4;
      const uint4 v = poll4(pr.ll[rk] + o, tag);
      const half2 h01 = __halves2half2(__ushort_as_half(v.x & 0xffffu),
                                       __ushort_as_half(v.y & 0xffffu));
      const half2 h23 = __halves2half2(__ushort_as_half(v.z & 0xffffu),
                                       __ushort_as_half(v.w & 0xffffu));
      *reinterpret_cast<uint2*>(out + o) =
          make_uint2(*reinterpret_cast<const uint32_t*>(&h01),
                     *reinterpret_cast<const uint32_t*>(&h23));
    }
  } else {
    for (int idx = tid; idx < total_m * 8; idx += 128) {
      const int row = idx >> 3;
      const int b = (idx >> 1) & 3;
      const int h4 = (idx & 1) * 4;
      const size_t o = static_cast<size_t>(row) * KD + b * HD +
                       phys.v[p1] * SW + tile * 8 + h4;
      const uint4 v = poll4(pr.ll[rk] + o, tag);
      st_vol4(pr.ll[p2] + o, v);
      const half2 h01 = __halves2half2(__ushort_as_half(v.x & 0xffffu),
                                       __ushort_as_half(v.y & 0xffffu));
      const half2 h23 = __halves2half2(__ushort_as_half(v.z & 0xffffu),
                                       __ushort_as_half(v.w & 0xffffu));
      *reinterpret_cast<uint2*>(out + o) =
          make_uint2(*reinterpret_cast<const uint32_t*>(&h01),
                     *reinterpret_cast<const uint32_t*>(&h23));
    }
    for (int idx = tid; idx < total_m * 16; idx += 128) {
      const int src = (idx & 1) ? p3 : p2;
      const int rem = idx >> 1;
      const int row = rem >> 3;
      const int b = (rem >> 1) & 3;
      const int h4 = (rem & 1) * 4;
      const size_t o = static_cast<size_t>(row) * KD + b * HD +
                       phys.v[src] * SW + tile * 8 + h4;
      const uint4 v = poll4(pr.ll[rk] + o, tag);
      const half2 h01 = __halves2half2(__ushort_as_half(v.x & 0xffffu),
                                       __ushort_as_half(v.y & 0xffffu));
      const half2 h23 = __halves2half2(__ushort_as_half(v.z & 0xffffu),
                                       __ushort_as_half(v.w & 0xffffu));
      *reinterpret_cast<uint2*>(out + o) =
          make_uint2(*reinterpret_cast<const uint32_t*>(&h01),
                     *reinterpret_cast<const uint32_t*>(&h23));
    }
  }

  __syncthreads();
  if (tid == 0) {
    __threadfence();
    if (atomicAdd(cnt, 1u) == 79u) {
      *cnt = 0u;
      st_vol(dirty_m + page, static_cast<unsigned>(total_m));
      st_vol(pr.seq, next_seq);
    }
  }
}

struct ArPeer4 {
  uint32_t* ar[4];
};

// 1 CTA per row, 640 threads (KPT = 4): full-mesh TP4 LL (16-bit tag | FP16)
// AllReduce + FP32 deterministic rank-order sum + hc_combine + optional
// GroupedGemmaRMSNorm in a single launch. Every row 0..total_m-1<=64 gets its
// own dedicated V100 SM simultaneously.
template <bool WITH_NORM>
__global__ void __launch_bounds__(640) allreduce_combine_norm_kernel(
    half* __restrict__ res_out, half* __restrict__ norm_out,
    const half* __restrict__ p0, const half* __restrict__ p1_in,
    const half* __restrict__ res, const half* __restrict__ inj,
    const half* __restrict__ nw, int nw_full, float eps,
    unsigned* __restrict__ cnt, unsigned* __restrict__ dirty_m,
    unsigned* __restrict__ seq, ArPeer4 peers, int rk, int total_m) {
  constexpr int THREADS = 640;
  constexpr int WARPS = THREADS / 32;
  constexpr int KPT = HD / THREADS;  // 4

  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int col0 = tid * KPT;
  const int peer1 = rk ^ 1, peer2 = rk ^ 2, peer3 = rk ^ 3;

  const unsigned next_ep = ld_vol(seq) + 1u;
  const unsigned page = next_ep & 1u;
  const uint32_t tag = tag_of(next_ep);
  const size_t page_words = static_cast<size_t>(page) * 4 * kPageRows * HD;
  uint32_t* ar[4];
#pragma unroll
  for (int q = 0; q < 4; ++q) ar[q] = peers.ar[q] + page_words;

  const int prev_m = static_cast<int>(ld_vol(dirty_m + page));
  if (total_m < prev_m) {
    const uint4 zero4 = make_uint4(0u, 0u, 0u, 0u);
    for (int clean_r = total_m + row; clean_r < prev_m; clean_r += total_m) {
      st_vol4(
          ar[rk] + static_cast<size_t>(peer1 * kPageRows + clean_r) * HD + col0,
          zero4);
      st_vol4(
          ar[rk] + static_cast<size_t>(peer2 * kPageRows + clean_r) * HD + col0,
          zero4);
      st_vol4(
          ar[rk] + static_cast<size_t>(peer3 * kPageRows + clean_r) * HD + col0,
          zero4);
    }
  }

  // 1. Load local partials and push 4 tagged FP16 words (1 uint4) per peer.
  const uint2 p0_vec = *reinterpret_cast<const uint2*>(
      p0 + static_cast<size_t>(row) * HD + col0);
  const half* p0_h = reinterpret_cast<const half*>(&p0_vec);
  half loc_h[4];
  float v[4];
  if (p1_in != nullptr) {
    const uint2 p1_vec = *reinterpret_cast<const uint2*>(
        p1_in + static_cast<size_t>(row) * HD + col0);
    const half* p1_h = reinterpret_cast<const half*>(&p1_vec);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const half s =
          __float2half_rn(__half2float(p0_h[j]) + __half2float(p1_h[j]));
      loc_h[j] = s;
      v[j] = __half2float(s);
    }
  } else {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      loc_h[j] = p0_h[j];
      v[j] = __half2float(p0_h[j]);
    }
  }

  uint4 wv;
  wv.x = pk(loc_h[0], tag);
  wv.y = pk(loc_h[1], tag);
  wv.z = pk(loc_h[2], tag);
  wv.w = pk(loc_h[3], tag);
  const size_t self_off = static_cast<size_t>(rk * kPageRows + row) * HD + col0;
  st_vol4(ar[peer1] + self_off, wv);
  st_vol4(ar[peer2] + self_off, wv);
  st_vol4(ar[peer3] + self_off, wv);

  // 2. Overlap local HBM reads of residual, norm weight, and injection while
  //    NVLink writes are in flight.
  uint2 res_vec[4];
  uint2 nw_vec[4];
  float g[4];
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    res_vec[s] = *reinterpret_cast<const uint2*>(
        res + static_cast<size_t>(row) * KD + s * HD + col0);
    if constexpr (WITH_NORM) {
      nw_vec[s] =
          *reinterpret_cast<const uint2*>(nw + (nw_full ? s * HD : 0) + col0);
    }
    g[s] = 2.0f * sigm(__half2float(inj[row * 4 + s]) * 0.25f);
  }
  asm volatile("" ::: "memory");

  // 3. Poll all 3 NVLink peers concurrently (1 uint4 each) and reduce in
  //    deterministic FP32 rank order (rank0 + rank1) + (rank2 + rank3).
  const uint32_t* p1_poll =
      ar[rk] + static_cast<size_t>(peer1 * kPageRows + row) * HD + col0;
  const uint32_t* p2_poll =
      ar[rk] + static_cast<size_t>(peer2 * kPageRows + row) * HD + col0;
  const uint32_t* p3_poll =
      ar[rk] + static_cast<size_t>(peer3 * kPageRows + row) * HD + col0;
  uint4 r1, r2, r3;
  const long long t0 = clock64();
  do {
    asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r1.x), "=r"(r1.y), "=r"(r1.z), "=r"(r1.w)
                 : "l"(p1_poll));
    asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r2.x), "=r"(r2.y), "=r"(r2.z), "=r"(r2.w)
                 : "l"(p2_poll));
    asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r3.x), "=r"(r3.y), "=r"(r3.z), "=r"(r3.w)
                 : "l"(p3_poll));
    if (clock64() - t0 > 4000000000LL) __trap();
  } while ((r1.x >> 16) != tag || (r1.y >> 16) != tag || (r1.z >> 16) != tag ||
           (r1.w >> 16) != tag || (r2.x >> 16) != tag || (r2.y >> 16) != tag ||
           (r2.z >> 16) != tag || (r2.w >> 16) != tag || (r3.x >> 16) != tag ||
           (r3.y >> 16) != tag || (r3.z >> 16) != tag || (r3.w >> 16) != tag);

  const uint32_t u1[4] = {r1.x, r1.y, r1.z, r1.w};
  const uint32_t u2[4] = {r2.x, r2.y, r2.z, r2.w};
  const uint32_t u3[4] = {r3.x, r3.y, r3.z, r3.w};
  float B[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float x1 = __half2float(__ushort_as_half(u1[j] & 0xffffu));
    const float x2 = __half2float(__ushort_as_half(u2[j] & 0xffffu));
    const float x3 = __half2float(__ushort_as_half(u3[j] & 0xffffu));
    const float near_s = (rk & 1) ? (x1 + v[j]) : (v[j] + x1);
    const float far_s = (rk & 1) ? (x3 + x2) : (x2 + x3);
    const float sum = (rk & 2) ? (far_s + near_s) : (near_s + far_s);
    B[j] = __half2float(__float2half_rn(sum));
  }

  // 4. Combine + optional GroupedGemmaRMSNorm.
  __shared__ float warp_sq[4][WARPS];
  __shared__ float s_rrms[4];
  float o[4][4];
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    const half* rh = reinterpret_cast<const half*>(&res_vec[s]);
    uint2 out_vec;
    half* oh = reinterpret_cast<half*>(&out_vec);
    float sq = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const half h = __float2half_rn(__half2float(rh[j]) + B[j] * g[s]);
      oh[j] = h;
      if constexpr (WITH_NORM) {
        const float of = __half2float(h);
        o[s][j] = of;
        sq = __fmaf_rn(of, of, sq);
      }
    }
    *reinterpret_cast<uint2*>(res_out + static_cast<size_t>(row) * KD + s * HD +
                              col0) = out_vec;
    if constexpr (WITH_NORM) {
#pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        sq += __shfl_down_sync(0xffffffffu, sq, off);
      if (lane == 0) warp_sq[s][warp] = sq;
    }
  }

  if constexpr (WITH_NORM) {
    __syncthreads();
    if (tid < 4) {
      float tot = 0.f;
#pragma unroll
      for (int w = 0; w < WARPS; ++w) tot += warp_sq[tid][w];
      s_rrms[tid] = rsqrtf(tot * (1.0f / 2560.0f) + eps);
    }
    __syncthreads();
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const float rrms = s_rrms[s];
      const half* wh = reinterpret_cast<const half*>(&nw_vec[s]);
      uint2 n_vec;
      half* nh = reinterpret_cast<half*>(&n_vec);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float y = o[s][j] * rrms;
        nh[j] = __float2half_rn(__fmaf_rn(y, __half2float(wh[j]), y));
      }
      *reinterpret_cast<uint2*>(norm_out + static_cast<size_t>(row) * KD +
                                s * HD + col0) = n_vec;
    }
  }

  __syncthreads();
  if (tid == 0) {
    __threadfence();
    if (atomicAdd(cnt, 1u) == static_cast<unsigned>(total_m - 1)) {
      *cnt = 0u;
      st_vol(dirty_m + page, static_cast<unsigned>(total_m));
      st_vol(seq, next_ep);
    }
  }
}
}  // namespace

void sm70_hc_ll_down_out(torch::Tensor x, torch::Tensor wd, torch::Tensor part,
                         torch::Tensor cnt, std::vector<int64_t> ll,
                         torch::Tensor seq, int64_t rank, int64_t variant,
                         bool optimized_loads) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.dim() == 2 &&
                  x.size(0) >= 1 && x.size(0) <= kMaxRows &&
                  x.size(1) == 10240 && x.is_contiguous(),
              "HC LL requires CUDA FP16 [1..64,10240]");
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
                  part.numel() >= 40 * kPageRows * 96 && cnt.numel() >= 4,
              "HC LL down workspace or weight shape mismatch");
  auto st = at::cuda::getCurrentCUDAStream().stream();
  Peer p = make_peer(ll, seq, rank);
  auto c = reinterpret_cast<unsigned*>(cnt.data_ptr<int>());
#define D(S, W, O)                                                             \
  down3<S, W, O><<<dim3(3, S), 32 * W, 0, st>>>(                               \
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
                  x.size(0) >= 1 && x.size(0) <= kMaxRows &&
                  x.size(1) == 10240 && x.is_contiguous(),
              "HC LL requires CUDA FP16 [1..64,10240]");
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
                  (lora_out.numel() == 0 ||
                   lora_out.sizes() == at::IntArrayRef({M, 320})) &&
                  inj_out.sizes() == at::IntArrayRef({M, 4}),
              "HC LL up workspace or shape mismatch");
  auto st = at::cuda::getCurrentCUDAStream().stream();
  Peer p = make_peer(ll, seq, rank);
  auto c = reinterpret_cast<unsigned*>(cnt.data_ptr<int>());
  half* lora_ptr =
      lora_out.numel() ? reinterpret_cast<half*>(lora_out.data_ptr()) : nullptr;
#define U(W, O)                                                    \
  up3<W, O><<<80, 32 * W, 0, st>>>(                                \
      reinterpret_cast<const uint32_t*>(ll_lora),                  \
      reinterpret_cast<const half*>(wu.data_ptr()),                \
      reinterpret_cast<const half*>(x.data_ptr()), c, p,           \
      reinterpret_cast<const unsigned*>(down_seq.data_ptr<int>()), \
      reinterpret_cast<half*>(out.data_ptr()), lora_ptr,           \
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

void sm70_hc_ll_add_all_gather_fc_out(torch::Tensor out, torch::Tensor el,
                                      torch::Tensor hl, torch::Tensor cnt,
                                      torch::Tensor dirty_m,
                                      std::vector<int64_t> ll,
                                      torch::Tensor seq, int64_t rank,
                                      std::vector<int64_t> phys_order) {
  TORCH_CHECK(el.is_cuda() && el.scalar_type() == at::kHalf && el.dim() == 2 &&
                  el.size(0) >= 1 && el.size(0) <= kMaxRows &&
                  el.size(1) == 640 && el.is_contiguous(),
              "FC LL gather requires CUDA FP16 el [1..64, 640]");
  const c10::cuda::CUDAGuard guard(el.device());
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major == 7 && properties->minor == 0,
              "SM70 required");
  const int64_t M = el.size(0);
  TORCH_CHECK(
      hl.device() == el.device() && hl.scalar_type() == at::kHalf &&
          hl.is_contiguous() && hl.sizes() == at::IntArrayRef({M, 4, 640}) &&
          out.device() == el.device() && out.scalar_type() == at::kHalf &&
          out.is_contiguous() && out.sizes() == at::IntArrayRef({M, 10240}),
      "FC LL gather tensor shape/dtype mismatch");
  for (const auto& t : {cnt, dirty_m, seq}) {
    TORCH_CHECK(t.device() == el.device() && t.scalar_type() == at::kInt &&
                    t.is_contiguous(),
                "FC LL gather counters must be contiguous int32");
  }
  TORCH_CHECK(cnt.numel() >= 1 && dirty_m.numel() >= 2 && seq.numel() == 1 &&
                  ll.size() == 4 && phys_order.size() == 4 && rank >= 0 &&
                  rank < 4 && ll[rank] && ll[rank ^ 1] && ll[rank ^ 2],
              "FC LL gather invalid peer pointers or rank");
  PhysOrder4 phys{};
  for (int i = 0; i < 4; ++i) {
    TORCH_CHECK(phys_order[i] >= 0 && phys_order[i] < 4,
                "Invalid phys_order entry");
    phys.v[i] = static_cast<int>(phys_order[i]);
  }
  auto st = at::cuda::getCurrentCUDAStream().stream();
  Peer p = make_peer(ll, seq, rank);
  fc_add_all_gather_kernel<<<80, 128, 0, st>>>(
      reinterpret_cast<half*>(out.data_ptr()),
      reinterpret_cast<const half*>(el.data_ptr()),
      reinterpret_cast<const half*>(hl.data_ptr()),
      reinterpret_cast<unsigned*>(cnt.data_ptr<int>()),
      reinterpret_cast<unsigned*>(dirty_m.data_ptr<int>()), p, phys,
      static_cast<int>(M));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sm70_hc_ll_allreduce_combine_norm_out(
    torch::Tensor res_out, torch::Tensor norm_out, torch::Tensor p0,
    torch::Tensor p1, torch::Tensor res, torch::Tensor inj, torch::Tensor nw,
    double eps, torch::Tensor cnt, torch::Tensor dirty_m, torch::Tensor seq,
    std::vector<int64_t> ar_ptrs, int64_t rank) {
  TORCH_CHECK(
      p0.is_cuda() && p0.scalar_type() == at::kHalf && p0.dim() == 2 &&
          p0.size(0) >= 1 && p0.size(0) <= kMaxRows && p0.size(1) == 2560 &&
          p0.is_contiguous(),
      "HC LL allreduce_combine_norm requires CUDA FP16 p0 [1..64, 2560]");
  const c10::cuda::CUDAGuard guard(p0.device());
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major == 7 && properties->minor == 0,
              "SM70 required");
  const int64_t M = p0.size(0);
  const bool has_p1 = p1.defined() && p1.numel() > 0;
  const bool with_norm = norm_out.defined() && norm_out.numel() > 0;
  if (has_p1) {
    TORCH_CHECK(p1.device() == p0.device() && p1.scalar_type() == at::kHalf &&
                    p1.is_contiguous() &&
                    p1.sizes() == at::IntArrayRef({M, 2560}) &&
                    reinterpret_cast<uintptr_t>(p1.data_ptr()) % 16 == 0,
                "Invalid secondary partial p1");
  }
  for (const auto& t : {res_out, res, inj}) {
    TORCH_CHECK(t.device() == p0.device() && t.scalar_type() == at::kHalf &&
                    t.is_contiguous(),
                "HC LL combine tensors must be contiguous FP16");
  }
  TORCH_CHECK(res_out.sizes() == at::IntArrayRef({M, 10240}) &&
                  res.sizes() == at::IntArrayRef({M, 10240}) &&
                  inj.sizes() == at::IntArrayRef({M, 4}) &&
                  reinterpret_cast<uintptr_t>(p0.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(res.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(res_out.data_ptr()) % 16 == 0,
              "HC LL combine shape/alignment mismatch");
  int nw_full = 0;
  const half* nw_ptr = nullptr;
  half* norm_ptr = nullptr;
  if (with_norm) {
    TORCH_CHECK(
        norm_out.device() == p0.device() &&
            norm_out.scalar_type() == at::kHalf && norm_out.is_contiguous() &&
            norm_out.sizes() == at::IntArrayRef({M, 10240}) &&
            reinterpret_cast<uintptr_t>(norm_out.data_ptr()) % 16 == 0 &&
            nw.device() == p0.device() && nw.scalar_type() == at::kHalf &&
            nw.is_contiguous() && (nw.numel() == 2560 || nw.numel() == 10240) &&
            reinterpret_cast<uintptr_t>(nw.data_ptr()) % 16 == 0,
        "HC LL norm tensor shape/alignment mismatch");
    nw_full = (nw.numel() == 10240) ? 1 : 0;
    nw_ptr = reinterpret_cast<const half*>(nw.data_ptr());
    norm_ptr = reinterpret_cast<half*>(norm_out.data_ptr());
  }
  for (const auto& t : {cnt, seq}) {
    TORCH_CHECK(t.device() == p0.device() && t.scalar_type() == at::kInt &&
                    t.is_contiguous() && t.numel() >= 1,
                "HC LL allreduce counters must be contiguous int32");
  }
  TORCH_CHECK(dirty_m.device() == p0.device() &&
                  dirty_m.scalar_type() == at::kInt &&
                  dirty_m.is_contiguous() && dirty_m.numel() >= 2,
              "HC LL allreduce dirty_m must have >= 2 int32 elements");
  TORCH_CHECK(ar_ptrs.size() == 4 && rank >= 0 && rank < 4 && ar_ptrs[0] &&
                  ar_ptrs[1] && ar_ptrs[2] && ar_ptrs[3],
              "HC LL allreduce requires full-mesh 4-rank peer pointers");
  ArPeer4 peers{};
  for (int i = 0; i < 4; ++i) {
    peers.ar[i] =
        reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(ar_ptrs[i]));
  }
  const half* p1_ptr =
      has_p1 ? reinterpret_cast<const half*>(p1.data_ptr()) : nullptr;
  auto st = at::cuda::getCurrentCUDAStream().stream();
  const int m = static_cast<int>(M);
  auto* ro = reinterpret_cast<half*>(res_out.data_ptr());
  const auto* p0_ptr = reinterpret_cast<const half*>(p0.data_ptr());
  const auto* res_ptr = reinterpret_cast<const half*>(res.data_ptr());
  const auto* inj_ptr = reinterpret_cast<const half*>(inj.data_ptr());
  auto* cnt_ptr = reinterpret_cast<unsigned*>(cnt.data_ptr<int>());
  auto* dm_ptr = reinterpret_cast<unsigned*>(dirty_m.data_ptr<int>());
  auto* seq_ptr = reinterpret_cast<unsigned*>(seq.data_ptr<int>());
  const float eps_f = static_cast<float>(eps);
  const int rk = static_cast<int>(rank);
  if (with_norm) {
    allreduce_combine_norm_kernel<true><<<m, 640, 0, st>>>(
        ro, norm_ptr, p0_ptr, p1_ptr, res_ptr, inj_ptr, nw_ptr, nw_full, eps_f,
        cnt_ptr, dm_ptr, seq_ptr, peers, rk, m);
  } else {
    allreduce_combine_norm_kernel<false><<<m, 640, 0, st>>>(
        ro, nullptr, p0_ptr, p1_ptr, res_ptr, inj_ptr, nullptr, 0, eps_f,
        cnt_ptr, dm_ptr, seq_ptr, peers, rk, m);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "sm70_hc_ll_add_all_gather_fc_out(Tensor(a!) out, Tensor el, Tensor hl, "
      "Tensor(b!) cnt, Tensor(c!) dirty_m, int[] ll_ptrs, Tensor(d!) seq, "
      "int logical_rank, int[] phys_order) -> ()");
  m.def(
      "sm70_hc_ll_allreduce_combine_norm_out(Tensor(a!) res_out, Tensor(b!) "
      "norm_out, Tensor p0, Tensor p1, Tensor res, Tensor inj, Tensor nw, "
      "float eps, Tensor(c!) cnt, Tensor(d!) dirty_m, Tensor(e!) seq, "
      "int[] ar_ptrs, int logical_rank) -> ()");
}

TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("sm70_hc_ll_add_all_gather_fc_out", &sm70_hc_ll_add_all_gather_fc_out);
  m.impl("sm70_hc_ll_allreduce_combine_norm_out",
         &sm70_hc_ll_allreduce_combine_norm_out);
}
