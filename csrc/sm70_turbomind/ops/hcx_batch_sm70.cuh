// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Included in namespace hcx, after the retained HCX primitives. Full-mesh
// TP4, separate output projection, local schedule, M9..40 only. Sixteen
// warps expose independent row exchange/reduction work; the five HC up warps
// and six HC down warps retain their original arithmetic. One resident
// grid performs the exchange and both grid rendezvous for the whole batch.
constexpr int BATCH_ROWS = 40;
constexpr int BATCH_THREADS = 512;

template <bool HalfPackets = false>
__global__ __launch_bounds__(BATCH_THREADS, 1) void hcx_batch_kernel(Args a) {
  constexpr int WARPS = BATCH_THREADS / 32;
  const int i = blockIdx.x, t = threadIdx.x, warp = t >> 5, lane = t & 31;
  const int rk = a.rank, p1 = rk ^ 1, p2 = rk ^ 2, p3 = rk ^ 3;
  const unsigned tag = ldv(a.seq) + 1, base = ldv(a.bar + 3);
  const int r8 = (lane & 3) + ((lane & 16) ? 4 : 0), quad = (lane >> 2) & 3;
  constexpr int AR_STRIDE = BATCH_ROWS * 32;
  __shared__ __align__(16) half xs[BATCH_ROWS][136];
  __shared__ __align__(16) half ls[BATCH_ROWS][328];
  // The LoRA input tile is not populated until after the down projection
  // and both grid rendezvous. Reuse its earlier lifetime for a coalesced
  // split-K partial tile instead of allocating another shared-memory bank.
  static_assert(sizeof(ls) >= 8 * 388 * sizeof(float));
  float* const partial_tile = reinterpret_cast<float*>(ls);
  __shared__ float red[5][8][33];
  __shared__ float rr[BATCH_ROWS][4];

  if constexpr (HalfPackets) {
    for (int row = warp; row < a.M; row += WARPS) {
      const int col = 32 * i + lane;
      const bool packed = a.p1 == nullptr;
      const int slot = (rk * NC + i) * AR_STRIDE + row * 32 + lane;
      if (!packed || lane < 16) {
        unsigned value;
        if (packed) {
          const half* pair = a.p0 + row * HD + 32 * i + 2 * lane;
          value = static_cast<unsigned>(__half_as_ushort(pair[0])) |
                  (static_cast<unsigned>(__half_as_ushort(pair[1])) << 16);
        } else {
          float v = __half2float(a.p0[row * HD + col]);
          v += __half2float(a.p1[row * HD + col]);
          value = __float_as_uint(v);
        }
        st2(a.ar[p1] + slot, value, tag);
        st2(a.ar[p2] + slot, value, tag);
        st2(a.ar[p3] + slot, value, tag);
      }
      // A tagged header in previously unused LoRA padding preserves each peer's
      // optional-secondary semantics, including mixed formats across TP ranks.
      // Every data word retains its existing payload+epoch validation.
      if (i == 0 && lane == 0) {
        const int header = row * LW + 324 + rk;
        st2(a.lora[p1] + header, packed, tag);
        st2(a.lora[p2] + header, packed, tag);
        st2(a.lora[p3] + header, packed, tag);
      }
    }
  } else {
    for (int row = warp; row < a.M; row += WARPS) {
      const int col = 32 * i + lane;
      float v = __half2float(a.p0[row * HD + col]);
      if (a.p1) v += __half2float(a.p1[row * HD + col]);
      const int slot = (rk * NC + i) * AR_STRIDE + row * 32 + lane;
      st2(a.ar[p1] + slot, __float_as_uint(v), tag);
      st2(a.ar[p2] + slot, __float_as_uint(v), tag);
      st2(a.ar[p3] + slot, __float_as_uint(v), tag);
    }
  }
  uint4 dlo[4], dhi[4], ulo[4], uhi[4];
  if (warp < 6) {
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const uint4* p = a.wd + ((i * 6 + warp) * 4 + s) * 64;
      dlo[s] = ldgv(p + lane);
      dhi[s] = ldgv(p + 32 + lane);
    }
  }
  if (warp >= 3 && warp < 8) {
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const uint4* p = a.wu + ((i * 5 + warp - 3) * 4 + s) * 64;
      ulo[s] = ldgv(p + lane);
      uhi[s] = ldgv(p + 32 + lane);
    }
  }
  for (int row = warp; row < BATCH_ROWS; row += WARPS) {
    const int col = 32 * i + lane;
    float sums[4] = {};
#pragma unroll
    for (int b = 0; b < 4; ++b) xs[row][b * 32 + lane] = __float2half_rn(0.f);
    if (row < a.M) {
      float v = __half2float(a.p0[row * HD + col]);
      if (a.p1) v += __half2float(a.p1[row * HD + col]);
      unsigned f1 = 0, f2 = 0, f3 = 0;
      if constexpr (HalfPackets) {
        f1 = poll2(a.lora[rk] + row * LW + 324 + p1, tag);
        f2 = poll2(a.lora[rk] + row * LW + 324 + p2, tag);
        f3 = poll2(a.lora[rk] + row * LW + 324 + p3, tag);
      }
      const int slot = i * AR_STRIDE + row * 32;
      // Retain the qualified small-HC schedule: issue all independent peer
      // loads before waiting for any one epoch. Arithmetic order is unchanged.
      unsigned u1, u2, u3, e1, e2, e3;
      const auto start = clock64();
      do {
        asm volatile("ld.volatile.global.v2.u32 {%0,%1}, [%2];"
                     : "=r"(u1), "=r"(e1)
                     : "l"(a.ar[rk] + p1 * NC * AR_STRIDE + slot +
                           (f1 ? lane / 2 : lane)));
        asm volatile("ld.volatile.global.v2.u32 {%0,%1}, [%2];"
                     : "=r"(u2), "=r"(e2)
                     : "l"(a.ar[rk] + p2 * NC * AR_STRIDE + slot +
                           (f2 ? lane / 2 : lane)));
        asm volatile("ld.volatile.global.v2.u32 {%0,%1}, [%2];"
                     : "=r"(u3), "=r"(e3)
                     : "l"(a.ar[rk] + p3 * NC * AR_STRIDE + slot +
                           (f3 ? lane / 2 : lane)));
        if (clock64() - start > 4000000000LL) __trap();
      } while (e1 != tag || e2 != tag || e3 != tag);
      const int shift = 16 * (lane & 1);
      const float x1 = f1 ? __half2float(__ushort_as_half(u1 >> shift))
                          : __uint_as_float(u1);
      const float x2 = f2 ? __half2float(__ushort_as_half(u2 >> shift))
                          : __uint_as_float(u2);
      const float x3 = f3 ? __half2float(__ushort_as_half(u3 >> shift))
                          : __uint_as_float(u3);
      const float near = (rk & 1) ? x1 + v : v + x1;
      const float far = (rk & 1) ? x3 + x2 : x2 + x3;
      const float reduced =
          __half2float(__float2half_rn((rk & 2) ? far + near : near + far));
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        const float g = 2.f * sigm(__half2float(a.inj[row * 4 + b]) / 4.f);
        const half h = __float2half_rn(
            __half2float(a.res[row * KD + b * HD + col]) + reduced * g);
        const float o = __half2float(h);
        const half nw = a.nw[a.nw_full ? b * HD + col : col];
        a.res_out[row * KD + b * HD + col] = h;
        xs[row][b * 32 + lane] = __float2half_rn(o + o * __half2float(nw));
        sums[b] = o * o;
      }
    }
#pragma unroll
    for (int s = 16; s > 0; s >>= 1)
#pragma unroll
      for (int b = 0; b < 4; ++b)
        sums[b] += __shfl_xor_sync(0xffffffff, sums[b], s);
    if (lane == 0 && row < a.M)
      *reinterpret_cast<float4*>(a.sq + (i * BATCH_ROWS + row) * 4) =
          make_float4(sums[0], sums[1], sums[2], sums[3]);
  }
  __syncthreads();
  const int tile = warp >> 1, kh = warp & 1;
#pragma unroll
  for (int token_tile = 0; token_tile < 5; ++token_tile) {
    if (token_tile * 8 >= a.M) break;
    if (warp < 6) {
      float acc0[8] = {}, acc1[8] = {};
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const int k = (kh * 4 + s) * 16;
        const uint4 x0 =
            *reinterpret_cast<const uint4*>(&xs[token_tile * 8 + r8][k]);
        const uint4 x1 =
            *reinterpret_cast<const uint4*>(&xs[token_tile * 8 + r8][k + 8]);
        float (&acc)[8] = s < 2 ? acc0 : acc1;
        mma(acc, x0.x, x0.y, dlo[s].x, dlo[s].y);
        mma(acc, x0.z, x0.w, dlo[s].z, dlo[s].w);
        mma(acc, x1.x, x1.y, dhi[s].x, dhi[s].y);
        mma(acc, x1.z, x1.w, dhi[s].z, dhi[s].w);
      }
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int rw =
            token_tile * 8 + (e & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
        const int n = tile * 32 + quad * 8 +
                      ((e & 1) | (((lane >> 1) & 1) << 1) | ((e >> 2) << 2));
        *reinterpret_cast<float2*>(partial_tile + (rw - token_tile * 8) * 388 +
                                   n * 4 + 2 * kh) =
            make_float2(acc0[e], acc1[e]);
      }
    }
    __syncthreads();
    for (int cell = t; cell < 8 * 96; cell += BATCH_THREADS) {
      const int row = cell / 96, col = cell % 96;
      if (token_tile * 8 + row < a.M)
        *reinterpret_cast<float4*>(
            a.dpart +
            ((i * BATCH_ROWS + token_tile * 8 + row) * 96 + col) * 4) =
            *reinterpret_cast<const float4*>(partial_tile + row * 388 +
                                             col * 4);
    }
    __syncthreads();
  }
  // Reserved padded down columns84..95 have no downstream readers. The
  // producing CTA resets each row flag after all its partial stores, before
  // the existing release/acquire grid barrier. No allocation/init dependency.
  if (i == NC - 1 && t < a.M) {
    float* slot = a.dpart + ((NC - 1) * BATCH_ROWS + t) * 96 * 4 + 84 * 4;
    *reinterpret_cast<unsigned*>(slot + 4) = tag - 1;
  }
  grid_bar_epoch(a.bar + 2, base + NC);
  // A resident owner computes the identical80-way statistic once per row.
  if (i < a.M && warp == 0) {
    const int row = i;
    float4 qs[3];
#pragma unroll
    for (int j = 0; j < 3; ++j)
      qs[j] = lane + 32 * j < NC
                  ? __ldcg(reinterpret_cast<const float4*>(a.sq) +
                           (lane + 32 * j) * BATCH_ROWS + row)
                  : make_float4(0.f, 0.f, 0.f, 0.f);
    float sums[4] = {
        (qs[0].x + qs[1].x) + qs[2].x, (qs[0].y + qs[1].y) + qs[2].y,
        (qs[0].z + qs[1].z) + qs[2].z, (qs[0].w + qs[1].w) + qs[2].w};
#pragma unroll
    for (int s = 16; s > 0; s >>= 1)
#pragma unroll
      for (int b = 0; b < 4; ++b)
        sums[b] += __shfl_xor_sync(0xffffffff, sums[b], s);

    if (lane == 0) {
      float* slot = a.dpart + ((NC - 1) * BATCH_ROWS + row) * 96 * 4 + 84 * 4;
      *reinterpret_cast<float4*>(slot) = make_float4(
          rsqrtf(sums[0] / HD + a.eps), rsqrtf(sums[1] / HD + a.eps),
          rsqrtf(sums[2] / HD + a.eps), rsqrtf(sums[3] / HD + a.eps));
      asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(slot + 4),
                   "r"(tag)
                   : "memory");
    }
  }
  for (int row = warp; row < a.M; row += WARPS) {
    float inverse_values[4] = {};
    if (lane == 0) {
      const float* slot =
          a.dpart + ((NC - 1) * BATCH_ROWS + row) * 96 * 4 + 84 * 4;
      const auto start = clock64();
      unsigned ready;
      do {
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
                     : "=r"(ready)
                     : "l"(slot + 4)
                     : "memory");
        if (clock64() - start > 4000000000LL) __trap();
      } while (ready != tag);
      const float4 value = __ldcg(reinterpret_cast<const float4*>(slot));
      inverse_values[0] = value.x;
      inverse_values[1] = value.y;
      inverse_values[2] = value.z;
      inverse_values[3] = value.w;
    }
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      const float inverse = __shfl_sync(0xffffffff, inverse_values[b], 0);
      if (lane == 0) rr[row][b] = inverse;
      const float o =
          __half2float(a.res_out[row * KD + b * HD + 32 * i + lane]);
      float y = o * inverse;
      y += y * __half2float(
                   a.nw[a.nw_full ? b * HD + 32 * i + lane : 32 * i + lane]);
      a.xn[row * KD + b * HD + 32 * i + lane] = __float2half_rn(y);
    }
  }
  __syncthreads();
  for (int oi = i + NC * warp; oi < a.M * 96; oi += NC * WARPS) {
    const int rw = oi / 96, n = oi % 96;
    if (!(n < 80 || (rk == 3 && n < 84))) continue;
    float pb[4] = {};
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      if (lane + 32 * j < NC) {
        const float4 pv = __ldcg(reinterpret_cast<const float4*>(a.dpart) +
                                 ((lane + 32 * j) * BATCH_ROWS + rw) * 96 + n);
        pb[0] += pv.x;
        pb[1] += pv.y;
        pb[2] += pv.z;
        pb[3] += pv.w;
      }
    }
#pragma unroll
    for (int s = 16; s > 0; s >>= 1)
#pragma unroll
      for (int b = 0; b < 4; ++b)
        pb[b] += __shfl_xor_sync(0xffffffff, pb[b], s);
    float q = 0.f;
#pragma unroll
    for (int b = 0; b < 4; ++b) q = fmaf(rr[rw][b], pb[b], q);
    half hv = __float2half_rn(q);
    if (n < 80) {
      const float v = divf(__half2float(hv), 4.f);
      hv = __float2half_rn(v * sigm(v));
    }
    const int pos = rw * LW + (n < 80 ? rk * 80 + n : 320 + n - 80);
    const unsigned value = __half_as_ushort(hv);
    if (lane == 0) st2(a.lora[rk] + pos, value, tag);
    if (lane == 1) st2(a.lora[p1] + pos, value, tag);
    if (lane == 2) st2(a.lora[p2] + pos, value, tag);
    if (lane == 3) st2(a.lora[p3] + pos, value, tag);
  }
  for (int idx = i * BATCH_THREADS + t; idx < a.M * 324;
       idx += NC * BATCH_THREADS)
    poll2(a.lora[rk] + (idx / 324) * LW + idx % 324, tag);
  grid_bar_epoch(a.bar + 2, base + 2 * NC);
  for (int idx = t; idx < BATCH_ROWS * 320; idx += BATCH_THREADS)
    ls[idx / 320][idx % 320] =
        idx < a.M * 320
            ? __ushort_as_half(static_cast<unsigned short>(
                  __ldcg(a.lora[rk] + (idx / 320) * LW + idx % 320).x))
            : __float2half_rn(0.f);
  if (i == 0 && t < a.M * 4)
    a.inj_out[t] = __ushort_as_half(static_cast<unsigned short>(
        __ldcg(a.lora[rk] + (t >> 2) * LW + 320 + (t & 3)).x));
  __syncthreads();
#pragma unroll
  for (int token_tile = 0; token_tile < 5; ++token_tile) {
    if (token_tile * 8 >= a.M) break;
    if (warp >= 3 && warp < 8) {
      const int uw = warp - 3;
      float acc[8] = {};
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const int k = (uw * 4 + s) * 16;
        const uint4 x0 =
            *reinterpret_cast<const uint4*>(&ls[token_tile * 8 + r8][k]);
        const uint4 x1 =
            *reinterpret_cast<const uint4*>(&ls[token_tile * 8 + r8][k + 8]);
        mma(acc, x0.x, x0.y, ulo[s].x, ulo[s].y);
        mma(acc, x0.z, x0.w, ulo[s].z, ulo[s].w);
        mma(acc, x1.x, x1.y, uhi[s].x, uhi[s].y);
        mma(acc, x1.z, x1.w, uhi[s].z, uhi[s].w);
      }
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int rw = (e & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
        red[uw][rw][quad * 8 + ((e & 1) | (lane & 2) | (e & 4))] = acc[e];
      }
    }
    __syncthreads();
    const int row = token_tile * 8 + warp;
    if (warp < 8 && row < a.M) {
      const int h = 640 * rk + 8 * i + (lane & 7);
      float acc = red[0][warp][lane];
#pragma unroll
      for (int q = 1; q < 5; ++q) acc += red[q][warp][lane];
      const float sg = sigm(__half2float(__float2half_rn(acc)));
      const float xv =
          __half2float(__ldcg(a.xn + row * KD + (lane >> 3) * HD + h));
      float mixed = 0.f;
#pragma unroll
      for (int b = 0; b < 4; ++b)
        mixed = fmaf(__shfl_sync(0xffffffff, sg, b * 8 + (lane & 7)),
                     __shfl_sync(0xffffffff, xv, b * 8 + (lane & 7)), mixed);
      const unsigned bits = __half_as_ushort(__float2half_rn(divf(mixed, 4.f)));
      const unsigned packed =
          bits | (__shfl_down_sync(0xffffffff, bits, 1) << 16);
      const unsigned upper = __shfl_down_sync(0xffffffff, packed, 2);
      if (lane < 8 && !(lane & 3)) {
        st4(a.hb[p1] + (row * HD + h) / 2, packed, upper, tag);
        st4(a.hb[p2] + (row * HD + h) / 2, packed, upper, tag);
        st4(a.hb[p3] + (row * HD + h) / 2, packed, upper, tag);
        *reinterpret_cast<uint2*>(a.blk_out + row * HD + h) =
            make_uint2(packed, upper);
      }
    }
    __syncthreads();
  }
  for (int u = t; u < 6 * a.M; u += BATCH_THREADS) {
    const int src = u % 3 == 0 ? p1 : u % 3 == 1 ? p2 : p3;
    const int row = (u / 3) >> 1, h = 640 * src + 8 * i + ((u / 3) & 1) * 4;
    *reinterpret_cast<uint2*>(a.blk_out + row * HD + h) =
        poll4(a.hb[rk] + (row * HD + h) / 2, tag);
  }
  if (i == 0 && t == 0) {
    a.bar[3] = base + 2 * NC;
    a.seq[0] = tag;
  }
}
