// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once
// Included inside namespace hcx after hcx_kernel.
//
// Concurrent verification batches (up to MAX_M rows): the same per-row
// arithmetic as hcx_kernel, but each phase covers every 8-row group before
// the next exchange or grid barrier. All groups' all-reduce words are pushed
// before any is polled, and one pair of grid barriers serves the launch, so
// the cross-GPU latency is paid once instead of once per group. Each group
// keeps its own exchange slots and scratch; weights stay in registers.

constexpr int GMAX = MAX_M / 8;

// G > 0 fixes the group count at compile time: the per-group loops then have
// no exits, so independent groups' loads and MMAs interleave. G = 0 keeps the
// runtime count (producer-fused variants).
template <bool FULL, int OF, int G = 0, bool LOCAL_SCHEDULE = false>
__global__ __launch_bounds__(256, 1) void hcx_batch_kernel(Args a) {
  TS(0);
  constexpr int GP = G > 0 ? G : GMAX;
  const int i = blockIdx.x, t = threadIdx.x, warp = t >> 5, lane = t & 31;
  const int MT = a.M, rk = a.rank, p1 = rk ^ 1, p2 = rk ^ 2, p3 = rk ^ 3;
  const unsigned ep = ldv(a.seq), tag = ep + 1;
  const unsigned base = LOCAL_SCHEDULE ? ldv(a.bar + 3) : 0;
  const int r8 = (lane & 3) + ((lane & 16) ? 4 : 0), quad = (lane >> 2) & 3;
  const int groups = G > 0 ? G : (MT + 7) / 8;
  const int row = warp, col = 32 * i + lane;
  auto rows_of = [&](int g) { return min(8, MT - 8 * g); };
  auto ar = [&](int g, int q) { return a.ar[q] + g * AR_WORDS; };
  auto lora = [&](int g, int q) { return a.lora[q] + g * LORA_WORDS; };
  auto hb = [&](int g, int q) { return a.hb[q] + g * HB_WORDS; };
  auto sq = [&](int g) { return a.sq + g * NC * 8 * 4; };
  auto dpart = [&](int g) { return a.dpart + g * NC * 8 * 96 * 4; };

  // Keep static shared memory strictly below Volta's 32 KB carveout boundary
  // (~12.6 KB across all G=1..5) by overlapping down-stage xs and up-stage
  // single-group ls + reduction buffers.
  union StageSmem {
    half xs[GP][8][128 + 8];
    struct {
      half ls[8][320 + 8];
      float red[5][8][LOCAL_SCHEDULE ? 33 : 32];
    } up;
  };
  __shared__ __align__(16) StageSmem smem;
  __shared__ __align__(16) half tout[GP][8][8];
  __shared__ float rr[GP][8][4];

  float v[GP];
#pragma unroll
  for (int g = 0; g < GP; ++g) v[g] = 0.f;

  // ---- 0. fused producer (o-proj / gated GDN out-proj) for every group.
  if constexpr (OF >= 0) {
    if (warp < 6)
      pf_l2(reinterpret_cast<const char*>(a.wd + (i * 6 + warp) * 4 * 64) +
            lane * 128);
    if (warp >= 3)
      pf_l2(reinterpret_cast<const char*>(a.wu + (i * 5 + warp - 3) * 4 * 64) +
            lane * 128);
    extern __shared__ uint4 dsm[];
    const int Kp = a.oG * 128;
    if (a.gz) {
      const int heads = Kp / 128, nrows = MT * heads;
      const uint2 wv = *reinterpret_cast<const uint2*>(a.gw + lane * 4);
      for (int rw = i * 8 + warp; rw < nrows; rw += NC * 8) {
        const int tok = rw / heads, hd = rw % heads;
        const uint2 xv = __ldg(reinterpret_cast<const uint2*>(
            a.ox + tok * a.oldx + hd * 128 + lane * 4));
        const uint2 zv = __ldg(reinterpret_cast<const uint2*>(
            a.gz + tok * a.ldz + hd * 128 + lane * 4));
        const half* xh = reinterpret_cast<const half*>(&xv);
        const half* zh = reinterpret_cast<const half*>(&zv);
        const half* wh = reinterpret_cast<const half*>(&wv);
        float values[4], gates[4], weights[4], sum = 0.0f;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          values[q] = __half2float(xh[q]);
          gates[q] = __half2float(zh[q]);
          weights[q] = __half2float(wh[q]);
          sum = __fadd_rn(sum, __fmul_rn(values[q], values[q]));
        }
        for (int off = 16; off > 0; off >>= 1)
          sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffU, sum, off));
        sum = __shfl_sync(0xffffffffU, sum, 0);
        const float mean = __fmul_rn(sum, 1.0f / 128);
        const float inverse = rsqrtf(__fadd_rn(mean, a.geps));
        half outv[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const float normalized =
              __fmul_rn(__fmul_rn(values[q], inverse), weights[q]);
          const float denominator = __fadd_rn(1.0f, expf(-gates[q]));
          outv[q] =
              __float2half_rn(__fmul_rn(normalized, gates[q] / denominator));
        }
        __stcg(
            reinterpret_cast<uint2*>(a.gscr + tok * Kp + hd * 128 + lane * 4),
            *reinterpret_cast<uint2*>(outv));
      }
      grid_bar(a.bar, NC);
    }
#pragma unroll
    for (int g = 0; g < GP; ++g) {
      if (G == 0 && g >= groups) break;
      const int M = rows_of(g), r0 = 8 * g;
      float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      body6<OF, 8, 1>(a.oseg, i, true, warp, 0, 0, a.oG, 4 * a.oG, a.oG,
                      a.gz ? a.gscr + r0 * Kp : a.ox + r0 * a.oldx,
                      a.gz ? Kp : a.oldx, M, dsm, nullptr, acc, nullptr);
      __syncthreads();
      float* rd = reinterpret_cast<float*>(dsm);
#pragma unroll
      for (int e = 0; e < 8; ++e) rd[warp * 256 + lane * 8 + e] = acc[e];
      __syncthreads();
      float sum = 0.f;
#pragma unroll
      for (int k = 0; k < 8; ++k) sum += rd[k * 256 + t];
      __syncthreads();
      {
        const int lv = t >> 3, e = t & 7;
        const int token = (e & 2) | ((lv & 16) ? 4 : 0) | (lv & 1);
        const int c = ((lv >> 2) & 3) * 8 +
                      ((e & 1) | (((lv >> 1) & 1) << 1) | ((e >> 2) << 2));
        rd[token * 32 + c] = __half2float(__float2half_rn(sum));
      }
      __syncthreads();
      if (row < M) v[g] = rd[row * 32 + lane];
      __syncthreads();
    }
  }

  // ---- 1. push every group's partial for columns [32i, 32i + 32). All
  // loads are issued before the (volatile, ordered) remote stores.
#pragma unroll
  for (int g = 0; g < GP; ++g) {
    if (G == 0 && g >= groups) break;
    if (row >= rows_of(g)) continue;
    const int gr = 8 * g + row;
    if constexpr (OF < 0) v[g] = __half2float(a.p0[gr * HD + col]);
    if (a.p1) v[g] += __half2float(a.p1[gr * HD + col]);
  }
#pragma unroll
  for (int g = 0; g < GP; ++g) {
    if (G == 0 && g >= groups) break;
    if (row >= rows_of(g)) continue;
    if (FULL) {
      const int mine = (rk * NC + i) * 256 + t;
      st2(ar(g, p1) + mine, __float_as_uint(v[g]), tag);
      st2(ar(g, p2) + mine, __float_as_uint(v[g]), tag);
      st2(ar(g, p3) + mine, __float_as_uint(v[g]), tag);
    } else {
      st2(ar(g, p1) + i * 256 + t, __float_as_uint(v[g]), tag);
    }
  }
  TS(1);
  // Small combine operands; the norm weight does not depend on the row.
  half nwv[4];
#pragma unroll
  for (int b = 0; b < 4; ++b) nwv[b] = a.nw[a.nw_full ? b * HD + col : col];
  half resv[GP][4];
  float gate[GP][4];
#pragma unroll
  for (int g = 0; g < GP; ++g) {
    if (G == 0 && g >= groups) break;
    if (row >= rows_of(g)) continue;
    const int gr = 8 * g + row;
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      resv[g][b] = a.res[gr * KD + b * HD + col];
      gate[g][b] = 2.0f * sigm(__half2float(a.inj[gr * 4 + b]) / 4.0f);
    }
  }
  // ---- weight prefetch (independent of every exchange)
  uint4 dlo[4], dhi[4], ulo[4], uhi[4];
  if (warp < 6) {
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const uint4* p = a.wd + ((i * 6 + warp) * 4 + s) * 64;
      dlo[s] = ldgv(p + lane);
      dhi[s] = ldgv(p + 32 + lane);
    }
  }
  if (!LOCAL_SCHEDULE && warp >= 3) {
    const char* p =
        reinterpret_cast<const char*>(a.wu + (i * 5 + warp - 3) * 4 * 64);
    pf_l2(p + lane * 128);
  }
  // ---- all-reduce completion for every group.
  if (FULL) {
#pragma unroll
    for (int g = 0; g < GP; ++g) {
      if (G == 0 && g >= groups) break;
      if (row >= rows_of(g)) continue;
      if constexpr (LOCAL_SCHEDULE) {
        unsigned u1, u2, u3, e1, e2, e3;
        const auto start = clock64();
        do {
          asm volatile("ld.volatile.global.v2.u32 {%0,%1}, [%2];"
                       : "=r"(u1), "=r"(e1)
                       : "l"(ar(g, rk) + (p1 * NC + i) * 256 + t));
          asm volatile("ld.volatile.global.v2.u32 {%0,%1}, [%2];"
                       : "=r"(u2), "=r"(e2)
                       : "l"(ar(g, rk) + (p2 * NC + i) * 256 + t));
          asm volatile("ld.volatile.global.v2.u32 {%0,%1}, [%2];"
                       : "=r"(u3), "=r"(e3)
                       : "l"(ar(g, rk) + (p3 * NC + i) * 256 + t));
          if (clock64() - start > 4000000000LL) __trap();
        } while (e1 != tag || e2 != tag || e3 != tag);
        const float x1 = __uint_as_float(u1), x2 = __uint_as_float(u2),
                    x3 = __uint_as_float(u3);
        const float near = (rk & 1) ? x1 + v[g] : v[g] + x1;
        const float far = (rk & 1) ? x3 + x2 : x2 + x3;
        v[g] = (rk & 2) ? far + near : near + far;
      } else {
        float x[4];
#pragma unroll
        for (int s = 0; s < 4; ++s)
          x[s] = s == rk ? v[g]
                         : __uint_as_float(
                               poll2(ar(g, rk) + (s * NC + i) * 256 + t, tag));
        v[g] = (x[0] + x[1]) + (x[2] + x[3]);
      }
    }
  } else {
    const int slot = (NC + i) * 256 + t;
#pragma unroll
    for (int g = 0; g < GP; ++g) {
      if (G == 0 && g >= groups) break;
      if (row >= rows_of(g)) continue;
      v[g] = v[g] + __uint_as_float(poll2(ar(g, rk) + i * 256 + t, tag));
      st2(ar(g, p2) + slot, __float_as_uint(v[g]), tag);
    }
#pragma unroll
    for (int g = 0; g < GP; ++g) {
      if (G == 0 && g >= groups) break;
      if (row >= rows_of(g)) continue;
      v[g] = v[g] + __uint_as_float(poll2(ar(g, rk) + slot, tag));
    }
  }

  TS(2);
  // ---- 2./3. combine and squares for every group, then HC down partials.
  float o[GP][4];
#pragma unroll
  for (int g = 0; g < GP; ++g) {
    if (G == 0 && g >= groups) break;
    const int M = rows_of(g), gr = 8 * g + row;
    const bool act = row < M;
    const float B = __half2float(__float2half_rn(v[g]));
    float sums[4];
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      float q = 0.f;
      if (act) {
        const half h =
            __float2half_rn(__half2float(resv[g][b]) + B * gate[g][b]);
        o[g][b] = __half2float(h);
        a.res_out[gr * KD + b * HD + col] = h;
        smem.xs[g][row][b * 32 + lane] =
            __float2half_rn(o[g][b] + o[g][b] * __half2float(nwv[b]));
        q = o[g][b] * o[g][b];
      } else {
        o[g][b] = 0.f;
        smem.xs[g][row][b * 32 + lane] = __float2half_rn(0.f);
      }
      if constexpr (LOCAL_SCHEDULE) {
        sums[b] = q;
      } else {
#pragma unroll
        for (int s = 16; s > 0; s >>= 1) q += __shfl_xor_sync(0xffffffff, q, s);
        if (lane == 0 && act) sq(g)[(i * 8 + row) * 4 + b] = q;
      }
    }
    if constexpr (LOCAL_SCHEDULE) {
#pragma unroll
      for (int s = 16; s > 0; s >>= 1) {
#pragma unroll
        for (int b = 0; b < 4; ++b)
          sums[b] += __shfl_xor_sync(0xffffffff, sums[b], s);
      }
      if (lane == 0 && act)
        *reinterpret_cast<float4*>(sq(g) + (i * 8 + row) * 4) =
            make_float4(sums[0], sums[1], sums[2], sums[3]);
    }
  }
  TS(10);
  __syncthreads();
  if (LOCAL_SCHEDULE && warp >= 3) {
    const char* p =
        reinterpret_cast<const char*>(a.wu + (i * 5 + warp - 3) * 4 * 64);
    pf_l2(p + lane * 128);
  }
  if (warp < 6) {
    const int tile = warp >> 1, kh = warp & 1;
#pragma unroll
    for (int g = 0; g < GP; ++g) {
      if (G == 0 && g >= groups) break;
      const int M = rows_of(g);
      float acc0[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f},
            acc1[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const int k = (kh * 4 + s) * 16;
        const uint4 x0 = *reinterpret_cast<const uint4*>(&smem.xs[g][r8][k]);
        const uint4 x1 =
            *reinterpret_cast<const uint4*>(&smem.xs[g][r8][k + 8]);
        float (&acc)[8] = s < 2 ? acc0 : acc1;
        mma(acc, x0.x, x0.y, dlo[s].x, dlo[s].y);
        mma(acc, x0.z, x0.w, dlo[s].z, dlo[s].w);
        mma(acc, x1.x, x1.y, dhi[s].x, dhi[s].y);
        mma(acc, x1.z, x1.w, dhi[s].z, dhi[s].w);
      }
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int rw = (e & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
        const int n = tile * 32 + quad * 8 +
                      ((e & 1) | (((lane >> 1) & 1) << 1) | ((e >> 2) << 2));
        if (rw < M) {
          *reinterpret_cast<float2*>(dpart(g) + ((i * 8 + rw) * 96 + n) * 4 +
                                     2 * kh) = make_float2(acc0[e], acc1[e]);
        }
      }
      if (g == 0) TS(15);
    }
  }
  TS(3);
  const unsigned gb = (OF >= 0 && a.gz) ? NC : 0;
  if constexpr (LOCAL_SCHEDULE)
    grid_bar_epoch(a.bar + 2, base + NC);
  else
    grid_bar(a.bar, gb + NC);
  TS(4);

  // ---- 4. rrms and xn for every group, then the split-K + stream
  // reduction and lora push over every group's outputs.
  auto valid = [](int q, int n) { return n < 80 || (q == 3 && n < 84); };
  auto gcol = [](int q, int n) { return n < 80 ? q * 80 + n : 320 + n - 80; };
  float4 qsg[GP][3];
#pragma unroll
  for (int g = 0; g < GP; ++g)
#pragma unroll
    for (int j = 0; j < 3; ++j)
      qsg[g][j] = g < groups && row < rows_of(g) && lane + 32 * j < NC
                      ? __ldcg(reinterpret_cast<const float4*>(sq(g)) +
                               (lane + 32 * j) * 8 + row)
                      : make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
  for (int g = 0; g < GP; ++g) {
    if (G == 0 && g >= groups) break;
    if (row >= rows_of(g)) continue;
    const int gr = 8 * g + row;
    const float4* qs = qsg[g];
    float sums[4];
    if constexpr (LOCAL_SCHEDULE) {
      sums[0] = (qs[0].x + qs[1].x) + qs[2].x;
      sums[1] = (qs[0].y + qs[1].y) + qs[2].y;
      sums[2] = (qs[0].z + qs[1].z) + qs[2].z;
      sums[3] = (qs[0].w + qs[1].w) + qs[2].w;
#pragma unroll
      for (int s = 16; s > 0; s >>= 1) {
#pragma unroll
        for (int b = 0; b < 4; ++b)
          sums[b] += __shfl_xor_sync(0xffffffff, sums[b], s);
      }
    }
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      float q;
      if constexpr (LOCAL_SCHEDULE) {
        q = sums[b];
      } else {
        auto comp = [&](const float4& f) {
          return b == 0 ? f.x : b == 1 ? f.y : b == 2 ? f.z : f.w;
        };
        q = (comp(qs[0]) + comp(qs[1])) + comp(qs[2]);
#pragma unroll
        for (int s = 16; s > 0; s >>= 1) q += __shfl_xor_sync(0xffffffff, q, s);
      }
      const float rrms = rsqrtf(q / HD + a.eps);
      if (lane == 0) rr[g][row][b] = rrms;
      float y = o[g][b] * rrms;
      y += y * __half2float(nwv[b]);
      a.xn[gr * KD + b * HD + col] = __float2half_rn(y);
    }
  }
  __syncthreads();
  TS(11);
  const int outputs = groups * 8 * 96;
  for (int oi0 = i + NC * warp; oi0 < outputs; oi0 += 2 * NC * 8) {
    // Both outputs' partial loads are issued before the first remote store
    // (whose memory clobber would otherwise serialize the next loads).
    float4 pvs[2][3];
    bool live[2];
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int oi = oi0 + u * NC * 8;
      const int gr = oi / 96, n = oi % 96, g = gr / 8, rw = gr % 8;
      live[u] = oi < outputs && rw < rows_of(g) && valid(rk, n);
#pragma unroll
      for (int j = 0; j < 3; ++j)
        pvs[u][j] = live[u] && lane + 32 * j < NC
                        ? __ldcg(reinterpret_cast<const float4*>(dpart(g)) +
                                 ((lane + 32 * j) * 8 + rw) * 96 + n)
                        : make_float4(0.f, 0.f, 0.f, 0.f);
    }
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      if (!live[u]) continue;
      const int oi = oi0 + u * NC * 8;
      const int gr = oi / 96, n = oi % 96, g = gr / 8, rw = gr % 8;
      float pb[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int j = 0; j < 3; ++j)
        if (lane + 32 * j < NC) {
          const float4 pv = pvs[u][j];
          pb[0] += pv.x;
          pb[1] += pv.y;
          pb[2] += pv.z;
          pb[3] += pv.w;
        }
#pragma unroll
      for (int s = 16; s > 0; s >>= 1)
#pragma unroll
        for (int b = 0; b < 4; ++b)
          pb[b] += __shfl_xor_sync(0xffffffff, pb[b], s);
      float q = 0.f;
#pragma unroll
      for (int b = 0; b < 4; ++b) q = fmaf(rr[g][rw][b], pb[b], q);
      half hv = __float2half_rn(q);
      if (n < 80) {
        const float xx = divf(__half2float(hv), 4.0f);
        hv = __float2half_rn(xx * sigm(xx));
      }
      const int pos = rw * LW + gcol(rk, n);
      const unsigned w = __half_as_ushort(hv);
      if (lane == 0) st2(lora(g, rk) + pos, w, tag);
      if (lane == 1) st2(lora(g, p1) + pos, w, tag);
      if (lane == 2) st2(lora(g, p2) + pos, w, tag);
      if (FULL && lane == 3) st2(lora(g, p3) + pos, w, tag);
    }
  }
  if (warp >= 3) {
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const uint4* p = a.wu + ((i * 5 + warp - 3) * 4 + s) * 64;
      ulo[s] = ldgv(p + lane);
      uhi[s] = ldgv(p + 32 + lane);
    }
  }
  TS(5);
  // ---- 5. collect every group's lora (+ injection) share, one barrier.
#pragma unroll
  for (int g = 0; g < GP; ++g) {
    if (G == 0 && g >= groups) break;
    const int M = rows_of(g);
    for (int idx = i * 256 + t; idx < M * 324; idx += NC * 256) {
      const int pos = (idx / 324) * LW + idx % 324, c = idx % 324;
      const unsigned w = poll2(lora(g, rk) + pos, tag);
      if (!FULL &&
          ((c >= 80 * p1 && c < 80 * p1 + 80) || (p1 == 3 && c >= 320)))
        st2(lora(g, p2) + pos, w, tag);
    }
  }
  TS(6);
  if constexpr (LOCAL_SCHEDULE)
    grid_bar_epoch(a.bar + 2, base + 2 * NC);
  else
    grid_bar(a.bar, gb + 2 * NC);
  TS(7);
  // ---- 6. HC up and gate-mix per group from single-group shared lora rows
  // (< 32 KB total static shared memory), pipelining hidden-slice pushes.
  if (i == 0 && t < MT * 4) {
    const int gr = t >> 2, g = gr / 8, rw = gr % 8;
    a.inj_out[t] = __ushort_as_half(static_cast<unsigned short>(
        __ldcg(lora(g, rk) + rw * LW + 320 + (t & 3)).x));
  }
  TS(14);
  const int me = warp;
  const int mrw = (me & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
  const int mhh = (me & 1) | (((lane >> 1) & 1) << 1) | ((me >> 2) << 2);
  const int mh = 640 * rk + 8 * i + mhh;
  const int lhh = lane & 7, lstream = lane >> 3;
  const int lh = 640 * rk + 8 * i + lhh;
  float xvp[GP];
#pragma unroll
  for (int g = 0; g < GP; ++g) {
    if constexpr (LOCAL_SCHEDULE) {
      xvp[g] = (G > 0 || g < groups) && warp < rows_of(g)
                   ? __half2float(
                         __ldcg(a.xn + (8 * g + warp) * KD + lstream * HD + lh))
                   : 0.f;
    } else {
      xvp[g] =
          (G > 0 || g < groups) && mrw < rows_of(g)
              ? __half2float(__ldcg(a.xn + (8 * g + mrw) * KD + quad * HD + mh))
              : 0.f;
    }
  }
  TS(12);
#pragma unroll
  for (int g = 0; g < GP; ++g) {
    if (G == 0 && g >= groups) break;
    const int M = rows_of(g);
    {
      // Five 16-byte loads per thread cover all 8 * 160 = 1280 lora pairs of
      // group g; all global loads issue before writing smem.up.ls.
      uint4 w[5];
#pragma unroll
      for (int u = 0; u < 5; ++u) {
        const int idx = u * 256 + t;
        const int rw = idx / 160, c = (idx % 160) * 2;
        w[u] = rw < M ? __ldcg(reinterpret_cast<const uint4*>(lora(g, rk) +
                                                              rw * LW + c))
                      : make_uint4(0, 0, 0, 0);
      }
#pragma unroll
      for (int u = 0; u < 5; ++u) {
        const int idx = u * 256 + t;
        const int rw = idx / 160, c = (idx % 160) * 2;
        *reinterpret_cast<unsigned*>(&smem.up.ls[rw][c]) =
            (w[u].x & 0xffffu) | (w[u].z << 16);
      }
    }
    __syncthreads();
    if (warp >= 3) {
      const int uw = warp - 3;
      float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const int k = (uw * 4 + s) * 16;
        const uint4 x0 = *reinterpret_cast<const uint4*>(&smem.up.ls[r8][k]);
        const uint4 x1 =
            *reinterpret_cast<const uint4*>(&smem.up.ls[r8][k + 8]);
        mma(acc, x0.x, x0.y, ulo[s].x, ulo[s].y);
        mma(acc, x0.z, x0.w, ulo[s].z, ulo[s].w);
        mma(acc, x1.x, x1.y, uhi[s].x, uhi[s].y);
        mma(acc, x1.z, x1.w, uhi[s].z, uhi[s].w);
      }
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        if constexpr (LOCAL_SCHEDULE) {
          const int rw = (e & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
          const int hh = (e & 1) | (lane & 2) | (e & 4);
          smem.up.red[uw][rw][quad * 8 + hh] = acc[e];
        } else {
          smem.up.red[uw][e][lane] = acc[e];
        }
      }
    }
    __syncthreads();
    if constexpr (LOCAL_SCHEDULE) {
      if (warp < M) {
        float acc = smem.up.red[0][warp][lane];
#pragma unroll
        for (int q = 1; q < 5; ++q) acc += smem.up.red[q][warp][lane];
        const float sg = sigm(__half2float(__float2half_rn(acc)));
        const float xv = xvp[g];
        float mixed = 0.f;
#pragma unroll
        for (int b = 0; b < 4; ++b)
          mixed = fmaf(__shfl_sync(0xffffffff, sg, b * 8 + lhh),
                       __shfl_sync(0xffffffff, xv, b * 8 + lhh), mixed);
        const unsigned bits =
            __half_as_ushort(__float2half_rn(divf(mixed, 4.0f)));
        const unsigned packed =
            bits | (__shfl_down_sync(0xffffffff, bits, 1) << 16);
        const unsigned upper = __shfl_down_sync(0xffffffff, packed, 2);
        if (lane < 8 && !(lane & 3)) {
          st4(hb(g, p1) + (warp * HD + lh) / 2, packed, upper, tag);
          st4(hb(g, p2) + (warp * HD + lh) / 2, packed, upper, tag);
          if (FULL) st4(hb(g, p3) + (warp * HD + lh) / 2, packed, upper, tag);
          *reinterpret_cast<uint2*>(a.blk_out + (8 * g + warp) * HD + lh) =
              make_uint2(packed, upper);
        }
      }
    } else {
      float acc = smem.up.red[0][me][lane];
#pragma unroll
      for (int q = 1; q < 5; ++q) acc += smem.up.red[q][me][lane];
      const float sg = sigm(__half2float(__float2half_rn(acc)));
      const float xv = xvp[g];
      float mixed = 0.f;
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        const int src = (lane & ~12) | (b << 2);
        mixed = fmaf(__shfl_sync(0xffffffff, sg, src),
                     __shfl_sync(0xffffffff, xv, src), mixed);
      }
      if (quad == 0) tout[g][mrw][mhh] = __float2half_rn(divf(mixed, 4.0f));
    }
  }
  TS(13);
  if constexpr (!LOCAL_SCHEDULE) {
    __syncthreads();
    if (t < 2 * MT) {
      const int gr = t >> 1, g = gr / 8, rw = gr % 8, h4 = (t & 1) * 4,
                h = 640 * rk + 8 * i + h4;
      const uint2 d = *reinterpret_cast<const uint2*>(&tout[g][rw][h4]);
      st4(hb(g, p1) + (rw * HD + h) / 2, d.x, d.y, tag);
      st4(hb(g, p2) + (rw * HD + h) / 2, d.x, d.y, tag);
      if (FULL) st4(hb(g, p3) + (rw * HD + h) / 2, d.x, d.y, tag);
      *reinterpret_cast<uint2*>(a.blk_out + gr * HD + h) = d;
    }
  }
  TS(8);
  // ---- 7. collect every group's peer hidden slices.
  if (FULL) {
    // Every group at once: thread t owns word t % 6 of row t / 6 (all rows).
    if (t < 6 * MT) {
      const int gr = t / 6, w = t % 6, k = w % 3;
      const int src = k == 0 ? p1 : k == 1 ? p2 : p3;
      const int g = gr / 8, rw = gr % 8;
      const int h = 640 * src + 8 * i + (w / 3) * 4;
      const uint2 d = poll4(hb(g, rk) + (rw * HD + h) / 2, tag);
      *reinterpret_cast<uint2*>(a.blk_out + gr * HD + h) = d;
    }
  } else {
#pragma unroll
    for (int g = 0; g < GP; ++g) {
      if (G == 0 && g >= groups) break;
      const int M = rows_of(g), r0 = 8 * g;
      if (t >= 32 && t < 32 + 2 * M) {
        const int u = t - 32, rw = u >> 1, h = 640 * p1 + 8 * i + (u & 1) * 4;
        const uint2 d = poll4(hb(g, rk) + (rw * HD + h) / 2, tag);
        st4(hb(g, p2) + (rw * HD + h) / 2, d.x, d.y, tag);
        *reinterpret_cast<uint2*>(a.blk_out + (r0 + rw) * HD + h) = d;
      }
      if (t >= 64 && t < 64 + 4 * M) {
        const int u = t - 64, src = (u & 2) ? p3 : p2, rw = u >> 2,
                  h = 640 * src + 8 * i + (u & 1) * 4;
        const uint2 d = poll4(hb(g, rk) + (rw * HD + h) / 2, tag);
        *reinterpret_cast<uint2*>(a.blk_out + (r0 + rw) * HD + h) = d;
      }
    }
  }
  TS(9);
  if constexpr (LOCAL_SCHEDULE) {
    if (i == 0 && t == 0) {
      a.bar[3] = base + 2 * NC;
      a.seq[0] = tag;
    }
  } else {
    __syncthreads();
    if (t == 0) {
      __threadfence();
      if (atomicAdd(a.bar + 1, 1u) == NC - 1) {
        a.bar[0] = 0;
        a.bar[1] = 0;
        __threadfence();
        atomicExch(a.seq, ep + 1);
      }
    }
  }
}
