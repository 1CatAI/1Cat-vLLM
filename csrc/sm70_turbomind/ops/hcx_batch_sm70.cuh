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

template <bool FULL, int OF>
__global__ __launch_bounds__(256, 1) void hcx_batch_kernel(Args a) {
  TS(0);
  const int i = blockIdx.x, t = threadIdx.x, warp = t >> 5, lane = t & 31;
  const int MT = a.M, rk = a.rank, p1 = rk ^ 1, p2 = rk ^ 2, p3 = rk ^ 3;
  const unsigned ep = ldv(a.seq), tag = ep + 1;
  const int r8 = (lane & 3) + ((lane & 16) ? 4 : 0), quad = (lane >> 2) & 3;
  const int groups = (MT + 7) / 8;
  const int row = warp, col = 32 * i + lane;
  auto rows_of = [&](int g) { return min(8, MT - 8 * g); };
  auto ar = [&](int g, int q) { return a.ar[q] + g * AR_WORDS; };
  auto lora = [&](int g, int q) { return a.lora[q] + g * LORA_WORDS; };
  auto hb = [&](int g, int q) { return a.hb[q] + g * HB_WORDS; };
  auto sq = [&](int g) { return a.sq + g * NC * 8 * 4; };
  auto dpart = [&](int g) { return a.dpart + g * NC * 8 * 96 * 4; };

  __shared__ __align__(16) half xs[GMAX][8][128 + 8];
  __shared__ __align__(16) half ls[GMAX][8][320 + 8];
  __shared__ float red[6][8][32];
  __shared__ __align__(16) half tout[GMAX][8][8];
  __shared__ float rr[GMAX][8][4];

  float v[GMAX];
#pragma unroll
  for (int g = 0; g < GMAX; ++g) v[g] = 0.f;

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
    for (int g = 0; g < GMAX; ++g) {
      if (g >= groups) break;
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
  for (int g = 0; g < GMAX; ++g) {
    if (g >= groups) break;
    if (row >= rows_of(g)) continue;
    const int gr = 8 * g + row;
    if constexpr (OF < 0) v[g] = __half2float(a.p0[gr * HD + col]);
    if (a.p1) v[g] += __half2float(a.p1[gr * HD + col]);
  }
#pragma unroll
  for (int g = 0; g < GMAX; ++g) {
    if (g >= groups) break;
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
  half resv[GMAX][4];
  float gate[GMAX][4];
#pragma unroll
  for (int g = 0; g < GMAX; ++g) {
    if (g >= groups) break;
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
  if (warp >= 3) {
    const char* p =
        reinterpret_cast<const char*>(a.wu + (i * 5 + warp - 3) * 4 * 64);
    pf_l2(p + lane * 128);
  }
  // ---- all-reduce completion for every group.
  if (FULL) {
#pragma unroll
    for (int g = 0; g < GMAX; ++g) {
      if (g >= groups) break;
      if (row >= rows_of(g)) continue;
      float x[4];
#pragma unroll
      for (int s = 0; s < 4; ++s)
        x[s] = s == rk ? v[g]
                       : __uint_as_float(
                             poll2(ar(g, rk) + (s * NC + i) * 256 + t, tag));
      v[g] = (x[0] + x[1]) + (x[2] + x[3]);
    }
  } else {
    const int slot = (NC + i) * 256 + t;
#pragma unroll
    for (int g = 0; g < GMAX; ++g) {
      if (g >= groups) break;
      if (row >= rows_of(g)) continue;
      v[g] = v[g] + __uint_as_float(poll2(ar(g, rk) + i * 256 + t, tag));
      st2(ar(g, p2) + slot, __float_as_uint(v[g]), tag);
    }
#pragma unroll
    for (int g = 0; g < GMAX; ++g) {
      if (g >= groups) break;
      if (row >= rows_of(g)) continue;
      v[g] = v[g] + __uint_as_float(poll2(ar(g, rk) + slot, tag));
    }
  }

  TS(2);
  // ---- 2./3. combine and squares for every group, then HC down partials.
  float o[GMAX][4];
#pragma unroll
  for (int g = 0; g < GMAX; ++g) {
    if (g >= groups) break;
    const int M = rows_of(g), gr = 8 * g + row;
    const bool act = row < M;
    const float B = __half2float(__float2half_rn(v[g]));
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      float q = 0.f;
      if (act) {
        const half h =
            __float2half_rn(__half2float(resv[g][b]) + B * gate[g][b]);
        o[g][b] = __half2float(h);
        a.res_out[gr * KD + b * HD + col] = h;
        xs[g][row][b * 32 + lane] =
            __float2half_rn(o[g][b] + o[g][b] * __half2float(nwv[b]));
        q = o[g][b] * o[g][b];
      } else {
        o[g][b] = 0.f;
        xs[g][row][b * 32 + lane] = __float2half_rn(0.f);
      }
#pragma unroll
      for (int s = 16; s > 0; s >>= 1) q += __shfl_xor_sync(0xffffffff, q, s);
      if (lane == 0 && act) sq(g)[(i * 8 + row) * 4 + b] = q;
    }
  }
  __syncthreads();
  if (warp < 6) {
    const int tile = warp >> 1, kh = warp & 1;
#pragma unroll
    for (int g = 0; g < GMAX; ++g) {
      if (g >= groups) break;
      const int M = rows_of(g);
      float acc0[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f},
            acc1[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const int k = (kh * 4 + s) * 16;
        const uint4 x0 = *reinterpret_cast<const uint4*>(&xs[g][r8][k]);
        const uint4 x1 = *reinterpret_cast<const uint4*>(&xs[g][r8][k + 8]);
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
    }
  }
  TS(3);
  const unsigned gb = (OF >= 0 && a.gz) ? NC : 0;
  grid_bar(a.bar, gb + NC);
  TS(4);

  // ---- 4. rrms and xn for every group, then the split-K + stream
  // reduction and lora push over every group's outputs.
  auto valid = [](int q, int n) { return n < 80 || (q == 3 && n < 84); };
  auto gcol = [](int q, int n) { return n < 80 ? q * 80 + n : 320 + n - 80; };
#pragma unroll
  for (int g = 0; g < GMAX; ++g) {
    if (g >= groups) break;
    if (row >= rows_of(g)) continue;
    const int gr = 8 * g + row;
    float4 qs[3];
#pragma unroll
    for (int j = 0; j < 3; ++j)
      qs[j] = lane + 32 * j < NC
                  ? __ldcg(reinterpret_cast<const float4*>(sq(g)) +
                           (lane + 32 * j) * 8 + row)
                  : make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      auto comp = [&](const float4& f) {
        return b == 0 ? f.x : b == 1 ? f.y : b == 2 ? f.z : f.w;
      };
      float q = (comp(qs[0]) + comp(qs[1])) + comp(qs[2]);
#pragma unroll
      for (int s = 16; s > 0; s >>= 1) q += __shfl_xor_sync(0xffffffff, q, s);
      const float rrms = rsqrtf(q / HD + a.eps);
      if (lane == 0) rr[g][row][b] = rrms;
      float y = o[g][b] * rrms;
      y += y * __half2float(nwv[b]);
      a.xn[gr * KD + b * HD + col] = __float2half_rn(y);
    }
  }
  __syncthreads();
  for (int oi = i + NC * warp; oi < groups * 8 * 96; oi += NC * 8) {
    const int gr = oi / 96, n = oi % 96, g = gr / 8, rw = gr % 8;
    if (rw >= rows_of(g) || !valid(rk, n)) continue;
    float pb[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int j = 0; j < 3; ++j)
      if (lane + 32 * j < NC) {
        const float4 pv = __ldcg(reinterpret_cast<const float4*>(dpart(g)) +
                                 ((lane + 32 * j) * 8 + rw) * 96 + n);
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
  for (int g = 0; g < GMAX; ++g) {
    if (g >= groups) break;
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
  grid_bar(a.bar, gb + 2 * NC);
  TS(7);
  // ---- 6. HC up and gate-mix per group from shared lora rows, then push
  // every group's hidden slice.
  for (int idx = t; idx < groups * 8 * 320; idx += 256) {
    const int gr = idx / 320, c = idx % 320, g = gr / 8, rw = gr % 8;
    ls[g][rw][c] = rw < rows_of(g)
                       ? __ushort_as_half(static_cast<unsigned short>(
                             __ldcg(lora(g, rk) + rw * LW + c).x))
                       : __float2half_rn(0.f);
  }
  if (i == 0 && t < MT * 4) {
    const int gr = t >> 2, g = gr / 8, rw = gr % 8;
    a.inj_out[t] = __ushort_as_half(static_cast<unsigned short>(
        __ldcg(lora(g, rk) + rw * LW + 320 + (t & 3)).x));
  }
  __syncthreads();
#pragma unroll
  for (int g = 0; g < GMAX; ++g) {
    if (g >= groups) break;
    const int M = rows_of(g), r0 = 8 * g;
    if (warp >= 3) {
      const int uw = warp - 3;
      float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const int k = (uw * 4 + s) * 16;
        const uint4 x0 = *reinterpret_cast<const uint4*>(&ls[g][r8][k]);
        const uint4 x1 = *reinterpret_cast<const uint4*>(&ls[g][r8][k + 8]);
        mma(acc, x0.x, x0.y, ulo[s].x, ulo[s].y);
        mma(acc, x0.z, x0.w, ulo[s].z, ulo[s].w);
        mma(acc, x1.x, x1.y, uhi[s].x, uhi[s].y);
        mma(acc, x1.z, x1.w, uhi[s].z, uhi[s].w);
      }
#pragma unroll
      for (int e = 0; e < 8; ++e) red[uw][e][lane] = acc[e];
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        float acc = red[0][e][lane];
#pragma unroll
        for (int q = 1; q < 5; ++q) acc += red[q][e][lane];
        const int rw = (e & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
        const int hh = (e & 1) | (((lane >> 1) & 1) << 1) | ((e >> 2) << 2);
        const int h = 640 * rk + 8 * i + hh;
        const float sg = sigm(__half2float(__float2half_rn(acc)));
        const float xv =
            rw < M ? __half2float(__ldcg(a.xn + (r0 + rw) * KD + quad * HD + h))
                   : 0.f;
        float mixed = 0.f;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          const int src = (lane & ~12) | (b << 2);
          mixed = fmaf(__shfl_sync(0xffffffff, sg, src),
                       __shfl_sync(0xffffffff, xv, src), mixed);
        }
        if (quad == 0) tout[g][rw][hh] = __float2half_rn(divf(mixed, 4.0f));
      }
    }
    __syncthreads();
  }
  if (t < 2 * MT) {
    const int gr = t >> 1, g = gr / 8, rw = gr % 8, h4 = (t & 1) * 4,
              h = 640 * rk + 8 * i + h4;
    const uint2 d = *reinterpret_cast<const uint2*>(&tout[g][rw][h4]);
    st4(hb(g, p1) + (rw * HD + h) / 2, d.x, d.y, tag);
    st4(hb(g, p2) + (rw * HD + h) / 2, d.x, d.y, tag);
    if (FULL) st4(hb(g, p3) + (rw * HD + h) / 2, d.x, d.y, tag);
    *reinterpret_cast<uint2*>(a.blk_out + gr * HD + h) = d;
  }
  TS(8);
  // ---- 7. collect every group's peer hidden slices.
#pragma unroll
  for (int g = 0; g < GMAX; ++g) {
    if (g >= groups) break;
    const int M = rows_of(g), r0 = 8 * g;
    if (FULL) {
      if (t >= 32 && t < 32 + 6 * M) {
        const int u = t - 32, k = u % 3,
                  src = k == 0   ? p1
                        : k == 1 ? p2
                                 : p3,
                  rw = (u / 3) >> 1;
        const int h = 640 * src + 8 * i + ((u / 3) & 1) * 4;
        const uint2 d = poll4(hb(g, rk) + (rw * HD + h) / 2, tag);
        *reinterpret_cast<uint2*>(a.blk_out + (r0 + rw) * HD + h) = d;
      }
    } else {
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
  __syncthreads();
  TS(9);
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
