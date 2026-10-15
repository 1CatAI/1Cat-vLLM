// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Included in namespace hcx, after the retained HCX primitives. Full-mesh
// TP4, separate output projection, local schedule, M9..40 only. One resident
// grid performs the exchange and both grid rendezvous for the whole batch.
constexpr int BATCH_ROWS = 40;

__global__ __launch_bounds__(256, 1) void hcx_batch_kernel(Args a) {
  const int i = blockIdx.x, t = threadIdx.x, warp = t >> 5, lane = t & 31;
  const int rk = a.rank, p1 = rk ^ 1, p2 = rk ^ 2, p3 = rk ^ 3;
  const unsigned tag = ldv(a.seq) + 1, base = ldv(a.bar + 3);
  const int r8 = (lane & 3) + ((lane & 16) ? 4 : 0), quad = (lane >> 2) & 3;
  constexpr int AR_STRIDE = BATCH_ROWS * 32;
  __shared__ __align__(16) half xs[BATCH_ROWS][136];
  __shared__ __align__(16) half ls[BATCH_ROWS][328];
  __shared__ float red[5][8][33];
  __shared__ float rr[BATCH_ROWS][4];

  for (int row = warp; row < a.M; row += 8) {
    const int col = 32 * i + lane;
    float v = __half2float(a.p0[row * HD + col]);
    if (a.p1) v += __half2float(a.p1[row * HD + col]);
    const int slot = (rk * NC + i) * AR_STRIDE + row * 32 + lane;
    st2(a.ar[p1] + slot, __float_as_uint(v), tag);
    st2(a.ar[p2] + slot, __float_as_uint(v), tag);
    st2(a.ar[p3] + slot, __float_as_uint(v), tag);
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
  if (warp >= 3) {
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const uint4* p = a.wu + ((i * 5 + warp - 3) * 4 + s) * 64;
      ulo[s] = ldgv(p + lane);
      uhi[s] = ldgv(p + 32 + lane);
    }
  }
  for (int row = warp; row < BATCH_ROWS; row += 8) {
    const int col = 32 * i + lane;
    float sums[4] = {};
#pragma unroll
    for (int b = 0; b < 4; ++b) xs[row][b * 32 + lane] = __float2half_rn(0.f);
    if (row < a.M) {
      float v = __half2float(a.p0[row * HD + col]);
      if (a.p1) v += __half2float(a.p1[row * HD + col]);
      const int slot = i * AR_STRIDE + row * 32 + lane;
      const float x1 =
          __uint_as_float(poll2(a.ar[rk] + p1 * NC * AR_STRIDE + slot, tag));
      const float x2 =
          __uint_as_float(poll2(a.ar[rk] + p2 * NC * AR_STRIDE + slot, tag));
      const float x3 =
          __uint_as_float(poll2(a.ar[rk] + p3 * NC * AR_STRIDE + slot, tag));
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
  if (warp < 6) {
    const int tile = warp >> 1, kh = warp & 1;
#pragma unroll
    for (int token_tile = 0; token_tile < 5; ++token_tile) {
      if (token_tile * 8 >= a.M) break;
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
        if (rw < a.M)
          *reinterpret_cast<float2*>(
              a.dpart + ((i * BATCH_ROWS + rw) * 96 + n) * 4 + 2 * kh) =
              make_float2(acc0[e], acc1[e]);
      }
    }
  }
  grid_bar_epoch(a.bar + 2, base + NC);
  for (int row = warp; row < a.M; row += 8) {
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
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      const float inverse = rsqrtf(sums[b] / HD + a.eps);
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
  for (int oi = i + NC * warp; oi < a.M * 96; oi += NC * 8) {
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
  for (int idx = i * 256 + t; idx < a.M * 324; idx += NC * 256)
    poll2(a.lora[rk] + (idx / 324) * LW + idx % 324, tag);
  grid_bar_epoch(a.bar + 2, base + 2 * NC);
  for (int idx = t; idx < BATCH_ROWS * 320; idx += 256)
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
    if (warp >= 3) {
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
    if (row < a.M) {
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
  for (int u = t; u < 6 * a.M; u += 256) {
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
