// Extracted verbatim from v41k.cu (base 8a7bd7b) by extract_old.py;
// the marked EDIT lines are the only changes.
namespace v41k {
template <int Type>
constexpr int block_bytes() {
  return Type == 16 ? 66 : Type == 17 ? 74 : 56;
}

template <int Type>
__global__ void build_table(uint32_t* out, int words) {
  using Book = turbomind::gemm::LatticeCodebook<Type>;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= words) return;
  uint32_t word = Book::word(i) ^ 0x80808080U;
  if constexpr (Type == 29) {
    word = __vadd4(word, word);
    word = __vadd4(word, word);
    word = __vadd4(word, word);
  }
  out[i] = word;
}
__device__ __forceinline__ void ld3(const uint8_t* p, uint32_t& w0,
                                    uint32_t& w1) {
  // 8 bytes at any 2-byte-aligned address: three aligned words + funnel shifts.
  const uintptr_t a = reinterpret_cast<uintptr_t>(p);
  const uint32_t* q = reinterpret_cast<const uint32_t*>(a & ~uintptr_t(3));
  const uint32_t sh = (a & 3) * 8;
  const uint32_t x0 = __ldg(q), x1 = __ldg(q + 1), x2 = __ldg(q + 2);
  w0 = __funnelshift_r(x0, x1, sh);
  w1 = __funnelshift_r(x1, x2, sh);
}
__device__ __forceinline__ uint32_t ld16(const uint8_t* p) {
  return *reinterpret_cast<const uint16_t*>(p);
}

template <int Type>
// Codebook entries as one 64-bit LDS; sign masks computed (PRMT), not looked
// up.
#ifndef V41K_SMEM_MASKS
  #define V41K_MASK(n) sign_mask(n)
#else
  #define V41K_MASK(n) masks[n]
#endif
struct Group3 {
  uint32_t w[8];
  float d;
  int m0, m1;
  __device__ __forceinline__ Group3(const uint8_t* row, int group,
                                    const uint32_t* book,
                                    const uint32_t* masks) {
    const uint8_t* b = row + (group >> 3) * block_bytes<Type>();
    const int sub = group & 7;
    if constexpr (Type == 29) {
      const uint32_t qs = __ldg(reinterpret_cast<const uint32_t*>(b + 4 * sub));
      const uint32_t qh = ld16(b + 32 + 2 * sub);
      const uint2 sc = __ldg(reinterpret_cast<const uint2*>(b + 48));
#pragma unroll
      for (int octet = 0; octet < 4; ++octet) {
        const int high = (qh >> (4 * octet)) & 15;
        const int index = ((qs >> (8 * octet)) & 255) | ((high & 7) << 8);
        const uint32_t bias = (high >> 3) ? 0xffffffffU : 0x01010101U;
        const uint2 e = reinterpret_cast<const uint2*>(book)[index];
        w[2 * octet] = __vadd4(e.x, bias);
        w[2 * octet + 1] = __vadd4(e.y, bias);
      }
      const uint32_t s01 = sc.x, s23 = sc.y;
      const uint16_t bits =
          ((s01 & 0xffff) >> 12) | (((s01 >> 16) >> 8) & 0xf0) |
          (((s23 & 0xffff) >> 4) & 0xf00) | ((s23 >> 16) & 0xf000);
      d = __half2float(__ushort_as_half(bits));
      const uint64_t all = (uint64_t(s23) << 32) | s01;
      const int f = sub * 2;
      // scales[f/4] >> 3*(f%4) on 16-bit lanes
      m0 = 2 * ((all >> (16 * (f / 4) + 3 * (f % 4))) & 7) + 1;
      m1 = 2 * ((all >> (16 * ((f + 1) / 4) + 3 * ((f + 1) % 4))) & 7) + 1;
    } else {
      uint32_t w0, w1;
      ld3(b + 2 + 8 * sub, w0, w1);
      d = __half2float(__ushort_as_half(ld16(b)));
      if constexpr (Type == 16) {
        const uint32_t meta = w1;
#pragma unroll
        for (int octet = 0; octet < 4; ++octet) {
          const int index = (w0 >> (8 * octet)) & 255;
          int signs = (meta >> (7 * octet)) & 127;
          signs |= (__popc(signs) & 1) << 7;
          const uint32_t lo = V41K_MASK(signs & 15), hi = V41K_MASK(signs >> 4);
          const uint2 e = reinterpret_cast<const uint2*>(book)[index];
          w[2 * octet] = __vsub4(e.x ^ lo, lo);
          w[2 * octet + 1] = __vsub4(e.y ^ hi, hi);
        }
        m0 = m1 = 1 + 2 * (meta >> 28);
      } else {
#pragma unroll
        for (int octet = 0; octet < 4; ++octet) {
          const uint32_t code =
              ((octet < 2 ? w0 : w1) >> (16 * (octet & 1))) & 0xffff;
          const int index = code & 511;
          int signs = code >> 9;
          signs |= (__popc(signs) & 1) << 7;
          const uint32_t lo = V41K_MASK(signs & 15), hi = V41K_MASK(signs >> 4);
          const uint2 e = reinterpret_cast<const uint2*>(book)[index];
          w[2 * octet] = __vsub4(e.x ^ lo, lo);
          w[2 * octet + 1] = __vsub4(e.y ^ hi, hi);
        }
        const int sc = b[66 + sub];
        m0 = 1 + 2 * (sc & 15);
        m1 = 1 + 2 * (sc >> 4);
      }
    }
  }
  __device__ __forceinline__ float apply(const int* qs, float xd) const {
    int s0 = 0, s1 = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) s0 = __dp4a(int(w[i]), qs[i], s0);
#pragma unroll
    for (int i = 4; i < 8; ++i) s1 = __dp4a(int(w[i]), qs[i], s1);
    return d * xd * float(s0 * m0 + s1 * m1);  // EDIT 1: the x0.125 is the FFMA below (old SASS)
  }
};

template <int Type>
constexpr int book_words() {
  return turbomind::gemm::LatticeCodebook<Type>::kBytes / 4;
}
__device__ __forceinline__ int find_distinct(const int* ids, int routes,
                                             int u) {
  int e = -1, distinct = 0;
  for (int r = 0; r < routes && e < 0; ++r) {
    bool seen = false;
    for (int q = 0; q < r; ++q) seen |= ids[q] == ids[r];
    if (!seen) {
      if (distinct == u) e = ids[r];
      ++distinct;
    }
  }
  return e;
}
__device__ __forceinline__ void load_q8(const int8_t* qs, const float* ds,
                                        int64_t idx, int (&q)[8], float& d) {
  const int4* p = reinterpret_cast<const int4*>(qs + idx * 32);
  const int4 a = __ldg(p), b = __ldg(p + 1);
  q[0] = a.x;
  q[1] = a.y;
  q[2] = a.z;
  q[3] = a.w;
  q[4] = b.x;
  q[5] = b.y;
  q[6] = b.z;
  q[7] = b.w;
  d = __ldg(ds + idx);
}

__device__ __forceinline__ void q8_warp_aligned(int8_t* qs, float* ds,
                                                int64_t idx, float value) {
  const int lane = threadIdx.x % 32;
  float m = fabsf(value);
#pragma unroll
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffU, m, o));
  const float d = m / 127.f;
  qs[idx * 32 + lane] = m == 0.f ? 0 : int8_t(roundf(value / d));
  if (!lane) ds[idx] = __half2float(__float2half_rn(d));
}
#ifndef V41K_GU_MINB
  #define V41K_GU_MINB 2
#endif
template <int Type>
__global__ void __launch_bounds__(512, V41K_GU_MINB)
    gate_up_em5(int8_t* __restrict__ hq, float* __restrict__ hd,
                const int8_t* __restrict__ xq, const float* __restrict__ xd,
                const int* ids, const float* __restrict__ prob,
                const uint8_t* __restrict__ gate,
                const uint8_t* __restrict__ up, int row_stride,
                const uint32_t* __restrict__ gbook, int routes, int top_k,
                const int* __restrict__ plan) {
  constexpr int GROUPS = 160, BW = book_words<Type>();
  __shared__ __align__(16) uint32_t book[BW];
  __shared__ int s_expert, s_cnt, s_route[6], s_tok[6];
  __shared__ uint32_t masks[16];
  __shared__ float red[6][2][32];
  const int t = threadIdx.x, warp = t / 32, lane = t % 32;
  if (plan != nullptr) {
    // per-layer plan: [0] distinct count, [1+u] expert, [37+u] routes,
    // [73+6u+c] route
    const int u = blockIdx.y;
    if (u >= __ldg(plan)) return;
    for (int i = t; i < BW; i += blockDim.x) book[i] = __ldg(gbook + i);
    if (t < 16) masks[t] = sign_mask(t);
    if (t < 6) {
      const int cnt = __ldg(plan + 37 + u);
      const int r = t < cnt ? __ldg(plan + 73 + 6 * u + t) : 0;
      s_route[t] = r;
      s_tok[t] = r / top_k;
      if (t == 0) {
        s_expert = __ldg(plan + 1 + u);
        s_cnt = cnt;
      }
    }
  } else if (t == 0) {
    const int e = find_distinct(ids, routes, blockIdx.y);
    int cnt = 0;
    if (e >= 0)
      for (int r = 0; r < routes; ++r)
        if (ids[r] == e && cnt < 6) {
          s_route[cnt] = r;
          s_tok[cnt] = r / top_k;
          ++cnt;
        }
    s_expert = e;
    s_cnt = cnt;
  }
  __syncthreads();
  const int e = s_expert, cnt = s_cnt;
  if (plan == nullptr) {
    if (e < 0) return;
    for (int i = t; i < BW; i += blockDim.x) book[i] = __ldg(gbook + i);
    if (t < 16) masks[t] = sign_mask(t);
    __syncthreads();
  }
  const int lrow = warp * 2 + lane / 16;
  const int row = blockIdx.x * 32 + lrow;
  const uint8_t* g = gate + (int64_t(e) * 576 + row) * row_stride;
  const uint8_t* u = up + (int64_t(e) * 576 + row) * row_stride;
  int tok[6];
#pragma unroll
  for (int c = 0; c < 6; ++c) tok[c] = c < cnt ? s_tok[c] : 0;
  float gs[6] = {}, us[6] = {};
#pragma unroll kGuUnroll
  for (int group = lane % 16; group < GROUPS; group += 16) {
    const Group3<Type> dg(g, group, book, masks);
    const Group3<Type> du(u, group, book, masks);
#pragma unroll
    for (int c = 0; c < 6; ++c)
      if (c < cnt) {
        int q[8];
        float d;
        load_q8(xq, xd, int64_t(tok[c]) * GROUPS + group, q, d);
        gs[c] = __builtin_fmaf(dg.apply(q, d), 0.125f, gs[c]);  // EDIT 2: FFMA(x, 0.125, acc) as in the old SASS
        us[c] = __builtin_fmaf(du.apply(q, d), 0.125f, us[c]);  // EDIT 3
      }
  }
#pragma unroll
  for (int c = 0; c < 6; ++c) {
#pragma unroll
    for (int o = 8; o; o >>= 1) {
      gs[c] += __shfl_xor_sync(0xffffffffU, gs[c], o);
      us[c] += __shfl_xor_sync(0xffffffffU, us[c], o);
    }
    if (c < cnt && lane % 16 == 0) {
      red[c][0][lrow] = gs[c];
      red[c][1][lrow] = us[c];
    }
  }
  __syncthreads();
  if (warp < cnt) {
    const int route = s_route[warp];
    float g16 = __half2float(__float2half_rn(red[warp][0][lane]));
    float u16 = __half2float(__float2half_rn(red[warp][1][lane]));
    g16 = g16 > 10.f ? 10.f : g16;
    u16 = u16 > 10.f ? 10.f : (u16 < -10.f ? -10.f : u16);
    const float act = __fdiv_rn(g16, __fadd_rn(1.f, expf(-g16)));
    const float value = __half2float(
        __float2half_rn(__fmul_rn(__fmul_rn(act, u16), prob[route])));
    q8_warp_aligned(hq, hd, int64_t(route) * 18 + blockIdx.x, value);
  }
}
// Row-split down (G3): each rank owns 1280 output rows of every expert with the
// full K = 2304 (72 Q8 groups). The routed Q8 inputs of all four ranks are
// gathered rank-major: rank r's block holds hq [routes, 18, 32] then hd
// [routes, 18] at `rank_stride` bytes. Only rows [row_base, row_base + rows)
// of out are written; the caller zeroes the rest before the TP all-reduce.
template <int Type, int NW, int SUB>
__global__ void __launch_bounds__(32 * NW, 8 / NW * 4)
    down_rows(half* __restrict__ out, const uint8_t* __restrict__ gathered,
              int64_t rank_stride, const uint8_t* __restrict__ down,
              int row_stride, int rows_local, int row_base,
              const uint32_t* __restrict__ gbook, int routes, int top_k,
              int tokens, int rows_per_cta, const int* __restrict__ plan) {
  constexpr int LG = 18, GROUPS = 72, N = 5120, BW = book_words<Type>(),
                RT = 32 / SUB;
  __shared__ __align__(16) uint32_t book[BW];
  __shared__ int distinct_e[36], s_nu, member_route[36][6], member_cnt[36];
  __shared__ uint32_t masks[16];
  __shared__ float part[NW][8][6];
  const int t = threadIdx.x, lane = t % 32, warp = t / 32;
  for (int i = t; i < BW; i += blockDim.x) book[i] = __ldg(gbook + i);
  if (t < 16) masks[t] = sign_mask(t);
  for (int i = t; i < 36 + 36 + 216; i += blockDim.x) {
    if (i < 36)
      distinct_e[i] = __ldg(plan + 1 + i);
    else if (i < 72)
      member_cnt[i - 36] = __ldg(plan + 1 + i);
    else
      member_route[(i - 72) / 6][(i - 72) % 6] = __ldg(plan + 1 + i);
  }
  if (t == 0) s_nu = __ldg(plan);
  __syncthreads();
  const int nu = s_nu;
  const int lrow = lane / SUB, sub = lane % SUB;
  const int64_t q_bytes = int64_t(routes) * LG * 32;
  for (int tile = 0; tile < rows_per_cta / RT; ++tile) {
    const int row0 = (blockIdx.x * rows_per_cta / RT + tile) * RT;
    const int row = row0 + lrow;
    float acc[6] = {};
    for (int q = warp; q < nu; q += NW) {
      const int e = distinct_e[q], cnt = member_cnt[q];
      const uint8_t* base = down + (int64_t(e) * rows_local + row) * row_stride;
      int rt[6];
#pragma unroll
      for (int c = 0; c < 6; ++c) rt[c] = c < cnt ? member_route[q][c] : 0;
      float dot[6] = {};
      for (int g = sub; g < GROUPS; g += SUB) {
        const Group3<Type> w(base, g, book, masks);
        const uint8_t* rank_base = gathered + (g / LG) * rank_stride;
        const int8_t* hq = reinterpret_cast<const int8_t*>(rank_base);
        const float* hd = reinterpret_cast<const float*>(rank_base + q_bytes);
        const int gl = g % LG;
#pragma unroll
        for (int c = 0; c < 6; ++c)
          if (c < cnt) {
            int qv[8];
            float d;
            load_q8(hq, hd, int64_t(rt[c]) * LG + gl, qv, d);
            dot[c] = __builtin_fmaf(w.apply(qv, d), 0.125f, dot[c]);  // EDIT 4: FFMA(x, 0.125, dot) as in the old SASS
          }
      }
#pragma unroll
      for (int c = 0; c < 6; ++c) {
#pragma unroll
        for (int o = 1; o < SUB; o <<= 1)
          dot[c] += __shfl_xor_sync(0xffffffffU, dot[c], o);
      }
#pragma unroll
      for (int c = 0; c < 6; ++c)
        if (c < cnt) {
          const int token = rt[c] / top_k;
          const float v = __half2float(__float2half_rn(dot[c]));
#pragma unroll
          for (int k = 0; k < 6; ++k)
            if (k == token) acc[k] += v;
        }
    }
    __syncthreads();
    if (sub == 0)
#pragma unroll
      for (int k = 0; k < 6; ++k) part[warp][lrow][k] = acc[k];
    __syncthreads();
    if (t < RT * tokens) {
      const int r = t / tokens, k = t % tokens;
      float v = 0.f;
#pragma unroll
      for (int w8 = 0; w8 < NW; ++w8) v += part[w8][r][k];
      out[k * N + row_base + row0 + r] = __float2half_rn(v);
#pragma unroll
      for (int o = 0; o < 4; ++o)
        if (o * rows_local != row_base)
          out[k * N + o * rows_local + row0 + r] = __float2half_rn(0.f);
    }
  }
}
}  // namespace v41k
