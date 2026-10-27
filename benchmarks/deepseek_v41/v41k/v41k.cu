#include <cstdlib>
#include <mma.h>
// V4.1 routed experts, expert-major (each distinct expert's weights are read
// and decoded once per layer for all tokens that selected it). Original GGUF
// IQ1_M / IQ2_XXS / IQ2_XS blocks, Q8_1 activations, dp4a integer dots.
// Codebooks live in a pre-transformed global table (L1-cached) instead of a
// per-CTA shared copy, so small CTAs pay no book initialization.
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>

#include "sm70_turbomind/ops/gguf_dp4a.cuh"

namespace v41k {
using vllm::sm70_gguf::Q8_1;

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

__device__ __forceinline__ uint32_t sign_mask(int s) {
  return vllm::sm70_gguf::LatticeDot<22>::sign_mask(s);
}

// One 32-weight group of an original block, decoded to 8 signed int8x4 words.
template <int Type>
struct Group {
  uint32_t w[8];
  float d;
  int m0, m1;
  __device__ __forceinline__ Group(const uint8_t* row, int group,
                                   const uint32_t* __restrict__ book,
                                   const uint32_t* masks) {
    const uint8_t* b = row + (group >> 3) * block_bytes<Type>();
    const int sub = group & 7;
    uint32_t meta = 0;
    if constexpr (Type == 16)
      meta = vllm::sm70_gguf::load_u32_2(b + 6 + sub * 8);
#pragma unroll
    for (int octet = 0; octet < 4; ++octet) {
      int index, signs = 0, delta = 0;
      if constexpr (Type == 16) {
        index = b[2 + sub * 8 + octet];
        signs = (meta >> (7 * octet)) & 127;
      } else if constexpr (Type == 17) {
        const uint16_t code =
            *reinterpret_cast<const uint16_t*>(b + 2 + (sub * 4 + octet) * 2);
        index = code & 511;
        signs = code >> 9;
      } else {
        const int t = sub * 4 + octet;
        const int high = (b[32 + t / 2] >> (4 * (t % 2))) & 15;
        index = b[t] | ((high & 7) << 8);
        delta = high >> 3;
      }
      uint32_t first = __ldg(book + index * 2),
               second = __ldg(book + index * 2 + 1);
      if constexpr (Type == 29) {
        const uint32_t bias = delta ? 0xffffffffU : 0x01010101U;
        first = __vadd4(first, bias);
        second = __vadd4(second, bias);
      } else {
        signs |= (__popc(signs) & 1) << 7;
        const uint32_t lo = masks[signs & 15], hi = masks[signs >> 4];
        first = __vsub4(first ^ lo, lo);
        second = __vsub4(second ^ hi, hi);
      }
      w[2 * octet] = first;
      w[2 * octet + 1] = second;
    }
    if constexpr (Type == 29) {
      const auto* scales = reinterpret_cast<const uint16_t*>(b + 48);
      const uint16_t bits = (scales[0] >> 12) | ((scales[1] >> 8) & 0xf0) |
                            ((scales[2] >> 4) & 0xf00) | (scales[3] & 0xf000);
      d = __half2float(__ushort_as_half(bits));
      const int f = sub * 2, s = f + 1;
      m0 = 2 * ((scales[f / 4] >> (3 * (f % 4))) & 7) + 1;
      m1 = 2 * ((scales[s / 4] >> (3 * (s % 4))) & 7) + 1;
    } else {
      d = __half2float(*reinterpret_cast<const half*>(b));
      if constexpr (Type == 16) {
        m0 = m1 = 1 + 2 * (meta >> 28);
      } else {
        m0 = 1 + 2 * (b[66 + sub] & 15);
        m1 = 1 + 2 * (b[66 + sub] >> 4);
      }
    }
  }
  // qs: 8 activation words; xd: activation Q8_1 scale.
  __device__ __forceinline__ float apply(const int* qs, float xd) const {
    int s0 = 0, s1 = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) s0 = __dp4a(int(w[i]), qs[i], s0);
#pragma unroll
    for (int i = 4; i < 8; ++i) s1 = __dp4a(int(w[i]), qs[i], s1);
    return d * xd * float(s0 * m0 + s1 * m1) * 0.125f;
  }
};

// Distinct expert `u` (first-occurrence order over routes) and its routes.
struct Members {
  int expert, count;
  int route[6];
};

__device__ __forceinline__ void find_members(const int* ids, int routes, int u,
                                             Members* out) {
  // Executed by one warp; routes <= 36.
  const int lane = threadIdx.x & 31;
  __shared__ int first[64];
  int distinct = 0;
  for (int r = 0; r < routes; ++r) {
    const int e = ids[r];
    bool seen = false;
    for (int q = lane; q < r; q += 32) seen |= ids[q] == e;
    seen = __any_sync(0xffffffffU, seen);
    if (!seen) {
      if (lane == 0) first[distinct] = e;
      ++distinct;
    }
  }
  __syncwarp();
  if (lane == 0) {
    out->expert = u < distinct ? first[u] : -1;
    out->count = 0;
    if (out->expert >= 0)
      for (int r = 0; r < routes; ++r)
        if (ids[r] == out->expert && out->count < 6)
          out->route[out->count++] = r;
  }
  __syncwarp();
}

template <class Index>
__global__ void ids_to_int(const Index* ids, int* out, int n) {
  const int i = threadIdx.x;
  if (i < n) out[i] = int(ids[i]);
}

// Gate/up for one distinct expert x 32-row tile; fused SwiGLU (clamp 10,
// route probability) and Q8_1 quantization of the down input.
// Block: 256 threads = 8 warps; warp w -> matrix w/4, rows (w%4)*8 .. +8;
// LANES lanes per row-dot.
template <int Type, int LANES>
__global__ void __launch_bounds__(256)
    gate_up_em(Q8_1* __restrict__ down_in, const Q8_1* __restrict__ x,
               const int* ids, const float* __restrict__ prob,
               const uint8_t* __restrict__ gate, const uint8_t* __restrict__ up,
               int row_stride, const uint32_t* __restrict__ book, int routes,
               int top_k) {
  constexpr int GROUPS = 160, ROWS_PER_WARP = 32 / LANES;
  extern __shared__ int smem_i[];
  __shared__ Members mem;
  __shared__ uint32_t masks[16];
  __shared__ float red[6][2][32];
  const int t = threadIdx.x, warp = t / 32, lane = t % 32;
  if (warp == 0) find_members(ids, routes, blockIdx.y, &mem);
  if (t < 16) masks[t] = sign_mask(t);
  __syncthreads();
  const int e = mem.expert, cnt = mem.count;
  if (e < 0) return;
  // Stage member activations: qs[c][g][8] ints, then d[c][g] floats.
  int* xs = smem_i;
  float* xd = reinterpret_cast<float*>(smem_i + 6 * GROUPS * 8);
  for (int i = t; i < cnt * GROUPS * 9; i += blockDim.x) {
    const int c = i / (GROUPS * 9), rest = i % (GROUPS * 9);
    const int g = rest / 9, word = rest % 9;
    const int token = mem.route[c] / top_k;
    const uint32_t* src =
        reinterpret_cast<const uint32_t*>(x + token * GROUPS + g);
    const uint32_t v = src[word];
    if (word == 0)
      xd[c * GROUPS + g] = __low2float(*reinterpret_cast<const half2*>(&v));
    else
      xs[(c * GROUPS + g) * 8 + word - 1] = int(v);
  }
  __syncthreads();
  const int mat = warp / 4;
  const int local = (warp % 4) * 8 + lane / LANES * (8 / ROWS_PER_WARP);
  // ROWS_PER_WARP row-dots per warp, each LANES lanes; with LANES=4 there
  // are 8 row-dots per warp (= 8 rows), so `local` indexes them directly.
  const int row = blockIdx.x * 32 + (warp % 4) * 8 + lane / LANES;
  const uint8_t* base =
      (mat ? up : gate) + (int64_t(e) * 576 + row) * row_stride;
  float acc[6] = {};
  for (int g = lane % LANES; g < GROUPS; g += LANES) {
    const Group<Type> w(base, g, book, masks);
#pragma unroll
    for (int c = 0; c < 6; ++c)
      if (c < cnt)
        acc[c] += w.apply(xs + (c * GROUPS + g) * 8, xd[c * GROUPS + g]);
  }
#pragma unroll
  for (int c = 0; c < 6; ++c) {
#pragma unroll
    for (int o = LANES / 2; o; o >>= 1)
      acc[c] += __shfl_xor_sync(0xffffffffU, acc[c], o);
    if (c < cnt && lane % LANES == 0)
      red[c][mat][(warp % 4) * 8 + lane / LANES] = acc[c];
  }
  (void)local;
  __syncthreads();
  if (warp < cnt) {
    const int route = mem.route[warp];
    float g16 = __half2float(__float2half_rn(red[warp][0][lane]));
    float u16 = __half2float(__float2half_rn(red[warp][1][lane]));
    g16 = g16 > 10.f ? 10.f : g16;
    u16 = u16 > 10.f ? 10.f : (u16 < -10.f ? -10.f : u16);
    const float act = __fdiv_rn(g16, __fadd_rn(1.f, expf(-g16)));
    const float value = __half2float(
        __float2half_rn(__fmul_rn(__fmul_rn(act, u16), prob[route])));
    vllm::sm70_gguf::quantize_q8_1_warp(down_in + route * 18 + blockIdx.x,
                                        value);
  }
}

// Down for a 32*WARPS-row output tile over a chunk of distinct experts.
// Lane = output row; groups serial (18 = 576/32). Chunk partials go to `ws`;
// the last CTA of a tile sums chunks in order (deterministic).
template <int Type, int WARPS>
__global__ void __launch_bounds__(32 * WARPS)
    down_em(half* __restrict__ out, const Q8_1* __restrict__ h, const int* ids,
            const uint8_t* __restrict__ down, int row_stride, int left_groups,
            const uint32_t* __restrict__ book, int routes, int top_k,
            int tokens, float* __restrict__ ws, int* counters) {
  constexpr int GROUPS = 18, N = 5120;
  __shared__ int distinct_e[36], s_nu, member_route[36][6], member_cnt[36];
  __shared__ uint32_t masks[16];
  __shared__ int xs[6][GROUPS][8];
  __shared__ float xd[6][GROUPS];
  __shared__ bool s_last;
  const int t = threadIdx.x, lane = t % 32, warp = t / 32;
  if (t < 16) masks[t] = sign_mask(t);
  if (t == 0) {
    int nu = 0;
    for (int r = 0; r < routes; ++r) {
      const int e = ids[r];
      int u = 0;
      while (u < nu && distinct_e[u] != e) ++u;
      if (u == nu) {
        distinct_e[nu] = e;
        member_cnt[nu] = 0;
        ++nu;
      }
      member_route[u][member_cnt[u]++] = r;
    }
    s_nu = nu;
  }
  __syncthreads();
  const int nu = s_nu, chunks = gridDim.y, chunk = blockIdx.y;
  const int row = blockIdx.x * 32 * WARPS + warp * 32 + lane;
  float acc[6] = {};  // per token (tokens <= 6)
  for (int u = chunk; u < nu; u += chunks) {
    const int e = distinct_e[u], cnt = member_cnt[u];
    __syncthreads();
    for (int i = t; i < cnt * GROUPS * 9; i += blockDim.x) {
      const int c = i / (GROUPS * 9), rest = i % (GROUPS * 9);
      const int g = rest / 9, word = rest % 9;
      const uint32_t v = reinterpret_cast<const uint32_t*>(
          h + member_route[u][c] * GROUPS + g)[word];
      if (word == 0)
        xd[c][g] = __low2float(*reinterpret_cast<const half2*>(&v));
      else
        xs[c][g][word - 1] = int(v);
    }
    __syncthreads();
    const uint8_t* base = down + (int64_t(e) * N + row) * row_stride;
    float dot[6] = {};
    for (int g = 0; g < GROUPS; ++g) {
      const Group<Type> w(base, g + left_groups, book, masks);
#pragma unroll
      for (int c = 0; c < 6; ++c)
        if (c < cnt) dot[c] += w.apply(xs[c][g], xd[c][g]);
    }
#pragma unroll
    for (int c = 0; c < 6; ++c)
      if (c < cnt) {
        const int token = member_route[u][c] / top_k;
        const float v = __half2float(__float2half_rn(dot[c]));
#pragma unroll
        for (int k = 0; k < 6; ++k)
          if (k == token) acc[k] += v;
      }
  }
  if (chunks == 1) {
    for (int k = 0; k < tokens; ++k) out[k * N + row] = __float2half_rn(acc[k]);
    return;
  }
  for (int k = 0; k < tokens; ++k)
    __stcg(ws + (int64_t(chunk) * tokens + k) * N + row, acc[k]);
  __threadfence();
  __syncthreads();
  if (t == 0) s_last = atomicAdd(counters + blockIdx.x, 1) == chunks - 1;
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  for (int k = 0; k < tokens; ++k) {
    float sum = 0.f;
    for (int c = 0; c < chunks; ++c)
      sum += __ldcg(ws + (int64_t(c) * tokens + k) * N + row);
    out[k * N + row] = __float2half_rn(sum);
  }
  if (t == 0) counters[blockIdx.x] = 0;
}

// ---- v2: Codex reuse structure (16 lanes per row), 32-row tiles for the
// fused SwiGLU+Q8 epilogue, global L1 codebook, compact member lists.
template <int Type>
__global__ void __launch_bounds__(512)
    gate_up_em2(Q8_1* __restrict__ down_in, const Q8_1* __restrict__ x,
                const int* ids, const float* __restrict__ prob,
                const uint8_t* __restrict__ gate,
                const uint8_t* __restrict__ up, int row_stride,
                const uint32_t* __restrict__ book, int routes, int top_k) {
  constexpr int GROUPS = 160;
  __shared__ int s_expert, s_cnt, s_route[6], s_tok[6];
  __shared__ uint32_t masks[16];
  __shared__ float red[6][2][32];
  const int t = threadIdx.x, warp = t / 32, lane = t % 32;
  if (t < 16) masks[t] = sign_mask(t);
  if (t == 0) {
    // u-th distinct expert in first-occurrence order.
    int u = blockIdx.y, e = -1, distinct = 0;
    for (int r = 0; r < routes && e < 0; ++r) {
      bool seen = false;
      for (int q = 0; q < r; ++q) seen |= ids[q] == ids[r];
      if (!seen) {
        if (distinct == u) e = ids[r];
        ++distinct;
      }
    }
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
  if (e < 0) return;
  // 16 warps x 2 rows: lanes 0-15 row 2w, lanes 16-31 row 2w+1.
  const int lrow = warp * 2 + lane / 16;
  const int row = blockIdx.x * 32 + lrow;
  const uint8_t* g = gate + (int64_t(e) * 576 + row) * row_stride;
  const uint8_t* u = up + (int64_t(e) * 576 + row) * row_stride;
  int tok[6];
#pragma unroll
  for (int c = 0; c < 6; ++c) tok[c] = c < cnt ? s_tok[c] : 0;
  float gs[6] = {}, us[6] = {};
  for (int group = lane % 16; group < GROUPS; group += 16) {
    const Group<Type> dg(g, group, book, masks);
    const Group<Type> du(u, group, book, masks);
#pragma unroll
    for (int c = 0; c < 6; ++c)
      if (c < cnt) {
        const Q8_1& a = x[tok[c] * GROUPS + group];
        const int* qs = reinterpret_cast<const int*>(a.qs);
        const float xd = __low2float(a.ds);
        gs[c] += dg.apply(qs, xd);
        us[c] += du.apply(qs, xd);
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
    vllm::sm70_gguf::quantize_q8_1_warp(down_in + route * 18 + blockIdx.x,
                                        value);
  }
}

// Down v2: CTA = 8 output rows (one warp-row set), 4 warps split the
// distinct experts; 4 lanes per row over the 18 groups; per-token sums are
// reduced across warps in shared memory (fixed order: deterministic).
template <int Type>
__global__ void __launch_bounds__(128)
    down_em2(half* __restrict__ out, const Q8_1* __restrict__ h, const int* ids,
             const uint8_t* __restrict__ down, int row_stride, int left_groups,
             const uint32_t* __restrict__ book, int routes, int top_k,
             int tokens) {
  constexpr int GROUPS = 18, N = 5120;
  __shared__ int distinct_e[36], s_nu, member_route[36][6], member_cnt[36];
  __shared__ uint32_t masks[16];
  __shared__ float part[4][8][6];
  const int t = threadIdx.x, lane = t % 32, warp = t / 32;
  if (t < 16) masks[t] = sign_mask(t);
  if (t == 0) {
    int nu = 0;
    for (int r = 0; r < routes; ++r) {
      const int e = ids[r];
      int q = 0;
      while (q < nu && distinct_e[q] != e) ++q;
      if (q == nu) {
        distinct_e[nu] = e;
        member_cnt[nu] = 0;
        ++nu;
      }
      member_route[q][member_cnt[q]++] = r;
    }
    s_nu = nu;
  }
  __syncthreads();
  const int nu = s_nu;
  const int lrow = lane / 4, sub = lane % 4;
  const int row = blockIdx.x * 8 + lrow;
  float acc[6] = {};
  for (int q = warp; q < nu; q += 4) {
    const int e = distinct_e[q], cnt = member_cnt[q];
    const uint8_t* base = down + (int64_t(e) * N + row) * row_stride;
    int rt[6];
#pragma unroll
    for (int c = 0; c < 6; ++c) rt[c] = c < cnt ? member_route[q][c] : 0;
    float dot[6] = {};
    for (int g = sub; g < GROUPS; g += 4) {
      const Group<Type> w(base, g + left_groups, book, masks);
#pragma unroll
      for (int c = 0; c < 6; ++c)
        if (c < cnt) {
          const Q8_1& a = h[rt[c] * GROUPS + g];
          dot[c] +=
              w.apply(reinterpret_cast<const int*>(a.qs), __low2float(a.ds));
        }
    }
#pragma unroll
    for (int c = 0; c < 6; ++c) {
      dot[c] += __shfl_xor_sync(0xffffffffU, dot[c], 1);
      dot[c] += __shfl_xor_sync(0xffffffffU, dot[c], 2);
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
  if (sub == 0)
#pragma unroll
    for (int k = 0; k < 6; ++k) part[warp][lrow][k] = acc[k];
  __syncthreads();
  if (t < 8 * tokens) {
    const int r = t / tokens, k = t % tokens;
    const float v =
        part[0][r][k] + part[1][r][k] + part[2][r][k] + part[3][r][k];
    out[k * N + blockIdx.x * 8 + r] = __float2half_rn(v);
  }
}

// ---- diagnostics: bandwidth floor and decode-only cost (gate+up, 18 tiles)
__global__ void __launch_bounds__(512)
    gu_read_only(float* sink, const int* ids, const uint8_t* __restrict__ gate,
                 const uint8_t* __restrict__ up, int row_stride, int routes) {
  __shared__ int s_expert;
  if (threadIdx.x == 0) {
    int u = blockIdx.y, e = -1, distinct = 0;
    for (int r = 0; r < routes && e < 0; ++r) {
      bool seen = false;
      for (int q = 0; q < r; ++q) seen |= ids[q] == ids[r];
      if (!seen) {
        if (distinct == u) e = ids[r];
        ++distinct;
      }
    }
    s_expert = e;
  }
  __syncthreads();
  const int e = s_expert;
  if (e < 0) return;
  const int64_t bytes = int64_t(32) * row_stride;  // 32 rows per tile
  const uint4* g = reinterpret_cast<const uint4*>(
      gate + (int64_t(e) * 576 + blockIdx.x * 32) * row_stride);
  const uint4* u = reinterpret_cast<const uint4*>(
      up + (int64_t(e) * 576 + blockIdx.x * 32) * row_stride);
  uint32_t acc = 0;
  for (int64_t i = threadIdx.x; i < bytes / 16; i += blockDim.x) {
    const uint4 a = __ldg(g + i), b = __ldg(u + i);
    acc ^= a.x ^ a.y ^ a.z ^ a.w ^ b.x ^ b.y ^ b.z ^ b.w;
  }
  if (acc == 0x12345678u) sink[0] = 1.f;
}

template <int Type>
__global__ void __launch_bounds__(512)
    gu_decode_only(float* sink, const int* ids,
                   const uint8_t* __restrict__ gate,
                   const uint8_t* __restrict__ up, int row_stride,
                   const uint32_t* __restrict__ book, int routes) {
  __shared__ int s_expert;
  __shared__ uint32_t masks[16];
  const int t = threadIdx.x, warp = t / 32, lane = t % 32;
  if (t < 16) masks[t] = sign_mask(t);
  if (t == 0) {
    int u = blockIdx.y, e = -1, distinct = 0;
    for (int r = 0; r < routes && e < 0; ++r) {
      bool seen = false;
      for (int q = 0; q < r; ++q) seen |= ids[q] == ids[r];
      if (!seen) {
        if (distinct == u) e = ids[r];
        ++distinct;
      }
    }
    s_expert = e;
  }
  __syncthreads();
  const int e = s_expert;
  if (e < 0) return;
  const int row = blockIdx.x * 32 + warp * 2 + lane / 16;
  const uint8_t* g = gate + (int64_t(e) * 576 + row) * row_stride;
  const uint8_t* u = up + (int64_t(e) * 576 + row) * row_stride;
  uint32_t acc = 0;
  float f = 0.f;
  for (int group = lane % 16; group < 160; group += 16) {
    const Group<Type> dg(g, group, book, masks);
    const Group<Type> du(u, group, book, masks);
#pragma unroll
    for (int i = 0; i < 8; ++i) acc ^= dg.w[i] ^ du.w[i];
    f += dg.d * dg.m0 + du.d * du.m1;
  }
  if (acc == 0x12345678u || f == 1234.5f) sink[0] = 1.f;
}

// ---- v3: vector loads + shared-memory codebook ----------------------------
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

#ifndef V41K_GU_UNROLL
  #define V41K_GU_UNROLL 2
#endif
constexpr int kGuUnroll = V41K_GU_UNROLL;
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
    return d * xd * float(s0 * m0 + s1 * m1) * 0.125f;
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

template <int Type>
__global__ void __launch_bounds__(512)
    gate_up_em3(Q8_1* __restrict__ down_in, const Q8_1* __restrict__ x,
                const int* ids, const float* __restrict__ prob,
                const uint8_t* __restrict__ gate,
                const uint8_t* __restrict__ up, int row_stride,
                const uint32_t* __restrict__ gbook, int routes, int top_k) {
  constexpr int GROUPS = 160, BW = book_words<Type>();
  __shared__ __align__(16) uint32_t book[BW];
  __shared__ int s_expert, s_cnt, s_route[6], s_tok[6];
  __shared__ uint32_t masks[16];
  __shared__ float red[6][2][32];
  const int t = threadIdx.x, warp = t / 32, lane = t % 32;
  if (t == 0) {
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
  if (e < 0) return;
  for (int i = t; i < BW; i += blockDim.x) book[i] = __ldg(gbook + i);
  if (t < 16) masks[t] = sign_mask(t);
  __syncthreads();
  const int lrow = warp * 2 + lane / 16;
  const int row = blockIdx.x * 32 + lrow;
  const uint8_t* g = gate + (int64_t(e) * 576 + row) * row_stride;
  const uint8_t* u = up + (int64_t(e) * 576 + row) * row_stride;
  int tok[6];
#pragma unroll
  for (int c = 0; c < 6; ++c) tok[c] = c < cnt ? s_tok[c] : 0;
  float gs[6] = {}, us[6] = {};
#pragma unroll 2
  for (int group = lane % 16; group < GROUPS; group += 16) {
    const Group3<Type> dg(g, group, book, masks);
    const Group3<Type> du(u, group, book, masks);
#pragma unroll
    for (int c = 0; c < 6; ++c)
      if (c < cnt) {
        const Q8_1& a = x[tok[c] * GROUPS + group];
        const int* qs = reinterpret_cast<const int*>(a.qs);
        const float xd = __low2float(a.ds);
        gs[c] += dg.apply(qs, xd);
        us[c] += du.apply(qs, xd);
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
    vllm::sm70_gguf::quantize_q8_1_warp(down_in + route * 18 + blockIdx.x,
                                        value);
  }
}

template <int Type>
__global__ void __launch_bounds__(128)
    down_em3(half* __restrict__ out, const Q8_1* __restrict__ h, const int* ids,
             const uint8_t* __restrict__ down, int row_stride, int left_groups,
             const uint32_t* __restrict__ gbook, int routes, int top_k,
             int tokens, int rows_per_cta) {
  constexpr int GROUPS = 18, N = 5120, BW = book_words<Type>();
  __shared__ __align__(16) uint32_t book[BW];
  __shared__ int distinct_e[36], s_nu, member_route[36][6], member_cnt[36];
  __shared__ uint32_t masks[16];
  __shared__ float part[4][8][6];
  const int t = threadIdx.x, lane = t % 32, warp = t / 32;
  for (int i = t; i < BW; i += blockDim.x) book[i] = __ldg(gbook + i);
  if (t < 16) masks[t] = sign_mask(t);
  if (t == 0) {
    int nu = 0;
    for (int r = 0; r < routes; ++r) {
      const int e = ids[r];
      int q = 0;
      while (q < nu && distinct_e[q] != e) ++q;
      if (q == nu) {
        distinct_e[nu] = e;
        member_cnt[nu] = 0;
        ++nu;
      }
      member_route[q][member_cnt[q]++] = r;
    }
    s_nu = nu;
  }
  __syncthreads();
  const int nu = s_nu;
  const int lrow = lane / 4, sub = lane % 4;
  for (int tile = 0; tile < rows_per_cta / 8; ++tile) {
    const int row0 = (blockIdx.x * rows_per_cta / 8 + tile) * 8;
    const int row = row0 + lrow;
    float acc[6] = {};
    for (int q = warp; q < nu; q += 4) {
      const int e = distinct_e[q], cnt = member_cnt[q];
      const uint8_t* base = down + (int64_t(e) * N + row) * row_stride;
      int rt[6];
#pragma unroll
      for (int c = 0; c < 6; ++c) rt[c] = c < cnt ? member_route[q][c] : 0;
      float dot[6] = {};
      for (int g = sub; g < GROUPS; g += 4) {
        const Group3<Type> w(base, g + left_groups, book, masks);
#pragma unroll
        for (int c = 0; c < 6; ++c)
          if (c < cnt) {
            const Q8_1& a = h[rt[c] * GROUPS + g];
            dot[c] +=
                w.apply(reinterpret_cast<const int*>(a.qs), __low2float(a.ds));
          }
      }
#pragma unroll
      for (int c = 0; c < 6; ++c) {
        dot[c] += __shfl_xor_sync(0xffffffffU, dot[c], 1);
        dot[c] += __shfl_xor_sync(0xffffffffU, dot[c], 2);
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
    if (t < 8 * tokens) {
      const int r = t / tokens, k = t % tokens;
      const float v =
          part[0][r][k] + part[1][r][k] + part[2][r][k] + part[3][r][k];
      out[k * N + row0 + r] = __float2half_rn(v);
    }
  }
}

template <int Type>
__global__ void __launch_bounds__(512)
    gu_decode3(float* sink, const int* ids, const uint8_t* __restrict__ gate,
               const uint8_t* __restrict__ up, int row_stride,
               const uint32_t* __restrict__ gbook, int routes) {
  constexpr int BW = book_words<Type>();
  __shared__ __align__(16) uint32_t book[BW];
  __shared__ int s_expert;
  __shared__ uint32_t masks[16];
  const int t = threadIdx.x, warp = t / 32, lane = t % 32;
  if (t == 0) s_expert = find_distinct(ids, routes, blockIdx.y);
  __syncthreads();
  const int e = s_expert;
  if (e < 0) return;
  for (int i = t; i < BW; i += blockDim.x) book[i] = __ldg(gbook + i);
  if (t < 16) masks[t] = sign_mask(t);
  __syncthreads();
  const int row = blockIdx.x * 32 + warp * 2 + lane / 16;
  const uint8_t* g = gate + (int64_t(e) * 576 + row) * row_stride;
  const uint8_t* u = up + (int64_t(e) * 576 + row) * row_stride;
  uint32_t acc = 0;
  float f = 0.f;
  for (int group = lane % 16; group < 160; group += 16) {
    const Group3<Type> dg(g, group, book, masks);
    const Group3<Type> du(u, group, book, masks);
#pragma unroll
    for (int i = 0; i < 8; ++i) acc ^= dg.w[i] ^ du.w[i];
    f += dg.d * dg.m0 + du.d * du.m1;
  }
  if (acc == 0x12345678u || f == 1234.5f) sink[0] = 1.f;
}

// ---- v4: v3 decode + activations staged once in aligned shared memory ----
// x layout in smem: qs as int4 pairs [tok][group][2] (32 B), d as float.
template <int Type>
__global__ void __launch_bounds__(512)
    gate_up_em4(Q8_1* __restrict__ down_in, const Q8_1* __restrict__ x,
                const int* ids, const float* __restrict__ prob,
                const uint8_t* __restrict__ gate,
                const uint8_t* __restrict__ up, int row_stride,
                const uint32_t* __restrict__ gbook, int routes, int top_k,
                int tokens) {
  constexpr int GROUPS = 160, BW = book_words<Type>();
  extern __shared__ int4 dyn[];
  int4* xq = dyn;                                              // [6][160][2]
  float* xd = reinterpret_cast<float*>(dyn + 6 * GROUPS * 2);  // [6][160]
  uint32_t* book = reinterpret_cast<uint32_t*>(xd + 6 * GROUPS);
  __shared__ int s_expert, s_cnt, s_route[6], s_tok[6];
  __shared__ uint32_t masks[16];
  __shared__ float red[6][2][32];
  const int t = threadIdx.x, warp = t / 32, lane = t % 32;
  if (t == 0) {
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
  if (e < 0) return;
  for (int i = t; i < BW; i += blockDim.x) book[i] = __ldg(gbook + i);
  if (t < 16) masks[t] = sign_mask(t);
  // stage only member tokens, compacted: slot c <- token s_tok[c]
  for (int i = t; i < cnt * GROUPS * 9; i += blockDim.x) {
    const int c = i / (GROUPS * 9), rest = i % (GROUPS * 9);
    const int g = rest / 9, word = rest % 9;
    const uint32_t v =
        reinterpret_cast<const uint32_t*>(x + s_tok[c] * GROUPS + g)[word];
    if (word == 0)
      xd[c * GROUPS + g] = __low2float(*reinterpret_cast<const half2*>(&v));
    else
      reinterpret_cast<uint32_t*>(xq + (c * GROUPS + g) * 2)[word - 1] = v;
  }
  __syncthreads();
  const int lrow = warp * 2 + lane / 16;
  const int row = blockIdx.x * 32 + lrow;
  const uint8_t* g = gate + (int64_t(e) * 576 + row) * row_stride;
  const uint8_t* u = up + (int64_t(e) * 576 + row) * row_stride;
  float gs[6] = {}, us[6] = {};
#pragma unroll 2
  for (int group = lane % 16; group < GROUPS; group += 16) {
    const Group3<Type> dg(g, group, book, masks);
    const Group3<Type> du(u, group, book, masks);
#pragma unroll
    for (int c = 0; c < 6; ++c)
      if (c < cnt) {
        const int4 a0 = xq[(c * GROUPS + group) * 2],
                   a1 = xq[(c * GROUPS + group) * 2 + 1];
        const int qs[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
        const float d = xd[c * GROUPS + group];
        gs[c] += dg.apply(qs, d);
        us[c] += du.apply(qs, d);
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
    vllm::sm70_gguf::quantize_q8_1_warp(down_in + route * 18 + blockIdx.x,
                                        value);
  }
  (void)tokens;
}

template <int Type>
__global__ void __launch_bounds__(128)
    down_em4(half* __restrict__ out, const Q8_1* __restrict__ h, const int* ids,
             const uint8_t* __restrict__ down, int row_stride, int left_groups,
             const uint32_t* __restrict__ gbook, int routes, int top_k,
             int tokens, int rows_per_cta) {
  constexpr int GROUPS = 18, N = 5120, BW = book_words<Type>();
  __shared__ __align__(16) uint32_t book[BW];
  __shared__ int4 hq[36 * GROUPS * 2];
  __shared__ float hd[36 * GROUPS];
  __shared__ int distinct_e[36], s_nu, member_route[36][6], member_cnt[36];
  __shared__ uint32_t masks[16];
  __shared__ float part[4][8][6];
  const int t = threadIdx.x, lane = t % 32, warp = t / 32;
  for (int i = t; i < BW; i += blockDim.x) book[i] = __ldg(gbook + i);
  if (t < 16) masks[t] = sign_mask(t);
  for (int i = t; i < routes * GROUPS * 9; i += blockDim.x) {
    const int rg = i / 9, word = i % 9;
    const uint32_t v = reinterpret_cast<const uint32_t*>(h + rg)[word];
    if (word == 0)
      hd[rg] = __low2float(*reinterpret_cast<const half2*>(&v));
    else
      reinterpret_cast<uint32_t*>(hq + rg * 2)[word - 1] = v;
  }
  if (t == 0) {
    int nu = 0;
    for (int r = 0; r < routes; ++r) {
      const int e = ids[r];
      int q = 0;
      while (q < nu && distinct_e[q] != e) ++q;
      if (q == nu) {
        distinct_e[nu] = e;
        member_cnt[nu] = 0;
        ++nu;
      }
      member_route[q][member_cnt[q]++] = r;
    }
    s_nu = nu;
  }
  __syncthreads();
  const int nu = s_nu;
  const int lrow = lane / 4, sub = lane % 4;
  for (int tile = 0; tile < rows_per_cta / 8; ++tile) {
    const int row0 = (blockIdx.x * rows_per_cta / 8 + tile) * 8;
    const int row = row0 + lrow;
    float acc[6] = {};
    for (int q = warp; q < nu; q += 4) {
      const int e = distinct_e[q], cnt = member_cnt[q];
      const uint8_t* base = down + (int64_t(e) * N + row) * row_stride;
      int rt[6];
#pragma unroll
      for (int c = 0; c < 6; ++c) rt[c] = c < cnt ? member_route[q][c] : 0;
      float dot[6] = {};
      for (int g = sub; g < GROUPS; g += 4) {
        const Group3<Type> w(base, g + left_groups, book, masks);
#pragma unroll
        for (int c = 0; c < 6; ++c)
          if (c < cnt) {
            const int rg = rt[c] * GROUPS + g;
            const int4 a0 = hq[rg * 2], a1 = hq[rg * 2 + 1];
            const int qs[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            dot[c] += w.apply(qs, hd[rg]);
          }
      }
#pragma unroll
      for (int c = 0; c < 6; ++c) {
        dot[c] += __shfl_xor_sync(0xffffffffU, dot[c], 1);
        dot[c] += __shfl_xor_sync(0xffffffffU, dot[c], 2);
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
    if (t < 8 * tokens) {
      const int r = t / tokens, k = t % tokens;
      const float v =
          part[0][r][k] + part[1][r][k] + part[2][r][k] + part[3][r][k];
      out[k * N + row0 + r] = __float2half_rn(v);
    }
  }
}

// ---- FP8 (e4m3, UE8M0 32x32 block scales) small-M GEMV for the DSpark draft.
// CTA = 32 output rows (one scale row block), 8 warps x 4 rows; lanes stride
// K in 16-byte chunks. e4m3 -> fp16 by bit placement, x 2^8 folded into the
// block scale (exact, subnormals included). Split-K partials are reduced in a
// fixed order by fp8_reduce (deterministic).
template <int M>
__global__ void __launch_bounds__(256)
    fp8_gemv_kernel(float* __restrict__ partials, half* __restrict__ out,
                    const half* __restrict__ x, const uint8_t* __restrict__ w,
                    const uint8_t* __restrict__ scales, int N, int K,
                    int direct, int ksplit) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int n0 = blockIdx.x * 32 + warp * 4;
  const int s = blockIdx.y;
  const int kbeg = s * ksplit, k_end = min(K, kbeg + ksplit);
  const int sgroups = K / 32;
  float acc[4][M];
#pragma unroll
  for (int r = 0; r < 4; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) acc[r][m] = 0.f;
  for (int k0 = kbeg + lane * 16; k0 < k_end; k0 += 512) {
    float xv[M][16];
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const uint4 a =
          __ldg(reinterpret_cast<const uint4*>(x + int64_t(m) * K + k0));
      const uint4 b =
          __ldg(reinterpret_cast<const uint4*>(x + int64_t(m) * K + k0 + 8));
      const half2* ha = reinterpret_cast<const half2*>(&a);
      const half2* hb = reinterpret_cast<const half2*>(&b);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float2 fa = __half22float2(ha[i]), fb = __half22float2(hb[i]);
        xv[m][2 * i] = fa.x;
        xv[m][2 * i + 1] = fa.y;
        xv[m][8 + 2 * i] = fb.x;
        xv[m][8 + 2 * i + 1] = fb.y;
      }
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int n = n0 + r;
      if (n >= N) break;
      const uint4 q =
          __ldg(reinterpret_cast<const uint4*>(w + int64_t(n) * K + k0));
      const int e = scales[(n / 32) * sgroups + k0 / 32];
      const float scale = e + 8 < 255 ? __int_as_float((e + 8) << 23)
                                      : __int_as_float(e << 23) * 256.f;
      const uint32_t words[4] = {q.x, q.y, q.z, q.w};
      float dot[M];
#pragma unroll
      for (int m = 0; m < M; ++m) dot[m] = 0.f;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const uint32_t b = (words[i / 4] >> (8 * (i % 4))) & 0xffu;
        const uint16_t bits = uint16_t(((b & 0x7fu) << 7) | ((b & 0x80u) << 8));
        const float v = __half2float(__ushort_as_half(bits));
#pragma unroll
        for (int m = 0; m < M; ++m) dot[m] = fmaf(v, xv[m][i], dot[m]);
      }
#pragma unroll
      for (int m = 0; m < M; ++m) acc[r][m] = fmaf(dot[m], scale, acc[r][m]);
    }
  }
#pragma unroll
  for (int r = 0; r < 4; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) {
      float v = acc[r][m];
#pragma unroll
      for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffU, v, o);
      acc[r][m] = v;
    }
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int n = n0 + r;
      if (n >= N) break;
#pragma unroll
      for (int m = 0; m < M; ++m) {
        if (direct)
          out[int64_t(m) * N + n] = __float2half_rn(acc[r][m]);
        else
          partials[(int64_t(s) * M + m) * N + n] = acc[r][m];
      }
    }
  }
}

__global__ void fp8_reduce(half* out, const float* partials, const half* bias,
                           int M, int N, int split) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M * N) return;
  float v = 0.f;
  for (int s = 0; s < split; ++s) v += partials[int64_t(s) * M * N + i];
  if (bias) v += __half2float(bias[i % N]);
  out[i] = __float2half_rn(v);
}

// ---- v5: aligned activations (int8 qs as two int4 + float scale arrays) ----
__global__ void q8_aligned_kernel(int8_t* __restrict__ qs,
                                  float* __restrict__ ds,
                                  const half* __restrict__ x, int K) {
  // one warp per 32-element group
  const int g = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  const int lane = threadIdx.x % 32, row = blockIdx.y, groups = K / 32;
  if (g >= groups) return;
  const float v = __half2float(x[int64_t(row) * K + g * 32 + lane]);
  float m = fabsf(v);
#pragma unroll
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffU, m, o));
  const float d = m / 127.f;
  qs[(int64_t(row) * groups + g) * 32 + lane] =
      m == 0.f ? 0 : int8_t(roundf(v / d));
  if (!lane) ds[int64_t(row) * groups + g] = __half2float(__float2half_rn(d));
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
        gs[c] += dg.apply(q, d);
        us[c] += du.apply(q, d);
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

template <int Type, int RB>
__global__ void __launch_bounds__(512)
    gate_up_em6(int8_t* __restrict__ hq, float* __restrict__ hd,
                const int8_t* __restrict__ xq, const float* __restrict__ xd,
                const int* ids, const float* __restrict__ prob,
                const uint8_t* __restrict__ gate,
                const uint8_t* __restrict__ up, int row_stride,
                const uint32_t* __restrict__ gbook, int routes, int top_k) {
  constexpr int GROUPS = 160, BW = book_words<Type>();
  __shared__ __align__(16) uint32_t book[BW];
  __shared__ int s_expert, s_cnt, s_route[6], s_tok[6];
  __shared__ uint32_t masks[16];
  __shared__ float red[6][2][32];
  const int t = threadIdx.x, warp = t / 32, lane = t % 32;
  if (t == 0) {
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
  if (e < 0) return;
  for (int i = t; i < BW; i += blockDim.x) book[i] = __ldg(gbook + i);
  if (t < 16) masks[t] = sign_mask(t);
  __syncthreads();
  const int lrow = warp * 2 + lane / 16;
  for (int rb = 0; rb < RB; ++rb) {
    const int block = blockIdx.x * RB + rb;
    const int row = block * 32 + lrow;
    const uint8_t* g = gate + (int64_t(e) * 576 + row) * row_stride;
    const uint8_t* u = up + (int64_t(e) * 576 + row) * row_stride;
    int tok[6];
#pragma unroll
    for (int c = 0; c < 6; ++c) tok[c] = c < cnt ? s_tok[c] : 0;
    float gs[6] = {}, us[6] = {};
#pragma unroll 2
    for (int group = lane % 16; group < GROUPS; group += 16) {
      const Group3<Type> dg(g, group, book, masks);
      const Group3<Type> du(u, group, book, masks);
#pragma unroll
      for (int c = 0; c < 6; ++c)
        if (c < cnt) {
          int q[8];
          float d;
          load_q8(xq, xd, int64_t(tok[c]) * GROUPS + group, q, d);
          gs[c] += dg.apply(q, d);
          us[c] += du.apply(q, d);
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
      q8_warp_aligned(hq, hd, int64_t(route) * 18 + block, value);
    }
    __syncthreads();
  }
}

template <int Type>
__global__ void __launch_bounds__(128)
    down_em5(half* __restrict__ out, const int8_t* __restrict__ hq,
             const float* __restrict__ hd, const int* ids,
             const uint8_t* __restrict__ down, int row_stride, int left_groups,
             const uint32_t* __restrict__ gbook, int routes, int top_k,
             int tokens, int rows_per_cta) {
  constexpr int GROUPS = 18, N = 5120, BW = book_words<Type>();
  __shared__ __align__(16) uint32_t book[BW];
  __shared__ int distinct_e[36], s_nu, member_route[36][6], member_cnt[36];
  __shared__ uint32_t masks[16];
  __shared__ float part[4][8][6];
  const int t = threadIdx.x, lane = t % 32, warp = t / 32;
  for (int i = t; i < BW; i += blockDim.x) book[i] = __ldg(gbook + i);
  if (t < 16) masks[t] = sign_mask(t);
  if (t == 0) {
    int nu = 0;
    for (int r = 0; r < routes; ++r) {
      const int e = ids[r];
      int q = 0;
      while (q < nu && distinct_e[q] != e) ++q;
      if (q == nu) {
        distinct_e[nu] = e;
        member_cnt[nu] = 0;
        ++nu;
      }
      member_route[q][member_cnt[q]++] = r;
    }
    s_nu = nu;
  }
  __syncthreads();
  const int nu = s_nu;
  const int lrow = lane / 4, sub = lane % 4;
  for (int tile = 0; tile < rows_per_cta / 8; ++tile) {
    const int row0 = (blockIdx.x * rows_per_cta / 8 + tile) * 8;
    const int row = row0 + lrow;
    float acc[6] = {};
    for (int q = warp; q < nu; q += 4) {
      const int e = distinct_e[q], cnt = member_cnt[q];
      const uint8_t* base = down + (int64_t(e) * N + row) * row_stride;
      int rt[6];
#pragma unroll
      for (int c = 0; c < 6; ++c) rt[c] = c < cnt ? member_route[q][c] : 0;
      float dot[6] = {};
      for (int g = sub; g < GROUPS; g += 4) {
        const Group3<Type> w(base, g + left_groups, book, masks);
#pragma unroll
        for (int c = 0; c < 6; ++c)
          if (c < cnt) {
            int qv[8];
            float d;
            load_q8(hq, hd, int64_t(rt[c]) * GROUPS + g, qv, d);
            dot[c] += w.apply(qv, d);
          }
      }
#pragma unroll
      for (int c = 0; c < 6; ++c) {
        dot[c] += __shfl_xor_sync(0xffffffffU, dot[c], 1);
        dot[c] += __shfl_xor_sync(0xffffffffU, dot[c], 2);
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
    if (t < 8 * tokens) {
      const int r = t / tokens, k = t % tokens;
      out[k * N + row0 + r] = __float2half_rn(part[0][r][k] + part[1][r][k] +
                                              part[2][r][k] + part[3][r][k]);
    }
  }
}

template <int Type, int NW>
__global__ void __launch_bounds__(32 * NW, 8 / NW * 4)
    down_em6(half* __restrict__ out, const int8_t* __restrict__ hq,
             const float* __restrict__ hd, const int* ids,
             const uint8_t* __restrict__ down, int row_stride, int left_groups,
             const uint32_t* __restrict__ gbook, int routes, int top_k,
             int tokens, int rows_per_cta, const int* __restrict__ plan) {
  constexpr int GROUPS = 18, N = 5120, BW = book_words<Type>();
  __shared__ __align__(16) uint32_t book[BW];
  __shared__ int distinct_e[36], s_nu, member_route[36][6], member_cnt[36];
  __shared__ uint32_t masks[16];
  __shared__ float part[NW][8][6];
  const int t = threadIdx.x, lane = t % 32, warp = t / 32;
  for (int i = t; i < BW; i += blockDim.x) book[i] = __ldg(gbook + i);
  if (t < 16) masks[t] = sign_mask(t);
  if (plan != nullptr) {
    for (int i = t; i < 36 + 36 + 216; i += blockDim.x) {
      if (i < 36)
        distinct_e[i] = __ldg(plan + 1 + i);
      else if (i < 72)
        member_cnt[i - 36] = __ldg(plan + 1 + i);
      else
        member_route[(i - 72) / 6][(i - 72) % 6] = __ldg(plan + 1 + i);
    }
    if (t == 0) s_nu = __ldg(plan);
  } else if (t == 0) {
    int nu = 0;
    for (int r = 0; r < routes; ++r) {
      const int e = ids[r];
      int q = 0;
      while (q < nu && distinct_e[q] != e) ++q;
      if (q == nu) {
        distinct_e[nu] = e;
        member_cnt[nu] = 0;
        ++nu;
      }
      member_route[q][member_cnt[q]++] = r;
    }
    s_nu = nu;
  }
  __syncthreads();
  const int nu = s_nu;
  const int lrow = lane / 4, sub = lane % 4;
  for (int tile = 0; tile < rows_per_cta / 8; ++tile) {
    const int row0 = (blockIdx.x * rows_per_cta / 8 + tile) * 8;
    const int row = row0 + lrow;
    float acc[6] = {};
    for (int q = warp; q < nu; q += NW) {
      const int e = distinct_e[q], cnt = member_cnt[q];
      const uint8_t* base = down + (int64_t(e) * N + row) * row_stride;
      int rt[6];
#pragma unroll
      for (int c = 0; c < 6; ++c) rt[c] = c < cnt ? member_route[q][c] : 0;
      float dot[6] = {};
      for (int g = sub; g < GROUPS; g += 4) {
        const Group3<Type> w(base, g + left_groups, book, masks);
#pragma unroll
        for (int c = 0; c < 6; ++c)
          if (c < cnt) {
            int qv[8];
            float d;
            load_q8(hq, hd, int64_t(rt[c]) * GROUPS + g, qv, d);
            dot[c] += w.apply(qv, d);
          }
      }
#pragma unroll
      for (int c = 0; c < 6; ++c) {
        dot[c] += __shfl_xor_sync(0xffffffffU, dot[c], 1);
        dot[c] += __shfl_xor_sync(0xffffffffU, dot[c], 2);
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
    if (t < 8 * tokens) {
      const int r = t / tokens, k = t % tokens;
      float v = 0.f;
#pragma unroll
      for (int w8 = 0; w8 < NW; ++w8) v += part[w8][r][k];
      out[k * N + row0 + r] = __float2half_rn(v);
    }
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
            dot[c] += w.apply(qv, d);
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

// One-shot TP4 gather over CUDA IPC (G3 decode). Every rank's slot r of every
// rank's buffer receives rank r's `bytes`; flags at `flag_offset` carry a
// per-device epoch (release/acquire at system scope, as in custom all-reduce).
// The MoE output all-reduce that follows bounds rank skew, so one buffer is
// enough: no rank can rewrite a slot before its owner left down_rows.
__device__ __forceinline__ void v41k_st_release(unsigned* p, unsigned v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v)
               : "memory");
}
__device__ __forceinline__ unsigned v41k_ld_acquire(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
               : "=r"(v)
               : "l"(p)
               : "memory");
  return v;
}

template <int WPD>
__global__ void __launch_bounds__(128 * WPD)
    peer_gather(uint8_t* p0, uint8_t* p1, uint8_t* p2, uint8_t* p3, int rank,
                int64_t stride, int64_t flag_offset,
                unsigned* __restrict__ epoch_counter) {
  __shared__ unsigned s_epoch;
  if (threadIdx.x == 0) s_epoch = *epoch_counter + 1;
  __syncthreads();
  const unsigned epoch = s_epoch;
  uint8_t* peers[4] = {p0, p1, p2, p3};
  const int dest = threadIdx.x / (32 * WPD), lane = threadIdx.x % (32 * WPD);
  if (dest != rank) {
    const int4* src =
        reinterpret_cast<const int4*>(peers[rank] + rank * stride);
    int4* dst = reinterpret_cast<int4*>(peers[dest] + rank * stride);
    for (int64_t i = lane; i < stride / 16; i += 32 * WPD) dst[i] = src[i];
  }
  __syncthreads();
  if (threadIdx.x < 4)
    v41k_st_release(
        reinterpret_cast<unsigned*>(peers[threadIdx.x] + flag_offset) + rank,
        epoch);
  if (threadIdx.x == 0) {
    const unsigned* own =
        reinterpret_cast<const unsigned*>(peers[rank] + flag_offset);
    for (int r = 0; r < 4; ++r)
      while (int(v41k_ld_acquire(own + r) - epoch) < 0) {
      }
    *epoch_counter = epoch;
  }
}

// ---- G4b decode indexer: scores over packed FP4 index keys ----------------
// Packed key row (68 B): 64 B E2M1 nibbles (even element in the low nibble)
// then 4 power-of-two scale exponents (biased) for groups of 32. Rows q/w are
// (token, head) pairs padded to 64. Output fp16 out[t, n] =
//   sum_h fp16(fp16(relu(fp16(q[t,h] . k[n]))) * w[t,h]), for n < live end.
__device__ __forceinline__ float v41k_e2m1(int code) {
  const int m = code & 7;
  const float mag = m < 2 ? 0.5f * m
                          : __uint_as_float(uint32_t(((m >> 1) + 126) << 23) |
                                            uint32_t((m & 1) << 22));
  return (code & 8) ? -mag : mag;
}

template <int KB>
__global__ void __launch_bounds__(256)
    index_scores_decode(const half* __restrict__ q, const half* __restrict__ w,
                        const uint8_t* __restrict__ keys,
                        const int64_t* __restrict__ positions,
                        half* __restrict__ out, int tokens, int heads,
                        int64_t n_out, int ratio) {
  __shared__ half qs[64][136];
  __shared__ float ks[KB][129];
  __shared__ float ws[64];
  const int t = threadIdx.x;
  const int64_t end = (positions[tokens - 1] + 1) / ratio;
  const int64_t n0 = int64_t(blockIdx.x) * KB;
  if (n0 >= end) return;
  const int rows = tokens * heads;
  for (int i = t; i < 64 * 128; i += 256) {
    const int r = i / 128, d = i % 128;
    qs[r][d] = r < rows ? q[r * 128 + d] : __float2half(0.f);
  }
  if (t < 64) ws[t] = t < rows ? __half2float(w[t]) : 0.f;
  for (int i = t; i < KB * 64; i += 256) {
    const int k = i / 64, b = i % 64;
    const int64_t n = n0 + k;
    float lo = 0.f, hi = 0.f;
    if (n < end) {
      const uint8_t* row = keys + n * 68;
      const int byte = row[b];
      const float scale =
          __uint_as_float(uint32_t(row[64 + (2 * b) / 32]) << 23);
      lo = v41k_e2m1(byte & 15) * scale;
      hi = v41k_e2m1((byte >> 4) & 15) * scale;
    }
    // Round like the FP16 cache values (exact for these products).
    ks[k][2 * b] = __half2float(__float2half_rn(lo));
    ks[k][2 * b + 1] = __half2float(__float2half_rn(hi));
  }
  __syncthreads();
  // Each thread: one key column (k), a stripe of (token) outputs.
  for (int item = t; item < KB * tokens; item += 256) {
    const int k = item % KB, tok = item / KB;
    const int64_t n = n0 + k;
    if (n >= end) continue;
    float total = 0.f;
    for (int h = 0; h < heads; ++h) {
      const int r = tok * heads + h;
      float dot = 0.f;
#pragma unroll 16
      for (int d = 0; d < 128; ++d)
        dot = fmaf(__half2float(qs[r][d]), ks[k][d], dot);
      float s = __half2float(__float2half_rn(dot));
      s = s > 0.f ? s : 0.f;
      total += __half2float(__float2half_rn(s * ws[r]));
    }
    out[int64_t(tok) * n_out + n] = __float2half_rn(total);
  }
}

// ---- HC transition (V4.1 mHC, 4 streams, H=5120): parts + finish ----
// parts: one CTA per 32-column chunk serves every token. thread (o = warp, c =
// lane) owns output stream o at column c; fn is read once per CTA.
template <bool MIX, int TOK>
__global__ void __launch_bounds__(128)
    hc_parts(float* __restrict__ residual, const half* __restrict__ y,
             const float* __restrict__ post, const float* __restrict__ comb,
             const float* __restrict__ fn, const float* __restrict__ cpre,
             float* __restrict__ cbuf, float* __restrict__ dots,
             float* __restrict__ squares, float* __restrict__ csquares) {
  constexpr int H = 5120, C = H / 32;
  const int chunk = blockIdx.x, o = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int col = chunk * 32 + lane;
  __shared__ float red[TOK][4][32];
  __shared__ float part[TOK][4][26];
  float r[TOK][4], yv[TOK];
  // issue every load first: all tokens' four sources, y and (once) fn
#pragma unroll
  for (int t = 0; t < TOK; ++t) {
#pragma unroll
    for (int s = 0; s < 4; ++s)
      r[t][s] = residual[(int64_t(t) * 4 + s) * H + col];
    yv[t] = __half2float(y[int64_t(t) * H + col]);
  }
  float w[MIX ? 24 : 1];
  if (MIX) {
#pragma unroll
    for (int k = 0; k < 24; ++k)
      w[k] = __ldg(fn + (int64_t(k) * 4 + o) * H + col);
  }
  __syncthreads();  // every source row read before any stream is overwritten
#pragma unroll
  for (int t = 0; t < TOK; ++t) {
    const float* cb = comb + t * 16;
    const float n = cb[o] * r[t][0] + cb[4 + o] * r[t][1] +
                    cb[8 + o] * r[t][2] + cb[12 + o] * r[t][3] +
                    post[t * 4 + o] * yv[t];
    residual[(int64_t(t) * 4 + o) * H + col] = n;
    red[t][o][lane] = n * cpre[t * 4 + o];
    if (MIX) {
      float acc[25];
#pragma unroll
      for (int k = 0; k < 24; ++k) acc[k] = w[k] * n;
      acc[24] = n * n;
#pragma unroll
      for (int k = 0; k < 25; ++k)
#pragma unroll
        for (int m = 16; m; m >>= 1)
          acc[k] += __shfl_xor_sync(0xffffffffU, acc[k], m);
      if (lane == 0)
#pragma unroll
        for (int k = 0; k < 25; ++k) part[t][o][k] = acc[k];
    }
  }
  __syncthreads();
  for (int t = o; t < TOK; t += 4) {
    const float coll =
        red[t][0][lane] + red[t][1][lane] + red[t][2][lane] + red[t][3][lane];
    cbuf[int64_t(t) * H + col] = coll;
    float q = coll * coll;
#pragma unroll
    for (int m = 16; m; m >>= 1) q += __shfl_xor_sync(0xffffffffU, q, m);
    if (lane == 0) csquares[t * C + chunk] = q;
  }
  if (MIX)
    for (int i = threadIdx.x; i < TOK * 25; i += 128) {
      const int t = i / 25, k = i % 25;
      const float v =
          part[t][0][k] + part[t][1][k] + part[t][2][k] + part[t][3][k];
      if (k < 24)
        dots[(int64_t(t) * C + chunk) * 24 + k] = v;
      else
        squares[t * C + chunk] = v;
    }
}

template <bool MIX>
__global__ void __launch_bounds__(256)
    hc_finish(const float* __restrict__ dots, const float* __restrict__ squares,
              const float* __restrict__ csquares,
              const float* __restrict__ cbuf, const float* __restrict__ norm,
              half* __restrict__ out, const float* __restrict__ scale,
              const float* __restrict__ basev, float* __restrict__ pre,
              float* __restrict__ post, float* __restrict__ comb,
              float norm_eps, float hc_eps, int iters) {
  constexpr int H = 5120, C = H / 32;
  const int t = blockIdx.x, part = blockIdx.y, tid = threadIdx.x,
            lane = tid % 32, warp = tid / 32;
  __shared__ float wsum[8][26];
  __shared__ float s_inv;
  // fixed-order reductions over the C chunk partials (independent of M)
  float cs = 0.f, sq = 0.f, d[24];
#pragma unroll
  for (int r = 0; r < 24; ++r) d[r] = 0.f;
  const bool mixer = MIX && part == 0;
  for (int c = tid; c < C; c += 256) {
    cs += csquares[t * C + c];
    if (mixer) {
      sq += squares[t * C + c];
#pragma unroll
      for (int r = 0; r < 24; ++r) d[r] += dots[(int64_t(t) * C + c) * 24 + r];
    }
  }
#pragma unroll
  for (int k = 16; k; k >>= 1) {
    cs += __shfl_xor_sync(0xffffffffU, cs, k);
    if (mixer) {
      sq += __shfl_xor_sync(0xffffffffU, sq, k);
#pragma unroll
      for (int r = 0; r < 24; ++r)
        d[r] += __shfl_xor_sync(0xffffffffU, d[r], k);
    }
  }
  if (lane == 0) {
    wsum[warp][24] = cs;
    if (mixer) {
      wsum[warp][25] = sq;
#pragma unroll
      for (int r = 0; r < 24; ++r) wsum[warp][r] = d[r];
    }
  }
  __syncthreads();
  if (tid == 0) {
    float total = 0.f;
    for (int w8 = 0; w8 < 8; ++w8) total += wsum[w8][24];
    s_inv = rsqrtf(total / H + norm_eps);
  }
  __syncthreads();
  const float inv = s_inv;
  for (int c = part * 1024 + tid; c < (part + 1) * 1024; c += 256)
    out[int64_t(t) * H + c] =
        __float2half_rn(cbuf[int64_t(t) * H + c] * inv * norm[c]);
  if (mixer && tid < 32) {
    // lane < 16 holds comb[i][j], i = lane / 4, j = lane % 4; lanes 16..23 hold
    // pre/post
    float tot = 0.f;
    for (int w8 = 0; w8 < 8; ++w8) tot += wsum[w8][25];
    const float rs = rsqrtf(tot / (4 * H) + norm_eps);
    float acc = 0.f;
    const int r = lane < 16 ? 8 + lane : (lane < 24 ? lane - 16 : 0);
    for (int w8 = 0; w8 < 8; ++w8) acc += wsum[w8][r];
    if (lane >= 16 && lane < 20)
      pre[t * 4 + lane - 16] =
          1.f / (1.f + expf(-(acc * rs * scale[0] + basev[lane - 16]))) +
          hc_eps;
    if (lane >= 20 && lane < 24)
      post[t * 4 + lane - 20] =
          2.f / (1.f + expf(-(acc * rs * scale[1] + basev[lane - 16])));
    float v = acc * rs * scale[2] + basev[r];
    const unsigned m16 = 0x0000ffffU;
    if (lane < 16) {
      // softmax over j within row i (4 lanes), then column normalize
      float mx = v;
      mx = fmaxf(mx, __shfl_xor_sync(m16, mx, 1));
      mx = fmaxf(mx, __shfl_xor_sync(m16, mx, 2));
      v = expf(v - mx);
      float rsum = v + __shfl_xor_sync(m16, v, 1);
      rsum += __shfl_xor_sync(m16, rsum, 2);
      v = v / rsum + hc_eps;
      float csum = v + __shfl_xor_sync(m16, v, 4);
      csum += __shfl_xor_sync(m16, csum, 8);
      v = v / (csum + hc_eps);
      for (int it = 1; it < iters; ++it) {
        rsum = v + __shfl_xor_sync(m16, v, 1);
        rsum += __shfl_xor_sync(m16, rsum, 2);
        v = v / (rsum + hc_eps);
        csum = v + __shfl_xor_sync(m16, v, 4);
        csum += __shfl_xor_sync(m16, csum, 8);
        v = v / (csum + hc_eps);
      }
      comb[t * 16 + lane] = v;
    }
  }
}

// ---- fp16 small-M GEMV: out[m, n] = sum_k x[m, k] * w[n, k] (M <= 8, K % 8 ==
// 0) ----
template <int M>
__global__ void __launch_bounds__(256)
    hgemv_kernel(half* __restrict__ out, const half* __restrict__ x,
                 const half* __restrict__ w, int N, int K) {
  // 8 warps x 4 rows per CTA; each lane issues all its weight loads first.
  constexpr int R = 4, MAXC = 3;  // K / 8 <= 96 chunks -> <= 3 per lane
  extern __shared__ half xs[];
  for (int i = threadIdx.x; i < M * K / 8; i += blockDim.x)
    reinterpret_cast<uint4*>(xs)[i] = reinterpret_cast<const uint4*>(x)[i];
  const int lane = threadIdx.x % 32;
  const int n0 = (blockIdx.x * 8 + threadIdx.x / 32) * R;
  const int chunks = K / 8;
  uint4 wv[R][MAXC];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int i = 0; i < MAXC; ++i) {
      const int c = lane + 32 * i;
      wv[r][i] =
          (n0 + r < N && c < chunks)
              ? __ldg(reinterpret_cast<const uint4*>(w + int64_t(n0 + r) * K) +
                      c)
              : make_uint4(0, 0, 0, 0);
    }
  __syncthreads();
  float acc[R][M] = {};
#pragma unroll
  for (int i = 0; i < MAXC; ++i) {
    const int c = lane + 32 * i;
    if (c >= chunks) break;
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const uint4 xv = reinterpret_cast<const uint4*>(xs + m * K)[c];
      const half2* xh = reinterpret_cast<const half2*>(&xv);
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const half2* wh = reinterpret_cast<const half2*>(&wv[r][i]);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 a = __half22float2(wh[j]), b = __half22float2(xh[j]);
          acc[r][m] += a.x * b.x + a.y * b.y;
        }
      }
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) {
#pragma unroll
      for (int o = 16; o; o >>= 1)
        acc[r][m] += __shfl_xor_sync(0xffffffffU, acc[r][m], o);
      if (lane == 0 && n0 + r < N)
        out[int64_t(m) * N + n0 + r] = __float2half_rn(acc[r][m]);
    }
}

// ---- fused sparse attention over gathered keys (V = K), 16 heads x 512 dims
// ---- grid (T, splits): each CTA takes 64 keys, computes S = Q K^T (fp32), a
// masked partial softmax, and the unnormalized P V partial; attn_combine folds
// splits + sink.
namespace attnw = nvcuda::wmma;
constexpr int kAH = 16, kAD = 512, kACH = 64;

__global__ void __launch_bounds__(128)
    attn_split(const half* __restrict__ q, const half* __restrict__ keys,
               const bool* __restrict__ unused, half* __restrict__ ows,
               float* __restrict__ mws, float* __restrict__ lws, int width,
               float scale) {
  constexpr int CH = 32;
  __shared__ __align__(32) half qs[kAH * kAD];     // 16 KB
  __shared__ __align__(32) float ss[2][kAH * CH];  // partial S per d-half
  __shared__ __align__(32) half ps[kAH * CH];
  __shared__ __align__(32)
      half kz[16 * kAD];  // zero rows for a short last chunk
  const int t = blockIdx.x, sp = blockIdx.y, splits = gridDim.y;
  const int tid = threadIdx.x, warp = tid / 32;
  const int k0 = sp * CH;
  const int nk = min(CH, width - k0);
  const uint4* qg = reinterpret_cast<const uint4*>(q + int64_t(t) * kAH * kAD);
  for (int i = tid; i < kAH * kAD / 8; i += 128)
    reinterpret_cast<uint4*>(qs)[i] = qg[i];
  if (nk < CH)
    for (int i = tid; i < 16 * kAD / 8; i += 128)
      reinterpret_cast<uint4*>(kz)[i] = make_uint4(0, 0, 0, 0);
  __syncthreads();
  const half* kg = keys + (int64_t(t) * width + k0) * kAD;
  // key tile kt (16 keys): global rows, or zeros when the tile runs past width
  auto ktile = [&](int kt) -> const half* {
    return kt * 16 < nk ? kg + kt * 16 * kAD : kz;
  };
  {  // S: warp -> key tile (warp & 1), d half (warp >> 1)
    const int kt = warp & 1, dh = warp >> 1;
    attnw::fragment<attnw::accumulator, 16, 16, 16, float> acc;
    attnw::fill_fragment(acc, 0.f);
    attnw::fragment<attnw::matrix_a, 16, 16, 16, half, attnw::row_major> a;
    attnw::fragment<attnw::matrix_b, 16, 16, 16, half, attnw::col_major> b;
    const half* kb = ktile(kt);
#pragma unroll 4
    for (int d = dh * 256; d < dh * 256 + 256; d += 16) {
      attnw::load_matrix_sync(a, qs + d, kAD);
      attnw::load_matrix_sync(b, kb + d, kAD);
      attnw::mma_sync(acc, a, b, acc);
    }
    attnw::store_matrix_sync(ss[dh] + kt * 16, acc, CH, attnw::mem_row_major);
  }
  __syncthreads();
  {  // masked partial softmax: 8 threads per head, 4 keys each
    const int h = tid / 8, part = tid % 8;
    float v[4], m = -INFINITY;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int key = part * 4 + j;
      const bool dead = key >= nk || unused[int64_t(t) * width + k0 + key];
      v[j] = dead ? -INFINITY
                  : (ss[0][h * CH + key] + ss[1][h * CH + key]) * scale;
      m = fmaxf(m, v[j]);
    }
#pragma unroll
    for (int o = 4; o; o >>= 1)
      m = fmaxf(m, __shfl_xor_sync(0xffffffffU, m, o));
    float l = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float e = m == -INFINITY ? 0.f : expf(v[j] - m);
      l += e;
      ps[h * CH + part * 4 + j] = __float2half_rn(e);
    }
#pragma unroll
    for (int o = 4; o; o >>= 1) l += __shfl_xor_sync(0xffffffffU, l, o);
    if (part == 0) {
      mws[(int64_t(t) * splits + sp) * kAH + h] = m;
      lws[(int64_t(t) * splits + sp) * kAH + h] = l;
    }
  }
  __syncthreads();
  {  // O partial (fp16): 32 dim tiles, 8 per warp, k = 32 keys
    attnw::fragment<attnw::matrix_a, 16, 16, 16, half, attnw::row_major> a0, a1;
    attnw::load_matrix_sync(a0, ps, CH);
    attnw::load_matrix_sync(a1, ps + 16, CH);
    attnw::fragment<attnw::matrix_b, 16, 16, 16, half, attnw::row_major> b;
    half* og = ows + (int64_t(t) * splits + sp) * kAH * kAD;
    const half* kb0 = ktile(0);
    const half* kb1 = ktile(1);
#pragma unroll 2
    for (int tile = warp; tile < kAD / 16; tile += 4) {
      attnw::fragment<attnw::accumulator, 16, 16, 16, float> acc;
      attnw::fill_fragment(acc, 0.f);
      attnw::load_matrix_sync(b, kb0 + tile * 16, kAD);
      attnw::mma_sync(acc, a0, b, acc);
      attnw::load_matrix_sync(b, kb1 + tile * 16, kAD);
      attnw::mma_sync(acc, a1, b, acc);
      attnw::fragment<attnw::accumulator, 16, 16, 16, half> hacc;
#pragma unroll
      for (int e = 0; e < acc.num_elements; ++e)
        hacc.x[e] = __float2half_rn(acc.x[e]);
      attnw::store_matrix_sync(og + tile * 16, hacc, kAD, attnw::mem_row_major);
    }
  }
}

__global__ void __launch_bounds__(256)
    attn_combine(const half* __restrict__ ows, const float* __restrict__ mws,
                 const float* __restrict__ lws, const float* __restrict__ sink,
                 half* __restrict__ out, int splits) {
  const int t = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
  float mx = sink[h];
  for (int s = 0; s < splits; ++s)
    mx = fmaxf(mx, mws[(int64_t(t) * splits + s) * kAH + h]);
  float total = expf(sink[h] - mx);
  for (int s = 0; s < splits; ++s) {
    const float m = mws[(int64_t(t) * splits + s) * kAH + h];
    total += m == -INFINITY
                 ? 0.f
                 : lws[(int64_t(t) * splits + s) * kAH + h] * expf(m - mx);
  }
  const float inv = 1.f / total;
  for (int d = tid; d < kAD; d += 256) {
    float acc = 0.f;
    for (int s = 0; s < splits; ++s) {
      const float m = mws[(int64_t(t) * splits + s) * kAH + h];
      if (m != -INFINITY)
        acc +=
            __half2float(ows[((int64_t(t) * splits + s) * kAH + h) * kAD + d]) *
            expf(m - mx);
    }
    out[(int64_t(t) * kAH + h) * kAD + d] = __float2half_rn(acc * inv);
  }
}

// One warp builds the per-layer distinct-expert plan (first-occurrence order),
// replacing the serial thread-0 prologue of every gate/up and down CTA.
__global__ void __launch_bounds__(32)
    moe_plan(int* __restrict__ plan, const int* __restrict__ ids, int routes) {
  __shared__ int s_ids[64], s_first_u[64];
  const int lane = threadIdx.x;
  for (int r = lane; r < 64; r += 32) s_ids[r] = r < routes ? ids[r] : -1;
  __syncwarp();
  bool first[2];
  int e[2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int r = lane + 32 * h;
    e[h] = s_ids[r];
    bool f = r < routes &&
             e[h] >= 0;  // id < 0: route dropped (truncated verify position)
    for (int q = 0; q < r && f; ++q) f = s_ids[q] != e[h];
    first[h] = f;
  }
  const unsigned b0 = __ballot_sync(0xffffffffU, first[0]);
  const unsigned b1 = __ballot_sync(0xffffffffU, first[1]);
  const unsigned below = (1u << lane) - 1u;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int r = lane + 32 * h;
    const int u = h == 0 ? __popc(b0 & below) : __popc(b0) + __popc(b1 & below);
    if (first[h]) s_first_u[r] = u;
  }
  __syncwarp();
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int r = lane + 32 * h;
    if (r >= routes || e[h] < 0) continue;
    int q0 = 0, c = 0;
    while (s_ids[q0] != e[h]) ++q0;  // first occurrence of this expert
    for (int q = q0; q < r; ++q) c += s_ids[q] == e[h];
    const int u = s_first_u[q0];
    if (c < 6) plan[73 + 6 * u + c] = r;
    if (first[h]) {
      int cnt = 0;
      for (int q = r; q < routes; ++q) cnt += s_ids[q] == e[h];
      plan[1 + u] = e[h];
      plan[37 + u] = cnt < 6 ? cnt : 6;
    }
  }
  if (lane == 0) plan[0] = __popc(b0) + __popc(b1);
}

// down with all row tiles of the CTA accumulated in registers and one reduction
// barrier at the end (down_em6 synchronized twice per 8-row tile). Same FP
// order.
template <int Type, int NW, int TL>
__global__ void __launch_bounds__(32 * NW, 8 / NW * 4)
    down_em8(half* __restrict__ out, const int8_t* __restrict__ hq,
             const float* __restrict__ hd, const uint8_t* __restrict__ down,
             int row_stride, int left_groups,
             const uint32_t* __restrict__ gbook, int top_k, int tokens,
             const int* __restrict__ plan) {
  constexpr int GROUPS = 18, N = 5120, BW = book_words<Type>();
  __shared__ __align__(16) uint32_t book[BW];
  __shared__ int distinct_e[36], s_nu, member_route[36][6], member_cnt[36];
  __shared__ uint32_t masks[16];
  __shared__ float part[NW][TL][8][6];
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
  const int lrow = lane / 4, sub = lane % 4;
  float acc[TL][6] = {};
  for (int q = warp; q < nu; q += NW) {
    const int e = distinct_e[q], cnt = member_cnt[q];
    int rt[6];
#pragma unroll
    for (int c = 0; c < 6; ++c) rt[c] = c < cnt ? member_route[q][c] : 0;
#pragma unroll
    for (int tile = 0; tile < TL; ++tile) {
      const int row = (blockIdx.x * TL + tile) * 8 + lrow;
      const uint8_t* base = down + (int64_t(e) * N + row) * row_stride;
      float dot[6] = {};
      for (int g = sub; g < GROUPS; g += 4) {
        const Group3<Type> w(base, g + left_groups, book, masks);
#pragma unroll
        for (int c = 0; c < 6; ++c)
          if (c < cnt) {
            int qv[8];
            float d;
            load_q8(hq, hd, int64_t(rt[c]) * GROUPS + g, qv, d);
            dot[c] += w.apply(qv, d);
          }
      }
#pragma unroll
      for (int c = 0; c < 6; ++c) {
        dot[c] += __shfl_xor_sync(0xffffffffU, dot[c], 1);
        dot[c] += __shfl_xor_sync(0xffffffffU, dot[c], 2);
      }
#pragma unroll
      for (int c = 0; c < 6; ++c)
        if (c < cnt) {
          const int token = rt[c] / top_k;
          const float v = __half2float(__float2half_rn(dot[c]));
#pragma unroll
          for (int k = 0; k < 6; ++k)
            if (k == token) acc[tile][k] += v;
        }
    }
  }
  if (sub == 0)
#pragma unroll
    for (int tile = 0; tile < TL; ++tile)
#pragma unroll
      for (int k = 0; k < 6; ++k) part[warp][tile][lrow][k] = acc[tile][k];
  __syncthreads();
  for (int i = t; i < TL * 8 * tokens; i += 32 * NW) {
    const int tile = i / (8 * tokens), r = (i / tokens) % 8, k = i % tokens;
    float v = 0.f;
#pragma unroll
    for (int w8 = 0; w8 < NW; ++w8) v += part[w8][tile][r][k];
    out[k * N + (blockIdx.x * TL + tile) * 8 + r] = __float2half_rn(v);
  }
}

// ---- H1 in-place expert swap (shared host arena, TP rank carrier banks) ----
// Plan contract (from publish): slots[op], loads[op] = incoming's host offset,
// which also receives the victim (in-place exchange). Per shard layout:
// S source row bytes, R source row start, F first byte, D device row bytes,
// W exclusive write bytes, N rows, P projection offset.
struct SwapShard {
  int64_t S, R, F, D, W, N, P, scratch;
};
struct SwapLayout {
  SwapShard sh[3];
  int64_t rows_total, scratch_per_op;
};

__device__ __forceinline__ void warp_copy(uint8_t* dst, const uint8_t* src,
                                          int64_t len, int lane) {
  const uintptr_t a = reinterpret_cast<uintptr_t>(dst) |
                      reinterpret_cast<uintptr_t>(src) | uintptr_t(len);
  if ((a & 15) == 0) {
    for (int64_t i = lane; i < len / 16; i += 32)
      reinterpret_cast<uint4*>(dst)[i] = reinterpret_cast<const uint4*>(src)[i];
  } else if ((a & 7) == 0) {
    for (int64_t i = lane; i < len / 8; i += 32)
      reinterpret_cast<uint2*>(dst)[i] = reinterpret_cast<const uint2*>(src)[i];
  } else if ((a & 3) == 0) {
    for (int64_t i = lane; i < len / 4; i += 32)
      reinterpret_cast<uint32_t*>(dst)[i] =
          reinterpret_cast<const uint32_t*>(src)[i];
  } else if ((a & 1) == 0) {
    for (int64_t i = lane; i < len / 2; i += 32)
      reinterpret_cast<uint16_t*>(dst)[i] =
          reinterpret_cast<const uint16_t*>(src)[i];
  } else {
    for (int64_t i = lane; i < len; i += 32) dst[i] = src[i];
  }
}

__device__ __forceinline__ void unit_of(const SwapLayout& L, int64_t unit,
                                        int64_t& op, int& shard, int64_t& row) {
  op = unit / L.rows_total;
  row = unit % L.rows_total;
  shard = 0;
  while (shard < 2 && row >= L.sh[shard].N) {
    row -= L.sh[shard].N;
    ++shard;
  }
}

__global__ void __launch_bounds__(256)
    swap_stage_kernel(uint8_t* b0, uint8_t* b1, uint8_t* b2,
                      const uint8_t* __restrict__ host, uint8_t* scratch,
                      const int* __restrict__ slots,
                      const int64_t* __restrict__ loads, const int* count,
                      const int* fast, SwapLayout L) {
  if (*fast != 0) return;
  const int64_t units = int64_t(*count) * L.rows_total;
  const int lane = threadIdx.x % 32;
  for (int64_t unit = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
       unit < units; unit += int64_t(gridDim.x) * blockDim.x / 32) {
    int64_t op, row;
    int shard;
    unit_of(L, unit, op, shard, row);
    const SwapShard& g = L.sh[shard];
    uint8_t* bank = shard == 0 ? b0 : (shard == 1 ? b1 : b2);
    uint8_t* dev = bank + (int64_t(slots[op]) * g.N + row) * g.D;
    // victim's exclusive bytes -> scratch, then incoming host row -> slot
    warp_copy(scratch + op * L.scratch_per_op + g.scratch + row * g.W, dev, g.W,
              lane);
    __syncwarp();
    warp_copy(dev, host + loads[op] + (row + g.R) * g.S + g.F + g.P, g.D, lane);
  }
}

__global__ void swap_signal_kernel(unsigned long long* control,
                                   const long long* counter, long long* wb_seq,
                                   const int* count, const int* fast,
                                   int flag_word) {
  if (*fast != 0 || *count == 0) return;
  const long long seq = *counter;
  wb_seq[0] = seq;
  wb_seq[1] =
      *count;  // private snapshot: the next layer may rewrite count/fast
  __threadfence_system();
  atomicExch(control + flag_word, (unsigned long long)seq);
  __threadfence_system();
}

__global__ void __launch_bounds__(256)
    swap_writeback_kernel(uint8_t* host, const uint8_t* __restrict__ scratch,
                          const int64_t* __restrict__ loads, const int* count,
                          const int* fast, const long long* wb_seq,
                          unsigned long long* control, int flag0,
                          int flag_stride, int ranks, int ack_word,
                          unsigned* done, SwapLayout L) {
  (void)count;
  (void)fast;
  const long long pending =
      reinterpret_cast<const volatile long long*>(wb_seq)[1];
  if (pending == 0) return;
  const unsigned long long seq = (unsigned long long)wb_seq[0];
  if (threadIdx.x == 0) {
    // every rank must finish reading the incoming bytes before any rank
    // overwrites them (ffn_down boundary blocks are read by two ranks).
    // Only CTA 0 polls host memory; the others wait on a device word.
    const long long begin = clock64();
    volatile unsigned* gate = done + 1;
    if (blockIdx.x == 0) {
      bool all = false;
      while (!all) {
        all = true;
        for (int r = 0; r < ranks; ++r)
          all &= *reinterpret_cast<volatile unsigned long long*>(
                     control + flag0 + r * flag_stride) >= seq;
        if (!all) __nanosleep(1000);
        if (clock64() - begin > (long long)60e9) asm volatile("trap;");
      }
      __threadfence();
      *gate = 1;
    } else {
      while (*gate == 0) {
        __nanosleep(500);
        if (clock64() - begin > (long long)60e9) asm volatile("trap;");
      }
    }
  }
  __syncthreads();
  const int64_t units = int64_t(pending) * L.rows_total;
  const int lane = threadIdx.x % 32;
  for (int64_t unit = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
       unit < units; unit += int64_t(gridDim.x) * blockDim.x / 32) {
    int64_t op, row;
    int shard;
    unit_of(L, unit, op, shard, row);
    const SwapShard& g = L.sh[shard];
    warp_copy(host + loads[op] + (row + g.R) * g.S + g.F + g.P,
              scratch + op * L.scratch_per_op + g.scratch + row * g.W, g.W,
              lane);
  }
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    if (atomicAdd(done, 1u) == gridDim.x - 1) {
      *done = 0;
      done[1] = 0;
      reinterpret_cast<volatile long long*>(const_cast<long long*>(wb_seq))[1] =
          0;  // consumed
      __threadfence_system();
      atomicExch(control + ack_word, seq);
      __threadfence_system();
    }
  }
}

// PCIe-friendly warp copies for arbitrary alignment (zero-copy host memory):
// host reads are whole aligned 16 B words (a superset of the range is read,
// harmless); host writes are aligned 16 B words in the middle and exact bytes
// only at the edges (neighbouring bytes belong to other ranks).
constexpr int kStage = 512;  // bytes staged per warp per step

__device__ __forceinline__ void warp_h2d(uint8_t* dst, const uint8_t* src,
                                         int64_t len, int lane,
                                         uint8_t* stage) {
  const uintptr_t s = reinterpret_cast<uintptr_t>(src);
  const uintptr_t a0 = s & ~uintptr_t(15), a1 = (s + len + 15) & ~uintptr_t(15);
  for (uintptr_t base = a0; base < a1; base += kStage) {
    const uintptr_t w = base + uintptr_t(lane) * 16;
    if (w < a1)
      reinterpret_cast<uint4*>(stage)[lane] =
          *reinterpret_cast<const uint4*>(w);
    __syncwarp();
    const int64_t lo =
        int64_t(base > s ? base - s : 0);  // dst index of stage[ (s+lo)-base ]
    const int64_t hi = min(len, int64_t(base + kStage - s));
    for (int64_t i = lo + lane; i < hi; i += 32) dst[i] = stage[s + i - base];
    __syncwarp();
  }
}

__device__ __forceinline__ void warp_d2h(uint8_t* dst, const uint8_t* src,
                                         int64_t len, int lane,
                                         uint8_t* stage) {
  const uintptr_t d = reinterpret_cast<uintptr_t>(dst);
  const uintptr_t a0 = (d + 15) & ~uintptr_t(15),
                  a1 = (d + len) & ~uintptr_t(15);
  if (a0 >= a1) {  // short: exact bytes
    for (int64_t i = lane; i < len; i += 32) dst[i] = src[i];
    return;
  }
  for (int64_t i = lane; i < int64_t(a0 - d); i += 32) dst[i] = src[i];  // head
  for (int64_t i = int64_t(a1 - d) + lane; i < len; i += 32)
    dst[i] = src[i];  // tail
  for (uintptr_t base = a0; base < a1; base += kStage) {
    const int64_t off = int64_t(base - d),
                  n = min(int64_t(kStage), int64_t(a1 - base));
    for (int64_t i = lane; i < n; i += 32) stage[i] = src[off + i];
    __syncwarp();
    if (lane * 16 < n)
      *reinterpret_cast<uint4*>(base + uintptr_t(lane) * 16) =
          reinterpret_cast<const uint4*>(stage)[lane];
    __syncwarp();
  }
}

// Free-slot exchange: per row, save the victim's exclusive bytes into its new
// host allocation, then read the incoming row (sources and destinations never
// alias, so no cross-rank ordering is needed). Full-duplex PCIe; ends with a
// system fence so the following commit may publish completion.
__global__ void __launch_bounds__(256)
    swap_combined_kernel(uint8_t* b0, uint8_t* b1, uint8_t* b2, uint8_t* host,
                         const int* __restrict__ slots,
                         const int64_t* __restrict__ loads,
                         const int64_t* __restrict__ saves, const int* count,
                         const int* fast, SwapLayout L) {
  if (*fast != 0) return;
  __shared__ __align__(16) uint8_t swap_stage_smem[8 * kStage];
  const int64_t units = int64_t(*count) * L.rows_total;
  const int lane = threadIdx.x % 32;
  for (int64_t unit = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
       unit < units; unit += int64_t(gridDim.x) * blockDim.x / 32) {
    int64_t op, row;
    int shard;
    unit_of(L, unit, op, shard, row);
    if (loads[op] < 0) continue;  // rescue: the expert never left its slot
    const SwapShard& g = L.sh[shard];
    uint8_t* bank = shard == 0 ? b0 : (shard == 1 ? b1 : b2);
    uint8_t* dev = bank + (int64_t(slots[op]) * g.N + row) * g.D;
    const int64_t rowoff = (row + g.R) * g.S + g.F + g.P;
    uint8_t* stage = swap_stage_smem + (threadIdx.x / 32) * kStage;
    if (saves[op] >= 0)
      warp_d2h(host + saves[op] + rowoff, dev, g.W, lane, stage);
    __syncwarp();
    warp_h2d(dev, host + loads[op] + rowoff, g.D, lane, stage);
  }
  __threadfence_system();
}

}  // namespace v41k

torch::Tensor v41k_table(int64_t type, torch::Device device) {
  int words = 0;
  if (type == 16)
    words = turbomind::gemm::LatticeCodebook<16>::kBytes / 4;
  else if (type == 17)
    words = turbomind::gemm::LatticeCodebook<17>::kBytes / 4;
  else if (type == 29)
    words = turbomind::gemm::LatticeCodebook<29>::kBytes / 4;
  else
    TORCH_CHECK(false, "type");
  auto out = torch::empty({words}, torch::dtype(torch::kInt32).device(device));
  auto stream = at::cuda::getCurrentCUDAStream();
  auto* p = reinterpret_cast<uint32_t*>(out.data_ptr());
  if (type == 16)
    v41k::build_table<16><<<(words + 255) / 256, 256, 0, stream>>>(p, words);
  if (type == 17)
    v41k::build_table<17><<<(words + 255) / 256, 256, 0, stream>>>(p, words);
  if (type == 29)
    v41k::build_table<29><<<(words + 255) / 256, 256, 0, stream>>>(p, words);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

void v41k_gate_up(torch::Tensor down_in, torch::Tensor x, torch::Tensor ids32,
                  torch::Tensor prob, torch::Tensor gate, torch::Tensor up,
                  int64_t type, torch::Tensor book, int64_t lanes) {
  const at::cuda::OptionalCUDAGuard guard(x.device());
  const int routes = ids32.numel(), top_k = 6;
  TORCH_CHECK(ids32.scalar_type() == torch::kInt32 && routes <= 36);
  const dim3 grid(18, routes);
  const size_t smem = 6 * 160 * 9 * 4;
  auto stream = at::cuda::getCurrentCUDAStream();
  auto* o = reinterpret_cast<vllm::sm70_gguf::Q8_1*>(down_in.data_ptr());
  auto* xi = reinterpret_cast<const vllm::sm70_gguf::Q8_1*>(x.data_ptr());
#define V41K_GU(T, L)                                                       \
  {                                                                         \
    auto k = v41k::gate_up_em<T, L>;                                        \
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize,    \
                         smem);                                             \
    k<<<grid, 256, smem, stream>>>(                                         \
        o, xi, ids32.data_ptr<int>(), prob.data_ptr<float>(),               \
        gate.data_ptr<uint8_t>(), up.data_ptr<uint8_t>(), gate.size(2),     \
        reinterpret_cast<const uint32_t*>(book.data_ptr()), routes, top_k); \
  }
  if (lanes == 4) {
    if (type == 16)
      V41K_GU(16, 4) else if (type == 17) V41K_GU(17, 4) else V41K_GU(29, 4)
  } else {
    TORCH_CHECK(false, "lanes");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void v41k_down(torch::Tensor out, torch::Tensor h, torch::Tensor ids32,
               torch::Tensor down, int64_t type, int64_t left,
               torch::Tensor book, torch::Tensor ws, torch::Tensor counters,
               int64_t chunks, int64_t warps) {
  const at::cuda::OptionalCUDAGuard guard(out.device());
  const int routes = ids32.numel(), tokens = out.size(0);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto* hi = reinterpret_cast<const vllm::sm70_gguf::Q8_1*>(h.data_ptr());
#define V41K_DN(T, W)                                                        \
  v41k::down_em<T, W><<<dim3(5120 / (32 * W), chunks), 32 * W, 0, stream>>>( \
      reinterpret_cast<half*>(out.data_ptr()), hi, ids32.data_ptr<int>(),    \
      down.data_ptr<uint8_t>(), down.size(2), left / 32,                     \
      reinterpret_cast<const uint32_t*>(book.data_ptr()), routes, 6, tokens, \
      ws.data_ptr<float>(), counters.data_ptr<int>())
  if (warps == 4) {
    if (type == 16)
      V41K_DN(16, 4);
    else if (type == 17)
      V41K_DN(17, 4);
    else
      V41K_DN(29, 4);
  } else if (warps == 2) {
    if (type == 16)
      V41K_DN(16, 2);
    else if (type == 17)
      V41K_DN(17, 2);
    else
      V41K_DN(29, 2);
  } else {
    if (type == 16)
      V41K_DN(16, 1);
    else if (type == 17)
      V41K_DN(17, 1);
    else
      V41K_DN(29, 1);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void v41k_gate_up2(torch::Tensor down_in, torch::Tensor x, torch::Tensor ids32,
                   torch::Tensor prob, torch::Tensor gate, torch::Tensor up,
                   int64_t type, torch::Tensor book) {
  const at::cuda::OptionalCUDAGuard guard(x.device());
  const int routes = ids32.numel();
  auto stream = at::cuda::getCurrentCUDAStream();
  auto* o = reinterpret_cast<vllm::sm70_gguf::Q8_1*>(down_in.data_ptr());
  auto* xi = reinterpret_cast<const vllm::sm70_gguf::Q8_1*>(x.data_ptr());
#define V41K_GU2(T)                                                   \
  v41k::gate_up_em2<T><<<dim3(18, routes), 512, 0, stream>>>(         \
      o, xi, ids32.data_ptr<int>(), prob.data_ptr<float>(),           \
      gate.data_ptr<uint8_t>(), up.data_ptr<uint8_t>(), gate.size(2), \
      reinterpret_cast<const uint32_t*>(book.data_ptr()), routes, 6)
  if (type == 16)
    V41K_GU2(16);
  else if (type == 17)
    V41K_GU2(17);
  else
    V41K_GU2(29);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void v41k_down2(torch::Tensor out, torch::Tensor h, torch::Tensor ids32,
                torch::Tensor down, int64_t type, int64_t left,
                torch::Tensor book) {
  const at::cuda::OptionalCUDAGuard guard(out.device());
  const int routes = ids32.numel(), tokens = out.size(0);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto* hi = reinterpret_cast<const vllm::sm70_gguf::Q8_1*>(h.data_ptr());
#define V41K_DN2(T)                                                       \
  v41k::down_em2<T><<<5120 / 8, 128, 0, stream>>>(                        \
      reinterpret_cast<half*>(out.data_ptr()), hi, ids32.data_ptr<int>(), \
      down.data_ptr<uint8_t>(), down.size(2), left / 32,                  \
      reinterpret_cast<const uint32_t*>(book.data_ptr()), routes, 6, tokens)
  if (type == 16)
    V41K_DN2(16);
  else if (type == 17)
    V41K_DN2(17);
  else
    V41K_DN2(29);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void v41k_diag(torch::Tensor sink, torch::Tensor ids32, torch::Tensor gate,
               torch::Tensor up, int64_t type, torch::Tensor book,
               int64_t mode) {
  const at::cuda::OptionalCUDAGuard guard(gate.device());
  const int routes = ids32.numel();
  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(18, routes);
  auto* bk = reinterpret_cast<const uint32_t*>(book.data_ptr());
  if (mode == 0) {
    v41k::gu_read_only<<<grid, 512, 0, stream>>>(
        sink.data_ptr<float>(), ids32.data_ptr<int>(), gate.data_ptr<uint8_t>(),
        up.data_ptr<uint8_t>(), gate.size(2), routes);
  } else {
#define V41K_DG(T)                                                             \
  v41k::gu_decode_only<T><<<grid, 512, 0, stream>>>(                           \
      sink.data_ptr<float>(), ids32.data_ptr<int>(), gate.data_ptr<uint8_t>(), \
      up.data_ptr<uint8_t>(), gate.size(2), bk, routes)
    if (type == 16)
      V41K_DG(16);
    else if (type == 17)
      V41K_DG(17);
    else
      V41K_DG(29);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void v41k_gate_up3(torch::Tensor down_in, torch::Tensor x, torch::Tensor ids32,
                   torch::Tensor prob, torch::Tensor gate, torch::Tensor up,
                   int64_t type, torch::Tensor book) {
  const at::cuda::OptionalCUDAGuard guard(x.device());
  const int routes = ids32.numel();
  auto stream = at::cuda::getCurrentCUDAStream();
  auto* o = reinterpret_cast<vllm::sm70_gguf::Q8_1*>(down_in.data_ptr());
  auto* xi = reinterpret_cast<const vllm::sm70_gguf::Q8_1*>(x.data_ptr());
#define V41K_GU3(T)                                                   \
  v41k::gate_up_em3<T><<<dim3(18, routes), 512, 0, stream>>>(         \
      o, xi, ids32.data_ptr<int>(), prob.data_ptr<float>(),           \
      gate.data_ptr<uint8_t>(), up.data_ptr<uint8_t>(), gate.size(2), \
      reinterpret_cast<const uint32_t*>(book.data_ptr()), routes, 6)
  if (type == 16)
    V41K_GU3(16);
  else if (type == 17)
    V41K_GU3(17);
  else
    V41K_GU3(29);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void v41k_down3(torch::Tensor out, torch::Tensor h, torch::Tensor ids32,
                torch::Tensor down, int64_t type, int64_t left,
                torch::Tensor book, int64_t rows_per_cta) {
  const at::cuda::OptionalCUDAGuard guard(out.device());
  const int routes = ids32.numel(), tokens = out.size(0);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto* hi = reinterpret_cast<const vllm::sm70_gguf::Q8_1*>(h.data_ptr());
  TORCH_CHECK(rows_per_cta % 8 == 0 && 5120 % rows_per_cta == 0);
#define V41K_DN3(T)                                                          \
  v41k::down_em3<T><<<5120 / rows_per_cta, 128, 0, stream>>>(                \
      reinterpret_cast<half*>(out.data_ptr()), hi, ids32.data_ptr<int>(),    \
      down.data_ptr<uint8_t>(), down.size(2), left / 32,                     \
      reinterpret_cast<const uint32_t*>(book.data_ptr()), routes, 6, tokens, \
      int(rows_per_cta))
  if (type == 16)
    V41K_DN3(16);
  else if (type == 17)
    V41K_DN3(17);
  else
    V41K_DN3(29);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void v41k_decode3(torch::Tensor sink, torch::Tensor ids32, torch::Tensor gate,
                  torch::Tensor up, int64_t type, torch::Tensor book) {
  const at::cuda::OptionalCUDAGuard guard(gate.device());
  const int routes = ids32.numel();
  auto stream = at::cuda::getCurrentCUDAStream();
  auto* bk = reinterpret_cast<const uint32_t*>(book.data_ptr());
#define V41K_D3(T)                                                             \
  v41k::gu_decode3<T><<<dim3(18, routes), 512, 0, stream>>>(                   \
      sink.data_ptr<float>(), ids32.data_ptr<int>(), gate.data_ptr<uint8_t>(), \
      up.data_ptr<uint8_t>(), gate.size(2), bk, routes)
  if (type == 16)
    V41K_D3(16);
  else if (type == 17)
    V41K_D3(17);
  else
    V41K_D3(29);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void v41k_gate_up4(torch::Tensor down_in, torch::Tensor x, torch::Tensor ids32,
                   torch::Tensor prob, torch::Tensor gate, torch::Tensor up,
                   int64_t type, torch::Tensor book) {
  const at::cuda::OptionalCUDAGuard guard(x.device());
  const int routes = ids32.numel(), tokens = x.size(0);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto* o = reinterpret_cast<vllm::sm70_gguf::Q8_1*>(down_in.data_ptr());
  auto* xi = reinterpret_cast<const vllm::sm70_gguf::Q8_1*>(x.data_ptr());
#define V41K_GU4(T)                                                        \
  {                                                                        \
    const size_t smem = 6 * 160 * 36 + v41k::book_words<T>() * 4;          \
    auto k = v41k::gate_up_em4<T>;                                         \
    static bool configured = false;                                        \
    if (!configured) {                                                     \
      cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, \
                           smem);                                          \
      configured = true;                                                   \
    }                                                                      \
    k<<<dim3(18, routes), 512, smem, stream>>>(                            \
        o, xi, ids32.data_ptr<int>(), prob.data_ptr<float>(),              \
        gate.data_ptr<uint8_t>(), up.data_ptr<uint8_t>(), gate.size(2),    \
        reinterpret_cast<const uint32_t*>(book.data_ptr()), routes, 6,     \
        tokens);                                                           \
  }
  // clang-format off
  if (type == 16) {
    V41K_GU4(16)
  } else if (type == 17) {
    V41K_GU4(17)
  } else {
    V41K_GU4(29)
  }
  // clang-format on
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void v41k_down4(torch::Tensor out, torch::Tensor h, torch::Tensor ids32,
                torch::Tensor down, int64_t type, int64_t left,
                torch::Tensor book, int64_t rows_per_cta) {
  const at::cuda::OptionalCUDAGuard guard(out.device());
  const int routes = ids32.numel(), tokens = out.size(0);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto* hi = reinterpret_cast<const vllm::sm70_gguf::Q8_1*>(h.data_ptr());
  TORCH_CHECK(rows_per_cta % 8 == 0 && 5120 % rows_per_cta == 0 &&
              routes <= 36);
#define V41K_DN4(T)                                                          \
  v41k::down_em4<T><<<5120 / rows_per_cta, 128, 0, stream>>>(                \
      reinterpret_cast<half*>(out.data_ptr()), hi, ids32.data_ptr<int>(),    \
      down.data_ptr<uint8_t>(), down.size(2), left / 32,                     \
      reinterpret_cast<const uint32_t*>(book.data_ptr()), routes, 6, tokens, \
      int(rows_per_cta))
  if (type == 16)
    V41K_DN4(16);
  else if (type == 17)
    V41K_DN4(17);
  else
    V41K_DN4(29);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void v41k_fp8_gemv(torch::Tensor out, torch::Tensor x, torch::Tensor w,
                   torch::Tensor scales, torch::Tensor partials, int64_t split,
                   c10::optional<torch::Tensor> bias) {
  const at::cuda::OptionalCUDAGuard guard(x.device());
  const int M = x.size(0), K = x.size(1), N = w.size(0);
  TORCH_CHECK(M >= 1 && M <= 8 && K % 32 == 0 && split >= 1);
  int ksplit = (K + split - 1) / split;
  ksplit = (ksplit + 15) / 16 * 16;
  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid((N + 31) / 32, split);
  const bool direct = split == 1 && !bias.has_value();
  float* part = direct ? nullptr : partials.data_ptr<float>();
  half* o = reinterpret_cast<half*>(out.data_ptr());
  auto* xi = reinterpret_cast<const half*>(x.data_ptr());
  auto* wi = reinterpret_cast<const uint8_t*>(w.data_ptr());
  auto* si = reinterpret_cast<const uint8_t*>(scales.data_ptr());
#define V41K_FP8(MM)                                   \
  v41k::fp8_gemv_kernel<MM><<<grid, 256, 0, stream>>>( \
      part, o, xi, wi, si, N, K, direct ? 1 : 0, ksplit)
  switch (M) {
    case 1:
      V41K_FP8(1);
      break;
    case 2:
      V41K_FP8(2);
      break;
    case 3:
      V41K_FP8(3);
      break;
    case 4:
      V41K_FP8(4);
      break;
    case 5:
      V41K_FP8(5);
      break;
    case 6:
      V41K_FP8(6);
      break;
    case 7:
      V41K_FP8(7);
      break;
    default:
      V41K_FP8(8);
      break;
  }
  if (!direct) {
    const int total = M * N;
    v41k::fp8_reduce<<<(total + 255) / 256, 256, 0, stream>>>(
        o, part,
        bias.has_value() ? reinterpret_cast<const half*>(bias->data_ptr())
                         : nullptr,
        M, N, split);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void v41k_q8_aligned(torch::Tensor qs, torch::Tensor ds, torch::Tensor x) {
  const at::cuda::OptionalCUDAGuard guard(x.device());
  const int M = x.size(0), K = x.size(1);
  auto stream = at::cuda::getCurrentCUDAStream();
  v41k::q8_aligned_kernel<<<dim3((K / 32 + 7) / 8, M), 256, 0, stream>>>(
      reinterpret_cast<int8_t*>(qs.data_ptr()), ds.data_ptr<float>(),
      reinterpret_cast<const half*>(x.data_ptr()), K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void v41k_moe5(torch::Tensor out, torch::Tensor xq, torch::Tensor xd,
               torch::Tensor hq, torch::Tensor hd, torch::Tensor ids32,
               torch::Tensor prob, torch::Tensor gate, torch::Tensor up,
               torch::Tensor down, int64_t gtype, int64_t dtype, int64_t left,
               torch::Tensor gbook, torch::Tensor dbook, int64_t rows) {
  const at::cuda::OptionalCUDAGuard guard(out.device());
  const int routes = ids32.numel(), tokens = out.size(0);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto i8 = [](torch::Tensor& t) {
    return reinterpret_cast<int8_t*>(t.data_ptr());
  };
#define V41K_GU5(T)                                                            \
  v41k::gate_up_em5<T><<<dim3(18, routes), 512, 0, stream>>>(                  \
      i8(hq), hd.data_ptr<float>(), i8(xq), xd.data_ptr<float>(),              \
      ids32.data_ptr<int>(), prob.data_ptr<float>(), gate.data_ptr<uint8_t>(), \
      up.data_ptr<uint8_t>(), gate.size(2),                                    \
      reinterpret_cast<const uint32_t*>(gbook.data_ptr()), routes, 6, nullptr)
  if (gtype == 16)
    V41K_GU5(16);
  else if (gtype == 17)
    V41K_GU5(17);
  else
    V41K_GU5(29);
#define V41K_DN5(T)                                                           \
  v41k::down_em5<T><<<5120 / rows, 128, 0, stream>>>(                         \
      reinterpret_cast<half*>(out.data_ptr()), i8(hq), hd.data_ptr<float>(),  \
      ids32.data_ptr<int>(), down.data_ptr<uint8_t>(), down.size(2),          \
      left / 32, reinterpret_cast<const uint32_t*>(dbook.data_ptr()), routes, \
      6, tokens, int(rows))
  if (dtype == 16)
    V41K_DN5(16);
  else if (dtype == 17)
    V41K_DN5(17);
  else
    V41K_DN5(29);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void v41k_moe6(torch::Tensor out, torch::Tensor xq, torch::Tensor xd,
               torch::Tensor hq, torch::Tensor hd, torch::Tensor ids32,
               torch::Tensor prob, torch::Tensor gate, torch::Tensor up,
               torch::Tensor down, int64_t gtype, int64_t dtype, int64_t left,
               torch::Tensor gbook, torch::Tensor dbook, int64_t rows,
               int64_t rb) {
  const at::cuda::OptionalCUDAGuard guard(out.device());
  const int routes = ids32.numel(), tokens = out.size(0);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto i8 = [](torch::Tensor& t) {
    return reinterpret_cast<int8_t*>(t.data_ptr());
  };
#define V41K_GU6(T, R)                                                         \
  v41k::gate_up_em6<T, R><<<dim3(18 / R, routes), 512, 0, stream>>>(           \
      i8(hq), hd.data_ptr<float>(), i8(xq), xd.data_ptr<float>(),              \
      ids32.data_ptr<int>(), prob.data_ptr<float>(), gate.data_ptr<uint8_t>(), \
      up.data_ptr<uint8_t>(), gate.size(2),                                    \
      reinterpret_cast<const uint32_t*>(gbook.data_ptr()), routes, 6)
#define V41K_GU6T(T) \
  if (rb == 1)       \
    V41K_GU6(T, 1);  \
  else if (rb == 2)  \
    V41K_GU6(T, 2);  \
  else if (rb == 3)  \
    V41K_GU6(T, 3);  \
  else if (rb == 6)  \
    V41K_GU6(T, 6);  \
  else               \
    V41K_GU6(T, 9)
  if (gtype == 16) {
    V41K_GU6T(16);
  } else if (gtype == 17) {
    V41K_GU6T(17);
  } else {
    V41K_GU6T(29);
  }
  if (dtype == 16)
    V41K_DN5(16);
  else if (dtype == 17)
    V41K_DN5(17);
  else
    V41K_DN5(29);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void v41k_hc_transition(
    torch::Tensor residual, torch::Tensor y, torch::Tensor post,
    torch::Tensor comb, torch::Tensor cpre, torch::Tensor norm,
    torch::Tensor out, torch::Tensor cbuf, torch::Tensor csq,
    std::optional<torch::Tensor> fn, std::optional<torch::Tensor> scale,
    std::optional<torch::Tensor> base, std::optional<torch::Tensor> dots,
    std::optional<torch::Tensor> sq, std::optional<torch::Tensor> pre_n,
    std::optional<torch::Tensor> post_n, std::optional<torch::Tensor> comb_n,
    double norm_eps, double hc_eps, int64_t iters) {
  const at::cuda::OptionalCUDAGuard guard(residual.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int tokens = residual.size(0);
  const bool mix = fn.has_value();
  auto f = [](const std::optional<torch::Tensor>& t) {
    return t ? t->data_ptr<float>() : nullptr;
  };
  const half* yp = reinterpret_cast<const half*>(y.data_ptr());
  if (mix) {
#define V41K_HCP(M, T)                                                     \
  v41k::hc_parts<M, T><<<160, 128, 0, stream>>>(                           \
      residual.data_ptr<float>(), yp, post.data_ptr<float>(),              \
      comb.data_ptr<float>(), M ? f(fn) : nullptr, cpre.data_ptr<float>(), \
      cbuf.data_ptr<float>(), M ? f(dots) : nullptr, M ? f(sq) : nullptr,  \
      csq.data_ptr<float>())
#define V41K_HCPT(M)  \
  switch (tokens) {   \
    case 1:           \
      V41K_HCP(M, 1); \
      break;          \
    case 2:           \
      V41K_HCP(M, 2); \
      break;          \
    case 3:           \
      V41K_HCP(M, 3); \
      break;          \
    case 4:           \
      V41K_HCP(M, 4); \
      break;          \
    case 5:           \
      V41K_HCP(M, 5); \
      break;          \
    case 6:           \
      V41K_HCP(M, 6); \
      break;          \
    case 7:           \
      V41K_HCP(M, 7); \
      break;          \
    default:          \
      V41K_HCP(M, 8); \
  }
    V41K_HCPT(true)
    v41k::hc_finish<true><<<dim3(tokens, 5), 256, 0, stream>>>(
        f(dots), f(sq), csq.data_ptr<float>(), cbuf.data_ptr<float>(),
        norm.data_ptr<float>(), reinterpret_cast<half*>(out.data_ptr()),
        f(scale), f(base), f(pre_n), f(post_n), f(comb_n), float(norm_eps),
        float(hc_eps), int(iters));
  } else {
    V41K_HCPT(false)
    v41k::hc_finish<false><<<dim3(tokens, 5), 256, 0, stream>>>(
        nullptr, nullptr, csq.data_ptr<float>(), cbuf.data_ptr<float>(),
        norm.data_ptr<float>(), reinterpret_cast<half*>(out.data_ptr()),
        nullptr, nullptr, nullptr, nullptr, nullptr, float(norm_eps),
        float(hc_eps), int(iters));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void v41k_moe5w(torch::Tensor out, torch::Tensor xq, torch::Tensor xd,
                torch::Tensor hq, torch::Tensor hd, torch::Tensor ids32,
                torch::Tensor prob, torch::Tensor gate, torch::Tensor up,
                torch::Tensor down, int64_t gtype, int64_t dtype, int64_t left,
                torch::Tensor gbook, torch::Tensor dbook, int64_t rows,
                int64_t dn_warps) {
  if (dn_warps == 4) {
    v41k_moe5(out, xq, xd, hq, hd, ids32, prob, gate, up, down, gtype, dtype,
              left, gbook, dbook, rows);
    return;
  }
  const at::cuda::OptionalCUDAGuard guard(out.device());
  const int routes = ids32.numel(), tokens = out.size(0);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto i8 = [](torch::Tensor& t) {
    return reinterpret_cast<int8_t*>(t.data_ptr());
  };
  if (gtype == 16)
    V41K_GU5(16);
  else if (gtype == 17)
    V41K_GU5(17);
  else
    V41K_GU5(29);
#define V41K_DN6(T, W)                                                        \
  v41k::down_em6<T, W><<<5120 / rows, 32 * W, 0, stream>>>(                   \
      reinterpret_cast<half*>(out.data_ptr()), i8(hq), hd.data_ptr<float>(),  \
      ids32.data_ptr<int>(), down.data_ptr<uint8_t>(), down.size(2),          \
      left / 32, reinterpret_cast<const uint32_t*>(dbook.data_ptr()), routes, \
      6, tokens, int(rows), nullptr)
  if (dn_warps == 8) {
    if (dtype == 16)
      V41K_DN6(16, 8);
    else if (dtype == 17)
      V41K_DN6(17, 8);
    else
      V41K_DN6(29, 8);
  } else {
    if (dtype == 16)
      V41K_DN6(16, 2);
    else if (dtype == 17)
      V41K_DN6(17, 2);
    else
      V41K_DN6(29, 2);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void v41k_hgemv(torch::Tensor out, torch::Tensor x, torch::Tensor w) {
  const at::cuda::OptionalCUDAGuard guard(x.device());
  const int M = x.size(0), K = x.size(1), N = w.size(0);
  TORCH_CHECK(M >= 1 && M <= 8 && K % 8 == 0 && K <= 768 && x.is_contiguous() &&
              w.is_contiguous());
  auto stream = at::cuda::getCurrentCUDAStream();
  const size_t smem = size_t(M) * K * 2;
  auto* o = reinterpret_cast<half*>(out.data_ptr());
  auto* xp = reinterpret_cast<const half*>(x.data_ptr());
  auto* wp = reinterpret_cast<const half*>(w.data_ptr());
  const dim3 grid((N + 31) / 32);
  switch (M) {
#define V41K_HG(MM)                                                       \
  case MM:                                                                \
    v41k::hgemv_kernel<MM><<<grid, 256, smem, stream>>>(o, xp, wp, N, K); \
    break;
    V41K_HG(1)
    V41K_HG(2) V41K_HG(3) V41K_HG(4) V41K_HG(5) V41K_HG(6) V41K_HG(7) V41K_HG(8)
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void v41k_attn_gathered(torch::Tensor out, torch::Tensor q, torch::Tensor keys,
                        torch::Tensor unused, torch::Tensor sink,
                        torch::Tensor ws, double scale) {
  const at::cuda::OptionalCUDAGuard guard(q.device());
  const int T = q.size(0), width = keys.size(1);
  TORCH_CHECK(q.size(1) == 16 && q.size(2) == 512 && keys.size(2) == 512);
  TORCH_CHECK(q.is_contiguous() && keys.is_contiguous() &&
              unused.is_contiguous());
  const int splits = (width + 31) / 32;
  // ws (fp32 elements): O partials as fp16 [T, splits, 16, 512], then m and l
  const int64_t o_floats = int64_t(T) * splits * 16 * 512 / 2;
  TORCH_CHECK(ws.numel() >= o_floats + int64_t(T) * splits * 16 * 2);
  half* ows = reinterpret_cast<half*>(ws.data_ptr<float>());
  float* mws = ws.data_ptr<float>() + o_floats;
  float* lws = mws + int64_t(T) * splits * 16;
  auto stream = at::cuda::getCurrentCUDAStream();
  v41k::attn_split<<<dim3(T, splits), 128, 0, stream>>>(
      reinterpret_cast<const half*>(q.data_ptr()),
      reinterpret_cast<const half*>(keys.data_ptr()), unused.data_ptr<bool>(),
      ows, mws, lws, width, float(scale));
  v41k::attn_combine<<<dim3(T, 16), 256, 0, stream>>>(
      ows, mws, lws, sink.data_ptr<float>(),
      reinterpret_cast<half*>(out.data_ptr()), splits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void v41k_moe7(torch::Tensor out, torch::Tensor xq, torch::Tensor xd,
               torch::Tensor hq, torch::Tensor hd, torch::Tensor ids32,
               torch::Tensor prob, torch::Tensor gate, torch::Tensor up,
               torch::Tensor down, int64_t gtype, int64_t dtype, int64_t left,
               torch::Tensor gbook, torch::Tensor dbook, int64_t rows,
               int64_t dn_warps, torch::Tensor plan) {
  const at::cuda::OptionalCUDAGuard guard(out.device());
  const int routes = ids32.numel(), tokens = out.size(0);
  TORCH_CHECK(routes <= 36 && plan.numel() >= 73 + 216);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto i8 = [](torch::Tensor& t) {
    return reinterpret_cast<int8_t*>(t.data_ptr());
  };
  int* pl = plan.data_ptr<int>();
  v41k::moe_plan<<<1, 32, 0, stream>>>(pl, ids32.data_ptr<int>(), routes);
#define V41K_GU7(T)                                                            \
  v41k::gate_up_em5<T><<<dim3(18, routes), 512, 0, stream>>>(                  \
      i8(hq), hd.data_ptr<float>(), i8(xq), xd.data_ptr<float>(),              \
      ids32.data_ptr<int>(), prob.data_ptr<float>(), gate.data_ptr<uint8_t>(), \
      up.data_ptr<uint8_t>(), gate.size(2),                                    \
      reinterpret_cast<const uint32_t*>(gbook.data_ptr()), routes, 6, pl)
  if (gtype == 16)
    V41K_GU7(16);
  else if (gtype == 17)
    V41K_GU7(17);
  else
    V41K_GU7(29);
#define V41K_DN7(T, W)                                                        \
  v41k::down_em6<T, W><<<5120 / rows, 32 * W, 0, stream>>>(                   \
      reinterpret_cast<half*>(out.data_ptr()), i8(hq), hd.data_ptr<float>(),  \
      ids32.data_ptr<int>(), down.data_ptr<uint8_t>(), down.size(2),          \
      left / 32, reinterpret_cast<const uint32_t*>(dbook.data_ptr()), routes, \
      6, tokens, int(rows), pl)
  static const bool em8 = [] {
    const char* v = std::getenv("VLLM_V41_EM8");
    return v == nullptr || v[0] != '0';
  }();
  if (em8) {
#define V41K_DN8(T, W, L)                                                    \
  v41k::down_em8<T, W, L><<<5120 / (8 * L), 32 * W, 0, stream>>>(            \
      reinterpret_cast<half*>(out.data_ptr()), i8(hq), hd.data_ptr<float>(), \
      down.data_ptr<uint8_t>(), down.size(2), left / 32,                     \
      reinterpret_cast<const uint32_t*>(dbook.data_ptr()), 6, tokens, pl)
#define V41K_DN8T(T)                    \
  if (dn_warps == 8 && rows == 16)      \
    V41K_DN8(T, 8, 2);                  \
  else if (dn_warps == 8 && rows == 32) \
    V41K_DN8(T, 8, 4);                  \
  else if (dn_warps == 4 && rows == 16) \
    V41K_DN8(T, 4, 2);                  \
  else if (dn_warps == 8)               \
    V41K_DN8(T, 8, 1);                  \
  else                                  \
    V41K_DN8(T, 4, 1)
    if (dtype == 16) {
      V41K_DN8T(16);
    } else if (dtype == 17) {
      V41K_DN8T(17);
    } else {
      V41K_DN8T(29);
    }
  } else if (dn_warps == 8) {
    if (dtype == 16)
      V41K_DN7(16, 8);
    else if (dtype == 17)
      V41K_DN7(17, 8);
    else
      V41K_DN7(29, 8);
  } else {
    if (dtype == 16)
      V41K_DN7(16, 4);
    else if (dtype == 17)
      V41K_DN7(17, 4);
    else
      V41K_DN7(29, 4);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static v41k::SwapLayout v41k_swap_layout(const std::vector<int64_t>& v) {
  TORCH_CHECK(v.size() == 21, "swap layout: 3 x (S, R, F, D, W, N, P)");
  v41k::SwapLayout L{};
  int64_t scratch = 0, rows = 0;
  for (int i = 0; i < 3; ++i) {
    auto& g = L.sh[i];
    g.S = v[7 * i];
    g.R = v[7 * i + 1];
    g.F = v[7 * i + 2];
    g.D = v[7 * i + 3];
    g.W = v[7 * i + 4];
    g.N = v[7 * i + 5];
    g.P = v[7 * i + 6];
    g.scratch = scratch;
    scratch += g.N * g.W;
    rows += g.N;
  }
  L.rows_total = rows;
  L.scratch_per_op = (scratch + 255) / 256 * 256;
  return L;
}

void v41k_swap_stage(torch::Tensor b0, torch::Tensor b1, torch::Tensor b2,
                     torch::Tensor host, torch::Tensor scratch,
                     torch::Tensor slots, torch::Tensor loads,
                     torch::Tensor count, torch::Tensor fast,
                     std::vector<int64_t> layout) {
  const at::cuda::OptionalCUDAGuard guard(b0.device());
  auto L = v41k_swap_layout(layout);
  TORCH_CHECK(scratch.numel() >= L.scratch_per_op * slots.numel(),
              "swap scratch too small");
  auto stream = at::cuda::getCurrentCUDAStream();
  v41k::swap_stage_kernel<<<1024, 256, 0, stream>>>(
      b0.data_ptr<uint8_t>(), b1.data_ptr<uint8_t>(), b2.data_ptr<uint8_t>(),
      host.data_ptr<uint8_t>(), scratch.data_ptr<uint8_t>(),
      slots.data_ptr<int>(), loads.data_ptr<int64_t>(), count.data_ptr<int>(),
      fast.data_ptr<int>(), L);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void v41k_swap_signal(torch::Tensor control, torch::Tensor counter,
                      torch::Tensor wb_seq, torch::Tensor count,
                      torch::Tensor fast, int64_t flag_word) {
  const at::cuda::OptionalCUDAGuard guard(counter.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  v41k::swap_signal_kernel<<<1, 1, 0, stream>>>(
      reinterpret_cast<unsigned long long*>(control.data_ptr<int64_t>()),
      reinterpret_cast<const long long*>(counter.data_ptr<int64_t>()),
      reinterpret_cast<long long*>(wb_seq.data_ptr<int64_t>()),
      count.data_ptr<int>(), fast.data_ptr<int>(), int(flag_word));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void v41k_swap_writeback(torch::Tensor host, torch::Tensor scratch,
                         torch::Tensor loads, torch::Tensor count,
                         torch::Tensor fast, torch::Tensor wb_seq,
                         torch::Tensor control, int64_t flag0,
                         int64_t flag_stride, int64_t ranks, int64_t ack_word,
                         torch::Tensor done, std::vector<int64_t> layout,
                         int64_t ctas) {
  const at::cuda::OptionalCUDAGuard guard(scratch.device());
  auto L = v41k_swap_layout(layout);
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(done.numel() >= 2, "swap_writeback: done needs 2 words");
  v41k::swap_writeback_kernel<<<int(ctas), 256, 0, stream>>>(
      host.data_ptr<uint8_t>(), scratch.data_ptr<uint8_t>(),
      loads.data_ptr<int64_t>(), count.data_ptr<int>(), fast.data_ptr<int>(),
      reinterpret_cast<const long long*>(wb_seq.data_ptr<int64_t>()),
      reinterpret_cast<unsigned long long*>(control.data_ptr<int64_t>()),
      int(flag0), int(flag_stride), int(ranks), int(ack_word),
      reinterpret_cast<unsigned*>(done.data_ptr<int>()), L);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int64_t v41k_swap_scratch_per_op(std::vector<int64_t> layout) {
  return v41k_swap_layout(layout).scratch_per_op;
}

void v41k_swap_combined(torch::Tensor b0, torch::Tensor b1, torch::Tensor b2,
                        torch::Tensor host, torch::Tensor slots,
                        torch::Tensor loads, torch::Tensor saves,
                        torch::Tensor count, torch::Tensor fast,
                        std::vector<int64_t> layout, int64_t ctas) {
  const at::cuda::OptionalCUDAGuard guard(b0.device());
  auto L = v41k_swap_layout(layout);
  auto stream = at::cuda::getCurrentCUDAStream();
  v41k::swap_combined_kernel<<<int(ctas), 256, 0, stream>>>(
      b0.data_ptr<uint8_t>(), b1.data_ptr<uint8_t>(), b2.data_ptr<uint8_t>(),
      host.data_ptr<uint8_t>(), slots.data_ptr<int>(),
      loads.data_ptr<int64_t>(), saves.data_ptr<int64_t>(),
      count.data_ptr<int>(), fast.data_ptr<int>(), L);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void v41k_gate_up7(torch::Tensor xq, torch::Tensor xd, torch::Tensor hq,
                   torch::Tensor hd, torch::Tensor ids32, torch::Tensor prob,
                   torch::Tensor gate, torch::Tensor up, int64_t gtype,
                   torch::Tensor gbook, torch::Tensor plan) {
  const at::cuda::OptionalCUDAGuard guard(xq.device());
  const int routes = ids32.numel();
  TORCH_CHECK(routes <= 36 && plan.numel() >= 73 + 216);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto i8 = [](torch::Tensor& t) {
    return reinterpret_cast<int8_t*>(t.data_ptr());
  };
  int* pl = plan.data_ptr<int>();
  v41k::moe_plan<<<1, 32, 0, stream>>>(pl, ids32.data_ptr<int>(), routes);
#define V41K_GU7R(T)                                                           \
  v41k::gate_up_em5<T><<<dim3(18, routes), 512, 0, stream>>>(                  \
      i8(hq), hd.data_ptr<float>(), i8(xq), xd.data_ptr<float>(),              \
      ids32.data_ptr<int>(), prob.data_ptr<float>(), gate.data_ptr<uint8_t>(), \
      up.data_ptr<uint8_t>(), gate.size(2),                                    \
      reinterpret_cast<const uint32_t*>(gbook.data_ptr()), routes, 6, pl)
  if (gtype == 16)
    V41K_GU7R(16);
  else if (gtype == 17)
    V41K_GU7R(17);
  else
    V41K_GU7R(29);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void v41k_down_rows(torch::Tensor out, torch::Tensor gathered,
                    int64_t rank_stride, int64_t routes, torch::Tensor down,
                    int64_t dtype, int64_t row_base, torch::Tensor dbook,
                    int64_t rows, int64_t dn_warps, torch::Tensor plan) {
  const at::cuda::OptionalCUDAGuard guard(out.device());
  const int tokens = out.size(0), rows_local = down.size(1);
  TORCH_CHECK(routes <= 36 && rows_local % rows == 0 && rows % 2 == 0 &&
              rank_stride % 16 == 0);
  TORCH_CHECK(gathered.numel() >= 4 * rank_stride && out.size(1) == 5120);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int sub = rows % 4 ? 16 : (rows % 8 ? 8 : 4);
  // clang-format off
#define V41K_DNR(T, W)     \
  do {                     \
    if (sub == 16) {       \
      V41K_DNR2(T, W, 16)  \
    } else if (sub == 8) { \
      V41K_DNR2(T, W, 8)   \
    } else {               \
      V41K_DNR2(T, W, 4)   \
    }                      \
  } while (0)
// clang-format on
#define V41K_DNR2(T, W, S)                                                   \
  v41k::down_rows<T, W, S><<<rows_local / rows, 32 * W, 0, stream>>>(        \
      reinterpret_cast<half*>(out.data_ptr()), gathered.data_ptr<uint8_t>(), \
      rank_stride, down.data_ptr<uint8_t>(), down.size(2), rows_local,       \
      int(row_base), reinterpret_cast<const uint32_t*>(dbook.data_ptr()),    \
      int(routes), 6, tokens, int(rows), plan.data_ptr<int>());
  if (dn_warps == 8) {
    if (dtype == 16)
      V41K_DNR(16, 8);
    else if (dtype == 17)
      V41K_DNR(17, 8);
    else
      V41K_DNR(29, 8);
  } else {
    if (dtype == 16)
      V41K_DNR(16, 4);
    else if (dtype == 17)
      V41K_DNR(17, 4);
    else
      V41K_DNR(29, 4);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void v41k_peer_gather(int64_t p0, int64_t p1, int64_t p2, int64_t p3,
                      int64_t rank, int64_t stride, int64_t flag_offset,
                      torch::Tensor epoch) {
  const at::cuda::OptionalCUDAGuard guard(epoch.device());
  TORCH_CHECK(stride % 16 == 0 && flag_offset % 16 == 0 && rank >= 0 &&
              rank < 4);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto ptr = [](int64_t v) { return reinterpret_cast<uint8_t*>(v); };
  static const int wpd = [] {
    const char* v = std::getenv("VLLM_V41_GATHER_WPD");
    return v ? std::atoi(v) : 4;
  }();
  unsigned* ep = reinterpret_cast<unsigned*>(epoch.data_ptr<int>());
  if (wpd == 1)
    v41k::peer_gather<1><<<1, 128, 0, stream>>>(
        ptr(p0), ptr(p1), ptr(p2), ptr(p3), int(rank), stride, flag_offset, ep);
  else if (wpd == 2)
    v41k::peer_gather<2><<<1, 256, 0, stream>>>(
        ptr(p0), ptr(p1), ptr(p2), ptr(p3), int(rank), stride, flag_offset, ep);
  else
    v41k::peer_gather<4><<<1, 512, 0, stream>>>(
        ptr(p0), ptr(p1), ptr(p2), ptr(p3), int(rank), stride, flag_offset, ep);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void v41k_index_scores_decode(torch::Tensor out, torch::Tensor q,
                              torch::Tensor w, torch::Tensor keys,
                              torch::Tensor positions, int64_t heads,
                              int64_t ratio) {
  const at::cuda::OptionalCUDAGuard guard(out.device());
  const int tokens = out.size(0);
  TORCH_CHECK(tokens * heads <= 64 && q.size(-1) == 128 && keys.size(1) == 68 &&
              positions.scalar_type() == torch::kInt64);
  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int KB = 32;
  const int blocks = (keys.size(0) + KB - 1) / KB;
  v41k::index_scores_decode<KB><<<blocks, 256, 0, stream>>>(
      reinterpret_cast<const half*>(q.data_ptr()),
      reinterpret_cast<const half*>(w.data_ptr()), keys.data_ptr<uint8_t>(),
      positions.data_ptr<int64_t>(), reinterpret_cast<half*>(out.data_ptr()),
      tokens, int(heads), out.size(1), int(ratio));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

TORCH_LIBRARY(v41k, m) {
  m.def(
      "index_scores_decode(Tensor(a!) out, Tensor q, Tensor w, Tensor keys, "
      "Tensor positions, "
      "int heads, int ratio) -> ()",
      &v41k_index_scores_decode);
  m.def(
      "peer_gather(int p0, int p1, int p2, int p3, int rank, int stride, int "
      "flag_offset, "
      "Tensor(a!) epoch) -> ()",
      &v41k_peer_gather);
  m.def(
      "gate_up7(Tensor xq, Tensor xd, Tensor(a!) hq, Tensor(b!) hd, Tensor "
      "ids32, Tensor prob, "
      "Tensor gate, Tensor up, int gtype, Tensor gbook, Tensor(c!) plan) -> ()",
      &v41k_gate_up7);
  m.def(
      "down_rows(Tensor(a!) out, Tensor gathered, int rank_stride, int routes, "
      "Tensor down, "
      "int dtype, int row_base, Tensor dbook, int rows, int dn_warps, Tensor "
      "plan) -> ()",
      &v41k_down_rows);
  m.def("table(int type, Device device) -> Tensor", &v41k_table);
  m.def(
      "gate_up(Tensor(a!) down_in, Tensor x, Tensor ids32, Tensor prob, "
      "Tensor gate, Tensor up, int type, Tensor book, int lanes) -> ()",
      &v41k_gate_up);
  m.def(
      "down(Tensor(a!) out, Tensor h, Tensor ids32, Tensor down, int type, "
      "int left, Tensor book, Tensor ws, Tensor counters, int chunks, "
      "int warps) -> ()",
      &v41k_down);
  m.def(
      "diag(Tensor(a!) sink, Tensor ids32, Tensor gate, Tensor up, int type, "
      "Tensor book, int mode) -> ()",
      &v41k_diag);
  m.def("q8_aligned(Tensor(a!) qs, Tensor(b!) ds, Tensor x) -> ()",
        &v41k_q8_aligned);
  m.def(
      "moe5(Tensor(a!) out, Tensor xq, Tensor xd, Tensor(b!) hq, Tensor(c!) "
      "hd, Tensor ids32, "
      "Tensor prob, Tensor gate, Tensor up, Tensor down, int gtype, int dtype, "
      "int left, "
      "Tensor gbook, Tensor dbook, int rows) -> ()",
      &v41k_moe5);
  m.def(
      "moe6(Tensor(a!) out, Tensor xq, Tensor xd, Tensor(b!) hq, Tensor(c!) "
      "hd, Tensor ids32, "
      "Tensor prob, Tensor gate, Tensor up, Tensor down, int gtype, int dtype, "
      "int left, "
      "Tensor gbook, Tensor dbook, int rows, int rb) -> ()",
      &v41k_moe6);
  m.def(
      "hc_transition(Tensor(a!) residual, Tensor y, Tensor post, Tensor comb, "
      "Tensor cpre, "
      "Tensor norm, Tensor(b!) out, Tensor(c!) cbuf, Tensor(d!) csq, Tensor? "
      "fn, Tensor? scale, "
      "Tensor? base, Tensor(e!)? dots, Tensor(f!)? sq, Tensor(g!)? pre_n, "
      "Tensor(h!)? post_n, "
      "Tensor(i!)? comb_n, float norm_eps, float hc_eps, int iters) -> ()",
      &v41k_hc_transition);
  m.def(
      "moe5w(Tensor(a!) out, Tensor xq, Tensor xd, Tensor(b!) hq, Tensor(c!) "
      "hd, Tensor ids32, "
      "Tensor prob, Tensor gate, Tensor up, Tensor down, int gtype, int dtype, "
      "int left, "
      "Tensor gbook, Tensor dbook, int rows, int dn_warps) -> ()",
      &v41k_moe5w);
  m.def("hgemv(Tensor(a!) out, Tensor x, Tensor w) -> ()", &v41k_hgemv);
  m.def(
      "attn_gathered(Tensor(a!) out, Tensor q, Tensor keys, Tensor unused, "
      "Tensor sink, "
      "Tensor(b!) ws, float scale) -> ()",
      &v41k_attn_gathered);
  m.def(
      "moe7(Tensor(a!) out, Tensor xq, Tensor xd, Tensor(b!) hq, Tensor(c!) "
      "hd, Tensor ids32, "
      "Tensor prob, Tensor gate, Tensor up, Tensor down, int gtype, int dtype, "
      "int left, "
      "Tensor gbook, Tensor dbook, int rows, int dn_warps, Tensor(d!) plan) -> "
      "()",
      &v41k_moe7);
  m.def(
      "swap_stage(Tensor(a!) b0, Tensor(b!) b1, Tensor(c!) b2, Tensor host, "
      "Tensor(d!) scratch, "
      "Tensor slots, Tensor loads, Tensor count, Tensor fast, int[] layout) -> "
      "()",
      &v41k_swap_stage);
  m.def(
      "swap_signal(Tensor(a!) control, Tensor counter, Tensor(b!) wb_seq, "
      "Tensor count, "
      "Tensor fast, int flag_word) -> ()",
      &v41k_swap_signal);
  m.def(
      "swap_writeback(Tensor(a!) host, Tensor scratch, Tensor loads, Tensor "
      "count, Tensor fast, "
      "Tensor wb_seq, Tensor(b!) control, int flag0, int flag_stride, int "
      "ranks, int ack_word, "
      "Tensor(c!) done, int[] layout, int ctas) -> ()",
      &v41k_swap_writeback);
  m.def("swap_scratch_per_op(int[] layout) -> int", &v41k_swap_scratch_per_op);
  m.def(
      "swap_combined(Tensor(a!) b0, Tensor(b!) b1, Tensor(c!) b2, Tensor(d!) "
      "host, Tensor slots, "
      "Tensor loads, Tensor saves, Tensor count, Tensor fast, int[] layout, "
      "int ctas) -> ()",
      &v41k_swap_combined);
  m.def(
      "fp8_gemv(Tensor(a!) out, Tensor x, Tensor w, Tensor scales, Tensor "
      "partials, "
      "int split, Tensor? bias) -> ()",
      &v41k_fp8_gemv);
  m.def(
      "gate_up4(Tensor(a!) down_in, Tensor x, Tensor ids32, Tensor prob, "
      "Tensor gate, Tensor up, int type, Tensor book) -> ()",
      &v41k_gate_up4);
  m.def(
      "down4(Tensor(a!) out, Tensor h, Tensor ids32, Tensor down, int type, "
      "int left, Tensor book, int rows_per_cta) -> ()",
      &v41k_down4);
  m.def(
      "gate_up3(Tensor(a!) down_in, Tensor x, Tensor ids32, Tensor prob, "
      "Tensor gate, Tensor up, int type, Tensor book) -> ()",
      &v41k_gate_up3);
  m.def(
      "down3(Tensor(a!) out, Tensor h, Tensor ids32, Tensor down, int type, "
      "int left, Tensor book, int rows_per_cta) -> ()",
      &v41k_down3);
  m.def(
      "decode3(Tensor(a!) sink, Tensor ids32, Tensor gate, Tensor up, int "
      "type, "
      "Tensor book) -> ()",
      &v41k_decode3);
  m.def(
      "gate_up2(Tensor(a!) down_in, Tensor x, Tensor ids32, Tensor prob, "
      "Tensor gate, Tensor up, int type, Tensor book) -> ()",
      &v41k_gate_up2);
  m.def(
      "down2(Tensor(a!) out, Tensor h, Tensor ids32, Tensor down, int type, "
      "int left, Tensor book) -> ()",
      &v41k_down2);
}
