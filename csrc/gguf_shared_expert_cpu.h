// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

// CPU execution of cold decode experts (VLLM_V41_CPU_EXPERTS). The owner
// lane computes chosen cold experts straight from the raw GGUF bytes in the
// shared host arena while the GPUs transfer the others and run resident
// experts, so a miss costs neither uplink bandwidth nor an eviction. Each
// 256-weight IQ2_XXS / IQ2_XS / IQ1_M block is unpacked once per 32-weight
// group into unsigned magnitudes, a sign carrier and odd per-16 scales
// (w = d / 8 * s16 * v, the ggml formats), then every token does
// sign / maddubs / madd against Q8_K activations. The expert matches the GPU
// semantics (clamped SwiGLU times the route probability rounded to FP16,
// then down); activations use Q8_K blocks of 256 instead of Q8_1 blocks of
// 32, so results are close but not bit-identical.
#include "gguf_cpu_iq_tables.h"

#include <immintrin.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

#pragma GCC push_options
#pragma GCC target("avx2,fma,f16c")

namespace vllm::gguf {

#define V41_CPU_TARGET

namespace cpu_moe {

struct BlockIq2xxs {
  uint16_t d;
  uint16_t qs[32];
};
struct BlockIq2xs {
  uint16_t d;
  uint16_t qs[32];
  uint8_t scales[8];
};
struct BlockIq1m {
  uint8_t qs[32];
  uint8_t qh[16];
  uint8_t scales[8];
};
struct BlockQ8K {
  float d;
  int8_t qs[256];
  int16_t bsums[16];
};
static_assert(sizeof(BlockIq2xxs) == 66 && sizeof(BlockIq2xs) == 74 &&
              sizeof(BlockIq1m) == 56);

inline size_t block_bytes(int type) {
  switch (type) {
    case 16:
      return sizeof(BlockIq2xxs);
    case 17:
      return sizeof(BlockIq2xs);
    case 29:
      return sizeof(BlockIq1m);
    default:
      return 0;
  }
}

inline const uint64_t* sign64() {
  static const auto table = [] {
    std::vector<uint64_t> t(128);
    for (int i = 0; i < 128; ++i) {
      uint64_t v = 0;
      for (int j = 0; j < 8; ++j)
        v |= uint64_t(cpu_iq::ksigns_iq2xs[i] & cpu_iq::kmask_iq2xs[j] ? 0xFF
                                                                       : 0x01)
             << (8 * j);
      t[i] = v;
    }
    return t;
  }();
  return table.data();
}

V41_CPU_TARGET inline float h2f(uint16_t h) { return _cvtsh_ss(h); }

// IQ1_M grid shift per qh byte: two 8-weight groups (bits 3 and 7), +1 / -1
inline const __m128i* iq1m_delta() {
  static const auto table = [] {
    std::vector<__m128i> t(256);
    for (int q = 0; q < 256; ++q)
      t[q] = _mm_set_epi64x(q & 0x80 ? -1LL : 0x0101010101010101LL,
                            q & 0x08 ? -1LL : 0x0101010101010101LL);
    return t;
  }();
  return table.data();
}

// ggml quantize_row_q8_K_ref
inline void quantize_q8k(const float* x, BlockQ8K* y, int k) {
  for (int i = 0; i < k / 256; ++i, x += 256) {
    float max = 0, amax = 0;
    for (int j = 0; j < 256; ++j)
      if (std::fabs(x[j]) > amax) {
        amax = std::fabs(x[j]);
        max = x[j];
      }
    if (!amax) {
      std::memset(&y[i], 0, sizeof(BlockQ8K));
      continue;
    }
    const float iscale = -127.f / max;
    for (int j = 0; j < 256; ++j) {
      const int v = int(std::nearbyint(iscale * x[j]));
      y[i].qs[j] = int8_t(v > 127 ? 127 : v);
    }
    for (int j = 0; j < 16; ++j) {
      int s = 0;
      for (int l = 0; l < 16; ++l) s += y[i].qs[16 * j + l];
      y[i].bsums[j] = int16_t(s);
    }
    y[i].d = 1 / iscale;
  }
}

struct GroupIq2xxs {
  using Block = BlockIq2xxs;
  V41_CPU_TARGET static inline float d(const Block* x) { return h2f(x->d); }
  V41_CPU_TARGET static inline void group(const Block* x, int ib, __m256i& u,
                                          __m256i& g, __m256i& sc) {
    uint32_t a[2];
    std::memcpy(a, x->qs + 4 * ib, 8);
    const uint8_t* i8 = reinterpret_cast<const uint8_t*>(a);
    const uint64_t* s = sign64();
    u = _mm256_set_epi64x(
        cpu_iq::iq2xxs_grid[i8[3]], cpu_iq::iq2xxs_grid[i8[2]],
        cpu_iq::iq2xxs_grid[i8[1]], cpu_iq::iq2xxs_grid[i8[0]]);
    g = _mm256_set_epi64x(s[(a[1] >> 21) & 127], s[(a[1] >> 14) & 127],
                          s[(a[1] >> 7) & 127], s[a[1] & 127]);
    sc = _mm256_set1_epi16(int16_t(2 * (a[1] >> 28) + 1));
  }
};

struct GroupIq2xs {
  using Block = BlockIq2xs;
  V41_CPU_TARGET static inline float d(const Block* x) { return h2f(x->d); }
  V41_CPU_TARGET static inline void group(const Block* x, int ib, __m256i& u,
                                          __m256i& g, __m256i& sc) {
    const uint16_t* q = x->qs + 4 * ib;
    const uint64_t* s = sign64();
    u = _mm256_set_epi64x(
        cpu_iq::iq2xs_grid[q[3] & 511], cpu_iq::iq2xs_grid[q[2] & 511],
        cpu_iq::iq2xs_grid[q[1] & 511], cpu_iq::iq2xs_grid[q[0] & 511]);
    g = _mm256_set_epi64x(s[q[3] >> 9], s[q[2] >> 9], s[q[1] >> 9],
                          s[q[0] >> 9]);
    sc = _mm256_set_m128i(
        _mm_set1_epi16(int16_t(2 * (x->scales[ib] >> 4) + 1)),
        _mm_set1_epi16(int16_t(2 * (x->scales[ib] & 0xf) + 1)));
  }
};

struct GroupIq1m {
  using Block = BlockIq1m;
  V41_CPU_TARGET static inline float d(const Block* x) {
    uint16_t sc[4];
    std::memcpy(sc, x->scales, 8);
    return h2f(uint16_t((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
                        ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000)));
  }
  V41_CPU_TARGET static inline void group(const Block* x, int ib, __m256i& u,
                                          __m256i& g, __m256i& sc) {
    uint16_t s[4];
    std::memcpy(s, x->scales, 8);
    const uint8_t* qs = x->qs + 4 * ib;
    const uint8_t* qh = x->qh + 2 * ib;
    const __m256i grid =
        _mm256_set_epi64x(cpu_iq::iq1s_grid[qs[3] | ((qh[1] << 4) & 0x700)],
                          cpu_iq::iq1s_grid[qs[2] | ((qh[1] << 8) & 0x700)],
                          cpu_iq::iq1s_grid[qs[1] | ((qh[0] << 4) & 0x700)],
                          cpu_iq::iq1s_grid[qs[0] | ((qh[0] << 8) & 0x700)]);
    const __m128i* dt = iq1m_delta();
    const __m256i delta = _mm256_set_m128i(dt[qh[1]], dt[qh[0]]);
    // (grid + delta) * 8 = 8 * grid +- 1 (grid in {-1, 0, 1}); never zero
    g = _mm256_add_epi8(_mm256_sign_epi8(_mm256_set1_epi8(8), grid), delta);
    u = _mm256_abs_epi8(g);
    sc = _mm256_set_m128i(
        _mm_set1_epi16(
            int16_t(2 * ((s[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1)),
        _mm_set1_epi16(
            int16_t(2 * ((s[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7) + 1)));
  }
};

V41_CPU_TARGET inline float hsum8(__m256 v) {
  __m128 x = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  x = _mm_add_ps(x, _mm_movehl_ps(x, x));
  x = _mm_add_ss(x, _mm_movehdup_ps(x));
  return _mm_cvtss_f32(x);
}

// out[m * ldo + r] = row r . q[m] for rows [r0, r1) of a row-major matrix of
// nb blocks per row.
template <class G, int M>
V41_CPU_TARGET void rows(const uint8_t* w, size_t row_bytes, int nb, int r0,
                         int r1, const BlockQ8K* const* q, float* out,
                         int ldo) {
  using Block = typename G::Block;
  for (int r = r0; r < r1; ++r) {
    const Block* row =
        reinterpret_cast<const Block*>(w + size_t(r) * row_bytes);
    __m256 accf[M];
    for (int m = 0; m < M; ++m) accf[m] = _mm256_setzero_ps();
    for (int b = 0; b < nb; ++b) {
      const Block* xb = row + b;
      __m256i acc[M];
      for (int m = 0; m < M; ++m) acc[m] = _mm256_setzero_si256();
      for (int ib = 0; ib < 8; ++ib) {
        __m256i u, g, sc;
        G::group(xb, ib, u, g, sc);
        for (int m = 0; m < M; ++m) {
          const __m256i q8 = _mm256_loadu_si256(
              reinterpret_cast<const __m256i*>(q[m][b].qs + 32 * ib));
          acc[m] = _mm256_add_epi32(
              acc[m],
              _mm256_madd_epi16(
                  _mm256_maddubs_epi16(u, _mm256_sign_epi8(q8, g)), sc));
        }
      }
      const float dw = G::d(xb);
      for (int m = 0; m < M; ++m)
        accf[m] = _mm256_fmadd_ps(_mm256_set1_ps(dw * q[m][b].d),
                                  _mm256_cvtepi32_ps(acc[m]), accf[m]);
    }
    for (int m = 0; m < M; ++m) out[m * ldo + r] = 0.125f * hsum8(accf[m]);
  }
}

template <class G>
void rows_any(int m, const uint8_t* w, size_t row_bytes, int nb, int r0, int r1,
              const BlockQ8K* const* q, float* out, int ldo) {
  switch (m) {
    case 1:
      return rows<G, 1>(w, row_bytes, nb, r0, r1, q, out, ldo);
    case 2:
      return rows<G, 2>(w, row_bytes, nb, r0, r1, q, out, ldo);
    case 3:
      return rows<G, 3>(w, row_bytes, nb, r0, r1, q, out, ldo);
    case 4:
      return rows<G, 4>(w, row_bytes, nb, r0, r1, q, out, ldo);
    case 5:
      return rows<G, 5>(w, row_bytes, nb, r0, r1, q, out, ldo);
    case 6:
      return rows<G, 6>(w, row_bytes, nb, r0, r1, q, out, ldo);
    case 7:
      return rows<G, 7>(w, row_bytes, nb, r0, r1, q, out, ldo);
    default:
      return rows<G, 8>(w, row_bytes, nb, r0, r1, q, out, ldo);
  }
}

inline void dot_rows(int type, int m, const uint8_t* w, size_t row_bytes,
                     int nb, int r0, int r1, const BlockQ8K* const* q,
                     float* out, int ldo) {
  if (type == 16)
    rows_any<GroupIq2xxs>(m, w, row_bytes, nb, r0, r1, q, out, ldo);
  else if (type == 17)
    rows_any<GroupIq2xs>(m, w, row_bytes, nb, r0, r1, q, out, ldo);
  else
    rows_any<GroupIq1m>(m, w, row_bytes, nb, r0, r1, q, out, ldo);
}

}  // namespace cpu_moe

// One cold expert of one plan: its raw bytes and the verify tokens routed to
// it.
struct SharedCpuExpert {
  const uint8_t* bytes;  // gate rows, up rows, down rows (original GGUF layout)
  int kinds[3];
  int tokens;
  int token[8];
  float weight[8];
};

// Persistent spinning pool. run() is called by the owner coordinator lane,
// which also works as thread 0; no CUDA calls are made here.
class SharedExpertCpu {
 public:
  static constexpr int kMaxTokens = 8;

  SharedExpertCpu(int hidden, int intermediate, int threads)
      : hidden_(hidden),
        inter_(intermediate),
        threads_(threads < 1 ? 1 : threads) {
    if (hidden % 256 || intermediate % 256 || hidden > 8192 ||
        intermediate > 8192)
      throw std::invalid_argument("CPU experts need 256-multiple dimensions");
    if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma") ||
        !__builtin_cpu_supports("f16c"))
      throw std::runtime_error("CPU experts need AVX2/FMA/F16C");
    qx_.resize(kMaxTokens * (hidden / 256));
    parts_.resize(threads_);
    for (auto& p : parts_) p.resize(size_t(kMaxTokens) * hidden);
    cpu_moe::sign64();
    cpu_moe::iq1m_delta();
    for (int t = 1; t < threads_; ++t)
      pool_.emplace_back([this, t] { worker(t); });
  }

  ~SharedExpertCpu() {
    stop_.store(true, std::memory_order_release);
    generation_.fetch_add(1, std::memory_order_acq_rel);
    for (auto& t : pool_) t.join();
  }

  // x: [tokens, hidden] FP16 (host); y: [tokens, hidden] FP16 (host), the sum
  // over experts of weight * expert(x[token]) for the routed tokens.
  void run(const uint16_t* x, int tokens,
           const std::vector<SharedCpuExpert>& experts, uint16_t* y) {
    if (tokens < 1 || tokens > kMaxTokens)
      throw std::invalid_argument("CPU expert tokens");
    x_ = x;
    y_ = y;
    tokens_ = tokens;
    experts_ = &experts;
    const size_t n = experts.size();
    if (gate_.size() < n * kMaxTokens * inter_) {
      gate_.resize(n * kMaxTokens * inter_);
      up_.resize(n * kMaxTokens * inter_);
      qh_.resize(n * kMaxTokens * (inter_ / 256));
    }
    if (acc_.size() < size_t(kMaxTokens) * hidden_)
      acc_.resize(size_t(kMaxTokens) * hidden_);
    const uint64_t job =
        generation_.fetch_add(1, std::memory_order_acq_rel) + 1;
    work(0, job);
  }

 private:
  // Arrivals only grow: job j phase p completes at ((j - 1) * 4 + p) * threads.
  void barrier(uint64_t job, int phase) {
    const uint64_t target = ((job - 1) * 4 + phase) * uint64_t(threads_);
    arrived_.fetch_add(1, std::memory_order_acq_rel);
    while (arrived_.load(std::memory_order_acquire) < target) _mm_pause();
  }

  void worker(int id) {
    uint64_t seen =
        0;  // jobs are numbered from 1; never miss one posted before start
    auto idle_since = std::chrono::steady_clock::now();
    while (true) {
      const uint64_t now = generation_.load(std::memory_order_acquire);
      if (now != seen) {
        seen = now;
        if (stop_.load(std::memory_order_acquire)) return;
        work(id, now);
        idle_since = std::chrono::steady_clock::now();
        continue;
      }
      if (std::chrono::steady_clock::now() - idle_since <
          std::chrono::milliseconds(4))
        _mm_pause();
      else
        std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
  }

  void work(int id, uint64_t job) {
    const auto& ex = *experts_;
    const int n = int(ex.size());
    const int hb = hidden_ / 256, ib = inter_ / 256;
    // phase 1: quantize the input tokens
    for (int t = id; t < tokens_; t += threads_) {
      float row[8192];
      for (int c = 0; c < hidden_; ++c)
        row[c] = cpu_moe::h2f(x_[size_t(t) * hidden_ + c]);
      cpu_moe::quantize_q8k(row, qx_.data() + t * hb, hidden_);
    }
    barrier(job, 1);
    // phase 2: gate and up rows of every expert (rows split across threads)
    {
      const int total = 2 * n * inter_;
      const int r0 = int(int64_t(total) * id / threads_),
                r1 = int(int64_t(total) * (id + 1) / threads_);
      for (int at = r0; at < r1;) {
        const int unit = at / inter_, e = unit / 2, mat = unit % 2,
                  row = at % inter_;
        const int stop = std::min(r1, (unit + 1) * inter_) - unit * inter_;
        const auto& x = ex[e];
        const cpu_moe::BlockQ8K* q[kMaxTokens];
        for (int m = 0; m < x.tokens; ++m) q[m] = qx_.data() + x.token[m] * hb;
        const size_t rb = hb * cpu_moe::block_bytes(x.kinds[mat]);
        const uint8_t* w =
            x.bytes +
            (mat ? size_t(inter_) * hb * cpu_moe::block_bytes(x.kinds[0]) : 0);
        float* out =
            (mat ? up_ : gate_).data() + size_t(e) * kMaxTokens * inter_;
        cpu_moe::dot_rows(x.kinds[mat], x.tokens, w, rb, hb, row, stop, q, out,
                          inter_);
        at = unit * inter_ + stop;
      }
    }
    barrier(job, 2);
    // phase 3: clamped SwiGLU x probability (FP16 rounding), quantize per
    // (expert, token)
    for (int job = id; job < n * kMaxTokens; job += threads_) {
      const int e = job / kMaxTokens, m = job % kMaxTokens;
      if (m >= ex[e].tokens) continue;
      float h[8192];
      const float* g = gate_.data() + (size_t(e) * kMaxTokens + m) * inter_;
      const float* u = up_.data() + (size_t(e) * kMaxTokens + m) * inter_;
      const float p = ex[e].weight[m];
      for (int c = 0; c < inter_; ++c) {
        const float gc = std::min(g[c], 10.f),
                    uc = std::max(std::min(u[c], 10.f), -10.f);
        const float v = gc / (1.f + std::exp(-gc)) * uc * p;
        h[c] = _cvtsh_ss(_cvtss_sh(v, 0));
      }
      cpu_moe::quantize_q8k(h, qh_.data() + (size_t(e) * kMaxTokens + m) * ib,
                            inter_);
    }
    barrier(job, 3);
    // phase 4: down rows (hidden rows split across threads), summed over
    // experts
    {
      const int r0 = int(int64_t(hidden_) * id / threads_),
                r1 = int(int64_t(hidden_) * (id + 1) / threads_);
      std::vector<float>& part = parts_[id];
      for (int t = 0; t < tokens_; ++t)
        std::memset(acc_.data() + size_t(t) * hidden_ + r0, 0,
                    sizeof(float) * (r1 - r0));
      for (int e = 0; e < n; ++e) {
        const auto& x = ex[e];
        const cpu_moe::BlockQ8K* q[kMaxTokens];
        for (int m = 0; m < x.tokens; ++m)
          q[m] = qh_.data() + (size_t(e) * kMaxTokens + m) * ib;
        const size_t gu = size_t(inter_) * hb *
                          (cpu_moe::block_bytes(x.kinds[0]) +
                           cpu_moe::block_bytes(x.kinds[1]));
        cpu_moe::dot_rows(x.kinds[2], x.tokens, x.bytes + gu,
                          ib * cpu_moe::block_bytes(x.kinds[2]), ib, r0, r1, q,
                          part.data(), hidden_);
        for (int m = 0; m < x.tokens; ++m)
          for (int r = r0; r < r1; ++r)
            acc_[size_t(x.token[m]) * hidden_ + r] +=
                part[size_t(m) * hidden_ + r];
      }
      for (int t = 0; t < tokens_; ++t)
        for (int r = r0; r < r1; ++r)
          y_[size_t(t) * hidden_ + r] =
              _cvtss_sh(acc_[size_t(t) * hidden_ + r], 0);
    }
    barrier(job, 4);
  }

  const int hidden_, inter_, threads_;
  std::vector<std::thread> pool_;
  std::atomic<uint64_t> generation_{0};
  std::atomic<uint64_t> arrived_{0};
  std::atomic<bool> stop_{false};
  const uint16_t* x_ = nullptr;
  uint16_t* y_ = nullptr;
  int tokens_ = 0;
  const std::vector<SharedCpuExpert>* experts_ = nullptr;
  std::vector<cpu_moe::BlockQ8K> qx_, qh_;
  std::vector<float> gate_, up_, acc_;
  std::vector<std::vector<float>> parts_;
};

}  // namespace vllm::gguf

#pragma GCC pop_options
