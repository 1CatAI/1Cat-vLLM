// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Native GGUF -> dense_mv (dmv13) decode+pack. Byte-identical to
// vllm/model_executor/layers/quantization/gguf_dmv13_dense.py
// (decode + pack, plus compact_lut4_scale for LUT4), without the NumPy
// whole-array uint32 temporaries: one pass per 32-row tile, tiles in parallel.

#include <torch/all.h>
#include <c10/util/Half.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

namespace {

enum Format { Q4K = 0, Q5K = 1, Q6K = 2, LUT4 = 3, Q8 = 4 };
enum GType {
  T_Q8_0 = 8,
  T_Q4_K = 12,
  T_Q5_K = 13,
  T_Q6_K = 14,
  T_IQ4_NL = 20,
  T_IQ4_XS = 23
};

// Lane L of a 32-row tile holds tile row ROWMAP[L].
constexpr int kRowMap[32] = {0,  1,  2,  3,  8,  9,  10, 11, 16, 17, 18,
                             19, 24, 25, 26, 27, 4,  5,  6,  7,  12, 13,
                             14, 15, 20, 21, 22, 23, 28, 29, 30, 31};

inline float f16_to_f32(const uint8_t* p) {
  uint16_t bits;
  std::memcpy(&bits, p, 2);
  return static_cast<float>(c10::Half(bits, c10::Half::from_bits()));
}

inline uint32_t f32_to_f16_bits(float v) {
  return static_cast<uint32_t>(c10::Half(v).x);  // round to nearest even
}

struct Layout {
  int fmt, gs, block_bytes, block_elems;
};

Layout layout_of(int64_t gtype) {
  switch (gtype) {
    case T_Q4_K:
      return {Q4K, 32, 144, 256};
    case T_Q5_K:
      return {Q5K, 32, 176, 256};
    case T_Q6_K:
      return {Q6K, 16, 210, 256};
    case T_IQ4_XS:
      return {LUT4, 32, 136, 256};
    case T_IQ4_NL:
      return {LUT4, 32, 18, 32};
    case T_Q8_0:
      return {Q8, 32, 34, 32};
  }
  TORCH_CHECK(false, "dmv13 pack: unsupported GGUF type ", gtype);
}

// Decode one row into codes q[K], scales s[K/gs] and (K-quants) minima m.
void decode_row(const uint8_t* row, int64_t nb, const Layout& L, uint8_t* q,
                float* s, float* m) {
  for (int64_t bi = 0; bi < nb; ++bi) {
    const uint8_t* b = row + bi * L.block_bytes;
    uint8_t* qb = q + bi * L.block_elems;
    switch (L.fmt) {
      case Q4K:
      case Q5K: {
        const float d = f16_to_f32(b), dmin = f16_to_f32(b + 2);
        const uint8_t* sc12 = b + 4;
        uint8_t sc[8], mn[8];
        for (int j = 0; j < 4; ++j) {
          sc[j] = sc12[j] & 63;
          mn[j] = sc12[j + 4] & 63;
        }
        for (int j = 4; j < 8; ++j) {
          sc[j] = (sc12[j + 4] & 0xF) | ((sc12[j - 4] >> 6) << 4);
          mn[j] = (sc12[j + 4] >> 4) | ((sc12[j] >> 6) << 4);
        }
        for (int j = 0; j < 8; ++j) {
          s[bi * 8 + j] = d * static_cast<float>(sc[j]);
          m[bi * 8 + j] = -dmin * static_cast<float>(mn[j]);
        }
        const bool q5 = L.fmt == Q5K;
        const uint8_t* qh = q5 ? b + 16 : nullptr;
        const uint8_t* qs = q5 ? b + 48 : b + 16;
        for (int i = 0; i < 4; ++i)
          for (int k = 0; k < 32; ++k) {
            uint8_t c = qs[32 * i + k];
            uint8_t lo = c & 0xF, hi = c >> 4;
            if (q5) {
              lo |= ((qh[k] >> (2 * i)) & 1) << 4;
              hi |= ((qh[k] >> (2 * i + 1)) & 1) << 4;
            }
            qb[64 * i + k] = lo;
            qb[64 * i + 32 + k] = hi;
          }
        break;
      }
      case Q6K: {
        const uint8_t *ql = b, *qh = b + 128;
        const int8_t* sc = reinterpret_cast<const int8_t*>(b + 192);
        const float d = f16_to_f32(b + 208);
        for (int j = 0; j < 16; ++j)
          s[bi * 16 + j] = d * static_cast<float>(sc[j]);
        for (int n = 0; n < 2; ++n) {
          const uint8_t *low = ql + 64 * n, *h = qh + 32 * n;
          uint8_t* o = qb + 128 * n;
          for (int k = 0; k < 32; ++k) {
            o[k] = (low[k] & 0xF) | (((h[k] >> 0) & 3) << 4);
            o[32 + k] = (low[32 + k] & 0xF) | (((h[k] >> 2) & 3) << 4);
            o[64 + k] = (low[k] >> 4) | (((h[k] >> 4) & 3) << 4);
            o[96 + k] = (low[32 + k] >> 4) | (((h[k] >> 6) & 3) << 4);
          }
        }
        break;
      }
      case LUT4: {
        const float d = f16_to_f32(b);
        if (L.block_elems == 32) {  // IQ4_NL
          s[bi] = d;
          for (int k = 0; k < 16; ++k) {
            qb[k] = b[2 + k] & 0xF;
            qb[16 + k] = b[2 + k] >> 4;
          }
        } else {  // IQ4_XS
          const int32_t sh = b[2] | (b[3] << 8);
          for (int ib = 0; ib < 8; ++ib) {
            const int32_t ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) |
                               (((sh >> (2 * ib)) & 3) << 4);
            // NumPy promotes to float64; one rounding, same as float32 here.
            s[bi * 8 + ib] =
                static_cast<float>(static_cast<double>(d) * (ls - 32));
            const uint8_t* c = b + 8 + 16 * ib;
            for (int k = 0; k < 16; ++k) {
              qb[32 * ib + k] = c[k] & 0xF;
              qb[32 * ib + 16 + k] = c[k] >> 4;
            }
          }
        }
        break;
      }
      case Q8: {
        s[bi] = f16_to_f32(b);
        std::memcpy(qb, b + 2, 32);
        break;
      }
    }
  }
}

}  // namespace

// Returns (codes, high, scale) uint8 tensors, as gguf_dmv13_dense.pack(
// *decode(raw, gtype)); LUT4 scales are compacted like compact_lut4_scale.
std::vector<torch::Tensor> gguf_dmv13_decode_pack(torch::Tensor raw,
                                                  int64_t gtype,
                                                  int64_t num_threads) {
  TORCH_CHECK(raw.device().is_cpu() && raw.scalar_type() == at::kByte &&
                  raw.dim() == 2 && raw.is_contiguous(),
              "dmv13 pack expects a contiguous 2-D uint8 CPU tensor");
  const Layout L = layout_of(gtype);
  const int64_t N = raw.size(0), row_bytes = raw.size(1);
  TORCH_CHECK(row_bytes % L.block_bytes == 0, "dmv13 pack: partial block");
  const int64_t nb = row_bytes / L.block_bytes;
  const int64_t K = nb * L.block_elems;
  const int64_t T = (N + 31) / 32, S = (K + 31) / 32, G = (S + 3) / 4,
                Sp = G * 4;
  const int64_t spg = 32 / L.gs;  // scales per 32-wide step
  const int64_t ns = K / L.gs;    // scales per row
  const bool has_m = L.fmt == Q4K || L.fmt == Q5K;

  const int64_t code_words = (L.fmt == Q8 ? 8 : 4);
  auto u8 = torch::TensorOptions().dtype(at::kByte);
  torch::Tensor codes = torch::empty({T * Sp * 32 * code_words * 4}, u8);
  torch::Tensor high;
  if (L.fmt == Q5K)
    high = torch::empty({T * G * 32 * 4 * 4}, u8);
  else if (L.fmt == Q6K)
    high = torch::empty({T * G * 2 * 32 * 4 * 4}, u8);
  else
    high = torch::empty({0}, u8);
  const int64_t scale_words = L.fmt == LUT4 ? 2 : 4;
  torch::Tensor scale = torch::empty({T * G * 32 * scale_words * 4}, u8);

  uint32_t* C = reinterpret_cast<uint32_t*>(codes.data_ptr<uint8_t>());
  uint32_t* H = reinterpret_cast<uint32_t*>(high.data_ptr<uint8_t>());
  uint32_t* SC = reinterpret_cast<uint32_t*>(scale.data_ptr<uint8_t>());
  const uint8_t* R = raw.data_ptr<uint8_t>();

  auto do_tile = [&](int64_t t) {
    // Decode the tile's 32 rows (zero padded beyond N and K).
    std::vector<uint8_t> q(32 * Sp * 32, 0);
    std::vector<float> s(32 * Sp * spg, 0.f), m(has_m ? 32 * Sp * spg : 0, 0.f);
    for (int r = 0; r < 32; ++r) {
      const int64_t n = t * 32 + r;
      if (n >= N) break;
      decode_row(R + n * row_bytes, nb, L, &q[r * Sp * 32], &s[r * Sp * spg],
                 has_m ? &m[r * Sp * spg] : nullptr);
      (void)ns;
    }
    for (int lane = 0; lane < 32; ++lane) {
      const int r = kRowMap[lane];
      const uint8_t* qr = &q[r * Sp * 32];
      for (int64_t sp = 0; sp < Sp; ++sp) {
        const uint8_t* Q = qr + sp * 32;
        // codes
        if (L.fmt == Q8) {
          for (int c = 0; c < 8; ++c) {
            uint32_t w = 0;
            for (int b = 0; b < 4; ++b) w |= uint32_t(Q[4 * c + b]) << (8 * b);
            // (T, Sp, 2, L, 4)
            C[(((t * Sp + sp) * 2 + c / 4) * 32 + lane) * 4 + c % 4] = w;
          }
        } else {
          for (int c = 0; c < 4; ++c) {
            uint32_t w = 0;
            if (L.fmt == LUT4) {
              for (int b = 0; b < 4; ++b) {
                const uint32_t byte =
                    Q[8 * c + 2 * b] | (uint32_t(Q[8 * c + 2 * b + 1]) << 4);
                w |= byte << (8 * b);
              }
            } else {
              for (int j = 0; j < 4; ++j) {
                w |= uint32_t(Q[8 * c + 2 * j] & 0xF) << (4 * j);
                w |= uint32_t(Q[8 * c + 2 * j + 1] & 0xF) << (4 * j + 16);
              }
            }
            C[((t * Sp + sp) * 32 + lane) * 4 + c] = w;  // (T, Sp, L, 4)
          }
        }
        // high bits
        const int64_t g = sp / 4, s4 = sp % 4;
        if (L.fmt == Q5K) {
          uint32_t w = 0;
          for (int c = 0; c < 4; ++c)
            for (int j = 0; j < 4; ++j) {
              w |= uint32_t((Q[8 * c + 2 * j] >> 4) & 1) << (4 * c + j);
              w |= uint32_t((Q[8 * c + 2 * j + 1] >> 4) & 1)
                   << (16 + 4 * c + j);
            }
          H[((t * G + g) * 32 + lane) * 4 + s4] = w;  // (T, G, L, 4)
        } else if (L.fmt == Q6K) {
          uint32_t w[2] = {0, 0};
          for (int c = 0; c < 4; ++c)
            for (int j = 0; j < 4; ++j) {
              const int p = 2 * (4 * (c & 1) + j);
              w[c >> 1] |= uint32_t((Q[8 * c + 2 * j] >> 4) & 3) << p;
              w[c >> 1] |= uint32_t((Q[8 * c + 2 * j + 1] >> 4) & 3)
                           << (16 + p);
            }
          // H (T, L, Sp, 2) viewed as (T, L, G, 2, 4) -> (T, G, a, L, b)
          for (int half = 0; half < 2; ++half) {
            const int64_t idx = s4 * 2 + half;  // position within the group
            const int64_t a = idx / 4, bb = idx % 4;
            H[(((t * G + g) * 2 + a) * 32 + lane) * 4 + bb] = w[half];
          }
        }
        // scale word
        const float* sr = &s[r * Sp * spg + sp * spg];
        uint32_t w;
        const uint32_t s16 = f32_to_f16_bits(sr[0]);
        if (has_m) {
          w = s16 | (f32_to_f16_bits(m[r * Sp * spg + sp]) << 16);
        } else if (L.fmt == Q6K) {
          w = s16 | (f32_to_f16_bits(sr[1]) << 16);
        } else {
          w = s16 | (s16 << 16);
        }
        if (L.fmt == LUT4) {
          // compact_lut4_scale: four duplicated halves -> two half2 words.
          uint16_t* out =
              reinterpret_cast<uint16_t*>(&SC[((t * G + g) * 32 + lane) * 2]);
          out[s4] = static_cast<uint16_t>(s16);
        } else {
          SC[((t * G + g) * 32 + lane) * 4 + s4] = w;  // (T, G, L, 4)
        }
      }
    }
  };

  const int64_t workers =
      std::max<int64_t>(1, std::min<int64_t>(num_threads, T));
  std::vector<std::thread> pool;
  for (int64_t w = 0; w < workers; ++w)
    pool.emplace_back([&, w] {
      for (int64_t t = w; t < T; t += workers) do_tile(t);
    });
  for (auto& th : pool) th.join();
  return {codes, high, scale};
}
