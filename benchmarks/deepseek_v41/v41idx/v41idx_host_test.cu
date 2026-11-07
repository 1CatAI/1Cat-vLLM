// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// v41idx_host_test.cu: CPU-only proof that the VLLM_V41_INDEX_TILED kernel
// bodies (stage_q / stage_keys / tile from v41idx.cu, compiled for the HOST,
// run with the kernel's own persistent-CTA schedule) give bitwise the same
// output as the public reference v41k::index_scores_decode.
//
// Expected values do not come from v41idx.cu:
//   1. key decode: every (nibble byte, scale byte) pair against the OCP MX
//      E2M1 table {0, .5, 1, 1.5, 2, 3, 4, 6} (sign bit 8) and the IEEE meaning
//      of the scale byte as an FP32 exponent field (0 -> 0, 255 -> inf);
//   2. scores: ref_scores(), a line-by-line transliteration of the public reference
//      kernel's loops (v41k.cu index_scores_decode: decode into float ks, one
//      fmaf chain per (row, key), relu/weight/head-sum epilogue), with its own
//      decode from item 1.
// Build and run (no GPU, no kernel launch):
//   nvcc -std=c++17 -O2 -DV41IDX_HOST_TEST v41idx_host_test.cu -o host_test && ./host_test
//   nvcc ... -DV41IDX_MUTANT_ORDER ... && ./host_test --expect-mismatch   (mutant must be caught)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

#include "v41idx.cu"

using v41idx::KB;
using v41idx::KEY_BYTES;

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint64_t rnd() {
  uint64_t x = rng_state;
  x ^= x << 13;
  x ^= x >> 7;
  x ^= x << 17;
  return rng_state = x;
}
static float uni() { return float(rnd() >> 40) / float(1ull << 24); }  // [0, 1)
static float gauss() {
  float u = uni() + 1e-7f, v = uni();
  return std::sqrt(-2.f * std::log(u)) * std::cos(6.2831853f * v);
}
static uint16_t hbits(__half h) {
  uint16_t u;
  memcpy(&u, &h, 2);
  return u;
}
static __half hfrom(uint16_t u) {
  __half h;
  memcpy(&h, &u, 2);
  return h;
}

// --- independent spec decode -------------------------------------------
static float spec_e2m1(int code) {
  static const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  const float m = mag[code & 7];
  return (code & 8) ? -m : m;  // code 8 -> -0.0f
}
static float spec_scale(int e) {
  if (e == 0) return 0.f;          // exponent field 0, mantissa 0
  if (e == 255) return INFINITY;   // exponent field all ones, mantissa 0
  return std::ldexp(1.f, e - 127);
}

// --- transliteration of the public reference kernel (v41k.cu, G4b decode indexer) --
static void ref_scores(const std::vector<__half>& q, const std::vector<__half>& w,
                       const std::vector<uint8_t>& keys, int64_t cap,
                       const std::vector<int64_t>& positions, std::vector<__half>& out,
                       int tokens, int heads, int64_t n_out, int ratio) {
  const int64_t end = (positions[tokens - 1] + 1) / ratio;
  const int RB = 32;
  const int64_t blocks = (cap + RB - 1) / RB;
  for (int64_t blk = 0; blk < blocks; ++blk) {
    const int64_t n0 = blk * RB;
    if (n0 >= end) continue;
    for (int k = 0; k < RB; ++k) {
      const int64_t n = n0 + k;
      if (n >= end) continue;
      float ks[128];
      for (int b = 0; b < 64; ++b) {
        const uint8_t* row = keys.data() + n * 68;
        const int byte = row[b];
        const float scale = spec_scale(row[64 + (2 * b) / 32]);
        const float lo = spec_e2m1(byte & 15) * scale;
        const float hi = spec_e2m1((byte >> 4) & 15) * scale;
        ks[2 * b] = __half2float(__float2half_rn(lo));
        ks[2 * b + 1] = __half2float(__float2half_rn(hi));
      }
      for (int tok = 0; tok < tokens; ++tok) {
        float total = 0.f;
        for (int h = 0; h < heads; ++h) {
          const int r = tok * heads + h;
          float dot = 0.f;
          for (int d = 0; d < 128; ++d)
            dot = std::fma(__half2float(q[r * 128 + d]), ks[d], dot);
          float s = __half2float(__float2half_rn(dot));
          s = s > 0.f ? s : 0.f;
          total += __half2float(__float2half_rn(s * __half2float(w[r])));
        }
        out[tok * n_out + n] = __float2half_rn(total);
      }
    }
  }
}

// --- the kernel bodies, run with the kernel's schedule ---------------------
template <int H>
static void tiled_host(const std::vector<__half>& q, const std::vector<__half>& w,
                       const std::vector<uint8_t>& keys, int64_t cap,
                       const std::vector<int64_t>& positions, std::vector<__half>& out,
                       int tokens, int64_t n_out, int ratio, int grid_cap) {
  const int64_t end = v41idx::live_end(positions[tokens - 1], ratio, cap);
  const int rows = tokens * H;
  const int nt = ((tokens * v41idx::KL + 31) / 32) * 32;
  const int64_t blocks = (cap + KB - 1) / KB;
  const int64_t grid = blocks < grid_cap ? blocks : grid_cap;
  std::vector<float> buf((v41idx::smem_bytes(rows) + 15) / 4 + 4);
  unsigned char* base = reinterpret_cast<unsigned char*>(buf.data());
  for (int64_t cta = 0; cta < grid; ++cta) {
    int64_t n0 = cta * KB;
    if (n0 >= end) continue;
    // fresh, poisoned shared memory per CTA
    for (auto& f : buf) f = NAN;
    const v41idx::Smem s = v41idx::carve(base, rows);
    for (int tid = 0; tid < nt; ++tid)
      v41idx::stage_q(tid, nt, q.data(), w.data(), rows, s.qs, s.ws);
    for (; n0 < end; n0 += grid * KB) {
      for (int tid = 0; tid < nt; ++tid)
        v41idx::stage_keys(tid, nt, keys.data(), n0, end, s.ks);
      for (int tid = 0; tid < nt; ++tid) {
        const int t = tid / v41idx::KL, kl = tid % v41idx::KL;
        if (t < tokens)
          v41idx::tile<H>(t, kl, s.qs, s.ks, s.ws, n0, end, out.data(), n_out);
      }
    }
  }
}

static long decode_test() {
  long bad = 0;
  std::vector<uint8_t> row(68 * KB, 0);
  std::vector<__half> ks(KB * v41idx::KS);
  // row k: nibble byte (k + 64 * rep) in every one of its 64 bytes, scale byte e
  for (int e = 0; e < 256; ++e) {
    for (int rep = 0; rep < 4; ++rep) {
      for (int k = 0; k < KB; ++k) {
        for (int b = 0; b < 64; ++b) row[k * 68 + b] = uint8_t(k + 64 * rep);
        for (int g = 0; g < 4; ++g) row[k * 68 + 64 + g] = uint8_t(e);
      }
      for (int tid = 0; tid < 96; ++tid)
        v41idx::stage_keys(tid, 96, row.data(), 0, KB, ks.data());
      for (int k = 0; k < KB; ++k) {
        const int byte = k + 64 * rep;
        const __half elo = __float2half_rn(spec_e2m1(byte & 15) * spec_scale(e));
        const __half ehi = __float2half_rn(spec_e2m1((byte >> 4) & 15) * spec_scale(e));
        for (int b = 0; b < 64; ++b) {
          if (hbits(ks[k * v41idx::KS + 2 * b]) != hbits(elo)) ++bad;
          if (hbits(ks[k * v41idx::KS + 2 * b + 1]) != hbits(ehi)) ++bad;
        }
      }
    }
  }
  // mixed scales per 32-element group (byte b uses scale 64 + b / 16)
  for (int k = 0; k < KB; ++k)
    for (int i = 0; i < 68; ++i) row[k * 68 + i] = uint8_t(rnd());
  for (int tid = 0; tid < 96; ++tid) v41idx::stage_keys(tid, 96, row.data(), 0, KB, ks.data());
  for (int k = 0; k < KB; ++k)
    for (int b = 0; b < 64; ++b) {
      const int byte = row[k * 68 + b], e = row[k * 68 + 64 + b / 16];
      if (hbits(ks[k * v41idx::KS + 2 * b]) != hbits(__float2half_rn(spec_e2m1(byte & 15) * spec_scale(e)))) ++bad;
      if (hbits(ks[k * v41idx::KS + 2 * b + 1]) != hbits(__float2half_rn(spec_e2m1(byte >> 4) * spec_scale(e)))) ++bad;
    }
  return bad;
}

// A stale live position must not consume the guarded allocation beyond capacity.
// The reference grid has exactly two 32-key blocks for this aligned capacity.
static int capacity_test() {
  constexpr int cap = 64, allocated = 128, heads = 8;
  std::vector<__half> q(heads * 128, __float2half_rn(1.f));
  std::vector<__half> w(heads, __float2half_rn(0.125f));
  // Guard rows make a missing bound deterministic instead of invoking host UB.
  // An actual keys allocation of cap rows would be read out of bounds.
  std::vector<uint8_t> keys(allocated * KEY_BYTES, 0);
  std::vector<int64_t> positions(1, allocated - 1);
  std::vector<__half> expected(allocated, __float2half_rn(-7.f));
  auto actual = expected;
  ref_scores(q, w, keys, cap, positions, expected, 1, heads, allocated, 1);
  tiled_host<heads>(q, w, keys, cap, positions, actual, 1, allocated, 1, 1);
  for (int n = 0; n < allocated; ++n) {
    if (hbits(actual[n]) != hbits(expected[n])) {
      std::printf("CAPACITY FAILED at column %d\n", n);
      return 1;
    }
  }
  std::printf("CAPACITY OK: stale end is bounded by 64 keys\n");
  return 0;
}

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "--capacity-only") return capacity_test();
  if (capacity_test() != 0) return 1;
  const bool expect_mismatch = argc > 1 && std::string(argv[1]) == "--expect-mismatch";
  long cases = 0, mismatch_cases = 0, mismatched = 0, compared = 0, nonfinite = 0, written = 0, written_finite = 0;
  long decode_bad = decode_test();
  std::printf("DECODE exhaustive 256 bytes x 256 scale bytes + mixed groups: %ld mismatches\n", decode_bad);
  const int caps[] = {1, 63, 64, 65, 300};
  const int grids[] = {1, 3, 160};
  for (int heads : {8, 4}) {
    for (int tokens = 1; tokens * heads <= 64; ++tokens) {
      for (int cap : caps) {
        for (int ratio : {1, 4}) {
          for (int mode = 0; mode < 2; ++mode) {
            const int64_t full = int64_t(cap) * ratio;  // positions[T-1] + 1 -> end = cap
            const int64_t ends_pos[] = {0, 1, int64_t(ratio), full / 2, full - 1, full};
            for (int64_t p1 : ends_pos) {
              for (int grid : grids) {
                const int rows = tokens * heads;
                std::vector<__half> q(rows * 128), w(rows);
                for (auto& v : q) {
                  float f = gauss();
                  if (mode == 1) {  // extremes: large, overflowing dots, some inf / nan
                    const float r = uni();
                    f = r < 0.02f ? (r < 0.01f ? INFINITY : NAN) : f * 30000.f;
                  }
                  v = __float2half_rn(f);
                }
                for (auto& v : w) v = __float2half_rn(mode == 1 ? gauss() * 300.f : gauss() * 0.1f);
                std::vector<uint8_t> keys(size_t(cap) * 68);
                for (int64_t n = 0; n < cap; ++n) {
                  for (int b = 0; b < 64; ++b) keys[n * 68 + b] = uint8_t(rnd());
                  for (int g = 0; g < 4; ++g)
                    keys[n * 68 + 64 + g] = mode == 1 ? uint8_t(rnd()) : uint8_t(118 + rnd() % 14);
                }
                std::vector<int64_t> positions(tokens);
                for (int t = 0; t < tokens; ++t) positions[t] = p1 - 1 - (tokens - 1 - t);
                const int64_t n_out = cap + 7;
                std::vector<__half> a(tokens * n_out), b(tokens * n_out);
                for (size_t i = 0; i < a.size(); ++i) a[i] = b[i] = hfrom(uint16_t(0x7E00 | (i & 0x1FF)));
                ref_scores(q, w, keys, cap, positions, a, tokens, heads, n_out, ratio);
                if (heads == 8)
                  tiled_host<8>(q, w, keys, cap, positions, b, tokens, n_out, ratio, grid);
                else
                  tiled_host<4>(q, w, keys, cap, positions, b, tokens, n_out, ratio, grid);
                long m = 0;
                for (size_t i = 0; i < a.size(); ++i) {
                  if (hbits(a[i]) != hbits(b[i])) ++m;
                  if (!std::isfinite(__half2float(a[i]))) ++nonfinite;
                  if (hbits(a[i]) != uint16_t(0x7E00 | (i & 0x1FF))) {
                    ++written;
                    if (std::isfinite(__half2float(a[i]))) ++written_finite;
                  }
                }
                compared += long(a.size());
                mismatched += m;
                ++cases;
                if (m) {
                  ++mismatch_cases;
                  if (!expect_mismatch && mismatch_cases <= 5)
                    std::printf("MISMATCH heads %d tokens %d cap %d ratio %d mode %d p1 %lld grid %d: %ld\n",
                                heads, tokens, cap, ratio, mode, (long long)p1, grid, m);
                }
              }
            }
          }
        }
      }
    }
  }
  std::printf("SCORES %ld cases, %ld values compared (%ld non-finite incl. poison; %ld written by the reference, %ld of them finite), %ld mismatched values in %ld cases\n",
              cases, compared, nonfinite, written, written_finite, mismatched, mismatch_cases);
  if (!expect_mismatch && (written < 100000 || written_finite < 50000)) {
    std::printf("FAILED: too few written values for a meaningful comparison\n");
    return 1;
  }
  if (expect_mismatch) {
    const bool killed = mismatched > 0;
    std::printf("%s\n", killed ? "MUTANT KILLED" : "MUTANT SURVIVED");
    return killed ? 0 : 1;
  }
  const bool ok = decode_bad == 0 && mismatched == 0;
  std::printf("%s\n", ok ? "ALL OK" : "FAILED");
  return ok ? 0 : 1;
}
