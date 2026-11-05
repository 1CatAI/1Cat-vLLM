// Single-threaded CPU check of the EXPERT_DIET decode pieces (v41k_diet.cuh)
// against the EXPERT_FAST ones they replace (v41k_fast.cuh): exact integer
// equality of every word / multiplier / scale and bitwise equality of the
// float dot. No codebook needed (any table works: both index the same one).
// From the repository root:
//   g++ -std=c++20 -O2 -ffp-contract=off -Icsrc/sm70_turbomind/lmdeploy \
//     -Ibenchmarks/deepseek_v41/v41k \
//     -o /tmp/diet_dectest benchmarks/deepseek_v41/v41k/emu/diet_dectest.cpp
//   /tmp/diet_dectest        (a few seconds; exit 1 on a mismatch)
#include "shim.h"
#include <cstdio>
#include <random>
#include "src/turbomind/kernels/gemm/lattice_codebooks.h"
#define __CUDACC__ 1
#define V41K_HOST_EMU 1
#include "old_extract.inc"
#include "v41k_fast.cuh"
#include "v41k_diet.cuh"

static long g_bad = 0;
#define CHECK(c)                                                    \
  do {                                                              \
    if (!(c) && g_bad++ < 10) std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); \
  } while (0)

int main() {
  std::mt19937_64 rng(20261103);
  // A random 4096-word IQ1_M selector table (dec29 and the diet lookup read
  // the same words, so equality over a random table covers every index).
  std::vector<uint32_t> book29(v41k_fast::FastBook<29>::kSmemWords);
  for (auto& w : book29) w = uint32_t(rng());
  long n = 0;

  // d29_bits == iq1m_d_bits: a bit permutation; random + all single bits.
  for (int i = 0; i < 4000000; ++i) {
    const uint32_t a = uint32_t(rng()), b = uint32_t(rng());
    CHECK((v41k_diet::d29_bits(a, b) & 0xFFFFu) == v41k_fast::iq1m_d_bits(a, b));
    ++n;
  }
  for (int bit = 0; bit < 64; ++bit) {
    const uint64_t x = uint64_t(1) << bit;
    const uint32_t a = uint32_t(x), b = uint32_t(x >> 32);
    CHECK((v41k_diet::d29_bits(a, b) & 0xFFFFu) == v41k_fast::iq1m_d_bits(a, b));
    ++n;
  }

  // idx29 + dec29_words + m29 == dec29: every qh (16 bits) x random qs, all 8
  // sub-blocks, random scale words.
  for (uint32_t qh = 0; qh < 65536; ++qh)
    for (int r = 0; r < 8; ++r) {
      const uint32_t qs = uint32_t(rng()), s01 = uint32_t(rng()), s23 = uint32_t(rng());
      const int sub = r;
      uint32_t w0[8], w1[8];
      int a0, a1, b0, b1;
      v41k_fast::dec29(qs, qh, s01, s23, sub, book29.data(), w0, a0, a1);
      uint32_t ia, ib;
      v41k_diet::idx29(qs, qh, ia, ib);
      v41k_diet::dec29_words(ia, ib, book29.data(), w1);
      const uint32_t sh = uint32_t(((sub >> 1) & 1) * 16 + (sub & 1) * 6);
      v41k_diet::m29(s01, s23, sub >= 4, sh, 1u, b0, b1);
      for (int i = 0; i < 8; ++i) CHECK(w0[i] == w1[i]);
      CHECK(a0 == b0);
      CHECK(a1 == b1);
      ++n;
    }

  // dot<false> == fast_dot always; dot<true> == fast_dot when m0 == m1 (the
  // IQ2_XXS case), over random words / activations / scales / accumulators,
  // including the extremes of the int8 ranges and the multiplier range 1..31.
  for (int i = 0; i < 3000000; ++i) {
    uint32_t w[8];
    int q[8];
    for (int j = 0; j < 8; ++j) {
      w[j] = uint32_t(rng());
      q[j] = int(uint32_t(rng()));
      if (i % 7 == 0) {  // extremes
        w[j] = (i & 8) ? 0x81818181u : 0x7F7F7F7Fu;
        q[j] = (i & 16) ? int(0x80808080u) : 0x7F7F7F7F;
      }
    }
    const int m0 = 1 + 2 * int(rng() % 16), m1 = 1 + 2 * int(rng() % 16);
    float d, xd, acc;
    const uint32_t bd = uint32_t(rng()), bx = uint32_t(rng()), ba = uint32_t(rng());
    // finite, varied magnitudes: fp16-range d, small xd, any accumulator
    d = float(int32_t(bd % 131072) - 65536) * 1e-5f;
    xd = float(bx % 100000) * 9e-8f + 1e-4f;
    acc = float(int32_t(ba % 2000001) - 1000000) * 1e-3f;
    const float f0 = v41k_fast::fast_dot(w, q, d, xd, m0, m1, acc);
    const float g0 = v41k_diet::dot<false>(w, q, d, xd, m0, m1, acc);
    const float f1 = v41k_fast::fast_dot(w, q, d, xd, m0, m0, acc);
    const float g1 = v41k_diet::dot<true>(w, q, d, xd, m0, m0, acc);
    uint32_t u0, v0, u1, v1;
    std::memcpy(&u0, &f0, 4);
    std::memcpy(&v0, &g0, 4);
    std::memcpy(&u1, &f1, 4);
    std::memcpy(&v1, &g1, 4);
    CHECK(u0 == v0);
    CHECK(u1 == v1);
    ++n;
  }
  std::printf("diet decode: %ld cases, %ld mismatches\n", n, g_bad);
  return g_bad ? 1 : 0;
}
