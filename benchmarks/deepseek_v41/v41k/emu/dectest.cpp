// Exhaustive decode check: old Group3<T> (verbatim, shared byte codebook) vs the
// new PRMT-selector decode (dec16/dec17/dec29 over the replicated fast table).
// Build as emu.cpp (same flags), run with no arguments; exit 1 on a mismatch.
#include "shim.h"
#include <cstdio>
#include <random>
#include "src/turbomind/kernels/gemm/lattice_codebooks.h"
#define __CUDACC__ 1
#include "v41k_fast.cuh"
#include "old_extract.inc"

template <class F> void run_serial(int grid, int block, F f) {
  blockDim.x = block; gridDim.x = grid;
  for (int b = 0; b < grid; ++b) for (int t = 0; t < block; ++t) { blockIdx.x = b; threadIdx.x = t; f(); }
}
template <int T> std::vector<uint32_t> old_book() {
  const int words = turbomind::gemm::LatticeCodebook<T>::kBytes / 4;
  std::vector<uint32_t> b(words);
  run_serial((words + 255) / 256, 256, [&] { v41k::build_table<T>(b.data(), words); });
  return b;
}
template <int T> std::vector<uint32_t> smem_book() {  // fast table, then fill_book's layout
  using FB = v41k_fast::FastBook<T>;
  std::vector<uint32_t> g(FB::kWords), s(FB::kSmemWords);
  run_serial((FB::kWords + 255) / 256, 256, [&] { v41k_fast::build_fast_table<T>(g.data()); });
  for (int i = 0; i < FB::kSmemWords; ++i) {
    int src = i;
    if constexpr (T != 29) src = i < FB::kSignBase ? i / FB::kGridR : FB::kGrid + (i - FB::kSignBase) / FB::kSignR;
    s[i] = g[src];
  }
  return s;
}

template <int T>
long check(const uint8_t* blk, int sub, int lane, const uint32_t* ob, const uint32_t* sb, long& n) {
  uint32_t masks[16];
  for (int i = 0; i < 16; ++i) masks[i] = sign_mask(i);
  const v41k::Group3<T> g(blk, sub, ob, masks);
  uint32_t w[8];
  int m0, m1;
  float d;
  if constexpr (T == 29) {
    uint32_t qs, s01, s23;
    uint16_t qh;
    std::memcpy(&qs, blk + 4 * sub, 4);
    std::memcpy(&qh, blk + 32 + 2 * sub, 2);
    std::memcpy(&s01, blk + 48, 4);
    std::memcpy(&s23, blk + 52, 4);
    v41k_fast::dec29(qs, qh, s01, s23, sub, sb, w, m0, m1);
    d = __half2float(__ushort_as_half(v41k_fast::iq1m_d_bits(s01, s23)));
  } else {
    uint32_t w0, w1;
    uint16_t dh;
    std::memcpy(&w0, blk + 2 + 8 * sub, 4);
    std::memcpy(&w1, blk + 6 + 8 * sub, 4);
    std::memcpy(&dh, blk, 2);
    const uint32_t lg = uint32_t(lane) * 4, ls = uint32_t(lane & 15) * 4;
    if constexpr (T == 16) v41k_fast::dec16(w0, w1, sb, lg, ls, w, m0, m1);
    else v41k_fast::dec17(w0, w1, blk[66 + sub], sb, ls, w, m0, m1);
    d = __half2float(__ushort_as_half(dh));
  }
  ++n;
  long bad = 0;
  for (int i = 0; i < 8; ++i) bad += w[i] != g.w[i];
  bad += m0 != g.m0 || m1 != g.m1;
  uint32_t a, b;
  std::memcpy(&a, &d, 4);
  std::memcpy(&b, &g.d, 4);
  bad += a != b;
  return bad;
}

template <int T> long run(std::mt19937& rng, long& n) {
  constexpr int BB = T == 16 ? 66 : T == 17 ? 74 : 56;
  const auto ob = old_book<T>();
  const auto sb = smem_book<T>();
  alignas(16) uint8_t blk[BB + 16];
  long bad = 0;
  const int codes = T == 16 ? 32768 : T == 17 ? 65536 : 4096;
  for (int sub = 0; sub < 8; ++sub)
    for (int i = 0; i < codes; ++i) {
      for (auto& x : blk) x = uint8_t(rng());
      // every octet position o sees every code as i runs (different offsets per o)
      for (int o = 0; o < 4; ++o) {
        const int c = (i + o * 7919) % codes;
        if constexpr (T == 16) {
          blk[2 + 8 * sub + o] = uint8_t(c & 255);
          uint32_t meta;
          std::memcpy(&meta, blk + 6 + 8 * sub, 4);
          meta = (meta & ~(127u << (7 * o))) | (uint32_t(c >> 8) << (7 * o));
          if (o == 3) meta = (meta & 0x0fffffffu) | (uint32_t(i & 15) << 28);
          std::memcpy(blk + 6 + 8 * sub, &meta, 4);
        } else if constexpr (T == 17) {
          const uint16_t c16 = uint16_t(c);
          std::memcpy(blk + 2 + 8 * sub + 2 * o, &c16, 2);
          blk[66 + sub] = uint8_t(i & 255);
        } else {
          blk[4 * sub + o] = uint8_t(c & 255);
          uint16_t qh;
          std::memcpy(&qh, blk + 32 + 2 * sub, 2);
          qh = uint16_t((qh & ~(15u << (4 * o))) | (uint32_t(c >> 8) << (4 * o)));
          std::memcpy(blk + 32 + 2 * sub, &qh, 2);
        }
      }
      bad += check<T>(blk, sub, int(rng() & 31), ob.data(), sb.data(), n);
    }
  if constexpr (T == 29)  // scale fields: every 16-bit value of each scale word, every sub
    for (int word = 0; word < 4; ++word)
      for (int v = 0; v < 65536; ++v) {
        for (auto& x : blk) x = uint8_t(rng());
        const uint16_t vv = uint16_t(v);
        std::memcpy(blk + 48 + 2 * word, &vv, 2);
        bad += check<T>(blk, v & 7, 0, ob.data(), sb.data(), n);
      }
  return bad;
}

int main() {
  std::mt19937 rng(12345);
  long total = 0;
  for (int T : {16, 17, 29}) {
    long n = 0;
    const long bad = T == 16 ? run<16>(rng, n) : T == 17 ? run<17>(rng, n) : run<29>(rng, n);
    std::printf("type %d: %ld sub-block decodes, %ld mismatching fields\n", T, n, bad);
    total += bad;
  }
  return total ? 1 : 0;
}
