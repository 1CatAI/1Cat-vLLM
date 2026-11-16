// CPU emulation, bitwise, on synthetic weights / route mixes: old
// down_rows<T, NW, 8> vs new v41k_fast::down_rows_fast<T, M>, and old
// gate_up_em5<T> (plan mode) vs new v41k_fast::gate_up_fast<T> (the old kernels
// verbatim from the base commit, see extract_old.py / old_extract.inc).
// Host-only (no GPU): each CUDA thread is a std::thread, CTAs run in shuffled
// order, outputs are NaN-poisoned, the plan tail is stale garbage, and routes
// no plan member names carry NaN hd scales. From the repository root:
//   g++ -std=c++20 -O2 -ffp-contract=off -pthread \
//     -Icsrc/sm70_turbomind/lmdeploy -Ibenchmarks/deepseek_v41/v41k \
//     -o /tmp/emu benchmarks/deepseek_v41/v41k/emu/emu.cpp
//   /tmp/emu <type 16|17|29> <m_lo> <m_hi> <kinds mask, 63 = all> <seed> [gu]
// (gu: gate_up_em5 vs gate_up_fast; default: down_rows vs down_rows_fast)
// prints one line per case and "... type T: N failing cases" (exit 1 if N > 0).
#include "shim.h"
#include <algorithm>
#include <cstdio>
#include <random>
#include "src/turbomind/kernels/gemm/lattice_codebooks.h"
#define __CUDACC__ 1
#include "old_extract.inc"
#include "v41k_fast_gu.cuh"

constexpr int E = 64, RL = 1280;

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
template <int T> std::vector<uint32_t> fast_book() {
  const int words = v41k_fast::FastBook<T>::kWords;
  std::vector<uint32_t> b(words);
  run_serial((words + 255) / 256, 256, [&] { v41k_fast::build_fast_table<T>(b.data()); });
  return b;
}

std::vector<int> make_ids(int m, int kind, std::mt19937& rng) {
  std::vector<int> ids(m * 6);
  auto rnd = [&](int n) { return int(rng() % n); };
  for (int k = 0; k < m; ++k) {
    std::vector<int> row;
    if (kind == 0) {  // distinct per row, random across rows
      while ((int)row.size() < 6) { int e = rnd(E); if (std::find(row.begin(), row.end(), e) == row.end()) row.push_back(e); }
    } else if (kind == 1) {  // pool of 8, distinct per row, drop a tail
      while ((int)row.size() < 6) { int e = rnd(8) + 3; if (std::find(row.begin(), row.end(), e) == row.end()) row.push_back(e); }
      for (int i = 6 - rnd(4); i < 6; ++i) row[i] = -1;
    } else if (kind == 2) {  // pool of 3 with repeats inside a row (cnt caps at 6)
      for (int i = 0; i < 6; ++i) row.push_back(rnd(3) + 20);
    } else if (kind == 3) {  // one expert in every row
      row.push_back(7);
      while ((int)row.size() < 6) { int e = rnd(E); if (std::find(row.begin(), row.end(), e) == row.end()) row.push_back(e); }
      std::shuffle(row.begin(), row.end(), rng);
    } else if (kind == 4) {  // all dropped
      row.assign(6, -1);
    } else {  // pool of 12, some drops anywhere
      while ((int)row.size() < 6) { int e = rnd(12) + 30; if (std::find(row.begin(), row.end(), e) == row.end()) row.push_back(e); }
      for (int i = 0; i < 6; ++i) if (rnd(4) == 0) row[i] = -1;
    }
    for (int i = 0; i < 6; ++i) ids[k * 6 + i] = row[i];
  }
  return ids;
}
// moe_plan semantics (first-occurrence order, routes ascending, cnt capped at 6);
// entries it does not write keep stale garbage.
void make_plan(const std::vector<int>& ids, std::vector<int>& plan) {
  std::vector<int> ex;
  for (int r = 0; r < (int)ids.size(); ++r) {
    const int e = ids[r];
    if (e < 0) continue;
    int u = int(std::find(ex.begin(), ex.end(), e) - ex.begin());
    if (u == (int)ex.size()) { ex.push_back(e); plan[37 + u] = 0; }
    plan[1 + u] = e;
    int c = 0;
    for (int q = 0; q < r; ++q) c += ids[q] == e;
    if (c < 6) plan[73 + 6 * u + c] = r;
    int cnt = 0;
    for (int q = 0; q < (int)ids.size(); ++q) cnt += ids[q] == e;
    plan[37 + u] = std::min(cnt, 6);
  }
  plan[0] = (int)ex.size();
}

template <int T> void fill_weights(std::vector<uint8_t>& w, int bb, std::mt19937& rng) {
  for (auto& x : w) x = uint8_t(rng());
  const size_t blocks = (w.size() - 64) / bb;
  for (size_t b = 0; b < blocks; ++b) {
    uint8_t* p = w.data() + b * bb;
    const uint16_t dbits = uint16_t(0x1000 + rng() % 0x1800);  // ~2^-11 .. 2^-4, positive
    if constexpr (T == 29) {
      uint16_t s[4];
      std::memcpy(s, p + 48, 8);
      for (int i = 0; i < 4; ++i) s[i] = uint16_t((s[i] & 0x0fff) | (((dbits >> (4 * i)) & 15) << 12));
      std::memcpy(p + 48, s, 8);
    } else {
      std::memcpy(p, &dbits, 2);
    }
  }
}

template <int T>
int run_case(int m, int kind, int classes, uint64_t seed, const std::vector<uint32_t>& ob,
             const std::vector<uint32_t>& fb, const std::vector<uint8_t>& down, bool verbose) {
  constexpr int bb = T == 16 ? 66 : T == 17 ? 74 : 56;
  std::mt19937 rng(seed);
  const int routes = 6 * m, rs = (routes * 648 + 255) / 256 * 256;
  const int rank = int(rng() % 4), row_base = rank * RL;
  std::vector<int> ids = make_ids(m, kind, rng);
  std::vector<int> plan(512);
  for (auto& x : plan) x = 0x40000000 + int(rng() % 1000);  // stale garbage
  make_plan(ids, plan);
  std::vector<uint8_t> gathered(4 * rs);
  for (int r = 0; r < 4; ++r) {
    uint8_t* slot = gathered.data() + r * rs;
    for (int i = 0; i < routes * 576; ++i) slot[i] = uint8_t(int8_t(int(rng() % 255) - 127));
    float* hd = reinterpret_cast<float*>(slot + routes * 576);
    for (int i = 0; i < routes * 18; ++i) hd[i] = 0.001f + float(rng() % 100000) * 5e-7f;
  }
  {  // routes no plan member names (dropped, or past the cnt cap): NaN scales
    std::vector<char> used(routes, 0);
    for (int u = 0; u < plan[0]; ++u)
      for (int c = 0; c < plan[37 + u]; ++c) used[plan[73 + 6 * u + c]] = 1;
    for (int r = 0; r < 4; ++r) {
      float* hd = reinterpret_cast<float*>(gathered.data() + r * rs + routes * 576);
      for (int q = 0; q < routes; ++q)
        if (!used[q])
          for (int i = 0; i < 18; ++i) hd[q * 18 + i] = __builtin_nanf("");
    }
  }
  std::vector<half> o_old(m * 5120), o_new(m * 5120), o_dirty(m * 5120);
  for (auto& h : o_old) h.x = 0x7e00;
  for (auto& h : o_new) h.x = 0x7e00;
  std::vector<int> ord_old(RL / 4), ord_new(RL / 16);
  for (int i = 0; i < (int)ord_old.size(); ++i) ord_old[i] = i;
  for (int i = 0; i < (int)ord_new.size(); ++i) ord_new[i] = i;
  std::shuffle(ord_new.begin(), ord_new.end(), rng);
  const int row_stride = 9 * bb;
  auto old_k = [&] {
    if (classes == 8)
      v41k::down_rows<T, 8, 8>((half*)o_old.data(), gathered.data(), rs, down.data(), row_stride, RL, row_base,
                               ob.data(), routes, 6, m, 4, plan.data());
    else
      v41k::down_rows<T, 4, 8>((half*)o_old.data(), gathered.data(), rs, down.data(), row_stride, RL, row_base,
                               ob.data(), routes, 6, m, 4, plan.data());
  };
  emu_launch(RL / 4, 32 * classes, old_k, ord_old);
  auto new_k = [&](half* out, const int* pl) {
    switch (m) {
#define C(M) case M: v41k_fast::down_rows_fast<T, M>(out, gathered.data(), down.data(), row_stride, RL, row_base, fb.data(), classes, pl); break;
      C(1) C(2) C(3) C(4) C(5) C(6)
#undef C
    }
  };
  // Dirty the static "shared memory" with a different plan first.
  {
    std::vector<int> ids2 = make_ids(m, 0, rng), plan2 = plan;
    make_plan(ids2, plan2);
    emu_launch(RL / 16, 512, [&] { new_k(o_dirty.data(), plan2.data()); }, ord_new);
  }
  emu_launch(RL / 16, 512, [&] { new_k(o_new.data(), plan.data()); }, ord_new);
  int bad = 0, nan = 0, nz = 0;
  for (int i = 0; i < m * 5120; ++i) {
    if (o_old[i].x != o_new[i].x) {
      if (bad < 5 && verbose)
        std::printf("  mismatch k=%d n=%d old=%04x new=%04x\n", i / 5120, i % 5120, o_old[i].x, o_new[i].x);
      ++bad;
    }
    nan += (o_new[i].x & 0x7c00) == 0x7c00;
    nz += o_new[i].x != 0;
  }
  std::printf("T=%d M=%d kind=%d classes=%d rank=%d nu=%d: mismatches=%d nan=%d nonzero=%d\n", T, m, kind,
              classes, rank, plan[0], bad, nan, nz);
  return bad + nan;
}

// gate/up: old gate_up_em5<T> (plan mode, grid 18 x routes) vs the persistent
// gate_up_fast<T> on grids of 80, 13 and 3 CTAs; hq / hd prefilled with
// garbage / NaN, compared whole (entries no member writes must stay as they
// were in both).
template <int T>
int run_case_gu(int m, int kind, uint64_t seed, const std::vector<uint32_t>& ob,
                const std::vector<uint32_t>& fb, const std::vector<uint8_t>& gate,
                const std::vector<uint8_t>& up, bool verbose) {
  constexpr int bb = T == 16 ? 66 : T == 17 ? 74 : 56;
  std::mt19937 rng(seed);
  const int routes = 6 * m, row_stride = 20 * bb;
  std::vector<int> ids = make_ids(m, kind, rng);
  std::vector<int> plan(512);
  for (auto& x : plan) x = 0x40000000 + int(rng() % 1000);  // stale garbage
  make_plan(ids, plan);
  std::vector<int8_t> xq(m * 160 * 32);
  std::vector<float> xd(m * 160), prob(routes);
  for (auto& v : xq) v = int8_t(int(rng() % 255) - 127);
  for (auto& v : xd) v = 0.001f + float(rng() % 100000) * 9e-8f;
  for (auto& v : prob) v = float(rng() % 100000) * 1e-5f;
  const int grids[3] = {80, 13, 3};
  std::vector<int8_t> hq0(routes * 18 * 32);
  for (auto& v : hq0) v = int8_t(rng());
  std::vector<float> hd0(routes * 18, __builtin_nanf(""));
  std::vector<int8_t> hq_old = hq0;
  std::vector<float> hd_old = hd0;
  std::vector<int> ord_old(18 * routes);
  for (int i = 0; i < (int)ord_old.size(); ++i) ord_old[i] = i;
  std::shuffle(ord_old.begin(), ord_old.end(), rng);
  emu_launch2(18, routes, 512, [&] {
    v41k::gate_up_em5<T>(hq_old.data(), hd_old.data(), xq.data(), xd.data(), ids.data(), prob.data(),
                         gate.data(), up.data(), row_stride, ob.data(), routes, 6, plan.data());
  }, ord_old);
  int fails = 0;
  for (int g : grids) {
    std::vector<int8_t> hq_new = hq0;
    std::vector<float> hd_new = hd0;
    std::vector<int> ord(g);
    for (int i = 0; i < g; ++i) ord[i] = i;
    std::shuffle(ord.begin(), ord.end(), rng);
    {  // dirty the static "shared memory" with a different plan first
      std::vector<int> ids2 = make_ids(m, 0, rng), plan2 = plan;
      make_plan(ids2, plan2);
      std::vector<int8_t> hq2 = hq0;
      std::vector<float> hd2 = hd0;
      emu_launch(g, 512, [&] {
        v41k_fast::gate_up_fast<T>(hq2.data(), hd2.data(), xq.data(), xd.data(), prob.data(), gate.data(),
                                   up.data(), row_stride, fb.data(), plan2.data());
      }, ord);
    }
    emu_launch(g, 512, [&] {
      v41k_fast::gate_up_fast<T>(hq_new.data(), hd_new.data(), xq.data(), xd.data(), prob.data(), gate.data(),
                                 up.data(), row_stride, fb.data(), plan.data());
    }, ord);
    int bad = 0, written = 0, clamp = 0;
    for (int i = 0; i < routes * 18; ++i) {
      uint32_t a, b;
      std::memcpy(&a, &hd_old[i], 4);
      std::memcpy(&b, &hd_new[i], 4);
      bad += a != b;
      written += !std::isnan(hd_old[i]);
      clamp += !std::isnan(hd_old[i]) && hd_old[i] > 9.9f / 127.f;
      for (int j = 0; j < 32; ++j) bad += hq_old[i * 32 + j] != hq_new[i * 32 + j];
    }
    if (verbose)
      std::printf("GU T=%d M=%d kind=%d grid=%d nu=%d: mismatches=%d groups_written=%d big=%d\n", T, m, kind, g,
                  plan[0], bad, written, clamp);
    fails += bad != 0;
  }
  return fails;
}

template <int T> int run_type_gu(uint64_t seed, int kinds_mask, int m_lo, int m_hi) {
  constexpr int bb = T == 16 ? 66 : T == 17 ? 74 : 56;
  std::mt19937 rng(seed);
  std::vector<uint8_t> gate(size_t(E) * 576 * 20 * bb + 64), up(gate.size());
  fill_weights<T>(gate, bb, rng);
  fill_weights<T>(up, bb, rng);
  const auto ob = old_book<T>();
  const auto fb = fast_book<T>();
  int fails = 0;
  for (int m = m_lo; m <= m_hi; ++m)
    for (int kind = 0; kind < 6; ++kind)
      if (kinds_mask >> kind & 1) fails += run_case_gu<T>(m, kind, seed * 1000 + m * 10 + kind, ob, fb, gate, up, true) != 0;
  return fails;
}

template <int T> int run_type(uint64_t seed, int kinds_mask, int m_lo, int m_hi) {
  constexpr int bb = T == 16 ? 66 : T == 17 ? 74 : 56;
  std::mt19937 rng(seed);
  std::vector<uint8_t> down(size_t(E) * RL * 9 * bb + 64);
  fill_weights<T>(down, bb, rng);
  const auto ob = old_book<T>();
  const auto fb = fast_book<T>();
  int fails = 0;
  for (int m = m_lo; m <= m_hi; ++m)
    for (int kind = 0; kind < 6; ++kind)
      if (kinds_mask >> kind & 1)
        fails += run_case<T>(m, kind, (m + kind) % 2 ? 8 : 4, seed * 1000 + m * 10 + kind, ob, fb, down, true) != 0;
  return fails;
}

int main(int argc, char** argv) {
  const int type = argc > 1 ? std::atoi(argv[1]) : 29;
  const int m_lo = argc > 2 ? std::atoi(argv[2]) : 1, m_hi = argc > 3 ? std::atoi(argv[3]) : 6;
  const int kinds = argc > 4 ? std::atoi(argv[4]) : 63;
  const uint64_t seed = argc > 5 ? std::atoll(argv[5]) : 1;
  const bool gu = argc > 6 && std::strcmp(argv[6], "gu") == 0;
  int fails = gu ? (type == 16 ? run_type_gu<16>(seed, kinds, m_lo, m_hi)
                    : type == 17 ? run_type_gu<17>(seed, kinds, m_lo, m_hi)
                                 : run_type_gu<29>(seed, kinds, m_lo, m_hi))
                 : (type == 16 ? run_type<16>(seed, kinds, m_lo, m_hi)
                    : type == 17 ? run_type<17>(seed, kinds, m_lo, m_hi)
                                 : run_type<29>(seed, kinds, m_lo, m_hi));
  std::printf("%s type %d: %d failing cases\n", gu ? "gate/up" : "down", type, fails);
  return fails ? 1 : 0;
}
