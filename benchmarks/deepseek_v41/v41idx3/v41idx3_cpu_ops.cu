// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// v41idx3_cpu_ops: the two decode index scorers' own bodies as CPU ops, for tests on a machine with no card.
// Test library only: never loaded by a server.
//
//   torch.ops.v41idx.index_scores_tiled(out, q, w, keys, positions, heads, ratio)
//       ../v41idx/v41idx.cu's own stage and tile bodies (#1372, the reference), run with its CTA schedule
//       (grid = min(blocks, 2 * 80): a V100 has 80 SMs) and its live end clamped to the key capacity
//   torch.ops.v41idx3.index_scores_tc(out, q, w, keys, positions, heads, ratio, variant, counters)
//       v41idx3.cu's own stage, guard, fragment, epilogue and fallback bodies, the device op's signature and checks,
//       grid = min(blocks, ctas * 80), with each mma of a warp replaced by the card's measured layout and adder
//       (v41idx3::tc_cell)
//   torch.ops.v41idx3.tc_cell(c, a, b) -> Tensor
//       the adder alone: c float32 [n], a and b float32 [n, 4] holding fp16 values; one output cell each
//
// Build (no card): build_cpu_ops.sh beside this file.

#define V41IDX_HOST_TEST
#include "v41idx.cu"
#include "v41idx3.cu"

#include <torch/all.h>
#include <algorithm>
#include <vector>

namespace {

constexpr int SMS = 80;

inline float half_value(uint32_t bits16) {
  const unsigned short u = static_cast<unsigned short>(bits16);
  __half h;
  memcpy(&h, &u, 2);
  return __half2float(h);
}

// One mma of a warp: four quad pairs, each its own 8 x 8 x 4 product. Thread slot s (lane qd * 4 + (s & 3), plus
// 16 for s >= 4) holds A's row s and B's column s; output (m, n) is in slot (m&1)|(n&2)|(m&4), element
// (n&1)|(m&2)|(n&4). Measured on a V100 by a probe of single mma instructions (0 of 64 off).
void host_mma(float acc[32][8], const v41idx3::Frag a[32], const v41idx3::Frag b[32]) {
  for (int qd = 0; qd < 4; ++qd) {
    float A[8][4], B[8][4];
    for (int s = 0; s < 8; ++s) {
      const int lane = qd * 4 + (s & 3) + ((s & 4) ? 16 : 0);
      A[s][0] = half_value(a[lane].x & 0xffffu);
      A[s][1] = half_value(a[lane].x >> 16);
      A[s][2] = half_value(a[lane].y & 0xffffu);
      A[s][3] = half_value(a[lane].y >> 16);
      B[s][0] = half_value(b[lane].x & 0xffffu);
      B[s][1] = half_value(b[lane].x >> 16);
      B[s][2] = half_value(b[lane].y & 0xffffu);
      B[s][3] = half_value(b[lane].y >> 16);
    }
    for (int m = 0; m < 8; ++m)
      for (int n = 0; n < 8; ++n) {
        const int so = (m & 1) | (n & 2) | (m & 4), e = (n & 1) | (m & 2) | (n & 4);
        const int lane = qd * 4 + (so & 3) + ((so & 4) ? 16 : 0);
        acc[lane][e] = v41idx3::tc_cell(acc[lane][e], A[m], B[n]);
      }
  }
}

struct Total {
  int64_t cells = 0, heads = 0;
};

template <bool GUARD, bool FORCE>
Total run(torch::Tensor& out, const torch::Tensor& q, const torch::Tensor& w, const torch::Tensor& keys,
          const torch::Tensor& positions, int ratio, int ctas) {
  using namespace v41idx3;
  const int tokens = int(out.size(0));
  const int rows = tokens * HEADS;
  const int nt = 32 * tokens;
  const int64_t end = live_end(positions.data_ptr<int64_t>()[tokens - 1], ratio, keys.size(0));
  const int64_t blocks = (keys.size(0) + KB - 1) / KB;
  const int grid = int(std::min<int64_t>(blocks, int64_t(ctas) * SMS));
  const auto* qp = reinterpret_cast<const __half*>(q.data_ptr());
  const auto* wp = reinterpret_cast<const __half*>(w.data_ptr());
  const uint8_t* kp = keys.data_ptr<uint8_t>();
  auto* op = reinterpret_cast<__half*>(out.data_ptr());
  std::vector<uint64_t> store((smem_bytes(tokens) + 64) / 8 + 2);  // 8-byte aligned, as the fragments need
  unsigned char* smem = reinterpret_cast<unsigned char*>(store.data());
  Total fell;
  for (int b = 0; b < grid; ++b) {
    int64_t n0 = int64_t(b) * KB;
    if (n0 >= end) continue;
    const Smem s = carve(smem, tokens);
    for (int tid = 0; tid < nt; ++tid) stage_q(tid, nt, qp, wp, rows, s.qs, s.ws);
    if (GUARD) {
      for (int tid = 0; tid < nt; ++tid) scan_group(qp, tid >> 2, tid & 3, s.ge, s.gl);
      for (int t = 0; t < tokens; ++t) s.at[t] = token_a(t, s.ge, s.gl);
    }
    for (; n0 < end; n0 += int64_t(grid) * KB) {
      for (int tid = 0; tid < nt; ++tid) stage_keys(tid, nt, kp, n0, end, s.ks);
      if (GUARD)
        for (int tid = 0; tid < nt; ++tid) stage_kb(tid, nt, kp, n0, end, s.kb, s.ksc);
      for (int t = 0; t < tokens; ++t) {
        if (!FORCE) {
          float acc[2][32][8] = {};
          for (int d = 0; d < 128; d += 4) {
            Frag a[32], b0[32], b1[32];
            for (int lane = 0; lane < 32; ++lane) {
              a[lane] = frag_a(s.qs, t, lane, d);
              b0[lane] = frag_b(s.ks, 0, lane, d);
              b1[lane] = frag_b(s.ks, 1, lane, d);
            }
            host_mma(acc[0], a, b0);
            host_mma(acc[1], a, b1);
          }
          for (int lane = 0; lane < 32; ++lane) {
            float mine[2][8];
            for (int e = 0; e < 8; ++e) {
              mine[0][e] = acc[0][lane][e];
              mine[1][e] = acc[1][lane][e];
            }
            heads_out(t, lane, mine, s.ws, s.hs);
          }
        }
        for (int lane = 0; lane < 32; ++lane) {
          const Fell f = cells_out<GUARD, FORCE>(t, lane, s, n0, end, op, out.size(1));
          fell.cells += f.cells;
          fell.heads += f.heads;
        }
      }
    }
  }
  return fell;
}

void index_scores_tc(torch::Tensor out, torch::Tensor q, torch::Tensor w, torch::Tensor keys,
                     torch::Tensor positions, int64_t heads, int64_t ratio, int64_t variant,
                     torch::Tensor counters) {
  const int64_t tokens = out.dim() == 2 ? out.size(0) : -1;
  TORCH_CHECK(!out.is_cuda(), "v41idx3_cpu_ops: CPU tensors only");
  TORCH_CHECK(out.dim() == 2 && out.scalar_type() == torch::kHalf && out.stride(1) == 1 &&
                  out.stride(0) == out.size(1),
              "v41idx3: out must be fp16 [tokens, n_out] with row stride n_out");
  TORCH_CHECK(heads == 8, "v41idx3: 8 index heads per rank only (a tile's 8 rows), got ", heads);
  TORCH_CHECK(tokens >= 1 && tokens <= v41idx3::MAX_TOKENS, "v41idx3: need 1 <= tokens <= 8");
  TORCH_CHECK(q.scalar_type() == torch::kHalf && q.is_contiguous() && q.dim() == 2 &&
                  q.size(0) == tokens * heads && q.size(1) == 128,
              "v41idx3: q must be contiguous fp16 [tokens * 8, 128]");
  TORCH_CHECK(w.scalar_type() == torch::kHalf && w.is_contiguous() && w.numel() == tokens * heads,
              "v41idx3: w must be contiguous fp16 [tokens * 8]");
  TORCH_CHECK(keys.scalar_type() == torch::kUInt8 && keys.is_contiguous() && keys.dim() == 2 &&
                  keys.size(1) == 68 && keys.size(0) > 0,
              "v41idx3: keys must be contiguous uint8 [capacity > 0, 68]");
  TORCH_CHECK(positions.scalar_type() == torch::kInt64 && positions.is_contiguous() &&
                  positions.numel() >= tokens,
              "v41idx3: positions must be contiguous int64 with >= tokens entries");
  TORCH_CHECK(counters.scalar_type() == torch::kInt64 && counters.is_contiguous() && counters.numel() >= 4,
              "v41idx3: counters must be contiguous int64 with >= 4 entries");
  TORCH_CHECK(out.size(1) >= keys.size(0), "v41idx3: output width must cover key capacity");
  TORCH_CHECK(ratio >= 1, "v41idx3: ratio must be >= 1");
  TORCH_CHECK(variant >= 0 && variant < 16 && (variant & 3) != 3,
              "v41idx3: variant must be 0..15 and not NOGUARD with FORCE, got ", variant);
  const int ctas = 2 + int((variant >> 2) & 3);
  Total fell;
  if (variant & 2) fell = run<true, true>(out, q, w, keys, positions, int(ratio), ctas);
  else if (variant & 1) fell = run<false, false>(out, q, w, keys, positions, int(ratio), ctas);
  else fell = run<true, false>(out, q, w, keys, positions, int(ratio), ctas);
  counters.data_ptr<int64_t>()[0] += fell.cells;
  counters.data_ptr<int64_t>()[1] += fell.heads;
  // the device kernel's CTA 0 adds these once a call that has a live key
  const int64_t end = v41idx3::live_end(positions.data_ptr<int64_t>()[tokens - 1], int(ratio), keys.size(0));
  if (end > 0) {
    counters.data_ptr<int64_t>()[2] += 1;
    counters.data_ptr<int64_t>()[3] += tokens * end;
  }
}

// The reference: v41idx.cu's kernel body (index_scores_tiled<H>) with its launch geometry, on the CPU.
template <int H>
void run_ref(torch::Tensor& out, const torch::Tensor& q, const torch::Tensor& w, const torch::Tensor& keys,
             const torch::Tensor& positions, int ratio) {
  const int tokens = int(out.size(0));
  const int rows = tokens * H;
  const int nt = ((tokens * v41idx::KL + 31) / 32) * 32;
  const int64_t end = v41idx::live_end(positions.data_ptr<int64_t>()[tokens - 1], ratio, keys.size(0));
  const int64_t blocks = (keys.size(0) + v41idx::KB - 1) / v41idx::KB;
  const int grid = int(std::min<int64_t>(blocks, 2 * SMS));
  const auto* qp = reinterpret_cast<const __half*>(q.data_ptr());
  const auto* wp = reinterpret_cast<const __half*>(w.data_ptr());
  auto* op = reinterpret_cast<__half*>(out.data_ptr());
  std::vector<uint64_t> store((v41idx::smem_bytes(rows) + 64) / 8 + 2);
  unsigned char* smem = reinterpret_cast<unsigned char*>(store.data());
  for (int b = 0; b < grid; ++b) {
    int64_t n0 = int64_t(b) * v41idx::KB;
    if (n0 >= end) continue;
    const v41idx::Smem s = v41idx::carve(smem, rows);
    for (int tid = 0; tid < nt; ++tid) v41idx::stage_q(tid, nt, qp, wp, rows, s.qs, s.ws);
    for (; n0 < end; n0 += int64_t(grid) * v41idx::KB) {
      for (int tid = 0; tid < nt; ++tid) v41idx::stage_keys(tid, nt, keys.data_ptr<uint8_t>(), n0, end, s.ks);
      for (int tid = 0; tid < nt; ++tid) {
        const int t = tid / v41idx::KL, kl = tid % v41idx::KL;
        if (t < tokens) v41idx::tile<H>(t, kl, s.qs, s.ks, s.ws, n0, end, op, out.size(1));
      }
    }
  }
}

void index_scores_tiled(torch::Tensor out, torch::Tensor q, torch::Tensor w, torch::Tensor keys,
                        torch::Tensor positions, int64_t heads, int64_t ratio) {
  const int64_t tokens = out.dim() == 2 ? out.size(0) : -1;
  TORCH_CHECK(!out.is_cuda(), "v41idx3_cpu_ops: CPU tensors only");
  TORCH_CHECK(out.dim() == 2 && out.scalar_type() == torch::kHalf && out.stride(1) == 1 &&
                  out.stride(0) == out.size(1),
              "v41idx: out must be fp16 [tokens, n_out] with row stride n_out");
  TORCH_CHECK(heads == 4 || heads == 8, "v41idx: tiled scorer supports 4 or 8 index heads per rank, got ", heads);
  TORCH_CHECK(tokens >= 1 && tokens * heads <= 64, "v41idx: need 1 <= tokens * heads <= 64");
  TORCH_CHECK(q.scalar_type() == torch::kHalf && q.is_contiguous() && q.dim() == 2 &&
                  q.size(0) == tokens * heads && q.size(1) == 128,
              "v41idx: q must be contiguous fp16 [tokens * heads, 128]");
  TORCH_CHECK(w.scalar_type() == torch::kHalf && w.is_contiguous() && w.numel() == tokens * heads,
              "v41idx: w must be contiguous fp16 [tokens * heads]");
  TORCH_CHECK(keys.scalar_type() == torch::kUInt8 && keys.is_contiguous() && keys.dim() == 2 &&
                  keys.size(1) == 68 && keys.size(0) > 0,
              "v41idx: keys must be contiguous uint8 [capacity > 0, 68]");
  TORCH_CHECK(positions.scalar_type() == torch::kInt64 && positions.is_contiguous() &&
                  positions.numel() >= tokens,
              "v41idx: positions must be contiguous int64 with >= tokens entries");
  TORCH_CHECK(out.size(1) >= keys.size(0), "v41idx: output width must cover key capacity");
  TORCH_CHECK(ratio >= 1, "v41idx: ratio must be >= 1");
  if (heads == 8) run_ref<8>(out, q, w, keys, positions, int(ratio));
  else run_ref<4>(out, q, w, keys, positions, int(ratio));
}

torch::Tensor tc_cell(torch::Tensor c, torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(c.scalar_type() == torch::kFloat && c.is_contiguous() && c.dim() == 1, "tc_cell: c float32 [n]");
  const int64_t n = c.size(0);
  for (const torch::Tensor* p : {&a, &b})
    TORCH_CHECK(p->scalar_type() == torch::kFloat && p->is_contiguous() && p->dim() == 2 && p->size(0) == n &&
                    p->size(1) == 4,
                "tc_cell: a and b float32 [n, 4]");
  torch::Tensor d = torch::empty({n}, torch::kFloat);
  const float* cp = c.data_ptr<float>();
  const float* ap = a.data_ptr<float>();
  const float* bp = b.data_ptr<float>();
  float* dp = d.data_ptr<float>();
  for (int64_t i = 0; i < n; ++i) dp[i] = v41idx3::tc_cell(cp[i], ap + 4 * i, bp + 4 * i);
  return d;
}

int64_t version() { return 1; }

}  // namespace

TORCH_LIBRARY(v41idx, m) {
  m.def(
      "index_scores_tiled(Tensor(a!) out, Tensor q, Tensor w, Tensor keys, Tensor positions, int heads, "
      "int ratio) -> ()",
      &index_scores_tiled);
  m.def("version() -> int", &version);
}

TORCH_LIBRARY(v41idx3, m) {
  m.def(
      "index_scores_tc(Tensor(a!) out, Tensor q, Tensor w, Tensor keys, Tensor positions, int heads, int ratio, "
      "int variant, Tensor(b!) counters) -> ()",
      &index_scores_tc);
  m.def("tc_cell(Tensor c, Tensor a, Tensor b) -> Tensor", &tc_cell);
  m.def("version() -> int", &version);
}
