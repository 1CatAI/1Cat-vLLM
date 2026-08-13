// SPDX-License-Identifier: MIT
// Adapted from ref/V100-SM_70-Flash-Attn-2-v1 (fa2_sm70), Copyright (c) 2026 xormal
// (Alexander Romanoff), MIT license:
//   - fa2_sm70/csrc/volta_gdn_h.cuh          (the chunk-state kernel and its analysis)
//   - fa2_src/fmha_kernel/gemm/volta_warp_mma.h  (the minimal warp-MMA subset below)
// Adaptation for monico-vllm (patches/0100): English comments, self-contained single
// translation unit, torch JIT-extension entry point, BV/warp-shape instantiations chosen
// for the TP4 per-rank GDN shape (H=8, Hg=4, K=V=128, BT=64) instead of the reference's
// H=24, and PIPE exposed as a runtime-selected instantiation.
//
// WHAT THIS REPLACES. chunk_gated_delta_rule_fwd_kernel_h_blockdim64 in
// vllm/model_executor/layers/fla/ops/chunk_delta_h.py. Measured on our tree (patches/0100
// gate 1, SASS census in patches/0094 lineage): Triton targeting sm_70 emits ZERO tensor
// instructions for tl.dot — the chunk GEMMs run as scalar FP32 FMA. Volta's HMMA
// (mma.sync.aligned.m8n8k4.f32.f16.f16.f32) is idle on this path. This kernel computes the
// SAME recurrence on the tensor cores, fp16 operands with fp32 accumulation — the same
// input/accumulate precision classes as the Triton kernel (which also feeds tl.dot fp16
// operands and accumulates fp32); only the accumulation ORDER differs, which is why the
// record gates it against an fp64 CPU reference.
//
// THE RECURRENCE (chunk BT=64; state H_c laid out [v][k]; identical to the Triton kernel,
// including both rounding traps):
//     (a) h[c]    = fp16(H_c)                       -- state at the START of the chunk
//     (b) y       = u - w x fp16(H_c)^T             -- GEMM1, [BT,DK] x [DK,BV]
//     (c) v_new   = fp16(y)                         -- UNGATED
//     (d) vt      = fp16(y * exp(gL - g_t))         -- gated, a SECOND independent rounding
//     (e) H_{c+1} = exp(gL) * H_c + k^T x vt        -- GEMM2, [BV,BT] x [BT,DK]
// Trap 1: v_new is written BEFORE the gate, so fp16(y) and fp16(y*gamma) are TWO distinct
// roundings of one fp32 value — do not "save" one. Trap 2: the fp16 rounding of H_c in (a)
// and in (b) is ONE AND THE SAME, so the state is converted once into shared memory and that
// buffer feeds both the h store and GEMM1's B operand. (Both match the Triton kernel:
// tl.store(h, b_h.to(f16)) and tl.dot(w, trans(b_h).to(f16)) round the same fp32 state.)
//
// WORK DECOMPOSITION. The recurrence is STRICTLY sequential across chunks; parallelism
// exists only over (sequence, value head, DV slice). Our per-rank shape has H=8 value heads,
// so at N=1 the grid is (DV/BV) * 8 blocks on 80 SMs: BV=64 -> 16 blocks, BV=32 -> 32,
// BV=16 -> 64. A block owns rows [i_v*BV, i_v*BV+BV) of V and ALL of K. Splitting adds no
// MACs but multiplies w/k reads linearly — which BV wins at OUR occupancy is measured by the
// PoC, not assumed from the reference (whose H=24 favored BV=64).
//
// OPERAND ORIENTATIONS (from the reference's measured maps; names are counter-intuitive):
//   GEMM1 y[t][v] = SUM_k w[t][k]*Hhat[v][k]: A=w is [M=t][K=k] -> kKMajor; B=Hhat is
//         [N=v][K=k], k contiguous -> kKMajor. Instruction .row.col.
//   GEMM2 H[v][k] += SUM_t vt[t][v]*k[t][k]: the accumulator is held as [v][k] — the SAME
//         layout needed for the h store and for GEMM1's operand, so the state is never
//         repacked. Then A=vt is [K=t][M=v], m contiguous -> kMMajor; B=k is [K=t][N=k],
//         n contiguous -> kNMajor. Instruction .col.row.
//
// ROW PADDING IS COMPUTED PER BANK-CONFLICT LAW, per operand (reference header, verified by
// their SASS/bank analysis; our numeric gate validates end-to-end on our silicon):
//   * k-major operands (sW, sHh): need LD halves with LD/2 = 2u, u odd -> LD = DK+4 = 132.
//     264 B row stride is not 16 B-aligned, so wide STS is illegal there — two uint2 stores.
//   * m/n-major operands (sVt, sK): need (LD/2) mod 32 == 8 -> LDK = DK+16 = 144 (288 B,
//     16 B-aligned: wide store legal), LDT = gdn_h_ldt(BV).
//   * sU is read on the ACCUMULATOR map; LDU = BV+4 spreads all 32 lanes across banks.
//
// v_new IS WRITTEN THROUGH sU: y lives in accumulator registers; a direct global store would
// be scattered STG.U16. Each (t,v) cell is owned by EXACTLY ONE lane (coverage property of
// the accumulator map), and that same lane read u for that cell — so fp16(y) is written OVER
// the already-consumed u in shared memory (same address, same thread: no race), then sU is
// flushed to global in one coalesced pass. Costs no extra buffer.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace monico_sm70_gdn {

// ------------------------------------------------------------------------------------------
// Minimal Volta warp-MMA subset (from volta_warp_mma.h; i8/swizzle/preload variants dropped).
// The operand-feed maps below were MEASURED by the reference (hmma_map_probe over all four
// layout variants, coverage-asserted: every output cell stamped exactly once); our PoC gate
// re-validates them end-to-end against an fp64 CPU reference on our silicon.
// ------------------------------------------------------------------------------------------
enum class BLayout { kKMajor, kNMajor };
enum class ALayout { kKMajor, kMMajor };

#define MONICO_HMMA_ASM(sfx)                                                                \
  asm("mma.sync.aligned.m8n8k4." sfx ".f32.f16.f16.f32 "                                    \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"         \
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),                                     \
        "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])                                      \
      : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(b[1]))

template <BLayout BL, ALayout AL = ALayout::kKMajor>
__device__ __forceinline__ void hmma_884_b(float (&d)[8], const uint32_t (&a)[2],
                                           const uint32_t (&b)[2]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700) && (__CUDA_ARCH__ < 800)
  if constexpr (AL == ALayout::kKMajor) {
    if constexpr (BL == BLayout::kKMajor) { MONICO_HMMA_ASM("row.col"); }
    else                                  { MONICO_HMMA_ASM("row.row"); }
  } else {
    if constexpr (BL == BLayout::kKMajor) { MONICO_HMMA_ASM("col.col"); }
    else                                  { MONICO_HMMA_ASM("col.row"); }
  }
#else
  (void)d; (void)a; (void)b;
#endif
}

// Operand feeds for a DENSE 16x16 per instruction (all measured by the reference).
__device__ __forceinline__ int wmma_a_row(int lane)  { return (lane & 3) | ((lane & 16) >> 2) | (lane & 8); }
__device__ __forceinline__ int wmma_a_krow(int lane) { return lane & 3; }
__device__ __forceinline__ int wmma_a_mbase(int lane) {
  const int q = lane >> 2;
  return 4 * ((q >> 2) & 1) + 8 * ((q >> 1) & 1);
}
__device__ __forceinline__ int wmma_b_col(int lane)  { return (lane & 3) | ((lane & 16) >> 2) | ((lane & 4) << 1); }
__device__ __forceinline__ int wmma_b_krow(int lane) { return lane & 3; }
__device__ __forceinline__ int wmma_b_nbase(int lane) {
  const int q = lane >> 2;
  return 4 * ((q >> 2) & 1) + 8 * (q & 1);
}
// Accumulator: the SAME map for every layout variant.
__device__ __forceinline__ int wmma_acc_row(int lane, int reg) {
  return ((lane & 1) | ((lane & 16) >> 2) | (lane & 8)) + 2 * ((reg >> 1) & 1);
}
__device__ __forceinline__ int wmma_acc_col(int lane, int reg) {
  return ((lane & 2) | ((lane & 4) << 1)) + (reg & 1) + 4 * ((reg >> 2) & 1);
}

// Warp-level MMA over a (16*MB) x (16*NB) tile; K walked in steps of 4, fragments
// double-buffered so shared-memory latency of step k+4 hides under step k's tensor ops.
template <int MB, int NB, BLayout BL, ALayout AL = ALayout::kKMajor>
struct WarpMma {
  float acc[MB][NB][8];

  __device__ __forceinline__ void clear() {
#pragma unroll
    for (int i = 0; i < MB; ++i)
#pragma unroll
      for (int j = 0; j < NB; ++j)
#pragma unroll
        for (int r = 0; r < 8; ++r) acc[i][j][r] = 0.f;
  }

  __device__ __forceinline__ const __half* a_base(const __half* sA, int lda, int row0,
                                                  int lane) const {
    if constexpr (AL == ALayout::kKMajor) return sA + (long)(row0 + wmma_a_row(lane)) * lda;
    else return sA + (long)wmma_a_krow(lane) * lda + row0 + wmma_a_mbase(lane);
  }
  __device__ __forceinline__ const __half* b_base(const __half* sB, int ldb, int col0,
                                                  int lane) const {
    if constexpr (BL == BLayout::kKMajor) return sB + (long)(col0 + wmma_b_col(lane)) * ldb;
    else return sB + (long)wmma_b_krow(lane) * ldb + col0 + wmma_b_nbase(lane);
  }
  __device__ __forceinline__ void load_a_p(uint32_t (&fa)[MB][2], const __half* pa, int lda,
                                           int k) const {
    const __half* p = (AL == ALayout::kKMajor) ? (pa + k) : (pa + (long)k * lda);
#pragma unroll
    for (int i = 0; i < MB; ++i) {
      const __half* q = (AL == ALayout::kKMajor) ? (p + (long)(i * 16) * lda) : (p + i * 16);
      const uint2 v = *reinterpret_cast<const uint2*>(q);
      fa[i][0] = v.x; fa[i][1] = v.y;
    }
  }
  __device__ __forceinline__ void load_b_p(uint32_t (&fb)[NB][2], const __half* pb, int ldb,
                                           int k) const {
    const __half* p = (BL == BLayout::kKMajor) ? (pb + k) : (pb + (long)k * ldb);
#pragma unroll
    for (int j = 0; j < NB; ++j) {
      const __half* q = (BL == BLayout::kKMajor) ? (p + (long)(j * 16) * ldb) : (p + j * 16);
      const uint2 v = *reinterpret_cast<const uint2*>(q);
      fb[j][0] = v.x; fb[j][1] = v.y;
    }
  }
  __device__ __forceinline__ void mma(const uint32_t (&fa)[MB][2], const uint32_t (&fb)[NB][2]) {
#pragma unroll
    for (int i = 0; i < MB; ++i)
#pragma unroll
      for (int j = 0; j < NB; ++j)
        hmma_884_b<BL, AL>(acc[i][j], fa[i], fb[j]);
  }

  template <int K>
  __device__ __forceinline__ void accumulate(const __half* sA, int lda, int row0,
                                             const __half* sB, int ldb, int col0, int lane) {
    const __half* pa = a_base(sA, lda, row0, lane);
    const __half* pb = b_base(sB, ldb, col0, lane);
    uint32_t fa[2][MB][2], fb[2][NB][2];
    load_a_p(fa[0], pa, lda, 0);
    load_b_p(fb[0], pb, ldb, 0);
#pragma unroll
    for (int k = 0; k < K; k += 4) {
      const int c = (k >> 2) & 1;
      if (k + 4 < K) {
        load_a_p(fa[c ^ 1], pa, lda, k + 4);
        load_b_p(fb[c ^ 1], pb, ldb, k + 4);
      }
      mma(fa[c], fb[c]);
    }
  }

  template <typename F>
  __device__ __forceinline__ void visit(int lane, F&& f) {
#pragma unroll
    for (int i = 0; i < MB; ++i)
#pragma unroll
      for (int j = 0; j < NB; ++j)
#pragma unroll
        for (int r = 0; r < 8; ++r)
          f(acc[i][j][r], i * 16 + wmma_acc_row(lane, r), j * 16 + wmma_acc_col(lane, r));
  }
};

// ------------------------------------------------------------------------------------------
// The chunk-state kernel (volta_gdn_h.cuh, adapted).
// ------------------------------------------------------------------------------------------

// sVt row stride: >= BV and (LD/2) mod 32 == 8 (see header).
__host__ __device__ constexpr int gdn_h_ldt(int bv) { return ((bv - 16 + 63) / 64) * 64 + 16; }

// Shared-memory size, computed by both host (for cudaFuncSetAttribute) and kernel.
template <int BT, int DK, int BV>
constexpr size_t gdn_h_smem() {
  return (size_t)(BT * (DK + 4) + BV * (DK + 4) + BT * (DK + 16) + BT * gdn_h_ldt(BV) +
                  BT * (BV + 4)) * sizeof(__half);
}

// BT — chunk size (64); DK/DV — head dims; BV — this block's V slice; WM1 x WN1 — warp grid
// over GEMM1's output [BT][BV]; WM2 x WN2 — over the state [BV][DK]; PIPE — separate the
// next chunk's LDG issue from its STS (the reference measured +20..46% from exactly this
// separation; the cost is (2*BT*DK/8 + BT*BV/4)/NTHR staging vectors per thread of register
// pressure, so PIPE=false is also instantiated and the PoC picks per BV).
template <int BT, int DK, int DV, int BV, int WM1, int WN1, int WM2, int WN2, bool PIPE = true>
__global__ __launch_bounds__(WM1 * WN1 * 32, 1) void volta_gdn_chunk_h(
    const __half* __restrict__ kg,     // k   [T, HG, DK]      -- key head
    const __half* __restrict__ wg,     // w   [T, H,  DK]
    const __half* __restrict__ ug,     // u   [T, H,  DV]
    const float*  __restrict__ gg,     // g   [T, H]           -- cumulative log gate, fp32
    const float*  __restrict__ h0g,    // h0  [N, H, DV, DK]   -- fp32, may be nullptr
    __half* __restrict__ hg,           // h   [NT_all, H, DV, DK]
    __half* __restrict__ vng,          // v_new [T, H, DV]
    float*  __restrict__ htg,          // ht  [N, H, DV, DK]   -- fp32, may be nullptr
    const int* __restrict__ cu,        // cu_seqlens    [N+1]
    const int* __restrict__ coff,      // chunk_offsets [N]
    int H, int HG) {
  constexpr int LDW = DK + 4;          // k-major A: conflict-free, wide store illegal
  constexpr int LDH = DK + 4;          // k-major B: same
  constexpr int LDK = DK + 16;         // n-major B: stride 16 (mod 64), wide store legal
  constexpr int LDT = gdn_h_ldt(BV);   // m-major A: same law
  constexpr int LDU = BV + 4;          // read on the accumulator map
  constexpr int NW = WM1 * WN1;
  constexpr int NTHR = NW * 32;
  constexpr int SM1 = BT / WM1, SN1 = BV / WN1;
  constexpr int SM2 = BV / WM2, SN2 = DK / WN2;
  static_assert(WM1 * WN1 == WM2 * WN2, "both phases run on the same warps");
  static_assert(SM1 % 16 == 0 && SN1 % 16 == 0, "GEMM1 warp tile multiple of fragment step");
  static_assert(SM2 % 16 == 0 && SN2 % 16 == 0, "GEMM2 warp tile multiple of fragment step");
  static_assert(DK % 8 == 0 && BV % 8 == 0, "staging moves vectors");

  extern __shared__ __half smem[];
  __half* sW  = smem;                  // [BT][LDW]  w window   (GEMM1 operand A)
  __half* sHh = sW  + BT * LDW;        // [BV][LDH]  fp16 state (GEMM1 operand B AND h output)
  __half* sK  = sHh + BV * LDH;        // [BT][LDK]  k window   (GEMM2 operand B)
  __half* sVt = sK  + BT * LDK;        // [BT][LDT]  gated value (GEMM2 operand A)
  __half* sU  = sVt + BT * LDT;        // [BT][LDU]  first u, THEN v_new (see header)
  __shared__ float sGam[BT];

  const int i_v = blockIdx.x, i_nh = blockIdx.y;
  const int i_n = i_nh / H, hv = i_nh % H;
  const int hk = hv / (H / HG);        // value head -> key head map, BLOCKED (as Triton's
                                       // i_h // (H // Hg))
  const int bos = cu[i_n], T = cu[i_n + 1] - bos;
  const int NT = (T + BT - 1) / BT;
  const int boh = coff[i_n];
  const int v0 = i_v * BV;

  const int sw = H * DK, su = H * DV, sk = HG * DK, sg = H;
  const long sh = (long)H * DV * DK;
  const __half* pW = wg + ((long)bos * H + hv) * DK;
  const __half* pK = kg + ((long)bos * HG + hk) * DK;
  const __half* pU = ug + ((long)bos * H + hv) * DV;
  __half*       pV = vng + ((long)bos * H + hv) * DV;
  const float*  pG = gg + (long)bos * H + hv;
  __half*       pH = hg + ((long)boh * H + hv) * (long)DV * DK;

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm1 = warp / WN1, wn1 = warp % WN1;
  const int wm2 = warp / WN2, wn2 = warp % WN2;
  const int r10 = wm1 * SM1, c10 = wn1 * SN1;   // this warp's tile corner in GEMM1
  const int r20 = wm2 * SM2, c20 = wn2 * SN2;   // ... and in the state

  // The state lives in REGISTERS for the whole chunk walk. Accumulator type/orientation is
  // fixed by GEMM2 (see header).
  WarpMma<SM2 / 16, SN2 / 16, BLayout::kNMajor, ALayout::kMMajor> st;
  st.clear();
  if (h0g != nullptr) {
    const float* p0 = h0g + (long)i_nh * DV * DK;
    st.visit(lane, [&](float& x, int r, int c) { x = p0[(long)(v0 + r20 + r) * DK + c20 + c]; });
  }

  // --- The chunk window travels through registers (PIPE): the next chunk's LDG is issued
  // right after the readiness barrier, its STS lands at the start of the NEXT iteration, so
  // both GEMMs, the epilogue and both global stores sit between issue and use.
  constexpr int NLW = BT * (DK / 8) / NTHR;
  constexpr int NLU = BT * (BV / 4) / NTHR;
  static_assert(NLW * NTHR == BT * (DK / 8), "w/k staging divides evenly across threads");
  static_assert(NLU * NTHR == BT * (BV / 4), "u staging divides evenly across threads");
  uint4 rw[NLW], rk[NLW];
  uint2 ru[NLU];
  float rgam = 0.f, rgexp = 1.f;

  // The gate is computed HERE, on the staging path: exp goes to the SFU (overlaps FP32 nearly
  // free) and leaves the critical path of the compute phase. Rows beyond T are zeroed BEFORE
  // the exponent — a neighboring sequence's g could be inf and a zero factor would not save it.
  auto ldg = [&](int tn) {
    const int rn = min(BT, T - tn), ln = tn + rn - 1;
    const float gl = pG[(long)ln * sg];
    rgexp = __expf(gl);
    if (tid < BT) rgam = (tid < rn) ? __expf(gl - pG[(long)(tn + tid) * sg]) : 0.f;
#pragma unroll
    for (int q = 0; q < NLW; ++q) {
      const int i = tid + q * NTHR, r = i / (DK / 8), c = (i % (DK / 8)) * 8;
      const bool ok = (tn + r) < T;
      rw[q] = ok ? *reinterpret_cast<const uint4*>(&pW[(long)(tn + r) * sw + c]) : uint4{0, 0, 0, 0};
      rk[q] = ok ? *reinterpret_cast<const uint4*>(&pK[(long)(tn + r) * sk + c]) : uint4{0, 0, 0, 0};
    }
#pragma unroll
    for (int q = 0; q < NLU; ++q) {
      const int i = tid + q * NTHR, r = i / (BV / 4), c = (i % (BV / 4)) * 4;
      ru[q] = ((tn + r) < T)
                  ? *reinterpret_cast<const uint2*>(&pU[(long)(tn + r) * su + v0 + c])
                  : uint2{0, 0};
    }
  };
  // Shared-store width is whatever the buffer's PADDING allows: sK's 288 B stride is 16 B
  // aligned (STS.128); sW's 264 B is not (two STS.64). Deliberate trade: conflict-free
  // fragment READS outnumber staging writes by an order of magnitude.
  auto sts = [&]() {
    if (tid < BT) sGam[tid] = rgam;
#pragma unroll
    for (int q = 0; q < NLW; ++q) {
      const int i = tid + q * NTHR, r = i / (DK / 8), c = (i % (DK / 8)) * 8;
      *reinterpret_cast<uint2*>(&sW[r * LDW + c])     = make_uint2(rw[q].x, rw[q].y);
      *reinterpret_cast<uint2*>(&sW[r * LDW + c + 4]) = make_uint2(rw[q].z, rw[q].w);
      *reinterpret_cast<uint4*>(&sK[r * LDK + c])     = rk[q];
    }
#pragma unroll
    for (int q = 0; q < NLU; ++q) {
      const int i = tid + q * NTHR, r = i / (BV / 4), c = (i % (BV / 4)) * 4;
      *reinterpret_cast<uint2*>(&sU[r * LDU + c]) = ru[q];
    }
  };

  if (PIPE && NT > 0) ldg(0);

  for (int it = 0; it < NT; ++it) {
    const int t0 = it * BT;   // chunk length and gL are staging-only — computed inside ldg()

    __syncthreads();                              // last chunk's GEMM2 finished reading smem

    if (!PIPE) ldg(t0);
    sts();
    const float gexp = rgexp;                     // snapshot: next staging overwrites rgexp

    // --- (a) state -> fp16 ONCE: feeds both the h store and GEMM1's operand B.
    // The DECAY (e) is applied IN THE SAME PASS over the accumulator: it does not depend on
    // the stored value, and folding it here removes a second sweep and pulls the FMUL chain
    // off the critical path before GEMM2. Register pairs (rg, rg+1) are ADJACENT columns
    // (accumulator map), so the store is 4 B wide.
#pragma unroll
    for (int i = 0; i < SM2 / 16; ++i)
#pragma unroll
      for (int j = 0; j < SN2 / 16; ++j)
#pragma unroll
        for (int rg = 0; rg < 8; rg += 2) {
          const int r = r20 + i * 16 + wmma_acc_row(lane, rg);
          const int c = c20 + j * 16 + wmma_acc_col(lane, rg);
          *reinterpret_cast<__half2*>(&sHh[r * LDH + c]) =
              __floats2half2_rn(st.acc[i][j][rg], st.acc[i][j][rg + 1]);
          st.acc[i][j][rg]     *= gexp;
          st.acc[i][j][rg + 1] *= gexp;
        }
    __syncthreads();

    // ISSUE EARLY: the next chunk's window is requested HERE and lands in shared memory only
    // at the start of the next iteration — both GEMMs sit in between.
    if (PIPE && it + 1 < NT) ldg(t0 + BT);

    // --- h store. Slice [v0, v0+BV) x all DK is CONTIGUOUS in global (row stride exactly
    // DK) — fully coalesced; sHh is read 4 values wide (the +4 padding forbids wider).
    {
      __half* dst = pH + (long)it * sh + (long)v0 * DK;
      for (int i = tid; i < BV * (DK / 4); i += NTHR) {
        const int r = i / (DK / 4), c = (i % (DK / 4)) * 4;
        *reinterpret_cast<uint2*>(&dst[(long)r * DK + c]) =
            *reinterpret_cast<const uint2*>(&sHh[r * LDH + c]);
      }
    }

    // --- (b) GEMM1: acc = w x Hhat^T. Both operands k-major.
    WarpMma<SM1 / 16, SN1 / 16, BLayout::kKMajor> m1;
    m1.clear();
    m1.template accumulate<DK>(sW, LDW, r10, sHh, LDH, c10, lane);

    // --- (c)+(d) epilogue. Unrolled EXPLICITLY (accumulator registers must be indexed by
    // compile-time constants); pairs (rg, rg+1) are adjacent columns of one row, so the u
    // read and both stores go in 4 B pairs.
#pragma unroll
    for (int i = 0; i < SM1 / 16; ++i)
#pragma unroll
      for (int j = 0; j < SN1 / 16; ++j)
#pragma unroll
        for (int rg = 0; rg < 8; rg += 2) {
          const int t = r10 + i * 16 + wmma_acc_row(lane, rg);
          const int c = c10 + j * 16 + wmma_acc_col(lane, rg);
          const float y0 = __half2float(sU[t * LDU + c])     - m1.acc[i][j][rg];
          const float y1 = __half2float(sU[t * LDU + c + 1]) - m1.acc[i][j][rg + 1];
          // v_new — UNGATED, written over the already-consumed u (same address, same thread)
          *reinterpret_cast<__half2*>(&sU[t * LDU + c]) = __floats2half2_rn(y0, y1);
          const float gm = sGam[t];
          // vt — a SECOND, independent rounding of the same fp32 value
          *reinterpret_cast<__half2*>(&sVt[t * LDT + c]) = __floats2half2_rn(y0 * gm, y1 * gm);
        }
    __syncthreads();

    // --- v_new flush, coalesced (see header for why not straight from registers)
    for (int i = tid; i < BT * (BV / 4); i += NTHR) {
      const int r = i / (BV / 4), c = (i % (BV / 4)) * 4;
      if ((t0 + r) < T)
        *reinterpret_cast<uint2*>(&pV[(long)(t0 + r) * su + v0 + c]) =
            *reinterpret_cast<const uint2*>(&sU[r * LDU + c]);
    }

    // --- (f) outer product. Decay (e) was already applied above, fused with the h unload.
    st.template accumulate<BT>(sVt, LDT, r20, sK, LDK, c20, lane);
  }

  if (htg != nullptr) {
    float* pt = htg + (long)i_nh * DV * DK;
    st.visit(lane, [&](float& x, int r, int c) { pt[(long)(v0 + r20 + r) * DK + c20 + c] = x; });
  }
}

// ------------------------------------------------------------------------------------------
// Host entry. Fixed to the shape this record proves: BT=64, DK=DV=128. BV and PIPE are
// runtime-selected among compiled instantiations (the PoC measures the winner at the served
// per-rank shape H=8; the Python wrapper's default encodes it).
// ------------------------------------------------------------------------------------------
namespace {

constexpr int BT = 64;
constexpr int DK = 128;
constexpr int DV = 128;

#define CHK(t, dt, name)                                                              \
  TORCH_CHECK((t).is_cuda() && (t).is_contiguous(), name " must be contiguous CUDA"); \
  TORCH_CHECK((t).scalar_type() == (dt), name " has wrong dtype")

template <int BV, int WM1, int WN1, int WM2, int WN2, bool PIPE>
void launch(const torch::Tensor& k, const torch::Tensor& w, const torch::Tensor& u,
            const torch::Tensor& g, const c10::optional<torch::Tensor>& h0,
            torch::Tensor& h, torch::Tensor& vn, torch::Tensor& ht, bool has_ht,
            const torch::Tensor& cu, const torch::Tensor& coff, int N, int H, int Hg) {
  const size_t sh = gdn_h_smem<BT, DK, BV>();
  auto kern = volta_gdn_chunk_h<BT, DK, DV, BV, WM1, WN1, WM2, WN2, PIPE>;
  // Opt-in above the 48 KB default once per instantiation (89 KB cap would fail loudly).
  static bool attr_done = false;
  if (!attr_done) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        (int)sh));
    attr_done = true;
  }
  kern<<<dim3(DV / BV, N * H), WM1 * WN1 * 32, sh, at::cuda::getCurrentCUDAStream()>>>(
      (const __half*)k.data_ptr(), (const __half*)w.data_ptr(), (const __half*)u.data_ptr(),
      g.data_ptr<float>(), h0.has_value() ? h0->data_ptr<float>() : nullptr,
      (__half*)h.data_ptr(), (__half*)vn.data_ptr(),
      has_ht ? ht.data_ptr<float>() : nullptr,
      cu.data_ptr<int>(), coff.data_ptr<int>(), H, Hg);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// h [1,NT,H,DV,DK] fp16 (state at the START of each chunk); v_new like u; ht [N,H,DV,DK]
// fp32 when output_final_state. Mirrors chunk_gated_delta_rule_fwd_h's varlen contract.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fwd_h(
    torch::Tensor k, torch::Tensor w, torch::Tensor u, torch::Tensor g,
    c10::optional<torch::Tensor> h0, torch::Tensor cu_seqlens, torch::Tensor chunk_offsets,
    int64_t NT, bool output_final_state, int64_t bv, bool pipe) {
  const c10::cuda::CUDAGuard guard(k.device());
  CHK(k, torch::kHalf, "k"); CHK(w, torch::kHalf, "w"); CHK(u, torch::kHalf, "u");
  CHK(g, torch::kFloat, "g");
  CHK(cu_seqlens, torch::kInt, "cu_seqlens"); CHK(chunk_offsets, torch::kInt, "chunk_offsets");
  // h0 is read as const float*: the dtype MUST be checked here — a half buffer read as float
  // is not a crash but a silently wrong answer.
  if (h0.has_value()) { CHK(h0.value(), torch::kFloat, "initial_state"); }
  const int Hg = (int)k.size(-2), H = (int)u.size(-2);
  TORCH_CHECK(k.size(-1) == DK && u.size(-1) == DV, "head dims must be 128");
  TORCH_CHECK(H % Hg == 0, "H must be a multiple of Hg");
  const int N = (int)cu_seqlens.size(0) - 1;
  auto h = torch::empty({1, (long)NT, H, DV, DK}, torch::dtype(torch::kHalf).device(k.device()));
  auto vn = torch::empty_like(u);
  auto ht = output_final_state
                ? torch::empty({N, H, DV, DK}, torch::dtype(torch::kFloat).device(k.device()))
                : torch::Tensor();
  switch (bv) {
    case 64:
      if (pipe) launch<64, 2, 4, 2, 4, true >(k, w, u, g, h0, h, vn, ht, output_final_state, cu_seqlens, chunk_offsets, N, H, Hg);
      else      launch<64, 2, 4, 2, 4, false>(k, w, u, g, h0, h, vn, ht, output_final_state, cu_seqlens, chunk_offsets, N, H, Hg);
      break;
    case 32:
      if (pipe) launch<32, 4, 2, 2, 4, true >(k, w, u, g, h0, h, vn, ht, output_final_state, cu_seqlens, chunk_offsets, N, H, Hg);
      else      launch<32, 4, 2, 2, 4, false>(k, w, u, g, h0, h, vn, ht, output_final_state, cu_seqlens, chunk_offsets, N, H, Hg);
      break;
    case 16:
      if (pipe) launch<16, 4, 1, 1, 4, true >(k, w, u, g, h0, h, vn, ht, output_final_state, cu_seqlens, chunk_offsets, N, H, Hg);
      else      launch<16, 4, 1, 1, 4, false>(k, w, u, g, h0, h, vn, ht, output_final_state, cu_seqlens, chunk_offsets, N, H, Hg);
      break;
    default:
      TORCH_CHECK(false, "unsupported BV (compiled: 16, 32, 64)");
  }
  return {h, vn, ht};
}

}  // namespace

}  // namespace monico_sm70_gdn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fwd_h", &monico_sm70_gdn::fwd_h,
        "SM70 tensor-core chunk_gated_delta_rule_fwd_h (varlen, BT=64, K=V=128)");
}
