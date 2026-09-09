# FlashAttention development focus

Current delivery uses the retained FlashAttention-V100 implementation on
already working H3 dense workflows. At the user's latest request, further
investigation of slower Attention prototypes and new workflow expansion are
paused. Validated FlashInfer remains available, with further FI optimization
and exhaustive FI acceptance paused. The unfinished matrix and unchanged
quality/performance gates remain recorded as incomplete.

See [CURRENT_STATUS.md](CURRENT_STATUS.md) for the retained configuration,
measured workflow coverage and delivery limits.

With native TP4 peer rows, shared pageable VAE weights and a complete request
warmup plus three unprofiled requests, FA records a minimum-card median
53.235745 useful TFLOP/s and 58.293218 seconds denoise. The corresponding FI
measurements are 50.898828 TFLOP/s and 60.969633 seconds. Both pass native
latent/RGB/PCM preservation and fail the >80 throughput gate. Independent
official and human quality gates remain pending.

## Evidence guiding the next operator change

A development-only FA timer samples thread zero in head zero and every 32nd
query CTA. Nine boundary lengths and the actual [1,34551,14,128] captured
input remain bitwise equal to the retained FA implementation. Instrumented
operator median is 150.045700 ms versus 149.676025 ms without instrumentation
(0.247% overhead). Main-shape compilation uses 250 registers and no spills.
The query-64/key-128 diagnostic specialization spills and is not measured.

| Sampled phase | Fraction of sampled warp spans |
| --- | ---: |
| QK including operand loads and barrier | 38.85% |
| V prefetch and softmax | 18.76% |
| Probability stores and publication | 7.94% |
| PV including operand loads and barrier | 33.20% |
| Iteration join | 1.25% |

These phase spans include scheduling and waits; they are not whole-kernel
critical-path percentages or formal model performance. The measurement guides
operand and register-lifetime work, without qualifying a faster configuration.

The primary 34,560-row GEMM diagnostic also compares all 18 returned eligible
zero-workspace, no-split, FP32-accumulation Lt choices across QKV, FC1, output
projection and FC2. Timed alternatives preserve outputs bitwise, and the
existing algorithm 21/tile 24 remains the fastest choice in each shape.
No GEMM plan change is justified by this measurement.

## Rejected candidates

- Normal-range hardware exp2 retains the library outside [-126,0]. Boundary
  and actual-input outputs are bitwise, but paired median regresses from
  149.407745 to 158.803970 ms. No full-model run or source promotion follows.
- A new explicit WMMA Q16/K64-owner prototype retains FA K128 softmax panels,
  FP32 accumulation and output scaling. It lowers register use from 248 to
  128 without spilling and doubles a Q128 CTA to 512 threads. Nine boundary
  lengths and actual input are bitwise, but operator median regresses from
  147.507202 to 323.808258 ms. Register count alone is insufficient evidence
  of a speedup. This is distinct from the old invalid CUTLASS warp16 shape.
- Adding vector Q/K loads, V lane-exchange transpose and swizzled probability
  storage to that prototype spills 160 bytes per thread at the 128-register
  limit. The resource gate rejects it before GPU timing. A separate scoped
  staging experiment reduces the spill to 116 bytes with a 120-byte stack,
  which still fails the resource gate. Neither staging variant is GPU timed.

Exact code, binary hashes, clocks, numerical results and paired measurements
are retained under `/data/minimax-h3/sm70-general-20260909/` in
`attention-fa-phase-clock`, `gemm-primary-heuristics`,
`attention-fa-normal-exp2`, `attention-fa-warp16-native` and
`attention-fa-warp16-staging`. None is installed as a production replacement.

The vector-staging Q96 follow-up raises the per-thread register budget to 168.
Its only 4-byte local spill is stored before the key loop and reloaded after
its final back edge, as verified in SASS. This admits a bounded numerical and
performance probe without claiming zero spills. Nine boundaries and the
actual 34,551-token input are bitwise, but median regresses from 148.462585 to
281.825287 ms. This closes the split-key warp16 staging route.

A separate one-owner FA K128 prototype keeps rounded probability fragments
in registers, preserves the FA K64-half sum order and retains Q across key
panels. It uses 199 registers, no spills and 96 KiB shared memory. Nine
boundaries and the actual input are bitwise, but median is 195.892136 ms
versus 147.519104 ms. Its V-prefetch follow-up compiles to 240 registers with
no spills, but is not GPU timed: the parent exceeds the predeclared 10%
slowdown limit. No full-model run or production promotion follows.

Evidence: `attention-fa-warp16-staging-q96`, `attention-fa-register-k128`,
`attention-fa-register-k128-vprefetch`, and `fa-vprefetch-after-register.json`.

The artifact-only D128 tiled-GEMM port also closes without promotion. Padding
only PV's reduction width to 32 fixes its nonaligned-key vector read; the
original failing boundary then passes Compute Sanitizer with zero errors.
Eleven boundary/stress cases and the actual input pass the independent FP32
operator gate. Actual-input relative L2 is 0.000231 against that oracle and
0.000335 against FA; output is not bitwise equal to FA. Its paired median is
212.961273 ms versus retained FA's 149.299194 ms, so no complete sampling
quality run is justified.

A bounded Nsight Systems attribution records 209.361943 ms operator wall
time: QK plus tile softmax 142.181160 ms, PV plus probability rescaling
61.613049 ms, and GPU idle/host gaps 0.582682 ms. API durations overlap GPU
execution and cannot be added to those kernel durations. Packing or CPU
scheduling does not explain this regression. A row-maximum epilogue follow-up
completed its CPU build but was stopped at the user's scope change before
any GPU numerical check or timing. Neither artifact replaces production FA.
Evidence: `attention-fa-tiled-gemm-d128` and `attention-fa-tiled-gemm-rowmax`.
