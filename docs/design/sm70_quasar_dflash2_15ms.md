# QUASAR DFlash2 TP4: quality-preserving 15 ms campaign

## Frozen contract

The target is a complete, unprofiled B1 verification round below 15 ms on
release1k and MBPP28. A round includes target execution, sampling, state
handling and the DFlash2 proposal. Target-only graph time is not this metric.

- Integration base: `56f534e672657a6c7599afd6c0dcb2e2c211b2e3`, `onecat/main`.
- Four V100-SXM2-32GB GPUs, TP4; no TP8 substitution.
- QUASAR NVFP4 target revision `d8e6fbfa3e3a78899b440222b827430045a05b44`;
  DFlash2 revision `dedf8df68adfb1afeaf7b7480c0a0243108177b4`.
- FP16 activation, E4M3 target KV, FP16 draft KV, FP32 logits, seven draft
  tokens and eight verification rows. Preserve the recurrent-state dtype.
- CUDA 12.8, Torch 2.10.0+cu128, Python 3.12.13; V2 runner and Flash-V100
  target/draft graphs. Model limit 262144, token budget 4096, capacity four,
  one live request, memory utilization 0.8, prefix caching and Mamba align.
- Temperature 1, top-k 20, top-p .95, natural EOS, thinking `xhigh`.
  release1k retains seed 20260925 and MBPP28 seed 0; speed output cap 1024.

The retained precision-preserving results, 18.892/18.435 ms, predate this
integration base. They establish the optimization gap, not the new baseline.
New measurements freeze source overlays and every loaded native library.

## First change: observe the actual recurrent input

`benchmarks.sm70_dflash2_state_audit.StateAuditExtension` is an explicit
diagnostic worker extension. It is enabled only when
`VLLM_SM70_DFLASH2_AUDIT_ROOT` is set. Its active-case file supplies a complete
forced token tape. It records native logits but forces continuation and
acceptance; these requests must never be used for speed, acceptance, or task
quality claims.

The earlier fixed-prefix audit did not preserve incoming conv/SSM state and
excluded prefill layer tensors with its eight-token dump limit. This extension
records convolution inputs and states, recurrent q/k/v/g/beta and states,
slot tables/selectors, positions and RNG states. In particular, the recurrent
input comes from `slot_table[request, accepted_selector - 1]`; column zero is
not a valid replacement. Padding remains marked invalid, distinct from live
slot zero. Snapshots own their storage and graph replays refresh them.

State tensors are copied to persistent device buffers inside existing opaque
GDN calls and exported at the sampler boundary. Set the ordinary Qwen layer
dump token limit high enough to include prefill, and collect all four ranks.
The extension changes diagnostic work and allocation; a diagnostic result is
not evidence that an uninstrumented serving path has identical timing or
arithmetic selection. Require same-configuration repeats before attribution.

## Ordered promotion gates

1. Close fixed-prefix A/A repeatability, including the first prefill difference.
2. Test the existing packed verifier and eliminate confirmed layout/state copies.
3. Profile and optimize the five actual TP4 QPN2 projection shapes.
4. Extend communication/Gemma normalization fusion to q8, then assess overlap.
5. Optimize draft FP16 computation and complete-round graph scheduling.

Copy/layout/scheduling changes require exact affected tensors, states, logits,
probabilities and acceptance. Arithmetic changes require an independent
FP32/FP64 oracle, distribution/EOS analysis, long-output checks and paired
acceptance noninferiority with no preallocated loss margin. The previous
4.33% same-configuration TV is an unresolved defect in reproducibility, never
a tolerance. Unexplained token or EOS changes block promotion.

Require three independent paired startups and five measured requests after
warmup per speed fixture. Report mean-round-cost medians, round tails, TTFT,
pure decode throughput, accepted drafts per round and emitted tokens per round
separately. Profiler service sums and overlapping phases are not additive
end-to-end savings. Long-context quality and performance remain separate gates.

## Provenance and current status

This scope differs from the open FP8 target campaign (#405) and independent
batch FlashInfer ports (#515/#523): it targets QUASAR NVFP4 B1/q8 and first
repairs the missing state evidence. FlashInfer mechanisms are studied at
`91bda04c66f7cb851e1ab3b78b9fecea644b9844`; no upstream SM75+ binary is used
as an SM70 replacement.

Artifacts for this campaign are retained under
`/data/minimax-h3/task-cache/v100-quasar-dflash2-15ms-20260908`.
`baseline-manifest.json` records the source overlay, native-library SHA256s
and loaded libraries on all four workers. The base vLLM DSO is an archived
compatible build, not a full rebuild of main. Flash-V100 and FlashQLA were
rebuilt from the frozen tree with CUDA 12.8/GCC 12. GPU clocks remain dynamic.
The initial campaign uses devices 4--7 with independent telemetry. Following
the September 8 host reboot, devices 4--7 host another service; new diagnostic
pairs use a fixed lease on devices 0--3. Results from the two GPU groups are
kept separate, and final speed pairs require a fresh baseline on the same group.

Fresh uninstrumented baseline, one independent startup and five measured
requests per fixture after warmup:

| Fixture | Median request-average complete round | Output tokens | Rounds |
| --- | ---: | ---: | ---: |
| release1k | 19.017 ms | 248 | 82 |
| MBPP28 | 18.567 ms | 270 | 60 |

Within this startup, each fixture's five output hashes match. The JSON smoke
passes. Three existing long-code cases stop naturally at 5235, 1084 and 1312
tokens; EvalPlus reports base 3/3 and plus 1/3. This small subset is a baseline,
not evidence of a quality improvement. The three-startup performance and
acceptance promotion gates remain outstanding.

### First reproducibility defect: autotuned prefill reduction

The corrected `audit-a1r3` and `audit-a2` captures each contain 144 records:
two fixed tapes, full prefill and 17 subsequent forwards, and four ranks.
The sampler may observe an extra pipelined forward beyond the API output cap;
these forced-accept diagnostics are not acceptance measurements.

`results/audit-aa.json` reports maximum post-sampling TV 0.043373242,
five changed top-p support rows and no top-1 flips. The first causal difference
is layer 0's prefill input RMSNorm on rank 2, before the GDN projection.
The input hidden states are bitwise equal, but 9 FP16 outputs differ for
MBPP28 and 17 for MBPP3, by at most 0.0009765625. Differences then enter the
prefill conv/SSM states and the first verifier's incoming SSM state.

Retained per-rank Inductor `.best_config` files establish that `audit-a1r3`
rank 2 selected R0_BLOCK=2048/16 warps, while the other three ranks and all
four `audit-a2` ranks selected R0_BLOCK=8192/16 warps. Replaying the actual
generated kernel with checkpoint weights and the captured MBPP28 input
exactly reproduces each arm, respectively (zero output-element mismatches).
Thus this observed drift comes from changing FP32 reduction order, not a
first difference in verifier state addressing. It does not establish that
every historical quality issue has the same cause.

Both reductions differ from a rounded FP64 oracle (46 and 51 elements in
this captured tensor). Selecting the more common configuration alone is not
a precision argument. An environment-only attempt with
`TORCHINDUCTOR_DETERMINISTIC=1` did not reach the current AOT compile path:
the generated kernel metadata still says `deterministic=False`. Its 2.517%
A/A TV therefore does not evaluate the actual deterministic mechanism. It is
recorded as a failed route hit, not a rejected numerical implementation.

### Opt-in fixed Gemma reduction

`VLLM_SM70_DFLASH2_FIXED_GEMMA_RMS=1` selects a fixed 8192-element, 16-warp
reduction for contiguous FP16 `[M, 5120]` inputs and FP16 weights, with either
no residual or an FP16 residual. The latter retains FP32 residual output.
The established FP32-residual fused path and unsupported shapes keep their
existing dispatch. The new flag defaults to zero.

The initial fixed-kernel A/A (`audit-fixed-norm-1r2` versus
`audit-fixed-norm-2`) has 144 records per arm: zero differing intermediates,
bitwise-equal logits and zero full/sampling TV. Comparing that candidate to
`audit-a2` exposed another arithmetic detail at MBPP3 step 8: three values
in layer 0 post-attention norm differ, eventually producing maximum sampling
TV 0.003018199. Top-p support and top-1 stay unchanged, which is insufficient
for acceptance. Its masked square and residual materialization boundary had
been removed, changing FMA contraction even with the same tile and warp count.

The corrected kernel preserves those boundaries. With the same checkpoint
weights and captured inputs, both norms and the residual now exactly match
`audit-a2` for all 36 case/step combinations (two full prefills plus all 17
verification steps per tape). The final corrected model comparison,
`audit-a2` versus `audit-fixed-norm-3`, also passes: all 144 records have
bitwise-equal intermediates and native logits, zero full/sampling TV and no
support or top-1 changes (`results/a2-versus-fixed-norm-3.json`).
No serving default is promoted.
The first AOT attempt also exposed an unresolved imported `tldevice` alias in
generated code. Using `tl.rsqrt` fixes code generation, and the test now runs
the actual Inductor backend rather than only Dynamo's eager backend.

Current focused norm/state tests: **28 pass on V100**, including graph replay,
irregular prefill versus q1/q8 row invariance, residual storage/precision and
FP64-reference checks. Recorded operator replay, source hashes and A/A results
are under `results/fixed-norm-*.json`; the actual model runs use isolated
compiler caches and the frozen native libraries.

Focused tests: seven pass on V100, including accepted-slot indexing, invalid
padding, owned snapshot storage, CUDA graph replay with changing selectors,
and incomplete/nonfinite capture rejection. Invalid early attempts are
retained separately: `audit-a1` failed to wrap a module; `audit-a1r2` had
stale warmup buffers and unreliable address-based layer identity. Neither
is used for attribution. Current records carry per-forward epochs and use
the caller's layer identity. Unused prefill conv-history bytes are not
automatically treated as live-state corruption.

Hardware NCU counters are currently unavailable: the driver sets
`RmProfilingAdminOnly=1`, and the available root helper only manages GPU
clocks. This does not block state, numerical, CUDA-event or Nsight Systems
work, but no counter-based bottleneck claim is made without those counters.

### Natural-output gate and packed verifier integration

The first uninstrumented fixed-norm startup has median complete-round costs
19.035 ms (release1k) and 18.719 ms (MBPP28), with five measured requests after
warmup. Its three long-code cases score base 3/3 and plus 1/3, matching the
initial baseline; JSON and all nine seed/structured-fixture pairs pass,
including the existing parallel-tool premature-EOS fixture. Token sequences
change versus the original unpinned startup: first flips are output positions
187 (release1k) and 8 (MBPP28), zero-based. Accepted drafts/round are 1.917 and
3.934, respectively. These are observations, not an acceptance noninferiority
pass; the small score set cannot clear the changed-output gate.

The initial packed on/off model run (`audit-packed-1`) is excluded as packed
parity evidence: it did not log an actual route hit and produced no additional
packed-verifier kernel specialization. Inspection finds two integration bugs:
the Qwen3.5 projection and in-place convolution retain a wider QKVZBA row
stride, rejected by the contiguous-only gate; and the packed bridge retains
the old FP16 beta default while the standard speculative path uses FP32 beta.

The candidate reads contiguous-feature, row-strided QKV directly using its
actual row stride, and explicitly materializes FP32 beta. The existing
default-off verifier flag still controls dispatch. Sixteen GPU parity cases
cover the old FP16-beta component contract and the actual runtime bridge with
FP32 beta, wider projection rows, q4/q8, B1/B2 and FP16/FP32 states. Projection
storage remains unchanged. The diagnostic now records per-forward kernel
route markers; the comparator can require a packed hit on every observed
layer/rank/verification step. Eight state-audit tests pass, including rejection
of an equal-output comparison with no candidate hit. The complete model
comparison (`audit-fixed-norm-3` versus `audit-packed-2`) now passes with
required per-forward packed hits: all 144 records, intermediate tensors,
states and logits are bitwise equal, with zero TV/support/top-1 changes.
See `results/packed-stride-ab.json` and the pinned source overlay in
`results/audit-packed-2-source.json`.

Twenty additional real-state cases exercise the new 4128-element QKV row
stride; all outputs, states, padding and projection storage match exactly.
Task-local NVIDIA Compute Sanitizer 2025.1.0 (CUDA package 12.8.93-1) memcheck
reports zero errors on these cases; racecheck reports zero errors or warnings.
This is an operator memory gate, not
a long-context or acceptance noninferiority gate.

Real-state component replay also covers 480 combinations of two layers, two
tapes, four ranks, three verifier steps, all eight accepted-slot selectors,
non-monotonic state IDs including zero, empty padded requests, strided state
pools and untouched retired slots. Those original runs supplied captured FP32
gates and contiguous QKV; they do not validate the previously incorrect runtime
bridge. All evidence remains independent of performance and natural acceptance.

The first uninstrumented packed startup measures 18.278 ms / 17.762 ms on
release1k / MBPP28 (five requests after warmup). It is **not promoted**:
relative to `fixed-speed-1`, first output flips occur at positions 123 / 8,
and accepted drafts/round change from 1.917 / 3.934 to 1.989 / 3.500.
The three-code subset still scores base 3/3 and plus 1/3, and nine structured
seed/fixture pairs pass, but these cannot clear the changed acceptance gate.
The original unpinned baseline also selected different first-layer reduction
blocks across ranks: 2048 on ranks 0/1/3 and 8192 on rank 2. The retained
compiler configurations are in `results/natural-baseline-norm-configs.json`.

The diagnostic supports `force_tokens=false` cases. These keep
the actual sampler outputs and record target auxiliary hidden states, incoming
draft logits, proposal candidates/scores and sampled/rejected counts. Requests
are bounded probes with synchronization and full-vocabulary dumps; neither
their latency nor their forced length cap is a performance/text-health result.
The first natural-mode startup failed because its proposal wrapper did not
preserve the `input_batch` keyword used by warmup; the signature is corrected.
The next failed before model loading because another task occupied GPU4--7.
Neither failed run contains a usable natural-sampling comparison. Per-launch
GPU availability is now rechecked in addition to the existing advisory locks.

The first successful natural control on GPU0--3 preserves the previous
uninstrumented fixed-norm output prefixes: all 32 MBPP28 and 144 release1k
tokens match. This bounds the observed diagnostic perturbation; it is not a
complete-output, acceptance noninferiority or performance result. The natural
comparator checks complete four-rank target/proposal coverage before finding
the first observed difference. Eleven CPU tests pass (one CUDA graph test
skipped), including missing-proposal rejection, proposal-before-target ordering
and exclusion of unwritten sampled-output padding.

The completed natural pair has 232 target records per arm (nine MBPP28 and
49 release1k forwards, each on four ranks). Its first observed difference is
already at the prefill target boundary, before the first packed q8 verifier:
layer 0/1 GDN observations remain equal, while all five auxiliary hidden-state
tensors and final logits differ. The first sampled row has zero post-top-k/p
TV in both cases, despite nonzero full-vocabulary TV. Later token/acceptance
changes therefore cannot be dismissed based on that first sampled row.
The investigation has moved to a bounded eight-token probe with layer 2/3
observations, including the first full-attention layer. Q/K reduction autotune
choices are being checked; no Q/K normalization cause is established yet.

At 2026-09-08 01:24:11 UTC, main merged PR #560 as
`e5d63c51f0fcc1ddf75d229e3df06bf52df206f5`. It routes DFlash2 E4M3 q8 to FP32
attention intermediates and changes the scalar/q1 precision path. The frozen
campaign results above predate that change. Preserve the old overlay until
the current numerical attribution is closed, then merge main, use matching
precision-revision-4 Flash-V100 binaries and establish a new baseline before
final performance acceptance. An independent build from the exact merge tree
has completed under `flash-v4-source` / `flash-v4-build`; it is not yet the active
runtime. The separate FP8-target model gate for #560 does not validate QUASAR.
The revision-4 Flash-V100 library SHA256 is
`a751fed902279b0de23537c4aad2dc4fee360146d7fce7ef0c4f255a77f48b02`;
the matching paged-KV utility SHA256 is
`571fe2a96b70d76737375eaed9fb8ad1cac3bc7eefadf139ea3d2437e0cfdb7d`.
