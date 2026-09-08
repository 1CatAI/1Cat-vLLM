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
The user subsequently reserved devices 0--3 for other work: all subsequent
campaign GPU execution is restricted to devices 4--7. The temporary 0--3 lease
and telemetry are released; those diagnostics remain historical evidence only.

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

The extended eight-token pair has 24 target records per arm. Its logical
target, auxiliary, proposal and acceptance tensors match exactly after aligning
physical request slots. The early proposal observer incorrectly sliced the
per-request seed/temperature arrays as if they were packed per draft row;
it now gathers by request slot. The comparator also aligns those older retained
captures and reports physical slot mappings separately. Twelve CPU tests pass
(one CUDA graph test skipped). The passing short pair does not clear the
earlier prefill drift or the complete acceptance gate.

From the user's subsequent scope clarification, this branch concentrates on
DFlash2 complete-round cost. Independent quality-root-cause investigation is
left for the other agents, with retained artifacts in `results/quality-handoff.md`.
Numerical parity and acceptance remain mandatory candidate promotion gates.

At 2026-09-08 01:24:11 UTC, main merged PR #560 as
`e5d63c51f0fcc1ddf75d229e3df06bf52df206f5`. It routes DFlash2 E4M3 q8 to FP32
attention intermediates and changes the scalar/q1 precision path. The frozen
campaign results above predate that change. The cost branch integrates that
main as `631780fb4229e3cc4f384571135f6fd86996ce3f`, with the old overlay and
libraries retained for the separate numerical investigation. An independent
build from the exact merge tree is active under `flash-v4-source` /
`flash-v4-build`. Its import reports precision revision 4; 138 attention-policy
tests pass on CPU, with one GPU test skipped. Establish a new unprofiled
baseline before interpreting complete-round gains. The separate FP8-target
model gate for #560 does not validate QUASAR.
The revision-4 Flash-V100 library SHA256 is
`a751fed902279b0de23537c4aad2dc4fee360146d7fce7ef0c4f255a77f48b02`;
the matching paged-KV utility SHA256 is
`571fe2a96b70d76737375eaed9fb8ad1cac3bc7eefadf139ea3d2437e0cfdb7d`.

### QPN2 cost measurement

`benchmarks/kernels/benchmark_sm70_qpn2_working_set.py` exports the prepared
runtime codes, scales and actual dispatch parameters from four consecutive
TP4 layers. Its benchmark replays all sixteen projections in model order,
covering the five production shapes. Activations are explicitly frozen
synthetic FP16 inputs; weights must come from the real loaded model. Both
arms are checked for finite, bitwise-equal outputs after the first graph
replay and after all alternating timing trials. Source-library and snapshot
hashes accompany the result. This is a working-set measurement, never a
complete-round result or a replacement for the candidate's full model gates.

The task-local trace parser now discovers the captured steady rounds and
kernel counts. A regression against the retained September 6 trace exactly
reproduces the recorded phase and complete-round timings.

### FP32-attention baseline and confirmed layout saving, September 8

The user authorized stopping the service on GPU4--7. That service is stopped;
the campaign holds the rear-four lease and does not allocate GPU0--3.
The following independent startups use main `e5d63c51f0`, the pinned revision-4
attention DSOs above, fixed Gemma reduction, and no profiler or tensor dumps.
Each fixture has one warmup and five measured requests. These are initial
screens, not the required three paired startups or final promotion evidence.

| Fixture | Packed off | Packed on | Packed + combined split |
| --- | ---: | ---: | ---: |
| release1k complete round | 19.336 ms | 18.506 ms | 17.366 ms |
| MBPP28 complete round | 18.923 ms | 18.024 ms | 16.953 ms |

Every output hash matches within and across these three arms. release1k emits
272 tokens in 91 rounds (1.989 accepted drafts/round; 2.989 emitted/round).
MBPP28 emits 634 tokens in 130 rounds (3.877 accepted drafts/round; 4.877
emitted/round). Both stop naturally. The combined-split startup has median
TTFT 354.2/115.7 ms and pure decode throughput 171.5/287.2 tokens/s.
See `results/v4-layout-initial-ab.json` and
`results/v4-combined-initial-ab.json`. Loaded DSO inventories for the packed
and combined startups were hashed after measured requests. The first baseline
retains the pinned library manifest but predates that extra process-map capture.

The opt-in `VLLM_SM70_DFLASH2_FUSED_GDN_COMBINED_SPLIT=1` reuses the existing
bitwise split kernel for the all-NVFP4 QKVZBA allocation. The old split flag
only covered checkpoints with a separate b/a projection. In the real combined
layout, each of 48 GDN layers instead launched three index creations and three
separate tail copies. The new path copies z/b/a in one launch before convolution
mutates QKV. It is limited to SM70 DFlash2 TP4/hidden5120, FP16 q8 and
QKV/z/ba widths 2560/1536/12; other shapes retain their existing route. The
new flag defaults to zero and is not automatically enabled by DFlash2 setup.
Nine GPU tests pass, including the real 4120-column view with row stride 4128,
aliased input arguments, compiled and ordinary CUDA graph replay, changed input
values, preserved tails after QKV mutation, and untouched padding.

The fresh packed trace is `profile/v4-packed-tp4.nsys-rep` and its SQLite;
`results/v4-packed-trace.json` retains the analysis. Ten steady rounds on all
four ranks show QPN2 service 7.033 ms, target communication 1.652 ms, target
copies 1.339 ms and normalization 0.831 ms. Draft service is 3.774 ms, including
1.831 ms of dense GEMMs. The 96 tiny b/a index-select kernels each launch one
eight-thread block. These measurements motivate the combined split above.
Profiler critical-rank round time is 21.618 ms; it is diagnostic, not an
endpoint performance result. The profiler stop/export request is also excluded.

Four QPN2 candidates were screened on rank-0 runtime weights from four
consecutive layers (214,087,680 bytes), all five production shapes, seven
alternating A/B trials and 50 graph replays per trial. All remain bitwise exact;
none is faster, so none is integrated into serving:

| Candidate | Control/candidate median working-set ms | Decision |
| --- | ---: | --- |
| Static K specialization | 0.389 / 0.405 | Reject; every paired trial slower |
| Precombined FP16 scales | 0.379 / 0.417 | Reject; scale traffic grows |
| 16-column CTA remapping | 0.406 / 0.429 | Reject; every paired trial slower |
| One-group codes/scale prefetch | 0.390 / 0.408 | Reject; every paired trial slower |

These are synthetic activations with real runtime weights, not full-layer or
model-quality results. The precombined-scale working set is 237,875,200 bytes;
the benchmark now records candidate scale dtype and footprint. The unchanged
source-library control also matches the archived production QPN2 outputs.
Source and DSO SHA256s accompany each `results/qpn2-*-real.json` result.

The existing small-message push route also passed its native gate after a
task-local build of the current main communicator: all four ranks, 13 message
sizes, 32 changing-input cycles for each of random/zero/special patterns,
mixed graph order, interleaved sum2, delayed ranks and canaries. For 128 q8
collectives, 80 versus 40 blocks measures 0.884 versus 0.882 ms in the same
communicator lifetime. This gain is too small to justify a full model candidate;
it is not promoted. See `results/custom-ar-v4-mixed-size-gate.json` and
`results/q8-push-grid-race.json`. No normalization arithmetic was changed.

### QPN2 publication and communication arrival audit

The private publication candidate moves the established 16-byte packet writes
into the unchanged QPN2 arithmetic epilogue. Producer CTAs never poll or wait.
A separate consumer preserves rank-ordered FP32 addition, FP16 materialization,
sentinel escaping/cleanup and both epochs of the existing push pool. The normal
projection output remains materialized. This differs from producer-poll fusion;
the first implementation spilled its dynamically indexed peer-pointer array
and was slower. Passing the local pointer directly eliminates those spills.
The exact q8 row-projection kernels use 48 registers with no stack or spills;
the consumer uses 40 registers with no stack or spills.

`benchmarks/kernels/build_sm70_qpn2_publish_candidate.py` generates the private
candidate from production source anchors and records source/DSO hashes. It does
not replace a serving operator. `--build` now keeps default CUDA math;
`--use-fast-math` explicitly reproduces historical experiments and is an
arithmetic change for gated SiLU. All extra CUDA flags are recorded in the
manifest. The generated CUDA source SHA256 is
`7360d578080c96350970b9ceb42fa5e470627949c28219013d1daa0861acc42a`.
Use a communicator built from the same header and set
`VLLM_SM70_CUSTOM_AR_LIBRARY` to that sidecar; do not mix opaque communicator
objects between libraries. The measured sidecar SHA256 is
`e32f156f606c47dc5863a7785065f1fef9ff3668788d9c44e5f85c2264e8a7d1`.

`benchmarks/kernels/benchmark_sm70_qpn2_publish.py` exercises sixteen real
prepared projections on each of four ranks, with eight dependent all-reduces
and an extra ordinary push call. Five changing-input cycles, alternating graph
order, delayed ranks and output canaries pass byte-for-byte checks on projected
and reduced tensors. The working-set screen measures 0.491/0.465 ms for
control/candidate; it is not a complete model round. A focused memcheck and
racecheck pass with zero errors using Gloo process coordination. The first
NCCL-coordinated sanitizer run exited on CUDA API 209 during NCCL's kernel
capability probing; it was not counted as a pass. Racecheck is a shared-memory
check, not proof of all inter-GPU global-memory ordering.

The model-screen DSO SHA256 is
`3e5afdbdb176cf4ed7460e125f48f0bb1f36e85fae7107d33f48deb157919a37`.
The reusable builder produces identical CUDA source; its independently rebuilt
DSO `007298ccc086ac3181a29b9d745f32c76809a9d2b90158af474b2b94f3efc457`
also passes the four-rank five-cycle correctness gate. Original sanitizer
evidence is tied to the model-screen DSO, not relabeled as a rebuilt-DSO run.

The task-local model integration preserves one opaque row-projection boundary
in both arms, reuses the same compiler/autotuner cache, and enables publication
only during q8 CUDA capture. All 128 target row projections hit on every rank;
draft projections retain their existing path. One independent startup per arm,
one warmup and five measured requests per fixture, without instrumentation:

| Fixture | Matching publication control | Publication enabled | Saved |
| --- | ---: | ---: | ---: |
| release1k complete round | 17.232 ms | 17.073 ms | 0.159 ms |
| MBPP28 complete round | 16.785 ms | 16.660 ms | 0.126 ms |

All output hashes, natural EOS and both acceptance-length definitions match the
earlier arms above. The matching control includes the new row-op boundary and
rebuilt communicator: do not attribute its difference from the earlier
17.366/16.953 ms result to publication. See
`results/v4-publish-initial-ab.json`, per-startup source manifests and mapped
library inventories. The model integration remains task-local and disabled
by default; full state/distribution and repeated-startup gates remain open.

Ordinal-matched communication in the four-rank packed trace shows that its
first target push has mean kernel duration 0.388 ms, rank arrival skew 0.684 ms,
and last-arrival-to-last-finish time 0.010 ms. The first draft push similarly
measures 0.236/0.473/0.009 ms. Much of these particular kernel durations is rank
waiting. Graph-node profiling can itself inflate arrival skew; these numbers
do not establish unprofiled host overhead. The CPU sparse-target probe span
includes waiting for queued target GPU work and must not be added again as
independent CPU cost. See `results/v4-collective-arrival-audit.json`.

Further exact QPN2 screens (shared-partial bank swizzle, cache policy and a
33-percent shared-memory carveout hint) show either noise-level savings or
regressions and remain unpromoted. A vectorized peer-read/Gemma prototype is
exact but slower (eight joins: 0.100/0.264 ms); a local-push consumer design
requires its own evidence. The existing draft TurboMind FP16 GEMM screen saves
about 0.200 ms across twenty real-weight projections but is not bitwise equal.
Its independent FP64-reference errors do not worsen in that primitive screen;
model-distribution and acceptance gates are still required, so it is not enabled.

The subsequent forced-tape publication pair (`v4-publish-audit-control` /
`v4-publish-audit-speed`) has 140 records per arm across all four ranks and
two 128-token tapes. Requested layer 0/1 intermediates and conv/SSM state,
target boundary tensors, and native logits are bitwise equal.
Sampling TV is zero with no changed top-p support or top-1 rows. This gate
retains the complete prefill records; it is not natural acceptance evidence.
See `results/v4-publish-audit-comparison.json` and its separate manifest.

The lower-overhead whole-graph trace in `profile/v4-publish-graph/tp4.sqlite`
does not collect individual graph nodes. Ten steady rounds have diagnostic
critical-rank mean interval 18.465 ms, GPU union 17.214 ms and uncovered time
1.252 ms. Target graph mean duration is 12.308 ms. Its host launch skew is
0.685 ms, but GPU start skew is only 0.005 ms because launches are queued.
The main draft graph has host/GPU start skew 0.292/0.297 ms and mean duration
3.862 ms. Do not treat the earlier target-node arrival skew as an established
unprofiled saving. The request containing profiler stop/export is excluded
from endpoint performance claims. Analysis:
`results/v4-publish-graph-trace.json` and
`results/v4-publish-graph-arrival-audit.json`.

Additional independent screens remain unpromoted:

| Candidate | Control/candidate working-set ms | Result |
| --- | ---: | --- |
| Local published-packet consumer + Gemma | 0.508 / 0.515 | Exact; slower in every pair |
| Global QPN2 partials, four warps per CTA | 0.389 / 0.539 | Exact; added traffic/launches do not pay back |
| Fixed-q8 input/output bounds | 0.380 / 0.477 | Exact; compiled register use rises to 70--72 |
| Fixed-q8 bounds, unroll two | 0.393 / 0.408 | Exact; 64 registers, still slower |
| K-group-major weight codes/scales | 0.379 / 0.392 | Exact; same footprint, still slower |

The local consumer preserves the existing packet protocol and Gemma reduction
topology and returns the materialized reduced tensor. Its successful gate
covers changing inputs, graph order, delayed ranks and canaries. It follows
two retained harness failures: incorrect packed inline-assembly return
constraints and an unregistered warmup-only ordinary collective buffer.
Neither failed run is counted as correctness or speed evidence.

An independent-stream context experiment gives each arm an identical context
capture stream, private graph pool and cuBLAS workspace. Only replay placement
differs. Context scratch reads wait for target output; accepted-slot KV writes
stay on the main stream and wait for context completion. The unprofiled model
pair is slower: release1k 17.088/17.547 ms and MBPP28 16.659/17.130 ms. All
output hashes and acceptance lengths match. See
`results/v4-context-overlap-ab.json`; the experiment remains disabled and no
further quality promotion work is justified by this negative speed result.

Rear-four telemetry during decode reports 1530-MHz SM clocks, 877-MHz memory,
roughly 171--183 W draw under the unchanged 300-W limit, and no active clock
event reason in the checked samples. Clock headroom is not credited as a
remaining optimization.

### Direct attention output: speed candidate held at the numerical gate

`VLLM_SM70_DFLASH2_DIRECT_ATTENTION_OUTPUT` defaults to zero. It is armed only
for SM70, DFlash2, TP4, FP16 dense Qwen3.5 with hidden size 5120. The decoder
can consume the projection tensor already returned by attention. The new GDN
opaque entry returns that allocation while retaining the full-forward operation
order and explicit conv/SSM mutation arguments. It does not enable the existing
long-prefill collective/norm switch. Other models keep the existing path.

The initial artifact prototype failed during compilation because the existing
GDN full-forward boundary required an output buffer even though the decoder
could accept a direct return. The new return-valued opaque entry resolves that
interface issue. Its schema marks both caches mutated and its return unaliased;
the fake implementation passes shape/dtype checks at 1, 8, 135 and 4097 rows.
Scoped Python lint, format and bytecode checks pass.

One unprofiled artifact A/B startup per arm, five measured requests per fixture,
gives 17.091/16.824 ms on release1k and 16.561/16.402 ms on MBPP28. All output
hashes, natural EOS and acceptance lengths match. The source-integrated version
also reaches this range, but **this candidate is not numerically cleared**.
Its 140-record fixed-prefix comparison has identical captured layer 0/1
intermediates and conv/SSM state, while all target-boundary hidden records and
native logits differ. Maximum sampling TV is 0.0104069 with no changed top-p
support or top-1 rows. These observations do not yet distinguish later-layer
arithmetic/compiler effects from diagnostic perturbation. See
`results/v4-direct-source-audit-comparison.json` and
`results/direct-output-quality-hold.json`. A bounded eight-token probe captures
GDN layer 2 and the first full-attention layer 3. Until the difference is
localized and resolved, exclude this candidate from promoted combinations.

The QPN2 input-layout screen is separate: arranging the same FP16 q8 input as
`[K/16, 8, 16]` gives 0.387/0.357 ms across the real four-layer weight working
set, bitwise equal in every projection. That screen excludes packing time and
does not establish model speed. Gemma producers that write this layout directly
pass 45 changing-input graph cases across no-residual, FP16-residual and
FP32-residual contracts and three magnitude ranges. The subsequent model
experiment is based on the cleared publication combination, with the direct
attention-output switch disabled. See `results/qpn2-input-packed-real.json`
and `results/packed-gemma-gate.json`.

The first complete-model packed-input pair fails admission: release1k changes
from 272 to 210 tokens and 1.989 to 1.800 accepted drafts/round; MBPP28 changes
from 634 to 754 tokens and 3.877 to 3.303 accepted drafts/round. Its apparent
16.823/16.443 ms timing is not a promoted gain. A diagnostic shadow then uses
the actual input and weight of every selected operator: per rank, 383 fresh
comparisons cover 128 norms, 127 residuals and 128 column projections. Norms,
residuals and ungated projections all match. Gated projections in later layers
show a few differing bytes, often one FP16 ULP, which the first-four-layer
synthetic-input screen missed. Retained evidence is
`results/packed-input-operator-shadow-summary.json` and each rank's raw report.

The experimental packed-input DSO used `--use_fast_math`. Disassembly of its
gated kernel has no FFMA correction instructions; the archived production
gated kernel and the default-math rebuild each contain 22. The current CMake
QPN2 path obtains Torch's common CUDA flags without adding fast math. A
same-input GEMM/activation shadow and default-math model rerun are underway;
the input-layout candidate remains held until those gates pass. The builder's
former implicit fast-math default has therefore been removed and its math mode
made explicit. Historical SHA256s and measured results are not relabeled as
default-math results. Publication's serving path used only its nongated
producer and already passed its complete 140-record comparison.

The bounded direct-output probe has eight four-rank records: GDN layer 2,
full-attention layer 3, target hidden and native logits all match, with zero TV.
This does not clear the earlier 140-record drift; the direct-output switch
remains disabled. See `results/v4-direct-tiny-comparison.json`.

The complete-round target below 15 ms, full distribution/state comparison for
the final combination, repeated-startup acceptance gates, and long-context
validation remain open. No new serving default or merge is claimed.
