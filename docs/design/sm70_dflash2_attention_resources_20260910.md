# DFlash2 attention resource experiments, September 10

## Scope and acceptance

This follow-up is based on the long-verification PR's
`1994baff8f188a2c915e883395ae96e682db0577`, with integration target `onecat/main`
(observed `c2ba9e2929cffe44e34cef8bb22dde99e2dc7b16` at branch creation).
It adds private operator experiments; it changes no default serving route.
The complete-round target remains **22 ms at 261888 input tokens and a
256-token output budget**, within 262144 service capacity. Decode time divided
by speculative rounds includes terminal q1 and partial-verifier steps.
An attention operator time is not this complete-round metric.

The target is QUASAR-QAT Qwen3.8-27B NVFP4 through TurboMind W4A16/QPN2,
TP4 on V100-SXM2-32GB, FP16 activations, E4M3 KV, FP32 state/logits,
CUDA 12.8, Torch 2.10.0+cu128 and Python 3.12.13. Its 16 full-attention
layers retain the complete context. The DFlash2 draft's 2048-token window
does not restrict target attention. Keep temperature 1, top-p 0.95, top-k 20,
the existing seeds, native FA2 prefill and original GDN prefill.

The existing visible-q8 parent preserves 80 logical splits, N32 updates,
K16 compensated QK products and probability-residual PV products. The
following experiments preserve that arithmetic and FP32 workspace contract.
All timings below use sixteen distinct layer KV allocations, page size 3296,
query shape `(8,6,256)`, alternating CUDA-graph A/B order and five event samples.

## Exact E4M3 lookup

`build_sm70_grouped_attention_candidate.py --e4m3-shared-lut` initializes
a 256-entry, 512-byte shared table with the existing E4M3 bit decoder.
The initial CTA barrier publishes the table before aligned 16-byte loads
or the existing stride fallback consume it. The option requires an isolated
six-head fixed-q8 build, vector loads and 80 splits. It is off by default.
Without the option, the generated parent source retains SHA256
`3b0c9688ce17e1870408ef81fb5cd9b63a677b7cfd7d4777b8df77dd0fc24132`.

| Sixteen-layer operator | Paired parent, ms | Lookup, ms |
| --- | ---: | ---: |
| 131072 tokens | 7.727 | 7.599 |
| 261888 tokens | 14.874 | 14.624 |

All 65 output/full-FP32-workspace/canary checks pass, including page crossings,
stride padding, zero/rejected rows, restored lengths and 262144 visibility.
The candidate's memcheck, racecheck and synccheck each pass 25 such checks
with zero errors or race hazards. Sanitizer runs contain no performance work.
The paired 3296-page q8 specialization uses 126 registers and 512 static
shared bytes, without reported stack or spills; dynamic shared memory is
73728 bytes. These are compiler resources, not achieved occupancy.

Source SHA256:
`e78e922b17ff44262c49e6664871d0692e0ecd048b676061516d0419d3819f92`.
DSO SHA256:
`3b5a53e413ce064cccac4c735f090fbd7f9e4beffb097120d15e1861f36a2ed2`.
Raw reports: `q8-kv-lut-screen.json`, `q8-kv-lut-{memcheck,racecheck,synccheck}.json`.
The first same-startup service A/B keeps compact scalar q1 in both arms.
Both endpoint lengths run one cold request followed by five repeats per arm.
All twelve request pairs retain identical tokens, finish reasons and acceptance;
each rank records 384 actual compact-scalar q1 calls per arm.

| Unprofiled endpoint | Parent | Lookup |
| --- | ---: | ---: |
| 1K complete round, ms | 15.870 | 15.691 |
| 261888 complete round, ms | 35.034 | 34.646 |
| 261888 pure decode, tokens/s | 145.575 | 147.202 |
| 261888 accepted drafts/round | 4.020 | 4.020 |
| 261888 emitted tokens/round | 5.120 | 5.120 |

The five-repeat 261888 prefill medians are 1.196/1.199 seconds and TTFT
1.429/1.430 seconds. These repeats retain the benchmark's prefix-cache policy;
they are not cold-prefill throughput. Cold requests remain in the raw reports.
The source is `1994baff`, whose runtime Python and Flash-V100 source match the
frozen `239d71c` service revision. Report: `q8-kv-lut-model-1-paired.json`.
This is one paired startup. The **22-ms target remains unmet**; repeated-startup
and new-candidate natural-output admission are incomplete.

## Rejected parallel QK products

`build_sm70_grouped_attention_qk_parallel.py` binds the hashed visible-q8
parent. Twelve warps compute independent K16 products into four shared slots;
six consume them with the original compensated addition order, and four
load V. The extra product storage raises dynamic shared memory to 98304 bytes.
The 3296-page q8 specialization uses 128 registers without reported spills.

All 65 byte/canary checks pass, but the added storage and barriers fail the
performance screen. At 131072 tokens the paired parent/candidate costs are
7.705/14.075 ms; at 261888 they are 14.877/27.464 ms. Reject before sanitizer
or model trials. More participating warps alone is not evidence of better
resource utilization. Preserve this negative result to avoid repeating it.

Source SHA256:
`71fa3ec257b358b6ef40956df2d4fe56c2777c6f0f9edd6e12a4b0a5a3d62134`.
DSO SHA256:
`50833f8b2fb025274635d719e0fe264cef41784dbe4acb0bdd758e4f522b5ca9`.
Raw report: `qk-parallel-screen.json`.

The existing two-head-group screen had not combined vector loads and QK
unroll2. Screening that combination with 3296-page specialization retains
all 65 byte/canary checks, but costs 11.635/22.766 ms at 128K/261888 against
its paired parent's 7.698/14.871 ms. Reject it before sanitizer or model runs.
Raw report: `heads2-vector-qk2-screen.json`. Reducing the per-CTA footprint
does not by itself overcome duplicated KV reads and padded three-head rows.

## Compact probability layout

The current work focuses on the approximately 15-ms **target q8 attention**
at 256K. Scalar q1 graphs and unrelated round costs are deferred. The private
`build_sm70_grouped_attention_swizzle.py` builder binds either the hashed
visible-q8 parent or its exact-LUT child and changes shared-memory layouts.
It copies and hashes all native inputs; no default service imports it.

The probability/residual layout packs each M16/K16 panel into two 128-half
planes. An SM70-specific row permutation feeds matrix-A fragments through
two aligned vector shared loads. Q remains in its original padded layout in
the selected candidate. Moving the P/residual stores before the warp sum
reduction retains every arithmetic operation and the publication barrier.
This targets data access; it does not truncate the context, change 80 logical
splits, remove residual compensation, or change online update order.

| Paired layout screen, 16 distinct layers | 128K, ms | 261888, ms |
| --- | ---: | ---: |
| Exact-LUT parent | 7.571 | 14.620 |
| Q layout only | 7.552 | 14.589 |
| Q and P layout | 6.836 | 13.193 |

Both candidates pass 65 byte/workspace/canary checks each. The next screen
holds the Q/P candidate as the control and separates P from Q:

| Layout screen | 1K, ms | 32K, ms | 128K, ms | 261888, ms |
| --- | ---: | ---: | ---: | ---: |
| Q and P control | 0.505 | 1.989 | 6.836 | 13.207 |
| P only | 0.502 | 1.977 | 6.779 | 13.084 |
| **P only, early stores** | **0.501** | **1.968** | **6.739** | **13.010** |

The two additional candidates each pass 65 checks. The selected early-store
DSO also passes 60 q2–q7 tail checks against the original visible-q8 parent.
Its memcheck, racecheck and synccheck each pass 25 checks, including the full
262144 boundary, with zero errors or race hazards. The 3296-page full-q8
specialization uses 118 registers and 512 static shared bytes, with no stack
or spills reported. This is a compiler observation, not achieved occupancy.

Selected source SHA256:
`eb7a85511f581fcd22cf13619c85ed2f42a8cbc8b216bb3e632bf448f6b820e1`.
Selected DSO SHA256:
`9db33737adb880cd4198266ac4f29785ce3e709936aa47dcc79011a9bce4b811`.
Regenerating with the final builder retains every source-file hash.
Raw reports are `qp-swizzle-screen.json`, `p-swizzle-screen.json`,
`p-swizzle-early-tail-checks.json` and
`p-swizzle-early-{memcheck,racecheck,synccheck}.{json,txt}`. The local handoff
records their absolute locations in the owned artifact directory.

The `--pv-m8n32` screen widens PV tiles and keeps three accumulator fragments
per warp. A native fragment probe validates all 512 matrix-A halves and 256
accumulator elements before the operator comparison; all 65 full-operator
checks then pass. Performance nevertheless regresses from 6.736/13.006 ms
to 7.130/13.805 ms at 128K/261888. Reject it before sanitizer or model trials.
Source SHA256 `549ed030aa129f496c84db6df9c74bfa84f186ea78cb7ecb624d84f6dd2bcdbd`,
DSO SHA256 `2be7d9fcaa728b9fc287457af35957bbda6c7dfa83534707b8cbd1283f2e37b5`.
Report: `p-swizzle-pv8-screen.json`.

A final direct comparison includes the original visible-q8 DSO, the selected
P layout, compact K/P, and a same-warp two-product QK schedule. All three
candidates pass 65 byte checks each. The K experiment preserves V's padded
layout; its native matrix-B word map is checked before the operator run.
An initial K build was discarded by source inspection because a zero-fill
rewrite also affected V; only the corrected `kp-swizzle-early-r2` is measured.
The QK schedule forms two K16 products before summing them in the unchanged
FP32 compensation order, without the earlier experiment's CTA barriers.

| Final paired operator screen | 1K, ms | 128K, ms | 261888, ms |
| --- | ---: | ---: | ---: |
| Original visible-q8 | 0.544 | 7.701 | 14.893 |
| **Selected P layout + early stores + lookup** | **0.505** | **6.738** | **12.983** |
| K and P layout + early stores + lookup | 0.526 | 7.330 | 14.149 |
| P layout + same-warp QK pipeline + lookup | 0.502 | 6.737 | 12.984 |

The selected 256K operator is **12.83% faster** in this direct comparison.
Reject K layout for its regression, and reject the extra QK schedule because
its long-context benefit is below measurement resolution. Neither requires
further sanitizer or model trials. Report: `p-k-qk-final-screen.json`.
K source/DSO SHA256:
`08f56749cd466a27d4a52f4fa796db70c4058f14a8cbb4f12162ca4e86a795ee` /
`b098987572665dd729818d52dbf9f6652dfdc105eaf1625eb01670fc36e9fb39`.
QK source/DSO SHA256:
`a93f18addd174f1cfb4520f2ebe6dde4a6cc43ff556ecb025a2ccbd70c75bde5` /
`456029c9274030af1f2af11c666d3713cc9a340b6a8a352ccfca0e53fe20cc64`.

The new full-q8 clock probe retains byte-identical outputs and all FP32
workspace elements for sixteen distinct layers at 128K and 261888. At 261888,
aggregated tile-cycle fractions are K load 15.43%, QK with overlapped V load
38.36%, online softmax 20.53%, and ordered PV 25.69%. These include probe
and synchronization costs and are not wall-time fractions or hardware
utilization. The next optimization should account for QK as the largest
measured phase rather than assuming PV still dominates. The old generic-only
probe regenerates to its original source hash after the builder refactor.
Report: `p-swizzle-phase.json`; source/DSO SHA256:
`de8a6520bae50e33ab7759501da864a69c7b57b13182abba2ef28f8d76a52f1c` /
`444200120b6a3dd4f9bbed605f960eff4456a65de693019409bc347e27607ed4`.

The selected layout completes one independent, same-startup service A/B
(24 requests, 12 pairs). Every pair retains identical tokens, finish reason,
sampling and draft acceptance. Both arms keep the existing compact scalar q1
route; graph replay counters establish actual q8 control/candidate route hits
on all four ranks. The five-repeat medians exclude the retained cold request.

| Unprofiled endpoint | Visible-q8 control | P layout + lookup |
| --- | ---: | ---: |
| 1K complete round, ms | 15.829 | 15.774 |
| 261888 complete round, ms | 36.134 | 33.263 |
| 261888 pure decode, tokens/s | 128.310 | 139.384 |
| 261888 accepted drafts/round | 3.563636 | 3.563636 |
| 261888 emitted tokens/round | 4.654545 | 4.654545 |

Report: `p-swizzle-early-model-1-paired.json`. This startup's generated
trajectory differs from the earlier lookup-only startup; do not use those
two startups for an unpaired speed claim. The measured service gain includes
all consequences of replacing q8 within the same runtime. It does not assign
the entire 2.871-ms round reduction to attention-kernel service time.
The **22-ms complete-round target is still unmet**. Repeated-startup and
new-candidate natural-output corpus admission remain incomplete.

The actual service CUDA graph-node trace now corroborates the operator
result: target grouped attention consumes **13.077 ms per rank/round** across
eight interior q8 rounds on four ranks (steps 9–16 of ten captured steps).
Every rank/round contains sixteen full-q8 kernels and sixteen combine kernels.
All 512 observed full-q8 launches use the intended 3296-page specialization,
80 CTAs, 512 threads, 118 registers, 72192 dynamic shared bytes, 512 static
shared bytes and zero reported local bytes per thread. Control/candidate
DSO hashes and graph route counters remain in the service capture report.
The warm and captured requests retain identical tokens, finish reason and
acceptance. These instrumented kernel timings corroborate route behavior;
they do not replace the unprofiled complete-round A/B above.

Reports: `p-swizzle-early-service-trace.json` and
`p-swizzle-early-service-attribution.json`. The SQLite trace has SHA256
`95601fc2228f963a2864c076e89cb5a4632fbc4186df8f80bafe90e97b4513e3`.
Collection used Nsight Systems 2022.4.2.50. The system package lacked its
importer, so the retained raw trace was converted offline with the matching
official NVIDIA 2022.4.2.50 bundle; no service rerun was needed. The local
worklog records tool/archive hashes and the conversion failure/recovery.
The profiling driver now binds the complete tool path explicitly.

## Reproduction and promotion gates

Build the lookup using the measured parent's options:

```bash
.venv/bin/python benchmarks/kernels/build_sm70_grouped_attention_candidate.py \
  --output-dir "$CANDIDATE" --head-groups 1 --qk-unroll 2 --vector-load \
  --page-specialize --prefetch-v --reuse-pv-values --specialize-full-q8 \
  --all-visible-tiles --e4m3-shared-lut --grouped-only --build
.venv/bin/python benchmarks/kernels/benchmark_sm70_grouped_attention_long.py \
  --baseline "$PARENT/manifest.json" --candidate "$CANDIDATE/manifest.json" \
  --output "$REPORT" --performance-page 3296 \
  --performance-contexts 131072 261888 --extended-boundary 262144
```

Build the probability layout from that exact-LUT manifest:

```bash
.venv/bin/python -m benchmarks.kernels.build_sm70_grouped_attention_swizzle \
  --base-manifest "$LOOKUP/manifest.json" --output-dir "$CANDIDATE" \
  --probabilities --early-store --build
```

The optional Q, K and wider-PV switches are independent screens, not defaults.
The phase-clock builder supports both the generic partial kernel and the
full-q8 specialization. Its extra clock writes and synchronization affect
timing; phase-cycle fractions must not be called wall-time or occupancy data.

Set the CUDA 12.8 toolkit, `TORCH_CUDA_ARCH_LIST=7.0`, owned compiler caches
and the GPU lease explicitly. For each sanitizer use the complete CUDA 12.8
bundle and replace timing options with `--sanitizer --correctness-only
--extended-boundary 262144`. Check source/DSO hashes in every report.

The owned worklog retains absolute artifact paths, environment, commands,
process ownership and service results. Native correctness and compiler-resource
counts do not establish full-model quality, repeatability, or the 22-ms target.
Promotion requires actual route hits, matching output tokens and acceptance,
natural-output gates, repeated unprofiled startups and the context sweep.

## Seven-millisecond attention target

The user tightened the acceptance target to **at most 7 ms for all sixteen
full-context target attention layers at 256K**. The complete-round target is
still 22 ms. Neither target has been met. The service-checked attention result
remains 13.077 ms; later producer timings below exclude softmax, PV and merge.
All screens retain the original K16 compensated FP32 arithmetic, N32 online
updates and 80 logical splits. No changed numerical contract is admitted.

The optional `--xor-planes` probability layout passes all 65 complete
output/workspace/canary checks against the selected P-layout DSO. Its paired
sixteen-layer costs are 6.760 to 6.662 ms at 128K and 13.004 to 12.827 ms at
261888. Applying XOR to K as well also passes 65 checks but costs 6.861 and
13.219 ms. Retain P-only XOR as a small native-only candidate; reject combined
K/P XOR. Neither has new sanitizer or service admission.
Report: `p-kp-xor-screen.json`.

### Independent QK producer

`build_sm70_grouped_attention_direct_qk.py` copies the hashed service-checked
P-layout source and exposes a separate `qk_stage` entrypoint. Its reference
arm is the original independent shared-panel QK producer. The ordinary `run`
entrypoint remains the parent full-attention kernel; timing it would not
measure these producer changes. Full FP32 scores use the original N32 tile
layout. A separate guard tile at each end and unused capacity retain sentinels.

Every row below passes **24 complete score-buffer comparisons**, including
zero/restored lengths, page boundaries, page sizes 3296/1648/848, 8-byte-only
stride padding, and lengths through 262144. Timing uses sixteen distinct layer
KV allocations, five warmup graph replays and five alternating trials of eight
replays. These are **QK-only** costs, not complete attention or service latency.

| Producer candidate | Paired reference at 261888, ms | Candidate, ms |
| --- | ---: | ---: |
| Direct operands, one warp per tile | 6.520 | 11.320 |
| Direct operands, context-major grid | 6.506 | 10.764 |
| Direct operands, six warps per CTA | 6.506 | 10.446 |
| Six warps, exactly decoded FP16 K mirror | 6.505 | 10.420 |
| Shared N64 / D128 panels | 6.522 | 8.737 |
| Shared N128 / D128 panels | 6.506 | 7.089 |
| N64 with coalesced output | 6.505 | 8.632 |
| N128 with coalesced output | 6.503 | 6.996 |
| N64 with K-fragment reuse across three Q tiles | 6.506 | 5.870 |
| N128 with K-fragment reuse across three Q tiles | 6.520 | 5.593 |
| N128 with reuse and separate independent products | 6.521 | **5.545** |

The FP16 mirror is a diagnostic that excludes conversion cost and consumes
additional memory. It does not justify changing the serving KV format.
Coalesced output reuses dead shared Q/K storage. Reusing each K fragment across
three M16 tiles gives the first substantial gain; computing their independent
products before compensated additions adds only a small further improvement.
Each output still accumulates its K16 products in the original order.

The last candidate's 128K result is 3.539 to 3.040 ms. Its source/DSO SHA256:
`70e79fc0dec2ff65bbf531dd976890c446532a60c9d5c1fc53d25bef39f8e114` /
`2f36e317be77e930a02c6cd674ef4d313d438807e166c7ff5af4477b72d08475`.
Reports are named `<candidate>-scores.json`; the local handoff maps every
candidate to its flags, full source manifest and library hash. The measured
runner was subsequently packaged as
`benchmark_sm70_grouped_attention_qk_stage.py` with explicit input/output paths.
The packaged runner also passes all 24 comparisons under CUDA 12.8 memcheck,
with zero errors (`panel-qk-n128-products-memcheck.json`). Racecheck,
synccheck, full-attention and model admission are not complete. The selected
3296-page paired producer uses 118 registers and 48384 static shared bytes,
with no reported stack or spills; its 8-byte-only fallback uses 122 registers
and also reports no stack or spills.

Ten later QK sources regenerate byte for byte from the checked-in builder.
The first `direct-qk` measurement used automatic shared-memory carveout; its
frozen source and manifest are retained. Later direct arms explicitly request
L1 preference, while their reference arms request shared-memory preference.
The historical first source also predates line-wrapping changes. Do not use
the current default builder to claim reproduction of that first source hash.

### Rejected unpadded head groups

`build_sm70_grouped_attention_compact_cta.py` specializes q8 into three groups
of two heads, with sixteen real rows and 256 threads per CTA. It retains the
P layout and the original logical partitions. A warp shares P across two D
output tiles. The smaller shared allocation permits two CTAs by shared-memory
capacity, but this is not an achieved-occupancy measurement.

All 65 output/workspace/canary checks pass. The paired complete-attention costs
are **6.735 to 12.034 ms at 128K**, and **13.002 to 23.632 ms at 261888**.
Reject before sanitizer or service trials. This smaller unpadded layout still
does not overcome its extra KV traffic and changed execution schedule.
Report: `compact-cta-screen.json`. Source/DSO SHA256:
`d7ee5ea2ef1c4b3c899f326e23aa8895b830ebab72977f660755990c783ec802` /
`fb87dac36992c9cf67c773950d50d73a357d3d5a8088ba7ef5437c8c158635cb`.

### Reproduction and remaining work

Under the same toolkit, cache and GPU lease settings documented above:

```bash
.venv/bin/python -m benchmarks.kernels.build_sm70_grouped_attention_direct_qk \
  --base-manifest "$P_LAYOUT/manifest.json" --output-dir "$CANDIDATE" \
  --context-panel 128 --coalesced-output --reuse-k --separate-products --build
.venv/bin/python -m benchmarks.kernels.benchmark_sm70_grouped_attention_qk_stage \
  --candidate "$CANDIDATE/manifest.json" --output "$REPORT"
```

The next useful milestone is a complete attention implementation benefiting
from independent QK production while accounting for score publication,
ordered softmax/PV, temporary storage and merge cost. A 5.545-ms QK result
alone leaves less than 1.5 ms of the requested budget for all remaining work;
it does not establish feasibility of a 7-ms complete kernel.
Nsight Compute counters remain unavailable under the current driver policy.
Do not infer long-scoreboard stalls, bank conflicts or achieved occupancy from
compiler limits, or introduce another software pipeline on that assumption.
Keep the numerical acceptance clarification pending and preserve the current
arithmetic order. Service defaults and the selected service-checked DSO remain
unchanged; the 7-ms attention and 22-ms round targets remain open.
