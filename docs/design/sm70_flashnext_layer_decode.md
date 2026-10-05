# Flash-Next whole-layer decode fusion screen

This campaign targets TP4 SM70, single-token decode and the M5 MTP4 verifier.
The acceptance targets are at most 600 kernels per step, no single-CTA kernels,
and ordinary pure decode at most 7.5 ms/token before the final 6 ms target.
These targets have not been achieved by this screen.

## Establishing a comparable endpoint

The integration base is `615710ae5106beeeab87950599fff28977e92962`.
The historical approximately 97.7 tok/s source reproduction at `b2042fb24b78`
used 8192 input tokens, 513 forced output tokens, FP16 KV, native FP32 recurrent
state and pinned-UVA PLE decode. The recent endpoint uses disk mmap PLE.
PR #831 records a source-complete mapped-result disk baseline of 10.98609 ms.
PR #885 requalification at `fd65333826e9` records 11.95157 ms for its reference
reader and 11.87902 ms for the default native reader. Earlier qualification of
that reader recorded 11.01475/10.86912 ms. The later pair is retained separately;
the earlier pair does not establish current-main speed.

The normal `fd65333826e9` package and this integration base have identical CUDA
sources and NVFP4 model implementation. Their only source differences are four
GGUF Python files. Its native artifact is therefore a source-matched NVFP4
control. A fresh normal native build from this owned tree is complete.
Historical pinned-UVA timing cannot by itself identify a regression commit in
the disk route; same-contract comparisons must locate the later disk slowdown.

The retained ordinary PR #831 package was rerun on the same host immediately
after current main, with the contract below. Its six decode samples were
11.979898, 11.920696, 11.863334, 11.889318, 11.874748 and 11.818351 ms/token,
median 11.882033. Current main measured 11.851278. All six pairs of 513 output
token IDs match exactly. The recent source regression is not reproduced by
this comparison; do not assign a regressing commit from unmatched old timings.
The PR #872 hardware record reports active SM clocks of 1507–1530 MHz, whereas
the current host reports 1290 MHz during load with the same GPU UUIDs and
300 W power limits. This is a measured condition difference, not a quantified
clock attribution or a completed regression fix.

The maintained regression entry records source, native hashes, resolved worker
routes, full timing token IDs and natural health outputs. It fixes FP16 dense
and KV, FP32 recurrent state, TP4, 262144 startup capacity, 94% memory budget,
8192 input / 513 output, one request, CUDA graphs, no prefix cache and no MTP.
Pure decode is separate from first-token and prefill time. EOS suppression is
confined to the fixed-length timing requests.

```bash
.venv/bin/python -m benchmarks.benchmark_sm70_flashnext_regression \
  --model /path/to/model --source-sha SOURCE_SHA --output baseline.json
```

## Research implementations

The shared-M1 candidate is included in the ordinary package build and enabled
by its dtype, shape and platform contract, with no new environment switch.
Its model distribution gate is pending. The other screen DSOs are research
artifacts; their results are not source-complete production performance
evidence. Admission requires the ordinary package and a clean-artifact model
gate.

| Segment | Screen | Synchronization and reduction |
| --- | --- | --- |
| HC | combine/norm/down/push plus up/mix, two kernels | Per-row tagged producer packets; no grid barrier; fixed FP32 sums |
| Router and experts | One router CTA inside W13, intermediate-tile W13/W2, one kernel | Cooperative resident launch without a grid barrier; independent producer flags and ordered sums |
| Shared expert | gate/up, SiLU, W2 and shared gate, one kernel | Ten resident intermediate-tile CTAs, release/acquire flags, ordered partial sums |
| GDN | Full value-head recurrence and gated RMSNorm, one kernel | One CTA owns every value column and all sequential M5 states |
| QSA | Selection, index expansion, sparse attention, merge and output gate, one kernel | Resident selector/core/merge CTAs and release/acquire flags |

HC covers real checkpoint weight banks, both M1 and M5, and changing width
allocations. Its M1 control uses the current TP4 operator. Its M5 control uses
the existing replicated FP32 MMA verifier implementation. An additional HC
screen incorporates the preceding core all-reduce into down. Shared expert
uses real checkpoint weights and includes
the small gate; its following all-reduce is excluded. GDN includes the complete
post-convolution core and output norm; projections, convolution and production
state-slot metadata are excluded. M1 compares FlashQLA plus norm; M5 compares
the actual mixed-QKV verifier core plus norm. Independent FP32 arithmetic checks
all M5 recurrent states. QSA excludes score construction, projection, rotary
and KV writes; it admits only the screened 8K geometry. Router/expert compute
excludes shared experts and the following TP reduction. These exclusions are
recorded in each result.

Prebuild on CPU while waiting for a complete GPU lease:

```bash
CUDA_VISIBLE_DEVICES= TORCH_CUDA_ARCH_LIST=7.0 \
  .venv/bin/python -m benchmarks.kernels.benchmark_sm70_hc_norm_push_screen \
  --build-only --output hc-build.json
```

Each screen records graph measurements and retained CUDA profiler evidence.
New captures retain the raw graph and count kernel nodes and grid dimensions
directly through the CUDA driver, including child graphs. Profiler completeness
is reported separately: one older M5 router trace dropped events, so its 23/3
event counts are not an authoritative graph count. Missing geometry is not
reported as zero single-CTA kernels. Keep kernels with insufficient whole
segment benefit out of model execution. A microbenchmark delta cannot be
subtracted from endpoint TPOT or credited as an end-to-end result.

## Quality and promotion

For this campaign, record teacher-forcing KL mean/p99/maximum over matching
prefixes and the full valid vocabulary, top-1 agreement and natural termination.
Maximum absolute logit difference is diagnostic and does not veto admission.
FP32 association differences do not require bitwise investigation. Moderate
precision changes require the same distribution and output-health gate.
This campaign contract supersedes the historical raw-logit maximum veto for
these changes; old evidence and thresholds remain historical records.

Compilation and Python syntax/lint checks have passed. Matched endpoint and
initial segment measurements are complete; additional segment screens and
candidate model distribution gates remain pending. No
endpoint speed improvement, kernel-count acceptance or production admission is
claimed yet.

## Initial measurements

The fresh integration-base endpoint records six samples at 11.889218,
11.884385, 11.892540, 11.687132, 11.818170 and 11.590987 ms/token: median
11.851278 ms/token. The arithmetic and exact-copy health prompts stop naturally
with the requested answers. All four worker reports confirm FULL M1 capture.
This establishes current-source reproduction, not a completed regression fix
or distribution admission for a candidate.

The complete-head GDN screen is rejected:

| Width | Native core plus norm, 36 calls (ms) | Fused complete heads (ms) | Graph kernels, native/fused |
| --- | ---: | ---: | ---: |
| M1 | 0.229089 | 0.302019 | 72 / 36 |
| M5 | 0.649751 | 1.098775 | 72 / 36 |

Its independent FP32 state oracle has maximum error 7.45e-9. M1 output error
is zero; M5 maximum output error is 3.05e-5, relative L2 2.23e-6. These are
segment arithmetic checks, not model distribution gates. Twelve complete-head
CTAs remove launch boundaries but lose the native state's tile parallelism.
Do not tune this variant. A distinct screen preserves 192 state-tile producers
and twelve fixed norm consumers in one resident kernel. The original negative
measurement used profiler event counts; no measured grid geometry was exported.

The initial shared-expert invocation failed before candidate execution because
its control called a removed Python wrapper. Its corrected control calls the
normal registered SiLU operator. This is benchmark plumbing, with no speed
measurement or numerical admission from the failed invocation.

## Segment screening results

These are complete **screened segments**, not complete model layers. Every
screen explicitly excludes the operations described above. Real checkpoint
weights use synthetic activations; QSA uses synthetic cache contents. GDN
repeats the same state buffers and therefore measures a hot-state proxy.
No delta below is credited against endpoint TPOT.

| Screen | Calls | M1 control / candidate (ms) | M5 control / candidate (ms) | M1 kernels control / candidate |
| --- | ---: | ---: | ---: | ---: |
| HC, redundant normalization | 16 | 0.286024 / 0.470241 | 0.552449 / 1.174292 | 64 / 32 |
| HC with preceding reduction | 16 | 0.384389 / 0.698982 | 0.716217 / 4.218812 | 96 / 32 |
| Shared expert, scalar intermediate tiles | 16 | 0.288256 / 0.546929 | 0.514108 / 1.721543 | 80 / 16 |
| GDN, complete heads | 36 | 0.229089 / 0.302019 | 0.649751 / 1.098775 | 72 / 36 |
| GDN, state tiles and fixed norm consumers | 36 | 0.231434 / 0.237076 | 0.649915 / 0.580351 | 72 / 36 |
| QSA selection through output gate | 12 | 0.521789 / 1.538847 | 1.018569 / 6.460572 | 48 / 12 |
| Router and complete experts | 4 | 0.126536 / 4.600648 | 0.367797 / 22.636609 | 16 / 4 |

All initial M1 candidates are rejected. The tile-preserving GDN variant improves
the M5 hot-state proxy by 10.7%, but neither improves M1 nor establishes a
cold-state whole-layer or model gain. Do not promote it on this evidence.
The earlier HC reduction control includes a staging copy and two reduction
kernels; its result is still negative, but it is not an exact production
reduction count. The later fixed-normalization screen uses the registered
production reduction control.

All candidates with retained grid geometry have zero single-CTA graph kernels.
Removing small grids alone is insufficient: redundant normalization, fewer
state tiles and serial row/head work can cost more than the removed launches.
The router and expert screen reports exact selected IDs and small numerical
errors, but its combined timing does not isolate the expert fusion cost.
An expert-only screen therefore freezes native routing outside the graph.

## Follow-up measurements

New captures count driver graph nodes directly. The fixed-normalization HC
uses four independent branch producers; each branch is normalized once.
The reduction control uses the production registered-buffer path. Both QSA
follow-ups rotate twelve separate cache allocations rather than repeatedly
reading one cache. The Tensor Core QSA screen retains the native exact selector
and FP32 QK/PV accumulation with FP16 probability materialization.

| Screen | Calls | M1 control / candidate (ms) | M5 control / candidate (ms) | M1 kernels control / candidate |
| --- | ---: | ---: | ---: | ---: |
| HC, fixed normalization producers | 16 | 0.286003 / 0.439747 | 0.553579 / 1.103773 | 64 / 32 |
| HC, fixed producers and preceding reduction | 16 | 0.351396 / 0.596746 | 0.663565 / 2.112250 | 80 / 32 |
| HC, vector loads | 16 | 0.285962 / 0.434831 | 0.553473 / 1.101521 | 64 / 32 |
| HC, vector loads and preceding reduction | 16 | 0.350290 / 0.586383 | 0.663231 / 2.078326 | 80 / 32 |
| HC, coalesced eight-row peer packets | 16 | 0.285737 / 0.563507 | 0.553627 / 1.540300 | 64 / 32 |
| HC, coalesced packets and preceding reduction | 16 | 0.355041 / 0.807260 | 0.665983 / 2.927887 | 80 / 32 |
| Packed per-token router and experts | 4 | 0.126536 / 0.497111 | 0.367809 / 0.857393 | 16 / 4 |
| Experts with frozen native routing | 4 | 0.081592 / 0.112855 | 0.289295 / 0.456638 | 8 / 4 |
| QSA, parallel heads | 12 | 0.554906 / 1.146860 | 1.032283 / 3.728270 | 48 / 12 |
| QSA, Tensor Core QK/PV | 12 | 0.556790 / 0.665313 | 1.032655 / 1.981976 | 48 / 12 |
| Shared expert, ten Tensor Core producers | 16 | 0.289188 / 0.282921 | 0.517016 / 0.470519 | 80 / 16 |
| Shared expert, forty K-partitioned producers | 16 | 0.289782 / 0.193219 | 0.308644 / 0.322171 | 80 / 16 |

All rows except forty-producer shared M1 are rejected for M1 admission. The
ten-producer shared M1 saving is only 0.006267 ms over sixteen calls and is
insufficient. The forty-producer M1 segment saves 0.096563 ms over sixteen
real weight banks (33.3%), with 32 single-CTA kernels reduced to zero.
This is a segment candidate, not an endpoint saving or completed layer gate.

The shared M5 numbers need their resolved control recorded. The ten-producer
row uses the strict FP32 cuBLAS fallback. Forty-producer M5 against that fallback
measures 0.516258 / 0.329388 ms, but this is **not** the shipped MTP speed
baseline. Against the shipped batched-up path and its existing precision
policy, M5 measures 0.308644 / 0.322171 ms, 96 / 16 kernels: a 4.4% regression.
Keep the existing M5 route. M1 maximum output error is 9.54e-7; the actual M5
control comparison has maximum error 1.91e-6. These arithmetic comparisons
do not substitute for teacher-forcing KL, top-1 or natural termination.

The first shared Tensor Core attempt encountered unaligned shared storage;
explicit sixteen-byte alignment fixes that correctness issue. Measurements
above come from the corrected implementations. A queue-script failure skipped
one invocation; it contributes no timing result.

The account cannot change application clocks, so the historical/current clock
difference remains unquantified. No clock setting changed and no candidate
has passed model distribution admission.

The packed-router follow-up uses a lossless layout matched to SM70 MMA
fragments and one multi-warp router CTA per token. Routing IDs match for all
four weight banks and all three activation scales; maximum output error is
3.81e-6. M5 driver graph counts are 24 / 4, with zero single-CTA kernels in
both arms. Packing removes much of the first router prototype's penalty but
the complete segment remains slower at both widths and is rejected. The HC
vector-load follow-up also remains slower at both widths, with and without
the preceding production reduction, and is rejected.

A forty-producer shared-M1 implementation is included in the ordinary
SM70 extension and opaque Python dispatch. M5 retains its existing native
batched-up route; prefill retains ordinary projections. Standalone complete
non-PLE decoder-layer captures retain actual M1 inputs and MTP4 verifier M5
metadata. Those layer graphs are explicitly separated from the compiled
whole-model endpoint. Ordinary-artifact build and M1 layer validation pass;
MTP4 layer and model distribution validation remain pending.

The coalesced HC experiment replaces one-row producers with two K partitions
per eight-row group and contiguous peer publications. Its first kernel has
twenty-two projection producers, four normalization producers and one fixed
gather consumer. Although its arithmetic checks pass and both complete HC
widths use two graph kernels, it remains slower and is rejected. Do not tune
this variant.

The ordinary CMake SM70 extension now builds the shared-expert operator. Its
paired sixteen-bank M1 segment measures 0.287642 / 0.192553 ms, 80 / 16
kernels and 32 / 0 single-CTA kernels; maximum output error is 9.54e-7. The
production M5 fallback measures 0.399836 / 0.401592 ms with identical outputs
and 96 kernels in both arms. These paired measurements use the same fresh
ordinary artifacts and do not replace the earlier artifact's endpoint
baseline. The reference source retains original Python model dispatch and
includes the unused additive CUDA registration, so reference and candidate
quality runs can use identical native artifacts.

Fake schemas preserve output shape/dtype at M1, M5, M17 and M33. Six
distribution-probe unit tests pass, including diagnostic-only raw logit
differences. Initial eager-model collection attempts selected a Python CUDA
norm wrapper that could not be traced, and then encountered mixed-dtype views
of recurrent-state storage that AOT could not functionalize. The successful
capture uses normal model initialization and independent benchmark state
storage with the original strides and active rows.

The ordinary-artifact complete M1 layer graphs measure:

| Non-PLE layer | Control / shared candidate (ms) | Kernels | Single-CTA kernels |
| --- | ---: | ---: | ---: |
| GDN layer 2 | 0.152791 / 0.149709 | 25 / 21 | 5 / 3 |
| QSA layer 3 | 0.304732 / 0.301912 | 37 / 33 | 9 / 7 |

These are critical-rank medians from seven alternating paired repetitions,
not endpoint TPOT. Hidden and injection outputs match; maximum MLP output
differences are 9.16e-5 and 3.05e-5, respectively. Each graph includes the
complete decoder layer, both HC transactions and production reductions. The
MTP4 M5 collection initially exhausted memory during 8K PLE prefill, before
layer measurement. Its layer-only retry uses 65536 startup capacity and 88%
memory budget, recorded separately from the 262144/94% M1 endpoint contract.
That retry also failed startup admission: 0.78 GiB KV was available, below the
1.1 GiB needed for 64K. The next MTP4 layer-only run uses 32768 capacity and
88% memory, retaining the actual 8192-token input. Neither failed collection
contributes performance evidence.

A separate QSA research phase combines the two input projections, both
norm/rotary preparation paths, main KV publication and output initialization.
It retains the native score, selection and sparse-attention core rather than
using the rejected coupled selector/core prototype. Offline SM70 compilation
passes at M1 and M5; complete-layer timing, selection and arithmetic checks
are pending.

A GDN research follow-up includes convolution, recurrent update and gated
normalization in one kernel, retaining the parallel value tiles. Norm consumers
delay convolution-state writes until their local value producers finish. The
Q/K commit additionally waits on the 48 producers sharing that query head;
there is no whole-grid barrier. It retains the MTP sliding convolution window
and commits every speculative recurrent state through the actual slot table.
CPU compilation passes; complete-layer output/state checks and timing at M1/M5
remain pending. No production dispatch is installed by either research import.

The first ordinary shared-chain model distribution run captures all 192
teacher-forcing rows: mean KL 0.002515, p99 0.032982, maximum 0.046212 and
top-1 agreement 97.396%. It fails the retained distribution gate; both natural
health requests nevertheless stop normally. Raw maximum logit difference
4.9375 is diagnostic only. All twelve first prefill rows match exactly, so
the prefill-path hypothesis is rejected. Inspection finds that the fused
shared chain omitted the native FP16 SiLU result boundary before multiplying
the up branch. Restoring that boundary requires no additional kernel. The
ordinary extension has been rebuilt; paired model requalification is pending.

The QSA layer prototype initially assumed MRoPE attributes, although this
language-only runtime uses ordinary RoPE. Its admission now follows the
existing pre-indexer for both forms. The GDN layer prototype also initially
assumed SiLU output gating; this model config selects sigmoid. That failed
entry check contributes no performance result. The complete-layer research
kernel now selects the actual output activation. Earlier SiLU GDN segment
screens remain research proxies and cannot qualify this sigmoid layer.

A new HC phase removes the first kernel's fixed gather consumer. Up/mix
consumes the peer-pushed down packets, computes only its TP hidden stripe and
publishes and collects output stripes inside its own resident kernel. The
previous coalesced M1 screen already used the native TP-sharded up operator;
its regression must not be attributed to replicated M1 up computation. The
replicated MMA up implementation belongs to its M5 research arm. CPU build
of the new two-phase HC passes; whole-layer M1/M5 timing is pending.

The corrected QSA preparation prototype passes the complete M1 layer screen:
critical-rank control/candidate medians are 0.285102/0.249477 ms, 33/23
kernels and 7/4 single-CTA kernels. All layer outputs and selection IDs match.
This paired run uses ordinary model initialization, the 8192-token prefix,
262144 startup capacity and 94% memory; the existing shared-M1 chain is enabled
in both arms. It does not replace the earlier artifact's endpoint baseline.
Actual MTP4 M5, at 32768 startup capacity and 88% memory with the same 8192-token
prefix, also passes: 0.457075/0.410872 ms, 41/30 kernels and 1/1 single-CTA
kernels. Layer outputs and selection IDs match on all four ranks. The shared
M5 fallback separately measures 0.269107/0.269425 ms for GDN layer 2 and
0.455547/0.456874 ms for QSA layer 3, with identical outputs and unchanged
28/41 kernel counts. These measurements describe verifier layers, not latency
per emitted speculative token.

The admitted preparation direction is being moved into a normal Python/Triton
module. Its M1 joint projection selects the original two weight pointers,
avoiding a persistent concatenated-weight copy. M5 preserves the two original
batch projections and fuses their subsequent preparation. Ordinary-route layer
and model qualification remain pending; the research M5 count cannot be
credited to this production variant before measurement.

The new HC peer-push/sharded-up phase reduces four graph kernels per decoder
layer but regresses M1 complete-layer time: GDN 0.147343/0.195369 ms and QSA
0.303332/0.339579 ms. Counts are 21/17 and 33/29, respectively. It is rejected
without further tuning. Actual MTP4 M5 also regresses: GDN
0.276275/0.447232 ms (28/24 kernels), QSA 0.453240/0.612452 ms (41/37).
The GDN M5 profiler includes one additional event in each arm, so its event
stream is marked incomplete; counts above come from retained CUDA driver graph
nodes. No production HC dispatch is added.

The SiLU-boundary shared requalification again captures all 192 rows and fails:
mean KL 0.002857, p99 0.030546, maximum 0.068376 and top-1 agreement 97.396%.
Raw maximum logit difference 5.046875 remains diagnostic. No further numerical
tuning is attempted for its small complete-layer gain. Normal model loading
no longer prepares or enables this chain. The ordinary native operator and
explicit benchmark preparation remain available solely for reproducing the
rejected experiment. QSA package layer and distribution tests now retain the
original shared-expert route in both arms and qualify QSA independently.

The corrected convolution/recurrence/sigmoid GDN prototype passes both complete
layer screens. M1 critical-rank medians are 0.148716/0.144978 ms with 21/18
kernels. Actual MTP4 M5 measures 0.275958/0.271739 ms with 28/25 kernels.
Convolution state matches; recurrent-state maximum differences are 1.49e-8
and 5.96e-8, respectively. Maximum hidden-output difference is 1.53e-5;
maximum MLP difference is 1.22e-4. These paired research graphs retain the
now-rejected shared chain in both arms. The M5 control profiler reports one
extra event and is marked incomplete; counts come from CUDA driver graph
nodes. Neither measurement qualifies an endpoint.

The GDN kernel is being moved into the ordinary native extension with explicit
device, dtype, layout and cooperative-residency checks. Normal preparation is
restricted to the validated sigmoid/FP16-convolution/FP32-state geometry.
Unsupported metadata retains the original complete forward method. Ordinary
M1/M5 layer checks and combined GDN/QSA distribution qualification are pending.

The ordinary QSA package passes its M1 complete-layer screen with the original
shared expert fixed in both arms: 0.285542/0.250563 ms, 37/27 kernels and 9/6
single-CTA kernels. All three layer outputs and selection IDs match on all four
ranks. Startup reports zero prepared shared chains. The graph profiler event
count agrees with retained CUDA driver nodes. Model distribution qualification
is still pending.

Ordinary QSA actual MTP4 M5 also passes: 0.448010/0.418263 ms, 41/32 kernels
and 1/1 single-CTA kernels. Layer outputs and selection IDs match on all four
ranks; profiler event counts agree with driver graph nodes. The production
variant deliberately preserves the original two batch projections, so its
32-kernel graph is distinct from the 30-kernel concatenated-weight research
variant. Model distribution and natural completion remain pending.

Ordinary QSA independent model qualification captures all 192 teacher-forcing
rows with the same native artifact as its original-dispatch reference. Every
full-vocabulary logit matches: mean/p99/maximum KL are zero, top-1 agreement is
100%, and raw maximum logit difference is zero. All category groups pass the
retained distribution gate. Natural completion is still running; combined GDN
qualification will use fresh original-dispatch reference and candidate captures
after installing the GDN native artifact under the complete GPU lease.
