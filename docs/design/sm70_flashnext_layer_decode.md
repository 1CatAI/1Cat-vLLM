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
control. A fresh normal native build from this owned tree is in progress.
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

No experimental runtime dispatch or new environment switch is installed.
The screen DSOs are research artifacts; they are not source-complete production
performance evidence. An admitted implementation must move into the ordinary
package build and pass a clean-artifact model gate.

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
speed improvement, kernel-count acceptance or default production admission is
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

Two distinct follow-up screens are compiled: fixed HC normalization producers
with native up/mix, and Tensor Core shared experts that reuse weights across
all M5 rows. They address the specific repeated-work and scalar-compute costs;
their GPU measurements remain pending. No runtime dispatch has changed and
no candidate has been submitted to model distribution admission.
