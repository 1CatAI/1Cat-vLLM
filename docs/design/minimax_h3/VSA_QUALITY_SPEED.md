# VSA quality and 31.3-second stage

The user resumed FastH3 VSA Data-Free development with original floating
weights, TP4 and the official top-k 64/four-step algorithm. This stage requires
independent FP32 full-sampling quality and human audiovisual review, plus a
full warmup and three unprofiled requests with median denoise <=31.3 seconds
and CV <=5%. Complete requests must beat a matched Dense control. Official
GPU-kernel validation and >80 useful TFLOP/s/card remain separate unfinished
objectives. No quality threshold is relaxed and no AUTO promotion is made.

## Frozen baseline and diagnosis

The source baseline is `970c5fb3f86d59440a3431e53029e98f8d690778`, which merges
the existing shared-kernel dependency into the owned VSA worktree. Commands,
source/binary hashes, input captures and diagnostic artifacts are retained in
`/home/ymzx/h3-sm70-artifacts-20260909/vsa-quality-speed-20260910/`.

The native host policy now matches the retained Dense configuration: pageable
shared VAE masters, prepared floating columns and explicit native TP4 peer
rows with a 4 GiB communication budget. A complete primary capture preserves
the previous VSA final video/audio latents, all pre-encoding RGB frames and
PCM bitwise. Its 34.895545-second denoise is a cold diagnostic including input
capture, not formal performance or a quality repair.

A separate profiled request records a 32.039245-second denoise NVTX span on
the slowest rank. Sparse Attention accounts for 7.343837 seconds and GEMM for
15.466638 seconds. The initial parser classified peer `reduce_rows` in other
kernels; its 3.654961 seconds must be included in communication, alongside
1.727143 seconds of other collectives. Profiler start/stop overhead is outside
this NVTX span but inside native stage timing; neither time is an unprofiled
acceptance result. No NCU occupancy/utilization claim is made.

The fixed-input diagnostic shows probability rounding contributes to error,
but a compensated-PV prototype only reduces actual operator relative L2 from
0.000203199 to 0.000135391. It fails its numerical admission criterion and is
not installed or run as a full-sampling candidate. Further localization keeps
QKV, selected blocks and gates fixed; FP32 output and QK/PV diagnostics are
separate from production precision.

## Deferred work accounting and internal validation

Dynamic selected-pair and selected-block counts stay as int64 device scalars
while layers execute. The request-owned counter retains their producing
tensors and step/layer association, then copies all counts to CPU together
after complete denoise. Public result fields remain Python integers and are
reconciled across steps, layers and totals. Counter completion remains inside
the complete-denoise timer and before the final TP barrier. Closing or failing
a request removes hooks and releases pending tensors.

H3-owned geometry and a nonempty mask produced by the official top-k/prefix
construction use a private prevalidated CUDA entrypoint. Device, layout,
dtype, shape, alignment and index-limit checks remain. The general sparse
operator still validates block-size values and rejects empty query rows.
Older wheels use its checked entrypoint until rebuilt; no user flag bypasses
validation. Kernel arithmetic is unchanged.

Validation so far: 73 CPU checks pass, 2 GPU checks skip in the CPU run;
23 leased-GPU checks pass, including real dense/sparse DiT work accounting,
tail/padding controls, strict public input rejection, exact public/private
output equality and absence of host scalar reads in the private entrypoint.
The rebuilt sparse kernel retains 215 registers and zero spills. Nine further
checks pass under both Compute Sanitizer memcheck and synccheck with zero
reported errors. The
complete primary capture preserves final video/audio latents, pre-encoding
RGB and PCM bitwise. Selected blocks, valid pairs and useful FLOPs also match
at every rank, layer and step. Its 34.458830-second cold captured denoise is
diagnostic only. This preserves the old native output, whose independent
FP32 quality still fails; it does not establish engineering quality.

## Full-sampling amplification diagnosis

A separate artifact-only prototype resets the FP32 PV accumulator for each
64-key block and adds a scaled low probability component. Fixed-input relative
L2 improves from 0.000203199 to 0.000046140, but this still misses the prototype's
fivefold numerical improvement criterion. It is not installed. A full run was
used specifically to localize amplification, not as acceptance or performance
evidence. The centered-exponent variant adds no useful numerical improvement.

The independent FP32 reference was run again and reproduces the original final
video and audio latents bitwise. With identical initial tensors, the block-local
candidate's video latent errors after the four steps are 0.003839, 0.016306,
0.056423 and 0.369510; final audio error is 0.050502. Both final latent gates fail.
Selection first differs in the second layer on all ranks. By the last layer of
the first step, 73.2--78.4% of query blocks have at least one changed selected
block. This is the fraction of affected queries, not the fraction of replaced
keys. Fixed-reference-map controls separate continuous arithmetic error from
dynamic-selection amplification. They cannot be shipped. Fixed reference maps
still give final video/audio errors of 0.189996/0.016187 with the native kernel
and 0.175657/0.014001 with compensated block-local PV. Exact FP32 prefix queries
with dynamic video selection give 0.401763/0.086767. All three fail; neither
fixed routing nor prefix precision alone solves the problem.

The native frontend using the same FP32 sparse operator matches the independent
official frontend bitwise on real inputs, including the official transport-only
partner block. The block-local diagnostic before final FP16 rounding has relative
L2 0.000005753 against FP32 (prefix 0.000015489, video 0.000005443). A further
artifact isolates exact FP32 QK/softmax and compensated 64-key PV partials; no
production sparse-math change is admitted from these local numbers.

## Exact layout fusion

One kernel gathers Q/K/V directly into their final padded tiles. Another
combines learned compression with the sparse output and restores the original
row order, preserving separate FP16 multiply/add rounding, including overflow.
The original pooling, top-k, sparse traversal and work reductions are retained.
An older extension or strided/unaligned inputs use the existing Python layout.

Only geometry indices are cached across requests. A denoise context owns QKV
scratch storage separately for each device/stream and releases it on success or
failure. Nested contexts restore their parent. Every returned output is freshly
allocated, so a later layer cannot overwrite an earlier result.

The artifact prototype preserves real primary outputs and useful counts bitwise.
The corrected comparison includes work counting on both sides: median 41.969666
ms baseline versus 37.358593 ms fused over seven paired operator measurements.
This is an operator result, not complete-denoise acceptance. The installed source
passes 30 GPU tests covering layout, sparse math and geometry; its eight layout
checks also pass memcheck and synccheck with zero errors. The complete primary
capture preserves final video/audio latents, all 124 pre-encoding RGB frames and
PCM bitwise; every rank/layer/step work count is identical. Its captured denoise
is 32.989041 seconds, which is diagnostic only. Independent FP32 engineering
quality remains incomplete; VSA is not promoted into default or AUTO selection.

## Stage timing evaluation

After separate full native benchmarks for VSA and Dense, run:

```bash
python -m vllm.video.vsa_acceptance --vsa VSA_BENCHMARK_DIR \
  --dense DENSE_BENCHMARK_DIR --output stage-performance.json
```

The tool reuses strict same-session warmup, three-request and rank/step/layer
accounting validation. It additionally requires the primary dimensions, TP4,
top-k 64, official FastH3 four-step sigma positions and matched request, GPU,
host-weight, VAE, communication and export settings. Only backend, query tile,
top-k and adapter path may differ between the two algorithm controls. It checks
the slowest rank's median denoise <=31.3 seconds, CV <=5%, and a strictly lower
complete-request median than Dense. The future 80-TF criterion is explicitly
reported separately. Passing this timing tool does not establish weight
identity, numerical/human quality, other-shape coverage or official hardware
validation.

## Matched formal timing: speed gate passes, joint stage incomplete

Source `450f9b9bc6458e31ff83150f33b79df31a89732d` with the retained
`native-layout-binaries.json` completed one full warmup and three consecutive
unprofiled, uncaptured single requests per backend. Both controls use the same
original H3 weights, request, TP4 GPUs, pageable shared VAE masters, prepared
FP16 columns, 4 GiB native peer-row communication budget and libx264 export.
Each uses its corresponding official FastH3 Data-Free adapter. VSA uses top-k
64 and query tile 64; Dense FA uses query tile 128.

| Measurement | VSA, existing FP16 sparse math | Dense FA |
| --- | ---: | ---: |
| Denoise run 1 (s), slowest rank | 30.948464 | 52.886819 |
| Denoise run 2 (s), slowest rank | 30.999521 | 52.873835 |
| Denoise run 3 (s), slowest rank | 30.990756 | 52.858062 |
| Median denoise (s) | 30.990756 | 52.873835 |
| Denoise CV | 0.071956% | 0.022239% |
| Median complete request (s) | 68.157817 | 87.426512 |
| Peak allocation bound incl. raw IPC (GiB/card) | 20.843828 | 19.554280 |
| Effective model TFLOP/s/card, lowest rank median | 54.611483 | 57.553944 |

Per-step medians of the maximum rank GPU time are 7.730810, 7.748937,
7.747062 and 7.753540 seconds for VSA, versus 13.183612, 13.219172,
13.240814 and 13.227989 seconds for Dense. These event timings do not replace
the complete-denoise wall-clock criterion, which includes counter completion.
Avoided sparse work is not included in useful throughput.

`native-layout-stage-performance.json` passes all three timing checks. The
underlying sparse arithmetic still fails the independent FP32 reference:
video/audio latent relative L2 is 0.390670/0.088636. This timing pass cannot be
combined with the quality pass of a different, slower kernel. No configuration
has yet passed the joint stage. `vsa-layout-formal-telemetry.json` and its Dense
counterpart retain per-device clocks, power, memory and utilization samples;
these span the complete campaign and are not denoise-only utilization figures.

## Exact FP32 sparse CUDA candidate

The compensated-PV experiment with exact FP32 QK/global softmax still fails
full sampling (video/audio latent errors 0.367458/0.074789), despite a fivefold
local improvement. Admission now requires real-input FP32 parity before another
full sample; a small local relative error is not sufficient evidence.

Sequential FP32 FFMA QK and PV match cuBLAS FP32 on seven real query subsets.
A directly indexed CUDA implementation avoids gathered K/V copies, compacts
selected blocks in ascending order, preserves prefix-dense/video-sparse queries
and uses a global FP32 softmax. Its 64-by-64 output-tile variant matches all
609 blocks of the actual primary operator bitwise, including five boundary,
extreme-score and poisoned-padding cases. Six small cases pass memcheck and
synccheck with zero errors.

The complete dynamic-selection seed-42 sample also matches the frozen reference
video/audio latents and all decoded RGB frames bitwise. PSNR is infinite,
SSIM is 1, audio spectral cosine is 0.999999999999945 and RMS ratio is
0.999999998643185. This is a numerical pass only: human review remains pending.
Captured denoise is approximately 109.6 seconds, so this kernel fails speed.

The 64-by-64 operator takes approximately 407 ms. Nsight Systems attributes
158.81 ms to QK, 215.65 ms to PV and 32.37 ms to global softmax. Shared-memory
padding (411 ms) and a 4-by-4 register tile (437 ms) preserve bitwise output but
are slower and rejected. A CUTLASS SIMT candidate preserves the five boundary
cases and full primary operator bitwise at approximately 233 ms. Six small
cases pass memcheck and synccheck with zero errors. Nsight Systems attributes
75.76 ms to QK, 116.46 ms to PV and 32.46 ms to global softmax. Its complete
seed-42 sample preserves initial video/audio noise, text tensors, every step
and final latents bitwise. All 124 RGB frames are identical; audio spectral
cosine is 0.999999999999945 and RMS ratio is 0.999999998643185. Captured denoise
is 74.545147 seconds and complete request is 225.097778 seconds; the latter
includes cold staging and capture and is not a matched formal measurement.
The primary numerical gate passes, but speed and human review do not. No
precision candidate changes the runtime backend or sampling algorithm.

Remaining acceptance work: one configuration satisfying both quality and speed,
seeds 43/44, 243-frame and 15-second boundaries, TP1/TP2 compatibility, representative
Dense/LightX2V/Ref2VA output regressions, and human audiovisual review. Official
original-hardware comparison remains explicitly deferred.

## Reproducing the precision diagnostic

The acceptance-only source is `benchmarks/kernels/h3_vsa_fp32.cu`. It has no
runtime backend registration. Build it separately with CUTLASS 4.4.2, CUDA
12.8.93, Torch 2.10.0+cu128, Python 3.12.13 and SM70:

```bash
CUDA_HOME=/path/to/cuda-12.8 TORCH_CUDA_ARCH_LIST=7.0 MAX_JOBS=2 \
  .venv/bin/python benchmarks/kernels/benchmark_h3_vsa_fp32.py \
  --build-directory /path/to/diagnostic-build \
  --cutlass-root /path/to/cutlass-4.4.2 --output /path/to/build.json
```

After acquiring a GPU lease, run the regression checks against that exact
binary. For sanitizer checks, put `compute-sanitizer --tool memcheck
--error-exitcode 86` or `--tool synccheck --error-exitcode 86` before Python:

```bash
H3_VSA_FP32_DIAGNOSTIC=/path/to/diagnostic-build/h3_vsa_cutlass_fp32.so \
  .venv/bin/python -m pytest -q tests/video/test_h3_vsa_fp32_diagnostic.py
```

The same benchmark accepts `--binary`, `--capture attention-input-rank-0.pt`
and `--output`. It compares the complete captured operator with independent
FP32 math before measuring it. Reports retain binary/source hashes and mark
operator measurements ineligible for complete-denoise acceptance. Supplying
an existing binary does not assert that it was built from the current source.
The diagnostic binary must never be substituted into a formal runtime timing
record without recording the actual operator override.

The initial versioned diagnostic passes 20 leased-GPU regression checks,
including six mathematical/boundary cases and fourteen public input rejection
cases. The complete captured primary operator also matches FP32 bitwise. All
20 checks pass memcheck and synccheck with zero errors. Its build source hash is
`f21ae1df63032be02853c2968dacdc001511bb5729942c8b006e316a0771d8fc`.
The full quality capture used the arithmetic-identical artifact build
`cafdc331dfbc2218e6e982a6ce8b4d8c4cd562ad45e4106ca5968de4fb911349`;
the separately built versioned extension is
`ef4a9b8ebee7091b71951692d46313a1b59c322f1970d1925759a285c1d8cb96`.
The source body is unchanged apart from inlining includes, using the base
CUTLASS header directly, formatting and comments. These binary identities
must remain distinct in retained measurement records.

## Batch independent FP32 queries

The initial prefix PV launch has only 14 threadblocks on an 80-SM V100. Its
query-chunk limit was inherited from the gathered-K/V oracle, although the
CUDA implementation indexes shared converted Q/K/V directly. The diagnostic
now submits at most 32 independent query blocks together, with a 2 GiB limit
on each score/probability allocation and a matching CUDA grid-size guard.
Requests with even one query exceeding that allocation limit are rejected.
Each query retains its own compact selected indices and ascending key order;
there is no selection union, changed probability precision or split-K reduction.

Source `cb33adbdd821a77a8f7f5575f0a6acb68fa5bb606d2b41dd02657d458e448e56`
builds binary
`8a0192009af23b25d559b91e9155385266f9574c6e92adba3570d67b539f5cd6`.
The exact versioned binary passes all 20 GPU, memcheck and synccheck checks,
including a 38-block case crossing the new query-batch boundary, with zero
sanitizer errors. The complete captured primary operator remains bitwise FP32,
with median 172.439 ms versus 232.740 ms before batching.

The complete primary sample preserves the bytes of initial inputs, all four
step outputs and final video/audio latents against the frozen reference. All
124 RGB frames match; PSNR is infinite, SSIM is 1, audio spectral cosine is
0.999999999999945 and RMS ratio is 0.999999998643185. Human review is pending.
Captured denoise is 61.826728 seconds, complete request 164.385431 seconds,
and peak allocation bound including raw IPC is 24,884,304,896 bytes/card.
These captured times are diagnostic; the original and batched captures are
not a formal warm-session comparison.

A separate one-warmup/three-request benchmark is pending. Its provenance must
explicitly name `CUTLASS_SIMT_FP32_BATCHED_DIAGNOSTIC` and the binary/source
hashes: the native sampler, work counters and timers are unchanged, but the
acceptance tool replaces the sparse operator. This remains an opt-in diagnostic
with no runtime/default/AUTO registration. The joint 31.3-second quality/speed
stage remains incomplete.
