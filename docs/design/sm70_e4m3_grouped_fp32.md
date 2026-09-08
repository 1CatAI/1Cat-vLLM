# Experimental E4M3 grouped attention with FP32 partial state

**2026-09-08 policy update:** [DFlash2 FP32 defaults](sm70_dflash2_fp32_defaults.md)
supersede the default-routing and KV-alias decisions recorded below. E4M3
DFlash2 verification now uses repaired FP32 state; revision 4 adds the
1728/3456 page layouts. Earlier measurements and failed model gates retain
their original artifact attribution.

## Scope and admission

**Current mainline audit decision (2026-09-07): enabled for compatible
SM70 E4M3 single-request small-Q attention.** The maintainer's acceptance
criterion is numerical/output quality and speed, not identical greedy tokens.
The historical strict-token failures below remain recorded and are not
reclassified as deterministic passes. They do not demonstrate semantic
collapse: the latest boundary case produces two coherent continuations with
a small top-two margin, while attention error is at the FP16 rounding floor.

Fresh integration against main `8d9c35189920` (including #548) passes 246
focused kernel/routing checks with a rebuilt CUDA 12.8 extension. The same
source's unchanged scalar E4M3 fallback and repaired grouped route were timed
in alternating five-sample CUDA Graph pairs on one V100-SXM2-32GB:

| Q / page / context | Scalar fallback (us) | Grouped FP32 (us) |
| --- | --- | --- |
| 2 / 800 / 8192 | 204.256 | 63.853 |
| 5 / 848 / 8192 | 405.286 | 69.882 |
| 8 / 1616 / 65536 | 6072.998 | 394.688 |
| 5 / 1648 / 131072 | 7329.069 | 668.326 |
| 5 / 3296 / 262144 | 14626.714 | 1303.667 |

These are synthetic-input operator measurements, not model throughput.
The 246 checks include same-quantized-KV FP64 references, changing row-length
graph replay, zero rows, misaligned Q and padded KV strides. No full E2E
suite was run for this integration. Old binaries retain the safe fallback;
the new route requires precision ABI revision 3. Main's separate v37 prefill
bridge and E5M2 paths are preserved. AI-assisted audit.

This change integrates the small-query part of a private E4M3 migration into
`onecat/main`, based on `755baae1d075ee04fa9096b23fc0225b23589a86`.
It does not change the default KV format, the existing prefill implementation,
the cache writer, B1 decode, or the default DFlash2 verifier route.

`VLLM_FLASH_V100_E4M3_GROUPED_FP32=0` rolls back the rebuilt native entry for
explicit `fp8_e4m3` KV. The flag defaults to **on**. Older extensions without
the new entry retain the ordinary backend route. Set the flag before worker
initialization; rebuild and restart task-owned workers to change the route.
The extension must report `grouped_e4m3_fp32_precision_version() >= 3`.
Presence of the original forward symbol alone is insufficient: older builds
contained the repeatable numerical counterexample described below. A stale
binary retains the ordinary backend fallback and emits a rebuild warning
when the experimental flag is requested.

The supported contract is SM70, FP16 Q/output, 2–8 query rows, six query heads,
one KV head, D256, and page sizes 800/848/1616/1648/3296. The parent attention
metadata must describe exactly one request. Independent-request batches,
sliding windows, explicit partition overrides, incompatible layouts, and
unsupported shapes retain the existing fallback. Valid physical block IDs
and per-query lengths within the block-table capacity are caller invariants.
The wrapper is not an API for interpreting E5M2 bytes as E4M3.

The selected entry logs `experimental E4M3 grouped FP32 route selected` and
records `prefill_smallq_e4m3_grouped_fp32`. A requested environment flag alone
is not evidence that an inference used this entry.

## Design

We reuse the existing grouped Tensor Core dataflow to scan KV once for a
packed query/head group. The repaired precision revision keeps accumulation
and unnormalized partition numerators in FP32, with these safeguards:

1. QK uses K16 Tensor Core products and compensated FP32 summation across D.
   Explicit round-to-nearest additions preserve the correction under the
   standard fast-math build.
2. PV consumes both the high FP16 probability and its FP16 rounding residual.
   The latest experiment stores the residual multiplied by 2048 and divides
   the second product's E4M3-derived V operands by 2048. The inverse scaling
   is exact in FP16 for every finite E4M3 encoding; the purpose is to retain
   small residuals that otherwise underflow. This experiment's model and
   performance admission is separate from the recorded revision-2 results.
   The softmax denominator remains the original FP32 probability sum.
3. Each N32 PV tile starts with a fresh FP32 Tensor Core accumulator. We update
   the longer-lived FP32 online state with one scalar FMA per output element,
   combining rescaling and addition. This avoids repeatedly feeding a large
   accumulator into Tensor Core products of increasingly small corrections.
4. Revision 3 preserves separate FP32 max/sum statistics and the unnormalized
   PV numerator until the final combine. It avoids rounding a normalized
   partial and reconstructing its weight from `max + log(sum)`. This is a
   representation change, not a new production FP64 path.

Tensor Core operands and the final output remain FP16. These changes reduce
arithmetic error on identical quantized KV; they do not remove FP8 quantization
loss or promise bitwise equivalence to FP64 on arbitrary inputs.

For q5, only 30 packed rows are live. We skip the wholly padded third M16 tile
in QK, PV, and softmax. A separate padded residual panel adds 3.75 KiB of
shared memory per CTA without additional KV reads or global workspace.
The measured q5 build still uses 128 registers/thread and one CTA per SM.
The q5 timing evidence is not a speed claim for every admitted query width.

The new entry consumes the metadata builder's per-query device lengths.
Those lengths, not the padded Q shape, determine each causal prefix. The
maximum live length controls partition assignment. Inactive rows explicitly
write zero before reading partial state, including when all rows are empty.
The same captured graph can therefore replay after padding or length changes.

The FP32 workspace is keyed separately from the old FP16 workspace, including
device and stream. Its partial tensor is `[80, 8, 6, 256]` FP32 (3.75 MiB),
plus `[80, 8, 6, 2]` FP32 max/sum statistics (30 KiB). Revision 2 used 15 KiB
of LSE statistics; old extensions require a rebuild for this ABI. The legacy partial tensor is
1.875 MiB. Keeping a reference to an existing graph also keeps its workspace
alive; this is not a claim of zero concurrent-memory cost.

Existing E5M2 q8/q16 and sparse-page4 instantiations retain their template
defaults and FP16 partial storage. The new entry does not modify their
public call signatures. Unlike PR #517's E4M3 DFlash2 q8 port, this scope adds
FP32 partial state and explicit row lengths for dense small-query MTP input;
it does not duplicate its logits, sampling, or context-pipeline changes.

## Evidence boundaries

The private prototype and this integration build are different artifacts.
Private measurements motivated this implementation; they do not automatically
approve the rebuilt integration artifact or its end-to-end performance.

The retained private bundle `e4m3-precision-parity-20260906` includes 24 actual
MTP q5/page848 groups (four context lengths, three layers, two rounds). The
FP32-partial prototype had arithmetic relative-L2 0.0199%–0.0224% against
PyTorch FP64 on the identical quantized KV, lower than its captured scalar
control in every group. This is **not total FP8 error**. Four TP4 short
retrieval cases matched the control's 11 output token IDs, but that is not a
full long-generation quality gate. Private activations, model data, binaries,
paper drafts, and machine-specific paths are deliberately excluded from Git.

The integration build passes 39 GPU tests (12 new FP32 cases, 25 existing
grouped cases, and two sparse-page4 cases) on physical GPU4. Its 12 new cases
also pass Compute Sanitizer memcheck on GPU5 with zero errors. An additional
24-group private real-input replay on GPU6 gives arithmetic relative-L2
0.0198771%–0.0224338%, lower than the captured scalar control in every group.
Only one of those 24 outputs is bitwise identical to the private prototype.
Current-main dataflow and the standard build's `--use_fast_math` differ from
the private artifact; this replay does not establish the cause of every
rounding difference and does not transfer its model-quality approval.

The complete routing-policy selection passes 138 tests on physical GPU4,
including explicit-length forwarding, disabled-entry fallback and multi-request
rejection. An earlier CPU-hidden run had two existing stream/device-query
failures (132 passed, one skipped); rerunning with an idle visible GPU resolves
both without modifying those tests or changing their assertions.

The native library SHA256 is
`953924a40df189dea60b4bbc7624bdb1bd6e1460176f2f111b21f3b2f9c959f0`.
The real-input replay result SHA256 is
`f3a444a60aab55f67964d0cfb40db80d9297f2905278d296ebe0fa001fbc48b5`.
The memcheck log SHA256 is
`a9d061cee538182f762e132abeea7bac7bddb844338f25751bbbb043f242c425`.
Build/test artifacts remain in the owned worktree's `.artifacts/e4m3-fp32/`;
they are not distributed as source or installed into running services.

Environment: V100-SXM2-32GB, Python 3.12.13, Torch 2.10.0+cu128, CUDA 12.8
runtime, GCC/G++12, and the locally available CUDA 12.0 compiler. The build
logs retain the CUDA minor-version warning; this is not a CUDA 12.8 compiler
validation. Both Flash-V100 and `paged_kv_utils` build successfully. The
runtime import is verified to come from the task's build directory.

Default enablement requires a fresh matching model-quality and unprofiled
performance gate on the integration build, including longer outputs. There
is no new model-throughput claim in this change.

## Reproduce focused checks

Use a task-owned virtual environment and an idle SM70 GPU. Build Flash-V100
with the repository's `setup.py build_ext`, using separate build and library
directories. Point `PYTHONPATH` at that library directory and this checkout's
`flash-attention-v100` source package; importing an installed stale extension
does not validate this source change.

```bash
.venv/bin/python -m pytest --confcutdir=tests/kernels/attention -q \
  tests/kernels/attention/test_sm70_grouped_e4m3_fp32.py \
  tests/kernels/attention/test_sm70_flash_v100_grouped_verify.py \
  tests/kernels/attention/test_sm70_qsa_grouped_page4.py

.venv/bin/python -m pytest --confcutdir=tests/v1/attention -q \
  tests/v1/attention/test_sm70_e4m3_grouped.py \
  tests/v1/attention/test_sm70_flash_v100_policy.py
```

The new operator tests use randomly relocated pages, non-unit KV scales,
FP64 reference outputs, and one graph with live/all-zero/tail-zero/first-zero/
restored row lengths. They cover every admitted query length and page family,
and the exact 262144-token boundary. Workspace tests check that FP32 and FP16
buffers do not alias. Policy checks cover default-off behavior, native
capability detection, invalid shapes and independent-request rejection.

Run the same operator checks under Compute Sanitizer memcheck before promotion.
The test suite needs a visible idle GPU: some existing policy tests query a
CUDA stream even when their tensors are on CPU.

## Remaining gates and rejected routes

- This is an experimental code integration, not default deployment approval.
- Long-output model quality and current-main TP4 performance remain unapproved.
- A separate private paged SplitKV E4M3 port corrected an inherited merge-row
  mismatch but remained slower than its E5M2 counterpart. It is not included.
- MTP5 dual-CTA equivalence, independent batches, and other GQA ratios need
  their own evidence; they are not covered by this single-request gate.
- AI assistance was used. Human line-by-line review and confirmation of the
  relevant tests remain required before merging, per the repository policy.

## Model admission update: a repeatable counterexample

The original integration artifact (`953924a...959f0`, CUDA 12.0 compiler)
has now been exercised in Qwen3.8-27B-FP8, TP4, native MTP4, explicit E4M3
KV, FP16 activation/SSM cache, 262144 maximum length and CUDA graphs. At an
8192-token prompt, an 11-token retrieval response matches across
control--candidate--control, but this short smoke hides a longer-output
counterexample. Greedy sampling is an explicit regression contract, not an
official-sampling quality evaluation; EOS is not suppressed.

Initially, the two controls themselves diverged at output token 177, while
the candidate differed from the first control at token 229 (one-based).
The unstable controls make that initial comparison inconclusive; per-start
GEMM tuning is one possible confounder. With
FP8/AWQ small-shape tuning and FP8 coordinated tuning disabled **only for
diagnostic isolation**, two independent control processes reproduce all 256
output tokens and MTP acceptance length in six runs including warmups. The
FP32 candidate instead reproducibly differs at output token 102 in all
three runs. All recorded top-five logprobs are finite and both texts remain
coherent, but the no-token-divergence gate fails.

A separate counterfactual replaces only admitted small-Q attention with
PyTorch FP64 over the identical quantized KV, retaining FP16 output and all
other model computation. Its three runs match the control's entire 256-token
sequence. At the first divergence, the logprob margin for token `343` over
token `5604` is +0.046875 in the control, -0.046875 in the candidate, and
+0.093750 in this reference. This is an attention-only reference, not a
full-model FP64 or unquantized-KV oracle. The reference used GPU0--3; the
fixed-dispatch native runs used GPU4--7. No absolute timings are pooled.

This counterexample blocks promotion; lower aggregate operator L2 alone
does not grant model acceptance. It is not evidence that E4M3 is worse than
E5M2, nor a verdict on the private long-prefill implementation, which was not
loaded. Do not expand to the long-context speed sweep or change defaults
before localizing this repeatable failure. Raw bundle:
`e4m3-mainline-model-gate-20260906`, indexed in the private handoff.

## E4M3 bridge prerequisite, without default routing

The backend's existing prefill bridge is still E5M2-only. The explicit
`fp8_e4m3_paged_kv_to_fp16` entry adds the missing conversion building block,
sharing the paged scheduling and unit-scale specialization with E5M2. It
preserves signed zero, scales in FP32, rounds to FP16, and zeroes the live
prefix's 16-token padding. It does not reinterpret E5M2 bytes, select a
backend route, or change the global KV format. Old extensions fail with an
explicit rebuild message when this new entry is requested.

The bridge build uses CUDA 12.8.93/GCC12, Torch 2.10+cu128, SM70 and standard
fast-math flags. Its native SHA256 is
`c1ce4140fe82a94fba8c351e219e72ca6676f4ebb1fcc54fc7444679e08b2003`.
The 30 bridge cases cover both formats, all 256 byte encodings, five page
sizes, three scale pairs, relocated pages, signed zero and graph length
changes. Together with 39 grouped/sparse regressions and one missing-native
entry check, 70 tests pass; the routing-policy suite also passes 138 tests.
Memcheck covers these 30 cases and 12 FP32 grouped cases with zero
errors. The same 24-group activation replay has relative-L2
0.0198771%--0.0224338%, all below the captured scalar control. These are
operator results, **not model approval for this updated artifact**.

Artifact directory: `.artifacts/e4m3-bridge-fp32/`. To reproduce the added
conversion checks, run the ordinary GPU and memcheck workflows above with
`tests/kernels/attention/test_sm70_fp8_bridge_formats.py`. The small-Q
selection message is now process-scoped so future gates can check every TP
rank rather than mistaking rank-zero-only logging for missing execution.

## Precision-repair update

The repair separates errors hidden by the final FP16 output floor. A replay
of 200 retained real groups includes the previous 24 groups and 176 additional
rank-zero target-MTP groups near the natural-prose regression. Instrumenting
the model to capture those groups changes its generation trajectory; captured
inputs are used for same-input arithmetic analysis, not uninstrumented model
admission or timing.

Probability compensation alone reduces operator error but does not pass the
model gate. With compensated P and a long-lived Tensor Core accumulator,
the median partition-state relative-L2 at approximately 256K is `2.95e-5`
after removing QK and final FP16 rounding effects. Replacing only the merge
weights with FP64-reference weights barely changes that error. Tile-local PV
and FP32 online-state updates reduce it to `2.92e-7`. The biased-V synthetic
regression reproduces the old failure (`2.93e-5` against a `3e-6` bound), while
the repaired build passes. Zero-mean random V alone did not catch this issue.

A separate model counterfactual retains the native-order QK computation but
uses FP64 softmax/PV. It still diverges at output token 177 in the new GPU0--3
cohort, supporting the additional compensated QK summation. The full FP64
attention-only reference again matches the entire control output in three
runs. This is not a claim about all-model FP64 arithmetic.

With all three repairs, the 8K natural-prose counterexample matches all 256
reference/control token IDs in three candidate runs. The native controls on
either side are token-stable. In the 200-group operator replay, disagreements
with the FP64 result rounded to FP16 decrease from 197210 elements to 2092;
these counts describe attention tensor elements, **not token errors**. The
maximum output relative-L2 divided by the unavoidable FP16-rounding floor is
`1.000018`.

The same-GPU, q5/page848, 20-warmup/100-ABBA CUDA-graph comparison against the
original FP32-partial candidate measures:

| Actual KV length | Original latency (ms) | Repaired latency (ms) |
|---:|---:|---:|
| 8197 | 0.071053 | 0.071027 |
| 65541 | 0.356045 | 0.353997 |
| 131077 | 0.687693 | 0.685722 |
| 261893 | 1.338496 | 1.335757 |

This establishes essentially unchanged operator speed in this contract,
not a substantial new speedup, a 60-TFLOP/s prefill result, or production
end-to-end throughput. Model isolation still disables per-start GEMM tuning;
the diagnostic setting is not promoted to a production default.

The repair-algorithm DSO is
`2f48d070dbe7536c011a6585334da61afd943d70a6ff68dc024ca90ababd2f43`.
The capability-guarded build is
`252da72e37a65ca40b4b821960c0d62d11a705e0d08e2a651217ad28f9d13e86`.
Their 200 replay outputs are bitwise identical. The final four-context
model cohort, broader production sampling/performance admission, and human
review remain separate gates. The experimental flag and global KV-format
defaults are unchanged. Artifacts remain in the owned worktree's
`.artifacts/e4m3-output-repair/`, with large private data outside Git.

### Remaining long-context counterexample and residual experiment

The subsequent four-context cohort does not establish promotion. Against its
attention-only FP64 reference, revision 2 matches all 256 tokens at 8K/64K but
first diverges at token 151 at 128K. The approximately 256K candidate completes
generation, while the corresponding reference/control are incomplete. The
ordinary control also differs from the FP64 reference (token 177 at 8K and
248 at 128K), so it is not treated as an exact numerical oracle. Cross-process
results must be distinguished from same-process reference brackets.

Retained 128K diagnostic frames expose another avoidable source: direct FP16
storage of an unscaled probability residual. On one real layer-27 frame,
with all other arithmetic in FP64, the residual representation contributes
relative-L2 `1.98e-7`; power-of-two scaling reduces this to `5.87e-9`. This is
a representation-only counterfactual, not a kernel or model result. The new
regression isolates scores 0/-8 and inspects FP32 partition outputs before
final FP16 rounding, including positive/negative minimum and maximum finite
E4M3 values to exercise subnormal scaled V operands on actual hardware.

The scaled-residual experimental library is
`87ffc1fd626e3d0fa46f6e2bda741a88d79d53809685a4d569c37e4d35b325cd`.
Its 89 GPU checks pass. The new small-probability test rejects revision 2
(`2.2877e-5` relative error against a `3e-6` bound), while the scaled-residual
build passes all six signed/range cases. In the same 200-group replay,
FP16 element disagreements with the FP64-rounded reference decrease from
2092 to 1872; in 32 additional real 128K frames they decrease from 348 to 320.
The four same-GPU q5/page848 100-ABBA speed ratios versus revision 2 are
1.0007/1.0000/1.0007/1.0007, effectively unchanged. These improvements do not
establish bitwise FP64 equivalence or model quality.

A completed revision-2 same-process 128K reference--candidate--reference
bracket matches all 256 token IDs in all three requests; the two references
are stable. It does not reproduce the earlier cross-process token-151
difference, whose full cause remains unlocalized. This instrumented,
cached-prefix bracket is not a cold-process or throughput benchmark.
The scaled-residual build also passes 62 Compute Sanitizer checks with zero
memory errors. Its same-process 128K bracket has now completed and **fails**
the deterministic token gate: the two FP64 reference requests match all 256
tokens, while the candidate first differs at token 77. Both references select
token 799 (`one`) and the candidate selects 23438 (`engineers`). The two
reference top logprobs are tied; the candidate favors 23438 by 0.015625. A
near-tie does not waive the stated no-divergence gate or establish broad
semantic degradation.

All four TP workers select the native entry. Rank-zero instrumentation covers
all 16 full-attention layers in all three requests, with finite outputs.
The archived result hash is
`911c9906aed7ecf48d2673ce8267f2d27d2de22da65e55a0ec7ae36bbbff1926`.
The reference sequence also differs from the previous revision-2 process's
reference sequence. This prevents attributing the model difference solely to
the residual scaling; it does not turn this failing bracket into a pass.
Accumulation remains FP32. There is no production FP64 path, default-format
promotion, or transfer of the private 60-TFLOP/s prefill acceptance to this
small-query implementation.

## Publication sync with current main

The repair commit `52dcdc4d523d8e7331897f073bada070ad468cc1` is followed by a
non-rewriting merge of main `95205a2d9952813aa7469f63ff65b8f2813c027a`.
Main's independent sparse-page4 allocation-invariance changes are preserved.
The fresh CUDA 12.8 extension hash is
`76aa9a19f197fe805bb97b037f8a7934129ef0752b5c66922246c241a7e97b42`.

The post-sync build passes 89 kernel checks and 138 routing-policy checks.
All 200 retained real-input small-Q outputs are finite and bitwise identical
to the archived scaled-residual library. The replay result hash is
`eb187e7fea4c81e4f01bb5ebb3752c888e583b0d62fc3bd63dec37bf2cde73f9`.
This is a source-integration regression, not a fresh performance/model gate.

The additional `tests/kernels/test_sm70_qsa_page4_plan.py` run passes 30 native
planner checks. Eight integration cases fail at import because this isolated
source worktree lacks `vllm._C`; they do not reach numerical assertions. The
setup-failure log is retained, and the full 38-case suite is not reported as
passed. The archived 62-case sanitizer/model results above refer to the
pre-sync library, not this new binary.

The update remains in Draft PR #524 with the experimental flag off. The
within-process token-77 failure and missing whole-model/human-review gates
block merge and default promotion. No installed extension or service is
replaced. Private FP32 long-prefill v18/v37 still require separate clean-source
integration and admission; this small-Q patch does not publish those paths
or replace the older stable 60/61T implementation already in main.

## 2026-09-07 integration repair

Main `099d9841f542f1b71121b4aff49e3aa29053a489` adds E4M3 paired loads to
the dense q8 verifier. The explicit-row FP32 entry still admits 8-byte KV
strides, so inheriting that 16-byte loader would truncate physical offsets.
For example, a valid token stride of 264 would address byte 256. Commit
`4d889d0c1dde3589bced4211b984047a3d028609` merges main, retains both native
capability entries, and keeps the original 8-byte loader for `ROW_SEQLENS`.
The q8 E4M3 paired path and sparse-page4/E5M2 routes are preserved.

Three new padded-stride cases cover token/block/both padding with changing
row lengths in CUDA Graph replay. Each output is bitwise equal to its
contiguous-layout control. The rebuilt extension hash is
`b65ee698fa992b6ff933f495c3827390aef3376641464c399b27939a3f9132b3`.
It passes 98 kernel tests, 142 routing tests, and all 38 planner tests.
The planner's previously missing core dependency is now explicitly pinned
to the archived native-base library; no full native-base rebuild is claimed.
The three new stride cases also pass memcheck with zero errors.

All 200 actual-input outputs match the previous R6 extension bitwise. The
same-GPU, q5/page848, 20-warmup/100-ABBA speed ratios are
1.00255/1.00029/1.00064/0.99990 at approximately 8K/64K/128K/256K. This is
essentially unchanged operator speed, not model throughput admission.

The refreshed 128K model bracket uses the same frozen prompt tokens and
diagnostic launch settings, with the new Flash-V100 DSO and current Python
source. All 256 tokens match across candidate and both stable references;
all four TP ranks select the route and all 16 full-attention layers are
recorded on rank zero with finite outputs. Result SHA256:
`79001ddfa0517c2fc8d1cfb2f017d174f2e3648506905847da69f867f905c22b`.
This bounded pass does not explain the previous cohort's token-77 failure.
The alignment change leaves all 200 retained operator outputs unchanged, so
it is not asserted to be the cause of the model difference.

### Revision-3 state representation screen

A private prototype retains separate max/sum and unnormalized PV numerators.
For uniform attention over 128K/256K positions, with one E4M3 value 1.125 per
256 positions and all others 1, the exact output is `1 + 2^-11`. The old
normalized/LSE path rounds up; the prototype obeys FP16 round-to-nearest-even
and returns 1. Both choices have the same max-absolute/L2 error at this
midpoint. This is an operator regression, not a reconstruction of token 77.

On 200 real inputs, FP16 disagreements with the FP64-rounded reference fall
from 1872 to 1640: 125 groups improve, 45 worsen, and 30 tie. Aggregate L2 is
dominated by final FP16 rounding; this is not uniform per-input improvement.
Four 100-ABBA q5 speed ratios are 0.99964/1.00007/1.00039/1.00008, effectively
unchanged. Prototype DSO:
`df4a03dc6aafec7e1c6b2a47d4b9a5a706b98a98de9fc8a1542539ea8305661e`;
result: `cd7ac6efed7068101e30e7a8bd98749089b6177128fa4a790a875bf27f109311`.
The new representation requires its own integration and model admission;
the revision-2 model pass is not transferred to it.

The integrated revision-3 source is
`beb172ebd0278b7faa4118e2e0caf12047aa6cb2`, with extension
`8880b0405d6d7d212c4a738d45ab441815b6c948ab6e4e8ac5e1b83d754a9d49`.
It passes 102 kernel, 142 routing, and 38 planner checks; 69 Compute Sanitizer
cases pass with zero errors. All 200 real-input and five midpoint outputs
match the screened prototype bitwise. Integrated 100-ABBA q5 speed ratios
are 0.99928/1.00000/1.00041/1.00007. The partial kernel retains 128 registers
per thread and zero reported local memory; combine uses 32 registers and
656 bytes of static shared memory. This is not an end-to-end speed claim.

### Revision-3 model gate: 128K pass, boundary-256K fail

The frozen-driver run completes both brackets without relaxing the sampling
or token criterion. At 128K, candidate and both references agree on all 256
tokens. At 261888+256, the reference sequences agree, but the candidate first
differs at one-based token 126: references choose `speed`, candidate chooses
`benchmark`. The reference logprobs of these two alternatives tie at
-1.2728908062; the candidate favors `benchmark` by 0.046875. This near-tie
does not waive the deterministic gate, nor is it proof of broad semantic
degradation. Raw result SHA256:
`a94d356ce29273fba3a202428ea77c737e7f51bd5152be1885d3ed46b6fb700e`.

All four TP ranks hit the native route, and rank-zero records cover all 16
full-attention layers with finite outputs. The reference computes admitted
small-Q attention in PyTorch FP64 on identical E4M3 KV and returns FP16;
the rest of the model is not FP64. This cohort retains FP16 SSM state and
FP16 LM-head output. Instrumented run times are not production performance.

Boundary execution emits repeated shared-memory broadcast wait warnings,
then recovers and completes. Late q3 rows appear as speculative drafting
approaches the model length limit; they occur after the first divergence.
The original summarizer incorrectly required only q5 and failed after model
completion. A preserved copy and the corrected q2–8-aware summary distinguish
this reporting error from the actual token failure; token/finite checks are
unchanged. The earlier startup with a mismatched driver hash remains aborted
and is not included in these results.

The next counterfactual enables main's existing opt-in FP32 LM-head output
identically for both references and candidate. A synthetic local TP4 head
screen (62080 by 5120, rows 1/5, 100 ABBA samples) reduces selected-column
relative-L2 from about 2.0e-4 to 2.5e-6 versus FP64, with essentially unchanged
projection latency. It is not real token-126 replay or model admission.
Keep the failed FP16-head cohort; a changed output precision is a new
explicit contract, not a retrospective pass of the old one.

That FP32-head-only counterfactual also fails: at 128K, both references are
stable, but candidate token 191 is `validation` instead of `scrutiny`.
The reference top-two margin is about 0.018314; the candidate reverses it
by about 0.006470. All four ranks confirm FP32 head output, and all recorded
attention outputs are finite. The predeclared 128K gate stops expansion to
256K. Result SHA256:
`9571a0b77a2dfa161b8e8f0aa90097e3e6f9115b89c5eab06f295394113b043c`.
Thus improving the final projection alone does not resolve model admission.

The diagnostic launch explicitly retained FP16 GDN recurrent state. Current
main's `gated_delta_net_state_dtype` instead resolves `auto` SSM state to
FP32 for this model family; convolution state remains model-dtype. A new
counterfactual will retain FP32 logits and change only SSM state to FP32 on
both sides. The corresponding cache page size may change. This tests a
plausible amplification mechanism and the current-main default, but does
not erase either explicitly FP16-SSM failure or establish causality yet.

### Contiguous Q does not imply an aligned vector-load base

A continuous FP16 view can have a storage offset of 1/4/7 half elements,
giving a byte-address remainder of 2/8/14 modulo 16. The native Q feed reads
`uint4`, so contiguity alone did not satisfy its load contract. The native
host entry now clones only such unaligned Q views after entering the correct
device/stream context. Normal aligned Q retains its original Tensor and the
same kernel launch. The exceptional q8 copy is at most 24 KiB; this is not
a zero-overhead claim for deliberately unaligned callers. Kernel arithmetic,
KV loaders, and workspace ABI are unchanged. Offset-Q tests update input
values and row lengths during CUDA Graph replay and compare bitwise with
aligned controls. This separate safety issue is not a demonstrated cause
of the token-126 or token-191 failures.

Fresh extension
`c33a84443d42621060a90ff7ecdbe2af4910e7774317c44c74777a13d7b56aaf`
passes 105 kernel checks and six targeted Q/KV-alignment memcheck cases with
zero errors. All 200 aligned real-input outputs match revision 3 bitwise.
The four 100-ABBA q5 speed ratios are 1.01834/1.00065/1.00036/1.00000;
the short-point difference is not a new arithmetic speedup claim. Previous
142 policy/38 planner/69 broader memcheck results retain their earlier
artifact attribution. No model pass is transferred to this build.

### FP32 SSM plus FP32 logits: boundary-256K gate still fails

The completed counterfactual uses source `0c34be5d60` and the guarded-Q DSO
above. All four ranks confirm 48 allocated GDN SSM caches in FP32, FP16
convolution state, FP32 logits, and page size 1616. Both context brackets
complete with stable references. At 128K, all three 256-token sequences
match. At 261888+256, candidate token 26 is `-level`, versus `-` in the
references: the prose continues as "low-level kernel changes" rather than
"low-precision arithmetic". Both are coherent; this is a failed strict token
gate, not proof of general semantic collapse. The reference top-two margin
is 0.00760078; candidate reverses it by 0.00913811. Raw result SHA256:
`452a5a1845cae49b8aa26470a2fffacc4e8009a715ebb7625ec3b0e990c3a722`.

| SSM state | LM-head output | Page | 128K | 261888+256 |
|---|---|---:|---|---|
| FP16 | FP16 | 848 | pass | fail, token 126 |
| FP16 | FP32 | 848 | fail, token 191 | not run: 128K gate stopped expansion |
| FP32 | FP32 | 1616 | pass | fail, token 26 |

These are different arithmetic/cache-layout contracts; token positions do
not rank their overall quality. The old failures are not retrospectively
removed, and FP32 SSM alone is not established as their cause or cure.
The latest candidate has finite rank-zero outputs across all 16 attention
layers at both lengths. Its worst L2-to-FP16-rounding-floor ratios are
1.00002884 at 128K and 1.00001732 at boundary-256K. Very small attention error
still does not guarantee a stable final token. A targeted replay around the
first-divergence prefix is the next localization step; do not merely repeat
this unchanged full-model cohort or relax its token criterion.

Two preceding FP32-SSM diagnostic startups produced no accepted quality
results: a private dtype hook entered Dynamo with a non-Tensor capture query,
then an exact-class-name filter omitted the Qwen3.5 GDN subclass. Their logs
are retained. The accepted recording setup inspects real cache metadata
outside compiled forward and recognizes the inheritance chain; all dtype,
layer-count and token assertions remain in place.

At the boundary, the first reference waits while the strided GDN extension
is compiled on demand; it resumes after compilation, then candidate and final
reference complete. The observed command builds this worktree's
`flash_qla_sm70_gdn_strided` with system CUDA 12.0.140, independently of the
CUDA-12.8 Flash-V100 build. Strided GDN DSO SHA256:
`89337e7055cc8ba8f9bd972341f43010ce23a5a7cb991a84eb48e60bc5bbfaf9`.
Its main CUDA source hash is
`fd6389cef9f1b38df7e122e582221d74d9ae1fba377ac3bec0da047fd3d30af8`.
This supports a cold-build explanation of this run's tail wait, not an
attention deadlock. Prebuild/warm this dependency before performance work;
instrumented elapsed times remain excluded from production speed claims.
