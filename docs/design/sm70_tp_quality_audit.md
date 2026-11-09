# SM70 local-layout acceleration: paired output audit

## Workload

2026-09-21, PR666. Source16b2afcd1b for the primary comparison;
316e04a1ec adds the Mamba tail-state repair;907c210b03 repairs QPN2 scales.
V100-SXM2-32GB,
GPU0-3, TP4, Python3.12, Torch2.10.0+cu128, CUDA12.8. Model:
QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4, DFlash2 draft revision
`dedf8df68adfb1afeaf7b7480c0a0243108177b4`, seven draft tokens,
target E4M3 KV and draft FP16 KV. Max context262144, chunk8192, maxseq32,
memory0.85, block2048, Mamba block8192/align, prefix caching enabled.
Both target and draft retain CUDA Graphs; no eager benchmark is used.

Primary quality uses the same deployment sampling in both arms:
temperature0.7/top_p0.8/top_k20, thinking enabled, seed20260921+item index,
concurrency4 and max output65536. This is a controlled serving comparison,
not a claim to use the checkpoint's generation_config defaults. Each of
GSM8K, MATH500 and sanitized MBPP has the same32 sampled items as the earlier
audit. Requests, dataset hashes, full responses and scores are retained.
MBPP tests stay out of prompts and generated code runs in the existing
Landlock/seccomp sandbox. All96 responses in each arm finish naturally;
there are no retries replacing primary scores and no truncated answers.

ON uses unmodified defaults. OFF explicitly sets these to0:

```text
VLLM_FLASH_V100_PREFILL_D256_GQA_ARCH_128K_EXPERIMENTAL
VLLM_FLASH_V100_PREFILL_DENSE_SPLITKV3
VLLM_FLASH_V100_DECODE_USE_XQA
VLLM_FLASH_V100_E4M3_BATCH_XQA
VLLM_FLASH_V100_E4M3_SCALAR_FAST
VLLM_FLASH_V100_E4M3_GROUPED_FP32
VLLM_FLASH_V100_DFLASH2_GROUPED_VERIFY
VLLM_SM70_E4M3_LONG_ATTENTION
VLLM_SM70_NVFP4_QPN2
VLLM_SM70_NVFP4_QPN2_PREFILL
VLLM_SM70_FP8_QPN8
VLLM_SM70_DFLASH2_QPN8_RERANK
```

The control disables related pre-existing acceleration as well as the newly
generalized paths. FP32 verification logits, model/KV quantization, sampling,
collectives, DFlash and graph policy remain unchanged. ON worker logs show
QK+PV FP32 dispatch, QPN2, FP32 candidate rerank and long-attention graph
capture. OFF worker configuration records the explicit rollbacks and retains
the ordinary scalar/dense paths.

## Paired results

| Dataset | OFF | ON | OFF-only correct | ON-only correct |
| --- | ---: | ---: | --- | --- |
| GSM8K | 30/32 | 30/32 | None | None |
| MATH500 | 32/32 | 32/32 | None | None |
| MBPP | 26/32 | 25/32 | indices11,144 | index73 |
| Total | 88/96 | 87/96 | 2 | 1 |

All paired request bodies match. Exact answer text matches on20/96 items;
sampling and numerical reduction differences do change generated text. The
one-item net score difference is not evidence of statistical equivalence or
of a systematic quality loss. Keep the measured difference visible.
GSM8K item830 has the correct1128 minutes written as18h48m in both arms;
the strict extractor scores it wrong. No adjusted score replaces the table.

A separate C1/temperature0 diagnostic on the three discordant MBPP items
scores OFF1/3 and ON0/3: both fail11/144; only OFF passes73. These results
also do not establish token-level equivalence. The ON diagnostic uses the
independently built wheel and proposal logging; it is not a speed benchmark.
The main audit and the diagnostics are retained separately, not cherry-picked.

Natural-EOS retrieval at32768,131072 and256000 input tokens returns identical
correct text in both arms:173,284,396,853. Every request resets prefix cache.
These checks cover long prefill and retrieval, not a comprehensive long-context
reasoning benchmark. Prior operator checks independently cover local KV heads
1/2/4, FP64 selected-row oracles and changed-input CUDA Graph replay.

## Capacity-tail failure and repair

At262128 input tokens with16 output tokens available, both arms originally
return the same wrong text, `173\n284\n3913`, then EOS. The clean wheel
reproduces it. At262112 input with32 slots available, both return the correct
16-token answer. A32768-input/16-output control also passes, separating the
context-capacity transition from the requested output budget alone.

The verifier moves from q8/q7 to a single-token decode tail. Speculative GDN
stores several candidate recurrent states, while ordinary decode reads the
canonical state. MRV2 align precopy previously returned early whenever the
source and destination Mamba columns matched. It therefore left the accepted
temporal slot and the shifted convolution window unmaterialized for that tail.

The repair copies the accepted state to the canonical position for q1 after
multiple accepted tokens, even within the same block, and resets its selector
to1. It preserves normal speculative q8 behavior, graph execution and the full
context capacity. Eleven GPU tests pass, including same-block graph replay
with accepted counts2/6/8 and q1/q8 controls. The original262128-input request
now returns all four correct values, naturally finishing with16 output tokens
and262144 total tokens. This repair is separate from FlashAttention precision;
the failure is not evidence of FP32 accumulation overflow.

## Projection scale-rounding investigation

Promotion is paused at the user's request. No further wheel builds are part
of this investigation; the changed CUDA translation unit is rebuilt in the
owned native build and installed into the owned source tree.

Three temperature-zero probes retain the original prompt and compare an
identical reasoning prefix. Full trajectories first diverge at generated
tokens212 and389 for MBPP indices73 and11. Forcing the shared prefix as input
gives the same top token in ON and OFF for all three cases. The previously
observed divergence of item144 is not stable across these fresh runs.
With attention acceleration still ON and all four projection acceleration
switches OFF, all three short token trajectories match the OFF control.
These bounded probes intentionally stop after the divergence point; they
are diagnostics, not completed-answer quality scores.

`nvfp4_qpn2_sm70.cu` rounded `global_scale * 16384` to FP16 before multiplying
the E4M3 group scale. The ordinary W4A16 path multiplies the original FP32
global scale by the exactly represented group scale, then rounds to FP16.
The different order changes the effective model weights. In12 real tensors,
13-59% of group scales differ; this exceeds a mere reordering of GEMM sums.
The original exponent rebias can also overflow for otherwise finite weights,
or skip the reference's subnormal rounding. The new basis-vector regression
fails on the old binary and passes all24 cases after repair, covering M8/16/32
and separate/shared layouts. All cases use changed-input CUDA Graph replay.

The repair rounds the combined global/group scale once and removes the FP4
exponent bias from the code instead of scaling up the group scale. It retains
the QPN2 layout, Tensor Core FP16 operands and FP32 accumulation for both
ordinary and fused-gate kernels. Six real projection families plus fused
gate/up, tested at M1/8/32, reduce relative error against the independent
FP64 effective-weight oracle from roughly0.00028-0.00061 to0.00020-0.00023.
For fused gate/up, the rounded-reference error falls from0.00074-0.00097 to
0-0.000004. Raw graph timings are retained; some kernels slow by up to15%,
so this is not a claim of unchanged end-to-end speed.

An additional same-input comparison evaluates both repaired QPN2 and ordinary
TurboMind against that FP64 oracle for six real projection families at M8.
Both have virtually identical relative errors,0.000199-0.000215. Against the
correctly rounded FP16 reference, QPN2 differs on10/6144 sampled elements and
TurboMind on20/6144. Small reduction-order differences remain; the repaired
QPN2 is not using lower-precision effective weights than the control.

After the repair, item11's first divergence disappears; item73 still differs.
At its first differing position the OFF top-two logit gap is0.0040, versus
0.0142 in the repaired candidate. The residual difference must not be hidden
by claiming token equivalence. The repaired32-item MBPP run completes without
truncation and scores23/32, versus25/32 in the original ON run and26/32 in
the original OFF run. A fresh matched OFF control on the repaired source
scores25/32; all64 paired responses finish naturally, with identical request
bodies. Only indices7 and144 are OFF-only correct in this new pair. The
earlier OFF-only item11 is now wrong in both modes, demonstrating that even
the control's item-level result is not stable across these runs. This does
not prove that every remaining difference is harmless or permit replacing
the original scores. The output-quality promotion gate remains open.
The original GSM8K/MATH500 scores are not new validation of this changed kernel.

A final natural-EOS, C1/temperature0 check on indices7 and73 scores OFF0/2
and repaired ON1/2. Item73 now returns the required concatenation with the
repaired path; item7 still uses the wrong Woodall formula in both modes.
This diagnostic reverses the earlier direction on item73, so it must not
be presented as a stable, one-way regression or used to replace the primary
sampled scores. Three cold-prefix repeats of its first1024 tokens in the
repaired ON process produce identical output hashes. The bounded repeat
does not reproduce request-state contamination; cross-process differences
and complete output-score parity are not resolved by that check alone.

The failed-answer audit also records limitations of this zero-shot MBPP
protocol: `is_octagonal` and `is_num_decagonal` invite a predicate despite
the request for the nth value; tuple conversion leaves its output format
unspecified; the directrix reference uses multiplication where the ordinary
parabola formula requires division. These do not excuse the ON/OFF difference
or change any score. Generated-code tests and all original failures remain.

Artifacts: `diag-{on,off}-r14-divergence.json`,
`diag-attn-only-r15-divergence.json`, `diag-fixed-r16-divergence.json`,
`qpn2-scale-rounding-audit-r15.json`, `qpn2-scale-{before,after}-test.log`,
`{before,after}-fix-r16-qpn2-real.json`, and
`audit-fixed-r16-mbpp32{.jsonl,-summary.json}`. Additional records:
`qpn2-scale-full-test-r17.log`, `qpn2-vs-tm-reference-r17.json`,
`audit-fixed-off-r17-mbpp32{.jsonl,-summary.json}` and
`qpn2-fix-mbpp-paired-r17.json`. The last operator-only correctness checks
overlap the OFF quality run; none of its timings is used as speed evidence.
The source-native `_C` hash
is recorded in `qpn2-scale-fix-native-r16.json`.
Final diagnostics: `postfix-greedy-{on,off}-r18-mbpp32.jsonl` and
`greedy-repro-r19.json`.

## Fixed-prefix localization after the scale repair

The next diagnostic fixes both the token tape and the target's q8 verification
schedule, so a changed draft proposal cannot explain a different target input.
This is a forced-accept diagnostic, not a quality score or speed measurement.
The same source-native libraries are used in both fresh TP4 processes; only
QPN2 dispatch changes. The 93-token MBPP73 prompt and 225 observed target
positions use normal CUDA Graph execution.

In the original capture (`r21`), all 256 target projections were checked on
real activations against independently decoded checkpoint weights and FP64
matrix products. Every sampled result is finite. Maximum relative L2 error
is 0.000230 for QPN2 and 0.000240 for ordinary TurboMind at the first q8 step.
Each arm is scored against its own captured input; these are not same-input
element-count comparisons. A separate actual-hidden-state LM-head audit
checks 900 vocabulary-shard rows: zero missing exact top-21 candidates and
zero local top-1 changes. Neither audit establishes full-model equivalence.

The expanded `r23` capture localizes the first prefill discrepancy before
layer 0's input projection. Seven FP16 normalized inputs differ on rank 0;
the other ranks have identical inputs and identical GDN chunk outputs across
the two processes. Within one process, ranks also disagree on those seven
normalization elements. This is the previously documented Inductor RMSNorm
reduction-order drift, which had an opt-in repair but no serving default.
The source already preserves FP32 accumulation; selecting a reduction by
timing independently on each rank makes that arithmetic order unstable.

The DFlash2 serving profile now selects the existing fixed 8192-element,
16-warp Gemma reduction for its supported no-residual/FP16-residual geometry.
It preserves the established FP32-residual path and CUDA Graph execution,
has no TP-count or weight-quantization gate, and honors an explicit zero
rollback. In `r24`, with the fixed reduction in both arms, all four ranks'
captured prefill projections, first-layer GDN intermediates and final hidden
states are bitwise equal. Across the 225 fixed positions there are no target
top-1 changes. Decode hidden states still differ by up to 0.01134 relative L2
because the matrix kernels retain different FP32 summation orders. This
remaining difference is recorded, not declared harmless by the prefill fix.

The source review also finds an independent heterogeneous-batch sampling
bug: the compact target cutoff guard uses only the first request's
temperature/top-p. Later requests can therefore miss a split top-p tie and
retain a different vocabulary support. The guard now expands each request's
parameters using its packed logit-row offsets. The original caller fails
the reordered-request/variable-row regression; the repaired guard passes.
All original quality requests used identical sampling parameters, so this
bug is not offered as the cause of their score difference.

Artifacts: `projection-capture-r{21,24}-comparison.json`,
`gdn-prefill-comparison-r{23,24}.json`,
`capture-qpn2-{on,off}-step1-oracle-r21.json`,
`capture-qpn2-on-step0-oracle-r21.json`, `captured-lm-head-r22.json`,
`mixed-cutoff-original-caller-r25.log`, `mixed-cutoff-after-r25.log`,
and `norm-tests-r25.log` (20 V100 tests, including changed-input graph replay
and an actual Inductor fullgraph). Captured tensors and compiler caches are
retained in the task cache. A fresh, uninstrumented MBPP32 ON/OFF pair,
standard vLLM bench, and the 262128+16 boundary gate follow these changes.
Their results, rather than forced-token diagnostics, determine promotion.

### Uninstrumented fixed-normalization pair

The r26/r27 pair uses commit `28389e1fa5`, the unchanged source-native
libraries, and the same deployment contract described above. Both arms
select the fixed normalization by default. All 32 request bodies are identical
between arms. ON scores **24/32**, OFF **25/32**; all 64 responses stop
naturally. The only OFF-only correct item is 144; there are no ON-only correct
items. ON emits 185389 tokens in total (maximum 41976 for one answer), OFF
161292 (maximum 28229). The former OFF-only item 7 now passes in both arms.
This is still an unresolved quality difference, not a passed promotion gate.

Item 144's two answers both contain the correct nth-decagonal formula
`n * (4*n - 3)`. ON assigns the required `is_num_decagonal` name to membership
testing and places the nth computation in another helper; OFF assigns the
required name to nth computation. This explains the assertion failure but
does not identify the low-level cause of the generation branch. Retain the
original score. A separate full natural-EOS, C1/temperature-zero comparison
is used to investigate this remaining case; it must not replace the C4
sampled results with a more favorable diagnostic.

That complete greedy diagnostic finishes naturally in both modes: ON passes
after 13413 output tokens; OFF fails after 14878, assigning the required name
to the predicate. Request bodies match, with C1 and temperature zero in both
arms. The direction reverses relative to the primary sampled pair. This does
not establish a stable one-way accuracy loss, but also does not erase the
primary 24/25 result or establish global equivalence. Do not repeat sampled
sweeps solely to obtain a favorable score; the next numerical investigation
needs a concrete same-input counterexample beyond the validated rounding floor.

An independent audit of item 84 also confirms that its generated answers
satisfy the stated integer equation, while the tests demand specific tuples
among multiple valid solutions. The sandboxed equation check passes; the
original strict score remains a failure. See `mbpp84-semantic-audit-r26.json`.

Standard `vllm bench serve`, random 2048-token input/256-token output, with
DFlash2 and normal CUDA Graphs, records:

| Mode | Concurrency | Median TTFT | Median TPOT | Full-request output throughput |
| --- | ---: | ---: | ---: | ---: |
| OFF | 1 | 0.6831 s | 5.7054 ms | 119.6903 tok/s |
| ON | 1 | 0.5927 s | 3.3647 ms | 176.3946 tok/s |
| OFF | 4 | 2.6441 s | 12.5228 ms | 160.0622 tok/s |
| ON | 4 | 2.2109 s | 11.0062 ms | 189.2527 tok/s |

These are fixed-output synthetic speed checks, separate from natural answer
quality. DFlash streaming events can contain multiple tokens: do not invert
the raw inter-event latency and call it per-token decode throughput.
Each arm also passes the cold 262128-input/16-output retrieval-and-sum
boundary, returning `173 / 284 / 396 / 853`, with natural EOS at 262144 total
tokens. ON TTFT is 132.3341 s (1980.8 input tok/s by input/TTFT); OFF TTFT is
174.3433 s (1503.5 input tok/s). These are measured first long requests in
their respective processes, without using prefix-cache hits as prefill speed.
The 16-token answer is not a long-context decode-throughput benchmark.

Artifacts: `norm-quality-pair-r27.json`,
`audit-norm-{on-r26,off-r27}-mbpp32{.jsonl,-summary.json}`,
the corresponding `-c{1,4}.json` and `-long-262128.json`,
`quality-repair-source-r26.json`, and `mbpp144-r27-analysis.json`.
The complete deterministic diagnostic is retained in
`greedy-fixed-{off-r28,on-r29}-mbpp32.jsonl` and `greedy-fixed-pair-r29.json`.

## Reproduction and scope

Raw artifacts and launch receipts are retained outside Git in the task artifact
directory recorded in the local handoff.
Primary files: `audit-{on,off}-r10-{gsm,math500,mbpp}32.jsonl`,
`audit-r10-comparison.json`, `audit-switches.json` and the corresponding
`*-launch.json`/`*-server.log`. Diagnostic prefixes:
`discordant-off-r10`, `discordant-on-wheel-r11`, `boundary-on-tailfix-r12`.
The retained `launch_audit_endpoint.py`, `run_audit_suite.py`,
`run_boundary_quality.py` and `compare_audit.py` contain the exact commands.

The normal source build produces a wheel including Flash-V100, FA2 and
FlashQLA. All15 native extensions have no RPATH/RUNPATH; a fresh process with
no private library overrides imports all three packages from the extracted
wheel and reports32-bit QK/PV accumulation and runtime-page revision1.
The optional Rust frontend is omitted because Rust is unavailable; Python
serving is used. Package hashes and import paths are in
`package-manifest-r10.json` and `package-import-r10.json`.
These packaging checks predate907c210b03 and are not a wheel containing the
QPN2 scale repair. Per the user's direction, that repair uses the source
native extension and no new wheel is built.

Candidate dispatch follows supported local layout, with explicit rollback
switches. Output-quality promotion remains pending. The audit complements the existing
TP1/TP2/TP4 operator/serving evidence; it does not prove every model, weight
quantization or memory configuration. Shared QPN2 weights remain opt-in.
Full QK/PV FP32 remains70-71T;75T recovery and the separate35B migration
baselines remain open. See [coverage and speed evidence](sm70_tp_shape_coverage.md).
