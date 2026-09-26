# SM70 DFlash2 batch latency, 2026-09-26

Status: implemented in Draft PR #697. Focused operator and real-input numerical
checks pass. A same-process diagnostic A/B shows an incremental C8 benefit, but
fresh uninstrumented default-configuration services do not reproduce it. C8
rolling acceptance misses the two-percentage-point gate. Keep the PR in Draft;
no production speed gain, PRO win or completed C8 speed target is claimed.

## Contract and ownership

- Integration: `onecat/main`, base
  `e889919e2192fa36b25c922366526fa3fe62edbc`.
- Branch: `codex/v100-decode-round-20260926-111504`.
- Retained artifacts: `/data/minimax-h3/task-cache/sm70-decode-round-20260926`.
- Target: Qwen3.8-27B-NVFP4, TP4 V100-SXM2-32GB, FP16 execution,
  DFlash2 q7 (eight verifier rows/request), target E4M3 KV, draft automatic KV,
  Flash-V100, prefix caching, max context 262144, memory utilization 0.8.
- Dataset: the existing 2048-input/256-output shared-prefix set, SHA256
  `1cea3c5dbbd22fde40ad08db21fae1065b74f994009882a37e89c16b82b3580e`.
- Sampling: temperature 0.7, top-p 0.8, top-k 20, min-p 0, repetition penalty
  1, seed 20260923. Fixed output length is for speed only; natural EOS quality
  remains a separate gate.
- CUDA 12.8, Torch 2.10.0+cu128, Python 3.12. Private caches and port 18879.
  Other sessions' services must finish before reserving four GPUs.

## Implemented candidates

### Batch context and metadata graphs

The previous pre-sampling context projection and graph metadata refresh only
admitted B1/q8. Capture exact context shapes already supported by the full draft
query graphs. Replay the matching graph for uniform no-prefill verifier batches;
ragged, prefill and uncaptured shapes retain their existing path.

Projection uses original positions before the acceptance decision. Only the
later store consumes accepted slot mappings. Capture initializes mappings to
`PAD_SLOT_ID`, and metadata replay copies all captured rows, including padding,
to prevent stale request metadata when a batch shrinks. Dispatch depends on
query shape and backend capabilities, not a model name or weight quantization.

### Request-local exact sampler fallback

Previously a tied/ambiguous cutoff in one request made the entire batch use
full-vocabulary sampling. Keep the original cutoff guard and full sampler for
each affected request. Other requests retain compact sampling. Packing preserves
request-slot IDs, positions, local draft steps and seeds. Full-logit gathering
and the all-ambiguous fallback remain unchanged. Penalties, grammar, logprobs,
and synthetic rejection are still outside the compact path's contract.

This removes unnecessary dense sampling work; it does not claim to remove a
second LM-head projection (the old fallback already reused its logits).

### Grouped attention partition-weight reuse

Compute each partition's exponential once in the denominator loop and reuse it
across output dimensions. Keep the existing max/sum order, FP32 arithmetic,
output accumulation order, thread count and synchronization. This common combine
kernel also serves the legacy FP16-partial and precise FP32-partial routes;
the optimization is not conditioned on a model or KV quantization name.

## Validation to date

| Gate | Result |
| --- | --- |
| Draft/sampler CPU tests | 32 passed, 19 skipped (CUDA unavailable in CPU run) |
| Existing DFlash2/alignment/structured-output/compact-aux CPU regressions | 192 passed, 21 skipped |
| Mixed request fallback GPU tests | 25 passed; C2/C4/C8, ragged rows, heterogeneous sampling, changed seeds, FP32/FP64 Gumbel, 32768/248320 vocabularies |
| Deferred context writes under graph replay | 5 passed; B1/B2/B3/B4/B8, 16 replays each |
| Complete grouped-attention GPU suites | 91 passed on the normal rebuilt extension |
| Automatic defaults and explicit overrides | 48 passed, 6 GPU tests skipped in the CPU policy run |
| Uninstrumented TP4 C1/C2/C4/C8 and acceptance | Three runs each complete; C8 rolling -0.11%, pure -0.63%, rolling acceptance -2.50 pp: not accepted |
| Natural-completion quality | Both services: 14/16 correct, 15/16 natural stops; the same length-capped case remains |

The sampler GPU tests compare valid output token IDs and accepted lengths with
the unchanged full-vocabulary sampler. The context tests compare cache contents
with eager projection/store and verify rejected slots are untouched. These are
operator tests, not model-level acceptance evidence.

## Fresh ordinary-service default A/B

Both independently started services clear inherited acceleration variables and
use their own normal rebuilt package. No diagnostic worker extension or step
observer is loaded. Each rolling cell has three runs: C1/16, C2/24, C4/32 and
C8/48 requests. The pure C8 comparison uses three ordered six-wave runs over
the same 48 prompts. All speed requests return 256 actual token IDs; 2K prefix
cache hits are zero. The medians are:

| Metric | Control tok/s | Candidate tok/s | Change | Acceptance delta (pp) |
| --- | ---: | ---: | ---: | ---: |
| Rolling C1 | 241.739 | 242.487 | +0.31% | 0.000 |
| Rolling C2 | 265.057 | 295.439 | +11.46% | +6.585 |
| Rolling C4 | 326.295 | 327.664 | +0.42% | +0.441 |
| Rolling C8 | 385.783 | 385.374 | -0.11% | -2.497 |
| Pure C8, full 48 prompts | 680.644 | 676.332 | -0.63% | +0.393 |

The C2 acceptance change prevents assigning its whole improvement to lower
compute latency. C8 fails both the positive-speed gate and the rolling
acceptance limit; these results supersede the diagnostic gain as the promotion
decision. Cross-process pure outputs differ for 21/48 requests in every repeat,
although the earlier same-process ablation has identical outputs. This does
not identify the cause: initialization/dispatch and changed code must still be
isolated before claiming an exact-output benefit across independent services.

The repeated natural-EOS pair again scores 14/16 on both services, with the same
per-case correctness and finish reasons. Question 16 reaches the 4096-token
limit in both, so the strict 16-natural-stop assertion still fails. The 4K
repeat records 3296 cached tokens on each service; the 32K C2 smoke completes
on both. Neither smoke is a long-context speed or complete quality baseline.

Raw medians/individual runs: `results/default-service-comparison.json`.
Output/quality comparison: `results/default-output-audit.json`.
Loaded modules and hashes: `results/default-{control,candidate}-runtime.json`.

A subsequent diagnostic restart exports the actual TurboMind cache before
serving. Comparing its 39 common descriptors with the prior default diagnostic
process confirms initialization-dependent choices, not just different binary
hashes from additional warmed shapes. For example, the `[16, 62080, 5120]`
compressed GEMM changes split-K from 3 to 1; one rank's fused-activation
`[64, 8704, 5120]` entry changes from 2 to 7. Tile and swizzle changes also
occur. Split changes can change the floating-point reduction tree. These are
a confirmed comparison confound, not proof that they explain all 21 changed
outputs or the entire speed difference. The completed ordinary services did
not export their LUTs, so their exact choices cannot be reconstructed from
this later process. Retain `results/route-restart-comparison.json` and the
binary caches under `results/routes/`; a controlled replay is still needed.

## TP-rank GEMM plan coordination follow-up

The first cache-broadcast implementation still had two holes. It exported
after the FP8 warmup, so FP4 dense entries were measured independently, and the
TurboMind cache importer appended duplicate `(GemmDesc, batch)` entries. If a
compile warmup had already inserted a local entry, lookup kept that old entry
because it was first. The FP8/FP4 dense warmups now run in one coordinated
phase. Rank 0 measures AWQ/FP8/FP4 routes, synchronizes, exports the complete
cache, and every rank—including rank 0—imports it before replaying warmup. The
importer replaces an existing batch record with the imported launch plan. The
default path has no new environment switch; the existing FP8 coordination
switch remains an explicit rollback/debug override.

The clean post-change service log reports 14 rank-0 LUT records loaded on all
four TP ranks. An immediate route export contains 28 records on each rank; the
decoded files have the same SHA256
`8c4d2d00c6a2a22a6e65e213b21a0efecba703f71676213f83e164d803088d14`. Before
the importer replacement, rank 1/2 differed at the M16 LM-head swizzle and
rank 0 diverged later at several M64 entries. This is now a route-consistency
gate, not a speed claim.

The first ordinary no-instrumentation C8/48 screen after this change completed
three runs in one fresh service. Decode-capacity results were 379.243,
383.912 and 371.862 tok/s (median 379.243 tok/s), with acceptance 56.006%,
56.623% and 52.167% (median 56.006%). Against the retained three-run control
median 385.783 tok/s, this is -1.70% in this screen; the route fix has not yet
provided an accepted end-to-end speed gain. It does, however, remove the
previous cross-rank plan drift and keeps the PR in Draft pending a route-quality
selection step and a fresh paired control.

## Initial endpoint screening and real-input audit

The initial clean services used identical newly built common extensions, with
the candidate's rebuilt Flash-V100 combine change. This control already includes
the previously merged GEMM and batched draft-attention work. These numbers are
incremental changes, not the cumulative benefit of those earlier PRs.

| Batch | Control rolling tok/s | Candidate | Change | Acceptance delta (pp) |
| --- | ---: | ---: | ---: | ---: |
| C1 | 245.112 | 242.242 | -1.17% | 0.000 |
| C2 | 284.709 | 292.441 | +2.72% | -0.053 |
| C4 | 329.160 | 336.871 | +2.34% | 0.000 |
| C8 | 368.923 | 360.105 | -2.39% | -3.516 |

This is one rolling run per cell. C8 fails the two-percentage-point acceptance
screen and is not silently discarded. A first-eight-prompts-only pure-decode
screen gives 866.011 -> 907.292 tok/s (three repetitions), but cannot replace the
48-prompt acceptance set or excuse the rolling failure.

A task-only worker extension then compares actual model inputs within one
process. On each of four TP ranks, 64 batch-context projection steps have
bitwise-equal K/V versus eager, and 64 mixed compact/dense sampling steps have
identical valid sampled token IDs and accepted lengths versus the unchanged
full-vocabulary sampler. These ranks are replicas, not 256 independent workloads.
The audit adds GPU work and its throughput is not a performance result.

With the audit disabled, a same-process Python-path ablation retains identical
GEMM tactics and candidate attention in both variants. Two rolling C8/48 runs
average 371.156 -> 390.063 tok/s (+5.09%). An ordered six-wave pass over all 48
prompts gives pure decode 678.099 -> 714.614 tok/s (+5.38%), with exactly the
same accepted/drafted counters (50.0207% acceptance) and all 12,288 returned
token IDs identical across the 48 requests. This supports a real
incremental benefit but remains diagnostic evidence: the worker extension is
loaded, there is only one full-48 pure pass, and independently started clean
services still require repeated comparison.

The natural-output screen has the same per-case correctness/finish status on
both services (14 correct, 15 natural stops). Only three complete texts match
exactly, so it is a relative quality check, not model-level bitwise equivalence.

## Reconciling full-q8 steps with rolling request latency

The earlier joint draft-attention/GDN change reduced the complete C8/q8 step
57.810 -> 50.744 ms. At fixed emitted tokens per step, that implies +13.92%
step throughput. Its recorded rolling capacity improvement was only +3.95%.
An attribution based solely on those full-q8 samples was insufficient.

The new task-only observer records every scheduled step, including prefill and
partial batches, using asynchronous CUDA events without synchronizing each
step. For each request it reconstructs the interval from its first sampled
output to its last, then intersects that interval with each step. This measures
request waiting time, not a sum of four GPUs' kernel service or GPU utilization.
Warmup requests are excluded. Forty-eight requests emit all 12,288 tokens;
12,240 remain after excluding each first token, matching the client calculation.

| Contribution to total request decode time | Warm control | Candidate |
| --- | ---: | ---: |
| Complete C8/q8 steps, no prefill | 32.81% | 31.20% |
| Other no-prefill batches | 9.08% | 7.23% |
| Mixed prefill/decode steps | 58.11% | 61.57% |
| Other and between-step gaps | <0.01% | <0.01% |
| Client vs reconstructed duration discrepancy | 0.137% | 0.122% |

Mixed steps have a median around 540 ms; admission of several new prompts can
produce a 1.7-second step. CPU launch/compilation gaps inside a step are included
in its event interval. The first control pass also contains cold sampler/JIT
stalls; it is retained separately and the table uses a warmed control repeat.

Applying the earlier 12.22% full-q8 latency reduction to only a 31–33% share of
request waiting predicts roughly 4% rolling speed improvement. That is consistent
with the scale of the earlier +3.95% result. It is not an exact reconstruction
of the old pair: those runs did not retain the all-step ledger, and their batch
admission and accepted outputs differ. The current ledger's changing acceptance
also prevents treating its raw endpoint difference as a clean speed claim.

For example, holding all other work and emitted tokens fixed gives
`1 / (1 - 0.3281 * (1 - 50.744 / 57.810)) - 1 = 4.18%`, rather than
`57.810 / 50.744 - 1 = 13.92%`. Excluding a request's own TTFT does not exclude
time it later spends waiting for replacement requests' prefill. The rolling
metric is `C * sum(output tokens after first chunk) / sum(request decode time)`;
it is a client decode-capacity estimate, not an all-C-alive GPU token counter.

Within the current observed process, full C8/q8 median target forward remains
32.01 -> 32.13 ms, sampling 6.04 -> 5.60 ms, draft 7.71 -> 5.79 ms, and the
complete step 46.02 -> 44.32 ms. The GEMM dispatch cache hashes are identical
before/after on every rank. This is consistent with this PR changing context
submission and sampling rather than adding a new GEMM kernel.

## Automatic runtime selection

The default-configuration diagnostic and ordinary services explicitly clear inherited
`VLLM_*` tuning settings before launching. No `VLLM_SM70_*` acceleration exports
or `VLLM_USE_V2_MODEL_RUNNER=1` are required. Normal model, speculative decoding,
TP, context, memory, sampling and backend arguments still define the workload.
Runtime logs confirm common batch-GEMM defaults, V2 DFlash, joint draft attention,
and context/metadata graphs for existing captured shapes through 16 requests.
In the ordinary fresh-process pair, graph memory is 0.81 -> 0.88 GiB/rank.
With the same memory utilization 0.8, runtime profiling assigns 11.00 ->
11.45 GiB/rank to KV, reporting 1,009,312 -> 1,050,885 cache tokens. This
profiling-dependent difference is recorded, not claimed as a memory optimization.
The same-process diagnostic pair necessarily retains one fixed KV allocation.

The implementation introduces no extra environment switch. The common attention
combine is selected by its backend; batch context and exact request-local
sampling use existing capability checks and automatic configuration. Explicit
diagnostic overrides remain respected. These defaults are present in this PR's
source; it remains Draft until the endpoint/quality evidence is reviewed.

The workload is selected using ordinary service arguments, for example:

```bash
vllm serve <target-checkpoint> \
  --tensor-parallel-size 4 --dtype half \
  --max-model-len 262144 --gpu-memory-utilization 0.8 \
  --attention-backend FLASH_ATTN_V100 --kv-cache-dtype fp8_e4m3 \
  --enable-prefix-caching --mamba-cache-mode align \
  --max-num-seqs 16 --max-num-batched-tokens 8192 \
  --speculative-config '{"method":"dflash","model":"<draft-checkpoint>","num_speculative_tokens":7,"kv_cache_dtype":"auto","attention_backend":"FLASH_ATTN_V100","draft_sample_method":"probabilistic","enforce_eager":false}'
```

Target and draft paths, GPU visibility, memory budget and desired concurrency
remain deployment choices. No manual acceleration environment variables are
part of this command; explicit debug/rollback overrides still take precedence.

Retained artifact groups include `results/screen-comparison.json`,
`results/real-input-audit/`, `results/diag-ab-comparison.json`,
`results/all-step-ledger/`, `results/routes/`, and the default-service logs under
the task artifact root. Do not compare first-eight and full-48 prompt sets.

## Research-only microbenchmarks and rejected candidates

Research DSOs live only in the retained artifact directory. They are never
loaded by endpoint benchmarks or required by the installed package.

### GEMM scale lifetime

Actual TP-local weights, M64, service split-K/swizzle choices, paired eager and
CUDA Graph runs: grouped-scale register reuse reduced FP8 M64 registers from
146 to 130. Moving the next-stage fetch after current MMA reduced this to 129.
All compared eager/replay outputs were bitwise identical for three activation
amplitudes. However, the paired seven-shape layer-weighted GEMM sum only changed
from 16.7295 to 16.4402 ms (1.7%), and some layers regressed. This is not admitted
to production dispatch. In particular register reduction alone did not cross
the two-CTA residency threshold for the M64 tile.

The native reference measured 17.0636 ms; do not mix that with the paired
prototype baseline to inflate the compiler/lifetime change's benefit. The
research FP8 output tile also uses a different scheduler group axis; it is not
an exact replacement for that service tactic.

A follow-up occupancy candidate used launch bounds after shortening the scale
lifetimes. The M64/N128/K64 variants reached 128 registers, two resident CTAs and
zero compiler-reported spill stores/loads. They did not become faster: FP8 input
62.833 -> 62.981 us, QKV 60.365 -> 60.831 us, FP4 down 54.953 -> 54.717 us, all
bitwise equal to the original. The complete seven-shape paired sum regressed
17.3572 -> 17.6370 ms. This candidate is also excluded. Higher theoretical
occupancy alone is not sufficient evidence of higher useful throughput.

### Attention combine

The first 256-thread candidate normalized shared weights in a separate phase.
It improved C8 but regressed C1 by 8–14%; it was rejected.

The retained 512-thread candidate reuses unnormalized exponential weights with
no extra barrier. Synthetic partition inputs, 7 paired timing repetitions,
six changed-input graph replays/case, all bitwise identical:

| Batch | 2K old/new (us) | 32K old/new (us) |
| --- | --- | --- |
| 1 | 6.768 / 6.214 | 12.877 / 11.293 |
| 2 | 6.662 / 6.282 | 24.630 / 19.469 |
| 4 | 10.784 / 8.806 | 31.450 / 27.389 |
| 8 | 26.899 / 23.354 | 60.186 / 51.318 |

The tested contexts also include 128, 8192 and 262136. This is a combine-kernel
improvement, not whole attention or emitted-token speed. At 16 attention layers,
the C8 2K microbenchmark predicts only about 0.057 ms saved per verifier step.

## Remaining acceptance

Run matched clean control/candidate services, with no profiling hooks or private
DSO overrides. Separate rolling-request decode from all-C-alive windows without
new prefill, and include every verifier width. Compare acceptance and natural
output quality before admitting any change as a default. Re-run exact-source
trace attribution only after an uninstrumented endpoint benefit is established.
Retain the 4K cache-hit, 32K routing, 262K boundary and affected 35B-A3B AWQ/FP8
regression obligations. Saved PRO data are iteration context, not a fresh win.
