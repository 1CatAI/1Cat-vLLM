# DFlash2 acceptance and quantization-independent schedules, 2026-09-09

The user requests acceptance of the current approximately 16.2/15.8-ms
combination, dataset-level decode speed, acceptance and quality, followed by
PR/main integration only if the gates pass. Quantization-independent
optimizations must be available to other weight formats. The original sub-15-ms
performance objective is not claimed achieved by this campaign.

## Current capacity contract: 256K

The user subsequently required 256K context without the evaluation's artificial
16K generation cutoff. The server already used `--max-model-len 262144`; the
cutoff came from the client request's `max_tokens=16384`. The old seed-one job
was stopped deliberately and its partial records retained. Its exit code 143
is an authorized protocol transition, not a numerical failure.

The new `acceptance-256k` campaign retains the same prompts, seeds, weights,
sampling and four GPUs. Before natural generation, the client uses the server's
`/tokenize` renderer, verifies `max_model_len=262144`, and explicitly sets
`max_tokens=262144-prompt_tokens`. The input is never truncated and EOS remains
natural. A 135-token prompt therefore has a 262009-token output budget. The
actual response's prompt count must match the tokenizer result. There is no
separate 16K/32K generation cap. Reaching the model's total context limit is
still reported as a length stop, never presented as natural completion.

All three paired dataset launches are restarted with this policy; previous
truncated cases run first. FP8, public-route and whole-stack controls use the
same remaining-capacity policy. Speed requests also use the full remaining
capacity while retaining the canonical natural-EOS fixtures. One-token prefix
warmups and bounded teacher-forced operator diagnostics remain explicitly
excluded from natural-generation quality and performance results.

The corpus, capacity-policy checks, launch-time source archives and new results
are separate from `acceptance-16ms`. The results below describe the historical
16K-cap campaign and do not certify the new 256K-capacity campaign.

## Frozen evaluation

The running reference checkout remains detached at
`a7cc5ae305149d7a9ffdf42fb224dff34e5606aa` in
`/home/ymzx/桌面/1cat-vllm/worktrees/v100-quasar-dflash2-15ms-20260907-161715`.
Implementation continues on the same owned PR #556 branch in
`/home/ymzx/桌面/1cat-vllm/worktrees/v100-quasar-dflash2-acceptance-20260909`.
Main `b6d91d61ff` was merged into that branch without conflicts; this is not a
merge of the candidate into main. New code does not alter the running reference
checkout or its native libraries.

Artifacts are under
`/data/minimax-h3/task-cache/v100-quasar-dflash2-15ms-20260908/acceptance-16ms`.
`freeze.json` records runtime Python hashes, source, sampling and GPU ownership.
Each startup retains its own four-worker runtime-library inventory after
measurement. Physical GPUs 4–7, TP4/B1/q8, E4M3 target KV, FP32 logits/state,
FP16 draft transport, weights, T1/k20/p.95/xhigh and natural EOS remain fixed.

The first fresh independent startup, with five warmups and five measurements
per fixture/arm, records:

| Fixture | BV8 / BV2 complete-round median | BV2 pure decode | BV2 TTFT |
| --- | ---: | ---: | ---: |
| release1k | 16.454723 / 16.219526 ms | 183.607 token/s | 350.235 ms |
| MBPP28 | 16.314348 / 16.096967 ms | 302.494 token/s | 127.903 ms |

Both arms use the already-frozen attention/context/QPN2/sparse-selection stack;
this comparison isolates GDN BV2. It does not substitute for a whole-stack
quality comparison. All measured token IDs, natural EOS and acceptance match.
MBPP28 is slower than the previous 15.872776-ms observation; retain the new
samples rather than selecting only the earlier minimum.

## Dataset protocol and open findings

The immutable corpus hash is
`6756091e4061b0b092ceeac71e691a79b2015ef2030548f74f7cc7cc2d1cb5ed`.
It contains 32 prompts each from GSM8K, MATH500, HumanEval and MBPP, 16 from
the existing stratified LiveCodeBench v6 subset, and four JSON/tool fixtures.
Seeds are 0, 1 and 2. These are subset results, not full benchmark scores.
Each paired case has a separate one-token prefix warmup per arm, excluded from
quality/performance scoring. Measured generations have a 16384-token cap and
do not ignore EOS. Pair order alternates. All responses, including failures,
remain retained.

Report actual accepted/proposed draft tokens, accepted draft tokens per round,
emitted tokens per round, position-specific acceptance, request-average complete
rounds, TTFT and pure decode separately. Aggregate decode is
`sum(output_tokens - 1) / sum(engine_decode_seconds)`; stream chunk intervals
are transport observations rather than instrumented GPU-round percentiles.

Seed zero completes all 148 pairs with identical token IDs, acceptance counts
(including per-position counts), finish reasons and semantic tool calls. Its
unprofiled aggregate decode measurements are:

| Subset | Cases | BV8 / BV2 pure decode (token/s) | Accepted / proposed | Accepted drafts / emitted tokens per round |
| --- | ---: | ---: | ---: | ---: |
| GSM8K | 32 | 309.791 / 316.749 | 57.9715% | 4.058008 / 5.058848 |
| MATH500 | 32 | 243.960 / 246.660 | 48.2514% | 3.377595 / 4.377883 |
| HumanEval | 32 | 225.650 / 229.195 | 42.8062% | 2.996437 / 3.996309 |
| MBPP | 32 | 231.882 / 234.678 | 43.3363% | 3.033541 / 4.033895 |
| LiveCodeBench v6 | 16 | 175.055 / 177.259 | 32.9424% | 2.305965 / 3.305909 |
| JSON/tool fixtures | 4 | 214.813 / 217.170 | 42.0974% | 2.824121 / 3.824121 |

These measurements isolate the GDN change inside the shared candidate stack.
They are not complete-stack acceptance evidence. Seed-zero mathematics is
provisionally scored at GSM8K 30/32 and MATH500 31/32 in both arms. Six cases
reach the 16K cap without final content: HumanEval/10 and LiveCodeBench subset
indices 21, 64, 93, 131 and 162. They remain failures at that cap, with no credit
from reasoning-only code. Disabling all performance candidates reproduces
identical tokens for the three mathematics errors and HumanEval/10. A separate
32K-cap diagnostic makes HumanEval/10 end naturally at 21162 tokens in both
arms, again with identical tokens. It does not replace the original truncated
sample. The five LiveCodeBench failures have a separate whole-stack check.

The retained LiveCodeBench wrapper incorrectly treated negative error codes as
truthy passes. The campaign's private corrected wrapper follows official
`lcb_runner/evaluation/pass_k_utils.py` at commit
`28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`: every test result must be greater
than zero. Eight synthetic sentinel checks pass. The original wrapper, both
source hashes and the correction are recorded; no affected score is credited.
Original HumanEval/MBPP assertion scores and EvalPlus scores are separate, with
EvalPlus's eligible-subset denominator reported explicitly. Other seeds,
executable scores and the final acceptance verdict remain pending.

A separate same-startup control/candidate/control diagnostic is queued for
fixed prefixes, all 48 GDN layers and all 64 target layer observations. It
retains full vocabulary logits and lossless SHA256 fingerprints of the other
tensor bytes to bound disk consumption. It is not a timing service and is not
yet quality evidence. Long-context admission remains open.

## Common schedules

`sm70_dflash2_common_candidate_route.py` provides an explicit manifest-based
entry point for GDN value tiling, context/probe overlap, grouped E4M3 attention
and exact sparse candidate gathering. It loads no QPN2 projection library and
does not require a target quantization name. Existing guards continue to require
the audited shapes, activation/state types and applicable graph path.
Native dependencies are hashed before installation. Unsupported calls retain
the existing operator.

`sm70_dflash2_qpn2_candidate_route.py` separately packages optional cap64 column
projections and TP4 row publication. Its representation and dimension guards
match the audited QPN2 calls. A model with other weight formats can install the
common routes without importing or loading any QPN2 projection implementation.
The combined public entry point is queued for an NVFP4 comparison with the
frozen private installer before admission.

The integration branch includes dependency PR #563 at
`b4334fc028593942d658854e94461546c40ee21b`. It preserves prefill classification
for initial one-token requests in speculative GDN, preventing reads from a
previous request's recycled state. Its 32 focused CPU metadata tests pass;
GPU singleton/history-reuse diagnostics are queued on the fixed integration
source. This dependency is not yet merged into main, and the unsafe original
singleton case is not rerun on the unpatched frozen evaluation checkout.

The FP8 model snapshot has all 66 indexed shards present. Independent control
and common-route candidate model jobs are queued, including the two speed
fixtures and 20 real quality cases. Other-quantization performance/quality is
not yet established. QPN2 compressed weight decoding remains NVFP4-specific.
No new route is enabled by default and PR #556 remains a draft pending gates.
