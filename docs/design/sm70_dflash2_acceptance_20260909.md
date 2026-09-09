# DFlash2 acceptance and quantization-independent schedules, 2026-09-09

The user requests acceptance of the current approximately 16.2/15.8-ms
combination, dataset-level decode speed, acceptance and quality, followed by
PR/main integration only if the gates pass. Quantization-independent
optimizations must be available to other weight formats. The original sub-15-ms
performance objective is not claimed achieved by this campaign.

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

Seed-zero mathematics is provisionally scored at GSM8K 30/32 and MATH500 31/32
in both arms, with identical failed questions. HumanEval/10 reaches the 16K cap
in both arms with no final answer. It is a quality failure at that cap and must
not receive credit from code present only in reasoning. A bounded whole-stack
control and a separate paired 32K-cap diagnostic are queued; the latter does
not replace or erase the original truncated samples. Other seeds, executable
code scores and the final acceptance verdict remain pending.

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

The FP8 model snapshot has all 66 indexed shards present. Independent control
and common-route candidate model jobs are queued, including the two speed
fixtures and 20 real quality cases. Other-quantization performance/quality is
not yet established. QPN2 compressed weight decoding remains NVFP4-specific.
No new route is enabled by default and PR #556 remains a draft pending gates.
