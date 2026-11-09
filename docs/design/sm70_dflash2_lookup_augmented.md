# SM70 DFlash2 lookup-augmented block drafting

## Purpose and provenance

This change ports lookup-augmented block drafting (LABD) to the public 1Cat
MRV2 DFlash2 path. The neural checkpoint remains a block-8 model (one anchor
plus seven trained draft positions), while the target may verify a q16 block
when the request is demonstrably reproducing its own context. Normal and
low-confidence traffic stays on q8.

The algorithm is adapted from
[`syv-ai/qwen38-27b-rtx3090`](https://github.com/syv-ai/qwen38-27b-rtx3090)
revision `69ba4d0688c6ae76cb9d3c4a5c3b36445e1b040c`, specifically the
Apache-2.0 `patches/dflash2-lookup-drafting.patch`. The implementation is
rebased onto 1Cat's Model Runner V2, sparse probabilistic rejection sampler,
SM70 selector, hybrid KV manager, and Flash-V100 verifier rather than applying
the patch mechanically. Development is tracked in public Draft PR #355.

## Runtime contract

- `method=dflash`, `ngram_assist=true`, and `num_speculative_tokens=15`
  activate LABD for a selector-capable DFlash2 checkpoint whose trained block
  is smaller than the configured verifier.
- The DFlash2 model, grouped convolution, selector lattice, and draft KV retain
  the checkpoint-native seven draft positions. Lookup owns at most the eight
  additional target positions.
- One Triton program per active request scans the authoritative UVA int32 token
  history. It selects the longest suffix match and breaks ties by recency;
  overlapping matches are legal.
- The lookup proposal is fused with the neural proposal. Filled probabilistic
  rows become point masses in the existing sparse draft-logit cache, including
  complete erase metadata for the following step. When a weak match relies on
  `VLLM_DFLASH2_LOOKUP_AGREE > 0`, the agreeing neural prefix retains its
  original proposal scores. Only subsequent positions become point masses:
  conditioning the prefix's own correction on its sampled tokens biases the
  target distribution. Strong history-only matches and the default agreement
  threshold of zero retain the existing behavior. See the
  [numerical audit](sm70_dflash2_fastpath_numerics.md) for the counterexample.
- Structured-output and prefill batches retain q8 and do not use lookup.
- The host controller enters q16 only after two consecutive strong copy
  signals. B1 may coast for three steps; batches larger than one never keep
  sticky state across a miss.
- Both q8 and q16 target graphs are captured. The DFlash draft graph remains
  q8 because that is the checkpoint contract.

The asynchronous scheduler cannot carry a worker-selected proposal count back
to the scheduler: it pads every step to the configured maximum. Adaptive LABD
therefore disables asynchronous scheduling by default. An explicit async
request fails at startup instead of silently becoming always-q16. Setting
`VLLM_DFLASH2_LOOKUP_ADAPTIVE=0` deliberately selects fixed-width q16 and may
retain async scheduling.

## Cache and Flash-V100 integration

The wider verifier changes the Mamba-align hybrid page from 1,648/3,296 to
1,728/3,456 elements. The hybrid KV reservation uses the target verification
width while all DFlash model allocations use the trained width. Flash-V100's
grouped verifier accepts all four page layouts. The existing CUDA extension's
runtime-stride fallback handles 1,728/3,456 exactly; a fixed specialization is
only justified if the paired microbenchmark shows a material gap.

## Evidence, 2026-08-28

CPU configuration, routing, controller, graph-layout, and policy tests pass.
The controller test proves the sequence q8, q8, q16 for two consecutive strong
signals and proves that a multi-request miss cannot coast on q16. The config
tests prove default async disablement and a targeted error for explicitly
enabled async scheduling.

On Tesla V100-SXM2-32GB, the new 1,728/3,456 page layouts match an FP32
reference for both q8 and q16 (four strict cases). The six-case GPU LABD suite
passes suffix hit/miss/overlap, request eligibility, 7-to-15 fusion, controller
signal boundaries, and sparse point-mass cache invariants.

The standalone lookup-plus-fusion CUDA Graph cost is:

| Context | B1 | B2 | B4 | B8 |
| ---: | ---: | ---: | ---: | ---: |
| 1K | 0.0111 ms | 0.0131 ms | 0.0139 ms | 0.0152 ms |
| 32K | 0.1210 ms | 0.1254 ms | 0.1274 ms | 0.1351 ms |
| 64K | 0.2336 ms | 0.2359 ms | 0.2317 ms | 0.2357 ms |
| 128K | 0.4182 ms | 0.4156 ms | 0.4124 ms | 0.4705 ms |

The external benchmark artifact is `lookup-uva-sm70.json`. The nearly flat
B1-to-B8 slope confirms that the lookup kernel exposes request parallelism
rather than serializing the batch.

The runtime-stride 3,456-page grouped-verifier CUDA Graph measurements are:

| Query/context | Grouped | Independent XQA | Speedup |
| --- | ---: | ---: | ---: |
| q8 / 1K | 0.0320 ms | 0.0564 ms | 1.76x |
| q8 / 32K | 0.1240 ms | 0.9098 ms | 7.34x |
| q8 / 128K | 0.4095 ms | 2.9715 ms | 7.26x |
| q16 / 1K | 0.0273 ms | 0.0479 ms | 1.76x |
| q16 / 32K | 0.2167 ms | 1.5883 ms | 7.33x |
| q16 / 128K | 0.7855 ms | 5.6914 ms | 7.25x |

The 32K/128K movement versus the old fixed 3,296 page tracks the 4.85% larger
page closely. A new fixed 3,456 specialization would recover only the earlier
address-path margin (roughly 1--2%), so it is deferred until a full-model trace
shows that margin on the critical path.

An initial q15 route smoke produced coherent lookup proposals and positions
beyond the neural seven-token block, but it ran with async scheduling. Its
`75` drafts over five rounds are proof that async pinned every round to 15, not
valid evidence for the adaptive policy or throughput. That run is retained as
a route/correctness diagnostic only.

The first synchronous end-to-end pair uses the production NVFP4 target, TP4,
E5M2 target KV, FP16 draft KV, prefix caching, Mamba align, a 4,096-token
prefill budget, probabilistic sampling (`temperature=1`, `top_p=0.95`,
`top_k=20`), and CUDA Graphs. Each ordinary-path number below is the second
256-token request in one engine process, after inference-time Triton JIT has
already completed:

| Workload | Verifier behavior | Mean acceptance length | Steady decode | TPOT |
| --- | --- | ---: | ---: | ---: |
| ordinary async rate-limiter task, q7 baseline | fixed q8 | 3.048 | 168.52 tok/s | 5.934 ms |
| same task, adaptive LABD | q8 throughout | 3.122 | 170.98 tok/s | 5.849 ms |
| repeated-context continuation, adaptive LABD | q8 then sustained q16 | mixed-run 4.896 | 316.27 tok/s | 3.162 ms |

The normal sample is 1.46% faster than the paired q7 sample, so this route did
not expose a normal-request throughput penalty. It is only one prompt and is
not a promotion-level confidence interval. The mixed normal/copy counters
contain 106 measured rounds and 862 proposals, which implies 91 q8 rounds and
15 q16 rounds. Positions 8--15 were accepted in 14 of those q16 rounds. The
controller log records the normal request as ineligible and changes to q16
only after the copy request produces the required consecutive strong signals.

The paired runs use probabilistic sampling and their target trajectories split
at token 73. They are therefore performance evidence, not an exact-token
quality claim. Dataset scores and parser validity remain the quality oracle.

The fixed-sample API quality battery gives the same scores as target-only/q7:

| Gate | Adaptive LABD | Target-only/q7 reference |
| --- | ---: | ---: |
| BFCL-style function calling | 29/32 | 29/32 |
| strict JSON Schema | 7/8 | 7/8 |
| ToolACE subset | 12/12 | 12/12 |
| NexusRaven subset | 13/16 | 13/16 |
| structured B1 | 12/12 | 12/12 |
| structured B4 | 12/12 | 12/12 |
| prefix-state/cache checks | 5/5 | 5/5 |

One BFCL parallel-call case initially appeared to regress to 28/32: the final
`math.factorial(number=15)` arrived as `{}`. Target-only and LABD emitted
identical token IDs, proving that this was not a model or acceptance failure.
The q15 burst placed a complete function body in the parser buffer before its
header delta was emitted, and the next/final delta contained only the closing
tag. The Qwen3-Coder streaming parser now consumes an already-complete body
when it emits that header. Its exact 13-burst reproducer and all 104 parser
tests pass; BFCL returns to 29/32.

A separate cold-cache run exposed the same MRV2 coverage gap in startup
warmup: the old model-module scan did not see the GDN cache bound in the static
forward context. A strict JSON request could therefore compile the ordinary
causal-conv update during inference and took 218.45 seconds. MRV2 now warms one
bound Qwen GDN layer with the production cache layout before graph capture. On
a new Triton and Inductor cache, the startup log confirms that warmup, the
first strict JSON request returns correct output in 4.14 seconds, and
inference-time `_causal_conv1d_update_kernel` JIT falls from one to zero. Other
first-request auxiliary kernels still account for most of those four seconds
and are tracked separately from decode performance.

## Remaining promotion gates

1. Compare q7 DFlash2, adaptive LABD, and fixed q16 at B1/B2/B4. Report target
   verify time, full round time, tokens per round, per-stream decode, aggregate
   decode, TTFT, resident requests, and preemptions.
2. Extend the now-passing protocol battery to broad coding and long-context
   suites, then sweep all 128 relevant CUDA-Graph/prefix-cache residues. No
   task-score or parser-validity regression is acceptable.

LABD remains opt-in until every gate above is complete. Drafter-free chaining
is a separate second-stage experiment and stays disabled in this change.
