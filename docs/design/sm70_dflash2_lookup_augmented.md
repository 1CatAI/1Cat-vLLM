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
  complete erase metadata for the following step.
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
reference for both q8 and q16 (four strict cases). The earlier GPU LABD suite
also passes suffix hit/miss/overlap, request eligibility, 7-to-15 fusion, and
sparse point-mass cache invariants.

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

An initial q15 route smoke produced coherent lookup proposals and positions
beyond the neural seven-token block, but it ran with async scheduling. Its
`75` drafts over five rounds are proof that async pinned every round to 15, not
valid evidence for the adaptive policy or throughput. That run is retained as
a route/correctness diagnostic only.

## Remaining promotion gates

1. Benchmark generic 3,456-page grouped verification against independent XQA
   at q8/q16 and 1K/32K/128K. Add a fixed CUDA specialization only if the
   generic path loses materially.
2. Run a normal miss workload and a context-copy workload with synchronous
   scheduling. Scheduler metrics and profile transitions must prove q8 remains
   the default and q16 is entered only for sustained hits.
3. Compare q7 DFlash2, adaptive LABD, and fixed q16 at B1/B2/B4. Report target
   verify time, full round time, tokens per round, per-stream decode, aggregate
   decode, TTFT, resident requests, and preemptions.
4. Run probabilistic quality gates against target-only and the accepted DFlash2
   baseline, including coding, tool calling, structured JSON, long context,
   and all 128 relevant CUDA-Graph/prefix-cache residues. No task-score or
   parser-validity regression is acceptable.

LABD remains opt-in until every gate above is complete. Drafter-free chaining
is a separate second-stage experiment and stays disabled in this change.
