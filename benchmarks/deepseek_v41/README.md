# DeepSeek V4.1 Flash on four V100s

> Phases 2–3 (prefill 4x, expert dedupe, 512K packed KV, prefix cache, thinking API):
> see [HANDOFF.md](HANDOFF.md) for launch commands, switches, results and open items.

Baseline: `6a143197e6e5f6fa77bd483df0c5a09dcea6aa20`.
Branch: `codex/ds41-flash-sm70-20261024`.

The first profile targets V100 32 GiB ×4, TP4, one request, 16,384 total tokens
and DSpark5. The current numerical profile completed 30 minutes of stability
qualification and OpenAI API checks, including a 16,255-token input.
See the [54633 qualification report](results/54633-20261024.md) for raw-run
performance, resource peaks, the startup registration incident and boundaries.

## Implementation and numerical boundaries

- Independent target/draft classes live in `vllm/models/deepseek_v41/`.
  The runtime contract validates geometry and rejects unsupported topology,
  prefix caching and asynchronous scheduling. Strict GGUF mapping covers all
  1008 backbone tensors, including 120 routed-expert tensors.
- Dense matrices retain original packed storage, with `prefill_min_m=1`,
  FP16 Tensor Core GEMM and a 32 MiB bounded dequantization workspace.
  Expert M1/M6 use DP4A. M2–5 use bounded FP16 GEMMs with device-side routes.
  Prefill transcodes tiles of 128 experts into one reusable workspace.
  There is no full FP16 expert library or CPU matrix fallback.
- Ratio-1/2 compression, four shared compressed KV sources, top-512 selection
  and delayed mHC follow the reviewed V4.1 sources. The 16K bypass skips only
  candidate-block filtering. FP16 caches hold values after official FP4/FP8 QDQ.
- Norms, routing, mHC/HC residuals, C2 projections/pooling and Engram gates retain
  necessary FP32 arithmetic. Draft expert projections remain FP16; routed sums
  and TP reduction enter HC in FP32. A real tool query produced finite card
  partials summing to about 66,019, beyond FP16 range.
- Engram has one node-level reader, row deduplication, bounded shared raw
  buffers, per-card dequantization and sharded projection. It preserves the
  explicit Engram pad token 2 and never replicates the complete tables.
- DSpark5 uses three 128-expert/top-3 layers and pre-block HC means at target
  layers 37/38/39, after Engram. Five noncausal query rows feed six target rows.
  Embedding/head storage is shared. Native FP8/MXFP4 shards are linked unchanged.
- Engram history, C2 partial groups, window and compressed KV retain candidate
  state. Runner acceptance selects the input prefix including its anchor.
  EOS/output-budget clipping precedes state commit.
- The independent tokenizer uses the reviewed official MIT prompt encoder.
  The renderer rejects unsupported media before media I/O. The parser supports
  automatic and streaming tool calls using V4.1's spaced DSML tags.
  Required/named constrained tool decoding is not admitted.

There is no full FP32 reference model. Numerical checks decode bounded weight
blocks and respect each operator's actual FP16/DP4A boundary.

## One original host expert pool

`shared_original` stores unsliced original GGUF bytes in one size-sealed memfd
per TP group, mapped with `MAP_SHARED`. Processes independently register the
same object with CUDA and obtain process-local device pointers. Messages carry
64-bit offsets and validate object identity, size and peer uid.

Rank 0 alone reads original payloads and owns the native coordination thread.
Four completion sequences and reader leases fence reuse. Evictions get distinct
host destinations; whole down blocks have one writeback owner. Completed bytes
are compacted by a bounded CPU byte copy. Unlock publication does not depend on
a CUDA stream. Owner death, timeouts and inconsistent plans fail the request.

| Offload | Cold experts/layer | Shared cold bytes | GPU slots/card | Raw GPU bytes/card |
| ---: | ---: | ---: | ---: | ---: |
| 30% | 115 | 39,553,228,800 | 277 | 26,362,255,178 |
| 25% | 96 | 33,018,347,520 | 296 | 28,170,495,818 |
| 20% | 77 | 26,483,466,240 | 315 | 29,978,736,458 |

Slots include eight separate GPU spares. There is one group-level 512 MiB
exchange reserve. The canonical prefill workspace is 566,231,040 bytes/card.
These exclude dense weights, KV, Engram and other buffers. The first version
replaces victim slots directly; overlap is not assumed.

Set `expert_cache_host_storage="shared_original"`,
`expert_cache_offload_fraction=0.30`, `expert_storage="original"`,
`expert_arena=true`, `expert_cache_fraction=0`, `expert_cache_host_slack=0`.
Conflicting legacy budgets and predictive-prefill controls are rejected.

The actual-size registration probe passed with 40,090,099,712 bytes. The
checkpoint probe verifies startup bytes and repeated M6 swaps on all 40 layers.
Matching device/inode and summed object PSS establish one physical pool.
Four RSS values overcount it. Storage probes do not establish full-model fit.

## Checkpoint and environment

Model files are read-only under `/home/ymzx/ds41-models` on 54633:

| Component | Bytes | Decimal GB |
| --- | ---: | ---: |
| Backbone, four GGUF shards | 138,238,893,056 | 138.24 |
| Engram GGUF including header | 110,595,290,816 | 110.60 |
| DSpark, three HF shards | 7,933,129,808 | 7.93 |
| Total | 256,767,313,680 | 256.77 |

Routed-expert payload: 132,073,390,080 bytes. The checkpoint audit reads headers
and checks the existing SHA256 log against its manifest; it does not rerun
complete file hashing. Sources are under `/home/ymzx/ds41-dev/src`, evidence
under `/home/ymzx/ds41-dev/state`.

GPU qualification uses `ds41-compute-20261024`, exposing only GPUs 0–3 with a
54 GiB memory limit. `ds41-dev-20261024` is GPU-hidden for CPU tests. Run Python
inside these containers through their mapped `.venv/bin/python`.

Native `_C` and `_moe_C` were rebuilt. The unchanged GGUF dense extension keeps
its seed binary; relevant source matches the baseline and bounded decoding was
checked independently. Seed package metadata may name `705aca212`; source HEAD,
native digests and per-file SHA256 manifests identify the running implementation.

## Reproduce qualification

From the source directory inside the appropriate container:

```bash
ds41_state=/home/ymzx/ds41-dev/state
.venv/bin/python -m benchmarks.deepseek_v41.checkpoint_audit \
  --model-root /home/ymzx/ds41-models --output "$ds41_state/checkpoint_audit.json"
.venv/bin/python -m benchmarks.deepseek_v41.prepare_dspark \
  --model-root /home/ymzx/ds41-models --destination /home/ymzx/ds41-dev/dspark-v41
.venv/bin/python -m benchmarks.deepseek_v41.shared_host_probe \
  --bytes 40090099712 --output "$ds41_state/shared-host-full.json"
.venv/bin/python -m benchmarks.deepseek_v41.shared_checkpoint_probe \
  --model-root /home/ymzx/ds41-models --layers 40 --offload 0.30 \
  --output "$ds41_state/shared-checkpoint-40.json"
.venv/bin/python -m benchmarks.deepseek_v41.acceptance \
  --model-root /home/ymzx/ds41-models --chunk 512 \
  --draft /home/ymzx/ds41-dev/dspark-v41 --context-checks --force-selectors \
  --soak-seconds 1800 --output "$ds41_state/acceptance.json"
```

Run performance comparisons sequentially in separate engine processes. Both use
KV 256 MiB, chunk 512 and 30% offload. Raw timing has two warmups and three
samples per length, without finite/stage hooks; a separate profile records
CUDA intervals, acceptance and byte counters.

```bash
ds41_state=/home/ymzx/ds41-dev/state
.venv/bin/python -m benchmarks.deepseek_v41.performance \
  --model-root /home/ymzx/ds41-models --output "$ds41_state/performance-no-spec.json"
.venv/bin/python -m benchmarks.deepseek_v41.performance \
  --model-root /home/ymzx/ds41-models --draft /home/ymzx/ds41-dev/dspark-v41 \
  --output "$ds41_state/performance-dspark.json"
```

Checks cover forced acceptance 0–5, C2/window boundaries, rejection continuation,
EOS and the 16K tail. Sampling-law tests use 32,768 synthetic chains with real
Markov proposals/rejection sampling; they are separate from checkpoint quality.
Stage totals include nested work. File/carrier bytes are logical work, not
measured PCIe transactions. Do not sum nested times or replicated group counters.

## Serving scope

`serve-54633.yaml` records FP16, TP4, DSpark5, synchronous eager execution,
single-request/16K limits, prefill 512, MoE sub-batches 512 and prefetch 1.
It binds `127.0.0.1:18000`. Combined target/draft chunks 2048/1024 exceeded
memory; 512 is the approved fallback with unchanged offload. Preserve at least
8 GiB host MemAvailable.

```bash
.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --config benchmarks/deepseek_v41/serve-54633.yaml
```

Multimodal inference, concurrency above one, lengths above 16K, asynchronous
scheduling, graph execution and other GPU/TP topologies are outside this first
profile. Full-model capacity at 25%/20% needs separate qualification even when
raw expert-storage checks pass.
