# QSA operator comparison with sglang-sxm2, 2026-10-11

SGLang is faster in most of these FP16 decode/verification operator cases.
1Cat's native device-history path wins the 8K single-request five-query case;
32K single-request verification is close. The gap is primarily index scoring
for multiple requests and exact top-k at long context, rather than a universal
attention-kernel deficit. These are synthetic single-layer measurements, not
model throughput, TPOT, TTFT, acceptance, or production engine qualification.

## Frozen contract

- 1Cat source: `c39f53abae51df6bf8ef122e34975a9f3596a93c`.
- [sglang-sxm2](https://github.com/dg1kjd/sglang-sxm2/tree/73abfe1dd7e7aae7a62b222a201e48f532fb298e):
  `73abfe1dd7e7aae7a62b222a201e48f532fb298e`.
- One Tesla V100-SXM2-16GB, physical GPU 3 of a four-card NV2 machine.
  Driver 570.211.01; CUDA toolkit/runtime 12.8; Torch 2.9.1+cu128;
  Triton 3.5.1; TileLang 0.1.10. No model is loaded. No GPU clocks are
  changed; observed active SM clock is 1530 MHz, memory clock 877 MHz.
- FP16 query and KV operands. Indexer H4/D128; attention H6/Hkv1/D256;
  compression 4; top-k 512 compressed blocks, expanded to 2048 plus up to
  three open-group tokens. This is a TP-local shape, not a TP4 execution.
- Contexts 8192/32768/131072; request/query-width pairs 1×1, 4×1, 1×5,
  4×5. Last-query causal positions, random physical pages, seed 20261011.
  Gaussian queries are synthetic, not replayed correlated model activations.
- Timed chain: paged index scores → top-k → token expansion → sparse attention.
  Query projection, normalization, RoPE, KV writes, output gate/projection,
  CPU/GPU metadata preparation, communication and runner scheduling are excluded.
  The attention-only probe uses identical selected indices. Full chains use
  each implementation's own selections, including its output ordering.

The harness compiles original native source/header bodies with binding glue
and loads original Python/Triton/TileLang function bodies while replacing
model-module imports, logging and configuration lookup. The final harness uses
the pinned `vllm.triton_utils` shim, including standard package initialization;
it constructs no serving engine. The earlier common-page diagnostic used the
same underlying Triton directly. The primary comparison is repeated with the
final shim. 1Cat uses its shared-key scorer where admitted (single request,
M2–8); other rows use the existing Triton scorer. Standard FP16 attention uses
Triton split/merge. The separately measured native device-history variant
requires a target-owned device history and retains FP32 probabilities/PV;
it is **not** the default paged-cache path or the speculative-draft path.
SGLang uses native H4/D128 scoring and native sparse decode; merge is native
for M1–4 and original TileLang for larger M.

Native device-history and SGLang partial/merge have different internal rounding
boundaries. Matching FP16 storage does not make their results bit-identical.
No algorithm, inference default or serving ABI is changed by this comparison.

## Primary comparison: each implementation's page layout

1Cat uses the previously observed 204-key indexer pages and 816-token KV pages;
SGLang's native scorer requires four-key pages and its main cache uses a flat
request-to-token map backed here by 16-token pages. Identical logical values
are repacked before timing. Repacking and initial table construction are not
charged to either operator. 1Cat's padded capacity is retained in its full
selection chain. This tests the observed 1Cat geometry, not every allocator
or model configuration.

GPU microseconds per layer/step, lower is better. Before each timed region a
32 MiB buffer is overwritten to reduce reuse of the previous invocation's L2
contents; scrub time is excluded by captured external CUDA events. Six groups
of ten samples alternate A/B then B/A; values are medians of group medians.
This is a cache-scrub sensitivity test, not a proof that every cache line is cold.
The two 1Cat columns are measured against SGLang separately; its device-history
pair is shown here. All samples, including the other pair, are retained in JSON.

| Context | Requests × query rows | 1Cat standard | 1Cat device history | SGLang |
| --- | --- | ---: | ---: | ---: |
| 8K | 1 × 1 | 59.39 | 51.20 | 47.10 |
| 8K | 4 × 1 | 126.21 | 89.09 | 64.51 |
| 8K | 1 × 5 | 90.11 | 66.56 | 78.85 |
| 8K | 4 × 5 | 403.71 | 187.14 | 128.26 |
| 32K | 1 × 1 | 87.30 | 77.82 | 48.64 |
| 32K | 4 × 1 | 155.65 | 122.88 | 72.70 |
| 32K | 1 × 5 | 111.62 | 88.83 | 87.04 |
| 32K | 4 × 5 | 493.57 | 340.99 | 179.20 |
| 128K | 1 × 1 | 202.50 | 194.56 | 77.82 |
| 128K | 4 × 1 | 329.22 | 292.61 | 138.24 |
| 128K | 1 × 5 | 195.58 | 177.15 | 162.82 |
| 128K | 4 × 5 | 1083.14 | 933.12 | 402.43 |

At 128K, four requests with five verification rows spend about 714 µs in
1Cat scoring versus 264 µs in SGLang; top-k/expansion takes 113 versus 51 µs.
1Cat's device-history attention is 115 versus 97 µs. In contrast, the
single-request M5 scorer is 37 versus 74 µs and device-history attention is
42 versus 51 µs: those advantages are outweighed by top-k/expansion at
111 versus 48 µs. Stage times are independent probes and need not sum exactly
to the measured chain.

The compiled Triton scorer PTX in this environment contains scalar FMA and
no `mma.sync`. Source-level `tl.dot` is not evidence of a Tensor Core hit.
Changing page geometry substantially changes its cost despite this same
instruction category. This observation does not establish the sole hardware
bottleneck or predict a different compiler version.

## Common small-page diagnostic

Both arms use four-key index pages and 16-token KV pages here. This is useful
for isolating source operators under one layout but **must not replace** the
primary comparison: it greatly penalizes 1Cat's generic multi-request scorer.
For example, the 128K 4×5 device-history chain moves from 6919.94 to 933.12 µs
when 1Cat is repacked to 204/816, with identical logical inputs.

With a 32 MiB scrub before each timed region:

| Context | Requests × query rows | 1Cat standard | 1Cat device history | SGLang |
| --- | --- | ---: | ---: | ---: |
| 8K | 1 × 1 | 59.39 | 50.18 | 47.10 |
| 8K | 4 × 1 | 244.74 | 191.49 | 64.51 |
| 8K | 1 × 5 | 89.09 | 66.56 | 77.82 |
| 8K | 4 × 5 | 779.26 | 582.91 | 125.44 |
| 32K | 1 × 1 | 86.02 | 77.82 | 48.13 |
| 32K | 4 × 1 | 506.37 | 463.87 | 73.73 |
| 32K | 1 × 5 | 109.06 | 89.09 | 86.78 |
| 32K | 4 × 5 | 2056.70 | 1891.84 | 179.20 |
| 128K | 1 × 1 | 202.24 | 191.49 | 77.82 |
| 128K | 4 × 1 | 1608.70 | 1558.02 | 137.73 |
| 128K | 1 × 5 | 195.33 | 177.15 | 162.82 |
| 128K | 4 × 5 | 7077.12 | 6919.94 | 401.41 |

With repeated warm-cache graph execution (eight calls per graph, six alternating
groups, forty replays per group, event envelope divided by calls):

| Context | Requests × query rows | 1Cat standard | 1Cat device history | SGLang |
| --- | --- | ---: | ---: | ---: |
| 8K | 1 × 1 | 43.87 | 36.25 | 33.63 |
| 8K | 4 × 1 | 224.71 | 175.62 | 53.78 |
| 8K | 1 × 5 | 73.48 | 55.16 | 62.30 |
| 8K | 4 × 5 | 750.18 | 584.17 | 121.04 |
| 32K | 1 × 1 | 76.98 | 69.07 | 39.76 |
| 32K | 4 × 1 | 507.72 | 456.94 | 68.64 |
| 32K | 1 × 5 | 109.46 | 85.52 | 81.76 |
| 32K | 4 × 5 | 2050.83 | 1895.18 | 177.07 |
| 128K | 1 × 1 | 193.07 | 184.59 | 74.04 |
| 128K | 4 × 1 | 1607.07 | 1538.81 | 133.23 |
| 128K | 1 × 5 | 194.34 | 173.60 | 158.63 |
| 128K | 4 × 5 | 7053.06 | 6918.37 | 414.30 |

The warm-cache experiment reuses a single layer's storage. Its advantage
cannot be multiplied by model layer count or converted into tokens/s.

## Correctness and limits

All twelve shape cases complete in each of the three runs. Scorers agree
with a FP32 oracle over visible columns. On these inputs, both identical-score
top-k and independently computed-score top-k select the same block sets;
expanded token sets also agree. SGLang's atomic collection order is not
required to match 1Cat's deterministic ordering. Equal-score cutoff tie
behavior is not qualified by these random inputs.

Every attention arm is compared with a FP32 softmax/value oracle, including
the full chain's own actual selected tokens. Maximum attention-only absolute
error is 8.02e-5; maximum full-chain error across the runs is 1.01e-4; maximum
attention relative RMS error is 3.41e-4. Graph replay is checked against eager
execution after the attention query is changed in place. This does not validate
model quality, acceptance, request-state lifecycle, or arbitrary padding.

The JSON also retains eager Python submission time and peak allocated bytes.
These are **harness measurements**, not engine host-overhead rankings: package
imports, configuration getters and logging are adapted, and concurrent CPU
compilation was present on the machine. Allocation increments exclude existing
KV storage and reserved workspace. The native history arm separately reserves
4,086,720 bytes for its FP32 partial/state workspace, following the M≤20 bound;
it is not a zero-workspace implementation.

Unmeasured: FP8 (1Cat E4M3 and SGLang E5M2 have different storage semantics),
prefill and dense-short-context dispatch, page4 large-M fusion, full engines,
weights, communication, PLE, DDTree and model output quality. In particular,
SGLang's dedicated sparse-prefill native kernel requires E5M2, so its decode
kernel is not relabelled as an FP16 prefill benchmark.

## Reproduce and retained evidence

Use source checkouts at the pinned revisions and a dedicated Python environment
with the versions above. The benchmark expects CUDA 12.x with SM70 support;
CUDA 13 is unsuitable for compiling these V100 kernels. Build Flash-V100 from
the supplied 1Cat tree with its existing setup.py. No external model files or
private prebuilt kernel overlays are required.

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=7.0
export CUDA_VISIBLE_DEVICES=3
export MAX_JOBS=1
export TRITON_CACHE_DIR="$PWD/qsa-ab-cache/triton"
export TILELANG_CACHE_DIR="$PWD/qsa-ab-cache/tilelang"
export TORCHINDUCTOR_CACHE_DIR="$PWD/qsa-ab-cache/inductor"
.venv/bin/python benchmarks/kernels/benchmark_qsa_sxm2_ab.py \
  --onecat /path/to/pinned/1Cat-vLLM --sglang /path/to/pinned/sglang-sxm2 \
  --cache "$PWD/qsa-ab-cache/native" --output native-layout.json \
  --onecat-index-page 204 --onecat-kv-page 816 --cold-cache --replays 10
```

Omit the page options for the common-page experiment; also omit `--cold-cache`
and `--replays 10` for the warm experiment. Native binaries are task-local
operator bindings, **not** a source-installed serving artifact or worker route
validation. Helpers and original kernels remain separate; no kernel arithmetic
is recreated by the harness.

- [Primary samples and correctness](native-layout.json)
- [Common-page scrubbed samples](cold.json)
- [Common-page warm samples](warm.json)
- [Source, binding, compiler command and binary SHA256 manifest](provenance.json)
- [Triton scorer code-generation metadata](scorer-codegen.json)
- [Benchmark driver](../../../benchmarks/kernels/benchmark_qsa_sxm2_ab.py)
- [Source adapters and native binding glue](../../../benchmarks/kernels/qsa_sxm2_adapters.py)

Initial harness bring-up needed SGLang's `SGL_CUDA_ARCH=700` compiler definition
and explicit TVM-FFI export expressions. A graph check was also corrected to
ignore unwritten masked scorer-tail columns, just as the real top-k consumer
does. These were harness failures, not engine/kernel correctness failures;
no timing from a failed case is used. Original logs remain in the task's
`native-build.log`, `smoke*.log`, `warm.log` and successful `warm2.log`,
`cold.log`, `native-layout.log`, `native-layout-final.log` artifacts outside Git.

## Next implementation priorities

1. Extend native index scoring to the supported multi-request shapes while
   retaining paged geometry, causal bounds, deterministic selection and fallback.
   Validate real 204/816 geometry as well as small-page diagnostic layouts.
2. Improve exact long-context and batched top-k. Do not replace the deterministic
   score/index tie contract with SGLang's unordered collection without a separate
   semantics decision and cutoff-tie tests.
3. Preserve the already useful M5 shared-key/device-history paths. Optimize the
   entire score/select/attention chain before adding another isolated attention
   variant. A later same-model comparison is needed for serving conclusions.
