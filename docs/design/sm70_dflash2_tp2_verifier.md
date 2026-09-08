# DFlash2 TP2 verification cost

## Scope and frozen baseline

The TP2 campaign targets approximately 25 ms per complete B1/q8 DFlash2 round
on two rear V100-SXM2-32GB GPUs. A round includes target, logits/sampling,
state handling, context work and draft. TP4 optimization is a separate campaign.

Integration base: `e5d63c51f0fcc1ddf75d229e3df06bf52df206f5`.
Use the QUASAR Qwen3.8-27B NVFP4 checkpoint at
`d8e6fbfa3e3a78899b440222b827430045a05b44` and the DFlash2 checkpoint at
`dedf8df68adfb1afeaf7b7480c0a0243108177b4`. The workload uses Python 3.12.13,
Torch 2.10.0+cu128, CUDA 12.8, TP2 on physical GPUs 4 and 7, FP16 activations,
E4M3 target KV, FP16 draft KV and FP32 logits. Both attention backends are
FLASH_ATTN_V100. Keep V2 runner, target/draft CUDA Graphs, context pipeline and
context KV graph enabled. Maximum context is 262144, batch-token budget 4096,
maximum sequences 4 and memory utilization 0.8. Only one request is active.

Sampling remains temperature 1, top-k 20, top-p 0.95, xhigh thinking, natural
EOS and at most 1024 output tokens. The release1k fixture uses seed 20260925
and 1019 input tokens; MBPP28 uses seed 0 and 135 input tokens. Startup and
model preparation are outside decode timing. The original baseline uses
frozen copies of existing native libraries; it is not a rebuild of all main
sources. Retained runtime manifests hash the actual mapped worker libraries.

One startup, one warmup and five measured requests per fixture gave:

| Metric | release1k | MBPP28 |
| --- | ---: | ---: |
| Median request-average complete round, ms | 44.973 | 35.119 |
| Median pure decode, tokens/s | 67.832 | 127.662 |
| Median warm TTFT, ms | 575.387 | 147.247 |
| Accepted drafts per round | 2.063291 | 3.500000 |
| Emitted tokens per round | 3.063291 | 4.500000 |
| Output tokens | 242 | 270 |
| Draft rounds | 79 | 60 |

Outputs repeat within this startup and finish naturally. MBPP28 passes its
three supplied assertions. These are short-context baselines, not a 256K
latency result or the three-startup final acceptance gate.

## Trace and first optimization

Ten steady rounds from both ranks show approximately 35.260 ms of target GPU
service, including 14.963 ms of TurboMind projections and 12.838 ms of scalar
attention. Draft GPU service is 6.638 ms; target head/sampling is 2.274 ms.
The profiled critical-rank round interval is 47.057 ms. Service sums and
profiled wall intervals are diagnostic, not unprofiled performance claims.

TP2's 12 query heads and two KV heads do not enter the existing six-head,
single-KV-head E4M3 grouped route. Its scalar attention uses 1024-token
partitions, FP32 partial output and FP32 partition statistics. The observed
launch is `(8, 12, 256)` CTAs, 256 threads/CTA, 40 registers/thread and
12880 bytes shared memory for the frozen control.

`VLLM_FLASH_V100_TP2_E4M3_SCALAR_FAST=1` selects an experimental specialization
only for q shape `[8,12,256]`, E4M3 KV with two heads, FP32 partial storage,
1024-token partitions and full attention without an anchored window. It is
off by default. Unverified shapes use the original route. A requested matching
route rejects a stale native library instead of silently reporting success.

The specialization constructs normal E4M3 values directly in FP32 bit fields,
retains the original signed zeros, subnormals and NaN payload, and unrolls the
PV loop by eight. Each output still follows the original ascending-token FMA
chain. It retains partition boundaries, score reductions, FP32 intermediate
storage, output rounding, KV scales and the original final reduction kernel.
The native launch counter proves host dispatch, including capture-time calls;
it does not count CUDA Graph replays or model rounds.

## Evidence and promotion status

The initial isolated implementation passes:

- All 256 E4M3 byte encodings, with bitwise equality against the original
  decoder, including both signed zeros and both NaN encodings.
- 33 fixed-operand comparisons across bit conversion alone and PV unroll
  factors four/eight. Final outputs and valid partial output/max/sum bits
  match the frozen native implementation. Lengths include zero, partition
  and page boundaries, 65537 and 262144; live CUDA Graph inputs change between
  replays.
- At length 3297, all variants retain the same FP64-reference error:
  maximum absolute `3.0444386e-5`, p99 absolute `1.4819749e-5`, relative L2
  `2.0301283e-4`.
- CUDA 12.8 Compute Sanitizer memcheck and racecheck on the winning u8
  partition kernel report zero errors and zero hazards, respectively.
- Live same-call shadow comparison on both model ranks: 27072 attention
  calls, 665321472 output elements, zero bit differences and zero nonfinite
  outputs. The original result drives generation. These runs contain
  diagnostic work and are excluded from speed evidence.

For sixteen distinct KV layer working sets, the operator median is 12.712 ms
for the frozen scalar implementation and 3.892 ms for exact bit conversion
with PV unroll eight. Conversion alone and unroll four are approximately
6.335/6.314 ms. These are attention operator results, not complete rounds.
The private u8 implementation is pinned by SHA256
`696545418c6dae261f0bc6a3a530b34464d040de8e404a3069cfd8c2a7762ad3`.
The integrated native build is separately pinned by SHA256
`f916e9e370eeb8d865b4de9d8b64f6e66d8831c0b4458dbe3087141dbadc1d19`;
its own GPU and runtime gates must pass before substituting it for the
isolated implementation.

The first contemporaneous control reproduces round cost at 44.986/35.075 ms.
Its release1k trajectory has 349 tokens rather than the original startup's
242, while within-startup repetitions match. The candidate was disabled in
this control. This pre-existing startup variation is not an allowed quality
tolerance; cross-startup token/acceptance comparisons must retain this limit.
The first separate-startup candidate measures 35.921/32.706 ms, with
release1k/MBPP28 outputs of 283/297 tokens. Its corresponding control produces
349/270 tokens. The trajectories and acceptance counts differ, so the
approximately 20.15%/6.75% latency reductions are provisional performance
observations, not accepted quality-preserving gains. A within-startup graph
comparison is used next to keep prefill and projection choices fixed.
The 25 ms target and final promotion remain outstanding.

The integrated native build passes 14 tests, including exhaustive byte
decoding, 262144-token graph replay, FP64 reference, unsupported-shape fallback
and stale-library rejection. The initial QPN2 TP2 projection screen uses
sixteen real matrices from four adjacent layers. Its best working-set median
is 0.685 ms versus 0.930 ms for TurboMind, but several reference-error metrics
grow. This arithmetic candidate is rejected for model use. Source inspection
identifies early FP16 rounding of the global scale as a separate precision
candidate; it requires fresh operator and model gates.

## Reproduction and retained negative results

Build Flash-V100 from this branch with the same CUDA/Torch/compiler flags and
select that module before running the tests. Set `CUDA_VISIBLE_DEVICES` only
to an owned rear GPU, and use private build/compiler caches.

```bash
TORCH_CUDA_ARCH_LIST=7.0 MAX_JOBS=2 .venv/bin/python -m pytest \
  --confcutdir=tests/kernels/attention \
  tests/kernels/attention/test_sm70_tp2_e4m3_scalar_fast.py \
  tests/kernels/attention/test_sm70_e4m3_scalar_fp32.py -q
```

The GPU gate includes an exhaustive decoder comparison, strided output
sentinels, changing page/sequence visibility, graph replay, FP64 reference
and fallback dispatch. The stale-library gate also runs without a GPU.

Task artifacts are retained under campaign identifier
`v100-quasar-dflash2-tp2-25ms-20260908`. They contain baseline contracts,
worker DSO inventories, Nsight data, raw endpoint responses, operator results,
sanitizer logs, source/build hashes and serial GPU queue records. The baseline
campaign identifier is `v100-quasar-dflash2-tp2-baseline-20260908`.

A capped partition-grid experiment passed 66 bitwise cases but did not improve
the sixteen-layer working set: 12.701 ms control, 13.641 ms at cap one and
approximately 12.719 ms at caps two through sixteen. It was rejected before
model testing. Do not repeat that path without new bottleneck evidence.

The first memcheck invocation loaded both experimental u4/u8 DSOs and reported
`CUDA_ERROR_INVALID_HANDLE` in `cuKernelGetFunction` at the second decoder-LUT
launch. The quality gate blocked model work. Running only the winning DSO
passed both sanitizer tools with API error checking retained. The failed
invocation remains recorded rather than counted as a pass.
