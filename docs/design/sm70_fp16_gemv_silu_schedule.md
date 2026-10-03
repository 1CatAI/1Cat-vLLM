# SM70 FP16 GEMV/SiLU scheduling

The retained operator improves a small-grid GEMV with long weight rows. More
independent lanes hide row-load latency. Inputs and weights remain FP16; dot
products accumulate in FP32, retain the FP16 projection boundary, then apply
FP32 SiLU to a prefix. FP32 reduction association can change. Exact top-k,
expert selection, dense-weight precision, and speculative decoding are unchanged.

`Sm70Fp16GemvSiluKernel` admits contiguous same-device FP16 matrices, M1–16,
positive K, two valid weight-row ranges, disjoint output storage, finite positive
division, and zero-padded outputs on SM70. It has no model or TP predicate.
For at most twice the device's SM count of active output rows and K4096–16384,
M1–2 use K2048/eight warps and M3–4 use K512/four warps. Other geometries retain
K256/four warps. The existing HC adapter enables the policy for no-speculation,
non-batch-invariant execution. Existing M2–16 packed HC dispatch runs first;
MTP/direct-call defaults and batch-invariant execution keep their old reductions.
Preparation appears in the existing worker acceleration report. There is no new
runtime environment variable.

## Measured contract

Both normal control/candidate wheels were packaged from integration base
`3a147164833e58df2f80203cd169cbd71cea41e1`. Native objects come from the declared
standard `dev1228+gcf2a1285e` precompiled artifact; this Python/Triton change adds
no native DSO. The wheels ship the same 15 native objects, inspected with
`readelf -d`; no build-host/task-cache RPATH or private library override is used.
Tests ran in a fresh installed environment outside the source root.

The model is Qwen3.8-Flash-Next-NVFP4 revision
`7b719225242aacd3dbd3f9407468c2ee9a9d2594`: checkpoint NVFP4 experts,
checkpoint dense weights cast to FP16, FP16 activation/KV, FP32 accumulation and
SSM state, TP4/PP1 on four fully connected NV2 V100-SXM2-32GB GPUs. Driver is
580.173.02, Torch 2.10.0+cu128, Triton 3.6.0, CUDA 12.8, Nsight Systems 2024.6.2.
The existing power limit is 185 W; memory clock is 877 MHz and observed decode
boost reaches 1530 MHz. Hardware settings were not changed.

Timing uses 8192 input/513 forced output tokens, max context 262144, batch
budget 8192, max sequences 1, memory utilization 0.90, prefix cache off,
no MTP, and FULL+PIECEWISE graphs. Forced EOS suppression is timing-only.
As requested, ngram tables use **pure disk mmap**, not a full pinned-host copy:
existing `VLLM_SM70_QWEN38_HYBRID_PLE=0`, `VLLM_PLE_CPU_OFFLOAD=1`, and
`VLLM_PLE_DISK_OFFLOAD=1`. Reclaimable OS page cache and small transport buffers
remain. This differs from historical pinned-UVA measurements near 98 tok/s.
Every GPU run held `/tmp/gpu0-3.lock`.

## Before/after measurements

The installed operator screen rotates 16 real checkpoint matrices (over L2),
alternates control/candidate order, and measures complete FP16 projection plus
SiLU and zero padding. The four TP rank row mappings run on one GPU.

| Measurement | Control | Candidate |
| --- | ---: | ---: |
| HC down rank-0 mapping, median us | 4.9540 | 4.1930 |
| HC down rank-1 mapping, median us | 4.9712 | 4.2252 |
| HC down rank-2 mapping, median us | 4.9660 | 4.1758 |
| HC down rank-3 mapping, median us | 4.9485 | 4.2072 |
| Model graph HC down service, ms/rank/step | 0.6110 | 0.5113 |
| HC down calls/full model graph | 96 | 96 |
| Kernels/full model graph | 1349 | 1349 |
| Unprofiled pure decode median, ms/token | 13.0500 | 12.8927 |
| Unprofiled pure decode median, tok/s | 76.63 | 77.56 |
| Unprofiled sample range, ms/token | 12.9732–13.0985 | 12.8096–13.0744 |

Graph services use graph-launch correlation IDs, not host-window kernel guesses:
23 full replays/rank, excluding first/last, give 84 middle rank replays per arm.
The new symbol executes exactly 96 times per full graph. Service sums are not
endpoint TPOT. The three control and six candidate unprofiled samples overlap;
the observed 1.22% endpoint difference is **not a demonstrated stable gain**.
Operator and graph-service improvements are the retained evidence. This change
does not reduce node count, achieve 5 ms/token, or establish C2–C16 throughput.

## Numerical and quality gate

Nine GPU tests pass: M1/2/3/4/8/16 geometries, changing and poisoned graph
inputs, row ranges/padding, FP16 subnormal projection, and invalid-layout/alias
rejection. The real-weight screen checks all 16 matrices at four activation
scales for all rank mappings against FP64 projection followed by the intended
FP16 boundary/SiLU. Maximum candidate relative L2 error is `7.36e-5`; all
outputs are finite and padding is zero. Ninety-three targeted HC/MTP/collective
host tests pass. Ruff and `git diff --check` pass.

The frozen regression set uses the first 12 sanitized MBPP tasks, first 12
GSM8K test questions, eight Chinese questions, and needle lengths 8192/32768/
131072/258048 at depth 0.5. Sampling is thinking on, temperature 1, top-p 0.95,
top-k 20, seed `4201 + case index`, max output 4096, and natural EOS.

| Quality subset | Control | Candidate |
| --- | ---: | ---: |
| MBPP executable tests | 11/12 | 12/12 |
| GSM8K numerical answer | 12/12 | 12/12 |
| Chinese QA | 8/8 | 8/8 |
| Needle retrieval | 4/4 | 4/4 |

The control's MBPP-16 repeats reasoning and reaches the output limit; the
candidate completes correctly. All 36 candidate outputs finish thinking and
stop naturally, with nonempty final answers, no replacement characters, and no
detected repeated long lines. Sixteen of 36 token sequences match: token
identity is diagnostic, not the gate. Control quality ran under nsys with
collection disabled; candidate quality was unprofiled. This small set supports
this regression gate, not a full-dataset or universal quality guarantee.

Frozen cases SHA256:
`62d3ae65ce754ee10d2bef31cd7548d3143362d9a6c4717beeedc48bb4fb363d`.
Official source files, recorded in that manifest:

- [Sanitized MBPP](https://raw.githubusercontent.com/google-research/google-research/master/mbpp/sanitized-mbpp.json), SHA256 `ca95deaa9a01ef0a6f439f88bcf0dd3db3563d22f22aad6cae04ebb9a8d8c8e9`.
- [GSM8K test](https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl), SHA256 `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`.

## Reproduction and artifacts

Install the normal wheel, then run these source benchmarks. Keep the generated
case manifest fixed between arms and use separate output/cache directories.

```bash
.venv/bin/python benchmarks/prepare_sm70_qwen38_quality_cases.py \
  --dataset-dir /path/to/official-datasets --out /path/to/fixed-cases.json
.venv/bin/python benchmarks/benchmark_sm70_qwen38_quality.py \
  --model /path/to/model --cases /path/to/fixed-cases.json \
  --output /path/to/arm/gate.json
.venv/bin/python benchmarks/kernels/benchmark_sm70_fp16_gemv_silu.py \
  --model /path/to/model --out /path/to/operator.json
```

The published scorer reproduces all 72 retained per-case results, and the
case generator reproduces the frozen manifest byte-for-byte. Historical TTFT
fields in the initial control JSON subtract different clock domains and are
invalid; pure decode subtracts two monotonic timestamps and is unaffected.
The retained benchmark uses `first_token_latency` for TTFT and reports prefill
separately. No invalid TTFT is used as performance evidence.

Task-retained artifacts (excluded from Git):
`.artifacts/remote_{control,candidate}_graph_nodes.{nsys-rep,sqlite}`,
`graph_down_correlated_comparison.json`, `quality_comparison.json`,
`remote_{control_quality_profile,candidate_quality}.json`,
`remote_installed_hc_shard_micro.json`, and `package-inspection/manifest.json`.
Control wheel SHA256 is
`3293f3461369c1c0abd9c03c89b644ffab81af8a4b3106df0de3ec9a43345ff4`;
candidate wheel SHA256 is
`6fe5eaf39bbf87a479f1366011e551884f51816be69a2598849b5dff7d7bb034`.

The remaining disk-mode profile has about 2.2 ms of gaps immediately before
PLE dequantization. CPU offload result DMA averages 2.6 us but enqueue-to-start
delay averages 1.14 ms. These are instrumented dependencies, not a causal
partition of unprofiled TPOT. A one-GPU synthetic worker-context copy prototype
did not show stable benefit and is not selected. Follow-up work must preserve
disk storage, measure the outstanding dependency, then evaluate larger fusions
and concurrent endpoints. Dense QPN8 remains off and separately qualified.
