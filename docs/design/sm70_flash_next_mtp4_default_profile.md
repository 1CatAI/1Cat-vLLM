# Flash-Next MTP4 shared-path baseline and trace, 2026-09-26

The owner accepted **27.3963 ms per complete MTP round** as the development
baseline. It includes target verification, four draft steps and host scheduling;
it is not target-forward-only latency. Reuse this recorded baseline and run only
the focused case affected by the next concrete optimization. No more wheel
packaging, no-MTP baseline sweeps or repeated broad quality matrices are needed.

## Shared implementation

PR #684 enables the direct M5 experts and TP4 push collective from merged
PR #398 together with shared/fused GDN metadata. It also admits exact MTP4 to
the existing Qwen3.8 common defaults: checkpoint-FP16 GEMV, fused GDN inputs,
fused HC, shared-expert overlap and MoE add/all-reduce. Existing shape guards
remain: an M=1 GEMV is not called with M=5 verifier inputs.

The drafter reuses those operators through a second compiled view of the same
parameters/state. Target and draft capture share the decode-compilation context.
The existing split graph manager is selected by default for this admitted
MTP4 contract: verifier sizes are `(5, 10)`, while single-token draft decode
retains `(1)`. Without this split, shared verifier-size normalization silently
removed the draft M=1 graph and its prepared fast operators never executed.
Explicit environment overrides remain supported. No new arithmetic kernel,
weight format, model replica or private runtime library is introduced by this
common-path extension.

## Frozen workload and provenance

- Model: `RadixArk/Qwen3.8-Flash-Next-NVFP4`, TP4/PP1, physical GPUs 0-3,
  V100-SXM2-32GB; driver 580.173.02, CUDA toolkit 12.8.93,
  Torch 2.10.0+cu128, Python 3.12.13.
- Checkpoint NVFP4 experts, FP16 activations/KV and native FP32 SSM state;
  V2 runner, FULL_AND_PIECEWISE graphs, one request, max length 32768,
  max batch tokens 8192, GPU memory utilization 0.95, prefix caching enabled.
- Four greedy MTP drafts. The fixed cost fixture has 8192 input tokens,
  513 output tokens, temperature 0, seed 0 and forced length. The short trace
  uses the identical input and 129 output tokens, exactly the endpoint prefix.
- Hybrid PLE: asynchronous disk-mmap prefill, local pinned-UVA decode.
  `VLLM_QWEN4EXP_PLE_HOST_GIB=12` per rank retains all PLE rows in host RAM:
  11.92 GiB per rank, 47.68 GiB total. This is not a disk-only low-RAM mode.
- All four #398/#684 acceleration switches and the common operator switches
  are unset at launch and resolve on through defaults, including dual
  compilation, hybrid PLE and split draft graphs.
- Measured source: `6b8cc4eaa7bed73b56176549ab6dbcce940b5887`, integrating
  main `4b8855c5a087df224058df3a090cf207682e4d47`, plus the retained source patch
  SHA256 `35f8cb9ac6ebea7ce4cf58562c503c376596ba0991532dfa8193376b348a5636`.
  A later typing-only decorator overload correction does not change execution.
- Source checkout and its normal in-place native build are used directly.
  No new wheel, copied sidecar, `LD_PRELOAD` or private library overlay is used.
  The native build is from `9fab7604f`; no native code changed between that
  build and this measured source. `_C.abi3.so` SHA256 is
  `616a7c79d6397b880bfc6fe540c1d544a1356a6eab14e14f3a8aac6327d59fda`.

## Accepted endpoint baseline

These requests run with Nsight attached but **capture disabled**, before
`cudaProfilerStart`. They are not an independent profiler-free replication.
Decode excludes prefill and the first output token; round cost is the request's
decode duration divided by its speculative-round count. One request per case
establishes an engineering reference, not a latency distribution.

| Case | Input/output tokens | TTFT ms | Prefill ms | Rounds | Full-round ms | Acceptance length | Decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Accepted fixed fixture | 8192/513 | 2589.48 | 2573.63 | 335 | **27.3963** | 1.5284 | 55.79 |
| Natural LIS | 70/284 | 265.39 | 201.61 | 74 | 28.2805 | 3.8243 | 135.23 |
| Natural arithmetic | 89/329 | 231.16 | 206.34 | 78 | 28.3442 | 4.2308 | 148.36 |
| Natural signed division | 72/421 | 203.47 | 185.69 | 171 | 28.2516 | 2.4561 | 86.94 |

The fixed fixture takes 9.177750 s of decode, or 17.9253 ms per emitted token.
It continues after normal EOS and is a round-cost fixture, not a natural-output
quality or general-throughput benchmark. Acceptance length is essential: the
older CPU-worker-PLE run's 126-128 tok/s used acceptance length 3.9237 and a
different output stream. Do not compare that throughput as pure kernel speed.

Relative to the first common-stack run without the M=1 draft graph, the final
run preserves every output token and speculative counter in all six cases.
The fixed fixture improves from 28.0489 to 27.3963 ms/round (+2.38% tok/s);
the three natural responses improve from 98.95/109.30/63.44 to
135.23/148.36/86.94 tok/s (+35.74% to +37.04%). These are focused single-run
observations, not a claimed universal speedup.

## Corresponding graph-node trace

Nsight Systems 2025.1.1 uses `--trace=cuda,nvtx --sample=none`,
`--cuda-graph-trace=node:host-only` and a CUDA-profiler-API capture range.
The short request has 85 rounds and **84 closed intervals on every TP rank**.
Each interval starts at the fused GDN group-metadata kernel and ends at its
next invocation. Unique sampling/draft markers and monotonic boundaries are
checked. Target wall spans the first through last captured target GPU node.

The table uses TP rank 0 throughout, so its stage means close to the measured
round. Other rank means are 36.52-36.54 ms; selecting the longest complete
same-rank interval per round gives 36.5414 ms. Stage percentiles are not additive.

| Nonoverlapping stage | Mean ms | p50 ms | p90 ms |
| --- | ---: | ---: | ---: |
| Metadata marker to target graph | 0.9242 | 0.8721 | 1.1201 |
| Target verification graph | **29.3167** | 29.1523 | 30.5007 |
| Target end to output gather | 0.0199 | 0.0195 | 0.0228 |
| Sampling/state and draft handoff | 0.9540 | 0.9543 | 0.9720 |
| Four drafts and proposal combine | **5.2779** | 5.2907 | 5.3176 |
| Next-round preparation | 0.0459 | 0.0456 | 0.0463 |
| Complete closed round | **36.5387** | 36.3604 | 37.5631 |

The draft interval starts at `_prepare_eagle_inputs_kernel`, before the first
draft forward: first draft/input work is 1.5651 ms and the remaining three drafts
plus combine are 3.7129 ms. Starting at `_prepare_eagle_decode_kernel` would
omit that first forward. Target sampling/state includes LM-head work and the
initial handoff before the draft marker.

Node capture perturbs timing, and the shorter request covers a different part
of the generation. **Do not replace 27.3963 ms with 36.5387 ms or rescale trace
stages into invented capture-disabled target/draft times.** The 29.3167-ms
verifier is a traced measurement only. Rank-0 target kernel busy-time union is
21.3554 ms; the remaining 7.9613 ms has no recorded kernel running. This residual
includes graph gaps and possible tracing/dependency effects; it is not a proven
CPU bottleneck or an independently removable latency budget.

## Where the GPU time goes

Service durations below can overlap and include collective dependency waits;
they do not add into the wall table. Rank maxima may come from different ranks.
There are 970368 graph-node kernel events across the four ranks' closed windows.

| Operator family | Mean round rank-max service ms | Calls/rank/round |
| --- | ---: | ---: |
| Dense BLAS/GEMM/GEMV and reductions | **12.8330** | 948 |
| TP communication, including waits | 4.8335 | 125 |
| Other elementwise/copy/index | 4.4813 | 1077 |
| Direct NVFP4 experts | 3.0191 | 96 |
| QSA/attention | 2.1782 | 112 |
| HC combine/gating | 1.3803 | 328 |
| GDN/convolution | 0.9740 | 109 |
| Draft MoE | 0.8374 | 8 |
| Routing/shared gate | 0.8181 | 103 |
| Metadata/state | 0.1866 | 53 |
| PLE pinned-UVA lookup | **0.1164** | 1 |

The hottest individual family is the CUTLASS SM70 FP16 16x16 WMMA GEMM:
304 calls/rank/round, 27.32 microseconds per launch, 8.07-8.54 ms of service
per rank. Rank-0 dense service splits into 9.31 ms in target/preparation,
2.51 ms in draft and 0.46 ms in sampling/state.

The trace proves dispatch, not merely enabled flags: each rank/round has
48 W13 split4 and 48 W2 split1 direct expert launches, 36 fused GDN projection
split kernels, 12 common FP16 row-GEMV calls, six native HC up/mix and six HC
down/all-gather calls. The latter GEMV/HC calls have nonzero graph-node IDs.
PLE is one local pinned gather (0.1144 ms on rank 0), already a small cost.

The next optimization should first attribute the dominant M=5 verifier GEMMs
to projection shapes/call sites using this retained trace, then test the
existing common operators' small-batch extension/fusion on those exact shapes.
Keep the common dispatch and numerical contract; do not clone the single-token
path or apply M=1 assumptions to M=5. Collective service is strongly skewed
across ranks (1.34-4.78 ms), so isolate readiness waits before blaming transfer
bandwidth. PLE and another metadata rewrite are not the leading candidates.

## Quality and retained evidence

Three natural responses use temperature 1.0, top-p 0.95, top-k 20, seed 20260828,
max 2048 tokens and normal EOS. All stop normally. Manual review finds correct
LIS, 240 km / 68.57 km/h arithmetic and negative integer-division answers;
five local LIS assertions pass. Output tokens and acceptance counters exactly
match the preceding common-stack run. This bounded screen does not replace
full quality evaluation. The earlier 9fab source passed HumanEval 0-7 (8/8)
and 467 affected tests; those are historical evidence, not rerun results for
this new integration. New focused coverage has 46 passes and one CUDA-role
policy skip, plus scoped lint/type checks.

The prior 73-75 tok/s no-MTP run used explicit CPU-worker RAM PLE and cannot
stand in for the historical 97.7-97.9 tok/s hybrid/pinned-UVA baseline. One
attempt at its 262144 capacity failed before generation: required KV 3.24 GiB,
available 1.25 GiB. Preserve `baseline_nomtp.log`; do not repeat that experiment
or claim a new matched 97 tok/s measurement. The owner explicitly narrowed
further work to shared-path integration, this baseline and its trace.

All local evidence is retained under:

```text
/home/ymzx/桌面/1cat-vllm/worktrees/v100-mtp4-full-defaults-20260926-103036/.artifacts/
```

- `mtp4_baseline_27_396ms.json`: accepted contract, source-patch/native hashes,
  raw artifact paths and hashes, metric definition and future comparison rule.
- `stacked_mtp_graph.json`, `_contract.json`, `_source.patch`, `.log`, `.exit`:
  endpoint text/tokens/stats, environment, measured source and clean exit.
- `stacked_mtp_graph_trace.nsys-rep` and `.sqlite`: raw graph-node capture.
- `stacked_mtp_graph_wall.json`, `_kernels.json`, `_target_busy.json`,
  `_summary.json`: closed intervals, service attribution and quality/parity.
- `run_probe.py`, `mtp_probe.py`, `analyze_stacked_trace.py`,
  `analyze_kernel_trace.py`: exact retained launcher, fixture and analysis.

The completed run command was
`.venv/bin/python .artifacts/run_probe.py stacked_mtp_graph --ple auto --ple-host-gib 12 --focused-trace`.
All task-owned model/profiler workers exited. The worktree is retained for
baseline/trace reuse; unrelated GPU workloads must not be stopped.
