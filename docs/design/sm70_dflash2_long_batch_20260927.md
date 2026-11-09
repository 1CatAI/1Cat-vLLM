# DFlash2 concurrent long-context attention, 2026-09-27

This follow-up extends the existing single-request SM70 E4M3 long-attention
kernel to request-major q8 batches. It is incremental to `818a2bab6002ee02cb9ff6858b4917b30445b785`
on `codex/v100-decode-round-20260926-111504`, within PR #697. It does not
supersede the earlier batch sampler/context acceptance failures documented in
[the preceding audit](sm70_dflash2_batch_latency_20260926.md).

## Why the single-request optimization was missing from concurrent decode

The native entrypoint admitted only B1; although its kernels already had a
request grid dimension, per-row sequence lengths and the FP32 max/sum workspace
were not offset for batched requests. The Python wrapper likewise required one
request. In addition, the graph selector accepted only one CPU upper-bound
value, so expanding capture alone would still have selected the ordinary graph
for live concurrent requests.

The existing full-q8 kernel now admits B2–B16, using independent request-major
workspace panels. The graph manager captures matching existing q8 batch shapes
and checks every live request's CPU length hint before selecting a bounded
graph. Padded requests and zero-length rows remain safe. B1 q2–q8 and the existing
scalar-tail route retain their behavior; batched q1–q7 retain the ordinary route.
The loaded extension reports its batch capacity, so stale B1 binaries cannot
be passed batched workspaces.

This is the same built-in default path, with no new enable variable, model-name
check or weight-quantization check. E4M3 KV, FP16 queries, D=256 and GQA=6 are the
kernel's numerical/shape contract. Multiple six-query-head groups are supported.
The served context limit still bounds capture, up to 262144. Existing explicit
disable overrides remain available.
Additional batch graphs are only captured for that dtype/head geometry. The
35B-A3B TP4 geometry (four local query heads, one replicated KV head), FP16 KV,
E5M2 KV and other head dimensions retain their existing graph set. This is a
dispatch regression check, not a new 35B AWQ/FP8 speed baseline.

Workspaces are shared across compatible batch sizes on the same device, stream
and context contract. Descending graph capture allocates the largest needed
request panel once; a smaller deployment does not reserve a B16 panel. If an
eager call or capture grows the bank, older panels stay alive because earlier
graphs can still reference them. Different streams never share a panel.

## Operator validation

- Torch 2.10.0+cu128, CUDA 12.8, V100-SXM2-32GB.
- `tests/v1/worker/test_sm70_long_attention_graphs.py` plus
  `tests/kernels/attention/test_sm70_long_attention_batch.py`: **41 passed**.
- CUDA Graph replay changes Q, each request's length, padding, individual zero
  rows and all-zero batches. Batched results equal independent single-request
  execution bitwise. An independent FP64 oracle checks the longest and shortest
  requests, including multiple KV heads.
- GPU coverage: B2/B4/B8/B16; 3297, 8192, 32768, 131072 and 262144 contexts;
  pages 1024/1536/1648/3296; interleaved and eight-byte-only aligned KV strides.
- Selector tests cover full per-request CPU hints, oversized peers, padded
  captures, served-window limits and stale native libraries.
- Additional graph tests capture growing and descending batch sizes, then
  replay older graphs after reusing/growing the workspace bank. All outputs
  remain bitwise equal to independent-request execution.
- Incremental normal CMake FA2 build succeeds. The changed kernel reports no
  spills. No task-only native overlay or `LD_PRELOAD` is used.
- After integrating main `b034648012244ab712df05b93e6d8fff877a6f2f`, the long
  operator/graph tests plus the pre-Ampere and quantized-draft context regression
  tests give **50 passed**. A separate CPU-only run including the expanded
  dtype/head-geometry admission matrix gives **49 passed**; these suites overlap.

Commands (from the owned worktree):

```bash
uv run --no-project --python .venv/bin/python .venv/bin/python -m pytest -q \
  tests/v1/worker/test_sm70_long_attention_graphs.py \
  tests/kernels/attention/test_sm70_long_attention_batch.py

CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$PWD:$PWD/flash-attention-v100 \
  uv run --no-project --python .venv/bin/python .venv/bin/python \
  benchmarks/kernels/benchmark_sm70_long_attention_batch.py \
  --contexts 32768 131072 --batches 4 8 --page 1648 \
  --output /path/to/long-batch-page1648-bench.json
```

## Attention-only measurements

Compare the ordinary Flash-V100 request-major kernel with the shipped optimized
full-q8 kernel, using 16 independent per-layer KV working sets. Each result is
the median of five alternating sample pairs, with eight replays per sample.
This excludes model GEMMs, GDN, TP communication, draft and sampling. All compared
outputs are bitwise equal in these fixtures. These are **not endpoint gains**.

Page size 1648, CUDA Graph replay:

| Context | Concurrent requests | Ordinary, ms | Optimized, ms | Attention latency reduction |
| --- | ---: | ---: | ---: | ---: |
| 32768 | 4 | 13.202 | 7.537 | 42.91% |
| 32768 | 8 | 26.009 | 14.783 | 43.16% |
| 131072 | 4 | 48.390 | 26.295 | 45.66% |
| 131072 | 8 | 96.392 | 52.244 | 45.80% |

A page-3296 sweep also covers 8K/32K/128K/256K and B1/B2/B4/B8. C8 drops
34.4%/43.3%/45.9%/46.4% respectively. The B1 comparison in that microbenchmark
uses the ordinary kernel versus the already-existing optimized B1 kernel; it
is not a new C1 gain from this change.

The interface selects a 1648-token base block. Final grouped verifier logs in
the endpoint use page 3296 after hybrid-cache grouping, so both page sweeps are
retained; the grouped verifier's actual C8/32K micro result is 25.886 → 14.687 ms.

## Matched endpoint gate

Control is archived
`818a2bab6002ee02cb9ff6858b4917b30445b785` with the frozen pre-change FA2 binary;
candidate changes only the long-attention implementation and admission above.
Both use GPUs 0–3, TP4, Qwen3.8-27B-NVFP4, FP16 execution, E4M3 KV, 262144 service
capacity, 8192 chunk size, max sequences 16, prefix caching, DFlash2 q7 and
sampling temperature 0.7/top-p 0.8/top-k 20 with fixed per-request seeds.

The fixed performance fixture is 32768 input/256 output, a shared technical
report prefix and distinct request suffixes. It uses forced output length only
for timing. Three repetitions each of C1/C4/C8 are admitted as one wave, with no
later arrivals. Throughput counts actual returned token IDs between the last
request's first token and the first request's last token. This is a **client
all-live decode window**, not synchronized worker-step throughput, and it
excludes initial prefill. TTFT and acceptance counters are recorded separately.
A separate eight-request long-document retrieval check uses natural EOS and
unforced output length (31692 input tokens per request).

Initial service pair, before sharing workspaces between batch graphs: each
concurrency has one separately recorded first-use warmup and three measured
repeats. The first C4 run exposed lazy Triton compilation; it is excluded in
both arms, and an additional control repeat supplies three warm samples.

| Concurrency | Control decode tok/s | Candidate decode tok/s | Gain | Aggregate acceptance, control → candidate |
| --- | ---: | ---: | ---: | ---: |
| C1 | 159.300 | 159.793 | +0.31% | 33.395% → 33.395% |
| C4 | 326.832 | 411.724 | +25.97% | 38.961% → 39.369% |
| C8 | 492.025 | 612.624 | +24.51% | 47.904% → 50.078% |

Median TTFT is 1.223/3.322/5.229 s for control and 1.225/3.322/5.207 s for
candidate. Prefix hits are identical (29664 of 32768 input tokens per request).
Both natural-EOS retrieval runs score **8/8 correct and 8/8 normal stops**.
C1 outputs and acceptance match exactly. C4/C8 acceptance and outputs vary
between repeated waves even in the control; these are emitted-token endpoint
gains, not a claim that all of the gain is isolated kernel compute savings.

The initial candidate increases the actual graph pool from 0.88 to 1.13 GiB.
Keep that first pair in `long-endpoint-initial-comparison.json`; it isolates the
long-attention port before the main integration.

The final service uses workspace sharing and main integration at
`7e80145857c7401701924836879c5b0f17094a87`. It has the same workload and launch
options, with one first-use warmup and three measured repeats per concurrency:

| Concurrency | Control decode tok/s | Final decode tok/s | Gain | Aggregate acceptance, control → final |
| --- | ---: | ---: | ---: | ---: |
| C1 | 159.300 | 159.324 | +0.02% | 33.395% → 33.395% |
| C4 | 326.832 | 429.134 | +31.30% | 38.961% → 41.889% |
| C8 | 492.025 | 624.863 | +27.00% | 47.904% → 53.968% |

The final C4 samples are 427.940/429.134/429.568 tok/s; C8 samples are
624.863/667.761/543.966 tok/s. Acceptance contributes to this variability, so
these gains must not be described as an equal-per-step compute comparison.
Request mean-TPOT medians (which can include peers' initial prefill after one
request has started decoding) are 6.277/13.355/24.432 → 6.277/10.982/21.796 ms.
TTFT medians are 1.223/3.322/5.229 → 1.226/3.323/5.208 s. This experiment is not a
rolling-arrival benchmark or a new PRO 6000 comparison.

The final natural-EOS run is again **8/8 correct and 8/8 normal stops**. C1
tokens and acceptance remain exact. All four TP ranks log actual C4 and C8
bounded-graph selection during generation, in addition to native route logs
during capture. The subsequent dtype/head-geometry admission guard preserves
this measured configuration and is covered by the CPU matrix above.

Graph pool consumption falls to **0.99 GiB**: about 0.14 GiB below the first
candidate, leaving about 0.11 GiB above control. Service maximum length remains
262144 at memory utilization 0.8. Record the separate pre-capture KV budget
difference as well: control reports 11.45 GiB / 1050885 tokens, both candidates
11.00 GiB / 1009312 tokens. That difference is present before long graphs are
captured and has not been isolated; do not claim unchanged total KV capacity
or four independent simultaneous 262K requests. Whole-service speed/quality was
measured at 32K; 128K/256K evidence here is operator validation and timing.

On 2026-09-27, the project owner requested merging PR #697 after disclosure of
the earlier rolling-decode acceptance failure and the KV-budget difference.
Both remain follow-up items; the merge decision does not mark those gates as
passed. These results qualify this incremental long-context route, not every
older change in that PR or the historical 35B performance target. Supported
routes remain enabled by the existing capability checks without new switches.

The first control startup failed because the systemd environment lacked Ninja
on PATH during the existing FlashQLA JIT build. The launcher now includes the
declared Python environment and CUDA toolkit binaries. This was a runtime
setup failure, not evidence about the new attention kernel; the failed log is
retained. Public API units and tunnel gateway remain stopped throughout.
The initial quality client also failed locally on a tokenizer `BatchEncoding`
being passed to JSON; converting its `input_ids` fixed the request construction.
The completed quality runs above are from the corrected client.

## Build and retained artifacts

Artifacts are retained under the isolated task root recorded in its local
`CURRENT.md`. Paths below are relative to that root.

- Native source SHA256:
  `e98db3a38dfae231ec164c3e14c9e755f362af9c089dcc36331b32b7b8d8e55f`.
- Candidate `_vllm_fa2_C.abi3.so` SHA256:
  `3e274c8da9f921b32e1318d17b9aa14af47a82b72a9c9e95498fa72b85c12187`.
- Control FA2 SHA256:
  `e0e712d3ac341b60fc91349227639857b3cbd328628be61357b205cf208e9298`.
- The unchanged FlashQLA source was compiled for the declared Torch/CUDA and
  installed in its normal `flash_qla/ops/gated_delta_rule/chunk/sm70` package
  location before candidate startup (no prebuilt-path override). SHA256:
  `833acf54ca22218771d5ee0a6b32be7cd93d3c951b9075da0d09744e30a20527`.
- Normal dynamic dependencies: Torch, CUDA, cuBLAS and system C++ runtime;
  no private task-cache library or RPATH/RUNPATH overlay.
- `results/long-batch-fa2-build.log`, `results/long-batch-quality-tests.log`.
- `results/long-batch-pooled-quality-tests.log`,
  `results/long-batch-main-tests.log`, `results/long-batch-admission-tests.log`.
- `results/long-batch-kernel-bench.json` and
  `results/long-batch-page1648-bench.json`: raw timing arrays and library hashes.
- `results/long-prompts-32k.json`: shared tokenized performance dataset.
- `results/long-service-{control,candidate,pooled}/`: every raw client wave,
  token ID, timestamp, speculative counter and natural-output result.
- `results/long-endpoint-pooled-comparison.json`: final three-repeat medians.
- `results/long-{control,candidate,pooled}-service.log`: configuration, route
  evidence and memory accounting; matching `*-idle-gpu-memory.csv` snapshots.
- `run-long-service.sh`, `long-eval.py`, `endpoint_barrier_client.py`:
  exact retained endpoint launch and measurement commands.
- `results/long-control-service-missing-ninja.log`: rejected startup attempt.
