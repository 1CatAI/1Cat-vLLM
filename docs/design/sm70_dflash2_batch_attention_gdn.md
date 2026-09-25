# SM70 DFlash batch attention and verifier-state scheduling

The q8 verifier's C1-to-C4 cost increase is mostly compressed GEMM, but after
the M64 tuning its C4-to-C8 GEMM increment is only 1.05 ms. A same-service
September 25 trace instead found five draft attention launches per step
at C1, twenty at C4 and forty at C8. Each launch had only eight CTAs
and ran on the same stream. Kernel time was 0.862/3.255/6.451 ms. CUDA Graph
replay retained the per-request loop.

## Implementation

- Uniform noncausal DFlash queries now use the existing paged attention
  kernel's batch dimension. Query length is static during capture; sequence
  lengths and page tables remain live GPU inputs. The original per-sequence
  path handles nonuniform batches, anchored masks, diagnostic tensor dumps,
  and noncontiguous output buffers. Single-request dispatch is unchanged.
  This route is independent of weight quantization and supports the existing
  KV formats without a new native kernel or environment variable.
- For the measured TP4 q8 packed GDN shapes (4/8 requests, QH4/VH12,
  K=V=128), BV8 replaces BV32. The K reduction, gating, FP32 recurrence,
  accepted-state selectors and every speculative state snapshot are unchanged.
  Explicit legacy schedule overrides retain precedence. This is local to the
  packed verifier, so ordinary FlashQLA prefill/decode dispatch is unchanged.

## Workload and source

The SM70 Qwen3.8 DFlash2 automatic configuration now enables compressed batch
GEMM layouts and sets AWQ warmup, FP8 dense tuning and NVFP4 dense tuning
limits to M64. It preserves every explicit environment override, including
`VLLM_SM70_BATCH_GEMM_LAYOUTS=0`. The existing contract admits all target
quantizations and KV dtypes; individual operators check their own capabilities.
Batch layouts still require capacity for at least eight requests, and C1/C4
retain their existing small-M kernels. Joint draft attention and the guarded
packed GDN schedule activate automatically. The same configuration enables
the packed GDN entry, combined projection split, draft context pipeline/KV
graph, and quantized LM-head fallback used by the measured service. Explicit
overrides still win for these prerequisites. The experimental FP8 prescaled
layout remains off by default.

Integration base: `d49e32b3587d4d34ffccb0ffd376e63974b06c88`, `onecat/main`.
Owned branch: `codex/v100-dflash2-concurrent-decode-20260923-075451`.
Pre-change HEAD: `b8e05bcbe4586ec4a123e1349ceab0f2821a817a`.
Native extension SHA256:
`2884a59db00563d686a5dd7bbfbad57edb424b7d43b22a79bb468f299b56cef5`.
Inference changes in this follow-up are Python dispatch/JIT scheduling; the
installed source-built native extensions are unchanged. Experimental GEMM
sidecars described below are research-only and are not loaded by services.

Service contract: Qwen3.8-27B-NVFP4, FP16 execution, target E4M3 KV,
FP16 draft KV, Flash-V100, DFlash2 q7, V100-SXM2-32GB GPUs 4–7 TP4,
300 W, Torch 2.10.0+cu128, CUDA runtime 12.8, driver 580.173.02.
Maximum context 262144, memory fraction 0.8, max sequences 16, batched
tokens 8192, prefix caching/aligned Mamba cache, async scheduling and
chunked prefill. Batch GEMM layouts enabled, all three tuning/warmup limits
64, FP8 batch prescaling disabled and single-request tail graphs disabled.

Performance requests: 2048 input/256 output, temperature/top-p/top-k
0.7/0.8/20, seed 20260923+request index, forced output length. All endpoint
modes use the same shared-1792 dataset, SHA256
`1cea3c5dbbd22fde40ad08db21fae1065b74f994009882a37e89c16b82b3580e`.
Forced-length requests are not the natural-completion quality gate.

## Focused evidence

The attention screen uses q8, Hq8/Hkv2, D128, page16 and noncausal SWA
(2047,2047). CUDA Graph medians in microseconds per layer:

| Context | Batch | Per-request loop | Joint batch | Numerical result |
| --- | ---: | ---: | ---: | --- |
| 2048 | 1 | 198.11 | 198.08 | Bitwise equal |
| 2048 | 4 | 897.30 | 210.26 | Bitwise equal |
| 2048 | 8 | 1559.14 | 202.75 | Bitwise equal |
| 32768 | 1 | 177.53 | 177.53 | Bitwise equal |
| 32768 | 4 | 814.35 | 208.21 | Bitwise equal |
| 32768 | 8 | 1637.14 | 212.79 | Bitwise equal |

These include layout copies, unlike the earlier Nsight kernel-only numbers;
they are not complete draft or endpoint latency. Reproduction:

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH="$PWD:$PWD/flash-attention-v100" \
  uv run --no-project --python .venv/bin/python .venv/bin/python \
  benchmarks/kernels/benchmark_sm70_dflash_batched_prefill.py
```

The packed GDN screen (including gating) measured C4 BV32/BV8 at
59.90/43.62 us and C8 at 91.82–97.31/80.96 us per layer. Output and FP32
state were bitwise equal. The GDN regression checks every stored state,
untouched slots, heterogeneous acceptance selectors and repeated graph
replays for TP2/TP4, C1/C4/C8 and packed/strided projections: 24 tests pass.
Attention regression covers C1/C4/C8, q1/q4/q8, FP16/E4M3 KV,
2K/4K/32K/262K positions, page sizes 16/1648, changing live lengths and page
tables, and untouched output padding: 20 tests pass. The service's hybrid
cache selects page size 1648 despite the CLI's initial block-size=16.

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH="$PWD:$PWD/flash-attention-v100" \
  uv run --no-project --python .venv/bin/python .venv/bin/python -m pytest -q \
  tests/kernels/attention/test_sm70_dflash_batched_prefill.py \
  tests/kernels/test_sm70_dflash2_packed_gdn_fp32.py
```

## Rejected alternatives

- Existing FlashQLA DDTree state API with linear parents: C1/C4/C8 takes
  27.51/43.01/87.89 us versus 21.33/43.52/78.95 us for the selected packed
  schedule. Outputs differ by up to 1.91e-6 and FP32 state by up to 2.24e-8
  on the synthetic screen. This establishes neither a speed nor numerical
  reason to replace the selected recurrence. It does not rule out a new
  FlashQLA verifier implementation.
- FP8 QPN8 M32: four split-K phases reduce shared memory but regress the
  real TP4 projection shapes. Reducing K-loop unrolling lowers registers
  138→127, but M32 out/QKV show no useful gain; linear input saves only
  about 5.36 us. More residency by itself is not an adequate promotion gate.
- FP4 QPN2: new ordered split-K phases preserve the original summation while
  reusing decoded weights across 16/32 rows per CTA. All tested outputs are
  bitwise equal, but M32 gate/up regresses from about 145 to 157–161 us and
  down remains near 65 us. Do not retry unchanged. No GEMM source change from
  this screen is included.

Raw scripts, compiler resource logs, failed attempts and service evidence
are retained under the task-private artifact key
`verification/batch-followup-20260925/`. The local handoff records its
absolute location. The preceding same-service C1/C4/C8 trace is in adjacent
`scaling-audit-20260925/SCALING_REPORT.md` and `service.nsys-rep`.

## Endpoint validation

Three endpoint runs per cell, using C1/16, C4/32 and C8/48 requests:

| Batch | Control rolling decode tok/s | Candidate | Change | Acceptance control/candidate |
| --- | ---: | ---: | ---: | --- |
| C1 | 243.633 | 246.891 | +1.34% | 56.60% / 57.49% |
| C4 | 293.667 | 298.459 | +1.63% | 56.70% / 56.54% |
| C8 | 343.050 | 356.606 | +3.95% | 56.22% / 55.18% |

Rolling decode is the existing threaded SSE client's request-level capacity
estimate, `C * sum(output_tokens - 1) / sum(last_stream - first_stream)`.
It includes pauses during replacement prefill and is not GPU-only decode.
No observer or Nsight instrumentation was loaded for these service results.
Prefix hits were zero. C8 acceptance varied between runs (candidate
48.75–56.62%); the table reports medians, not identical token sequences.
C1 dispatch is unchanged, so its small rate difference is not attributed to
this patch.

Median complete output rates are C1 167.86→169.32, C4 235.78→240.57 and
C8 279.82→282.27 tok/s. Median TTFT is 0.478→0.479, 0.610→0.603 and
0.663→0.645 s. Median of each request's mean ITL is
3.995→3.998, 13.653→12.513 and 22.297→21.822 ms;
P90 of those request means is 5.050→4.584, 18.668→18.828 and
31.338→30.288 ms. These are not the distribution of individual token gaps.

One isolated wave per mode additionally estimated the common client decode
window: C4 418.12→494.03 tok/s, C8 746.67→784.79 tok/s. Token counts are
estimated from streamed-text retokenization, with at most one final token of
discrepancy. Do not label these single waves an exact pure-decode GPU counter
or a three-run admission result.

The same 16 GSM8K questions (indices 8–23), xhigh reasoning, seed 20260923,
temperature/top-p/top-k 0.7/0.8/20 and natural EOS score 15/16 on both
services. The same question is wrong. Control has one 4096-token length cap;
candidate stops naturally on all 16. Acceptance is 51.29→51.58%. This is a
relative regression check, not proof of general model quality.

All 44 focused GPU tests and targeted pre-commit hooks pass. Set
`MYPYPATH="$PWD/flash-attention-v100"` when running the source-tree mypy hook
so it resolves the bundled Flash-V100 exports. No production native extension
rebuild or task-private sidecar is needed for this follow-up.

## Complete q8 step and kernel trace

The candidate uses the same task-only CUDA-event observer as the preceding
September 25 scaling audit. Both select full-batch, no-prefill q8 steps and
discard the first three. The candidate has 36/35/32 samples per rank for
C1/C4/C8. These diagnostic measurements are separate from the uninstrumented
endpoint runs above. Rank-0 medians in milliseconds, previous → candidate:

| Batch | Target forward | Draft | Complete GPU step |
| --- | ---: | ---: | ---: |
| C1 | 14.055 → 14.065 | 4.296 → 4.270 | 20.000 → 20.371 |
| C4 | 31.253 → 30.365 | 9.322 → 6.760 | 45.861 → 42.255 |
| C8 | 37.648 → 36.863 | 13.490 → 7.309 | 57.810 → 50.744 |

Each number is the wall time for the whole concurrent batch, not a separate
serial cost paid by each request. Complete-step medians need not equal the
sum of phase medians. Per-step maximum across all four ranks, then median,
is 20.556/42.360/51.105 ms for candidate C1/C4/C8. C1 dispatch is unchanged;
the 0.371-ms rank-0 complete-step difference is retained rather than hidden.

Nsight CUDA Graph node traces classify six interior steps per batch and
rank. Rank-0 target kernel service sums (not wall latency) in milliseconds:

| Batch | FP4 GEMM | FP8 GEMM | GDN | TP communication | Target attention | Other | Sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 | 4.537 | 3.968 | 1.481 | 1.373 | 0.638 | 2.188 | 14.184 |
| C4 | 11.238 | 9.040 | 2.661 | 3.152 | 1.774 | 2.482 | 30.347 |
| C8 | 12.451 | 8.977 | 4.474 | 4.441 | 3.281 | 3.286 | 36.911 |

Draft paged attention now takes five launches at every batch size. Its
grid is (1,1,8), (1,1,32), and (1,1,64) at C1/C4/C8, replacing the old
five/twenty/forty launches of grid (1,1,8). This confirms concurrent
requests share a kernel launch. Attention service time is now
0.859/0.876/0.910 ms, versus 0.862/3.255/6.451 ms previously. Successive
draft layers still depend on each other; this is parallelism across
requests, not parallel execution of dependent layers. Draft projection
GEMMs were already batched.

Target GDN service falls from 3.689 to 2.661 ms at C4 and 5.345 to
4.474 ms at C8. Compressed target GEMM remains essentially unchanged and
accounts for 66.8%/58.1% of C4/C8 target kernel service. C1→C4 GEMM still
grows from 8.506 to 20.278 ms. Existing QPN8 M32 already reuses a decoded
weight tile across 32 rows; it does not launch four independent GEMMs.
Its larger accumulator set and two ordered K phases increase register and
shared-memory pressure. QPN2 M32 instead uses separate 8-row CTAs, with
repeated weight loading/decode across those CTAs. Increasing reuse must be
balanced against accumulator residency and data delivery; the rejected
real-weight screens above demonstrate that increasing reuse alone is not
sufficient.

Artifacts are `candidate-trace/service.nsys-rep`, `service.sqlite`,
`phases-rank{0,1,2,3}.jsonl`, `comparison.json` and `comparison-summary.json`
under the task-private key above. `summarize.py` identifies each target
graph by its launch correlation ID; it does not sum all four GPUs into a
purported step latency. Nsight 2022 required explicit QDSTRM import using
its bundled importer; the capture itself completed successfully.

The C1→C4 GEMM problem remains open. The modest rolling gains do not meet
the requested 5%-ahead-PRO target, and no new PRO or 35B-A3B AWQ/FP8
model speed admission was performed.
