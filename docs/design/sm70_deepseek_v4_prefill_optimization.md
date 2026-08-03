# SM70 DeepSeek V4 Prefill Optimization

## Scope

- Integration base: `48e89751b4b98c18e1be6506dca15f015155d068`
- Model: DeepSeek-V4-Flash, FP8 dense and MXFP4 routed experts
- Hardware: TP8 on eight V100-SXM2-32GB GPUs
- Quantization backend: TurboMind only; Marlin is out of scope
- KV cache: `fp8_ds_mla`
- Runtime: CUDA Graph enabled, no eager execution, no speculative decoding
- Initial workload: exactly 1024 input tokens and one naturally sampled output
- Sampling: official `temperature=1.0`, `top_p=1.0`

The first gate is a same-source, unprofiled single-request prefill baseline.
Report tokenized prompt length, request TTFT, worker prefill wall time, and
prompt throughput separately. Decode timing must not be included in the
prefill claim.

## Measurement Definitions

- Request TTFT is measured by the streaming OpenAI client from request start
  to the first non-empty text delta. Effective prompt throughput is
  `1024 / TTFT`; it includes frontend, scheduling, LM-head, sampling, and
  streaming overhead and is not pure GPU prefill throughput.
- The Nsight GPU envelope is the interval from the first to the last CUDA
  kernel on one rank during the captured request. The critical rank is the GPU
  with the longest envelope.
- GPU busy time is the interval union across streams. `envelope - busy` is the
  explicit GPU idle/dependency gap and closes the per-rank wall clock.
- Kernel and category service sums describe composition. They are kept
  separate from the envelope because concurrent streams can overlap.

## Measurement Order

1. Verify worker logs select DeepSeek V4, TurboMind FP8/MXFP4, SM70 sparse
   attention, FP8 MLA KV, TP8, and non-eager execution.
2. Run warm and cold unprofiled 1024-token prefill samples and retain raw JSON.
3. Capture one Nsight Systems request with CUDA Graph node and NVTX data.
4. Split prefill GPU service into FP8 dense, MXFP4 MoE, sparse/SWA attention,
   compressor and KV work, mHC, collectives, routing, and residual categories.
5. Use Nsight Compute only on a confirmed dominant kernel and preserve an
   explicit unattributed residual between GPU service and worker/request wall.
6. Admit an optimization only after an exact-shape microbenchmark and numerical
   oracle pass, then rerun the unprofiled endpoint and output-quality gates.

## Baseline

The task-owned endpoint used TP8, `fp8_ds_mla`, prefix caching disabled,
`max_model_len=2048`, `max_num_batched_tokens=2048`, no MTP, and CUDA Graph.
The first cold request took 11681.327 ms and compiled eight prefill Triton
kernels. It is retained as cold-start evidence and excluded from the warm
baseline. Two additional warmups measured 1652.083 and 1653.033 ms.

Five measured requests then produced:

| Seed | TTFT (ms) | Prompt tokens | Completion tokens |
| ---: | --------: | ------------: | ----------------: |
| 6101 | 1651.805 | 1024 | 1 |
| 6102 | 1651.730 | 1024 | 1 |
| 6103 | 1651.969 | 1024 | 1 |
| 6104 | 1654.750 | 1024 | 1 |
| 6105 | 1649.518 | 1024 | 1 |

The median is 1651.805 ms, the mean is 1651.954 ms, and the median effective
request-level prompt throughput is 619.93 token/s. All samples used official
`temperature=1.0` and `top_p=1.0` sampling. A separate 1024/64 smoke decoded
readable text but drifted into an HTML fragment near its token-limit stop, so
it is retained only as baseline text-health evidence, not as a full quality
pass.

Raw artifacts are retained under
`/home/fudanwl/v100-worktrees/runs/dsv4-prefill-trace-20260803/`.

## Nsight Systems Trace

After the cold JITs and two profiler-server warmups, one exact 1024/1 request
was captured with CUDA Graph node tracing. The profiled request had 1713.796
ms TTFT. Its critical rank was device 5:

| Same-request interval | Time (ms) | Share of request TTFT |
| --- | ---: | ---: |
| Critical-rank GPU envelope | 1686.097 | 98.38% |
| GPU busy interval union | 1541.559 | 89.95% |
| GPU idle/dependency gap | 144.538 | 8.43% |
| Request residual outside GPU envelope | 27.699 | 1.62% |

The second and third rows close the GPU envelope. The last row is only the
same-profile request residual; it must not be interpreted as a pure scheduler
measurement. Across all eight ranks, envelopes were tightly grouped from
1685.221 to 1686.097 ms.

Critical-rank kernel service was composed as follows. Percentages use total
kernel service as their denominator and therefore do not close the GPU wall
clock when streams overlap.

| Category | Service (ms) | Service share | Launches |
| --- | ---: | ---: | ---: |
| TurboMind MXFP4 MoE GEMM | 1128.102 | 72.69% | 22016 |
| SM70 sparse MLA/SWA attention | 212.538 | 13.69% | 43 |
| NCCL collectives | 64.355 | 4.15% | 88 |
| TurboMind FP8 dense GEMM | 48.321 | 3.11% | 258 |
| mHC | 43.237 | 2.79% | 259 |
| FP16/CUTLASS GEMM | 17.856 | 1.15% | 148 |
| KV compression/indexer/rope | 17.060 | 1.10% | 346 |
| MoE routing | 15.349 | 0.99% | 344 |
| All remaining categories | 5.147 | 0.33% | 574 |

The dominant MXFP4 service splits into exactly two repeated launch shapes:

| Stage | Service (ms) | Launches | Mean launch | Launch geometry |
| --- | ---: | ---: | ---: | --- |
| W13 gate/up, local N=512 | 886.889 | 11008 | 80.57 us | `grid=(49,4,2)`, 128 threads, 255 registers, 32784 B dynamic shared memory |
| W2 down, local N=4096 | 241.213 | 11008 | 21.91 us | `grid=(49,32,1)`, 128 threads, 255 registers, 32784 B dynamic shared memory |

The count is structural: 43 MoE layers times 256 experts times two stages is
22016 launches. `mxfp4_moe_dense_stage_sm70_out()` currently loops over every
expert and invokes TurboMind with `num_experts=1`. This launch decomposition,
not sparse attention, is the first prefill optimization target.

The first candidate uses four 64-expert TurboMind dispatches per stage instead
of 256 one-expert dispatches. PR #179 contained a default-off unvalidated sketch
of this direction. The sketch was ported to the current integration base, but
its unbounded 256-expert grouped call failed the numerical gate despite a large
speedup:

| Candidate | W13 legacy | W13 grouped | Speedup | Numerical result |
| --- | ---: | ---: | ---: | --- |
| 256 experts/launch | 29.66 ms | 1.299 ms | 22.83x | Rejected: max abs 0.00390625 |
| 256 experts/launch, legacy dispatch policy | 28.95 ms | 2.074 ms | 13.96x | Rejected: same max abs 0.00390625 |

The second failure used the exact same TurboMind kernel, split count, swizzle,
CTA shape, and stages as the legacy loop. The difference therefore comes from
the large grouped scheduler execution shape, not a quantization or dispatch
policy change. It affected 2645 of 3145728 W13 elements across all 256 experts,
with no sign flips or row-wise argmax changes. It is still rejected because
the full-model effect of repeating the difference across 43 layers is not
proven safe.

Sweeping the number of experts handled per grouped launch found a strict
bitwise boundary:

| Experts/launch | W13 grouped GPU median | Speedup | Cross-route result |
| ---: | ---: | ---: | --- |
| 1 | 28.273 ms | 1.06x | bitwise |
| 2 | 16.644 ms | 1.81x | bitwise |
| 4 | 9.767 ms | 2.92x | bitwise |
| 8 | 5.040 ms | 5.54x | bitwise |
| 16 | 2.557 ms | 11.00x | bitwise |
| 32 | 2.489 ms | 11.31x | bitwise |
| 64 | 2.464 ms | 11.51x | bitwise |
| 128 | 2.372 ms | 11.92x | rejected, max abs 0.00390625 |
| 256 | 2.081 ms | 13.56x | rejected, max abs 0.00390625 |

The candidate is therefore hard-limited to at most 64 experts per launch. At
that width the 1024-token stage benchmark passed cross-route and repeated-run
bitwise equality for W13 and W2, balanced routing and a half-active routing
stress, across seeds 29, 101, and 202. Balanced-route W13 speedup ranged from
10.14x to 12.40x; W2 ranged from 5.64x to 5.89x. The structural request count
becomes 43 layers times two stages times four launches, or 344 launches instead
of 22016.

The original projection put MXFP4 service near 123 ms. The matched endpoint and
post-candidate trace have now measured the result rather than relying on that
projection.

## Grouped-Prefill Candidate

The candidate keeps the route default-off, requires at least 6144 routed rows
(1024 tokens times top-k 6), and hard-clamps the group width to 64. This keeps
unvalidated high-concurrency decode shapes out of the prefill route. Worker
logs proved that a 1024-token API request reached the grouped C++ path with a
12288-row internal staging shape. Decode graph capture remained on the existing
dense-stage path with 12 rows and did not enter grouped prefill.

After a cold request and two warmups, five unprofiled requests produced:

| Seed | Candidate TTFT (ms) |
| ---: | ---: |
| 6601 | 567.607 |
| 6602 | 533.535 |
| 6603 | 531.873 |
| 6604 | 531.416 |
| 6605 | 568.867 |

The candidate median is 533.535 ms and the mean is 546.660 ms. Relative to the
same-contract 1651.805 ms baseline median, TTFT falls by 1118.270 ms or 67.70%,
a 3.096x speedup. Effective request-level prompt throughput rises from 619.93
to 1919.27 token/s. These are unprofiled endpoint measurements, not kernel
service projections.

The matched post-candidate Nsight request measured 596.478 ms TTFT. Device 5
was again the critical rank:

| Same-request interval | Time (ms) | Share of request TTFT |
| --- | ---: | ---: |
| Critical-rank GPU envelope | 572.791 | 96.03% |
| GPU busy interval union | 545.071 | 91.38% |
| GPU idle/dependency gap | 27.721 | 4.65% |
| Request residual outside GPU envelope | 23.687 | 3.97% |

The candidate busy interval and idle gap close the candidate GPU envelope.
Critical-rank envelopes ranged from 572.114 to 572.791 ms across all ranks.
Its service composition was:

| Category | Service (ms) | Service share | Launches |
| --- | ---: | ---: | ---: |
| SM70 sparse MLA/SWA attention | 228.628 | 41.18% | 43 |
| TurboMind MXFP4 MoE GEMM | 133.226 | 24.00% | 344 |
| TurboMind FP8 dense GEMM | 50.828 | 9.16% | 258 |
| mHC | 45.335 | 8.17% | 259 |
| NCCL collectives | 39.792 | 7.17% | 88 |
| KV compression/indexer/rope | 18.161 | 3.27% | 346 |
| FP16/CUTLASS GEMM | 17.989 | 3.24% | 148 |
| MoE routing | 15.729 | 2.83% | 344 |
| All remaining categories | 5.467 | 0.98% | 574 |

MXFP4 service therefore falls from 1128.102 to 133.226 ms, an 88.19%
reduction, while its launch count falls exactly 64x from 22016 to 344. W13
accounts for 86.319 ms and 172 launches; W2 accounts for 46.906 ms and 172
launches. The new first optimization target is the 228.628 ms sparse MLA/SWA
attention kernel, not further grouped-MoE launch reduction.

### Numerical And Text Gates

The operator oracle now covers the internal 2048-token execution shape seen in
the route log: 12288 routed rows, all 256 experts active, random non-uniform
expert counts, and the entire output buffer initialized with a sentinel. At
64 experts per launch, both W13 and W2 match the legacy path bit-for-bit,
including the full-buffer tail, and repeated grouped runs are bitwise stable.

The candidate also completed repeated official-sampling 1024/64 requests with
stable, readable text, no repeated-token collapse, and unchanged steady decode
latency. A separate greedy cross-process comparison shared its first 25 output
tokens with the default-off run and then selected a different valid
continuation. Because that comparison crossed server restarts, it is not a
valid operator-level numerical oracle; a second default-off restart intended
to measure restart variance was blocked when an unrelated TP8 service acquired
the GPUs. The feature therefore remains default-off until same-process model
output or logit equality is closed. No quality-pass claim is made from the
semantic smoke alone.

## Exact 8K Chunk Comparison

The next workload uses exactly 8192 prompt tokens, one officially sampled
output token, TP8, `fp8_ds_mla`, no prefix cache, no MTP, and non-eager
breakable CUDA Graph execution. The benchmark prompt builder preserves its
existing token prefix and repeats the already-tokenized context only when the
requested length exceeds the original 80-paragraph fixture.

The first 4096-token chunk request exposed a correctness bug before timing:
the SM70 indexer dequantization kernel has a `uint8` AOT signature and performs
software E4M3 decoding, but its runtime call passed the same storage with a
native `float8_e4m3fn` type. Triton rejects that native FP8 type on SM70 before
entering the kernel. Passing a zero-copy `uint8` view at the call boundary
preserves every value and scale. A direct GPU oracle covering weighted-Q,
software FP8 K dequantization, and FP16 HMMA was bitwise equal to the reference
with zero mismatched elements.

The endpoint comparison measured:

| Chunk | Grouped prefill | Median TTFT | Mean TTFT | Prompt throughput |
| ---: | :---: | ---: | ---: | ---: |
| 4096 | off | 7458.383 ms | 7457.075 ms | 1098.36 token/s |
| 4096 | group-64 | 3179.100 ms | 3178.883 ms | 2576.83 token/s |
| 8192 | group-64 | 2997.086 ms | 2997.686 ms | 2733.32 token/s |

Each row uses five measured requests after a cold request and two warmups. The
4096 rows used `gpu_memory_utilization=0.90`. An 8192-token profile run leaves
only 2.92 GiB for KV at that setting, below the 3.32 GiB startup admission
requirement for `max_model_len=10240`, so the 8192 row used 0.95 and retained
15,667 KV tokens. GPU memory utilization changes the allocated KV pool rather
than the active 8192-token execution, but a strict 0.95-versus-0.95 endpoint
repeat remains outstanding. Subject to that caveat, one 8192-token chunk is
5.73% faster than two 4096-token chunks.

The exact 8192-token, 49,152-routed-row operator gate passed before the model
run. Group-64 W13 measured 5.124 ms versus 54.956 ms legacy (10.72x); W2
measured 3.237 ms versus 14.504 ms (4.48x). Both stages were cross-route and
repeated-run bitwise equal over the complete output buffer.

An official-sampling 8192/64 request completed all 64 tokens with 20.330 ms
mean decode interval, but its continuation ended in an unrelated TypeScript
fragment. It is not a text-quality pass. The speed route remains experimental
until the same prompt and seed are compared with the 4096-token chunk path and
the broader model-quality investigation is closed.

Raw artifacts include:

- `baseline-c4096-seed820{1..5}-fixed-i8192-o1.json`
- `group64-c4096-seed840{1..5}-i8192-o1.json`
- `group64-c8192-u095-seed860{1..5}-i8192-o1.json`
- `microbench-grouped-prefill-chunk8192-random-b64.log`
- `group64-c8192-u095-quality-i8192-o64.json`

They are retained in
`/home/fudanwl/v100-worktrees/runs/dsv4-prefill-trace-20260803/`.

As a same-source preliminary reference, an earlier no-MTP run used runtime
source `6f946b603a`, the same TP8 and KV-cache configuration, and 256 output
tokens. Its exact 1024-token prompt had these request TTFT values:

| Seed | TTFT (ms) |
| ---: | --------: |
| 4201 | 1818.623 |
| 4202 | 1826.996 |
| 4203 | 1817.802 |

The median is 1818.623 ms, or 563.06 prompt tok/s at request level. This is a
reference rather than the final task-owned result because the requested
prefill contract uses one output token.

## Rejected Evidence

The earlier report
`dsv4-combined-latest-graphtrace-20260803/graph_node_combined_i1024_o64`
cannot serve as the full-prefill baseline. That server had prefix caching
enabled, and the captured 1024-token request repeated the warmup prompt. The
trace therefore measured a partial prefix hit: request TTFT was 1039.958 ms
and the critical-rank GPU envelope was 1019.709 ms. Its kernel data remains
useful only for validating the SQLite parser and category rules.

The prefix-cache diagnosis is independently visible in same-source request
results: a repeated 1024-token prompt with cache enabled measured about
932.83-933.25 ms TTFT, while the no-prefix request measured 1872.07 ms. The
new trace must therefore keep prefix caching disabled.

## Runtime Setup Notes

- A task-private empty TileLang cache exposed two startup prerequisites before
  performance measurement: set `TILELANG_TARGET=cuda` (resolved as `sm_70`)
  and point `CUDA_HOME` at the Conda CUDA 12.8 toolkit containing `nvcc`.
- Two attempted launches correctly aborted when another registered TP8 task
  acquired the GPUs first. These are ownership failures, not model startup or
  performance regressions, and their logs are retained with the raw artifacts.
