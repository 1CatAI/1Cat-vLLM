# SM70 AWQ Exact-M5 Batched-GEMV

## Status

This document records the benchmark-only TensorRT-style exact-`M=5` AWQ
batched-GEMV experiment for Qwen3.6-27B-AWQ, TP2, on V100/SM70. It is kept
separate from the target-verifier overview so implementation and failed-path
details remain searchable.

Current decision as of 2026-07-11:

- the dataflow has a real performance benefit on three of the four projection
  shapes;
- pure FP16 accumulation improves the all-real 236-call microbenchmark by
  `10.60%` over two paired baseline/candidate rounds;
- a shape hybrid that retains TurboMind for MLP down reaches `9.8299 ms`, or
  `13.34%` below the paired baseline mean;
- neither result clears the `15%` first decision gate;
- FP16 and FP32-partial variants do not satisfy the projection-output quality
  gate;
- full FP32 accumulation nearly matches TurboMind output but is much slower;
- therefore no production dispatch or default was changed.

The quality-preserving successor is now implemented and accepted. It keeps
TurboMind FP32-accumulating HMMA, adds `CTA_N=32/64`, and fixes the A shared
layout. Its separate record is
`docs/design/sm70_awq_small_n_hmma_operator.md`; the accepted all-real result is
`9.7688 ms` with 236/236 bitwise-equal outputs.

This route is TurboMind-adjacent AWQ operator research. It does not use Marlin.

## Scope And Acceptance Contract

The fixed rank-local call mix is:

| projection family | shape | calls/rank |
|---|---:|---:|
| merged MLP gate/up | `5x17408x5120` | 63 |
| MLP down | `5x5120x8704` | 63 |
| linear-attention qkv/z | `5x8192x5120` | 47 |
| all attention output projections | `5x5120x3072` | 63 |

The experiment must use all 236 real checkpoint weights for the accepted
aggregate number. Representative-layer timing is only for shape attribution.
The first performance gate is at least `15%` aggregate improvement with no
shape regression. Projection output comparison precedes any full-model quality
run; eventual official long-output acceptance loss may not exceed `2%`.

## Implementation

The prototype is intentionally isolated from the production extension:

- CUDA source:
  `benchmarks/csrc/sm70_awq_m5_batched_gemv.cu`
- harness:
  `benchmarks/benchmark_sm70_awq_verifier_micro.py`
- CLI selector:
  `--m5-batched-gemv off|fp16|fp16-hybrid|fp32|fp32-full`
- JIT namespace: `sm70_awq_m5_micro`

`--op-library` lets the harness load the archived fixed TurboMind baseline
without replacing the worktree `vllm/_C.abi3.so`. This was necessary because
the current worktree binary had moved to a different experiment.

### Prepared layout

The checkpoint tensors have the AWQ contract:

```text
qweight: [K, N/8] int32
qzeros:  [K/128, N/8] int32
scales:  [K/128, N] fp16
weight = (uint4 - zero) * scale
```

The offline prepare kernel converts weights to the Turing-style K64,
interleave-4 column stream. It can be viewed as packed uint4 `[N/4, 4K]`.
Each aligned 128-bit runtime load contains 32 K values for one output-column
stream. Zero points are unpacked once and stored as FP16
`zero_bias = -zero * scale`, matching TurboMind's FP16 dequantization form.

The production `tm_weight` and `tm_scales` cannot be reused: they are in the
SM70 HMMA-884 operand and statistic layouts, not a linear column stream.

### Exact-M5 execution map

The kernel uses the TensorRT-LLM Turing family parameters:

```text
CtaM       = 5
CtaN       = 4 logical column streams
Interleave = 4
Threads    = 128
TileK      = 64
StepK      = 32/thread
CtaK       = 1024 physical K values
```

One CTA therefore emits 5 rows by 16 physical output columns. On SM70 the
launch counts are:

| N | output CTAs | current TurboMind CTA example |
|---:|---:|---:|
| 17408 | 1088 | 68 for the fixed gate/up kernel |
| 8192 | 512 | shape/config dependent |
| 5120 | 320 | shape/config dependent |

The earlier `17408/4=4352` estimate applies to an interleave-1 column-major
variant, not the SM70/Turing interleave-4 layout. The implemented route exposes
16 times more gate/up CTAs than the current 68-CTA kernel while reading each W4
value once and reusing it across all five rows.

Within a CTA, four warps cover one 1024-K segment. Each thread computes four
column streams for all five rows with `half2` FMA. Warp reductions use XOR
offsets `16, 8, 1`; warp partials are combined in FP32 shared memory before the
FP16 output store.

### Accumulation variants

| CLI mode | accumulation | purpose |
|---|---|---|
| `fp16` | persistent `half2` across K, then FP32 warp/CTA reduction | TensorRT-style speed ceiling |
| `fp16-hybrid` | `fp16`, except `5x5120x8704` uses TurboMind | shape-tactic ceiling |
| `fp32` | 32-K `half2` dot products, accumulated in FP32 per 1024-K segment | lower-drift partial variant |
| `fp32-full` | every dequantized FP16 weight is accumulated with scalar FP32 FMA | numerical control |

Compiled SM70 resource use is 88 registers/thread and 1280 bytes shared memory
for `fp16`; both FP32 variants use 128 registers/thread and 1280 bytes shared.

## Benchmark Contract

Environment:

```text
GPU: Tesla PG503-216 / V100 / SM70
physical GPU: 2
Torch: 2.10.0+cu128
model: /home/ymzx/models/Qwen3.6-27B-AWQ
TP: 2, rank: 0
M: 5
group size: 128
baseline library:
  bench_results/awq_m5_structural_20260710/warp8_candidate/
  _C.baseline_fixed.abi3.so
```

Representative shapes use `--batch-repeats 20`; single-call event/synchronize
samples let the V100 downclock between samples and are not comparable. The
accepted aggregate queues all 236 calls inside one event interval, with five
warmups and 100 measured iterations.

## Results

### Representative shape timing

All values are mean microseconds per call on the same GPU. Negative delta is
faster than the fixed TurboMind baseline.

| shape/family | TurboMind | FP16 M5 | delta | FP32 partial | delta |
|---|---:|---:|---:|---:|---:|
| gate/up `5x17408x5120` | 88.620 | 71.705 | -19.09% | 82.446 | -6.97% |
| down `5x5120x8704` | 39.550 | 44.798 | +13.27% | 52.590 | +32.97% |
| qkv/z `5x8192x5120` | 46.362 | 39.493 | -14.82% | 41.567 | -10.34% |
| linear-attn out `5x5120x3072` | 24.644 | 18.685 | -24.18% | 19.839 | -19.50% |
| full-attn out `5x5120x3072` | 24.077 | 18.833 | -21.78% | 19.779 | -17.85% |

The down projection is the only shape where the existing HMMA kernel is
decisively better. A production-quality implementation would require
shape-by-shape tactic selection rather than globally replacing TurboMind.

### NCU bottleneck localization

Full Nsight Compute reports were collected on 2026-07-11 for the real layer-1
weights. NCU locks/replays the kernel, so its duration is used only for
same-report attribution and not as the accepted microbenchmark latency.

Gate/up comparison:

| metric | fixed TurboMind | M5 FP16 | M5 FP32 partial |
|---|---:|---:|---:|
| NCU duration | 99.456 us | 85.056 us | 88.704 us |
| DRAM throughput | 481.3 GB/s | 576.3 GB/s | 552.5 GB/s |
| DRAM peak percentage | 43.24% | 51.44% | 48.95% |
| SM throughput | 31.22% | 63.38% | 62.26% |
| Tensor-pipe active | 16.77% | 0% | 0% |
| registers/thread | 159 | 88 | 128 |
| achieved occupancy | 6.25% | 28.53% | 23.17% |
| scheduler cycles with no eligible warp | 65.29% | 46.53% | 46.23% |
| shared-memory bank conflicts | 2,096,576 | 1,800 | 1,753 |

The stock `8x256x64` TurboMind gate/up kernel launches only 68 CTAs for 72 SMs.
It therefore has one four-warp CTA on each active SM, leaves four SMs idle,
uses 159 registers/thread and 20.51 KiB dynamic shared memory, and reaches only
6.25% achieved occupancy. Its shared loads average a 4.8-way conflict. This is
why neither HBM nor HMMA reaches its device ceiling: the primary baseline issue
is insufficient M=5 grid/residency plus shared-memory serialization, not peak
FLOP or HBM capacity.

The other fixed TurboMind tactics have the same resource footprint and do not
remove the structural issue:

| shape | fixed split | CTA grid | NCU duration | DRAM | Tensor pipe | occupancy | issue active | no eligible | shared bank conflicts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gate/up | 1 | 68 | 99.456 us | 481.3 GB/s | 16.77% | 6.25% | 34.71% | 65.29% | 2,096,576 |
| down | 7 | 140 | 50.624 us | 478.6 GB/s | 16.44% | 11.41% | 36.13% | 63.87% | 1,061,103 |
| qkv/z | 2 | 64 | 57.024 us | 399.8 GB/s | 14.82% | 6.24% | 31.32% | 68.68% | 990,208 |
| out | 3 | 60 | 32.832 us | 269.6 GB/s | 10.94% | 6.22% | 24.25% | 75.75% | 375,360 |

Gate/up, qkv/z, and out do not even launch one CTA per SM. Down launches almost
two CTAs/SM after split-K but still exposes only about 1.84 active warps per
scheduler. The production-quality baseline is therefore latency/parallelism
bound across the complete shape mix, not only on gate/up.

The exact-M5 FP16 kernel removes that first bottleneck with 1,088 output-column
CTAs and almost eliminates shared-memory conflicts, but exposes a different
limit:

| shape | NCU duration | DRAM | DRAM peak | SM peak | occupancy | issue active | long scoreboard | FP16 math throttle | waves/SM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gate/up `5x17408x5120` | 85.056 us | 576.3 GB/s | 51.44% | 63.38% | 28.53% | 53.47% | 25.53% | 15.60% | 3.02 |
| down `5x5120x8704` | 54.304 us | 464.2 GB/s | 42.39% | 51.56% | 25.29% | 44.67% | 31.7% | 10.8% | 0.89 |
| qkv/z `5x8192x5120` | 49.920 us | 477.4 GB/s | 43.19% | 51.69% | 26.22% | 46.54% | 27.1% | 13.1% | 1.42 |
| out `5x5120x3072` | 25.600 us | 372.1 GB/s | 34.06% | 36.82% | 25.56% | 34.93% | 28.5% | 9.0% | 0.89 |

Across every shape, `long scoreboard` is the largest sampled stall. On gate/up,
744 of 910 long-scoreboard samples are attributed to the first `LOP3` that
consumes a newly loaded packed-weight word. The load itself is coalesced
`LDG.E.128`; the problem is insufficient load-to-use overlap. The 88-register
footprint limits the FP16 kernel to five CTAs/SM (31.25% theoretical occupancy),
leaving too few independent warps to hide that latency. The compiler also keeps
many packed/converted columns live and emits runtime integer division for the
fixed group size.

The reported 63% compute utilization is not Tensor Core GEMM utilization. The
candidate executes no HMMA instructions; it is FP16-pipe work from uint4
conversion, dequantization, and dependent `HFMA2` chains. The generic NCU
19.7-byte/sector warning is secondary: source counters show no excessive L2
sectors on the 128-bit packed-weight loads, and measured DRAM reads are close to
the unavoidable W4 payload plus metadata.

There is still avoidable on-chip load work. Four interleave lanes reload the
same activation K segment, so each 1024-K tile issues about 40 KiB of activation
load instructions for 10 KiB of unique FP16 activation data. Scale/zero metadata
has a similar four-way thread-group reread. Cache/coalescer reuse prevents these
copies from becoming extra HBM traffic, but they consume LSU, L1, issue slots,
and registers. They are a secondary target after the packed-weight dependency.
The down projection also has a final half-full K tile (`8704 % 1024 = 512`), in
which only two of four warps remain active and latency hiding drops further.

The resulting bottleneck order is:

1. packed-weight load-to-first-use latency (`long scoreboard`);
2. register-limited latency hiding;
3. FP16 conversion/dequant/accumulation pipe pressure and dependency chains;
4. redundant cached activation/metadata loads and address work;
5. per-shape wave tails and fixed launch/drain cost, especially for the smaller
   qkv/out grids.

This does not make the FP16 path acceptable: it only explains its performance
ceiling. A quality-preserving successor must retain FP32 accumulation and the
baseline K reduction order while creating more N-direction work. It should not
repeat the already rejected uniform `8x128x32`, `TG=1x8x1`, `TG_K=4`, or changed
split-K routes. The new structural branch is a finer-N SM70 HMMA kernel (below
the currently registered CTA-N=128 floor) or an equivalent exact-M5 schedule
that uses `mma.sync.m8n8k4.f32.f16.f16.f32`, software-pipelines packed weights,
and controls live ranges before any production integration.

### All-real 236-call aggregate

| route | mean | p50 | p90 | delta vs paired baseline |
|---|---:|---:|---:|---:|
| fixed TurboMind round 1 | 11.3419 ms | 11.3388 | 11.3603 | reference |
| fixed TurboMind round 2 | 11.3445 ms | 11.3403 | 11.3725 | reference |
| FP16 M5 round 1 | 10.1336 ms | 10.1366 | 10.1806 | -10.65% |
| FP16 M5 round 2 | 10.1470 ms | 10.1586 | 10.2011 | -10.55% |
| FP32 partial | 10.7853 ms | 10.7909 | 10.8452 | -4.92% |
| FP16 hybrid, down on TurboMind | 9.8299 ms | 9.8232 | 9.8888 | -13.34% |

The two-round means are `11.3432 ms` for TurboMind and `10.1403 ms` for pure
FP16 M5, a stable `10.60%` improvement. The hybrid is the fastest measured
route but still misses the `<=9.6417 ms` value corresponding to the `15%` gate.
This is a non-graph microbenchmark ratio, not a replacement for the calibrated
`10.181 ms` production AWQ bucket. Even an optimistic direct transfer of the
hybrid ratio would yield about `8.823 ms`, still above the `7.184 ms` target and
still subject to the unresolved numerical-quality failure.

### Projection-output comparison

All-real values compare one output from every real layer against the archived
fixed TurboMind baseline with identical deterministic inputs.

| route | exact elements | max abs diff | weighted mean abs diff | sign mismatches |
|---|---:|---:|---:|---:|
| FP16 M5 | 12.48% | 0.0390625 | 0.0007424 | 5768 |
| FP32 partial | 24.05% | 0.0156250 | 0.0003491 | 2861 |
| FP16 hybrid | 26.19% | 0.0390625 | 0.0005809 | 4545 |
| full FP32, representative shapes only | 99.376% | 0.0009766 | 4.87e-7 | 0 |

The full FP32 control costs `16.9706 ms` on the representative weighted mix,
versus `11.7972 ms` for TurboMind. It proves the prepared layout and AWQ math
are correct; the remaining difference is accumulation order. It is not a
performance candidate.

## Decision

This experiment proves that output-column exact-M parallelism is useful on
V100. It does not prove that the current TensorRT FP16 accumulation semantics
are quality-safe for this model.

Closed conclusions:

1. Do not replace all four shapes with the FP16 GEMV. It misses the aggregate
   gate, regresses MLP down, and has material projection-output drift.
2. Do not continue the full scalar FP32 path. Its numerical result is good but
   it is slower than TurboMind.
3. Do not report `9.8299 ms` as production AWQ latency. It is a benchmark-only
   shape hybrid and still uses the unsafe FP16 accumulation on 173 calls.
4. Do not repeat the old M=1 NVFP4 raw-GEMV conclusion here. This M=5 AWQ
   implementation reuses each W4 value over five rows and does obtain a real
   speedup.

The next valid branches are:

1. Keep TurboMind for `5x5120x8704`; optimize only gate/up, qkv/z, and output
   projections.
2. Find an accumulation scheme between persistent FP16 and 128-register FP32
   partials. It must materially reduce sign changes without losing the FP16
   kernel's occupancy.
3. Profile the FP16 gate/up kernel with NCU before changing CTA geometry. The
   first questions are achieved HBM bandwidth, half2 pipe utilization, and
   whether activation rereads or register dependencies now limit the 71-us
   result.
4. Only after projection drift is controlled, integrate an environment-gated
   production prepare/dispatch path and run the official sampling quality gate.

## Reproduction

Representative FP16 run:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
PYTHONPATH=. TORCH_CUDA_ARCH_LIST=7.0 \
uv run --no-project \
  --python /home/ymzx/miniconda3/envs/vllm-0.0.5-t210/bin/python \
  python benchmarks/benchmark_sm70_awq_verifier_micro.py \
  --op-library \
    bench_results/awq_m5_structural_20260710/warp8_candidate/_C.baseline_fixed.abi3.so \
  --model /home/ymzx/models/Qwen3.6-27B-AWQ \
  --device cuda:0 --m 5 --tp-size 2 --tp-rank 0 \
  --m5-batched-gemv fp16 --warmup 30 --iters 100 --batch-repeats 20
```

All-real aggregate:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
PYTHONPATH=. TORCH_CUDA_ARCH_LIST=7.0 \
uv run --no-project \
  --python /home/ymzx/miniconda3/envs/vllm-0.0.5-t210/bin/python \
  python benchmarks/benchmark_sm70_awq_verifier_micro.py \
  --op-library \
    bench_results/awq_m5_structural_20260710/warp8_candidate/_C.baseline_fixed.abi3.so \
  --model /home/ymzx/models/Qwen3.6-27B-AWQ \
  --device cuda:0 --m 5 --tp-size 2 --tp-rank 0 --all-real-layers \
  --m5-batched-gemv fp16 --skip-case-timing \
  --aggregate-warmup 5 --aggregate-iters 100
```

Primary artifacts are under:

```text
bench_results/awq_m5_trt_gemv_20260711/
```
