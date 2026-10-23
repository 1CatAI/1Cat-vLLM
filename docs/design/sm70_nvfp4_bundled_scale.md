# Bundled native NVFP4 groups on SM70

The qualified native QPN2 path stores one allocation containing 256 FP4 code
bytes and 32 E4M3 scale bytes per N32/K16 group. Code and scale tensors are
strided views of that allocation. Decode uses the existing reduction and
FP16 rounding order with adjacent group addresses.

Automatic selection is restricted to exact SM70, the existing qualified
shared native layout policy, and the TP4 MLP dimensions K5120/N8704 and
K4352/N5120. Other shapes retain their existing layout. No runtime switch
is required and neither target nor draft precision changes.

For prefill, a single kernel restores compact codes and rounded scale
operands into shared scratch. All serialized layers and captured graphs use
one bounded code buffer (21.25 MiB for these shapes), together with the
existing FP16 and scale workspaces. No layer retains a second weight layout.
If the code buffer cannot be reserved during preparation, the existing
native layout remains available. Worker shutdown releases these buffers.

## Validation

```bash
.venv/bin/python -m pytest -q --noconftest --import-mode=importlib \
  tests/kernels/quantization/test_sm70_nvfp4_native_layout.py
.venv/bin/python benchmarks/kernels/benchmark_sm70_nvfp4_bundle.py \
  --model /path/to/original-model --out /tmp/bundle-paired.json --rows 8 32
```

The paired benchmark uses real TP4 layer-zero weights, 128 MiB cache eviction,
CUDA graph events, randomized arm order and 200 samples. Packing and eviction
are excluded from timing. It compares integer bit patterns at four input
amplitudes. Retaining both layouts is confined to the benchmark.

A standalone build of the complete CUDA source passes 16 bit-pattern cases
for M1/7/8/9/16/24/32/64 and repeated graphs. Its Python dispatch passes M64
and M256 scalar/gated prefill comparisons, including zero and subnormal group
scales, and a 24-layer graph scratch/replay check. This private screen does
not constitute release-artifact or serving admission.

Matched CUDA 12.8, Torch 2.10/cu128, V100-SXM2-32GB cold MLP graphs measure
73.144 to 69.842 microseconds at M8 (paired saving 3.302, 95% interval
3.072–3.548), and 133.268 to 133.914 at M32 (saving -0.645, interval
-0.865–-0.389). Both layouts have two compute kernels. The M32 result requires
an explicit same-service C4 check before promotion; projection timings must
not be presented as model latency or measured DRAM throughput.

The complete TP4 GDN layer with this CUDA source measures 169.643
to 163.772 microseconds on the per-replay critical rank
(paired saving 5.871, 95% interval 4.943–6.813).
Output, residual, FP32 rollback state and convolution history bit patterns match.
Both graphs contain ten timed compute kernels. This is a layer measurement;
This estimate predicts approximately 0.293 ms per model round; serving measurements
below supersede that estimate.

## Release and serving validation

A complete CUDA 12.8 SM70 wheel passes all 23 native-layout GPU tests, schema/fake
operator checks and the workspace cleanup test. The installed attention and
projection extensions load successfully without task-local native libraries.

The serving comparison uses one process on four V100-SXM2-32GB GPUs, TP4,
300 W, application graphics clock 1290 MHz and memory clock 877 MHz, Torch
2.10/cu128, original unsloth Qwen3.8-27B-NVFP4 weights and DFlash2 draft7.
Maximum context is 262144, maximum sequences 4, FP8 E4M3 KV, block size 2048
and GPU memory utilization 0.8. Each arm uses eight prompts with 600 output
tokens, seed 123, temperature 0.7, top-p 0.9, top-k 20 and thinking disabled.
The measurement excludes prefill and the first 20 streamed rounds; no profiler
or CUDA-event instrumentation runs during these timings.

| Input | Separate layouts, ms/round | Bundled, ms/round | Tokens/round, both | Reference-path fraction, both |
| --- | ---: | ---: | ---: | ---: |
| 1K | 14.823 | 14.390 | 2.980 | 1.35% |
| 8K | 15.258 | 14.842 | 2.948 | 1.83% |

The 95% prompt-bootstrap intervals for tokens/round are identical across arms:
[2.790, 3.186] at 1K and [2.877, 3.030] at 8K. Three natural prompts per arm
terminate normally. All four ranks select 112 bundled projections. Their target
M8 graphs retain 606 kernel nodes and draft graphs retain 142. The selected
compact-path graph components sum to 775 kernel nodes in both arms; this count
excludes eager launches and is not the total number of kernels in a round.

The first warmed C4 pair measures 409.79 versus 408.83 tokens/s (-0.24%). A
targeted follow-up alternates control/candidate/control/candidate in one service,
warming each visit and measuring twice. The four measurements per arm average
403.12 versus 406.68 tokens/s; their ranges are 393.75–408.46 and 406.28–407.14.
This supports throughput parity, rather than a claim of a C4 speedup. M32 remains
a small regression in isolation and should remain covered by future C4 checks.

Additional, untimed CUDA-event requests measure the critical-rank target M8 graph
at 11.473 versus 11.041 ms, confirming that the approximately 0.43 ms round saving
is in GPU execution. The target M32 graph measures 20.461 versus 20.513 ms.
These envelopes include event overhead and exclude the rest of the verify round.
The measured M8 saving exceeds the cold-layer estimate by 47%; subsequent model
budgets should use the observed 0.43 ms, not extrapolate isolated kernels again.
No 700 GB/s or sub-12-ms claim follows from these measurements.
