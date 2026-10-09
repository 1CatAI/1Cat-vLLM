# SM70 dense GGUF weight reuse

The M20 projection previously launched three independent eight-row token
tiles. Each tile loaded and decoded the complete weight matrix. The new
weight-major kernel loads each group once and feeds three independent Volta
MMA accumulators. FP16 weight reconstruction, FP16 activations, FP32 K order,
warp reduction and final output rounding are preserved.

Admission is limited to the measured M20, split1, W4/W8 shapes:

| Projection | Local K | Local output widths | Layers |
| --- | ---: | --- | ---: |
| GDN qkv + z | 2560 | 2560 + 1536 | 36 |
| Attention q/k/v | 2560 | 3072 + 256 + 256 | 12 |
| GDN output | 1536 | 2560 | 36 |
| Attention output | 1536 | 2560 | 12 |

The two 256-dimensional attention KV heads are replicated over TP4; cutting
each GGUF K/V tensor into four 128-row shards would benchmark the wrong
shape. M5, other M, split-K and other dimensions retain the old kernel.
No weight expansion, second resident bank or additional graph allocation is
introduced. Existing segment dispatch and scratch ownership are retained.

## Community references and structural screens

[Marlin](https://github.com/IST-DASLab/marlin) discusses register reuse,
dequantization/MMA scheduling and offline layouts. Its Ampere asynchronous
loads are not used here. [FLUTE](https://github.com/HanGuo97/flute) studies
lookup layout and shared-memory lookup costs.
[GemLite](https://github.com/dropbox/gemlite) separates small-batch algorithms
and packing widths. These are design references; no external implementation
or license-incompatible source is incorporated. The decoder derives from
the existing Apache-2.0 SM70 projection implementation.

The tested alternatives preserve the current arithmetic, including
FP16 coefficient rounding. They are research screens, not model results:

| Screen | M5 total ms | M20 total ms | Decision |
| --- | ---: | ---: | --- |
| Matched installed control | 1.2466 | 3.1011 | Reference |
| Register IQ4 lookup | 1.2579 | 3.0651 | No useful overall gain |
| Register activation exchange | 1.3301 | 3.1930 | Reject |
| Lossless Q6/IQ4 byte expansion | 1.3313 | 3.1878 | Reject; more storage and traffic |
| Three token groups sharing weights | 1.3309 | 2.4392 | Select M20 only |
| Format-combination specialization | 1.3210 | 3.0068 | Reject as an overall policy |

Format specialization reduces registers for LUT-only kernels from about 144
to 95, but this does not improve the complete mixed-format M5 workload.
Another screen interleaves four decoded half2 operands with each pair of
MMA instructions; its M5 improvement is negligible and it is not selected.
Lower register counts and fewer lookup instructions alone are not latency
evidence.

An initial CUDA 12.0 screen is retained separately. Its control was faster
than the installed CUDA 12.8 kernel, so those cross-compiler differences are
not counted as optimization gains. All results in the table use CUDA 12.8.

## Microbenchmark contract

Four V100-SXM2-32GB GPUs with full NV2 connectivity; one locked GPU is used
for these independent TP4 shard projections. Torch 2.10.0+cu128 and NVCC
12.8.93; SM70, FP16 operands and FP32 accumulation. No fast-math flag is
introduced. The weights come from all 48 layers of Flash-Next GSQ-RCO
IQ3_S, retaining their actual mixed Q4_K, Q5_K, Q6_K and IQ4_XS types.

Each graph rotates through 12 or 36 distinct layer banks, substantially more
than V100's 6 MB L2 capacity. Graphs are warmed before CUDA-event timing.
Research arms are interleaved in forward/reverse order, ten observations per
arm, with eight graph replays per observation. Warp counts and split counts
are fixed to production choices, not selected from a tuning sweep.

| M20 role | Control us/layer | Weight-major us/layer |
| --- | ---: | ---: |
| GDN input | 43.68 | 36.97 |
| GDN output | 22.74 | 15.26 |
| Attention input | 38.71 | 32.80 |
| Attention output | 20.47 | 13.77 |

The weighted total improves 3.1011 to 2.4392 ms, about 21.3%. This is the sum
of 96 independently supplied dense projections, not end-to-end C4 latency.
All banks contain 540.34 MB of packed weight planes per rank. M20 previously
issued approximately 1.62 GB of weight loads across its three token tiles;
the new schedule issues approximately 540 MB. Cache hits can change measured
DRAM traffic; these are logical issued bytes, not Nsight DRAM counters.
Unique-weight bytes per service time improve approximately 174 to 222 GB/s.

The M5 installed control measures 1.2466 ms, about 433 GB/s of logical
weight traffic. The historical 2.3-ms dense service sum comes from a different
profile and execution context; it cannot be replaced with this isolated
microbenchmark total or subtracted from the 17.4-ms round baseline.

## Correctness and reproducibility

The initial six-screen CUDA 12.8 comparison passes 2,304 real-shard output
bit comparisons, across M5/M20 and two different activation inputs. Maximum
relative L2 error against official FP32 GGUF dequantization is 0.0008975;
candidate outputs are bit-identical to the installed control. Only the M20
weight-major schedule is incorporated into native dispatch.

The additional GPU suite covers all six dense source types, real admitted
dimensions, mixed segments, strided output, shared-gate rounding, poisoned
output/scratch, and changed-input CUDA graph replay. The reproducible native
benchmark uses only the normal CUDA extension:

```bash
flock /tmp/gpu0-3.lock python benchmarks/benchmark_gguf_dense_projection.py \
  --model "$GGUF_MODEL_DIR" --source-sha "$SOURCE_SHA" --output result.json
```

Run the same command with complete control and candidate source runtimes.
The JSON includes the native library hash, output fingerprints, official
dequantization error, exact N/K/type lists, logical weight bytes and every
timing observation. No whole model, prefill, attention, MTP sampling or
acceptance benchmark is run by this command.
