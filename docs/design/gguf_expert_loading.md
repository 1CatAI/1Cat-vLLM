# GGUF expert loading

Canonical expert loading previously converted an entire expert on every TP
rank and then discarded the other ranks' values. The loader now slices packed
rows or whole K blocks before converting them. When TP cuts an original block,
it retains canonical slicing. In particular, Q2_0 K=640 with TP4 still uses its
expanded group32 representation to support local K=160.

IQ MMA carriers contain exactly 16 bits. Expanding their eight U2 codes with
uint16 operations avoids the previous uint64 temporary. Indices, signs, FP16
coefficients, metadata, GPU layouts and inference arithmetic are unchanged.
Original-block gate/up readers continue to receive their original TP slices.

## CPU conversion measurement

The comparison uses real Flash-Next IQ3_S expert tensors, TP4, RAM-resident
inputs, and seven same-process ABBA repetitions. Both paths are checked on all
four ranks, including every canonical array and the final IQ MMA carriers.
All comparisons are byte exact. Torch 2.10.0+cu128 and NumPy use one CPU thread;
the installed runtime targets V100-SXM2-32GB with CUDA 12.8.

| Projection | Original ms/expert | Updated ms/expert | Speedup |
| --- | ---: | ---: | ---: |
| IQ4_NL down | 3.425 | 0.943 | 3.63x |
| IQ3_XXS gate/up | 4.800 | 2.612 | 1.84x |
| IQ2_S gate/up | 5.063 | 2.407 | 2.10x |
| IQ3_S gate/up | 5.058 | 2.709 | 1.87x |
| IQ4_XS gate/up | 3.733 | 1.027 | 3.63x |
| Q2_0 down | 7.443 | 7.406 | 1.00x |

Weighting these medians by the actual 73,728 expert projections gives
345.55 seconds before and 176.53 seconds after per rank. These are CPU
conversion estimates, excluding GPU uploads/preparation, bank finalization,
dense weights, name mapping and I/O. They are not model startup wall times.
The previous complete weight load was 709.32 seconds; a complete updated load
must be measured independently.

## Storage diagnosis

A bounded 1-GiB direct read of the same model file completed in 0.391882
seconds, about 2.7 GB/s. During the previous model load, physical reads were
about 70–80 MB/s, disk utilization was 2–4%, and each of four loading workers
occupied one CPU core. The disk was not saturated. Demand-driven mmap reads
and conversion work must be distinguished from a sequential storage benchmark.

The CPU benchmark is `benchmarks/benchmark_gguf_expert_transcode.py`. Pass
`--candidate-dir` and, when comparing revisions explicitly, `--control-dir`
pointing to directories containing each revision's `gguf_transcode.py` and
`gguf_lattice_transcode.py`, followed by all GGUF shards and `--output`.
The reference LUT codec remains unchanged between these revisions.

## Validation

The targeted transcode suites pass 113 tests, including 25 source formats,
both TP axes, every rank, original-block misalignment, IQ carrier round trips
and reconstruction against official GGUF dequantization. No GPU kernel or
runtime dispatch changes are included.
