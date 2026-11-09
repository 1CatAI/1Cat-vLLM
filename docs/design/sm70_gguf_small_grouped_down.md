# SM70 canonical small grouped down projections

MTP verification sends roughly one routed token to each active expert.
The IQ4_NL and Q2_0 vector operator consumes the existing TurboMind N32/K8
canonical banks without adding an original-weight copy or padding each
expert to an M=8 tile. Existing expert offsets identify active intervals.
The same TurboMind register transforms reconstruct FP16 operands; products
and reductions use FP32. Each CTA computes 16 output columns with eight
K partitions, then combines four warp partials in FP32.

Q2_0 has a 64-value original block. Canonical group32 expansion precedes
TP slicing, so Flash-Next TP4 K=160 never cuts a packed original block.
IQ4_NL uses the existing nonlinear LUT decoder. This complements the joint
original-block gate/up operator; no additional lattice decoder is introduced.

## Correctness

Nine CPU capability checks and 24 GPU cases pass: both formats, all four TP ranks, original M=1/5/20,
empty experts, singleton and repeated expert intervals, official FP32
GGUF dequantization before TP slicing, changed-input graph replay and exact
repeated replay. Real-weight outputs remain finite. IQ4_NL relative L2 is
0.000292–0.000294 and Q2_0 is 0.0002078–0.0002089; canonical GEMM and the
new vector errors are effectively equal because they share decoded operands.
The vector path does not reduce accumulation precision.

## Real-weight timings

V100-SXM2-32GB, SM70, CUDA 12.8.93, Torch 2.10.0+cu128,
Python 3.12.3, FP16 activations, FP32 accumulation, TP4 rank0,
512 experts, local N=2560/K=160, top-k=10. Inputs use the real
`blk.0.ffn_down_exps.weight` and `blk.1.ffn_down_exps.weight` from
Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S. Ten rotated weight banks exceed twice
L2 capacity even for the Q2_0 M=1 active set. Timings use CUDA graph replay,
100 replays after eager and graph warmup, normalized by ten banks.
M below counts original input tokens; capability M counts routed rows.

| Source type | Original M | Active experts | Grouped GEMM µs | Vector µs | Saved µs |
|---|---:|---:|---:|---:|---:|
| 20 | 1 | 10 | 34.721 | 12.517 | 22.203 |
| 20 | 5 | 48 | 47.137 | 40.567 | 6.570 |
| 20 | 20 | 165 | 89.232 | 144.918 | -55.685 |
| 42 | 1 | 10 | 34.493 | 12.085 | 22.408 |
| 42 | 5 | 48 | 50.121 | 40.645 | 9.476 |
| 42 | 20 | 165 | 83.942 | 137.066 | -53.124 |

The M=5 projection across 39 IQ4_NL and nine Q2_0 layers is 0.342 ms.
This is an operator estimate, not a measured model-round improvement.
M=20 regresses by 53–56 µs per layer and remains on grouped GEMM.
Only measured routed M=10 and M=50 are declared by default; unmeasured
batches and descriptors retain the canonical fallback. Missing operator,
non-SM70, dtype, disabled-policy and uncalibrated geometry reasons are
reported by the capability declaration.

## Artifact and reproduction

The ordinary source-containing wheel is `1.5.2.dev530+g585a4d4bf`.
The graph benchmark is the two-line replay correction in `d04a9e6cbd`;
the native operator is unchanged. All 213 installed packages pass dependency
checking. Installed `_C` has only standard Torch/CUDA dependencies and no
RPATH/RUNPATH. Wheel SHA256:
`80cd89483ba6a1d3a317711175765f8ff33e75fba804f36549ea17486b6f20e9`.
Native `_C` SHA256:
`254ca8ceb40a6c07ecabe45494f9537b6a934c87cfc8ad33daa7a54d8e2b9c57`.

```bash
.venv/bin/python -m pytest -q tests/kernels/quantization/test_gguf_small_grouped_vec.py
.venv/bin/python -m pytest -q tests/kernels/quantization/test_gguf_small_grouped_capabilities.py
.venv/bin/python benchmarks/kernels/benchmark_gguf_small_down.py MODEL.gguf --layer 0 --output iq4nl.json
.venv/bin/python benchmarks/kernels/benchmark_gguf_small_down.py MODEL.gguf --layer 1 --output q2.json
```

GPU runs require the shared GPU lock and an idle device. Model dispatch is
integrated separately after this calibration; these data do not establish
end-to-end throughput or quality.
