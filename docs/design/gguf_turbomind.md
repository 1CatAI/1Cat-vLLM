# Canonical GGUF weights in TurboMind on SM70

GGUF is a weight source for TurboMind. Model scheduling, FP16 activations,
mma884 arithmetic, batching, expert routing and CUDA graph replay retain
their existing contracts. The packaged llama.cpp operators are fallback and
reference implementations; they are not the primary optimization target.

## Decoder families

| Family | Source formats | Canonical representation | Implementation state |
| --- | --- | --- | --- |
| Affine integers | Q4_0, Q4_1, Q8_0, Q4_K | u4/u8, group32 FP16 scale and additive min | Implemented operators |
| Affine integers | Q1_0, Q2_0, Q5_0/1, Q2_K, Q3_K, Q5_K, Q6_K | u2 and composed bit planes, group16/32 coefficients | Next operator scope |
| LUT4 | IQ4_NL, IQ4_XS, MXFP4, NVFP4 | Four-bit index with format-specific decode table | Separate operator scope |
| Lattice codebooks | IQ1_S/M, IQ2_XXS/XS/S, IQ3_XXS/S | Shared codebook plus indices/signs and scales | Separate operator scope |
| Ternary | TQ1_0, TQ2_0 | Lossless conversion to affine u2 | Depends on u2 scope |

No model is connected to the new operators in this change. The first model
integration is Qwen3.8-27B Q4_K_M, followed by Flash-Next IQ3_XXS with TP4.
Other source families retain fallback capability until their canonical
operators and numerical checks are available.

## Group32 affine storage

The CPU transcode emits independent projection descriptors:

- Integer codes `[N,K]`, with no rounding of the code values.
- FP16 scale/min `[N,K/32]`, evaluating `scale * code + min`.
- Source format, integer width and group size.

Q4_0 becomes unsigned codes plus `min=-8*d`. Q4_1 keeps its scale/min.
Q8_0 codes are shifted by 128 with `min=-128*d`. These coefficients are exact
when representable in FP16. Q4_K's nested scales/mins are multiplied in FP32
and rounded to FP16; reconstruction error is reported against gguf-py's
official dequantizer. Coefficient overflow rejects this representation.
Additive min is stored directly, including zero-scale constant groups.

GPU preparation uses the existing TurboMind converters and packed layouts.
Scale/min pairs occupy one uint32. The integer u8 decoder reverses the middle
byte interleave used by `Converter<uint16_t,uint8_t>`; KV-cache uint8 storage
has a different order. The decoder changes no FP16 activation or MMA precision.

TP slices canonical groups rather than the original GGUF superblocks. A
Q4_K row of K=768 therefore supports TP4 K=192 without cutting a canonical
group, even though 192 is not a multiple of the original 256-value block.
Mixed projections retain their own source type, coefficients and descriptor.
Their outputs can be concatenated without reinterpreting one projection's
packed bytes as another format.

The current GPU converter requires N divisible by 32 and K divisible by 32.
The framework reports an output-pack or group-boundary rejection for smaller
tails. It does not pass an incomplete N pack to the converter or silently
change computation precision. N padding is a subsequent storage feature.

## Kernel lifecycle and capability

`TurboMindGgufAffineKernel` participates in the existing mixed-precision
selector. Canonical storage is admitted only by its corresponding kernel;
GPTQ/AWQ checkpoint loaders do not reinterpret it. The operator capability
records family, source format, M interval and graph support, and appears in
the existing prepared-kernel startup report. Unsupported layout, activation
dtype, source codec, missing packaged operator or disabled policy gives a
specific rejection reason. The default is enabled; no new runtime environment
variables are introduced.

Dense u4 group32 and integer u8 group32 are registered in the existing
sm70_884_4/8 registries. Dense and grouped bridges use the existing workspace
and GEMM implementation. Graph capture and fake implementations preserve the
operator's mutation schema; preparation releases the temporary unpacked codes
and separate min tensor once the packed parameters exist.

## Numerical and runtime checks

Runtime on 2026-10-03: V100-SXM2-32GB, SM70, driver 580.173.02, CUDA toolkit
12.8.93, Torch 2.10.0+cu128, Python 3.12.3. Tests use GPU 0 under the shared
GPU flock. There is no attention, KV cache, sampling, MTP or model TP workload
in these operator measurements.

- Six CPU transcode tests compare official dequantization, report Q4_K
  coefficient rounding, and validate TP4 reblocking across original blocks.
- Twenty-one GPU tests pass, covering four formats, M=1 through M=512,
  dense and grouped GEMM, K=160, empty experts, graph replay, constant blocks,
  framework selection and its rejection reasons. One Q4_K raw-row case is
  skipped because K=160 cannot hold its original superblock; canonical
  reblocking is checked separately.
- The framework lifecycle also passes a full-graph `torch.compile` smoke
  using the eager compiler backend. This proves tracing compatibility, not
  Inductor scheduling or whole-model performance.

For actual Flash-Next Q4_K tensors, expanding scale/min to FP16 gives:

| Tensor | N | K | Max absolute weight error | Relative L2 weight error |
| --- | ---: | ---: | ---: | ---: |
| Layer 2 shared expert gate | 640 | 2560 | 0.000199795 | 0.000845949 |
| Layer 2 GDN output | 2560 | 6144 | 0.000379562 | 0.000691617 |

Output relative L2 against official dequantization with FP32 accumulation is
0.000875–0.000946 for the shared gate and 0.000754–0.000771 for GDN output in
the initial M sweep. These are operator measurements, not model quality gates.

## Benchmark methodology and current findings

`benchmark_gguf_turbomind.py` reads actual checkpoint bytes, records every
tensor's shape/type, and measures M=1,2,4,8,16,32,64,128,512,2048,8192.
It compares the canonical operator, existing TurboMind AWQ group128 at the
same N/K, admitted llama.cpp MMVQ/MMQ, dequantization plus cuBLAS, and cached
FP16 as a lower bound. AWQ is a shape/implementation comparator; its codes
are not claimed to be an independently quantized copy of the same model.
Both eager and graph timings use CUDA events, with 100 ms per-route warmup
and 20 measured iterations. Graph warmup and capture use the same stream;
three replay warmups exclude the first graph upload. A single shared-gate
M=2 AWQ outlier was rerun for 100 iterations; both raw runs are retained.

The downloaded 27B UD-Q4_K_M checkpoint mixes IQ4_XS, Q3_K and Q4_K among
its FFN projections. Use actual Q4_K layers for this first affine scope;
the filename alone does not establish a tensor's quantization format.

The initial narrow group32 tile repertoire was substantially slower than
AWQ at M=512. A CUDA trace identified a grouped 128x128 tile where the
reference selected a dense 128x256 tile. Adding dense group32 candidates
closed most of that prefill gap. Canonical u4/u8 also use distinct measured
cache keys for M <= 32, enabled by default without runtime environment gates.
Uncached capture uses a feasible fallback without tuning inside capture.

Current graph timings in microseconds follow. The eager GGUF column is
reported separately. MMVQ/MMQ entries are omitted when the explicit upstream
operator rejects that M interval. Cached FP16 excludes dequantization.

### Qwen3.8-27B-UD-Q4_K_M.gguf / blk.3.ffn_gate.weight

Q4_K; N=17408, K=5120. These are full projections, not TP4 slices.

| M | GGUF eager | GGUF graph | AWQ graph | MMVQ graph | MMQ graph | DQ+cuBLAS graph | Cached FP16 graph |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 79.16 | 100.51 | 66.15 | 73.78 | — | 974.64 | 310.43 |
| 2 | 79.16 | 99.64 | 72.76 | 76.85 | — | 970.50 | 310.94 |
| 4 | 79.82 | 100.40 | 72.96 | 111.41 | — | 973.57 | 311.30 |
| 8 | 76.44 | 102.09 | 70.09 | 193.69 | 140.29 | 972.44 | 314.52 |
| 16 | 83.97 | 101.53 | 95.03 | — | 189.95 | 984.88 | 320.82 |
| 32 | 117.20 | 129.79 | 115.51 | — | 269.31 | 995.48 | 313.65 |
| 64 | 231.63 | 262.50 | 208.13 | — | — | 1104.13 | 441.34 |
| 128 | 458.50 | 478.87 | 462.34 | — | — | 1278.57 | 589.88 |
| 512 | 1505.89 | 1507.07 | 1436.42 | — | — | 1853.49 | 1108.38 |
| 2048 | 5290.14 | 5219.58 | 5063.12 | — | — | 5152.36 | 4167.94 |
| 8192 | 21499.03 | 21498.27 | 20459.01 | — | — | 18475.72 | 16725.81 |

### Qwen3.8-27B-UD-Q4_K_M.gguf / blk.1.attn_gate.weight

Q4_K; N=6144, K=5120. These are full projections, not TP4 slices.

| M | GGUF eager | GGUF graph | AWQ graph | MMVQ graph | MMQ graph | DQ+cuBLAS graph | Cached FP16 graph |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 42.39 | 46.23 | 37.58 | 31.90 | — | 333.93 | 92.93 |
| 2 | 33.89 | 43.42 | 38.04 | 33.43 | — | 344.58 | 96.51 |
| 4 | 34.25 | 43.62 | 37.89 | 45.06 | — | 340.94 | 97.02 |
| 8 | 35.69 | 43.78 | 38.66 | 76.39 | 61.54 | 347.14 | 99.84 |
| 16 | 40.86 | 48.54 | 45.21 | — | 78.39 | 347.75 | 100.30 |
| 32 | 54.68 | 54.78 | 56.83 | — | 109.36 | 350.57 | 104.50 |
| 64 | 93.08 | 92.77 | 96.67 | — | — | 421.99 | 172.70 |
| 128 | 140.29 | 139.47 | 153.24 | — | — | 472.37 | 213.66 |
| 512 | 572.11 | 570.88 | 559.41 | — | — | 697.24 | 432.18 |
| 2048 | 1880.52 | 1880.27 | 1805.98 | — | — | 1832.81 | 1506.46 |
| 8192 | 7781.84 | 7758.39 | 7475.56 | — | — | 6629.43 | 5993.01 |

### Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf / blk.2.ffn_gate_shexp.weight

Q4_K; N=640, K=2560. These are full projections, not TP4 slices.

| M | GGUF eager | GGUF graph | AWQ graph | MMVQ graph | MMQ graph | DQ+cuBLAS graph | Cached FP16 graph |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 12.90 | 12.54 | 15.56 | 8.76 | — | 23.40 | 7.22 |
| 2 | 13.38 | 12.20 | 15.52 | 9.01 | — | 28.11 | 11.84 |
| 4 | 12.80 | 12.44 | 15.97 | 10.19 | — | 28.16 | 12.13 |
| 8 | 13.77 | 13.41 | 16.28 | 14.23 | 17.61 | 28.06 | 12.19 |
| 16 | 13.88 | 13.47 | 18.33 | — | 20.02 | 28.42 | 12.49 |
| 32 | 14.28 | 14.13 | 22.99 | — | 24.93 | 27.49 | 11.72 |
| 64 | 44.13 | 44.60 | 39.01 | — | — | 30.77 | 12.80 |
| 128 | 545.08 | 39.68 | 39.27 | — | — | 37.68 | 19.30 |
| 512 | 55.40 | 54.78 | 54.48 | — | — | 78.90 | 38.30 |
| 2048 | 129.89 | 130.46 | 129.13 | — | — | 145.31 | 93.08 |
| 8192 | 443.44 | 445.34 | 417.74 | — | — | 470.73 | 323.69 |

### Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf / blk.2.ssm_out.weight

Q4_K; N=2560, K=6144. These are full projections, not TP4 slices.

| M | GGUF eager | GGUF graph | AWQ graph | MMVQ graph | MMQ graph | DQ+cuBLAS graph | Cached FP16 graph |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 25.04 | 29.29 | 24.47 | 20.53 | — | 169.73 | 44.34 |
| 2 | 25.14 | 25.34 | 24.63 | 21.15 | — | 191.64 | 56.88 |
| 4 | 25.40 | 25.45 | 24.78 | 26.93 | — | 195.17 | 56.68 |
| 8 | 26.11 | 26.06 | 26.93 | 44.70 | 39.53 | 194.92 | 56.78 |
| 16 | 30.41 | 30.41 | 33.64 | — | 47.92 | 181.20 | 54.63 |
| 32 | 39.83 | 39.42 | 45.06 | — | 65.02 | 188.67 | 61.70 |
| 64 | 56.78 | 57.50 | 73.11 | — | — | 202.34 | 79.16 |
| 128 | 101.99 | 101.02 | 98.36 | — | — | 228.76 | 104.60 |
| 512 | 232.81 | 231.78 | 227.12 | — | — | 320.51 | 196.20 |
| 2048 | 891.80 | 889.91 | 877.41 | — | — | 872.91 | 716.24 |
| 8192 | 3666.02 | 3662.28 | 3569.36 | — | — | 3168.77 | 2882.87 |

The results support the tensor-core path for concurrent and prefill
workloads, but not a uniform AWQ-equivalent claim: the full 27B FFN gate
still trails group128 AWQ at M=1–8 and M=64. Expanded group32 coefficients
occupy four bytes per 32 weights, versus four bytes per 128 weights for this
AWQ comparator, in addition to different selected tiles. The exact eager
and capture descriptors match in the diagnostic trace; a graph-specific
kernel mismatch is not established. Further coefficient-traffic and tile
profiling is required before connecting this path to models.

At M=2048/8192, DQ+cuBLAS also beats the current TurboMind tile on several
large dense projections. Keep that measured candidate available; an
automatic model-level M policy is not yet selected. No activation precision
is reduced to match the reference timings.

The owned-source `_C` target builds through normal CMake. Wheel assembly
uses normal setup staging with that target installed into the package, while
unchanged auxiliary extensions come from the packaged fallback wheel. The
wheel `_C` hash matches the installed source target. Dynamic libraries are
not replaced while a benchmark is running.

Core extension SHA256:
`154effa3bcf0fab51c32d730cbf17c4b91af925ef1929971080ee91178451d5c`.
Wheel SHA256:
`cbf091977bc2cd1b22c230a9c3f4f09354ee56564fa7f7fbecdc04ce35002650`.

Reproduce with a normally installed source-built wheel:

```bash
flock /tmp/gpu0-3.lock env CUDA_VISIBLE_DEVICES=0 \
  python benchmarks/kernels/benchmark_gguf_turbomind.py "$MODEL_GGUF" \
  --tensor blk.2.ffn_gate_shexp.weight --tensor blk.2.ssm_out.weight \
  --cuda-graph --output affine-projections.json
```

## Sources

Block definitions and CPU reference formulas follow gguf-py and the pinned
MIT llama.cpp source used by the fallback operators. GPU packing and mma884
use the bundled OpenMMLab TurboMind implementation; its original copyright
and license are preserved. New bridges and canonical codecs carry the vLLM
Apache-2.0 headers. No upstream kernel body is copied into this affine decoder.
