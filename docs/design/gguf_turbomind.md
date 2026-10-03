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
and 20 measured iterations. Graph warmup and capture use the same stream.

The downloaded 27B UD-Q4_K_M checkpoint mixes IQ4_XS, Q3_K and Q4_K among
its FFN projections. Use actual Q4_K layers for this first affine scope;
the filename alone does not establish a tensor's quantization format.

An initial GDN-output M=512 measurement was 399 us versus AWQ's 218 us.
The CUDA trace showed a grouped 128x128 tile where AWQ used a dense 128x256
tile. Adding the existing dense tile repertoire for group32 and using the
existing dispatch policy reduced the canonical operator to 233 us versus
230 us for AWQ. Shared-gate M=512 was 56 us versus 55 us. These are individual
operator graph timings; the complete final tables are recorded separately
after correcting graph warmup stream initialization.

The first graph's workspace initialization had been captured in a preliminary
measurement because warmup ran on a different stream. Those first-point graph
figures are not used to choose a default. Small-M and M=64 routing still need
the corrected sweep. No universal speedup or final default-M policy is claimed.

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
