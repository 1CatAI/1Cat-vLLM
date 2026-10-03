# Packaged native GGUF operators on SM70

This change packages fallback and reference operators for GGUF formats.
TurboMind canonical-format operators are the performance path. Standalone
architecture adapters and TP expert storage are separate review scopes.

## Sources and runtime contract

- llama.cpp is pinned to `002a12ad25503a93501b2e188c360029830a241a` (MIT).
- The Torch bridge starts from vllm-gguf-plugin PR 141 at
  `aa09d6522f29325d64d999d7d7c794f79836de07` (Apache-2.0).
- Unmodified upstream files, original bridge hashes, and local changes are
  recorded in `csrc/quantization/gguf_upstream/source_manifest.json`.
  Both licenses and binary provenance ship inside the wheel.
- The new TQ1/TQ2 CUDA decoder follows the upstream ternary block format.
  These types have dequant/BLAS capability, not MMVQ capability.

`KernelConfig.sm70_gguf` defaults to enabled. Each prepared projection reports
its operator capabilities, local shape, format, and rejected-route reason.
Existing formats retain their previous production operators. Formats absent
from those operators use the new fallback. Its explicit reference selector
uses dequantization followed by cuBLAS for M >= `prefill_min_m` (default 8),
and admitted MMVQ for smaller batches, otherwise BLAS. This threshold is not
a TurboMind default or evidence that the fallback wins at those shapes.
FP32 activations retain FP32 BLAS compute; they are not narrowed through FP16.
Mixed projections retain separate logical types, GPU allocations and storage
tails. Padding guards do not change the logical K dimension.

The local reader adds ISTA-DASLab Q2_0 (type 42, 64 values / 18 bytes), without
mutating the gguf package's global enum. Its tensor directory uses mmap; it
does not dequantize or copy complete expert stacks while reading metadata.

## Validation on 2026-10-03

Runtime: Tesla V100-SXM2-32GB, SM70, driver 580.173.02, CUDA toolkit 12.8.93,
Torch 2.10.0+cu128, Python 3.12.3. Operator tests and timings use GPU 0 only,
UUID `GPU-6d23ab90-d6d2-b22b-8486-7fec517f07b9`, under the shared GPU flock.
There is no model TP, attention, sampling, MTP or KV-cache workload here.

- Four Q2_0 reader/dequant/bounds tests pass on CPU.
- Eleven configuration, guarded storage and startup-report tests pass on CPU.
- Ten GPU tests pass: eight format cases (Q2_0, Q1_0, MXFP4, NVFP4,
  TQ1_0, TQ2_0, IQ3_XXS, Q4_K), CPU-loaded mixed shards moved to GPU,
  and large FP32 activations that would overflow FP16.
- Each format case compares FP32 dequantization to CPU reference, FP16 BLAS
  for M=1/8/32 to an FP32 accumulation reference, and admitted MMVQ under its
  Q8 activation numerical contract.
- The owned-source extension builds through normal CMake and ships as
  `vllm._C_gguf`. A fresh venv installs the resulting wheel, imports the
  packaged module and licenses, and passes all ten GPU tests. No preload,
  private library override or kernel sidecar is used.
- The focused wheel uses the existing normal precompiled wheel for unchanged
  base operators; the new GGUF extension is compiled from this owned source.
  A full build of every unchanged base operator is not claimed.

Extension SHA256:
`74ed944b8abb0f8679757a4e1bf0acef453f4a9803002f2c47b35b47f39d163f`.
Wheel SHA256:
`2539b29895ec139a2044c16e211f5dbb7ec520118e07f7651e84efbca582ee86`.
The wheel includes the prepared MoE capability declarations and preservation
of existing format routes, and passes ten tests in a fresh installed runtime.
The extension has no RPATH/RUNPATH; NEEDED entries are standard CUDA/cuBLAS,
Torch and system libraries. Torch supplies its own standard library loading.

## Real checkpoint projection timings

The IQ3_XXS Flash-Next checkpoint is mixed precision. These measurements use
the actual first expert bytes in layer 0 (IQ4_NL down, IQ2_XS gate) and layer 1
(Q2_0 down). They are not whole-model C1/C4 or prefill throughput.
Three warmups and 20 iterations are used per shape. Eager timings include
host launch gaps; graph timings replay one captured operator.

| Format / N / K | M | Native eager us | Legacy eager us | Native graph us | Legacy graph us |
| --- | ---: | ---: | ---: | ---: | ---: |
| IQ4_NL / 2560 / 640 | 1 | 44.24 | 19.61 | 16.18 | 9.83 |
| IQ4_NL / 2560 / 640 | 4 | 44.19 | 24.73 | 12.80 | 22.89 |
| IQ4_NL / 2560 / 640 | 8 | 50.79 | 41.98 | 25.29 | 40.14 |
| IQ4_NL / 2560 / 640 | 16 | 49.66 | 76.08 | 26.57 | 74.50 |
| IQ4_NL / 2560 / 640 | 256 | 51.92 | 40.29 | 45.47 | 36.71 |
| IQ2_XS / 640 / 2560 | 1 | 44.80 | 19.35 | 10.91 | 10.60 |
| IQ2_XS / 640 / 2560 | 4 | 43.78 | 20.68 | 12.95 | 18.38 |
| IQ2_XS / 640 / 2560 | 8 | 57.19 | 31.54 | 43.26 | 30.52 |
| IQ2_XS / 640 / 2560 | 16 | 54.17 | 54.68 | 43.42 | 53.91 |
| IQ2_XS / 640 / 2560 | 256 | 67.23 | 60.01 | 63.13 | 58.57 |
| Q2_0 / 2560 / 640 | 1 | 44.80 | unavailable | 9.83 | unavailable |
| Q2_0 / 2560 / 640 | 4 | 44.24 | unavailable | 14.39 | unavailable |
| Q2_0 / 2560 / 640 | 8 | 44.80 | unavailable | 18.43 | unavailable |
| Q2_0 / 2560 / 640 | 16 | 46.69 | unavailable | 18.74 | unavailable |
| Q2_0 / 2560 / 640 | 256 | 44.13 | unavailable | 39.12 | unavailable |

Native relative L2 versus dequantized FP16 weights with FP32 accumulation is
0.00497-0.00549 for MMVQ and 0.000205-0.000209 for BLAS. These figures do not
establish a model quality tolerance.

The native route is slower in several cases, including eager C1 and the
IQ2_XS M=8 graph. Graphs recover part of the host overhead, but a general
speedup is not established. Repack, fused dequant GEMM and grouped MoE remain
required performance work. Do not promote these operator results as meeting
the AWQ/NVFP4 whole-model acceptance target.

Reproduce using the normally installed wheel, with an actual shard path:

```bash
flock /tmp/gpu0-3.lock env CUDA_VISIBLE_DEVICES=0 \
  python benchmarks/kernels/benchmark_gguf_sm70.py "$MODEL_GGUF" \
  --cuda-graph --output projections.json
```

## Open gates

New raw MoE operators are packaged but the existing Python GGUFMoEMethod is
still the legacy implementation. Independent gate/up/down type storage,
TP storage and reduction, quantized PLE lookup, full model logits/greedy
comparison, common quality sets, and model C1/C4/C8/C16 plus 8K/32K prefill
are not completed by this kernel PR.
