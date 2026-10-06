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
