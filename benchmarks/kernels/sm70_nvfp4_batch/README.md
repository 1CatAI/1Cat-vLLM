# SM70 batched NVFP4 tile screen

The standalone N128 prototype shares direct row-major activations between
column warps and gate/up, uses software double buffering, and writes FP32
split-K partials. It supports M16, M32 and M64 without input packing. The
prototype is not selected by any serving dispatcher.

Run the driver with a sharded QUASAR checkpoint and a source-built SM70
installation. Acquire exclusive GPU ownership before running:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=7.0 \
python benchmarks/benchmark_sm70_nvfp4_batch_tiles.py \
  --model /path/to/checkpoint --out /path/to/result.json --m 16 32 64 \
  --candidate benchmarks/kernels/sm70_nvfp4_batch/n128_splitk.cu
```

The driver reads layer 55 TP rank 0 weights, times CUDA graph replays with a
128 MiB cache eviction outside the timing events, and records projection
error relative to the installed compressed operator. Nsight Compute can
profile only the candidate region with `--ncu --ncu-candidate`; use
`--candidate-so` to reuse the exact measured extension binary. Preserve the
source and binary hashes with counter results. Full model numerical and
concurrent decode gates remain required before production admission.
