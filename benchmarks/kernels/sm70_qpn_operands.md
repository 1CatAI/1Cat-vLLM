# SM70 M8 projection and producer-layout screen

These benchmarks extract the production NVFP4/FP8 kernel bodies into isolated
research extensions. They test operand dependencies, activation layout, decode
simplification, and the cost of preparing inputs in a real TP4 GDN layer.
They do not replace the installed vLLM extension or register a serving route.

## Variants and numerical scope

| Variant | Purpose | Numerical requirement |
| --- | --- | --- |
| Production / source control | Calibrate the extracted kernel | Bitwise |
| Constant activation / weight | Remove one operand dependency | Invalid model output; diagnostic only |
| Fixed shape | Specialize M8 dimensions | Bitwise |
| Packed input / packed chain | K16 activation layout; publish packed gate output | Bitwise |
| Packed scaled | Fold the exact power-of-two decode factor into a bounded scale | Bitwise; reject overflowing expanded scales |
| Swizzled warp assignment | Change physical warp assignment without changing logical sums | Bitwise |
| N16 tile | Increase resident CTAs after reducing register demand | Changes reduction order; model KL required |
| FP8 postscale | Apply channel scales after FP32 accumulation | Changes rounding; model KL required |
| Norm producer packing | Change the fused norm's output address to K16 layout | Bitwise, including residual output |

The NVFP4 dimensions are M8, TP-local gate/up N8704/K5120 and down
N5120/K4352. They describe the first 56 MLPs of the original mixed NVFP4/FP8
checkpoint. The FP8 screen covers GDN QKVZ+a/b, GDN output, attention QKV and
a final FP8 MLP down projection. Draft weights and both LM heads are untouched.

The K16 layout reuses the existing activation-packing convention. The scale
fold retains the production FP16 effective-scale rounding, then multiplies that
scale by 16384 before decoding the small FP16 code. It is valid only where this
expanded scale is finite. It is not a general unbounded-scale replacement.

## Run

Use CUDA 12.8, Torch 2.10.0+cu128, V100/SM70, a task-owned build directory and
an original checkpoint. Acquire the GPU lease and verify process ownership.
The standalone screens use a real rank-0 TP4 weight shard on one GPU.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 \
CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST=7.0 MAX_JOBS=1 \
.venv/bin/python benchmarks/kernels/benchmark_sm70_qpn_operands.py \
  --source-root "$PWD" --output "$PWD/.cache/operand-screen" \
  --model /path/to/original-checkpoint --layers 0 16 32 55
```

`--build-only` compiles without creating GPU tensors. `--generate-only` checks
source anchors without importing Torch. `--load-existing` verifies the retained
source text before loading an already-built extension. Preserve the generated
source, compiler resource report, library hash and raw timing samples together.

The screen interleaves variants and reads a 128MiB eviction buffer outside the
CUDA-event interval. Ten contiguous graph replays precede each recorded sample.
Packing is excluded from the isolated projection measurement, explicitly
included in `benchmark_sm70_qpn_mlp_chain.py`, and either separate or fused into
the producer in `benchmark_sm70_qpn_layer.py`.

Build `benchmark_sm70_qpn_norm_layout.py` separately for the producer control.
The layer benchmark runs under `torch.distributed.run --nproc-per-node=4` and
requires `--source-root`, `--model`, `--operand-library`, `--norm-library`, and
`--output`. It uses real layer weights and synthetic inputs, includes both TP4
all-reduce/norm boundaries, and leaves GDN core parallelism unchanged. It checks
four input amplitudes, captures each variant, reads CUDA graph node types, and
interleaves timing. State restoration and eviction are outside the timed region.

The FP8 builder is `benchmark_sm70_qpn8_operands.py`; its real-weight driver is
`benchmark_sm70_qpn8_operand_screen.py`. FP32 dense projection error is recorded
for rounding-changing candidates, but is **not** a model teacher-forcing gate.

For NCU, use `--profile-from-start off --graph-profiling node`. The standalone
NVFP4 and FP8 drivers expose a `--profile` range. The TP4 layer uses `--ncu` for
that range and `--profile` for a separate Torch trace. Do not combine the two
profilers. A neighbor-state investigation must preserve preceding cache state
and record the number of replay passes; cache flushing changes the experiment.

## V100 results and admission decision

Measured with four V100-SXM2-16GB GPUs at 300W, full NVLink, Torch 2.10.0+cu128,
CUDA 12.8, native layouts, M8, TP4 shards of the original mixed checkpoint.
Default application clocks were 877/1312MHz. Temporary clock controls were
restored. These are research measurements, not serving benchmarks.

| Scope | Production | Packed + bounded scale fold | Interpretation |
| --- | ---: | ---: | --- |
| Isolated gate/up | 44.42µs | 37.86µs | Excludes packing |
| Isolated down | 23.58µs | 22.68µs | Excludes packing |
| Complete local MLP, mean of layers 16/32/55 | 59.44µs | 57.34µs | Includes the separate input pack |
| Complete TP4 GDN layer, rank 0 | 152.35µs | 151.94µs | Norm writes packed output directly |

All four ranks pass the complete-layer bitwise check at four input amplitudes.
However, every rank's paired bootstrap 95% interval for the whole-layer timing
improvement includes zero. **The packed producer is not admitted as a speedup.**
Its graph has the same 11 kernel nodes as production, including the untimed
eviction kernel; explicit packing has 12. The two untimed tensor copies are
separate graph memcpy nodes. No per-model-step kernel reduction is claimed.

The layer trace narrows the discrepancy: rank-0 gate/up is 43.70µs for production
and 41.86µs for the candidate; down is 26.90/26.80µs. These are profiled kernel
intervals, not the unprofiled event intervals in the table. In particular, the
isolated candidate's 37.86µs must not be multiplied by layer count as a gain.

NCU confirms that activation request processing matters in isolation. Removing
weights leaves only about 82KB of DRAM reads yet still takes about 36µs under
profiling, with high L1TEX activity. K16 packing reduces gate load wavefronts
from about 975k to 348k. That does not establish a deployable model improvement.
Activation request bytes are also not additional HBM weight bytes.

The TP4 neighbor capture preserves preceding cache state (`cache-control none`,
three replay passes). It still reduces gate load wavefronts from 974902 to
348160, while DRAM reads remain 25.16/25.08MB and eligible warps per scheduler
remain 1.106/1.134. Kernel replay can change cache residency, so these counters
do not replace the unprofiled layer admission result. Reducing this request
overhead alone has not solved the remaining operand-readiness bottleneck.

Other controls were not admitted:

- Fixed dimensions regress down; physical warp reassignment adds little.
- N16 gate regresses to 40.91µs; N16 down improves to 21.66µs but changes
  reduction order. No model KL gate was run for it.
- FP8 postscale/packing has small QKV gains and output/down regressions.
- FP8 N16 output has 39 registers, 12KiB shared memory and four resident CTAs,
  but measures 15.10µs versus 15.23µs production. Higher occupancy alone did
  not provide a meaningful event-time gain.

No full wheel, C1/C4 serving rollout, acceptance-rate result or model KL pass
is claimed. Reuse these negative controls and whole-layer results before
proposing further kernel variants.
