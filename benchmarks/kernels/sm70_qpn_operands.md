# SM70 M8 operand-dependency screen

This benchmark separates activation and weight dependencies in the native
NVFP4 M8 gate/up and down kernels. It extracts the current production bodies
from an explicitly selected source tree and builds a separate research library.
It does not register a serving route or replace the installed vLLM extension.

| Variant | Purpose | Valid model output |
| --- | --- | --- |
| Production | Installed operator used for calibration | Yes |
| Control | Source-extracted kernel with the original runtime dimensions | Yes |
| Constant activation | Remove activation loads while retaining weight decode | No |
| Constant weight | Remove weight loads and decode while retaining activation loads | No |
| Fixed shape | Specialize the dimensions without changing HMMA or reduction order | Requires bitwise validation |

The fixed dimensions are M8, TP-local gate/up N8704/K5120 and down N5120/K4352.
The wrapper rejects other shapes. These shapes describe the first 56 MLPs of
the original mixed NVFP4/FP8 checkpoint, not its final eight FP8 MLPs.

## Run

Use CUDA 12.8, Torch 2.10.0+cu128, a V100/SM70 runtime, a task-owned output
directory, and an existing original checkpoint. The screen itself uses rank0's
TP4 shard on one GPU. It does not implement communication or a whole layer.
Acquire the machine's GPU lease and verify process ownership before running.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 \
CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST=7.0 MAX_JOBS=1 \
.venv/bin/python benchmarks/kernels/benchmark_sm70_qpn_operands.py \
  --source-root "$PWD" --output "$PWD/.cache/operand-screen" \
  --model /path/to/original-checkpoint --layers 0 16 32 55
```

`--build-only` compiles without constructing GPU tensors. `--generate-only`
needs no Torch import and verifies that the expected source bodies are found.
The build emits ptxas register, shared-memory and spill statistics. Preserve
these with the result: ablations change resource demand as well as memory work.

The screen checks the control and fixed-shape variant bitwise against the
installed operator at four input amplitudes. It then interleaves graph replays
of all five variants. Each replay reads a 128MiB eviction buffer outside its
CUDA-event timing interval. Outputs include samples, weight/scale payload,
source and library hashes, and explicit labels for invalid diagnostic outputs.

For NCU, add `--profile --layers 0` and collect only kernels matching
`operand_.*_(gate|down)` with `--profile-from-start off --graph-profiling node`.
Keep NCU replay timing separate from unprofiled event timing. Record SM/memory
clocks and the power limit; do not infer in-kernel clocks from an idle sample.

## Admission

Removing an operand is not an optimization and cannot be deployed. A faster
ablation bounds a hypothesis but does not establish an additive decomposition
of kernel time: instruction scheduling and register lifetimes also change.

First require source control to match the installed operator numerically and
calibrate its timing. Then inspect SASS and NCU dependency/throughput counters
before designing a deployable change. A valid candidate needs a cold real-weight
whole-layer graph improvement, followed by matched model validation. This
microbenchmark does not establish a model-level latency or throughput result.

The initial CUDA 12.8 SM70 compile passes with zero spills. Control gate/down
use 50/48 registers per thread; fixed-shape gate/down use 48/49. All four use
16KiB shared memory per CTA. GPU numerical, timing and model admission results
remain pending; no speedup is claimed from compilation.
