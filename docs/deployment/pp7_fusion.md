# Optional SM70 gated RMSNorm extension

The PP decode changes do not require this optional extension. Production keeps
`sm70_rmsnorm_gated_exact=false`: the existing Triton kernel was faster for larger
batches in the measured PP7 workload. The extension allows explicit comparison
of the original FP32 reduction schedule at up to 3,072 rows of 128 FP16 values.

The additional library registers `vllm_pp7::rmsnorm_gated_exact_out` and is loaded
from `vllm/_pp7_fusion.so`. It never replaces the main vLLM native extension.
Rows 1–192 still use the existing `_C::sm70_rmsnorm_gated_exact_out` operator.
The CUDA dispatch falls back to Triton if the additional library is missing.

## Build in a separate checkout

Use the checkout's existing CUDA-enabled PyTorch environment, a compatible CUDA
toolkit with `nvcc`, Ninja, and a C++17 compiler. Set `CUDA_HOME` if the toolkit is
not discoverable. The validated deployment used PyTorch 2.10.0+cu128, its C++11
ABI, GCC 11.4, and nvcc 12.9.86. Compiler paths and versions are not fixed by the
script; rebuilding with another toolchain need not produce an identical binary.

```bash
# Show source, output, architecture, and build settings without importing torch.
.venv/bin/python tools/build_pp7_fusion.py

# Build only the additional SM70 library; CUDA devices remain hidden.
.venv/bin/python tools/build_pp7_fusion.py --execute

# Optional overrides:
.venv/bin/python tools/build_pp7_fusion.py --execute \
  --cuda-home /path/to/cuda --build-dir /path/to/build
```

The kernel is compiled for SM70 with `-O3 --fmad=false` and without fast-math.
Do not add contraction or fast-math flags: the arithmetic order is intentional.
The builder refuses symlink outputs and existing binaries with different bytes.
Use a separate checkout when an installed binary must remain available. The
normal repository build does not build this optional library automatically.

## Validation and opt-in

CPU routing and fallback checks do not require a GPU. The focused conftest
boundary avoids loading the repository-wide optional test dependencies:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest \
  tests/kernels/test_pp7_fusion_candidate.py \
  tests/kernels/test_actual_norm_dispatch.py \
  --confcutdir=tests/kernels -k cpu -q
```

Run GPU checks only on an idle SM70 device, after building the extension:

```bash
CUDA_VISIBLE_DEVICES=0 PP7_FUSION_GPU_TESTS=1 .venv/bin/python -m pytest \
  tests/kernels/test_pp7_fusion_candidate.py \
  tests/kernels/test_actual_norm_dispatch.py \
  --confcutdir=tests/kernels -k gpu -q
```

For experiments, set `sm70_rmsnorm_gated_exact=true` in `--kernel-config`, keeping
the other required configuration keys. Admission requires SM70, contiguous FP16
inputs with 128 columns, no bias or grouping, normalization before gating, and
SiLU/swish or sigmoid. Keep the flag false for the validated production setting;
microbenchmark gains against a multi-operation reference do not establish an
improvement over the existing fused Triton implementation.
