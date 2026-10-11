# V100 hardware validation, 2026-10-11

The user supplied machine 54633 and authorized stopping its Studio inference
service. At validation start that service was already absent and all four V100s
were idle. This supersedes the earlier GPU deferral. Delivery is consolidated
in Draft PR #1177 against `onecat/main`; #1175 is closed with its commits retained.

## Independent installed artifact

Library source: `564e536ea5`, integration/benchmark source: `b41c0c7aa884`.
The only test change after that source is the startup-namespace correction below.
Main integration base remains `be3722bc095b9792bf9833068595f0b1ef70ad37`.

- Python 3.12.14; Torch `2.10.0+cu128`; TVM-FFI `0.1.10`.
- CUDA toolkit `12.8.93`, driver `580.173.02`, Tesla V100-SXM2-32GB, SM70.
- A fresh operator environment has no vLLM. It installs the ordinary source/JIT
  wheel, then compiles shipped sources into its own cache with CUDA 12.8.
- Wheel: `flashinfer_python-0.7.2+1cat.sm70.2-py3-none-any.whl`.
  SHA256: `b18477c8b332de3fcb07a8fbae05340cdbbc67411fd01ff63afd97bac0fe279a`.
- No `PYTHONPATH`, `LD_PRELOAD`, private `LD_LIBRARY_PATH`, borrowed engine DSO,
  source overlay, or engine-private extension is used.
- GPU 2 was held with the shared ownership lock during tests. Other tasks' builds and GPU
  processes are preserved. The P400 display device is excluded.

The loaded CUDA 12.8 artifact identities are:

| Component | Source SHA256 | Native library SHA256 |
| --- | --- | --- |
| basics | `b5cf0cf71251350587e2842cef29912fb2cf14bf9a9d89589808921c3e4823a8` | `d784b3afebd73318e37faefa226dbb8dd7210f1d645170acd16c8aa80ceb7cb6` |
| qsa | `9b83a952aa60a5e39d54d393d097d8b646aac788db0cb85dc4e220b6682838a8` | `f036fcfd77b30540070038d1079fe398f3dfadf2a5437bcd5e5f5c1df592c5ff` |

## Correctness result

**40 passed in 10.86 seconds**: 9 packaging/ownership tests, 16 GPU primitive
cases and 15 GPU QSA cases. Dependency consistency passes for all 38 installed
packages. Commands, from a directory containing the standalone tests:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST=7.0
.venv-operators/bin/python -I -c \
  'from flashinfer.sm70 import prepare; prepare("basics"); prepare("qsa")'
.venv-operators/bin/python -I -m pytest -x -v \
  -c tests/sm70/pytest.ini --confcutdir=tests/sm70 tests/sm70 \
  --junitxml=artifacts/operator-gpu-tests.xml
uv pip check --python .venv-operators/bin/python
```

RMSNorm/Gemma widths 72/128/2560/8192, residual rounding, activations and partial
strided/in-place RoPE pass. QSA covers C1/C4, M1/M5/M20/M32, strided layouts,
interleaved requests, page size 204, invalid physical pages, ties, numerical
extremes, int32/int64 positions, empty/padded rows, changing replay metadata,
independent captures and cache rebinding. The FP64 reference retains the existing
`rtol=atol=2e-6`; selected IDs and visibility are exact. No tolerances changed.

An expected warning occurs when testing rejection of plan allocation inside an
otherwise empty CUDA graph. It is not a failed capture of an execution plan.

The first run stopped at the fresh-process lazy-import assertion: NVIDIA's
`cuda-bindings` startup `.pth` creates the `cuda` namespace before FlashInfer is
imported. A before/after module inventory proves FlashInfer adds no backend
modules. The test now checks newly imported modules, including backend submodules,
rather than attributing interpreter startup work to this library. The failed log
and JUnit record are retained alongside the passing run; no production change
was needed for this failure.

## Performance and remaining gates

No speed claim follows from these correctness tests. A normal engine wheel is
being built from the owned source for a paired comparison against the existing
QSA route. Its native `csrc`, CMake file and CMake directory are identical to
the main baseline; no previous task's compiled engine extension is reused.

Score/selection screening, complete QSA, text C1/C4, MTP rollback/acceptance,
D72 vision, HC/MoE/GDN/PLE/output chains and shared Qwen3.6 regressions remain
open. Default selection stays off. Complete-chain and integrated acceptance
remain at least 5% improvement beyond noise, with at most 2% other-point
regression and unchanged numerical/quality criteria.

Raw artifacts are retained in the task's `gpu-54633` bundle: installed package
and loaded-native identities, test logs/JUnit, initial failure, dependency
inventories, GPU topology, dynamic linkage and disassembly. They are not build
or runtime dependencies.
