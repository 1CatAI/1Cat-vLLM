# 1Cat FlashInfer SM70 module

This is an independently installable `flashinfer-python` provider, retaining
`import flashinfer`. Upstream sources are pinned by `UPSTREAM.lock.json` to
`34bb66e4d0b420b5a03c99b2b6a1c0831dfaa738` (0.7.2). The repository contains the
complete upstream snapshot and all four pinned dependency snapshots, including
their licenses. Local changes follow the separate vendor commit.

**Status: development; independent GPU correctness passed, performance admission pending.**
The supplied V100-SXM2 machine passes 40 standalone tests (31 execute GPU kernels)
on Python 3.12 / Torch 2.10 / CUDA 12.8. See the
[hardware validation record](gpu-validation-20261011.md). No engine backend is enabled
by default. The request-grouped QSA scorer is an explicit candidate described
in [QSA development and validation](qsa.md).

## Install and test independently

From this module directory, with Python 3.12 and the existing Torch 2.10/CUDA 12.8
environment (do not upgrade the engine stack):

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python build setuptools wheel packaging
.venv/bin/python -m build --outdir dist
# Use the CUDA 12.8 Torch index and the engine's pinned dependencies on V100.
uv pip install --python .venv/bin/python dist/flashinfer_python-*.whl
uv pip install --python .venv/bin/python pytest
.venv/bin/python -m pytest -c tests/sm70/pytest.ini --confcutdir=tests/sm70 tests/sm70
```

Use `-m 'not gpu'` for CPU packaging tests. The clean-process test intentionally
requires an environment without vLLM. Run GPU tests from the same ordinary
installed wheel in a fresh process; no source overlay or private library mount.
The wheel includes the CUDA sources and headers used for JIT. It is currently a
source/JIT wheel, **not an accepted AOT binary release**. The sdist contains the
build backend, pinned provenance, licenses and required build sources.

The base install does not install CuTe DSL, CUDA 13, cuda-tile or EP backends and
does not execute the upstream installer hooks that modify Torch/NCCL. The
upstream optional implementations and exports remain in the source;
`[upstream]`, `[b12x]`, `[ptx]`, `[cu12]` and `[cu13]` opt into their dependencies.
They retain their original architecture restrictions. Enabling an extra does not
make its kernels eligible for SM70. EP native builds are not performed by this
fork's minimal build backend.

## Compute and lifecycle boundary

```python
import flashinfer

flashinfer.sm70.prepare("basics")  # compilation and loading before capture
# Allocate x, weight, and y before capture.
flashinfer.rmsnorm(x, weight, out=y)
print(flashinfer.sm70.provenance("basics"))
```

SM70 public dispatch covers FP16 RMSNorm, Gemma RMSNorm, their residual variants,
SiLU/GeLU-and-multiply, and position-ID RoPE (including in-place, partial rotary
dimensions and interleaved layout). Norm/activation vector storage must be
aligned; unsupported dtypes/layouts raise. Norm supports contiguous 2D/3D input
with last dimension 1..8192. The explicit APIs also live in `flashinfer.sm70`.
RoPE accepts strided Q/K with a contiguous last dimension. PDL is rejected on
SM70. Other public APIs keep upstream capability checks; there is no global
`>=70` relaxation.

The library uses TVM-FFI and CUDA, with no vLLM imports, vLLM extension calls or
Torch C++ ABI dependency. Torch custom operators declare mutation and fake
implementations. Callers own outputs and persistent state. Replaying a captured
graph uses the captured tensor addresses; changing storage addresses requires
recapture. Reloading tensor contents in place is supported by the contract and
covered by the standalone GPU tests. No derived weight tensor is retained globally.

Offline compilation (does not run a GPU kernel):

```bash
.venv/bin/python -c 'from flashinfer.sm70 import build; print(build("basics"))'
```

Use `CUDA_HOME` to select CUDA 12.8. CUDA 13 is rejected. The development compiler
also accepts other CUDA 12 toolkits; these do not qualify a CUDA 12.8 release.
`FLASHINFER_SM70_CACHE_DIR` optionally changes the cache location; the default is
`~/.cache/flashinfer/sm70`. Each module's key includes source/header hashes,
upstream/dependency pins, version, SM70 architecture, compiler versions,
compiler environment, flags and FFI version. Builds run under an interprocess
lock, verify the cached DSO digest, and use an isolated child compiler environment.
Package directories are never runtime build destinations. `FLASHINFER_DISABLE_JIT`
requires a previously built matching artifact and fails otherwise.

## Ownership and update procedure

The engine owns scheduling, model/quantization configuration, weight loading,
PLE storage, cache allocation, TP/PP/DCP process groups and sampling policy.
The library owns native computation, explicit execution plans and workspaces.
Engine adapters must not move engine ownership into the module.

Keep upstream imports, compatibility changes, optimized kernels and engine
adapters in separate commits. For an upstream update, download an **explicit
full commit** into a new staging directory, verify its archive and gitlink
dependency SHAs, compare against the pinned vendor commit, update the lock, and
reapply/review local changes. Never update from a floating branch, overwrite a
dirty tree, or auto-upgrade dependencies during installation. Keep the upstream
`README.md`, licenses and notices. Parent pre-commit excludes this independently
managed module to avoid reformatting vendored code; lint owned SM70 Python/CUDA
and changed upstream interfaces using the module's configuration.

See [coverage and admission](coverage.md). The former primitive-only module is
documented in [SM70_PRIMITIVES.md](../../SM70_PRIMITIVES.md); the old preparation
script now resolves this owned source directory and never clones a different
FlashInfer version into `.deps`.
