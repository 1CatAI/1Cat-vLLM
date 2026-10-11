# Foundation validation — 2026-10-11

Status: source/install/build checks passed; GPU correctness, CUDA graphs,
performance, output quality and release admission are pending. GPU execution was
explicitly deferred by the user. No throughput or latency improvement is claimed.

| Check | Evidence |
| --- | --- |
| Build | Standard PEP 517 sdist, then wheel built from that sdist with `python -m build --no-isolation` |
| Independent installation | Ordinary wheel installed in Python 3.12.3, Torch 2.10.0+cpu, TVM-FFI 0.1.10; no vLLM installed; dependency consistency check passed |
| Package/API checks | 8 passed: lazy imports, source/provenance packaging, public signatures, unchanged upstream SM70 rejection, unknown-module rejection, capture preparation guard, source/toolchain cache invalidation |
| Engine dependency metadata | 2 passed: exactly one pinned fork provider on SM70, official pins retained on non-SM70 |
| Offline native build | Installed wheel sources compiled with nvcc 12.0.140 to an SM70 ELF; FFI norm/activation/RoPE exports loaded in a fresh process |
| Native linkage | `libtvm_ffi.so`, `libcudart.so.12`, standard C++/C libraries; no Torch C++ or vLLM extension dependency |
| GPU tests | 16 primitive cases prepared, not executed: norm, residual rounding, activation, partial/strided position-ID RoPE, graph replay and in-place weight reload |
| Lint | Owned module Ruff and CUDA formatting checked; relevant parent hooks passed after fixing the dependency test's mypy annotation |

CUDA 12.0 offline compilation is an early source check only. It does not replace
the required CUDA 12.8/V100-SXM2 build/runtime validation. The wheel is a portable
source/JIT package; no task-local DSO is embedded or presented as a release binary.

Artifacts:

- `flashinfer_python-0.7.2+1cat.sm70.1-py3-none-any.whl`
  SHA256 `7607899cbc1504c30c37f9d86d763933cc500b02c11a1e0e31fd076fd2b50c0b`
- `flashinfer_python-0.7.2+1cat.sm70.1.tar.gz`
  SHA256 `2bacec38c6e973a28a94927b2e6c08c355bacfa4f28d1ecb092d95fbf4a5e15d`

The artifacts precede this report-only commit. Build/test logs and artifacts are
retained in the task's `flashinfer-sm70-module-20261011/foundation` artifact folder.

The RMS-only compilation profile excludes unused TRT-LLM LayerNorm/BF16 helper
definitions, which failed an FP16-only SM70 compilation. It reuses the upstream
RMS kernels and does not emulate BF16 Tensor Cores or loosen global arch guards.

Next acceptance work uses the [coverage ledger](coverage.md). Freeze a normal
CUDA 12.8 mainline control, establish A/A stability, then run paired complete
chains and model C1/C4/visual workloads. Keep the 5% gain and 2% regression gates,
original numeric tolerances, exact index/state checks, MTP acceptance and the
27B/35B-A3B AWQ/FP8 regressions. No automatic performance-admitted routes exist.
