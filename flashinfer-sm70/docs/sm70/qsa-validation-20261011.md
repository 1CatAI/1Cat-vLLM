# QSA candidate validation — 2026-10-11

Historical record of the initial CPU/offline phase, before the user supplied
machine 54633. The hardware deferral has ended. The current standalone GPU
results and paired partial-chain measurements are in the
[hardware validation record](gpu-validation-20261011.md). This initial record
does not establish acceleration or release admission. Delivery is now
consolidated in Draft PR #1177; #1175 is closed with its commits retained.

## Source and review boundary

- Integration line: `onecat/main`, audited `8d4fd940`, implementation base
  `be3722bc095b9792bf9833068595f0b1ef70ad37` (retains #1172).
- Foundation: Draft [#1175](https://github.com/1CatAI/1Cat-vLLM/pull/1175),
  head `6a6c48cb45`.
- Library implementation: `564e536ea5`; engine adapter: `3e58378b30`.
  Owned branch/worktree: `codex/v100-flashinfer-sm70-qsa-20261011` /
  `v100-flashinfer-sm70-qsa-20261011`.
- Upstream snapshot and all dependency pins remain in `UPSTREAM.lock.json`.
  Fork version: `0.7.2+1cat.sm70.2`.
- Main was fetched again before publication. No mainline, canonical-checkout,
  other worktree, model weight or runtime environment was modified.

Draft [#1176](https://github.com/1CatAI/1Cat-vLLM/pull/1176) appeared during this
batch. It moves the existing engine-owned provider and extends **single-request**
row tiling with legacy-binary compatibility. This batch provides an independently
installed TVM-FFI owner and **multiple-request** grouping, per-instance plans and
default-off engine selection. It does not edit that PR's native target/provider
or claim its implementation as new performance evidence. When integrating,
preserve its shared-provider import in the legacy fallback, reconcile the
migration log, and rerun the affected source/packaging checks. Neither candidate
is performance-admitted. The older #523 GDN/QSA probe remains a distinct failed
quality/performance experiment; it supplies no admission evidence here.

## Executed checks

| Check | Result |
| --- | --- |
| Standard artifacts | PEP 517 sdist, then wheel from that sdist; no source overlay |
| Independent installation | Python 3.12.3, Torch 2.10.0+cpu, TVM-FFI 0.1.10; ordinary wheel, no installed vLLM; dependency check passed |
| Library CPU suite | **9 passed**; **31 GPU cases deselected**, not executed |
| Engine adapter/dependency tests | **11 passed**; reject forced fallback, retain candidate reason, bounded independent plan reuse, version lock, actual selector handoff |
| Native build/load | Both `basics` and `qsa` compiled from installed wheel sources with nvcc **12.0.140**, SM70 ELF; FFI exports loaded without launching GPU work |
| Native identity | QSA SASS contains Volta `HMMA.884`; no inspected LDSM/LDGSTS/CPASYNC/UTMA instructions |
| Native dependencies | `libtvm_ffi.so`, `libcudart.so.12`, standard C++/C libraries; no Torch C++ or engine extension |
| Source distribution | QSA Python/CUDA/headers, standalone tests, documentation and lock included; wheel contains no task-built DSO |
| Static checks | Owned module Ruff/format and CUDA format; applicable parent hooks, generated architecture reference and runtime configuration ownership |

The initial parent hook run found the adapter's unannotated optional plan and
stale generated line references after the new kernel configuration field. Both
were corrected and the affected checks rerun. The configuration-ownership check
itself passed; its first hook invocation observed concurrent owned-file edits
and was rerun after the tree stopped changing.

Native loads are **not kernel execution**. The local CUDA 12.0 compiler is not
the required CUDA 12.8 release toolchain. All 31 deselected GPU cases (16
primitives and 15 QSA) still need the provided V100-SXM2 environment.

Reproduce with a task-owned Python 3.12 environment, existing Torch 2.10 and
TVM-FFI 0.1.10:

```bash
.venv/bin/python -m build --no-isolation --outdir dist-fi flashinfer-sm70
uv pip install --python .venv/bin/python --no-deps dist-fi/*.whl
uv pip check --python .venv/bin/python
.venv/bin/python -I -m pytest -c flashinfer-sm70/tests/sm70/pytest.ini \
  --confcutdir=flashinfer-sm70/tests/sm70 flashinfer-sm70/tests/sm70 -m 'not gpu'
.venv/bin/python -m pytest --confcutdir=tests/tools \
  tests/tools/test_flashinfer_sm70_adapter.py tests/tools/test_flashinfer_sm70_requirements.py
.venv/bin/python -I -c 'from flashinfer.sm70 import prepare; prepare("basics"); prepare("qsa")'
```

The installed-process checks used no `PYTHONPATH`, `LD_PRELOAD`, private
`LD_LIBRARY_PATH`, mounted source, or external native kernel. JIT outputs live
in the normal configurable cache and derive entirely from installed sources.

## Retained artifacts

Artifacts/logs are retained under the workspace's
`.artifacts/flashinfer-sm70-module-20261011/qsa/` directory:

| Artifact | SHA256 |
| --- | --- |
| `flashinfer_python-0.7.2+1cat.sm70.2-py3-none-any.whl` | `b18477c8b332de3fcb07a8fbae05340cdbbc67411fd01ff63afd97bac0fe279a` |
| `flashinfer_python-0.7.2+1cat.sm70.2.tar.gz` | `dbab43356fc0431ee861bb101916d71225325aebbcab355a25d187b67b500e66` |
| Locally JIT-built `basics` DSO, not a release artifact | `a5ba4175951d256f2ceb547da79011d79a96abaa0d1461bb6223bccb4b3bce15` |
| Locally JIT-built `qsa` DSO, not a release artifact | `542a571fa603cd0bb267e4715da1341da338a866732b7d87fa154f3c2baf3abc` |

The wheel and sdist precede this report-only commit. Loaded source/compiler
identity and resolved installed-package path are in `installed-identity.json`.
No GPU service, profiler, benchmark, API port or persistent task process was
started.

## Open gates and next action

The engine default is `off`. Candidate mode logs nonselection reasons; forced
mode is for supported score shapes and rejects unsupported prefills. No automatic
performance registry entry exists. Full QSA, HC/MoE, GDN, PLE/MTP/output and D72
vision are **unfinished**, as recorded in the [coverage ledger](coverage.md).
Neither these source changes nor the score/selection screen qualify a full chain.

On the supplied V100-SXM2 machine, build ordinary baseline/candidate artifacts
with Python 3.12 / Torch 2.10 / CUDA 12.8, establish stable A/A results, and run the
small exact-shape QSA numerical/replay screen first. Only passing candidates
advance to complete QSA and C1/C4 workloads, supported long-context and
TP/PP/DCP/cache/quantization combinations, MTP acceptance/rollback and model
quality. The 5% complete-chain/integrated gain and 2% other-key-point regression
limits remain unchanged. Vision and 27B/35B-A3B AWQ/FP8 regressions remain required
before any all-path or release claim.
