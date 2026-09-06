# FlashInfer SM70 batch integration probe

Purpose: combine operator-screen winners, then perform one consolidated
control/candidate model comparison. No release default is changed.

Integration base: `onecat/main` at `755baae1d075ee04fa9096b23fc0225b23589a86`.
This owned branch combines #504 at `5a049230cd` (existing batch HC/projection
and TP baseline), #513 at `17fa36e35a` (native FlashInfer QSA), and #515 at
`9ea54c19f4` (fused gate/conv/GDN prototype). Combination commit `f8e822c335`.
The runtime bridge is new, not a duplicate of those component-only PRs.

## Decisions and contracts

- Retain existing M1, HC projection/TP and HC norm paths. Both FlashInfer HC
  staging variants passed numerical tests but lost to the existing Triton
  component. On GPU 4, FP16 B16 baseline 3.164 us, register variant 3.779 us;
  no HC replacement is admitted.
- `VLLM_SM70_FLASHINFER_BATCH=0` by default. Explicit prebuilt GDN and QSA
  library paths are used only in the probe; no compiler or JIT during capture.
- GDN preserves FP16 activations/projections and FP32 recurrent state. Fuse at
  the existing opaque input/core boundary to remove, not duplicate, the BA
  GEMM. Ordinary QKVZ projection and output norm/projection remain unchanged.
- GDN dispatch uses actual non-spec, uniform-decode metadata and compatible
  tensor geometry (2..64 rows). M1, larger shapes, prefill/mixed/MTP and other
  state/activation layouts fall back. No global TP, scheduler budget or KV
  dtype binding. Initial model validation is no-MTP; spec engines are not armed.
- QSA accepts the proven small-batch FP16 sparse-page family (4..16 rows),
  preserving index order, repeats, invalid-page masking and the FP16 output
  boundary before the existing output gate. E4M3 and other shapes fall back.
- Scratch allocations are call-local, never a cross-stream global mutable
  workspace. The zero page is persistent/read-only. Derived-weight reloads
  copy in place and reject changed geometry rather than replacing graph pointers.
- Do not change selector, expert routing, sampling, state precision or output
  length to manufacture a throughput gain.

## Validation and measurement

The component worklogs retain micro timings and reference/source/binary hashes.
On GPU 4, #515's updated suite passed **20 tests**, including register-HC
cases (66.92 s including rebuild). GDN targeted memcheck passed one B16 case
with **0 errors**; racecheck passed one B4 case with **0 errors/0 warnings**.
These are targeted operator checks, not whole-engine sanitizer coverage.

Initial bridge suite: **21 passed** on GPU 4 (11.67 s), including actual QSA
output gating, GDN input/core state evolution and eager/graph equality. A CPU
no-op initially imported the GDN module before its guard; fixed by guarding
first, so unsupported devices do not load native libraries.

The next consolidated run uses the unchanged 8192-input/256-output fixed-live-
width benchmark, C1/4/8/16, no MTP, NVFP4 FlashNext, TP4, 256K capacity, prefix
cache + Mamba align, FP16 QSA KV, FP32 GDN state. CUDA Graph, GPU 4--7; both
arms use identical baseline flags and NUMA-node-1 CPU affinity. Keep automatic
GPU boost; do not compare absolute timings with prior GPU 0--3 runs.

Use full EngineCore timestamp intervals, not client receive blocking time.
The fixed denominator remains 70 tok/s: C4/C8/C16 targets 238/420/728 tok/s.
The deterministic performance workload remains unchanged and is **not** quality
evidence. Its historical forced-length policy does not apply to quality cases.

Quality: reuse the fixed GSM8K first-16 health screen (temperature 1, top-k20,
top-p.95, natural EOS, max16384), plus the retained 80-case BFCL/schema manifest
as an offline generation/parser screen. Explicitly distinguish offline model
outputs from HTTP/SSE transport testing and official benchmark scores. Report
scores, incomplete outputs and regressions; do not claim broad quality from
a short smoke. No candidate model result has been produced yet.

Artifacts are task-private under this worktree's `.artifacts/`: native build
caches, `bridge-tests-gpu4-v2.log`, `env.sh`, `run-e2e.sh`, `run-model.py` and
`e2e/`. Frozen native vLLM/Flash-V100/HC libraries are pinned reference binaries;
this is not a freshly built release wheel. No shared checkout is modified.

The user freed GPU 4--7. The old **local** `1cat-qwen38-flash-next.service`
was stopped and disabled; remote delivery services were not changed. Stop all
task-owned engine workers after each comparison. Keep this PR Draft until
the model and quality gates pass. AI-assisted (Codex), DCO sign-off and human
review required before merge.

## First model comparison and compiled-boundary correction (September 6)

Draft integration PR: #523. Both arms used the frozen speed runner and
environment above on GPU 4--7. The control reproduced the historical baseline:

| Concurrency | Control tok/s | First candidate tok/s |
|---|---:|---:|
| 1 | 87.750 | 87.864 |
| 4 | 217.026 | 219.474 |
| 8 | 364.521 | 375.581 |
| 16 | 587.789 | 612.823 |

These first candidate numbers are **QSA only**, not combined GDN + QSA.
Although all four ranks prepared 36 GDN layers, no fused GDN route was hit.
The `1 < num_tokens <= 64` test was outside the opaque input/core boundary:
the initial large-prefill trace specialized that branch away. Move all shape
and metadata selection inside the runtime op. Preserve the existing fused
FP16 input projection on fallback, including M1, instead of substituting
separate QKVZ/BA GEMMs. The boundary has an extra Z output copy; its M1 cost
and full-model behavior still need the corrected combined comparison.

Added an export regression using the actual boundary predicate: one large
example must support the full 1..2048 row range. Reinstating the legacy shape
guard demonstrably fails export's shape constraints. The corrected predicate
and original-projection fallback checks pass: **24 CPU tests**, 5.63 seconds.
The first test attempt had a missing CPU-only op stub; fixed the fixture,
not the model implementation. The native operator binaries are unchanged.

Quality is **not admitted**: GSM8K is 15/16 with no truncation in both arms,
but the fixed offline BFCL screen is **52/64 -> 50/64** (irrelevance 13 -> 11).
JSON Schema remains 16/16. Retain all per-case records; a small stochastic
screen cannot establish noninferiority, and this negative signal cannot be
ignored in exchange for C16's 4.3% gain. Do not enable by default.

Artifacts: `.artifacts/e2e/control-*`,
`.artifacts/e2e/qsa-only-190e5f0225/candidate-*`,
`.artifacts/control-e2e-v1.log`, `.artifacts/candidate-e2e-v2.log`,
and `.artifacts/compiled-boundary-cpu-v2.log`.
Candidate attempt v1 exited at the GPU lease gate (75), before a model run;
it produced no performance result. Per-device external campaign leases are
also respected by the launcher now. All first-comparison workers exited;
GPU 4 was subsequently acquired by an unrelated sanitizer job, which is not
terminated by this task.

The next combined attempt at `45711b5e21` passed compilation but failed during
FULL graph capture, before any quality or speed request. AOT compilation
preceded KV allocation, so the input/core op received empty 1-D state
placeholders. The new bridge tried to transpose them. The existing standard
recurrent core already resolves this case from the scheduler-bound layer
cache; reuse that contract inside the new runtime bridge. Explicit nonempty
state inputs must never be replaced. Add both placeholder and explicit-cache
graph/state tests, including a different layer cache to catch accidental
replacement, and guard malformed/unbound states before transpose.

Failure log: `.artifacts/candidate-e2e-v3.log`. It is a startup failure, not a
performance or quality result. Its workers exited. Run the expanded GPU
bridge suite under the same GPU lease before the next model launch.
