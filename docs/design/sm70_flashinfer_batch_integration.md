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
