# Qwen3.8 exact projection and W13 follow-up

Integration: public/main755baae1d075ee04fa9096b23fc0225b23589a86.
Stacked on #507, bbdf0af5c999fa14e2167620921b1173c4ccad76, to preserve the
98.965175 tok/s no-MTP baseline. AI assistance: OpenAI Codex; human review is
required. No integration/main mutation or resident service is authorized here.

## Frozen scope

V100-SXM2-32GB TP4 GPU0–3, Torch2.10.0+cu128, native NVFP4 experts,
FP16 activation/KV/projections, V2 dual graph, hybrid PLE, no MTP or prefix
cache, max262144, chunk8192, M1, deterministic8K/513 performance samples.
Official-sampling natural-output checks are separate. Existing output drift
also occurs on clean old source; see #507's old-source control. Do not hide it
or change arithmetic/NCCL policy to force a particular token sequence.

The latest trace separates row GEMV1.448ms/rank/token into output projections
0.626694 (48 calls), router projection0.420866 (48), QSA QKV0.317551 (12),
and indexer0.082991 (12). W130.670ms, W20.449ms. These are diagnostic GPU
service sums, not additive endpoint wall time.

## Ordered decisions

1. Test the GDN/QSA same-shape plan collision: GDN output load-policy1 is
   overwritten by QSA load-policy0 in the shape-only dictionary. Compare
   actual48 checkpoint output projections with the original FP32 tree and
   CUDA Graph. Do not assume the intended role policy is faster.
2. Only screen a materially different exact loading schedule if the first
   screen and counters justify it. Do not repeat rejected rowblock/CUDA dot
   trees, GDN row-tiling, output-projection/HC fusion, or HC cache sweeps.
3. Optimize native NVFP4 W13 unpack/load scheduling without changing current
   split-K, MMA order, group scaling, FP16 rounding or SwiGLU boundaries.
   Establish narrow kernel counters if proposing software pipelining.

First operator screens; admit only bitwise, changing-input graph-stable
candidates with paired timing gains. A full model run is conditional on
useful gains; zero additional model launches is preferable for failed ideas.
Do not claim100 tok/s or HC1.5ms without matching endpoint evidence.

Open PRs were checked: #504 is batchedHC/QSA, #509 is load-time conversion
cache release, #494 is page4 allocation ordering. None implements this scope.

## Evidence, 2026-09-06

Draft #510 is stacked on #507. No model initialization or service has run in
this follow-up. Task-owned raw artifacts are under `.artifacts/`; compiler
caches, GPU leases and outputs are isolated. The project Python environment
and native frozen DSOs are borrowed read-only.

### Role-aware output projection

The custom op now accepts an optional role string, passed by the model layer.
Explicit roles use the existing role plans; legacy two-argument callers keep
their former shape-only behavior. Unmatched roles/shapes or unsupported inputs
fall back. Dtype, reduction tree, tile size and arithmetic are unchanged.

Real checkpoint weights from all48 layers (36GDN/12QSA), TP rank0's
2560x1536 column shard,64 changing/poisoned CUDA Graph replays: bitwise equal.
Eight warmed, alternating-order paired timing samples measure48 calls at
0.551990 ->0.543309ms, saving0.008681ms. The production registered custom op
also passes16 changing-input graph replays over all48 real weights. This is
a dispatch repair with a small operator gain, not an endpoint speed claim.

CPU routing/allowlist/fallback suite:29 passed,3 unrelated HC GPU tests
deselected. The first CPU-only invocation included those GPU tests and failed
because the NVML-based SM70 marker did not respect hidden CUDA devices; it
was not a source/numerical failure. Do not repeat that invocation or broaden
this task into unrelated HC test-marker edits. Targeted Ruff checks pass.
One additional FakeTensor/export test passes and retains the role string in
the exported custom-op node; this is not a full-model graph compilation gate.

Artifacts: `projection_policy.json`, `projection_dispatch.log`, `cpu_tests.log`.

### W13 small-grid and load-dependency diagnosis

Installed frozen W13 binary and rebuilt generic control are bitwise equal.
The fixed-shape candidate keeps split16/MMA/SwiGLU boundaries and passes64
changing-input graph replays (48 calls each, changing experts/invalid IDs,
duplicates, signed zero and input scales). It reduces registers60 ->48, but
paired latency is0.537928 ->0.537213ms: only0.000714ms saved, not a useful
standalone change. It is not enabled in production.

One model-free privileged NCU capture profiles the dynamic/static kernels.
Original:100 CTAs x512 threads, achieved occupancy30.55%, DRAM352.19GB/s,
no-eligible scheduler cycles66.95%, long-scoreboard stalls46.6% of average
cycles between issued instructions. The grid covers only about0.6 theoretical
waves. Static geometry still gives353.05GB/s and72.32% no-eligible cycles.
NCU durations16.29/15.58us have differing clocks and replay overhead; do not
replace paired graph timings with those values or claim a peak-bandwidth wall.
Do not follow NCU's generic FMA/precision advice: reduction order stays fixed.

These counters justify the next small experiment: prefetch the next packed
weight pair and scale into registers before consuming the current group.
The prefetched candidate also passes64 changing graph replays against the
installed oracle. Paired48-call latency0.539162 ->0.531574ms saves0.007588ms;
registers56, no spills. Still not a sufficient endpoint gain by itself.

Next screen distributes the SAME16 ordered FP32 partials across2/4 CTAs,
then merges in the original0..15 order and retains FP16/SwiGLU boundaries.
This is not changing the numerical split-K grouping. It adds a small FP32
workspace and one merge launch, both included in timing. Pending validation;
no candidate flag is selected by a production launcher.

Artifacts: `w13_static.json`, `w13_prefetch.json`, `w13_counters.ncu-rep`,
`w13_counters_details.txt`, and separate static/prefetch/split-block build logs.
The two initial NCU attempts stopped before profiling (protected `/tmp` lock
open, then per-terminal sudo authentication); use read-only lock descriptors
and explicit sudo authentication. The subsequent capture completed and exited.

Avoid more speculative row-GEMV vectorization: the actual output-projection
PTX already uses64-bit vectorized loads (`ld.global.v2.b32`). A rewrite must
show a different confirmed bottleneck, not assume scalar weight loads.
