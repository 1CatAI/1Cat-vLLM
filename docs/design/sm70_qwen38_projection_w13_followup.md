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

## Evidence

Pending. Task-owned raw artifacts are under `.artifacts/`; compiler caches,
GPU leases and outputs are isolated. The Python environment and unchanged
native DSOs are borrowed read-only from the previous accepted runtime.
