# TurboMind GGUF dense projection integration

GGUF linear layers prepare independent canonical projections during loading.
Each projection retains its source codec and chooses the existing affine,
LUT4 or lattice mixed-precision kernel. The model scheduler, attention and
CUDA graph lifecycle retain their existing contracts. Mixed fused projections
are evaluated in their logical order and concatenated without treating one
packed format as another.

The first model workload is Qwen3.8-27B UD-Q4_K_M with TP4. Its mixed FFN
weights require both affine and LUT4 preparation. Small output tails and
unsupported canonical coefficients retain the packaged fallback and report
their rejection reason. GDN input layout transforms compose with canonical
storage. Embedding and packed-row PLE preparation are separate integration
scopes. Flash-Next stays TP4.

## Validation

The underlying operators have official dequantization oracles and matched
real-shape timings in the family design documents. This layer additionally
requires mixed projection and layout checks, an ordinary installed-wheel
route check, greedy/logit distribution comparisons and model quality checks.
Model timings must separate prefill from steady decode and report C1/C4/C8/C16
and 8K/32K prompts. Operator timings are not model throughput evidence.

## Layer checks

Fifteen GPU checks pass on V100 32GB, CUDA 12.8 and Torch 2.10.0+cu128.
Independent affine, LUT4 and all seven lattice formats match official
dequantization with FP32 accumulation (relative L2 below 0.003), including
graph replay and full-graph tracing. Mixed affine/LUT4/lattice/FP16 projections
retain their logical order when loaded in a different file order; GDN input
tiling and bias compose with those projections. Preparation preserves shared
source parameters and reports incomplete output packs explicitly.

All 13 existing Qwen3.5 adapter tests also pass. The first layer check uses
the normal main operator artifact with SHA256
`4910c47ab1aaed253001d5950bf44dd40a350b2b087202a8ea2b13f2c5457782`.
Installed-wheel and model-level results remain pending.
