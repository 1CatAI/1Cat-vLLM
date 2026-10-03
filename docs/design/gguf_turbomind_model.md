# TurboMind GGUF dense projection integration

GGUF linear layers prepare independent canonical projections during loading.
Each projection retains its source codec and chooses the existing affine,
LUT4 or lattice mixed-precision kernel. The model scheduler, attention and
CUDA graph lifecycle retain their existing contracts. Mixed fused projections
are evaluated in their logical order and concatenated without treating one
packed format as another.

The first model workload is Qwen3.8-27B UD-Q4_K_M with TP4. Its mixed FFN
weights require affine, LUT4 and lattice preparation. Four FFN projections
use IQ3_S despite the checkpoint's Q4_K_M name. Small output tails and
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

## Installed artifact and model correctness

An ordinary wheel in a fresh runtime passes all 15 layer GPU checks with
210 compatible dependencies. The source and installed canonical extension
are the normal main artifact, SHA256
`5cd0fa29e533f92644e012c57fe7b439293bf360e8988b8d73d7bbef54839f6a`.
Wheel `1cat_vllm-1.5.2.dev416+gba6295209.precompiled-cp312-cp312-linux_x86_64.whl`
has SHA256 `b25b8acbcb370df4a0e07a3316720aca4edb39cf746e9f7d625e8f1f28c81c47`.
No private extension or Python-path override is required.

The Qwen3.5-0.8B TP4 regression check keeps 64/64 English and arithmetic
tokens, with first-logit RMSE reduced to 0.109217/0.118177/0.144899 across
the three fixed prompts. Chinese still diverges after token 23; this issue
is not resolved. Embedded chat results remain `Paris`, `4`, and `你好`.

Qwen3.8-27B UD-Q4_K_M completes installed-wheel TP4 inference with FP16,
maxlen 2048, maxbatch 256, maxseqs 4, memory utilization 0.3, eager and no MTP.
Worker logs select the three canonical families and Flash-V100/FlashQLA.
The four fixed GGUF/HF chat tokenizations match exactly. Greedy output bodies
match pinned CPU llama.cpp for `Paris`, arithmetic, translation and a
29-token Chinese explanation of lunar phases; all stop normally.

| Prompt | First-logit RMSE | Relative L2 | Reference-to-actual KL | Top-20 overlap |
| --- | ---: | ---: | ---: | ---: |
| Paris | 0.114157 | 0.036342 | 0.00006944 | 18/20 |
| Arithmetic | 0.175268 | 0.065481 | 0.00003251 | 18/20 |
| Translation | 0.127382 | 0.046972 | 0.00005601 | 20/20 |
| Lunar phases | 0.077471 | 0.036478 | 0.00374037 | 19/20 |

All four top-1 logits match. These short checks establish model operation;
the common quality set and C1/C4/C8/C16 plus 8K/32K performance comparisons
are still pending. The 27B IQ3_S gate/down TP4 shapes additionally require
prefill crossover measurements; the existing Flash-Next calibration does
not establish their crossover.
