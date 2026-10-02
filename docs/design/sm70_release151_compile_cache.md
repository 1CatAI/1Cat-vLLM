# SM70 1.5.1 compilation-cache policy

The release requires compilation caching on by default. Existing PR
[#621](https://github.com/1CatAI/1Cat-vLLM/pull/621) removes main's two forced
cache opt-outs. This change depends on it and does not duplicate those edits.

AOT FX-graph restoration and reuse of compiled subgraphs are separate cache
paths. The release's compressed-tensors NVFP4, FP16, TP4/E4M3, q7 DFlash2
verifier contract uses compiled-subgraph caching without AOT graph restoration.
The existing verifier admission and NVFP4 detection are reused. CUDA graph
mode and native kernel arithmetic remain unchanged. Explicit AOT selection is
preserved with a warning about its failed quality gate. Other contracts retain
their existing defaults until their own validation is available.

## Reproduction and measured result

Base: `d30469863287471a7082842500ae73299a697e0d`. Independent main wheel:
`1cat_vllm-1.5.1-cp312-cp312-linux_x86_64.whl`, SHA256
`b3cd873677f17cc779dbf3bb7ffcc03d64d243378df331b9a3ab38e97c32a2f7`.
Torch 2.10.0+cu128, four V100-SXM2 32GB GPUs, QUASAR-QAT 27B NVFP4,
pinned DFlash2 revision `dedf8df68adfb1afeaf7b7480c0a0243108177b4`,
262144 context, 8192 prefill budget, four sequences, memory 0.80,
block2048/mamba8192, prefix caching and FULL_AND_PIECEWISE graphs.

Use the installed official launcher with E4M3 and the pinned local draft.
The diagnostic controller starts fresh server processes with isolated caches
and requests the same complete LRU implementation and tests at temperature 0,
seed 0 and max_tokens 3072. No source overlay or private native library is used.
Explicit cache/AOT variables simulate the candidate policies on the baseline
wheel; this is not final no-variable wheel qualification.

| Cache path | First startup | Subsequent startups | Complete output |
| --- | --- | --- | --- |
| AOT reload | 237.164 s | 76.778 / 73.268 s | Fails parity and a generated test |
| Cache-off control | 182.033 s | Not measured | Matches original cold output |
| Compiled subgraphs, AOT off | 221.114 s | 92.317 / 92.842 s | All three match original cold output |

AOT reload produces 2106 tokens versus the cold reference's 2181, with the
first difference at position 1370. The generated code fails one of 13 tests.
The original complete code passes all 11 tests. Both compiled-subgraph warm
starts preserve every token, load three compiled graphs and compile zero
graphs. Each still loads weights and captures CUDA graphs.

Raw evidence is retained under
`/data/minimax-h3/task-cache/release151-qualification-20260930/compile-cache-20261002/`.
The separately truncated binary-search request and failed AOT reloads are
preserved. Do not repeat the rejected #675 rank or #682 subgraph isolation
experiments as a proposed repair.

## Remaining qualification

The single-request three-start result does not establish seed 1/2 behavior,
MBPP/needle regression, concurrent quality, pure decode speed, or cross-model
quality. The combined source configuration with #621 selects cache-on/AOT-off
without performance variables, but that source-only diagnostic does not
replace a rebuilt wheel. The final combined wheel, Studio user path and full
release matrix remain required before promotion. No PR merge, tag or public
release is performed by this task.
