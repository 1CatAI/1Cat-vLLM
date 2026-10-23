# GGUF DFlash2 on two and four SM70 GPUs

The split-window draft attention implementation shares its FP32 score, PV,
and reduction arithmetic across local Q/KV head layouts 16/4 (TP2) and 8/2
(TP4). Native concurrent attention remains restricted to the qualified 8/2
layout. The target model exposes its GDN output collective at both TP sizes;
the shared graph pattern selects the existing TP2 or TP4 allreduce/Gemma
RMSNorm operator and preserves the exact configured epsilon. The normal
`fuse_allreduce_rms` pass switch controls both paths.

## Memory and execution check, 2026-10-07

Hardware: four V100-SXM2-16GB GPUs, pairwise NV2 links, driver 580.178.04,
CUDA 12.8, Torch 2.10.0+cu128, Python 3.12.3. TP2 uses GPUs 0 and 1;
GPUs 2 and 3 remain unused. SM clocks are not locked and vary during these
checks; memory clocks are 877 MHz. These are execution checks, not matched
performance baselines against earlier 1290 MHz runs on 32GB GPUs.

Target: Qwen3.8-27B-GSQ-RCO-IQ3_S, 11,771,546,784 bytes (10.963 GiB),
SHA256 `64b53b64c7aa39f20a7e54bd80582fe595b1d745624ee8a72e92508c0326d810`.
Draft: Qwen3.8-27B-DFlash2-Q8_0, 2,056,414,816 bytes (1.915 GiB),
SHA256 `c18e800daedc59ca68fd13b6a856d795746af6d399a9279ac6a277d1d422f87e`.

Both use FP16 activations, seven speculative draft tokens, CUDA Graph,
max length 32768, max sequences 1, batch budget 1024, GPU memory utilization
0.88, target E4M3 KV, draft FP16 KV, and FP32 SSM state. Native accumulation
precision is unchanged. The control wheel contains runtime source
`2b140abbf1`; the shared-path candidate contains `dfdb2b7fb8` with byte-identical
native libraries. Subsequent test-location and documentation changes do not
alter installed runtime code.

| TP size | Target + draft loading, GiB/rank | Target unique tensor storage, GiB/rank | Final reserved, GiB/rank | Peak GPU memory, MiB |
| --- | ---: | ---: | ---: | --- |
| 2, control | 9.53 | 7.559 | 13.574 | 15169 / 15201 |

The TP2 hybrid KV/state budget is 3.42 GiB/rank and reports capacity for
61,440 tokens. This does not establish execution at that context length:
the configured and tested maximum is 32K, with actual input lengths 1K/8K.
Model loading, graph/workspace allocation, cache backing, and allocator
reservation all contribute to GPU usage; file size alone is insufficient.

The same two deterministic health prompts finish normally in the control
and candidate: the arithmetic answer is `391`, and the English answer is
a coherent unit-test explanation. This establishes text health, not a
full quality-set result or a model-logit KL measurement.

## Short C1 execution results

Each request generates 128 tokens with temperature 0.7, top-p 0.9, top-k 20,
seed 123. EOS is ignored only for the fixed-length timing fixture; the health
prompts respect EOS. Twenty full rounds are discarded. The remaining sample
count is small, and clocks and accepted lengths vary; no performance gain
is attributed to the shared-path change from these measurements.

| Runtime | TP | Input | Full round, ms | Emitted tokens/full round |
| --- | ---: | ---: | ---: | ---: |
| control | 2 | 1024 | 31.015 | 3.538 |
| shared boundaries | 2 | 1024 | 31.050 | 4.182 |
| control | 2 | 8192 | 31.936 | 3.083 |
| shared boundaries | 2 | 8192 | 31.650 | 3.250 |
| control | 4 | 1024 | 14.272 | 4.182 |
| control | 4 | 8192 | 14.647 | 3.083 |

The candidate logs confirm TP2 allreduce/Gemma RMSNorm pattern replacement.
Projection-plane admission still requires separately measured TP2 shapes;
sharing attention and norm boundaries does not establish that all projection
acceleration is selected.

Installed-artifact tests: 56 CPU dispatch, norm-epsilon and model-boundary
checks pass; 34 GPU attention checks pass. GPU coverage includes both local
head layouts, four page sizes, indirection, live and zero lengths, changed
inputs and graph replay, with an FP64 oracle. Native symbol absence and
unsupported concurrent layouts retain fallback.
