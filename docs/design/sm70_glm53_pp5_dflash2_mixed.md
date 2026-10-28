# GLM-5.3-Flash on three V100 and two RTX 8000: PP5 with DFlash2

This note records how GLM-5.3-Flash (NVFP4 experts with csf_32x2_v1 scales, MXFP8 dense and attention
projections) runs on a mixed Volta/Turing box, five pipeline stages with one request at a time, DFlash2 as the draft
model, and what the memory accounting looked like before it did. The numbers are single boots on the fork's tree at
the time, not A/B series unless a line says so, and predate the last commits of this series (five-bit scale planes,
CSF records written by the expansion kernel), whose own measurements are in their commit messages.

## Layout

- Cards in pipeline order (`CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=0,1,4,3,2`): RTX 8000, V100, V100, V100,
  RTX 8000. `VLLM_PP_LAYER_PARTITION=14,7,7,7,10`: a V100 holds at most seven MoE layers, the RTX cards take the rest.
- `--pipeline-parallel-size 5 --max-num-seqs 1 --max-num-batched-tokens 512 --max-model-len 524288
  --kv-cache-dtype fp8_e4m3 --gpu-memory-utilization 0.98 --enable-prefix-caching`.
- Draft: `--speculative-config` with `method dflash`, the DFlash2 drafter, `num_speculative_tokens 7`,
  `draft_sample_method greedy`, `kv_cache_dtype auto`, `attention_backend FLASH_ATTN` (FA2). The draft runs on the last
  stage, an RTX 8000, where the V100-only `flash_attn_v100` kernels refuse to start (`Kernel supports only Volta GPUs`).
- The token embedding (1.18 GiB, and its replica for the draft on the last stage) lives in pinned host memory:
  `--cpu-offload-gb 1.4 --cpu-offload-params embed_tokens`. A step reads a few rows; the table does not need to occupy a
  card. Measured cost: 8 tokens +20 us per step, a 512-token chunk +1.24 ms.

## Memory accounting that had to change

The first DFlash2 boot served 15,591 tokens of context. Four findings: the first two are fixed by this series, the
other two are limits to plan around.

1. The profile run reserved the up-projected context output of `kv_b_proj` (`chunked_prefill_workspace_size x heads x
   (qk_nope + v) x 2 B`) on every stage: 1.07 GB at a 16k maximum length, 4 GiB from 64k tokens. Sparse MLA
   implementations only run `forward_mqa` and never use it. The V100 graph reserve is defined as the activation peak, so
   it shrank with it. Activation peak per card: 1.16 GB to 0.57 GiB.
2. The sparse indexer gathered keys into a buffer of `40 x max_model_len` entries of 132 B (5.16 GiB at 1M tokens). It
   now holds `min(40, max_num_seqs) x max_model_len` entries.
3. A request needs `groups x (2 + k)` fixed pool blocks for the KDA state (six groups, k = 7: 54 blocks of 15.3 MB on the
   first stage). The group count is global and set by the stage with the fewest MLA layers.
4. `persistent_topk` (indexer top-k) needs all of its blocks resident: 72 SMs on an RTX 8000 times 8,909 entries per
   block, about 641k tokens. The `FilteredTopK` fallback needs 128 KB of shared memory, Turing has 64 KB.

Result: 667,683 tokens of KV cache with `--max-model-len 524288` (auto-fit to 801,792 failed on finding 4).

## Results

Chunk 512, one request, greedy, prompts of 8,687 tokens for prefill. One boot each, not an A/B: the run without draft
used `VLLM_PP_LAYER_PARTITION=13,7,7,7,11` and CUDA graphs for one token, the DFlash2 run 14,7,7,7,10 and graphs for
eight tokens. Units: tok/s (higher is better), s and ms (lower is better).

| | no draft | DFlash2, k = 7 |
|---|---|---|
| KV cache pool as reported by the engine (tokens; `--max-model-len 524288`) | 1,167,414 | 667,683 |
| Prefill, 8,687 tokens (s) | 4.26 | 4.54 |
| Decode, 192 tokens after a 30-token prompt (tok/s) | 24.02 | 32.3 |
| Decode after 8.7k tokens of context (tok/s) | 23.73 | 44.1 |
| GSM8K, 60 questions, thinking on, temperature 0 (tok/s) | 23.26 | 53.35 |
| GSM8K, correct | 59 / 60 | 59 / 60 |
| GSM8K, mean acceptance length (gate 4.85, reference 5.53) | - | 5.511 |
| per-position acceptance, 1..7 | - | 0.920 0.813 0.719 0.622 0.554 0.471 0.411 |

The one wrong GSM8K answer is question 12 in both runs ("how many years until he starts earning money": the
reference counts the year after break-even, 12 versus 13).

Decode step time at long context, ms per step (the draft run verifies several tokens per step; at the measured
acceptance this is 24.8 to 41.8 tok/s against 14.4 to 23.0 tok/s without the draft):

| prompt tokens | 31k | 170k | 283k | 516k |
|---|---|---|---|---|
| no draft, ms per token | 43.4 | 49.2 | 56.0 | 69.6 |
| DFlash2, ms per round | 96.4 | 104.4 | 119.1 | 129.9 |
| time to first token, DFlash2 (s) | 16.7 | 85.5 | 144.8 | 273.1 |

## Variants that were measured and not kept

- Last stage on a V100 (`CUDA_VISIBLE_DEVICES=0,1,3,2,4`, partition 14,7,7,11,6, draft attention `FLASH_ATTN_V100`):
  faster in one boot each against the RTX 8000 last stage (short decode 37.15 against 32.33 tok/s, GSM8K 54.3 against
  51.6 tok/s, acceptance 5.614 against 5.461), but 10 % less KV (597,105 tokens) and 0.8 GB of free memory on the last
  card; the RTX 8000 last stage was kept for the KV pool.
- The GLM verifier switches that the TP8 route sets (`VLLM_SM70_GLM53_MHC_NATIVE_VERIFY`, `..._MHC_FUSED_POST_DOT_Q8`,
  `VLLM_SM70_GLM_MHC_PRE_THREADS=1024`, `VLLM_SM70_DFLASH2_VERIFY_FASTPATH`, `VLLM_SM70_DFLASH2_FUSED_GDN_METADATA`):
  A/B/A/B shows +1.8 % short decode, -2 % after 8.7k, +2.2 % GSM8K, texts differ; no consistent gain.

## Operating notes

- Weights load in 7 to 12 minutes depending on swap pressure (31 GB of host RAM, five workers of 3 to 5.5 GB each).
- `--max-model-len` must stay at or below about 641k on RTX 8000 stages (finding 4).
- The first stage keeps 2.5 GiB for KV, the last 3.4 GiB; peak `memory.used` per card (MiB, PCI order):
  47515 / 30408 / 47809 / 29712 / 30452.

## Prefix cache

The hybrid cache keeps prefix blocks of 9,216 tokens (one Mamba page). With the draft model attached, a 62,372-token history
reaches its first token in 32.1 s the first time and in 4.97 s the second time (55,296 tokens served from the cache); a
follow-up question on the same history takes 4.31 s.
