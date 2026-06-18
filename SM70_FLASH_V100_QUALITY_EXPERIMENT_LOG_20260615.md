# SM70 Flash-V100 Quality Experiment Log - 2026-06-15

## Scope

This log records the experiments already run for the Qwen3.6 35B AWQ
long-output numeric gibberish issue. Keep this file updated before running
another variant so we do not repeat the same analysis.

Primary model:

```text
/home/ymzx/models/Qwen3.6-35B-A3B-AWQ
```

Primary runtime:

```text
/home/ymzx/miniconda3/envs/vllm-0.0.5-t210/bin/python
PYTHONPATH=/home/ymzx/桌面/1cat-vllm/vllm
```

Primary prompt:

```text
Please write a technical design document of at least 3000 Chinese characters
about implementing stable long-context inference quality regression tests in
vLLM. Include background, problem definition, metrics, test data, engineering
implementation, risks, and acceptance criteria. Write continuously, do not
produce meaningless numbers, and do not stop early.
```

Sampling used for most checks:

```text
max_tokens=1024
temperature=1.0
top_p=0.95
top_k=20
sampling_seed=20260615
skip_special_tokens=False
```

## Current Conclusion

The reproduced failure was narrowed to:

```text
TP4 + FLASH_ATTN_V100 + FULL decode CUDA graph replay
```

Root cause found:

```text
FlashAttnV100MetadataBuilder.build_for_cudagraph_capture() bound q=1 FULL
decode graph replay to persistent block_table/seq_lens/query_start_loc buffers,
but the normal runtime build() path did not refresh those buffers before replay.
The FULL graph could therefore read capture/dummy decode metadata instead of
the current request metadata.
```

Fix:

```text
Refresh the captured q=1 decode metadata buffers in runtime build() whenever
the buffers already exist from CUDA graph capture.
```

The following are already excluded as sole root causes:

- FP8 KV cache. It can produce numeric gibberish, but the user's API command
  used default KV cache and still reproduced the issue.
- First-token Flash/Triton divergence. 0.0.3 also diverges on the first token,
  so first-token mismatch is not enough to explain the late numeric gibberish.
- 262K max model length. TP4 8K also reproduces.
- Decode dynamic partition selection.
- LM head top1 path.
- torch.compile alone.
- Packed recurrent decode.
- GDN FlashQLA decode route alone.

The strongest pre-fix evidence was that TP4 Flash-V100 eager output was normal,
while TP4 Flash-V100 FULL decode graph output was bad. TP2 Flash-V100 FULL graph
was also normal. The fix directly updates the graph replay metadata buffers
instead of falling back to Triton or disabling CUDA graph.

## Reproducer Baseline

The user reproduced the issue with an OpenAI API server command equivalent to:

```bash
/home/ymzx/miniconda3/envs/vllm-0.0.5-t210/bin/python \
  -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model /home/ymzx/models/Qwen3.6-35B-A3B-AWQ \
  --served-model-name qwen36-35b-awq \
  --quantization awq \
  --tensor-parallel-size 4 \
  --dtype half \
  --max-model-len 262144 \
  --max-num-batched-tokens 16324 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.88 \
  --attention-backend FLASH_ATTN_V100 \
  --trust-remote-code \
  --disable-log-requests
```

Changing only the attention backend to `TRITON_ATTN` made the output normal in
the user's API test.

## Experiment Matrix

| ID | Variant | Artifact | Result | Conclusion |
| --- | --- | --- | --- | --- |
| A | User API, TP4, default KV, Flash-V100 | User API test | Output becomes numeric/date gibberish | Real serving-like Flash path reproduces |
| B | User API, TP4, default KV, Triton | User API test | Output normal | Common AWQ/MoE path is not enough to fail |
| C | 0.0.3 Flash vs Triton first-token comparison | Local first-token test | First token also diverges in 0.0.3 | First-token divergence is not the final root |
| D | TP4, Flash-V100, FP8 E5M2 KV | `/tmp/vllm_flash_fp8kv_35b_awq_long_result.json` | 1024 tokens, digit_ratio 0.5925, hash `bb2e2de3d07aea8b9a81a524b9206e1d94abfb7727aa8d4996ac5a6cfa3fd4c0` | FP8 KV is a separate bad path, not the user's default-KV root |
| E | TP4, 262K, default KV, Flash-V100 | `/tmp/vllm_flash_defaultkv_35b_awq_tp4_262k_long_result.json` | 766 tokens, digit_ratio 0.1286, weird_runs 6, hash `041eeb92c2b601e91410a6f6c97de064a52adf471b50c7dceea1a1366717795e` | Reproduces numeric gibberish without FP8 KV |
| F | TP4, 262K, default KV, Triton | `/tmp/vllm_triton_defaultkv_35b_awq_tp4_262k_long_result.json` | 1024 tokens, digit_ratio 0.0036, weird_runs 0, hash `8d73ffe16362da5f1de6e44732810be98e53c723a98c1e0f3331e268b4578053` | Triton quality baseline is clean |
| G | TP4, 262K, Flash-V100, dynamic partitions off | `/tmp/vllm_flash_defaultkv_35b_awq_tp4_262k_dynpart0_long_result.json` | Same bad hash as ID E | Dynamic partition count is not root |
| H | TP4, 8K, Flash-V100, FULL graph | `/tmp/vllm_flash_defaultkv_35b_awq_tp4_8k_long_result.json` | 1024 tokens, digit_ratio 0.1296, weird_runs 5, hash `07a2237ac6197cff4b63d098dd4aa54a3dc802c732695ff84f5142d8fddf62c1` | 262K max length is not required |
| I | TP2, 8K, Flash-V100, FULL graph | `/tmp/vllm_flash_defaultkv_35b_awq_long_result.json` | 767 tokens, digit_ratio 0.0141, weird_runs 0, hash `9913c3ac982dfbaf3807326c9020001daae66b854629b03dfcce516afca51e20` | Failure is TP4-sensitive |
| J | TP4, 8K, Flash-V100, enforce eager | `/tmp/vllm_flash_defaultkv_35b_awq_tp4_8k_eager_long_result.json` | 1024 tokens, digit_ratio 0.0064, weird_runs 0, hash `23309b95652def8e3e365a5f333a22e5e28b813a0aa0e5227f31fd5f886f1fc7` | Flash kernel path is not intrinsically bad in eager |
| K | TP4, 8K, Flash-V100, eager, LM_HEAD_TOP1=0 | `/tmp/vllm_flash_defaultkv_35b_awq_tp4_8k_eager_lmhead0_long_result.json` | Same clean hash as ID J | LM_HEAD_TOP1 is not root |
| L | TP4, 8K, Flash-V100, no torch compile but FULL decode graph | `/tmp/vllm_flash_defaultkv_35b_awq_tp4_8k_no_compile_graph_long_result.json` | 1024 tokens, digit_ratio 0.2227, weird_runs 3, hash `35879abcf3f2553c4759f83e83e367c54a4e250900d8c297a459dc76133ff342` | torch.compile is not required; FULL decode graph is sufficient |
| M | TP4, 8K, Flash-V100, packed recurrent decode off | `/tmp/vllm_flash_defaultkv_35b_awq_tp4_8k_no_packed_recurrent_long_result.json` | 1024 tokens, digit_ratio 0.5254, weird_runs 1, hash `72f98b3e4986695f6fe7fe2e6014d8167b195b6e2580bf3c2d4b7288ce381b41` | Packed recurrent decode is not root |
| N | TP4, 8K, Flash-V100, GDN FlashQLA decode off | `/tmp/vllm_flash_defaultkv_35b_awq_tp4_8k_no_gdn_flashqla_long_result.json` | 1024 tokens, digit_ratio 0.3962, weird_runs 2, hash `6184b947abd84e4f97521f6cfa925252edcca23c944a09955d88d55b639a348f` | GDN FlashQLA route alone is not root |
| O | Fixed TP4, 8K, default KV, Flash-V100, compile + FULL graph | `/tmp/vllm_flash_defaultkv_35b_awq_tp4_8k_graph_metadata_fix_long_result.json` | 1024 tokens, digit_ratio 0.0060, weird_runs 0, steady_decode_tps 121.269, hash `bfdaac78c6b9de453d838bc3f44b0740e28136b27ffa97d123ffe2fe94ccdadd` | Numeric gibberish fixed while keeping Flash FULL graph speed |
| P | Fixed TP4, 262K, default KV, Flash-V100, compile + FULL graph | `/tmp/vllm_flash_defaultkv_35b_awq_tp4_262k_graph_metadata_fix_long_result.json` | 1024 tokens, digit_ratio 0.0112, weird_runs 0, steady_decode_tps 119.431, hash `f20764a41806a6bbdf29efd521a237774ec90d8e3272569cb6a5ef60e6e5cf62` | User API-scale 262K configuration fixed while keeping Flash FULL graph speed |
| Q | Fixed TP4, 262K, 8-prompt dataset, default KV, Flash-V100, compile + FULL graph, natural EOS | `vllm/bench_results/flash_v100_quality_fix_20260615/qwen36_35b_awq_tp4_flash_fullgraph_262k_quality_prompts8_o1024_no_ignore_eos.json` | `ignore_eos=False`, 4095 total output tokens, per-prompt token counts `[75, 1024, 1024, 209, 295, 1024, 323, 121]`, max digit_ratio 0.0195, weird_runs 0, repeated_num_runs 0, max_digit_run 4, aggregate output throughput 109.571 tok/s | Dataset regression no longer shows numeric/date gibberish under the production Flash FULL graph lane |

The aborted `--ignore-eos` dataset attempt is intentionally not used as quality
evidence. Output-quality judgment should respect natural EOS because forcing the
model to continue after it wants to stop can pollute the failure signal.

## Do Not Repeat

Do not rerun these as generic proof:

- Flash vs Triton first-token A/B. It is already known to split and is not
  sufficient to explain late numeric gibberish.
- 262K vs 8K max length. 8K already reproduces.
- Dynamic partition on/off. It already produced an identical bad hash.
- LM_HEAD_TOP1 on/off in eager. It already produced an identical clean hash.
- Packed recurrent decode off. It does not fix the issue.
- GDN FlashQLA off. It does not fix the issue.

Only rerun a previous variant if a code change specifically targets that
variant and the run is used as a regression check.

## Next Regression Checks

If this area regresses again, do not start with broad backend A/B tests. First
check whether q=1 graph replay metadata buffers are refreshed before
`run_fullgraph()`, then rerun ID O and ID P.
