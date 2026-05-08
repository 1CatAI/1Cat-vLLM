# FA2 vs Triton Long-Context Regression

Date: 2026-05-07

## Scope

Target version: `1Cat-vLLM-0.0.3`

Environment:
- Python: `/home/ymzx/miniconda3/envs/1cat-vllm-0.0.3/bin/python`
- vLLM: `0.0.3.dev0+g72bb24e2d.d20260430.cu128`
- Torch/CUDA: `torch 2.9.1+cu128`, CUDA 12.8
- GPUs: V100-class `Tesla PG503-216`, sm70, GPUs `1,2,3,4`

Model-level targets:
- Model: `/home/ymzx/models/Qwen3.5-35B-A3B-AWQ`
- TP: 4
- dtype: float16
- 32k run: `max_model_len=32768`, `max_num_batched_tokens=32768`, `max_num_seqs=4`
- 256k run: `max_model_len=262144`, `max_num_batched_tokens=8192`, `max_num_seqs=1`
- `disable_custom_all_reduce=True`
- `prompt_logprobs=0`, because prompt logprob all-gather OOMs on V100 TP4.

## Artifacts

- Operator script: `tools/fa2_triton_long_context_regression.py`
- Model script: `tools/vllm_v100_backend_regression.py`
- Operator quality: `fa2_triton_ops_quality_longctx_blk528.json`
- Operator speed: `fa2_triton_ops_speed_longctx_blk528_clean.json`
- Operator 256k: `fa2_triton_ops_256k.json`
- Operator TP4-local decode: `fa2_triton_ops_decode_tp4local_scriptcheck_20260508.json`
- Model combined result: `model_backend_longctx_qwen35_35b_tp4_disablecar/combined.json`
- Model logs: `model_backend_longctx_qwen35_35b_tp4_disablecar/*.log`
- Model 256k combined result: `model_backend_256k_qwen35_35b_tp4_disablecar_mbt8192_fast/combined.json`
- Model 256k logs: `model_backend_256k_qwen35_35b_tp4_disablecar_mbt8192_fast/*.log`
- Model 256k FA2 decode65 result: `model_backend_256k_fa2_decode65_20260508/FLASH_ATTN_V100.json`
- Model 256k FA2 decode257 result: `model_backend_256k_fa2_decode257_20260508/FLASH_ATTN_V100.json`

## Operator Results

All operator quality cases passed, including Qwen3.5 MoE shape `q_heads=16`, `kv_heads=2`, `head_dim=256`, and model-real attention block size `528`.

Quality max error:
- Dense full prefill: max_abs <= `9.77e-4`
- Prefix/chunked prefill: max_abs <= `6.10e-5`
- Decode: max_abs <= `3.05e-5`

Speed median on one V100:

| Case | FA2 | Triton | FA2 speedup |
|---|---:|---:|---:|
| dense full prefill, b1 s4096, block16 | 10.035 ms | 63.175 ms | 6.30x |
| dense full prefill, b1 s4096, block528 | 10.048 ms | 64.824 ms | 6.45x |
| prefix prefill, q512 ctx8192, paged FA2 | 6.939 ms | 32.646 ms | 4.71x |
| prefix prefill, q512 ctx8192, default gather+dense, block16 | 5.508 ms | 32.646 ms | 5.93x |
| prefix prefill, q512 ctx8192, default gather+dense, block528 | 5.506 ms | 33.036 ms | 6.00x |
| prefix prefill, q1024 ctx16384, paged FA2 | 27.104 ms | 130.605 ms | 4.82x |
| prefix prefill, q1024 ctx16384, default gather+dense | 20.712 ms | 130.605 ms | 6.31x |
| decode, b1 ctx8192 | 0.148 ms | 0.562 ms | 3.79x |
| decode, b4 ctx8192, block16 | 0.468 ms | 0.754 ms | 1.61x |
| decode, b4 ctx8192, block528 | 0.471 ms | 0.748 ms | 1.59x |
| decode, b1 ctx32768 | 0.486 ms | 1.397 ms | 2.87x |

Operator conclusion: FA2 is not slower than Triton for long-context full attention prefill/decode. The model-real `block_size=528` path is also correct and faster.

### Operator 256k Results

The 256k operator run used the Qwen3.5 MoE attention shape `q_heads=16`, `kv_heads=2`, `head_dim=256`, and the real model block size `528`. All quality checks passed.

| Case | FA2 | Triton | FA2 speedup | max_abs |
|---|---:|---:|---:|---:|
| prefix prefill, q512 ctx262144, block528 | 173.500 ms | 1115.976 ms | 6.43x | 7.63e-06 |
| prefix prefill, q1024 ctx262144, block528 | 343.027 ms | 2219.047 ms | 6.47x | 7.63e-06 |
| decode, b1 ctx262144, block528 | 3.412 ms | 9.306 ms | 2.73x | 7.63e-06 |
| decode, b4 ctx262144, block528 | 13.249 ms | 15.803 ms | 1.19x | 7.63e-06 |

Operator 256k conclusion: FA2 is substantially faster for prefill and decode at 262144 context. The b4 decode case is only modestly faster because non-attention overhead and batching reduce the relative attention share, but it is still not slower.

### TP4-Local Decode Follow-Up

The model-level run uses TP4. For Qwen3.5-35B-A3B-AWQ, the full-attention config is total `q_heads=16`, `kv_heads=2`, `head_dim=256`, so each TP rank runs local `q_heads=4`, `kv_heads=1`, `head_dim=256`.

The TP4-local operator decode run passed quality checks and produced:

| Case | FA2 | Triton | FA2 speedup |
|---|---:|---:|---:|
| decode, b1 ctx8192, h4 kv1 | 0.084 ms | 0.611 ms | 7.28x |
| decode, b1 ctx32768, h4 kv1 | 0.182 ms | 1.615 ms | 8.86x |
| decode, b1 ctx65536, h4 kv1 | 0.331 ms | 2.950 ms | 8.92x |
| decode, b1 ctx131072, h4 kv1 | 0.618 ms | 5.124 ms | 8.30x |
| decode, b1 ctx262144, h4 kv1 | 1.089 ms | 9.964 ms | 9.15x |
| decode, b4 ctx262144, h4 kv1 | 3.566 ms | 15.281 ms | 4.29x |

TP4-local decode conclusion: the FA2 decode kernel itself does not show an abnormal 256k regression. At the real per-rank shape, one 256k full-attention layer costs about `1.09 ms`; the model has 10 full-attention layers, so the full-attention part is roughly `10.9 ms/token` before projection, MLP/MoE, linear-attention, communication, scheduler, and sampling overhead.

## Model Results

The model logs confirmed real backend use:
- FA2 decode path: `FLASH_ATTN_V100 decode path active (paged KV kernel, CUDA-graph safe)`
- FA2 no-prefix prefill: `FLASH_ATTN_V100 prefill path active (no prefix/chunked context)`
- FA2 prefix/chunked prefill: `FLASH_ATTN_V100 prefill path active (prefix/chunked via paged-KV gather)`
- Triton run used `AttentionBackendEnum.TRITON_ATTN`

Speed median:

| Case | FA2 | Triton | FA2 speedup |
|---|---:|---:|---:|
| batch1 prefill512 decode1 | 0.1693 s | 0.1488 s | 0.88x |
| batch1 prefill512 decode64 | 0.8299 s | 0.8253 s | 0.99x |
| batch1 prefill8192 decode1 | 0.6722 s | 1.2135 s | 1.81x |
| batch1 prefill8192 decode64 | 1.3484 s | 2.0812 s | 1.54x |
| batch4 mixed decode32 | 0.7321 s | 0.7057 s | 0.96x |
| batch4 decode-heavy128 | 1.9224 s | 1.9198 s | 1.00x |

Derived long-context decode increment:
- FA2: `(1.3484 - 0.6722) / 63 = 10.73 ms/token`
- Triton: `(2.0812 - 1.2135) / 63 = 13.77 ms/token`
- FA2 incremental long-context decode speedup: about `1.28x`

Model speed conclusion:
- Long prefill is clearly faster with FA2.
- Long-context decode is also faster, but end-to-end decode-heavy cases are mostly bottlenecked by non-attention work, so the full model throughput is near equal.
- Short prefill can be slightly slower with FA2 in this model-level setup.

### Model 256k Results

The 256k model run used Qwen3.5-35B-A3B-AWQ with TP4 on GPUs `1,2,3,4`, `max_model_len=262144`, `max_num_batched_tokens=8192`, and generated prompts of about 258k tokens after the Qwen3.5 chat template.

The first 256k attempt with `max_num_batched_tokens=32768` failed in the SM70 AWQ MoE workspace allocation, not in attention:
- failing allocation: `torch.empty(total_slots, ...)` in `awq_sm70_moe.py`
- error: `torch.OutOfMemoryError: Tried to allocate 1024.00 MiB`

Reducing the chunk size to `8192` allowed the 256k model run to complete for both backends.

Speed, one iteration per case:

| Case | Input tokens | FA2 | Triton | FA2 speedup |
|---|---:|---:|---:|---:|
| batch1 prefill258k decode1 | 258012 | 114.942 s | 685.590 s | 5.96x |
| batch1 prefill258k decode8 | 258012 | 115.695 s | 686.628 s | 5.93x |

Derived 256k decode increment:
- FA2: `(115.6946 - 114.9424) / 7 = 107.45 ms/token`
- Triton: `(686.6276 - 685.5902) / 7 = 148.20 ms/token`
- FA2 incremental 256k decode speedup: about `1.38x`

The `decode8 - decode1` estimate only uses a 7-token difference between two independent long-prefill requests, so it is sensitive to prefill jitter.

A follow-up FA2-only run used the same 258k prompt but compared `decode65` against `decode1`:

| Case | Input tokens | Latency | Output tokens |
|---|---:|---:|---:|
| batch1 prefill258k decode1 | 258012 | 114.610 s | 1 |
| batch1 prefill258k decode65 | 258012 | 115.713 s | 65 |

Derived from the 64-token difference:
- FA2 decode: `(115.7133 - 114.6103) / 64 = 17.23 ms/token`
- FA2 decode rate: about `58.0 tokens/s`

A stricter follow-up skipped quality generation and compared `decode257` against `decode1`, using a 256-token difference:

| Case | Input tokens | Latency | Output tokens |
|---|---:|---:|---:|
| batch1 prefill258k decode1 | 258012 | 115.948 s | 1 |
| batch1 prefill258k decode257 | 258012 | 119.904 s | 257 |

Derived from the 256-token difference:
- FA2 decode: `(119.9036 - 115.9485) / 256 = 15.45 ms/token`
- FA2 decode rate: about `64.7 tokens/s`

Model 256k speed conclusion: the real model confirms the operator result. At 256k context, FA2 is not worse than Triton; it is about `5.9x` faster end-to-end for long prefill. The earlier `107 ms/token` decode estimate from `decode8 - decode1` was too noisy. The better `decode65 - decode1` and `decode257 - decode1` estimates are `58.0` to `64.7 tokens/s` for single-request 256k decode on TP4 V100.

## Quality Results

Operator correctness passed.

Model strict greedy-token quality did not fully pass:
- Passed exact token/text: `short_zh_reasoning`, `short_en_reasoning`
- Failed exact token/text: `code`, `medium_mixed`, `long_prefill`
- No non-finite generated logprobs were observed.

Failure details:
- `code`: first token mismatch at generated token index 18.
- `medium_mixed`: first generated token differs. FA2 selected `基于`; Triton selected `您`.
- `long_prefill`: first mismatch at generated token index 12.

256k strict quality result:
- Case: `long_prefill`
- Prompt tokens: FA2 `258011`, Triton `258011`
- FA2 output token IDs: `[16, 25944, 343, 3950]`, text prefix `1Cat vLL`
- Triton output token IDs: `[760, 1879, 3766, 369]`, text prefix `The input provided is`
- Generated logprobs were finite; generated non-finite count was `0`.
- The first generated token differed. FA2 top-1 was token `16` (`1`) with logprob `-1.3355`; Triton top-1 was token `760` (`The`) with logprob `-1.0765`.

Interpretation:
- This is not an operator-level correctness failure: FA2 and Triton attention outputs match tightly on long-context Qwen3.5 shapes, including block size 528.
- It is also not a NaN/inf quality failure.
- It is a strict model-level deterministic regression failure: small backend numerical differences can change greedy generation on ambiguous prompts.
- Therefore, current FA2 cannot be called fully token-exact equivalent to Triton at model level.

## Blockers Found

1. `prompt_logprobs=5` with an 8k prompt OOMs on TP4 V100 due logits all-gather, needing about 3.79 GiB extra allocation per failing rank.
2. TP4 V100 with vLLM custom all-reduce failed during CUDA graph capture with `custom_all_reduce.cuh:455 invalid argument`. The valid model run used `disable_custom_all_reduce=True`.
3. The first 256k model run with `max_num_batched_tokens=32768` OOMed in SM70 AWQ MoE buffer allocation. Reducing to `8192` completed the 256k run.

## Conclusion

For long-context speed, current FA2 is better than Triton on both operator-level and real Qwen3.5 model-level tests, including 256k context.

For quality, the attention kernels themselves pass strict numerical checks, but the full model does not pass strict exact greedy-token regression against Triton on all prompts. Treat model-level exact-output parity as unresolved until either the numerical drift is reduced or the acceptance criterion is changed from exact token equality to a top-k/logprob tolerance policy.
