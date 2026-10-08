# Initial Flash-Next QSA request errors

The ordinary source120 wheel captured12 real layer/rank samples from Flash-Next IQ3_XXS on four V100-SXM2-32GB GPUs with NVLink. The request contains157 verified chat token IDs. Diagnostic execution uses FP16 KV, eager mode, no MTP/prefix reuse, one greedy output token, and skips tokenizer initialization. This is not text-quality, EOS, acceptance or serving-speed evidence.

Each sample retains16 final queries `[16,6,256]` and157 K/V rows `[157,1,256]`, actual indexer selections, addressed raw/compressed state, token/sample hashes and19 installed runtime file hashes. Every rank executes12 grouped-page4 and12 XQA-page4 calls in the real request; each captured layer records one of each. Warmup is excluded. All own GPU/PLE workers exited normally and the bounded TP4 lease released.

The selected layers are3/27/47. All have compression ratio4; the addressed states have4 raw and39 compressed rows. Every final query selects all its causal history (142 through157 keys). Thus the real QSA route/mask is exercised, but this short request does not exercise top-k pruning or the ratio128 layer family. Future capture selection explicitly includes each distinct compression ratio while retaining first/middle/last depth representatives.

## Arithmetic comparison

These are means over12 layer/rank samples with an FP32 masked oracle over captured FP16 Q/K/V. TP rank samples share one request and some replicated KV; they are not12 independent requests. K-only retains FP16 V; V-only retains FP16 K. Errors are not additive. Captured FP16 layer scalars remain unchecked E4M3 calibration.

| Scheme | Both RMSE | K-only RMSE | V-only RMSE | FP16 staged RMSE | Candidate bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fp16` | 0 | 0 | 0 | 0 | 160768 |
| `e4m3_layer_scale` | 0.01095393 | 0.005946365 | 0.009318224 | 0.01095393 | 80384 |
| `int8_token_head_fp32` | 0.004157149 | 0.001644019 | 0.003790535 | 0.004157349 | 81640 |
| `int8_token_head_fp16` | 0.004156754 | 0.001645957 | 0.003792292 | 0.004158157 | 81012 |
| `int8_token_head_legacy_trunc_fp32` | 0.01143756 | 0.004500896 | 0.009858052 | 0.01143468 | 81640 |
| `int8_token_head_affine_fp32` | 0.003258087 | 0.001359624 | 0.002952846 | 0.003260305 | 82896 |
| `int8_feature_group32_fp16` | 0.002482061 | 0.001137268 | 0.002188429 | 0.002484555 | 85408 |
| `int8_k_channel_token_group32_v_token_fp32` | 0.003897756 | 0.0008622834 | 0.003790535 | 0.003896276 | 86900 |
| `int8_feature_group64_fp16` | 0.002915484 | 0.001313318 | 0.002585939 | 0.002917687 | 82896 |
| `int8_k_channel_token_group64_v_token_fp32` | 0.003923504 | 0.0009566682 | 0.003790535 | 0.003922708 | 93044 |
| `int8_k_token_fp32_v_feature_group32_fp16` | 0.002757444 | 0.001644019 | 0.002188429 | 0.002758811 | 83524 |
| `int8_k_token_fp32_v_feature_group64_fp16` | 0.003086059 | 0.001644019 | 0.002585939 | 0.003089316 | 82268 |

Nearest-even token/head INT8 has mean attention RMSE0.00415715, while legacy truncation gives0.0114376. V-only error is greater than K-only, consistent with the two27B short-request corpora. Mixed token/head K plus group32 V gives0.00275744, about33.7% lower than token/head FP32 for2.31% more candidate bytes. Grouping both K and V by32 gives still lower error, at greater metadata cost. None of these measurements selects a runtime/default format.

All144 candidate/sample FP16 staging simulations remain finite. This simulation does not reproduce native QK epilogues, probabilities or tensor-core/split-K arithmetic. Candidate bytes include inline scale/group padding, but physical page stride/alignment, residual groups, Mamba allocation and offload remain unimplemented. The prior finite-extreme FP16-scale overflow regression still applies.

## Reproduction and remaining gates

Run `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python tools/kv_codec/evaluate.py --manifest DATA/manifest.json --out DATA/errors.json --ablate-kv --simulate-fp16-staging`. The [JSON ledger](sm70_kv_flashnext_request_errors.json) contains all144 measurements, evaluator hash, actual routes, sample provenance and diagnostic configuration. Raw tensors/logs remain outside Git.

Engine initialization takes803.17s and the hooked request1.35s; both are diagnostic durations, not performance results. The previous15-minute cold attempt expired without samples. The successful token-ID/batch157 path completes within the same15-minute cap; no long-lived service remains.

Longer real requests, ratio128 and actual sparse-pruning cases, calibrated E4M3, native codec cost/rounding and the three-model KL/top1/acceptance/retrieval/EOS/C4 gates remain necessary. Keep FP8 optional and do not set an INT8 default from these short-request errors.
