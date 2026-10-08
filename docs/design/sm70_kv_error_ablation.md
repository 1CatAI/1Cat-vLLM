# Real-request K/V error ablation

Same 12 captured layer/rank samples as the [initial comparison](sm70_kv_initial_request_errors.md): one 157-token request, two 27B models, FP16 KV, TP2, final 16 queries. This CPU analysis adds no GPU reservation and does not select a runtime format or default.

K-only quantizes K while retaining FP16 V; V-only retains FP16 K and quantizes V. Both use the same actual request mask and FP32 attention oracle. These errors are not additive because K changes the softmax. Mean RMSE averages six layer/rank samples per model. Captured FP16-layer E4M3 scales remain uncalibrated.

## 27B NVFP4

| Scheme | Both RMSE | K-only RMSE | V-only RMSE | Candidate bytes/sample |
| --- | ---: | ---: | ---: | ---: |
| `e4m3_layer_scale` | 0.0370931 | 0.0217498 | 0.0278949 | 160768 |
| `int8_token_head_fp32` | 0.0160022 | 0.00709227 | 0.0140705 | 163280 |
| `int8_feature_group32_fp16` | 0.00926477 | 0.00416037 | 0.00803542 | 170816 |
| `int8_k_channel_token_group32_v_token_fp32` | 0.0144131 | 0.00288285 | 0.0140705 | 173800 |
| `int8_k_token_fp32_v_feature_group32_fp16` | 0.0111704 | 0.00709227 | 0.00803542 | 167048 |
| `int8_k_token_fp32_v_feature_group64_fp16` | 0.0124174 | 0.00709227 | 0.00970747 | 164536 |

## 27B GGUF

| Scheme | Both RMSE | K-only RMSE | V-only RMSE | Candidate bytes/sample |
| --- | ---: | ---: | ---: | ---: |
| `e4m3_layer_scale` | 0.0422691 | 0.0259277 | 0.027898 | 160768 |
| `int8_token_head_fp32` | 0.0158804 | 0.00703687 | 0.0138042 | 163280 |
| `int8_feature_group32_fp16` | 0.00901186 | 0.00377374 | 0.00795867 | 170816 |
| `int8_k_channel_token_group32_v_token_fp32` | 0.014172 | 0.00295461 | 0.0138042 | 173800 |
| `int8_k_token_fp32_v_feature_group32_fp16` | 0.0112854 | 0.00703687 | 0.00795867 | 167048 |
| `int8_k_token_fp32_v_feature_group64_fp16` | 0.0124683 | 0.00703687 | 0.00955049 | 164536 |

## Decision implications

In this request, nearest-even token/head INT8 has roughly twice as much V-only as K-only attention RMSE. Better channel-grouped K alone leaves the larger V error. This motivates two additional arithmetic candidates: FP32 token/head K scales with FP16 feature-group V scales at widths 32 and 64. They are offline candidates, not implemented runtime codecs.

The V-group32 candidate reduces combined attention RMSE by about 30% relative to token/head FP32 in both models, with 2.31% more candidate bytes. It still has greater error than grouping both K and V by 32. V-group64 uses 0.77% more candidate bytes than token/head FP32 and gives a smaller error improvement.

The byte figures include candidate payload, scale and feature/token-group padding. They do not prove a physical page layout: different K/V metadata widths need distinct offsets/strides or shared padding that can erase these savings. Page alignment, Mamba geometry, partial channel-group residual state, prefix restore and host offload must be included in the implemented codec budget before selecting a scheme. E4M3 also has 8 fixed layer-scale bytes.

The original ten-scheme metrics match exactly when the old and new tools run on the same local CPU/Torch/thread contract. Cross-host FP32 attention rounding slightly differs from the older ledger; this document retains its own evaluation provenance rather than silently replacing old values. All twelve schemes and per-sample hashes/metrics are in [the JSON ledger](sm70_kv_error_ablation.json).

Next decisions require Flash-Next/QSA and longer real requests, independently calibrated E4M3, native reader staging/throughput and the model quality gates. Short-request RMSE cannot admit a default or establish decode performance.

## Decoded FP16 tile check

`--simulate-fp16-staging` casts reconstructed K/V to FP16 before the FP32 masked oracle. This checks one proposed reader staging contract; it does not model native QK scale epilogues, FP16 probabilities or tensor-core/split-K accumulation order. All12 real samples stay finite across the12 schemes. The [staging ledger](sm70_kv_fp16_staging.json) retains per-layer hashes, errors and counts.

| Candidate | NVFP4 staged RMSE | GGUF staged RMSE |
| --- | ---: | ---: |
| `int8_token_head_fp32` | 0.0159953 | 0.0158901 |
| `int8_token_head_fp16` | 0.0159958 | 0.0158754 |
| `int8_k_token_fp32_v_feature_group32_fp16` | 0.0111747 | 0.0112875 |

A finite-extreme regression exposes an encoder/reader contract that these short requests do not exercise:65504/127 rounds to stored FP16 scale516, and decoded127×516=65532 becomes infinity in a FP16 tile. The FP32-scale candidate stays finite. A FP16-scale codec needs an explicit scale-rounding or saturating-decoder rule before native admission; neither change is implemented or silently applied to the existing arithmetic candidates. Invalid simulated tiles report nonfinite counts without evaluating infinite attention or emitting NaN JSON. The regression is synthetic contract coverage, not format-selection data.
