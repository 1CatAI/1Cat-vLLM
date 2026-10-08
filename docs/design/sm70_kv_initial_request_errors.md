# Initial real-request KV arithmetic comparison

This is an initial data gate, not a runtime default or kernel admission.

Runtime: normal source wheel at `477e79507b09498b011ebcfa625b8b67aad321e7`,
Torch 2.10.0+cu128 / CUDA 12.8, two NVLink V100-SXM2-32GB GPUs on the
authorized 54633 host. FP16 KV, eager execution, no MTP or prefix reuse.
One real user request (157 chat tokens) reaches three dense attention layers
on both ranks. Per sample: Q [16,12,256], K/V [157,2,256], explicit causal mask.
Values were captured after RoPE and before quantization. Diagnostic copying
and synchronization exclude this run from timing evidence.

E4M3 uses the captured **FP16-cache layer scalars**; independent production
E4M3 calibration remains unchecked. The oracle is FP32 masked attention over
the captured unquantized FP16 tensors. Mean RMSE averages six layer/rank samples
per model; maximum attention error is the largest element error over those six.
Payload bytes include scale/group padding; E4M3 adds 8 fixed layer-scale bytes.

Raw per-sample metrics and hashes are retained in
[the JSON ledger](sm70_kv_initial_request_errors.json). Request tensors stay on
the task data volume. No model weights or private cache paths are committed.

## 27B NVFP4

| Scheme | Mean K RMSE | Mean V RMSE | Mean attention RMSE | Max attention error | Bytes/sample |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fp16` | 0 | 0 | 0 | 0 | 321536 |
| `e4m3_layer_scale` | 0.0378525 | 0.0636015 | 0.0370931 | 1.19784 | 160768 |
| `int8_token_head_fp32` | 0.0146841 | 0.0249883 | 0.0160022 | 0.668939 | 163280 |
| `int8_token_head_fp16` | 0.0146843 | 0.0249877 | 0.0159966 | 0.681967 | 162024 |
| `int8_token_head_legacy_trunc_fp32` | 0.0291899 | 0.0493601 | 0.0437854 | 1.67092 | 163280 |
| `int8_token_head_affine_fp32` | 0.0126148 | 0.0210328 | 0.0129432 | 0.461189 | 165792 |
| `int8_feature_group32_fp16` | 0.00933378 | 0.0161027 | 0.00926477 | 0.40541 | 170816 |
| `int8_k_channel_token_group32_v_token_fp32` | 0.00653091 | 0.0249883 | 0.0144131 | 0.174549 | 173800 |
| `int8_feature_group64_fp16` | 0.0109336 | 0.0188504 | 0.0112148 | 0.462101 | 165792 |
| `int8_k_channel_token_group64_v_token_fp32` | 0.00708224 | 0.0249883 | 0.0144351 | 0.149326 | 186088 |

## 27B GGUF

| Scheme | Mean K RMSE | Mean V RMSE | Mean attention RMSE | Max attention error | Bytes/sample |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fp16` | 0 | 0 | 0 | 0 | 321536 |
| `e4m3_layer_scale` | 0.0379345 | 0.0628093 | 0.0422691 | 2.17532 | 160768 |
| `int8_token_head_fp32` | 0.0147296 | 0.024606 | 0.0158804 | 0.614545 | 163280 |
| `int8_token_head_fp16` | 0.0147299 | 0.0246057 | 0.0159269 | 0.615461 | 162024 |
| `int8_token_head_legacy_trunc_fp32` | 0.0292973 | 0.0486421 | 0.0434619 | 1.71436 | 163280 |
| `int8_token_head_affine_fp32` | 0.012637 | 0.0208319 | 0.0123942 | 0.41814 | 165792 |
| `int8_feature_group32_fp16` | 0.0093523 | 0.0159104 | 0.00901186 | 0.208286 | 170816 |
| `int8_k_channel_token_group32_v_token_fp32` | 0.00650833 | 0.024606 | 0.014172 | 0.198286 | 173800 |
| `int8_feature_group64_fp16` | 0.0109685 | 0.0185907 | 0.0107758 | 0.287826 | 165792 |
| `int8_k_channel_token_group64_v_token_fp32` | 0.00706017 | 0.024606 | 0.0142179 | 0.205702 | 186088 |

## Decision implications

The current truncating INT8 writer cannot stand in for nearest-even INT8 in
format selection: its numerical error is materially different. Extraction keeps
that existing rounding contract; a new rounding policy needs separate writer
and quality gates. Feature groups improve this initial attention error at the
cost of metadata and a more complex reader. Channel groups across tokens also
need a partial-group/residual policy for incremental writes.

Before choosing an INT8 codec, collect Flash-Next with its actual QSA/indexer
mask and compression state, expand the real prompts and context lengths,
recheck production E4M3 scales, and measure native-kernel arithmetic and cost.
All format/path and model quality/performance gates remain required.

The later [K/V error ablation](sm70_kv_error_ablation.md) evaluates which side
drives attention error and adds two mixed K/V granularity candidates. This
initial ten-scheme ledger remains unchanged; the new measurements have their
own CPU evaluation provenance.
