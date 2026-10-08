# SM70 KV source inventory

Reproduce using `tools/kv_codec/inventory.py --out <artifact.json>`. Full broad-match evidence stays outside Git; this document retains every tracked source file explicitly mentioning `kv_cache_dtype`. Broad fp8/e4m3/uint8 matches include weight quantization and are not all KV decisions.

Base `c4f6245f841466782752a8c3283e4727565cf17a`: 1361 broad-match source files, 210 explicit KV dtype files, including 73 package, 15 native, 76 tests, 39 benchmarks and 4 vendored Flash-V100 files. Backend: 134 exact `kv_cache_dtype` tokens, 26 `if`/conditional-expression/assert predicates mentioning cache dtype, 89 environment read names, 44 literal route names, 52 route call sites. The reported 160 branches / 78 switches do not match this reproducible counting definition.

| File | Explicit dtype line numbers at baseline |
|---|---|
| `.buildkite/lm-eval-harness/test_lm_eval_correctness.py` | 56, 61 |
| `benchmarks/attention_benchmarks/benchmark.py` | 592, 593, 656, 714, 868, 892, 924 |
| `benchmarks/attention_benchmarks/common.py` | 217, 375, 389 |
| `benchmarks/attention_benchmarks/mla_runner.py` | 68, 155, 524, 573, 706, 740, 741, 743, 755, 796, 933, 934, 938, 939, 956, 970, 1027 |
| `benchmarks/attention_benchmarks/runner.py` | 143, 218, 359, 545 |
| `benchmarks/benchmark_flash_v100_9b_awq_e2e.py` | 215, 400 |
| `benchmarks/benchmark_flash_v100_bfla_longbench.py` | 338 |
| `benchmarks/benchmark_flashnext_acceptance.py` | 115 |
| `benchmarks/benchmark_gguf_model.py` | 105 |
| `benchmarks/benchmark_qwen38_dcp_numerics.py` | 73, 102 |
| `benchmarks/benchmark_qwen38_dcp_quality.py` | 296 |
| `benchmarks/benchmark_sm70_79t_cold.py` | 76, 256 |
| `benchmarks/benchmark_sm70_attention_exactness.py` | 3309, 3356, 3516, 3527, 3693, 3716, 4462, 4557, 4588, 4620, 6062, 6082, 6207, 6216 |
| `benchmarks/benchmark_sm70_decode.py` | 241, 658, 749, 750, 752, 753, 755, 861, 862, 863, 871, 1739, 1970 |
| `benchmarks/benchmark_sm70_dflash2_batched_grouped_verify.py` | 119 |
| `benchmarks/benchmark_sm70_dflash2_fp32_attention.py` | 44 |
| `benchmarks/benchmark_sm70_dflash2_gsm8k.py` | 395, 406 |
| `benchmarks/benchmark_sm70_mixed_decode_rows.py` | 42 |
| `benchmarks/benchmark_sm70_model_tokens.py` | 478, 959, 1049, 1050, 1052, 1053, 1055, 1151, 1152, 1153, 1161, 1743, 1908 |
| `benchmarks/benchmark_sm70_mtp4_round.py` | 150 |
| `benchmarks/benchmark_sm70_qwen38_distribution.py` | 111 |
| `benchmarks/benchmark_sm70_qwen38_quality.py` | 308 |
| `benchmarks/benchmark_sm70_serving_quality.py` | 587 |
| `benchmarks/benchmark_sm70_source_inventory.py` | 81 |
| `benchmarks/benchmark_sm70_turboquant_quality.py` | 255, 259, 324, 333, 358, 550, 551, 609, 616, 633, 637, 703 |
| `benchmarks/compare_flashnext_acceptance.py` | 19 |
| `benchmarks/kernels/benchmark_flash_v100_fp8_kv.py` | 206, 237, 262, 275 |
| `benchmarks/kernels/benchmark_paged_attention.py` | 40, 77, 128, 149, 170, 250 |
| `benchmarks/kernels/benchmark_reshape_and_cache.py` | 29, 36, 60, 78, 123 |
| `benchmarks/kernels/benchmark_reshape_and_cache_flash.py` | 32, 41, 73, 93, 104, 149 |
| `benchmarks/kernels/benchmark_sm70_dflash2_grouped_verify.py` | 115 |
| `benchmarks/kernels/benchmark_sm70_flash_v100_xqa_layout.py` | 36, 89, 114, 140, 167, 200, 394, 457, 484, 506, 542, 585, 611 |
| `benchmarks/kernels/benchmark_sm70_multihead_attention.py` | 68 |
| `benchmarks/kernels/benchmark_trtllm_decode_attention.py` | 204, 227, 272 |
| `benchmarks/kernels/benchmark_trtllm_prefill_attention.py` | 220, 243, 287 |
| `benchmarks/probe_sm70_prefix_cache.py` | 77, 78, 81, 125 |
| `benchmarks/run_sm70_quality_speed_matrix.py` | 428, 444, 445, 585, 700, 800, 801, 821 |
| `benchmarks/run_sm70_release_matrix.py` | 286, 297, 310, 326, 329, 336, 347, 350, 353, 365, 371, 380, 388, 394, 403, 411, 417, 448, 461, 1078, 1107, 1124, 1125, 1972, 1983, 2048, 2082, 2219, 2435 |
| `benchmarks/sm70_qwen38_baseline.py` | 215 |
| `benchmarks/verify_sm70_artifact.py` | 832 |
| `csrc/attention/sm70_grouped_long/include/fused_mha.h` | 27, 38, 49, 57, 78, 92, 99, 113, 138, 146 |
| `csrc/attention/sm70_grouped_long/kernel/grouped-attention.cu` | 25, 26, 27, 30, 33 |
| `csrc/attention/sm70_grouped_long/kernel/scalar-attention.cu` | 25, 26, 27, 30, 33 |
| `csrc/cache.h` | 21, 28, 33, 41, 45, 53 |
| `csrc/cpu/cpu_attn.cpp` | 3, 6, 7, 9, 92, 101, 183, 190 |
| `csrc/cpu/torch_bindings.cpp` | 164, 176, 502, 511 |
| `csrc/libtorch_stable/attention/paged_attention_v1.cu` | 177, 184 |
| `csrc/libtorch_stable/attention/paged_attention_v2.cu` | 190, 196 |
| `csrc/libtorch_stable/cache_kernels.cu` | 773, 791, 819, 840, 890, 925, 941, 964, 972, 977, 1007, 1023, 1040, 1059, 1079, 1202, 1253, 1256, 1584, 1585 |
| `csrc/libtorch_stable/cache_kernels_fused.cu` | 227, 305 |
| `csrc/libtorch_stable/ops.h` | 377, 391, 412, 419, 426, 435, 441, 450 |
| `csrc/libtorch_stable/torch_bindings.cpp` | 498, 511, 670, 679, 687, 701, 707, 716, 731 |
| `csrc/rocm/attention.cu` | 3656, 3663, 3674, 3699 |
| `csrc/rocm/ops.h` | 37 |
| `csrc/rocm/torch_bindings.cpp` | 69 |
| `flash-attention-v100/flash_attn_v100/flash_attn_interface.py` | 1011, 1025, 1108, 1224, 1254, 1275, 1315, 1325, 1335, 1356, 1396, 1433, 1450, 1474, 1486, 1510, 1573, 1598, 1687, 1773, 1805, 1823, 1854, 1874, 1900 |
| `flash-attention-v100/include/fused_mha.h` | 27, 38, 49, 57, 81, 95, 102, 116, 141, 149 |
| `flash-attention-v100/kernel/flash_decode_paged.cu` | 26, 27, 28, 31, 34, 4598, 4607, 4616, 4618, 4620, 4799, 4845, 4853, 4860, 4862, 4956, 4970, 4972, 5111, 5170, 5183, 6154, 6172, 6241, 6247, 6249, 6314 |
| `flash-attention-v100/kernel/fused_mha_forward_paged.cu` | 39, 40, 43, 46, 3396, 3401, 3543, 3548, 3550, 3640, 3921, 3925, 3992, 3995, 3997, 4070 |
| `tests/basic_correctness/test_cumem.py` | 255 |
| `tests/benchmarks/test_sm70_qwen38_baseline.py` | 47, 137 |
| `tests/benchmarks/test_sm70_release_quality_gate.py` | 103, 108 |
| `tests/compile/fullgraph/test_full_graph.py` | 212 |
| `tests/compile/fusions_e2e/models.py` | 16, 64 |
| `tests/compile/passes/test_fusion_attn.py` | 90 |
| `tests/compile/passes/test_mla_attn_quant_fusion.py` | 67, 81, 127, 169, 483, 511 |
| `tests/compile/passes/test_mla_rope_kvcache_cat_fusion.py` | 144, 145, 146, 184, 221, 255, 272, 286 |
| `tests/compile/passes/test_rope_kvcache_fusion.py` | 98, 99, 100, 139, 204, 219, 234 |
| `tests/config/test_prefix_anchored_swa.py` | 144 |
| `tests/config/test_sm70_release_profile.py` | 45, 62 |
| `tests/entrypoints/llm/test_accuracy.py` | 88 |
| `tests/kernels/attention/test_attention.py` | 123, 135, 139, 148, 191, 216, 235, 279, 301, 330, 353, 364, 403 |
| `tests/kernels/attention/test_attention_selector.py` | 417, 426, 472, 480, 514, 520, 530, 535, 538, 540, 548 |
| `tests/kernels/attention/test_cache.py` | 56, 68, 70, 90, 102, 120, 132, 137, 155, 175, 190, 205, 242, 257, 263, 278, 297, 299, 302, 304, 310, 319, 331, 346, 351, 393, 395, 401, 419, 440, 453, 455, 457, 485, 498, 625, 661, 677, 689, 766, 769, 775, 776, 779, 781, 794, 805, 821, 832, 834, 840, 844, 846, 849, 853, 885, 904, 950, 954, 986, 996, 1005, 1008, 1011, 1012, 1050, 1061, 1067, 1069, 1105, 1113, 1133, 1147, 1162, 1174, 1179, 1181, 1250, 1264, 1275, 1277, 1283, 1287 |
| `tests/kernels/attention/test_cpu_attn.py` | 203, 227, 278, 390, 418, 442, 456, 460, 485, 499, 585, 613, 627, 631, 655, 669, 673, 697, 711, 715, 739, 753 |
| `tests/kernels/attention/test_flashinfer.py` | 516, 531, 573, 624, 638, 639, 680 |
| `tests/kernels/attention/test_prefix_prefill.py` | 105, 116, 121, 129, 164, 167, 232, 251, 320, 328, 337, 342, 350, 407, 410, 475, 494, 580, 596, 607, 617, 628, 637, 642, 671 |
| `tests/kernels/attention/test_sm70_decode_workspace_arena.py` | 150 |
| `tests/kernels/attention/test_sm70_dflash_batched_prefill.py` | 37, 108 |
| `tests/kernels/attention/test_sm70_e4m3_scalar_fp32.py` | 16, 68 |
| `tests/kernels/attention/test_sm70_flash_v100_decode_planner.py` | 246, 284, 293, 302, 331 |
| `tests/kernels/attention/test_sm70_flash_v100_grouped_verify.py` | 290 |
| `tests/kernels/attention/test_sm70_flash_v100_mtp_smallq_exactness.py` | 78, 101 |
| `tests/kernels/attention/test_sm70_flash_v100_multihead.py` | 105, 133, 164, 227 |
| `tests/kernels/attention/test_sm70_flash_v100_prefix_decode_rows.py` | 28, 38, 46, 65, 67, 110, 124, 135, 141, 169, 216 |
| `tests/kernels/attention/test_sm70_flash_v100_varlen_layout.py` | 68, 110, 111, 114, 124, 135, 479, 592, 661, 743, 824, 926 |
| `tests/kernels/attention/test_sm70_grouped_e4m3.py` | 36, 81 |
| `tests/kernels/attention/test_sm70_qsa_grouped_page4.py` | 32, 34, 37, 45, 76 |
| `tests/kernels/attention/test_sm70_tp2_e4m3_scalar_fast.py` | 43 |
| `tests/kernels/attention/test_use_trtllm_attention.py` | 35, 176, 181, 186, 191 |
| `tests/kernels/core/test_rotary_embedding_mla_cache_fused.py` | 25, 42, 104, 116, 119, 135, 150, 154, 160, 164 |
| `tests/kernels/test_cache_kernels.py` | 55 |
| `tests/kernels/test_sm70_qsa_page4_plan.py` | 345 |
| `tests/models/glm5next/test_sm70_sparse.py` | 54, 85, 100, 115, 146 |
| `tests/models/multimodal/generation/test_maverick.py` | 54 |
| `tests/models/quantization/test_fp8.py` | 22, 54, 68, 69, 79, 93, 105, 123, 138, 158, 168 |
| `tests/models/quantization/test_mxfp4.py` | 33 |
| `tests/models/quantization/test_per_token_kv_cache.py` | 34, 45, 69, 81, 93 |
| `tests/models/qwen4_exp/test_e4m3_mtp.py` | 129 |
| `tests/models/qwen4_exp/test_e4m3_mtp_capture_gpu.py` | 58 |
| `tests/models/qwen4_exp/test_e4m3_mtp_gpu.py` | 69, 131, 184 |
| `tests/models/qwen4_exp/test_qsa_cache.py` | 47, 166, 178, 190, 214 |
| `tests/models/qwen4_exp/test_qsa_dcp_attention.py` | 71, 108 |
| `tests/models/qwen4_exp/test_qsa_dcp_local_width.py` | 133, 152 |
| `tests/models/qwen4_exp/test_qsa_dcp_packed_cache.py` | 119 |
| `tests/models/qwen4_exp/test_qsa_e4m3.py` | 120, 163, 237, 296, 397 |
| `tests/models/qwen4_exp/test_qsa_metadata_padding.py` | 556 |
| `tests/models/qwen4_exp/test_qsa_ops.py` | 290, 296, 395 |
| `tests/models/qwen4_exp/test_qsa_xqa_page4_workspace.py` | 30, 33, 35 |
| `tests/models/qwen4_exp/test_weight_loading.py` | 339, 365, 369, 401, 422 |
| `tests/plugins/vllm_add_dummy_platform/vllm_add_dummy_platform/dummy_platform.py` | 28 |
| `tests/quantization/test_fp8.py` | 105, 140, 149, 167, 173, 483, 528, 562, 583, 584, 589, 590, 597 |
| `tests/quantization/test_per_token_kv_cache.py` | 60, 70, 78, 377, 379, 387 |
| `tests/quantization/test_quark.py` | 61, 63, 68 |
| `tests/quantization/test_sm70_quantized_model_compat.py` | 46, 71, 86 |
| `tests/quantization/test_turboquant.py` | 220 |
| `tests/v1/attention/test_attention_backends.py` | 294 |
| `tests/v1/attention/test_flash_attn_turing_gates.py` | 27 |
| `tests/v1/attention/test_mla_backends.py` | 160, 177, 195, 196, 243, 356, 357, 366, 370, 494, 495, 502, 506, 590, 623, 718, 728, 754, 758, 766, 822, 1108, 1134, 1156 |
| `tests/v1/attention/test_prefix_anchored_swa_mask.py` | 64 |
| `tests/v1/attention/test_rocm_attention_backends_selection.py` | 142, 269, 278, 293, 322, 346 |
| `tests/v1/attention/test_sm70_e4m3_grouped.py` | 38, 57 |
| `tests/v1/attention/test_sm70_e4m3_scalar_tail.py` | 53, 71 |
| `tests/v1/attention/test_sm70_flash_v100_policy.py` | 351, 355, 359, 363, 386, 408, 498, 525, 528, 781, 1465, 2022, 2092, 2195, 2293, 2393, 2470, 2531, 2588, 2597, 2604, 2664, 2731, 2780, 2799, 2978, 3063, 3125, 3187, 3243, 3408, 3463, 3523, 3592, 3670 |
| `tests/v1/attention/test_sm70_flashinfer_backend.py` | 86 |
| `tests/v1/attention/test_sm70_fp16_grouped_admission.py` | 28, 76 |
| `tests/v1/attention/test_sm70_mixed_decode_rows.py` | 50, 52, 180, 194, 220, 259 |
| `tests/v1/attention/test_sm70_v37_prefill.py` | 98, 125 |
| `tests/v1/attention/test_sparse_mla_backends.py` | 151, 181, 190, 197, 198, 202, 203, 225, 418, 438, 472, 523 |
| `tests/v1/attention/test_trtllm_attention_integration.py` | 282, 295, 362, 398, 449, 534 |
| `tests/v1/spec_decode/test_acceptance_length.py` | 120 |
| `tests/v1/spec_decode/test_dflash2.py` | 728, 757 |
| `tests/v1/spec_decode/test_dflash_mrv2_config.py` | 334, 356 |
| `tests/v1/spec_decode/test_draft_profile_scope.py` | 41, 91 |
| `tests/v1/worker/test_gpu_model_runner.py` | 85, 1658 |
| `tools/pre_commit/generate_attention_backend_docs.py` | 550, 551, 553, 557, 558, 559, 794, 797, 799, 806, 1076, 1079, 1126, 1128, 1300 |
| `tools/qwen4_exp/qsa_kv_calibration.py` | 300, 351 |
| `vllm/_aiter_ops.py` | 2480, 2510 |
| `vllm/_custom_ops.py` | 128, 149, 175, 199, 226, 248, 2675, 2685, 2697, 2707, 2718, 2722, 2735, 2747, 2819, 2830, 2892, 2895, 4002, 4013, 4034, 4054 |
| `vllm/compilation/passes/fusion/mla_rope_kvcache_cat_fusion.py` | 33, 49, 62, 88, 155, 174, 203, 229 |
| `vllm/config/cache.py` | 130, 132 |
| `vllm/config/speculative.py` | 175 |
| `vllm/config/vllm.py` | 3373 |
| `vllm/distributed/kv_transfer/kv_connector/utils.py` | 354 |
| `vllm/engine/arg_utils.py` | 120, 141, 468, 710, 711, 1192, 1994, 1995, 1996, 1998, 1999, 2019, 2030, 2039, 2040 |
| `vllm/model_executor/layers/attention/attention.py` | 32, 169, 264, 267, 272, 274, 282, 285, 298, 305, 310, 313, 316, 318, 322, 323, 325, 353, 415, 462, 504, 506, 514, 556, 662, 683, 715, 722 |
| `vllm/model_executor/layers/attention/chunked_local_attention.py` | 98, 100, 102, 127 |
| `vllm/model_executor/layers/attention/cross_attention.py` | 207, 209, 219, 240 |
| `vllm/model_executor/layers/attention/encoder_only_attention.py` | 68, 70, 75 |
| `vllm/model_executor/layers/attention/mla_attention.py` | 251, 371, 374, 389, 399, 400, 404, 413, 421, 448, 570, 589, 661, 674, 965, 966, 971, 978, 1012, 1027, 1038, 1958, 1981, 2074 |
| `vllm/model_executor/layers/attention/sm70_qwen38_qk_rope.py` | 223 |
| `vllm/model_executor/layers/attention/static_sink_attention.py` | 134, 136, 141, 208, 228 |
| `vllm/model_executor/layers/quantization/kv_cache.py` | 76, 92, 138, 177 |
| `vllm/model_executor/models/extract_hidden_states.py` | 27, 101, 194, 200, 205, 256, 259, 262, 264, 266, 267, 279 |
| `vllm/model_executor/models/whisper_causal.py` | 294, 296, 301 |
| `vllm/models/deepseek_v4/attention.py` | 802, 804, 806, 815, 816, 820, 823, 848 |
| `vllm/models/glm5next/sm70/sparse.py` | 49, 90, 97, 120, 134, 150, 151, 205, 214 |
| `vllm/models/qwen4_exp/amd/qsa.py` | 35, 70, 113, 268, 269, 270, 286, 325 |
| `vllm/models/qwen4_exp/common/qsa_cache.py` | 813 |
| `vllm/models/qwen4_exp/nvidia/model.py` | 563, 1158, 1375 |
| `vllm/models/qwen4_exp/nvidia/mtp.py` | 312, 719 |
| `vllm/models/qwen4_exp/nvidia/ops/qsa.py` | 138, 146, 1817, 1821, 1957, 1976, 1984, 1998, 2044, 2066, 2084, 2088, 2105, 2130, 2150, 2167, 2177, 2180, 2192, 2199, 2224, 2241, 2248, 2258, 2284, 2313, 2322, 2323, 2393 |
| `vllm/models/qwen4_exp/nvidia/qsa.py` | 40, 184, 231, 284, 323, 392, 567, 568, 575, 576, 578, 587, 606, 679, 725 |
| `vllm/platforms/cuda.py` | 88, 96, 318 |
| `vllm/platforms/interface.py` | 584, 586, 596, 619, 623, 641 |
| `vllm/platforms/rocm.py` | 323, 351 |
| `vllm/platforms/xpu.py` | 65, 66 |
| `vllm/sm70_profiles/acceleration.py` | 685, 723 |
| `vllm/utils/flashinfer.py` | 378, 439, 444 |
| `vllm/utils/torch_utils.py` | 124, 126, 127, 128, 132, 133, 134, 424, 425, 427, 430, 431, 445, 446, 448, 451 |
| `vllm/v1/attention/backend.py` | 60, 168, 169, 171, 172, 266, 280, 298, 299, 333, 823, 828, 839, 912, 973, 985, 1019, 1053, 1065 |
| `vllm/v1/attention/backends/cpu_attn.py` | 50, 162, 167, 247, 279, 282, 361, 395, 540, 542 |
| `vllm/v1/attention/backends/flash_attn.py` | 70, 183, 184, 186, 193, 212, 335, 459, 610, 629, 760, 887, 1015 |
| `vllm/v1/attention/backends/flash_attn_diffkv.py` | 139, 232 |
| `vllm/v1/attention/backends/flash_attn_v100.py` | 622, 625, 766, 787, 795, 800, 809, 810, 812, 813, 821, 827, 1173, 1174, 1177, 1180, 1191, 1192, 1202, 1203, 2869, 2870, 2872, 2874, 2880, 2884, 2886, 5037, 5038, 5053, 5061, 5102, 5164, 5293, 5477, 5612, 5648, 5757, 5770, 5862, 5863, 5954, 5976, 5988, 6094, 6136, 6163, 6174, 6179, 6212, 6217, 6222, 6329, 6352, 6369, 6378, 6404, 6432, 6528, 6789, 6833, 6913, 6941, 6971, 7172, 7200, 7223, 7250, 7286, 7312, 7455, 7531, 7536, 7541, 7562, 7569, 7574, 7587, 7591, 7609, 7640, 7657, 7902, 7940, 8245, 8256, 8305, 8393, 8631, 8684, 8689, 8694, 8732, 8752, 8771, 8795, 8826, 8999, 9112, 9344, 9496, 9502, 9513, 9548, 9573, 9596, 9678, 9817 |
| `vllm/v1/attention/backends/flashinfer.py` | 244, 263, 328, 392, 393, 395, 397, 400, 626, 628, 630, 637, 655, 1081, 1155, 1184, 1242, 1271, 1291, 1292, 1352, 1398, 1403, 1458, 1460, 1636, 1852 |
| `vllm/v1/attention/backends/flashinfer_sm70.py` | 439 |
| `vllm/v1/attention/backends/flex_attention.py` | 86, 1057, 1085, 1098, 1137 |
| `vllm/v1/attention/backends/mla/aiter_triton_mla.py` | 25, 39 |
| `vllm/v1/attention/backends/mla/cutlass_mla.py` | 40, 115, 129, 218 |
| `vllm/v1/attention/backends/mla/flashattn_mla.py` | 45, 80, 266, 280, 304, 326 |
| `vllm/v1/attention/backends/mla/flashinfer_mla.py` | 40, 73, 115, 129, 182, 187 |
| `vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py` | 63, 109, 268, 296, 344, 348 |
| `vllm/v1/attention/backends/mla/flashmla.py` | 49, 82, 215, 229, 275, 305 |
| `vllm/v1/attention/backends/mla/flashmla_sparse.py` | 96, 704, 717, 731, 732, 737, 1014 |
| `vllm/v1/attention/backends/mla/rocm_aiter_mla.py` | 56, 196, 197, 198, 200, 201, 658, 672 |
| `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py` | 266, 417, 418, 419, 421, 422, 626, 641, 725 |
| `vllm/v1/attention/backends/mla/tokenspeed_mla.py` | 61, 91, 141, 155, 177, 181 |
| `vllm/v1/attention/backends/mla/triton_mla.py` | 38, 92, 106, 131 |
| `vllm/v1/attention/backends/mla/xpu_mla_sparse.py` | 39, 182, 195, 234 |
| `vllm/v1/attention/backends/rocm_aiter_fa.py` | 279, 288, 712, 787, 803, 860, 955, 1052, 1310, 1344, 1374, 1401, 1412, 1440 |
| `vllm/v1/attention/backends/rocm_aiter_unified_attn.py` | 110, 123, 224, 279, 306 |
| `vllm/v1/attention/backends/rocm_attn.py` | 171, 275, 293, 332, 421, 440, 488, 503, 532 |
| `vllm/v1/attention/backends/triton_attn.py` | 61, 66, 432, 619, 640, 652, 667, 761, 853, 912, 916, 917, 926, 931, 940, 965 |
| `vllm/v1/attention/backends/turboquant_attn.py` | 192, 242, 256, 257, 259, 357, 368, 377, 529, 530, 641, 711, 1422, 1423 |
| `vllm/v1/attention/ops/chunked_prefill_paged_decode.py` | 271, 304, 333, 337, 339, 343, 361, 408 |
| `vllm/v1/attention/ops/paged_attn.py` | 38, 48 |
| `vllm/v1/attention/ops/prefix_prefill.py` | 653, 682, 686, 688, 691, 699, 702 |
| `vllm/v1/attention/ops/sm70_e4m3_grouped.py` | 53, 110 |
| `vllm/v1/attention/ops/sm70_e4m3_scalar.py` | 118, 151 |
| `vllm/v1/attention/ops/sm70_fp16_grouped.py` | 81 |
| `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py` | 19, 20, 21, 335, 361, 362, 366, 371, 377, 382, 526, 540, 541, 545, 549, 553, 558 |
| `vllm/v1/attention/selector.py` | 24, 40, 57, 68, 70, 71, 93 |
| `vllm/v1/kv_cache_interface.py` | 36, 59, 60, 61, 63, 65, 67, 72, 73, 76, 77, 78 |
| `vllm/v1/spec_decode/dflash.py` | 62 |
| `vllm/v1/spec_decode/llm_base_proposer.py` | 2126, 2131 |
| `vllm/v1/utils.py` | 666 |
| `vllm/v1/worker/gpu/model_runner.py` | 166, 169 |
| `vllm/v1/worker/gpu/spec_decode/dflash/utils.py` | 73 |
| `vllm/v1/worker/gpu_model_runner.py` | 131, 1352, 13026 |

The [path matrix](sm70_kv_path_matrix.md) adds QSA/host/scale consumers whose interfaces need not contain the `kv_cache_dtype` spelling. Inventory original and final sources independently; do not count a weight INT8 kernel as an INT8 KV route.
