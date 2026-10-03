# Flash-Next Step 0 kernel weight budget

This audit uses the retained mapped-transport graph: graphs 2–6 on four ranks,
1349 nodes per graph, FP16 dense/input, FP32 state/accumulation, NVFP4 routed
experts, TP4, CUDA graphs, no MTP, disk-mapped ngrams, 8192 input / 513 output,
262144 startup capacity. Source eac8c525d, CUDA 12.8, Torch 2.10/cu128,
four V100 SXM2 32-GB cards with full NVLink connectivity.

Rows are sorted by service time minus the parameter-read floor at 750 GB/s.
`Actual us/step` is mean kernel service across the 20 selected graphs, not a
closed wall-clock contribution. Weight bytes count unique accessed prepared
parameters per call, not NCU DRAM bytes. Zero means no learned weights;
activation/state/KV/communication traffic is not zero. Shared gate/up and final
HC down have the same cuBLAS symbol/grid, so that row uses their weighted mean.
HC down reads 81 rows per rank, excluding padding; HC up addresses the local
2560 output rows. Routed-expert bytes include prepared FP16 scales for all ten
selected experts. QSA affine weights and PLE norms were verified against the
cached generated decode source. GDN bias is FP16 and A_log FP32, confirmed by
the instantiated trace template.

Rank-masked embedding aliases remain pending input-to-node attribution. The
complete actual-DRAM table and effective bandwidth require counters in the
model; this weight audit does not substitute estimates for those counters.
The [acceptance document](sm70_qwen38_distribution_acceptance.md) records
component priorities, isolated counters and current endpoint qualification.

| Kernel | Grid | Calls/step | Weight B/call | Floor us/step | Actual us/step | Actual minus floor us |
|---|---|---:|---:|---:|---:|---:|
| vllm::sm70_qwen38_hc_up_mix_push | [160, 1, 1] | 96.0 | 1638400 | 209.715 | 744.103 | 534.388 |
| _qsa_sparse_paged_gqa_splitk_kernel | [1, 1, 64] | 12.0 | 0 | 0.000 | 462.251 | 462.251 |
| void gemv2T_kernel_val<int, int, __half, float, float, float,  | [40, 1, 10] | 49.0 | 1738710 | 113.596 | 517.156 | 403.561 |
| void <unnamed>::nvfp4_qpn_m1_sm70_kernel< | [10, 10, 1] | 48.0 | 5120000 | 327.680 | 724.751 | 397.071 |
| _sm70_qwen38_router_topk_kernel | [1, 1, 1] | 48.0 | 0 | 0.000 | 384.019 | 384.019 |
| void vllm::sm70_cross_device_reduce_1stage_push< | [3, 1, 1] | 49.0 | 0 | 0.000 | 374.597 | 374.597 |
| _hc_combine_norm_kernel | [1, 4, 1] | 95.0 | 20480 | 2.594 | 365.109 | 362.515 |
| void gemv2T_kernel_val<int, int, __half,__half, __half, float,  | [320, 1, 1] | 48.0 | 819200 | 52.429 | 392.530 | 340.101 |
| void vllm::sm70_cross_device_reduce_sum2_1stage_push< | [3, 1, 1] | 48.0 | 0 | 0.000 | 324.762 | 324.762 |
| _qwen38_fp16_row_gemv_kernel | [512, 1, 1] | 48.0 | 2621440 | 167.772 | 475.885 | 308.113 |
| <unnamed>::nvfp4_qwen38_w2_direct_reduce_kernel | [80, 1, 1] | 48.0 | 2560000 | 163.840 | 471.295 | 307.455 |
| _fp16_gemv_silu_ranges_kernel | [1, 88, 1] | 96.0 | 1658880 | 212.337 | 512.276 | 299.939 |
| void <unnamed>::gdn_decode_mixed_qkv_global_state_kernel<c10::Half, c10::Half, float,  | [12, 1, 16] | 36.0 | 72 | 0.003 | 292.782 | 292.778 |
| _qsa_mqa_paged_kernel | [1, 2052, 1] | 12.0 | 0 | 0.000 | 264.336 | 264.336 |
| <unnamed>::qwen38_shared_gate_exact_kernel | [1, 1, 1] | 48.0 | 5120 | 0.328 | 250.708 | 250.381 |
| void vllm::act_and_mul_kernel<c10::Half, __half2, &vllm::silu_kernel<c10::Half>, &vllm::pa | [1, 1, 1] | 48.0 | 0 | 0.000 | 202.896 | 202.896 |
| void vllm::qsa::qsa_lexicographic_decode_topk_kernel< | [1, 1, 1] | 12.0 | 0 | 0.000 | 197.484 | 197.484 |
| void cublasLt::splitKreduce_kernel< | [1, 20, 1] | 49.0 | 0 | 0.000 | 183.176 | 183.176 |
| void vllm::sm70_qwen38_hc_down_push_allgather< | [1, 1, 1] | 24.0 | 0 | 0.000 | 144.233 | 144.233 |
| _causal_conv1d_update_kernel | [1, 10, 1] | 36.0 | 20480 | 0.983 | 136.286 | 135.303 |
| void vllm::sm70_qwen38_hc_down_push_allgather< | [1, 1, 1] | 24.0 | 0 | 0.000 | 131.957 | 131.957 |
| _qsa_merge_splitk_kernel | [1, 6, 1] | 12.0 | 0 | 0.000 | 123.359 | 123.359 |
| void <unnamed>::rmsnorm_gated_exact_kernel< | [3, 1, 1] | 36.0 | 256 | 0.012 | 120.908 | 120.896 |
| void vllm::sm70_qwen38_hc_down_push_allgather< | [1, 1, 1] | 24.0 | 0 | 0.000 | 110.762 | 110.762 |
| _qwen38_fp16_row_gemv_kernel | [2560, 1, 1] | 48.0 | 7864320 | 503.316 | 603.000 | 99.683 |
| void vllm::sm70_qwen38_hc_down_push_allgather< | [1, 1, 1] | 24.0 | 0 | 0.000 | 97.346 | 97.346 |
| void vllm::reshape_and_cache_flash_kernel<unsigned short, unsigned short,  | [1, 1, 1] | 12.0 | 0 | 0.000 | 69.409 | 69.409 |
| _qsa_pre_indexer_kernel | [3, 1, 1] | 12.0 | 512 | 0.008 | 66.866 | 66.858 |
| _qwen38_fp16_gdn_input_kernel | [4120, 1, 1] | 36.0 | 21094400 | 1012.531 | 1064.503 | 51.972 |
| triton_poi_fused_zeros_0 | [12, 1, 1] | 21.5 | 0 | 0.000 | 42.926 | 42.926 |
| _expand_qsa_indices_kernel | [1, 9, 1] | 12.0 | 0 | 0.000 | 35.673 | 35.673 |
| _qwen38_fp16_row_gemv_kernel | [640, 1, 1] | 12.0 | 3276800 | 52.429 | 86.003 | 33.574 |
| void at::native::vectorized_elementwise_kernel< | [2, 1, 1] | 12.0 | 0 | 0.000 | 31.779 | 31.779 |
| triton_per_fused_2 | [7, 1, 1] | 12.0 | 0 | 0.000 | 31.053 | 31.053 |
| triton_poi_fused_copy__1 | [20, 1, 1] | 12.75 | 0 | 0.000 | 30.702 | 30.702 |
| _qwen38_fp16_row_gemv_kernel | [3584, 1, 1] | 12.0 | 18350080 | 293.601 | 322.712 | 29.111 |
| triton_poi_fused_3 | [15, 1, 1] | 7.75 | 1024 | 0.011 | 29.118 | 29.107 |
| triton_poi_fused_zeros_0 | [6, 1, 1] | 12.5 | 0 | 0.000 | 25.049 | 25.049 |
| triton_poi_fused_copy__1 | [10, 1, 1] | 10.25 | 0 | 0.000 | 24.908 | 24.908 |
| triton_poi_fused_3 | [9, 1, 1] | 4.25 | 1024 | 0.006 | 18.048 | 18.042 |
| triton_poi_fused_1 | [6, 1, 1] | 6.5 | 0 | 0.000 | 14.966 | 14.966 |
| void cutlass::Kernel2<cutlass_70_wmma_tensorop_f16_s161616gemm_f16_16x16_64x2_tn_align8> | [8, 80, 1] | 1.0 | 52428800 | 69.905 | 84.623 | 14.718 |
| triton_poi_fused_copy__0 | [10, 1, 1] | 6.25 | 0 | 0.000 | 13.685 | 13.685 |
| triton_poi_fused_1 | [12, 1, 1] | 5.5 | 0 | 0.000 | 12.686 | 12.686 |
| triton_poi_fused_copy__0 | [20, 1, 1] | 5.75 | 0 | 0.000 | 12.656 | 12.656 |
| _grouped_gemma_rmsnorm_kernel | [4, 1, 1] | 2.0 | 20480 | 0.055 | 8.481 | 8.427 |
| triton_red_fused__to_copy_abs_add_clamp_min_div_mean_mul_pow_rsqrt_sigmoid_sign_sqrt_sum_u | [4, 1, 1] | 1.0 | 40960 | 0.055 | 6.781 | 6.726 |
| void gemv2T_kernel_val<int, int, __half,__half, __half, float,  | [320, 1, 1] | 1.0 | 13107200 | 17.476 | 23.694 | 6.218 |
| _hc_combine_kernel | [1, 5, 1] | 1.0 | 0 | 0.000 | 5.872 | 5.872 |
| void gemv2T_kernel_val<int, int, __half,__half, __half, float,  | [1280, 1, 1] | 1.0 | 6553600 | 8.738 | 14.246 | 5.508 |
| _hc_gate_mix_kernel | [1, 5, 1] | 1.0 | 0 | 0.000 | 4.744 | 4.744 |
| void at::native::vectorized_elementwise_kernel< | [10, 1, 1] | 1.0 | 0 | 0.000 | 4.389 | 4.389 |
| _dequantize_ple_fp8_bytes_kernel | [3, 1, 1] | 1.0 | 0 | 0.000 | 4.170 | 4.170 |
| _qwen38_ple_m1_short_conv_kernel | [40, 1, 1] | 1.0 | 81920 | 0.109 | 3.366 | 3.257 |
| _hc_silu_kernel | [1, 1, 1] | 1.0 | 0 | 0.000 | 2.699 | 2.699 |
| triton_poi_fused_copy__3 | [10, 1, 1] | 0.75 | 0 | 0.000 | 1.794 | 1.794 |
| triton_poi_fused_zeros_like_2 | [80, 1, 1] | 0.75 | 0 | 0.000 | 1.648 | 1.648 |
| triton_poi_fused__to_copy_add_mean_mul_pow_rsqrt_view_1 | [40, 1, 1] | 0.5 | 20480 | 0.014 | 1.314 | 1.300 |
| triton_poi_fused__to_copy_add_mean_mul_pow_rsqrt_view_1 | [80, 1, 1] | 0.5 | 20480 | 0.014 | 1.245 | 1.231 |
| triton_poi_fused_repeat_1 | [80, 1, 1] | 0.5 | 0 | 0.000 | 1.178 | 1.178 |
| triton_poi_fused_repeat_1 | [40, 1, 1] | 0.5 | 0 | 0.000 | 1.107 | 1.107 |
| triton_poi_fused_copy__3 | [20, 1, 1] | 0.25 | 0 | 0.000 | 0.602 | 0.602 |
| triton_poi_fused_zeros_like_2 | [40, 1, 1] | 0.25 | 0 | 0.000 | 0.488 | 0.488 |
| triton_poi_fused_0 | [32, 1, 1] | 0.5 | pending | pending | 1.624 | pending |
| triton_poi_fused_0 | [92, 1, 1] | 0.25 | pending | pending | 0.624 | pending |
| triton_poi_fused_0 | [16, 1, 1] | 0.5 | pending | pending | 1.536 | pending |
| triton_poi_fused_0 | [46, 1, 1] | 0.75 | pending | pending | 1.749 | pending |
