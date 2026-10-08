# SM70 KV path × format ledger

Base: `c4f6245f841466782752a8c3283e4727565cf17a`. This is the source inventory, not a GPU acceptance result. `review` means the exact format/shape admission still needs auditing. No INT8 Flash-V100 route is claimed implemented. Dense routes can consume codec output; a `bridge` cell is not native quantized arithmetic.

The JSON beside this document is the machine-readable source of truth. Tests check every literal and dynamic `_record_route` call and the required four format columns.

| Neutral route | Existing counter | FP16 | E4M3 | E5M2 | INT8 |
|---|---|---|---|---|---|
| decode_dense_cache | `decode_dense_cache` | native | unsupported | unsupported | pending |
| decode_dense_reference | `decode_dense_reference` | reference | reference | reference | pending |
| decode_compact_scalar_tail | `decode_e4m3_compact_scalar_tail` | unsupported | native | unsupported | pending |
| decode_paged_prefill | `decode_paged_prefill` | review | review | review | pending |
| decode_scalar_paged | `decode_scalar_paged` | native | native | native | pending |
| decode_triton_no_flash_decode | `decode_triton_no_flash_decode` | fallback | fallback | fallback | pending |
| decode_triton_scalar_disabled | `decode_triton_scalar_disabled` | fallback | fallback | fallback | pending |
| decode_xqa_paged | `decode_xqa_paged` | native | native | native | pending |
| metadata_none_zero_output | `metadata_none_zero_output` | metadata | metadata | metadata | pending |
| prefill_capture_dflash_noncausal_paged | `prefill_capture_dflash_noncausal_paged` | metadata | metadata | metadata | pending |
| prefill_capture_smallq | `prefill_capture_smallq` | metadata | metadata | metadata | pending |
| prefill_capture_smallq_ddtree_metadata | `prefill_capture_smallq_ddtree_metadata` | metadata | metadata | metadata | pending |
| prefill_capture_smallq_no_ddtree_metadata | `prefill_capture_smallq_no_ddtree_metadata` | metadata | metadata | metadata | pending |
| prefill_ddtree_dense | `prefill_ddtree_dense` | review | review | review | pending |
| prefill_ddtree_triton | `prefill_ddtree_triton` | fallback | fallback | fallback | pending |
| prefill_dense_d256_gqa_79t_fp32 | `prefill_dense_d256_gqa_79t_fp32` | review | review | review | pending |
| prefill_dense_d256_gqa_79t_fp32_fringe_fallback | `prefill_dense_d256_gqa_79t_fp32_fringe_fallback` | review | review | review | pending |
| prefill_dense_d256_gqa_79t_fp32_q8192 | `prefill_dense_d256_gqa_79t_fp32_q8192` | review | review | review | pending |
| prefill_dense_d256_gqa_79t_fp32_q8192_pad | `prefill_dense_d256_gqa_79t_fp32_q8192_pad` | review | review | review | pending |
| prefill_dense_d256_gqa_arch_long | `prefill_dense_d256_gqa_arch_long` | review | review | review | pending |
| prefill_dense_d256_gqa_v37 | `prefill_dense_d256_gqa_v37` | review | review | review | pending |
| prefill_dense_splitd_d256 | `prefill_dense_splitd_d256` | review | review | review | pending |
| prefill_dense_splitd_d256_splitkv3_kernel | `prefill_dense_splitd_d256_splitkv3_kernel` | review | review | review | pending |
| prefill_no_prefix_dense_flash | `prefill_no_prefix_dense_flash` | review | review | review | pending |
| prefill_no_prefix_paged_cache_flash | `prefill_no_prefix_paged_cache_flash` | review | review | review | pending |
| prefill_prefix_bfla | `prefill_prefix_bfla` | review | review | review | pending |
| prefill_prefix_contig_dense | `prefill_prefix_contig_dense` | review | review | review | pending |
| prefill_prefix_contig_dense_bhmd | `prefill_prefix_contig_dense_bhmd` | review | review | review | pending |
| prefill_prefix_contig_dense_fa2_d256 | `prefill_prefix_contig_dense_fa2_d256` | review | review | review | pending |
| prefill_prefix_decode_rows_grouped_fp32 | `prefill_prefix_decode_rows_e4m3_grouped_fp32` | unsupported | native | unsupported | pending |
| prefill_prefix_dflash_noncausal_batch | `prefill_prefix_dflash_noncausal_batch` | review | review | review | pending |
| prefill_prefix_flash | `prefill_prefix_flash` | review | review | review | pending |
| prefill_prefix_kv_bridge_exact_d256 | `prefill_prefix_fp8_bridge_exact_d256` | unsupported | bridge | bridge | pending |
| prefill_prefix_kv_bridge_exact_d256_tailpad | `prefill_prefix_fp8_bridge_exact_d256_tailpad` | unsupported | bridge | bridge | pending |
| prefill_prefix_kv_bridge_exact_dense_d256 | `prefill_prefix_fp8_bridge_exact_dense_d256` | unsupported | bridge | bridge | pending |
| prefill_prefix_kv_bridge_exact_dense_d256_tailpad | `prefill_prefix_fp8_bridge_exact_dense_d256_tailpad` | unsupported | bridge | bridge | pending |
| prefill_prefix_paged_anchored | `prefill_prefix_paged_anchored` | review | review | review | pending |
| prefill_prefix_splitkv | `prefill_prefix_splitkv` | review | review | review | pending |
| verify_decode_scalar | `prefill_smallq_decode_scalar` | native | native | native | pending |
| verify_decode_xqa | `prefill_smallq_decode_xqa` | native | native | native | pending |
| verify_dflash2_grouped_verify | `prefill_smallq_dflash2_grouped_verify` | review | review | review | pending |
| verify_grouped_fp32 | `prefill_smallq_e4m3_grouped_fp32` | unsupported | native | unsupported | pending |
| verify_grouped_fp32 | `prefill_smallq_fp16_grouped_fp32` | native | unsupported | unsupported | pending |
| prefill_triton_safe | `prefill_triton_safe` | fallback | fallback | fallback | pending |

## Dynamic route families

These eight expressions include conditional fallbacks, bridge observer counters, dynamic page/partition labels, variable mixed-row routes and the split-D alias. They are not omitted just because they are not literal arguments.

- `'dflash_draft_triton_fallback' if is_dflash_draft_attn else 'unsupported_triton_fallback'`
- `'prefill_prefix_fp8_e4m3_bridge' if self.kv_cache_dtype == 'fp8_e4m3' else 'prefill_prefix_fp8_e5m2_bridge'`
- `f'decode_xqa_e4m3_dynamic_page{key_cache.shape[1]}'`
- `f'decode_xqa_p{partition_size_hint}_page{key_cache.shape[1]}'`
- `f'fp8_kv_{stage}'`
- `f'fp8_kv_{stage}_{route}'`
- `fa2_route or 'prefill_prefix_splitd_d256'`
- `route`

## Additional consumers

| Path | Scope | INT8 status | Source |
|---|---|---|---|
| verify_grouped | DFlash2 grouped verify and combined-copy layouts | pending | `vllm/v1/attention/ops/sm70_fp16_grouped.py` |
| prefix_decode_rows | variable scalar/XQA mixed rows | pending | `vllm/v1/attention/backends/flash_attn_v100.py` |
| decode_dynamic | dynamic page/partition route counters | pending | `flash-attention-v100/kernel/flash_decode_paged.cu` |
| kv_write_cuda | reshape_and_cache_flash | pending | `csrc/libtorch_stable/cache_kernels.cu` |
| kv_write_triton | reshape_and_cache_flash Triton / per-token-head writer | upstream_only | `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py` |
| kv_write_fused_rope | DFlash fused QK RoPE cache writes | pending | `csrc/libtorch_stable` |
| qsa_prepare | Flash-Next fused preparation / MTP4 | pending | `vllm/models/qwen4_exp/nvidia/qsa.py` |
| qsa_attention | sparse QSA attention / indexer | pending | `vllm/models/qwen4_exp/nvidia/ops/qsa.py` |
| qsa_policy | automatic cache policy and calibration | pending | `vllm/models/qwen4_exp/common/kv_policy.py` |
| host_transport | host offload, pinned history, hot cache restoration | pending | `vllm/models/qwen4_exp/common/qsa_cache.py` |
| cache_budget | page size, asymmetric K/V dimensions, scale bytes | upstream_only | `vllm/v1/kv_cache_interface.py` |
| hybrid_alignment | Mamba grid, hybrid cache page uniformity | pending | `vllm/v1/core/kv_cache_utils.py` |
| cache_allocate | payload and scale tensor allocation | upstream_only | `vllm/v1/worker/gpu_model_runner.py` |
| prefix_restore | hash/cache groups and offload restore | pending | `vllm/v1/core/kv_cache_manager.py` |
| graph_variants | graph workspace/active partition envelopes | pending | `vllm/v1/cudagraph_dispatcher.py` |
| draft_cache | target/draft cache dtype configuration | pending | `vllm/config/speculative.py` |
| packed_decode | existing K8V4 codec and attention | pending | `flash-attention-v100/kernel/flash_decode_turboquant.cu` |
| prefill_dense_8192 | 75T/79T and FA2 D256 after conversion | pending | `csrc/attention/sm70_v37/bridge.cu` |

## Qualification dimensions

Every row needs q=1 and applicable verify/prefill shapes, GQA and head dimensions, TP2/TP4 local KV heads, standard and compact pages, random/prefix-restored block tables, tails, graph replay with changing lengths, zero padding, anchored sliding windows and target/draft format combinations. Host transport must round-trip payload **and** scales. Add evidence to the row only after numerical, operator and route-asserted model gates pass.

The old counter names remain compatibility identifiers during the first extraction. The neutral names describe the eventual routing API; this PR does not rename live counters or claim format-independent dispatch is already complete.
