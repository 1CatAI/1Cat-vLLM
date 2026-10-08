# SM70 KV environment inventory

Inventory at `c4f6245f841466782752a8c3283e4727565cf17a`: **89 distinct backend read names**, counting `envs.VLLM_*` and literal `getenv/get` reads. These counts exclude documentation-only mentions and include inherited non-KV diagnostics. No control is retired or default changed by the first extraction.

Move validated defaults into per-engine typed attention policy. Retain legacy parsers as warning aliases for one release, with explicit old overrides preserving values. Keep experimental scheduling controls in one module. Debug observers should use the existing `VLLM_SM70_DEBUG` channels. Actual removals require parser, hash, graph and route parity evidence.

| Current name | Destination / purpose |
|---|---|
| `VLLM_DFLASH_DDTREE_TRACE_JSONL` | Unified diagnostics; retain legacy compatibility alias |
| `VLLM_DFLASH_DDTREE_TRACE_KV_CACHE_DIFF` | Unified diagnostics; retain legacy compatibility alias |
| `VLLM_DFLASH_DDTREE_TRITON_BRANCH_ATTN` | Existing per-engine speculative policy or graph contract |
| `VLLM_DFLASH_DDTREE_TRITON_BRANCH_ATTN_STRICT` | Existing per-engine speculative policy or graph contract |
| `VLLM_DFLASH_DDTREE_WORKER_PROFILE` | Unified diagnostics; retain legacy compatibility alias |
| `VLLM_FLASH_V100_ALLOW_TRITON_FALLBACK` | Typed backend policy; audit default and override behavior before retirement |
| `VLLM_FLASH_V100_BFLA_KEEP_MASS` | Typed prefill policy; protect 8192-chunk 75T/FA2 admission |
| `VLLM_FLASH_V100_BFLA_KEEP_RATIO` | Typed prefill policy; protect 8192-chunk 75T/FA2 admission |
| `VLLM_FLASH_V100_BFLA_LOCAL_BLOCKS` | Typed prefill policy; protect 8192-chunk 75T/FA2 admission |
| `VLLM_FLASH_V100_BFLA_MASK_BLOCK_N` | Typed prefill policy; protect 8192-chunk 75T/FA2 admission |
| `VLLM_FLASH_V100_BFLA_MIN_KEEP_BLOCKS` | Typed prefill policy; protect 8192-chunk 75T/FA2 admission |
| `VLLM_FLASH_V100_BFLA_MIN_KV` | Typed prefill policy; protect 8192-chunk 75T/FA2 admission |
| `VLLM_FLASH_V100_BFLA_MIN_Q` | Typed prefill policy; protect 8192-chunk 75T/FA2 admission |
| `VLLM_FLASH_V100_BFLA_POOL` | Typed prefill policy; protect 8192-chunk 75T/FA2 admission |
| `VLLM_FLASH_V100_BFLA_PREFILL` | Typed prefill policy; protect 8192-chunk 75T/FA2 admission |
| `VLLM_FLASH_V100_BFLA_SPEC_PROB` | Typed prefill policy; protect 8192-chunk 75T/FA2 admission |
| `VLLM_FLASH_V100_BFLA_SPEC_SEED` | Typed prefill policy; protect 8192-chunk 75T/FA2 admission |
| `VLLM_FLASH_V100_BFLA_SPEC_STRIDE` | Typed prefill policy; protect 8192-chunk 75T/FA2 admission |
| `VLLM_FLASH_V100_BFLA_THRESHOLD` | Typed prefill policy; protect 8192-chunk 75T/FA2 admission |
| `VLLM_FLASH_V100_COMPARE_BHMD_OUT_DIR` | Unified diagnostics; retain legacy compatibility alias |
| `VLLM_FLASH_V100_COMPARE_BHMD_OUT_MAX_CALLS` | Unified diagnostics; retain legacy compatibility alias |
| `VLLM_FLASH_V100_COMPARE_TRITON_OUT_DIR` | Unified diagnostics; retain legacy compatibility alias |
| `VLLM_FLASH_V100_COMPARE_TRITON_OUT_MAX_CALLS` | Unified diagnostics; retain legacy compatibility alias |
| `VLLM_FLASH_V100_COMPARE_TRITON_TENSOR_DUMP_DIR` | Unified diagnostics; retain legacy compatibility alias |
| `VLLM_FLASH_V100_COMPARE_TRITON_TENSOR_DUMP_MAX_TOKENS` | Unified diagnostics; retain legacy compatibility alias |
| `VLLM_FLASH_V100_DEBUG_PREFILL_COMPARE` | Unified diagnostics; retain legacy compatibility alias |
| `VLLM_FLASH_V100_DEBUG_ROUTE_SUMMARY` | Unified diagnostics; retain legacy compatibility alias |
| `VLLM_FLASH_V100_DECODE_DENSE_CACHE` | Typed backend policy; audit default and override behavior before retirement |
| `VLLM_FLASH_V100_DECODE_DENSE_REFERENCE` | Typed backend policy; audit default and override behavior before retirement |
| `VLLM_FLASH_V100_DECODE_DYNAMIC_PARTITIONS` | Central scheduling experiments; promote defaults only after exact-shape A/B |
| `VLLM_FLASH_V100_DECODE_FP8_XQA_MIN_SEQ_LEN` | Codec capability or compatibility gate; remove format-specific scheduling after parity |
| `VLLM_FLASH_V100_DECODE_PARTITION_SIZE` | Central scheduling experiments; promote defaults only after exact-shape A/B |
| `VLLM_FLASH_V100_DECODE_USE_BHMD_OUT` | Typed backend policy; audit default and override behavior before retirement |
| `VLLM_FLASH_V100_DECODE_USE_PAGED_PREFILL` | Typed prefill policy; protect 8192-chunk 75T/FA2 admission |
| `VLLM_FLASH_V100_DECODE_USE_SCALAR_PAGED` | Typed backend policy; audit default and override behavior before retirement |
| `VLLM_FLASH_V100_DECODE_USE_WMMA_WRAPPER` | Typed backend policy; audit default and override behavior before retirement |
| `VLLM_FLASH_V100_DECODE_USE_XQA` | Typed backend policy; audit default and override behavior before retirement |
| `VLLM_FLASH_V100_DECODE_XQA_Q4_MIN_SEQ_LEN` | Typed backend policy; audit default and override behavior before retirement |
| `VLLM_FLASH_V100_DFLASH2_BATCHED_GROUPED_VERIFY` | Existing per-engine speculative policy or graph contract |
| `VLLM_FLASH_V100_DFLASH2_GROUPED_VERIFY` | Existing per-engine speculative policy or graph contract |
| `VLLM_FLASH_V100_DFLASH2_GROUPED_VERIFY_MIN_MODEL_LEN` | Existing per-engine speculative policy or graph contract |
| `VLLM_FLASH_V100_DFLASH_PREFIX_DUMP` | Unified diagnostics; retain legacy compatibility alias |
| `VLLM_FLASH_V100_DISABLE_PAGED_PREFILL` | Typed prefill policy; protect 8192-chunk 75T/FA2 admission |
| `VLLM_FLASH_V100_DRAFT_GRAPH_DEBUG` | Unified diagnostics; retain legacy compatibility alias |
| `VLLM_FLASH_V100_DRAFT_GRAPH_DEBUG_LIMIT` | Unified diagnostics; retain legacy compatibility alias |
| `VLLM_FLASH_V100_E4M3_BATCH_XQA` | Codec capability or compatibility gate; remove format-specific scheduling after parity |
| `VLLM_FLASH_V100_E4M3_GROUPED_FP32` | Codec capability or compatibility gate; remove format-specific scheduling after parity |
| `VLLM_FLASH_V100_ENABLE_PAGED_PREFILL` | Typed prefill policy; protect 8192-chunk 75T/FA2 admission |
| `VLLM_FLASH_V100_FA2_D256_PREFILL` | Typed prefill policy; protect 8192-chunk 75T/FA2 admission |
| `VLLM_FLASH_V100_FP8_PREFILL_BRIDGE` | Typed prefill policy; protect 8192-chunk 75T/FA2 admission |
| `VLLM_FLASH_V100_KERNEL_BLOCK_SIZE16` | Typed backend policy; audit default and override behavior before retirement |
| `VLLM_FLASH_V100_PREFILL_CHUNK_PROFILE` | Unified diagnostics; retain legacy compatibility alias |
| `VLLM_FLASH_V100_PREFILL_CONTIG_DENSE` | Typed prefill policy; protect 8192-chunk 75T/FA2 admission |
| `VLLM_FLASH_V100_PREFILL_CONTIG_DENSE_ALLOW_COPY` | Typed prefill policy; protect 8192-chunk 75T/FA2 admission |
| `VLLM_FLASH_V100_PREFILL_CONTIG_DENSE_MIN_KV` | Typed prefill policy; protect 8192-chunk 75T/FA2 admission |
| `VLLM_FLASH_V100_PREFILL_CONTIG_DENSE_MIN_Q` | Typed prefill policy; protect 8192-chunk 75T/FA2 admission |
| `VLLM_FLASH_V100_PREFILL_D256_GQA_ARCH_128K_EXPERIMENTAL` | Typed prefill policy; protect 8192-chunk 75T/FA2 admission |
| `VLLM_FLASH_V100_PREFILL_D256_GQA_V37` | Typed prefill policy; protect 8192-chunk 75T/FA2 admission |
| `VLLM_FLASH_V100_PREFILL_DENSE_SPLITKV3` | Typed prefill policy; protect 8192-chunk 75T/FA2 admission |
| `VLLM_FLASH_V100_PREFILL_DENSE_SPLITKV3_MIN_KV` | Typed prefill policy; protect 8192-chunk 75T/FA2 admission |
| `VLLM_FLASH_V100_PREFILL_DENSE_SPLITKV3_Q8000_EXPERIMENTAL` | Typed prefill policy; protect 8192-chunk 75T/FA2 admission |
| `VLLM_FLASH_V100_PREFILL_GATHER_DENSE` | Typed prefill policy; protect 8192-chunk 75T/FA2 admission |
| `VLLM_FLASH_V100_PREFILL_GATHER_DENSE_MIN_KV` | Typed prefill policy; protect 8192-chunk 75T/FA2 admission |
| `VLLM_FLASH_V100_PREFILL_GATHER_DENSE_MIN_Q` | Typed prefill policy; protect 8192-chunk 75T/FA2 admission |
| `VLLM_FLASH_V100_PREFILL_PREFIX_DECODE_ROWS` | Typed prefill policy; protect 8192-chunk 75T/FA2 admission |
| `VLLM_FLASH_V100_PREFILL_SPLIT_KV` | Typed prefill policy; protect 8192-chunk 75T/FA2 admission |
| `VLLM_FLASH_V100_PREFILL_SPLIT_KV_MAX_Q` | Typed prefill policy; protect 8192-chunk 75T/FA2 admission |
| `VLLM_FLASH_V100_PREFILL_SPLIT_KV_MIN_KV` | Typed prefill policy; protect 8192-chunk 75T/FA2 admission |
| `VLLM_FLASH_V100_PREFILL_SPLIT_KV_MIN_Q` | Typed prefill policy; protect 8192-chunk 75T/FA2 admission |
| `VLLM_FLASH_V100_PREFILL_SPLIT_KV_TOKENS` | Typed prefill policy; protect 8192-chunk 75T/FA2 admission |
| `VLLM_FLASH_V100_PREFILL_USE_PAGED_CACHE` | Typed prefill policy; protect 8192-chunk 75T/FA2 admission |
| `VLLM_FLASH_V100_PREFILL_USE_TRITON` | Typed prefill policy; protect 8192-chunk 75T/FA2 admission |
| `VLLM_FLASH_V100_ROUTE_SUMMARY` | Unified diagnostics; retain legacy compatibility alias |
| `VLLM_FLASH_V100_SMALLQ_DECODE_MAX_MODEL_LEN` | Typed backend policy; audit default and override behavior before retirement |
| `VLLM_FLASH_V100_SMALLQ_DECODE_MAX_Q` | Typed backend policy; audit default and override behavior before retirement |
| `VLLM_FLASH_V100_SMALLQ_DECODE_USE_XQA` | Typed backend policy; audit default and override behavior before retirement |
| `VLLM_FLASH_V100_SMALLQ_DECODE_XQA_MIN_SEQ_LEN` | Typed backend policy; audit default and override behavior before retirement |
| `VLLM_FLASH_V100_TRACE_DECODE_ACTIVE` | Unified diagnostics; retain legacy compatibility alias |
| `VLLM_FLASH_V100_XQA_BATCH_CONTEXT_ROUTING` | Typed backend policy; audit default and override behavior before retirement |
| `VLLM_FLASH_V100_XQA_E4M3_G6_P64_P256_AUTO` | Central scheduling experiments; promote defaults only after exact-shape A/B |
| `VLLM_FLASH_V100_XQA_G6_P1024_SAWTOOTH` | Central scheduling experiments; promote defaults only after exact-shape A/B |
| `VLLM_FLASH_V100_XQA_MTP5_DUAL_CTA` | Central scheduling experiments; promote defaults only after exact-shape A/B |
| `VLLM_FLASH_V100_XQA_MTP5_PARTITION_SIZE` | Central scheduling experiments; promote defaults only after exact-shape A/B |
| `VLLM_SM70_DEBUG` | Unified diagnostics; retain legacy compatibility alias |
| `VLLM_SM70_DFLASH2_SCALAR_ATTENTION_MANIFEST` | Existing per-engine speculative policy or graph contract |
| `VLLM_SM70_DFLASH2_TAIL_CUDAGRAPHS` | Existing per-engine speculative policy or graph contract |
| `VLLM_SM70_FA2_D256_LIBRARY` | Typed backend policy; audit default and override behavior before retirement |
| `VLLM_SM70_MTP_CONTEXT_BUCKET_PARTITION_SIZE` | Central scheduling experiments; promote defaults only after exact-shape A/B |
| `VLLM_SM70_PROFILE_TRACE` | Unified diagnostics; retain legacy compatibility alias |
