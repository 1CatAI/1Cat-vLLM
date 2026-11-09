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

The four previously unresolved `triton_poi_fused_0` families are now attributed
using their cached decode source and Triton IR. Grids 16/32 gather one
2560-element FP16 embedding row and zero a 1536-element activation buffer.
The embedding load precedes the rank mask, so every rank reads either the
selected row or row zero. Grids 46/92 combine three 10240-element residual
inputs and zero 1536 activation elements; they read no learned weights.
These byte counts describe addressed parameters, including a cached row-zero
load, rather than asserting that every byte reaches DRAM on every token.

The attributed graph weights total 2.407 GB/card/token. Adding the FP16
317.850-MB local LM-head matrix gives 2.725 GB/card/token and a 3.633-ms
weight-only floor at 750 GB/s. The complete actual-DRAM table and effective
bandwidth require counters in the model; this weight audit does not substitute
estimates for those counters.
The [acceptance document](sm70_qwen38_distribution_acceptance.md) records
component priorities, isolated counters and current endpoint qualification.

| Kernel | Grid | Calls/step | Weight B/call | Median us/call | Floor us/step | Actual us/step | Actual minus floor us |
|---|---|---:|---:|---:|---:|---:|---:|
| `vllm::sm70_qwen38_hc_up_mix_push` | [160, 1, 1] | 96.0 | 1638400 | 7.680 | 209.715 | 744.103 | 534.388 |
| `_qsa_sparse_paged_gqa_splitk_kernel` | [1, 1, 64] | 12.0 | 0 | 38.495 | 0.000 | 462.251 | 462.251 |
| `cuBLAS shared gate/up + final HC down` | [40, 1, 10] | 49.0 | 1738710 | 10.496 | 113.596 | 517.156 | 403.561 |
| `void <unnamed>::nvfp4_qpn_m1_sm70_kernel<` | [10, 10, 1] | 48.0 | 5120000 | 14.783 | 327.680 | 724.751 | 397.071 |
| `_sm70_qwen38_router_topk_kernel` | [1, 1, 1] | 48.0 | 0 | 7.472 | 0.000 | 384.019 | 384.019 |
| `void vllm::sm70_cross_device_reduce_1stage_push<` | [3, 1, 1] | 49.0 | 0 | 6.272 | 0.000 | 374.597 | 374.597 |
| `_hc_combine_norm_kernel` | [1, 4, 1] | 95.0 | 20480 | 3.584 | 2.594 | 365.109 | 362.515 |
| `cuBLAS shared down` | [320, 1, 1] | 48.0 | 819200 | 6.880 | 52.429 | 392.530 | 340.101 |
| `void vllm::sm70_cross_device_reduce_sum2_1stage_push<` | [3, 1, 1] | 48.0 | 0 | 6.368 | 0.000 | 324.762 | 324.762 |
| `_qwen38_fp16_row_gemv_kernel` | [512, 1, 1] | 48.0 | 2621440 | 9.904 | 167.772 | 475.885 | 308.113 |
| `<unnamed>::nvfp4_qwen38_w2_direct_reduce_kernel` | [80, 1, 1] | 48.0 | 2560000 | 9.759 | 163.840 | 471.295 | 307.455 |
| `_fp16_gemv_silu_ranges_kernel` | [1, 88, 1] | 96.0 | 1658880 | 5.343 | 212.337 | 512.276 | 299.939 |
| `void <unnamed>::gdn_decode_mixed_qkv_global_state_kernel<c10::Half, c10::Half, float,` | [12, 1, 16] | 36.0 | 72 | 7.648 | 0.003 | 292.782 | 292.778 |
| `_qsa_mqa_paged_kernel` | [1, 2052, 1] | 12.0 | 0 | 21.952 | 0.000 | 264.336 | 264.336 |
| `<unnamed>::qwen38_shared_gate_exact_kernel` | [1, 1, 1] | 48.0 | 5120 | 4.992 | 0.328 | 250.708 | 250.381 |
| `void vllm::act_and_mul_kernel<c10::Half, __half2, &vllm::silu_kernel<c10::Half>, &vllm::pa` | [1, 1, 1] | 48.0 | 0 | 4.224 | 0.000 | 202.896 | 202.896 |
| `void vllm::qsa::qsa_lexicographic_decode_topk_kernel<` | [1, 1, 1] | 12.0 | 0 | 16.416 | 0.000 | 197.484 | 197.484 |
| `void cublasLt::splitKreduce_kernel<` | [1, 20, 1] | 49.0 | 0 | 3.680 | 0.000 | 183.176 | 183.176 |
| `void vllm::sm70_qwen38_hc_down_push_allgather<` | [1, 1, 1] | 24.0 | 0 | 4.992 | 0.000 | 144.233 | 144.233 |
| `_causal_conv1d_update_kernel` | [1, 10, 1] | 36.0 | 20480 | 3.776 | 0.983 | 136.286 | 135.303 |
| `void vllm::sm70_qwen38_hc_down_push_allgather<` | [1, 1, 1] | 24.0 | 0 | 4.832 | 0.000 | 131.957 | 131.957 |
| `_qsa_merge_splitk_kernel` | [1, 6, 1] | 12.0 | 0 | 10.208 | 0.000 | 123.359 | 123.359 |
| `void <unnamed>::rmsnorm_gated_exact_kernel<` | [3, 1, 1] | 36.0 | 256 | 3.312 | 0.012 | 120.908 | 120.896 |
| `void vllm::sm70_qwen38_hc_down_push_allgather<` | [1, 1, 1] | 24.0 | 0 | 4.160 | 0.000 | 110.762 | 110.762 |
| `_qwen38_fp16_row_gemv_kernel` | [2560, 1, 1] | 48.0 | 7864320 | 12.320 | 503.316 | 603.000 | 99.683 |
| `void vllm::sm70_qwen38_hc_down_push_allgather<` | [1, 1, 1] | 24.0 | 0 | 3.776 | 0.000 | 97.346 | 97.346 |
| `void vllm::reshape_and_cache_flash_kernel<unsigned short, unsigned short,` | [1, 1, 1] | 12.0 | 0 | 5.760 | 0.000 | 69.409 | 69.409 |
| `_qsa_pre_indexer_kernel` | [3, 1, 1] | 12.0 | 512 | 4.976 | 0.008 | 66.866 | 66.858 |
| `_qwen38_fp16_gdn_input_kernel` | [4120, 1, 1] | 36.0 | 21094400 | 29.168 | 1012.531 | 1064.503 | 51.972 |
| `triton_poi_fused_zeros_0` | [12, 1, 1] | 21.5 | 0 | 2.000 | 0.000 | 42.926 | 42.926 |
| `_expand_qsa_indices_kernel` | [1, 9, 1] | 12.0 | 0 | 2.976 | 0.000 | 35.673 | 35.673 |
| `_qwen38_fp16_row_gemv_kernel` | [640, 1, 1] | 12.0 | 3276800 | 7.168 | 52.429 | 86.003 | 33.574 |
| `void at::native::vectorized_elementwise_kernel<` | [2, 1, 1] | 12.0 | 0 | 2.656 | 0.000 | 31.779 | 31.779 |
| `triton_per_fused_2` | [7, 1, 1] | 12.0 | 0 | 2.592 | 0.000 | 31.053 | 31.053 |
| `triton_poi_fused_copy__1` | [20, 1, 1] | 12.75 | 0 | 2.400 | 0.000 | 30.702 | 30.702 |
| `_qwen38_fp16_row_gemv_kernel` | [3584, 1, 1] | 12.0 | 18350080 | 26.895 | 293.601 | 322.712 | 29.111 |
| `triton_poi_fused_3` | [15, 1, 1] | 7.75 | 1024 | 3.776 | 0.011 | 29.118 | 29.107 |
| `triton_poi_fused_zeros_0` | [6, 1, 1] | 12.5 | 0 | 2.016 | 0.000 | 25.049 | 25.049 |
| `triton_poi_fused_copy__1` | [10, 1, 1] | 10.25 | 0 | 2.432 | 0.000 | 24.908 | 24.908 |
| `triton_poi_fused_3` | [9, 1, 1] | 4.25 | 1024 | 4.256 | 0.006 | 18.048 | 18.042 |
| `triton_poi_fused_1` | [6, 1, 1] | 6.5 | 0 | 2.272 | 0.000 | 14.966 | 14.966 |
| `void cutlass::Kernel2<cutlass_70_wmma_tensorop_f16_s161616gemm_f16_16x16_64x2_tn_align8>` | [8, 80, 1] | 1.0 | 52428800 | 84.511 | 69.905 | 84.623 | 14.718 |
| `triton_poi_fused_copy__0` | [10, 1, 1] | 6.25 | 0 | 2.176 | 0.000 | 13.685 | 13.685 |
| `triton_poi_fused_1` | [12, 1, 1] | 5.5 | 0 | 2.304 | 0.000 | 12.686 | 12.686 |
| `triton_poi_fused_copy__0` | [20, 1, 1] | 5.75 | 0 | 2.176 | 0.000 | 12.656 | 12.656 |
| `_grouped_gemma_rmsnorm_kernel` | [4, 1, 1] | 2.0 | 20480 | 4.224 | 0.055 | 8.481 | 8.427 |
| `triton_red_fused__to_copy_abs_add_clamp_min_div_mean_mul_pow_rsqrt_sigmoid_sign_sqrt_sum_u` | [4, 1, 1] | 1.0 | 40960 | 6.832 | 0.055 | 6.781 | 6.726 |
| `cuBLAS PLE value` | [320, 1, 1] | 1.0 | 13107200 | 23.696 | 17.476 | 23.694 | 6.218 |
| `_hc_combine_kernel` | [1, 5, 1] | 1.0 | 0 | 5.856 | 0.000 | 5.872 | 5.872 |
| `cuBLAS final HC up` | [1280, 1, 1] | 1.0 | 6553600 | 14.256 | 8.738 | 14.246 | 5.508 |
| `_hc_gate_mix_kernel` | [1, 5, 1] | 1.0 | 0 | 4.768 | 0.000 | 4.744 | 4.744 |
| `void at::native::vectorized_elementwise_kernel<` | [10, 1, 1] | 1.0 | 0 | 4.320 | 0.000 | 4.389 | 4.389 |
| `_dequantize_ple_fp8_bytes_kernel` | [3, 1, 1] | 1.0 | 0 | 3.968 | 0.000 | 4.170 | 4.170 |
| `_qwen38_ple_m1_short_conv_kernel` | [40, 1, 1] | 1.0 | 81920 | 3.328 | 0.109 | 3.366 | 3.257 |
| `_hc_silu_kernel` | [1, 1, 1] | 1.0 | 0 | 2.704 | 0.000 | 2.699 | 2.699 |
| `triton_poi_fused_copy__3` | [10, 1, 1] | 0.75 | 0 | 2.400 | 0.000 | 1.794 | 1.794 |
| `triton_poi_fused_0` | [46, 1, 1] | 0.75 | 0 | 2.336 | 0.000 | 1.749 | 1.749 |
| `triton_poi_fused_zeros_like_2` | [80, 1, 1] | 0.75 | 0 | 2.208 | 0.000 | 1.648 | 1.648 |
| `triton_poi_fused_0` | [32, 1, 1] | 0.5 | 5120 | 3.456 | 0.003 | 1.624 | 1.621 |
| `triton_poi_fused_0` | [16, 1, 1] | 0.5 | 5120 | 3.104 | 0.003 | 1.536 | 1.533 |
| `triton_poi_fused__to_copy_add_mean_mul_pow_rsqrt_view_1` | [40, 1, 1] | 0.5 | 20480 | 2.592 | 0.014 | 1.314 | 1.300 |
| `triton_poi_fused__to_copy_add_mean_mul_pow_rsqrt_view_1` | [80, 1, 1] | 0.5 | 20480 | 2.480 | 0.014 | 1.245 | 1.231 |
| `triton_poi_fused_repeat_1` | [80, 1, 1] | 0.5 | 0 | 2.400 | 0.000 | 1.178 | 1.178 |
| `triton_poi_fused_repeat_1` | [40, 1, 1] | 0.5 | 0 | 2.208 | 0.000 | 1.107 | 1.107 |
| `triton_poi_fused_0` | [92, 1, 1] | 0.25 | 0 | 2.496 | 0.000 | 0.624 | 0.624 |
| `triton_poi_fused_copy__3` | [20, 1, 1] | 0.25 | 0 | 2.400 | 0.000 | 0.602 | 0.602 |
| `triton_poi_fused_zeros_like_2` | [40, 1, 1] | 0.25 | 0 | 1.952 | 0.000 | 0.488 | 0.488 |

## Boundary baselines and actual shared-down traffic

The standalone boundary benchmark uses CUDA 12.8, Torch 2.10/cu128 and the
full-NVLink V100-SXM2-32GB machine. It holds the shared GPU lock. This is a
research microbenchmark, with no model computation or runtime dispatch change.

```bash
CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST=7.0 \
  python benchmarks/benchmark_sm70_decode_boundaries.py --output boundaries.json
```

The graph contains 64 dependent producer/consumer pairs. Kernel-side global
timestamps bracket each boundary; the retained interior intervals exclude
the first and last pair. The global timer is coarse, so use the mean and
range alongside the quantized median. These intervals include the producer's
last store and timestamp instructions; they are not a pure scheduling-gap
measurement and cannot be multiplied by the model kernel count to close TPOT.

| Probe | Median us | Mean us | Sample range us |
| --- | ---: | ---: | ---: |
| Instrumented dependent graph boundary, 558 intervals | 1.024 | 1.2894 | 0.608–2.048 |
| GPU 0–1 system release/acquire flag round trip | 4.8647 | 4.8686 | 4.8320–4.8925 |

The flag test measures nine samples of 4096 round trips after warmup. The two
kernels run on separate GPU streams and validate every final generation;
launch skew is amortized over the loop. It is a two-GPU baseline without
model contention, not a four-rank all-reduce or per-layer synchronization time.

An isolated cold-cache NCU replay of the ordinary shared-down cuBLAS GEMV
uses a real captured input and checkpoint weight, FP16 operands and FP32
accumulation. The addressed weight is 819,200 bytes; actual DRAM reads include
other traffic and are reported separately.

| Grid | Registers/thread | Actual DRAM read B | Actual DRAM write B | NCU us | Read GB/s | Read floor us at 750 GB/s | NCU minus read floor us |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 320×1×1 | 50 | 825152 | 0 | 6.624 | 124.57 | 1.1002 | 5.5238 |

Zero DRAM writes does not mean zero logical output: stores can remain in
cache. NCU cold replay and profiler clock control differ from the graph-node
trace above; these counters do not replace its service times. Ordinary-user
counter access was unavailable, so this isolated sample ran with elevated
profiling permission, without changing driver permissions. Other kernel
families still require actual traffic counters.

## C1 SiLU/down fusion decision

The cold 48-layer operator comparison improved from 7.7733 to 6.5920 us,
with two kernels becoming one. The matched installed-artifact model gate
rejects it. Both endpoint arms use TP4, 262144 startup capacity, fixed 8192
input / 513 output tokens, FP16 dense/activations/KV, FP32 state, CUDA graphs,
disk-mapped ngrams and no MTP. The source-complete artifact is built from
native source `f551e0afea` with Python kernel source `6400e9a04d`.

| Arm | C1 median ms/token | Sample range ms/token | Task scores | Natural EOS |
| --- | ---: | ---: | ---: | ---: |
| Existing registry disables the fusion | 11.04115 | 10.92117–11.17972 | 36/36 | 36/36 |
| Fusion enabled | 11.06298 | 10.86482–11.24090 | 36/36 | 35/36 |

There is no demonstrated endpoint speedup. MBPP-16 stops at 628 tokens in
the control, but reaches 4096 tokens and repeats a long final-answer line
eleven times in the candidate. The 32-position C1 focus also fails the
unchanged KL/raw-logit gates, despite 100% top-1 agreement. Further full
distribution collection, C4 smoke and promotion tracing were skipped after
the C1 quality failure. [PR #872](https://github.com/1CatAI/1Cat-vLLM/pull/872)
is closed without merge; its model kernel count is not presented as measured.

The accepted path therefore retains the earlier graph budget and priority
order. Using its 2.7250-GB addressed-weight audit gives a 3.6334-ms weight-only
floor at 750 GB/s, approximately 7.41 ms below this new control endpoint;
this difference is not a wall-clock decomposition. The 7.5/6-ms phase targets
remain unmet. PLE phase timings are being collected before selecting a cache
or publish-path change.

## PLE publish, lookup, writeback and flag phases

An observer-only C1 run uses the same installed artifact with the rejected
SiLU/down kernel disabled. It captures a fixed 8K/64 timing-only request and
four seeded natural requests, yielding 1196 single-token CPU requests. The
four natural requests finish normally. CPU and notifier records are joined
by request ordinal, with matching token counts checked. These CPU timings
include hook overhead and are not GPU critical-path waits.

| Instrumented phase | Median us | p95 us |
| --- | ---: | ---: |
| Notifier start to socket send, including D2H-event wait | 9372.9 | 10097.4 |
| Socket send call | 188.7 | 295.5 |
| Socket-send start to CPU dispatch | 246.5 | 337.6 |
| Previous-result consumption/reset wait | 46.3 | 66.4 |
| Ngram key computation | 143.6 | 192.7 |
| Mmap row gather | 1610.2 | 1945.6 |
| Staging overhead beyond row gather | 72.0 | 98.5 |
| Result fan-out plus four flag publications | 89.0 | 177.6 |
| Four flag calls, included in fan-out above | 23.2 | 47.7 |
| Complete CPU handler | 1942.0 | 2432.6 |

The notifier's 9.37-ms wait includes queued preceding GPU work. It must not
be labeled a 9.37-ms PLE lookup penalty. Socket-call and dispatch intervals
overlap; nested phase medians must not be added to form a wall budget.
The 100-ms process samples enclosing the decode campaign, including its
interleaved prefills, record 19008 major and 22501 minor faults. This supports
investigating cold file-page reads before treating flag publication as the
main PLE cost. A fresh graph trace is still needed to measure the resulting
GPU critical-path wait.

A decode-only cold-start LRU simulation, without prefill warming, gives
22.7% individual-row hits and 14.5% all-row hits with 8192 rows. Increasing
to 65536 rows gives 24.4% row hits and 16.1% all-row hits. This is a trace
simulation, not a measured UVA cache speedup; a cache still needs an efficient
CPU miss path and byte/sequence checks.

CPU-only probes rotate 16 recorded row sets from the real checkpoint. Cold
arms discard only the selected file pages, with major faults confirming
physical cold reads. Warm arms retain them. Nine alternating repetitions
check every output byte and retain the table as mmap throughout.

| Lookup experiment | Warm mean-per-case median us | Cold mean-per-case median us |
| --- | ---: | ---: |
| Serial row copies | 28.24 | 1450.12 |
| Sixteen worker threads | 409.59 | 846.63 |
| Selected-page read-ahead then serial copies | 80.30 | 246.13 |
| Residency check then read-ahead for cold pages | 89.29 | 295.65 |

The thread-pool variant has excessive warm overhead. Selected-page read-ahead
is promising for misses, but these are CPU microbenchmarks; no model speedup
or default admission follows from them. The probe never prefaults the whole
table. Residency is a snapshot, and a page can be reclaimed before copying;
the original file-backed read remains the correctness path. See the Linux
[mincore API](https://man7.org/linux/man-pages/man2/mincore.2.html) and
[madvise API](https://man7.org/linux/man-pages/man2/madvise.2.html).
