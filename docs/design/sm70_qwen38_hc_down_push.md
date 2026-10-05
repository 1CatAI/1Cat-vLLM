# Flash-Next retained critical-stream budget

The retained step0-mapped graph-node trace is reused until a major HC or QSA
change is merged. No new trace was collected for this accounting. SHA256:
`19409a9c6d3ec760d8543418c6caee1955530501e14344d1b56d04541a0ecdc6`. The correction from traced service to an
unprofiled estimate is 0.88, as specified for this capture.

The parser selects the five interior complete replays per rank (20 total),
splits layers at their HC input boundaries, and accounts separately for stream
364 and the auxiliary stream. The first layers cross the capture stream
transition and must be retained individually. Intersections are timestamp
unions, so overlapping auxiliary work does not become projected endpoint gain.
Auxiliary work in critical-stream gaps is only an exposed-service upper bound;
a dependency claim still requires the producer/consumer ordering.

| Regular layer | Span us | Stream 364 service us | Auxiliary service us | Auxiliary overlap us | Weight floor us at 750 GB/s |
|---|---:|---:|---:|---:|---:|
| GDN | 185.561 | 154.274 | 35.629 | 29.164 | 64.506 |
| QSA | 304.679 | 268.577 | 31.799 | 29.237 | 65.190 |

The floor includes all prepared layer parameters, including the overlapping
shared branch. It is not a sum of measured HBM bytes. The GDN span includes
its final gap up to the next layer boundary; this distinguishes it from a
kernel-only service sum or the earlier 182-us layer observation.

HC down/push on stream 364 has 94 calls per step, totaling 973.037 us
(10.351 us/call); the two earlier calls remain on the stream-transition side.
A 7.5-us fused target projects 0.236 ms/token of savings after correction.
No additional launch gaps or overlapped service are credited. Each call
addresses 1658880 bytes of FP16 down weights, a 2.212-us weight floor;
transport and activation traffic remain separate.

Existing allreduce/Gemma norm work is tracked by PR #884. This change focuses
on down/SiLU/push and weight prefetch before the next input wait. The prototype
uses per-output generation tags and two packet slots, without a cooperative
whole-HC launch or a global grid barrier. The standalone up/gather retains
its existing cooperative residency contract; grids beyond resident capacity
use separate publish/consume kernels, selected by occupancy. It retains FP16 projection boundaries and
FP32 accumulation. Its four-branch up kernel compiles with 32 registers and
no spills; PTX places the four vector weight loads before the first packet
poll. These are compilation findings, not GPU correctness or speed admission.

## Closeout and shared-layer scope

The old down/push prototype receives one cold-weight, full-HC-boundary screen
before further implementation. Rotate all 48 layers and both HC boundaries,
include combine/norm in both arms, alternate graph timings, and report M1,
M4 and M5 plus graph kernel counts. Its existing occupancy fallback is retained
only to measure this historical prototype. Timing uses deterministic synthetic
activations with real checkpoint weights; it cannot substitute for a
real-activation or model-quality gate. An unsuccessful result is retained and
the prototype discarded rather than tuned further.

After closeout, the shared-layer implementation covers both single-token
decode and MTP4 target verification at M5. Draft generation and round
integration remain separate. The next HC change removes combine/norm and the
standalone down all-gather together: each down CTA redundantly computes the
four-stream combine and norm, CTA zero publishes the materialized normalized
mixer input, and each down CTA pushes its projected output. Up consumes the
tagged inputs. Multitoken scheduling must reuse weights and fit every waiting
CTA within the measured resident capacity; M4 must retain its accelerated
route and must not regress. Do not carry the old large-batch fallback forward
as the solution for M5.

No grid-wide barrier or cooperative whole-HC segment is proposed. Permitted
dependencies are redundant small computations, producer epilogue publication,
and consumer prologue waits. Floating-point partials use scratch and a fixed
reduction order, without floating atomic addition. A standalone cooperative
launch can enforce residency for a waiting consumer; it does not authorize
inter-CTA grid synchronization.

Each accepted step requires a complete layer graph with real weights and cold
L2 at M1 and M5, kernel counts and changed-input replays. Communication paths
also test generation wrap and simultaneous residency. Model integration must
report total graph kernels and single-CTA kernels. Accumulation-order changes
record teacher-forcing KL mean/p99/max and top-1 agreement, and require healthy
natural completions plus 128K/258K needle retrieval. Maximum logit difference
is recorded without a veto threshold. FP16 inputs/weights and FP32 accumulation
remain unchanged. C4 cannot regress. Quantization retains its separate gates.

Run model endpoints after approximately 1 ms of cumulative exposed savings,
or at a PR admission boundary; measure M1 token latency and the complete MTP4
round separately. Discount auxiliary-stream overlap and compare observed with
projected savings, investigating deviations beyond 15%. Do not repeat the
existing trace until a substantial HC or QSA change is merged.

The initial node-removal ledger is a projection, not a measured result:

| Segment | Planned node removal per token |
|---|---:|
| 96 HC boundaries, four nodes to two | 192 |
| Router top-k into expert W13 | 48 |
| Expert W13 and W2 into one partial-producing kernel | 48 |
| Shared expert, five nodes to one | 192 |
| Attention and MoE reduction producers/consumers | Approximately 96 |
| 36 GDN cores, three nodes to one | 72 |
| 12 QSA chains, fourteen nodes to six | Approximately 96 |

These steps total approximately 744 removed nodes, leaving about 605 of the
1349-node baseline. Meeting the 600-node target still requires accounting for
the final mixer and remaining clear/copy nodes. The zero-single-CTA target
also requires removing QSA fill and final-mixer activation nodes. Neither
target follows automatically from summing the main layer changes.

## Per-layer accounting

| Layer (zero based) | Kind | Kernels | Stream 364 kernels | Sync kernels on 364 | Span us | Auxiliary overlap us | Weight bytes | Floor us | Span minus floor us |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| embedding | embedding | 3 | 0 | 0 | 38.489 | 0.000 | 5120 | 0.007 | 38.482 |
| 0 | GDN | 24 | 5 | 1 | 206.556 | 31.368 | 48379208 | 64.506 | 142.051 |
| 1 | GDN | 35 | 30 | 6 | 1368.908 | 30.923 | 114058568 | 152.078 | 1216.830 |
| 2 | GDN | 27 | 22 | 6 | 188.083 | 26.646 | 48379208 | 64.506 | 123.577 |
| 3 | QSA | 33 | 28 | 6 | 307.685 | 29.278 | 48892416 | 65.190 | 242.495 |
| 4 | GDN | 25 | 20 | 6 | 185.384 | 28.509 | 48379208 | 64.506 | 120.878 |
| 5 | GDN | 26 | 21 | 6 | 183.331 | 26.437 | 48379208 | 64.506 | 118.825 |
| 6 | GDN | 27 | 22 | 6 | 187.587 | 30.579 | 48379208 | 64.506 | 123.081 |
| 7 | QSA | 33 | 28 | 6 | 305.766 | 29.395 | 48892416 | 65.190 | 240.576 |
| 8 | GDN | 25 | 20 | 6 | 182.964 | 28.587 | 48379208 | 64.506 | 118.459 |
| 9 | GDN | 26 | 21 | 6 | 182.272 | 27.859 | 48379208 | 64.506 | 117.766 |
| 10 | GDN | 27 | 22 | 6 | 187.686 | 31.283 | 48379208 | 64.506 | 123.180 |
| 11 | QSA | 33 | 28 | 6 | 305.282 | 29.486 | 48892416 | 65.190 | 240.092 |
| 12 | GDN | 25 | 20 | 6 | 186.673 | 30.801 | 48379208 | 64.506 | 122.168 |
| 13 | GDN | 26 | 21 | 6 | 183.057 | 28.046 | 48379208 | 64.506 | 118.552 |
| 14 | GDN | 27 | 22 | 6 | 186.944 | 27.184 | 48379208 | 64.506 | 122.438 |
| 15 | QSA | 33 | 28 | 6 | 302.921 | 29.497 | 48892416 | 65.190 | 237.732 |
| 16 | GDN | 25 | 20 | 6 | 184.027 | 31.080 | 48379208 | 64.506 | 119.521 |
| 17 | GDN | 26 | 21 | 6 | 182.248 | 32.756 | 48379208 | 64.506 | 117.742 |
| 18 | GDN | 27 | 22 | 6 | 186.635 | 28.101 | 48379208 | 64.506 | 122.129 |
| 19 | QSA | 33 | 28 | 6 | 303.659 | 28.558 | 48892416 | 65.190 | 238.469 |
| 20 | GDN | 25 | 20 | 6 | 185.224 | 26.646 | 48379208 | 64.506 | 120.718 |
| 21 | GDN | 26 | 21 | 6 | 183.115 | 30.041 | 48379208 | 64.506 | 118.609 |
| 22 | GDN | 27 | 22 | 6 | 186.972 | 29.121 | 48379208 | 64.506 | 122.467 |
| 23 | QSA | 33 | 28 | 6 | 304.416 | 29.309 | 48892416 | 65.190 | 239.226 |
| 24 | GDN | 25 | 20 | 6 | 184.661 | 28.044 | 48379208 | 64.506 | 120.155 |
| 25 | GDN | 26 | 21 | 6 | 183.144 | 31.201 | 48379208 | 64.506 | 118.638 |
| 26 | GDN | 27 | 22 | 6 | 187.352 | 29.921 | 48379208 | 64.506 | 122.846 |
| 27 | QSA | 33 | 28 | 6 | 305.123 | 29.413 | 48892416 | 65.190 | 239.933 |
| 28 | GDN | 25 | 20 | 6 | 182.934 | 30.889 | 48379208 | 64.506 | 118.428 |
| 29 | GDN | 26 | 21 | 6 | 183.022 | 31.416 | 48379208 | 64.506 | 118.516 |
| 30 | GDN | 27 | 22 | 6 | 187.187 | 29.201 | 48379208 | 64.506 | 122.681 |
| 31 | QSA | 33 | 28 | 6 | 305.217 | 29.565 | 48892416 | 65.190 | 240.028 |
| 32 | GDN | 25 | 20 | 6 | 184.377 | 26.357 | 48379208 | 64.506 | 119.872 |
| 33 | GDN | 26 | 21 | 6 | 183.636 | 30.888 | 48379208 | 64.506 | 119.131 |
| 34 | GDN | 27 | 22 | 6 | 187.435 | 27.864 | 48379208 | 64.506 | 122.929 |
| 35 | QSA | 33 | 28 | 6 | 305.040 | 29.080 | 48892416 | 65.190 | 239.850 |
| 36 | GDN | 25 | 20 | 6 | 184.612 | 27.291 | 48379208 | 64.506 | 120.107 |
| 37 | GDN | 26 | 21 | 6 | 181.953 | 29.035 | 48379208 | 64.506 | 117.448 |
| 38 | GDN | 27 | 22 | 6 | 188.111 | 29.640 | 48379208 | 64.506 | 123.606 |
| 39 | QSA | 33 | 28 | 6 | 304.606 | 29.022 | 48892416 | 65.190 | 239.417 |
| 40 | GDN | 25 | 20 | 6 | 184.107 | 28.792 | 48379208 | 64.506 | 119.601 |
| 41 | GDN | 26 | 21 | 6 | 181.860 | 29.196 | 48379208 | 64.506 | 117.355 |
| 42 | GDN | 27 | 22 | 6 | 187.240 | 29.675 | 48379208 | 64.506 | 122.734 |
| 43 | QSA | 33 | 28 | 6 | 303.437 | 29.862 | 48892416 | 65.190 | 238.247 |
| 44 | GDN | 25 | 20 | 6 | 184.375 | 31.547 | 48379208 | 64.506 | 119.870 |
| 45 | GDN | 26 | 21 | 6 | 183.304 | 27.715 | 48379208 | 64.506 | 118.798 |
| 46 | GDN | 27 | 22 | 6 | 186.579 | 27.020 | 48379208 | 64.506 | 122.073 |
| 47 | QSA | 33 | 27 | 5 | 302.994 | 28.377 | 48892416 | 65.190 | 237.804 |
| final_mixer | final_mixer | 6 | 0 | 0 | 46.843 | 0.000 | 13127680 | 17.504 | 29.339 |

## Stream 364 kernel priority

Sorted by service minus weight floor. Template rank instances remain separate
in this table; aggregate their service before estimating a fused collective.

| Kernel | Grid | Calls/step | Weight B/call | Floor us/call | Measured us/call | Excess us/step |
|---|---|---:|---:|---:|---:|---:|
| `vllm::sm70_qwen38_hc_up_mix_push` | [160, 1, 1] | 94 | 1638400 | 2.185 | 7.712 | 519.576 |
| `_qsa_sparse_paged_gqa_splitk_kernel` | [1, 1, 64] | 12 | 0 | 0.000 | 38.521 | 462.251 |
| `void <unnamed>::nvfp4_qpn_m1_sm70_kernel<` | [10, 10, 1] | 48 | 5120000 | 6.827 | 15.099 | 397.071 |
| `_sm70_qwen38_router_topk_kernel` | [1, 1, 1] | 48 | 0 | 0.000 | 8.000 | 384.019 |
| `_hc_combine_norm_kernel` | [1, 4, 1] | 93 | 20480 | 0.027 | 3.820 | 352.684 |
| `void vllm::sm70_cross_device_reduce_1stage_push<` | [3, 1, 1] | 47 | 0 | 0.000 | 7.105 | 333.941 |
| `void vllm::sm70_cross_device_reduce_sum2_1stage_push<` | [3, 1, 1] | 47 | 0 | 0.000 | 6.750 | 317.240 |
| `_qwen38_fp16_row_gemv_kernel` | [512, 1, 1] | 48 | 2621440 | 3.495 | 9.914 | 308.113 |
| `<unnamed>::nvfp4_qwen38_w2_direct_reduce_kernel` | [80, 1, 1] | 48 | 2560000 | 3.413 | 9.819 | 307.455 |
| `_fp16_gemv_silu_ranges_kernel` | [1, 88, 1] | 94 | 1658880 | 2.212 | 5.328 | 292.952 |
| `void <unnamed>::gdn_decode_mixed_qkv_global_state_kernel<c10::Half, c10::Half, float,` | [12, 1, 16] | 35 | 72 | 0.000 | 8.077 | 282.696 |
| `_qsa_mqa_paged_kernel` | [1, 2052, 1] | 12 | 0 | 0.000 | 22.028 | 264.336 |
| `void vllm::qsa::qsa_lexicographic_decode_topk_kernel<` | [1, 1, 1] | 12 | 0 | 0.000 | 16.457 | 197.484 |
| `void vllm::sm70_qwen38_hc_down_push_allgather<` | [1, 1, 1] | 23.5 | 0 | 0.000 | 6.005 | 141.109 |
| `_causal_conv1d_update_kernel` | [1, 10, 1] | 35 | 20480 | 0.027 | 3.767 | 130.895 |
| `void vllm::sm70_qwen38_hc_down_push_allgather<` | [1, 1, 1] | 23.5 | 0 | 0.000 | 5.482 | 128.826 |
| `_qsa_merge_splitk_kernel` | [1, 6, 1] | 12 | 0 | 0.000 | 10.280 | 123.359 |
| `void <unnamed>::rmsnorm_gated_exact_kernel<` | [3, 1, 1] | 35 | 256 | 0.000 | 3.341 | 116.925 |
| `void vllm::sm70_qwen38_hc_down_push_allgather<` | [1, 1, 1] | 23.5 | 0 | 0.000 | 4.588 | 107.813 |
| `_qwen38_fp16_row_gemv_kernel` | [2560, 1, 1] | 47 | 7864320 | 10.486 | 12.570 | 97.953 |
| `void vllm::sm70_qwen38_hc_down_push_allgather<` | [1, 1, 1] | 23.5 | 0 | 0.000 | 4.018 | 94.425 |
| `void vllm::reshape_and_cache_flash_kernel<unsigned short, unsigned short,` | [1, 1, 1] | 12 | 0 | 0.000 | 5.784 | 69.409 |
| `_qsa_pre_indexer_kernel` | [3, 1, 1] | 12 | 512 | 0.001 | 5.572 | 66.858 |
| `_qwen38_fp16_gdn_input_kernel` | [4120, 1, 1] | 35 | 21094400 | 28.126 | 29.502 | 48.162 |
| `triton_poi_fused_zeros_0` | [12, 1, 1] | 21.5 | 0 | 0.000 | 1.997 | 42.926 |
| `_expand_qsa_indices_kernel` | [1, 9, 1] | 12 | 0 | 0.000 | 2.973 | 35.673 |
| `_qwen38_fp16_row_gemv_kernel` | [640, 1, 1] | 12 | 3276800 | 4.369 | 7.167 | 33.574 |
| `void at::native::vectorized_elementwise_kernel<` | [2, 1, 1] | 12 | 0 | 0.000 | 2.648 | 31.779 |
| `triton_per_fused_2` | [7, 1, 1] | 12 | 0 | 0.000 | 2.588 | 31.053 |
| `triton_poi_fused_copy__1` | [20, 1, 1] | 12.75 | 0 | 0.000 | 2.408 | 30.702 |
| `_qwen38_fp16_row_gemv_kernel` | [3584, 1, 1] | 12 | 18350080 | 24.467 | 26.893 | 29.111 |
| `triton_poi_fused_3` | [15, 1, 1] | 7.75 | 1024 | 0.001 | 3.757 | 29.107 |
| `triton_poi_fused_zeros_0` | [6, 1, 1] | 12.5 | 0 | 0.000 | 2.004 | 25.049 |
| `triton_poi_fused_copy__1` | [10, 1, 1] | 10.25 | 0 | 0.000 | 2.430 | 24.908 |
| `triton_poi_fused_3` | [9, 1, 1] | 4.25 | 1024 | 0.001 | 4.247 | 18.042 |
| `triton_poi_fused_1` | [6, 1, 1] | 6.5 | 0 | 0.000 | 2.303 | 14.966 |
| `void cutlass::Kernel2<cutlass_70_wmma_tensorop_f16_s161616gemm_f16_16x16_64x2_tn_align8>` | [8, 80, 1] | 1 | 52428800 | 69.905 | 84.623 | 14.718 |
| `triton_poi_fused_copy__0` | [10, 1, 1] | 6.25 | 0 | 0.000 | 2.190 | 13.685 |
| `triton_poi_fused_1` | [12, 1, 1] | 5.5 | 0 | 0.000 | 2.307 | 12.686 |
| `triton_poi_fused_copy__0` | [20, 1, 1] | 5.75 | 0 | 0.000 | 2.201 | 12.656 |
| `triton_red_fused__to_copy_abs_add_clamp_min_div_mean_mul_pow_rsqrt_sigmoid_sign_sqrt_sum_unsqueeze_view_0` | [4, 1, 1] | 1 | 40960 | 0.055 | 6.781 | 6.726 |
| `void gemv2T_kernel_val<int, int, __half, __half, __half, float,` | [320, 1, 1] | 1 | 13107200 | 17.476 | 23.694 | 6.218 |
| `_hc_combine_kernel` | [1, 5, 1] | 1 | 0 | 0.000 | 5.872 | 5.872 |
| `void at::native::vectorized_elementwise_kernel<` | [10, 1, 1] | 1 | 0 | 0.000 | 4.389 | 4.389 |
| `_grouped_gemma_rmsnorm_kernel` | [4, 1, 1] | 1 | 20480 | 0.027 | 4.237 | 4.209 |
| `_dequantize_ple_fp8_bytes_kernel` | [3, 1, 1] | 1 | 0 | 0.000 | 4.170 | 4.170 |
| `_qwen38_ple_m1_short_conv_kernel` | [40, 1, 1] | 1 | 81920 | 0.109 | 3.366 | 3.257 |
| `triton_poi_fused_copy__3` | [10, 1, 1] | 0.75 | 0 | 0.000 | 2.391 | 1.794 |
| `triton_poi_fused_0` | [46, 1, 1] | 0.75 | 0 | 0.000 | 2.332 | 1.749 |
| `triton_poi_fused_zeros_like_2` | [80, 1, 1] | 0.75 | 0 | 0.000 | 2.197 | 1.648 |
| `triton_poi_fused__to_copy_add_mean_mul_pow_rsqrt_view_1` | [40, 1, 1] | 0.5 | 20480 | 0.027 | 2.627 | 1.300 |
| `triton_poi_fused__to_copy_add_mean_mul_pow_rsqrt_view_1` | [80, 1, 1] | 0.5 | 20480 | 0.027 | 2.490 | 1.231 |
| `triton_poi_fused_0` | [92, 1, 1] | 0.25 | 0 | 0.000 | 2.496 | 0.624 |
| `triton_poi_fused_copy__3` | [20, 1, 1] | 0.25 | 0 | 0.000 | 2.406 | 0.602 |
| `triton_poi_fused_zeros_like_2` | [40, 1, 1] | 0.25 | 0 | 0.000 | 1.952 | 0.488 |
