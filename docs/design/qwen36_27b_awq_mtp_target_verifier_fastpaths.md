# Qwen3.6-27B-AWQ MTP Target Verifier Fast-Path Audit

Date: 2026-07-10

This document separates exact optimization work that remains possible on
V100/SM70 from fast paths that require Ampere, Hopper, or Blackwell. The scope
is the quality-safe native-MTP4 route: Qwen3.6-27B-AWQ, TP2, `M=5`, CUDA graph,
non-eager execution, TurboMind AWQ, Flash-V100, and official sampling.
Marlin is out of scope. CUTLASS or FlashInfer candidates below mean importing
or wrapping the kernel under the existing TurboMind/vLLM target route, not
changing the accepted quantization backend to Marlin.

## Accepted Baseline

The calibrated target verifier costs are:

| scope | ms/MTP round |
|---|---:|
| target forward | `22.997` |
| target full-vocabulary logits | `2.038` |
| rejection and official sampling | `1.116` |
| complete verifier | `26.151` |

The target-forward composition is:

| category | critical ms/round | forward share |
|---|---:|---:|
| TurboMind AWQ GEMM | `10.181` | `44.27%` |
| TP custom all-reduce | `2.083` | `9.06%` |
| copy/cast | `1.776` | `7.72%` |
| GDN recurrent core plus causal conv | `1.595` | `6.94%` |
| TurboMind FP16 GEMM | `1.462` | `6.36%` |
| other Triton/elementwise | `1.447` | `6.29%` |
| CUTLASS/cuBLAS FP16 GEMM/GEMV | `1.192` | `5.18%` |
| index/reduce/scatter | `1.160` | `5.04%` |
| FlashAttn V100 verifier attention | `0.967` | `4.20%` |
| RMSNorm/residual | `0.603` | `2.62%` |
| SiLU/gating | `0.456` | `1.98%` |
| KV-cache update | `0.076` | `0.33%` |

Artifacts:

- `bench_results/nsys_27b_awq_mtp_default_dynamic_20260710/mtp_round_kernel_breakdown.md`
- `bench_results/nsys_27b_awq_mtp_default_dynamic_20260710/graph_node_i512_o16.nsys-rep`
- `bench_results/ddtree_realcode_20260625/ncu_awq_uint4_step5_deep_sudo.ncu-rep`
- `bench_results/ddtree_realcode_20260625/awq_gemm_deep_bottleneck_20260630.md`

## NCU Verdict For The AWQ Bucket

Do not apply the historical `192 registers/thread` result to this verifier.
That report sampled a `M=17`, `32x256x32` kernel. The current fixed-dispatch
`M=5` gate/up kernel is `8x256x64`, `TG=1x4x1`, split `1`, swizzle `0` and
uses `159 registers/thread`.

The same-invocation NCU comparison for the current gate/up shape is:

| metric | fixed `TG=1x4x1` | CTA-local `TG_K=2` |
|---|---:|---:|
| NCU duration | `98.976 us` | `89.504 us` |
| threads/CTA | `128` | `256` |
| grid CTAs | `68` | `68` |
| registers/thread | `159` | `120` |
| theoretical occupancy | `18.75%` | `25.00%` |
| achieved occupancy | `6.25%` | `12.50%` |
| active warps/SM | `4.00` | `8.00` |
| eligible warps/scheduler | `0.343` | `0.637` |
| issue active | `34.30%` | `41.26%` |
| no-eligible cycles | `65.70%` | `58.74%` |

The earlier full-bucket profile still establishes that the AWQ work is not
DRAM- or Tensor-Core-saturated, but its `192`-register launch must remain
historical evidence only. The current limitation is small-`M` parallelism and
latency hiding: gate/up launches only 68 CTAs on 72 SMs, and one 4-warp CTA per
active SM does not expose enough ready work.

## Required Delta

Reducing target forward from `22.997` to `20.000 ms` needs `2.997 ms`.
If every other bucket remained unchanged, AWQ GEMM would have to fall from
`10.181` to `7.184 ms`, a `29.44%` reduction or `1.417x` speedup.

Sensitivity, not a hardware prediction:

| AWQ-only speedup | resulting target forward |
|---:|---:|
| `1.417x` | `20.000 ms` |
| `1.50x` | `19.604 ms` |
| `1.70x` | `18.805 ms` |
| `2.00x` | `17.907 ms` |

The complete-verifier `<=20 ms` target is harder. Keeping logits and sampling
unchanged requires target forward `<=16.846 ms`. If AWQ were the only changed
bucket, it would need to reach `4.030 ms`, or `2.526x`; that is not a credible
SM70 kernel-tuning-only target.

## 2026-07-11 CTA-Local K-Parallel Prototype

The exact TP-rank shape contract is 236 AWQ calls, not the older 283-call or
unsharded checkpoint approximation:

| projection | `M x N x K` per TP rank | calls/rank | fixed split/swizzle |
|---|---:|---:|---:|
| merged MLP gate/up | `5x17408x5120` | `63` | `1/0` |
| MLP down | `5x5120x8704` | `63` | `7/0` |
| merged linear-attention qkv/z | `5x8192x5120` | `47` | `2/0` |
| linear-attention out | `5x5120x3072` | `47` | `3/0` |
| full-attention o | `5x5120x3072` | `16` | `3/0` |

`benchmark_sm70_awq_verifier_micro.py --all-real-layers` loads every actual
layer weight and executes these calls in model layer order. It avoids the
previous representative-layer cache approximation while remaining a
kernel-only benchmark.

The new registry prototype keeps CTA `8x256x64`, HMMA, packed AWQ layout and
the outer split-K count, but changes the thread-group map from `1x4x1` to
`1x4x2`. Two warp groups partition the CTA K loop and `Rearrange` combines the
two FP32 partial accumulators in shared memory. This is an intra-CTA reduction;
it does not add another global split-K workspace or kernel launch.

The best measured mix is:

| shape | selected implementation |
|---|---|
| gate/up | stock `TG=1x4x1`, split `3`, swizzle `3` |
| down | new `TG=1x4x2`, original split `7`, swizzle `0` |
| qkv/z | new `TG=1x4x2`, original split `2`, swizzle `0` |
| both out projections | new `TG=1x4x2`, original split `3`, swizzle `0` |

Five same-GPU, independent-process ABBA rounds over all 236 real weights:

| round | fixed baseline | candidate | saved | speedup |
|---:|---:|---:|---:|---:|
| 1 | `11.3290 ms` | `9.4878 ms` | `1.8412 ms` | `16.25%` |
| 2 | `11.3381 ms` | `9.4930 ms` | `1.8451 ms` | `16.27%` |
| 3 | `11.3402 ms` | `9.4946 ms` | `1.8456 ms` | `16.27%` |
| 4 | `11.3434 ms` | `9.4848 ms` | `1.8585 ms` | `16.38%` |
| 5 | `11.3321 ms` | `9.4997 ms` | `1.8324 ms` | `16.17%` |
| **mean** | **`11.3366 ms`** | **`9.4920 ms`** | **`1.8446 ms`** | **`16.27%`** |

The separate-rank runs agree: rank0 `11.3442 -> 9.5029 ms`, rank1
`11.3423 -> 9.4860 ms`. Against the fixed output tensors, all 236 layers have
small numerical differences. Rank0 is `99.3499%` bit-exact with maximum
absolute difference `0.001953125` and 5 near-zero sign flips; rank1 is
`99.3469%` bit-exact with maximum difference `0.0009765625` and no sign flips.
This passes the component speed gate but not the release quality gate by
itself.

Artifacts are under
`bench_results/awq_m5_structural_20260710/warp8_candidate/`.

The 256K, full-GDN, non-eager, official-sampling `macos_6k_code` A/B rejects
the best component mix even though both 6000-token outputs passed the text
health gate. With the same current binary, GPUs, prompt, sampling seed, and
dynamic draft vocabulary, the fixed-dispatch baseline measured mean
acceptance length `3.96` and steady decode `94.014 tok/s`; the best mix
measured `3.61` and `88.381 tok/s`. The `8.84%` acceptance-length loss exceeds
the agreed `2%` limit and erases the operator-speed gain at request level.
Artifacts:

- baseline:
  `bench_results/awq_m5_structural_20260710/fullgdn_fixed_baseline_macos6000_seed20260620/`;
- rejected best mix:
  `bench_results/awq_m5_structural_20260710/fullgdn_best_mix_macos6000_seed20260620_retry/`.

Do not promote gate/up `split3/swizzle3`. The next quality/profile gate keeps
every original split/swizzle and changes only the CTA thread-group layout to
`TG=1x4x2`; it completed as user unit
`vllm-awq-m5-tgk2-native-quality-profile-20260711.service`. The profiled
6000-token run passed text-health checks and reported mean acceptance length
`4.02`, but the later unprofiled gate shows that this synchronized trajectory
is not release acceptance evidence. Its all-real-weight same-GPU microbenchmark measured
`11.3565 -> 10.8536 ms`, saving `0.5029 ms` or `4.43%`; `99.341%` of outputs
were bit-exact, maximum absolute difference was `0.001953125`, and there were
10 near-zero sign flips. This is a smaller but cleaner candidate, not a claim
that the rejected `1.8446 ms` best-mix delta remains available.

The request-level throughput from that run is not an A/B speed result because
the candidate enabled per-step CUDA event profiling and the fixed quality run
did not. The completed fixed-dispatch profile control gives the aligned target
forward comparison:

| profiler calls | fixed dispatch | original-split `TG_K=2` | saved |
|---:|---:|---:|---:|
| 128 | `24.206 ms` | `23.467 ms` | `0.739 ms` |
| 256 | `24.506 ms` | `23.691 ms` | `0.815 ms` |
| 384 | `25.384 ms` | `24.685 ms` | `0.699 ms` |
| 512 | `25.483 ms` | `24.793 ms` | `0.690 ms` |
| **mean** | **`24.895 ms`** | **`24.159 ms`** | **`0.736 ms` (2.96%)** |

The values rise with generated context length, so only aligned call intervals
are compared. Artifacts are
`fullgdn_fixed_profile_macos2048_seed20260620/` and
`fullgdn_tgk2_native_profile_macos6000_seed20260620/` under the same
`awq_m5_structural_20260710` directory.

The unprofiled four-prompt gate rejects this candidate for production despite
all four generated texts passing the health checks. Mean acceptance length
was only `3.32`; the macOS record measured `88.529 tok/s`, below the exact
unprofiled fixed-dispatch macOS control at `94.014 tok/s`. In the macOS phase,
the six complete 10-second windows before the next-request boundary have a
weighted acceptance length of about `3.65`, versus `3.96` in the fixed
single-prompt control. The earlier profiled `4.02` result is not an acceptance
gate: synchronizing CUDA timing events every step changed the asynchronous
sampling trajectory. Artifact:
`bench_results/awq_m5_structural_20260710/fullgdn_tgk2_native_quality4_seed20260620/`.

Decision: keep `TG_K=2` as opt-in component/graph evidence, but do not
default-enable it and do not count its `0.736 ms` target-forward delta toward
accepted performance. Both the split3 best mix and the original-split
candidate fail the agreed acceptance/throughput requirement.

### Closed Prototype Branches

- N-direction `TG=1x8x1` lowered registers to `115` and doubled achieved
  occupancy, but NCU duration stayed `99.328 us`; duplicated per-warp dataflow
  erased the latency-hiding gain. Forcing it across all shapes raised the
  representative aggregate to about `20.6 ms`.
- `TG_K=4` added two more shared reductions/barriers and regressed gate/up to
  about `88-92 us`; it was removed from the registry.
- A compile-time plain/no-tile-allreduce specialization is closed by its own
  precondition. Current M=5 already has `159 registers/thread` and a register
  block limit of 3, so removing the inactive M=1 tile-allreduce branch cannot
  cross the expected residency threshold.
- Uniform `8x128x32` and uniform `8x256x64` forcing remain rejected. Per-shape
  split and tile contracts must be preserved.

## 2026-07-11 SM70 TP2 All-Reduce + Gemma RMSNorm Prototype

This is the first structural candidate after the AWQ `TG_K=2` acceptance
rejection. It keeps the existing TP2 peer-read order and computes
`AR(local_projection) + residual`; it must not use `all_reduce_sum2`, which
would reduce the replicated residual twice. The same kernel writes the exact
FP32 residual, performs the production Gemma RMSNorm reduction for hidden size
5120, applies `1 + weight`, and writes the FP16 normalized output.

The isolated op is
`_C_custom_ar::sm70_tp2_all_reduce_gemma_rms_norm`. Its implementation and
model-free harness are in `csrc/custom_all_reduce.{cu,cuh}` and
`benchmarks/benchmark_sm70_custom_all_reduce.py`. It is not wired into the
model or enabled by default.

The first run compared against the generic/native RMSNorm oracle and found one
FP16 element different by one ULP. That was an oracle error, not accepted
numeric slack: the production SM70 graph uses the `vllm_c` RMSNorm kernel.
After changing the benchmark baseline to the exact production sequence
`custom AR -> FP32 residual add -> vllm_c RMSNorm -> FP16 cast`, both ranks
match bitwise for normalized output and FP32 residual.

| CUDA graph scope | baseline | fused candidate | saved | speedup | exactness |
|---|---:|---:|---:|---:|---|
| one `[5,5120]` join | `0.029766 ms` | `0.023492 ms` | `0.006274 ms` | `1.267x` | both outputs bit-exact |
| 80 graph-visible joins | `2.186214 ms` | `1.453199 ms` | `0.733015 ms` | `1.504x` | both outputs bit-exact |

Artifacts:

- corrected one-join oracle:
  `bench_results/awq_m5_structural_20260710/allreduce_rmsnorm_prototype_tp2_gpu01_vllmc_oracle.json`;
- 80-join CUDA graph:
  `bench_results/awq_m5_structural_20260710/allreduce_rmsnorm_prototype_80joins_tp2_gpu01.json`.

The 80 joins model the 64 MLP-down boundaries and 16 full-attention output
boundaries visible to the compiled graph. The 48 GDN attention boundaries are
inside the full-GDN opaque op and remain out of scope; splitting that op would
reopen the known recurrent-state quality failure. The `0.733 ms` result clears
the component `>=0.40 ms` gate, but it is not yet an accepted target-forward
delta. Next gates are explicit compile-matcher route hit, an observed count of
80 logical joins, and same-condition full-graph A/B with the feature default
off.

## Architecture Fast-Path Matrix

| fast path | minimum useful architecture | affected current bucket | current AWQ checkpoint reusable | V100 status |
|---|---|---|---|---|
| TurboMind async-copy multistage W4A16 mainloop, HMMA `16x8x16` | SM80/A100 | AWQ `10.181`, part of FP16 `2.654` | likely yes after registry/layout validation; no retraining | impossible: SM70 has no `cp.async` or larger HMMA atom |
| CUTLASS mixed-input FP16/BF16 x INT4 with fast converters, group scales, TMA/WGMMA | SM90/H100 | AWQ `10.181` | mathematical route fits W4A16; AWQ zero-point/layout adapter must be proved | impossible: no TMA/WGMMA |
| FlashInfer `gated_delta_rule_mtp` | SM90/H100 | GDN `1.595` and some state copy/index work | weights yes; state layout/update semantics require exact adapter | explicitly requires SM90 |
| FlashInfer all-reduce + RMSNorm fusion with PDL | SM90/H100, SM100 | TP `2.083` plus RMS `0.603` | yes | local vLLM gate explicitly excludes SM70/SM80 |
| XQA/TRTLLM-gen verifier attention with MTP query length | SM90/SM100+ | attention `0.967` | attention weights yes; KV layout adapter needed | unsupported; even a perfect replacement saves less than `0.967 ms` |
| W4A8 AWQ / FP8 activation route | Ada/Hopper/Blackwell in current TRT-LLM matrix | AWQ and FP16 projections | requires activation quantization/calibration and quality validation | unsupported by the current backend matrix |
| native NVFP4 block-scaled TCGen05/TMEM GEMM and FP4 epilogues | SM100/Blackwell | most dense GEMM and surrounding materialization | no: AWQ INT4 is not NVFP4 E2M1; create a new checkpoint | impossible |

## 2026-07-11 SM90/SM100 AWQ GEMM Source Audit

This audit looks specifically for a better implementation of the fixed
`M=5` target-verifier linear algebra. It does not propose switching the
accepted route to Marlin. External kernels are references to import below the
TurboMind dispatch boundary or to reproduce inside the TurboMind kernel
library.

### Raw traffic floor

The 236 calls per TP rank contain `5,692,456,960` bytes (`5.692 GB`) of raw
INT4 weights before scales, zero-points, activation traffic, and output
traffic:

| projection family | raw W4 bytes/rank/round |
|---|---:|
| merged MLP gate/up | `2.808 GB` |
| MLP down | `1.404 GB` |
| linear-attention qkv/z | `0.986 GB` |
| all out projections | `0.495 GB` |

At the calibrated `10.181 ms`, raw-weight traffic alone corresponds to about
`559 GB/s`. Reaching `7.184 ms` would require about `792 GB/s` before adding
scale/zero and other traffic. This does not contradict the NCU scheduler
result: the current kernel leaves bandwidth unused because it exposes too
little parallel work. It does show that the final SM70 target is close to a
weight-streaming limit, so a candidate must improve parallelism without
duplicating weight reads.

### P0: exact-M CUDA-core batched GEMV

Implementation and measured SM70 results are maintained separately in
[`sm70_awq_exact_m5_batched_gemv.md`](sm70_awq_exact_m5_batched_gemv.md).

Current TensorRT-LLM contains a separate weight-only batched-GEMV family for
small `M`. This is not a generic GEMM tile adjustment:

- its dispatcher instantiates every exact `M` from 1 through 15;
- for asymmetric groupwise W4A16 (`zeros != nullptr`), `M=5` uses
  `CtaM=5`, `CtaN=4`, and 128 threads;
- on the SM70/Turing K64 interleave-4 layout, one CTA emits 16 physical
  columns, so gate/up exposes `17408 / 16 = 1088` CTAs rather than the current
  TurboMind gate/up kernel's 68 CTAs; the 4352-CTA count applies to the
  interleave-1 column-major variant;
- each CTA loads a weight slice once and reuses it across all five rows, so
  this is not five independent GEMVs and does not multiply weight traffic by
  five;
- TensorRT-LLM profiles this CUDA-core candidate against its CUTLASS GEMM
  tactics and only admits the CUDA candidate for `M < 16`;
- the launcher has separate Hopper interleaved and SM100 column-major layouts,
  and explicitly dispatches both W4A16 and W4A8 variants on SM90 and SM100.

This is the most directly aligned external design for our current
`M=5` scheduler-starvation evidence. It is also a numerical-risk candidate:
the current TensorRT-LLM implementation stores FP16/BF16 partial accumulators
and uses `__hfma2`/BF16 `__hfma2`, then performs the cross-warp reduction in
FP32. It therefore does not preserve the current TurboMind accumulation order
or precision by construction. The upstream launcher starts at SM75, so an
SM70 version is a port/redesign, not an existing disabled path.

Required experiment:

1. Reproduce the exact TensorRT layout and half2 dataflow in a model-free
   `M=5` microbenchmark to establish the performance ceiling.
2. Compare all 236 real-weight outputs, not only random matrices.
3. If speed clears the gate but output drift is material, add an FP32-partial
   variant and measure the cost before any full-model quality run.
4. Do not promote either variant unless the official long-output acceptance
   loss remains within 2%.

Primary source paths:

- https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/weightOnlyBatchedGemv/kernelDispatcher.h
- https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/weightOnlyBatchedGemv/kernelLauncher.h
- https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/weightOnlyBatchedGemv/kernel.h
- https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/weightOnlyBatchedGemv/utility.h
- https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/plugins/weightOnlyGroupwiseQuantMatmulPlugin/weightOnlyGroupwiseQuantMatmulPlugin.cpp

### P1: SM90 Machete-style mixed-input WGMMA

The local source tree already contains the strongest direct SM90 W4A16
reference. Machete is a CUTLASS mixed-input kernel, not a Marlin route. Its
mainloop:

- prepacks W4 weights to the WGMMA fragment layout so multiple narrow loads
  become one 128-bit shared-memory load;
- transposes `Y = XW^T` into `Y^T = W^T X^T`, making the quantized weights the
  register-sourced WGMMA operand;
- overlaps TMA movement, register dequantization, and WGMMA through warp
  specialization;
- uses Stream-K, and the local H100-tuned heuristic selects a `128x16` tile
  with cluster `1x1x1` for `M=1..16` at all our `N` values;
- supports AWQ `uint4`, group size 128, zero-points, FP16 activations, and FP32
  accumulation. Every current per-rank dimension satisfies its 64-by-128
  prepack divisibility contract.

This is the best SM90 checkpoint-compatible Tensor-Core candidate. It should
be raced shape-by-shape against the exact-M CUDA-core GEMV, not assumed to win
because it uses WGMMA. Tiny-`M` decode remains dominated by weight streaming,
launch geometry, and latency hiding. Machete is gated to compute capability 90
and cannot be executed on the local V100 or SM89 device.

Local references:

- `csrc/quantization/machete/machete_mainloop.cuh`
- `csrc/quantization/machete/generate.py`
- `vllm/model_executor/kernels/linear/mixed_precision/machete.py`
- `vllm/model_executor/layers/quantization/utils/machete_utils.py`

### P2: SM100 has two distinct routes

**Checkpoint-compatible W4A16.** CUTLASS now has an SM100 mixed-input INT4 x
BF16 example with group scales and zero-points. It explicitly uses a
`KernelTmaWarpSpecialized2SmMixedInputSm100` mainloop, a two-SM cluster, and a
TMA epilogue. TMEM moves accumulator residency out of the ordinary register
file and the 2-SM schedule provides a new way to distribute a tile. This can
consume AWQ-style integer weights after a layout adapter, but the published
example's `256x128x128` tile is not an `M=5` tuning result. A production route
must profile 1-SM, 2-SM, Stream-K, and exact-M GEMV tactics for each of the
four real shapes.

Source:
https://github.com/NVIDIA/cutlass/blob/main/examples/86_blackwell_mixed_dtype_gemm/86_blackwell_mixed_dtype.cu

**Native NVFP4/MXFP4.** Blackwell's `tcgen05.mma.blockscaled` consumes native
block-scaled FP4 and accumulates through TMEM. CUTLASS documents 2x instruction
throughput versus Blackwell FP8 MMA and 4x versus Hopper FP8 WGMMA, and also
ships a dedicated block-scaled FP4 GEMV example. Those are instruction
throughput ratios, not an expected `M=5` wall-time ratio: the same 4-bit weight
volume must still be streamed. More importantly, AWQ affine UINT4 values are
not NVFP4 E2M1 values. This route requires a separately quantized checkpoint
and a separate quality baseline; repacking the current AWQ checkpoint is not
valid conversion.

Sources:

- https://github.com/NVIDIA/cutlass/blob/main/examples/72_blackwell_narrow_precision_gemm/72a_blackwell_nvfp4_bf16_gemm.cu
- https://github.com/NVIDIA/cutlass/blob/main/examples/91_fp4_gemv/91_fp4_gemv.cu
- https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/llm_ptq/README.md

### TurboMind SM90 ceiling

Upstream TurboMind does register `sm90_16816_{4,8,16}` kernels, so porting the
current upstream registry is still a useful baseline. However,
`sm90_16816_4.cu` imports `config_sm80_s16816.h` and registers the 16816
family. It is not equivalent to the Machete TMA/WGMMA register-source
mainloop. Therefore, "enable TurboMind SM90" and "exploit Hopper's full mixed
input path" are separate milestones.

Source:
https://github.com/InternLM/lmdeploy/blob/main/src/turbomind/kernels/gemm/kernel/sm90_16816_4.cu

### Ranked benchmark order

| order | candidate | checkpoint | first decision gate |
|---:|---|---|---|
| 1 | exact `M=5` CUDA-core batched GEMV, half2 then FP32-partial variant | current AWQ | beat each current real shape and save at least 15% over the 236-call micro; then pursue the `1.417x` final gate |
| 2 | SM90 Machete WGMMA/TMA/Stream-K versus exact-M GEMV | current AWQ | FP32 output comparison first; aggregate AWQ ratio, not a representative GEMM |
| 3 | SM100 1-SM/2-SM mixed-input W4A16 versus exact-M GEMV | current AWQ | per-shape tactic winner plus all-236 aggregate |
| 4 | SM90/SM100 W4A8 AWQ | newly calibrated activation route | official ModelOpt marks W4A8 experimental and Qwen3 support is not established; full quality gate required |
| 5 | SM100 native NVFP4/MXFP4 | new checkpoint | separate accuracy baseline; never report as an exact AWQ kernel win |

There is no accepted speedup number for these candidates on Qwen3.6-27B
`M=5` yet. Machete's published serving gains use different models and request
rates, and Blackwell's FP4 ratios describe Tensor-Core instruction throughput.
The next valid number must come from the existing all-real-weight 236-call
microbenchmark on the target architecture.

The first row is not hypothetical source work. This tree already contains
`config_sm80_s16816.h`, `MainloopSm80_v2`, `IteratorSm80`, CUDA pipeline
commit/wait calls, and packed `uint4_t` operand transforms. The current local
kernel registry contains only `sm70_884_{4,8,16}.cu`, so the SM80 implementation
cannot be selected by the V100 build. Current upstream TurboMind already ships
`sm80_16816_{4,8,16}.cu` and `sm90_16816_{4,8,16}.cu`; the first implementation
step is therefore a controlled upstream registry/build port, not a new GEMM
design.

Ampere native INT4 IMMA must not be confused with the current AWQ W4A16
operation. IMMA consumes integer activations and accumulates to INT32. The
current TurboMind path consumes FP16 activations, converts packed W4 weights,
and executes FP16 HMMA. The directly relevant Ampere advantages are async
global-to-shared copies, lower copy-register demand, the wider HMMA atom,
larger cache/shared memory, and higher NVLink/HBM capability. A native INT4
route would instead be W4A4/W4A8 and changes the numerical contract.

## V100 Work That Still Has A Path

These candidates do not require a new GPU, but their savings overlap and must
not be added before a graph trace proves disjoint critical-path deltas.

| priority | exact candidate | buckets exposed | first go/no-go gate |
|---|---|---:|---|
| P0 | SM70 custom all-reduce + residual + RMSNorm in one peer-read kernel | `2.083 + 0.603 ms` | save `>=0.40 ms/round`, preserve reduction order, exact isolated output |
| P0 | exact full-GDN `M=5` kernel that fuses causal-conv, recurrent update, gating, and indexed state write | `1.595 ms` plus adjacent copy/index | save `>=0.60 ms/round`; output and every committed state slot must match before a long quality run |
| P1 | fuse GDN boundary casts/scatters and remove repeated temporary materialization | copy/index/elementwise pool of `4.383 ms` | save `>=0.75 ms/round` in a graph-node microbenchmark |
| P1 | port the exact-`M=5` TensorRT-style batched W4A16 GEMV, first half2 then FP32 partials | AWQ `10.181 ms` | all-236 micro saves `>=15%`, no shape regresses, then pass the 2% acceptance-loss gate |
| P1 | lower-register, no-splitK small-`M` AWQ schedule or persistent intra-GEMM tile scheduler | AWQ `10.181 ms` | aggregate 236-call/rank suite saves `>=1.0 ms`; preserve the accepted accumulation path |
| P2 | pack compatible FP16 projections that share the same input and fuse their epilogues | FP16 `2.654 ms` | save `>=0.30 ms/round` with projection-output parity |

The unsafe deep-MTP split-GDN path is not a candidate. It changed state
semantics and caused long-output loops. Fixed AWQ split/swizzle points are also
closed unless a new numerical design appears: the faster split-K gate/up
points failed the official long-output quality gate, and split-1 had negligible
speed value.

The exact-`M=5` batched-GEMV candidate is not a repeat of the rejected raw
`M=1` NVFP4 GEMV. It targets AWQ, reuses each W4 value across five activation
rows, uses an offline interleaved conversion layout, and creates thousands of
output-column CTAs. The prior result remains a warning that an unstructured
CUDA-core GEMV is insufficient, not evidence against this different dataflow.

The current generic custom all-reduce is active. The missing operation is
fusion with the consumer. This tree's upstream FlashInfer fusion is gated to
SM90/SM100, but an SM70-specific peer-read reduction can still compute the
residual and RMS statistics before writing the normalized result. That is a
new custom kernel, not a flag that is currently disabled by mistake.

## New-GPU Microbenchmark Order

Do not start with a full-model launch.

The local machine has four SM70 devices and one 8 GB SM89 RTX 4060 Ti. The
SM89 device can prove that a ported SM80-family operator registers, executes,
and matches projection outputs on synthetic real shapes. Its memory capacity
cannot host the TP2 27B route, and its result must not be reported as an A100
or H100 target-forward baseline.

1. **A100/SM80, current AWQ weights:** compile/register the existing TurboMind
   SM80 packed-W4 path and replay the exact 236-call/rank `M=5` shape mix. Go
   only if aggregate AWQ time is `<=7.184 ms` or if the measured secondary
   savings establish the same `2.997 ms` forward delta. Compare projection
   outputs before any quality run.
2. **H100/SM90, current AWQ math:** compare TurboMind SM80-style W4A16 against
   CUTLASS SM90 mixed-input W4A16. Separately test FlashInfer GDN-MTP and
   allreduce+RMS. Do not hide a failed dense path behind a faster attention
   result.
3. **Blackwell, separate model lane:** evaluate an NVFP4 checkpoint as a new
   quantization baseline. It is not an optimization result for the current AWQ
   checkpoint until the same official-sampling quality suite passes.
4. Run the non-eager target-only verifier microbenchmark, then the native-MTP4
   6k-token natural-stop gate, only after component thresholds pass.

## Non-Goals And Closed Interpretations

- CUDA graph is already enabled; graph replay does not provide the missing
  SM80/SM90 data-movement instructions.
- FlashInfer sampling is outside the `22.997 ms` target-forward span.
- Attention is only `0.967 ms`; an attention-only backend migration cannot
  close a `2.997 ms` forward gap.
- Async tensor parallel GEMM/communication fusion is not an automatic win for
  this model. The local policy enables sequence parallelism only for SM90/100
  models with hidden size at least `8192`; Qwen3.6-27B uses hidden size `5120`,
  and `M=5` is below the intended large-message regime.
- FP8, W4A8, relaxed acceptance, and NVFP4 alter precision or acceptance
  semantics. They require separate quality baselines and cannot be counted as
  exact current-AWQ wins.

## Primary References

- NVIDIA Ampere tuning guide: https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html
- NVIDIA Hopper tuning guide: https://docs.nvidia.com/cuda/archive/12.8.0/hopper-tuning-guide/index.html
- CUTLASS changelog and mixed-input GEMM support: https://github.com/NVIDIA/cutlass/blob/main/CHANGELOG.md
- Upstream TurboMind architecture-specific GEMM registry: https://github.com/InternLM/lmdeploy/tree/main/src/turbomind/kernels/gemm/kernel
- FlashInfer GDN decode/MTP API: https://docs.flashinfer.ai/api/gdn_decode.html
- FlashInfer GDN SM90 implementation contract: https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/gdn_decode.py
- FlashInfer/TensorRT-LLM MTP attention API: https://docs.flashinfer.ai/generated/flashinfer.decode.trtllm_batch_decode_with_kv_cache.html
- TensorRT-LLM quantization hardware matrix: https://nvidia.github.io/TensorRT-LLM/latest/features/quantization.html
