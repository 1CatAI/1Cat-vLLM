# SM70 Qwen3.8-27B-NVFP4 Decode Recovery

Date: 2026-08-23; acceptance updated 2026-08-29

## Decision

The accepted Qwen3.8-27B-NVFP4 TP4, no-MTP decode route keeps every eligible
FP8 and NVFP4 projection on an SM70 native path. Channel-FP8 projections use
the memory-neutral QPN8 decode kernels for the accepted gate/up, down, and
output shapes. Remaining channel-FP8 projections, including the LM head, use
TurboMind W8A16. NVFP4 projections use TurboMind W4A16. No accepted result in
this document credits a Marlin fallback.

The recovery route also adds E4M3 KV support to Flash-V100 XQA and selects a
p64 partition for the exact batch-one, G6/D256 layout. This is the change that
makes the frozen native-sampler result exceed 70 tok/s. The p64 route is
restricted to E4M3, batch one, G6/D256; the existing E5M2 and FP16 policies
are unchanged. The later no-MTP 80 tok/s acceptance adds exact QPN4, TP4
all-reduce, QPN8 specialization, and an exact-Philox chunked sampler sidecar;
that acceptance is recorded at the end of this document.

## Frozen Contract

- Model: Qwen3.8-27B-NVFP4, config SHA256
  `1b3c71868d1299e52df6fc907deb202d5132b1ef0f72aae0ef6d15185dd53a5c`.
- Hardware/runtime: four V100-SXM2-32GB GPUs, TP4, Python 3.12, Torch
  2.10.0+cu128, CUDA 12.8.
- Engine: compressed-tensors, FP16 activation/output, E4M3 KV,
  `FLASH_ATTN_V100`, `max_model_len=262144`, `max_num_batched_tokens=8192`,
  `max_num_seqs=1`, prefix caching, aligned Mamba cache, chunked prefill, and
  `FULL_AND_PIECEWISE` CUDA graphs.
- Request: input 1024, natural output cap 256, no `ignore_eos`, no MTP.
- Sampling: temperature 1.0, top-p 0.95, top-k 20, request seed 20260815,
  engine seed 0. The 2026-08-23 recovery baseline uses the unmodified native
  sampler. The 2026-08-24 acceptance uses the separately registered,
  exact-Philox chunked sampler described below.
- Pure decode uses the 255 steady token intervals and excludes TTFT/prefill.

## Route Map

| Layer family | TP-local shape | Accepted decode route |
|---|---|---|
| FP8 gate/up | K5120 x N8704 | fused QPN8 gate/SiLU/up |
| FP8 down | K4352 x N5120 | QPN8 split-16 |
| FP8 output | K1536 x N5120 | QPN8 split-12 |
| FP8 GDN input/full-attention QKV and LM head | model shapes | TurboMind W8A16 |
| NVFP4 gate/up | K5120 x N8704 | TurboMind N32, lookahead 1 |
| NVFP4 down | K4352 x N5120 | TurboMind N32, lookahead 2 |
| E4M3 G6/D256 attention | B1, page 1568 | Flash-V100 XQA p64 |

Channelwise `[N,1]` FP8 scales are admitted and packed without retaining a
second permanent weight copy. The existing QPN8 model/shape/concurrency gates
remain in force. The NVFP4 selector is exact-shape gated by
`VLLM_SM70_NVFP4_QWEN38_TP4_M1_FAST_SELECTOR`; setting it to zero restores
dynamic tuning. Setting `VLLM_FLASH_V100_DECODE_PARTITION_SIZE=128` restores
the previous E4M3 attention partition.

## Pure Decode Result

| Milestone | Steady decode | TPOT |
|---|---:|---:|
| Original mixed checkpoint route | 29.35 tok/s | about 34.07 ms |
| Channel-FP8 TurboMind | 57.285 tok/s | 17.457 ms |
| E4M3 XQA p256 | 62.615 tok/s | 15.971 ms |
| Exact channel-FP8 QPN8 | 65.522 tok/s | 15.262 ms |
| E4M3 XQA p128 | 67.880 tok/s | 14.732 ms |
| NVFP4 N32 selectors | 69.991 tok/s | 14.288 ms |
| Clean final binaries, native sampler, E4M3 p128 | 69.904 tok/s | 14.305 ms |
| Clean pre-merge binaries, native sampler, E4M3 p64 | 71.732 tok/s | 13.941 ms |
| Merged-source confirmation, native sampler, E4M3 p64 | **71.342 tok/s** | **14.017 ms** |

Both p64 requests generated all 256 tokens, contained no EOS, and finished by
length. Relative to the immediately preceding clean-p128 control, the
pre-merge p64 run saves 0.365 ms/token and improves pure decode by 2.61%.
After merging `onecat/main` at `675a12dedc`, rebuilding Flash-V100, and
repeating the frozen request under an exclusive four-GPU reservation, p64
measures 71.342 tok/s at 14.017 ms/token. This is the conservative 2026-08-23
baseline and is superseded by the 2026-08-24 acceptance below.

Sampled token identity is not an attention quality criterion: one-output-ULP
changes can flip a low-margin random sample. The final p64 stream is coherent
and, independently, reproduces an earlier accepted full-length stream. The
operator gate below is the numerical criterion.

## Numerical and Operator Gates

The E4M3 XQA p64/p128 race uses page size 1568, G6/D256, checkpoint-style
E4M3 bytes, K/V scales 0.04/0.25, and the scalar paged decoder as reference.

| Sequence length | p64 | p128 | p64 gain | Maximum error |
|---:|---:|---:|---:|---:|
| 1025 | 46.633 us | 63.614 us | 16.981 us | 7.63e-6 |
| 1152 | 45.896 us | 59.346 us | 13.450 us | 7.63e-6 |
| 1280 | 46.072 us | 55.625 us | 9.553 us | 7.63e-6 |
| 2049 | 45.807 us | 56.751 us | 10.945 us | 7.63e-6 |

Every p64 and p128 output is within one representable FP16 output ULP of the
scalar reference. The focused GPU regression covers p64/p128/p256 with unit
and non-unit K/V scales and passes 6/6.

The retained NVFP4 selectors also passed independent FP32-oracle checks:
relative L2 is `2.701e-4` for gate/up and `2.924e-4` for down, with cosine
approximately one. The QPN8 source already passed its model-quality and
operator gates recorded in `sm70_qwen38_qpn8_decode.md`.

## Per-Token Profile

The latest short Nsight Systems node trace uses the merged p64 route. It is
composition evidence; its 14.635 ms traced TPOT is not used as the absolute
speed result because the unprofiled frozen request measures 14.017 ms. The
trace captures 63 decode replays on each TP rank and uses the 61 middle steps
for steady statistics. Graph-node kernel coverage is 93.48%.

| Component | Time/token |
|---|---:|
| TurboMind NVFP4 gate/up | 2.762 ms |
| QPN8 split-16 projections | 2.234 ms |
| TurboMind NVFP4 down | 1.706 ms |
| TP all-reduce | 1.697 ms |
| QPN8 split-12 projections | 0.895 ms |
| E4M3 XQA p64 | 0.587 ms |
| QPN8 fused gate/up | 0.508 ms |
| TurboMind FP8 dense/LM head | 0.407 ms |

Across rank-token samples, the replay interval is 14.621 ms and GPU activity
union is 14.055 ms, or 96.136% of the interval. Idle gaps total 0.565 ms/token.
The stream still launches 1140.8 kernels per rank per token; half are shorter
than 5 us. The grid-limited static occupancy ceiling places 69.40% of service
below 25% occupancy and 28.50% at 25-50%. Nearly continuous GPU work therefore
does not imply high useful compute utilization: batch-one launch geometry and
the serial projection/communication chain remain the main limit.

Fifty-millisecond NVML samples report 99.47% mean GPU busy-window duty and
48.26% memory-active-window duty. Per-GPU power averages 175.9-180.2 W and
peaks at 221.4 W; runtime allocation peaks at 29.06-29.36 GiB/GPU. Model loading
accounts for 5.77 GiB/GPU, while the configured cache budget reports 21.29
GiB/GPU; the remaining allocation includes CUDA graphs, workspaces, state, and
runtime context.

NVML memory duty is not achieved HBM bandwidth. Current trace durations imply
effective minimum packed-weight rates of 451.2 GB/s for NVFP4 gate/up, 365.1
GB/s for NVFP4 down, and 717.4 GB/s for QPN8 down, with useful arithmetic rates
of 1.805, 1.461, and 1.435 TFLOP/s/GPU respectively. These omit scales,
activation/output traffic, caches, and implementation-internal work. A current
NCU capture was attempted, but the host rejected non-root performance-counter
access with `ERR_NVGPUCTRPERM`; exact current SM, Tensor Core, occupancy, and
DRAM counters therefore require administrative profiling permission. The
previously accepted QPN8 counter evidence remains in
`sm70_qwen38_qpn8_decode.md`, but it is not credited as current NVFP4 or p64
XQA counter evidence.

## Rejected Experiments

- NVFP4 N64 and K32 candidates lost to N32 and were removed. The retained
  exact shapes use split 3/swizzle 4 with shape-specific lookahead.
- QPN8 split 17/20/24 lost. Split 12 won only for K1536 x N5120, saving about
  12.7 us/token across the 64 output projections.
- The first compact top-k20 Python sampler and first fused CUDA top20
  candidate did not preserve native Philox sampling and were removed. Both
  2026-08-23 p64 results use the native sampler. They are distinct from the
  later canonicalized, generator-state-preserving implementation accepted on
  2026-08-24.
- Replacing only the full-vocabulary sort while retaining both native
  full-vocabulary softmax operations saved 53.9 us in isolation, insufficient
  for a stable acceptance margin. A one-full-softmax hybrid stayed
  experimental because the p64 attention route solved the target without
  changing sampling math.

## Build, Tests, and Evidence

- Final `_C` SHA256:
  `e0ea14d0e40330b08a9951e67634e50b722540d5d1bbb48500f690323ab07624`.
- Final p64 Flash-V100 SHA256:
  `b418fed86b9c1ab9297c8795c24732818239b9a3aaca5ec9efb60933853d8ce7`.
- Focused CPU policy/dispatch tests: 12/12 passed (nine FP8/QPN8 and three
  E4M3 p64 policy cases).
- Focused E4M3 XQA GPU numerical tests: 6/6 passed.
- Ruff lint/format, Python byte compilation, shell syntax, and
  `git diff --check` pass for the changed files.
- Retained task evidence is under
  `.artifacts/qwen38_nvfp4_speed_20260823/`, notably the final p128/p64 JSON
  results, the merged confirmation
  `final_merged_qwen38_nvfp4_tm_e4m3_xqa_p64_native_sampler_full_graph_i1k_o256.json`,
  `e4m3_xqa_p64_vs_p128_clean.json`, and the parsed per-token Nsight tables
  under `profiles/`.

## 2026-08-24 No-MTP 80 tok/s Acceptance

### Accepted deployment and route

The accepted deployment is deliberately a three-binary composition. It uses
the compatible primary `_C` module with SHA256
`a0a0cd9ddeccc73fa3d920c7a869450c4b33d001f97637c35b75b966d89ad36d`,
the production CMake C++17 sampler sidecar with SHA256
`cdbfdd87dfa9119e52acc88d5787063202561a879a63334145a5616344a549ae`,
and Flash-V100 p64 with SHA256
`b418fed86b9c1ab9297c8795c24732818239b9a3aaca5ec9efb60933853d8ce7`.
MTP remains disabled.

The exact FP8 gate/up, down, and output shapes stay on the proven-faster QPN8
route. GDN input, full-attention QKV, and the LM head stay on TurboMind W8A16.
The accepted batch-one NVFP4 shapes use the native QPN4 decode route; unsupported
or non-admitted cases retain the TurboMind fallback. Thus every eligible FP8
sublayer still uses TurboMind unless the frozen shape has a measured faster
QPN8 specialization.

The sampler sidecar replaces a full-vocabulary sampling fragment with an
80-chunk top-20 reduction while preserving canonical sort order, top-p math,
Philox draws, and generator state. It is built separately so sampler operator
registration does not relink the quality- and speed-frozen primary `_C`.
`setup.py` extracts either CPython-SOABI or abi3 sidecar names, and
`vllm/_sm70_ops.py` loads the bundled module or the explicitly configured
`VLLM_SM70_SAMPLER_LIBRARY`.

### Speed result

All absolute results use the frozen TP4, input-1024/output-256 contract and
255 steady decode intervals. TTFT and prefill are excluded.

| Binary/measurement | Steady decode | TPOT | Disposition |
|---|---:|---:|---|
| Merged native-sampler baseline | 71.342 tok/s | 14.017 ms | baseline |
| Hand sidecar repeat A2 | 80.177 tok/s | 12.472 ms | accepted |
| Hand sidecar repeat A3 | 80.164 tok/s | 12.474 ms | accepted |
| Hand sidecar repeat A4, physical GPUs 4-7 | 80.026 tok/s | 12.496 ms | accepted |
| Production CMake C++17 sidecar A2 | **80.624 tok/s** | **12.403 ms** | accepted |

The hand-sidecar accepted repeats preserve the frozen 256-token SHA256
`0b2d335ddce9b282e45eea1b6c86525bc61eeb5ba1655e8228e6ef3bd1ce823b`.
The production C++17 sidecar changes a low-margin full-model random trajectory,
so random token identity is not treated as the sole quality gate. A direct
cross-build sampler diagnostic covers 100 independent seeds and 256 sequential
draws: hand and production sidecars produce identical token selections and
identical final generator-state SHA256
`4257d205138503840fea92fe6b15ddfa276df82c315513da4bbd785393c98c96`.

### Output-quality gate

The quality contract is the frozen first 250 GSM8K test questions, five-shot
prompting, greedy generation, maximum 256 output tokens, and per-question
record retention. It is intentionally independent of the random performance
prompt.

| Route | Correct | Accuracy | Invalid |
|---|---:|---:|---:|
| Frozen pre-optimization baseline | 226/250 | 90.4% | 0 |
| Hand-sidecar candidate | 227/250 | 90.8% | 0 |
| Production CMake C++17 sidecar | **226/250** | **90.4%** | **0** |

The production result therefore equals the frozen accuracy baseline with no
invalid outputs. Its JSON contains all questions, generated texts, extracted
answers, labels, correctness flags, token IDs, and per-item output hashes. The
result file SHA256 is
`143b2b345b830a858bff2568f9cf51148c8185acb42bb129e2dfc2d421a196d2`.

### Accepted-route trace and resource use

The latest Nsight Systems trace uses the accepted compatible main library and
the behavior-equivalent hand sidecar. It captures 63 graph replays on each of
four ranks and reports the middle 61 steps. Its 13.465 ms node-traced TPOT is
composition evidence; the unprofiled 12.474 ms result remains the absolute
speed evidence. Graph-node kernel coverage is 94.88%.

| Critical component | GPU service/token |
|---|---:|
| QPN8 split-16 projections | 2.213 ms |
| QPN4 fused gate/up | 2.165 ms |
| TP4 pack32 all-reduce | 1.587 ms |
| QPN4 down | 1.441 ms |
| QPN8 output projection | 0.915 ms |
| Flash-V100 E4M3 XQA p64 | 0.618 ms |
| QPN8 fused gate/up | 0.496 ms |
| TurboMind FP8 dense/LM head | 0.410 ms |

The mean replay interval is 13.430 ms and GPU activity union is 12.887 ms,
or 95.958%; idle gaps total 0.543 ms/token and the largest mean gap is only
9.538 us. The route still launches 1061.9 kernels per rank per token. The
grid-limited occupancy ceiling assigns 38.46% of service below 25% occupancy,
47.86% at 25-50%, and 13.63% at 50-75%, showing that batch-one kernel geometry
and the serial projection/communication chain remain the main headroom.

NVML's coarse windows report 100% GPU-busy duty, 53.25% memory-active duty,
and 56.36% of the 300 W power limit. Runtime memory is about 28.97 GiB/GPU;
SM and memory clocks are 1530 and 877 MHz. These are duty indicators, not
achieved SM, Tensor Core, or HBM throughput. Nsight Compute counters remain
unavailable because the driver returns `ERR_NVGPUCTRPERM`. A payload-only
model estimate is about 491 GB/s/GPU and useful arithmetic is about 4.33
TFLOP/s across TP4, but neither value is a hardware-counter measurement.

### Reproducibility boundary and evidence

The production sidecar is reproducible through the CMake target
`_sm70_sampler_C` and its focused microbenchmark selects exactly the reference
tokens across five distributions, 100 explicit-seed trials, and 100 default
generator trials. It reduces the measured fragment from 102.427 to 62.972 us.
The final Python regression run passes all 94 tests in the sampler, NVFP4
admission, and SM70 TurboMind adapter files. Ruff lint/format, Python byte
compilation, the sidecar wheel-name gate, and `git diff --check` also pass.

Freshly relinking the primary `_C` was tested separately. The first cold run
measured 77.314 tok/s, while the matching second run measured 80.181 tok/s at
12.472 ms/token. Both used a current-source C++17 sampler sidecar and produced
the exact same 256 token IDs as the accepted production C++17 run. The earlier
roughly 77-80 tok/s spread was therefore startup-state variation, not a decode
route or output regression.

The clean source snapshot is `7bd1d783e4861af5f5396a721156c44f623f88b0`.
Its primary `_C` SHA256 is
`c5d7c6a9a0fa714c1e97427c3d16369404ae5af6ee80f48b165595081cc43e56`,
and its C++17 sampler SHA256 is
`3058051f59949d88f5982d0c5bd80121f3c75a0f5307ca901fa6aebdec4cbc74`.
The primary and sampler `.nv_fatbin` SHA256 values are respectively
`dbba740219f00db9a528310fa84c0defb97c03f5cf99b519af5975609eb926e6`
and
`9a89e8cc22d1f8e36378005c08e2b32b315d79728989d6b4b2fe44fedfa5cb2c`;
both are byte-identical to the accepted binaries. The full primary ELF differs
because merged main adds two unrelated ModelOpt NVFP4 MoE host entry points,
and full sidecar hashes include host build-ID differences. Those differences
do not enter this dense batch-one graph.

The clean second-run result SHA256 is
`4e6ed3306772efca1a9c84b3681d50a268c644b8d6327df0fab2341581236bb1`.
Its serialized token-array SHA256 is
`f50e0ebab44ae350c7921bf91ba709fa896dcfb253c06842323b80de5b8bde32`,
exactly matching the accepted C++17 result. It records TP4, physical GPUs 0-3,
custom all-reduce enabled, both FP8 and NVFP4 TurboMind enabled,
`speculative_config=None`, and `spec_decoding_metrics=null`. Because both
device binaries and the full sampled stream are identical, the existing
production C++17 GSM8K result, 226/250 with zero invalid outputs, remains the
quality gate; repeating the same 250-question run would add no new path
coverage. Keeping the sampler in its own sidecar remains the packaging
boundary, but a current-source primary rebuild is now accepted.

CMake component installation was also checked without rebuilding unrelated
wheel targets. It installs `vllm/_C.abi3.so` and
`vllm/_sm70_sampler_C.cpython-312-x86_64-linux-gnu.so`, removes their build-tree
RPATHs, and preserves both accepted fatbin hashes. The installed-file SHA256
values are `d7ed490ef41e55f3858a45b3a92d62b2d527343fa14b04053cbe65ba16646749`
and `8ee8466359dc4432234dc3daa92918b2226879caad563d23a85b980c2a7c755e`.
A loader probe against the staged files registered the QPN4, QPN8, and
`_C::sm70_sample_packed_top20_out` schemas successfully. A broad wheel rebuild
of unrelated extensions is not required to establish this route's packaging
boundary.

Primary evidence paths are:

- `.artifacts/qwen38_nvfp4_speed_20260823/results/candidate_qpn4_fold_qpn8_m1_arpack32_official_cxx17_sidecar_chunk80_i1k_o256_a2.json`
- `.artifacts/qwen38_nvfp4_speed_20260823/results/candidate_qwen38_nvfp4_clean_head_cxx17_chunk80_i1k_o256_a2.json`
- `.artifacts/qwen38_nvfp4_speed_20260823/results/qwen38_nvfp4_gsm8k_250_official_cxx17_sidecar.json`
- `.artifacts/qwen38_nvfp4_speed_20260823/results/sampler_stream_hand.json`
- `.artifacts/qwen38_nvfp4_speed_20260823/results/sampler_stream_cxx17.json`
- `.artifacts/qwen38_nvfp4_speed_20260823/wheel_stage_clean_head_7bd1d783/`
- `.artifacts/qwen38_nvfp4_speed_20260823/profiles/candidate_qwen38_nvfp4_sidecar_chunk80_nsys_nvml_i1k_o64_per_token.md`
- `.artifacts/qwen38_nvfp4_speed_20260823/profiles/candidate_qwen38_nvfp4_sidecar_chunk80_resource_summary.md`

## Quality-safe GDN decode fusion

The next no-MTP batch-one optimization keeps the mixed checkpoint route map
unchanged and removes work at the GDN boundary. The exact Qwen3.8 TP4 decode
shape can issue its channel-FP8 QKV/Z projection and FP16 b/a projection from
one QPN8 launch, writing the four destination tensors directly. A separate
one-pass Triton kernel performs the 12-by-128 gated RMSNorm without changing
the accepted arithmetic. The accepted pair is opt-in through both
`VLLM_SM70_GDN_QPN8_BA_SPLIT=1` and
`VLLM_SM70_GDN_RMSNORM_ONEPASS=1`; the split projection cannot activate
without its paired RMSNorm gate. Other operator shapes and prefill retain the
existing operators. These are hardware/layout/operator gates; the measured
checkpoint identifies the evidence workload and does not select the route.
The archived JSON predates this audit naming cleanup and records the former
Qwen-prefixed flag names; kernel arithmetic is unchanged. The final source
also rechecks the loaded QPN8 code/scales, workspace pointer, bias, dtype,
contiguity, and same-device contracts before dispatch.

The matched endpoint used the same four V100-SXM2-32GB GPUs, TP4, input 1024,
output cap 256, official random sampling, E4M3 KV, Flash-V100, full CUDA graph,
and no MTP. Each result is the stable median of three sequential requests and
pure decode excludes TTFT/prefill.

| Route | Pure TPOT | Steady decode | Output gate |
|---|---:|---:|---|
| Current control | 12.249652 ms | 81.634970 tok/s | frozen 256-token stream |
| QPN8 QKV/Z plus b/a only | 11.941616 ms | 83.740763 tok/s | reject: first divergence at token 202 |
| QPN8 QKV/Z plus b/a and one-pass RMSNorm | **11.921648 ms** | **83.881019 tok/s** | pass: 3/3 exact |

The accepted pair improves steady decode by 2.751% and TPOT by 2.678% versus
the matched control. All three accepted requests reproduce the control's full
256-token SHA256
`8b37337f4c393711cb8550db6bae909b1e85de8df1cf5ba8c60d8c000749c0a2`.
The one-pass RMSNorm 48-layer microbenchmark is bitwise exact and saves
55.946 us/token. The QKV projection is exact on all layers; b/a differs only
at normal FP32-reduction roundoff (`6.368e-8` relative L2, maximum absolute
error `2.384e-7`) and the paired endpoint restores exact sampled output.

The post-rebase admission recheck uses the final source-built extension and
the default Qwen3.5 model wrapper rather than a diagnostic GDN boundary. The
C++ split-route oracle fired on all four TP ranks. Its matching latest-main
control/candidate stable medians are `12.261727/11.956949 ms` and
`81.554583/83.633383 tok/s`, a 2.549% decode-throughput improvement. All
three candidate requests exactly match all three latest-main control
256-token streams with SHA256
`ca77db3b032a1600a8567adea706108c1bd8c5472b3ace2318c41ab17c66c1f9`.

A fused Q/K RMSNorm, MRoPE, and KV-write experiment reached 85.451 and
85.015 tok/s on its two stable repeats but is rejected: it diverged from the
control at token 1 and stopped after 208 output tokens. That source and its
route flag are intentionally absent from the submitted change.

Primary local evidence paths are:

- `.artifacts/qwen38_nvfp4_speed_20260823/results/gdn_endpoint_control_r3_i1k_o256.json`
- `.artifacts/qwen38_nvfp4_speed_20260823/results/gdn_endpoint_ba_split_rms_r3_i1k_o256.json`
- `.artifacts/qwen38_nvfp4_speed_20260823/results/current_main_gdn_control_r3_i1k_o256.json`
- `.artifacts/qwen38_nvfp4_speed_20260823/results/current_main_gdn_route_recheck_r3_i1k_o256.json`
- `.artifacts/qwen38_nvfp4_speed_20260823/results/gdn_rmsnorm_production_op_bound_48layers.json`
- `.artifacts/qwen38_nvfp4_speed_20260823/results/qpn8_fused_ba_split_bound_48layers.json`
- `.artifacts/qwen38_nvfp4_speed_20260823/results/qknorm_mrope_cache_gdn_ba_rms_r3_i1k_o256.json`

## 2026-08-25 Long-context E4M3 page800 decode

### Scope and route

This optimization targets the exact SM70, E4M3, grouped-query decode route
used when aligned Mamba state makes the attention page 800 tokens. The
production KV allocation is interleaved as `[blocks, 2, 800, 1, 256]`; its K
and V views each have strides `[409600, 256, 256, 1]`. Admission additionally
requires G6/D256 and the existing dual-CTA batch route. Other page sizes,
layouts, head dimensions, KV dtypes, batch routes, and prefill keep the prior
implementation.

The specialized kernel uses the compile-time 800-token page and physical
block stride, replaces repeated general page division with the bounded
one/two-page comparison for a 256-token partition, and loads adjacent shared
V values as aligned `half2`. Each output dimension retains the same token and
FMA order as the fallback. The strict layout predicate is checked before
dispatch; setting `VLLM_FLASH_V100_E4M3_PAGE800_FASTPATH=0` restores the
general-stride kernel. The fast path is enabled by default on an admitted
route. `VLLM_FLASH_V100_E4M3_PAGE800_FASTPATH_TRACE=1` emits a once-per-process
route diagnostic.

The measured contract uses Qwen3.8-27B-NVFP4, four V100-SXM2-32GB GPUs with
TP4, compressed-tensors NVFP4 weights, FP16 activations, E4M3 KV, Flash-V100,
FlashQLA, prefix caching, aligned Mamba cache, FULL decode graphs, and no MTP.
The workload is NVIDIA SPEED-Bench revision
`487aa718444e816458d1a0a52bfce7a454285cf4`: the official 32K `low_entropy`
JSONL has SHA256
`c3ee8a3f63b6cce18d063a2ff9992b6b96cb72c2b6408970fa78d2335b542f8e`.
The explicitly derived 64K workload pairs two official rows and has SHA256
`0dcc877ecc7ccf66450c1f168ff7493dcf50f52da89d321dba394144a70e9800`.
Performance sampling is temperature 1.0, top-k 20, top-p 0.95, request seed
20260822, natural EOS, and a 512-token cap. Pure decode excludes queueing,
TTFT, and prefill; each cell has one warmup and three measured repeats.

### Long-context result

The matched rollback/candidate A/B was measured on source `05d5aa4e57`. B1
references are 52.052 token/s at 32K and 40.152 token/s at 64K. The accepted
16K scaling references are 70.736% at B8 and 52.281% at B16.

| Context | Batch | Rollback token/s | Candidate token/s | Candidate TPOT | Gain | Efficiency vs B1 |
|---:|---:|---:|---:|---:|---:|---:|
| 32K | 8 | 292.03 | **326.47** | 24.505 ms | **11.79%** | 78.40% |
| 32K | 16 | 392.84 | **451.92** | 35.405 ms | **15.04%** | 54.26% |
| 64K | 8 | 209.94 | **244.51** | 32.719 ms | **16.47%** | 76.12% |
| 64K | 16 | 260.36 | **311.74** | 51.325 ms | **19.74%** | 48.52% |

B8's two-context geometric-mean gain is 14.11%, B16's is 17.36%, and the
four-cell geometric mean is 15.72%; no cell regresses. Relative to the 16K
references, 64K loses no efficiency at B8 and 3.76 percentage points at B16,
within the ten-point budget. Every one of the 144 candidate performance
requests completed with its full 512 tokens and without empty or replacement-
character output. Route logs show FULL graphs and the expected long-context
`dual_cta_split` B8 and `dual_cta` B16 dispatches on all four ranks.

The endpoint reports that the exact D256 prefill extension is unavailable and
uses its established fallback. That warning is common to rollback and
candidate and is excluded from the pure-decode metric. This result must not be
cited as a prefill improvement.

### Counter and correctness evidence

An Nsight Compute rollback/candidate comparison on B16, 64K, p256 attributes
the endpoint gain to less address/control and shared-memory pressure rather
than added occupancy. Kernel duration falls 22.47%, executed instructions
11.92%, shared-bank conflicts 48.16%, MIO-throttle stalls 95.92%, long-
scoreboard stalls 8.61%, and wait stalls 11.06%. Eligible warps per active
cycle rise 0.565 to 0.699. Peak-relative DRAM and SM utilization rise from
24.49% to 28.11% and from 39.11% to 44.79%, respectively; registers rise only
from 166 to 168 and dynamic shared memory is unchanged. Thus the retained
change follows the measured copy/addressing bottleneck and does not depend on
an occupancy increase.

The production-layout operator A/B is bitwise equal to the rollback path for
B8/B16 and p64/p128/p256. At p256, its direct kernel gains are 22.10% at
32K/B8, 21.37% at 32K/B16, 22.59% at 64K/B8, and 22.84% at 64K/B16. The two
focused GPU suites pass 15/15 and 14/14; the production interleaved cases use
`torch.equal`, not a tolerance. All 55 environment tests, Ruff lint/format,
the SM70 extension build, and `git diff --check` also pass.

Because the production path uses official stochastic sampling, independent
server processes are not expected to reproduce every generated text. The
quality gate therefore pairs rollback and candidate over the same 32 GSM8K
main/test prompts at B8 and B16, temperature 1.0/top-k 20/top-p 0.95, natural
EOS, and sampling seeds 20260822, 20260823, and 20260824. This protocol was
fixed before the two additional seeds ran; both sides were rerun rather than
repeating only the candidate.

| Batch | Rollback numeric EM | Candidate numeric EM | Delta |
|---:|---:|---:|---:|
| 8 | 71/96 | **74/96** | +3 |
| 16 | 74/96 | **74/96** | 0 |

All 384 quality requests succeed, and the candidate has no empty or Unicode
replacement output. The first seed alone was one item lower in each batch;
the predeclared three-seed aggregate is recorded to make that sampling
variance visible rather than discarding it. The final source patch rebases
without conflict onto `onecat/main@d62ef5cb20`. Its rebuilt Flash-V100 ELF has
SHA256 `70816f9655e36fcd04b07d75be1b3abafabc81a30c261d4fd99b06389cbf82a2`.
The full SM70 SASS dump is byte-identical to the measured binary, with SHA256
`b30031fa4f29181b8e1c757bc408c608bae97df380893e613b96a2cf7d3e9a51`;
the differing full-ELF hash is build metadata, not device code.

## 2026-08-25 No-MTP Long-Context Decode Acceptance

### Contract and route

The long-context gate keeps the accepted Qwen3.8-27B-NVFP4 computation and
sampling paths unchanged. It uses four V100-SXM2-32GB GPUs with TP4,
compressed-tensors weights, FP16 activations/output, E4M3 KV, TurboMind for
the admitted FP8/NVFP4 projections, Flash-V100 attention, full CUDA Graph,
prefix caching, aligned Mamba state, and no MTP. Each request ends at an exact
128K or 256K context after 64 generated tokens. Pure decode is the 63 steady
token intervals and excludes prefill and TTFT. The acceptance thresholds are
60 tok/s at 128K and 45 tok/s at 256K.

The new Flash-V100 route is restricted to batch one, G6/D256, 1,568-token
interleaved paged KV, and E4M3. It selects p64 below 12,288 tokens, p256 below
49,152, p512 below 98,304, p896 below 196,608, and p1664 thereafter. A single
long-wave partition kernel and matching reducer select p512/p896/p1664 from
device-resident sequence lengths, so CUDA Graph replay does not need a host
readback or three long inactive launch pairs. The existing short p64/p256
launch remains separate. Other batch sizes, KV formats, head layouts, page
sizes, explicit partition overrides, and attention routes are unchanged.

Full CUDA Graph capture has two semantic batch-one variants at the p512
boundary. Contexts below 49,152 tokens replay the default graph, which contains
only the p64/p256 launch and reducer pair. Contexts at or above 49,152 tokens
replay the long-context graph, where `batch_context_routing` admits the wave
launch and reducer pair. The dispatcher uses its existing host scheduler
context length to select an already-captured graph; the attention kernel still
reads the live sequence length from device memory. This keeps the 1K decode
graph free of inactive long-wave launches without adding a device-to-host
sequence-length readback.

E4M3 conversion uses a 256-entry exact half-bit lookup table initialized in
CTA shared memory. It adds 512 bytes of dynamic shared memory and avoids the
long integer conversion chain on every KV element. The final SM70 cubin uses
168 registers/thread and 32 bytes of stack for the 192-thread long-wave
partition kernel; total dynamic shared memory is about 44.8 KiB, retaining two
CTAs/SM. Its reducer uses 32 registers/thread and 80 bytes of static shared
memory. The route is enabled with
`VLLM_FLASH_V100_XQA_E4M3_G6_P64_P256_AUTO=1`,
`VLLM_FLASH_V100_XQA_E4M3_G6_WAVE_PARTITIONS=1`, and
`VLLM_FLASH_V100_XQA_E4M3_G6_MERGED_WAVE_LAUNCH=1`. Setting the wave flag to
zero restores the prior p64/p256 policy. Setting the merged-launch flag to zero
retains the wave thresholds but restores the separate long-wave launches.
Setting the auto flag to zero restores the original explicit-partition route.

### Exact endpoint result

The control and candidate use the same model, projections, sampler, TP/GPU
set, max length, input/output lengths, attention backend, KV type, graph mode,
and no-MTP state. Only the Flash-V100 E4M3 attention partition policy changes.

| Final context | Route | Prefill | TTFT | Pure TPOT | Steady decode | Target |
|---:|---|---:|---:|---:|---:|---:|
| 128K | p64/p256 control | 283.296 s | 283.327 s | 24.654 ms | 40.561 tok/s | 60 tok/s |
| 128K | LUT merged-wave | 283.127 s | 283.154 s | **16.172 ms** | **61.834 tok/s** | pass, +3.06% |
| 256K | p64/p256 control | 1085.628 s | 1085.673 s | 36.422 ms | 27.456 tok/s | 45 tok/s |
| 256K | LUT merged-wave | 1085.519 s | 1085.568 s | **19.851 ms** | **50.376 tok/s** | pass, +11.95% |

At 128K, throughput improves 52.45% and TPOT falls 34.40%. At 256K,
throughput improves 83.48% and TPOT falls 45.50%. Both requests emitted all
64 tokens, finished by length, and reported no corruption. The complete token
streams are exactly unchanged from control: 128K SHA256 is
`66a130ea9d68faaf40f7c7cfb942c579f759b9535394cce958699bd620df2d6f`;
256K SHA256 is
`e1bc9ab8a979e43cda22348c0ff0056b5a7c3212ba57e830b731f3cf51906524`.
This is an attention-only optimization and does not credit a sampler change.

The post-integration 1K regression guard keeps the accepted TurboMind
projection routes, FlashQLA GDN BA-split route, official sampler, disabled
compile cache, and 256 output tokens. After excluding the first cold request,
the accepted GDN BA-split control is 83.761/83.721 tok/s and the
graph-specialized candidate is 83.184/83.187 tok/s, a 0.66% median delta. All
three candidate repetitions are token-for-token identical to one another and
to that accepted GDN control; their compact token-list SHA256 is
`c243818da5cced8b8653df4c441f713e978f8dfe7d8343d1b78c0eb3aeffe8ef`.
The short default graph therefore removes the inactive long-wave pair without
changing the admitted output stream or causing a material short-context
regression. A cache-reload diagnostic is not used as quality evidence.

### Operator and resource evidence

A same-GPU operator A/B uses the production interleaved K/V layout. At 128K,
the merged route takes 370.087 us versus 377.429 us for separate long-wave
dispatch; at 256K it takes 599.839 us versus 611.925 us. The selected explicit
p896/p1664 controls take 362.175/593.954 us respectively. Merged-route eager
and CUDA Graph outputs are bit-exact to those explicit controls. The small
remaining dispatch delta is preferable to three long kernel/reducer pairs in
the captured full-model graph.

The final 256K Nsight Systems CUDA-profiler-range trace uses the same
production interleaved layout and final extension. The active merged p1664
partition kernel takes 0.576603 ms and its reducer takes 0.033280 ms. The
device-routed short guard kernel/reducer take 0.004544/0.003552 ms, for
0.617979 ms across Flash attention kernels. The old p256 trace takes
1.391781 ms in the partition kernel and 0.316769 ms in the reducer, or
1.708550 ms total. The retained route therefore cuts traced Flash attention
time by 63.83% (2.76x), while remaining bit-exact to the explicit p1664
control. The trace is an isolated per-rank attention operator capture, not a
full-model TPOT decomposition.

During the exact decode windows NVML reports 100% GPU-busy duty on every rank.
At 128K, memory-active duty averages about 48.2%, allocation is
29.45-29.75 GiB/GPU, and mean power is 200.7-215.8 W. At 256K, memory-active
duty averages about 45.5%, allocation is unchanged, and mean power is
211.6-226.8 W. SM and memory clocks remain 1530/877 MHz. NVML memory-active
duty is not achieved HBM bandwidth; exact SM, Tensor Core, occupancy, and DRAM
counters remain unavailable because Nsight Compute returns
`ERR_NVGPUCTRPERM` on this host.

The production-layout correction is material: treating the unbound K/V views
as contiguous produced full-model output corruption and is rejected. A QK
software pipeline slowed the p896/p1664 kernels by roughly 30-38%; a split
reducer did not beat the retained route; paired E4M3 conversion measured
409.958/683.428 us at 128K/256K and lost to the shared LUT. All three
experiments are absent from the admitted path.

### Post-acceptance kernel headroom

Three additional changes retain the same admitted route and the real unbound
interleaved K/V views. First, the long-wave CTA stages its partition page IDs
once instead of reloading them for every tile. Second, a strict host-side
stride gate specializes the production `(blocks, 2, 1568, 1, 256).unbind(1)`
layout: the physical block stride is `2 * 1568 * 256`, while token and head
strides remain 256. Any other view uses the generic stride path. Third, the
fixed-layout E4M3 PV loop reads adjacent shared-memory values as `half2` while
retaining one FP32 FMA per dimension and the original token accumulation
order. It only changes lane ownership and restores dimensions explicitly at
the output store.

Each stage was measured as a four-arm control/candidate/candidate/control A/B
on one otherwise idle V100. Every arm contains five repetitions for each of
three page-table layouts; the table reports the median of the ten samples per
layout followed by the median across layouts. The exact partition-boundary
companions at 131,071 and 262,080 tokens pass as well.

| Exact context | Stage | Control | Candidate | Kernel delta |
|---:|---|---:|---:|---:|
| 128K | partition page IDs | 373.076 us | 368.604 us | -1.199% |
| 256K | partition page IDs | 602.017 us | 594.632 us | -1.227% |
| 128K | fixed interleaved stride | 370.102 us | 364.585 us | -1.491% |
| 256K | fixed interleaved stride | 598.257 us | 587.909 us | -1.730% |
| 128K | shared-V `half2` PV | 362.739 us | 348.265 us | -3.990% |
| 256K | shared-V `half2` PV | 584.745 us | 558.515 us | -4.486% |

All 12 sequence/layout cases have one output SHA256 across all four arms and
are bit-exact to their explicit p896/p1664 controls (`max_abs=0`). The final
partition kernel remains at 168 registers/thread and two CTAs/SM; the PV
change removes the prior 32-byte stack frame. Relative to the fixed-stride
binary, its static SASS instruction count falls from 2,424 to 2,248 and shared
loads fall from 118 to 90, with the same 57 FFMA instructions. These are
static cubin counts, not NCU throughput counters. The timed binary SHA256 is
`43f742cc228100dd0a58aed37bf1cb6007447129bcf3f1d28148bee71b6ff2a5`.
The clang-formatted final build SHA256 is
`5e66a5331aaf83e5e15080c5122c78047b94d3cb47a721572613bbbba9172d47`;
its target-kernel SASS is byte-identical to the timed build (normalized SASS
SHA256 `0684d453072ed630aa68477b618ab4c090eb18c87422a6de90af3fd43740a1ee`)
and its resource record is unchanged. The final build is bit-exact in eager
and CUDA Graph replay at 128K/p896 and 256K/p1664 (`max_abs=0`); all eight
parameterized page-1568 wave-boundary regressions pass.

Applying the three independently matched operator ratios to the measured
5.921/9.597 ms attention slices, while leaving the measured 10.251/10.253 ms
non-attention residuals unchanged, projects 15.784 ms (63.354 tok/s) at 128K
and 19.151 ms (52.216 tok/s) at 256K. These are decomposition-based
projections, not replacement end-to-end measurements; the accepted measured
endpoint remains 61.834/50.376 tok/s.

A packed pair-LUT initialization experiment is rejected because it is about
1.0-1.2% slower at 128K and 0.78% slower at 256K. Replacing constant division
with compare/subtract address normalization passed 2,801,664 exhaustive host
address checks and all GPU exactness cases but was statistically neutral. A
scalar PV-accumulator rewrite removed the stack frame but was neutral or
slower at 256K. None of these rejected variants is present in the admitted
path.

Primary retained evidence is:

- `.artifacts/qwen38_nvfp4_speed_20260823/results/e4m3_xqa_auto_single_p12288_long_i16k_64k_128k_256k_o64.json`
- `.artifacts/qwen38_nvfp4_speed_20260823/results/e4m3_xqa_lut_merged_wave_p12288_long_i128k_256k_o64.json`
- `.artifacts/qwen38_nvfp4_speed_20260823/telemetry/e4m3_xqa_lut_merged_wave_p12288_long_i128k_256k_o64.nvml.csv`
- `.artifacts/qwen38_e4m3_xqa_long_route/results/e4m3_xqa_lut_separate_same_gpu4.json`
- `.artifacts/qwen38_e4m3_xqa_long_route/results/e4m3_xqa_lut_merged_same_gpu4.json`
- `.artifacts/qwen38_e4m3_xqa_long_route/results/e4m3_xqa_wave_boundaries_graph_workspace.json`
- `.artifacts/qwen38_e4m3_xqa_long_route/results/page_ids/`
- `.artifacts/qwen38_e4m3_xqa_long_route/results/page_ids_fixed_stride/`
- `.artifacts/qwen38_e4m3_xqa_long_route/results/page_ids_fixed_stride_pv_half2/`
- `.artifacts/qwen38_e4m3_xqa_long_route/results/page_ids_packed_lut/`
- `.artifacts/qwen38_e4m3_xqa_long_route/results/page_ids_fast_address/`
- `.artifacts/qwen38_e4m3_xqa_long_route/profiles/e4m3_xqa_final_merged_wave_p1664_l262144_nsys.nsys-rep`

## 2026-08-29 Flash-Next Checkpoint-Native FP16-KV Decode

This result is a separate contract from the 2026-08-24 80 tok/s acceptance
above. The earlier result used input 1,024/output 256, E4M3 KV, admitted QPN8
projections, and a sampler sidecar. The current
`/data/models/RadixArk/Qwen3.8-Flash-Next-NVFP4` target keeps FP16 KV and
checkpoint-native NVFP4 experts, explicitly disables online QPN8 and the
top1-only LM-head shortcut, and measures input 8,192/output 512. The two
numbers must not be compared as though they were the same baseline.

### Matched contract and trace decision

The locked endpoint uses four V100-SXM2-32GB GPUs, TP4/PP1, the V2 runner,
FP16 activation and KV, ModelOpt FP4, Flash-V100, prefix caching with aligned
Mamba state, chunked prefill, `max_model_len=262144`,
`max_num_batched_tokens=2048`, `max_num_seqs=1`, and full CUDA Graph.
The speed request is deterministic greedy decode with input 8,192, output 512,
five repetitions, prefix-cache reset between repetitions, and `ignore_eos`
so every sample contributes the same 511 steady intervals. Pure decode
excludes TTFT and prefill. Natural-EOS official sampling is a separate quality
gate below.

The matched control is 65.853-65.882 tok/s across five repetitions, with mean
**65.864 tok/s** and **15.183 ms** TPOT. Its Nsight Systems graph-node trace
shows dense GEMV/GEMM and compressor service at 7.502 ms/rank/token and 812
launches/rank/token. The two largest cuBLAS GEMV families alone contribute
3.656 and 2.254 ms/rank/token. That evidence selected projection and
HyperConnection work instead of another attention or sampler experiment.

The retained exact-topology route is default-off and contains:

- checkpoint-FP16 SM70 row GEMV for the eight audited Qwen3.8 projection
  roles; 288 projections are prepared in the base model;
- a two-launch HyperConnection projection/mix route for the exact
  `hc_count=4`, hidden 2,560, low-rank 320 shape; 96 modules are marked;
- exact QSA lexicographic top-k using four score-radix passes followed by
  increasing-index pivot-tie compaction, replacing the redundant four index
  radix passes while retaining score-descending/index-ascending semantics;
- a single GDN input launch that computes QKVZ and b/a and writes the consumed
  qkv/z/b/a splits directly for the 36 exact non-interleaved TP4 GDN modules.

The final GDN operator screen uses 256 changing real-weight inputs. QKV and z
are bitwise equal in all 256 cases; b and a are bitwise equal in 253 and 254
cases, with worst absolute differences 0.00024414 and 0.00003052. The fused
launch measures 35.840 us versus 39.936 us for two GEMVs plus the split,
projecting 0.147 ms/token over 36 layers. QSA exactness, dense ties,
prefill-batch behavior, and CUDA Graph replay were also covered by the three
focused GPU tests.

### Endpoint result

| Route | Five steady decode samples (tok/s) | Mean TPOT | Mean speed |
|---|---|---:|---:|
| checkpoint-native control | 65.872, 65.882, 65.856, 65.855, 65.853 | 15.183 ms | 65.864 tok/s |
| retained FP16/QSA/GDN/HC route | 80.367, 80.451, 80.463, 81.560, 80.822 | **12.387 ms** | **80.732 tok/s** |

The gain is 14.869 tok/s, or **22.57%**, while TPOT falls **18.41%**. Every
candidate repetition exceeds 80 tok/s, population standard deviation is
0.442 tok/s, and all five 512-token sequences are identical. The first EOS is
at index 8 in every forced-length sample; the repeated chat markers after that
point are expected `ignore_eos` behavior and are not used as quality
evidence. Warm 8K prefill median is 2,766.4 ms versus 2,777.6 ms for control,
so the decode route does not show a material warm-prefill regression.

### Natural-output quality

The quality contract freezes GSM8K indices 8-23 with the official xhigh chat
template, temperature 1.0, top-p 0.95, top-k 20, seed 20260828, output cap
4,096, and natural EOS. It uses the same TP4 model route with MTP off.
Results are **15/16 raw and 15/16 strict**, 16/16 natural stops, 16/16 closed
thinking sections, zero length caps, and zero structurally invalid outputs.
The frozen same-prompt MTP4 reference is 15/16 raw and 14/16 strict. The only
candidate miss is the same reference miss: GSM8K item 12 predicts 12 instead
of 13, so it is not a new corruption signature.

Repository long-output health checks pass all 16 candidate responses: zero
replacement characters or bad markers, longest same-token run 3, longest
same-line run 1, and maximum repeated 50-character window count 6 against the
failure threshold of 40. The quality run itself reports weighted pure decode
at 80.935 tok/s and mean request decode at 81.017 tok/s.

The three feature switches are
`VLLM_SM70_QWEN38_FP16_GEMV`,
`VLLM_SM70_QWEN38_FUSED_HC_FP16`, and
`VLLM_SM70_QWEN38_FUSED_GDN_INPUT_FP16`; all default to false and require
the exact SM70, TP4, no-speculation, FP16 Qwen3.8 topology. Unsupported shapes,
prefill, other dtypes, online QPN8, and speculative decoding retain the
ordinary path.

Rejected work is intentionally absent from the retained source. An opaque
whole-GDN projection route fell to 73.277 tok/s; one-pass GDN RMSNorm was
neutral and changed output; CUB and sort-compaction QSA variants were slower;
the fused shared-expert screen was neutral; and the unaccepted fused W13
prototype was removed. Online QPN8 reached about 82.27 tok/s but changed token
trajectories and produced a repetition outlier. The top1-only LM-head shortcut
also changed long output and remains off.

Primary evidence is:

- `.artifacts/qwen38_exact_decode80/control/control_i8192_o512_r5.json`
- `.artifacts/qwen38_exact_decode80/trace_control_matched/control_i8192_o32_per_token.md`
- `.artifacts/qwen38_exact_decode80/trace_gemv_router_ba_index_hc_gate_overlap_matched/trace_gemv_router_ba_index_hc_gate_overlap_matched_i8192_o32_per_token.md`
- `.artifacts/qwen38_exact_decode80/operator_fp16_gdn_dual.json`
- `.artifacts/qwen38_exact_decode80/gemv_router_ba_index_hc_bk256_qsa4_gdndual_gate_overlap_full/gemv_router_ba_index_hc_bk256_qsa4_gdndual_gate_overlap_full_i8192_o512_r5.json`
- `.artifacts/qwen38_exact_decode80/gsm16_exact_decode80_official_xhigh/audit.json`
- `.artifacts/qwen38_exact_decode80/gsm16_exact_decode80_official_xhigh/health.json`

## 2026-09-04 current-main single-request decode trace

The unified dual-compile/hybrid-PLE service was reprofiled at public-main SHA
`05910abb97446128a259fbd5fbe2bf9ece70a492`. The locked route is TP4/V2,
no MTP, FP16 activation/KV, checkpoint-native NVFP4 experts, full decode CUDA
Graph, prefix caching off, and input 8,192/output 513. One model load ran a
513-token low-overhead baseline outside the profiler capture and then captured
only a 32-token graph-node diagnostic.

The 8K baseline measured `83.3749 tok/s`, or `11.9940 ms/token`; the accepted
short-prompt service point remains `86.07 tok/s`, so context length must stay
in every decode comparison. The node trace measured a `12.783 ms` middle-token
replay interval, `12.762 ms` GPU activity envelope, and only `0.050 ms` mean TP
replay-start skew. It covered `97.09%` of graph-node kernels and contained about
1,644 kernels/rank/token. Half of those kernels were shorter than 5 us. The
unprofiled decode samples reported 100% GPU utilization but only 140-150 W per
board, consistent with HBM traffic and small-kernel/graph-node issue cost rather
than FP16 compute saturation.

Rank-average service attribution, which is not additive wall time because the
shared-expert stream overlaps the main stream, is:

| Subsystem | Service ms/rank/token |
| --- | ---: |
| HyperConnection | 2.603 |
| NVFP4 MoE expert/router/activation | 2.203 |
| checkpoint-FP16 row GEMV | 1.441 |
| QSA sparse attention | 1.263 |
| remaining dense/cuBLAS, chiefly shared expert | 1.256 |
| fused GDN input | 1.064 |
| elementwise/metadata/copy | 1.022 |
| TP communication | 0.854 |
| GDN recurrent/core | 0.658 |
| LM head/sample | 0.582 |

The checkpoint-FP16 HC down/up, fused GDN input, and remaining row-GEMV kernels
read at least 2.788 GB of weights per rank and token. Exact tensor sizes and
trace duration imply 552-596 GB/s for HC, 529 GB/s for row GEMV, and 714 GB/s
for fused GDN input. These are traffic lower bounds rather than NCU counters;
the current host blocks performance counters with `ERR_NVGPUCTRPERM`. The GDN
input is therefore not the first target. The ordered implementation candidates
are exact router projection/top-k, NVFP4 W2 plus weighted-reduce fusion, QSA
decode fusion, critical-path graph-node reduction around HC, and an exact or
guarded greedy LM-head route. Full-model startup is deferred until standalone
real-weight candidates project at least 0.4 ms/token combined savings.

Raw reports remain outside Git under
`.artifacts/qwen38_nomtp_token_trace/`, including the `.nsys-rep`, exported
SQLite database, parsed per-token JSON/CSV/Markdown, route contract, and GPU
samples.

### Exact post-trace candidates

Three lossless single-token changes have passed focused operator gates after
the trace. They retain checkpoint FP16 activations and HC weights, native
NVFP4 expert weights, FP32 accumulation, and the existing FP16 materialization
boundaries:

- The Qwen3.8 W2 kernel now forms each route's FP16 result before applying the
  top-k weight and rank-ordered reduction in the same launch. Its real-weight
  CUDA Graph gate is bitwise and projects `0.098 ms/token` savings over 48
  layers.
- HC up reuses the same 320-element low-rank vector across four independent
  output rows. The selected row-four schedule is bitwise in all 128 changing
  input cases and reduces the 96-call HC cycle from `2.139 ms` to `2.062 ms`,
  saving `0.077 ms/token`.
- Qwen3.8 M1 `all_reduce_sum2` now reuses the registered SM70 TP4 push buffers.
  Both paths first form the local FP16 sum, accumulate ranks 0 through 3 in
  FP32, and round once to FP16. The four-rank CUDA Graph gate is bitwise for
  integer, model-distribution, and signed-zero patterns. Forty-eight
  collectives fall from `0.459 ms` to `0.136 ms`, saving `0.323 ms/token`.
  `VLLM_SM70_TP4_PUSH_ALLREDUCE_SUM2_M1=0` is the rollback.
- Single-row QSA decode now performs one coarse score-radix pass, compacts only
  that bucket, and refines the remaining radix bytes in shared memory. The
  final increasing-index scan is unchanged, so lower-index score ties and the
  downstream accumulation order remain exact. Twelve real-shape launches at
  lengths 2,048-2,169 fall from `0.2169 ms` to `0.1238 ms`, saving
  `0.0931 ms/token` or 1.75x. Random scores, dense ties, signed zero, Inf/NaN,
  the 2,304-entry boundary, and the 2,305/4,096/16,384 device fallback are
  bitwise equal to the original selector. Multi-row prefill retains the
  original kernel. Source-overlay validation may supply the same compiled
  fragment through `VLLM_SM70_QSA_TOPK_LIBRARY`; release wheels link it into
  `_C_stable_libtorch` normally.

The isolated savings sum to `0.591 ms/token`; they are not an end-to-end TPOT
claim because the shared-expert and main streams overlap. One full-model A/B is
still required before treating the operator gains as service throughput.

Privileged NCU counters confirm why HC needs traffic/issue improvements rather
than lower precision. The down projection reaches `488 GB/s` DRAM throughput
with `24.23%` achieved occupancy and spends `86.11%` of scheduler cycles with
no eligible warp. HC up reaches `481 GB/s`, `64.31%` occupancy, and `63.45%`
no-eligible cycles. More warps, split-K down, down row tiling, and the SGLang
persistent atomic-grid HC implementation are slower on V100. HC
combine-plus-RMSNorm is already only about `0.305 ms` per 96-call graph cycle;
an 8-warp variant saves just `0.014 ms` and changes the reduction result, so it
is rejected. A warp-per-dot HC-up kernel is `0.132 ms/token` slower and changes
89 of 245,760 FP16 outputs by at most `0.000488`. A same-precision FP16 Tensor
Core/QPN layout is also `0.012-0.035 ms/token` slower, consumes about 0.6 GiB
more packed weights per rank, and does not reproduce the established FP16
materialization boundary. Both are rejected. The retained HC changes do not
quantize FP16 tensors or relax any quality gate.

Further exact HC screens close the inexpensive schedule space. Bypassing L1 for
streaming weights is bitwise but `0.054 ms/token` slower. Changing Triton
pipeline stages is bitwise and neutral within `0.002 ms/token`; 8/16-row HC-up
tiles and paired-stream prefetch are bitwise but `0.043-0.097 ms/token` slower.
Larger down reduction tiles save at most `0.022 ms/token` while changing FP16
outputs by one ULP, so they are rejected. Fusing the attention output projection
with HC combine, followed by an exact norm-only kernel, is bitwise for both
multi-stream and normalized outputs but is `0.013 ms/token` slower over 48
calls. These paths should not be rescanned without a different kernel
architecture.

### Exact TP4 HyperConnection compute sharding

The next retained HC candidate changes work placement, not model precision.
For each M=1 HC down projection, rank `r` computes low-rank rows
`[80r, 80(r+1))` and injection row `320+r` directly from the existing
replicated checkpoint-FP16 weight. A rank-ordered push all-gather reconstructs
the original 320 low-rank and four injection values. Each rank then computes
the corresponding 2,560 rows of the FP16 HC-up projection, and a second push
kernel applies the established FP16 gate boundary, FP32 sigmoid and
rank-ordered FMA, and final FP16 materialization. The implementation keeps the
full weights resident, so prefill and unsupported cases use the original
replicated path without a weight-loader or memory-layout change.

On four V100-SXM2-32GB GPUs, the real-shape 96-HC CUDA Graph cycle falls from
`2.042378 ms` to `1.748982 ms`, saving `0.293396 ms/token` or 16.78%. All
block and injection outputs are bitwise equal on all four ranks. A separate
production-dispatch smoke covers 16 changing inputs through the registered
custom op and CUDA Graph lifecycle; all four ranks report zero FP16 bit
mismatches. The route requires the existing checkpoint-FP16 HC opt-in, exact
Qwen3.8 topology, fully connected TP4 SM70 custom all-reduce, and registered
push buffers. Otherwise it falls back before launching a sharded kernel.

This candidate does not use FP8, INT8, QPN, altered activation types, or a
reduced-precision accumulator. Together with the preceding isolated exact
screens, projected operator savings are `0.884 ms/token`; this is still not an
end-to-end throughput claim, and it does not by itself establish the 100
tok/s target.

Two additional no-lower-precision screens were rejected. A deterministic
E512/K10 top-10 selector preserves all outputs bitwise across random inputs,
dense ties, signed zero, NaN, and infinities, but loses `0.023 ms/token` with
hot logits and `0.049 ms/token` after a 64-MiB L2 scrub. GDN input row tiling
is bitwise but saves only `0.0046 ms/token`. Checkpoint-native NVFP4 W13
split-16 retains FP32 MMA accumulation and FP16 output but changes FP32
summation grouping; it differs from split-8 by one FP16 ULP in about 0.28% of
sampled outputs, so it is not enabled without a full model quality gate.

### Hidden-coordinate HC sharding, 2026-09-05

The next exact M=1 candidate assigns each TP rank 640 hidden coordinates and
computes all four branch gates for those coordinates. It preserves the
checkpoint FP16 weight layout, the two-K-warp FP32 reduction, the FP16 gate
boundary, FP32 sigmoid and branch-ordered FMA, and the final FP16 output.
The following collective gathers final hidden slices instead of branch gates:
each rank sends 1,280 rather than 5,120 bytes to each peer. No extra weight
copy or precision change is introduced, and prefill is unchanged.

It uses the existing `VLLM_SM70_QWEN38_FUSED_HC_FP16` opt-in and exact TP4
admission checks. A source-matched custom-AR extension enables the new
`sm70_qwen38_hc_output_allgather` op; an older extension retains the existing
gate-sharded path. Capability discovery and dispatch use the DSO that owns the
opaque communicator, never a different extension's fallback symbol. The new
gather reuses the isolated HC channel, not the concurrent MoE channel.

The initial screen loads all 96 real HC weight pairs, not a repeated layer-0
weight. Four V100-SXM2-32GB ranks each report zero FP16 bit mismatches for
block and injection outputs over 16 changing inputs. After 1,000 warmup graph
replays, three alternating paired timing groups give the following medians:

| Variant | 96 Mix calls (ms) | Change from control (ms) |
| --- | ---: | ---: |
| Current gate-sharded control | 1.743988 | — |
| Hidden shard, two hidden rows / eight warps | 1.703158 | -0.040830 |
| Producer-only down publication, coalesced revision | 2.037357 | +0.293369 |
| Exact down partials + fused tail/gather, one part | 1.842709 | +0.098720 |
| Same, two parts | 1.850873 | +0.106885 |
| Same, four parts | 1.864315 | +0.120327 |

Only hidden sharding is retained. The first producer-only version was still
slower at 2.229951 ms. Coalescing its peer writes reduced that overhead but
did not beat the control. Fixed-order down splitting also remained slower
after half2 loads and a one-warp gather tail. None of those losing prototypes
is part of the production dispatch. Alternative hidden tiles / warp counts
were bitwise but slower than the selected two-row/eight-warp schedule.

These are Mix-only graph measurements: they exclude HC combine/RMSNorm and
must not be subtracted directly from the 2.658-ms full-HC trace service sum.
The initial prototype improvement is about 2.3%, not the initial 20% screening
target and not an end-to-end tokens/s claim.

The committed production implementation (`aaf63696b6`) subsequently passes
the registered-op gate: 96 real weight pairs x 16 changing inputs x four
ranks, including 512 graph replays overlapping the actual sum2 route on an
auxiliary stream. All HC block, injection, and sum2 outputs have zero FP16 bit
mismatches. The independent hidden-shard GPU unit test passes, as do all 13
CPU dispatch tests. With Torch `2.10.0+cu128`, runtime CUDA `12.8`, and the
SM70 extension compiled by NVCC `12.0.140`, three paired timings are:

| Production dispatch | Paired samples (ms) | Median (ms) |
| --- | --- | ---: |
| Existing gate shard | 1.738315 / 1.739291 / 1.738595 | 1.738595 |
| New hidden shard | 1.689020 / 1.690590 / 1.690003 | 1.690003 |

The final Mix-only saving is **0.048592 ms (2.79%)**; each variant's sample
range is below 0.1%. No full-model startup is justified by this small increment
alone. Combine it with other admitted exact candidates for the next matched
endpoint gate; natural-output quality and the required 256K boundary remain
part of endpoint promotion. The 100-tok/s target remains unproven by this
change.

Reproduce the production-dispatch gate without loading the entire model:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_SM70_TP4_PUSH_ALLREDUCE=1 \
  .venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=4 \
  benchmarks/kernels/benchmark_sm70_hc_tp4.py \
  --model /path/to/Qwen3.8-Flash-Next-NVFP4 --out /path/to/hc-result.json
```

Use a source-matched wheel or set `VLLM_SM70_CUSTOM_AR_LIBRARY` to an extension
that contains both the new op and the complete communicator lifecycle. The
benchmark compares old and new registered-op dispatch, checks all 96 real
weight pairs, overlaps HC with the actual sum2 CUDA Graph route on an auxiliary
stream, and reports three paired Mix-only timings separately from correctness.

### Exact fused HC up/mix/gather candidate (2026-09-05)

The source now includes the selected 160-CTA FP16 up/mix/gather kernel. It
uses 128-bit weight/input reads, parallel branch sigmoids, and the same
eight-term FP32 FMA chains, XOR reduction tree, FP16 gate boundary, and
branch-ordered FP32 mixing. Each CTA publishes exact FP16 outputs together
with a generation tag; its two packet slots and generation counter are
isolated from both legacy HC collectives and the auxiliary MoE channel.
The existing communicator allocation grows by 21,120 bytes per rank; there
is no weight copy, additional communicator, or new user tuning switch.

The existing `VLLM_SM70_QWEN38_FUSED_HC_FP16` opt-in and TP4/SM70/M=1 gates
still apply. A source-matched extension selects fused up/mix/gather; an older
extension retains hidden-sharded split up/gather, or the legacy gate-sharded
route if needed. Optional-op capability and dispatch must come from the DSO
that owns the communicator, including the extended allocation layout.

The preceding **prototype** complete-HC screen measures `2.108826 ->
1.999374 ms` (5.19%) with bitwise intermediate/final outputs. The subsequent
registered production gate at `0303b82d1e` measures **`2.109529 -> 1.994807 ms`
(5.44%)**. All four ranks pass the 16-input intermediate/final checks, 512
auxiliary sum2 replays, and post-timing checks after packet generation wrap.
Fused samples are `1.994807/1.993735/1.996370 ms`, versus split-hidden
`2.109556/2.109529/2.109242 ms`. Runtime is Torch `2.10.0+cu128`, CUDA `12.8`,
TP4 V100-SXM2-32GB; the sidecar was compiled with NVCC `12.0.140`.
The `1.5-ms` whole-HC target and endpoint speed are not established by this
microbenchmark. The old `2.658-ms` whole-model trace uses a different scope
and must not be compared directly to it.

Run the complete registered-op gate, without loading attention/MoE weights:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  VLLM_SM70_TP4_PUSH_ALLREDUCE=1 VLLM_SM70_TP4_PUSH_ALLREDUCE_SUM2_M1=1 \
  .venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=4 \
  benchmarks/kernels/benchmark_sm70_hc_full_chain.py --fused-up \
  --model /path/to/Qwen3.8-Flash-Next-NVFP4 --out /path/to/hc-full-result.json
```

This compares forced split-hidden and fused registered dispatch, includes all
HC norms and final projections, checks 16 changing inputs and 512 auxiliary
sum2 graph replays, then rechecks outputs after timing crosses packet-tag
wrap. Timings exclude the auxiliary stress workload. Both this benchmark and
the older Mix-only gate explicitly freeze their control routes so a newer
extension cannot silently replace both sides of the comparison.
