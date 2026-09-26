# Flash-Next MTP4 batch decode qualification

The accepted reference is **27.3963 ms per complete MTP4 round**. The target
verifier uses M5/M10 matrix batches; the draft loop uses M1. This work ports
the packed GDN input candidate from [PR #692](https://github.com/1CatAI/1Cat-vLLM/pull/692)
at `bcf0efa914e5e84b38164359563931e8a82e5f57` onto the shared MTP4 defaults,
then extends grouped expert reuse to M5 and fuses native gated RMSNorm.
Verification continues to use matrix batches. The requested acceptance
threshold is **less than 20 ms per complete unprofiled round**.

## Implementation and scope

- Integration base: `b034648012244ab712df05b93e6d8fff877a6f2f` (`onecat/main`).
- `VLLM_SM70_QWEN38_GDN_INPUT_BATCH=1` selects the existing native Tensor Core
  QKVZ/b/a projection with direct final-output stores, for M2..16 only.
  The flag remains opt-in because the packed weight copy reduces KV capacity.
- The existing exact Qwen3.8 topology, TP4, FP16, SM70 and no-MTP/MTP4
  admissions apply. Prefill and M1 retain their shared paths. There is no new
  activation quantization or accumulation-order change.
- Original weights remain available for prefill/M1. Packing adds
  **725.625 MiB per rank** for the 36 target GDN layers. Engine memory and KV
  capacity must be reported; component speed alone does not admit a default.
- Normal CMake registration builds the kernel into `vllm._C`. Development
  uses `setup.py build_ext --inplace`; no wheel or private kernel DSO is used.

PR #692 covers general no-MTP batch candidates and has an unresolved engine
quality gate. The first qualification below isolates its GDN input kernel.
The subsequent MTP5 grouped-expert route reuses its W2/reduction kernel from
`2c5b584468d52bb8391a5609c19bf7657390f67c`, preserving MTP's W13 split4.
Its other W13 scheduling changes, HC and collective experiments are excluded.
PR #687's split-copy optimization is already represented by the mainline
shared GDN split path and is not added again.

## Measurement contract

Python 3.12.13, Torch 2.10.0+cu128, CUDA toolkit 12.8.93, V100-SXM2-32GB,
checkpoint `RadixArk/Qwen3.8-Flash-Next-NVFP4`, TP4, FP16 activations/KV,
FP32 SSM, MTP4 with greedy draft, max length 32768, batch-token cap 8192,
one request, memory utilization 0.95, prefix caching, V2 runner,
FULL_AND_PIECEWISE graphs and 12 GiB/rank pinned host PLE.

Run M5/M10 real-weight component checks first, covering all 36 layers and
four TP slices with changing inputs and exact FP16-bit comparisons. Only
then run a matched full-model pair on one GPU set, with two fixed 8192/513
greedy requests and natural EOS prompts. Keep emitted token IDs, accepted
draft counts, pure decode, TTFT and round latency separate.

Node-trace kernel service is diagnostic, not the accepted round latency.
CUDA-event phase timing is collected after ordinary endpoint timing.
The old 27.3963 ms reference is retained; a current-build control determines
the candidate's measured gain.

## First qualification: packed GDN input

The normal CUDA source build succeeds. The optional Rust frontend is not built
in this environment. The new operator is registered in the ordinary `_C`:
SHA256 `2c96d0ccffd5c7af505179585f240da2f7a02c6b73d77b4c5725de458d4ac8fd`.
Fresh-process imports resolve Torch/CUDA/cuBLAS from the declared environment,
without `LD_PRELOAD` or private library overrides. cuBLAS is 12.8.4.1,
CUDA runtime 12.8.90 and Triton 3.6.0.

### Component gate

**114 targeted tests pass**, covering changing-input CUDA Graph replay,
M5/M10 output canaries, reloadable packed weights, unsupported-shape and
alignment fallbacks, the shared split-copy path and MTP4 graph admission.
Changed-file pre-commit hooks pass, including mypy and CUDA API checks.

All 36 real target GDN weight pairs, all four TP weight slices, M5/M10 and
six synthetic input scales produce **zero FP16-bit differences**. These are
independent component runs on physical GPU4, not simultaneous TP4 model time.
Each measurement includes both projections and final output layout, with
seven alternating graph trials over all 36 distinct weight pairs.

| TP weight slice | M5 control / candidate (ms) | M10 control / candidate (ms) |
| --- | ---: | ---: |
| 0 | 1.969408 / 1.141824 | 1.964416 / 1.145216 |
| 1 | 1.929344 / 1.122496 | 1.964800 / 1.144704 |
| 2 | 1.929472 / 1.123136 | 1.964736 / 1.145152 |
| 3 | 1.930112 / 1.135744 | 1.965184 / 1.147776 |

The component saving is **0.794--0.828 ms**, approximately 41%. It is not
subtracted from the historical 27.3963-ms number to invent a model result.

Nsight Compute collection was attempted once, but the driver returned
`ERR_NVGPUCTRPERM`; no hardware counters were collected. The compiled kernel
uses 126 registers/thread, 4096 bytes of shared memory and no local memory.
These are static resources, not measured occupancy, SM or HBM utilization.

```bash
CUDA_VISIBLE_DEVICES=4 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  OMP_NUM_THREADS=1 VLLM_SM70_QWEN38_GDN_INPUT_BATCH=1 \
  .venv/bin/python -m benchmarks.kernels.benchmark_sm70_gdn_input_batch \
  --model "$MODEL" --rows 5,10 --rank 0 --out micro_rank0.json
```

Repeat the rank argument for TP weight slices 1--3; it selects checkpoint
columns and does not start distributed workers. GPU ownership must be checked
before running the command.

## Full-model A/B

Physical GPUs0--3 are occupied by another service, so both current-build arms
use GPUs4--7. The control has the batch flag explicitly disabled and the
candidate has it explicitly enabled; every other acceleration flag and engine
setting is identical. The fixed fixture uses two ordinary 8192/513 greedy
requests. The value below is the median complete-round time, not TTFT and not
the intrusive trace timing.

| Fixture | Control (ms) | Batch GDN (ms) | Saving (ms) |
| --- | ---: | ---: | ---: |
| fixed8192, repeat A | 27.322924 | 26.421336 | 0.901587 |
| fixed8192, repeat B | 27.296220 | 26.570200 | 0.726020 |
| fixed8192 median | **27.309572** | **26.495768** | **0.813804 (2.980%)** |
| natural EOS 0 | 28.298227 | 27.692327 | 0.605900 |
| natural EOS 1 | 28.217924 | 27.653630 | 0.564294 |
| natural EOS 2 | 28.215669 | 27.665865 | 0.549804 |

The candidate is faster than both the same-build control and the accepted
27.396270-ms reference. Both fixed responses emit 513 tokens and have 335
drafts, 177 accepted tokens, mean acceptance length 1.528358, and position
acceptance `[125, 36, 10, 6]`. The three natural responses emit 284, 329 and
421 tokens respectively. Token IDs, finish reasons and all MTP acceptance
statistics match in every paired case. The trace request is intentionally
short and intrusive: it measures 35.622097 ms (control) versus 33.821767 ms
(candidate), so it is used for composition only and is not reported as the
endpoint baseline.

The new path increases model storage because the packed QKVZ and BA buffers
are retained alongside the original weights. The engine logs show:

| Engine resource | Control | Batch GDN | Change |
| --- | ---: | ---: | ---: |
| Model loading | 22.9 GiB | 23.61 GiB | +~0.71 GiB/rank |
| Available KV cache | 3.34 GiB | 2.66 GiB | -0.68 GiB/rank |
| KV cache tokens | 153,910 | 122,631 | -20.3% |
| Maximum concurrency | 4.70x | 3.74x | -20.4% |

The C=1 fixture fits and passes, but this capacity loss is a material engine
trade-off. The quality and latency gates pass; the memory gate does not yet
justify changing the default, so `VLLM_SM70_QWEN38_GDN_INPUT_BATCH` stays
default-off in this change.

## Trace decomposition

Nsight Systems captured 84 steady target-verifier windows per rank. Kernel
service is a sum of overlapping launches, while the graph wall time is the
critical rank's elapsed graph window; they must not be added together. The
control critical rank was rank 1 and the candidate critical rank was rank 2:

| Critical-rank target graph | Control | Batch GDN |
| --- | ---: | ---: |
| Graph wall (ms) | 28.389041 | 26.740710 |
| Kernel service sum (ms) | 22.830981 | 22.149391 |
| Kernel busy union (ms) | 20.544991 | 19.855439 |
| No recorded kernel (ms) | 7.844050 | 6.885271 |
| Kernel launches / round | 2,612 | 2,504 |

Rank-0 category service shows where the change lands:

| Rank-0 category (ms/round) | Control | Batch GDN | Delta |
| --- | ---: | ---: | ---: |
| Dense BLAS / GEMM / GEMV and reductions | 9.735184 | 7.667470 | -2.067714 |
| GDN / convolution | 0.960875 | 0.886225 | -0.074650 |
| GDN fused batched projections | 0 | 1.306008 | +1.306008 |
| TP communication including waits | 1.279297 | 1.213016 | -0.066281 |
| Other elementwise / copy / index | 3.648207 | 3.631834 | -0.016374 |
| PLE pinned-UVA lookup | 0.115087 | 0.114930 | -0.000158 |
| No recorded kernel (exclusive) | 8.054524 | 7.236864 | -0.817659 |

The top rank-0 CUTLASS 16x16 service falls from 6.013796 ms over 291 calls to
4.514411 ms over 255 calls. The new `gdn_input_batch_kernel` contributes
1.306008 ms over 36 calls, one per target GDN layer per round. The trace thus
shows the intended batch-decode fusion removing launch and dense-GEMM work;
it does not show a switch to row GEMV. The phase diagnostic collected after
the candidate endpoint reports 21.664982 ms target forward, 22.510001 ms
target-verifier wall and 5.079019 ms draft work on its critical rank. The
control phase RPC failed after all ordinary and trace cases completed, so that
diagnostic is not used as a cross-arm timing comparison.

Nsight Compute was rejected by the driver with `ERR_NVGPUCTRPERM`; SM
occupancy, tensor utilization and HBM bandwidth therefore remain unmeasured.
Static compilation reports 126 registers/thread, 4096 bytes shared memory and
zero local-memory bytes. The next optimization should remove the duplicate
packed-weight allocation, for example by an in-place or original-layout batch
kernel, before reconsidering a default-on setting.

Local raw reports, launch contracts, traces and analysis are retained under
`.artifacts/` in the owned worktree. Generated binaries and model data are
not committed.

## Continuing toward a complete-round latency below 20 ms

The requested acceptance threshold is now **less than 20 ms per complete
unprofiled MTP4 round**, with the same workload and healthy output/acceptance.
The measured 26.495768-ms candidate does not meet that threshold. A target-only
graph time, profiled launch gaps, or projected component savings cannot close
the remaining 6.495768-ms gap.

The following source-isolated screens retain the failed paths without another
full-model timing run:

- An original-layout batch Tensor Core HC down/SiLU kernel initially changed
  bits because the existing cuBLAS split-K workspace rounds every partial to
  FP16 before the ordered FP32 reduction. Restoring that boundary makes the
  screened outputs exact, but costs 18.640 us versus 17.068 us at M5 and
  31.412 versus 17.460 us at M10. Reject the schedule. The eight checkpoint
  pairs and five changing-input scales are retained in
  `hc_batch_fusion_v{1,2}.{json,log}` and `hc_batch_down_audit.py`.
- HC up plus gate mix preserves all screened bits, but the best M5 pair is
  only 14.000 -> 12.584 us and M10 regresses 14.404 -> 15.572 us. It is not
  integrated or presented as a model gain.
- Reusing the existing batch W13/SwiGLU and W2/reduce fusions with MTP's
  unchanged split4 is exact in five changed-input scales but mostly neutral
  on the complete MoE chain. Existing expert grouping is more promising:
  synthetic 50/35/20/10 unique-expert cases measure direct versus grouped
  71.515/74.115, 62.088/52.147, 50.507/38.525 and 47.965/32.504 us.
  These are checkpoint-layer0 component times, not real-route or model gains.
  `mtp_moe_fusion_v1.{json,log}` retains the full three-arm comparison.

The next decision uses captured MTP routes before admitting grouping. All
private research extensions remain confined to component screens; engine
validation uses the normal source-built `_C` only. The first diagnostic
route-capture launcher mistakenly inherited a 256K length / 0.9 memory
configuration from a general helper and failed KV capacity checks before
generation. Its failed log is retained as `mtp_routes_diagnostic.log`.
The corrected launcher reads the saved 32K / 0.95 endpoint configuration
directly and retains 12-GiB host PLE. Route-copy instrumentation is diagnostic
and is never used as new speed evidence.

### Grouped MTP5 experts and native gated RMSNorm

`VLLM_SM70_NVFP4_MOE_GROUPED_MTP5=1` reuses the existing expert grouping for
TP4/E512/H2560/I160/top10, exactly five verifier rows. W13 retains four ordered
K partitions and its FP16 activation boundaries. The grouped W2 kernel shares
weights between routes, stores FP16 projections inside the CTA and performs
the original ordered FP32 weighted reduction. It does not quantize activations
or route M5 through five M1 calls. Both flags remain opt-in during qualification.

`VLLM_SM70_RMSNORM_GATED_EXACT=1` fuses the native FP32 N128 RMSNorm and
sigmoid/SiLU chain, shared by M1 and batches. The implementation reproduces
ATen's contiguous vector4 mean reduction and explicit pointwise rounding;
using an arbitrary reduction tree or approximate sigmoid changes the oracle.
Admission requires contiguous FP16 tensors, 1..192 rows, SM70, no grouped norm,
normalization before gating and batch-invariant mode disabled. Other shapes
keep the existing path. This adds no persistent weight copy.

The ordinary source-built `_C` SHA256 for these two additions is
`b4c3ca4e7085f82eafb3b33d28446b8de2520420e293b864118f6b89fdb4071c`.
No private research extension is loaded by model workers. Native GPU gates
pass: 62 norm/group-dispatch cases in total, including changed-input CUDA
Graphs, output canaries and all 65,536 FP16 gate payloads for sigmoid and SiLU.
The initial run passed 52 cases and failed ten before executing the kernel
because the test omitted `default_vllm_config`; after adding that fixture,
only those ten cases were rerun and all passed. The separate CPU integration
and dispatch suite passes 56 cases.

The instrumented route snapshot captures five M5 cases over all 48 target
layers. Their mean distinct expert counts among 50 slots are 24.69, 26.13,
27.92, 29.21 and 34.71. This confirms substantial reuse for the retained
workload; instrumented generation times are not performance evidence.
Eight real weight layers with those routes and three input scales are bit
exact. The complete grouping/W13/activation/W2/reduction component means are:

| Captured route case | Direct (us) | Grouped and fused reduction (us) |
| --- | ---: | ---: |
| Fixed 8K / 65 | 53.875 | 40.650 |
| Fixed 8K / 129 | 54.385 | 42.238 |
| Fixed 8K / 257 | 54.941 | 43.223 |
| Natural code | 57.471 | 47.021 |
| Natural math | 61.149 | 52.308 |

The source-built MTP5 benchmark independently covers all four TP weight
slices, changing inputs/routes, poisoned intermediates and invalid expert IDs.
All intermediate and output bits agree. At 35 distinct experts it measures
about 59.2 -> 48.9 us; at 20 experts, 48.3--48.9 -> 34.2--34.4 us. The 50-unique
case also improves slightly. These are component timings, not round savings.

The source-built norm benchmark covers all 36 real norm weights and six input
scales. Every output bit agrees; the 36-layer M5 chain measures
**0.899328 -> 0.068582 ms**. M1 and M10 chains measure 0.886118 -> 0.063872 and
0.897126 -> 0.071040 ms respectively. Reproduce with the two committed
`benchmark_sm70_mtp5_grouped` and `benchmark_sm70_rmsnorm_gated_exact` modules,
passing `--model` and `--out`, plus `--rank 0..3` for expert weight slices.

A broader shared-expert gate fusion stress test rejects the vector8 dot
candidate: 1,440 checked M5 logits already contain one FP16-bit difference.
Even though the sampled final gated outputs agree, the dot is not admitted.
Keep `shared_gate_batch_v2.{json,log}` as a rejected arithmetic change.

### Matched source-built stack result

On physical GPUs0--3, the same `_C` build and workload above, with packed GDN
input enabled in both arms, the additional expert/norm flags measure:

| Fixed request | Control round (ms) | Grouped experts + norm (ms) |
| --- | ---: | ---: |
| 8192 / 513, A | 26.807870 | 23.837849 |
| 8192 / 513, B | 26.584251 | 23.862098 |
| Median | **26.696061** | **23.849974** |

This is a **2.846087-ms / 10.6611%** reduction in complete unprofiled rounds.
The previous 26.495768-ms run used GPUs4--7; the new same-GPU control is the
basis of the incremental claim. The historical 27.3963-ms reference is kept
separately. **The less-than-20-ms goal is not met; 3.849974 ms remains.**

Every token and acceptance statistic agrees for both fixed repeats, warmup
and three natural EOS cases. Natural output lengths are 284/329/421 tokens;
rounds improve from 28.048752/28.118647/27.994233 to
25.096709/25.112601/25.222245 ms. KV allocation remains 2.66 GiB/rank and
122,631 tokens. These two new changes add no persistent weight copy.

CUDA-event phase timing runs only after ordinary performance and quality
requests, in the same candidate engine. The rank with the largest mean total
wall time (rank1) reports target forward 18.208780 ms, target sample 0.738557,
target state update 0.016567, four drafts 5.038821, total GPU 24.078845 and
total wall 24.155889 ms. Its target verifier GPU sum is 18.963904 ms. These
separately instrumented phases explain the remaining work; the 18.96-ms
verifier alone does not satisfy the complete-round threshold. Do not subtract
phase measurements or profiler launch gaps from 23.849974 ms.

Raw evidence: `mtp_stack_{control,candidate}_0123.{json,log}`, their launch
contracts/source patches and `mtp_stack_comparison.json`. The local launcher
sets every candidate flag explicitly, retains the 32K/0.95/12-GiB-PLE
contract and acquires GPU leases. Both model runs complete successfully.

A subsequent packed HC screen retains exact outputs but offers only about
4.9 us per M5 down/SiLU + up/mix pair (eight real pairs, five input scales).
Down costs 16.296 -> 13.484 us and up/mix 13.624 -> 11.548 us. M10 down is
slightly slower. Duplicating both weight matrices for 96 modules would consume
roughly 1.20 GiB/rank, so this schedule is not integrated. See
`hc_batch_packed_v3.{json,log}`; it is not an additional full-model gain.

A packed batch output-projection screen preserves the M5 split2 FP16
workspace boundary, but every tested schedule is slower: 22--28 us versus
16--17 us. M10 uses a different cuBLAS numerical contract and also fails bit
parity. Keep `out_projection_batch.json` and `out_projection_batch_gpu0.log`
and reject without an engine run.

### Draft follow-up and trace validation

The low-overhead phase observation above leaves about 5.04 ms in four drafts.
The previous node trace (`mtp_batch_candidate_draft_breakdown.json`) contains
1.883523 ms/rank0/round of four local vocabulary projections and 0.822872 ms
of unquantized draft-MoE projections. These are old-trace service sums, not
an additive decomposition of the new 23.849974-ms endpoint.

The current SM70 Triton compilation for the draft MoE contains FP32 FMA and
no Tensor Core `mma.sync` instructions. The M1 W13 grid has only 30 CTAs for
80 SMs. Its tuned `BM2/BN128/BK64`, four-warp configuration is already
enabled; reenabling the historical tile is not a new optimization. The
checkpoint's BF16 draft weights are converted to runtime FP16; target NVFP4
expert kernels are not a direct replacement. Preserve sequential FP32 dot
accumulation and the W2 router-weight multiplication before FP16 conversion.

The first new node capture, `mtp_stack_trace`, completes generation but its
SQLite has no `CUPTI_ACTIVITY_KIND_KERNEL` table and cannot qualify a kernel
breakdown. Retain the failed report and parser error rather than assigning
its wall time to kernels. The follow-up capture explicitly starts/stops and
flushes profiling in all four TP workers before engine shutdown. The normal
extension rebuilt after formatting only has SHA256
`4d9bcd5ac535883f5b7608a3b8e9c37de2fcac15e2165ddfa720b94ff35a22ad`;
the measured endpoint pair keeps its original build hash above.
