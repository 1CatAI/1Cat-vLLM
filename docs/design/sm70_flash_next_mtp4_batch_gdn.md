# Flash-Next MTP4 batch decode qualification

The post-main-merge audit measures **21.411234 ms** per complete MTP4
round (ordinary controls 21.421255 / 21.401212 ms). All fixed and natural
token IDs, acceptance statistics and normal EOS match the retained oracle.
The requested **less-than-20-ms** threshold remains unmet. New flags remain
opt-in. The earlier 21.507750-ms result and 21.427792-ms same-engine
confirmations are retained below as historical evidence.

The current calibrated outer trace closes at **21.515078 ms** with +0.4688%
request perturbation: target forward 15.479071, sampling/state 0.784805,
four drafts 4.664252 and preparation 0.586950 ms. Current internal timestamp
observers still fail perturbation calibration. Neither the retained
29.391136-ms Nsight capture nor internal marker sums are an absolute
breakdown of ordinary execution. See the final merge-audit section for the
latest source, native hashes, quality gates and raw reports.

The original accepted reference is **27.3963 ms per complete MTP4 round**.
The target verifier uses M5/M10 matrix batches; draft step0 uses M5 and its
three continuations use M1. This work ports
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

The old trace also closes the draft wall on rank0, using 84 identical windows
from `_prepare_eagle_inputs_kernel` to the sampled/draft-combine entry. First
draft costs 1.572600 ms and the three continuations about 1.23 ms each. All
298 kernels occupy a 4.790127-ms union inside a 5.271298-ms window; the
remaining 0.481171 ms has no recorded kernel. Not every gap is removable host
work. The exclusive category table closes without adding overlapping service:

| Old-trace draft category | Exclusive wall (ms/round) |
| --- | ---: |
| Four full-vocabulary projections | 1.883523 |
| Draft MoE projections | 0.794247 |
| Other dense projections and reductions | 0.634174 |
| Sampling, state and elementwise | 0.547145 |
| Attention and indexer | 0.344693 |
| TP communication including waits | 0.322858 |
| HC postops and fused M1 projections | 0.196853 |
| Overlapping different kernel families | 0.066633 |
| No recorded kernel | 0.481171 |
| Total | **5.271298** |

The checkpoint vocabulary is 248,320, hidden size 2,560, TP4. Each full local
FP16 head has 317,849,600 weight bytes. Dividing that nominal traffic by the
roughly 0.47-ms projection gives about 675 GB/s; this is a traffic estimate,
not an NCU DRAM counter or achieved hardware-utilization measurement. Head
work is consistent with a substantial bandwidth cost. In contrast, the M1
MoE W13 projection issues only 30 CTAs and executes sequential SIMT FMA with
repeated shared-memory staging, which gives a concrete scheduling target.
Local argmax reduction and the previous tuned MoE tile are already enabled.

### Integration refresh

Draft PR [#703](https://github.com/1CatAI/1Cat-vLLM/pull/703) contains the
qualified implementation at `45248dc8d4abdd172f6d59e15ab8bb786466bfe1`.
Main `1e90d17f2c75e443b2a85a576ed68fa04c5f9dd6` is subsequently merged in
`2e4369373a3cfecbe84c401917c870951e140707`, preserving both migration ledgers.
The normal source rebuild succeeds, including native attention. Its `_C`
SHA256 is `eda731ee07bafaae3692226812c8a31833958dd2a2baef7ef55935bbbadc7151`
and `_vllm_fa2_C` is
`4fff2e872eeaa48337475c37f546e9b0fdc333c67b1ec9289fbf8d7eef29c86a`.
Only standard CUDA/Torch dependencies appear in `readelf -d`.

Post-merge CPU regression: **108 passed, 8 GPU cases skipped**, using
`CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q` with
`tests/compile/test_sm70_decode_graph.py`,
`tests/models/qwen4_exp/test_sm70_gdn_projection_split.py` and
`tests/quantization/test_sm70_nvfp4_grouped_decode_dispatch.py`.
This CPU result does not replace the prior GPU gates or establish new model
performance. The accepted 23.849974-ms pair keeps its original source/build
identity. Subsequent draft-MoE and original-layout component screens are
recorded below; their timings do not replace this endpoint pair.

### Refreshed node trace and exact draft-MoE screen

`mtp_stack_trace_flush` uses the refreshed source/native build above. The
report contains all four TP workers and **84 closed rounds**. Generation
finishes, but the primary exits during capture shutdown (exit 143) and
leaves workers waiting; only these owned processes are cleaned up. The
post-trace phase request does not run. Retain this limitation: the kernel
intervals are usable, but this is not a successful endpoint or phase run.
The next capture launcher explicitly sets `--kill=none --wait=primary`.

The Nsight diagnostic table warns for all four TP workers that the driver's
CUDA 13.0 version is unsupported by Nsight Systems 2025.1.1 (tracing uses
12.8 libraries), and that CUDA and NVTX records may be incomplete. Each
selected target/draft window contains 2,060/298 recorded kernels, but empty
intervals must remain **no recorded kernel**, not proven GPU idle time.

The rank0 intervals close as follows; these are **profiled milliseconds**:

| Interval | Mean ms |
| --- | ---: |
| Metadata to first target graph node | 0.782884 |
| Target graph, first to last node | 22.614401 |
| Target end to gather | 0.020994 |
| Sampling/state handoff | 0.949621 |
| Four drafts including combine | 5.266901 |
| Next-round preparation | 0.044608 |
| Complete closed interval | **29.679408** |

The mean per-round slowest-rank interval is 29.681317 ms. The target graph
contains 2,060 kernels: kernel union 16.933081 ms, no-kernel intervals
5.681320 ms. Of the latter, 3.661182 ms overlaps the long target
`cudaGraphLaunch` API. This is evidence of substantial profiling/launch
interference, not 3.66 ms of newly removable model computation. It cannot
replace or be rescaled to the accepted 23.849974-ms unprofiled endpoint.
Target dense service remains 7.320460 ms; the largest repeated grid groups
are HC down `(8,3,20)` / 1.525945 ms and HC up `(8,80,1)` / 1.354346 ms.
Service sums can overlap and are not an additive critical-path table.

The draft wall excludes the final combine kernel and closes independently:

| Draft category, rank0 | Exclusive ms |
| --- | ---: |
| Four full-vocabulary projections | 1.812865 |
| MoE projections | 0.690810 |
| Other dense projections/reductions | 0.614226 |
| Sampling, state and elementwise | 0.529273 |
| TP communication including waits | 0.524088 |
| Attention and indexer | 0.342716 |
| HC postops and fused M1 projections | 0.201451 |
| Overlapping kernel families | 0.065627 |
| No recorded kernel | 0.481110 |
| Total | **5.262167** |

The first admitted draft screen preserves original FP16 weights, sequential
FP32 FMA, and W2's router multiplication before FP16 storage. On GPU1 with
real rank0 checkpoint weights, five activation scales, changing routes and
poisoned outputs, every tested variant is bit-exact. Selected projections:

| Projection | Tuned Triton us | Selected native us |
| --- | ---: | ---: |
| M1 W13, direct vector8 loads / 64 columns | 105.395 | 61.395 |
| M1 W2, tile64 / K64 | 24.960 | 13.024 |
| M5 W13, tile32 / K128, captured fixed routes | 170.989 | 88.070 |
| M5 W13, same tile, captured code routes | 172.147 | 96.800 |
| M5 W2, tile64 / K64, captured fixed routes | 95.258 | 52.294 |
| M5 W2, same tile, captured code routes | 94.874 | 51.130 |

These are isolated component results (`draft_moe_exact.{json,cu,py}`), not
complete-round savings. M5 captured routes come from the first draft pass;
continuation routes are synthetic. The implementation is now registered in
the normal `_C` build, guarded by opt-in `VLLM_SM70_MTP_MOE_FP16_EXACT=1`,
exact M1/M5/E512/H2560/I160/top10 geometry, FP16 contiguous tensors, the
audited BM2 tile, SM70, and no bias/quantization/sorted assignment. Unsupported
inputs retain their existing route. It adds no persistent weight copy and
leaves target batch decode unchanged. Normal-build component gates are
reported below; do not promote projection times to an endpoint result.

Reject the original-layout GDN screen without another engine run: all 36
real rank0 layers remain exact, but the M5 chain costs 1.104200 ms packed
versus 1.510728 ms original / 1.444454 ms with read-only loads; M10 costs
1.126441 versus 1.929830 / 1.706721 ms. Record
`gdn_input_original.{json,cu,py}` so this layout change is not repeated.

### Normal-build draft gates and modular dispatch

The normal `_C` build has SHA256
`647649e8d5ee1ef6ab0952e8e29b8967967ca647a27bd86d55af62e58ca65179`.
Its dependencies resolve only to the declared Torch/CUDA runtime. The
draft test suite passes 14 cases (9 metadata/default tests and 5 GPU tests).
The normal-build benchmark covers all four real TP weight slices, M1/M5,
W13/W2, six activation scales, changed routes and poisoned graph outputs:
all 16 projection cases have zero FP16-bit differences.

```bash
.venv/bin/python -m benchmarks.kernels.benchmark_sm70_mtp_moe_fp16 \
  --model /data/models/RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --out .artifacts/draft_moe_native_checkpoint.json
```

These are isolated projection times across the four weight slices on one
leased GPU, not four-rank engine timings:

| Projection | Tuned Triton us | Native us |
| --- | ---: | ---: |
| M1 W13 | 105.267--105.434 | 58.931--63.526 |
| M1 W2 | 24.858--25.472 | 12.922--13.536 |
| M5 W13 | 173.338--174.502 | 101.875--109.670 |
| M5 W2 | 95.206--95.706 | 54.394--55.328 |

The first model candidate revealed that `TritonExperts.apply` bypassed the
common dispatcher and called the low-level Triton implementation directly.
It never selected the new operator and was stopped (exit -15); retain
`mtp_draft_exact_candidate` as an invalid route-hit attempt, not performance
evidence. Both modular projections now use the common dispatcher. W13
receives the routing-weight pointer with multiplication still disabled.
Two additional GPU tests exercise this actual modular entry at M1/M5,
assert both native operator calls, and preserve the entire MoE output bits
under changed-input CUDA Graph replay. This correction is Python-only;
the native extension hash is unchanged.

The same-GPUs0--3 control on this native build completes at
23.955188/23.945170 ms, median **23.950179 ms**, with the three previously
qualified optimizations enabled and the new draft flag disabled. The
corrected candidate is queued for the same physical GPU group. Keep this
new pair distinct from the earlier 23.849974-ms build identity; no new
complete-round improvement is claimed before its candidate gate finishes.

### Additional HC screens rejected before model testing

`hc_up_original_shared` preserves gate and mixed-output bits on eight real
weight pairs and five scales, but staging original-layout weights in shared
memory regresses M5 from 13.664 us to 18.052--21.776 us per pair and M10
from 14.424 us to 20.924--23.592 us. Reject all four tile/padding variants.

`hc_up_cache` compares identical original-layout arithmetic with default
cache configuration and `cudaFuncCachePreferL1`/zero preferred shared
carveout. M5 changes 12.696 -> 12.716 us; M10 15.528 -> 15.588 us, with
zero bit differences. L1 preference supplies no gain. The separate M5
fusion is only about 0.99 us faster than the unfused reference and regresses
M10; this is no qualified engine improvement. The cache hypothesis follows
the [Volta unified-cache documentation](https://docs.nvidia.com/cuda/volta-tuning-guide/index.html#unified-shared-memory-l1-texture-cache),
not an observed hardware counter. NCU remains blocked by
`ERR_NVGPUCTRPERM`. Retain both screen reports and skip engine runs for them.

### Full-vocabulary head screen

`draft_head_screen` uses the real rank0 FP16 head `[62080,2560]`, M1,
six synthetic activation scales and alternating graph measurements on GPU4.
The only bit-exact cuBLASLt heuristic costs 475.104 us versus 474.944 us for
the existing full projection. Other heuristics and the packed full-head
operator change FP16 logits, so they are rejected without timing or engine
tests. This screen does not justify changing the target's batch GEMM.

The existing raw/packed fused top1 operators take 399.264/418.240 us and
match the selected value/index on these six inputs. That is a narrower
check than full-logit bit parity. The raw operator uses `__hmul2`, rounding
each product to FP16 before summation, so it fails this task's arithmetic
contract outright. The packed operator also changes accumulation order;
historical model tests have reported output divergence. Neither shortcut is
enabled by this change. Preserve the full-vocabulary arithmetic.

### Endpoint resource handoff

The modular fix is committed at `6bcffbb7968711babe44dfcf0986500377fb13ab`.
GPU0--3 remained occupied by another evaluation, so an additional same-source
control completed on GPU4--7: **23.880598/23.867778 ms**, median
**23.874188 ms**. Its emitted tokens and acceptance records all match the
GPU0--3 control. Keep timings paired within a physical GPU group.

`mtp_draft_exact_candidate_4567` failed before weight loading: another task
started on GPU4--7 after the launcher's idle check, leaving 24.38/31.73 GiB
free, below the unchanged 0.95 memory-utilization request. Retain this failed
startup; do not reduce memory utilization or terminate the foreign workers.
`mtp_draft_exact_candidate_auto` subsequently used GPU4--7 and its matching
control, as reported below. No additional control was run.

The first isolated NVIDIA Nsight Systems **2025.3.2.474** attempt completes
its 129-token model request and all four profiler RPC start/stop calls, but
prints **No reports were generated**. It supplies no kernel timing evidence.
Its package SHA256 is
`c7cfe27e2250eb91e1a67e7feb5f2c490c7f598e3b3a3d047aff000bc49f9d6b`.
A small single-GPU reproduction finds the cause: this package loads CUPTI
13.1, reports `CUPTI_ERROR_INVALID_DEVICE`, and produces zero CUDA events.
Whole-process tracing yields only NVTX events; API-range tracing yields no
report. Retain `nsys_smoke_{api,all}2532` and the failed model capture log.

An isolated **2025.3.1.90** CLI package, SHA256
`d2484ad0faf6831b11fa0bf73c54232d9ea8beafb50414019e6ba299c4ed5718`, loads
CUPTI 12.9 instead. The identical API-range smoke captures all **40 expected
kernels** (20 graph replays of GEMM + sine) and 20 NVTX ranges. It still warns
about the driver's CUDA 13.0 version and possible incomplete events at range
closure. Whole-process smoke captures its 47 expected warmup/capture/replay
kernels with no incomplete-events warning. This local count check justifies
one model capture, not a general guarantee of trace completeness.

The source build and CUDA 12.8 runtime remain unchanged; both profilers live
only in the task cache. The model capture uses `--cuda-event-trace=false`,
node tracing, all-rank flush and `--kill=none --wait=primary`.
[Nsight's release notes](https://docs.nvidia.com/nsight-systems/ReleaseNotes/index.html)
exclude Volta from 2025.4 and later, so upgrading to those versions is not a
recovery route. Do not interpret an unrecorded interval as proven GPU idle.

### Complete draft-MoE endpoint gate

`mtp_draft_exact_candidate_auto` obtains GPU4--7 after the unrelated service
exits. Compare it only with `mtp_draft_exact_control_4567`: both use source
`6bcffbb7968711babe44dfcf0986500377fb13ab`, the same native `_C` hash, GPU
group and engine contract. The only environment difference is
`VLLM_SM70_MTP_MOE_FP16_EXACT=0/1`; retained source patches contain only this
report. Both processes exit successfully. Workers log the actual native
route. Model memory stays 23.61 GiB and KV capacity stays 122,631 tokens.

| Ordinary request | Control round ms | Exact draft round ms |
| --- | ---: | ---: |
| Fixed 8192/513 A | 23.880598 | 23.632925 |
| Fixed 8192/513 B | 23.867778 | 23.682790 |
| Fixed median | **23.874188** | **23.657858** |
| Natural EOS code, 284 tokens | 24.957797 | 25.067507 |
| Natural EOS arithmetic, 329 tokens | 24.920565 | 25.013588 |
| Natural EOS explanation, 421 tokens | 24.962826 | 24.962961 |

The fixed-fixture saving is **0.216330 ms (0.9061%)**. The natural requests
show no gain (two are 0.110/0.093 ms slower); do not claim a universal speedup
or enable the flag by default. The below-20-ms objective remains
**3.657858 ms away** on the fixed fixture.

Every token ID, finish reason and acceptance statistic matches in both
fixed repeats and all three natural requests, as well as warmup and the
separate phase request. The fixed fixture still emits 513 tokens with 335
draft rounds and 177 accepted drafts, position counts `[125,36,10,6]`.
Natural outputs stop normally. The generated LIS function passes empty,
duplicate, negative and example inputs; the arithmetic answer is 240 km /
68.57 km/h, and the signed-division example reconstructs its dividend.
These are three text-health cases, not broad model-quality qualification.

Separate CUDA-event phase measurements show per-rank draft means changing
from **5.011584--5.034362 ms** to **4.879756--4.902546 ms**. On the same rank1,
draft is 5.011584 -> 4.902546 ms and target verifier GPU is
19.008792 -> 18.872125 ms. The highest-mean-total rank changes from rank1 to
rank3; do not mix category maxima into a synthetic round. Instrumented phase
requests and the upcoming node trace do not replace the ordinary endpoint.

Retain `mtp_draft_exact_comparison.{json,log}`, both endpoint reports/logs,
contracts/source patches and `draft_moe_output_health.json`. The component
gains only partly appear in the engine; the next node capture checks the
native projections' actual scheduling and the remaining dense/HC work.

### Successful current-build node capture and remaining bottlenecks

`mtp_draft_exact_trace2531` uses the same source, normal `_C`, GPU4--7 and
all four flags as the qualified candidate. Generation and profiler exit 0;
all four worker start/stop RPCs return success. Its 129 tokens and acceptance
(85 drafts, 43 accepted, positions `[16,13,8,6]`) match the preceding capture.
No owned GPU workers remain after shutdown. Keep the report, SQLite export,
contracts, diagnostics and analysis JSON under that artifact prefix.

Four ranks each contain 212,131 recorded kernels over the whole range.
All **84 closed steady rounds** contain exactly **2,060 target kernels** and
**298 draft kernels**. Every rank and selected round contains exactly one
native M5 W13, three native M1 W13 and four native W2 calls. The new path is
actually used; no draft step silently falls back to the old Triton projection.
Driver-version and possible incomplete-CUDA/NVTX-events diagnostics remain.
Stable selected counts support these route observations; they do not prove
that every unrecorded interval is idle.

Rank0's *profiled* cycle closes at **29.391136 ms** (mean per-cycle slowest
rank 29.393058): metadata-to-target 0.845020, target graph 22.668388,
target-to-gather 0.009949, sampling/state handoff 0.788612, four drafts plus
combine 5.032932, next-round preparation 0.046235. The target graph contains
16.806997 ms of recorded-kernel union and 5.861392 ms without a recorded
kernel; 3.834410 ms of the latter overlaps its 4.116641-ms graph-launch API.
This is **not a new 29-ms endpoint** and is not rescaled or subtracted from
the measured 23.657858-ms ordinary round. Separate phase observations
remain around 19 ms for target verification and 4.9 ms for the four drafts;
their enclosing request is 26.885772 ms/round, so they are diagnostic rather
than a calibrated low-overhead decomposition of the ordinary endpoint.

The rank0 draft interval, excluding final combine, closes independently:

| Current traced draft category | Exclusive wall ms/round |
| --- | ---: |
| Four full-vocabulary heads | 1.881527 |
| Other dense projections/reductions | 0.629245 |
| Draft MoE projections | 0.613238 |
| Sampling, state and elementwise | 0.544494 |
| TP communication and dependency waits | 0.405687 |
| Attention/indexer | 0.343567 |
| HC postops/fused M1 projections | 0.199544 |
| Overlapping kernel families | 0.067603 |
| No recorded kernel | 0.343043 |
| **Draft wall** | **5.027947** |

Draft steps 0/1/2/3 close at 1.384795/1.266408/1.175003/1.201741 ms.
The first step is M5; subsequent steps are M1. Each local vocabulary head
still reads a nominal 317,849,600 weight bytes and takes about 0.469--0.471 ms
on rank0. The byte/time quotient is about 675--678 GB/s of *nominal weight
traffic*, not a measured memory-bandwidth counter. The already rejected
non-exact head alternatives cannot establish an available speedup.

Target's remaining dense service is **7.565656 ms**, including HC down
97 calls / 1.627876 ms and HC up 98 calls / 1.394273 ms. HC postops add
1.055227 ms service. Other target services include grouped experts 2.498326,
QSA 1.805013, packed GDN input 1.299056 and GDN core 0.887810 ms. These service
sums overlap and must not be added to the target wall. This still makes the
repeated target dense/HC operations the main place to pursue the several
milliseconds needed for <20-ms complete rounds.

#### M1 W13 long-tail investigation

The new native M1 W13 trace is not uniformly as fast as its isolated screen.
Across 252 calls per rank, medians are 72.559/62.064/63.248/71.584 us, but
means are 146.829/97.180/87.968/123.266 us. **171 of 1,008 calls exceed
250 us**; maxima reach 306.748--340.413 us. The slow calls can overlap only
about 4--8 us of shared-expert work on another stream, so that recorded work
alone does not explain a roughly 260-us tail. This is a trace observation,
not yet a proven unprofiled scheduling or hardware root cause.

Targeted follow-up screens avoid another model startup:

- `draft_moe_context_screen`: real rank0 weights, 96 changing routes/scales
  in each warm/cold condition, alternating arm order and one-replay CUDA
  event timing. Native direct64 means 80.811/88.544 us; maxima 89.088/98.304 us.
  A 64-MiB cache-clearing memset does not reproduce the 300-us tail.
- `draft_moe_context_screen_nsys`: the same screen under 2025.3.1 node
  tracing also does not reproduce that tail; cold direct64 mean 89.681 us,
  maximum 100.384 us. Nsight on/off alone is insufficient to explain it.
- `draft_moe_tlb_screen`: a 24-GiB pressure tensor is touched every 64 KiB
  before each projection, outside the timed window. Across 48 changed
  routes, direct64 mean/max are 88.021/95.232 us. This artificial address
  pressure also does not reproduce the model trace's tail. It is not a
  measurement of TLB misses or proof that model residency is irrelevant.
- All screened outputs match the Triton control bit for bit. Shared32/K256
  gives 75.904 us under the last condition, but its earlier warm screen is
  essentially tied with direct64. This is not a full-model admission; no
  further kernel switch or default change is made from these component data.

The next discriminating measurement is graph-internal CUDA-event timing of
M1 W13 in the resident engine without CUPTI, with actual route IDs and rank
arrival times, to establish whether these tails exist in ordinary execution.
It should reuse the qualified workload and control instead of repeating the
full endpoint suite. Then choose a scheduling change only if that evidence
supports it. In parallel as an optimization direction, target batch dense/HC
fusion and launch-count reduction have substantially more total headroom
than another isolated draft tile sweep. Historical rejected shared staging,
packing, prefetch and arithmetic variants remain closed until a concrete new
mechanism addresses their recorded failure. No achieved SM/HBM utilization
is claimed while Nsight Compute counter access is unavailable.

The task-local `mtp_kernel_observer` implements that follow-up using external
CUDA event nodes around the unchanged normal `_C` projection, plus small
route/counter copies after the end event. A standalone replay smoke passes:
three event pairs, three changed-route replays, exact route/counter tracking
and bit-identical outputs. This observer synchronizes after each proposal,
so its whole-request latency would remain diagnostic only. It does not
replace a kernel or alter routing/weights.

The first resident attempt, `mtp_draft_kernel_events`, exits 1 before weights
load: a foreign GPU4--7 job appears after preflight, leaving only
20.43/31.73 GiB free on one rank versus the unchanged 30.15-GiB request.
No resident event result exists from this failed attempt. At this checkpoint
both four-GPU groups are occupied; the bounded retry waits for an idle group.
Do not treat the observer's standalone smoke as evidence about model tails.

#### Resident graph-event observation, without Nsight

After GPU4--7 becomes idle, `mtp_draft_kernel_events_retry` completes with
exit 0 on the same runtime source/build. All three requests' token IDs,
finish reasons and acceptance statistics match `mtp_draft_exact_trace2531`.
Actual expert IDs also agree across all four ranks. The resource-failed
attempt above is retained separately.

There is an important scope correction: the runtime captures **one M1
single-step graph and replays it three times** in `multi_step_decode`.
Consequently, each worker reports one event pair and a counter increment of
three per proposal. The event pair and route buffer retain only the **last
M1 step**. This observation does not time the first two M1 steps separately.
The standalone smoke's three distinct captured calls had three event pairs;
it did not establish how the resident engine reuses a captured graph.

Each rank records 87 proposals. Aligning the full-range M1 call order to the
84 closed trace rounds gives the following final-step comparison:

| TP rank | Node trace mean/max us | No-CUPTI graph-event mean/max us |
| --- | ---: | ---: |
| 0 | 122.073 / 338.844 | 75.447 / 82.944 |
| 1 | 85.656 / 306.973 | 65.938 / 70.656 |
| 2 | 81.641 / 305.628 | 66.280 / 72.704 |
| 3 | 113.559 / 335.421 | 75.106 / 82.944 |

The final-step node trace has 39/336 calls above 250 us; the aligned event
observation has **zero**. Therefore, do not treat the node trace's 300-us
W13 calls as an established ordinary-kernel bottleneck. These runs differ
in instrumentation: the graph-event observer adds event/route/counter nodes
and fences each proposal, so this is evidence of measurement sensitivity,
not proof that CUPTI alone caused every tail. The trace's overlapping runtime
APIs do not distinguish slow and fast W13 calls: both overlap an existing
`cudaStreamSynchronize`, so that correlation also supplies no causal fix.

Retain `mtp_draft_kernel_events_retry{,_analysis}.json`, its observer source
hash, `mtp_draft_last_step_profile_comparison.json` and
`mtp_draft_exact_trace2531_native_api_overlap.json`. The next observer should
collect the pair immediately after **each** `decode_cudagraph_manager`
`run_fullgraph` call, before the following replay overwrites it, preferably
within a resident engine that stays open for diagnostic RPCs. Do not repeat
an identical end-of-proposal observer expecting three independent timings.
All owned workers shut down; subsequent GPU4--7 owners belong to another task.
The qualified 23.657858-ms endpoint and the <20-ms remaining objective are
unchanged by this diagnostic run.

### Reuse the batch router key and remove MTP QKV copies

The next screens preserve the 23.657858-ms endpoint contract. They do not
rerun no-MTP, change draft length, enable reduced precision, or substitute
component sums for a new complete-round measurement.

- The existing M1 router's lossless 32-bit key is also exact at M1/2/4/5/8/10/16.
  Tests cover all 65,536 FP16 payloads at the actual M specialization and
  24 changing-input graph replays per width. FP32 weights, expert IDs and
  rank-major source indices match the original 64-bit sort exactly. Keeping
  eight warps preserves the normalization reduction. Forty-eight distinct
  M5 calls improve 0.254016 -> 0.204256 ms; M10 improves
  0.254496 -> 0.204800 ms. Extend only the existing admitted FP16 M<=16
  dispatcher; this approximately 0.05-ms component saving is not a measured
  endpoint gain. Retain `router_batch_packed.{py,json,log}`.
- Ordinary MTP currently copies three flattened Q/K/V splits and concatenates
  them before invoking the already fused sigmoid recurrence. Reuse the
  existing mixed-QKV loader with the same BV32/four-warp math, FP32 state,
  accepted-state selection and FP16 output. In 36 distinct layer buffers,
  ten changing-input/state graph checks per width preserve every output and
  stored-state bit, including untouched slots. The current copy path, one
  fused packing copy and direct mixed loader measure respectively
  **1.097771 / 0.642859 / 0.615125 ms at M5**, and
  **1.256832 / 0.797653 / 0.784512 ms at M10**. These are five alternating
  component timing pairs, not new engine speed. Retain
  `gdn_mtp_mixed_loader.{py,json}` and `gdn_mtp_mixed_loader_retry2.log`.

The existing opt-in `VLLM_SM70_FUSED_SIGMOID_MIXED_QKV` now admits this small
SM70 pure-spec verifier geometry (four QK heads, twelve V heads, D128,
FP16 input, FP32 state and 2..16 rows). DFlash2's separate numerical bridge,
tree verification, mixed prefill/decode and unsupported shapes keep their
previous dispatch. Core output allocation/merge and padding remain as before.
No new native library or persistent weight storage is required.
The CPU route gate passes 18 cases; ten GPU integration cases are pending at
this checkpoint. They exercise the public router and real GDN convolution,
accepted-state selectors, both conv-cache layouts and output canaries.
Whole-model token/acceptance and complete-round gates remain pending.

#### Further HC evidence before another model run

The original-layout split-stage down hypothesis retains twenty K512
FP16-rounded partials and ordered FP32 reduction, with reduction/SiLU fused.
Eight real checkpoint pairs and seven input scales are bitwise exact. It is
nevertheless slower: the best M5 case is 17.024 -> 18.702 us and M10 is
16.098 -> 20.422 us. Two/four-warp variants are slower still. Reject these
schedules without an engine run; retain `hc_down_split_stage.{cu,py,json,log}`.

Existing trace launch metadata identifies the cuBLAS HC down kernel as
`16x16_64x2_tn_align8`, 480 CTAs, one warp/CTA, 114 registers/thread and
8,704 shared bytes/CTA. The new one-warp screen had 220 CTAs, whereas its
slower two-warp version had 120. A follow-up screens 420/840 independent
CTAs by masking unused Volta MMA quads while preserving every K512 partial;
compiled variants use 62 registers and no shared memory or spills. This is
a parallelism hypothesis, not measured occupancy or bandwidth. Its GPU
screen remains pending.

[PR #504](https://github.com/1CatAI/1Cat-vLLM/pull/504)'s existing batch HC
sharding changes the original replicated GEMM association and previously
saved only 0.174--0.277 ms in a different no-MTP endpoint. It cannot be
blindly ported as an exact MTP improvement. A separate component screen
reuses [PR #704](https://github.com/1CatAI/1Cat-vLLM/pull/704)'s unchanged
register-level up/mix kernel at fixed source
`590d78e0b168d38703f94333d240606900821f85`, with this task's original M5/M10
precision settings. Its result is pending; the research extension is not
loaded by model workers.

GPU ownership checks rejected the first mixed-loader attempt and its first
bounded retry before GPU work. The second retry acquired idle GPU4 and
completed; other services were left untouched. Subsequent integration/HC
gates share a finite queued lease. Preserve resource failures as such rather
than counting them as correctness failures or speed samples.

Source `8a99ccb4ea37e820840e9b7c3d1d45b335dfc94f` passes all changed-file
pre-commit hooks, including mypy. The combined 15-minute GPU wait then expires
because all eight cards remain under foreign leases/compute owners; none of
the ten new GPU integration cases or the two pending HC screens starts.
Retain `mtp_pending_gates.log`, `mtp_loader_cpu_gate.log` and
`mtp_loader_precommit.log`. No owned GPU worker or waiting process remains.
Resume the prepared `run_mtp_pending_gates.py` under an idle-card lease, inspect
all three exit/result files, and only then qualify admitted runtime changes
with the frozen full-model contract. Do not rerun the successful component
screens or report a new endpoint/trace from this resource failure.

#### CPU-only recurrence layout audit

The retained target trace launches the fused sigmoid recurrence as
`grid=(1,4,12)`, 128 threads/CTA, 119 registers/thread and 256 shared bytes:
only 48 CTAs for 80 V100 SMs. This establishes a limit on independent
parallel work, not achieved utilization or the size of an available gain.

A CPU-only Triton compilation isolates **BV32 -> BV16 while retaining four
warps and three stages**. Both TTGIR programs use the same K reduction
layout: state tile `sizePerThread=[1,4]`, `threadsPerWarp=[1,32]`,
`warpsPerCTA=[4,1]`; Q/K normalization keeps its original four-warp 1-D
layout. All four FP32 reduction regions retain the same axes/layouts.
The isolated compiled pair uses 117 -> 80 registers, one barrier, 256
shared bytes and zero spill/stack bytes. Its M5 grid grows from 48 to 96
CTAs. The isolated 117-register build is not relabeled as the traced
119-register binary, and neither compile result is a timing or bit-parity
result.

This differs from the previously rejected BV8/recurrent-schedule overrides;
no global runtime schedule has changed. Retain
`gdn_value_tile_compile.{py,json,log}`, both emitted IR/PTX/cubins and the
prepared `gdn_mtp_value_tile.py` GPU screen. That screen compares every
FP16 output and FP32 state bit over changing inputs, accepted selectors and
live slot zero, then times only an exact candidate. The pending gate runner
now has four independent jobs, including this screen. Admission still
requires GPU evidence before any runtime integration.

The subsequent 600-second lease wait (`mtp_pending_gates_retry1.log`) also
expires before any CUDA work. Both foreign TP4 groups remain occupied;
all four prepared jobs and whole-model qualification are pending. The wait
process has exited. The accepted 23.657858-ms endpoint is unchanged.

### Capture representativeness audit, 2026-09-27

The current trace is retained for route identification and recorded ordering,
but is **not admitted for absolute normal-execution latency attribution**.
Checking the saved request metrics independently of the SQLite parser gives:

| Measurement | Complete round ms | Scope |
| --- | ---: | --- |
| Qualified ordinary endpoint | 23.657858 | Median of two 8192/513 requests |
| Node-captured request's own decode metrics | 29.350420 | 8192/129, 85 drafts |
| Same node trace, closed intervals | 29.391136 | Rank0, 84 closed rounds |
| Separate CUDA-event phase request | 26.885772 | 8192/513, 335 drafts |

The captured request and trace agree within 0.139%, supporting that the
captured application really ran slower. This does not identify the cause of
the 24.06% difference from the qualified endpoint: output length and warmup
history differ, and capture overhead has no matched off/on/off control.
Do not attribute the entire difference to CUPTI or subtract it from kernels.

The phase observer's enclosing request is 13.64% slower than the ordinary
endpoint. Source inspection finds an event synchronization after the target
phase and another at round reporting. Worker-local timing averages exclude
some enclosing request cost; they cannot be added into a purported exact
23.657858-ms decomposition. This observer was off during the node trace.

An independent correlation-ID join within the same 84 closed rank0 windows
finds 84 target graph-launch calls with 2,060 kernels each, mean host API
duration 4.116641 ms. Draft graph-launch calls carry 81 nodes / 0.232171 ms
and 65 nodes / 0.163631 ms. The target's 3.834410 ms of unrecorded-kernel
intervals overlapping graph launch remains a diagnostic lead, not confirmed
idle time or removable overhead. See `mtp_capture_graphlaunch_audit.json`.

The original Nsight 2025.3.1 tool uses CUPTI 12.9 and warns about the CUDA-13
driver interface. The previously attempted 2025.3.2/CUPTI-13.1 smoke emits
`CUPTI_ERROR_INVALID_DEVICE` and no CUDA events; do not repeat that failed
upgrade or claim that a version change has fixed the capture. NVIDIA's
[CUDA graph tracing documentation](https://docs.nvidia.com/nsight-systems/UserGuide/)
also distinguishes lower-overhead whole-graph tracing from potentially
expensive node tracing. That documented possibility is not a causal result
for this workload.

Prepared `run_mtp_trace_calibration.py` uses the current owned runtime and a
fresh same-source clean control, then graph and node captures only while the
preceding control passes. Every arm warms the full 8192/513 workload and
measures identical 513-token requests before/during/after collection. It
keeps the phase profiler off, records host graph replay duration without
CUDA events or per-round synchronization, and checks every token, finish
reason and acceptance statistic against the saved fixed fixture. A 3% bound
on control drift, inactive injection overhead and capture perturbation is a
local admission gate, not an assumed profiler guarantee. The newer router
source is explicitly recorded; a new run is not relabeled as source
`6bcffbb796`.

The first launch attempt fails before CUDA at the per-GPU lease check.
GPU4--7 had briefly released memory between foreign jobs but were reserved
by PID2418031 (`serve_with_lock.py`); workers2418667--2418670 then loaded.
No calibration inference, new trace, owned GPU worker or waiter exists at
this checkpoint. Syntax checks pass for the prepared artifact scripts.
Retain `mtp_capture_representativeness_audit.json` and the calibration
scripts. Corrected plot titles explicitly mark the old trace uncalibrated.

### Calibrated deferred-event trace, 2026-09-27

The subsequent queue obtains GPUs4--7 after foreign jobs exit naturally.
No foreign service is stopped. All arms use source
`e7df523a3c68c6d1a586d34adb54f84fc1e63c1d`, normal `_C` SHA256
`647649e8d5ee1ef6ab0952e8e29b8967967ca647a27bd86d55af62e58ca65179`,
the same environment, physical GPUs4--7, and warmed 8192/513 fixture.
The four qualified acceleration flags stay enabled; mixed-QKV stays disabled.
The batch router source is newer than the paired `6bcffbb796` endpoint above.
These controls qualify measurement and the fixed fixture; they are not a
new paired optimization speedup or a new natural-EOS quality campaign.

| Arm | Before ms/round | During ms/round | After ms/round |
| --- | ---: | ---: | ---: |
| Independent clean process | 23.686521 | 23.704239 | 23.700977 |
| Nsight whole-graph process | 25.176410 | 25.350714 | 25.460508 |
| Deferred CUDA events, no Nsight | 23.602294 | 23.799931 | 23.691980 |

The clean mean is **23.697246 ms**. Whole-graph collection changes only
0.127% against its adjacent controls, but its collection-disabled process
already differs from clean by **+6.841%**. It fails admission; no node arm
is launched. Its metadata markers close 335 cycles/rank at
25.354080--25.354105 ms, agreeing with the captured request. One target-graph
record is missing on ranks0/2/3 at cycle316/310/305. Do not drop these cycles
silently or reuse this graph trace for normal absolute attribution.

The deferred observer precreates event handles and reads them after the
request. It overrides only the optional diagnostic profiler's event fence
and reporting; real model, output, and collective synchronization is intact.
Each graph invocation receives unique event handles, including all three
replays of the same M1 draft graph. No per-round timing read or diagnostic
synchronization occurs. The original phase profiler flag remains zero.
The event request differs by **+0.646%** from its own off-control mean
(23.647137 ms), and **+0.433%** from the independent clean mean. Off-control
drift is 0.380%. All three requests preserve all 513 output IDs, finish
reasons, and the complete acceptance statistics: 335 drafts, 177 accepted
tokens, positions `[125, 36, 10, 6]`.

Each rank records **336 target-forward starts and 6,720 unique events**.
Those starts close **335 consecutive intervals**. The initial parser
incorrectly required the number of starts to equal the request draft
counter; its failed log and source are retained. The corrected parser
distinguishes those quantities, validates all four ranks and five replay
envelopes per invocation, and retains the final invocation as raw data and
the final closing boundary. This is a CPU-only parser correction: no raw
measurement changes and no GPU rerun.

All components below come from the same 335 rank0 intervals. Event envelopes
include GPU execution, host submission gaps, and dependency waits; they are
not isolated kernel service times. No timing is rescaled to the endpoint.

| Rank0 phase | Mean ms | p50 ms | p90 ms | p99 ms |
| --- | ---: | ---: | ---: | ---: |
| Target forward | 17.781912 | 17.795044 | 17.914063 | 18.081792 |
| Sampling and state handoff | 0.780647 | 0.781250 | 0.785156 | 0.791504 |
| Four drafts | 4.627054 | 4.554688 | 4.800293 | 5.014648 |
| Next-round preparation/scheduling | 0.593635 | 0.589844 | 0.607178 | 0.655273 |
| Complete closed cycle | **23.783247** | 23.781738 | 24.015137 | 24.270752 |

The four rank-local cycle means are 23.783247, 23.783418, 23.783387,
and 23.783259 ms, within **0.071%** of the event request's 23.799931 ms.
Selecting the longest whole interval at each aligned ordinal gives
23.813001 ms. Its components, each taken from the same selected rank, are
17.730917 / 0.773139 / 4.635154 / 0.673792 ms. This is kept separate from
fixed-rank means; independently maximizing each category would not close.
Different GPUs retain separate event-clock origins.

Rank0's target replay envelope is 17.741128 ms; forward work around it is
0.040784 ms. The four draft replay envelopes are **1.329739, 1.081356,
1.058597, 1.051968 ms** (M5, then three M1), plus 0.105394 ms outside those
replays. Within the 0.780647-ms sampling/handoff interval, direct event
pairs measure 0.754719 ms sampling and 0.017420 ms state update. Remaining
gaps stay explicit in the enclosing interval. Prefill/TTFT are excluded;
the sampled request's pure decode is 7.972977 s for 512 steady emitted
tokens (64.216917 tokens/s, 15.572221 ms/emitted token).

Host replay observation supports a profiler-sensitive launch path: ordinary
target replay host means are 0.074--0.101 ms/rank before event collection,
versus 1.048--1.422 ms/rank in the inactive Nsight process. These are CPU
API durations, not GPU graph costs. This does not assign every difference
in the old node trace to one cause. Node-level kernel attribution and
SM/HBM counters remain unqualified; do not carry old GEMM service sums into
the calibrated wall table.

The target forward accounts for about 74.8% of rank0's complete cycle and
draft for 19.5%. The next measurement should split HC, GDN, experts, and
attention inside the target while retaining this perturbation/closure gate.
The <20-ms goal remains unmet. No new runtime optimization, default, wheel,
or model numerical contract is introduced by this measurement.

Retained artifacts in the owned worktree's `.artifacts/`:

- `mtp_capture_calibration_{none,graph,deferred}_20260927` reports,
  commands/contracts, logs, and exit codes;
- `mtp_capture_calibration_graph_20260927_audit.json`, including the rejected
  graph trace's missing records and diagnostic warnings;
- `mtp_deferred_{worker,probe}.py`, `run_mtp_deferred_trace.py`, and the
  CPU plumbing check;
- `mtp_deferred_independent_audit.json`, verifying identities, all output
  IDs, monotonic timestamps, and unique handles independently of plotting;
- `mtp_capture_calibration_deferred_20260927_{wall,chrome}.json`, all raw
  intervals and a Chrome/Perfetto event trace;
- `calibrated_mtp_trace_view/trace_breakdown.{png,svg}` and its selection
  manifest. The shown raw interval is round252, chosen by distance to the
  median over all 335 rank0 intervals.

The model processes and resource waiter exit after measurement. The first
automatic postprocessor reports the strict parser failure described above;
CPU analysis/plotting are then completed and reviewed without another
model load. The current report/status supersede that historical failure.

### HC planning audit and pending component screens, 2026-09-27

At the user's request for an estimated current forward composition, a CPU
analysis reconstructs module ownership in the two retained node traces
`mtp_draft_exact_trace2531` and `mtp_stack_trace_flush`. Every rank/window
has 97 HC norm-to-gate boundaries and 2,060 kernel records. Timestamp
unions remove same-module concurrent work; TP collectives and cross-module
overlap stay separate. This corrects two coarse classifications: HC includes
its dense projections, and one of the 98 `(8,80,1)` GEMMs belongs to PLE.

The planning budget inherits the newer old trace's rank0 busy intervals:
HC 4.293, MoE 4.307, GDN 3.774, QSA 2.908, PLE with preparation 0.607,
TP communication/dependency waits 0.909 ms. Current target replay minus
those historical busy intervals leaves a 0.934-ms balancing residual.
**This assumes transferable kernel times/overlap and is not a newly measured
internal trace. The residual is not measured idle time.** The old 129-token
and current 513-token windows differ; no uniform scaling or independent
rank-category maxima are used. Both historical captures' four ranks support
HC/MoE as the largest modules, followed by GDN and QSA. Retain
`mtp_forward_module_estimate_20260927.{json,md}`, its CPU parser and plot.

HC's ordinary M5 path uses replicated FP16 weights: down is
`[5,10240] x [10240,336]`, up is `[5,320] x [320,10240]`, and the final
mixer's down has 320 outputs. The 96 regular modules plus final mixer
contain 1,302,855,680 nominal weight bytes per rank per round and about
6.514 GFLOPs at M5, before postops. This is weight-volume accounting, not
measured HBM traffic. The existing exact TP4 HC sharding implementation
requires `x.shape == (1,10240)`; M5 uses the ordinary replicated fallback.
Extending output-dimension sharding to batch shapes is a candidate to avoid
duplicated work, but it must preserve each row's original reduction and
materialization boundaries and account for the additional gathers. The
previous PR #504 result does not establish that new route's parity or speed.

GPU4 later becomes idle. The existing single-card lease runs only the two
previously unexecuted HC screens, without a model load or profiler:

```bash
MTP_RESEARCH_GPU=4 MTP_WAIT_SECONDS=0 .venv/bin/python \
  .artifacts/run_research.py .artifacts/run_hc_evidence_20260927.py
```

Source is `63ad3e255b` (runtime unchanged from `e7df523a3c`), Torch
2.10.0+cu128, CUDA toolkit 12.8.93, driver 580.173.02. The normal native
extension SHA remains the calibrated value above. Both arms use eight real
checkpoint HC pairs, seven activation scales and alternating graph timing
pairs; all tested raw/intermediate/final FP16 bits agree.

| Component | M | Control us | Candidate us | Decision |
| --- | ---: | ---: | ---: | --- |
| Down, 420 independent CTAs at M5 | 5 | 17.022 | 18.510 | Reject |
| Down, 840 independent CTAs at M5 | 5 | 16.324 | 18.116 | Reject |
| Same down schedules | 10 | 16.128 / 16.124 | 19.960 / 21.698 | Reject |
| PR #704 register up/mix, original precision | 5 | 14.516 | 10.738 | Component passes, -26.0% |
| Same up/mix | 10 | 15.422 | 11.946 | Component passes, -22.5% |

Increasing independent CTA count alone does not rescue the original-layout
down schedule. Do not repeat those variants or run an engine gate for them.
PR #704's selected up/mix kernel at source
`590d78e0b168d38703f94333d240606900821f85` passes the original MTP precision
screen; its benchmark's global precision override is not copied. Extrapolating
the M5 difference across 96 ordinary HC modules gives about **0.363 ms**,
not an observed full-round gain. Keeping both original and packed up weights
would add about 0.586 GiB/rank for those modules; loader memory and fallback
layout need resolution before integration.

Both kernels here are task-local research extensions, not runtime dispatch
or a source-complete speed claim. `hc_evidence_20260927_contract.json` records
source/JIT/native hashes, commands, environment and GPU state; raw results
are `hc_down_cta_parallel.json`, `hc_up_register_pr704.json` and matching
`*_20260927.{log,exit}`. Both processes exit 0 and release GPU4. No foreign
service is stopped, no wheel is built, and no runtime/default changes.
GDN integration/value-tile gates and current internal trace calibration
remain pending. Prioritize preserving the admitted up/mix component while
evaluating batch HC sharding; the complete-round <20-ms target is unmet.

### Exact MTP batch HC, cooperative execution and router, 2026-09-27

The normal source-built runtime now reuses PR #704's packed batch Tensor Core
HC computation and communicator-owned half-plus-tag transport. The MTP
numerical contract differs from that PR's concurrent-decode screen: keep
`allow_fp16_reduced_precision_reduction=True`, round each of the twenty K512
down partials to FP16, then add in the original FP32 order. Do not copy a
global precision override or substitute M1 GEMV for M5/M10. The loader keeps
original checkpoint weights for M1/prefill and adds TP4 packed shards only
under the explicit MTP4 gate (about 330 MiB/rank).

`VLLM_SM70_MTP_HC_BATCH=1` enables this M5/M10 route. Its optional
`VLLM_SM70_MTP_HC_COOPERATIVE=1` execution combines down, gather/SiLU, up/mix
and output gather in one cooperative launch. All arithmetic and transport
are shared with the four-launch implementation. Dedicated batch channels,
per-block epochs, and clearing consumed packets preserve FP16 bits and
support changing graph widths without collision with M1 or auxiliary MoE.
Both flags default off pending broader promotion.

The separate `VLLM_SM70_MTP_ROUTER_BATCH=1` reuses the same packed batch-MMA
implementation file for the replicated E512 router projection. Four Volta
quad pairs compute the original four contiguous K640 partials; an ordered
FP32 shuffle reduction replaces the separate workspace-reduction launch.
This retains MTP's FP16 output and adds about 122.5 MiB/rank of packed weights.
Its M5/M10 checkpoint component screen is 12.562->7.778 and 11.882->8.270 us.

The HC normal-`_C` TP4 gates cover all 96 checkpoint pairs, all four ranks,
M5/M10, seven activation scales and alternating graph widths. LoRA, mixed
output and injection are bitwise exact. Full pair times, including both
gathers, are:

| Execution | M5, us/pair | M10, us/pair |
| --- | ---: | ---: |
| Replicated control | 33.568 | 34.573 |
| Four-launch TP4 batch HC | 29.013 | 33.730 |
| Cooperative TP4 batch HC | 24.082 | 26.674 |

The mixed-QKV GDN integration and router gates pass 28 GPU tests. HC loader
contracts pass 24 tests; router native/dispatcher graph checks plus the
three existing GPU HC checks pass six tests. An earlier CPU-only invocation
of those three existing tests fails because it hides CUDA; the subsequent
GPU run resolves that invocation error. Changed-file pre-commit passes.

Two resident-engine checks use the unchanged frozen Qwen3.8 Flash Next
NVFP4 / V100-SXM2-32GB TP4 GPUs4--7 / FP16 KV / FP32 SSM / MTP4 /
32K max length / 8192 batch tokens / one sequence / memory 0.95 / 12-GiB
pinned host PLE contract. CUDA 12.8, Torch 2.10.0+cu128, Triton 3.6.0,
driver 580.173.02, greedy 8192-token prompt and 513 output tokens are retained.
No wheel, private preload or sidecar is used by either engine.

| Candidate | Ordinary before | Deferred events | Ordinary after | Ordinary mean |
| --- | ---: | ---: | ---: | ---: |
| Four-launch HC + mixed QKV, `7b0b303a60` | 22.752404 | 22.915771 | 22.746107 | 22.749255 |
| Cooperative HC + batch router + mixed QKV, `3b7365925f` | 22.077260 | 22.139499 | 22.111224 | **22.094242** |

All values are complete-round milliseconds. The second candidate improves
the independent clean 23.697246-ms baseline by 6.76%, and still misses the
20-ms target by 2.094242 ms. Native SHA256 is
`1cd56edf45492ec12a8feed3a087473abd926f11af8145e827c512a280391d8b`.
All fixed token IDs, finish reasons and acceptance counters match the oracle:
335 drafts, 1,340 draft tokens, 177 accepted, positions `[125,36,10,6]`.
Three natural prompts also retain every token ID, normal EOS and acceptance
record (284/329/421 output tokens, temperature 1, top-p 0.95, top-k 20,
seed 20260828, maximum 2,048 tokens).

The second candidate's same-engine event perturbation is **+0.205%** and
ordinary-before/after drift **+0.154%**. Rank0's unscaled, 335 closed cycles
average **22.127934 ms**, only -0.052% from the enclosing observed request:

| Phase envelope | ms |
| --- | ---: |
| Target forward (replay 16.143267) | 16.183720 |
| Sampling and state handoff | 0.766432 |
| Four drafts | 4.607666 |
| Next-round preparation | 0.570115 |

Draft replay envelopes are 1.327611/1.065933/1.060104/1.049167 ms. These
include dependency/submission waits and are not isolated kernel service.
Do not rescale the previous Nsight category table into current measurements.
The optimized ordinary time is intentionally different from the old clean
baseline: remove the parser's inappropriate <=3% equality requirement
against that older implementation. Same-engine perturbation, control drift,
cycle closure and output equality remain required. An independent clean run
of the final candidate is still pending; this trace's admission scope is
the matched resident-engine off/on/off comparison.

Retain `mtp_hc_mixed_candidate_20260927*`,
`mtp_hc_coop_router_candidate_20260927*`, `hc_mtp_native_tp4_20260927*`,
`hc_mtp_coop_tp4_20260927*`, the source/native contracts, and
`analyze_mtp_candidate_trace.py`. The latter writes unscaled per-rank closed
intervals and a Chrome/Perfetto timeline. Do not rerun these completed gates
unless subsequent changes invalidate them.

Further component decisions, without model admission:

- GDN BV16 changes FP32 state and some FP16 output bits; reject it without
  timing. Keep BV32 (`gdn_mtp_value_tile.json`).
- Grouped MoE W13 full unroll-40 is exact but 3--5% slower on the real-route
  screen. N16 tiles are exact but mixed: layer0 averages slightly slower,
  layers24/42 improve by 2.629/1.138 us per complete expert chain. Neither
  justifies another engine run (`moe_w13_unroll40*`, `moe_w13_n16*`).
- Cooperative HC half tiles preserve all TP4 bits but M5 only improves
  24.082->23.595 us; M10 is neutral. Publishing each up tile without the
  final grid barrier is also neutral (23.941/26.825 us). Revert both, retaining
  their source patches, native contracts and `hc_mtp_coop_*_tp4_20260927*`.
- Reusing the M1 HC norm's 1024-wide reduction at M5/M10 changes output bits.
  Retain the 512-wide batch reduction. Only extending weight prefetch is
  exact across all 96 norm weights/seven scales, taking M5 0.352085->0.320811
  and M10 0.352128->0.315904 ms per 96-component chain. This narrow change
  is included in `82ac7f970d`; it is not a measured endpoint saving yet.
- The optional QSA scorer in `82ac7f970d` is rejected and removed. Its
  complete-round ordinary times are 23.308872/23.264122 ms (mean 23.286497),
  despite all fixed/natural outputs and acceptance remaining exact. The
  engine uses page204/8,364 score columns: twelve scorer calls regress
  0.223392->1.186528 ms at M5 and 0.351824->1.477632 ms at M10. The earlier
  page4 synthetic geometry is not representative. Compiler inspection also
  corrects the earlier Tensor Core explanation: both Triton variants lower
  `tl.dot` to scalar FP32 FMA, with zero MMA instructions. The rejected
  variant has a 3,224-byte/thread stack (1,016 LDL and 570 STL instructions),
  whereas the original has no stack. Preserve `qsa_batch_compiler_audit.json`,
  `qsa_mqa_engine_page*`, and `mtp_qsa_batch_candidate_20260927*`.
- A native shared-key scorer preserves the original sequential 128 FP32
  FMAs and all screened scores, without spills, but does not improve M5.
  Unroll8 measures 0.237264->0.244368 ms per twelve calls; full unroll128
  measures 0.260672->0.261776 ms. Stop this experiment (`qsa_batch_exact*`).
- Full cooperative HC down/up K-loop unrolling preserves all 96 real pairs,
  four ranks, seven activation scales, and M5/M10 changing graph widths.
  The TP4 component on GPU0--3 measures M5 33.552223->21.110666 us and M10
  34.607112->23.565778 us against the ordinary reference. The previous
  cooperative candidate was 24.082445/26.674223 us on GPU4--7. This is a
  component result, not a measured complete-round saving. Retain
  `hc_mtp_coop_full_unroll_tp4_20260927*` and `hc_up_full_unroll*`.
- Reuse the existing single-row exact QSA selector's bucket compaction for
  M5/M10 rows with at most 2,304 visible blocks. A union shares the compact
  and normal workspaces; longer rows retain the original four-pass body.
  Buffers over 9,216 columns retain the original kernel, since the combined
  kernel regresses the 64K-context component. At M10, 2,176 blocks improve
  24.586668->11.250667 us, 8,192 blocks are neutral
  (33.708001->33.537333 us). Native schema adds optional `decode_batch=False`;
  `VLLM_SM70_QSA_MTP_TOPK=1` opts in. Nine native GPU tests pass, including
  M1/prefill regressions, changing captured M5/M10 inputs, threshold crossing,
  ties/infinities, noncontiguous rows, and the long-buffer fallback. Preserve
  `qsa_topk_adaptive*`, `qsa_topk_shared*` and
  `qsa_mtp_topk_native_gate_20260927.log`. Whole-model admission is pending.

The pinned PR704 GDN output projection uses FP32 split partials, whereas the
current MTP M5 projection requires FP16 materialization. Do not port it by
changing the global precision contract. The earlier packed output projection
screen already preserves M5 bits but regresses latency, and changes M10 bits
(`out_projection_batch*`); do not repeat it without a new numerical/scheduling
hypothesis. No wheel was built.

### Combined HC unroll / selector result and resident comparison

Source `732e184173` completes the frozen GPU4--7 gate. Ordinary before/after
rounds are **22.359054/22.292406 ms**, mean **22.325730 ms**; observed is
22.432099 ms (+0.476% perturbation, -0.298% control drift). All fixed and
natural token IDs, finish reasons and acceptance counters match. Rank0's
unscaled closed cycle is 22.439080 ms: target 16.322280, sampling/state
0.783965, four drafts 4.684935, preparation 0.647900. The 500-ms telemetry
samples during all fixed requests show 1,530-MHz SM clocks and no throttle
flags. This candidate does **not** beat the earlier qualified 22.094242-ms
process; neither component savings nor the different-process difference
establish an isolated HC/selector gain. Keep both HC schedules selectable
with `VLLM_SM70_MTP_HC_FULL_UNROLL=0/1` (default 0) for a resident A/B/A.
Retain `mtp_hc_unroll_topk_candidate_20260927*`, including both native library
hashes, raw events, full outputs and GPU telemetry. The <20-ms goal is unmet.

Two more component decisions:

- Splitting the current packed HC down tile from N32 to N8 keeps all partial
  bits but regresses M5 5.406->5.848 us and M10 6.078->7.876 us. Reject it
  (`hc_down_n8*`); its initial fixture omitted the separate injection rows,
  which is corrected before arithmetic testing.
- Full-width up (a possible way to omit the final gather) takes
  10.704/10.776 us at M5 with unroll4/20, versus the sharded up's 5.388 us.
  No complete-pair gain is established; do not add another packed full-weight
  copy to the engine (`hc_up_full_width*`).
- Selecting only the first 16 lossless router keys, instead of sorting all
  512, preserves all 65,536 FP16 payloads and changed graph inputs at
  M1/2/4/5/8/10/16. Keep the eight-warp FP32 normalization and original top10
  outputs. The M5 48-call chain improves 0.241504->0.173168 ms, M10
  0.205488->0.147520 ms. The opt-in `VLLM_SM70_MTP_ROUTER_TOP16=1` admits
  FP16 M5/M10 only. Public router and HC integration tests pass 46 cases.
  Whole-model admission remains pending (`router_top16*`,
  `mtp_router_top16_hc_dispatch_gate_20260927.log`).

The task-local resident harness retains ordinary CUDA graph definitions,
clears inactive request/prefix state before variant recapture, and records
command-by-command source/flag identity and graph metadata. It is intended
to amortize model loading and compare schedules under one process. Graph
recapture and timing readback remain outside measured requests. This harness
is a measurement tool, not a serving default or a new latency claim.

### Resident controls and exact shared-expert batch fusion, 2026-09-27

The same-process A/B/A at source `1ef9f45a58` qualifies a small combined
improvement: A (HC unroll4, original QSA selector and full router sort)
averages **22.156515 ms**, B (HC full unroll, adaptive QSA top-k and partial
router sort) **22.005602 ms**. Common HC norm prefetch stays enabled in both.
B ordinary before/after are 21.865026/22.146178 ms; observed is 22.146999 ms
(+0.643% perturbation, +1.286% control drift). Its 335 closed rank0 cycles
average 22.130003 ms: target 16.114291, sampling/state 0.780309, four drafts
4.548976 and preparation 0.686427. Fixed and all three natural-EOS token IDs,
finish reasons and acceptance records remain exact. Do not promote the
single 21.865-ms observation as the endpoint. The <20-ms goal is unmet.

Retain `mtp_resident_ab_20260927*`. A later diagnostic callable-RPC attempt
terminates that resident run after the valid A/B and quality data have been
saved. The replacement uses a named, restricted diagnostic RPC; insecure
serialization is not enabled. Subsequent raw-graph timing attempts establish:

- The first observer selected M5 with no context bucket, while the request
  uses M5/bucket32768. Its zero internal event count invalidates attribution;
  only outer events were collected (`mtp_resident_modules_20260927*`).
- After selecting the actual graph and requiring 336 unique invocations,
  194 event nodes inflate ordinary **21.915471 ms** to **27.121042 ms**
  (+23.753%). All token/acceptance records remain exact, but the internal
  breakdown is rejected for ordinary absolute timing. An empty cloned graph
  stays near the ordinary control (22.123230 versus 21.970648 ms).
- The next sparse event probe fails before generation on an unnamed graph
  tail node; this is a harness error, not model behavior. Diagnostic errors
  now return data instead of escaping the worker RPC. The normal model
  kernels are unchanged. Preserve `mtp_resident_modules_hit_20260927*`;
  do not rescale the rejected internal times.
- A research-only globaltimer kernel-node observer passes 16 changed-input
  tiny-graph replays with unique GPU-counter slots. Model perturbation and
  output gates are pending. It is not an admitted trace or serving kernel.

`VLLM_SM70_MTP_SHARED_BATCH=1` adds a source-built M5/M10 shared-expert
path, using the existing fused-activation hook. Its Tensor Core gate/up
projection preserves eight K320 partitions, the FP16 split partials,
left-to-right FP32 reduction, FP16 projection and FP16 SiLU before multiply.
A second fusion leaves the scalar projection untouched and combines only
its FP16 sigmoid and output multiply. The original M1/prefill paths remain
available. Packing adds **76.5625 MiB/rank** for all 49 target/draft modules.
The flag defaults off pending complete-model admission.

The normal native build passes all 49 real weights, four TP slices, M5/M10,
seven input scales and changing graph inputs bit-for-bit. The 49-call
projection/activation chains save 0.270--0.325 ms at M5 and 0.267--0.277 ms
at M10. Separate sigmoid/multiply research covers every FP16 payload and
changing graphs: the 48-call chain changes 0.270432->0.110112 ms at M5 and
0.191424->0.082944 ms at M10. These component savings are not summed into
an endpoint claim. Native/public fallback tests pass 36 cases; source build
and applicable pre-commit checks pass. Retain `shared_up_batch*`,
`shared_sigmoid_mul*`, `shared_batch_native_gate*` and
`shared_batch_dispatch_gate_20260927.log`. The full model gate is running
under the original GPU4--7 contract. No wheel or foreign service stop.

A separate one-warp GDN input screen retains exact QKVZ and four-partition
b/a outputs for all 36 real rank0 layers and six scales. It removes the
three inactive QKVZ warps and shared-memory b/a reduction, but improves only
1.102438->1.045955 ms for the complete M5 component chain (M10
1.124393->1.095516 ms). Keep it as research pending a larger structural
benefit; do not infer large end-to-end savings from thread occupancy alone.
The nominal checkpoint-weight byte/time quotient is about 689 GB/s for the
original M5 chain, not a measured HBM utilization counter. NCU counters
remain unavailable. Retain `gdn_input_warp*`; unroll16 regresses M10.

The shared-expert candidate completes ordinary/event/ordinary and natural
quality before a later profiling harness failure. Ordinary rounds are
**21.662596/21.782064 ms**, mean **21.722330 ms**; outer deferred timing is
21.788783 ms (+0.306% perturbation, +0.551% control drift). All 513 fixed
IDs, 335 drafts, 177 accepted tokens and the three natural EOS sequences
(284/329/421 tokens) match the retained oracle exactly. Retain
`mtp_shared_batch_resident_20260927*` and its normalized completed-segment
audit. The internal timestamp-template capture then fails because it ran
outside inference mode; the following request encounters an illegal address
after that failed capture. These diagnostic attempts are rejected. The
harness now enters inference mode for diagnostic calls and stops on any
reported diagnostic error before further generation. A fresh ordinary
confirmation and admitted internal breakdown remain pending. <20 ms is unmet.

### Exact PLE rollback/conv fusion, 2026-09-27

`VLLM_SM70_MTP_PLE_CONV=1` combines the single-request MTP4 PLE rollback,
four ordered dilated-convolution taps, FP16 convolution boundary, SiLU and
state commit in a normal source-built kernel. It supports M5/M10 graph
padding, H10240, dilation3 and a 13-position FP16/FP32 cache in both supported
layouts. Null state IDs never write state. The existing generic path handles
all other shapes. The flag remains default off. No precision or accumulation
policy changes, sidecar runtime library, or wheel are involved.

Ten native/public tests pass, including 30 successive rollback/commit rounds
per dtype/layout/padding case, accepted lengths 0/1/3/5/8, query lengths
0/1/3/5 and null/live states. Effective output bits and the entire state match;
padded rows are zero (zero sign is not promised). The component chain falls
from about 124--144 us to 4--7 us. Retain `ple_spec_conv*`,
`mtp_ple_native_gate_20260927.log` and `mtp_ple_conv_build_20260927.log`.
Changed-file pre-commit checks pass.

The same complete model contract with shared-expert fusion and PLE enabled
qualifies ordinary **21.371631/21.643868 ms**, mean **21.507750 ms**. Deferred
outer events measure 21.537974 ms, only +0.1405% relative to controls, with
+1.2738% control drift. All fixed 513 token IDs, 335 drafts, 177 accepted
tokens and three natural normal-EOS sequences (284/329/421 tokens) match the
oracle. Two final ordinary requests after disabling the diagnostic observer
measure **21.463874/21.391711 ms**, mean **21.427792 ms**, with the same exact
quality. These are confirmations within this loaded engine, not independent
fresh-process replicates. The resident process exits normally and releases
GPU4--7. Retain `mtp_ple_shared_resident_20260927*` including quality audit.

The admitted outer rank0 closed cycle is **21.542208 ms**:

| Stage | ms |
| --- | ---: |
| Target forward | 15.534391 |
| Sampling, state and handoff | 0.780901 |
| Four drafts | 4.574895 |
| Next-round preparation | 0.652021 |

Target replay alone is 15.488772 ms; draft replay envelopes are
1.307042/1.049184/1.058051/1.054877 ms. These are dependency-aware replay
walls, not isolated kernel service times. Internal HC attribution is still
unqualified. The 194-globaltimer-node diagnostic measures 22.884433 ms
versus 21.383865/21.451629-ms controls (+6.85%), so its HC and branch sums
must not be presented as ordinary execution costs. A two-marker capture
measures 88.799896 ms with multi-second rank stalls, while surrounding
controls remain 21.976152/21.617862 ms. Preserve the entire failed capture;
do not drop outliers, assume their cause, align independent clock origins,
or rescale either capture to the baseline.

The run uses source `69eac6d8e806ad30e41639863aacddae827199ac` plus the saved
PLE patch; `_C` SHA256 is
`95f0af7adcb248d5227162c75e48c1e0e0df227ad2d8cc594be01aa28697987d` and
stable-native SHA256 is
`ea479867dce18c14b91c16045b62ef1ab20634a211069c3d89ace7d252ddda63`.
The exact diff and untracked-source hashes are in the run contract.
The complete-round **<20 ms goal remains unmet**.

### Follow-up screens after PLE

- Pairing two expert groups in each W2 warp (N16 tiles) preserves all output
  bits on three real layers, five captured routing sets and four activation
  scales, but regresses every component case. The 16-warp variant is about
  3--10% slower; 32 warps regress by up to 19.6%. Do not integrate this screen
  or repeat it. Its first compile missed an output argument and produced no
  benchmark evidence; the corrected gate is `moe_w2_n16_gate_20260927.log`.
- The rejected GDN BV16 state's first divergence is now localized to value
  coordinates 12--15 modulo32. Both shapes have the same K reduction tree;
  LLVM/PTX contraction rounds a different first product in the final four V
  rows of each tile. Pinning the BV32 product/FMA sequence explicitly restores
  every FP16 output and FP32 state bit for M5/M10 across changing inputs,
  accepted positions and six scales. This is a new diagnosis, not permission
  to accept the original non-exact BV16 schedule. `gdn_fma_probe*` and
  `gdn_fma_order*` retain the evidence.
- Exact pinned BV8/BV16 recurrence components save only about 0.04--0.06 ms
  per 36-layer chain. A warp-local Q/K norm also preserves the original
  reduction/FMA sequence but does not improve the chain; BV4 regresses M10.
  Keep these as research; no full-model reload or claimed endpoint saving.
  Retain `gdn_warp_norm*` and `gdn_fma_parallel*`.

- Replacing the first two cooperative HC grid barriers with per-producer
  ready flags and coherent LoRA loads is exact on eight real pairs, four
  ranks, M5/M10, seven scales and alternating widths. It is slower:
  23.216->24.576 us at M5 and 25.467->29.157 us at M10. Both variants use
  254 registers without spills. Do not integrate this scheduling change;
  fewer whole-grid barriers alone does not establish lower latency. Retain
  `hc_ready_flags_gate_20260927*`, `hc_flags*` and their separate IPC channels.

- Cooperative HC with 2/4 warps per CTA (40/20 CTAs at M5) retains exact
  outputs but loses to the 80-CTA, one-warp schedule: M5 23.237 us versus
  25.088/28.597 us; M10 24.021 versus 25.339/30.597 us. All use 254 registers
  without spills. More warps per CTA is not an admitted utilization fix.
  Keep `hc_warp_geometry_gate_20260927*` and `hc_warps*`.
- Grouped W13's warp-shuffle activation epilogue is exact but 2.4--4.8%
  slower across the real-route chain. Moving the four K partitions into
  the four quad pairs also preserves exact bits but regresses 25--51%.
  Both skip unnecessary full-model runs; retain `moe_w13_epilogue*` and
  `moe_w13_quad_split*`. The original packed MTP5 expert path stays selected.

- HC reciprocal-only sigmoid is bitwise identical on all 65,536 FP16
  payloads and their quarter-scaled forms, including NaN payload results.
  However its complete TP4 pair is nearly neutral at M5 (23.221->23.045 us)
  and slower at M10 (25.920->26.357 us). Do not integrate or extrapolate a
  meaningful endpoint gain. Preserve `hc_sigmoid_rcp*` and
  `hc_fast_rcp_gate_20260927*`. All follow-up research workers exited; owned
  GPU leases are released. Runtime source remains PLE commit `763189d9a8`.

## Merge audit against main (2026-09-27)

The user explicitly requested an audit and merge of PR #703. The integration
line is `onecat/main`, fetched at
`ef6909830cbb7b40a24413bfe74ab49a4f7e1b90` (#706). It was merged into the
published owned branch without rebasing, using merge commit
`b772a0e96271dc11b21509b80aa0fb8999885d26`. There were no conflicts. Main's
medium-message TP4 all-reduce tuning and concurrent/draft projection changes
remain intact. The native source and Python candidate were subsequently
validated together, not just against the earlier PR base.

The review covered native registration/build integration, tensor geometry and
alignment guards, communicator-owned HC packets and changing graph widths,
PLE rollback/cache writes, ordered projection/activation rounding, shared
classic/modular MoE dispatch, mixed-QKV state preservation, and QSA/router
selection fallbacks. One actionable numerical-contract defect was found:
GDN batch dispatch did not reject `allow_fp16_accumulation=True`. A five-row
screen produced 12,002 / 7,192 / 54 / 56 differing half elements in QKV / Z /
B / A against the original projection. The admitted FP32-accumulation modes
had zero differences. Commit `4d4cb68dac0976e6c8c1394729a01ac98ff18146`
adds that fallback and GPU M5/M10 regression checks. It does not change the
frozen performance contract, which uses FP32 accumulation.

Normal native rebuild succeeds (`bash .artifacts/build.sh`); the optional
Rust frontend remains unavailable in this environment. No wheel was built.
Fresh-process import with both `LD_PRELOAD` and `LD_LIBRARY_PATH` unset,
`readelf -d`, and the resolved Torch/CUDA/cuBLAS mappings show no private
kernel dependency. Rebuilt `_C` SHA256:
`790ca7b49e83c2289c98b167ca070146d10643f63dab653ee6a54e95f56a175f`.
Stable-native SHA256 remains
`ea479867dce18c14b91c16045b62ef1ab20634a211069c3d89ace7d252ddda63`.
Changed-file pre-commit passes, including the audit fix.

The following focused source-built GPU/CPU suite passes **185 tests** in
53.50 seconds, including graph replay, full cache/state bits, dispatch and
fallback checks. It was run through the task GPU-lease wrapper on GPU0:

```bash
MTP_RESEARCH_GPU=0 .venv/bin/python .artifacts/run_research.py -m pytest -q \
  tests/models/qwen4_exp/test_sm70_gdn_input_batch.py \
  tests/models/qwen4_exp/test_sm70_mtp_hc_batch.py \
  tests/kernels/test_sm70_mtp_ple_conv.py \
  tests/kernels/test_sm70_mtp_shared_batch.py \
  tests/kernels/test_sm70_mtp_router_batch.py \
  tests/kernels/moe/test_sm70_mtp_moe_fp16.py \
  tests/kernels/moe/test_sm70_router_key_dispatch.py \
  tests/kernels/test_sm70_mtp_gdn_mixed_qkv.py \
  tests/models/qwen4_exp/test_hc_norm_dispatch.py \
  tests/quantization/test_sm70_nvfp4_grouped_decode_dispatch.py
```

The first test invocation used an incorrect directory for the grouped-decode
file and collected no tests; its log is retained separately. The corrected
command above is the accepted result.

Raw audit artifacts remain under
`/home/ymzx/桌面/1cat-vllm/worktrees/v100-mtp4-batch-gemm-20260926-155343/.artifacts/`:
`pr703_merge_build_20260927.log`,
`pr703_merge_runtime_audit_20260927.json`,
`pr703_gdn_precision_audit.log`,
`pr703_merge_targeted_tests_corrected_20260927.log`,
`pr703_merge_precommit_20260927.log`, and
`pr703_audit_fix_precommit_20260927.log`.

### Matched integration gate and merge disposition

A fresh engine at source `4d4cb68dac0976e6c8c1394729a01ac98ff18146`
uses the unchanged workload contract above, GPU4–7 and every admitted MTP
acceleration enabled. The source diff is empty; no untracked source file or
private library override participates. The command is:

```bash
MTP_WAIT_SECONDS=900 .venv/bin/python .artifacts/run_mtp_optimization_gate.py \
  pr703_merge_all_accel_20260927 --hc-batch 1 --gdn-mixed 1 \
  --hc-cooperative 1 --router-batch 1 --hc-full-unroll 1 --qsa-topk 1 \
  --router-top16 1 --shared-batch 1 --ple-conv 1 \
  --trace deferred --natural-full
```

Ordinary/event/ordinary complete rounds are **21.421255 / 21.511603 /
21.401212 ms**, ordinary mean **21.411234 ms**. Ordinary pure decode is
71.347743 / 71.414562 tokens/s; TTFT is 0.434015 / 0.408487 seconds and
prefill is 0.423256 / 0.396150 seconds. Observer perturbation is +0.468769%;
ordinary control drift is -0.093565%. No timing is rescaled.

All four fixed 513-token cases and three natural cases preserve every token,
finish reason and acceptance statistic. Fixed requests retain 335 drafts,
1,340 drafted tokens, 177 accepted tokens and positions `[125,36,10,6]`;
natural requests retain 284 / 329 / 421 tokens with normal EOS. This is the
same focused quality gate, not a claim of broad benchmark accuracy.

The rank0 closed 335-cycle outer timeline sums to **21.515078 ms**:
15.479071 target forward + 0.784805 sampling/state + 4.664252 draft +
0.586950 preparation. Selecting the slowest complete interval at each
aligned round ordinal gives 21.542033 ms; every phase in an interval comes
from that same rank. Per-rank cycle/request closure error is at most
0.016269%. These are replay/dependency envelopes, not kernel service sums;
no new claim about absolute internal HC cost is made.

The full run exits zero, and its workers and GPU leases are released. Logs
confirm the actual batch GDN/HC/router/shared-expert, PLE, exact draft MoE,
gated norm, mixed-QKV, grouped-MTP and QSA routes. The raw contract, quality
audit, telemetry, deferred trace, Chrome timeline and closure report use
prefix `pr703_merge_all_accel_20260927` in the artifact directory above.

The audit fix and integration gates admit PR #703 for the explicitly
requested merge into `onecat/main`. New opt-in defaults and numerical
contracts remain unchanged. The <20-ms goal and calibrated internal kernel
attribution remain follow-up work; neither blocks the user's authorized
integration of this measured improvement. No wheel was produced.
