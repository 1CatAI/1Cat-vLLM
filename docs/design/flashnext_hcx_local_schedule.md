# Flash-Next HCX local schedule on full-mesh V100

The full-mesh HCX schedule delays up-weight prefetch until combine finishes,
and assigns the eight gate-mix accumulator slots to eight warps. Previously,
up-weight prefetch overlapped the initial down-weight loads, and warp 0 mixed
all eight slots serially. Each output retains the original five-part FP32
sum, FP16 casts, sigmoid and four-stream FMA order.

The normal `_C` implementation offers this experimental schedule for TP4
full-mesh HCX, M1–8, with a separate output projection. Fused output projections
and the two-hop exchange retain their existing schedule. Large batches still use the
existing model fallback. The kernel policy `sm70_hcx_local_schedule=true`
opts in before graph capture. The default is false at both the Python policy
and native operator schema because model qualification has not passed.
HCX itself remains controlled by `sm70_hcx`.

## Related work and interpretation

[mHC's kernel-fusion design](https://arxiv.org/html/2512.24880v2#S4.SS3.SSS1)
moves division by the norm after the projection and fuses scans of the
expanded residual stream. HCX already computes per-stream down partials
before applying reciprocal RMS. The transferable opportunity is further
local data reuse; mHC's Sinkhorn mapping and training overhead are not the
Qwen HC graph or a V100 decode latency prediction.

[TensorRT-LLM's all-reduce API](https://nvidia.github.io/TensorRT-LLM/python-api/tensorrt_llm.functional.html#tensorrt_llm.functional.AllReduceFusionOp)
exposes residual/RMSNorm fusion, including a MoE-finalize variant. HCX already
uses this general approach; its remaining local dependency chain must be
measured inside the fused kernel. The retained change shortens that chain
without changing the collective or the reduction tree.

[MSCCL++'s synchronization model](https://microsoft.github.io/mscclpp/dsl/concepts.html#synchronization)
distinguishes a thread-block barrier from asynchronous semaphore signaling.
That makes selective consumer waiting a relevant experiment, but does not
establish that it will be faster on this topology. The direct-polling screen
below loses on these V100s. Likewise, removing several operations in an
ablation does not make their individual latency differences additive: overlap,
cache pressure and scheduling change together. The 12–14 µs target is not an
achieved result of this work.

## Measurement contract

Integration base: `17fa23e576fd12d2b3ce527558539058ab3024e6` on
`codex/v100-flashnext-host-fp8-kv-20261007-080157`. The implementation commit
is `3cf6194254798013425f4ddc6f434c3066f8742c`.
The compiled-model cache policy and test-harness corrections are in
`9e0451404f04f7b95d25df03bd299316a4c74306`.

The screen uses four Tesla V100-SXM2-32GB GPUs, every pair connected by NV2,
CUDA 12.8, Torch 2.10.0+cu128 and driver 580.173.02. Eight actual HC weight
pairs rotate through a 512-boundary CUDA graph. Activations are synthetic.
Each arm runs twice per sample with rotating/reversed order. The statistic
is the median of the per-sample maximum across all four ranks. The full
all-reduce/combine/norm/down/up/output exchange is timed.
The eight-pair weight file has SHA256
`8ef327c7e24f110147ee2e6fde9c631fb8576018236bed6ac434569d14266aac`.

The research control core SHA256 is
`784d1447f4f5f5593fa77db525e6b40e841366d654202aab5a56beec67398a0c`.
These research measurements are separate from installed-wheel validation.

| Rows | Installed control, µs | Copied control, µs | Selected schedule, µs |
| --- | ---: | ---: | ---: |
| 1 | 18.773 | 18.770 | 16.431 |
| 5 | 23.750 | 23.731 | 21.052 |
| 8 | 30.516 | 30.452 | 27.697 |

The research checks cover all eight weight pairs, changed inputs, an optional
second producer contribution, shared/per-branch norm weights and changed-input
graph replays. The packaged benchmark additionally compares the raw FP16 bit
patterns, including zero inputs and the operator's default dispatch.

## Installed-wheel validation

The normal CMake `_C` target was built from the owned implementation source,
installed into the wheel, and checked in a fresh task runtime. Both schedules
are in that same extension. No research DSO is loaded for these results.
The extension has no RPATH/RUNPATH and depends on the declared standard
Torch/CUDA/system libraries.

Wheel: `1cat_vllm-1.5.2.dev1194+g3cf6194254.cu128-cp312-cp312-linux_x86_64.whl`.
Wheel SHA256:
`901310b460e14f7f218ebc644f8331552532eee163b0ad3fe9f691db3332dbb6`.
Core SHA256:
`d56e4ec2d07cec0698dcb21f1ecba53f7cd761d18d804b5ae29820214fd930d0`.
Both native schedules use 92 registers and 13,824 bytes of shared memory,
with zero local memory or stack reported by `cuobjdump`.

| Rows | Reference schedule, µs | Local schedule, µs | Reduction |
| --- | ---: | ---: | ---: |
| 1 | 18.723 | 16.332 | 12.8% |
| 5 | 23.710 | 21.131 | 10.9% |
| 8 | 30.454 | 27.876 | 8.5% |

M5 has 24 observations per arm: reference 23.652–23.958 µs and candidate
21.060–21.342 µs. These are observed ranges, not confidence intervals.

Each M1–8 point passes 67 raw-FP16-bit comparisons per rank: four eager input
sets across eight actual weight pairs, both explicit and default dispatch,
and three changed-input graph replays. The M2/3/4/6/7 follow-up uses fewer
timing samples and is a correctness check, not a quoted speed measurement.
The CPU dispatch/configuration tests pass 25 cases after updating two test
doubles and checking that both schedules reuse the compiled-model cache.

The compiler/graph payload test passes on both the earlier installed wheel
and the new wheel, including M5 HCX and the M20 fallback. Its first run exposed
a test-harness omission: raw CUDA graph capture did not register custom
all-reduce peer addresses. Both wheels failed at the first M20 graph replay,
before HCX was selected. The test now uses the same distributed
`graph_capture` context as the model runner. Both reruns pass and report
identical errors against their independent reference implementation.

The final default-off policy is built from
`84e41a1d1489e5f358c78658a7ab337611a17d8b` into
`1cat_vllm-1.5.2.dev1197+g84e41a1d14.cu128-cp312-cp312-linux_x86_64.whl`.
Its SHA256 is
`bd5a10fb8355a5bc88bb8ac9759bdb9f2bc98ff25a40dab34fc978adf4623970`;
the rebuilt core SHA256 is
`26b02babd7a3680ddb3e85f1b035688ad595873ce4c80b30ebf9c3ff4debf6e6`.
A fresh installed audit confirms the false default in the Python configuration
and native schema, with equal compile keys for explicit true/false settings.
All 25 CPU cases and focused pre-commit checks pass again. The entire GPU
device-code section is byte-identical to the measured wheel, with SHA256
`8b6e60a06b8139f22819adb17ba4af44fe897da59365a909ad9ae21deac1c1d4`.
The core hash changes because of the C++ schema default; the GPU kernels did
not change. A follow-up with four timing samples (eight observations per arm)
against the installed default-off wheel passes the same 67 bitwise checks per
rank at M1, M5 and M8. Its paired
medians are 18.827/16.352, 23.733/21.146 and 30.540/27.877 µs respectively.
This confirms the packaged routes; the longer initial run remains the quoted
performance dataset. This check does not replace model quality qualification.

## Model measurement contract

The model comparison uses the same installed wheel for both arms and switches
only `sm70_hcx_local_schedule`. It runs Flash-Next GSQ-RCO IQ3_S GGUF with
checkpoint-FP16 HC weights and a FP16 MTP4 draft on TP4. Model dtype is FP16;
the QSA KV policy is E4M3 with `qsa_host_kv_device_reference=true`, an 8192-token
hot window and FP16 draft KV. Maximum model length is 9216, maximum batch
tokens 512, maximum concurrent sequences 4 and GPU memory utilization 0.95.
Prefix caching is disabled. Output projection remains separate from HCX.

The eight natural prompts use greedy sampling with a 600-token limit and
ordinary EOS handling. C1 measures fixed I8192/O256 and C4 fixed I128/O600,
with EOS ignored for those two performance probes only. Initial loading,
compilation, prefill and teacher captures are excluded from the pure-decode
statistics. The C1 observer-off measurements bracket the GPU-envelope probe.
Teacher captures use 64 fixed prompt/position pairs and two repetitions per
arm. Candidate conditioning comes from the reference token sequences.

The first fresh-process pair used separate compilation caches. Actual routes
match on all four ranks after accounting for the scheduling flag: FULL decode,
95 HCX modules, no deferred output projections, and captured M5/10/15/20.

| Initial fresh-process observation | Reference | Local schedule |
| --- | ---: | ---: |
| C1 unobserved round, ms (mean of two cohorts) | 19.8828 | 19.6105 |
| C1 emitted tokens/round | 4.8857 | 4.8857 |
| C4 unobserved round, ms | 43.3138 | 43.2972 |
| Observed target GPU envelope mean, ms | 15.6929 | 15.5179 |
| Eight-prompt mean draft acceptance | 45.3421% | 44.7988% |

The three C1 output sequences and all four C4 output sequences match, but
only one of eight natural sequences matches. At 64 aligned teacher positions,
cross-arm mean/p99/max KL is 0.001207/0.017931/0.029911 and top-1 agreement is
63/64. This fails the existing mean-KL, p99-KL and top-1 gates. Both within-arm
repetitions have identical logits at all 64 positions. The acceptance delta
is -0.543 percentage points, with a paired prompt-bootstrap 95% interval
[-3.607, +2.600]; this does not establish acceptance equivalence. The observed
0.272-ms C1 reduction is therefore not an admitted model optimization result.

The initial configuration field also partitioned the compiled-model cache,
although its dispatch occurs inside the opaque HC operator. Excluding it
from that cache key preserves reuse of unrelated compiled operators; fresh
CUDA graph capture still records the selected native schedule. This does not
by itself establish the cause of the numerical differences. A same-process
original/recaptured-reference/candidate/original-again diagnostic holds the
loaded weights and compiled operators fixed to localize them.

The cache-policy build is
`1cat_vllm-1.5.2.dev1195+g9e0451404f.cu128-cp312-cp312-linux_x86_64.whl`,
SHA256 `46992d012ff5bc96b3e8722d72d1e29d180aa091b1686b97298a76120c6e8c9a`.
It contains the same native core SHA256 as the measured initial wheel.
Installation into a separate task runtime confirms the native schema and
equal cache keys for the two schedules. The first diagnostic stopped before
any output comparison because its trusted local callable RPC lacked the
serialization opt-in. The corrected offline harness checks this before model
loading. A second attempt stopped before comparing outputs because engine
initialization mutated a nested configuration object in the report. The
report now retains an independent configuration copy. Neither harness failure
produced a model quality result; both logs are retained. The corrected run
uses the cache-policy wheel and the original benchmark's matmul precision,
sampling seed and warmup.

The corrected diagnostic completes its original-schedule phase: eight natural
continuations and all 64 teacher positions. Recapture then stops on an overly
strict invocation-count assertion (188 captured calls versus a threshold of
190); no new-schedule or roundtrip phase completes. These are not successful
same-process controls. The completed original phase is retained separately
as the reference for a fresh candidate process using the same cache-policy
wheel and the same compilation cache.

Both the initial reference and this completed phase select the reference
native schedule, yet their mean/p99/max KL is 0.000740/0.006074/0.008173, top-1
agreement is 63/64 and none of the eight natural continuations is identical.
The Python cache-key policy and compilation artifacts differ between those
versions; this is not a same-artifact A/A experiment. It shows why the initial
cross-version observations cannot uniquely identify a native schedule effect.

The subsequent fresh candidate process reuses the same cache-policy wheel
and compilation cache as that completed reference phase. All actual routes,
prompt tokenization and conditioning positions match apart from the schedule.
Mean/p99/max KL is 0.001018/0.006454/0.006540, top-1 agreement is 63/64 and
none of the eight natural continuations is identical. Candidate repetitions
remain exactly equal at all 64 positions. This still fails the mean-KL and
top-1 gates, so separate compilation caches are not a sufficient explanation.
Mean acceptance changes from 46.6689% to 46.1004%; the paired 95% interval
for the -0.568-percentage-point change is [-1.769, +0.966]. The completed
reference phase is usable for this comparison; its enclosing four-phase
diagnostic remains incomplete. These results do not qualify a default change
or a model latency claim.

## Rejected or superseded screens

Every row is a same-process comparison within its recorded run. Do not add
individual improvements or compare raw timings across unrelated runs.

| Screen at M5 | Control, µs | Candidate, µs | Decision |
| --- | ---: | ---: | --- |
| CTA-local norm tile | 23.742 | 23.575 | Small standalone gain; omit from final combination |
| Last-arrival norm reduction, with local tile | 23.742 | 23.597 | No added benefit |
| Partial-buffer CTA grouping 4/16/80 | 23.801 | 24.056 / 24.220 / 24.030 | Reject |
| Fixed M5, with local tile | 23.744 | 23.562 | Only 0.056 below dynamic local-tile arm |
| Second barrier counts only receiving CTAs | 23.744 | 23.611 | No added benefit over local tile |
| Up prefetch after first barrier, with local tile | 23.744 | 22.665 | Retain scheduling direction |
| Parallel gate-mix, delayed prefetch, local tile | 23.710 | 21.354 | Retain warp parallelism |
| Earlier delayed prefetch, global norm tile | 23.750 | 21.052 | Selected compact implementation |
| Direct per-CTA LoRA polling, local norm tile | 23.735 | 45.797 | Reject even on full mesh |
| Statically single-slot mix loop, against selected schedule | 21.025 | 21.113 | Reject; installed selected schedule in this run is 21.082 |

The single-slot rewrite reduces the research kernel's static instruction
count by about 13%, with the same register/shared-memory usage, but does not
improve measured latency. All 54 changed-input checks per rank pass. This
screen compares against the already selected schedule, not the original
23.7-µs schedule; its instruction-count reduction is not a speedup claim.

The K-shard rewrite gives each rank 640 hidden columns and each CTA H columns
of all four residual streams. It retains its norm/mix input in shared memory,
reduces squared norms and down partials across ranks, and includes delivery
of the complete residual state in the measured output exchange.
That delivery preserves the current model interface. Keeping the residual
state sharded across multiple model boundaries would require a broader
layout change and is not measured by these prototypes; the results below
do not establish a lower bound for that architecture.

| K-shard receiver | Control, µs | H8, µs | H16, µs | H32, µs |
| --- | ---: | ---: | ---: | ---: |
| One receiving CTA | 23.753 | 77.239 | 77.678 | 82.129 |
| Distributed receiving CTAs | 23.710 | 43.853 | 44.063 | 47.924 |

The distributed rewrite is deterministic and its largest screened relative
L2 error is 7.07e-5, but it loses on latency. It is rejected before model
quality evaluation. It changes the reduction tree and is not part of the
native implementation. Research sources and per-rank JSON are retained in
the task's `hcx-local-fuse` artifacts under `central-receiver`,
`distributed-receiver`, `group-layout`, `scheduling`, `mix`, `final-screen`,
`direct-poll` and `prototypes`; none is loaded by the normal model route.

## Reproduce the installed-operator comparison

Build the owned source using the normal SM70 CMake `_C` target and package it
in the wheel. Install into a fresh task runtime. Use the same model weights,
GPU set and topology for both arms, with no private kernel library overrides.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  .venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=4 \
  benchmarks/kernels/benchmark_sm70_hcx_schedule.py \
  --weights "$HC_WEIGHTS" --output "$HCX_RESULTS" --rows 1 5 8
```

Acquire the campaign's GPU ownership locks before launching. Check M2/3/4/6/7
with the same benchmark and a smaller sample count, and run
`tests/kernels/test_sm70_hcx_compiled_payload.py` for compiler/graph payload
ownership and the M20 fallback. Boundary savings do not establish a model
round saving: compare `sm70_hcx_local_schedule` true/false in the same wheel
with the rest of the model contract fixed.
