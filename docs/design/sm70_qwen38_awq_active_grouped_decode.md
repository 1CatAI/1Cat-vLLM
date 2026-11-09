# Qwen3.8 SM70 AWQ active grouped decode

## Scope and implementation

This is a narrow replacement for the larger implementation in
[1CatAI/1Cat-vLLM#491](https://github.com/1CatAI/1Cat-vLLM/pull/491),
which was closed over maintenance cost versus measured benefit, not a
correctness failure. It reuses `awq_moe_active_dense_stage_sm70_out`, its
existing active-segment builder, and the existing grouped GEMM implementation.
There is no new public operator, Python operator wrapper, or GEMM kernel.

Admission requires SM70 AWQ, TP4, 512 experts, top-k 10, effective group size 32,
hidden size 2560, W13 `(K,N)=(2560,320)`, W2 `(K,N)=(160,2560)`, and 2–8
input tokens. Runtime and pre-capture warmup share the admission policy.
GPU validation uses a native-g32 checkpoint; remapped checkpoints are not a
separate validated quality claim. Single-token and unmatched shapes retain
their existing routes. Explicit
dense/exact diagnostic routes and the existing decode token cap take priority.

Repeated expert IDs form multi-row segments; unused trailing offsets describe
empty segments. The scheduler remains offsets-based, not a one-row-per-route
NVFP4 dispatch. Scratch offsets and IDs are marked mutable in the existing
Torch schema. Rebuild the extension with the Python changes: an older binary
can still expose the same operator name without the new implementation.

The model-specific grouped route defaults on for admitted contracts. To roll
back, set this before starting a fresh engine:

```bash
export VLLM_SM70_AWQ_QWEN38_MOE_COMPACT_GROUPED_DECODE=0
```

## Optional existing autotune

For the validated Qwen3.8 TP4 deployment, the candidate configuration is:

```bash
export VLLM_SM70_AWQ_QWEN38_MOE_COMPACT_GROUPED_DECODE=1
export VLLM_SM70_AWQ_TUNE_SMALL_SHAPES=1
```

`VLLM_SM70_AWQ_TUNE_SMALL_SHAPES` already exists and remains **off by default**.
This PR does not change the global tuning, FP16 reduction, NCCL, or scheduler
defaults. The tuning flag is broader than the narrow admission gate, so its
results must not be generalized to all AWQ models. Retain the existing
preserve-default-splits settings. For a clean comparison, start independent
engines with private empty caches and no imported GEMM LUT; toggling a flag
inside a process does not invalidate previously selected tactics. Roll tuning
back separately with `VLLM_SM70_AWQ_TUNE_SMALL_SHAPES=0` before startup.

## Measurement contract and separate attribution

Evidence was collected on four V100 PCIe 32 GB GPUs, Qwen3.8 Flash-Next AWQ
g32, TP4/MTP0, FP16 activations and KV, maximum length 131328, eight maximum
sequences, 8192 batched tokens, chunked prefill on, prefix caching and async
scheduling off, GPU memory utilization 0.89, language-only MRv2
`FULL_AND_PIECEWISE`. All arms use the same QSA page4 logical-order correction
tracked separately in
[1CatAI/1Cat-vLLM#494](https://github.com/1CatAI/1Cat-vLLM/pull/494).
That attention correction is not included here. Runtime results are from the
frozen `fbcef6e2f9`-based validation stack, not a retest of later upstream main.

All measured arms also freeze the existing `VLLM_SM70_QWEN38_FP16_GEMV=1`,
`VLLM_SM70_QWEN38_FUSED_HC_FP16=1`,
`VLLM_SM70_QWEN38_FUSED_GDN_INPUT_FP16=1`, and
`VLLM_SM70_QSA_INDEXER_CUBLAS=1` opt-ins. Runtime-lossy online QPN8 is off.
Actual collectives on this PCIe validation stack use PyNCCL; the later
fully-connected custom-AR HC-sharding results from SXM2 systems do not imply
that the same optimized route was active here.

Each cell has a 16-token warmup and one scored request batch with a 320-token
output limit, `ignore_eos=false`, `min_tokens=0`, greedy decoding, and frozen
prompt token IDs. The numbered-list prompts request enough output to reach
steady state without suppressing EOS. All scored requests reached 320 tokens.
Pure aggregate throughput counts token deliveries strictly inside the common
window after every request has started decoding and before any request ends.
Each request contributes 319/294/304 tokens for C1/C4/C8 respectively; there
were no multi-token deliveries. This is neither per-request speed nor E2E.

### Grouped route alone: matched r3 comparison

Autotune is off in both AWQ arms. These are the implementation-only gains.

| Cell | Grouped OFF tok/s | Grouped ON tok/s | Gain | NVFP4 tok/s |
|---|---:|---:|---:|---:|
| C1×64K | 48.65 | 48.60 | −0.09% | 54.65 |
| C4×64K | 110.02 | 116.20 | +5.62% | 130.91 |
| C8×16K | 207.29 | 211.94 | +2.24% | 237.58 |

### Existing autotune: subsequent fresh-process comparison

Grouped decode is on in both arms. This isolates the configuration benefit,
not additional code added by this PR.

| Cell | Tuning OFF tok/s | Tuning ON tok/s | Gain | Historical r3 NVFP4 tok/s |
|---|---:|---:|---:|---:|
| C1×64K | 48.7874 | 48.5898 | −0.40% | 54.6544 |
| C4×64K | 116.1290 | 129.2741 | +11.32% | 130.9051 |
| C8×16K | 212.0154 | 242.7240 | +14.48% | 237.5780 |

The new OFF baseline differs from the previous grouped ON by only
+0.38%/−0.06%/+0.04%. NVFP4 was not rerun as a third arm in this second
experiment; one score per cell does not establish statistical superiority.
Do not add percentages from the two experiments or extrapolate to C8×64K.

| Cell | Prefill+mixed seconds OFF→ON | Pure-window seconds OFF→ON | E2E seconds OFF→ON |
|---|---:|---:|---:|
| C1×64K | 15.4253→15.4243 | 6.5386→6.5652 | 21.9639→21.9895 |
| C4×64K | 83.8265→83.5213 | 10.1267→9.0970 | 94.6550→93.2887 |
| C8×16K | 32.8447→32.8341 | 11.4709→10.0196 | 44.8195→43.3021 |

E2E also includes the final drain. C4/C8 E2E duration falls by only
1.44%/3.39%; mixed-phase mean ITL remains about 2.868/2.092 seconds. This is
not a fix for long-prefill interference or a scheduler-policy change.

## Kernel evidence, quality and costs

After scoring, each tuning arm ran a separate C4 diagnostic profile. All eight
rank traces contain 16 CPU execute annotations, 16 GPU execute annotations,
16 CUDA Graph replays and 48 W13/W2 pairs per step, with four generation
requests and zero prefill requests. W13 changes from M64×128×32 to M8×256×64
on all ranks. Its mean per-rank GPU time falls from 5.531 to 2.109 ms/step;
W2 stays near 1.31 ms, FP16 projections/HC near 10.42 ms, and QSA near
3.49 ms. W2 launch grids are not identical across all ranks. These are
profiled kernel durations, not new unprofiled ITLs or a sum across ranks.

- Grouped-only r3: 54 short quality requests across three arms passed basic
  answer checks and stopped naturally at EOS. AWQ OFF/ON complete token IDs
  matched in 17/18 cases; the remaining arithmetic wording was `and` versus
  `+`, with the same answer 156.
- Tuning A/B: 36/36 short requests passed basic answer checks, stopped at EOS,
  and had finite recorded logprobs. C4 self-repeat matched complete token IDs
  and recorded top-5 logprobs in 4/4 cases in each arm. Across arms 16/18
  complete token sequences matched; two C8 responses differed in wording.
  Both new arms also differ from the older grouped-ON reference in 2/18
  cases, so fresh-process baseline drift precludes assigning every change
  exclusively to tuning.
- Real-weight operator replay showed small W13 rounding differences; a
  separate tuning probe reached maximum absolute difference 0.0009765625.
  This is not proof of model equivalence. No new functional failure was
  observed in the bounded prompts, but bitwise equality and broad quality
  acceptance are **not** claimed.
- The precision-sensitive cross-batch/process issue is recorded and deferred.
  This PR deliberately does not change HC/NCCL precision or attempt a general
  numerical-determinism repair. It is separate from QSA ordering correctness.
- Tuning OFF/ON both report 386,392 KV tokens, 509 blocks and 5,210,075,136
  KV tensor bytes per rank, and 0.37 GiB graph memory. Initialization-to-ready
  took 431.090/427.330 seconds. There was no observed extra total startup or
  KV-capacity cost in this pair; host cache variability and full memory peaks
  were not controlled well enough to claim startup acceleration or identical
  peak memory.

## Validation and follow-up

The measured source checkpoint is `5cceeaad89d6ead1474ca834afd9aaf3a7bd413c`.
The extension built successfully; prior validation retained 51 GPU-directed
test passes and 32 dynamic-route comparisons, including repeated experts and
graph replay. The focused CPU policy/warmup regression command is:

```bash
.venv/bin/python -m pytest -q tests/quantization/test_sm70_awq_active_grouped_decode.py
```

The natural-EOS, per-token, startup/LUT and per-rank trace artifacts are
retained under experiment IDs `awq-narrow-qsa-fixed-ab-20260904` (r3),
`awq-nvfp4-decode-gap-profile-20260904` (r3), and
`awq-autotune-model-ab-20260905`. No production deployment is implied.

For fork review, the patch was ported onto synchronized main
`755baae1d075ee04fa9096b23fc0225b23589a86`. Conflict resolution preserves the
new indexed-prefill admission and compact-metadata initialization alongside
the grouped-decode flag. Added boundary tests verify that indexed prefill and
grouped decode never both admit the same token count. The original validated
branch is retained. The new base also changes HC, NVFP4 dispatch and scratch
lifetimes: the historical GPU results above are **not** a GPU acceptance of
this new integrated tree. The separate integration run below closes the bounded
model acceptance gate; final human review remains required before promotion.

### Aligned-tree integration acceptance (2026-09-05, r2)

The native extensions were rebuilt from `b50bb1f5037d` on base `755baae1d075`.
Both formats additionally use the separate QSA ordering port `05c46b2cd0` and
its matching Flash-V100 Python package and rebuilt extension. No production
changes were made after these binaries were built. Each AWQ cell was scored
once with grouped decode and the optional existing autotune enabled; shared
expert overlap was disabled. This is integration acceptance, not a new matched
grouped-OFF/ON experiment or evidence that all changes come from this PR.

| AWQ cell | Pure aggregate tok/s | Mean pure ITL ms | Prefill+mixed s | Pure window s | E2E s |
|---|---:|---:|---:|---:|---:|
| C1×64K | 51.2897 | 19.4971 | 14.5356 | 6.2196 | 20.7552 |
| C4×64K | 131.7116 | 30.3694 | 79.8826 | 8.9286 | 89.4635 |
| C8×16K | 247.5878 | 32.3118 | 31.3695 | 9.8228 | 41.6304 |

The frozen inputs, natural-EOS 320-token cap and pure-window accounting are
unchanged: 319/294/304 tokens per request, no multi-token deliveries. All
18 short quality requests stopped at EOS, passed basic answer checks and had
finite recorded logprobs. Same-arm C4 repeats matched complete token IDs and
top-5 logprobs in 4/4 cases. The post-score C1 profile matched the scored output.
This is bounded quality evidence, not cross-version bitwise or broad quality
acceptance; the precision deferral above remains unchanged.

All four workers reported 48 admitted grouped layers and tuning enabled.
Startup took 438.4613 seconds. The engine reported 427,385 KV tokens, with
563 configured cache blocks and 0.37 GiB graph memory. Correction: the earlier
563×784 extrapolation was not the engine-reported token capacity.
The model exited with status 0 and no OOM. Python
resource-tracker cleanup warnings at shutdown are retained in the raw log;
the completed scores and profile are not reclassified as a clean shutdown log.

The first integration attempt failed before long-context scoring: QSA imported
an image-local 18-argument XQA binding, while the hash check inspected a
different top-level 19-argument module. Fixing the harness package path and
checking actual binding identity plus argument conversion resolved that
runtime mismatch. No attention route, quality gate or precision setting was
disabled. Failed and successful evidence remain separate under experiment ID
`qwen38-c1-main-20260905-r2` (successful AWQ/NVFP4) and its unsuffixed failed run.
This investigation did not require another change to the PR's five production
files. GPU integration acceptance does not imply production deployment.

C1 is a separate follow-up for **both AWQ and NVFP4**, not merely an attempt
to reach NVFP4's current speed. Investigate shared projection/HC, attention
and launch/reduction overhead alongside format-specific MoE preparation.
Neither the common costs nor NVFP4's current performance establish how much
can actually be recovered. Keep that investigation out of this PR.
