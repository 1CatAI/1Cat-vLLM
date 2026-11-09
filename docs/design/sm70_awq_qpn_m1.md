# SM70 AWQ QPN single-token operator

## Kernel layer

`_C::awq_moe_qpn_m1_sm70_out` implements the native-group-32 Qwen3.8
TP4 routed-expert geometry. This layer registers an inference-only operator;
it does not select it in the model runtime or change any default route.

The quadpair-N Tensor Core dataflow is derived from the existing NVFP4 QPN
implementation and its retained `LICENSE.v100-skinny` notice. This is a
model-specific AWQ adaptation, not an enablement of a generic Skinny backend.

The two launches are:

1. W13: selected experts in original router order, CTA-local FP32 reduction,
   FP16 gate/up materialization, then FP16 SwiGLU intermediate output.
2. W2: FP32 dot products, per-route FP16 output materialization, then ordered
   FP32 router-weight accumulation into the FP16 output.

No selected-weight bank, input replication, checkpoint rewrite or persistent
weight copy is added. The operator consumes the existing prepared banks and
caller-owned intermediate/output buffers.

### Admission contract

| Argument | Shape | Type |
| --- | --- | --- |
| Input / output | `(1, 2560)` | FP16 |
| Intermediate | `(10, 160)` | FP16 |
| W13 prepared weight | `(512, 2560, 40)` | INT32 |
| W2 prepared weight | `(512, 160, 320)` | INT32 |
| W13 4-byte metadata | `(512, 80, 320)` | INT32 |
| W2 4-byte metadata | `(512, 5, 2560)` | INT32 |
| W13 3-byte metadata | `(512, 80, 320, 3)` | UINT8 |
| W2 3-byte metadata | `(512, 5, 2560, 3)` | UINT8 |
| Expert IDs / router weights | `(1, 10)` | INT32 / FP32 |

All arguments must be contiguous on the same SM70 CUDA device. Weight,
metadata and activation pointers require 16-byte alignment; IDs and router
weights require 4-byte alignment. Output/intermediate must not overlap each
other or any input. Negative or out-of-range expert IDs contribute zero;
duplicate valid expert IDs retain their separate router weights.

The 4-byte layout stores the existing FP16 scale and rounded FP16 bias.
The 3-byte layout stores a FP16 scale and UINT8 zero point. The kernel-only
base reads it scalarly; the independent cooperative-load layer below changes
only W13 metadata reads. Bias is reconstructed at the same FP16 boundary. Dequantization
retains `half_fma(q, scale, half(-zero * scale))`; replacing this with
`half((q - zero) * scale)` is not an equivalent rounding contract.

## Numerical boundary and tests

The acceptance objective is **controlled additional numerical perturbation**.
Both implementations approximate the same computation; legacy output is not
ground truth. Holding the AWQ checkpoint fixed separates the arithmetic
change from the quantization error already present in both paths. Compare
each path against the same independently decoded-weight/FP64 reference,
then assess their paired differences and task-level behavior. This does not
measure either path's error against the original unquantized model.
Passing the retained local bounds means both paths have bounded error on
the tested inputs, not that their errors are identical or cancel. Whole-model
quality and amplification across states/routing remain separate checks.

The CTA-local reduction changes FP32 summation order relative to the legacy
TurboMind split-K route. Bitwise equality to legacy AWQ is not promised, and
neither path is declared the mathematical reference merely because it existed
first. Full-model acceptance must examine fixed-prefix raw logits and paired
quality, separately from speed and free-running token-stream equality.

Run the portable prepared-layout test on a V100 native build:

```bash
.venv/bin/python -m pytest -q tests/kernels/test_sm70_awq_qpn_m1.py
```

It independently constructs prepared metadata/weight tiles for both layouts,
checks one-hot reads across K/group boundaries, an FP64 W2 dot reference with
explicit FP16 rounding allowance, changing CUDA Graph inputs, duplicate and
invalid expert IDs, aliased/misaligned arguments, and the registered fake op.
It requires SM70; a CPU skip is not a GPU test pass. Shape-specific kernel
tests do not by themselves establish full-model quality or throughput.

## Runtime layer

`VLLM_SM70_AWQ_QWEN38_QPN_M1` defaults to `1` at model initialization for
supported layer contracts and native builds. Set `0` for rollback; other
values are rejected. Use a native build containing the operator
and restart the engine after changing the setting, including for rollback.
Changing an environment variable does not replace an already captured graph.
There is no research sidecar, runtime compilation or external DSO loader.

Admission requires the existing TP4/E512/native-group-32 geometry, batched
TurboMind weights, interleaved W13 and the legacy single-token compact path.
Both the checkpoint and prepared group sizes must be 32. An explicit opt-in
with an unsupported layer contract or missing native operator fails closed.
When the variable is unset, those cases automatically retain the legacy route.
Admission uses tensor geometry and capabilities, not checkpoint identity.
At execution, only contiguous FP16 `(1, 2560)` inputs with ten INT32 expert
IDs and FP32 router weights select QPN. Other physical batch sizes, including
padded CUDA Graph batches, retain their existing route. This does not change
grouped decode, prefill, attention, shared experts, router selection or MTP.
The existing prepared banks and per-call buffers are passed by reference.

Run CPU admission and neighboring dispatch tests:

```bash
.venv/bin/python -m pytest -q \
  tests/quantization/test_awq_qpn_sm70.py \
  tests/quantization/test_sm70_awq_active_grouped_decode.py \
  tests/quantization/test_sm70_awq_indexed_prefill.py \
  tests/quantization/test_sm70_awq_compact_metadata.py
```

### Validation snapshot and limitations

The September 5-6, 2026 investigation used four V100s, TP4/MTP0, the same
native-group-32 AWQ checkpoint and frozen prompt token IDs, FP16 activations
and KV, 4-byte metadata, prefix caching off, and `ignore_eos=false`. The
runtime inherited the separate grouped-decode change and used a separate
QSA page4 logical-order fix in both arms. Those are baseline dependencies,
not changes made by this proposal. Raw artifacts distinguish the built core,
Python sources, QSA extension, checkpoint, tokenizer and evaluation tools.

Uninstrumented full-model cells were each measured once with an output limit
of 320 tokens; these are observations, not confidence intervals. The metric
below is aggregate **pure-decode** throughput, excluding prefill overlap:

| Workload | QPN off (tok/s) | QPN on (tok/s) |
| --- | --- | --- |
| C1 x 64K | 50.0326 | 59.0313 |
| C4 x 64K | 131.3485 | 129.5357 |
| C8 x 16K | 246.3619 | 245.7809 |

The separate NVFP4 reference was 60.2609 tok/s for C1 x 64K: about a 2.04%
gap, not exact parity. All these full-model numbers use 4-byte metadata;
they must not be attributed to cooperative 3-byte metadata loading. The
older 54.6544 tok/s baseline and its 68.3180 tok/s (+25%) research target
remain distinct from this paired experiment; that target is not achieved.
Instrumented dual-path diagnostic timings are not performance measurements.

The initial cross-process quality pair had an IFEval pass-to-fail change
(4/5 to 3/5); its unchanged-route controls also differed, so it could not
isolate QPN as the cause. A later same-runtime 65-case pair recorded:

| Evaluation subset | QPN off | QPN on |
| --- | --- | --- |
| HumanEval | 5/5 | 5/5 |
| MBPP | 4/5 | 4/5 |
| IFEval strict | 3/5 | 4/5 |
| GSM8K | 29/32 | 28/32 |
| Tool selection | 10/12 | 10/12 |
| Needle retrieval | 6/6 | 6/6 |

There was one GSM regression and one IFEval improvement; 47/65 output token
streams were exact. This is **not** a zero-regression or statistically
non-inferior quality result. The original budgets and failed cases remain
part of the evidence, rather than being replaced by the focused diagnostic.

For attribution, the focused same-process/same-graph experiment executed
both local MoE implementations on identical inputs/routes and selected the
returned output with a device flag. Shapes were prewarmed and GEMM LUTs
remained fixed. Same-arm repeats and unchanged-route C4 controls were exact.
Independent checkpoint/FP64 checks covered 1,152 captured layer/rank/arm
samples at three fixed prefixes; both paths' W13 and W2 passed the retained
rounding bounds. Neither implementation was used as the other's oracle.

The GSM first divergence occurred at index 120, where the legacy logits for
token IDs 4003 and 16526 were exactly tied at 21.59375. The focused legacy
run also reproduced the earlier QPN failed answer at the unchanged 256-token
budget. This supports numerical trajectory variation, not a demonstrated
kernel defect. It does not make a truncated or incorrect answer acceptable
by definition.

Teacher forcing prevents different sampled prefixes from confounding the
comparison; it does not force hidden states or expert membership to match.
At the captured 64K step, 13/48 layers changed expert membership while each
selection respected its own scores. The first change reversed a 0.01171875
score margin. Global logit differences can therefore be larger than local
rounding errors. The focused 64K maximum was 1.751953, and an earlier full
trace reached 5.052734; these are not described as a few ULPs. Not every
historical worst-logit step was captured for independent local replay.

The analysis tool was also corrected to use `argmax`, not the first index
from `topk(2)`, for the greedy tie rule. Reanalysis found an IFEval flip at
index 80 where QPN's top two logits tie; an earlier short-prompt flip was
a tie-order reporting artifact. Raw tensors and task scores did not change.

The original runtime proposal was opt-in. The tested local numerical
evidence does not justify rewriting the kernel solely to reproduce legacy
tokens, but broad quality non-inferiority and production readiness remain
unproven. Future acceptance must retain task-level quality checks alongside
numerical bounds; neither single-question changes nor their aggregate
cancellation alone settle that decision.

The September 6 integration enables the measured shape by default under the
repository owner's acceptance policy: preserve the observed C1 speedup and
independently checked arithmetic without requiring greedy equality. The
65-case quality observations and their limitations above remain unchanged;
this policy decision is not a new model benchmark or a statistical quality
claim. Both the grouped-decode baseline and the QSA allocation-order repair
must be retained when comparing to that paired evidence. The integration
adds CPU coverage for implicit unsupported-shape/build fallback, explicit
rollback and interaction with the loading-cache lifecycle regression.

## Cooperative 3-byte metadata layer

W13 loads one 96-byte tile using 24 aligned 32-bit loads per warp, then
shuffles the packed words to reconstruct each lane's 3-byte record. The last
record (byte offset 93) needs no read or shuffle beyond the 24 loaded words.
W2 keeps the scalar 3-byte reader; the 4-byte path is unchanged. Weight
layout, bias reconstruction, MMA order and all rounding boundaries remain
identical to the kernel-only base. No new runtime option or resident buffer
is introduced.

The scalar and cooperative builds each passed the same two portable SM70
GPU tests and 256 real-checkpoint/dynamic-graph comparisons to the validated
prototype, with 72 independent FP64 stage checks and no rounding-bound
violations per build. A single-layer graph probe measured 21.5280 us for
scalar 3-byte W13/W2 and 20.6160 us for the cooperative build (about 4.2%
lower); this is a component observation, not a model-level speedup claim.
The full-model results above use 4-byte metadata and cannot validate or
measure deployment of the cooperative 3-byte path.

Since the cooperative reader removes the speed cost of the 3-byte layout on
the M=1 route, `VLLM_SM70_AWQ_MOE_COMPACT_METADATA` now defaults to `1` for the
supported Qwen3.8 TP4 E512 native-g32 shape. Unsupported builds or shapes fall
back to the 4-byte layout with an info log; an explicit `=1` still fails
closed, and `=0` keeps the 4-byte layout. On the same-day regression contract
(max model length 131,328, GPU memory utilization 0.89) the engine-reported
KV cache grew from 427,385 to 497,983 tokens with C1 x 64K pure decode within
0.2% of the 4-byte layout and C4/C8 within run-to-run noise.
