# SM70 serialized block-FP8 linear integration

Serialized 128 x 128 block-FP8 loading previously selected TurboMind, QPN8,
prescaled decode and exact-dense prefill inside `Fp8LinearMethod`.
`TurboMindFp8LinearKernel` now owns that preparation and dispatch in the existing
scaled-MM kernel framework. Its CUDA priority and `turbomind` provider use the
same selector as upstream FP8 kernels. An SM70 W8A16 configuration is distinct
from an activation-FP8 configuration, so another kernel cannot accidentally
consume a prepared TurboMind layout.

The initial migration preserves the numerical bodies and existing admission
boundaries. It does not change scales, accumulators, precision, workspace sizes,
native operator signatures or tuning tables. QPN8 and grouped-BMM are retained
variants of the weight-only dispatcher. They are not the general block-FP8
activation kernel proposed in #750.

## Configuration and compatibility

`KernelConfig.sm70_fp8` resolves eleven Python policy aliases once per engine.
Explicit fields take precedence. In particular, generic QPN8 rollback still
wins over a specific legacy PP2/TP4 enable. Defaults are resolved after the
existing verifier compatibility defaults, until that category is migrated.
Loading and forward dispatch consume the resolved object and never write the
environment. Unused FP8 policy is excluded from the compilation hash, preserving
NVFP4 and AWQ graph fingerprints.

The old names remain for one version and warn when used for serialized FP8
linears. Other checkpoint admission, MoE, compressed-tensors and online loaders
still share some names; their compatibility reads are not removed in this PR.
Early `Fp8Config.get_min_capability()` admission still occurs before an engine's
kernel config exists. Explicit fields override aliases at layer loading; they
do not bypass that admission check.

Two controls are mirrored as read-only resolved legacy state:
`VLLM_SM70_FP8_PREFILL_FAST_SELECTOR` and
`VLLM_SM70_FP8_GROUPED_BMM_DECODE`. Native host dispatch also reads these names.
Offering a Python override would leave native and Python dispatch inconsistent.
They remain environment controls until native host APIs can carry the values.
This is a routing interface change; no numerical kernel changes are needed.

The common `VLLM_DISABLED_KERNELS` mechanism can disable this class. If the
weight-only layout has no remaining implementation, the common selector fails
with its reasons rather than handing prepared codes to an incompatible kernel.
The existing backend preference still decides whether the weight-only loader
is used; an explicit incompatible `linear_backend` now receives the selector's
normal error. These explicit selector controls are new, separately tested
behavior; the preserved default and legacy-override matrix has zero differences.

## Capability and retained validation limits

`can_implement()` checks the weight-only configuration, FP16 activations, block
scale format and native prepare operator. Existing local layout and native
variant checks remain beside preparation. Model identity is not added to
capability checks.

The repeated PP2/TP4 single-request qualification is now defined once in
`models/config.py`. It is a measured pipeline/workspace boundary, not proof that
the GEMM needs that scheduling layout. Widening it requires simultaneous-stream
workspace checks, paired complete outputs and matched pipeline decode timing.

The ordinary QPN8 projection-name whitelist can be replaced by block layout,
shape and fused-output checks after tests cover renamed and additional
projections. Exact shapes in the QPN8 tuning table are measured tuning entries;
the table already has a generic fallback and is not an admission whitelist.
Shared-expert fused activation retains its exclusion because the existing
qualification notes explicitly identify it as numerically unsafe. Reversible
FP16 exponent shifts remain checked at weight loading, with UE8M0 and pipeline
qualification retained. No untested boundary is widened here.

The 35B-A3B AWQ/FP8 end-to-end gate needs those actual checkpoints. NVFP4 35B or a
30B FP8 checkpoint cannot substitute for the requested model/quantization pair.
Do not promote a route based on the CPU snapshots alone.

## Regression command

```bash
python -m tools.sm70_route_snapshot --category fp8 --baseline-ref BASE_SHA
```

The tool imports the actual historical quantization module independently of
the candidate. CPU/meta tensors and native doubles record preparation and
opaque dispatch calls, including scalar plans and fused activation. The 324
standard configurations are supplemented by 216 dense/MoE FP8 configurations
and ten override, missing-operator, workspace and PP2/TP4 edges. Plans are
stored once by content hash to keep the retained snapshot readable and small.
This proves Python dispatch equivalence; it does not execute CUDA arithmetic,
native tuning or complete model quality checks.

A pure migration expects zero differences. A measured widening records only its
approved row IDs with `--expected-changes`. The GPU acceptance gate remains an
installed source-complete wheel pair, release-profile 27B 32K C1/C4, an FP8
route/numerical oracle, and paired MBPP, needle and Chinese QA requests.
