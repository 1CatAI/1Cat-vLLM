# SM70 linear kernel integration

SM70 acceleration uses the existing linear kernel framework. Native dtype,
layout, shape and operator checks belong beside the implementation in
`vllm/model_executor/kernels/linear/`. A kernel implements `is_supported` and
`can_implement`, returning a reason when it cannot run. The normal priority
list selects the implementation; `VLLM_DISABLED_KERNELS` disables a class.

The first migration covers compressed-tensors NVFP4 weight-only linears:
`Qpn4NvFp4LinearKernel`, `Qpn2NvFp4LinearKernel` and
`TurboMindNvFp4LinearKernel`, in that priority order. Each uses the existing
native arithmetic and prepared layouts. The quantization scheme still loads
checkpoint parameters and converts its global scale representation; the
kernel owns preparation and execution. Other formats, MoE, attention,
DFlash2 pipelines and collectives remain separate migration scopes.

## Configuration and qualification

### Turing ModelOpt linears

On SM75, the common selectors can choose `TuringQpn2NvFp4LinearKernel`
for checkpoint-native NVFP4 and `TuringQpn8Fp8LinearKernel` for static
per-tensor E4M3 weights. Both keep activations and output in FP16 and require
the corresponding native QPN operators. NVFP4 accepts positive output widths
with zero padding to a 32-column tile and K divisible by 128; FP8 requires
positive N divisible by 32 and K divisible by 128. These contracts do not
depend on model names, tensor parallel size or speculative width.

The paths are selected by default when supported. `sm70_nvfp4.dense_qpn2`
and `sm70_fp8.enabled` can disable their respective routes through
`--kernel-config`; `VLLM_DISABLED_KERNELS` can disable an individual class.
The `turbomind` provider includes these registry-independent QPN kernels.
Legacy backend disable overrides remain compatible. The new NVFP4 policy
participates in compilation identity, invalidating older cached graphs.

NVFP4 retains one QPN2 layout: rows up to 32 use native decode, and larger
batches dequantize into invocation-owned FP16 storage for dense GEMM. FP8
uses the native QPN8 dispatcher with invocation-owned dense scratch. Runtime
dispatch happens inside opaque operations, and cached graphs retain no
serialized workspace address. Volta retains its existing implementations.

This support concerns linear weights. It does not establish FP8 KV-cache or
attention-backend support on Turing. A conservative serving configuration
uses `--dtype float16 --kv-cache-dtype float16` with an attention backend
supported by the installed build, such as `TRITON_ATTN`.

`KernelConfig.sm70_nvfp4` resolves once before layers load. Runtime QPN2
execution consumes those resolved values; it does not re-read environment
variables. This state participates in `KernelConfig.compute_hash` and is
carried with the engine configuration to workers. Creating another engine
cannot change the first engine's resolved QPN2 values.

Explicit configuration wins over deprecated environment aliases. For example:

```bash
vllm serve MODEL --kernel-config '{"sm70_nvfp4":{"qpn2":false}}'
VLLM_DISABLED_KERNELS=Qpn2NvFp4LinearKernel vllm serve MODEL
```

`--linear-backend turbomind` identifies the SM70 CT weight-only provider.
The first migration retains the CT weight-only path's earlier precedence over
activation-quantized NVFP4 selection. It does not change legacy Marlin,
emulation, batch-invariant or format override precedence as an incidental
cleanup. Expanding explicit backend selection to other SM70 formats is a
separate migration with its own before/after route table.

The new class names intentionally make `VLLM_DISABLED_KERNELS` effective for
these CT paths: disabling QPN2 falls back to TurboMind, and disabling every
eligible implementation reports an error. The new `turbomind` backend spelling
is accepted for this provider. These explicit-control additions do not enlarge
the default admitted configurations.

Five existing names remain compatible for one release:

| Deprecated name | Configuration field |
| --- | --- |
| `VLLM_SM70_NVFP4_QPN2` | `sm70_nvfp4.qpn2` |
| `VLLM_SM70_NVFP4_QPN2_PREFILL` | `sm70_nvfp4.prefill` |
| `VLLM_SM70_NVFP4_QPN2_SHARED_WEIGHT` | `sm70_nvfp4.shared_weight` |
| `VLLM_SM70_NVFP4_QPN2_SHARED_SCALES` | `sm70_nvfp4.shared_scales` |
| `VLLM_SM70_NVFP4_QPN2_PREFILL_MIN_M` | `sm70_nvfp4.prefill_min_m` |

Reading an alias warns once. Remove aliases in the release after their full
compatibility release, together with their registrations and release notes.
The prefill threshold retains 1024; changing it remains an explicit user
configuration. Existing split-K and accumulator settings move verbatim into
the implementation's tuning table. This migration does not retune them.

Operator capability and quality qualification have different owners.
`can_implement` checks the packed local K/N layout and required native symbols.
The retained draft/state qualification, QPN4 C1/no-speculation boundary, dense
projection dimensions and projection-role whitelist live in
`model_executor/models/config.py`. They retain the previous admitted set;
renaming a projection currently produces a visible qualification rejection.
Removing that restriction requires a measured broadening change. Hardware,
model revision, precision, KV and accepted-state behavior must be recorded.

Compact scales require the existing native scratch ABI. Batch GEMM layouts
supersede compact scales because large-M TurboMind consumes the prepared FP16
scales. A true `shared_scales` configuration is permission to use that layout,
not a promise that the loaded layer selected it. Selection reasons come from
`select_sm70_nvfp4_linear_kernel`; the startup acceleration report exposes the
resolved policy as `runtime_guarded`, without asserting a request kernel hit.

The rest of SM70 still has legacy environment policies. This first migration
isolates QPN2 policy; it does not claim complete multi-engine isolation for
unmigrated attention, GEMM tuning or collective routes.

## Route regression command

Run from a checkout with the normal Python dependencies installed:

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m tools.sm70_route_snapshot \
  --check tests/quantization/data/sm70_nvfp4_routes.json
```

For an independent comparison with the parent implementation:

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m tools.sm70_route_snapshot \
  --baseline-ref PARENT_SHA --output routes.json
```

The tool executes both complete CT NVFP4 loading/dispatch adapters with CPU
and meta tensor doubles and records the native entrypoint plus its scalar
plan. The 324 configurations cover three model/format scopes, three KV types,
TP2/4, no speculation/MTP/DFlash, concurrency 1/4/8 and budget 4096/8192.
The Flash-Next ModelOpt and 35B AWQ scopes are recorded as not applicable to
this CT category; their selectors are migrated and tested in later PRs.
These fixtures do not validate every EngineArgs combination, and do not
prove GPU arithmetic or graph replay.

The 18 edge cases cover QPN4 preparation and workspace failure, missing native operators, old compact-scale ABI, independent
codes, batch/compact precedence, padding, unsupported K, new projection names
and all five deprecated overrides. CPU unit tests additionally check disabled
kernel names, reasons, hash differences and engine isolation. GPU validation
uses the release profile's matched 27B 32K C1 and C4 service workloads.

A pure structural PR must leave route snapshots unchanged. A measured
broadening PR passes `--expected-changes changes.json`, an exact JSON list of
changed row IDs, and supplies quality and speed evidence for those rows.
Unexpected additions and missing intended changes both fail. The retained
snapshot must be generated from the independent baseline before changing
implementation or expected rows.

## AWQ dense linear migration

`TurboMindAwqLinearKernel` uses the existing `MPLinearKernel` lifecycle and
CUDA priority list. The AWQ loader retains checkpoint loading and its
architecture-specific fallback; preparation, bounded exact-dense scratch and
execution belong to the kernel. The legacy AWQ GEMM packing differs from the
GPTQ packing accepted by other MP kernels, so the selector filters that layout
family before applying provider priority. Native arithmetic is unchanged.

`KernelConfig.sm70_awq` resolves the dense controls once. The three legacy names
remain compatible for one release:

| Legacy name | Configuration field |
| --- | --- |
| `VLLM_SM70_AWQ_TURBOMIND` | `sm70_awq.enabled` |
| `VLLM_SM70_AWQ_PREFILL_EXACT_DENSE` | `sm70_awq.prefill_exact_dense` |
| `VLLM_SM70_AWQ_MLP_ENGINE` | `sm70_awq.fused_silu` |

The shared TurboMind switch still controls unmigrated AWQ MoE. The dense
compatibility warning states that scope. Format admission and legacy Marlin
conversion remain in the quantization config. `VLLM_DISABLED_KERNELS` can
disable `TurboMindAwqLinearKernel`, selecting the retained Triton fallback.
Missing native preparation and unsupported group sizes retain fail-closed
behavior. The default exact-dense role boundary is recorded in model policy;
the fused epilogue remains experimental with its existing TP2/M1 limit.
Neither qualification is broadened without an AWQ quality pair.

Unused AWQ policy is excluded from the graph fingerprint. Actual AWQ policy is
resolved before compilation and included in the fingerprint. An independent
pre-migration hash fixture protects NVFP4 from changes to an unused format.

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m tools.sm70_route_snapshot \
  --category awq --baseline-ref PARENT_SHA
```

The AWQ category executes the independent legacy loader and candidate loader,
recording preparation and native decode/prefill arguments. It covers the same
324 model/KV/TP/speculation/concurrency/budget scopes and ten boundary cases.
NVFP4 models are explicitly inapplicable to this format; these doubles do not
replace native arithmetic or whole-model AWQ validation.

## Adding another optimization

1. Extend an existing kernel or add an implementation to its existing
   provider/format registry. Keep capability checks next to the operator and
   return a concrete rejection reason.
2. Put whole-model quality restrictions in the corresponding model config
   policy. Explain the unvalidated case and the smallest gate that can admit it.
3. Carry resolved engine decisions through `KernelConfig` or the relevant
   configuration object. Do not write `os.environ` to communicate a decision.
4. Use measured tuning constants/tables. User-facing options must explain an
   actual control; prefer configuration fields over new environment variables.
5. Extend the route snapshot for the migrated category. Record intentional
   changes, native ABI and numerical contract, CPU checks and matched GPU
   quality/performance. Add the same fields to the PR description.

Attention variants use `validate_configuration`; model-specific adjustments
use `VerifyAndUpdateConfig`. MoE uses its kernel/backend selectors. Pipelines
and collectives outside those frameworks should follow the same capability,
policy and reason interfaces. There is no parallel SM70 acceleration registry.

The migrated-variable guard complements the environment registration check
in PR #711. It rejects reading migrated names outside the configuration
compatibility adapter, including variables that are already registered.

## Capability moves and qualification backlog

The following checks can be owned by existing implementations without
enlarging the admitted routing set:

| Check | Owner | Evidence required to enlarge its admitted set |
| --- | --- | --- |
| Local packed K/N, rank, padding and gated alignment | Linear `can_implement` | Native arithmetic comparison at the newly admitted local shapes |
| Native symbols and compact-scale ABI revision | Linear implementation | Installed-artifact symbol/ABI smoke |
| Attention head dimensions, KV dtype and page layout | Attention `validate_configuration` | Native attention oracle and captured-graph checks at each new layout |
| Hardware architecture and workspace admission | Kernel/backend implementation | Actual hardware route hit and bounded-memory check |

Whole-model restrictions retain their existing routes until quality and speed
evidence admits a particular replacement:

| Restriction/candidate | Why it is retained | Smallest promotion gate |
| --- | --- | --- |
| DFlash2 selector, draft length and state/ubatch limits | Accepted-state and verifier behavior exceed a local GEMM contract | Same-request token/logit and acceptance comparison; MBPP subset, needle and Chinese QA; matched C1/C4 decode |
| Flash-Next MTP4 model/layout limits | State tracking, MoE and MTP quality were audited together | MTP acceptance and natural-stop outputs; the same quality sets and matched decode contract |
| QPN2 projection-role whitelist | Existing real-weight audit covered the listed roles | New-role native oracle, followed by the model quality gate; no new numerical precision |
| QPN4 dense dimensions and C1/no-speculation limit | Existing bounded-workspace and full-model audit has that scope | New-shape oracle, memory admission, graph replay and speculative-state quality |
| QWEN38 batch fast path | The +29%/+18% report used a historical pre-repair candidate whose output-parity gate failed | Audit current repaired source first; paired quality sets and C2/C4 speed, with C1 control |
| #703 opt-in batch GDN paths | Complete MTP4 validation exists for the measured contract; extra resident layouts change the memory budget | Reproduce that contract, verify current symbols and quality, then measure KV capacity and C1/C4 before default promotion |

Screen #703's measured contract first, then QPN2 role restrictions and repaired
QWEN38 batch paths according to validation cost. A token difference alone is
not a quality failure under the maintainer's accepted criteria; numerical
changes require the stated quality-set gate. Reducing accumulation precision
requires a separate maintainer discussion.

The initial structural change deliberately makes no promotion decisions.
Moving a guard into model policy is not evidence that its untested scope is
safe. Subsequent category PRs add their own full route capture to the command.
