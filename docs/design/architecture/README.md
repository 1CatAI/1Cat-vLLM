# 1Cat-vLLM architecture and maintenance

This is the current architecture entry point. Component READMEs describe
maintained contracts; phase reports retain historical decisions and measurements.
Phase E starts from main `18784e027` (D1–D6 merged). Merged scope is not a claim
that every model, topology or experiment is qualified. PLE and DDTree follow-up
remain deferred.

For implementation steps and validation commands, use the
[1Cat development guide](../../contributing/1cat-development.md).

## Responsibilities and dependency direction

```text
initialization: compatibility inputs -> ordered defaults / validation
                                    -> engine-owned typed policy
model adapter: eligibility, weight relationships, state contract
platform/provider: device, format, layout, dtype and native capability
assembly: bind existing selector, operations and resource owners
execution: dynamic inputs -> selected stages -> state commit
observations: existing checkpoints -> engine-owned diagnostics
```

Runner V1 and V2 retain their selection and execution order. Models declare
contracts; providers implement them; runners sequence execution. Shared stages
receive prepared inputs and explicit bindings, not an entire runner or a callback
into an old format-specific implementation. Resource owners retain buffers,
events and workspaces; contexts borrow them without copying ownership. Dynamic
token counts, request order and prefill/decode decisions stay at their existing
safe boundaries.

Maintenance rules:

1. Model qualification belongs in model adapters; hardware/native admission
   belongs in platform or provider code. Generic modules expose existing
   extension points. The layering ratchet prevents new debt; old coupling remains
   visible rather than being declared eliminated.
2. Capture compatibility inputs during initialization. Explicit typed settings
   win, subject to preserved safety constraints and validation order. Workers
   receive resolved policy; execution does not reread legacy settings.
3. Put parameters in their responsible configuration below. Computation affects
   effective cache hashes; diagnostics do not. Do not communicate policy through
   process-environment writes or introduce a second global configuration store.
4. Reuse codecs and stages while retaining real layout, algorithm and rounding
   differences. New codecs still need writers/readers, bindings, admission,
   fallback and tests; registration alone enables no native route.
5. Reuse existing selectors and declarations. Preserve compatible route labels,
   including historical format-bearing names. Static declaration and selector
   prediction are distinct from observed native execution.
6. Keep one implementation and mutable resource owner per responsibility.
   Compatibility modules forward to that owner. Default-off experiments remain
   available; deprecation requires a reason, evidence and replacement.

## Where things live

| Responsibility | Maintained entry / extension point |
| --- | --- |
| KV storage identity and conversion | [KV codecs](../../../vllm/v1/attention/kv_codecs.README.md); independent of weight quantization |
| Flash-V100 selection and execution | [Attention component](../../../vllm/v1/attention/backends/flash_v100/README.md); assembly, executors and feature-owned metadata |
| Shared-key sparse index scoring | [SM70 QSA provider](../../../vllm/v1/attention/ops/sm70_qsa_shared_key.py); model-independent tensor admission and native row tiles; former model module forwards imports |
| MoE stages, codecs and resources | [SM70 MoE component](../../../vllm/model_executor/layers/fused_moe/sm70/README.md); existing `sm70_moe_router` selector |
| Linear dispatch and QPN | `vllm/model_executor/kernels/linear/`, including `qpn/`; existing selector and prepared provider state |
| Native SM70 bindings | `vllm/_sm70/`; `_sm70_ops.py` is a compatibility facade |
| GDN computation | `vllm/model_executor/layers/fla/ops/`; existing selector produces `GdnExecutionPlan`; model layer adapts projections and stages |
| GDN metadata and state | `vllm/v1/attention/ops/gdn_state.py`, `GDNSpecDecodeStateContract` and `ModelState`; builders own metadata, model state owns conv/SSM updates |
| Model defaults and shared weights | `vllm/model_executor/models/runtime_defaults.py`, `shared_weights.py`, and existing `MODELS_CONFIG_MAP` adapters |
| Platform defaults | `vllm/platforms/runtime_defaults.py` and existing platform hooks |
| Ordered configuration resolution | `vllm/config/policy_defaults.py`; retain early defaults and late validation checkpoints |
| Runners and ordinary speculation | `vllm/v1/worker/gpu_model_runner.py`, `vllm/v1/worker/gpu/model_runner.py` and `BaseSpeculator`; ordinary DFlash uses V2 |
| Warmup and staged input | `vllm/model_executor/warmup/plan.py`, `sm70_runtime.py`, and `vllm/v1/worker/runtime/input_transfer.py` |
| Graphs, communication and layer providers | Existing graph dispatcher/runner and communicator; `kernels/lm_head/`, `kernels/norm/`, `kernels/linear/`; graph tables and IPC lifetimes retain their owners |
| Diagnostics and resource binding | `vllm/diagnostics.py`, `vllm/runtime_resources.py`; engine-owned budgets, observations and runtime resources |

## Configuration ownership

| Policy | Canonical configuration |
| --- | --- |
| Linear, MoE, GDN and provider computation | `KernelConfig` family fields, `sm70_moe`, `gdn`, `layer_execution`, `sm70_sparse` |
| Attention algorithms and native resources | `AttentionConfig.flash_v100`; existing shared graph/attention fields remain authoritative |
| Ordinary speculative sampling / DFlash2 | `SpeculativeConfig.sampling_policy` / `sm70_dflash2` |
| Graph selection / collectives | `CompilationConfig.runtime` / `ParallelConfig.communication` |
| Warmup and staged input | `KernelConfig.sm70_runtime` |
| Trace, compare, dump and timing | `ObservabilityConfig.runtime_trace` and existing diagnostic/profiler fields |
| PLE placement | `OffloadConfig.ple`; outstanding PLE test adaptation is deferred |

Legacy names are initialization adapters. The
[environment reference](../../configuration/env_var_reference.md) documents their
metadata; [Phase D](sm70_phase_d.md) records parsing, provenance, effective hashes,
native ABI and retained process/standalone boundaries.

## Generalizing existing paths

Keep the owners above as the integration targets. New models supply layout,
head geometry, codec and state requirements to those owners; they should inherit
qualified implementations without copying another model's dispatch chain.
Consolidating ownership and widening qualification are separate deliverables.
The following work queue records decisions, not a second runtime registry.

Classify a restriction before changing it:

| Restriction | Required action |
| --- | --- |
| Hardware, ABI or layout correctness | Implement the missing capability or retain a reasoned fallback; deleting the guard does not add support |
| Implementation coverage, such as one query tile | Extend the existing algorithm over a range with tails and lifecycle handling |
| Numerical/model qualification | Use bounded operator error and representative task-quality evidence; identical generated tokens are not a universal requirement |
| Performance tuning, such as a crossover or measured geometry | Keep tuning separate from correctness admission; compare the affected real route before selecting a new range |

Prioritize a missing capability that blocks multiple inputs or models. Keep
preserved cases as controls and test the changed boundary/interior plus the
affected startup or graph interaction. Stop after the decision is established;
do not remeasure unrelated historical paths.

| Family / current source | Restriction to resolve | Reuse target and next decision |
| --- | --- | --- |
| [QSA shared-key scoring](../../../vllm/v1/attention/ops/sm70_qsa_shared_key.py) | Old native ABI handles one eight-query tile; FP16/H4/D128 and one request remain implementation constraints | Common attention provider now tiles query rows; validate the added row range before qualification. Older binaries keep their row limit and fallback |
| [QSA Top-K](../../../csrc/qsa_lexicographic_topk.cuh) | M5/M10 score-capacity gap between short MTP and long compact routes | First compare the same selector around 9216/9217 and 32767/32768 score columns, distinguishing live length from capacity; extend interval selection only if useful |
| [QSA sparse page4](../../../vllm/models/qwen4_exp/nvidia/ops/qsa.py) | H6/D256, index budget and page layout | Parameterize the existing sparse attention provider after checking its native layout; preserve exact selection/masking semantics for the supplied scores |
| [Grouped long attention](../../../vllm/v1/attention/ops/sm70_grouped_long.py) | D256/GQA6, fixed per-request query width in batches, context qualification | Extend request metadata and existing workspace/graph plan to ragged groups; do not conflate a certified context limit with a hardware limit |
| [AWQ active grouped MoE](../../../vllm/model_executor/layers/quantization/awq_sm70_moe.py) | Model geometry, TP4, group32, top-k and M2..8 bundled in qualification | Separate native geometry capability from quality/tuning in the existing MoE plan; qualify one useful neighboring geometry against the same AWQ/FP8 migration criteria |
| [DFlash2 FP16 projections](../../../vllm/model_executor/layers/quantization/sm70_dflash2_fp16.py) | Five measured geometries and M8-specific packed provider | Reuse the linear preparation/provider interface; retain geometry data as tuning and keep the ordinary GEMM fallback. Avoid another model-specific linear dispatch chain |
| [GDN native verifier](../../../vllm/model_executor/layers/fla/ops/sm70/gdn_verify.py) | One request, at most eight rows, K/V128 | Extend the existing `GdnExecutionPlan` provider and metadata/state contract; qualify batches and tails before attempting a new head dimension |
| [DFlash2 defaults](../../../vllm/model_executor/models/config.py) and [online QPN8](../../../vllm/model_executor/layers/quantization/sm70_online_qpn8.py) | Model-quality packages and explicit precision conversion | Separate independently reusable operators from package qualification; evaluate precision changes through task quality, preserving the fallback |

### First delivery: shared QSA scorer

The public provider accepts tensor metadata without a model name. The old model
import is a compatibility facade. The packaged CUDA operator uses an eight-row
query tile over `grid.y`, with a masked last tile and 64-bit output offsets.
`MAX_QUERY_ROWS` reports the native launch bound; it is not a measured throughput
or model-quality envelope. Native libraries predating this capability remain
limited to eight rows, even with newer Python.

CPU admission tests cover tile boundaries, invalid metadata and old binaries.
The 22 CPU cases and 19 distinct GPU cases pass. GPU checks exercise non-aligned
score widths, query tails, padded rows, changed live lengths and replay, against
both the existing scorer and independent native tiles. Actual 32K/256K scorer
and score-to-selection measurements are recorded in the
[migration control log](../sm70_v100_migration_control.md#shared-path-generalization-qsa-query-tiles-2026-10-11).
Model task quality, startup parity and model speed remain unqualified. The
change does not establish multi-request native scoring or other-head support.

```bash
.venv/bin/python -m pytest --confcutdir=tests/v1/attention -q \
  tests/v1/attention/test_sm70_qsa_shared_key_admission.py
# Build/install standard _C, _C_stable_libtorch and _sm70_qsa_indexer_C targets:
.venv/bin/python -m pytest --confcutdir=tests/kernels -q \
  tests/kernels/test_sm70_qsa_shared_key.py
```

The existing scorer benchmark accepts query rows and live length independently
of allocated capacity. Its defaults retain the former M5/C1 and M20/C4 workloads.
It records actual native dispatcher calls outside timed capture, numerical error
and same-process ABBA CUDA Graph timings. Use the model's 204-key page layout to
compare actual 32K and 256K prefixes; larger capacity alone does not test a
longer prefix:

```bash
.venv/bin/python -m benchmarks.kernels.benchmark_sm70_qsa_shared_key \
  --out /tmp/qsa-scorer-32k.json --component scorer --requests 1 \
  --query-rows 8 9 17 33 65 --page-size 204 --pages-per-request 41 \
  --live-length 32768 --layers 12
.venv/bin/python -m benchmarks.kernels.benchmark_sm70_qsa_shared_key \
  --out /tmp/qsa-scorer-256k.json --component scorer --requests 1 \
  --query-rows 8 9 17 33 65 --page-size 204 --pages-per-request 322 \
  --live-length 262144 --layers 12
```

The benchmark records `rtol=2e-6, atol=1e-5` for FP32 scores from FP16 inputs.
An FP64 diagnostic established rounding error in both scorers; unchanged unit
fixtures still pass their tighter comparison. For changed selection rows, each
arm must produce the exact Top-K set for its own scores, retain causal tails and
padding, and confine exchanges to score-rounding uncertainty. Overlap and
boundary gaps are reported; malformed-tail and non-Top-K negative controls are
rejected. This operator criterion does not replace model task-quality evidence.

## Source-derived reference

The [architecture reference](runtime_reference.md) is generated from reachable
configuration types, KV codecs, attention routes and MoE stage declarations.
It describes source contracts, not effective engine policy or observed launches.
After changing those declarations, run:

```bash
.venv/bin/python -m tools.generate_architecture_reference --write
.venv/bin/python -m tools.generate_architecture_reference --check
```

The default invocation checks without writing. `--json` emits the same facts.
Checks cover this overview, the three component READMEs, the INT8 proposal, E
record, generated reference, development guide, contributing entry points and
PR template. Historical evidence is linked, not regenerated
or subjected to a repository-wide link cleanup.

## Inspection and evidence

Run from the repository root with the project environment:

```bash
.venv/bin/python tools/pre_commit/check_layering.py --report
.venv/bin/python -m tools.config_inventory --check
.venv/bin/python -m tools.config_inventory --json
.venv/bin/python -m tools.sm70.flash_v100_audit
```

These are source audits. The layering report includes the full parameter
inventory; avoid repeating that inventory just to collect the same evidence.
[B](sm70_phase_b.md), [C](sm70_phase_c.md) and [D](sm70_phase_d.md) explain their
different count scopes. Counts are not per-token reads or observed kernel hits.
The per-file baseline may only decrease: `--update` records reductions;
`--accept-moves` allows redistribution only when aggregate counts do not grow.
Neither permits widening exclusions to conceal new coupling.

## Delivery status and retained work

| Scope | Status / evidence |
| --- | --- |
| A3 attention ownership and common stages | Merged; [record](flash_v100_refactor_progress.md) and [limitations](flash_v100_known_issues.md) retain exact baselines |
| B MoE/linear paths and bindings | Merged; [integration and operator evidence](sm70_phase_b.md) |
| C model/platform, execution and state | Seven deliveries merged; [protocol/resource accounting](sm70_phase_c.md#phase-c-structural-accounting-and-retained-boundaries) |
| D configuration lifetime and deprecation | D1–D6 merged; [closure](sm70_phase_d.md), with PLE follow-up deferred and validation limitations retained |
| E maintained documentation and checks | E1–E3 delivered; [validation and retained boundaries](sm70_phase_e.md) |
| DDTree / PLE follow-up | Deferred; no repair or new qualification in E |
| INT8-G64 | [Design handoff](int8_g64_codec.md) only; not implemented or qualified |

A3's merge does not imply every original A0–A6 proposal was delivered. Native
codec follow-ups and experimental paths need individual evidence. No model
throughput, TTFT or 35B speed conclusion follows from B–D operator acceptance.
The [migration control log](../sm70_v100_migration_control.md) keeps experiments
at their original links. Use component contracts for current development and
follow evidence links for the original decision data.
