# 1Cat-vLLM architecture and maintenance

This is the current architecture entry point. Component READMEs describe
maintained contracts; phase reports retain historical decisions and measurements.
Phase E starts from main `18784e027` (D1–D6 merged). Merged scope is not a claim
that every model, topology or experiment is qualified. PLE and DDTree follow-up
remain deferred.

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
record and generated reference. Historical evidence is linked, not regenerated
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
| E maintained documentation and checks | In progress; [delivery record](sm70_phase_e.md) |
| DDTree / PLE follow-up | Deferred; no repair or new qualification in E |
| INT8-G64 | [Design handoff](int8_g64_codec.md) only; not implemented or qualified |

A3's merge does not imply every original A0–A6 proposal was delivered. Native
codec follow-ups and experimental paths need individual evidence. No model
throughput, TTFT or 35B speed conclusion follows from B–D operator acceptance.
The [migration control log](../sm70_v100_migration_control.md) keeps experiments
at their original links. Use component contracts for current development and
follow evidence links for the original decision data.
