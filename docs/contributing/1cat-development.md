# Developing 1Cat runtime extensions

Start with the [architecture map](../design/architecture/README.md) to choose an
owner. Use the component contracts for behavior, the
[generated reference](../design/architecture/runtime_reference.md) for source
declarations, and phase reports for historical evidence. Update the primary
source of each fact and link to it from other documents.

## Runtime acceptance contract

This contract is required for changes to runtime behavior, model integration,
dispatch, configuration or native loading/builds. Record it in the existing PR
template and component tests; do not create a second capability registry.
Documentation-only changes use the checks below. Reviewers must resolve missing
evidence within the claimed scope before marking a runtime change ready.

1. **Declare coverage before implementation.** Separate the correctness support
   envelope, accelerated subset and measured cases. Include affected dimensions:
   device, model/state contract, weight/KV dtype, layout/head geometry, token and
   request counts, context range, TP/EP/PP, speculation, graph and launch mode.
   State units: live tokens, compressed score columns, allocated page capacity
   and graph padding are different. A 32K measurement proves no other length.
2. **Extend a family path.** New models map their capabilities to existing
   attention, MoE, linear and GDN providers, stages and resource owners. Identify
   reusable stages and the missing capability before adding code. Keep algorithm,
   tiling and fusion specializations behind that family's selector; do not copy a
   model's complete fast path or introduce another dispatcher/configuration
   store. A new provider needs an algorithm/layout/numerical reason and explicit
   bindings. Fewer files or one giant kernel is not a performance acceptance test.
3. **Separate admission, qualification and tuning.** Hardware/layout constraints
   determine whether an implementation can execute; model/state evidence
   determines qualification; measurements choose among eligible implementations.
   Preserve numerical and state contracts. Never remove a safety/quality guard
   or change precision merely to widen a fast path. Each exact-shape restriction
   must identify which constraint it represents and the qualified fallback.
4. **Define startup equivalence.** For supported entry points with equivalent
   effective inputs, resolved computation policy, worker policy, computation
   hashes and numerical/state behavior must agree. Compare values, not command
   text. Defaults versus explicit values, CLI versus `LLM`, legacy inputs versus
   typed configuration, process initialization and warm/AOT cache reuse belong in
   the affected matrix. Graph/eager execution may use different kernels; explain
   intentional differences. A developer shell, import order or stale native
   library must not be required for a released path. Follow the configuration
   and native-binding recipes below for ownership and compatibility checks.
5. **Make boundaries safe and visible.** Every admitted input must reach a valid
   implementation. Outside a specialized range, retain a qualified fallback with
   a diagnosable rejection reason; if no implementation supports the input,
   reject at the earliest boundary with enough information. Do not silently
   change precision. Check interval endpoints, an interior case, valid neighbors
   of each changed threshold, page/tile tails and capacity growth. Explicitly
   account for gaps between fast ranges; a selector flag is not a native hit.
6. **Test affected interactions.** Use cheap selector/configuration cases for
   range coverage, then the smallest GPU cases that exercise changed computation
   and fallback. Include affected interactions among mixed request lengths,
   batch/token counts, prefill/decode/verification, graph capture/replay and state
   reorder/commit. Startup changes need paired fresh-process entry tests; native
   packaging changes need the supported clean artifact entry. Justify omitted
   dimensions. Do not require an unrelated Cartesian benchmark sweep or repeat
   recorded experiments. A regression must remain in the owning test suite.
7. **Keep evidence distinct.** Report selection, observed native execution,
   output/state correctness and measured performance separately. Record each
   source/binary identity; match inputs and effective configuration across the
   baseline and candidate except for the declared change under comparison.
   Preserve the component's deterministic comparison or documented numerical
   tolerance; changing that contract needs separate justification. Performance
   acceptance includes the affected real execution path, with TTFT/prefill and
   pure decode reported separately; a faster isolated kernel can add conversion,
   allocation, synchronization or launch overhead elsewhere. Model-reuse claims
   need a second compatible model's evidence; structural reuse alone does not
   qualify that model. Retain the migration's separate 27B and 35B AWQ/FP8 targets.
8. **Bound incomplete work.** A narrow optimization may be accepted with its
   exact coverage and correct fallback, but must not be advertised as general
   acceleration. Record uncovered intervals/interactions, the responsible
   component and follow-up decision in existing component or migration records.
   Missing evidence remains unverified; it is not a passing result. Default-on
   promotion or a broader support claim requires the corresponding evidence.

For example, extending a score-column fast path through `N` requires cases at
the valid neighbors of `N`, a non-aligned interior length and different live
lengths sharing padded capacity. Record the expected native branch or fallback
for each. A token-context label alone cannot establish that coverage.

## Choose the extension point

The following recipes retain existing selectors, initialization order and
resource owners. They do not broaden model or hardware qualification. PLE and
DDTree work is deferred; INT8-G64 is a design proposal, not a supported codec.

### Add a codec

1. Distinguish KV storage from weight quantization. For KV storage, extend
   [`KV_CODECS`](../../vllm/v1/attention/kv_codecs.py) and follow the
   [KV contract](../../vllm/v1/attention/kv_codecs.README.md). For MoE weights,
   extend the prepared-weight codecs in the
   [SM70 MoE component](../../vllm/model_executor/layers/fused_moe/sm70/README.md).
2. Declare storage/element dtype, scale and packing layout, writer/reader
   semantics, and rounding/accumulation. Bind native operations and retain the
   existing allocator/workspace owner. A descriptor alone does not enable an op.
3. Extend the existing admission and binding declarations. Specify unsupported
   combinations and fallback. Preserve aliases, weight release and old imports;
   do not duplicate an entire attention executor or MoE `apply`.
4. Regenerate the architecture reference. Run the component's minimal tests,
   then writer/reader and capture/replay tests on the supported hardware for an
   actual implementation. Compare deterministic outputs and state exactly.

```bash
.venv/bin/python -m pytest -q tests/v1/attention/test_kv_codecs.py \
  tests/v1/attention/test_flash_v100_routes.py
# For MoE codecs instead, start with the component's mainflow/resource tests.
.venv/bin/python -m tools.generate_architecture_reference --write
```

### Add a path or provider

1. Reuse the family selector: Flash-V100 routing/executors, `sm70_moe_router`,
   the linear kernel selector, or
   [`gdn_selector`](../../vllm/model_executor/layers/fla/ops/gdn_selector.py).
   Keep model qualification in its adapter and hardware/native admission in
   the provider. Dynamic M and prefill/decode decisions keep their safe boundary.
2. Declare covered stages, input/output layout and numerical semantics. MoE
   uses `STAGE_BINDINGS` / `FP4_STAGE_BINDINGS`; attention uses `ROUTE_SPECS`.
   Reuse common stages and initialization-bound policy. Explain any independent
   fusion/algorithm; default-off status alone is not deprecation evidence.
3. Add admission rejection, missing-op and fallback cases to the existing
   snapshot/test matrix. Keep historical route names. Do not infer candidate
   priority from the declaration table or claim native execution from a snapshot.
4. Compare snapshots with the parent baseline, test the affected stage sequence,
   and run affected operator tests if computation changes. Record shape, format,
   TP/EP and graph conditions, including threshold neighbors and empty input.

```bash
.venv/bin/python tools/sm70_route_snapshot.py --category moe --output /tmp/moe-before.json
# Run after the change with the same environment and baseline file:
.venv/bin/python tools/sm70_route_snapshot.py --category moe --check /tmp/moe-before.json
.venv/bin/python -m pytest -q tests/quantization/test_sm70_moe_mainflow.py \
  tests/quantization/test_sm70_fp4_mainflow.py tests/tools/test_sm70_path_explanations.py
# GDN stage changes use these contracts instead:
.venv/bin/python -m pytest -q tests/model_executor/layers/test_gdn_execution_plan.py \
  tests/kernels/test_gdn_execution_stages.py
```

### Add model qualification

1. Use [`MODELS_CONFIG_MAP`](../../vllm/model_executor/models/config.py) and
   [model runtime defaults](../../vllm/model_executor/models/runtime_defaults.py).
   Shared embedding/LM-head eligibility belongs to
   [`shared_weights.py`](../../vllm/model_executor/models/shared_weights.py).
2. Preserve the early/late checkpoints in `PolicyDefaults`; platform rules use
   the existing platform hooks. Declare the required model/state contract rather
   than adding architecture-name checks to runners, common stages or providers.
3. Test accepted and rejected model contracts, explicit overrides, and both
   engine initialization orders. Keep default provenance, weight sharing, TP
   restrictions and the existing unqualified-model fallback.

```bash
.venv/bin/python -m pytest -q tests/config/test_runtime_default_ownership.py \
  tests/v1/spec_decode/test_dflash_shared_placeholders.py
```

Qualification changes still need their affected operator/state evidence. These
CPU contracts do not establish model throughput, quality or 35B speed parity.

### Add a parameter

1. Choose the responsible field in the
   [configuration ownership table](../design/architecture/README.md#configuration-ownership).
   Extend that typed policy and its initialization adapter; do not introduce
   another global registry or execution-time environment getter.
2. Preserve parsing, empty-string behavior, alias priority, safety overrides and
   validation order. Pass resolved policy/provenance to workers. Test explicit
   typed values against legacy inputs and environment changes after resolution.
3. Include effective computation in its hash and remove migrated legacy aliases
   from duplicate environment hashing. Diagnostics and unused features must not
   perturb unrelated paths. Bind native policy explicitly when applicable.
4. If adding a legacy name, register metadata in
   [`envs.py`](../../vllm/envs.py) / [`envs_metadata.py`](../../vllm/envs_metadata.py).
   Deprecation requires category, reason, scoped evidence and replacement, plus
   a once-per-process warning when explicitly set. Reuse the existing environment
   reference generator rather than making another parameter table.

```bash
.venv/bin/python -m tools.generate_env_reference --check
.venv/bin/python -m tools.config_inventory --check
.venv/bin/python tools/pre_commit/check_env_metadata.py
.venv/bin/python -m pytest -q tests/config/test_projection_sampling_policy.py \
  tests/config/test_flash_v100_lifecycle.py
```

Choose the affected family's tests; the two examples cover projection/sampling
and attention initialization. Add cases for worker serialization, conflicting
aliases, effective hashes and independent engines when extending that policy.

### Change state or workspace

1. Identify the one mutable owner and who borrows its buffers. Reuse
   [`GDNSpecDecodeStateContract`](../../vllm/config/gdn_state.py), `ModelState`,
   [GDN state operations](../../vllm/v1/attention/ops/gdn_state.py), or the existing
   component workspace. Shared runtime resources use
   [`runtime_resources.py`](../../vllm/runtime_resources.py).
2. Keep request reorder, padding, accepted-token lengths and `-1` non-decode
   markers explicit. Preserve conv/SSM commit ordering. Input transfer owns
   pinned source lifetimes, events and exception recovery; runners sequence it.
3. Test capacity growth/release, capture/replay with changed inputs and indices,
   AOT address rebinding, and two independent engines. Compatibility facades must
   not duplicate mutable state or retain stale addresses.

```bash
.venv/bin/python -m pytest -q tests/v1/attention/test_gdn_state_ownership.py \
  tests/v1/worker/test_input_transfer.py \
  tests/quantization/test_sm70_engine_workspaces.py \
  tests/quantization/test_sm70_fp8_workspace_aot_reload.py
```

Use the affected subset. CPU ownership tests precede GPU capture/lifetime and
deterministic state tests; they cannot substitute for those tests.

### Change a native binding

1. Update the existing native source and build target together with its schema,
   fake implementation and capability probe. SM70 bindings live in
   [`vllm/_sm70/`](../../vllm/_sm70/); `_sm70_ops.py` keeps compatibility imports.
   Attention uses its existing versioned Flash-V100 policy binding.
2. Preserve old custom-op names, signatures and registration/load order. New
   explicit policy needs initialization-time rejection if an old binary cannot
   express it. Reuse parsing/projection declarations; native hot paths consume
   prebound values, not environment strings.
3. Keep library discovery in the loader and temporary/capture resources with
   their existing owner. Update stage/route declarations and tests for actual
   availability, missing operators and ABI mismatch.
4. Build the affected extension using the
   [incremental build workflow](incremental_build.md), then run its operator
   correctness/registration tests on matching hardware. Record source/binary
   identity, output/state comparison and graph lifecycle evidence.

```bash
.venv/bin/python -m pytest -q tests/config/test_sm70_native_runtime.py \
  tests/config/test_flash_v100_lifecycle.py
.venv/bin/python -m tools.sm70.path_inventory --bindings --markdown
```

Those commands validate policy contracts and inspect declared bindings; neither
builds nor exercises a native binary. Add the changed operator's GPU test.

## Task-to-tool index

Run from the repository root. Runtime tests/snapshots use the project environment;
the architecture generator and its tests need only the lightweight dependencies
installed by their pre-commit hooks, with no Torch/CUDA/native extension.

| Question | Existing command / entry | Evidence and boundary |
| --- | --- | --- |
| Did ownership or a declaration change? | `python -m tools.generate_architecture_reference --check` (default); `--write` to refresh; `--json` to inspect | Source-only; checks the explicit maintained-document list below |
| Where is a parameter read and owned? | `python -m tools.config_inventory --json`; `--check` for the gate | Static Python/native inventory and lifecycle ownership; no getter execution |
| What is declared for B/C paths? | `python -m tools.sm70.path_inventory --phase b --markdown`; use `--phase c` for C or `--bindings --markdown` for binding facts | Historical comparison accepts `--ref`; declaration is not capability proof |
| What would an existing selector choose? | `python tools/sm70_route_snapshot.py --category moe`; other categories: `awq`, `fp8`, `fp8-policy`, `nvfp4`, `dflash2` | CPU/meta test sandbox; compare with `--check FILE`; no GPU-hit claim |
| How is a selected plan explained? | [`explain_moe_plan` / `explain_linear`](../../tools/sm70/explain.py), called with an existing plan/policy | Parameters, admission and predicted operators; observed execution is a separate field |
| What policy did this engine resolve? | [`runtime_policy_report(cfg)`](../../vllm/config/policy_defaults.py) on the initialized config | Provenance/effective policy; not a second parser or CLI that builds an engine |
| Which native operations actually ran? | [`NativeDispatchTrace`](../../tools/sm70/native_trace.py); attach its JSON using snapshot `--observed-trace FILE` | Instrumented operator evidence; record the same inputs/configuration and loaded binary |
| Did layering regress? | `python tools/pre_commit/check_layering.py`; `--report` for explanations | Existing ratchet; report already includes the full parameter inventory |
| Are attention boundaries preserved? | `python -m tools.sm70.flash_v100_audit` | Dependency/source audit; no selector execution or GPU qualification |

Do not repeat the full inventory when a layering report already supplies it.
Do not widen exclusions or raise baselines to hide new coupling. The architecture
check does not run that inventory; it derives config type ownership from source.

## Documentation-only validation and CI

```bash
pre-commit run check-architecture-reference --all-files
pre-commit run test-architecture-reference --all-files
# With the project documentation dependencies installed:
API_AUTONAV_EXCLUDE=vllm mkdocs build
```

The test hook runs the focused CPU suite with `--confcutdir=tests/tools`, avoiding
inference fixtures. The check hook defaults to read-only operation. The generated
reference, overview, three component READMEs, INT8 proposal, E record, this guide,
two contributing entry points and PR template are the explicit link-check scope
in [`MAINTAINED_DOCS`](../../tools/generate_architecture_reference.py). It checks
local targets, heading anchors and source line bounds, not historical or external
URLs across the repository.

Local hooks use matching file filters. The `architecture-docs` CI workflow uses
matching path filters for declarations/configuration, maintained documents,
generator/readers, documentation hooks and check wiring. The existing all-files
lint job skips only these two new hooks because the filtered workflow owns them.
Existing layering, ownership and deprecation gates retain their constraints.

Use the [PR template](../../.github/PULL_REQUEST_TEMPLATE.md) to report path ×
format × key conditions, expected selection/fallback, evidence type and gaps.
For documentation-only changes, write `N/A` with the reason; GPU tests are not
required. For runtime changes, select the smallest affected tests, retain exact
baselines and record limitations. A static plan, an actual kernel hit and measured
performance are three different kinds of evidence.
