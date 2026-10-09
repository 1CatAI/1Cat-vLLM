# Phase B: SM70 parameters and execution paths

Integration base: `fc9a518654d8e2dc5dce32507aaece474930ea17`.
The acceptance scope is structural consolidation with unchanged defaults,
operator selection, numerical arithmetic and model qualification. On
2026-10-09 the project owner explicitly accepted operator tests for this round;
35B AWQ/FP8 model throughput and TTFT are not claimed here.

## B0: inventory and ownership

The [generated parameter and native-path ledger](sm70_phase_b_parameters.md)
freezes the pre-change tree: 26 source files, 93 legacy parameters, 155 reads,
and 247 native call sites. It includes all four quantized MoE methods, GGUF,
skinny, quantization loaders, QPN, linear providers, workspace registration and
native bindings. Each row links to the exact baseline source and records the
getter/default, consumer and enclosing route conditions. JSON output also
retains native arguments and function sizes:

```bash
python tools/sm70/path_inventory.py --ref fc9a518654d8e2dc5dce32507aaece474930ea17
python tools/sm70/path_inventory.py --summary
```

Counts are static evidence: aliases moved into a configuration adapter must not
be interpreted as deleted parameters, and a call site is not an independently
reachable end-to-end path. The migration disposition below complements the
source ledger; remaining work is explicit.

| Family / entry | Preparation and eligibility | Execution and fallback | Resource / disposition |
|---|---|---|---|
| AWQ `AWQSM70MoEMethod` | 4-bit asymmetric g32/64/128; align logical input/output; per-expert TurboMind banks, optional compact scale/zero metadata | Single active/compact/indexed; per-expert grouped or dense W13/W2; active-exact W2 limited to 128 slots then dense; batched token ceiling preserved | Source weights released after preparation; persistent decode capacity bounded at 32; dummy/capture and overflow temporary buffers retained. Common codec and stage executor in delivery 1 |
| AWQ Qwen3.8 adapter | Exact TP4/E512/H2560/I160/top-k10/g32 and original native checks; indexed W13 prefill starts at M128 | M2..8 active grouped; indexed prefill; W2 chunks 4096/6144 only with indexed W13 and actual memory saving; explicit unavailable indexed/compact layouts still fail closed | No qualification broadening. Chunked W2 owns W2 plus reduction; indexed W13 avoids materialized input. Adapter stays distinct; stage order is shared |
| AWQ legacy compact / QPN M1 | Original interleaved W13 prepared bank and QPN qualification | Original monolithic call; QPN shortcut has priority within the admitted legacy lane | Experimental/native differences retained; default-off is not deprecation. QPN policy normalization in delivery 2; monolithic arithmetic retained |
| FP8 `Fp8SM70MoEMethod` | Existing block/tensor scale normalization and FP8-to-TurboMind conversion; static input-scale handling unchanged | Single active/indexed/compact and batched/per-expert/dense W13/W2; absent requested single-token op falls back in the old order | Same prepared FP8 banks and metadata; 32-token resident workspace, unchanged overflow allocation/aliases. Common codec/stages and independent workspace owner in delivery 1 |
| FP8 legacy compact | Original separate interleaved banks, exact-layout/decomposed/native-unpermute options | Explicit experiment remains accessible; compare diagnostics remain observational; strict-compare-fail retains its historical no-op warning | Distinct layout and arithmetic retained, not promoted to a default; decomposed execution joins the common stages in delivery 2 |
| NVFP4 `ModelOptNvFp4SM70MoEMethod` | Original model, shape, scale, TP and native gates in linked ledger | QPN M1/batch/MTP5; dense/grouped; indexed/fused-SwiGLU prefill; direct W2 reduction and GLM Q8 variants | Raw/prescaled and prepared banks, split-K and graph-safe buffers are separate contracts. Delivery 2 consolidates these choices without changing eligibility |
| MXFP4 `Mxfp4SM70MoEMethod` | Existing E8M0 scale/layout and model geometry checks | QPN/tiny-batch/grouped and dense fallback; exact flags and native constraints in ledger | Preserve scale convention and supported geometry. Delivery 2; independent geometry broadening PR #1030 is not included |
| GGUF TurboMind MoE | Mixed per-projection weight formats, expert-map handling, separate gate/up | Existing mixed-format calls and FP32 weighted reduction | Keep split gate/up and arithmetic. Share contracts/reusable stages in delivery 2 |
| skinny modular MoE | Existing modular interface, grouping/sorting, scale modes and QPN kernels | Shape-driven chunks and split-K; original modular backend admission | Keep its modular ownership, share reusable contracts/stages in delivery 2 |
| AWQ/FP8/NVFP4/GGUF linear providers | Existing common kernel selector and ordered candidates; per-format typed linear config | Existing provider preparation and `op_kind` branches, logical output clipping and bias | Reuse existing AOT runtime workspace address resolution. Delivery 3 consolidates provider dispatch and QPN ownership |
| Native `_sm70_ops.py` | Existing loading, registration and fake implementations | Stable public names and lazy native checks | Delivery 3 splits loading/linear/MoE/auxiliary ownership without changing import order |

Weight packing, group size, scale/zero point and preparation layout belong to
the codec contract. Device, dtype, dimensions and native availability belong to
capability checks. Token ceilings, stage modes and split-K/chunks belong to
execution policy. Workspace capacity and persistent/capture allocations belong
to workspace owners. Model qualification stays in adapters. Diagnostics and
legacy aliases do not belong in numerical executors.

## Delivery 1: AWQ and FP8 shared stages

Both formats now use `Sm70MoEMethodBase` for policy/lifecycle adaptation,
`Sm70MoEWeightCodec.prepare_weights` for alternating W13/W2 expert preparation,
`execute_single_token` and `execute_routed` for the two actual execution flows.
The codec receives tensors and layout arguments, not a complete method object.
There is no callback from either executor into the old format modules.

`Sm70MoeRoutePlan` is extended rather than replaced by a parallel dispatcher.
`STAGE_BINDINGS` is the common source for native binding names and snapshot
explanations. Native fusion/layout/numerical differences stay explicit:

- Single-token W13 produces routing metadata and compact input as well as W13.
- Indexed prefill reads original input through row indices.
- AWQ grouped-active W2 uses active expert offsets; full dense W2 uses all experts.
- AWQ chunked W2 includes weighted reduction and returns directly.
- AWQ's interleaved activation and logical output slice remain in their original positions.
- FP8 does not acquire AWQ's output slice or change its scale preparation.

The ordinary decomposed format flows reduce from four independently maintained
flows (two formats × single/routed) to two shared flows. Monolithic legacy and
reference-comparison algorithms are counted separately, not hidden as common
flow reductions. AWQ's reference calculations retain their original stage
positions in a diagnostic observer; disabled diagnostics create no observer.
FP8's resident/overflow workspace extraction reuses the verified #1065 code and
behavioral tests. Native pointer registration and AOT reload stay unchanged.

### Configuration and compatibility

`KernelConfig.sm70_moe.awq` and `.fp8` capture policy once for the engine.
Explicit typed fields override legacy aliases. Unsupported per-format typed
fields fail rather than silently becoming ineffective switches. Legacy getters
still control parsing, default values and malformed-value behavior.

`single_token_w13` is a candidate set normalized to compact → indexed → dense.
Compact missing does not erase a requested indexed fallback. Strict AWQ mode
still disables indexed stages while retaining the historical compact-W13
precedence. Combined indexed flags use the original OR rules; FP8 also retains
its format-specific indexed-W2 default. Weighted reduction falls back to native
unpermute when its op is absent. Explicit compact metadata and indexed prefill
retain their old missing-op error behavior.

Only resolved format policies enter the graph hash. Diagnostics and source
attribution are excluded. Two engines receive separate policies and no process
environment mutation is used. Static single-token availability/plans are bound
at initialization; dynamic M remains in the existing `apply` boundary.

Example explicit equivalent of the dense single-token stage request:

```json
{"sm70_moe":{"awq":{"single_token_w13":["dense"],"single_token_w2":"dense"}}}
```

This requests stage candidates; it does not override legacy monolithic admission
or make an ineligible model/shape eligible. Set `legacy_compact: false` when
explicitly selecting the decomposed path.

### Explanation and validation

The existing snapshot command accepts `--category moe`. It records configuration,
selected stages, predicted native bindings and fallback reason, and compares
against old helper implementations loaded from a git revision:

```bash
python tools/sm70_route_snapshot.py --category moe \
  --baseline-ref fc9a518654d8e2dc5dce32507aaece474930ea17 \
  --output /tmp/sm70-moe-routes.json
```

This CPU prediction is explicitly separate from GPU native-call observations.
The operator A/B harness records actual public native calls, bit-exact outputs,
prepared banks, eager/graph replay and alternating graph-event timings.
The final validation record is maintained in the migration control document.

## Remaining delivery boundaries

Delivery 2 implements the FP4 Python stage migration and GGUF/skinny reusable
reduction. Delivery 3 owns linear dispatch/QPN/native module ownership, the
remaining native policy consumers and combined explanation reports. The old #1065/#1066 drafts are superseded by the
first main-targeted delivery; they are not dependencies or stacked merge gates.
DDTree and repository-wide C/D refactors remain outside this scope.

## Delivery 2: FP4 stages, workspace ownership and retained interfaces

Integration base: `fe63db651e3cc4055bf4349411d372ad801714aa` (delivery 1,
PR #1124, merged with remote CI passing). NVFP4's direct, routed and grouped
flows and MXFP4's QPN, direct-order, direct-prepare and ordinary routed flows
now use one `execute_fp4` stage sequence: seven independent production sequences
become one. `Fp4MoECodec` binds weight preparation, W13 and W2 without reading
policy or calling the old format modules. The existing `Sm70MoeRoutePlan`
expresses fusion coverage, split-K, interleaving, MTP binding and reduction.

The [generated bindings and alias table](sm70_phase_b_bindings.md) comes from
the same declarations consumed by the codec. Fused-stage coverage also controls
whether the shared executor performs activation and reduction. Distinct kernels
retain their actual differences: raw E4M3/global scales, E8M0 packing, split-K,
interleaved SwiGLU, clamp arithmetic, grouped route metadata, split N256/N64
prefill, GLM's exact reduction tree and ordered Triton/native reductions.

`NvFp4MoEWorkspace` and `MxFp4MoEWorkspace` own allocation/view/overflow rules.
Their original layer attributes remain the storage owners; the codec borrows
views, so rebinding remains visible. Raw-scale storage still uses the original
per-device shared expansion workspace and rejects microbatching before allocation.
MXFP4 retains its separate immutable direct-order offsets after M2..8 compaction.
No new persistent weight copy or global pointer registry is introduced.

`KernelConfig.sm70_moe.nvfp4` and `.mxfp4` capture 27 legacy inputs once, and
AWQ's remaining QPN M1 input now joins `.awq`. Loaded layers pass captured policy
to the existing shape selectors. Direct legacy helper callers may still resolve
an independent compatibility policy. The common execution modules have no
configuration/environment reads. The Qwen route debug switch is captured once
and excluded from the **KernelConfig** calculation hash.

FP8's explicit legacy decomposed experiment reuses `execute_routed`, including
its original zero-before-unpermute ordering. The monolithic AWQ/FP8 experiments
remain separate because of their native fusion/layout/reduction semantics;
they are not deprecated merely because a flag defaults off. GGUF's mixed
projection banks and separate gate/up, and skinny's modular interface, remain
intact. Their matching FP32 slot-major weighted reduction now shares one stage;
skinny's FP16 per-expert index-add reference is deliberately separate.

### Native policy boundary still owned by delivery 3

The native audit found dual consumers in TurboMind's GEMM selector/feasibility
checks and MoE permutation kernels. Migrating a Python getter alone cannot make
a conflicting typed override control those native decisions. Until explicit
native policy arguments land, dual-consumer boolean overrides must match the
legacy native value; conflicts fail with a clear error and never modify the
process environment. This compatibility guard is temporary, not a completed
implementation of typed precedence across the native boundary.

Affected groups include AWQ active-exact/grouped W2, MXFP4 grouped/verifier
selection, NVFP4 grouped prefill/expert rows/fast prefill and common single-token
permutation switches. C++-only tuning switches also remain for the native-binding
inventory in delivery 3. The overall AOT environment hash still contains these
legacy consumers; unused-format/diagnostic isolation currently applies to the
KernelConfig hash, not yet the entire AOT cache key. Delivery 3 must complete
that boundary, the FP8 diagnostic adapter and the combined execution explanation,
in addition to the linear/provider/native module work.

### Validation and measurable changes

- Frozen-main native call/argument/view traces cover 213 supported FP4 cases;
  configuration, workspace, rebinding, conflict and reduction checks bring the
  focused suite to 223 passes. Ordinary AWQ/FP8 plus AWQ QPN compatibility remains
  covered separately; 168 existing route snapshots are unchanged.
- Final NVFP4 Qwen operator A/B: 84 eager cases and 180 changed-input/route graph
  replays, exact output/native order. MXFP4: 20 eager cases and 51 replays, exact.
  Separate weight loads match packed banks; native pointer descriptors compare
  address offsets and strides, excluding their uninitialized padding bytes.
- The final same-GPU graph timing median change is -0.158% for NVFP4 and -0.033%
  for MXFP4; maximum increases are +1.55% and +0.625%. No model speed claim follows.
- GLM TP8 fused permutation/QPN and the FP8 legacy decomposed route have separate
  observed-hit A/B records; grouped/MTP records exercise the newly shared stage
  sequence. Full counts and artifact locations are in the migration control log.
- Static source audit after delivery 1: 95 parameter reads / 217 direct native
  call sites. After this extraction: 48 / 179; alias declarations and
  indirect codec calls are counted separately. These counts do **not** mean
  parameters or supported native paths were removed. The meaningful reduction
  is seven FP4 execution sequences to one, plus removal of FP8's duplicated
  decomposed production sequence.

Operator testing is the owner's accepted gate. No new model geometry, TP/EP
qualification, attention configuration, 35B throughput or TTFT claim is made.
