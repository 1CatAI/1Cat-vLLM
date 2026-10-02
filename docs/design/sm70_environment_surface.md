# SM70 environment control surface

SM70 public controls are explicit metadata entries with `user_visible=True`.
They cover the acceleration off switches, the separate online-requantization
experiment, memory limits and one debug entry. The generated user reference
omits internal schedules, tuning and compatibility aliases. Classification
changes do not change routing or remove those implementation controls.

Use `KernelConfig`, `VLLM_DISABLED_KERNELS` and the attention backend option
for migrated kernels. Internal controls remain inspectable for maintenance:

```bash
.venv/bin/python -m tools.generate_env_reference \
  --include-internal --output /tmp/environment-internals.md
```

CI requires literal visibility metadata for every variable, rejects consumer
placeholder text in public SM70 descriptions and checks generated documentation.
The registration check now covers stable module-constant aliases in Python,
including long attention's previously missed opt-out. It respects local
shadowing; dynamically constructed names still require a separate audit.

## Debug compatibility

`VLLM_SM70_DEBUG` accepts comma-separated `trace`, `mtp`, `events` and `routing`.
It defaults to empty, so observation adds no work. Set only the channels needed
for an investigation; timing with observers is not ordinary throughput.
An explicitly set unified entry, including an empty value, takes precedence
over the migrated aliases. Without it, legacy parsing and OR precedence remain
unchanged and use of an alias produces a deprecation warning.

| Old variable | Replacement |
| --- | --- |
| `VLLM_SM70_PROFILE_TRACE` | `VLLM_SM70_DEBUG=trace` |
| `VLLM_SM70_DECODE_TILE_PROFILE` | `VLLM_SM70_DEBUG=trace` |
| `VLLM_SM70_MTP_PROFILE` | `VLLM_SM70_DEBUG=mtp` |
| `VLLM_SM70_DECODE_EVENT_TRACE` | `VLLM_SM70_DEBUG=events` |
| `VLLM_FLASH_V100_ROUTE_SUMMARY` | `VLLM_SM70_DEBUG=routing` |
| `VLLM_FLASH_V100_DEBUG_ROUTE_SUMMARY` | `VLLM_SM70_DEBUG=routing` |

These aliases remain for one full released compatibility version and are
removed in the following version. The cycle starts with the first release
containing this migration; the ongoing release/1.5.1 branch is untouched. This first debug batch consolidates these observers; tensor
capture directories and the remaining specialized observers still need their
category migrations. Hiding their catalog entries does not consolidate them.

## QSA tuning compatibility

`SM70_QSA_TUNING` contains the existing measured defaults: 64-MiB score tiles,
512-row cuBLAS crossover, 1048576 score elements and 64-row page4 XQA crossover.
The conservative XQA threshold follows the measured approximately 48-row
crossover, independent of the server token budget. No kernel arithmetic,
threshold, warmup route or tensor output changes in this migration.

The three former `VLLM_SM70_QSA_INDEXER_*` numeric overrides and
`VLLM_SM70_QSA_XQA_PAGE4_MIN_ROWS` retain their original parser and explicit
values for one released compatibility cycle, with warnings. Default tuning lives in
code; these variables are omitted from the public reference. They are scheduled
for removal in the following release, including the old benchmark's explicit 4096-row XQA
setting. Historical benchmark recipes remain replayable during this cycle.

## Removed inactive entries

The source/native audit finds no execution consumer for these six entries:

- `VLLM_DFLASH_SYNC_CONTEXT_KV`
- `VLLM_DFLASH_SKIP_CONTEXT_KV_PRECOMPUTE`
- `VLLM_DFLASH_DUMP_LAYER_HIDDENS`
- `VLLM_DFLASH_DUMP_LAYER0_COMPONENTS`
- `VLLM_DFLASH_DUMP_ATTN_COMPONENTS`
- `VLLM_SM70_FLASH_V100_0DOT3_BENCHMARK_COMBO_KERNEL`

The last alias survived only in benchmark reporting. Configuration already
forces `benchmark_combo_kernel=True` because the unbenchmarked choice changed
greedy output. Removing the alias does not admit that rejected schedule.
Historical reports mentioning it remain unchanged. The five DFlash entries had
no consumer beyond their getter/declaration; removing them changes no execution
path and enables no replacement experiment.

Registering long attention's stable opt-out and removing inactive entries
intentionally changes the environment portion of compile-cache fingerprints.
None of these changes changes a native dispatcher, mathematical operation or
precision setting. Pure cleanup uses route snapshots and exact operator checks;
GPU speed A/B is reserved for admission/default changes.
