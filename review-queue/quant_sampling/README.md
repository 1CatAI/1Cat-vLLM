# Repair QPN2 dispatch and graph sampling state

This draft retains related proposals and their review records. Original patches are **unapplied** unless a separate corrected-source section explicitly identifies applied files. Do not merge this archive as accepted runtime code. CI on the archive baseline does not qualify the proposed operators, models, policies or dependencies.

## Retained sources

| Original PR | Exact head | Files | Status |
| --- | --- | --- | --- |
| [#1441](https://github.com/1CatAI/1Cat-vLLM/pull/1441) | `135f815612e2fdccb31689cba57f46bb57200716` | 4 | retained, unqualified |
| [#1442](https://github.com/1CatAI/1Cat-vLLM/pull/1442) | `de0cafa591a329e8452a33e72fcf7394578bac0d` | 5 | retained, unqualified |

The publication commit retains each exact original head as a parent or ancestor, preserving complete trees and authorship. `patches/` contains original full diffs; `records/` preserves original metadata, all comments, reviews and review-thread comments at the audit snapshot. Original PR links retain subsequent discussion. No source branch is deleted.

## Audit and remaining qualification

A corrected source candidate fixes the32-row native versus128-row Python dispatch mismatch by legal row chunks, binds lm_head execution policy once, and snapshots live UVA sampling parameters inside the DFlash2 graph only. Test native row thresholds, dynamic compile/CUDA Graph replay, rotating live request slots/seeds and precision/acceptance. #1261 LoRA sparse-target/mapping integration remains unapplied until its own qualification.

Conflicting alternatives must be reconciled explicitly, preserving the qualified main behavior until a corrected implementation passes its own admission gates. Numeric/kernel changes require paired quality and acceptance checks as well as whole-workload measurements. Closing an old entry as superseded is **not a merge or an approval of its runtime changes**.

Prepared with Codex under the repository owner's explicit consolidation instruction.
