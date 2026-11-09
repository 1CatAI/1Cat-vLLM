# Heterogeneous seven-GPU serving qualification

This draft retains related proposals and their review records. Original patches are **unapplied** unless a separate corrected-source section explicitly identifies applied files. Do not merge this archive as accepted runtime code. CI on the archive baseline does not qualify the proposed operators, models, policies or dependencies.

## Retained sources

| Original PR | Exact head | Files | Status |
| --- | --- | --- | --- |
| [#1260](https://github.com/1CatAI/1Cat-vLLM/pull/1260) | `53c88a5ba55bbd9f5b148226801881bd871bec40` | 215 | retained, unqualified |

The publication commit retains each exact original head as a parent or ancestor, preserving complete trees and authorship. `patches/` contains original full diffs; `records/` preserves original metadata, all comments, reviews and review-thread comments at the audit snapshot. Original PR links retain subsequent discussion. No source branch is deleted.

## Audit and remaining qualification

Original #1260 requires seven heterogeneous GPUs and matching GGUF checkpoints.54633 has4xV100. Its complete215-file merge-base/head diff was reconstructed and replay-verified. Need the intended heterogeneous PP/TP mapping, transport failure/slow-rank, memory and model-quality acceptance before integration.

Conflicting alternatives must be reconciled explicitly, preserving the qualified main behavior until a corrected implementation passes its own admission gates. Numeric/kernel changes require paired quality and acceptance checks as well as whole-workload measurements. Closing an old entry as superseded is **not a merge or an approval of its runtime changes**.

Prepared with Codex under the repository owner's explicit consolidation instruction.
