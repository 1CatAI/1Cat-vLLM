# DeepSeek-V4.1 8-GPU checkpoint qualification

This draft retains related proposals and their review records. Original patches are **unapplied** unless a separate corrected-source section explicitly identifies applied files. Do not merge this archive as accepted runtime code. CI on the archive baseline does not qualify the proposed operators, models, policies or dependencies.

## Retained sources

| Original PR | Exact head | Files | Status |
| --- | --- | --- | --- |
| [#1243](https://github.com/1CatAI/1Cat-vLLM/pull/1243) | `70bb718828f3c0620c0fb44ec5f671dd67194e47` | 124 | retained, unqualified |
| [#1030](https://github.com/1CatAI/1Cat-vLLM/pull/1030) | `19f25c48ee95eeda0ba402cdea8e29f963a5fce0` | 4 | retained, unqualified |
| [#1031](https://github.com/1CatAI/1Cat-vLLM/pull/1031) | `a36df7c2239a13ae8da3f257151bdf92f335cc5f` | 106 | retained, unqualified |

The publication commit retains each exact original head as a parent or ancestor, preserving complete trees and authorship. `patches/` contains original full diffs; `records/` preserves original metadata, all comments, reviews and review-thread comments at the audit snapshot. Original PR links retain subsequent discussion. No source branch is deleted.

## Audit and remaining qualification

Original #1243 requires an 8xV100 DeepSeek-V4.1 EXL3 topology and its checkpoint;54633 has4xV100. The complete124-file diff was reconstructed from exact Git trees and replay-verified, including file modes. Original source commits #1030/#1031 remain ancestors. This establishes source retention, not 1M-context/model acceptance.

Conflicting alternatives must be reconciled explicitly, preserving the qualified main behavior until a corrected implementation passes its own admission gates. Numeric/kernel changes require paired quality and acceptance checks as well as whole-workload measurements. Closing an old entry as superseded is **not a merge or an approval of its runtime changes**.

Prepared with Codex under the repository owner's explicit consolidation instruction.
