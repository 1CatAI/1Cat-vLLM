# Repair tool parsing, DeepSeek rendering and usage reporting

This draft retains related proposals and their review records. Original patches are **unapplied** unless a separate corrected-source section explicitly identifies applied files. Do not merge this archive as accepted runtime code. CI on the archive baseline does not qualify the proposed operators, models, policies or dependencies.

## Retained sources

| Original PR | Exact head | Files | Status |
| --- | --- | --- | --- |
| [#1414](https://github.com/1CatAI/1Cat-vLLM/pull/1414) | `4d5711ed3e1fbc7b7801c2346c41aa70d01273af` | 11 | retained, unqualified |
| [#1413](https://github.com/1CatAI/1Cat-vLLM/pull/1413) | `37376ba12e227ef582737e526ef5160408e96904` | 7 | retained, unqualified |

The publication commit retains each exact original head as a parent or ancestor, preserving complete trees and authorship. `patches/` contains original full diffs; `records/` preserves original metadata, all comments, reviews and review-thread comments at the audit snapshot. Original PR links retain subsequent discussion. No source branch is deleted.

## Audit and remaining qualification

A corrected source candidate includes iterative empty-block handling, exact whole-number parsing and DeepSeek-V4 dialect boundaries, with both original test sets retained. Run real parser/tokenizer regression tests and the serving cache-usage paths on54633. CPU parser success is scoped evidence, not a DeepSeek model quality benchmark.

Conflicting alternatives must be reconciled explicitly, preserving the qualified main behavior until a corrected implementation passes its own admission gates. Numeric/kernel changes require paired quality and acceptance checks as well as whole-workload measurements. Closing an old entry as superseded is **not a merge or an approval of its runtime changes**.

Prepared with Codex under the repository owner's explicit consolidation instruction.
