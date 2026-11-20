# Multimodal and MRV2 pipeline qualification queue

This draft retains related proposals and their review records. Original patches are **unapplied** unless a separate corrected-source section explicitly identifies applied files. Do not merge this archive as accepted runtime code. CI on the archive baseline does not qualify the proposed operators, models, policies or dependencies.

## Retained sources

| Original PR | Exact head | Files | Status |
| --- | --- | --- | --- |
| [#1415](https://github.com/1CatAI/1Cat-vLLM/pull/1415) | `a43f9b0448fcc24b99c09b5f985265daa155259d` | 6 | retained, unqualified |
| [#1365](https://github.com/1CatAI/1Cat-vLLM/pull/1365) | `af22bbe21c1b0f6d205193519922c327230cdee6` | 10 | retained, unqualified |

The publication commit retains each exact original head as a parent or ancestor, preserving complete trees and authorship. `patches/` contains original full diffs; `records/` preserves original metadata, all comments, reviews and review-thread comments at the audit snapshot. Original PR links retain subsequent discussion. No source branch is deleted.

## Audit and remaining qualification

Image hashing broken-EXIF cases can be unit-tested, but GLM vision position/image budgets and MRV2 pipeline transfer need the relevant vision checkpoints plus PP layout. Preserve all source and qualify multimodal positions, cache identity and prompt/output behavior before integration.

Conflicting alternatives must be reconciled explicitly, preserving the qualified main behavior until a corrected implementation passes its own admission gates. Numeric/kernel changes require paired quality and acceptance checks as well as whole-workload measurements. Closing an old entry as superseded is **not a merge or an approval of its runtime changes**.

Prepared with Codex under the repository owner's explicit consolidation instruction.
