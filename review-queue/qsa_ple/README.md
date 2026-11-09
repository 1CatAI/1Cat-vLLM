# QSA and PLE prefill integration queue

This draft retains related proposals and their review records. Original patches are **unapplied** unless a separate corrected-source section explicitly identifies applied files. Do not merge this archive as accepted runtime code. CI on the archive baseline does not qualify the proposed operators, models, policies or dependencies.

## Retained sources

| Original PR | Exact head | Files | Status |
| --- | --- | --- | --- |
| [#828](https://github.com/1CatAI/1Cat-vLLM/pull/828) | `431232b61c01b9f619241d24cb9935111d311b90` | 3 | retained, unqualified |
| [#1176](https://github.com/1CatAI/1Cat-vLLM/pull/1176) | `f0a3f3c91e91afb1bbfcef28858d97231a901463` | 10 | retained, unqualified |
| [#1235](https://github.com/1CatAI/1Cat-vLLM/pull/1235) | `d1c15914b21f279b2984173e69cf11f9ab921c0f` | 12 | retained, unqualified |
| [#1236](https://github.com/1CatAI/1Cat-vLLM/pull/1236) | `b1a646d80b18438f037909cf3a06a90c8819cc5b` | 12 | retained, unqualified |
| [#1217](https://github.com/1CatAI/1Cat-vLLM/pull/1217) | `27a394faf934e9abf7e0ac736bad8ab821615e66` | 14 | retained, unqualified |

The publication commit retains each exact original head as a parent or ancestor, preserving complete trees and authorship. `patches/` contains original full diffs; `records/` preserves original metadata, all comments, reviews and review-thread comments at the audit snapshot. Original PR links retain subsequent discussion. No source branch is deleted.

## Audit and remaining qualification

Require Qwen4Exp QSA/PLE checkpoints, 12/24-query-head and several-KV-head cases, DCP ranks, page tails, concurrent prefetch lifetimes and full prefill/decode comparison. The exact #828 merge-base/head trees differ in only3files; its API file list of262 includes unchanged entries. Its regenerated diff is replay-verified. No inference gain is inferred from that reconstruction.

Conflicting alternatives must be reconciled explicitly, preserving the qualified main behavior until a corrected implementation passes its own admission gates. Numeric/kernel changes require paired quality and acceptance checks as well as whole-workload measurements. Closing an old entry as superseded is **not a merge or an approval of its runtime changes**.

Prepared with Codex under the repository owner's explicit consolidation instruction.
