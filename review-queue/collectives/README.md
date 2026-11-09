# Collectives, PP overlap and P2P protocol qualification queue

This draft retains related proposals and their review records. Original patches are **unapplied** unless a separate corrected-source section explicitly identifies applied files. Do not merge this archive as accepted runtime code. CI on the archive baseline does not qualify the proposed operators, models, policies or dependencies.

## Retained sources

| Original PR | Exact head | Files | Status |
| --- | --- | --- | --- |
| [#1245](https://github.com/1CatAI/1Cat-vLLM/pull/1245) | `205ac6f550a5870b7ac06dc9286ea1a16069559e` | 24 | retained, unqualified |
| [#1216](https://github.com/1CatAI/1Cat-vLLM/pull/1216) | `01f1c7d2bf6976e4be4e4955fb95ae537be6e821` | 7 | retained, unqualified |
| [#963](https://github.com/1CatAI/1Cat-vLLM/pull/963) | `c0c25bd70ecd64f1af41ab1ac9a1cba5a567cc75` | 4 | retained, unqualified |
| [#1294](https://github.com/1CatAI/1Cat-vLLM/pull/1294) | `4f369ab020081e9fc478f21d8b6b24e579eb671b` | 2 | retained, unqualified |
| [#1250](https://github.com/1CatAI/1Cat-vLLM/pull/1250) | `c02f3edc29cf6a25b7fea10fee2de79fdbae683a` | 3 | retained, unqualified |

The publication commit retains each exact original head as a parent or ancestor, preserving complete trees and authorship. `patches/` contains original full diffs; `records/` preserves original metadata, all comments, reviews and review-thread comments at the audit snapshot. Original PR links retain subsequent discussion. No source branch is deleted.

## Audit and remaining qualification

Need actual NVLink/P2P, non-P2P two-rank and supported topology coverage, CUDA Graph replay, concurrent buffers, timeouts and slow ranks. #1250 checks an unknown communicator after allocating a pool block; repair ordering and exercise producer/restart/listener protocol before promotion. #1294 send overlap requires lifetime and shutdown validation. Do not infer all-reduce wins from a single topology.

Conflicting alternatives must be reconciled explicitly, preserving the qualified main behavior until a corrected implementation passes its own admission gates. Numeric/kernel changes require paired quality and acceptance checks as well as whole-workload measurements. Closing an old entry as superseded is **not a merge or an approval of its runtime changes**.

Prepared with Codex under the repository owner's explicit consolidation instruction.
