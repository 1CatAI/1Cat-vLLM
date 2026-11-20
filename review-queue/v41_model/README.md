# DeepSeek-V4.1 model and speculative state queue

This draft retains related proposals and their review records. Original patches are **unapplied** unless a separate corrected-source section explicitly identifies applied files. Do not merge this archive as accepted runtime code. CI on the archive baseline does not qualify the proposed operators, models, policies or dependencies.

## Retained sources

| Original PR | Exact head | Files | Status |
| --- | --- | --- | --- |
| [#1379](https://github.com/1CatAI/1Cat-vLLM/pull/1379) | `ab911bf837779d954d0ab7cba2ab019dca39b4e8` | 62 | retained, unqualified |
| [#1375](https://github.com/1CatAI/1Cat-vLLM/pull/1375) | `2db180f0900c6096e26af03d5b873621255c8d67` | 9 | retained, unqualified |
| [#1376](https://github.com/1CatAI/1Cat-vLLM/pull/1376) | `bcfeaaffbf6b8281d011f201991450989106ee9b` | 38 | retained, unqualified |
| [#1381](https://github.com/1CatAI/1Cat-vLLM/pull/1381) | `38d9ecf8a6a242795adf9b6fd69273ce1ba9b51a` | 5 | retained, unqualified |
| [#1221](https://github.com/1CatAI/1Cat-vLLM/pull/1221) | `450e65c5262eae2853d97144e5f6fb12cde854e9` | 14 | retained, unqualified |
| [#1373](https://github.com/1CatAI/1Cat-vLLM/pull/1373) | `78f90ccb70278c3a17dc7ac309333a14a44138e7` | 18 | retained, unqualified |
| [#1370](https://github.com/1CatAI/1Cat-vLLM/pull/1370) | `49744bc71d2071fc852c1cecb5761f9d841b7dae` | 34 | retained, unqualified |
| [#1377](https://github.com/1CatAI/1Cat-vLLM/pull/1377) | `dfbc25b1af4cc2620196c17494d32bbc1015b38f` | 43 | retained, unqualified |

The publication commit retains each exact original head as a parent or ancestor, preserving complete trees and authorship. `patches/` contains original full diffs; `records/` preserves original metadata, all comments, reviews and review-thread comments at the audit snapshot. Original PR links retain subsequent discussion. No source branch is deleted.

## Audit and remaining qualification

DeepSeek-V4.1 checkpoint, DSpark drafter, stochastic coupling, stateful prefix-hit and session replay require model-level acceptance. #1376 skips model layers during prefill and cannot be admitted on speed evidence alone. #1373 persistent PP batches are kept out of the generic bugfix candidate pending PP2+speculation/session qualification. Generic step-ownership fix #1371 is tested separately.

Conflicting alternatives must be reconciled explicitly, preserving the qualified main behavior until a corrected implementation passes its own admission gates. Numeric/kernel changes require paired quality and acceptance checks as well as whole-workload measurements. Closing an old entry as superseded is **not a merge or an approval of its runtime changes**.

Prepared with Codex under the repository owner's explicit consolidation instruction.
