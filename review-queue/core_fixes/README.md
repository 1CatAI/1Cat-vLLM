# Repair scheduler, cache, loader and pipeline ownership

This draft retains related proposals and their review records. Original patches are **unapplied** unless a separate corrected-source section explicitly identifies applied files. Do not merge this archive as accepted runtime code. CI on the archive baseline does not qualify the proposed operators, models, policies or dependencies.

## Retained sources

| Original PR | Exact head | Files | Status |
| --- | --- | --- | --- |
| [#1408](https://github.com/1CatAI/1Cat-vLLM/pull/1408) | `3affd3e9df3a5221d4a5416822780e9e1162c786` | 17 | retained, unqualified |
| [#1371](https://github.com/1CatAI/1Cat-vLLM/pull/1371) | `0bd5c7954cc585fa30bd41fbe2c6483e79057c00` | 9 | retained, unqualified |
| [#1276](https://github.com/1CatAI/1Cat-vLLM/pull/1276) | `bd3b78bcd4b11981dd45fbb968e9f55ffd5833e3` | 14 | retained, unqualified |
| [#1225](https://github.com/1CatAI/1Cat-vLLM/pull/1225) | `f9d1134a6413c49a9d4ab1890ee42ef3b9add80a` | 5 | retained, unqualified |
| [#1362](https://github.com/1CatAI/1Cat-vLLM/pull/1362) | `924bd588cdba31ee878512334afb903671a40c94` | 4 | retained, unqualified |
| [#1319](https://github.com/1CatAI/1Cat-vLLM/pull/1319) | `8b7fb496a4e812f38a2ed7235a56937a92ceb662` | 5 | retained, unqualified |
| [#1320](https://github.com/1CatAI/1Cat-vLLM/pull/1320) | `9efdccd897c4a2192b42176c10928b117dbff659` | 8 | retained, unqualified |
| [#1300](https://github.com/1CatAI/1Cat-vLLM/pull/1300) | `7b3ffb8711168c6cf855c5329e6469bd169b827f` | 5 | retained, unqualified |
| [#1161](https://github.com/1CatAI/1Cat-vLLM/pull/1161) | `a66fcda49f1133b54dbbe8d14daf38e2a6731da4` | 2 | retained, unqualified |

The publication commit retains each exact original head as a parent or ancestor, preserving complete trees and authorship. `patches/` contains original full diffs; `records/` preserves original metadata, all comments, reviews and review-thread comments at the audit snapshot. Original PR links retain subsequent discussion. No source branch is deleted.

## Audit and remaining qualification

A corrected source candidate is being tested on54633 for broadcast timeouts, KV lookahead hashes, PP step ownership, prompt-logprobs memory, loader/direct-I/O and provider lifecycle. #1373 and #1250 are excluded and tracked in their feature/protocol groups. Generated architecture references and main CI must pass; prefix-cache and PP model-level checks remain admission gates where needed.

Conflicting alternatives must be reconciled explicitly, preserving the qualified main behavior until a corrected implementation passes its own admission gates. Numeric/kernel changes require paired quality and acceptance checks as well as whole-workload measurements. Closing an old entry as superseded is **not a merge or an approval of its runtime changes**.

Prepared with Codex under the repository owner's explicit consolidation instruction.
