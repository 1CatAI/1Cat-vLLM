# Serving API and alternative parser qualification queue

This draft retains related proposals and their review records. Original patches are **unapplied** unless a separate corrected-source section explicitly identifies applied files. Do not merge this archive as accepted runtime code. CI on the archive baseline does not qualify the proposed operators, models, policies or dependencies.

## Retained sources

| Original PR | Exact head | Files | Status |
| --- | --- | --- | --- |
| [#1437](https://github.com/1CatAI/1Cat-vLLM/pull/1437) | `065705ab71474c9b7dae539f28e31a3dc084d97d` | 4 | retained, unqualified |
| [#1438](https://github.com/1CatAI/1Cat-vLLM/pull/1438) | `50dd333b8c0b8a026b8a16416c5a67b2361ce870` | 5 | retained, unqualified |
| [#1439](https://github.com/1CatAI/1Cat-vLLM/pull/1439) | `6f16090081012e3a8aea4e1d5a4ac08fe2f3bd1f` | 5 | retained, unqualified |
| [#1440](https://github.com/1CatAI/1Cat-vLLM/pull/1440) | `ec3db4a8923aa1e559ca341f9e61afc32e4d4437` | 4 | retained, unqualified |
| [#1443](https://github.com/1CatAI/1Cat-vLLM/pull/1443) | `ee1216fc8a67eeadcc4b27664ed07d16b8ecd5c6` | 2 | retained, unqualified |
| [#1446](https://github.com/1CatAI/1Cat-vLLM/pull/1446) | `17411f25a40b2b6927683475e4eb76ba62f47452` | 6 | retained, unqualified |
| [#241](https://github.com/1CatAI/1Cat-vLLM/pull/241) | `463406253a7cb57d320ef652288fb48fc2084b87` | 33 | retained, unqualified |

The publication commit retains each exact original head as a parent or ancestor, preserving complete trees and authorship. `patches/` contains original full diffs; `records/` preserves original metadata, all comments, reviews and review-thread comments at the audit snapshot. Original PR links retain subsequent discussion. No source branch is deleted.

## Audit and remaining qualification

Keep service/reload/parser defaults unchanged pending API integration tests. #1446 supplies a large alternative parser with no test files in its diff and flips the default switch; do not promote it unqualified. #1439 hot reload and #1438 cache reclamation need concurrent-request and failure-path checks. #241 is a broad older replay, not a ready-to-merge patch stack.

Conflicting alternatives must be reconciled explicitly, preserving the qualified main behavior until a corrected implementation passes its own admission gates. Numeric/kernel changes require paired quality and acceptance checks as well as whole-workload measurements. Closing an old entry as superseded is **not a merge or an approval of its runtime changes**.

Prepared with Codex under the repository owner's explicit consolidation instruction.
