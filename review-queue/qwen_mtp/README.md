# Qwen MTP, draft heads and GDN integration queue

This draft retains related proposals and their review records. Original patches are **unapplied** unless a separate corrected-source section explicitly identifies applied files. Do not merge this archive as accepted runtime code. CI on the archive baseline does not qualify the proposed operators, models, policies or dependencies.

## Retained sources

| Original PR | Exact head | Files | Status |
| --- | --- | --- | --- |
| [#1427](https://github.com/1CatAI/1Cat-vLLM/pull/1427) | `431a20ad8751d4c8e3e026c9c27fa5941e257612` | 20 | retained, unqualified |
| [#1218](https://github.com/1CatAI/1Cat-vLLM/pull/1218) | `7308b33f9152762ee3a12928516d673411b92818` | 7 | retained, unqualified |
| [#979](https://github.com/1CatAI/1Cat-vLLM/pull/979) | `0a0dbe99f7536290bde1094ce0ecf9a720f13988` | 8 | retained, unqualified |
| [#953](https://github.com/1CatAI/1Cat-vLLM/pull/953) | `5a7c65327c5e5166155e6af18c8dddaaf45d7f03` | 32 | retained, unqualified |
| [#983](https://github.com/1CatAI/1Cat-vLLM/pull/983) | `6dd4c56ba4482813034370a6f7f288673de435c6` | 12 | retained, unqualified |
| [#1301](https://github.com/1CatAI/1Cat-vLLM/pull/1301) | `d8ed559e21705566f983c58dc5de0f361b63f3ac` | 4 | retained, unqualified |
| [#1345](https://github.com/1CatAI/1Cat-vLLM/pull/1345) | `c6ae9c5d7b2ebbb33d796e33083a7e36730aec6e` | 5 | retained, unqualified |
| [#844](https://github.com/1CatAI/1Cat-vLLM/pull/844) | `35b84b870571dcc030096a1ce3a1da6b27de599f` | 3 | retained, unqualified |
| [#629](https://github.com/1CatAI/1Cat-vLLM/pull/629) | `27e5f8e574ba7f46e755e66f94796d3438ea8867` | 6 | retained, unqualified |
| [#1261](https://github.com/1CatAI/1Cat-vLLM/pull/1261) | `609b31a1b8dc74dcd3bdd0e56db79aa320540ee8` | 3 | retained, unqualified |

The publication commit retains each exact original head as a parent or ancestor, preserving complete trees and authorship. `patches/` contains original full diffs; `records/` preserves original metadata, all comments, reviews and review-thread comments at the audit snapshot. Original PR links retain subsequent discussion. No source branch is deleted.

## Audit and remaining qualification

Validate combined draft vocabulary/head, GDN state publication and attention changes with the same weights, seed, sampling policy, C1/C8, 2K/long-context inputs and tuning settings. Component gains cannot establish <=10ms or unchanged acceptance. The historical14.8ms and ninfer11.5ms figures are not new measurements. Keep reduced-vocabulary and numerical changes unpromoted until paired output/acceptance and quality gates pass.

Conflicting alternatives must be reconciled explicitly, preserving the qualified main behavior until a corrected implementation passes its own admission gates. Numeric/kernel changes require paired quality and acceptance checks as well as whole-workload measurements. Closing an old entry as superseded is **not a merge or an approval of its runtime changes**.

Prepared with Codex under the repository owner's explicit consolidation instruction.
