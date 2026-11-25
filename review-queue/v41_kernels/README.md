# DeepSeek-V4.1 native kernel qualification queue

This draft retains related proposals and their review records. Original patches are **unapplied** unless a separate corrected-source section explicitly identifies applied files. Do not merge this archive as accepted runtime code. CI on the archive baseline does not qualify the proposed operators, models, policies or dependencies.

## Retained sources

| Original PR | Exact head | Files | Status |
| --- | --- | --- | --- |
| [#1382](https://github.com/1CatAI/1Cat-vLLM/pull/1382) | `d488114cd1753a411f26248945ebeb822fe891ae` | 6 | retained, unqualified |
| [#1385](https://github.com/1CatAI/1Cat-vLLM/pull/1385) | `2fe80a0f046cce68fcb3d2ca43903b8708216428` | 17 | retained, unqualified |
| [#1386](https://github.com/1CatAI/1Cat-vLLM/pull/1386) | `7b1b689f528382ee20041d5fc27a4f05d61060dc` | 6 | retained, unqualified |
| [#1387](https://github.com/1CatAI/1Cat-vLLM/pull/1387) | `1872a3882a71b7089597c826cedbe6b54fc86175` | 6 | retained, unqualified |
| [#1388](https://github.com/1CatAI/1Cat-vLLM/pull/1388) | `90eccca2679040fd6729f2ecc720caf31819f1b5` | 39 | retained, unqualified |
| [#1389](https://github.com/1CatAI/1Cat-vLLM/pull/1389) | `04b5f9d4f41672caafdd4b8c7fc09789561f0c3c` | 40 | retained, unqualified |
| [#1390](https://github.com/1CatAI/1Cat-vLLM/pull/1390) | `c9a380b7fae39728b395dac945bec65f7c513676` | 7 | retained, unqualified |
| [#1372](https://github.com/1CatAI/1Cat-vLLM/pull/1372) | `e3acc82f3a14c2d890a8a47fed05a24346df9169` | 15 | retained, unqualified |
| [#1374](https://github.com/1CatAI/1Cat-vLLM/pull/1374) | `a19f07f72bb381738785ca1938963267d1dc4181` | 11 | retained, unqualified |
| [#1273](https://github.com/1CatAI/1Cat-vLLM/pull/1273) | `9a71c5c6271e4b8db665f407511ecd9a9a410457` | 12 | retained, unqualified |
| [#1383](https://github.com/1CatAI/1Cat-vLLM/pull/1383) | `45021a037847d42e4ff3c3c3a3df2ab0f3c19cd5` | 5 | retained, unqualified |
| [#1384](https://github.com/1CatAI/1Cat-vLLM/pull/1384) | `c32b74a62cd14fda8ea0b5d356ea048598dce26e` | 14 | retained, unqualified |

The publication commit retains each exact original head as a parent or ancestor, preserving complete trees and authorship. `patches/` contains original full diffs; `records/` preserves original metadata, all comments, reviews and review-thread comments at the audit snapshot. Original PR links retain subsequent discussion. No source branch is deleted.

## Audit and remaining qualification

Many operators are default-off alternatives with shared dependencies. Preserve their opt-in admission and numerical contracts. Need native builds plus target-checkpoint route-hit, TP4 graph replay, slow-rank/concurrent-state, prefill/decode tails and quality checks; component timing alone does not qualify model integration. Do not enable these defaults during consolidation.

Conflicting alternatives must be reconciled explicitly, preserving the qualified main behavior until a corrected implementation passes its own admission gates. Numeric/kernel changes require paired quality and acceptance checks as well as whole-workload measurements. Closing an old entry as superseded is **not a merge or an approval of its runtime changes**.

Prepared with Codex under the repository owner's explicit consolidation instruction.
