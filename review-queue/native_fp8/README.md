# Native FP8/QPN and SM75 kernel qualification

This draft retains related proposals and their review records. Original patches are **unapplied** unless a separate corrected-source section explicitly identifies applied files. Do not merge this archive as accepted runtime code. CI on the archive baseline does not qualify the proposed operators, models, policies or dependencies.

## Retained sources

| Original PR | Exact head | Files | Status |
| --- | --- | --- | --- |
| [#1448](https://github.com/1CatAI/1Cat-vLLM/pull/1448) | `f165b6ac47eeea932d6c7b482d7c45a144f97693` | 8 | retained, unqualified |
| [#1157](https://github.com/1CatAI/1Cat-vLLM/pull/1157) | `dd4c9f85059c62af1774c17d2d332a1516b4b12e` | 1 | retained, unqualified |
| [#1412](https://github.com/1CatAI/1Cat-vLLM/pull/1412) | `c28f814964913fc40cc7a378cd5e77ca5da3fa7a` | 2 | retained, unqualified |
| [#1021](https://github.com/1CatAI/1Cat-vLLM/pull/1021) | `13825030f926dd92ddf60061d507b8325daf9590` | 7 | retained, unqualified |
| [#1187](https://github.com/1CatAI/1Cat-vLLM/pull/1187) | `739083e52cf3f3871d6e1eb91eda814d66f2f943` | 4 | retained, unqualified |
| [#1346](https://github.com/1CatAI/1Cat-vLLM/pull/1346) | `68aeae6b705adffed48ae6bfca2117195cfb9084` | 1 | retained, unqualified |
| [#1278](https://github.com/1CatAI/1Cat-vLLM/pull/1278) | `c15d1446fa00d3196243da5de951d8ed3586de0a` | 20 | retained, unqualified |
| [#1322](https://github.com/1CatAI/1Cat-vLLM/pull/1322) | `03485453bee261228e6a4591e3ca93cfb9551fcd` | 3 | retained, unqualified |
| [#1069](https://github.com/1CatAI/1Cat-vLLM/pull/1069) | `8837881a4dd8441229b9646cd5d73c4a31e945c3` | 1 | retained, unqualified |

The publication commit retains each exact original head as a parent or ancestor, preserving complete trees and authorship. `patches/` contains original full diffs; `records/` preserves original metadata, all comments, reviews and review-thread comments at the audit snapshot. Original PR links retain subsequent discussion. No source branch is deleted.

## Audit and remaining qualification

The complete development native library scored192/200 on GSM8K-200 against the release-native193/200 control; question153 newly hit the length limit. Keep it isolated. #1187 changes runtime tuning defaults and needs a matched frozen-tuning control, not attribution to kernels. #1278/#1346 require actual SM75 qualification; a V100 cannot establish Turing behavior. Validate native correctness, graph replay, concurrent resource ownership, matched model quality and whole-round timing before promotion.

Conflicting alternatives must be reconciled explicitly, preserving the qualified main behavior until a corrected implementation passes its own admission gates. Numeric/kernel changes require paired quality and acceptance checks as well as whole-workload measurements. Closing an old entry as superseded is **not a merge or an approval of its runtime changes**.

Prepared with Codex under the repository owner's explicit consolidation instruction.
