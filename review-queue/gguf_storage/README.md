# GGUF/EXL3 storage and precision integration queue

This draft retains related proposals and their review records. Original patches are **unapplied** unless a separate corrected-source section explicitly identifies applied files. Do not merge this archive as accepted runtime code. CI on the archive baseline does not qualify the proposed operators, models, policies or dependencies.

## Retained sources

| Original PR | Exact head | Files | Status |
| --- | --- | --- | --- |
| [#959](https://github.com/1CatAI/1Cat-vLLM/pull/959) | `f02e37e081029791542a5dd43ca41d3a661ecc79` | 5 | retained, unqualified |
| [#1102](https://github.com/1CatAI/1Cat-vLLM/pull/1102) | `b96d43e5c8df201992a181836d631198ef827edf` | 11 | retained, unqualified |
| [#1167](https://github.com/1CatAI/1Cat-vLLM/pull/1167) | `403a76e9ad09fd0466b698e9a7b24f01a1a3f26b` | 104 | retained, unqualified |
| [#1020](https://github.com/1CatAI/1Cat-vLLM/pull/1020) | `55915c7f16c593b2dc7388c80ded0ff52094edc8` | 41 | retained, unqualified |
| [#894](https://github.com/1CatAI/1Cat-vLLM/pull/894) | `73bc8ab85bd2b611fd5984422a0a6ce1ef6dfae2` | 7 | retained, unqualified |
| [#890](https://github.com/1CatAI/1Cat-vLLM/pull/890) | `ac7c9eab16b2d37538e50c9fc397edfac0513030` | 13 | retained, unqualified |
| [#822](https://github.com/1CatAI/1Cat-vLLM/pull/822) | `813bf2e5e8b2c0c04452c5354d510ea7d2a9c83e` | 15 | retained, unqualified |
| [#897](https://github.com/1CatAI/1Cat-vLLM/pull/897) | `881ad90eae7ee4c7159ff7973bad0261b1a30ac8` | 24 | retained, unqualified |

The publication commit retains each exact original head as a parent or ancestor, preserving complete trees and authorship. `patches/` contains original full diffs; `records/` preserves original metadata, all comments, reviews and review-thread comments at the audit snapshot. Original PR links retain subsequent discussion. No source branch is deleted.

## Audit and remaining qualification

Original-byte expert banks, EXL3 loading, host KV and GGUF memory lifetime changes require the corresponding checkpoints and cold/hot cache, eviction, graph replay and memory-capacity runs. #894/#890 alter numerical storage/logit paths: require matched quality and sampling checks. Preserve each alternative; do not blindly concatenate overlapping model/provider changes.

Conflicting alternatives must be reconciled explicitly, preserving the qualified main behavior until a corrected implementation passes its own admission gates. Numeric/kernel changes require paired quality and acceptance checks as well as whole-workload measurements. Closing an old entry as superseded is **not a merge or an approval of its runtime changes**.

Prepared with Codex under the repository owner's explicit consolidation instruction.
