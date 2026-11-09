# DeepSeek-V4 attention and indexing qualification queue

This draft retains related proposals and their review records. Original patches are **unapplied** unless a separate corrected-source section explicitly identifies applied files. Do not merge this archive as accepted runtime code. CI on the archive baseline does not qualify the proposed operators, models, policies or dependencies.

## Retained sources

| Original PR | Exact head | Files | Status |
| --- | --- | --- | --- |
| [#1411](https://github.com/1CatAI/1Cat-vLLM/pull/1411) | `efc21383c0247990387132ab4ea81c505a618286` | 2 | retained, unqualified |
| [#1347](https://github.com/1CatAI/1Cat-vLLM/pull/1347) | `6968a8b40e7f859f63d16cff26497316f05f7d06` | 74 | retained, unqualified |
| [#1344](https://github.com/1CatAI/1Cat-vLLM/pull/1344) | `5e76f22ed4339bf373c55271eb9b50e4e4a5105f` | 1 | retained, unqualified |
| [#1328](https://github.com/1CatAI/1Cat-vLLM/pull/1328) | `bfc5bcf30c088577237f290fe6dbac480dca7839` | 2 | retained, unqualified |
| [#1325](https://github.com/1CatAI/1Cat-vLLM/pull/1325) | `24c5ede84ca62a32a76f8dc5e1375deb1a6cb353` | 2 | retained, unqualified |
| [#1417](https://github.com/1CatAI/1Cat-vLLM/pull/1417) | `c1cd04d9d694e38f49b35211f67836f529279002` | 2 | retained, unqualified |
| [#1282](https://github.com/1CatAI/1Cat-vLLM/pull/1282) | `b9fbe637c823635d583c0dceaac9abf8475d08c2` | 76 | retained, unqualified |
| [#1321](https://github.com/1CatAI/1Cat-vLLM/pull/1321) | `7110faceace926ce92d6187bf996f5e1c666a3b2` | 11 | retained, unqualified |
| [#1425](https://github.com/1CatAI/1Cat-vLLM/pull/1425) | `bf2d1e24cf4b07a8bd6f927adf06df8dda97fc3c` | 10 | retained, unqualified |
| [#1279](https://github.com/1CatAI/1Cat-vLLM/pull/1279) | `4a95e6beaff7f5e3882e8fed5a05d15e0fb4276a` | 48 | retained, unqualified |
| [#1280](https://github.com/1CatAI/1Cat-vLLM/pull/1280) | `61dddbfa0b0935342b59c8c9e60d1d85cfd08d4f` | 60 | retained, unqualified |

The publication commit retains each exact original head as a parent or ancestor, preserving complete trees and authorship. `patches/` contains original full diffs; `records/` preserves original metadata, all comments, reviews and review-thread comments at the audit snapshot. Original PR links retain subsequent discussion. No source branch is deleted.

## Audit and remaining qualification

Persistent top-k CTA synchronization, float4 tails, FP32 scoring, compressed KV and context buckets require rebuilt native kernels and threshold/tail/context coverage. Validate both SM70 and SM75 paths where claimed, TP/DCP layouts, FP8 KV, graph replay and target-checkpoint quality before selecting new defaults. #1344/#1417 small diffs are not automatically safe without path qualification.

Conflicting alternatives must be reconciled explicitly, preserving the qualified main behavior until a corrected implementation passes its own admission gates. Numeric/kernel changes require paired quality and acceptance checks as well as whole-workload measurements. Closing an old entry as superseded is **not a merge or an approval of its runtime changes**.

Prepared with Codex under the repository owner's explicit consolidation instruction.
