# GLM-5.3 model and KDA kernel integration queue

This draft retains related proposals and their review records. Original patches are **unapplied** unless a separate corrected-source section explicitly identifies applied files. Do not merge this archive as accepted runtime code. CI on the archive baseline does not qualify the proposed operators, models, policies or dependencies.

## Retained sources

| Original PR | Exact head | Files | Status |
| --- | --- | --- | --- |
| [#1286](https://github.com/1CatAI/1Cat-vLLM/pull/1286) | `75fd75d70817ad50d9e1aa5436faee9b9e6c9b06` | 116 | retained, unqualified |
| [#1348](https://github.com/1CatAI/1Cat-vLLM/pull/1348) | `ceceb328eb4a816f4963ad30af80ed3bcf7a1ac1` | 117 | retained, unqualified |
| [#1410](https://github.com/1CatAI/1Cat-vLLM/pull/1410) | `f03e254d782a6ad9d8ae724cd3685b689e430385` | 8 | retained, unqualified |
| [#1125](https://github.com/1CatAI/1Cat-vLLM/pull/1125) | `da13d2d9ca4bba2cf7757442afde63c38b407d78` | 8 | retained, unqualified |
| [#1159](https://github.com/1CatAI/1Cat-vLLM/pull/1159) | `44d292046a7b206fe3504315fd1de505aea89c3c` | 5 | retained, unqualified |
| [#1165](https://github.com/1CatAI/1Cat-vLLM/pull/1165) | `60dd820a7cba6a3388934e791ac72cbfe50606a1` | 2 | retained, unqualified |
| [#1166](https://github.com/1CatAI/1Cat-vLLM/pull/1166) | `74d2cf6da8412fa0a6bc98da0c02b36686be2b84` | 3 | retained, unqualified |
| [#1228](https://github.com/1CatAI/1Cat-vLLM/pull/1228) | `593fd096604900f59cc09a0279d5101dbfc570f7` | 4 | retained, unqualified |
| [#1092](https://github.com/1CatAI/1Cat-vLLM/pull/1092) | `587fac6eb37b8aee1cd42df76f4af284d30f503d` | 8 | retained, unqualified |

The publication commit retains each exact original head as a parent or ancestor, preserving complete trees and authorship. `patches/` contains original full diffs; `records/` preserves original metadata, all comments, reviews and review-thread comments at the audit snapshot. Original PR links retain subsequent discussion. No source branch is deleted.

## Audit and remaining qualification

GLM-5.3-Flash checkpoints and KDA recurrent/tail-ring state are not qualified by Qwen runs. Require speculative-width, padded-stride, recurrent-state, prompt/decode and image-independent model tests with matched precision. SM75 TileLang admission needs an SM75 device. Overlapping base-model stacks remain unapplied proposals until reconciled.

Conflicting alternatives must be reconciled explicitly, preserving the qualified main behavior until a corrected implementation passes its own admission gates. Numeric/kernel changes require paired quality and acceptance checks as well as whole-workload measurements. Closing an old entry as superseded is **not a merge or an approval of its runtime changes**.

Prepared with Codex under the repository owner's explicit consolidation instruction.
