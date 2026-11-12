# Dependency and build matrix reconciliation queue

This draft retains related proposals and their review records. Original patches are **unapplied** unless a separate corrected-source section explicitly identifies applied files. Do not merge this archive as accepted runtime code. CI on the archive baseline does not qualify the proposed operators, models, policies or dependencies.

## Retained sources

| Original PR | Exact head | Files | Status |
| --- | --- | --- | --- |
| [#1350](https://github.com/1CatAI/1Cat-vLLM/pull/1350) | `2e6e79d0d9594da11b2567476a347ad95281cba1` | 9 | retained, unqualified |
| [#1351](https://github.com/1CatAI/1Cat-vLLM/pull/1351) | `6f9475bb59c4fce2b3024ef0f0854a9efbede264` | 1 | retained, unqualified |
| [#1281](https://github.com/1CatAI/1Cat-vLLM/pull/1281) | `f7e8157be362800c529aeb2ca67ad6f5941227f5` | 2 | retained, unqualified |
| [#1013](https://github.com/1CatAI/1Cat-vLLM/pull/1013) | `1c3592f23a27304f8384a2675c1003fe4581effd` | 1 | retained, unqualified |
| [#1014](https://github.com/1CatAI/1Cat-vLLM/pull/1014) | `fa361081e391d2456e0929defb576b108dbc3372` | 1 | retained, unqualified |
| [#1015](https://github.com/1CatAI/1Cat-vLLM/pull/1015) | `4da969c9cc24c3319440fcf1d436d01bc5fc6df4` | 1 | retained, unqualified |

The publication commit retains each exact original head as a parent or ancestor, preserving complete trees and authorship. `patches/` contains original full diffs; `records/` preserves original metadata, all comments, reviews and review-thread comments at the audit snapshot. Original PR links retain subsequent discussion. No source branch is deleted.

## Audit and remaining qualification

#1350 mixes torch2.10.0 with torchaudio2.11.1+cpu and FlashInfer Python0.7.2 with cubin0.6.11. Reconcile ABI/version constraints and regenerate maintained lock variants rather than merging lock fragments. Validate resolver/import/build matrices for CUDA/ROCm/CPU and major redis/plotly/typepy changes; workflow setup-python also needs its own CI execution.

Conflicting alternatives must be reconciled explicitly, preserving the qualified main behavior until a corrected implementation passes its own admission gates. Numeric/kernel changes require paired quality and acceptance checks as well as whole-workload measurements. Closing an old entry as superseded is **not a merge or an approval of its runtime changes**.

Prepared with Codex under the repository owner's explicit consolidation instruction.
