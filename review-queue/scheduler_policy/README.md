# Scheduler and prefix-cache policy qualification queue

This draft retains related proposals and their review records. Original patches are **unapplied** unless a separate corrected-source section explicitly identifies applied files. Do not merge this archive as accepted runtime code. CI on the archive baseline does not qualify the proposed operators, models, policies or dependencies.

## Retained sources

| Original PR | Exact head | Files | Status |
| --- | --- | --- | --- |
| [#943](https://github.com/1CatAI/1Cat-vLLM/pull/943) | `83701788a2467aa37e07a6c77bce3365ec8af16e` | 5 | retained, unqualified |
| [#1444](https://github.com/1CatAI/1Cat-vLLM/pull/1444) | `86a3ef1062490d12bf2bf4cf23c191afb2f089e1` | 2 | retained, unqualified |
| [#1445](https://github.com/1CatAI/1Cat-vLLM/pull/1445) | `8275b418c00400349c5bffd98e0ef9223ca3d2c3` | 3 | retained, unqualified |

The publication commit retains each exact original head as a parent or ancestor, preserving complete trees and authorship. `patches/` contains original full diffs; `records/` preserves original metadata, all comments, reviews and review-thread comments at the audit snapshot. Original PR links retain subsequent discussion. No source branch is deleted.

## Audit and remaining qualification

Scheduling/cache policy changes need correctness under preemption, mixed prefill/decode, prefix eviction, recurrent state and concurrent requests; retaining free hashes must preserve allocator invariants. Measure fairness, TTFT/prefill and throughput with identical workloads before any default phase switch.

Conflicting alternatives must be reconciled explicitly, preserving the qualified main behavior until a corrected implementation passes its own admission gates. Numeric/kernel changes require paired quality and acceptance checks as well as whole-workload measurements. Closing an old entry as superseded is **not a merge or an approval of its runtime changes**.

Prepared with Codex under the repository owner's explicit consolidation instruction.
