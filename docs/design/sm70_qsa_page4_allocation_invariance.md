# SM70 QSA page4 allocation invariance

## Defect and contract

The selected logical tokens and their K/V values can remain identical while
the physical KV page allocation changes between requests. The grouped page4
planner formerly emitted pages in physical hash-slot order, within active-row
categories. Changing allocation therefore changed the online-softmax and
tensor-core reduction order. This can perturb FP16 attention outputs and
downstream logits without an incorrect selected set or incorrect K/V values.

The fix guarantees a stable logical plan for fixed query grouping, logical
selections, visibility, and page-alias relationships. Physical addresses are
payload, not ordering keys. It does **not** promise bitwise equality across
different batch shapes, changed prefix-sharing relationships, different
attention implementations, or quantization formats.

## Implementation history

| Change | Role in this path |
|---|---|
| [#378](https://github.com/1CatAI/1Cat-vLLM/pull/378), integrated with [#382](https://github.com/1CatAI/1Cat-vLLM/pull/382) | Introduced the SM70 Flash-V100 virtual-page4 path; the single-row path sorted physical microblock IDs for locality. |
| [#387](https://github.com/1CatAI/1Cat-vLLM/pull/387), commit `94ce990ce85abeb12e3948ee1c4f518594bacf25` | Added eight-query K/V sharing, physical-page hash deduplication, mask merging, category packing, and hash-slot-order emission. |
| [#466](https://github.com/1CatAI/1Cat-vLLM/pull/466), commit `186c9e3585b109b88c070603a181dd6826400153` | Lowered the default page4 admission from 4096 to 64 actual query rows and admitted grouped prefixes with XQA tails. It widened exposure, rather than introducing the hash planner. |

These are source-history findings, not a GPU regression bisection of every
historical release. This defect is independent of AWQ grouped **decode**:
the observed first difference arose during QSA grouped **prefill**, with the
experimental AWQ grouped-decode gate disabled throughout the diagnosis.

## Causal investigation

The diagnostic contract used one frozen AWQ checkpoint/runtime, four V100s,
TP4, MTP0, FP16 activations and KV, MRv2, FULL_AND_PIECEWISE graphs,
8192 batched tokens, prefix caching off, asynchronous scheduling off,
fixed prompt token IDs, fixed enqueue order, and fixed request-slot order.
The four prompts contained 42/41/58/44 tokens. Each diagnostic request generated
only its first token, with temperature 0, `min_tokens=0`, and `ignore_eos=false`.
This was numerical diagnosis, not an output-quality or throughput benchmark.

1. In repeated C4 batches, all four ranks had identical inputs through the first
   three layers. The first different boundary was Layer3 QSA (zero-based layer
   numbering). Attention output max absolute difference was `0.0009765625`;
   full-logit max absolute difference was `0.01171875`, with unchanged argmax.
2. Finer observations found identical Q/K/V, logical positions, request mapping,
   selected token indices, and all effective K/V read in logical order.
   Physical block tables differed. All 23 query groups retained the same
   logical page/mask sets, but 17 had a different order. QSA core max absolute
   difference was `0.00048828125` on each rank.
3. Holding physical allocation fixed, 24 extra attention replays were bitwise
   identical. The measured symptom was not fixed-input kernel randomness.
4. Diagnostic CPU sorting of only Layer3's plan restored equality there and
   moved the first difference to Layer7, the next QSA layer.
5. Applying the same control to all 12 QSA layers made all four ranks' 175
   observed boundaries and complete logits bitwise identical.
6. Removing sorting while retaining observation and CPU synchronization brought
   the Layer3 difference back. Extra synchronization alone did not explain the
   result.

The actual C4 step had 185 query rows: 184 grouped rows plus one XQA tail.
The short C1 control had 44 rows and did not enter page4. Long C1 prefill can
still enter the path; concurrency labels are not route evidence.

Instrumentation can affect compilation boundaries. Causal attribution rests on
same-instrumentation intervention/reversal and local fixed-input replays, not
on interpreting diagnostic timings as performance. The earlier freely batched
70-versus-61-token answer divergence and the independent AWQ W13 operator
microdifference are not claimed to be completely explained by this experiment.

## Upstream comparison and duplicate-work check

Original vLLM QSA iterates logical selection positions, then uses the block
table for addressing. It does not use this SM70 physical-hash union planner.
Related upstream work must not be conflated with this defect:

- [1Cat #394](https://github.com/1CatAI/1Cat-vLLM/pull/394) already provides
  exact lexicographic QSA top-k selection on the tested SM70 contract.
  Those selections were bitwise equal during this investigation.
- [vLLM #55122](https://github.com/vllm-project/vllm/pull/55122) addresses
  generic `persistent_topk` membership/tie/order nondeterminism, not ordering
  subsequently introduced by the page4 planner.
- [vLLM #54873](https://github.com/vllm-project/vllm/pull/54873) skips unused
  sparse-attention selection entries and tunes launch profiles. It does not
  fix the 1Cat planner.
- [vLLM RFC #55394](https://github.com/vllm-project/vllm/issues/55394) proposes
  a related query-tile union. Its prototype sorts logical blocks before
  physical mapping. That principle is useful here; the GB10 single-request
  prototype is not a ready-made SM70 concurrent replacement and is not ported.

At the 2026-09-04 duplicate check, PR #55122 was open, PR #54873 was merged, and
RFC #55394 was open. No direct repair of this planner was found. The existing
Triton fallback remains a correctness/performance control; upstream use does
not by itself establish V100 performance or cross-batch invariance.

## Narrow repair

The grouped planner retains physical-page deduplication and OR-merged token
masks. Each entry also records its smallest logical owner:
`(first contributing query within the group, logical four-token block)`.
An atomic minimum makes shared-page ownership independent of insertion order.
CUB block radix sort orders entries by active-row category and logical owner.
The original category packing, eight-page padding, and attention kernel remain.
This uses CUB's existing sorting primitive, not a new sorting algorithm.

The single-row XQA path, including non-grouped tails, also needs repair. Its
existing GPU `torch.sort` now sorts packed logical keys carrying physical IDs
as payload. The causal partial page remains after complete pages and invalid
slots remain last. Only integer planning changes; attention arithmetic,
weights, quantization, scheduler policy, and route thresholds are unchanged.

The grouped hash and owner arrays occupy 96 KiB of shared memory. Once entries
are held in registers, that storage is reused for CUB sorting and category
scans. There is no added global-memory grouped workspace or weight/KV copy.
The single-row sorting keys grow from int32 to int64, so temporary metadata
memory is not claimed to be unchanged. Resource use and latency require GPU
measurement; absence of a global grouped allocation does not imply zero cost.
The tested CUDA 12.8 SM70 binary reports 128 registers per planner thread,
zero local memory, and zero stack bytes. Dynamic shared memory is 96 KiB per
CTA; the resource dump's zero static-shared value does not include it.

## Validation

The regression is `tests/kernels/test_sm70_qsa_page4_plan.py`. It checks exact
logical-reference plans, shared physical pages, collisions, invalid rows,
all-empty groups, wide unions, selection permutations, page sizes 4/16/32,
graph replay with relocated inputs, contiguous/interleaved FP16 and E4M3 KV, and a
185-row grouped-plus-XQA-tail batch.

```bash
.venv/bin/python -m pytest -q tests/kernels/test_sm70_qsa_page4_plan.py
.venv/bin/python -m pytest -q tests/models/qwen4_exp/test_qsa_ops.py
```

Before this repair, the initial 20-case regression produced 17 failures and
three passes (the all-empty controls). Both attention relocation checks failed
at the bitwise-output assertion. The later single-row and 185-row integration
tests were added separately and must not be counted as part of that initial run.

The candidate builds with CUDA 12.8 / SM70. On one V100, all 26 new regression
cases and all 14 existing QSA-ops tests pass (40 total). All applicable
pre-commit hooks pass. Captured Layer3 inputs from all four TP ranks reproduce
the baseline relocation difference and become bitwise equal after the repair:

| Rank | Baseline different output elements | Baseline max absolute difference | Repaired different elements |
|---|---:|---:|---:|
| 0 | 1625 | 0.00048828125 | 0 |
| 1 | 1852 | 0.00048828125 | 0 |
| 2 | 1515 | 0.00048828125 | 0 |
| 3 | 1866 | 0.00048828125 | 0 |

The existing Triton fallback is also bitwise allocation-invariant for these
four captured pairs. This is direct replay evidence, in addition to the
upstream source inspection above.

### Bounded operator timings

CUDA-event medians, three warmups and 20 samples per call, one V100, FP16 KV,
Hq/Hkv/D = 6/1/256, page size 16. The synthetic cases have four 64K requests,
512 shared selected logical blocks per query, randomized physical allocation,
and the stated total query-row count. They favor K/V sharing and are **not**
a server concurrency contract or full-model throughput measurement.

| Input | Old page4 (ms) | Repaired page4 (ms) | Change | Existing Triton (ms) |
|---|---:|---:|---:|---:|
| Captured 185-row step, allocation A | 0.3518 | 0.3615 | +2.8% | 2.6552 |
| Captured 185-row step, allocation B | 0.3600 | 0.3635 | +1.0% | 2.4049 |
| Synthetic 1024 rows | 1.5130 | 1.5811 | +4.5% | 14.3713 |
| Synthetic 8192 rows | 9.1136 | 9.7823 | +7.3% | 99.3587 |

The Triton column comes from the candidate probe. Baseline probe Triton
medians were 2.6563/2.6588/14.4645/99.3930 ms respectively; short host-driven
operator calls show noise and these single-session numbers are not confidence
intervals. For 8192 rows, planner-only time grows from 0.6610 to 1.3251 ms.
The cost is real; retaining page4 with logical ordering is still preferable
to the measured full Triton fallback on this V100 contract. Synthetic
reference errors versus Triton remain small (repaired maximum absolute
difference 0.0001220703125 in both cases), not bitwise cross-kernel equality.

### Natural-EOS full-model sanity

A bounded AWQ / TP4 / MTP0 / FP16-KV run used the same frozen compatible
runtime as the diagnosis, including its existing AWQ wrapper admission repair
and disabled experimental AWQ grouped-decode gate. No model trace, sampler
replacement, or attention intervention was installed. Prompt IDs, enqueue
order, and free request-slot order were fixed; each C1/C4/C8 shape ran twice
with temperature 0, `min_tokens=0`, `ignore_eos=false`, and a 96-token limit.

All 26 request outputs stopped naturally, had finite reported logprobs, and
passed basic answer checks. Token IDs matched exactly between the two runs
of each shape. Worker provenance confirmed the repaired binary and Python
source on all four ranks. Actual page4 route logs reported 184 grouped + 1
XQA row for C4 and 368 grouped + 2 XQA rows for C8. KV capacity was 386,392
tokens, unchanged from the prior same-configuration run.

Cross-shape/position differences **remain**: the same open-ended Chinese
prompt produced 61 tokens at C1, 70 at C4, and 61/71 at its two C8 positions.
Each position reproduced its own sequence on the repeat. This run validates
short same-shape repeatability, not cross-batch or cross-position invariance;
the remaining divergence has not been localized by this model sanity check.
It also does not establish long-context quality, NVFP4 model acceptance, E2E
performance, or a resolution of every prior generation divergence.
