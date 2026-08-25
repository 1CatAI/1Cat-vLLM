# SM70 DFlash2 acceptance research without retraining

> **Status:** the default-off alignment capture and offline analyzer are
> development tooling. They do not change the production proposal policy and
> do not claim an acceptance or performance improvement by themselves.

## Scope and frozen contracts

- Date: 2026-08-25.
- Integration base: `onecat/main@34403018d917054dd7765d5e820ad29c8d342348`.
- Branch: `codex/v100-v100-dflash2-acceptance-rd-20260825-113155`.
- No draft or target retraining and no checkpoint edits.
- Target validation remains the complete MRV2 block-eight workload: seven draft
  tokens plus one target bonus row. DDTree index reuse and wider verifier trees
  are outside this campaign.
- Target sampling remains exact standard rejection at temperature 1.0, top-p
  0.95, and top-k 20. A candidate proposal distribution is admissible only if
  the selector draw and the rejection sampler consume the same exact `q`.

The score gate remains the retained 128-row mixed32 dataset with SHA256
`85ce10d84735ec981c001663d5748b1939a81892ef3ccd4549aec66177795607`.
Its private filesystem location is intentionally not committed.
It uses sequential B1, a fixed seed, xhigh reasoning, at most 2,048 generated
tokens, and natural EOS. Corrected MBPP-32 and WikiText-2 PPL retain their
existing independent gates. The historical native-grouped result is 83/96 on
GSM8K/MATH-500/HumanEval, 25/32 on corrected MBPP, request-mean completion
length 4.238430, and pooled acceptance length 3.749104.

The production performance gate remains the PR #288 practical contract: TP4
V100, 256K maximum context, `max_num_batched_tokens=4096`, prefix caching,
Mamba align mode, FP8 E5M2 target KV, FP16 draft KV, Flash-V100, probabilistic
DFlash2, tool/reasoning parsers, and FULL CUDA Graph. Its short-context complete
round is 18.465--18.603 ms. Acceptance work may not increase verifier rows or
regress that round by more than measurement noise.

## First target and promotion gate

The first research target is at least +0.5 request-mean completion tokens per
verification round on the unchanged mixed32 contract. Promotion additionally
requires:

1. no aggregate task-score loss against the paired target-only and current
   DFlash2 controls;
2. no corrected-MBPP or WikiText PPL regression;
3. exact rejection-distribution tests and normalized proposal rows;
4. unchanged eight-row target validation and at most 0.05 ms incremental
   selector latency on SM70;
5. no material acceptance or latency regression at 1K, 32K, 128K, or 256K.

## Selector headroom audit

The first implementation is diagnostic-only. With both compact sparse
rejection and `VLLM_SPEC_DUMP_ALIGNMENT=1`, the speculator keeps the checkpoint
top16 IDs, unary logits, full `7 x 16 x 16` selector lattice, and exact realized
proposal rows. The target sampler records its top20 rows and the actual strict
rejection outcome. Disabled diagnostics allocate no shadow lattice and add no
copy or sampler operation.

`benchmarks/analyze_dflash2_selector_alignment.py` reports, by draft depth:

- current overlap `sum(min(p, q))`;
- target probability mass covered by the fixed top16 candidate support;
- the gap between support mass and current overlap;
- one-step counterfactual sweeps over proposal temperature, proposal nucleus,
  unary/edge calibration, and backward log-sum-exp future messages.

Even sampler steps tune the candidates and odd steps are a held-out report.
Counterfactual values after a changed path are explicitly only recorded-prefix
overlap proxies, not end-to-end acceptance claims. Survivors must be rerun on
the frozen dataset.

## Initial verification

- Focused Ruff lint and format: passed.
- Diagnostic analyzer synthetic checks: passed.
- CPU DFlash2 suite with CUDA hidden: 77 passed, 12 expected CUDA skips.
- No GPU performance or end-to-end acceptance result has been claimed yet.
