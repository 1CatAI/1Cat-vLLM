# SM70 NVFP4 DFlash2 output-quality audit

## Scope and provenance

This audit starts from private integration commit
`d63e9490f65f9e01f6649053c1ab72922034b931` on branch
`codex/v100-dflash2-quality-audit-20260826-124851`. It covers the practical
Qwen3.8-27B NVFP4 target plus official BF16 DFlash2 drafter on TP4 V100. Remote
deployment and packaging are out of scope.

The production contract remains temperature 1.0, top-p 0.95, top-k 20,
`xhigh` reasoning, seven probabilistic draft tokens, FP8 E5M2 target KV, FP16
draft KV, FULL target/draft CUDA Graphs, prefix caching, Mamba align, 256K
maximum context, 4096 maximum batched tokens, and the Qwen tool/reasoning
parsers. Quality is judged by scored distributions rather than greedy or
bitwise identity.

Upstream was checked once on 2026-08-26. vLLM PR
[`#52816`](https://github.com/vllm-project/vllm/pull/52816) is now merged. Its
following draft-logit cache-stride fix, vLLM PR `#53017`, does not apply to
this DFlash2 path: DFlash2 uses its own sparse top-16 cache kernel and passes
both request and step strides explicitly. Open quantized-drafter fixes are
also outside this contract because the retained drafter is the official BF16
checkpoint.

The structured-output audit found a separate relevant upstream chain after
the local May 29 fork point: vLLM PRs
[`#44297`](https://github.com/vllm-project/vllm/pull/44297),
[`#44993`](https://github.com/vllm-project/vllm/pull/44993),
[`#52805`](https://github.com/vllm-project/vllm/pull/52805), and
[`#53046`](https://github.com/vllm-project/vllm/pull/53046). They constrain
draft rows after a reasoning boundary, advance only the post-boundary suffix,
stop XGrammar at termination, and avoid logging expected invalid pre-mask
drafts as FSM errors. This is directly relevant to the still-open upstream
DFlash2 `json_object` report
[`#53777`](https://github.com/vllm-project/vllm/issues/53777); only this minimal
behavioral closure is backported here.

## Existing evidence

- The repaired stable RMSNorm extension SHA `7689e5...d449` passes its 13-case
  batched-weight suite. The stale SHA `fe986a...9ba8c` reproduced the historical
  row-zero context-K normalization bug and is not valid quality evidence.
- Draft-MLP QPN8 was removed after request-mean acceptance changed by
  `-0.06397`, beyond the fixed `-0.05` gate. It remains disabled.
- The corrected MBPP-32 16K pair scores target-only/DFlash2 `31/32` versus
  `32/32`; EvalPlus base `30/31` versus `31/31`; and EvalPlus plus `27/31`
  versus `30/31`. HumanEval-32 EvalPlus is `32/32` for both. This is useful but
  is only one sampling realization.
- The practical selector-QPN8 run on 64 GSM8K rows changed request-mean
  acceptance `5.39851 -> 5.43951`, but its speed comparison used different
  physical GPU groups. The design record explicitly kept the route opt-in.
- The packaged practical launch nevertheless forced
  `VLLM_SM70_DFLASH2_QPN8_RERANK=1` and
  `VLLM_SM70_DFLASH2_QPN8_DENSE_ORDER=0`. This bypasses the source defaults
  (`rerank=0`, `dense_order=1`) and must not be treated as a promoted quality
  contract.
- Historical real-hidden candidate-order shadows contain no missing dense
  top-20 token from the QPN8 top-64 support in the measured sample, and their
  returned FP16 values match. They still change the local top-20 token set on
  69/4640 top-20 rows (1.49%) and top-1 on 6/4640 rows because equal-valued
  cutoff tokens are resolved in candidate order.
- An offline scan of 2,048 retained selector-alignment dumps covers 16,384
  target rows. The 19th and 20th returned FP16 logits are equal on 544 rows
  (3.3203%); four rows expose three equal cutoff entries within the returned
  support. Visible cutoff probability mass averages about `0.000445`. This is
  small per step but can accumulate in multi-thousand-token coding outputs.
- The retained 32K eager LongBench pair has four Qasper/NarrativeQA records.
  Target-only and DFlash2 have identical token hashes on all four records and
  the same average score, `36.6667`. This is useful long-prompt parity evidence,
  but it does not replace Graph, prefix-cache, or 128K/256K validation.
- The retained grouped verifier is lower risk: its dedicated operator and
  graph-replay tests are bitwise equal to the accepted fallback under the
  admitted SM70 shape. Fixed interleaved page addressing and staged page IDs
  still require the paired state-pollution checks below.

## Hypotheses, in test order

1. **Experimental selector launch configuration.** Candidate-order QPN8 may
   alter low-probability target support at FP16 ties or very rarely miss the
   dense top-20 support. Dense-order QPN8 should retain most of the speed while
   restoring the local dense-vocabulary tie contract.
2. **Compact TP top-k boundary.** Sparse target rejection merges per-rank
   candidates rather than materializing the full vocabulary. Non-tied rows are
   exact; tied cutoff rows need a real-hidden global shadow before the path is
   called distribution-equivalent to target-only.
3. **Graph/prefix/request-slot state.** Same-seed requests must be checked cold,
   warm, after mixed-length contamination, and after prefix-cache reuse. A
   target-only repeat is required before attributing cross-process stochastic
   variation to DFlash2.
4. **Sparse rejection arithmetic.** Temperature application, target top-k then
   top-p masking, conditional selector scores, token-keyed acceptance uniforms,
   and `log1p` residual sampling are statically consistent. Dense versus sparse
   same-source scoring remains the decisive dynamic check.

## Minimal isolation matrix

Every arm uses the same source, extension hashes, physical GPU group, model,
prompt order, request seeds, natural EOS, and practical engine parameters.

| Arm | DFlash2 | Sparse target rejection | Selector QPN8 | Dense order | Purpose |
| --- | --- | --- | --- | --- | --- |
| `T0` | no | n/a | n/a | n/a | target-only score and repeat variance |
| `D0` | yes | off | off | n/a | dense DFlash2 reference |
| `D1` | yes | on | off | n/a | isolate compact rejection |
| `D2` | yes | on | on | on | quality-first accelerated selector |
| `D3` | yes | on | on | off | reproduce current packaged launch |

The first screen uses a small fixed coding/reasoning subset and at least three
predeclared seeds. It records token/text hashes as diagnostics but promotes by
paired scores, natural-stop/invalid rates, acceptance, and confidence
intervals. Eager real-hidden shadows are run only long enough to measure QPN8
support coverage and compact-global top-k boundary behavior. Graph runs then
check immediate repeats, mixed-length contamination, cold/warm prefix cache,
and request-slot reuse.

The full gate uses HumanEval and MBPP with EvalPlus, stratified LiveCodeBench,
GSM8K, MATH-500, WikiText prompt PPL, tool-call validity, and long-output coding
health. At least three predeclared sampling seeds are required for executable
coding scores. No suite may show a statistically supported regression versus
the same NVFP4 target-only route. DFlash2 request-mean acceptance may not fall
more than `0.05` from the accepted dense DFlash2 reference.

## Performance gate

Quality fixes are measured on the same practical 256K/4096/prefix/tool contract.
The accepted short baseline is approximately 18 ms per complete speculative
round; high-acceptance prompts can exceed 220 output token/s. A safe change
must preserve the retained grouped long-context verifier and should keep the
complete-round regression below 1%. Candidate-order QPN8 is not retained merely
to save tens of microseconds if dense-order or dense-selector scoring is more
stable.

## 2026-08-26 bounded multi-seed isolation result

The same physical V100 group 4--7 ran all five arms sequentially. The bounded
screen replays two GSM8K, two MATH-500, two HumanEval, and two embedded MBPP
rows at three predeclared request seeds, for 24 outputs per arm. Sampling is
temperature 1.0/top-p 0.95/top-k 20 with `xhigh` reasoning, natural EOS, and a
2,048-token cap. The embedded MBPP rows are useful only for relative trajectory
health: both target-only and every DFlash arm score 0/6 because this cap ends
reasoning before valid code or because the embedded prompt lacks the corrected
independent-test contract. They are not an absolute coding score.

| Arm | Valid score pattern | Total | Natural stop | Request acceptance | Mean steady decode |
| --- | --- | ---: | ---: | ---: | ---: |
| `T0` | GSM 5/6, MATH 3/6, HumanEval 6/6 | 14/24 | 15/24 | n/a | 75.246 tok/s |
| `D0` | GSM 6/6, MATH 3/6, HumanEval 6/6 | 15/24 | 16/24 | 4.64385 | 237.849 tok/s |
| `D1` | GSM 5/6, MATH 3/6, HumanEval 6/6 | 14/24 | 16/24 | 4.64085 | 244.086 tok/s |
| `D2` | GSM 5/6, MATH 3/6, HumanEval 6/6 | 14/24 | 16/24 | 4.62132 | 253.552 tok/s |
| `D3` | GSM 4/6, MATH 3/6, HumanEval 6/6 | 13/24 | 15/24 | 4.67840 | 259.833 tok/s |

`T0` and `D1` have exactly the same 24-row pass/fail set despite zero equal
token hashes. This is direct evidence that random trajectory divergence alone
is not a quality regression. `D0 -> D1` changes request-mean acceptance by only
`-0.002999` while improving mean steady decode by 2.62%; compact rejection is
retained. `D1 -> D2` exchanges one win in each direction, keeps aggregate score
unchanged, changes acceptance by `-0.019532`, and improves mean steady decode
by 3.88%; dense-order selector QPN8 remains the quality-first accelerated
candidate.

`D2 -> D3` yields one D2-only pass and no D3-only pass in this small screen,
reducing 14/24 to 13/24 for a 2.48% mean-decode gain. McNemar `p=1.0` and the
wide bootstrap interval do not prove a population regression, but they also do
not authorize the packaged candidate-order override. Combined with retained
real-hidden evidence, the quality/risk trade is unfavorable. The source now
ignores `VLLM_SM70_DFLASH2_QPN8_DENSE_ORDER=0` unless the separate explicit
benchmark-only `VLLM_SM70_DFLASH2_QPN8_ALLOW_CANDIDATE_ORDER=1` is present.
This converts stale production launch files to the D2 path while preserving
the main QPN8 speedup and leaves D3 available only for controlled research.

This screen loaded stale stable-ABI RMSNorm SHA `fe986a...9ba8c`; the runtime
capability guard correctly selected the bitwise per-layer fallback. Therefore
its score evidence is valid and all arms are comparable, but its throughput is
directional rather than a practical-endpoint claim. The formal API and speed
gates force repaired SHA `7689e5...d449`.

## Structured-output safety backport

The local scheduler previously derived the new reasoning window from stale
async placeholders, fed the whole boundary step to the grammar, advanced FSM
state through drafts produced before their masks existed, and allowed
XGrammar to consume tokens after termination. Those are state-machine bugs,
not model-sampling variation. The targeted backport now:

- uses the exact `new_token_ids` appended by the scheduler step;
- records the reasoning-end token and advances only the suffix after it;
- grammar-validates post-marker drafts before temporary bitmask-state advance;
- keeps the bonus row constrained after invalid `-1` draft padding;
- stops XGrammar acceptance/validation at termination and resets termination
  state explicitly.

The combined DFlash2, reasoning-structured-output, XGrammar termination, and
quality-comparator gate first passed `110` tests. The final CPU-only rerun,
which deliberately hides GPUs while another process owns them, passes `106`
and skips 12 explicit CUDA cases. The separate V100 compact-rejection gate
passes all eight temperature 0.6/1.0 cases. The two exact failure shapes
reported upstream pass without `Failed to advance FSM`.

The repaired-extension TP4 Graph API gate then ran with prefix caching, Mamba
align, `qwen3` reasoning, and `qwen3_coder` tools. At B1 and B4 it repeated
`json_object`, strict JSON schema, and required `get_weather` tool calls four
times each: both lanes pass `12/12`, all 24 requests finish successfully, and
the server log contains no FSM-advance, traceback, EngineCore error, HTTP 500,
or `FINISHED_ERROR` signature. The launch intentionally supplied the stale
`DENSE_ORDER=0` value; the runtime logged that it ignored it and enabled
`dense_order=True`, proving the production safety admission works. These short
prompts did not produce a prefix-cache hit. A second gate therefore uses the
actual 256K/current-Flash contract and alternates two long ALPHA/BETA prefixes.
All five requests return the exact per-request sentinel and checksum. Prefix
metrics record 59,396 queried tokens and 32,960 hit tokens, far above the
1,024-token admission threshold. The server log remains free of FSM, HTTP 500,
EngineCore, traceback, and `FINISHED_ERROR` signatures.

The official Qwen model card also distinguishes general thinking sampling from
precise coding: the latter recommends temperature 0.6 with top-p 0.95/top-k 20
and no presence penalty. Temperature 1.0 remains the general-thinking and
distribution-equivalence contract. Temperature 0.6 is evaluated below as a
separate client-visible precise-coding profile; sampling profiles must not mask
a verifier or grammar defect.

## Practical 16K coding gate

The quality-first D2 route completed MBPP-32, HumanEval-32, and stratified
LiveCodeBench-16 at three predeclared seed bases. Each request uses the
practical 256K/4096/prefix/Mamba-align/Graph engine contract, temperature
1.0/top-p 0.95/top-k 20, `xhigh` reasoning, and a 16K natural-EOS cap. MBPP
excludes the one upstream EvalPlus row without a valid independent-test
contract, leaving 31 scored rows per seed.

| Suite | Samples | Base/score | Plus | Natural stop | Aggregate output | Mean steady decode | Pooled/request acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MBPP | 96 (93 scored) | 89/93 | 80/93 | 95/96 | 213.539 tok/s | 236.902 tok/s | 4.061/4.318 |
| HumanEval | 96 | 94/96 | 92/96 | 91/96 | 208.978 tok/s | 245.645 tok/s | 3.972/4.476 |
| LiveCodeBench | 48 | 33/48 | n/a | 32/48 | 170.283 tok/s | 196.541 tok/s | 3.277/3.694 |

MBPP Base/Plus by seed are `31/29`, `27/24`, and `31/27` out of 31.
HumanEval Base/Plus are `31/30`, `31/31`, and `32/31` out of 32. The middle
MBPP seed demonstrates substantial sampling variance even though its
acceptance and speed are not worse; this is why a single output or acceptance
mean is not an intelligence metric.

The first seed exactly matches the historical no-DFlash request-seed contract.
Across MBPP plus HumanEval, both routes score Base `62/63` and Plus `59/63`.
The accelerated route gains one Base and two Plus rows on MBPP, while losing
one Base and two Plus rows on HumanEval. Thus the matched aggregate shows no
net score regression, although the changed sampled trajectory moves individual
failures. MBPP/HumanEval expose six 16K cases among 192 outputs, with three
still extractable and correct. LiveCodeBench adds 16 length cases among 48
outputs. The temperature-0.6 follow-up below improves executable score but not
natural-stop behavior, so it is retained as an optional precise-coding profile
rather than a forced API default.

LiveCodeBench scores `11/16` for every seed. Aggregated difficulty scores are
easy `18/18`, medium `12/18`, and hard `3/12`. The historical no-DFlash and
old-DFlash first-seed controls also score `11/16`; the current route exchanges
one passing task with each but adds one natural stop (`11/16` versus `10/16`).
Across all three current seeds, the first two have 11 stops and the third has
10. Every failure is a 16K length case; all stopped cases in the first two
seeds pass. This isolates excessive thinking as the remaining coding-product
risk, rather than verifier corruption or a score regression.

## Precise-coding sampling and PPL distribution gate

The same practical D2 engine contract was repeated at temperature 0.6 for the
predeclared middle seed, which was the weakest temperature-1.0 MBPP
realization. All other engine, dataset, prompt, seed, reasoning, and 16K
natural-EOS settings were unchanged.

| Suite | Temperature 1.0 | Temperature 0.6 | Natural stop, 1.0 -> 0.6 |
| --- | --- | --- | ---: |
| MBPP | Base 27/31, Plus 24/31 | Base 29/31, Plus 27/31 | 31/32 -> 30/32 |
| HumanEval | Base 31/32, Plus 31/32 | Base 31/32, Plus 31/32 | 30/32 -> 30/32 |
| LiveCodeBench | 11/16 | 11/16 | 11/16 -> 10/16 |

Across the 80 requests, temperature 0.6 changes request-mean acceptance
`4.27345 -> 4.51356`, pooled acceptance `3.70902 -> 3.84470`, mean steady
decode `233.187 -> 244.520 token/s`, and output-token/decode-time throughput
`195.817 -> 201.852 token/s`. It improves MBPP executable scores and lowers no
suite's aggregate score, but natural stops fall from `72/80` to `70/80`.
LiveCodeBench exchanges one pass in each direction: one temperature-1.0 stop
becomes a temperature-0.6 length failure while a different 16K length output
becomes executable and passes. This is sampling variation, not evidence that
temperature 0.6 fixes long-thinking termination. Therefore temperature 0.6 is
accepted as an optional precise-coding profile, not as a forced global API
default; the general-thinking profile remains temperature 1.0.

A separate current-source Graph PPL pair compares target-only and DFlash2 on
eight fixed 2,048-token WikiText segments. Both arms use the NVFP4 target,
E5M2 target KV, TP4 V100, 256K maximum context, 4096 chunking, prefix caching,
Mamba align, and the repaired extension set. The 16,376 scored prompt tokens
produce weighted target-only/DFlash2 PPL `5.4993116/5.4993622`, an absolute
change of `+0.0000506` (`+0.00092%`). Maximum per-segment PPL difference is
`0.0062143`; prompt-logprob mean/max absolute differences are
`0.0048521/0.481627`. The predeclared `0.01` maximum segment-PPL gate passes
with no failed evidence. The generic comparator labels the artifact
`B-pending` only because this prompt-PPL experiment intentionally does not
collect output-logprob or sampler-logit tensors; the scored generation,
compact-rejection equivalence, and API state gates cover those separate
behaviors. The aggregate PPL result rejects a distribution-level degradation
of the target model from enabling DFlash2.

## Current status

The source worktree is isolated. The five-arm screen, safe selector admission,
minimal structured-output backport, comparator, focused CPU/GPU tests, actual
structured API gate, three-seed practical coding gate, 256K long-prefix state
gate, temperature-0.6 repeat, and current Graph PPL pair are complete. The
benchmark supports index-derived request seeds so its first lane exactly
matches the historical target-only seed contract, and it records suite
metadata for official EvalPlus/LiveCodeBench scoring. Compact rejection
matches dense rejection at both temperature `1.0` and precise-coding
temperature `0.6`. The same-card bounded D1/D2 screen already attributes a
3.88% steady-decode gain to quality-safe dense-order selector QPN8; a larger
practical D1/D2 performance pair is optional follow-up rather than a quality
blocker. Development and Draft PR #18 are private; no public branch is used.
