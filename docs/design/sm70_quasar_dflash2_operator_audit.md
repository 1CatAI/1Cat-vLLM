# QUASAR + DFlash2 operator quality audit, 2026-09-06

This is the retained first-stage audit. The
[E4M3 KV and FP32-logit follow-up](sm70_quasar_e4m3_fp32_logits.md) addresses
the precision findings and records subsequent performance and quality checks.

This audit fixes two sampling-boundary defects and two defects in the
measurement path. It does **not** establish that the checkpoint, FP8 KV
cache, or approximate LM-head candidate search preserves unquantized model
quality. Keep the change in Draft: reference fallback has a measured latency
cost, and one of three selected MBPP Plus cases regressed at the fixed seed.

## Frozen contract

- Integration: `onecat/main`, base
  `755baae1d075ee04fa9096b23fc0225b23589a86`.
- Target: `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4`, revision
  `d8e6fbfa3e3a78899b440222b827430045a05b44`.
- Draft revision: `dedf8df68adfb1afeaf7b7480c0a0243108177b4`;
  checkpoint SHA256:
  `67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c`.
- Four V100-SXM2-32GB GPUs, TP4, FP16 activations, NVFP4 checkpoint weights
  executed through TurboMind W4A16/QPN2; this is not native NVFP4 arithmetic.
  Python 3.12.13, Torch 2.10.0+cu128, CUDA 12.8, Triton 3.6.0.
- V2 runner, `FLASH_ATTN_V100` for both models, target E5M2 KV, draft FP16
  KV, seven probabilistic speculative tokens, QPN8 top-64 candidate search
  followed by FP16 rerank with dense-vocabulary ordering.
- Maximum length 262144, batch-token budget 4096, maximum sequences 4,
  GPU memory fraction 0.8, prefix caching enabled, Mamba cache mode
  `align`, FULL_AND_PIECEWISE graphs. The audited verifier has eight rows.
- Sampling: temperature 1, top-k 20, top-p 0.95, natural EOS. Speed requests
  use seed 20260925 and maximum output 1024. The three diagnostic MBPP cases
  use seed 0 and maximum output 16384. Thinking is enabled, effort xhigh.

Worker logs confirm Flash-V100 grouped q8 verification, XQA paged decode,
FlashQLA, TurboMind QPN2 and QPN8 rerank dispatch. Launch scripts and queue
manifests retain the complete environment. Speed runs contain no capture
hooks. Native binaries were reused; this change needs no native rebuild.
An attempted final live-worker library snapshot occurred after shutdown and
captured zero workers; it is not evidence of loaded-library identity.
Resolved native-file hashes and the original baseline loaded-library
manifest are retained separately.

## Reference and evidence rules

The C2 capture contains 64 target layers, all four TP ranks, and verification
steps 0, 7 and 31 of one real request: 1131 saved tensors per rank and step.
References use the **same saved input, logical TP weight shard, position,
physical cache pages, gate, and state selection**. They do not compare
unrelated generations. Linear references dequantize the same checkpoint
into FP32; they do not measure the loss from the original unquantized model.

Relative L2 is `||actual - reference||2 / ||reference||2`. Tables show the
maximum across cases, not an average and not a single shared worst case.
Residual and transport checks use their staged dtype contracts; matrix,
attention and recurrence references otherwise use FP32 arithmetic.
Distribution error uses total variation `sum(abs(p-q))/2`.

Small nonzero errors are measurements, not automatic acceptance. In
particular, a representation or reduction-order difference requires its own
propagation evidence before promotion.

## Confirmed defects and changes

1. **Compact top-20 loses the full-vocabulary tie contract.** PyTorch top-k
   masking retains all entries equal to the kth threshold, potentially more
   than 20. A fixed list of 20 cannot represent that distribution. A top-p
   boundary can also split equal logits in a different vocabulary order.
   Request a 21st reranked candidate, detect kth ties, split nucleus ties and
   near-boundary CDF rounding, then use existing full-vocabulary verification
   for an ambiguous block. Keep separate contiguous buffers for top-16,
   top-20 and top-21. The check is outside model graph capture.
2. **Dense Triton masking mishandles ambiguous pivots.** On a compiled V100
   fixture with two high logits and 18 tied logits, top-p 0.95 retained 20
   candidates where PyTorch retained 19; distribution TV was 0.04266838.
   The kernel now flags ambiguous rows and the public wrapper remasks those
   rows using the untouched full-vocabulary PyTorch reference. Direct graph
   capture uses that reference without a host decision inside capture.
   The internal pivot algorithm is not claimed to have been replaced.
3. **Layer-dump custom-op output aliased its input against its schema.**
   `torch.library.opcheck(test_schema)` rejected the old implementation.
   Its AOT buffer reuse produced false residual errors at layers 6, 34 and
   62 in the first capture. Return owned storage, including the fake
   implementation. This only repairs enabled diagnostics; it is not evidence
   that ordinary serving previously had those residual errors.
4. **GDN QKV oracle sharded the concatenation incorrectly.** The checkpoint
   concatenates logical Q/K/V. Shard each logical segment before
   concatenation, matching the runtime loader, and test all four ranks.

The compact guard adds one host decision and copies only eight by 21 values.
An exact-shape microbenchmark reduced guard median time from 255.85 us for
multiple GPU operations plus a host flag to 62.33 us for one copy and CPU
calculation, with zero decision disagreements on 300 random checks. This
does not remove the cost of full-vocabulary fallback.

## Target operators on real inputs

| Operator | Cases | Maximum absolute error | Maximum relative L2 |
|---|---:|---:|---:|
| Attention QKV projection | 192 | 0.02303 | 5.482e-4 |
| Attention output projection | 192 | 0.02396 | 5.854e-4 |
| GDN QKV/Z/B/A projection | 576 | 0.02952 | 5.802e-4 |
| GDN output projection | 576 | 0.01605 | 7.005e-4 |
| MLP gate/up with fused SiLU | 768 | 0.09627 | 1.283e-3 |
| MLP down projection | 768 | 0.03246 | 5.977e-4 |
| Input RMSNorm, staged reference | 768 | 0.001953 | 1.184e-5 |
| Post-attention RMSNorm, staged reference | 768 | 0.001953 | 1.333e-5 |
| Residual addition, staged reference | 768 | 0 | 0 |
| Q normalization and partial RoPE | 192 | 0.008986 | 2.458e-4 |
| K normalization and partial RoPE | 192 | 0.007598 | 3.032e-4 |
| Attention on identical cached KV | 192 | 0.03008 | 3.042e-4 |
| Attention sigmoid gate | 192 | 0.0002441 | 2.130e-4 |
| Attention gated multiply | 192 | 0.007690 | 2.461e-4 |
| KV key encoder vs E5M2 roundtrip | 192 | 0 | 0 |
| KV value encoder vs E5M2 roundtrip | 192 | 0 | 0 |
| E5M2 key representation vs original FP16 K | 192 | 0.9922 | 5.840e-2 |
| E5M2 value representation vs original FP16 V | 192 | 3.969 | 5.932e-2 |
| FP16 LM head vs FP32 same-weight arithmetic | 12 shards | 0.007971 | 2.085e-4 |
| TP reduction vs FP32 sum | 384 | 0.03125 | 2.736e-4 |

All 7116 target reference cases are finite. All 384 TP reductions exactly
equal FP32 summation followed by FP16 rounding, and agree across ranks.

The KV encoder is exact under its E5M2 contract. The roughly 5.9% conversion
loss belongs to that precision choice. A synthetic attention comparison
against original FP16 KV gives 4.34–5.34% relative output error, substantially
larger than attention arithmetic on the same cached KV. This does not prove
that KV precision caused the user's observed text-quality regression.

The LM-head example illustrates why scalar tolerances and dataset scores
are insufficient: all 24 real rows preserve top-1, but one row changes its
top-p support against FP32 arithmetic. Maximum full-softmax TV is 0.002119;
after top-20/top-p it is **0.021532**. Separately, indexed FP16 rerank differs
from dense FP16 logits by 0.015625 on four of those rows, while all captured
top-20 ID sets agree. Neither FP32 references nor the tie guard establish
global coverage of the approximate QPN8 top-64 candidate search.

## State, boundary and draft operators

| Operator or boundary | Evidence | Result |
|---|---|---|
| GDN QKV packing | strided M=1/8/137 | Bitwise equal |
| GDN causal convolution + SiLU | all 48 layer weights, TP0 real q8 input, synthetic history; accepted selectors 1/8 | 96 cases; max relative L2 3.384e-4; whole updated state exact |
| GDN gated RMS | 48 real layer weights and real Z; synthetic core input | max relative L2 2.224e-4 |
| FlashQLA prefill recurrence | T=8/64/256/1024, FP32 state, independent sequential reference | output max abs 7.624e-6; state 1.192e-7 |
| FlashQLA fused decode | two seeded inputs, untouched-slot check | output max abs 1.763e-6; state 2.235e-8 |
| Packed q8 GDN verification | 64 graph replays, permuted state slots, accepted selectors 1–8; same precomputed gates and production matching flags | Output and whole state bitwise equal to split recurrent reference on every replay |
| Target grouped attention | q8, Hq6/Hkv1/D256, randomized physical pages; contexts 1032/32768/131072/262144 | same-cache FP32 relative L2 3.18–3.34e-4; XQA comparison is not bitwise equal |
| Draft noncausal SWA | q8, Hq8/Hkv2/D128, window 2047 each side, contexts 1024/2048/2055/4097/32768 | max relative L2 about 2.88e-4 |
| Draft FP16 projections | 120 real-input layer/rank cases | max relative L2 3.713e-4; max abs 1.849 on large intermediate values |
| Draft BF16 SwiGLU transport | 20 real-input cases | Captured input, rerun kernel and staged reference exact; row scales exact |
| Draft grouped convolution | 20 real-weight/input cases, both sides | max relative L2 6.11e-4 |
| Draft BF16 RMS/residual | 10 real-weight cases, two magnitudes | Exact, including residual magnitudes beyond FP16 finite range |
| Draft per-layer context K RMS | all five layer weights | max relative L2 2.065e-4 |
| Draft selector edge scores | real codebooks, candidate fixture, synthetic hidden rows | max relative L2 1.693e-6 |

The 1024 initial real-weight M8 projection probes had exact repeated eager
and graph results. A separate 32-case dynamic-input replay check is exact.
These are not 1024 real-input or production-aligned GDN measurements:
the first GDN oracle had the sharding defect described above.

Real-logit sampling replay covers 24 rows at top-p 1/0.95/0.6, isolating
sampling from LM-head arithmetic. Protected masks match the reference in
all 72 row cases and measured distribution TV is zero. The kth tie in this
capture carried negligible mass, so the large distribution defect is
established by the compiled adversarial fixtures, not overstated as a
large observed error on these 24 real rows.

This is coverage of the principal q8 verifier and draft operator families,
not exhaustive acceptance of every prefill shape, embedding/context-FC
path, scheduler transition, sampling penalty, arbitrary RNG trajectory,
or checkpoint quantization loss. Those unmeasured contracts remain open.

## Validation and cost

Focused GPU validation:

- Oracle and layer-dump tests: 6 passed.
- New tied-cutoff tests plus existing top-k/top-p suite:
  112 passed, 35 platform/configuration skips.
- Existing DFlash2 rerank/compact-top-k regression selection: 22 passed,
  144 deselected.
- After reducing guard overhead: 14 tied-cutoff tests passed again.
- Focused Ruff checks/format and `git diff --check` pass.

Example commands from the owned checkout:

```bash
python -m pytest -q tests/benchmarks/test_sm70_quasar_nvfp4_oracle.py tests/kernels/core/test_sm70_qwen_layer_dump.py
python -m pytest -q tests/v1/sample/test_topk_topp_tied_cutoffs.py tests/v1/sample/test_topk_topp_sampler.py
python -m pytest -q tests/v1/spec_decode/test_dflash2.py -k 'rerank or compact_topk'
```

Unprofiled, one warmup then three measured repetitions; the following uses
the median consistently. Complete-round cost is engine pure-decode time
divided by speculative rounds, **not target-verifier GPU time**.

| Request | Baseline round ms | Final round ms | Baseline / final decode tok/s | Baseline / final TTFT s | Emitted tokens / rounds |
|---|---:|---:|---:|---:|---:|
| Release 1K, actual prompt 1019 | 18.037 | 20.781 | 152.826 / 132.646 | 0.3522 / 0.3464 | 318 / 115 |
| MBPP 28, actual prompt 135 | 17.588 | 19.941 | 268.541 / 236.857 | 0.1051 / 0.1004 | 308 / 65 |

Median round cost increases about 15.2% and 13.4%. The initial GPU-operation
guard was slower still: 21.037/20.068 ms. Request token counts and acceptance
counts are unchanged on these two speed fixtures. No end-to-end speedup or
restoration of the 17.6 ms release baseline is claimed.

Three selected diagnostic MBPP cases all stop naturally: Base 3/3, Plus
0/3. The old DFlash baseline scored Plus 1/3 on these same cases; target-only
scored Plus 0/3. A real JSON-object request returns valid JSON with result
42. This small smoke cannot justify a model-quality improvement.

## Retained artifacts and rejected evidence

Artifact bundle: `v100-quasar-dflash2-operator-audit-20260906`.
The local Chinese audit report records its absolute location and the owned
worktree. Raw tensors, model paths and compiler caches are not checked in.

- `captures-v2/`: C2 tensors, norms, route inventory, cache pages, full and
  compact logits, per-shard FP32 head references.
- `results/real-v2-rank{0,1,2,3}.json`: per-layer/operator metrics;
  `operator-summary.json` provides the aggregate.
- `math-scan.json`, `draft-math.json`, `gdn-q8-state.json`,
  `gdn-conv-norm.json`, `tp-reduction.json`,
  `lm-head-propagation.json`: independent operator and state evidence.
- `sampler-before.json`, `sampler-after.json`,
  `sampler-real-final.json`: pre-fix and protected distribution checks.
  The after file intentionally still reports the unguarded compact helper,
  alongside the repaired dense public operator.
- `final-speed-*.json`, `fixed-quality-subset.json`,
  `fixed-evalplus-subset.json`, `fixed-json.json`: service evidence.
- `scripts/`, `queue/`, `logs/`, `contract.json`,
  `final-native-files.json`, `cleanup-final.json`: exact commands,
  environments, negative attempts, binary hashes, and cleanup.

Do not reuse C1 numerical conclusions: it was affected by the diagnostic
alias bug. Do not reuse the original FP16 multiply/divide restore probe as
graph corruption evidence: that transform was not lossless. The initial
uniform-logit masking test failed and was fixed; the first convolution/norm
probe failed because it instantiated a CustomOp without a vLLM config and
was corrected before producing the final 144 cases. An occupied test port
was avoided without interrupting its owner.

Task-owned services and GPU leases have been released; the unrelated
service on port 8000 remained healthy. Before promotion, resolve the
fallback performance cost and investigate precision propagation at the
LM head and KV boundary with matched inputs. Do not replace those checks
with a larger aggregate-score run.
