# Qwen3.8 NVFP4 allocation-order quality repair

This integration reuses the original implementation from
[#494](https://github.com/1CatAI/1Cat-vLLM/pull/494), commit
`5fa8a605dab12cc9ee15459d9ac6b88d95c7be3a`. It does not propose a competing page
planner repair. Its separate scope is NVFP4 validation on the frozen
[#507](https://github.com/1CatAI/1Cat-vLLM/pull/507) single-request lane, whose
earlier recorded throughput was 98.965175 tokens/s. That speed record alone
did not establish output-quality acceptance.

## Frozen subject

- Public integration base: `755baae1d075ee04fa9096b23fc0225b23589a86`.
- Model-tested source: `d2c8401c220849b18ae5b858798305af5d904d42`;
  `bbdf0af5c999fa14e2167620921b1173c4ccad76` has identical production source.
- Qwen3.8-Flash-Next-NVFP4, TP4/PP1, four V100-SXM2-32GB GPUs; no MTP,
  prefix caching disabled, 262144 context, 8192 chunk, max sequences 1.
- FP16 activations and attention KV; checkpoint-native NVFP4 expert weights
  through TurboMind W4A16. No online QPN8 or approximate LM-head.
- V2 runner; dynamic prefill and full static decode graphs. PLE uses mmap
  during prefill and pinned-UVA during decode.
- Torch 2.10.0+cu128, CUDA compiler 12.0.140, driver 580.173.02.
  Only Flash-V100 was rebuilt in an isolated directory; no dependency upgrade.

The frozen speed launcher explicitly used FP16 recurrent SSM state although
the checkpoint specifies FP32. The matched allocation intervention deliberately
preserves that setting to avoid confounding two changes. It is **not** the
native-state quality contract. Normal validation must use
`mamba_ssm_cache_dtype=auto` (resolves to FP32 for this checkpoint), or explicit
`float32`; do not inherit the frozen launcher's FP16 override.

## Root cause and actual-input proof

The original audit observed identical Layer3 prefill hidden states, Q/K/V,
gates, selected logical token IDs and positions in all four TP ranks. Physical
KV block allocation differed. Grouped page4 planning emitted physical hash
order, changing the online-softmax/reduction order despite identical logical
attention. The observed final token fork was at generated index 8, before
natural EOS; the EOS and `This` candidate logits exchanged order. The full
vocabulary logit delta there reached 3.125, not merely a near tie.

`benchmarks/kernels/verify_sm70_qsa_nvfp4_relocation.py` reconstructs the
**actual** captured Layer3 KV and runs the following controls in one process:

1. Frozen planner and arithmetic must reproduce both saved model outputs
   bitwise. This establishes that the replay represents the observed failure.
2. Rebuilt attention with the frozen plan must preserve frozen arithmetic.
3. Replace only the planner while retaining frozen attention arithmetic.
4. Run the complete rebuilt extension and compare it to step 3.
5. Restore the frozen planner and require the old outputs to return exactly.

All controls passed on all four rank captures (8192 queries per rank,
6 query heads, 1 KV head, head dimension 256, interleaved 400-token pages).

| Rank | Old allocation-dependent output elements / 12582912 | Old max abs delta | Fixed differing elements |
|---:|---:|---:|---:|
| 0 | 632957 | 0.001953125 | 0 |
| 1 | 512203 | 0.0009765625 | 0 |
| 2 | 752346 | 0.001953125 | 0 |
| 3 | 615824 | 0.0009765625 | 0 |

1023 of 1024 groups changed logical page order in the old planner. Logical
sorting removed the allocation dependence. Arithmetic-rebuild comparisons
and native reversals each had zero differing output bits in all ranks.
This is an actual-input **operator** causal proof, not by itself a full-model
or universal no-token-flip certificate.

The extension SHA256 values for these controls were:

- Frozen: `a3684c1b379c6992a4e85e2e3474c54c1a832d908e8c5c331e1b3cf77cd641ed`.
- Repaired: `fb2f38915825d8c9c56e0fe8732162d27b6c3e7a867bf54a6e7ed0b90706eaf0`.

## Focused regression

The extra regression uses actual 400/784-token page geometry (FP16/native-FP32
SSM contracts), contiguous and interleaved strides, contexts 8192/32768/262144
and graph replay with physical
layout A/B/A. Every emitted page and mask is checked against a CPU logical
reference. Combined with the existing page4 and QSA-ops suites:

```bash
.venv/bin/python -m pytest -q \
  tests/kernels/test_sm70_qsa_page4_plan.py \
  tests/models/qwen4_exp/test_qsa_ops.py
```

Result: **53 passed**, no skips, in 17.62 seconds on V100 GPU0.
The 262144 cases here are planner tests, not 256K model-quality measurements.
After observing the native-state784-token geometry, the six added784 cases
passed separately in3.13 seconds (`-k 'hybrid_page and 784'`): **59 distinct
passed cases** across the two targeted invocations.

For retained actual-input captures, replay with explicitly selected extensions:

```bash
.venv/bin/python benchmarks/kernels/verify_sm70_qsa_nvfp4_relocation.py \
  --capture-dir /path/to/per-rank-captures \
  --frozen-dso /path/to/frozen/flash_attn_v100_cuda.so \
  --out /path/to/results/real_replay.json
```

The repaired Python package and rebuilt extension must be first on the runtime
import path. Updating Python source while loading the old CUDA extension does
not deploy the grouped-planner repair. Keep private paths and raw tensors out
of Git; retain them in the task's local artifact directory.

## Model gate and remaining scope

The completed one-instance FP16-state control used GPU0–3, the original speed
baseline's GPU group. It switched only the grouped planner in all 12 QSA layers
(12 confirmed planner calls per rank per 8192-token request):

| Intervention | Repetitions | First EOS, zero-based | Full 513-token hashes |
|---|---:|---|---|
| Frozen planner | 3 | 8, 8, 28 | 3 different |
| Fixed planner | 3 | 28, 28, 28 | Identical |
| Restore frozen planner | 2 | 28, 8 | Old allocation dependence returns |
| Restore fixed planner | 3 | 28, 28, 28 | Identical to first fixed group |

All six fixed requests had the same 513-token hash:
`603abc75ff871d582a4ec032c5f7c388207447ee3f4bffb17c8beb5e022190fc`.
Their actual full-vocabulary logits and final hidden states were also bitwise
equal at every observed position (first 12 generated positions). Across all
132 observed positions including controls, logits were finite, all four ranks
agreed on final hidden, and the actual selected token matched the concatenated
full-vocabulary argmax. Comparisons stop at the first differing input prefix;
later divergent trajectories are not treated as equivalent-input comparisons.

The same-instrumentation intervention and reversal close the causal link from
physical-page planning to this actual NVFP4 token-fork reproduction. This is
not a claim that arbitrary inputs or alternative batch shapes cannot diverge.

Two official-sampling natural-EOS checks (arithmetic and exact record copy)
passed before and after the intervention. After fully detaching observers,
three more 513-token runs retained the fixed token hash. Their total request
times were 6.35348 / 6.35953 / 6.35402 seconds. Request-level statistics had not
been enabled for this diagnostic instance, so these **include prefill** and
must not be reported as a pure-decode comparison against 98.965 tokens/s.
This missing-metrics limitation is retained explicitly; the native-state run
enables statistics and rejects missing separated metrics.

### Native FP32-state validation and separated performance

A second model instance used `mamba_ssm_cache_dtype=auto`. All four workers
confirmed **float32** recurrent state; FP16 activations, short-convolution state
and attention KV remained unchanged. The runtime consequently selected
784-token attention pages instead of 400. All other production optimization
gates and sidecars remained the frozen ones. Request statistics were enabled.

- Four 8192+513 deterministic requests had identical complete token sequences;
  all48 observed positions had bitwise equal complete logits and final hidden
  across repetitions. Finite/argmax/TP-replica checks passed.
- All11 official-sampling natural-EOS requests
  passed: arithmetic and copy before/after, three additional Chinese/Python/set
  questions, and dispersed retrieval at8191/8192/8193/261632 input tokens.
  The first two checks preceded observation installation; the rest followed
  detachment. These are nine distinct cases, not a broad dataset evaluation.
- The261632-input case placed three unique records at token offsets13079,
  130797 and248514 (approximately5%,50%,95% of context). All three codes were
  recovered correctly; the answer stopped naturally after217 generated tokens.
  The answer depends on the interior records, not on the final question alone.
- Following the long request and a shape warmup, three unobserved8192+513
  requests again matched the earlier native-state token hash. This also checks
  this particular long-to-short request reuse transition.

| Measurement | Run1 | Run2 | Run3 | Mean |
|---|---:|---:|---:|---:|
| Pure decode, tokens/s | 97.90434 | 97.90695 | 97.90411 | **97.90513** |
| TPOT, ms/token | 10.21405 | 10.21378 | 10.21408 | **10.21397** |
| Prefill, tokens/s (8192 input) | 6965.58 | 6976.74 | 6970.63 | **6970.98** |

Decode is512 tokens divided by last-token minus first-token time; prefill is
8192 divided by scheduled-to-first-token time. Do not substitute the engine's
periodic aggregate throughput log. This is a warmed repeated benchmark prompt,
not a universal prefill number for arbitrary content or261632-token inputs.
The new native-state pure-decode result is1.07% below the historical98.965175
FP16-state record. That comparison changes state precision and planner together;
it does not isolate either change's cost or claim a contemporary A/B timing.

With the fixed planner, native FP32 versus the old FP16 state override gave
bitwise identical first-prefill logits. Subsequent common-prefix decode logits
differed (maximum2.3193359375 across the first12 observed positions), despite
unchanged sampled tokens there. The first token-sequence difference was at37,
**after both natural EOS positions at28**. This is not evidence of a natural
answer failure, nor does it justify treating the lower-state-precision override
as numerically equivalent. Keep native FP32 state for this quality contract.

The two bounded model loads ended normally and released their GPU workers;
no API service or model residency remains. Current evidence closes the
reproduced allocation-induced token-fork defect and these bounded native-state
quality gates. Existing GDN gate-reduction and W13 split-order differences still
need independent actual-input references and propagated-ranking evidence;
this repair does not certify all kernels or arbitrary inputs as error-free.
Human review and wider quality acceptance remain required before promotion.

## Main integration, 2026-09-06

The user requested merging this PR after the bounded results and remaining
numerical-audit scope were reported. PR #494 is already on public main.
PR #525 now targets main directly and retains the tested #506/#507 ancestor
commits, avoiding a merge that would stop in an intermediate feature branch.
The synchronized main base is `4366d9d5fe80eeaf79575b51ec36a6a032673df0`.

The merge preserves the tested QSA, HC and router Python source, Flash-V100
kernel source and page400/page784 regressions unchanged. It retains main's
independently merged AWQ/PP and model-load allocator cleanup changes. CI fixes
only format C++/Markdown, spell out an existing commit SHA to avoid a typo
false positive, and use `torch.accelerator` device/synchronization/cache APIs
in three standalone verification scripts. No runtime arithmetic is modified.

The synchronized source passes27 CPU dispatch tests covering HC norm prefetch,
QSA resolved-address admission/fallback, and router packed-key admission.
Targeted pre-commit checks pass. No model was restarted for this integration;
the97.90513tokens/s result remains attributed to the earlier tested source and
binary contract, not a fresh benchmark of the updated main tree. The broader
GDN/W13 numerical-reference work remains open after merge.
