# QUASAR QAT: TP2/TP4 arithmetic and distribution audit

## Findings

TP size changes both matrix partitions and the current SM70 execution route.
The audit found and repaired two TP2 defects, then demonstrated remaining
sampling-token changes on identical prefixes after those repairs. Greedy top-1
agreement does not establish distribution agreement.

1. **NVFP4 output alignment:** the TP2 GDN input projection has logical N=8240.
   It satisfies the old 16-column adapter alignment but corrupts the native
   packed TurboMind result. Padding to N=8256 (32 columns) restores accuracy.
   Eight real shards across layers 0, 1, 32 and 62 had relative L2
   **0.326777–0.512499 before, 0.000249–0.000323 after**. TP4's existing
   4120→4128 physical layout is unchanged.
2. **FP32 LM-head admission:** the precision flag previously took effect only
   while preparing the TP4 QPN8 candidate layout. TP2 fell back to FP16 dense
   logits even when the flag was set. The explicit FP32 flag now independently
   admits the pinned vocabulary/hidden shape on TP2 and TP4. QPN8 admission
   remains TP4-only; no global precision default changes.
3. **Remaining TP drift:** after fixing both defects, two same-prefix probes
   retained all 258 greedy top-1 IDs, but target sampling distributions reached
   **1.029% / 2.208% maximum TV**. Production token-keyed Gumbel noise produced
   **21 differences in 16,512 paired draws**, including one with seed 0.
4. **QAT semantics:** the checkpoint describes W4A4, while the SM70 route uses
   W4A16. Applying its activation quantizer on the same captured inputs changes
   individual projection outputs by up to **12.24% relative L2**. Changing only
   the last MLP down projection to that activation contract produces up to
   **5.042% sampling TV** downstream. These are execution differences, not
   percentages of text-quality loss or proof that W4A4 is better on this host.

The audit does not establish overall QAT quality relative to its BF16 teacher:
that exact teacher checkpoint is unavailable locally. The operator reference
decodes the *QAT checkpoint's* weights; it is not an unquantized-model oracle.

## Frozen inputs and source

- Integration: `onecat/main`, `755baae1d075ee04fa9096b23fc0225b23589a86`.
- Starting precision/performance branch: `6f07be1cce338b87e7e8d1a714bf9e30e5424fd8`.
- Production-code repairs: `fcb6dada58` in owned Draft PR #517.
- QUASAR checkpoint revision: `d8e6fbfa3e3a78899b440222b827430045a05b44`.
- DFlash2 revision: `dedf8df68adfb1afeaf7b7480c0a0243108177b4`.
- V100-SXM2-32GB; TP4 uses GPUs 0–3, TP2 uses GPUs 0–1. CUDA 12.8,
  Torch 2.10.0+cu128, Python 3.12.13, Triton 3.6.0, FP16 activations.
- Target/draft FLASH_ATTN_V100, target E4M3 KV, draft FP16 KV; V2 runner;
  draft7/q8; full/piecewise target and full draft graphs; context pipeline on.
- Max length 262144, batch tokens 4096, capacity four requests, one live
  request, memory utilization 0.8, prefix cache and Mamba align enabled.

The matrix audit reuses alias-corrected C2 captures at step 7 from the preceding
operator audit. Their trajectories were generated with E5M2, but the exact same
input vector is supplied to each compared TP partition. Column projections
use rank-zero input on every shard; row inputs concatenate all four logical
shards. Logical Q/K/V and gate/up segments are sharded independently, and
packed checkpoint bytes are checked against the full projection.

The end-to-end probes use fresh E4M3 services. Token tapes come from the prior
natural-stop production MBPP28 and MBPP3 outputs. Each comparison covers the
prefill's last position plus 16 q8 verification groups: 129 rows per input.
Every compared position, token ID and group boundary is checked. The runner
forces the tape and accepts its query rows, preventing an early sample change
from feeding different prefixes to the two models. These requests are capped
diagnostics, not free generation, quality scores or performance benchmarks.

## Matrix dimensions and execution

Dimensions below are local logical `(N, K)` for `X[M,K] @ W[N,K].T`, with M=8
for the decode operator comparison.

| Projection | TP2 `(N,K)` | TP4 `(N,K)` | Partition |
| --- | --- | --- | --- |
| GDN qkvzba | 8240, 5120 | 4120, 5120 | Output rows, logical segments |
| Attention qkv | 7168, 5120 | 3584, 5120 | Output rows, logical segments |
| MLP gate/up | 17408, 5120 | 8704, 5120 | Output rows, gate/up separately |
| GDN/attention output | 5120, 3072 | 5120, 1536 | Input columns and sum |
| MLP down | 5120, 8704 | 5120, 4352 | Input columns and sum |
| LM-head | 124160, 5120 | 62080, 5120 | Vocabulary rows |

TP4's admitted q8 target projections use QPN2, with fixed split-K/accumulator
settings. TP2 uses TurboMind. QPN2 internally uses FP32 HMMA accumulators and
FP32 partial reduction, then writes FP16. Consequently, row-parallel layers
round each local partial before the cross-rank sum. An accurate collective
cannot recover those discarded bits.

The comparison therefore includes both the production route difference
(TP2 TurboMind versus TP4 QPN2) and an isolated partition test using the same
kernel family on both sides. Direct QPN2 calls on TP2 are counterfactual
operator probes; this change does not enable QPN2 for TP2 production.

## Operator results

All 64 layers were inspected: 256 fused projection invocations, covering the
checkpoint's 496 logical linear layers. The table reports maximum relative L2
between the two TP assemblies on identical input vectors, after alignment repair.

| Projection | Cases | Same TurboMind family | Same QPN2 configuration rule | TP2 TM vs TP4 QPN2 |
| --- | ---: | ---: | ---: | ---: |
| GDN qkvzba | 48 | 0.00323% | 0 | 0.06678% |
| Attention qkv | 16 | 0.00242% | 0 | 0.06513% |
| MLP gate/up, before activation | 64 | 0.00182% | 0 | 0.06416% |
| GDN output | 48 | 0.04404% | 0.04775% | 0.08108% |
| Attention output | 16 | 0.04654% | 0.04564% | 0.07763% |
| MLP down | 64 | 0.04423% | 0.04420% | 0.07423% |

The fused gate/SILU/up output reaches 0.11928% between production routes.
Changing only the rounding of exact FP32 local partials already causes up to
0.05107% TP2/TP4 difference in MLP down. Thus there is an arithmetic mechanism
independent of a broken collective or changed checkpoint shards.

Real partial tensors from three row projections, two kernel routes and both
TP sizes were also passed through native captured collectives. All six cases
per rank matched FP32 summation followed by FP16 bitwise. This validates the
assembly reference on those inputs; it does not prove every collective shape.

Older C2 replicated norm inputs contained a few rank-dependent rounded values.
The matrix oracle explicitly fixes those inputs instead of comparing different
vectors. In the fresh end-to-end probes, replicated final hidden states are
bitwise identical among ranks within each TP configuration on all compared rows.

## Distribution and sampled-token changes

Both end-to-end services use the repaired FP32 dense LM-head for observation.
An independent FP32 dot-product oracle is also computed per vocabulary shard.
The head-only comparison evaluates the old FP16 path on the *same* hidden states.

| Check | MBPP28, 129 positions | MBPP3, 129 positions |
| --- | ---: | ---: |
| Final hidden relative L2, TP2 vs TP4 | 0.9576% | 0.9778% |
| Greedy top-1 flips, TP2 vs TP4 | 0 | 0 |
| Changed top-p support, TP2 vs TP4 | 0 | 1 |
| Maximum sampling TV, FP32 oracle TP2 vs TP4 | 1.0291% | 2.2077% |
| Old TP2 FP16 head vs same-hidden FP32: maximum TV | 2.5391% | 1.4526% |
| Repaired TP2 head vs same-hidden FP32: maximum TV | 0.001708% | 0.002081% |
| Changed head-only top-p supports, old → repaired | 1 → 0 | 1 → 0 |

The first decode group's layer-zero input matches exactly between TP sizes.
Its MLP output already differs by about 0.16%; differences then propagate
through attention, recurrent state, norms and subsequent projections. The
final-state difference is not attributable solely to one matrix or solely to
TP count: production kernel families and shapes both differ.

Sampling uses T=1, k=20 and p=0.95 with seeds fixed in advance to 0–63. The
production `gumbel_noised_argmax` primitive keys noise by seed, absolute
position and token ID. Sparse support enumeration is checked against the
full-vocabulary `gumbel_sample` at seed 0 before counting flips.

| Paired draws | MBPP28 / 8256 | MBPP3 / 8256 |
| --- | ---: | ---: |
| TP2 vs TP4 FP32 distributions: different sampled IDs | 10 | 11 |
| Old TP2 FP16 head vs FP32 oracle | 5 | 5 |
| Repaired TP2 FP32 head vs FP32 oracle | 0 | 0 |

For example, at MBPP28 position 167 and seed 0, TP2 and TP4 choose token IDs
13 and 1132 respectively under the same target-sampling noise. This is a
reproduced sampling flip with unchanged greedy top-1. It is not a full
DFlash rejection/acceptance trace or a claim about free-running answer quality.

## QAT execution contract

The pinned model card and `quantization_config` declare W4A4 NVFP4 with
16-element activation groups, E4M3 group scales and a checkpoint global scale.
The accepted SM70 TurboMind/QPN2 route retains quantized weights but consumes
FP16 activations; it does not apply that activation quantizer.

The audit applies the repository's NVFP4 activation quantize/dequantize
reference using the stored input global divisor, then multiplies by the same
FP32-dequantized QAT weights. Maximum output differences are 6.88% for GDN
input, 8.90% for GDN output, 5.96% for attention qkv, 12.24% for attention
output, 9.11% for MLP gate/up and 9.01% for MLP down.

To trace this through logits without replacing every upstream state, a second
probe replaces only layer 63's MLP down result on the same captured input and
residual, then runs the final norm and FP32 LM-head. Across 24 rows at three
captured steps, top-1 stays unchanged, one top-p support changes, and maximum
TV reaches 5.042%. By contrast, changing only that projection between repaired
TP2 TurboMind and TP4 QPN2 reaches 0.04112% maximum TV in these probes.

This evidence supports auditing the QAT activation contract separately from
ordinary GEMM rounding. It does **not** justify automatically enabling costly
W4A4 emulation, assuming higher activation precision always improves this
trained model, or quoting the model card's native-FP4 scores for the V100 route.
No new BF16-teacher comparison or end-to-end W4A4 quality score was obtained.

## Validation, reproducibility and limits

- Native padding and adapter regressions: **19 passed**, including N=8240,
  M=1/8/17 and changed-input CUDA Graph replay.
- FP32 head admission: **6 passed**. Logical GDN TP2/TP4 sharding: **6 passed**.
- The tracked matrix-audit CLI ran successfully against real layer-zero data;
  full 64-layer results were produced by its retained task-script precursor.
- Ruff, mypy and applicable commit hooks pass for the fixes.
- No native source or native binary changed. Existing TP4 physical widths,
  QPN2 accumulation and FP32 head execution stay the same. The preceding
  18.892/18.435 ms result is historical evidence; no new latency claim is made.

```bash
PYTHONPATH=. .venv/bin/python benchmarks/kernels/benchmark_sm70_quasar_tp_quality.py \
  --model <pinned-quasar-directory> --capture-dir <alias-corrected-c2-captures> \
  --step 7 --tm-alignment 32 --out <external-results.json>
```

The old alignment can be selected in this diagnostic CLI with
`--tm-alignment 16`; it is not a serving rollback recommendation.

One TP4 diagnostic service stalled during a request transition. The initial
hook reread a global active-case file during every sampler call; the async
runner can still have a round in flight after the API reaches its output cap.
Removing/switching that file can send ranks through different diagnostic
collectives. The retained hook now binds the tape to request ID. The accepted
comparisons validate the complete common prefill/q8 groups and exclude extra
in-flight groups; MBPP3 TP4 was collected in a separate startup. The revised
request-binding hook has not received a fresh multi-request endpoint gate.
This diagnostic failure is not classified as a production model defect.

The first standalone TP2 collective probe lacked a current `VllmConfig` and
failed before validation; a configured rerun passed. A debugger attach was
unavailable on this host. Failed paths and task-script revisions are retained.

Artifacts: `v100-quasar-tp2-tp4-quality-audit-20260906`, including
`operator-all-layers-fixed.json`, `padding-probe.json`,
`teacher-comparison.json`, `sampling-flips.json`,
`last-layer-propagation-summary.json`, actual module manifests, native/JIT
inventory, exact token tapes, launch scripts and logs. Results are under its
`results/` directory; large captures remain external to Git. Task services and
GPU leases have been released.

The remaining quality gate is to compare the pinned QAT W4A4 contract and the
V100 W4A16 path against the exact BF16 teacher on shared prefixes, then test
whether delaying row-partial FP16 rounding reduces the measured drift without
losing the established round-time budget. Neither is declared solved here.
