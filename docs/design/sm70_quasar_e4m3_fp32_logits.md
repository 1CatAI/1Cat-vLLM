# QUASAR + DFlash2: E4M3 KV and FP32 logits

This is the precision follow-up to the
[operator audit](sm70_quasar_dflash2_operator_audit.md). It addresses the
observed 0.021532 top-p distribution error and adds E4M3 storage to the
grouped q8 verifier. The opt-in mode improves measured operator precision;
it is not a claim that the original unquantized model or aggregate answer
quality has been recovered.

## Implementation

`VLLM_SM70_DFLASH2_FP32_LOGITS=1` retains FP32 logits for the existing
SM70 TP4 DFlash2 LM-head contract: FP16 weights, local shape 62080 by 5120,
QPN8 top-64 candidate search, and up to eight verifier rows. A small Triton
kernel evaluates each selected original weight row with FP32 multiplication
and reduction, storing directly into FP32 candidate buffers. It avoids both
FP16 output rounding and expanded cross-row products in the old rerank.
Dense-vocabulary ordering and the audited sampling cutoff protection remain
enabled. Full-vocabulary fallback also returns FP32 logits through
`torch.mm(..., out_dtype=torch.float32)`; this is tested with Torch 2.10.
The flag defaults off, making the changed precision contract explicit.

Explicit `--kv-cache-dtype fp8_e4m3` now uses the grouped one-pass q8 kernel
for supported Hq6/Hkv1/D256 layouts, instead of forcing independent XQA
rows. Reuse the existing exact E4M3 paired conversion used by XQA. Other
grouped sparse-page paths retain their conversion. The new entry requires
16-byte-aligned KV block/token strides; the backend gates that contract.
The E4M3 route is limited to q8/one-pass in model dispatch and the native
API. Other query lengths retain their existing fallback.

The native module advertises E4M3 grouped support. Python paired with an
older native extension keeps the older routing behavior rather than
calling an unsupported entry. Rebuild Flash-V100 to use the new route;
no TurboMind native rebuild is needed.

## Same-input numerical evidence

References reuse the previous valid C2 captures: all four TP ranks,
verification steps 0/7/31, original checkpoint weights and hidden states.
They measure runtime arithmetic against that same quantized checkpoint,
not checkpoint quantization loss against an unavailable original model.

| Measurement | Previous FP16/E5M2 | New FP32/E4M3 |
|---|---:|---:|
| Top-p TV on the previously problematic row | 0.02153214 | 3.6694e-7 |
| Maximum top-p TV across the 24 real rows | 0.02153214 | 1.1325e-6 |
| Top-p support matches FP32 reference | 23/24 rows | 24/24 rows |
| Candidate logit maximum absolute error, actual QPN8 support | FP16 rounding up to one logit ULP | 3.8147e-6 |
| Missing local top-21 in actual QPN8 top-64 support | separately audited | 0 across 12 rank/step cases |
| Maximum real KV conversion relative L2 | 0.0593241 | 0.0290236 |
| Native KV encoder vs matching Torch FP8 conversion | exact | exact |

The problematic distribution difference is reduced by about five orders
of magnitude on that row, not mathematically zero. Dense FP32 fallback has
a different reduction order from the independent FP32 reference: maximum
absolute logit error 0.00051308 and maximum top-p TV 2.9933e-5, with matching
support on all 24 rows. Candidate-path results must not be substituted for
dense-fallback results in a precision claim.

KV checks use 384 K/V cases per dtype across the 16 full-attention layers,
four ranks and three captured steps. No checked source value exceeds E4M3's
finite magnitude range, and no NaN/Inf was produced. Native cache write
and decode roundtrips match the corresponding dtype reference. This is
evidence for these inputs, not a proof that all future KV values fit an
unscaled E4M3 cache. The checkpoint supplies no KV scale tensors.

Independent randomized-page attention checks cover 1032, 32768, 131072 and
262144 context lengths. At 262144, representation-induced relative output
error falls from about 0.04856 to 0.02556; arithmetic on the same decoded
cache stays around 3.3e-4. Graph and eager outputs agree.

The faster E4M3 conversion is bitwise equal to the scalar conversion on all
four context lengths and a fixture exercising every finite E4M3 encoding,
including signed zeros and subnormals. No attention arithmetic or reduction
schedule is changed by that optimization.

## Performance evidence and limits

A real-hidden candidate microbenchmark on V100, M=8/K=5120/64 candidates,
measures about 100.76 us for packed FP16 rerank and 5.68 us for FP32 candidate
dots. This is a graph microbenchmark with repeated inputs and resident
candidate weights; it is not an end-to-end speedup claim.

Same-input native E4M3 conversion A/B:

| Context | Scalar grouped us | Paired grouped us | Output |
|---|---:|---:|---|
| 1032 | 45.57 | 36.25 | Bitwise equal |
| 32768 | 225.18 | 159.33 | Bitwise equal |
| 131072 | 824.52 | 562.28 | Bitwise equal |
| 262144 | 1617.41 | 1089.13 | Bitwise equal |

These are complete grouped-attention kernel-path times, not isolated
conversion instructions. Absolute timings from different microbenchmark
executions are kept separate; use the same-input A/B for the paired
conversion delta.

The retained final service result is recorded below after validation.
Complete-round cost is engine pure-decode time divided by speculative
rounds. It is not the target verifier alone. Changed precision can change
generated length and acceptance, so include them with throughput and TTFT.

| Configuration | Release 1K round ms / decode tok/s | MBPP28 round ms / decode tok/s | Output tokens (1K / MBPP) |
|---|---:|---:|---:|
| Original E5M2/FP16 baseline | 18.037 / 152.826 | 17.588 / 268.541 | 318 / 308 |
| E5M2/FP32 control | 20.457 / 166.521 | 19.979 / 252.811 | 529 / 299 |
| E4M3/FP32 scalar conversion | 20.516 / 146.230 | 20.254 / 224.833 | 283 / 297 |
| E4M3/FP32 paired, final confirmation | 20.149 / 135.033 | 19.745 / 267.695 | 303 / 260 |

Final confirmation TTFT medians are 0.3512/0.1033 s. Accepted/emitted
lengths are 2.7297/5.3061 tokens per round, with 111/49 rounds. An earlier
paired-service startup measured 20.099/19.623 ms; the confirmation measured
20.149/19.745 ms and reproduced the complete token hashes of both speed
fixtures. The original 17.6–18.0 ms round cost is **not fully recovered**.

Scalar and paired service transcripts differ, even though all same-input
conversion probes agree bitwise. Added replay on 192 real-query cases with
C2 cached KV re-encoded to E4M3 also agrees bitwise. These checks do not prove
end-to-end trajectory equivalence, so the final paired service was separately
checked for natural-stop text and JSON quality instead of inheriting the
scalar service score.

E5M2/FP32, scalar E4M3/FP32 and final paired E4M3/FP32 services completed the same three
diagnostic MBPP requests naturally. Base tests pass 3/3, Plus tests pass
0/3, and a JSON-object request returns valid JSON with result 42. These
selected hard cases do not establish an improvement in answer quality.
The original DFlash baseline had Plus 1/3 on those cases, and its target-only
control had 0/3. Keep this limitation visible.

## Validation and reproduction

Environment: four V100-SXM2-32GB GPUs, TP4, Python 3.12.13, Torch 2.10.0+cu128,
CUDA 12.8, Triton 3.6.0. Target checkpoint revision
`d8e6fbfa3e3a78899b440222b827430045a05b44`; draft revision
`dedf8df68adfb1afeaf7b7480c0a0243108177b4`.
Starting precision-audit commit:
`7faa7f682ea4b23921241ae01c0179dcc3fde058`, based on
`onecat/main` at `755baae1d075ee04fa9096b23fc0225b23589a86`.

Focused checks:

- FP32 candidate graph replay/rounding and E4M3 scaled random-page tests:
  7 passed.
- Existing grouped E5M2 regression tests: 25 passed.
- Routing and DFlash2 rerank/compact-top-k selection:
  31 passed, 257 deselected.
- After paired conversion: new E4M3 and existing grouped tests:
  29 passed.
- Final native shape/stride contract and E4M3 reference/graph checks: 6 passed.
- Mypy, Ruff and required commit hooks are checked on the final source.

Example serving contract, using the pinned local checkpoints and this
checkout's rebuilt Flash-V100:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
VLLM_USE_V2_MODEL_RUNNER=1 \
VLLM_SM70_DFLASH2_FP32_LOGITS=1 \
vllm serve "$QUASAR_MODEL" \
  --dtype half --tensor-parallel-size 4 \
  --attention-backend FLASH_ATTN_V100 --kv-cache-dtype fp8_e4m3 \
  --max-model-len 262144 --max-num-batched-tokens 4096 --max-num-seqs 4 \
  --gpu-memory-utilization 0.8 --enable-prefix-caching --mamba-cache-mode align \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --default-chat-template-kwargs '{"enable_thinking":true,"reasoning_effort":"xhigh"}' \
  --speculative-config "{\"method\":\"dflash\",\"model\":\"$DFLASH2_MODEL\",\"revision\":\"dedf8df68adfb1afeaf7b7480c0a0243108177b4\",\"num_speculative_tokens\":7,\"kv_cache_dtype\":\"auto\",\"attention_backend\":\"FLASH_ATTN_V100\",\"draft_sample_method\":\"probabilistic\",\"enforce_eager\":false}" \
  --seed 0
```

Speed requests use temperature 1, top-k 20, top-p 0.95, seed 20260925,
maximum output 1024 and natural EOS, with one warmup and three measured
repetitions. Quality requests use seed 0 and maximum output 16384.

Raw bundle: `v100-quasar-e4m3-fp32logits-20260906`. The local Chinese
report retains absolute locations. Important files include:

- `results/head-pipeline-real.json`: actual QPN8 search, FP32 candidate and
  dense-fallback comparisons.
- `results/kv-real.json`: native encoder and representation errors.
- `results/e4m3-boundary.json`, `results/e4m3-pair-compare.json`: long
  boundaries and bitwise paired-conversion A/B.
- `results/*-speed-*.json`, `*-quality-subset.json`, `*-evalplus.json`:
  unprofiled service results, text, tokens and scores.
- `scripts/`, `queue/`, `logs/`: full commands and negative attempts.
- Native build manifests and live worker manifests identify the loaded
  extension, source hashes and task-owned cache locations.

Rejected attempts are retained: the first head probe imported the wrong
Python operator facade; the first prepared-buffer replay lacked inference
mode; and a paired-conversion probe was queued before its library finished
building. These failed probes produced no accepted numerical evidence.
The initial scalar E4M3 route is retained as the slower correctness control.
No new model weights or generated binaries are committed.

The final native host validation admits exactly q8. Its extracted CUDA
device fatbinary is SHA256-identical to the library used for final service
validation; only the host shape guard was tightened.
