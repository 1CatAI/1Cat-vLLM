# QUASAR DFlash2 quality-preserving latency recovery

## Result and acceptance boundary

The E4M3 KV / FP32-logit configuration recovers complete verification-round
latency from about 20 ms to **18.892 ms on release1k and 18.435 ms on MBPP28**
in an independent production startup without diagnostic worker extensions.
These are unprofiled medians, with one warmup and three measured requests.
The previous E5M2 / FP16-logit peaks were 18.037 and 17.588 ms respectively.
**The 17.6–18 ms objective is not yet met.** The remaining gap is about 0.9 ms.

All control/candidate comparisons within each service retained the complete
token sequence and acceptance counts. The long-output comparison also retained
all 3,559 / 1,400 / 1,856 tokens for the three selected MBPP inputs. This is
evidence about these optimizations, not proof that every source of model
quality loss has been eliminated. The production confirmation reproduced both
speed-fixture hashes and all 4,939 / 5,904 / 633 quality-fixture tokens from the
previous precision configuration. Task-only diagnostic services produced other
fixed-seed trajectories; the cause of that difference remains unresolved.

The preceding [precision repair](sm70_quasar_e4m3_fp32_logits.md) remains active:
FP32 candidate logits, E4M3 target KV, and exact-reference fallback for ambiguous
top-k/top-p boundaries. This change does not relax their numerical contract.

## Implementation

1. Capture the fixed eight-row draft context computation in private CUDA graph
   pools. Variable prefill shapes retain the ordinary path.
2. Separate context projection from cache insertion. For B1/q8 decode, project
   the target hidden states and compute context K/V before the target sampling
   fence. Only insert K/V after acceptance has populated the original slot
   mappings. Rejected and evicted rows retain `PAD_SLOT_ID` and do not write.
3. Replay the three persistent metadata copies used by the captured non-causal
   Flash-V100 paged draft graph. Other shapes, causal models, CP, alternate
   backends, and anchored attention keep the normal metadata builder.
4. Preserve the top-20 view of the 21-column cutoff probe. The rejection kernel
   already accepts a row stride; this removes two copies. B1 queries whose
   sampling rows cover all real tokens also avoid two identity index gathers.

The projection, normalization, RoPE, KV write operators, and sampling arithmetic
remain the same. The eager context method and the split methods share the same
implementation. Persistent graph outputs are retained explicitly; compute,
cache insertion, and draft query graphs have separate allocation pools.

The additional opt-in is:

```bash
VLLM_SM70_DFLASH2_FP32_LOGITS=1 \
VLLM_SM70_DFLASH2_CONTEXT_PIPELINE=1 \
vllm serve <pinned-quasar-checkpoint> \
  --tensor-parallel-size 4 --dtype half \
  --attention-backend FLASH_ATTN_V100 --kv-cache-dtype fp8_e4m3 \
  --max-model-len 262144 --max-num-batched-tokens 4096 \
  --max-num-seqs 4 --gpu-memory-utilization 0.8 \
  --enable-prefix-caching --mamba-cache-mode align \
  --speculative-config '<pinned-dflash2-config>'
```

`VLLM_SM70_DFLASH2_CONTEXT_KV_GRAPH=1` enables only the context-graph stage for
isolated A/B comparison. The pipeline flag includes that stage. Both new flags
default off; these measurements do not promote other checkpoints or hardware.

## Frozen workload

- Integration base: `755baae1d075ee04fa9096b23fc0225b23589a86`.
- Precision control: `6ec27bec9c2e0aec24037597710045b3b8b25e5d`, Draft PR #517.
- Target: `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4`, revision
  `d8e6fbfa3e3a78899b440222b827430045a05b44`.
- DFlash2 revision: `dedf8df68adfb1afeaf7b7480c0a0243108177b4`; draft weight
  SHA256 `67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c`.
- Four V100-SXM2-32GB GPUs, TP4, CUDA 12.8, Torch 2.10.0+cu128,
  Python 3.12.13, Triton 3.6.0. Target NVFP4 uses TurboMind W4A16/QPN2.
- V2 runner; target and draft Flash-V100; target E4M3 KV, draft FP16 KV;
  seven probabilistic draft tokens, eight-row verification; full/piecewise
  target graphs and full draft graphs.
- Context limit 262,144; token budget 4,096; capacity four requests; one live
  request; memory utilization 0.8; prefix cache and Mamba align enabled.
- Temperature 1, top-k 20, top-p 0.95; thinking enabled, effort `xhigh`;
  no image/video inputs. Natural EOS, 1,024 output-token cap for speed.
- release1k uses 1,019 prompt tokens and seed 20260925. MBPP28 uses 135 prompt
  tokens and seed 0. Preserve these different seeds when reproducing.

## Unprofiled comparison

The independent production confirmation uses implementation commit
`f22ac115d0ac0cc8a13bd042cf1472c33add00c4`, no profiler or development mode, and
fresh task-owned compile caches. Against the preceding precision repair:

| Workload | Precision control round ms | Optimized round ms | Control / optimized decode tok/s | Tokens / rounds |
| --- | ---: | ---: | ---: | ---: |
| release1k | 20.149 | 18.892 | 135.033 / 144.015 | 303 / 111 |
| MBPP28 | 19.745 | 18.435 | 267.695 / 286.721 | 260 / 49 |

The full token hashes and acceptance counts match the preceding control on all
four requests (warmup plus three measured runs) for both fixtures. Final TTFT
medians are 0.350 / 0.109 s. The round cost drops by 6.24% / 6.64%; the remaining
gap against the old E5M2/FP16 peaks is 0.855 / 0.847 ms.

The control below is the repeated control at the end of the same-process
control → pipeline → control experiment. Both arms include the removal of
identity gathers and stride copies; only the new graphs/pipeline are toggled.

| Workload | Control round ms | Pipeline round ms | Control decode tok/s | Pipeline decode tok/s | Tokens / rounds |
| --- | ---: | ---: | ---: | ---: | ---: |
| release1k | 20.035 | 18.925 | 142.457 | 150.817 | 275 / 96 |
| MBPP28 | 19.642 | 18.467 | 243.926 | 259.445 | 437 / 91 |

Median TTFT was 0.349 / 0.352 s for release1k and 0.101 / 0.106 s for MBPP28
(control / pipeline). These are small samples, not a TTFT improvement claim.
Decode rates exclude TTFT/prefill. Round latency is not latency per emitted
token; acceptance is shown separately and remained identical in the A/B arms.

The earlier context-graph-only experiment recovered 20.168 / 19.834 ms to
19.421 / 19.166 ms. A direct same-service comparison then measured context-only
19.424 / 19.124 ms versus staged computation 19.304 / 18.991 ms. Do not combine
different startup trajectories into an acceptance-rate comparison.

## Operator and text validation

- Actual loaded draft weights: 16 changing hidden-state/position/slot cases on
  each TP rank, covering accepted lengths 1–8 and positions near 1K, 32K,
  128K, and the 256K boundary. Context graph and split compute/write graph
  produced bitwise-identical cache entries to eager execution on all ranks.
- Real metadata builders: all three persistent inputs matched ordinary rebuilds
  bitwise at 1K, 32K, 128K, and 256K on all four ranks, including changed
  physical block-table contents.
- Context microbenchmark: approximately 0.63 ms eager submission/completion
  versus 0.066–0.069 ms graph replay, using 200 repeated resident-input calls.
  This is a component measurement, not the whole-round speedup.
- Selected MBPP indices 3, 7, and 24: all full token hashes identical between
  control and pipeline. Base tests 3/3, Plus tests 1/3; identical transcripts
  imply identical scores for the control. The score is not evidence of an
  improvement over the previous startup, which had different transcripts.
- Structured JSON returned the requested integer 42. All quality cases ended
  naturally, without truncation or replacement characters.
- Independent production confirmation: MBPP outputs 4,939 / 5,904 / 633 tokens,
  bitwise identical token IDs to the preceding precision configuration; Base
  3/3, Plus 0/3, and JSON result 42. The diagnostic service's Plus 1/3 result
  above must not be reported as a production quality improvement.
- Focused GPU regressions: 32 passed. Flash-V100 metadata/policy tests:
  15 passed. Ruff and local mypy passed.

```bash
.venv/bin/python -m pytest -q \
  tests/kernels/attention/test_dflash2_context_pipeline.py \
  tests/v1/spec_decode/test_rejection_sampler_utils.py \
  tests/v1/spec_decode/test_dflash2.py \
  -k 'context or dflash2_sparse_topk or compact'
.venv/bin/python -m pytest -q \
  tests/v1/attention/test_sm70_flash_v100_policy.py -k 'metadata or draft'
```

The new kernel regression checks that early computation does not mutate the
cache, then changes acceptance and physical slots before every write replay.
It compares the entire cache, including invalid entries, with masked eager
execution. Padded-candidate rejection tests exercise the retained sentinel
stride while placing a large excluded value in the sentinel column.

256K validation here is at operator/metadata level. No new 256K endpoint speed
or full long-context model-quality result is claimed.

## Profiling and rejected paths

Nsight Systems 2025.3.1 captured CUDA graph nodes on all four TP ranks. Aligning
the explicit verification-round ordinals gives 12 complete groups; excluding
the two edges leaves 10 steady groups. Diagnostic round intervals averaged
20.815 ms, and host `DFlash2Speculator.propose` ranges averaged 0.801 ms.
These traced values include profiling overhead and are not accepted speed.
The compact cutoff range includes the preceding GPU-completion wait; its
12.443 ms average must not be called CPU boundary-check computation.

- Nsight 2026.4 produced no CUDA report even for a minimal probe. Its bundled
  release notes explicitly remove Pascal/Volta support starting at 2025.4.
  A verified NVIDIA 2025.3.1 package fixed this environment issue.
- The first 2025.3 trace had all worker NVTX ranges but GPU events only for
  rank zero. It is retained as partial evidence, not a TP4 critical-path table.
  Starting/stopping the CUDA profiler on every rank produced the complete trace.
- Pinning the four worker main threads to separate NUMA-local cores changed
  medians by only about 0.03–0.04 ms in a direct test. Original affinities were
  restored; pinning is not part of the accepted configuration.
- The boundary guard still detects real ambiguous inputs. Disabling it, lowering
  logit precision, or reverting KV dtype was not admitted as a speed fix.

## Retained evidence

Task bundle: `v100-quasar-quality-speed-recovery-20260906`.

- `results/metadata-ab-{control,control-repeat,pipeline}-speed-*.json`.
- `results/final-{control,pipeline}-quality-subset.json`,
  `results/final-pipeline-json.json`, `results/final-evalplus.json`.
- `results/production-confirm-{speed-release1k,speed-mbpp28,quality-subset,json,evalplus}.json`,
  `production-worker-provenance.json` (four workers, native hashes and 184 JIT cubins).
- `results/metadata-context-real-oracle.json`, `results/metadata-real-oracle.json`.
- `profile/final-tp4.nsys-rep`, original SQLite, explicitly aligned SQLite,
  `final-tp4.rounds.json`, and `final-gpu-breakdown.{json,csv,md}`.
  Each trace row is a verification round, not an emitted token.
- `scripts/serve-pipeline.sh`, A/B clients, real-weight worker oracles, task
  lease/ownership record, worker/native provenance, and the tested source diff.

The next performance gate is the remaining sampling/state/launch dependency
after target completion. It needs its own exact-distribution and mutable-state
validation; a projected saving is not evidence that the old peak is restored.
