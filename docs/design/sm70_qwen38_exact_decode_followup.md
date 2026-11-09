# Qwen3.8 exact single-request decode follow-up

Integration line: `public/main`, base `755baae1d075ee04fa9096b23fc0225b23589a86`.
This task is stacked on HC PR #506 (`4ae6a0005a`) to preserve the measured
single-request contract; it does not duplicate the M4/M8/M16 work in #504 or
the Page4 relocation-order fix in #494. Human review is required. AI assistance
was used (OpenAI Codex).

## Baseline and scope

TP4 V100-SXM2-32GB GPU0–3, Torch2.10.0+cu128, Qwen3.8-Flash-Next-NVFP4,
FP16 activations/KV, native NVFP4 experts, no MTP/prefix cache, hybrid PLE,
V2 dual CUDA graph, max length262144, prefill chunk8192, single request.
8K/513 before-capture engine baseline: 96.394713/96.417843/96.420508 tok/s,
aggregate96.411020, TPOT10.372258ms. Configured256K is not an input-quality gate.
Three greedy baseline token sequences equal the previous baseline; two short
official-sampling/thinking/natural-EOS smokes passed. Full256K quality pending.

The matching 8K/32 graph-node trace has 29 middle replay windows/rank. Complete
HC is2.033180ms; QSA1.219981ms including top-k0.307393ms; router top-k0.486903ms.
These are GPU service times, not additive end-to-end latency.

Raw baseline directory (outside Git):
`/home/ymzx/桌面/1cat-vllm/worktrees/v100-qwen38-nomtp-token-trace-20260903-173451/.artifacts/hc_trace_vector_20260905_retry1/`.

## Three bounded directions

1. Build the existing QSA single-row decode top-k from the frozen source and
   verify that the actual runtime uses it. The baseline trace contains the
   generic kernel even at grid1; source selects the specialized kernel for M1.
2. Screen a lossless 32-bit router sort key for FP16 logits/E512/K10/M1.
   Keep max/exp/normalization reduction, tie rules and invalid-row semantics.
   Other dtypes and widths retain the original 64-bit-key path.
3. Screen precomputed sparse KV addresses, keeping logical token order,
   split/merge arithmetic, dtype and FP16 boundaries. Do not assume a gather
   or persistent-kernel rewrite is profitable.

No lower precision, changed accumulation order, top-k truncation, MTP or
top1-only LM-head shortcuts. First prove numerical/graph/shape gates on small
operators, then use a combined whole-model run when justified. Never preempt
unrelated GPU owners. Keep failed paths and raw evidence.

## Current status

Draft PR #507 is stacked on #506. The three operator paths are implemented
and hit the combined model trace. A whole-model run measured98.965175 tok/s.
The frozen baseline's greedy output was not reproducible on an isolated
old-source control either. No new operator mismatch was found, but do not
promote this as complete deterministic-output or broad quality acceptance.
Detailed evidence and the existing repeatability issue follow.

The router keeps the original 8-warp FP32 max/exp/normalization tree. Its
32-bit key is selected only for existing FP16 E512/K10 M1 routing. Signed-zero,
NaN/Inf, ties, all raw half encodings and finite shuffled inputs are covered:
1,280 rows are bitwise equal, plus16 changed-input/poisoned graph replays.
After warmup, eight alternating-order A/B pairs measure48 calls at
**0.252652 ->0.203530ms** (save0.049121ms). An earlier non-interleaved screen
drifted in clock state and is retained but not used for the accepted delta.

The QSA sidecar builder compiles the production header, exposes a version
marker and a literal old generic-kernel control.72 cases cover M1/M2, lengths
0/1/511/512/2048/2304/2305/4096/65536, ties, signed-zero and non-finite scores;
all selected IDs are exact.16 changed-input/poisoned graph replays pass.
Twelve calls at live lengths2048–2169 measure **0.258196 ->0.111665ms**.
Source-overlay services must pin the freshly built library through existing
`VLLM_SM70_QSA_TOPK_LIBRARY`; do not infer native-binary freshness from Python
source or an old unversioned log message. The endpoint trace must prove the
decode-specialized symbol is present. Header SHA256:
`e09d4af611894d2c3613ea1d5ac50e1fd2606f729e6fa7c45eb8079fcc6b9508`.
Sidecar SHA256:
`e56d7874877dddb88589a7e08ecbe9074f7f7532af1be764ab5f47a9b8aa8165`.

QSA address resolution retains logical ordering/duplicates and validity,
precomputes physical token slots, and removes the dependent page-table load
from the unchanged partial attention arithmetic. It is limited to SM70,
FP16 M1/Q6/KV1/D256/page400/selection2051 and signed-int32-safe physical slots.
Other shapes, cache formats and prefill keep the original path. Eight changing
graph scenarios per length test page relocation, invalid pages/indices and
requests, duplicates and poisoned outputs; all outputs are bitwise identical.
Public production-dispatch screen (12 attention+merge calls, resolver included):

| Cache context | Original ms | Resolved ms |
|---|---:|---:|
|8192|0.345016|0.318909|
|32768|0.384205|0.367094|
|262144|0.365860|0.323968|

These are operator service times, not additive endpoint savings. The256K row
is an operator cache-size check, not a256K model-input quality acceptance.
CPU dispatcher/QSA launch suites:49 passed,1 skipped (GPU-only). Targeted Ruff
checks pass. No new lower-precision weights, KV or arithmetic introduced.
An additional correctness-only run uses the real worker's interleaved K/V
layout, strides `[204800,256,256,1]`; eight graph scenarios at each of8K/32K/
256K are bitwise equal. The timing table above uses separate contiguous K/V
storage; do not relabel it as an interleaved-layout latency measurement.
Reproduce the additional gate with `--interleaved-kv --skip-timing`; raw output
is `.artifacts/three_paths/address_interleaved_exact.json`.

Reproduction (project Python environment, SM70 GPU ownership required):

```bash
CUDA_HOME=/usr TORCH_CUDA_ARCH_LIST=7.0 .venv/bin/python \
  benchmarks/kernels/build_sm70_qsa_topk_sidecar.py --build-dir .artifacts/qsa-build
.venv/bin/python benchmarks/kernels/verify_sm70_qsa_router_exact.py \
  --qsa-library .artifacts/qsa-build/vllm_qsa_decode_topk_sm70.so \
  --out .artifacts/operators.json
.venv/bin/python benchmarks/kernels/verify_sm70_qsa_resolved.py \
  .artifacts/address.json
```

Raw task artifacts: `.artifacts/three_paths/operators_interleaved.json`,
`address_production.json`, build/queue logs. All standalone GPU screens exited.

## Combined endpoint, source d2c8401c22

One model initialization produced `.artifacts/endpoint/result.json`,
`quality.json`, `contract.txt`, `run.log`, and the8K/32 graph-node report.
Workers explicitly log QSA source-overlay decode specialization version1;
the trace contains `qsa_lexicographic_decode_topk_kernel` and the new
`_qsa_resolve_physical_indices_kernel`. The old HC/W2 binaries and M2-disable
compatibility pin are unchanged.

| Unprofiled8K/513, three repeats | Frozen baseline | Candidate |
|---|---:|---:|
| Steady decode tok/s |96.411020|98.965175|
| TPOT ms |10.372258|10.104565|

Candidate repeats:98.939675/98.977150/98.978711 tok/s. The observed difference
is0.267694ms/token (2.65% throughput), not a three-repeat same-output delta.
This does not meet100 tok/s. Mean TTFT1.169634s and prefill1.166232s; the
fixed8192-token prompt corresponds to approximately7024 prefill tok/s.

The graph trace has31 replays/rank,29 middle windows. Rank-average diagnostic
GPU service (ms/token; do not sum these to close endpoint wall time):

| Work | Frozen trace | Candidate trace |
|---|---:|---:|
| QSA selected-block top-k |0.307393|0.175693|
| Router top-k |0.486903|0.356512|
| QSA partial attention |0.436250|0.382302|
| QSA split merge |0.117537|0.116910|
| New physical-address resolver |0|0.028287|
| Whole QSA category |1.219981|1.060166|

The candidate trace includes a21.585578ms interval and7.718280ms following
rank-start skew. Keep these samples: mean rank-max interval11.328738ms,
median10.840832ms. Complete HC mean2.100456ms/p502.022327ms versus baseline
2.033180ms/p502.031939ms; its code is unchanged and waiting distorts the mean.
Do not call this an HC regression/improvement or replace unprofiled TPOT with
the trace interval. The previous HC1.5ms target remains unmet.

Health checks use official temperature1/top-p0.95/top-k20, seed0, thinking,
natural EOS. Short arithmetic, exact record copy, and261632-token padded
arithmetic all pass (125/110/140 output tokens). The task is at the end of the
long prompt: this is not long-range retrieval or comprehensive quality.
The262143-input +1-output request completes the262144 boundary; its50.758678s
prefill is approximately5164 tok/s, and it has no steady decode iteration.

### Whole-model repeatability investigation

Do not hide this difference behind the three passing health checks. The old
greedy/ignore-EOS513-token performance repeats were identical. All three new
repeats have different hashes. Repeats1/3 match the old natural prefix through
first EOS at index28 and diverge at index37, after forced continuation.
Repeat2 emits EOS at index8, so the mismatch is **not only after EOS**.
Precision/arithmetic source contracts and operator checks pass; that alone
does not prove unchanged full-model output.

Frozen token hash:
`7385dacbed6a3d06576993bda99375b51c6a2e6132ee4f2fc0079646b461fca1`.
Candidate hashes:

- `f1ad6ecc74c99ff5980d993884b44e7e147d477637353c9e750310c9ccaa08a8`
- `8419c18737e8ef9135e1a9f5e3cd788ea6561d63b3168a8f307f8deb48874899`
- `1888a2872baf55ec768df79cb7b8877b4e5911970ee949ffa213f815cdc55bb2`

Model-free follow-up against the **installed old native QSA binary**, rather
than only the rebuilt control:72 cases and16 changing-logit graph replays
pass. An additional128 changing-length graph replays/1536 row comparisons
pass, including fast/fallback transitions, ReLU zero ties, near ties,
non-finite scores and padded row storage. Public reproduction function:
`qsa_dynamic_screen(torch.ops._C.qsa_lexicographic_topk)` after loading the
candidate sidecar. Raw files are `installed_topk_oracle.json` and
`dynamic_topk_oracle.json` under `.artifacts/three_paths/`.

Known test-condition differences: the new run uses task-owned fresh compiler
caches and performs the261632-token health request **before** the speed
repeats; the old baseline had only short health requests. A second and final
model initialization tested these conditions on clean old-source4ae6a0005a,
without any of the three candidates and without Nsight capture.

Control `.artifacts/control/`: two short health requests,513/32-token warmups,
three8K/513 repeats, the same261632-token health request, then three8K/513
repeats. Both sets use the same loaded engine. All health checks pass naturally
(109/167/90 tokens). Before the long request the repeats have EOS indices
8/28/28 and two distinct hashes; after it they have8/8/8 and three hashes.
Therefore long-request history is **not necessary** for the observed drift.
The old-source control reproduces the candidate's early-EOS hash
`8419c18737e8ef9135e1a9f5e3cd788ea6561d63b3168a8f307f8deb48874899`
in all513 tokens, both before and after the long request. This specific
repeatability symptom demonstrably predates the three optimizations; it is
not evidence by itself of a new precision reduction or new wrong top-k.
It does not prove that every other generation difference is harmless.

Control mean decode is96.107235 tok/s before and96.083709 after the long
request. One identical-output old/new pair measures96.106397 versus98.977150
tok/s (0.301793ms/token less); this is one pair, not a replacement three-repeat
controlled acceptance benchmark. The primary observed candidate result remains
98.965175 tok/s. The CPU audit is `.artifacts/repeatability_audit.json`.

Related existing [PR #494](https://github.com/1CatAI/1Cat-vLLM/pull/494) fixes
page4 physical-allocation-dependent attention ordering and is still open,
not merged, at this audit. Its established issue is a relevant next suspect,
not a proven explanation for every NVFP4 output above. Reuse/review that fix
before inventing another planner rewrite; do not silently adopt a new FP32
arithmetic or NCCL policy to force matching text. Actual-tensor capture and a
same-engine intervention would be needed for causal attribution here.

All two model jobs, operator jobs, samplers and queues are terminal. GPU0–3
were released; GPU4–7 belong to an unrelated service. Final integration is
`public/main`; #507 remains Draft, stacked on #506, pending human review and
resolution of the relevant existing quality/integration gate. No main push
or resident API was performed. Both100 tok/s and HC1.5ms remain unmet.

Retained SHA256 checksums:

- Candidate result: `e4b27b6e54298cf26cbd5acde28046a5dfed6210265b752ce7df1ac75ad2c8cb`
- Candidate health: `2a79fdfc172d709c9245d911a195d88c285dffd947011a2e306b4914b3ec204f`
- Candidate trace: `d1afc22e0d91d5c7ce94bfaa3132675302af4ef74e2be18456dec4983b897cd2`
- Old control result: `95c577b208090aa605db3402641e563a8d3892ef55823d0dc4bcb80478786ccb`
- Old control health: `fe3b26835418ca7508fcbef275560c293073f2f611872656785f5fa5631c9944`
