# SM70 attention and quantized projection coverage across TP sizes

## Scope

Extend the existing V100 acceleration to the local tensor layouts produced by
TP1, TP2 and TP4. Dispatch must follow dtype, head grouping, matrix dimensions,
alignment and supported arithmetic, rather than a tensor-parallel-size allowlist.
Use FP16 inputs with FP32 accumulation for both QK and PV. The inherited
75T recipe used FP32 only for PV; see the precision audit below.

Integration: `onecat/main`. Base: `b711d5304525dfc0cca6bc8a0bb005f33fe1bbf8`.
Owned branch/worktree: `codex/v100-tp-generalize-20260921-021932` /
`worktrees/v100-tp-generalize-20260921-021932`.

## Quality investigation; promotion remains pending, 2026-09-21

PR666 remains a draft. The [paired output audit](sm70_tp_quality_audit.md)
records the96-item ON/OFF comparison, the repaired DFlash context-boundary
failure, and the subsequent NVFP4 scale-rounding investigation. Do not promote
the candidate on the strength of finite outputs or operator checks alone.
The local-layout routes have no TP-size allowlist, but the candidate still
needs output-quality acceptance. Full QK/PV FP32 currently measures70-71T;
the75T target remains open. The earlier75T measurement used FP16 QK.

| Check | Current evidence | Outstanding |
| --- | --- | --- |
| TP1/2/4 local attention/projection geometry | GPU operator and changed-input CUDA Graph checks pass | Whole-model results are separate |
| QK and PV accumulation | Both FP32 in candidate r8; FP16 inputs/intermediates/output | Recover 75T without reducing precision |
| TP1 | Default no-DFlash64K, cold retrieval and C1-C32 requests pass | DFlash2/8192-token profiling still exceeds memory |
| TP2, 256K | Full-FP32 default no-DFlash cold retrieval and C1-C32 pass | DFlash2 shared-layout r5 remains earlier QK arithmetic |
| TP4 27B + DFlash2, 256K | r8 completes96 quality items, cold256K and C1-C32 bench | C16/C32 queue; matched audit: see linked report |
| Actual simultaneous decode | TP2 r5 C2/C4 measured; C8 queues | Do not relabel queued C8-C32 as resident decode |
| Shared QPN2 weight default | Operator equivalence passes | Paired model quality before changing default |
| 35B-A3B AWQ/FP8 migration target | No matching model found locally | Matched baseline still required; not claimed here |

TP4 r8 measured server (now stopped): TP4 on GPU4-7, target QUASAR-QAT Qwen3.8-27B NVFP4,
DFlash2 draft revision `dedf8df68adfb1afeaf7b7480c0a0243108177b4`,
num_speculative_tokens=7, target E4M3 KV, draft FP16 KV, FP16 compute,
max_model_len=262144, chunk8192, maxseq32, memory utilization0.85,
block_size2048, mamba_block_size8192/align, prefix caching on, FULL target/draft
CUDA Graphs. Cold tests reset prefix cache and disable benchmark ready-check
and warmup requests. Cache capacity is 922965 tokens after prefill workspace
profiling; graph memory is 1.75 GiB/rank. Other-task GPU0-3 remains untouched.

### Accumulation audit and rejected tuning

`PREFIX_PV_FP32_MMA_ACCUMULATE` was present in the inherited recipe, but
`PREFIX_QK_CUBLAS_FP32_ACCUM` was absent. Adding the latter selects FP32 QK
accumulation in prefix and triangular-tail GEMMs. Prefix uses cuBLAS default
Tensor Op (99), and tail uses Tensor Op algorithm11. These are numerical
changes; same-kernel parity alone is not a sufficient quality gate.

Q8192/KV262144, logical causal FLOPs, graph replay including per-head copies,
on V100-SXM2-32GB, Torch2.10.0+cu128/CUDA12.8:

| Recipe | Hkv1 (TP4 local) | Hkv2 (TP2 local) | Hkv4 (TP1 local) |
| --- | ---: | ---: | ---: |
| r5 QK FP16 / PV FP32 | 76.625T | 75.748T | 75.296T |
| r6 QK/PV FP32, original algorithms | 70.876T | 69.996T | 69.749T |
| r7 QK/PV FP32, PV M64 tile (rejected) | 63.024T | 62.386T | 62.100T |
| r8 QK/PV FP32, tuned algorithms | 71.111T | 70.221T | 69.982T |

All are single-GPU operators for the local shapes, not multi-GPU end-to-end
throughput. r8 times are 182.645/369.916/742.363 ms. Its 16 numerical/graph
checks pass; selected-row FP64 relative L2 ranges up to 0.002785. FP32
accumulation does not remove FP16 intermediate/output rounding. Raw evidence:
`prefill-graph-qk32-r8.json`, `prefill-numeric-qk32-r8.log`,
`precision-error-r8.log`.

Bounded cuBLAS algorithms99-115, four input layouts and cuBLASLt heuristics
were screened. No prefix candidate exceeds the selected layout. Tail
algorithm11 improves its isolated graph timing from about2.06ms to1.50ms.
An initial tail screen captured an empty graph because the cuBLAS handle
used the wrong stream; those files are explicitly renamed
`tail-qk32-empty-graph-invalid.*` and excluded. The corrected harness binds
cuBLAS to the capture stream. Torch profiler graph traces confirm prefix QK/PV
dominate the workload; apparent tail duration includes scheduling overlap and
is not the isolated compute time. Nsight Systems2022.4 lacks QdstrmImporter
on this host; its raw capture is not claimed as a readable trace.

### Compiled-library precision admission

The native library now reports `sm70_d256_gqa_accumulation_bits()`, checking
compile-time QK/PV flags in both Q8000 and Q8192 translation units. The Python
loaders require32 for this architecture route. Missing/old FP16-QK libraries
fall back to exact dense attention with a rebuild message; the independent
v37 operator retains its admission. Eight loader-policy cases pass, and the
rebuilt native library reports32. This adds capability reporting without
changing r8 GPU arithmetic; running r8 servers keep their original mapped
library. The replacement was installed atomically for fresh processes.

A final bounded CUTLASS prefix-QK screen also loses to cuBLAS: graph replay
at M49152/N24576/K256 and physical K stride262144 measures10.53-12.42ms
for four FP32-accumulating threadblock/warp shapes, versus paired cuBLAS
8.61/8.68ms. All output elements match the cuBLAS FP16 output in this screen.
It is rejected, not installed in serving (`qk32-cutlass-screen.log`).
The final extension was rebuilt after formatting at source4038f83009;
SHA256 `95db86166a28f3e0533b1e3931a860134293f32687cda0f24d4ef2cce4e13c39`.
Four changed-input/length graph regressions pass on the final binary. Serving
r9 loaded the pre-format capability build, SHA256
`08f089644ce3221bb80b3269cc7250fd06af14e8a95f10165a9f3df14417cdb2`;
the only later CUDA-source difference is formatting. No process hot reload
or private preloaded library is used.

### Corrected TP2 r5 long context and concurrency

r5 includes the captured-state fix and startup workspace profiling committed
in `fe630d3f4c`. Cold retrieval at32768/131072/256000 input token IDs returns
all three exact expected answers, naturally stopping at16 output tokens.
TTFT is13.466/70.832/181.628 seconds. Only the `r5b` stream timings are valid;
the earlier byte-at-a-time client parsed a large prompt-token-ID event too
slowly and delayed observations. Artifacts: `tp2-prefill-r5b-long-*.json`.

Cold `vllm bench serve`, input256000/output256, C1: 1 completed/0 failed,
TTFT183.500s (1395.1 input tok/s), request185.580s, complete-output1.379tok/s.
The post-TTFT synthetic output rate is122.59tok/s with mean DFlash accepted
length6.07. This is fixed-length random-input/ignore-EOS timing, not a claim
about ordinary natural-language decode speed. DFlash ITL measures stream
chunks and cannot be inverted as per-token decode throughput.

TP2 C1/2/4/8/16/32 cold benchmark requests all complete. However, server logs
show at most4 resident requests under this memory/page configuration; C8+
queues. A separate emitted-token-ID common-window measurement at2048 input /
2048 output records C2=165.43 and C4=195.90 aggregate decode tok/s. C8 has no
common decode window (last TTFT51.97s exceeds first completion43.34s), so the
harness rejects it instead of printing misleading throughput. Artifacts:
`tp2-prefill-r5-steady-c{2,4}.json`, `tp2-prefill-r5-steady.log`.

The DFlash capture-size policy also had a separate16-request cap. It now
includes verifier shapes through32 requests, independently of TP size; the
TP4 startup confirms q8*C32=256 is captured. Sixteen focused policy checks
pass. This fixes graph coverage, not memory capacity or scheduler residency.

## TP4 r8 serving results, 2026-09-21 13:12 CST

The full-FP32 TP4 configuration above completes96 sampled quality items:
GSM8K28/32 (strict extractor), MATH50032/32 (`math-verify`0.9.0), and
sanitized MBPP25/32 (all provided tests held out). MBPP initially has one
32768-token cutoff; item164 is rerun with ceiling65536 and naturally stops
at28149 tokens, still failing its assertions. Retain both records. Other
items finish naturally. These are sample scores, not a matched regression
comparison or proof of unchanged model quality. GSM8K item830 contains the
correct1128 minutes reformatted as18h48m, which the strict extractor misses.

Generated Python is evaluated in a subprocess with Landlock filesystem
isolation, seccomp network/process restrictions and CPU/memory/time limits.
All32 reference solutions pass the same evaluator. No model tests or expected
outputs are included in MBPP prompts; the entry function signature is provided.
Raw evidence: `tp4-default-qk32-r8-{gsm,math500,mbpp}32.jsonl`,
`tp4-default-qk32-r8-longanswer-mbpp32.jsonl`,
`mbpp32-reference-sandbox.json`.

Cold natural-EOS retrieval at32768/131072/256000 input tokens returns the
four expected values each time and stops at16 output tokens. TTFT is
7.612/39.673/100.595 seconds. Formal `vllm bench serve`256000/256 C1:
TTFT101.955s, request104.716s, prefill estimate2510.91tok/s, complete-output
2.445tok/s. Post-TTFT output rate92.35tok/s is a synthetic ignore-EOS result;
DFlash mean accepted length is2.857. It is not directly comparable to the
TP2 r5 result with accepted length6.071 or to natural-language decode.

Cold `vllm bench serve`,2048 input and256 output tokens/request, random dataset,
seed20260921+C, temperature0.7/top_p0.8/top_k20, no warmup/ready-check, prefix
reset before each case:

| Client concurrency | Total input | TTFT median(s) | TPOT median(ms) | Complete output(tok/s) | Request median(s) | Observed max resident |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 | 2048 | 0.523 | 3.260 | 188.877 | 1.354 | 1 |
| C2 | 4096 | 0.788 | 7.818 | 173.392 | 2.781 | 2 |
| C4 | 8192 | 1.828 | 8.889 | 232.275 | 3.718 | 4 |
| C8 | 16384 | 2.990 | 18.130 | 238.556 | 7.159 | 8 |
| C16 | 32768 | 4.844 | 30.052 | 248.126 | 13.652 | 13 |
| C32 | 65536 | 11.364 | 29.056 | 267.356 | 20.891 | 13 |

All requests complete with zero failures and zero observed prefix-cache hits.
C16/C32 each incur one preemption/recompute and queue behind the cache-capacity
limit. They are not32 simultaneous GPU decode. Complete emitted-token chunks inside an all-requests decode window measure
C2=245.96, C4=306.12 and C8=403.27 aggregate tok/s (2048 output tokens/request).
Window durations are15.60/22.70/38.58 seconds. C16/C32 are excluded because
the resident-capacity evidence does not support those simultaneous counts. TPOT is a
per-request post-TTFT average; stream ITL is a DFlash chunk interval and is not
inverted to obtain per-token throughput. Benchmark JSON and sampled residency
are retained under `tp4-default-qk32-r8-cold-c*`; consolidated values are in
`tp4-r8-serving-summary.json`.

## TP1/TP2 default no-DFlash checks, 2026-09-21 13:36 CST

Fresh capability-checked services use the same27B NVFP4 target, FP16 compute,
E4M3 KV, chunk8192/maxseq32, graphs enabled, default weight layouts and no
acceleration overrides. DFlash is **disabled** in these two configurations;
they must not be compared as matched scaling points against TP4 DFlash.
TP1 uses GPU4, maxlen65536 and memory0.92; TP2 uses GPU5-6, maxlen262144 and
memory0.85. Native workspace profiling is logged in both. Model/cache/graph
memory for TP1 is19.67/3.03/0.65 GiB, with71493 token cache capacity (the initial
32K launch); TP2 cache is12.21 GiB/762956 tokens and graph memory1.84 GiB.
The initial TP1 cache estimate supported a64K retry, which starts and serves.
That final64K launch retains3.03 GiB cache memory, reports82782 effective
cache tokens, and captures1.04 GiB of graphs.

Natural-EOS cold retrieval passes on TP1 at64000 input tokens (TTFT83.182s)
and TP2 at256000 (TTFT213.286s). Both return all four expected values and stop
at16 tokens; logs explicitly show QK+PV FP32 architecture/FP8-bridge dispatch.
These one-shot cold timings include any first-use compilation.

Cold `vllm bench serve`,2048 input/256 output per request, same protocol as
TP4, all requests completed without errors and no observed prefix-cache hits:

| Concurrency | TP1 TTFT median(s) | TP1 complete output(tok/s) | TP2 TTFT median(s) | TP2 complete output(tok/s) |
| --- | ---: | ---: | ---: | ---: |
| C1 | 1.728 | 23.933 | 1.000 | 37.914 |
| C2 | 3.705 | 38.028 | 5.385 | 30.769 |
| C4 | 7.295 | 58.321 | 3.866 | 94.477 |
| C8 | 20.670 | 64.272 | 8.252 | 105.723 |
| C16 | 20.569 | 69.850 | 10.510 | 145.241 |
| C32 | 52.949 | 68.333 | 18.182 | 182.143 |

TP1 reaches9 resident requests and queues higher client concurrency. TP2
reaches32 resident requests. Neither run preempts. TP2 C2 logs first-use
`_topk_topp_kernel` compilation; a single diagnostic repeat after clearing
prefix cache records TTFT1.541s/output58.158tok/s with warmed kernels. Keep
both observations, do not replace the slower first-use result or interpret
warm kernels as prefix-cache hits. Raw records use
`tp1-fp32-cap-64k-r9-c*`, `tp2-fp32-cap-r9-c*`, and
`tp2-fp32-cap-r9-warmkernel-c2`.

A separate512-output-token run measures actual common-window aggregate
TP2 decode at C2/4/8/16/32:76.02/145.56/272.64/474.05/570.35tok/s.
All five cases have a valid window after every first token and before any
last token; durations13.39/13.95/14.88/17.04/28.09s. These exclude prefill.
Per-request median emitted-token intervals are26.25/27.38/29.28/33.67/55.97ms.
Raw records: `tp2-fp32-cap-r9-steady-c*.json`. This confirms resident C32
serving on the TP2 no-DFlash configuration; it does not change the DFlash
memory limits documented above.

TP1 27B+DFlash2 remains unaccepted due to memory, and TP2 DFlash2/256K still
uses explicit shared QPN2 in its earlier r5 checks. The new default no-DFlash
results do not erase those limitations. QPN4 retains its pre-existing
single-sequence/no-MTP admission; the generic LM-head/attention/local-projection
changes must not be described as removing every specialized kernel contract.

## Acceptance and worklog

- Correct E4M3 XQA partition selection for multiple KV heads at 32K through 256K.
- Admit GQA6/D256 prefill, batched decode and grouped FP32 verification across
  multiple local KV heads; retain the existing single-head implementation.
- Replace explicit TP4 quantized-projection gates with native layout capabilities
  and extend the measured projection configurations for the TP1/TP2 layouts.
- Validate arithmetic against independent references, CUDA Graph replay and
  route selection; separate prefill, pure decode and end-to-end serving results.
- Keep controls and candidate settings matched. Do not benchmark with eager mode.
- Promote defaults only after the affected quality and performance checks pass.

2026-09-21: Source audit found TP1/TP2 E4M3 C1 planning selects partition 1024 at
32K+, while native XQA rejects large partitions when Hkv>1. The 75T prefill,
E4M3 batched XQA and grouped FP32 gates require Hq=6/Hkv=1; NVFP4 QPN2 and FP8
QPN8/prefill additionally have explicit TP4 gates. Source policy checks are
not GPU performance or quality measurements.

GPU use authorized by the user: stop both existing services and use GPU 0–7.
Stopped-service launch records and raw validation artifacts are retained in
the task-local `tp-generalize-20260921/artifacts` directory. No unrelated service
code, model files, or canonical checkout changes belong to this task.

## Implementation and validation in progress, 2026-09-21

Draft PR: <https://github.com/1CatAI/1Cat-vLLM/pull/666>. The changes have not been
promoted to main. All paths below are relative to the artifact directory above.

The attention dispatcher now admits GQA6/D256 with multiple local KV heads.
The 75T prefill and compensated long-context kernels retain their original
six-head arithmetic, with ordered per-head calls. XQA's generic kernels already
carry KV-head strides; their admission and partition planner now agree. E4M3
scalar exact conversion/PV unrolling is independent of TP size and batch size.
`VLLM_FLASH_V100_E4M3_SCALAR_FAST` defaults on, preserving the old TP2 flag as a
fallback override and requiring rebuilt native revision 3.

NVFP4 QPN2, channel/block FP8 QPN8 and AWQ/FP8 dense prefill select compatible
local projection geometry instead of TP4. Workspace capacity follows the local
matrix size and preserves old allocations referenced by raw pointers. Shared
QPN2 codes are being tested explicitly; their global default is still unchanged.

The LM-head follow-up admits aligned local vocabularies and FP32 logits on one,
two and four ranks. Wider local vocabularies retain 64 candidates per 62,080
rows, preserving the support of the original vocabulary chunks. The packed
FP16 rerank has a separate native 64-candidate contract; the wider route uses
the existing FP32 indexed-dot implementation.

Completed focused checks (Torch 2.10.0+cu128, CUDA 12.8, V100-SXM2-32GB):

- 36 E4M3 XQA/grouped tests, Hkv=1/2/4, C1/2/4/8/16/32 and 32K/128K/256K,
  changed-input/length CUDA Graph replay against FP64: passed.
- 17 scalar native GPU tests, including all byte encodings, bitwise output and
  FP32 intermediate-state parity, and graph replay at 256K: passed. Three stale
  library admission tests pass separately. Initial test refactoring introduced
  an undefined mock variable; fixed, without changing native arithmetic.
- Six built-in long/scalar multi-head graph cases at 262144: passed.
- Four Q8192/KV262144 prefill cases (B1/B2, Hkv2/Hkv4): every element matches
  separate calls of the original single-KV-head kernel. Initial FP64 bounds
  incorrectly borrowed the tighter v37 tolerance: the original 75T operator
  itself has roughly 0.25% relative L2 on these random inputs. The retained gate
  requires bitwise parity with that operator and independent relative L2<0.007.
  This is not evidence of zero error or whole-model quality acceptance.
- 27 channel-FP8 QPN8 graph cases, local TP1/2/4 projection geometry and
  M8/16/32, force route selection without a dense workspace: passed.
- Shared/dual QPN2 layouts: six real projection families for TP1 and TP2,
  M1/8/16/32/33/8192, ordinary and gated output, all bitwise equal after changing
  inputs in graphs. `nvfp4-tp{1,2}-shared.json` records timings. Small gated M8
  shared readers can cost 6–9%; M32 generally improves. This is operator-only.
- LM-head graph checks on 36 retained actual hidden-state inputs/shard layouts:
  top-21 IDs match full FP32 logits; maximum indexed-dot error versus FP64 is
  2.812e-6. Raw cuBLAS FP32 logits can differ by about 4.8e-4 due to reduction
  order; exact token/logit parity must not be asserted for the complete model.
  See `lm-head-local-vocab-r3.json`. An initial harness omitted the service's
  BF16-checkpoint-to-FP16 conversion; the corrected harness applies it.
- Policy checks: attention 194, planner 17, quantization/scalar 57, LM-head/AWQ
  23 pass. Broader final checks remain pending as implementation continues.

The exact scalar/PV graph microbenchmark on two distinct TP2 KV layers measures
2.034 to 0.876 ms at 4K and 89.885 to 45.352 ms at 256K, with bitwise outputs.
These are sums of two attention operators, **not** emitted-token latency.

## Serving status and rejected setups

The originally authorized services were stopped. Another task subsequently
started a service on GPU0–3; leave it untouched. This task uses GPU4–7.
`launch_endpoint.py` records owned process IDs, ports, flags and isolated caches.
No private DSO/preload overlays are used. Native extensions and the bundled
FlashQLA extension were built from this worktree with CUDA 12.8.
The full packaging command reached Rust after native compilation but could not
find a Rust compiler. The first attempt also lacked `patchelf` on PATH; the
native rebuild supplied the existing environment's `patchelf`. The installed
environment provides the unchanged Rust dependency; a complete new wheel is
not yet claimed.

TP1 no-MTP, E4M3 KV, chunk8192, maxseq32 and memory utilization 0.92 cannot fit
the 27B model at max length262144: weights load 19.67 GiB, available KV5.46 GiB,
required KV8.38 GiB. This is a capacity failure, not successful 256K admission.
TP1 is being checked at max length131072, with CUDA Graph enabled. Natural-EOS
arithmetic, translation and coding smokes pass. `vllm bench serve` C1–C32,
2048 input/256 output per request, completes all requests; C32 reaches only
17 simultaneously resident sequences with the current coarse cache pages.
Report client concurrency separately from worker residency. Fixed-length
synthetic bench outputs use ignore-EOS solely for timing, never quality scoring.

TP2 DFlash2, max length262144, shared QPN2, E4M3 target KV, FP16 draft KV and
maxseq32 starts with full target/draft CUDA Graphs. A fixed random sample of
32 GSM8K items is running with official sampling, thinking enabled, natural
EOS and a 16384-token ceiling; full responses and cutoff counts are retained.
The initial endpoint is the control before the local-vocabulary head extension.
Strict numeric extraction has already exposed a semantically correct mixed-unit
answer that it marks wrong; preserve raw answers and distinguish extractor
scores from a claim of numerical corruption.

Next: complete paired model quality and long-context checks, validate TP2/TP4
serving concurrency with `vllm bench`, inspect remaining geometry-specific gates,
publish the implementation/results and promote only accepted defaults.

## Additional findings, 2026-09-21 11:40 CST

- The TP2 shared-QPN2 control completed all 32 sampled GSM8K questions: strict
  extraction 28/32, with one answer cut off at 16384 tokens. One strict failure
  is a correct 1128-minute answer reformatted as 18h48m; other failures include
  ambiguous question interpretations. Raw answers remain in
  `tp2-before-head-gsm32.jsonl`. This is not an untruncated quality pass.
- TP2 C4 initially died with CUDA OOM while allocating a 206 MiB gated-prefill
  temporary, with 204.5 MiB free. CUDA Graph capture had consumed 1.95 GiB and
  the existing SM70 startup policy reserves no graph memory. No arithmetic
  overflow was reported. `tp2-before-head-c4.json` is an invalid speed result.
- With memory utilization reduced from 0.92 to 0.85 (same max length262144,
  chunk8192, maxseq32, graphs on), `tp2-head-r1` completed C2/4/8/16/32 without
  request failures. Total output throughput: 125.52/104.47/145.66/140.85/143.86
  tokens/s. These include prefill and are not pure decode. Native long-page
  widening below was not loaded by that server. Logs and detailed bench JSON
  preserve request timings and speculative acceptance.
- Source inspection disproved the old comments claiming the long kernel was
  compiled only for pages1648/3296. It already contains a runtime-page kernel;
  host admission now admits positive 16-aligned pages, advertised by a native
  capability operator. Stale libraries retain their previous page admission.
  The scalar compact map stores two pages per 1024-token partition and now
  accepts pages>=1024 instead of fixing page3296. Experimental manifest
  overrides retain their original qualified pages.
- Thirty long-context graph checks (Hkv2/4, Q1/3/8, pages1024/2048/3296/4096/8192,
  KV262144) pass against FP64. Four FP16/E5M2 multi-head graph regressions pass.
  Nine block-FP8 CUTLASS Q8192 tests cover TP1/2/4 output/down/gate-up layouts,
  including interleaved gated-SiLU, with changed inputs and FP64 references.
- Sixty focused quantization policy/head/workspace checks pass. Growing a
  workspace preserves all previously handed-out raw pointers. Existing split-K
  arithmetic and QPN2 shared-layout defaults remain unchanged.
- Native rebuild r3 completed `_C` and `_vllm_fa2_C`; installed atomically from
  this worktree's build output with build RUNPATH removed. No running process
  was asked to reload a different native file.

Current owned launches: `tp2-default-r3` (GPU5–6, port18522, DFlash, default
separate QPN2 weights, maxlen262144, memory0.85), and `tp1-dflash-r3` (GPU4,
port18521, DFlash, explicit shared QPN2, maxlen65536, memory0.88). The prior
TP1/TP2 endpoints have been stopped. GPU0–3 still belong to the other task.
Next: default-route E2E concurrency/quality, cold 256K, TP4 acceptance, and
matched performance evidence before promoting Draft #666.

Startup follow-up: `tp2-default-r3` failed capacity admission: separate QPN2
weights load19.54 GiB/rank, leave3.8 GiB KV at utilization0.85, while256K requires
6.0 GiB. `tp1-dflash-r3` instead failed during draft post-processing: the
checkpoint has no LM head, but the generic loader tried to build an approximate
1.19 GiB QPN8 copy of its uninitialized placeholder immediately before target
sharing. The loader now releases missing embedding/head placeholders after
checkpoint loading and before generic post-processing; checkpoint-owned weights
remain intact. Eight focused loading/sharing tests pass. The fresh TP1 retry
loads successfully (26.06 GiB weights) and has reached graph compilation;
capacity/serving acceptance is still pending. The fresh TP2 retry uses explicit
shared QPN2 until the separate/shared model acceptance is resolved.

Implementation checkpoint5810c3c8ce is pushed to Draft #666; commit hooks all
pass, including mypy after correcting the batch-aware workspace key type.
Attention policy checks228 passed. Native artifact SHA256:

- `_C.abi3.so`: b6fca82a75e6eb0a77ae31ec2ff59469ea59e7b6d4d2fe90c371b17e2ecadd65
- `_vllm_fa2_C.abi3.so`: 53d18b1a4a9f7cae81c938ad2b3986512b2d76ba468c20f8a46ccadb8629d530
- FlashV100: 4a1157b24e4eb75d8311149b81e62efdb2652eaaa1898a7f104d0e379eab11e3

### Graph replay and long-prefill follow-up

The TP1 DFlash placeholder fix is committed as `96e2f28b67`. The retried model
loads (26.06 GiB), but initial profiling leaves a negative KV budget (-3.58 GiB)
with chunk8192/maxseq32. TP1 DFlash serving has **not** passed; lowering max length
alone cannot fix this activation/weight capacity failure.

TP2 `tp2-shared-r4` completed vLLM bench C1/2/4/8/16/32 and GSM8K32 (30/32,
zero truncation, all natural stops, max_tokens32768). Its first 32768-token
request failed while creating the long-prefill cuBLAS workspace. The initial
memory profile skipped attention, so the KV allocator had not reserved the
75T workspace. The candidate now initializes the selected Q8000/Q8192 core
once during the attention memory-profile call; no TP-size condition is added.
Fresh `tp2-prefill-r5` startup confirms Q8192 workspace inclusion. Long-context
serving acceptance remains pending until that fresh run completes.

A separate operator CUDA Graph microbenchmark exposed dangling host addresses
in the existing native core: captured symbol copies referenced stack locals,
and later calls overwrote host-side tail metadata. The fix passes symbol values
as kernel arguments and retains immutable tail metadata per KV length/value
buffer. Four regressions pass for Q8000/Q8192, Hkv2/4 and B1/2, replaying an old
graph after another KV length is captured and all inputs change. Replay output
is bitwise equal to an ordinary invocation of the same arithmetic.

Q8192/KV262144 graph replay, including per-head copies, measured Hkv1/2/4 at
76.625/75.748/75.296 logical causal TFLOP/s (169.502/342.925/689.974 ms).
These are single-GPU operator results for the three local layouts, not TP
serving throughput. Raw data: `prefill-graph-r5.log`; failure retained
in `prefill-graph-local-heads.log`; passing trace `prefill-graph-r5.log`.

Measurement correction: vLLM bench's initial ready-check request reuses the
first benchmark prompt. All subsequent cold-prefill runs set
`--ready-check-timeout-sec 0 --num-warmups 0` and explicitly reset prefix cache.
Previous 2048-token cases are retained as originally measured, not relabeled as
proven cold-cache evidence. DFlash stream ITL measures chunk arrivals, so it
must not be inverted as per-token pure-decode TPS. Long-quality streaming now
records emitted token IDs per chunk and rejects error/unfinished streams.
