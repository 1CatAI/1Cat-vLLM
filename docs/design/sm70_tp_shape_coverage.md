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

## Latest acceptance state: full FP32 candidate, 2026-09-21

This PR remains a draft. Removing TP-size admission gates is implemented, but
blanket default promotion is not accepted. In particular, full QK/PV FP32 is
currently 70-71T, below the requested 75T minimum. The earlier 75T measurement
must not be attributed to full FP32 accumulation.

| Check | Current evidence | Outstanding |
| --- | --- | --- |
| TP1/2/4 local attention/projection geometry | GPU operator and changed-input CUDA Graph checks pass | Whole-model results are separate |
| QK and PV accumulation | Both FP32 in candidate r8; FP16 inputs/intermediates/output | Recover 75T without reducing precision |
| TP1 27B + DFlash2 | Loader placeholder fix passes; weights load | 8192-token profiling exceeds available memory |
| TP2 27B + DFlash2, 256K | r5 cold 256K and natural-EOS retrieval pass | r5 uses QK FP16; not full-FP32 acceptance |
| TP4 27B + DFlash2, 256K | r8 starts with default separate weights and full graphs | 96-item quality, long context and serving concurrency running |
| Actual simultaneous decode | TP2 r5 C2/C4 measured; C8 queues | Do not relabel queued C8-C32 as resident decode |
| Shared QPN2 weight default | Operator equivalence passes | Paired model quality before changing default |
| 35B-A3B AWQ/FP8 migration target | No matching model found locally | Matched baseline still required; not claimed here |

Current server: TP4 on GPU4-7, target QUASAR-QAT Qwen3.8-27B NVFP4,
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
