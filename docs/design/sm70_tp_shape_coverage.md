# SM70 attention and quantized projection coverage across TP sizes

## Scope

Extend the existing V100 acceleration to the local tensor layouts produced by
TP1, TP2 and TP4. Dispatch must follow dtype, head grouping, matrix dimensions,
alignment and supported arithmetic, rather than a tensor-parallel-size allowlist.
Preserve FP32 accumulation where required by the accepted E4M3/75T contract.

Integration: `onecat/main`. Base: `b711d5304525dfc0cca6bc8a0bb005f33fe1bbf8`.
Owned branch/worktree: `codex/v100-tp-generalize-20260921-021932` /
`worktrees/v100-tp-generalize-20260921-021932`.

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
`/home/ymzx/1cat-build/tp-generalize-20260921/artifacts`. No unrelated service
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
FlashQLA extension were built from this worktree with `/data/minimax-h3/cuda128`.
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
