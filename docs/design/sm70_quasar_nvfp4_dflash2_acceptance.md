# SM70 QUASAR NVFP4 DFlash2 acceptance

> Status: implementation, matched target/DFlash acceptance, and fresh-wheel
> release validation complete. All numerical, performance, quality, protocol,
> packaging, and memory findings below are measured on the four-V100 release
> host.

## Scope and frozen contract

This work admits the fully quantized
`QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` checkpoint on four V100 GPUs without
narrowing DFlash2 to `max_num_seqs=1`, a 512-token prefill budget, or one target
KV dtype. The measured practical contract is TP4, FP16 execution, Flash-V100,
FP8 E5M2 target KV, FP16 draft KV, prefix caching, Mamba alignment, a 256K
declared context, `max_num_batched_tokens=4096`, `max_num_seqs=4`, seven model
drafts, selector top-16, probabilistic draft sampling, and full target/draft
CUDA graphs. The final wheel gate measures cold prefill only through 64K, as
required by the release test plan.

The checkpoint differs materially from the earlier mixed NVFP4 baseline. All
attention, GDN, and MLP linears are checkpoint-native NVFP4. On Volta, the
accepted QPN2 and TurboMind kernels execute those weights as W4A16: they retain
the checkpoint's FP4 weights and group scales but consume FP16 activations.
They do not emulate the checkpoint's W4A4 activation quantizer. This is an
existing SM70 execution contract, not a property of speculative decoding, and
is why dataset scores rather than route logs are a mandatory release gate.

## Correctness root cause

The initial target-only smoke for `17 + 25` emitted a long run of `!`. A real
checkpoint oracle covered all six TP-local projection shapes. Five shapes
matched a dense dequantized-weight reference, while the GDN fused input
projection did not:

| Projection | TP-local shape | Old result |
|---|---:|---:|
| GDN `in_proj_qkvz` | K=5120, N=4120 | relative L2 about 0.5 |
| GDN output | K=1536, N=5120 | within FP16 tolerance |
| attention QKV | K=5120, N=3584 | within FP16 tolerance |
| attention output | K=1536, N=5120 | within FP16 tolerance |
| MLP gate/up | K=5120, N=8704 | within FP16 tolerance |
| MLP down | K=4352, N=5120 | within FP16 tolerance |

N=4120 is divisible by eight but not by the TurboMind converter's required
16-column physical layout. The converter silently produced corrupt values.
Padding weights and scales with zero rows to N=4128, executing the physical
shape, and cropping before bias reduced relative L2 to about `3e-4` with cosine
similarity rounding to 1.0 on every TP rank. The exact arithmetic smoke then
returned `42`.

The implementation records logical and physical output widths in the prepared
linear state, uses the physical width for warmup and execution, and crops to
the logical width before bias, reshape, or a row-parallel collective consumes
the result. Padded fused gate-SiLU remains rejected because no current model
needs that unsupported layout.

## QPN2 all-projection verification path

The same real-weight oracle was extended to the six QUASAR projection shapes
at M=1 and M=8 on all four TP shards. QPN2 passed every cell. In the final M=8
run, the worst observed relative L2 was `5.20e-4`; cosine similarity was at
least `0.99999988`. Depending on the projection and TP shard, QPN2 was
`1.29x--3.94x` faster than the corrected TurboMind path. The N=4120 GDN fused
input projection used a physical N=4128 buffer on all four ranks. A forced
M=16 run separately covered the ephemeral dense-prefill algorithm; its output
differed from the established TurboMind reference by at most one FP16 ULP.
M=16 timing is deliberately not a production claim because the actual
large-M threshold is 1024.

The runtime admission is intentionally exact: compressed-tensors NVFP4,
recognized Qwen3.8 TP4 projection suffix and packed shape, DFlash block 8,
selector top-16, PP1, and no DBO. It does not require one resident sequence,
E5M2 target KV, or a particular prefill chunk. Unsupported shapes retain the
corrected TurboMind path.

A production-shape Split-K/accumulator sweep found only about 0.15 ms of
additional whole-model opportunity. That reduction-order change is not being
promoted: the practical round is already below the target and preserving the
current quality candidate is more valuable than a marginal timing claim.

## Scheduler and prefill isolation

MRV2 DFlash owns a separate K+1 query batch. Its K mask slots must size the
draft `InputBuffers`, hidden-state buffer, context-position buffer, and
per-group slot mappings; they must not be subtracted from the target
scheduler's prefill budget. A target-only scheduled-token property now returns
zero extra target slots for `method=dflash` while retaining the old accounting
for Eagle, MTP, and `dflash_ddtree`.

The DFlash speculator independently raises draft buffer capacity to at least
`max_num_seqs * (draft_block + 1)`. Its runtime and captured CUDA graph also
use a DFlash-owned persistent slot-mapping buffer rather than borrowing the
target `BlockTables` allocation. This closes the out-of-bounds case where
`max_num_seqs * 8` exceeds `max_num_batched_tokens`. The SM70 interactivity
defaults no longer force `max_num_seqs=1` or
`max_num_batched_tokens=4096`; normal server defaults and every explicit user
override survive.

With prefix caching and Mamba alignment enabled, two cold token-ID repetitions
measured 4,138/4,155 token/s at 32K and 3,612/3,607 token/s at 63,488 tokens.
A later cold chat-contract confirmation, with a marker before the repeated
body to prevent accidental cross-case prefix reuse, measured:

| Prompt | Prefill throughput |
|---:|---:|
| 32,724 tokens | 4,049 token/s |
| 63,482 tokens | 3,607 token/s |

These are slightly above the same-machine mixed-NVFP4 DFlash2 history of about
4,039 and 3,567 token/s, so enabling DFlash2 no longer removes the accepted
large-M QPN2-packed prefill path.

## Complete-round evidence

Graph-node tracing is diagnostic overhead, not a release timing. Under the
same Nsight instrumentation, the old mixed target and QUASAR all-QPN2 target
had nearly identical stable target-replay intervals (about 21.31 and 21.16 ms)
and rank-max GPU service (about 19.39 and 19.19 ms). This establishes that the
additional checkpoint-native NVFP4 projections are no longer an unexplained
gap.

The profiler-free practical service then ran five identical, warmed,
single-request measurements on the final candidate:

`17.543, 17.545, 17.517, 17.624, 17.532 ms/round`

The mean is `17.552 ms`, with a `17.517--17.624 ms` range. The requested
sub-20-ms gate therefore passes without binding server capacity to B1.

## Quality gates

The following gates were applied before packaging:

1. target-only and DFlash2 use the same QUASAR checkpoint, sampling contract,
   dataset order, seed policy, 16K output cap, practical cache settings, and
   score harness;
2. executable MBPP with long xhigh reasoning and token-level WikiText
   likelihood must not show a meaningful loss versus target-only;
3. BFCL-style simple, multiple, parallel, and irrelevance cases plus strict
   JSON Schema cases must be compared with the target-only service;
4. prefix-hit follow-ups and mixed-length batch 1/2/4 runs must show no stale
   GDN state, KV corruption, graph residue, or memory growth;
5. the DFlash2 seven-position acceptance vector must be reported in full.

Target-only and DFlash both pass six of eight fixed BFCL cases. Both misses are
the same two parallel fixtures, where each arm emits one valid call instead of
the fixture's expected two calls; simple, multiple, and irrelevance cases all
pass. Target-only passes all four fixed JSON Schema cases at a 2K budget.
DFlash passes three at 2K; the remaining complex schema consumes the 2K budget
in valid reasoning and emits no content, then passes and stops naturally at
2,655 tokens with a 4K budget. This is a budget-sensitive reasoning trajectory,
not malformed JSON or a speculative-decoding protocol error.

The first matched MBPP-32 run uses temperature 1.0, top-p 0.95, top-k 20,
`xhigh` reasoning, request seed 0, and a 16K natural-EOS cap. Target-only
passes 32/32 independent tests; accelerated DFlash2 passes 31/32. EvalPlus
excludes the one row without an independent contract and scores target-only
versus DFlash2 Base `31/31` versus `30/31` and Plus `29/31` versus `27/31`.
The only base discordance is one valid but incorrectly terminating comb-sort
implementation. With one discordant pair, the exact McNemar p-value is 1.0;
this run is evidence to investigate rather than evidence of a population
regression.

A full same-seed QPN2 rollback arm isolates the newly widened target operator
from the rest of DFlash2. It also scores 31/32, EvalPlus Base 30/31, and Plus
27/31, but fails a different row after a 16K reasoning-length termination.
Thus disabling QPN2 does not recover aggregate quality. It reduces mean steady
decode from 251.93 to 225.44 token/s and aggregate output throughput from
228.75 to 208.06 token/s. Disabling QPN2 therefore does not repair the sampled
failure but costs about 11.8% steady decode throughput.

At request seed `20260925`, target-only and all-QPN2 DFlash2 both score MBPP
32/32, EvalPlus Base 31/31, and Plus 28/31, with identical Plus failure rows.
Across the two predeclared seeds, target-only scores Base 62/62 and Plus 57/62;
DFlash scores Base 61/62 and Plus 55/62. There is only one paired discordance
for each score class, which is not significant evidence of a population
regression. The second seed's exact score and failure agreement also rejects
the hypothesis that QPN2 systematically degrades code quality.

On eight matched 2,048-token WikiText prompts, target-only and DFlash emit the
same first output token and text in every case. The maximum prompt PPL
difference is `0.00851`; the maximum per-token prompt/output log-probability
difference is `0.938/0.190`, within the predeclared model-quality bounds of
`1.0/0.25/0.01` for prompt logprob, output logprob, and PPL.

Structured-output batch gates pass target-only and DFlash at B1/B2/B4
(`42/42` each). The final DFlash protocol suite passes tool-chain 6/6, nested
and escaped JSON schemas, and stream/non-stream parity 3/3. Explicit-history
Responses replay passes; server-side Responses storage remains disabled
because the current store has no eviction policy and is not required for tool
calling. Prefix-state follow-ups pass 5/5 with and without thinking for both
arms.

A 96-request mixed B4 structured-output stress passes 96/96. V100 memory is
28,855 MiB per rank before and after the run; the subsequent first long-prefill
allocation leaves only a stable 4 MiB one-time increase. Logs contain no OOM,
illegal access, traceback, stale-buffer symptom, or graph replay corruption.

The earlier gross repeated-punctuation failure is closed by the N=4120
physical-layout repair. No quality conclusion is inferred from that smoke or
from greedy token equality.

## Rollback and diagnostics

- `VLLM_SM70_NVFP4_QPN2=0` retains corrected TurboMind execution.
- `VLLM_SM70_NVFP4_QPN2_PREFILL=0` retains the established large-M fallback.
- `VLLM_SM70_DFLASH2_VERIFY_FASTPATH=0` remains a diagnostic rollback, not a
  production quality workaround.
- Missing QPN2 operators or an unrecognized shape fall back explicitly; no
  padded logical width may silently reach a converter that cannot represent
  it.

The real-checkpoint oracle is
`benchmarks/kernels/benchmark_sm70_quasar_nvfp4_oracle.py`. The serving-quality
reporter now discovers every speculative position dynamically instead of
truncating DFlash2 at four positions. Long quality runs also retain each
completed response in an fsync'd partial JSONL sidecar before final summary
metadata is assembled.

## Fresh 1.5.0 wheel release proof

The CPython 3.12 SM70 wheel is
`1cat_vllm-1.5.0-cp312-cp312-linux_x86_64.whl`, size `147,980,225` bytes,
SHA256 `2a4d6bee4e19d315b142f2c563059f3064ddeeca563a6bdc828c33e1073c825b`.
It was built with the CUDA 12.8 toolkit and version metadata `1.5.0`. All ten
native libraries have empty RPATH/RUNPATH and resolve their Torch/CUDA
dependencies with the normal runtime library set. The wheel retains the
release dependency bound
`prometheus-fastapi-instrumentator>=8.1.0,<9.0.0`.

On the four-V100 host, the prior environment was cloned to a distinct Conda
prefix, the old package was uninstalled, and this wheel was installed without
dependencies or a source/PYTHONPATH overlay. `pip check` passes after satisfying
the package's existing `setuptools>=77.0.3,<81.0.0` bound, and `vllm.__file__`
resolves inside the new prefix. The core extension, QPN2 operators, SM70
compact sampler, Flash-V100, and FlashQLA all load from that prefix.

The source-independent TP4 server used FP16 execution, a 262,144-token declared
context, E5M2 target KV, automatic FP16 draft KV, prefix caching with Mamba
alignment, `max_num_batched_tokens=4096`, `max_num_seqs=4`, tool/reasoning
parsers, and target plus draft CUDA graphs. No private fast-path environment
variables were supplied. Startup logs automatically enabled the quality-audited
DFlash2 defaults, loaded all checkpoint-native NVFP4 projections, selected
QPN2 for compatible decode projections and QPN2-packed ephemeral prefill for
M>=1024, and activated the exact q8 grouped verifier.

The final targeted CPU contract suite passes 159 tests under the wheel's
Torch 2.10/CUDA 12.8 ABI. Ruff lint, Ruff format, and `git diff --check` pass
for every changed Python source and test file.

The first 1K request against an empty Triton cache compiled seven bounded
request-shape kernels and measured `23.055 ms/round`. With the cache warm, four
identical requests measured `17.641`, `17.663`, `17.638`, and `17.637`
ms/round: mean `17.645 ms`, range `17.637--17.663 ms`. The first and steady
runs had the same 512 output tokens, 203 draft rounds, and `2.522` emitted
tokens per round; the cold-only difference is compilation latency rather than
an acceptance or execution-route change.

Unique cold long-context requests measured:

| Prompt | Prefill | Complete round | Pure decode | Emitted/round |
|---:|---:|---:|---:|---:|
| 32,724 tokens | 4,038.5 token/s | 18.827 ms | 134.9 token/s | 2.540 |
| 63,482 tokens | 3,596.5 token/s | 20.358 ms | 105.7 token/s | 2.186 |

Both requests contained the required suffix marker, had no replacement
characters, and passed repetition/output-health checks. Their prefill rates are
within about 0.3% of the accepted source measurements, so the wheel does not
lose the promoted prefill path. These ordinary technical-writing prompts have
lower acceptance than the historical easy prompt; complete-round latency, not
their task-dependent decode throughput, is the release comparison.

Two identical 9,994-token prefix requests returned the same healthy output.
Wall time fell from `2.624 s` cold to `0.311 s` cached and TTFT from `2.506 s`
to `0.210 s`; the server recorded 9,888 cached prompt tokens. A four-way
concurrent strict-JSON test passed 16/16 requests. Per-rank GPU memory changed
by at most 2 MiB, and the log contained no traceback, OOM, illegal access,
assertion failure, stale-buffer symptom, or graph corruption. Graceful shutdown
returned GPUs 0--3 to 6 MiB each.

Fresh-wheel protocol probes returned `42.0`, emitted the requested
`get_weather({"city":"Guangzhou"})` call, and produced a schema-valid
`{"name":"小林","age":27}` object. The full client also passed the six-step
tool chain, nested/escaped structured output, stream/non-stream parity, and
explicit-history Responses replay. Its aggregate flag remains false only for
`previous_response_id`: server-side Responses storage is intentionally disabled
because the current store has no eviction policy. The first stored response is
valid and the expected follow-up receives HTTP 404; ordinary Chat Completions
tool calling and explicit-history Responses are unaffected.
