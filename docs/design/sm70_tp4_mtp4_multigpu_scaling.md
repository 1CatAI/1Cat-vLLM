# SM70 TP4 MTP4 Multi-GPU Scaling Ledger

## Scope

This ledger covers Qwen3.6-27B-AWQ target-verifier MTP4 decode on four V100
SM70 GPUs with TurboMind, Flash-V100, CUDA graphs, and pairwise NVLink `NV2`.
It excludes Marlin and eager-mode experiments.

## Accepted Evidence

The matching `i768/o64`, M=5 FULL CUDA-graph traces are:

- TP2: `bench_results/mtp4_current_binary_20260712/nsys_tp2_current_m5_i768/`
- TP4: `bench_results/mtp4_current_binary_20260712/nsys_tp4_current_m5_i768/`

CUDA-event target-forward p50 is `22.670 ms` at TP2 and `18.433 ms` at TP4.
The exact all-real AWQ M=5 portion improves `9.085 -> 5.827 ms`; subtracting
it leaves the non-AWQ chain at `13.585 -> 12.606 ms`, only `7.2%` faster.

| component | TP2 ms | TP4 ms | TP2 -> TP4 | conclusion |
|---|---:|---:|---:|---|
| TurboMind AWQ M=5 | 9.085 | 5.827 | -35.9% | improved but not ideal |
| Flash-V100 target decode | 2.617 | 2.588 | -1.1% | primary fixed compute chain |
| TP communication | 2.271 | 2.522 | +11.1% | many small synchronization-bound P2P reductions |
| non-AWQ residual (event) | 13.585 | 12.606 | -7.2% | P0 multi-GPU defect |

NVLink is not the bandwidth limiter here. The target graph has 128 small
`[5,5120]` row-parallel collectives per rank, while attention is replicated on
each rank and underfilled after head sharding.

## Flash-V100 Static-Graph Root Cause

Nsight shows every M=5 Flash partition kernel uses a fixed workspace grid:

| topology | grid | CTAs per layer | mean partition-kernel duration |
|---|---|---:|---:|
| TP2 | `(5, 12, 257)` | 15,420 | about 159 us |
| TP4 | `(5, 6, 257)` | 7,710 | about 158 us |

The exact graph template is `D=256, PARTITION_SIZE=1024`, not 256. The 257 Z
slots correspond to a 263,168-token workspace. At input length 768 there is
only one live partition: TP2 has 60 useful CTAs (`5*12*1`) and TP4 has 30
(`5*6*1`), both below the V100's 80 SMs. The other grid slots return early.

This corrects an invalid first microbenchmark that used a 65,792-token
workspace and therefore captured `PARTITION_SIZE=1024, gridZ=65`. Its P256
numbers must not be used for this MTP graph.

## Rejected: Persistent Partition-Grid Cap

The default-off `VLLM_FLASH_V100_DECODE_PARTITION_GRID_CAP` prototype reduced
the graph's Z grid and made each CTA process a Z-stride of partitions. It was
bitwise exact at 768, 4096, 65536, and 262144 tokens, but failed performance.

The correct NSYS micrograph proves route hit and rejection:

| route | partition grid | regs/thread | partition kernel us | decision |
|---|---|---:|---:|---|
| baseline | `(5,6,257)` | 44 | 115.312 | reference |
| cap 8 persistent | `(5,6,8)` | 66 | 117.112 | reject |

The grid reduction removes empty CTAs but the extra persistent control and
register growth more than consume the saving. A fixed cap is also unsafe for
the complete context range: at 65K, cap 64 regresses `1.575 -> 1.978 ms`; at
262K, cap 192 regresses `5.994 -> 7.575 ms`. The prototype was removed from
the extension and must not be reintroduced as a global environment override.

Artifacts:

- `bench_results/mtp4_current_binary_20260712/flash_v100_m5_micro/persistent_grid_cap_v3_tp4_m5_real_graph_sweep.json`
- `bench_results/mtp4_current_binary_20260712/flash_v100_m5_micro/nsys_grid_cap_v3/`

## Active P0: Context-Bucketed Attention Graphs

The viable direction is not a fixed global grid cap. A short-context MTP graph
must use a smaller workspace/partition policy and a distinct CUDA-graph key,
or update graph kernel launch parameters through a supported API. The existing
256K graph remains the long-context fallback. This must first prove that graph
selection is based on a safe context bucket, does not exhaust graph memory, and
preserves the exact scalar output before any full-model run.

### Implemented Candidate: Bounded Short-Context Graph

The first implementation is deliberately opt-in through
`VLLM_SM70_MTP_CONTEXT_BUCKETS=4096`; the empty/default setting retains the
pre-existing graph only. It adds `attention_context_bucket` to the CUDA-graph
descriptor, so a short graph cannot alias the 256K graph. Dispatch selects the
smallest configured bucket that is at least the actual MTP context length; an
over-bucket request falls back to the ordinary long graph. Capture receives
the bucket only as the Flash-V100 workspace capacity cap. The stable full block
table pointer and runtime bounds remain unchanged.

The graph-capture metadata unit test covers the bounded workspace policy and
the dispatcher unit test covers selection/fallback. A direct physical-V100
M=5 Flash graph replay at `i=768` gives the following kernel evidence:

| workspace policy | graph grid | workspace slots | mean ms | relative |
|---|---|---:|---:|---:|
| baseline P1024 | `(5,6,257)` | 263,168 | 0.12426 | reference |
| bucket P256 | `(5,6,16)` | 4,096 | 0.05319 | 2.34x |
| bucket P512 | `(5,6,16)` | 8,192 | 0.07768 | 1.60x |

P256/P512 are replay-stable but not bitwise equal to P1024: the different
partition/reduction order gives maximum absolute error `1.2207e-4`. Therefore
the full-model validation must use the normal official sampling and long
natural-stop quality gate, not only a layer-equality assertion.

Same-source full-model diagnostic evidence uses Qwen3.6-27B-AWQ, TP4,
TurboMind, Flash-V100, MTP4, full-GDN, CUDA graphs, 256K context, dynamic
vocabulary, `i768/o64`, and official sampling with `ignore_eos` solely to
hold the trajectory length constant. The candidate captures two FULL graphs
(`0.16 GiB`, 432 custom-all-reduce addresses) instead of one (`0.12 GiB`, 304
addresses). It selects the bounded graph exactly once on each rank.

| metric | 256K graph only | P4096 short graph | delta |
|---|---:|---:|---:|
| interval target verifier forward | 17.720 ms | 15.964 ms | -1.756 ms (-9.9%) |
| steady TPOT | 10.993 ms | 10.616 ms | -3.4% |
| steady decode | 90.963 tok/s | 94.199 tok/s | +3.56% |
| MTP mean acceptance length | 4.1875 | 4.1875 | exact |

Both arms emitted exactly the same 64 token IDs and text. This proves the
short diagnostic trajectory, but it is not a release-quality result because
EOS was suppressed. Keep the graph bucket opt-in until the natural-stop long
output gate passes, including the transition from P4096 to the 256K fallback.

Artifacts:

- `bench_results/mtp4_current_binary_20260712/flash_v100_m5_micro/workspace_bucket_p1024_p256_p512_i768_gpu0_20260712.json`
- `bench_results/mtp4_current_binary_20260712/context_bucket_tp4_m5_i768/baseline_current_source/route_off_i768_o64.json`
- `bench_results/mtp4_current_binary_20260712/context_bucket_tp4_m5_i768/candidate_clean/route_on_i768_o64.json`

### Current P256 Validation: Quality-Pass, Opt-In

The previous P256 natural-stop result was rejected by a narrow `-2.094%`
acceptance miss against an older source baseline. The current source must be
judged only against a freshly captured no-bucket baseline. The fresh paired
evidence below uses the same model, TP4 GPUs 0--3, TurboMind AWQ,
Flash-V100, MTP4, 256K model length, dynamic vocabulary asset, official
sampling (`temperature=1.0`, `top_p=0.95`, `top_k=20`, seed `20260620`), and
non-eager CUDA graphs.

The M=5 fixed-length diagnostic is the low-overhead target-forward authority:

| metric | no bucket P1024 | P4096 + P256 | delta |
|---|---:|---:|---:|
| target verifier forward | 17.784 ms | 16.065 ms | -1.719 ms (-9.67%) |
| steady TPOT | 10.472 ms | 9.423 ms | -1.049 ms (-10.02%) |
| steady decode | 95.490 tok/s | 106.121 tok/s | +11.13% |
| MTP mean acceptance length | 4.200 | 4.200 | exact |
| 64 generated token IDs | identical | identical | pass |

The P256 graph-node trace confirms the intended utilization change. Node-trace
wall times are composition evidence only; kernel grid and per-launch duration
are direct route evidence.

| item | P1024 long graph | P256 bounded graph | change |
|---|---:|---:|---:|
| partition kernel | `D256/P1024`, `(5,6,257)` | `D256/P256`, `(5,6,16)` | separate graph, no in-place mutation |
| useful partition CTAs at i768 | 30 | 90 | 3x more useful CTA work |
| partition-kernel mean | 157.715 us | 56.069 us | 2.81x faster |
| Flash critical path, graph-node trace | 2.587 ms | 0.983 ms | -1.604 ms (-62.0%) |
| Flash TP GPU-sum, graph-node trace | 10.339 ms | 3.916 ms | -62.1% |

Natural-EOS `macos_6k_code` validation is the quality authority. Outputs are
not required to have identical sampled token IDs because P256 changes the
attention partition/reduction order; the gate instead requires healthy text
and no more than 2% acceptance loss against the paired fresh baseline.

| route | EOS tokens | mean acceptance | relative acceptance | TPOT | steady decode | text-health |
|---|---:|---:|---:|---:|---:|---|
| no bucket P1024 | 5,994 | 4.03838 | reference | 7.837 ms | 127.597 tok/s | pass |
| P4096 + P512 | 5,903 | 4.03418 | -0.104% | 7.672 ms | 130.353 tok/s | pass |
| P4096 + P256 | 5,843 | 4.11989 | +2.018% | 7.439 ms | 134.436 tok/s | pass |

For all three natural outputs, the health check found no replacement
characters, no bad markers, no pathological repeated windows, and complete
HTML/CSS/JavaScript code fences with closing `</style>`, `</script>`, and
`</html>` tags. P256 is therefore the current strongest quality-passing
short-context candidate, with `+5.36%` natural-stop throughput versus the
fresh baseline.

Decision: retain the mechanism as opt-in for now
(`VLLM_SM70_MTP_CONTEXT_BUCKETS=4096` and
`VLLM_SM70_MTP_CONTEXT_BUCKET_PARTITION_SIZE=256`). Do not make it a global
MTP default until the `>4096` long-graph fallback and additional natural
prompts pass under the same source/quality rules.

Artifacts:

- `bench_results/mtp4_current_binary_20260712/context_bucket_tp4_m5_i768/baseline_refresh/`
- `bench_results/mtp4_current_binary_20260712/context_bucket_tp4_m5_i768/p256_profile_refresh/`
- `bench_results/mtp4_current_binary_20260712/context_bucket_tp4_m5_i768/p256_nsys/`
- `bench_results/mtp4_current_binary_20260712/context_bucket_tp4_m5_macos6k_p512_refresh/`
- `bench_results/mtp4_current_binary_20260712/context_bucket_tp4_m5_macos6k_p256_refresh/`

### Rejected: Raw TP4 M5 No-End All-Reduce Barrier

The M=5 verifier has exactly 128 FP16 `[5,5120]` row-parallel reductions:
one attention/GDN output reduction and one MLP-down reduction for each of the
64 dense layers. The normal payload has 3,200 packed 16-byte values, so the
512-thread reduction launches only seven CTAs on each 80-SM V100. This is a
latency/synchronization problem, not an NVLink bandwidth problem.

An experimental raw no-end-barrier kernel improved the synthetic 128-call
CUDA-graph microbenchmark from `1.485 ms` to `1.374 ms` (`-7.5%`, about
`0.88 us` per reduction) and preserved the current custom reduction order for
exact-int, rank-marker, and model-like inputs. It is rejected for model use:
the compiled M=5 graph immediately reuses the all-reduce input allocation as
RMSNorm output storage. A faster rank can therefore overwrite an input while a
slower peer still reads it. The graph buffer registry explicitly allows those
addresses to be reused, so the next all-reduce start barrier cannot provide a
valid source-lifetime guarantee.

Do not revive this as an environment-only path. Any future deferred-end design
must carry an explicit source-lifetime token through the compiler graph, wait
before the first possible source reuse, and pass an overwrite-after-allreduce
stress test before a model run. Even if made safe, its credible target-forward
ceiling is only about `0.11 ms` per M=5 verifier round, so it cannot be the
primary solution for the `12.606 ms` non-AWQ TP4 residual.

Artifacts:

- `bench_results/mtp4_current_binary_20260712/tp4_m5_relaxed_ar_micro/baseline.json`
- `bench_results/mtp4_current_binary_20260712/tp4_m5_relaxed_ar_micro/relaxed.json`

## Long-Output MTP4 Decay: Verifier First, Acceptance Later

The long macOS coding run is a 10-second-window diagnostic, not a replacement
for a dedicated long-context trace. It is nevertheless sufficient to separate
the two terms in the MTP throughput equation:

`output_tokens_per_second = mean_acceptance_length / MTP_round_seconds`.

The route used TP4, MTP4, official sampling, TurboMind AWQ, Flash-V100, and
CUDA graphs. `route_on_macos6k_mtp4.log` continued past 9K generated tokens,
so its periodic engine metrics expose the length trend.

| generated-token region | gen tok/s | mean acceptance length | inferred MTP round ms | diagnosis |
|---|---:|---:|---:|---|
| about 2.9K | 125.7 | 4.31 | 34.3 | short-context reference |
| about 3.8K | 119.2 | 4.39 | 36.8 | verifier cost increasing, acceptance improves |
| about 6.3K | 108.0 | 4.33 | 40.1 | verifier-cost dominated decline |
| about 7.0K | 99.0 | 4.06 | 41.0 | verifier cost still dominates |
| about 9.1K | 77.0 | 3.35 | 43.5 | acceptance and verifier cost both hurt |

Thus the first roughly 20% throughput loss from 2.9K to 7.0K occurs while
acceptance changes only `4.31 -> 4.06`; the MTP round grows `34.3 -> 41.0 ms`
instead. This is primarily target-verifier cost, not poor drafting. The
profiled short-context composition explains why: target verifier forward is
`18.433 ms` or `71.8%` of measured GPU spans, whereas MTP draft total is only
`4.713 ms`.

From about 7.0K to 9.1K, the average round rises only `41.0 -> 43.5 ms`, but
acceptance falls `4.06 -> 3.35`. Holding the 7.0K round time fixed would yield
about `81.7 tok/s`; the actual `77.0 tok/s` means about four fifths of this
late-interval loss is acceptance, and one fifth is additional verifier cost.
All four draft positions weaken together in the final window
(`0.883/0.648/0.457/0.361`), so it is not solely the fourth MTP position.

The 4K boundary is a concrete verifier-path discontinuity. With
`VLLM_SM70_MTP_CONTEXT_BUCKETS=4096`, contexts through 4096 use the bounded
P256 workspace graph. The dispatcher intentionally returns the original 256K
FULL graph above that bound. The drop beginning around the corresponding
4K-output window is therefore compatible with the long-graph fallback, but
the 10-second windows cannot assign an exact millisecond share to that
transition.

Next measurement, after the interactive API is no longer needed, must keep
the route and sampling fixed and collect MTP CUDA-event bins at output-context
`2K, 4K-, 4K+, 6K, 8K, 10K`. Each bin must report target forward, target
logits, target rejection/sample, draft total, and all four acceptance
positions. In parallel, record dynamic-vocabulary shortlist/tail coverage per
draft position. This separates fallback attention work from a real long-code
draft-distribution or vocabulary-coverage loss without changing output
semantics.

Artifact: `bench_results/mtp4_current_binary_20260712/context_bucket_tp4_m5_macos6k/route_on_macos6k_mtp4.log`.

## Resolved: Streaming HTML Prefix Loss

The reported malformed macOS HTML was not a target-model, MTP, dynamic-vocab,
or P256-attention error. It was an OpenAI chat streaming error that affected
both TP4 MTP4 and no-MTP controls whenever `qwen3_coder` tool parsing was
enabled. Non-streaming output preserved `<meta>`, `<title>`, and `<style>`;
SSE output dropped the leading `<` and returned `meta`, `title>`, and
`style>`, which made rendered code look like natural language had entered the
HTML.

Root cause: `Qwen3CoderToolParser` delayed a trailing prefix such as `<` or
`<t` to see whether it completed `<tool_call>` or `<function=`. When the next
delta proved it was an ordinary HTML tag, the delayed prefix was never added
back to the SSE content. The fix preserves the pending prefix until it either
completes a real tool marker or is emitted with the first non-marker delta.
The real tool-call marker remains withheld.

Validation:

- Unit regression: `3 passed` for split `<meta>`, `<title>`, and repeated
  `<` prefixes in `tests/tool_parsers/test_qwen3coder_tool_parser.py`.
- Live no-MTP control reproduced the failure before the fix, proving that it
  was unrelated to speculative verification.
- Live patched TP4 MTP4, Qwen3.6-27B-AWQ, TurboMind AWQ, Flash-V100, CUDA
  graphs, dynamic vocabulary, P256, 256K, prefix caching, and
  `qwen3_coder` tool calling returned raw SSE deltas containing
  `<meta charset=...>`, `<title>`, and `<style>`.

Artifacts:

- Before fix:
  `bench_results/mtp4_current_binary_20260713/api_quality_regression/nomtp_macos1024_stream_seed20260620.sse`
- After fix:
  `bench_results/mtp4_current_binary_20260713/api_quality_regression/mtp4_p256_dynamic_macos1024_stream_fixed_seed20260620.sse`

This resolves the visible code corruption. It does not convert the dynamic
draft vocabulary from an experimental proposal route into an independent
model-quality acceptance result; future MTP quality work must still use
semantic/code-validity checks in addition to repetition checks.

## Closed Paths

| path | result | decision |
|---|---|---|
| TP4 XQA Tensor Core small-query route in a compact P256 graph | 0.0597 ms/layer vs scalar 0.0533 ms at 768 tokens; max diff to paged-prefill 0.01184 | closed only for explicit compact context-bucket graphs; the accepted P1024 long-graph XQA route is documented in `sm70_qwen36_27b_awq_mtp4_optimization.md` |
| fixed global persistent partition cap | route hit, bitwise exact, but `115.312 -> 117.112 us` under NSYS and large long-context regressions | removed and closed |
| decode-as-paged-prefill | prior M=8/seq4096 gate is `0.593 ms` vs scalar `0.173 ms` | closed; do not repeat as a short-MTP shortcut |
| TP4 custom all-reduce + RMS fusion | standalone gain did not reach compiled graph; endpoint regressed | closed |
| custom all-reduce block-count sweep | default 7 CTA remains fastest | closed |
| raw TP4 M5 no-end all-reduce barrier | synthetic graph `1.485 -> 1.374 ms`, but source allocation can be reused before peer reads finish | rejected: unsafe without compiler-enforced liveness |
