# DFlash2 capacity-tail graph recovery

## Problem and contract

The TP4 QUASAR DFlash2 request at 261888 input tokens and 256 output tokens
reaches the 262144-token capacity. Its normal q8 path remains approximately
linear in context length, but the final partial-verifier and non-speculative
q1 steps lose both full CUDA graphs and optimized attention dispatch.

The inherited tail flag was nested inside adaptive lookup initialization.
Ordinary seven-draft DFlash2 therefore never registered its q1/q6 graphs.
The original long-attention loader also admitted only q8 through 132096.

The workload is QUASAR Qwen3.8-27B NVFP4 revision
`d8e6fbfa3e3a78899b440222b827430045a05b44`, DFlash2 revision
`dedf8df68adfb1afeaf7b7480c0a0243108177b4`, four V100-SXM2-32GB GPUs, TP4,
FP16 activations, E4M3 target KV, FP32 state/logits, CUDA 12.8 and
Torch 2.10.0+cu128. Sequential requests use T=1, top-k=20, top-p=0.95,
seven probabilistic drafts, Flash-V100 and CUDA graphs. The server retains
max-num-seqs=4, max-num-batched-tokens=4096, memory utilization 0.8,
prefix caching and aligned Mamba caching. This is a bounded speed window,
not natural-EOS dataset scoring. Physical devices and local manifests are
recorded in the retained handoff.

Integration base: `fe67339ddf862df0441a4e844fc664d9cafdffd0`.
Inherited PR596 head: `9930ebe8e6bff1fb031740d5787f86f6193d2cd9`.

## Implementation

`VLLM_SM70_DFLASH2_TAIL_CUDAGRAPHS=1` registers target-only B1 q1 through q7
alongside q8. It does not depend on lookup augmentation and does not add
tail shapes to the drafter, other speculative methods or multi-request
uniform batches. Sequence-parallel configurations retain the previous path.
The flag remains disabled by default.

The grouped manifest can explicitly declare `max_context` and `query_rows`.
Without those fields the contract remains 132096 and q8. A selected
256K manifest uses `max_context: 262144` and `query_rows: [2,3,4,5,6,7,8]`.
The CPU bound controls graph selection; device row lengths remain
authoritative. Oversized or device-only bounds use the ordinary graph.
No attention truncation, capacity increase or arithmetic change is made.

`VLLM_SM70_DFLASH2_SCALAR_ATTENTION_MANIFEST` optionally loads the existing
compact E4M3 q1 library. The loader checks the DSO hash and the compact
six-head/lookup/256K contract. Its persistent FP32 numerator, maximum and
sum buffers are allocated before profiling and capture, outside the graph
pool. The operator requires the explicit 256K graph descriptor and the
admitted B1/H6/D256/page3296 E4M3 layout. Short q1 contexts below 128K,
windows, anchors and partition overrides retain the original operator.

GDN runtime code is unchanged. Capture intentionally uses PAD_SLOT_ID to
avoid mutating live state; tests verify that runtime preparation updates the
same captured storage for all q1-q8 widths under both `none` and `align`
Mamba cache modes. A comparison of capture placeholder values with live
values alone would incorrectly diagnose this protection as a state bug.

## Native prerequisites

The grouped candidate retains 80 logical splits, K16 compensated QK,
N32 online updates, compensated P/residual products and the complete FP32
workspace. The scalar candidate retains its 256 partitions of 1024 tokens
and complete FP32 partition buffers. No native source is changed here.

| Native candidate | Source SHA256 | Measured DSO SHA256 |
| --- | --- | --- |
| P layout, early store, E4M3 lookup | `eb7a85511f581fcd22cf13619c85ed2f42a8cbc8b216bb3e632bf448f6b820e1` | `9db33737adb880cd4198266ac4f29785ce3e709936aa47dcc79011a9bce4b811` |
| Compact scalar, lookup, 4 KiB shared reservation | `5e68578c0252c7525496abab98aef5ed5f774528ac1dab6c5e1c587912cfa632` | `6d2b2b1ec5e0501de9abf49a81670cca2fb2ee700be66365f98db118a62fee21` |

The existing builders are `build_sm70_grouped_attention_swizzle.py` and
`build_sm70_scalar_attention_candidate.py`. Keep their frozen source inputs
and manifests; new compilations have their own DSO hashes. The former uses
the admitted exact-LUT parent with `--probabilities --early-store`; the
latter uses `--share-kv-six-heads --e4m3-lut --compact-page-map
--dynamic-shared-bytes 4096`. Retained full workspace, tail and sanitizer
evidence is documented in the
[attention worklog](sm70_dflash2_attention_resources_20260910.md) and
[long verification worklog](sm70_dflash2_long_verify_curve_20260909.md).

Enable the measured runtime with explicit startup configuration:

```bash
export VLLM_SM70_DFLASH2_TAIL_CUDAGRAPHS=1
export VLLM_SM70_E4M3_LONG_ATTENTION_MANIFEST=/path/to/grouped-tail-manifest.json
export VLLM_SM70_DFLASH2_SCALAR_ATTENTION_MANIFEST=/path/to/compact-scalar/manifest.json
```

This change does not enable the separately frozen QPN2/context performance
harness by default. Both A/B arms retain that same harness, FA2 sidecar,
original FlashQLA prefill and q8 P-layout candidate. Legacy experimental
hooks which overwrite MAX_CONTEXT with 262152 are not part of this runtime.

## Verification

First final-code startup: one warmup per arm, then three alternating measured
requests per arm, at each of the four workload points below. All 16 request
pairs preserve token IDs, finish reason and acceptance records. Each table
entry is a median of request-average complete-round measurements.

| Input | Seed | Control ms | Tail graphs ms | Saved ms |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 0 | 15.808 | 15.804 | 0.004 |
| 131072 | 0 | 22.573 | 22.577 | -0.004 |
| 261888 | 0 | 35.464 | 31.390 | 4.073 |
| 261888 | 2 | 37.884 | 31.335 | 6.549 |

Seed 0 reaches q1 tails. Seed 2 also reaches q6 and q7; all four ranks
record actual FULL dispatch for these shapes. The first final-code startup
records 36 q1, 12 q6 and 12 q7 dispatches over the three measured seed-2
candidate requests across four ranks. Short-context runs have no tail
dispatches. Native binding counters describe capture/warmup; graph selection
and replay are separately observed.

An earlier final-runtime integration smoke uses three seeds and twelve
requests, all exact. Its warmed seed-2 result is 37.453 to 31.275 ms and
154.739 to 185.307 pure decode tokens/s. Its scalar admission was subsequently
restricted to the explicit long-context descriptor; the table above tests
that final restriction. These startups are not interchangeable controls.

Focused CPU checks cover actual graph initialization/dispatch, default-off,
non-DFlash, non-SM70, draft manager and sequence-parallel fallbacks, manifest
bounds, scalar layout/scale admission, and capture-to-runtime GDN buffer
refresh. All applicable pre-commit hooks pass. Expanded final startup
results and aggregate distributions are appended below when complete.

## Rejected paths and limits

The first private integration screen attempted to allocate the q6 workspace
on its first eligible call, which occurred inside capture. The assertion
failed before inference. The retained failure led to preallocating scalar
buffers and keeping grouped workspaces in the existing graph-safe loader.

Do not skip the drafter simply because the current q1 has no proposals:
the observed scheduler can return from q1 to q8 after rejection correction.
This change preserves scheduler accounting and all tail cost in the
complete-round denominator. It does not alter capacity guards or sampling.

The original 6.3-ms residual was not a pure non-attention stage: scalar
attention had been classified as miscellaneous work, and q1 consumes time
without incrementing the speculative-round counter. The q6 trace also shows
rank arrival skew inside collectives. Unprofiled endpoint latency, traced
kernel service and host time must remain separate.

The 7-ms full-attention and 22-ms complete-round goals remain unmet.
This scope recovers the boundary regression. It is not a complete numerical
audit across every model, quantization, topology or natural-output corpus.

## Three-startup result

Three independent final-runtime startups complete 96 requests and 48 exact
control/candidate pairs. Each startup and workload warms both arms before
three alternating measurements. The runtime file hashes are identical.
The third startup uses the packaged `benchmarks.sm70_dflash2_tail_audit`
extension and additionally counts selected context buckets. Every measured
q1/q6/q7 candidate tail selects the 262144 descriptor on all four ranks.

| Input | Seed | Control round ms | Candidate round ms | Saved ms | Control/candidate pure decode tokens/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 0 | 15.823 | 15.824 | -0.001 | 298.432 / 298.419 |
| 131072 | 0 | 22.573 | 22.577 | -0.004 | 191.441 / 191.374 |
| 261888 | 0 | 35.464 | 31.396 | 4.067 | 143.809 / 162.337 |
| 261888 | 2 | 37.403 | 31.335 | 6.068 | 154.899 / 184.953 |

Round values are medians of the three startup medians; throughput values
are medians across measured requests. Seed 2 preserves 4.727273 accepted
drafts and 5.818182 emitted tokens per speculative round. Its latency drops
16.22%. Seed 0 preserves 4.02 accepted drafts and 5.12 emitted tokens per
speculative round; its latency drops 11.47%. Tail counts differ by generation
trajectory, so the two seeds must remain separate comparisons.

At 261888/seed2, warmed control/candidate prefill medians are 1200.716 and
1201.185 ms; TTFT medians are 1439.936 and 1440.823 ms. These prefix-cached
prefill values are separate from decode and are not cold-prefill throughput.
Candidate request-average round mean/p50/p90/p99 are
31.304/31.335/31.353/31.362 ms. Complete distributions, per-request tokens,
acceptance and per-rank counters are retained in `final-summary-3starts.json`
and `tail-final-{1,2,3}.json` under the local artifact root.

After this repeated measurement, a narrow guard was added in
`GPUModelRunner.execute_model`: q1-q7 prompts or prefill chunks do not use
the decode-only tail graphs. Six actual-runner dispatch tests prove the
prefill/decode distinction for q1/q6/q7. This guard does not affect the
measured workload's 4096-token prefill chunks or decode. A final service
check covers one-, six- and seven-token prompts plus the 261888/seed2 tail.
The result of that check is recorded below.

The final guard check completes 16 requests and eight exact pairs, including
one-, six- and seven-token prompts. Its 261888/seed2 trajectory has 43
speculative rounds and 4.837209 accepted drafts per round, differing from
the prior startup sequence; retain this as a separate matched comparison.
The warmed final-code pair improves 38.338 to 32.228 ms, saving 6.110 ms.
It exercises q1/q6 FULL graphs and the 262144 context descriptors;
q7 coverage comes from the preceding three-startup campaign.
No short prefill is dispatched as a tail graph. Raw report:
`tail-prefill-guard-1.json`.

Final focused CPU suites pass 111 tests, with two GPU-only metadata tests
skipped. The service validations above provide the actual V100 graph,
operator-route and paired-output evidence. All relevant pre-commit checks
pass. Every task-owned service is stopped and the rear-GPU leases released.
