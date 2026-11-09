# SM70 Qwen3.8 No-MTP Concurrency Optimization

## Scope

This work improves Qwen3.8-Flash-Next NVFP4 no-MTP decode scaling on TP4 V100.
It does not change MTP, sampling, KV-cache precision, or model weights.

Integration base: `onecat/main@45a58ab6749096248dc15b1263bdf5faf51f5c70`.

## Frozen baseline and targets

The existing single-request contract is 70 tokens/s. The measured aggregate
throughput from the same `max_num_seqs=16` engine is:

| Concurrency | Aggregate tokens/s | Efficiency vs. 70 tokens/s | Target |
| ---: | ---: | ---: | ---: |
| 4 | 163.727 | 58.5% | 238 tokens/s (85%) |
| 8 | 280.841 | 50.2% | 420 tokens/s (75%) |
| 16 | 441.229 | 39.4% | 728 tokens/s (65%) |

Raw baseline:
`.artifacts/qwen38_flash_next_no_mtp_concurrency/result.json`.

Measurement correction (2026-09-05): the initial fixed-width tables below
used client `get_output()` blocking wait, not complete inter-token intervals.
They remain historical diagnostics and must not be described as end-to-end
throughput. The reaccounted engine-timestamp results below supersede that
accounting, without rerunning or changing any prompt, seed, output, or frozen
70 tokens/s reference. The original request-level baseline and fixed-width
steady-state measurements also remain distinct contracts.

## Evidence-first sequence

1. Generalize the native NVFP4 QPN expert route from token counts 1 and 5 to
   exact no-MTP decode widths 4, 8, and 16. Compare the complete routed MoE
   operation with the existing grouped path, including numerical error.
2. Raise compact routed-slot coverage only if the 160-route C16 microbenchmark
   beats the dense 512-expert dispatch and passes the same numerical oracle.
3. Measure exact TP4 20/40/80 KiB collectives before adding push or fused-sum
   variants.
4. Screen small-M projection, HyperConnection, and GDN candidates only after
   the MoE and collective contributions are known.
5. Run one C1/C4/C8/C16 end-to-end acceptance test after admitted candidates
   have enough projected savings; do not repeatedly relaunch the full model.

## Baseline trace and contract separation

The historical E4M3-KV SPEED-Bench result (`C8=465.24`, `C16=772.44`) is not
the baseline above. It used 1K input, E4M3 KV, the official low-entropy prompt
set, stochastic generation, and natural EOS. The frozen baseline here uses 8K
independent prompts, checkpoint-native FP16 KV, greedy generation, and 512
forced output tokens. The historical result proves that the requested scaling
is possible on these GPUs, but it is not interchangeable evidence.

A fixed-live-width C16 graph-node trace was captured after the microbenchmarks
justified profiling. Ubuntu installed the Nsight target and importer in
different roots, so the raw QDSTRM was recovered directly with
`/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter`; the model was not run a
second time. The accepted trace contains 12 graph replays on four TP ranks.
After removing two head and tail samples, eight steady steps average 28.847 ms
maximum-rank replay interval and 28.304 ms maximum-rank summed GPU service.
Wall minus GPU service is only 0.543 ms, so the remaining C16 gap is on the GPU
critical path rather than in request timing or scheduler bookkeeping.

The principal service terms are direct NVFP4 W13 (5.583 ms), direct NVFP4 W2
(2.573 ms), sparse QSA attention (1.840 ms), GDN recurrent update (1.351 ms),
and TP communication (1.625 ms). Dense FP16 CUTLASS/cuBLAS kernels account for
roughly another 8.9 ms when the broad 16x16 signature and the split-K
main/reduction signatures are combined. The parser's `LM-head/sample` label
includes that broad CUTLASS signature and must not be read as 5.736 ms of LM
head alone. The accepted artifacts are under
`.artifacts/qwen38_nomtp_concurrency/profiles/`
`c16_graph_node_batch_qpn_v6_static_fallback*`.

## Microbenchmark decisions

### Native NVFP4 direct batch route: admitted

The existing checkpoint NVFP4 packing is unchanged. Direct dispatch reads the
same packed weights with FP16 inputs and FP32 HMMA accumulation while avoiding
expert sort and replicated-input materialization. Production grouped-GEMM
accumulation was screened independently at each graph width. The retained
performance splits are:

| Tokens | W13 split-K | Warm saving over 48 layers | Cold diagnostic saving |
| ---: | ---: | ---: | ---: |
| 4 | 5 | 1.59--2.13 ms | 5.41--8.06 ms |
| 8 | 4 | 1.53--2.80 ms | 4.62--7.08 ms |
| 16 | 1 | 0.05--4.09 ms | 17.01--17.60 ms |

The ranges cover the two endpoint patterns: all rows reuse ten experts versus
every routed row selecting a distinct expert. The cold numbers are planning
diagnostics, not an additive endpoint claim. Direct-kernel output is replay
deterministic. A same-process grouped comparison was bitwise equal for the
reported runs, but that is not a portable quality proof: TurboMind can choose
a different CTA/split reduction order when autotuning in another process.
Final admission therefore requires an independent FP32 reference plus endpoint
dataset scores. The M16 overlap case is effectively neutral while distinct
routes improve strongly. The exact shape/capability fallback remains available
through `VLLM_SM70_NVFP4_QWEN38_MOE_QPN_BATCH_DECODE=0`.

### Router top-k: admitted pending endpoint quality battery

The SM70 E512/K10 router is now selected for every live width from M1 through
M16 instead of only M1/M5 and the three target widths. At M4/M8/M16 it reduces
a 48-layer micro-round by 0.335/0.342/0.343 ms (2.33--2.35x). A separate sweep
of M2/M3/M6/M7/M9--M15 saves 0.330--0.433 ms per 48 layers. Expert IDs and
source-row maps are exact. Normalized FP32 route weights differ by at most
`2.24e-8`, inside the existing M1/M5 `1e-7` contract. This closes the generic
router fallback while requests enter or leave a batch without binding the
optimization to a configured concurrency. The endpoint quality battery remains
mandatory before this extension is called release-ready.

### TP4 push collective: admitted

For one regular and one shared+routed sum2 reduction per layer, 48 layers:

| Tokens | Pull | Push | Saving | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 1.169 ms | 0.355 ms | 0.814 ms | 3.30x |
| 8 | 1.289 ms | 0.441 ms | 0.848 ms | 2.92x |
| 16 | 1.183 ms | 0.651 ms | 0.532 ms | 1.82x |

All four ranks are bitwise equal for exact-integer, signed-zero, and
model-scale dynamic graph inputs. Dispatch is restricted to fully connected
SM70 TP4, CUDA Graph capture, FP16 payloads of exactly 20/40/80 KiB, and has an
explicit `VLLM_SM70_TP4_PUSH_ALLREDUCE_QWEN38_BATCH=0` rollback.

### Sparse QSA partial launch: admitted; split cap rejected

Keeping the production split count and changing only the partial kernel from
four to two warps is bitwise equal at M2/M4/M8/M16. The projected 12-layer
savings are approximately 0.19/0.24/0.34/0.84 ms against the old four-warp
launch. A broader BLOCK_N sweep confirms that 16 remains the SM70 winner;
BLOCK_N=8 does not satisfy Triton's tensor-core K requirement and BLOCK_N=32
is slower. An M8/M16 split-count cap to 32/16 projected another 0.14/0.16 ms
per 12-layer round, but changed the FP32 merge association and produced a
maximum FP16 output difference of `6.11e-5`. In the fixed C16 endpoint run,
the cap plus the exact expert-pack candidate measured 28.861 ms versus the
28.815 ms baseline and retained only 4 of 16 greedy completion hashes. The cap
was therefore removed; only the bitwise two-warp partial launch remains.

### M4/M8/M16 W13/SwiGLU fusion: admitted pending endpoint battery

For the production interleaved gate/up layout, one CTA now computes a gate/up
tile pair and applies the existing FP16 SwiGLU epilogue without writing the
320-column intermediate. M4 retains the admitted five-way K reduction, M8 the
four-way reduction, and M16 the single-way reduction; each split partial is
summed and rounded in the same order as the direct QPN route before SwiGLU.
Static and changed-expert CUDA Graph replay are bitwise equal to that direct
route at all three widths. The fused route inherits any tiny direct-vs-grouped
reduction difference; it introduces no additional numerical difference.

Against direct W13 plus the already fused W2 path, the incremental 48-layer
saving is about 0--0.12 ms at M4, 0.46--0.54 ms at M8, and 0.25--0.28 ms at
M16. The route is limited to exact M4/M8/M16, TP4, E512/K10, H2560/I160 and
has a default-on rollback switch. The kernel explicitly supports both the
ordinary and interleaved W13 layouts; selection therefore depends on its own
operator capability, not on whether the separate large-prefill fused-SwiGLU
operator is present in an overlay build.

### Parallel W2 plus weighted reduction: admitted pending endpoint battery

The first W2-fusion prototype serialized ten experts in one warp and was
removed. The retained design assigns one warp to each of the ten routed slots
inside a token/output-tile CTA, rounds each HMMA result to the same FP16 route
value, then lets warp zero accumulate those values in fixed slot order with
FP32 FMA. It removes the routed W2 global write/read and standalone reduction
without changing weights or activation precision.

At M1/M4/M8/M16 it saves another 1.56/4.10--6.39/5.69--6.79/6.70--9.32 us
per layer over the direct QPN path. All eight static endpoint comparisons are
bitwise equal to that direct path. Dynamic CUDA Graph replay with changed
expert IDs and route weights is exact at M4/M8/M16; M1 differs from the
grouped baseline by at most `4.77e-7`, but remains bitwise equal to the direct
path and is therefore not selected by the new batch gate. The M2/M4/M8/M16 route
is default-on only when the exact SM70 TP4 E512/K10/H2560/I160 contract and
operator capability are present; setting
`VLLM_SM70_NVFP4_QWEN38_MOE_QPN_BATCH_FUSED_W2=0` restores the two-stage path.

Earlier trace interpretation incorrectly attributed M1/M2/M4 observations to
TP sequence splitting. The fixed-width trace and four-rank tensor dump prove
that a global C16 step executes M16 expert tensors on every TP rank. The lower
widths came from requests leaving the batch in the earlier request-level
trace. M2 remains useful for those live-width transitions: a checkpoint-backed
screen selected split-K 10 for direct M2 W13. Direct W13/W2 saves
1.69--1.72 ms per 48-layer M2 round; adding the fixed-order fused W2 raises
that to 1.80--1.83 ms. Dynamic expert-ID and route-weight replay remains
bitwise equal to the direct path; the maximum difference from grouped dispatch
is `2.38e-7`. Fusing W13+SwiGLU at M2 regresses about 0.13 ms per round, so M2
uses direct W13 plus fused W2 while the W13 epilogue fusion remains M4/M8/M16.

### Cross-token expert packing: rejected after endpoint validation

A four-rank C16 tensor capture records identical E512/K10 routes on every TP
replica. Across all 48 layers, 160 routes select a mean 87.3 unique experts;
66.6% of routes belong to an expert selected more than once. Packing up to
eight rows for the same expert therefore has a real reuse opportunity.

The first compact implementation lost that opportunity to an O(routes squared)
planner. Replacing it with one shared-memory expert counter per expert reduces
planning from 15.15 to roughly 4--6 us per layer. A fixed-grid packed W13 plus
four-warps-per-CTA W2 implementation is bitwise equal to the current direct
route on all 48 captured layers. A repeated same-environment run measured
165.74 us direct versus 156.57 us packed, or 0.440 ms projected saving across
48 layers. A 128-group grid severely regresses synthetic routes above 128
groups, so it is not a production choice.

The route distribution is request-dependent and the speed crossover is around
96 groups. More importantly, the combined fixed C16 endpoint run did not turn
the micro saving into wall-time benefit: 28.861 ms / 554.38 tok/s versus the
28.815 ms / 555.26 tok/s baseline. The production dispatch and build wiring
were removed rather than retaining a speculative opt-in. The task-local
evidence is in
`.artifacts/qwen38_nomtp_concurrency/c16_topk_ids_analysis.json` and
`.artifacts/qwen38_nomtp_concurrency/expert_pack_m16_route_sweep_production.json`.

### Shared-expert gate fusion at M1--M16: admitted pending endpoint battery

The existing native shared-expert gate kernel was artificially restricted to
one row even though its implementation is row-independent. Selecting it by the
measured live shape (`1 <= M <= 16`) fuses the FP16 gate dot, sigmoid, and
in-place output multiply for all small CUDA-graph widths. On checkpoint gate
weights it saves 0.250--0.320 ms per 48-layer round at M4/M8/M16. The first
model-scale comparison was bitwise equal; a broader dynamic replay sweep found
one non-bitwise case out of 32 at M16, bounded to `1.22e-4` maximum absolute
error and `1.45e-4` relative L2. Extreme synthetic activation scaling is not
bitwise invariant, so this is a score-gated scheduling/numerical candidate, not
an exact-arithmetic claim. The existing
`VLLM_SM70_QWEN3NEXT_SHARED_GATE_FUSION=0` rollback remains available.

The first fixed-width endpoint log exposed an integration miss: construction
compared the global shared-expert intermediate width (`640`) with the local
TP4 width (`160`), so the measured kernel existed but the model never selected
it. Eligibility now checks the actual local projection contract instead:
gate/up partitions `(160, 160)`, down input `160`, and hidden input/output
`2560`. This is a capability/shape check rather than a model-name or
configured-batch check. A subsequent endpoint must print the shared-gate route
log before its projected saving is credited.

### Raw E4M3 scale bandwidth: experimental, default off

The checkpoint stores one-byte E4M3 block scales plus one FP32 global scale per
expert, while the previous loader permanently expanded every block scale to
FP16. Keeping only the checkpoint representation removes about 1.875 GiB of
persistent scale tensors per TP rank. A reusable per-device FP16 workspace of
about 50 MiB materializes scales for prefill and other generic shapes. Decode
widths 2--16 instead reconstruct the effective scale in-register and never pay
the workspace expansion on the hot path.

The fast SM70 decode reconstruction uses a high/low half split of the global
scale. This is not a universal identity for every legal E4M3 code and global
scale. Production therefore does not infer safety from the model name or
quantization label. During loading, each layer temporarily constructs the old
prepared-FP16 tensors, reconstructs every W13 and W2 scale with both the strict
prefill arithmetic and the proposed decode arithmetic, and requires
`torch.equal` over the complete tensors. Decode comparison is against the
effective `prepared_half * 2**14` consumed by HMMA; dividing a candidate back
before comparison can hide mismatches through FP16 subnormal rounding. A
mismatch warns and retains the established prepared-FP16 route in automatic
mode; an explicitly forced raw route fails startup. Generic/prefill expansion
uses the strict FP32 multiply and FP16 rounding order independently of this
fast-path capability result.

The checkpoint audit sampled 3,072,000 physical block codes across layer 0 and
ten experts: all codes were in the normal positive interval 73--126, with no
zero, subnormal, sign, infinity, or NaN encodings. Inverse recovery from the
prepared scales had zero mismatches. All 73,728 gate/up/down global-scale
tensors across 48 layers were audited, yielding 470 distinct global scales.
Simulation over those globals and every observed-range code had zero mismatch;
smaller legal codes do produce counterexamples and are the reason the runtime
full-tensor capability check is mandatory.

On actual routed checkpoint weights, strict expansion and fast validation were
bitwise equal. The fixed fused route changed one 48-layer round by
0.121/0.584/0.340 ms at M4/M8/M16 respectively. The generic raw-kernel screen for
otherwise uncovered M3/M5/M7/M10/M15 remained bitwise equal and changed the
round by +0.008/+0.302/+0.456/+0.431/+0.241 ms versus already prepared scales;
its benefit is avoiding a much larger per-entry scale expansion, not beating a
resident prepared tensor. These are microbenchmark results only. The raw route
remains pending full-model quality admission before it can be credited to the
production baseline. That micro comparison used direct kernels on both sides;
it did not prove that enabling raw storage preserved the production dispatcher.

The production-loader/apply audit found that M3 and M7 were changing from
grouped TurboMind to QPN when raw storage was enabled. On actual layer-0/rank-0
weights this produced up to `2.38e-7` absolute output differences, despite
exact scales. Storage and dispatch are now independent:

- `VLLM_SM70_NVFP4_QWEN38_MOE_RAW_SCALE=1` changes only scale storage.
- `VLLM_SM70_NVFP4_QWEN38_MOE_QPN_DYNAMIC_DECODE=1` independently extends QPN
  to every live width 2--16, for both prepared and raw storage. Default off;
  the existing 2/4/8/16 split choices remain unchanged.
- Both switches remain experimental pending quality scores. Dynamic extension
  must not silently accompany a memory-format optimization.

After separation, real layer-0/rank-0 loader/apply comparisons are bitwise
equal at all M1--M16 plus M784 prefill with dynamic extension off, and at
all M1--M16 with it on. Small-width eager and CUDA Graph results agree.
Earlier boundary coverage also passed all 30 combinations of layers 0/23/47,
TP ranks 0/3, and M1/4/8/16/784 using all 512 experts per sampled layer.
These are real checkpoint weights with synthetic activations, not a full
model quality score. All 256 E4M3 encodings, signed zero, subnormal/NaN
rejection, W13 global-slot layout, effective-scale admission, and changed-input
Graph replay have focused native tests.

Task evidence:
`.artifacts/raw-audit/{decoupled_storage_v2,decoupled_dynamic_v2,`
`layers_boundary_ranks_v1}.json` and `pytest-admission-v2.log` (52 passed).
The updated native `_C` SHA256 is
`6db72e5d72ed051423ce5dc62f5a935e716c02e85c3c423c92edc96fcce51fa3`.

The portable reproduction is now in
`benchmarks/kernels/verify_sm70_nvfp4_moe_raw_storage.py`, taking an explicit
`--model` directory instead of a developer-specific model path. It fails on
numerical mismatch and uses one GPU to inspect the selected TP4 shards:

```bash
CUDA_VISIBLE_DEVICES=<idle-gpu> .venv/bin/python \
  benchmarks/kernels/verify_sm70_nvfp4_moe_raw_storage.py \
  --model <checkpoint-directory> --layers 0,23,47 --ranks 0,3 \
  --tokens 1,4,8,16,784 --out raw-storage.json
```

Add `--dynamic` for the independently controlled M2--M16 QPN extension.
CPU regression coverage across MoE admission, shared-gate/QSA shape policies,
and shutdown cleanup passes 108 tests; six GPU-only cases were skipped in
that CPU run. This is separate from the 52 focused tests above, which included
the new native raw-scale tests on an idle V100.

### Endpoint run: directional only

One post-candidate 8K/512 run measured C1/C4/C8/C16 at
79.906/196.021/318.883/475.165 aggregate tokens/s. This is a directional
improvement of 12.9/19.7/13.5/7.7% over the frozen run, but tokenizer and kernel
caches were not identical between launches. It therefore does not replace the
frozen baseline or satisfy the final paired acceptance gate. A subsequent
call-chain audit found that this run also predated effective production-namespace
registration of the parallel W2 operator, and an intermediate edit had disabled
the shared-gate fusion at model construction. Both integration defects are now
covered by focused tests; the directional result must not be credited with
either projected saving.

The next endpoint result uses `EngineCoreOutputs` rather than request-level
first/last timestamps. A sample is admitted only when scheduler running width
equals C, no requests are waiting, exactly C outputs are returned, every
request emits one token, and neither a prefill marker nor a finished request is
present. Eight head and tail samples are discarded. This avoids charging the
large staggered admission spans (up to about two seconds at C16) to steady
decode and makes the 16.81/19.05/21.98 ms C4/C8/C16 target thresholds directly
testable.

The first fixed-live-width candidate completed with all intended graph routes
visible in the production call chain. M4/M8/M16 selected the widened router,
direct NVFP4 QPN expert path, fixed-order W2/weighted-reduce fusion, sparse-QSA
partial specialization, and TP4 push collective; M16 additionally selected the
W13/SwiGLU fusion. After discarding eight head and tail samples, the medians
were:

| Live width | Legacy receive wait | Reciprocal-wait estimate | Fixed-70 estimate | Gate |
| ---: | ---: | ---: | ---: | --- |
| 1 | 12.181 ms | 82.096 tok/s | 117.3% | reference |
| 4 | 18.522 ms | 215.963 tok/s | 77.1% | fail 85% |
| 8 | 22.469 ms | 356.054 tok/s | 63.6% | fail 75% |
| 16 | 28.815 ms | 555.259 tok/s | 49.6% | fail 65% |

This is substantially better than the request-level frozen result, but it does
not meet the 238/420/728 tok/s acceptance targets. The residual grows with the
number of live routed rows, so subsequent work stays focused on cold expert
weight service and batch dense projections rather than scheduler constants.
Raw evidence is in
`.artifacts/qwen38_nomtp_concurrency/fixed_width_candidate_v1.json`.

### Paired production build and corrected timing (2026-09-05)

The raw/prepared pair uses the same production-built MoE operators and
Python code, TP4 V100, Torch 2.10/CUDA 12.8, 8K prompts, 256 forced output
tokens, no MTP, FP16 KV, Prefix Cache/Mamba align, 256K maximum length, 2048
prefill chunk and max-num-seqs 16. Its compiled `_C` SHA256 is
`04818aa069b7098927fdb4a47d516d2b316f081afd68f3612634fd2e7af5fdb3`.
The later admission fix above was rebuilt separately; these endpoint numbers
must not be presented as measurements of that newer binary.
Both arms still use the same pinned custom-allreduce and FlashQLA sidecars;
this is not a clean-wheel deployment gate. Their respective SHA256 values are
`4fdf9148b4d21951b7f40edd54a43be0ce387bbf4eb86ce010845f702531e55d` and
`c8bd7650444ec56cfe2576c044d8f5f438b0a352877064bbb51ac0510dc2ea2c`.

Client receive wait excludes processing between receives. The corrected
accounting includes only consecutive eligible engine output timestamps, trims
eight head/tail samples, and computes aggregate tokens divided by summed
interval time. No GPU rerun was needed. Four accounting regression tests cover
client wait, prefill gaps, insufficient data, and mean-vs-median throughput.

| C | Prepared mean step | Raw mean step | Prepared tok/s | Raw tok/s | Raw fixed-70 efficiency | Target |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 12.398 ms | 12.361 ms | 80.66 | 80.90 | reference | 70 |
| 4 | 18.826 ms | 18.769 ms | 212.48 | 213.11 | 76.1% | 85% |
| 8 | 22.386 ms | 22.022 ms | 357.37 | 363.28 | 64.9% | 75% |
| 16 | 29.829 ms | 29.593 ms | 536.38 | 540.67 | 48.3% | 65% |

These are paired-run diagnostics, not passed performance/quality gates. Raw
storage frees about 1.73 GiB per worker (reported model memory 21.27 ->
19.54 GiB), but its incremental C8/C16 aggregate gain is only 1.66%/0.80%.
Full-model greedy completion hashes differ in some requests, including C1.
The dynamic-width dispatcher explains an identified difference, not the C1
case. Autotuned prefill reduction order is a hypothesis, not an established
root cause. Full-model quality and the remaining mismatch audit are still
required; keep raw storage off by default and do not promote this Draft PR.

Original paired artifacts are
`.artifacts/qwen38_nomtp_concurrency/fixed_width_{prepared,raw}_production_v1.json`.
The unchanged records are reaccounted in
`.artifacts/raw-audit/engine_intervals_reaccounted_v2.json`. The benchmark now
records raw-storage and dynamic-dispatch switches separately and uses complete
engine intervals for acceptance. The 238/420/728 tok/s targets are unchanged
and remain unmet.

### Related work and next bounded screen

PR #481 (`codex/v100-qwen38-nomtp-token-trace-20260903-173451`) is a related
single-request no-MTP campaign, not another source of accepted batch numbers.
Its latest inspected commit is `30f81105621e9f39e6b3bf9d816f77d63acd8307`.
It adds exact PLE primitives, a different shared-gate kernel, and a QSA
output-gate epilogue. Reuse and widen those implementations only after a
paired M4/M8/M16 operator screen, rather than writing competing kernels.
Its shared-gate and W13 edits overlap this branch and need explicit
consolidation before integration; do not merge both blindly or count their
savings twice. Single-row acceptance does not prove multi-row graph safety.
In particular, the inspected #481 shared-gate construction still checks the
global `intermediate_size == 160`, while this checkpoint supplies 640 and TP4
local projection partitions are 160. Preserve this branch's local-shape
admission fix when consolidating; an operator-only win is not a route-hit.

No additional endpoint run is justified by the small raw-scale gain alone.
Keep the existing C16 trace as the optimization guide (about 8 ms direct MoE
and 8.9 ms dense FP16 service), finish the C1 mismatch localization, then
screen reusable batch epilogues before another full-model measurement.

### Rejected screens

- GDN CTA-group-block sweeps change a 36-layer round by only about 0.01 ms;
  the current setting remains unchanged.
- The apparent M8 FP16-padding gain was allocator/order noise. A fixed-pointer,
  alternating A/B test reduces the real weighted saving to 0.338 ms; M4 is
  only 0.128 ms and two shapes are not bitwise equal. No production padding
  route is added.
- The prior MTP campaign already measured the SM70 port of SGLang's persistent
  HC-mix kernel at 40.5x slower and found only 0.190 ms from unquantized FP16
  projection substitution. Those dead ends are not rerun here.
- A second HC experiment fused the FP16 HC up projection with the following
  gate-mix operation. It regressed one call from 12.29/13.00/15.10 us to
  163.94/164.81/166.40 us at M4/M8/M16, or about 14.5 ms across the 96 calls
  in one round. It also changed some FP32 results, so it was rejected before
  production integration.
- A follow-up exact launch-geometry screen covered the existing HC gate-mix
  and combine-norm Triton kernels at M4/M8/M16. The best bitwise gate-mix
  schedules save only about 0.04--0.09 ms per complete round. The faster M16
  combine-norm schedule changes the normalized tensor, while the fastest
  bitwise schedule is effectively the current one. No production policy was
  added for this sub-millisecond noise-floor result. Raw evidence is in
  `.artifacts/qwen38_nomtp_concurrency/hc_postops_batch_v1.json`.
- Real-checkpoint generic TurboMind FP16 projection substitutions were either
  slower in the hot-cache regime or changed outputs by about `1e-3`; none are
  admitted.
- A cuBLASLt heuristic sweep covered all algorithms returned for the eight
  dominant M16 FP16 projection shapes. The exact algorithms are already at or
  within measurement noise of the PyTorch choice. Permitting alternate FP32
  association order projects only about 0.08 ms total saving while producing
  relative-L2 differences around `3.6e-4`; this does not justify a new 64 MiB
  workspace or a numerical risk. The apparent 0.93 ms of split-K reduction in
  the trace is part of the fastest complete algorithm, not removable overhead
  in isolation.
- FP16 GDN recurrent state saves only about 0.04/0.12/0.77 ms at M4/M8/M16,
  while state and output relative-L2 errors are roughly 26--29%. It remains a
  capacity experiment and is rejected under the quality gate.
- A per-expert 256-entry FP16 scale LUT is bitwise exact but adds a dependent
  random load; it regresses the M16 direct path and is rejected.
- Concatenating the FP16 GDN qkvz and b/a projections saves only
  0.41--0.44 ms in the serial 36-layer cost model. It also changes b/a at M8
  and M16 (maximum absolute difference up to `2.44e-4`) because cuBLAS chooses
  a different reduction for the wider output. The small saving cannot justify
  the extra packed weights and numerical change, so this route is rejected.
- Running the existing FP16 GDN qkvz and b/a projections concurrently is
  bitwise exact before and after changed-input CUDA Graph replay. The actual
  production custom-op boundary, however, saves only 0.071/0.072/0.067 ms per
  36-layer M4/M8/M16 round, far below the 0.2 ms integration threshold. The
  initial bare-`torch.mm` screen overestimated the realizable overlap, so the
  multi-stream production edit was removed. Evidence is in
  `.artifacts/qwen38_nomtp_concurrency/gdn_parallel_input_production_v1.json`.
- Sorting QSA selector indices before sparse attention does not provide useful
  locality at these widths. Excluding the sort itself, the projected saving is
  only 0.020/0.002/0.006 ms per 12-layer M4/M8/M16 round, and restoring the
  original order still changes the FP16 result by up to `1.22e-4`. The runtime
  route remains unsorted. Evidence is in
  `.artifacts/qwen38_nomtp_concurrency/qsa_index_order_v1.json`.

## Batch fusion implementation screen (2026-09-05)

Control remains `d20a077bf4cc22da15850c03527ac323f30ff0cc`, prepared
scales, raw-storage/dynamic-dispatch experiments off. The integration line is
still `onecat/main`, original base
`45a58ab6749096248dc15b1263bdf5faf51f5c70`, owned Draft PR #474.
Protect the **measured 80.657 tok/s C1** within 1%; 70 tok/s is only the fixed
denominator for the user's concurrency targets. Targets are 238/420/728 tok/s
at C4/C8/C16, requiring complete steps <=16.807/19.048/21.978 ms. They are not
met by the existing endpoint measurements.

Source audit of the frozen control:

| Component | M1 | M4/M8/M16 | Remaining batch limitation |
| --- | --- | --- | --- |
| FP16 dense projection | custom row GEMV | cuBLAS linear fallback | M1 kernel cannot be enabled merely by deleting its row guard |
| HC input/up/gate mix | fused M1 operators | two GEMMs plus SiLU and gate mix | true multi-row projection/epilogue still pending |
| GDN input qkvz + b/a | fused M1 projection | two GEMMs plus contiguous copies | prior overlap and concatenate experiments did not earn integration |
| Routed W13 + SiLU | direct QPN | fused direct route, split5/4/1 | each route feeds only one logical HMMA row; repeated experts reload weights |
| Routed W2 + ordered reduce | direct QPN | fused multi-route CTA | no cross-request weight reuse; preserve original top-k reduction order |
| Router/shared gate/QSA | existing paths | batch candidates already on this branch | their gains must not be counted a second time |

Trace correction: C16 dense matrix kernels account for **8.864 ms** of GPU
service, plus **0.929 ms** split-K reduction = **9.792 ms**. The broad 5.736 ms
cuBLAS signature is not exclusively LM-head work. These are service sums, not
independent wall-clock savings or a valid subtraction from a differently
configured C1 trace.

### Implemented: grouped W13 plus intra-CTA Split-K prototype

New portable benchmark and source:

- `benchmarks/kernels/benchmark_sm70_moe_packed_w13.py`
- `benchmarks/kernels/sm70_moe_packed_w13.cu`
- `tests/kernels/quantization/test_sm70_moe_packed_w13.py`

This is **benchmark-only**, with no production dispatch, CLI, model loader,
prefill, or M1 change. It is materially different from the rejected packed
W13+packed W2 path: gather up to eight original input rows per expert directly
inside W13, divide K across CTA warps, retain the FP16 projection and SiLU
materialization boundaries, scatter to original route slots, then use the
unchanged fused W2/reduce. There is no duplicated top-k input tensor, floating
atomic reduction, CPU routing readback, or persistent FP16 expert copy.
The integer grouping plan and all writeback costs are inside the timed graph.
An optional benchmark-only `--packed-w2` variant now reuses that metadata for
W2 as described below; the default screen still uses the existing fused W2.

Algorithm reference: [PyTorch's locality-aware vLLM MoE study](https://pytorch.org/blog/accelerating-moe-model/)
supports investigating expert locality and Split-K, but its A100/H100 numbers
and floating atomic implementation are not imported as V100 evidence. It also
reports unresolved end-to-end expert mapping inconsistencies. This prototype
therefore explicitly tests changing route groups and original-slot restoration.

One idle V100 (physical GPU1), Torch 2.10/CUDA 12.8, SM70-only task-owned
extension; all 512 experts from actual checkpoint **layer 0 / TP4 shard 0**,
interleaved native NVFP4 weights/prepared FP16 scales, synthetic activations.
Replay the 48 captured C16 routing patterns from the existing fixed baseline:

| Screen | Control mean per-layer call | Candidate mean | Sum of paired savings across 48 route patterns |
| --- | ---: | ---: | ---: |
| grouped W13 split8 + unchanged W2 | 164.596 us | 124.579 us | 1.921 ms |
| additionally mask inactive shared-memory rows | 164.408 us | 124.456 us | 1.918 ms |

All 48 route patterns improved in the first screen (28.38--49.20 us).
The second change is neutral and has been removed for simplicity. **This is
not an actual 48-layer execution**: it reuses layer-0 weights with each layer's
captured routing, and is only a phase-admission estimate. No endpoint speed
or capacity improvement is claimed. This W13-only candidate is below the
>=2 ms C16 screen gate; the subsequent grouped-W2 combination is assessed
separately below.

At the layer-0 captured routing, split8 saves only 3.63 us at M4. M8's
same-split4 variant saves 3.28 us; forcing split1 on M4/M8 loses 36.12/25.80 us.
Thus widening a guard is not a batch implementation or a universal speedup.

Quality status: grouping with the **same split as the control** is exact in
the exercised real-weight M4/M8/M16 cases, including interleaved W13. Split8
at C16 changes FP32 association; the 48-pattern screen's maximum final-output
absolute difference is `1.90735e-6`, maximum relative L2 `1.97128e-4`.
These small operator errors are **not** a model-quality pass. Full HumanEval,
MBPP, GSM8K, perplexity, tool-call and structured-output score gates remain
pending; greedy hashes remain diagnostics rather than the score criterion.

Graph checks poison scratch/output buffers and replay changed inputs with
distinct, repeated, reversed and invalid routes. Dedicated tests also cover
every width M1--M16, one expert filling multiple packs, and both physical W13
layouts. Unsupported production widths still use the unchanged production
fallback because this extension is not registered into the model dispatcher.
W13-only focused GPU pytest: **22 passed** in 33.23 seconds (two unrelated Swig
deprecation warnings). Run:
`.venv/bin/python -m pytest -q tests/kernels/quantization/test_sm70_moe_packed_w13.py`.

#### Follow-up: reuse the grouping in W2

The new `--packed-w2` screen processes **both singleton and repeated groups in
one kernel**, followed by one fixed original-slot weighted reduction. This
removes the old repeated/singleton split and its redundant launches. It uses
a bounded FP16 routed-output buffer (0.78125 MiB at M16), not a second grouping
pass. The initial layer-0 / rank-0 real-weight C16 screen gives:

| Route pattern | Direct production MoE | Grouped W13 split8 + grouped W2 | Saving |
| --- | ---: | ---: | ---: |
| all 160 experts different | 191.478 us | 178.256 us | 13.222 us |
| all requests share 10 experts | 106.326 us | 33.245 us | 73.082 us |
| captured layer-0 route, 99 experts | 170.886 us | 125.130 us | 45.757 us |

Same-split1 W13 plus grouped W2 remains exact against the native control in
all five changed-input/group-count checks. Split8 retains the previously
described FP32 association difference. These are not endpoint or model-score
results. The 48-route sweep and the expanded M1--M16 W2 pytest were initially
deferred because another task occupied GPUs 0--3; the preflight correctly
returned 75 before launching. No foreign processes were terminated. The
W13-only 22-pass result above is not attributed to those expanded tests.

After the other task exited, the expanded **22 GPU tests passed in 33.59 s**.
Grouped W2 is bitwise equal to native W2 at every width M1--M16, including
changing route counts and invalid slots after graph capture. The full
48-route-pattern screen then measured **164.404 -> 112.939 us** per MoE call,
**31.30% lower**, and **2.470 ms** summed paired saving. All 48 improved
(30.22--68.55 us); final-output max-abs/relative-L2 error remained
`1.90735e-6`/`1.97128e-4`. This crosses the >=2 ms *microbenchmark admission*
gate, not an endpoint or quality-score gate. Layer-0 weights are still reused
with all captured routes, so no actual full-model latency reduction is claimed.
The combination is now eligible for production-wrapper integration followed
by the planned paired endpoint/score tests; keep it out of default dispatch
until those are completed.

### Measurement hygiene and retained evidence

Initial short-call timing used one operator sequence per graph and showed
large C4/C8 fluctuations. The accepted screen captures 16 complete calls
inside each graph, uses fixed pointers and alternating A/B order, and divides
CUDA-event time by both unroll count and replay count. Initial v1 artifacts
are retained but are not speed gates. A targeted recheck of the prior dense
cuBLASLt sweep using 32 calls per graph still projects only 0.237 ms across
all shapes, with several non-exact winners. It does not justify a new 64 MiB
workspace or replacement of the production dense path.

Artifacts relative to this owned worktree:

- `.artifacts/raw-audit/packed-w13-real-layer0-unrolled-v2.json`
- `.artifacts/raw-audit/packed-w13-real-48routes-v3.json`
- `.artifacts/raw-audit/packed-w13-real-48routes-masked-v4.json`
- `.artifacts/raw-audit/packed-w13-w2-real-v5.json`
- `.artifacts/raw-audit/packed-w13-w2-real-48routes-v6.json`
- `.artifacts/raw-audit/cublaslt-unrolled/cublaslt_fp16_algorithms_m16.json`

Each portable report records CUDA/Torch, model layer/shard, source hash,
route-file hashes (v3+), paired timing samples and numerical errors. Build
with a task-owned `TORCH_EXTENSIONS_DIR`, CUDA 12.8 `CUDA_HOME`, and
`TORCH_CUDA_ARCH_LIST=7.0`; use the task uv Python and frozen native `_C`.
The tested native `_C` SHA256 is
`6db72e5d72ed051423ce5dc62f5a935e716c02e85c3c423c92edc96fcce51fa3`;
the W13-only unmasked prototype source SHA256 was
`9bad8d300a3d8f96c697528a74cfd40e3a117c5a73c2c94cf4d494d94d4063e2`.
The final W13+W2 source SHA256 is
`b2ca971714153b83e81ff03cf7c51b8293e9723bfcaa5e8dd1427a424fdc125a`,
and its tested extension SHA256 is
`ebb7202f4462727db1345ad551f7d6f984363344024f84e440c9ba51a0768dc3`.
All GPU runs used `.artifacts/raw-audit/.venv/bin/python`, a task-created uv
environment, and the task native bootstrap. GPU1 was released after testing;
no task-owned API or resident GPU worker was started. Other GPU workloads
were not touched.

Example benchmark arguments (model and route paths supplied locally):

```bash
.venv/bin/python benchmarks/kernels/benchmark_sm70_moe_packed_w13.py \
  --model "$MODEL" --layer 0 --rank 0 --tokens 16 --splits 8 --interleaved --packed-w2 \
  --route-glob "$ROUTE_GLOB" --samples 5 --repeats 10 --out "$RESULT_JSON"
```

Next: integrate the grouped W13/W2 candidate behind an experimental local
capability/shape gate, then perform the planned control/candidate/control
endpoint and score tests. Preserve C1 and prefill; no scheduler, KV-type or
maximum-sequence-count binding. Dense multi-row fusion remains a separate
unfinished optimization track.
Keep Draft; neither default enablement nor a release claim is authorized by
these microbenchmarks.

## Guarded production integration (2026-09-05)

Starting source is `992a9375153d307c119c7b2dc6311a938bae3791`, owned PR #474,
same original integration base and performance contract as above. The former
benchmark CUDA source is moved to
`csrc/sm70_turbomind/ops/nvfp4_grouped_decode_sm70.cu` and included in the SM70
native build. Both operations register in `_C`, with Python wrappers and fake
implementations; tests and the standalone screen reuse this source rather
than maintaining a second kernel copy.

`VLLM_SM70_NVFP4_MOE_GROUPED_DECODE=1` enables the experimental route at load
time; **the default remains 0**. Admission uses the actual local
E512/H2560/I160/top10 shape, native packed weights and prepared FP16 scales,
without a model-name, TP-degree, KV-dtype, maximum-sequence-count or prefill
chunk-size restriction. Raw-scale storage, clamped SwiGLU and unsupported
shapes retain the old route. An eligible explicit opt-in with missing native
operations fails at startup. Runtime initially admits the screened M8/split4
and M16/split8 specializations; all other widths retain their prior paths.

Crucially, an M16 input is not necessarily 16 single-token decode requests.
The opaque MoE call checks existing CPU attention metadata to reject prefill,
mixed batches and multi-token verification. Missing/list/unknown metadata
fails closed. The decision is cached only within that `ForwardContext`, not
across requests or graph captures. There is no `.item()` or device-to-host
transfer in this route selection. Per-layer integer metadata is allocated
before capture (6,404 bytes/layer); W2 reuses the existing routed-output
workspace. There is no new process-global tensor cache.

Native rebuild succeeded. Tested `_C` SHA256:
`76f106f86f7e7bdf5f8a51b64378fee7ee09ba8a6d3cb51e699a944985711858`.
Validation so far:

- Existing mixed-NVFP4 and initial dispatch suites: 91 passed / 5 GPU-only
  skipped in the CPU run. Final dispatch suite including fake-op execution
  and CPU-integer metadata guards: 33 passed. These suites overlap; do not
  sum them.
- The 22 grouped-operator GPU tests pass against the production `_C`, not the
  former benchmark namespace.
- All applicable staged pre-commit hooks pass, including mypy, Python/CUDA
  formatting, Markdown lint and source-header checks.
- Real checkpoint layer-0/rank-0 **loader + `ModelOptNvFp4SM70MoEMethod.apply`**
  audit selects M8/M16 in logs. M1/M4/M8 and fallback M17/M32/M784 are exact
  against disabled control. M16 max-abs `2.38419e-7`, relative L2 `1.38547e-4`;
  repeated calls and CUDA Graph versus eager are exact. These are operator
  checks with synthetic activations, not model-quality acceptance.
- Artifact: `.artifacts/raw-audit/grouped-production-apply-v1.json`; command:
  `benchmarks/kernels/verify_sm70_nvfp4_moe_raw_storage.py --grouped --model
  <model-dir> --layers 0 --ranks 0 --tokens 1,4,8,16,17,32,784 --out <result>`.

The control/candidate/control engine experiment completed under
`.artifacts/grouped-runtime/{control_before,candidate,control_after}.{log,json}`.
All arms use the same new `_C`, source, uv Python, custom-AR/FlashQLA sidecars,
TP4, no MTP, FP16 KV, FP32 GDN state, max length 262144, prefix/Mamba align,
2048 prefill chunk, max-seqs16, 8192 input / 256 forced output, and widths
1/4/8/16. Only the grouped-decode opt-in differs; raw storage and dynamic QPN
dispatch stay off. This source-build experiment is not a clean-wheel gate.
The launcher waits for cooperative locks **and actual idle GPUs** and never
terminates another task to make room. Report engine intervals, not receive
wait or the microbenchmark projection.

### Actual engine results

Baseline below is tokens divided by the mean complete step time of the two
disabled runs, not a historical speed figure. All four workers in the enabled
arm used full CUDA graphs; M8/M16 grouped dispatch is confirmed in capture
logs. The QSA page4-XQA import fallback warning also occurs in the retained old
baseline logs: all these arms use the same Triton sparse QSA path. Do not claim
that the standalone page4-XQA implementation was exercised.

| C | Control-before step | Candidate step | Control-after step | Baseline tok/s | Candidate tok/s | Throughput change | Fixed-70 efficiency / goal |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 12.275 ms | 12.280 ms | 12.267 ms | 81.493 | 81.435 | -0.07% | reference |
| 4 | 18.671 ms | 18.645 ms | 18.694 ms | 214.108 | 214.535 | +0.20% | 76.62% / 85% |
| 8 | 22.212 ms | 22.040 ms | 22.239 ms | 359.942 | 362.981 | +0.84% | 64.82% / 75% |
| 16 | 29.566 ms | 27.371 ms | 29.589 ms | 540.951 | 584.568 | +8.06% | 52.19% / 65% |

C16 saves **2.207 ms per complete step**, confirming most of the 2.470 ms
microbenchmark projection in this fixed workload. C1 satisfies the <=1%
regression screen. C4 does not select the new route, so its small difference
is not credited as an optimization. C8 is only a small gain; none of the
238/420/728 tok/s goals is met. This one bracketed experiment is not a
confidence interval across independent repeated campaigns or other contexts.

All C1 completions are identical. C4/C8/C16 completions differ even between
the two disabled controls; enabled C4 matches the second disabled control.
That establishes baseline run-to-run token variability, **not its cause or
quality impact**. Do not label all differences as new-kernel damage, harmless
rounding, or proof of quality. Preserve the token IDs for subsequent score
tests/first-divergence diagnosis. Greedy identity is not the release criterion.

All three engines and their workers shut down; no task API remains resident.
The longstanding Python resource-tracker shared-memory cleanup warning occurs
at shutdown, as in prior baseline logs; it is not evidence of a new GPU leak.
GPU0--3 were rechecked after completion and only the desktop process remained.
Final source adds a CPU-integer metadata type guard (non-integer metadata
fails closed without tensor comparisons); this guard does not change the
screened host-integer decisions. No second full-model run is claimed for that
defensive type check or whitespace-only CUDA formatting.

Quality admission remains pending. In particular, prefill perplexity alone
does not exercise a decode-only optimization. Any PPL claim must include
teacher-forced decode through the new route; coding/tool/schema score tests
must actually contain active M8/M16 batches. Keep default dispatch unchanged
until end-to-end, quality and C1-regression gates pass.

## HC batch follow-up and attribution correction (2026-09-05)

Source anchor remains `94dd55c899be8c4725eab55374393b10315ca718`; there is
no new production route or new endpoint result in this follow-up. Reaching
C16 728 tok/s from the retained 27.371 ms step requires a further 5.393 ms
step reduction, not another sub-millisecond dispatch tweak.

Offline reanalysis uses complete `(globalPid, correlationId)` graph groups:
four ranks each have nine complete 1,976-node graphs in the retained
pre-grouped trace. Dropping the first/last complete graph leaves seven steps
on all four ranks. Exact launch geometry, same-stream reduction adjacency and
source order identify 97 HC down/up calls (48*2 plus final mixer), 36 GDN
inputs, 12 QSA inputs/index projections and 48 shared-expert/router calls.
This is inferred role attribution, not module NVTX measurement.

Dense model-graph service, excluding outside-graph LM-head/sampling:

| Role, including its split-K reduction | Rank/graph mean ms |
| --- | ---: |
| HC down/inject + up | 3.558 |
| GDN qkvz + b/a | 2.023 |
| Attention/PLE output | 1.192 |
| Shared expert gate/up + down | 0.992 |
| Router projection | 0.816 |
| QSA qkvg + index projection | 0.661 |
| PLE input projection | 0.091 |
| Total | 9.334 |

HC postops add 1.074 ms of service. These terms overlap other streams and
are not independently removable wall time. They describe the old graph,
not the grouped candidate's current per-kernel costs.

**Correction to the early trace interpretation:** wall minus summed GPU
service (about 0.543 ms in the old interval parser) is not total device idle
time or a closed host residual. Complete-graph kernel interval union gives
27.449 ms activity envelope, 26.854 ms service sum and 25.520 ms union busy:
1.334 ms of overlapped service and 1.929 ms of no-kernel gaps. The latter
contains 1.434 ms across about 1,533 gaps shorter than 2 us, 0.362 ms of
2--10 us gaps and 0.133 ms of longer gaps. Copies/dependencies may occupy
those gaps; they are not proof of CPU overhead or guaranteed fusion savings.
Reproducer/report: `.artifacts/raw-audit/attribute_c16_complete_graphs.py`
and `c16-bottleneck-audit-20260905.md` in the same directory.

### FP16 tensor-core HC fusion screen: rejected

A materially different candidate from the prior slow SIMT/Triton pointwise
projection uses Volta `mma.m8n8k4`, packed FP16 weights, 8-row tiles, shared
FP16 gates and fused ordered sigmoid/mix. Down uses global/intra-CTA Split-K
and a tail that retains FP16 rounding before SiLU/injection. No activation
or weight quantization is introduced. The later vector-load variant shares
weights across all 16 rows and replaces scalar half loads with 128-bit loads.

Actual layer-0 attention-HC weights, synthetic activations at scales
0.25/1/3, alternating A/B, 32 complete HC chains per graph, five samples:

| Initial screen | Baseline complete chain | Candidate | Decision |
| --- | ---: | ---: | --- |
| M8, fuse up/mix only | 33.658 us | 42.706 us | Slower |
| M16, fuse up/mix only | 33.726 us | 40.970 us | Slower |
| M8, full chain, best screened split20 | 32.698 us | 39.966 us | Slower |
| M16, full chain, best screened split20 | 34.571 us | 42.443 us | Slower |

Repeated changed-input graph replay after poisoning workspaces matches eager
candidate outputs. Up-only injection is exact; full-chain injection differs
because FP32 reduction association changes. These numerical checks are not
model-quality scores. No candidate is admitted. The prototype also requires
13.125 MiB of extra packed weights per HC pair (1.23 GiB across 96 pairs),
so packing must not be described as a free runtime improvement.

Artifacts: `.artifacts/raw-audit/hc-batch-v1.json` (native SHA256
`ac87b87aad8116332c02b83ea3a614ca24d63a461ab1bab90cd8f1de4ba9ea97`).
Vector-load screen: `hc-batch-v2-vector.{json,log}`. Another workload entered
GPU0--3 during the vector test despite cooperative locks; its unstable timing
samples are contaminated and cannot support a performance claim. It did not
establish a gain; do not promote it or repeat a full model run for this path.
Subsequent distributed tests check foreign GPU processes before and after
each timed group and fail closed if exclusivity is lost.

### Batched TP4 HC output sharding: also rejected

The next bounded screen borrows the output-sharding idea of PR #481, but
uses batched local linears and existing disjoint-output SM70 reductions,
including communication and data movement. It does not copy the M1 kernel
or communicator ABI, and does not widen its production guard.
Each rank computes 80 low-rank coordinates (rank3 also publishes injection),
then optionally computes/mixes 640 hidden coordinates across all HC branches.
Zero-filled disjoint publication uses the existing communicator's reductions.
All four ranks use the same inputs and retain the original weight precision.

Actual layer-0 attention-HC pair, M16, five alternating timing samples, 16
complete chains per graph, 40 replays/sample, maximum rank duration:

| Distributed screen | Matched replicated baseline | Candidate | Decision |
| --- | ---: | ---: | --- |
| Shard down and up; two reductions | 33.992 us | 41.787 us | Slower |
| Shard down only; one reduction | 33.285 us | 43.504 us | Slower |

Baseline sample ranges are 33.960--34.003 and 33.261--33.314 us; candidate
ranges are 41.414--41.950 and 42.846--43.552 us respectively. Runtime logs
confirm the existing SM70 communicator and push-buffer registration. Foreign
process checks before/after each timed group pass in both distributed runs.
This is not an NCCL-only surrogate or a sum of independently timed pieces.

Changed-input, poisoned-workspace Graph replay equals candidate eager output
on all four ranks at all three activation scales. Against the replicated
baseline, block relative-L2 reaches `2.981e-4` and injection `4.385e-4`; max
absolute differences are `0.00390625` and `0.0078125`. There is no model-score
pass. Both variants fail the performance screen before endpoint admission.

The combination of smaller local GEMMs, scatter and collectives is slower;
the complete-chain test alone does not separate their individual causality.
Do not claim that dividing weight bytes by four yields a fourfold compute
speedup, or that all the regression is communication. No more context or
batch sweeps are warranted for these unmodified implementations. The next
useful diagnostic is a short **single-HC** stage timeline of the local GEMMs,
publication and reductions before designing a fused compute/publication op.

Portable commands (task uv Python/torchrun, real checkpoint supplied locally):

```bash
.venv/bin/python benchmarks/kernels/benchmark_sm70_hc_batch.py \
  --model "$MODEL" --pairs attn --tokens 8,16 --out hc-packed.json
.venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=4 \
  benchmarks/kernels/benchmark_sm70_hc_batch_tp4.py \
  --model "$MODEL" --tokens 16 --mode full --out hc-tp4.json
```

Use `--mode down` for the one-reduction ablation. Actual launcher/cache and
results live in `.artifacts/raw-audit/run_hc{,_tp4}_screen.sh`,
`hc-batch-tp4-v1.{json,log}` and `hc-batch-tp4-down-v1.{json,log}`.
The task-created uv Python is `.artifacts/raw-audit/.venv/bin/python`,
Torch `2.10.0+cu128`, CUDA toolkit 12.8, V100-SXM2-32GB. The communicator is
the same frozen sidecar as the earlier engine tests, not a new ABI/build.
All applicable staged pre-commit hooks pass. Only benchmark/diagnostic files
are added: no production defaults, model weights, sampling or kernel routes
change. All microbenchmark workers/lock holders exit and GPU0--3 are released;
unrelated services are untouched. Current accepted endpoint evidence remains
the prior grouped candidate's 27.371 ms, still pending model-quality admission.

### HC sharding trace and generic small-message admission (2026-09-05)

The next diagnostic is a **single HC pair**, not another full-model launch.
`benchmark_sm70_hc_batch_tp4.py --tokens 16 --profile` captures three paired
graph replays on all four ranks, each containing 16 complete HC calls.
Nsight Systems 2022.4.2 uses `--cuda-graph-trace=node` on this host; its
standalone `QdstrmImporter` recovered the report without rerunning GPUs.
Each NVTX range is joined to `cudaGraphLaunch` and then to kernels by
`(globalPid, correlationId)`, not clipped to host launch duration. There are
192 occurrences of each stage across all ranks and repeats.

Mean **profiled kernel service per HC call**, not accepted absolute speed:

| Stage | Replicated baseline | Sharded candidate |
| --- | ---: | ---: |
| Down GEMM | 16.363 us | 8.631 us |
| Down split-K reduction | 4.370 us | 4.200 us |
| Scatter down result | absent | 2.103 us |
| Down-result all-reduce | absent | 16.365 us |
| SiLU | 2.581 us | 2.374 us |
| Up GEMM | 13.892 us | 6.830 us |
| Gate mix / disjoint scatter | 4.563 us | 2.946 us |
| Output all-reduce | absent | 8.775 us |

This establishes that the local GEMMs are faster, while publication and
collectives erase the saving. In particular, the `16 * 336 * 2 = 10752` byte
down result misses the existing push size whitelist and selects ordinary
`cross_device_reduce_1stage<half,4>` with grid `(2,1,1)`. The 80-KiB output
already selects the push kernel. Do not subtract these profiled service
numbers from the 27.371-ms full-model candidate result.

The bounded next screen adds default-**off**
`VLLM_SM70_TP4_PUSH_ALLREDUCE_SMALL_MESSAGES`. It admits ordinary FP16
all-reduces of positive, 16-byte-aligned messages up to the existing 80-KiB
storage capacity. SM70, fully connected TP4, registered push storage and
active graph capture are still required by the native caller. Established
payload launch choices and the `sum2` admission remain unchanged. This is a
message-size capability gate, not a model/max-sequence/chunk/KV condition.
The explicit experiment can also admit the existing 25-KiB MTP payload;
without it, the separate legacy MTP flag retains its old semantics.

Source review also found a correctness bug in the old batch block-count
override: an override below `ceil(bytes / 2048)` leaves a tail unwritten,
because the push kernel processes one 16-byte pack per thread and has no
grid-stride loop. Such overrides now fall back to the established safe
launch count. Unset/default overrides are unaffected.

`benchmark_sm70_tp4_small_message_push.py` exercises 13 sizes from 16 bytes
through 80 KiB + 16 bytes (ordinary pull fallback), partial CTAs, 10.5-KiB HC
payloads, forward/reverse mixed-size graphs and interleaved graph replay.
Changed random inputs, signed zero, subnormals, finite extremes, infinities
and NaNs (including the reserved sentinel payload) are checked against a
rank-ordered FP32 sum with FP16 output. Outputs are poisoned before replay;
prefix/suffix canaries and rank-skew delays cover stale buffers and tails.
Finite values require bitwise equality; NaN payload identity is not required.
An initial **8 cycles per pattern**, three graphs, all four ranks passes,
including an intentionally undersized `QWEN38_BATCH_BLOCKS=1` override.
This is a native numerical/graph gate, **not a model-quality score pass**.
The existing CPU allocator/dispatch suite also passes: **21 tests**.

Task-owned sidecar, same communicator lifecycle in one DSO, Torch 2.10.0
cu128 / CUDA 12.8, native SM70 only. Candidate v2 binary SHA256:
`68a14e492cc20e9fd5f8801051eabdd91a8d048537e4de32d4d564495a3491fd`.
The earlier flag-off timing attempt was rejected before any accepted timing
because an external GPU process appeared during the run. Do not count it or
compare different CUDA-built binaries as if only the new flag changed.
The same v2 sidecar was subsequently used for flag 0/1 screening. At M16,
flag-off baseline/candidate were 31.872/37.363 us; a later flag-on process
measured 33.790/33.552 us. The replicated baseline itself moved between
processes, so do not attribute the raw 37.363-to-33.552 difference entirely
to the flag. Within-process flag-on paired savings were 3.536/2.338/0.238 us
at M4/M8/M16. The first flag-on attempt also detected foreign workers and
was discarded; only the `on-retry` artifact supplies those results.

Artifacts relative to this worktree:

- `.artifacts/raw-audit/hc-tp4-profile-v1.{qdstrm,nsys-rep,sqlite,log}`
- `.artifacts/raw-audit/hc-tp4-smallmsg-off.log` (discarded timing)
- `.artifacts/raw-audit/smallmsg-mixed-v2.{json,log}` (8-cycle native gate)
- `.artifacts/raw-audit/hc-tp4-smallmsg-v2-off.{json,log}` (timing job)
- `.artifacts/raw-audit/hc-tp4-smallmsg-v2-on-retry.{json,log}`

Primary upstream references rechecked for the next design decision:
[vLLM fusion design](https://github.com/vllm-project/vllm/blob/main/docs/design/fusions.md)
describes sequence parallel and compute/collective fusion, while
[Triton-distributed's end-to-end guide](https://github.com/ByteDance-Seed/Triton-distributed/blob/main/docs/getting-started/e2e/e2e_dense.md)
explicitly reports that extra AG/RS work can erase overlap gains for small
tensors. These motivate measuring the complete local chain; neither is V100
performance evidence or justification for a blanket distributed rewrite.

### Follow-up: remove empty push CTAs and reject the HC cache-footprint bias

The 80-KiB legacy push launch uses 80 CTAs, but only 40 CTAs have active
16-byte packs. The current experimental admission therefore uses
`ceil(bytes / 2048)` for **all ordinary** covered messages, including known
sizes. This happens only when `SMALL_MESSAGES=1`; ordinary flag-off launches
remain unchanged. The helper's explicit `allow_generic` argument keeps
`sum2` admission and launch geometry unchanged even while the experiment is
active. No new scratch space or communicator ABI is introduced.

Candidate **v4** sidecar SHA256:
`348b782113785d374d397362b37cc93dc06f445d595b7cd4a0d5e5f3fdaf3888`.
The intermediate v3 DSO is not used for accepted GPU evidence: source review
separated `sum2` from generic grid selection before launching v4.
The expanded native gate passes **64 cycles per pattern / 13 sizes / four
graph sequences / four ranks**. The fourth graph interleaves ordinary and
`sum2` calls sharing the same push storage. Rank-skew, poisoned output,
canaries, finite-bit equality and NaN classification all pass, still with
the intentionally undersized legacy block override. This does not establish
model scores. Artifact: `.artifacts/raw-audit/smallmsg-mixed-v4.{json,log}`.

The HC screen now offers `--weight-copies`: rotate distinct allocations of
the **same checkpoint HC pair**, not an actual multi-layer model. This
separates hot single-pair behavior from a working set too large for cache.
With one copy, the candidate uses 3,440,640 bytes of local weights versus
13,434,880 bytes in the replicated path. With 16 copies those working sets
are 55,050,240 and 214,958,080 bytes. Both paths still execute the same
16-call graph and include GEMMs, SiLU, scatter, mix and both communications.
This allocation test is not a claim about production model memory savings.

Five alternating unprofiled samples, per-sample maximum rank duration:

| M | One-copy baseline | One-copy candidate | 16-copy baseline | 16-copy candidate |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 31.235 us | 27.187 us | 31.370 us | 31.198 us |
| 8 | 31.872 us | 28.843 us | 32.018 us | 33.462 us |
| 16 | 33.781 us | 32.486 us | 33.947 us | 36.520 us |

One M4 candidate timing outlier is retained in each report rather than
trimmed; the table uses medians. M8/M16 rotating-weight regressions occur in
all five samples. Foreign-process checks pass in these completed runs.
Changed-input/poisoned graph results equal candidate eager outputs, but the
earlier small non-exact differences versus the replicated model remain.

**Decision: reject this HC sharding implementation for production.** The
single-pair cache benefit is not robust to a layer-like working set; removing
the 10.5-KiB pull fallback and inactive CTAs does not overcome the complete
chain's communication/data-movement cost. Do not integrate it, launch a full
model for it, or count its hot-cache micro gain toward 238/420/728 tok/s.
The generic push experiment stays off by default in the WIP branch.
The next bounded implementation would need to fuse local pointwise work and
disjoint publication/gather, eliminating zero-filled scatter and redundant
traffic/launches, rather than merely split more GEMMs. Screen it with the
rotating-weight case first. Reuse existing push storage/protocol, keep all
communicator lifecycle calls within one DSO, and retain the native/model
quality gates. No such fused operator has been implemented or measured yet.

Artifacts: `.artifacts/raw-audit/hc-tp4-smallmsg-v4-copies{1,16}.{json,log}`.
The first old-binary undercoverage reproducer never reached the GPU because
an edit to its waiting shell launcher invalidated the shell's read position;
that log is a tooling failure, not a kernel result. The retry is a separate
job/artifact. Do not modify launch scripts while their waiting processes
are live. The launcher now also honors the paper campaign's full-job GPU
reservation, in addition to per-GPU locks and actual process checks.
All completed GPU workers exit. Current endpoint evidence remains the
grouped-MoE candidate's 27.371-ms C16 result, pending model-quality admission;
the overall concurrency and C1/quality goals remain **unmet**.

## Acceptance gates

- A microbenchmark candidate must improve median CUDA Graph replay time at its
  production shape and must not regress any covered shape by more than 1%.
- Native NVFP4 weights and FP16 activations remain unchanged. Report bitwise
  equality, maximum absolute error, and relative L2 error against the current
  production route.
- Before default enablement, run long-output text health plus coding, tool-call,
  and structured-output quality checks. A speedup with a quality regression is
  rejected.
- Runtime selection must be capability- and shape-based. It must not bind the
  model to one concurrency, KV dtype, maximum sequence count, or scheduler
  setting.
