# Flash-Next HCX local schedule on full-mesh V100

The full-mesh HCX schedule delays up-weight prefetch until combine finishes,
issues independent peer loads together, interleaves the four stream norm
reductions, and assigns the final gate-mix to one warp per row. Each output
retains the original FP32 sum trees, FP16 casts, sigmoid and four-stream FMA
order. Persistent barrier counters remove the completion-counter/reset phase;
acquire/release atomics publish the two remaining grid barriers.

The normal `_C` implementation offers this experimental schedule for TP4
full-mesh HCX, M1–8, with a separate output projection or a fused Q4_K/Q6_K
projection whose input normalization is already complete. Other projection
formats, the optional fused gated-normalization prefix and the two-hop
exchange retain their existing schedule. Large batches still use the
existing model fallback. The kernel policy `sm70_hcx_local_schedule=true`
opts in before graph capture. The default is false at both the Python policy
and native operator schema pending repeatable fresh-process model qualification.
HCX itself remains controlled by `sm70_hcx`.

## Installed projection-plus-HC route, 2026-10-10

Source `b0d7af22b584a272dfea362b0fefffaf97760054` adds the local schedule to
the normal fused Q4_K/Q6_K producer dispatch. It instantiates the existing
device template; it does not change its arithmetic. The gated-normalization
prefix retains its legacy barrier protocol. Eligible model projections also
require `sm70_hcx_output_projection`, whose existing default is true.

The source-complete CMake `_C` build is installed in a fresh runtime as
`1cat_vllm-1.5.2.dev1419+gb0d7af22b5.cu128-cp312-cp312-linux_x86_64.whl`.
Wheel SHA256 is
`71f9eb9f88bd32472a313ee57ba6b515f8f217266b5359f8d51ddc0a338486d8`;
core SHA256 is
`b2b4ff426e06e234f747c67d39ae096d7bf6f0c05be9818bf04bd89ffb844679`.
The route needs no research library or private native overlay. Startup loader
overrides are unset and `_C` has no RPATH/RUNPATH. The audit separately records
OpenCV's normal post-import library-path addition and the resolved standard
Torch/CUDA dependencies. Q4_K uses 126 registers and Q6_K 181; both use
13,888 static plus 32,768 dynamic shared-memory bytes and no stack/spills.

`benchmarks/kernels/benchmark_sm70_hcxo_schedule.py` compares complete chains
using eight real matching projection/HC weight pairs, normalized synthetic
inputs, K=1536 per rank and N=2560. Production dense-segments projection uses
eight warps and split=1. Each formal graph has 512 boundaries and each arm
has 24 observations with rotating/reversed order and critical-rank timing.

| Rows | Separate production projection + local HCX, µs | Existing fused schedule, µs | Fused local schedule, µs | Saving vs separate |
| --- | ---: | ---: | ---: | ---: |
| 1 | 23.031 | 24.485 | 20.712 | 10.1% |
| 5 | 28.505 | 30.205 | 26.395 | 7.4% |
| 8 | 35.447 | 36.776 | 32.628 | 8.0% |

The M5 gain is 2.110 µs over the complete production chain. Producer-only
time is 8.930 µs; subtracting it from fused time does not measure an independent
HC kernel. A 48-boundary multiplication of the saving gives 0.101 ms under
this microbenchmark contract, not a measured model saving. The real paired
weight file SHA256 is
`86071aa93080816cbf50c77375b2e355830a5e3a3763ef5f37aa48f441e25176`.

Every M1–8 shape passes 35 eager/changed-input-graph comparisons for each
fused schedule per rank, plus 12 counter-wrap/interleaving and four fallback
comparisons. All three outputs are checked as raw FP16 bits. Separate-path
partial output is poisoned with NaNs before the fused eager/graph checks,
so stale producer output cannot satisfy the oracle. Cases include zeros,
shared/per-branch norms and both quantization formats. The fallback cases
cover gated normalization and the two-hop algorithm on full-mesh hardware;
they do not qualify an actual two-hop topology. M2/3/4/6/7 use fewer timing
samples as a correctness follow-up. A separate CUDA trace observes both
reference and selected fused kernel specializations on every rank: six
Q6_K and two Q4_K calls per schedule. Profiling is outside reported timings.

The unfused regression is 23.766 to 18.993 µs at M5, with 73 raw-bit cases
per rank. Both unfused specializations' extracted SASS match V6 exactly.
The 25 focused dispatch/cache-policy tests and applicable pre-commit checks
pass. The first runner accidentally selected the standalone benchmark with
FFN weights; those passing M1–8 results are retained under
`standalone-ffn-*` and are not fused-chain evidence. Corrected results are
`round3-results/native-v7/native.json`, `other-rows.json` and `route.json`.

These results validate installed native operators. They do not exercise the
model's Python projection-layout adapter or qualify model logits, acceptance
or latency. Full-model work remains deferred and the local-schedule default
remains false. Halving the current approximately 19-µs HC boundary is still
unachieved.

## Unfused boundary: second optimization round

Source `71fce5ca51a923c9bf52cd4016d0a63df63d32e4` is built by the normal
CMake `_C` target and installed in
`1cat_vllm-1.5.2.dev1202+g71fce5ca51.cu128-cp312-cp312-linux_x86_64.whl`.
Wheel SHA256 is
`c65f0b5f9da6475a39386defc5fd727544fccf0b05011a2d1ab025756011f420`;
core SHA256 is
`8f33440ccead32d95d2e3be6b4f7fa3b0c06bba97155ee5c977b75af8afa2096`.
The fresh runtime has no research DSO, RPATH/RUNPATH or private loader
override. The selected kernel uses 94 registers, 13,888 bytes of shared
memory and zero stack/spills. The reference full-mesh kernel's extracted
SASS is identical across the V3–V6 wheels.

| Rows | Reference, µs | Local schedule, µs | Reduction |
| --- | ---: | ---: | ---: |
| 1 | 18.680 | 14.424 | 22.8% |
| 5 | 23.753 | 19.013 | 20.0% |
| 8 | 30.525 | 25.927 | 15.1% |

The measurement contract below is unchanged: eight actual weight pairs,
synthetic activations, 512 complete boundaries per graph, 24 observations
per arm, and the critical rank for each observation. M5 observed ranges
are 23.628–24.078 µs and 18.912–19.216 µs. All M1–8 shapes pass 73 raw FP16
bit-pattern comparisons per rank and shape, including changed-input graph
replays, optional secondary inputs, shared/per-branch norm weights and zero
inputs. Six of those checks exercise counter wrap and interleaved reference /
candidate calls. A separate M5 process measures 23.737 to 19.004 µs. The M2/3/4/6/7
follow-up is a correctness run with fewer timing samples. Targeted pre-commit
checks and fresh CLI startup pass.

V4 (`ce2606b95f`) measured 23.812 to 20.140 µs at M5. V5 (`0f1361e81f`)
removed the final arrival/reset phase and measured 23.714 to 19.599 µs.
V6 replaces thread-wide fences around barrier arrival with a CTA join,
GPU-scope acquire/release RMW chain, acquire load, and consumer CTA join.
Each benchmark arm remains in the normal installed `_C`; research libraries
are not needed. Do not subtract results from these different runs to infer
the independent cost of either change.

At the user's request, the second-round model diagnostic was stopped during
model loading to prioritize kernel scheduling experiments toward a 30–50%
boundary reduction. It produced no new model-quality or model-speed result.
The same-process quality evidence later in this report applies to the older
wheel only. V4–V6 have no new model-quality or model-speed result. The default
remains false. The achieved boundary reduction is approximately 20% at M5;
30–50% and less than 1 ms of HC per round remain targets.

The diagnostic now accepts an explicit `--expected-core-sha256` when reusing
a reference workload with a rebuilt core. It records the current and
reference core identities separately and reruns all four quality phases;
it does not reuse the older candidate's quality result.

## Second-round research screens

The following measurements use isolated research DSOs. They are complete
boundary screens, not installed-route or model-speed claims. Controls and
candidates are paired within each row; different rows are different runs.

| M5 screen | Paired control, µs | Candidate, µs | Decision |
| --- | ---: | ---: | --- |
| Norm ILP + independent peer loads + row-owned wide output pushes | 21.094 | 20.088 | Integrated into the normal wheel above |
| Grouped readiness flags, groups 8 / 16 / all 80 | 21.082 | 22.141 / 22.128 / 22.972 | Reject |
| Raw LoRA-to-register operands, prefetched / streamed | 20.079 | 22.272 / 22.517 | Reject |
| Five compact LoRA receivers, shared / register consumer | 20.090 | 24.505 / 22.042 | Reject |
| Interleaved up ownership, norm input retained by its producer CTA | 20.058 | 19.804 | Small research gain; not integrated |
| Interleaved ownership plus direct batched LoRA polling | 20.058 | 51.143 | Reject |
| Row-aligned norm/down reduction, separate / interleaved trees | 19.879 | 19.893 / 19.763 | No substantial extra gain; not integrated |
| Persistent counter, without final CTA join | 20.071 | 19.203 | Integrated in V5 |
| Acquire/release barrier against the persistent-counter arm | 19.262 | 18.700 | Integrated in V6 |
| Four counter banks against acquire/release | 18.700 | 19.532 | Reject |
| Cooperative grid synchronization | 19.173 | 18.703 | Similar to acquire/release; additional launch constraints |
| Six warps against eight-warp acquire/release | 18.609 | 18.575 | Too small to justify a separate route |

The interleaved ownership experiment assigns up columns `32 * CTA + 8 * rank`
and repacks up weights accordingly. It preserves the arithmetic and keeps
norm/mix inputs in the same CTA, without changing the K summation order.
These ownership and reduction screens pass changing-input raw-bit checks,
including graph replays. Simply removing the second global barrier performs
poorly: duplicated packet polling and readiness publication can cost more
than the barrier it replaces. Retain the negative results when choosing the
next structural rewrite.

Raw second-round source, commands, hashes and per-rank results are retained
in the task's `hcx-local-fuse-20261009` artifacts, including `norm-ilp`,
`selected-followup`, `barrier-screen`, `direct-up`, `lora-pipeline`,
`hidden-owner` and `row-reduce`.

## Structural reassessment toward less than 1 ms

For 94 target HCX boundaries, an isolated average below 10.64 µs is a
necessary microbenchmark budget for a 1-ms sum. It is not sufficient to
establish a model wall-time result and excludes draft and remaining target HC.
The current installed 19.004-µs M5 result corresponds to 1.786 ms when simply
multiplied by 94. The historical model trace below is a different measurement.

The new diagnostic records `clock64` from every warp in every CTA, including
the output-receiving warp that thread-0-only timing misses. It contrasts one
warm weight pair with eight rotating pairs. In the eight-pair capture, the
longest CTA's phase spans, averaged across four ranks and four captures, are:

| Instrumented phase | Mean SM cycles |
| --- | ---: |
| Entry, input exchange and down-weight readiness | 5,246 |
| Down MMA and partial stores | 2,458 |
| First grid join | 3,629 |
| Partial preloads, norm reduction and normalized-input publication | 3,652 |
| Down reduction and LoRA publication | 975 |
| Second grid join | 3,019 |
| LoRA copy to shared memory and CTA join | 2,679 |
| Up MMA | 514 |
| Mix/output publication and remote-output collection | 2,495 |

These spans include concurrent work and straggler waits. They are not
independent causal costs or a complete additive model ledger. `clock64`
differences are taken only within a CTA; clocks on different SMs or ranks are
not aligned. Raw results retain all phases and snapshots. The old JSON field
name `stage_timestamps_ns` is historical: the new capture contains cycles.
No-instrumentation controls in this diagnostic measure 23.820 µs original,
18.693 µs installed V6, and 18.867 µs copied/instrumentable V6. The numerical
checks are bit-exact; the instrumented source changes register allocation and
does not replace the formal installed result.

The large data-handoff spans justify testing the partition, but their
elimination is not a free saving. The following rewrites preserve complete
residual, block-output and injection delivery in the timed boundary:

| Structural M5 screen | Installed V6 control, µs | Candidate, µs | Decision |
| --- | ---: | ---: | --- |
| Rank-local K tiles, overlapping norm exchange with down MMA | 19.123 | 19.433 | No gain |
| Joint norm statistics and four FP32 down-stream partials, direct consumers | 19.123 | 66.804 | Reject |
| Joint payload, compact receivers | 19.123 | 25.800 | Reject |
| Joint payload with contiguous consumer planes, direct / compact | 19.062 | 30.652 / 21.881 | Better than the strided prototype, still reject |
| One down column per CTA, full K reduced inside the CTA | 19.005 | 24.716 | Reject |
| Tensor Core down with 20 K partitions instead of 80 | 19.025 | 23.093 | Reject |

The joint-payload variants remove a communication round but exchange four
FP32 stream partials per latent element and may duplicate their consumption
across 80 CTAs. The one-column variant removes the global down-partial table
but uses SIMT dot products and repeatedly reads the residual. The coarser
Tensor Core variant reduces the partial table fourfold, while duplicating
combine work across N groups and adding a CTA-local reduction. Their paired
results reject these concrete implementations; they do not prove that every
sharded or coarser layout must be slower.

Each changed-reduction candidate passes 27 numerical cases and repeated-run
determinism checks per rank, using real weights and changed synthetic inputs.
The largest relative L2 error across these screens is 7.07e-5; the residual
output is bit-exact. These are screening tolerances, not teacher-forcing or
acceptance-rate approval. None of these rewrites enters the model route.
The one-column prototype first failed its numerical gate because the reused
K-shard harness reserved too little space for its larger all-reduce packet
layout. The corrected buffer partition passes; its failed log is preserved.
Its initial one-block-per-SM build was never launched because 84 CTAs on
80 SMs could not all be resident; the measured build has an occupancy guard,
82 registers, and zero stack/spills.

Raw sources, build commands, hashes and per-rank results are retained under
`/home/ymzx/codex-artifacts/hcx-local-fuse-20261009` in `warp-ledger`,
`joint-statistics`, `joint-statistics-coalesced`, `column-owned-down`, and
`coarse-splitk`. The same task directory on the remote host contains the
executed runners and logs.

### Half-current-latency follow-up, 2026-10-10

The next target is at most approximately 9.5 µs per complete M5 boundary,
half the current approximately 19-µs V6 result. This is a target, not a new
measurement or a model wall-time claim. The following screens use the same
installed V6 control, eight rotating real weight pairs and synthetic inputs.
All timings include residual, block and injection delivery unless the
explicit recurrent contract below applies. They remain isolated research
libraries; no new implementation is selected for the normal model route.

| M5 screen | Installed V6, µs | Candidate, µs | Decision |
| --- | ---: | ---: | --- |
| Idle warp reduces tagged norm statistics during down MMA | 18.730 | 19.663 | Reject |
| Read final mix input immediately after the second join | 18.730 | 18.824 | Reject |
| Overlapped norm plus mix input retained in registers | 18.730 | 19.083 | Reject |
| Replace first grid join with per-producer release/acquire readiness | 18.672 | 23.952 | Reject; overlapped-norm control is 19.645 |
| FP16 payload and 16-bit epoch in one 32-bit LoRA word | 18.747 | 19.790 | Reject this all-row-refresh implementation |
| Count only LoRA receiver CTAs at the second join, retaining mix locally | 18.680 | 19.598 | Reject; retained-mix control is 19.052 |
| Full-K down column per CTA using Tensor Cores, prefetched / streamed weights | 19.080 | 24.756 / 29.448 | Reject |
| Compact LoRA packets with separate buffers and epochs for each M | 18.818 | 20.289 | Reject; copied control is 18.853 |
| Stage LoRA into shared memory four / eight FP16 values per thread | 18.609 | 22.615 / 20.827 | Reject; copied control is 18.617 |
| Four full-K Tensor Core columns per down CTA, prefetched / streamed | 19.016 | 21.258 / 27.039 | Reject; one-column control is 24.726 |
| Two / eight full-K columns per down CTA, prefetched | 19.094 | 21.437 / 22.243 | Reject; four-column control is 21.252 |

The norm, readiness, compact-packet and selective-arrival candidates pass
raw-FP16-bit comparisons against installed V6:
24 eager cases and three changed-input graph cases per candidate per rank,
plus two counter-wrap/interleaving cases. The compact packet additionally
passes 18 row-count/short-epoch cases and a graph stress exceeding 65,536
calls. It refreshes all eight packet rows even when M is smaller, preventing
old inactive slots from matching a wrapped epoch. That extra work is part of
its measured cost. The follow-up separates buffers and counters by M, so every
active packet is refreshed without padding work. It still regresses, passes
54 raw-bit comparisons and four counter cases per rank, and additionally
passes 36 interleaved row-count/short-epoch cases plus a graph stress exceeding
65,536 calls. The result does not support padding refresh as the sole reason
the compact protocol loses.

The full-K Tensor Core variant assigns one down output column to each of 80
CTAs and uses otherwise unused MMA columns for the four injection outputs.
Every K reduction stays in shared memory; the global down-partial table is
eliminated. Prefetching all weights uses 254 registers; streaming them uses
82, with no spills in either case. A resident-grid occupancy check precedes
launch. Both variants pass 27 changed-input numerical cases and repeat-run
determinism per rank; the maximum relative L2 error against V6 is 6.19e-5.
They preserve full residual/block/injection delivery but change the K sum
tree, so this is only a numerical screen. Removing the partial table this
way does not offset the new full-K work and data access schedule.

Grouping four down columns shares one input scan among them. Twenty CTAs
perform down, while all 80 still combine inputs and produce up outputs. It
reduces repeated input reads fourfold and improves the one-column prototype
by 3.468 µs, but remains slower than V6. Its numerical and determinism checks
pass with maximum relative L2 error 6.19e-5. Prefetched weights still require
254 registers with zero spills. The separate vector-staging screen preserves
the arithmetic and passes 81 raw-bit plus six counter cases per rank, but
its wider packet reads, packing and shared stores regress at both widths.

The follow-up tests two and eight columns with 40 and 10 down-owner CTAs;
the eight-column path assigns injection to one additional CTA on rank three.
The complete grid remains 80 CTAs. Neither improves on the paired four-column
control, and all three lose to native V6. All use 254 registers without spills
and pass the same numerical/determinism screen, with maximum relative L2
error 6.19e-5. More input reuse alone does not produce the desired latency.

The readiness rewrite publishes complete down partials with device-scoped
release stores and acquires each producer's flag before loading its payload.
Its compiled SM70 polling loop includes `CCTL.IVALL` after the acquire load.
This is an additional cache operation absent from ordinary payload loads;
the timing does not isolate its contribution from the other costs of the
new schedule. Fewer named grid barriers alone do not establish a shorter
dependency path. The source retains the required memory ordering described
in the [PTX memory consistency model](https://docs.nvidia.com/cuda/archive/12.5.0/parallel-thread-execution/index.html#memory-consistency-model).

The residual-carry experiment uses a recurrent synthetic chain. Each boundary
delivers the full block output and injection; an identity producer on rank
zero consumes that block at the following boundary. Other ranks contribute
zero after the first boundary. Residual slices remain local between calls,
and the last call delivers the full residual. The first input is already
replicated, so entry requires no layout conversion. All kernel launches and
the final materialization remain inside the timing.

| Chain length | Native V6, µs/boundary | K-shard, materialize every call | K-shard, materialize at exit |
| --- | ---: | ---: | ---: |
| 1 | 19.016 | 19.360 | 19.326 |
| 8 | 19.315 | 19.890 | 19.305 |
| 94 | 19.350 | 19.766 | 19.028 |

At length 94, deferring residual materialization saves 0.739 µs against the
same K-shard arithmetic, but only 0.322 µs (1.7%) against V6. It does not
support the earlier hoped-for multi-microsecond gain from residual carry
alone. Full final outputs and every block/injection in an eight-step trace
are bit-identical between the two K-shard layouts. Repeated runs are exact;
the largest relative L2 error against V6 across the changed-input chain
checks is 0.001397, larger than the earlier single-boundary discrepancy.
This is a numerical screen with a synthetic identity producer, not a model
quality gate or a 94-layer model benchmark. PLE, fallback and final-mixer
layout transitions are not exercised by this chain.

Sources, build hashes, logs and per-rank results are retained in
`residual-carry`, `norm-overlap`, `norm-handoff`, `compact-lora`,
`selective-lora`, `tensor-column`, `compact-lora-shaped`, `lora-vectorstage`,
`tensor-column-group4`, `tensor-column-widths`, `current-ablation`,
`round3-build-manifest.json` and `round3-results` under the
existing artifact root. Each measured variant has zero stack/spills. These
results leave the admitted native V6 latency and its opt-in/default status
unchanged; neither halving current latency nor sub-1-ms HC is demonstrated.

### Current-schedule diagnostic ablations

These are deliberately incorrect-output controls, not usable HCX variants.
Each arm owns separate IPC buffers, scratch and counters. Peer-wait removal
loads each packet once; grid joins are bypassed only together with peer
waits. Weight removal also removes prefetch, but every build retains 128
static HMMA instructions. Communication writes, intermediate staging,
arithmetic and kernel boundaries remain. Register allocation changes between
65 and 96 registers without spills, so the differences are neither additive
causal costs nor a rigorous latency lower bound.

| M5 diagnostic arm | Median, µs |
| --- | ---: |
| Installed V6 | 18.632 |
| Copied complete control | 19.037 |
| Weight reads/prefetch removed | 17.733 |
| Peer waits bypassed | 17.901 |
| Both grid joins and peer waits bypassed | 14.893 |
| Weight reads/prefetch, grid joins and peer waits bypassed | 13.206 |

Only the copied valid control has a numerical gate: 27 raw-bit and two
counter cases per rank. The 13.206-µs combined ablation still exceeds the
9.5-µs target while doing less required work. It motivates reducing the
remaining arithmetic/data-staging path or changing the surrounding operator
contract; it does not prove that a correct 9.5-µs implementation is impossible.

### Output projection plus HC

The `producer-fusion` screen expands the boundary to include the preceding
output projection. It extracts the first eight real Q4_K/Q6_K output-projection
tensors and their matching FFN HC weights from the GGUF, with K=1536 per rank
and N=2560. Inputs are synthetic, in GGUF column order, after any GDN input
normalization. It does not load the model or exercise the Python projection
layout adapter. Both separate producer implementations use eight warps,
split=1 and one output tile per CTA, matching the configured K=1536 projection.

| M5 complete-chain arm | Median, µs |
| --- | ---: |
| DMV13 projection + original HCX schedule | 32.840 |
| DMV13 projection + installed V6 HCX schedule | 28.316 |
| Production dense-segments projection + installed V6 HCX | 28.481 |
| Existing fused HCXO | 30.225 |
| Fused HCXO with V6 local scheduling, research DSO | 26.464 |

The production producer alone measures 8.898 µs. Subtracting it from the
fused graph is an incremental chain-cost estimate, not an independently
observed HC kernel or model wall-time partition. The directly measured
complete-chain saving against the production producer plus V6 is 2.017 µs,
or 7.1%. Multiplying that saving by 48 attention boundaries gives a 0.097-ms
projection under this microbenchmark contract; it is not a model result or
the earlier assumed 3.8-µs saving per fusion.

The three original/fused arms pass 81 raw-FP16-bit comparisons per rank
against separate DMV13 + V6, including changed-input graph replays. The
production dense-segments chain also matches all three outputs bit-for-bit
for the 24 eager cases per rank. This source supports Q4_K/Q6_K and already
normalized producer inputs only. This first screen has no model quality gate
or clean normal package; installed-route validation is recorded separately.

The follow-up `producer-register` variant moves down-weight loads into
registers before the output projection, instead of relying on L2 prefetch.
It passes the same raw-bit comparisons but regresses: 28.001 µs versus
26.498 µs for the fused control in that run. Q6_K register use rises from
181 to 243 with no spills. The paired complete-chain result rejects this
placement; early weight fetch cannot be assumed to disappear from the
producer's critical path. The register-prefetch variant remains research-only.

The owned branch subsequently merges integration commit `4825b6831f`.
The sole conflict is the equivalent peer-buffer registration fix in the
compiled-payload test; the integration spelling is retained. At that merge,
the HCX CUDA source is byte-identical to measured V6 source `71fce5ca51`. This merge
does not make the older V6 wheel a new integration-wide runtime qualification.
The 25 focused HCX dispatch and compile-cache policy tests pass again on the
merged source, and the merge's pre-commit checks pass.

## Related work and interpretation

[mHC's kernel-fusion design](https://arxiv.org/html/2512.24880v2#S4.SS3.SSS1)
moves division by the norm after the projection and fuses scans of the
expanded residual stream. HCX already computes per-stream down partials
before applying reciprocal RMS. The transferable opportunity is further
local data reuse; mHC's Sinkhorn mapping and training overhead are not the
Qwen HC graph or a V100 decode latency prediction.

[TensorRT-LLM's all-reduce API](https://nvidia.github.io/TensorRT-LLM/python-api/tensorrt_llm.functional.html#tensorrt_llm.functional.AllReduceFusionOp)
exposes residual/RMSNorm fusion, including a MoE-finalize variant. HCX already
uses this general approach; its remaining local dependency chain must be
measured inside the fused kernel. The retained change shortens that chain
without changing the collective or the reduction tree.

[MSCCL++'s synchronization model](https://microsoft.github.io/mscclpp/dsl/concepts.html#synchronization)
distinguishes a thread-block barrier from asynchronous semaphore signaling.
That makes selective consumer waiting a relevant experiment, but does not
establish that it will be faster on this topology. The direct-polling screen
below loses on these V100s. Likewise, removing several operations in an
ablation does not make their individual latency differences additive: overlap,
cache pressure and scheduling change together. The 12–14 µs target is not an
achieved result of this work.

## Historical 17.4-ms model baseline and reporting scope

The [repaired HCX model report](flashnext_mtp4_round12.md#repaired-hcx-matched-endpoint-comparison)
records 17.3988 ms per unprofiled complete MTP4 round. Its separate graph-node
trace records 2.9623375625 ms of target HCX kernel service per round, averaged
over four ranks and 36 stable composition windows. This is the source of the
historical approximately 3-ms HC figure. The retained trace is
`round12/trace-hcx-materialized/steady-service.json`; the complete ledger is
`ledger-closed.json` in the same artifact directory. Draft HC service is
separate: combine/mix, down and up total approximately 0.2794 ms per round.
The trace itself has a median 18.5346-ms window; its kernel-service categories
are not an additive partition of the unprofiled 17.3988-ms endpoint.

Those historical workers report missing direct links for rank pairs 0/3 and
1/2 and execute the two-hop HCX kernel. The current scheduling experiments
use a different machine with a full NV2 mesh. The local schedule is admitted
only by full-mesh branches; the latest extension also supports eligible
fused output projections. Its benefits have not been measured or implemented
for the historical two-hop branch.

Multiplying 94 target boundaries by an isolated approximately 19-us graph
measurement gives an approximately 1.8-ms microbenchmark service estimate.
It does not establish the latest model's HC latency and cannot be subtracted
from the historical 3-ms trace as an observed optimization. Cache state,
surrounding work, rank arrival times, topology and instrumentation differ.
A comparison against the historical endpoint must preserve its deployment
contract and distinguish target HCX, remaining target HC and draft HC. No
new whole-model measurement is claimed while model work remains deferred.

## Measurement contract

Integration base: `17fa23e576fd12d2b3ce527558539058ab3024e6` on
`codex/v100-flashnext-host-fp8-kv-20261007-080157`. The implementation commit
is `3cf6194254798013425f4ddc6f434c3066f8742c`.
The compiled-model cache policy and test-harness corrections are in
`9e0451404f04f7b95d25df03bd299316a4c74306`.

The screen uses four Tesla V100-SXM2-32GB GPUs, every pair connected by NV2,
CUDA 12.8, Torch 2.10.0+cu128 and driver 580.173.02. Eight actual HC weight
pairs rotate through a 512-boundary CUDA graph. Activations are synthetic.
Each arm runs twice per sample with rotating/reversed order. The statistic
is the median of the per-sample maximum across all four ranks. The full
all-reduce/combine/norm/down/up/output exchange is timed.
The eight-pair weight file has SHA256
`8ef327c7e24f110147ee2e6fde9c631fb8576018236bed6ac434569d14266aac`.

The research control core SHA256 is
`784d1447f4f5f5593fa77db525e6b40e841366d654202aab5a56beec67398a0c`.
These research measurements are separate from installed-wheel validation.

| Rows | Installed control, µs | Copied control, µs | Selected schedule, µs |
| --- | ---: | ---: | ---: |
| 1 | 18.773 | 18.770 | 16.431 |
| 5 | 23.750 | 23.731 | 21.052 |
| 8 | 30.516 | 30.452 | 27.697 |

The research checks cover all eight weight pairs, changed inputs, an optional
second producer contribution, shared/per-branch norm weights and changed-input
graph replays. The packaged benchmark additionally compares the raw FP16 bit
patterns, including zero inputs and the operator's default dispatch.

## Installed-wheel validation

The normal CMake `_C` target was built from the owned implementation source,
installed into the wheel, and checked in a fresh task runtime. Both schedules
are in that same extension. No research DSO is loaded for these results.
The extension has no RPATH/RUNPATH and depends on the declared standard
Torch/CUDA/system libraries.

Wheel: `1cat_vllm-1.5.2.dev1194+g3cf6194254.cu128-cp312-cp312-linux_x86_64.whl`.
Wheel SHA256:
`901310b460e14f7f218ebc644f8331552532eee163b0ad3fe9f691db3332dbb6`.
Core SHA256:
`d56e4ec2d07cec0698dcb21f1ecba53f7cd761d18d804b5ae29820214fd930d0`.
Both native schedules use 92 registers and 13,824 bytes of shared memory,
with zero local memory or stack reported by `cuobjdump`.

| Rows | Reference schedule, µs | Local schedule, µs | Reduction |
| --- | ---: | ---: | ---: |
| 1 | 18.723 | 16.332 | 12.8% |
| 5 | 23.710 | 21.131 | 10.9% |
| 8 | 30.454 | 27.876 | 8.5% |

M5 has 24 observations per arm: reference 23.652–23.958 µs and candidate
21.060–21.342 µs. These are observed ranges, not confidence intervals.

Each M1–8 point passes 67 raw-FP16-bit comparisons per rank: four eager input
sets across eight actual weight pairs, both explicit and default dispatch,
and three changed-input graph replays. The M2/3/4/6/7 follow-up uses fewer
timing samples and is a correctness check, not a quoted speed measurement.
The CPU dispatch/configuration tests pass 25 cases after updating two test
doubles and checking that both schedules reuse the compiled-model cache.

The compiler/graph payload test passes on both the earlier installed wheel
and the new wheel, including M5 HCX and the M20 fallback. Its first run exposed
a test-harness omission: raw CUDA graph capture did not register custom
all-reduce peer addresses. Both wheels failed at the first M20 graph replay,
before HCX was selected. The test now uses the same distributed
`graph_capture` context as the model runner. Both reruns pass and report
identical errors against their independent reference implementation.

The final default-off policy is built from
`84e41a1d1489e5f358c78658a7ab337611a17d8b` into
`1cat_vllm-1.5.2.dev1197+g84e41a1d14.cu128-cp312-cp312-linux_x86_64.whl`.
Its SHA256 is
`bd5a10fb8355a5bc88bb8ac9759bdb9f2bc98ff25a40dab34fc978adf4623970`;
the rebuilt core SHA256 is
`26b02babd7a3680ddb3e85f1b035688ad595873ce4c80b30ebf9c3ff4debf6e6`.
A fresh installed audit confirms the false default in the Python configuration
and native schema, with equal compile keys for explicit true/false settings.
All 25 CPU cases and focused pre-commit checks pass again. The entire GPU
device-code section is byte-identical to the measured wheel, with SHA256
`8b6e60a06b8139f22819adb17ba4af44fe897da59365a909ad9ae21deac1c1d4`.
The core hash changes because of the C++ schema default; the GPU kernels did
not change. A follow-up with four timing samples (eight observations per arm)
against the installed default-off wheel passes the same 67 bitwise checks per
rank at M1, M5 and M8. Its paired
medians are 18.827/16.352, 23.733/21.146 and 30.540/27.877 µs respectively.
This confirms the packaged routes; the longer initial run remains the quoted
performance dataset. This check does not replace model quality qualification.

## Model measurement contract

The model comparison uses the same installed wheel for both arms and switches
only `sm70_hcx_local_schedule`. It runs Flash-Next GSQ-RCO IQ3_S GGUF with
checkpoint-FP16 HC weights and a FP16 MTP4 draft on TP4. Model dtype is FP16;
the QSA KV policy is E4M3 with `qsa_host_kv_device_reference=true`, an 8192-token
hot window and FP16 draft KV. Maximum model length is 9216, maximum batch
tokens 512, maximum concurrent sequences 4 and GPU memory utilization 0.95.
Prefix caching is disabled. Output projection remains separate from HCX.

The eight natural prompts use greedy sampling with a 600-token limit and
ordinary EOS handling. C1 measures fixed I8192/O256 and C4 fixed I128/O600,
with EOS ignored for those two performance probes only. Initial loading,
compilation, prefill and teacher captures are excluded from the pure-decode
statistics. The C1 observer-off measurements bracket the GPU-envelope probe.
Teacher captures use 64 fixed prompt/position pairs and two repetitions per
arm. Candidate conditioning comes from the reference token sequences.

The first fresh-process pair used separate compilation caches. Actual routes
match on all four ranks after accounting for the scheduling flag: FULL decode,
95 HCX modules, no deferred output projections, and captured M5/10/15/20.

| Initial fresh-process observation | Reference | Local schedule |
| --- | ---: | ---: |
| C1 unobserved round, ms (mean of two cohorts) | 19.8828 | 19.6105 |
| C1 emitted tokens/round | 4.8857 | 4.8857 |
| C4 unobserved round, ms | 43.3138 | 43.2972 |
| Observed target GPU envelope mean, ms | 15.6929 | 15.5179 |
| Eight-prompt mean draft acceptance | 45.3421% | 44.7988% |

The three C1 output sequences and all four C4 output sequences match, but
only one of eight natural sequences matches. At 64 aligned teacher positions,
cross-arm mean/p99/max KL is 0.001207/0.017931/0.029911 and top-1 agreement is
63/64. This fails the existing mean-KL, p99-KL and top-1 gates. Both within-arm
repetitions have identical logits at all 64 positions. The acceptance delta
is -0.543 percentage points, with a paired prompt-bootstrap 95% interval
[-3.607, +2.600]; this does not establish acceptance equivalence. The observed
0.272-ms C1 reduction is therefore not an admitted model optimization result.

The initial configuration field also partitioned the compiled-model cache,
although its dispatch occurs inside the opaque HC operator. Excluding it
from that cache key preserves reuse of unrelated compiled operators; fresh
CUDA graph capture still records the selected native schedule. This does not
by itself establish the cause of the numerical differences. A same-process
original/recaptured-reference/candidate/original-again diagnostic holds the
loaded weights and compiled operators fixed to localize them.

The cache-policy build is
`1cat_vllm-1.5.2.dev1195+g9e0451404f.cu128-cp312-cp312-linux_x86_64.whl`,
SHA256 `46992d012ff5bc96b3e8722d72d1e29d180aa091b1686b97298a76120c6e8c9a`.
It contains the same native core SHA256 as the measured initial wheel.
Installation into a separate task runtime confirms the native schema and
equal cache keys for the two schedules. The first diagnostic stopped before
any output comparison because its trusted local callable RPC lacked the
serialization opt-in. The corrected offline harness checks this before model
loading. A second attempt stopped before comparing outputs because engine
initialization mutated a nested configuration object in the report. The
report now retains an independent configuration copy. Neither harness failure
produced a model quality result; both logs are retained. The corrected run
uses the cache-policy wheel and the original benchmark's matmul precision,
sampling seed and warmup.

The corrected diagnostic completes its original-schedule phase: eight natural
continuations and all 64 teacher positions. Recapture then stops on an overly
strict invocation-count assertion (188 captured calls versus a threshold of
190); no new-schedule or roundtrip phase completes. These are not successful
same-process controls. The completed original phase is retained separately
as the reference for a fresh candidate process using the same cache-policy
wheel and the same compilation cache.

Both the initial reference and this completed phase select the reference
native schedule, yet their mean/p99/max KL is 0.000740/0.006074/0.008173, top-1
agreement is 63/64 and none of the eight natural continuations is identical.
The Python cache-key policy and compilation artifacts differ between those
versions; this is not a same-artifact A/A experiment. It shows why the initial
cross-version observations cannot uniquely identify a native schedule effect.

The subsequent fresh candidate process reuses the same cache-policy wheel
and compilation cache as that completed reference phase. All actual routes,
prompt tokenization and conditioning positions match apart from the schedule.
Mean/p99/max KL is 0.001018/0.006454/0.006540, top-1 agreement is 63/64 and
none of the eight natural continuations is identical. Candidate repetitions
remain exactly equal at all 64 positions. This still fails the mean-KL and
top-1 gates, so separate compilation caches are not a sufficient explanation.
Mean acceptance changes from 46.6689% to 46.1004%; the paired 95% interval
for the -0.568-percentage-point change is [-1.769, +0.966]. The completed
reference phase is usable for this comparison; its enclosing four-phase
diagnostic remains incomplete. These results do not qualify a default change
or a model latency claim.

### Completed same-process controls

The final diagnostic completes all four phases against the cache-policy wheel:
original graphs, recaptured reference, recaptured candidate, and original
graphs again. The loaded weights and compiled operators remain fixed. Both
new captures record 188 native HCX calls per rank across two M5 descriptors,
with identical counts across schedules and ranks. The 95 prepared modules
include a final mix-only boundary; each descriptor executes 94 full HCX calls.

| Same-process comparison | Exact teacher logits | Exact natural outputs | Exact acceptance counters | Mean/p99/max KL |
| --- | ---: | ---: | ---: | ---: |
| Original / recaptured reference | 64/64 | 8/8 | 8/8 | 0 / 0 / 0 |
| Recaptured reference / candidate | 64/64 | 8/8 | 8/8 | 0 / 0 / 0 |
| Original / original again | 64/64 | 8/8 | 8/8 | 0 / 0 / 0 |

These controls pass the model distribution gate and preserve the full-vocabulary
logits bitwise at the tested positions. Mean draft acceptance is 45.4753% in
every phase. The conditional per-operator snapshot diagnostic is not invoked
because all three controls pass.

There is now also a same-artifact fresh-process A/A comparison: the completed
original phase from the earlier diagnostic versus this final original phase.
Both use the reference schedule, the same wheel/core, shared compilation
cache, configuration, actual routes, prompts, seed and warmup. The earlier
enclosing harness later fails, but its original eight natural rows and 64
teacher positions are complete. A/A mean/p99/max KL is
0.000838/0.007006/0.007195, top-1 agreement is 63/64, only 7/64 logits are
exact, and none of the eight natural outputs is identical. A/A itself fails
the top-1 gate. This establishes variation between starts without changing
the HCX schedule; it does not identify its underlying cause.

Using this later reference phase with the earlier fresh candidate instead
gives mean/p99/max KL 0.000835/0.005859/0.007425 and top-1 agreement 64/64,
which passes the distribution limits, but only one natural sequence matches.
This additional comparison does not replace the failed original pair. The
results together show why one fresh-process A/B pair cannot isolate the native
schedule's effect. The schedule remains opt-in until startup repeatability
and the associated promotion gate are resolved.

### Exploratory same-process model timing

After the four quality phases, the same loaded model runs two warmups and
eight C1 I8192/O256 cohorts, ordered ABBA then BAAB. GPU observation is off.
All eight output sequences match. Each cohort contributes 35 steady-decode
intervals and 171 emitted tokens after the benchmark's edge trimming;
initialization, compilation and prefill are excluded.

| C1 pure-decode round statistic | Reference, ms | Candidate, ms |
| --- | ---: | ---: |
| Mean of four cohort means | 19.6543 | 19.2568 |
| Median of four cohort means | 19.8190 | 19.3727 |
| Cohort mean range | 19.0438–19.9355 | 18.7460–19.5358 |

The two symmetric blocks give candidate-minus-reference differences of
-0.0242 and -0.7708 ms. Their large difference prevents admitting the overall
-0.3975-ms observation as a stable model speedup. The reproducible performance
claim remains the installed HCX boundary result, with same-process model
numerical equivalence. A longer controlled timing window and resolution of
fresh-process A/A variation are required before default promotion.

The retained final artifacts are `model-inprocess-qualified/result.json`,
`qualified-comparison.json`, `loaded-to-shared-comparison.json`, and
`qualified-timing.json` in the task's model evidence directory. The diagnostic
and comparison scripts are retained with the raw captures and source hashes.
All task-owned model workers exit and release the GPU ownership locks.

## Rejected or superseded screens

Every row is a same-process comparison within its recorded run. Do not add
individual improvements or compare raw timings across unrelated runs.

| Screen at M5 | Control, µs | Candidate, µs | Decision |
| --- | ---: | ---: | --- |
| CTA-local norm tile | 23.742 | 23.575 | Small standalone gain; omit from final combination |
| Last-arrival norm reduction, with local tile | 23.742 | 23.597 | No added benefit |
| Partial-buffer CTA grouping 4/16/80 | 23.801 | 24.056 / 24.220 / 24.030 | Reject |
| Fixed M5, with local tile | 23.744 | 23.562 | Only 0.056 below dynamic local-tile arm |
| Second barrier counts only receiving CTAs | 23.744 | 23.611 | No added benefit over local tile |
| Up prefetch after first barrier, with local tile | 23.744 | 22.665 | Retain scheduling direction |
| Parallel gate-mix, delayed prefetch, local tile | 23.710 | 21.354 | Retain warp parallelism |
| Earlier delayed prefetch, global norm tile | 23.750 | 21.052 | Selected compact implementation |
| Direct per-CTA LoRA polling, local norm tile | 23.735 | 45.797 | Reject even on full mesh |
| Statically single-slot mix loop, against selected schedule | 21.025 | 21.113 | Reject; installed selected schedule in this run is 21.082 |

The single-slot rewrite reduces the research kernel's static instruction
count by about 13%, with the same register/shared-memory usage, but does not
improve measured latency. All 54 changed-input checks per rank pass. This
screen compares against the already selected schedule, not the original
23.7-µs schedule; its instruction-count reduction is not a speedup claim.

The K-shard rewrite gives each rank 640 hidden columns and each CTA H columns
of all four residual streams. It retains its norm/mix input in shared memory,
reduces squared norms and down partials across ranks, and includes delivery
of the complete residual state in the measured output exchange.
That delivery preserves the current model interface. Keeping the residual
state sharded across multiple model boundaries would require a broader
layout change and is not measured by these prototypes; the results below
do not establish a lower bound for that architecture.

| K-shard receiver | Control, µs | H8, µs | H16, µs | H32, µs |
| --- | ---: | ---: | ---: | ---: |
| One receiving CTA | 23.753 | 77.239 | 77.678 | 82.129 |
| Distributed receiving CTAs | 23.710 | 43.853 | 44.063 | 47.924 |

The distributed rewrite is deterministic and its largest screened relative
L2 error is 7.07e-5, but it loses on latency. It is rejected before model
quality evaluation. It changes the reduction tree and is not part of the
native implementation. Research sources and per-rank JSON are retained in
the task's `hcx-local-fuse` artifacts under `central-receiver`,
`distributed-receiver`, `group-layout`, `scheduling`, `mix`, `final-screen`,
`direct-poll` and `prototypes`; none is loaded by the normal model route.

## Reproduce the installed-operator comparison

Build the owned source using the normal SM70 CMake `_C` target and package it
in the wheel. Install into a fresh task runtime. Use the same model weights,
GPU set and topology for both arms, with no private kernel library overrides.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  .venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=4 \
  benchmarks/kernels/benchmark_sm70_hcx_schedule.py \
  --weights "$HC_WEIGHTS" --output "$HCX_RESULTS" --rows 1 5 8
```

Acquire the campaign's GPU ownership locks before launching. Check M2/3/4/6/7
with the same benchmark and a smaller sample count, and run
`tests/kernels/test_sm70_hcx_compiled_payload.py` for compiler/graph payload
ownership and the M20 fallback. Boundary savings do not establish a model
round saving: compare `sm70_hcx_local_schedule` true/false in the same wheel
with the rest of the model contract fixed.

The completed same-process procedure is available as
`benchmarks/diagnose_sm70_hcx_schedule.py`. First produce a completed reference
with `benchmark_flashnext_acceptance.py` using the same installed wheel,
HCX enabled, local scheduling disabled, separate output projection, TP4 and
MTP4. Retain the model contract above and acquire all GPU ownership locks.
Then run the trusted offline callbacks with:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
  .venv/bin/python benchmarks/diagnose_sm70_hcx_schedule.py \
  --reference "$HCX_REFERENCE_JSON" --output "$HCX_QUALITY_JSON"
```

The checked-in worker callbacks and comparison logic match the executed
diagnostic. Its entry point additionally rejects incompatible TP/MTP/policy
contracts before loading and imports the installed package before adding
repository benchmark helpers to the module search path.
