# Flash-Next prefill: 6K investigation

Integration baseline: `onecat/main`, `6f3557c973b4602905037e086fd4acba9b0d867f`
(PR #1203). Work is isolated on
`codex/v100-flashnext-prefill-6k-9683-20261013-160548`.

## Workload and evidence boundaries

- Four V100-SXM2 16 GB GPUs, NV2 links, CUDA 12.8, driver 570.211.01,
  Torch 2.10.0+cu128, Python 3.12.3. One 18-core/36-thread E5-2686 v4 socket.
- Flash-Next IQ3_S, TP4, MTP4, maximum context **262144 tokens**,
  maximum scheduled chunk **8192 tokens**, prefix caching and thinking enabled.
  Experts retain 80% GPU / 20% host residency; QSA KV stays on host, PLE on disk.
- Production sampling remains temperature 0.6 / top-p 0.95 / top-k 20.
  The historical `ttft.py` is an explicitly greedy, one-output-token timing test.
- Historical `ttft.py 32000` uses random words, approximately 34700 tokens,
  **five** chunks (7344 x 4 plus approximately 5347). Its 7.50–7.55 s figure
  measures a complete HTTP response, not streaming TTFT or pure prefill.
- Historical `profile_request.py` recreates its RNG inside the word generator,
  producing one repeated word and **four** chunks. Its 6.48 s trace cannot close
  the wall-clock accounting for the different 7.5 s request. The old table also
  omitted 0.91 s of other kernels when only its first five rows were summed.
- For approximately 34700 tokens, 6000 tokens/s corresponds to approximately
  **5.8 s**: approximately 1.7 s / 23% below the historical response latency.
  No new end-to-end speed claim is established by the microbenchmarks below.

## Resource findings from the existing DMA trace

The critical GPU's main-stream kernel/copy union occupies 87.35% of the
6.482 s trace interval. Across all streams, kernel union occupies 88.98% and
kernel/copy union 97.56%. Main-stream gaps are not whole-device idle time.
Cold-expert copies overlap kernels for 85–87% of their duration; their summed
3.45–3.70 s per GPU must not be added to end-to-end latency.

Matching collective ordinals across all four ranks gives 400 all-reduces
and 812 all-gathers per rank. GPU0 arrives last at 278/400 all-reduces.
The sum from the latest rank arrival to the latest completion is 400.8 ms
for all-reduce, compared with GPU0's 419.5 ms kernel sum. The earliest-to-latest
arrival spreads sum to 258.6 ms; those waits mostly occur on earlier ranks.
They cannot simply be subtracted from the critical rank's latency. These are
trace timing estimates, not link bandwidth counters or a new workload result.

The original analysis joined CUDA correlation IDs across TP processes.
`nsys_breakdown.py` now keys them by process and correlation ID, and keys the
main stream by process and stream ID. Replaying the existing DMA trace moves
20 ms from the host-gap bucket to GPU waits: **133 ms host gaps, 687 ms GPU
waits**, with the same 6482 ms span and kernel totals. These are gap
attributions, not hardware utilization counters.

The trace has launch geometry and registers/shared-memory metadata, but no SM,
Tensor Core, DRAM, or NVLink hardware utilization counters. Static occupancy
limits are not measured resource saturation. For example, QSA rows use
120 registers/thread and approximately 34 KiB shared memory per block, allowing
two 256-thread blocks/SM. Increasing this to three was already rejected in the
previous investigation.

The saved real QSA input has 8160 queries, six heads, dimension 256, and 2051
index positions per query. Only 12240 of 16736160 indices are negative.
Reading each valid K/V pair once would request approximately 8.56 GB; dividing
by kernel time is **requested payload bandwidth**, not measured HBM traffic
(cache hits are not known).

## Experiments and decision gates

### 1. Narrow grouped MoE tiles

Hypothesis: expert batches contain roughly 100–160 rows at the current chunks.
Additional 64x64 tiles may reduce masked work, accumulator registers, and N=320
padding. Reuse the existing TurboMind grouped provider, codec transforms,
autotuner, workspace, and group-sized K iteration. Existing kernels remain
available as fallback. No graph, cache-state, or sampling behavior is changed.

The experimental tiles use the existing narrow-prefill admission
(`M >= 512` routed rows, `N <= 512`, `N % 64 == 0`). Existing provider validation
still applies. Measured cases are much narrower: TP4 rank-zero real weights,
512 experts, top-10, hidden 2560, intermediate 160, N=320 gate/up; down executes
the production 4096-token sub-batches and unroute operation. This does not
qualify every admitted shape or other models.

`moe_tiles.py` tests all gate/up types present in this model:
18 (17 layers), 22 (20 layers), 21 (10 layers), 23 (one layer). Down uses
20 (39 layers) or 42 (nine layers). Timing excludes routing, transcode,
host DMA, and multi-GPU communication. It includes both GEMMs, SiLU, down
sub-batching, allocation calls, and output unroute.

Predeclared microbenchmark numerical gates: finite hidden/output tensors;
maximum checked-row relative L2 <= 0.01 against independently dequantized
FP32 gate/up and complete MoE references; candidate/control checked-row
relative L2 <= 0.003. These tolerate FP16 accumulation/rounding, but do not
replace paired model task-quality, repetition/EOS, or long-context checks.
Require a stable chain improvement before spending a service restart on it.

Baseline, median of five CUDA-event samples, five iterations/sample:

| Layer / gate type | Tokens | Gate/up + SiLU ms | Down + unroute ms | Complete chain ms |
|---|---:|---:|---:|---:|
| 0 / 18 | 7344 | 4.327 | 3.156 | 7.367 |
| 0 / 18 | 5347 | 3.759 | 2.294 | 5.628 |
| 1 / 22 | 7344 | 4.042 | 2.908 | 6.952 |
| 1 / 22 | 5347 | 2.482 | 2.061 | 4.506 |
| 17 / 21 | 7344 | 4.458 | 3.150 | 7.465 |
| 17 / 21 | 5347 | 3.373 | 2.280 | 5.700 |
| 47 / 23 | 7344 | 3.875 | 3.200 | 6.833 |
| 47 / 23 | 5347 | 2.924 | 2.323 | 5.175 |

All eight baseline cases pass; maximum checked output error is 0.000844.
Native dispatch confirms type-18 currently selects 32x128, so a benefit cannot
be inferred merely from reducing the M tile. Separate-stage times should not
be summed as a replacement for the measured chain (GPU clocks vary).

Rejected before GPU testing: 64x64 with four warps cannot instantiate the
current metadata thread map (zero-width access for this layout). Retain the
two-warp candidate for measurement; do not change shared iterator contracts
just to admit the extra tuning candidate.

A second artifact adds padded activation supply using the existing quantized
batch-family operand. Hypothesis: at K32/K64, the unpadded shared-memory row
stride repeats banks across the distinct rows of each eight-lane MMA load
group. Padding by eight half values separates those accesses. This is a
static layout hypothesis, not hardware-counter evidence. Candidates are
32x128x32 with 32x64 epilogue subtiles and 64x64x64 with four warps. The K64
metadata tile has enough elements for four warps. Decoder, group size,
routing, bank storage, and fallback kernels remain shared with the existing
family. Separate artifacts retain the first candidate for comparison.

Measured complete MoE chains (ms):

| Layer / tokens | Control | 64x64 | Padded candidates | Padded reduction |
|---|---:|---:|---:|---:|
| 0 / 7344 | 7.367 | 7.047 | 6.995 | 5.1% |
| 0 / 5347 | 5.628 | 5.124 | 4.956 | 11.9% |
| 1 / 7344 | 6.952 | 6.868 | 6.318 | 9.1% |
| 1 / 5347 | 4.506 | 4.537 | 4.404 | 2.3% |
| 17 / 7344 | 7.465 | 7.224 | 6.711 | 10.1% |
| 17 / 5347 | 5.700 | 5.172 | 5.006 | 12.2% |
| 47 / 7344 | 6.833 | 6.736 | 6.329 | 7.4% |
| 47 / 5347 | 5.175 | 4.808 | 4.507 | 12.9% |

All eight cases pass for both candidates. Checked hidden/output rows are
bit-exact to control. Weighting these isolated chain deltas by 48 layer types
and five chunks gives only about 0.13 s for the padded artifact; this is a
screening estimate, not an end-to-end result. No claim of 6K follows.
The added grouped-prefill regression covers M=511/512/513, interior M=2049,
N=64/320/512, N32 and wide fallbacks, K=256/512/768/2560, uneven/empty experts,
indexed and direct input, fused SiLU, output guards, and changed-input graph
replay. Its predeclared maximum-row relative-L2 limit is 0.005 versus an
independent GGUF-dequantized FP32 reference.

**The second padded artifact is rejected despite those eight passes.** The
boundary regression finds non-finite/wrong results in five of 24 cases.
Its 32x128 CTA used a 32x64 epilogue subtile to distinguish the tuning key,
but that subtile is narrower than the output thread map's horizontal span.
Some threads therefore read outside the intended shared-memory tile. The
third candidate removes it and uses a full 32x64x64 CTA/epilogue with a 2x2
warp map, alongside the full 64x64x64 candidate. No shared epilogue or
iterator contract is changed. A 1x4 warp map was rejected at compilation:
its N=16/warp is below this MMA family's N=32 atom.

The verified original artifact passes **24/24**. Pytest initially prepended
the candidate source root through the tests package, so a nominal control
invocation actually loaded the padded artifact; that comparison is invalid
and retained only as a harness failure. The corrected runner imports vLLM
and `_C` before invoking pytest, asserts their paths against the requested
artifact, and prints the native SHA256. It confirms 24 original passes and
five failures for the second padded artifact. Microbenchmark JSON now also
records loaded package/native paths and native hash.

The corrected third artifact passes **24/24 boundary cases**. Its full-chain
times for the eight rows above are 6.723, 4.753, 6.201, 4.437, 6.672, 4.824,
6.404, and 4.592 ms: reductions of 1.5–15.5%, with all checked hidden/output
rows bit-exact to control. The layer/chunk-weighted isolated delta is 0.163 s.
This artifact, rather than the rejected second one, proceeds to model testing.
Its `_C` SHA256 is
`5d389732b5fb828b947ffdb74a3ec6c80aa70cb60bdc6b1ca65e437a9514148e`.

Two smaller hypotheses were rejected on the original artifact:

- Increasing down sub-batches from 4096 to 8192 tokens: layers 0/1 at
  7344/5347 tokens give complete-chain times of 7.590/5.921 and 7.269/4.878 ms,
  versus 7.367/5.628 and 6.952/4.506 ms. All checked outputs are bit-exact to
  control; speed is 3–8% worse, with a larger temporary output allocation.
- Tuning gate/up at the actual 7344 rows rather than 8192: complete chains
  are 7.378 and 6.908 ms for layers 0/1, versus 7.367 and 6.952 ms. There is
  no stable material improvement; checked outputs are bit-exact to control.

### 2. HC narrow projection

Hypothesis: the M~8K, N=88, K=10240 HC projection exposes too few concurrent
blocks. `hc_projection.py` compares the existing cuBLAS call with two, four,
and eight K partitions, FP32 partials and FP32 reduction. It includes strided
batch handling and reduction costs. Synthetic data screens throughput only;
any admitted candidate needs real activations/weights and paired quality.
The numerical screen is finite output and maximum row relative L2 <= 0.001
against FP32 and against the current projection.

Rejected: for 7344 rows, control is 0.503 ms and K partitions 2/4/8 are
0.532/0.537/0.568 ms; for 5347 rows, control is 0.360 ms and partitions are
0.425/0.419/0.411 ms. All outputs are finite; maximum relative L2 is 0.000298
against FP32 and 0.000602 against control. More parallel blocks do not offset
the partial-output/reduction overhead.

### 3. Larger custom all-reduce admission

`collectives.py` uses all four idle GPUs, rank-max CUDA-event times, FP16
rows x 2560 messages, and the production PyNccl/CustomAllreduce providers.
It includes eager custom-AR staging into its IPC buffer. Raising its capacity
to 48 MiB adds 80 MiB of GPU buffers per rank relative to the default 8 MiB.

| Rows | NCCL AR ms | Custom AR ms | NCCL AG, width 88 ms | NCCL AG, width 640 ms |
|---|---:|---:|---:|---:|
| 7344 | 0.547 | 0.585 | 0.084 | 0.318 |
| 5347 | 0.411 | 0.431 | 0.076 | 0.254 |

Rejected: increasing custom-AR capacity alone is 5–7% slower. Checked rank-zero
outputs are finite, relative L2 <= 0.000370. This is a synthetic throughput
screen, not full collective correctness or model qualification. Isolated NCCL
is faster than historical in-service call averages: rank arrival skew and
concurrent resource use need to be separated from transfer cost before another
communication change.

### 4. Existing QSA rows route

The saved real prefill input confirms the current rows route at 12.54 ms,
versus 23.07 ms for the old tiled route (1.84x). Maximum absolute output error
is 0.000977 and maximum per-head relative L2 is 0.000774; no NaNs. Requested
payload divided by runtime is about 683 GB/s, not a measured HBM counter.
This gain is already in the baseline and must not be counted again.

### 5. HC communication/compute pipeline

`hc_pipeline.py` uses the existing HC packet, gate-mix, and assembly kernels
with cuBLAS and PyNccl. It splits rows into two/four chunks and overlaps the
two gathers with subsequent projection work on a side stream. The timing
includes submissions, events, assembly, and final concatenation.

Rejected: 7344-row sequential time is 1.343 ms, versus 2.390/4.985 ms for
two/four chunks. At 5347 rows, times are 1.056 versus 2.642/4.913 ms. This
Python pipeline adds more overhead than it hides. Output versus the same
tiles executed serially is bit-exact, isolating stream ordering from cuBLAS
shape-dependent arithmetic. The original maximum-row relative-L2 screen
of 0.001 also fails for the four-element injection output in three cases
(maximum 0.004036); it is not relaxed. Model-output errors are <= 0.0000761
and all outputs are finite. No runtime change or quality qualification follows.

### 6. Per-projection prefill streams

Hypothesis: the shared cold stream can hold down DMA behind a gate/up bank
wait, and the shared resident stream orders next-layer gate/up transcode
before current-layer down transcode. `moe_streams.py` screens separate cold
streams and separate cold/resident streams independently, using the actual
native expert cache with 402 resident experts plus eight spare slots and
136 host slots. Canonical banks and staging bytes do not increase.

Eight logical layers repeat four real TP4-rank-zero layer formats, wrapping
the four-view ring. Output and restarted-pass checks must be bit-exact;
native decode misses then change residency before another comparison. The
split-resident experiment explicitly joins each view's down readers before
reusing it, including leftover prefetch after an interrupted pass. Optional
GPU sleep models only a between-layer delay, not model compute contention.
This remains a benchmark-only hypothesis pending measured benefit.

Rejected: with no artificial gap, eight-layer pass medians are 130.18 ms
shared, 130.59 ms cold-only split, 127.23 ms fully split, and 130.86 ms on
the repeated shared control. With a 20-million-cycle layer gap, all modes
take 186.3–186.9 ms. The small isolated benefit disappears once between-layer
work hides transfers. All output/restart comparisons are bit-exact, including
after 256 expert loads, 208 evictions/write-backs, and zero cache errors.
There is no runtime stream change.

### 7. Reuse the existing dequantize-plus-cuBLAS dense provider

The old trace attributes 450 ms to Q6_K bit-plane dense kernels and 149 ms to
Q5_K, across four chunks. They use 128x256 CTAs, approximately 248–255
registers/thread, and one resident CTA/SM. These are static resource limits;
no Tensor Core utilization counter was collected. Dispatch inspection confirms
the affine fused route uses the default policy for M > 32, so an 8192-row
warmup does not autotune the measured 7344/5347 shapes.

`dense_blas.py` uses real rank-zero TP4 weights and includes weight
dequantization into temporary scratch on every call. It compares fused,
dequantize+BLAS, then fused again. Maximum row-relative L2 against the FP32
reference and paired output must be <= 0.003, with finite outputs. All ten
large-M cases and 25 threshold-screen cases pass.

| Projection / source | M | Fused ms | Dequantize+BLAS ms | Fused repeat ms |
|---|---:|---:|---:|---:|
| QKV / Q6_K | 7344 | 1.867 | 1.191 | 1.839 |
| QKV / Q6_K | 5347 | 1.370 | 0.917 | 1.362 |
| QKV / Q5_K | 7344 | 1.908 | 1.143 | 1.921 |
| QKV / Q5_K | 5347 | 1.411 | 0.900 | 1.408 |
| SSM output / Q6_K | 7344 | 1.024 | 0.711 | 1.049 |
| SSM output / Q6_K | 5347 | 0.797 | 0.563 | 0.795 |
| Attention gate / Q6_K | 7344 | 1.929 | 0.697 | 1.975 |
| Attention gate / Q6_K | 5347 | 0.979 | 0.587 | 0.975 |

Only four physical layouts are added to the existing calibration table:
(bits, group, K, N) = (5,32,2560,2560), (6,16,2560,2560),
(6,16,1536,2560), and (6,16,2560,1536), all at **M >= 2048 tokens**.
At M=512/1024 several layouts regress, so those retain fused computation.
The N=160 shared-expert screen is inconsistent and is not admitted.
Uncalibrated layouts, missing native operators, and failed workspace allocation
retain the existing fused fallback. No codec, provider, startup flag, or
workspace owner is added. The existing shared maximum-size scratch is retained;
its model-level memory impact is part of the pending service check.

The affected GPU suite passes **15 tests**, including actual native route
observations at M=2047/2048/2049/4097, both eager and compiled calls, the
serving opaque projection, shared scratch, and changed-input graph replay.
Correctness support remains the existing affine provider's valid FP16 matrix
contract. Acceleration qualification is those four layouts; measured speed
points are M=256/512/1024/2048/4096/5347/7344, not every possible M or TP rank.

### 8. Skip cold-expert holes while preserving staging offsets

After decode, 110 cold experts occupy an extent of 135–136 host slots. The
old staging copy transfers that whole extent, including slots that the cold
transcode never reads. `cold_dma.py` first screens fragmented 110/104-slot
maps at real shard sizes: full copies of 73.98 MB take 16.05–16.37 ms;
selected copies take 13.31/12.71 ms including Python/native submissions.

`ExpertCacheLane.stage_cold` now derives contiguous occupied ranges from the
same saved residency map consumed by the transcode, keeping every destination
offset unchanged. It reuses the existing host arena, staging buffer, stream
dependencies, and cache owner. It does not change residency, arithmetic,
spare slots, or GPU memory. A host slot must be valid for every cold expert;
invalid mappings raise an explicit error.

`moe_dma.py` compares both copy policies in the actual eight-layer prefill
pipeline, with all other operations identical. Times are medians in ms:

| Cache state / between-layer gap | Full extent | Occupied ranges | Full repeat |
|---|---:|---:|---:|
| Initial contiguous / none | 132.857 | 133.717 | 134.331 |
| Initial contiguous / 20M GPU cycles | 181.523 | 182.574 | 182.467 |
| Fragmented after decode / none | 161.027 | 136.131 | 162.643 |
| Fragmented after decode / 20M GPU cycles | 187.707 | 182.438 | 187.692 |

All outputs and interrupted-pass restarts are bit-exact, including after 888
expert loads, evictions, and write-backs, with zero cache errors. The gap is
synthetic delay, not real model compute. The complete affected cache suite
passes **9 tests**, including graph decode, a single hardware queue, mixed
prefill/decode, fragmented/slack-free host arenas, untouched staging holes,
and using the saved snapshot after the live CPU residency table changes.
Full-model latency remains a separate acceptance gate.

### 9. Reject padded expert-batch FP16 BLAS

`moe_blas.py` screens real layer-zero TP4 weights at 7344 tokens, top-10,
512 experts and 192 padded rows/expert. It includes per-batch dequantization,
padded activation gather, gate/up, SiLU, down, scatter and weighted unroute.
Contiguous compressed GPU banks are an optimistic input layout; production
would also need to gather resident/cold slots. Both arms omit routing and DMA.

The current quantized chain takes 6.414 ms (repeat 6.478). Batches of 32, 64,
and 128 experts take 19.860, 17.301 and 16.972 ms respectively. All outputs
are finite with paired maximum row-relative L2 0.001142, within the preset
0.003 bound, but performance rejects this implementation. For 64 experts,
gate/up and down GEMMs alone take 0.688 and 0.255 ms, or about 7.54 ms for
all eight batches before dequantization/gather/scatter. FP16 gate/up output
only reduces its 0.688 ms to 0.661; it cannot close the gap. No runtime route
is added. Raw artifacts: `moe-blas.json` and `moe-blas-first.json`.

### 10. Restore the calibrated policy after segment-bank restoration

The matched serving trace exposed a missing integration path: no affine BLAS
or affine dequantization kernels execute, despite the four admitted physical
layouts. With `small_m_hmma=true`, initialization replaces canonical weights
with segment storage. Its prefill fallback restores canonical weights but
always invokes fused GEMM, and serialization discards the BLAS bands.
Thus the first 7.099 s result below does not demonstrate dense BLAS execution.
The earlier direct-kernel and ordinary opaque-path tests did not cover this
production storage mode.

The follow-up carries the existing admitted bands through single and mixed
segment wrappers, retaining the small-M route. After canonical restoration,
it reuses the existing affine BLAS operator and engine-owned scratch only
inside those bands; all other inputs retain fused GEMM. Mixed output views
use a contiguous temporary because the existing native BLAS API requires it.
No native operator, configuration flag or extra persistent workspace is added.
`dense_segments.py` measures this actual restoration chain on real weights;
`test_gguf_segment_prefill.py` covers threshold neighbors, a range interior,
return to decode, compiled opaque calls, changed-input graph replay, mixed
output views and an uncalibrated fallback. **44 GPU/dispatch tests pass**,
including the existing segment codec, restoration, small-M graph and runtime
opacity regressions. Actual native BLAS dispatch is observed in the restored
chain; opaque and compiled surfaces produce the same qualified outputs.

Real-weight restored-chain medians (ms), including canonical restoration and
weight dequantization on every call:

| Projection / source | M | Restored fused | Restored BLAS | Fused repeat |
|---|---:|---:|---:|---:|
| blk.0.attn_qkv.weight / 14 | 7344 | 1.959 | 1.309 | 1.938 |
| blk.0.attn_qkv.weight / 14 | 5347 | 1.464 | 1.016 | 1.398 |
| blk.1.attn_qkv.weight / 13 | 7344 | 1.991 | 1.229 | 1.918 |
| blk.1.attn_qkv.weight / 13 | 5347 | 1.440 | 0.980 | 1.475 |
| blk.0.ssm_out.weight / 14 | 7344 | 1.148 | 0.824 | 1.144 |
| blk.0.ssm_out.weight / 14 | 5347 | 0.882 | 0.610 | 0.882 |
| blk.1.attn_gate.weight / 14 | 7344 | 2.077 | 0.814 | 2.038 |
| blk.1.attn_gate.weight / 14 | 5347 | 1.061 | 0.777 | 1.063 |

All eight cases are finite and meet the preset 0.003 maximum row-relative
L2 bounds against both the FP32 oracle and the paired fused output. Full-model
route observation, quality and unprofiled speed with this fix remain pending.

### Matched request measurement

`measure_prefill.py` uses deterministic word sequences and per-request hashes,
clears prefix caching on an exclusively owned server, and records streaming
first choice, first text, complete response, actual token count, and service
metrics. Service metric sums are attributed only when their count increases
by exactly one. Server prefill includes scheduled wall time, not just GPU
kernel time. `--profile` uses identical request generation; profiled wall time
is not an unprofiled speed result.

Fresh baseline on the owned server, after two excluded warmups (13.35/8.01 s):

| Prompt tokens | Client first text s | Server prefill s | Server TTFT s |
|---:|---:|---:|---:|
| 34688 | 7.895 | 7.544 | 7.757 |
| 34606 | 7.513 | 7.186 | 7.448 |
| 34631 | 7.429 | 7.165 | 7.376 |

Median client TTFT is **7.513 s / 4605.9 tokens/s**. In the middle request,
server TTFT exceeds server prefill by 0.261 s, and client first text follows
server TTFT by 0.066 s. These metrics share one request, unlike the historical
four-chunk profile. The first measured request still shows warming drift.
The existing 30K-token retrieval passes 2/2. `quality.py` adds paired archive
retrieval plus arithmetic, production T0.6/top-p0.95/top-k20 with thinking,
strict expected JSON, normal EOS, and a repetition check. The 32761-token
baseline passes with 166 generated tokens; near-256K qualification is pending.

The first near-256K baseline was interrupted before completion to keep the
cheap optimization screen first. Above the configured 131072-token QSA
history-stage limit, the original runtime falls back to direct host-history
attention and slows sharply (around 32 layer snapshots per 30 s, all GPUs
busy but around 85–90 W). No long-context quality result or candidate
regression is inferred. Complete this baseline/candidate boundary check
before promotion; the configured limit and deployment remain unchanged.

First candidate (before the segment-policy fix), same five prompt hashes and
token counts as control:

| Prompt tokens | Client first text s | Server prefill s | Server TTFT s |
|---:|---:|---:|---:|
| 34688 | 7.502 | 7.171 | 7.387 |
| 34606 | 7.091 | 6.823 | 7.034 |
| 34631 | 7.099 | 6.790 | 7.021 |

Median TTFT is **7.099 s / 4878.1 tokens/s**, improving 0.414 s (5.5%) from
the paired 7.513 s baseline. Excluded warmups take 14.13 and 7.56 s.
This is unprofiled; microbenchmark deltas are not substituted for this result.
It does **not** meet 6K tokens/s. The 32K quality prompt has the same hash and
passes the same expected JSON, EOS and repetition checks (167 generated tokens
versus control's 166). The 131070-token candidate also passes (161 generated
tokens, 60.44 s complete response); a matching 128K control is still pending,
so this does not establish a long-context performance comparison.

The isolated cold-cache candidate startup took about 22 minutes, including
SM70+SM75 FlashQLA fallback compilation, checkpoint loading, and fresh main/MTP
graph compilation. This is not the warmed deployment's startup time. No OOM
occurred. The owned candidate was stopped after these checks for a bounded
expert-batch FP16 MoE screen and a matched Nsight profile. Its descendants
were recorded and only its verified API process received SIGTERM.

### Matched candidate trace and boundary quality

The first-candidate profile uses the same index-2 request as the unprofiled
runs: 34688 tokens, hash
`126bc5319be9170da4f07bd2bca263e083d2d636da17d0697d7a19d746b04342`.
Client first text is 7.312 s, server prefill 6.952 s and server TTFT 7.209 s.
The four main-stream spans are 6.914 s. This capture also enables NCCL
INIT/TUNING logging, so its wall time is not an unprofiled performance result.
The profiled prefill chunk row counts are 7344, 6528 and 6128; do not assume
that every request uses the historical 7344/5347 microbenchmark shapes.

GPU0 closes as 6053.5 ms main-stream kernel/copy/memset union, approximately
785.6 ms submitted GPU gaps and 75.1 ms host gap. The initial analysis omitted
7.6 ms of memsets and therefore counted them as gaps; `nsys_breakdown.py` now
includes them. The 397.9 ms outside the stream span remains
an unattributed HTTP/server residual; it is not added to kernel categories.
Main-stream categories from `nsys_breakdown.py`, in ms:

| Work | Kernel/copy time |
|---|---:|
| TurboMind dense projections | 987 |
| MoE gate/up + down | 662 + 501 |
| MoE unroute + routing | 122 + 68 |
| QSA sparse partial + other + merge | 624 + 354 + 12 |
| AllReduce + AllGather | 501 + 381 |
| HC combine/norm + other HC | 348 + 125 |
| Other cuBLAS, including HC down | 589 |
| GDN | 350 |
| Elementwise/other, MTP, memcpy, dense helpers | 258 + 66 + 55 + 42 |
| Memset | 8 |

The HC down cuBLAS kernel alone is 272.2 ms / 500 launches. GPU0 union
coverage is 87.44% main-stream kernel/copy (87.55% including memsets),
89.88% all kernels, and 97.88%
all kernels/copies. These are timeline coverage, not SM, Tensor Core or HBM
utilization counters. NCU counters remain unavailable. Cold-stream H2D is
13.620 GB/rank over 3.553–3.766 s, about 3.62–3.83 GB/s. It overlaps
computation and must not be added to the wall-clock table.

`nsys_wait_dependencies.py` further distinguishes recorded event dependencies
from submitted gaps. It joins event records by process/event/synchronization
ID, follows their source-stream work and earlier dependencies, and reconstructs
a lower bound for event readiness. Synchronization activity timestamps are CPU
API durations and are not used as GPU wait durations. In this trace every
kernel/copy/memset/event record has a process-matched runtime correlation.
The directly attributable event stalls total **108.0 ms**: 61.4 ms before the
first down projection and 46.6 ms before the first gate/up projection, over
five chunks. Thus the approximately 786 ms gap bucket does not establish an
equally large opportunity for cold-expert double buffering.

Among queued gaps of 20–100 microseconds, 342.2 ms total overlaps side-stream
DMA for 331.0 ms. Many larger gaps overlap `_stage_pages`, which gathers QSA
history directly from mapped host memory (98.5 ms of side-stream kernels).
Overlap alone is not causation. Two cheap pending screens distinguish these
hypotheses: `dma_dispatch.py` compares fresh processes with 8/32/8 CUDA work
queues, and `qsa_stage_dma.py` compares the existing mapped-gather and pinned
host-copy staging branches, including concurrent HC-shaped computation.
Neither changes production runtime or establishes a model speedup.

Across ranks, 500 main FP16 all-reduces spend 434.5 ms after the last rank's
arrival; arrival spread sums to 245.1 ms, with GPU0 arriving last 366 times.
The 1030 all-gathers have 335.9 ms after the last arrival and 133.3 ms spread.
NCCL 2.27.5 TUNING logs explicitly select RING/SIMPLE for the prefill payloads
(37,601,280-byte reduce; 1,292,544/9,400,320-byte per-rank gather inputs).
The trace kernel name ends in RING_LL even for these calls: that name alone
is not evidence of actual protocol. This explains why forcing Simple in the
previous handoff gave no benefit; no protocol override is promoted.

The dominant grouped down kernel uses 255 registers/thread, 128 threads
and 32784 bytes shared memory per CTA. Register capacity bounds it to two
resident CTAs/SM, or eight warps (12.5% of 64); this is a resource ceiling,
not measured achieved occupancy. It motivates a bounded short-K tile screen
inside the existing provider, with smaller row tiles and the existing padded
activation loader. Its accelerated subset is grouped M>=512 routed rows,
N=2560 and K=160, for IQ4/group32 and affine U2/group32; other shapes retain
the existing tuner candidates.

The first short-K build used CTA_K=64, which is ineligible for K160 because
`Kernel::is_feasible` requires K divisible by CTA_K. All 40 tests passed but
trace records show the old down tactic still executing. Its small timing
movement is not credited as candidate benefit. Artifact SHA256:
`df27efc944ddebbc238ddf0dc6252b74bc5a5807d6e7057b7fee7f1508f1bd01`;
logs are archived with the `ineligible-k64-` prefix. The corrected experiment
uses K32 and distinct 2x2 warp layouts, preserving that eligibility constraint.
It builds as `041038bcecae84a18e845bdc48e8999c86aa83c8e2de7b767f40f8a4cf1c0485`
and passes 40 boundary/graph tests and all eight real-weight numerical checks.
The 32-row variants reduce register use to 126/132, but tuning still chooses
the old 128x128x16 kernel at 40960 routed rows. At 20480 routed rows it chooses
the new 64x128x32 IQ4 kernel (184 registers, two resident CTAs maximum).
Complete chains at layers 0/1/17/47 and 7344/5347 tokens are
6.634/4.871, 6.075/4.317, 6.968/5.172 and 6.327/4.928 ms. Compared with the
earlier accepted artifact, several tails regress and there is no stable
material gain. **Reject** these tactics; restore the accepted native artifact
and source before model testing. The rejected source/native files are retained
in the owned experiment archive; `moe-down.json`, `down-prefill-tests.log` and
`down-tile-resources.json` distinguish execution from resource estimates.

The GDN core's 180 launches total 271.8 ms; each launches 48 CTAs with 128
threads, 255 registers/thread and approximately 90 KiB shared memory. This
motivates a separate value-width screen, not a claim of measured SM occupancy.
`gdn_value_tiles.py` compares the existing TileLang core at value widths 32
and 16, using normalized synthetic TP4 operands and indexed FP32 state.
State reset is included in every timing; operand preparation is excluded.
The predeclared output/state maximum row-relative L2 limit is 0.003, with
finite output and exact untouched state slots required. Both cases are in
fact bit-exact. Width 16 takes 2.065/1.438 ms at 7344/5347 tokens, versus
width 32 at 1.853/1.175 ms (repeated controls 1.694/1.177). **Reject** the
finer tiling: extra CTAs do not compensate for duplicated work. No GDN
runtime change or wider quality qualification follows.

The first candidate's 260089-token quality prompt passes retrieval, arithmetic,
normal EOS and repetition checks (181 output tokens; maximum repeated ten-word
span 1). Hash:
`2c78b9e46e325dfd708b4d8c8f8c7c8fa2b9696bcb3d54f2a62313c70fb76665`.
The full response takes 821.64 s, under the existing 131072-token staging limit
and direct-host fallback beyond it. This is a boundary quality pass, not a
paired speed comparison; matching control is still pending. The owned API,
workers and Nsight launcher are stopped, with process ids archived in
`profile-stop-tree.json`. The remote trace files were removed only after
matching local SHA256 checks; local raw artifacts are `candidate-matched.*`.

## Execution discipline

Microbenchmarks acquire the existing GPU locks and start only on an empty GPU.
Other tasks' services are left to their owners. Build output, native tuning,
Triton, and Python compiler caches are task-owned. The shared venv is unchanged.
Candidate native code is built through the normal CMake `_C` target; no private
sidecar kernel or preload is an accepted artifact.

The first native candidate installs `_C` SHA256
`02dc9ff2fca7fb2c154448b5f21d537ee29a547b0ade6f5b5e4018759a56bb63`;
control is `5f0d7a83415532ec817c3e23a7b24b5015d7cbacafcd3d452fabbc28efeadd5b`.
The padded candidate installs `_C` SHA256
`fc0c457f39c2bb7b788c614b1d67c8226922a34ec44aef6e20d005c3c7c0e7cf`.
The build reused three unchanged Marlin objects after verifying identical
normalized compiler commands, all 15 relocated source dependencies per
object, object hashes, and an up-to-date original Ninja dependency graph
(5864 dependencies/object). Those objects were copied into an owned cache;
an audited compiler launcher lets the normal CMake `_C` target consume them
and records the relocated dependencies. It invokes the real compiler on any
mismatch. All modified code was freshly compiled. This is an incremental
build, not a claim that every unchanged object was rebuilt from scratch.
The installed extension has no RPATH/RUNPATH or private DSO dependencies;
its dependencies are the declared CUDA/Torch libraries and system C++/C
runtime. Build-cache files are not runtime dependencies.

The first isolated baseline startup compiled FlashQLA in an empty task cache.
When checkpoint loading overlapped four native compiler processes, the 5 GiB
host-memory guard stopped only the owned service tree. No benchmark request
completed in that attempt. Subsequent full-model startup must wait until the
large native build completes; low-memory microbenchmarks can run concurrently.
The effective unchanged residency log is 402/512 experts plus eight spare GPU
slots, 136 host slots, 9.46 GiB GPU expert slots and 3.14 GiB pinned host per
rank, matching the historical deployment with configured fraction 0.8.

## Segment-policy full-model result

Runtime source `3faaa58d08`, native `_C` SHA256 `5d389732...`, preserves all
deployment settings above. With profiling disabled, the same three measured
prompts take **6.977 / 6.907 / 6.885 s** to first text. Median TTFT is
**6.907 s / 5010 tokens/s**, versus baseline **7.513 s / 4606 tokens/s**:
8.1% lower latency, and another 0.192 s below the first candidate. Server
prefill times are 6.647 / 6.613 / 6.609 s. The five prompt hashes/token counts,
including two warmups, match the earlier runs. This remains below 6000 tokens/s.

After unprofiled timing, a separate 8667-token Torch profile verifies actual
execution: each of the four workers records **136 native affine-BLAS calls
and 136 affine-dequantization CUDA kernels**. Its timing is not credited as
unprofiled performance. Raw evidence is `service-candidate-final-measure.json`,
`service-candidate-final-route.json`, `final-route-check.json`, and four
compressed `final-route` worker traces.

The matching original baseline passes both 131070- and 260089-token quality
tasks, with identical prompt hashes to the first candidate. It returns the
expected retrieval/arithmetic JSON, normal EOS and maximum ten-word repetition
of one; completion lengths are 173 and 152 tokens. These are quality results,
not matched throughput tests. The segment fix passes all three matched tasks:
32761, 131070 and 260089 tokens, with 166, 155 and 155 output tokens.
Retrieval/arithmetic, normal EOS and repetition checks all pass. Full response
times are 8.917, 56.448 and 774.606 s. The long task retains the same
131072-token staging cap and host fallback as the original baseline
(802.810 s); this is a quality boundary, not the 32K throughput workload.
`paired-quality-summary.json` verifies matching hashes. The owned service
was stopped after these checks.

Draft PR: <https://github.com/1CatAI/1Cat-vLLM/pull/1205>. The original deployment
is unchanged. The following experiments are separate from that measured runtime.

## Subsequent root-cause screens

An eight-warp 64x256x32 down tactic uses 168/175 registers per thread for
U2/IQ4, allowing only one CTA and still eight warps per SM. All 40 GPU checks
and four real-weight chains pass (bit-exact), but complete chains show no
stable gain. Layer 0 at 7344 tokens is 6.622 / 6.672 / 6.691 ms for control /
candidate / repeated control; layer 1 is 6.163 / 6.178 / 6.270 ms. The tuner
still selects the old tactic for the important 40960/32480-route down calls.
**Reject** the tactic. The source and installed `_C` are restored to the
qualified implementation; `wide-down-rejected.json` records archive and hash
checks. Installed rejected SHA256 was
`7b79c4267de2e9f8f8ec98a669429c27efbd5254e6e3f974c4a6ad6a9bc1dfbc`.

Fresh-process CUDA work-queue counts 8 / 32 / 8 give foreground medians
1.520 / 1.515 / 1.522 ms during background H2D (idle about 1.37 ms).
`dma_dispatch.py` uses 12 streams at each priority, 16 x 16 MiB copies, and
32 queued four-million-float adds. No material benefit appears in this
screen; **do not promote** a queue-count override. It does not establish that
every copy size or model dependency pattern behaves identically.

### Contiguous QSA history staging

Hypothesis: mapped-host `_stage_pages` uses SMs while waiting for PCIe;
a contiguous copy-engine transfer can overlap the main compute stream.
Existing host-copy versus mapped-gather branches at 1785 FP8 pages give
5.444 / 3.338 / 5.357 ms for a concurrent four-HC-projection chain. Isolated
copy time is not faster. About 53 separate copy runs regress to 6.517 ms;
1785 individual runs regress to 153.915 ms. Admit only one contiguous run.

The candidate adds `stage_host_run` to the existing QSA native component,
uses CUDA's mapped-host alias and the caller's current stream, and copies
encoded history and scales exactly. `HostQSAKV` selects it for one run of
at least 512 KiB. Fragmentation, smaller transfers, actual device history,
other hardware or an older native artifact preserve the existing gather.
It retains the same stage owner, capacity and ready/free events, and allocates
no extra device storage. Native binding is independent of decode admission.

The 47 focused GPU tests pass, covering FP8/FP16, pinned/raw host allocations,
offsets, invalid ranges/scales, threshold neighbors, fallbacks, prefetch and
partial-page updates, plus graph replay with changed source bytes/scales.
Actual-native microbenchmarks confirm one successful DMA call for contiguous
layouts and zero for fragmented fallbacks. FP8 concurrent-chain medians are
5.463 / 3.309 / 5.218 ms; FP16 is 8.207 / 6.501 / 8.407 ms. These are synthetic
single-GPU overlap results, not a model latency claim.

The normal `_sm70_qsa_device_C` CMake component builds and installs SHA256
`cf09ff3f9624897f66a6590bc4eea8381aba40790c55697c33308ab6e71d6716`
after whitespace formatting (tested pre-format artifact `fa4dc0e8...`). The
qualified control's QSA SHA256 remains
`ed1d3a854e673cec3c646076bbc7d3b3010d5840fa4c17867d1fc445a2e41eac`.
The first full-model screen at source `481cf2b1e4` gives median TTFT
**6.979 s / 4971 tokens/s**, not an improvement over the qualified control.
A separate 8667-token profile shows **zero native DMA calls on every worker**, while
the 136 affine-BLAS calls/worker remain present. Therefore this does not
measure the proposed DMA path; **do not promote or credit a model speedup**.
The performance screen fails its admission criterion, so longer quality
runs are not warranted. Follow-up work records actual stage-plan geometry
and screens the normal native copy operation on 2/4/8/approximately 53 runs.
Those measurements must distinguish failed admission from copy contention.

### Bounded-run QSA admission and merged dense projections

A temporary DEBUG probe on the owned server shows native-copy binding present
on all four workers, but the full-model page plans contain two or three runs.
One 1785-page plan is `(source, target, count)` = `(1989,0,51)`,
`(3672,51,459)`, `(4182,510,1275)`. This explains the zero-call result above;
the copy operation itself was available. Diagnostic logging is removed from
the candidate after capturing this geometry.

The follow-up clips each run to the actual `[first,last)` transfer before
counting, and admits at most four active runs with at least 512 KiB total.
It checks the run count before submitting any copy. More fragmented transfers,
small copies, device history and unavailable native bindings retain gather.
The existing constructor still requires D256 and positive page sizes divisible
by four; these are not new geometry support claims.

`qsa_stage_dma.py --native-copy --native-run-screen 4 --layouts
serving_three_runs` reproduces the captured layout (with a common source-page
translation) through the normal native artifact. Concurrent four-HC-projection
chain medians, gather / DMA / gather repeat, are **5.029 / 3.311 / 5.343 ms**
for encoded FP8 and **8.410 / 6.379 / 8.178 ms** for FP16. Exactly three native
copy calls execute and bytes/scales match exactly. A two-run full plan whose
copied 892-page prefix lies in only one run gives **3.715 / 2.207 / 3.599 ms**
for FP8 and **4.934 / 3.243 / 5.327 ms** for FP16, with one actual native copy.
These are overlap microbenchmarks, not serving latency savings.

The earlier fragmented-copy rejection used Python tensor views/copy calls.
The native 2/4/8/approximately-53-run screen does improve this overlap workload;
it does not establish that fragmentation inherently defeats native DMA. The
four-run bound conservatively covers the observed full-model geometry.
The extended tests cover clipped ranges, 3/4/5-run neighbors and few-run
prefetch/partial-page updates. The first attempt reports 69 passes and two
incorrect test assumptions about unsupported constructor geometry; those two
tests are removed rather than widening runtime support. The clean rerun passes
all **69** GPU tests on the normal `cf09ff3f...` native artifact.

The qualified short serving trace also exposes merged Q5/Q6 matrices that
miss the original four-layout BLAS calibration. `dense_blas.py --merged`
constructs actual rank-zero GDN QKV/gate and QSA Q/K/V groups from real weights,
including independent GDN Q/K/V TP slices and replicated QSA KV heads.
All ten M7344/M5347 cases pass the predeclared finite/FP32/paired 0.003 gate;
paired outputs are bit-exact. Timings include weight dequantization each call:

| Physical projection | M | Fused ms | DQ + BLAS ms | Fused repeat ms |
|---|---:|---:|---:|---:|
| GDN4 Q6 QKV + gate, N4096 | 7344 | 2.806 | 1.854 | 2.791 |
| GDN4 Q6 QKV + gate, N4096 | 5347 | 2.002 | 1.367 | 2.068 |
| GDN38 Q5 QKV + gate, N4096 | 7344 | 2.942 | 1.772 | 3.055 |
| GDN38 Q5 QKV + gate, N4096 | 5347 | 2.225 | 1.381 | 2.233 |
| QSA31 Q6 QKV, N3584 | 7344 | 2.302 | 1.619 | 2.552 |
| QSA31 Q6 QKV, N3584 | 5347 | 1.779 | 1.192 | 1.766 |
| QSA19 Q5 QV, N3328 | 7344 | 2.285 | 1.473 | 2.515 |
| QSA19 Q5 QV, N3328 | 5347 | 1.851 | 1.096 | 1.866 |
| QSA23 Q5 Q, N3072 | 7344 | 2.296 | 1.380 | 2.310 |
| QSA23 Q5 Q, N3072 | 5347 | 1.731 | 1.083 | 1.708 |

The candidate adds these five physical layouts at M >= 2048 to the same
affine provider, without increasing its shared workspace maximum. Threshold
microbenchmarks, restored-segment execution and affected graph/compiled/mixed
output-view checks gate the next owned full-model run. Selection and speed
are still separate from paired task quality; no new serving result is claimed.

`moe_reduce_traffic.py` separately rejects a routed FP32-atomic output
epilogue traffic screen before a large native build. At 4096 tokens, the
materialize-plus-unroute control takes 0.814 ms (repeat 0.810), versus
1.203/2.225 ms for 256/1024-column atomic tiles including zeroing and the
final FP16 conversion. At 3248 tokens, control is 0.655/0.656 ms versus
0.863/1.678 ms. Maximum paired row-relative L2 is at most 0.000057, within
the predeclared 0.001 limit, with finite outputs. Both paths read an identical
synthetic FP16 GEMM result; this does not predict exact fused-GEMM timing,
but the proposed output traffic does not justify implementation.

Raw local evidence is retained under
`/tmp/flashnext-prefill-6k-20261014/`; remote task logs are under
`/home/test/flashnext-256k/codex-prefill-6k-20261014/logs/`.
Relevant files: `wide-down-*.json`, `dma-dispatch-*.json`,
`qsa-stage-{64,256}.json`, `qsa-dma-native-micro.json`, and
`qsa-dma-tests.log`. No shared venv or production source was changed.
