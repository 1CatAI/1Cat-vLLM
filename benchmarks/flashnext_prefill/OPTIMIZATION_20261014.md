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

The restored-segment screen rejects two initial M2048 admissions: QSA31 Q6
gives only 4.3% improvement and QSA23 Q5 about 1%, below the predeclared 5%
margin. Follow-up QSA31 at M6144 is 2.067 / 2.064 / 2.108 ms, so its calibration
is removed entirely. QSA23 passes at M4096 (1.298 / 0.882 / 1.291 ms) and
M6144 (1.996 / 1.794 / 1.994 ms); its minimum is raised to 4096 tokens.
The two GDN layouts and QSA19 retain M >= 2048. There are thus eight new
Flash-Next layouts in total, including the original four, with no increase in
the shared workspace maximum. The original eleven segment tests pass; two
follow-up threshold tests cover M4095/4096/4097/8193, eager/compiled execution,
changed-input graph replay, native route hits and small-M decode. Removing
QSA31 leaves ten cases in the current affected test file.

The combined bounded-run QSA and corrected dense candidate gives **6.889 s /
5023 tokens/s** on the five matched cold-prefix requests at chunk size 8192.
Its three measured TTFTs are 6.905 / 6.889 / 6.857 s. The independent short
profile proves selection on all four workers: 26 native staging calls, 50
linked H2D copies (35,564,544 bytes), and 156 affine-BLAS calls per worker.
Despite these route hits, it fails the predeclared 6.86 s serving gate against
the qualified 6.907 s runtime. **Do not claim a new serving improvement or
run expensive long-context quality on this result.** The owned service exits
normally. The earlier 6.907 s result remains the fully qualified result.

### Chunk size and autotuner coverage

The 34,688-token trace has five main prefill chunks: 7344, 7344, 7344, 6528
and 6128 tokens. A first larger-M MoE screen that still tunes at 8192 is not
a valid test of a correctly initialized larger-chunk service. At M8976, the
layer-zero gate/up takes 10.581 ms and the complete chain 14.155 ms under 8K
tuning. Fresh-process tuning at 9216 reduces these to 3.688 / 7.536 ms;
repeating 8K tuning restores 10.611 / 14.251 ms. All checks against the
predeclared FP32 numerical bound pass. Four real expert formats at M8976
give complete chains of 7.536 / 7.503 / 7.753 / 7.476 ms with 9216 tuning;
at M7344 they give 6.560 / 6.275 / 6.645 / 6.449 ms.

The existing runtime tuner already takes its first actual prefill row count
into account. No tuning-policy source change is justified by this finding.
The next service experiment uses a separate cache and chunk size 9216; all
other startup settings, including 256K capacity, expert residency, MTP4,
prefix caching and FULL graphs, remain fixed. Actual chunk shapes, native
execution, unprofiled TTFT and task quality must establish its result.

Six remaining affine shapes pass a direct DQ-plus-BLAS screen, but they are
not added to runtime calibration. A benchmark-only forced-band option in
`dense_segments.py` allows the serving restoration cost to be tested before
admission and records both runtime and forced bands separately.

`qsa_fp16_tiles.py` tests the existing FP16 tile-loop parameter on real main
QSA selections, decoding saved FP8 history to FP16. Tiles 1 / 2 / 4 / 8 / 1
take 25.140 / 23.688 / 24.788 / 25.019 / 25.137 ms. Finite, paired and
independent FP32 row checks pass the predeclared relative-L2 0.003 limit.
This is not a draft-model quality qualification; the best 6% kernel gain
would contribute less than approximately 10 ms at the measured model's
draft attention share. No runtime tile default is changed.

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

### Initial GPU queue: indexer and HC screens (resolved below)

At 2026-10-14 00:31 UTC the other task's paired test has completed, but its
port-8001 service remains resident on all four GPUs. No preemption is authorized;
the owned queue waits for the locks and empty devices. Runtime source for the
queued chunk-9216 experiment matches `ae2f6da127` (nine changed runtime files
checked by SHA256). The installed indexer remains the original
`c5b7bb4abe18118488418081d085498049a99e0c665e913fb07559beb8052aba` artifact.

The pending microbenchmarks are hypothesis screens, not qualified improvements:

- `hc_combine_norm.py`: the existing HC4/D2560 combine/norm kernel at M7344
  and M8976, testing warp/tile parameters with identical preallocated outputs.
  Finite and FP32/paired maximum-row relative-L2 must be <= 0.001. Rounding the
  materialized combine result to FP16 remains part of the reference.
- `qsa_indexer_tiles.py`: the existing 32/64/128 MiB score budget and a repeated
  32 MiB control. It includes the actual outer logits-cap loop and the existing
  native top-512 selector; at N8672 the 32 MiB policy limits each outer chunk
  to 1934 rows. Timing a single M7344 score matrix would not represent this path.
- The same script's `--native` arm screens a source-owned fused prefill
  scorer, reusing the decode kernel's Volta MMA fragment layout for 8/32 query
  rows per CTA. It writes reduced FP32 logits directly without a four-head
  score temporary. No Python runtime admission changes are made. Its normal
  CMake component is installed only in the independent `indexer-probe` tree,
  SHA256 `2926a624623cad653c8afb63a18528099f42ae5602da2e9b0cbd5f194fea3b38`.
  The two variants use 34/66 registers with no stack/local memory. Dependencies
  are standard CUDA/Torch libraries; CPU import/registration succeeds. These
  checks are not evidence of CUDA execution or correctness.
- `qsa_indexer_tp.py`: the model uses a replicated index projection. Partition
  query rows over TP4, score/select with the existing provider, then all-gather
  512 block indices per row. Timings include padding, copies, native selection
  and PyNccl communication, and use maximum-rank CUDA-event time. Test M7344,
  M8976 and M7345 (uneven partition). A second pair includes token expansion
  and all-gathers the full 2051-token selection, measuring the simpler runtime
  integration's larger transfer. Expansion from the same selected blocks must
  be exact on every rank. Key gathering is common to both arms and excluded.
  No cache-state writes or runtime dispatch change.

Before any indexer candidate runs, the numerical screen is fixed as follows:
exact finite/negative-infinity mask; maximum row-relative L2 <= 0.0001 against
control and independently computed FP32 checked rows; distinct valid selected
indices; at most one block replacement per row and <= 0.00001 of all selected
blocks. Exact selection equality and replacement counts are reported separately.
The selection allowance accounts for FP32 reduction ordering at the ranking
boundary; it is not output-quality acceptance. Any promising candidate still
requires representative paired task/EOS/repetition checks, including long
context, before promotion. No failed numerical limit has been relaxed.

Remote queue order: `chunk9k-queue.sh` (restored dense screen, then owned service
and its checker), `hc-combine-queue.sh`, `qsa-indexer-tiles-queue.sh`,
`indexer-prefill-queue.sh`, `qsa-indexer-tp-queue.sh`. Each micro runner checks
GPU locks and occupancy again before executing. The experimental fused scorer
is not installed into the serving candidate or production tree.

The native candidate must improve the complete bounded score/selection chain
by at least 20% at both measured query sizes before
`indexer-prefill-tests-queue.sh` runs its 32 new GPU boundary/graph/invalid-input
cases and 16 existing shared-key decode regressions. These tests are prepared,
not yet executed. All applicable source checks pass; no GPU test pass is
inferred from compilation or import success. The current service's startup
state is recorded privately for exact model/configuration restoration only if
the user authorizes a temporary pause; no signal has been sent to that service.

### Authorized four-GPU session: QSA partition and PLE prefill

The user subsequently authorized stopping the resident service and using all
four GPUs. Its exact startup state was saved privately; only that idle service
and its owned guard were stopped. The shared venv and production source remain
unchanged. Current raw evidence is retained outside the checkout under
`artifacts/flashnext-prefill-20261014/` in the workspace; the remote task log
directory above remains authoritative. The earlier local `/tmp` directory is
not available on this desktop host.

The queued screens now have measured decisions:

- HC combine/norm launch tuning is rejected. At M7344 the control/repeat are
  0.780/0.775 ms and the best variant is 0.775 ms; at M8976 the control/repeat
  are 0.942/0.945 ms and variants span 0.942--0.948 ms. Numerical checks pass,
  but these changes do not establish a material gain.
- Increasing QSA score budget from 32 to 64/128 MiB reduces score/selection
  time, but adds transient memory. At M7344/N8672 the times are
  10.172/5.855/5.410 ms, with a 10.998 ms repeated 32 MiB control. This is
  evidence of repeated tiled work, not an admission to raise the memory cap.
- The experimental fused native scorer is rejected by its predeclared gate.
  At M7344 its best 7.230 ms result improves only 7.4% over the faster 7.808 ms
  control, below the required 20%. M8976 improves from 11.646 to 8.766 ms,
  but both sizes must pass. The proposed CUDA changes and unrun tests are
  archived outside the checkout; the original indexer source and installed
  serving artifact are retained. The conditional GPU test queue exits 77
  (skipped), not passed.
- Query-row partition over TP4 justifies integration. The existing scorer,
  native selector and an AllGather of 512 compressed block indices take
  2.531/3.572/2.887 ms at M7344/M8976/M7345, versus faster replicated controls
  of 9.766/11.564/9.422 ms. All four ranks produce exactly equal selections.
  This screen excludes the common key gather; the complete selector is
  measured separately below.

The opt-in `kernel_config.sm70_sparse.qsa_prefill_tp_rows` path binds the
existing TP group in the QSA model adapter, which qualifies replicated queries
and index-cache content. It partitions queries, runs the existing bounded
cuBLAS/native-top-k stages and gathers compressed indices over the existing
communication provider before expansion. It preserves the 32 MiB score budget,
cache writes and output-buffer identity. Admission is SM70, FP16, TP4 with
DCP=PCP=1, one request, H4/D128, 2048 selected tokens, compression ratio 4,
4096--9216 query-token rows, and sufficient existing cuBLAS work per partition.
Other shapes, mixed requests, smaller/larger chunks and CUDA graph capture use
the original selector. Context remains bounded by the existing paged-cache
contract; the 4096-row boundary is a query-chunk size, not a context limit.
The flag defaults to false and participates in the active QSA policy hash.

`qsa_prefill_tp.py` tests the complete serving selector, including key gather,
masks, padding, selection, compressed-index AllGather and expansion. Maximum
rank times (control / candidate / repeated control, ms) are:

| Query rows / compressed key columns | Control | TP4 | Repeat |
|---|---:|---:|---:|
| 7344 / 1836 | 3.659 | 2.110 | 3.276 |
| 7345 / 8672 | 12.057 | 4.777 | 11.288 |
| 9216 / 8672 | 14.460 | 4.361 | 14.204 |

All four ranks have exact output equality, valid causal masks and unchanged
guard regions. Coverage includes query rows 0, 5, 4095/4096/4097, 7344/7345,
9216/9217, mixed requests, 65536 compressed columns (256K-token context), and
changed-input graph replay through the fallback. Actual AllGather counts prove
one call per admitted selector and zero for fallback. An initial harness run
omitted policy resolution and did not enter the fast path; its artifacts are
marked `unresolved-policy-*` and are not performance evidence.

The first chunk-9216 service attempt failed before producing a timing: PLE's
padded-history convolution needed another 176 MiB with only 73.94 MiB free.
The new opt-in `kernel_config.layer_execution.ple_prefill_conv` removes that
history materialization. A Triton stage reads request-local rows and old state
directly; a separate stage commits valid base states, then native FP16 SiLU
retains the existing rounding boundary. Empty/null slots and speculative state
tails are preserved. The initial acceleration contract is SM70 FP16,
10240 channels, four taps, dilation 3, base state length 9, contiguous channels
and device metadata, including strided state layouts. Other layouts and the
default false policy retain the existing convolution. Policy hashes distinguish
the enabled path on models with PLE consumers.

The complete PLE convolution/state/SiLU benchmark gives:

| Requests / query rows | Faster control ms | Candidate ms | Peak temporary MiB, control / candidate |
|---|---:|---:|---:|
| 1 / 7344 | 7.827 | 0.941 | 287.7 / 143.4 |
| 1 / 8976 | 8.634 | 1.081 | 351.4 / 175.3 |
| 4 / 1600 each | 6.277 | 0.805 | recorded in raw JSON |
| Mixed 4, 0, 13, 7 with a null slot | 2.575 | 0.491 | recorded in raw JSON |

The predeclared output limit is finite maximum row-relative L2 <= 0.001 with
exact state updates; all measured outputs and states are actually bit-exact.
The GPU suite covers initial/no-initial state, lengths 0/1/8/9/10/16/17/2049,
mixed empty/null requests, decode offsets, transposed cache layout, changed
inputs and reordered state slots during repeated graph replay. The initial
combined run gives 48 passed and one test-only missing-import failure; after
fixing the import, all three new policy tests pass. The extended graph cases
give 2 passed. Together these runs cover all 27 PLE cases and 23 policy cases;
there is no claim of a single clean combined test log. All affected pre-commit
checks pass, including architecture, runtime ownership and generated reference.

An owned chunk-9216 service now tests both flags with the prior dense and QSA
DMA changes. It retains 256K context, TP4, 80% GPU / 20% host experts, host QSA
KV, disk PLE, MTP4, prefix caching, thinking, tools and FULL decode graphs.
All 16 candidate runtime files are frozen and SHA256-matched to the local
checkout before startup. The benchmark uses the same five prompt hashes and
actual token counts as the qualified 6.907 s result. Its predeclared screen is
median TTFT < 6.78 s, followed by paired 32K/128K/near-256K retrieval/arithmetic,
EOS and repetition checks. An independent full-prefill trace establishes
execution and chunk shapes; profiled timing is not used as serving performance.
At this checkpoint startup is still running: no end-to-end gain or default
promotion is inferred from the microbenchmarks.

The first combined service then passes startup and PLE but fails on the first
request in HC's second AllGather: a 44 MiB allocation finds only 33.94 MiB
free. It produces no valid timing. The HC projection still holds its completed
gate, LoRA, packet and unpacked-weight tensors while allocating gathered and
assembled outputs. Releasing each temporary after its last queued use changes
no arithmetic, transport or storage contract. `hc_lifetime.py` measures the
actual serving function with packed weights and real four-rank NCCL transport:
temporary peak falls from 131.0 to 71.8 MiB at M7344, 159.3 to 87.7 MiB at
M8976, and 163.4 to 90.1 MiB at M9216. Outputs match exactly on every rank,
including three changed-input graph replays at M7344. Timing is variable and
does not establish a compute speedup; this is a memory-lifetime fix.

The updated `nvlink9k-mem` service freezes and matches 17 runtime files,
retains the same model settings, and repeats the same serving/quality gate.
No other process owns the GPUs. The first attempt's logs are retained separately.
An additional benchmark-only HC row-partition screen is prepared: it includes
two transient weight gathers and the final result gather, without increasing
resident weight storage. It must beat both faster repeated controls by 10%
at M7344 and M8976, with finite block/injection outputs and paired row-relative
L2 <= 0.001, before any runtime integration is considered.

That HC row-partition screen is rejected: complete chains take 1.693 / 1.687 /
1.992 ms at M7344 / M8976 / M7345, versus faster repeated controls of 1.363 /
1.567 / 1.334 ms. The additional weight transport and layouts outweigh the
improved GEMM geometry. All numerical checks pass; maximum block error is
0.000074 and injection error 0.000944. No row-partition HC path is added.

The `nvlink9k-mem` service also fails on its first request, now at the HC output
assembly allocation. Reclaiming HC temporaries alone does not make 9216-token
capacity viable under the fixed GPU/host expert and KV settings. Its timing and
quality are unqualified. The next owned service, `nvlink8k`, retains the original
8192-token limit and the same 17-file candidate to isolate QSA/PLE/HC-lifetime
changes from larger-chunk memory pressure. It uses the same predeclared 6.78 s
performance screen and subsequent output-quality checks. No GPU-residency or
context-capacity concession is used to make the larger chunk fit.

The 8K candidate passes the predeclared serving screen with median TTFT
**6.746936 s / 5129.143 tokens/s**, versus the fully qualified 6.907495 s
result (2.32% lower) and original 7.513460 s baseline (10.20% lower).
Three measured TTFTs are 7.064238 / 6.746936 / 6.722450 s on the same
34688 / 34606 / 34631-token prompts. The first measured request still has
warming drift, as in the earlier control. Pure server prefill times are
6.623776 / 6.418930 / 6.478647 s (median 6.478647 s). All five prompt hashes
and actual token counts match the qualified comparison. These measurements
are unprofiled and remain below the 6000-token/s objective.

The paired 32761-, 131070- and 260089-token tasks pass with 166 / 155 / 155
generated tokens,
correct retrieval/arithmetic JSON, normal EOS and maximum ten-word repetition
of one. The near-256K check takes 816.461 seconds through the original
host-KV fallback above 131072 context tokens; it is not a 32K speed result. The
independent full-prefill trace sees 65 compressed-block AllGathers per worker
and ten new PLE kernels (five convolution/state pairs). Actual query chunks
are 7344, 7344, 7344, 6528 and 6108 tokens for its separate 34668-token prompt.
Its profiled TTFT is 7.596847 s, so its larger collective waits are not an
unprofiled communication-cost estimate. Native execution is established;
hardware SM/Tensor Core/link utilization counters have not been collected.

## HC residual-row ownership candidate (not yet checkpoint-qualified)

The user has authorized bounded numerical deviations when paired task quality
shows no material regression. Bitwise tensors and identical generated token
sequences are not the admission criterion. The existing 0.16-second increment
does not satisfy the user's request for a substantial improvement. Additional
PR updates are paused while testing a structural HC change; the 6.746936-second
qualified result remains the serving reference, not a claim of 6000 tokens/s.

This candidate assigns one contiguous quarter of multi-stream residual query
rows to each physical TP rank. Attention outputs reach owners with FP16
ReduceScatter. Routed and shared MoE contributions remain separate until an
FP32 sum and ReduceScatter, then narrow to FP16. HC combine and normalization
operate on local residual rows. Projection uses transient gathers of the
existing lossless weight shards, then gathers the single-stream block input.
Injection factors and residuals stay local. PLE receives a gathered residual
at its original state boundary. The final MTP hidden-buffer copy gathers all
rows and retains its original capacity and layout. This is a complete stage
change, unlike the rejected HC projection-only row-partition screen above.

`kernel_config.layer_execution.hc_prefill_sequence_parallel` defaults to false.
The accelerated subset is SM70 FP16, HC4 / hidden 2560 / lowrank 320, existing
sharded HC storage and HCX, TP4/PP1/DCP1/PCP1, one request, no LoRA or HCX
diagnostics, backed dynamic shapes, and FP32 GEMM accumulation without FP16
partial reductions. Query chunks must have 4096--16384 rows
divisible by four, without intermediate/deepstack inputs or graph capture.
Other cases keep the original backbone. These are query-token chunk units,
not context-token limits; neither expert residency nor the 256K context limit
changes. A separate compiled view shares the original parameter/state owner,
so original prefill and decode graphs remain distinct. One-forward-call
context binding restores deferred-reduction state after nested calls and
exceptions and is isolated across threads. The opt-in policy participates in
the consuming engine's hash and is excluded on models without HC consumers.

The complete-stage microbenchmark includes the producer AllReduce or
ReduceScatter, combine/norm, both HC projections, every weight/result gather
and all layout copies. Maximum-rank repeated-control and candidate times are:

| Query rows | Control ms | FP16 ReduceScatter ms | FP32 ReduceScatter ms |
|---|---:|---:|---:|
| 7344 | 2.650624 | 1.989990 | 2.329498 |
| 8976 | 3.126064 | 2.108058 | 2.602342 |
| 11424 | 3.954688 | 2.480230 | 3.149824 |

At 8976 rows, temporary peak falls from 578132480 to 195336192 bytes, and
resident residual state for the call falls from 183828480 to 45957120 bytes.
The timing screen is meaningful only for this complete synthetic HC stage;
it does not establish a checkpoint prefill gain or hardware utilization.

The original paired maximum-row-relative-L2 screen of 0.001 is preserved as
failed: four-component injection logits near zero give ratios of approximately
0.00145--0.00325. Before subsequent runs, an independent FP32 reference and
operation-aware screen were declared: finite outputs; hidden/block maximum
row-relative L2 <= 0.001; injection-logit aggregate relative L2 <= 0.001;
maximum logit error relative to each reference column's RMS <= four FP16
epsilons (0.00390625); and absolute error of the actual
`2*sigmoid(logit/4)` injection factor <= 0.0005. The control and both candidate
reduction variants pass. FP16 candidate aggregate logit error is approximately
0.00026, similar to the control; FP32 reduction is approximately 0.00021.
This independent-reference screen does not re-label the failed paired metric.

The continuous-state screen declares state/block/logit relative L2 <= 0.003,
injection-factor absolute error <= 0.0005 and finite values before execution.
Checkpoints 1/8/16/32/48 all pass against independent FP32 arithmetic. At step
48 the maximum hidden-row relative L2 is 0.0008562 and maximum injection-factor
absolute error is 0.00045955. This synthetic recurrence is not a substitute
for checkpoint long-context and output-quality evidence.

`hc_sequence_runtime.py` reuses the actual model/decoder forward, packed HC
weights, producer overrides and NCCL operators in a two-layer synthetic
attention/MoE/PLE backbone. Rows 4096/7344/11424 pass on all four ranks:
maximum block/MTPhidden row-relative L2 is approximately 0.00044/0.00046,
PLE-state aggregate error is approximately 0.00025--0.00027, capacity tails
remain untouched, and each case executes zero AllReduces, four ReduceScatters
and seventeen AllGathers. Fullgraph dynamic compilation and changed-input
low-level graph replay pass at 7344 rows. High-level entry intentionally keeps
ordinary CUDA graph calls on the original backbone. The first harness attempt
failed because its PP fixture created a new SimpleNamespace inside Dynamo;
that failure is archived separately and is not a runtime failure or a pass.

The model-forward/producer CPU checks initially report 34 passed and one stale
fixture missing `model_config.quantization`. The repaired fixture passes its
focused retest. Configuration-only checks subsequently pass on a local
UnspecifiedPlatform harness without native kernels: HC policy engine/hash
isolation and the affected PLE-policy regression, 2 passed. An initial hash
test compared different consumer-filter sets; after fixing the fixture to
compare equal sets, it passes. The subsequent complete model-forward/producer
CPU run on that local harness passes all 37 cases, including explicit
unqualified FP16 accumulation-policy fallback. No native GPU execution is
inferred from this CPU run. All affected pre-commit checks pass before the
last qualification/test additions; the focused rerun for those additions
passes all affected checks, including lint, mypy, architecture,
policy/configuration, documentation and runtime ownership.

Next checkpoint experiment: retain fixed expert/KV/PLE/MTP/sampling settings
and test 9216-token chunks with HC row ownership, using the same five cold
prompt hashes, warmups and actual token counts. The substantial-gain screen
is median TTFT < 6.24 seconds (at least 0.5 seconds below the qualified 8K
reference), followed by paired 32K/128K/near-256K retrieval/arithmetic,
repetition/EOS checks and native execution evidence. The final objective
remains >=6000 actual prompt tokens per first-choice second. Every quality
case must retain correct answers, normal EOS and a maximum ten-word repeat
count <=5; generated wording and token counts may vary. Larger chunks and
wider QSA admission are not yet qualified. SSH handshake timeouts currently
prevent starting this checkpoint experiment; no serving gain is inferred.
The local artifact directory also contains `hcsp9k-source-manifest.json` with
22 runtime-file SHA256 values and `hcsp9k-experiment-plan.json` with the exact
matched performance and semantic-quality gates. They describe prepared work;
they are not remote deployment or checkpoint measurement artifacts.

## Local four-GPU continuation with fixed chunk 8192

The user moved the experiment to the local host and fixed the query chunk
limit at 8192. The remote 9216 attempt was cancelled during startup; its
owned process tree is stopped and the previously active Strata user service
is restored. It supplies no completed serving performance or quality result.

Local physical GPUs 0--3 are V100-SXM2-16GB; 4--7 remain with the existing
service. GPUs 0--3 have a mixture of NV1 and NV2 links, unlike the remote
all-NV2 topology. Driver is 580.178.04, Torch is 2.10.0+cu128, CUDA toolkit
12.8, and CPU affinity is NUMA0's 0--23,48--71. The user explicitly approved
the locally cached abliterated IQ3_S checkpoint. Consequently all speed and
quality comparisons use a fresh local control and the same checkpoint;
the remote 6.746936-second result is not a same-machine comparison.

The local runner retains 256K context, configured 80% GPU / 20% host expert
residency, 8192 device-hot KV tokens, host QSA history, disk PLE, MTP4 with the
local stock safetensors drafter, thinking, tools, prefix caching, FULL decode
graphs, and the same sampling. Both control and candidate include the prior
QSA/PLE/affine changes. The only A/B policy difference is
`hc_prefill_sequence_parallel=false/true`. The same five cold prompt hashes
and actual token counts must match; task answers, EOS and repetition remain
the quality criterion rather than generated-token equality.

Normal package native artifacts already built from this task's owned source
are reused from its qualified remote runtime, SHA256-checked in a fresh local
process. `_C` remains `5d389732...`; all ten packaged native modules and their
hashes are retained in `native-artifacts.json`. No private sidecar or preload
is a runtime dependency. Fresh imports pass under local Torch/CUDA. Compiler,
FlashQLA, Inductor and Triton caches are separate per local A/B phase.

The complete-stage local screen reports:

| Query rows | Faster control ms | Row ownership ms | Temporary peak bytes, control / candidate |
|---|---:|---:|---:|
| 6528 | 2.718874 | 1.832397 | 419410944 / 142062592 |
| 7344 | 2.997709 | 2.092083 | 471837696 / 159820288 |
| 8192 | 3.306189 | 2.256947 | 526319616 / 178274304 |

All four ranks pass the independent-reference screen, including the actual
injection-factor error <=0.0005. The earlier paired near-zero logit metric
still fails and remains explicitly recorded. The actual model-adapter
two-layer screen also passes 4096/7344/11424 rows, compiled execution, changed
input replay, MTP/PLE capacity tails and expected transport counts on every
rank. These runs establish native operation and the local micro gain.

The completed matched local checkpoint A/B gives:

| Phase | Median first-choice seconds | Actual prompt tok/s | Measured pure-prefill seconds |
|---|---:|---:|---|
| Local control | 6.474688 | 5344.813 | 6.335764 / 6.264022 / 6.262821 |
| HC row ownership | 6.241688 | 5544.334 | 6.072138 / 6.039716 / 6.039958 |

The gain is 0.233000 seconds (3.6%), below the substantial-gain and 6000 tok/s
screens. All five prompt hashes and actual token counts match. Both 32761-token
retrieval/arithmetic cases answer correctly, reach normal EOS and have maximum
ten-word repeat count one; generated wording and lengths differ (181/173).
Long quality is deferred until a candidate passes the performance screen.
This small gain is not promoted through another PR update.

The candidate trace has a 6.073-second main GPU span. Rank-zero category time
is MoE 1.5194 s, dense 0.94465 s, QSA 0.90058 s, AllGather 0.55157 s,
HC 0.51150 s, GDN 0.36643 s, ReduceScatter 0.35985 s, miscellaneous kernels
0.19018 s, memcpy 0.05129 s and AllReduce 0.01517 s. Submitted GPU execution
gaps total 0.578681 s; separate host submission gaps are 0.060516 s. This is a
profiled, divisible-by-four prompt and not the unprofiled speed table. Two of
three measured prompt tails currently fall back to the original HC path.
The trace confirms 480 ReduceScatters and 485 row-owned HC projection calls.
FP32 MoE reductions take 0.23169 s and FP16 attention reductions 0.12816 s.
The two-plane MoE payload copy takes 0.07983 s. Adjacent gap classification
guides investigation but does not establish a gap's cause. Hardware SM/HBM
utilization counters are unavailable with the current profiling permissions;
static resource ceilings are not measured utilization percentages.

Forcing NCCL Simple yields the same complete-stage times as the default and
passes the same reference checks: reject this protocol override. A kernel
name ending in RING_LL alone does not establish the selected wire protocol.

The next benchmark-only screen stages TP partials in existing CUDA IPC buffers
and lets each row owner read the four peers through NVLink while combining
residuals and computing grouped RMSNorm. Both scalar NCCL ordering barriers,
the staging copy and the fused kernel are inside the measured prefix. Scaled
FP16 staging (FP32 local sum / 8, restored in FP32 before the FP16 boundary)
reduces 8192-row prefix time from 1.76860 to 0.71204 ms. The complete HC chain,
including projections, transient weight gathers, layout copies and block
AllGather, drops from 3.329 to 2.28495 ms (31.4%); 4096/7344-row gains are
30.0%/30.2%. All four ranks pass the previously declared independent FP32
reference limits, including injection-factor absolute error <=0.0005.
This synthetic screen does not qualify full-range cancellation, continuous
state, request isolation, graph/resource lifetime or checkpoint quality.
FP32 staging instead makes the complete stage 6--11% slower, despite passing
reference checks, so that transport variant is rejected. A packed pair of
unrounded FP16 contributions is being screened to distinguish numerical
rounding benefit from communication/layout benefit before runtime integration.

The packed unrounded FP16 pair passes arithmetic checks but improves the
complete stage only 1.6--2.1%, so it is rejected. Keeping the up bank resident
reduces an additional approximately 0.1 ms per synthetic stage; that small
benefit does not justify a new permanent weight-memory budget here.

The admitted peer-stage prototype instead caps each rank's scaled-FP16
reconstruction error at 2**-14 per element. A row with any larger error or
non-finite source uses an FP32 IPC bank. Both banks and row flags belong to
one engine resource, and two scalar NCCL barriers order publication and bank
reuse. Thus the transport error before the original FP16 narrowing is at most
2**-12 per element across four ranks, apart from FP32 addition association.
The complete synthetic chain retains 27.7--30.9% gain at 4096/7344/8192 rows;
8192 rows trigger one FP32 fallback row on one rank. This is a computation
policy with bounded numerical differences, not an exact transport codec.

Large finite sums with opposite-rank cancellation, a small remainder after
65504-scale cancellation, mixed rows and subnormal inputs all pass. The small
remainder case sends the two problematic ranks entirely through FP32; the
mixed case sends 586 rows on each of those ranks through FP32. Changed-input
CUDA graph replay passes. A 48-step independent-reference recurrence passes
the predeclared state/block/logit relative-L2 <=0.003 and injection-factor
absolute-error <=0.0005 gates. At step 48 the maximum hidden-row error is
0.0008573 and the maximum factor error is 0.0004309.

Runtime integration is opt-in `hc_prefill_peer_norm`, dependent on
`hc_prefill_sequence_parallel`. `sm70_hc_prefill_peer.py` reuses normal package
IPC allocation/close and the existing PyNccl communicator. It fuses FP32 peer
sum, the original FP16 boundary, residual combination and optional grouped
Gemma RMSNorm; projections remain the existing sharded HC implementation.
Prepared resources belong to `runtime_resources_for(config)` and are released
per engine, including peer mappings. Missing operators, NCCL or full NVLink
mesh retain the original stages. Supported HC row ownership remains query
rows 4096--16384 under the previously stated model/dtype/TP/graph contract;
peer acceleration is limited to padded query rows <=8192. No native binary,
shared environment, expert split, context, KV/PLE placement or chunk limit
changes. The current packaged native hashes remain the qualified hashes.

HC alone now pads up to three tail rows. GDN, QSA, MoE and PLE receive only
actual rows, and final sampled/MTP hidden outputs are cropped before storage.
The opaque shard fake expresses `(M+3)//4` without a Dynamo modulo branch.
The actual two-layer model adapter passes 4092/4096/4097/4098/4099/4100,
7344/7345, 8188/8191/8192/8193/8196, 11427 and 16384 query rows on every rank.
Qualified peer cases execute zero original ReduceScatters; threshold/capacity
fallbacks execute four. PLE/MTP capacity tails are preserved. The same dynamic
compiled view crosses divisible/ragged and peer-capacity boundaries, and
changed-input graph replay also passes for 7345 rows. A separate peer-disabled
fallback run passes 4097/7344/7345/8191/8193/11427 rows. The affected CPU tests
pass 45 cases; all affected pre-commit checks pass, including mypy, policy,
architecture and resource ownership. The matched local checkpoint phase
`hcpeer8192` is now starting; no end-to-end peer benefit is inferred yet.

Local artifacts are under
`/home/ymzx/桌面/1cat-vllm/artifacts/flashnext-prefill-local-20261014`:
`source-manifest.json`, `experiment-contract.json`, native hashes, launch
commands, `logs/hc_sequence_{rows,runtime}.{json,log,exit}`, and owned GPU/PID
records. The substantial-gain screen remains at least 0.5 seconds from the
local control, with the final objective >=6000 actual prompt tokens per
first-choice second. Expensive long quality follows a candidate that passes
both this gain screen and the objective; smaller gains first guide further
critical-path work rather than another PR update.

### Local peer result, dynamic compilation and rejected DMA schedules

The first checkpoint peer run (`hcpeer8192`) passes the 32K retrieval/arithmetic,
EOS and repetition checks, but median TTFT is **7.647611 s / 4528.34 tok/s**.
Profiling an already-seen prompt gives 6.138639 s, identifying a per-new-tail
compilation cost rather than a transport win that can be reported to users.
The unused `_prepare_row` ROWS constexpr and `_peer_combine_norm` LOCAL
constexpr produce 20 and 120 shape-specialized kernels. Removing ROWS and
making LOCAL a non-specialized runtime scalar retains all affected arithmetic
and ownership checks. The actual adapter's dynamic ragged/capacity cases pass.

The corrected `hcpeer-dynamic8192` run gives **6.082314 s / 5693.72 tok/s**
(unprofiled trials 6.077313, 6.097683, 6.082314 s; server pure prefill
5.880743, 5.893051, 5.887453 s). All five cold prompt hashes and token counts
match the 6.474688-second local control. The gain is **0.392374 s**, below the
0.5-second screening threshold and the >=6000 tok/s goal. Its 32K quality
check passes. It does not trigger the expensive 128K/near-256K qualification
or another PR update. The initial loading failure from allocating the IPC bank
before the existing allocator-cache release is retained as
`service-hcpeer8192-loading-oom.log`; moving allocation after that loading
boundary is covered by a focused lifecycle test. The corrected service starts
with 256K context and unchanged expert/KV/PLE policies and stops its own PIDs
on completion. The current dirty source is experimental, not shared/deployed.

The corrected main-stream profile spans 5.945506 seconds: MoE 1.520147 s,
dense 0.946963 s, QSA 0.903057 s, HC 0.721746 s, AllGather 0.553742 s,
GDN 0.366420 s, miscellaneous 0.148209 s, AllReduce 0.074073 s, copies
0.051451 s and other BLAS 0.015088 s. Main-stream gaps are 0.636126 s.
They are not whole-GPU idle time. The earlier row-owned trace's all-stream
union shows only 0.112772 s idle: 0.526425 s of its main-stream gaps overlap
other work, with about 0.422 s copy-only intervals. Copy overlap alone does
not attribute critical-path causality. No hardware SM/HBM/Tensor Core/link
utilization counter is available under this account's profiling permissions.

Two production-shape, four-concurrent-GPU MoE schedule screens are rejected.
They use the existing provider, real rank-zero layer codecs, hidden 2560,
intermediate 160, top-10 routing, 8192 rows, 80% resident experts, the normal
native artifact, and include all staging/compute work. Deferring next gate/up
DMA until the current down acquisition is 105.909 versus 102.412/102.829 ms
without an artificial gap; at a 5M-cycle gap it remains slower, and its
20M-cycle gap improvement is only about 1.2%. A second staging bank increases
storage from 444416000 to 798310400 bytes for at most 0.5% at gap zero and
regresses both delayed cases. Exact restart and post-decode host-writeback
checks pass. GPU sleeps here distinguish schedules; they are not a model
compute overlap reproduction. Neither change enters runtime source.

The user's latest clock decision is fixed **1312 MHz** on GPUs 0--3; retain
chunk 8192 and 300 W limits. GPU 4--7 and their other task remain untouched.
The 32/64/128 MiB QSA scorer-budget screen includes top-512 selection, token
expansion, ragged padding and TP4 AllGather. At 8192 rows/8672 blocks, 32 MiB
is 2.615/2.494 ms, 64 MiB 2.370 ms, 128 MiB 2.344 ms. At 6933 rows the
32 MiB repeated controls are 2.420/2.417 ms and 128 MiB is 2.345 ms. All
selected blocks are identical, valid and distinct. The expected total saving
is only a few milliseconds over this prompt; this knob alone cannot close
the remaining approximately 0.31-second gap.

A direct IPC block-gather prototype fuses HC gate mixing into publication,
then reads the four peer banks with two scalar NCCL ordering fences. The full
peer-prefix-plus-projection stage is bit-exact versus the existing stage,
but the first 8192-row variant is **3.242 ms versus 2.309/2.305 ms**; 7344
and 4096 rows also regress. Generated PTX reveals scalar 16-bit remote loads
(and scalar output stores), motivating an explicit alignment/vectorization
screen before rejecting or integrating this transport. This screen remains
outside runtime source. Artifacts: `logs/moe-priority.json`, `moe-banks.json`,
`qsa-tiles.json`, `hc-peer-gather.json`, `hcpeer-dynamic8192-measure.json`,
`hcpeer-dynamic8192-attribution.json`, retained exits/logs and owned GPU records.

### Rejected local shared-head QSA and peer launch/row-tile screens

A benchmark-only TP4 H24 reader passes paired/independent numerical checks and
cuts its synthetic one-K/V-head chain by 49--51% with AllToAll. The checkpoint
adapter correctly refuses this route: Flash-Next's actual contract is 24 query
heads and **four distinct KV heads**, with one KV head per TP rank, not one
replicated KV head across all ranks. The requested opt-in therefore executes
zero shared-head kernels. Its matched measurement is 6.082194 s / 5690.47 tok/s,
and 32K quality passes only through the existing route. This is fallback
coverage, not evidence of checkpoint acceleration. Never remove the KV-head
qualification to apply this synthetic result. All shared-head runtime/API/
policy/test additions are removed from the candidate and retained under
`rejected-shared-heads` in the local artifact directory; original native QSA
artifact is restored. No corresponding PR update is made.

The direct IPC block gather remains slower after pointer alignment and 64-bit
copy screens. At 8192 rows it is about 2.96--3.07 ms versus 2.30--2.31 ms;
its arithmetic is exact. Pointer hints are insufficient to improve this
transport and are not applied to runtime. A complete peer-HC stage screen of
4/8/16 warps and 256/512/1024 column blocks also shows no stable material
benefit; control is 2.303/2.309 ms at 8192 rows, best variant 2.299 ms. None
of these launch variants enters source. Their retained logs/caches distinguish
compiled scalar/vector instructions and complete-stage timing.

A separate 128-thread, 32-key QSA reader keeps the actual H6/KV1-per-rank
contract and reuses the existing QK/PV MMA. It tests 3/4 resident-CTA launch
bounds, with finite paired/independent FP32 maximum per-head L2 <=0.003 fixed
before runs. Both variants pass numerically but regress: at 7344 rows,
13.519/11.908 ms versus 11.398/11.427 ms; at 8192 rows, 15.219/13.380 ms
versus 12.911 ms. The extra softmax/synchronization work outweighs increased
query-row residency. Reject; source and native are archived in
`rejected-qsa-rows32`, then original QSA source/artifact restored.

An isolated HC FP16-accumulation screen (no serving/global-policy change)
gives complete-stage 1.929 versus 2.142 ms at 7344 rows and 2.064 versus
2.306 ms at 8192 rows. The predeclared operation-aware paired limits are
hidden row L2 0.001, block row L2 0.03, injection-factor absolute error 0.01,
logit column-RMS error 0.04, plus finite tensors. Actual factor error is at
most 0.00321 and block row L2 about 0.000201. This is a synthetic numeric/
performance screen, not real-weight or task qualification. A global Torch
backend toggle is not a production implementation; it would affect unrelated
operators and existing admission guards. The next cheap screen uses actual
dense checkpoint weights and the exact NN BLAS layout before considering a
source-owned, explicitly scoped precision variant. Current best qualified
local speed remains 6.082314 s / 5693.72 tok/s at fixed 1312 MHz, chunk8192.

The real-weight dense NN-layout FP16-accumulation screen includes the existing
dequantization and BLAS stages on all four concurrent GPUs. The FP32 Torch proxy
is exact versus the native call. GDN QKV/gate and QSA Q/QV gains are 5.7--12.9%,
with paired row L2 about 0.0041. However, QSA layer 23 at 7344 rows exceeds the
predeclared column-RMS limit: 0.061711 versus 0.06. Do not loosen that limit or
promote this global backend toggle. No runtime precision change is made; logs
and all per-case pass flags remain in `logs/dense-fp16-accum.json`. Exit 0 means
the screen completed, not that all numeric cases passed.

### Full-path follow-up and revised 8K target

The user raised the objective from 6K to **8K actual prompt tok/s over client
TTFT**, while retaining local GPUs0--3 at 1312 MHz, chunk limit8192, TP4,
256K capacity, IQ3_S, MTP4, the 80%/20% expert placement and host QSA/disk PLE.
At approximately 34.6K actual prompt tokens this requires approximately 4.33 s;
do not stop or promote a small PR merely after crossing 6K. All of the
approximately six-second path is in scope, including CPU preprocessing.

The existing MoE custom op already exposes two producer tensors. An explicit
runner/layer `forward_partials` API lets qualified HC sequence forwards pass
routed/shared dependencies directly to the opaque HC consumer. Other forwards
retain the original combine/transform path; reduced and zero-expert runners
explicitly reject this partial API. Overflow fallback keeps the FP32 sum and
private ragged carriers. Compiler liveness, no hidden temporary references,
threshold/capacity fallback, PLE and MTP tails are covered. Complete four-GPU
HC stage timing is 2.141 versus 2.331/2.366 ms at 7344 rows and 2.304 versus
2.514/2.518 ms at 8192 rows, bit-exact on all ranks. Actual matched serving
is **6.025139 s / 5743.60 tok/s**, pure prefill approximately 5.838 s;
32K retrieval/arithmetic/EOS/repetition gates pass. Gain versus original local
control is 0.449549 s. Still no PR update or 8K claim.

Two HC projection transpose/contiguous materializations are unnecessary.
Gate mixing already accepts gathered columns. The existing packet-SiLU family
now accepts explicit rank-column strides, and sequence projection passes both
BLAS outputs directly. Full stage is 2.027 versus repeated control 2.144 ms
at 7344 rows, 2.170 versus 2.300/2.303 ms at 8192 rows. Outputs are bit-exact
on all ranks and three physical/logical rank permutations. Actual model-adapter
rows4096/4097/7344/7345/8192 and capacity fallback8193 preserve state/tails.

The frontend trace locates approximately 134 ms in long-prompt tokenization,
with scheduling queue time below 0.1 ms. A Tiktoken prototype gives 9 ms and
309 sample parity passes, but its static qualification rejects the actual
checkpoint's unusual later-parent BPE merges (including ASCII `anjutkan`).
This is not qualified checkpoint acceleration. Retain the original HF BPE:
the candidate memoizes its own reference-generated pieces on long ASCII
token-ID calls. Unicode, pairs, tensors, unsupported options or >128 new
pieces use HF. Added markers split first; mutations invalidate memoization;
entries/tokens are bounded per tokenizer. HF `__call__` preserves Encoding and
offset APIs. Async token-ID-only singletons use `encode`, while full encoding
and compatible batch calls retain their existing API. Real checkpoint paired
screen: **17.939 versus 126.820/127.287 ms**, 1201 exact-ID cases, including
the unusual merge and changed texts. This does not establish a Unicode speed
gain. Sources, interpreter and normal extension identity are explicit; running
an external artifact script without the owned source path instead imports the
older site package and is invalid evidence. Affected CPU contracts:114 passed.

Real-weight HC raw FP16 accumulation is rejected without changing serving
precision. On six attention/MLP HC branches at layers0/23/47 and two row sizes,
full stages save about 0.21--0.25 ms, but 10 of 12 cases exceed the predeclared
injection-factor allowance0.01 (maximum0.03301); some also exceed logit-column
RMS0.04. Exit0 only means completed; preserve all flags. Do not loosen failed
limits or use a process-global backend toggle in serving. Next main-path
work prioritizes MoE register/padding pressure, dense projection and QSA
memory/compute efficiency over further HC-only polishing.

The first combined layout/token-cache model startup overlaps a CPU contract
process that initializes a CUDA context. It fails the unchanged 0.95 free-memory
startup gate (14.69/15.77 GiB free versus14.98 GiB requested). Failed log/run/done
are retained with `.startup-memory-failed`; retry only after that owned test
process exits and GPUs0--3 are idle. Do not reduce expert placement or 256K
capacity to conceal this startup interference.

The combined lossless HC layouts, explicit MoE partials and reference HF piece
memoization pass the matched local 32K serving screen: **5.897010 s /
5870.43 actual prompt tok/s**, versus original control6.474688 s. The gain is
0.577678 s (8.9%); measured pure-prefill trials are5.835568/5.800153/5.802420 s.
The first warm request is7.610 s pure prefill, and the next5.923 s; retain this
startup/JIT limitation. Five prompt hashes/token counts match control, and the
32K retrieval/arithmetic/EOS/repetition gate passes. This is still below8K;
long-context paired qualification and a PR update remain outstanding.

The newest retained Torch profile covers GPU kernels on ranks1--3 only;
rank0 contains CPU activity but no GPU events. Their main-stream spans are
approximately6.309 s under profiling, with rank1 MoE1.562, dense0.960, QSA0.950,
HC0.711, AllGather0.702, GDN0.376, AllReduce0.121 s. Submitted/host gaps are
0.692/0.079 s; these are not all-stream idle or unprofiled TTFT. Do not add
profiled durations to the unprofiled5.897 s result or claim rank0 GPU coverage.

Standalone source-complete MoE probes reuse the actual TurboMind indexed
LUT4/group32 decoder, scheduler and epilogue, with unchanged FP32 weighted
unroute. The private benchmark DSO is never installed or registered as a
serving provider. Float128 is bit-exact to the owned production native.
The first half-accumulator probe has an incorrect lane/value layout (the
SM70 FP16 layout differs from FP32); its large error is a probe bug, not
numeric evidence. The corrected v2 probe uses the CUTLASS SM70_8x8_16b layout.
Static registers fall from255 to218 (M128), or186 to141 (M64), with zero spills;
this is a residency bound, not achieved utilization. Across real layers0/17/47
and rows7344/8192, full-chain half128 gains are only approximately1--2%.
Half64 regresses about3% at8192. Layer0/47 at7344 exceed the predeclared
column-RMS allowance0.03 (0.033725/0.034281); reject without loosening limits.
All per-case flags are retained in `logs/moe-half-probe-v2.json`.

A dense projection screen retains FP32 accumulation while permitting FP16
chunk reduction in an isolated benchmark process. Outputs are exact to the
native on all eight real-weight/row cases, but full dequantization+BLAS gains
range from-1.30% to+0.24%. Reject; no runtime backend policy is changed.
`logs/dense-fp16-reduction.json` retains all measurements.

MoE v3 compares column/row raster orders using both128-row and64-row FP32
tiles at real layers0/17, rows7344/8192, on four concurrent GPUs. All outputs
are exact, but row128 gives at most approximately0.4%;64-row variants regress
4--11%. Reject raster changes. The old4096-to8192 down-batch screen already
regressed3--8%; do not repeat it alone. Next hypotheses are an intermediate
row tile near the actual expert occupancy, vector IPC AllGather measured
inside the complete HC stage, and actual adjacent-query selected-index reuse
before designing a QSA shared-reader kernel.

The intermediate96-row FP32 down tile follows the actual approximately80-row
expert groups more closely than either64/128. At real layers0/17 and
rows7344/8192, the padded96 variant gives6.510/6.801 and6.552/6.810 ms versus
6.754/7.033 and6.812/7.057 ms. All outputs are exact on four concurrent GPUs.
The80-row tile regresses at8192. This is a3--4% whole-chain compute screen;
the normal serving tuner, other codecs, capacity and end-to-end impact have
not yet been qualified. Keep96 as a cumulative candidate, not a standalone PR.

A native16-byte vector IPC AllGather with interleaved peer sources includes
publication, both existing NCCL ownership fences and the complete HC stage.
It is exact, but4/8-vector variants give2.080/2.098 versus2.027 ms at7344,
2.248/2.247 versus2.165 ms at8192. Reject. This rules out scalar/vectorization
alone as the reason the earlier direct-IPC gather lost to NCCL.

An owned ordinary model request captures60 QSA selected-index tensors:
layers3/23/47, five prefill chunks, four workers. Counts0..4 have
7344/7344/7344/6528/6108 rows, total34668 actual prompt tokens; output width
is2051 expanded token indices, not512 compressed blocks. Sets are identical
across workers and contain no duplicate valid tokens. Ideal adjacent4-query
union savings are approximately40--57% in later chunks,65--68% in the first;
2-query savings23--35% later,41--44% first. These exclude union, shared-memory
and kernel costs, so are a hypothesis, not measured attention acceleration.
The capture wrapper misses a `json` import while recording the already-complete
request; it still stops its owned service in `finally`. Retained tensors and
request metadata recover the manifest; no request re-run is needed. The9.142 s
diagnostic first request includes cold compilation/dumps and is excluded from
performance evidence. Logs retain the finalization error.

The actual-shard TP-versus-expert-partition micro uses the same grouped GGUF
family, replicated inputs, masked sentinel routes and FP32 TP summation.
Each GPU's weights differ correctly:512 experts xI160 for TP,128 experts xI640
for the expert partition. Layers0/17 at7344/8192 give7.510/7.824 and7.565/7.841
ms versus repeated TP controls8.119/8.546 and8.254/8.666 ms. Predeclared paired
row-L2<=0.015 and column-RMS<=0.04 pass (approximately0.00045/0.00875 maximum).
This is resident compute plus routing/summation only. Original GGUF serving
requires DP=PCP=EP=1; loader,80%/20% caching and decode are not qualified by
this screen. Do not bypass that guard or claim expert-parallel serving support.

The next QSA source experiment stays inside the existing staged-history reader
family. A GPU union stores compressed block IDs plus query-specific membership
bits; two/four adjacent queries share KV loads while retaining individual
causality, scales, softmax and output. Duplicate/malformed block groups,
different requests or incomplete groups use the original per-row reader.
The grouped subset has FP8,H6/D256,width2051, at most131072 staged history
slots; other contexts/layouts retain existing behavior. Before runs, paired
and independent FP32 maximum head-L2 limit remains0.003. Actual selected sets
and synthetic finite Q/K/V screen the union-plus-native stage first; no serving
policy is enabled until a meaningful gain and numerical qualification exist.

QSA grouped union readers are rejected. All paired and independent FP32
head-L2 checks pass (about0.0005 versus0.003), but the complete stage loses:
at layer47's6108-row final chunk, original9.516 ms; groups2/4 raw11.996/15.199
ms. Union-only cost is1.051/2.439 ms and native-only10.849/12.697 ms. Sharing
raw loads while computing every union key for each query increases arithmetic
and conversion. Decoding once into shared FP16 tiles preserves the numbers but
needs more than48KiB shared memory, reducing residency to one CTA; native-only
15.119/16.114 ms, complete16.175/18.606 ms. A32-key decoded tile restores the
two-CTA resource ceiling and skips PV blocks absent from each query, but still
loses: native12.410/12.918 ms, complete13.480/15.395 versus9.516 ms. Static
resource counts are not achieved utilization. Source snapshots, native, logs
and exact limits remain in `rejected-query-groups` and the corresponding
campaign snapshots. Original QSA source and native
`cf09ff3f9624897f66a6590bc4eea8381aba40790c55697c33308ab6e71d6716` are restored;
no serving policy/default is enabled. A v2 output-path error overwrites the
v1 JSON; v1 is recovered from its retained stdout and marked accordingly.

GDN changes the leading hypothesis: its existing indexed TileLang core uses
four warps per CTA. A standalone source-complete256-thread factory, with
unchanged arithmetic and state reset included, gives1.079/1.127 ms at7344/8192
versus conservative repeated controls1.578/1.749 ms on four concurrent GPUs.
Rows4095/4096/4097/6108/6528/8193 pass exact output/state/untouched-slot checks;
actual6108/6528 chunks give0.857/0.908 versus1.317/1.392 ms. The normal bundled
factory/public API now accepts an explicit thread count, owned by the existing
GDN policy/plan; default128, qualified256 subset is SM70 FP16,H4/HV12, one
indexed request, FP32 state/gate/beta,4096--8192 query-chunk rows. Other tensors
keep128. No global math state/private serving factory is used. Policy/lifecycle
CPU contracts:51 passed; ordinary API GPU first/continued state, absent/present
state and private slot regressions:4 passed. Actual serving A/B remains next.

The gate/up raster screen is distinct from the rejected down raster screen.
It tests the activation-reuse hypothesis in the existing padded64x64x64
lattice family. Layers0/17 at7344/8192 pass exact hidden/output checks but row
raster gives6.707/6.975 and6.756/7.020 ms versus production6.768/7.049 and
6.777/7.058 ms: at most about1%, not the material gain hypothesized. The
private column control is slightly slower than the actual tuner. Reject this
raster change. The first macro-based standalone compile fails; the corrected
explicit-instantiation artifact runs these four cases, then layer47'sLUT4
codec lies outside its declared lattice subset and rejects. Partial completed
case evidence is retained; the reproducer now skips unqualified codecs rather
than misidentifying that failure as a model/kernel regression. No serving
native or registry changes are made.

### Continued full-prefill optimization (2026-10-15, local)

The target is now 8,000 actual prompt tokens/client TTFT second. About 34.6K
actual tokens require roughly 4.33 seconds. The whole request remains in scope;
no clock, chunk, context, residency, QSA-host-KV, PLE-disk or MTP setting changes.
The normal serving natives remain _C 5d389732... and QSA cf09ff3f... .

The matched GDN128 control gives 5.910302 s / 5,855.20 tok/s, quality32 passes.
An indexed-only256 policy gives 5.919706 s / 5,845.90 tok/s but is NOT accelerated:
ordinary serving uses non-indexed state. Logs and the180 native launches show
128 threads, grid48, batch1. This is an admission miss, not evidence against
256 threads. The new ordinary non-indexed API regression passes four first/
continued-state cases and preserves input-state ownership exactly. Four-GPU
core screens at4095/4096/4097/6108/6528/7344/8192/8193 are exact; at7344/8192,
repeated128 controls1.382/1.537 ms versus256 cores1.030/1.143 ms. The provider
now admits one direct-state sequence (state leading dimension1, cu_seqlens
length2) as well as one indexed sequence, with the existing4096--8192 query
chunk, FP16,H4/HV12,FP32 state/gate/beta contract. Other calls keep128.

A normal complete HC stage compares default8,4,8,16 channels and Simple with
fresh processes. All outputs match all four rank hashes exactly. At8192 rows,
default8 is2.173 ms,16 is2.129,4 is2.356 and Simple8 is2.226. Default actual
AllGather entry symbols have an LL suffix; this does not establish the
selected protocol (see the later NCCL diagnostic correction).16 channels
is an isolated candidate only: both default and2-MiB buffer serving launches
fail before loading on GPU0's fixed0.95 startup memory budget (14.85/14.88 GiB
free versus14.98 requested).2-MiB micro still gives2.132 ms but does not
remove the startup rejection. Retain8 in serving; do not change KV capacity
or lower the quota to qualify this small isolated win.

The existing tuner early-exits when a descriptor already exists, so a first
same-process retune design would not test the hypothesis. The corrected
uniform/profile arms use fresh processes/caches per layer. Prior checkpoint
routing counts (not current-request routes) produce complete8192-row chains:
layer0 7.714/7.713 ->7.299/7.311 ms;17 7.560/7.591 ->7.310/7.331;
47 7.358/7.340 ->7.034/7.088. All paired finite/row-L2<=0.002/column-RMS<=0.03
checks pass. Native traces confirm down128x128x16 becomes existing32x128x32;
GU keeps64x64x64 with changed swizzle. Runtime reuses the existing profile,
seeded Gumbel top-k without duplicates and existing descriptor-family tuner.
Explicit expert_cache_prefill_tuning=profile, defaultuniform; absent, non-finite,
negative, empty or unqualified profile counts fall back to uniform groups.
No kernel/provider/math switch is added. Four ordinary GPU cases cover both
modes, sub-batch boundaries, independent FP32 references and decode interleave.
Policy tests cover snapshots and inactive/canonical/MMQ/no-profile hashes.

The actual GDN-direct plus profile-tuning serving run retains NCCL8 and gives
**5.834715833 s / 5,931.051484 tok/s**, quality32 passes, all five prompt hashes
and actual counts match original control. Total gain versus6.474688 control:
0.639972 s. This is still below6K and far below8K; do not promote a small PR
or claim the objective. The256-thread core actually executes in ordinary
requests. Runtime/native source-manifest drift is empty. Profile timing and
operator attribution remain separate from unprofiled client measurements.

Additional rejected structural screens:

- Exact predecoded-unscaled FP16 QSA history: actual selected layer47 final
  chunk6108,816-token pages. Four-rank outputs are bit-exact; independent FP32
  head-L2 <=0.000458 versus predeclared0.003. Decode itself0.071 ms, but complete
 12.945 versus9.460 ms. FP16 core increases shared memory to49,088 bytes and
  needs128 registers with48-byte stack. Reject; original source/native restored.
-128-key/512-thread QSA: same selected data, unchanged FP32 QK/PV and FP8 storage.
 121 registers,65,216 dynamic shared bytes, no stack; one CTA limit is static,
  not achieved utilization. Paired head-L2 <=0.000494 and independent<=0.000445
  pass0.003, but complete10.766 versus repeated9.387 ms. Reject and restore.
- Two canonical expert bank sets after HC memory reductions: normal GPU API
  four cases pass. Four concurrent eight-layer caches preserve output/restart/
  changed-residency exactly. Banks/staging444,416,000 ->798,310,400 bytes.
  No-gap pass101.256/101.297 versus100.681 ms;10M gap123.689/123.753 versus129.335;
 20M gap172.192/172.257 versus179.271. Gap is synthetic, not model work. Reject;
  bank-count option removed. Exact source snapshots/natives are under the
  corresponding rejected-* artifact directories, with hashes.

Current trace evidence: rank1 cold H2D stream has15,075 copies,13,620,288,000
bytes and3.122 s summed copy duration. This is already background work, not
3.122 seconds to add to the critical path. Main-stream waits must be associated
with those dependencies before estimating saved TTFT.

Next compute screen reuses existing indexed FP16 grouped GEMM and fused SiLU,
with the normal FP16 weight pack and real layer0 weights, four concurrent GPUs.
It excludes decode/pack/load cost to reject the idea cheaply if its maximum
compute saving cannot cover that cost. It removes padded activation batching
from the previously rejected BLAS screen; it does not repeat that implementation.
The private source-complete DSO is benchmark-only and never installed/registered.

### Full-path screens after the 5931 tok/s result

The fresh route trace has a 5.817 s main-stream span. Across traced ranks,
exclusive categories average approximately MoE 1.546 s, dense 0.954 s, QSA
0.900 s, HC 0.695 s, AllGather 0.559 s and GDN 0.303 s. About 0.575 s of
submitted main-stream gaps includes cold-bank/transcode dependencies; this is
not equivalent to all-GPU idle time. GDN falls about 63 ms versus the 128-thread
trace, but profile-tuned MoE slightly regresses (about 17 ms). The combined
client result does not establish a positive tuner contribution. A normal
256-thread/uniform-tuning serving run is queued to isolate that component.

New compute and transport decisions, all benchmark-only:

- Resident indexed FP16 expert weights remove the previously tested padded
  activation batching. Four existing FP32-accumulating grouped tiles give
  6.503/4.318/5.156/5.131 ms versus original quantized GU 3.714 ms (repeat
  3.720 ms), excluding the additional decode/pack/load cost. Row-L2 <=0.00205
  passes the predeclared 0.003 limit. Reject the resident-FP16 implementation.
- Widening padded GU from 64x64x64 to 32x128x64 reduces repeated A loads but
  increases full-chain latency: layer17 8.324/8.963 ms at 7344/8192 rows versus
  6.807/7.063 ms. Hidden/output tensors are exact. Reject the wider tile.
- Dense FP16 accumulation in two K segments, with the original NN layout and
  dequantization, passes the existing row-L2 0.03 / column-RMS 0.06 allowances
  but regresses 2–11% on four real merged projections at 7344/8192 rows.
  Reject; no serving-global math policy is changed.
- Group32 Q8 activations fail the predeclared hidden-row 0.025, output-row
  0.02 and output-column-RMS 0.10 screen. Layer47 output-column error reaches
  0.3196. Group32 signed 10-bit activations pass the same allowances in all
  six layer/row cases on four GPUs; layer47 maxima are 0.0414/0.0306. This is
  only a fixed-router precision proxy, not model task qualification.
- The source-complete 10-bit native A iterator reduces storage from 2 to
  1.375 bytes/value but makes the complete resident chain about 50% slower:
  10.263/10.744 ms versus 6.754/7.036 ms on layer0. Encoding itself costs
  0.099/0.110 ms. It also fails the additional exact proxy-equivalence check;
  no correctness claim or promotion is made. Static register counts rise
  from 153 to 179 for the lattice GU; these are not achieved utilization.
  Reject this packed iterator before further model-quality work.
- FP24 peer packets retain the original per-element 2^-14 source-error guard
  and the original FP32 row fallback. Complete HC at amplitude0.3 is about
  10% faster (8192: 2.488 versus 2.775/2.779 ms) but injection error 0.001953
  exceeds the predeclared 0.0005 stage limit. Low-amplitude scaled-half rows
  remain exact with no speed gain. Reject this codec under the existing gate.
- Four concurrent cold-DMA ranks use real layer0 shard sizes
  156800/156800/230400 bytes per expert and the same 136-slot buffers.
  Synthetic post-decode occupancy has 110 used slots: merging one-slot holes
  cuts 60 copies to18 but increases 59.84 to67.46 MB and 13.278 to14.661 ms;
  a single span per shard gives15.462 ms. All used bytes are exact. Initial
  contiguous occupancy is12.603 ms across modes. Reject coalescing unused
  holes; the current four-rank DMA is predominantly bandwidth limited.

Serving _C/QSA identities remain 5d389732.../cf09ff3f... . Standalone private
libraries are never installed or registered for serving. Subsequent probes
test immutable FP16 peer-weight reads in the existing grouped GEMM family and
split-row QSA using the existing row reader and FP32 merge. Their generated
source identities and complete-stage costs are recorded before any promotion.

### Isolated GDN result and subsequent root-cause screens

The normal 256-thread GDN run with the original uniform tuner gives
**5.831594319 s / 5941.041357 tok/s**, quality32 passes and all five prompt
hashes/counts match control. Runtime source drift is empty during that cohort.
The three worker traces average a 5.812847 s main span: MoE1525.309 ms,
dense955.826, QSA903.108, HC693.070, AllGather559.656, GDN303.382,
AllReduce94.869; submitted gaps581.064 and host gaps55.795 ms. Profile tuning
does not demonstrate a positive client benefit. Its runtime option and tests
are archived with source hashes under rejected-moe-profile-tuning; the three
owned files are restored to HEAD. GDN direct-state tuning and prior HC changes
remain. Architecture/GDN policy tests pass31 cases after regeneration.

Additional complete-stage decisions:

| Screen | Result | Decision |
| --- | --- | --- |
| QSA split row, existing FP32 merge | At6108 rows, 2/4/8 splits9.631/9.927/10.503 ms versus repeated9.453. All head errors<=0.000618; source one-split control is exact. | Reject partitioning/merge overhead. |
| Immutable IPC FP16 HC weights | Corrected existing indexed-A/blocked-B path gives3.871/4.173 ms for the best tile versus2.031/2.168 repeated controls; block-row error<=0.000072, but raw injection error0.001953 also fails0.0005. | Reject remote repeated GEMM reads. The first experimental indexed packed-B attempt is invalid and its illegal-access log is retained; it never runs in serving. |
| Preexpanded half lattice codebook | Exact on all four GPUs/layers0/17. Full chains6.86–7.23 ms versus6.73–7.10 production controls. | No material gain; reject. |
| Resident FP16 down | Existing K16/K32 tiles preserve weighted outputs exactly, but best3.164/3.289 ms versus2.939/3.045 at7344/8192, before extra decode/pack cost. Bank419.43 MB versus canonical117.96 MB. K64 is outside K160's valid tiled range and is skipped. | Reject. |
| Compact M64 full tiles plus M32 tails | Exact under uniform and skewed routing. Uniform7344 about ties;8192 regresses about4%. Skewed cases regress3–4%. Metadata preparation is included. | Reject this scheduling implementation. |
| Two-slot QSA metadata/RKP264 | Exact, 31,936 shared bytes versus35,008. Two-CTA core9.509 versus9.451; three-CTA core12.991, with80-byte stack. | Higher static residency alone loses; reject. |
| QSA direct V staging | Exact, 80 registers/8-byte stack, 31,936 shared bytes;9.022 versus9.451 repeated control. Direct K+V is9.363 with three CTAs,11.778 with two. | Keep V-only as an unpromoted4.5% operator candidate. |
| One raw HC weight packet | Outputs are exact, but complete2.051/2.177 ms versus repeated2.015/2.166. Packet unpack is included. | Reject consolidation; an isolated fewer-call count is not a stage gain. |
| Fixed GU epilogue / wider M64N128 | Exact but fixed64x64 is7.074/7.404 and wide7.398/7.714 versus6.751/7.018 production controls on layer0. Local descriptors produce368/384-byte stacks. | Reject this entry implementation; test compiler inlining separately before revisiting its cause. |

QSA selections are already ordered among valid indices: the6108x2051 saved
case has zero descending adjacent valid pairs. The lower ascending-all-row
fraction comes from negative tail padding. Sorting these inputs cannot test
the hypothesized locality improvement; do not spend another GPU screen on it.

NCCL correction: the explicit untimed TUNING log for the ordinary complete HC
stage selects RING/SIMPLE with8 channels for1.802240/1.638400/9.400320 MB
AllGather sends; four-byte publication/reuse fences use RING/LL with1 channel.
The generic CUDA entry's LL suffix was incorrectly treated as protocol evidence.
NCCL2.27.5 [device dispatch](https://github.com/NVIDIA/nccl/blob/v2.27.5-1/src/device/common.h)
can call a different work function from the entry's specialization;
[enqueue diagnostics](https://github.com/NVIDIA/nccl/blob/v2.27.5-1/src/enqueue.cc)
log the actual algorithm/protocol. Forced LL complete HC is3.509 ms at8192;
LL128 is2.283 versus fresh automatic2.171. All four rank hashes match across
these protocol controls. Retain automatic selection, not a forced protocol.

A bounded-input QSA numerical probe uses the same0.003 paired/independent
head-L2 gate, FP32 softmax and FP32 accumulation between64-key tiles. FP16
QK quarters are64 dimensions; PV partials return to FP32 after each64-key tile.
The standalone range is finite |q|<=64 and finite value scale<=1.0. This is
not general runtime qualification. The first FP16-to-FP32 fragment conversion
selects the wrong lane's register and fails; the corrected two-shuffle transpose
matches SM70's distinct accumulator layouts. At6108 rows, QK-only8.844 ms,
PV-only9.343, both9.002 versus repeated9.462. QK-only paired head-L2<=0.000655
and sampled FP32<=0.000466 on rank0; all four rank gates pass. No model task
quality, larger support range or serving enablement is claimed yet.

### Adaptive INT16 transport: retained raw failure and existing functional gate

The original INT16 screen retains its failures: at amplitude 0.3, complete HC
7344/8192-row times are 2.048/2.193 ms versus repeated 2.574/2.774 ms controls;
hidden/block row L2 is below 0.000224, but raw injection absolute difference
0.001953125 exceeds that screen's 0.0005 raw-logit limit. No runtime promotion
is made from that result.

Before the next run, use the already established HC contract from
`hc_sequence_rows.reference_errors`: finite outputs; maximum hidden/block row
L2 and logit aggregate L2 <= 0.001; maximum logit error divided by its reference
column RMS <= four FP16 epsilons (0.00390625); maximum absolute error of the
consumed `2*sigmoid(logit/4)` factor <= 0.0005. Apply it both to control/candidate
versus independent FP32 arithmetic with explicit original FP16 boundaries and
to the pair. These are existing operation-aware bounds, not a relaxed raw
limit. Preserve the raw failure in the new report. Source transport guard
remains 2**-14 per element. A passed operator screen still requires paired
model/task, recurrence and boundary evidence before enablement.

Private fixed-epilogue inlining, including a second header overlay forcing
fragment and matrix-combination operators inline, does not change the
368/384-byte stacks or register counts. Stop that compiler screen; no normal
header or serving artifact is changed.

### INT16 full-model rejection and conversion-cost ceiling

The public INT16 candidate passes independent FP32/factor checks, cancellation,
subnormals, non-finite/fallback storage, mixed-mode graph replay, 48-stage drift
(maximum factor error 0.0003113), and model-adapter query boundaries
4095/4096/4097/7345/8192/8193. Thirty-five CPU policy/ownership tests plus three
GPU codec cases and thirteen architecture tests pass after correcting a test
assumption: 65504+65504 is exactly representable in scaled FP16 and should not
be forced through FP32. No numerical limit is changed.

Matched five prompt hashes/counts on the real model give 5.828532 s / 5941.63
actual prompt tok/s versus 5.831594 s / 5941.04 for the original peer transport
and uniform tuner. The 32K retrieval/arithmetic/EOS check passes. The 3 ms TTFT
difference is not material evidence of a benefit. Reject the runtime feature;
remove its typed field, hashes, adapter and tests from active source. The peer
source is restored exactly to fbbbf803cf73161a87e1e562432743bb35d6af9814d389da8b41490514cc10f8.
The source-complete rejected runtime and tests are archived in
`rejected-hc-int16-runtime`; benchmark-only codec/state probes remain available.
No long-context quality expansion or default promotion is justified.

A fresh three-GPU-worker trace has a 5.805 s mean main-stream span: MoE1.529,
dense0.956, QSA0.900, HC0.694, AllGather0.563, GDN0.304, AllReduce0.090 s.
The fourth worker trace again lacks CUDA events, so this is not a complete
four-rank resource-utilization report.

A private, exact GU scratch-lifetime screen retires scheduler scratch before
the mainloop and overlays the unused U operand. Shared bytes fall from24608
to24576, enough for four128-thread CTAs in96KiB. A four-CTA register cap spills:
complete7344/8192-row chains take7.221/7.517 ms versus repeated6.827/7.096;
a five-CTA cap is23.379/24.461. Uncapped scratch reuse gives no gain. Reject;
normal headers/native remain unchanged. Existing custom-AR zero-sum fences
in place of the two NCCL fences save only about1–6 microseconds per complete
HC stage, with existing functional gates passed. Reject that replacement too.

A four-rank upper-bound screen repeats one real layer across eight logical
layers so that a primed canonical bank remains exact. Cold DMA,80% residency,
allocation, stream fences and restart checks remain. With20M-cycle inter-layer
gaps, resident-conversion omission is173.548 ms and all-conversion omission
171.508 versus176.737/177.640 controls. Without gaps they are98.652/93.817
versus101.816/101.743. Outputs/restarts are exact on all four ranks. This bounds
conversion removal in that isolated workload; it is not a valid optimization
for different model layers. Prioritize compute-side lattice decoding before
engineering a direct-original reader whose conversion-only ceiling is modest.

### Lossless lattice-byte compute screen and actual-route capture

Benchmark-only byte pre-expansion moves the IQ3 grid/sign decode out of the
GEMM and reuses the SM70 FP32 grouped family with centered byte operands and
the original FP16 group scales. Real rank-zero layers0/17, four concurrent
GPUs, produce exact hidden/weighted outputs on both7344/8192-row cases.
The64x64x64 full-chain times are6.362/6.701 and6.396/6.703 ms, versus repeated
6.790/7.049 and6.815/7.106 controls. The32x128x32 byte tile is slower.
This excludes expansion/packing and adds a larger bank; it is an upper-bound
compute candidate, not serving evidence. Type22 is outside this probe.

The fresh trace's down kernels account for approximately0.741 s (LUT4+U2),
versus0.639 s for padded lattice gate/up. Capture real routes with the existing
`--enable-return-routed-experts` serving API at unchanged TP4, chunk8192,
256K context, IQ3_S, MTP4 and residency settings. Capture responses are
diagnostic and their timings are not performance results. The source-complete
`capture_moe_routes.py` validates complete prompt routing and the existing
32K task. `moe_real_down_tiles.py` will compare seven existing FP32 down tile
shapes against the normal uniform-tuned path on those exact route IDs;
activation and mixture weights remain synthetic and are labeled as such.

The existing routed-expert-return API explicitly falls back to the legacy
worker (V2 rejects this diagnostic). Its first capture startup fails before
requests: QSA's compressed side-cache stores204 rows per816-token physical
page, but legacy reshaping uses six32-row virtual pages, dropping12 rows.
The observed9243648 FP16 elements cannot reshape to[2124,32,1,128]. The
V2 cache helper already respects `uses_physical_block_table`; apply the same
backend contract in legacy initialization, retaining the paged-MQA indexer
path. Fourteen CPU checks pass across32/64/128/204/208/288 stored rows,
padded/unpadded physical pages, both worker views, neighbour writes, and the
existing virtual-page indexer regression. This is an initialization fix, not
a performance result or a bypass of V2's diagnostic qualification guard.
Preserve the failed startup log; retry capture from the normal public API.

### Real routes, grouped compute screens and selective preparation

The normal public routed-expert-return API now starts successfully with the
physical QSA page fix. Two captured 32K speed prompts match the original hashes
and token counts; the existing 32K retrieval/arithmetic/EOS check passes.
Route capture is diagnostic (legacy worker), excluded from speed comparisons.
For speed-0 the actual chunk rows are7344,7344,7344,6528,6108. Across48 layers
and5 chunks, mean active experts are218.1; using the initial resident profile,
only40.43 of110 cold experts are used (max78). Speed-4 is219.94/40.675,
quality32 is384.84/61.38. These are initial-placement estimates, not a claim
about later decode-mutated cache placement.

Four-rank exact-FP32 down tile and persistent expert-stripe screens retain
real captured IDs, GU, routing, both DN sub-batches and weighted unroute.
Activations/mixture weights are synthetic and DMA is excluded. Seven ordinary
tiles are mostly slower/tied, with only layer17 last-chunk4.6% improvement.
Six persistent M16/M32 stripe schedules are also mostly slower/tied; layer17
last-chunk improves7.3%, without a consistent cross-layer benefit. All existing
finite/row-L2/column-RMS gates pass. Do not promote either schedule.

Copy-stream priority and resident-transcode CTA-cap screens preserve original
math/placement and restart semantics. Eight logical layers on four ranks show
at most1.6% improvement with20M-cycle gaps; negligible without gaps. Do not
promote the priority/cap changes.

Benchmark-only active preparation uses actual current routes to filter both
resident/cold conversion lists and cold DMA. It retains ordinary canonical
banks, 80% raw residency and original FP32 GEMMs. CPU route-offset download,
synchronization and on-demand preparation are included in timing. Eight layers
use captured layer0/1/17/47 IDs twice and derived matching residency profiles.
Cold H2D per pass/rank falls482380800->149004800 bytes. Without gaps the repeated
controls101.287/101.861 ms fall to67.977 ms; all outputs, interruption/restart
and post-decode cache changes are exact on all ranks. However20M-cycle gaps
regress168.454/168.389->171.844 ms: losing advance prefetch and CPU synchronization
outweigh saved work. A busy-GEMM inter-layer screen (3x7344x4096x4096 FP16
matmuls/layer) gives137.943/138.099->133.222 ms (~3.5%). This is not serving
evidence and does not justify default promotion. An asynchronous implementation
would need the existing native cache snapshot/lifetime protocol, not Python
slot-map assumptions or private provider injection.

A further actual-route compute ceiling compares the already-existing FP16
family against quantized GU/DN before engineering selective hot-expert banks.
It excludes conversion, packing, larger-bank and DMA cost, explicitly retaining
the prior finite/row-L2<=.003/column-RMS<=.03 admission limits.

The real-route FP16-family screen completes on layers0/17/47 and first/last
chunks. Most full-chain cases regress or tie. The best GU tile gives modest
last-chunk wins (layer47 4.284 vs4.488 ms), but layer0 last-chunk GU fails the
unchanged output-column-RMS limit (.0374196 > .03) for all four FP16 tiles.
The down-only best tile improves selected last chunks by4–6%, mostly ties on
full chunks; this excludes its419MB bank and all decode/pack costs. Reject
full FP16 banks; the route skew does not justify this implementation.

An asynchronous-active benchmark uses a background host lane with pinned
route offsets, system-scope GPU publish/acquire and ordinary lifetime events.
The first attempt hangs during its first active correctness pass and is
stopped with PID/start-time-scoped cleanup of this task's processes only.
Source/logs are retained in rejected-active-async-first. It is not serving
evidence, and no runtime setting/native provider has been added.

The asynchronous active-prep screen is now exact on four ranks, interrupted
passes and post-decode changes. First-use Triton module loading must occur
before polling kernels (including pointer-alignment specializations). In
addition, the host preparation queue must be drained after all layers are
submitted and before pageable blocking output copies; otherwise the CUDA copy
submission blocks while the model stream waits for that submission. Failed
stacks/sources are retained. The completed diagnostic gives65.838 ms versus
101.595/101.515 without gaps, but171.789 versus168.379/168.260 with20M-cycle
gaps. Removing the per-layer main-thread wait does not recover advance
preparation; do not promote this implementation.

The complete HC up-weight-gather overlap screen is exact on all four ranks.
At7344 rows the candidate2.016 ms versus warm repeated control2.024 is noise;
8192 gives2.167 versus2.171/2.172. NCCL/compute overlap alone does not establish
a material stage benefit; reject before model testing.

Captured speed-route adjacent chunks reuse88--92% of active experts; the
mean newly active count is17--26/layer. A fixed previous-chunk prediction screen
therefore retains original one-layer prefetch for predicted experts and adds
exact GU/DN completion for every current-route miss. The predictor remains
fixed during repeated timing, so warm repetition cannot erase miss cost.
Original raw placement, canonical/staging capacity, arithmetic and snapshot
maps remain. Route download, patch launch and every miss DMA are included.

### Down hybrid compute ceiling and actual cache overlap

A compact GPU task table separates full M128 down tiles from M16/M32 tails
and schedules each with the existing FP32 GEMM/epilogue family. Metadata and
queue overhead are included. Captured first/last routes on layers0/17/47 give
exact four-rank full-chain outputs. Tail32 times are5.457/4.494,5.721/5.069,
5.095/4.150 ms versus controls5.687/4.849,6.209/5.610,5.319/4.509.
These are4--10% compute gains, excluding DMA. Extra8/16-warp partitions of the
original tile all regress (e.g. layer17 first best7.167 vs6.232); reject those.

Putting tail32 into the complete eight-layer cache pipeline retains full
cold DMA and confirms exact outputs/restarts/post-decode across IQ4 and U2.
Without gaps100.789 vs101.571/101.443 ms is negligible;20M-cycle gaps166.450
vs168.612/168.364 save only~2 ms/8 layers. Keep as a cumulative compute
candidate; do not infer the4--10% isolated gain on client TTFT or add a new
production/native policy yet.

Fixed previous-chunk prediction with exact miss completion gives128.420 ms
versus136.848/136.363 with three busy FP16 GEMMs between layers (~6%).
The sleep-gap result is164.210 versus167.173/167.250; all cache/state checks
remain exact. This requires full-model evidence before any source/default
promotion. No csrc production change, _C replacement or new runtime option has
been made in this continuation.

A QSA128-key/256-thread screen reuses the normal decoder and selection table,
two metadata slots, direct V staging, and one shared score buffer reduced
across two K halves. Shared bytes46400 permit two CTAs; the first variant
spills120 bytes at128 registers. A direct-K variant removes the register
prefetch array to distinguish load latency from that spill cost. It retains
the existing finite/paired+independent-FP32 head L2<=.003 screen. The generated
namespace and first missing-source selection errors are fixed before any GPU
run. Private benchmark libraries remain research-only and are never installed.

### Source-complete predictive preparation and full-model A/B

GPU-only mapped completion is materially better than per-layer CPU miss
completion in the fixed previous-chunk workload. Eight logical layers with
20M-cycle gaps give158.298 ms versus167.281/167.271; no-gap61.681 versus
101.433/101.177. DMA bytes149004800 exclude mapped miss reads. All outputs,
interrupted restarts and post-decode checks are exact on four ranks.
The128-key QSA screen fails performance:16.121 ms with register prefetch,
11.457 with direct K, versus9.464/9.452 ordinary controls. Its numeric checks
pass (head L2<=.000494), but it is rejected. No production QSA change is made.

Add a normal typed expert_cache_prefill_prepare=all|predict option, defaultall.
The existing TurboMind owner keeps its forwarding/GEMM stages and owns one
PredictivePrefillPreparation. CPU hints are double-buffered, adopted only after
their CUDA event completes, and only performance hints. Every routed miss is
converted from original GPU slots/mapped host storage. The native lane's new
stage_cold_selected uses the same mutex/residency snapshot/slot offsets as full
DMA. Close drains mapped reads before host unregister. Eager SM70 batches
4096--8192 and<=512 experts qualify; graph capture and other batches fall back.
No_C/QSA replacement or private serving library is required. Existing helper
builds normally from packaged source in the task's standard Torch cache.

Two config/hash/serialization checks pass. Ten GPU contracts pass in32.57 s,
including full/sparse/empty selected DMA, holes, post-decode snapshots,
invalid masks,4095/4096/4097/8191/8192/8193 rows, wrong hints, interruption,
post-decode and pending-transfer teardown. Four concurrent ranks then use the
ordinary TurboMind class (no benchmark provider/DSO) with actual512-expert
IQ3/IQ4/U2 data: all three real route chunks, empty hints, restart and post-decode
outputs are exact. Architecture generation and Ruff checks pass.

Native_C remains5d389732... and QSA remainscf09ff3f... . The new CPU lane helper
links only standard Torch/CUDA/C++ libraries and has no private RPATH/RUNPATH.
Runtime source is frozen for fresh normal full-model control/candidate starts,
at unchanged TP4/1312MHz/chunk8192/256K/IQ3_S/MTP4/80%-20% placement/hostQSA/
diskPLE/thinking/tool/prefix and official sampling. predict-ab-run.py writes
source hashes/drift and compares fresh client medians against each other,
separately from the older6.474 s baseline. Both runs include the existing32K
quality gate and an untimed route trace. No PR/default promotion is claimed.

### Residency cost screen and HC packet precision rejection

The fresh predictive-preparation control is5.854889573 s/5910.615 tok/s,
with matched prompt hashes/counts and the32K quality gate passing. Its candidate
is still starting; no end-to-end predictive benefit is established yet.

Captured routing shows40.4--40.7 active cold experts/layer on average and
15.36--15.74% of routed assignments using cold experts. Experts with<=4 rows
are30.6--31.5% of active cold experts but only0.22--0.23% of cold assignments.
Do not assume that mapped cold weights are all rare. A prefill-only LRU
simulation under the original410-slot/402-initial-resident budget reduces
mean cold experts from47.4 to8.46/6.10/4.21/3.31 across five chunks. This
ignores decode changes and admission/movement costs and is not a speed claim.
Prepare an optimistic component screen using existing resolve, plan copying
and commit, with precomputed current-route lists. Include all GPU protocol,
raw moves and a committed CPU-snapshot boundary. This determines whether
engineering a normal current-route planner can pay; it never qualifies TTFT.

HC temporary4/6-bit weight packets fail a CPU numeric proxy on actual
layer0/23/47 attention/FFN weights and synthetic normalized activations.
Predeclared block-row L2<=.03, injection-factor absolute<=.01 and sigmoid-gate
absolute<=.01 are unchanged. Four-bit block errors are.20--.224; six-bit
errors.044--.0635, with substantial sigmoid errors. Reject before GPU or
full-model testing; this proxy does not establish model quality either way.

The normal predictive A/B completes without source drift. Fresh control
5.854889573 s versus predictive5.840365224 s gives only14.52 ms (~0.25%),
not material evidence for standalone promotion. Both32K quality gates pass;
all five speed-prompt outputs match. Logs confirm the48-layer normal route and
adoption of completed hints. New component admission work may retain it as a
research dependency, but do not describe prediction alone as a useful gain.
The old helper's printed6.474 s comparison is historical, not this experiment's
control. GPU4--7 had a separate task start during candidate startup, so retain
that possible shared-host interference limitation. Only GPUs0--3 are owned.

### Original-byte residency and DMA transport root cause

Current-route admission through ordinary resolve includes full LRU/raw moves
and CPU-snapshot commit. Five real chunks/eight logical layers with20M-cycle
gaps give controls787.85/791.14/798.25 ms, first admission1136.20, and later
745.62 ms. A side stream reduces first admission to1026.52, later742.43.
Requests are precomputed from the current captured routes, so these are
optimistic component screens. The first-pass mapped-copy penalty blocks
promotion even though outputs/restarts are exact on all four ranks.

An independent transport-only screen uses real layer-zero shard byte sizes,
original mapped-word swap versus bounded DMA, including aliased host-slot
write-back and exact bytes on every rank. Eight experts/4.352 MB each direction
take6.162/6.297 ms mapped versus1.983 DMA. Forty experts/21.760 MB take
52.097/52.543 versus9.319 ms with only230400 temporary GPU bytes. Planner and
commit are excluded here. The initial unsupported Torch runtime enum attempt
is retained as rejected-expert-swap-api; the successful test uses standard
installed CUDA runtime functions only.

The existing native lane now shares its original LRU planner between decode
and a caller-owned eager DMA method. The decode polling lane still issues no
CUDA work. Pinned per-layer plans survive their stream; the existing GPU remap
commits after copies. CPU admission first retires published recency entries.
Eight targeted checks pass in40.79 s: original bytes in GPU/host slots, full
host swaps, invalid-storage rejection before mutation, decode graph replay,
prefill/decode interleave and the one-hardware-queue deadlock regression.

After warming the larger whole-entry remap, DMA admission first sequence is
788.636 versus789.296 control median; later742.980 and no-further-admission
control after warming745.333. The older889.209 first result included new remap
JIT and is explicitly retained under dma-unwarmed-remap. This removes the
first mapped-copy penalty; it does not demonstrate a first-client speed gain.

Add opt-in predict_resident through the same typed preparation owner. It uses
completed CPU hints rather than benchmark route lists, retains exact miss
completion and the80% slot budget, and fences decode graph replay through an
external event. It pays the layer-zero synchronization cost. Four CPU hash/
serialization checks and three GPU threshold/restart/post-decode/teardown plus
preexisting-graph-after-interruption checks pass. Ordinary four-rank real-codec
qualification and a full model measurement remain required.

### Dense projection and independent placement screens

An equal normalized blend of the existing9683 profile and independently
captured quality32 routing reduces held-out speed-route cold assignments only
15.74%->13.80% and15.36%->13.59%. Keep the original profile for runtime A/B;
this CPU placement prediction does not justify a service restart or a new
default. Neither speed route participates in calibration.

Existing forced DQ/BLAS bands on six remaining real layouts are rechecked on
four1312MHz GPUs with complete segment restoration. At7344/8192, output
projections improve44--50%, Q4 merged GDN about25--30%; numeric limits remain
.003 paired/independent row L2. At6108, several input projections regress
6--13%. Padding to64 rows makes them worse; reject padding before runtime.

The installed standard cuBLAS FP32 NN algorithms identify the6108 slowdown:
algorithm103 gives GDN4 1.613 ms vs2.361 default and QSA31 1.434 vs2.071,
including DQ, exact paired outputs. It regresses/ties at6528/7344. A targeted
66-case four-rank boundary screen preserves FP32 accumulation and no reduced
precision reductions, with every .003 check passing. Gains around6108/6143/
6144/6145 are30--34% for N1536/3072/4096 at K2560. Do not install a blanket
algorithm override; qualify a bounded window and restored serving layouts.
The normal native_C/QSA artifacts remain unchanged at this point.

The ordinary predict_resident component passes actual512-expert IQ3/IQ4/U2
routes on four concurrent ranks, including empty hints, interrupted restarts
and post-decode. Its warm final6108-row chunk/eight logical layers with20M gaps
is144.466 ms versus160.651/160.628 controls (~10.1%). Unlike the optimistic
admission screen this includes the real hint owner, CPU LRU, DMA, GPU commit
and layer-zero synchronization. It remains a synthetic-activation component
measurement and is not a client TTFT claim.

Fresh all/predict_resident normal service A/B starts with reused owned control
cache and a separate candidate cache. Runtime files are frozen throughout;
the installed native_C stays5d389732... and QSAcf09ff3f... . The next dense
algorithm window exists in source but is inactive in this retained native_C;
neither dense calibration nor native installation changes during this LRU A/B.
Both arms use this same installed artifact. A fresh ordinary CMake_C build,
from the owned source and copied declared dependency sources, proceeds on
separate CPU cores. It is not installed until this A/B ends and affected GPU
qualification can run. No private probe or historical CMake _C is deployed.

The fresh residency A/B control is5.821445322 s/5944.57 tok/s; its32K quality
gate passes. Application-clock configuration is still1312MHz/877MHz and was
not changed. New NVML sampling reveals actual SM clocks can reach1530MHz;
measured control request samples are predominantly1530 (a few GPU1 samples
1507/1522). Earlier1312MHz labels identify the application setting, not proof
of a locked execution clock. Compare the candidate's sampled clocks before
attributing gains. `measure_prefill.py` now records wall-clock request windows.
The owned monitor exits with the A/B owner PID/starttime and reads only0--3.
NVML request activity is about97--99%, memory activity38--41%, power245--265W,
and PCIe RX roughly4.3--5.7GiB/s/rank. Activity is not achieved Tensor Core,
HBM or NVLink saturation. Hardware profiling counters require admin privileges
(RmProfilingAdminOnly=1); NVLink payload counters return N/A. No clock or
driver setting is changed to obtain counters.

The existing interrupted-graph contract exercises layer-zero decode. Before
promotion, extend it to all cache layers and explicitly retire speculative raw
readers before decode replay, as well as admission copies. A separate external
completion event joined on an owner stream can establish this without adding
the speculative-read join to the admission stream's critical path. Keep runtime
files frozen until the current complete-pass A/B finishes; this pending edge
does not widen the measured complete-pass support claim.

The complete-pass residency A/B finishes with no runtime source drift:
5.821445322 s/5944.57 tok/s control versus5.344897598 s/6474.59 candidate,
a0.476548 s/8.19% reduction. Both32K quality gates pass. Candidate measured
requests are5.428548/5.344898/5.303017, reflecting warming residency; do not
claim the same latency for arbitrary cold request history. Both request sets
retain the same hashes/token counts and configuration. NVML samples in the
measured windows have1530MHz on93/96 control and80/84 candidate GPU samples,
with remaining clocks1492--1522. Mean PCIe RX drops4832->3140 MiB/s/rank;
activity98.76->96.30%, power256.1->267.1W. These are sampled whole-request
metrics, not per-kernel saturation counters. The8K objective remains unmet.

After the frozen A/B, add a separate owned decode-fence stream/external event
that joins speculative raw readers as well as admission copies. It does not
serialize the admission stream behind speculative conversion. Extend the
preexisting decode graph to all cache layers and delay a future cold-stream
reader; synchronizing graph replay must observe that reader completed.
All three GPU tests pass in4.71 s after this change. The pre-fence A/B above
is retained explicitly; final implementation speed still needs measurement.

The exact M128 gate/up full-tile plus M32-tail screen regresses on four real
layer0/17 first/last cases:5.807/4.991/6.346/5.732 ms versus controls
5.681/4.846/6.222/5.616. Checks pass, but larger tiles use242--255 registers;
reject before production. The first run then encounters unsupported layer47
decoder (CUDA invalid-value), retained in its log; the benchmark now guards
its declared lattice decoder subset. No widened-decoder rerun is warranted.

The reader-fence ordinary component remains144.510 ms versus160.681/160.550
with four-rank exact output/state checks. Its preexisting all-layer decode
graph also passes with CUDA_DEVICE_MAX_CONNECTIONS=1 (3.00 s). The lattice
eight-replica shared codebook is a separate rejected compute screen: even
against its matching recompiled64x64 control it regresses2.4--5.9% on four
captured layer0/17 cases, with all numerical checks passing.

The fresh normal CMake build completes89 targets from declared owned source
and copied dependency sources. Standard component installation stages native_C
SHA518957bfaefcb248d01c28d50ecf635c94dff66f3fe697f436ec0d8c28f8ae56;
RPATH/RUNPATH are empty and NEEDED libraries are standard Torch/CUDA/system
only. Atomic normal-package installation retains original5d389732... for
rollback. Fresh ordinary import,18 affine numeric/graph checks and three
resident contracts pass. No private standalone probe library is installed.

The new native window is then measured with complete segment restoration on
all four ranks: six remaining layouts at4096/4097/6108/6528/7344/8192 pass
all unchanged .003 checks. Gains are21--50% (the7344 Q5-gate outlier63% is
retained in raw data). Six4096-row calibration entries are added to the
existing family table. Eleven framework dispatch/workspace/compiled/graph
checks pass in7.60 s, including4095/4096/4097 and6108. A default-band run,
without forced admission, confirms all six physical serving layouts use
[4096,-1] and pass on every rank. The full model result remains pending.

The first residency trace occurs after the quality request and gives5.814 s
GPU main-stream span/5.920 s profiled client TTFT. It is not a decomposition
of the earlier5.345 s warm median. Cache history and profiling both differ;
no causal split between them is established. The next full A/B adds three
unprofiled post-quality requests, then profiles the warmed state. Keep the
task-switch first request separate from warm repeated throughput.

The prior resident profile is decomposed across all streams on the three
GPU-observed ranks (TP0 remains CPU-only in this trace). Of829--838 ms main
stream gaps,607--611 ms contains side-stream activity; actual all-stream idle
is222--227 ms. The long `_sub_bounds` waits follow expert transcoding and DMA;
the chunk-boundary `_prepare_row` waits include admission copies. This is a
5.814 s profiled request after a quality task switch, not the5.345 s warm
unprofiled median. Keep this distinction when prioritizing preparation overlap.
Raw: resident-candidate8192-all-streams.json and worker-gaps.json.

While the new native-C dense/resident A/B is frozen, two private research-only
probes are compiled on separate CPU cores without editing runtime headers:
CTA-local shared decoded-B fragments (same lattice decoder/FP32 MMA), and
fused down-GEMM FP32 route reduction (round individual contributions to FP16
as control, include route metadata, output zero and final cast). The second
also tests whether removing the~200 MiB down temporary can make deeper weight
prefetching feasible. Fixed operation limits remain finite, row L2<=.003,
column RMS<=.03; no model/default change is authorized by these microprobes.
They are queued after the complete-model cohort under the same0--3 locks.

The dense/all control on the new ordinary native-C measures5.824908 s /
5942.59 tok/s, with32K retrieval/arithmetic/EOS gate passing. Three requests
after the quality task give6.034595 s median, so cache-history cost is real
and must stay separate from the earlier warm-speed block. The fresh candidate
is still starting; no combined dense/resident full-model gain is established.
Control trace is6.017--6.018 s GPU span /6.106 s profiled client TTFT.

Concrete native arguments confirm major unqualified Q8 hot shapes:
[M,1536] -> [M,2560] and[M,160] -> [M,2560], bits8/group32, each240
calls/request. The new six Q4/Q5 layouts hit BLAS in this same trace but cover
only part of the dense work. A standard-cuBLAS TN screen is queued with these
two real Q8 weights plus existing merged GDN/QSA, includes DQ and an explicit
transpose (a favorable result can justify fusing that layout into DQ). FP32
accumulation and the fixed .003 independent/paired/fused row-L2 gate remain.
No runtime-header change is made during this frozen A/B.

The final reader-fenced/new-native dense residency A/B finishes without runtime
source drift:5.824908116 s/5942.59 tok/s control versus5.274614855 s/6565.59
candidate (0.550293 s/9.45%). Both32K quality checks pass. Measured candidate
requests are5.375135/5.270817/5.274615. Both arms' actual SM samples remain
predominantly1530MHz (control98/104, candidate88/96), without setting changes.
Candidate's three unprofiled requests after the quality task are5.860097,
5.882230,5.504519; profiled client5.531756/GPU critical observed rank5.422747 s.
The three post-quality requests have not fully restored the5.275 s warm state.
Its trace breakdown is MoE~1.50s, QSA~.963, Dense~.955, HC~.65, AG~.365,
GDN~.289, AR~.05--.077, main submitted gap~.46, host gap~.054. This table
belongs to that same profiled request, not the warm median. TP0 remains
CPU-only; the critical observed rank isTP3. Direct HC kernel attribution fixes
missing parent scopes for _prepare_row/_peer_combine_norm in the new trace.

Five research decisions follow the source-frozen model cohort:

- Shared decoded-B gate/up is exact but~10x slower (local arrays/extra shared
  staging); reject. M64N128 with one M warp is only2--3% better; keep unpromoted.
- FP32 fused down+route reduction passes every unchanged numeric check but
  full real-route chains regress~13--21%; reject, including its memory premise.
- Two canonical slots in ordinary predict_resident pass exact/restart/decode
  checks but144.768 vs144.576/144.578 ms with20M inter-layer gaps; reject.
  No new pipeline-depth or workspace-budget runtime configuration is added.
- Native stream-ordered copy batching passes exact-byte checks on all ranks.
  Forty expert swaps: serial9.2658 ms, group8 individual8.1638, group8 batch
  API7.7469, one whole batch8.4012. Eight experts1.9830->1.7809. Batch8 needs
  4,352,000B temporary instead of230,400. Keep only independent copies within
  each batch; separate H2D saves, D2H write-backs and final GPU admission.
- Standard-cuBLAS TN plus explicit transpose regresses merged GDN/QSA by
  ~10--20% versus NN; reject. Crucially NN itself halves the hot Q8 output
  projection time versus the uncalibrated fused path, without changing layout.

The Q8 full segment-restoration screen on four ranks at4096/4097/6108/6528/
7344/8192 passes independent/paired .003 limits. Large K1536/N2560 outputs
improve~41--50%, but shared K160/N2560 regresses at4096/4097. Shared6143/6144/
6145 boundary screen supports a6144 minimum (rank0's6143 anomaly is retained).
Two existing-table entries are added: bits8/group32/K1536/N2560 min4096 and
K160/N2560 min6144. Two native/framework/workspace/compiled/graph tests pass
in3.10s; default-band four-rank serving-segment checks also pass. Normal_C
stays518957bf...; no private kernel or artifact installation is needed.

Batch transport is folded into the ordinary source-built expert-cache helper,
with one-shard legacy fallback, individual copies on legacy default/captured
streams and compile-time fallback for CUDA<12.8. The owner allocates bounded
batch8 scratch. Twenty-one GPU byte/state/threshold/capture/interruption tests
pass15.11s, plus single-hardware-queue decode and interrupted graph tests pass
15.93s. A CPU-only normal helper build uses declared CUDA12.8 and Torch.
A full-five-chunk warm component is743.775 vs743.297/742.731ms: no gain when
these four real layers' expert unions181/166/367/114 all fit402 slots. Across
all48 layers only layer15's speed0 union427 exceeds402. This does not establish
transient task-switch benefit; do not extrapolate transport speed to client TTFT.

The next frozen full model A/B compares previous-best source with Q8+batch.
The control's three switched runtime files match the previous model manifest
byte-for-byte. Each arm freezes all runtime files, uses normal_C518957bf...,
and runs six unprofiled post-quality requests before profiling. Source swaps
are only between stopped owned services; finalization restores the candidate.
While it runs, do not edit runtime files: the worktree temporarily contains
control source. Active script q8-batch-ab-run.py/session89776. GPU0--3 only.

A new research-only QSA pair prototype partitions common and exclusive selected
tokens exactly. A512-thread CTA runs the existing FP32-MMA reader twice,
retaining online softmax/PV accumulators: shared keys load once for both
queries, then each query processes its compact residual. This avoids the old
union scheme's extra dot products and needs no partial-output merge buffers.
A GPU sorter emits three513-entry packed lists per pair (~19--23MiB), with
original-reader fallback for malformed/mixed-request groups. Standalone compile
uses116 registers, no stack/spills,32,128B static plus37,888B dynamic shared.
Fixed paired/independent head-L2<=.003 and exact all-row selected-multiset
checks precede timing. GPU screen is queued after the full A/B/session40870.
It is not installed in serving. Runtime QSA source/native remain unchanged.

The Q8+batch same-machine cohort completes with no runtime source drift:
previous-best5.280235784s/6569.40 tok/s versus candidate5.159037682s/6707.84
(0.121198102s/2.295%). Both32K retrieval/arithmetic/EOS checks pass.
Six post-quality requests give medians5.484479346s versus5.378152938s;
keep task-switch behavior separate from the primary warm-speed block.
Ordinary native_C remains518957bf..., QSA nativecf09ff3f.... Actual clocks
remain predominantly1530MHz under load with unchanged1312MHz app setting.
Batch transport has no warm component gain on fitting expert unions, so the
combined gain must not be attributed independently to batching. Full latest
profile attribution is being recomputed. Raw:q8-batch-ab.json and each arm's
runtime manifest/hardware samples.

The QSA pair prototype's final producer is GPU binary membership on validated
sorted primary selections, with original-reader fallback for unsafe groups.
All four ranks pass exact selected multisets, finite outputs and fixed .003
paired/independent head-L2 gates (maximum paired .000648). Ideal saved KV
reads29.926%, but11.238ms versus9.515/9.535ms control. One justified followup
trims compacted phase loops:10.6068ms versus9.4547/9.5061 control. Preparation
alone.21284ms; core alone10.3975ms. The core itself regresses, so reject the
algorithm rather than optimize its producer. No private QSA library is installed.
Raw:qsa-pair-partition[-trimmed].json; all serving QSA source/native unchanged.

A separate whole-chunk down+route fusion followup eliminates split-padding and
tests both FP32 and FP16x2 atomic reduction under the same .003 row/.03 column
limits. All six real-layer first/last cases pass (half maximum row.000799,
column.02934), yet FP32/half complete chains regress~7--21%:e.g.layer0first
6.709/6.609ms versus5.715;layer17last6.067/6.076 versus5.654. Reject both;
no full-model quality run or native integration is justified. Raw:
moe-real-fused-whole.json. No further atomic dtype/tile grid is warranted.

### Whole-path followups after the6708 tok/s cohort

The latest candidate Torch traces contain no GPU activities on any rank; both
critical-path JSON reports are therefore empty, not valid new breakdowns.
The service logs record CUPTI_ERROR_NOT_INITIALIZED(15). A fresh four-rank
small profiling smoke records eight kernels per rank. A separate fresh-process
CUDA-graph reproducer leaves exactly3,866,624 bytes free and reproduces the
same initialization error/no GPU events. With~14MiB free it records kernels
and consumes12,582,912 additional bytes. NVML records16,380.3125MiB used at
the failed model attach. This supports device-space exhaustion, not a permission
claim. A new same-source model cohort releases unused allocator cache before
constructing the Torch profiler. That explicit diagnostic-only wrapper edit
is restored in the owned supervisor's finalizer; normal CLI, model, quotas,
chunk8192 and computation policy are retained. No profiled result is substituted
for unprofiled TTFT. Raw:cupti-memory-exact-4.json, cupti-memory-4.json,
q8-batch-critical-path launch/source manifest and service log.

Independent immutable-weight NVLink CE gathering is screened inside complete
HC stages, including norm, projections and final activation gather. Serial
and batch copies regress; four copy streams give1.701/2.031/2.190ms at
6108/7344/8192, versus faster repeated NCCL controls1.684/2.018/2.173.
Outputs are bit-exact on all four ranks. Reject; no runtime/native installation.
Raw:peer-ce-gather.json. This is distinct from the earlier SM-vector IPC reads.

The EP compute ceiling is repeated with captured routes and different real
physical rank shards, retaining original .015 row/.04 column gates. Contiguous
EP4 regresses especially at layer47:8.293/6.883ms versus6.759/5.683 for
first/last chunks. Its route counts34,737/8,610/7,246/22,847 reveal the
imbalance hidden by the earlier uniform-random-route8--10% gain. TP2xEP2
also lacks consistent benefit:layer0first8.047 vs7.104, layer17last6.578
vs6.794, layer47last5.601 vs5.660. Both screens pass finite/numeric checks.
CPU-only fixed-stripe and preexisting9683-profile ownership maps also leave
large held-out imbalance; no new mapping is fitted to the measured prompt.
Reject serving EP expansion on this evidence; loader/cache/decode guards remain.
Raw:moe-real-partition.json and moe-real-hybrid-partition.json.

A current-DN-before-next-GU scheduling hypothesis uses the same single bank,
real codecs/routes, full prediction/LRU/DMA/state commits, and three busy
6108x4096x4096 FP16/FP32-accumulation matmuls between layers. Complete eight-
layer times103.495 versus103.730/104.086ms are not material. All four-rank
exact/interruption/empty-hint/post-decode checks pass. Reject this simple
reordering; do not promote the apparent stream-dependency explanation alone.
Its first harness attempt fails on missing imported _MEMLOG before candidate
measurement; the corrected harness is measured and both logs are retained.
Raw:moe-resident-reordered.json. A separate two-bank test under real compute
pressure remains pending; the previous no-gain evidence used sleeping gaps.

A full-width GU reuse hypothesis is compiled research-only: one CTA holds
all320 output columns, avoiding repeated indexed activation loads across five
64-column tiles. The initial K64/M16/N320 variant spills~340B/thread and is
not sent to the GPU. K32 bounds operand live ranges; M16/N320(two warps) uses
247 registers/no spills, and padded N384(four warps)168/no spills. Both retain
the existing lattice decoder, FP32 MMA and gated epilogue. Numerical and complete-
chain speed checks are pending after the owned model profile. No serving source
or installed native changes are made while the trace cohort is frozen.

### Complete four-rank critical path and rejected column-streaming promotion

The fresh same-source cohort restores GPU events on all four ranks by releasing
unused allocator cache before profiler construction. Its diagnostic wrapper is
restored byte-for-byte afterward and the runtime source-drift record is empty.
Unprofiled median5.167457189s/6701.75 tok/s corroborates the prior5.159s best;
profiled client5.205132136s, critical observed GPU main-stream span5.129182399s.
TP3 exclusive main-stream categories:MoE1492.242ms,QSA893.863,Dense773.492,
HC642.839,AllGather390.333,GDN288.076,AllReduce102.938,other/memory141.865,
submitted gaps367.446 and host gaps36.086. These belong to the same profiled
request. All-stream per-card idle is258--294ms;TP3 is262.207ms. Aligning all
four traces with their identical baseTimeNanoseconds yields only46.867ms when
all four GPUs have no kernel/copy/memset active. Do not describe262ms as
four-card idle or kernel-active time as achieved SM/HBM utilization. Raw:
q8-batch-critical-path-{breakdown,all-streams,cross-rank}.json.

The ordinary two-bank pipeline is also measured under three busy FP16/FP32-
accumulation matmuls per layer, not sleeping gaps. Eight-layer control103.831/
104.280ms versus104.173 candidate: no material gain despite353,894,400 extra
canonical bytes. Exact/restart/empty-hint/post-decode checks pass on all ranks.
Reject further bank-depth work without a new critical-path reason.

A different DN dataflow streams output columns instead of tokens. Two1280-
column passes retain all query routes, avoid subroute sorting and indexed-A
DN reads, and reuse the existing native grouped family/FP32 unroute. No atomics
or extra wide output are used. One-pass down tensors188,006,400/156,364,800B
at7344/6108 rows versus the previous209,715,200B maximum (not a claim about
whole-process reserved-memory peaks). Four different physical TP rank shards
on layers0/17/47, first/last actual chunks, pass bit-exact outputs. Complete
resident chains improve~3--8% with actual-route calibration. A normal source
candidate covers eager4097--8192 query rows forE512/N2560/K160 canonical
LUT4/U2 with a4096-token scratch capacity; other rows/layouts/capture retain
token streaming. One strided-output/canary/changed-input graph check and five
affected cache/state checks pass(6 tests/10.63s). U2 real-shard boundaries
4095/4096/4097/6143/6144/8191/8192/8193 also pass exactly on all four ranks.

Promotion fails the ordinary whole-five-chunk cache pipeline with busy GEMMs:
603.920ms candidate versus603.195/605.633 control. Selection evidence explains
part of the difference:uniform warmup calibration regresses first chunks on
layers0/47(5.970vs5.746 and5.582vs5.318) while real-route calibration had won;
unmeasured default selection is worse still(~6.1--7.0ms). Do not fit a cached
prompt or introduce serving-path measurement to manufacture a gain. Reject
promotion on current evidence. All candidate runtime/tests are archived under
artifact moe-column-runtime/candidate; ordinary runtime is restored exactly to
the previous-best control. No new knob, native artifact or serving PR is added.

An independent earlier exact-DN-miss completion probe starts current DN patches
before GU, then waits after GU, preserving one bank and all reader fences.
Including columns, its whole-five-chunk busy pipeline602.594ms versus602.960/
606.454 is not material; exact/interruption/decode checks pass. Reject. Raw:
moe-early-down.json. The earlier source-neutral ordering probe is separately
recorded and not counted again.

Full-width GU activation reuse is also rejected:spill-free M16/N320/K32 and
M16/N384/K32 regress complete real-route chains~16--26%, despite unchanged
numerical checks. Original K64 wide prototype spilled~340B/thread and never
ran. A final bounded-codebook replica test addresses the concrete Type21
shared-memory occupancy loss of the old8-replica version:1/2/4 replicas use
24,608/26,656/30,752B shared and153/158/158 registers, all permitting three
CTAs by static resource limits. Both2/4 replicas still regress~2.5--3.5%
against the matching rebuilt control on layer17 first/last chunks. Reject;
resource occupancy is theoretical, not an achieved-utilization measurement.

The next layout ceiling preencodes immutable DN weights outside timing and
reads them only for resident experts; cold DN DMA/conversion and every other
ordinary cache/state step remain. It deliberately uses additional canonical
GPU storage and cannot qualify serving or80/20 memory acceptance. Its purpose
is to decide whether an eventual in-place canonical resident layout is worth
the decode/host-writeback engineering. In IQ4_NL DN the canonical weight+scale
byte count equals the original bank's logical byte count; decode and original-
byte contracts still require separate proof. No such production layout change
has been made. Active artifact:moe_direct_down_ceiling.py.

The resident-canonical DN ceiling is now rejected:600.3125ms versus602.3349/
606.0431ms for the whole five-chunk/eight-layer pipeline with busy GEMMs. It
uses432,537,600 extra canonical GPU bytes and cannot qualify80/20 serving.
Exact/restart/empty-hint/post-decode checks pass all four ranks. Removing
resident DN transcoding alone is not a material critical-path win; do not
undertake a production resident-storage/decode/writeback conversion from this
evidence. Raw:moe-direct-down-ceiling.json.

A final Type22 IQ2_S gate/up screen covers a different codebook family and
width8 using actual layer1 routes and four different physical TP shards.
Planar32-bit codebook halves and M32/N128 tiling both regress:at7344 rows,
complete chain5.578/5.729ms versus native5.395 and matching rebuilt5.520;
at6108 rows4.635/4.786 versus4.530/4.606. Fixed.003 row-L2/.03 column-RMS
checks pass. Reject; installed native/source stay unchanged.
Raw:moe-iq2s-gateup.json.

### Cross-stage communication and alternate compute screens (15 October)

The valid TP3 trace attributes76.800/74.442ms to485 down/up immutable-weight
AllGathers, versus194.530ms to485 row-owned2560-wide activation gathers.
Raw:q8-batch-critical-path-ag-shapes.json. This bounds weight-only changes;
do not assign all390ms of AllGather time to fixed weights.

Cross-HC weight prefetch includes eight distinct HC modules, preceding one or
three busy FP16/FP32-accumulation GEMMs, both peer reader fences, projections,
weight/output gathers and the prologue. Same-communicator and separate-
communicator NCCL prefetch both tie/regress. Four-stream copy-engine prefetch
using immutable IPC weight publication saves~0.5--0.9ms per8 modules:at7344,
37.316vs38.197/38.382ms with one busy GEMM,84.838vs85.340/85.687 with three;
at6108,41.448vs42.369/42.367 and102.382vs103.017/103.159. Outputs are exact
on all ranks. Extra published copies make this research-only; no serving
IPC lifecycle/peak-memory admission is claimed. Its estimated client ceiling
is tens of milliseconds, not the8000 tok/s gap. Raw:hc-next-weight-prefetch
{,-dedicated,-ce}.json.

Parallelizing only GDN local QK and decay/A terms removes repeated work across
value tiles, but adds prepared P reads to the serial recurrence. Including
A restoration/preparation/state reset,7344 takes1.220ms versus1.189/1.023
controls;6108 takes1.053 versus.997/.902. Outputs and private state are exact
all ranks. Reject. Raw:gdn-precomputed-screen.json. The independent existing
auto_cp screen returns the non-partitioned fallback (exact outputs, higher
wrapper overhead); it does not measure an executing CP speedup. The warmup
mask kernel does not initialize fallback_mask for ht_mask rows, so no CP
admission/quality claim follows without separately fixing and qualifying
that public path. No runtime CP change has been made.

A new lossless exception-plane peer codec retains the existing half guard and
all ownership fences. FP32 rows use high16+middle8 and load low16 only for
escaped elements; it fits the original6-byte-per-element IPC allocation and
reconstructs the existing FP32 values exactly. Both7344/6108 and amplitudes
.03/.3/3.0 pass complete-stage bit-exact checks on all four GPUs. At amplitudes
.3/3 the extra encoding/read instructions cancel transport savings(e.g.2.588
vs2.573/2.575ms at7344). Reject. This is distinct from the earlier lossy
FP24 rejection; no failed numerical limit is relaxed. Raw:hc-lossless24-screen.

Standard cuBLASLt NN heuristics use FP32 accumulation and COMPUTE_TYPE
reduction, with numerical capability flag66050(input16F/accumulator32F/HMMA).
Four distinct physical rank shards and original per-call DQ are included.
At6108 the returned implementation regresses1.599->2.362ms(GDN4) and
1.423->2.059(QSA31). At7344, algo21/tile24 improves1.889/1.904->1.777
and1.693/1.706->1.635; other choices regress/tie. Every fixed.003 paired/
independent row-L2 check passes. This is a small-size-specific candidate,
not a global library override or end-to-end gain. Raw:dense-lt-screen.json.

The first hot-expert dense-GEMM ceiling excludes online selection and B
construction, includes GPU route/gather/scatter and complete DN. Eight hot
experts cover~20--28K routes. All four completed layer0/17 first/last cases
pass unchanged.003/.03 limits, but regress~.23--.30ms per chain. The initial
noncontiguous active-group scheduler still launches empty grid regions for
skipped hot rows. A compact scheduling-offset version tests this concrete
confounder before rejecting dense hot-expert lowering. Type23 is outside
the private lattice-only probe; its rejected invalid-value attempt is
retained, and no invalid layout is dispatched in serving.

The compact hot/cold scheduler uses independent scheduling prefix offsets
(counts of hot experts zeroed) while matrix/input/output offsets retain their
original positions. This removes the empty hot-row grid regions without
changing the grouped family or source layout. Whole-chain timings still
regress:layer0 first5.995vs5.766/5.765ms,last5.102vs4.882/4.901;
layer17first6.484vs6.218/6.216,last5.893vs5.642/5.653. All four ranks pass
fixed.003/.03 checks. Reject the eight-hot dense lowering even before paying
online selection and B construction. Raw:moe-hot-cold-compact.json.

The per-layer profile contains49 MoE roots per chunk(48 target+1 MTP),245
over5 chunks. Do not use modulo48 to label them. With modulo49, MTP accounts
for83.624ms main-stream span/82.127ms kernel time, versus target layer15
45.039ms and layer044.218ms. Trace arguments show MTP's Q8_0 GU/DN use19
2048-token-or-tail calls and38 old ggml_moe_a8 launches, with real dimensions
GU[512,320,2720] and DN[512,2560,170]. This is the next independent compute
hypothesis: original raw Q8 blocks read directly by FP16 Tensor Core GEMMs,
without a canonical replica or increasing the2048-token DN capacity. The
research screen loads four physical TP slices of the actual BF16 MTP file
through the existing quantize_q8_0 function; captured target-layer routes are
only proxies, not captured MTP routes. No serving route is installed yet.
Raw:q8-batch-critical-path-moe-layers.json and mtp_raw_q8_screen.py.

### Correctness root cause in the remaining MTP prefill baseline

The raw-Triton Q8 prototype is rejected before timing completion:its SM70
SASS has FFMA and extensive local spills, with no HMMA. This does not test
a Tensor Core implementation. The explicit shipped TileLang SM70 gemm_v1
prototype executes, but its26.471/21.418ms versus14.681/12.378 controls is
NOT a valid performance comparison:the old control has non-finite/unwritten
outputs. Candidate independent FP32 row-L2 is~.00052--.00056; the failed
control cannot establish numerical quality or a valid baseline.

The source root cause is the shared legacy MMQ moe_q guard exp_idx>255.
MTP has512 experts, so every routed expert256--511 is skipped. This is
separate from Q8 K-tail speculation. A deterministic normal-operator CUDA
graph reproducer poisons its graph-owned output with NaN before replay.
E256/id255 writes all4160 values correctly;E257/id256 andE512/id511 leave
all4160 values NaN on all four V100s. No allocator poisoning or private
provider is used. Existing MTP tests use only1/5/20 tokens and miss MMQ's
large-batch route. Raw:repro-mtp-expert-capacity.json.

Fix the existing legacy family by passing the actual positive int32 expert
bank capacity from W.size(0) through its internal launchers, replacing the
constant guard with exp_idx>=num_experts. Public op schemas, weight storage,
math, routing and default policy are unchanged. Add normal-op Q8/Q4 MMQ
capacity255/256/511 and graph changed-index coverage. A normal CMake build
of_C_stable_libtorch is in progress; the old installed standard stable DSO
and both modified sources are saved under native-before-expert-capacity.
Do not call the earlier5.159s client run a fully qualified MTP-prefill
computation baseline:target-task output gates do not prove correct draft
expert execution. Corrected full-model timing and paired task/acceptance
evidence are required before further speed promotion.

The normal29-object_C_stable_libtorch build is complete. Standard CMake
installation stages it without runtime overrides; publication replaces the
owned checkout file atomically so existing foreign processes retain their
old mapped inode. Active stable DSO SHA256
`d3dd92653fe967054e9a6a09f3e7b9ede4a1409021d7921cf1fe7cf5a7b2a702`;ordinary_C518957... and QSAcf09ff... stay
unchanged. The capacity regression passes10 tests on each of four GPUs in
~6.25s, including six Q8/Q4 graph cases and prior Q8 byte/vector tests.

The first added Q4 test incorrectly compared an A8 MMQ operator to an
unquantized-activation FP32 product with a universal2% maximum-row bound.
That also fails the unchanged low-expert E256 case(2.02%;other Q4 cases
2.46--2.49%). No numerical bound is relaxed. The capacity test now compares
the same physical quantized weights as a one-expert bank, retaining the
operator's A8 arithmetic and requiring bit-exact results after poisoned-output
graph replay and changed live expert indices. Existing independent Q8 tests
remain unchanged. The failed original reference check is retained under
mtp-capacity-qualification-q4-reference-mismatch.log;it is not claimed as
new Q4 numerical coverage.

Original MTP Q8 bank preparation also reuses pad_weight_tail to supply the
existing GGUF reader's zero allocation tail for K160, preserving every
logical packed byte. It does not pad each row or expand K to512. The per-rank
storage increment is374bytes, and initialization retains no duplicate bank.
A CPU byte/storage-tail check is added;GPU qualification is rerun because
this loading-boundary source changed.

The existing upstream native MMQ family is screened on the same physical
MTP shards with guarded storage and original activation/weighting stages.
It takes151.180/126.261ms at7344/6108, far beyond the old invalid control
and the direct raw implementation. Reject the upstream route before any
production admission change. The prequalification report marks the old
control invalid explicitly and makes no accepted speed comparison.
Raw:mtp-upstream-q8-prequal.json.
