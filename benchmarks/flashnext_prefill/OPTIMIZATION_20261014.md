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

First combined candidate, same five prompt hashes and token counts as control:

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

Pending: candidate model quality, matched unprofiled end-to-end measurement,
and paired near-256K qualification. The original 256K deployment has not been
promoted to a candidate. The isolated candidate server is starting with the
corrected MoE tiles, four dense calibrations, and occupied-slot staging.
