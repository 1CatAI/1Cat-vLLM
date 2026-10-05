# Native CPU reads for disk-mapped PLE rows

The C1 PLE phase trace identifies mmap gathering as the largest CPU lookup
phase: median 1.610 ms across 1196 single-token requests. Key computation is
0.144 ms; result fan-out and four flag publications take 0.089 ms. These are
instrumented CPU intervals, not GPU critical-path wait measurements.

The new CPU operator resolves every requested row, sorts and deduplicates
only its covering pages, advises read-ahead for cold pages, then copies the
original bytes directly into the result buffer. It retains all checkpoint
shards as mmap. It never allocates a whole-table copy or prefaults the whole
mapping. Existing page-release policy, output publication and consumer waits
remain in their existing order. Dense weights, activations and accumulation
precision are unaffected; existing quantized embedding bytes are unchanged.

`MappedRowGatherKernel` lives in the PLE kernel framework and is selected by
Linux/file-mapping/operator capability. Startup verifies actual mapped row
bytes and records warm reference/candidate timings without consuming RNG.
`KernelConfig.ple_disk_row_gather` defaults on; it is CPU I/O policy and does
not change the compiled model hash. No environment variable, architecture,
tensor-parallel, model-name or batch-width restriction is added. The native
operator validates indices and output geometry before writing results.

The normal `_C` extension ships the CPU registration. A missing operator or
failed startup check retains the existing reader. Both whole-table output
gather and cascade disk-tier reads consult the provider. CPU-reader admission
is included in the offload-worker READY message and startup logs.

Research-only real-row probes use 16 retained row sets, nine alternating
repetitions, and byte comparisons. Cold arms evict only selected file pages;
warm arms retain them. The table stays file-backed in both arms.

| CPU lookup | Warm us/case | Cold us/case |
| --- | ---: | ---: |
| Serial ctypes row copies | 28.27 | 1446.62 |
| Native cold-page advice and gather | 53.95 | 282.08 |

The native research measurement includes creating its ID and output tensors;
the production path writes the existing output directly. Cold controls
average 16.69 major faults/case; native prefetch initiates I/O ahead of copies.
All compared row bytes match. These numbers do not establish a model speedup.

Fourteen research C++ correctness cases cover M=1/5/17/33, 17/160/8193-byte
rows, shard/page boundaries, repeated IDs, buffer reuse and rejection of
invalid or strided IDs. This CPU scheduling change preserves gathered bytes
and numerical operations. Promotion requires installed-artifact byte checks
and one matched C1 timing comparison; distribution and quality suites are
reserved for changes to accumulation or numerical boundaries. UVA hot-row caching remains a separate
follow-up: the current 8192-row trace simulation has only 14.5% all-row hits,
so its miss path must be efficient as well.

Before the C1 experiment, the projected critical-path saving is
`1 lookup/token × 873.7515 us pre-copy wait × 0.80 lookup reduction × 0.9`
= **0.629 ms/token**. The 80% reduction is rounded from the cold real-row
measurement, not from warm startup timings. This projection caps credit at
the observed pre-copy GPU wait; CPU service time is not added to GPU time.
The historical trace and the current cache state may differ. Compare the
observed saving against this estimate and investigate deviations over 15%.

The maintained endpoint entry is
`benchmarks/benchmark_sm70_qwen38_quality.py --timing-only`. It takes the GPU
lock without blocking and fails before model loading on busy GPUs, insufficient
disk, missing checkpoint files, Python/header mismatch, disabled request metrics,
or an unguarded spawn entry. Every launch writes a completion summary, including
failures. Normal quality mode retains anomaly reports; a single anomalous output
requires three different seeds in both arms before a candidate-specific failure
can be established. Existing health detection alone is not that comparison.

The normal installed SM70 wheel passes all fourteen CPU dispatch tests without
skips. The same retained real-row cold/warm test gives the following results;
no private operator binding is used in this measurement.

| Installed CPU lookup | Warm us/case | Cold us/case | Cold payload GB/s | Byte error |
| --- | ---: | ---: | ---: | ---: |
| Reference row copies | 29.97 | 1502.12 | 0.00170 | 0 |
| Native selected-page gather | 62.21 | 292.52 | 0.00875 | 0 |

Each case copies exactly 2560 payload bytes (16 × 160). Its covering-page
footprint averages 68352 bytes, ranging from 65536 to 73728; neither is a
measurement of physical storage traffic. Payload GB/s is the copied bytes
divided by CPU elapsed time, not GPU DRAM bandwidth or SSD throughput. Cold
reference major faults average 16.69 per case; prefetch queues I/O before
copying in the candidate. The warm regression is retained explicitly and the
C1 endpoint gate remains pending. The first endpoint attempt correctly failed
before model loading because another task held the GPU lock.

A queued pair may hold one reservation across both processes with an inherited
file descriptor: `--gpu-lock-fd FD`. The entry verifies that it refers to the
required lock and takes a nonblocking exclusive lock on that same open file
description. The caller keeps its original descriptor open until both arms
finish. This avoids a release/reacquire race while preserving the ordinary
entry's fail-fast behavior and GPU-idle check. Normal launches need no option.

The normal installed-artifact C1 pair completed with FP16 dense/KV, FP32 state,
NVFP4 experts, TP4, CUDA graphs, disk ngrams, no MTP, 262144 startup capacity,
8192 input tokens and 513 fixed-length output tokens. Both arms use the same
wheel and six samples; only `ple_disk_row_gather` changes. The candidate's
CPU-worker report selects `MappedRowGatherKernel` with a passing byte check;
the control reports `disabled_by_policy`.

| C1 pure decode | Median ms/token | Range ms/token |
| --- | ---: | ---: |
| Reference reader | 11.014752 | 10.932574–11.068880 |
| Native reader | 10.869117 | 10.773957–10.975267 |

The measured saving is **0.145635 ms/token (1.32%)**, against the pre-recorded
0.629101 ms estimate: a **76.85% relative estimation error**. Sample ranges
overlap; this is one matched installed-artifact comparison, not a guarantee
for every prompt or cache state. Numerical operations are unchanged and all
fourteen native byte tests pass; no distribution or task-quality campaign was
run for this scheduling change under the graded policy.

The estimate incorrectly applies a forced-cold CPU lookup ratio directly to
an older GPU pre-copy wait. Startup's actual mapped-row warm sample already
shows a different ratio: 156.90 us reference versus 77.30 us native for 64
rows. Neither startup ratio nor forced-cold service time establishes the
fraction of a normal decode's GPU wait that can be removed. A short observer
run is pending to quantify per-request page faults and gather/staging time
under the same repeated-input C1 contract before closing calibration.
Future I/O estimates must retain cache-state and critical-path uncertainty;
use 0.146 ms as this workload's measured benefit rather than 0.629 ms.

Use `--ple-phase-probe` in the maintained endpoint entry for diagnostic phases
and faults. Such reports set `instrumented=true` and `speed_acceptance=false`
and cannot be compared as endpoint speed evidence. `--runtime-cache` permits
reuse of the compatible compiled baseline graphs for this CPU-only policy;
`--timing-repeats 2` limits the diagnostic cohort. Completion summaries keep
compact CPU-reader decisions and link to the full route report.

## Three-seed requalification and lookup calibration

The updated admission contract requires matched three-seed task and health
results for this PR, including 128K and 258048-token needles. Previous
byte-preserving scheduling admission and single-seed evidence do not complete
this requested requalification.

A diagnostic control run records 1057 single-token CPU requests: only 45
requests (4.26%) incur mmap major faults. Median mmap gather is 112.223 us,
gather plus staging 188.567 us, key computation 105.974 us, previous-consumer
wait 33.826 us, and complete CPU request 683.796 us. Four flag publications
per request have a per-publication median of 8.483 us. These nested timings
must not be summed. Python observation overhead is present; this instrumented
run is not speed admission and does not establish GPU exposed wait. The
previous 0.629-ms projection applied cold-page reduction to all requests;
cache residency and overlap must be accounted for before further projection.
