# Flash-Next active host KV on SM70

QSA attention history uses pinned host E4M3 bytes with one FP32 scale per
token, local KV head and K/V vector. This format does not require checkpoint
calibration scales. Encoding uses round-to-nearest division and E4M3 conversion.
The GPU retains a bounded per-layer four-way hot-page cache; FP16 attention
staging is shared by serial QSA owners. Indexer history and active GDN/PLE
states retain their existing storage and arithmetic.

`KernelConfig.qsa_host_kv` selects this route. It is disabled while model-level
quality and latency are being measured. Host storage's default format is
E4M3; the generic cache dtype remains independent. This permits preserving
FP16 on other attention owners. Admission requires SM70, FP16 activations,
one local KV head, D256 and DCP1/PCP1. The startup report includes the reason
when the requested route is unavailable. No environment variable is added.

The default hot capacity is 8,192 tokens per owner, configurable through
`qsa_host_kv_hot_tokens`. At D256 this costs 4 MiB of E4M3 codes, plus scales,
tags and metadata. Thirty-two query rows share approximately 64 MiB of FP16
staging across owners. Prefill is processed in bounded query tiles. Exact
selected positions and causal masks are retained; no new sparsity heuristic
or draft attention window is introduced.

## Storage and replay

The hybrid allocator separates host attention from device recurrent pools.
Recurrent pages retain the original scheduler block IDs, but no longer inherit
the larger attention page's padding solely to share its allocation. Prefix
and speculative ownership remain under the existing cache managers.

New and tentative K/V writes update authoritative host storage and invalidate
the corresponding four-token hot page. A GPU epoch and protection pass keep
every current-call cache hit immutable during gathering. Misses use a bucket
lock with no waiting; contention and full protected sets read authoritative
host data. This bounds storage without dropping selected positions or
overflowing a miss queue. Fixed workspaces and mapped pointers are initialized
before graph capture. Each CTA owns its diagnostic counters, avoiding global
counter contention on the replay path.

The page-pool approach is informed by
[Strata](https://github.com/Niko1221/Strata/blob/82f46a8c8f475f001ad76d92f58f4a4f8ffb0253/include/strata/kernels/kv_stream.hpp).
No Strata kernel source is copied. Its published single-GPU INT8 measurements
are not FP8 TP4 performance evidence.

## Validation

CPU tests check mixed host/resident allocation, unique owners, unchanged
scheduler capacity, and reduced device pool bytes at the same block count.
GPU tests compare encoded bytes and scales with the PyTorch E4M3 oracle,
then compare every gathered value after eviction, contention, tentative-token
rewrites and captured replay. M1/M5/M20 are covered.

Model promotion requires a normal packaged runtime, matched host-on/off
C1/C4 measurements, teacher-forcing KL/top-1, natural completion checks and
eight-prompt acceptance intervals. Read actual host/device pool sizes, hot
cache bytes, staging bytes and hit/miss counters from workers outside timed
replay. A successful allocator test alone is not a model-level result.

The restored control uses Flash-Next IQ3_S, FP16 MTP4, TP4 on four
V100-SXM2-32GB GPUs, Torch 2.10.0+cu128 and CUDA 12.8. Cache is FP16 and
recurrent state is FP32, FULL decode graph, maximum context 9,216,
prefill budget 512, maximum concurrency four and greedy sampling. The wheel
source is `3577384357caac62f9822246b8a140dc552c24cf`. Eight natural prompts use
up to 600 output tokens with EOS enabled; timing probes use the unchanged
acceptance benchmark's fixed cohorts.

| Control | C1 ms/round | C1 tokens/round | C4 ms/round |
| --- | --- | --- | --- |
| Historical | 17.406 | 4.886 | 45.025 |
| Restored | 17.387 | 4.886 | 46.047 |

The restored eight-prompt mean draft acceptance is 45.59%; both short natural
completion checks end normally. Historical and restored natural token IDs
differ, so the historical transcript is not a bitwise oracle for a new run.
Use a fresh matched control and teacher conditions for host-cache qualification.
These numbers are resident-KV controls; host-KV model results are pending.
