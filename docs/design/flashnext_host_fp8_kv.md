# Flash-Next active host KV on SM70

QSA attention history uses pinned host E4M3 bytes with one FP32 scale per
token, local KV head and K/V vector. This format does not require checkpoint
calibration scales. Encoding uses round-to-nearest division and E4M3 conversion.
The GPU retains a bounded per-layer page-slot cache. Attention reads protected
FP16 hot entries; unresolved host misses are decoded into shared FP16 scratch
before entering the existing FP32 accumulation and softmax loop. Indexer history and active GDN/PLE
states retain their existing storage and arithmetic.

`KernelConfig.qsa_host_kv` selects this route. It is disabled while model-level
quality and latency are being measured. Host storage's default format is
E4M3; the generic cache dtype remains independent. This permits preserving
FP16 on other attention owners. Admission requires SM70, FP16 activations,
one local KV head, D256 and DCP1/PCP1. The startup report includes the reason
when the requested route is unavailable. No environment variable is added.

`qsa_host_kv_dtype` defaults to `fp8_e4m3` for target history. Draft history
uses the independent `qsa_host_kv_draft_dtype`, initially `float16`; this
preserves speculative precision while target quantization is evaluated.
Selecting FP16 for both isolates placement and allocator changes from FP8
rounding. Both host formats reconstruct identical FP16 hot/staging layouts.

The default hot capacity is 8,192 tokens per owner, configurable through
`qsa_host_kv_hot_tokens`. At D256 this costs 8 MiB of reconstructed FP16 K/V,
plus tags and metadata. FP8 decoding occurs on misses, rather than every hot
read. Thirty-two query rows share bounded resolution maps and approximately
64 MiB of FP16 miss scratch across owners. Hot hits do not copy into that
scratch. Prefill is processed in bounded query tiles. Exact
selected positions and causal masks are retained; no new sparsity heuristic
or draft attention window is introduced.

## Storage and replay

The hybrid allocator separates host attention from device recurrent pools.
Recurrent pages retain the original scheduler block IDs, but no longer inherit
the larger attention page's padding solely to share its allocation. Prefix
and speculative ownership remain under the existing cache managers.
Without prefix caching, the scheduler pool is bounded by admitted concurrency
at maximum context, including recurrent speculative pages and alignment slack.
Freed memory remains available instead of being consumed by unused state pages.
An explicit block-count override retains precedence.

New and tentative K/V writes update authoritative host storage and any resident
hot entry from the encoded representation. A GPU epoch and protection pass keep
every current-call cache hit immutable during gathering. A physical-page-to-slot map deduplicates misses and a bounded clock sweep
reserves unprotected slots. Pending copies and capacity overflow use decoded
host data in miss scratch, with no waiting between CTAs. This bounds storage without dropping selected positions or
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
These numbers are resident-KV controls. The matched host experiment follows.

## First matched host experiment

A packaged source integrates the qualified HCX implementation with the current
main Python paths. Both runs use the same wheel and declared native provider.
Only host storage is enabled in the candidate; in this first experiment both
target and draft history use per-vector E4M3. The configured 9,216 context and
four requests are unchanged. The actual attention page is 816 tokens.

| Storage | C1 ms/round | C1 tokens/round | C4 ms/round | Mean draft acceptance |
| --- | --- | --- | --- | --- |
| Resident FP16 | 17.396 | 4.886 | 46.009 | 46.32% |
| Host E4M3, target and draft | 18.825 | 4.886 | 47.389 | 43.46% |

The eight-prompt paired acceptance difference is -2.86 percentage points;
bootstrap 95% CI [-4.63, -0.79]. This rejects default promotion. On 64 equally
conditioned target positions, mean/max KL is 0.002121/0.029512 and top-1 agrees
at 62/64 positions. Every logit is finite; both short completion checks produce
the same answers and stop normally. Coherent text alone does not meet the
acceptance requirement.

Device pools fall from 2.9344 to 1.5420 GiB per rank. The candidate additionally
retains 105.48 MiB of hot-cache buffers/metadata and 64.125 MiB of shared staging.
Total Torch allocation falls by 1.229 GiB per rank. Authoritative host pools
occupy 0.7942 GiB per rank, with per-vector scale arrays accounted separately.
Indexer and recurrent storage remains on device. The physical scheduler pool
contains 157 blocks rather than 273; it still admits the declared concurrency.

The next precision ablation preserves FP16 in the draft and provides an FP16
host reference. Direct reading removes query-private copying from the hot
path while retaining the staged reference for isolated comparisons. Eight
FP8/FP16 direct-reader tests cover M1/M5/M20/M32, misses, causal padding,
output gating and graph replay; the first source-only run agrees exactly with
the staged reader. Model latency and acceptance for this change remain pending.
On a separate V100-SXM2-16GB, 19 packaged GPU tests pass, including the 816-token
page, empty padded rows and captured replay. This is operator validation, not
four-card model capacity evidence.

## Precision ablation and cache experiments

Keeping the draft in FP16 measures C1 18.735 ms/round with 4.886 tokens/round
and C4 47.558 ms/round. Target top-1 agrees at all 64 equally conditioned
positions; mean/max KL is 0.002124/0.036558. Both short completions end normally.
Eight-prompt acceptance is 44.14% versus 46.32%; paired difference -2.19
percentage points, 95% CI [-4.07, +0.11]. This does not establish equivalence,
so FP8 remains unqualified for default promotion. Device pools remain 1.542
GiB/rank; target-FP8/draft-FP16 host pools occupy 0.855 GiB/rank.

Inlining host FP8 reconstruction in the attention loop is rejected: the
packaged M5 experiment takes about 196 microseconds versus 107 for the staged
reader at 8K. Four-way hashing also reaches only 75% hits for the fixed
four-request selection despite sufficient aggregate capacity. Changing the
hash alone does not resolve that structural conflict.

The page-slot prototype passes 38 GPU tests, including physical-page aliases,
tentative rewrites and captured replay. The initial direct/staged tests used artificial fixed tail columns and missed
a real selector boundary: the causal tail is compacted immediately after the
selected complete blocks. A fixed-tail loop drops live tokens at boundaries
such as 257 and 513. The corrected loop keeps the original split assignment
and truncates only the empty suffix. Regression tests use the actual QSA
expansion kernel at contexts 14, 255, 256, 257, 511, 512, 513 and 2050, for
M1/M5/M20/M32 and FP8/FP16 storage. A conditional around tensor-core arithmetic
is slower and is rejected.

Same-card synthetic graph timings for the current source prototype:

| Context | Rows | Hot tokens | Resident us | Staged us | Hot/miss reader us | Hit rate |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | 5 | 8192 | 57.9 | 68.2 | 41.0 | 100% |
| 128 | 20 | 8192 | 294.1 | 309.7 | 71.0 | 100% |
| 8192 | 5 | 8192 | 55.0 | 85.6 | 61.3 | 100% |
| 8192 | 20 | 8192 | 274.3 | 360.6 | 302.1 | approximately 100% |
| 32768 | 5 | 32768 | 51.0 | 81.8 | 58.1 | 100% |
| 32768 | 20 | 32768 | 259.0 | 344.5 | 281.8 | 100% |

These selections are fixed and shared by five queries per request. They do
not prove model hit rates or endpoint gains. Packaged qualification and an
FP16 host model reference are the next checks. Teacher capture additionally
saves small post-RoPE KV samples outside the timed replay for format analysis;
resident controls do not change their capture behavior.

## Rejected FP16 host model reference

The first slot-cache FP16 model run, before the compact-tail correction, measures
C1 18.608 ms/round, 4.886 tokens/round and C4 44.332 ms/round. Mean acceptance
is 45.53%, with paired difference -0.79 percentage points and 95% CI
[-1.77, +0.29] against the resident control. However, 64 target teacher positions
show mean/max KL 0.012746/0.222238 and top-1 agreement 62/64. These results
reject the implementation even with unquantized host storage; the compact-tail
bug must be corrected before assessing format-induced quality loss.

The reported C1 is the benchmark's arithmetic mean of its before/after probes,
including the after probe's large outlier. The approximately 18.14 ms median
is diagnostic only. Whole-workload cache hits are about 98.97–99.11% for target
owners and 99.02% for the draft; these include prefill and generation rather than
an isolated C1 cohort. High hit rate does not establish correct attention.
