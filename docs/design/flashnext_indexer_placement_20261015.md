# Restore Flash-Next GPU indexer history on 16 GB V100

The 9683 deployment and the qualified local deployment both keep authoritative
main QSA KV history in host memory. The remote deployment additionally moved
compressed indexer history to host memory to avoid a long-prefill PLE allocation
failure. That additional placement change substantially increases decode cost.

## Matched baseline and cause

The remote baseline uses SC117 Qwen3.8-Flash-Next GSQ-RCO abliterated IQ3_S,
TP4 on V100-SXM2-16GB, Torch 2.10.0+cu128, CUDA 12.8, greedy MTP4, FULL M5
graphs, 262144 total-token context, prefix caching, thinking, T0.6/P0.95/K20,
80% GPU / 20% host experts, E4M3 host main history, encoded target HOT32768,
FP16 draft HOT16384 and disk PLE. Source baseline is `699c864d3c`.

The same CPU instruction-execution prompt, seed 20261014 and 192-output limit
are measured after two warmups, with three measured requests per context.
Stable decode excludes TTFT and the first output token. Accepted length is
derived from the actual speculative counters; round costs are HTTP boundary
estimates including draft and scheduling, not isolated GPU verifier timers.

| Actual input tokens | Host-indexer output tok/s | TPOT ms | Round estimate ms | Accepted length |
| ---: | ---: | ---: | ---: | ---: |
| 32774 | 65.03 | 15.378 | 33.376 | 2.170 |
| 65604 | 40.24 | 24.852 | 52.741 | 2.144 |
| 131124 | 26.27 | 38.066 | 82.621 | 2.182 |

A same-kernel placement screen uses the production SM70 shared-key indexer,
M5/H4/D128, identical FP16 history and query values, compression ratio 4,
graph replay, five warmups and nine samples of twenty replays. Visible scores
and visibility counts are bitwise equal across placements.

| Context tokens | Mapped-host indexer ms | GPU indexer ms |
| ---: | ---: | ---: |
| 32768 | 0.7628 | 0.01869 |
| 65536 | 1.4752 | 0.01859 |
| 131072 | 2.9278 | 0.03389 |
| 261000 | 5.8650 | 0.05837 |

These are isolated single-rank operator times. They are not whole-model
speedups, and multiplying them by layers does not establish TP4 wall time.

## Device memory repair

Reuse the channel-chunked PLE implementation from PR #1209, extending its
admission to single prefills of at least 4096 tokens. Each such temporary is
bounded to one quarter of the unpadded step, rounded to 64-channel groups;
short single prefills retain the original path. Existing multi-prefill
half-step admission remains unchanged. There is no new environment switch,
precision change or native provider.

On the actual 6944-token, hidden-size-10240 FP16 V100 case, peak additional
device allocation drops from 285793280 to 214490624 bytes: 68 MiB released.
Output and committed convolution state are bitwise equal. This repairs the
single-prefill transient that prevented retaining GPU indexer history.

Correctness scope is the existing NVIDIA depthwise PLE convolution contract;
execution qualification includes CPU FP32 and V100 FP16, initial/fresh/null
state, empty caches, skewed multi-request padding and the actual large
single-prefill shape. CPU partition-dependent rounding is compared with one
thread, following the existing oracle. The selector checks 4095/4096 and 6944
tokens. These operator checks alone do not qualify all models or serving
concurrency. No native sources or native ABI change; the installed package's
source-matching bundled native extensions remain in use.

Restore the deployment's existing `QSA_HOST_INDEXER_HISTORY=false` setting.
Main history stays on host; compressed indexer history now uses device storage,
as in the local qualified deployment. The API address and model aliases are
unchanged. Shared environment packages are not modified.

## Validation and retained evidence

- CPU: 18 passed, 3 CUDA checks skipped.
- Destination GPU/CPU suite: 32 passed, including state equality and peak-memory
  assertions. The operator placement screen passes bitwise equality at all four
  context points.
- End-to-end candidate decode, cold prefill, tools and 256K capacity/quality:
  pending service restart and matched measurements; no speed claim yet.

Artifacts are retained in `reports/flashnext-9683-decode-parity-20261015/` and
the remote deployment's `decode-parity-20261015/` subdirectory. The original
PLE implementation and launcher are backed up for rollback. The baseline's
host-indexer workaround solved capacity but was not decode-speed qualified;
capacity and performance acceptance must remain separate.
