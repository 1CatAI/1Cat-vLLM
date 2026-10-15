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

The first GPU-indexer attempt passes 32K and 64K decode, reaching 119.53 and
101.94 output tok/s with unchanged measured acceptance lengths. At 128K,
`hc_assemble_ranks` fails a 40 MiB allocation with 11.94 MiB device memory free.
The PLE repair alone is therefore insufficient for the full-context workload.
This failed attempt is retained separately and is not the final deployment.

A separate four-stream, graph-captured FP16 cuBLAS score-matrix screen measures
34079744 workspace bytes under the default configuration versus 525312 with
`CUBLAS_WORKSPACE_CONFIG=:16:8`. Output SHA256 is identical. One timing screen
measures 0.1217 versus 0.0967 ms; this is only a rejection screen for a large
slowdown, not an accepted performance gain. The final deployment additionally
uses this standard CUDA library workspace configuration, requiring a fresh
process and new graph capture. Whole-model speed, task quality and long-prefill
headroom remain the final acceptance gate.

The second attempt restores 32K/64K decode to 121.63/109.16 tok/s but still
fails the same 40 MiB HC allocation at 128K, now with 39.94 MiB free. Retain
that failure rather than treating a near-fitting allocation as accepted capacity.

Reuse the existing history-attention workspace across decode and tiled FP16
prefill, processing the latter in 32-row slices instead of 64. Rows retain the
same native split states and accumulation order. The FP8 target's direct
`run_rows` path remains unchanged. Removing the separately cached 64-row
workspace releases about 12.5 MiB at selection width 2051. The shared allocator
retains sufficient reference and native capacity regardless of initialization
order. Existing attention checks pass 43 cases; two allocation-order checks and
five bitwise tiled/capture/replay checks at rows 31/32/33/64/65 pass. The first
five failures were an invalid FP16 test scale fixture (per-token scales belong
to encoded history); the corrected fixture uses the existing two-scale FP16
contract. No runtime limits or numerical tolerances were loosened.

Artifacts are retained in `reports/flashnext-9683-decode-parity-20261015/` and
the remote deployment's `decode-parity-20261015/` subdirectory. The original
PLE implementation and launcher are backed up for rollback. The baseline's
host-indexer workaround solved capacity but was not decode-speed qualified;
capacity and performance acceptance must remain separate.

## Takeover: memory headroom and the 128K staging boundary

The fourth attempt sets NCCL's standard buffer to 1 MiB. Its complete decode
sweep reaches 132.73 / 116.50 / 105.20 output tok/s at the same 32774 /
65604 / 131124 inputs. Round estimates are 18.216 / 21.292 / 21.111 ms;
measured acceptance is now 2.418 / 2.481 / 2.233, so unchanged acceptance is
not claimed. Arithmetic, tool roundtrip and natural-EOS retrieval/arithmetic
pass at 32761 / 131070 / 260977 inputs. The last quality request takes
345.94 seconds. A second separate boundary-speed request is cancelled to
proceed with the memory repair. This attempt reaches only 6 MiB free in
200-ms driver sampling and is not accepted as a deployment with headroom.

Both machines use Torch 2.10.0+cu128, NCCL 2.27.5, cuDNN 9.10.2,
CUDA runtime 12.8.90 and cuBLAS 12.8.4.1. Their four-card topology differs:
9683 has NV2 between every pair; the local group has mixed NV1/NV2 links.
The same two-process-group all-gather probe with a 1 MiB buffer uses
406847488 non-Torch bytes locally and 591396864 on 9683: 176 MiB additional
usage. NCCL INFO reports local 8 collective / 8 P2P channels, versus remote
12 collective / 16 P2P channels. Setting `NCCL_MAX_CTAS=8` on 9683 makes
both counts 8 and matches the local 406847488-byte footprint. The 40 MiB
all-gather screen changes from about 0.37--0.40 to 0.50 ms; it is not a
whole-model speed qualification. A 4 MiB buffer with that limit uses
440401920 bytes and measures 0.467 ms. Four CTAs with a 1 MiB buffer uses
272629760 bytes but measures 0.783 ms, so it is not chosen solely for memory.
These are standard NCCL configuration inputs, not a replacement communicator.

Two additional lifetime repairs retain existing operators and arithmetic:

- HC sharded fallback releases the local projections, packet and gates as
  soon as consumed. At 8192 rows, the same-operator GPU comparison reduces
  additional peak allocation from 152436736 to 83951616 bytes (65.3125 MiB),
  with bitwise-equal finite outputs. Existing eager and changed-input graph
  checks cover both packed layouts and rank orders.
- QSA scoring releases each cuBLAS score tile and each consumed logits tile
  before the next allocation. CPU lifetime checks exercise multiple column
  and row tiles; the CUDA test compares the same tiled arithmetic bitwise
  and replays changed query/key inputs.

`kernel_config.qsa_host_kv_prefill_stage_slots` is a typed initialization
policy, accepting only 1 or 2, defaulting to the existing 2. It is serialized
and included in the effective hash only when host prefill staging is selected.
One slot retains prefix prefetch on the existing side stream: the preceding
attention completion event fences reuse, while the following non-QSA work can
still overlap the next prefix copy. No new kernel, precision, cache storage,
request indexing or state commit algorithm is introduced.

For E4M3 authoritative history, one 262144-token stage occupies the same raw
bytes as two 131072-token stages (128 MiB total, plus scales), and removes the
128K fallback to per-query host reads. These context units are tokens in
encoded target history. FP16 history consumes twice the bytes, so a byte
budget described by 262144 E4M3 tokens holds only 131072 FP16 tokens; the
existing fallback remains in effect outside that capacity. This does not
change the model's attention algorithm or establish constant-time prefill.

Destination validation: 63 passing cases, including configuration transfer
and hashing, exact staged prefix/current-token writes, HC output/graph tests,
QSA tiled score/graph tests, and delayed-reader reuse for both slot counts at
4096 / 131088 / 262144 E4M3 tokens. The unused-policy hash checks pass locally.
The new whole-model run retains TP4, MTP4, FULL M5 graphs, encoded HOT32768,
FP16 draft HOT16384, 80/20 expert placement, 256K context and chunk8192.
Whole-model peak live/driver memory, verifier distribution, cold prefill,
long-context quality and fresh-process reproduction remain pending; no
512-MiB headroom or 16-ms verifier acceptance is claimed from operator tests.

## Ordinary verifier window and peer prefill port

The standard Worker restart, without step-event fences, passes the matched
32K/64K/128K sweep at 152.15 / 139.99 / 110.49 output tok/s. Natural-EOS
quality again passes through 260977 tokens, taking 113.85 seconds at the long
boundary. The equivalent preceding single-stage diagnostic run took114.02s,
compared with345.94s on the 128K-stage workaround; full staging repairs most of
that cliff. A diagnostic private CPU-pool screen gives 15.96ms rank-max verifier
mean at32K (p99 23.97ms); this is kept separate from ordinary tracing.

Ordinary CUDA graph-envelope tracing measures504 committed target rounds over
eight matched192-token32K requests. Every rank has512 state-commit triples:
one prefill plus63 metric-reconciled target verifications per request. There
is no extra pipeline-end commit in this trace. All504 target graph starts are
present and associate uniquely with their following same-stream commit. This
is distinguished from the older trace's extra pipeline-end replay; the parser
has an explicit tail-count selection, rather than imputing or dropping events.

From the earliest target GPU start to the latest target state-commit end across
TP4, mean is **15.216741ms**, p50 14.5569745ms, p90 16.915299ms, p99 24.133488ms,
maximum40.109622ms. All504 windows are exact; there are no conservative missing
start bounds. This passes the historical mean<=16ms criterion, not an every
round or p99 target. The source is996e3a161c, main history on host, compressed
indexing on GPU, MTP4/FULL M5, encoded HOT32768 and FP16 draft HOT16384. Helper,
EngineCore and PLE threads use disjoint physical-core pools. Standard NCCL
MAX_CTAS6/buffer4MiB uses390070272 bytes in the two-group screen,48MiB less
than8CTAs/buffer4MiB, with0.499ms all-gather. Unprofiled output throughput,
round estimates and this independently traced verifier window remain separate.

Further reuse of peer commit31edb1fe3d adds only its qualified QSA TP-row and PLE
prefill stages. Target/decode graphs and native sources remain unchanged. The
QSA adapter qualifies replicated queries/history for TP4/DCP1/PCP1; the selector
admits single-request FP16/H4/D128 prefill at4096..9216 query rows with sufficient
score work, gathers compressed512-block selections and expands globally.
Decode, capture, mixed requests, threshold neighbors and other layouts retain
the existing selector. The per-tile logits lifetime repair stays inside the
loop for both paths. The opt-in remains a typed active-QSA computation policy.

The PLE stage reuses the generic PLE kernel owner, consumes existing input and
state directly, commits valid base state separately and retains the native FP16
SiLU boundary. SM70/FP16/10240channels/4taps/dilation3/base9 are the accelerated
subset; unsupported layouts use the existing convolution. Both opt-ins default
to false in the library. No private kernel artifact or native ABI update is
required: the Triton source is shipped with the ordinary package.

Local port checks:64CPU cases pass with24GPU skips;50 PLE GPU cases pass. At the
actual7344-row geometry, compared with this branch's bounded quarter-channel
control, fused PLE reduces additional peak from227123712 to150405120 bytes,
with bitwise-equal output and state, and0.85545 versus7.34761ms in the repeated
control. Destination affected suites pass98cases. The TP4 selector passes all
four ranks at query threshold neighbors4095/4096/4097 and9216/9217, 65536
compressed history columns, mixed requests and captured fallback.

The combined serving candidate passes arithmetic, a tool roundtrip and
natural-EOS retrieval/arithmetic at32761/131070/260977 input tokens. The same
cold-prefill hashes give median5126.6801tok/s and6.766172s, versus4881.6560tok/s
and7.088988s before the opt-ins. A further261044-input,192-output SSE passes.
Across all four ranks, the minimum estimated live headroom is530.963MiB;
driver sampling reaches60MiB. These are different quantities: unused allocator
reservation explains part of the gap, but the physical margin is still too
small for final deployment acceptance. Shared packages and native artifacts
are unchanged, and all ten changed Python runtime files match the owned source.

## Further HC memory candidate

Port the peer's row-owned HC stages into this checkout while keeping the
ordinary decode and fallback backbones. The candidate partitions the
multi-stream residual by physical query-row owner, defers attention and MoE
reductions to that owner, and gathers block inputs and the final MTP state at
their existing boundaries. Partial routed/shared MoE tensors remain separate
until their FP32 sum and reduction. Unaligned tails pad only the HC carrier;
PLE and MTP copy only actual query rows and retain untouched capacity tails.

The typed opt-ins default to false. Deployment qualification is deliberately
restricted to SM70 FP16 GGUF with HC4/H2560/lowrank320, sharded HC storage,
HCX, TP4/PP1/DCP1/PCP1, one sequence, backed dynamic shapes, no LoRA,
4096..16384 query-token chunks, no graph capture and no FP16 partial
accumulation/reduction. Other inputs retain the existing backbone. A separate
compiled view borrows the original parameter/state owner; nested/exceptional
forward calls restore the reduction context. Native sources and ABI are
unchanged. The optional peer-normalization owner is kept **off** for this
deployment candidate; its allocation is not required for row partitioning.

The imported synthetic contract is declared before destination runs: finite
hidden/block outputs, maximum row-relative L2<=0.001 against independent FP32
arithmetic, injection-logit aggregate L2<=0.001, column-RMS-normalized maximum
error<=four FP16 epsilons, and injection-factor absolute error<=0.0005. A
48-step recurrence uses relative L2<=0.003 and the same factor allowance.
This preserves the peer's earlier failed near-zero paired-logit metric as a
failure; it is not relabeled or relaxed after a destination run. Checkpoint
task answers, EOS, repetition and state semantics remain required gates.

Local configuration/model/producer checks pass74cases on an explicit
UnspecifiedPlatform harness. One initial fixture failed because that harness
already registered the CPU custom-op implementation; the fixture now adds its
test implementation only when absent, and the entire affected suite passes.
Destination native stage/model-adapter tests, whole-model peak memory, paired
long-context quality and fresh-process verifier measurements are pending.
The library defaults and production defaults are not promoted from CPU tests.

Destination HC qualification subsequently passes79affected cases and the
four-rank model-adapter cases4096/4097/7344/7345/8192, including padding,
compilation, changed-input replay and preserved PLE/MTP capacity tails.
Independent FP32-reference and48-step recurrence checks pass7344/8192. At8192,
the stage's temporary peak drops502.28125to170.015625MiB. The paired near-zero
logit metric remains explicitly failed. Whole-model retrieval/arithmetic/EOS
passes through260977 tokens, a further261044-input192-output SSE passes, and
cold-prefill median is5196.6917tok/s /6.664047s.

The ordinary source760context curve reconciles2480exact committed windows:
means15.893678/16.259436/16.365536/17.544996ms at32774/65604/131124/261044
input tokens. Corresponding p99 values are31.449087/31.458691/30.764757/
28.706752ms. The long mean increases10.39%; graph-to-state residual remains
about0.676ms. This is not an all-context16ms result. Short node traces are
used only for composition: rank-zero Top-K service grows0.5225to1.8246ms,
index scoring0.2093to0.7054ms and hot resolution0.9679to1.4489ms. These
overlapping/category service measurements are not an additive wall-clock table.

## Startup allocator reservation

The HC run reduces live allocation, but its driver-free memory remains low.
An allocator snapshot shows1257.7MiB unused in ordinary expandable segments
and another50MiB reserved in captured private pools. Several CUDA streams
retain roughly170--204MiB unused each. These are allocator reservations rather
than additional live model weights or a change in expert placement.

`kernel_config.qsa_host_kv_trim_warmup_cache` defaults to false. When requested
and an actual host-QSA consumer exists, the common GPU Worker returns unused
allocator pages once after model/graph and auxiliary kernel warmup. Inference
does not trim. Active tensor/state ownership and captured pools are retained.
This startup resource policy is excluded from the computation hash and is
serialized through the normal per-engine configuration; no shell/import-order
requirement or package installation is introduced. Both V1 and V2 reach the
same final warmup boundary. Models without host-QSA consumers keep a no-op.

Five configuration/consumer CPU cases and all six destination cases pass.
The GPU case creates auxiliary scratch after capture, returns that unused
reservation, preserves active state bitwise and replays changed inputs.
The initial fixture allocated scratch before capture, which itself already
cleared the cache; that failed memory-drop assertion is retained and corrected
to reproduce post-capture auxiliary warmup. Whole-model memory headroom,
unprofiled performance and fresh-process public serving remain pending.

## Exact decode Top-K partition candidate

The Top-K node is the leading long-context growth hypothesis. Warp-aggregated
histogram atomics fail the rejection screen because random/tied cases regress.
A warp-local final emission screen preserves all90exact cases but releases
only about10% of kernel cost, insufficient to justify a native rebuild. Both
remain rejected research artifacts; qualified native sources/artifacts are
restored, and no private kernel DSO is a runtime dependency.

The new opt-in `sm70_sparse.qsa_decode_partition_topk` instead composes the
existing exact selector over8192-column FP32 partitions, then selects the
global Top512 from their union. A globally selected item must be in its local
Top512. Local and merged selections retain lower-index ties and increasing
index order; gathering candidates in partition/index order preserves the
original global tie and accumulation order. The contiguous causal prefix
produces a contiguous candidate prefix, including a short tail. No score
precision, index validity or state/rollback semantics change.

Admission is SM70, single-request M5/M10, tokenTop2048/compression4 through
the existing exact selector and32768..69632physical compressed-score columns.
These are capacity-column units, not input-token limits. Other batches,
devices, widths and prefill rows retain the existing selector. The policy
defaults to false, is serialized normally and contributes to the consuming
QSA engine hash only when active. Both native stages are shipped normal
package operations; three source Triton layout kernels prepare/gather/remap.

Destination28cases pass, including5/10rows,32768/65600/69632capacities,
strided input, guarded caller output, changed-input/length graph replay,
negative/empty/511/512/513/8191/8192/8193 live-length neighbors, narrow scores,
ties, signed zero, infinities, an independent stable-sort oracle and existing
selector regressions. Configuration transfer and qualified/unused hash checks
pass. The first hash fixture named a nonexistent policy field; its corrected
fixture does not change runtime behavior or relax a gate.

The8192-tile prototype at65536live columns measures0.0704/0.0831/0.0881/
0.1063ms versus0.1099/0.2602/0.2370/0.2955ms for random/narrow/tied/zero rows,
with exact outputs. At8192live columns it is slightly slower, so whole-model
short/long verifier distributions remain required. A unique64-bit key final
merge using Torch Top-K is slower and rejected. The combined model candidate
retains HC/PLE/TP-row acceleration and enables startup cache trimming;
task quality, physical/live headroom and all-context16ms acceptance are pending.
