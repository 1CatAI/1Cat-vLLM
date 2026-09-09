# Native MiniMax H3 migration control

## Current campaign direction

New Attention development and exhaustive workflow acceptance concentrate on
FlashAttention-V100 at the user's request. The already validated FlashInfer
implementation remains available, but new FI optimization and exhaustive FI
qualification are paused. The historical parallel-backend decisions below
are retained as evidence, not current work allocation.

The full-request TP4 four-step native peer-row FA measurement records a
minimum-card median of 53.235745 useful TFLOP/s, 58.293218 seconds denoise,
and CV 0.076959%. It does not meet the >80 gate. Original floating weights,
W8A16, legal adapters/reference inputs, explicit VSA/cache algorithms and
independent official/human quality remain part of the campaign scope.
Only configurations satisfying the unchanged quality and full performance
gates may enter automatic selection.

See [FA_DEVELOPMENT.md](FA_DEVELOPMENT.md) for current bottleneck evidence and
rejected kernel candidates, and [CAMPAIGN_RESULTS.md](CAMPAIGN_RESULTS.md)
for the per-workflow qualification table. The owned stack remains Draft;
local operator results are not end-to-end model acceptance.

Latest FlashInfer change:
[FLASHINFER_LOCAL_ROTATION.md](FLASHINFER_LOCAL_ROTATION.md).
Rotate FP16 rows on their owner before all-gather, avoiding duplicated QKV/MLP
ConvRot while retaining exact matrix operations and TP sums. The unchanged
39-frame/20-update run takes 61.538397 seconds, 56.286054 useful TFLOPS/card;
final video/audio latents and fresh MP4 remain bitwise equal. The 123-test
suite and both independently launched TP4 regressions pass. Current tracing
attributes 41.10% to GEMM, 39.81% to attention, 13.52% to communication and
1.02% to ConvRot. Gather/projection overlap, Q64 aliased K/V and updated Q32/N64
reuse controls do not improve the selected path and are rejected. The combined
distributed test launcher hang is retained; use separate torchrun lifetimes.
Human quality, <50 seconds and >80 TFLOPS remain open.

Latest FlashInfer change: [FLASHINFER_V4.md](FLASHINFER_V4.md).
Four-row exact V transposition permits 64-bit shared stores, reducing the
unchanged 39-frame/20-update denoise to 62.434779 seconds and 55.477950 useful
TFLOPS/card. Final video/audio latents and fresh MP4 remain bitwise equal.
The 122-test suite and three 12-case sanitizers pass. Human quality review,
<50 seconds and >80 TFLOPS remain open; residual sharding still defaults off.
The communication-overlap prototype is rejected after FP32 reduction error
amplifies during sampling. Future optimization prioritizes bitwise equality;
see [FLASHINFER_OVERLAP.md](FLASHINFER_OVERLAP.md) for the fresh baseline trace
and failed paths.

Latest FlashInfer integration: [FLASHINFER_SILU.md](FLASHINFER_SILU.md).
Dependency `9764b6c202`'s shared FP32 SiLU/FP16 preparation fusion works with
our FP32 residual reduce-scatter. The unchanged 39-frame/20-update run takes
63.565580 seconds (54.491025 useful TFLOPS/card), with bitwise-equal video/audio
latents and fresh MP4. The 122-test suite and corrected four-rank block test
pass. Attention/W8 binaries remain unchanged. Human quality, <50 seconds and
>80 TFLOPS are still not accepted.

Latest FlashInfer warp transpose: [FLASHINFER_VTRANSPOSE.md](FLASHINFER_VTRANSPOSE.md).
Warp-local exact FP16 pair exchange removes a shared staging round trip and
one CTA barrier. The unchanged residual-sharded 39-frame/20-update run takes
64.920336 seconds (53.353907 useful TFLOPS/card), down from 65.804661 seconds.
Video/audio latents and fresh MP4 remain bitwise equal. The 117-test suite and
three 12-case sanitizers pass. Fresh NCU shows 37.57% Tensor pipe activity;
direct V stores introduce bank conflicts, so MIO remains a bottleneck.
Human quality review, <50 seconds and >80 TFLOPS remain open.

Latest FlashInfer operator change:
[FLASHINFER_OPERANDS.md](FLASHINFER_OPERANDS.md). A 68-half probability stride
and 64-bit fragment loads remove almost all repeating shared-memory bank
conflicts. Full unchanged 39-frame/20-update denoise with residual sharding
falls from 66.863312 to 65.804661 seconds (52.636903 useful TFLOPS/rank).
Video/audio latents and the fresh MP4 are bitwise equal. The 117-test suite and
three 12-case sanitizers pass. Fresh NCU verifies excessive wavefronts fall
from 235,879,168 to 1,042,944; Tensor pipe activity reaches 36.33%.
The first same-name native A/B import collided and is marked invalid; qualified
module names and identity assertions fix the harness, and corrected short/long
controls establish the gain. Human quality review, <50 seconds and >80 TFLOPS
remain open. Residual sharding still defaults off.

Latest opt-in FlashInfer improvement:
[FLASHINFER_RESIDUAL.md](FLASHINFER_RESIDUAL.md). TP4 FP32 residual sharding
reduces the unchanged 39-frame/20-update denoise to 66.863312 seconds,
51.803500 useful TFLOPS/rank. The native implementation matches the prototype
bitwise; it differs from the replicated baseline (video SSIM 0.986179).
Automatic media checks and the four-rank block/padding/FP32-range regression
pass. Human quality acceptance remains pending, so the new option defaults
off. The below-50-second and >80-TFLOPS/card targets remain incomplete.
Fresh counters from the retained attention binary show 34.91% Tensor pipe
activity and 50.09% scheduler cycles without an eligible warp. Three query
reuse variants regress; the Q128/K128 alias variant spills and is not run.
Q96/K128 removes spills but remains slower. Replicated INT8 MLP weights with
local token rows also regress (two updates 6.645113 -> 7.030977 seconds), despite
removing two MLP collectives; this artifact-only path is rejected without a
full-video run. Its per-rank valid-row accounting and failed harness attempt
are recorded in the same document.

Latest retained FlashInfer change: [FLASHINFER_TO50.md](FLASHINFER_TO50.md).
Q shared-memory swizzling and an exact transposed FP16/cuBLASLt weight cache
reduce the unchanged 39-frame/20-update denoise to 70.828264 seconds,
48.903550 useful TFLOPS per rank. Video/audio latents and fresh MP4 remain
bitwise equal. 105 tests pass after dependency reconciliation; previous
FlashInfer sanitizer checks and seven added memory checks pass. The below-50s
milestone, human quality review and per-card >80-TFLOPS acceptance remain open.

Latest FlashInfer investigation: [FLASHINFER_ROOT_CAUSE.md](FLASHINFER_ROOT_CAUSE.md).
Fused QK preparation and an 8x8 V transpose reduce the same 39-frame/20-update
denoise to 75.140751 seconds, or 46.096872 useful TFLOPS per rank. Video/audio
latents and freshly decoded MP4 are unchanged from the preceding K64 run.
79 tests pass. The next short-run milestone is below 50 seconds with unchanged
parameters and no quality regression; >80 TFLOPS per rank remains incomplete.

FlashInfer feeding follow-up on its own branch: see
[FLASHINFER_FEEDING.md](FLASHINFER_FEEDING.md). Q128/K64 reduces complete
39-frame/20-update denoise to 78.968862 s, or 43.862270 useful TFLOPS per rank.
65 tests and focused sanitizers pass; fresh decoding passes automatic checks.
Output rounding changes, human quality review and the >80-TFLOPS gate remain
open. The parallel FlashAttention task is outside this branch's scope.

Status: implementation in progress; no video quality or 80 TFLOPS acceptance yet.

Workflow and adapter expansion is tracked in [WORKFLOWS.md](WORKFLOWS.md) and
[ADAPTATION.md](ADAPTATION.md), with validation entries at the end of this file.

Current decision: development proceeds on separate FlashAttention and
FlashInfer branches. FlashAttention remains the default H3 backend;
independent FlashInfer optimization and reproduction explicitly select
`FLASHINFER_SM70`. Both paths target >80 useful TFLOPS/card and below 50
seconds for the unchanged 39-frame, 20-update development workload. Preserve
D128 MHA, its original scale and valid-token FLOP accounting, including when
reusing the existing D256 TensorOp architecture; padding never increases
useful FLOPs. Residual sharding remains opt-in, and integrating development
code does not establish the pending quality or performance acceptance.

## 2026-09-09 FlashInfer integration with main

PR #564 synchronizes with `onecat/main@da65a6b23e9c`, which already contains
native H3 (#557), SM70 operators (#558), workflows/adapters (#565), shared
residual/local-rotation support (#568) and validation tools (#559). Both
backend histories and the workflow records are retained. Configuration keeps
main's dual-backend residual support and adapter rejection. The local-rotation
regression checks both backends within one distributed fixture lifetime.
Shared model dataflow and native FlashInfer/W8 CUDA sources and binaries are
unchanged from the measured local-rotation build.

The integrated video suite passes **266 tests, 2 skipped, 1 deselected**, using
the preceding FlashInfer suite command against `e7fa44deb253`. After including
PRs #568/#559, **24 targeted tests pass** for the changed configuration, metrics
and regressions. The final INT8 local-rotation and default unquantized TP4
block modules each pass on every rank, covering both attention backends in
separate torchrun processes. Raw commands and logs are retained in the
campaign's `feeding-gather/merge-main/` and its `final/` subdirectory.
The 61.538397-second result remains the preceding
development measurement; this integration does not add a performance claim
or complete the pending human quality, 50-second or 80-TFLOPS gates.

## 2026-09-08 FlashAttention-V100 D128 native route

### Local ConvRot before gather: 58.19 seconds

The optional residual-sharded route now rotates normalized FP16 rows on their
owning rank before all-gather. QKV and gate/up projections consume those exact
gathered bits without repeating ConvRot. Each rank rotates one quarter of the
rows; the original ConvRot256 arithmetic, INT8 scales, projection shapes and
FLOP hooks are preserved. CPU/unquantized paths retain ordinary forwards.
This reuses a read-only working-tree snapshot from the independent FI task,
based on `53780abef0e9ef190aa55a1f61d8fd5fbd390aa1`, with the regression adapted
to native FlashAttention. Snapshot patch SHA256:
`ce41592338adeef0388c7b5b826351d378c468bfa58e7df3def662cf7056dc4b`.
The source was uncommitted in that tree at capture; no FI attention delegation
or H3 CUDA binary change is involved.

The unchanged 1344x768, 39-frame/24-FPS, seed42, INT8 ConvRot FL2VA, TP4 GPU0-3
workload completes **20 actual updates in 58.190385 s**, or **2.909519 s/update
and 59.524500 useful model TFLOPS/card**. This reduces the previous optional
58.794562 s result by 0.604177 s (1.03%). Per-card useful FLOPs remain
3,463,753,579,661,312. Peak denoise Torch allocation is unchanged at
**6.195001125 GiB/card**; NVML peak device-used memory is 8.046386719 GiB/card.
Persistent FP16 cache and Lt workspace remain zero. This is one unprofiled,
cached-text development run after a one-call warmup, not formal acceptance.
Three paired two-update checks give medians 5.878070 ->5.813375 s (1.10%).

Both complete video/audio latent tensors are **bitwise equal** to the previous
58.79-second run, finite and identical across ranks. Its decoded media is
reused only after these checks and verification of the MP4 hash below. There
is no new VAE or end-to-end timing. Automatic validity is preserved; the prior
human audiovisual review remains pending. The option still defaults to false;
the flag-off baseline remains 62.804019 s. Omit the flag to restore that path,
or use parent `2a560b0a7fa0` to revert only the local-rotation change.

Validation: 62 targeted tests pass, one GPU case is deselected. A separate
torchrun invocation runs one actual four-rank test, passing on every rank:
cached/uncached INT8 projections, two valid/padded lengths, consecutive blocks,
rank-specific weights, residuals above 65504, exact output and unchanged FLOP
hooks. Pre-commit passes. Do not combine distributed GPU test modules in one
pytest lifetime: the repository fixture tears down distributed state between
tests. Earlier sanitizer runs cover unchanged operators, not a new run here.

Eight further attention variants remain artifact-only. All match the native
output exactly across 12 tested lengths and pass sampled FP32 references;
none establishes a stable paired speedup including wrapper/copy costs:

| Candidate | Native control ms | Candidate ms | Decision |
| --- | ---: | ---: | --- |
| Register row max/sum | 20.210688 | 20.071424 | No stable paired gain |
| Separate max/sum exchange | 20.093952 | 20.232191 | No gain |
| Head-contiguous Q/K/V packing | 19.866625 | 20.254721 | Packing loses |
| Direct Q/K vector iterators | 20.256767 | 21.655552 | Hot register spills |
| Direct Q iterator | 20.048897 | 20.049919 | No gain |
| Direct K iterator | 19.971071 | 20.135937 | No gain |
| Direct V iterator | 20.254721 | 19.984385 | Clock drift; unstable gain |
| PV-only 64x32 warp | 20.199425 | 23.879681 | Slower after mapping fix |

Query-window attention/projection overlap also loses, including a version
retaining the full-shape attention specialization and validated cloned GEMM
plans. A three-window TP4 block pipeline takes about 6.04 s versus 5.87 s for
two updates and introduces nonzero latent deltas; it is not integrated. Its
raw hook counter includes 29 padding rows and is explicitly invalid for
acceptance accounting. See `flashattention-register50/failed-paths.json` for
exact jobs, sources and outcomes; do not repeat unchanged candidates.

Latest evidence: `flashattention-register50/REPORT.md`, `report.json`, source
snapshot hashes, paired probes, test logs and NVML curves. Media:
`outputs/quality39-int8-flashattn-localrot-20steps/FLASH_ATTN_V100/`.
**Under 50 seconds, attention 60 TFLOPS, formal per-card 80 TFLOPS and human
quality acceptance remain incomplete.** GEMM/attention operand reuse remains
the main optimization target; this small ConvRot gain does not resolve it.

### Optional FP32 residual row sharding: 58.79 seconds

`--residual-sequence-parallel` enables an experimental TP4 FL2VA INT8 path
for either native SM70 attention backend. It defaults to false in both
`video generate` and `video serve`. This reuses the committed residual dataflow
from the independent FlashInfer branch at `5b4f0bba31`, adapting its backend
gate and tests for FlashAttention-V100. No FlashInfer attention code is ported
or delegated to when FlashAttention is selected.

Each rank retains only its FP32 residual rows through the DiT blocks. The
existing normalization boundary produces FP16 rows, which are gathered before
QKV and MLP projection. Output projections retain FP32 partial sums and use
FP32 reduce-scatter. Final heads receive the gathered FP32 rows. This reduces
repeated normalization/gating work and intermediate data movement, preserving
INT8/scale information, all GEMMs, valid attention tokens and denoise updates.
The feature rejects unsupported TP sizes, BF16 checkpoints, Ref2VA, adapters, multiple
requests and simultaneous Ulysses hooks. The original path remains available
by omitting the flag; source rollback is kernel parent `9764b6c20259`.

On the unchanged 1344x768, 39-frame/24-FPS, seed42, INT8 ConvRot FL2VA,
TP4 GPU0-3 request, **20 actual updates take 58.794562 s**:
**2.939728 s/update and 58.912822 useful model TFLOPS/card**. This is a 6.38%
time reduction from the 62.804019 s default. Per-card useful FLOPs remain
3,463,753,579,661,312. Peak denoise Torch allocation falls from 6.566735744 to
**6.195001125 GiB/card**; NVML peak device-used memory is 8.171386719 GiB/card.
Persistent FP16 weight cache and cuBLASLt workspace stay zero. This is one
unprofiled development run with a one-call warmup and prompt-verified cached
text, not an end-to-end or formal repeated measurement.

Fresh VAE decoding takes 8.478661 s, with 34.904292 s VAE loading reported
separately. All automatic media checks pass. FP32 reduction order changes:
video/audio latent relative RMS differences from the 62.80-second baseline
are 3.87465% /1.20270%. Decoded-video SSIM is 0.983989 and PSNR 38.238221 dB;
these are auxiliary comparisons, not quality thresholds. Eight sampled frames
show a consistent red boat, yellow duck and background. Human audiovisual
scoring remains pending, so the option is not promoted to the default.
MP4 SHA256 is
`f00ba75587f58e2a63a105647cb634eebd60b154ab7b3551f385ca2df96e5243`.

Separate two-update Nsight Systems traces explain the gain. Rank0 exclusive
wall seconds close to each trace's own denoise NVTX span:

| Category | Replicated residual | Sharded residual |
| --- | ---: | ---: |
| GEMM | 2.539663 | 2.550870 |
| Attention | 2.005245 | 2.023926 |
| TP collectives | 1.076831 | 0.845654 |
| Other GPU kernels | 0.440786 | 0.237076 |
| ConvRot | 0.125264 | 0.126346 |
| Weight dequantization | 0.059483 | 0.059765 |
| Copies | 0.012490 | 0.012507 |
| FP16 preparation outside fused kernels | 0.000026 | 0.000025 |
| No recorded GPU activity | 0.031895 | 0.035798 |
| Complete trace span | 6.291683 | 5.891967 |

The fused activation's preparation is included in other GPU kernels. GEMM
and attention still consume about 78% of the new trace. This is not an idle
optimization. Explicit copies are small; the profile does not establish an
HBM-copy bottleneck. At 60 useful attention TFLOPS alone, the old trace still
projects roughly 61 seconds for 20 updates with all other costs held fixed;
this is an Amdahl illustration, not a measurement or hardware lower bound.

NCU on the unchanged native wide attention reports 246 registers/thread,
34,304 shared bytes/block, 12.47% achieved occupancy, 49.23% tensor-pipe
activity, 66.46% L1/shared data-path throughput and 4.57% HBM throughput.
SASS samples point to QK shared stores waiting for operands, shared-load/HMMA
dependencies and PV address instructions. Higher occupancy alone is not a
sufficient fix. Eight isolated paired candidates pass sampled FP32 references
but fail to establish a speedup; all remain outside the installed extension:

| Candidate | Native control ms | Candidate ms | Decision |
| --- | ---: | ---: | --- |
| Pretranspose V to column layout | 20.215809 | 21.629951 | Packing/zeroing loses |
| Warp/shared row maximum | 20.073471 | 20.227072 | No gain |
| Q32/K256 | 20.044800 | 24.824833 | Reuse/traffic regression |
| Unroll fixed QK loop | 20.204544 | 20.252672 | No gain |
| Unroll fixed PV loop | 19.892223 | 20.021248 | No gain |
| Eight warps | 20.045824 | 26.279936 | Register spills |
| Global-load cache policy cg | 20.264959 | 20.225023 | 0.2% noise |
| Q32/K128, four warps, three CTAs | 20.711424 | 26.468351 | Reuse regression |

A further artifact-only GEMM/reduce-scatter overlap probe uses zero persistent
cache and workspace, unlike the earlier independent 10-GiB-cache probe. Three
paired two-update runs give medians 5.872043 ->5.742729 s (2.20%). It preserves
useful FLOPs and reduces temporary memory, but has nonzero latent differences
and no complete-video quality validation. Extra packing, streams and GEMM-plan
APIs are not integrated for this modest prefix gain. Do not report 57.43 s as
measured full-20 denoise, or repeat these variants without a new hypothesis.

Validation: 70 targeted numerics/service/configuration tests pass. One actual
four-rank collective test passes on every rank, covering both native attention
backends, three valid/padded lengths, consecutive blocks, rank-dependent
weights, FP32 residual values above 65504 and poisoned padding. Pre-commit
checks pass. No CUDA source or installed binary changes in this follow-up;
prior attention sanitizer coverage is retained, not presented as a new run.
The actual full-20 run records all ranks' finite, mutually identical final
latents and unchanged FLOP counts.

Evidence root: `flashattention-pipeline50/` (exact jobs, NCU/SASS, both Nsight
traces, candidate source/binaries, negative results, tests, reports and NVML
curves). Fresh output:
`outputs/quality39-int8-flashattn-residual-20steps/FLASH_ATTN_V100/`.
Installed attention SHA256 remains
`bdb3dcf0eea8c23f992536182a06d26f7b61c7bb6ddefa4b3c88590b81ad0005`.

**Under 50 seconds, attention 60 TFLOPS, formal per-card 80 TFLOPS and human
quality acceptance remain incomplete.** Keep the arithmetic operand-reuse
focus; do not mistake 100% NVML utilization for Tensor Core saturation.

### Wide QK reuse and fused MLP preparation: 62.80 seconds

The retained native implementation completes the unchanged 1344x768,
39-frame/24-FPS, seed42, INT8 ConvRot FL2VA, TP4 GPU0-3 workload in
**62.804019 s for 20 actual updates** (21 sigma positions), or
**3.140201 s/update and 55.151783 useful model TFLOPS/card**. This is 9.06%
less time than 69.062335 s. Peak denoise Torch allocation falls from
6.647851467 to **6.566735744 GiB/card**; NVML peak device-used memory is
8.419433594 GiB/card. Persistent FP16 cache and Lt workspace remain zero.
This is one unprofiled development measurement after a one-call warmup with
prompt-verified cached text, not end-to-end or formal repeated acceptance.

Two changes address arithmetic operand reuse and intermediate data movement:

- Q64/K128 uses four 32x64 QK warps, reusing each Q fragment across twice as
  many keys. Direct batch/head indexing removes grouped pointer metadata and
  register pressure. Softmax distributes rows by actual `WarpCount::kCount`.
  The previous wide-QK failure incorrectly inferred eight warps from tile
  area despite launching four, leaving rows 32..63 unnormalized. This fixes
  the length-63 NaNs; the prior mapping-only diagnosis below is superseded.
  Two original 32x32 accumulator subfragments retain the probability store
  layout. Large sequences select K128; small refiners retain K64. The N12323,
  H14 specialization and V-load/softmax overlap use 246 registers/thread,
  34,304 shared bytes/block and no spills. There is no extra global workspace.
- INT8 MLPs fuse FP32 SiLU/product evaluation with power-of-two FP16 input
  preparation, avoiding the large FP32 intermediate write/read. A model-local
  row-parallel subclass accepts the explicit scale, preserves module/FLOP
  hooks and restores scale in FP32 before the ordinary TP sum. Releasing the
  gate/up buffer before projection avoids retaining an extra allocation.
  CPU, unquantized and FP32-input MLPs use the previous implementation.

The installed native attention's independently loaded, paired control is
**25.572351 ->20.110336 ms**, or **42.565744 ->54.126701 useful TFLOPS** at
B1/N12323/H14/D128. The earlier 19.772415 ms /55.051756 TFLOPS artifact
prototype is a separate build/run. A corrected Q128/K128 prototype passes
numerics but only improves 20.046848 ->19.749887 ms against the new native
path (1.48%); it is not enabled or treated as a full-model gain. The wider-M
64x32 warp spills 40 bytes in each direction and is slower, so it is rejected.

Quality provenance matters: changing K64 to K128 changes FP32 online reduction
and FP16 probability rounding. The native attention-only run (64.284648 s)
was freshly decoded; relative RMS changes against the previous video/audio
latents are 10.5741% /1.8484%. All automatic media checks pass. Eight sampled
frames show one red paper boat and one yellow duck, stable scene/color/count
and no obvious geometric corruption; this is not a full audiovisual score.
The subsequent SiLU fusion run (62.852804 s) and final buffer-release run
(62.804019 s) are both bitwise equal to the attention-only latents, so their
decode reuse is explicit. MP4 SHA256 is
`97b9196c49a8e1bf61cb8368f4db7d7013e99bc107650945dbe92b4b59b0184a`.
Human five-axis review remains pending; do not claim quality accepted.

Validation: 103 targeted attention/activation/INT8/numerics tests pass. New
coverage includes both key tiles, half-warp row boundaries, the CUDA grid-y
limit, hot-shape graph replay with changed values, fused activation scaling
and scale restoration before reduction. Long N73483 uses sampled FP32 rows,
never a full square reference allocation. K64 is bitwise equal to the frozen
V-prefetch primitive on all 16 checked lengths. Isolated CUDA12.8 memcheck,
racecheck and synccheck each pass 25 attention cases without errors/hazards;
graph replay is covered by ordinary pytest, not isolated sanitizer capture.
The previous full-runtime sanitizer loader issue remains documented below.

Two initial native control probes loaded both extensions under the same Python
module name, causing import caching to reuse the candidate as control. Their
timings and apparent bitwise failure are invalid. `native-verified-results.json`
uses distinct qualified module names and checks both function schemas before
comparison. Preserve the invalid logs; do not reuse them as evidence.

Evidence: `flashattention-qk50/` contains source prototypes, exact job JSON,
build/test/sanitizer logs, `report.json`, `REPORT.md`, `failed-paths.json`,
NVML curves and the sampled contact sheet. Final media and raw rank/NVML data:
`outputs/quality39-int8-flashattn-native-wide-silu-release-20steps/FLASH_ATTN_V100/`.
The measured attention binary is
`98b2e902208fae496b03ee5d66e9718931ccf056b4ae8d65eeb16d168c18f8a4`.
After repository formatting, the rebuilt binary is
`bdb3dcf0eea8c23f992536182a06d26f7b61c7bb6ddefa4b3c88590b81ad0005`;
its entire cuobjdump SASS output is byte-for-byte identical to the measured
binary (`formatted-build-comparison.json`). No new denoise timing is inferred
from the rebuild.
Rollback the combined change to kernel parent `f825756607db` and rebuild the
FlashAttention extension; `key_tile=64` is also available for primitive
numerical comparison. The existing `--int8-weight-layout row` independently
rolls back the column-major GEMM path.

**The under-50-second, attention-60-TFLOPS, formal per-card-80-TFLOPS and human
quality gates remain incomplete.** Next work should continue the measured
attention operand-reuse/shared-load/HMMA focus. Do not inflate useful FLOPs
with padding, rotations or dequantization, or use NVML's 100% median GPU
utilization as a Tensor Core saturation claim.

### Attention-focused diagnosis and V prefetch: 69.06 seconds

The user redirects the next optimization to GEMM/attention arithmetic and data
movement, with **60 useful TFLOPS for D128 attention** as the next operator
milestone. Preserve the under-50-second complete-denoise and >80 TFLOPS/card
model targets; none of these targets has passed.

A fresh two-update trace of the 69.92-second native baseline records 2.515795 s
of GEMM service (94.413296 useful TFLOPS) and 2.580667 s of attention service
(42.179368 useful TFLOPS) on the critical rank. These are service rates, not
complete-model throughput. The actual B1/N12323/H14/D128 noncausal attention
call has 1,088,506,166,272 useful FLOPs; 60 TFLOPS requires 18.141769 ms.

NCU on the same baseline binary finds 234 registers/thread, 26,128 shared
bytes/block, two resident blocks and 12.5% occupancy. Tensor pipe activity is
36.99%; HBM throughput is only 1.51%, while the L1/shared data path reaches
61.64%. SASS-correlated samples identify long-scoreboard consumers at the
QK/V shared stores after global loads, and short-scoreboard consumers at QK
HMMA instructions. This supports addressing operand movement and latency
hiding *inside* attention. It does not establish an HBM bandwidth limit.

The retained change loads the first V fragment before softmax and passes it
to a small adaptation of the existing SM70 PV software pipeline. Residual
masks, FP16 rounding, FP32 accumulation and math order are unchanged. It keeps
234 registers without spills, the same shared footprint and zero persistent
FP16 cache / zero Lt workspace. The native helper's paired operator median
is 24.522753 versus 25.702400 ms (44.387601 versus 42.350370 TFLOPS); clocks
vary, so this is a development microbenchmark, not formal model acceptance.

The new binary's independent NCU run reduces long-scoreboard stalls per
issued instruction from 0.646730 to 0.342021. Tensor pipe activity rises from
36.99% to 37.91%; short-scoreboard stalls remain. Do not interpret that stall
ratio change as an equivalent wall-time reduction. Remaining work should
address QK's shared-load/HMMA dependencies and input reuse.

The installed CMake build completes the unchanged 39-frame/20-update TP4
workload in **69.062335 s**, or 3.453117 s/update and 50.154018 useful
TFLOPS/card: a 1.23% reduction from 69.923404 s. All ranks retain exactly
6.647851467 GiB peak Torch allocation. Video/audio latents are bitwise equal;
reuse of the baseline decode is explicit, with MP4 SHA256
`17ac6de78b7bc280ce91a0c6ca018785d131b3798856ca3ea3cdc55a10811988`.
Automatic media checks inherit the identical baseline output; human scoring
is pending. The artifact prototype's 68.980006 s is a separate build/run.
Neither timing is end-to-end or the formal three-run 243-frame acceptance.

Eighteen native GPU tests pass, including newly added 32/33/96/97-key residual
boundaries, strided storage, poisoned padding, online rescaling and graph
replay. The standard CMake FlashAttention component builds. Binary SHA256 is
`540474cfad758d4328ad0e0f00c745be62d3d8caaebea1d4bc19b86354442b9f`.

CUDA12.8 isolated memcheck, racecheck and synccheck each pass thirteen cases
with zero errors/hazards. Full pytest under memcheck completed its seventeen
selected tests but reported a CUDA `cuKernelGetFunction` invalid-handle API
error during module loading, as in the earlier full-environment diagnostic.
Retain that failed log; the isolated run loads the exact installed extension
without importing the full vLLM runtime. Ordinary GPU graph replay is covered
by pytest; the isolated sanitizer harness does not test graph capture.

Rejected or paused probes, all outside production source:

| Probe | Paired candidate / control | Decision |
| --- | --- | --- |
| Manual QK shared fill | 44.529663 / 24.239103 ms | Exact but slower; retain original pipeline |
| QK internal K64 stage | 25.759745 / 25.328640 ms | Exact, no improvement |
| PV internal K64 stage | 26.507263 / 24.967169 ms | Exact, slower |
| QK first-tile persistence / next-K prefetch | 25.236481 / 25.414656 ms | Only 0.7%, registers 234 to 254; not retained |
| Q64/K128 with a 32x64 QK warp | NaN at length 63 in the earlier probe | Superseded by the actual-warp-count fix above |

At this earlier checkpoint, split-32 conversion still failed length 63.
The newer investigation above identifies and fixes the separate softmax warp-count
error; do not repeat the original variants unchanged. Earlier
Q128/K32 wide-PV variants corrupt the length-one output; their direct-store
workaround spills and slows down. Manual PV fill and larger Q/K tiles remain
rejected in the retained artifact records.

Evidence is under `flashattention-warp50/`: `report.json`,
`attention60-root-cause.json`, `native-steps.nsys-rep`, both NCU reports and
SASS-correlated samples, candidate source/build logs, native tests and quality
commands. Media, rank/phase CSVs and NVML curves are under
`outputs/quality39-int8-flashattn-native-prefetch-20steps/FLASH_ATTN_V100/`.
Rollback is the kernel parent `8cacbb70219c` plus a rebuild of
`_h3_flashattn_C`; keep the existing column-major INT8 path.

### Low-memory follow-up: 69.92 seconds, under-50 target incomplete

The user requires no substantial memory increase. Keep the persistent FP16
weight cache at zero; the development guard allows at most 1 GiB/card above
the prior 6.647851467 GiB denoise allocation peak. This allowance is a working
interpretation of the memory request, not a new user-specified numeric limit.

Two exact changes are now implemented:

- Specialize the softmax bounds check once per complete 64-key tile. The tail
  still uses valid-key masking; FP32 accumulation and exp2f are unchanged.
- Repack DiT INT8 weights into column-major physical storage during loading.
  Logical [N,K] coordinates, signed codes, channel scales and ConvRot remain
  unchanged. Per-call FP16 decode keeps this layout. A bounded host descriptor
  cache selects cuBLASLt algorithm 21 / tile 24 / split 1 / reduction 0 with
  **zero device workspace**. Missing plans fall back to the original row-major
  GEMM. No persistent FP16 cache is required. Roll back the layout and GEMM with
  `--int8-weight-layout row`.

| Fixed 39-frame / 20-update workload | Prior QK/RoPE baseline | Native low-memory route |
| --- | ---: | ---: |
| Complete synchronized denoise | 74.411059 s | 69.923404 s |
| Seconds / actual update | 3.720553 | 3.496170 |
| Effective useful TFLOPS / each rank | 46.548909 | 49.536398 |
| Peak allocated / each rank | 6.647851 GiB | 6.647851 GiB |
| Persistent FP16 cache / each rank | 0 | 0 |

The native route reduces denoise time by 6.03%; **50 seconds is not achieved**.
Video and audio latents are bitwise equal to the baseline, as are decoded
PCM samples and the MP4 (SHA256
`17ac6de78b7bc280ce91a0c6ca018785d131b3798856ca3ea3cdc55a10811988`).
A WAV container can differ while its PCM is identical. All automatic media
checks pass; five-axis human scoring remains pending. Driver-reported peak
used memory is 8.583496 GiB/card, which includes allocations outside Torch's
allocator. NVML busy percentage is diagnostic and is not Tensor Core FLOPS.

This is one unprofiled complete run after a one-invocation warmup, with cached
prompt-verified text. It is not end-to-end timing or the primary 243-frame,
three-measurement >80 TFLOPS/card acceptance. An artifact prototype measured
69.868952 s with the same zero-cache policy; do not combine different builds
into the formal three-run statistic.

Validation: 86 targeted GPU/CPU tests pass, including all signed INT8 codes,
scale/offset tails, all four padded-M12352 TP4 projections, forced fallback,
layout idempotence and CUDA Graph replay. Isolated CUDA12.8 memcheck, racecheck
and synccheck pass with zero errors/hazards for the new decode and Lt path.
The standard CMake W8A16 and FlashAttention components build successfully.

Evidence: `flashattention-lowmem50/report.json`, `production-binaries.json`,
`production-quality-job.json` and
`outputs/quality39-int8-flashattn-lowmem-native-20steps/FLASH_ATTN_V100/`.
GPU logs are in the serialized queue's `flashattention-under50/` directory.

Preserve failed or unselected experiments rather than repeat them unchanged:

- A 9.63-GB/card FP16 cache plus Lt measured 69.456427 s; it fails the latest
  memory policy and is not the forward configuration.
- Alternative cuBLAS row-layout algorithms and larger CUTLASS GEMM tiles were
  slower; some algorithm IDs also failed exact FP16-output checks.
- Row-layout GEMM/TP overlap improved standalone row projections only modestly
  and changed FP32 reduction rounding; it is not enabled.
- Q32/K64 attention measured 28.025 ms versus 27.575 ms control. A persistent
  shared-Q prototype measured 24.784 versus 25.083 ms with clock variation;
  this does not establish a useful complete-model gain. Neither is enabled.
- Manually staging the full PV value tile took 87.811 versus 23.785 ms;
  128-bit vector loads/stores reduced this to 31.587 versus 24.302 ms, still
  slower. Both matched control values but are rejected; retain these prototypes
  in `flashattention-lowmem50/attention-failed-paths.json`.
- Head-major and sequence-padding copies add storage traffic with no clear
  benefit. Approximate exp2 variants add no clear gain over the exact full-tile
  specialization and are not enabled.

Attention and serialized TP transfers remain the main opportunities alongside
GEMM. The prior trace shows less than 1% GPU idle time; allocating more weight
cache or only reducing Python launches cannot supply the remaining 19.9 seconds.

### Feeding diagnosis and exact QK/RoPE fusion follow-up

The current development target is **under 50 seconds for 20 actual updates**
at the unchanged 1344x768, 39-frame, seed42, INT8 ConvRot, TP4 GPU0-3 workload,
with output quality retained. The primary 243-frame >80 TFLOPS/card acceptance
is separate and remains incomplete.

Before QK/RoPE fusion, a four-rank Nsight Systems trace of the first two updates
from the unchanged 20-update schedule gives the following exclusive GPU wall
breakdown. The traced intervals include synchronized denoise boundaries; they
are profiling evidence, not unprofiled acceptance timing.

| Rank | Wall | Attention | GEMM | TP transfer + waiting | Other GPU work | No GPU activity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 7.744 s | 2.795 s | 2.716 s | 1.058 s | 1.125 s | 0.049 s |
| 1 | 7.744 s | 2.612 s | 2.651 s | 1.332 s | 1.108 s | 0.041 s |
| 2 | 7.744 s | 2.645 s | 2.641 s | 1.275 s | 1.128 s | 0.053 s |
| 3 | 7.744 s | 2.641 s | 2.655 s | 1.268 s | 1.131 s | 0.049 s |

Other GPU work includes normalization, conversion, ConvRot, INT8 decode and
copies. Rank0 performs 108,850,886,402,048 attention FLOPs and
237,524,454,107,136 linear FLOPs in these two updates. GEMM service throughput
is 87.46 TFLOPS and attention service throughput is 38.94 TFLOPS; neither is
complete-denoise throughput. Less than 1% is unoccupied GPU time. INT8 weight
decode is only 0.063 s, so weight caching and host launch optimization are not
the first targets. Faster ranks spend longer in NCCL: kernel duration includes
waiting for rank0 and must not all be attributed to network transfer.

With every other traced cost held constant, making attention take zero time
would yield only 69.99 TFLOPS for this short shape. This is an Amdahl illustration,
not a hardware limit or achieved result. The long primary shape has a different
compute/communication balance. NCCL auto versus forced Simple at real FP32
payloads gives approximately 5.44 ms for 264,993,792 bytes and 31 ms for
1,580,178,432 bytes; forcing the protocol is not promoted.

The retained fix fuses Q/K RMSNorm and 96-of-128-channel RoPE into two Triton
launches, replacing 32 PyTorch kernels per DiT attention block. Normalization
and both rotary products retain the reference FP16 rounding boundaries; FP32
arithmetic fusion is disabled. Unsupported shapes and autograd keep the
reference path. On N12323/H14, paired Q/K preparation drops from 3.44 ms to
0.30 ms, with bitwise-equal outputs at normal and high input amplitudes.

An unprofiled complete 20-update run now takes **74.411058897 s**, or
**3.720552945 s/update** and **46.548908603 useful TFLOPS/card**. The prior
77.342982236 s control uses the same attention/W8A16 binaries, prompt, seed,
weights and cache-off settings. The speedup is 3.94%; this is one development
measurement with a one-call warmup, not the formal three-run gate. DiT peak
allocation remains 6.647851467 GiB/card. Both final video/audio latents are
bitwise equal to the control. Fresh VAE decoding produces the identical MP4
SHA256 `17ac6de78b7bc280ce91a0c6ca018785d131b3798856ca3ea3cdc55a10811988`.
Automatic media checks pass; human audiovisual quality review is still pending.

Validation: 24 targeted tests pass, including 10 QK/RoPE cases for real length,
strided projection views, FP32 weights, high-range/zero inputs, rotary tails,
reference fallback and changed-input CUDA Graph replay. Three isolated
offset/tail cases pass CUDA12.8 memcheck, racecheck and synccheck. The first
model launcher failed during NCCL initialization with CUDA_MODULE_LOADING=EAGER,
before model inference. The successful model run uses the prior LAZY mode;
that single failure does not establish EAGER as its cause.

Rejected attention probes, each against an interleaved unchanged control:
direct fixed-D128 dispatch 27.189 vs 26.739 ms (209 registers); Q32/K128
30.757 vs 27.088 ms (198 registers); Q64/K128 30.810 vs 26.655 ms
(190 registers); warp-reduced max 27.237 vs 26.797 ms. All passed sampled
FP32 references; none is installed. Lower register count alone did not improve
throughput. Preserve these results instead of repeating unchanged tile sweeps.

Artifacts: `/data/minimax-h3/native-h3-20260908/flashattention-feeding-round2/`,
including `denoise-steps.nsys-rep`, its SQLite export, `breakdown.csv`,
`module-kernels.json`, `rootcause.json`, `denoise-breakdown.png`, prototype
sources/build logs, tests and sanitizers. The new media and NVML/phase reports
are under `outputs/quality39-int8-flashattn-rope-20steps/FLASH_ATTN_V100/`.
Reproduce the retained change with `run_rope_quality.py`; source and environment
are recorded in the task handoff. Roll back only QK/RoPE fusion by using
`qk_norm_rope_reference` at its dispatch point; attention and weights remain fixed.

GPU0-3 jobs were preempted under the user's explicit priority authorization.
One subsequent cleanup mistakenly targeted a GPU4-7 audit process before
checking its GPU UUID. This was outside that scope. The incident and recovery
are recorded in `preemption-followup.json`: the original supervisor relaunched
the same audit configuration with an independent recovery output label.
Always verify physical GPU UUIDs and process environment before any signal.

The new `_h3_flashattn_C` extension selects a dedicated CUTLASS SM70 fused
kernel under `FLASH_ATTN_V100`. Its QK tile is 64x64 and PV tile is 64x128;
all 128 output channels remain in FP32 registers across online-softmax updates.
It uses 128 threads, 232 registers without spills in the initial build, and
26,128 bytes of shared memory. Each MHA head is independent. There is no D256
head padding, global square score allocation, or FlashInfer delegation.
The wrapper supports masked query/key tails, storage-offset/strided inputs and
stream-local metadata. Provenance and BSD notices are in
`flash-attention-v100/kernel/h3/UPSTREAM.md`. Standard CMake, setuptools and
the source-only extension helper include the new operator.

The existing v37 QK/PV implementation was separately adapted to D128 with
FP32 scale and masked KV padding. At actual H3 N12323 it passes the sampled
FP32 reference and takes 34.279 ms against a 34.993 ms FlashInfer control.
The serial bounded-score prototype is slower than the fused path and is not
installed. Fused Q32/K64 and Q128/K64 candidates also pass numerical checks
but are slower than Q64/K64 in their respective matched comparisons.

An unprofiled GPU0 B1/H14/D128 comparison at actual valid H3 lengths measures:

| Tokens | New Flash-V100 | FlashInfer control | Generic Flash-V100 |
| --- | ---: | ---: | ---: |
| 12323 | 26.185728 ms | 35.336193 ms | 61.659138 ms |
| 73483 | 942.884888 ms | 1229.690918 ms | 2245.138428 ms |

Both lengths pass spread-out FP32 query references and whole-output finiteness.
These are operator measurements, not full-denoise acceptance. Clocks were
recorded, not locked: at the long length new Flash-V100 observed 1455-1462 MHz
and the controls 1530 MHz. Useful operator throughput is 41.57/41.05 TFLOPS;
the requested 80 TFLOPS/card model gate remains pending.

Validation: 73 video-suite tests passed on the initial JIT binary, including
14 new cases for tail masking, independent batches, rescaling, strided/offset
storage, native dispatch, graph replay and rejected inputs. The final standard
CMake binary SHA256 is
`84a41632bbc8e99459a938b90ca5981ded5600f336ce72b1e10e294613bc986a`.
Eight independent offset/tail/rescaling cases pass CUDA12.8 memcheck,
racecheck and synccheck with zero reported errors/hazards and
`CUDA_MODULE_LOADING=EAGER`. The broader pytest sanitizer harness reports
`cuKernelGetFunction` error 400 despite passing numerical assertions; the
instrumented graph/API issue remains open. It is not suppressed or counted
as a clean all-tests sanitizer result. A temporary diagnostic named `profile.py`
also shadowed Python's standard library; it was renamed `profile_native.py`.
No production Python dependency was changed to address that harness error.

Raw artifacts: `/data/minimax-h3/native-h3-20260908/flashattention-mainline/`.
They include all prototype sources/build logs, `bench-results.json`, GPU
ownership records, tests and sanitizer logs. The first ordinary-user NCU
attempt lacked GPU counter permission; it produced no counters. Complete
short-denoise measurements and the privileged profile are recorded below
when completed. Use `FLASHINFER_SM70` as the measured rollback path.

### Complete short-denoise and counter evidence

GPU0-3 INT8 FL2VA, 1344x768, 39 frames, seed42 and 20 actual updates completes
in **77.342982236 s** (3.867149112 s/update). Each rank performs
3,463,753,579,661,312 useful FLOPs and achieves **44.784329224 TFLOPS** against
the slowest rank's complete denoise time. The same retained W8A16 binary and
cache-off settings were used; the prior FlashInfer checkpoint took 86.200372206 s
and achieved 40.182582639 TFLOPS. The throughput gain is 11.4521%. This is one
unprofiled development measurement after a one-call warmup, with cached,
prompt-verified real text encoding; it is not the formal primary three-run gate.
DiT peak allocation is 6.647851467 GiB/card, not whole-pipeline memory acceptance.

The VAE was freshly loaded and decoded this candidate: load 35.935474 s,
decode 5.650908 s. All automatic media checks pass. The 39-frame MP4 SHA256 is
`17ac6de78b7bc280ce91a0c6ca018785d131b3798856ca3ea3cdc55a10811988`.
First/last frames show one red paper boat and one yellow duck; complete temporal
and audio review remains pending. Compared with FlashInfer, final video/audio
latents have relative L2 differences 0.107541/0.018019. They are not bitwise equal,
so no decode reuse or final quality equivalence is claimed.

The privileged NCU run records the actual `h3_flash_v100_d128` kernel:
Tensor pipe active 34.64%, 232 registers/thread, 26,128 bytes shared and maximum
active-warps fraction 12.5%. Long/short-scoreboard stalls are 15.43%/12.91%.
The profile used `--clock-control none` and is excluded from timing. Register
pressure and memory-latency hiding remain concrete optimization targets;
these counter percentages do not establish full-model throughput. NVML during
denoise reports mean GPU utilization above 99% but only 44.78 useful TFLOPS/card.

A post-profile register-limit probe requested three resident CTAs. It reduced
registers from 232 to 168 but introduced a 232-byte stack with 260/356 bytes of
spill stores/loads. Outputs remain bitwise equal to the new FlashAttention
control, but the exact N12323 operator slows from 26.385 to 47.712 ms. It is
rejected and retained as `reg3.cu`, `reg3-build.log` and `reg3-results.json`.
Do not repeat a register-cap-only change as an occupancy optimization.

The first two short-run launch attempts found another H3 task's GPU0-3 lease;
no timing/model run started. That independently owned FlashInfer task completed
before the FlashAttention run acquired the group. Its source and result were
not used as this change's matched baseline. No other job was terminated.
The initial privileged NCU wrapper hit Linux's protected `/tmp` lock-file
policy; the working wrapper holds the GPU lease as the user and elevates only
the profiling child. No global driver or power settings were changed.

Artifacts: `flashattention-mainline/report.json`, `native-ncu-summary.json`,
`native-ncu-privileged.ncu-rep`, `quality3.log`, and
`outputs/quality39-int8-flashattn-fused-20steps/FLASH_ATTN_V100/` under the task
artifact root. The output directory includes regenerated video/audio, latents,
rank/phase CSV, NVML JSONL and plots. **80 TFLOPS/card remains unmet.**

## Fixed scope

- Base: onecat/main 56f534e672657a6c7599afd6c0dcb2e2c211b2e3.
- Native model/pipeline/service; no vllm-omni runtime dependency.
- Model sources, licenses and revisions: see
  `vllm/model_executor/models/minimax_h3/UPSTREAM.md`.
- FL2VA and Ref2VA, original BF16 checkpoint and Comfy INT8 ConvRot AdaLN-pruned checkpoint.
- Four V100-SXM2-32GB, TP4 DiT/text encoder, native four-rank VAE tile parallel.
- FP16 compute with FP32 latent projections, timestep computation, pruned AdaLN and output heads.
- Acceptance: 1344x768, 243 frames, 24 FPS, seed 42, 50 sigma positions,
  no LoRA/step skipping/approximate cache; record actual DiT forwards.
- Each rank median useful model FLOPs / complete-denoise wall time >80 TFLOPS;
  warmup once, then three unprofiled measurements; denoise CV <=5%.
- Two attention backends must pass quality. Fastest qualified backend carries
  performance acceptance. BF16 checkpoint performance is reported separately.

## Implementation evidence

- Created isolated owned worktree and branch; canonical checkout left untouched.
- Native H3 packed layouts, reference processing, scheduler, transformer, Qwen3VL
  encoder and VAE adapter ports in progress. Omni framework, sequence-parallel,
  LoRA and approximate cache dependencies removed from native execution path.
- Signed INT8/FP32 scales, runtime QKV order, fused FFN shard loading and
  pruned AdaLN interpolation ported from open PR 6894.
- W8A16 currently has an explicit PyTorch numerical reference. This is not
  TurboMind kernel completion or a performance result.
- Flash-V100 path connected; FlashInfer D128 non-causal path must be implemented
  before backend support or qualification is claimed.

## Environment and failed paths

- Owned Python 3.12 environment uses Torch 2.10.0+cu128, Transformers 5.15.1,
  Diffusers 0.40.0. System nvcc is 12.0; owned CUDA 12.8 toolkit pending.
- Initial dependency resolution tried to upgrade Torch/CUDA; interrupted before
  installation and used pinned packages without replacing the shared environment.
- Bootstrap vLLM binaries are read-only links from the local 1Cat 1.5.0 environment;
  hashes are recorded in task artifacts. They are not a new source-build result.
- Upstream modulation and QK/RoPE kernels contain explicit BF16 roundings. These
  require FP16 adaptation and numerical tests on SM70 before performance use.

## Remaining gates

1. Targeted format, packing, TP/shard, FP16 numeric, padding and reference tests.
2. Native serial CLI/API, dependency installation, media export and job lifecycle.
3. TurboMind signed W8A16, bounded cache and real FlashInfer-SM70 non-causal D128.
4. Actual checkpoint loading and both-partition GPU functional generation.
5. Full video quality, fixed-shape performance measurements and profiler evidence.
6. Three review scopes/Draft PRs with signed commits and reproducible artifacts.

Every failed GPU experiment must record configuration, result and the resulting
implementation decision here or in linked benchmark evidence. Do not repeat an
unchanged experiment. No acceptance result may be inferred from route imports,
GPU utilization, synthetic operator peaks, or estimated hardware capabilities.

## 2026-09-08 native and operator checkpoint

- `tests/video`: 14 passed, including signed INT8/row-scale restoration,
  Kronecker ConvRot256/inverse, original QKV reorder, curve interpolation,
  partition metadata, 243-frame/49-forward schedule, poisoned padding,
  both non-causal attention backends on V100, W8A16 FP32 reduction, encoder
  causal GQA, and serial HTTP job lifecycle.
- Synthetic two-block INT8 curve DiT: TP1 versus TP4 on GPU 0–3 passed on every
  rank. Repeated after replacing reference linear execution with native W8A16;
  all four ranks passed (atol 0.0005, rtol 0.02). This is a correctness test,
  not full-model or performance acceptance.
- Native W8A16 decodes signed INT8 with original FP32 per-row scales, rotates
  activations in FP32 with FP16 output and calls cuBLAS FP16 GEMM from the
  TurboMind SM70 operator module. Uses explicit FP32 compute and disables
  reduced-precision reductions. Initial cuBLAS math-mode comparison failed;
  explicitly disabling reduced-precision reductions fixed the mismatch.
- FlashInfer-SM70 now owns a D128 non-causal online-softmax WMMA operator using
  the repository's Volta QK/PV primitives. Lengths 1,17,63,64,65,243,1025 passed
  the FP32 reference (largest observed absolute error 0.0009765625). No
  Flash-V100 delegation occurs in this route. Large-sequence optimization and
  full-video quality remain pending.
- CLI `vllm video generate/serve`, serial worker engine, HTTP jobs, optional
  video dependencies, fixed-list FP16 cache, useful-FLOP hooks, NVML sampling,
  media export and automated video checks are implemented. Runtime validation
  with actual full checkpoints remains pending while weights download.
- CUDA Toolkit 12.8.93 nvcc installed in the task artifact directory; independent
  extension builds passed against Torch 2.10.0+cu128. Wheel targets added, but
  complete wheel build has not yet been validated.
- Nsight Compute on GPU 0 returned `ERR_NVGPUCTRPERM`. No counters were collected.
  Do not claim Tensor Core activity/occupancy from this run. Nsight Systems
  tracing is being checked separately. No global driver settings were changed.

Raw evidence (not committed): task artifact directory
`/data/minimax-h3/native-h3-20260908/`, including `native-tests.log`,
`tp1-smoke.log`, `tp4-smoke.log`, `tp4-w8a16-smoke.log`,
`build-owned-extensions.log`, `flashinfer-numerics.log`, and `profiles/`.

### Actual components and loader recovery

The first real TP4 Qwen3VL encode exposed a missing optional residual argument
in the native RMSNorm adapter. Restored the reference residual-add/FP16 rounding
contract and added a focused regression; full TP4 encode is being repeated to
validate that fix. The VAE component checks produced finite 22-frame 64x64 video
(15,278,413,824 peak allocated bytes on GPU 0) and finite 32-kHz mono audio. T=2
was rejected by the VAE streaming decoder; the valid component smoke uses T=7.
The original text-encoder shard 10 stalled in Xet reconstruction. Recovered it
through HTTP, verified SHA256 `aded5a4d1d5e22dbd8b6f79266b6eb88c840411b09527c53917a1419ace22e2f`,
and stopped only the owned stalled downloader. INT8 downloads continue separately.
Nsight Systems 2025.1 CLI produced a usable trace; the W8A16 projection launched
`cutlass_70_tensorop_f16_s884gemm_relu_f16_128x128_tn_align8`. This confirms the
Tensor Core kernel route, not measured Tensor Core activity or full-denoise speed.

The next real-encoder check exposed an integration mismatch: Omni's encoder
ignored the return value of `group.all_reduce`, while native GroupCoordinator
can return a new tensor. Fixed both embedding and row-parallel reductions and
added a functional-collective regression. All four real TP4 encoder ranks now
have finite layer-50 output with identical amax 16088. The diagnostic comparison
initially attempted NCCL broadcast on the encoder's CPU result; the test is
being corrected to broadcast a CUDA copy before checking all-rank identity.

### Real INT8 FP16-range fixes (2026-09-08)

- The corrected text-encoder comparison passed bitwise identity across all
  four TP ranks, with finite layer-50 FP16 output. The full FL2VA INT8 file and
  Ref2VA INT8 file are downloaded. Original BF16 DiT shards are still downloading.
- Native explicit width/height requests previously failed Omni's separate
  aspect-ratio requirement. The native canvas now supplies that ratio; two
  canvas regressions pass, including the exact 1344x768 primary shape.
- The first real single-step pipeline reached DiT, then rejected non-finite
  velocity. Layer diagnostics found condition projection values up to 76615.5,
  block residuals above four million, and overflowing row projections and
  gated MLP products. FP16 accumulation settings alone cannot represent these
  values. Preserve condition projection, residuals and gated products in FP32.
  Normalize and convert attention/MLP GEMM inputs to FP16; use exact power-of-two
  row scaling for wide MLP activations, restoring scale in FP32 GEMM output.
  INT8 weights, FP32 checkpoint scales and ConvRot256 remain intact.
- A real 50-block INT8 DiT forward at the 256x256 diagnostic shape now has finite
  output on all four ranks (video amax 10.8243, audio amax 4.36589). This is one
  forward, not a complete schedule, quality pass or performance result.
- The current native suite passes 23 tests, including actual CUDA checks for
  FP32 residuals, above-FP16-range Tensor Core outputs, restored activation
  scales, signed INT8, padding and alias-preserving CPU/GPU staging.
- Pinned staging now replaces one host allocation at a time. The old snapshot
  retained all pageable weights while building the entire pinned copy, raising
  transient TP4 host memory and swap pressure.
- Nsight Compute succeeded with the user's sudo authorization. The earlier
  ERR_NVGPUCTRPERM result is superseded: the isolated FP16-output GEMM recorded
  462422016 Tensor Core instructions and 86.177% active tensor-pipe cycles.
  The new FP32-output variant needs its own profile. Complete-denoise >80 TFLOPS,
  main-shape memory, full-video quality and both-backend quality remain pending.
- Review scopes: native #557, kernels #558 (stacked). The next end-to-end run
  initially found both GPU groups occupied by other vLLM workers; the owned
  launcher waits for a free group without changing those processes.

Evidence: `encoder-real-tp4-final.log`, `pipeline-route-canvas.log`,
`dit-nan-diagnostic.log`, `dit-nan-fp32-condition.log`,
`dit-nan-fp32-residual.log`, `dit-nan-fp32-output.log`,
`dit-nan-fp32-gated.log`, `native-tests-fp32-islands.log`,
`kernel-build-fp32-output.log`, and `profiles/w8a16-hmma-root.ncu-rep`
under the task artifact directory. Do not repeat superseded failing routes
unless a new change requires them.

### Short development workloads and attention evidence (2026-09-08)

- User requested 1–2 second development clips and no repeated full-video runs.
  Native requests now accept 22 and 39 aligned frames; 39 frames at 24 FPS is
  1.625 seconds. The primary 243-frame/49-forward acceptance contract is unchanged.
  Ten targeted shape/schedule tests pass, including short audio latent alignment
  and rejection below the streaming VAE's minimum temporal chunk.
- The earlier primary run was interrupted before completion; the host rebooted.
  Its log reaches 22/49 DiT calls. Preserve it as partial diagnostic evidence,
  never as a completed run or acceptance measurement. Initial stable calls took
  about 129.5 seconds; late slow calls are not a reproducible speed baseline.
- NVML medians were 100% GPU utilization. Nsight Compute separately measured
  13.886% tensor-pipe activity in main-shape Flash-V100 attention and 78.150% in
  the FP16-input/FP32-output Tensor Core GEMM. Utilization is not useful TFLOPS.
  Attention at 73483 tokens took about 2.262 seconds without profiler, explaining
  most of the observed DiT time. The performance gate remains unqualified.
- Native TP4 video VAE decoded 39 frames at 1344x768 using 28 tiles. All four
  outputs were finite with the expected shape; each peak allocation was
  17805820416 bytes (16.58 GiB). Decode took 4.62–5.50 seconds per rank, without
  a synchronized performance protocol; this is a functional result only.
- CMake configure/build/install succeeded for both H3 extension targets. Five
  targeted CUDA numerical tests passed against those installed modules. This
  does not yet establish a complete release-wheel build.
- All fixed-revision original and Comfy weights have downloaded and checksum
  manifests are retained. Host memory is shared with another service, so current
  short denoise diagnostics reuse the previously validated TP4 text embeddings
  and load the VAE after releasing DiT, rather than retaining every component.
  Such runs must explicitly report cached text and separate loading costs.

Evidence under `/data/minimax-h3/native-h3-20260908/`:
`short-clip-contract-tests.log`, `vae-tp4-short.log`,
`cmake-numerics-tests.log`, `interrupted-baseline-summary.json`,
`original-checkpoint-manifest.json`, `comfy-checkpoint-manifest.json`,
`profiles/flashv100-attention.summary.json`, and
`profiles/w8a16-fp32-output.summary.json`.

### Independent FlashInfer register accumulation

The selected SM70 kernel keeps QK scores, online-softmax reductions and PV
accumulators in registers. Only cross-warp row partials and probabilities use
shared memory. A 128-query by 32-key tile uses 512 threads and 64 KiB shared
memory. The explicit backend continues to execute its own Volta WMMA kernel.

- Full FP32 checks at 17, 243, 1025 and 8192 tokens passed. Spread-out query
  checks at 12323 and 73483 tokens passed without a square attention matrix.
  Added regression coverage for batch indexing, 127/128/129 query boundaries,
  short-video token count and changing softmax maxima across key tiles.
- The final formatted implementation passes all 38 native video tests on GPU 0.
- At 12323 tokens (39-frame canvas), candidate median attention time was
  53.220 ms versus 61.108 ms for Flash-V100. At 73483 tokens the same isolated
  comparison was 1.857 seconds versus 2.246 seconds. No full long video was run.
- Cached-text TP4 INT8 denoise at 1344x768, 39 frames, seed 42 and three sigma
  positions (two forwards), after a one-forward warmup: Flash-V100 11.29494 s,
  FlashInfer candidate 10.55566 s. Each rank counted 346375340509184 useful
  FLOPs, giving 30.6664 and 32.8142 useful TFLOPS respectively. All four ranks
  matched bitwise within each backend and all latents were finite.
  This is development timing, not the full-schedule >80 TFLOPS acceptance.
- Rejected candidates: 128x64 requested too many launch resources; 32x64 was
  slower; keeping only PV in registers was slower than also reducing softmax
  in registers. Their code/binary and raw measurements remain in artifacts,
  while the public extension retains only the selected implementation.
- The first cached-text diagnostic attempted to export the raw BCTHW VAE
  tensor. Its script omitted the pipeline's output conversion to BTHWC RGB8.
  The native pipeline already performs that conversion; correct the diagnostic
  and save latents before export to avoid repeating denoise on export failures.

Artifacts: `attention-short-candidates.json`, `flashinfer-tested-candidates.cu`,
`flashinfer-tested-candidates.so`, `flashinfer-register-formatted-build.log`,
`kernel-register-native-tests.log`, `denoise-short-int8.log` and
`outputs/dev39-int8/{FLASH_ATTN_V100,FLASHINFER_SM70}/` under the task artifact
directory. Those output directories contain timing and NVML curves; the initial
export failed and must not be presented as generated-video quality evidence.

### Short-video export and GPU ownership diagnostics

The register-softmax FlashInfer implementation in kernel draft #558 passes 38
native numerical/service tests. Its cached-text INT8 TP4 development run exported
a 1344x768, 39-frame, 24-FPS video with 1.632 seconds of decoded audio. Automatic
frame/dimension/audio/finite/black/static checks pass. Visual inspection of the
first screenshot shows pronounced ghosting and grid artifacts; the two-forward
schedule is an execution check and is not a quality pass.

The first export retry encountered other workers taking 27 GiB per GPU and ran
out of memory. A subsequent diagnostic shell performed a preflight but continued
after Python returned an error; it ran beside other workers using about 13 GiB
each. That run's timing is explicitly invalidated in its JSON. The native engine
already propagates selection errors; the separate diagnostic launcher now also
propagates them and checks external GPU processes before and after denoise.
NVML records now include compute PIDs and their memory to expose interference.
A 12-sample read-only telemetry check passed on the live GPU inventory.

Retained evidence: `outputs/dev39-int8-final/FLASHINFER_SM70/` (video, original
audio, latents, frames and invalidated timing), `denoise-short-int8-final.log`
(OOM), `denoise-short-int8-final-retry.log` (export passed, timing excluded),
`run_short_guarded.py`, and `nvml-process-record-smoke.jsonl` in the artifact
directory. As of this checkpoint GPU groups 0–3 and 4–7 have other vLLM workers.
No owned long-video or waiting-GPU process is being retained. BF16 and reference
generation checks still need an available four-GPU group.

### Twenty-step short-clip quality check requested

The user prioritizes normal output quality over further throughput tuning and
requested a 20-step check. The diagnostic now fixes 39 frames at 1344x768,
24 FPS, seed 42, the same INT8 checkpoint and FlashInfer implementation, and
20 actual DiT calls (`num_inference_steps=21` under the reference sigma-point
convention). Video/audio shifts remain 12/3, Turbo LoRA is off, and FP16 weight
cache is off. Reuse the verified text conditioning for the same prompt to keep
the comparison focused on the changed step count. Assert the completed call
count and retain latents before export.

At preparation time both GPU groups were occupied. A later idle GPU snapshot
was still covered by another task's group lease. Capacity-only preflight can
race such a task during service replacement. H3 now acquires the shared 1Cat
per-card/group locks, checks capacity again, and retains the lease until its
workers have exited. Failed acquisition/startup releases its own partial locks.
The quality diagnostic uses the same lease and propagates launch errors. Ten
CPU lease/service tests pass, including partial-lock rollback, whole-group
fallback and worker-startup failure. No unrelated process was stopped.

The 20-step GPU/video result is pending; do not infer that low step count is
the sole cause of the two-step clip's artifacts. Prepared contract and commands
are retained as `quality39-20steps-contract.json`, `denoise_quality.py`,
`run_quality_guarded.py`, and `quality39-20steps-launch.log` in the task artifact
directory. `gpu-lease-tests.log` records the focused regression result.

### GPU 0–3 priority authorization

The user explicitly prioritizes this H3 task on GPU 0–3 and authorizes stopping
conflicting jobs there. This authorization persists across turns; do not ask
again for the same GPU allocation. GPU 4–7 services remain outside that scope.
After checking PID/command identity and preserving its active-job metadata,
stopped the conflicting quasar lease scheduler PID 17903 with SIGTERM. Its
GPU child had already exited; no model/log/queue files were deleted. Acquired
the shared 0–3 lease for H3 and started the fixed 20-forward quality check.
The CPU text conditioning, INT8 weight encoding, shift rules and FP16 cache
settings match the earlier short clip. Keep quality as pending until the video
has decoded and been reviewed.

The interruption record is `gpu03-priority-handoff.json`; the active test log is
`quality39-20steps-run.log` in the retained artifact directory.

### Completed twenty-forward INT8 short quality comparison

Both backends completed the fixed 1344x768, 39-frame, 24-FPS, seed-42 request
with 20 actual denoise calls, after one single-call warmup. Original INT8/FP32
scale and ConvRot data, cached verified text conditioning, sigma shifts 12/3
and cache-off settings were retained. Each GPU had only its corresponding
H3 rank in the recorded NVML compute-process samples.

| Backend | Complete denoise | Seconds/call | Useful TFLOPS/rank |
| --- | ---: | ---: | ---: |
| FLASH_ATTN_V100 | 113.459387 s | 5.672969 | 30.5286 |
| FLASHINFER_SM70 | 105.642574 s | 5.282129 | 32.7875 |

All video/audio latents are finite and bitwise identical between TP ranks within
each backend. Both exported videos pass full decoding, 39-frame dimensions/FPS,
valid finite audio duration, no-black and no-prolonged-static checks. Inspecting
frames 0/19/38 for FlashInfer and 0/38 for Flash-V100 shows a clear red paper
boat, yellow duck, reflections and stable foliage; the broad grid and ghosting
from the two-call sample are absent. The controlled step-count comparison
supports undersampling as the main cause of those severe artifacts. It does
not prove universal quality or complete the primary-video acceptance gate.

Backend outputs are similar, not identical: decoded RGB PSNR over all 39 frames
is 30.888 dB; final video/audio latent relative L2 differences are 0.0777004 and
0.0153296. These numbers are auxiliary, not quality acceptance criteria. Audio
is valid 32-kHz stereo; semantic listening review is still pending, as is the
user's five-axis final review. No 80 TFLOPS qualification is claimed.

Evidence: `outputs/quality39-int8-20steps/` contains both videos, original audio,
latents, NVML samples/curves, rank/phase CSV, automated checks, visual-review
notes and backend comparison JSON. `quality-two-vs-twenty.png` compares the
same-seed two- and twenty-call outputs. Raw logs are
`quality39-20steps-run.log` and `quality39-20steps-flashv100.log`.

### Original-checkpoint short quality and FlashInfer mainline profiling

The original BF16 checkpoint also completed FL2VA text-to-video at 1344x768,
39 frames, seed 42 and 20 updates through Flash-V100, using FP16 matrix inputs
and FP32 sensitive intermediates on V100. Complete denoise took 110.569167 s
(5.528458 s/update), with 31.328872 useful TFLOPS per rank. DiT-only peak Torch
allocation was 17.186255 GiB; this does not include a whole-pipeline memory peak.
All automatic media checks pass and inspected first/last frames show a clear
boat and duck without the severe two-update artifacts. Human audio review,
Ref2VA/reference generation and primary acceptance remain pending. Evidence is
`outputs/quality39-original-20steps/FLASH_ATTN_V100/` in the artifact directory.

The user subsequently fixed FlashInfer-SM70 as the optimization mainline and
reaffirmed >80 TFLOPS/card. The CLI/config default now follows that choice.
A four-rank Nsight Systems trace captures the first two updates from the
unchanged 20-update schedule, with cached verified text, 39 frames and cache
off. It truncates the schedule for profiling and makes no quality or acceptance
claim. The rank-0 synchronized denoise span is 10.600430 s:

| Exclusive wall category | Two-update seconds |
| --- | ---: |
| FlashInfer attention | 5.466013 |
| Model GEMM | 2.692784 |
| TP communication | 1.088080 |
| Other GPU kernels | 1.011361 |
| ConvRot | 0.218711 |
| Weight dequantization | 0.061064 |
| Copies | 0.012464 |
| No recorded GPU activity | 0.049953 |

The parser keeps kernel service and exclusive wall coverage separate, including
an overlap category if present. This trace prioritizes attention and then TP
communication over weight caching or launch-overhead tuning. It must not be
substituted for the unprofiled 20-update result or primary three-run gate.

An initial warp-owned-query prototype is rejected: all sampled FP32 references
pass, but its Q64/K32, Q64/K64 and Q128/K32 variants take about 89.36, 87.88 and
59.46 ms at 12323 tokens, versus 53.27 ms for the retained kernel. Registers rise
to 198/230 per thread. Do not repeat those unchanged variants. A transposed V
layout and software-prefetch follow-up are being evaluated separately.

Evidence: `profile_h3_steps.py`, `run_profile_h3_steps.py`,
`profiles/h3-flashinfer-mainline-steps.nsys-rep`, the matching SQLite,
`profiles/h3-flashinfer-step-breakdown.json`, `query-owned-results.json`, and
`flashv100-60t-route-audit.md`. GPU 0–3 priority preemption is authorized; the
active development lease is recorded in `flashinfer-development-lease.json`.

### FlashInfer V-layout and K/V-prefetch checkpoint

Retained the 128x32 register-softmax layout and moved V into column-major
shared storage for PV. Vectorized loads fetch the next K/V tile while the
current PV tile computes; the next iteration joins before consuming it.
Contiguous storage-offset views retain a scalar load path when their K/V
pointers are not 16-byte aligned. The launch remains 512 threads, 128
registers/thread, with no register spills.

After warmup, three ABBA groups at each shape give these prototype operator
medians (padding excluded from useful FLOPs):

| Length | Previous kernel | Prefetch candidate |
| --- | ---: | ---: |
| 12323 | 53.202433 ms | 38.789122 ms |
| 73483 | 1859.025452 ms | 1354.961914 ms |

The final source also handles unaligned storage and passes all 47 video tests.
CUDA 12.8 Compute Sanitizer memcheck and synccheck each pass all six new tail
and unaligned-view cases with zero errors. The system sanitizer was incomplete
and could not find its injection library; the verified CUDA 12.8 redistributable
was used instead. The ordinary CMake H3 FlashInfer target builds successfully.

The final binary SHA256 is
`67d542efaaf9bf3fa4e360bfe4c32e9537832a239d0d12b5488460cf9bf464c1`.
With this binary, the same seed-42/39-frame/20-update cached-text INT8 request
completes denoise in 90.880784 s, versus 105.642574 s before: 4.544039 s/update
and 38.113157 useful TFLOPS per rank. Both video and audio latents are bitwise
identical to the earlier register-softmax result. Automatic export checks pass;
the MP4 SHA256 is also identical:
`db77e1ab34999a1727b962edc77d42780b203c4cd23a835e6e05540956c6a078`.
DiT-only peak allocation remains 6.647851 GiB/rank. This is one short diagnostic,
not primary quality or the three-run >80 TFLOPS qualification.

Matched-shape NCU reports tensor-pipe activity rising from 16.976% to 22.950%,
and long-scoreboard stalls falling from 28.329% to 0.531%. Shared load conflicts
fall from 1.208 billion to 0.671 billion, while shared store conflicts rise to
1.105 billion. The next experiment targets the transpose stores; no further
speedup is assumed. NCU durations/counters remain separate from formal timing.

Evidence: `prefetch-long-results.json`, `prefetch-final-tests.log`,
`prefetch-memcheck-12.8.log`, `prefetch-synccheck.log`,
`prefetch-cmake-build.log`, `profiles/flashinfer-prefetch-short.*`, and
`outputs/quality39-int8-prefetch-20steps/`. The previous binary is retained as
`flashinfer-register-baseline.so` for exact rollback/AB comparison.

### Selected cooperative V transpose

The selected follow-up reuses the consumed probability tile as temporary
row-major V storage, synchronizes, and writes the column-major V tile
cooperatively. This removes the direct strided-store pattern without changing
floating-point arithmetic. The final binary SHA256 is
`339ceba5f296faba8c4a13a0ebed7788a5f2f795ab39d7a5c6c3d3353dbb0186`;
compilation uses 127 registers/thread and reports no spills.

At 12323 tokens with observed SM clocks at 1530 MHz throughout the timed
comparison, medians are 38.209343 ms for direct prefetch stores, 34.983711 ms
for cooperative transpose and 35.778271 ms for an additional XOR-swizzle
candidate. Outputs are bitwise equal; reject the slower extra-swizzle variant.
An earlier warmed long-shape bracket at 73483 tokens gives 1334.623779 versus
1223.218689 ms for direct versus cooperative stores. These remain isolated
operator results and are not used as model acceptance figures.

The same complete 20-update INT8 short request takes 87.824099 s
(4.391205 s/update), giving 39.439671 useful TFLOPS on every rank. Both final
latent tensors are bitwise equal to the prior validated video/audio tensors.
The diagnostic therefore reuses the unchanged decoder's prior output only
after verifying both equalities and the MP4 hash. Its explicit mode is
`short_clip_denoise_quality_with_reused_decode`; no fresh VAE or end-to-end
service timing is claimed. The original register-softmax baseline was
105.642574 s and 32.787478 TFLOPS/rank. The >80 gate remains unmet.

All 47 video tests pass with the final binary. CUDA12.8 memcheck, racecheck and
synccheck each pass all six tail/unaligned cases, with zero errors/hazards.
The standard CMake component build passes. Final matched-shape NCU measures
25.091% tensor-pipe activity, 0.580% long-scoreboard stalls, 0.872 billion
shared-load conflicts and 0.101 billion shared-store conflicts. The remaining
25% theoretical active-warp limit, shared-load stalls and matrix scheduling
need further attention; the previous full-step trace also records nontrivial
TP communication. Hardware counters are not full-model TFLOPS.

Evidence: `transpose-candidates-warm.json`, `cooperative-long-results.json`,
`cooperative-final-tests.log`, `cooperative-{memcheck,racecheck,synccheck}.log`,
`cooperative-cmake-build.log`, `profiles/flashinfer-cooperative-short.*`, and
`outputs/quality39-int8-cooperative-20steps/`. The prior accepted prefetch
binary is retained as `flashinfer-prefetch-baseline.so`.

## FlashInfer mainline: conversion and ConvRot follow-up

The retained W8A16 change replaces block-wide ConvRot256 shared-memory
butterflies with warp shuffles and register permutations. FP32 intermediates,
radix-four addition order and the final FP16 rounding stay intact. A separate
CUDA operator fuses FP32 row maximum, exact power-of-two scaling and FP16
conversion before GEMM. CPU reference behavior remains available.

On a synthetic `[12323,7168]` matrix, ConvRot decreases from 1.054720 to
0.571392 ms and FP16 preparation from 2.870272 to 1.302528 ms. Both conversions
and row scales are bitwise equal to their controls. These operator results
are distinct from the real H3 layer dimensions and model timing below.

The final W8A16 binary SHA256 is
`c7eb5d619eb6a3a5e260acae04f17e7f88a172649717ad26478a5a9d6d6e9a1b`.
The FlashInfer attention binary stays at the cooperative-prefetch revision
above. The same 39-frame/20-update INT8 TP4 request completes denoise in
86.200372 s (4.310019 s/update), or 40.182583 useful TFLOPS on each rank.
Video/audio latents are bitwise equal to the previous result; decoder reuse
is explicitly recorded. This single short run is still below the primary
>80 TFLOPS/card gate. DiT-only peak remains 6.647851 GiB.

Validation: 58 video tests pass, plus the subsequently added nonfinite-input
test passes separately. CUDA12.8 memcheck, racecheck and synccheck each pass
the eleven new rotation/scaling cases, including grid-stride boundaries.
The standard CMake W8A16 component builds. Evidence is under
`feeding-round2/` and `outputs/quality39-int8-fusedprep-20steps/`.

Rejected attention controls at D128/N12323: expanding BK to 64 spills
registers and takes 58.25 ms; limiting unrolling lowers that to 38.63 ms but
still loses to the 34.99 ms baseline. BK32 without unrolling takes 40.46 ms.
Halving BQ to 64 gives 41.07 ms (BK32) or 45.72 ms (BK64). Explicit P-fragment
reuse gives no improvement; skipping unit output rescaling gives only about
1% and is not retained. CUTLASS FMHA controls take about 30 ms, with different
rounding, and are not installed or counted as FlashInfer results. An asymmetric
QK/PV CUTLASS control compiles but remains unmeasured after the user's ComfyUI
audit request. Preserve these controls rather than repeat them unchanged.

## Recovered ComfyUI V100 source audit

The user identified unmounted local disks as the likely source location.
Read-only inspection recovered a ComfyUI/Raylight tree containing
`flash-attention-v100-1cat-0.0.5`. Its dense source SHA256 is
`82e4a09dda874034bf0ca76482d670fe22676315e9143232f63c742dc475683f`.
The recovered D128 route uses Q32/K176, 512 threads and WMMA. CUDA12.8 emits
64 registers/thread with spills. The original files remain unmodified.

An unaligned 65-token request hangs: a block barrier is inside a conditional
that excludes invalid query-row threads. Padding Q alone avoids that hang
but still produces NaNs with 65 valid K/V tokens. Do not use this recovered
kernel directly for H3's unaligned lengths or infer quality from its timing.

Fully aligned controls isolate the old kernel's performance. At B1/H14/D128,
noncausal FP16 N12320, current FlashInfer takes 34.900993 ms versus recovered
ComfyUI's 67.394562 ms; at N73568, 1224.818726 versus 2334.326904 ms. All
outputs are finite and sampled-row FP32 references pass. Three alternating
measurements after warmup observed 1530 MHz throughout. These aligned lengths
differ from H3's actual 12323/73483 and are only operator controls. Their
effective attention rates are approximately 31.2/31.7 TFLOPS for FlashInfer
versus 16.1/16.6 for recovered ComfyUI, not full-model acceptance figures.

The old Raylight documentation's large cold/hot speedup includes worker/model
startup. Its archived logs also show PyTorch/xformers routes, while LTX TP
can call ComfyUI's selected attention directly. A workflow's FLASH_ATTN label
alone is insufficient route evidence. The audit does not justify replacing
the current H3 FlashInfer mainline. Source snapshots, licenses, reproduction
scripts, failures and results are retained in `comfy-audit/`.

### Official workflows and LightX2V Turbo expansion (2026-09-08)

Owned branch: `codex/v100-h3-workflows-lora-20260908-091941`.
Stack base: native PR #557 at `1d201f41344f1a9a50d91197a8ad3a5525e51190`;
integration remains `onecat/main` (observed `e5d63c51f0fcc1ddf75d229e3df06bf52df206f5`).
This scope adds workflow/LoRA execution and does not change the other tasks'
attention kernels. [Omni coverage](../omni_workflow_coverage.md) records the
90-row upstream support inventory, relevant task history and remaining families.

- Added complete-layout LightX2V Diffusers Turbo loading for FL2V and Ref2V,
  four/eight updates, 544p/768p training variants, metadata alpha and per-artifact
  modality shifts. Every A/B tensor must be consumed; ambiguous directories,
  wrong partitions, other export layouts and malformed shapes are rejected.
- TP-local Q/K/V delta slices and reordered MLP gate/value rows use the existing
  FP16-input/FP32-output GEMM with range scaling. INT8 ConvRot base weights stay
  unchanged; LoRA consumes unrotated activations. Buffers join pinned staging.
- CLI and HTTP expose task, flow shifts, reference-video offsets and request
  LoRA scale. Omitted sampling values come from the adapter; explicit mismatches
  fail before dispatch. Scale zero restores base defaults and bypasses deltas.
- Reference audio/video metadata is validated before queueing using the same
  source checks as preprocessing. The service remains healthy after rejected
  media. LoRA GEMM work is counted separately from base work.

Validation environment: Python 3.12.13, Torch 2.10.0+cu128, CUDA 12.8,
Transformers 5.15.1, Diffusers 0.40.0, V100-SXM2-32GB TP4, GPU0-3.
Signed Comfy INT8 and FP32 scales, AdaLN pruning and ConvRot256 retained;
FP16 weight cache and approximate caches off. Denoising uses the copied,
hash-recorded #558 FlashInfer-SM70 development binary; text encoder keeps its
causal Flash-V100 path. This is not a rebuilt release wheel. All generation
runs below encode their actual prompt/media and decode/export fresh audio/video.

| Workflow | Adapter | Frames | Calls/rank | Encode | Complete denoise | VAE | Generation total |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| T2VA | FL2V 4-step v1.2 768p, alpha 8, shifts 6/3 | 39 | 4 | 5.76 s | 27.57 s | 6.65 s max rank | 43.94 s |
| First+last FL2VA | same | 39 | 4 | 12.97 s | 34.36 s | 6.11 s max rank | 57.40 s |

Both outputs are 1344x768, 24 FPS, seed 42, with native finite 32-kHz audio.
All automatic checks pass. Inspected first/last screenshots show the red paper
boat and yellow duck; the FL2VA endpoints follow the supplied frames. Human
listening/temporal review is pending. The peak per-rank allocation is 16.59 GiB
for T2VA and 16.61 GiB for FL2VA. Cold worker startup is separate: maximum
94.26 s and 106.30 s, respectively. These are single functional runs with no
warmup/three-repeat performance protocol, not formal speed or quality gates.
The T2VA run preceded the separate LoRA FLOP counter addition; its recorded
numerator excludes adapter work and must not be used for TFLOPS claims.

Commands (from the owned worktree, with task-owned caches and media tools):

```bash
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -m pytest --confcutdir=tests/video tests/video -q
.venv/bin/ruff check \
  vllm/entrypoints/cli/video.py \
  vllm/model_executor/models/minimax_h3/{config,lora,pipeline,reference_video,validation}.py \
  vllm/video/{server,metrics}.py \
  tests/video/{test_h3_lora,test_h3_workflows}.py
```

Result: 84 passed, 8 GPU tests skipped by the CPU-only environment. Real TP4
CLI generation above provides GPU validation of both active adapter loading
and full pipeline execution. The new tests cover all eight official media
combinations, malformed media before dispatch, all eight Turbo filename
contracts, metadata alpha, TP1/2/4 algebra, ConvRot basis and exact zero-scale
bypass. The first fixture attempt lacked FFmpeg on PATH; rerunning with the
existing task-local FFmpeg/FFprobe 7.0.2 passed. Do not diagnose that setup error
as a model failure.

Ref2VA four-step v0.1 weights were downloaded and validated (624 tensors,
alpha 8, shifts 12/3). The mixed-reference real run reached four-rank adapter
binding and the 10337-token Qwen presentation for one image, one video and
standalone audio, then received SIGTERM (exit 143) before denoise. It produced
no completed video and is not counted as passing. Another H3 task held GPU0-3
immediately afterwards. A prior launch was correctly rejected while the group
was leased. No other task's process was stopped by this scope.

For short mixed-reference development use 56 frames: the existing post-encode
reference-audio length check rejects a 39-frame embedded soundtrack after it
is truncated below two seconds. This does not affect the upstream 4–15-second
output contract; it remains an explicit short-development limitation.

Retained artifacts: `/data/minimax-h3/workflows-lora-20260908/`:
`ownership.txt`, `worktree-create.log`, `binary-hashes.txt`, `lora-sha256.txt`,
`omni-supported-models.md`, `omni-h3-recipe.md`, `omni-lora.py`,
`cpu-final.log`, `workflow-tests.log`, `t2va-turbo4.log`, `fl2va-turbo4.log`,
`ref2va-turbo4-wait.log`, `run_reference.py`, and `outputs/*-turbo4/`.
Model and adapter weights remain outside Git. Original/Comfy and Turbo source
revisions are in `UPSTREAM.md`.

Remaining gates: completed mixed Ref2VA, original BF16-base + LoRA generation,
eight-step real outputs and other artifact versions, additional seeds, full
quality/audio review and release-wheel validation. FlashGen, FastH3, combined
partition serving and upstream multipart upload semantics remain separate
work. Keep the change Draft until its required review/quality gates pass.

### Direct frontend API expansion (2026-09-08)

The user authorized all official workflow capabilities and clarified that the
application frontend calls native vLLM APIs directly. ComfyUI integration is
not required. No ComfyUI source or runtime has been added. The full remaining
scope is retained in [ADAPTATION.md](ADAPTATION.md).

Continuing the owned PR #565 branch from `92e8c18e2beda42303268979b89519908740fd1c`:

- Added JSON and multipart request normalization, typed HTTP(S)/data URL
  references and request-owned temporary media. File names cannot choose staging
  destinations; media size/type/metadata checks run before worker dispatch.
- Added synchronous MP4 return, multiple outputs with seed offsets, async job
  listing/deletion, indexed downloads and model discovery. The frontend contract
  and examples are in [API.md](API.md), and `/openapi.json` includes request schemas.
- Preserved native adapter defaults consistently across transports: a startup
  adapter is active unless `lora_scale=0`. A request `lora` object must select
  that loaded file. This differs from upstream's preload-only PEFT default and
  is documented explicitly; adding `model` does not toggle activation.
- Tests cover 11 input combinations, source-file preservation, staged-file
  cleanup, malformed requests, sync/async results, output indexing and schemas.

CPU command and result: the existing `PATH="$PWD/.venv/bin:$PATH"
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest --confcutdir=tests/video
tests/video -q` passed **110 tests**, with 8 GPU-only tests skipped. The complete
pre-commit checks on changed files passed. Raw records are
`/data/minimax-h3/workflows-lora-20260908/api-cpu-all.log` and
`api-precommit.log`; changed API source hashes are in `api-source-hashes.json`.

The new ASGI-to-native-engine mixed Ref2VA test was attempted once using
`run_api_reference.py`, with actual image/video/audio uploads, INT8 ConvRot TP4,
Ref2V four-step v0.1, 1344x768, 4.4 requested seconds, seed 42 and shifts 12/3.
It exited with status 75 before model loading because neither GPU group was
free and unleased. Record: `api-ref2va.log`. No GPU generation result is claimed
for this API revision; no owned workers, service ports or GPU leases remain.
Retry only after resource ownership changes. The other tasks were not stopped.

### FlashGen native four-step adapter and AdaLN restoration (2026-09-08)

Continues the owned PR #565 branch from
`79398b0b5a08e0ea6ed7cac37358f55624b651a7`, with the same base, Python/Torch
environment and direct native API scope.

- Added the official FlashGen rank-64/alpha-64 T2VA artifact as a separate
  native layout. The loader validates all 518 tensors / 259 pairs, including
  grouped QKV, native gate/up FFN and dense AdaLN targets. Runtime deltas use
  staged FP16 A/B buffers, FP32 intermediates and the unrotated input basis.
- Metadata supplies `[1, 0.7, 0.4, 0.15, 0]`; shifts are 12/3. CLI/HTTP defaults
  select four intervals (`num_inference_steps=4`). LightX2V's five/nine-point
  convention is unchanged. Wrong active tasks, layouts and schedules fail.
- A pruned INT8 base restores 106 original AdaLN/time tensors; all backbone
  signed INT8 weights and FP32 scales remain unchanged. Original transformer
  shards are required. Restored weights add about 6.1 GiB per TP4 rank before
  adapter/activation costs. This is a weight-size estimate; actual GPU peak
  remains pending. Scale zero retains restored AdaLN/time components; omit the
  adapter and restart to recover the exact earlier pruned deployment.
- The downloaded official artifact's revision, byte size and verified SHA256
  are recorded in `UPSTREAM.md`. Production TP4 meta-model binding consumed all
  259 targets with 518 actual CPU buffers, totaling 435,126,272 bytes per rank.
  This verifies real checkpoint shapes and FP16 representability, not GPU
  execution. The CPU audit explicitly simulates TP configuration and groups.

The same CPU test command now passes **142 tests**, with 8 GPU-only tests
skipped. Thirty-two FlashGen tests cover metadata errors, exact schedule/task
semantics, TP1/2/4 projection algebra, complete binding, missing original
restoration tensors, INT8 tensor preservation, scale-zero bypass and HTTP
defaults. All changed-file pre-commit checks pass. The first standalone binding
audit needed the owned worktree on `PYTHONPATH` and explicit simulated TP config;
those setup-only failures are retained separately from the successful audit.

Evidence under `/data/minimax-h3/workflows-lora-20260908/flashgen/`:
`manifest.json`, `files.json`, `cpu-all.log`, `precommit.log`,
`inspect_binding.py`, `production-binding-tp4.log`, and `production-binding.json`.
No FlashGen GPU result or output-quality acceptance is claimed.

The real native-engine API mixed-Ref2VA test was retried after an idle snapshot
and successful native GPU lease acquisition. INT8 TP4, Ref2V four-step v0.1,
actual image/video/audio uploads, 1344x768, 107 frames at 24 FPS, seed 42 and
shifts 12/3 reached startup (107.824344 s), all-rank binding and 10,266-token
Qwen media encoding. The process then received SIGTERM (exit 143), at the
start of the four-call denoise loop. There is no completed media or valid
generation timing. The sender was not identified. `api-ref2va-run2.log` retains
the record; no owned workers or GPU leases remain. Do not repeat this unchanged
GPU run without a resource-ownership change or coordinated validation window.

Next acceptance: uninterrupted mixed-reference API generation and FlashGen
T2VA, with measured restoration residency, every-rank call counts, output decode
and video/audio review. FastH3 and the rest of the authorized workflow tracker
remain outstanding; this checkpoint is not full workflow completion.

### FastH3 Dense native fusion and four-step API (2026-09-08)

Continues owned PR #565 from `54d72466f3a81426fc8afe0bf481da7c8e60c4dd`.
Native model/frontend scope and the Python 3.12.13, Torch 2.10.0+cu128,
CUDA 12.8 environment remain unchanged.

- Added explicit FastH3 Dense release identification, rank-64 pairing, declared
  tensor counts and complete 50-block / 2-refiner coverage checks. The official
  artifact's 809 tensors contain 362 low-rank pairs and 85 full-rank weight/bias
  deltas, targeting 343 native parameters.
- Fusion reconstructs FP32 deltas in the original grouped-QKV and gate/up
  layout, rounds to the source checkpoint dtype and enters normal native TP
  loading before pinned staging snapshots the host model. Mapped FP32 adapter
  tensors are not modified. Missing, duplicate and unapplied edits fail startup.
- The CLI/HTTP contract fixes T2VA-only FL2VA, four intervals, shifts 12/3 and
  `[0.999, 0.749, 0.5, 0.25, 0]`. Omitted sampling values follow that contract.
  Fused weights require request scale 1; request `lora` selection is rejected.
  Restart without the adapter to recover the base model.
- This implementation requires original weights. Serialized INT8 fusion and
  VSA execution are rejected and remain separate implementation work. Native
  staging runs after fusion; it does not use Omni's fusion-bypassing host-plan
  path, which is why the upstream offload restriction is not copied blindly.

The complete official Dense artifact was downloaded and SHA256-verified.
`UPSTREAM.md` pins the revision and hash. The artifact's declared original base
and the native pinned base have identical hashes for all 81 FL2VA files.

Validation from the owned worktree:

```bash
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -m pytest --confcutdir=tests/video tests/video -q
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES='' .venv/bin/python \
  /data/minimax-h3/workflows-lora-20260908/fasth3/validate_production_fusion.py
```

The final video suite passes **171 tests**, with **9 GPU-only tests skipped**.
The 29 new CPU tests cover release metadata, partial layouts, checkpoint dtype
rounding, full-rank/bias edits, original INT8 refusal, malformed base streams,
actual native TP1/2/4 loaders, CPU staging snapshot/restore and API restrictions.
A CUDA-versus-CPU fusion regression is present but has not run without a lease.

The real-weight CPU audit fused **all 343 parameters** (66,279,468,032 bytes
of reconstructed source-dtype weights, streamed without retaining a full copy).
Nine representative QKV, row projection, FFN, AdaLN, refiner and patch-projection
weights/biases match the pinned upstream fusion implementation bit for bit.
Every reconstructed tensor is finite and every edit is accounted for. The
four-thread CPU diagnostic took 171.429758 s and peaked at 9.180790 GiB process
RSS. It includes selected oracle reconstruction and comparison; it is not
native worker startup time, deployed host residency, GPU peak or inference speed.

Both four-GPU groups were already leased when inspected. GPU0-3 belonged to
the separate H3 kernel task (PID 526342); GPU4-7 belonged to the other lease
scheduler (PID 48286). No GPU test was launched for this revision. FlashGen,
FastH3 and mixed-reference API generation, CUDA fusion numerics, actual staging
transfers, memory peaks and audio/video quality remain pending GPU acceptance.

Evidence under `/data/minimax-h3/workflows-lora-20260908/fasth3/`:
`download-verified.json`, `base-compatibility.json`, `native-index.json`,
`cpu-all-final.log`, `cpu-final.log`, `precommit.log`,
`validate_production_fusion.py`, `production-fusion.log` and
`production-fusion.json`. Model weights and raw artifacts remain outside Git.
Keep this work Draft; the wider authorized tracker remains in `ADAPTATION.md`.

### User-requested mainline integration (2026-09-08)

The user subsequently requested merging the implemented workflow scope into
`main`. This source integration supersedes the earlier Draft-only instruction;
it does not close the GPU, full-duration, human audiovisual or performance gates.
Those statuses remain explicit in `ADAPTATION.md`.

Required source dependencies are native model PR #557 at
`1d201f41344f1a9a50d91197a8ad3a5525e51190` and SM70 operator PR #558 at
`9764b6c202594158e109c7d7dd4bc3d7124ad25d`. The workflow head before integration
was `4fc5ba8539bed16e00ecf0935a6ca2bf105dae8c`. These were combined in the owned
workflow worktree and synchronized with `onecat/main`
`e5d63c51f0fcc1ddf75d229e3df06bf52df206f5` before publishing the merge candidate.

Conflict resolution retains both `lora_path` and `int8_weight_layout`, the
FlashGen AdaLN restoration branch and the operator's INT8 layout configuration.
Both workflow and kernel evidence histories are retained. The optimized INT8
MLP preparation requires an unwrapped `Int8ConvRotLinearMethod`; a dynamic
LoRA wrapper follows the existing FP32 activation path and cannot be bypassed.

The combined mainline candidate passes **173 CPU video tests**, with **93 GPU
tests skipped** (the count includes the operator dependency's GPU tests).
All changed-file pre-commit checks pass. The three H3 extensions were rebuilt
from this source with Torch 2.10.0+cu128, CUDA Toolkit 12.8, SM70 and CUTLASS
v4.4.2, using an owned compiler cache. This component build does not establish
a complete release wheel or a fresh GPU numerical/quality result.

Integration evidence is under
`/data/minimax-h3/workflows-lora-20260908/merge/`: `cpu-main.log`,
`build-extensions.log`, `extension-imports.json`, `cli-serve-help.log`,
`precommit-kernels.log` and `precommit-main.log`. The exact build command and
final GitHub merge SHAs are retained in that task's merge/handoff artifacts.
