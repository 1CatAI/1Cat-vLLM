# Shared SM70 projection epilogues

This development branch is stacked on the common prepared execution (#571)
and workflow accounting (#578) branches. No configuration has passed the
campaign's >80 useful TFLOP/s/card and complete official quality gates.

The separate [FI register-probability update](FLASHINFER_REGISTER_PROBABILITY.md)
now also has full native media preservation and a formal warmup-plus-three
result of 49.795 useful TFLOP/s/card. It remains below the FA query-128 result
and the campaign target. Both use the same shared projection interface.

## Measured problem and implementation

The matching four-step FA denoise profile spends 6.890 seconds in miscellaneous
elementwise kernels, including FP32 row-scale restoration, adapter additions
and output casts. Attention, GEMM and communication separately consume
31.439, 16.041 and 8.453 seconds. These are profiled service diagnostics;
the unprofiled audited baseline is 47.091839–47.091855 useful TFLOP/s/card.

`sm70_diffusion.fp16_linear_add` prepares the projection input at the existing
FP16 boundary and keeps GEMM accumulation in FP32. A shared CUDA epilogue
restores row scales with a rounded FP32 multiplication, adds the scaled delta
with the same FP32 fused multiply-add as PyTorch, and stores FP16 or FP32.
It writes only the requested output slice. Delta/output and scale/output
storage overlap is rejected, including differently typed views of one buffer.

The H3 adapter uses this interface when output hardware, precision and layout
support it and adapter slices are disjoint. Overlapping contributions retain
the original FP32 buffer until the final cast. Wheels without the new extension
ABI retain the ordinary path. No quantization label or adapter name controls
dispatch. The first LoRA projection still uses unrotated inputs; row-parallel
increments still join the partial result before the collective.

## Development evidence

Environment: Python 3.12.13, Torch 2.10.0+cu128, CUDA toolkit 12.8.93,
V100 SXM2 32GB. All GPU tests use an owned native lease. Evidence root:
`/data/minimax-h3/sm70-general-20260909/`.

- `epilogue-integration-v1.log`: 43 checks pass, including explicit comparison
  with the old unfused adapter path for original/W8A16 bases, FP16/FP32 output,
  prepared/ordinary inputs, negative scales, wide intermediates, untouched
  slices, overlap fallback, unaligned storage and CUDA Graph replay.
- `epilogue-cpu-v1.log`: 94 adapter, workflow and strict acceptance checks pass.
- `epilogue-micro.json`: paired postprocessing-only medians for 34,560 rows,
  output width 5,376: three FP16 QKV slices 7.524352 -> 2.015232 ms; one FP32
  row projection 4.562944 -> 3.094528 ms. Results match bitwise. These timings
  exclude GEMM and do not establish complete-request speedup.
- `epilogue-binaries.json` retains the first tested binary. The strengthened
  alias check is in `epilogue-binaries-v3.json`; validation is recorded separately.

The final alias guard passed on GPU0 (`epilogue-alias-v3-gpu0.log`). An earlier
GPU4 attempt was refused by an existing lease before starting the test.

## Complete four-step control and measurements

Source `7ea8908d83827dd8d82c34ba6a60b2beaa8057d6`, TP4, LightX2V four-step v1.2,
W8A16, FA, exact residual sharding, no persistent FP16 weight cache, and the
original 1280x736/124-frame internal canvas for the five-second sample:

- `epilogue-quality.json`: final video/audio latents and all pre-encoding RGB
  frames match frozen mainline bitwise; video SSIM 1, spectral cosine 1 and RMS
  ratio 1. This is numerical preservation, not independent official acceptance.
- `epilogue-720p-three-runs/performance.json`: one full warmup (64.373997 seconds
  denoise) followed by three complete requests without profiler or captures.
  Denoise times are 62.245159 / 62.183194 / 62.162648 seconds; CV 0.056387%.
  Every rank reports median **49.905491–49.905510 useful TFLOP/s**. The declared
  >80 gate fails and remains incomplete.
- Complete request times are 84.363265 / 91.992465 / 88.620830 seconds. Peak
  allocation remains 19,501,498,880 bytes per card. Exact source/kernel hashes,
  per-step records and NVML samples are retained beside each run.
- The 62.183194-second median is 5.64% below the audited original FA baseline
  (65.898529 seconds). This measures the **combined** prepared/residual/epilogue
  changes, not an isolated attribution to this CUDA epilogue. A separate matched
  profile is necessary for attribution.

Independent official reference, full audiovisual review, other adapters and
the complete shape/TP matrix remain required. No AUTO selection is qualified.

## Explicit attention query geometry

`attention_query_tile=128` / `--attention-query-tile 128` opts into a 128-query
FlashAttention-V100 CTA. The default remains 64 and retains the previous call
ABI. Both sizes use the same 32x64 warp arithmetic and key-tile selection.
The option applies independently of model weights and adapters; other attention
backends reject this explicit tiling option rather than ignoring it.

The separate prototype retains exact outputs at nine boundary lengths and
the actual 34,551-token Q/K/V capture. Full four-step video/audio latents also
match frozen mainline bitwise (`q128-profile-quality.json`). A matching pair of
full-denoise profiles is retained in `epilogue-profile-breakdown/` and
`q128-epilogue-profile-breakdown/`; profiler timings are not acceptance results.
The public kernel/CLI implementation additionally passes 69 GPU tail, storage,
cross-attention-length and graph checks; 53 strengthened comparisons also
require exact equality between query geometries and reject invalid query tiles.

The native option at source `6b39f1c23ca6834e9beead89b4cdf57548101e97` passes
the complete latent/RGB/PCM comparison (`query-tile-quality.json`): both final
latents and all 124 frames match frozen mainline bitwise; SSIM 1 and all audio
gates pass. The same explicit configuration completed one full warmup plus
three unprofiled requests (`query-tile-720p-three-runs/performance.json`):

| Configuration | Median denoise seconds | Useful TFLOP/s/card | Denoise CV |
| --- | ---: | ---: | ---: |
| Audited original FA baseline | 65.898529 | 47.091839–47.091855 | 0.055668% |
| Prepared/residual/epilogue, query tile 64 | 62.183194 | 49.905491–49.905510 | 0.056387% |
| Same path, explicit query tile 128 | 59.748563 | 51.939038–51.939057 | 0.003793% |

The last three denoise times are 59.743807 / 59.748563 / 59.748666 seconds.
Complete request times are 81.805001 / 82.780855 / 84.272880 seconds and peak
allocation remains 19,501,498,880 bytes/card. The 128-query setting reduces
median denoise by 3.92% relative to the same prepared 64-query configuration,
and the combined change reduces it by 9.33% relative to the original baseline.
These measurements cover one five-second workflow only. **The >80 gate still
fails; official reference, human review and the wider matrix remain pending.**

## Eight-step numerical preservation

The same explicit query-128/prepared/residual/epilogue configuration at
`bd5e1898265eb1783fcc413de321125230fbe594` also completed a matched TP4
LightX2V eight-step FL2V v1.0_768p comparison with frozen mainline `4f19ef7`.
Both runs use W8A16, seed 42, the same five-second request, nine sigma points,
flow shift 6 and audio flow shift 3. The official adapter SHA256 is
`9b0efe3613b43a84e30febaa43af27432ea9d0711eac7bba904b2556b175f6d4`.

`light8-720p-quality.json` passes every declared numerical gate: both final
latents, all 124 RGB frames and decoded PCM match bitwise. This extends
preservation evidence beyond the four-step adapter, using the same shared
operators without an adapter-specific dispatch exception. It remains a frozen
native control, not independent official-model acceptance.

The captured cold requests took 141.160054 and 121.541891 seconds in denoise;
complete request times were 174.840862 and 203.651622 seconds respectively.
These single captured runs have different staging conditions and no full
warmup, so they do not establish a formal performance result or an overall
request speedup. Source hashes, binary manifests and commands are recorded
in `light8-pair.json`. The eight-step >80 gate remains unmeasured.

An additional 16-row warp experiment (`attention-warp16/hypothesis.json`)
was rejected at compilation: CUTLASS Volta MMA requires a multiple of its
interleaved tile shape. No GPU run or production change followed. Supporting
that geometry requires new MMA and accumulator iterators, not another
configuration-only benchmark of the rejected shape.

## Mixed-reference eight-step control

Source `2063b09f2d75c4a63af3f90a3b7803744ffc6e02` completes Ref2VA with the
official eight-step v1.0_768p adapter, W8A16 Ref2VA base, seed 42 and one image,
one 2.5-second reference video plus one standalone audio reference. The video
start time is zero. The output remains 1280x736/124 internal frames for the
five-second request. This control uses 69,325 valid DiT tokens and a 10,273-token
Qwen presentation, exercising mixed reference indices and padded residual rows.

`ref8-mixed-quality.json` passes all gates against frozen native mainline:
video/audio latents, all RGB frames and PCM are bitwise equal; PSNR infinity,
SSIM 1, RMS ratio 1. Candidate settings include prepared execution, exact
residual sharding, query tile 128 and explicit shared pageable VAE host weights.
The frozen control uses its ordinary query-64 path and pinned host masters.
Host residency changes byte ownership/transfers, not GPU arithmetic.

Single captured cold denoise times are 421.173489 seconds for the frozen
control and 366.799932 seconds for the candidate; request times are
586.794155 and 454.461091 seconds. Peak GPU allocation is 19,623,684,096 and
19,615,279,104 bytes/card respectively. The candidate reports corrected useful
throughput 52.919225–52.919231 TFLOP/s/card. These runs lack full warmup and
three measurements and have different host staging policies, so they do not
qualify either performance or attribution to an individual optimization.
Independent official quality, audiovisual/reference review and other reference
combinations remain pending. Raw contracts and media are retained in
`/home/ymzx/h3-sm70-artifacts-20260909/runs/ref8-mixed-720p-{baseline,candidate}/`.

Two additional CTA-barrier coalescing candidates preserve bitwise edge and
34,551-token results. The second passes 20 synccheck and racecheck geometries
with zero errors/hazards. Their paired operator gains are only 0.3–0.6%, so
neither is retained or promoted to a full-request performance claim. Evidence:
`attention-barrier-coalesce/`, `attention-barrier-coalesce-v2/` and the
`barrier-v2-*.log` files under the campaign root.

Further isolated operand probes are also rejected:

- `attention-fi-p-reuse/` interchanges PV loops in the retained FlashInfer
  Q128/K64 kernel to share a loaded P fragment across four output fragments.
  Nine boundary shapes and the real 34,551-token input remain bitwise equal,
  but paired latency regresses from 183.285767 to 185.328644 ms.
- `attention-rescale-identity/` uses a warp-uniform check to skip accumulator
  multiplication when every applicable online-softmax scale equals one.
  The same numerical cases remain bitwise equal; 150.328323 to 149.099518 ms
  is less than 1% and includes clock variation, so it is not retained.

Neither probe changes production kernels or establishes a full-denoise gain.
Their source, build logs, hashes and actual-input timings remain in the
campaign artifact root to prevent repeating unchanged experiments.

## Larger canvas and duration compatibility

Source `9d2489fc4e2f32ea500ec13478bd68ef9000c1cc` completes two additional
TP4 W8A16 LightX2V four-step requests with prepared execution, exact residual
sharding, query tile 128 and shared pageable VAE masters. Both use seed 42,
the original paper-boat prompt, five sigma points and flow shifts 6/3.

| Requested shape | Actual frames | Denoise seconds | Request seconds | Peak GPU allocation bytes/card | Useful TFLOP/s/card |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1344x768, 243 frames | 243 | 203.431920 | 254.138019 | 24,341,115,904 | 52.603224–52.603229 |
| 1344x768, 15 seconds | 362 | 401.026288 | 469.032456 | 28,777,495,040 | 53.561244–53.561247 |

The 15-second request resolves to the model's 362-frame aligned output; it is
not claimed to be an exactly 15.000-second encoded clip. Both complete native
media validation and strict actual-work checks, and remove their owned shared
weight directories after shutdown. Full captures, source/binary manifests,
per-rank stages and NVML samples are retained under
`/home/ymzx/h3-sm70-artifacts-20260909/runs/official-243-frame/` and
`boundary-15-second/`; `large-canvas-summary.json` summarizes the evidence.

These first captured requests establish shape and memory compatibility only.
They have no matched quality reference, full warmup or three post-warmup
measurements. Both are below 80 and remain unqualified. The independent
official reference and human review gates are also pending.

## Remaining FL2V Turbo versions

Source `be89a26d1c` completes four more matched frozen-mainline comparisons on
TP4 at the same 1280x736/124-frame internal canvas for a five-second request.
All use W8A16, seed 42, pageable host masters and no fixed FP16 weight cache.
The candidate uses prepared execution, exact residual sharding, shared LoRA
epilogues and query tile 128. The frozen `4f19ef7` control uses its ordinary
query-64 path. Each artifact retains its official alpha and flow shift.

| Official artifact | Intervals / sigma points | Video shift / alpha | Candidate denoise seconds | Request seconds |
| --- | ---: | ---: | ---: | ---: |
| FL2V four-step v1.0_768p | 4 / 5 | 6 / 128 | 61.263482 | 96.285706 |
| FL2V four-step v1.1_768p | 4 / 5 | 6 / 128 | 61.018722 | 96.491105 |
| FL2V four-step v0.1 | 4 / 5 | 12 / 8 | 61.000133 | 94.938178 |
| FL2V eight-step v1.0 (non-768p) | 8 / 9 | 12 / 8 | 120.478234 | 156.924670 |

For every pair, final video/audio latents, all 124 pre-encoding RGB frames and
decoded PCM match bitwise. SSIM and RMS ratio are 1. Strict per-rank workload
validators pass, including actual intervals, block counts and duplicate-work
exclusion. Peak allocation is 19,501,498,880 bytes/card for the four-step cases
and 19,502,023,168 bytes/card for eight-step. Single-request useful throughput
is approximately 50.65–51.52 TFLOP/s/card, below 80.

These are captured cold quality controls, not warmed three-run performance.
Together with the existing v1.2 four-step and v1.0_768p eight-step controls,
all six official FL2V Turbo artifacts now have a complete W8A16 T2VA numerical
preservation result. Original-weight and keyframe combinations remain separate
pending coverage. The subsequent Ref2V four-step control is recorded below.
No independent official quality or human acceptance is inferred.

Evidence: `remaining-turbo-pairs.json`, `remaining-turbo-summary.json`,
`light4-v{10,11,01}-720p-quality.json`, `light8-v10-non768-720p-quality.json`
and the corresponding captured runs under the campaign's artifact root.

## Four-step mixed references and complete adapter inventory

Source `c69cfc7024460e314e79a0bba37a3b736340bc6e` completes the official
Ref2V four-step v0.1 adapter with a W8A16 Ref2VA base, one image, one
2.5-second video and one standalone audio reference. The video starts at zero;
seed 42, five sigma points, video/audio shifts 12/3 and alpha 8 are retained.
The candidate uses the same general FA query-128 path as the other adapters.

`ref4-mixed-quality.json` passes all declared numerical gates against frozen
native `4f19ef7`: final video/audio latents, all 124 RGB frames and PCM match
bitwise, PSNR is infinite, SSIM is 1 and RMS ratio is 1. Spectral cosine is
0.9999999999999695. Both native generations complete. The candidate's strict
actual-work checks pass; its complete denoise is 184.563686 seconds, request
269.210367 seconds, and peak allocation 19,613,711,360 bytes/card. Corrected
useful throughput is 52.585556-52.585562 TFLOP/s/card.

The frozen control takes 214.108869 seconds denoise and 321.250580 seconds
request with the same peak allocation. These are captured cold requests with
different host VAE sharing policies; they are not formal speed acceptance.
The old control script did not embed Git metadata. The separate
`baseline-source-audit.json` verifies all 2,483 tracked `vllm` files in the
frozen archive against `4f19ef7`, without changing the historical contract.

All eight official LightX2V artifacts now have complete native numerical
preservation evidence: six FL2V adapters on T2VA and both Ref2V adapters with
mixed references. This is not the full task/weight/reference cross-product,
independent official quality, human review or >80 acceptance. Exact paths,
source identities and timing scope are retained in `ref4-mixed-pair.json`,
`ref4-mixed-summary.json` and the campaign result index.

## First, last and both-frame controls

Source `4ed70419e7f42c6e9f4625f7fa92cf8dea2126d8` also completes all three
FL2VA keyframe modes using the official Light4 v1.2_768p adapter and W8A16,
with one immutable engine per implementation. The candidate uses the same
general FA query-128/prepared/exact-residual/epilogue path. First and last
images are the retained frames 0 and 123 of the campaign sample, selected
with indices `[0]`, `[-1]` and `[0,-1]` respectively.

| Constraint | Frozen-control denoise seconds | Candidate denoise seconds | Candidate request seconds | Candidate useful TFLOP/s/card, minimum |
| --- | ---: | ---: | ---: | ---: |
| First frame | 77.275887 | 66.419303 | 106.123582 | 50.697486 |
| Last frame | 71.537050 | 64.852263 | 102.142091 | 51.922501 |
| First and last frames | 77.299085 | 69.615145 | 108.431893 | 52.304580 |

Every pair passes complete numerical preservation: final video/audio latents,
all 124 decoded frames and PCM match bitwise; SSIM and RMS ratio are 1.
Peak allocation is unchanged within each pair: 19,512,957,952 bytes/card
for a single image and 19,522,796,544 bytes/card for both images.

These are captured requests with different first-use state, reference lengths
and host VAE sharing policies. They establish full execution and native
preservation, not a formal performance comparison or independent verification
of reference fidelity. Original-weight/other-adapter keyframes and the full
legal Ref2VA combination matrix remain pending.

Evidence: `keyframe-pairs.json`, `keyframe-summary.json`, and
`keyframe-{first,last,first-last}-quality.json`. The candidate contract retains
shared-operator, model and video-source hashes plus clean Git provenance.

## Original floating FlashGen and FastH3 Dense controls

Both four-interval T2VA variants now have complete native comparisons with
frozen mainline `4f19ef7a20db60bb0685e599bd3f4dd156202eed`. Each pair uses
original floating FL2VA weights, its matching official adapter, seed 42,
TP4, flow shifts 12/3 and the same 1280x736/124-frame internal canvas.
Candidate source and individual file hashes are retained in each contract;
the frozen source audit covers all 2,483 tracked package files with no mismatch.

Both final video/audio latents, all 124 pre-encoding RGB frames and PCM match
bitwise for both variants. SSIM and RMS ratio are 1; spectral cosine exceeds
0.99999999999996. This establishes native numerical preservation for these
configurations, not independent official-model or human quality acceptance.

| Variant | Baseline denoise / request seconds | Candidate denoise / request seconds | Candidate useful TFLOP/s/card | Candidate peak bytes/card |
| --- | ---: | ---: | ---: | ---: |
| FlashGen four-step | 73.290174 / 162.679376 | 59.488178 / 121.697365 | 51.620900 | 20,781,940,224 |
| FastH3 Dense data-free | 61.574164 / 121.787763 | 56.223567 / 114.632281 | 54.124949 | 20,023,082,496 |

These are captured cold controls. Both arms use pageable host weights; the
candidate also shares host VAE weights and enables prepared column weights,
exact residual sharding, shared epilogues and explicit FA query tile 128.
The measurements combine these changes and do not isolate a kernel effect.
Neither variant has completed a full warmup plus three unprofiled requests.

Evidence: `original-variant-pairs.json`, `original-variant-summary.json`,
`flashgen-original-quality.json`, `fasth3-dense-original-quality.json` and
`baseline-source-audit.json` in the campaign artifact root. Complete media and
contracts reside in the corresponding `*-original-{baseline,candidate}` runs.
The [campaign table](CAMPAIGN_RESULTS.md) separates these diagnostic timings
from formal acceptance measurements. Every configuration remains unqualified.
