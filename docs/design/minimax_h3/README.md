# Native MiniMax H3 (development)

The latest optional residual-sharded FlashInfer development run completes
39 frames and 20 denoise updates in 61.538397 seconds, with unchanged
video/audio latents and fresh MP4. See
[FLASHINFER_LOCAL_ROTATION.md](FLASHINFER_LOCAL_ROTATION.md) for local ConvRot,
the current bottleneck trace, validation and remaining gates.
The numerical-drift overlap experiment is rejected; see
[FLASHINFER_OVERLAP.md](FLASHINFER_OVERLAP.md).

This is an in-progress native integration. Full checkpoint video quality and
four-card 80 useful TFLOPS acceptance are not yet established. The control log
records completed tests and remaining gates. No separate vllm-omni installation
is required.

For explicit workflow selection, reference-video offsets, four/eight-step
LightX2V Turbo, FlashGen and FastH3 Dense, see [Workflows and distilled LoRA](WORKFLOWS.md).
The application frontend calls the [native video API](API.md) directly.
The [adaptation tracker](ADAPTATION.md) records the remaining official workflows.

The supported deployment contract is Python 3.12, Torch 2.10.0+cu128, CUDA Toolkit
12.8 and V100/SM70. Install the normal 1Cat source build with its `video` extra;
`tools/minimax_h3/build_extensions.py` builds the three independent H3 operators
when working on top of an existing compatible 1Cat installation. It requires
`CUDA_HOME`, `TORCH_CUDA_ARCH_LIST=7.0`, and `VLLM_CUTLASS_SRC_DIR` pointing to
CUTLASS v4.4.2 source. It does not rebuild the rest of vLLM. The normal CMake
build fetches that pinned CUTLASS version automatically.
The development tests use Transformers 5.15.1 and Diffusers 0.40.0.

## Host memory and temporary disk storage

Four 32-GB GPUs do not imply that the host can retain the complete CPU model.
Pinned DiT and text-encoder masters exhausted a 64-GiB host during TP4 loading.
`--host-memory-mode auto` therefore uses reclaimable, disk-backed CPU weights
on hosts with less than 128 GiB of physical RAM. Larger hosts retain the pinned
path. `--host-memory-mode mmap` or `pinned` selects a policy explicitly.

The disk-backed path preserves tensor bytes, aliases, strides and dtypes;
quantization, attention and denoise schedules are unchanged. Parameters are
mapped before streaming checkpoint writes, avoiding a complete anonymous CPU
copy before offload. Storage transformed during quantization and adapter
preparation is mapped again before staging. Device transfers from these CPU
masters may take longer than transfers from pinned RAM.

Temporary storage defaults to `VLLM_CACHE_ROOT/h3-host`. Use
`--host-memory-directory /path/on/local-ssd` to select another directory.
Each allocation reserves disk space before mapping so a full disk reports a
loading error instead of a later SIGBUS. Files are immediately unlinked after
mapping; their space remains occupied until the worker releases the tensors,
including after an abrupt worker exit. Original model files remain read-only.
Provide enough free SSD space for the selected model's CPU weights. This mode
reduces weight residency; it does not eliminate activation or media-processing
memory requirements.

```bash
export CUDA_HOME=/path/to/cuda-12.8
export TORCH_CUDA_ARCH_LIST=7.0
uv pip install -e '.[video]' --no-build-isolation
vllm video generate \
  --model MiniMaxAI/MiniMax-H3 \
  --partition fl2va \
  --transformer-path /path/to/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
  --tensor-parallel-size 4 \
  --attention-backend FLASH_ATTN_V100 \
  --output-dir ./h3-output
```

The default prompt is the approved paper-boat/duck scene; defaults are 1344x768,
243 frames, 24 FPS, seed 42 and 50 sigma positions (49 actual DiT calls).
For short development quality checks, use
`--num-frames 39 --num-inference-steps 21`: this keeps the 1344x768 canvas,
generates 1.625 seconds and performs 20 DiT forwards. The sigma-point convention
means N positions give N-1 denoise updates for this checkpoint. The fixed
acceptance workload still uses 50 positions (49 forwards). One- or two-forward
runs are only execution/numeric diagnostics: they produced severe ghosting and
grid artifacts that disappeared from the inspected 20-forward samples. The minimum is 22 frames (0.917
seconds), required by the streaming VAE. Requests align upward to 17n+5 frames:
`--duration 1` produces 39 frames and `--duration 2` produces 56 frames.
Use an aligned `--num-frames` value when an exact short duration is wanted.
Omit `--transformer-path` to load the original BF16 checkpoint into the FP16/FP32
mixed-precision model. The reference checkpoint's VAE Python code executes from
its downloaded model directory. Frozen revisions and port licenses are recorded
in the model package's `UPSTREAM.md`. FFmpeg and FFprobe must be on `PATH` for
reference-video/audio processing.

For hosts that cannot hold every component's pinned CPU copy, use
`--disable-host-weight-pinning` (Python: `host_weight_pin_memory=False`). This
keeps pageable immutable masters for the DiT, text encoder and both VAEs.
It preserves weights, layouts and computation, while allowing the OS to page
inactive components; transfers and request startup can be slower. The original
TP4 deployment holds about 157 GiB of CPU masters before loader temporaries and
allocator caches, so all-pinned startup exhausted the tested 188 GiB host.
See [GENERAL_SM70.md](GENERAL_SM70.md) for the complete floating-weight control.

Use `--image first.png --keyframe-indices 0`, `--image last.png
--keyframe-indices -1`, or two `--image` arguments with `--keyframe-indices 0 -1`
for FL2VA. A Ref2VA instance uses `--partition ref2va` and the matching transformer
file; `--image`, `--video` and `--audio` are repeatable reference arguments.
A serving instance binds one partition and processes one request at a time.

```bash
vllm video serve \
  --model /path/to/MiniMax-H3 \
  --partition fl2va \
  --transformer-path /path/to/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
  --tensor-parallel-size 4 --host 127.0.0.1 --port 8000
curl -sS http://127.0.0.1:8000/v1/videos \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"一艘红色纸船漂过公园浅水池，轻柔流水声。","seed":42}'
```

`POST /v1/videos` returns a job ID. Query `GET /v1/videos/{id}` and download
`GET /v1/videos/{id}/content` after completion. The service also supports
multipart uploads, typed reference URLs, synchronous MP4 responses, multiple
outputs, job listing/deletion and OpenAPI request schemas. See [API.md](API.md)
for the frontend contract, ownership/cleanup behavior and adapter defaults.

`FLASH_ATTN_V100` is the default denoiser and the main performance-development
path. With the H3 SM70 extensions it selects the dedicated non-causal D128
TensorOp operator. It preserves all independent MHA heads and the D128 scale;
it does not pad the model's head dimension to D256. `FLASHINFER_SM70` remains
an explicit comparison and rollback option. The Torch
reference backend is for numerical investigation. Text encoding uses TP4 and
Flash-V100 causal GQA; the selectable denoiser backends use non-causal MHA.

FP16 weight caching defaults to zero. After measuring candidate layers, pass a
budget with `--fp16-weight-cache-gib` and repeat `--fp16-cache-layer` for the
fixed layer list. Cached weights retain ConvRot coordinates. Cache/staging
preparation is separately timed; dequantization remains inside denoise timing
for uncached weights. Both original INT8 tensors and FP32 scales are retained.
The measured 39-frame/20-update cache list and native cuBLASLt configuration
are documented in [FLASHINFER_TO50.md](FLASHINFER_TO50.md), including exact
commands, output comparison and the still-incomplete 50-second target.

The earlier experimental FlashInfer residual route changed reduction order;
its historical results and unaccepted differences are recorded in
[FLASHINFER_RESIDUAL.md](FLASHINFER_RESIDUAL.md). Current residual sharding uses
the ordinary full FP32 all-reduce before selecting local residual rows. It
preserves the complete four-step control bitwise, and does not claim the old
reduce-scatter communication saving. It remains explicitly enabled with
`--residual-sequence-parallel` and requires wider quality/performance validation.

The subsequent probability-tile layout change reduces this same optional
route to 65.804661 seconds and preserves its video/audio outputs bitwise.
See [FLASHINFER_OPERANDS.md](FLASHINFER_OPERANDS.md) for the instruction-level
bottleneck, corrected A/B measurements and new sanitizer/profiler evidence.

`--int8-weight-layout column` is the default for DiT INT8 projections. Loading
reorders physical INT8 storage without changing logical weights or scales.
Each invocation decodes transient FP16 weights and selects the validated SM70
cuBLASLt plan with FP32 accumulation and zero workspace. This does not create
a persistent FP16 cache. Unsupported plans use the original GEMM; warm up a
shape before CUDA graph capture. Use `--int8-weight-layout row` to restore the
original weight layout and GEMM route. Original BF16 checkpoints are unaffected.

For INT8 MLPs, the native path combines FP32 SiLU/product evaluation and
power-of-two FP16 input preparation in one kernel. The following projection
restores the scale in FP32 before the normal TP reduction. This removes the
large FP32 activation intermediate without lowering arithmetic precision or
adding a persistent cache. Prepared inputs now also support original floating
weights and active LoRA. LoRA restores the first A projection's row scale before
preparing B, retaining both rounding boundaries and pre-reduction delta addition.
The shared FP16 GEMM and preparation operators live outside the H3 model package.
Original floating projections can opt into `--fp16-weight-layout column`; the
default remains `row`. Layout preparation preserves logical weight coordinates
and storage size. Full-workflow qualification is tracked in [GENERAL_SM70.md](GENERAL_SM70.md).

For the measured TP4 FL2VA INT8 development workload, optionally add
`--residual-sequence-parallel` to `video generate` or `video serve`. This keeps
FP32 residual rows sharded across ranks and gathers normalized FP16 inputs at
the existing precision boundary. Both native SM70 attention backends support
this option. It now accepts original weights, Ref2VA and matching adapters on
TP2/TP4; TP1 uses ordinary execution. It defaults to false. Runtime guards still
require one request, FP32 residuals, aligned metadata and no Ulysses hooks.
For projections that can consume only a rotated input, normalized FP16 rows
are rotated locally before all-gather, avoiding
four copies of the same ConvRot work while preserving gathered bits and GEMMs.
An active adapter also needs the unrotated input, so the ordinary gather is
retained there until a validated combined preparation removes that dependency.
The following measurements predate the generalized route and do not qualify it:
The native FlashAttention 39-frame/20-update run measures 58.190385 s
and 6.195001125 GiB peak denoise allocation/card, versus 62.804019 s and
6.566735744 GiB without the flag, with zero persistent cache in both cases.
Its complete audio/video latents equal the earlier 58.794562-second sharded
run bitwise, so that run's checked decode is reused. Automatic checks pass;
the sharded route changes FP32 reduction order from the default and its human
quality review remains pending. Omit the flag to restore the default.
These are short development measurements, not the 243-frame acceptance run.

The generalized four-step/full-canvas check exposed substantial latent drift
from reduce-scatter's FP32 addition order. Residual sharding now performs the
same full all-reduce as the replicated path before selecting each rank's rows.
This preserves the reference sum order and does not claim a communication-volume
reduction. The old reduce-scatter speed figures above are not qualified outputs.

Outputs include `video.mp4`, original decoded `audio.wav`, `run.json`, sampled
`nvml.jsonl`, `quality.json` and frame screenshots. Automatic checks do not
replace the five-axis human quality review. Useful TFLOPS use actual local
matrix dimensions and valid attention tokens, exclude padding/rotation/dequant,
and divide by the maximum complete denoise duration across all four ranks.
NVML utilization and standalone operator speed are diagnostic evidence only.

The native engine reserves a whole available GPU group with the shared 1Cat
V100 per-card and group locks. An explicit `CUDA_VISIBLE_DEVICES` selection
(NVML indices or GPU UUIDs) bounds the candidate groups; a busy selected group
never falls back to unselected devices. UUIDs are recommended for launchers.
Without a selection, devices with at least 30 GiB capacity form ordered groups,
so a small display card does not shift the four-V100 grouping. Workers receive
the leased devices as UUIDs to avoid CUDA/NVML ordinal-order differences.
The NVML platform resolves these UUIDs for physical-device queries before
pipeline imports, so capability checks use the same selected boards. Custom
all-reduce topology checks use this mapping as well.
The lease remains held until
its workers exit. A group reserved by another cooperating task is unavailable
even while that task is between CUDA processes. If both groups are occupied or
reserved, startup fails before model loading.
