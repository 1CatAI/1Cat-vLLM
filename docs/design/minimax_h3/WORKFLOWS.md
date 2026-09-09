# Native H3 workflows and distilled LoRA

This extends the native 1Cat pipeline from PR #557. It does not require a
separate vllm-omni runtime. Upstream comparison is pinned to vLLM-Omni
`b58ff5cb8b17250b76f9cdf9b9b46385cdda4376` (2026-09-08):
[H3 recipe](https://github.com/vllm-project/vllm-omni/blob/b58ff5cb8b17250b76f9cdf9b9b46385cdda4376/recipes/MiniMaxAI/MiniMax-H3.md).
The wider model inventory is in [Omni coverage](../omni_workflow_coverage.md).

## Workflow contract

One engine serves one DiT partition. Both partitions use the shared FL2VA
tokenizer, Qwen3-VL encoder and VAEs. Use a separate engine configuration when
switching between FL2VA and Ref2VA; simply changing the request task cannot
change the loaded checkpoint.

| Workflow | Partition | Request fields / CLI arguments |
| --- | --- | --- |
| Text to video + audio | `fl2va` | `task=t2va`; no media |
| First frame to video + audio | `fl2va` | `task=fl2va`, one image, indices `[0]` |
| Last frame to video + audio | `fl2va` | `task=fl2va`, one image, indices `[-1]` |
| First and last frames | `fl2va` | `task=fl2va`, two images, indices `[0,-1]` |
| Image reference | `ref2va` | `task=ref2va`, one or more images |
| Image and audio reference | `ref2va` | images and standalone audio |
| Video reference | `ref2va` | video, with its embedded audio when present |
| Mixed reference | `ref2va` | images, videos and standalone audio |

Ref2VA requires a visual reference; audio-only is not an upstream-supported
workflow. Limits are 9 images, 3 videos, 3 standalone audio files and 12 total
files. Source videos/audio must each last 2–15 seconds, with a maximum total
duration of 15 seconds for each input type. Video start times must leave at
least two seconds in each source. Native validation uses the same metadata
checks as preprocessing, before worker dispatch. FFmpeg and FFprobe must be
on `PATH`. Corrupt media returns HTTP 422 without closing the worker group.

CLI media flags are repeatable: `--image`, `--video`, `--audio`.
`--keyframe-indices` selects first/last frames;
`--reference-video-start-times` takes one offset in seconds per video.
Explicit `--task`, `--flow-shift`, and `--audio-flow-shift` are now exposed.
Without `--task`, the existing partition/media-based task inference remains.

## LightX2V Turbo family

Download the **Diffusers export**, retaining its published filename:

```bash
hf download lightx2v/Minimax-h3-Turbo \
  minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors \
  --revision 2f015e66b37c585cea9dc4ae6f1850ea8788e742 \
  --local-dir ./h3-turbo
```

Every filename below starts with `minimax_h3_` and ends in `.safetensors`.
All use rank 128 and audio flow shift 3. Alpha is read from file metadata,
with LightX2V's reference default 8 when metadata omits it.

| Filename middle | Tasks | Actual denoiser calls | Sigma points | Video shift | Alpha |
| --- | --- | ---: | ---: | ---: | ---: |
| `fl2v_turbo_4step_v0.1` | T2VA / FL2VA | 4 | 5 | 12 | absent → 8 |
| `fl2v_turbo_4step_v1.0_768p_bf16` | T2VA / FL2VA | 4 | 5 | 6 | 128 |
| `fl2v_turbo_4step_v1.1_768p_bf16` | T2VA / FL2VA | 4 | 5 | 6 | 128 |
| `fl2v_turbo_4step_v1.2_768p_bf16` | T2VA / FL2VA | 4 | 5 | 6 | 8 |
| `fl2v_turbo_8step_v1.0_bf16` | T2VA / FL2VA | 8 | 9 | 12 | 8 |
| `fl2v_turbo_8step_v1.0_768p_bf16` | T2VA / FL2VA | 8 | 9 | 6 | 8 |
| `ref2v_turbo_4step_v0.1_bf16` | Ref2VA | 4 | 5 | 12 | 8 |
| `ref2v_turbo_8step_v1.0_768p_bf16` | Ref2VA | 8 | 9 | 6 | 8 |

`--lora-path` accepts one local file, or a directory containing exactly one
recognized artifact. ComfyUI fused exports and renamed files are refused instead
of being interpreted as this layout. FlashGen and FastH3 Dense use their separate
loaders described below. FL2V and Ref2V adapters require their matching base partition.

One immutable adapter is loaded at engine startup. Requests apply a multiplier
`--lora-scale` / `lora_scale` (default 1) to `alpha / rank`. Setting it to zero
skips all LoRA matmuls and uses base-model sampling defaults. There is no hot
loading of arbitrary files, composition of adapters, or weight prefusion.

When the CLI/HTTP request omits steps or shifts, native 1Cat selects the active
adapter's values and records them in the request. Explicit values are preserved
and checked: **four-step LightX2V needs `num_inference_steps=5`**, and eight-step
needs `9`. This is the inherited sigma-point convention. A base request still
defaults to 50 points / 49 updates. Python callers can use
`sampling_for_deployment(config, ...)`; directly constructed
`H3SamplingParams` retains its explicit/default 50-point contract.

```bash
vllm video generate \
  --model /path/to/MiniMax-H3 --partition fl2va \
  --transformer-path /path/to/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
  --tensor-parallel-size 4 --attention-backend FLASHINFER_SM70 \
  --lora-path ./h3-turbo/minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors \
  --task fl2va --image first.png --image last.png --keyframe-indices 0 -1 \
  --num-frames 39 --output-dir ./h3-fl2va-turbo
```

Omit media and select `--task t2va` for text generation. Omit
`--transformer-path` for the original BF16 checkpoint's FP16/FP32 runtime.
The 39-frame command is a short development check, not the full quality gate.
For Ref2VA, select the Ref2VA partition, checkpoint and `ref2v` adapter together.
Use at least 56 output frames in short tests containing embedded reference
audio: the current pipeline's encoded reference-audio check requires 80 latent
positions, so the 39-frame audio truncation is too short. Official 4–15-second
output requests do not encounter that development-only boundary.

Serving uses the same deployment flags with `vllm video serve`. Native HTTP
accepts JSON, local reference paths, typed URLs/data URLs and multipart uploads.
The application frontend calls this API directly; see [API.md](API.md) for
sync/async requests, multiple outputs, task cleanup and OpenAPI. For example,
on a Ref2VA server with its matching Turbo adapter:

```json
{
  "task": "ref2va",
  "prompt": "图中的纸船继续向右漂移，保留参考视频的环境和流水声。",
  "image": ["/path/to/boat.png"],
  "video": ["/path/to/reference.mp4"],
  "audio": ["/path/to/water.wav"],
  "reference_video_start_times": [0.0],
  "num_frames": 107,
  "lora_scale": 1.0
}
```

## FlashGen four-step T2VA

The native loader accepts the official
[FlashGen artifact](https://modelscope.cn/models/FlashGen/Minimax-H3-4step-lora-flashgen)
named `minimax_h3_t2va_flashgen_4step_v1.0_768p_bf16.safetensors`.
Retain that filename and use the FL2VA base partition with `task=t2va`.
Active FlashGen rejects keyframe and reference tasks. Its rank and alpha are
both 64; metadata supplies the DMD2 schedule `[1, 0.7, 0.4, 0.15, 0]`.
Video/audio flow shifts are 12/3. **FlashGen uses `num_inference_steps=4`**,
counting intervals, unlike LightX2V's five-point API convention. Omitting the
steps and shifts selects these defaults.

```bash
vllm video serve \
  --model /path/to/MiniMax-H3 --partition fl2va \
  --transformer-path /path/to/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
  --tensor-parallel-size 4 --attention-backend FLASHINFER_SM70 \
  --lora-path ./h3-flashgen/minimax_h3_t2va_flashgen_4step_v1.0_768p_bf16.safetensors
```

The adapter has 518 tensors / 259 A/B pairs, including dense AdaLN targets.
The loader converts fused grouped QKV rows to the native TP layout; native
FC1 already has `[gate, up]` order. Every tensor is validated and consumed.
Runtime deltas reuse registered staged buffers and the unrotated-activation
SM70 path below. No dense shortcut can bypass them.

For an AdaLN-pruned INT8 checkpoint, startup restores 106 original AdaLN and
time-embedder tensors before applying FlashGen. The original FL2VA transformer
shards must therefore be present under the model directory. A remote model ID
downloads those shards as well as the shared components. Only restoration
tensors are read from them; backbone INT8 weights and FP32 scales are retained.
This adds approximately **6.1 GiB of weight storage per TP4 rank**, excluding
adapter buffers, activations and other components. This is a tensor-size budget,
not a measured peak. GPU memory and output quality remain unvalidated.

`lora_scale=0` bypasses FlashGen and restores base sampling defaults, but keeps
the restored dense AdaLN/time modules. It does not return to the earlier pruned
base bit for bit. Restart without `--lora-path` to recover that exact deployment.
Original-checkpoint deployments do not need AdaLN restoration.

## FastH3 Dense four-step T2VA

Download the explicit dense adapter from the
[FastVideo release](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA):

```bash
hf download FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA \
  dense-datafree/adapter_model.safetensors \
  --revision f509e629374cac104e7f62daecce6d1488a3041d \
  --local-dir ./fasth3

vllm video serve \
  --model /path/to/MiniMax-H3 --partition fl2va \
  --tensor-parallel-size 4 --attention-backend FLASHINFER_SM70 \
  --lora-path ./fasth3/dense-datafree/adapter_model.safetensors
```

The leaf directory containing that file is also accepted; a repository root
containing several variants is not guessed. Native FastH3 Dense requires the
**original transformer weights**; `--transformer-path` is rejected. Serialized
INT8 fusion and VSA execution remain unimplemented.

This artifact has 809 tensors: 362 rank-64 A/B pairs and 85 full-rank weight/bias
edits, affecting 343 native parameters. The loader validates release identity,
metadata counts, pairing and coverage of all 50 DiT and 2 refiner blocks. It
reconstructs the deltas in FP32, adds them to the original weights, and rounds
to the checkpoint dtype before the usual native FP16/FP32 TP loading. There is
no PEFT alpha scaling. Separate Q/K/V deltas are placed into grouped checkpoint
order; FFN value/gate rows are swapped into native gate/up order.

Fusion runs once during startup, before pinned CPU staging snapshots the model.
It reads one parameter's edits at a time and verifies that all edits reached the
model. Native staging therefore retains fused weights. Actual startup cost and
GPU peak still need measurement; this is not a qualified GPU deployment yet.

Only `task=t2va` is supported. Omitting request steps/shifts selects four
intervals, `num_inference_steps=4`, video/audio shifts 12/3 and the released
schedule `[0.999, 0.749, 0.5, 0.25, 0]`. These positions differ from FlashGen's
four-step schedule. The fused model is immutable: request `lora` selection and
`lora_scale` values other than 1 return errors. Restart without `--lora-path`
to recover the base model. Merely setting the request scale to zero cannot undo
full-rank norm, bias and projection edits.

## Numerical and memory contract

The LightX2V loader consumes all 624 tensors / 312 A/B pairs, covering 50 DiT blocks
and two token-refiner blocks. They bind to 208 native linears because Q/K/V
share one base projection. Q/K/V keep independent A/B pairs and local output
slices. MLP B rows are converted from `[value, gate]` to native `[gate, value]`
before TP partitioning. Row-parallel A matrices are sharded over input channels;
their deltas join the base result before the existing all-reduce.

For a ConvRot layer, the computation is:

```text
base = Linear(ConvRot(x), dequantized_rotated_INT8_weight)
output = base + lora_scale * alpha/rank * B(A(x))
```

The adapter sees the **unrotated** activation. No full weight delta is merged
into signed INT8. FP16 GEMM inputs and FP32 outputs use the existing SM70 H3
operator, including power-of-two scaling for wide-range intermediates. A/B
buffers are registered before the pinned stager snapshots the model, so they
follow its load/offload lifecycle. Dense shortcuts are disabled on adapted
layers so they cannot bypass the delta. FLOP accounting records active LoRA
matmuls separately with `.lora` keys and counts none when scale is zero.

## Validation and remaining work

See [CONTROL.md](CONTROL.md) for the exact environment, source base, commands,
measured results and evidence paths. Passing a parser or synthetic shape test
is not an end-to-end generation result. Each artifact/partition/quantization
combination requires its own evidence before being promoted.

FlashGen's loader, schedule and AdaLN restoration have CPU validation, including
production-sized adapter binding; completed GPU generation remains pending.
FastH3 Dense has native original-weight fusion and CPU validation; its GPU
generation remains pending. FastH3 INT8/VSA, combined-partition serving,
step batching, DLO and approximate caches remain pending in the authorized
[adaptation tracker](ADAPTATION.md). Multipart input and the broader video task
API are implemented, with GPU acceptance tracked separately.
