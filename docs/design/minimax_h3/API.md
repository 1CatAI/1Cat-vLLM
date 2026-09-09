# Native H3 API

The application frontend calls the native `vllm video serve` service directly.
There is no ComfyUI dependency or intermediate workflow service. Interactive
request documentation is available at `/docs`, with JSON and multipart schemas
in `/openapi.json`.

## Start the service

Use the model, transformer, parallelism and operator settings in
[the native deployment guide](README.md). For example:

```bash
vllm video serve \
  --model /path/to/MiniMax-H3 \
  --partition fl2va \
  --transformer-path /path/to/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
  --tensor-parallel-size 4 --attention-backend FLASHINFER_SM70 \
  --host 127.0.0.1 --port 8000 --output-dir ./h3-jobs
```

This remains one partition per engine. For reference generation, select
`--partition ref2va` and the corresponding transformer. Keep a stable public API
URL in the frontend's own reverse proxy when deploying it separately.

### GPU video encoding

Both `vllm video generate` and `vllm video serve` accept
`--video-encoder h264_nvenc`. Select an FFmpeg executable built with NVENC:

```bash
IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg vllm video serve \
  --model /path/to/MiniMax-H3 \
  --transformer-path /path/to/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
  --tensor-parallel-size 4 --video-encoder h264_nvenc \
  --host 127.0.0.1 --port 8000 --output-dir ./h3-jobs
```

The encoder runs on the rank-0 GPU inside the engine's leased group. RGBA
packing runs on the input tensor's device; NVENC handles RGB-to-YUV420 conversion
and H.264 compression. The current subprocess pipe still transfers raw frames
through host memory. Audio encoding, MP4 muxing, and file I/O remain on the CPU.
This does not distribute one video's encoding across the DiT tensor-parallel
group. See [NVIDIA's FFmpeg guide](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/ffmpeg-with-nvidia-gpu/index.html).

The default remains `libx264` with CRF 18 / medium. NVENC uses VBR CQ 18 / p4 /
HQ; equal numeric quality settings do not guarantee equal bitrate or quality.
The selected FFmpeg must support `h264_nvenc`, packed RGBA, and `-rgb_mode`.
Some imageio-ffmpeg bundled executables omit NVENC. A requested NVENC export
fails with an `encode.log` diagnostic if unavailable; it never silently switches
to CPU encoding.

Rank-0 run metadata records `video_encoder`, `encoding_device`,
`preparation_device`, `ffmpeg_executable`, and `export_stage_seconds`.
The latter separates frame preparation/copy, WAV writing, FFmpeg startup/feed,
and encoder/muxer completion. Feed time includes pipe backpressure and overlaps
encoding, so these are sequential wall intervals rather than GPU kernel timings.
The frontend video API and output audio/video contract are unchanged.
See [the V100 export measurement](GPU_EXPORT.md) for measured latency, output
quality, file-size tradeoffs, and the remaining host-memory transfer cost.

## Endpoints

| Method | Path | Result |
| --- | --- | --- |
| GET | `/v1/models` | Served model identifier |
| POST | `/v1/videos` | HTTP 202 with a video job ID |
| GET | `/v1/videos/{id}` | Status, output metadata and download URLs |
| GET | `/v1/videos` | Job list; `limit`, `after`, `order` pagination |
| GET | `/v1/videos/{id}/content` | First MP4, or `?output_index=N` for another output |
| DELETE | `/v1/videos/{id}` | Delete queued or finished job and its owned files |
| POST | `/v1/videos/sync` | Block and return the first MP4 directly |
| GET | `/health` | Engine readiness |
| GET | `/metrics` | Queue and job counters |

Generation is serial within an engine. Asynchronous jobs progress through
`queued`, `in_progress`, and `completed` or `failed`. The progress value counts
completed outputs, not denoising steps. Deleting a running job returns HTTP 409;
it does not terminate a distributed worker group. Downloading an unfinished
job also returns 409. Unknown IDs or output indices return 404.

Job records live in memory for the process lifetime. Asynchronous outputs stay
on disk until deleted. The synchronous endpoint does not publish a job in the
list and removes its temporary results after sending the MP4. Uploaded and
downloaded reference files are request-owned and removed after success, failure,
or rejection. Original local reference files are never deleted.

## JSON requests from the frontend

```javascript
const submitted = await fetch("/v1/videos", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    prompt: "A red paper boat floats across a pond with gentle water sounds.",
    task: "t2va",
    width: 1344,
    height: 768,
    duration: 4.4,
    seed: 42,
    num_outputs_per_prompt: 2,
  }),
});
if (!submitted.ok) throw new Error(await submitted.text());
const { id } = await submitted.json();

// Poll this URL until status is completed or failed.
const response = await fetch(`/v1/videos/${id}`);
if (!response.ok) throw new Error(await response.text());
const job = await response.json();
if (job.status === "completed") {
  // Each output.url is a relative MP4 download URL for the video player.
  console.log(job.outputs.map((output) => output.url));
} else if (job.status === "failed") {
  throw new Error(job.error.message);
}
```

`num_outputs_per_prompt` accepts 1–10; output N uses `seed + N`. Each output
runs through the same native engine independently. This is multiple sampling,
not DiT co-batching. The sync endpoint performs the requested outputs but returns
only the first, matching the official raw-MP4 behavior; use async for all outputs.

`duration` and `seconds` are aliases. Upstream `extra_params.duration` is also
accepted. Contradictory values are rejected. Frame counts align upward to H3's
17n+5 constraint; 4.4 requested seconds produces 107 frames at 24 FPS. Width and
height must be supplied together and be multiples of 32. `size="1344x768"` is an
alternative. With dimensions omitted, the native canvas planner uses the image
ratio for FL2VA and the selected/default ratio for other tasks.

## Reference media

For remote clients, JSON accepts `image_reference`, `video_reference`, and
`audio_reference`. Each field accepts one typed object or a list. The URL value
can be HTTP(S) or a base64 data URL:

```json
{
  "prompt": "The subject moves naturally while preserving the reference sound.",
  "task": "ref2va",
  "duration": 4.4,
  "image_reference": [{"image_url": "https://assets.example/subject.png"}],
  "video_reference": [{"video_url": "https://assets.example/motion.mp4"}],
  "audio_reference": [{"audio_url": "https://assets.example/voice.wav"}],
  "reference_video_start_times": [0.0]
}
```

For browser uploads, repeat the `input_references` multipart field. The server
uses MIME types to classify images, videos and audio, retaining per-type order:

```javascript
const form = new FormData();
form.set("prompt", "Move naturally between the first and last images.");
form.set("extra_params", JSON.stringify({
  task: "fl2va", frame_indices: [0, -1], duration: 4.4,
}));
form.append("input_references", firstImageFile);
form.append("input_references", lastImageFile);
const response = await fetch("/v1/videos", { method: "POST", body: form });
if (!response.ok) throw new Error(await response.text());
const job = await response.json();
```

The singular `input_reference` is accepted too. Ref2VA supports all combinations
in [the workflow matrix](WORKFLOWS.md), provided at least one visual reference is
present. Existing JSON `image`, `video` and `audio` arrays refer to server-local
paths. Remote frontends should use uploads or typed URLs.

Per-file limits are 30 MiB for images, 50 MiB for videos and 15 MiB for audio.
The existing model validator checks formats, reference dimensions and durations,
counts, and segment offsets before GPU dispatch. Bad references return 422 and
leave the engine usable.

## Distilled LoRA requests

Start with a supported adapter in `--lora-path`. JSON and multipart use the same
default: the loaded adapter is active at scale 1, and omitted steps/shifts follow
its contract. For LightX2V and FlashGen, `lora_scale=0` bypasses the adapter and
restores base sampling defaults. Fused FastH3 Dense requires scale 1.
Adding a `model` or `extra_params` field never changes adapter activation.

For LightX2V/FlashGen, the optional `lora` object accepts `path` (or `local_path`), `name`, `scale`, and
`int_id`. Its path must resolve to the preloaded adapter. It selects that adapter
and scale; arbitrary hot loading is not yet implemented. This native startup
default differs from upstream PEFT's preload-only behavior.

```json
{
  "prompt": "A red paper boat floats across a pond.",
  "task": "t2va",
  "duration": 4.4,
  "num_inference_steps": 5,
  "flow_shift": 6,
  "audio_flow_shift": 3,
  "lora_scale": 1
}
```

The example assumes FL2V four-step 768p Turbo. Ref2V and other artifacts need
their own task and sampling contract. Four-step LightX2V means five sigma points;
eight-step means nine. Unsupported adapters or mismatched settings are rejected.

On a server started with the official FlashGen adapter, use `task=t2va` and
`num_inference_steps=4`, with video/audio shifts 12/3, or omit steps/shifts to
select these defaults. Its four intervals use the schedule stored in adapter
metadata. Active FlashGen rejects FL2VA keyframes and Ref2VA references.
For FlashGen over pruned INT8, scale zero keeps the restored original AdaLN/time
modules; restart without the adapter to return to the pruned base. See
[FlashGen deployment requirements](WORKFLOWS.md#flashgen-four-step-t2va) for
original-weight and memory requirements. CPU validation has passed; actual
FlashGen GPU generation and quality acceptance remain pending.

FastH3 Dense uses `task=t2va`, `num_inference_steps=4` and video/audio shifts
12/3, also selected when omitted. Its released sigma positions are
`[0.999, 0.749, 0.5, 0.25, 0]`. It is fused at startup: per-request `lora` and
`lora_scale` values other than 1 return HTTP 422. Restart without the adapter
to recover the base model. Use the original weights; native INT8 fusion and
VSA variants remain unsupported. See [FastH3 deployment](WORKFLOWS.md#fasth3-dense-four-step-t2va).
GPU generation and quality acceptance remain pending.

## Scope and validation

The interface changes have CPU coverage for real media parsing and fake-engine
job execution, including 11 input combinations, cleanup, downloads, multi-output
seeds and OpenAPI. These tests do not establish GPU generation or video quality.
Current model evidence and pending acceptance are recorded in [CONTROL.md](CONTROL.md).

`quality=lossless` selects the existing reference path. `high`, cache policies,
FastH3 VSA, combined partition routing and additional execution modes remain
tracked implementation work in [ADAPTATION.md](ADAPTATION.md), not accepted
no-op parameters. Generic neutral CFG fields are accepted at 1; H3 has no
negative CFG branch. Native VAE tiling remains deployment-controlled.
