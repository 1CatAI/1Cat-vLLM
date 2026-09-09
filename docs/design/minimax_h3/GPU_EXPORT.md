# H3 NVENC export measurement

The opt-in `h264_nvenc` route completed a real GPU export on a
V100-SXM2-32GB. For this five-second 720p sample, mean export latency fell
from 1.652 s to 1.377 s (16.6%), while the output file grew by 40.8%.
This is an export-only result, not an end-to-end H3 generation speedup.

## Workload and environment

- Implementation SHA: `642925223d0a97ac48aa598ef6e0559493f71084` (clean tree).
- Integration base: `e7fa44deb253746a11b06be1cd3a75ef7dcdb67c`.
- One V100-SXM2-32GB, physical GPU 0, CUDA-visible device 0. The existing
  GPU-owner queue ran this bounded test serially under its own lease.
- Python 3.12.13, PyTorch 2.10.0+cu128, CUDA 12.8, driver 580.173.02.
- System FFmpeg 6.1.1-3ubuntu5, selected with `IMAGEIO_FFMPEG_EXE`.
- Common input: 120 RGB24 frames at 1280x720 and 24 FPS, resident on the GPU,
  plus the same five-second 32-kHz audio tensor. The frames were decoded from
  the earlier delivered H3 sample, not retained original VAE output tensors.
- Initial frame upload, model startup, denoising, VAE decoding and VAE output
  quantization are excluded. Export includes raw-frame device-to-host copying,
  WAV writing, FFmpeg startup/feed, compression, AAC encoding and MP4 muxing.
- Two runs per encoder, ordered CPU / NVENC / NVENC / CPU. Torch used four
  CPU threads; FFmpeg retained its normal encoder thread defaults.
- CPU: libx264 CRF 18 / medium. GPU: NVENC VBR CQ 18 / p4 / HQ. These are
  different encoder quality controls, not matched bitrate or matched quality.

## Results

| Metric | CPU libx264 | GPU h264_nvenc |
| --- | ---: | ---: |
| Export samples (s) | 1.613, 1.691 | 1.391, 1.363 |
| Mean export (s) | 1.652 | 1.377 |
| Mean frame preparation and copy (s) | 0.209 | 0.292 |
| Mean WAV preparation (s) | 0.017 | 0.016 |
| Mean FFmpeg startup and feeding (s) | 0.954 | 0.804 |
| Mean FFmpeg completion (s) | 0.449 | 0.233 |
| SSIM versus common input | 0.993359 | 0.992695 |
| MP4 size (bytes) | 3,563,768 | 5,018,993 |

Feeding includes pipe backpressure and overlaps encoding. These are sequential
wall intervals, not disjoint CPU/GPU kernel timings; small Python/cleanup costs
remain outside the sub-intervals but inside the export total. RGBA adds GPU
packing and increases raw transfer volume compared with RGB24; both costs are
included in the GPU route's preparation/copy interval.

All four outputs passed automatic frame, dimension, FPS, audio, finite-value,
black-frame and static-frame checks. FFprobe confirms H.264 YUV420P, 120 frames,
24 FPS, and exactly 5.000 s for both video and AAC audio. NVENC samples had
identical size and SSIM. Inspection of frames 48 and 119 found no obvious color
or channel-order corruption; this is not a full human temporal/audio review.
NVML recorded hardware encoder activity, peaking at 15% during this short test.

The current exporter packs RGBA on the source GPU, then uses a host-memory
subprocess pipe. NVENC handles RGB-to-YUV420 conversion and H.264 compression.
This does not eliminate the GPU-to-CPU-to-GPU raw-frame transfer or parallelize
one video across several GPUs. Audio encoding, muxing and file I/O remain on
the CPU. The CPU default is retained; use the explicit deployment option in
[the API guide](API.md#gpu-video-encoding) to select NVENC.

The earlier full-pipeline packaging measurement of 20.56 s was not reproduced
under this isolated contract. It must not be compared directly with 1.377 s
as a claimed speedup. An already-CPU-resident input exported through the original
implementation in 1.298 s and passed checks (SSIM 0.993359), also a different
timing boundary.

## Reproduction and retained evidence

Artifacts are under `/data/minimax-h3/h3-gpu-export-20260908/`:
`benchmark_export.py`, `benchmark-contract.json`, `export-ab.json`,
`encoder-utilization.json`, `quality-comparison.json`, `summary.json`,
`coordinated-probe.json`, and the four `export-*/` output directories.
`runtime-binaries.sha256` records the pre-existing native binaries used for
Python imports; this Python-only change does not claim a new CUDA build.

Run the benchmark only inside an owned GPU lease, with `PYTHONPATH` pointing
to this worktree and `CUDA_VISIBLE_DEVICES` restricted to the allocated GPU.
After GPU work completes, `check_quality.py` performs CPU output validation
and SSIM comparisons. The bounded coordinated job exited successfully and
returned control to its parent lease; no unrelated process was interrupted.

Validation: 36 targeted tests passed, covering export contracts, a real CPU
export, and native HTTP API regressions. Local pre-commit and GitHub checks
passed on the implementation commit; the actual NVENC run provides separate
hardware evidence beyond the CPU tests' simulated encoder process.
