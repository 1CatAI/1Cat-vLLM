# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Export H3 video and its synchronous 32 kHz audio."""

import subprocess
import time
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import soundfile as sf


def export_video(video, audio, output_dir, *, fps=24, encoder="libx264", gpu_index=0):
    if encoder not in ("libx264", "h264_nvenc"):
        raise ValueError(f"unsupported video encoder {encoder!r}")
    nvenc = encoder == "h264_nvenc"
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    stages = {}
    started = time.perf_counter()
    # Pipeline video is B,T,H,W,C, audio is B,C,samples.
    video = video.detach()
    if video.ndim != 5 or video.shape[0] != 1 or video.shape[-1] != 3:
        raise ValueError(f"unexpected H3 video shape {tuple(video.shape)}")
    if audio.ndim != 3 or audio.shape[0] != 1 or audio.shape[1] not in (1, 2):
        raise ValueError(f"unexpected H3 audio shape {tuple(audio.shape)}")
    preparation_device = str(video.device)
    if nvenc:
        # NVENC accepts packed RGBA and converts it to YUV420 on the GPU.
        # Pack on the source device before the subprocess's host-memory pipe;
        # do not request FFmpeg's CPU RGB-to-YUV conversion with -pix_fmt.
        rgba = video.new_full((*video.shape[:-1], 4), 255)
        rgba[..., :3].copy_(video)
        video = rgba
    video = video.cpu().contiguous()
    audio = audio.detach().float().cpu()
    stages["prepare_and_copy"] = time.perf_counter() - started
    started = time.perf_counter()
    if not np.isfinite(audio.numpy()).all():
        raise ValueError("H3 produced non-finite audio")
    waveform = audio[0].numpy().T
    wav = root / "audio.wav"
    sf.write(wav, waveform, 32000, subtype="FLOAT")
    stages["audio_wav"] = time.perf_counter() - started
    _, frames, height, width, _ = video.shape
    output = root / "video.mp4"
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    codec_args = (
        [
            "-c:v",
            "h264_nvenc",
            "-gpu",
            str(gpu_index),
            "-preset",
            "p4",
            "-tune",
            "hq",
            "-rc",
            "vbr",
            "-cq",
            "18",
            "-b:v",
            "0",
            "-rgb_mode",
            "yuv420",
        ]
        if nvenc
        else [
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
        ]
    )
    command = [
        ffmpeg,
        "-nostdin",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgba" if nvenc else "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-i",
        str(wav),
        "-map",
        "0:v",
        "-map",
        "1:a",
        *codec_args,
        "-c:a",
        "aac",
        "-b:a",
        "320k",
        "-movflags",
        "+faststart",
        str(output),
    ]
    started = time.perf_counter()
    with (root / "encode.log").open("wb") as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        try:
            assert process.stdin is not None
            for frame in video[0]:
                process.stdin.write(memoryview(frame.numpy()).cast("B"))
            process.stdin.close()
            # Pipe feeding overlaps encoder work and includes backpressure.
            # These are sequential wall intervals, not CPU/GPU kernel times.
            stages["ffmpeg_start_and_feed"] = time.perf_counter() - started
            started = time.perf_counter()
            if process.wait() != 0:
                raise RuntimeError(
                    f"ffmpeg video export failed; see {root / 'encode.log'}"
                )
            stages["ffmpeg_finish"] = time.perf_counter() - started
        except BaseException as exc:
            process.kill()
            process.wait()
            if isinstance(exc, BrokenPipeError):
                raise RuntimeError(
                    f"ffmpeg {encoder} export failed; see {root / 'encode.log'}"
                ) from exc
            raise
    return {
        "video": str(output),
        "audio": str(wav),
        "frames": frames,
        "width": width,
        "height": height,
        "fps": fps,
        "video_encoder": encoder,
        "encoding_device": f"cuda:{gpu_index}" if nvenc else "cpu",
        "preparation_device": preparation_device,
        "ffmpeg_executable": ffmpeg,
        "export_stage_seconds": stages,
    }
