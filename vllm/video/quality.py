# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic media checks; human video-quality review remains a separate gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image


def inspect_video(
    path,
    *,
    expected_frames=243,
    expected_width=1344,
    expected_height=768,
    expected_fps=24,
):
    path = Path(path)
    frame_means = []
    differences = []
    black_frames = []
    previous = None
    sizes = set()
    screenshots = []
    folder = path.parent / "frames"
    folder.mkdir(exist_ok=True)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        for index, frame in enumerate(container.decode(stream)):
            rgb = frame.to_ndarray(format="rgb24")
            sizes.add((frame.width, frame.height))
            sampled = rgb[::8, ::8].astype(np.float32)
            mean = float(sampled.mean())
            frame_means.append(mean)
            if mean < 1 and float(sampled.std()) < 1:
                black_frames.append(index)
            if previous is not None:
                differences.append(float(np.abs(sampled - previous).mean()))
            previous = sampled
            if index % (expected_fps * 2) == 0 or index == expected_frames - 1:
                output = folder / f"frame-{index:04d}.png"
                Image.fromarray(rgb).save(output)
                screenshots.append(str(output))
    samples = 0
    audio_rates = set()
    audio_energy = 0.0
    audio_finite = True
    with av.open(str(path)) as container:
        if container.streams.audio:
            for frame in container.decode(audio=0):
                array = frame.to_ndarray().astype(np.float64)
                audio_finite = audio_finite and bool(np.isfinite(array).all())
                audio_energy += float(np.square(array).sum())
                samples += frame.samples
                audio_rates.add(frame.sample_rate)
    longest_static = current_static = 0
    for difference in differences:
        current_static = current_static + 1 if difference < 0.02 else 0
        longest_static = max(longest_static, current_static)
    audio_duration = (
        samples / next(iter(audio_rates)) if len(audio_rates) == 1 else None
    )
    checks = {
        "frame_count": len(frame_means) == expected_frames,
        "dimensions": sizes == {(expected_width, expected_height)},
        "fps": abs(fps - expected_fps) < 0.001,
        "audio_present_finite_nonzero": samples > 0
        and audio_finite
        and audio_energy > 0,
        "audio_duration": audio_duration is not None
        and abs(audio_duration - expected_frames / expected_fps) < 0.15,
        "no_black_frames": not black_frames,
        "no_prolonged_static_run": longest_static < expected_fps,
    }
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    result = {
        "checks": checks,
        "automatic_passed": all(checks.values()),
        "frames": len(frame_means),
        "sizes": sorted(sizes),
        "fps": fps,
        "audio_duration_seconds": audio_duration,
        "black_frames": black_frames,
        "longest_static_run": longest_static,
        "frame_mean": frame_means,
        "adjacent_frame_difference": differences,
        "screenshots": screenshots,
        "sha256": digest.hexdigest(),
        "human_review": {
            "status": "pending",
            "scores": {
                key: None
                for key in (
                    "prompt_following",
                    "subject_consistency",
                    "temporal_continuity",
                    "detail",
                    "audio",
                )
            },
        },
    }
    (path.parent / "quality.json").write_text(json.dumps(result, indent=2))
    return result
