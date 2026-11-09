# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate client metadata before it reaches distributed H3 workers."""

from os import PathLike
from pathlib import Path
from subprocess import CalledProcessError
from types import SimpleNamespace

from .config import H3Config, H3InputError, H3Request


def validate_request(config: H3Config, request: H3Request) -> None:
    # Reuse the pipeline's reference contract, without constructing a model or
    # touching CUDA. Media encoding remains in the distributed worker stage.
    from .pipeline import (
        MiniMaxH3Pipeline,
        _resolve_fl2va_keyframe_indices,
        _validate_ref2va_reference_counts,
        _validate_reference_image,
    )
    from .preprocessing import load_minimax_h3_images

    supported = {"t2va", "fl2va"} if config.partition == "fl2va" else {"ref2va"}
    deployment = SimpleNamespace(partition=config.partition, supported_tasks=supported)
    task = MiniMaxH3Pipeline._resolve_task(
        deployment, request.sampling.extra_args.get("task"), request.media
    )
    if config.lora_path:
        from .lora import inspect_deployment_adapter, validate_adapter_sampling

        validate_adapter_sampling(
            inspect_deployment_adapter(config),
            task,
            request.sampling,
        )
    references = {}
    for key in ("image", "video", "audio"):
        value = request.media.get(key)
        values = (
            []
            if value is None
            else (list(value) if isinstance(value, (list, tuple)) else [value])
        )
        if value is not None and not values:
            raise H3InputError(f"{key} references must not be empty")
        if key == "audio" and len(values) == 2 and isinstance(values[1], int):
            values = [value]  # A waveform/sample-rate pair is one reference.
        for item in values:
            if isinstance(item, (str, PathLike)) and not Path(item).is_file():
                raise H3InputError(f"{key} reference does not exist: {item}")
        references[key] = values

    counts = [len(references[key]) for key in ("image", "video", "audio")]
    if task == "ref2va":
        _validate_ref2va_reference_counts(*counts)
        from .pipeline import _load_audios
        from .reference_video import (
            validate_reference_audio_files,
            validate_reference_audio_waveforms,
            validate_reference_video_files,
        )

        try:
            if references["video"]:
                validate_reference_video_files(
                    references["video"],
                    start_time_seconds=request.sampling.extra_args.get(
                        "start_time_seconds"
                    ),
                )
            if references["audio"]:
                validate_reference_audio_files(request.media["audio"])
                validate_reference_audio_waveforms(_load_audios(request.media["audio"]))
        except (OSError, CalledProcessError, ValueError) as exc:
            raise H3InputError(f"invalid reference media: {exc}") from exc
    elif task == "t2va" and any(counts):
        raise H3InputError("t2va does not accept reference media")
    elif task == "fl2va":
        if not 1 <= counts[0] <= 2 or counts[1] or counts[2]:
            raise H3InputError("fl2va accepts one or two image keyframes only")
        _resolve_fl2va_keyframe_indices(request.sampling.extra_args, counts[0])
    if task != "fl2va" and request.sampling.extra_args.get("frame_indices") is not None:
        raise H3InputError("keyframe indices require image keyframes")
    if (
        request.sampling.extra_args.get("start_time_seconds") is not None
        and not counts[1]
    ):
        raise H3InputError("reference start times require video references")
    images = []
    try:
        if references["image"]:
            images = load_minimax_h3_images(references["image"])
            for image in images:
                _validate_reference_image(image)
    except (OSError, ValueError) as exc:
        raise H3InputError(f"invalid reference image: {exc}") from exc
    MiniMaxH3Pipeline._resolve_shape(
        deployment, task, request.sampling, images[0] if images else None
    )
