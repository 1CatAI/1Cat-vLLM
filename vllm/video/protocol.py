# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Normalize native JSON and Omni video forms before worker admission."""

from __future__ import annotations

import binascii
import json
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote_to_bytes

import httpx
import pybase64 as base64
from fastapi import Request
from pydantic import BaseModel, ConfigDict, Field
from starlette.datastructures import UploadFile

from vllm.model_executor.models.minimax_h3.config import (
    DEFAULT_PROMPT,
    H3Config,
    H3InputError,
    H3Request,
    sampling_for_deployment,
)

_MIME = {
    "image/jpeg": ("image", ".jpg"),
    "image/png": ("image", ".png"),
    "image/webp": ("image", ".webp"),
    "image/heic": ("image", ".heic"),
    "image/heif": ("image", ".heif"),
    "video/mp4": ("video", ".mp4"),
    "video/quicktime": ("video", ".mov"),
    "audio/wav": ("audio", ".wav"),
    "audio/x-wav": ("audio", ".wav"),
    "audio/wave": ("audio", ".wav"),
    "audio/mpeg": ("audio", ".mp3"),
    "audio/mp3": ("audio", ".mp3"),
}
_LIMIT = {"image": 30 * 1024**2, "video": 50 * 1024**2, "audio": 15 * 1024**2}
_REFERENCE_FIELDS = {
    "input_reference": None,
    "input_references": None,
    "image_reference": "image",
    "video_reference": "video",
    "audio_reference": "audio",
}


class RemoteLoRA(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = ""
    path: str | None = None
    local_path: str | None = None
    scale: float = Field(default=1.0, allow_inf_nan=False)
    int_id: int = 0


class VideoRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str | None = None
    prompt: str = DEFAULT_PROMPT
    width: int | None = None
    height: int | None = None
    size: str | None = None
    aspect_ratio: str | None = None
    short_edge: Literal[768] = 768
    fps: Literal[24] = 24
    num_frames: int = 243
    duration: float | None = Field(default=None, allow_inf_nan=False)
    seconds: float | None = Field(default=None, allow_inf_nan=False)
    seed: int = 42
    num_inference_steps: int | None = None
    num_outputs_per_prompt: int = Field(default=1, ge=1, le=10)
    task: Literal["t2va", "fl2va", "ref2va"] | None = None
    quality: Literal["lossless", "high"] | None = None
    lora: RemoteLoRA | None = None
    lora_scale: float | None = Field(default=None, allow_inf_nan=False)
    flow_shift: float | None = None
    audio_flow_shift: float | None = None
    image: list[str] = Field(default_factory=list)
    video: list[str] = Field(default_factory=list)
    audio: list[str] = Field(default_factory=list)
    keyframe_indices: list[int] | None = None
    reference_video_start_times: list[float] | None = None
    extra_params: dict[str, Any] = Field(default_factory=dict)
    # H3 is CFG-distilled. These neutral fields occur in generic Omni clients.
    guidance_scale: float = Field(default=1.0, ge=1.0, le=1.0)
    true_cfg_scale: float = Field(default=1.0, ge=1.0, le=1.0)
    negative_prompt: Literal[""] = ""
    vae_use_slicing: Literal[False] = False
    vae_use_tiling: bool = Field(
        default=False,
        description="Compatibility field; native VAE tiling is deployment-controlled",
    )

    def request(self, config: H3Config) -> H3Request:
        from vllm.model_executor.models.minimax_h3.preprocessing import (
            load_minimax_h3_images,
            resolve_minimax_h3_aspect_ratio,
            resolve_minimax_h3_output_canvas,
        )

        if self.model is not None and self.model not in (
            config.model,
            "MiniMaxAI/MiniMax-H3",
        ):
            raise H3InputError(f"model is not served: {self.model}")
        extra = dict(self.extra_params)
        aliases = {
            "task": "task",
            "duration": "duration_seconds",
            "seconds": "duration_seconds",
            "flow_shift": "flow_shift",
            "audio_flow_shift": "audio_flow_shift",
            "keyframe_indices": "frame_indices",
            "reference_video_start_times": "start_time_seconds",
            "aspect_ratio": "aspect_ratio",
        }
        if "duration" in extra:
            if (
                "duration_seconds" in extra
                and extra["duration"] != extra["duration_seconds"]
            ):
                raise H3InputError("conflicting duration fields")
            extra["duration_seconds"] = extra.pop("duration")
        for field, key in aliases.items():
            value = getattr(self, field)
            if value is not None:
                if key in extra and extra[key] != value:
                    raise H3InputError(f"conflicting {field} and extra_params.{key}")
                extra[key] = value
        allowed = set(aliases.values()) | {
            "short_edge",
            "force_refresh_step_hint",
            "force_refresh_step_policy",
        }
        if unknown := extra.keys() - allowed:
            raise H3InputError(f"unsupported H3 extra_params: {sorted(unknown)}")
        if extra.get("short_edge", 768) != 768:
            raise H3InputError("H3 short_edge must be 768")
        scale = self.lora_scale
        if self.lora is not None:
            if config.lora_path:
                from vllm.model_executor.models.minimax_h3.fasth3 import FastH3Spec
                from vllm.model_executor.models.minimax_h3.lora import (
                    inspect_deployment_adapter,
                )

                if isinstance(inspect_deployment_adapter(config), FastH3Spec):
                    raise H3InputError(
                        "FastH3 is fused; per-request lora is unavailable"
                    )
            if (
                self.lora.path is not None
                and self.lora.local_path is not None
                and Path(self.lora.path).resolve()
                != Path(self.lora.local_path).resolve()
            ):
                raise H3InputError("conflicting lora.path and lora.local_path")
            path = self.lora.path or self.lora.local_path
            if not path or not config.lora_path:
                raise H3InputError("request LoRA requires a preloaded adapter path")
            from vllm.model_executor.models.minimax_h3.lora import select_adapter_file

            deployed = select_adapter_file(config.lora_path)
            requested = select_adapter_file(path)
            if deployed.resolve() != requested.resolve():
                raise H3InputError("requested LoRA is not loaded on this server")
            if scale is not None and scale != self.lora.scale:
                raise H3InputError("conflicting lora.scale and lora_scale")
            scale = self.lora.scale
        if scale is None:
            scale = 1.0
        width, height = self.width, self.height
        if self.size:
            try:
                size_width, size_height = map(int, self.size.lower().split("x"))
            except ValueError as exc:
                raise H3InputError("size must be WIDTHxHEIGHT") from exc
            if (width is not None and width != size_width) or (
                height is not None and height != size_height
            ):
                raise H3InputError("size conflicts with width/height")
            width, height = size_width, size_height
        if (width is None) != (height is None):
            raise H3InputError("width and height must be supplied together")
        if width is None:
            task = extra.get("task") or (
                "ref2va"
                if config.partition == "ref2va" or self.video or self.audio
                else "fl2va"
                if self.image
                else "t2va"
            )
            image = load_minimax_h3_images(self.image[:1])[0] if self.image else None
            ratio = resolve_minimax_h3_aspect_ratio(
                task, extra.get("aspect_ratio", "16:9"), image
            )
            height, width = resolve_minimax_h3_output_canvas(ratio, self.short_edge)
        return H3Request(
            prompt=self.prompt,
            sampling=sampling_for_deployment(
                config,
                width=width,
                height=height,
                fps=self.fps,
                num_frames=self.num_frames,
                seed=self.seed,
                num_inference_steps=self.num_inference_steps,
                lora_scale=scale,
                quality=self.quality,
                extra_args=extra,
            ),
            media={
                key: getattr(self, key)
                for key in ("image", "video", "audio")
                if getattr(self, key)
            },
        )


def _media_type(mime: str, expected: str | None) -> tuple[str, str]:
    mime = mime.partition(";")[0].strip().lower()
    if mime not in _MIME:
        raise H3InputError(f"unsupported reference content type: {mime}")
    kind, suffix = _MIME[mime]
    if expected is not None and kind != expected:
        raise H3InputError(f"expected {expected} reference, got {mime}")
    return kind, suffix


async def _save_reference(value: Any, expected: str | None, root: Path, index: int):
    if isinstance(value, UploadFile):
        kind, suffix = _media_type(value.content_type or "", expected)
        data = await value.read(_LIMIT[kind] + 1)
    else:
        if isinstance(value, dict):
            candidates = [
                (key.removesuffix("_url"), value[key])
                for key in ("image_url", "video_url", "audio_url")
                if key in value
            ]
            if len(candidates) != 1:
                raise H3InputError(
                    "reference must have one image_url, video_url or audio_url"
                )
            kind, value = candidates[0]
            if expected is not None and expected != kind:
                raise H3InputError("reference kind does not match field")
            expected = kind
            if isinstance(value, dict):
                value = value.get("url")
        if not isinstance(value, str):
            raise H3InputError("reference must contain a URL or data URL")
        if value.startswith("data:"):
            header, separator, payload = value.partition(",")
            if not separator:
                raise H3InputError("invalid reference data URL")
            kind, suffix = _media_type(header[5:], expected)
            if len(payload) > _LIMIT[kind] * 4 // 3 + 4:
                raise H3InputError("reference exceeds size limit")
            try:
                data = (
                    base64.b64decode(payload, validate=True)
                    if header.endswith(";base64")
                    else unquote_to_bytes(payload)
                )
            except (ValueError, binascii.Error) as exc:
                raise H3InputError("invalid reference base64") from exc
        elif value.startswith(("http://", "https://")):
            async with (
                httpx.AsyncClient(follow_redirects=True, timeout=30) as client,
                client.stream("GET", value) as response,
            ):
                response.raise_for_status()
                kind, suffix = _media_type(
                    response.headers.get("content-type", ""), expected
                )
                chunks = bytearray()
                async for chunk in response.aiter_bytes():
                    chunks.extend(chunk)
                    if len(chunks) > _LIMIT[kind]:
                        raise H3InputError("reference exceeds size limit")
                data = bytes(chunks)
        else:
            raise H3InputError("reference URL must use http, https or data")
    if not data or len(data) > _LIMIT[kind]:
        raise H3InputError("reference is empty or exceeds size limit")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{index}{suffix}"
    path.write_bytes(data)
    return kind, str(path)


async def parse_video_request(request: Request, root: Path) -> VideoRequest:
    """Keep staged media alive until completion; caller owns root cleanup."""
    content_type = request.headers.get("content-type", "").partition(";")[0]
    if content_type == "application/json":
        body = await request.json()
        if not isinstance(body, dict):
            raise H3InputError("video request must be an object")
        pairs = list(body.items())
    elif content_type in ("multipart/form-data", "application/x-www-form-urlencoded"):
        form = await request.form(
            max_files=12, max_fields=64, max_part_size=70 * 1024**2
        )
        pairs = list(form.multi_items())
    else:
        raise H3InputError("expected JSON or multipart video request")
    body = {}
    reference_count = 0
    try:
        for key, value in pairs:
            if key in _REFERENCE_FIELDS:
                if isinstance(value, str) and value.lstrip().startswith(("{", "[")):
                    value = json.loads(value)
                references = value if isinstance(value, list) else [value]
                for reference in references:
                    reference_count += 1
                    if reference_count > 12:
                        raise H3InputError("at most 12 uploaded references")
                    kind, path = await _save_reference(
                        reference, _REFERENCE_FIELDS[key], root, reference_count
                    )
                    body.setdefault(kind, []).append(path)
            else:
                if key in body:
                    raise H3InputError(f"duplicate field: {key}")
                if isinstance(value, UploadFile):
                    raise H3InputError(f"unexpected file field: {key}")
                if isinstance(value, str) and key in (
                    "extra_params",
                    "lora",
                    "keyframe_indices",
                    "reference_video_start_times",
                ):
                    value = json.loads(value)
                if isinstance(value, str) and key in {
                    "short_edge",
                    "fps",
                    "guidance_scale",
                    "true_cfg_scale",
                    "vae_use_slicing",
                    "vae_use_tiling",
                }:
                    value = json.loads(value.lower())
                body[key] = value
        return VideoRequest.model_validate(body)
    finally:
        if content_type != "application/json":
            await form.close()
