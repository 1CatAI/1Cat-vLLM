# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request and deployment contracts for native MiniMax H3."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

BASE_MODEL = "MiniMaxAI/MiniMax-H3"
BASE_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
COMFY_MODEL = "Comfy-Org/MiniMax-H3"
COMFY_REVISION = "a98869194787969724c7425d95d0ed73ce9202af"
# The streaming VAE needs at least one full temporal chunk (22 frames).
# Short clips are useful development workloads; acceptance stays at 243 frames.
MIN_OUTPUT_FRAMES = 22
MAX_OUTPUT_FRAMES = 362
MIN_OUTPUT_SECONDS = MIN_OUTPUT_FRAMES / 24
MAX_OUTPUT_SECONDS = 15
DEFAULT_PROMPT = (
    "一个连续的写实镜头。白天自然光下的公园浅水池，一艘红色纸船从画面左侧"
    "缓慢漂向右侧，一只黄色橡皮鸭从纸船后方经过。微风在水面形成细小涟漪，"
    "背景绿树保持稳定。镜头缓慢向前推进，纸船与橡皮鸭的颜色、形状和数量"
    "始终一致。音轨包含轻柔流水声和远处鸟鸣，无人声、无音乐、无字幕。"
)


class H3InputError(ValueError):
    """Invalid media or generation parameters, safe to report to a client."""


@dataclass(frozen=True)
class H3Config:
    model: str = BASE_MODEL
    revision: str = BASE_REVISION
    partition: Literal["fl2va", "ref2va"] = "fl2va"
    transformer_path: str | None = None
    tensor_parallel_size: int = 4
    attention_backend: str = "FLASH_ATTN_V100"
    vsa_topk: int = 64
    attention_query_tile: Literal[64, 128] = 64
    fp16_weight_cache_gib: float = 0.0
    fp16_cache_layers: tuple[str, ...] = ()
    lora_path: str | None = None
    int8_weight_layout: str = "column"
    fp16_weight_layout: Literal["row", "column"] = "row"
    residual_sequence_parallel: bool = False
    residual_reduction: Literal["native", "peer"] = "native"
    residual_reduction_memory_gib: float = 4.0
    host_weight_pin_memory: bool = True
    share_host_vae_weights: bool = False
    weight_offload: Literal["component", "layer"] = "component"
    video_encoder: Literal["libx264", "h264_nvenc"] = "libx264"

    def __post_init__(self) -> None:
        if self.residual_reduction not in ("native", "peer"):
            raise H3InputError("residual reduction must be native or peer")
        if self.residual_reduction == "peer" and not self.residual_sequence_parallel:
            raise H3InputError("peer reduction requires residual sequence parallelism")
        if (
            isinstance(self.residual_reduction_memory_gib, bool)
            or not math.isfinite(self.residual_reduction_memory_gib)
            or not math.isfinite(self.residual_reduction_memory_gib * 2**30)
            or self.residual_reduction_memory_gib <= 0
        ):
            raise H3InputError("residual communication budget must be finite and > 0")
        if self.attention_query_tile not in (64, 128):
            raise H3InputError("Attention query tile must be 64 or 128")
        if (
            self.attention_query_tile != 64
            and self.attention_backend != "FLASH_ATTN_V100"
        ):
            raise H3InputError("Explicit query tiling requires FLASH_ATTN_V100")
        if self.weight_offload not in ("component", "layer"):
            raise H3InputError("weight offload must be component or layer")
        if self.weight_offload == "layer" and self.fp16_cache_layers:
            raise H3InputError("layer offload cannot retain a fixed GPU weight cache")
        if not isinstance(self.host_weight_pin_memory, bool):
            raise H3InputError("host weight pinning must be a boolean")
        if not isinstance(self.share_host_vae_weights, bool):
            raise H3InputError("shared host VAE weights must be a boolean")
        if (
            self.share_host_vae_weights
            and self.tensor_parallel_size > 1
            and self.host_weight_pin_memory
        ):
            raise H3InputError("shared host VAE weights require pageable host masters")
        if self.video_encoder not in ("libx264", "h264_nvenc"):
            raise H3InputError("video encoder must be libx264 or h264_nvenc")
        if self.partition not in ("fl2va", "ref2va"):
            raise H3InputError("partition must be fl2va or ref2va")
        if self.int8_weight_layout not in ("row", "column"):
            raise H3InputError("H3 INT8 weight layout must be row or column")
        if self.fp16_weight_layout not in ("row", "column"):
            raise H3InputError("H3 FP16 weight layout must be row or column")
        if self.tensor_parallel_size not in (1, 2, 4):
            raise H3InputError("native H3 supports TP1, TP2, or TP4")
        if self.attention_backend not in (
            "FLASH_ATTN_V100",
            "FLASHINFER_SM70",
            "TORCH_SDPA",
            "FASTVIDEO_VSA",
        ):
            raise H3InputError(f"unsupported H3 attention: {self.attention_backend}")
        if (
            isinstance(self.vsa_topk, bool)
            or not isinstance(self.vsa_topk, int)
            or self.vsa_topk <= 0
        ):
            raise H3InputError("VSA topk must be a positive integer")
        if self.attention_backend == "FASTVIDEO_VSA" and not self.lora_path:
            raise H3InputError("H3 VSA requires an explicit FastH3 VSA artifact")
        if (
            not math.isfinite(self.fp16_weight_cache_gib)
            or self.fp16_weight_cache_gib < 0
        ):
            raise H3InputError("FP16 weight-cache budget must be finite and >= 0")
        if self.fp16_weight_cache_gib and not self.fp16_cache_layers:
            raise H3InputError("select measured cache layers with --fp16-cache-layer")
        if len(set(self.fp16_cache_layers)) != len(self.fp16_cache_layers):
            raise H3InputError("FP16 cache layers must be unique")


@dataclass
class H3SamplingParams:
    height: int = 768
    width: int = 1344
    fps: int = 24
    num_frames: int = 243
    num_inference_steps: int = 50
    seed: int = 42
    num_outputs_per_prompt: int = 1
    quality: str | None = None
    lora_scale: float = 1.0
    extra_args: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.lora_scale):
            raise H3InputError("LoRA scale must be finite")
        if self.height <= 0 or self.width <= 0:
            raise H3InputError("video dimensions must be positive")
        if self.height % 32 or self.width % 32:
            raise H3InputError("H3 video dimensions must be multiples of 32")
        if not 0.25 <= self.width / self.height <= 4:
            raise H3InputError("H3 canvas aspect ratio must be in [1:4, 4:1]")
        if self.fps != 24:
            raise H3InputError("H3 generates at 24 FPS")
        if not MIN_OUTPUT_FRAMES <= self.num_frames <= MAX_OUTPUT_FRAMES:
            raise H3InputError("H3 requires between 22 and 362 frames")
        if self.num_inference_steps < 2:
            raise H3InputError("H3 needs at least two sigma positions")
        if self.num_outputs_per_prompt != 1:
            raise H3InputError("native H3 generates one video per request")
        if self.quality not in (None, "lossless"):
            raise H3InputError("native H3 does not enable approximate step caches")
        for key in ("flow_shift", "audio_flow_shift"):
            value = self.extra_args.get(key)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise H3InputError(f"{key} must be finite and positive")
        duration = self.extra_args.get("duration_seconds")
        if duration is not None and (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or not MIN_OUTPUT_SECONDS <= duration <= MAX_OUTPUT_SECONDS
        ):
            raise H3InputError("H3 duration must be between 22/24 and 15 seconds")


@dataclass
class H3Request:
    prompt: str = DEFAULT_PROMPT
    sampling: H3SamplingParams = field(default_factory=H3SamplingParams)
    media: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise H3InputError("prompt must not be empty")


def sampling_for_deployment(
    config: H3Config,
    *,
    num_inference_steps: int | None = None,
    lora_scale: float = 1.0,
    **kwargs,
) -> H3SamplingParams:
    """Choose omitted CLI/HTTP steps from the active adapter's contract.

    Explicit step/shift values are preserved and validated before dispatch.
    Python callers can also use this helper instead of specifying sigma points.
    """
    spec = None
    if config.lora_path:
        from .fasth3 import FastH3Spec
        from .lora import inspect_deployment_adapter

        candidate = inspect_deployment_adapter(config)
        if isinstance(candidate, FastH3Spec) and lora_scale != 1:
            raise H3InputError("FastH3 is fused; request lora_scale must be 1")
        if lora_scale != 0:
            spec = candidate
    if spec is not None:
        extra = dict(kwargs.get("extra_args") or {})
        extra.setdefault("flow_shift", spec.video_shift)
        extra.setdefault("audio_flow_shift", spec.audio_shift)
        kwargs["extra_args"] = extra
    if num_inference_steps is None:
        num_inference_steps = spec.api_steps if spec is not None else 50
    return H3SamplingParams(
        num_inference_steps=num_inference_steps, lora_scale=lora_scale, **kwargs
    )
