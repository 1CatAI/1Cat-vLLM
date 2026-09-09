# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native serial MiniMax H3 pipeline. See UPSTREAM.md for port provenance."""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch import nn
from tqdm.auto import tqdm
from transformers import Qwen2TokenizerFast, Qwen3VLProcessor

from vllm import envs
from vllm.distributed import get_tp_group, get_world_group
from vllm.logger import init_logger
from vllm.utils.mem_utils import get_cpu_memory
from vllm.video.metrics import DenoiseWorkCounter

from .attention import Attention, attention_backend
from .comfy_checkpoint import inspect_comfy_checkpoint, resolve_comfy_checkpoint_path
from .condition_noise import (
    minimax_h3_audio_cond_noise_aug_rows,
    minimax_h3_imgvid_cond_noise_aug_rows,
)
from .conditioning import MiniMaxH3TextConditioning
from .config import (
    MAX_OUTPUT_FRAMES,
    MAX_OUTPUT_SECONDS,
    MIN_OUTPUT_FRAMES,
    MIN_OUTPUT_SECONDS,
    H3Config,
    H3InputError,
    H3Request,
)
from .denoise_loop import MiniMaxH3DenoiseBranch, minimax_h3_denoise_loop
from .encoder import MiniMaxH3Qwen3VLEncoder
from .packed_sequence import (
    minimax_h3_packed_sequence,
    minimax_h3_packed_sequence_ref2va_blocks,
)
from .packed_tokens import (
    minimax_h3_patchify_video_latent,
    minimax_h3_unpack_audio_tokens,
    minimax_h3_unpatchify_video_tokens,
)
from .preprocessing import (
    load_minimax_h3_images as _load_images,
)
from .preprocessing import (
    minimax_h3_multi_image_presentation,
    minimax_h3_ref2va_presentation,
    minimax_h3_ref2va_video_presentation,
    minimax_h3_text_only_ids,
)
from .preprocessing import (
    resolve_minimax_h3_aspect_ratio as _resolve_minimax_h3_aspect_ratio,
)
from .preprocessing import (
    resolve_minimax_h3_output_canvas as _resolve_output_canvas,
)
from .preprocessing import (
    resolve_minimax_h3_reference_image_shape as _reference_image_shape,
)
from .quantization import DiffusionInt8ConvRotConfig
from .reference_video import (
    load_audio_file,
    load_video_audio,
    load_video_frames,
    prepare_reference_videos,
    sample_reference_video_frames,
    validate_reference_audio_files,
    validate_reference_audio_waveforms,
)
from .residency import LayerwiseModuleStager, MMapHostWeights, PinnedModuleStager
from .sigma_schedule import DMD2SigmaSchedule
from .time_request import (
    MINIMAX_H3_SHAPE_PLANNER,
    minimax_h3_align_frame_count,
    minimax_h3_time_shift_sigmas,
)
from .transformer import MiniMaxH3DiTBlock, MiniMaxH3DiTModel
from .vae import MiniMaxH3AudioVAE, MiniMaxH3VideoVAE
from .weight_cache import FP16WeightCache
from .weights import iter_checkpoint_weights, resolve_model_root

logger = init_logger(__name__)

MINIMAX_H3_FPS = 24


MINIMAX_H3_AUDIO_SAMPLE_RATE = 32000


MINIMAX_H3_IMGVID_COND_TIMESTEP = 0.999


MINIMAX_H3_AUDIO_REF_COND_TIMESTEP = 1.0


MINIMAX_H3_OUTPUT_SHORT_EDGE = 768


MINIMAX_H3_OUTPUT_MAX_PIXELS = 768 * 1344


MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE = 2048


MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE = 32


MINIMAX_H3_SUPPORTED_ASPECT_RATIOS = {
    "21:9": 21.0 / 9.0,
    "16:9": 16.0 / 9.0,
    "4:3": 4.0 / 3.0,
    "1:1": 1.0,
    "3:4": 3.0 / 4.0,
    "9:16": 9.0 / 16.0,
}


MINIMAX_H3_MAX_REFERENCE_IMAGE_BYTES = 30 * 1024 * 1024


MINIMAX_H3_REFERENCE_IMAGE_FORMATS = frozenset({"jpeg", "png", "webp", "heic", "heif"})


_MINIMAX_H3_DENOISE_INPUT_KEYS = (
    "task",
    "text_embeddings",
    "text_tags",
    "seed",
    "latent_t",
    "latent_h",
    "latent_w",
    "audio_t",
    "num_frames",
    "num_steps",
    "video_shift",
    "audio_shift",
    "base_schedule",
    "visual_condition",
    "visual_condition_shape",
    "audio_condition",
    "ref_audio_t",
    "ref_blocks",
    "visual_condition_shapes",
    "audio_condition_lengths",
    "keyframe_frame_indices",
)


def _read_base_schedule(release: Mapping[str, Any]) -> DMD2SigmaSchedule | None:
    """Read a partition's distilled schedule. An absent key means legacy uniform."""
    return DMD2SigmaSchedule.from_metadata(release)


def _prepare_minimax_h3_video_output(video: torch.Tensor) -> torch.Tensor:
    """Quantize decoded frames in place before worker-to-engine transfer."""
    if not torch.isfinite(video).all():
        raise RuntimeError("H3 VAE produced non-finite video")
    video = video.detach().float()
    video.clamp_(0, 1).mul_(255).round_()
    return video.permute(0, 2, 3, 4, 1).to(
        dtype=torch.uint8,
        memory_format=torch.contiguous_format,
    )


def _align_multiple(value: float, multiple: int = 32) -> int:
    return max(multiple, int(round(float(value) / multiple)) * multiple)


def _load_image(value: Any) -> Image.Image:
    images = _load_images(value)
    if len(images) != 1:
        raise H3InputError(f"MiniMax H3 expected one image, got {len(images)}")
    return images[0]


def _load_audio(value: Any) -> tuple[torch.Tensor, int]:
    if isinstance(value, (list, tuple)) and not (
        len(value) == 2 and isinstance(value[1], (int, np.integer))
    ):
        audios = _load_audios(value)
        if len(audios) != 1:
            raise H3InputError(f"MiniMax H3 expected one audio, got {len(audios)}")
        return audios[0]
    if isinstance(value, (str, os.PathLike)):
        return load_audio_file(str(value))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        waveform, sample_rate = value
        waveform = torch.as_tensor(waveform).float()
        return waveform, int(sample_rate)
    if isinstance(value, dict):
        waveform = value.get("waveform", value.get("array"))
        sample_rate = value.get("sample_rate", value.get("sampling_rate"))
        if waveform is not None and sample_rate is not None:
            return torch.as_tensor(waveform).float(), int(sample_rate)
    raise H3InputError(
        "MiniMax H3 audio input must be a path, (waveform, sample_rate), or a waveform "
        "mapping"
    )


def _load_audios(value: Any) -> list[tuple[torch.Tensor, int]]:
    if isinstance(value, (list, tuple)) and not (
        len(value) == 2 and isinstance(value[1], (int, np.integer))
    ):
        if not value:
            raise H3InputError("MiniMax H3 audio input must not be empty")
        return [_load_audio(item) for item in value]
    return [_load_audio(value)]


def _as_int_list(value: Any, *, name: str) -> list[int]:
    if isinstance(value, bool):
        raise H3InputError(f"{name} must be an integer or a list of integers")
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = list(value)
        if not result:
            raise H3InputError(f"{name} must not be empty")
        if any(
            isinstance(item, bool) or not isinstance(item, (int, np.integer))
            for item in result
        ):
            raise H3InputError(f"{name} must contain only integers")
        return [int(item) for item in result]
    raise H3InputError(f"{name} must be an integer or a list of integers")


def _resolve_fl2va_keyframe_indices(
    extra: Mapping[str, Any], image_count: int
) -> list[int]:
    target = extra.get("target")
    target = target if isinstance(target, Mapping) else {}
    raw = extra.get("frame_indices", extra.get("frame_index"))
    if raw is None:
        raw = target.get("frame_indices", target.get("frame_index"))
    if raw is None:
        raw_indices = [0] if image_count == 1 else [0, -1]
    else:
        raw_indices = _as_int_list(raw, name="frame_indices")
    if len(raw_indices) != image_count:
        raise H3InputError(
            f"MiniMax H3 FL2VA requires one frame index per image: got {raw_indices!r} "
            f"for {image_count} image(s)"
        )
    if tuple(raw_indices) not in ((0,), (-1,), (0, -1)):
        raise H3InputError(
            "MiniMax H3 FL2VA frame_indices must be [0], [-1], or [0, -1]"
        )
    return raw_indices


def _reuse_prepared_reference_videos(
    prepared: list[dict[str, Any]] | None,
    *,
    expected_count: int,
) -> list[dict[str, Any]] | None:
    if prepared is None:
        return None
    if len(prepared) != expected_count:
        raise H3InputError(
            "MiniMax H3 prepared-reference-video count does not match the request"
        )
    for item in prepared:
        if not os.path.isfile(item["prepared_path"]):
            raise H3InputError(
                f"MiniMax H3 prepared reference video is unavailable: "
                f"{item['prepared_path']}"
            )
    return prepared


def _validate_ref2va_reference_counts(
    image_count: int,
    video_count: int,
    audio_count: int,
) -> None:
    """Validate the official Ref2VA reference-count contract."""
    if image_count < 0 or video_count < 0 or audio_count < 0:
        raise H3InputError("MiniMax H3 reference counts must be non-negative")
    if image_count + video_count == 0:
        raise H3InputError("ref2va requires at least one image or video reference")
    if image_count > 9:
        raise H3InputError("ref2va accepts at most 9 image references")
    if video_count > 3:
        raise H3InputError("ref2va accepts at most 3 video references")
    if audio_count > 3:
        raise H3InputError("ref2va accepts at most 3 standalone audio references")
    if image_count + video_count + audio_count > 12:
        raise H3InputError("ref2va accepts at most 12 total references")


def _resolve_minimax_h3_num_outputs(value: Any) -> int:
    if value is None:
        return 1
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise H3InputError(
            "MiniMax H3 num_outputs_per_prompt must be an integer in [1, 10]"
        )
    value = int(value)
    if not 1 <= value <= 10:
        raise H3InputError(
            f"MiniMax H3 num_outputs_per_prompt must be in [1, 10], got {value}"
        )
    return value


def _minimax_h3_output_seeds(seed: int, num_outputs: int) -> list[int]:
    return [int(seed) + output_index for output_index in range(int(num_outputs))]


def _validate_reference_image(image: Image.Image) -> None:
    width, height = image.size
    if min(width, height) < 256 or max(width, height) > 5760:
        raise H3InputError(
            f"MiniMax H3 reference image dimensions must be in [256, 5760] pixels, got "
            f"{width}x{height}"
        )
    ratio = width / height
    if not 0.4 <= ratio <= 2.5:
        raise H3InputError(
            f"MiniMax H3 reference image aspect ratio must be in [0.4, 2.5], got "
            f"{width}x{height}"
        )


def _dit_rank_world() -> tuple[Any, int, int]:
    if not dist.is_initialized():
        return None, 0, 1
    group = get_world_group().device_group
    return group, dist.get_rank(group), dist.get_world_size(group)


def _broadcast_rank0_exception(exc: Exception | None) -> None:
    """Synchronize a rank-0-only exception across every DiT rank.

    H3 reference-video preparation runs only on rank 0; the other DiT ranks
    return ``None`` without touching disk. When rank 0 raises inside that
    path it exits :meth:`prepare_encode` before reaching the downstream
    ``dist.broadcast`` calls, and non-zero ranks then hang on those
    collectives forever. Every rank calls this helper right after the
    rank-0-only work, before any subsequent collective, so all ranks either
    raise the same error together or all continue.
    """
    group, rank, world_size = _dit_rank_world()
    if world_size == 1:
        if exc is not None:
            raise exc
        return
    if rank == 0:
        if exc is None:
            payload: list[Any] = [None]
        else:
            payload = [
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "status_code": getattr(exc, "status_code", None),
                    "error_type": getattr(exc, "error_type", None),
                }
            ]
    else:
        payload = [None]
    dist.broadcast_object_list(payload, src=0, group=group)
    info = payload[0]
    if info is None:
        return
    if rank == 0:
        assert exc is not None
        raise exc
    # Rebuild a matching client-facing error on non-zero ranks so the runner's
    # per-request try/except records the same 4xx status as rank 0. The exact
    # subclass need not survive the wire; the message and status suffice.
    message = f"[rank 0] {info['type']}: {info['message']}"
    raise H3InputError(message)


def _broadcast_tensor(
    tensor: torch.Tensor | None,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    group, rank, world_size = _dit_rank_world()
    if world_size == 1:
        if tensor is None:
            raise ValueError("source tensor is required for single-rank execution")
        return tensor.to(device=device, dtype=dtype)

    shape = torch.zeros(5, dtype=torch.long, device=device)
    if rank == 0:
        if tensor is None:
            raise ValueError("rank 0 must provide a tensor to broadcast")
        shape[0] = tensor.ndim
        shape[1 : tensor.ndim + 1] = torch.tensor(
            tensor.shape,
            device=device,
        )
    dist.broadcast(shape, src=0, group=group)
    ndim = int(shape[0].item())
    tensor_shape = tuple(int(v) for v in shape[1 : ndim + 1].tolist())
    if rank == 0:
        output = tensor.to(device=device, dtype=dtype).contiguous()
    else:
        output = torch.empty(tensor_shape, device=device, dtype=dtype)
    dist.broadcast(output, src=0, group=group)
    return output


class MiniMaxH3Pipeline(nn.Module):
    def __init__(self, config: H3Config, *, shared_weights_dir: str | None = None):
        super().__init__()
        if (
            config.share_host_vae_weights
            and config.tensor_parallel_size > 1
            and shared_weights_dir is None
        ):
            raise H3InputError(
                "shared host VAE weights require an engine-owned directory"
            )
        self.config = config
        self.partition = config.partition
        from vllm.media.progress import report_loading

        def loading(done, component):
            group = get_tp_group()
            report_loading(
                done,
                4,
                component,
                rank=group.rank_in_group,
                world_size=group.world_size,
            )

        loading(0, "transformer")
        self.device = torch.device("cuda", torch.accelerator.current_device_index())
        use_mmap = config.host_memory_mode == "mmap" or (
            config.host_memory_mode == "auto" and get_cpu_memory() < 128 * 1024**3
        )
        self._host_backing = (
            MMapHostWeights(
                config.host_memory_directory or Path(envs.VLLM_CACHE_ROOT) / "h3-host"
            )
            if use_mmap
            else None
        )
        if self._host_backing is not None:
            logger.info(
                "H3 uses reclaimable disk-backed host weights at %s; "
                "weights and compute precision are unchanged",
                self._host_backing.directory,
            )
        from .fasth3 import FastH3Fusion, FastH3Spec
        from .flashgen import FlashGenSpec, restore_dense_adaln_weights
        from .lora import inspect_deployment_adapter, select_adapter_file

        adapter_spec = inspect_deployment_adapter(config) if config.lora_path else None
        self.model_root = resolve_model_root(
            config,
            require_original_transformer=isinstance(adapter_spec, FlashGenSpec),
        )
        path = self.model_root / ("FL2VA" if self.partition == "fl2va" else "Ref2VA")
        shared = self.model_root / "FL2VA"
        if not shared.is_dir():
            shared = path
        release = json.loads((path / "model_index.json").read_text())["_minimax_h3"]
        if release["partition"].lower() != self.partition:
            raise H3InputError("checkpoint partition does not match deployment")
        self.supported_tasks = frozenset(
            release.get(
                "tasks", ["ref2va"] if self.partition == "ref2va" else ["t2va", "fl2va"]
            )
        )
        shifts = release.get("sigma_shift_scales", {})
        self.default_video_shift = float(shifts.get("video", 12.0))
        self.default_audio_shift = float(shifts.get("audio", 3.0))
        self._base_schedule_by_partition = {
            self.partition: _read_base_schedule(release)
        }
        if (
            adapter_spec is not None
            and self._base_schedule_by_partition[self.partition] is not None
        ):
            raise H3InputError(
                "H3 LoRA cannot overlay an already distilled base_schedule"
            )
        if isinstance(adapter_spec, FastH3Spec):
            if (self.default_video_shift, self.default_audio_shift) != (12.0, 3.0):
                raise H3InputError("FastH3 requires base video/audio shifts 12/3")
            self.supported_tasks = adapter_spec.supported_tasks
        transformer_path = path / "transformer"
        quant = None
        overrides = None
        restore_adaln = False
        if config.transformer_path:
            transformer_path = resolve_comfy_checkpoint_path(config.transformer_path)
            checkpoint = inspect_comfy_checkpoint(
                transformer_path, expected_partition=self.partition
            )
            overrides = checkpoint.arch_overrides
            restore_adaln = (
                isinstance(adapter_spec, FlashGenSpec)
                and checkpoint.adaln_curve_grid is not None
            )
            if restore_adaln:
                overrides = None
            quant = DiffusionInt8ConvRotConfig(
                layer_configs=checkpoint.layer_configs,
                weight_layout=config.int8_weight_layout,
            )
        architecture = json.loads((path / "transformer/config.json").read_text())
        token = attention_backend.set(config.attention_backend)
        try:
            self.transformer = MiniMaxH3DiTModel(
                architecture,
                quant_config=quant,
                arch_overrides=overrides,
                residual_sequence_parallel=config.residual_sequence_parallel,
            )
        finally:
            attention_backend.reset(token)
        if self._host_backing is not None:
            PinnedModuleStager.map_cpu_weights(
                self.transformer, self._host_backing, preserve_parameters=False
            )
        for module in self.transformer.modules():
            if isinstance(module, Attention):
                module.query_tile = config.attention_query_tile
        self._residual_reduction = None
        if config.residual_reduction == "peer":
            from .collectives import H3ResidualReduction

            self._residual_reduction = H3ResidualReduction(
                get_tp_group(),
                memory_budget_bytes=int(config.residual_reduction_memory_gib * 2**30),
            )
            for module in self.transformer.modules():
                if isinstance(module, MiniMaxH3DiTBlock):
                    module.residual_reducer = self._residual_reduction
        weights = iter_checkpoint_weights(transformer_path)
        if restore_adaln:
            weights = restore_dense_adaln_weights(weights, path / "transformer")
        fusion = None
        if isinstance(adapter_spec, FastH3Spec):
            fusion = FastH3Fusion(
                select_adapter_file(config.lora_path),
                partition=config.partition,
                head_dim=self.transformer.arch.attention_head_dim,
                device=self.device,
            )
            weights = fusion.apply(weights)
        loaded = self.transformer.load_weights(weights)
        if fusion is not None:
            fusion.validate_fully_applied(loaded)
        required = set(dict(self.transformer.named_parameters()))
        required.update(dict(self.transformer.named_buffers()))
        missing = required - loaded
        if missing:
            raise RuntimeError(f"H3 DiT checkpoint missing tensors: {sorted(missing)}")
        for layer in self.transformer.modules():
            method = getattr(layer, "quant_method", None)
            if method is not None:
                layer.h3_fp16_weight_layout = config.fp16_weight_layout
                method.process_weights_after_loading(layer)
                if self._host_backing is not None:
                    PinnedModuleStager.map_cpu_weights(layer, self._host_backing)
        self.transformer.post_load_weights()
        self.turbo_spec = None
        if fusion is not None:
            self.turbo_spec = fusion.spec
        elif config.lora_path:
            from .lora import install_adapter

            self.turbo_spec = install_adapter(
                self.transformer, config.lora_path, self.partition
            )
        self._dit_stager = PinnedModuleStager(
            self.transformer,
            self.device,
            host_backing=self._host_backing,
            pin_memory=config.host_weight_pin_memory,
        )
        self._weight_cache = FP16WeightCache(
            self.transformer,
            budget_gib=config.fp16_weight_cache_gib,
            layers=config.fp16_cache_layers,
        )
        self.text_encoder_group = get_tp_group()
        loading(1, "text_encoder")
        self.text_encoder_tp_size = self.text_encoder_group.world_size
        self._dit_rank = self.text_encoder_group.rank_in_group
        self.tokenizer = Qwen2TokenizerFast.from_pretrained(
            str(shared / "tokenizer"), local_files_only=True
        )
        self.processor = Qwen3VLProcessor.from_pretrained(
            str(shared / "processor"), local_files_only=True
        )
        self.text_encoder = MiniMaxH3Qwen3VLEncoder(
            str(shared / "text_encoder"),
            device=self.device,
            load_model=True,
            encoder_group=self.text_encoder_group,
        )
        if self._host_backing is not None:
            PinnedModuleStager.map_cpu_weights(
                self.text_encoder, self._host_backing, preserve_parameters=False
            )
        self.text_encoder.load_weights(iter_checkpoint_weights(shared / "text_encoder"))
        self._encoder_stager = PinnedModuleStager(
            self.text_encoder,
            self.device,
            pin_memory=config.host_weight_pin_memory,
            host_backing=self._host_backing,
        )
        self._dit_layer_stager: LayerwiseModuleStager | None = None
        self._encoder_layer_stager: LayerwiseModuleStager | None = None
        if config.weight_offload == "layer":
            self._dit_layer_stager = LayerwiseModuleStager(
                self._dit_stager,
                (*self.transformer.token_refiner.blocks, *self.transformer.blocks),
                # Cache decision probes may consume these outside block.forward.
                resident_modules=(
                    self.transformer.blocks[0].norm1,
                    self.transformer.blocks[0].adaln_proj,
                ),
            )
            self._encoder_layer_stager = LayerwiseModuleStager(
                self._encoder_stager,
                (
                    *self.text_encoder.vision.blocks,
                    *self.text_encoder.text_model.layers,
                ),
            )
        loading(2, "video_vae")
        self.video_vae = MiniMaxH3VideoVAE(
            str(shared / "video_vae"),
            device=self.device,
            load_device=torch.device("cpu"),
            pin_memory=config.host_weight_pin_memory,
            shared_weights_dir=shared_weights_dir,
        )
        self.video_vae.set_parallel_size(config.tensor_parallel_size)
        loading(3, "audio_vae")
        self.audio_vae = MiniMaxH3AudioVAE(
            str(shared / "audio_vae"),
            device=self.device,
            load_device=torch.device("cpu"),
            pin_memory=config.host_weight_pin_memory,
            shared_weights_dir=shared_weights_dir,
        )
        self.stage_durations = {}
        self.actual_dit_calls = 0
        self.eval()
        loading(4, "ready")

    def _transformer_for_task(self, task):
        return self.transformer

    def _resolve_sigma_positions(self, task, sampling):
        if getattr(self, "turbo_spec", None) is not None:
            from .lora import validate_adapter_sampling

            validate_adapter_sampling(self.turbo_spec, task, sampling)
            if sampling.lora_scale != 0:
                return self.turbo_spec.base_schedule, self.turbo_spec.api_steps
        schedule = self._base_schedule_for_task(task)
        if schedule is None:
            return None, sampling.num_inference_steps
        if sampling.num_inference_steps != schedule.num_inference_steps:
            raise H3InputError("checkpoint pins its own inference-step count")
        return schedule.base_schedule, schedule.num_inference_steps

    def _encode_text_hidden(self, input_ids, vision_kwargs):
        with self._component_on_device(self.text_encoder):
            return self.text_encoder.encode_ids(input_ids, **vision_kwargs)

    @contextmanager
    def _component_on_device(self, component):
        if (
            component is self.text_encoder
            and getattr(self, "_encoder_layer_stager", None) is not None
        ):
            plan = self._encoder_layer_stager
            try:
                with plan.on_device():
                    yield
            finally:
                self.stage_durations["encoder_layer_weight_staging"] = plan.load_seconds
                self.stage_durations["encoder_layer_weight_offload"] = (
                    plan.offload_seconds
                )
            return
        stager = self._encoder_stager if component is self.text_encoder else None
        if stager is not None:
            stager.load()
        else:
            component.load_to_device()
        try:
            yield
        finally:
            if stager is not None:
                stager.offload()
            else:
                component.offload_to_cpu()

    @contextmanager
    def _resident_dit_layers_on_device(self, *, enabled=True):
        started = time.perf_counter()
        if getattr(self, "_dit_layer_stager", None) is not None:
            plan = self._dit_layer_stager
            try:
                with plan.on_device():
                    self.stage_durations["dit_staging_and_weight_cache"] = (
                        time.perf_counter() - started
                    )
                    yield
            finally:
                self.stage_durations["dit_layer_weight_staging"] = plan.load_seconds
                self.stage_durations["dit_layer_weight_offload"] = plan.offload_seconds
            return
        self._dit_stager.load()
        try:
            self._weight_cache.prepare()
            torch.accelerator.synchronize()
            self.stage_durations["dit_staging_and_weight_cache"] = (
                time.perf_counter() - started
            )
            yield
        finally:
            self._weight_cache.clear()
            self._dit_stager.offload()

    def progress_bar(self, *, total):
        return tqdm(total=total, desc="H3 denoise", disable=self._dit_rank != 0)

    def residual_reduction_stats(self):
        reducer = getattr(self, "_residual_reduction", None)
        if reducer is None:
            return {"configured_backend": "native", "raw_ipc_peak_bytes": 0}
        return reducer.snapshot()

    def close(self):
        reducer = getattr(self, "_residual_reduction", None)
        if reducer is not None:
            reducer.close()

    @torch.inference_mode()
    def forward(self, request: H3Request):
        from vllm.media.progress import report

        self.stage_durations = {}
        reducer = getattr(self, "_residual_reduction", None)
        if reducer is not None:
            reducer.begin_request()
        self.actual_dit_calls = 0
        report("encoding")
        started = time.perf_counter()
        context = self._prepare_request_inputs(
            prompt=request.prompt,
            multi_modal_data=request.media,
            sampling=request.sampling,
        )
        torch.accelerator.synchronize()
        self.stage_durations["encode"] = time.perf_counter() - started
        started = time.perf_counter()
        from .lora import lora_scale

        token = lora_scale.set(request.sampling.lora_scale)
        try:
            video_latent, audio_latent = self.diffuse(**self._denoise_kwargs(context))
        finally:
            lora_scale.reset(token)
        torch.accelerator.synchronize()
        self.stage_durations["denoise_including_staging"] = (
            time.perf_counter() - started
        )
        started = time.perf_counter()
        report("decoding")
        video, audio = self.decode(
            video_latent, audio_latent, height=context["height"], width=context["width"]
        )
        torch.accelerator.synchronize()
        self.stage_durations["vae"] = time.perf_counter() - started
        return _prepare_minimax_h3_video_output(video), audio

    def _base_schedule_for_task(self, task: str) -> DMD2SigmaSchedule | None:
        """Return the distilled schedule of the partition that serves ``task``."""
        partition = "ref2va" if task == "ref2va" else "fl2va"
        return self._base_schedule_by_partition.get(partition)

    def _resolve_task(
        self,
        requested: str | None,
        multi_modal_data: dict[str, Any],
    ) -> str:
        if requested is None:
            # A Ref2VA-only startup has no FL2VA transformer; preserve its
            # historical implicit default even for image-only references.
            if self.partition == "ref2va" or (
                multi_modal_data.get("video") is not None
                or multi_modal_data.get("audio") is not None
            ):
                requested = "ref2va"
            elif multi_modal_data.get("image") is not None:
                requested = "fl2va"
            else:
                requested = "t2va"
        task = str(requested).lower()
        if task not in self.supported_tasks:
            raise H3InputError(
                f"checkpoint partition {self.partition!r} supports "
                f"{sorted(self.supported_tasks)}, got task={task!r}"
            )
        return task

    def _resolve_shape(
        self,
        task: str,
        sampling: Any,
        image: Image.Image | None,
    ) -> tuple[int, int, int, int, int]:
        fps = int(sampling.fps or MINIMAX_H3_FPS)
        if fps != MINIMAX_H3_FPS:
            raise H3InputError(f"MiniMax H3 output fps is fixed at {MINIMAX_H3_FPS}")
        extra = sampling.extra_args or {}
        target = extra.get("target")
        if target is not None and not isinstance(target, Mapping):
            raise H3InputError("MiniMax H3 extra_args['target'] must be an object")
        target = target if isinstance(target, Mapping) else {}
        duration = target.get(
            "duration_seconds", extra.get("duration_seconds", extra.get("duration"))
        )
        if duration is not None:
            if isinstance(duration, bool):
                raise H3InputError(
                    f"MiniMax H3 output duration must be in [22/24, 15] seconds, got "
                    f"{duration!r}"
                )
            try:
                duration = float(duration)
            except (TypeError, ValueError) as exc:
                raise H3InputError(
                    f"MiniMax H3 output duration must be in [22/24, 15] seconds, got "
                    f"{duration!r}"
                ) from exc
            if (
                not math.isfinite(duration)
                or not MIN_OUTPUT_SECONDS <= duration <= MAX_OUTPUT_SECONDS
            ):
                raise H3InputError(
                    f"MiniMax H3 output duration must be in [22/24, 15] seconds, got "
                    f"{duration}"
                )
            requested_frames = int(round(duration * fps))
        elif int(sampling.num_frames or 1) > 1:
            requested_frames = int(sampling.num_frames)
        else:
            requested_frames = 124 if task == "ref2va" else 209
            duration = requested_frames / fps
        if not MIN_OUTPUT_FRAMES <= requested_frames <= MAX_OUTPUT_FRAMES:
            raise H3InputError(
                f"MiniMax H3 requires between 22 and 362 frames, got {requested_frames}"
            )
        num_frames = minimax_h3_align_frame_count(requested_frames)

        height = sampling.height
        width = sampling.width
        aspect_ratio = target.get("aspect_ratio", extra.get("aspect_ratio"))
        raw_short_edge = target.get(
            "short_edge", extra.get("short_edge", MINIMAX_H3_OUTPUT_SHORT_EDGE)
        )
        if isinstance(raw_short_edge, bool) or not isinstance(
            raw_short_edge, (int, np.integer)
        ):
            raise H3InputError(
                f"MiniMax H3 target.short_edge must be {MINIMAX_H3_OUTPUT_SHORT_EDGE}, "
                f"got {raw_short_edge!r}"
            )
        short_edge = int(raw_short_edge)

        if height is not None and width is not None and aspect_ratio is None:
            # Native requests supply the final canvas, including the aligned
            # 1344x768 shape whose ratio is slightly different from 16:9.
            aspect_ratio = width / height
        else:
            aspect_ratio = _resolve_minimax_h3_aspect_ratio(
                task,
                aspect_ratio,
                image,
            )
        if not 0.25 <= aspect_ratio <= 4.0:
            raise H3InputError(
                f"MiniMax H3 canvas aspect ratio must be in [1:4, 4:1], got "
                f"{aspect_ratio}"
            )

        if height is None or width is None:
            height, width = _resolve_output_canvas(aspect_ratio, short_edge)
        height = int(height) // 32 * 32
        width = int(width) // 32 * 32
        if min(height, width) <= 0:
            raise H3InputError(f"invalid MiniMax H3 canvas {width}x{height}")
        if width > 4 * height or height > 4 * width:
            raise H3InputError("MiniMax H3 canvas aspect ratio must be in [1:4, 4:1]")

        latent_t = MINIMAX_H3_SHAPE_PLANNER.video_latent_t(num_frames)
        audio_t = MINIMAX_H3_SHAPE_PLANNER.audio_latent_t(num_frames / fps)
        return height, width, num_frames, latent_t, audio_t

    def encode_prompt(
        self,
        *,
        task: str,
        prompt: str,
        image: Image.Image | None = None,
        images: list[Image.Image] | None = None,
        prepared_videos: list[dict[str, Any]] | None = None,
        condition_labels: list[tuple[str, int]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, rank, _ = _dit_rank_world()
        hidden = None
        tags = None
        ids = None
        vision_kwargs: dict[str, torch.Tensor] = {}
        images = (
            list(images)
            if images is not None
            else ([image] if image is not None else [])
        )
        if rank == 0:
            if task == "t2va":
                ids = minimax_h3_text_only_ids(self.tokenizer, prompt)
                tags = torch.ones(ids.shape[0], dtype=torch.long)
                vision_kwargs = {}
            else:
                image_token_counts: list[int] = []
                if images:
                    vision = self.processor.image_processor(
                        images=images,
                        return_tensors="pt",
                    )
                    image_grid = vision["image_grid_thw"]
                    merge = int(self.processor.image_processor.merge_size) ** 2
                    image_token_counts = [
                        int(grid.prod().item()) // merge for grid in image_grid
                    ]
                    vision_kwargs.update(
                        {
                            "pixel_values": vision["pixel_values"],
                            "image_grid_thw": image_grid,
                        }
                    )

                video_block_counts: list[list[int]] = []
                video_block_timestamps: list[list[float]] = []
                if prepared_videos:
                    videos = []
                    sampled_videos = []
                    for index, item in enumerate(prepared_videos):
                        sampled = sample_reference_video_frames(item["prepared_path"])
                        videos.append(np.stack(sampled["frames"]))
                        sampled_videos.append(sampled)
                    vision = self.processor.video_processor(
                        videos=videos,
                        do_sample_frames=False,
                        return_tensors="pt",
                    )
                    video_grid = vision["video_grid_thw"]
                    merge = int(self.processor.image_processor.merge_size) ** 2
                    for index, sampled in enumerate(sampled_videos):
                        blocks = int(video_grid[index, 0])
                        per_block = (
                            int(video_grid[index, 1])
                            * int(video_grid[index, 2])
                            // merge
                        )
                        timestamps = sampled["block_timestamps"]
                        if len(timestamps) != blocks:
                            raise ValueError(
                                f"video block count mismatch: processor={blocks}, "
                                f"timestamps={len(timestamps)}"
                            )
                        video_block_counts.append([per_block] * blocks)
                        video_block_timestamps.append(timestamps)
                    vision_kwargs.update(
                        {
                            "pixel_values_videos": vision["pixel_values_videos"],
                            "video_grid_thw": video_grid,
                        }
                    )

                if not images and not prepared_videos:
                    raise H3InputError(f"{task} requires an image or video condition")
                if condition_labels is None:
                    condition_labels = []
                    for image_index in range(1, len(images) + 1):
                        condition_labels.append(("image", image_index))
                    audio_index = 0
                    for video_index, item in enumerate(prepared_videos or (), start=1):
                        if item["input_has_audio"]:
                            audio_index += 1
                            condition_labels.append(("audio", audio_index))
                        condition_labels.append(("video", video_index))

                if task == "fl2va":
                    if prepared_videos:
                        raise H3InputError("fl2va does not accept video conditions")
                    ids, tags = minimax_h3_multi_image_presentation(
                        self.tokenizer,
                        prompt=prompt,
                        image_token_counts=image_token_counts,
                    )
                elif prepared_videos:
                    ids, tags = minimax_h3_ref2va_video_presentation(
                        self.tokenizer,
                        prompt=prompt,
                        condition_labels=condition_labels,
                        image_token_count=image_token_counts or None,
                        video_block_token_counts=video_block_counts,
                        video_block_timestamps=video_block_timestamps,
                    )
                else:
                    ids, tags = minimax_h3_ref2va_presentation(
                        self.tokenizer,
                        prompt=prompt,
                        condition_labels=condition_labels,
                        image_token_count=image_token_counts or None,
                    )

            logger.info(
                "MiniMax H3 %s Qwen presentation: %d tokens%s",
                task,
                int(ids.shape[0]),
                (
                    f", {len(images)} reference images"
                    + (
                        f", {len(prepared_videos)} reference videos"
                        if prepared_videos
                        else ""
                    )
                    if images
                    else (
                        f", {len(prepared_videos)} reference videos"
                        if prepared_videos
                        else ""
                    )
                ),
            )

        if rank < self.text_encoder_tp_size:
            # Distribute the encode inputs from the DiT main rank to the other
            # encoder TP ranks, then run the distributed encode on all of them.
            ids = self._distribute_encode_inputs(ids, vision_kwargs)
            hidden = self._encode_text_hidden(ids, vision_kwargs)

        hidden = _broadcast_tensor(
            hidden,
            dtype=torch.float16,
            device=self.device,
        )
        tags = _broadcast_tensor(
            tags,
            dtype=torch.long,
            device=self.device,
        )
        return hidden, tags

    def _encoder_group_broadcast_tensor(
        self,
        tensor: torch.Tensor | None,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Broadcast a tensor from encoder rank 0 over the encoder TP group."""
        group = self.text_encoder_group
        if group.world_size == 1:
            if tensor is None:
                raise ValueError("source tensor is required for single-rank execution")
            return tensor.to(device=device, dtype=dtype)

        shape = torch.zeros(8, dtype=torch.long, device=device)
        if group.rank_in_group == 0:
            if tensor is None:
                raise ValueError("encoder rank 0 must provide a tensor to broadcast")
            shape[0] = tensor.ndim
            shape[1 : tensor.ndim + 1] = torch.tensor(tensor.shape, device=device)
        torch.distributed.broadcast(shape, src=group.ranks[0], group=group.device_group)
        ndim = int(shape[0].item())
        tensor_shape = tuple(int(value) for value in shape[1 : ndim + 1].tolist())
        if group.rank_in_group == 0:
            output = tensor.to(device=device, dtype=dtype).contiguous()
        else:
            output = torch.empty(tensor_shape, device=device, dtype=dtype)
        torch.distributed.broadcast(
            output, src=group.ranks[0], group=group.device_group
        )
        return output

    def _distribute_encode_inputs(
        self,
        ids: torch.Tensor | None,
        vision_kwargs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Fan out encode inputs from encoder rank 0 to the encoder TP ranks.

        Mutates ``vision_kwargs`` in place so every encoder rank ends up with
        the same vision tensors, and returns the broadcast ``input_ids``.
        """
        keys = (
            "pixel_values",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
        )
        key_dtypes = {
            "pixel_values": torch.float16,
            "pixel_values_videos": torch.float16,
            "image_grid_thw": torch.long,
            "video_grid_thw": torch.long,
        }
        group = self.text_encoder_group
        device = self.device
        if group.world_size == 1:
            if ids is None:
                raise ValueError("encoder rank 0 must produce input ids")
            return ids.to(device=device, dtype=torch.long)

        mask = torch.zeros(len(keys), dtype=torch.long, device=device)
        if group.rank_in_group == 0:
            for index, key in enumerate(keys):
                mask[index] = 1 if key in vision_kwargs else 0
        torch.distributed.broadcast(mask, src=group.ranks[0], group=group.device_group)

        if group.rank_in_group == 0:
            ids = self._encoder_group_broadcast_tensor(
                ids, dtype=torch.long, device=device
            )
        else:
            ids = self._encoder_group_broadcast_tensor(
                None, dtype=torch.long, device=device
            )
        for index, key in enumerate(keys):
            if mask[index].item() == 0:
                continue
            source = vision_kwargs.get(key) if group.rank_in_group == 0 else None
            vision_kwargs[key] = self._encoder_group_broadcast_tensor(
                source,
                dtype=key_dtypes[key],
                device=device,
            )
        return ids

    def _prepare_reference_videos(
        self,
        values: Any,
        *,
        target_frame_count: int,
        workdir: str,
        start_time_seconds: Any = None,
    ) -> list[dict[str, Any]] | None:
        _, rank, _ = _dit_rank_world()
        if rank != 0:
            return None
        return prepare_reference_videos(
            values,
            target_frame_count=target_frame_count,
            workdir=workdir,
            start_time_seconds=start_time_seconds,
        )

    def _encode_visual_conditions(
        self,
        images: list[Image.Image],
        prepared_videos: list[dict[str, Any]] | None,
        *,
        video_count: int,
    ) -> tuple[torch.Tensor | None, list[tuple[int, int, int]]]:
        rows: list[torch.Tensor] = []
        shapes: list[tuple[int, int, int]] = []
        _, rank, _ = _dit_rank_world()
        # Keep image and video references in one residency window when both
        # appear in a request; otherwise the video branch would reload the VAE.
        needs_video_vae = video_count > 0 or (rank == 0 and bool(images))
        video_vae_context = (
            self._component_on_device(self.video_vae)
            if needs_video_vae
            else nullcontext()
        )
        with video_vae_context:
            if images:
                image_rows = None
                if rank == 0:
                    image_rows = torch.cat(
                        [self.video_vae.encode_image(image) for image in images]
                    )
                rows.append(
                    _broadcast_tensor(
                        image_rows,
                        dtype=torch.float32,
                        device=self.device,
                    )
                )
                shapes.extend(
                    (1, image.height // 16, image.width // 16) for image in images
                )
            if video_count:
                video_rows, video_shapes = self._encode_video_conditions_resident(
                    prepared_videos,
                    count=video_count,
                )
                rows.append(video_rows)
                shapes.extend(video_shapes)
        return (torch.cat(rows) if rows else None), shapes

    def _encode_audio_conditions_resident(
        self,
        audios: list[tuple[torch.Tensor, int]],
        *,
        max_duration_seconds: float | None = None,
    ) -> tuple[torch.Tensor | None, list[int]]:
        if not audios:
            return None, []
        if max_duration_seconds is not None:
            max_duration_seconds = float(max_duration_seconds)
            if max_duration_seconds <= 0:
                raise ValueError("max_duration_seconds must be positive")
        _, rank, _ = _dit_rank_world()
        rows = None
        lengths = torch.zeros(len(audios), dtype=torch.long, device=self.device)
        if rank == 0:
            bounded_audios = []
            for waveform, sample_rate in audios:
                if max_duration_seconds is not None:
                    max_samples = int(round(max_duration_seconds * int(sample_rate)))
                    waveform = waveform[..., :max_samples]
                bounded_audios.append((waveform, sample_rate))
            encoded = [
                self.audio_vae.encode_waveform(*audio) for audio in bounded_audios
            ]
            rows = torch.cat([item[0] for item in encoded])
            lengths = torch.tensor(
                [int(item[1]) for item in encoded],
                dtype=torch.long,
                device=self.device,
            )
        group, _, world_size = _dit_rank_world()
        if world_size > 1:
            dist.broadcast(lengths, src=0, group=group)
        return (
            _broadcast_tensor(rows, dtype=torch.float32, device=self.device),
            [int(value) for value in lengths.tolist()],
        )

    def _encode_video_conditions_resident(
        self,
        prepared_videos: list[dict[str, Any]] | None,
        *,
        count: int,
    ) -> tuple[torch.Tensor, list[tuple[int, int, int]]]:
        group, rank, world_size = _dit_rank_world()
        distributed_encode = self.video_vae.is_distributed_enabled()
        if distributed_encode:
            # Native tiled encode uses collectives, so every VPP rank must
            # enter each reference encode in the same input order.
            prepared_videos_list = [prepared_videos]
            dist.broadcast_object_list(
                prepared_videos_list,
                src=0,
                group=group,
                device=self.device,
            )
            prepared_videos = prepared_videos_list[0]

        rows = None
        shapes = torch.zeros((count, 3), dtype=torch.long, device=self.device)
        if rank == 0 or distributed_encode:
            if prepared_videos is None or len(prepared_videos) != count:
                raise ValueError("reference-video preparation is incomplete")
            encoded = [
                self.video_vae.encode_video(load_video_frames(item["prepared_path"]))
                for item in prepared_videos
            ]
            rows = torch.cat([item[0] for item in encoded])
            shapes = torch.tensor(
                [item[1] for item in encoded],
                dtype=torch.long,
                device=self.device,
            )
        if distributed_encode:
            return (
                rows.to(device=self.device, dtype=torch.float32),
                [tuple(int(value) for value in item) for item in shapes.tolist()],
            )

        if world_size > 1:
            dist.broadcast(shapes, src=0, group=group)
        return (
            _broadcast_tensor(rows, dtype=torch.float32, device=self.device),
            [tuple(int(value) for value in item) for item in shapes.tolist()],
        )

    def _encode_video_audio_conditions_resident(
        self,
        prepared_videos: list[dict[str, Any]] | None,
        *,
        has_audio: list[bool],
    ) -> tuple[torch.Tensor | None, list[int]]:
        _, rank, _ = _dit_rank_world()
        count = sum(has_audio)
        if count == 0:
            return None, []
        rows = None
        lengths = torch.zeros(count, dtype=torch.long, device=self.device)
        if rank == 0:
            if prepared_videos is None:
                raise ValueError("rank 0 reference-video preparation is incomplete")
            encoded = [
                self.audio_vae.encode_waveform(
                    *load_video_audio(
                        item["original_path"],
                        start_time_seconds=float(item.get("start_time_seconds", 0.0)),
                        duration_seconds=item.get(
                            "audio_duration_seconds",
                            item.get("duration_seconds"),
                        ),
                    )
                )
                for item in prepared_videos
                if item["input_has_audio"]
            ]
            rows = torch.cat([item[0] for item in encoded])
            lengths = torch.tensor(
                [item[1] for item in encoded],
                dtype=torch.long,
                device=self.device,
            )
        group, _, world_size = _dit_rank_world()
        if world_size > 1:
            dist.broadcast(lengths, src=0, group=group)
        return (
            _broadcast_tensor(rows, dtype=torch.float32, device=self.device),
            [int(value) for value in lengths.tolist()],
        )

    def _encode_reference_audio_conditions(
        self,
        prepared_videos: list[dict[str, Any]] | None,
        *,
        has_audio: list[bool],
        standalone_audios: list[tuple[torch.Tensor, int]],
        max_duration_seconds: float,
    ) -> tuple[torch.Tensor | None, list[int], torch.Tensor | None, list[int]]:
        # Embedded and standalone audio are consecutive direct Audio-VAE
        # calls. Keep the component resident across both paths.
        needs_audio_vae = any(has_audio) or bool(standalone_audios)
        audio_vae_context = (
            self._component_on_device(self.audio_vae)
            if needs_audio_vae
            else nullcontext()
        )
        with audio_vae_context:
            embedded_condition, embedded_lengths = (
                self._encode_video_audio_conditions_resident(
                    prepared_videos,
                    has_audio=has_audio,
                )
            )
            external_condition, external_lengths = (
                self._encode_audio_conditions_resident(
                    standalone_audios,
                    max_duration_seconds=max_duration_seconds,
                )
            )
        return (
            embedded_condition,
            embedded_lengths,
            external_condition,
            external_lengths,
        )

    def _initial_noise(
        self,
        *,
        seed: int,
        latent_t: int,
        latent_h: int,
        latent_w: int,
        audio_t: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_generator = torch.Generator(device="cpu").manual_seed(seed)
        video = torch.randn(
            1,
            24,
            latent_t,
            latent_h,
            latent_w,
            generator=video_generator,
            dtype=torch.float32,
        )
        video_rows = minimax_h3_patchify_video_latent(
            video,
            patch_size=(1, 2, 2),
        )
        audio_generator = torch.Generator(device="cpu").manual_seed(seed)
        audio_rows = torch.randn(
            audio_t * 2,
            32,
            generator=audio_generator,
            dtype=torch.float32,
        )
        return video_rows, audio_rows

    def _build_denoise_inputs(
        self,
        *,
        task: str,
        text_embeddings: torch.Tensor,
        text_tags: torch.Tensor,
        seed: int,
        latent_t: int,
        latent_h: int,
        latent_w: int,
        audio_t: int,
        num_frames: int,
        num_steps: int,
        video_shift: float,
        audio_shift: float,
        base_schedule: Sequence[float] | None,
        visual_condition: torch.Tensor | None,
        visual_condition_shape: tuple[int, int, int] | None,
        audio_condition: torch.Tensor | None,
        ref_audio_t: int | None,
        ref_blocks: list[dict[str, Any]] | None = None,
        visual_condition_shapes: list[tuple[int, int, int]] | None = None,
        audio_condition_lengths: list[int] | None = None,
        keyframe_frame_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        """Build the packed layout, initial rows, anchors, and sigma schedules.

        Shared by request-mode :meth:`diffuse` and step-mode
        :meth:`prepare_encode` so both paths start from identical state.
        """
        initial_video, initial_audio = self._initial_noise(
            seed=seed,
            latent_t=latent_t,
            latent_h=latent_h,
            latent_w=latent_w,
            audio_t=audio_t,
        )
        if task == "ref2va":
            if ref_blocks is None:
                if visual_condition_shape is None or ref_audio_t is None:
                    raise ValueError("ref2va condition metadata is missing")
                _, ref_h, ref_w = visual_condition_shape
                ref_blocks = [
                    {"kind": "image", "latent_h": ref_h, "latent_w": ref_w},
                    {"kind": "audio", "ref_audio_t": ref_audio_t},
                ]
            packed = minimax_h3_packed_sequence_ref2va_blocks(
                text_len=int(text_embeddings.shape[0]),
                latent_t=latent_t,
                latent_h=latent_h,
                latent_w=latent_w,
                audio_t=audio_t,
                ref_blocks=ref_blocks,
            )
        else:
            packed = minimax_h3_packed_sequence(
                text_len=int(text_embeddings.shape[0]),
                latent_t=latent_t,
                latent_h=latent_h,
                latent_w=latent_w,
                audio_t=audio_t,
                include_keyframe_cond=task == "fl2va",
                keyframe_frame_indices=keyframe_frame_indices
                if task == "fl2va"
                else None,
                frame_count=num_frames if task == "fl2va" else None,
            )

        tags = packed["token_tags"].clone()
        tags[packed["text_pos"]] = text_tags.cpu()
        branch = MiniMaxH3DenoiseBranch(
            packed=packed,
            text_embeddings=text_embeddings,
            token_tags=tags,
            device=self.device,
        )

        visual_anchor = visual_condition
        if visual_anchor is not None:
            condition_shapes = visual_condition_shapes
            if condition_shapes is None and visual_condition_shape is not None:
                condition_shapes = [visual_condition_shape]
            if not condition_shapes:
                raise ValueError("visual condition shape is missing")
            visual_anchor = minimax_h3_imgvid_cond_noise_aug_rows(
                visual_anchor,
                condition_shapes=condition_shapes,
                target_latent_t=latent_t,
                imgvid_cond_num_frames=len(condition_shapes),
                seed=seed,
                noise_aug=MINIMAX_H3_IMGVID_COND_TIMESTEP,
            )
            full_video = torch.zeros(
                branch.img_pos.shape[0],
                96,
                dtype=torch.float32,
            )
            full_video[branch.update_mask] = initial_video
            initial_video = full_video

        audio_anchor = audio_condition
        if audio_anchor is not None:
            condition_audio_t = audio_condition_lengths
            if condition_audio_t is None and ref_audio_t is not None:
                condition_audio_t = [ref_audio_t]
            if not condition_audio_t:
                raise ValueError("reference audio length is missing")
            audio_anchor = minimax_h3_audio_cond_noise_aug_rows(
                audio_anchor,
                condition_audio_t=condition_audio_t,
                seed=seed,
                noise_aug=MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
            )
            full_audio = torch.zeros(
                branch.audio_pos.shape[0],
                32,
                dtype=torch.float32,
            )
            full_audio[branch.audio_update_mask] = initial_audio
            initial_audio = full_audio

        video_sigmas = minimax_h3_time_shift_sigmas(
            num_steps=num_steps,
            shift_scale=video_shift,
            base_schedule=base_schedule,
        )
        audio_sigmas = minimax_h3_time_shift_sigmas(
            num_steps=num_steps,
            shift_scale=audio_shift,
            base_schedule=base_schedule,
        )
        return {
            "branch": branch,
            # The request-mode loop moves these onto the device itself; step mode
            # keeps them resident across steps, so normalize once for both.
            "video_rows": initial_video.to(device=self.device, dtype=torch.float32),
            "audio_rows": initial_audio.to(device=self.device, dtype=torch.float32),
            "cond_anchor": (
                None
                if visual_anchor is None
                else visual_anchor.to(device=self.device, dtype=torch.float32)
            ),
            "audio_anchor": (
                None
                if audio_anchor is None
                else audio_anchor.to(device=self.device, dtype=torch.float32)
            ),
            "sigmas_video": video_sigmas,
            "sigmas_audio": audio_sigmas,
        }

    def _unpack_denoised_rows(
        self,
        branch: MiniMaxH3DenoiseBranch,
        video_rows: torch.Tensor,
        audio_rows: torch.Tensor,
        *,
        latent_t: int,
        latent_h: int,
        latent_w: int,
        audio_t: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select the target rows and unpack them back into VAE latents."""
        target_video = video_rows[branch.update_mask_dev]
        video_latent = minimax_h3_unpatchify_video_tokens(
            target_video,
            latent_shape=(
                latent_t,
                latent_h // 2,
                latent_w // 2,
                24,
            ),
            patch_size=(1, 2, 2),
        )
        target_audio = audio_rows[branch.audio_update_mask_dev]
        audio_latent = minimax_h3_unpack_audio_tokens(
            target_audio,
            audio_t=audio_t * 2,
            audio_channel=2,
        )
        return video_latent, audio_latent

    def diffuse(
        self,
        *,
        task: str,
        text_embeddings: torch.Tensor,
        text_tags: torch.Tensor,
        seed: int,
        latent_t: int,
        latent_h: int,
        latent_w: int,
        audio_t: int,
        num_frames: int,
        num_steps: int,
        video_shift: float,
        audio_shift: float,
        base_schedule: Sequence[float] | None,
        visual_condition: torch.Tensor | None,
        visual_condition_shape: tuple[int, int, int] | None,
        audio_condition: torch.Tensor | None,
        ref_audio_t: int | None,
        ref_blocks: list[dict[str, Any]] | None = None,
        visual_condition_shapes: list[tuple[int, int, int]] | None = None,
        audio_condition_lengths: list[int] | None = None,
        keyframe_frame_indices: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self._build_denoise_inputs(
            task=task,
            text_embeddings=text_embeddings,
            text_tags=text_tags,
            seed=seed,
            latent_t=latent_t,
            latent_h=latent_h,
            latent_w=latent_w,
            audio_t=audio_t,
            num_frames=num_frames,
            num_steps=num_steps,
            video_shift=video_shift,
            audio_shift=audio_shift,
            base_schedule=base_schedule,
            visual_condition=visual_condition,
            visual_condition_shape=visual_condition_shape,
            audio_condition=audio_condition,
            ref_audio_t=ref_audio_t,
            ref_blocks=ref_blocks,
            visual_condition_shapes=visual_condition_shapes,
            audio_condition_lengths=audio_condition_lengths,
            keyframe_frame_indices=keyframe_frame_indices,
        )
        branch = inputs["branch"]
        transformer = self._transformer_for_task(task)
        counter = DenoiseWorkCounter(
            transformer,
            used_length=branch.used_len,
            video_outputs=int(branch.update_mask.sum()),
            audio_outputs=int(branch.audio_update_mask.sum()),
        )
        from vllm.media.progress import report

        report("staging_model")
        self.denoise_workload = {
            "work_accounting": "dense_tp_lora_v2",
            "partition": self.partition,
            "task": task,
            "adapter": (
                type(self.turbo_spec).__name__ if self.turbo_spec is not None else None
            ),
            "video_sigmas": list(inputs["sigmas_video"]),
            "audio_sigmas": list(inputs["sigmas_audio"]),
            "used_length": branch.used_len,
            "blocks_per_call": counter.blocks_per_call,
            "attention_algorithm": "dense",
            "cache_algorithm": None,
            "actual_backends": sorted(
                {
                    module.backend
                    for module in transformer.modules()
                    if isinstance(module, Attention)
                }
            ),
        }
        with counter, self._resident_dit_layers_on_device(enabled=True):
            torch.accelerator.synchronize()
            dist.barrier()
            torch.accelerator.synchronize()
            started = time.perf_counter()
            total = len(inputs["sigmas_video"]) - 1
            report("denoising", completed=0, total=total)
            with self.progress_bar(total=total) as progress:

                def on_step(step, video, audio):
                    progress.update()
                    report("denoising", completed=step + 1, total=total)

                video_rows, audio_rows = minimax_h3_denoise_loop(
                    model=transformer,
                    positive=branch,
                    initial_video_rows=inputs["video_rows"],
                    initial_audio_rows=inputs["audio_rows"],
                    keyframe_cond_rows=inputs["cond_anchor"],
                    audio_ref_rows=inputs["audio_anchor"],
                    sigmas_video=inputs["sigmas_video"],
                    sigmas_audio=inputs["sigmas_audio"],
                    device=self.device,
                    imgvid_cond_noise_aug_for_inference=(
                        MINIMAX_H3_IMGVID_COND_TIMESTEP
                    ),
                    audio_cond_noise_aug_for_inference=(
                        MINIMAX_H3_AUDIO_REF_COND_TIMESTEP
                    ),
                    on_step=on_step,
                    step_profiler=counter.step,
                )
            torch.accelerator.synchronize()
            dist.barrier()
            torch.accelerator.synchronize()
            self.stage_durations["denoise"] = time.perf_counter() - started
            self.useful_denoise_flops = counter.flops
            self.actual_dit_calls = counter.calls
            self.denoise_flops_by_layer = counter.by_layer
            self.redundant_denoise_flops = counter.redundant_flops
            self.redundant_flops_by_layer = counter.redundant_by_layer
            self.denoise_steps = counter.finish_steps()
            self.denoise_executed_blocks = dict(counter.blocks)

        return self._unpack_denoised_rows(
            branch,
            video_rows,
            audio_rows,
            latent_t=latent_t,
            latent_h=latent_h,
            latent_w=latent_w,
            audio_t=audio_t,
        )

    def decode(
        self,
        video_latent: torch.Tensor,
        audio_latent: torch.Tensor,
        *,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with (
            self._component_on_device(self.video_vae),
            torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=True,
            ),
        ):
            video = self.video_vae.decode_latent(video_latent)
        video = video[..., :height, :width].contiguous()
        with self._component_on_device(self.audio_vae):
            audio = self.audio_vae.decode_latent(audio_latent)
        return video, audio

    def _prepare_request_inputs(
        self,
        *,
        prompt: str,
        multi_modal_data: dict[str, Any],
        sampling: Any,
        text_conditioning: MiniMaxH3TextConditioning | None = None,
        prepared_reference_videos: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Resolve the task and output shape, then run every request-level encode.

        Shared by request-mode :meth:`forward` and step-mode
        :meth:`prepare_encode`; the returned mapping feeds :meth:`diffuse` and
        :meth:`_build_denoise_inputs` unchanged.
        """
        quality = sampling.quality
        logger.debug("MiniMax H3 request quality=%s", quality)
        extra = sampling.extra_args or {}
        task = self._resolve_task(extra.get("task"), multi_modal_data)
        turbo = getattr(self, "turbo_spec", None)
        if turbo is not None:
            from .lora import validate_adapter_sampling

            validate_adapter_sampling(turbo, task, sampling)

        raw_image = multi_modal_data.get("image")
        raw_videos = multi_modal_data.get("video")
        raw_audio = multi_modal_data.get("audio")
        images = _load_images(raw_image) if raw_image is not None else []
        video_values = (
            list(raw_videos) if isinstance(raw_videos, (list, tuple)) else raw_videos
        )
        audio_values = (
            list(raw_audio) if isinstance(raw_audio, (list, tuple)) else raw_audio
        )

        if task == "t2va" and (
            images or raw_videos is not None or raw_audio is not None
        ):
            raise H3InputError("t2va does not accept image, video, or audio conditions")
        if task == "fl2va":
            if not images:
                raise H3InputError("fl2va requires multi_modal_data.image")
            if len(images) > 2:
                raise H3InputError("fl2va accepts at most first and last images")
            if raw_videos is not None or raw_audio is not None:
                raise H3InputError("fl2va accepts image keyframes only")
        if task == "ref2va":
            video_count = (
                len(video_values)
                if isinstance(video_values, (list, tuple))
                else int(video_values is not None)
            )
            audio_is_waveform_pair = (
                isinstance(raw_audio, (list, tuple))
                and len(raw_audio) == 2
                and isinstance(raw_audio[1], (int, np.integer))
            )
            audio_count = (
                len(audio_values)
                if isinstance(audio_values, (list, tuple))
                and not audio_is_waveform_pair
                else int(raw_audio is not None)
            )
            _validate_ref2va_reference_counts(len(images), video_count, audio_count)
        elif raw_videos is not None:
            raise H3InputError(f"{task} does not accept a video condition")

        image = images[0] if images else None
        height, width, num_frames, latent_t, audio_t = self._resolve_shape(
            task, sampling, image
        )
        if task == "fl2va":
            for item in images:
                _validate_reference_image(item)
            prepared_images = [
                item.resize((width, height), Image.Resampling.LANCZOS)
                for item in images
            ]
            keyframe_frame_indices = _resolve_fl2va_keyframe_indices(extra, len(images))
        elif task == "ref2va":
            prepared_images = []
            for item in images:
                ref_width, ref_height = _reference_image_shape(item)
                prepared_images.append(
                    item.resize((ref_width, ref_height), Image.Resampling.LANCZOS)
                )
            keyframe_frame_indices = None
        else:
            prepared_images = []
            keyframe_frame_indices = None

        visual_condition = None
        visual_shape = None
        visual_shapes = None
        audio_condition = None
        ref_audio_t = None
        audio_lengths = None
        ref_blocks = None
        with tempfile.TemporaryDirectory(prefix="minimax_h3_ref2va_") as workdir:
            prepared_videos = None
            has_audio: list[bool] = []
            video_count = 0
            if raw_videos is not None:
                video_count = (
                    len(raw_videos) if isinstance(raw_videos, (list, tuple)) else 1
                )
                # File-based reference-video prep runs only on rank 0; other
                # ranks return None without touching disk. If rank 0 raises
                # (e.g. invalid file, unsupported codec) it must not exit
                # ``prepare_encode`` before the downstream broadcasts below --
                # non-zero ranks would then deadlock on them forever. Capture
                # the exception here and let every rank agree on the outcome
                # before starting any subsequent collective.
                prep_error: Exception | None = None
                try:
                    _, rank, _ = _dit_rank_world()
                    if prepared_reference_videos is not None:
                        if rank == 0:
                            prepared_videos = _reuse_prepared_reference_videos(
                                prepared_reference_videos,
                                expected_count=video_count,
                            )
                    else:
                        prepared_videos = self._prepare_reference_videos(
                            raw_videos,
                            target_frame_count=num_frames,
                            workdir=workdir,
                            start_time_seconds=extra.get("start_time_seconds"),
                        )
                except Exception as exc:
                    prep_error = exc
                _broadcast_rank0_exception(prep_error)
                has_audio_tensor = torch.zeros(
                    video_count,
                    dtype=torch.long,
                    device=self.device,
                )
                _, rank, world_size = _dit_rank_world()
                if rank == 0:
                    has_audio_tensor = torch.tensor(
                        [
                            int(item["input_has_audio"])
                            for item in prepared_videos or []
                        ],
                        dtype=torch.long,
                        device=self.device,
                    )
                if world_size > 1:
                    dist.broadcast(
                        has_audio_tensor,
                        src=0,
                        group=get_world_group().device_group,
                    )
                has_audio = [bool(value) for value in has_audio_tensor.tolist()]

            if raw_audio is not None:
                validate_reference_audio_files(raw_audio)
            standalone_audios = _load_audios(raw_audio) if raw_audio is not None else []
            validate_reference_audio_waveforms(standalone_audios)
            condition_labels: list[tuple[str, int]] = []
            for image_index in range(1, len(prepared_images) + 1):
                condition_labels.append(("image", image_index))
            audio_index = 0
            for video_index, item in enumerate(prepared_videos or (), start=1):
                if item["input_has_audio"]:
                    audio_index += 1
                    condition_labels.append(("audio", audio_index))
                condition_labels.append(("video", video_index))
            for _ in standalone_audios:
                audio_index += 1
                condition_labels.append(("audio", audio_index))

            if text_conditioning is not None:
                text_embeddings = text_conditioning.hidden_states.to(
                    device=self.device,
                    dtype=torch.float16,
                )
                text_tags = text_conditioning.token_tags.to(
                    device=self.device,
                    dtype=torch.long,
                )
            elif getattr(self, "text_encoder", None) is not None:
                text_embeddings, text_tags = self.encode_prompt(
                    task=task,
                    prompt=prompt,
                    images=prepared_images,
                    prepared_videos=prepared_videos,
                    condition_labels=condition_labels if task == "ref2va" else None,
                )
            else:
                raise H3InputError(
                    "MiniMax H3 diffusion stage requires text_encoder_output when "
                    "text_encoder is not loaded"
                )

            # ``prepared_videos`` is intentionally ``None`` on non-zero DiT
            # ranks; the distributed video encoder broadcasts the prepared
            # metadata inside ``_encode_visual_conditions``.  Use the global
            # video count here so video + standalone-audio Ref2VA requests do
            # not look like audio-only requests on those ranks.
            if video_count or prepared_images:
                visual_condition, visual_shapes = self._encode_visual_conditions(
                    prepared_images,
                    prepared_videos,
                    video_count=video_count,
                )
                (
                    embedded_audio_condition,
                    embedded_audio_lengths,
                    external_audio_condition,
                    external_audio_lengths,
                ) = self._encode_reference_audio_conditions(
                    prepared_videos,
                    has_audio=has_audio,
                    standalone_audios=standalone_audios,
                    max_duration_seconds=float(num_frames)
                    / float(sampling.fps or MINIMAX_H3_FPS),
                )
                audio_parts = [
                    item
                    for item in (embedded_audio_condition, external_audio_condition)
                    if item is not None
                ]
                audio_condition = torch.cat(audio_parts) if audio_parts else None
                audio_lengths = embedded_audio_lengths + external_audio_lengths
                ref_blocks = []
                image_shapes = visual_shapes[: len(prepared_images)]
                video_shapes = visual_shapes[len(prepared_images) :]
                for shape in image_shapes:
                    ref_blocks.append(
                        {
                            "kind": "image",
                            "latent_h": shape[1],
                            "latent_w": shape[2],
                        }
                    )
                audio_iterator = iter(embedded_audio_lengths)
                for shape, contributes_audio in zip(
                    video_shapes, has_audio, strict=True
                ):
                    ref_audio = next(audio_iterator) if contributes_audio else 0
                    ref_blocks.append(
                        {
                            "kind": "video_audio" if ref_audio else "video",
                            "ref_audio_t": ref_audio,
                            "latent_t": shape[0],
                            "latent_h": shape[1],
                            "latent_w": shape[2],
                        }
                    )
                for ref_audio_t in external_audio_lengths:
                    ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t})
            elif standalone_audios:
                raise H3InputError(
                    "standalone audio references require a Ref2VA visual reference"
                )

            if visual_shapes and len(visual_shapes) == 1:
                visual_shape = visual_shapes[0]
            if audio_lengths:
                if any(length < 80 or length > 600 for length in audio_lengths):
                    raise H3InputError(
                        "MiniMax H3 audio references must each be between 2 and 15 "
                        "seconds"
                    )
                if sum(audio_lengths) > 600:
                    raise H3InputError(
                        "MiniMax H3 audio references must be at most 15 seconds in "
                        "total"
                    )
                if len(audio_lengths) == 1:
                    ref_audio_t = audio_lengths[0]

        seed = int(sampling.seed if sampling.seed is not None else 42)
        base_schedule, num_steps = self._resolve_sigma_positions(task, sampling)
        active_turbo = turbo is not None and sampling.lora_scale != 0
        video_shift = float(
            extra.get(
                "flow_shift",
                turbo.video_shift if active_turbo else self.default_video_shift,
            )
        )
        audio_shift = float(
            extra.get(
                "audio_flow_shift",
                turbo.audio_shift if active_turbo else self.default_audio_shift,
            )
        )
        num_outputs = _resolve_minimax_h3_num_outputs(sampling.num_outputs_per_prompt)
        return {
            "task": task,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "latent_t": latent_t,
            "latent_h": height // 16,
            "latent_w": width // 16,
            "audio_t": audio_t,
            "text_embeddings": text_embeddings,
            "text_tags": text_tags,
            "visual_condition": visual_condition,
            "visual_condition_shape": visual_shape,
            "audio_condition": audio_condition,
            "ref_audio_t": ref_audio_t,
            "ref_blocks": ref_blocks,
            "visual_condition_shapes": visual_shapes,
            "audio_condition_lengths": audio_lengths,
            "keyframe_frame_indices": keyframe_frame_indices,
            "seed": seed,
            "num_steps": num_steps,
            "video_shift": video_shift,
            "audio_shift": audio_shift,
            "base_schedule": base_schedule,
            "num_outputs": num_outputs,
        }

    @staticmethod
    def _denoise_kwargs(context: dict[str, Any]) -> dict[str, Any]:
        """Select the denoise-input arguments from a prepared request context."""
        return {key: context[key] for key in _MINIMAX_H3_DENOISE_INPUT_KEYS}
