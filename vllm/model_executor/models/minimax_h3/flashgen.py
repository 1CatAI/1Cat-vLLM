# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native FlashGen four-step adapters and restoration of pruned AdaLN.

Contract: Omni's minimax_h3/npu/lora.py at the revision in UPSTREAM.md.
The export is device-independent. Native execution uses H3's staged TP layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from vllm.logger import init_logger

from .config import H3InputError
from .sigma_schedule import DMD2SigmaSchedule

logger = init_logger(__name__)
FLASHGEN_FILENAME = "minimax_h3_t2va_flashgen_4step_v1.0_768p_bf16.safetensors"
_RANK = 64
_DIMS = {
    "attn.qkv_proj": (5376, 21504),
    "attn.out_proj": (7168, 5376),
    "mlp.fc1": (5376, 28672),
    "mlp.fc2": (14336, 5376),
    "adaln_proj.linear": (2688, 96768),
}
_TARGETS = {
    f"blocks.{index}.{suffix}": dims
    for index in range(50)
    for suffix, dims in _DIMS.items()
}
_TARGETS.update(
    {
        f"token_refiner.blocks.{index}.{suffix}": dims
        for index in range(2)
        for suffix, dims in _DIMS.items()
        if suffix != "adaln_proj.linear"
    }
)
_TARGETS["final_layer.adaln_proj.linear"] = (2688, 10752)
_A = ".lora_A.default.weight"
_B = ".lora_B.default.weight"


@dataclass(frozen=True)
class FlashGenSpec:
    filename: str
    base_schedule: tuple[float, ...]
    rank: int = 64
    alpha: float = 64.0
    video_shift: float = 12.0
    audio_shift: float = 3.0
    denoise_steps: int = 4
    api_steps: int = 4
    supported_tasks: frozenset[str] = frozenset({"t2va"})


def inspect_flashgen_lora(path: str | Path, partition: str) -> FlashGenSpec:
    path = Path(path)
    if path.name != FLASHGEN_FILENAME or not path.is_file():
        raise H3InputError(f"FlashGen requires {FLASHGEN_FILENAME}")
    if partition != "fl2va":
        raise H3InputError("FlashGen requires the FL2VA partition and task=t2va")
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
        for key, expected in (
            ("key_format", "minimax-h3-native"),
            ("qkv_layout", "grouped"),
            ("tasks", "t2va"),
        ):
            if metadata.get(key) != expected:
                raise H3InputError(f"FlashGen requires {key}={expected}")
        try:
            rank = int(metadata.get("lora_rank", ""))
            alpha = float(metadata.get("lora_alpha", ""))
            positions = tuple(
                float(value) for value in metadata["base_schedule"].split(",")
            )
            schedule = DMD2SigmaSchedule.from_positions(positions)
        except (KeyError, TypeError, ValueError) as exc:
            raise H3InputError(
                "FlashGen requires rank/alpha and a valid base_schedule"
            ) from exc
        if rank != _RANK or alpha != _RANK or schedule.num_inference_steps != 4:
            raise H3InputError(
                "FlashGen requires rank=alpha=64 and four schedule intervals"
            )
        expected = {
            f"transformer.{target}{side}" for target in _TARGETS for side in (_A, _B)
        }
        keys = set(checkpoint.keys())
        if keys != expected:
            raise H3InputError(
                "Incomplete FlashGen tensor set: "
                f"missing={sorted(expected - keys)[:3]}, "
                f"unexpected={sorted(keys - expected)[:3]}"
            )
        for target, (inputs, outputs) in _TARGETS.items():
            for side, shape in ((_A, (rank, inputs)), (_B, (outputs, rank))):
                name = f"transformer.{target}{side}"
                value = checkpoint.get_slice(name)
                if tuple(value.get_shape()) != shape or value.get_dtype() not in (
                    "F16",
                    "BF16",
                    "F32",
                ):
                    raise H3InputError(
                        f"Invalid FlashGen tensor {name}: expected {shape}"
                    )
    return FlashGenSpec(
        filename=path.name, base_schedule=positions, rank=rank, alpha=alpha
    )


def shard_native_pair(a, b, name, tp_rank, tp_size, *, heads=56, head_dim=128):
    if name.endswith(("attn.out_proj", "mlp.fc2")):
        a = a.chunk(tp_size, dim=1)[tp_rank]
    elif name.endswith("attn.qkv_proj"):
        # Export: [head, Q/K/V, channel]. Runtime: [Q_local, K_local, V_local].
        grouped = b.reshape(heads, 3, head_dim, b.shape[1])
        b = torch.cat(
            [
                part.reshape(heads * head_dim, b.shape[1]).chunk(tp_size)[tp_rank]
                for part in grouped.unbind(dim=1)
            ]
        )
    elif name.endswith("mlp.fc1"):
        # This native export is already [gate, up], unlike LightX2V Diffusers.
        b = torch.cat([part.chunk(tp_size)[tp_rank] for part in b.chunk(2)])
    else:
        b = b.chunk(tp_size)[tp_rank]
    return a.contiguous(), b.contiguous()


def install_flashgen_lora(model, path: str | Path, partition: str) -> FlashGenSpec:
    from .lora import TurboLinearMethod

    spec = inspect_flashgen_lora(path, partition)
    modules = dict(model.named_modules())
    pending = {}
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        for name in _TARGETS:
            layer = modules.get(name)
            if layer is None or isinstance(layer.quant_method, TurboLinearMethod):
                raise H3InputError(
                    f"Missing or already adapted FlashGen target: {name}"
                )
            a, b = shard_native_pair(
                checkpoint.get_tensor(f"transformer.{name}{_A}"),
                checkpoint.get_tensor(f"transformer.{name}{_B}"),
                name,
                layer.tp_rank,
                layer.tp_size,
                heads=model.arch.num_attention_heads,
                head_dim=model.arch.attention_head_dim,
            )
            a, b = a.half(), b.half()
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise H3InputError(
                    f"FlashGen target cannot be represented in FP16: {name}"
                )
            if (b.shape[0], a.shape[1]) != tuple(layer.weight.shape):
                raise H3InputError(
                    f"FlashGen target shape mismatch: {name}; pruned AdaLN must be "
                    "restored from original weights"
                )
            pending[name] = (a, b)
    for name, (a, b) in pending.items():
        layer = modules[name]
        layer._sm70_f16_forbidden = True
        layer.register_buffer("h3_lora_a_0", a)
        layer.register_buffer("h3_lora_b_0", b)
        layer.quant_method = TurboLinearMethod(
            layer.quant_method, [(0, 0, b.shape[0])], spec.alpha / spec.rank
        )
    logger.info(
        "H3 FlashGen: %d native targets, rank=%d alpha=%g, four denoiser calls, "
        "base schedule=%s",
        len(pending),
        spec.rank,
        spec.alpha,
        spec.base_schedule,
    )
    return spec


def is_dense_adaln_weight(name: str) -> bool:
    return name.startswith("time_embedder.") or ".adaln_proj.linear." in name


def restore_dense_adaln_weights(pruned_weights, original_dir: str | Path):
    """Keep INT8 weights untouched and restore the original AdaLN/time modules.

    The rank-64 adapter needs 2688-wide time features; a pruned 8-wide curve
    cannot represent them. Restoration changes only those base components.
    """
    from .weights import iter_checkpoint_weights

    expected = {
        f"{name}.{side}"
        for name in _TARGETS
        if name.endswith("adaln_proj.linear")
        for side in ("weight", "bias")
    } | {
        f"time_embedder.{name}.{side}"
        for name in ("proj_in", "proj_out")
        for side in ("weight", "bias")
    }
    for name, weight in pruned_weights:
        if name != "adaln_t_table" and not is_dense_adaln_weight(name):
            yield name, weight
    restored = set()
    for name, weight in iter_checkpoint_weights(original_dir, include=expected):
        restored.add(name)
        yield name, weight
    if restored != expected:
        raise H3InputError(
            "Original H3 checkpoint is missing AdaLN restoration tensors: "
            f"{sorted(expected - restored)}"
        )
    logger.info(
        "H3 FlashGen restored %d original AdaLN/time tensors; INT8 backbone retained",
        len(restored),
    )
