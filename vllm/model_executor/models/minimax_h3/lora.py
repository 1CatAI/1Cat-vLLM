# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""LightX2V Turbo adapters for native H3, including signed INT8 ConvRot bases.

Artifact contracts follow Omni's MiniMax H3 loader (see UPSTREAM.md). Deltas
run on unrotated activations, before the base layer's TP collective. They never
modify quantized weights. Registered buffers participate in native CPU staging.
"""

from __future__ import annotations

import math
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path

import regex as re
import torch
from safetensors import safe_open

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearMethodBase

from .config import H3InputError
from .fasth3 import FASTH3_FILENAME, FastH3Spec
from .flashgen import FLASHGEN_FILENAME, FlashGenSpec
from .quantization import fp16_gemm_input

logger = init_logger(__name__)
lora_scale: ContextVar[float] = ContextVar("h3_lora_scale", default=0.0)

_NAME = re.compile(
    r"^minimax_h3_(?P<task>fl2v|ref2v)_turbo_(?P<steps>[48])step"
    r"_v\d+\.\d+(?P<res>_768p)?(?:_bf16)?\.safetensors$"
)
_A = ".lora_A.default.weight"
_B = ".lora_B.default.weight"
_DIMS = {
    "attn.to_q": (5376, 7168),
    "attn.to_k": (5376, 7168),
    "attn.to_v": (5376, 7168),
    "attn.to_out.0": (7168, 5376),
    "ff.net.0.proj": (5376, 28672),
    "ff.net.2": (14336, 5376),
}
_TARGETS = {
    f"{prefix}.{index}.{suffix}": suffix
    for prefix, count in (
        ("transformer_blocks", 50),
        ("token_refiner.refiner_blocks", 2),
    )
    for index in range(count)
    for suffix in _DIMS
}


@dataclass(frozen=True)
class TurboSpec:
    filename: str
    task_family: str
    denoise_steps: int
    video_shift: float
    audio_shift: float = 3.0
    rank: int = 128
    alpha: float = 8.0
    base_schedule: tuple[float, ...] | None = None

    @property
    def api_steps(self) -> int:
        return self.sigma_points

    @property
    def sigma_points(self) -> int:
        return self.denoise_steps + 1

    @property
    def supported_tasks(self) -> frozenset[str]:
        return (
            frozenset({"ref2va"})
            if self.task_family == "ref2v"
            else frozenset({"t2va", "fl2va"})
        )


def parse_turbo_filename(name: str) -> TurboSpec | None:
    match = _NAME.fullmatch(name)
    if match is None:
        return None
    return TurboSpec(
        filename=name,
        task_family=match["task"],
        denoise_steps=int(match["steps"]),
        video_shift=6.0 if match["res"] else 12.0,
    )


def select_turbo_file(artifact: str | Path) -> Path:
    path = Path(artifact)
    if path.is_dir():
        candidates = sorted(
            p for p in path.glob("*.safetensors") if parse_turbo_filename(p.name)
        )
        if len(candidates) != 1:
            raise H3InputError("--lora-path must select exactly one Turbo artifact")
        path = candidates[0]
    if not path.is_file() or parse_turbo_filename(path.name) is None:
        raise H3InputError(
            "--lora-path requires a LightX2V 4/8-step Diffusers safetensors file; "
            "ComfyUI, FlashGen and FastH3 exports have different layouts"
        )
    return path


def select_adapter_file(artifact: str | Path) -> Path:
    path = Path(artifact)
    if path.is_dir():
        candidates = sorted(
            file
            for file in path.glob("*.safetensors")
            if file.name in (FLASHGEN_FILENAME, FASTH3_FILENAME)
            or parse_turbo_filename(file.name)
        )
        if len(candidates) != 1:
            raise H3InputError("--lora-path must select exactly one H3 adapter")
        path = candidates[0]
    if path.is_file() and path.name in (FLASHGEN_FILENAME, FASTH3_FILENAME):
        return path
    return select_turbo_file(path)


def inspect_adapter(
    artifact: str | Path, partition: str
) -> TurboSpec | FlashGenSpec | FastH3Spec:
    path = select_adapter_file(artifact)
    if path.name == FASTH3_FILENAME:
        from .fasth3 import inspect_fasth3_lora

        return inspect_fasth3_lora(path, partition)
    if path.name == FLASHGEN_FILENAME:
        from .flashgen import inspect_flashgen_lora

        return inspect_flashgen_lora(path, partition)
    return inspect_turbo_lora(path, partition)


def inspect_deployment_adapter(config):
    spec = inspect_adapter(config.lora_path, config.partition)
    if isinstance(spec, FastH3Spec) and config.transformer_path:
        raise H3InputError(
            "FastH3 Dense fusion requires original weights; omit --transformer-path"
        )
    return spec


def install_adapter(model, artifact: str | Path, partition: str):
    path = select_adapter_file(artifact)
    if path.name == FASTH3_FILENAME:
        raise H3InputError("FastH3 must be fused before loading and CPU staging")
    if path.name == FLASHGEN_FILENAME:
        from .flashgen import install_flashgen_lora

        return install_flashgen_lora(model, path, partition)
    return install_turbo_lora(model, path, partition)


def inspect_turbo_lora(artifact: str | Path, partition: str) -> TurboSpec:
    """Validate names, metadata and global shapes without loading tensor data."""
    path = select_turbo_file(artifact)
    spec = parse_turbo_filename(path.name)
    assert spec is not None
    expected_partition = "ref2va" if spec.task_family == "ref2v" else "fl2va"
    if partition != expected_partition:
        raise H3InputError(f"{path.name} requires partition={expected_partition}")
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
        if metadata.get("key_format") not in (None, "minimax-h3-diffusers"):
            raise H3InputError("Turbo requires key_format=minimax-h3-diffusers")
        try:
            alpha = float(metadata.get("alpha", 8.0))
        except (TypeError, ValueError) as exc:
            raise H3InputError("Turbo alpha must be finite and positive") from exc
        if not math.isfinite(alpha) or alpha <= 0:
            raise H3InputError("Turbo alpha must be finite and positive")
        spec = replace(spec, alpha=alpha)
        expected = {target + side for target in _TARGETS for side in (_A, _B)}
        keys = set(checkpoint.keys())
        if keys != expected:
            raise H3InputError(
                "Incomplete or unsupported Turbo tensor set: "
                f"missing={sorted(expected - keys)[:5]}, "
                f"unexpected={sorted(keys - expected)[:5]}"
            )
        for target, suffix in _TARGETS.items():
            input_dim, output_dim = _DIMS[suffix]
            for side, shape in (
                (_A, (spec.rank, input_dim)),
                (_B, (output_dim, spec.rank)),
            ):
                tensor = checkpoint.get_slice(target + side)
                if tuple(tensor.get_shape()) != shape or tensor.get_dtype() not in (
                    "F16",
                    "BF16",
                    "F32",
                ):
                    raise H3InputError(
                        f"Invalid Turbo tensor shape/dtype: {target + side}, "
                        f"expected {shape}"
                    )
    return spec


def validate_turbo_sampling(spec: TurboSpec, task: str, sampling) -> None:
    validate_adapter_sampling(spec, task, sampling)


def validate_adapter_sampling(
    spec: TurboSpec | FlashGenSpec | FastH3Spec, task: str, sampling
) -> None:
    if isinstance(spec, FastH3Spec) and sampling.lora_scale != 1:
        raise H3InputError("FastH3 is fused; request lora_scale must be 1")
    if sampling.lora_scale == 0:
        return
    if task not in spec.supported_tasks:
        raise H3InputError(
            f"{spec.filename} supports {sorted(spec.supported_tasks)}, got {task}"
        )
    if sampling.num_inference_steps != spec.api_steps:
        raise H3InputError(
            f"{spec.filename} requires num_inference_steps={spec.api_steps} "
            f"({spec.denoise_steps} actual denoiser calls)"
        )
    for key, expected in (
        ("flow_shift", spec.video_shift),
        ("audio_flow_shift", spec.audio_shift),
    ):
        if not math.isclose(float(sampling.extra_args.get(key, expected)), expected):
            raise H3InputError(f"{spec.filename} requires {key}={expected:g}")


def _shard_pair(a, b, suffix, tp_rank, tp_size):
    """Return TP-local A/B and the output slice within a fused QKV layer."""
    offset = 0
    if suffix in ("attn.to_out.0", "ff.net.2"):
        a = a.chunk(tp_size, dim=1)[tp_rank]
    elif suffix == "ff.net.0.proj":
        # Diffusers is [value, gate]; native SiluAndMul is [gate, value].
        value, gate = b.chunk(2, dim=0)
        b = torch.cat((gate.chunk(tp_size)[tp_rank], value.chunk(tp_size)[tp_rank]))
    else:
        b = b.chunk(tp_size)[tp_rank]
        offset = ("attn.to_q", "attn.to_k", "attn.to_v").index(suffix) * b.shape[0]
    return a.contiguous(), b.contiguous(), offset


def _linear_fp32(x, weight):
    if not x.is_cuda:
        return torch.nn.functional.linear(x.float(), weight.float())
    from .cuda_ops import w8a16_extension

    values, scale = fp16_gemm_input(x)
    output = w8a16_extension().gemm(values, weight, True)
    if scale is not None:
        output = output * scale
    return output.reshape(*x.shape[:-1], weight.shape[0])


class TurboLinearMethod(LinearMethodBase):
    """Apply the adapter before Column/RowParallelLinear gathers or reduces."""

    def __init__(self, base, parts, alpha_over_rank):
        self.base = base
        self.parts = parts
        self.alpha_over_rank = alpha_over_rank

    def create_weights(self, *args, **kwargs):
        raise RuntimeError("Turbo is installed after base checkpoint loading")

    def apply(self, layer, x, bias=None):
        output = self.base.apply(layer, x, bias)
        scale = lora_scale.get() * self.alpha_over_rank
        if scale == 0:
            return output
        dtype = output.dtype
        output = output.float()
        for index, offset, width in self.parts:
            a = getattr(layer, f"h3_lora_a_{index}")
            b = getattr(layer, f"h3_lora_b_{index}")
            delta = _linear_fp32(_linear_fp32(x, a), b)
            output[..., offset : offset + width].add_(delta, alpha=scale)
        return output.to(dtype)


def install_turbo_lora(model, artifact: str | Path, partition: str) -> TurboSpec:
    """Install one immutable adapter; request-local scale zero bypasses it."""
    path = select_turbo_file(artifact)
    spec = inspect_turbo_lora(path, partition)
    modules = dict(model.named_modules())
    pending = {}
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        for target, suffix in _TARGETS.items():
            name = target.replace("transformer_blocks.", "blocks.").replace(
                "token_refiner.refiner_blocks.", "token_refiner.blocks."
            )
            prefix = name[: -len(suffix)]
            native_suffix = {
                "attn.to_q": "attn.qkv_proj",
                "attn.to_k": "attn.qkv_proj",
                "attn.to_v": "attn.qkv_proj",
                "attn.to_out.0": "attn.out_proj",
                "ff.net.0.proj": "mlp.fc1",
                "ff.net.2": "mlp.fc2",
            }[suffix]
            name = prefix + native_suffix
            layer = modules.get(name)
            if layer is None or isinstance(layer.quant_method, TurboLinearMethod):
                raise H3InputError(f"Missing or already adapted Turbo target: {name}")
            a, b, offset = _shard_pair(
                checkpoint.get_tensor(target + _A),
                checkpoint.get_tensor(target + _B),
                suffix,
                layer.tp_rank,
                layer.tp_size,
            )
            a, b = a.half(), b.half()
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise H3InputError(
                    f"Turbo target cannot be represented in FP16: {target}"
                )
            if (
                a.shape[1] != layer.weight.shape[1]
                or offset + b.shape[0] > layer.weight.shape[0]
            ):
                raise H3InputError(
                    f"Turbo target does not match local base dimensions: {name}"
                )
            pending.setdefault(name, []).append((a, b, offset))
    # No partial installation on malformed input; validate every target first.
    for name, tensors in pending.items():
        layer = modules[name]
        # Generic dense dispatch can bypass quant_method.apply after preparing
        # a base weight. The adapter must remain in the executed route.
        layer._sm70_f16_forbidden = True
        parts = []
        for index, (a, b, offset) in enumerate(tensors):
            layer.register_buffer(f"h3_lora_a_{index}", a)
            layer.register_buffer(f"h3_lora_b_{index}", b)
            parts.append((index, offset, b.shape[0]))
        layer.quant_method = TurboLinearMethod(
            layer.quant_method, parts, spec.alpha / spec.rank
        )
    logger.info(
        "H3 Turbo %s: %d raw targets bound to %d TP-local layers, rank=%d alpha=%g, "
        "%d denoiser calls, shifts=%g/%g; original activation basis, staged buffers",
        spec.filename,
        len(_TARGETS),
        len(pending),
        spec.rank,
        spec.alpha,
        spec.denoise_steps,
        spec.video_shift,
        spec.audio_shift,
    )
    return spec
