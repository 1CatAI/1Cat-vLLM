# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""FastH3 Dense checkpoint fusion before native TP loading and CPU staging.

Adapted from Omni's minimax_h3/fasth3.py at the revision in UPSTREAM.md.
The release adds B @ A and full-rank weight/bias deltas to original weights.
It has no request-switchable base path and cannot edit serialized INT8 bytes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors import safe_open

from vllm.logger import init_logger

from .config import H3InputError

logger = init_logger(__name__)
FASTH3_FILENAME = "adapter_model.safetensors"
_DENSE_IDENTITY = "fastvideo/fastvideo-fasth3-dense-4-step-v1"
_VSA_IDENTITIES = frozenset(
    f"fastvideo/fastvideo-fasth3-4-step-{version}" for version in ("v1", "v1.1", "v1.2")
)
_MODEL_TARGETS = {
    "proj_in": "video_patch_proj",
    "proj_out": "final_layer.video_out",
    "audio_proj_in": "audio_patch_proj",
    "audio_proj_out": "final_layer.audio_out",
    "context_embedder": "condition_proj",
    "time_embedder.linear_1": "time_embedder.proj_in",
    "time_embedder.linear_2": "time_embedder.proj_out",
    "norm_out.linear": "final_layer.adaln_proj.linear",
    "norm_out.norm": "final_layer.norm",
}
_BLOCK_TARGETS = {
    "attn.to_q": ("attn.qkv_proj", "q"),
    "attn.to_k": ("attn.qkv_proj", "k"),
    "attn.to_v": ("attn.qkv_proj", "v"),
    "attn.to_out.0": ("attn.out_proj", "plain"),
    "attn.to_gate_compress": ("attn.to_gate_compress", "plain"),
    "ff.net.0.proj": ("mlp.fc1", "swap"),
    "ff.net.2": ("mlp.fc2", "plain"),
    "adaln_proj.linear": ("adaln_proj.linear", "plain"),
    "norm1": ("norm1", "plain"),
    "norm2": ("norm2", "plain"),
}
_PREFIXES = {
    "transformer_blocks.": ("blocks.", 50),
    "token_refiner.refiner_blocks.": ("token_refiner.blocks.", 2),
}
_ROLES = {
    ".lora_A.weight": "a",
    ".lora_B.weight": "b",
    ".diff_b": "bias",
    ".diff": "diff",
    ".set_weight": "set",
}


@dataclass(frozen=True)
class FastH3Spec:
    filename: str = FASTH3_FILENAME
    base_schedule: tuple[float, ...] = (0.999, 0.749, 0.5, 0.25, 0.0)
    rank: int = 64
    video_shift: float = 12.0
    audio_shift: float = 3.0
    denoise_steps: int = 4
    api_steps: int = 4
    supported_tasks: frozenset[str] = frozenset({"t2va"})
    requires_vsa: bool = False


@dataclass
class _Patch:
    layout: str
    pairs: dict[str, dict[str, str]] = field(default_factory=dict)
    diff: str | None = None
    assigned: str | None = None


def _native_target(module: str):
    if module in _MODEL_TARGETS:
        return _MODEL_TARGETS[module], "plain", None
    for prefix, (native, count) in _PREFIXES.items():
        if module.startswith(prefix):
            index, _, suffix = module[len(prefix) :].partition(".")
            if index.isdigit() and 0 <= int(index) < count and suffix in _BLOCK_TARGETS:
                target, layout = _BLOCK_TARGETS[suffix]
                return f"{native}{index}.{target}", layout, (prefix, int(index))
    raise H3InputError(f"Unknown FastH3 target: {module}")


def _read_index(path: str | Path, partition: str):
    path = Path(path)
    if path.name != FASTH3_FILENAME or not path.is_file():
        raise H3InputError("FastH3 requires its explicit adapter_model.safetensors")
    if partition != "fl2va":
        raise H3InputError("FastH3 Dense requires the FL2VA base and task=t2va")
    patches: dict[str, _Patch] = {}
    coverage = {prefix: set() for prefix in _PREFIXES}
    counted = {"low_rank_tensors": 0, "diff_tensors": 0, "set_weight_tensors": 0}
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
        identity = metadata.get("finetuned_model", "").lower()
        requires_vsa = identity in _VSA_IDENTITIES
        if (
            metadata.get("format") != "fastvideo-lora-v2"
            or identity not in (_DENSE_IDENTITY, *_VSA_IDENTITIES)
            or metadata.get("base_model", "").lower() != "minimaxai/minimax-h3"
            or metadata.get("rank") != "64"
        ):
            raise H3InputError(
                "Only the official rank-64 FastH3 Dense/VSA releases are supported"
            )
        for name in checkpoint.keys():  # noqa: SIM118
            match = next(
                (
                    (name[: -len(suffix)], role)
                    for suffix, role in _ROLES.items()
                    if name.endswith(suffix)
                ),
                None,
            )
            if match is None:
                raise H3InputError(f"Unknown FastH3 tensor role: {name}")
            module, role = match
            native, layout, block = _native_target(module)
            if block is not None:
                coverage[block[0]].add(block[1])
            param = f"{native}.{'bias' if role == 'bias' else 'weight'}"
            patch = patches.setdefault(param, _Patch(layout=layout))
            value = checkpoint.get_slice(name)
            shape = value.get_shape()
            if value.get_dtype() not in ("F16", "BF16", "F32"):
                raise H3InputError(f"FastH3 requires floating-point deltas: {name}")
            is_gate = native.endswith(".attn.to_gate_compress")
            if role == "set" or is_gate:
                if (
                    not requires_vsa
                    or role != "set"
                    or not is_gate
                    or not native.startswith("blocks.")
                    or len(shape) != 2
                    or min(shape) <= 0
                    or patch.assigned is not None
                ):
                    raise H3InputError(f"Invalid FastH3 VSA compression gate: {name}")
                patch.assigned = name
                counted["set_weight_tensors"] += 1
                continue
            if role in ("a", "b"):
                if len(shape) != 2 or shape[0 if role == "a" else 1] != 64:
                    raise H3InputError(
                        f"FastH3 requires rank-64 matrix factors: {name}"
                    )
                patch.pairs.setdefault(layout, {})[role] = name
                counted["low_rank_tensors"] += 1
            else:
                if layout != "plain" or patch.diff is not None:
                    raise H3InputError(
                        f"Unsupported fused-layout/duplicate FastH3 delta: {name}"
                    )
                patch.diff = name
                counted["diff_tensors"] += 1
        for field_name, actual in counted.items():
            try:
                declared = int(metadata[field_name])
            except (KeyError, TypeError, ValueError) as exc:
                raise H3InputError(f"FastH3 must declare {field_name}") from exc
            if declared != actual:
                raise H3InputError(
                    f"FastH3 {field_name}: declared {declared}, found {actual}"
                )
        for prefix, (_, count) in _PREFIXES.items():
            if coverage[prefix] != set(range(count)):
                raise H3InputError(f"FastH3 must edit every {prefix} block")
        expected_gates = (
            {f"blocks.{i}.attn.to_gate_compress.weight" for i in range(50)}
            if requires_vsa
            else set()
        )
        actual_gates = {key for key, patch in patches.items() if patch.assigned}
        if actual_gates != expected_gates:
            raise H3InputError(
                "FastH3 VSA must assign every main-block compression gate"
            )
        for param, patch in patches.items():
            if any(set(pair) != {"a", "b"} for pair in patch.pairs.values()):
                raise H3InputError(f"FastH3 has an unpaired factor for {param}")
            if patch.layout in ("q", "k", "v") and set(patch.pairs) != {"q", "k", "v"}:
                raise H3InputError(
                    f"FastH3 grouped QKV requires all three projections: {param}"
                )
    return FastH3Spec(requires_vsa=requires_vsa), patches


def inspect_fasth3_lora(path: str | Path, partition: str) -> FastH3Spec:
    return _read_index(path, partition)[0]


class FastH3Fusion:
    """Read one parameter's edits at a time; round before normal TP loading."""

    def __init__(self, path, *, partition="fl2va", head_dim=128, device="cpu"):
        self.path = Path(path)
        self.spec, self.patches = _read_index(path, partition)
        self.head_dim = head_dim
        self.device = torch.device(device)
        self.applied: set[str] = set()
        self.started = False

    def _fuse(self, checkpoint, name, weight):
        patch = self.patches.get(name)
        if patch is None:
            return weight
        if patch.assigned is not None:
            raise H3InputError("FastH3 gate must be new, not replace a base parameter")
        if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise H3InputError("FastH3 fusion requires original floating-point weights")

        def read(key):
            return checkpoint.get_tensor(key).to(self.device).float()

        delta = None
        if patch.pairs:
            products = {}
            for layout, pair in patch.pairs.items():
                a, b = read(pair["a"]), read(pair["b"])
                if layout == "swap":
                    if b.shape[0] % 2:
                        raise H3InputError(
                            f"FastH3 FFN rows cannot split in half: {name}"
                        )
                    up, gate = b.chunk(2)
                    b = torch.cat((gate, up))
                products[layout] = b @ a
            if patch.layout in ("q", "k", "v"):
                parts = [products[key] for key in ("q", "k", "v")]
                if any(part.shape != parts[0].shape for part in parts):
                    raise H3InputError(f"FastH3 QKV dimensions disagree: {name}")
                if parts[0].shape[0] % self.head_dim:
                    raise H3InputError(f"FastH3 QKV rows do not match head_dim: {name}")
                delta = torch.stack(
                    [part.reshape(-1, self.head_dim, part.shape[1]) for part in parts],
                    dim=1,
                ).flatten(0, 2)
            else:
                delta = products[patch.layout]
        if patch.diff is not None:
            diff = read(patch.diff)
            if delta is not None:
                if diff.shape != delta.shape:
                    raise H3InputError(
                        f"FastH3 low/full-rank delta shapes disagree: {name}"
                    )
                delta.add_(diff)
            else:
                # FP32 CPU tensors may alias the mapped adapter. Fusion must
                # not modify a source tensor retained by the checkpoint reader.
                delta = diff.clone()
        if delta is None or delta.shape != weight.shape:
            raise H3InputError(f"FastH3 delta does not match base weight shape: {name}")
        # Match the release's reconstruction/rounding before native FP16/FP32
        # parameter loading. Returning CPU data lets staging capture fused weights.
        fused = delta.add_(weight.to(self.device)).to(weight.dtype)
        if not torch.isfinite(fused).all():
            raise H3InputError(f"FastH3 fusion produced non-finite weights: {name}")
        self.applied.add(name)
        return fused.to(weight.device)

    def apply(self, weights):
        if self.started:
            raise H3InputError("FastH3 fusion is single-use")
        self.started = True
        with safe_open(self.path, framework="pt", device="cpu") as checkpoint:
            for name, weight in weights:
                if name in self.applied:
                    raise H3InputError(f"Duplicate FastH3 base parameter: {name}")
                yield name, self._fuse(checkpoint, name, weight)
            for name, patch in self.patches.items():
                if patch.assigned is not None:
                    value = checkpoint.get_tensor(patch.assigned)
                    if not torch.isfinite(value).all():
                        raise H3InputError(f"FastH3 gate is non-finite: {name}")
                    self.applied.add(name)
                    yield name, value
        self.validate_fully_applied()

    def validate_fully_applied(self, loaded=None):
        arrived = self.applied if loaded is None else self.applied & set(loaded)
        if missing := self.patches.keys() - arrived:
            raise H3InputError(
                f"FastH3 edits did not reach the model: {sorted(missing)[:5]}"
            )
        logger.info(
            "H3 FastH3 Dense fused %d parameters before CPU staging", len(arrived)
        )
