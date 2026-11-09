# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preserve checkpoint FP8 blocks across padded MTP tensor-parallel shards."""

from collections.abc import Iterable, Iterator

import torch


def padded_mtp_fp8_width(intermediate: int, tp_size: int) -> int:
    """Use one block-aligned physical width for all logical TP slices."""
    if intermediate <= 0 or tp_size <= 0:
        raise ValueError("MTP expert width and TP size must be positive")
    max_offset = max((rank * intermediate) % 128 for rank in range(tp_size))
    return (intermediate + max_offset + 127) // 128 * 128


def pad_mtp_fp8_checkpoint_matrix(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    down_projection: bool,
    tp_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand a global matrix for the ordinary block-aware TP loader.

    Retain the original FP8 bytes and block scales. Each TP slice starts at
    its original offset within a 128-wide block; zero padding masks elements
    owned by neighboring ranks. Gate/up rows and down columns therefore use
    the same physical intermediate coordinates, without requantization.
    """
    if weight.ndim != 2 or weight.dtype != torch.float8_e4m3fn:
        raise ValueError("MTP checkpoint weights must be 2D FP8 E4M3 tensors")
    expected = tuple((dim + 127) // 128 for dim in weight.shape)
    if tuple(scale.shape) != expected or not scale.is_floating_point():
        raise ValueError(f"MTP checkpoint scales must have block shape {expected}")
    if not torch.isfinite(weight.float()).all():
        raise ValueError("MTP checkpoint weights must be finite")
    if not torch.isfinite(scale).all() or (scale < 0).any():
        raise ValueError("MTP checkpoint scales must be finite and nonnegative")
    kernel_scale = scale.to(torch.float16)
    if (
        not torch.isfinite(kernel_scale).all()
        or ((scale > 0) & (kernel_scale == 0)).any()
    ):
        raise ValueError("MTP checkpoint scales are outside the SM70 FP16 scale range")
    axis = 1 if down_projection else 0
    if tp_size <= 0 or weight.shape[axis] % tp_size:
        raise ValueError("MTP checkpoint intermediate width must divide TP size")
    if weight.shape[1 - axis] % 128:
        raise ValueError("MTP checkpoint hidden size must be divisible by 128")
    logical = weight.shape[axis] // tp_size
    padded = padded_mtp_fp8_width(logical, tp_size)
    shape = list(weight.shape)
    shape[axis] = padded * tp_size
    result = weight.new_zeros(shape)
    scale_shape = list(expected)
    scale_shape[axis] = padded // 128 * tp_size
    result_scale = scale.new_ones(scale_shape)
    for rank in range(tp_size):
        start = rank * logical
        offset = start % 128
        result.narrow(axis, rank * padded + offset, logical).copy_(
            weight.narrow(axis, start, logical)
        )
        block_count = (offset + logical + 127) // 128
        result_scale.narrow(axis, rank * (padded // 128), block_count).copy_(
            scale.narrow(axis, start // 128, block_count)
        )
    return result, result_scale


def prepare_mtp_fp8_checkpoint(
    weights: Iterable[tuple[str, torch.Tensor]],
    expert_prefixes: set[str],
    *,
    tp_size: int,
    num_experts: int,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Pair checkpoint weights/scales and retain their FP8 representation.

    Other model tensors pass through unchanged. Check completeness per expert,
    not just per fused runtime parameter, before accepting the checkpoint.
    """
    if not expert_prefixes:
        yield from weights
        return
    pending: dict[str, dict[str, torch.Tensor]] = {}
    seen: set[str] = set()
    projections = {"gate_proj", "up_proj", "down_proj"}
    expected = {
        f"{prefix}.{expert}.{projection}.{suffix}"
        for prefix in expert_prefixes
        for expert in range(num_experts)
        for projection in projections
        for suffix in ("weight", "weight_scale_inv")
    }
    for name, tensor in weights:
        if name not in expected:
            if any(name.startswith(prefix + ".") for prefix in expert_prefixes):
                raise ValueError(f"Unexpected MTP FP8 expert tensor: {name}")
            yield name, tensor
            continue
        if name in seen:
            raise ValueError(f"Duplicate MTP FP8 checkpoint tensor: {name}")
        seen.add(name)
        base, suffix = name.rsplit(".", 1)
        pair = pending.setdefault(base, {})
        pair[suffix] = tensor
        if len(pair) == 2:
            weight, scale = pad_mtp_fp8_checkpoint_matrix(
                pair["weight"],
                pair["weight_scale_inv"],
                down_projection=base.endswith(".down_proj"),
                tp_size=tp_size,
            )
            del pending[base]
            yield base + ".weight", weight
            yield base + ".weight_scale_inv", scale
    missing = expected - seen
    if missing:
        raise ValueError(
            "Missing MTP FP8 checkpoint tensors: " + ", ".join(sorted(missing)[:5])
        )
