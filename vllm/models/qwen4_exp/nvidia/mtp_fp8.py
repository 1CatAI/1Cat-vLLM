# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load checkpoint block-FP8 MTP experts into the SM70 FP16 draft path."""

from collections.abc import Iterable, Iterator

import torch


def dequantize_mtp_fp8_weight(
    weight: torch.Tensor, scale: torch.Tensor, block_size: int
) -> torch.Tensor:
    """Apply checkpoint scales before TP sharding, without requantization."""
    if weight.dtype != torch.float8_e4m3fn or weight.ndim != 2:
        raise ValueError("MTP FP8 expert weights must be 2D float8_e4m3fn tensors")
    if block_size != 128:
        raise ValueError("SM70 MTP FP8 loading requires 128x128 checkpoint blocks")
    expected = tuple((dim + block_size - 1) // block_size for dim in weight.shape)
    if tuple(scale.shape) != expected or not scale.is_floating_point():
        raise ValueError(
            f"MTP FP8 scale shape/dtype mismatch: expected {expected} floating "
            f"scales, got {tuple(scale.shape)} {scale.dtype}"
        )
    expanded = scale.to(device=weight.device, dtype=torch.float32)
    expanded = expanded.repeat_interleave(block_size, 0).repeat_interleave(
        block_size, 1
    )
    expanded = expanded[: weight.shape[0], : weight.shape[1]]
    return (weight.float() * expanded).to(torch.float16)


def dequantize_mtp_fp8_experts(
    weights: Iterable[tuple[str, torch.Tensor]], expert_blocks: dict[str, int]
) -> Iterator[tuple[str, torch.Tensor]]:
    """Pair per-expert weights/scales in either order; yield only FP16 weights.

    Names have already been mapped into the standalone draft model. Other
    weights pass through unchanged. No checkpoint file is modified and no
    conversion is installed on the inference path.
    """
    if not expert_blocks:
        yield from weights
        return
    pending: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in weights:
        parts = name.rsplit(".", 3)
        if (
            len(parts) != 4
            or parts[0] not in expert_blocks
            or not parts[1].isdigit()
            or parts[2] not in ("gate_proj", "up_proj", "down_proj")
            or parts[3] not in ("weight", "weight_scale_inv")
        ):
            yield name, tensor
            continue
        base = name.rsplit(".", 1)[0]
        pair = pending.setdefault(base, {})
        if parts[3] in pair:
            raise ValueError(f"Duplicate MTP FP8 checkpoint tensor: {name}")
        pair[parts[3]] = tensor
        if len(pair) == 2:
            del pending[base]
            yield (
                base + ".weight",
                dequantize_mtp_fp8_weight(
                    pair["weight"], pair["weight_scale_inv"], expert_blocks[parts[0]]
                ),
            )
    if pending:
        missing = ", ".join(sorted(pending)[:5])
        raise ValueError(f"Unpaired MTP FP8 expert weights/scales: {missing}")
