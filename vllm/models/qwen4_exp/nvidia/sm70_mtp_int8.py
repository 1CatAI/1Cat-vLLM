# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark-only INT8 expert weights with FP16 operands and FP32 sums."""

import torch


@torch.inference_mode()
def prepare_int8_expert_weight(weight):
    """Pack row-scaled integers into the existing 32-by-16 MMA layout.

    Zero rows use scale one. Startup FP32 scratch is bounded to 4096 rows.
    This helper installs no serving dispatch and never replaces FP16 weights.
    """
    if weight.ndim != 2 or weight.dtype != torch.float16:
        raise ValueError("Expected a two-dimensional FP16 weight")
    rows, width = weight.shape
    if rows % 32 or width % 16:
        raise ValueError("Expert weight must align to 32 output and 16 input rows")
    integer = torch.empty_like(weight, dtype=torch.int8)
    scales = weight.new_empty(rows)
    for begin in range(0, rows, 4096):
        values = weight[begin : begin + 4096].float()
        scale = values.abs().amax(-1, keepdim=True) / 127
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        integer[begin : begin + 4096].copy_(
            (values / scale).round().clamp(-127, 127).to(torch.int8)
        )
        scales[begin : begin + 4096].copy_(scale.flatten())
    lane = torch.arange(32, device=weight.device)
    columns = ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) >> 2)
    physical = torch.tensor(
        [0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15],
        device=weight.device,
    )
    packed = integer.reshape(rows // 32, 32, width // 16, 16)
    packed = packed.index_select(1, columns).index_select(3, physical)
    return packed.permute(0, 2, 1, 3).contiguous().view(torch.uint8), scales
