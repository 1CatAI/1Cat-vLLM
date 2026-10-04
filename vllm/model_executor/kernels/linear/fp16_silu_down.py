# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP16 SiLU/multiply and row projection with FP32 accumulation."""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, tldevice, triton


@triton.jit
def _silu_down(
    gate_up, weight, out, K: tl.constexpr, N: tl.constexpr, ROWS: tl.constexpr
):
    tiles: tl.constexpr = triton.cdiv(N, ROWS)
    token = tl.program_id(0) // tiles
    row = (tl.program_id(0) % tiles) * ROWS + tl.arange(0, ROWS)
    lanes = tl.arange(0, 8)
    vectors = tl.arange(0, 2)
    columns = lanes[:, None] * 2 + vectors[None, :]
    acc = tl.zeros((ROWS, 8, 2), tl.float32)
    for start in tl.static_range(0, K, 16):
        col = start + columns
        gate = tl.load(gate_up + token * 2 * K + col, col < K, 0).to(tl.float32)
        up = tl.load(gate_up + token * 2 * K + K + col, col < K, 0).to(tl.float32)
        activated = (
            tl.div_rn(gate, 1.0 + tldevice.exp(-gate)).to(tl.float16).to(tl.float32)
        )
        value = (activated * up).to(tl.float16).to(tl.float32)
        w = tl.load(
            weight + row[:, None, None] * K + col[None, :, :],
            (row[:, None, None] < N) & (col[None, :, :] < K),
            0,
        ).to(tl.float32)
        acc = tl.fma(value[None, :, :], w, acc)
    projected = tl.sum(tl.sum(acc, 2), 1)
    tl.store(out + token * N + row, projected, row < N)


def apply(gate_up: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    m, twice_k = gate_up.shape
    n, k = weight.shape
    if not (
        m > 0
        and twice_k == 2 * k
        and k > 0
        and n > 0
        and gate_up.dtype == weight.dtype == torch.float16
        and gate_up.is_contiguous()
        and weight.is_contiguous()
        and gate_up.is_cuda
        and weight.device == gate_up.device
        and current_platform.is_device_capability(70)
    ):
        raise ValueError("Unsupported SM70 FP16 SiLU/down tensors")
    rows = 16
    out = gate_up.new_empty((m, n))
    _silu_down[(m * triton.cdiv(n, rows),)](
        gate_up, weight, out, K=k, N=n, ROWS=rows, num_warps=4
    )
    return out
