# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GGUF nonlinear nibbles to canonical TurboMind LUT4 groups.

Layouts and formulas follow gguf-py/llama.cpp. Integer indices are preserved;
the two tables are IQ4 nonlinear values and standard E2M1 values.
"""

from dataclasses import dataclass, replace

import gguf
import numpy as np

from vllm.model_executor.layers.quantization.gguf_transcode import _fp16_coefficients
from vllm.transformers_utils.gguf_tensor_reader import quant_size

LUT4_TYPES = frozenset((20, 23, 39, 40))
LUT4_IQ = 0
LUT4_E2M1 = 1
LUT4_TABLES = (
    np.array(gguf.quants.IQ4_NL.kvalues, dtype=np.float32),
    np.array(gguf.quants.MXFP4.kvalues, dtype=np.float32) * 0.5,
)


@dataclass(frozen=True)
class Lut4GGUFProjection:
    source_type: int
    group_size: int
    lut_id: int
    codes: np.ndarray  # [N,K], preserved unsigned nibble indices
    scales: np.ndarray  # [N,K/group], FP16

    @property
    def bits(self) -> int:
        return 4

    def dequantize(self) -> np.ndarray:
        values = LUT4_TABLES[self.lut_id][self.codes].reshape(
            *self.scales.shape, self.group_size
        )
        return (values * self.scales.astype(np.float32)[..., None]).reshape(
            self.codes.shape
        )

    def tp_slice(self, rank: int, size: int, *, axis: int):
        if not 0 <= rank < size or axis not in (0, 1):
            raise ValueError("Invalid GGUF TP rank, size or axis")
        span, remainder = divmod(self.codes.shape[axis], size)
        if remainder or (axis == 1 and span % self.group_size):
            raise ValueError("GGUF canonical TP boundary cuts a LUT4 group")
        selection = [slice(None), slice(None)]
        selection[axis] = slice(rank * span, (rank + 1) * span)
        codes = np.ascontiguousarray(self.codes[tuple(selection)])
        if axis == 1:
            selection[axis] = slice(
                rank * span // self.group_size, (rank + 1) * span // self.group_size
            )
        return replace(
            self,
            codes=codes,
            scales=np.ascontiguousarray(self.scales[tuple(selection)]),
        )


def transcode_lut4(data: np.ndarray, weight_type: int) -> Lut4GGUFProjection:
    """Normalize block layouts and expand nested scales, without requantizing."""
    if weight_type not in LUT4_TYPES:
        raise ValueError(f"GGUF type {weight_type} has no LUT4 codec")
    block, size = quant_size(weight_type)
    if data.dtype != np.uint8 or data.ndim != 2 or data.shape[1] % size:
        raise ValueError("GGUF projection needs complete packed rows [N,bytes]")
    rows, width = data.shape
    k = width // size * block
    blocks = np.ascontiguousarray(data).reshape(-1, size)
    count = blocks.shape[0]
    group = 16 if weight_type == 40 else 32
    lut_id = LUT4_IQ if weight_type in (20, 23) else LUT4_E2M1
    if weight_type == 20:
        scales = blocks[:, :2].copy().view("<f2").astype(np.float32)
        payload = blocks[:, 2:]
    elif weight_type == 23:
        d = blocks[:, :2].copy().view("<f2").astype(np.float32)
        hi = blocks[:, 2:4].copy().view("<u2")
        hi = (hi >> (2 * np.arange(8, dtype=np.uint16))) & 3
        lo = blocks[:, 4:8].reshape(count, 4, 1)
        lo = (lo >> np.array([0, 4], np.uint8)[None, None, :]) & 15
        scale_codes = lo.reshape(count, 8) | (hi.astype(np.uint8) << 4)
        scales = d * (scale_codes.astype(np.int16) - 32).astype(np.float32)
        payload = blocks[:, 8:]
    elif weight_type == 39:
        # gguf-py's doubled-integer convention uses E8M0 / 2. Use the
        # standard E2M1 table with the full E8M0 scale instead.
        with np.errstate(over="ignore"):
            scales = gguf.quants.MXFP4.e8m0_to_fp32_half(blocks[:, :1]) * 2
        payload = blocks[:, 1:]
    else:
        scales = gguf.quants.NVFP4.ue4m3_to_fp32(blocks[:, :4]) * 2
        payload = blocks[:, 4:]
    lanes = payload.reshape(count, -1, 1, group // 2)
    codes = (lanes >> np.array([0, 4], np.uint8)[None, None, :, None]) & 15
    scales = scales.reshape(rows, k // group)
    if not np.isfinite(scales).all():
        raise ValueError("GGUF LUT4 scale overflows finite canonical coefficients")
    converted = _fp16_coefficients(scales, "LUT4 scale")
    if np.any((scales != 0) & (converted == 0)):
        raise ValueError("GGUF LUT4 scale underflows canonical FP16 coefficients")
    return Lut4GGUFProjection(
        weight_type,
        group,
        lut_id,
        np.ascontiguousarray(codes.reshape(rows, k)),
        converted,
    )
