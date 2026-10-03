# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GGUF blocks to canonical TurboMind integer codes and affine coefficients.

Block layouts follow gguf-py/llama.cpp. This module changes weight storage;
activation precision and the TurboMind MMA arithmetic are unchanged.
"""

from dataclasses import dataclass

import gguf
import numpy as np


@dataclass(frozen=True)
class AffineGGUFProjection:
    source_type: int
    bits: int
    group_size: int
    codes: np.ndarray  # [N,K], integer codes before TurboMind layout packing
    scales: np.ndarray  # [N,K/group], FP16
    mins: np.ndarray  # [N,K/group], additive FP16 bias, not a zero-point ratio

    def dequantize(self) -> np.ndarray:
        grouped = self.codes.reshape(*self.scales.shape, self.group_size)
        return (
            grouped.astype(np.float32) * self.scales.astype(np.float32)[..., None]
            + self.mins.astype(np.float32)[..., None]
        ).reshape(self.codes.shape)

    def tp_slice(self, rank: int, size: int, *, axis: int):
        if not 0 <= rank < size or axis not in (0, 1):
            raise ValueError("Invalid GGUF TP rank, size or axis")
        span, remainder = divmod(self.codes.shape[axis], size)
        if remainder or (axis == 1 and span % self.group_size):
            raise ValueError("GGUF canonical TP boundary cuts an affine group")
        selection = slice(rank * span, (rank + 1) * span)
        codes = self.codes[selection] if axis == 0 else self.codes[:, selection]
        stats = (
            selection
            if axis == 0
            else slice(
                rank * span // self.group_size, (rank + 1) * span // self.group_size
            )
        )
        scales = self.scales[stats] if axis == 0 else self.scales[:, stats]
        mins = self.mins[stats] if axis == 0 else self.mins[:, stats]
        return AffineGGUFProjection(
            self.source_type,
            self.bits,
            self.group_size,
            np.ascontiguousarray(codes),
            np.ascontiguousarray(scales),
            np.ascontiguousarray(mins),
        )


AFFINE_GROUP32_TYPES = frozenset(
    (
        gguf.GGMLQuantizationType.Q4_0,
        gguf.GGMLQuantizationType.Q4_1,
        gguf.GGMLQuantizationType.Q8_0,
        gguf.GGMLQuantizationType.Q4_K,
    )
)


def _fp16_coefficients(value: np.ndarray, name: str) -> np.ndarray:
    with np.errstate(over="ignore"):
        result = value.astype(np.float16)
    if np.any(np.isfinite(value) & ~np.isfinite(result)):
        raise ValueError(f"GGUF {name} overflows canonical FP16 coefficients")
    return result


def transcode_affine_group32(
    data: np.ndarray, weight_type: int
) -> AffineGGUFProjection:
    """Keep exact integer codes; expand nested scales into group-32 scale/min.

    Q4_0/Q4_1/Q8_0 coefficients are exact when representable. Q4_K products
    may round when expanded to FP16. Report reconstruction error against the
    official dequantizer before selecting this representation.
    """
    if weight_type not in AFFINE_GROUP32_TYPES:
        raise ValueError(f"GGUF type {weight_type} has no group-32 affine codec")
    block, size = gguf.GGML_QUANT_SIZES[weight_type]
    if data.dtype != np.uint8 or data.ndim != 2 or data.shape[1] % size:
        raise ValueError("GGUF projection needs complete packed rows [N,bytes]")
    rows, width = data.shape
    k = width // size * block
    blocks = np.ascontiguousarray(data).reshape(-1, size)
    d = blocks[:, :2].copy().view("<f2").astype(np.float32)
    bits = 4
    if weight_type == gguf.GGMLQuantizationType.Q4_K:
        dmin = blocks[:, 2:4].copy().view("<f2").astype(np.float32)
        scales, mins = gguf.quants.Q4_K.get_scale_min(blocks[:, 4:16])
        scales = d * scales.astype(np.float32)
        mins = -dmin * mins.astype(np.float32)
        payload = blocks[:, 16:].reshape(-1, 4, 32)
        codes = np.stack((payload & 15, payload >> 4), axis=2).reshape(rows, k)
    elif weight_type == gguf.GGMLQuantizationType.Q8_0:
        bits = 8
        codes = (blocks[:, 2:].view(np.int8).astype(np.int16) + 128).astype(np.uint8)
        codes = codes.reshape(rows, k)
        scales, mins = d, -128 * d
    else:
        affine = weight_type == gguf.GGMLQuantizationType.Q4_1
        payload = blocks[:, 4 if affine else 2 :]
        codes = np.stack((payload & 15, payload >> 4), axis=1).reshape(rows, k)
        scales = d
        mins = (
            blocks[:, 2:4].copy().view("<f2").astype(np.float32) if affine else -8 * d
        )
    return AffineGGUFProjection(
        int(weight_type),
        bits,
        32,
        np.ascontiguousarray(codes),
        _fp16_coefficients(scales.reshape(rows, k // 32), "scale"),
        _fp16_coefficients(mins.reshape(rows, k // 32), "min"),
    )


def reconstruction_error(projection: AffineGGUFProjection, reference: np.ndarray):
    actual = projection.dequantize().astype(np.float64)
    reference = reference.astype(np.float64)
    if actual.shape != reference.shape:
        raise ValueError("GGUF reference and canonical shapes differ")
    difference = actual - reference
    reference_norm = np.linalg.norm(reference)
    return {
        "max_abs": float(np.max(np.abs(difference), initial=0)),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "relative_l2": float(np.linalg.norm(difference) / reference_norm)
        if reference_norm
        else float(np.linalg.norm(difference)),
    }
