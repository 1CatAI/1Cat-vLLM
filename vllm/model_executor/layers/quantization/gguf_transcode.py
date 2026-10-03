# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GGUF blocks to canonical TurboMind integer codes and affine coefficients.

Block layouts follow gguf-py/llama.cpp. This module changes weight storage;
activation precision and the TurboMind MMA arithmetic are unchanged.
"""

from dataclasses import dataclass

import gguf
import numpy as np

from vllm.transformers_utils.gguf_tensor_reader import quant_size


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

AFFINE_U2_TYPES = frozenset((10, 34, 35, 41, 42))
AFFINE_BITPLANE_TYPES = frozenset((6, 7, 11, 13, 14))


def transcode_affine(data: np.ndarray, weight_type: int) -> AffineGGUFProjection:
    """Normalize affine and ternary blocks without quantizing code values.

    Small source blocks expand to group32 so TP4 K=160 does not cut Q2_0's
    original 64-value blocks. Q2_K retains its group16 subblock boundaries.
    Ternary formats use the same unsigned two-bit decoder and additive bias.
    """
    if weight_type in AFFINE_GROUP32_TYPES:
        return transcode_affine_group32(data, weight_type)
    if weight_type in AFFINE_BITPLANE_TYPES:
        return transcode_affine_bitplanes(data, weight_type)
    if weight_type not in AFFINE_U2_TYPES:
        raise ValueError(f"GGUF type {weight_type} has no affine codec")
    block, size = quant_size(weight_type)
    if data.dtype != np.uint8 or data.ndim != 2 or data.shape[1] % size:
        raise ValueError("GGUF projection needs complete packed rows [N,bytes]")
    rows, width = data.shape
    k = width // size * block
    blocks = np.ascontiguousarray(data).reshape(-1, size)
    if weight_type == 10:  # Q2_K: affine metadata for each 16-value subblock.
        d = blocks[:, 80:82].copy().view("<f2").astype(np.float32)
        dmin = blocks[:, 82:84].copy().view("<f2").astype(np.float32)
        scales = d * (blocks[:, :16] & 15).astype(np.float32)
        mins = -dmin * (blocks[:, :16] >> 4).astype(np.float32)
        shifts = np.arange(0, 8, 2, dtype=np.uint8).reshape(1, 1, 4, 1)
        codes = (blocks[:, 16:80].reshape(-1, 2, 1, 32) >> shifts) & 3
        group = 16
    else:
        group = 32
        if weight_type in (41, 42):
            d = blocks[:, :2].copy().view("<f2").astype(np.float32)
            shifts = np.arange(0, 8, 1 if weight_type == 41 else 2, dtype=np.uint8)
            codes = (blocks[:, 2:, None] >> shifts) & (1 if weight_type == 41 else 3)
            if weight_type == 41:
                # Codes 0/2 avoid doubling the FP16 scale at its range limit.
                codes = codes * 2
        else:
            d = blocks[:, -2:].copy().view("<f2").astype(np.float32)
            if weight_type == 35:
                shifts = np.arange(0, 8, 2, dtype=np.uint8).reshape(1, 1, 4, 1)
                codes = (blocks[:, :64].reshape(-1, 2, 1, 32) >> shifts) & 3
            else:
                # TQ1's base-3 lanes wrap in uint8, followed by floor(3*x/256).
                # This is an integer storage conversion, including zero scales.
                powers = np.array([1, 3, 9, 27, 81], dtype=np.uint8)
                sections = []
                for payload, lane_width, factors in (
                    (blocks[:, :32], 32, powers),
                    (blocks[:, 32:48], 16, powers),
                    (blocks[:, 48:52], 4, powers[:4]),
                ):
                    lanes = payload.reshape(-1, 1, lane_width) * factors[None, :, None]
                    sections.append(lanes.reshape(-1, lane_width * len(factors)))
                wrapped = np.concatenate(sections, axis=1)
                codes = ((wrapped.astype(np.uint16) * 3) >> 8).astype(np.uint8)
        scales = np.repeat(d, block // group, axis=1)
        mins = -scales
    return AffineGGUFProjection(
        weight_type,
        2,
        group,
        np.ascontiguousarray(codes.reshape(rows, k)),
        _fp16_coefficients(scales.reshape(rows, k // group), "scale"),
        _fp16_coefficients(mins.reshape(rows, k // group), "min"),
    )


def transcode_affine_bitplanes(
    data: np.ndarray, weight_type: int
) -> AffineGGUFProjection:
    """Preserve 3/5/6-bit integer codes and source affine group boundaries.

    The storage decoder combines a U2/U4 low plane with a one/two-bit high
    plane. These formulas follow gguf-py's official reconstruction; no code
    values are rounded or requantized.
    """
    if weight_type not in AFFINE_BITPLANE_TYPES:
        raise ValueError(f"GGUF type {weight_type} has no bit-plane codec")
    block, size = quant_size(weight_type)
    if data.dtype != np.uint8 or data.ndim != 2 or data.shape[1] % size:
        raise ValueError("GGUF projection needs complete packed rows [N,bytes]")
    rows, width = data.shape
    k = width // size * block
    blocks = np.ascontiguousarray(data).reshape(-1, size)
    count = blocks.shape[0]
    if weight_type in (6, 7):
        bits, group = 5, 32
        d = blocks[:, :2].copy().view("<f2").astype(np.float32)
        start = 4 if weight_type == 7 else 2
        high = blocks[:, start : start + 4].copy().view("<u4")
        high = (high >> np.arange(32, dtype=np.uint32)) & 1
        low = blocks[:, start + 4 :].reshape(-1, 1, 16)
        low = (low >> np.array([0, 4], np.uint8)[None, :, None]) & 15
        codes = low.reshape(count, 32) | (high.astype(np.uint8) << 4)
        scales = d
        mins = (
            blocks[:, 2:4].copy().view("<f2").astype(np.float32)
            if weight_type == 7
            else -16 * d
        )
    elif weight_type == 13:
        bits, group = 5, 32
        d = blocks[:, :2].copy().view("<f2").astype(np.float32)
        dmin = blocks[:, 2:4].copy().view("<f2").astype(np.float32)
        scale_codes, min_codes = gguf.quants.Q4_K.get_scale_min(blocks[:, 4:16])
        scales, mins = (
            d * scale_codes.astype(np.float32),
            -dmin * min_codes.astype(np.float32),
        )
        low = blocks[:, 48:].reshape(count, 4, 1, 32)
        low = (low >> np.array([0, 4], np.uint8)[None, None, :, None]) & 15
        high = blocks[:, 16:48].reshape(count, 1, 32)
        high = (high >> np.arange(8, dtype=np.uint8)[None, :, None]) & 1
        codes = low.reshape(count, 8, 32) | (high << 4)
    elif weight_type == 14:
        bits, group = 6, 16
        d = blocks[:, -2:].copy().view("<f2").astype(np.float32)
        scales = d * blocks[:, 192:208].view(np.int8).astype(np.float32)
        mins = -32 * scales
        low = blocks[:, :128].reshape(count, 2, 1, 64)
        low = (low >> np.array([0, 4], np.uint8)[None, None, :, None]) & 15
        high = blocks[:, 128:192].reshape(count, 2, 1, 32)
        high = (high >> np.array([0, 2, 4, 6], np.uint8)[None, None, :, None]) & 3
        codes = low.reshape(count, 8, 32) | (high.reshape(count, 8, 32) << 4)
    else:
        bits, group = 3, 16
        d = blocks[:, -2:].copy().view("<f2").astype(np.float32)
        packed_scales = blocks[:, 96:108]
        low_scales = packed_scales[:, :8, None].transpose(0, 2, 1)
        low_scales = low_scales >> np.array([0, 4], np.uint8)[None, :, None]
        high_scales = packed_scales[:, 8:, None].transpose(0, 2, 1)
        high_scales = high_scales >> np.array([0, 2, 4, 6], np.uint8)[None, :, None]
        scale_codes = (low_scales.reshape(count, 16) & 15) | (
            (high_scales.reshape(count, 16) & 3) << 4
        )
        scales = d * (scale_codes.astype(np.int16) - 32).astype(np.float32)
        mins = -4 * scales
        low = blocks[:, 32:96].reshape(count, 2, 1, 32)
        low = (low >> np.array([0, 2, 4, 6], np.uint8)[None, None, :, None]) & 3
        high = blocks[:, :32].reshape(count, 1, 32)
        high = (high >> np.arange(8, dtype=np.uint8)[None, :, None]) & 1
        codes = low.reshape(count, 8, 32) | (high << 2)
    return AffineGGUFProjection(
        weight_type,
        bits,
        group,
        np.ascontiguousarray(codes.reshape(rows, k)),
        _fp16_coefficients(scales.reshape(rows, k // group), "scale"),
        _fp16_coefficients(mins.reshape(rows, k // group), "min"),
    )


def pack_affine_high_plane(projection: AffineGGUFProjection) -> np.ndarray:
    """Little-endian high-code bits per canonical group, in 32-bit words."""
    if projection.source_type not in AFFINE_BITPLANE_TYPES:
        raise ValueError("Projection does not use the bit-plane decoder")
    low_bits = 2 if projection.bits == 3 else 4
    high_bits = projection.bits - low_bits
    high = (
        (projection.codes >> low_bits)
        .reshape(*projection.scales.shape, projection.group_size)
        .astype(np.uint64)
    )
    shifts = np.arange(projection.group_size, dtype=np.uint64) * high_bits
    return np.ascontiguousarray((high << shifts).sum(-1).astype(np.uint32))


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
