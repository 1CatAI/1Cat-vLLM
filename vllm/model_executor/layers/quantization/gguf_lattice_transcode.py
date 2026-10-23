# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GGUF lattice indices, sign masks and canonical FP16 scale groups.

Source layouts and tables come from gguf-py/llama.cpp (MIT). Codebook indices
and signs are retained rather than quantizing reconstructed weight values.
"""

from dataclasses import dataclass, replace

import gguf
import numpy as np

from vllm.model_executor.layers.quantization.gguf_transcode import _fp16_coefficients
from vllm.transformers_utils.gguf_tensor_reader import quant_size

LATTICE_TYPES = frozenset((16, 17, 18, 19, 21, 22, 29))


def lattice_grid(weight_type: int) -> np.ndarray:
    name = gguf.GGMLQuantizationType(weight_type).name
    codec = getattr(gguf.quants, name)
    codec.init_grid()
    return codec.grid.reshape(codec.grid_shape).astype(np.float32)


@dataclass(frozen=True)
class LatticeGGUFProjection:
    source_type: int
    group_size: int
    grid_width: int
    indices: np.ndarray  # [N,K/grid_width], preserved uint16 indices
    signs: np.ndarray  # [N,K/8], eight sign bits per byte
    deltas: np.ndarray  # [N,K/8], IQ1 delta polarity; ignored for IQ2/IQ3
    scales: np.ndarray  # [N,K/group], FP16 expanded coefficients

    @property
    def shape(self):
        return self.indices.shape[0], self.indices.shape[1] * self.grid_width

    @property
    def bits(self):
        return 2  # Physical operand width; signs/indices also occupy metadata.

    @property
    def codes(self):
        return self.mma884_storage()[0]

    def dequantize(self) -> np.ndarray:
        values = lattice_grid(self.source_type)[self.indices].reshape(self.shape)
        if self.source_type in (19, 29):
            delta = (1 - 2 * self.deltas.astype(np.int8)) * np.float32(0.125)
            values = values + np.repeat(delta, 8, axis=1)
        else:
            shifts = np.arange(8, dtype=np.uint8)
            signs = ((self.signs[..., None] >> shifts) & 1).astype(np.int8)
            values = values * (1 - 2 * signs).reshape(self.shape)
        return (
            values.reshape(*self.scales.shape, self.group_size)
            * self.scales.astype(np.float32)[..., None]
        ).reshape(self.shape)

    def mma884_storage(self) -> tuple[np.ndarray, np.ndarray]:
        """U2 operand stream and scale/sign/index metadata, without requantizing.

        Eight logical U2 codes become one 16-bit MMA packet. IQ2 places the
        low index byte and sign mask there; IQ3 places two low index bytes.
        IQ1 stores its full eleven-bit index and delta polarity. Metadata
        retains high index bits and signs alongside one FP16 group scale.
        """
        n, k = self.shape
        indices = self.indices.astype(np.uint64)
        scales = self.scales.view(np.uint16).astype(np.uint64)
        if self.source_type in (19, 29):
            packets = indices | (self.deltas.astype(np.uint64) << 15)
            metadata = scales.astype(np.uint16)
        elif self.grid_width == 8:
            packets = (indices & 255) | (self.signs.astype(np.uint64) << 8)
            high = (indices >> 8).reshape(n, k // self.group_size, -1)
            shifts = 16 + 2 * np.arange(high.shape[-1], dtype=np.uint64)
            metadata = (scales | np.bitwise_or.reduce(high << shifts, axis=-1)).astype(
                np.uint32
            )
        else:
            pairs = indices.reshape(n, k // 8, 2)
            packets = (pairs[..., 0] & 255) | ((pairs[..., 1] & 255) << 8)
            sign_groups = self.signs.astype(np.uint64).reshape(
                n, k // self.group_size, -1
            )
            sign_shifts = 16 + 8 * np.arange(sign_groups.shape[-1], dtype=np.uint64)
            high = (indices >> 8).reshape(n, k // self.group_size, -1)
            high_shifts = 48 + np.arange(high.shape[-1], dtype=np.uint64)
            metadata = scales | np.bitwise_or.reduce(
                sign_groups << sign_shifts, axis=-1
            )
            metadata |= np.bitwise_or.reduce(high << high_shifts, axis=-1)
        # Invert Converter<uint16_t,uint2_t>'s adjacent-half pairing so it
        # preserves the packet verbatim after operand packing.
        # Every carrier is exactly 16 bits; widening the eight-way expansion
        # to uint64 needlessly quadruples its temporary memory traffic.
        shifts = np.array([0, 8, 2, 10, 4, 12, 6, 14], dtype=np.uint16)
        packets = packets.astype(np.uint16)
        codes = ((packets[..., None] >> shifts) & 3).astype(np.uint8).reshape(n, k)
        return np.ascontiguousarray(codes), np.ascontiguousarray(metadata)

    def tp_slice(self, rank: int, size: int, *, axis: int):
        if not 0 <= rank < size or axis not in (0, 1):
            raise ValueError("Invalid GGUF TP rank, size or axis")
        span, remainder = divmod(self.shape[axis], size)
        if remainder or (axis == 1 and span % self.group_size):
            raise ValueError("GGUF canonical TP boundary cuts a lattice group")
        updates = {}
        for name, stride in (
            ("indices", self.grid_width),
            ("signs", 8),
            ("deltas", 8),
            ("scales", self.group_size),
        ):
            selection = [slice(None), slice(None)]
            width = span if axis == 0 else span // stride
            selection[axis] = slice(rank * width, (rank + 1) * width)
            updates[name] = np.ascontiguousarray(getattr(self, name)[tuple(selection)])
        return replace(self, **updates)


def _nibbles(data):
    return ((data[..., None] >> np.array([0, 4], np.uint8)) & 15).reshape(
        data.shape[0], -1
    )


def _parity_signs(indices):
    table = np.frombuffer(gguf.quants.IQ2_XXS.ksigns, dtype=np.uint8)
    return table[indices]


def transcode_lattice(data: np.ndarray, weight_type: int) -> LatticeGGUFProjection:
    if weight_type not in LATTICE_TYPES:
        raise ValueError(f"GGUF type {weight_type} has no lattice codec")
    block, size = quant_size(weight_type)
    if data.dtype != np.uint8 or data.ndim != 2 or data.shape[1] % size:
        raise ValueError("GGUF projection needs complete packed rows [N,bytes]")
    rows, width = data.shape
    k = width // size * block
    blocks = np.ascontiguousarray(data).reshape(-1, size)
    count = blocks.shape[0]
    group = 16 if weight_type in (17, 22, 29) else 32
    grid_width = 4 if weight_type in (18, 21) else 8
    signs = np.zeros((count, 32), np.uint8)
    deltas = np.zeros_like(signs)
    if weight_type == 29:
        words = blocks[:, 48:].copy().view("<u2")
        bits = (words & 0xF000) >> np.array([12, 8, 4, 0], np.uint16)
        d = (
            np.bitwise_or.reduce(bits, axis=1)
            .copy()
            .view("<f2")
            .astype(np.float32)[:, None]
        )
        local = ((words[..., None] >> np.array([0, 3, 6, 9], np.uint16)) & 7).reshape(
            count, 16
        )
        scales = d * (2 * local + 1)
        high = (blocks[:, 32:48, None] >> np.array([0, 4], np.uint8)).reshape(count, 32)
        indices = blocks[:, :32].astype(np.uint16) | ((high.astype(np.uint16) & 7) << 8)
        deltas = ((high >> 3) & 1).astype(np.uint8)
    else:
        d = blocks[:, :2].copy().view("<f2").astype(np.float32)
        if weight_type == 19:
            high = blocks[:, 34:].copy().view("<u2")
            scales = d * (2 * ((high >> 12) & 7) + 1)
            deltas = np.repeat(((high >> 15) & 1).astype(np.uint8), 4, axis=1)
            high_indices = (
                (high[..., None] >> np.array([0, 3, 6, 9], np.uint16)) & 7
            ).reshape(count, 32)
            indices = blocks[:, 2:34].astype(np.uint16) | (high_indices << 8)
        elif weight_type in (16, 18):
            if weight_type == 16:
                words = blocks[:, 2:].copy().view("<u4").reshape(count, 8, 2)
                indices = (
                    words[..., 0]
                    .copy()
                    .view(np.uint8)
                    .reshape(count, 32)
                    .astype(np.uint16)
                )
                meta = words[..., 1]
            else:
                indices = blocks[:, 2:66].astype(np.uint16)
                meta = blocks[:, 66:].copy().view("<u4")
            scales = d * (np.float32(0.5) + (meta >> 28).astype(np.float32))
            scales *= np.float32(0.25 if weight_type == 16 else 0.5)
            sign_indices = (
                (meta[..., None] >> np.array([0, 7, 14, 21], np.uint32)) & 127
            ).reshape(count, 32)
            signs = _parity_signs(sign_indices)
        elif weight_type == 17:
            codes = blocks[:, 2:66].copy().view("<u2")
            indices = codes & 511
            signs = _parity_signs(codes >> 9)
            scales = d * (np.float32(0.5) + _nibbles(blocks[:, 66:])) * np.float32(0.25)
        elif weight_type == 22:
            high = (
                (blocks[:, 66:74, None] >> np.array([0, 2, 4, 6], np.uint8)) & 3
            ).reshape(count, 32)
            indices = blocks[:, 2:34].astype(np.uint16) | (high.astype(np.uint16) << 8)
            signs = blocks[:, 34:66]
            scales = d * (np.float32(0.5) + _nibbles(blocks[:, 74:])) * np.float32(0.25)
        else:
            high = (
                (blocks[:, 66:74, None] >> np.arange(8, dtype=np.uint8)) & 1
            ).reshape(count, 64)
            indices = blocks[:, 2:66].astype(np.uint16) | (high.astype(np.uint16) << 8)
            signs = blocks[:, 74:106]
            scales = d * (1 + 2 * _nibbles(blocks[:, 106:]))
    converted = _fp16_coefficients(scales.reshape(rows, k // group), "lattice scale")
    return LatticeGGUFProjection(
        weight_type,
        group,
        grid_width,
        np.ascontiguousarray(indices.reshape(rows, k // grid_width)),
        np.ascontiguousarray(signs.reshape(rows, k // 8)),
        np.ascontiguousarray(deltas.reshape(rows, k // 8)),
        converted,
    )
