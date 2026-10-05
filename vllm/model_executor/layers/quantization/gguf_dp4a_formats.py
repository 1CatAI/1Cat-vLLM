# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integer-dot storage retaining original GGUF scale levels without rounding."""

from dataclasses import dataclass

import gguf
import numpy as np

from vllm.transformers_utils.gguf_tensor_reader import quant_size


@dataclass(frozen=True)
class GGUFIntegerDotProjection:
    source_type: int
    codes: np.ndarray
    d: np.ndarray
    small_scales: np.ndarray
    dmin: np.ndarray | None = None
    small_mins: np.ndarray | None = None

    @property
    def shape(self):
        return self.codes.shape

    @property
    def group_size(self):
        return 16 if self.source_type == 14 else 32

    def dequantize(self):
        n, k = self.shape
        scales = self.d.astype(np.float32).repeat(256 // self.group_size, axis=1)
        scales *= self.small_scales.astype(np.float32)
        if self.source_type == 23:
            values = np.asarray(gguf.quants.IQ4_NL.kvalues, dtype=np.float32)[
                self.codes
            ]
        else:
            values = self.codes.astype(np.float32)
        out = values * scales.repeat(self.group_size, axis=1)
        if self.dmin is not None:
            assert self.small_mins is not None
            mins = self.dmin.astype(np.float32).repeat(8, axis=1)
            mins *= self.small_mins.astype(np.float32)
            out -= mins.repeat(32, axis=1)
        return out.reshape(n, k)

    def packed(self, input_layout=None):
        """N32 packets: signed int8 Q6 values, or logical eight-nibble U4 words.

        Expanding Q6 codes to int8 avoids decode arithmetic. Original signed
        group16 scales and the superblock d remain separate, unlike expanded
        FP16 scale/min carriers. No floating weight or coefficient is rounded.
        """
        n, k = self.shape
        if n % 32 or k % 256:
            raise ValueError("Integer dot storage requires N32 and complete K256")
        codes = self.codes
        original_d = self.d.repeat(256 // self.group_size, axis=1)
        small_scales = self.small_scales
        original_min = None if self.dmin is None else self.dmin.repeat(8, axis=1)
        small_mins = self.small_mins
        if input_layout is not None:
            import torch

            span, remainder = divmod(input_layout.head_dim, self.group_size)
            if remainder:
                raise ValueError("Head layout cuts an integer-dot scale group")

            def restore(values, head_dim):
                return input_layout.weight_to_vllm(
                    torch.from_numpy(values), dim=1, head_dim=head_dim
                ).numpy()

            codes = restore(codes, input_layout.head_dim)
            original_d = restore(original_d, span)
            small_scales = restore(small_scales, span)
            if original_min is not None:
                assert small_mins is not None
                original_min = restore(original_min, span)
                small_mins = restore(small_mins, span)
        if self.source_type == 14:
            packets = codes.reshape(n, k // 4, 4).view("<i4").squeeze(-1)
        else:
            groups = codes.reshape(n, k // 8, 8).astype(np.uint32)
            packets = (
                (groups << (4 * np.arange(8, dtype=np.uint32))).sum(-1).astype("<i4")
            )
        packets = packets.reshape(n // 32, 32, -1).transpose(0, 2, 1)
        empty_d = np.empty((0,), dtype=np.float16)
        empty_s = np.empty((0,), dtype=np.int8)
        return (
            np.ascontiguousarray(packets),
            np.ascontiguousarray(original_d.T),
            np.ascontiguousarray(small_scales.T),
            empty_d if original_min is None else np.ascontiguousarray(original_min.T),
            empty_s if small_mins is None else np.ascontiguousarray(small_mins.T),
        )


def transcode_integer_dot(data: np.ndarray, source_type: int):
    if source_type not in (12, 14, 23):
        raise ValueError("No calibrated integer-dot codec for this GGUF type")
    _, size = quant_size(source_type)
    if (
        data.dtype != np.uint8
        or data.ndim != 2
        or not data.size
        or data.shape[1] % size
    ):
        raise ValueError("Integer-dot codec needs complete original packed rows")
    n, width = data.shape
    k = width // size * 256
    blocks = np.ascontiguousarray(data).reshape(-1, size)
    if source_type == 14:
        low = blocks[:, :128].reshape(-1, 2, 1, 64)
        low = (low >> np.array([0, 4], np.uint8)[None, None, :, None]) & 15
        high = blocks[:, 128:192].reshape(-1, 2, 1, 32)
        high = (high >> np.array([0, 2, 4, 6], np.uint8)[None, None, :, None]) & 3
        codes = low.reshape(-1, 8, 32) | (high.reshape(-1, 8, 32) << 4)
        codes = (codes.astype(np.int16) - 32).astype(np.int8).reshape(n, k)
        d = blocks[:, 208:210].copy().view("<f2").reshape(n, k // 256)
        scales = blocks[:, 192:208].copy().view(np.int8).reshape(n, k // 16)
        return GGUFIntegerDotProjection(source_type, codes, d, scales)
    d = blocks[:, :2].copy().view("<f2").reshape(n, k // 256)
    if source_type == 12:
        payload = blocks[:, 16:].reshape(-1, 4, 32)
        codes = np.stack((payload & 15, payload >> 4), axis=2).reshape(n, k)
        scales, mins = gguf.quants.Q4_K.get_scale_min(blocks[:, 4:16])
        dmin = blocks[:, 2:4].copy().view("<f2").reshape(n, k // 256)
        return GGUFIntegerDotProjection(
            source_type,
            codes,
            d,
            scales.astype(np.int8).reshape(n, k // 32),
            dmin,
            mins.astype(np.int8).reshape(n, k // 32),
        )
    payload = blocks[:, 8:].reshape(-1, 8, 16)
    codes = np.concatenate((payload & 15, payload >> 4), axis=-1).reshape(n, k)
    hi = blocks[:, 2:4].copy().view("<u2").astype(np.uint32)
    lo = blocks[:, 4:8].copy().view("<u4")
    groups = np.arange(8, dtype=np.uint32)
    scales = ((lo >> (4 * groups)) & 15) | (((hi >> (2 * groups)) & 3) << 4)
    scales = (scales.astype(np.int16) - 32).astype(np.int8).reshape(n, k // 32)
    return GGUFIntegerDotProjection(source_type, codes, d, scales)
