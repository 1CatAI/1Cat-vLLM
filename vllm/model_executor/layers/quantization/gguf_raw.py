# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Original GGUF rows with at most seven alignment bytes, without transcoding."""

from dataclasses import dataclass

import numpy as np

from vllm.transformers_utils.gguf_tensor_reader import quant_size

RAW_LATTICE_TYPES = frozenset((18, 21, 22))


@dataclass(frozen=True)
class RawGGUFProjection:
    source_type: int
    data: np.ndarray
    logical_k: int
    payload_bytes_per_row: int

    @classmethod
    def from_rows(cls, data: np.ndarray, source_type: int):
        if source_type not in RAW_LATTICE_TYPES:
            raise ValueError("Raw lattice kernel does not support this GGUF type")
        block, size = quant_size(source_type)
        if data.dtype != np.uint8 or data.ndim != 2 or data.shape[1] % size:
            raise ValueError("Raw GGUF requires complete packed rows [N,bytes]")
        payload = data.shape[1]
        stride = (payload + 7) // 8 * 8
        if stride == payload and data.flags.c_contiguous:
            storage = data
        else:
            storage = np.zeros((data.shape[0], stride), dtype=np.uint8)
            storage[:, :payload] = data
        return cls(source_type, storage, payload // size * block, payload)

    @property
    def shape(self):
        return self.data.shape[0], self.logical_k

    @property
    def padding_bytes_per_row(self):
        return self.data.shape[1] - self.payload_bytes_per_row

    @property
    def storage_bytes(self):
        return self.data.nbytes

    def tp_slice(self, rank: int, size: int, *, axis: int):
        if not 0 <= rank < size or axis not in (0, 1):
            raise ValueError("Invalid raw GGUF TP slice")
        span, remainder = divmod(self.shape[axis], size)
        if remainder:
            raise ValueError("Raw GGUF TP dimension is not divisible")
        payload = self.data[:, : self.payload_bytes_per_row]
        if axis == 0:
            payload = payload[rank * span : (rank + 1) * span]
        else:
            block, block_bytes = quant_size(self.source_type)
            if span % block:
                raise ValueError("Raw GGUF TP boundary cuts a source block")
            row_bytes = span // block * block_bytes
            payload = payload[:, rank * row_bytes : (rank + 1) * row_bytes]
        return type(self).from_rows(payload, self.source_type)
