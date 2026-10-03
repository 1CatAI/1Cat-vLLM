# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GGML storage compatibility for types newer than the Python gguf package.

Q2_0's enum/block contract follows ggml-org/llama.cpp
002a12ad25503a93501b2e188c360029830a241a. This does not modify gguf's global
enumeration or register a process-wide reader monkeypatch.
"""

from typing import cast

import gguf
import numpy as np

Q2_0 = 42


def quant_type_name(value: int) -> str:
    if value == Q2_0:
        return "Q2_0"
    return gguf.GGMLQuantizationType(value).name


def quant_size(value: int) -> tuple[int, int]:
    if value == Q2_0:
        return 64, 18
    return gguf.GGML_QUANT_SIZES[value]


def dequantize(data: np.ndarray, value: int) -> np.ndarray:
    if value != Q2_0:
        return gguf.quants.dequantize(data, gguf.GGMLQuantizationType(value))
    block, size = quant_size(value)
    if data.dtype != np.uint8 or data.shape[-1] % size:
        raise ValueError("Q2_0 rows must contain whole 18-byte blocks")
    blocks = data.reshape(-1, size)
    scales = blocks[:, :2].copy().view("<f2").astype(np.float32)
    codes = (blocks[:, 2:, None] >> np.arange(0, 8, 2, dtype=np.uint8)) & 3
    result = (codes.reshape(-1, block).astype(np.float32) - 1) * scales
    return result.reshape(*data.shape[:-1], data.shape[-1] // size * block)


class GGUFReader(gguf.GGUFReader):
    def _build_tensors(self, start_offs, fields):
        extra = [field for field in fields if int(field.parts[4][0]) == Q2_0]
        regular = [field for field in fields if int(field.parts[4][0]) != Q2_0]
        super()._build_tensors(start_offs, regular)
        names = {tensor.name for tensor in self.tensors}
        for field in extra:
            _, raw_name, _, dims, _, offset = field.parts
            name = bytes(raw_name).decode("utf-8")
            if name in names:
                raise ValueError(f"Duplicate GGUF tensor {name!r}")
            names.add(name)
            shape = tuple(reversed(dims.tolist()))
            block, size = quant_size(Q2_0)
            if not shape or shape[-1] <= 0 or shape[-1] % block:
                raise ValueError(
                    f"Q2_0 {name}: K must be a positive multiple of {block}"
                )
            count = int(np.prod(dims, dtype=np.int64))
            byte_count = count // block * size
            data_offset = int(start_offs + offset[0])
            if data_offset < start_offs or data_offset + byte_count > self.data.nbytes:
                raise ValueError(f"Truncated GGUF Q2_0 tensor {name!r}")
            packed_shape = (*shape[:-1], shape[-1] // block * size)
            self.tensors.append(
                gguf.ReaderTensor(
                    name=name,
                    tensor_type=cast(gguf.GGMLQuantizationType, Q2_0),
                    shape=dims,
                    n_elements=count,
                    n_bytes=byte_count,
                    data_offset=data_offset,
                    data=self._get(data_offset, np.uint8, byte_count).reshape(
                        packed_shape
                    ),
                    field=field,
                )
            )
