# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read selected packed GGUF rows without expanding an embedding table."""

import numpy as np

from vllm.transformers_utils.gguf_tensor_reader import dequantize, quant_size


class PackedGGUFRowReader:
    def __init__(
        self,
        data: np.ndarray,
        source_type: int,
        hidden_size: int,
        logical_rows: int | None = None,
    ):
        block, size = quant_size(source_type)
        if data.dtype != np.uint8 or data.ndim != 2:
            raise ValueError("GGUF row storage must be packed uint8 [rows,bytes]")
        if hidden_size <= 0 or hidden_size % block:
            raise ValueError("GGUF row width cuts a source quantization block")
        if data.shape[1] != hidden_size // block * size:
            raise ValueError("GGUF packed row width disagrees with logical width")
        rows = data.shape[0] if logical_rows is None else logical_rows
        if not 0 < rows <= data.shape[0]:
            raise ValueError("Invalid GGUF logical row count")
        # Keep the reader/mmap owner's view. Only requested rows are copied.
        self.data = data
        self.source_type = int(source_type)
        self.hidden_size = hidden_size
        self.logical_rows = rows

    def lookup(self, ids: np.ndarray, dtype=np.float16) -> np.ndarray:
        ids = np.asarray(ids)
        if not np.issubdtype(ids.dtype, np.integer):
            raise TypeError("GGUF row IDs must be integers")
        dtype = np.dtype(dtype)
        if dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
            raise ValueError("GGUF row output requires FP16 or FP32")
        output_shape = (*ids.shape, self.hidden_size)
        if not ids.size:
            return np.empty(output_shape, dtype=dtype)
        if ids.min() < 0 or ids.max() >= self.logical_rows:
            raise IndexError("GGUF row ID outside the logical vocabulary")
        unique, inverse = np.unique(ids.reshape(-1), return_inverse=True)
        values = dequantize(self.data[unique], self.source_type)
        if not np.isfinite(values).all():
            raise ValueError("GGUF selected rows contain nonfinite values")
        if dtype == np.float16 and np.any(np.abs(values) > np.finfo(dtype).max):
            raise ValueError("GGUF selected rows overflow FP16 output")
        return values.astype(dtype)[inverse].reshape(output_shape)
