# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict, architecture-independent GGUF tensor boundary checks."""

from collections.abc import Sequence

import gguf
import numpy as np
import torch

from vllm.transformers_utils.gguf_tensor_reader import dequantize


def require_tensor_shape(tensor, expected: Sequence[int], name: str) -> None:
    """Compare a GGUF tensor with its expected logical (PyTorch-order) shape.

    GGUF's dimension directory uses the reverse order. This check changes no
    payload bytes and does not infer or perform a matrix transpose.
    """
    actual = tuple(int(size) for size in tensor.shape[::-1])
    expected = tuple(expected)
    if actual != expected or any(size <= 0 for size in actual):
        raise ValueError(f"GGUF {name}: expected shape {expected}, got {actual}")


def decode_dense_tensor(tensor, dtype: torch.dtype, name: str) -> torch.Tensor:
    """Decode a dense destination without silently losing range or layout.

    The caller chooses the destination precision, including FP32 parameters;
    there is no intermediate cast through the model's activation dtype.
    Model-specific reshapes and numerical inversions belong in the adapter.
    """
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"Unsupported GGUF dense destination dtype: {dtype}")
    if tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
        raw = torch.from_numpy(tensor.data.view(np.uint16).copy())
        decoded = raw.view(torch.bfloat16)
    elif tensor.tensor_type in (
        gguf.GGMLQuantizationType.F16,
        gguf.GGMLQuantizationType.F32,
    ):
        decoded = torch.from_numpy(tensor.data.copy())
    else:
        decoded = torch.from_numpy(dequantize(tensor.data, tensor.tensor_type))
    expected = tuple(int(size) for size in tensor.shape[::-1])
    if tuple(decoded.shape) != expected:
        raise ValueError(
            f"GGUF {name}: decoded shape {tuple(decoded.shape)} differs from "
            f"directory shape {expected}"
        )
    if not torch.isfinite(decoded).all():
        raise ValueError(f"GGUF {name}: non-finite weight values")
    converted = decoded.to(dtype)
    if not torch.isfinite(converted).all():
        raise ValueError(f"GGUF {name}: values overflow destination {dtype}")
    return converted
