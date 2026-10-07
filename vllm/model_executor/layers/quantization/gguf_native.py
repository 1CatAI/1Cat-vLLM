# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packaged GGUF operators and Volta dispatch, with explicit capability admission."""

import importlib
from functools import lru_cache
from typing import Any

import torch

from vllm.platforms import current_platform
from vllm.transformers_utils.gguf_tensor_reader import quant_size, quant_type_name

NATIVE_TYPES = frozenset(
    (
        2,
        3,
        6,
        7,
        8,
        10,
        11,
        12,
        13,
        14,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        29,
        34,
        35,
        39,
        40,
        41,
        42,
    )
)


@lru_cache(maxsize=1)
def native_available() -> bool:
    try:
        importlib.import_module("vllm._C_gguf")
    except ImportError:
        return False
    return hasattr(torch.ops._C_gguf, "ggml_dense_upstream_capabilities")


def pad_weight_tail(
    weight: torch.Tensor, weight_type: int, *, storage_has_zero_tail: bool = False
) -> torch.Tensor:
    """Preserve logical rows, with the upstream MATRIX_ROW_PADDING storage tail.

    The storage contract follows vllm-gguf-plugin PR #141 (Apache-2.0); it pads
    the final allocation, not each logical row or the matrix K dimension.
    """
    if weight_type not in NATIVE_TYPES or weight.dtype != torch.uint8:
        return weight
    block, size = quant_size(weight_type)
    if weight.shape[-1] % size:
        raise ValueError("GGUF packed row ends inside a quantization block")
    logical_k = weight.shape[-1] // size * block
    missing = (-logical_k) % 512
    if missing % block:
        raise ValueError("GGUF upstream storage tail is not block aligned")
    tail_bytes = missing // block * size
    contiguous = weight.contiguous()
    if tail_bytes == 0:
        return contiguous
    if storage_has_zero_tail:
        available = contiguous.untyped_storage().nbytes() - contiguous.storage_offset()
        if contiguous is not weight or available < contiguous.numel() + tail_bytes:
            raise ValueError("GGUF guarded weight does not own its zero storage tail")
        return contiguous
    # Extra storage can belong to the next merged projection. Do not assume
    # it is a zero tail or overwrite it: NaN scales there can contaminate even
    # a zero-padded activation. Give this projection its own guarded allocation.
    storage = torch.zeros(
        contiguous.numel() + tail_bytes, dtype=torch.uint8, device=weight.device
    )
    logical = storage[: contiguous.numel()].view_as(contiguous)
    logical.copy_(contiguous)
    return logical


def empty_guarded_weight(shape, weight_type: int, device) -> torch.Tensor:
    """Allocate one packed bank and a zero safety tail before filling its rows."""
    block, size = quant_size(weight_type)
    if weight_type not in NATIVE_TYPES or shape[-1] % size:
        raise ValueError(
            "GGUF guarded bank requires complete native quantization blocks"
        )
    k = shape[-1] // size * block
    tail_bytes = ((-k) % 512) // block * size
    elements = 1
    for dimension in shape:
        elements *= dimension
    storage = torch.empty(elements + tail_bytes, dtype=torch.uint8, device=device)
    storage[elements:].zero_()
    return storage[:elements].view(shape)


def packed_tp_span(
    full_k: int, weight_type: int, tp_size: int, tp_rank: int
) -> tuple[int, int, int, int]:
    """Cover a TP slice with whole original blocks and zero activation margins.

    Return packed byte bounds, left activation padding and physical K. Boundary
    blocks may be shared between ranks; each rank uses only its logical inputs.
    """
    block, size = quant_size(weight_type)
    if full_k <= 0 or tp_size <= 0 or full_k % tp_size or full_k % block:
        raise ValueError("GGUF TP span requires complete source rows and equal slices")
    if not 0 <= tp_rank < tp_size:
        raise ValueError("GGUF TP rank is outside the group")
    local_k = full_k // tp_size
    start = tp_rank * local_k
    first = start // block
    last = (start + local_k + block - 1) // block
    return first * size, last * size, start - first * block, (last - first) * block


def native_dense(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_type: int,
    prefill_min_m: int = 8,
    max_dequant_bytes: int = 0,
) -> torch.Tensor | None:
    if weight_type not in NATIVE_TYPES or not native_available():
        return None
    if not x.is_cuda or not weight.is_cuda:
        return None
    x = x.contiguous()
    native = torch.ops._C_gguf
    capabilities = native.ggml_dense_upstream_capabilities(
        weight, x, weight_type, weight.shape[0]
    )
    # Volta has FP16 tensor cores and dp4a, but no INT8 tensor cores. Admit
    # cuBLAS before MMQ for prefill and concurrent batches.
    device_index = x.device.index
    if device_index is None:
        device_index = torch.accelerator.current_device_index()
    volta = current_platform.is_device_capability((7, 0), device_id=device_index)

    def blas():
        if max_dequant_bytes:
            block, size = quant_size(weight_type)
            k = weight.shape[-1] // size * block
            rows = max(1, max_dequant_bytes // (k * x.element_size()))
            if rows < weight.shape[0]:
                output = x.new_empty((x.shape[0], weight.shape[0]))
                for start in range(0, weight.shape[0], rows):
                    end = min(start + rows, weight.shape[0])
                    output[:, start:end].copy_(
                        native.ggml_dense_blas(
                            weight[start:end], x, weight_type, end - start
                        )
                    )
                return output
        return native.ggml_dense_blas(weight, x, weight_type, weight.shape[0])

    if (capabilities & 16) and volta and x.shape[0] >= prefill_min_m:
        return blas()
    if capabilities & 4:
        return native.ggml_dense_mmvq(weight, x, weight_type, weight.shape[0])
    if capabilities & 16:
        return blas()
    if (capabilities & 8) and not volta:
        return native.ggml_dense_mmq(weight, x, weight_type, weight.shape[0])
    return None


def dense_admission(weight, weight_type, dtype, prefill_min_m):
    """Inspect prepared storage at startup; this does not execute a GEMM."""
    result: dict[str, Any] = {"weight_type": quant_type_name(weight_type)}
    if weight_type in (0, 1, 30):
        result["route"] = "unquantized_torch_gemm"
        return result
    if weight_type not in NATIVE_TYPES:
        result["reason"] = "unsupported_native_format"
        return result
    if not weight.is_cuda or not native_available():
        result["reason"] = "native_extension_or_cuda_unavailable"
        return result
    block, size = quant_size(weight_type)
    k = weight.shape[-1] // size * block
    native = torch.ops._C_gguf
    result["local_shape"] = [weight.shape[0], k]
    result["capabilities"] = {}
    for m in (1, prefill_min_m):
        probe = torch.empty((m, k), dtype=dtype, device=weight.device)
        bits = native.ggml_dense_upstream_capabilities(
            weight, probe, weight_type, weight.shape[0]
        )
        result["capabilities"][f"m{m}"] = {
            "mmvq": bool(bits & 4),
            "mmq": bool(bits & 8),
            "dequant_blas": bool(bits & 16),
        }
        if not bits:
            result["reason"] = "native_shape_or_storage_not_admitted"
    return result


def native_dequantize(
    weight: torch.Tensor, weight_type: int, rows: int, columns: int, dtype
) -> torch.Tensor | None:
    if weight_type not in NATIVE_TYPES or not native_available() or not weight.is_cuda:
        return None
    return torch.ops._C_gguf.ggml_dequantize_upstream(
        weight, weight_type, rows, columns, dtype
    )
