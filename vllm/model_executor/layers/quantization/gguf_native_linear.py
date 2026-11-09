# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measured small-M single projections with opaque runtime-M fallback."""

from dataclasses import asdict

import torch
from torch.nn import Parameter

from vllm.model_executor.kernels.gguf import native_linear_capability
from vllm.model_executor.layers.quantization.gguf_native_pair import (
    _SOURCE_BLOCK_BYTES,
    _SOURCE_PACKERS,
)
from vllm.model_executor.layers.quantization.gguf_turbomind import (
    _prepared_gguf_projection,
    prepared_projection_arguments,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op


def _native_linear(
    x: torch.Tensor,
    records: torch.Tensor,
    partials: torch.Tensor,
    counters: torch.Tensor,
    source_type: int,
    codes: list[torch.Tensor],
    stats: list[torch.Tensor],
    caches: list[torch.Tensor | None],
    descriptors: list[int],
    cache_bands: list[int],
    blas_bands: list[int],
) -> torch.Tensor:
    rows = x.reshape(-1, x.shape[-1]).contiguous()
    if rows.shape[0] == 8:
        out = rows.new_empty((8, 5120))
        if source_type == 10:
            torch.ops._C.gguf_canonical_linear_n64_sm70_out(
                out, rows, codes[0], stats[0], partials, counters, 2, 16
            )
        else:
            torch.ops._C.gguf_native_linear_n64_sm70_out(
                out, rows, records, partials, counters, source_type
            )
    else:
        out = _prepared_gguf_projection(
            x,
            codes[0],
            stats[0],
            caches[0],
            descriptors[0],
            descriptors[1],
            descriptors[2],
            descriptors[3],
            descriptors[4],
            descriptors[5],
            descriptors[6],
            cache_bands,
            blas_bands,
        )
    return out.reshape(*x.shape[:-1], 5120)


def _native_linear_fake(
    x: torch.Tensor,
    records: torch.Tensor,
    partials: torch.Tensor,
    counters: torch.Tensor,
    source_type: int,
    codes: list[torch.Tensor],
    stats: list[torch.Tensor],
    caches: list[torch.Tensor | None],
    descriptors: list[int],
    cache_bands: list[int],
    blas_bands: list[int],
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], 5120))


direct_register_custom_op(
    op_name="gguf_native_linear",
    op_func=_native_linear,
    mutates_args=["partials", "counters"],
    fake_impl=_native_linear_fake,
)


def prepare_native_linear(layer, sources, projections, enabled: bool):
    source_type = sources[0][1] if len(sources) == 1 else -1
    kernel = projections[0].kernel if len(projections) == 1 else None
    shape = kernel.config.partition_weight_shape if kernel else (0, 0)
    device = sources[0][0].device if sources else torch.device("cpu")
    capability = (
        current_platform.get_device_capability(device.index)
        if device.type == "cuda"
        else None
    )
    admission = native_linear_capability(
        source_type,
        shape[0],
        shape[1],
        kernel.config.act_type if kernel else None,
        enabled,
        capability.to_int() if capability else 0,
    )
    reason = admission.reason
    if reason is None:
        if not layer.prefix.endswith(".down_proj"):
            reason = "requires_calibrated_down_projection"
        elif sources[0][0].dtype != torch.uint8 or sources[0][0].shape != (
            5120,
            17 * _SOURCE_BLOCK_BYTES[source_type],
        ):
            reason = "single_projection_source_shape_has_no_calibration"
        elif projections[0].logical_output_size != 5120:
            reason = "canonical_fallback_unavailable"
    if reason is None:
        source = sources[0][0]
        records = (
            torch.empty(0, dtype=torch.uint8, device=device)
            if source_type == 10
            else torch.from_numpy(
                _SOURCE_PACKERS[source_type](source.detach().cpu().numpy())
            ).to(device)
        )
        layer.register_parameter(
            "gguf_native_linear_records", Parameter(records, False)
        )
        layer.register_buffer(
            "gguf_native_linear_partials",
            torch.empty((80, 2, 512), dtype=torch.float32, device=device),
            persistent=False,
        )
        layer.register_buffer(
            "gguf_native_linear_counters",
            torch.zeros(80, dtype=torch.int32, device=device),
            persistent=False,
        )
        layer.gguf_native_linear_type = source_type
    result = asdict(admission)
    result.update(reason=reason, n=5120, k=4352, fallback="canonical")
    return result


def apply_native_linear(layer, x):
    return torch.ops.vllm.gguf_native_linear(
        x,
        layer.gguf_native_linear_records,
        layer.gguf_native_linear_partials,
        layer.gguf_native_linear_counters,
        layer.gguf_native_linear_type,
        *prepared_projection_arguments(layer.gguf_tm_projections),
    )
