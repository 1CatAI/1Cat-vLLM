# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measured output projections with runtime-M and head-layout fallback."""

from dataclasses import asdict

import torch
from torch.nn import Parameter

from vllm.model_executor.kernels.gguf import small_output_capability
from vllm.model_executor.layers.quantization.gguf_layout import GGUFHeadTilingLayout
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


def _small_output(
    x: torch.Tensor,
    records: torch.Tensor,
    partials: torch.Tensor,
    counters: torch.Tensor,
    source_type: int,
    gdn_head_tiling: bool,
    fallback_head_tiling: bool,
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
        torch.ops._C.gguf_small_output_sm70_out(
            out, rows, records, partials, counters, source_type, 1, gdn_head_tiling
        )
    else:
        if fallback_head_tiling:
            rows = GGUFHeadTilingLayout(3, 128).input_to_gguf(rows)
        out = _prepared_gguf_projection(
            rows,
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


def _small_output_fake(
    x: torch.Tensor,
    records: torch.Tensor,
    partials: torch.Tensor,
    counters: torch.Tensor,
    source_type: int,
    gdn_head_tiling: bool,
    fallback_head_tiling: bool,
    codes: list[torch.Tensor],
    stats: list[torch.Tensor],
    caches: list[torch.Tensor | None],
    descriptors: list[int],
    cache_bands: list[int],
    blas_bands: list[int],
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], 5120))


direct_register_custom_op(
    op_name="gguf_small_output",
    op_func=_small_output,
    # The admitted one-partition kernel writes directly to its new output;
    # it never touches reduction storage. Keep those operands read-only so
    # functionalization does not clone/copy unused workspace during replay.
    mutates_args=[],
    fake_impl=_small_output_fake,
)


def prepare_small_output(layer, sources, projections, enabled, layout):
    source_type = sources[0][1] if len(sources) == 1 else -1
    kernel = projections[0].kernel if len(projections) == 1 else None
    shape = kernel.config.partition_weight_shape if kernel else (0, 0)
    gdn = layer.prefix.endswith(".linear_attn.out_proj")
    role = "gdn_out" if gdn else "attention_o"
    device = sources[0][0].device if sources else torch.device("cpu")
    hardware = (
        current_platform.get_device_capability(device.index)
        if device.type == "cuda"
        else None
    )
    admission = small_output_capability(
        source_type,
        shape[0],
        shape[1],
        kernel.config.act_type if kernel else None,
        enabled,
        hardware.to_int() if hardware else 0,
    )
    reason = admission.reason
    if reason is None:
        if not gdn and not layer.prefix.endswith(".self_attn.o_proj"):
            reason = "requires_calibrated_output_projection"
        elif (gdn and layout != GGUFHeadTilingLayout(3, 128)) or (
            not gdn and layout is not None
        ):
            reason = "output_projection_head_layout_has_no_calibration"
        elif sources[0][0].dtype != torch.uint8 or sources[0][0].shape != (
            5120,
            6 * _SOURCE_BLOCK_BYTES[source_type],
        ):
            reason = "output_projection_source_storage_has_no_calibration"
        elif projections[0].logical_output_size != 5120:
            reason = "canonical_fallback_unavailable"
    if reason is None:
        records = torch.from_numpy(
            _SOURCE_PACKERS[source_type](sources[0][0].detach().cpu().numpy())
        ).to(device)
        layer.register_parameter("gguf_small_output_records", Parameter(records, False))
        layer.register_buffer(
            "gguf_small_output_partials",
            torch.empty((80, 2, 512), dtype=torch.float32, device=device),
            persistent=False,
        )
        layer.register_buffer(
            "gguf_small_output_counters",
            torch.zeros(80, dtype=torch.int32, device=device),
            persistent=False,
        )
        layer.gguf_small_output_type = source_type
        layer.gguf_small_output_head_tiling = gdn
        layer.gguf_small_output_fallback_tiling = (
            gdn and not projections[0].input_layout_restored
        )
    result = asdict(admission)
    result.update(
        reason=reason,
        role=role,
        n=5120,
        k=1536,
        split_k=1,
        input_head_tiling=gdn,
        fallback="canonical",
    )
    return result


def apply_small_output(layer, x):
    return torch.ops.vllm.gguf_small_output(
        x,
        layer.gguf_small_output_records,
        layer.gguf_small_output_partials,
        layer.gguf_small_output_counters,
        layer.gguf_small_output_type,
        layer.gguf_small_output_head_tiling,
        layer.gguf_small_output_fallback_tiling,
        *prepared_projection_arguments(layer.gguf_tm_projections),
    )
