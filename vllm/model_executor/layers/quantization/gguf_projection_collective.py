# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qualified projection/TP/norm boundary with live-M canonical fallback."""

import operator

import torch
import torch.fx as fx
from torch._higher_order_ops.auto_functionalize import (
    auto_functionalized,
    auto_functionalized_v2,
)

from vllm.config import get_current_vllm_config_or_none
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)
_FORMAT_NAMES = {0: "Q4_K", 3: "IQ4_XS", 5: "IQ3_S"}


def _project_norm(
    x: torch.Tensor,
    codes: list[torch.Tensor],
    high: list[torch.Tensor],
    scales: list[torch.Tensor],
    formats: list[int],
    widths: list[int],
    kw: int,
    tn: int,
    split: int,
    pair: bool,
    gdn_heads: bool,
    partials: torch.Tensor,
    counters: torch.Tensor,
    floating: torch.Tensor | None,
    fallback_floating: list[torch.Tensor],
    fallback_codes: list[torch.Tensor],
    fallback_stats: list[torch.Tensor],
    fallback_caches: list[torch.Tensor | None],
    descriptors: list[int],
    cache_bands: list[int],
    blas_bands: list[int],
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    group_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.distributed.parallel_state import _groups

    from .gguf_dmv import _project, table

    group = _groups[group_name]()
    if group is None:
        raise RuntimeError("GGUF projection collective group has been destroyed")
    communicator = group.device_communicator
    ca = getattr(communicator, "ca_comm", None)
    reason = None
    if x.ndim != 2 or tuple(x.shape) != (8, 4352):
        reason = "live_shape_requires_m8_k4352"
    elif formats not in ([0], [3], [5]) or widths != [5120]:
        reason = "projection_format_or_width_not_calibrated"
    elif (kw, tn, split, pair, gdn_heads) != (4, 2, 1, False, False):
        reason = "projection_schedule_not_calibrated"
    elif floating is not None or fallback_floating:
        reason = "floating_segments_not_supported"
    elif not hasattr(torch.ops._C, "gguf_dmv_allreduce_norm_sm70_out"):
        reason = "packaged_projection_collective_operator_missing"
    elif (
        group.world_size != 4
        or ca is None
        or ca.disabled
        or not ca.fully_connected
        or not getattr(communicator, "use_custom_allreduce", False)
        or ca.sm70_tp4_push_buffer_ptrs is None
    ):
        reason = "requires_tp4_custom_ar_and_full_nvlink"
    elif (
        not x.is_cuda
        or x.dtype != torch.float16
        or residual.dtype != torch.float32
        or weight.dtype != torch.float32
        or not all(t.is_contiguous() for t in (x, residual, weight))
        or tuple(residual.shape) != (8, 5120)
        or tuple(weight.shape) != (5120,)
    ):
        reason = "requires_contiguous_fp16_input_and_fp32_residual_norm"
    elif torch.cuda.get_device_capability(x.device) != (7, 0):
        reason = "requires_sm70"
    cfg = get_current_vllm_config_or_none()
    if cfg is not None:
        key = f"gguf_projection_collective:{formats}:{tuple(x.shape)}"
        cfg.kernel_config.collective_kernel_selections[key] = {
            "operator": "gguf_dmv_allreduce_norm_sm70_out",
            "source_type": _FORMAT_NAMES.get(formats[0], "uncalibrated"),
            "min_m": 8,
            "max_m": 8,
            "graph_safe": True,
            "reason": reason,
            "fallback": "projection_then_collective_norm",
        }
    if reason is not None:
        logger.debug_once("GGUF projection collective fallback: %s", reason)
        projected = _project(
            x,
            codes,
            high,
            scales,
            formats,
            widths,
            kw,
            tn,
            split,
            pair,
            gdn_heads,
            partials,
            counters,
            floating,
            fallback_floating,
            fallback_codes,
            fallback_stats,
            fallback_caches,
            descriptors,
            cache_bands,
            blas_bands,
        )
        return group._sm70_tp4_all_reduce_gemma_rms_norm_out_place(
            projected, residual, weight, epsilon
        )
    assert ca is not None
    output = x.new_empty((8, 5120))
    residual_out = torch.empty_like(residual)
    partial = torch.empty_like(output)
    torch.ops._C.gguf_dmv_allreduce_norm_sm70_out(
        x,
        codes[0],
        scales[0],
        table(x.device),
        partial,
        ca.sm70_tp4_push_buffer_ptrs,
        ca.rank,
        residual,
        weight,
        output,
        residual_out,
        formats[0],
        epsilon,
    )
    logger.info_once("SM70 GGUF projection/TP4/norm tile pipeline route hit.")
    return output, residual_out


def _project_norm_fake(
    x,
    codes,
    high,
    scales,
    formats,
    widths,
    kw,
    tn,
    split,
    pair,
    gdn_heads,
    partials,
    counters,
    floating,
    fallback_floating,
    fallback_codes,
    fallback_stats,
    fallback_caches,
    descriptors,
    cache_bands,
    blas_bands,
    residual,
    weight,
    epsilon,
    group_name,
):
    return x.new_empty((*x.shape[:-1], widths[0])), torch.empty_like(
        residual, dtype=torch.float32
    )


direct_register_custom_op(
    "gguf_projection_collective_norm",
    _project_norm,
    fake_impl=_project_norm_fake,
    mutates_args=["partials", "counters"],
)


def fuse_projection_collectives(graph: fx.Graph) -> int:
    """Fold sole-consumer projections, preserving functionalized scratch outputs.

    M stays a runtime guard so shared C4/prefill graphs retain canonical routing.
    """
    count = 0
    projection_op = torch.ops.vllm.gguf_dmv_projection.default
    fused_op = torch.ops.vllm.gguf_projection_collective_norm.default
    argument_names = [a.name for a in projection_op._schema.arguments]
    for norm in list(graph.nodes):
        if norm.op != "call_function" or norm.target != (
            torch.ops.vllm.sm70_tp4_all_reduce_gemma_rms_norm.default
        ):
            continue
        projection = norm.args[0]
        if not isinstance(projection, fx.Node) or len(projection.users) != 1:
            continue
        functional = None
        if projection.op == "call_function" and projection.target == projection_op:
            args = projection.args
        elif (
            projection.op == "call_function"
            and projection.target == operator.getitem
            and projection.args[1] == 0
            and isinstance(projection.args[0], fx.Node)
            and projection.args[0].target
            in (auto_functionalized, auto_functionalized_v2)
            and projection.args[0].args == (projection_op,)
        ):
            functional = projection.args[0]
            args = tuple(functional.kwargs.get(name) for name in argument_names)
            if any(
                user.op != "call_function"
                or user.target != operator.getitem
                or user.args[1] not in (0, 1, 2)
                for user in functional.users
            ):
                continue
            if any(
                user.op != "call_function"
                or user.target != operator.getitem
                or user.args[1] not in (0, 1)
                for user in norm.users
            ):
                continue
        else:
            continue
        if (
            len(args) != 21
            or list(args[4]) not in ([0], [3], [5])
            or list(args[5]) != [5120]
            or tuple(args[6:11]) != (4, 2, 1, False, False)
            or args[13] is not None
            or list(args[14])
        ):
            continue
        value = args[0].meta.get("val")
        if value is None or value.ndim != 2 or value.shape[-1] != 4352:
            continue
        group_name = norm.kwargs.get("group_name")
        if group_name is None and len(norm.args) == 5:
            group_name = norm.args[4]
        if not isinstance(group_name, str):
            continue
        anchor = functional if functional is not None else projection
        positions = {node: i for i, node in enumerate(graph.nodes)}
        if any(
            isinstance(argument, fx.Node) and positions[argument] >= positions[anchor]
            for argument in norm.args[1:4]
        ):
            continue
        with graph.inserting_before(anchor):
            if functional is None:
                replacement = graph.call_function(
                    fused_op, args=(*args, *norm.args[1:4], group_name)
                )
                replacement.meta = norm.meta.copy()
                norm.replace_all_uses_with(replacement)
            else:
                kwargs = dict(functional.kwargs)
                kwargs.update(
                    residual=norm.args[1],
                    weight=norm.args[2],
                    epsilon=norm.args[3],
                    group_name=group_name,
                )
                replacement = graph.call_function(
                    functional.target, args=(fused_op,), kwargs=kwargs
                )
                replacement.meta = functional.meta.copy()
                norm_values = norm.meta.get("val")
                original_values = functional.meta.get("val")
                if norm_values is not None and original_values is not None:
                    replacement.meta["val"] = (*norm_values, *original_values[1:])
                for user in list(norm.users):
                    item = graph.call_function(
                        operator.getitem, args=(replacement, user.args[1])
                    )
                    item.meta = user.meta.copy()
                    user.replace_all_uses_with(item)
                    graph.erase_node(user)
                for user in list(functional.users):
                    if user is not projection:
                        user.args = (replacement, user.args[1] + 1)
        graph.erase_node(norm)
        graph.erase_node(projection)
        if functional is not None:
            graph.erase_node(functional)
        count += 1
    return count
