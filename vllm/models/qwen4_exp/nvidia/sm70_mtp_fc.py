# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preserve local FP16 residual bytes while sharing the draft FC gather."""

import torch

import vllm.envs as envs
from vllm.compilation.sm70_decode_graph import is_sm70_decode_graph_compiling
from vllm.distributed import get_tp_group, tensor_model_parallel_all_gather
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op


def _fc_add_all_gather(el: torch.Tensor, hl: torch.Tensor) -> torch.Tensor:
    if el.is_cuda and 1 <= el.shape[0] <= 64 and el.shape[1] == 640:
        try:
            group = get_tp_group()
            comm = getattr(group.device_communicator, "hc_ll_comm", None)
        except (AssertionError, AttributeError, RuntimeError):
            comm = None
        if comm is not None and getattr(comm, "can_add_all_gather_fc", False):
            out = comm.add_all_gather_fc(
                el if el.is_contiguous() else el.contiguous(),
                hl if hl.is_contiguous() else hl.contiguous(),
            )
            if out is not None:
                return out
    return tensor_model_parallel_all_gather(el.unsqueeze(-2) + hl).flatten(-2)


def _fc_add_all_gather_fake(el: torch.Tensor, hl: torch.Tensor) -> torch.Tensor:
    return el.new_empty((el.shape[0], 10240))


direct_register_custom_op(
    op_name="qwen38_sm70_fc_add_all_gather",
    op_func=_fc_add_all_gather,
    fake_impl=_fc_add_all_gather_fake,
)


def _fc_shard_ready(layer) -> bool:
    if (
        type(layer) is not ColumnParallelLinear
        or type(layer.quant_method) is not UnquantizedLinearMethod
        or layer.tp_size not in (2, 4)
        or not layer.gather_output
        or layer.return_bias
        or layer.bias is not None
    ):
        return False
    expected = (2560 // layer.tp_size, 2560)
    if getattr(layer, "_sm70_mtp_draft_qpn8_ready", False):
        return getattr(layer, "_sm70_mtp_draft_qpn8_shape", None) == expected
    return layer.weight.dtype == torch.float16 and layer.weight.shape == expected


def _can_combine_fc(model, embedding, hidden, *, allow_eager: bool = False):
    if (
        envs.VLLM_BATCH_INVARIANT
        or (not allow_eager and not is_sm70_decode_graph_compiling())
        or embedding.ndim != 2
        or hidden.ndim != 2
        or embedding.shape[-1] != 2560
        or hidden.shape[-1] != 10240
        or embedding.dtype != torch.float16
        or hidden.dtype != torch.float16
        or not embedding.is_cuda
        or not hidden.is_cuda
        or not current_platform.is_device_capability(70)
    ):
        return False
    return all(
        _fc_shard_ready(layer) for layer in (model.fc_embedding, model.fc_hidden)
    )


def maybe_combine_fc(model, embedding, hidden, *, allow_eager: bool = False):
    if not _can_combine_fc(model, embedding, hidden, allow_eager=allow_eager):
        return None
    rows = embedding.shape[0]
    if allow_eager and not is_sm70_decode_graph_compiling() and rows >= 64:
        from .ops.hc import grouped_gemma_rmsnorm

        e = grouped_gemma_rmsnorm(
            embedding.contiguous(),
            model.pre_fc_norm_embedding.weight,
            model.pre_fc_norm_embedding.variance_epsilon,
            1,
        )
        h = grouped_gemma_rmsnorm(
            hidden.contiguous(),
            model.pre_fc_norm_hidden.weight,
            model.pre_fc_norm_hidden.variance_epsilon,
            4,
        ).view(rows, 4, 2560)
    else:
        e = model.pre_fc_norm_embedding(embedding)
        h = model.pre_fc_norm_hidden(hidden).view(rows, 4, 2560)
    el = model.fc_embedding.quant_method.apply(model.fc_embedding, e)
    hl = model.fc_hidden.quant_method.apply(model.fc_hidden, h)
    # Addition commutes with the feature gather, with exactly the same FP16
    # rounding per element. Norms/projections and TP ownership are unchanged.
    if (
        el.is_cuda
        and el.shape[-1] == 640
        and tensor_model_parallel_all_gather.__module__.startswith("vllm.")
    ):
        return torch.ops.vllm.qwen38_sm70_fc_add_all_gather(el, hl)
    return tensor_model_parallel_all_gather(el.unsqueeze(-2) + hl).flatten(-2)


def maybe_sequence_rows_fc(model, embedding, hidden):
    """Project owned prefill rows without gathering the full HC activation.

    Borrow the existing column shards only for this call. The small weight
    gathers replace activation gathers whose size grows with the query. The
    caller already qualified TP4 HC prefill; other linear methods keep their
    original forward and row-shard fallback.
    """
    if envs.VLLM_BATCH_INVARIANT or not all(
        type(layer) is ColumnParallelLinear
        and type(layer.quant_method) is UnquantizedLinearMethod
        and layer.tp_size == 4
        and layer.gather_output
        and not layer.return_bias
        and layer.bias is None
        and layer.weight.shape == (640, 2560)
        and layer.weight.dtype == torch.float16
        for layer in (model.fc_embedding, model.fc_hidden)
    ):
        return None
    from .sm70_hc_prefill_peer import hc_sequence_shard_rows

    group = get_tp_group()
    embedding = hc_sequence_shard_rows(embedding)
    hidden = hc_sequence_shard_rows(hidden)
    embedding = model.pre_fc_norm_embedding(embedding)
    hidden = model.pre_fc_norm_hidden(hidden).view(-1, 4, 2560)
    weight = group.all_gather(model.fc_embedding.weight, dim=0)
    embedding = torch.nn.functional.linear(embedding, weight)
    del weight
    weight = group.all_gather(model.fc_hidden.weight, dim=0)
    hidden = torch.nn.functional.linear(hidden, weight)
    return (embedding.unsqueeze(-2) + hidden).flatten(-2)
