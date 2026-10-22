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


@torch.compiler.assume_constant_result
def _sm70_device():
    return current_platform.is_device_capability(70)


def _can_combine_fc(model, embedding, hidden):
    if (
        envs.VLLM_BATCH_INVARIANT
        or not is_sm70_decode_graph_compiling()
        or embedding.ndim != 2
        or hidden.ndim != 2
        or embedding.shape[-1] != 2560
        or hidden.shape[-1] != 10240
        or embedding.dtype != torch.float16
        or hidden.dtype != torch.float16
        or not embedding.is_cuda
        or not hidden.is_cuda
        or not _sm70_device()
    ):
        return False
    return all(
        type(layer) is ColumnParallelLinear
        and type(layer.quant_method) is UnquantizedLinearMethod
        and layer.tp_size == 4
        and layer.gather_output
        and not layer.return_bias
        and layer.bias is None
        and layer.weight.shape == (640, 2560)
        and layer.weight.dtype == torch.float16
        for layer in (model.fc_embedding, model.fc_hidden)
    )


def maybe_combine_fc(model, embedding, hidden):
    if not _can_combine_fc(model, embedding, hidden):
        return None
    rows = embedding.shape[0]
    e = model.pre_fc_norm_embedding(embedding)
    h = model.pre_fc_norm_hidden(hidden).view(rows, 4, 2560)
    el = model.fc_embedding.quant_method.apply(model.fc_embedding, e)
    hl = model.fc_hidden.quant_method.apply(model.fc_hidden, h)
    # Addition commutes with the feature gather, with exactly the same FP16
    # rounding per element. Norms/projections and TP ownership are unchanged.
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
