# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preserve local FP16 residual bytes while sharing the draft FC gather."""

import torch

import vllm.envs as envs
from vllm.compilation.sm70_decode_graph import is_sm70_decode_graph_compiling
from vllm.distributed import tensor_model_parallel_all_gather
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
