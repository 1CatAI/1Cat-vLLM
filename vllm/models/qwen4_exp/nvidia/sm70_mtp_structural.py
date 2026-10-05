# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Structural MTP candidates, installed only by benchmark control workers.

Promotion to default requires shared model gates. The shared candidate keeps
TP reduction outside this local expert chain, exactly as the original MLP.
"""

from types import MethodType

import torch

from vllm.compilation.sm70_decode_graph import use_sm70_decode_graph_semantics
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP
from vllm.utils.torch_utils import direct_register_custom_op


def _shared_chain(
    x: torch.Tensor,
    packed_up: torch.Tensor,
    packed_down: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(x)
    partial = x.new_empty((8, x.shape[0], 320), dtype=torch.float32)
    gate_logits = x.new_empty(x.shape[0], dtype=torch.float32)
    torch.ops._C.qwen38_shared_chain_sm70_out(
        out, partial, gate_logits, x, packed_up, packed_down, gate
    )
    return out


direct_register_custom_op(
    op_name="sm70_mtp_shared_chain_probe",
    op_func=_shared_chain,
    mutates_args=[],
    fake_impl=lambda x, packed_up, packed_down, gate: torch.empty_like(x),
)


def _shared_forward(layer, x):
    if (
        use_sm70_decode_graph_semantics()
        and x.ndim == 2
        and x.shape == (5, 2560)
        and x.dtype == torch.float16
        and x.is_cuda
        and x.is_contiguous()
    ):
        return torch.ops.vllm.sm70_mtp_shared_chain_probe(
            x,
            layer.gate_up_proj._sm70_mtp_shared_packed,
            layer._sm70_shared_down_probe,
            layer.expert_gate.weight,
        )
    return layer._sm70_shared_original_forward(x)


def prepare_shared_chain_probe(model):
    if not hasattr(torch.ops._C, "qwen38_shared_chain_sm70_out"):
        raise RuntimeError("Rebuild the normal SM70 extension for shared-chain probe")
    prepared = 0
    for layer in model.modules():
        if not isinstance(layer, Qwen2MoeMLP) or layer.expert_gate is None:
            continue
        packed = getattr(layer.gate_up_proj, "_sm70_mtp_shared_packed", None)
        if (
            packed is None
            or tuple(layer.down_proj.weight.shape) != (2560, 160)
            or layer.down_proj.reduce_results
            or layer.down_proj.bias is not None
            or layer.gate_up_proj.bias is not None
            or tuple(layer.expert_gate.weight.shape) != (1, 2560)
            or layer.expert_gate.bias is not None
        ):
            continue
        layer.register_buffer(
            "_sm70_shared_down_probe",
            layer.down_proj.weight.detach().t().contiguous(),
            persistent=False,
        )
        object.__setattr__(layer, "_sm70_shared_original_forward", layer.forward)
        layer.forward = MethodType(_shared_forward, layer)  # type: ignore[method-assign]
        prepared += 1
    return prepared
