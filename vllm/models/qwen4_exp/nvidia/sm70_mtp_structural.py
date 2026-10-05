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

_probe_calls = {
    "shared": 0,
    "draft-qpn8": 0,
    "draft-int8": 0,
    "draft-int8-block32": 0,
    "draft-int8-packed": 0,
    "target-head-qpn8": 0,
}
_probe_widths: dict[str, set[int]] = {name: set() for name in _probe_calls}


def structural_route_proof():
    """Actual opaque-op invocations during execution/capture, not registration."""
    return {
        name: {"calls": count, "widths": sorted(_probe_widths[name])}
        for name, count in _probe_calls.items()
    }


def _shared_chain(
    x: torch.Tensor,
    packed_up: torch.Tensor,
    packed_down: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    _probe_calls["shared"] += 1
    _probe_widths["shared"].add(int(x.shape[0]))
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
            layer.down_proj.weight.detach()
            .reshape(80, 32, 10, 2, 8)
            .permute(0, 2, 3, 1, 4)
            .contiguous(),
            persistent=False,
        )
        object.__setattr__(layer, "_sm70_shared_original_forward", layer.forward)
        layer.forward = MethodType(_shared_forward, layer)  # type: ignore[method-assign]
        prepared += 1
    return prepared


def _draft_expert_chain(
    x: torch.Tensor,
    codes13: torch.Tensor,
    scales13: torch.Tensor,
    codes2: torch.Tensor,
    scales2: torch.Tensor,
    ids: torch.Tensor,
    probabilities: torch.Tensor,
    integer: bool = False,
    block32: bool = False,
    canonical: bool = False,
) -> torch.Tensor:
    role = (
        "draft-int8-packed"
        if canonical
        else "draft-int8-block32"
        if block32
        else "draft-int8"
        if integer
        else "draft-qpn8"
    )
    _probe_calls[role] += 1
    _probe_widths[role].add(int(x.shape[0]))
    out = torch.empty_like(x)
    activation = x.new_empty((x.shape[0] * 10, 160))
    chain = (
        torch.ops._C.sm70_mtp_moe_int8_block32_chain_out
        if block32
        else torch.ops._C.sm70_mtp_moe_int8_chain_out
        if integer
        else torch.ops._C.sm70_mtp_moe_qpn8_chain_out
    )
    chain(out, activation, x, codes13, scales13, codes2, scales2, ids, probabilities)
    return out


direct_register_custom_op(
    op_name="sm70_mtp_draft_qpn8_probe",
    op_func=_draft_expert_chain,
    mutates_args=[],
    fake_impl=lambda x,
    codes13,
    scales13,
    codes2,
    scales2,
    ids,
    probabilities,
    integer=False,
    block32=False,
    canonical=False: (torch.empty_like(x)),
)


def _draft_packed_fallback(
    x: torch.Tensor,
    codes13: torch.Tensor,
    scales13: torch.Tensor,
    codes2: torch.Tensor,
    scales2: torch.Tensor,
    ids: torch.Tensor,
    probabilities: torch.Tensor,
) -> torch.Tensor:
    from .sm70_mtp_int8 import packed_int8_fallback_experts

    _probe_calls["draft-int8-packed"] += 1
    _probe_widths["draft-int8-packed"].add(int(x.shape[0]))
    return packed_int8_fallback_experts(
        x, codes13, scales13, codes2, scales2, probabilities, ids
    )


direct_register_custom_op(
    op_name="sm70_mtp_draft_packed_fallback_probe",
    op_func=_draft_packed_fallback,
    mutates_args=[],
    fake_impl=lambda x,
    codes13,
    scales13,
    codes2,
    scales2,
    ids,
    probabilities: torch.empty_like(x),
)


def _draft_expert_apply(
    method, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
):
    if (
        use_sm70_decode_graph_semantics()
        and x.ndim == 2
        and x.shape[0] in (1, 5)
        and x.shape[1] == 2560
        and x.is_cuda
        and x.dtype == torch.float16
        and x.is_contiguous()
        and topk_ids.dtype == torch.int32
        and topk_weights.dtype == torch.float32
        and topk_ids.shape == topk_weights.shape == (x.shape[0], 10)
        and topk_ids.is_contiguous()
        and topk_weights.is_contiguous()
    ):
        # Retain the runner's shared-expert ordering and outer TP reduction.
        # NO_OVERLAP and auxiliary-stream experts are handled by the runner;
        # only an MK-internal shared expert needs an explicit call here.
        if shared_experts is not None:
            from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
                SharedExpertsOrder,
            )

            shared_experts.apply(
                shared_experts_input, SharedExpertsOrder.MK_INTERNAL_OVERLAPPED
            )
        return torch.ops.vllm.sm70_mtp_draft_qpn8_probe(
            x,
            layer._sm70_mtp_qpn8_codes13,
            layer._sm70_mtp_qpn8_scales13,
            layer._sm70_mtp_qpn8_codes2,
            layer._sm70_mtp_qpn8_scales2,
            topk_ids,
            topk_weights,
            layer._sm70_mtp_expert_integer,
            layer._sm70_mtp_expert_block32,
            layer._sm70_mtp_expert_canonical,
        )
    if layer._sm70_mtp_expert_canonical:
        if shared_experts is not None:
            from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
                SharedExpertsOrder,
            )

            shared_experts.apply(
                shared_experts_input, SharedExpertsOrder.MK_INTERNAL_OVERLAPPED
            )
        return torch.ops.vllm.sm70_mtp_draft_packed_fallback_probe(
            x,
            layer._sm70_mtp_qpn8_codes13,
            layer._sm70_mtp_qpn8_scales13,
            layer._sm70_mtp_qpn8_codes2,
            layer._sm70_mtp_qpn8_scales2,
            topk_ids,
            topk_weights,
        )
    return method._sm70_mtp_original_apply(
        layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
    )


def prepare_draft_expert_qpn8_probe(
    draft_model, *, integer=False, block32=False, canonical=False
):
    """Prepare only the provided proposer subtree, retaining FP16 fallback."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
        UnquantizedFusedMoEMethod,
    )
    from vllm.model_executor.layers.quantization.sm70_online_qpn8 import (
        prepare_channel_qpn8_weight,
    )

    if block32 and not integer:
        raise ValueError("Block scaling requires INT8 weights")
    if canonical and (not block32 or not integer):
        raise ValueError("Canonical storage requires block32 INT8")
    if canonical and getattr(draft_model, "_sm70_decode_graph_model", None) is not None:
        raise ValueError("Canonical probe must precede any shared-weight decode clone")
    if integer:
        from vllm.models.qwen4_exp.nvidia.sm70_mtp_int8 import (
            prepare_int8_expert_weight,
        )

        prepare = lambda w: prepare_int8_expert_weight(w, block32=block32)
    else:
        prepare = prepare_channel_qpn8_weight
    op_name = (
        "sm70_mtp_moe_int8_block32_chain_out"
        if block32
        else "sm70_mtp_moe_int8_chain_out"
        if integer
        else "sm70_mtp_moe_qpn8_chain_out"
    )
    if not hasattr(torch.ops._C, op_name):
        raise RuntimeError("Rebuild the normal SM70 extension for draft QPN8 probe")
    prepared = 0
    for layer in draft_model.modules():
        if not isinstance(layer, FusedMoE):
            continue
        method = layer.quant_method
        if (
            not isinstance(method, UnquantizedFusedMoEMethod)
            or method.is_monolithic
            or layer.activation != MoEActivation.SILU
            or layer.apply_router_weight_on_input
            or layer.expert_map is not None
            or layer.w13_weight.shape != (512, 320, 2560)
            or layer.w2_weight.shape != (512, 2560, 160)
            or layer.w13_weight.dtype != torch.float16
            or layer.w2_weight.dtype != torch.float16
            or not layer.w13_weight.is_cuda
            or not layer.w2_weight.is_cuda
            or not layer.w13_weight.is_contiguous()
            or not layer.w2_weight.is_contiguous()
            or getattr(layer, "w13_bias", None) is not None
            or getattr(layer, "w2_bias", None) is not None
        ):
            continue
        if hasattr(method, "_sm70_mtp_original_apply"):
            raise RuntimeError("Draft QPN8 probe method was already prepared")
        for role, weight, k, n in (
            ("13", layer.w13_weight, 2560, 320),
            ("2", layer.w2_weight, 160, 2560),
        ):
            codes, scales = prepare(weight.view(-1, k))
            layer.register_buffer(
                "_sm70_mtp_qpn8_codes" + role, codes, persistent=False
            )
            layer.register_buffer(
                "_sm70_mtp_qpn8_scales" + role,
                scales.view(k // 32, 512, n) if block32 else scales.view(512, n),
                persistent=False,
            )
        layer._sm70_mtp_expert_integer = integer
        layer._sm70_mtp_expert_block32 = block32
        layer._sm70_mtp_expert_canonical = canonical
        if canonical:
            # Byte buffers and parameter views share storage. Do not save the
            # checkpoint FP16 tensors in a fallback closure or module field.
            layer.w13_weight = torch.nn.Parameter(
                layer._sm70_mtp_qpn8_codes13.view(torch.int8).view(512, 320, 2560),
                requires_grad=False,
            )
            layer.w2_weight = torch.nn.Parameter(
                layer._sm70_mtp_qpn8_codes2.view(torch.int8).view(512, 2560, 160),
                requires_grad=False,
            )
        object.__setattr__(method, "_sm70_mtp_original_apply", method.apply)
        method.apply = MethodType(_draft_expert_apply, method)  # type: ignore[method-assign]
        prepared += 1
    return prepared


def _target_head_qpn8(
    x: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    from vllm import _sm70_ops as ops

    _probe_calls["target-head-qpn8"] += 1
    _probe_widths["target-head-qpn8"].add(int(x.shape[0]))
    out = x.new_empty(x.shape[0], codes.shape[1])
    ops.fp8_qpn8_gemm_sm70_out(out, x, codes, scales, 8, 2, True, False)
    return out


direct_register_custom_op(
    op_name="sm70_mtp_target_head_qpn8_probe",
    op_func=_target_head_qpn8,
    mutates_args=[],
    fake_impl=lambda x, codes, scales: x.new_empty(x.shape[0], codes.shape[1]),
)


def prepare_target_head_qpn8_probe(target_model, draft_model):
    """Benchmark-only target head view; target output projections are unchanged."""
    from vllm.models.qwen4_exp.nvidia.sm70_mtp_head import MTPQPN8Head
    from vllm.v1.worker.gpu.spec_decode.eagle.utils import get_target_lm_head

    language = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    head = get_target_lm_head(target_model, language)
    shared = draft_model._sm70_draft_head
    if not isinstance(shared, MTPQPN8Head) or shared.head is not head:
        raise RuntimeError("Target probe requires the identical admitted draft head")
    view = MTPQPN8Head(head, shared_view=shared)

    def apply(view, layer, x, bias=None):
        rows = x.numel() // x.shape[-1]
        if x.dtype != torch.float16 or not 1 <= rows <= 8:
            return MTPQPN8Head.apply(view, layer, x, bias)
        out = torch.ops.vllm.sm70_mtp_target_head_qpn8_probe(
            x.reshape(rows, 2560).contiguous(), view.codes, view.scales
        )
        if bias is not None:
            out.add_(bias)
        return out.reshape(*x.shape[:-1], head.weight.shape[0])

    view.apply = MethodType(apply, view)  # type: ignore[method-assign]
    owner = language if getattr(language, "lm_head", None) is head else target_model
    owner.lm_head = view
    return view
