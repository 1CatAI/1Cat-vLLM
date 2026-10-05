# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M1 shared-expert chain with ordinary projection fallbacks."""

import torch
import torch.nn.functional as F

from vllm.compilation.sm70_decode_graph import use_sm70_decode_graph_semantics
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op


def _shared_expert_chain(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    gate: torch.Tensor,
    packed_up: torch.Tensor,
    packed_down: torch.Tensor,
    projection: torch.Tensor,
    partial: torch.Tensor,
    flags: torch.Tensor,
    epochs: torch.Tensor,
    native_packed_up: torch.Tensor | None,
    exact_gate: bool,
    batch_gate: bool,
    batch_up: bool,
    decode: bool,
) -> torch.Tensor:
    from vllm import _sm70_ops as ops

    if (
        decode
        and x.ndim == 2
        and x.shape == (1, 2560)
        and x.dtype == torch.float16
        and x.is_cuda
        and x.is_contiguous()
        and x.data_ptr() % 16 == 0
        and not torch.backends.cuda.matmul.allow_fp16_accumulation
    ):
        out = torch.empty_like(x)
        torch.ops._C.qwen38_shared_expert_chain_sm70_out(
            out, x, packed_up, packed_down, gate, projection, partial, flags, epochs
        )
        return out

    # Keep the shape decision inside this opaque op. Prefill-first compilation
    # must not bake its large-M branch into subsequent single-token graphs.
    if batch_up and decode:
        activated = torch.ops.vllm.qwen38_sm70_shared_up(x, w13, native_packed_up)
    else:
        projected = F.linear(x, w13)
        activated = projected.new_empty((*projected.shape[:-1], 160))
        torch.ops._C.silu_and_mul(activated, projected)
    out = F.linear(activated, w2)
    shape_ok = (
        x.ndim == 2
        and 0 < x.shape[0] <= 16
        and x.is_contiguous()
        and out.is_contiguous()
        and x.dtype == out.dtype == torch.float16
    )
    if (
        exact_gate
        and x.shape[0] == 1
        and shape_ok
        and ops.has_qwen38_shared_gate_exact()
    ):
        ops.qwen38_shared_gate_exact_out(out, x, gate)
        return out
    logits = F.linear(x, gate)
    if (
        batch_gate
        and decode
        and x.shape[0] > 1
        and shape_ok
        and ops.has_qwen38_shared_gate_sigmoid_mul()
    ):
        ops.qwen38_shared_gate_sigmoid_mul_out(out, logits)
        return out
    if batch_up and decode:
        return torch.ops.vllm.qwen38_sm70_shared_gate_mul(logits, out)
    return torch.sigmoid(logits) * out


def _shared_expert_chain_fake(x: torch.Tensor, *args) -> torch.Tensor:
    return torch.empty_like(x)


direct_register_custom_op(
    op_name="qwen38_sm70_shared_expert_chain",
    op_func=_shared_expert_chain,
    fake_impl=_shared_expert_chain_fake,
    mutates_args=["projection", "partial", "flags", "epochs"],
)


def prepare_shared_expert_chains(module: torch.nn.Module, vllm_config) -> int:
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP

    from .sm70_fp16_gemv import _exact_runtime_contract

    if (
        not _exact_runtime_contract(vllm_config)
        or vllm_config.model_config.quantization != "modelopt_fp4"
        or vllm_config.parallel_config.use_ubatching
        or not current_platform.is_device_capability(70)
        or not hasattr(torch.ops._C, "qwen38_shared_expert_chain_sm70_out")
    ):
        return 0
    prepared = 0
    for child in module.modules():
        if not isinstance(child, Qwen2MoeMLP):
            continue
        if child.down_proj.reduce_results or any(
            layer.bias is not None
            for layer in (child.gate_up_proj, child.down_proj, child.expert_gate)
            if layer is not None
        ):
            continue
        if hasattr(child, "_sm70_qwen38_shared_chain"):
            continue
        up = getattr(getattr(child, "gate_up_proj", None), "weight", None)
        down = getattr(getattr(child, "down_proj", None), "weight", None)
        gate = getattr(getattr(child, "expert_gate", None), "weight", None)
        if (
            not isinstance(up, torch.Tensor)
            or not isinstance(down, torch.Tensor)
            or not isinstance(gate, torch.Tensor)
        ):
            continue
        if (
            up.shape != (320, 2560)
            or down.shape != (2560, 160)
            or gate.shape != (1, 2560)
        ):
            continue
        if not all(
            t.is_cuda and t.dtype == torch.float16 and t.is_contiguous()
            for t in (up, down, gate)
        ) or not (up.device == down.device == gate.device):
            continue
        packed_up = (
            torch.stack((up[:160], up[160:]), dim=1)
            .reshape(10, 32, 160, 2, 8)
            .permute(0, 2, 3, 1, 4)
            .contiguous()
        )
        packed_down = down.reshape(80, 32, 10, 2, 8).permute(0, 2, 3, 1, 4).contiguous()
        tensors = (
            packed_up,
            packed_down,
            torch.empty(40, 1, 32, dtype=torch.float32, device=up.device),
            torch.empty(10, 1, 2560, dtype=torch.float32, device=up.device),
            torch.zeros(80, dtype=torch.int32, device=up.device),
            torch.zeros(40, dtype=torch.int32, device=up.device),
        )
        for name, tensor in zip(
            ("up", "down", "projection", "partial", "flags", "epochs"), tensors
        ):
            child.register_buffer(
                f"_sm70_shared_chain_{name}", tensor, persistent=False
            )
        child._sm70_qwen38_shared_chain = True
        prepared += 1
    return prepared


def shared_expert_chain_forward(
    module: torch.nn.Module, x: torch.Tensor
) -> torch.Tensor:
    return torch.ops.vllm.qwen38_sm70_shared_expert_chain(
        x,
        module.gate_up_proj.weight,
        module.down_proj.weight,
        module.expert_gate.weight,
        module._sm70_shared_chain_up,
        module._sm70_shared_chain_down,
        module._sm70_shared_chain_projection,
        module._sm70_shared_chain_partial,
        module._sm70_shared_chain_flags,
        module._sm70_shared_chain_epochs,
        getattr(module.gate_up_proj, "_sm70_mtp_shared_packed", None),
        module._sm70_exact_shared_expert_gate,
        module._sm70_batch_shared_expert_gate,
        getattr(module.gate_up_proj, "_sm70_mtp_prepare_shared_batch", False),
        use_sm70_decode_graph_semantics(),
    )
