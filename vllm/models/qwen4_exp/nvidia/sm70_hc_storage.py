# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lossless TP4 HC storage shared by decode and prefill.

Storage and large-batch fallback are reused from the constrained-memory GGUF
implementation. The existing LL kernels remain the small-batch implementation.
"""

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

from .sm70_fp16_hc import _pack_hc_batch_weight

logger = init_logger(__name__)


def _unpack_hc_storage(down: torch.Tensor, up: torch.Tensor):
    """Recover local checkpoint rows, never the full replicated matrices."""
    return (
        down.permute(0, 3, 1, 2, 4).contiguous().reshape(96, 10240)[:88],
        up.permute(3, 0, 4, 1, 2, 5).contiguous().reshape(2560, 320),
    )


def _sharded_hc_project(
    x: torch.Tensor, down: torch.Tensor, up: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.distributed.parallel_state import get_tp_group

    from .ops.hc import hc_gate_mix, hc_silu

    group = get_tp_group()
    communicator = getattr(group.device_communicator, "hc_ll_comm", None)
    assert communicator is not None and communicator.status["enabled"]
    result = communicator.apply(x, down, up)
    if result is not None:
        return result
    # Keep large-M prefill usable without retaining replicated checkpoint banks.
    # The LL topology can reorder ranks; gathered rows must follow logical order.
    local_down, local_up = _unpack_hc_storage(down, up)
    local = torch.mm(x, local_down.T)
    packet = group.all_gather(local, dim=-1).reshape(x.shape[0], 4, 88)
    packet = packet[:, list(communicator.order)]
    lora = hc_silu(packet[:, :, :80].reshape(x.shape[0], 320), 4)
    injection = packet[:, 3, 80:84].contiguous()
    local_gate = torch.mm(lora, local_up.T)
    gate = group.all_gather(local_gate, dim=-1).reshape(x.shape[0], 4, 4, 640)
    gate = gate[:, list(communicator.order)]
    gate = gate.permute(0, 2, 1, 3).reshape(x.shape[0], 10240)
    return hc_gate_mix(x, gate, 4), injection


def _sharded_hc_project_fake(
    x: torch.Tensor, down: torch.Tensor, up: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return x.new_empty((x.shape[0], 2560)), x.new_empty((x.shape[0], 4))


direct_register_custom_op(
    op_name="qwen38_sharded_hc_project",
    op_func=_sharded_hc_project,
    fake_impl=_sharded_hc_project_fake,
)
sharded_hc_project = torch.ops.vllm.qwen38_sharded_hc_project


def prepare_sharded_hc_storage(model, vllm_config) -> None:
    """Replace replicated FP16 HC matrices with the existing lossless LL pack."""
    if vllm_config.kernel_config.hc_weight_storage != "sharded":
        return
    from vllm.distributed.parallel_state import get_tp_group

    from .hyperconnection import GatedResidual

    group = get_tp_group()
    communicator = getattr(group.device_communicator, "hc_ll_comm", None)
    if communicator is None or not communicator.status["enabled"]:
        reason = (
            "communicator_unavailable"
            if communicator is None
            else communicator.status["reason"]
        )
        raise ValueError(
            f"Sharded HC storage requires qualified TP4 transport: {reason}"
        )
    if vllm_config.lora_config is not None:
        raise ValueError("Sharded HC storage does not support LoRA adapters")
    count = 0
    for module in model.modules():
        if not isinstance(module, GatedResidual) or hasattr(module, "hc_shard_down"):
            continue
        if (module.hc_count, module.hidden_size, module.lora_rank) != (4, 2560, 320):
            raise ValueError("Sharded HC storage requires HC4, hidden 2560, rank 320")
        down_layer = (
            module.input_mix_weight_down_block_inject
            if module.use_combine
            else module.input_mix_weight_down
        )
        up_layer = module.input_mix_weight_up
        down = down_layer.weight
        up = up_layer.weight
        staged = all(
            getattr(layer, "_hc_cpu_staging", False) for layer in (down_layer, up_layer)
        )
        if (
            (not staged and not down.is_cuda)
            or down.dtype != torch.float16
            or up.dtype != torch.float16
        ):
            raise ValueError("Sharded HC storage requires FP16 checkpoint matrices")
        if not module.use_combine:
            down = torch.nn.functional.pad(down, (0, 0, 0, 16))
        rank = communicator.logical_rank
        module.register_buffer(
            "hc_shard_down",
            _pack_hc_batch_weight(down, "down", rank).to(communicator.device),
            persistent=False,
        )
        module.register_buffer(
            "hc_shard_up",
            _pack_hc_batch_weight(up, "up", rank).to(communicator.device),
            persistent=False,
        )
        for layer in (down_layer, up_layer):
            layer.register_parameter(
                "weight",
                nn.Parameter(layer.weight.new_empty((0,)), requires_grad=False),
            )
            if hasattr(layer, "_sm70_qwen38_hc_batch_packed"):
                delattr(layer, "_sm70_qwen38_hc_batch_packed")
        count += 1
    logger.info(
        "Prepared %d lossless HC shards; replicated projection banks released", count
    )
