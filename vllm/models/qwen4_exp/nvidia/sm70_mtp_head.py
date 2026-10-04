# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Draft-only QPN8 view of the shared Flash-Next LM head.

The target retains its original head, method and checkpoint parameter. Small
draft batches read the channel-QPN8 pack; full logits and compact top1 use the
same view so numerical probes observe the actual candidate distribution.
"""

import torch
from torch import nn

import vllm.envs as envs
from vllm import _sm70_ops as ops
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.sm70_online_qpn8 import (
    prepare_channel_qpn8_weight,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)


class MTPQPN8Head(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head
        self.shard_indices = head.shard_indices
        # LogitsProcessor calls method.apply(view, hidden). This view owns no
        # replacement checkpoint parameter and never mutates the shared head.
        codes, scales = prepare_channel_qpn8_weight(head.weight)
        self.register_buffer("codes", codes, persistent=False)
        self.register_buffer("scales", scales, persistent=False)

    @property
    def quant_method(self):
        return self

    @property
    def weight(self):
        return self.head.weight

    def apply(self, layer, x, bias=None):
        rows = x.numel() // x.shape[-1]
        if x.dtype != torch.float16 or not 1 <= rows <= 8:
            return self.head.quant_method.apply(self.head, x, bias)
        x2 = x.reshape(rows, 2560).contiguous()
        out = x.new_empty(rows, self.weight.shape[0])
        ops.fp8_qpn8_gemm_sm70_out(out, x2, self.codes, self.scales, 8, 2, True, False)
        if bias is not None:
            out.add_(bias)
        return out.reshape(*x.shape[:-1], self.weight.shape[0])

    def maybe_get_sm70_lm_head_top1(self, hidden_states, bias=None):
        # Use the existing compact value/ID reduction on these QPN8 logits,
        # including the existing vocabulary padding mask and global ID offset.
        return None


def prepare_mtp_qpn8_head(head):
    if (
        envs.VLLM_BATCH_INVARIANT
        or not current_platform.is_cuda()
        or not current_platform.is_device_capability(70)
        or get_tensor_model_parallel_world_size() != 4
        or not head.weight.is_cuda
        or head.weight.dtype != torch.float16
        or tuple(head.weight.shape) != (62080, 2560)
        or any(
            not hasattr(torch.ops._C, name)
            for name in ("fp8_qpn8_prepare_sm70", "fp8_qpn8_gemm_sm70_out")
        )
    ):
        return None
    view = MTPQPN8Head(head)
    logger.info_once("SM70 Flash-Next draft-only channel-QPN8 head prepared (M1..8).")
    return view
