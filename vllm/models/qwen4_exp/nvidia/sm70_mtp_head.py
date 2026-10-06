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
        codes, scales = prepare_channel_qpn8_weight(head.weight)
        self.register_buffer("codes", codes, persistent=False)
        self.register_buffer("scales", scales, persistent=False)
        self.hot_ids: torch.Tensor | None = None

    def prepare_hot_vocabulary(self, count: int) -> None:
        """FR-Spec: keep a balanced share of the ``count`` hottest rows."""
        import torch.distributed as dist

        from vllm.distributed import get_tp_group

        group = get_tp_group()
        world, rank = group.world_size, group.rank_in_group
        local = self.head.weight
        rows = local.shape[0]
        vocab = self.shard_indices.org_vocab_end_index - (
            self.shard_indices.org_vocab_start_index
        )
        if rows * world != 248320 or vocab != rows:
            logger.info_once("SM70 draft hot vocabulary skipped: padded shards.")
            return
        count = min(count, rows * world)
        gathered = [torch.empty_like(local) for _ in range(world)]
        dist.all_gather(gathered, local.contiguous(), group=group.device_group)
        full = torch.cat(gathered)
        del gathered
        ids = torch.arange(rank, count, world, device=local.device)
        pad = (-ids.numel()) % 32
        if pad:
            ids = torch.cat([ids, ids[-1:].expand(pad)])
        codes, scales = prepare_channel_qpn8_weight(full[ids].contiguous())
        del full
        self.register_buffer("hot_codes", codes, persistent=False)
        self.register_buffer("hot_scales", scales, persistent=False)
        self.hot_ids = ids.to(torch.int32)
        logger.info_once(
            "SM70 draft hot vocabulary: %d ids, %d rows per rank.",
            count,
            ids.numel(),
        )

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

    def maybe_get_sm70_lm_head_top1_pair(self, hidden_states, bias=None):
        """Emit the existing compact packet without full-row serial selection."""
        if (
            hidden_states.ndim != 2
            or hidden_states.dtype != torch.float16
            or not hidden_states.is_cuda
            or not 1 <= hidden_states.shape[0] <= 8
            or getattr(self.shard_indices, "num_added_elements", 0)
            or not hasattr(torch.ops._C, "qwen38_mtp_local_top1_sm70_out")
        ):
            return None
        start = self.shard_indices.org_vocab_start_index
        valid = self.shard_indices.org_vocab_end_index - start
        if valid < 1:
            return None
        rows = hidden_states.shape[0]
        if self.hot_ids is not None and bias is None:
            hot = self.hot_ids.numel()
            logits = hidden_states.new_empty(rows, hot)
            ops.fp8_qpn8_gemm_sm70_out(
                logits,
                hidden_states.contiguous(),
                self.hot_codes,
                self.hot_scales,
                8,
                2,
                True,
                False,
            )
            pairs = logits.new_empty((rows, 2), dtype=torch.float32)
            partial = logits.new_empty((rows, (hot + 511) // 512, 2), dtype=torch.float32)
            torch.ops._C.qwen38_mtp_local_top1_sm70_out(pairs, partial, logits, hot, 0)
            local = pairs[:, 1].long()
            pairs[:, 1] = self.hot_ids[local].float()
            return pairs
        logits = self.apply(self, hidden_states, bias)
        pairs = logits.new_empty((rows, 2), dtype=torch.float32)
        partial = logits.new_empty((rows, (valid + 511) // 512, 2), dtype=torch.float32)
        torch.ops._C.qwen38_mtp_local_top1_sm70_out(
            pairs, partial, logits, valid, start
        )
        return pairs

    def maybe_get_sm70_lm_head_top1(self, hidden_states, bias=None):
        # Use the existing compact value/ID reduction on these QPN8 logits,
        # including the existing vocabulary padding mask and global ID offset.
        return None


def prepare_mtp_qpn8_head(head, hot: int = 0):
    weight = getattr(head, "weight", None)
    if not isinstance(weight, torch.Tensor):
        return None
    if (
        envs.VLLM_BATCH_INVARIANT
        or not current_platform.is_cuda()
        or not current_platform.is_device_capability(70)
        or get_tensor_model_parallel_world_size() != 4
        or not weight.is_cuda
        or weight.dtype != torch.float16
        or tuple(weight.shape) != (62080, 2560)
        or any(
            not hasattr(torch.ops._C, name)
            for name in ("fp8_qpn8_prepare_sm70", "fp8_qpn8_gemm_sm70_out")
        )
    ):
        return None
    view = MTPQPN8Head(head)
    logger.info_once("SM70 Flash-Next draft-only channel-QPN8 head prepared (M1..8).")
    if hot > 0:
        view.prepare_hot_vocabulary(hot)
    return view
