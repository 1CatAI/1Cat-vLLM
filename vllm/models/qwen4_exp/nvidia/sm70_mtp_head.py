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
        self._shortlist_size = None

    def prepare_shortlist(self, token_ids):
        """Prepare an evaluated candidate; full-vocabulary probes stay intact.

        Original global IDs are retained. Only greedy draft decisions use this
        pack; target logits and stochastic correction retain the full head.
        Call before capture, never while a captured graph is serving requests.
        """
        ids = torch.as_tensor(token_ids, dtype=torch.int64, device=self.weight.device)
        if ids.ndim != 1 or not ids.numel():
            raise ValueError("Draft shortlist must be a nonempty vector")
        ids = ids.unique(sorted=True)
        start = self.shard_indices.org_vocab_start_index
        end = self.shard_indices.org_vocab_end_index
        local = ids[(ids >= start) & (ids < end)]
        self._shortlist_size = local.numel()
        self.register_buffer("shortlist_ids", local, persistent=False)
        if self._shortlist_size == 0:
            return
        selected = self.weight.index_select(0, local - start)
        padded = (self._shortlist_size + 31) // 32 * 32
        if padded != self._shortlist_size:
            selected = torch.nn.functional.pad(
                selected, (0, 0, 0, padded - self._shortlist_size)
            )
        codes, scales = prepare_channel_qpn8_weight(selected)
        self.register_buffer("shortlist_codes", codes, persistent=False)
        self.register_buffer("shortlist_scales", scales, persistent=False)
        self._shortlist_padded = padded

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
        rows = hidden_states.numel() // hidden_states.shape[-1]
        if (
            self._shortlist_size is not None
            and hidden_states.dtype == torch.float16
            and 1 <= rows <= 8
        ):
            shape = hidden_states.shape[:-1]
            if self._shortlist_size == 0:
                values = hidden_states.new_full(shape, -float("inf"))
                ids = torch.full(
                    shape,
                    self.shard_indices.org_vocab_start_index,
                    dtype=torch.int64,
                    device=hidden_states.device,
                )
                return values, ids
            logits = hidden_states.new_empty(rows, self._shortlist_padded)
            ops.fp8_qpn8_gemm_sm70_out(
                logits,
                hidden_states.reshape(rows, 2560).contiguous(),
                self.shortlist_codes,
                self.shortlist_scales,
                8,
                2,
                True,
                False,
            )
            valid = logits[:, : self._shortlist_size]
            if bias is not None:
                indices = self.shortlist_ids - self.shard_indices.org_vocab_start_index
                valid = valid + bias.index_select(0, indices)
            values, indices = valid.max(dim=-1)
            return values.reshape(shape), self.shortlist_ids[indices].reshape(shape)
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
