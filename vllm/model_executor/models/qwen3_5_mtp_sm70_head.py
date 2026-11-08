# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Draft-only hot-vocabulary QPN8 head for the shared Qwen3.5/3.8 LM head on SM70.

Greedy MTP drafts pick their token among the most frequent vocabulary rows
(FR-Spec). Each rank keeps the frequency-ranked ids of its own contiguous
vocabulary shard, padded to the largest shard so all ranks run one GEMM width,
and stores only those rows as a channel-QPN8 pack. Verification keeps the
full target head, so a draft outside the shortlist costs acceptance, not
correctness.
"""

from pathlib import Path

import torch
from torch import nn

import vllm.envs as envs
from vllm import _sm70_ops as ops
from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.sm70_online_qpn8 import (
    prepare_channel_qpn8_weight,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)

# The QPN8 GEMM admits 1..32 rows; MTP drafts one row per request.
_MAX_ROWS = 32
_DEFAULT_RANKING = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "sm70_mtp_dynamic_vocab_qwen36_27b_tp4.pt"
)


def load_hot_ranking(path: str) -> torch.Tensor:
    """Token ids, most frequent first: a .pt ranking asset, .npy, or one id per line."""
    if path.endswith(".pt"):
        value = torch.load(path, map_location="cpu", weights_only=True)
        ranking = value["global_ranking"] if isinstance(value, dict) else value
        return torch.as_tensor(ranking, dtype=torch.int64)
    if path.endswith(".npy"):
        import numpy as np

        return torch.from_numpy(np.load(path).astype("int64"))
    with open(path) as f:
        return torch.tensor([int(line) for line in f if line.strip()])


def shard_hot_ids(
    ranking: torch.Tensor, count: int, world: int, rank: int, rows_per_rank: int
) -> torch.Tensor:
    """This rank's share of the ``count`` top-ranked ids, sorted and padded to the
    largest shard's count rounded up to 32 by repeating its last id."""
    hot = ranking[:count].long()
    owner = hot // rows_per_rank
    largest = int(torch.bincount(owner, minlength=world).max())
    ids = hot[owner == rank].sort().values
    width = largest + (-largest) % 32
    if ids.numel() < width:
        fill = ids[-1:] if ids.numel() else ids.new_full((1,), rank * rows_per_rank)
        ids = torch.cat([ids, fill.expand(width - ids.numel())])
    return ids


class MTPHotQPN8Head(nn.Module):
    def __init__(self, head: nn.Module, ids: torch.Tensor):
        super().__init__()
        self.head = head
        self.shard_indices = head.shard_indices
        start = self.shard_indices.org_vocab_start_index
        ids = ids.to(head.weight.device)
        codes, scales = prepare_channel_qpn8_weight(
            head.weight[ids - start].contiguous()
        )
        self.register_buffer("codes", codes, persistent=False)
        self.register_buffer("scales", scales, persistent=False)
        self.register_buffer("hot_ids", ids.to(torch.int32), persistent=False)

    @property
    def quant_method(self):
        return self

    @property
    def weight(self):
        return self.head.weight

    def apply(self, layer, x, bias=None):
        return self.head.quant_method.apply(self.head, x, bias)

    def maybe_get_sm70_lm_head_top1_pair(self, hidden_states, bias=None):
        """Local (value, global id) of the best hot row, or None to use the full head."""
        if (
            bias is not None
            or hidden_states.ndim != 2
            or hidden_states.dtype != torch.float16
            or not 1 <= hidden_states.shape[0] <= _MAX_ROWS
        ):
            return None
        rows = hidden_states.shape[0]
        hot = self.hot_ids.numel()
        logits = hidden_states.new_empty(rows, hot)
        ops.fp8_qpn8_gemm_sm70_out(
            logits, hidden_states.contiguous(), self.codes, self.scales, 8, 2, True, False
        )
        pairs = logits.new_empty((rows, 2), dtype=torch.float32)
        partial = logits.new_empty((rows, (hot + 511) // 512, 2), dtype=torch.float32)
        torch.ops._C.qwen38_mtp_local_top1_sm70_out(pairs, partial, logits, hot, 0)
        pairs[:, 1] = self.hot_ids[pairs[:, 1].long()].float()
        return pairs

    def maybe_get_sm70_lm_head_top1(self, hidden_states, bias=None):
        return None


def prepare_mtp_hot_qpn8_head(
    head: nn.Module, hot: int, ranking_path: str = ""
) -> MTPHotQPN8Head | None:
    weight = getattr(head, "weight", None)
    shard = getattr(head, "shard_indices", None)
    missing = [
        name
        for name in (
            "fp8_qpn8_prepare_sm70",
            "fp8_qpn8_gemm_sm70_out",
            "qwen38_mtp_local_top1_sm70_out",
        )
        if not hasattr(torch.ops._C, name)
    ]
    reason = (
        "disabled"
        if hot <= 0
        else "no dense weight"
        if not isinstance(weight, torch.Tensor) or shard is None
        else "batch invariant"
        if envs.VLLM_BATCH_INVARIANT
        else "not SM70"
        if not current_platform.is_device_capability(70)
        else f"weight {weight.dtype} {tuple(weight.shape)} on {weight.device}"
        if not weight.is_cuda
        or weight.dtype != torch.float16
        or weight.ndim != 2
        or weight.shape[1] % 16
        else "added vocabulary"
        if getattr(shard, "num_added_elements", 0)
        else f"missing ops {missing}"
        if missing
        else None
    )
    if reason is not None:
        if hot > 0:
            logger.info_once("SM70 MTP hot draft head skipped: %s.", reason)
        return None
    group = get_tp_group()
    rows = shard.org_vocab_end_index - shard.org_vocab_start_index
    vocab = rows * group.world_size
    if rows != weight.shape[0] or getattr(shard, "num_org_vocab_padding", 0):
        logger.info_once("SM70 MTP hot draft head skipped: padded vocabulary shards.")
        return None
    path = ranking_path or str(_DEFAULT_RANKING)
    ranking = load_hot_ranking(path)
    ranking = ranking[(ranking >= 0) & (ranking < vocab)]
    count = min(hot, ranking.numel())
    ids = shard_hot_ids(ranking, count, group.world_size, group.rank_in_group, rows)
    view = MTPHotQPN8Head(head, ids)
    logger.info_once(
        "SM70 MTP hot-vocabulary QPN8 draft head: %d ranked ids, %d rows per rank.",
        count,
        ids.numel(),
    )
    return view
