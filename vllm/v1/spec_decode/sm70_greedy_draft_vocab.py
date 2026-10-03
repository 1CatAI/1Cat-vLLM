# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Greedy proposals using the existing bounded dynamic draft vocabulary."""

from pathlib import Path

import torch
import torch.nn.functional as F

from vllm.distributed import tensor_model_parallel_all_gather
from vllm.v1.spec_decode.static_draft_vocab import (
    DynamicDraftVocabRuntime,
    initialize_dynamic_draft_vocab,
    prepare_dynamic_draft_vocab_prefill_candidates,
)


def select_local_pair(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Break shortlist ties using original token IDs, independent of row order."""
    values = logits.max(dim=-1).values
    ids = (
        torch.where(logits == values.unsqueeze(-1), token_ids, 2**24 - 1)
        .min(dim=-1)
        .values
    )
    return torch.stack((values.float(), ids.float()), dim=-1)


class Sm70GreedyDraftVocab:
    def __init__(self, head, device, *, tp_size: int, size: int):
        ranking = (
            Path(__file__).resolve().parents[2]
            / "assets"
            / f"sm70_mtp_dynamic_vocab_qwen36_27b_tp{tp_size}.pt"
        )
        self.runtime: DynamicDraftVocabRuntime = initialize_dynamic_draft_vocab(
            head, ranking, size, 512, 0, True, 4, device, gpu_lru_enabled=True
        )
        assert self.runtime.tp_group is not None
        self._base_start = self.runtime.tp_group.rank_in_group * (
            self.runtime.lm_head.num_embeddings_per_partition
        )
        self._base_end = (
            self._base_start + self.runtime.lm_head.num_embeddings_per_partition
        )

    def observe_target_logits(self, logits: torch.Tensor, *, prefill: bool) -> None:
        # Bootstrap rare IDs once per prefill; subsequently keep a small rolling
        # set of target predictions. All selection and row refresh stay on GPU.
        candidates = prepare_dynamic_draft_vocab_prefill_candidates(
            logits, min(2048 if prefill else 20, logits.shape[-1])
        )
        self.runtime.update(candidates)

    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        runtime = self.runtime
        base = runtime.lm_head.quant_method.apply(runtime.lm_head, hidden_states)
        tail = F.linear(hidden_states, runtime.local_tail_weight)
        tail.masked_fill_(runtime.local_tail_token_ids.unsqueeze(0) < 0, -torch.inf)
        base_ids = runtime.token_id_map[self._base_start : self._base_end]
        pair = select_local_pair(
            torch.cat((base, tail), dim=-1),
            torch.cat((base_ids, runtime.local_tail_token_ids)),
        )
        # Reuse the same admitted IPC reduction as the normal full head.
        custom = runtime.logits_processor._maybe_custom_top1_argmax(pair)
        if custom is not None:
            return custom
        # Same compact value/ID transport as full-vocabulary local argmax.
        gathered = tensor_model_parallel_all_gather(pair, dim=-1).reshape(
            hidden_states.shape[0], -1, 2
        )
        return (
            select_local_pair(gathered[:, :, 0], gathered[:, :, 1].long())
            .select(-1, 1)
            .long()
        )
