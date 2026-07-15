# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static reduced-vocabulary helpers for speculative draft LM-heads."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

import vllm.envs as envs

if TYPE_CHECKING:
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead


def _get_sm70_ops() -> Any:
    from vllm import _sm70_ops

    return _sm70_ops


@dataclass(frozen=True)
class StaticDraftVocabTPPlan:
    source_token_ids: tuple[tuple[int, ...], ...]
    destination_token_ids: tuple[tuple[int, ...], ...]
    split_matrix: tuple[tuple[int, ...], ...]
    gathered_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class MTPDraftVocabConfig:
    ranking_path: str | None
    shortlist_size: int
    dynamic_tail_size: int
    full_refresh_interval: int
    fused_proposal_enabled: bool
    gpu_lru_enabled: bool
    prefill_topk: int
    using_defaults: bool = False


def resolve_mtp_draft_vocab_config(
    method: str,
    tensor_parallel_size: int = 2,
) -> MTPDraftVocabConfig:
    """Resolve explicit vocabulary controls or the default MTP GPU-LRU route."""
    config = MTPDraftVocabConfig(
        ranking_path=envs.VLLM_SM70_MTP_STATIC_DRAFT_VOCAB_RANKING,
        shortlist_size=envs.VLLM_SM70_MTP_STATIC_DRAFT_VOCAB_SIZE,
        dynamic_tail_size=envs.VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_TAIL_SIZE,
        full_refresh_interval=(
            envs.VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_FULL_REFRESH_INTERVAL
        ),
        fused_proposal_enabled=(envs.VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_FUSED_PROPOSAL),
        gpu_lru_enabled=envs.VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_GPU_LRU,
        prefill_topk=envs.VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_PREFILL_TOPK,
    )
    explicit_config = any(
        (
            config.ranking_path is not None,
            config.shortlist_size != 0,
            config.dynamic_tail_size != 0,
            config.full_refresh_interval != 0,
            config.fused_proposal_enabled,
            config.gpu_lru_enabled,
            config.prefill_topk != 0,
        )
    )
    if (
        method != "mtp"
        or not envs.VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_DEFAULT
        or explicit_config
    ):
        return config

    ranking_path = (
        Path(__file__).resolve().parents[2]
        / "assets"
        / f"sm70_mtp_dynamic_vocab_qwen36_27b_tp{tensor_parallel_size}.pt"
    )
    if not ranking_path.is_file():
        raise FileNotFoundError(
            f"Default MTP dynamic-vocabulary ranking asset is missing: {ranking_path}"
        )
    return MTPDraftVocabConfig(
        ranking_path=str(ranking_path),
        shortlist_size=98_304,
        dynamic_tail_size=512,
        full_refresh_interval=0,
        fused_proposal_enabled=True,
        gpu_lru_enabled=True,
        prefill_topk=2048,
        using_defaults=True,
    )


def validate_dynamic_draft_vocab_prefill_topk(
    topk: int,
    *,
    gpu_lru_enabled: bool,
    full_vocab_size: int,
) -> None:
    if topk < 0:
        raise ValueError("Dynamic draft vocabulary prefill top-k cannot be negative.")
    if topk == 0:
        return

    max_topk = min(4096, full_vocab_size)
    if topk > max_topk:
        raise ValueError(
            "Dynamic draft vocabulary prefill top-k must be in "
            f"[1, {max_topk}], got {topk}."
        )
    if not gpu_lru_enabled:
        raise ValueError(
            "Dynamic draft vocabulary prefill top-k requires the GPU LRU route."
        )


def prepare_dynamic_draft_vocab_prefill_candidates(
    target_logits: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    if target_logits.ndim != 2:
        raise ValueError("Dynamic prefill bootstrap logits must be two-dimensional.")
    if topk <= 0 or topk > target_logits.shape[-1]:
        raise ValueError("Dynamic prefill bootstrap top-k exceeds the logits width.")

    ranked = torch.topk(target_logits, k=topk, dim=-1)
    candidate_ids = ranked.indices.masked_fill(
        ~torch.isfinite(ranked.values),
        -1,
    )
    return candidate_ids.flip(-1).contiguous()


@dataclass
class DynamicDraftVocabPrefillBootstrapState:
    consumed_request_id: str | None = None

    def maybe_prepare_candidates(
        self,
        request_id: str,
        target_logits: torch.Tensor | None,
        *,
        topk: int,
        num_computed_tokens: int,
        num_scheduled_tokens: int,
        num_prompt_tokens: int,
        spec_decode_active: bool,
    ) -> torch.Tensor | None:
        is_final_prefill = (
            num_computed_tokens
            < num_prompt_tokens
            <= num_computed_tokens + num_scheduled_tokens
        )
        if (
            target_logits is None
            or spec_decode_active
            or not is_final_prefill
            or self.consumed_request_id == request_id
        ):
            return None
        return prepare_dynamic_draft_vocab_prefill_candidates(target_logits, topk)

    def mark_consumed(self, request_id: str) -> None:
        self.consumed_request_id = request_id

    def clear_finished_requests(self, finished_request_ids: set[str]) -> None:
        if self.consumed_request_id in finished_request_ids:
            self.consumed_request_id = None


def update_dynamic_draft_vocab_lru_state(
    tail_lru: OrderedDict[int, None],
    observed_output_ids: torch.Tensor | None,
    target_candidate_ids: torch.Tensor,
    *,
    full_vocab_size: int,
    base_token_ids: frozenset[int],
    tail_size: int,
    local_shard_start: int | None = None,
    local_shard_end: int | None = None,
) -> bool:
    if (local_shard_start is None) != (local_shard_end is None):
        raise ValueError("Dynamic draft vocabulary shard bounds must be paired.")
    local_shard = None
    if local_shard_start is not None and local_shard_end is not None:
        if not 0 <= local_shard_start < local_shard_end <= full_vocab_size:
            raise ValueError("Dynamic draft vocabulary shard bounds are invalid.")
        local_shard = (local_shard_start, local_shard_end)

    old_ids = tuple(tail_lru)
    for tensor in (observed_output_ids, target_candidate_ids):
        if tensor is None:
            continue
        for token_id in tensor.detach().reshape(-1).cpu().tolist():
            token_id = int(token_id)
            if (
                token_id < 0
                or token_id >= full_vocab_size
                or token_id in base_token_ids
                or (
                    local_shard is not None
                    and not local_shard[0] <= token_id < local_shard[1]
                )
            ):
                continue
            tail_lru.pop(token_id, None)
            tail_lru[token_id] = None
            if len(tail_lru) > tail_size:
                tail_lru.popitem(last=False)
    return tuple(tail_lru) != old_ids


def select_dynamic_draft_vocab_shard_seed(
    ranked_tail_token_ids: Sequence[int],
    *,
    base_token_ids: frozenset[int],
    local_shard_start: int,
    local_shard_end: int,
    tail_size: int,
) -> tuple[int, ...]:
    """Select one rank's highest-ranked non-base tail tokens."""
    if tail_size <= 0:
        raise ValueError("Dynamic draft vocabulary tail size must be positive.")
    if local_shard_start < 0 or local_shard_end <= local_shard_start:
        raise ValueError("Dynamic draft vocabulary shard bounds are invalid.")

    selected: list[int] = []
    selected_set: set[int] = set()
    for value in ranked_tail_token_ids:
        token_id = int(value)
        if (
            token_id in base_token_ids
            or token_id in selected_set
            or token_id < local_shard_start
            or token_id >= local_shard_end
        ):
            continue
        selected.append(token_id)
        selected_set.add(token_id)
        if len(selected) == tail_size:
            return tuple(selected)

    raise ValueError(
        "Dynamic GPU LRU ranking cannot fill the shard-local tail: "
        f"range=({local_shard_start}, {local_shard_end}) "
        f"required={tail_size} available={len(selected)}."
    )


def compact_dynamic_draft_vocab_tail_state(
    tail_token_ids: Sequence[int],
    local_shard_start: int,
    local_shard_end: int,
    local_tail_capacity: int,
) -> tuple[list[int], list[int]]:
    """Keep local tail rows in global LRU order for the host fallback."""
    local_ids = [
        token_id
        for token_id in tail_token_ids
        if local_shard_start <= token_id < local_shard_end
    ]
    if len(local_ids) > local_tail_capacity:
        raise RuntimeError("Dynamic draft vocabulary exceeded local tail capacity.")
    return local_ids, [token_id - local_shard_start for token_id in local_ids]


@dataclass
class StaticDraftVocabRuntime:
    lm_head: ParallelLMHead
    logits_processor: LogitsProcessor
    token_id_map: torch.Tensor
    full_vocab_size: int
    shortlist_size: int
    ranking_fingerprint: str | None


@dataclass
class DynamicDraftVocabRuntime(StaticDraftVocabRuntime):
    base_size: int
    tail_size: int
    local_tail_capacity: int
    base_token_ids: frozenset[int]
    tail_lru: OrderedDict[int, None]
    source_weight: torch.Tensor
    local_shard_start: int
    local_shard_end: int
    local_tail_weight: torch.Tensor
    local_tail_token_ids: torch.Tensor
    gathered_tail_token_ids: torch.Tensor
    tail_active_mask: torch.Tensor
    tp_device_group: Any
    full_refresh_interval: int = 0
    proposal_count: int = 0
    full_head_active: bool = False
    full_head_candidate_ids: list[torch.Tensor] = field(default_factory=list)
    fused_proposal_enabled: bool = False
    num_speculative_tokens: int = 0
    tp_group: Any | None = None
    prepared_base_weight: torch.Tensor | None = None
    prepared_base_k_ld: int = 0
    local_base_start: int = 0
    fused_base_values: torch.Tensor | None = None
    fused_base_ids: torch.Tensor | None = None
    fused_tail_logits: torch.Tensor | None = None
    fused_local_pairs: torch.Tensor | None = None
    fused_gathered_pairs: torch.Tensor | None = None
    fused_sampled_tokens: torch.Tensor | None = None
    fused_sparse_ids: torch.Tensor | None = None
    fused_sparse_probs: torch.Tensor | None = None
    fused_exponentials: torch.Tensor | None = None
    fused_dense_probs: torch.Tensor | None = None
    gpu_lru_enabled: bool = False
    gpu_lru_token_ids: torch.Tensor | None = None
    gpu_base_token_mask: torch.Tensor | None = None
    local_tail_source_row_indices: torch.Tensor | None = None
    gpu_lru_empty_observed_ids: torch.Tensor | None = None
    gpu_lru_empty_candidate_ids: torch.Tensor | None = None

    def begin_proposal(self) -> None:
        self.full_head_active = (
            self.full_refresh_interval > 0
            and self.proposal_count % self.full_refresh_interval == 0
        )
        self.proposal_count += 1
        self.full_head_candidate_ids.clear()

    def observe_full_logits(self, logits: torch.Tensor) -> None:
        if self.full_head_active:
            k = min(20, logits.shape[-1])
            self.full_head_candidate_ids.append(torch.topk(logits, k=k, dim=-1).indices)

    def end_proposal(self) -> None:
        if self.full_head_active and self.full_head_candidate_ids:
            self.update(torch.cat(self.full_head_candidate_ids, dim=-1))
        self.full_head_candidate_ids.clear()
        self.full_head_active = False

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from vllm.distributed.communication_op import tensor_model_parallel_all_gather

        base_logits = self.logits_processor(self.lm_head, hidden_states)
        if base_logits is None:
            raise RuntimeError("Dynamic draft vocabulary requires all-gathered logits.")
        local_tail_logits = F.linear(hidden_states, self.local_tail_weight)
        tail_logits = tensor_model_parallel_all_gather(local_tail_logits, dim=-1)
        tail_logits.masked_fill_(
            ~self.tail_active_mask.view(1, -1),
            -torch.inf,
        )
        return torch.cat((base_logits, tail_logits), dim=-1)

    def fused_sample(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int,
        top_p: float,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.fused_proposal_enabled:
            raise RuntimeError("Dynamic fused proposal is not enabled.")
        if hidden_states.shape != (1, self.lm_head.embedding_dim):
            raise ValueError("Dynamic fused proposal requires one hidden-state row.")
        if not 0 <= spec_step_idx < self.num_speculative_tokens:
            raise ValueError("Dynamic fused proposal step is out of range.")
        tensors = (
            self.prepared_base_weight,
            self.fused_base_values,
            self.fused_base_ids,
            self.fused_tail_logits,
            self.fused_local_pairs,
            self.fused_gathered_pairs,
            self.fused_sampled_tokens,
            self.fused_sparse_ids,
            self.fused_sparse_probs,
            self.fused_exponentials,
            self.fused_dense_probs,
        )
        if self.tp_group is None or any(tensor is None for tensor in tensors):
            raise RuntimeError("Dynamic fused proposal buffers are not initialized.")

        sm70_ops = _get_sm70_ops()
        sm70_ops.sm70_f16_lm_head_top20_tc_out(
            self.fused_base_values,
            self.fused_base_ids,
            hidden_states,
            self.prepared_base_weight,
            self.prepared_base_k_ld,
            self.local_base_start,
            0,
        )
        torch.mm(
            hidden_states,
            self.local_tail_weight.t(),
            out=self.fused_tail_logits,
        )
        sm70_ops.sm70_merge_tail_top20_pack_out(
            self.fused_local_pairs,
            self.fused_base_values,
            self.fused_base_ids,
            self.token_id_map[: self.base_size],
            self.fused_tail_logits,
            self.local_tail_token_ids,
            self.base_size + self.tp_group.rank_in_group * self.tail_size,
        )
        torch.distributed.all_gather_into_tensor(
            self.fused_gathered_pairs,
            self.fused_local_pairs,
            group=self.tp_device_group,
        )

        exponential = self.fused_exponentials[spec_step_idx]
        exponential.exponential_(generator=generator)
        sampled_token = self.fused_sampled_tokens[spec_step_idx : spec_step_idx + 1]
        sparse_ids = self.fused_sparse_ids[spec_step_idx]
        sparse_probs = self.fused_sparse_probs[spec_step_idx]
        sm70_ops.sm70_sample_packed_top20_out(
            sampled_token,
            sparse_ids,
            sparse_probs,
            self.fused_gathered_pairs,
            exponential,
            top_p,
        )
        self.tp_group.broadcast(sampled_token, src=0)

        dense_probs = self.fused_dense_probs[spec_step_idx : spec_step_idx + 1]
        dense_probs.zero_()
        dense_probs.scatter_(1, sparse_ids.view(1, -1), sparse_probs.view(1, -1))
        return sampled_token, dense_probs

    def update(
        self,
        target_candidate_ids: torch.Tensor,
        observed_output_ids: torch.Tensor | None = None,
    ) -> None:
        if self.gpu_lru_enabled:
            self._refresh_gpu_tail(
                (
                    self.gpu_lru_empty_observed_ids
                    if observed_output_ids is None
                    else observed_output_ids.reshape(-1)
                ),
                target_candidate_ids.reshape(-1),
            )
            return

        if update_dynamic_draft_vocab_lru_state(
            self.tail_lru,
            observed_output_ids,
            target_candidate_ids,
            full_vocab_size=self.full_vocab_size,
            base_token_ids=self.base_token_ids,
            tail_size=self.tail_size,
        ):
            self._refresh_tail()

    def _refresh_gpu_tail(
        self,
        observed_output_ids: torch.Tensor | None,
        target_candidate_ids: torch.Tensor | None,
    ) -> None:
        tensors = (
            self.gpu_lru_token_ids,
            self.gpu_base_token_mask,
            self.local_tail_source_row_indices,
            self.gpu_lru_empty_observed_ids,
            self.gpu_lru_empty_candidate_ids,
        )
        if not self.gpu_lru_enabled or any(tensor is None for tensor in tensors):
            raise RuntimeError("Dynamic GPU LRU buffers are not initialized.")

        assert self.gpu_lru_token_ids is not None
        assert self.gpu_base_token_mask is not None
        assert self.local_tail_source_row_indices is not None
        assert self.gpu_lru_empty_observed_ids is not None
        assert self.gpu_lru_empty_candidate_ids is not None
        gpu_lru_token_ids = self.gpu_lru_token_ids
        gpu_base_token_mask = self.gpu_base_token_mask
        local_tail_source_row_indices = self.local_tail_source_row_indices
        empty_observed_ids = self.gpu_lru_empty_observed_ids
        empty_candidate_ids = self.gpu_lru_empty_candidate_ids
        sm70_ops = _get_sm70_ops()
        sm70_ops.sm70_dynamic_draft_vocab_update_tail_out(
            gpu_lru_token_ids,
            self.local_tail_token_ids,
            local_tail_source_row_indices,
            empty_observed_ids if observed_output_ids is None else observed_output_ids,
            empty_candidate_ids
            if target_candidate_ids is None
            else target_candidate_ids,
            gpu_base_token_mask,
            self.full_vocab_size,
            self.local_shard_start,
            self.local_shard_end,
        )
        sm70_ops.sm70_dynamic_draft_vocab_refresh_tail_weight_out(
            self.local_tail_weight,
            self.source_weight,
            local_tail_source_row_indices,
        )
        torch.distributed.all_gather_into_tensor(
            self.gathered_tail_token_ids,
            self.local_tail_token_ids,
            group=self.tp_device_group,
        )
        torch.ge(
            self.gathered_tail_token_ids,
            0,
            out=self.tail_active_mask,
        )
        torch.clamp(
            self.gathered_tail_token_ids,
            min=0,
            out=self.token_id_map[self.base_size :],
        )

    def _refresh_tail(self) -> None:
        local_ids, local_source_indices = compact_dynamic_draft_vocab_tail_state(
            tuple(self.tail_lru),
            self.local_shard_start,
            self.local_shard_end,
            self.local_tail_capacity,
        )

        self.local_tail_token_ids.fill_(-1)
        self.local_tail_weight.zero_()
        if local_ids:
            local_token_ids = torch.tensor(
                local_ids,
                dtype=torch.int64,
                device=self.local_tail_token_ids.device,
            )
            count = local_token_ids.numel()
            self.local_tail_token_ids[:count].copy_(local_token_ids)
            local_indices = torch.tensor(
                local_source_indices,
                dtype=torch.int64,
                device=self.local_tail_token_ids.device,
            )
            torch.index_select(
                self.source_weight,
                0,
                local_indices,
                out=self.local_tail_weight[:count],
            )

        torch.distributed.all_gather_into_tensor(
            self.gathered_tail_token_ids,
            self.local_tail_token_ids,
            group=self.tp_device_group,
        )
        active = self.gathered_tail_token_ids >= 0
        self.tail_active_mask.copy_(active)
        self.token_id_map[self.base_size :].copy_(
            self.gathered_tail_token_ids.clamp_min(0)
        )


def build_static_draft_vocab_tp_plan(
    global_ranking: Sequence[int],
    shard_ranges: Sequence[tuple[int, int]],
    shortlist_size: int,
) -> StaticDraftVocabTPPlan:
    world_size = len(shard_ranges)
    if world_size <= 0:
        raise ValueError("At least one vocabulary shard is required.")
    if shortlist_size <= 0 or shortlist_size > len(global_ranking):
        raise ValueError("Shortlist size must fit in the global ranking.")

    selected = tuple(int(token_id) for token_id in global_ranking[:shortlist_size])
    if len(set(selected)) != shortlist_size:
        raise ValueError("The selected global ranking contains duplicate token IDs.")

    for rank, (start, end) in enumerate(shard_ranges):
        if start < 0 or end <= start:
            raise ValueError(
                f"Invalid vocabulary range for rank {rank}: {(start, end)}"
            )
        if rank and start != shard_ranges[rank - 1][1]:
            raise ValueError("Vocabulary shard ranges must be contiguous.")

    source_buckets: list[list[int]] = [[] for _ in range(world_size)]
    for token_id in selected:
        source = next(
            (
                rank
                for rank, (start, end) in enumerate(shard_ranges)
                if start <= token_id < end
            ),
            None,
        )
        if source is None:
            raise ValueError(f"Token ID {token_id} is outside all vocabulary shards.")
        source_buckets[source].append(token_id)

    destination_sizes = [
        shortlist_size // world_size + (rank < shortlist_size % world_size)
        for rank in range(world_size)
    ]
    source_offsets = [0] * world_size
    split_matrix = [[0] * world_size for _ in range(world_size)]
    destination_buckets: list[list[int]] = [[] for _ in range(world_size)]
    for destination, destination_size in enumerate(destination_sizes):
        needed = destination_size
        for source in range(world_size):
            available = len(source_buckets[source]) - source_offsets[source]
            take = min(needed, available)
            if take:
                begin = source_offsets[source]
                end = begin + take
                destination_buckets[destination].extend(
                    source_buckets[source][begin:end]
                )
                source_offsets[source] = end
                split_matrix[source][destination] = take
                needed -= take
        if needed:
            raise ValueError("Selected rows cannot fill balanced TP destinations.")
    if source_offsets != [len(bucket) for bucket in source_buckets]:
        raise ValueError("Selected rows remain after TP destination assignment.")

    source_ordered = []
    for source in range(world_size):
        offset = 0
        ordered: list[int] = []
        for destination in range(world_size):
            count = split_matrix[source][destination]
            ordered.extend(source_buckets[source][offset : offset + count])
            offset += count
        source_ordered.append(tuple(ordered))

    gathered = tuple(token_id for bucket in destination_buckets for token_id in bucket)
    if len(gathered) != shortlist_size or set(gathered) != set(selected):
        raise AssertionError("TP plan changed the selected vocabulary set.")
    return StaticDraftVocabTPPlan(
        source_token_ids=tuple(source_ordered),
        destination_token_ids=tuple(tuple(ids) for ids in destination_buckets),
        split_matrix=tuple(tuple(row) for row in split_matrix),
        gathered_token_ids=gathered,
    )


def remap_reduced_draft_output(
    reduced_token_ids: torch.Tensor,
    reduced_probs: torch.Tensor,
    token_id_map: torch.Tensor,
    full_vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if reduced_probs.ndim != 2:
        raise ValueError("Reduced draft probabilities must be two-dimensional.")
    if reduced_probs.shape[-1] != token_id_map.numel():
        raise ValueError("Reduced probability width does not match token map.")
    if full_vocab_size <= 0:
        raise ValueError("Full vocabulary size must be positive.")

    global_token_ids = token_id_map[reduced_token_ids.to(torch.int64)]
    full_probs = torch.zeros(
        (reduced_probs.shape[0], full_vocab_size),
        dtype=reduced_probs.dtype,
        device=reduced_probs.device,
    )
    expanded_map = token_id_map.view(1, -1).expand(reduced_probs.shape[0], -1)
    full_probs.scatter_(1, expanded_map, reduced_probs)
    return global_token_ids, full_probs


def remap_dynamic_draft_output(
    reduced_token_ids: torch.Tensor,
    reduced_probs: torch.Tensor,
    token_id_map: torch.Tensor,
    full_vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    global_token_ids = token_id_map[reduced_token_ids.to(torch.int64)]
    full_probs = torch.zeros(
        (reduced_probs.shape[0], full_vocab_size),
        dtype=reduced_probs.dtype,
        device=reduced_probs.device,
    )
    expanded_map = token_id_map.view(1, -1).expand(reduced_probs.shape[0], -1)
    full_probs.scatter_add_(1, expanded_map, reduced_probs)
    return global_token_ids, full_probs


def initialize_static_draft_vocab(
    target_lm_head: Any,
    ranking_path: str | Path,
    shortlist_size: int,
    device: torch.device,
) -> StaticDraftVocabRuntime:
    from vllm.distributed.parallel_state import get_tp_group
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    payload = torch.load(ranking_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Static draft vocabulary artifact must be a dictionary.")
    global_ranking = payload.get("global_ranking")
    if not isinstance(global_ranking, torch.Tensor) or global_ranking.ndim != 1:
        raise ValueError("Artifact is missing a one-dimensional global_ranking.")

    tp_group = get_tp_group()
    tp_size = tp_group.world_size
    tp_rank = tp_group.rank_in_group
    artifact_tp_size = int(payload.get("tp_size", -1))
    if artifact_tp_size != tp_size:
        raise ValueError(
            f"Ranking artifact TP={artifact_tp_size}, runtime TP={tp_size}."
        )

    full_vocab_size = int(payload.get("model_vocab_size", -1))
    if full_vocab_size <= 0:
        raise ValueError("Artifact has an invalid model vocabulary size.")
    if target_lm_head.num_embeddings_padded != full_vocab_size:
        raise ValueError(
            "Target LM-head and ranking artifact vocabulary sizes differ: "
            f"{target_lm_head.num_embeddings_padded} != {full_vocab_size}."
        )
    if target_lm_head.num_added_embeddings != 0:
        raise NotImplementedError("Static draft vocabulary does not support LoRA rows.")
    if target_lm_head.shard_indices.num_org_vocab_padding != 0:
        raise NotImplementedError(
            "Static draft vocabulary currently requires unpadded target shards."
        )
    if shortlist_size % tp_size:
        raise ValueError("Static draft shortlist must be divisible by TP size.")

    shard_width = target_lm_head.num_embeddings_per_partition
    shard_ranges = tuple(
        (rank * shard_width, min((rank + 1) * shard_width, full_vocab_size))
        for rank in range(tp_size)
    )
    plan = build_static_draft_vocab_tp_plan(
        global_ranking.tolist(), shard_ranges, shortlist_size
    )

    local_start, local_end = shard_ranges[tp_rank]
    source_ids = plan.source_token_ids[tp_rank]
    if any(not local_start <= token_id < local_end for token_id in source_ids):
        raise AssertionError("TP plan assigned nonlocal rows to a source rank.")
    local_indices = torch.tensor(
        [token_id - local_start for token_id in source_ids],
        dtype=torch.int64,
        device=device,
    )
    source_weight = torch.index_select(
        target_lm_head.weight, 0, local_indices
    ).contiguous()

    reduced_head = ParallelLMHead(
        shortlist_size,
        target_lm_head.embedding_dim,
        params_dtype=target_lm_head.weight.dtype,
        org_num_embeddings=shortlist_size,
        quant_config=None,
        prefix="sm70_static_draft_lm_head",
    ).to(device)
    destination_rows = len(plan.destination_token_ids[tp_rank])
    if reduced_head.weight.shape[0] != destination_rows:
        raise AssertionError(
            "Reduced LM-head partition does not match TP destination plan."
        )

    input_splits = list(plan.split_matrix[tp_rank])
    output_splits = [plan.split_matrix[source][tp_rank] for source in range(tp_size)]
    if tp_size == 1:
        reduced_head.weight.data.copy_(source_weight)
    else:
        torch.distributed.all_to_all_single(
            reduced_head.weight.data,
            source_weight,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=tp_group.device_group,
        )
    torch.cuda.synchronize(device)

    fingerprints = payload.get("fingerprints")
    ranking_fingerprint = None
    if isinstance(fingerprints, dict):
        value = fingerprints.get("global_ranking_sha256")
        if isinstance(value, str):
            ranking_fingerprint = value

    token_id_map = torch.tensor(
        plan.gathered_token_ids,
        dtype=torch.int64,
        device=device,
    )
    return StaticDraftVocabRuntime(
        lm_head=reduced_head,
        logits_processor=LogitsProcessor(shortlist_size).to(device),
        token_id_map=token_id_map,
        full_vocab_size=full_vocab_size,
        shortlist_size=shortlist_size,
        ranking_fingerprint=ranking_fingerprint,
    )


def initialize_dynamic_draft_vocab(
    target_lm_head: Any,
    ranking_path: str | Path,
    base_size: int,
    tail_size: int,
    full_refresh_interval: int,
    fused_proposal_enabled: bool,
    num_speculative_tokens: int,
    device: torch.device,
    gpu_lru_enabled: bool = False,
) -> DynamicDraftVocabRuntime:
    if tail_size <= 0:
        raise ValueError("Dynamic draft vocabulary tail size must be positive.")
    if gpu_lru_enabled:
        if not fused_proposal_enabled:
            raise ValueError(
                "Dynamic GPU LRU is validated only with fused TP2/TP4 proposal."
            )
        if base_size != 98_304 or tail_size != 512:
            raise ValueError(
                "Dynamic GPU LRU requires the validated 98,304 base plus "
                "512 shard-local tail rows per rank."
            )
        if target_lm_head.weight.dtype != torch.float16:
            raise ValueError("Dynamic GPU LRU currently requires an FP16 LM-head.")
        if full_refresh_interval != 0:
            raise ValueError("Dynamic GPU LRU requires full_refresh_interval=0.")
    base_runtime = initialize_static_draft_vocab(
        target_lm_head,
        ranking_path,
        base_size,
        device,
    )
    payload = torch.load(ranking_path, map_location="cpu", weights_only=True)
    global_ranking = payload.get("global_ranking")
    if not isinstance(global_ranking, torch.Tensor) or global_ranking.ndim != 1:
        raise ValueError("Artifact is missing a one-dimensional global_ranking.")
    if base_size + tail_size > global_ranking.numel():
        raise ValueError("Dynamic draft base plus tail exceeds the ranking.")

    ranking = [int(token_id) for token_id in global_ranking.tolist()]
    base_token_ids = frozenset(ranking[:base_size])
    tail_lru: OrderedDict[int, None] = OrderedDict()
    for token_id in ranking[base_size : base_size + tail_size]:
        if token_id not in base_token_ids:
            tail_lru[token_id] = None

    gpu_lru_token_ids = None
    gpu_base_token_mask = None
    local_tail_source_row_indices = None
    gpu_lru_empty_observed_ids = None
    gpu_lru_empty_candidate_ids = None
    from vllm.distributed.parallel_state import get_tp_group

    tp_group = get_tp_group()
    tp_size = tp_group.world_size
    tp_rank = tp_group.rank_in_group
    shard_width = target_lm_head.num_embeddings_per_partition
    local_shard_start = tp_rank * shard_width
    local_shard_end = min(
        local_shard_start + shard_width,
        base_runtime.full_vocab_size,
    )
    if gpu_lru_enabled:
        shard_seed = select_dynamic_draft_vocab_shard_seed(
            ranking[base_size:],
            base_token_ids=base_token_ids,
            local_shard_start=local_shard_start,
            local_shard_end=local_shard_end,
            tail_size=tail_size,
        )
        gpu_lru_token_ids = torch.tensor(
            shard_seed,
            dtype=torch.int64,
            device=device,
        )
        gpu_base_token_mask = torch.zeros(
            base_runtime.full_vocab_size,
            dtype=torch.bool,
            device=device,
        )
        gpu_base_token_mask[base_runtime.token_id_map[:base_size]] = True
        local_tail_source_row_indices = torch.empty(
            tail_size,
            dtype=torch.int64,
            device=device,
        )
        gpu_lru_empty_observed_ids = torch.empty(
            0,
            dtype=torch.int32,
            device=device,
        )
        gpu_lru_empty_candidate_ids = torch.empty(
            0,
            dtype=torch.int64,
            device=device,
        )

    local_tail_weight = torch.empty(
        (tail_size, target_lm_head.embedding_dim),
        dtype=target_lm_head.weight.dtype,
        device=device,
    )
    local_tail_token_ids = torch.empty(
        tail_size,
        dtype=torch.int64,
        device=device,
    )
    gathered_tail_token_ids = torch.empty(
        tp_size * tail_size,
        dtype=torch.int64,
        device=device,
    )
    tail_active_mask = torch.empty(
        tp_size * tail_size,
        dtype=torch.bool,
        device=device,
    )
    token_id_map = torch.empty(
        base_size + tp_size * tail_size,
        dtype=torch.int64,
        device=device,
    )
    token_id_map[:base_size].copy_(base_runtime.token_id_map)

    prepared_base_weight = None
    prepared_base_k_ld = 0
    fused_base_values = None
    fused_base_ids = None
    fused_tail_logits = None
    fused_local_pairs = None
    fused_gathered_pairs = None
    fused_sampled_tokens = None
    fused_sparse_ids = None
    fused_sparse_probs = None
    fused_exponentials = None
    fused_dense_probs = None
    if fused_proposal_enabled:
        sm70_ops = _get_sm70_ops()
        if tp_size not in (2, 4):
            raise ValueError("Dynamic fused proposal currently requires TP2 or TP4.")
        if num_speculative_tokens != 4:
            raise ValueError("Dynamic fused proposal currently requires MTP4.")
        if not hasattr(torch.ops._C, "sm70_f16_lm_head_top20_tc_out"):
            raise RuntimeError("Dynamic fused proposal SM70 ops are not built.")
        if gpu_lru_enabled and (
            not hasattr(torch.ops._C, "sm70_dynamic_draft_vocab_update_tail_out")
            or not hasattr(
                torch.ops._C,
                "sm70_dynamic_draft_vocab_refresh_tail_weight_out",
            )
        ):
            raise RuntimeError("Dynamic GPU LRU SM70 ops are not built.")
        prepared = sm70_ops.sm70_f16_prepare(base_runtime.lm_head.weight)
        prepared_base_weight = prepared[0]
        prepared_base_k_ld = int(prepared[1][0].item())
        fused_base_values = torch.empty((1, 20), dtype=torch.float32, device=device)
        fused_base_ids = torch.empty((1, 20), dtype=torch.int64, device=device)
        fused_tail_logits = torch.empty(
            (1, tail_size), dtype=torch.float16, device=device
        )
        fused_local_pairs = torch.empty((20, 3), dtype=torch.float32, device=device)
        fused_gathered_pairs = torch.empty(
            (tp_size * 20, 3), dtype=torch.float32, device=device
        )
        fused_sampled_tokens = torch.empty(
            num_speculative_tokens, dtype=torch.int64, device=device
        )
        fused_sparse_ids = torch.empty(
            (num_speculative_tokens, 20), dtype=torch.int64, device=device
        )
        fused_sparse_probs = torch.empty(
            (num_speculative_tokens, 20), dtype=torch.float32, device=device
        )
        fused_exponentials = torch.empty(
            (num_speculative_tokens, token_id_map.numel()),
            dtype=torch.float32,
            device=device,
        )
        fused_dense_probs = torch.empty(
            (num_speculative_tokens, base_runtime.full_vocab_size),
            dtype=torch.float32,
            device=device,
        )

    runtime = DynamicDraftVocabRuntime(
        lm_head=base_runtime.lm_head,
        logits_processor=base_runtime.logits_processor,
        token_id_map=token_id_map,
        full_vocab_size=base_runtime.full_vocab_size,
        shortlist_size=token_id_map.numel(),
        ranking_fingerprint=base_runtime.ranking_fingerprint,
        base_size=base_size,
        tail_size=tail_size,
        local_tail_capacity=tail_size,
        base_token_ids=base_token_ids,
        tail_lru=tail_lru,
        source_weight=target_lm_head.weight,
        local_shard_start=local_shard_start,
        local_shard_end=local_shard_end,
        local_tail_weight=local_tail_weight,
        local_tail_token_ids=local_tail_token_ids,
        gathered_tail_token_ids=gathered_tail_token_ids,
        tail_active_mask=tail_active_mask,
        tp_device_group=tp_group.device_group,
        full_refresh_interval=full_refresh_interval,
        fused_proposal_enabled=fused_proposal_enabled,
        num_speculative_tokens=num_speculative_tokens,
        tp_group=tp_group,
        prepared_base_weight=prepared_base_weight,
        prepared_base_k_ld=prepared_base_k_ld,
        local_base_start=tp_rank * (base_size // tp_size),
        fused_base_values=fused_base_values,
        fused_base_ids=fused_base_ids,
        fused_tail_logits=fused_tail_logits,
        fused_local_pairs=fused_local_pairs,
        fused_gathered_pairs=fused_gathered_pairs,
        fused_sampled_tokens=fused_sampled_tokens,
        fused_sparse_ids=fused_sparse_ids,
        fused_sparse_probs=fused_sparse_probs,
        fused_exponentials=fused_exponentials,
        fused_dense_probs=fused_dense_probs,
        gpu_lru_enabled=gpu_lru_enabled,
        gpu_lru_token_ids=gpu_lru_token_ids,
        gpu_base_token_mask=gpu_base_token_mask,
        local_tail_source_row_indices=local_tail_source_row_indices,
        gpu_lru_empty_observed_ids=gpu_lru_empty_observed_ids,
        gpu_lru_empty_candidate_ids=gpu_lru_empty_candidate_ids,
    )
    if gpu_lru_enabled:
        assert gpu_lru_empty_observed_ids is not None
        assert gpu_lru_empty_candidate_ids is not None
        runtime._refresh_gpu_tail(
            gpu_lru_empty_observed_ids,
            gpu_lru_empty_candidate_ids,
        )
    else:
        runtime._refresh_tail()
    torch.cuda.synchronize(device)
    return runtime
