# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DDTree verifier helpers.

The helpers here consume the payload produced by DFlash and the compact logits
produced by a future tree verifier. They do not run the target model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.v1.spec_decode.ddtree_metadata import (
    DDTreeVerifierMetadata,
    make_prefill_tree_attention_mask,
)
from vllm.v1.spec_decode.ddtree_payload import (
    DDTreeDraftPayload,
    tree_from_payload,
)


@dataclass(frozen=True)
class DDTreeVerificationResult:
    accepted_node_indices: tuple[int, ...]
    accepted_token_ids: tuple[int, ...]
    bonus_token_id: int
    output_token_ids: tuple[int, ...]

    @property
    def num_accepted_tokens(self) -> int:
        return len(self.accepted_token_ids)


@dataclass(frozen=True)
class DDTreeAttentionVerifierInputs:
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    attention_mask: torch.Tensor
    compact_logits_indices: torch.Tensor
    metadata: DDTreeVerifierMetadata


def _payload_child_maps(payload: DDTreeDraftPayload) -> tuple[dict[int, int], ...]:
    """Build official-style row-index child maps from verifier payload arrays."""

    num_nodes = payload.num_tree_nodes
    if len(payload.parent_indices) != num_nodes:
        raise ValueError("payload parent_indices length mismatch")
    if len(payload.node_depths) != num_nodes:
        raise ValueError("payload node_depths length mismatch")
    if len(payload.node_scores) != num_nodes:
        raise ValueError("payload node_scores length mismatch")

    child_maps: list[dict[int, int]] = [dict() for _ in range(num_nodes + 1)]
    for node_row, (token_id, parent_index, depth) in enumerate(
        zip(
            payload.tree_token_ids,
            payload.parent_indices,
            payload.node_depths,
            strict=True,
        ),
        start=1,
    ):
        if parent_index < -1 or parent_index >= node_row - 1:
            raise ValueError(
                "payload parent_indices must reference an earlier verifier node"
            )
        parent_row = 0 if parent_index == -1 else parent_index + 1
        parent_depth = 0 if parent_row == 0 else payload.node_depths[parent_row - 1]
        expected_depth = int(parent_depth) + 1
        if depth != expected_depth:
            raise ValueError(
                f"payload depth mismatch for node {node_row}: "
                f"expected {expected_depth}, got {depth}"
            )
        child_maps[parent_row][int(token_id)] = node_row
    return tuple(child_maps)


def _greedy_verify_payload_from_target_tokens(
    *,
    payload: DDTreeDraftPayload,
    target_tokens: list[int],
) -> DDTreeVerificationResult:
    """Follow the DDTree with one target token per compact verifier row."""

    expected_rows = payload.num_tree_nodes + 1
    if len(target_tokens) != expected_rows:
        raise ValueError(
            f"compact target token row mismatch: expected {expected_rows}, "
            f"got {len(target_tokens)}"
        )

    child_maps = _payload_child_maps(payload)
    cursor = 0
    accepted_nodes: list[int] = [0]
    accepted_tokens: list[int] = []

    while True:
        next_token = int(target_tokens[cursor])
        child_index = child_maps[cursor].get(next_token)
        if child_index is None:
            output_tokens = accepted_tokens + [next_token]
            return DDTreeVerificationResult(
                accepted_node_indices=tuple(accepted_nodes),
                accepted_token_ids=tuple(accepted_tokens),
                bonus_token_id=next_token,
                output_token_ids=tuple(output_tokens),
            )

        accepted_nodes.append(child_index)
        accepted_tokens.append(next_token)
        cursor = child_index


def greedy_verify_payload_from_target_tokens(
    *,
    payload: DDTreeDraftPayload,
    target_tokens: list[int],
) -> DDTreeVerificationResult:
    """Greedy-verify a DDTree payload from compact argmax token ids."""

    return _greedy_verify_payload_from_target_tokens(
        payload=payload,
        target_tokens=target_tokens,
    )


def metadata_from_payload(
    *,
    prompt_len: int,
    payload: DDTreeDraftPayload,
) -> DDTreeVerifierMetadata:
    return DDTreeVerifierMetadata.from_tree(
        prompt_len=prompt_len,
        tree=tree_from_payload(payload),
    )


def make_attention_verifier_inputs(
    *,
    prompt_token_ids: torch.Tensor,
    payload: DDTreeDraftPayload,
    mask_dtype: torch.dtype,
) -> DDTreeAttentionVerifierInputs:
    """Build prompt-plus-tree tensors for an attention-only verifier oracle."""

    if prompt_token_ids.ndim != 1:
        raise ValueError("prompt_token_ids must be a 1D tensor")
    if prompt_token_ids.numel() < 1:
        raise ValueError("prompt_token_ids must contain at least one token")

    tree = tree_from_payload(payload)
    metadata = DDTreeVerifierMetadata.from_tree(
        prompt_len=int(prompt_token_ids.numel()),
        tree=tree,
    )
    tree_token_ids = torch.tensor(
        metadata.tree_token_ids,
        dtype=prompt_token_ids.dtype,
        device=prompt_token_ids.device,
    )
    input_ids = torch.cat((prompt_token_ids, tree_token_ids), dim=0)
    position_ids = torch.tensor(
        metadata.all_position_ids(),
        dtype=torch.long,
        device=prompt_token_ids.device,
    )
    attention_mask = make_prefill_tree_attention_mask(
        prompt_len=int(prompt_token_ids.numel()),
        tree=tree,
        device=prompt_token_ids.device,
        dtype=mask_dtype,
    )
    compact_logits_indices = torch.tensor(
        metadata.compact_logits_indices,
        dtype=torch.long,
        device=prompt_token_ids.device,
    )

    return DDTreeAttentionVerifierInputs(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        compact_logits_indices=compact_logits_indices,
        metadata=metadata,
    )


def greedy_verify_payload_from_compact_logits(
    *,
    payload: DDTreeDraftPayload,
    compact_logits: torch.Tensor,
) -> DDTreeVerificationResult:
    """Greedy-verify a DDTree payload from compact root-plus-node logits."""

    if compact_logits.ndim != 2:
        raise ValueError("compact_logits must have shape [rows, vocab]")
    expected_rows = payload.num_tree_nodes + 1
    if compact_logits.shape[0] != expected_rows:
        raise ValueError(
            f"compact_logits row mismatch: expected {expected_rows}, "
            f"got {compact_logits.shape[0]}"
        )
    target_tokens = compact_logits.argmax(dim=-1).detach().cpu().tolist()
    return _greedy_verify_payload_from_target_tokens(
        payload=payload,
        target_tokens=target_tokens,
    )


def greedy_verify_payload_from_compact_top_tokens(
    *,
    payload: DDTreeDraftPayload,
    compact_top_tokens: torch.Tensor,
) -> DDTreeVerificationResult:
    """Greedy-verify a DDTree payload from compact root-plus-node argmax tokens."""

    if compact_top_tokens.ndim != 1:
        raise ValueError("compact_top_tokens must have shape [rows]")
    expected_rows = payload.num_tree_nodes + 1
    if compact_top_tokens.shape[0] != expected_rows:
        raise ValueError(
            f"compact_top_tokens row mismatch: expected {expected_rows}, "
            f"got {compact_top_tokens.shape[0]}"
        )
    target_tokens = compact_top_tokens.detach().cpu().tolist()
    return _greedy_verify_payload_from_target_tokens(
        payload=payload,
        target_tokens=target_tokens,
    )
