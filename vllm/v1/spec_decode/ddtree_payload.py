# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DDTree draft payload construction.

This module converts a batched DFlash logits block into per-request DDTree
payloads. It is intentionally separate from the verifier hot path so the tree
builder can be tested and evolved without changing scheduler semantics.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import torch

from vllm.logger import init_logger
from vllm.v1.spec_decode.ddtree_tree import (
    DDTree,
    DDTreeBuildMode,
    DDTreeNode,
    build_ddtree,
)

logger = init_logger(__name__)


@dataclass(frozen=True)
class DDTreeDraftPayload:
    """Flattened per-request DDTree payload produced by a draft model."""

    tree_token_ids: tuple[int, ...]
    parent_indices: tuple[int, ...]
    node_depths: tuple[int, ...]
    node_scores: tuple[float, ...]
    top1_chain_token_ids: tuple[int, ...]
    flat_draft_token_ids: tuple[int, ...]
    budget: int
    top_k: int
    chain_seed: bool
    tree_mode: DDTreeBuildMode = "best_first"
    topk_token_ids_by_depth: tuple[tuple[int, ...], ...] = ()
    topk_logprobs_by_depth: tuple[tuple[float, ...], ...] = ()

    @property
    def num_tree_nodes(self) -> int:
        return len(self.tree_token_ids)

    def flat_chain_matches_top1(self) -> bool:
        return self.flat_draft_token_ids == self.top1_chain_token_ids

    def is_flat_chain(self) -> bool:
        """Return whether verifier nodes are exactly the linear draft chain."""
        num_nodes = self.num_tree_nodes
        expected_parents = () if num_nodes == 0 else (-1,) + tuple(range(num_nodes - 1))
        expected_depths = tuple(range(1, num_nodes + 1))
        return (
            self.tree_token_ids == self.flat_draft_token_ids
            and self.parent_indices == expected_parents
            and self.node_depths == expected_depths
        )


def payload_from_tree(
    *,
    tree: DDTree,
    top1_chain_token_ids: tuple[int, ...],
    flat_draft_token_ids: tuple[int, ...],
    budget: int,
    top_k: int,
    chain_seed: bool,
    tree_mode: DDTreeBuildMode = "best_first",
    topk_token_ids_by_depth: tuple[tuple[int, ...], ...] = (),
    topk_logprobs_by_depth: tuple[tuple[float, ...], ...] = (),
) -> DDTreeDraftPayload:
    return DDTreeDraftPayload(
        tree_token_ids=tree.token_ids_for_verifier(),
        parent_indices=tree.parent_indices_for_verifier(),
        node_depths=tree.node_depths_for_verifier(),
        node_scores=tuple(node.score for node in tree.non_root_nodes),
        top1_chain_token_ids=top1_chain_token_ids,
        flat_draft_token_ids=flat_draft_token_ids,
        budget=budget,
        top_k=top_k,
        chain_seed=chain_seed,
        tree_mode=tree_mode,
        topk_token_ids_by_depth=topk_token_ids_by_depth,
        topk_logprobs_by_depth=topk_logprobs_by_depth,
    )


def tree_from_payload(
    payload: DDTreeDraftPayload,
    *,
    root_token_id: int = -1,
) -> DDTree:
    """Rebuild a DDTree from verifier-coordinate payload arrays."""

    num_nodes = len(payload.tree_token_ids)
    if len(payload.parent_indices) != num_nodes:
        raise ValueError("payload parent_indices length mismatch")
    if len(payload.node_depths) != num_nodes:
        raise ValueError("payload node_depths length mismatch")
    if len(payload.node_scores) != num_nodes:
        raise ValueError("payload node_scores length mismatch")

    nodes: list[DDTreeNode] = [
        DDTreeNode(
            index=0,
            parent_index=None,
            token_id=root_token_id,
            depth=0,
            score=0.0,
        )
    ]
    for verifier_index, (
        token_id,
        parent_index,
        depth,
        score,
    ) in enumerate(
        zip(
            payload.tree_token_ids,
            payload.parent_indices,
            payload.node_depths,
            payload.node_scores,
            strict=True,
        ),
        start=1,
    ):
        if parent_index < -1 or parent_index >= verifier_index - 1:
            raise ValueError(
                "payload parent_indices must reference an earlier verifier node"
            )
        tree_parent_index = 0 if parent_index == -1 else parent_index + 1
        expected_depth = nodes[tree_parent_index].depth + 1
        if depth != expected_depth:
            raise ValueError(
                f"payload depth mismatch for node {verifier_index}: "
                f"expected {expected_depth}, got {depth}"
            )
        nodes.append(
            DDTreeNode(
                index=verifier_index,
                parent_index=tree_parent_index,
                token_id=token_id,
                depth=depth,
                score=score,
            )
        )

    return DDTree(nodes=tuple(nodes))


def build_ddtree_payloads_from_logits(
    *,
    logits: torch.Tensor,
    batch_size: int,
    num_speculative_tokens: int,
    budget: int,
    top_k: int | None,
    chain_seed: bool,
    tree_mode: DDTreeBuildMode = "best_first",
    flat_draft_token_ids: torch.Tensor | None = None,
) -> tuple[DDTreeDraftPayload, ...]:
    """Build per-request DDTree payloads from DFlash first-pass logits.

    Args:
        logits: Tensor with shape ``[batch_size * num_speculative_tokens, vocab]``.
        batch_size: Number of active requests.
        num_speculative_tokens: DFlash proposal depth.
        budget: Maximum number of non-root DDTree verifier nodes.
        top_k: Candidate count per depth. ``None`` uses ``budget``, matching
            the DDTree reference implementation.
        chain_seed: Whether to seed the tree with the top-1 chain.
        tree_mode: DDTree topology. ``best_first`` preserves the original
            cumulative-score expansion. ``root_leaf``/``spine_leaf`` preserve
            the top-1 chain and attach alternative candidates only as leaves
            under the top-1 spine.
        flat_draft_token_ids: Existing flat draft output, shape
            ``[batch_size, num_speculative_tokens]``. This is carried for
            parity checks and does not change scheduler behavior.
    """

    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch * k, vocab]")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if num_speculative_tokens < 1:
        raise ValueError("num_speculative_tokens must be >= 1")
    if budget < 1:
        raise ValueError("budget must be >= 1")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be >= 1")

    expected_rows = batch_size * num_speculative_tokens
    if logits.shape[0] != expected_rows:
        raise ValueError(
            f"logits row mismatch: expected {expected_rows}, got {logits.shape[0]}"
        )
    vocab_size = logits.shape[1]
    requested_top_k = budget if top_k is None else top_k
    effective_top_k = min(requested_top_k, vocab_size)

    profile_enabled = os.getenv("VLLM_DFLASH_DDTREE_WORKER_PROFILE", "0") == "1"
    profile_t0 = time.perf_counter() if profile_enabled else 0.0
    profile_stage_t0 = profile_t0
    float_logits = logits.float()
    float_ms = (
        (time.perf_counter() - profile_stage_t0) * 1000.0 if profile_enabled else 0.0
    )
    profile_stage_t0 = time.perf_counter() if profile_enabled else 0.0
    topk_logits, topk_token_ids = torch.topk(
        float_logits,
        k=effective_top_k,
        dim=-1,
    )
    topk_launch_ms = (
        (time.perf_counter() - profile_stage_t0) * 1000.0 if profile_enabled else 0.0
    )
    profile_stage_t0 = time.perf_counter() if profile_enabled else 0.0
    log_normalizer = torch.logsumexp(float_logits, dim=-1, keepdim=True)
    logsumexp_launch_ms = (
        (time.perf_counter() - profile_stage_t0) * 1000.0 if profile_enabled else 0.0
    )
    profile_stage_t0 = time.perf_counter() if profile_enabled else 0.0
    topk_logprobs = topk_logits - log_normalizer
    logprob_launch_ms = (
        (time.perf_counter() - profile_stage_t0) * 1000.0 if profile_enabled else 0.0
    )

    profile_stage_t0 = time.perf_counter() if profile_enabled else 0.0
    topk_token_ids_cpu = (
        topk_token_ids.view(
            batch_size,
            num_speculative_tokens,
            effective_top_k,
        )
        .detach()
        .cpu()
    )
    ids_cpu_ms = (
        (time.perf_counter() - profile_stage_t0) * 1000.0 if profile_enabled else 0.0
    )
    profile_stage_t0 = time.perf_counter() if profile_enabled else 0.0
    topk_logprobs_cpu = (
        topk_logprobs.view(
            batch_size,
            num_speculative_tokens,
            effective_top_k,
        )
        .detach()
        .cpu()
    )
    logprobs_cpu_ms = (
        (time.perf_counter() - profile_stage_t0) * 1000.0 if profile_enabled else 0.0
    )

    profile_stage_t0 = time.perf_counter() if profile_enabled else 0.0
    if flat_draft_token_ids is None:
        flat_draft_token_ids_cpu = topk_token_ids_cpu[:, :, 0]
        flat_cpu_ms = 0.0
    else:
        if flat_draft_token_ids.shape != (batch_size, num_speculative_tokens):
            raise ValueError(
                "flat_draft_token_ids must have shape "
                f"[{batch_size}, {num_speculative_tokens}]"
            )
        flat_draft_token_ids_cpu = flat_draft_token_ids.detach().cpu()
        flat_cpu_ms = (
            (time.perf_counter() - profile_stage_t0) * 1000.0
            if profile_enabled
            else 0.0
        )

    payloads: list[DDTreeDraftPayload] = []
    profile_stage_t0 = time.perf_counter() if profile_enabled else 0.0
    topk_token_ids_list = topk_token_ids_cpu.tolist()
    topk_logprobs_list = topk_logprobs_cpu.tolist()
    flat_draft_token_ids_list = flat_draft_token_ids_cpu.tolist()
    for req_idx in range(batch_size):
        req_topk_token_ids = topk_token_ids_list[req_idx]
        req_topk_logprobs = topk_logprobs_list[req_idx]
        candidates_by_depth = [
            [
                (int(token_id), float(logprob))
                for token_id, logprob in zip(depth_token_ids, depth_logprobs)
            ]
            for depth_token_ids, depth_logprobs in zip(
                req_topk_token_ids, req_topk_logprobs
            )
        ]
        topk_token_ids_by_depth = tuple(
            tuple(token_id for token_id, _ in candidates)
            for candidates in candidates_by_depth
        )
        topk_logprobs_by_depth = tuple(
            tuple(logprob for _, logprob in candidates)
            for candidates in candidates_by_depth
        )
        tree = build_ddtree(
            candidates_by_depth,
            budget=budget,
            top_k=effective_top_k,
            chain_seed=chain_seed,
            tree_mode=tree_mode,
        )
        top1_chain = tuple(
            int(depth_token_ids[0]) for depth_token_ids in req_topk_token_ids
        )
        flat_chain = tuple(
            int(token_id) for token_id in flat_draft_token_ids_list[req_idx]
        )
        payloads.append(
            payload_from_tree(
                tree=tree,
                top1_chain_token_ids=top1_chain,
                flat_draft_token_ids=flat_chain,
                budget=budget,
                top_k=effective_top_k,
                chain_seed=chain_seed,
                tree_mode=tree_mode,
                topk_token_ids_by_depth=topk_token_ids_by_depth,
                topk_logprobs_by_depth=topk_logprobs_by_depth,
            )
        )

    if profile_enabled:
        logger.info(
            "DFLASH_DDTREE_WORKER_PROFILE payload_detail "
            "total_ms=%.3f float_ms=%.3f topk_launch_ms=%.3f "
            "logsumexp_launch_ms=%.3f logprob_launch_ms=%.3f "
            "ids_cpu_ms=%.3f logprobs_cpu_ms=%.3f flat_cpu_ms=%.3f "
            "tree_cpu_ms=%.3f batch=%d spec_tokens=%d top_k=%d vocab=%d",
            (time.perf_counter() - profile_t0) * 1000.0,
            float_ms,
            topk_launch_ms,
            logsumexp_launch_ms,
            logprob_launch_ms,
            ids_cpu_ms,
            logprobs_cpu_ms,
            flat_cpu_ms,
            (time.perf_counter() - profile_stage_t0) * 1000.0,
            batch_size,
            num_speculative_tokens,
            effective_top_k,
            vocab_size,
        )

    return tuple(payloads)


def build_ddtree_payloads_from_topk(
    *,
    topk_token_ids: torch.Tensor,
    topk_logprobs: torch.Tensor,
    batch_size: int,
    num_speculative_tokens: int,
    budget: int,
    chain_seed: bool,
    tree_mode: DDTreeBuildMode = "best_first",
    flat_draft_token_ids: torch.Tensor | None = None,
) -> tuple[DDTreeDraftPayload, ...]:
    """Build per-request DDTree payloads from precomputed top-k candidates."""

    if topk_token_ids.ndim != 2:
        raise ValueError("topk_token_ids must have shape [batch * k, top_k]")
    if topk_logprobs.shape != topk_token_ids.shape:
        raise ValueError("topk_logprobs must match topk_token_ids shape")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if num_speculative_tokens < 1:
        raise ValueError("num_speculative_tokens must be >= 1")
    if budget < 1:
        raise ValueError("budget must be >= 1")

    expected_rows = batch_size * num_speculative_tokens
    if topk_token_ids.shape[0] != expected_rows:
        raise ValueError(
            f"top-k row mismatch: expected {expected_rows}, "
            f"got {topk_token_ids.shape[0]}"
        )
    effective_top_k = int(topk_token_ids.shape[1])
    if effective_top_k < 1:
        raise ValueError("top-k candidate count must be >= 1")

    profile_enabled = os.getenv("VLLM_DFLASH_DDTREE_WORKER_PROFILE", "0") == "1"
    profile_t0 = time.perf_counter() if profile_enabled else 0.0
    profile_stage_t0 = profile_t0
    topk_token_ids_cpu = (
        topk_token_ids.view(
            batch_size,
            num_speculative_tokens,
            effective_top_k,
        )
        .detach()
        .cpu()
    )
    ids_cpu_ms = (
        (time.perf_counter() - profile_stage_t0) * 1000.0 if profile_enabled else 0.0
    )
    profile_stage_t0 = time.perf_counter() if profile_enabled else 0.0
    topk_logprobs_cpu = (
        topk_logprobs.view(
            batch_size,
            num_speculative_tokens,
            effective_top_k,
        )
        .detach()
        .cpu()
    )
    logprobs_cpu_ms = (
        (time.perf_counter() - profile_stage_t0) * 1000.0 if profile_enabled else 0.0
    )

    profile_stage_t0 = time.perf_counter() if profile_enabled else 0.0
    if flat_draft_token_ids is None:
        flat_draft_token_ids_cpu = topk_token_ids_cpu[:, :, 0]
        flat_cpu_ms = 0.0
    else:
        if flat_draft_token_ids.shape != (batch_size, num_speculative_tokens):
            raise ValueError(
                "flat_draft_token_ids must have shape "
                f"[{batch_size}, {num_speculative_tokens}]"
            )
        flat_draft_token_ids_cpu = flat_draft_token_ids.detach().cpu()
        flat_cpu_ms = (
            (time.perf_counter() - profile_stage_t0) * 1000.0
            if profile_enabled
            else 0.0
        )

    payloads: list[DDTreeDraftPayload] = []
    profile_stage_t0 = time.perf_counter() if profile_enabled else 0.0
    topk_token_ids_list = topk_token_ids_cpu.tolist()
    topk_logprobs_list = topk_logprobs_cpu.tolist()
    flat_draft_token_ids_list = flat_draft_token_ids_cpu.tolist()
    for req_idx in range(batch_size):
        req_topk_token_ids = topk_token_ids_list[req_idx]
        req_topk_logprobs = topk_logprobs_list[req_idx]
        candidates_by_depth = [
            [
                (int(token_id), float(logprob))
                for token_id, logprob in zip(depth_token_ids, depth_logprobs)
            ]
            for depth_token_ids, depth_logprobs in zip(
                req_topk_token_ids, req_topk_logprobs
            )
        ]
        topk_token_ids_by_depth = tuple(
            tuple(token_id for token_id, _ in candidates)
            for candidates in candidates_by_depth
        )
        topk_logprobs_by_depth = tuple(
            tuple(logprob for _, logprob in candidates)
            for candidates in candidates_by_depth
        )
        tree = build_ddtree(
            candidates_by_depth,
            budget=budget,
            top_k=effective_top_k,
            chain_seed=chain_seed,
            tree_mode=tree_mode,
        )
        top1_chain = tuple(
            int(depth_token_ids[0]) for depth_token_ids in req_topk_token_ids
        )
        flat_chain = tuple(
            int(token_id) for token_id in flat_draft_token_ids_list[req_idx]
        )
        payloads.append(
            payload_from_tree(
                tree=tree,
                top1_chain_token_ids=top1_chain,
                flat_draft_token_ids=flat_chain,
                budget=budget,
                top_k=effective_top_k,
                chain_seed=chain_seed,
                tree_mode=tree_mode,
                topk_token_ids_by_depth=topk_token_ids_by_depth,
                topk_logprobs_by_depth=topk_logprobs_by_depth,
            )
        )

    if profile_enabled:
        logger.info(
            "DFLASH_DDTREE_WORKER_PROFILE payload_topk_detail "
            "total_ms=%.3f ids_cpu_ms=%.3f logprobs_cpu_ms=%.3f "
            "flat_cpu_ms=%.3f tree_cpu_ms=%.3f batch=%d spec_tokens=%d "
            "top_k=%d",
            (time.perf_counter() - profile_t0) * 1000.0,
            ids_cpu_ms,
            logprobs_cpu_ms,
            flat_cpu_ms,
            (time.perf_counter() - profile_stage_t0) * 1000.0,
            batch_size,
            num_speculative_tokens,
            effective_top_k,
        )

    return tuple(payloads)
