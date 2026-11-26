# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What the profile run reserves for MLA must be what the layer later uses."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.model_executor.layers.attention.mla_attention import (
    MLAAttention,
    SparseMLAAttentionImpl,
)
from vllm.v1.attention.backends.mla.indexer import get_max_prefill_buffer_size

NUM_HEADS, QK_NOPE, V_HEAD, WORKSPACE_TOKENS = 3, 5, 6, 7


def _profile_run(impl, monkeypatch) -> list[tuple[int, ...]]:
    """Run the metadata-free branch of forward_impl and list the shapes it allocates."""
    allocated: list[tuple[int, ...]] = []
    real_empty = torch.empty

    def recording_empty(*args, **kwargs):
        shape = args[0] if args and isinstance(args[0], tuple) else args
        allocated.append(tuple(shape))
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", recording_empty)
    layer = SimpleNamespace(
        impl=impl,
        num_heads=NUM_HEADS,
        v_head_dim=V_HEAD,
        qk_nope_head_dim=QK_NOPE,
        chunked_prefill_workspace_size=WORKSPACE_TOKENS,
    )
    tokens = torch.zeros(2, 4)
    MLAAttention.forward_impl(
        layer, tokens, tokens, tokens, None, None, torch.zeros(2, NUM_HEADS * V_HEAD)
    )
    return allocated


def test_dense_mla_reserves_the_up_projected_context(monkeypatch):
    shapes = _profile_run(object(), monkeypatch)
    assert (WORKSPACE_TOKENS, NUM_HEADS, QK_NOPE + V_HEAD) in shapes


def test_sparse_mla_reserves_no_up_projected_context(monkeypatch):
    # A sparse impl only runs forward_mqa: kv_b_proj never sees the cached context.
    shapes = _profile_run(Mock(spec=SparseMLAAttentionImpl), monkeypatch)
    assert (WORKSPACE_TOKENS, NUM_HEADS, QK_NOPE + V_HEAD) not in shapes


@pytest.mark.parametrize(
    "max_num_seqs,expected_sequences",
    [(1, 1), (4, 4), (40, 40), (256, 40)],
)
def test_indexer_gather_buffer_holds_the_sequences_a_batch_can_hold(
    max_num_seqs, expected_sequences
):
    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=262_144),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
    )
    assert get_max_prefill_buffer_size(config) == 262_144 * expected_sequences
