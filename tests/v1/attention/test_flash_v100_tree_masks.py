# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Feature-owned masks preserve ancestry, capture guards and live injection."""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends.flash_v100 import impl, prefill, prefill_candidates
from vllm.v1.attention.backends.flash_v100.spec import tree_masks

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("window", [(-1, -1), (2, 0)])
def test_branch_visibility_excludes_siblings_and_respects_window(window):
    mask = tree_masks.build_visibility_mask(
        q_len=4,
        seq_len=6,
        prefix_len=2,
        tree_len=3,
        parent_row=torch.tensor([0, 0, 0, 2]),
        device=torch.device("cpu"),
        window_size=window,
    )
    expected = (
        [[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 1, 0], [1, 1, 1, 0, 1, 1]]
        if window == (-1, -1)
        else [
            [1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 1, 1],
        ]
    )
    assert torch.equal(mask, torch.tensor(expected, dtype=torch.bool))


@pytest.mark.parametrize("kind", ["seq_lens", "query_start_loc"])
def test_capture_requires_restoration_without_host_comparison(monkeypatch, kind):
    source = torch.tensor([0, 4, 8], dtype=torch.int32)
    restored = "ddtree_" + kind + "_restored_for_triton"
    metadata = SimpleNamespace(**{kind: source, restored: False})
    check = (
        tree_masks.triton_seq_lens_match
        if kind == "seq_lens"
        else tree_masks.triton_query_start_loc_match
    )

    def unexpected(*args):
        raise AssertionError("capture must not compare values on the host")

    monkeypatch.setattr(tree_masks._routing, "_is_cuda_graph_capturing", lambda t: True)
    monkeypatch.setattr(torch, "equal", unexpected)
    assert check(metadata, source, 2)
    assert not check(metadata, source.clone(), 2)
    setattr(metadata, restored, True)
    assert check(metadata, source.clone(), 2)


def test_mixed_linear_rows_decline_capture_without_mutating_parents():
    parents = torch.tensor([[0, 0, 0, 2], [0, 0, 0, 0]], dtype=torch.int32)
    original = parents.clone()
    lengths = torch.tensor([3, 0])
    starts = torch.tensor([0, 4, 8])
    assert (
        tree_masks.triton_parent_ids_for_query(
            parents, lengths, starts, is_capturing=True
        )
        is None
    )
    result = tree_masks.triton_parent_ids_for_query(
        parents, lengths, starts, is_capturing=False
    )
    assert result is not None and result.data_ptr() != parents.data_ptr()
    assert result.tolist() == [[0, 0, 0, 2], [0, 0, 1, 2]]
    assert torch.equal(parents, original)


def test_legacy_patch_is_consumed_by_owned_batch_admission(monkeypatch):
    from vllm.v1.attention.backends import flash_attn_v100 as legacy

    original = tree_masks.parent_metadata_requires_branch
    calls = []

    def tracked(metadata, starts):
        calls.append(metadata)
        return original(metadata, starts)

    monkeypatch.setattr(legacy, "_ddtree_parent_metadata_requires_branch", tracked)
    instance = object.__new__(impl.FlashAttnV100Impl)
    instance.scale = 0.125
    instance.kv_cache_dtype = "auto"
    owner = instance._new_prefill_executor()
    executor = prefill.create_prefill_executor(owner)
    metadata = SimpleNamespace(
        ddtree_parent_ids=torch.tensor([[0, 0, 1]]),
        ddtree_num_tree_tokens_cpu=torch.tensor([2]),
    )
    request = SimpleNamespace(
        causal=True, attn_metadata=metadata, query_start_loc=torch.tensor([0, 3])
    )
    assert prefill_candidates.TreeBatch(executor).admit(request)
    assert calls == [metadata]
    assert owner.ops.tree_requires_branch is tree_masks.parent_metadata_requires_branch
    ops = instance._new_verification_executor().ops
    assert ops.tree_seq_lens_match is tree_masks.triton_seq_lens_match
    assert ops.tree_query_start_match is tree_masks.triton_query_start_loc_match
    assert ops.tree_parent_ids is tree_masks.triton_parent_ids_for_query
    assert ops.tree_visibility is tree_masks.build_visibility_mask
