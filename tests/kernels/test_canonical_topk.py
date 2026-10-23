# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for selected-ID ordering; CUDA graph checks are separate."""

import ast
from pathlib import Path

import pytest
import torch

from vllm.config import KernelConfig
from vllm.v1.attention.ops.canonical_topk import canonicalize_selected_indices


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_sort_preserves_every_negative_slot_and_duplicates(dtype):
    selected = torch.tensor(
        [[9, -1, 2, 2, -7, 4], [-1, -1, -2, -1, -1, -1]], dtype=dtype
    )
    frozen = selected.clone()
    actual = canonicalize_selected_indices(selected)
    assert actual.tolist() == [[2, -1, 2, 4, -7, 9], [-1, -1, -2, -1, -1, -1]]
    assert torch.equal(selected, frozen)
    assert torch.equal(selected.sort(-1).values, actual.sort(-1).values)
    assert torch.equal(selected < 0, actual < 0)


def test_partial_rows_and_noncontiguous_columns_leave_other_storage_unchanged():
    backing = torch.tensor([[8, 99, -1, 99, 3, 99, 2, 99]] * 6, dtype=torch.int32)
    frozen = backing.clone()
    selected = backing[2:4, ::2]
    assert not selected.is_contiguous()
    selected.copy_(canonicalize_selected_indices(selected))
    assert selected.tolist() == [[2, -1, 3, 8], [2, -1, 3, 8]]
    assert torch.equal(backing[:2], frozen[:2])
    assert torch.equal(backing[4:], frozen[4:])
    assert torch.equal(backing[:, 1::2], frozen[:, 1::2])


@pytest.mark.parametrize("shape", [(0, 8), (3, 0), (0, 0), (1, 1)])
def test_empty_and_singleton(shape):
    selected = torch.full(shape, -1, dtype=torch.int32)
    assert torch.equal(canonicalize_selected_indices(selected), selected)


def test_maximum_integer_remains_a_valid_value():
    maximum = torch.iinfo(torch.int32).max
    selected = torch.tensor([[maximum, -1, 2]], dtype=torch.int32)
    assert canonicalize_selected_indices(selected).tolist() == [[2, -1, maximum]]


@pytest.mark.parametrize("pool_size", [1, 4])
def test_logical_token_or_pool_selection_preserves_expanded_multiset(pool_size):
    selected = torch.tensor([[8, 2, -1, 5, 2]], dtype=torch.int32)
    actual = canonicalize_selected_indices(selected)

    # A pool is an indivisible consecutive group; the separately appended
    # incomplete tail is not a member of this selected-ID tensor.
    def expanded(ids):
        return [
            int(value) * pool_size + offset
            for value in ids[0]
            if value >= 0
            for offset in range(pool_size)
        ]

    tail = [40, 41]
    assert sorted(expanded(selected) + tail) == sorted(expanded(actual) + tail)
    assert torch.equal(selected < 0, actual < 0)


def test_order_does_not_resolve_different_tied_memberships():
    first = torch.tensor([[9, 2, 5]], dtype=torch.int32)
    second = torch.tensor([[8, 2, 5]], dtype=torch.int32)
    assert not torch.equal(
        canonicalize_selected_indices(first), canonicalize_selected_indices(second)
    )


@pytest.mark.parametrize("sparse_active", [False, True])
def test_root_policy_hash_is_independent_of_sm70_qualification(sparse_active):
    native, canonical = KernelConfig(), KernelConfig(canonical_prefill_topk_order=True)
    assert native.canonical_prefill_topk_order is False
    for config in (native, canonical):
        config.sm70_sparse.active = sparse_active
        config.sm70_sparse.qualify("indexer")
        config.sm70_sparse.resolve()
    assert native.compute_hash() != canonical.compute_hash()


def test_fullgraph_cpu_capture_reads_new_values_and_masks():
    # A CPU Dynamo full graph rules out data-dependent Python branching. It
    # does not replace CUDA graph replay validation on each supported device.
    compiled = torch.compile(
        canonicalize_selected_indices, backend="eager", fullgraph=True
    )
    selected = torch.tensor([[7, -1, 3, 3], [-1, 9, 2, 5]], dtype=torch.int32)
    first = compiled(selected)
    assert first.tolist() == [[3, -1, 3, 7], [-1, 2, 5, 9]]
    selected.copy_(torch.tensor([[-1, 4, 1, 2], [8, -2, 3, 1]], dtype=torch.int32))
    second = compiled(selected)
    assert second.tolist() == [[-1, 1, 2, 4], [1, -2, 3, 8]]


def test_invalid_dtype_and_rank_rejected():
    for selected in (torch.zeros((2, 4)), torch.zeros(4, dtype=torch.int32)):
        with pytest.raises(ValueError):
            canonicalize_selected_indices(selected)


def test_prefill_hook_is_after_both_native_selectors_and_before_expansion():
    root = Path(__file__).resolve().parents[2]
    path = root / "vllm/model_executor/layers/sparse_attn_indexer_kpool.py"
    tree = ast.parse(path.read_text())
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "sparse_attn_indexer_kpool"
    )
    prefill = next(
        n
        for n in function.body
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Name)
        and n.test.id == "has_prefill"
    )
    decode = next(
        n
        for n in function.body
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Name)
        and n.test.id == "has_decode"
    )
    loop = next(n for n in prefill.body if isinstance(n, ast.For))
    select = next(
        i
        for i, n in enumerate(loop.body)
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Call)
        and isinstance(n.test.func, ast.Attribute)
        and n.test.func.attr == "is_xpu"
    )
    order = next(
        i
        for i, n in enumerate(loop.body)
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Name)
        and n.test.id == "canonical_prefill_order"
    )
    assert order == select + 1
    assert isinstance(loop.body[order + 1], ast.If)
    assert "pool_ids" in ast.unparse(loop.body[order + 1])
    assert "canonicalize_selected_indices" not in ast.unparse(decode)
    assert "vllm.v1.worker.gpu.causal_" not in path.read_text()
