# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
from vllm.v1.sample.ops.topk_topp_triton import (
    apply_top_k_top_p_triton,
    sort_topk_with_vocab_ties,
)
from vllm.v1.worker.gpu.spec_decode.dflash2 import sparse_rejection
from vllm.v1.worker.gpu.spec_decode.dflash2.sparse_rejection import (
    _compact_target_requires_reference,
)


@pytest.mark.parametrize("top_p", [1.0, 0.95, 0.6])
@pytest.mark.parametrize("case", ["k_tie", "p_tie", "uniform", "unique"])
@pytest.mark.parametrize("rows", [1, 8])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_tied_cutoffs_match_full_vocabulary_reference(case, top_p, rows):
    x = torch.full((rows, 32768), -20.0, device="cuda")
    if case == "k_tie":
        x[:, :24] = 1.0
        x[:, :18] = 2.0
    elif case == "p_tie":
        x[:, :20] = 1.0
        x[:, :2] = 2.0
    elif case == "uniform":
        x.fill_(1.0)
    else:
        x[:, :32] = torch.arange(32, 0, -1, device="cuda") / 8
    k = torch.full((rows,), 20, dtype=torch.int32, device="cuda")
    p = torch.full((rows,), top_p, device="cuda")
    expected = apply_top_k_top_p_pytorch(x.clone(), k, p)
    actual = apply_top_k_top_p_triton(x.clone(), k, p)
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows", [1, 19, 20])
@pytest.mark.parametrize("top_k,top_p", [(20, 0.95), (50, 0.9)])
def test_compact_fallback_bounds_full_vocabulary_sort_workspace(
    monkeypatch, rows, top_k, top_p
):
    from vllm.v1.sample.ops import topk_topp_sampler

    # MTP4 at C4 has 20 target rows. Uniform logits force the ambiguous tie
    # fallback, which previously sorted the entire expanded batch together.
    logits = torch.ones((rows, 248320), device="cuda")
    k = torch.full((rows,), top_k, dtype=torch.int32, device="cuda")
    p = torch.full((rows,), top_p, device="cuda")
    expected = apply_top_k_top_p_pytorch(logits.clone(), k, p)
    reference_rows = []

    def bounded_reference(x, k, p, **kwargs):
        reference_rows.append(x.shape[0])
        if rows > 1:
            assert kwargs["rowwise_sort"]
        return apply_top_k_top_p_pytorch(x, k, p, **kwargs)

    monkeypatch.setattr(
        topk_topp_sampler, "apply_top_k_top_p_pytorch", bounded_reference
    )
    actual = apply_top_k_top_p_triton(logits, k, p)
    assert reference_rows == ([1] if rows == 1 else [2] * ((rows + 1) // 2))
    assert torch.equal(actual, expected)
    if rows > 1:
        assert torch.equal(logits, torch.ones_like(logits))
    else:
        # The established singleton reference route permits in-place masking.
        assert actual.data_ptr() == logits.data_ptr()


@pytest.mark.parametrize("mask_value", [-float("inf"), -123.0])
@pytest.mark.parametrize("use_top_k", [False, True])
def test_single_row_direct_entry_keeps_grammar_ties_on_reference(
    monkeypatch, mask_value, use_top_k
):
    from vllm.v1.sample.ops import topk_topp_triton

    def unsafe_pivot(*args):
        raise AssertionError("Single-row call reached the unsafe pivot kernel")

    monkeypatch.setattr(topk_topp_triton, "num_compute_units", unsafe_pivot)
    # A strided row with masked vocabulary and a nucleus split inside a tie.
    x = torch.full((1, 65536), -float("inf"))[:, ::2]
    x[:, :24] = 1.0
    x[:, :2] = 2.0
    k = torch.tensor([20], dtype=torch.int32) if use_top_k else None
    p = torch.tensor([0.8])
    expected = apply_top_k_top_p_pytorch(x.clone(), k, p)
    if mask_value != -float("inf"):
        expected.masked_fill_(torch.isneginf(expected), mask_value)
    actual = apply_top_k_top_p_triton(x, k, p, mask_value=mask_value)
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_standalone_topp_ties_and_graph_capture():
    x = torch.full((2, 32768), -20.0, device="cuda")
    x[:, :20] = 1.0
    x[:, :2] = 2.0
    p = torch.full((2,), 0.95, device="cuda")
    expected = apply_top_k_top_p_pytorch(x.clone(), None, p)
    actual = apply_top_k_top_p_triton(x.clone(), None, p)
    assert torch.equal(actual, expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = apply_top_k_top_p_triton(x.clone(), None, p)
    graph.replay()
    assert torch.equal(captured, expected)


@pytest.mark.parametrize("use_top_k", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mixed_cutoffs_keep_row_parameters(use_top_k):
    x = torch.full((8, 32768), -20.0, device="cuda")
    x[:, :32] = torch.arange(32, 0, -1, device="cuda") / 8
    x[1::2, :24] = 1.0
    x[1::2, :2] = 2.0
    k = torch.tensor([8, 12, 16, 20] * 2, dtype=torch.int32, device="cuda")
    p = torch.tensor([0.8, 0.95, 0.6, 1.0] * 2, device="cuda")
    if not use_top_k:
        k = None
    expected = apply_top_k_top_p_pytorch(x.clone(), k, p)
    actual = apply_top_k_top_p_triton(x.clone(), k, p)
    assert torch.equal(actual, expected)


def test_compact_guard_only_rejects_ambiguous_cutoffs():
    unique = torch.arange(21, 0, -1, dtype=torch.float32)[None] / 8
    assert not _compact_target_requires_reference(unique, 1.0, 0.95)
    k_tie = unique.clone()
    k_tie[:, -1] = k_tie[:, -2]
    assert _compact_target_requires_reference(k_tie, 1.0, 1.0)
    p_tie = torch.ones(1, 21)
    p_tie[:, :2] = 2.0
    p_tie[:, -1] = -20.0
    assert _compact_target_requires_reference(p_tie, 1.0, 0.95)
    assert not _compact_target_requires_reference(p_tie, 1.0, 1.0)


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
def test_compact_guard_checks_each_rows_sampling_parameters(temperature):
    unique = torch.arange(21, 0, -1, dtype=torch.float32) / 8
    tied = torch.ones(21)
    tied[:2] = 2.0
    tied[-1] = -20.0
    probe = torch.stack((unique, tied, tied))
    temperatures = np.array([1.0, temperature, temperature], dtype=np.float32)
    top_p = np.array([1.0, 1.0, 0.95], dtype=np.float32)
    # Using the first request's top_p for the whole batch misses the final
    # row's split tie, changing the retained vocabulary support.
    assert not _compact_target_requires_reference(probe, 1.0, 1.0)
    expected = any(
        _compact_target_requires_reference(probe[i : i + 1], t, p)
        for i, (t, p) in enumerate(zip(temperatures, top_p))
    )
    assert expected
    assert _compact_target_requires_reference(probe, temperatures, top_p) == expected


@pytest.mark.parametrize(
    ("temperatures", "top_ps"),
    [([0.5, 2.0, 1.0], [0.95, 0.9, 1.0]), ([1.0, 2.0, 0.1], [0.8, 0.9, 0.8])],
)
def test_compact_rejection_uses_request_mapping_and_variable_row_counts(
    monkeypatch, temperatures, top_ps
):
    class Speculator:
        def get_sparse_draft_logits(self):
            return None, None

    monkeypatch.setattr(sparse_rejection, "DFlash2Speculator", Speculator)
    monkeypatch.setattr(
        sparse_rejection.envs, "VLLM_SM70_DFLASH2_SPARSE_TARGET_REJECTION", True
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (7, 0))
    monkeypatch.setattr(
        sparse_rejection, "_supports_sparse_sampling_contract", lambda *args: True
    )
    probe = torch.ones(3, 21)
    probe[:, :2] = 2.0
    probe[:, -1] = -20.0
    states = SimpleNamespace(
        temperature=SimpleNamespace(np=np.array(temperatures)),
        top_p=SimpleNamespace(np=np.array(top_ps)),
    )
    batch = SimpleNamespace(
        has_structured_output_reqs=False,
        idx_mapping_np=np.array([2, 0]),
        cu_num_logits_np=np.array([0, 1, 3]),
    )
    # Request 2 is unambiguous; the two rows of request 0 need the reference.
    # No GPU sampling should be attempted after detecting the split tie.
    result = sparse_rejection.try_dflash2_sparse_target_rejection(
        SimpleNamespace(get_topk_tokens_and_logits=lambda *args: (None, probe)),
        Speculator(),
        SimpleNamespace(sampler=SimpleNamespace(sampling_states=states)),
        SimpleNamespace(device=SimpleNamespace(type="cuda")),
        batch,
        None,
    )
    assert result is None


@pytest.mark.parametrize("rows", [8, 16, 32, 64])
@pytest.mark.parametrize("vocab", [32768, 248320])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compact_large_batch_matches_reference(rows, vocab):
    # Mix cutoff ties, more ties than the shortlist can hold, grammar masks,
    # distinct cutoffs, and heterogeneous request parameters in one batch.
    torch.manual_seed(1527)
    x = (torch.randn(rows, vocab, device="cuda") * 1.5).half().float()
    x[0].fill_(1.0)
    x[1].fill_(-float("inf"))
    x[1, :24] = 1.0
    x[1, :2] = 2.0
    x[2].fill_(-20.0)
    x[2, -160:] = 3.0
    x[3].fill_(-float("inf"))
    x[3, 7] = 1.0
    k = torch.tensor([1, 20, 64, 127] * (rows // 4), device="cuda")
    p = torch.tensor([0.6, 0.8, 0.95, 1.0] * (rows // 4), device="cuda")
    expected = apply_top_k_top_p_pytorch(x.clone(), k, p)
    actual = apply_top_k_top_p_triton(x.clone(), k, p)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("direction", [-1, 0, 1])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compact_top_p_rounding_boundary(direction):
    torch.manual_seed(1528)
    x = torch.randn(16, 32768, device="cuda")
    k = torch.full((16,), 20, dtype=torch.int32, device="cuda")
    ordered = x.sort(dim=-1).values
    cutoff = ordered[:, -20:-19]
    ordered.masked_fill_(ordered < cutoff, -float("inf"))
    probabilities = ordered.softmax(-1).cumsum(-1)
    p = 1 - probabilities[:, -9]
    if direction:
        p = torch.nextafter(p, torch.full_like(p, float("inf") * direction))
    expected = apply_top_k_top_p_pytorch(x.clone(), k, p)
    actual = apply_top_k_top_p_triton(x.clone(), k, p)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("rows", [33, 64])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compact_dense_fallback_bounds_temporary_rows(monkeypatch, rows):
    from vllm.v1.sample.ops import topk_topp_sampler
    from vllm.v1.sample.ops.topk_topp_triton import _apply_top_k_top_p_compact

    # Every row exceeds the shortlist at its top-k cutoff. Different top-p
    # parameters exercise row mapping across complete and partial chunks.
    logits = torch.ones((rows, 248320), device="cuda")
    logits[:, :2] = 2.0
    k = torch.full((rows,), 20, dtype=torch.int32, device="cuda")
    p = torch.linspace(0.6, 1.0, rows, device="cuda")
    expected = apply_top_k_top_p_pytorch(logits.clone(), k, p)
    dense_batch_sizes = []

    def reference(chunk, chunk_k, chunk_p):
        dense_batch_sizes.append(chunk.shape[0])
        return apply_top_k_top_p_pytorch(chunk, chunk_k, chunk_p)

    monkeypatch.setattr(topk_topp_sampler, "apply_top_k_top_p_pytorch", reference)
    actual = _apply_top_k_top_p_compact(logits, k, p, -float("inf"))
    assert torch.equal(actual, expected)
    assert actual.data_ptr() == logits.data_ptr()
    assert sum(dense_batch_sizes) == rows
    assert max(dense_batch_sizes) <= 16


@pytest.mark.parametrize("rows", [2, 8, 32])
@pytest.mark.parametrize("vocab", [32768, 248320])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compact_ties_preserve_vocabulary_order(rows, vocab):
    torch.manual_seed(1530)
    x = torch.full((rows, vocab), -20.0, device="cuda")
    # Scatter a split nucleus tie across radix-sort tiles and the shortlist's
    # arbitrary topk order. It must keep the reference's exact token support.
    for row in range(rows):
        ids = torch.randperm(vocab, device="cuda")[:24]
        x[row, ids] = 1.0
        x[row, ids[:2]] = 2.0
    k = torch.full((rows,), 20, dtype=torch.int32, device="cuda")
    p = torch.full((rows,), 0.95, device="cuda")
    expected = apply_top_k_top_p_pytorch(x.clone(), k, p)
    actual = apply_top_k_top_p_triton(x.clone(), k, p)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("mask_value", [float("-inf"), -1e9])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compact_changed_inputs_strides_and_large_k(mask_value):
    torch.manual_seed(1529)
    storage = torch.randn(16, 65536, device="cuda")
    x = storage[:, ::2]
    k = torch.full((16,), 20, dtype=torch.int32, device="cuda")
    p = torch.full((16,), 0.8, device="cuda")
    for amplitude in (0.1, 1.0, 3.0):
        x.normal_(0, amplitude)
        expected = apply_top_k_top_p_pytorch(x.clone(), k, p)
        if mask_value != float("-inf"):
            expected.masked_fill_(torch.isneginf(expected), mask_value)
        actual = apply_top_k_top_p_triton(x, k, p, mask_value)
        assert torch.equal(actual, expected)
    # Large k must keep the existing general implementation.
    k[::2] = 512
    expected = apply_top_k_top_p_pytorch(x.clone(), k, p)
    actual = apply_top_k_top_p_triton(x.clone(), k, p)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("width", [21, 64, 128])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_shortlist_order_preserves_radix_ties_and_value_bits(width, descending):
    torch.manual_seed(1540)
    ids = torch.stack(
        [torch.randperm(248320, device="cuda")[: width + 3] for _ in range(8)]
    )[:, :width]
    storage = torch.randn(8, width + 3, device="cuda")
    values = storage[:, :width]
    output = sort_topk_with_vocab_ties(
        values, ids, vocab_size=248320, descending=descending
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = sort_topk_with_vocab_ties(
            values, ids, vocab_size=248320, descending=descending
        )
    for amplitude in (0.1, 2.0):
        values.normal_(0, amplitude)
        values[:, :12] = 1.0
        values[:, 12:18] = torch.tensor(
            [-0.0, 0.0, -float("inf"), float("inf"), float("nan"), -float("nan")],
            device="cuda",
        )
        ordered_ids, permutation = ids.sort(dim=-1, descending=descending)
        ordered_values = values.gather(1, permutation)
        ordered_values, permutation = ordered_values.sort(
            dim=-1, descending=descending, stable=True
        )
        ordered_ids = ordered_ids.gather(1, permutation)
        graph.replay()
        assert torch.equal(output[1], ordered_ids)
        assert torch.equal(
            output[0].view(torch.int32), ordered_values.view(torch.int32)
        )
