# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.qwen3_5_mtp_refined_head import (
    _rerank_candidates,
    _select_candidates,
    prepare_mtp_refined_head,
)


def test_refined_head_cpu_fallback():
    head = torch.nn.Module()
    head.weight = torch.nn.Parameter(torch.zeros(1))
    assert (
        prepare_mtp_refined_head(head, SimpleNamespace(speculative_config=None)) is None
    )


@pytest.mark.parametrize("rows", [1, 8])
@pytest.mark.parametrize("kind", ["random", "zero", "nonfinite", "negative_infinity"])
def test_refined_head_candidate_order(rows, kind):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    torch.manual_seed(653)
    logits = torch.randn(rows, 62080, device="cuda", dtype=torch.float16)
    if kind == "zero":
        logits.fill_(-1)
        logits[:, :100] = 0
        logits[:, 1:100:2] = -0.0
    elif kind == "nonfinite":
        logits[:, 100:180] = float("inf")
        logits[:, 200:220] = float("nan")
    elif kind == "negative_infinity":
        logits.fill_(-float("inf"))
    expected = torch.argsort(logits, dim=-1, descending=True, stable=True)[:, :32]
    actual = _select_candidates(logits)
    assert torch.equal(actual.long(), expected)


def test_refined_head_rerank_ties_and_replay():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    x = torch.zeros(8, 5120, device="cuda", dtype=torch.float16)
    weight = torch.zeros(64, 5120, device="cuda", dtype=torch.float16)
    weight[:, 0] = torch.arange(64, device="cuda", dtype=torch.float16)
    ids = torch.arange(31, -1, -1, device="cuda", dtype=torch.int32).expand(8, -1)
    ids = ids.contiguous()
    _rerank_candidates(x, weight, ids, 62080)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = _rerank_candidates(x, weight, ids, 62080)
    for value, best_id in [(0, 62080), (1, 62111), (-1, 62080)]:
        x[:, 0] = value
        graph.replay()
        assert torch.equal(output[:, 1], torch.full_like(output[:, 1], best_id))
        assert torch.equal(
            output[:, 0], torch.full_like(output[:, 0], 31 if value == 1 else 0)
        )


@pytest.mark.parametrize("rows", [1, 5, 8, 9, 17, 32, 33, 40, 64])
def test_fp16_candidate_scores_match_dense_tensor_core_order(rows):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    operation = getattr(torch.ops._C, "sm70_fp16_candidate_scores_out", None)
    if operation is None:
        pytest.skip("requires the candidate scoring operator")
    assert operation is not None
    torch.manual_seed(71328)
    weight = torch.randn(62080, 5120, device="cuda", dtype=torch.float16)
    x = torch.randn(rows, 5120, device="cuda", dtype=torch.float16)
    ids = torch.randint(0, 62080, (rows, 32), device="cuda", dtype=torch.int32)
    scores = torch.empty(rows, 32, device="cuda", dtype=torch.float16)
    operation(scores, x, weight, ids)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operation(scores, x, weight, ids)
    for cycle in range(32):
        x.normal_().mul_(0.1 if cycle < 16 else 2)
        graph.replay()
        # The reference head uses the dense WMMA reduction. Pad the one-row
        # oracle so cuBLAS keeps that path independently of GEMV heuristics.
        dense_input = x.expand(8, -1).contiguous() if rows == 1 else x
        dense = torch.nn.functional.linear(dense_input, weight)[:rows]
        expected = dense.gather(1, ids.long())
        assert torch.equal(scores.view(torch.int16), expected.view(torch.int16))


def test_fp16_candidate_scores_invalid_ids_are_bounded():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    operation = getattr(torch.ops._C, "sm70_fp16_candidate_scores_out", None)
    if operation is None:
        pytest.skip("requires the candidate scoring operator")
    assert operation is not None
    x = torch.ones(1, 5120, device="cuda", dtype=torch.float16)
    weight = torch.ones(32, 5120, device="cuda", dtype=torch.float16)
    ids = torch.tensor([[-1, 0, 31, 32]], device="cuda", dtype=torch.int32)
    scores = torch.empty_like(ids, dtype=torch.float16)
    operation(scores, x, weight, ids)
    assert torch.isnan(scores[0, [0, 3]]).all()
    assert torch.equal(scores[0, 1:3], torch.full_like(scores[0, 1:3], 5120))
