# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact decode Top-K: row isolation, strides, ties and dynamic graph lengths."""

import pytest
import torch

from vllm import _custom_ops as _ops  # noqa: F401


@pytest.fixture(scope="module", autouse=True)
def require_sm70():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 required")


@pytest.mark.parametrize("rows", (1, 2, 3, 4, 8, 16, 17))
@pytest.mark.parametrize("padding", (0, 19))
def test_exact_batch_topk_dynamic_graph(rows, padding):
    torch.manual_seed(20927)
    columns = 65600  # Padded compressed-key capacity for a 256K context.
    storage = torch.empty(rows, columns + padding, device="cuda")
    logits = storage[:, :columns]
    lengths = torch.full((rows,), 2048, dtype=torch.int32, device="cuda")
    output_storage = torch.full(
        (rows * 512 + 16,), -77, dtype=torch.int32, device="cuda"
    )
    candidate = output_storage[8:-8].view(rows, 512)
    control = torch.empty_like(candidate)
    op = torch.ops._C.qsa_lexicographic_topk
    logits.normal_()
    op(logits, lengths, candidate, 512, True)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(logits, lengths, candidate, 512, True)

    for case, length in enumerate((-1, 0, 511, 512, 513, 2048, 2304, 2305, 65536)):
        logits.normal_()
        if case % 3 == 1:
            logits.copy_(torch.arange(columns, device="cuda") % 7 - 3)
        elif case % 3 == 2:
            logits.zero_()
            logits[:, ::2] = -0.0
        logits[:, 5::31] = -float("inf")
        logits[:, 7::43] = float("inf")
        # Neighbouring rows deliberately use different lengths/score order.
        row_lengths = [length + row * 17 for row in range(rows)]
        lengths.copy_(torch.tensor(row_lengths, dtype=torch.int32, device="cuda"))
        odd = torch.arange(rows, device="cuda") % 2 == 1
        logits.mul_(torch.where(odd, -1, 1)[:, None])
        candidate.fill_(-77)
        graph.replay()
        op(logits, lengths, control, 512)
        assert torch.equal(candidate, control)
        assert torch.all(output_storage[:8] == -77)
        assert torch.all(output_storage[-8:] == -77)
        # Stable score sort supplies an independent lower-index tie oracle;
        # the kernel emits the selected set in increasing index order.
        for row, raw_length in enumerate(row_lengths):
            n = min(max(raw_length, 0), columns)
            selected = torch.argsort(logits[row, :n], descending=True, stable=True)
            selected = selected[:512].sort().values.to(torch.int32)
            expected = torch.full((512,), -1, dtype=torch.int32, device="cuda")
            expected[: selected.numel()] = selected
            assert torch.equal(candidate[row], expected)


def test_empty_batch_topk():
    logits = torch.empty(0, 2048, device="cuda")
    lengths = torch.empty(0, dtype=torch.int32, device="cuda")
    output = torch.empty(0, 512, dtype=torch.int32, device="cuda")
    torch.ops._C.qsa_lexicographic_topk(logits, lengths, output, 512, True)


@pytest.mark.parametrize(
    "rows,columns",
    [(1, 32767), (4, 32768), (5, 8364), (20, 32844), (32, 65600), (33, 32844)],
)
def test_long_score_selection_keeps_exact_cutoff_and_order(rows, columns):
    """Narrow score bands and FP16 overflow cannot discard FP32 candidates."""
    torch.manual_seed(20261011)
    logits = torch.empty(rows, columns + 7, device="cuda")[:, :columns]
    lengths = torch.full((rows,), columns, device="cuda", dtype=torch.int32)
    output = torch.empty(rows, 512, device="cuda", dtype=torch.int32)
    op = torch.ops._C.qsa_lexicographic_topk
    logits.normal_()
    op(logits, lengths, output, 512)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(logits, lengths, output, 512)
    for distribution in (
        "random",
        "narrow",
        "overflow",
        "cutoff",
        "short",
        "uniform_prefix",
        "uniform_tail",
        "uniform_middle",
    ):
        logits.normal_()
        lengths.fill_(columns)
        if distribution == "narrow":
            logits.copy_(1 + torch.rand_like(logits) / 10000)
        elif distribution == "overflow":
            logits.copy_(70000 + torch.rand_like(logits) * 10000)
        elif distribution == "cutoff":
            logits.fill_(1)
            logits[:, 100:500] = float("inf")
        elif distribution == "short":
            lengths.copy_(torch.arange(rows, device="cuda", dtype=torch.int32) + 500)
        elif distribution == "uniform_prefix":
            logits[:, :1024] = 0
        elif distribution == "uniform_tail":
            logits[:, 1024:] = 10
        elif distribution == "uniform_middle":
            logits[:, 1024:-1024] = 10
        graph.replay()
        for row, length in enumerate(lengths.tolist()):
            chosen = (
                torch.argsort(logits[row, :length], descending=True, stable=True)[:512]
                .sort()
                .values.to(torch.int32)
            )
            expected = torch.full((512,), -1, device="cuda", dtype=torch.int32)
            expected[: chosen.numel()] = chosen
            assert torch.equal(output[row], expected), (distribution, row)


def test_independent_selection_graphs_keep_their_own_addresses():
    """A later, larger capture cannot redirect an earlier graph's buffers."""
    op = torch.ops._C.qsa_lexicographic_topk
    captures = []
    for rows, columns in ((1, 32844), (4, 65688)):
        logits = torch.arange(columns, device="cuda", dtype=torch.float32)
        logits = logits.repeat(rows, 1)
        lengths = torch.full((rows,), columns, device="cuda", dtype=torch.int32)
        output = torch.empty(rows, 512, device="cuda", dtype=torch.int32)
        op(logits, lengths, output, 512)
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            op(logits, lengths, output, 512)
        captures.append((graph, logits, lengths, output))
    for step in range(3):
        for graph, logits, lengths, output in reversed(captures):
            lengths.fill_(logits.shape[1] - step * 1000)
            graph.replay()
            expected = torch.arange(
                logits.shape[1] - step * 1000 - 512,
                logits.shape[1] - step * 1000,
                device="cuda",
                dtype=torch.int32,
            )
            assert torch.equal(output, expected.expand_as(output))
