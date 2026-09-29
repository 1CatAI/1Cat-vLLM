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
