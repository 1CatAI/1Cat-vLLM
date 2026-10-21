# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm import _custom_ops as _ops  # noqa: F401
from vllm.models.qwen4_exp.nvidia.ops.qsa_partition_topk import qsa_partition_topk


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("rows", [5, 10])
@pytest.mark.parametrize("columns", [32768, 65600, 69632, 131072, 262144])
def test_partition_preserves_exact_selection_guards_and_dynamic_replay(rows, columns):
    torch.manual_seed(20261015)
    logits = torch.randn(rows, columns + 19, device="cuda")[:, :columns]
    lengths = torch.full((rows,), 8192, device="cuda", dtype=torch.int32)
    storage = torch.full((rows * 512 + 16,), -77, device="cuda", dtype=torch.int32)
    out = storage[8:-8].view(rows, 512)
    control = torch.empty_like(out)
    qsa_partition_topk(logits, lengths, out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        qsa_partition_topk(logits, lengths, out)
    for pattern, live in enumerate(
        (-1, 0, 511, 512, 513, 2305, 8191, 8192, 8193, 32768, 65536, columns)
    ):
        logits.normal_()
        if pattern % 4 == 1:
            logits.copy_(torch.arange(columns, device="cuda") % 7 - 3)
        elif pattern % 4 == 2:
            logits.copy_(1 + torch.rand_like(logits) / 100000)
        elif pattern % 4 == 3:
            logits.zero_()
            logits[:, ::2] = -0.0
        logits[:, 5::31] = float("inf")
        logits[:, 7::43] = -float("inf")
        row_lengths = [live + 17 * r for r in range(rows)]
        lengths.copy_(torch.tensor(row_lengths, device="cuda", dtype=torch.int32))
        graph.replay()
        torch.ops._C.qsa_lexicographic_topk(logits, lengths, control, 512, True)
        assert torch.equal(out, control)
        assert torch.all(storage[:8] == -77) and torch.all(storage[-8:] == -77)
        for row, raw in enumerate(row_lengths):
            n = min(max(raw, 0), columns)
            selected = torch.argsort(logits[row, :n], stable=True, descending=True)[
                :512
            ]
            expected = torch.full((512,), -1, device="cuda", dtype=torch.int32)
            expected[: selected.numel()] = selected.sort().values.to(torch.int32)
            assert torch.equal(out[row], expected)
