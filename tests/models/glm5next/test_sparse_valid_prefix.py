# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exercise sparse FP16 attention as device-side valid lengths change."""

import pytest
import torch

from vllm.models.deepseek_v4.common.portable_sparse import use_portable_sparse
from vllm.models.deepseek_v4.sm70.sparse_kernels import (
    sm70_sparse_attention_gathered,
)

pytestmark = pytest.mark.skipif(
    not torch.accelerator.is_available() or not use_portable_sparse(),
    reason="requires an admitted portable sparse CUDA device",
)


def _reference(q, kv, indices, lengths, sink):
    rows = []
    for row, length in enumerate(lengths.cpu().tolist()):
        selected_ids = indices[row, : max(0, min(length, indices.shape[1]))]
        selected_ids = selected_ids[(selected_ids >= 0) & (selected_ids < len(kv))]
        selected = kv[selected_ids.long()].float()
        if len(selected) == 0:
            rows.append(torch.zeros_like(q[row]))
            continue
        scores = q[row].float() @ selected.T * 512**-0.5
        if sink is not None:
            scores = torch.cat((scores, sink.float()[:, None]), dim=-1)
        probs = torch.softmax(scores, dim=-1)[:, : len(selected)]
        rows.append((probs @ selected).half())
    return torch.stack(rows)


@pytest.mark.parametrize("width", [35, 2048])
@pytest.mark.parametrize("has_sink", [False, True])
def test_sparse_graph_updates_valid_prefix(width, has_sink):
    """A captured empty request must handle growing, shrinking and padded rows."""
    torch.manual_seed(23)
    device = torch.device("cuda")
    tokens, heads = 5, 13
    # Nontrivial row strides and a partial head tile exercise pointer arithmetic.
    q = torch.randn(tokens, heads + 2, 512, device=device, dtype=torch.float16)
    q = q[:, :heads]
    kv = torch.randn(4096, 512, device=device, dtype=torch.float16) * 0.2
    indices = torch.randint(
        0, len(kv), (tokens, width + 7), device=device, dtype=torch.int32
    )[:, :width]
    lengths = torch.zeros(tokens, device=device, dtype=torch.int32)
    sink = torch.linspace(-2, 2, heads, device=device).half() if has_sink else None
    out = torch.empty_like(q)
    eager = torch.empty_like(q)

    def run(destination):
        sm70_sparse_attention_gathered(
            q, kv, indices, lengths, 512**-0.5, sink, destination
        )

    run(out)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(out)

    for valid_lengths in (
        [0, 1, 15, 16, 17],
        [width, width - 1, 33, width + 32, -1],
        [17, 0, 1, 32, width],
    ):
        q.normal_(std=0.2)
        kv.normal_(std=0.2)
        indices.random_(0, len(kv))
        # Invalid indices inside the valid prefix must stay masked as before.
        indices[:, 3] = -1
        indices[:, 8] = len(kv) + 1
        lengths.copy_(torch.tensor(valid_lengths, dtype=torch.int32, device=device))
        graph.replay()
        run(eager)
        torch.accelerator.synchronize()
        assert torch.equal(out, eager)
        torch.testing.assert_close(
            out, _reference(q, kv, indices, lengths, sink), rtol=2e-2, atol=2e-3
        )
        assert torch.isfinite(out).all()


def test_sparse_masked_tail_is_irrelevant():
    """Inactive slots can contain valid KV addresses without affecting output."""
    torch.manual_seed(29)
    q = torch.randn(3, 64, 512, dtype=torch.float16, device="cuda") * 0.2
    kv = torch.randn(4096, 512, dtype=torch.float16, device="cuda") * 0.2
    indices = torch.randint(0, len(kv), (3, 2048), dtype=torch.int32, device="cuda")
    lengths = torch.tensor([1, 17, 257], dtype=torch.int32, device="cuda")
    first, second = torch.empty_like(q), torch.empty_like(q)
    sm70_sparse_attention_gathered(q, kv, indices, lengths, 512**-0.5, None, first)
    for row, count in enumerate([1, 17, 257]):
        indices[row, count:] = -1
    sm70_sparse_attention_gathered(q, kv, indices, lengths, 512**-0.5, None, second)
    torch.accelerator.synchronize()
    assert torch.equal(first, second)
