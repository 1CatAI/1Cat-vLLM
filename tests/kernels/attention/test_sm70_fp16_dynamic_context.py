# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP16 dynamic context: same-KV FP64 references and live CUDA graphs."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform.is_device_capability(70),
    reason="requires SM70",
)


def _inputs(monkeypatch, context, rows, batch=1):
    from vllm.v1.attention.ops import sm70_fp16_grouped as fp16
    from vllm.v1.attention.ops import sm70_grouped as grouped

    call = fp16.load_grouped_fp16_fp32()
    assert call is not None
    monkeypatch.setattr(
        grouped,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(model_config=SimpleNamespace(max_model_len=context)),
    )
    assert fp16.grouped_context_limit() == context, "rebuild the FP16 grouped DSO"
    page = 832
    pages = (context + page - 1) // page
    torch.manual_seed(20261110 + rows + batch)
    storage = torch.randn(
        batch * pages, 2, page, 1, 256, device="cuda", dtype=torch.float16
    )
    k, v = storage.unbind(1)
    q = torch.randn(batch * rows, 6, 256, device="cuda", dtype=torch.float16)
    table = torch.randperm(batch * pages, device="cuda").int().reshape(batch, pages)
    lengths = torch.arange(
        context - rows + 1, context + 1, device="cuda", dtype=torch.int32
    ).repeat(batch)
    return call, (q, k, v, table, lengths, torch.empty_like(q))


def _oracle(inputs):
    q, k, v, table, lengths, actual = inputs
    expected = torch.empty_like(q, dtype=torch.float64)
    rows = q.shape[0] // table.shape[0]
    for request in range(table.shape[0]):
        section = slice(request * rows, (request + 1) * rows)
        limit = int(lengths[section].max())
        if limit == 0:
            expected[section].zero_()
            continue
        key = k[table[request].long()].reshape(-1, 256)[:limit].double()
        value = v[table[request].long()].reshape(-1, 256)[:limit].double()
        scores = (q[section].double() @ key.T) * 0.0625
        scores.masked_fill_(
            torch.arange(limit, device="cuda")[None, None]
            >= lengths[section, None, None],
            -torch.inf,
        )
        expected[section] = scores.softmax(-1).nan_to_num(0) @ value
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.double(), expected, rtol=0.03, atol=3e-4)
    relative = float(
        (actual.double() - expected).norm() / expected.norm().clamp_min(1e-30)
    )
    assert relative < 0.007
    return relative


def _live_graph(call, inputs, context):
    def run(out=inputs[-1]):
        return call(*inputs[:-1], out=out, softmax_scale=0.0625)

    run()
    error = _oracle(inputs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    eager = torch.empty_like(inputs[-1])
    for length in (min(262145, context), 0, context):
        inputs[0].normal_(0, 0.7)
        inputs[3].copy_(inputs[3].flip(1))
        inputs[4].fill_(length)
        run(eager)
        inputs[-1].fill_(float("nan"))
        graph.replay()
        assert torch.equal(inputs[-1], eager)
        if length == 0:
            assert torch.count_nonzero(inputs[-1]) == 0
    _oracle(inputs)
    print(f"context={context} rows={inputs[0].shape[0]} relative_l2={error:.8g}")


@pytest.mark.parametrize("context", [32768, 262144, 524288, 1048576])
@pytest.mark.parametrize("rows", range(2, 9))
def test_fp16_dynamic_fp64_and_live_graph(monkeypatch, context, rows):
    call, inputs = _inputs(monkeypatch, context, rows)
    _live_graph(call, inputs, context)


@pytest.mark.parametrize("context", [266240, 266241, 832 * 321 + 1, 1048593, 2097152])
def test_fp16_page_and_old_capacity_boundaries(monkeypatch, context):
    call, inputs = _inputs(monkeypatch, context, 8)
    _live_graph(call, inputs, context)


@pytest.mark.parametrize("batch", [2, 4])
def test_fp16_independent_request_and_padding_graphs(monkeypatch, batch):
    call, inputs = _inputs(monkeypatch, 524288, 8, batch=batch)
    inputs[4][8:] = 32768
    inputs[4][-1] = 0
    call(*inputs[:-1], out=inputs[-1], softmax_scale=0.0625)
    _oracle(inputs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call(*inputs[:-1], out=inputs[-1], softmax_scale=0.0625)
    inputs[4][-8:] = 0
    graph.replay()
    _oracle(inputs)
    inputs[4][-8:] = 524288
    graph.replay()
    _oracle(inputs)


def test_fp16_reads_suffix_beyond_one_million(monkeypatch):
    context = 2097152
    call, inputs = _inputs(monkeypatch, context, 8)
    q, k, v, table, lengths, out = inputs
    q.zero_()
    v.zero_()
    # Place the boundary on a physical page to preserve its known suffix mean.
    start = ((1048576 + 831) // 832) * 832
    v[table[0, start // 832 :].long()] = 1
    lengths.fill_(context)
    call(q, k, v, table, lengths, out=out, softmax_scale=0.0625)
    torch.testing.assert_close(
        out, torch.full_like(out, (context - start) / context), rtol=0, atol=0
    )
