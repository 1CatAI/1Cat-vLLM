# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gguf
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (7, 0)
    or not hasattr(torch.ops._C, "sm70_dmv13_out"),
    reason="requires SM70 and sm70_dmv13_out",
)

BLOCK = {8: 34, 12: 144, 13: 176, 14: 210, 20: 18, 23: 136}


def _raw(rng, rows, k, qtype):
    size = BLOCK[qtype]
    block = 32 if qtype in (8, 20) else 256
    raw = rng.integers(0, 256, (rows, k // block, size), dtype=np.uint8)
    d = np.full((rows, k // block), 0.002, np.float16).view(np.uint8)
    if qtype in (12, 13):
        raw[..., 0:2] = d.reshape(rows, k // block, 2)
        raw[..., 2:4] = d.reshape(rows, k // block, 2)
    elif qtype == 14:
        raw[..., 208:210] = d.reshape(rows, k // block, 2)
    else:
        raw[..., :2] = d.reshape(rows, k // block, 2)
    return raw.reshape(rows, -1)


class _Shard:
    def __init__(self, raw, qtype):
        self.qweight = torch.from_numpy(raw)
        self.qweight_type = type("T", (), {"weight_type": qtype})()
        self.prefix = f"test.{qtype}.{raw.shape[0]}"


class _Extra:
    def __init__(self, weight):
        self.weight = weight


class _MergedBf16Extra:
    def __init__(self, weight):
        pieces = [part.contiguous().view(torch.uint8) for part in weight.chunk(2)]
        self.qweight = pieces[0]
        self.qweight.data_container = pieces
        self.qweight.shard_id = [0, 1]
        self.qweight.shard_id_map = {0: 0, 1: 1}
        self.qweight_type = type("T", (), {"shard_weight_type": {0: 30, 1: 30}})()


@pytest.mark.parametrize("qtype", [12, 14])
@pytest.mark.parametrize("tokens", [1, 5, 8])
@pytest.mark.parametrize("merged_bf16", [False, True])
def test_side_projection_matches_dequant(qtype, tokens, merged_bf16):
    from vllm.model_executor.layers.quantization.sm70_dmv13_projection import (
        _PROJECTIONS,
        Dmv13Projection,
    )

    rng = np.random.default_rng(qtype + tokens)
    n, k, extra_n = 256, 2560, 24
    raw = _raw(rng, n, k, qtype)
    extra = (torch.randn(extra_n, k, device="cuda") * 0.02).half()
    if merged_bf16:
        extra = extra.bfloat16()
    side = _MergedBf16Extra(extra) if merged_bf16 else _Extra(extra)
    proj = Dmv13Projection(_Shard(raw, qtype), side)
    assert proj.ready
    _PROJECTIONS[proj.name] = proj
    x = (torch.randn(tokens, k, device="cuda") * 0.5).half()
    out, extra_out = proj(x)
    w = gguf.quants.dequantize(raw, gguf.GGMLQuantizationType(qtype))
    ref = x.float().cpu() @ torch.from_numpy(w).float().T
    err = (out.float().cpu() - ref).norm() / ref.norm()
    assert float(err) < 2e-3
    torch.testing.assert_close(
        extra_out.float(), (x.float() @ extra.half().float().T), atol=2e-2, rtol=2e-2
    )


@pytest.mark.parametrize("qtype", [8, 12, 13, 14, 20, 23])
def test_shared_segment_planes_preserve_side_and_m20_graph(qtype):
    from types import SimpleNamespace

    from vllm.model_executor.layers.quantization.gguf_dense_hmma_formats import (
        decode,
        pack,
    )
    from vllm.model_executor.layers.quantization.sm70_dmv13_projection import (
        Dmv13Projection,
        _register_banks,
        share_prepared_banks,
    )

    n, k = 256, 2560
    raw = _raw(np.random.default_rng(3000 + qtype), n, k, qtype)
    fmt, codes, scales, minimum, group = decode(raw, qtype)
    payload = [
        torch.from_numpy(t).cuda() for t in pack(fmt, codes, scales, minimum, group)
    ]
    p = torch.nn.Module()
    p.codes, p.segment_high, p.stats = payload
    p.kernel = SimpleNamespace(config=SimpleNamespace(partition_weight_shape=(k, n)))
    p.source_output_sizes = (n,)
    p.segment_format, p.output_padding, p.input_layout_restored = fmt, 0, False
    layer = torch.nn.Module()
    layer.gguf_tm_projections = torch.nn.ModuleList([p])
    extra = (torch.randn(24, k, device="cuda") * 0.02).half()
    proj = Dmv13Projection(_Shard(raw, qtype), _Extra(extra))
    assert proj.ready
    layer.sm70_side_projection = proj
    _register_banks(layer, "sm70_side_projection", proj)
    x = torch.randn(5, k, device="cuda", dtype=torch.float16)
    before = tuple(t.clone() for t in proj.run(x))
    assert share_prepared_banks(layer) == 1
    after = proj.run(x)
    for got, expected in zip(after, before):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = proj.run(x)
    for _ in range(3):
        x.normal_()
        graph.replay()
        for got, expected in zip(captured, proj.run(x)):
            torch.testing.assert_close(got, expected, rtol=0, atol=0)
    # C4 still reads the same resident planes through the existing M20 route.
    x20 = torch.randn(20, k, device="cuda", dtype=torch.float16)
    y20 = torch.empty(20, n, device="cuda", dtype=torch.float16)
    workspace = torch.empty(8192, device="cuda", dtype=torch.float32)
    counters = torch.zeros(32, device="cuda", dtype=torch.int32)

    def run20():
        torch.ops._C.gguf_dense_segments_sm70_out(
            x20,
            [p.codes],
            [p.segment_high],
            [p.stats],
            [y20],
            [fmt],
            [n],
            k,
            1,
            4,
            workspace,
            counters,
            None,
        )

    run20()
    graph20 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph20):
        run20()
    for _ in range(3):
        x20.normal_()
        graph20.replay()
        got = y20.clone()
        run20()
        torch.testing.assert_close(got, y20, rtol=0, atol=0)
