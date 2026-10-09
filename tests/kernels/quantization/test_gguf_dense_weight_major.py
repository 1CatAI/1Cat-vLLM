# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M20 weight reuse: official references, strided output and graph rewrites."""

import gguf
import numpy as np
import pytest
import torch

from vllm.model_executor.layers.quantization.gguf_dense_hmma_formats import decode, pack


def _raw(source, n, k):
    block, size = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(source)]
    rng = np.random.default_rng(2300 + source)
    data = rng.integers(0, 256, (n, k // block, size), dtype=np.uint8)
    offset = 208 if source == 14 else 0
    data[..., offset : offset + 2] = np.array([0.0007], np.float16).view(np.uint8)
    if source in (12, 13):
        data[..., 2:4] = np.array([0.0003], np.float16).view(np.uint8)
    return data.reshape(n, -1)


@pytest.mark.parametrize(
    "sources", [(8,), (12,), (13,), (14,), (20,), (23,), (12, 14), (23, 14, 13)]
)
@pytest.mark.parametrize("m", [5, 20])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_actual_k_changed_graph(sources, m):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 required")
    import vllm._C  # noqa: F401

    k = 2560
    n_shards = {1: [4096], 2: [2560, 1536], 3: [3072, 256, 256]}[len(sources)]
    widths, fmts, codes, high, scales, official = [], [], [], [], [], []
    for source, n in zip(sources, n_shards):
        raw = _raw(source, n, k)
        fmt, q, s, minimum, group = decode(raw, source)
        payload = [torch.from_numpy(p).cuda() for p in pack(fmt, q, s, minimum, group)]
        widths.append(n)
        fmts.append(fmt)
        codes.append(payload[0])
        high.append(payload[1])
        scales.append(payload[2])
        official.append(
            torch.from_numpy(
                gguf.quants.dequantize(raw, gguf.GGMLQuantizationType(source))
            ).cuda()
        )
    x = torch.randn(m, k, device="cuda", dtype=torch.float16)
    storage = torch.empty(m, sum(widths) + 7, device="cuda", dtype=torch.float16)
    out = storage[:, 3 : 3 + sum(widths)]
    views = list(out.split(widths, dim=1))
    tiles = sum((v + 31) // 32 for v in widths)
    ws = torch.full((tiles * ((m + 7) // 8) * 256,), float("nan"), device="cuda")
    counter = torch.zeros(tiles * ((m + 7) // 8), device="cuda", dtype=torch.int32)
    gate = torch.randn(m, device="cuda", dtype=torch.float16)

    def run():
        torch.ops._C.gguf_dense_segments_sm70_out(
            x, codes, high, scales, views, fmts, widths, k, 1, 4, ws, counter, gate
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for trial in range(4):
        x.normal_(mean=trial / 20)
        out.fill_(float("nan"))
        graph.replay()
        reference = torch.cat([x.float() @ w.T for w in official], dim=1).half()
        reference = (reference.float() * torch.sigmoid(gate.float())[:, None]).half()
        torch.accelerator.synchronize()
        assert torch.isfinite(out).all()
        assert (
            float((out.float() - reference.float()).norm() / reference.float().norm())
            < 0.003
        )
        assert not counter.any()


@pytest.mark.parametrize("source", [12, 13, 14])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_output_projection_k1536(source):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 required")
    import vllm._C  # noqa: F401

    raw = _raw(source, 2560, 1536)
    fmt, q, s, minimum, group = decode(raw, source)
    payload = [torch.from_numpy(p).cuda() for p in pack(fmt, q, s, minimum, group)]
    x = torch.randn(20, 1536, device="cuda", dtype=torch.float16)
    y = torch.empty(20, 2560, device="cuda", dtype=torch.float16)
    ws = torch.empty(80 * 3 * 256, device="cuda")
    counter = torch.zeros(80 * 3, device="cuda", dtype=torch.int32)
    torch.ops._C.gguf_dense_segments_sm70_out(
        x,
        [payload[0]],
        [payload[1]],
        [payload[2]],
        [y],
        [fmt],
        [2560],
        1536,
        1,
        8,
        ws,
        counter,
        None,
    )
    w = torch.from_numpy(
        gguf.quants.dequantize(raw, gguf.GGMLQuantizationType(source))
    ).cuda()
    reference = x.float() @ w.T
    assert float((y.float() - reference).norm() / reference.norm()) < 0.003
