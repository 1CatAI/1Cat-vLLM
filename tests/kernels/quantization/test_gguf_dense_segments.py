# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Official GGUF numerical references and changed-input graph replay."""

import gguf
import numpy as np
import pytest
import torch

from vllm.model_executor.layers.quantization.gguf_dense_hmma_formats import (
    decode,
    pack,
    reconstruct,
)


def raw_weights(source, n=64, k=256):
    block, size = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(source)]
    rng = np.random.default_rng(1700 + source)
    data = rng.integers(0, 256, (n, k // block, size), dtype=np.uint8)
    offset = 208 if source == 14 else 0
    data[..., offset : offset + 2] = np.array([0.0007], dtype=np.float16).view(np.uint8)
    if source in (12, 13):
        data[..., 2:4] = np.array([0.0003], dtype=np.float16).view(np.uint8)
    return data.reshape(n, -1)


@pytest.mark.parametrize("source", [8, 12, 13, 14, 20, 23])
def test_codec_official_reference(source):
    raw = raw_weights(source)
    fmt, q, scale, minimum, group = decode(raw, source)
    reference = gguf.quants.dequantize(raw, gguf.GGMLQuantizationType(source))
    actual = reconstruct(fmt, q, scale, minimum, group).astype(np.float32)
    assert np.isfinite(actual).all()
    assert np.linalg.norm(actual - reference) / np.linalg.norm(reference) < 0.001
    codes, high, stats = pack(fmt, q, scale, minimum, group)
    assert codes.dtype == high.dtype == stats.dtype == np.uint8


@pytest.mark.parametrize("source", [8, 12, 13, 14, 20, 23])
@pytest.mark.parametrize("m,split", [(1, 1), (5, 2), (8, 4), (20, 2)])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_projection_official_and_graph(source, m, split):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 required")
    import vllm._C  # noqa: F401

    k = 512 if split == 4 else 256
    raw = raw_weights(source, k=k)
    fmt, q, scale, minimum, group = decode(raw, source)
    payload = [torch.from_numpy(p).cuda() for p in pack(fmt, q, scale, minimum, group)]
    x = torch.randn(m, k, device="cuda", dtype=torch.float16)
    output = torch.empty(m, 69, device="cuda", dtype=torch.float16)[:, 3:67]
    ws = torch.empty(
        2 * ((m + 7) // 8) * split * 256, device="cuda", dtype=torch.float32
    )
    counter = torch.zeros(2 * ((m + 7) // 8), device="cuda", dtype=torch.int32)
    official = torch.from_numpy(
        gguf.quants.dequantize(raw, gguf.GGMLQuantizationType(source))
    ).cuda()

    def run():
        torch.ops._C.gguf_dense_segments_sm70_out(
            x,
            [payload[0]],
            [payload[1]],
            [payload[2]],
            [output],
            [fmt],
            [64],
            k,
            split,
            4,
            ws,
            counter,
            None,
        )

    run()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for i in range(12):
        x.normal_(mean=i / 20)
        graph.replay()
        reference = x.float() @ official.T
        torch.accelerator.synchronize()
        error = (output.float() - reference).norm() / reference.norm()
        assert float(error) < 0.003
        assert not bool(counter.any())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_reject_m33_without_launch():
    import vllm._C  # noqa: F401

    x = torch.empty(33, 256, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="1..32"):
        torch.ops._C.gguf_dense_segments_sm70_out(
            x,
            [],
            [],
            [],
            [],
            [],
            [],
            256,
            1,
            4,
            torch.empty(1, device="cuda"),
            torch.empty(1, device="cuda", dtype=torch.int32),
            None,
        )


@pytest.mark.parametrize("source", [8, 12, 13, 14, 20, 23])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_transient_restore_matches_turbomind_storage(source):
    import vllm._C  # noqa: F401

    from vllm.model_executor.layers.quantization.gguf_turbomind import (
        prepare_gguf_projections,
    )

    raw = raw_weights(source)
    projection = prepare_gguf_projections(
        [(torch.from_numpy(raw).cuda(), source)], torch.float16, True, 8
    )[0]
    fmt, q, scale, minimum, group = decode(raw, source)
    payload = [torch.from_numpy(p).cuda() for p in pack(fmt, q, scale, minimum, group)]
    weight = torch.empty_like(projection.codes)
    stats = torch.empty_like(projection.stats)
    torch.ops._C.gguf_dense_restore_canonical_sm70_out(
        weight, stats, *payload, fmt, 256, 64
    )
    torch.testing.assert_close(weight, projection.codes, rtol=0, atol=0)
    torch.testing.assert_close(stats, projection.stats, rtol=0, atol=0)
