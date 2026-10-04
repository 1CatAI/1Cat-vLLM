# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gguf
import numpy as np
import pytest
import torch
import vllm._C  # noqa: F401

from vllm.model_executor.layers.quantization.gguf_raw import RawGGUFProjection
from vllm.transformers_utils.gguf_tensor_reader import quant_size


def packed(kind, n=7, k=768):
    block, size = quant_size(kind)
    rng = np.random.default_rng(8431 + kind)
    data = rng.integers(0, 256, (n, k // block, size), dtype=np.uint8)
    scales = rng.uniform(0.005, 0.03, (n, k // block)).astype("<f2")
    data[:, :, :2] = scales[..., None].view(np.uint8)
    return data.reshape(n, -1)


@pytest.mark.parametrize("kind", [21, 22])
def test_original_payload_and_tp_slicing(kind):
    data = packed(kind, n=12, k=1024)
    raw = RawGGUFProjection.from_rows(data, kind)
    assert raw.shape == (12, 1024)
    assert raw.padding_bytes_per_row <= 7
    assert raw.storage_bytes == 12 * ((data.shape[1] + 7) // 8 * 8)
    assert np.array_equal(raw.data[:, : data.shape[1]], data)
    for axis in (0, 1):
        shards = [raw.tp_slice(rank, 4, axis=axis) for rank in range(4)]
        payload = [s.data[:, : s.payload_bytes_per_row] for s in shards]
        assert np.array_equal(np.concatenate(payload, axis=axis), data)
        assert all(s.padding_bytes_per_row <= 7 for s in shards)
    with pytest.raises(ValueError, match="cuts a source block"):
        RawGGUFProjection.from_rows(packed(kind, k=768), kind).tp_slice(0, 4, axis=1)


@pytest.mark.parametrize("kind", [21, 22])
def test_raw_dequant_and_vector_graph(kind):
    data = packed(kind)
    raw = RawGGUFProjection.from_rows(data, kind)
    w = torch.from_numpy(raw.data).cuda()
    expected = torch.from_numpy(
        gguf.quants.dequantize(data, gguf.GGMLQuantizationType(kind))
    ).cuda()
    fp32 = torch.empty_like(expected)
    torch.ops._C.gguf_lattice_raw_dequantize_sm70_out(fp32, w, kind)
    torch.testing.assert_close(fp32, expected, rtol=0, atol=0)
    fp16 = torch.empty_like(expected, dtype=torch.float16)
    torch.ops._C.gguf_lattice_raw_dequantize_sm70_out(fp16, w, kind)
    torch.testing.assert_close(fp16, expected.half(), rtol=0, atol=0)
    x = torch.randn(1, raw.logical_k, device="cuda", dtype=torch.float16)
    out = torch.empty((1, raw.shape[0]), device="cuda", dtype=torch.float16)
    partial = torch.empty((3, raw.shape[0]), device="cuda", dtype=torch.float32)
    for splits in (1, 3):
        run = lambda splits=splits: torch.ops._C.gguf_lattice_raw_vec_sm70_out(
            out, x, w, kind, partial, splits
        )
        for _ in range(3):
            run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        x.copy_(torch.randn_like(x))
        graph.replay()
        torch.testing.assert_close(
            out.float(), x.float() @ expected.T, rtol=0.001, atol=0.002
        )


@pytest.mark.parametrize("kind", [21, 22])
@pytest.mark.parametrize("m", [5, 8, 16])
def test_raw_mma_matches_fp32_accumulation_and_graph(kind, m):
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    data = packed(kind, n=37)
    raw = RawGGUFProjection.from_rows(data, kind)
    w = torch.from_numpy(raw.data).cuda()
    reference = (
        torch.from_numpy(gguf.quants.dequantize(data, gguf.GGMLQuantizationType(kind)))
        .half()
        .cuda()
    )
    x = torch.randn(m, raw.logical_k, device="cuda", dtype=torch.float16)
    out = torch.empty((m, raw.shape[0]), device="cuda", dtype=torch.float16)
    partial = torch.empty((3, m, raw.shape[0]), device="cuda", dtype=torch.float32)
    for tile in (8, 32):
        for splits in (1, 3):
            run = lambda splits=splits, tile=tile: (
                torch.ops._C.gguf_lattice_raw_mma_sm70_out(
                    out, x, w, kind, partial, splits, tile
                )
            )
            for _ in range(3):
                run()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            x.copy_(torch.randn_like(x))
            graph.replay()
            torch.testing.assert_close(
                out.float(), x.float() @ reference.float().T, rtol=0.003, atol=0.01
            )


@pytest.mark.parametrize("kind", [21, 22])
def test_raw_blas_graph(kind):
    data = packed(kind)
    raw = RawGGUFProjection.from_rows(data, kind)
    w = torch.from_numpy(raw.data).cuda()
    reference = (
        torch.from_numpy(gguf.quants.dequantize(data, gguf.GGMLQuantizationType(kind)))
        .half()
        .cuda()
    )
    x = torch.randn(512, raw.logical_k, device="cuda", dtype=torch.float16)
    out = torch.empty((512, raw.shape[0]), device="cuda", dtype=torch.float16)
    scratch = torch.empty(reference.T.shape, dtype=torch.float16, device="cuda")
    run = lambda: torch.ops._C.gguf_lattice_raw_blas_sm70_out(out, x, w, kind, scratch)
    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    x.copy_(torch.randn_like(x))
    graph.replay()
    torch.testing.assert_close(scratch, reference.T, rtol=0, atol=0)
    torch.testing.assert_close(
        out.float(), x.float() @ reference.float().T, rtol=0.003, atol=0.01
    )
