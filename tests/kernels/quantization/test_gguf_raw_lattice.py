# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial

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


def restore_compact_bytes(storage, kind, n, k):
    """Independent bitstream inverse: verify every source bit survives."""
    block_bytes, bits, small_bytes = (110, 26, 4) if kind == 21 else (82, 18, 8)
    blocks = k // 256
    restored = np.zeros((n, blocks, block_bytes), dtype=np.uint8)
    for first in range(0, n, 32):
        width = min(32, n - first)
        packet_bytes = width * bits * 4
        for block in range(blocks):
            start = first * blocks * block_bytes + block * width * block_bytes
            tile = storage[start : start + width * block_bytes]
            stream = int.from_bytes(tile[:packet_bytes].tobytes(), "little")
            target = restored[first : first + width, block]
            target[:, :2] = tile[packet_bytes : packet_bytes + width * 2].reshape(
                width, 2
            )
            target[:, 106 if kind == 21 else 74 :] = tile[
                packet_bytes + width * 2 :
            ].reshape(width, small_bytes)
            for octet in range(32):
                for col in range(width):
                    packet = (stream >> ((octet * width + col) * bits)) & (
                        (1 << bits) - 1
                    )
                    if kind == 21:
                        target[col, 2 + octet * 2] = packet & 255
                        target[col, 3 + octet * 2] = (packet >> 9) & 255
                        target[col, 66 + octet // 4] |= (
                            ((packet >> 8) & 1) | (((packet >> 17) & 1) << 1)
                        ) << (2 * (octet % 4))
                        target[col, 74 + octet] = packet >> 18
                    else:
                        target[col, 2 + octet] = packet & 255
                        target[col, 66 + octet // 4] |= ((packet >> 8) & 3) << (
                            2 * (octet % 4)
                        )
                        target[col, 34 + octet] = packet >> 10
    return restored.reshape(n, -1)


@pytest.mark.parametrize("kind", [21, 22])
@pytest.mark.parametrize("n", [1, 7, 37, 64])
def test_equal_byte_gpu_reorder_and_compact_graph(kind, n):
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    data = packed(kind, n=n)
    raw = RawGGUFProjection.from_rows(data, kind)
    source = torch.from_numpy(raw.data).cuda()
    count = (data.nbytes + 7) // 8 * 8
    weight = torch.empty(count, dtype=torch.uint8, device="cuda")
    torch.ops._C.gguf_lattice_compact_reorder_sm70_out(
        weight, source, kind, raw.logical_k
    )
    encoded = weight.cpu().numpy()
    assert count - data.nbytes <= 7
    assert np.array_equal(restore_compact_bytes(encoded, kind, n, raw.logical_k), data)
    assert not encoded[data.nbytes :].any()
    reference = torch.from_numpy(
        gguf.quants.dequantize(data, gguf.GGMLQuantizationType(kind))
    ).cuda()
    decoded = torch.empty_like(reference)
    torch.ops._C.gguf_lattice_compact_dequantize_sm70_out(decoded, weight, kind)
    torch.testing.assert_close(decoded, reference, rtol=0, atol=0)
    for m in (1, 5, 8, 16, 512):
        x = torch.randn(m, raw.logical_k, device="cuda", dtype=torch.float16)
        out = torch.empty((m, n), device="cuda", dtype=torch.float16)
        scratch = torch.empty(reference.T.shape, dtype=torch.float16, device="cuda")
        plans = (
            [(split, variant) for split in (1, 3) for variant in (False, True)]
            if m <= 64
            else [(1, False)]
        )
        for split, variant in plans:
            partials = torch.empty((split, m, n), dtype=torch.float32, device="cuda")
            if m == 512:
                run = partial(
                    torch.ops._C.gguf_lattice_compact_blas_sm70_out,
                    out,
                    x,
                    weight,
                    kind,
                    scratch,
                )
            else:
                op = (
                    torch.ops._C.gguf_lattice_compact_vec_sm70_out
                    if m == 1
                    else torch.ops._C.gguf_lattice_compact_mma_sm70_out
                )
                run = partial(
                    op,
                    out,
                    x,
                    weight,
                    kind,
                    partials,
                    split,
                    variant,
                )
            for _ in range(3):
                run()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            x.copy_(torch.randn_like(x))
            graph.replay()
            expected = x.float() @ (reference if m == 1 else reference.half().float()).T
            torch.testing.assert_close(out.float(), expected, rtol=0.003, atol=0.01)
            if m == 512:
                torch.testing.assert_close(scratch, reference.half().T, rtol=0, atol=0)
