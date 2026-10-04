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
@pytest.mark.parametrize(
    "factor_scale,float_grid", [(False, False), (True, False), (True, True)]
)
def test_raw_dequant_and_vector_graph(kind, factor_scale, float_grid):
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
        for prefetch in (False, True):
            run = lambda splits=splits, prefetch=prefetch: (
                torch.ops._C.gguf_lattice_raw_vec_sm70_out(
                    out, x, w, kind, partial, splits, prefetch, factor_scale, float_grid
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
            [
                (split, variant, False, 0)
                for split in (1, 3)
                for variant in (False, True)
            ]
            if m <= 64
            else [(1, natural, False, 0) for natural in (False, True)]
        )
        if 1 < m <= 64 and n % 32 == 0:
            plans += [(split, True, True, 0) for split in (1, 3)]
        if m == 16:
            plans += [(split, True, False, 8) for split in (1, 3)]
            if n % 32 == 0:
                plans += [(split, True, True, 8) for split in (1, 3)]
        bounded_plans = [(*plan, False) for plan in plans]
        if m == 16:
            bounded_plans += [(*plan, True) for plan in plans if plan[3] == 0]
        algorithm_plans = [
            (*plan, algorithm)
            for plan in bounded_plans
            for algorithm in ((99, 102) if m == 512 else (99,))
        ]
        activation_plans = [(*plan, False) for plan in algorithm_plans]
        if m == 16:
            activation_plans += [
                (*plan, True)
                for plan in algorithm_plans
                if not plan[2] and plan[3] == 0 and not plan[4]
            ]
        for (
            split,
            variant,
            staged,
            row_tile,
            bounded,
            algorithm,
            stage_activation,
        ) in activation_plans:
            partials = torch.empty((split, m, n), dtype=torch.float32, device="cuda")
            if m == 512:
                scratch = torch.empty(
                    reference.shape if variant else reference.T.shape,
                    dtype=torch.float16,
                    device="cuda",
                )
                run = partial(
                    torch.ops._C.gguf_lattice_compact_blas_sm70_out,
                    out,
                    x,
                    weight,
                    kind,
                    scratch,
                    variant,
                    algorithm,
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
                    *([staged, row_tile, bounded, stage_activation] if m != 1 else []),
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
                expected_scratch = reference.half() if variant else reference.half().T
                torch.testing.assert_close(scratch, expected_scratch, rtol=0, atol=0)


@pytest.mark.parametrize("kind", [21, 22])
def test_all_codebook_entries_and_signs(kind):
    count = 512 if kind == 21 else 1024
    _, size = quant_size(kind)
    data = np.zeros((count * 2, size), dtype=np.uint8)
    data[:, :2] = np.array([1.0], dtype="<f2").view(np.uint8)
    for row in range(count * 2):
        index = row % count
        if kind == 21:
            data[row, 2:66] = index & 255
            data[row, 66:74] = 255 if index & 256 else 0
            data[row, 74:106] = 255 if row >= count else 0
        else:
            data[row, 2:34] = index & 255
            data[row, 66:74] = (index >> 8) * 85
            data[row, 34:66] = 255 if row >= count else 0
    raw = RawGGUFProjection.from_rows(data, kind)
    weight = torch.from_numpy(raw.data).cuda()
    compact = torch.empty((data.nbytes + 7) // 8 * 8, dtype=torch.uint8, device="cuda")
    torch.ops._C.gguf_lattice_compact_reorder_sm70_out(compact, weight, kind, 256)
    reference = torch.from_numpy(
        gguf.quants.dequantize(data, gguf.GGMLQuantizationType(kind))
    ).cuda()
    for layout, storage in (("raw", weight), ("compact", compact)):
        out = torch.empty_like(reference)
        getattr(torch.ops._C, f"gguf_lattice_{layout}_dequantize_sm70_out")(
            out, storage, kind
        )
        torch.testing.assert_close(out, reference, rtol=0, atol=0)
        rounded = torch.empty_like(reference, dtype=torch.float16)
        getattr(torch.ops._C, f"gguf_lattice_{layout}_dequantize_sm70_out")(
            rounded, storage, kind
        )
        torch.testing.assert_close(
            rounded.view(torch.int16),
            reference.half().view(torch.int16),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("kind", [21, 22])
def test_final_half_rounding_for_every_finite_block_scale(kind):
    # Include signed zero, every subnormal, overflow boundaries and both signs.
    bits = np.arange(65536, dtype=np.uint16)
    bits = bits[(bits & 0x7C00) != 0x7C00]
    count = len(bits)
    _, size = quant_size(kind)
    blocks = 2 if kind == 21 else 1
    data = np.zeros((count, blocks, size), dtype=np.uint8)
    data[:, :, :2] = bits.view(np.uint8).reshape(count, 1, 2)
    quant = getattr(gguf.quants, "IQ3_S" if kind == 21 else "IQ2_S")
    quant.init_grid()
    grid = quant.grid.reshape(-1, 4 if kind == 21 else 8)
    if kind == 21:
        indices = [
            int(np.flatnonzero(np.any(grid == value, axis=1))[0])
            for value in np.unique(grid)
        ]
        for block in range(blocks):
            for octet in range(32):
                first, second = indices[(2 * octet) % 8], indices[(2 * octet + 1) % 8]
                data[:, block, 2 + 2 * octet] = first & 255
                data[:, block, 3 + 2 * octet] = second & 255
                data[:, block, 66 + octet // 4] |= (
                    (first >> 8) << (2 * (octet % 4))
                ) | ((second >> 8) << (2 * (octet % 4) + 1))
            data[:, block, 74:106] = 0x55
            for byte in range(4):
                low = block * 8 + byte * 2
                data[:, block, 106 + byte] = low | ((low + 1) << 4)
    else:
        index = next(i for i, row in enumerate(grid) if len(np.unique(row)) == 3)
        data[:, :, 2:34] = index & 255
        data[:, :, 66:74] = (index >> 8) * 85
        data[:, :, 34:66] = 0x55
        for byte in range(8):
            data[:, :, 74 + byte] = 2 * byte | ((2 * byte + 1) << 4)
    data = data.reshape(count, -1)
    raw = RawGGUFProjection.from_rows(data, kind)
    weight = torch.from_numpy(raw.data).cuda()
    compact = torch.empty((data.nbytes + 7) // 8 * 8, dtype=torch.uint8, device="cuda")
    torch.ops._C.gguf_lattice_compact_reorder_sm70_out(
        compact, weight, kind, raw.logical_k
    )
    reference = (
        torch.from_numpy(gguf.quants.dequantize(data, gguf.GGMLQuantizationType(kind)))
        .cuda()
        .half()
    )
    for layout, storage in (("raw", weight), ("compact", compact)):
        out = torch.empty_like(reference)
        getattr(torch.ops._C, f"gguf_lattice_{layout}_dequantize_sm70_out")(
            out, storage, kind
        )
        torch.testing.assert_close(
            out.view(torch.int16), reference.view(torch.int16), rtol=0, atol=0
        )


@pytest.mark.parametrize("kind", [21, 22])
def test_compact_blas_cancellation_retains_fp32_partials(kind):
    _, size = quant_size(kind)
    n, k = 1536, 2560
    blocks = np.zeros((n, k // 256, size), dtype=np.uint8)
    blocks[:, :, :2] = np.array([1.0], dtype="<f2").view(np.uint8)
    data = blocks.reshape(n, -1)
    # Grid index zero and zero small-scale bits reconstruct exactly one.
    expected = gguf.quants.dequantize(data, gguf.GGMLQuantizationType(kind))
    assert np.array_equal(expected, np.ones((n, k), dtype=np.float32))
    raw = RawGGUFProjection.from_rows(data, kind)
    source = torch.from_numpy(raw.data).cuda()
    weight = torch.empty(data.nbytes, device="cuda", dtype=torch.uint8)
    torch.ops._C.gguf_lattice_compact_reorder_sm70_out(weight, source, kind, k)
    x = torch.full((512, k), 128.0, device="cuda", dtype=torch.float16)
    x[:, k // 2 :] = -128.0
    for natural in (False, True):
        scratch = torch.empty(
            (n, k) if natural else (k, n), device="cuda", dtype=torch.float16
        )
        for algorithm, dtype in [
            (algo, dtype)
            for algo in (99, 102)
            for dtype in (torch.float16, torch.float32)
        ]:
            out = torch.empty((512, n), device="cuda", dtype=dtype)
            run = partial(
                torch.ops._C.gguf_lattice_compact_blas_sm70_out,
                out,
                x,
                weight,
                kind,
                scratch,
                natural,
                algorithm,
            )
            for _ in range(3):
                run()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            # Intermediate positive/negative partials exceed FP16 range,
            # while the final dot product cancels exactly to zero.
            graph.replay()
            torch.testing.assert_close(out, torch.zeros_like(out), rtol=0, atol=0)
            x.neg_()
            graph.replay()
            torch.testing.assert_close(out, torch.zeros_like(out), rtol=0, atol=0)


@pytest.mark.parametrize("kind", [21, 22])
def test_compact_blas_fp32_output_and_final_cast_graph(kind):
    data = packed(kind, n=37, k=2560)
    raw = RawGGUFProjection.from_rows(data, kind)
    source = torch.from_numpy(raw.data).cuda()
    weight = torch.empty((data.nbytes + 7) // 8 * 8, device="cuda", dtype=torch.uint8)
    torch.ops._C.gguf_lattice_compact_reorder_sm70_out(weight, source, kind, 2560)
    reference = (
        torch.from_numpy(gguf.quants.dequantize(data, gguf.GGMLQuantizationType(kind)))
        .half()
        .cuda()
    )
    x = torch.randn((512, 2560), device="cuda", dtype=torch.float16)
    result = torch.empty((512, 37), device="cuda", dtype=torch.float32)
    out = torch.empty_like(result, dtype=torch.float16)
    for natural in (False, True):
        scratch = torch.empty(
            reference.shape if natural else reference.T.shape,
            device="cuda",
            dtype=torch.float16,
        )

        def run(scratch=scratch, natural=natural):
            torch.ops._C.gguf_lattice_compact_blas_sm70_out(
                result,
                x,
                weight,
                kind,
                scratch,
                natural,
            )
            out.copy_(result)

        for _ in range(3):
            run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        x.copy_(torch.randn_like(x))
        graph.replay()
        expected = x.float() @ reference.float().T
        torch.testing.assert_close(result, expected, rtol=5e-5, atol=0.001)
        torch.testing.assert_close(out, result.half(), rtol=0, atol=0)


@pytest.mark.parametrize("kind", [21, 22])
@pytest.mark.parametrize("cancellation", [False, True])
@pytest.mark.parametrize("dq_partitions", [1, 2, 4])
def test_compact_turbomind_fp16_workspace_graph(kind, cancellation, dq_partitions):
    n, k, m = (1536, 2560, 512) if cancellation else (64, 768, 512)
    data = packed(kind, n=n, k=k)
    if cancellation:
        data.fill(0)
        data.reshape(n, k // 256, -1)[:, :, :2] = np.array([1.0], dtype="<f2").view(
            np.uint8
        )
    expected = (
        torch.from_numpy(gguf.quants.dequantize(data, gguf.GGMLQuantizationType(kind)))
        .cuda()
        .half()
    )
    raw = RawGGUFProjection.from_rows(data, kind)
    source = torch.from_numpy(raw.data).cuda()
    weight = torch.empty(data.nbytes, device="cuda", dtype=torch.uint8)
    torch.ops._C.gguf_lattice_compact_reorder_sm70_out(weight, source, kind, k)
    scratch = torch.empty_like(expected)
    official = torch.empty_like(expected)
    torch.ops._C.gguf_workspace_f16_prepare_sm70_out(official, expected)
    oracle = expected.reshape(n // 32, 32, k // 8, 8).permute(0, 2, 1, 3)
    torch.testing.assert_close(official.flatten(), oracle.flatten(), rtol=0, atol=0)
    pointers, _ = torch.ops._C.awq_moe_build_strided_ptrs(
        scratch.unsqueeze(0), scratch.unsqueeze(0), k * 32, k * 32, 1
    )
    offsets = torch.tensor([0, m], device="cuda", dtype=torch.int32)
    x = torch.randn((m, k), device="cuda", dtype=torch.float16)
    if cancellation:
        x.fill_(128.0)
        x[:, k // 2 :] = -128.0
    out = torch.empty((m, n), device="cuda", dtype=torch.float16)
    run = partial(
        torch.ops._C.gguf_lattice_compact_tm_f16_sm70_out,
        out,
        x,
        weight,
        kind,
        scratch,
        offsets,
        pointers,
        dq_partitions,
    )
    blas_scratch = torch.empty((k, n), device="cuda", dtype=torch.float16)
    torch.ops._C.gguf_lattice_compact_blas_sm70_out(
        out, x, weight, kind, blas_scratch, False, 102, dq_partitions
    )
    torch.testing.assert_close(blas_scratch.T, expected, rtol=0, atol=0)
    for _ in range(3):
        run()
    torch.testing.assert_close(scratch, official, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(2):
        if cancellation:
            x.neg_()
        else:
            x.copy_(torch.randn_like(x))
        graph.replay()
        if cancellation:
            torch.testing.assert_close(out, torch.zeros_like(out), rtol=0, atol=0)
        else:
            torch.testing.assert_close(
                out.float(), x.float() @ expected.float().T, rtol=0.003, atol=0.01
            )
