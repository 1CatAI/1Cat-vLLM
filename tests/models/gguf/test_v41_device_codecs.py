# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Portable tensor codecs must preserve the established NumPy lattice ABI."""

import gguf
import numpy as np
import pytest
import torch

from vllm.model_executor.layers.quantization.gguf_device_transcode import (
    lattice_storage,
)
from vllm.model_executor.layers.quantization.gguf_lattice_transcode import (
    transcode_lattice,
)


def source(kind, rows, k, scale, seed=41):
    block, unit = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(kind)]
    data = np.random.default_rng(seed).integers(
        0, 256, size=(rows, k // block, unit), dtype=np.uint8
    )
    d = np.full((rows, k // block), scale, dtype="<f2")
    if kind == 29:
        words = data[..., 48:].copy().view("<u2").reshape(rows, k // block, 4)
        bits = d.view("<u2")[..., None]
        words = (words & 0x0FFF) | (
            ((bits >> (4 * np.arange(4, dtype=np.uint16))) & 15) << 12
        )
        data[..., 48:] = words.astype("<u2").view(np.uint8).reshape(rows, k // block, 8)
    else:
        data[..., :2] = d.view(np.uint8).reshape(rows, k // block, 2)
    return data.reshape(rows, -1)


@pytest.mark.parametrize("kind", [16, 17, 18, 21, 22, 29])
@pytest.mark.parametrize("scale", [0.0, -0.0, 2**-24, 0.00137, -0.25])
def test_tensor_codec_matches_numpy_packets_and_stats_on_cpu(kind, scale):
    data = source(kind, 3, 768, scale)
    codes, stats = transcode_lattice(data, kind).mma884_storage()
    signed = {2: np.int16, 4: np.int32, 8: np.int64}[stats.itemsize]
    actual_codes, actual_stats = lattice_storage(torch.from_numpy(data), kind)
    np.testing.assert_array_equal(actual_codes.numpy(), codes)
    np.testing.assert_array_equal(actual_stats.numpy(), stats.view(signed))


@pytest.mark.parametrize("kind", [16, 17, 29])
def test_v41_gate_tp_rows_and_down_carriers_share_the_host_abi(kind):
    # Down's 576 logical columns are covered by three source blocks, with
    # offsets 0/64/128/192. Compare the entire carrier before native prepare.
    for k in (5120, 768):
        data = source(kind, 32, k, 0.00137)
        codes, stats = transcode_lattice(data, kind).mma884_storage()
        actual_codes, actual_stats = lattice_storage(torch.from_numpy(data), kind)
        signed = {2: np.int16, 4: np.int32}[stats.itemsize]
        np.testing.assert_array_equal(actual_codes.numpy(), codes)
        np.testing.assert_array_equal(actual_stats.numpy(), stats.view(signed))


@pytest.mark.parametrize("kind", [16, 17, 29])
def test_scale_overflow_is_rejected_before_kernel_dispatch(kind):
    data = source(kind, 2, 256, 60000)
    with pytest.raises(ValueError, match="overflows"):
        transcode_lattice(data, kind)
    with pytest.raises(ValueError, match="overflows"):
        lattice_storage(torch.from_numpy(data), kind)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA qualification required")
@pytest.mark.parametrize("kind", [16, 17, 29])
@pytest.mark.parametrize("k", [768, 5120])
def test_v41_cuda_codec_matches_host_before_device_admission(kind, k):
    data = source(kind, 32, k, 0.00137)
    codes, stats = transcode_lattice(data, kind).mma884_storage()
    actual_codes, actual_stats = lattice_storage(
        torch.from_numpy(data).to("cuda"), kind
    )
    signed = {2: np.int16, 4: np.int32}[stats.itemsize]
    np.testing.assert_array_equal(actual_codes.cpu().numpy(), codes)
    np.testing.assert_array_equal(actual_stats.cpu().numpy(), stats.view(signed))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA qualification required")
@pytest.mark.parametrize("kind", [16, 17, 29])
@pytest.mark.parametrize(
    "offset,gated", [(0, True), (0, False), (64, False), (128, False), (192, False)]
)
def test_v41_fused_transcode_preserves_packets_tp_offsets_and_graph_replay(
    kind, offset, gated
):
    from vllm import _custom_ops  # noqa: F401

    n, source_k, k = 32, 5120 if gated else 768, 5120 if gated else 576
    rows = n // 2 if gated else n
    parts = [
        source(kind, rows, source_k, 0.00137, seed=41 + part)
        for part in range(2 if gated else 1)
    ]
    projection = np.stack(parts, axis=1).reshape(n, -1) if gated else parts[0]
    decoded = transcode_lattice(projection, kind)
    codes, stats = decoded.mma884_storage()
    codes = codes[:, offset : offset + k].reshape(n, k // 8, 8).astype(np.uint16)
    shifts = np.array([0, 8, 2, 10, 4, 12, 6, 14], dtype=np.uint16)
    packets = np.bitwise_or.reduce(codes << shifts, axis=-1)
    expected_w = (
        np.ascontiguousarray(packets.reshape(n // 32, 32, k // 8).transpose(0, 2, 1))
        .view(np.uint8)
        .ravel()
    )
    group = decoded.group_size
    expected_s = (
        np.ascontiguousarray(stats[:, offset // group : (offset + k) // group].T)
        .view(np.uint8)
        .ravel()
    )
    device_parts = [torch.from_numpy(data).to("cuda") for data in parts]
    pointers = [
        torch.tensor([part.data_ptr()] * 2, dtype=torch.int64, device="cuda")
        for part in device_parts
    ]
    w_bytes, s_bytes = expected_w.nbytes, expected_s.nbytes
    weights = torch.full((2 * w_bytes,), 165, dtype=torch.uint8, device="cuda")
    metadata = torch.full((2 * s_bytes,), 165, dtype=torch.uint8, device="cuda")
    sequence = torch.tensor([1, 0], dtype=torch.int32, device="cuda")
    count = torch.zeros(1, dtype=torch.int32, device="cuda")
    error = torch.zeros(1, dtype=torch.int32, device="cuda")

    def run():
        torch.ops._C.gguf_prefill_transcode_sm70(
            weights,
            metadata,
            w_bytes,
            s_bytes,
            pointers[0],
            pointers[1] if gated else None,
            sequence,
            count,
            16,
            kind,
            n,
            k,
            offset,
            parts[0].shape[-1],
            error,
        )

    run()
    assert bool((weights == 165).all()) and bool((metadata == 165).all())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    count.fill_(1)
    graph.replay()
    assert error.item() == 0
    assert bool((weights[:w_bytes] == 165).all())
    assert bool((metadata[:s_bytes] == 165).all())
    np.testing.assert_array_equal(weights[w_bytes:].cpu().numpy(), expected_w)
    np.testing.assert_array_equal(metadata[s_bytes:].cpu().numpy(), expected_s)
