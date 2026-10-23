# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent canonical-storage and fused-output oracles for DMV projections."""

import gguf
import numpy as np
import pytest
import torch

from vllm.model_executor.layers.quantization import gguf_dmv_formats as iq
from vllm.model_executor.layers.quantization.gguf_dense_hmma_formats import decode, pack
from vllm.model_executor.layers.quantization.gguf_lattice_transcode import (
    transcode_lattice,
)
from vllm.model_executor.layers.quantization.gguf_lut_transcode import transcode_lut4

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def planes(kind, k=512):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 required")
    import vllm._C  # noqa: F401

    n = 64
    size = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(kind)][1]
    raw = np.random.default_rng(kind).integers(
        0, 256, (n, k // 256, size), dtype=np.uint8
    )
    raw[..., :2] = np.array([0.0007], np.float16).view(np.uint8)
    if kind == 12:
        raw[..., 2:4] = np.array([0.0003], np.float16).view(np.uint8)
    raw = raw.reshape(n, -1)
    if kind in iq.IQ2_FORMATS:
        fmt, codes, scale = iq.pack_iq2(raw, kind)
        high = iq.iq2_reverse_table(kind)
    elif kind in (18, 21):
        fmt, codes, scale = iq.pack(raw, kind)
        high = np.empty(0, np.uint8)
    else:
        fmt, q, s, m, group = decode(raw, kind)
        codes, high, scale = pack(fmt, q, s, m, group)
        scale = iq.compact_lut4_scale(scale) if fmt == 3 else scale
    buffers = [torch.from_numpy(a).cuda() for a in (codes, high, scale)]
    reference = torch.from_numpy(
        gguf.quants.dequantize(raw, gguf.GGMLQuantizationType(kind))
    ).cuda()
    return fmt, buffers, reference, raw


@pytest.mark.parametrize("kind", [16, 17, 18, 21, 22, 23])
def test_restore_canonical_storage_exactly(kind):
    fmt, (codes, high, scale), reference, raw = planes(kind)
    if kind in (16, 17, 18, 21, 22):
        converted = transcode_lattice(raw, kind)
        c, s = converted.mma884_storage()
        original = torch.ops._C.gguf_lattice_sm70_prepare(
            torch.from_numpy(c).cuda(),
            torch.from_numpy(s.view(np.int64 if kind in (18, 21) else np.int32)).cuda(),
            kind,
            converted.group_size,
        )
    else:
        converted = transcode_lut4(raw, kind)
        original = torch.ops._C.gguf_lut4_sm70_prepare(
            torch.from_numpy(converted.codes).cuda(),
            torch.from_numpy(converted.scales).cuda(),
            0,
            32,
        )
    weight, stats = original[:2]
    restored_weight, restored_stats = torch.empty_like(weight), torch.empty_like(stats)
    if kind in iq.IQ2_FORMATS:
        torch.ops._C.gguf_dmv_restore_iq2_sm70_out(
            restored_weight, restored_stats, codes, scale, high, kind, 512, 64
        )
    else:
        torch.ops._C.gguf_dmv_restore_sm70_out(
            restored_weight, restored_stats, codes, scale, fmt, 512, 64
        )
    assert torch.equal(weight, restored_weight)
    assert torch.equal(stats, restored_stats)


@pytest.mark.parametrize(
    "gate,up", [(17, 16), (22, 21), (17, 18), (16, 22), (22, 17), (18, 22), (21, 22)]
)
def test_iq2_pair_preserves_native_m8_arithmetic(gate, up):
    from vllm.model_executor.layers.quantization.gguf_native_pair import _SOURCE_PACKERS

    ga, gb, _, gr = planes(gate, k=5120)
    ua, ub, _, ur = planes(up, k=5120)
    raw_gate = torch.from_numpy(_SOURCE_PACKERS[gate](gr)).cuda()
    raw_up = torch.from_numpy(_SOURCE_PACKERS[up](ur)).cuda()
    x = torch.randn(8, 5120, device="cuda", dtype=torch.float16)
    expected = torch.empty(8, 64, device="cuda", dtype=torch.float16)
    actual = torch.empty_like(expected)
    workspace = torch.empty(2048, device="cuda", dtype=torch.float32)
    counters = torch.zeros(2, device="cuda", dtype=torch.int32)
    table = torch.from_numpy(iq.tables()).cuda()

    def call():
        torch.ops._C.gguf_dmv_sm70_out(
            x,
            [gb[0], ub[0]],
            [gb[1], ub[1]],
            [gb[2], ub[2]],
            [actual, actual],
            [ga, ua],
            [64, 64],
            5120,
            1,
            8,
            workspace,
            counters,
            2,
            None,
            table,
            None,
            None,
            actual,
            False,
        )

    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for amplitude in (0.125, 1.0, 4.0):
        x.normal_().mul_(amplitude)
        torch.ops._C.gguf_native_pair_sm70_out(expected, x, raw_gate, raw_up, gate, up)
        call()
        assert torch.equal(actual, expected)
        graph.replay()
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("kind", [17, 22])
def test_iq2_down_preserves_native_m8_arithmetic(kind):
    from vllm.model_executor.layers.quantization.gguf_native_pair import _SOURCE_PACKERS

    fmt, buffers, _, raw = planes(kind, k=4352)
    records = torch.from_numpy(_SOURCE_PACKERS[kind](raw)).cuda()
    book = torch.empty(4096 if kind == 17 else 8192, device="cuda", dtype=torch.uint8)
    torch.ops._C.gguf_dmvq_book_sm70_out(book, kind)
    x = torch.randn(8, 4352, device="cuda", dtype=torch.float16)
    expected = torch.empty(8, 64, device="cuda", dtype=torch.float16)
    actual = torch.empty_like(expected)
    workspace = torch.empty(4096, device="cuda", dtype=torch.float32)
    counters = torch.zeros(2, device="cuda", dtype=torch.int32)
    for amplitude in (0.125, 1.0, 4.0):
        x.normal_().mul_(amplitude)
        torch.ops._C.gguf_dmvq_sm70_out(
            expected, x, records, workspace, counters, book, kind, 8, 1
        )
        torch.ops._C.gguf_dmv_sm70_out(
            x,
            [buffers[0]],
            [buffers[1]],
            [buffers[2]],
            [actual],
            [fmt],
            [64],
            4352,
            1,
            8,
            workspace,
            counters,
            2,
            None,
            None,
            None,
            None,
            None,
            False,
        )
        assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "gate,up",
    [(a, b) for a in (12, 18, 21, 23) for b in (12, 18, 21, 23)]
    + [(a, b) for a in (16, 17, 22) for b in (16, 17, 18, 21, 22, 23)]
    + [(a, b) for a in (18, 21, 23) for b in (16, 17, 22)],
)
@pytest.mark.parametrize("split", [1, 2])
def test_pair_official_and_changed_input_graph(gate, up, split):
    ga, gb, gw, _ = planes(gate)
    ua, ub, uw, _ = planes(up)
    x = torch.randn(8, 512, device="cuda", dtype=torch.float16)
    out = torch.empty(8, 64, device="cuda", dtype=torch.float16)
    workspace = torch.empty(2048, device="cuda", dtype=torch.float32)
    counters = torch.zeros(1, device="cuda", dtype=torch.int32)
    table = torch.from_numpy(iq.tables()).cuda()

    def call():
        torch.ops._C.gguf_dmv_sm70_out(
            x,
            [gb[0], ub[0]],
            [gb[1], ub[1]],
            [gb[2], ub[2]],
            [out, out],
            [ga, ua],
            [64, 64],
            512,
            split,
            4,
            workspace,
            counters,
            4,
            None,
            table,
            None,
            None,
            out,
            False,
        )

    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for _ in range(10):
        x.normal_()
        call()
        expected = out.clone()
        graph.replay()
        assert torch.equal(out, expected)
        g = (x.float() @ gw.T).half().float()
        u = (x.float() @ uw.T).half().float()
        reference = g * torch.sigmoid(g) * u
        assert (out.float() - reference).norm() / reference.norm() < 0.003
        assert not counters.any()


@pytest.mark.parametrize("kind", [18, 21, 23])
@pytest.mark.parametrize("k", [1536, 3072])
def test_gdn_heads_are_loaded_in_original_gguf_order(kind, k):
    fmt, buffers, weight, _ = planes(kind, k=k)
    x = torch.randn(8, k, device="cuda", dtype=torch.float16)
    out = torch.empty(8, 64, device="cuda", dtype=torch.float16)
    partials = torch.empty(1024, device="cuda", dtype=torch.float32)
    counters = torch.zeros(1, device="cuda", dtype=torch.int32)
    table = torch.from_numpy(iq.tables()).cuda()
    torch.ops._C.gguf_dmv_sm70_out(
        x,
        [buffers[0]],
        [buffers[1]],
        [buffers[2]],
        [out],
        [fmt],
        [64],
        k,
        1,
        4,
        partials,
        counters,
        2,
        None,
        table,
        None,
        None,
        None,
        True,
    )
    tiled = x.reshape(8, k // (3 * 128), 3, 128).transpose(1, 2).reshape(8, k)
    reference = tiled.float() @ weight.T
    assert (out.float() - reference).norm() / reference.norm() < 0.003


@pytest.mark.parametrize("kind", [12, 21])
def test_tp2_coalesced_bank_restores_large_canonical_fallback(kind):
    from vllm.model_executor.layers.quantization.gguf_turbomind import (
        GGUFPreparedProjection,
    )

    n, k = 17408, 5120
    size = gguf.GGML_QUANT_SIZES[kind][1]
    blocks = np.random.default_rng(kind).integers(
        0, 256, (n, k // 256, size), dtype=np.uint8
    )
    blocks[..., :2] = np.array([0.0007], np.float16).view(np.uint8)
    if kind == 12:
        blocks[..., 2:4] = np.array([0.0003], np.float16).view(np.uint8)
    raw = torch.from_numpy(blocks.reshape(n, -1)).cuda()
    control = GGUFPreparedProjection(raw, kind, torch.float16, True, 512)
    candidate = GGUFPreparedProjection(
        raw, kind, torch.float16, True, 512, dmv_enabled=True
    )
    assert hasattr(candidate, "dmv_format")
    for m in (1, 16):
        x = torch.randn(m, k, device="cuda", dtype=torch.float16)
        assert torch.equal(candidate(x), control(x))


@pytest.mark.parametrize("kind", [18, 21])
def test_large_scale_retains_canonical(kind):
    from types import SimpleNamespace

    from vllm.model_executor.layers.quantization.gguf_dmv import prepare_bank

    projection = SimpleNamespace(
        source_type=kind,
        output_padding=0,
        kernel=SimpleNamespace(
            config=SimpleNamespace(partition_weight_shape=(5120, 4352))
        ),
    )
    canonical = SimpleNamespace(scales=np.array([[64.0]], dtype=np.float16))
    raw = torch.empty((4352, 2200), dtype=torch.uint8, device="meta")
    assert not prepare_bank(projection, raw, canonical)
    assert (
        projection.dmv_rejection_reason == "iq3_scale_exceeds_decode_cancellation_range"
    )


def test_rejected_shard_rolls_back_every_plane(monkeypatch):
    from types import SimpleNamespace

    from vllm.model_executor.layers.quantization import gguf_turbomind as tm

    calls = []

    def fake(
        weight, kind, dtype, enabled, threshold, input_layout=None, dmv_enabled=False
    ):
        calls.append(dmv_enabled)
        result = SimpleNamespace(
            source_type=kind, kernel=object(), input_layout_restored=False
        )
        if dmv_enabled and kind == 21:
            result.dmv_format = 5
        elif dmv_enabled:
            result.dmv_rejection_reason = "rejected_test_scale"
        return result

    monkeypatch.setattr(tm, "GGUFPreparedProjection", fake)
    sources = [
        (torch.empty((32, size), dtype=torch.uint8, device="meta"), kind)
        for kind, size in ((21, 110), (18, 98))
    ]
    projections = tm.prepare_gguf_projections(
        sources, torch.float16, True, 8, dmv_enabled=True
    )
    assert calls == [True, True, False, False]
    assert all(not hasattr(p, "dmv_format") for p in projections)
    assert all(p.dmv_rejection_reasons == ["rejected_test_scale"] for p in projections)


def test_iq2_plane_policy_changes_graph_hash():
    from vllm.config.kernel import KernelConfig, Sm70GgufConfig

    enabled = KernelConfig(sm70_gguf=Sm70GgufConfig(iq2_signed_nibbles=True))
    disabled = KernelConfig(sm70_gguf=Sm70GgufConfig(iq2_signed_nibbles=False))
    assert enabled.compute_hash() != disabled.compute_hash()


@pytest.mark.parametrize("kind", [16, 17, 22])
def test_iq2_non_m8_retains_canonical_output_bitwise(kind):
    from vllm.model_executor.layers.quantization.gguf_turbomind import (
        GGUFPreparedProjection,
    )

    _, _, _, raw = planes(kind)
    weight = torch.from_numpy(raw).cuda()
    baseline = GGUFPreparedProjection(weight, kind, torch.float16, True, 512)
    candidate = GGUFPreparedProjection(
        weight, kind, torch.float16, True, 512, dmv_enabled=True
    )
    assert candidate.dmv_format == iq.IQ2_FORMATS[kind]
    for m in (1, 2, 4, 16, 32, 512):
        x = torch.randn(m, 512, device="cuda", dtype=torch.float16)
        assert torch.equal(baseline(x), candidate(x)), (kind, m)
