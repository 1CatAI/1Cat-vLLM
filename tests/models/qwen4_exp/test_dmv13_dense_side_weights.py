# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization.sm70_dmv13_projection import _fp16_weight


def _layer(parts):
    tensors = [part.contiguous().view(torch.uint8) for part, _ in parts]
    raw = tensors[0]
    raw.data_container = tensors
    raw.shard_id = list(range(len(tensors)))
    raw.shard_id_map = {i: i for i in raw.shard_id}
    types = {i: kind for i, (_, kind) in enumerate(parts)}
    return SimpleNamespace(
        qweight=raw,
        qweight_type=SimpleNamespace(shard_weight_type=types),
    )


@pytest.mark.parametrize("kinds", [(1,), (30,), (30, 30), (1, 30)])
def test_merged_dense_rows_use_existing_fp16_weight_contract(kinds):
    parts = []
    for i, kind in enumerate(kinds):
        dtype = torch.bfloat16 if kind == 30 else torch.float16
        values = (torch.arange(24).reshape(3, 8).float() / 37 + i).to(dtype)
        parts.append((values, kind))
    got = _fp16_weight(_layer(parts))
    expected = torch.cat([part.half() for part, _ in parts])
    assert got is not None and got.dtype == torch.float16
    assert torch.equal(got.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("value", [65536.0, float("inf"), float("nan")])
def test_unrepresentable_bf16_weight_declines_fp16_fusion(value):
    parts = [(torch.full((3, 8), value, dtype=torch.bfloat16), 30)]
    assert _fp16_weight(_layer(parts)) is None


def test_mixed_width_rows_decline_fusion():
    assert (
        _fp16_weight(
            _layer([(torch.ones(3, 8).half(), 1), (torch.ones(3, 16).half(), 1)])
        )
        is None
    )


@pytest.mark.parametrize("source", [8, 12, 13, 14, 20, 23])
def test_side_projection_shares_prepared_segment_backing(source):
    import numpy as np

    from vllm.model_executor.layers.quantization import gguf_dmv13_dense as dense
    from vllm.model_executor.layers.quantization.gguf_dense_hmma_formats import pack
    from vllm.model_executor.layers.quantization.gguf_lut_transcode import (
        transcode_lut4,
    )
    from vllm.model_executor.layers.quantization.gguf_transcode import transcode_affine
    from vllm.model_executor.layers.quantization.sm70_dmv13_projection import (
        Dmv13Projection,
        _register_banks,
        share_prepared_banks,
    )
    from vllm.transformers_utils.gguf_tensor_reader import quant_size

    n, k = 96, 256
    block, size = quant_size(source)
    rng = np.random.default_rng(2300 + source)
    raw = rng.integers(0, 256, (n, k // block, size), dtype=np.uint8)
    offset = 208 if source == 14 else 0
    raw[..., offset : offset + 2] = np.array([0.0007], np.float16).view(np.uint8)
    if source in (12, 13):
        raw[..., 2:4] = np.array([0.0003], np.float16).view(np.uint8)
    raw = raw.reshape(n, -1)
    fmt, codes, scales, minimum, group = dense.decode(raw, source)
    canonical = (
        transcode_lut4(raw, source)
        if source in (20, 23)
        else transcode_affine(raw, source)
    )
    q = canonical.codes
    if source == 8:
        q = ((q.astype(np.int16) - 128) & 255).astype(np.uint8)
    packed = [
        torch.from_numpy(value)
        for value in pack(
            fmt,
            q,
            canonical.scales.astype(np.float32),
            canonical.mins.astype(np.float32) if minimum is not None else None,
            canonical.group_size,
        )
    ]
    p = torch.nn.Module()
    p.codes, p.segment_high, p.stats = packed
    p.kernel = SimpleNamespace(config=SimpleNamespace(partition_weight_shape=(k, n)))
    p.segment_format, p.output_padding, p.input_layout_restored = fmt, 0, False
    p.source_output_sizes = (32, 64)
    projection = object.__new__(Dmv13Projection)
    projection.ready, projection.k = True, k
    projection.widths, projection.formats = [32, 64], [fmt, fmt]
    projection.codes, projection.high, projection.scale = [], [], []
    for first, end in ((0, 32), (32, 96)):
        old = dense.pack(
            fmt,
            codes[first:end],
            scales[first:end],
            minimum[first:end] if minimum is not None else None,
            group,
        )
        if fmt == dense.LUT4:
            from vllm.model_executor.layers.quantization.gguf_dmv_formats import (
                compact_lut4_scale,
            )

            old = (*old[:2], compact_lut4_scale(old[2]))
        for family, value in zip(("codes", "high", "scale"), old):
            getattr(projection, family).append(torch.from_numpy(value))
    projection.workspace = torch.empty(16)
    projection.counters = torch.zeros(3, dtype=torch.int32)
    projection.extra = torch.empty(24, k, dtype=torch.float16)
    layer = torch.nn.Module()
    layer.gguf_tm_projections = torch.nn.ModuleList([p])
    layer.sm70_side_projection = projection
    _register_banks(layer, "sm70_side_projection", projection)
    old = {
        family: [x.clone() for x in getattr(projection, family)]
        for family in ("codes", "high", "scale")
    }
    assert share_prepared_banks(layer) == 1
    for family, reference in old.items():
        for actual, expected in zip(getattr(projection, family), reference):
            assert torch.equal(actual, expected)
    for family, backing in (("codes", p.codes), ("high", p.segment_high)):
        for actual in getattr(projection, family):
            assert (
                actual.untyped_storage().data_ptr()
                == backing.untyped_storage().data_ptr()
            )
    if fmt != dense.LUT4:
        assert (
            projection.scale[0].untyped_storage().data_ptr()
            == p.stats.untyped_storage().data_ptr()
        )
    assert layer._sm70_side_projection_codes_0 is projection.codes[0]
    # A reader with a restored input head layout must keep its original planes.
    p.input_layout_restored = True
    assert share_prepared_banks(layer) == 0
