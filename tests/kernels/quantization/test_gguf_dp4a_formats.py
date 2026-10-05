# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest
import torch

from vllm.model_executor.layers.quantization.gguf_dp4a_formats import (
    pack_mixed_u4_pair,
    transcode_integer_dot,
)
from vllm.model_executor.layers.quantization.gguf_layout import GGUFHeadTilingLayout
from vllm.transformers_utils.gguf_tensor_reader import dequantize, quant_size


@pytest.mark.parametrize("kinds", [(12, 23), (23, 12)])
def test_mixed_u4_pair_retains_packets_and_original_coefficients(kinds):
    codecs = []
    for kind in kinds:
        _, size = quant_size(kind)
        raw = np.random.default_rng(kind).integers(
            0, 256, (160, 10 * size), dtype=np.uint8
        )
        codecs.append(transcode_integer_dot(raw, kind))
    pair = pack_mixed_u4_pair(*codecs)
    for index, codec in enumerate(codecs):
        packed = codec.packed()
        rows = slice(160 * index, 160 * (index + 1))
        np.testing.assert_array_equal(pair[0][5 * index : 5 * (index + 1)], packed[0])
        for field in (1, 2):
            np.testing.assert_array_equal(pair[field][:, rows], packed[field])
        for field in (3, 4):
            if codec.source_type == 23:
                assert not pair[field][:, rows].any()
            else:
                np.testing.assert_array_equal(pair[field][:, rows], packed[field])


@pytest.mark.parametrize("kind", [12, 14, 23])
@pytest.mark.parametrize("k", [1536, 2560])
@pytest.mark.parametrize("head_tiled", [False, True])
def test_integer_dot_preserves_official_fp32_weights_and_packed_codes(
    kind, k, head_tiled
):
    n = 32
    _, size = quant_size(kind)
    rng = np.random.default_rng(kind + k)
    blocks = rng.integers(0, 256, (n, k // 256, size), dtype=np.uint8)
    d = rng.uniform(0.001, 0.02, blocks.shape[:2]).astype("<f2")
    start = 208 if kind == 14 else 0
    blocks[:, :, start : start + 2] = d[..., None].view(np.uint8)
    if kind == 12:
        blocks[:, :, 2:4] = (d * np.float16(0.5))[..., None].view(np.uint8)
    raw = blocks.reshape(n, -1)
    codec = transcode_integer_dot(raw, kind)
    np.testing.assert_array_equal(codec.dequantize(), dequantize(raw, kind))
    layout = GGUFHeadTilingLayout(2, 128) if head_tiled else None
    packed, original_d, small, dmin, mins = codec.packed(layout)

    def restored(array, head_dim):
        if layout is None:
            return array
        return layout.weight_to_vllm(
            torch.from_numpy(array), dim=1, head_dim=head_dim
        ).numpy()

    packets = packed.transpose(0, 2, 1).copy().reshape(n, -1)
    if kind == 14:
        decoded = packets.view(np.int8).reshape(n, k)
    else:
        decoded = (
            (
                packets[..., None].astype(np.uint32)
                >> (4 * np.arange(8, dtype=np.uint32))
            )
            & 15
        ).reshape(n, k)
    np.testing.assert_array_equal(decoded, restored(codec.codes, 128))
    np.testing.assert_array_equal(
        original_d.T,
        restored(
            codec.d.repeat(256 // codec.group_size, axis=1), 128 // codec.group_size
        ),
    )
    np.testing.assert_array_equal(
        small.T, restored(codec.small_scales, 128 // codec.group_size)
    )
    assert original_d.dtype == np.float16 and small.dtype == np.int8
    if kind == 12:
        np.testing.assert_array_equal(dmin.T, restored(codec.dmin.repeat(8, axis=1), 4))
        np.testing.assert_array_equal(mins.T, restored(codec.small_mins, 4))
    else:
        assert not dmin.size and not mins.size

    if kind == 23:
        import gguf

        values = np.asarray(gguf.quants.IQ4_NL.kvalues, dtype=np.float32)[decoded]
    else:
        values = decoded.astype(np.float32)
    reconstructed = values * (original_d.T.astype(np.float32) * small.T).repeat(
        codec.group_size, axis=1
    )
    if kind == 12:
        reconstructed -= (dmin.T.astype(np.float32) * mins.T).repeat(32, axis=1)
    np.testing.assert_array_equal(reconstructed, restored(dequantize(raw, kind), 128))

    row_codes, row_scales, _, row_mins, _ = codec.row_storage(layout)
    if kind == 14:
        row_values = row_codes.view(np.int8).astype(np.float32)
    else:
        nibble_words = row_codes.copy().view("<u4")
        indices = (
            (nibble_words[..., None] >> (4 * np.arange(8, dtype=np.uint32))) & 15
        ).reshape(n, k)
        if kind == 23:
            row_values = np.asarray(gguf.quants.IQ4_NL.kvalues, dtype=np.float32)[
                indices
            ]
        else:
            row_values = indices.astype(np.float32)
    row_reference = row_values * row_scales.repeat(codec.group_size, axis=1)
    if kind == 12:
        row_reference -= row_mins.repeat(32, axis=1)
    np.testing.assert_array_equal(row_reference, restored(dequantize(raw, kind), 128))
