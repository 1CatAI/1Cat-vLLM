# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
from pathlib import Path

import gguf
import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "dmv_formats",
    Path(__file__).parents[3]
    / "vllm/model_executor/layers/quantization/gguf_dmv_formats.py",
)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


@pytest.mark.parametrize("kind,size", [(18, 98), (21, 110)])
def test_indices_signs_and_subnormal_scales(kind, size):
    rng = np.random.default_rng(123)
    raw = rng.integers(0, 256, (64, size * 3), dtype=np.uint8)
    blocks = raw.reshape(64, 3, size)
    # Cover subnormals, normals and positive/negative zero explicitly.
    d = np.array([1, 3, 31, 255, 1024, 2048, 0, 0x8000], np.uint16)
    blocks[..., :2] = np.resize(d, (64, 3)).copy().view(np.uint8).reshape(64, 3, 2)
    fmt, codes, meta = _module.pack(raw, kind)
    groups = 6
    inverse = np.argsort(_module.ROWMAP)
    packets = codes.view(np.uint32).reshape(2, groups, 3, 32, 4)
    packets = packets.transpose(0, 3, 1, 2, 4)[:, inverse].reshape(64, groups, 3, 4)
    indices = np.concatenate((packets[:, :, 0], packets[:, :, 1]), axis=-1)
    indices = indices.copy().view(np.uint8).reshape(64, groups, 4, 8)
    sign_word = packets[:, :, 2]
    bits = np.arange(32, dtype=np.uint32)
    natural_shifts = bits // 2 + (bits % 2) * 16
    signs = ((sign_word[..., None] >> natural_shifts) & 1).reshape(64, -1)
    if kind == 21:
        metadata = meta.view(np.uint32).reshape(2, groups, 32, 2)
        metadata = metadata.transpose(0, 2, 1, 3)[:, inverse].reshape(64, groups, 2)
        high = metadata[..., 0]
        high = (high[..., None] >> np.arange(0, 32, 8, dtype=np.uint32)) & 255
        indices = indices.astype(np.uint16) | (
            ((high[..., None] >> np.arange(8)) & 1) << 8
        )
        scale_word = metadata[..., 1]
        coefficient = 1.0
    else:
        scale_word = meta.view(np.uint32).reshape(2, groups, 32)
        scale_word = scale_word.transpose(0, 2, 1)[:, inverse].reshape(64, groups)
        coefficient = 0.25
    nib = ((scale_word[..., None] >> np.arange(16, 32, 4)) & 15).astype(np.float32)
    base = (scale_word & 65535).astype(np.uint16).view(np.float16).astype(np.float32)
    scale = (base[..., None] * (1 + 2 * nib) * coefficient).astype(np.float16)
    cls = getattr(gguf.quants, gguf.GGMLQuantizationType(kind).name)
    cls.init_grid()
    grid = cls.grid.reshape(-1, 4)
    weight = grid[indices].reshape(64, -1).astype(np.float32) * (
        1 - 2 * signs.astype(np.int8)
    )
    restored = (weight.reshape(64, -1, 32) * scale.reshape(64, -1, 1)).reshape(64, -1)
    official = gguf.quants.dequantize(raw, gguf.GGMLQuantizationType(kind))
    # Group-scale FP16 rounding is the only accepted discrepancy; a wrong
    # index or sign cannot be hidden by a GEMM-level relative-error check.
    magnitude = np.max(np.abs(official), axis=1, keepdims=True)
    np.testing.assert_allclose(
        restored, official, atol=float(magnitude.max()) * 0.001 + 1e-6, rtol=0.001
    )
    if kind == 18:
        np.testing.assert_array_equal(
            (scale_word & 65535).astype(np.uint16),
            np.repeat(blocks[..., :2].copy().view(np.uint16)[..., 0], 2, axis=1),
        )


def test_compact_lut4_scale_is_lossless():
    half = np.arange(128, dtype=np.uint32)
    duplicated = (half | (half << 16)).view(np.uint8)
    compact = _module.compact_lut4_scale(duplicated)
    np.testing.assert_array_equal(compact.view(np.uint16), half.astype(np.uint16))
