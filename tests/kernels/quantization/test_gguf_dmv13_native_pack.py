# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The native GGUF->dmv13 packer must be byte-identical to the NumPy one."""

import numpy as np
import pytest
import torch

from vllm.model_executor.layers.quantization import gguf_dmv13_dense as dense
from vllm.model_executor.layers.quantization import sm70_dmv13_projection as proj
from vllm.model_executor.layers.quantization.gguf_dmv_formats import (
    compact_lut4_scale,
)

# gtype -> (block bytes, offsets of FP16 scale fields within a block)
BLOCKS = {
    8: (34, (0,)),
    12: (144, (0, 2)),
    13: (176, (0, 2)),
    14: (210, (208,)),
    20: (18, (0,)),
    23: (136, (0,)),
}

pytestmark = pytest.mark.skipif(
    proj._native_packer() is None, reason="native dmv13 packer not built"
)


def _raw(gtype, rows, blocks, seed):
    rng = np.random.default_rng(seed)
    block_bytes, halves = BLOCKS[gtype]
    raw = rng.integers(0, 256, (rows, blocks, block_bytes), dtype=np.uint8)
    for off in halves:  # finite FP16 scales (NaN payloads are not compared)
        vals = rng.uniform(-2.0, 2.0, (rows, blocks)).astype(np.float16)
        raw[..., off : off + 2] = vals.view(np.uint8).reshape(rows, blocks, 2)
    return raw.reshape(rows, -1)


def _reference(raw, gtype):
    fmt, q, s, m, gs = dense.decode(raw, gtype)
    codes, high, scale = dense.pack(fmt, q, s, m, gs)
    if fmt == dense.LUT4:
        scale = compact_lut4_scale(scale)
    return fmt, q.shape[0], q.shape[1], codes, high, scale


@pytest.mark.parametrize("gtype", sorted(BLOCKS))
@pytest.mark.parametrize("rows", [32, 40, 96])
@pytest.mark.parametrize("threads", [1, 4])
def test_native_pack_matches_numpy(gtype, rows, threads):
    blocks = 2 if BLOCKS[gtype][0] > 40 else 12  # covers K padding to 128
    raw = _raw(gtype, rows, blocks, seed=gtype * 1000 + rows)
    expected = _reference(raw, gtype)
    got = proj._decode_pack(torch.from_numpy(raw), gtype, threads=threads)
    assert got[:3] == expected[:3]
    for a, b, name in zip(got[3:], expected[3:], ("codes", "high", "scale")):
        assert a.dtype == np.uint8 and a.shape == b.shape, name
        assert np.array_equal(a, b), name
