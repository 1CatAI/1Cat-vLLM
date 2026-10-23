# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
from pathlib import Path

import numpy as np
import pytest

# The layout needs no native module or CUDA context, even in a source checkout.
_spec = importlib.util.spec_from_file_location(
    "iq2_xs_records",
    Path(__file__).parents[3]
    / "vllm/model_executor/layers/quantization/gguf_iq2_xs_records.py",
)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
pack_iq2_xs_records = _module.pack_iq2_xs_records


@pytest.mark.parametrize("n,blocks", [(32, 1), (64, 3), (96, 20)])
def test_every_original_bit_survives_independent_reader(n, blocks):
    rng = np.random.default_rng(1290)
    raw = rng.integers(0, 256, (n, blocks * 74), dtype=np.uint8)
    before = raw.copy()
    records = pack_iq2_xs_records(raw)
    assert records.dtype == np.uint8 and records.flags.c_contiguous
    assert records.nbytes == raw.nbytes
    np.testing.assert_array_equal(raw, before)
    recovered = np.empty_like(raw)
    for tile in range(n // 32):
        macro = tile * blocks * 32 * 74
        for block in range(blocks):
            for col in range(32):
                row = recovered[tile * 32 + col, block * 74 : (block + 1) * 74]
                d = macro + blocks * 2304 + block * 64 + col * 2
                row[:2] = records[d : d + 2]
                for half in range(2):
                    cursor = macro + (block * 2 + half) * 1152 + col * 16
                    assert cursor % 16 == 0
                    row[2 + half * 32 : 18 + half * 32] = records[cursor : cursor + 16]
                    row[18 + half * 32 : 34 + half * 32] = records[
                        cursor + 512 : cursor + 528
                    ]
                    scale = macro + (block * 2 + half) * 1152 + 1024 + col * 4
                    assert scale % 4 == 0
                    row[66 + half * 4 : 70 + half * 4] = records[scale : scale + 4]
    np.testing.assert_array_equal(recovered, raw)


def test_noncontiguous_tp_rows_keep_original_bytes():
    backing = np.arange(64 * 3 * 74 * 2, dtype=np.uint32).astype(np.uint8)
    raw = backing.reshape(64, 3 * 74 * 2)[:, ::2]
    assert not raw.flags.c_contiguous
    np.testing.assert_array_equal(
        pack_iq2_xs_records(raw), pack_iq2_xs_records(raw.copy())
    )


@pytest.mark.parametrize("shape", [(31, 74), (32, 99), (0, 74), (32, 0)])
def test_rejects_partial_quant_blocks_or_column_tiles(shape):
    with pytest.raises(ValueError):
        pack_iq2_xs_records(np.zeros(shape, dtype=np.uint8))


def test_rejects_converted_weights():
    with pytest.raises(TypeError):
        pack_iq2_xs_records(np.zeros((32, 74), dtype=np.float16))
