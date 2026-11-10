# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gguf
import numpy as np
import pytest


@pytest.mark.parametrize("quant", [gguf.quants.IQ2_XS, gguf.quants.IQ2_XXS])
def test_shared_iq2_s_operand_formation_matches_original_formula(quant):
    quant.init_grid()
    grid = np.unique(quant.grid)
    assert grid.tolist() == [8, 25, 43]
    signs = np.frombuffer(gguf.quants.IQ2_XXS.ksigns, dtype=np.uint8)
    assert all(int(signs[i]) == (i | ((i.bit_count() & 1) << 7)) for i in range(128))
    d16 = np.arange(65536, dtype=np.uint16).view(np.float16)
    d16 = d16[np.isfinite(d16)]
    d32 = d16.astype(np.float32)
    # Check all finite original d patterns, every scale and every grid integer.
    # Only the final operand may round, including correct overflow/subnormals.
    with np.errstate(over="ignore"):
        for nibble in range(16):
            small = (np.float32(0.5) + np.float32(nibble)) * np.float32(0.25)
            for value in grid:
                for sign in (np.float32(1), np.float32(-1)):
                    exact_factor = np.float16(
                        np.float16(value * sign) * np.float16(small)
                    )
                    assert np.float32(exact_factor) == value * sign * small
                    formed = (d32 * np.float32(exact_factor)).astype(np.float16)
                    original = (
                        d32
                        * (np.float32(0.5) + np.float32(nibble))
                        * np.float32(0.25)
                        * value
                        * sign
                    ).astype(np.float16)
                    np.testing.assert_array_equal(
                        formed.view(np.uint16), original.view(np.uint16)
                    )
