# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVQuantMode,
    get_kv_quant_mode,
)

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize("mode", list(KVQuantMode))
def test_layout_cache_dtype_round_trips_every_quant_mode(mode):
    """Every KVQuantMode must have a spelling that maps back to itself.

    ``layout_cache_dtype`` is called with the engine-wide ``kv_cache_dtype``
    during KV allocation. A mode without an entry raises KeyError there, which
    reports the enum value rather than the mode the user configured.
    """
    spec = AttentionSpec(
        block_size=16,
        num_kv_heads=4,
        head_size=128,
        dtype=torch.uint8,
        kv_quant_mode=mode,
    )
    spelling = spec.layout_cache_dtype("auto")
    assert get_kv_quant_mode(spelling) is mode


def test_layout_cache_dtype_keeps_matching_configuration():
    """A spec that already matches the engine dtype keeps its exact spelling."""
    spec = AttentionSpec(
        block_size=16,
        num_kv_heads=4,
        head_size=128,
        dtype=torch.uint8,
        kv_quant_mode=KVQuantMode.NVFP4,
    )
    assert spec.layout_cache_dtype("nvfp4") == "nvfp4"
