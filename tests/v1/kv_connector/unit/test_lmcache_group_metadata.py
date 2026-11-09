# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional real-LMCache metadata checks; no GPU transfer or serving claims.

Requires LMCache's compiled native extension and group conversion API.
Tensor shapes are small synthetic fixtures: these tests check group semantics,
not the physical layout of a particular model's attention or recurrent state.
"""

import pytest
import torch

from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)


def _convert(specs, dcp_size=1):
    pytest.importorskip("lmcache.lmcache_native")
    conversion = pytest.importorskip("lmcache.integration.vllm.kv_cache_groups")
    config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec([f"layer{i}"], spec) for i, spec in enumerate(specs)
        ],
    )
    caches = {
        f"layer{i}": torch.zeros(2, 8, 16, 1, 16, dtype=torch.float16)
        for i in range(len(specs))
    }
    return conversion.create_engine_group_infos_from_vllm(
        config, caches, dcp_size=dcp_size
    )


def _attention():
    return FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=16, dtype=torch.float16
    )


def _mamba():
    return MambaSpec(
        block_size=16,
        shapes=((16,),),
        dtypes=(torch.float16,),
        mamba_cache_mode="align",
    )


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("dcp_size", [1, 2])
def test_lmcache_real_group_metadata(hybrid, dcp_size):
    specs = [_attention(), _mamba()] if hybrid else [_attention()]
    groups = _convert(specs, dcp_size)
    from lmcache.v1.multiprocess.group_view import expand_engine_block_ids

    assert [g.engine_group_id for g in groups] == list(range(len(specs)))
    assert [g.layer_indices for g in groups] == [(i,) for i in range(len(specs))]
    assert [g.tokens_per_block for g in groups] == (
        [16 * dcp_size, 16] if hybrid else [16 * dcp_size]
    )
    assert [g.sw_size_tokens for g in groups] == ([-1, 16] if hybrid else [-1])
    assert [g.recurrent_state for g in groups] == ([False, True] if hybrid else [False])
    block_ids = [[3, 7], [2, 5]] if hybrid else [[3, 7]]
    assert expand_engine_block_ids(groups, block_ids) == block_ids


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="LMCache b5d109ea includes non-prefix-cacheable scratch groups",
)
def test_lmcache_qsa_scratch_group_exclusion():
    ring = CircularBufferSpec(
        block_size=4, num_kv_heads=1, head_size=16, dtype=torch.float16
    )
    assert not ring.prefix_cacheable
    groups = _convert([_attention(), ring, _mamba()])
    # Preserve original engine IDs; never persist scratch state as prefix KV.
    assert [g.engine_group_id for g in groups] == [0, 2]
