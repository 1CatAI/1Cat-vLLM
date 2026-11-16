# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KVBlockZeroer.copy_block_ids copies whole KV blocks (source -> destination)
across every segment in one launch. The partial prefix cache of hybrid
Mamba/GDN models relies on it, and a wrong copy yields silently wrong states.

The layout mimics a hybrid model: one attention layer (K and V are two
segments) and one Mamba state layer whose page is larger than the kernel
tile, so the copy runs in several chunks and the last chunk is a partial one.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
from vllm.v1.worker.utils import KVBlockZeroer

NUM_BLOCKS = 12
# Mamba page in int32 elements: above the 1024-element tile and not a multiple
# of it (5 chunks, the last one holds 904 elements).
MAMBA_PAGE_EL = 5000


class _KVFirstAttentionBackend:
    """Layout (2, num_blocks, block_size, heads, head_size)."""

    @staticmethod
    def get_kv_cache_block_dim(*args, **kwargs) -> int:
        return 1


def _random_int16(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    return torch.randint(-(2**15), 2**15, shape, dtype=torch.int16, device=device)


class _HybridCache:
    """Random attention + Mamba buffers wired into an initialized zeroer."""

    def __init__(self, device: torch.device) -> None:
        attn_spec = FullAttentionSpec(
            block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float16
        )
        mamba_spec = MambaSpec(
            block_size=16,
            shapes=((1,),),
            dtypes=(torch.int32,),
            page_size_padded=MAMBA_PAGE_EL * 4,
        )
        # Raw int16 storage so random bit patterns compare bitwise; the
        # attention view is fp16 as in a real cache.
        self.attn_raw = _random_int16((2, NUM_BLOCKS, 16, 2, 8), device)
        self.mamba = torch.randint(
            -(2**31),
            2**31,
            (NUM_BLOCKS, MAMBA_PAGE_EL),
            dtype=torch.int32,
            device=device,
        )
        attn_group = SimpleNamespace(
            kv_cache_spec=attn_spec,
            kv_cache_group_id=0,
            layer_names=["attn"],
            backend=_KVFirstAttentionBackend,
        )
        mamba_group = SimpleNamespace(
            kv_cache_spec=mamba_spec,
            kv_cache_group_id=1,
            layer_names=["mamba"],
            backend=None,
        )
        context = {
            "attn": SimpleNamespace(kv_cache=self.attn_raw.view(torch.float16)),
            "mamba": SimpleNamespace(kv_cache=[self.mamba]),
        }
        self.zeroer = KVBlockZeroer(device, pin_memory=False)
        self.zeroer.init_meta(
            [attn_group, mamba_group],
            kernel_block_sizes=[16, 16],
            cache_dtype="auto",
            runner_only_attn_layers=set(),
            static_forward_context=context,
        )

    def snapshot(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.attn_raw.clone(), self.mamba.clone()

    def expected_after(
        self, before: tuple[torch.Tensor, torch.Tensor], pairs: list[tuple[int, int]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn, mamba = before[0].clone(), before[1].clone()
        for src, dst in pairs:
            attn[:, dst] = before[0][:, src]
            mamba[dst] = before[1][src]
        return attn, mamba

    def assert_equals(self, expected: tuple[torch.Tensor, torch.Tensor]) -> None:
        torch.accelerator.synchronize()
        assert torch.equal(self.attn_raw, expected[0])
        assert torch.equal(self.mamba, expected[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_copy_block_ids_meta_has_three_segments_and_chunked_pages() -> None:
    """Guards the premise of the other tests: K, V and Mamba are separate
    segments with different page sizes, and the Mamba page needs the rest path."""
    cache = _HybridCache(torch.device("cuda"))
    assert cache.zeroer._meta is not None
    page_sizes = cache.zeroer._meta[2].tolist()
    max_chunks, blk_size, n_segs = cache.zeroer._meta[4:]
    assert sorted(page_sizes) == [128, 128, MAMBA_PAGE_EL]
    assert n_segs == 3
    assert max_chunks * blk_size > MAMBA_PAGE_EL > (max_chunks - 1) * blk_size
    assert MAMBA_PAGE_EL % blk_size != 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_copy_block_ids_copies_whole_blocks_in_every_segment() -> None:
    cache = _HybridCache(torch.device("cuda"))
    before = cache.snapshot()
    # Not ascending; source block 0; destination at the end of the pool.
    pairs = [(5, 1), (0, NUM_BLOCKS - 1), (9, 4), (2, 7)]
    cache.zeroer.copy_block_ids(pairs)
    expected = cache.expected_after(before, pairs)
    cache.assert_equals(expected)
    # The test must be able to see a change: every destination really differs.
    for _, dst in pairs:
        assert not torch.equal(before[1][dst], cache.mamba[dst])
        assert not torch.equal(before[0][:, dst], cache.attn_raw[:, dst])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_copy_block_ids_single_pair_leaves_all_other_blocks_alone() -> None:
    cache = _HybridCache(torch.device("cuda"))
    before = cache.snapshot()
    cache.zeroer.copy_block_ids([(NUM_BLOCKS - 1, 0)])
    cache.assert_equals(cache.expected_after(before, [(NUM_BLOCKS - 1, 0)]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_copy_block_ids_empty_pairs_change_nothing() -> None:
    cache = _HybridCache(torch.device("cuda"))
    before = cache.snapshot()
    cache.zeroer.copy_block_ids([])
    cache.assert_equals(before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_copy_block_ids_skips_out_of_range_ids() -> None:
    cache = _HybridCache(torch.device("cuda"))
    before = cache.snapshot()
    cache.zeroer.copy_block_ids(
        [(NUM_BLOCKS, 2), (3, NUM_BLOCKS), (-1, 2), (3, -1), (6, 8)]
    )
    cache.assert_equals(cache.expected_after(before, [(6, 8)]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_copy_block_ids_requires_initialized_metadata() -> None:
    zeroer = KVBlockZeroer(torch.device("cuda"), pin_memory=False)
    zeroer.copy_block_ids([])
    with pytest.raises(RuntimeError, match="need initialized block metadata"):
        zeroer.copy_block_ids([(0, 1)])
