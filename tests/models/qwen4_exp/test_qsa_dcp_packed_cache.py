# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QSA DCP: target layers packed two per physical page write and read their
K/V through the real cache kernels exactly as unpacked caches do."""

import pytest
import torch

from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.worker.gpu.attn_utils import _reshape_kv_cache
from vllm.v1.worker.utils import AttentionGroup

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="QSA cache kernels need CUDA"
)

KERNEL_BLOCK, LOCAL_PAGE, BLOCKS, HEADS, DIM = 32, 64, 24, 12, 256


def _packed_views(e4m3: bool, monkeypatch) -> tuple[torch.Tensor, list[torch.Tensor]]:
    from vllm.v1.attention.backends import flash_attn
    from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend

    monkeypatch.setattr(flash_attn, "get_kv_cache_layout", lambda: "NHD")
    spec = FullAttentionSpec(
        block_size=LOCAL_PAGE,
        num_kv_heads=1,
        head_size=DIM,
        dtype=torch.uint8 if e4m3 else torch.float16,
        dcp_sharded=True,
    )
    members = ["model.layers.3.self_attn", "model.layers.7.self_attn"]
    raw = torch.zeros(
        BLOCKS * len(members) * spec.page_size_bytes, dtype=torch.int8, device="cuda"
    )
    views = _reshape_kv_cache(
        attn_groups=[AttentionGroup(FlashAttentionBackend, members, spec, 0)],
        kv_cache_raw_tensors={name: raw for name in members},
        cache_dtype="fp8_e4m3" if e4m3 else "auto",
        kernel_block_sizes=[KERNEL_BLOCK],
        shared_kv_cache_layers={},
        packed_members={name: (i, len(members)) for i, name in enumerate(members)},
    )
    return raw, [views[name] for name in members]


@pytest.mark.parametrize("e4m3", [False, True])
def test_packed_members_match_unpacked_caches(e4m3: bool, monkeypatch):
    from vllm._custom_ops import reshape_and_cache_flash
    from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_sparse_paged_attention

    raw, packed = _packed_views(e4m3, monkeypatch)
    cache_dtype = "fp8_e4m3" if e4m3 else "auto"
    kernel_blocks = BLOCKS * LOCAL_PAGE // KERNEL_BLOCK
    generator = torch.Generator().manual_seed(1600 + int(e4m3))
    scale = torch.ones(1, dtype=torch.float32, device="cuda")

    # One request per member, each over a different permutation of kernel
    # blocks, so writes cross physical pages in an arbitrary order.
    tokens, rows = kernel_blocks * KERNEL_BLOCK - 7, 5
    expected = []
    for member, view in enumerate(packed):
        reference = torch.zeros_like(view.contiguous())
        assert reference.stride(0) * 2 == view.stride(0)
        order = torch.randperm(kernel_blocks, generator=generator)
        positions = torch.arange(tokens)
        slots = (order[positions // KERNEL_BLOCK] * KERNEL_BLOCK) + (
            positions % KERNEL_BLOCK
        )
        key = torch.randn((tokens, 1, DIM), generator=generator).half().cuda()
        value = torch.randn((tokens, 1, DIM), generator=generator).half().cuda()
        for cache in (view, reference):
            key_cache, value_cache = cache.unbind(1)
            reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                slots.cuda(),
                cache_dtype,
                scale,
                scale,
            )
        expected.append((reference, order))

    # The second member's writes left the first member's kernel blocks intact,
    # and together the members cover the physical pages without overlap.
    for view, (reference, _) in zip(packed, expected):
        assert torch.equal(view, reference)
    covered = sum(
        torch.count_nonzero(reference.view(torch.uint8)) for reference, _ in expected
    )
    assert torch.count_nonzero(raw) == covered

    for member, view in enumerate(packed):
        reference, order = expected[member]
        selection = torch.full((rows, 1026), -1, dtype=torch.int32)
        for row in range(rows):
            chosen = torch.randperm(tokens, generator=generator)[:1000]
            selection[row, : chosen.numel()] = chosen.int()
        selection = selection.cuda()
        query = torch.randn((rows, HEADS, DIM), generator=generator).half().cuda()
        block_table = order.int().cuda()[None]
        token_to_req = torch.zeros(rows, dtype=torch.int32, device="cuda")
        results = []
        for cache in (view, reference):
            key_cache, value_cache = cache.unbind(1)
            out = torch.empty(query.shape, dtype=torch.float32, device="cuda")
            lse = torch.empty(query.shape[:2], dtype=torch.float32, device="cuda")
            qsa_sparse_paged_attention(
                query,
                key_cache,
                value_cache,
                selection,
                block_table,
                token_to_req,
                out=out,
                lse=lse,
                kv_cache_dtype=cache_dtype,
            )
            results.append((out, lse))
        # Same kernel and data, only the block stride differs: bitwise equal.
        assert torch.equal(results[0][0], results[1][0])
        assert torch.equal(results[0][1], results[1][1])
        assert torch.isfinite(results[0][0]).all()
