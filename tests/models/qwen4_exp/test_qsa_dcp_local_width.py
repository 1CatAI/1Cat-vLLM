# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QSA DCP: the localized selection fits the per-rank width bound."""

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops.qsa_dcp import (
    qsa_dcp_local_selection_width,
    qsa_localize_dcp_indices,
)

TOKEN_TOPK, RATIO = 2048, 4
FULL_WIDTH = TOKEN_TOPK + RATIO - 1


def _structured_selection(
    rows: int, context: int, generator: torch.Generator
) -> torch.Tensor:
    """Selections shaped like _expand_qsa_indices_kernel's output: up to 512
    whole groups among the (q + 1) // RATIO complete ones, plus the 0-3 token
    causal tail of the open group."""
    out = torch.full((rows, FULL_WIDTH), -1, dtype=torch.int32)
    for row in range(rows):
        position = int(torch.randint(RATIO, context, (1,), generator=generator))
        complete = (position + 1) // RATIO
        chosen = torch.randperm(complete, generator=generator)[: TOKEN_TOPK // RATIO]
        tokens = (chosen[:, None] * RATIO + torch.arange(RATIO)).flatten()
        tail = torch.arange(complete * RATIO, position + 1)
        assert tail.numel() <= RATIO - 1
        ids = torch.cat([tokens, tail]).int()
        out[row, : ids.numel()] = ids
    return out


def test_width_bound_values():
    assert qsa_dcp_local_selection_width(TOKEN_TOPK, RATIO, 2, 1, FULL_WIDTH) == 1026
    assert qsa_dcp_local_selection_width(TOKEN_TOPK, RATIO, 2, 2, FULL_WIDTH) == 1026
    # A group no longer splits evenly: keep the full width.
    assert qsa_dcp_local_selection_width(TOKEN_TOPK, RATIO, 2, 4, FULL_WIDTH) == (
        FULL_WIDTH
    )
    assert qsa_dcp_local_selection_width(TOKEN_TOPK, RATIO, 1, 1, FULL_WIDTH) == (
        FULL_WIDTH
    )


@pytest.mark.parametrize("interleave", [1, 2])
@pytest.mark.parametrize("context", [300, 2100, 30030])
def test_owned_tokens_never_exceed_bound(interleave: int, context: int):
    generator = torch.Generator().manual_seed(57431 + context + interleave)
    selection = _structured_selection(64, context, generator)
    bound = qsa_dcp_local_selection_width(TOKEN_TOPK, RATIO, 2, interleave, FULL_WIDTH)
    valid = selection >= 0
    for rank in range(2):
        owned = valid & ((selection // interleave) % 2 == rank)
        assert int(owned.sum(dim=1).max()) <= bound


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("interleave", [1, 2])
def test_localized_columns_after_bound_are_padding(interleave: int):
    generator = torch.Generator().manual_seed(1234 + interleave)
    selection = _structured_selection(128, 30030, generator).cuda()
    bound = qsa_dcp_local_selection_width(TOKEN_TOPK, RATIO, 2, interleave, FULL_WIDTH)
    for rank in range(2):
        local = torch.empty_like(selection)
        qsa_localize_dcp_indices(
            selection,
            local,
            dcp_world_size=2,
            dcp_rank=rank,
            interleave_size=interleave,
            local_block_size=1600,
        )
        assert bool((local[:, bound:] == -1).all())
        owned = (selection >= 0) & ((selection // interleave) % 2 == rank)
        assert torch.equal((local >= 0).sum(dim=1), owned.sum(dim=1).to(torch.int64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("e4m3", [False, True])
def test_narrowed_dcp2_decode_matches_unsharded(e4m3: bool):
    """The DCP2 decode path as served: narrowed local width, LSE merge, then
    the fused Triton gate, against the unsharded kernel with its fused gate."""
    import math

    from vllm.models.qwen4_exp.nvidia.ops.qsa import (
        _qsa_output_gate,
        qsa_sparse_paged_attention,
    )

    generator = torch.Generator().manual_seed(4096 + int(e4m3))
    context, page, heads, dim, rows = 8192, 32, 12, 256, 4
    dtype = "fp8_e4m3" if e4m3 else "auto"
    key = torch.randn((context, 1, dim), generator=generator).half()
    value = torch.randn((context, 1, dim), generator=generator).half()
    if e4m3:
        key = key.to(torch.float8_e4m3fn).view(torch.uint8)
        value = value.to(torch.float8_e4m3fn).view(torch.uint8)
    key, value = key.cuda(), value.cuda()
    selection = _structured_selection(rows, context, generator).cuda()
    query = torch.randn((rows, heads, dim), generator=generator).half().cuda()
    gate = torch.randn((rows, heads, dim), generator=generator).half().cuda()
    token_to_req = torch.zeros(rows, dtype=torch.int32, device="cuda")
    bound = qsa_dcp_local_selection_width(TOKEN_TOPK, RATIO, 2, 1, FULL_WIDTH)

    partials, lses = [], []
    for rank in range(2):
        owned = torch.arange(context, device="cuda") % 2 == rank
        local_key = key[owned].reshape(-1, page, 1, dim)
        local_value = value[owned].reshape(-1, page, 1, dim)
        local = torch.empty_like(selection)
        qsa_localize_dcp_indices(
            selection,
            local,
            dcp_world_size=2,
            dcp_rank=rank,
            interleave_size=1,
            local_block_size=page,
        )
        partial = torch.empty(query.shape, dtype=torch.float32, device="cuda")
        lse = torch.empty(query.shape[:2], dtype=torch.float32, device="cuda")
        qsa_sparse_paged_attention(
            query,
            local_key,
            local_value,
            local[:, :bound],
            torch.arange(context // 2 // page, dtype=torch.int32, device="cuda")[None],
            token_to_req,
            out=partial,
            lse=lse,
            kv_cache_dtype=dtype,
        )
        partials.append(partial)
        lses.append(lse)
    partials, lses = torch.stack(partials), torch.stack(lses)
    merged_lse = torch.logsumexp(lses * math.log(2), dim=0) / math.log(2)
    merged = (partials * torch.exp2(lses - merged_lse)[..., None]).sum(dim=0)
    served = torch.empty_like(query)
    served.copy_(merged)
    _qsa_output_gate(served, gate)

    reference = qsa_sparse_paged_attention(
        query,
        key.reshape(-1, page, 1, dim),
        value.reshape(-1, page, 1, dim),
        selection,
        torch.arange(context // page, dtype=torch.int32, device="cuda")[None],
        token_to_req,
        output_gate=gate,
        kv_cache_dtype=dtype,
    )
    torch.testing.assert_close(served, reference, rtol=5e-3, atol=3e-3)
