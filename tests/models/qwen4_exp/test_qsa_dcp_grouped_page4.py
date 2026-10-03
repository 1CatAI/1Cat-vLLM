# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops
from vllm.models.qwen4_exp.nvidia.ops.qsa import (
    qsa_dcp_sparse_paged_attention_sm70_grouped_page4,
    qsa_sparse_paged_attention,
)
from vllm.models.qwen4_exp.nvidia.ops.qsa_dcp import (
    qsa_dcp_local_selection_width,
    qsa_localize_dcp_indices,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skip_global_cleanup

TOPK, RATIO, WORLD, INTERLEAVE = 2048, 4, 2, 1
WIDTH = TOPK + RATIO - 1


def test_small_batches_stay_on_triton():
    # Decode-sized batches never reach the grouped route, on any platform.
    rows = qsa_ops._SM70_QSA_DCP_GROUPED_PAGE4_MIN_ROWS - 1
    q = torch.zeros((rows, 12, 256), dtype=torch.float16)
    written = qsa_dcp_sparse_paged_attention_sm70_grouped_page4(
        q,
        torch.zeros((4, 16, 1, 256), dtype=torch.uint8),
        torch.zeros((4, 16, 1, 256), dtype=torch.uint8),
        torch.full((rows, WIDTH), -1, dtype=torch.int32),
        torch.zeros((1, 4), dtype=torch.int32),
        torch.zeros(rows, dtype=torch.int32),
        torch.zeros(q.shape, dtype=torch.float32),
        torch.zeros(q.shape[:2], dtype=torch.float32),
        "fp8_e4m3",
        1.0,
        1.0,
    )
    assert written == 0


def _selection(positions, generator):
    """Model-format global selection: score-ordered blocks plus the open tail."""
    rows = []
    for position in positions.tolist():
        visible = (position + 1) // RATIO
        blocks = torch.randperm(visible, generator=generator)[: TOPK // RATIO]
        tokens = (blocks[:, None] * RATIO + torch.arange(RATIO)).flatten().tolist()
        tail_start = visible * RATIO
        tokens += list(range(tail_start, position + 1))
        rows.append(tokens + [-1] * (WIDTH - len(tokens)))
    return torch.tensor(rows, dtype=torch.int32)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform.is_device_capability(70),
    reason="SM70 grouped page4 route",
)
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("e4m3", [False, True])
def test_grouped_route_matches_triton_lse_route(e4m3, packed, monkeypatch):
    flash = pytest.importorskip("flash_attn_v100.flash_attn_interface")
    # Keep the batch small; the row gate is a speed choice, not a contract.
    monkeypatch.setattr(qsa_ops, "_SM70_QSA_DCP_GROUPED_PAGE4_MIN_ROWS", 64)
    kv_dtype = "fp8_e4m3" if e4m3 else "auto"
    if not qsa_ops._qsa_grouped_page4_supported(flash.flash_attn_v100_cuda, kv_dtype):
        pytest.skip("Flash-V100 build has no grouped page4 route for this cache")
    generator = torch.Generator().manual_seed(816)
    page, pages, heads, dim = 16, 96, 12, 256
    # Request 0 selects 512 of its visible blocks; request 1 is short enough
    # to select every visible position. 77 rows leave a five-row remainder.
    positions = torch.cat([torch.arange(2400, 2440), torch.arange(100, 137)])
    token_to_req = torch.tensor([0] * 40 + [1] * 37, dtype=torch.int32)
    indices = _selection(positions, generator)
    indices[3] = -1  # A row without any selected position.
    indices, token_to_req = indices.cuda(), token_to_req.cuda()
    q = (torch.randn((77, heads, dim), generator=generator) * 0.3).half().cuda()
    # Interleaved K/V blocks, as the model's cache ABI stores them. A sharded
    # DCP cache packs one page of each member layer into every physical block.
    members = 2 if packed else 1
    cache = torch.randn((2 * pages + 3, members, 2, page, 1, dim), generator=generator)
    cache *= 0.5
    k_scale, v_scale = (0.25, 0.5) if e4m3 else (1.0, 1.0)
    if e4m3:
        cache[:, :, 0] /= k_scale
        cache[:, :, 1] /= v_scale
        cache = cache.to(torch.float8_e4m3fn).view(torch.uint8)
    else:
        cache = cache.half()
    k_cache, v_cache = cache.cuda()[:, members - 1].unbind(1)
    assert k_cache.stride(0) == members * 2 * page * dim
    block_table = (
        torch.randperm(2 * pages + 3, generator=generator)[: 2 * pages]
        .view(2, pages)
        .to(torch.int32)
        .cuda()
    )
    local_width = qsa_dcp_local_selection_width(TOPK, RATIO, WORLD, INTERLEAVE, WIDTH)

    for rank in range(WORLD):
        local = torch.empty_like(indices)
        qsa_localize_dcp_indices(
            indices,
            local,
            dcp_world_size=WORLD,
            dcp_rank=rank,
            interleave_size=INTERLEAVE,
            local_block_size=page,
        )
        expected_out = torch.empty(q.shape, dtype=torch.float32, device="cuda")
        expected_lse = torch.empty(q.shape[:2], dtype=torch.float32, device="cuda")
        qsa_sparse_paged_attention(
            q,
            k_cache,
            v_cache,
            local[:, :local_width],
            block_table,
            token_to_req,
            expected_out,
            kv_cache_dtype=kv_dtype,
            k_scale=k_scale,
            v_scale=v_scale,
            lse=expected_lse,
        )
        out = torch.full_like(expected_out, torch.nan)
        lse = torch.full_like(expected_lse, torch.nan)
        rows = qsa_dcp_sparse_paged_attention_sm70_grouped_page4(
            q,
            k_cache,
            v_cache,
            local,
            block_table,
            token_to_req,
            out,
            lse,
            kv_dtype,
            k_scale,
            v_scale,
        )
        assert rows == 72
        assert torch.isnan(out[rows:]).all() and torch.isnan(lse[rows:]).all()
        assert torch.isneginf(lse[3]).all() and not out[3].any()
        torch.testing.assert_close(lse[:rows], expected_lse[:rows], rtol=0, atol=1e-3)
        torch.testing.assert_close(
            out[:rows], expected_out[:rows], rtol=1e-2, atol=3e-3
        )
