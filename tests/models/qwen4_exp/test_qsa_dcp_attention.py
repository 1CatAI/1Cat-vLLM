# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_sparse_paged_attention
from vllm.models.qwen4_exp.nvidia.ops.qsa_dcp import qsa_localize_dcp_indices

pytestmark = [
    pytest.mark.skip_global_cleanup,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.mark.parametrize("e4m3", [False, True])
@pytest.mark.parametrize("width", [3, 2051])
def test_dcp2_g12_attention_lse_matches_fp32_reference(e4m3, width):
    generator = torch.Generator().manual_seed(57431)
    tokens, page, heads, dim = 128, 16, 12, 256
    world, interleave = 2, 4
    k = torch.randn((tokens, 1, dim), generator=generator).half()
    v = torch.randn((tokens, 1, dim), generator=generator).half()
    q = torch.randn((3, heads, dim), generator=generator).half().cuda()
    k_scale, v_scale = (0.125, 0.25) if e4m3 else (1.0, 1.0)
    if e4m3:
        k = k.to(torch.float8_e4m3fn)
        v = v.to(torch.float8_e4m3fn)
    k_ref, v_ref = k.float().cuda() * k_scale, v.float().cuda() * v_scale
    if e4m3:
        k, v = k.view(torch.uint8), v.view(torch.uint8)
    k, v = k.cuda(), v.cuda()
    selection = torch.full((3, width), -1, dtype=torch.int32)
    count = min(width, tokens)
    selection[0, :count] = torch.randperm(tokens, generator=generator)[:count]
    selection[1, :3] = torch.tensor([0, 1, 32])  # Only rank 0 owns these.
    # Row 2 has no selected positions on either rank.
    indices = selection.cuda()
    token_to_req = torch.zeros(3, dtype=torch.int32, device="cuda")
    local_ids = torch.empty_like(indices)
    partials, lses = [], []
    for rank in range(world):
        owned = (torch.arange(tokens, device="cuda") // interleave) % world == rank
        local_k = k[owned].reshape(-1, page, 1, dim)
        local_v = v[owned].reshape_as(local_k)
        # Nonidentity physical page mapping catches accidental direct addressing.
        table = torch.tensor([[3, 1, 0, 2]], dtype=torch.int32, device="cuda")
        physical_k, physical_v = torch.empty_like(local_k), torch.empty_like(local_v)
        physical_k[table[0].long()] = local_k
        physical_v[table[0].long()] = local_v
        qsa_localize_dcp_indices(
            indices,
            local_ids,
            dcp_world_size=world,
            dcp_rank=rank,
            interleave_size=interleave,
            local_block_size=page,
        )
        out = torch.empty(q.shape, dtype=torch.float32, device="cuda")
        lse = torch.empty(q.shape[:2], dtype=torch.float32, device="cuda")
        qsa_sparse_paged_attention(
            q,
            physical_k,
            physical_v,
            local_ids,
            table,
            token_to_req,
            out=out,
            lse=lse,
            kv_cache_dtype="fp8_e4m3" if e4m3 else "auto",
            k_scale=k_scale,
            v_scale=v_scale,
        )
        assert torch.isfinite(out).all()
        assert torch.isneginf(lse[2]).all()
        assert torch.count_nonzero(out[2]) == 0
        if rank == 1:
            assert torch.isneginf(lse[1]).all()
            assert torch.count_nonzero(out[1]) == 0
        partials.append(out)
        lses.append(lse)

    partials, lses = torch.stack(partials), torch.stack(lses)
    global_lse = torch.logsumexp(lses * math.log(2), dim=0) / math.log(2)
    weights = torch.where(torch.isfinite(lses), torch.exp2(lses - global_lse), 0)
    merged = (partials * weights[..., None]).sum(dim=0)
    reference = torch.zeros_like(merged)
    expected_lse = torch.full_like(global_lse, -torch.inf)
    for row in range(3):
        ids = indices[row][indices[row] >= 0].long()
        if len(ids) == 0:
            continue
        logits = q[row].float() @ k_ref[ids, 0].T / math.sqrt(dim)
        reference[row] = logits.softmax(dim=-1) @ v_ref[ids, 0]
        expected_lse[row] = torch.logsumexp(logits, dim=-1) / math.log(2)
    torch.testing.assert_close(merged, reference, rtol=3e-3, atol=2e-3)
    torch.testing.assert_close(global_lse, expected_lse, rtol=1e-4, atol=2e-3)
    # Exercise the unchanged DCP1 call signature and its FP16 output route.
    full_table = torch.arange(tokens // page, dtype=torch.int32, device="cuda")[None]
    baseline = qsa_sparse_paged_attention(
        q,
        k.reshape(-1, page, 1, dim),
        v.reshape(-1, page, 1, dim),
        indices,
        full_table,
        token_to_req,
        kv_cache_dtype="fp8_e4m3" if e4m3 else "auto",
        k_scale=k_scale,
        v_scale=v_scale,
    )
    torch.testing.assert_close(baseline.float(), reference, rtol=3e-3, atol=2e-3)
    assert torch.equal(indices.cpu(), selection)
