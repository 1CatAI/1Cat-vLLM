# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact 256K logical-address boundaries and empty DCP shard math."""

import math

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_sparse_paged_attention

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("interleave", [1, 4, 16])
@pytest.mark.parametrize("kv_dtype", ["float16", "fp8_e4m3"])
@pytest.mark.parametrize(
    ("width", "length"),
    [(2051, 262143), (2051, 262144)],
)
@torch.inference_mode()
def test_dcp_sparse_attention_lse_and_merge(interleave, kv_dtype, width, length):
    torch.manual_seed(921)
    rows, heads, dim, page_size = 4, 12, 256, 16
    pages = (length + page_size * 2 - 1) // (page_size * 2)
    storage_length = pages * page_size * 2
    q = torch.randn(rows, heads, dim, device="cuda", dtype=torch.float16) * 0.3
    k = torch.randn(storage_length, 1, dim, device="cuda", dtype=torch.float16) * 0.5
    v = torch.randn_like(k) * 0.6
    k_scale, v_scale = (0.007, 0.009) if kv_dtype == "fp8_e4m3" else (1.0, 1.0)
    if kv_dtype == "fp8_e4m3":
        k = (
            (k.float() / k_scale)
            .clamp(-448, 448)
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )
        v = (
            (v.float() / v_scale)
            .clamp(-448, 448)
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )
        decoded_k = k.view(torch.float8_e4m3fn).float() * k_scale
        decoded_v = v.view(torch.float8_e4m3fn).float() * v_scale
    else:
        decoded_k, decoded_v = k.float(), v.float()
    # Noncontiguous physical pages plus shuffled selections and invalid tails.
    table = torch.randperm(pages, dtype=torch.int32, device="cuda")[None]
    selected = torch.randint(length, (rows, width), device="cuda", dtype=torch.int32)
    selected[:, -1] = -1
    selected[0, 0] = length - 1
    selected[1].fill_(-1)  # Empty on all shards (graph padding).
    selected[2].fill_(0)  # Empty only on rank 1.
    owners = (torch.arange(storage_length, device="cuda") // interleave) % 2
    local_outs, local_lses = [], []
    for rank in range(2):
        local_k = torch.empty(pages, page_size, 1, dim, dtype=k.dtype, device="cuda")
        local_v = torch.empty_like(local_k)
        local_k[table[0].long()] = k[owners == rank].view_as(local_k)
        local_v[table[0].long()] = v[owners == rank].view_as(local_v)
        out = torch.empty_like(q, dtype=torch.float32)
        lse = torch.empty(q.shape[:2], device="cuda", dtype=torch.float32)
        from vllm.models.qwen4_exp.nvidia.ops.qsa_dcp import qsa_localize_dcp_indices

        local_ids = torch.empty_like(selected)
        qsa_localize_dcp_indices(
            selected,
            local_ids,
            dcp_world_size=2,
            dcp_rank=rank,
            interleave_size=interleave,
            local_block_size=page_size,
        )
        qsa_sparse_paged_attention(
            q,
            local_k,
            local_v,
            local_ids,
            table,
            torch.zeros(rows, dtype=torch.int32, device="cuda"),
            out,
            kv_cache_dtype=kv_dtype,
            k_scale=k_scale,
            v_scale=v_scale,
            lse=lse,
        )
        assert torch.isfinite(out).all()
        assert torch.isneginf(lse[1]).all()
        if rank == 1:
            assert torch.isneginf(lse[2]).all()
            assert torch.count_nonzero(out[2]) == 0
        local_outs.append(out)
        local_lses.append(lse)
    lses = torch.stack(local_lses)
    weights = torch.softmax(lses * math.log(2), dim=0).nan_to_num()
    actual = (torch.stack(local_outs) * weights[..., None]).sum(0)
    ks = decoded_k[selected.clamp_min(0).long(), 0]
    vs = decoded_v[selected.clamp_min(0).long(), 0]
    scores = torch.einsum("rhd,rkd->rhk", q.float(), ks) / math.sqrt(dim)
    scores.masked_fill_(selected[:, None, :] < 0, -torch.inf)
    expected_lse = torch.logsumexp(scores, dim=-1) / math.log(2)
    merged_lse = torch.logsumexp(lses * math.log(2), dim=0) / math.log(2)
    expected = torch.einsum("rhk,rkd->rhd", scores.softmax(-1).nan_to_num(), vs)
    torch.testing.assert_close(merged_lse, expected_lse, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(actual, expected, atol=5e-4, rtol=5e-3)
