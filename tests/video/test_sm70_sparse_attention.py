# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent masked FP32 reference for real sparse SM70 execution."""

import pytest
import torch

from vllm.model_executor.layers.sm70_sparse_attention import block_sparse_attention

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a leased SM70 GPU"
)


def _reference(q, k, v, block_map, sizes):
    result = torch.zeros_like(q)
    for batch in range(q.shape[0]):
        for head in range(q.shape[2]):
            for query_block, query_size in enumerate(sizes):
                indices = [
                    torch.arange(i * 64, i * 64 + size, device=q.device)
                    for i, size in enumerate(sizes)
                    if block_map[batch, head, query_block, i]
                ]
                selected = torch.cat(indices)
                start = query_block * 64
                queries = q[batch, start : start + query_size, head].float()
                keys = k[batch, selected, head].float()
                values = v[batch, selected, head].float()
                scores = (queries @ keys.T) * 128**-0.5
                result[batch, start : start + query_size, head] = (
                    scores.softmax(-1) @ values
                ).half()
    return result


@pytest.mark.parametrize("sizes", [(1,), (63, 1), (17, 64, 3, 64, 1)])
@pytest.mark.parametrize("dense", [False, True])
def test_sparse_masks_and_nonterminal_edges(sizes, dense):
    torch.manual_seed(711)
    blocks = len(sizes)
    q, k, v = [
        torch.randn(2, blocks * 64, 3, 128, device="cuda", dtype=torch.float16)
        for _ in range(3)
    ]
    # Poison every padding region, including holes before later valid blocks.
    for i, size in enumerate(sizes):
        for operand in (q, k, v):
            operand[:, i * 64 + size : (i + 1) * 64] = float("nan")
    block_map = torch.ones(2, 3, blocks, blocks, device="cuda", dtype=torch.bool)
    if not dense and blocks > 1:
        block_map[:, :, 1:, ::2] = False
        block_map[:, :, 1:, -1] = True
        block_map[1, 2, 0, :-1] = False
    expected = _reference(q, k, v, block_map, sizes)
    actual = block_sparse_attention(
        q,
        k,
        v,
        block_map,
        torch.tensor(sizes, device="cuda", dtype=torch.int32),
        scale=128**-0.5,
    )
    assert torch.isfinite(actual).all()
    assert (actual.float() - expected.float()).norm() / expected.float().norm() < 0.001
    torch.testing.assert_close(actual, expected, rtol=0.003, atol=0.003)
    for i, size in enumerate(sizes):
        assert not torch.count_nonzero(actual[:, i * 64 + size : (i + 1) * 64])


def test_sparse_rejects_empty_rows_and_invalid_sizes_before_launch():
    x = torch.zeros(1, 128, 2, 128, device="cuda", dtype=torch.float16)
    mask = torch.ones(1, 2, 2, 2, device="cuda", dtype=torch.bool)
    sizes = torch.tensor([64, 1], device="cuda", dtype=torch.int32)
    mask[:, :, 1] = False
    with pytest.raises(RuntimeError, match="selected key"):
        block_sparse_attention(x, x, x, mask, sizes, scale=128**-0.5)
    mask.fill_(True)
    sizes[1] = 65
    with pytest.raises(RuntimeError, match=r"\[1,64\]"):
        block_sparse_attention(x, x, x, mask, sizes, scale=128**-0.5)


def test_sparse_all_blocks_matches_existing_64_key_arithmetic():
    from vllm.model_executor.models.minimax_h3.cuda_ops import flashattn_extension

    torch.manual_seed(509)
    q, k, v = [
        torch.randn(1, 256, 2, 128, device="cuda", dtype=torch.float16)
        for _ in range(3)
    ]
    expected = flashattn_extension().forward(q, k, v, 128**-0.5, 64)
    actual = block_sparse_attention(
        q,
        k,
        v,
        torch.ones(1, 2, 4, 4, device="cuda", dtype=torch.bool),
        torch.full((4,), 64, device="cuda", dtype=torch.int32),
        scale=128**-0.5,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_h3_owned_sparse_route_preserves_output_without_host_scalar_reads():
    from vllm.model_executor.layers.sm70_sparse_attention import (
        _h3_block_sparse_attention,
        sparse_extension,
    )

    assert hasattr(sparse_extension(), "_forward_prevalidated")
    torch.manual_seed(814)
    q, k, v = [
        torch.randn(1, 192, 2, 128, device="cuda", dtype=torch.float16)
        for _ in range(3)
    ]
    sizes = torch.tensor([17, 64, 3], device="cuda", dtype=torch.int32)
    mask = torch.ones(1, 2, 3, 3, device="cuda", dtype=torch.bool)
    mask[:, :, 1:, 1] = False
    expected = block_sparse_attention(q, k, v, mask, sizes, scale=128**-0.5)
    _h3_block_sparse_attention(q, k, v, mask, sizes, scale=128**-0.5)
    torch.accelerator.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as prof:
        actual = _h3_block_sparse_attention(q, k, v, mask, sizes, scale=128**-0.5)
    assert not any(
        event.key == "aten::_local_scalar_dense" for event in prof.key_averages()
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="block map"):
        _h3_block_sparse_attention(q, k, v, mask[:, :, :2], sizes, scale=128**-0.5)
