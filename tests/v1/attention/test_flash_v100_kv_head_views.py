# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.v1.attention.backends.flash_v100.spec.verifier import per_kv_head_views


def test_single_kv_head_passes_the_batch_through():
    query = torch.randn(4, 6, 256)
    key = torch.randn(3, 800, 1, 256)
    value = torch.randn(3, 800, 1, 256)
    out = torch.empty_like(query)

    (view,) = per_kv_head_views(query, key, value, out)

    assert view.query is query
    assert view.key is key
    assert view.value is value
    assert view.out is out


def test_two_kv_heads_get_their_own_query_heads_and_cache():
    query = torch.randn(4, 12, 256)
    key = torch.randn(3, 800, 2, 256)
    value = torch.randn(3, 800, 2, 256)
    out = torch.empty_like(query)

    views = per_kv_head_views(query, key, value, out)

    assert len(views) == 2
    for head, view in enumerate(views):
        assert view.query.shape == (4, 6, 256)
        assert view.query.is_contiguous()
        assert view.key.shape == (3, 800, 1, 256)
        assert view.out.shape == (4, 6, 256)
        assert view.out is not out
        assert torch.equal(view.query, query[:, 6 * head : 6 * (head + 1)])
        assert torch.equal(view.key[:, :, 0], key[:, :, head])
        assert torch.equal(view.value[:, :, 0], value[:, :, head])
        out[:, view.query_heads] = float(head)
    assert torch.all(out[:, :6] == 0) and torch.all(out[:, 6:] == 1)
