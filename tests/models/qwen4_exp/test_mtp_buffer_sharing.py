# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Target/draft sharing keeps one owner and respects independent engines."""

from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn

from vllm.models.qwen4_exp.nvidia import mtp


def make_predictor(indices):
    owner = mtp.Qwen4ExpMultiTokenPredictor.__new__(mtp.Qwen4ExpMultiTokenPredictor)
    nn.Module.__init__(owner)
    attention = nn.Module()
    attention.indexer = NS(skip_topk=False)
    attention.register_buffer("topk_indices_buffer", indices)
    layer = nn.Module()
    layer.self_attn = attention
    owner.layers = nn.ModuleList([layer])
    return owner, attention


def test_topk_share_reselect_compact_and_next_target_preserve_capacity():
    target = torch.full((8, 4), -99, dtype=torch.int32)
    owner, attention = make_predictor(torch.empty(6, 4, dtype=torch.int32))
    assert owner.adopt_target_topk_indices(target) == 6 * 4 * 4
    assert attention.topk_indices_buffer is target
    assert owner.adopt_target_topk_indices(target) == 0
    target[:6].copy_(torch.arange(24).reshape(6, 4))
    expected = target[[4, 1]].clone()
    owner.compact_topk_indices(torch.tensor([4, 1]))
    owner.set_skip_topk(True)
    assert torch.equal(attention.topk_indices_buffer[:2], expected)
    assert torch.all(target[6:] == -99)
    owner.set_skip_topk(False)
    target[:6].fill_(17)
    assert torch.all(attention.topk_indices_buffer[:6] == 17)
    assert not attention.indexer.skip_topk


@pytest.mark.parametrize(
    "target",
    [
        torch.empty(4, 4, dtype=torch.int32),
        torch.empty(8, 3, dtype=torch.int32),
        torch.empty(8, 4, dtype=torch.int64),
    ],
)
def test_incompatible_topk_keeps_private_buffer(target):
    original = torch.empty(6, 4, dtype=torch.int32)
    owner, attention = make_predictor(original)
    assert owner.adopt_target_topk_indices(target) == 0
    assert attention.topk_indices_buffer is original


def test_rope_and_topk_only_share_with_their_own_target():
    owners, targets = [], []
    for _ in range(2):
        target = nn.Module()
        target.register_buffer(
            "topk_indices_buffer", torch.zeros(8, 4, dtype=torch.int32)
        )
        target.register_buffer("cos_sin_cache", torch.arange(32).reshape(8, 4).half())
        owner = mtp.Qwen4ExpMTP.__new__(mtp.Qwen4ExpMTP)
        nn.Module.__init__(owner)
        owner.model, _ = make_predictor(torch.empty(6, 4, dtype=torch.int32))
        owner.rope = nn.Module()
        owner.rope.register_buffer("cos_sin_cache", target.cos_sin_cache.clone())
        owner.other_rope = nn.Module()
        distinct = target.cos_sin_cache + 1
        owner.other_rope.register_buffer("cos_sin_cache", distinct)
        owner.adopt_target_buffers(target)
        assert owner.rope.cos_sin_cache is target.cos_sin_cache
        assert owner.other_rope.cos_sin_cache is distinct
        owner.adopt_target_buffers(target)
        owners.append(owner)
        targets.append(target)
    assert (
        owners[0].rope.cos_sin_cache.data_ptr()
        != owners[1].rope.cos_sin_cache.data_ptr()
    )
    targets[0].topk_indices_buffer.fill_(3)
    assert torch.all(owners[0].model.layers[0].self_attn.topk_indices_buffer == 3)
    assert torch.all(owners[1].model.layers[0].self_attn.topk_indices_buffer == 0)
