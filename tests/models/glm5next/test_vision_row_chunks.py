# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GLM-5.3 vision tower runs its token-wise steps in row chunks; every row
must come out as in one piece."""

import pytest
import torch
from torch import nn

from vllm.models.glm5next.nvidia import multimodal

_ROWS = 10
_ONE_PIECE = 1 << 30


class _EchoAttention(nn.Module):
    """Stands in for the encoder attention so the test sees exactly what the
    chunked projections hand to it."""

    def forward(self, query, key, value, cu_seqlens, max_seqlen):
        return query * 0.5 + key * 0.25 + value


def _chunked_and_one_piece(monkeypatch, run):
    monkeypatch.setattr(multimodal, "_ROWS_PER_CHUNK", 3)
    chunked = run()
    monkeypatch.setattr(multimodal, "_ROWS_PER_CHUNK", _ONE_PIECE)
    return chunked, run()


@torch.inference_mode()
def test_patch_merger_rows_match_one_piece(monkeypatch, dist_init, default_vllm_config):
    torch.manual_seed(0)
    merger = multimodal.Glm5NextPatchMerger(
        d_model=16, context_dim=32, swiglu_limit=7.0
    ).cuda()
    for parameter in merger.parameters():
        parameter.normal_(std=0.2)
    x = torch.randn(_ROWS, 16, device="cuda")

    chunked, one_piece = _chunked_and_one_piece(monkeypatch, lambda: merger(x))

    assert chunked.shape == (_ROWS, 16)
    torch.testing.assert_close(chunked, one_piece, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("rotary", [False, True])
@torch.inference_mode()
def test_attention_projections_match_one_piece(
    monkeypatch, dist_init, default_vllm_config, rotary
):
    torch.manual_seed(0)
    attention = multimodal.Glm5NextVisionAttention(
        embed_dim=16, num_heads=2, projection_size=16
    ).cuda()
    for parameter in attention.parameters():
        parameter.normal_(std=0.2)
    attention.attn = _EchoAttention()
    x = torch.randn(_ROWS, 1, 16, device="cuda")
    cos = sin = None
    if rotary:
        angles = torch.rand(_ROWS, attention.head_dim // 2, device="cuda")
        cos, sin = angles.cos(), angles.sin()
    cu_seqlens = torch.tensor([0, _ROWS], dtype=torch.int32, device="cuda")

    chunked, one_piece = _chunked_and_one_piece(
        monkeypatch, lambda: attention(x, cu_seqlens, cos, sin)
    )

    assert chunked.shape == (_ROWS, 1, 16)
    torch.testing.assert_close(chunked, one_piece, rtol=1e-5, atol=1e-6)
