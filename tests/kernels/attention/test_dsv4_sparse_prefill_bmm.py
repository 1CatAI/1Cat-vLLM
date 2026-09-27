# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The batched-matmul sparse MLA prefill attention against a float64 reference."""

import pytest
import torch

from vllm.models.deepseek_v4.common.ops import (
    sparse_attn_prefill_bmm,
    sparse_prefill_bmm_workspace_specs,
)
from vllm.models.deepseek_v4.common.ops.sparse_prefill_bmm import MAX_TOKENS_PER_PASS
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="measured and enabled on CUDA only"
)

HEAD_DIM = 512


def _reference(q, kv, indices, lengths, scale, attn_sink) -> torch.Tensor:
    out = torch.zeros(q.shape, dtype=torch.float64, device=q.device)
    for token in range(q.shape[0]):
        row = indices[token, : int(lengths[token])]
        keys = kv[row[row >= 0].long()].double()
        logits = torch.einsum("hd,kd->hk", q[token].double(), keys) * scale
        logits = torch.cat([logits, attn_sink.double()[:, None]], dim=1)
        probs = torch.softmax(logits, dim=1)[:, :-1]
        out[token] = probs @ keys
    return out


# 300 tokens take three passes, the last one partial.
@pytest.mark.parametrize("num_tokens", [7, 300])
@pytest.mark.parametrize("num_heads", [8, 64])
@pytest.mark.parametrize("width", [128, 640])
@torch.inference_mode()
def test_sparse_attn_prefill_bmm_matches_reference(
    num_tokens: int, num_heads: int, width: int
):
    torch.manual_seed(0)
    num_kv = 900
    q = torch.randn((num_tokens, num_heads, HEAD_DIM), device="cuda").half()
    kv = (0.3 * torch.randn((num_kv, HEAD_DIM), device="cuda")).half()
    # Row 0 stands for workspace memory no sequence has written yet (a short
    # prompt has no compressed entries); unused slots must not read it.
    kv[0] = float("nan")
    indices = torch.randint(
        1, num_kv, (num_tokens, width), dtype=torch.int32, device="cuda"
    )
    lengths = torch.tensor(
        [width, 0, 1, width // 2, width - 1, 5, width], dtype=torch.int32
    ).cuda()
    lengths = lengths.repeat(-(-num_tokens // lengths.numel()))[:num_tokens]
    indices[torch.arange(width, device="cuda")[None, :] >= lengths[:, None]] = -1
    indices[3, 2] = -1  # an unused slot inside the valid length
    attn_sink = torch.randn(num_heads, dtype=torch.float32, device="cuda")
    scale = HEAD_DIM**-0.5

    # Reserved for more tokens and a wider index tensor than this call uses.
    buffers = [
        torch.full(shape, float("nan"), dtype=dtype, device="cuda")
        for shape, dtype in sparse_prefill_bmm_workspace_specs(
            num_tokens + 9, num_heads, HEAD_DIM, width + 128, torch.float16
        )
    ]
    output = torch.empty_like(q)
    sparse_attn_prefill_bmm(q, kv, indices, lengths, scale, attn_sink, output, *buffers)

    expected = _reference(q, kv, indices, lengths, scale, attn_sink)
    assert torch.isfinite(output).all()
    assert not output[1].any()  # no keys: the sink takes all the weight
    assert buffers[0].shape[0] == min(num_tokens + 9, MAX_TOKENS_PER_PASS)
    torch.testing.assert_close(output.double(), expected, atol=1e-3, rtol=1e-3)
