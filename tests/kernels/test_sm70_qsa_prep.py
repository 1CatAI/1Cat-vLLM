# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (7, 0)
    or not hasattr(torch.ops._C, "qsa_prep_sm70_out"),
    reason="requires SM70 and qsa_prep_sm70_out",
)

HQ, D, ROT, BS = 6, 256, 64, 16


def _gemma(x, w, eps):
    x = x.float()
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (y * (1.0 + w.float())).half()


def _rope(x, cos, sin):
    rot = x[..., :ROT].float()
    a, b = rot[..., : ROT // 2], rot[..., ROT // 2 :]
    out = torch.cat([a * cos - b * sin, b * cos + a * sin], -1)
    return torch.cat([out.half(), x[..., ROT:]], -1)


@pytest.mark.parametrize("tokens", [1, 5, 20])
def test_prep_matches_reference(tokens):
    torch.manual_seed(tokens)
    dev, eps = "cuda", 1e-6
    qkv = torch.randn(tokens, HQ * 2 * D + 2 * D, device=dev).half()
    pos = torch.randint(0, 4000, (tokens,), device=dev)
    inv = 1.0 / (10000 ** (torch.arange(0, ROT, 2, device=dev).float() / ROT))
    freqs = torch.arange(4096, device=dev).float()[:, None] * inv
    cos_sin = torch.cat([freqs.cos(), freqs.sin()], -1)
    qw = torch.randn(D, device=dev).half() * 0.1
    kw = torch.randn(D, device=dev).half() * 0.1
    blocks = 8
    kc = torch.zeros(blocks, BS, 1, D, device=dev).half()
    vc = torch.zeros_like(kc)
    slot = torch.randperm(blocks * BS, device=dev)[:tokens]
    slot[0] = -1  # padded row: no cache write
    query = torch.empty(tokens, HQ, D, device=dev).half()
    torch.ops._C.qsa_prep_sm70_out(qkv, pos, cos_sin, qw, kw, eps, query, kc, vc, slot)

    cos = cos_sin[pos, : ROT // 2][:, None]
    sin = cos_sin[pos, ROT // 2 :][:, None]
    q = qkv[:, : HQ * 2 * D].view(tokens, HQ, 2 * D)[..., :D]
    k = qkv[:, HQ * 2 * D : HQ * 2 * D + D].view(tokens, 1, D)
    v = qkv[:, HQ * 2 * D + D :].view(tokens, 1, D)
    q_ref = _rope(_gemma(q, qw, eps), cos, sin)
    k_ref = _rope(_gemma(k, kw, eps), cos, sin)
    torch.testing.assert_close(query.float(), q_ref.float(), atol=4e-3, rtol=4e-3)
    for t in range(1, tokens):
        b, o = divmod(int(slot[t]), BS)
        torch.testing.assert_close(kc[b, o].float(), k_ref[t].float(), atol=4e-3, rtol=4e-3)
        assert torch.equal(vc[b, o], v[t])
