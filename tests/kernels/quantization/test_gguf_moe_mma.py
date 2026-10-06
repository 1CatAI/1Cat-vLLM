# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gguf
import numpy as np
import pytest
import torch

from vllm.model_executor.layers.quantization import gguf_moe_planes as planes_lib

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (7, 0)
    or not hasattr(torch.ops._C, "gguf_moe_gate_up_sm70_out"),
    reason="requires SM70 and the gguf_moe_gate_up_sm70_out operator",
)

GGUF = {
    18: (98, gguf.GGMLQuantizationType.IQ3_XXS),
    21: (110, gguf.GGMLQuantizationType.IQ3_S),
    22: (82, gguf.GGMLQuantizationType.IQ2_S),
}


def _raw(rng, rows, k, size):
    raw = rng.integers(0, 256, (rows, k // 256, size), dtype=np.uint8)
    scale = np.full((rows, k // 256), 0.01, np.float16).view(np.uint8)
    raw[..., :2] = scale.reshape(rows, k // 256, 2)
    return raw.reshape(rows, -1)


def _q8_reference(hidden):
    """Production Q8_1 blocks (gguf_dp4a.cuh) for [M, top_k, n] fp16 values."""
    m, top_k, n = hidden.shape
    v = hidden.float().reshape(m, top_k, n // 32, 32)
    maximum = v.abs().amax(-1, keepdim=True)
    d = maximum / 127.0
    safe = torch.where(maximum == 0, torch.ones_like(d), d)
    q = torch.where(maximum == 0, torch.zeros_like(v), torch.sign(v) * torch.floor(v.abs() / safe + 0.5))
    ds = torch.cat([d, v.sum(-1, keepdim=True)], -1).half().contiguous()
    out = torch.empty(m, top_k, n // 32, 36, dtype=torch.uint8, device=hidden.device)
    out[..., :4] = ds.view(torch.uint8).reshape(m, top_k, n // 32, 4)
    out[..., 4:] = q.to(torch.int8).view(torch.uint8)
    return out


def _q8_decode(blocks):
    d = blocks[..., :2].contiguous().view(torch.half)[..., 0].float()
    q = blocks[..., 4:].contiguous().view(torch.int8).float()
    return q * d[..., None]


@pytest.mark.parametrize("kind", [18, 21, 22])
@pytest.mark.parametrize("tokens,distinct", [(1, 10), (5, 31), (8, 50), (20, 12), (20, 60)])
def test_grouped_gate_up_matches_dequantized_reference(kind, tokens, distinct):
    rng = np.random.default_rng(kind * 100 + tokens)
    experts, n, k, top_k = 64, 64, 2560, 10
    size, qtype = GGUF[kind]
    raw = {name: _raw(rng, experts * n, k, size) for name in "gu"}
    planes = {name: planes_lib.expert_planes(raw[name], kind) for name in "gu"}
    dev = "cuda"
    codes, scale = {}, {}
    for name in "gu":
        _, c, s = planes[name]
        codes[name] = torch.from_numpy(c).to(dev).view(experts, -1)
        scale[name] = torch.from_numpy(s).to(dev).view(experts, -1)
    pool = rng.permutation(experts)[:distinct]
    ids = np.stack(
        [rng.choice(pool, top_k, replace=False) for _ in range(tokens)]
    ).astype(np.int32)
    x = (torch.randn(tokens, k, device=dev) * 0.5).half()
    hidden = torch.zeros(tokens, top_k, n, dtype=torch.half, device=dev)
    table = torch.from_numpy(planes_lib.expert_table(kind)).to(dev)
    torch.ops._C.gguf_moe_gate_up_sm70_out(
        hidden,
        x,
        torch.from_numpy(ids).to(dev),
        codes["g"],
        scale["g"],
        codes["u"],
        scale["u"],
        planes["g"][0],
        table,
        4,
    )
    quantized = torch.zeros(tokens, top_k, n // 32, 36, dtype=torch.uint8, device=dev)
    torch.ops._C.gguf_moe_gate_up_sm70_out(
        quantized,
        x,
        torch.from_numpy(ids).to(dev),
        codes["g"],
        scale["g"],
        codes["u"],
        scale["u"],
        planes["g"][0],
        table,
        4,
    )
    torch.testing.assert_close(_q8_decode(_q8_reference(hidden)), _q8_decode(quantized))
    assert torch.equal(_q8_reference(hidden), quantized)
    xf = x.float().cpu()
    ref = torch.zeros(tokens, top_k, n)
    for t in range(tokens):
        for j in range(top_k):
            e = int(ids[t, j])
            rows = slice(e * n, (e + 1) * n)
            wg = torch.from_numpy(gguf.quants.dequantize(raw["g"][rows], qtype))
            wu = torch.from_numpy(gguf.quants.dequantize(raw["u"][rows], qtype))
            g, u = wg.float() @ xf[t], wu.float() @ xf[t]
            ref[t, j] = torch.nn.functional.silu(g) * u
    error = (hidden.float().cpu() - ref).norm() / ref.norm()
    assert float(error) < 2e-3
