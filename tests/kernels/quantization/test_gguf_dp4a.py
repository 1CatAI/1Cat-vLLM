# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest
import torch
import vllm._C  # noqa: F401

from vllm.model_executor.layers.quantization.gguf_raw import RawGGUFProjection
from vllm.transformers_utils.gguf_tensor_reader import dequantize

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def quantize_reference(x):
    groups = x.float().reshape(x.shape[0], -1, 32)
    maximum = groups.abs().amax(-1, keepdim=True)
    d = maximum / 127
    ratio = torch.where(maximum == 0, 0, groups / d)
    q = (ratio.sign() * (ratio.abs() + 0.5).floor()).to(torch.int8)
    scales = d.squeeze(-1).half()
    sums = groups.sum(-1).half()
    return q, scales, sums


@pytest.mark.parametrize("m", [1, 5, 20])
def test_q8_1_matches_round_away_oracle_and_graph(m):
    torch.manual_seed(970 + m)
    x = torch.randn((m, 768), device="cuda", dtype=torch.float16)
    x[:, :32] = 0
    x[0, 32:64] = torch.tensor([127, 0.5, -0.5, 63.5, -63.5] + [0] * 27, device="cuda")
    out = torch.empty((m, 24, 36), device="cuda", dtype=torch.uint8)

    def run():
        torch.ops._C.gguf_quantize_q8_1_sm70_out(out, x)

    def check():
        q, d, s = quantize_reference(x)
        torch.testing.assert_close(out[:, :, 4:].view(torch.int8), q, rtol=0, atol=0)
        ds = out[:, :, :4].contiguous().view(torch.float16)
        torch.testing.assert_close(ds[:, :, 0], d, rtol=0, atol=0)
        torch.testing.assert_close(ds[:, :, 1], s, rtol=0, atol=0)

    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    x.copy_(torch.randn_like(x))
    graph.replay()
    check()


@pytest.mark.parametrize("m", [1, 5, 20])
@pytest.mark.parametrize("activated", [False, True])
def test_iq3_s_dot_matches_official_weight_and_q8_oracle(m, activated):
    experts, n, k, top_k = 4, 7, 768, 2
    rng = np.random.default_rng(21)
    weights, reference = [], []
    for _ in range(2):
        data = rng.integers(0, 256, (experts * n, k // 256, 110), dtype=np.uint8)
        d = rng.uniform(0.001, 0.005, data.shape[:2]).astype("<f2")
        data[:, :, :2] = d[..., None].view(np.uint8)
        data = data.reshape(experts * n, -1)
        raw = RawGGUFProjection.from_rows(data, 21)
        weights.append(torch.from_numpy(raw.data.reshape(experts, n, -1)).cuda())
        reference.append(
            torch.from_numpy(dequantize(data, 21)).reshape(experts, n, k).cuda()
        )
    torch.manual_seed(970 + m)
    x = (torch.randn((m, k), device="cuda") * 0.125).half()
    ids = torch.stack(
        (torch.zeros(m, dtype=torch.int64), torch.arange(m) % 3 + 1), 1
    ).cuda()
    q8 = torch.empty((m, k // 32, 36), dtype=torch.uint8, device="cuda")
    out = torch.empty(
        (m, top_k, n) if activated else (m, top_k, 2, n),
        dtype=torch.float16,
        device="cuda",
    )

    def run():
        torch.ops._C.gguf_quantize_q8_1_sm70_out(q8, x)
        torch.ops._C.gguf_dp4a_gate_up_sm70_out(out, q8, ids, *weights, 21, activated)

    def check():
        q, d, _ = quantize_reference(x)
        quantized = (q.float() * d[..., None].float()).reshape(m, k)
        gate, up = [
            torch.einsum("mk,mtnk->mtn", quantized, w[ids]).half() for w in reference
        ]
        expected = (
            torch.nn.functional.silu(gate.float()) * up.float()
            if activated
            else torch.stack((gate, up), 2)
        )
        torch.testing.assert_close(
            out.float(), expected.float(), rtol=0.002, atol=0.001
        )
        assert torch.isfinite(out).all()

    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    x.copy_(torch.randn_like(x) * 0.125)
    graph.replay()
    check()
