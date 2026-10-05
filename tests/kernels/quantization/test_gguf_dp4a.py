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
    # CUDA uses FP32 division; Torch division by a scalar can multiply by
    # its rounded reciprocal instead. Explicit rounding avoids a different
    # side of a half-integer boundary in this independent oracle.
    d = (maximum.double() / 127).float()
    ratio = torch.where(maximum == 0, 0, (groups.double() / d.double()).float())
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
@pytest.mark.parametrize("source_type,block_bytes", [(18, 98), (21, 110), (22, 82)])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_lattice_dot_matches_official_weight_and_q8_oracle(
    m, activated, source_type, block_bytes, index_dtype
):
    experts, n, k, top_k = 4, 7, 768, 2
    rng = np.random.default_rng(21)
    weights, reference = [], []
    for _ in range(2):
        data = rng.integers(
            0, 256, (experts * n, k // 256, block_bytes), dtype=np.uint8
        )
        d = rng.uniform(0.001, 0.005, data.shape[:2]).astype("<f2")
        data[:, :, :2] = d[..., None].view(np.uint8)
        data = data.reshape(experts * n, -1)
        raw = RawGGUFProjection.from_rows(data, source_type)
        weights.append(torch.from_numpy(raw.data.reshape(experts, n, -1)).cuda())
        reference.append(
            torch.from_numpy(dequantize(data, source_type))
            .reshape(experts, n, k)
            .cuda()
        )
    torch.manual_seed(970 + m)
    x = (torch.randn((m, k), device="cuda") * 0.125).half()
    ids = torch.stack(
        (torch.zeros(m, dtype=torch.int64), torch.arange(m) % 3 + 1), 1
    ).to(device="cuda", dtype=index_dtype)
    q8 = torch.empty((m, k // 32, 36), dtype=torch.uint8, device="cuda")
    out = torch.empty(
        (m, top_k, n) if activated else (m, top_k, 2, n),
        dtype=torch.float16,
        device="cuda",
    )

    def run():
        torch.ops._C.gguf_quantize_q8_1_sm70_out(q8, x)
        torch.ops._C.gguf_dp4a_gate_up_sm70_out(
            out, q8, ids, *weights, source_type, activated
        )

    def check():
        q, d, _ = quantize_reference(x)
        quantized = (q.float() * d[..., None].float()).reshape(m, k)
        gate, up = [
            torch.einsum("mk,mtnk->mtn", quantized, w[ids]).half() for w in reference
        ]
        expected = (
            (torch.nn.functional.silu(gate) * up)
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


@pytest.mark.parametrize("m", [5, 20])
@pytest.mark.parametrize("source_type", [20, 42])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_down_unroute_matches_tp4_official_weights(m, source_type, index_dtype):
    from vllm.model_executor.layers.quantization.gguf_turbomind_moe import (
        GGUFExpertBank,
    )
    from vllm.transformers_utils.gguf_tensor_reader import quant_size

    experts, n, full_k, k, top_k = 3, 64, 640, 160, 2
    block, size = quant_size(source_type)
    rng = np.random.default_rng(source_type)
    bank = GGUFExpertBank(source_type, experts, torch.device("cuda"), torch.float16)
    reference = []
    for expert in range(experts):
        data = rng.integers(0, 256, (n, full_k // block, size), dtype=np.uint8)
        d = rng.uniform(0.001, 0.005, data.shape[:2]).astype("<f2")
        data[:, :, :2] = d[..., None].view(np.uint8)
        data = data.reshape(n, -1)
        bank.add(expert, torch.from_numpy(data), rank=1, size=4, axis=1)
        reference.append(torch.from_numpy(dequantize(data, source_type)[:, k : 2 * k]))
    bank.finalize()
    weights = torch.stack(reference).cuda()
    torch.manual_seed(970 + m)
    hidden = torch.randn((m, top_k, k), device="cuda", dtype=torch.float16)
    ids = torch.stack((torch.arange(m) % experts, torch.zeros(m)), 1).to(
        device="cuda", dtype=index_dtype
    )
    probabilities = torch.softmax(torch.randn((m, top_k), device="cuda"), 1)
    out = torch.empty((m, n), device="cuda", dtype=torch.float16)

    def run():
        torch.ops._C.gguf_dp4a_down_unroute_sm70_out(
            out,
            hidden,
            ids,
            probabilities,
            bank.weight_ptrs,
            bank.stat_ptrs,
            source_type,
            experts,
        )

    def check():
        q, d, _ = quantize_reference(hidden.reshape(m * top_k, k))
        quantized = (q.float() * d[..., None].float()).reshape(m, top_k, k)
        down = torch.einsum("mtk,mtnk->mtn", quantized, weights[ids]).half()
        expected = (down.float() * probabilities[..., None]).sum(1).half()
        torch.testing.assert_close(out, expected, rtol=0.002, atol=0.002)
        assert torch.isfinite(out).all()

    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    hidden.copy_(torch.randn_like(hidden))
    probabilities.copy_(torch.softmax(torch.randn_like(probabilities), 1))
    ids.copy_((ids + 1) % experts)
    graph.replay()
    check()
