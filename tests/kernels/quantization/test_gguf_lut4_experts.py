# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gguf
import numpy as np
import pytest
import torch
import vllm._C  # noqa: F401

from vllm.model_executor.layers.quantization.gguf_lut_transcode import transcode_lut4
from vllm.model_executor.layers.quantization.gguf_turbomind_moe import GGUFExpertBank

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("source_type", [20, 23])
@pytest.mark.parametrize("m", [1, 5, 20])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_canonical_iq4_integer_dot_and_changed_graph(source_type, m, index_dtype):
    experts, n, k, top_k = 3, 64, 512, 2
    generator = np.random.default_rng(1000 + source_type)
    banks, integers, scales = [], [], []
    for _ in range(2):
        bank = GGUFExpertBank(source_type, experts, torch.device("cuda"), torch.float16)
        codes, coefficients = [], []
        block, width = (32, 18) if source_type == 20 else (256, 136)
        for expert in range(experts):
            raw = generator.integers(0, 256, (n, k // block, width), dtype=np.uint8)
            d = generator.uniform(0.0001, 0.001, raw.shape[:2]).astype("<f2")
            raw[:, :, :2] = d[..., None].view(np.uint8)
            raw = raw.reshape(n, -1)
            canonical = transcode_lut4(raw, source_type)
            official = gguf.quants.dequantize(
                raw, gguf.GGMLQuantizationType(source_type)
            )
            np.testing.assert_allclose(
                canonical.dequantize(), official, rtol=0.001, atol=0
            )
            codes.append(torch.from_numpy(canonical.codes.copy()))
            coefficients.append(torch.from_numpy(canonical.scales.copy()))
            bank.add(expert, torch.from_numpy(raw), rank=0, size=1, axis=0)
        bank.finalize()
        book = torch.tensor(gguf.quants.IQ4_NL.kvalues, dtype=torch.int32)
        integers.append(
            book[torch.stack(codes).long()].reshape(experts, n, k // 32, 32)
        )
        scales.append(torch.stack(coefficients).float())
        banks.append(bank)
    torch.manual_seed(1000 + m)
    x = torch.randn((m, k), dtype=torch.float16, device="cuda") * 0.125
    x[:, :32] = 0
    ids = torch.randint(experts, (m, top_k), device="cuda", dtype=index_dtype)
    activation = torch.empty((m, k // 32, 36), device="cuda", dtype=torch.uint8)
    hidden = torch.empty((m, top_k, n), device="cuda", dtype=torch.float16)

    def run(output=hidden, lanes=16):
        torch.ops._C.gguf_quantize_q8_1_sm70_out(activation, x)
        torch.ops._C.gguf_dp4a_lut4_gate_up_sm70_out(
            output,
            activation,
            ids,
            banks[0].weight_ptrs,
            banks[0].stat_ptrs,
            banks[1].weight_ptrs,
            banks[1].stat_ptrs,
            experts,
            lanes,
        )

    def expected():
        # Independent CPU integer-dot oracle; use the actual public Q8_1
        # activation packet, then multiply its FP16 scale after the integer sum.
        packet = activation.cpu()
        q = packet[:, :, 4:].contiguous().view(torch.int8).int()
        dx = packet[:, :, :4].contiguous().view(torch.float16)[:, :, 0].float()
        selected = ids.cpu().long()
        projections = []
        for weight, coefficient in zip(integers, scales):
            dot = (weight[selected] * q[:, None, None]).sum(-1).float()
            group_scale = coefficient[selected] * dx[:, None, None]
            projections.append((dot * group_scale).sum(-1).half())
        gate, up = projections
        return (torch.nn.functional.silu(gate.float()).half() * up).half()

    run()
    torch.testing.assert_close(hidden.cpu(), expected(), rtol=0.003, atol=0.003)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    x.copy_(torch.randn_like(x) * 0.125)
    ids.copy_((ids + 1) % experts)
    graph.replay()
    reference = expected()
    torch.testing.assert_close(hidden.cpu(), reference, rtol=0.003, atol=0.003)
    for lanes in (4, 8, 16):
        output = torch.empty((m, top_k, n // 32, 36), device="cuda", dtype=torch.uint8)
        run(output, lanes)
        packet = output.cpu()
        d = packet[..., :4].contiguous().view(torch.float16)[..., 0].float()
        q = packet[..., 4:].contiguous().view(torch.int8).float()
        reconstructed = (q * d[..., None]).reshape_as(reference)
        torch.testing.assert_close(
            reconstructed, reference.float(), rtol=0.015, atol=float(d.max()) + 0.003
        )
        assert torch.isfinite(reconstructed).all()
