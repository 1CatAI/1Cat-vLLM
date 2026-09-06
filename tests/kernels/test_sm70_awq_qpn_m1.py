# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent prepared-layout oracle for the opt-in Qwen3.8 AWQ M1 op."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires SM70 and its native extension",
)


def _bank(k, n, compact, experts):
    weight = torch.zeros((512, k, n // 8), dtype=torch.int32, device="cuda")
    metadata = torch.zeros(
        (512, k // 32, n, 3) if compact else (512, k // 32, n),
        dtype=torch.uint8 if compact else torch.int32,
        device="cuda",
    )
    decoded = {}
    for expert in experts:
        codes = torch.randint(0, 16, (k, n), device="cuda")
        zeros = torch.randint(0, 16, (k // 32, n), device="cuda")
        scales = (torch.rand(k // 32, n, device="cuda") * 0.01 + 0.001).half()
        bias = (-zeros.half() * scales).half()
        # Independently build the N32/K8 prepared tile layout and nibble order.
        values = codes.reshape(k // 8, 8, n // 32, 32).permute(2, 0, 3, 1)
        packed = torch.zeros((n // 32, k // 8, 32), dtype=torch.int64, device="cuda")
        for logical, physical in enumerate((0, 4, 1, 5, 2, 6, 3, 7)):
            packed |= values[..., logical] << (4 * physical)
        weight[expert].copy_(packed.int().reshape(k, n // 8))
        scale_bits = scales.view(torch.int16).int() & 0xFFFF
        if compact:
            metadata[expert].copy_(
                torch.stack((scale_bits & 255, scale_bits >> 8, zeros), -1).byte()
            )
        else:
            bias_bits = bias.view(torch.int16).int() & 0xFFFF
            metadata[expert].copy_(scale_bits | (bias_bits << 16))
        group = torch.arange(k, device="cuda") // 32
        decoded[expert] = (
            codes.double() * scales[group].double() + bias[group].double()
        ).half()
    return weight, metadata, decoded


@pytest.mark.parametrize("compact", [False, True])
def test_awq_qpn_m1_reference_graph_and_admission(compact):
    from vllm import _sm70_ops as ops

    assert hasattr(torch.ops._C, "awq_moe_qpn_m1_sm70_out")
    torch.manual_seed(731)
    experts = (0, 1, 257, 511)
    w13, s13, ref13 = _bank(2560, 320, compact, experts)
    w2, s2, ref2 = _bank(160, 2560, compact, experts)
    x = torch.zeros((1, 2560), dtype=torch.float16, device="cuda")
    ids = torch.tensor(
        [[511, 0, 257, 1, 511, 1, 0, 257, 0, 511]],
        dtype=torch.int32,
        device="cuda",
    )
    topk = torch.softmax(torch.randn(1, 10, device="cuda"), dim=-1)
    out = torch.empty_like(x)
    intermediate = torch.empty((10, 160), dtype=x.dtype, device="cuda")

    def run():
        ops.awq_moe_qpn_m1_sm70_out(out, intermediate, x, w13, s13, w2, s2, ids, topk)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    # One-hot inputs isolate weight reads at group/partition boundaries.
    for index in (0, 31, 32, 2559):
        x.zero_()
        x[0, index] = 1
        run()
        eager_out, eager_mid = out.clone(), intermediate.clone()
        expected_mid, expected_routes = [], []
        for expert in ids[0].tolist():
            value = ref13[expert][index].float()
            silu = (value[::2] / (1 + torch.exp(-value[::2]))).half()
            activation = (silu * value[1::2].half()).half()
            expected_mid.append(activation)
            expected_routes.append((activation.double() @ ref2[expert].double()).half())
        torch.testing.assert_close(
            intermediate, torch.stack(expected_mid), rtol=0, atol=0
        )
        reference = (
            (torch.stack(expected_routes).double() * topk[0].double().unsqueeze(1))
            .sum(0)
            .half()
            .unsqueeze(0)
        )
        # FP64 dot oracle is not the legacy reduction. Allow the materialized
        # FP16 route/output roundings, including the minimum subnormal floor.
        difference = (out.double() - reference.double()).abs()
        assert difference.max() <= reference.double().abs().max() * 2e-3 + 2**-24
        assert (
            difference.norm() <= reference.double().norm() * 2e-3 + 2560**0.5 * 2**-24
        )
        graph.replay()
        assert torch.equal(out, eager_out)
        assert torch.equal(intermediate, eager_mid)
    ids.fill_(-1)
    graph.replay()
    assert torch.count_nonzero(out) == 0
    assert torch.count_nonzero(intermediate) == 0
    ids.fill_(512)
    graph.replay()
    assert torch.count_nonzero(out) == 0
    assert torch.count_nonzero(intermediate) == 0
    # Tracing must resolve the native fake implementation without an external
    # research DSO. Actual model Inductor/graph validation is a separate gate.
    torch.compile(run, backend="eager", fullgraph=True)()
    valid = [out, intermediate, x, w13, s13, w2, s2, ids, topk]
    unaligned = torch.empty(2561, dtype=x.dtype, device="cuda")[1:].view_as(x)
    for index, replacement in (
        (0, x),
        (2, unaligned),
        (7, ids.long()),
        (8, topk.half()),
    ):
        args = list(valid)
        args[index] = replacement
        with pytest.raises(RuntimeError):
            ops.awq_moe_qpn_m1_sm70_out(*args)
