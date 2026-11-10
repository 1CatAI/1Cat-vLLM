# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboMind E4M3 group-size registration, arithmetic and graph coverage."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="SM70 required",
)


@pytest.mark.parametrize("group", [32, 64, 128])
@pytest.mark.parametrize("rows", [1, 8, 17, 32, 65])
@pytest.mark.parametrize("n,k", [(256, 256), (264, 264)])
def test_e4m3_group_sizes_against_fp64_and_graph(group, rows, n, k):
    import vllm._C  # noqa: F401

    torch.manual_seed(group + rows + n)
    weight = (torch.randn(n, k, device="cuda") * 2).to(torch.float8_e4m3fn)
    scales = 2.0 ** torch.randint(
        -8, -2, ((n + group - 1) // group, (k + group - 1) // group), device="cuda"
    )
    packed, packed_scales, meta = torch.ops._C.fp8_sm70_prepare(
        weight, scales, group, False
    )
    k_ld, q_ld = meta.tolist()
    dense = (
        weight.cpu().double()
        * scales.cpu()
        .double()
        .repeat_interleave(group, 0)
        .repeat_interleave(group, 1)[:n, :k]
    )
    x = torch.empty(rows, k, device="cuda", dtype=torch.float16)
    output = torch.empty(rows, n, device="cuda", dtype=torch.float16)

    def run():
        torch.ops._C.fp8_gemm_sm70_out(
            output, x, packed, packed_scales, group, k_ld, q_ld, False
        )

    x.normal_(0, 0.1)
    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for amplitude in (0.0, 0.1, 1.0):
        x.normal_(0, amplitude)
        output.fill_(float("nan"))
        graph.replay()
        assert torch.isfinite(output).all()
        reference = x.cpu().double() @ dense.T
        magnitude = x.cpu().double().abs() @ dense.abs().T
        rounded = reference.half()
        spacing = (
            torch.nextafter(rounded.abs(), torch.full_like(rounded, float("inf")))
            - rounded.abs()
        ).double()
        bound = spacing + magnitude * (32 * torch.finfo(torch.float32).eps)
        assert bool(((output.cpu().double() - reference).abs() <= bound).all())


@pytest.mark.parametrize("group", [32, 64, 128])
@pytest.mark.parametrize("rows", [5, 33])
def test_e4m3_group_sizes_fused_gate(group, rows):
    import vllm._C  # noqa: F401

    torch.manual_seed(group + rows)
    weight = torch.randn(256, 256, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.full((256, 1), 0.03125, device="cuda")
    packed, packed_scales, meta = torch.ops._C.fp8_sm70_prepare(
        weight, scales, group, True
    )
    x = torch.randn(rows, 256, device="cuda", dtype=torch.float16) * 0.1
    actual = torch.empty(rows, 128, device="cuda", dtype=torch.float16)
    torch.ops._C.fp8_gemm_sm70_out(
        actual, x, packed, packed_scales, group, *meta.tolist(), True
    )
    dot = x.double() @ (weight.double() * scales.double()).T
    reference = torch.nn.functional.silu(dot[:, :128]) * dot[:, 128:]
    relative_error = (actual.double() - reference).norm() / reference.norm()
    assert float(relative_error) < 1e-3


@pytest.mark.parametrize("group", [16, 256])
def test_e4m3_unsupported_group_is_rejected(group):
    import vllm._C  # noqa: F401

    weight = torch.empty(256, 256, device="cuda", dtype=torch.float8_e4m3fn)
    scales = torch.ones(256, 1, device="cuda")
    with pytest.raises(RuntimeError, match="group_size"):
        torch.ops._C.fp8_sm70_prepare(weight, scales, group, False)
