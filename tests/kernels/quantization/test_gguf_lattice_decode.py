# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from test_gguf_turbomind_lattice import prepare, projection

from vllm import _custom_ops  # noqa: F401

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("weight_type", [16, 17, 18, 19, 21, 22, 29])
def test_grouped_lattice_vec_fp32_oracle_empty_and_graph(weight_type):
    torch._dynamo.reset()
    e, n, k = 4, 160, 2560
    projections = [projection(weight_type, expert=i) for i in range(e)]
    prepared = [prepare(p) for p in projections]
    weights = torch.stack([p[0] for p in prepared])
    stats = torch.stack([p[1] for p in prepared])
    wp, sp = torch.ops._C.awq_moe_build_strided_ptrs(
        weights, stats, *prepared[0][2:], e
    )
    dense = [torch.from_numpy(p.dequantize()).half().cuda() for p in projections]
    for m in (1, 8, 64, 128):
        boundaries = [0, m // 4, m // 4, m, m]
        offsets = torch.tensor(boundaries, dtype=torch.int32, device="cuda")
        x = (torch.randn((m, k), device="cuda") * 0.125).half()
        out = torch.empty((m, n), dtype=torch.float16, device="cuda")
        expected = torch.empty((m, n), device="cuda")
        for expert, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            if start < end:
                expected[start:end] = x[start:end].float() @ dense[expert].float().T

        def run(out=out, x=x, offsets=offsets):
            torch.ops._C.gguf_lattice_grouped_vec_sm70_out(
                out, x, offsets, wp, sp, weight_type, e, projections[0].group_size
            )
            return out

        run()
        torch.testing.assert_close(out.float(), expected, rtol=0.003, atol=0.003)
        assert (out.float() - expected).norm() <= expected.norm() * 0.003 + 1e-6
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        graph.replay()
        torch.testing.assert_close(out.float(), expected, rtol=0.003, atol=0.003)
        compiled = torch.compile(run, backend="eager", fullgraph=True)
        torch.testing.assert_close(compiled().float(), expected, rtol=0.003, atol=0.003)


def test_grouped_lattice_capability_bands_and_reasons():
    from vllm.model_executor.kernels.gguf import (
        lattice_grouped_capabilities,
        select_lattice_grouped_capability,
    )

    for source in (17, 18):
        for e, cases in (
            (4, ((8, True), (16, False))),
            (512, ((128, True), (256, False), (512, True), (8192, False))),
        ):
            caps = lattice_grouped_capabilities(source, 2560, 160, e, torch.float16)
            for m, vector in cases:
                chosen = select_lattice_grouped_capability(caps, m)
                assert ("_vec_" in chosen.operator) == vector
    unknown = lattice_grouped_capabilities(18, 2560, 160, 8, torch.float16)
    assert unknown[1].reason == "grouped_vector_shape_has_no_calibration"
    assert "_gemm_" in select_lattice_grouped_capability(unknown, 1).operator
    disabled = lattice_grouped_capabilities(
        18, 2560, 160, 512, torch.float16, enabled=False
    )
    assert all(c.reason == "disabled_by_kernel_config" for c in disabled)
    with pytest.raises(ValueError, match="No prepared lattice"):
        select_lattice_grouped_capability(disabled, 1)
    bad_dtype = lattice_grouped_capabilities(18, 2560, 160, 512, torch.bfloat16)
    assert all(c.reason == "requires_fp16_activations" for c in bad_dtype)
