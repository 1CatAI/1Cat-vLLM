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
