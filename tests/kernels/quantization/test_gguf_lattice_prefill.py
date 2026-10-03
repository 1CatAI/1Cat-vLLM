# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from test_gguf_turbomind_lattice import prepare, projection

from vllm import _custom_ops  # noqa: F401

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("weight_type", [16, 17, 18, 19, 21, 22, 29])
def test_lattice_dequant_fp32_blas_and_graph(weight_type):
    torch._dynamo.reset()
    p = projection(weight_type)
    w, s, _, _ = prepare(p)
    n, k = p.shape
    scratch = torch.empty((k, n), dtype=torch.float16, device="cuda")
    expected_weight = torch.from_numpy(p.dequantize().T.copy()).half().cuda()
    torch.ops._C.gguf_lattice_dequantize_sm70_out(
        scratch, w, s, weight_type, p.group_size
    )
    torch.testing.assert_close(scratch, expected_weight, rtol=0, atol=0)
    for m in (1, 128, 512):
        torch.manual_seed(20261003 + m)
        x = (torch.randn((m, k), device="cuda") * 0.125).half()
        out = torch.empty((m, n), dtype=torch.float16, device="cuda")
        expected = x.float() @ expected_weight.float()

        def run(out=out, x=x):
            torch.ops._C.gguf_lattice_blas_sm70_out(
                out, x, w, s, weight_type, scratch, p.group_size
            )
            return out

        run()
        torch.testing.assert_close(out.float(), expected, rtol=0.003, atol=0.0002)
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(graph, stream=stream):
            run()
        graph.replay()
        torch.testing.assert_close(out.float(), expected, rtol=0.003, atol=0.0002)
        compiled = torch.compile(run, backend="eager", fullgraph=True)
        torch.testing.assert_close(
            compiled().float(), expected, rtol=0.003, atol=0.0002
        )
