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


@pytest.mark.parametrize(
    "weight_type,n,k,minimum",
    [
        (17, 160, 2560, 8),
        (18, 160, 2560, 8),
        (21, 1536, 2560, 2048),
        (21, 4352, 5120, 512),
        (21, 5120, 4352, 512),
    ],
)
def test_calibrated_lattice_routes_share_affine_workspace(weight_type, n, k, minimum):
    import numpy as np
    from test_gguf_lattice_transcode import source

    from vllm.model_executor.kernels.linear import (
        Sm70GgufLatticeConfig,
        TurboMindGgufLatticeKernel,
    )
    from vllm.model_executor.kernels.linear.mixed_precision.sm70_gguf import (
        _get_affine_blas_workspace,
    )
    from vllm.model_executor.layers.quantization.gguf_lattice_transcode import (
        transcode_lattice,
    )
    from vllm.scalar_type import scalar_types

    torch._dynamo.reset()
    p = transcode_lattice(
        source(weight_type, n=n, k=k, scale=0.0009765625), weight_type
    )
    config = Sm70GgufLatticeConfig(
        (k, n),
        (k, n),
        scalar_types.uint2,
        torch.float16,
        p.group_size,
        False,
        False,
        source_type=weight_type,
    )
    kernel = TurboMindGgufLatticeKernel(config, "codes", "scales")
    layer = torch.nn.Module()
    codes, metadata = p.mma884_storage()
    signed = metadata.view({2: np.int16, 4: np.int32, 8: np.int64}[metadata.itemsize])
    for name, data in (("codes", codes), ("scales", signed)):
        layer.register_parameter(
            name, torch.nn.Parameter(torch.from_numpy(data).cuda(), requires_grad=False)
        )
    kernel.process_weights_after_loading(layer)
    assert all(c.reason is None for c in kernel.prefill_capabilities)
    assert (
        layer.gguf_tm_blas_workspace.data_ptr()
        == _get_affine_blas_workspace(layer.codes).data_ptr()
    )
    boundaries = (
        (128, 512)
        if minimum == 512
        else (1024, 2048)
        if weight_type == 21
        else (4, 8, 1024, 2048, 4096)
    )
    dense = torch.from_numpy(p.dequantize()).half().cuda()
    for m in boundaries:
        admitted = any(c.supports_m(m) for c in kernel.prefill_capabilities)
        assert admitted == (
            m >= minimum if weight_type == 21 else 8 <= m <= 1024 or m >= 4096
        )
        x = (torch.randn((m, k), device="cuda") * 0.125).half()
        expected = x.float() @ dense.float().T
        run = lambda x: kernel.apply_weights(layer, x)
        torch.testing.assert_close(run(x).float(), expected, rtol=0.003, atol=0.003)
        compiled = torch.compile(run, backend="eager", fullgraph=True)
        torch.testing.assert_close(
            compiled(x).float(), expected, rtol=0.003, atol=0.003
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = run(x)
        graph.replay()
        torch.testing.assert_close(out.float(), expected, rtol=0.003, atol=0.003)
