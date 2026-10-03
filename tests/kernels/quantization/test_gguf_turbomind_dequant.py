# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm import _custom_ops  # noqa: F401
from vllm.model_executor.kernels.linear import (
    Sm70GgufAffineConfig,
    TurboMindGgufAffineKernel,
)
from vllm.model_executor.layers.quantization.gguf_transcode import AffineGGUFProjection
from vllm.scalar_type import scalar_types
from vllm.sm70_profiles.acceleration import loaded_linear_kernels

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_canonical_prefill_capability_is_reported():
    config = Sm70GgufAffineConfig(
        (256, 128),
        (256, 128),
        scalar_types.uint4,
        torch.float16,
        32,
        True,
        False,
        source_type=3,
    )
    kernel = TurboMindGgufAffineKernel(config, "codes", "scales", "mins")
    layer = torch.nn.Module()
    for name, data in {
        "codes": torch.zeros((128, 256), dtype=torch.uint8, device="cuda"),
        "scales": torch.ones((128, 8), dtype=torch.float16, device="cuda"),
        "mins": torch.zeros((128, 8), dtype=torch.float16, device="cuda"),
    }.items():
        layer.register_parameter(name, torch.nn.Parameter(data, requires_grad=False))
    kernel.process_weights_after_loading(layer)
    layer.quant_method = SimpleNamespace(kernel=kernel)
    candidates = next(iter(loaded_linear_kernels(layer).values()))[
        "operator_candidates"
    ]
    blas = next(c for c in candidates if c["operator"] == "gguf_affine_blas_sm70_out")
    assert blas["source_type"] == "Q4_1" and blas["family"] == "affine_integer"
    assert blas["min_m"] == 512 and blas["graph_safe"] and blas["reason"] is None


@pytest.mark.parametrize(
    "bits,group", [(2, 16), (2, 32), (3, 16), (4, 32), (5, 32), (6, 16), (8, 32)]
)
def test_canonical_dequant_blas_and_graph(bits, group):
    # Each storage contract is an independent tracing test. Avoid exhausting
    # Dynamo's shared code-object variant budget across parametrized cases.
    torch._dynamo.reset()
    n, k = 160, 2560
    rng = np.random.default_rng(20261003 + bits + group)
    codes = rng.integers(0, 1 << bits, (n, k), dtype=np.uint8)
    scales = (rng.integers(-7, 8, (n, k // group)) / 2048).astype(np.float16)
    mins = (rng.integers(-7, 8, scales.shape) / 2048).astype(np.float16)
    if bits == 3:
        mins = (scales.astype(np.float32) * -4).astype(np.float16)
    else:
        scales[:, 0], mins[:, 0] = 0, 0.125
    p = AffineGGUFProjection(0, bits, group, codes, scales, mins)
    w, s, _ = torch.ops._C.gguf_affine_sm70_prepare(
        torch.from_numpy(codes).cuda(),
        torch.from_numpy(scales).cuda(),
        torch.from_numpy(mins).cuda(),
        bits,
        group,
    )
    scratch = torch.empty((k, n), dtype=torch.float16, device="cuda")
    expected_weight = torch.from_numpy(p.dequantize().T.copy()).half().cuda()
    torch.ops._C.gguf_affine_dequantize_sm70_out(scratch, w, s, bits, group)
    torch.testing.assert_close(scratch, expected_weight, rtol=0, atol=0)
    for m in (1, 512):
        torch.manual_seed(20261003 + m)
        x = (torch.randn((m, k), device="cuda") * 0.125).half()
        out = torch.empty((m, n), dtype=torch.float16, device="cuda")
        expected = x.float() @ expected_weight.float()

        def run(x=x, out=out):
            torch.ops._C.gguf_affine_blas_sm70_out(out, x, w, s, bits, scratch, group)
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
