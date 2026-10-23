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
    assert (
        blas["min_m"] == 512
        and blas["graph_safe"]
        and blas["reason"] == "local_shape_has_no_prefill_calibration"
    )


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


@pytest.mark.parametrize(
    "bits,group,n",
    [(4, 32, 1536), (5, 32, 3072), (5, 32, 3328), (6, 16, 3584), (6, 16, 4096)],
)
def test_bounded_prefill_blas_window_neighbors_and_graph(bits, group, n):
    """The algorithm window keeps the family's FP32/FP16 numeric contract.

    This checks arithmetic, tails and graph replay; source/binary identity and
    actual native dispatch/performance are recorded separately in the worklog.
    """
    k = 2560
    rng = np.random.default_rng(9683 + n + bits)
    codes = rng.integers(0, 1 << bits, (n, k), dtype=np.uint8)
    scales = (rng.integers(-7, 8, (n, k // group)) / 2048).astype(np.float16)
    mins = (rng.integers(-7, 8, scales.shape) / 2048).astype(np.float16)
    w, s, _ = torch.ops._C.gguf_affine_sm70_prepare(
        torch.from_numpy(codes).cuda(),
        torch.from_numpy(scales).cuda(),
        torch.from_numpy(mins).cuda(),
        bits,
        group,
    )
    scratch = torch.empty(k, n, dtype=torch.float16, device="cuda")
    for m in (5999, 6000, 6001, 6108, 6199, 6200, 6201):
        x = (torch.randn(m, k, device="cuda") * 0.125).half()
        out = torch.empty(m, n, dtype=torch.float16, device="cuda")

        def run(x=x, out=out):
            torch.ops._C.gguf_affine_blas_sm70_out(out, x, w, s, bits, scratch, group)

        def check(x=x, out=out, m=m):
            # Independently check both ends, including the final partial row tile.
            sample = torch.cat((x[:16], x[m - 16 :])).float()
            expected = sample @ scratch.float()
            observed = torch.cat((out[:16], out[m - 16 :])).float()
            row_l2 = (observed - expected).norm(dim=1) / expected.norm(dim=1).clamp_min(
                1e-6
            )
            assert torch.isfinite(out).all()
            assert row_l2.max() <= 0.003

        run()
        check()
        if m == 6108:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            x.mul_(-0.5)
            graph.replay()
            check()


@pytest.mark.parametrize(
    "bits,group,k,n,minimum",
    [
        (5, 32, 5120, 1536, 2048),
        (5, 32, 2560, 2560, 2048),
        (6, 16, 2560, 2560, 2048),
        (6, 16, 2560, 1536, 2048),
        (6, 16, 1536, 2560, 2048),
        (4, 32, 2560, 1536, 4096),
        (4, 32, 1536, 2560, 4096),
        (4, 32, 2560, 3072, 4096),
        (4, 32, 2560, 4096, 4096),
        (5, 32, 2560, 1536, 4096),
        (5, 32, 1536, 2560, 4096),
        (8, 32, 1536, 2560, 4096),
        (8, 32, 160, 2560, 6144),
    ],
)
def test_calibrated_prefill_framework_routes_and_shared_workspace(
    bits, group, k, n, minimum
):
    from tools.sm70.native_trace import NativeDispatchTrace
    from vllm.model_executor.layers.quantization import gguf_turbomind  # noqa: F401

    torch._dynamo.reset()
    config = Sm70GgufAffineConfig(
        (k, n),
        (k, n),
        {
            4: scalar_types.uint4,
            5: scalar_types.uint5,
            6: scalar_types.uint6,
            8: scalar_types.uint8,
        }[bits],
        torch.float16,
        group,
        True,
        False,
        source_type={4: 12, 5: 13, 6: 14, 8: 8}[bits],
    )
    layers, kernels = [], []
    for _ in range(2):
        layer = torch.nn.Module()
        kernel = TurboMindGgufAffineKernel(config, "codes", "scales", "mins")
        for name, data in {
            "codes": torch.zeros((n, k), dtype=torch.uint8, device="cuda"),
            "scales": torch.ones((n, k // group), dtype=torch.float16, device="cuda"),
            "mins": torch.full(
                (n, k // group), 0.00390625, dtype=torch.float16, device="cuda"
            ),
        }.items():
            layer.register_parameter(
                name, torch.nn.Parameter(data, requires_grad=False)
            )
        kernel.process_weights_after_loading(layer)
        assert kernel.prefill_capability.min_m == minimum
        assert kernel.prefill_capability.reason is None
        layers.append(layer)
        kernels.append(kernel)
    assert (
        layers[0].gguf_tm_blas_workspace.data_ptr()
        == layers[1].gguf_tm_blas_workspace.data_ptr()
    )
    cases = (minimum - 1, minimum, minimum + 1, 6108 if minimum == 4096 else 4097)
    for m in (512, 2048) if k == 5120 else cases:
        x = (torch.randn((m, k), device="cuda") * 0.125).half()
        expected = (x.float().sum(-1, keepdim=True) * 0.00390625).expand(m, n)
        run = lambda x: kernels[0].apply_weights(layers[0], x)
        with NativeDispatchTrace() as trace:
            eager = run(x)
        operator = "blas" if m >= minimum else "gemm"
        assert any(
            r["operator"] == f"_C::gguf_affine_{operator}_sm70_out"
            and r["evidence"] == "cuda_dispatch"
            for r in trace.records
        )
        torch.testing.assert_close(eager.float(), expected, rtol=0.003, atol=0.0002)
        compiled = torch.compile(run, backend="eager", fullgraph=True)
        torch.testing.assert_close(
            compiled(x).float(), expected, rtol=0.003, atol=0.0002
        )
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run(x)
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(graph, stream=stream):
            out = run(x)
        graph.replay()
        torch.testing.assert_close(out.float(), expected, rtol=0.003, atol=0.0002)
        # The serving graph passes the calibrated bands through its opaque op.
        layer = layers[0]
        opaque = torch.ops.vllm.prepared_gguf_projection(
            x,
            layer.codes,
            layer.scales,
            None,
            0,
            bits,
            group,
            layer.gguf_tm_k_ld,
            layer.gguf_tm_q_ld,
            n,
            n,
            [],
            [minimum, -1],
        )
        torch.testing.assert_close(opaque.float(), expected, rtol=0.003, atol=0.0002)
        x.mul_(-0.5)
        graph.replay()
        expected = (x.float().sum(-1, keepdim=True) * 0.00390625).expand(m, n)
        torch.testing.assert_close(out.float(), expected, rtol=0.003, atol=0.0002)
