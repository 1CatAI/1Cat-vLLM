# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Segment storage retains calibrated prefill policy across runtime M changes."""

from types import SimpleNamespace

import gguf
import pytest
import torch

from tests.kernels.quantization.test_gguf_dense_segments import raw_weights
from tools.sm70.native_trace import NativeDispatchTrace
from vllm.config import set_current_vllm_config
from vllm.config.kernel import KernelConfig
from vllm.model_executor.layers.quantization.gguf_turbomind import (
    _prepared_gguf_mixed_projection,
    apply_prepared_gguf_projections,
    prepare_gguf_projections,
    prepared_projection_arguments,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize(
    "source,k,n",
    [
        (13, 2560, 2560),
        (14, 2560, 2560),
        (14, 2560, 1536),
        (14, 1536, 2560),
        (13, 2560, 4096),
        (14, 2560, 4096),
        (14, 2560, 3584),
        (13, 2560, 3328),
        (13, 2560, 3072),
    ],
)
def test_segment_storage_prefill_and_decode(source, k, n):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 required")
    torch._dynamo.reset()
    config = SimpleNamespace(
        kernel_config=KernelConfig(),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    raw = raw_weights(source, n, k)
    reference_weight = torch.from_numpy(
        gguf.quants.dequantize(raw, gguf.GGMLQuantizationType(source))
    ).cuda()
    with set_current_vllm_config(config):
        projections = prepare_gguf_projections(
            [(torch.from_numpy(raw).cuda(), source)], torch.float16, True, 8
        )
        projection = projections[0]
        assert hasattr(projection, "segment_format")
        serialized = prepared_projection_arguments(projections)
        assert serialized[3][0] == 3 and serialized[5] == [2048, -1]

        def run(x):
            return apply_prepared_gguf_projections(x, projections)

        compiled = torch.compile(run, backend="eager", fullgraph=True, dynamic=True)
        for m in (2047, 2048, 2049, 4097, 8, 20, 33):
            x = (torch.randn(m, k, device="cuda") * 0.125).half()
            operator = (
                "gguf_affine_blas_sm70_out"
                if m >= 2048
                else "gguf_dense_segments_sm70_out"
                if m in (8, 20)
                else "gguf_affine_gemm_sm70_out"
            )
            with NativeDispatchTrace() as trace:
                out = _prepared_gguf_mixed_projection(x, *serialized)
            assert any(
                r["operator"] == f"_C::{operator}" and r["evidence"] == "cuda_dispatch"
                for r in trace.records
            )

            def check(value, x=x):
                expected = x[:32].float() @ reference_weight.float().T
                error = (value[:32].float() - expected).norm(dim=-1)
                relative = error / expected.norm(dim=-1).clamp_min(1e-12)
                assert bool(torch.isfinite(value).all())
                assert float(relative.max()) <= 0.003

            check(out)
            check(compiled(x))
            check(projection(x))
            if m == 2048:
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    run(x)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    graph_out = run(x)
                x.mul_(-0.5)
                graph.replay()
                check(graph_out)


@pytest.mark.parametrize("segments", [((14, 1536), (12, 128)), ((13, 3328), (14, 256))])
def test_mixed_segment_output_views_and_unqualified_fallback(segments):
    config = SimpleNamespace(
        kernel_config=KernelConfig(),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    with set_current_vllm_config(config):
        # A calibrated projection plus an uncalibrated projection:
        # only one segment uses BLAS, whose native API requires contiguous out.
        sources = [
            (torch.from_numpy(raw_weights(source, n, 2560)).cuda(), source)
            for source, n in segments
        ]
        projections = prepare_gguf_projections(sources, torch.float16, True, 8)
        assert all(hasattr(p, "segment_format") for p in projections)
        x = (torch.randn(2049, 2560, device="cuda") * 0.125).half()
        with NativeDispatchTrace() as trace:
            mixed = _prepared_gguf_mixed_projection(
                x, *prepared_projection_arguments(projections)
            )
        operators = [
            r["operator"] for r in trace.records if r["evidence"] == "cuda_dispatch"
        ]
        assert "_C::gguf_affine_blas_sm70_out" in operators
        assert "_C::gguf_affine_gemm_sm70_out" in operators
        independent = torch.cat([p(x) for p in projections], dim=-1)
        torch.testing.assert_close(mixed, independent, rtol=0, atol=0)
        assert mixed.is_contiguous() and bool(torch.isfinite(mixed).all())
