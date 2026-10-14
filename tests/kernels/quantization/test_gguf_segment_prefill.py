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


@pytest.mark.parametrize("source,dmv", [(12, True), (23, True), (14, False)])
def test_batch_storage_uses_native_then_retained_prefill(source, dmv):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 required")
    k, n = 2560, 2560
    raw = raw_weights(source, n, k)
    reference_weight = torch.from_numpy(
        gguf.quants.dequantize(raw, gguf.GGMLQuantizationType(source))
    ).cuda()
    config = SimpleNamespace(
        kernel_config=KernelConfig(sm70_gguf={"small_m_hmma_batch": True}),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    with set_current_vllm_config(config):
        projections = prepare_gguf_projections(
            [(torch.from_numpy(raw).cuda(), source)],
            torch.float16,
            True,
            8,
            dmv_enabled=dmv,
        )
        assert projections[0].gguf_batch_stats is not None
        serialized = prepared_projection_arguments(projections)
        assert len(serialized) == 6 and serialized[3][0] == (6 if dmv else 5)
        for m in (5, 8, 9, 10, 20, 30, 40, 41):
            x = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.125
            apply_prepared_gguf_projections(x, projections)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = apply_prepared_gguf_projections(x, projections)
            for shift in (0.0, 0.125):
                x.normal_(mean=shift, std=0.125)
                graph.replay()
                expected = x.float() @ reference_weight.T
                torch.accelerator.synchronize()
                assert bool(torch.isfinite(out).all())
                assert float((out.float() - expected).norm() / expected.norm()) < 0.003


@pytest.mark.parametrize(
    "source,k,n",
    [(13, 2560, 2560), (14, 2560, 2560), (14, 2560, 1536), (14, 1536, 2560)],
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


def test_mixed_segment_output_views_and_unqualified_fallback():
    config = SimpleNamespace(
        kernel_config=KernelConfig(),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    with set_current_vllm_config(config):
        # A calibrated Q6 projection plus an uncalibrated Q4 projection:
        # only one segment uses BLAS, whose native API requires contiguous out.
        sources = [
            (torch.from_numpy(raw_weights(source, n, 2560)).cuda(), source)
            for source, n in ((14, 1536), (12, 128))
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
