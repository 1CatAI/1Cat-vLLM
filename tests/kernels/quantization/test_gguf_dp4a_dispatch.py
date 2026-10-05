# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest
import torch

from vllm.model_executor.kernels.gguf import dense_dp4a_capabilities
from vllm.model_executor.layers.quantization.gguf_turbomind import (
    GGUFPreparedProjection,
    _prepared_gguf_mixed_projection,
    apply_prepared_gguf_projections,
    prepared_projection_arguments,
)
from vllm.transformers_utils.gguf_tensor_reader import quant_size


@pytest.mark.parametrize("kind", [12, 23])
def test_dense_dp4a_reasons_and_exact_bands(kind):
    caps = dense_dp4a_capabilities(kind, 2560, 160, torch.float16)
    assert [c.min_m for c in caps] == [5, 20]
    assert all(c.max_m == c.min_m for c in caps)
    assert all(c.reason is None for c in caps)
    assert not any(c.supports_m(8) for c in caps)
    for kwargs, reason in [
        ({"enabled": False}, "disabled_by_kernel_config"),
        ({"compute_capability": 80}, "requires_sm70_device"),
    ]:
        assert all(
            c.reason == reason
            for c in dense_dp4a_capabilities(kind, 2560, 160, torch.float16, **kwargs)
        )
    assert all(
        c.reason == "requires_fp16_activations"
        for c in dense_dp4a_capabilities(kind, 2560, 160, torch.bfloat16)
    )
    assert all(
        c.reason == "integer_dense_shape_or_source_has_no_calibration"
        for c in dense_dp4a_capabilities(kind, 2560, 2560, torch.float16)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mixed_projection_encodes_once_and_keeps_canonical_fallback():
    projections = []
    for kind in (12, 23):
        _, size = quant_size(kind)
        rng = np.random.default_rng(kind)
        blocks = rng.integers(0, 256, (160, 10, size), dtype=np.uint8)
        d = rng.uniform(0.0001, 0.001, (160, 10)).astype("<f2")
        blocks[:, :, :2] = d[..., None].view(np.uint8)
        if kind == 12:
            blocks[:, :, 2:4] = (d * np.float16(0.5))[..., None].view(np.uint8)
        projections.append(
            GGUFPreparedProjection(
                torch.from_numpy(blocks.reshape(160, -1)).cuda(),
                kind,
                torch.float16,
                True,
                8,
                dp4a_enabled=True,
            )
        )
    assert all(len(p.dp4a_payload) == 5 for p in projections)
    for m in (1, 5, 20, 64):
        x = torch.randn(m, 2560, device="cuda", dtype=torch.float16)
        out = apply_prepared_gguf_projections(x, projections)
        assert out.shape == (m, 320) and out.is_contiguous()
        assert torch.isfinite(out).all()
        if m not in (5, 20):
            reference = _prepared_gguf_mixed_projection(
                x, *prepared_projection_arguments(projections)
            )
            torch.testing.assert_close(out, reference, rtol=0, atol=0)
        else:
            expected = torch.cat([p(x) for p in projections], dim=-1)
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                replay = apply_prepared_gguf_projections(x, projections)
            x.copy_(torch.randn_like(x))
            graph.replay()
            expected = apply_prepared_gguf_projections(x, projections)
            torch.testing.assert_close(replay, expected, rtol=0, atol=0)
