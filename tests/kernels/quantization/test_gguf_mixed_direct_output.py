# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from test_gguf_turbomind_bitplanes import source as bitplane_source
from test_gguf_turbomind_model import source as model_source

from vllm.model_executor.layers.quantization.gguf_turbomind import (
    GGUFPreparedProjection,
    apply_prepared_gguf_projections,
    mixed_projection_capabilities,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def source(t, n=64):
    if t == 14:
        return bitplane_source(t, n=n, k=512)
    data = model_source(t, n=n, k=512)
    return data[:, :288] if t == 12 else data


@pytest.mark.parametrize("types", [(14, 12), (12, 23), (18, 20)])
@pytest.mark.parametrize("m", [5, 20, 32])
def test_mixed_output_matches_independent_projections_in_graph(types, m):
    torch.manual_seed(20261005)
    projections = torch.nn.ModuleList(
        [
            GGUFPreparedProjection(
                torch.from_numpy(source(t, n=64)).cuda(), t, torch.float16, True, 8
            )
            for t in types
        ]
    )
    assert all(c.reason is None for c in mixed_projection_capabilities(projections))
    x = (torch.randn((1, m, 512), device="cuda") * 0.125).half()
    expected = torch.cat([p(x) for p in projections], dim=-1)
    actual = apply_prepared_gguf_projections(x, projections)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.is_contiguous()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = apply_prepared_gguf_projections(x, projections)
    graph.replay()
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)
    compiled = torch.compile(
        lambda x: apply_prepared_gguf_projections(x, projections),
        backend="eager",
        fullgraph=True,
    )
    torch.testing.assert_close(compiled(x), expected, rtol=0, atol=0)


def test_padded_projection_preserves_fallback_and_reports_reason():
    projections = torch.nn.ModuleList(
        [
            GGUFPreparedProjection(
                torch.from_numpy(source(t, n=n)).cuda(), t, torch.float16, True, 8
            )
            for t, n in ((12, 40), (23, 64))
        ]
    )
    capabilities = mixed_projection_capabilities(projections)
    assert capabilities[0].reason == "mixed_output_has_padded_projection"
    x = torch.randn((5, 512), device="cuda").half()
    expected = torch.cat([p(x) for p in projections], dim=-1)
    torch.testing.assert_close(
        apply_prepared_gguf_projections(x, projections), expected, rtol=0, atol=0
    )
