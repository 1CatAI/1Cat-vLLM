# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
LAYER_NAME = "language_model.model.layers.0.linear_attn"


def _dump(x):
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn  # noqa: F401

    return torch.ops.vllm.sm70_gdn_projection_dump(x, "proj_core_in", LAYER_NAME)


def test_gdn_projection_dump_obeys_nonaliasing_schema():
    x = torch.randn(8, 128, device="cuda", dtype=torch.float16)
    out = _dump(x)
    torch.library.opcheck(
        torch.ops.vllm.sm70_gdn_projection_dump.default,
        (x, "proj_core_in", LAYER_NAME),
        test_utils=("test_schema", "test_faketensor"),
    )
    assert torch.equal(out, x)
    out.zero_()
    assert torch.count_nonzero(x) > 0


def test_gdn_projection_dump_preserves_compiled_live_input():
    def forward(x, z):
        saved = x.float() * z.float()
        projected = _dump(x)
        return saved, projected.float() + z.float()

    x = torch.randn(8, 128, device="cuda", dtype=torch.float16)
    z = torch.randn_like(x)
    expected = forward(x, z)
    actual = torch.compile(forward, backend="aot_eager", fullgraph=True)(x, z)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, atol=0, rtol=0)


def test_gdn_projection_dump_graph_keeps_owned_output():
    x = torch.randn(8, 128, device="cuda", dtype=torch.float16)
    _dump(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _dump(x)
    for value in (1.0, -2.0, 4.0):
        x.fill_(value)
        graph.replay()
        assert torch.equal(out, x)
        out.zero_()
        assert torch.all(x == value)
