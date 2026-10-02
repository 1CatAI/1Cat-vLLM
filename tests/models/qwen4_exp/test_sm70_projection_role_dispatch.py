# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.models.qwen4_exp.nvidia import sm70_fp16_gemv as gemv


@pytest.mark.parametrize(
    "role,policy",
    [("layers.0.linear_attn.out_proj", 1), ("layers.3.self_attn.o_proj", 0), ("", 0)],
)
def test_same_shape_roles_reach_kernel(monkeypatch, role, policy):
    launches = []

    class Kernel:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: launches.append((grid, kwargs))

    monkeypatch.setattr(gemv, "_runtime_ok", lambda *args: True)
    monkeypatch.setattr(gemv, "_qwen38_fp16_row_gemv_kernel", Kernel())
    x = torch.empty(1, 1536, dtype=torch.float16, device="meta")
    w = torch.empty(2560, 1536, dtype=torch.float16, device="meta")
    out = gemv._qwen38_sm70_fp16_gemv(x, w, role)
    assert out.shape == (1, 2560)
    assert launches == [
        ((2560,), {"K": 1536, "BLOCK_K": 512, "LOAD_POLICY": policy, "num_warps": 4})
    ]


def test_linear_method_forwards_role(monkeypatch):
    seen = []

    def record_role(x, weight, role, packed_router, dense_batch):
        assert packed_router is None and dense_batch is False
        seen.append(role)
        return x

    monkeypatch.setattr(gemv, "use_sm70_decode_graph_semantics", lambda: True)
    monkeypatch.setattr(
        torch.ops.vllm,
        "qwen38_sm70_fp16_gemv",
        record_role,
    )
    x = torch.empty(1, 1536, device="meta")
    layer = SimpleNamespace(weight=x, prefix="layers.0.linear_attn.out_proj")
    gemv.Qwen38SM70FP16LinearMethod().apply(layer, x)
    assert seen == [layer.prefix]


@pytest.mark.parametrize("role", ["other.proj", "layers.0.linear_attn.out_proj"])
def test_unsupported_input_keeps_linear_fallback(role):
    x = torch.arange(6, dtype=torch.float32).view(2, 3)
    w = torch.arange(12, dtype=torch.float32).view(4, 3)
    assert torch.equal(gemv._qwen38_sm70_fp16_gemv(x, w, role), x @ w.T)


def test_role_is_preserved_through_fake_export():
    class Projection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.empty(2560, 1536, dtype=torch.float16, device="meta")
            )

        def forward(self, x):
            return torch.ops.vllm.qwen38_sm70_fp16_gemv(
                x, self.weight, "layers.0.linear_attn.out_proj"
            )

    program = torch.export.export(
        Projection(), (torch.empty(1, 1536, dtype=torch.float16, device="meta"),)
    )
    calls = [
        node
        for node in program.graph.nodes
        if node.target == torch.ops.vllm.qwen38_sm70_fp16_gemv.default
    ]
    assert len(calls) == 1
    assert calls[0].args[2] == "layers.0.linear_attn.out_proj"


@pytest.mark.parametrize("shape", [(64, 1537), (768, 5120), (1024, 2560)])
def test_other_tp_shards_and_widths_have_a_row_kernel_plan(shape):
    assert gemv._plan_for("model.layers.0.mlp.gate", shape) is not None
    assert gemv._plan_for("unrelated.layer", shape) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_generic_row_plan_masks_tails_and_replays_changing_inputs():
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    torch.manual_seed(511)
    x = torch.randn(1, 1537, dtype=torch.float16, device="cuda")
    weight = torch.randn(65, 1537, dtype=torch.float16, device="cuda")
    role = "model.layers.0.mlp.gate"
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            gemv._qwen38_sm70_fp16_gemv(x, weight, role)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = gemv._qwen38_sm70_fp16_gemv(x, weight, role)
    for _ in range(8):
        x.normal_()
        graph.replay()
        reference = (x.float() @ weight.float().T).half()
        torch.testing.assert_close(output, reference, atol=2e-3, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_role_policy_preserves_values_in_changing_input_graph():
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    torch.manual_seed(510)
    x = torch.randn(1, 1536, device="cuda", dtype=torch.float16)
    weight = torch.randn(2560, 1536, device="cuda", dtype=torch.float16)

    def project():
        return tuple(
            torch.ops.vllm.qwen38_sm70_fp16_gemv(x, weight, role)
            for role in (
                "",
                "layers.0.linear_attn.out_proj",
                "layers.3.self_attn.o_proj",
            )
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            project()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = project()
    for _ in range(16):
        x.normal_()
        graph.replay()
        eager = project()
        for actual in (*outputs, *eager):
            assert torch.equal(actual, eager[0])
