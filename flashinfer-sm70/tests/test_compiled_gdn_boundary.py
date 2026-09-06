# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A large prefill trace must not erase the small-decode runtime dispatch."""

import ast
import inspect
import textwrap
from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest
import torch

from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn


@pytest.mark.parametrize("legacy_shape_guard", [False, True])
def test_boundary_exports_the_whole_prefill_decode_range(legacy_shape_guard):
    # Export the actual first branch predicate, not a duplicate of its logic.
    # The old `1 < num_tokens <= 64` condition constrains this graph to rows
    # >=65 and fails export's requested 1..2048 contract after a prefill trace.
    source = ast.parse(
        textwrap.dedent(inspect.getsource(gdn.QwenGatedDeltaNetAttention.forward_cuda))
    )
    branch = next(node for node in source.body[0].body if isinstance(node, ast.If))
    probe = ast.parse(
        "def forward(self, x):\n"
        "    num_tokens = x.size(0)\n"
        "    if True:\n"
        "        return x + 1\n"
        "    return x - 1\n"
    )
    probe.body[0].body[1].test = branch.test
    if legacy_shape_guard:
        probe.body[0].body[1].test = ast.BoolOp(
            op=ast.And(),
            values=[branch.test, ast.parse("1 < num_tokens <= 64", mode="eval").body],
        )
    scope = {
        "_sm70_qwen_gdn_input_core_boundary_enabled": lambda: False,
        "use_sm70_decode_graph_semantics": lambda: True,
    }
    exec(compile(ast.fix_missing_locations(probe), __file__, "exec"), scope)
    module = type("BoundaryProbe", (torch.nn.Module,), {"forward": scope["forward"]})()
    module._sm70_fi_ready = True
    expected_error = (
        pytest.raises(torch._dynamo.exc.UserError, match="Constraints violated")
        if legacy_shape_guard
        else nullcontext()
    )
    with expected_error:
        exported = torch.export.export(
            module,
            (torch.ones(128, 2),),
            dynamic_shapes={"x": {0: torch.export.Dim("rows", min=1, max=2048)}},
        ).module()
    if legacy_shape_guard:
        return
    for rows in (1, 4, 16, 128, 2048):
        torch.testing.assert_close(
            exported(torch.ones(rows, 2)), torch.full((rows, 2), 2.0)
        )


@pytest.mark.parametrize("rows", [1, 4, 65])
def test_unsupported_runtime_keeps_existing_fp16_projection(monkeypatch, rows):
    layer = NS(
        _sm70_fi_ready=True,
        sm70_qwen38_fp16_fused_input=True,
        in_proj_qkvz=NS(weight=torch.empty(0)),
        in_proj_ba=NS(weight=torch.empty(0)),
    )
    ctx = NS(no_compile_layers={"layer": layer}, attn_metadata=None)
    monkeypatch.setattr(gdn, "get_forward_context", lambda: ctx)
    monkeypatch.setattr(gdn, "_resolve_layer_name", lambda name: name)
    monkeypatch.setattr(
        gdn, "_sm70_dump_gdn_projection_tensor", lambda _, __, tensor: tensor
    )
    monkeypatch.setattr(gdn, "_sm70_gdn_projection_dump_requested", lambda _: False)
    monkeypatch.setattr(gdn, "_sm70_gdn_qpn8_ba_split_eligible", lambda *_: False)
    monkeypatch.setattr(gdn, "use_sm70_decode_graph_semantics", lambda: True)
    qkv = torch.randn(rows, 5)
    z = torch.randn(rows, 6)
    b, a = torch.randn(rows, 3), torch.randn(rows, 3)
    calls = []

    def project(*args):
        calls.append("original_projection")
        return qkv, z, b, a

    def core(self, **kwargs):
        assert kwargs["mixed_qkv"] is qkv
        assert kwargs["b"] is b and kwargs["a"] is a
        calls.append("original_core")
        kwargs["core_attn_out"].fill_(2)

    monkeypatch.setattr(
        torch.ops.vllm, "qwen38_sm70_fp16_gdn_input", project, raising=False
    )
    monkeypatch.setattr(gdn, "_qwen_gdn_run_recurrent_core", core)
    z_out = torch.empty(rows, 3, 2)
    core_out = torch.empty_like(z_out)
    result = gdn.qwen_gdn_input_projection_core(
        torch.randn(rows, 7), z_out, core_out, None, None, "layer"
    )
    assert calls == ["original_projection", "original_core"]
    assert result[0] is z_out and result[1] is core_out
    torch.testing.assert_close(z_out.flatten(1), z)
    torch.testing.assert_close(core_out, torch.full_like(core_out, 2))
