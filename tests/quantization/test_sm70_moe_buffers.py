# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Differential buffer lifetime/layout checks against the frozen FP8 parent."""

import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.model_executor.layers.fused_moe.sm70.base import Sm70MoEMethodBase
from vllm.model_executor.layers.quantization import fp8_sm70_moe as fp8
from vllm.model_executor.layers.quantization.utils import sm70_layer_workspaces as ws

pytestmark = pytest.mark.cpu_test
FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/sm70_fp8_buffers_legacy.json").read_text()
)


def _legacy():
    tree = ast.parse(
        "class Legacy:\n"
        + "\n".join(
            "\n".join("    " + line for line in text.splitlines())
            for text in FIXTURE["buffers"].values()
        )
    )
    namespace: dict[str, Any] = {
        "torch": torch,
        "RoutedExperts": Any,
        "_DEFAULT_PERSISTENT_MAX_TOKENS": fp8._DEFAULT_PERSISTENT_MAX_TOKENS,
    }
    exec(compile(tree, "<frozen_fp8_buffers>", "exec"), namespace)
    return namespace["Legacy"]()


def _case(method, top_k, scratch):
    layer: Any = torch.nn.Module()
    layer.w13_tm_weight = torch.empty(1)
    layer.sm70_hidden_logical_size = 64
    layer.sm70_num_experts = 4
    layer.sm70_intermediate_size = 32
    layer.sm70_w13_n_dim = 64
    layer.sm70_ptr_row_bytes = 8
    layer.global_num_experts = 4
    method.moe = SimpleNamespace(experts_per_token=top_k)
    method.use_permute_with_scratch = scratch
    method._allocate_buffers(layer)
    return layer


def _descriptor(tensor):
    return tensor.shape, tensor.dtype, tensor.stride(), tensor.device


@pytest.mark.parametrize("scratch", [False, True])
@pytest.mark.parametrize("top_k", [1, 2, 8])
@pytest.mark.parametrize("tokens", [0, 1, 32, 33])
def test_fp8_allocations_and_buffer_aliases_match_parent(
    monkeypatch, scratch, top_k, tokens
):
    monkeypatch.setattr(
        torch.ops._moe_C,
        "moe_permute_sort_workspace_size",
        lambda slots, experts: slots * 4 + experts,
        raising=False,
    )
    parent = _legacy()
    current = object.__new__(fp8.Fp8SM70MoEMethod)
    old_layer = _case(parent, top_k, scratch)
    new_layer = _case(current, top_k, scratch)
    names = [name for name in vars(old_layer) if name.startswith("_fp8_buf_")]
    assert set(names) == {
        name for name in vars(new_layer) if name.startswith("_fp8_buf_")
    }
    for name in names:
        old, new = getattr(old_layer, name), getattr(new_layer, name)
        if isinstance(old, torch.Tensor):
            assert _descriptor(old) == _descriptor(new), name
        else:
            assert old == new, name
    old = parent._get_buffers(old_layer, tokens * top_k, tokens)
    new = current._get_buffers(new_layer, tokens * top_k, tokens)
    assert old.keys() == new.keys()
    for name in old:
        assert _descriptor(old[name]) == _descriptor(new[name]), name
        for candidate in names:
            old_source, new_source = (
                getattr(old_layer, candidate),
                getattr(new_layer, candidate),
            )
            if not isinstance(old_source, torch.Tensor):
                continue
            # Storage identity works for empty buffers too; data_ptr()==0 does not.
            old_alias = (
                old[name].untyped_storage()._cdata
                == old_source.untyped_storage()._cdata
            )
            new_alias = (
                new[name].untyped_storage()._cdata
                == new_source.untyped_storage()._cdata
            )
            assert old_alias == new_alias, (name, candidate)
    assert torch.equal(old["token_expert_indices"], new["token_expert_indices"])
    assert torch.equal(old["active_expert_offsets"], new["active_expert_offsets"])


def test_layer_view_keeps_legacy_rebinding_and_registry_identity():
    layer = torch.nn.Module()
    before = dict(ws._layer_workspaces)
    view = ws.LayerWorkspaceView(layer, "_fp8_buf_")
    view.output = torch.ones(1)
    assert layer._fp8_buf_output is view.output
    layer._fp8_buf_output = torch.zeros(2)
    assert view.output is layer._fp8_buf_output
    assert ws._layer_workspaces == before


def test_legacy_capacity_constant_is_read_when_allocating(monkeypatch):
    monkeypatch.setattr(fp8, "_DEFAULT_PERSISTENT_MAX_TOKENS", 5)
    current = object.__new__(fp8.Fp8SM70MoEMethod)
    layer = _case(current, top_k=2, scratch=False)
    assert layer._fp8_buf_max_tokens == 5
    assert layer._fp8_buf_max_slots == 10


def test_persistent_buffers_trace_and_observe_rebound_layer_storage():
    current = object.__new__(fp8.Fp8SM70MoEMethod)
    layer = _case(current, top_k=2, scratch=False)
    layer._fp8_buf_output.fill_(2)

    def run(x):
        buffers = current._get_buffers(layer, x.shape[0] * 2, x.shape[0])
        return buffers["output"] + x

    compiled = torch.compile(run, backend="eager", fullgraph=True)
    assert torch.equal(compiled(torch.ones(1, 64)), torch.full((1, 64), 3.0))
    layer._fp8_buf_output = torch.full_like(layer._fp8_buf_output, 5)
    assert torch.equal(compiled(torch.ones(1, 64)), torch.full((1, 64), 6.0))


class _InlineBuffers(ast.NodeTransformer):
    def __init__(self, values):
        self.values = values

    def visit_Name(self, node):
        if node.id in self.values:
            return copy.deepcopy(self.values[node.id])
        return node

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id == "buffers":
            return ast.Attribute(
                value=ast.Name(id="layer", ctx=ast.Load()),
                attr="_fp8_buf_" + node.attr,
                ctx=node.ctx,
            )
        return self.generic_visit(node)


def test_all_method_bodies_match_parent_after_expanding_buffer_wrappers():
    original = ast.parse(inspect.getsource(fp8.Fp8SM70MoEMethod)).body[0]
    assert isinstance(original, ast.ClassDef)
    base = ast.parse(inspect.getsource(Sm70MoEMethodBase)).body[0]
    assert isinstance(base, ast.ClassDef)
    common = {
        node.name: node for node in base.body if isinstance(node, ast.FunctionDef)
    }
    hashes = {}
    for node in original.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name in FIXTURE["buffers"]:
            statement = node.body[0]
            assert isinstance(statement, (ast.Expr, ast.Return))
            call = statement.value
            assert isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            values = {k.arg: k.value for k in call.keywords}
            expanded = copy.deepcopy(common[call.func.attr])
            expanded.name = node.name
            expanded.args = node.args
            expanded.body = expanded.body[1:]  # remove the non-owning view
            node = _InlineBuffers(values).visit(expanded)
        hashes[node.name] = hashlib.sha256(ast.dump(node).encode()).hexdigest()
    assert hashes == FIXTURE["method_hashes"]
