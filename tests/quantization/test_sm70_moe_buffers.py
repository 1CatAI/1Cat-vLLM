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
from unittest.mock import MagicMock

import pytest
import torch

from vllm.model_executor.layers.fused_moe.sm70.base import Sm70MoEMethodBase
from vllm.model_executor.layers.quantization import fp8_sm70_moe as fp8
from vllm.model_executor.layers.quantization import fp8_sm70_weight_codec as codec_impl
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
            for text in (FIXTURE["buffers"] | FIXTURE.get("stages", {})).values()
        )
    )
    namespace: dict[str, Any] = {
        **vars(fp8),
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


class _InlineStages(ast.NodeTransformer):
    def visit_Name(self, node):
        if node.id == "method":
            return ast.Name(id="self", ctx=node.ctx)
        return node

    def visit_Attribute(self, node):
        value = ast.unparse(node)
        if value in ("self.native_ops", "codec.native_ops"):
            return ast.Name(id="sm70_ops", ctx=node.ctx)
        if ast.unparse(node.value) == "self._legacy":
            return ast.Name(id=node.attr, ctx=node.ctx)
        if isinstance(node.value, ast.Name) and node.value.id == "policy":
            return ast.Attribute(
                value=ast.Name(id="layer", ctx=ast.Load()),
                attr="sm70_fp8_moe_" + node.attr,
                ctx=node.ctx,
            )
        return self.generic_visit(node)

    def visit_Call(self, node):
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "codec"
        ):
            if node.func.attr == "message":
                assert isinstance(node.args[0], ast.Constant)
                assert isinstance(node.args[0].value, str)
                return ast.Constant(value="SM70 FP8 " + node.args[0].value)
            if node.func.attr == "log":
                node.func = ast.Name(id="_log_runtime_route_once", ctx=ast.Load())
                node.args[0] = ast.Constant(value="SM70 FP8 " + node.args[0].value)
            elif node.func.attr == "enabled":
                node.func = ast.Name(
                    id=codec_impl.CAPABILITIES[node.args[0].value], ctx=ast.Load()
                )
                node.args = []
            elif node.func.attr == "buffer":
                assert ast.unparse(node.args[0]) == "layer"
                return ast.Attribute(
                    value=node.args[0],
                    attr="_fp8_buf_" + node.args[1].value,
                    ctx=ast.Load(),
                )
            elif node.func.attr in ("gemm_w13", "gemm_w2"):
                table = (
                    codec_impl.W13_OPS
                    if node.func.attr == "gemm_w13"
                    else codec_impl.W2_OPS
                )
                assert ast.unparse(node.args[1]) == "layer"
                node.func = ast.Attribute(
                    value=ast.Name(id="sm70_ops", ctx=ast.Load()),
                    attr=table[node.args[0].value],
                    ctx=ast.Load(),
                )
                node.args = node.args[2:]
        return self.generic_visit(node)


def test_all_method_bodies_match_parent_after_expanding_buffer_wrappers():
    original = ast.parse(inspect.getsource(fp8.Fp8SM70MoEMethod)).body[0]
    assert isinstance(original, ast.ClassDef)
    base = ast.parse(inspect.getsource(Sm70MoEMethodBase)).body[0]
    assert isinstance(base, ast.ClassDef)
    common = {
        node.name: node for node in base.body if isinstance(node, ast.FunctionDef)
    }
    prepare = ast.parse(
        inspect.getsource(codec_impl.Fp8MoEWeightCodec.prepare_weights).strip()
    ).body[0]
    assert isinstance(prepare, ast.FunctionDef)
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
        elif node.name in FIXTURE.get("stages", {}):
            expanded = copy.deepcopy(
                common["_apply_moe"] if node.name == "apply" else prepare
            )
            expanded.name = node.name
            expanded.args = node.args
            if node.name == "apply":
                expanded.body = expanded.body[2:]  # codec/policy bindings
            node = _InlineStages().visit(expanded)
        hashes[node.name] = hashlib.sha256(ast.dump(node).encode()).hexdigest()
    assert hashes == FIXTURE["method_hashes"]


@pytest.mark.parametrize(
    "tokens,batched", [(0, True), (1, False), (1, True), (2, False), (2, True)]
)
@pytest.mark.parametrize("variant", ["dense", "indexed", "compact", "per_expert"])
def test_fp8_stage_calls_and_cpu_stub_outputs_match_parent(
    monkeypatch, tokens, batched, variant
):
    calls = []
    logs = []

    def signature(args):
        return [(_descriptor(a) if isinstance(a, torch.Tensor) else a) for a in args]

    def native(name):
        def run(*args):
            calls.append((name, signature(args)))
            args[0].fill_(0.25)

        return run

    for name in (
        set(codec_impl.W13_OPS.values())
        | set(codec_impl.W2_OPS.values())
        | {"awq_moe_single_token_weighted_reduce_out"}
    ):
        monkeypatch.setattr(fp8.sm70_ops, name, native(name))
    for key, name in codec_impl.CAPABILITIES.items():
        enabled = (
            ("indexed" in key and variant == "indexed")
            or ("compact_w13" in key and variant == "compact")
            or (key == "single_token_weighted_reduce" and variant == "compact")
        )
        monkeypatch.setattr(fp8, name, lambda enabled=enabled: enabled)
    monkeypatch.setattr(fp8, "_log_runtime_route_once", lambda *args: logs.append(args))

    def permute(
        x,
        ids,
        indices,
        mapping,
        global_experts,
        local_experts,
        top_k,
        sorted_x,
        offsets,
        inverse,
        permutation,
        *scratch,
    ):
        calls.append(("permute", (bool(scratch),)))
        sorted_x.copy_(x.repeat_interleave(top_k, dim=0))
        offsets.zero_()
        offsets[-1] = ids.numel()
        inverse.copy_(torch.arange(ids.numel(), dtype=torch.int32).view_as(ids))

    def activate(out, gate_up):
        calls.append(("silu", signature((out, gate_up))))
        size = out.shape[1]
        out.copy_(torch.nn.functional.silu(gate_up[:, :size]) * gate_up[:, size:])

    def unpermute(sorted_out, weights, inverse, offsets, top_k, out):
        calls.append(
            (
                "unpermute",
                signature((sorted_out, weights, inverse, offsets, top_k, out)),
            )
        )
        out.copy_(
            (sorted_out.view(out.shape[0], top_k, -1) * weights.unsqueeze(-1)).sum(1)
        )

    monkeypatch.setattr(torch.ops._moe_C, "moe_permute", permute, raising=False)
    monkeypatch.setattr(
        torch.ops._moe_C, "moe_permute_with_scratch", permute, raising=False
    )
    monkeypatch.setattr(torch.ops._C, "silu_and_mul", activate, raising=False)
    monkeypatch.setattr(torch.ops._moe_C, "moe_unpermute", unpermute, raising=False)
    monkeypatch.setattr(
        torch.ops._moe_C,
        "moe_permute_sort_workspace_size",
        lambda *args: 8,
        raising=False,
    )
    outputs, traces, messages = [], [], []
    for method in (_legacy(), object.__new__(fp8.Fp8SM70MoEMethod)):
        layer = _case(method, top_k=2, scratch=variant == "per_expert")
        layer.apply_router_weight_on_input = False
        layer.expert_map = None
        layer.local_num_experts = layer.global_num_experts
        layer.sm70_fp8_moe_batched_gemm = batched
        layer.sm70_fp8_moe_permute_with_scratch = variant == "per_expert"
        layer.sm70_fp8_moe_batched_w13_per_expert_dispatch = variant == "per_expert"
        layer.sm70_fp8_moe_batched_w2_per_expert_dispatch = variant == "per_expert"
        layer.sm70_w13_k_dim, layer.sm70_w2_k_dim, layer.sm70_w2_n_dim = 64, 32, 64
        for name in (
            "w13_strided_ptrs_w",
            "w13_strided_ptrs_s",
            "w2_strided_ptrs_w",
            "w2_strided_ptrs_s",
        ):
            setattr(layer, name, torch.empty(4, 8, dtype=torch.uint8))
        method.group_size = 128
        method.compact_compare_reference = False
        calls.clear()
        logs.clear()
        output = method.apply(
            layer,
            torch.ones(tokens, 64, dtype=torch.float16),
            torch.full((tokens, 2), 0.5),
            torch.zeros(tokens, 2, dtype=torch.int32),
            None,
            None,
        )
        outputs.append(output.clone())
        traces.append(list(calls))
        messages.append(list(logs))
    assert torch.equal(*outputs)
    assert traces[0] == traces[1]
    assert messages[0] == messages[1]


@pytest.mark.parametrize("batched", [False, True])
def test_fp8_weight_preparation_layout_and_lifecycle_match_parent(monkeypatch, batched):
    def prepare(weight, scales, group_size):
        assert group_size == 128
        packed = weight.t().contiguous()
        return packed, scales.clone(), torch.tensor([packed.shape[0], packed.shape[1]])

    def pointers(weight, scales, k_ld, q_ld, experts):
        return torch.zeros(experts, 8, dtype=torch.uint8), torch.zeros(
            experts, 8, dtype=torch.uint8
        )

    monkeypatch.setattr(fp8.sm70_ops, "fp8_sm70_prepare", prepare)
    monkeypatch.setattr(fp8.sm70_ops, "awq_moe_build_strided_ptrs", pointers)
    layers = []
    for method in (_legacy(), object.__new__(fp8.Fp8SM70MoEMethod)):
        method.quant_config = SimpleNamespace(activation_scheme="dynamic")
        method.block_quant = True
        method.group_size = 128
        method.use_batched_gemm = batched
        method.use_batched_w13_per_expert_dispatch = True
        method.use_batched_w2_per_expert_dispatch = False
        method.use_permute_with_scratch = False
        method.moe = SimpleNamespace(experts_per_token=2)
        layer: Any = torch.nn.Module()
        layer.global_num_experts = 2
        for name, shape in (
            ("w13_weight", (2, 128, 128)),
            ("w2_weight", (2, 128, 64)),
            ("w13_weight_scale_inv", (2, 1, 1)),
            ("w2_weight_scale_inv", (2, 1, 1)),
        ):
            setattr(
                layer, name, torch.nn.Parameter(torch.ones(shape), requires_grad=False)
            )
        method.process_weights_after_loading(layer)
        assert not hasattr(layer, "w13_weight")
        assert not hasattr(layer, "w2_weight_scale_inv")
        layers.append(layer)
    assert layers[0]._parameters.keys() == layers[1]._parameters.keys()
    for name in layers[0]._parameters:
        old, new = getattr(layers[0], name), getattr(layers[1], name)
        assert _descriptor(old) == _descriptor(new), name
        assert torch.equal(old, new), name
    for name, value in vars(layers[0]).items():
        if name.startswith("sm70_"):
            assert value == getattr(layers[1], name), name


def test_legacy_compact_callback_retains_early_return(monkeypatch):
    monkeypatch.setattr(fp8, "_legacy_single_token_compact_enabled", lambda: True)
    current = object.__new__(fp8.Fp8SM70MoEMethod)
    layer = _case(current, top_k=2, scratch=False)
    layer.apply_router_weight_on_input = False
    layer.sm70_fp8_moe_batched_gemm = True
    expected = torch.full((1, 64), 7, dtype=torch.float16)
    callback = MagicMock(return_value=expected)
    current._apply_legacy_single_token_compact = callback
    result = current.apply(
        layer,
        torch.ones(1, 64, dtype=torch.float16),
        torch.ones(1, 2),
        torch.zeros(1, 2, dtype=torch.int64),
        None,
        None,
    )
    assert result is expected
    callback.assert_called_once()
    args = callback.call_args.args
    assert args[0] is layer
    assert args[3].dtype == torch.int32
    assert args[5] == 2
    assert (
        args[6].untyped_storage()._cdata
        == layer._fp8_buf_output.untyped_storage()._cdata
    )


def test_codec_preserves_native_module_rebinding(monkeypatch):
    operator = MagicMock()
    monkeypatch.setattr(
        fp8, "sm70_ops", SimpleNamespace(fp8_moe_gemm_sm70_out=operator)
    )
    codec = fp8.Fp8SM70MoEMethod.weight_codec
    layer: Any = SimpleNamespace()
    payload = torch.ones(1)
    codec.gemm_w13("batched", layer, payload, 3, False)
    codec.gemm_w2("batched", layer, payload, 4, False)
    assert operator.call_args_list[0].args == (payload, 3, False)
    assert operator.call_args_list[1].args == (payload, 4, False)
    with pytest.raises(KeyError):
        codec.gemm_w13("unimplemented", layer, payload)
