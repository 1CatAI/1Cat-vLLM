# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Method calculation/descriptor invariants after implementation extraction."""

import ast
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.v1.attention.backends import flash_attn_v100 as legacy
from vllm.v1.attention.backends.flash_v100 import impl, state
from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

pytestmark = pytest.mark.cpu_test


class _Normalize(ast.NodeTransformer):
    def visit_Global(self, node):
        return None

    def visit_FunctionDef(self, node):
        node = self.generic_visit(node)
        if node.args.args and node.args.args[0].arg == "self":
            node.args.args[0].annotation = None
        node.decorator_list = []
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body[0].value.value = inspect.cleandoc(node.body[0].value.value)
        return node

    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == "_state":
            return ast.Name(id=node.attr, ctx=node.ctx)
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "super" and node.args:
            assert [ast.unparse(a) for a in node.args] == ["_impl._super_owner", "self"]
            node.args = []
        return node


def test_all_method_bodies_and_static_descriptors_match_parent():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/flash_v100_impl_methods.json").read_text()
    )["methods"]
    actual = {}
    for path in Path(impl.__file__).parent.glob("*.py"):
        for node in ast.parse(path.read_text()).body:
            candidates = (
                node.body
                if isinstance(node, ast.ClassDef) and node.name == "FlashAttnV100Impl"
                else [node]
            )
            for fn in candidates:
                if not isinstance(fn, ast.FunctionDef) or fn.name not in fixture:
                    continue
                assert fn.name not in actual
                actual[fn.name] = hashlib.sha256(
                    ast.dump(_Normalize().visit(fn)).encode()
                ).hexdigest()
    assert actual == {k: v["sha256"] for k, v in fixture.items()}
    for name, descriptor in fixture.items():
        member = vars(impl.FlashAttnV100Impl)[name]
        assert isinstance(member, staticmethod) == descriptor["static"]


def test_legacy_state_rebinding_reaches_moved_decode_method(monkeypatch):
    instance = object.__new__(impl.FlashAttnV100Impl)
    instance._decode_cache_k = torch.ones(1)
    instance._decode_cache_v = torch.ones(1)
    instance._decode_cache_len = 3
    instance._decode_cache_capacity = 8
    instance._reset_decode_cache()
    assert instance._decode_cache_len == 0
    monkeypatch.setattr(legacy, "_logged_decode_dense_cache", True)
    assert state._logged_decode_dense_cache
    assert "_logged_decode_dense_cache" not in vars(impl)
    logger = MagicMock()
    monkeypatch.setattr(legacy, "logger", logger)
    monkeypatch.setattr(legacy, "_logged_decode_dense_reference", False)
    query = torch.empty((0, 6, 256), dtype=torch.float16)
    metadata = SimpleNamespace(num_actual_tokens=0)
    for _ in range(2):
        assert (
            instance._flash_v100_decode_dense_reference(
                None, query, query, metadata, query
            )
            is query
        )
    assert state._logged_decode_dense_reference
    assert legacy._logged_decode_dense_reference
    logger.warning.assert_called_once()


def test_extracted_compare_super_uses_original_class_cell(monkeypatch):
    original = impl.FlashAttnV100Impl
    instance = object.__new__(original)
    instance._reserve_triton_compare_call = lambda: 0
    instance._maybe_write_triton_tensor_dump = lambda *args: {}
    instance._write_triton_compare_report = MagicMock()
    metadata = SimpleNamespace(num_actual_tokens=1, max_query_len=1, max_seq_len=1)
    query = torch.zeros((1, 6, 256), dtype=torch.float16)
    output = torch.zeros_like(query)
    layer = SimpleNamespace()
    calls = []

    def reference(self, layer, q, k, v, cache, metadata, out, *args):
        assert self is instance
        calls.append(self)
        out.fill_(1)
        return out

    monkeypatch.setattr(TritonAttentionImpl, "forward", reference)
    # A module export patch must not change the old function's __class__ cell.
    monkeypatch.setattr(legacy, "FlashAttnV100Impl", object)
    instance._maybe_compare_triton_output(
        layer, query, query, query, query, metadata, output, None, None, "decode"
    )
    assert calls == [instance]
    report = instance._write_triton_compare_report.call_args.args
    assert torch.equal(report[1], torch.ones_like(output))
