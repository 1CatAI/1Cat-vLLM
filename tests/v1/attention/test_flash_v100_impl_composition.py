# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Method calculation/descriptor invariants after implementation extraction."""

import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.v1.attention.backends import flash_attn_v100 as legacy
from vllm.v1.attention.backends.flash_v100 import impl, state, workspace
from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

pytestmark = pytest.mark.cpu_test


class _InlineFeatureHooks(ast.NodeTransformer):
    """Expand mechanical feature extractions before comparing original bodies."""

    def __init__(self):
        source = Path(impl.__file__).parent / "spec/attention.py"
        self.hooks = {
            node.name: node
            for node in ast.parse(source.read_text()).body
            if isinstance(node, ast.FunctionDef)
        }

    def _hook(self, call):
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "ATTENTION_HOOKS"
        ):
            hook = self.hooks[call.func.attr]
            names = {
                "feature_fallback": "is_dflash_draft_attn",
                "capture_prefix": "is_dflash_non_causal",
            }
            arguments = [names.get(ast.unparse(a), ast.unparse(a)) for a in call.args]
            assert not call.keywords
            assert arguments == [a.arg for a in hook.args.args]
            return hook
        return None

    def visit_Expr(self, node):
        hook = self._hook(node.value)
        if hook is not None:
            # Statement hooks use the original local names; predicate hooks
            # return an expression and are handled separately below.
            assert not any(isinstance(n, ast.Return) for n in hook.body)
            return copy.deepcopy(hook.body)
        return self.generic_visit(node)

    def visit_Call(self, node):
        hook = self._hook(node)
        if hook is not None:
            assert len(hook.body) == 1 and isinstance(hook.body[0], ast.Return)
            return copy.deepcopy(hook.body[0].value)
        return self.generic_visit(node)

    def visit_Name(self, node):
        names = {
            "feature_fallback": "is_dflash_draft_attn",
            "capture_prefix": "is_dflash_non_causal",
        }
        if node.id in names:
            return ast.Name(id=names[node.id], ctx=node.ctx)
        return node


_CACHE_METHODS = {
    "invalidate": "_reset_decode_cache",
    "ensure_capacity": "_ensure_decode_cache_capacity",
    "get_kv_single_seq": "_get_decode_kv_single_seq",
}
_CACHE_FIELDS = {
    "key": "_decode_cache_k",
    "value": "_decode_cache_v",
    "length": "_decode_cache_len",
    "capacity": "_decode_cache_capacity",
}


class _Normalize(ast.NodeTransformer):
    in_cache = False

    def visit_ImportFrom(self, node):
        if node.module == "vllm.v1.attention.ops.sm70_grouped_scalar":
            node.module = "vllm.v1.attention.ops.sm70_e4m3_scalar"
        return node

    def visit_Global(self, node):
        return None

    def visit_FunctionDef(self, node):
        self.in_cache = node.name in _CACHE_METHODS
        if node.name == "get_kv_single_seq":
            assert [a.arg for a in node.args.kwonlyargs] == ["extract"]
            node.args.kwonlyargs = []
            node.args.kw_defaults = []
        node.name = _CACHE_METHODS.get(node.name, node.name)
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
        if ast.unparse(node) == "self.config.policy":
            return ast.Name(id="self", ctx=ast.Load())
        owner = ast.unparse(node.value)
        if owner in ("self.config.policy", "self.config"):
            node.value = ast.Name(id="self", ctx=ast.Load())
        if owner == "self.ops":
            node.value = ast.Name(id="self", ctx=ast.Load())
            node.attr = {
                "dense": "flash_attn_func",
                "paged": "flash_attn_decode_paged",
                "xqa": "flash_attn_decode_paged_xqa",
                "wmma": "flash_attn_decode_paged_wmma",
                "prefill": "flash_attn_prefill_paged",
                "prefill_bhmd": "flash_attn_prefill_paged_bhmd",
                "paged_keywords": "_flash_decode_paged_kwargs",
                "reserve_bhmd_compare": "_reserve_bhmd_compare_call",
                "write_bhmd_compare": "_write_bhmd_compare_report",
                "compare_bhmd": "_maybe_compare_bhmd_out",
            }.get(node.attr, node.attr)
        node = self.generic_visit(node)
        if ast.unparse(node) == "self.scalar_tail":
            return ast.parse(
                'getattr(self, "_sm70_scalar_tail_attention", None)', mode="eval"
            ).body
        if self.in_cache and ast.unparse(node.value) == "self":
            node.attr = {**_CACHE_FIELDS, **_CACHE_METHODS}.get(node.attr, node.attr)
        if ast.unparse(node.value) == "self.workspace.decode_cache":
            node.value = ast.Name(id="self", ctx=ast.Load())
            node.attr = _CACHE_METHODS.get(node.attr, node.attr)
        if ast.unparse(node.value) == "_workspace":
            names = {
                "MixedDecodeRowsPlan": "_MixedDecodeRowsPlan",
                "mixed_decode_rows_plan": "_mixed_decode_rows_plan",
                "MIXED_ROWS_GROUP": "_MIXED_ROWS_GROUP",
            }
            if node.attr in names:
                node.value = ast.Name(id="_metadata", ctx=ast.Load())
                node.attr = names[node.attr]
        if isinstance(node.value, ast.Name) and node.value.id == "_state":
            return ast.Name(id=node.attr, ctx=node.ctx)
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if self.in_cache and ast.unparse(node.func) == "extract":
            node.func = ast.parse(
                "_kv_layout._extract_contiguous_kv_from_paged_cache", mode="eval"
            ).body
        if ast.unparse(node.func) == "self._get_decode_kv_single_seq":
            assert len(node.keywords) == 1 and node.keywords[0].arg == "extract"
            assert (
                ast.unparse(node.keywords[0].value)
                == "_kv_layout._extract_contiguous_kv_from_paged_cache"
            )
            node.keywords = []
        if (
            ast.unparse(node.func) == "getattr"
            and node.args
            and ast.unparse(node.args[0]) == "self.config.policy"
        ):
            node.args[0] = ast.Name(id="self", ctx=ast.Load())
        if ast.unparse(node.func) == "_config.registered":
            assert len(node.args) == 1 and isinstance(node.args[0], ast.Constant)
            assert isinstance(node.args[0].value, str)
            return ast.Attribute(
                value=ast.Name(id="envs", ctx=ast.Load()),
                attr=node.args[0].value,
                ctx=ast.Load(),
            )
        if ast.unparse(node.func) == "_config.raw":
            node.func = ast.Attribute(
                value=ast.Name(id="os", ctx=ast.Load()), attr="getenv", ctx=ast.Load()
            )
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
                if isinstance(node, ast.ClassDef)
                and node.name in ("FlashAttnV100Impl", "DecodeCache", "DecodeExecutor")
                else [node]
            )
            for fn in candidates:
                if not isinstance(fn, ast.FunctionDef):
                    continue
                if path.name == "decode.py" and fn.name in (
                    "__init__",
                    "_flash_v100_window_size",
                    "_xqa_kv_codec",
                ):
                    continue
                if any(
                    isinstance(n, ast.Call)
                    and ast.unparse(n.func) == "self._new_decode_executor"
                    for n in ast.walk(fn)
                ):
                    # Only a direct typed delegate may replace the original body.
                    assert len(fn.body) == 1 and isinstance(fn.body[0], ast.Return)
                    assert isinstance(fn.body[0].value, ast.Call)
                    assert (
                        ast.unparse(fn.body[0].value.func)
                        == "self._new_decode_executor()." + fn.name
                    )
                    continue
                name = _CACHE_METHODS.get(fn.name, fn.name)
                if name not in fixture:
                    continue
                assert name not in actual
                actual[name] = hashlib.sha256(
                    ast.dump(
                        _Normalize().visit(_InlineFeatureHooks().visit(fn))
                    ).encode()
                ).hexdigest()
    # A4b deliberately changes policy capture and the two hint consumers.
    # Preserve every other calculation body; route planning has its own oracle.
    changed_policy = {
        "__init__",
        "_flash_v100_decode",
        "_run_prefill_prefix_decode_rows",
    }
    assert actual.keys() == fixture.keys()
    assert {k: v for k, v in actual.items() if k not in changed_policy} == {
        k: v["sha256"] for k, v in fixture.items() if k not in changed_policy
    }
    for name, descriptor in fixture.items():
        cache_name = {v: k for k, v in _CACHE_METHODS.items()}.get(name)
        owner = workspace.DecodeCache if cache_name else impl.FlashAttnV100Impl
        member = inspect.getattr_static(owner, cache_name or name)
        assert isinstance(member, staticmethod) == descriptor["static"]


def test_legacy_state_rebinding_reaches_moved_decode_method(monkeypatch):
    instance = object.__new__(impl.FlashAttnV100Impl)
    instance.workspace = workspace.V100Workspace(
        workspace.DecodeCache(torch.ones(1), torch.ones(1), 3, 8)
    )
    instance.workspace.decode_cache.invalidate()
    assert instance.workspace.decode_cache.length == 0
    monkeypatch.setattr(legacy, "_logged_decode_dense_cache", True)
    assert state._logged_decode_dense_cache
    assert "_logged_decode_dense_cache" not in vars(impl)
    logger = MagicMock()
    monkeypatch.setattr(legacy, "logger", logger)
    monkeypatch.setattr(legacy, "_logged_decode_dense_reference", False)
    query = torch.empty((0, 6, 256), dtype=torch.float16)
    metadata = SimpleNamespace(num_actual_tokens=0)
    instance.kv_cache_dtype = "auto"
    assert (
        instance._flash_v100_decode_dense_cache(
            None, query, query, query, query, metadata, query
        )
        is query
    )
    logger.warning.assert_not_called()
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
    assert legacy.FlashAttnV100Backend.get_impl_cls() is object
    instance._maybe_compare_triton_output(
        layer, query, query, query, query, metadata, output, None, None, "decode"
    )
    assert calls == [instance]
    report = instance._write_triton_compare_report.call_args.args
    assert torch.equal(report[1], torch.ones_like(output))
