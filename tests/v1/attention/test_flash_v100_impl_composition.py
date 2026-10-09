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

from tests.v1.attention.flash_v100_sequence_oracle import (
    batch_calculations,
    prefill_debug_calculations,
    sequence_calculations,
)
from vllm.v1.attention.backends import flash_attn_v100 as legacy
from vllm.v1.attention.backends.flash_v100 import decode, impl, state, workspace
from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

pytestmark = pytest.mark.cpu_test


class _InlineFeatureHooks(ast.NodeTransformer):
    """Expand mechanical feature extractions before comparing original bodies."""

    def __init__(self):
        source = Path(impl.__file__).parent / "spec/attention.py"
        self.hooks = {
            node.name: node
            for path in (source, source.with_name("attention_policy.py"))
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.FunctionDef)
        }

    def _hook(self, call):
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and (
                ast.unparse(call.func.value) == "_feature"
                or (
                    ast.unparse(call.func.value) == "self.ops"
                    and call.func.attr
                    in (
                        "capture_prefix_kind",
                        "record_capture_prefix",
                        "record_capture_layout",
                    )
                )
            )
        ):
            hook = self.hooks[call.func.attr]
            names = {
                "feature_fallback": "is_dflash_draft_attn",
                "capture_prefix": "is_dflash_non_causal",
                "self._policy()": "policy",
                "self._contract_validator()": "validator",
            }
            arguments = [names.get(ast.unparse(a), ast.unparse(a)) for a in call.args]
            assert not call.keywords
            assert arguments == [a.arg for a in hook.args.args]

            class Inputs(ast.NodeTransformer):
                def visit_Name(self, node):
                    if node.id == "policy":
                        return ast.Name(id="self", ctx=node.ctx)
                    if node.id == "validator":
                        return ast.parse(
                            "self._validate_dflash_attention_contract", mode="eval"
                        ).body
                    return node

            return Inputs().visit(copy.deepcopy(hook))
        return None

    def _prefill_branches(self, helper):
        assert len(helper.body) == 1 and isinstance(helper.body[0], ast.Return)
        dispatch = helper.body[0].value
        assert isinstance(dispatch, ast.Call)
        assert ast.unparse(dispatch.func) == "self._new_prefill_executor().forward"
        assert not dispatch.keywords
        arguments = [a.arg for a in helper.args.args]
        assert [ast.unparse(a) for a in dispatch.args] == arguments[1:]
        tree = ast.parse((Path(impl.__file__).parent / "prefill.py").read_text())
        forward = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "forward"
        )
        assert [a.arg for a in forward.args.args] == arguments
        executor = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "PrefillExecutor"
        )
        assert any(
            isinstance(n, ast.Assign) and ast.unparse(n) == "forward = forward"
            for n in executor.body
        )
        block = ast.Module(body=copy.deepcopy(forward.body), type_ignores=[])
        return self.visit(block).body

    def _decode_branches(self, helper):
        assert len(helper.body) == 1 and isinstance(helper.body[0], ast.Return)
        dispatch = helper.body[0].value
        assert isinstance(dispatch, ast.Call)
        assert ast.unparse(dispatch.func) == "self._new_decode_executor().forward"
        assert len(dispatch.args) == 1 and not dispatch.keywords
        request = dispatch.args[0]
        assert isinstance(request, ast.Call)
        assert ast.unparse(request.func) == "_decode.DecodeRequest"
        assert [ast.unparse(a) for a in request.args] == [
            a.arg for a in helper.args.args[1:]
        ]
        tree = ast.parse(Path(decode.__file__).read_text())
        classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
        executor = classes["DecodeExecutor"]
        forward = next(
            n
            for n in executor.body
            if isinstance(n, ast.FunctionDef) and n.name == "forward"
        )
        assert len(forward.body) == 1 and isinstance(forward.body[0], ast.Return)
        assert forward.body[0].value is not None
        assert ast.unparse(forward.body[0].value) == (
            "_plan.execute(request, (candidate(self) "
            "for candidate in DECODE_CANDIDATES))"
        )
        order = next(
            n.value
            for n in tree.body
            if isinstance(n, ast.Assign)
            and ast.unparse(n.targets[0]) == "DECODE_CANDIDATES"
        )
        assert isinstance(order, ast.Tuple)
        result = []
        for index, name in enumerate(order.elts):
            methods = {
                n.name: n
                for n in classes[ast.unparse(name)].body
                if isinstance(n, ast.FunctionDef)
            }
            admit, run = methods["admit"], methods["run"]
            assert len(admit.body) == 1 and isinstance(admit.body[0], ast.Return)
            condition = admit.body[0].value
            assert condition is not None
            if index == len(order.elts) - 1:
                assert ast.unparse(condition) == "True"
                result.extend(copy.deepcopy(run.body))
            else:
                result.append(
                    ast.If(
                        test=copy.deepcopy(condition),
                        body=copy.deepcopy(run.body),
                        orelse=[],
                    )
                )
        return result

    def visit_Assign(self, node):
        if not isinstance(node.value, ast.Call):
            return self.generic_visit(node)
        callee = ast.unparse(node.value.func)
        if callee not in ("execute_prefill_sequence", "execute_prefill_batch"):
            return self.generic_visit(node)
        sequence = callee == "execute_prefill_sequence"
        assert ast.unparse(node.targets[0]) == (
            "(out_seq, out_is_destination, skip_debug)"
            if sequence
            else "(batch_complete, batch_output, decode_rows)"
        )
        assert not node.value.keywords
        helper = next(
            n
            for n in ast.parse(
                (Path(impl.__file__).parent / "prefill.py").read_text()
            ).body
            if isinstance(n, ast.FunctionDef) and n.name == callee
        )
        arguments = [a.arg for a in helper.args.args]
        assert [ast.unparse(a) for a in node.value.args] == [
            "dflash_dump" if a == "dump_enabled" else a for a in arguments
        ]
        constructor = (
            "_sequence.PrefillRequest" if sequence else "_sequence.PrefillBatchRequest"
        )
        request = next(
            n
            for n in ast.walk(helper)
            if isinstance(n, ast.Call) and ast.unparse(n.func) == constructor
        )
        assert not request.keywords
        assert [ast.unparse(a) for a in request.args] == arguments[1:]
        return sequence_calculations() if sequence else batch_calculations()

    def visit_If(self, node):
        if ast.unparse(node.test) == "skip_debug":
            assert len(node.body) == 1 and isinstance(node.body[0], ast.Continue)
            assert not node.orelse
            return []
        if ast.unparse(node.test) == "batch_complete":
            assert [ast.unparse(n) for n in node.body] == ["return batch_output"]
            assert not node.orelse
            return []
        return self.generic_visit(node)

    def visit_Return(self, node):
        if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) in (
            "self._forward_decode",
            "self._forward_prefill",
        ):
            assert isinstance(node.value.func, ast.Attribute)
            cls = next(
                n
                for n in ast.parse(Path(impl.__file__).read_text()).body
                if isinstance(n, ast.ClassDef) and n.name == "FlashAttnV100Impl"
            )
            helper = next(
                n
                for n in cls.body
                if isinstance(n, ast.FunctionDef) and n.name == node.value.func.attr
            )
            assert not node.value.keywords
            assert [ast.unparse(a) for a in node.value.args] == [
                a.arg for a in helper.args.args[1:]
            ]
            if helper.name == "_forward_prefill":
                return self._prefill_branches(helper)
            return self._decode_branches(helper)
        return self.generic_visit(node)

    def visit_Expr(self, node):
        if (
            isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == "observe_prefill_reference"
        ):
            return prefill_debug_calculations(node.value)
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


_VERIFY_METHODS = {
    "validate_contract": "_validate_dflash_attention_contract",
    "grouped_verify_allowed": "_dflash2_grouped_verify_allowed",
    "call_grouped_verify": "_call_dflash2_grouped_verify",
    "smallq_xqa_allowed": "_smallq_decode_xqa_allowed",
    "call_smallq_decode_paged": "_call_flash_attn_smallq_decode_paged",
    "small_query_enabled": "_small_query_decode_enabled",
    "tree_prefill": "_flash_v100_ddtree_small_query_prefill_dense",
    "small_query_prefill": "_flash_v100_small_query_prefill_as_decode",
}


class _Normalize(ast.NodeTransformer):
    in_cache = False

    def _log_flag(self, node):
        assert isinstance(node, ast.Constant)
        names = {key: name for name, key in state.LOG_KEYS.items()}
        assert node.value in names
        return names[node.value]

    def visit_Expr(self, node):
        call = node.value
        if (
            isinstance(call, ast.Call)
            and ast.unparse(call.func) == "set_log_once_state"
        ):
            assert not call.keywords and len(call.args) == 2
            assert ast.unparse(call.args[1]) == "True"
            return ast.Assign(
                targets=[ast.Name(self._log_flag(call.args[0]), ast.Store())],
                value=ast.Constant(True),
            )
        return self.generic_visit(node)

    def visit_Name(self, node):
        if node.id == "seen_contracts":
            return ast.Name(id="_logged_dflash_attention_contracts", ctx=node.ctx)
        return node

    def visit_ImportFrom(self, node):
        if node.module == "vllm.v1.attention.ops.sm70_grouped_scalar":
            node.module = "vllm.v1.attention.ops.sm70_e4m3_scalar"
        return node

    def visit_Global(self, node):
        return None

    def visit_FunctionDef(self, node):
        if node.name == "validate_contract":
            assert [a.arg for a in node.args.args] == [
                "layer",
                "attn_metadata",
                "window_size",
            ]
            node.args.args = [ast.arg(arg="self"), *node.args.args[:-1]]
        node.name = _VERIFY_METHODS.get(node.name, node.name)
        self.in_cache = node.name in _CACHE_METHODS
        if node.name in ("_call_flash_attn_decode_paged", "_flash_v100_decode"):
            pairs = list(zip(node.args.kwonlyargs, node.args.kw_defaults))
            for argument, default in pairs:
                if argument.arg == "record":
                    assert default is not None
                    assert ast.unparse(default) == "_plan.record_legacy"
            pairs = [
                (argument, default)
                for argument, default in pairs
                if argument.arg != "record"
            ]
            node.args.kwonlyargs = [argument for argument, _ in pairs]
            node.args.kw_defaults = [default for _, default in pairs]
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
        expression = ast.unparse(node)
        verification = {
            "self.ops.partition_hint": (
                "_routing._mtp5_xqa_dual_cta_partition_size_hint"
            ),
            "self.ops.branch_enabled": (
                "_debug._dflash_ddtree_triton_branch_attn_enabled"
            ),
            "self.ops.branch_strict": "_debug._dflash_ddtree_triton_branch_attn_strict",
            "self.ops.tree_trace_enabled": "_routing._ddtree_trace_enabled",
            "self.ops.tree_trace_event": "_routing._ddtree_trace_event",
            "self.ops.prefix_dump_enabled": "_debug._dflash_prefix_dump_enabled",
            "self.ops.tree_seq_lens_match": "_masks._ddtree_triton_seq_lens_match",
            "self.ops.tree_query_start_match": (
                "_masks._ddtree_triton_query_start_loc_match"
            ),
            "self.ops.tree_parent_ids": "_masks._ddtree_triton_parent_ids_for_query",
            "self.ops.tree_visibility": "_masks._build_ddtree_visibility_mask",
            "self.ops.compare_triton": "self._maybe_compare_triton_output",
            "self.ops.small_query_enabled": "self._small_query_decode_enabled",
            "self.config.grouped_max_query": (
                "self.dflash2_grouped_verify_max_query_tokens"
            ),
            "self.config.grouped_request_major_abi": (
                "self.dflash2_grouped_verify_request_major_abi_version"
            ),
            "self.config.grouped_min_model_len": (
                "self.dflash2_grouped_verify_min_model_len"
            ),
            "self.config.grouped_enabled": "self.use_dflash2_grouped_verify",
            "self.config.grouped_batch_enabled": (
                "self.use_dflash2_batched_grouped_verify"
            ),
            "self.ops.grouped": "self.flash_attn_grouped_verify_paged",
            "self.ops.fp16_grouped": (
                'getattr(self, "flash_attn_grouped_fp16_fp32_paged", None)'
            ),
            "self.ops.e4m3_grouped": (
                'getattr(self, "flash_attn_grouped_e4m3_fp32_paged", None)'
            ),
            "self.ops.xqa": "self.flash_attn_decode_paged_xqa",
            "self.ops.window_size": "self._flash_v100_window_size",
            "self.ops.layer_info": "self._layer_debug_info",
            "self.ops.xqa_codec": "self._xqa_kv_codec",
            "self.ops.decode": "self._call_flash_attn_decode_paged",
            "self.admit_grouped": "self._dflash2_grouped_verify_allowed",
            "self.run_grouped": "self._call_dflash2_grouped_verify",
            "self.admit_xqa": "self._smallq_decode_xqa_allowed",
            "self.run_smallq": "self._call_flash_attn_smallq_decode_paged",
            "self.grouped_admission": "self",
        }
        if expression in verification:
            return ast.parse(verification[expression], mode="eval").body
        if expression.startswith("self.executor."):
            node = ast.parse(
                expression.replace("self.executor.", "self.", 1), mode="eval"
            ).body
            assert isinstance(node, ast.Attribute)
        if ast.unparse(node.value) == "request":
            return ast.Name(id=node.attr, ctx=node.ctx)
        if ast.unparse(node) == "self.ops.triton_forward":
            return ast.parse("super().forward", mode="eval").body
        debug_ops = {
            "profile_trace": "_sm70_profile_trace",
            "draft_debug_enabled": "_draft_graph_debug_enabled",
            "draft_debug_log": "_draft_graph_debug_log",
            "format_debug": "_format_tensor_debug",
        }
        if ast.unparse(node.value) == "self.ops" and node.attr in debug_ops:
            return ast.Attribute(
                value=ast.Name(id="_debug", ctx=ast.Load()),
                attr=debug_ops[node.attr],
                ctx=node.ctx,
            )
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
                "compare_triton": "_maybe_compare_triton_output",
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
        if ast.unparse(node.func) == "log_once_seen":
            assert not node.keywords and len(node.args) == 1
            return ast.Name(self._log_flag(node.args[0]), ast.Load())
        if ast.unparse(node.func) in ("logger.info_once", "logger.warning_once"):
            explicit = [k for k in node.keywords if k.arg == "key"]
            if explicit:
                assert len(explicit) == 1
                self._log_flag(explicit[0].value)
                assert len(node.keywords) == 2
                assert ast.unparse(node.keywords[0].value) == "'process'"
                assert isinstance(node.func, ast.Attribute)
                node.func.attr = node.func.attr.removesuffix("_once")
                node.keywords = []
        if ast.unparse(node.func) == "ddtree_branch_attention_correction":
            argument = next(k for k in node.keywords if k.arg == "impl")
            assert ast.unparse(argument.value) == "self.config"
            argument.value = ast.Name(id="self", ctx=ast.Load())
        if ast.unparse(node.func) == "window_size":
            node.func = ast.parse("self._flash_v100_window_size", mode="eval").body
        if ast.unparse(node.func) == "record":
            node.func = ast.parse("_routing._record_route", mode="eval").body
        if ast.unparse(node.func) in (
            "self._flash_v100_decode",
            "_routing._log_fp8_kv_cache_route",
        ):
            for keyword in node.keywords:
                if keyword.arg == "record":
                    assert ast.unparse(keyword.value) == "record"
            node.keywords = [
                keyword for keyword in node.keywords if keyword.arg != "record"
            ]
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
    from vllm.v1.attention.backends.flash_v100.spec import attention

    assert {
        legacy_name: method for method, legacy_name in _VERIFY_METHODS.items()
    } == attention.VERIFICATION_METHODS
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/flash_v100_impl_methods.json").read_text()
    )["methods"]
    actual = {}
    for path in (
        *Path(impl.__file__).parent.glob("*.py"),
        Path(impl.__file__).parent / "spec/contracts.py",
        Path(impl.__file__).parent / "spec/verifier.py",
        Path(impl.__file__).parent / "spec/diagnostics.py",
    ):
        for node in ast.parse(path.read_text()).body:
            candidates = (
                node.body
                if isinstance(node, ast.ClassDef)
                and node.name
                in (
                    "FlashAttnV100Impl",
                    "DecodeCache",
                    "DecodeExecutor",
                    "VerificationExecutor",
                )
                else [node]
            )
            for fn in candidates:
                if not isinstance(fn, ast.FunctionDef):
                    continue
                if path.name == "prefill.py" and fn.name == "forward":
                    # Expanded and checked through the common forward above.
                    continue
                if path.name == "decode.py" and fn.name in (
                    "__init__",
                    "_flash_v100_window_size",
                    "_xqa_kv_codec",
                    "forward",
                ):
                    continue
                if (
                    path.name in ("verify.py", "verifier.py")
                    and fn.name
                    in ("validate_contract", "_validate_dflash_attention_contract")
                    and fn.args.args[0].arg == "self"
                ):
                    assert len(fn.body) == 1
                    statement = fn.body[0]
                    assert isinstance(statement, (ast.Expr, ast.Return))
                    call = statement.value
                    assert isinstance(call, ast.Call)
                    assert ast.unparse(call.func) == "self.ops.validate_contract"
                    assert not call.keywords
                    assert [ast.unparse(a) for a in call.args] == [
                        "layer",
                        "attn_metadata",
                        "self.ops.window_size"
                        if fn.name == "validate_contract"
                        else "self._flash_v100_window_size",
                    ]
                    continue
                if path.name in ("verify.py", "verifier.py") and fn.name == "__init__":
                    continue
                if any(
                    isinstance(n, ast.Call)
                    and ast.unparse(n.func) == "self._new_verification_executor"
                    for n in ast.walk(fn)
                ):
                    assert len(fn.body) == 1 and isinstance(fn.body[0], ast.Return)
                    call = fn.body[0].value
                    assert isinstance(call, ast.Call)
                    assert isinstance(call.func, ast.Attribute)
                    assert _VERIFY_METHODS[call.func.attr] == fn.name
                    assert [ast.unparse(a) for a in call.args] == [
                        a.arg for a in fn.args.args[1:]
                    ]
                    assert [(k.arg, ast.unparse(k.value)) for k in call.keywords] == [
                        (a.arg, a.arg) for a in fn.args.kwonlyargs
                    ]
                    continue
                if path.name == "impl.py" and fn.name == "_run_prefill_paged_call":
                    # The typed assembly adapter is not a second calculation body.
                    # Check its complete delegation, then hash the original owner.
                    assert [a.arg for a in fn.args.args] == ["self"]
                    assert [a.arg for a in fn.args.kwonlyargs] == ["route"]
                    assert fn.args.kwarg is not None
                    assert fn.args.kwarg.arg == "kwargs"
                    assert len(fn.body) == 1 and isinstance(fn.body[0], ast.Return)
                    assert fn.body[0].value is not None
                    assert ast.unparse(fn.body[0].value) == (
                        "_prefill.PrefillExecutor.run_paged_call("
                        "self._new_prefill_executor(), route=route, **kwargs)"
                    )
                    continue
                if fn.name in ("_forward_decode", "_forward_prefill"):
                    # Expanded and validated with the forward body above.
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
                name = _VERIFY_METHODS.get(
                    fn.name, _CACHE_METHODS.get(fn.name, fn.name)
                )
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
    logger.warning_once.assert_not_called()
    for _ in range(2):
        assert (
            instance._flash_v100_decode_dense_reference(
                None, query, query, metadata, query
            )
            is query
        )
    assert state._logged_decode_dense_reference
    assert legacy._logged_decode_dense_reference
    logger.warning_once.assert_called_once()


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
