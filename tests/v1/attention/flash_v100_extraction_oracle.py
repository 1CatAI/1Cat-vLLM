# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inline final owner extractions into the unchanged calculation fixtures."""

import ast
import copy
from pathlib import Path

from vllm.v1.attention.backends import flash_v100

PACKAGE = Path(flash_v100.__file__).parent
HELPERS = {
    "_prefix_host_metadata": ("prefill.py", "_prefix_host_metadata"),
    "_observe_capture_smallq": ("prefill.py", "_observe_capture_smallq"),
    "_observe_smallq_update": ("spec/verify_metadata.py", "_observe_smallq_update"),
    "self._observe_forward": ("impl.py", "_observe_forward"),
    "self._run_bhmd_decode": ("decode.py", "_run_bhmd_decode"),
    "self._eager_small_query_prefill": (
        "spec/verifier.py",
        "_eager_small_query_prefill",
    ),
    "_forward_with_prefix": ("prefill.py", "_forward_with_prefix"),
    "_validate_prefix_mask": ("prefill.py", "_validate_prefix_mask"),
    "_expand_decode_rows": ("spec/verify_metadata.py", "_expand_decode_rows"),
}


class InlineFinalHelpers(ast.NodeTransformer):
    def inline(self, call):
        if not isinstance(call, ast.Call):
            return None
        name = ast.unparse(call.func)
        if name not in HELPERS:
            return None
        path, function = HELPERS[name]
        helper = next(
            n
            for n in ast.walk(ast.parse((PACKAGE / path).read_text()))
            if isinstance(n, ast.FunctionDef) and n.name == function
        )
        parameters = [a.arg for a in helper.args.args]
        assert not helper.args.defaults and not helper.args.kwonlyargs
        assert not call.keywords
        assert [ast.unparse(a) for a in call.args] == (
            parameters[1:] if name.startswith("self.") else parameters
        )
        return copy.deepcopy(helper.body)

    def visit_Expr(self, node):
        body = self.inline(node.value)
        if body is None:
            return self.generic_visit(node)
        assert not any(
            isinstance(n, ast.Return) for n in ast.walk(ast.Module(body, []))
        )
        return self.visit(ast.Module(body, [])).body

    def visit_Return(self, node):
        body = self.inline(node.value)
        return (
            self.generic_visit(node)
            if body is None
            else self.visit(ast.Module(body, [])).body
        )

    def visit_Assign(self, node):
        body = self.inline(node.value)
        if body is None:
            return self.generic_visit(node)
        returned = body.pop()
        assert isinstance(returned, ast.Return)
        assert returned.value is not None
        assert len(node.targets) == 1
        assert ast.unparse(node.targets[0]) == ast.unparse(returned.value)
        assert not any(
            isinstance(n, ast.Return) for n in ast.walk(ast.Module(body, []))
        )
        return self.visit(ast.Module(body, [])).body


PUBLIC_NAMES = {
    public: old
    for module in flash_v100.SUBMODULES
    for old, public in vars(module).get("LEGACY_ALIASES", {}).items()
}


class LegacyNames(ast.NodeTransformer):
    """Resolve live owner aliases, including calls; never replace calculations."""

    def visit_FunctionDef(self, node):
        node.name = PUBLIC_NAMES.get(node.name, node.name)
        return self.generic_visit(node)

    def visit_Name(self, node):
        if node.id == "feature_dump":
            node.id = "dflash_dump"
        return node

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            node.func.id = PUBLIC_NAMES.get(node.func.id, node.func.id)
        return self.generic_visit(node)

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id in (
            "_debug",
            "_debug_compare",
            "_kv_layout",
            "_routing",
            "_dense_prefill",
            "_metadata",
            "_smallq_metadata",
            "_masks",
            "_ops",
        ):
            node.attr = PUBLIC_NAMES.get(node.attr, node.attr)
        return self.generic_visit(node)
