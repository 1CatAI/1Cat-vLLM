# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static lifecycle classification shared by policy checks and inventory."""

import ast


def initialization_library_loaders(tree: ast.AST) -> set[ast.AST]:
    """Identify top-level initialization helpers whose only effect loads a DSO.

    This is a structural boundary, not an excluded directory or parameter list.
    Adding a getter wrapper, an operator call or a callback makes the function
    subject to the normal execution-policy check again.
    """
    initial_calls = {
        node.value.func.id
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
    }
    result = set()
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in initial_calls:
            continue
        calls = {
            ast.unparse(child.func)
            for child in ast.walk(node)
            if isinstance(child, ast.Call)
        }
        if "torch.ops.load_library" in calls and calls <= {
            "os.getenv",
            "os.environ.get",
            "torch.ops.load_library",
        }:
            result.add(node)
    return result
