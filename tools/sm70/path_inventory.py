# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inventory parameter reads, implementation paths and native calls without a GPU.

This is source evidence, not a claim that a route executed. ``--ref`` reads the
same inventory from a git revision, so moves cannot erase the migration ledger.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUANT = "vllm/model_executor/layers/quantization/"
LINEAR = "vllm/model_executor/kernels/linear/"
SOURCES = (
    *(QUANT + name + "_sm70_moe.py" for name in ("awq", "fp8", "nvfp4", "mxfp4")),
    QUANT + "gguf_turbomind_moe.py",
    "vllm/model_executor/layers/fused_moe/experts/skinny_sm70_moe.py",
    QUANT + "sm70_turbomind.py",
    QUANT + "utils/sm70_layer_workspaces.py",
    QUANT + "utils/nvfp4_qpn2_dequant.py",
    LINEAR + "pre_ampere_qpn.py",
    LINEAR + "scaled_mm/qpn8_blk.py",
    LINEAR + "scaled_mm/sm70_fp8.py",
    LINEAR + "mixed_precision/sm70_awq.py",
    LINEAR + "mixed_precision/sm70_gguf.py",
    LINEAR + "mixed_precision/sm70_gguf_lattice.py",
    LINEAR + "mixed_precision/sm70_gguf_lut4.py",
    LINEAR + "nvfp4/sm70.py",
    "vllm/_sm70_ops.py",
    QUANT + "awq.py",
    QUANT + "fp8.py",
    QUANT + "modelopt.py",
    QUANT + "gguf.py",
    QUANT + "mxfp4.py",
    QUANT + "awq_qpn_sm70.py",
    QUANT + "compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py",
    "vllm/config/kernel.py",
)


def read_source(path: str, ref: str | None) -> str:
    if ref:
        return subprocess.check_output(
            ["git", "show", f"{ref}:{path}"], cwd=ROOT, text=True
        )
    return (ROOT / path).read_text()


def registrations(source: str) -> dict[str, dict[str, str]]:
    """Retain parsing expressions as well as documented defaults/conditions."""
    result = {}
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                continue
            if not key.value.startswith("VLLM_"):
                continue
            fields = {"getter": ast.unparse(value)}
            if isinstance(value, ast.Call) and ast.unparse(value.func) == "env_var":
                fields = {k.arg: ast.unparse(k.value) for k in value.keywords if k.arg}
                fields["getter"] = ast.unparse(value.args[0])
            result[key.value] = fields
    return result


class Inventory(ast.NodeVisitor):
    def __init__(self, path: str):
        self.path = path
        self.scope: list[str] = []
        self.conditions: list[str] = []
        self.parameters: list[dict] = []
        self.calls: list[dict] = []
        self.functions: list[dict] = []

    def location(self, node: ast.AST) -> dict:
        return {
            "file": self.path,
            "line": node.lineno,
            "scope": ".".join(self.scope) or "<import>",
        }

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.scope.append(node.name)
        self.functions.append(
            {**self.location(node), "lines": node.end_lineno - node.lineno + 1}
        )
        self.generic_visit(node)
        self.scope.pop()

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        condition = ast.unparse(node.test)
        for body, expression in (
            (node.body, condition),
            (node.orelse, f"not ({condition})"),
        ):
            self.conditions.append(expression)
            for child in body:
                self.visit(child)
            self.conditions.pop()

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "envs"
            and node.attr.startswith("VLLM_")
        ):
            self.parameters.append(
                {**self.location(node), "name": node.attr, "read": "envs"}
            )
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if ast.unparse(node.value) == "os.environ" and isinstance(
            node.slice, ast.Constant
        ):
            self.parameters.append(
                {
                    **self.location(node),
                    "name": node.slice.value,
                    "read": "os.environ[]",
                }
            )
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        if (
            isinstance(node.left, ast.Constant)
            and isinstance(node.left.value, str)
            and node.left.value.startswith("VLLM_")
            and any(ast.unparse(value) == "os.environ" for value in node.comparators)
        ):
            self.parameters.append(
                {
                    **self.location(node),
                    "name": node.left.value,
                    "read": "explicit override presence",
                }
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = ast.unparse(node.func)
        if name in ("os.getenv", "os.environ.get") and node.args:
            key = node.args[0]
            if (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and key.value.startswith("VLLM_")
            ):
                self.parameters.append(
                    {
                        **self.location(node),
                        "name": key.value,
                        "read": name,
                        "expression": ast.unparse(node),
                    }
                )
        if name.startswith(("sm70_ops.", "torch.ops.")):
            self.calls.append(
                {
                    **self.location(node),
                    "operator": name,
                    "arguments": [ast.unparse(arg) for arg in node.args],
                    "conditions": self.conditions.copy(),
                }
            )
        self.generic_visit(node)


def source_paths(ref: str | None) -> list[str]:
    if ref:
        available = set(
            subprocess.check_output(
                ["git", "ls-tree", "-r", "--name-only", ref], cwd=ROOT, text=True
            ).splitlines()
        )
    else:
        available = {str(p.relative_to(ROOT)) for p in (ROOT / "vllm").rglob("*.py")}
    common = {
        path
        for path in available
        if path.startswith("vllm/model_executor/layers/fused_moe/sm70/")
    }
    common.add("vllm/config/sm70_moe.py")
    return sorted((set(SOURCES) | common) & available)


def inventory(ref: str | None = None) -> dict:
    parameters, calls, functions = [], [], []
    registry = registrations(read_source("vllm/envs.py", ref))
    paths = source_paths(ref)
    for path in paths:
        visitor = Inventory(path)
        visitor.visit(ast.parse(read_source(path, ref)))
        parameters.extend(visitor.parameters)
        calls.extend(visitor.calls)
        functions.extend(visitor.functions)
    names = sorted({row["name"] for row in parameters})
    return {
        "source": ref or "working-tree",
        "evidence": "static call sites; not runtime route hits",
        "counts": {
            "files": len(paths),
            "parameter_reads": len(parameters),
            "parameters": len(names),
            "native_call_sites": len(calls),
            "functions": len(functions),
        },
        "parameters": {name: registry.get(name) for name in names},
        "source_files": paths,
        "reads": parameters,
        "native_calls": calls,
        "functions": functions,
    }


def markdown(result: dict) -> str:
    lines = [
        "# B0 source parameter and native-path ledger",
        "",
        "Generated by `tools/sm70/path_inventory.py --markdown`. "
        "This is a static audit; it does not assert native execution or speed.",
        "",
        "Source: `" + result["source"] + "`.",
        "",
        "## Parameters",
        "",
        "Legacy getter expressions preserve defaults and parsing (including "
        "invalid-value errors). Consumer links identify read timing and the "
        "actual loader, selector, execution or diagnostic function.",
        "",
        "| Legacy name | Parsing/default | Consumers |",
        "|---|---|---|",
    ]
    source_root = (
        "../../.."
        if result["source"] == "working-tree"
        else "https://github.com/1CatAI/1Cat-vLLM/blob/" + result["source"]
    )
    for name, metadata in result["parameters"].items():
        reads = [r for r in result["reads"] if r["name"] == name]
        expression = (metadata or {}).get("getter")
        if expression is None:
            expression = next(
                (r.get("expression") for r in reads if r.get("expression")),
                "presence check / indirect adapter",
            )
        consumers = sorted(
            {f"[{r['scope']}]({source_root}/{r['file']}#L{r['line']})" for r in reads}
        )
        lines.append(
            "| `"
            + name
            + "` | `"
            + expression.replace("|", "\\|")
            + "` | "
            + "<br>".join(consumers)
            + " |"
        )
    lines += [
        "",
        "## Native paths",
        "",
        "Each row is a source call site. Enclosing conditions keep the "
        "candidate order and fallback branches inspectable. Attribute "
        "contracts are prepared by the linked format loader. Full argument "
        "expressions and function sizes are available in the JSON output.",
        "",
        "| Consumer | Native entry | Conditions |",
        "|---|---|---|",
    ]
    for row in result["native_calls"]:
        conditions = (
            " and ".join(row["conditions"]) or "unconditional at this call site"
        )
        lines.append(
            f"| [{row['scope']}]({source_root}/{row['file']}#L{row['line']}) "
            f"| `{row['operator']}` | `" + conditions.replace("|", "\\|") + "` |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()
    result = inventory(args.ref)
    if args.markdown:
        print(markdown(result))
    else:
        print(json.dumps(result["counts"] if args.summary else result, indent=2))


if __name__ == "__main__":
    main()
