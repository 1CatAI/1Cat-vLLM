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
from contextlib import suppress
from pathlib import Path

import regex as re

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


C_SOURCES = {
    "C1": (
        "vllm/v1/worker/gpu_model_runner.py",
        "vllm/v1/worker/gpu/model_runner.py",
        "vllm/v1/spec_decode/llm_base_proposer.py",
    ),
    "C2": (
        "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
        "vllm/v1/attention/backends/gdn_attn.py",
        "vllm/v1/worker/gpu/model_states/mamba_hybrid.py",
        "vllm/v1/worker/mamba_utils.py",
    ),
    "C3": ("vllm/config/vllm.py", "vllm/config/speculative.py"),
    "C4": (
        "vllm/model_executor/layers/vocab_parallel_embedding.py",
        "vllm/model_executor/layers/layernorm.py",
        "vllm/model_executor/layers/linear.py",
        "vllm/v1/core/sched/scheduler.py",
        "vllm/distributed/device_communicators/custom_all_reduce.py",
        "vllm/v1/cudagraph_dispatcher.py",
        "vllm/compilation/passes/fusion/allreduce_rms_fusion.py",
    ),
}
C_OWNERS = (
    "vllm/v1/worker/runtime/",
    "vllm/model_executor/warmup/",
    "vllm/model_executor/layers/fla/ops/sm70/",
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
        if name.startswith(
            (
                "sm70_ops.",
                "torch.ops.",
                "self.native_ops.",
                "codec.operators.",
                "self.operators.",
                "state.native_ops.",
                "binding.native.",
            )
        ):
            self.calls.append(
                {
                    **self.location(node),
                    "operator": name,
                    "arguments": [ast.unparse(arg) for arg in node.args],
                    "conditions": self.conditions.copy(),
                }
            )
        self.generic_visit(node)


def source_paths(ref: str | None, phase: str = "b") -> list[str]:
    if ref:
        available = set(
            subprocess.check_output(
                ["git", "ls-tree", "-r", "--name-only", ref], cwd=ROOT, text=True
            ).splitlines()
        )
    else:
        available = {str(p.relative_to(ROOT)) for p in (ROOT / "vllm").rglob("*.py")}
    if phase == "c":
        sources = {path for group in C_SOURCES.values() for path in group}
        sources.update(path for path in available if path.startswith(C_OWNERS))
        sources.update(
            {"vllm/config/sm70_runtime.py", "vllm/v1/spec_decode/profiling.py"}
        )
        return sorted(sources & available)
    common = {
        path
        for path in available
        if path.startswith(
            (
                "vllm/model_executor/layers/fused_moe/sm70/",
                "vllm/model_executor/kernels/linear/qpn/",
                "vllm/_sm70/",
            )
        )
    }
    common.update(
        {
            "vllm/config/sm70_moe.py",
            "vllm/config/sm70_native.py",
            LINEAR + "sm70_provider.py",
            QUANT + "compressed_tensors/schemes/compressed_tensors_w8a16_fp8.py",
            QUANT + "awq_marlin.py",
            "vllm/model_executor/warmup/sm70_native_cache.py",
            "vllm/model_executor/warmup/awq_sm70_warmup.py",
        }
    )
    return sorted((set(SOURCES) | common) & available)


def runtime_catalog() -> dict:
    """Read initialization aliases from the same dictionaries used by runtime config."""
    tree = ast.parse(read_source("vllm/config/sm70_runtime.py", None))
    aliases = []
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef):
            continue
        for node in ast.walk(cls):
            if (
                isinstance(node, ast.Call)
                and ast.unparse(node.func) == "resolve_legacy_fields"
            ):
                fields = ast.literal_eval(node.args[1])
                aliases.extend(
                    dict(
                        legacy=legacy,
                        typed=f"{cls.name}.{field}",
                        timing="initialization only",
                    )
                    for field, legacy in fields.items()
                )
    return {
        "evidence": "configuration declarations; not runtime launches",
        "aliases": aliases,
    }


def inventory(ref: str | None = None, phase: str = "b") -> dict:
    parameters, calls, functions = [], [], []
    registry = registrations(read_source("vllm/envs.py", ref))
    paths = source_paths(ref, phase)
    for path in paths:
        visitor = Inventory(path)
        visitor.visit(ast.parse(read_source(path, ref)))
        parameters.extend(visitor.parameters)
        calls.extend(visitor.calls)
        functions.extend(visitor.functions)
    names = sorted({row["name"] for row in parameters})
    declared = (
        (binding_catalog() if phase == "b" else runtime_catalog())
        if ref is None
        else {}
    )
    for row in declared.get("aliases", []) + declared.get("native_parameters", []):
        if row["legacy"] not in names:
            names.append(row["legacy"])
    return {
        "phase": phase,
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
        "initialized_declarations": declared,
        "source_files": paths,
        "reads": parameters,
        "native_calls": calls,
        "functions": functions,
    }


def markdown(result: dict) -> str:
    lines = [
        (
            f"# {result.get('phase', 'b').upper()}0 source parameter "
            "and native-path ledger"
        ),
        "",
        (
            "Generated by `tools/sm70/path_inventory.py --markdown`. "
            "This is a static audit; it does not assert native execution or speed."
        ),
        "",
        "Source: `" + result["source"] + "`.",
        "",
        "## Parameters",
        "",
        (
            "Legacy getter expressions preserve defaults and parsing (including "
            "invalid-value errors). Consumer links identify read timing and the "
            "actual loader, selector, execution or diagnostic function."
        ),
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
        (
            "Each row is a source call site. Enclosing conditions keep the "
            "candidate order and fallback branches inspectable. Attribute "
            "contracts are prepared by the linked format loader. Full argument "
            "expressions and function sizes are available in the JSON output."
        ),
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


def binding_catalog() -> dict:
    """Read the very declarations consumed by codecs, without importing Torch."""

    def assignments(path):
        result = {}
        for node in ast.parse(read_source(path, None)).body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                with suppress(ValueError, TypeError):
                    result[node.targets[0].id] = ast.literal_eval(node.value)
        return result

    policy = assignments("vllm/config/sm70_moe.py")
    aliases = dict(
        policy["ALIASES"], nvfp4=policy["NVFP4_ALIASES"], mxfp4=policy["MXFP4_ALIASES"]
    )
    linear = assignments("vllm/config/kernel.py")
    for family in ("awq", "fp8", "nvfp4"):
        aliases["linear_" + family] = linear[
            "SM70_" + family.upper() + "_LINEAR_ALIASES"
        ]
    registry = registrations(read_source("vllm/envs.py", None))
    bindings = assignments("vllm/model_executor/layers/fused_moe/sm70/declarations.py")[
        "FP4_STAGE_BINDINGS"
    ]
    native = assignments("vllm/config/sm70_native.py")["NATIVE_FIELDS"]
    cpp_paths = (
        "csrc/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.cu",
        "csrc/sm70_turbomind/ops/awq_sm70_gemm.cu",
        "csrc/sm70_turbomind/ops/nvfp4_qpn2_sm70.cu",
        "csrc/sm70_turbomind/ops/qwen38_prefill_cutlass.cu",
        "csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/gemm.cu",
        "csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/kernel/sm70_884_4.cu",
    )
    consumers = {field: [] for field, *_ in native}
    for path in cpp_paths:
        for line, source in enumerate(read_source(path, None).splitlines(), 1):
            for field in consumers:
                if re.search(r"PolicyField::" + re.escape(field) + r"\b", source):
                    consumers[field].append({"file": path, "line": line})
    return {
        "evidence": "configuration and binding declarations; no native execution claim",
        "native_parameters": [
            dict(
                legacy=alias,
                field=field,
                families=families,
                diagnostic=diagnostic,
                getter=registry.get(alias, {}).get("getter", "native unset default"),
                consumers=consumers[field],
                timing="owner initialization; frozen native argument at execution",
                precedence=(
                    "typed parent/native (conflict rejected) > "
                    "captured legacy > original native default"
                ),
            )
            for field, alias, families, diagnostic in native
        ],
        "aliases": [
            {
                "legacy": name,
                "typed": (
                    ("sm70_" + family.removeprefix("linear_"))
                    if family.startswith("linear_")
                    else "sm70_moe." + family
                )
                + "."
                + field,
                "getter": registry[name]["getter"],
                "timing": "engine initialization",
                "precedence": "explicit typed value > legacy getter/default",
            }
            for family, fields in aliases.items()
            for field, name in fields.items()
        ],
        "bindings": [
            dict(
                family=key[0],
                stage=key[1],
                mode=key[2],
                operator=value[0],
                covers=value[1],
                layout=value[2],
                arithmetic=value[3],
            )
            for key, value in bindings.items()
        ],
    }


def binding_markdown(catalog: dict) -> str:
    lines = [
        "# Phase B declared configuration and FP4 stage bindings",
        "",
        "Generated by `tools/sm70/path_inventory.py --bindings --markdown`.",
        "",
        (
            "Typed values override legacy getters at engine initialization. "
            "Selectors keep model/shape/native gates and their existing fallbacks. "
            "These declarations identify implementations; they do not assert execution."
        ),
        "",
        "## Format-specific aliases",
        "",
        "| Legacy name | Typed option | Original getter/default |",
        "|---|---|---|",
    ]
    for row in catalog["aliases"]:
        getter = row["getter"].replace("|", "\\|")
        lines.append(f"| `{row['legacy']}` | `{row['typed']}` | `{getter}` |")
    lines += [
        "",
        (
            "Common AWQ/FP8 single-token aliases retain the OR/priority rules "
            "described in the [main design](sm70_phase_b.md)."
        ),
        "",
        "## FP4 bindings",
        "",
        "| Format / mode | Native operator | Covered stages | Layout | Arithmetic |",
        "|---|---|---|---|---|",
    ]
    for row in catalog["bindings"]:
        lines.append(
            f"| {row['family']} / {row['stage']} / {row['mode']} | "
            f"`{row['operator']}` | {row['covers']} | {row['layout']} | "
            f"{row['arithmetic']} |"
        )
    lines += [
        "",
        "## Native policy arguments",
        "",
        (
            "Native policy is captured once. An unset value remains a null sentinel, "
            "preserving each native consumer's own default (which can differ from "
            "the Python compatibility getter). Consumer links are authoritative. "
            "Explicit parent and native requests for the same alias must agree. "
            "FP16 auxiliary aliases retain their independent legacy owner "
            "outside Phase B."
        ),
        "",
        (
            "| Legacy alias / field | Families | Compatibility getter | "
            "Native consumers | Hash role |"
        ),
        "|---|---|---|---|---|",
    ]
    for row in catalog["native_parameters"]:
        links = "<br>".join(
            f"[source](../../../{site['file']}#L{site['line']})"
            for site in row["consumers"]
        )
        getter = row["getter"].replace("|", "\\|")
        lines.append(
            f"| `{row['legacy']}` / `{row['field']}` | "
            f"{', '.join(row['families'])} | `{getter}` | {links} | "
            + ("diagnostic" if row["diagnostic"] else "calculation")
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref")
    parser.add_argument("--phase", choices=("b", "c"), default="b")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--bindings", action="store_true")
    args = parser.parse_args()
    if args.bindings:
        catalog = binding_catalog()
        print(
            binding_markdown(catalog).rstrip()
            if args.markdown
            else json.dumps(catalog, indent=2)
        )
        return
    result = inventory(args.ref, args.phase)
    if args.markdown:
        print(markdown(result))
    else:
        print(json.dumps(result["counts"] if args.summary else result, indent=2))


if __name__ == "__main__":
    main()
