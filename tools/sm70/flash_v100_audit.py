# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reproducible A3 dependency, environment and function-size inventory."""

import ast
import json
from pathlib import Path

import networkx as nx
import regex as re

ROOT = Path(__file__).resolve().parents[2]
PREFIX = "vllm.v1.attention.backends.flash_v100"
PACKAGE = ROOT / PREFIX.replace(".", "/")


# These are construction/compatibility edges, never execution-layer back imports.
# Keep the exact allowlist visible in audit output; all other edges use the rules.
COMPOSITION_EDGES = {
    (PREFIX + source, PREFIX + target)
    for source, target in (
        (".__init__", ".debug"),
        (".__init__", ".debug_compare"),
        (".impl", ".debug_compare"),
        (".impl", ".dense_prefill"),
        (".impl", ".masks"),
        (".impl", ".ops"),
        (".impl", ".plan.diagnostics"),
        (".impl", ".routing"),
        (".impl", ".spec.attention"),
        (".impl", ".spec.verifier"),
        (".verify", ".spec.verifier"),
    )
}
DEFERRED_FUNCTIONS = {"spec/verifier.py:tree_prefill": 321}


def forbidden(source, target):
    if (source, target) in COMPOSITION_EDGES:
        return False

    def layer(module):
        name = module.removeprefix(PREFIX + ".").split(".")[0]
        return {
            "decode": "exec",
            "prefill": "exec",
            "prefill_candidates": "exec",
            "verify": "exec",
            "routing": "plan",
            "debug_compare": "debug",
            "dense_prefill": "ops",
        }.get(name, name)

    source, target = layer(source), layer(target)
    allowed = {
        "impl": {"exec", "config"},
        "exec": {
            "exec",
            "plan",
            "workspace",
            "kv_layout",
            "masks",
            "ops",
            "config",
            "codec",
        },
        "spec": {"spec", "plan", "workspace", "kv_layout", "config"},
    }
    return (target == "debug" and source != "debug") or (
        source in allowed and target not in allowed[source]
    )


def audit():
    graph = nx.DiGraph()
    env = []
    functions = {}
    function_sizes = []
    private = []
    model_hits = 0
    flags = 0
    for path in sorted(PACKAGE.rglob("*.py")):
        module = ".".join(path.relative_to(ROOT).with_suffix("").parts)
        text = path.read_text()
        tree = ast.parse(text)
        aliases = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for item in node.names:
                    target = node.module + "." + item.name
                    if not (ROOT / (target.replace(".", "/") + ".py")).exists():
                        target = node.module
                    if target.startswith(PREFIX) and target != module:
                        graph.add_edge(module, target)
                        aliases[item.asname or item.name] = target
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions[f"{path.relative_to(PACKAGE)}:{node.name}"] = (
                    node.end_lineno - node.lineno + 1
                )
                function_sizes.append(node.end_lineno - node.lineno + 1)
            if (
                isinstance(node, ast.Attribute)
                and node.attr.startswith("_")
                and isinstance(node.value, ast.Name)
                and node.value.id in aliases
            ):
                private.append(f"{path.relative_to(PACKAGE)}:{node.lineno}")
            expression = ast.unparse(node) if isinstance(node, ast.Call) else ""
            is_env = expression.startswith(("os.getenv(", "os.environ.get("))
            via_config = expression.startswith(
                ("_config.registered(", "_config.raw(", "_config.env_is_set(")
            )
            is_env |= (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "envs"
                and node.attr.isupper()
            )
            if is_env or via_config:
                owners = [
                    fn
                    for fn in ast.walk(tree)
                    if isinstance(fn, ast.FunctionDef)
                    and fn.lineno <= node.lineno <= fn.end_lineno
                ]
                owner = (
                    min(owners, key=lambda fn: fn.end_lineno - fn.lineno)
                    if owners
                    else None
                )
                env.append(
                    dict(
                        file=str(path.relative_to(PACKAGE)),
                        line=node.lineno,
                        expression=ast.unparse(node),
                        via_config=via_config,
                        owner=owner.name if owner else "<module>",
                        policy="captured"
                        if owner
                        and owner.name
                        in (
                            "__init__",
                            "_initialize_native_ops",
                            "_initialize_prefill_policy",
                            "_initialize_decode_policy",
                        )
                        else "dynamic",
                    )
                )
        if "spec" not in path.relative_to(PACKAGE).parts:
            for node in tree.body:
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = (
                        node.targets if isinstance(node, ast.Assign) else [node.target]
                    )
                    if any(
                        isinstance(t, ast.Name) and t.id == "ROUTE_SPECS"
                        for t in targets
                    ):
                        node.value = ast.Constant(value="")
            model_hits += len(
                re.findall(r"dflash|ddtree|mtp|qwen|glm", ast.unparse(tree), re.I)
            )
        if path.name == "state.py":
            flags = sum(
                isinstance(n, ast.Assign)
                and isinstance(n.value, ast.Constant)
                and isinstance(n.value.value, bool)
                for n in tree.body
            )
    cycles = sorted(sorted(cycle) for cycle in nx.simple_cycles(graph))
    return {
        "metrics": dict(
            forward=functions["impl.py:forward"],
            largest_function=max(function_sizes),
            largest_active_function=max(
                size
                for name, size in functions.items()
                if name not in DEFERRED_FUNCTIONS
            ),
            cross_module_private=len(private),
            import_cycles=len(cycles),
            model_names_outside_spec=model_hits,
            env_outside_config=sum(
                e["file"] != "config.py" and not e["via_config"] for e in env
            ),
            state_flags=flags,
        ),
        "cycles": cycles,
        "deferred_functions": DEFERRED_FUNCTIONS,
        "composition_edges": sorted(
            [list(e) for e in graph.edges if e in COMPOSITION_EDGES]
        ),
        "edges": sorted([list(edge) for edge in graph.edges]),
        "forbidden_edges": sorted([list(e) for e in graph.edges if forbidden(*e)]),
        "environment": env,
        "functions": functions,
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
