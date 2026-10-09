# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static parameter/consumer inventory. Never imports or evaluates vLLM getters.

Run with --json for individual parameters, parser expressions, typed declarations
and consumer locations. Findings describe source references, not operator hits or
a proven call graph. Unresolved dynamic readers remain visible for review.
"""

import argparse
import ast
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from tools.pre_commit.check_env_metadata import read_metadata, registrations
from tools.pre_commit.check_env_registration import (
    NATIVE_SUFFIXES,
    native_reads,
    python_reads,
)
from tools.pre_commit.config_lifecycle import initialization_library_loaders

ROOT = Path(__file__).resolve().parents[1]
PREFIXES = ("VLLM_", "TM_", "FLASH_QLA_", "PREFIX_")


def python_references(source: str) -> list[dict]:
    tree = ast.parse(source)
    loading_nodes = {
        child for fn in initialization_library_loaders(tree) for child in ast.walk(fn)
    }
    parents = {
        child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)
    }
    nodes = {node.lineno: node for node in ast.walk(tree) if hasattr(node, "lineno")}

    def site(name, line, kind):
        node = nodes.get(line)
        scope = []
        while node is not None:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                scope.append(node.name)
            node = parents.get(node)
        entry = dict(name=name, line=line, kind=kind, scope=".".join(reversed(scope)))
        if nodes.get(line) in loading_nodes:
            entry["lifecycle"] = "process_library_loading"
        return entry

    result = [site(name, line, "raw") for name, line in python_reads(source)]
    env_modules = {"envs"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            env_modules.update(
                a.asname or a.name for a in node.names if a.name == "vllm.envs"
            )
        elif isinstance(node, ast.ImportFrom) and node.module == "vllm":
            env_modules.update(
                a.asname or a.name for a in node.names if a.name == "envs"
            )
        elif isinstance(node, ast.ImportFrom) and node.module == "vllm.envs":
            result.extend(
                site(a.name, node.lineno, "registered_import")
                for a in node.names
                if a.name.startswith(PREFIXES)
            )
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr.startswith(PREFIXES):
            if ast.unparse(node.value) in env_modules:
                result.append(site(node.attr, node.lineno, "registered"))
        elif isinstance(node, ast.Call):
            function = ast.unparse(node.func)
            method = function.rsplit(".", 1)[-1]
            argument = None
            if method in ("registered", "raw", "env_is_set", "legacy_qsa_tuning"):
                argument = node.args[0] if node.args else None
            elif function == "getattr" and len(node.args) > 1:
                if ast.unparse(node.args[0]) in env_modules:
                    argument = node.args[1]
            elif isinstance(node.func, ast.Subscript) and ast.unparse(
                node.func.value
            ).endswith("environment_variables"):
                argument = node.func.slice
            if argument is not None:
                name = argument.value if isinstance(argument, ast.Constant) else None
                result.append(
                    site(name if isinstance(name, str) else None, node.lineno, "getter")
                )
    return [
        item
        for item in result
        if item["name"] is None or item["name"].startswith(PREFIXES)
    ]


def typed_declarations(source: str) -> dict[str, list[dict]]:
    """Index existing alias declarations without establishing another registry."""
    result = defaultdict(list)
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if (
                    isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                    and key.value.startswith(PREFIXES)
                ):
                    key, value = value, key
                if (
                    isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                    and key.value.isidentifier()
                    and key.value.islower()
                ):
                    # An initialization declaration may carry a parser/default
                    # tuple or multiple historical aliases for the same field.
                    aliases = value.elts if isinstance(value, ast.Tuple) else [value]
                    for alias in aliases:
                        if (
                            isinstance(alias, ast.Constant)
                            and isinstance(alias.value, str)
                            and alias.value.startswith(PREFIXES)
                        ):
                            result[alias.value].append(
                                dict(field=key.value, line=alias.lineno)
                            )
        elif isinstance(node, ast.Tuple) and len(node.elts) >= 3:
            first, second = node.elts[:2]
            if (
                isinstance(first, ast.Constant)
                and isinstance(first.value, str)
                and first.value.isidentifier()
                and first.value.islower()
                and isinstance(second, ast.Constant)
                and isinstance(second.value, str)
                and second.value.startswith(PREFIXES)
            ):
                result[second.value].append(dict(field=first.value, line=second.lineno))
    return result


def destination(name: str, metadata: dict) -> str:
    """Migration grouping, not a replacement for the owning typed declarations."""
    if "DDTREE" in name:
        return "deferred_ddtree"
    if name.endswith(("_LIBRARY", "_LIBRARY_PATH", "_BUILD_DIR", "_SRC_DIR")):
        return "process_loading"
    if metadata.get("category") == "debug" or any(
        word in name for word in ("DUMP", "TRACE", "PROFILE", "COMPARE", "DEBUG")
    ):
        return "observability_config.runtime_trace"
    if name.startswith("PREFIX_") or any(
        word in name for word in ("FLASH_V100", "TURBOQUANT")
    ):
        return "attention_config.flash_v100 / compilation_config.runtime"
    if "GDN" in name or "FLASH_QLA" in name:
        return "kernel_config.gdn"
    if "DFLASH2" in name:
        return "speculative_config.sm70_dflash2"
    if any(word in name for word in ("MTP", "REJECTION", "DRAFT")):
        return "speculative_config.sampling_policy"
    if any(
        word in name
        for word in ("ALLREDUCE", "ALL_REDUCE", "TILE_OVERLAP", "MLP_ENGINE")
    ):
        return "parallel_config.communication"
    if any(word in name for word in ("QSA", "INDEXER", "SPARSE")):
        return "kernel_config.sm70_sparse"
    if any(
        word in name
        for word in (
            "MOE",
            "AWQ",
            "NVFP4",
            "MXFP4",
            "FP8",
            "GGUF",
            "HC",
            "GEMM",
            "GEMV",
            "QPN",
        )
    ):
        return "kernel_config format/layer provider"
    return "review_required"


def collect(root: Path = ROOT) -> dict:
    source = (root / "vllm/envs.py").read_text()
    metadata, errors = read_metadata(source)
    if errors:
        raise ValueError("\n".join(errors))
    getters = registrations(source)
    names = {
        name
        for name, data in metadata.items()
        if data["acceleration_paths"]
        or name.startswith(("VLLM_SM70_", "VLLM_FLASH_V100_", "FLASH_QLA_", "TM_"))
    }
    paths = subprocess.check_output(
        ["git", "ls-files", "vllm", "csrc", "flash-attention-v100"], cwd=root, text=True
    ).splitlines()
    consumers = defaultdict(list)
    declarations = defaultdict(list)
    unresolved = []
    for filename in paths:
        path = root / filename
        if not path.is_file() or filename == "vllm/envs.py":
            continue
        if path.suffix not in NATIVE_SUFFIXES | {".py"}:
            continue
        text = path.read_text(errors="replace")
        if path.suffix == ".py":
            references = python_references(text)
            if filename.startswith("vllm/config/"):
                for name, entries in typed_declarations(text).items():
                    declarations[name].extend(
                        dict(path=filename, **entry) for entry in entries
                    )
        else:
            references = [
                dict(name=name, line=line, kind="native", scope="")
                for name, line in native_reads(text)
            ]
        for item in references:
            name = item.pop("name")
            entry = dict(path=filename, **item)
            if name is None:
                unresolved.append(entry)
            elif name in names or name.startswith(("TM_", "FLASH_QLA_", "PREFIX_")):
                names.add(name)
                consumers[name].append(entry)
    names.update(declarations)
    parameters = {}
    for name in sorted(names):
        data = metadata.get(name, {})
        getter = getters.get(name)
        parameters[name] = dict(
            metadata=data,
            parser=ast.unparse(getter.args[0])
            if isinstance(getter, ast.Call)
            else None,
            destination=destination(name, data),
            typed_declarations=declarations[name],
            consumers=consumers[name],
        )
    return dict(parameters=parameters, unresolved_dynamic_readers=unresolved)


def summary(inventory: dict) -> dict:
    rows = inventory["parameters"]
    return dict(
        parameters=len(rows),
        references=dict(
            Counter(site["kind"] for row in rows.values() for site in row["consumers"])
        ),
        destinations=dict(Counter(row["destination"] for row in rows.values())),
        unresolved_dynamic_readers=len(inventory["unresolved_dynamic_readers"]),
        deprecated=[
            name for name, row in rows.items() if row["metadata"].get("deprecated")
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", action="store_true", help="include every parameter and consumer"
    )
    args = parser.parse_args()
    inventory = collect()
    print(
        json.dumps(
            inventory if args.json else summary(inventory), indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
