# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate source declarations and check maintained documentation, without vLLM.

Default/--check is read-only. --write updates only the generated reference;
--json prints the same source facts. No runtime getter or selector is executed.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

import regex as re

from tools.config_ownership import ConfigOwnership
from tools.sm70.path_inventory import stage_binding_declarations

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = "docs/design/architecture/runtime_reference.md"
CODECS = "vllm/v1/attention/kv_codecs.py"
ROUTES = "vllm/v1/attention/backends/flash_v100/routing.py"
BINDINGS = "vllm/model_executor/layers/fused_moe/sm70/declarations.py"
# Documentation scope only, never an execution/capability registry.
MAINTAINED_DOCS = (
    "docs/design/architecture/README.md",
    "docs/design/architecture/int8_g64_codec.md",
    "docs/design/architecture/sm70_phase_e.md",
    "docs/contributing/1cat-development.md",
    "docs/contributing/README.md",
    "CONTRIBUTING.md",
    ".github/PULL_REQUEST_TEMPLATE.md",
    "vllm/v1/attention/kv_codecs.README.md",
    "vllm/v1/attention/backends/flash_v100/README.md",
    "vllm/model_executor/layers/fused_moe/sm70/README.md",
    OUTPUT,
)


class DeclarationError(ValueError):
    pass


def expression(node, values):
    """Read the small expression vocabulary in existing declarations, not Python."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in values:
        return values[node.id]
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        items = [expression(item, values) for item in node.elts]
        return set(items) if isinstance(node, ast.Set) else items
    if isinstance(node, ast.Dict):
        return {
            expression(k, values): expression(v, values)
            for k, v in zip(node.keys, node.values)
        }
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return expression(node.left, values) | expression(node.right, values)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "frozenset"
        and len(node.args) == 1
        and not node.keywords
    ):
        return set(expression(node.args[0], values))
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "torch"
    ):
        return ast.unparse(node)
    raise DeclarationError(f"unsupported declaration: {ast.unparse(node)}")


def assignment(tree, name):
    matches = []
    for node in tree.body:
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
            if isinstance(node, ast.AnnAssign)
            else []
        )
        if any(
            isinstance(target, ast.Name) and target.id == name for target in targets
        ):
            matches.append(node.value)
    if len(matches) != 1:
        raise DeclarationError(f"expected one declaration of {name}")
    return matches[0]


def fields(tree, name, values):
    cls = next(
        (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name), None
    )
    if cls is None:
        raise DeclarationError(f"missing {name}")
    return {
        n.target.id: (
            n.value is not None,
            expression(n.value, values) if n.value is not None else None,
        )
        for n in cls.body
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
    }


def construct(spec, args, keywords):
    names = list(spec)
    if len(args) > len(names) or any(key not in spec for key in keywords):
        raise DeclarationError("unknown declaration field")
    supplied = dict(zip(names, args))
    if supplied.keys() & keywords.keys():
        raise DeclarationError("duplicate declaration field")
    supplied.update(keywords)
    missing = [
        key for key, (default, _) in spec.items() if not default and key not in supplied
    ]
    if missing:
        raise DeclarationError(f"missing declaration fields: {missing}")
    return {key: supplied.get(key, value) for key, (_, value) in spec.items()}


def source_location(path, node):
    return f"{path}:{node.lineno}"


def read_codecs(root):
    tree = ast.parse((root / CODECS).read_text())
    spec = fields(tree, "KVCodec", {})
    if set(spec) != {"name", "aliases", "storage_dtype", "element_dtype", "quantized"}:
        raise DeclarationError("KVCodec schema changed; update its reference renderer")
    declared = assignment(tree, "KV_CODECS")
    if not isinstance(declared, (ast.Tuple, ast.List)):
        raise DeclarationError("KV_CODECS must list codec symbols")
    result = []
    for symbol in declared.elts:
        if not isinstance(symbol, ast.Name):
            raise DeclarationError("KV_CODECS contains a non-symbol")
        node = assignment(tree, symbol.id)
        if (
            not isinstance(node, ast.Call)
            or not isinstance(node.func, ast.Name)
            or node.func.id != "KVCodec"
        ):
            raise DeclarationError(f"{symbol.id}: expected KVCodec declaration")
        data = construct(
            spec,
            [expression(a, {}) for a in node.args],
            {k.arg: expression(k.value, {}) for k in node.keywords},
        )
        result.append(
            dict(symbol=symbol.id, source=source_location(CODECS, node), **data)
        )
    if len({row["symbol"] for row in result}) != len(result):
        raise DeclarationError("duplicate codec symbol")
    return result


def read_routes(root, codecs):
    tree = ast.parse((root / ROUTES).read_text())
    values = {row["symbol"]: row["symbol"] for row in codecs}
    for name in ("_NATIVE_CODECS", "_ALL_CODECS"):
        values[name] = expression(assignment(tree, name), values)
    spec = fields(tree, "RouteSpec", values)
    node = assignment(tree, "ROUTE_SPECS")
    if (
        not isinstance(node, ast.Call)
        or not isinstance(node.func, ast.Name)
        or node.func.id != "_build_specs"
        or len(node.args) != 1
        or node.keywords
    ):
        raise DeclarationError("ROUTE_SPECS must use the declared _build_specs table")
    rows = expression(node.args[0], values)
    if not isinstance(rows, list):
        raise DeclarationError("route declarations must be a sequence")
    result = []
    for row in rows:
        if not isinstance(row, list) or len(row) != 3:
            raise DeclarationError("route row must contain names, stage and options")
        names, stage, options = row
        if not isinstance(names, str) or not isinstance(options, dict):
            raise DeclarationError("route names must be a string and options a mapping")
        for name in names.split():
            data = construct(spec, [name, stage], options)
            data["codecs"] = sorted(data["codecs"])
            result.append(dict(source=source_location(ROUTES, node), **data))
    if len({row["name"] for row in result}) != len(result):
        raise DeclarationError("duplicate route name")
    return sorted(result, key=lambda row: row["name"])


def collect(root=ROOT):
    codecs = read_codecs(root)
    routes = read_routes(root, codecs)
    sources = {
        str(p.relative_to(root)): p.read_text()
        for p in sorted((root / "vllm/config").rglob("*.py"))
    }
    ownership = ConfigOwnership(sources)
    if "vllm.config.vllm.VllmConfig" not in ownership.classes:
        raise DeclarationError("VllmConfig ownership root is missing")
    configurations = []
    for name, data in sorted(ownership.contracts().items()):
        info = ownership.classes[name]
        configurations.append(
            dict(
                name=name,
                owners=data["owners"],
                source=source_location(info["path"], info["node"]),
                methods={
                    key: value["source"] for key, value in data["methods"].items()
                },
            )
        )

    def binding_fields(value):
        if (
            not isinstance(value, (tuple, list))
            or len(value) != 4
            or not all(isinstance(field, str) for field in value)
        ):
            raise DeclarationError(f"invalid MoE binding fields: {value!r}")
        return dict(zip(("binding", "covers", "layout", "arithmetic"), value))

    bindings = []
    tables = stage_binding_declarations(root)
    for stage, modes in tables["STAGE_BINDINGS"].items():
        if not isinstance(stage, str) or not isinstance(modes, dict):
            raise DeclarationError("MoE stages must map names to mode mappings")
        for mode, value in modes.items():
            if not isinstance(mode, str):
                raise DeclarationError("MoE mode must be a string")
            bindings.append(
                dict(
                    family="awq/fp8",
                    stage=stage,
                    mode=mode,
                    source=BINDINGS,
                    **binding_fields(value),
                )
            )
    for key, value in tables["FP4_STAGE_BINDINGS"].items():
        if (
            not isinstance(key, tuple)
            or len(key) != 3
            or not all(isinstance(part, str) for part in key)
        ):
            raise DeclarationError("FP4 binding key must be (family, stage, mode)")
        family, stage, mode = key
        bindings.append(
            dict(
                family=family,
                stage=stage,
                mode=mode,
                source=BINDINGS,
                **binding_fields(value),
            )
        )
    return dict(
        evidence="source declarations; no selector or native execution",
        codecs=codecs,
        routes=routes,
        configurations=configurations,
        bindings=sorted(bindings, key=lambda r: (r["family"], r["stage"], r["mode"])),
    )


def render(report):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    def link(location):
        path, _, line = location.partition(":")
        return f"[{path}](../../../{path}" + (f"#L{line}" if line else "") + ")"

    lines = [
        "# Runtime architecture reference",
        "",
        "Generated by `.venv/bin/python -m "
        "tools.generate_architecture_reference --write`.",
        "Do not edit this file by hand. This describes the accompanying source,",
        "not environment-derived effective settings or observed execution.",
        "",
        "## Configuration owners",
        "",
        "Reachable types are derived from `VllmConfig` annotations. Method links",
        "identify initialization/hash code; they do not evaluate its semantics.",
        "",
        "| Type | Owner paths | Lifecycle / hash methods | Source |",
        "| --- | --- | --- | --- |",
    ]
    for row in report["configurations"]:
        methods = "; ".join(
            f"{key}: {link(value)}" for key, value in row["methods"].items()
        )
        lines.append(
            f"| `{row['name']}` | {cell(', '.join(row['owners']) or '(root)')} "
            f"| {methods} | {link(row['source'])} |"
        )
    lines += [
        "",
        "## KV codecs",
        "",
        "A registered storage descriptor does not establish native support.",
        "",
        "| Symbol | Canonical / aliases | Storage / element | Quantized | Source |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in report["codecs"]:
        lines.append(
            f"| {row['symbol']} | {cell(row['name'])} / "
            f"{cell(', '.join(row['aliases']))} | {row['storage_dtype']} / "
            f"{row['element_dtype']} | {row['quantized']} | {link(row['source'])} |"
        )
    lines += [
        "",
        "## Attention route declarations",
        "",
        "Sorted by name, not execution priority. The executor owns candidate order.",
        "These structural constraints also require native/ABI, metadata and policy",
        "admission. Observer and fallback labels preserve historical accounting.",
        "",
        "| Route | Stage | Codecs | Declared shape / flags | Source |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in report["routes"]:
        shape = {
            key: value
            for key, value in row.items()
            if key not in ("name", "stage", "codecs", "source")
        }
        lines.append(
            f"| `{row['name']}` | {row['stage']} | {', '.join(row['codecs'])} "
            f"| {cell(json.dumps(shape, sort_keys=True))} | {link(row['source'])} |"
        )
    lines += [
        "",
        "## MoE stage bindings",
        "",
        "AWQ/FP8 rows retain the declared suffix; `native_binding` constructs the",
        "native entry name. FP4 rows declare full native names. These tables are",
        "not a Cartesian product of supported model/shape combinations. Existing",
        "selectors enforce supported plans; GGUF and skinny retain their own adapters.",
        "",
        "| Family | Stage / mode | Declared binding | Covers | Layout "
        "| Arithmetic | Source |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["bindings"]:
        columns = [
            row["family"],
            row["stage"] + " / " + row["mode"],
            row["binding"],
            row["covers"],
            row["layout"],
            row["arithmetic"],
        ]
        lines.append(
            "| "
            + " | ".join(cell(value) for value in columns)
            + f" | {link(row['source'])} |"
        )
    return "\n".join(lines) + "\n"


def prose(text):
    return re.sub(r"(?m)^\s*(`{3,}|~{3,}).*?^\s*\1\s*$", "", text, flags=re.S)


def anchors(text):
    result, counts = set(), {}
    for heading in re.findall(r"(?m)^#{1,6}\s+(.+?)\s*#*\s*$", prose(text)):
        slug = re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        count = counts.get(slug, 0)
        counts[slug] = count + 1
        result.add(slug + (f"-{count}" if count else ""))
    result.update(re.findall(r'\b(?:id|name)=["\']([^"\']+)["\']', text))
    return result


def check_links(root, paths=MAINTAINED_DOCS):
    errors = []
    for name in paths:
        source = (root / name).resolve()
        if not source.is_file():
            errors.append(f"{name}: maintained document is missing")
            continue
        for raw in re.findall(
            r"\[[^\]]*\]\(<?([^\s)>]+)>?(?:\s+[^)]*)?\)", prose(source.read_text())
        ):
            url = urlsplit(raw)
            if url.scheme or url.netloc:
                continue
            target = (
                (source.parent / unquote(url.path)).resolve() if url.path else source
            )
            if not target.is_relative_to(root.resolve()) or not target.exists():
                errors.append(f"{name}: missing local target {raw}")
                continue
            fragment = unquote(url.fragment)
            if target.is_dir():
                target = target / "README.md"
            if not fragment:
                continue
            if not target.is_file():
                errors.append(f"{name}: no anchor target {raw}")
            elif target.suffix == ".md":
                if fragment not in anchors(target.read_text()):
                    errors.append(f"{name}: missing anchor {raw}")
            elif re.fullmatch(r"L\d+(?:-L\d+)?", fragment):
                numbers = [int(n) for n in re.findall(r"\d+", fragment)]
                if (
                    not 1
                    <= numbers[0]
                    <= numbers[-1]
                    <= len(target.read_text().splitlines())
                ):
                    errors.append(f"{name}: invalid source lines {raw}")
            else:
                errors.append(f"{name}: unsupported source anchor {raw}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--json", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        report = collect(args.root)
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        generated = render(report)
        output = args.root / OUTPUT
        if args.write:
            output.write_text(generated)
        elif not output.exists() or output.read_text() != generated:
            print(
                f"{OUTPUT} is stale; run "
                ".venv/bin/python -m tools.generate_architecture_reference --write"
            )
            return 1
        errors = check_links(args.root)
        if errors:
            print("\n".join(errors))
            return 1
    except (ValueError, SyntaxError, OSError, KeyError, TypeError) as exc:
        print(f"Architecture reference: {exc}")
        return 1
    print("Architecture declarations and maintained documentation links match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
