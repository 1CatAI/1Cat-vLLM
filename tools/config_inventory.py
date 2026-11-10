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

import regex as re

from tools.config_boundaries import (
    consumer_lifecycle,
    dynamic_domain,
    parameter_boundary,
)
from tools.config_ownership import ConfigOwnership
from tools.pre_commit.check_env_metadata import read_metadata, registrations
from tools.pre_commit.check_env_registration import (
    NATIVE_SUFFIXES,
    native_reads,
)
from tools.pre_commit.config_lifecycle import initialization_library_loaders
from tools.pre_commit.environment_readers import (
    PREFIXES,
    forwarding_getters,
    registered_and_wrapped_reads,
)

ROOT = Path(__file__).resolve().parents[1]


def python_references(source: str, *, module="", getters=None) -> list[dict]:
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

    result = []
    for key, node, kind in registered_and_wrapped_reads(
        source, module=module, getters=getters
    ):
        name = key.value if isinstance(key, ast.Constant) else None
        name = name if isinstance(name, str) else None
        entry = site(name, node.lineno, kind)
        entry["reader_expression"] = ast.unparse(node)
        if name is None:
            entry["expression"] = ast.unparse(key)
        result.append(entry)
    return [
        item
        for item in result
        if item["name"] is None or item["name"].startswith(PREFIXES)
    ]


def typed_declarations(source: str) -> dict[str, list[dict]]:
    """Index existing alias declarations without establishing another registry."""
    result = defaultdict(list)
    tree = ast.parse(source)
    parents = {
        child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)
    }

    def groups(node):
        result = []
        parent = parents.get(node)
        while parent is not None:
            if isinstance(parent, ast.Dict):
                for key, value in zip(parent.keys, parent.values):
                    if (
                        value is node
                        and isinstance(key, ast.Constant)
                        and isinstance(key.value, str)
                    ):
                        result.append(key.value)
            node, parent = parent, parents.get(parent)
        return result

    for node in ast.walk(tree):
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
                                dict(
                                    field=key.value,
                                    line=alias.lineno,
                                    groups=groups(node),
                                )
                            )
        elif isinstance(node, ast.Tuple) and len(node.elts) >= 2:
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
                entry = dict(field=first.value, line=second.lineno, groups=groups(node))
                if len(node.elts) >= 4 and isinstance(node.elts[2], ast.Tuple):
                    entry["families"] = ast.literal_eval(node.elts[2])
                    entry["diagnostic"] = ast.literal_eval(node.elts[3])
                result[second.value].append(entry)
    # Direct assignments are declarations too (for example gated norm and
    # state diagnostics). Follow only one stable local assignment, never execute
    # a helper or infer a field from a parameter's spelling.
    reads = {node: key for key, node, _ in registered_and_wrapped_reads(tree)}
    for assignment in ast.walk(tree):
        if not isinstance(assignment, ast.Assign) or len(assignment.targets) != 1:
            continue
        target = assignment.targets[0]
        if not (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and target.attr not in ("sources", "errors")
        ):
            continue
        scope = parents.get(assignment)
        while scope is not None and not isinstance(
            scope, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            scope = parents.get(scope)
        definitions = defaultdict(list)
        if scope is not None:
            for item in ast.walk(scope):
                if isinstance(item, ast.Assign):
                    for lhs in item.targets:
                        if isinstance(lhs, ast.Name):
                            definitions[lhs.id].append(item.value)
        pending, visited = [assignment.value], set()
        while pending:
            expression = pending.pop()
            if expression in visited:
                continue
            visited.add(expression)
            for item in ast.walk(expression):
                key = reads.get(item)
                if (
                    isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                    and key.value.startswith(PREFIXES)
                ):
                    result[key.value].append(
                        dict(field=target.attr, line=item.lineno, groups=[])
                    )
                if isinstance(item, ast.Name) and len(definitions[item.id]) == 1:
                    pending.append(definitions[item.id][0])
    return result


def native_policy_fields(root: Path) -> dict[str, str]:
    """Use the shipped ABI declarations, without loading either native library."""
    result = {}
    manifest = root / "csrc/sm70_policy_fields.inc"
    if manifest.is_file():
        result.update(
            ("PolicyField::" + field, alias)
            for field, alias in re.findall(
                r'SM70_POLICY_FIELD\(\s*(\w+)\s*,\s*"([^"]+)"', manifest.read_text()
            )
        )
    header = root / "flash-attention-v100/include/flash_v100_policy.h"
    if header.is_file():
        source = header.read_text()
        enum = re.search(r"enum class Field\s*\{([^}]+)\}", source)
        names = re.search(r"\bnames\[\]\s*=\s*\{([^}]+)\}", source)
        if enum and names:
            fields = [
                field.strip()
                for field in enum[1].split(",")
                if field.strip() != "count"
            ]
            aliases = re.findall(r'"([^"]+)"', names[1])
            if len(fields) != len(aliases):
                raise ValueError("Flash-V100 native policy manifest lengths differ")
            result.update(
                ("flash_v100::policy::Field::" + field, alias)
                for field, alias in zip(fields, aliases)
            )
    return result


def native_policy_references(source: str, fields: dict[str, str]) -> list[dict]:
    """Static bound-field references are distinct from compatibility env reads."""
    # Strings and comments cannot establish an executed policy consumer.
    masked = re.sub(
        r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*[\s\S]*?\*/',
        lambda match: re.sub(r"[^\n]", " ", match[0]),
        source,
    )
    rows = []
    for match in re.finditer(
        r"(?:\bPolicyField::|\bflash_v100::policy::Field::)(\w+)", masked
    ):
        key = match[0]
        if key not in fields:
            continue
        rows.append(
            dict(
                name=fields[key],
                line=source.count("\n", 0, match.start()) + 1,
                kind="native_bound",
                scope="",
                binding=key,
            )
        )
    return rows


def native_scopes(source: str) -> list[tuple[int, int, str]]:
    """Lexical function body ranges, not a reachability or C++ type analysis."""
    masked = re.sub(
        r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*[\s\S]*?\*/',
        lambda match: re.sub(r"[^\n]", " ", match[0]),
        source,
    )
    pairs, stack = {}, []
    for match in re.finditer(r"[{}]", masked):
        if match[0] == "{":
            stack.append(match.start())
        elif stack:
            pairs[stack.pop()] = match.start()
    functions = re.finditer(
        r"\b(\w+)\s*(\((?:[^()]|(?2))*\))\s*(?:const\s*)?(?:noexcept\s*)?\{",
        masked,
    )
    result = []
    for match in functions:
        if match[1] in ("if", "for", "while", "switch", "catch"):
            continue
        start = match.end() - 1
        if start in pairs:
            result.append(
                (
                    source.count("\n", 0, start) + 1,
                    source.count("\n", 0, pairs[start]) + 1,
                    match[1],
                )
            )
    return result


def native_retained_lifecycles(source, scopes):
    """Prove the retained copied helper boundary from shipped registrations.

    Conservatively follow every local function-name reference, including function
    pointers. This is a lexical call graph; unrecognized definitions or additional
    dynamic linkage need review and are never claimed as runtime kernel hits.
    """
    roots = set(re.findall(r"ops\.impl\([^;]+&([a-zA-Z_]\w*)\s*\)", source))
    if not roots:
        return set()
    lines = source.splitlines()
    bodies = {}
    for first, last, name in scopes:
        bodies.setdefault(name, "")
        bodies[name] += "\n".join(lines[first - 1 : last])
    if not roots <= bodies.keys():
        return set()
    reached, pending = set(), list(roots)
    while pending:
        name = pending.pop()
        if name in reached:
            continue
        reached.add(name)
        # Includes address-taking/template calls, over-approximating calls.
        pending.extend(
            set(re.findall(r"\b\w+\b", bodies[name])) & bodies.keys() - reached
        )
    return bodies.keys() - reached


def native_compile_guards(source):
    guards, active = {}, []
    for line, text in enumerate(source.splitlines(), 1):
        directive = re.match(r"\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)", text)
        if directive:
            kind, condition = directive.groups()
            if kind in ("if", "ifdef", "ifndef"):
                active.append(
                    condition.strip() if kind == "if" else f"{kind} {condition.strip()}"
                )
            elif active and kind == "else":
                active[-1] = f"!({active[-1]})"
            elif active and kind == "elif":
                active[-1] = condition.strip()
            elif active and kind == "endif":
                active.pop()
        guards[line] = tuple(active)
    return guards


def collect(root: Path = ROOT, *, strict_metadata=True) -> dict:
    source = (root / "vllm/envs.py").read_text()
    metadata, errors = read_metadata(source)
    if errors and strict_metadata:
        raise ValueError("\n".join(errors))
    getters = registrations(source)
    registry_readers = forwarding_getters(ast.parse(source), "vllm.envs")
    registry_readers.update(
        {key.rsplit(".", 1)[-1]: value for key, value in list(registry_readers.items())}
    )
    dependencies = {
        name: sorted(
            {
                ref["name"]
                for ref in python_references(
                    ast.unparse(getter.args[0]), getters=registry_readers
                )
                if ref["name"] is not None and ref["name"] != name
            }
        )
        for name, getter in getters.items()
        if isinstance(getter, ast.Call)
    }
    names = {
        name
        for name, data in metadata.items()
        if data["acceleration_paths"]
        or name.startswith(("VLLM_SM70_", "VLLM_FLASH_V100_", "FLASH_QLA_", "TM_"))
    }
    paths = subprocess.check_output(
        [
            "git",
            "ls-files",
            "vllm",
            "csrc",
            "flash-attention-v100",
            "flash_qla",
            "lmdeploy",
        ],
        cwd=root,
        text=True,
    ).splitlines()
    sources = {
        filename: (root / filename).read_text(errors="replace")
        for filename in paths
        if (root / filename).is_file()
        and Path(filename).suffix in NATIVE_SUFFIXES | {".py"}
        and filename != "vllm/envs.py"
    }
    wrapped_getters = {}
    reader_trees = {
        filename.removesuffix(".py").replace("/", "."): ast.parse(text)
        for filename, text in sources.items()
        if filename.endswith(".py")
        and any(token in text for token in ("env", "config"))
    }
    while True:
        previous = len(wrapped_getters)
        for module, tree in reader_trees.items():
            wrapped_getters.update(forwarding_getters(tree, module, wrapped_getters))
        if len(wrapped_getters) == previous:
            break
    ownership = ConfigOwnership(sources)
    native_fields = native_policy_fields(root)
    consumers = defaultdict(list)
    declarations = defaultdict(list)
    unresolved = []
    for filename, text in sources.items():
        path = root / filename
        if not path.is_file() or filename == "vllm/envs.py":
            continue
        if path.suffix not in NATIVE_SUFFIXES | {".py"}:
            continue
        if path.suffix == ".py":
            references = python_references(
                text,
                module=filename.removesuffix(".py").replace("/", "."),
                getters=wrapped_getters,
            )
            if filename.startswith("vllm/config/"):
                for name, entries in typed_declarations(text).items():
                    declarations[name].extend(
                        dict(path=filename, **entry) for entry in entries
                    )
        else:
            scopes = native_scopes(text)
            references = [
                dict(
                    name=name,
                    line=line,
                    kind="native",
                    scope=next(
                        (
                            scope
                            for first, last, scope in reversed(scopes)
                            if first <= line <= last
                        ),
                        "",
                    ),
                )
                for name, line in native_reads(text, include_unresolved=True)
            ]
            references.extend(native_policy_references(text, native_fields))
            if filename in (
                "csrc/attention/sm70_grouped_long/kernel/grouped-attention.cu",
                "csrc/attention/sm70_grouped_long/kernel/scalar-attention.cu",
            ):
                unreachable = native_retained_lifecycles(text, scopes)
                for entry in references:
                    if entry["scope"] in unreachable:
                        entry["lifecycle"] = "unreachable_copied_native_helper"
            if filename == "csrc/attention/sm70_79t/prefill.cu":
                guards = native_compile_guards(text)
                for entry in references:
                    entry["compile_guards"] = guards[entry["line"]]
                    if any(
                        guard
                        in (
                            "!defined(PREFIX_TORCH_EXTENSION)",
                            "!(defined(PREFIX_TORCH_EXTENSION))",
                        )
                        for guard in entry["compile_guards"]
                    ):
                        entry["lifecycle"] = "standalone_benchmark"
        for item in references:
            name = item.pop("name")
            entry = dict(path=filename, **item)
            if name is None:
                entry["input_domain"] = dynamic_domain(entry)
                unresolved.append(entry)
            elif name in names or name.startswith(
                ("TM_", "FLASH_QLA_", "PREFIX_", "SM70_MARLIN_")
            ):
                names.add(name)
                consumers[name].append(entry)
    names.update(declarations)
    parameters = {}
    for name in sorted(names):
        data = metadata.get(name, {})
        getter = getters.get(name)
        owners = [
            owner
            for entry in declarations[name]
            for owner in ownership.owners(entry["path"], entry, name)
        ]
        owners = list({(row["owner"], row["field"]): row for row in owners}.values())
        parameters[name] = dict(
            metadata=data,
            registered_dependencies=dependencies.get(name, []),
            parser=ast.unparse(getter.args[0])
            if isinstance(getter, ast.Call)
            else None,
            destination=(
                "; ".join(sorted({row["owner"] for row in owners}))
                if owners
                else (parameter_boundary(name) or {}).get("lifecycle", "unassigned")
            ),
            owners=owners,
            boundary=parameter_boundary(name),
            typed_declarations=declarations[name],
            consumers=consumers[name],
        )
    # Historical aliases can feed a registered getter which itself has a typed
    # owner (for example DECODE_TILE_PROFILE -> PROFILE_TRACE). Preserve that
    # edge instead of inventing a second owner based on the alias spelling.
    for _ in range(len(parameters)):
        changed = False
        for parent, row in parameters.items():
            for name in row["registered_dependencies"]:
                if name not in parameters or not row["owners"]:
                    continue
                child = parameters[name]
                for owner in row["owners"]:
                    if any(
                        (old["owner"], old["field"]) == (owner["owner"], owner["field"])
                        for old in child["owners"]
                    ):
                        continue
                    child["owners"].append({**owner, "via_registration": parent})
                    changed = True
        if not changed:
            break
    for row in parameters.values():
        if row["owners"]:
            row["destination"] = "; ".join(
                sorted({item["owner"] for item in row["owners"]})
            )
    for name, row in parameters.items():
        for site in row["consumers"]:
            site["lifecycle"] = consumer_lifecycle(name, site, row["boundary"])
    return dict(
        parameters=parameters,
        unresolved_dynamic_readers=unresolved,
        metadata_errors=errors,
        policy_contracts=ownership.contracts(),
    )


def summary(inventory: dict) -> dict:
    rows = inventory["parameters"]
    named_sites = {
        (site["path"], site["line"], site["kind"])
        for row in rows.values()
        for site in row["consumers"]
        if site["kind"] != "native_bound"
    }
    dynamic_sites = {
        (site["path"], site["line"], site["kind"])
        for site in inventory["unresolved_dynamic_readers"]
    }
    return dict(
        parameters=len(rows),
        unassigned_parameters=[
            name
            for name, row in rows.items()
            if not row["owners"] and not row.get("boundary")
        ],
        unique_read_sites=len(named_sites | dynamic_sites),
        unique_named_read_sites=len(named_sites),
        unique_dynamic_read_sites=len(dynamic_sites),
        references=dict(
            Counter(site["kind"] for row in rows.values() for site in row["consumers"])
        ),
        destinations=dict(Counter(row["destination"] for row in rows.values())),
        consumer_lifecycles=dict(
            Counter(
                site["lifecycle"] for row in rows.values() for site in row["consumers"]
            )
        ),
        unresolved_dynamic_readers=len(inventory["unresolved_dynamic_readers"]),
        unregistered_dynamic_readers=sum(
            site.get("input_domain") is None
            for site in inventory["unresolved_dynamic_readers"]
        ),
        deprecated=[
            name for name, row in rows.items() if row["metadata"].get("deprecated")
        ],
    )


def closure_errors(inventory: dict) -> list[str]:
    errors = list(inventory.get("metadata_errors", ())) + [
        f"{name}: no typed owner or reviewed retained boundary"
        for name, row in inventory["parameters"].items()
        if not row["owners"] and not row.get("boundary")
    ]
    errors.extend(
        f"{site['path']}:{site['line']}: dynamic reader lacks an input domain "
        f"({site['scope']})"
        for site in inventory["unresolved_dynamic_readers"]
        if site.get("input_domain") is None
    )
    errors.extend(
        f"{site['path']}:{site['line']}: {name} has an unclassified legacy consumer "
        f"({site['scope']})"
        for name, row in inventory["parameters"].items()
        for site in row["consumers"]
        if site.get("lifecycle") == "unclassified"
    )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", action="store_true", help="include every parameter and consumer"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="require an owner/boundary and registered dynamic consumer domain",
    )
    parser.add_argument(
        "--root", type=Path, default=ROOT, help="source checkout to inspect statically"
    )
    args = parser.parse_args()
    inventory = collect(args.root)
    if args.check:
        errors = closure_errors(inventory)
        if errors:
            print("\n".join(errors))
            raise SystemExit(1)
        print("Parameter ownership and dynamic consumer domains are complete.")
        return
    print(
        json.dumps(
            inventory if args.json else summary(inventory), indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
