# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Derive configuration ownership from annotations and alias declarations.

This is an explanation tool, not a policy registry. It never imports vLLM or
constructs a configuration. Unknown bindings remain visible for the audit.
"""

import ast
from collections import defaultdict

from tools.pre_commit.environment_readers import ReaderScopes, qualified


class ConfigOwnership:
    def __init__(self, sources: dict[str, str]):
        self.classes = {}
        self.paths = defaultdict(set)
        self.files = {}
        self.nodes = {}
        self.exports = {}
        for path, source in sources.items():
            if not path.startswith("vllm/config/") or not path.endswith(".py"):
                continue
            tree = ast.parse(source)
            module = path.removesuffix(".py").replace("/", ".")
            scopes = ReaderScopes(tree, module)
            self.files[path] = tree
            self.exports.update(
                {
                    module + "." + name: target
                    for name, target in scopes.bindings[tree].items()
                }
            )
            for cls in tree.body:
                if not isinstance(cls, ast.ClassDef):
                    continue
                identity = module + "." + cls.name
                bindings = scopes.bindings[cls]
                fields = {
                    field.target.id: field
                    for field in cls.body
                    if isinstance(field, ast.AnnAssign)
                    and isinstance(field.target, ast.Name)
                    and "ClassVar" not in ast.unparse(field.annotation)
                }
                self.classes[identity] = dict(
                    path=path,
                    node=cls,
                    fields=fields,
                    bindings=bindings,
                    bases=[qualified(base, bindings) for base in cls.bases],
                )
                self.nodes[cls] = identity
        self._visit("vllm.config.vllm.VllmConfig", "", set())

    def resolve(self, identity):
        seen = set()
        while (
            identity not in self.classes
            and identity in self.exports
            and identity not in seen
        ):
            seen.add(identity)
            identity = self.exports[identity]
        return identity

    def fields(self, identity, visited=None):
        identity = self.resolve(identity)
        visited = set() if visited is None else visited
        if identity not in self.classes or identity in visited:
            return {}
        visited.add(identity)
        info = self.classes[identity]
        fields = {}
        for base in info["bases"]:
            fields.update(self.fields(base, visited))
        fields.update(info["fields"])
        return fields

    def _visit(self, identity, path, ancestors):
        identity = self.resolve(identity)
        if identity not in self.classes or identity in ancestors:
            return
        self.paths[identity].add(path)
        info = self.classes[identity]
        for field, node in self.fields(identity).items():
            for target in ast.walk(node.annotation):
                child = self.resolve(qualified(target, info["bindings"]))
                if child in self.classes:
                    self._visit(
                        child,
                        (path + "." if path else "") + field,
                        ancestors | {identity},
                    )
                    break

    def contracts(self):
        """Link resolver/validation/hash implementations, including inheritance.

        Source expressions remain source, not predictions that every option is
        active. An engine report supplies its actual effective hash fields.
        """
        result = {}
        for identity, paths in self.paths.items():
            if not paths or identity not in self.classes:
                continue
            methods, pending, visited = {}, [identity], set()
            while pending:
                current = self.resolve(pending.pop())
                if current in visited or current not in self.classes:
                    continue
                visited.add(current)
                info = self.classes[current]
                for node in info["node"].body:
                    if isinstance(node, ast.FunctionDef) and node.name in (
                        "__post_init__",
                        "resolve",
                        "resolve_fields",
                        "capture_inputs",
                        "compute_hash",
                        "compile_ignored_aliases",
                        "graph_options",
                        "hash_options",
                        "finalize_hash",
                        "bind_consumers",
                        "value",
                        "gib_bytes",
                    ):
                        methods.setdefault(
                            node.name,
                            {
                                "source": f"{info['path']}:{node.lineno}",
                                "implementation": ast.unparse(node),
                            },
                        )
                pending.extend(reversed(info["bases"]))
            result[identity] = {"owners": sorted(paths), "methods": methods}
        return result

    def owners(self, filename: str, entry: dict, alias: str) -> list[dict]:
        """Map a declared field to reachable config instances, not name guesses."""
        field = entry["field"]
        enclosing = [
            identity
            for identity, info in self.classes.items()
            if info["path"] == filename
            and info["node"].lineno <= entry["line"] <= info["node"].end_lineno
        ]
        candidates = enclosing or [
            identity
            for identity, info in self.classes.items()
            if info["path"] == filename and field in self.fields(identity)
        ]
        instances = []
        for identity in candidates:
            instances.extend((identity, owner) for owner in self.paths[identity])
            if field not in self.fields(identity):
                # Channel labels may map to differently named child fields.
                # Read the actual {label: self.child} declaration.
                for node in ast.walk(self.classes[identity]["node"]):
                    if not isinstance(node, ast.Dict):
                        continue
                    for key, value in zip(node.keys, node.values):
                        if not (
                            isinstance(key, ast.Constant)
                            and key.value in entry.get("groups", ())
                            and isinstance(value, ast.Attribute)
                            and isinstance(value.value, ast.Name)
                            and value.value.id == "self"
                        ):
                            continue
                        child_field = self.fields(identity).get(value.attr)
                        if child_field is None:
                            continue
                        for target in ast.walk(child_field.annotation):
                            child = self.resolve(
                                qualified(target, self.classes[identity]["bindings"])
                            )
                            if child in self.classes:
                                instances.extend(
                                    (child, owner + "." + value.attr)
                                    for owner in self.paths[identity]
                                )
                                break
        result = []
        for identity, owner in instances:
            if field not in self.fields(identity):
                continue
            node = self.fields(identity)[field]
            # The same format config class is instantiated for AWQ and FP8.
            # Its declaration has both alias tables; their key is the format.
            families = ("awq", "fp8", "nvfp4", "mxfp4", "gguf", "marlin")
            alias_families = [
                family for family in families if f"_{family.upper()}_" in alias
            ]
            path_families = [
                family
                for family in families
                if family in owner.split(".") or f"sm70_{family}" in owner.split(".")
            ]
            groups = entry.get("groups", ())
            if (
                identity.endswith(".TensorDumpConfig")
                and groups
                and owner.rsplit(".", 1)[-1] not in groups
                and not enclosing
            ):
                continue
            if "families" in entry:
                family = path_families[0] if path_families else "f16"
                if family not in entry["families"]:
                    continue
            if (
                alias_families
                and path_families
                and not set(alias_families) & set(path_families)
            ):
                continue
            result.append(
                dict(
                    owner=owner,
                    field=field,
                    config_class=identity,
                    declared_value=ast.unparse(node.value) if node.value else None,
                    declaration=f"{self.classes[identity]['path']}:{node.lineno}",
                )
            )
        return result
