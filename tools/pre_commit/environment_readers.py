# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolve reader identities without importing or evaluating configuration code.

Import/assignment aliases and argument-forwarding helpers retain their identity.
An arbitrary object's ``raw`` method is not classified as an environment reader.
Computed keys remain explicit findings for the lifecycle inventory.
"""

import ast
from collections.abc import Mapping

PREFIXES = ("VLLM_", "TM_", "FLASH_QLA_", "PREFIX_", "SM70_MARLIN_")
BUILTIN_GETTERS = {"os.getenv": 0, "os.environ.get": 0, "os.environ.setdefault": 0}
# Public compatibility helpers: identities, not exempt consumers. Source-tree
# scanning additionally discovers forwarding helpers from their actual bodies.
COMPAT_GETTERS = {
    "vllm.v1.attention.backends.flash_v100.config.registered": 0,
    "vllm.v1.attention.backends.flash_v100.config.raw": 0,
    "vllm.v1.attention.backends.flash_v100.config.env_is_set": 0,
    "vllm.models.qwen4_exp.common.ple.env_gib_bytes": 0,
    "vllm.models.qwen4_exp.nvidia.ops.sm70_qsa_tuning.legacy_qsa_tuning": 0,
}
FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
SCOPES = (*FUNCTIONS, ast.ClassDef)


def qualified(node: ast.AST, bindings: Mapping[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        return bindings.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        base = qualified(node.value, bindings)
        return base + "." + node.attr if base else None
    return None


def scope_nodes(scope):
    for child in ast.iter_child_nodes(scope):
        yield child
        if not isinstance(child, SCOPES):
            yield from scope_nodes(child)


class ReaderScopes:
    """Lexical import, alias and constant bindings, including local shadowing."""

    def __init__(self, tree, module=""):
        self.module = module
        self.bindings = {}
        self.constants = {}
        self.functions = {}
        self._visit(tree, {"os": "os", "envs": "vllm.envs"}, {}, module)

    def _visit(self, scope, inherited, constants, prefix):
        nodes = list(scope_nodes(scope))
        bindings = dict(inherited)
        inherited_constants = constants
        constants = dict(constants)
        stores = {
            n.id
            for n in nodes
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
        }
        if isinstance(scope, FUNCTIONS):
            stores.update(a.arg for a in ast.walk(scope.args) if isinstance(a, ast.arg))
        for name in stores:
            bindings[name] = "__local__." + name
            constants.pop(name, None)
        assignments = {}
        for node in nodes:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    bindings[alias.asname or alias.name.split(".")[0]] = (
                        alias.name if alias.asname else alias.name.split(".")[0]
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                base = node.module
                if node.level:
                    base = ".".join(self.module.split(".")[: -node.level] + [base])
                for alias in node.names:
                    bindings[alias.asname or alias.name] = base + "." + alias.name
            elif isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                name = prefix + "." + node.name if prefix else node.name
                bindings[node.name] = name
                if not isinstance(node, ast.ClassDef):
                    self.functions[node] = name
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                for target in targets:
                    if isinstance(target, ast.Name):
                        assignments.setdefault(target.id, []).append(node.value)
        # Only stable single assignments have a statically usable identity.
        for _ in range(len(assignments) + 1):
            changed = False
            for name, values in assignments.items():
                if len(values) != 1:
                    continue
                value = values[0]
                if isinstance(value, ast.Constant):
                    constants[name] = value
                elif isinstance(value, ast.Name) and value.id in constants:
                    constants[name] = constants[value.id]
                identity = qualified(value, bindings)
                if (
                    identity
                    and not identity.startswith("__local__.")
                    and bindings.get(name) != identity
                ):
                    bindings[name] = identity
                    changed = True
            if not changed:
                break
        for node in [scope, *nodes]:
            self.bindings[node] = bindings
            self.constants[node] = constants
        for node in nodes:
            if isinstance(node, SCOPES):
                name = self.functions.get(
                    node, prefix + "." + getattr(node, "name", "")
                )
                # A method resolves globals, not unqualified class attributes.
                parent_bindings = (
                    inherited if isinstance(scope, ast.ClassDef) else bindings
                )
                parent_constants = (
                    self.constants[scope]
                    if not isinstance(scope, ast.ClassDef)
                    else inherited_constants
                )
                self._visit(node, parent_bindings, parent_constants, name)


def reader_argument(node: ast.AST, bindings, getters) -> ast.AST | None:
    if isinstance(node, ast.Call):
        name = qualified(node.func, bindings)
        index = getters.get(name)
        if index is not None:
            return (
                node.args[index]
                if len(node.args) > index
                else next(
                    (kw.value for kw in node.keywords if kw.arg in ("key", "name")),
                    None,
                )
            )
        if (
            name in ("getattr", "hasattr")
            and len(node.args) >= 2
            and qualified(node.args[0], bindings) == "vllm.envs"
        ):
            return node.args[1]
        if name == "vllm.envs.is_set" and node.args:
            return node.args[0]
        if (
            isinstance(node.func, ast.Subscript)
            and qualified(node.func.value, bindings)
            == "vllm.envs.environment_variables"
        ):
            return node.func.slice
    elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
        if qualified(node.value, bindings) == "os.environ":
            return node.slice
    elif isinstance(node, ast.Compare) and len(node.ops) == 1:
        if (
            isinstance(node.ops[0], (ast.In, ast.NotIn))
            and qualified(node.comparators[0], bindings) == "os.environ"
        ):
            return node.left
    return None


def forwarding_getters(tree: ast.AST, module: str, inherited=None) -> dict[str, int]:
    """Discover functions forwarding a named argument to a proven reader."""
    scopes = ReaderScopes(tree, module)
    getters = {**BUILTIN_GETTERS, **COMPAT_GETTERS, **(inherited or {})}
    found = {}
    for _ in range(len(scopes.functions) + 1):
        changed = False
        for function, name in scopes.functions.items():
            arguments = [
                a.arg for a in (*function.args.posonlyargs, *function.args.args)
            ]
            if arguments and arguments[0] in ("self", "cls"):
                continue  # Requires receiver-type proof, not a method-name guess.
            for node in scope_nodes(function):
                key = reader_argument(node, scopes.bindings[node], getters)
                if (
                    isinstance(key, ast.Name)
                    and key.id in arguments
                    and name not in found
                ):
                    found[name] = getters[name] = arguments.index(key.id)
                    changed = True
        if not changed:
            break
    return found


def registered_and_wrapped_reads(source: str, *, module="", getters=None):
    """Yield (key, node, kind); unresolved key expressions remain explicit."""
    tree = ast.parse(source) if isinstance(source, str) else source
    scopes = ReaderScopes(tree, module)
    readers = {**BUILTIN_GETTERS, **COMPAT_GETTERS, **(getters or {})}
    readers.update(forwarding_getters(tree, module, readers))
    for node in ast.walk(tree):
        bindings = scopes.bindings[node]
        if isinstance(node, ast.ImportFrom) and node.module == "vllm.envs":
            for alias in node.names:
                if alias.name.startswith(PREFIXES):
                    yield ast.Constant(alias.name), node, "registered_import"
        elif isinstance(node, ast.Attribute):
            if qualified(node.value, bindings) == "vllm.envs" and node.attr.startswith(
                PREFIXES
            ):
                yield ast.Constant(node.attr), node, "registered"
        else:
            key = reader_argument(node, bindings, readers)
            if key is not None:
                name = (
                    qualified(node.func, bindings)
                    if isinstance(node, ast.Call)
                    else None
                )
                kind = (
                    "raw"
                    if name in BUILTIN_GETTERS
                    or isinstance(node, (ast.Compare, ast.Subscript))
                    else "getter"
                )
                if isinstance(key, ast.Name):
                    key = scopes.constants[node].get(key.id, key)
                yield key, node, kind
