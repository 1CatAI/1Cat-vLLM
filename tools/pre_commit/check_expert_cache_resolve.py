# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep every routed-expert cache call-compatible with its users.

``GGUFNativeMoEMethod.expert_cache`` is dispatched dynamically:
``gguf_expert_storage`` binds either the plain slot cache
(``gguf_expert_cache.ExpertCache``, built by ``create_cache``) or the
shared-original storage cache
(``gguf_shared_storage.SharedOriginalExpertCache``).  ``gguf_moe.py`` calls
``cache.resolve(layer, ids, topk_weights)`` for both, so every class that can be
bound to ``expert_cache`` must accept that call shape.

A mismatch is a ``TypeError`` raised while CUDA graphs are being captured --
after minutes of weight loading, with nothing in the log naming the expert
cache -- so it reads as a generic "engine core initialization failed".  The
tests for this module are CUDA-only, which is why this is checked here, from
the AST, on CPU-only machines.

The check follows ``expert_cache`` bindings rather than every ``X.resolve()``
call in the package, so unrelated ``resolve()`` methods (``gguf_shared_runtime``,
the SM70 linear policy adapters) are never compared against one another.
"""

import ast
import sys
from pathlib import Path

PACKAGE = Path("vllm/model_executor/layers/quantization")
BINDING = "expert_cache"
METHOD = "resolve"


def _positional_arity(function):
    """Positional arguments a method accepts, ignoring ``self``.

    Returns ``None`` when the method takes ``*args``.
    """
    args = function.args
    if args.vararg is not None:
        return None
    positional = len(args.posonlyargs) + len(args.args)
    if args.args and args.args[0].arg in {"self", "cls"}:
        positional -= 1
    return positional


def _callee_name(node):
    """The name a call or assignment value refers to, without its module."""
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def _annotation_name(node):
    """The bare class name in a return annotation such as ``-> ExpertCache``."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _attribute_targets(node):
    """Assignment targets of ``node`` inside an ``Assign``/``AnnAssign``."""
    if isinstance(node, ast.Assign):
        return node.targets
    if isinstance(node, ast.AnnAssign):
        return [node.target]
    return []


class _FileScan(ast.NodeVisitor):
    """Per-file facts: classes, factories, `expert_cache` bindings and calls."""

    def __init__(self, path):
        self.path = path
        self.classes = {}  # class name -> (arity, line)
        self.factories = {}  # function name -> returned class name
        self.bindings = []  # (raw class-or-factory name, line)
        self.calls = []  # (line, positional args)

    def visit_ClassDef(self, node):
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == METHOD:
                self.classes[node.name] = (_positional_arity(item), item.lineno)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        returned = _annotation_name(node.returns)
        if returned is not None:
            self.factories.setdefault(node.name, returned)
        self._scan_body(node)
        self.generic_visit(node)

    def _scan_body(self, node):
        constructors = {}  # local name -> constructed class-or-factory name
        caches = set()  # local names holding an `expert_cache` value
        for child in ast.walk(node):
            for target in _attribute_targets(child):
                value = getattr(child, "value", None)
                if not isinstance(target, ast.Name) or value is None:
                    continue
                name = _callee_name(value)
                if name is not None and name != BINDING:
                    constructors[target.id] = name
        # Fixpoint: collect locals that hold an `expert_cache` value, so that
        # `cache = self.expert_cache` and `inner = cache` are followed.
        for _ in range(len(constructors) + 1):
            for child in ast.walk(node):
                value = getattr(child, "value", None)
                if value is None:
                    continue
                held = isinstance(value, ast.Attribute) and value.attr == BINDING
                held = held or (isinstance(value, ast.Name) and value.id in caches)
                if not held:
                    continue
                for target in _attribute_targets(child):
                    if isinstance(target, ast.Name):
                        caches.add(target.id)
        # Record the classes the dispatcher can bind to `expert_cache`.
        for child in ast.walk(node):
            for target in _attribute_targets(child):
                if not (isinstance(target, ast.Attribute) and target.attr == BINDING):
                    continue
                value = getattr(child, "value", None)
                bound = _callee_name(value)
                if isinstance(value, ast.Name):
                    bound = constructors.get(value.id)
                if bound is not None and bound != BINDING:
                    self.bindings.append((bound, getattr(child, "lineno", 0)))
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            func = child.func
            if not isinstance(func, ast.Attribute) or func.attr != METHOD:
                continue
            receiver = func.value
            reaches = isinstance(receiver, ast.Attribute) and receiver.attr == BINDING
            reaches = reaches or (
                isinstance(receiver, ast.Name) and receiver.id in caches
            )
            if reaches:
                self.calls.append((child.lineno, len(child.args)))


def violations(directory=PACKAGE):
    """Every expert cache ``resolve`` problem found under ``directory``."""
    scans = []
    for path in sorted(directory.glob("*.py")):
        scan = _FileScan(path)
        scan.visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        scans.append(scan)

    classes = {}
    factories = {}
    for scan in scans:
        for name, (arity, line) in scan.classes.items():
            classes[name] = (arity, f"{scan.path}:{line}")
        factories.update(scan.factories)

    # Both construction routes (`SharedOriginalExpertCache(...)` and
    # `create_cache(...)`) must end at a class that defines `resolve`.
    implementations = {}
    messages = []
    for scan in scans:
        for bound, line in scan.bindings:
            target = factories.get(bound, bound)
            if target in classes:
                implementations[target] = classes[target]
            else:
                messages.append(
                    f"::error file={scan.path},line={line}::'{bound}' is bound "
                    f"to '{BINDING}' but no class defining '{METHOD}' could be "
                    f"found for it"
                )

    for scan in scans:
        for line, arity in scan.calls:
            for name, (accepted, defined_at) in sorted(implementations.items()):
                if accepted is None or arity <= accepted:
                    continue
                messages.append(
                    f"::error file={scan.path},line={line}::'{name}.{METHOD}' "
                    f"(defined at {defined_at}) accepts {accepted} positional "
                    f"argument(s) but this call passes {arity}; every class "
                    f"bound to '{BINDING}' must accept the same call"
                )
    return messages


def main():
    messages = violations()
    for message in messages:
        print(message)
    if messages:
        print(
            f"{len(messages)} expert cache problem(s): the GGUF MoE dispatch "
            f"would raise TypeError at CUDA graph capture."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
