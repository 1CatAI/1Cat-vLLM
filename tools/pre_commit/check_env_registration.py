# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reject new VLLM_* environment reads that bypass vllm/envs.py.

`envs.compile_factors()` hashes every registered variable into the
torch.compile cache key. A switch read straight from `os.environ` is invisible
to it: flipping such a switch can load an AOT artifact compiled for the other
setting. Registering the variable in `vllm/envs.py` keeps it in the key; a
variable that never changes compiled code also belongs in `ignored_factors`
there, so it does not invalidate the cache.
"""

import ast
import sys
from pathlib import Path

import regex as re

ENVS_FILE = "vllm/envs.py"

# Direct reads that existed when this check was added. Register them in
# vllm/envs.py (or delete them) and drop them from this list over time; do not
# add new entries.
BASELINE: frozenset[str] = frozenset()

NATIVE_SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp"}


def native_reads(
    content: str, *, include_unresolved: bool = False
) -> list[tuple[str | None, int]]:
    """Find literal keys and constant aliases passed to native env readers.

    Preserve quoted strings and line numbers while removing C/C++ comments.
    Native wrappers such as env_flag_enabled and *_from_env are readers too.
    This is a lexical check, not an evaluator of dynamic string construction.
    """
    tokens = re.compile(
        r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*[\s\S]*?\*/'
    )

    def mask_comments(match):
        token = match.group()
        return re.sub(r"[^\n]", " ", token) if token.startswith(("//", "/*")) else token

    source = tokens.sub(mask_comments, content)
    key_pattern = r"(?:VLLM_|TM_|FLASH_QLA_|PREFIX_|SM70_MARLIN_)[A-Z0-9_]+"
    aliases = list(re.finditer(r'\b(\w+)\s*=\s*"(' + key_pattern + r')"', source))
    arrays = list(re.finditer(r"\b(\w+)\s*\[\s*\]\s*=\s*\{([^}]+)\}", source))
    calls = re.finditer(
        r'\b(\w+)\s*(?:<[^>\n]*>)?\s*\(\s*(?:"(' + key_pattern + r')"|(\w+)\b)',
        source,
    )
    reads = []
    for call in calls:
        if "env" not in call[1].lower():
            continue
        name = call[2]
        if name is None:
            name = next(
                (
                    alias[2]
                    for alias in reversed(aliases)
                    if alias[1] == call[3] and alias.start() < call.start()
                ),
                None,
            )
        line = source.count("\n", 0, call.start()) + 1
        if name:
            reads.append((name, line))
        elif call[3] is not None:
            array = next(
                (
                    item
                    for item in reversed(arrays)
                    if item[1] == call[3] and item.start() < call.start()
                ),
                None,
            )
            if array is not None:
                reads.extend(
                    (key, line)
                    for key in dict.fromkeys(
                        re.findall(r'"(' + key_pattern + r')"', array[2])
                    )
                )
            elif include_unresolved and call[1] in (
                "getenv",
                "secure_getenv",
                "getenv_s",
            ):
                # A computed key must remain visible even when no declaration
                # lets the lexical scanner associate it with a known alias.
                reads.append((None, line))
    return reads


def scan_native_file(path: str, known: set[str]) -> int:
    content = Path(path).read_text(encoding="utf-8")
    missing = {
        (name, line)
        for name, line in native_reads(content)
        if name is not None and name.startswith("VLLM_") and name not in known
    }
    for name, line in sorted(missing):
        print(
            f"{path}:{line}: error: {name} is read by native code but not "
            f"registered in {ENVS_FILE}. Register it with complete metadata."
        )
    return int(bool(missing))


def registered_variables() -> set[str]:
    with open(ENVS_FILE, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.AnnAssign):
            target = node.target
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        else:
            continue
        if (
            isinstance(target, ast.Name)
            and target.id == "environment_variables"
            and isinstance(node.value, ast.Dict)
        ):
            return {
                key.value
                for key in node.value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
    raise RuntimeError(f"environment_variables not found in {ENVS_FILE}")


def python_reads(content: str) -> list[tuple[str | None, int]]:
    """Return literal/constant input names, retaining unresolved dynamic reads."""
    tree = ast.parse(content)
    # Resolve stable module constants, such as DISABLE_ENV. Never substitute
    # a module alias through a function argument/local binding with that name.
    module_constants: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        elif isinstance(node, ast.AnnAssign):
            target = node.target
        else:
            continue
        if (
            isinstance(target, ast.Name)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            bindings = [
                n
                for n in tree.body
                if isinstance(n, (ast.Assign, ast.AnnAssign))
                and any(
                    isinstance(t, ast.Name) and t.id == target.id
                    for t in (n.targets if isinstance(n, ast.Assign) else [n.target])
                )
            ]
            if len(bindings) == 1:
                module_constants[target.id] = node.value.value
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }

    def constant_alias(key: ast.Name) -> str | None:
        parent = parents.get(key)
        while parent is not None:
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                parameters = (
                    *parent.args.posonlyargs,
                    *parent.args.args,
                    *parent.args.kwonlyargs,
                )
                if any(arg.arg == key.id for arg in parameters):
                    return None
                if any(
                    isinstance(n, ast.Name)
                    and n.id == key.id
                    and isinstance(n.ctx, ast.Store)
                    for n in ast.walk(parent)
                ):
                    return None
            parent = parents.get(parent)
        return module_constants.get(key.id)

    os_names = {"os"}
    environ_names: set[str] = set()
    getenv_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            os_names.update(
                alias.asname or alias.name for alias in node.names if alias.name == "os"
            )
        elif isinstance(node, ast.ImportFrom) and node.module == "os":
            for alias in node.names:
                if alias.name == "environ":
                    environ_names.add(alias.asname or alias.name)
                elif alias.name == "getenv":
                    getenv_names.add(alias.asname or alias.name)

    def is_os(node: ast.AST) -> bool:
        return isinstance(node, ast.Name) and node.id in os_names

    def is_environ(node: ast.AST) -> bool:
        return (isinstance(node, ast.Name) and node.id in environ_names) or (
            isinstance(node, ast.Attribute)
            and node.attr == "environ"
            and is_os(node.value)
        )

    reads = []
    for node in ast.walk(tree):
        key: ast.AST | None = None
        if isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and (
                    (func.attr == "getenv" and is_os(func.value))
                    or (func.attr in {"get", "setdefault"} and is_environ(func.value))
                )
            ) or (isinstance(func, ast.Name) and func.id in getenv_names):
                key = (
                    node.args[0]
                    if node.args
                    else next(
                        (kw.value for kw in node.keywords if kw.arg == "key"), None
                    )
                )
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.ctx, ast.Load)
            and is_environ(node.value)
        ):
            key = node.slice
        name = (
            key.value
            if isinstance(key, ast.Constant)
            else constant_alias(key)
            if isinstance(key, ast.Name)
            else None
        )
        if key is not None:
            reads.append((name if isinstance(name, str) else None, node.lineno))
    return reads


def scan_file(path: str, known: set[str]) -> int:
    if Path(path).suffix in NATIVE_SUFFIXES:
        return scan_native_file(path, known)
    reads = python_reads(Path(path).read_text(encoding="utf-8"))
    missing = [
        (name, line)
        for name, line in reads
        if name is not None
        and name.startswith("VLLM_")
        and name not in known | BASELINE
    ]
    for name, line in missing:
        print(
            f"{path}:{line}: error: {name} is read from os.environ but not "
            f"registered in {ENVS_FILE}. Register it there (and add it to "
            "ignored_factors if it never changes compiled code)."
        )
    return int(bool(missing))


def main() -> int:
    known = registered_variables()
    returncode = 0
    for filename in sys.argv[1:]:
        if filename == ENVS_FILE:
            continue
        returncode |= scan_file(filename, known)
    return returncode


if __name__ == "__main__":
    sys.exit(main())
