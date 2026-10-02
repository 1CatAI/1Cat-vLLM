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

import regex as re

ENVS_FILE = "vllm/envs.py"

_READ_PATTERN = re.compile(
    r"""(?:os\.getenv|os\.environ\.get|os\.environ\.setdefault)\(\s*"""
    r"""["'](VLLM_[A-Z0-9_]+)["']"""
    r"""|os\.environ\[\s*["'](VLLM_[A-Z0-9_]+)["']\s*\](?!\s*=[^=])"""
)

# Direct reads that existed when this check was added. Register them in
# vllm/envs.py (or delete them) and drop them from this list over time; do not
# add new entries.
BASELINE: frozenset[str] = frozenset()


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


def scan_file(path: str, known: set[str]) -> int:
    with open(path, encoding="utf-8") as f:
        content = f.read()
    returncode = 0
    for match in _READ_PATTERN.finditer(content):
        name = match.group(1) or match.group(2)
        if name in known or name in BASELINE:
            continue
        line_num = content[: match.start()].count("\n") + 1
        print(
            f"{path}:{line_num}: \033[91merror:\033[0m {name} is read from "
            f"os.environ but not registered in {ENVS_FILE}. Register it there "
            "(and add it to ignored_factors if it never changes compiled code)."
        )
        returncode = 1
    return returncode


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
