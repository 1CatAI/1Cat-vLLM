# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep migrated NVFP4 environment compatibility in its configuration adapter.

This complements #711's registration check: registered names can still be
incorrectly read at runtime rather than from a resolved per-engine config.
"""

import ast
import sys
from pathlib import Path

NAMES = {
    "VLLM_SM70_NVFP4_QPN2",
    "VLLM_SM70_NVFP4_QPN2_PREFILL",
    "VLLM_SM70_NVFP4_QPN2_SHARED_WEIGHT",
    "VLLM_SM70_NVFP4_QPN2_SHARED_SCALES",
    "VLLM_SM70_NVFP4_QPN2_PREFILL_MIN_M",
}
ALLOWED = {"vllm/envs.py", "vllm/config/kernel.py"}


def violations(path: Path) -> list[str]:
    if path.as_posix() in ALLOWED or "vllm" not in path.parts:
        return []
    tree = ast.parse(path.read_text())
    errors = []
    for node in ast.walk(tree):
        name = None
        if isinstance(node, ast.Attribute):
            name = node.attr
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            name = node.value
        if name in NAMES:
            errors.append(
                f"{path}:{node.lineno}: {name} belongs to the deprecated compatibility "
                "adapter; consume kernel_config.sm70_nvfp4 instead"
            )
    return errors


def main():
    paths = [Path(name) for name in sys.argv[1:]]
    if not paths:
        paths = list(Path("vllm").rglob("*.py"))
    errors = [
        error for path in paths if path.suffix == ".py" for error in violations(path)
    ]
    print("\n".join(errors))
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
