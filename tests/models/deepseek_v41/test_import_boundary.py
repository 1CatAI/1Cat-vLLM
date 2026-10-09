# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Import-boundary gate for vllm/models/deepseek_v41 (PORT_DESIGN §2.1 rule 3, decisions A1 + A4).

V4.1 must not import the V4 attention stack (its layer typing, per-head q-norm and ratio asserts would be
silently wrong for V4.1) nor the mHC kernels (FP16-stream fused kernels apply the wrong pre, A4). Of the
V4 package it may import only the pieces rule 3 lists. The scan is AST-based (absolute and resolved
relative imports) plus a text grep so ``importlib.import_module("...")`` strings are caught too.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

V4 = "vllm.models.deepseek_v4"

FORBIDDEN = (
    f"{V4}.attention",
    f"{V4}.compressor",
    f"{V4}.sm70.sparse",
    f"{V4}.sm70.indexer",
    f"{V4}.sm70.qnorm_rope_kv_fp8_insert",
    "vllm.model_executor.kernels.mhc",
)

# the only V4 modules rule 3 lets V4.1 import (read-only)
ALLOWED_V4 = (
    f"{V4}.common.ops.fp8_software",
    f"{V4}.sm70.projection",
    f"{V4}.sm70.gemv",
    f"{V4}.nvidia.model",
)


def _matches(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(prefix + ".")


def _module_name(path: Path, package_dir: Path) -> str:
    rel = path.relative_to(package_dir.parents[2]).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve_relative(current: str, is_package: bool, level: int, module: str | None) -> str:
    base = current.split(".") if is_package else current.split(".")[:-1]
    if level > 1:
        base = base[: len(base) - (level - 1)]
    return ".".join(base + ([module] if module else []))


def _imports(path: Path, package_dir: Path) -> list[tuple[int, str]]:
    current = _module_name(path, package_dir)
    is_package = path.name == "__init__.py"
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = (_resolve_relative(current, is_package, node.level, node.module)
                    if node.level else (node.module or ""))
            found.append((node.lineno, base))
            # `from vllm.models.deepseek_v4 import attention` imports the submodule too
            found.extend((node.lineno, f"{base}.{alias.name}") for alias in node.names if alias.name != "*")
    return found


def _violations(module: str) -> list[str]:
    reasons = [f"forbidden import {prefix}" for prefix in FORBIDDEN if _matches(module, prefix)]
    if _matches(module, V4) and module != V4 and not any(_matches(module, a) for a in ALLOWED_V4):
        # `from vllm.models.deepseek_v4.nvidia.model import DeepseekV4MLP` also yields
        # "...nvidia.model.DeepseekV4MLP": accept names below an allowed module
        reasons.append(f"V4 module {module} is not in the rule-3 allow-list")
    return reasons


def _python_files(package_dir: Path) -> list[Path]:
    return sorted(p for p in package_dir.rglob("*.py") if "__pycache__" not in p.parts)


def test_package_has_python_files(ds41_package_dir: Path) -> None:
    assert (ds41_package_dir / "common" / "contracts.py").is_file()


def test_no_forbidden_imports(ds41_package_dir: Path) -> None:
    problems: list[str] = []
    for path in _python_files(ds41_package_dir):
        for lineno, module in _imports(path, ds41_package_dir):
            problems.extend(f"{path}:{lineno}: {why}" for why in _violations(module))
    assert not problems, "deepseek_v41 import-boundary violations:\n" + "\n".join(problems)


def test_no_forbidden_module_strings(ds41_package_dir: Path) -> None:
    problems: list[str] = []
    for path in _python_files(ds41_package_dir):
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            for prefix in FORBIDDEN:
                for token in (prefix, prefix.replace(".", "/")):
                    if token in line:
                        problems.append(f"{path}:{lineno}: mentions {token}")
    assert not problems, "deepseek_v41 sources name forbidden modules:\n" + "\n".join(problems)


@pytest.mark.parametrize("module, bad", [
    (f"{V4}.attention", True),
    (f"{V4}.attention.DeepseekV4Attention", True),
    (f"{V4}.compressor", True),
    (f"{V4}.sm70.sparse", True),
    (f"{V4}.sm70.sparse_kernels", True),        # not in the allow-list
    (f"{V4}.sm70.indexer", True),
    (f"{V4}.sm70.qnorm_rope_kv_fp8_insert", True),
    ("vllm.model_executor.kernels.mhc.tilelang", True),
    (f"{V4}.sm70.projection", False),
    (f"{V4}.sm70.gemv", False),
    (f"{V4}.common.ops.fp8_software", False),
    (f"{V4}.nvidia.model.DeepseekV4MLP", False),
    (V4, False),                                 # the bare package name (yielded by `from ... import x`)
    ("vllm.models.deepseek_v41.common.contracts", False),
    ("vllm.model_executor.layers.fused_moe", False),
])
def test_violation_classifier(module: str, bad: bool) -> None:
    assert bool(_violations(module)) is bad


def test_relative_import_resolution(tmp_path: Path) -> None:
    pkg = tmp_path / "vllm" / "models" / "deepseek_v41"
    (pkg / "sm70").mkdir(parents=True)
    (pkg / "sm70" / "model.py").write_text(
        "from ..common import contracts\nfrom ...deepseek_v4 import attention\nfrom . import moe\n")
    found = {m for _, m in _imports(pkg / "sm70" / "model.py", pkg)}
    assert "vllm.models.deepseek_v41.common.contracts" in found
    assert f"{V4}.attention" in found
    assert "vllm.models.deepseek_v41.sm70.moe" in found
