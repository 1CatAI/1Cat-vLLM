# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    "installed,explicit", [(True, False), (True, True), (False, False)]
)
def test_metrics_helper_preserves_installed_imports(monkeypatch, installed, explicit):
    root = Path(__file__).resolve().parents[2]
    origin = (
        "/runtime/site-packages/vllm/__init__.py"
        if installed
        else str(root / "vllm/__init__.py")
    )
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(__file__=origin))
    monkeypatch.setattr(sys, "path", ["/runtime/site-packages"])
    monkeypatch.delenv("SM70_FLASH_V100_ROOT", raising=False)
    if explicit:
        monkeypatch.setenv("SM70_FLASH_V100_ROOT", "/explicit/flash")
    runpy.run_path(str(root / "benchmarks/benchmark_sm70_model_tokens.py"))
    if installed and not explicit:
        assert sys.path == ["/runtime/site-packages"]
    elif explicit:
        assert sys.path == ["/explicit/flash", "/runtime/site-packages"]
    else:
        assert sys.path[:2] == [str(root), str(root / "flash-attention-v100")]
