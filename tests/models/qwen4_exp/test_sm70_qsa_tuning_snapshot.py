# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Evaluate historical/current tuning assignments, without importing CUDA kernels."""

import ast
import os

import pytest

from vllm.models.qwen4_exp.nvidia.ops.sm70_qsa_tuning import (
    SM70_QSA_TUNING,
    legacy_qsa_tuning,
)

NAMES = (
    "_SM70_INDEXER_SCORE_TILE_BYTES",
    "_SM70_INDEXER_CUBLAS_MIN_ROWS",
    "_SM70_INDEXER_CUBLAS_MIN_SCORE_ELEMENTS",
    "_SM70_QSA_XQA_PAGE4_MIN_ROWS",
)


def evaluate(source):
    selected: list[ast.stmt] = [
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.Assign)
        and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in NAMES
    ]
    namespace = {
        "os": os,
        "SM70_QSA_TUNING": SM70_QSA_TUNING,
        "legacy_qsa_tuning": legacy_qsa_tuning,
    }
    exec(compile(ast.Module(selected, type_ignores=[]), "qsa.py", "exec"), namespace)
    return tuple(namespace[name] for name in NAMES)


@pytest.mark.parametrize("override", (None, "0", "64", "4096", "-1"))
def test_actual_tuning_assignments_match_historical_source(override, monkeypatch):
    import subprocess
    from pathlib import Path

    from vllm import envs

    path = "vllm/models/qwen4_exp/nvidia/ops/qsa.py"
    original = subprocess.check_output(
        ["git", "show", f"cf2a1285e40c09ae498ab8bee5443fb2a147199d:{path}"], text=True
    )
    for name in (
        "VLLM_SM70_QSA_INDEXER_SCORE_TILE_MB",
        "VLLM_SM70_QSA_INDEXER_CUBLAS_MIN_ROWS",
        "VLLM_SM70_QSA_INDEXER_CUBLAS_MIN_SCORE_ELEMENTS",
        "VLLM_SM70_QSA_XQA_PAGE4_MIN_ROWS",
    ):
        monkeypatch.delenv(name, raising=False)
        if override is not None:
            monkeypatch.setenv(name, override)
    envs.disable_envs_cache()
    assert evaluate(original) == evaluate(Path(path).read_text())
    envs.disable_envs_cache()
