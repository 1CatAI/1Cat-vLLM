# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from pathlib import Path

from tools.sm70_fp8_policy_snapshot import POLICY_NAMES, policy_code, snapshot


def test_default_routes_and_explicit_overrides():
    result = snapshot()
    assert len(result["cases"]) == 324
    assert len(result["edge_cases"]) == 270
    assert all(not row["applicable"] for row in result["cases"])
    assert all(
        row["dense_turbomind"]
        and not row["dense_dequant"]
        and not row["moe_dequant_requested"]
        for row in result["edge_cases"][:216]
    )
    rows = {row["config"]: row for row in result["edge_cases"]}
    assert rows["backend=auto/tm=0/moe=1/dequant=1"]["dense_dequant"]
    assert rows["backend=auto/tm=0/moe=1/dequant=1"]["moe_dequant_requested"]
    assert rows["backend=turbomind/tm=0/moe=1/dequant=1"]["dense_turbomind"]
    assert rows["backend=marlin/tm=1/moe=1/dequant=1"]["forced_marlin"]
    assert not rows["backend=marlin/tm=1/moe=1/dequant=1"]["dense_dequant"]


def test_removed_policy_writes_cannot_return():
    code = policy_code(Path("vllm/config/vllm.py").read_text())
    assert not any(name in code.co_consts for name in POLICY_NAMES)


def test_historical_statement_extraction_is_scoped():
    source = """
class VllmConfig:
    def __post_init__(self):
        raise RuntimeError("other startup code must not execute")
        if True:
            os.environ["VLLM_SM70_FP8_TURBOMIND"] = "0"
"""
    code = policy_code(source)
    tree = ast.parse(source)
    assert tree and "VLLM_SM70_FP8_TURBOMIND" in code.co_consts

    class Environment:
        environ = {}

    exec(code, {"os": Environment})
    assert Environment.environ == {"VLLM_SM70_FP8_TURBOMIND": "0"}
