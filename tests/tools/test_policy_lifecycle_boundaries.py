# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ast

from tools.config_inventory import python_references
from tools.pre_commit.check_layering import measure
from tools.pre_commit.config_lifecycle import initialization_library_loaders


def test_alias_declarations_do_not_hide_execution_dependencies():
    source = """
aliases = {"projection": "VLLM_SM70_QWEN38_FP16_GEMV"}
value = envs.VLLM_SM70_QWEN38_FP16_GEMV
if model_type == "qwen38":
    run_sm70()
"""
    counts = measure("vllm/generic.py", source)
    assert counts["model"] == 2
    assert counts["platform"] == 2
    assert python_references(source) == [
        dict(name="VLLM_SM70_QWEN38_FP16_GEMV", line=3, kind="registered", scope="")
    ]
    # An actual getter in a table must still be counted.
    counts = measure("vllm/generic.py", 'aliases = {"x": os.getenv("VLLM_SM70_X")}')
    assert counts["platform"] == counts["env"] == 1


def test_loading_boundary_requires_a_direct_initialization_call_and_only_loading():
    source = """
def load():
    path = os.getenv("VLLM_SM70_TEST_LIBRARY")
    enabled = os.getenv("VLLM_SM70_TEST_ENABLE") == "1"
    if path and enabled:
        torch.ops.load_library(path)
load()
"""
    assert len(initialization_library_loaders(ast.parse(source))) == 1
    rows = python_references(source)
    assert all(row["lifecycle"] == "process_library_loading" for row in rows)
    no_initialization = source.rsplit("load()", 1)[0]
    assert not initialization_library_loaders(ast.parse(no_initialization))
    executing = source.replace(
        "        torch.ops.load_library(path)",
        "        torch.ops.load_library(path)\n        torch.ops._C.compute()",
    )
    assert not initialization_library_loaders(ast.parse(executing))
