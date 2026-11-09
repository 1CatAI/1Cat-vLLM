# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise CPU loading orchestration without importing CPU-only native ops."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("drafter_state", ["absent", "none", "present"])
def test_cpu_load_model_allows_non_last_pp_drafter(drafter_state):
    path = Path(__file__).resolve().parents[3] / "vllm/v1/worker/cpu_model_runner.py"
    tree = ast.parse(path.read_text())
    runner_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CPUModelRunner"
    )
    method = next(
        node
        for node in runner_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "load_model"
    )
    # Execute the real method body; only its tracing decorator and native
    # model factory are excluded from this CPU-independent lifecycle test.
    method.decorator_list = []
    model = object()
    get_model = Mock(return_value=model)
    namespace = {"logger": Mock(), "get_model": get_model}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    runner = SimpleNamespace(
        model_config=SimpleNamespace(model="test"),
        vllm_config=object(),
        lora_config=None,
        _setup_eagle3_aux_hidden_state_outputs=Mock(),
    )
    if drafter_state != "absent":
        runner.drafter = Mock() if drafter_state == "present" else None
    namespace["load_model"](runner)
    assert runner.model is model
    get_model.assert_called_once_with(vllm_config=runner.vllm_config)
    if drafter_state == "present":
        runner.drafter.load_model.assert_called_once_with(model)
    runner._setup_eagle3_aux_hidden_state_outputs.assert_called_once_with()
