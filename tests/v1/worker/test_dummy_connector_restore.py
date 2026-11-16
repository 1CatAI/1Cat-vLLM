# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dummy model execution restores the connector before returning or raising."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.v1.worker.gpu.model_runner import GPUModelRunner


@pytest.mark.parametrize("is_profile", [False, True])
def test_failed_dummy_execution_restores_connector(is_profile):
    disabled: list[bool] = []
    error = RuntimeError("injected model failure")
    execute = Mock(side_effect=error)
    runner = SimpleNamespace(
        max_num_reqs=8,
        is_first_pp_rank=True,
        execute_model=execute,
        kv_connector=SimpleNamespace(set_disabled=disabled.append),
    )
    with pytest.raises(RuntimeError) as exc:
        GPUModelRunner._dummy_run(runner, 8, is_profile=is_profile)
    assert exc.value is error
    assert disabled == [True, False]
    assert execute.call_args.kwargs["is_profile"] is is_profile
    assert execute.call_args.kwargs["dummy_run"] is True


@pytest.mark.parametrize("is_profile", [False, True])
def test_successful_dummy_execution_restores_connector(is_profile):
    disabled: list[bool | str] = []
    execute = Mock(side_effect=lambda *args, **kwargs: disabled.append("execute"))
    runner = SimpleNamespace(
        max_num_reqs=8,
        is_first_pp_rank=True,
        is_last_pp_rank=False,
        execute_model=execute,
        kv_connector=SimpleNamespace(set_disabled=disabled.append),
    )
    assert GPUModelRunner._dummy_run(
        runner, 8, is_profile=is_profile, skip_eplb=True
    ) == (None, None)
    assert disabled == [True, "execute", False]
