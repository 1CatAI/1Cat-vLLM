# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PORT_DESIGN §9 AM-2 (runner side): dummy forwards are marked with ForwardContext.is_dummy_run.

GPUModelRunner._dummy_run (memory profiling, CUDA-graph capture, kernel_warmup, DP dummy batches) sets it; the real
execute_model forward does not. Engram keys on this flag. The guards below pin those call sites so a refactor that
drops the flag fails here instead of silently feeding zeros (or raising) inside Engram."""

from __future__ import annotations

import inspect
import re
from types import SimpleNamespace

import pytest

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context, set_forward_context


def test_forward_context_flag_contract() -> None:
    vc = VllmConfig()
    with set_forward_context(None, vc, num_tokens=4, is_dummy_run=True):
        assert get_forward_context().is_dummy_run is True
    with set_forward_context(None, vc, num_tokens=4):
        assert get_forward_context().is_dummy_run is False


def _forward_context_calls(source: str) -> list[str]:
    calls, start = [], 0
    while (i := source.find("set_forward_context(", start)) >= 0:
        depth, j = 0, i + len("set_forward_context")
        for j in range(j, len(source)):
            depth += {"(": 1, ")": -1}.get(source[j], 0)
            if depth == 0:
                break
        calls.append(source[i: j + 1])
        start = j + 1
    return calls


def test_dummy_run_sets_flag_and_execute_model_does_not() -> None:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    dummy = _forward_context_calls(inspect.getsource(GPUModelRunner._dummy_run))
    assert dummy and all(re.search(r"is_dummy_run\s*=\s*True", call) for call in dummy)
    real = _forward_context_calls(inspect.getsource(GPUModelRunner.execute_model))
    assert real and not any("is_dummy_run" in call for call in real)


def test_profile_capture_and_warmup_go_through_dummy_run() -> None:
    from vllm.model_executor.warmup import kernel_warmup
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    assert "self._dummy_run(" in inspect.getsource(GPUModelRunner.profile_run)
    assert "_dummy_run(" in inspect.getsource(kernel_warmup.kernel_warmup)
    capture_sources = "".join(inspect.getsource(fn) for name, fn in inspect.getmembers(GPUModelRunner)
                              if "captur" in name and inspect.isfunction(fn))
    assert "_dummy_run(" in capture_sources


def test_worker_refuses_engram_with_v2_runner() -> None:
    from vllm.v1.worker.gpu_worker import Worker

    worker = Worker.__new__(Worker)
    worker.vllm_config = SimpleNamespace(scheduler_config=SimpleNamespace(async_scheduling=False))
    model = SimpleNamespace(engram_service=object())
    worker.model_runner = SimpleNamespace(model=model, get_model=lambda: model)
    worker.use_v2_model_runner = True
    with pytest.raises(NotImplementedError, match="V1"):
        worker._engram_service()
