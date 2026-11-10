# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The speculative target-forward profile scope reads the runner's own trace.

The scope opens before the forward context exists, so a module-level lookup
falls back to building a standalone trace config, on every decode step.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.config.sm70_runtime import RuntimeTraceConfig
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def _runner(trace: RuntimeTraceConfig) -> SimpleNamespace:
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            use_dflash_ddtree=lambda: False, use_dspark=lambda: True
        ),
        device=torch.device("cuda"),
        vllm_config=SimpleNamespace(
            observability_config=SimpleNamespace(runtime_trace=trace)
        ),
    )


def _enter_scope(runner: SimpleNamespace) -> None:
    with GPUModelRunner._dflash_ddtree_target_forward_profile_scope(
        runner,
        use_spec_decode=True,
        num_tokens=6,
        num_reqs=1,
        cudagraph_mode=CUDAGraphMode.FULL,
    ):
        pass


def _forbid_standalone_trace(monkeypatch):
    def fail():
        raise AssertionError("profile scope rebuilt a standalone trace config")

    monkeypatch.setattr("vllm.config.sm70_runtime._standalone_runtime_trace", fail)


def test_scope_does_not_rebuild_a_standalone_trace(monkeypatch):
    trace = RuntimeTraceConfig(spec_target_nvtx=False, spec_target_profiler_step=0)
    _forbid_standalone_trace(monkeypatch)
    _enter_scope(_runner(trace))


def test_scope_follows_the_engine_trace(monkeypatch):
    trace = RuntimeTraceConfig(spec_target_nvtx=True, spec_target_profiler_step=0)
    _forbid_standalone_trace(monkeypatch)
    ranges: list[str] = []
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", ranges.append)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: ranges.append("pop"))
    _enter_scope(_runner(trace))
    assert len(ranges) == 2
    assert ranges[0].startswith("dspark_target_forward:step=1:tokens=6:reqs=1")
    assert ranges[1] == "pop"


@pytest.mark.parametrize("nvtx", [False, True])
def test_scope_ignores_the_environment_after_capture(monkeypatch, nvtx):
    trace = RuntimeTraceConfig(spec_target_nvtx=nvtx, spec_target_profiler_step=0)
    monkeypatch.setenv("VLLM_SM70_SPEC_TARGET_FORWARD_NVTX", "0" if nvtx else "1")
    _forbid_standalone_trace(monkeypatch)
    ranges: list[str] = []
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", ranges.append)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: ranges.append("pop"))
    _enter_scope(_runner(trace))
    assert bool(ranges) is nvtx
