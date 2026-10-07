# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.distributed.device_communicators.cuda_communicator import (
    _sm70_tp4_long_prefill_buffer_bytes,
)

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize(
    ("budget", "capture", "expected_mib"),
    [
        (2048, 32, 20),
        (2048, None, 20),
        (4096, 32, 40),
        (8192, 32, 80),
        (16384, 32, 80),
        (128, 32, 8),
        (2048, 4096, 40),
        (2048, 16384, 80),
    ],
)
def test_fused_collective_covers_scheduler_and_capture(
    monkeypatch: pytest.MonkeyPatch,
    budget: int,
    capture: int | None,
    expected_mib: int,
) -> None:
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=budget),
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=capture),
    )
    monkeypatch.setattr("vllm.config.get_current_vllm_config_or_none", lambda: config)
    assert _sm70_tp4_long_prefill_buffer_bytes() == expected_mib * 2**20


def test_fused_collective_without_engine_retains_full_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("vllm.config.get_current_vllm_config_or_none", lambda: None)
    assert _sm70_tp4_long_prefill_buffer_bytes() == 80 * 2**20
