# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MLA prefill selection asks for the SM70 dense prefill entry while the model is
built, before the engine binds its Flash-V100 runtime."""

import pytest

from vllm.v1.attention.backends.flash_v100 import (
    dense_prefill,
    ops,
    runtime,
)


@pytest.fixture
def unbound_runtime(monkeypatch):
    # Resources exist (an engine is being built) but hold no Flash-V100 runtime.
    monkeypatch.setattr(runtime, "current_runtime_resources", lambda: {})
    monkeypatch.setattr(ops, "_flash_attn_forward_lse", object())
    monkeypatch.setattr(ops, "_flash_attn_forward_lse_checked", True)


def test_availability_does_not_need_the_bound_runtime(unbound_runtime):
    assert dense_prefill.flash_v100_dense_prefill_lse_available()


def test_execution_still_requires_the_bound_runtime(unbound_runtime):
    with pytest.raises(RuntimeError, match="must be bound before execution"):
        ops.get_flash_dense_forward()
