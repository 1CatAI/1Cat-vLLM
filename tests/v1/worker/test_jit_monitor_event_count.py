# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Repeated compilation variants must remain observable after the first warning."""

from types import SimpleNamespace
from unittest.mock import Mock

from vllm.triton_utils import jit_monitor


def test_counts_repeated_kernel_and_preserves_existing_hook(monkeypatch):
    from triton import knobs

    previous = Mock(return_value="preserved")
    warning = Mock()
    monkeypatch.setattr(jit_monitor, "HAS_TRITON", True)
    monkeypatch.setattr(jit_monitor, "_jit_event_count", 0)
    monkeypatch.setattr(jit_monitor.logger, "warning", warning)
    monkeypatch.setattr(knobs.runtime, "jit_post_compile_hook", previous)
    jit_monitor._setup_triton_jit_hook()
    hook = knobs.runtime.jit_post_compile_hook
    for n in range(1, 4):
        payload = dict(fn=SimpleNamespace(name="same_kernel"), future_kwarg=n)
        assert hook(**payload) == "preserved"
        previous.assert_called_with(**payload)
    assert warning.call_count == 3
    assert [c.args[-1] for c in warning.call_args_list] == [1, 2, 3]
