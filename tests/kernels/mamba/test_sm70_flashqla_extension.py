# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SM70 FlashQLA's JIT extension bootstrap."""

from __future__ import annotations

from types import ModuleType

import pytest

from flash_qla.ops.gated_delta_rule.chunk.sm70 import fused_fwd


@pytest.fixture(autouse=True)
def reset_extension_load_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fused_fwd, "_EXT", None)
    monkeypatch.setattr(fused_fwd, "_EXT_LOAD_ERROR", None)


def test_sm70_flashqla_jit_passes_explicit_gencode_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_extension = ModuleType("flash_qla_sm70_gdn_strided")
    load_kwargs = {}

    def fake_load(**kwargs):
        load_kwargs.update(kwargs)
        return loaded_extension

    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "")
    monkeypatch.setattr(fused_fwd.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(fused_fwd, "load", fake_load)

    assert fused_fwd._load_ext() is loaded_extension
    assert load_kwargs["extra_cuda_cflags"] == [
        "-O3",
        "-gencode=arch=compute_70,code=sm_70",
        "-gencode=arch=compute_75,code=sm_75",
    ]
    assert fused_fwd.os.environ["TORCH_CUDA_ARCH_LIST"] == ""


def test_sm70_flashqla_jit_retains_initial_build_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial_error = RuntimeError("nvcc failed")
    load_calls = 0

    def failing_load(**kwargs):
        nonlocal load_calls
        load_calls += 1
        raise initial_error

    monkeypatch.setattr(fused_fwd.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(fused_fwd, "load", failing_load)

    with pytest.raises(RuntimeError, match="nvcc failed"):
        fused_fwd._load_ext()

    with pytest.raises(
        RuntimeError,
        match="initialization previously failed",
    ) as retry_error:
        fused_fwd._load_ext()

    assert retry_error.value.__cause__ is initial_error
    assert load_calls == 1
