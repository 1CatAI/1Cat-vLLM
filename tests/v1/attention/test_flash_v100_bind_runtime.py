# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The dense-forward and grouped-verify getters prepare the FA2 runtime.

Regression tests for the hybrid-model init crash: GLM5next resolved the
LSE-capable dense forward and the SM70 grouped verifier through these
getters, which bound Flash-V100 operations without preparing the attention
runtime (same contract get_flash_ops() already honors).
"""

import sys
from unittest import mock

import vllm.v1.attention.backends.flash_v100.ops as ops


def _install_fake_fa2(monkeypatch):
    interface = mock.MagicMock()
    interface._flash_attn_forward = mock.MagicMock(name="_flash_attn_forward")
    fa = mock.MagicMock()
    fa.flash_attn_interface = interface
    monkeypatch.setitem(sys.modules, "flash_attn_v100", fa)
    monkeypatch.setitem(sys.modules, "flash_attn_v100.flash_attn_interface", interface)


def test_get_flash_dense_forward_prepares_runtime(monkeypatch):
    _install_fake_fa2(monkeypatch)
    called = []
    monkeypatch.setattr(ops, "prepare_attention_runtime", lambda: called.append(True))
    monkeypatch.setattr(ops, "bind_attention_operation", lambda op: op)
    ops._flash_attn_forward_lse_checked = False
    try:
        ops.get_flash_dense_forward()
    finally:
        ops._flash_attn_forward_lse_checked = False
    assert called == [True]


def test_get_flash_grouped_verify_op_prepares_runtime(monkeypatch):
    _install_fake_fa2(monkeypatch)
    called = []
    monkeypatch.setattr(ops, "prepare_attention_runtime", lambda: called.append(True))
    monkeypatch.setattr(ops, "bind_attention_operation", lambda op: op)
    ops._flash_attn_grouped_verify_checked = False
    ops._flash_attn_grouped_verify_paged = None
    try:
        ops.get_flash_grouped_verify_op()
    finally:
        ops._flash_attn_grouped_verify_checked = False
    assert called == [True]


def test_get_flash_ops_prepares_runtime_on_success(monkeypatch):
    """The main ops getter already prepared the runtime; lock the contract."""
    _install_fake_fa2(monkeypatch)
    called = []
    monkeypatch.setattr(ops, "prepare_attention_runtime", lambda: called.append(True))
    monkeypatch.setattr(ops, "bind_attention_operation", lambda op: op)
    monkeypatch.setattr(ops, "get_sm70_splitd_d256_ops", lambda: None)
    monkeypatch.setattr(
        ops._config, "options", lambda: mock.MagicMock(value=lambda *_: False)
    )
    ops._flash_attn_func = mock.MagicMock()
    ops._flash_attn_decode_paged = mock.MagicMock()
    ops._flash_attn_prefill_paged = mock.MagicMock()
    try:
        ops.get_flash_ops()
    finally:
        ops._flash_attn_func = None
        ops._flash_attn_decode_paged = None
        ops._flash_attn_prefill_paged = None
    assert called == [True]
