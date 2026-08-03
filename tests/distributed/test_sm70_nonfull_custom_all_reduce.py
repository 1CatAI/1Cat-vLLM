# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.distributed.device_communicators.custom_all_reduce import (
    _allow_sm70_tp8_nonfull_custom_ar,
)


def test_sm70_tp8_nonfull_custom_ar_is_explicit(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_SM70_TP8_NONFULL_CUSTOM_AR", raising=False)
    assert not _allow_sm70_tp8_nonfull_custom_ar(8, False, (7, 0))

    monkeypatch.setenv("VLLM_SM70_TP8_NONFULL_CUSTOM_AR", "1")
    assert _allow_sm70_tp8_nonfull_custom_ar(8, False, (7, 0))


def test_sm70_tp8_nonfull_custom_ar_rejects_other_contracts(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_SM70_TP8_NONFULL_CUSTOM_AR", "1")
    assert not _allow_sm70_tp8_nonfull_custom_ar(4, False, (7, 0))
    assert not _allow_sm70_tp8_nonfull_custom_ar(8, True, (7, 0))
    assert not _allow_sm70_tp8_nonfull_custom_ar(8, False, (8, 0))
