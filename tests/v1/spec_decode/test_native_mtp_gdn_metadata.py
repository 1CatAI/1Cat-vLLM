# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 Qwen4Exp native MTP shares GDN metadata across cache groups."""

import os
from types import SimpleNamespace

import pytest

from vllm.config.vllm import (
    _SM70_QWEN4EXP_MTP_GDN_METADATA_DEFAULTS,
    _apply_sm70_qwen4exp_mtp_gdn_metadata_defaults,
    _is_sm70_qwen4exp_native_mtp_contract,
)
from vllm.v1.worker.gpu.spec_decode import uses_native_mtp


def _contract_args():
    return (
        SimpleNamespace(architectures=["Qwen4ExpForConditionalGeneration"]),
        SimpleNamespace(method="mtp", num_speculative_tokens=3),
        SimpleNamespace(pipeline_parallel_size=1, enable_dbo=False, ubatch_size=0),
    )


def test_uses_native_mtp_follows_the_speculative_method():
    assert uses_native_mtp(SimpleNamespace(speculative_config=_contract_args()[1]))
    for method in ("dflash", "eagle", "ngram"):
        config = SimpleNamespace(speculative_config=SimpleNamespace(method=method))
        assert not uses_native_mtp(config)
    assert not uses_native_mtp(SimpleNamespace(speculative_config=None))


def test_sm70_qwen4exp_native_mtp_contract_is_narrow():
    assert _is_sm70_qwen4exp_native_mtp_contract(*_contract_args())
    causal = _contract_args()
    causal[0].architectures = ["Qwen4ExpForCausalLM"]
    assert _is_sm70_qwen4exp_native_mtp_contract(*causal)

    for index, attribute, value in (
        (0, "architectures", ["Qwen3_5ForConditionalGeneration"]),
        (1, "method", "dflash"),
        (2, "pipeline_parallel_size", 2),
        (2, "enable_dbo", True),
        (2, "ubatch_size", 2),
    ):
        args = _contract_args()
        setattr(args[index], attribute, value)
        assert not _is_sm70_qwen4exp_native_mtp_contract(*args)
    for index in range(3):
        args = list(_contract_args())
        args[index] = None
        assert not _is_sm70_qwen4exp_native_mtp_contract(*args)


@pytest.mark.parametrize("overridden", sorted(_SM70_QWEN4EXP_MTP_GDN_METADATA_DEFAULTS))
def test_sm70_qwen4exp_mtp_gdn_metadata_defaults_preserve_overrides(
    monkeypatch, overridden
):
    for name in _SM70_QWEN4EXP_MTP_GDN_METADATA_DEFAULTS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(overridden, "0")

    applied = _apply_sm70_qwen4exp_mtp_gdn_metadata_defaults()

    assert overridden not in applied
    assert os.environ[overridden] == "0"
    for name, value in _SM70_QWEN4EXP_MTP_GDN_METADATA_DEFAULTS.items():
        if name != overridden:
            assert name in applied
            assert os.environ[name] == value
