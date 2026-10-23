# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import warnings

from vllm import envs_metadata
from vllm.config.execution_policy import CommunicationPolicy
from vllm.config.sm70_dflash2 import Sm70DFlash2Config


def test_typed_override_warns_without_parsing_legacy_input(monkeypatch):
    monkeypatch.setattr(envs_metadata, "_warned_names", set())
    name = "VLLM_SM70_AWQ_MLP_DOWN_TILE_OVERLAP_KERNEL_REDUCER_BLOCKS"
    monkeypatch.setenv(name, "invalid-but-overridden")
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        first = CommunicationPolicy(awq_overlap_kernel_reducer_blocks=0)
        first.resolve()
        second = CommunicationPolicy(awq_overlap_kernel_reducer_blocks=4)
        second.resolve()
        assert len([w for w in observed if name in str(w.message)]) == 1
    assert first.awq_overlap_kernel_reducer_blocks == 0
    assert second.awq_overlap_kernel_reducer_blocks == 4
    assert first.sources["awq_overlap_kernel_reducer_blocks"] == "typed"


def test_existing_alias_log_does_not_duplicate_structured_warning(monkeypatch, caplog):
    monkeypatch.setattr(envs_metadata, "_warned_names", set())
    name = "VLLM_SM70_DFLASH2_VERIFY_FASTPATH"
    monkeypatch.setenv(name, "0")
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        legacy = Sm70DFlash2Config()
        legacy.resolve(qualified=False)
        typed = Sm70DFlash2Config(verify_fastpath=True)
        typed.resolve(qualified=False)
        assert len([w for w in observed if name in str(w.message)]) == 1
    assert not legacy.verify_fastpath and typed.verify_fastpath
    assert name not in caplog.text
