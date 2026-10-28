# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The pynccl all-reduce stays out-of-place unless a caller opts in."""

import pytest

from vllm.envs import environment_variables

inplace_all_reduce_min = environment_variables["VLLM_INPLACE_AR_MIN"]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("VLLM_INPLACE_AR_MIN", raising=False)
    monkeypatch.delenv("VLLM_V41_PACKED_KV", raising=False)


def test_unset_is_out_of_place():
    assert inplace_all_reduce_min() == "0"


def test_deepseek_v41_packed_kv_keeps_in_place_prefill(monkeypatch):
    monkeypatch.setenv("VLLM_V41_PACKED_KV", "1")
    assert inplace_all_reduce_min() == "4194304"


@pytest.mark.parametrize("packed", ["0", "1"])
def test_explicit_value_wins(monkeypatch, packed):
    monkeypatch.setenv("VLLM_V41_PACKED_KV", packed)
    monkeypatch.setenv("VLLM_INPLACE_AR_MIN", "0")
    assert inplace_all_reduce_min() == "0"
    monkeypatch.setenv("VLLM_INPLACE_AR_MIN", "1024")
    assert inplace_all_reduce_min() == "1024"
