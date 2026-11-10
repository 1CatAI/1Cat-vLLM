# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from multiprocessing import Pipe

import pytest

from vllm import envs
from vllm.config.execution_policy import LayerExecutionPolicy

ALIAS = "VLLM_SM70_GLM53_MHC_FUSED_POST_DOT_Q8"


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.delenv(ALIAS, raising=False)
    envs.disable_envs_cache()
    yield
    envs.disable_envs_cache()


def test_default_is_resolved_and_hashed():
    automatic = LayerExecutionPolicy()
    automatic.resolve()
    explicit = LayerExecutionPolicy(mhc_fused_post_dot=True)
    explicit.resolve()
    disabled = LayerExecutionPolicy(mhc_fused_post_dot=False)
    disabled.resolve()
    assert automatic.mhc_fused_post_dot is True
    assert automatic.compute_hash() == explicit.compute_hash()
    assert automatic.compute_hash() != disabled.compute_hash()


@pytest.mark.parametrize("enabled", [False, True])
def test_legacy_typed_and_worker_equivalence(monkeypatch, enabled):
    monkeypatch.setenv(ALIAS, str(int(enabled)))
    legacy = LayerExecutionPolicy()
    legacy.resolve()
    monkeypatch.setenv(ALIAS, str(int(not enabled)))
    typed = LayerExecutionPolicy(mhc_fused_post_dot=enabled)
    typed.resolve()
    assert typed.mhc_fused_post_dot is legacy.mhc_fused_post_dot is enabled
    assert typed.compute_hash() == legacy.compute_hash()
    sender, receiver = Pipe()
    try:
        sender.send(legacy)
        worker = receiver.recv()
    finally:
        sender.close()
        receiver.close()
    worker.resolve()
    assert worker.compute_hash() == legacy.compute_hash()
    assert worker.mhc_fused_post_dot is enabled
