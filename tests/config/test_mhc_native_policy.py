# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from multiprocessing import Pipe

import pytest

from vllm import envs
from vllm.config.execution_policy import LayerExecutionPolicy


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for name in ("VLLM_SM70_GLM53_MHC_NATIVE_VERIFY", "VLLM_SM70_GLM_MHC_PRE_THREADS"):
        monkeypatch.delenv(name, raising=False)
    envs.disable_envs_cache()
    yield
    envs.disable_envs_cache()


def test_defaults_are_resolved_and_hashed():
    automatic = LayerExecutionPolicy()
    automatic.resolve()
    explicit = LayerExecutionPolicy(mhc_native_verify=True, mhc_pre_threads=1024)
    explicit.resolve()
    disabled = LayerExecutionPolicy(mhc_native_verify=False, mhc_pre_threads=256)
    disabled.resolve()
    assert automatic.mhc_native_verify is True
    assert automatic.mhc_pre_threads == 1024
    assert automatic.compute_hash() == explicit.compute_hash()
    assert automatic.compute_hash() != disabled.compute_hash()


@pytest.mark.parametrize("enabled,threads", [(False, 256), (True, 512), (True, 1024)])
def test_explicit_inputs_survive_worker_transfer(monkeypatch, enabled, threads):
    monkeypatch.setenv("VLLM_SM70_GLM53_MHC_NATIVE_VERIFY", str(int(enabled)))
    monkeypatch.setenv("VLLM_SM70_GLM_MHC_PRE_THREADS", str(threads))
    legacy = LayerExecutionPolicy()
    legacy.resolve()
    monkeypatch.setenv("VLLM_SM70_GLM53_MHC_NATIVE_VERIFY", str(int(not enabled)))
    monkeypatch.setenv("VLLM_SM70_GLM_MHC_PRE_THREADS", "128")
    explicit = LayerExecutionPolicy(mhc_native_verify=enabled, mhc_pre_threads=threads)
    explicit.resolve()
    sender, receiver = Pipe()
    try:
        sender.send(legacy)
        worker = receiver.recv()
    finally:
        sender.close()
        receiver.close()
    worker.resolve()
    assert worker.mhc_native_verify is enabled
    assert worker.mhc_pre_threads == threads
    assert worker.compute_hash() == explicit.compute_hash()
