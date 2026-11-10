# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thread policy lifecycle and the bounded original-prefill tensor contract."""

import multiprocessing
from types import SimpleNamespace as NS

import pytest
import torch

from vllm.config import gdn as gdn_config
from vllm.config.gdn import GdnConfig
from vllm.model_executor.layers.fla.ops.sm70 import gdn_prefill


def test_policy_snapshot_hash_and_inactive_backend(monkeypatch):
    monkeypatch.setattr(gdn_config, "_tilelang_nvcc_available", lambda: True)
    policies = [GdnConfig(original_prefill_threads=value) for value in (None, 128, 256)]
    for p in policies:
        p.resolve()
        p.active_prefill_backend = "flashqla_sm70"
    assert policies[0].compute_hash() == policies[1].compute_hash()
    assert policies[1].compute_hash() != policies[2].compute_hash()
    recv, send = multiprocessing.Pipe(duplex=False)
    try:
        send.send(policies[2])
        restored = recv.recv()
    finally:
        send.close()
        recv.close()
    monkeypatch.setenv("VLLM_SM70_FLASHQLA_ORIGINAL_PREFILL", "0")
    restored.resolve()
    assert restored.original_prefill and restored.original_prefill_threads == 256
    assert restored.sources["original_prefill_threads"] == "typed"
    for p in policies:
        p.active_prefill_backend = "triton"
    assert policies[1].compute_hash() == policies[2].compute_hash()
    for p in policies:
        p.active_prefill_backend = "flashqla_sm70"
        p.original_prefill = False
    assert policies[1].compute_hash() == policies[2].compute_hash()
    for p in policies:
        p.active_prefill_backend = None
    assert policies[1].compute_hash() == policies[2].compute_hash()


@pytest.mark.parametrize("value", [0, 64, 512])
def test_invalid_launch_is_rejected_before_worker_start(value):
    with pytest.raises(ValueError):
        GdnConfig(original_prefill_threads=value)


@pytest.mark.parametrize("rows", [1, 4095, 4096, 4097, 6108, 7344, 8192, 8193])
def test_qualified_query_chunk_and_rollback(monkeypatch, rows):
    monkeypatch.setattr(
        gdn_prefill.current_platform, "is_device_capability", lambda c: c == 70
    )
    q = NS(shape=(1, rows, 4, 128), dtype=torch.float16, is_cuda=True)
    v = NS(shape=(1, rows, 12, 128))
    state = NS(dtype=torch.float32, shape=(1, 12, 128, 128))
    gate = NS(dtype=torch.float32)
    indices = torch.tensor([2])
    choose = gdn_prefill.select_original_prefill_threads
    assert choose(q, v, state, indices, 256, gate, gate) == (
        256 if 4096 <= rows <= 8192 else 128
    )
    assert choose(q, v, state, indices, 128, gate, gate) == 128
    assert choose(q, v, state, None, 256, gate, gate) == 128
    cu = torch.tensor([0, rows], dtype=torch.int32)
    assert choose(q, v, state, None, 256, gate, gate, cu) == (
        256 if 4096 <= rows <= 8192 else 128
    )
    assert choose(q, v, state, None, 256, gate, gate, cu.repeat(2)) == 128
    state.shape = (2, 12, 128, 128)
    assert choose(q, v, state, None, 256, gate, gate, cu) == 128
    assert choose(q, v, state, indices.repeat(2), 256, gate, gate) == 128


@pytest.mark.parametrize(
    "change", ["cpu", "q_heads", "v_heads", "state", "gate", "device"]
)
def test_unqualified_family_tensor_uses_original_core(monkeypatch, change):
    monkeypatch.setattr(
        gdn_prefill.current_platform,
        "is_device_capability",
        lambda c: change != "device",
    )
    q = NS(shape=(1, 7344, 4, 128), dtype=torch.float16, is_cuda=change != "cpu")
    v = NS(shape=(1, 7344, 12, 128))
    state = NS(dtype=torch.float16 if change == "state" else torch.float32)
    gate = NS(dtype=torch.float16 if change == "gate" else torch.float32)
    if change == "q_heads":
        q.shape = (1, 7344, 8, 128)
    if change == "v_heads":
        v.shape = (1, 7344, 24, 128)
    assert (
        gdn_prefill.select_original_prefill_threads(
            q, v, state, torch.tensor([2]), 256, gate, gate
        )
        == 128
    )


@pytest.mark.parametrize("toolkit", ["nvcc", "runtime_only", "missing"])
def test_original_prefill_needs_a_cuda_toolkit(monkeypatch, tmp_path, toolkit):
    """A bare wheel install has no nvcc for TileLang: use the prebuilt prefill."""
    home = tmp_path / "cuda"
    if toolkit != "missing":
        (home / "lib64").mkdir(parents=True)
    if toolkit == "nvcc":
        (home / "bin").mkdir()
        (home / "bin" / "nvcc").write_text("")
    monkeypatch.setenv("CUDA_HOME", str(home))
    for name in (
        "VLLM_SM70_FLASHQLA_ORIGINAL_PREFILL",
        "FLASH_QLA_SM70_USE_ORIGINAL_TILELANG",
    ):
        monkeypatch.delenv(name, raising=False)
    policy = GdnConfig()
    policy.resolve()
    assert policy.original_prefill is (toolkit == "nvcc")
    assert policy.sources["original_prefill"] == (
        "default" if toolkit == "nvcc" else "default:no_nvcc"
    )

    # An explicit request keeps the original path; the JIT error is then
    # the user's to resolve.
    monkeypatch.setenv("VLLM_SM70_FLASHQLA_ORIGINAL_PREFILL", "1")
    policy = GdnConfig()
    policy.resolve()
    assert policy.original_prefill is True
