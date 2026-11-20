# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The QSA host-KV resolver normalizes GLM5next selector metadata."""

from types import SimpleNamespace as NS

import pytest
import torch

from vllm import platforms
from vllm.config import KernelConfig
from vllm.models.qwen4_exp.common.kv_policy import resolve_qsa_host_kv


@pytest.fixture(autouse=True)
def sm70_platform(monkeypatch):
    monkeypatch.setattr(
        platforms,
        "current_platform",
        NS(
            is_cuda=lambda: True,
            is_device_capability=lambda capability, device_id=0: capability == 70,
        ),
    )


def _config(hf, *, kv_heads=1, cache_dtype="auto", qsa_host_kv=True):
    model = NS(
        hf_text_config=hf,
        dtype=torch.float16,
        get_num_kv_heads=lambda parallel_config: kv_heads,
    )
    return NS(
        kernel_config=KernelConfig(qsa_host_kv=qsa_host_kv),
        model_config=model,
        parallel_config=NS(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            tensor_parallel_size=1,
        ),
        cache_config=NS(cache_dtype=cache_dtype, kv_offloading_size=None),
        kv_transfer_config=None,
    )


def test_glm5_metadata_is_admitted():
    # GLM5next: index_* selector names, qk_nope_head_dim instead of head_dim,
    # and latent (MLA) KV that is one shared head per rank regardless of the
    # raw num_key_value_heads/TP ratio.
    hf = NS(
        index_n_heads=8,
        index_kpool=4,
        index_topk=64,
        head_dim=0,
        qk_nope_head_dim=256,
        kv_lora_rank=512,
    )
    cfg = _config(hf, kv_heads=2)
    assert resolve_qsa_host_kv(cfg) is True
    assert cfg.kernel_config.qsa_host_kv_reason is None


def test_qwen4_metadata_still_admitted():
    hf = NS(
        indexer_n_heads=8,
        indexer_compress_ratio=4,
        indexer_budget=64,
        head_dim=256,
    )
    cfg = _config(hf, kv_heads=1)
    assert resolve_qsa_host_kv(cfg) is True
    assert cfg.kernel_config.qsa_host_kv_reason is None


def test_multi_head_non_mla_kv_is_rejected():
    hf = NS(
        indexer_n_heads=8,
        indexer_compress_ratio=4,
        indexer_budget=64,
        head_dim=256,
    )
    cfg = _config(hf, kv_heads=4)
    assert resolve_qsa_host_kv(cfg) is False
    assert cfg.kernel_config.qsa_host_kv_reason == (
        "host writer requires one TP-local KV head"
    )


def test_missing_selector_metadata_is_rejected():
    cfg = _config(NS(head_dim=256))
    assert resolve_qsa_host_kv(cfg) is False
    assert cfg.kernel_config.qsa_host_kv_reason == "no QSA selector metadata"


def test_disabled_policy_is_rejected():
    hf = NS(
        indexer_n_heads=8,
        indexer_compress_ratio=4,
        indexer_budget=64,
        head_dim=256,
    )
    cfg = _config(hf, qsa_host_kv=False)
    assert resolve_qsa_host_kv(cfg) is False
    assert cfg.kernel_config.qsa_host_kv_reason == "disabled by KernelConfig"


def test_wrong_glm5_head_dim_falls_back_and_rejects():
    hf = NS(
        index_n_heads=8,
        index_kpool=4,
        index_topk=64,
        head_dim=0,
        qk_nope_head_dim=128,
        kv_lora_rank=512,
    )
    cfg = _config(hf, kv_heads=2)
    assert resolve_qsa_host_kv(cfg) is False
    assert cfg.kernel_config.qsa_host_kv_reason == "host staging requires D256"
