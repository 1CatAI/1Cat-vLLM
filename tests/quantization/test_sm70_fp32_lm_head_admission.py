# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers import vocab_parallel_embedding as vocab


@pytest.mark.parametrize("enabled", [False, True])
def test_explicit_fp32_flag_independent_of_other_head_fastpaths(monkeypatch, enabled):
    layer = SimpleNamespace(
        prefix="language_model.lm_head",
        weight=SimpleNamespace(
            dtype=torch.float16,
            is_cuda=True,
            device=torch.device("cuda", 0),
            ndim=2,
            shape=(124160, 5120),
        ),
    )
    monkeypatch.setattr(vocab, "_sm70_env_bool", lambda *args: False)
    monkeypatch.setattr(vocab, "_sm70_dflash2_qpn8_rerank_requested", lambda: False)
    monkeypatch.setattr(vocab.current_platform, "is_cuda_alike", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (7, 0))
    monkeypatch.setattr(vocab.envs, "VLLM_SM70_DFLASH2_FP32_LOGITS", enabled)
    assert vocab._is_sm70_lm_head_fastpath_eligible(layer) == enabled


@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize("enabled", [False, True])
def test_fp32_head_admission_without_qpn8_layout(monkeypatch, tp, enabled):
    layer = SimpleNamespace(
        tp_size=tp,
        weight=torch.empty((248320 // tp, 5120), device="meta"),
    )
    monkeypatch.setattr(vocab, "_is_sm70_lm_head_fastpath_eligible", lambda _: True)
    monkeypatch.setattr(
        vocab, "_sm70_lm_head_packed_layout_requested", lambda *args: False
    )
    monkeypatch.setattr(vocab.envs, "VLLM_SM70_DFLASH2_FP32_LOGITS", enabled)
    assert vocab.maybe_prepare_sm70_lm_head_top1(layer)
    assert getattr(layer, "_sm70_dflash2_fp32_logits", False) == enabled
    assert not getattr(layer, "_sm70_dflash2_qpn8_rerank_prepared", False)


@pytest.mark.parametrize(
    "rows,hidden,fp32,expected",
    [
        (248320, 5120, True, True),
        (124160, 5120, True, True),
        (62080, 5120, False, True),
        (124160, 5120, False, False),
        (62080, 4096, True, True),
        (62080, 4096, False, False),
        (63, 5120, True, False),
        (62081, 5120, True, False),
    ],
)
def test_rerank_local_layout_contract(monkeypatch, rows, hidden, fp32, expected):
    layer = SimpleNamespace(
        weight=SimpleNamespace(shape=(rows, hidden)),
        shard_indices=SimpleNamespace(num_org_vocab_padding=0),
    )
    monkeypatch.setattr(vocab, "_sm70_dflash2_qpn8_rerank_requested", lambda: True)
    monkeypatch.setattr(vocab.envs, "VLLM_SM70_DFLASH2_FP32_LOGITS", fp32)
    assert vocab._is_sm70_dflash2_qpn8_rerank_eligible(layer) == expected
    layer.shard_indices.num_org_vocab_padding = 1
    assert not vocab._is_sm70_dflash2_qpn8_rerank_eligible(layer)
