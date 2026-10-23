# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.config.execution_policy import LayerExecutionPolicy
from vllm.config.sm70_dflash2 import Sm70DFlash2Config
from vllm.config.sm70_runtime import RuntimeTraceConfig
from vllm.model_executor.kernels.lm_head import sm70 as vocab


def make_state(weight, *, fp32, rerank=False):
    return vocab.Sm70LMHeadState(
        weight,
        vocab.VocabShard(),
        is_lm_head=True,
        policy=LayerExecutionPolicy(
            lm_head_dense=False, lm_head_top1=False, lm_head_top1_tc=False
        ),
        dflash=Sm70DFlash2Config(
            fp32_logits=fp32, qpn8_rerank=rerank, qpn8_rerank_shadow=False
        ),
        trace=RuntimeTraceConfig(profile_trace=False, greedy_token_trace=False),
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_explicit_fp32_flag_independent_of_other_head_fastpaths(monkeypatch, enabled):
    state = make_state(
        SimpleNamespace(
            dtype=torch.float16,
            is_cuda=True,
            device=torch.device("cuda", 0),
            ndim=2,
            shape=(124160, 5120),
        ),
        fp32=enabled,
    )
    monkeypatch.setattr(vocab.current_platform, "is_cuda_alike", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (7, 0))
    assert vocab._is_sm70_lm_head_fastpath_eligible(state) == enabled


@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize("enabled", [False, True])
def test_fp32_head_admission_without_qpn8_layout(monkeypatch, tp, enabled):
    state = make_state(torch.empty((248320 // tp, 5120), device="meta"), fp32=enabled)
    monkeypatch.setattr(vocab, "_is_sm70_lm_head_fastpath_eligible", lambda _: True)
    assert vocab.maybe_prepare_sm70_lm_head_top1(state)
    assert getattr(state, "_sm70_dflash2_fp32_logits", False) == enabled
    assert not getattr(state, "_sm70_dflash2_qpn8_rerank_prepared", False)
    assert not getattr(state, "_sm70_f16_prepared", False)


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
    state = make_state(SimpleNamespace(shape=(rows, hidden)), fp32=fp32, rerank=True)
    for name in (
        "fp8_qpn8_prepare_sm70",
        "fp8_qpn8_gemm_sm70_out",
        "sm70_f16_indexed_rerank_packed_out",
        "sm70_f16_rerank_keys_out",
        "sm70_f16_rerank_topk_out",
    ):
        monkeypatch.setattr(torch.ops._C, name, object(), raising=False)
    assert vocab._is_sm70_dflash2_qpn8_rerank_eligible(state) == expected
    state.shard_indices = vocab.VocabShard(num_org_vocab_padding=1)
    assert not vocab._is_sm70_dflash2_qpn8_rerank_eligible(state)
