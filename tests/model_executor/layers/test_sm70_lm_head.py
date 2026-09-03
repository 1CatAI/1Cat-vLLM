# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm.model_executor.layers.vocab_parallel_embedding as vocab_embedding
from vllm import envs

pytestmark = pytest.mark.skip_global_cleanup


class _FakeCudaTensor:
    def __init__(self, shape, dtype=torch.float16):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = torch.device("cuda:0")
        self.is_cuda = True

    def reshape(self, *shape):
        if shape and shape[0] == -1:
            shape = (self.shape[0], *shape[1:])
        self.shape = tuple(shape)
        return self

    def size(self, dim):
        return self.shape[dim]

    def is_contiguous(self):
        return True

    def stride(self, dim):
        return 1 if dim == 1 else self.shape[1]


def _set_lm_head_routes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    raw_top1: bool = False,
    tc_top1: bool = False,
    dense: bool = False,
    qpn8: bool = False,
) -> None:
    monkeypatch.setenv("VLLM_SM70_LM_HEAD_TOP1", str(int(raw_top1)))
    monkeypatch.setenv("VLLM_SM70_LM_HEAD_TOP1_TC", str(int(tc_top1)))
    monkeypatch.setenv("VLLM_SM70_ENABLE_LM_HEAD_FASTPATH", str(int(dense)))
    monkeypatch.setattr(envs, "VLLM_SM70_DFLASH2_QPN8_RERANK", qpn8)
    monkeypatch.setattr(envs, "VLLM_SM70_DFLASH2_QPN8_RERANK_SHADOW", False)


def test_raw_top1_does_not_prepare_packed_lm_head(monkeypatch) -> None:
    _set_lm_head_routes(monkeypatch, raw_top1=True)
    monkeypatch.setattr(
        vocab_embedding,
        "_is_sm70_lm_head_fastpath_eligible",
        lambda _layer: True,
    )
    prepare = Mock(side_effect=AssertionError("raw top1 must not pack the LM head"))
    monkeypatch.setattr(vocab_embedding.sm70_ops, "sm70_f16_prepare", prepare)
    layer = SimpleNamespace(weight=object())

    assert vocab_embedding.maybe_prepare_sm70_lm_head_top1(layer)
    assert layer._sm70_f16_raw_top1_ready
    assert not hasattr(layer, "_sm70_f16_prepared")
    assert not hasattr(layer, "_sm70_f16_tm_weight")
    prepare.assert_not_called()


def test_raw_top1_dispatch_uses_original_weight(monkeypatch) -> None:
    _set_lm_head_routes(monkeypatch, raw_top1=True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (7, 0))
    monkeypatch.setattr(
        torch.ops._C,
        "sm70_f16_lm_head_top1_out",
        Mock(),
        raising=False,
    )
    top1_out = Mock()
    monkeypatch.setattr(
        vocab_embedding.sm70_ops,
        "sm70_f16_lm_head_top1_out",
        top1_out,
    )
    monkeypatch.setattr(
        vocab_embedding.torch,
        "empty",
        lambda shape, dtype, device: _FakeCudaTensor(shape, dtype),
    )
    weight = _FakeCudaTensor((64, 16))
    layer = SimpleNamespace(
        weight=weight,
        _sm70_f16_raw_top1_ready=True,
        shard_indices=SimpleNamespace(
            org_vocab_start_index=32,
            num_org_vocab_padding=0,
        ),
    )

    result = vocab_embedding._maybe_sm70_lm_head_top1(
        layer,
        _FakeCudaTensor((1, 16)),
    )

    assert result is not None
    top1_out.assert_called_once()
    assert top1_out.call_args.args[3] is weight


@pytest.mark.parametrize(
    ("route", "tc_top1", "dense", "qpn8"),
    [
        ("tc_top1", True, False, False),
        ("dense", False, True, False),
        ("qpn8", False, False, True),
    ],
)
def test_packed_lm_head_routes_still_prepare_layout(
    monkeypatch,
    route,
    tc_top1,
    dense,
    qpn8,
) -> None:
    _set_lm_head_routes(
        monkeypatch,
        tc_top1=tc_top1,
        dense=dense,
        qpn8=qpn8,
    )
    monkeypatch.setattr(
        vocab_embedding,
        "_is_sm70_lm_head_fastpath_eligible",
        lambda _layer: True,
    )
    packed_weight = object()
    prepare = Mock(return_value=(packed_weight, torch.tensor([32])))
    monkeypatch.setattr(torch.ops._C, "sm70_f16_prepare", Mock(), raising=False)
    monkeypatch.setattr(vocab_embedding.sm70_ops, "sm70_f16_prepare", prepare)
    prepare_qpn8 = Mock(return_value=qpn8)
    monkeypatch.setattr(
        vocab_embedding,
        "_prepare_sm70_dflash2_qpn8_rerank",
        prepare_qpn8,
    )
    layer = SimpleNamespace(weight=object())

    assert vocab_embedding.maybe_prepare_sm70_lm_head_top1(layer), route
    assert layer._sm70_f16_prepared
    assert layer._sm70_f16_tm_weight is packed_weight
    assert layer._sm70_f16_k_ld == 32
    prepare.assert_called_once_with(layer.weight)
    prepare_qpn8.assert_called_once_with(layer)


def test_raw_top1_readiness_does_not_enable_dense_fastpath(monkeypatch) -> None:
    _set_lm_head_routes(monkeypatch, raw_top1=True, dense=True)
    layer = SimpleNamespace(_sm70_f16_raw_top1_ready=True)

    assert (
        vocab_embedding._maybe_sm70_lm_head_forward(
            layer,
            torch.empty((1, 16), dtype=torch.float16),
        )
        is None
    )


def test_disabled_lm_head_routes_prepare_nothing(monkeypatch) -> None:
    _set_lm_head_routes(monkeypatch)
    layer = SimpleNamespace(weight=object())

    assert not vocab_embedding.maybe_prepare_sm70_lm_head_top1(layer)
    assert not hasattr(layer, "_sm70_f16_raw_top1_ready")
    assert not hasattr(layer, "_sm70_f16_prepared")
