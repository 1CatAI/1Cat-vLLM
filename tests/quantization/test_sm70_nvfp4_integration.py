# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from vllm import _sm70_ops as ops
from vllm.model_executor.layers.quantization import nvfp4_sm70_moe as moe
from vllm.model_executor.models import qwen2_moe


def test_raw_scale_shared_workspace_rejects_microbatching(monkeypatch):
    monkeypatch.setattr(
        moe,
        "get_current_vllm_config_or_none",
        lambda: NS(parallel_config=NS(use_ubatching=True)),
    )
    allocate = Mock(side_effect=AssertionError("must reject before allocating"))
    monkeypatch.setattr(torch, "empty", allocate)
    with pytest.raises(NotImplementedError, match="DBO or microbatching"):
        moe._get_qwen38_raw_scale_workspace(torch.device("cuda", 0))
    allocate.assert_not_called()


@pytest.mark.parametrize("raw_scale", [False, True])
@pytest.mark.parametrize("fused", [False, True])
def test_m1_fusions_never_consume_raw_scale_workspace(monkeypatch, raw_scale, fused):
    # Execute host dispatch on CPU tensors; every GPU operator is intercepted.
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(moe, "is_exact_sm70_cuda", lambda *a, **k: True)
    for name in (
        "_use_qwen38_indexed_prefill",
        "_use_grouped_decode",
        "_use_qwen38_qpn_batch_decode",
        "_use_qwen38_qpn_mtp5_decode",
        "_use_qwen38_qpn_batch_fused_w13",
        "_use_qwen38_qpn_batch_fused_w2",
    ):
        monkeypatch.setattr(moe, name, lambda *a: False)
    monkeypatch.setattr(moe, "_use_qwen38_qpn_m1_decode", lambda *a: True)
    monkeypatch.delenv("VLLM_SM70_QWEN38_QPN_ROUTE_DEBUG", raising=False)
    layer = NS(
        sm70_nvfp4_hidden_size=2560,
        sm70_nvfp4_top_k=10,
        sm70_nvfp4_qwen38_fused_swiglu_prefill=True,
        sm70_nvfp4_qwen38_fused_swiglu_decode=fused,
        sm70_nvfp4_qwen38_w2_direct_reduce=fused,
        sm70_nvfp4_qwen38_raw_scale=raw_scale,
        w13_tm_weight=object(),
        w2_tm_weight=object(),
        w13_tm_scales=object(),
        w2_tm_scales=object(),
        w13_raw_scale_codes=object(),
        w2_raw_scale_codes=object(),
        w13_raw_global_scales=object(),
        w2_raw_global_scales=object(),
    )
    buffers = {
        key: torch.empty(1)
        for key in ("output", "intermediate", "gate_up", "sorted_output")
    }
    method = object.__new__(moe.ModelOptNvFp4SM70MoEMethod)
    method._get_buffers = Mock(return_value=buffers)
    method._apply_swiglu = Mock()
    calls = {}
    for name in (
        "nvfp4_moe_qpn_m1_sm70_out",
        "nvfp4_moe_qpn_raw_scale_sm70_out",
        "nvfp4_qwen38_w13_fused_swiglu_out",
        "nvfp4_qwen38_w2_direct_reduce_out",
    ):
        calls[name] = Mock()
        monkeypatch.setattr(ops, name, calls[name])
    reduce = Mock()
    monkeypatch.setattr(moe, "_single_token_weighted_reduce", reduce)
    result = method.apply(
        layer,
        torch.zeros(1, 2560, dtype=torch.float16),
        torch.ones(1, 10),
        torch.zeros(1, 10, dtype=torch.int32),
        None,
        None,
    )
    assert result is buffers["output"]
    use_fused = fused and not raw_scale
    assert calls["nvfp4_qwen38_w13_fused_swiglu_out"].call_count == int(use_fused)
    assert calls["nvfp4_qwen38_w2_direct_reduce_out"].call_count == int(use_fused)
    assert calls["nvfp4_moe_qpn_raw_scale_sm70_out"].call_count == (
        2 if raw_scale else 0
    )
    assert calls["nvfp4_moe_qpn_m1_sm70_out"].call_count == (
        0 if raw_scale or fused else 2
    )
    assert reduce.call_count == int(not use_fused)


@pytest.mark.parametrize("rows", [1, 4, 8, 16])
@pytest.mark.parametrize("available", [False, True])
def test_shared_gate_m1_kernel_is_not_called_for_batch(monkeypatch, rows, available):
    monkeypatch.setattr(
        qwen2_moe, "_sm70_dump_qwen_mlp_tensor", lambda label, idx, x: x
    )
    monkeypatch.setattr(ops, "has_qwen38_shared_gate_exact", lambda: available)
    exact = Mock()
    monkeypatch.setattr(ops, "qwen38_shared_gate_exact_out", exact)
    gate = Mock(return_value=(torch.ones(rows, 1, dtype=torch.float16), None))
    gate.weight = torch.zeros(1, 2560, dtype=torch.float16)
    layer = NS(
        layer_idx=0,
        _sm70_exact_shared_expert_gate=True,
        gate_up_proj=NS(forward_fused_silu_and_mul=lambda x: x),
        down_proj=lambda x: (x, None),
        expert_gate=gate,
    )
    x = torch.ones(rows, 2560, dtype=torch.float16)
    result = qwen2_moe.Qwen2MoeMLP.forward(layer, x)
    use_fused = rows == 1 and available
    assert exact.call_count == int(use_fused)
    assert gate.call_count == int(not use_fused)
    if not use_fused:
        torch.testing.assert_close(result, torch.sigmoid(torch.ones_like(x)))
