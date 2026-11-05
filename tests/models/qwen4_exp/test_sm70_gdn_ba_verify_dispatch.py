# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU route-boundary checks for the upstream SM70 GDN verifier projection."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn
from vllm.model_executor.layers.mamba.gdn import sm70_verify_dispatch as dispatch
from vllm.model_executor.layers.quantization import sm70_gdn_ba_verify


def test_ba_verify_dispatch_uses_upstream_helper_first(monkeypatch):
    layer = object()
    hidden_states = torch.empty((1, 0), dtype=torch.float16)
    outputs = tuple(torch.empty((0,)) for _ in range(4))
    calls = []

    monkeypatch.setattr(gdn, "_sm70_gdn_projection_dump_requested", lambda _: False)

    def apply(layer_arg, hidden_arg):
        calls.append((layer_arg, hidden_arg))
        return outputs

    monkeypatch.setattr(sm70_gdn_ba_verify, "apply_gdn_ba_verify", apply)

    result = dispatch.try_projection(gdn, layer, hidden_states, "layer.0")

    assert result is outputs
    assert len(calls) == 1
    assert calls[0][0] is layer
    assert calls[0][1] is hidden_states


def test_ba_verify_dispatch_preserves_fallback_when_upstream_declines(monkeypatch):
    calls = []
    monkeypatch.setattr(gdn, "_sm70_gdn_projection_dump_requested", lambda _: False)

    def decline(*_):
        calls.append(True)
        return None

    monkeypatch.setattr(sm70_gdn_ba_verify, "apply_gdn_ba_verify", decline)

    assert (
        dispatch.try_projection(
            gdn, object(), torch.empty((1, 0), dtype=torch.float16), "layer.0"
        )
        is None
    )
    assert calls == [True]


def test_ba_verify_dispatch_keeps_projection_dump_on_eager_route(monkeypatch):
    monkeypatch.setattr(gdn, "_sm70_gdn_projection_dump_requested", lambda _: True)

    def unexpected_call(*_):
        raise AssertionError("upstream fusion must not bypass projection dumps")

    monkeypatch.setattr(sm70_gdn_ba_verify, "apply_gdn_ba_verify", unexpected_call)

    assert (
        dispatch.try_projection(
            gdn, object(), torch.empty((1, 0), dtype=torch.float16), "layer.0"
        )
        is None
    )


@pytest.mark.parametrize("rows", [7, 8, 9, 16, 20, 32, 33])
@pytest.mark.parametrize("preallocated", [False, True])
def test_ba_verify_dispatch_preserves_upstream_concurrent_rows(
    monkeypatch, rows, preallocated
):
    """Exercise retained dispatch plus the real upstream guard with meta tensors."""
    from vllm.model_executor.layers.linear import LinearBase

    qkvz = LinearBase(5120, 4096, disable_tp=True)
    qkvz.sm70_fp8_qpn8 = True
    qkvz.weight = torch.empty((5120, 4096), device="meta")
    qkvz.weight_scale_inv = torch.empty((1, 4096), device="meta")
    ba = LinearBase(5120, 24, disable_tp=True)
    ba.weight = torch.empty((24, 5120), dtype=torch.float16, device="meta")
    layer = SimpleNamespace(
        in_proj_qkvz=qkvz,
        in_proj_ba=ba,
        enable_sm70_dflash2_fused_gdn_verify=True,
        enable_sm70_gdn_ba_verify=True,
        gqa_interleaved_layout=False,
        disable_tp_for_ba_proj=False,
    )
    x = torch.empty((rows, 5120), dtype=torch.float16, device="meta")
    monkeypatch.setattr(gdn, "_sm70_gdn_projection_dump_requested", lambda _: False)
    monkeypatch.setattr(
        sm70_gdn_ba_verify.current_platform,
        "is_device_capability",
        lambda capability: capability == 70,
    )
    calls = []
    monkeypatch.setattr(
        torch.ops._C,
        "fp8_qpn8_dispatch_ba_split_sm70_out",
        lambda *args: calls.append(args),
        raising=False,
    )
    shapes = ((rows, 2560), (rows, 1536), (rows, 12), (rows, 12))
    outputs = tuple(x.new_empty(shape) for shape in shapes)
    if preallocated:
        result = sm70_gdn_ba_verify.apply_gdn_ba_verify(layer, x, outputs)
    else:
        result = dispatch.try_projection(gdn, layer, x, "layer.0")
    if not 8 <= rows <= 32:
        assert result is None
        assert calls == []
        return
    assert len(calls) == 1
    assert tuple(t.shape for t in result) == shapes
    if preallocated:
        assert all(actual is expected for actual, expected in zip(result, outputs))
    assert calls[0][4].shape == (rows, 4096)
    assert calls[0][5].shape == (rows, 24)
    assert calls[0][7] is x
