# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU route-boundary checks for the upstream SM70 GDN verifier projection."""

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
