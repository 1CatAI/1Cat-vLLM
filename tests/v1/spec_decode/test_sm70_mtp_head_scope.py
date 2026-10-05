# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import torch
from torch import nn

from vllm.models.qwen4_exp.nvidia import sm70_mtp_head as head_ops


def test_draft_head_preserves_shared_target_and_uses_actual_probe_path(monkeypatch):
    calls = []

    class ReferenceMethod:
        def apply(self, layer, x, bias=None):
            calls.append("reference")
            return torch.nn.functional.linear(x, layer.weight, bias)

    shared = nn.Module()
    shared.weight = nn.Parameter(torch.randn(32, 2560, dtype=torch.float16))
    shared.quant_method = ReferenceMethod()
    shared.shard_indices = SimpleNamespace(num_org_vocab_padding=0)
    original = shared.weight.clone()
    method = shared.quant_method
    monkeypatch.setattr(
        head_ops,
        "prepare_channel_qpn8_weight",
        lambda w: (torch.zeros_like(w, dtype=torch.uint8), torch.ones(32)),
    )

    def gemm(out, x, *args):
        calls.append("qpn8")
        out.copy_(torch.nn.functional.linear(x, shared.weight))

    monkeypatch.setattr(head_ops.ops, "fp8_qpn8_gemm_sm70_out", gemm)
    view = head_ops.MTPQPN8Head(shared)
    assert view.weight is shared.weight
    assert shared.quant_method is method
    assert list(view.modules()) == [view, shared]  # No self-referential method child.
    for rows in (1, 4, 8):
        x = torch.randn(rows, 2560, dtype=torch.float16)
        assert torch.equal(view.quant_method.apply(view, x), method.apply(shared, x))
        assert view.maybe_get_sm70_lm_head_top1(x) is None
    assert calls == ["qpn8", "reference"] * 3
    view.apply(view, torch.randn(9, 2560, dtype=torch.float16))
    assert calls[-1] == "reference"
    assert torch.equal(shared.weight, original)


def test_default_head_preparation_precedes_graph_mode_guard(monkeypatch):
    from vllm.models.qwen4_exp.nvidia import mtp

    target_head, draft_view = object(), object()
    model = SimpleNamespace(_sm70_draft_head=None, lm_head=target_head)
    seen = []

    def prepare_head(head):
        seen.append(head)
        return draft_view

    monkeypatch.setattr(head_ops, "prepare_mtp_qpn8_head", prepare_head)
    monkeypatch.setattr(mtp.envs, "VLLM_SM70_QWEN38_DUAL_COMPILE", False)
    prepare = mtp.Qwen4ExpMTP.prepare_sm70_decode_graph_model
    assert not prepare(model)
    assert not prepare(model)
    assert seen == [target_head]
    assert model._sm70_draft_head is draft_view
