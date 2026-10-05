# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import torch
from torch import nn

from vllm.models.qwen4_exp.nvidia import sm70_mtp_head as head_ops


def test_missing_pipeline_head_skips_draft_preparation():
    assert head_ops.prepare_mtp_qpn8_head(nn.Module()) is None


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


def test_shortlist_keeps_global_ids_and_excludes_padding(monkeypatch):
    shared = nn.Module()
    shared.weight = nn.Parameter(torch.zeros(32, 2560, dtype=torch.float16))
    shared.weight.data[:, 0] = -torch.arange(1, 33)
    shared.shard_indices = SimpleNamespace(
        num_org_vocab_padding=0, org_vocab_start_index=100, org_vocab_end_index=132
    )
    shared.quant_method = SimpleNamespace(
        apply=lambda layer, x, bias=None: torch.nn.functional.linear(x, layer.weight)
    )
    monkeypatch.setattr(
        head_ops, "prepare_channel_qpn8_weight", lambda w: (w.clone(), torch.ones(32))
    )

    def gemm(out, x, codes, *args):
        out.copy_(torch.nn.functional.linear(x, codes))

    monkeypatch.setattr(head_ops.ops, "fp8_qpn8_gemm_sm70_out", gemm)
    view = head_ops.MTPQPN8Head(shared)
    view.prepare_shortlist([131, 106, 103, 106, 999])
    assert view._shortlist_size == 3
    x = torch.zeros(5, 2560, dtype=torch.float16)
    x[:, 0] = 1
    values, ids = view.maybe_get_sm70_lm_head_top1(x)
    assert ids.tolist() == [103] * 5
    assert values.tolist() == [-4] * 5
    # Full-vocabulary diagnostic logits still include the original best ID 100.
    assert view.apply(view, x).argmax(dim=-1).tolist() == [0] * 5
    view.prepare_shortlist([999])
    values, ids = view.maybe_get_sm70_lm_head_top1(x)
    assert torch.isneginf(values).all()
    assert ids.tolist() == [100] * 5
