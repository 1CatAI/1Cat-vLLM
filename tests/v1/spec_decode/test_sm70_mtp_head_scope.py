# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
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
    model.prepare_sm70_draft_head = lambda: mtp.Qwen4ExpMTP.prepare_sm70_draft_head(
        model
    )
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


def test_parallel_packet_cpu_keeps_original_route(monkeypatch):
    view = object.__new__(head_ops.MTPQPN8Head)
    nn.Module.__init__(view)
    view._shortlist_size = None
    monkeypatch.setattr(
        head_ops.torch.ops._C,
        "qwen38_mtp_local_top1_sm70_out",
        lambda *args: None,
        raising=False,
    )
    assert (
        view.maybe_get_sm70_lm_head_top1_pair(torch.zeros(1, 2560, dtype=torch.float16))
        is None
    )


def test_older_extension_does_not_attempt_parallel_packet(monkeypatch):
    view = object.__new__(head_ops.MTPQPN8Head)
    nn.Module.__init__(view)
    view._shortlist_size = None
    view.shard_indices = SimpleNamespace(num_added_elements=0)
    hidden = SimpleNamespace(ndim=2, dtype=torch.float16, is_cuda=True, shape=(1, 2560))
    monkeypatch.setattr(head_ops.torch.ops, "_C", SimpleNamespace())
    assert view.maybe_get_sm70_lm_head_top1_pair(hidden) is None


@pytest.mark.parametrize("use_custom_ipc", [False, True])
def test_parallel_packet_reuses_compact_transport_without_recomputing_logits(
    monkeypatch, use_custom_ipc
):
    from unittest.mock import Mock

    from vllm.model_executor.layers import logits_processor as module

    monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(module, "_maybe_sync_top1_all_gather", lambda *args: None)
    packet = torch.tensor([[4.0, 102.0], [2.0, 100.0]])
    ids = torch.tensor([102, 100])
    head = SimpleNamespace(
        maybe_get_sm70_lm_head_top1_pair=Mock(return_value=packet),
        maybe_get_sm70_lm_head_top1=Mock(
            side_effect=AssertionError("duplicate projection")
        ),
    )
    proc = module.LogitsProcessor(256)
    proc._maybe_dump_top_token_margin = Mock()
    proc._maybe_custom_top1_argmax = Mock(return_value=ids if use_custom_ipc else None)
    gather = Mock(side_effect=lambda pair, dim: torch.cat([pair] * 4, dim=dim))
    monkeypatch.setattr(module, "tensor_model_parallel_all_gather", gather)
    assert torch.equal(proc.get_top_tokens(head, torch.zeros(2, 2560)), ids)
    assert proc._maybe_custom_top1_argmax.call_args.args[0] is packet
    assert gather.call_count == (0 if use_custom_ipc else 1)
    head.maybe_get_sm70_lm_head_top1.assert_not_called()


@pytest.mark.parametrize("scale,cap", [(2.0, None), (1.0, 2.0)])
def test_parallel_packet_preserves_logit_transform_fallback(monkeypatch, scale, cap):
    from unittest.mock import Mock

    from vllm.model_executor.layers import logits_processor as module

    monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(module, "_maybe_sync_top1_all_gather", lambda *args: None)
    monkeypatch.setattr(
        module,
        "tensor_model_parallel_all_gather",
        lambda pair, dim: torch.cat([pair] * 4, dim=dim),
    )
    head = SimpleNamespace(
        maybe_get_sm70_lm_head_top1_pair=Mock(),
        maybe_get_sm70_lm_head_top1=Mock(return_value=None),
        quant_method=SimpleNamespace(
            apply=lambda *args, **kwargs: torch.tensor([[0.0, 1.0, 3.0]])
        ),
        shard_indices=SimpleNamespace(
            num_org_vocab_padding=0, org_vocab_start_index=100
        ),
    )
    proc = module.LogitsProcessor(256, scale=scale, soft_cap=cap)
    proc._maybe_custom_top1_argmax = lambda pair: None
    proc._maybe_dump_top_token_margin = lambda *args: None
    assert proc.get_top_tokens(head, torch.zeros(1, 2560)).tolist() == [102]
    head.maybe_get_sm70_lm_head_top1_pair.assert_not_called()
