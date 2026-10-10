# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.layers import sm70_draft47 as draft_ops
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
    from vllm.v1.attention.backends import fa_utils

    # This CPU load-order test does not exercise an attention backend.
    monkeypatch.setattr(fa_utils, "get_flash_attn_version", lambda: 2)
    from vllm.models.qwen4_exp.nvidia import mtp

    target_head, draft_view = object(), object()
    from vllm.config.execution_policy import GraphPolicy

    model = SimpleNamespace(
        _sm70_draft_head=None,
        lm_head=target_head,
        vllm_config=SimpleNamespace(
            kernel_config=SimpleNamespace(sm70_draft_hot_vocab=True),
            compilation_config=SimpleNamespace(runtime=GraphPolicy(dual_compile=False)),
        ),
    )
    model.prepare_sm70_draft_head = lambda: mtp.Qwen4ExpMTP.prepare_sm70_draft_head(
        model
    )
    seen = []

    def prepare_head(head, hot_vocab):
        assert hot_vocab
        seen.append(head)
        return draft_view

    monkeypatch.setattr(head_ops, "prepare_mtp_qpn8_head", prepare_head)
    monkeypatch.setenv("VLLM_SM70_QWEN38_DUAL_COMPILE", "0")
    prepare = mtp.Qwen4ExpMTP.prepare_sm70_decode_graph_model
    assert not prepare(model)
    assert not prepare(model)
    assert seen == [target_head]
    assert model._sm70_draft_head is draft_view


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


@pytest.mark.parametrize("d1a_enabled", [False, True])
@pytest.mark.parametrize("use_custom_ipc", [False, True])
def test_parallel_packet_reuses_compact_transport_without_recomputing_logits(
    monkeypatch, use_custom_ipc, d1a_enabled
):
    from unittest.mock import Mock

    from vllm.model_executor.layers import logits_processor as module

    monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(module, "_maybe_sync_top1_all_gather", lambda *args: None)
    monkeypatch.setattr(draft_ops, "enabled", lambda _: d1a_enabled)
    monkeypatch.setattr(
        draft_ops,
        "top_tokens",
        Mock(side_effect=AssertionError("upstream first")),
    )
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


def test_local_top1_precedes_d1a_full_logits_fallback(monkeypatch):
    from unittest.mock import Mock

    from vllm.model_executor.layers import logits_processor as module

    monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(module, "_maybe_sync_top1_all_gather", lambda *args: None)
    monkeypatch.setattr(draft_ops, "enabled", lambda _: True)
    monkeypatch.setattr(
        draft_ops,
        "top_tokens",
        Mock(side_effect=AssertionError("upstream first")),
    )
    monkeypatch.setattr(
        module,
        "tensor_model_parallel_all_gather",
        lambda pair, dim: torch.cat([pair] * 4, dim=dim),
    )
    head = SimpleNamespace(
        maybe_get_sm70_lm_head_top1_pair=Mock(return_value=None),
        maybe_get_sm70_lm_head_top1=Mock(
            return_value=(torch.tensor([4.0]), torch.tensor([102]))
        ),
        quant_method=SimpleNamespace(
            apply=Mock(side_effect=AssertionError("duplicate projection"))
        ),
    )
    proc = module.LogitsProcessor(256)
    proc._maybe_custom_top1_argmax = lambda pair: None
    proc._maybe_dump_top_token_margin = lambda *args: None
    assert proc.get_top_tokens(head, torch.zeros(1, 2560)).tolist() == [102]
    head.quant_method.apply.assert_not_called()


@pytest.mark.parametrize("fused_result", [None, [102]])
def test_d1a_only_receives_transformed_full_logits(monkeypatch, fused_result):
    from unittest.mock import Mock

    from vllm.model_executor.layers import logits_processor as module

    monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(module, "_maybe_sync_top1_all_gather", lambda *args: None)
    monkeypatch.setattr(draft_ops, "enabled", lambda _: True)
    fallback = Mock(
        return_value=None if fused_result is None else torch.tensor(fused_result)
    )
    monkeypatch.setattr(draft_ops, "top_tokens", fallback)
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
    proc = module.LogitsProcessor(256, scale=2.0)
    proc._maybe_custom_top1_argmax = lambda pair: None
    proc._maybe_dump_top_token_margin = lambda *args: None
    assert proc.get_top_tokens(head, torch.zeros(1, 2560)).tolist() == [102]
    torch.testing.assert_close(
        fallback.call_args.args[0], torch.tensor([[0.0, 2.0, 6.0]])
    )
    assert fallback.call_args.args[1:] == (100, 4)
    head.maybe_get_sm70_lm_head_top1_pair.assert_not_called()
