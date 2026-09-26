# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from vllm import envs
from vllm.model_executor.layers.quantization import nvfp4_sm70_moe as moe


def set_context(monkeypatch, metadata):
    ctx = NS(attn_metadata=metadata, additional_kwargs={})
    monkeypatch.setattr(moe, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(moe, "get_forward_context", lambda: ctx)
    return ctx


def test_grouped_decode_defaults_off(monkeypatch):
    monkeypatch.delenv("VLLM_SM70_NVFP4_MOE_GROUPED_DECODE", raising=False)
    envs.disable_envs_cache()
    assert not envs.VLLM_SM70_NVFP4_MOE_GROUPED_DECODE


@pytest.mark.parametrize("tokens", [1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 32, 64, 2048])
def test_runtime_shape_not_scheduler_configuration(monkeypatch, tokens):
    # No TP, model name, KV dtype, max-num-seqs or chunk-size fields needed.
    layer = NS(sm70_nvfp4_grouped_decode=True)
    set_context(monkeypatch, {"attn": NS(max_query_len=1)})
    x = torch.empty(tokens, 2560, dtype=torch.float16)
    ids = torch.empty(tokens, 10, dtype=torch.int32)
    assert moe._use_grouped_decode(layer, x, ids) == (tokens in (8, 16))
    layer.sm70_nvfp4_grouped_decode = False
    assert not moe._use_grouped_decode(layer, x, ids)


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {},
        [],
        [{"attn": NS(max_query_len=1)}],
        {"unknown": NS()},
        {"prefill": NS(max_query_len=16)},
        {"verify": NS(max_query_len=5)},
        {"attn": NS(max_query_len=1), "gdn": NS(num_prefills=1)},
        {"gdn": NS(num_prefills=0, num_decodes=2, num_decode_tokens=8)},
        {"gdn": NS(num_prefills=0, num_prefill_tokens=4)},
        {"attn": NS(max_query_len=torch.tensor(1))},
        {"gdn": NS(num_decodes=8, num_decode_tokens=torch.tensor(8))},
    ],
)
def test_metadata_fails_closed(monkeypatch, metadata):
    set_context(monkeypatch, metadata)
    assert not moe._grouped_decode_context_ok()


def test_no_context_falls_back(monkeypatch):
    monkeypatch.setattr(moe, "is_forward_context_available", lambda: False)
    assert not moe._grouped_decode_context_ok()


@pytest.mark.parametrize("batch_reduce", [False, True])
def test_grouped_ops_have_fake_implementations(batch_reduce):
    from torch._subclasses.fake_tensor import FakeTensorMode

    from vllm import _sm70_ops as ops

    if not ops.has_nvfp4_grouped_decode_dispatch():
        pytest.skip("Build native grouped-decode ops first")
    if batch_reduce and not ops.has_nvfp4_grouped_batch_reduce_dispatch():
        pytest.skip("Build native grouped batch reduction first")
    with FakeTensorMode():
        x = torch.empty(16, 2560, dtype=torch.float16)
        mid = torch.empty(160, 160, dtype=torch.float16)
        routed = torch.empty(160, 2560, dtype=torch.float16)
        ids = torch.empty(160, dtype=torch.int32)
        rows = torch.empty(160, 8, dtype=torch.int32)
        experts, sizes = torch.empty_like(ids), torch.empty_like(ids)
        total = torch.empty(1, dtype=torch.int32)
        w13 = torch.empty(512, 2560, 40, dtype=torch.int32)
        s13 = torch.empty(512, 160, 320, dtype=torch.float16)
        ops.nvfp4_grouped_w13_sm70_out(
            mid, x, w13, s13, ids, rows, experts, sizes, total, 8, True
        )
        w2_op = (
            ops.nvfp4_grouped_w2_batch_reduce_sm70_out
            if batch_reduce
            else ops.nvfp4_grouped_w2_sm70_out
        )
        w2_op(
            torch.empty_like(x),
            routed,
            mid,
            torch.empty(512, 160, 320, dtype=torch.int32),
            torch.empty(512, 10, 2560, dtype=torch.float16),
            torch.empty(16, 10),
            rows,
            experts,
            sizes,
            total,
        )


def test_cached_decision_is_per_forward(monkeypatch):
    old = set_context(
        monkeypatch, {"gdn": NS(num_prefills=0, num_decodes=16, num_decode_tokens=16)}
    )
    assert moe._grouped_decode_context_ok()
    assert old.additional_kwargs["sm70_grouped_moe_decode"]
    new = set_context(monkeypatch, {"attn": NS(max_query_len=32)})
    assert not moe._grouped_decode_context_ok()
    assert not new.additional_kwargs["sm70_grouped_moe_decode"]


@pytest.mark.parametrize("bad", ["dtype", "ids", "strides", "hidden"])
def test_bad_tensor_contract_falls_back(monkeypatch, bad):
    set_context(monkeypatch, {"attn": NS(max_query_len=1)})
    x = torch.empty(16, 2560, dtype=torch.float16)
    ids = torch.empty(16, 10, dtype=torch.int32)
    if bad == "dtype":
        x = x.float()
    elif bad == "ids":
        ids = ids.long()
    elif bad == "hidden":
        x = x[:, :-1]
    else:
        x = torch.empty(2560, 16, dtype=torch.float16).t()
    assert not moe._use_grouped_decode(NS(sm70_nvfp4_grouped_decode=True), x, ids)


@pytest.mark.parametrize("tokens,split", [(8, 4), (16, 1)])
def test_grouped_apply_preserves_direct_k_split(monkeypatch, tokens, split):
    from vllm import _sm70_ops as ops

    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(moe, "is_exact_sm70_cuda", lambda *a, **k: True)
    monkeypatch.setattr(moe, "_use_qwen38_indexed_prefill", lambda *a: False)
    monkeypatch.setattr(moe, "_use_grouped_decode", lambda *a: True)
    layer = NS(
        sm70_nvfp4_hidden_size=2560,
        sm70_nvfp4_top_k=10,
        sm70_nvfp4_qwen38_fused_swiglu_prefill=True,
        w13_tm_weight=object(),
        w13_tm_scales=object(),
        w2_tm_weight=object(),
        w2_tm_scales=object(),
        _nvfp4_grouped_rows=object(),
        _nvfp4_grouped_experts=object(),
        _nvfp4_grouped_sizes=object(),
        _nvfp4_grouped_total=object(),
    )
    buffers = {
        key: torch.empty(1) for key in ("output", "intermediate", "sorted_output")
    }
    method = object.__new__(moe.ModelOptNvFp4SM70MoEMethod)
    method._get_buffers = Mock(return_value=buffers)
    w13, w2 = Mock(), Mock()
    monkeypatch.setattr(ops, "nvfp4_grouped_w13_sm70_out", w13)
    monkeypatch.setattr(ops, "nvfp4_grouped_w2_batch_reduce_sm70_out", w2)
    actual = method.apply(
        layer,
        torch.zeros(tokens, 2560, dtype=torch.float16),
        torch.ones(tokens, 10),
        torch.zeros(tokens, 10, dtype=torch.int32),
        None,
        None,
    )
    assert actual is buffers["output"]
    assert w13.call_args.args[-2:] == (split, True)
    w2.assert_called_once()
