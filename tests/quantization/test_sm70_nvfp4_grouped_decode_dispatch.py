# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace as NS

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


@pytest.mark.parametrize("tokens", [1, 2, 4, 5, 8, 10, 16, 17])
def test_grouped_mtp5_preserves_direct_admission(monkeypatch, tokens):
    monkeypatch.setenv("VLLM_SM70_NVFP4_QWEN38_MOE_QPN_MTP5_DECODE", "1")
    envs.disable_envs_cache()
    layer = NS(
        sm70_nvfp4_grouped_mtp5=True,
        moe_config=NS(tp_size=4),
        sm70_nvfp4_num_experts=512,
        sm70_nvfp4_hidden_size=2560,
        sm70_nvfp4_intermediate_size=160,
        sm70_nvfp4_top_k=10,
    )
    x = torch.empty(tokens, 2560, dtype=torch.float16)
    ids = torch.empty(tokens, 10, dtype=torch.int32)
    assert moe._use_grouped_mtp5(layer, x, ids) == (tokens == 5)
    layer.moe_config.tp_size = 2
    assert not moe._use_grouped_mtp5(layer, x, ids)
    layer.moe_config.tp_size = 4
    layer.sm70_nvfp4_grouped_mtp5 = False
    assert not moe._use_grouped_mtp5(layer, x, ids)


def test_grouped_mtp5_defaults_on(monkeypatch):
    monkeypatch.delenv("VLLM_SM70_NVFP4_MOE_GROUPED_MTP5", raising=False)
    envs.disable_envs_cache()
    assert envs.VLLM_SM70_NVFP4_MOE_GROUPED_MTP5


def test_grouped_mtp5_unprepared_layer_falls_back():
    x = torch.empty(5, 2560, dtype=torch.float16)
    ids = torch.empty(5, 10, dtype=torch.int32)
    assert not moe._use_grouped_mtp5(NS(), x, ids)


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


def test_grouped_ops_have_fake_implementations():
    from torch._subclasses.fake_tensor import FakeTensorMode

    from vllm import _sm70_ops as ops

    if not ops.has_nvfp4_grouped_decode_dispatch():
        pytest.skip("Build native grouped-decode ops first")
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
        ops.nvfp4_grouped_w2_sm70_out(
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


@pytest.mark.parametrize("intermediate", [160, 320, 640])
@pytest.mark.parametrize("tokens", [1, 2, 5, 10, 20, 40, 41])
def test_grouped_batch_supports_tp1_tp2_tp4_shapes(monkeypatch, intermediate, tokens):
    monkeypatch.setenv("VLLM_SM70_NVFP4_QWEN38_MOE_QPN_MTP5_DECODE", "1")
    envs.disable_envs_cache()
    layer = NS(
        sm70_nvfp4_grouped_batch=True,
        sm70_nvfp4_num_experts=512,
        sm70_nvfp4_hidden_size=2560,
        sm70_nvfp4_intermediate_size=intermediate,
        sm70_nvfp4_top_k=10,
    )
    x = torch.empty(tokens, 2560, dtype=torch.float16)
    ids = torch.empty(tokens, 10, dtype=torch.int32)
    assert moe._use_grouped_batch(layer, x, ids) == (2 <= tokens <= 40)
    layer.sm70_nvfp4_intermediate_size = 256
    assert not moe._use_grouped_batch(layer, x, ids)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires SM70 CUDA",
)
@pytest.mark.parametrize("intermediate", [160, 320, 640])
@pytest.mark.parametrize("tokens", [5, 10, 20, 40])
def test_grouped_nvfp4_ops_match_across_tp_shards(intermediate, tokens):
    from vllm import _sm70_ops as ops
    from vllm.model_executor.layers.fused_moe.sm70.reduction import weighted_reduce_rows

    if not ops.has_nvfp4_grouped_decode_dispatch():
        pytest.skip("Build native grouped-decode ops first")
    torch.manual_seed(intermediate * 100 + tokens)
    dev = "cuda"
    experts, hidden, top_k = 512, 2560, 10
    slots = tokens * top_k
    x = (
        torch.randn(tokens, hidden, device=dev, dtype=torch.float16) * 0.25
    ).contiguous()
    ids = torch.stack(
        [torch.randperm(experts, device=dev)[:top_k] for _ in range(tokens)]
    ).to(torch.int32)
    weights = torch.rand(tokens, top_k, device=dev, dtype=torch.float32).softmax(-1)
    w13 = torch.randint(
        -(2**31),
        2**31 - 1,
        (experts, hidden, (2 * intermediate) // 8),
        device=dev,
        dtype=torch.int32,
    )
    s13 = (
        torch.rand(
            experts, hidden // 16, 2 * intermediate, device=dev, dtype=torch.float16
        )
        * 0.05
    )
    w2 = torch.randint(
        -(2**31),
        2**31 - 1,
        (experts, intermediate, hidden // 8),
        device=dev,
        dtype=torch.int32,
    )
    s2 = (
        torch.rand(experts, intermediate // 16, hidden, device=dev, dtype=torch.float16)
        * 0.05
    )

    rows = torch.empty(400, 8, device=dev, dtype=torch.int32)
    grp_experts = torch.empty(400, device=dev, dtype=torch.int32)
    grp_sizes = torch.empty(400, device=dev, dtype=torch.int32)
    grp_total = torch.empty(1, device=dev, dtype=torch.int32)

    mid = torch.empty(slots, intermediate, device=dev, dtype=torch.float16)
    split_k = 4 if intermediate == 160 or tokens <= 8 else 8
    ops.nvfp4_grouped_w13_sm70_out(
        mid,
        x,
        w13,
        s13,
        ids.view(-1),
        rows,
        grp_experts,
        grp_sizes,
        grp_total,
        split_k,
        False,
    )
    routed = torch.empty(slots, hidden, device=dev, dtype=torch.float16)
    out = torch.empty(tokens, hidden, device=dev, dtype=torch.float16)
    ops.nvfp4_grouped_w2_sm70_out(
        out, routed, mid, w2, s2, weights, rows, grp_experts, grp_sizes, grp_total
    )
    ref = weighted_reduce_rows(
        routed.view(tokens, top_k, hidden), weights, torch.float16
    )
    torch.testing.assert_close(out, ref, atol=2e-3, rtol=2e-3)
    if intermediate == 160 and ops.has_nvfp4_grouped_batch_reduce_dispatch():
        out_br = torch.empty_like(out)
        ops.nvfp4_grouped_w2_batch_reduce_sm70_out(
            out_br,
            routed,
            mid,
            w2,
            s2,
            weights,
            rows,
            grp_experts,
            grp_sizes,
            grp_total,
        )
        torch.testing.assert_close(out_br, ref, atol=2e-3, rtol=2e-3)
