# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for prefill state ownership and the production warmup matrix."""

from types import SimpleNamespace as NS

import pytest
import torch

from vllm.model_executor.layers.fla.ops import gdn_diagnostics as diagnostics
from vllm.model_executor.layers.fla.ops import gdn_preparation as preparation
from vllm.model_executor.layers.fla.ops.gdn_prefill import GdnPrefill
from vllm.model_executor.layers.fla.ops.gdn_selector import (
    GDN_BACKEND_STAGES,
    GdnExecutionPlan,
)
from vllm.model_executor.layers.fla.ops.gdn_stages import GdnHeadContract
from vllm.model_executor.layers.fla.ops.sm70.gdn_decode import FlashQlaDecodeAdmission
from vllm.model_executor.warmup import gdn as warmup


@pytest.mark.parametrize(
    "backend,indexed",
    [("triton", False), ("flashqla_sm70", False), ("flashqla_sm70", True)],
)
@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_prefill_state_indices_commit_and_output_alias(
    monkeypatch, backend, indexed, direct, empty
):
    events = []
    profiler = NS(
        start=lambda: None, end=lambda layer, stage, *a, **k: events.append(stage)
    )
    plan = GdnExecutionPlan(
        backend, GDN_BACKEND_STAGES[backend], True, indexed, direct, None
    )
    provider = GdnPrefill(plan, profiler)
    for name in ("capture_tensor", "capture_state_slice"):
        monkeypatch.setattr(diagnostics, name, lambda *a, **k: None)
    pool = torch.arange(6 * 4, dtype=torch.float16).reshape(6, 1, 2, 2)
    before = pool.clone()
    indices = torch.tensor([] if empty else [4, 1, 3], dtype=torch.long)
    has_state = torch.tensor([] if empty else [True, False, True], dtype=torch.bool)
    count = len(indices)
    expected = before[indices].clone()
    expected[~has_state] = 0
    expected = (expected.float() + 0.1234).to(pool.dtype)
    output = torch.empty(count, 1, 2) if direct else None
    q = torch.zeros(1, count, 1, 2)

    def kernel(**kwargs):
        events.append("kernel")
        assert kwargs["cu_seqlens"].shape == (count + 1,)
        assert kwargs["gate_is_exp"] is False
        if indexed:
            assert kwargs["initial_state"] is pool
            assert kwargs["state_indices"] is indices
            assert kwargs["has_initial_state"] is has_state
            assert kwargs["inplace_final_state"]
            assert not kwargs["output_final_state"]
            pool[indices] = expected
            final = None
        else:
            torch.testing.assert_close(
                kwargs["initial_state"],
                (before[indices] * has_state[:, None, None, None]),
                rtol=0,
                atol=0,
            )
            assert kwargs["initial_state"].is_contiguous()
            assert kwargs["output_final_state"]
            final = kwargs["initial_state"].float() + 0.1234
            if backend == "triton":
                assert "state_indices" not in kwargs
            else:
                assert kwargs["state_indices"] is None
                assert not kwargs["inplace_final_state"]
        result = torch.ones_like(q)
        if output is not None:
            output.copy_(result.squeeze(0))
            result = output.unsqueeze(0)
        return result, final

    provider._call_prefill = kernel
    result, final = provider.execute_prefill(
        q,
        q,
        q,
        q[..., 0],
        q[..., 0],
        ssm_state=pool,
        state_indices=indices,
        has_initial_state=has_state,
        cu_seqlens=torch.arange(count + 1, dtype=torch.int32),
        chunk_indices=None,
        chunk_offsets=None,
        use_qk_l2norm_in_kernel=False,
        gate_is_exp=False,
        core_attn_out=output,
        layer_name="layers.0",
        num_tokens=count,
    )
    expected_pool = before.clone()
    expected_pool[indices] = expected
    torch.testing.assert_close(pool, expected_pool, rtol=0, atol=0)
    assert (final is None) == indexed
    if direct:
        assert output is not None
        assert result.data_ptr() == output.data_ptr()
    assert events == ["state_gather", "kernel", "core_call", "state_writeback"]


@pytest.mark.parametrize(
    "legacy,gate_exp", [(False, False), (False, True), (True, False)]
)
def test_preparation_keeps_gate_and_normalization_contract(
    monkeypatch, legacy, gate_exp
):
    heads = GdnHeadContract(2, 4, 4, 4, 2)
    x = torch.arange(3 * 20, dtype=torch.float16).reshape(3, 20)[:, :16]
    a, b = torch.zeros(3, 2), torch.ones(3, 2, dtype=torch.float16)
    events = []

    def post_conv(**kwargs):
        events.append("fused")
        assert kwargs["apply_l2norm"]
        assert kwargs["output_g_exp"] == gate_exp
        assert kwargs["conv_output"] is x
        return (torch.zeros(3, 1, 4), torch.zeros(3, 1, 4), torch.zeros(3, 2, 4), a, b)

    def gates(A_log, aa, bb, dt_bias):
        events.append("legacy")
        assert aa is a and bb is b
        return a.unsqueeze(0), b.unsqueeze(0)

    monkeypatch.setattr(preparation, "fused_post_conv_prep", post_conv)
    monkeypatch.setattr(preparation, "fused_gdn_gating", gates)
    bound = preparation.GdnPreparation(heads, legacy=legacy, gate_is_exp=gate_exp)
    q, k, v, g, beta = bound.prefill(x, a, b, None, None)
    assert events == ["legacy" if legacy else "fused"]
    assert q.shape == k.shape == (1, 3, 1, 4)
    assert v.shape == (1, 3, 2, 4)
    assert g.shape == beta.shape == (1, 3, 2)
    assert beta.dtype == b.dtype
    if legacy:
        for actual, expected in zip((q, k, v), x.split([4, 4, 8], -1)):
            torch.testing.assert_close(
                actual.flatten(), expected.flatten(), rtol=0, atol=0
            )
            assert actual.is_contiguous()


@pytest.mark.parametrize("failure", [None, "convolution", "prefill", "decode"])
def test_warmup_order_strides_failure_and_engine_ownership(monkeypatch, failure):
    records: list[tuple] = []
    heads = GdnHeadContract(1, 1, 4, 4, 1)

    def conv(x, weight, bias, **kwargs):
        records.append(
            ("conv", x.shape[1], x.stride(), kwargs["cache_indices"].stride())
        )
        if failure == "convolution":
            raise RuntimeError("synthetic conv error")
        return x

    def prepare(contract, x, a, b, *weights):
        assert contract is heads
        records.append(("prep", x.shape[0], x.stride(), a.stride(), b.stride()))
        return (x, x, x, a, b)

    class Prefill:
        gdn_prefill_backend = "flashqla_sm70"
        execution_plan = NS(
            original_prefill=True, direct_prefill_output=True, indexed_prefill=True
        )

        def __call__(self, **kwargs):
            records.append(
                (
                    "prefill",
                    kwargs["q"].shape[1],
                    kwargs["cu_seqlens"].tolist(),
                    kwargs["output_final_state"],
                )
            )
            assert kwargs["core_attn_out"] is not None
            if failure == "prefill":
                raise RuntimeError("synthetic prefill error")

    def decode(contract, **kwargs):
        records.append(
            ("decode", kwargs["mixed_qkv"].shape[0], kwargs["mixed_qkv"].stride())
        )
        assert contract is heads and kwargs["out"].shape[1] == 1
        if failure == "decode":
            raise RuntimeError("synthetic decode error")
        return kwargs["out"], kwargs["initial_state"]

    monkeypatch.setattr(warmup, "causal_conv1d_fn", conv)
    monkeypatch.setattr(warmup, "prepare_prefill", prepare)
    monkeypatch.setattr(warmup, "mixed_qkv_recurrence", decode)
    monkeypatch.setattr(diagnostics, "log_decode_route", lambda **kwargs: None)
    monkeypatch.setattr(
        torch.accelerator, "empty_cache", lambda: records.append(("flush",))
    )
    task = warmup.GdnWarmup(
        heads=heads,
        prefill=Prefill(),
        decode_admission=FlashQlaDecodeAdmission(heads, False, None),
        native_policy=None,
        schedule=None,
        conv_weight=torch.ones(12, 1, 4),
        conv_bias=None,
        activation="silu",
        A_log=torch.zeros(1),
        dt_bias=torch.zeros(1),
        state_dtype=torch.float32,
        device=torch.device("cpu"),
        dtype=torch.float16,
        input_width=16,
        qkv_dim=12,
        qkv_row_stride=20,
        spec_cache_stride=3,
        conv_dim_first=True,
        prefix="layers.0",
        trace=False,
        decode_warmup=False,
    )
    owner: set[tuple] = set()
    task.run(owner)
    assert len(owner) == 1
    phases = [row[0] for row in records]
    assert phases == ["conv"] * (1 if failure == "convolution" else 8) + [
        "prep"
    ] * 12 + ["prefill"] * (1 if failure == "prefill" else 2) + ["decode"] * 4 + [
        "flush"
    ]
    assert {row[1] for row in records if row[0] == "prep"} == {63, 64}
    assert {row[2] for row in records if row[0] == "decode"} == {(12, 1), (16, 1)}
    assert all(
        row[2] == [0, 64 if failure == "convolution" else 63]
        for row in records
        if row[0] == "prefill"
    )
    recorded = list(records)
    task.run(owner)
    assert records == recorded
    task.run(set())  # The same geometry in another engine has its own budget.
    assert records == recorded * 2
