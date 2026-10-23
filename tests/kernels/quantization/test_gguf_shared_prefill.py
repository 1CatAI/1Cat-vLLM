# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared expert alignment must preserve decode and the calibrated MMQ bounds."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.kernels.gguf import (
    original_moe_prefill_block_size,
    original_moe_prefill_capability,
)


@pytest.mark.parametrize(
    "change",
    [
        {"source_types": (23, 23, 14)},
        {"source_types": (21, 21, 14)},
        {"source_types": (21, 23, 23)},
        {"hidden_size": 2560},
        {"intermediate_size": 1024},
        {"experts": 512},
        {"top_k": 4},
        {"dtype": torch.bfloat16},
        {"compute_capability": 80},
        {"compute_capability": None},
    ],
)
def test_unmeasured_layout_keeps_independent_routes(change):
    args = dict(
        source_types=(21, 21, 23),
        hidden_size=4096,
        intermediate_size=2048,
        experts=288,
        top_k=8,
        dtype=torch.float16,
        compute_capability=70,
    )
    args.update(change)
    assert original_moe_prefill_block_size(**args) == 0


@pytest.mark.parametrize("sm,minimum", [(70, 384), (86, 384), (89, 256)])
@pytest.mark.parametrize("offset", [-383, -1, 0, 1, 384, 385])
def test_shared_plan_requires_every_projection_in_original_range(
    monkeypatch, sm, minimum, offset
):
    from vllm.model_executor.layers.fused_moe import MoEActivation
    from vllm.model_executor.layers.fused_moe import moe_align_block_size as align
    from vllm.model_executor.layers.quantization.gguf_moe import GGUFNativeMoEMethod

    m = max(1, minimum + offset)
    calls, plans = [], []
    route_plan = (object(), object(), object())

    def make_plan(ids, block_size, experts, *, pad_sorted_ids):
        assert ids.shape == (m, 8)
        assert (block_size, experts, pad_sorted_ids) == (16, 288, True)
        plans.append(1)
        return route_plan

    def operation(name):
        def run(x, weight, ids, kind, rows, topk, count, *metadata):
            if metadata:
                assert (ids, *metadata) == route_plan
            else:
                assert ids.shape == (count, topk)
            calls.append((name, count, topk, bool(metadata)))
            return x.new_ones((count * topk, rows))

        return run

    monkeypatch.setattr(align, "moe_align_block_size", make_plan)
    monkeypatch.setattr(
        torch.ops,
        "_C_gguf",
        SimpleNamespace(ggml_moe_mmq=operation("mmq"), ggml_moe_mmvq=operation("mmvq")),
    )
    method = object.__new__(GGUFNativeMoEMethod)
    method.dp4a_admitted = False
    method.expert_trace_tokens = 0
    method.expert_cache = None
    method.hidden_size, method.intermediate_size = 8, 4
    method.num_experts = 288
    method.input_padding = {}
    method.weight_types = {"w1": 21, "w3": 21, "w2": 23}
    method.prefill_block_size = original_moe_prefill_block_size(
        (21, 21, 23), 4096, 2048, 288, 8, torch.float16, sm
    )
    assert method.prefill_block_size == 16
    method.prefill_capabilities = {
        shard: original_moe_prefill_capability(
            kind,
            2048 if shard == "w2" else 4096,
            4096 if shard == "w2" else 2048,
            288,
            torch.float16,
            is_sm70=sm == 70,
            compute_capability=sm,
            top_k=8,
            mmq_supported=True,
        )
        for shard, kind in method.weight_types.items()
    }
    method.projection_capabilities = {
        shard: replace(cap, operator="ggml_moe_mmvq", min_m=1, max_m=None)
        for shard, cap in method.prefill_capabilities.items()
    }
    layer = SimpleNamespace(
        apply_router_weight_on_input=False,
        activation=MoEActivation.SILU,
        expert_map=None,
        swiglu_limit=10.0,
        gguf_w1=None,
        gguf_w3=None,
        gguf_w2=None,
    )
    x = torch.ones((m, 8), dtype=torch.float16)
    ids = torch.zeros((m, 8), dtype=torch.int32)
    weights = torch.full((m, 8), 0.125)
    output = method.apply(layer, x, weights, ids, None, None)
    assert torch.equal(output, torch.ones_like(output))
    admitted = minimum <= m <= 768
    selected = "mmq" if admitted else "mmvq"
    assert calls == [
        (selected, m, 8, admitted),
        (selected, m, 8, admitted),
        (selected, m * 8, 1, admitted),
    ]
    assert len(plans) == int(admitted)

    # A partially admitted layer must never use a shared plan. Existing mixed
    # source formats retain their independently prepared operators.
    calls.clear()
    plans.clear()
    method.prefill_capabilities["w3"] = replace(
        method.prefill_capabilities["w3"], reason="unmeasured_source"
    )
    method.apply(layer, x, weights, ids, None, None)
    assert not plans and all(not call[-1] for call in calls)
