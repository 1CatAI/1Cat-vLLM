# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch._inductor.pattern_matcher import PatternMatcherPass
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

from vllm.compilation.passes.fusion import allreduce_rms_fusion as fusion
from vllm.compilation.passes.fusion.sm70_gemma_fusion import (
    mark_closed_residual_chain,
)
from vllm.model_executor.layers import layernorm  # noqa: F401


@pytest.mark.parametrize(
    "epsilon,pattern_epsilon,expected_matches",
    [
        (1e-6, 1e-6, 1),
        (9.999999974752427e-7, 9.999999974752427e-7, 1),
        (1e-5, 1e-5, 1),
        (9.999999974752427e-7, 1e-6, 0),
        (1e-5, 1e-6, 0),
    ],
)
@pytest.mark.parametrize("keep_residual", [False, True])
@pytest.mark.parametrize("tp", [2, 4])
def test_sm70_push_norm_matches_config_epsilon(
    monkeypatch, epsilon, pattern_epsilon, expected_matches, keep_residual, tp
):
    monkeypatch.setattr(
        fusion, "get_tp_group", lambda: SimpleNamespace(unique_name="tp:0")
    )
    monkeypatch.setattr(fusion, "get_tensor_model_parallel_world_size", lambda: tp)
    monkeypatch.setattr(
        fusion,
        "tensor_model_parallel_all_reduce",
        lambda x: torch.ops.vllm.all_reduce(x, group_name="tp:0"),
    )
    with FakeTensorMode():
        inputs = (
            torch.empty(8, 5120, dtype=torch.float16),
            torch.empty(8, 5120, dtype=torch.float32),
            torch.empty(5120, dtype=torch.float16),
        )

        def model(x, residual, weight):
            reduced = torch.ops.vllm.all_reduce(x.relu(), group_name="tp:0")
            result = torch.ops.vllm.sm70_dflash2_gemma_fused_add_rms_norm(
                reduced, residual, weight, epsilon
            )
            return result if keep_residual else result[0]

        graph = make_fx(model)(*inputs)
        patterns = PatternMatcherPass()
        fusion.Sm70PushGemmaRMSNormPattern(
            torch.float16, "cpu", pattern_epsilon, tp
        ).register(patterns)
        assert patterns.apply(graph.graph) == expected_matches
        collective = torch.ops.vllm.all_reduce_gemma_rms_norm.default
        fused = [n for n in graph.graph.nodes if n.target == collective]
        assert len(fused) == expected_matches
        if fused:
            assert fused[0].args[3] == epsilon
        assert any(
            n.target == torch.ops.vllm.all_reduce.default for n in graph.graph.nodes
        ) == (expected_matches == 0)


def test_sm70_closed_residual_chain_marking(monkeypatch):
    monkeypatch.setattr(
        fusion, "get_tp_group", lambda: SimpleNamespace(unique_name="tp:0")
    )
    monkeypatch.setattr(fusion, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(
        fusion,
        "tensor_model_parallel_all_reduce",
        lambda x: torch.ops.vllm.all_reduce(x, group_name="tp:0"),
    )
    collective = torch.ops.vllm.all_reduce_gemma_rms_norm.default

    with FakeTensorMode():
        inputs = (
            torch.empty(256, 5120, dtype=torch.float16),
            torch.empty(256, 5120, dtype=torch.float32),
            torch.empty(5120, dtype=torch.float16),
        )

        # 1) Closed 3-boundary chain ending in final norm (residual_out discarded):
        def closed_model(x, residual, weight):
            r0 = torch.ops.vllm.all_reduce(x.relu(), group_name="tp:0")
            h0, res0 = torch.ops.vllm.sm70_dflash2_gemma_fused_add_rms_norm(
                r0, residual, weight, 1e-6
            )
            r1 = torch.ops.vllm.all_reduce(h0.relu(), group_name="tp:0")
            h1, res1 = torch.ops.vllm.sm70_gemma_long_prefill_fused_add_rms_norm(
                r1, res0, weight, 1e-6
            )
            r2 = torch.ops.vllm.all_reduce(h1.relu(), group_name="tp:0")
            h2, _ = fusion.Sm70AllReduceGemmaRMSNormPattern(
                1e-6, torch.float16, "cpu", torch.float32
            ) and (
                torch.ops.vllm.sm70_dflash2_gemma_fused_add_rms_norm(
                    r2, res1, weight, 1e-6
                )
            )
            return h2

        graph_closed = make_fx(closed_model)(*inputs)
        patterns = PatternMatcherPass()
        fusion.Sm70PushGemmaRMSNormPattern(torch.float16, "cpu", 1e-6, 2).register(
            patterns
        )
        fusion.Sm70Tp4LongPrefillFusedNormPattern(
            1e-6,
            torch.float16,
            "cpu",
            collective=torch.ops.vllm.all_reduce_gemma_rms_norm,
        ).register(patterns)
        assert patterns.apply(graph_closed.graph) == 3
        marked = mark_closed_residual_chain(graph_closed.graph)
        assert marked == 3
        fused_closed = [n for n in graph_closed.graph.nodes if n.target == collective]
        assert len(fused_closed) == 3
        assert all(bool(n.args[5]) is True for n in fused_closed)

        # 2) Auxiliary reader on middle residual (e.g. Eagle3/DFlash2 aux layer):
        # nodes before/at the aux tap stay full-residual (False); suffix after tap
        # ending in dead residual_out is still safe (True).
        def aux_tap_model(x, residual, weight):
            r0 = torch.ops.vllm.all_reduce(x.relu(), group_name="tp:0")
            h0, res0 = torch.ops.vllm.sm70_dflash2_gemma_fused_add_rms_norm(
                r0, residual, weight, 1e-6
            )
            aux = h0.float() + res0
            r1 = torch.ops.vllm.all_reduce(h0.relu(), group_name="tp:0")
            h1, _ = torch.ops.vllm.sm70_dflash2_gemma_fused_add_rms_norm(
                r1, res0, weight, 1e-6
            )
            return h1, aux

        graph_aux = make_fx(aux_tap_model)(*inputs)
        assert patterns.apply(graph_aux.graph) == 2
        marked_aux = mark_closed_residual_chain(
            graph_aux.graph, allow_piecewise_output=True
        )
        fused_aux = [n for n in graph_aux.graph.nodes if n.target == collective]
        assert marked_aux == 1
        assert len(fused_aux[0].args) == 5 or bool(fused_aux[0].args[5]) is False
        assert bool(fused_aux[1].args[5]) is True

        # 3) Piecewise subgraph returning residual_out across attention split:
        def piecewise_sub(x, residual, weight):
            r0 = torch.ops.vllm.all_reduce(x.relu(), group_name="tp:0")
            h0, res0 = torch.ops.vllm.sm70_dflash2_gemma_fused_add_rms_norm(
                r0, residual, weight, 1e-6
            )
            return h0, res0

        graph_pw = make_fx(piecewise_sub)(*inputs)
        assert patterns.apply(graph_pw.graph) == 1
        assert (
            mark_closed_residual_chain(graph_pw.graph, allow_piecewise_output=False)
            == 0
        )
        assert (
            mark_closed_residual_chain(graph_pw.graph, allow_piecewise_output=True) == 1
        )
        fused_pw = [n for n in graph_pw.graph.nodes if n.target == collective]
        assert bool(fused_pw[0].args[5]) is True
