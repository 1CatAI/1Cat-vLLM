# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch._inductor.pattern_matcher import PatternMatcherPass
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

from vllm.compilation.passes.fusion import allreduce_rms_fusion as fusion
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
def test_sm70_push_norm_matches_config_epsilon(
    monkeypatch, epsilon, pattern_epsilon, expected_matches, keep_residual
):
    monkeypatch.setattr(
        fusion, "get_tp_group", lambda: SimpleNamespace(unique_name="tp:0")
    )
    monkeypatch.setattr(fusion, "get_tensor_model_parallel_world_size", lambda: 4)
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
        fusion.Sm70Tp4PushGemmaRMSNormPattern(
            torch.float16, "cpu", pattern_epsilon
        ).register(patterns)
        assert patterns.apply(graph.graph) == expected_matches
        fused = [
            n
            for n in graph.graph.nodes
            if n.target == torch.ops.vllm.sm70_tp4_all_reduce_gemma_rms_norm.default
        ]
        assert len(fused) == expected_matches
        if fused:
            assert fused[0].args[3] == epsilon
        assert any(
            n.target == torch.ops.vllm.all_reduce.default for n in graph.graph.nodes
        ) == (expected_matches == 0)
