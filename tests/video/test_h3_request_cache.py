# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request isolation and official cache algorithm/refresh contracts."""

import numpy as np
import pytest
import torch
from torch import nn

from vllm.model_executor.models.minimax_h3.config import (
    H3Config,
    H3InputError,
    H3SamplingParams,
)
from vllm.model_executor.models.minimax_h3.request_cache import (
    CACHE_DIT_DEFAULTS,
    TEA_COEFFICIENTS,
    CachePlan,
    TeaCacheState,
    request_cache,
    resolve_cache_plan,
)


@pytest.mark.parametrize(
    "backend,options",
    [
        ("invalid", {}),
        ("none", {"rel_l1_thresh": 0.17}),
        ("tea_cache", {"rel_l1_thresh": float("nan")}),
        ("cache_dit", {"Fn_compute_blocks": 1.5}),
        ("cache_dit", {"max_warmup_steps": -1}),
        ("cache_dit", {"max_cached_steps": True}),
        ("cache_dit", {"enable_taylorseer": 1}),
        ("cache_dit", {"taylorseer_order": -1}),
        ("cache_dit", {"taylorseer_order": True}),
        ("cache_dit", {"scm_steps_mask_policy": "custom"}),
        ("cache_dit", {"scm_steps_policy": "unknown"}),
    ],
)
def test_invalid_cache_configuration(backend, options):
    with pytest.raises(H3InputError):
        H3Config(cache_backend=backend, cache_config=options)


def test_official_quality_and_partition_policy():
    with pytest.raises(H3InputError, match="FL2VA"):
        H3Config(partition="ref2va", cache_backend="tea_cache")
    config = H3Config(cache_backend="cache_dit")
    assert (
        resolve_cache_plan(
            config, H3SamplingParams(quality="lossless"), calls=49
        ).backend
        == "none"
    )
    high = resolve_cache_plan(H3Config(), H3SamplingParams(quality="high"), calls=49)
    assert high.options == {
        **CACHE_DIT_DEFAULTS,
        "residual_diff_threshold": 0.04,
        "max_continuous_cached_steps": 1,
    }
    advanced = H3Config(
        cache_backend="cache_dit",
        cache_config={"enable_taylorseer": True, "scm_steps_mask_policy": "fast"},
    )
    assert (
        resolve_cache_plan(advanced, H3SamplingParams(quality="high"), calls=8).options
        == high.options
    )
    with pytest.raises(H3InputError, match="mutually exclusive"):
        resolve_cache_plan(
            H3Config(cache_backend="tea_cache"),
            H3SamplingParams(quality="high"),
            calls=49,
        )


@pytest.mark.parametrize(
    "extra",
    [
        {"force_refresh_step_hint": True},
        {"force_refresh_step_hint": 0},
        {"force_refresh_step_hint": 5},
        {"force_refresh_step_policy": "once"},
        {"force_refresh_step_hint": 1, "force_refresh_step_policy": "bad"},
    ],
)
def test_refresh_uses_actual_intervals(extra):
    with pytest.raises(H3InputError):
        resolve_cache_plan(
            H3Config(cache_backend="cache_dit"),
            H3SamplingParams(extra_args=extra),
            calls=4,
        )


def test_refresh_requires_active_cache():
    with pytest.raises(H3InputError, match="active Cache-DiT"):
        resolve_cache_plan(
            H3Config(),
            H3SamplingParams(extra_args={"force_refresh_step_hint": 2}),
            calls=4,
        )
    result = resolve_cache_plan(
        H3Config(cache_backend="cache_dit"),
        H3SamplingParams(extra_args={"force_refresh_step_hint": 4}),
        calls=4,
    )
    assert result.options["force_refresh_step_policy"] == "once"


def test_tea_matches_official_polynomial_and_residual_updates():
    state = TeaCacheState(0.5)
    prior_input = residual = None
    accumulated = 0.0
    for i in range(12):
        hidden = torch.arange(24, dtype=torch.float32).reshape(6, 4) + i / 40
        modulated = hidden / 2 + 0.1
        if i == 0:
            compute = True
        else:
            assert prior_input is not None
            distance = float(
                (
                    (modulated - prior_input).abs().mean()
                    / (prior_input.abs().mean() + 1e-8)
                ).cpu()
            )
            accumulated += abs(float(np.poly1d(TEA_COEFFICIENTS)(distance)))
            compute = accumulated >= 0.5
        if compute:
            accumulated = 0.0
            expected = hidden * 1.01 + 0.04
            residual = expected - hidden.clone()
        else:
            expected = hidden + residual
        actual = state.execute(hidden, modulated, lambda x: x * 1.01 + 0.04)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert state.records[-1]["computed"] == compute
        prior_input = modulated.detach()
    assert any(not item["computed"] for item in state.records)
    state.clear()
    assert state.previous_input is None and state.previous_residual is None


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(4) * 1.001)
        self.calls = 0

    def forward(self, hidden, **kwargs):
        self.calls += 1
        return hidden @ self.weight + 0.01


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block() for _ in range(6)])

    def forward(self, hidden):
        for block in self.blocks:
            hidden = block(hidden)
        return hidden


@pytest.mark.parametrize("hint,policy", [(None, "once"), (3, "once"), (3, "repeat")])
@pytest.mark.parametrize(
    "advanced",
    [
        {},
        {"enable_taylorseer": True, "taylorseer_order": 2},
        {"scm_steps_mask_policy": "medium"},
        {"scm_steps_mask_policy": "medium", "scm_steps_policy": "static"},
        {
            "enable_taylorseer": True,
            "scm_steps_mask_policy": "medium",
            "scm_steps_policy": "static",
        },
    ],
)
def test_actual_cachedit_repeated_requests_and_teardown(hint, policy, advanced):
    model = Model().eval()
    options = {
        **CACHE_DIT_DEFAULTS,
        "Fn_compute_blocks": 1,
        "Bn_compute_blocks": 1,
        "max_warmup_steps": 1,
        "residual_diff_threshold": 1,
        "max_continuous_cached_steps": 2,
        **advanced,
    }
    if hint is not None:
        options.update(force_refresh_step_hint=hint, force_refresh_step_policy=policy)
    plan = CachePlan("cache_dit", options, 8)
    outputs, patterns = [], []
    for request in range(2):
        original = list(model.blocks)
        results, steps = [], []
        with torch.inference_mode(), request_cache(model, plan):
            for step in range(8):
                before = [b.calls for b in original]
                results.append(model(torch.ones(5, 4) * (1 + step * 0.0001)).clone())
                steps.append([b.calls - before[i] for i, b in enumerate(original)])
        assert list(model.blocks) == original
        assert not model._h3_cache_active
        assert not getattr(model, "_is_cached", False)
        if advanced.get("scm_steps_mask_policy") and hint == 3 and policy == "repeat":
            # Every third call refreshes before medium SCM reaches its first
            # reuse slot. The official policy legitimately computes all blocks.
            assert [sum(step) for step in steps] == [6] * 8
        else:
            assert any(sum(step) < 6 for step in steps)
        outputs.append(results)
        patterns.append(steps)
    assert patterns[0] == patterns[1]
    for a, b in zip(*outputs):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    if advanced.get("scm_steps_policy") == "static" and hint is None:
        # Official medium mask for eight actual calls: 11110101.
        assert [sum(step) for step in patterns[0]] == [6, 6, 6, 6, 2, 6, 2, 6]


@pytest.mark.parametrize("calls", [3, 4, 5, 6, 7, 8, 49])
def test_scm_refresh_uses_official_mask_for_actual_calls(monkeypatch, calls):
    import cache_dit

    observed = []
    refresh = cache_dit.refresh_context

    def record(model, **kwargs):
        observed.append(kwargs)
        return refresh(model, **kwargs)

    monkeypatch.setattr(cache_dit, "refresh_context", record)
    config = H3Config(
        cache_backend="cache_dit",
        cache_config={"scm_steps_mask_policy": "fast"},
    )
    plan = resolve_cache_plan(config, H3SamplingParams(), calls=calls)
    with request_cache(Model(), plan):
        pass
    assert len(observed) == 1
    if calls >= 8 or calls in (4, 6):
        actual = observed[0]["cache_config"]
        assert actual.num_inference_steps == calls
        assert actual.steps_computation_mask == cache_dit.steps_mask(
            mask_policy="fast", total_steps=calls
        )
        assert actual.steps_computation_mask[-1] == 1
    else:
        # Match Omni's ordinary refresh for unsupported short SCM schedules.
        assert observed[0] == {"num_inference_steps": calls, "verbose": False}


def test_exception_and_reentrant_request_cleanup():
    model = Model()
    plan = CachePlan("tea_cache", {"rel_l1_thresh": 0.17}, 4)
    with (
        pytest.raises(RuntimeError, match="cannot overlap"),
        request_cache(model, plan),
        request_cache(model, plan),
    ):
        pass
    assert not model._h3_cache_active
    assert not hasattr(model, "_h3_tea_cache")
    with request_cache(model, plan):
        assert model._h3_tea_cache.previous_residual is None
