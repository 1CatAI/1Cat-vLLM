# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The new engine entry retains each legacy norm's bits and capture lifetime."""

from unittest.mock import Mock

import pytest
import torch

from vllm import envs
from vllm.model_executor.layers.fla.ops.sm70 import gdn_norm  # noqa: F401


@pytest.mark.parametrize("onepass", [False, True])
@pytest.mark.parametrize("rows", [12, 24])
def test_configured_norm_matches_legacy_and_replays_without_policy_reads(
    monkeypatch, rows, onepass
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 required")
    torch.manual_seed(123)
    x = torch.randn(rows, 128, device="cuda", dtype=torch.float16)
    z = torch.randn_like(x)
    weight = torch.randn(128, device="cuda", dtype=torch.float16)
    args = (x, z, weight, 1e-6, -1, True, "silu")
    name = "VLLM_SM70_GDN_RMSNORM_ONEPASS"
    monkeypatch.setenv(name, str(int(onepass)))
    envs.disable_envs_cache()
    legacy = torch.ops.vllm.sm70_qwen_gdn_rmsnorm_gated
    configured = torch.ops.vllm.sm70_qwen_gdn_rmsnorm_gated_configured
    expected = legacy(*args)
    actual = configured(*args, onepass)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = configured(*args, onepass)
    for offset in (0.25, -0.5):
        x.add_(offset)
        expected = legacy(*args)
        # Both eager execution and captured replay use their supplied policy.
        with monkeypatch.context() as scope:
            scope.setitem(
                envs.environment_variables,
                name,
                Mock(side_effect=AssertionError("execution read legacy policy")),
            )
            scope.setenv(name, "invalid-after-capture")
            actual = configured(*args, onepass)
            graph.replay()
            torch.accelerator.synchronize()
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
            assert torch.equal(captured.view(torch.int16), expected.view(torch.int16))


def test_configured_norm_schema_fake_and_aot_dispatch():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 required")
    x = torch.randn(12, 128, device="cuda", dtype=torch.float16)
    torch.library.opcheck(
        torch.ops.vllm.sm70_qwen_gdn_rmsnorm_gated_configured,
        (
            x,
            torch.randn_like(x),
            torch.ones(128, device="cuda", dtype=x.dtype),
            1e-6,
            -1,
            True,
            "silu",
            True,
        ),
    )
