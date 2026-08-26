# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterator

import pytest
import torch

import vllm.envs as envs
from vllm.models.deepseek_v4.sm70.gemv import (
    can_use_sm70_dsv4_fp16_gemv,
    maybe_sm70_dsv4_fp16_gemv,
)


@pytest.fixture(autouse=True)
def reset_env_cache() -> Iterator[None]:
    envs.disable_envs_cache()
    yield
    envs.disable_envs_cache()


def test_sm70_dsv4_fp16_gemv_is_default_off(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_SM70_DSV4_FP16_GEMV", raising=False)
    x = torch.empty((1, 4096), dtype=torch.float16)
    weight = torch.empty((256, 4096), dtype=torch.float16)
    assert not can_use_sm70_dsv4_fp16_gemv(x, weight, torch.float32)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="requires NVIDIA V100/SM70",
)
@pytest.mark.parametrize(
    ("n", "output_dtype"),
    [
        (64, torch.float16),
        (256, torch.float32),
        (512, torch.float32),
        (1024, torch.float32),
        (2048, torch.float32),
    ],
)
def test_sm70_dsv4_fp16_gemv_graph(
    monkeypatch, n: int, output_dtype: torch.dtype
) -> None:
    monkeypatch.setenv("VLLM_SM70_DSV4_FP16_GEMV", "1")
    torch.manual_seed(20260802 + n)
    x = torch.randn((1, 4096), device="cuda", dtype=torch.float16)
    weight = torch.randn((n, 4096), device="cuda", dtype=torch.float16) * 0.01

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        candidate = maybe_sm70_dsv4_fp16_gemv(x, weight, output_dtype)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    assert candidate is not None

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = maybe_sm70_dsv4_fp16_gemv(x, weight, output_dtype)
    assert captured is not None

    x.copy_(torch.randn_like(x))
    graph.replay()
    torch.cuda.synchronize()
    if output_dtype == torch.float16:
        reference = torch.mm(x, weight.T)
        torch.testing.assert_close(captured, reference, rtol=0, atol=0)
    else:
        reference = torch.mm(x, weight.T, out_dtype=torch.float32)
        torch.testing.assert_close(captured, reference, rtol=2e-4, atol=5e-5)
