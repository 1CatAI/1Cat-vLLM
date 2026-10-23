# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepared native launch policy retains arithmetic and graph resource ownership."""

import pytest
import torch

from vllm._sm70 import auxiliary


def staging_inputs(tokens, device="cuda"):
    generator = torch.Generator(device=device).manual_seed(20261009)
    mul = torch.randn((8, tokens, 24), device=device, generator=generator) * 0.01
    sqr = torch.rand((8, tokens), device=device, generator=generator) + 4096
    scale = torch.tensor([0.5, 0.75, 0.25], device=device)
    base = torch.randn((24,), device=device, generator=generator) * 0.1
    residual = (
        torch.randn(
            (tokens, 4, 4096), device=device, dtype=torch.float16, generator=generator
        )
        * 0.1
    )
    weight = (
        torch.randn((4096,), device=device, dtype=torch.float16, generator=generator)
        * 0.1
    )
    outputs = (
        torch.empty((tokens, 4), device=device),
        torch.empty((tokens, 4, 4), device=device),
        torch.empty((tokens, 4096), device=device, dtype=torch.float16),
    )
    args = (
        mul,
        sqr,
        scale,
        base,
        residual,
        *outputs,
        weight,
        1e-6,
        1e-6,
        1e-6,
        1.0,
        20,
        1e-6,
    )
    return args, outputs


def test_configured_launch_does_not_fall_back_to_environment(monkeypatch):
    calls = []

    def lookup(name):
        if name.endswith("configured_out"):
            raise RuntimeError("configured operator unavailable")
        return lambda *args: calls.append(name)

    monkeypatch.setattr(auxiliary, "_op", lookup)
    args = (None,) * 9 + (1e-6, 1e-6, 1e-6, 1.0, 20, 1e-6)
    with pytest.raises(RuntimeError, match="configured operator unavailable"):
        auxiliary.sm70_glm_mhc_pre_norm_out(*args, threads=256)
    assert not calls
    auxiliary.sm70_glm_mhc_pre_norm_out(*args)
    assert calls == ["sm70_glm_mhc_pre_norm_out"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA operator contract")
@pytest.mark.parametrize("tokens", [1, 8])
@pytest.mark.parametrize("threads", [128, 256, 512, 1024, 17])
def test_configured_native_matches_legacy_bitwise(monkeypatch, tokens, threads):
    args, outputs = staging_inputs(tokens)
    monkeypatch.setenv("VLLM_SM70_GLM_MHC_PRE_THREADS", str(threads))
    auxiliary.sm70_glm_mhc_pre_norm_out(*args)
    expected = tuple(t.clone() for t in outputs)
    # Another engine's legacy environment cannot override the captured launch.
    monkeypatch.setenv("VLLM_SM70_GLM_MHC_PRE_THREADS", "512")
    auxiliary.sm70_glm_mhc_pre_norm_out(*args, threads=threads)
    for actual, reference in zip(outputs, expected):
        assert torch.equal(actual, reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph contract")
@pytest.mark.parametrize("tokens", [1, 8])
def test_configured_native_graph_replays_updated_inputs(tokens):
    args, outputs = staging_inputs(tokens)

    def run():
        auxiliary.sm70_glm_mhc_pre_norm_out(*args, threads=256)

    run()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    args[0].mul_(1.5)
    args[4].mul_(0.5)
    run()
    expected = tuple(t.clone() for t in outputs)
    graph.replay()
    torch.accelerator.synchronize()
    for actual, reference in zip(outputs, expected):
        assert torch.equal(actual, reference)
