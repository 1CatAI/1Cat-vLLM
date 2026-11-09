# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.layernorm import (
    GemmaRMSNorm,
    _sm70_dflash2_fixed_gemma_rms_norm,
    _sm70_dflash2_gemma_fused_add_rms_norm,
    _use_sm70_dflash2_fixed_gemma_rms,
    _use_sm70_dflash2_gemma_fused_add_rms,
)


@pytest.mark.parametrize("has_residual", [False, True])
def test_fixed_gemma_norm_is_row_invariant_and_preserves_residual(has_residual):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 CUDA device required")
    torch.manual_seed(20260908)
    x = torch.randn((153, 5120), dtype=torch.float16, device="cuda")
    weight = torch.randn(5120, dtype=torch.float16, device="cuda").mul_(0.05)
    residual = torch.randn_like(x) if has_residual else None
    x_before = x.clone()
    residual_before = residual.clone() if residual is not None else None
    actual = _sm70_dflash2_fixed_gemma_rms_norm(x, residual, weight, 1e-6)
    if residual is not None:
        actual, residual_out = actual
        torch.testing.assert_close(
            residual_out, x.float() + residual.float(), atol=0, rtol=0
        )
        assert residual_out.dtype == torch.float32
        torch.testing.assert_close(residual, residual_before, atol=0, rtol=0)
    torch.testing.assert_close(x, x_before, atol=0, rtol=0)

    # The same rows must not acquire another reduction order at q1/q8 or at
    # an irregular prefill boundary. Include the final masked tile's values.
    for first, last in ((0, 1), (0, 8), (8, 143), (143, 153)):
        r = residual[first:last] if residual is not None else None
        part = _sm70_dflash2_fixed_gemma_rms_norm(x[first:last], r, weight, 1e-6)
        if has_residual:
            part = part[0]
        assert torch.equal(part, actual[first:last])

    values = x.double()
    if residual is not None:
        # Residual storage retains the existing FP32 addition contract.
        values = (x.float() + residual.float()).double()
    reference = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
    reference = (reference * (weight.double() + 1)).half()
    lower = torch.nextafter(reference, torch.full_like(reference, -float("inf")))
    upper = torch.nextafter(reference, torch.full_like(reference, float("inf")))
    assert torch.all((actual >= lower) & (actual <= upper))


def test_fixed_gemma_norm_fullgraph_replay_and_dispatch(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 CUDA device required")
    monkeypatch.setenv("VLLM_SM70_DFLASH2_FIXED_GEMMA_RMS", "1")
    monkeypatch.setenv("VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH", "1")
    x = torch.randn((8, 5120), device="cuda", dtype=torch.float16)
    weight = torch.zeros(5120, device="cuda", dtype=torch.float16)
    residual = torch.randn_like(x)

    @torch.compile(backend="inductor", fullgraph=True)
    def apply(values, skip, norm_weight):
        if _use_sm70_dflash2_fixed_gemma_rms(values, skip, norm_weight):
            return _sm70_dflash2_fixed_gemma_rms_norm(values, skip, norm_weight, 1e-6)
        return values, skip

    apply(x, residual, weight)
    assert not _use_sm70_dflash2_fixed_gemma_rms(x, residual.float(), weight)
    assert not _use_sm70_dflash2_fixed_gemma_rms(x[:, :128], None, weight[:128])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, skip = apply(x, residual, weight)
    for _ in range(3):
        x.copy_(torch.randn_like(x))
        residual.copy_(torch.randn_like(residual))
        graph.replay()
        expected, expected_skip = _sm70_dflash2_fixed_gemma_rms_norm(
            x, residual, weight, 1e-6
        )
        assert torch.equal(actual, expected)
        assert torch.equal(skip, expected_skip)


@pytest.mark.parametrize("num_tokens", [1, 8, 32, 137, 512])
@pytest.mark.parametrize("weight_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_sm70_dflash2_gemma_fused_add_rms_is_within_one_fp16_ulp(
    num_tokens: int,
    weight_dtype: torch.dtype,
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 CUDA device required")

    torch.manual_seed(20260823)
    x = torch.randn((num_tokens, 5120), dtype=torch.float16, device="cuda").mul_(0.125)
    residual = torch.randn((num_tokens, 5120), dtype=torch.float32, device="cuda").mul_(
        0.125
    )
    weight = torch.randn(5120, dtype=torch.float32, device="cuda").mul_(0.05)
    weight = weight.to(weight_dtype)

    expected_normalized, expected_residual = GemmaRMSNorm._forward_static_with_residual(
        weight, 1e-6, x, residual
    )
    actual_normalized, actual_residual = _sm70_dflash2_gemma_fused_add_rms_norm(
        x, residual, weight, 1e-6
    )

    lower = torch.nextafter(
        expected_normalized,
        torch.full_like(expected_normalized, -float("inf")),
    )
    upper = torch.nextafter(
        expected_normalized,
        torch.full_like(expected_normalized, float("inf")),
    )
    assert torch.all((actual_normalized >= lower) & (actual_normalized <= upper))
    assert torch.equal(actual_residual, expected_residual)


def test_sm70_dflash2_gemma_fused_add_rms_graph_replay_reads_current_inputs():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 CUDA device required")

    torch.manual_seed(20260823)
    x = torch.randn((8, 5120), dtype=torch.float16, device="cuda").mul_(0.125)
    residual = torch.randn((8, 5120), dtype=torch.float32, device="cuda").mul_(0.125)
    weight = torch.randn(5120, dtype=torch.float32, device="cuda").mul_(0.05)
    _sm70_dflash2_gemma_fused_add_rms_norm(x, residual, weight, 1e-6)
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual_normalized, actual_residual = _sm70_dflash2_gemma_fused_add_rms_norm(
            x, residual, weight, 1e-6
        )

    x.copy_(torch.randn_like(x).mul_(0.125))
    residual.copy_(torch.randn_like(residual).mul_(0.125))
    weight.copy_(torch.randn_like(weight).mul_(0.05))
    graph.replay()
    torch.accelerator.synchronize()

    expected_normalized, expected_residual = GemmaRMSNorm._forward_static_with_residual(
        weight, 1e-6, x, residual
    )
    lower = torch.nextafter(
        expected_normalized,
        torch.full_like(expected_normalized, -float("inf")),
    )
    upper = torch.nextafter(
        expected_normalized,
        torch.full_like(expected_normalized, float("inf")),
    )
    assert torch.all((actual_normalized >= lower) & (actual_normalized <= upper))
    assert torch.equal(actual_residual, expected_residual)


def test_sm70_dflash2_gemma_fused_add_rms_allows_aot_warmup_shape(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 CUDA device required")

    monkeypatch.setenv("VLLM_SM70_DFLASH2_FUSED_GEMMA_RMS", "1")
    monkeypatch.setenv("VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH", "1")
    x = torch.empty((512, 5120), dtype=torch.float16, device="cuda")
    residual = torch.empty_like(x, dtype=torch.float32)
    weight = torch.empty(5120, dtype=torch.float16, device="cuda")

    assert _use_sm70_dflash2_gemma_fused_add_rms(x, residual, weight)


def test_sm70_dflash2_gemma_fused_predicate_is_fullgraph_safe(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 CUDA device required")

    monkeypatch.setenv("VLLM_SM70_DFLASH2_FUSED_GEMMA_RMS", "1")
    monkeypatch.setenv("VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH", "1")
    x = torch.zeros((8, 5120), dtype=torch.float16, device="cuda")
    residual = torch.empty_like(x, dtype=torch.float32)
    weight = torch.empty(5120, dtype=torch.float16, device="cuda")

    @torch.compile(backend="eager", fullgraph=True)
    def apply_predicate(
        values: torch.Tensor,
        residual_values: torch.Tensor,
        norm_weight: torch.Tensor,
    ) -> torch.Tensor:
        if _use_sm70_dflash2_gemma_fused_add_rms(values, residual_values, norm_weight):
            return values + 1
        return values

    torch.testing.assert_close(apply_predicate(x, residual, weight), x + 1)
