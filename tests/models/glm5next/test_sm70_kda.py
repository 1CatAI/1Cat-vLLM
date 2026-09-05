# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

import vllm._sm70_ops as sm70_ops
from vllm.model_executor.layers.fla.ops.kda import (
    fused_recurrent_kda,
    layer_norm_gated_fwd,
)
from vllm.models.glm5next.nvidia.kda import (
    _sm70_exact_kda_gemv_enabled,
    _sm70_glm53_tp8_cublaslt_enabled,
    _sm70_glm53_tp8_fused_fg_b_enabled,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, True), ("1", True), ("0", False)],
)
def test_sm70_exact_kda_gemv_gate(
    monkeypatch: pytest.MonkeyPatch,
    value: str | None,
    expected: bool,
) -> None:
    if value is None:
        monkeypatch.delenv("VLLM_SM70_GLM53_EXACT_KDA_GEMV", raising=False)
    else:
        monkeypatch.setenv("VLLM_SM70_GLM53_EXACT_KDA_GEMV", value)
    assert _sm70_exact_kda_gemv_enabled() is expected


@pytest.mark.parametrize(
    ("value", "enabled"),
    [(None, False), ("1", True), ("0", False)],
)
def test_sm70_glm53_tp8_cublaslt_gate(
    monkeypatch: pytest.MonkeyPatch,
    value: str | None,
    enabled: bool,
) -> None:
    if value is None:
        monkeypatch.delenv("VLLM_SM70_GLM53_TP8_CUBLASLT", raising=False)
    else:
        monkeypatch.setenv("VLLM_SM70_GLM53_TP8_CUBLASLT", value)
    expected = enabled and torch.version.cuda == "12.8"
    assert _sm70_glm53_tp8_cublaslt_enabled() is expected


@pytest.mark.parametrize(
    ("value", "enabled"),
    [(None, False), ("1", True), ("0", False)],
)
def test_sm70_glm53_tp8_fused_fg_b_gate(
    monkeypatch: pytest.MonkeyPatch,
    value: str | None,
    enabled: bool,
) -> None:
    if value is None:
        monkeypatch.delenv("VLLM_SM70_GLM53_TP8_FUSED_FG_B", raising=False)
    else:
        monkeypatch.setenv("VLLM_SM70_GLM53_TP8_FUSED_FG_B", value)
    assert _sm70_glm53_tp8_fused_fg_b_enabled() is enabled


@pytest.mark.skipif(
    not (
        current_platform.is_cuda()
        and current_platform.get_device_capability() == DeviceCapability(7, 0)
        and hasattr(torch.ops._C, "sm70_glm53_tp8_cublaslt_out")
    ),
    reason="native NVIDIA V100/SM70 GLM-5.3 cuBLASLt op required",
)
@pytest.mark.parametrize("seed", range(2))
@pytest.mark.parametrize("n, k", [(3336, 4096), (4096, 1024)])
def test_sm70_glm53_tp8_cublaslt_matches_torch_and_graph(
    seed: int,
    n: int,
    k: int,
) -> None:
    torch.manual_seed(seed)
    device = current_platform.device_type
    input = torch.randn((8, k), device=device, dtype=torch.float16)
    weight = torch.randn((n, k), device=device, dtype=torch.float16)
    output = torch.empty((8, n), device=device, dtype=torch.float16)
    expected = F.linear(input, weight)

    sm70_ops.sm70_glm53_tp8_cublaslt_out(output, input, weight)
    torch.accelerator.synchronize()
    assert torch.equal(output, expected)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sm70_ops.sm70_glm53_tp8_cublaslt_out(output, input, weight)
    input.copy_(torch.randn_like(input))
    expected = F.linear(input, weight)
    graph.replay()
    torch.accelerator.synchronize()
    assert torch.equal(output, expected)


@pytest.mark.skipif(
    not (
        current_platform.is_cuda()
        and current_platform.get_device_capability() == DeviceCapability(7, 0)
        and hasattr(torch.ops._C, "sm70_glm_kda_fg_b_out")
    ),
    reason="native NVIDIA V100/SM70 GLM KDA CUDA op required",
)
@pytest.mark.parametrize("output_rows", [1024, 2048])
@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8])
def test_sm70_glm_kda_fused_fg_b_matches_fp16_and_graph(
    num_tokens: int,
    output_rows: int,
) -> None:
    torch.manual_seed(20260827)
    device = current_platform.device_type
    projected_rows = 3336 if output_rows == 1024 else 6416
    projected = torch.randn(
        (num_tokens, projected_rows), device=device, dtype=torch.float16
    ).mul_(0.1)
    f_input = projected[:, -256:-128]
    g_input = projected[:, -128:]
    assert f_input.stride() == (projected_rows, 1)
    assert g_input.stride() == (projected_rows, 1)
    f_weight = torch.randn((output_rows, 128), device=device, dtype=torch.float16).mul_(
        0.01
    )
    g_weight = torch.randn((output_rows, 128), device=device, dtype=torch.float16).mul_(
        0.01
    )
    f_out = torch.empty((num_tokens, output_rows), device=device, dtype=torch.float16)
    g_out = torch.empty_like(f_out)

    def run() -> None:
        sm70_ops.sm70_glm_kda_fg_b_out(
            f_out, g_out, f_input, g_input, f_weight, g_weight
        )

    run()
    torch.accelerator.synchronize()
    expected = (F.linear(f_input, f_weight), F.linear(g_input, g_weight))
    rowwise_f = []
    rowwise_g = []
    for token_idx in range(num_tokens):
        f_row = torch.empty((1, output_rows), device=device, dtype=torch.float16)
        g_row = torch.empty_like(f_row)
        sm70_ops.sm70_glm_kda_fg_b_out(
            f_row,
            g_row,
            f_input[token_idx : token_idx + 1],
            g_input[token_idx : token_idx + 1],
            f_weight,
            g_weight,
        )
        rowwise_f.append(f_row)
        rowwise_g.append(g_row)
    assert torch.equal(f_out, torch.cat(rowwise_f))
    assert torch.equal(g_out, torch.cat(rowwise_g))
    torch.testing.assert_close(f_out, expected[0], rtol=2e-3, atol=2e-4)
    torch.testing.assert_close(g_out, expected[1], rtol=2e-3, atol=2e-4)
    eager = (f_out.clone(), g_out.clone())

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.accelerator.synchronize()

    torch.testing.assert_close(f_out, eager[0], rtol=0, atol=0)
    torch.testing.assert_close(g_out, eager[1], rtol=0, atol=0)


@pytest.mark.skipif(
    not (
        current_platform.is_cuda()
        and current_platform.get_device_capability() == DeviceCapability(7, 0)
        and hasattr(torch.ops._C, "sm70_glm53_fp16_gemv_out")
    ),
    reason="native NVIDIA V100/SM70 GLM-5.3 exact FP16 GEMV op required",
)
@pytest.mark.parametrize("seed", range(2))
@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8])
def test_sm70_glm53_fp16_gemv_matches_cublas_and_graph(
    seed: int, num_tokens: int
) -> None:
    torch.manual_seed(seed)
    device = current_platform.device_type
    input = torch.randn((num_tokens, 4096), device=device, dtype=torch.float16)
    weight = torch.randn((6416, 4096), device=device, dtype=torch.float16)
    output = torch.empty((num_tokens, 6416), device=device, dtype=torch.float16)

    sm70_ops.sm70_glm53_fp16_gemv_out(output, input, weight)
    torch.accelerator.synchronize()
    expected_rows = []
    for token_idx in range(num_tokens):
        expected_row = torch.empty((1, 6416), device=device, dtype=torch.float16)
        sm70_ops.sm70_glm53_fp16_gemv_out(
            expected_row, input[token_idx : token_idx + 1], weight
        )
        expected_rows.append(expected_row)
    expected = torch.cat(expected_rows)
    assert torch.equal(output, expected)
    if num_tokens == 1:
        assert torch.equal(output, F.linear(input, weight))

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sm70_ops.sm70_glm53_fp16_gemv_out(output, input, weight)
    input.copy_(torch.randn_like(input))
    expected_rows = []
    for token_idx in range(num_tokens):
        expected_row = torch.empty((1, 6416), device=device, dtype=torch.float16)
        sm70_ops.sm70_glm53_fp16_gemv_out(
            expected_row, input[token_idx : token_idx + 1], weight
        )
        expected_rows.append(expected_row)
    expected = torch.cat(expected_rows)
    graph.replay()
    torch.accelerator.synchronize()
    assert torch.equal(output, expected)


@pytest.mark.skipif(
    not (
        current_platform.is_cuda()
        and current_platform.get_device_capability() == DeviceCapability(7, 0)
        and hasattr(torch.ops._C, "sm70_glm53_fp16_gemv_out")
    ),
    reason="native NVIDIA V100/SM70 GLM-5.3 exact FP16 GEMV op required",
)
def test_sm70_glm53_fp16_gemv_swizzle_matches_baseline_and_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(20260902)
    device = current_platform.device_type
    input = torch.randn((8, 4096), device=device, dtype=torch.float16)
    weight = torch.randn((6416, 4096), device=device, dtype=torch.float16)
    half2_output = torch.empty((8, 6416), device=device, dtype=torch.float16)
    swizzled_output = torch.empty_like(half2_output)

    monkeypatch.setenv("VLLM_SM70_GLM53_EXACT_KDA_HALF2_ROWS", "-2")
    sm70_ops.sm70_glm53_fp16_gemv_out(half2_output, input, weight)
    monkeypatch.setenv("VLLM_SM70_GLM53_EXACT_KDA_HALF2_ROWS", "-3")
    sm70_ops.sm70_glm53_fp16_gemv_out(swizzled_output, input, weight)
    torch.accelerator.synchronize()
    assert torch.equal(swizzled_output, half2_output)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sm70_ops.sm70_glm53_fp16_gemv_out(swizzled_output, input, weight)
    input.copy_(torch.randn_like(input))
    monkeypatch.setenv("VLLM_SM70_GLM53_EXACT_KDA_HALF2_ROWS", "-2")
    sm70_ops.sm70_glm53_fp16_gemv_out(half2_output, input, weight)
    monkeypatch.setenv("VLLM_SM70_GLM53_EXACT_KDA_HALF2_ROWS", "-3")
    graph.replay()
    torch.accelerator.synchronize()
    assert torch.equal(swizzled_output, half2_output)


@pytest.mark.skipif(
    not (
        current_platform.is_cuda()
        and current_platform.get_device_capability() == DeviceCapability(7, 0)
    ),
    reason="NVIDIA V100/SM70 KDA precision regression test",
)
def test_sm70_glm_kda_fp32_staging_avoids_fp16_overflow_and_graph() -> None:
    device = torch.device(current_platform.device_type)
    tokens, heads, dim = 8, 1, 128
    q = torch.ones(1, tokens, heads, dim, device=device, dtype=torch.float16)
    k = torch.ones_like(q)
    v = torch.zeros_like(q)
    gate = torch.zeros_like(q)
    beta = torch.full((1, tokens, heads), -20.0, device=device, dtype=torch.float16)
    a_log = torch.zeros(1, 1, heads, 1, device=device, dtype=torch.float32)
    g_bias = torch.zeros(heads * dim, device=device, dtype=torch.float32)
    base_state = torch.full(
        (tokens, heads, dim, dim), 1e6, device=device, dtype=torch.float32
    )
    query_start = torch.tensor([0, tokens], device=device, dtype=torch.int32)
    state_indices = torch.arange(tokens, device=device, dtype=torch.int32).unsqueeze(0)
    accepted = torch.ones(1, device=device, dtype=torch.int32)
    common = dict(
        q=q,
        k=k,
        v=v,
        g=gate,
        beta=beta,
        cu_seqlens=query_start,
        ssm_state_indices=state_indices,
        num_accepted_tokens=accepted,
        sigmoid_beta=True,
        a_log=a_log,
        g_bias=g_bias,
        compute_gate=True,
        lower_bound=-5.0,
    )

    fp16_output = torch.empty_like(k)
    fused_recurrent_kda(initial_state=base_state.clone(), out=fp16_output, **common)
    assert not torch.isfinite(fp16_output).all()

    state = base_state.clone()
    fp32_output = torch.empty_like(k, dtype=torch.float32)
    norm_gate = torch.zeros_like(k)
    norm_weight = torch.ones(dim, device=device, dtype=torch.float16)

    def run() -> torch.Tensor:
        state.copy_(base_state)
        fused_recurrent_kda(initial_state=state, out=fp32_output, **common)
        normalized, _, _, _ = layer_norm_gated_fwd(
            x=fp32_output.reshape(-1, dim),
            g=norm_gate.reshape(-1, dim),
            weight=norm_weight,
            bias=None,
            activation="sigmoid",
            eps=1e-5,
            out_dtype=torch.float16,
            is_rms_norm=True,
        )
        return normalized

    normalized = run()
    torch.accelerator.synchronize()
    assert fp32_output.abs().max() > torch.finfo(torch.float16).max
    assert torch.isfinite(normalized).all()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = run()
    graph.replay()
    torch.accelerator.synchronize()
    assert torch.isfinite(graph_output).all()
    assert torch.equal(graph_output, normalized)
