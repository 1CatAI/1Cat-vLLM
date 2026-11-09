# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

import vllm._sm70_ops as sm70_ops
from vllm import envs
from vllm.model_executor.kernels.mhc import tilelang as mhc_tilelang
from vllm.model_executor.kernels.mhc.torch import (
    mhc_fused_post_pre_torch,
    mhc_post_torch,
    mhc_pre_torch,
)
from vllm.model_executor.kernels.mhc.triton import sm70_mhc_pre_norm_from_staging
from vllm.model_executor.layers import mhc as mhc_layer
from vllm.platforms.interface import DeviceCapability
from vllm.utils.import_utils import has_tilelang


class _FakeCudaPlatform:
    def __init__(self, supports_bf16: bool) -> None:
        self.supports_bf16 = supports_bf16

    def is_cuda(self) -> bool:
        return True

    def get_device_capability(self) -> DeviceCapability:
        return DeviceCapability(8 if self.supports_bf16 else 7, 0)


def test_mhc_sm70_requires_explicit_fp16(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mhc_tilelang, "current_platform", _FakeCudaPlatform(False))

    fp16 = torch.empty((1,), dtype=torch.float16)
    assert mhc_tilelang._require_mhc_activation_dtype(fp16) == torch.float16

    with pytest.raises(RuntimeError, match="SM70 has no native BF16"):
        mhc_tilelang._require_mhc_activation_dtype(
            torch.empty((1,), dtype=torch.bfloat16)
        )


def test_mhc_bf16_is_preserved_on_supported_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mhc_tilelang, "current_platform", _FakeCudaPlatform(True))

    bf16 = torch.empty((1,), dtype=torch.bfloat16)
    assert mhc_tilelang._require_mhc_activation_dtype(bf16) == torch.bfloat16


def test_mhc_fp16_rejects_implicit_norm_weight_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mhc_tilelang, "current_platform", _FakeCudaPlatform(False))

    with pytest.raises(ValueError, match="FP16 requires norm_weight"):
        mhc_tilelang._prepare_mhc_norm_weight(
            torch.empty((1,), dtype=torch.bfloat16), torch.float16
        )


def test_mhc_native_bf16_preserves_norm_weight_conversion() -> None:
    prepared = mhc_tilelang._prepare_mhc_norm_weight(
        torch.empty((1,), dtype=torch.float32), torch.bfloat16
    )
    assert prepared is not None
    assert prepared.dtype == torch.bfloat16


def test_mhc_fake_paths_preserve_fp16_graph_metadata() -> None:
    residual = torch.empty((2, 4, 128), dtype=torch.float16, device="meta")
    x = torch.empty((2, 128), dtype=torch.float16, device="meta")
    fn = torch.empty((24, 512), dtype=torch.float32, device="meta")
    scale = torch.empty((3,), dtype=torch.float32, device="meta")
    base = torch.empty((24,), dtype=torch.float32, device="meta")
    post = torch.empty((2, 4, 1), dtype=torch.float32, device="meta")
    comb = torch.empty((2, 4, 4), dtype=torch.float32, device="meta")

    pre = torch.ops.vllm.mhc_pre_tilelang(
        residual, fn, scale, base, 1e-6, 1e-6, 1e-6, 1.0, 2
    )
    broadcast = torch.ops.vllm.mhc_pre_broadcast_tilelang(
        x,
        fn,
        scale,
        base,
        1e-6,
        1e-6,
        1e-6,
        1.0,
        2,
        1,
        torch.empty((128,), dtype=torch.float16, device="meta"),
        1e-6,
        torch.empty((24, 128), dtype=torch.float32, device="meta"),
    )
    fused = torch.ops.vllm.mhc_fused_post_pre_tilelang(
        x, residual, post, comb, fn, scale, base, 1e-6, 1e-6, 1e-6, 1.0, 2
    )
    head = torch.ops.vllm.hc_head_fused_kernel_tilelang(
        residual,
        torch.empty((4, 512), dtype=torch.float32, device="meta"),
        torch.empty((1,), dtype=torch.float32, device="meta"),
        torch.empty((4,), dtype=torch.float32, device="meta"),
        1e-6,
        1e-6,
    )

    assert pre[2].dtype == torch.float16
    assert pre[2].shape == (2, 128)
    assert broadcast[0].shape == (2, 4, 128)
    assert broadcast[3].dtype == torch.float16
    assert fused[0].dtype == torch.float16
    assert fused[3].dtype == torch.float16
    assert fused[3].shape == (2, 128)
    assert head.dtype == torch.float16
    assert head.shape == (2, 128)


def test_mhc_fp16_fake_path_compiles_fullgraph() -> None:
    def run(
        residual: torch.Tensor,
        fn: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.ops.vllm.mhc_pre_tilelang(
            residual, fn, scale, base, 1e-6, 1e-6, 1e-6, 1.0, 2
        )

    compiled = torch.compile(run, backend="eager", fullgraph=True)
    output = compiled(
        torch.empty((2, 4, 128), dtype=torch.float16, device="meta"),
        torch.empty((24, 512), dtype=torch.float32, device="meta"),
        torch.empty((3,), dtype=torch.float32, device="meta"),
        torch.empty((24,), dtype=torch.float32, device="meta"),
    )

    assert output[2].dtype == torch.float16
    assert output[2].shape == (2, 128)


def test_mhc_broadcast_fake_path_compiles_fullgraph() -> None:
    def run(
        residual: torch.Tensor,
        fn: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
        norm_weight: torch.Tensor,
        fn_broadcast: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.ops.vllm.mhc_pre_broadcast_tilelang(
            residual,
            fn,
            scale,
            base,
            1e-6,
            1e-6,
            1e-6,
            1.0,
            2,
            1,
            norm_weight,
            1e-6,
            fn_broadcast,
        )

    compiled = torch.compile(run, backend="eager", fullgraph=True)
    output = compiled(
        torch.empty((2, 128), dtype=torch.float16, device="meta"),
        torch.empty((24, 512), dtype=torch.float32, device="meta"),
        torch.empty((3,), dtype=torch.float32, device="meta"),
        torch.empty((24,), dtype=torch.float32, device="meta"),
        torch.empty((128,), dtype=torch.float16, device="meta"),
        torch.empty((24, 128), dtype=torch.float32, device="meta"),
    )

    assert output[0].shape == (2, 4, 128)
    assert output[3].dtype == torch.float16


def test_mhc_torch_fp16_fused_prefill_matches_fp32_reference() -> None:
    torch.manual_seed(17)
    num_tokens = 17
    hidden_size = 16
    hc_mult = 4
    hc_mult3 = 24
    residual = torch.randn(num_tokens, hc_mult, hidden_size, dtype=torch.float16)
    x = torch.randn(num_tokens, hidden_size, dtype=torch.float16)
    post = torch.randn(num_tokens, hc_mult, 1, dtype=torch.float32)
    comb = torch.randn(num_tokens, hc_mult, hc_mult, dtype=torch.float32)
    fn = torch.randn(hc_mult3, hc_mult * hidden_size, dtype=torch.float32)
    scale = torch.tensor([0.5, 0.75, 0.25], dtype=torch.float32)
    base = torch.randn(hc_mult3, dtype=torch.float32)
    norm_weight = torch.randn(hidden_size, dtype=torch.float16)

    output = mhc_fused_post_pre_torch(
        x,
        residual,
        post,
        comb,
        fn,
        scale,
        base,
        1e-6,
        1e-6,
        1e-6,
        1.0,
        3,
        norm_weight=norm_weight,
        norm_eps=1e-6,
    )
    residual_ref = (
        torch.einsum("bij,bih->bjh", comb, residual.float())
        + post * x.float().unsqueeze(1)
    ).half()
    pre_ref = mhc_pre_torch(
        residual_ref,
        fn,
        scale,
        base,
        1e-6,
        1e-6,
        1e-6,
        1.0,
        3,
        norm_weight=norm_weight,
        norm_eps=1e-6,
    )

    torch.testing.assert_close(output[0], residual_ref, rtol=0, atol=0)
    for actual, expected in zip(output[1:], pre_ref):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert output[-1].dtype == torch.float16


def test_mhc_sm70_fallback_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mhc_layer, "current_platform", _FakeCudaPlatform(False))
    decode = torch.empty((1, 4, 4096), dtype=torch.float16)
    prefill = torch.empty((17, 4, 4096), dtype=torch.float16)
    unsupported = torch.empty((1, 4, 8), dtype=torch.float16)

    assert not mhc_layer._use_sm70_fp16_mhc_fallback(decode, prefill_only=True)
    assert not mhc_layer._use_sm70_fp16_mhc_fallback(prefill, prefill_only=True)
    assert not mhc_layer._use_sm70_fp16_mhc_fallback(decode, prefill_only=False)
    assert mhc_layer._use_sm70_fp16_mhc_fallback(unsupported, prefill_only=False)


@pytest.mark.skipif(
    not (
        mhc_tilelang.current_platform.is_cuda()
        and mhc_tilelang.current_platform.get_device_capability()
        == DeviceCapability(7, 0)
        and hasattr(torch.ops._C, "sm70_glm_mhc_pre_norm_out")
    ),
    reason="native NVIDIA V100/SM70 mHC CUDA op required",
)
@pytest.mark.parametrize("num_tokens", [1, 8])
def test_mhc_sm70_native_final_stage_graph_matches_eager_bitwise(
    monkeypatch: pytest.MonkeyPatch,
    num_tokens: int,
) -> None:
    torch.manual_seed(20260831)
    device = mhc_tilelang.current_platform.device_type
    n_splits = 8
    gemm_mul = torch.randn(
        (n_splits, num_tokens, 24), device=device, dtype=torch.float32
    ).mul_(0.01)
    gemm_sqrsum = torch.rand(
        (n_splits, num_tokens), device=device, dtype=torch.float32
    ).add_(4096)
    scale = torch.tensor([0.5, 0.75, 0.25], device=device, dtype=torch.float32)
    base = torch.randn((24,), device=device, dtype=torch.float32).mul_(0.1)
    residual = torch.randn(
        (num_tokens, 4, 4096), device=device, dtype=torch.float16
    ).mul_(0.1)
    norm_weight = torch.randn((4096,), device=device, dtype=torch.float16).mul_(0.1)
    monkeypatch.setenv("VLLM_SM70_GLM53_MHC_NATIVE_VERIFY", "1")
    envs.disable_envs_cache()

    def run(
        post_mix: torch.Tensor,
        comb_mix: torch.Tensor,
        layer_input: torch.Tensor,
    ) -> None:
        sm70_mhc_pre_norm_from_staging(
            gemm_mul,
            gemm_sqrsum,
            scale,
            base,
            residual,
            post_mix,
            comb_mix,
            layer_input,
            norm_weight,
            1e-6,
            1e-6,
            1e-6,
            1.0,
            20,
            1e-6,
        )

    eager_outputs = (
        torch.empty((num_tokens, 4), device=device, dtype=torch.float32),
        torch.empty((num_tokens, 4, 4), device=device, dtype=torch.float32),
        torch.empty((num_tokens, 4096), device=device, dtype=torch.float16),
    )
    run(*eager_outputs)
    expected = tuple(output.clone() for output in eager_outputs)

    graph_outputs = tuple(torch.empty_like(output) for output in eager_outputs)
    run(*graph_outputs)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(*graph_outputs)
    graph.replay()
    torch.accelerator.synchronize()

    for output, reference in zip(graph_outputs, expected):
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
    envs.disable_envs_cache()


@pytest.mark.skipif(
    not (
        mhc_tilelang.current_platform.is_cuda()
        and has_tilelang()
        and mhc_tilelang.current_platform.get_device_capability()
        == DeviceCapability(7, 0)
        and hasattr(torch.ops._C, "sm70_glm_mhc_post_dot_q8_out")
    ),
    reason="native NVIDIA V100/SM70 mHC post+dot op and TileLang required",
)
def test_mhc_sm70_native_q8_post_dot_matches_fp32_stage_bitwise() -> None:
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        sm70_mhc_dot_from_fp32_stage_tilelang,
        sm70_mhc_post_fp32_stage_tilelang,
    )

    torch.manual_seed(20260902)
    device = mhc_tilelang.current_platform.device_type
    num_tokens = 8
    hidden_size = 4096
    hc_mult = 4
    hc_out = 24
    n_splits = 4
    tile_n = 6
    x = torch.randn((num_tokens, hidden_size), device=device, dtype=torch.float16)
    residual = torch.randn(
        (num_tokens, hc_mult, hidden_size), device=device, dtype=torch.float16
    )
    post_mix = torch.sigmoid(
        torch.randn((num_tokens, hc_mult), device=device, dtype=torch.float32)
    )
    comb_mix = torch.softmax(
        torch.randn((num_tokens, hc_mult, hc_mult), device=device, dtype=torch.float32),
        dim=1,
    )
    fn = torch.randn(
        (hc_out, hc_mult, hidden_size), device=device, dtype=torch.float32
    ).mul_(1e-4)

    staged_fp32 = torch.empty_like(residual, dtype=torch.float32)
    staged_residual = torch.empty_like(residual)
    staged_gemm = torch.empty(
        (n_splits, num_tokens, hc_out), device=device, dtype=torch.float32
    )
    staged_sqrsum = torch.empty(
        (n_splits, num_tokens), device=device, dtype=torch.float32
    )
    sm70_mhc_post_fp32_stage_tilelang(
        comb_mix,
        residual,
        post_mix,
        x,
        staged_fp32,
        staged_residual,
        staged_sqrsum,
        hidden_size,
        hc_mult,
        n_splits=n_splits,
    )
    sm70_mhc_dot_from_fp32_stage_tilelang(
        staged_fp32,
        fn,
        staged_gemm,
        hidden_size,
        hc_mult,
        hc_out,
        tile_n=tile_n,
        n_splits=n_splits,
    )

    native_residual = torch.empty_like(staged_residual)
    native_gemm = torch.empty_like(staged_gemm)
    native_sqrsum = torch.empty_like(staged_sqrsum)
    sm70_ops.sm70_glm_mhc_post_dot_q8_out(
        native_residual,
        native_gemm,
        native_sqrsum,
        comb_mix,
        residual,
        post_mix,
        x,
        fn,
        tile_n,
    )
    torch.accelerator.synchronize()

    torch.testing.assert_close(native_residual, staged_residual, rtol=0, atol=0)
    torch.testing.assert_close(native_gemm, staged_gemm, rtol=0, atol=0)
    torch.testing.assert_close(native_sqrsum, staged_sqrsum, rtol=0, atol=0)


@pytest.mark.parametrize("num_tokens", [1, 4, 16, 17])
@pytest.mark.skipif(
    not (
        mhc_tilelang.current_platform.is_cuda()
        and has_tilelang()
        and mhc_tilelang.current_platform.get_device_capability()
        == DeviceCapability(7, 0)
    ),
    reason="NVIDIA V100/SM70 and TileLang required",
)
def test_mhc_sm70_fused_fast_paths_match_torch(num_tokens: int) -> None:
    torch.manual_seed(20260827 + num_tokens)
    hidden_size = 4096
    hc_mult = 4
    device = mhc_tilelang.current_platform.device_type

    x = torch.randn((num_tokens, hidden_size), device=device, dtype=torch.float16).mul_(
        0.1
    )
    residual = torch.randn(
        (num_tokens, hc_mult, hidden_size), device=device, dtype=torch.float16
    ).mul_(0.1)
    post_mix = torch.sigmoid(
        torch.randn((num_tokens, hc_mult, 1), device=device, dtype=torch.float32)
    )
    comb_mix = torch.softmax(
        torch.randn((num_tokens, hc_mult, hc_mult), device=device, dtype=torch.float32),
        dim=-1,
    )
    fn = torch.randn(
        (24, hc_mult * hidden_size), device=device, dtype=torch.float32
    ).mul_(0.01)
    scale = torch.tensor([0.5, 0.75, 0.25], device=device, dtype=torch.float32)
    base = torch.randn((24,), device=device, dtype=torch.float32).mul_(0.1)
    norm_weight = torch.randn((hidden_size,), device=device, dtype=torch.float16).mul_(
        0.1
    )

    actual = mhc_tilelang.mhc_fused_post_pre_tilelang(
        x,
        residual,
        post_mix,
        comb_mix,
        fn,
        scale,
        base,
        1e-6,
        1e-6,
        1e-6,
        1.0,
        20,
        norm_weight=norm_weight,
        norm_eps=1e-6,
    )
    expected = mhc_fused_post_pre_torch(
        x,
        residual,
        post_mix,
        comb_mix,
        fn,
        scale,
        base,
        1e-6,
        1e-6,
        1e-6,
        1.0,
        20,
        norm_weight=norm_weight,
        norm_eps=1e-6,
    )

    tolerances = ((2e-3, 2e-3), (2e-3, 2e-4), (3e-3, 3e-4), (5e-3, 5e-4))
    for output, reference, (rtol, atol) in zip(actual, expected, tolerances):
        torch.testing.assert_close(output, reference, rtol=rtol, atol=atol)


@pytest.mark.parametrize("num_tokens", [1, 17])
@pytest.mark.skipif(
    not (
        mhc_tilelang.current_platform.is_cuda()
        and has_tilelang()
        and mhc_tilelang.current_platform.get_device_capability()
        == DeviceCapability(7, 0)
    ),
    reason="NVIDIA V100/SM70 and TileLang required",
)
def test_mhc_sm70_standalone_pre_post_fast_paths_match_torch(
    num_tokens: int,
) -> None:
    torch.manual_seed(20260828 + num_tokens)
    hidden_size = 4096
    hc_mult = 4
    device = mhc_tilelang.current_platform.device_type
    x = torch.randn((num_tokens, hidden_size), device=device, dtype=torch.float16).mul_(
        0.1
    )
    residual = torch.randn(
        (num_tokens, hc_mult, hidden_size), device=device, dtype=torch.float16
    ).mul_(0.1)
    post_mix = torch.sigmoid(
        torch.randn((num_tokens, hc_mult, 1), device=device, dtype=torch.float32)
    )
    comb_mix = torch.softmax(
        torch.randn((num_tokens, hc_mult, hc_mult), device=device, dtype=torch.float32),
        dim=-1,
    )
    fn = torch.randn(
        (24, hc_mult * hidden_size), device=device, dtype=torch.float32
    ).mul_(0.01)
    scale = torch.tensor([0.5, 0.75, 0.25], device=device, dtype=torch.float32)
    base = torch.randn((24,), device=device, dtype=torch.float32).mul_(0.1)
    norm_weight = torch.randn((hidden_size,), device=device, dtype=torch.float16).mul_(
        0.1
    )

    actual_post = mhc_tilelang.mhc_post_tilelang(x, residual, post_mix, comb_mix)
    expected_post = mhc_post_torch(x, residual, post_mix, comb_mix)
    actual_pre = mhc_tilelang.mhc_pre_tilelang(
        residual,
        fn,
        scale,
        base,
        1e-6,
        1e-6,
        1e-6,
        1.0,
        20,
        norm_weight=norm_weight,
        norm_eps=1e-6,
    )
    expected_pre = mhc_pre_torch(
        residual,
        fn,
        scale,
        base,
        1e-6,
        1e-6,
        1e-6,
        1.0,
        20,
        norm_weight=norm_weight,
        norm_eps=1e-6,
    )

    torch.testing.assert_close(actual_post, expected_post, rtol=2e-3, atol=2e-4)
    tolerances = ((2e-3, 2e-4), (3e-3, 3e-4), (5e-3, 1e-3))
    for output, reference, (rtol, atol) in zip(actual_pre, expected_pre, tolerances):
        torch.testing.assert_close(output, reference, rtol=rtol, atol=atol)


@pytest.mark.parametrize("num_tokens", [1, 17])
@pytest.mark.skipif(
    not (
        mhc_tilelang.current_platform.is_cuda()
        and has_tilelang()
        and mhc_tilelang.current_platform.get_device_capability()
        == DeviceCapability(7, 0)
    ),
    reason="NVIDIA V100/SM70 and TileLang required",
)
def test_mhc_sm70_broadcast_pre_fast_path_matches_expanded_torch(
    num_tokens: int,
) -> None:
    torch.manual_seed(20260830 + num_tokens)
    hidden_size = 4096
    hc_mult = 4
    device = mhc_tilelang.current_platform.device_type
    x = torch.randn((num_tokens, hidden_size), device=device, dtype=torch.float16).mul_(
        0.1
    )
    residual = x.unsqueeze(1).expand(-1, hc_mult, -1).contiguous()
    fn = torch.randn(
        (24, hc_mult * hidden_size), device=device, dtype=torch.float32
    ).mul_(0.01)
    fn_broadcast = fn.view(24, hc_mult, hidden_size).sum(dim=1)
    scale = torch.tensor([0.5, 0.75, 0.25], device=device, dtype=torch.float32)
    base = torch.randn((24,), device=device, dtype=torch.float32).mul_(0.1)
    norm_weight = torch.randn((hidden_size,), device=device, dtype=torch.float16).mul_(
        0.1
    )

    actual = mhc_tilelang.mhc_pre_broadcast_tilelang(
        x,
        fn,
        scale,
        base,
        1e-6,
        1e-6,
        1e-6,
        1.0,
        20,
        norm_weight=norm_weight,
        norm_eps=1e-6,
        fn_broadcast=fn_broadcast,
    )
    expected = mhc_pre_torch(
        residual,
        fn,
        scale,
        base,
        1e-6,
        1e-6,
        1e-6,
        1.0,
        20,
        norm_weight=norm_weight,
        norm_eps=1e-6,
    )

    torch.testing.assert_close(actual[0], residual, rtol=0, atol=0)
    tolerances = ((2e-3, 2e-4), (3e-3, 3e-4), (5e-3, 1e-3))
    for output, reference, (rtol, atol) in zip(actual[1:], expected, tolerances):
        torch.testing.assert_close(output, reference, rtol=rtol, atol=atol)


@pytest.mark.skipif(
    not (mhc_tilelang.current_platform.is_cuda() and has_tilelang())
    or mhc_tilelang.current_platform.get_device_capability() == DeviceCapability(7, 0),
    reason="SM70 uses the FP16 torch fallback; native TileLang CUDA required",
)
def test_mhc_fp16_tilelang_block_m_prenorm_keeps_rows_independent() -> None:
    """Exercise the >=1024-token block-M path with non-identical row pairs."""
    hidden_size = 128
    hc_mult = 4
    num_tokens = 1024
    hc_hidden_size = hc_mult * hidden_size
    n_out = 2 * hc_mult + hc_mult * hc_mult
    device = mhc_tilelang.current_platform.device_type

    # The prior scalar reduction could carry the first row into the second.
    # Exact integer sums make that failure unambiguous: 512/1536, not 512/2048.
    x = torch.ones((num_tokens, hc_hidden_size), dtype=torch.float16, device=device)
    x[1::2].fill_(3)
    fn = torch.ones((n_out, hc_hidden_size), dtype=torch.float32, device=device)
    out = torch.empty((1, num_tokens, n_out), dtype=torch.float32, device=device)
    sqrsum = torch.empty((1, num_tokens), dtype=torch.float32, device=device)
    out_ref = torch.empty_like(out)
    sqrsum_ref = torch.empty_like(sqrsum)

    mhc_tilelang._torch_hc_prenorm_gemm(x, fn, out_ref, sqrsum_ref)
    mhc_tilelang._tilelang_hc_prenorm_gemm(x, fn, out, sqrsum, hidden_size, hc_mult)

    torch.testing.assert_close(out, out_ref, rtol=0, atol=0)
    torch.testing.assert_close(sqrsum, sqrsum_ref, rtol=0, atol=0)


@pytest.mark.skipif(
    not (
        mhc_tilelang.current_platform.is_cuda()
        and has_tilelang()
        and mhc_tilelang.current_platform.get_device_capability()
        == DeviceCapability(7, 0)
    ),
    reason="NVIDIA V100/SM70 and TileLang required",
)
@pytest.mark.parametrize("num_tokens", [1, 4, 7, 8])
def test_mhc_sm70_fp32_stage_matches_fused_decode_bitwise(
    num_tokens: int,
) -> None:
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        mhc_fused_tilelang,
        sm70_mhc_dot_from_fp32_stage_tilelang,
        sm70_mhc_post_fp32_stage_tilelang,
    )

    torch.manual_seed(20260803)
    hidden_size = 4096
    hc_mult = 4
    hc_out = 24
    tile_n = 2 if num_tokens < 8 else 12
    n_splits = 8 if num_tokens < 8 else 4
    device = mhc_tilelang.current_platform.device_type

    x = torch.randn((num_tokens, hidden_size), device=device, dtype=torch.float16)
    residual = torch.randn(
        (num_tokens, hc_mult, hidden_size), device=device, dtype=torch.float16
    )
    post_mix = torch.randn((num_tokens, hc_mult), device=device, dtype=torch.float32)
    comb_mix = torch.randn(
        (num_tokens, hc_mult, hc_mult), device=device, dtype=torch.float32
    )
    fn = torch.randn((hc_out, hc_mult, hidden_size), device=device, dtype=torch.float32)

    baseline_gemm = torch.empty(
        (n_splits, num_tokens, hc_out), device=device, dtype=torch.float32
    )
    baseline_sqrsum = torch.empty(
        (n_splits, num_tokens), device=device, dtype=torch.float32
    )
    baseline_residual = torch.empty_like(residual)
    candidate_gemm = torch.empty_like(baseline_gemm)
    candidate_sqrsum = torch.empty_like(baseline_sqrsum)
    candidate_residual = torch.empty_like(residual)
    candidate_residual_fp32 = torch.empty_like(residual, dtype=torch.float32)

    mhc_fused_tilelang(
        comb_mix,
        residual,
        post_mix,
        x,
        fn,
        baseline_gemm,
        baseline_sqrsum,
        baseline_residual,
        hc_mult,
        hidden_size,
        hc_out,
        tile_n=tile_n,
        n_splits=n_splits,
        use_fp16=True,
    )
    sm70_mhc_post_fp32_stage_tilelang(
        comb_mix,
        residual,
        post_mix,
        x,
        candidate_residual_fp32,
        candidate_residual,
        candidate_sqrsum,
        hidden_size,
        hc_mult,
        n_splits=n_splits,
    )
    sm70_mhc_dot_from_fp32_stage_tilelang(
        candidate_residual_fp32,
        fn,
        candidate_gemm,
        hidden_size,
        hc_mult,
        hc_out,
        tile_n=tile_n,
        n_splits=n_splits,
    )
    torch.accelerator.synchronize()

    torch.testing.assert_close(candidate_residual, baseline_residual, rtol=0, atol=0)
    torch.testing.assert_close(candidate_gemm, baseline_gemm, rtol=0, atol=0)
    torch.testing.assert_close(candidate_sqrsum, baseline_sqrsum, rtol=0, atol=0)
