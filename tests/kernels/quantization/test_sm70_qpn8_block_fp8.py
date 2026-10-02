# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block-FP8 linears on SM70/SM75 through the native QPN8 operators."""

import pytest
import torch

from vllm.model_executor.kernels.linear.scaled_mm.qpn8_blk import (
    QPN8Fp8BlockScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
)
from vllm.utils.torch_utils import current_stream

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((7, 0), (7, 5)),
    reason="requires an SM70 or SM75 GPU",
)


def _config(n: int, k: int) -> FP8ScaledMMLinearLayerConfig:
    return FP8ScaledMMLinearLayerConfig(
        weight_quant_key=kFp8Static128BlockSym,
        activation_quant_key=kFp8Dynamic128Sym,
        weight_shape=(n, k),
        input_dtype=torch.float16,
        out_dtype=torch.float16,
    )


def _layer(n: int, k: int, scale_dtype=torch.float32):
    torch.manual_seed(n + k)
    raw = torch.randint(0, 256, (n, k), dtype=torch.uint8, device="cuda")
    # 0x7F/0xFF are e4m3 NaN codes; a checkpoint never stores them.
    raw[raw == 0x7F] = 0x7E
    raw[raw == 0xFF] = 0xFE
    weight = raw.view(torch.float8_e4m3fn)
    scales = (2.0 ** (torch.rand(n // 128, k // 128, device="cuda") * 3 - 12)).float()
    scales = scales.to(scale_dtype)
    full = scales.float().repeat_interleave(128, 0).repeat_interleave(128, 1)
    reference = weight.float() * full
    layer = torch.nn.Module()
    layer.prefix = f"test.{n}x{k}"
    layer.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
    layer.weight_scale_inv = torch.nn.Parameter(scales.clone(), requires_grad=False)
    return layer, reference


def test_admits_only_block_128_geometry():
    assert QPN8Fp8BlockScaledMMLinearKernel.can_implement(_config(1536, 4096))[0]
    assert not QPN8Fp8BlockScaledMMLinearKernel.can_implement(_config(1536, 4000))[0]
    assert not QPN8Fp8BlockScaledMMLinearKernel.can_implement(_config(1500, 4096))[0]


@pytest.mark.parametrize(
    ("n", "k"), [(128, 128), (768, 1408), (1536, 4096), (8192, 1024), (4096, 8192)]
)
@pytest.mark.parametrize("m", [0, 1, 6, 8, 9, 20, 64, 512])
@pytest.mark.parametrize("scale_dtype", [torch.float32, torch.float8_e8m0fnu])
def test_matches_dequantized_reference(
    default_vllm_config, n: int, k: int, m: int, scale_dtype
):
    layer, reference = _layer(n, k, scale_dtype)
    kernel = QPN8Fp8BlockScaledMMLinearKernel(_config(n, k))
    kernel.process_weights_after_loading(layer)
    # Raw checkpoint weights are released after packing.
    assert layer.weight.numel() == 0
    x = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.1
    y = kernel.apply_weights(layer, x)
    expected = x.float() @ reference.t()
    assert y.shape == (m, n) and y.dtype == torch.float16
    if m:
        assert ((y.float() - expected).norm() / expected.norm()).item() < 1e-3


@pytest.mark.parametrize("dense_only", [False, True])
def test_concurrent_streams_keep_their_own_prefill_buffer(
    default_vllm_config, dense_only
):
    # DeepSeek-V4 runs the indexer's wq_b on an aux stream next to the main
    # wq_b; with one shared dense buffer the prefill of one overwrote the
    # other's dequantized weight.
    layer_a, reference_a = _layer(8192, 1024)
    layer_b, reference_b = _layer(4096, 1024)
    kernel_a = QPN8Fp8BlockScaledMMLinearKernel(_config(8192, 1024))
    kernel_b = QPN8Fp8BlockScaledMMLinearKernel(_config(4096, 1024))
    kernel_a.process_weights_after_loading(layer_a)
    kernel_b.process_weights_after_loading(layer_b)
    if dense_only:
        for layer in (layer_a, layer_b):
            for name in (
                "_sm70_block_fp8_turbomind_packed_weight",
                "_sm70_block_fp8_turbomind_packed_scales",
            ):
                if hasattr(layer, name):
                    delattr(layer, name)
    x = torch.randn(256, 1024, device="cuda", dtype=torch.float16) * 0.1
    aux = torch.cuda.Stream()
    # vLLM's stream, as the model code takes it: leaving the aux context below
    # restores it, while torch's default stream would stay recorded as vLLM's
    # current stream and break later graph captures in this process.
    main = current_stream()
    aux.wait_stream(main)
    for _ in range(20):
        with torch.cuda.stream(aux):
            y_b = kernel_b.apply_weights(layer_b, x)
        y_a = kernel_a.apply_weights(layer_a, x)
    main.wait_stream(aux)
    torch.accelerator.synchronize()
    for y, reference in ((y_a, reference_a), (y_b, reference_b)):
        expected = x.float() @ reference.t()
        assert ((y.float() - expected).norm() / expected.norm()).item() < 1e-3


@pytest.mark.parametrize("m", [9, 64])
def test_volta_medium_prefill_keeps_original_turbomind_output(default_vllm_config, m):
    if torch.cuda.get_device_capability()[0:2] not in ((7, 0), (7, 2)):
        pytest.skip("Volta retains the TurboMind fallback layout")
    from vllm import _sm70_ops as ops

    layer, _ = _layer(1536, 4096)
    kernel = QPN8Fp8BlockScaledMMLinearKernel(_config(1536, 4096))
    kernel.process_weights_after_loading(layer)
    x = torch.randn(m, 4096, device="cuda", dtype=torch.float16)
    expected = torch.empty((m, 1536), device="cuda", dtype=torch.float16)
    ops.fp8_gemm_sm70_out(
        expected,
        x,
        layer._sm70_block_fp8_turbomind_packed_weight,
        layer._sm70_block_fp8_turbomind_packed_scales,
        128,
        layer._qpn8_fallback_k_ld,
        layer._qpn8_fallback_q_ld,
        False,
    )
    torch.testing.assert_close(kernel.apply_weights(layer, x), expected, rtol=0, atol=0)
