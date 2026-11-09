# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block-FP8 linears on SM70/SM75 through the native QPN8 operators."""

from types import SimpleNamespace

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


@pytest.mark.parametrize("m", [9, 64, 512])
def test_volta_without_turbomind_copy_keeps_one_layout(default_vllm_config, m):
    # A 32 GiB V100 pipeline stage of DeepSeek-V4 runs out of memory when every
    # block FP8 weight is held twice; without the copy, rows beyond M=8 take
    # the dense prefill operator that Turing uses.
    default_vllm_config.kernel_config.sm70_fp8.block_qpn8_volta_turbomind_prefill = (
        False
    )
    layer, reference = _layer(1536, 4096)
    kernel = QPN8Fp8BlockScaledMMLinearKernel(_config(1536, 4096))
    kernel.process_weights_after_loading(layer)
    assert not hasattr(layer, "_sm70_block_fp8_turbomind_packed_weight")
    assert not hasattr(layer, "_qpn8_fallback_k_ld")
    x = torch.randn(m, 4096, device="cuda", dtype=torch.float16) * 0.1
    y = kernel.apply_weights(layer, x)
    expected = x.float() @ reference.t()
    assert ((y.float() - expected).norm() / expected.norm()).item() < 1e-3


@pytest.mark.parametrize("turbomind_copy", [True, False])
def test_volta_warmup_follows_the_held_layout(default_vllm_config, turbomind_copy):
    # The coordinated Volta warmup used to read TurboMind attributes that block
    # QPN8 layers never carry and failed the engine start.
    if torch.cuda.get_device_capability()[0:2] not in ((7, 0), (7, 2)):
        pytest.skip("the coordinated dense warmup runs on Volta only")
    from vllm.model_executor.warmup.awq_sm70_warmup import (
        _iter_unique_fp8_dense_layers,
        _warmup_fp8_dense_layers,
    )

    default_vllm_config.kernel_config.sm70_fp8.block_qpn8_volta_turbomind_prefill = (
        turbomind_copy
    )
    layer, _ = _layer(1536, 4096)
    QPN8Fp8BlockScaledMMLinearKernel(_config(1536, 4096)).process_weights_after_loading(
        layer
    )
    model = torch.nn.Module()
    model.proj = layer
    layers = list(_iter_unique_fp8_dense_layers(model))
    assert layers == ([(layer, False)] if turbomind_copy else [])
    # Only rows beyond the native QPN8 bound use the TurboMind copy.
    assert _warmup_fp8_dense_layers(layers, [1, 8, 9, 64]) == (
        2 if turbomind_copy else 0
    )


@pytest.mark.parametrize(
    ("max_num_seqs", "speculative_tokens", "explicit", "copy"),
    [
        (1, 5, None, False),  # one request verifying six rows stays on QPN8
        (8, 0, None, False),
        (9, 0, None, True),
        (2, 5, None, True),  # twelve decode rows reach the TurboMind copy
        (1, 5, True, True),
        (4, 5, False, False),
    ],
)
def test_volta_turbomind_copy_follows_decode_rows(
    default_vllm_config, max_num_seqs, speculative_tokens, explicit, copy
):
    if torch.cuda.get_device_capability()[0:2] not in ((7, 0), (7, 2)):
        pytest.skip("only Volta holds the TurboMind copy")
    default_vllm_config.scheduler_config.max_num_seqs = max_num_seqs
    if speculative_tokens:
        default_vllm_config.speculative_config = SimpleNamespace(
            num_speculative_tokens=speculative_tokens
        )
    default_vllm_config.kernel_config.sm70_fp8.block_qpn8_volta_turbomind_prefill = (
        explicit
    )
    layer, _ = _layer(1536, 4096)
    QPN8Fp8BlockScaledMMLinearKernel(_config(1536, 4096)).process_weights_after_loading(
        layer
    )
    assert hasattr(layer, "_sm70_block_fp8_turbomind_packed_weight") == copy
