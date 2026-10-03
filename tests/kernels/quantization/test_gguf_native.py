# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib

import numpy as np
import pytest
import torch

from vllm.model_executor.kernels.gguf import admit_moe_fallback
from vllm.model_executor.layers.quantization.gguf_native import pad_weight_tail
from vllm.transformers_utils.gguf_tensor_reader import dequantize, quant_size

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def fixture(weight_type, rows=16):
    block, size = quant_size(weight_type)
    k = 640 if weight_type == 42 else 256
    rng = np.random.default_rng(20261003 + weight_type)
    blocks = rng.integers(0, 256, size=(rows * k // block, size), dtype=np.uint8)
    scale = np.frombuffer(np.float16(0.125).tobytes(), dtype=np.uint8)
    if weight_type in (34, 35):
        blocks[:, -2:] = scale
    elif weight_type == 39:
        blocks[:, 0] = 127
    elif weight_type == 40:
        blocks[:, :4] = 0x38
    else:
        blocks[:, :2] = scale
        if weight_type == 12:
            blocks[:, 2:4] = scale
    packed = blocks.reshape(rows, -1)
    if weight_type == 41:
        codes = np.unpackbits(blocks[:, 2:], axis=-1, bitorder="little").astype(
            np.int16
        )
        dense = ((codes * 2 - 1) * np.float32(0.125)).reshape(rows, k)
    else:
        dense = dequantize(packed, weight_type)
    return packed, dense


def load_native():
    try:
        importlib.import_module("vllm._C_gguf")
    except ImportError as error:
        pytest.skip(f"Packaged native GGUF extension unavailable: {error}")
    return torch.ops._C_gguf


@pytest.mark.parametrize("weight_type", [42, 41, 39, 40, 34, 35, 18, 12])
def test_native_dequant_and_dense_projection_against_cpu(weight_type):
    native = load_native()
    packed, reference = fixture(weight_type)
    weight = torch.from_numpy(packed).cuda()
    rows, k = reference.shape
    decoded = native.ggml_dequantize_upstream(
        weight, weight_type, rows, k, torch.float32
    )
    torch.testing.assert_close(
        decoded.cpu(), torch.from_numpy(reference), rtol=1e-6, atol=0
    )
    padded = pad_weight_tail(weight, weight_type)
    bank = pad_weight_tail(torch.stack((weight, weight)), weight_type)
    admitted = admit_moe_fallback(bank, weight_type, torch.float16)
    assert admitted.supports_m(1) and admitted.supports_m(8192)
    if weight_type in (34, 35):
        assert admitted.operator == "ggml_moe_grouped_dense"
        assert not admitted.graph_safe
        assert admitted.reason == "graph_safe_moe_operator_unavailable"
    for m in (1, 8, 32):
        generator = torch.Generator().manual_seed(20261003 + m)
        x = (torch.randn(m, k, generator=generator) * 0.125).half().cuda()
        caps = native.ggml_dense_upstream_capabilities(padded, x, weight_type, rows)
        assert caps & 16
        if weight_type in (34, 35):
            assert not caps & 4
            assert not native.ggml_should_use_mmvq(weight_type, 700, m)
        output = native.ggml_dense_blas(padded, x, weight_type, rows)
        # Quantized weights are dequantized to FP16 before tensor-core GEMM.
        expected = x.float().cpu() @ torch.from_numpy(reference).half().float().T
        torch.testing.assert_close(output.float().cpu(), expected, rtol=1e-3, atol=3e-3)
        if m == 1 and caps & 4:
            mmvq = native.ggml_dense_mmvq(padded, x, weight_type, rows)
            # MMVQ uses Q8 activation quantization; it has a distinct numerical
            # contract from the FP16 prefill path.
            error = (mmvq.float().cpu() - expected).norm()
            assert error <= expected.norm() * 0.025 + 0.02


def test_cpu_loaded_mixed_shards_are_prepared_on_the_worker_gpu():
    from vllm.model_executor.layers.quantization.gguf import (
        GGUFConfig,
        GGUFLinearMethod,
    )

    load_native()
    packed, reference = fixture(12)
    layer = torch.nn.Module()
    method = GGUFLinearMethod(GGUFConfig())
    k = reference.shape[1]
    with torch.device("cuda"):
        method.create_weights(layer, k, [2, 16], k, 18, torch.float16)
    floating = torch.ones((2, k), dtype=torch.float16)
    layer.qweight.data_container.extend([torch.from_numpy(packed), floating])
    layer.qweight.shard_id.extend([1, 0])
    layer.qweight.shard_id_map.update({1: 0, 0: 1})
    layer.qweight_type.shard_weight_type.update({1: 12, 0: 1})
    method.process_weights_after_loading(layer)
    assert all(weight.is_cuda for weight in layer.gguf_native_shard_weights)
    x = torch.full((8, k), 0.125, dtype=torch.float16, device="cuda")
    output = method.apply(layer, x)
    expected_weight = torch.cat(
        [floating.float(), torch.from_numpy(reference).half().float()]
    )
    expected = x.float().cpu() @ expected_weight.T
    torch.testing.assert_close(output.float().cpu(), expected, rtol=1e-3, atol=3e-3)


def test_fp32_blas_does_not_round_large_activations_to_fp16():
    native = load_native()
    packed, reference = fixture(41)
    weight = pad_weight_tail(torch.from_numpy(packed).cuda(), 41)
    x = torch.full((8, reference.shape[1]), 100000.0, device="cuda")
    output = native.ggml_dense_blas(weight, x, 41, reference.shape[0])
    expected = x.cpu() @ torch.from_numpy(reference).T
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output.cpu(), expected, rtol=1e-5, atol=0.1)
