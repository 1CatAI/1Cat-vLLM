# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense M1/M6 retains FP16 operands, independently of expert DP4A."""

import gguf
import numpy as np
import pytest
import torch

from vllm.model_executor.layers.quantization.gguf_native import (
    native_available,
    native_dense,
    pad_weight_tail,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform.is_device_capability(70),
    reason="SM70 qualification required",
)


@pytest.mark.parametrize("kind", [8, 13])
@pytest.mark.parametrize("rows", [1, 6])
def test_dense_small_batches_use_bounded_fp16_blas_not_q8_mmvq(monkeypatch, kind, rows):
    assert native_available()
    n, k, maximum = 1537, 5120, 4 * 1024**2
    block, size = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(kind)]
    packed = np.random.default_rng(41).integers(
        0, 256, (n, k // block, size), dtype=np.uint8
    )
    packed[..., :2] = np.frombuffer(np.float16(0.0037).tobytes(), dtype=np.uint8)
    if kind == 13:
        packed[..., 2:4] = np.frombuffer(np.float16(0.001).tobytes(), dtype=np.uint8)
    packed = packed.reshape(n, -1)
    dense = torch.from_numpy(
        gguf.quants.dequantize(packed, gguf.GGMLQuantizationType(kind))
    ).half()
    weight = pad_weight_tail(torch.from_numpy(packed).to("cuda"), kind)
    generator = torch.Generator().manual_seed(41)
    x = (torch.randn(rows, k, generator=generator) * 0.1).half().to("cuda")
    native = torch.ops._C_gguf
    blas = native.ggml_dense_blas
    chunks = []

    def checked(weight, values, kind, count):
        assert count * k * 2 <= maximum
        chunks.append(count)
        return blas(weight, values, kind, count)

    def forbidden(*args, **kwargs):
        raise AssertionError("Dense FP16 qualification reached Q8 MMVQ")

    monkeypatch.setattr(native, "ggml_dense_blas", checked)
    monkeypatch.setattr(native, "ggml_dense_mmvq", forbidden)
    actual = native_dense(x, weight, kind, prefill_min_m=1, max_dequant_bytes=maximum)
    assert actual is not None and actual.dtype == torch.float16
    assert sum(chunks) == n and len(chunks) > 1
    expected = (x.cpu().float() @ dense.float().T).half()
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-3, atol=1e-3)
