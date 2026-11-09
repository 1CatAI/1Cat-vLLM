# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup preparation paths must keep their host results byte for byte."""

from types import SimpleNamespace

import gguf
import numpy as np
import pytest
import torch

from vllm.model_executor.layers.quantization import gguf_dense_hmma_formats as F
from vllm.model_executor.model_loader.gguf_adapters import qwen35


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "fmt,gs,high",
    [
        (F.Q4K, 32, 16),
        (F.Q5K, 32, 32),
        (F.Q6K, 16, 64),
        (F.LUT4, 32, 16),
        (F.Q8, 32, 256),
    ],
)
@pytest.mark.parametrize("n,k", [(32, 128), (37, 160), (160, 2560), (257, 96)])
def test_dense_pack_device_matches_host(fmt, gs, high, n, k):
    rng = np.random.default_rng(fmt * 1000 + n + k)
    q = rng.integers(0, high, (n, k), dtype=np.uint8)
    s = (
        rng.standard_normal((n, k // gs)) * 10.0 ** rng.integers(-8, 6, (n, k // gs))
    ).astype(np.float32)
    s.flat[::97] = 7e4  # rounds to FP16 infinity on both paths
    m = None
    if fmt in (F.Q4K, F.Q5K):
        m = (rng.standard_normal((n, k // gs)) * 3).astype(np.float32)
    expected = F.pack(fmt, q, s, m, gs)
    actual = F.pack_device(fmt, q, s, m, gs, torch.device("cuda"))
    for ref, out in zip(expected, actual):
        out = out.cpu().numpy()
        assert out.dtype == ref.dtype and out.shape == ref.shape
        np.testing.assert_array_equal(out, ref)


def test_threaded_embedding_decode_matches_sequential():
    rng = np.random.default_rng(3)
    rows, columns = 2600, 64
    values = rng.standard_normal((rows, columns)).astype(np.float32)
    data = gguf.quants.quantize(values, gguf.GGMLQuantizationType.Q8_0)
    tensor = SimpleNamespace(
        data=data,
        tensor_type=gguf.GGMLQuantizationType.Q8_0,
        shape=(columns, rows),
    )
    actual = qwen35._dequantize_embedding(tensor, torch.float16, "t", 256)
    expected = torch.empty((rows, columns), dtype=torch.float16)
    for start, stop, chunk in qwen35._embedding_chunks(tensor, torch.float16, "t", 256):
        expected[start:stop] = chunk
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def test_threaded_embedding_decode_reports_overflow():
    values = np.full((300, 32), 1e6, dtype=np.float32)
    data = gguf.quants.quantize(values, gguf.GGMLQuantizationType.Q8_0)
    tensor = SimpleNamespace(
        data=data, tensor_type=gguf.GGMLQuantizationType.Q8_0, shape=(32, 300)
    )
    with pytest.raises(ValueError, match="overflow"):
        qwen35._dequantize_embedding(tensor, torch.float16, "t", 64)
