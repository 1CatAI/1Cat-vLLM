# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from vllm.transformers_utils.gguf_tensor_reader import dequantize, quant_size


def packed_rows(qtype, seed):
    block, size = quant_size(qtype)
    data = np.random.default_rng(seed).integers(
        0, 256, (5, 256 // block, size), dtype=np.uint8
    )
    scale = np.array([1 / 1024], dtype=np.float16).view(np.uint8)
    if qtype == 14:
        data[:, :, -2:] = scale
    else:
        data[:, :, :2] = scale
    if qtype in (3, 12, 23):
        data[:, :, 2:4] = scale
    return data.reshape(5, -1)


@pytest.mark.parametrize("qtype", [2, 3, 8, 12, 14, 20, 23, 42])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_ordered_rows_match_official_decoder_and_changed_graph(qtype, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA test")
    from vllm.model_executor.layers.quantization.gguf import dequantize_gguf_rows

    original = packed_rows(qtype, 971)
    changed = packed_rows(qtype, 972)
    packet = torch.from_numpy(original).cuda()
    for _ in range(3):
        actual = dequantize_gguf_rows(packet, qtype, 256, dtype)
    expected = torch.from_numpy(dequantize(original, qtype)).to("cuda", dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = dequantize_gguf_rows(packet, qtype, 256, dtype)
    packet.copy_(torch.from_numpy(changed))
    graph.replay()
    expected = torch.from_numpy(dequantize(changed, qtype)).to("cuda", dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_legacy_ordered_rows_match_official_decoder():
    if not torch.cuda.is_available():
        pytest.skip("CUDA test")
    from vllm.model_executor.layers.quantization.gguf import dequantize_gguf_rows

    packet = packed_rows(20, 973)
    actual = dequantize_gguf_rows(
        torch.from_numpy(packet).cuda(), 20, 256, torch.float16, False
    )
    expected = torch.from_numpy(dequantize(packet, 20)).to("cuda", torch.float16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
