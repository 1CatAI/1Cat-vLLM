# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm import _sm70_ops as ops
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW8A16Fp8,
)
from vllm.platforms import current_platform


@pytest.mark.skipif(
    not current_platform.is_device_capability(70), reason="SM70 FP8 projection"
)
def test_graph_projection_writes_destination():
    torch.manual_seed(71)
    weight = (torch.randn(5120, 1536, device="cuda") * 0.05).to(torch.float8_e4m3fn)
    scales = torch.rand(5120, 1, device="cuda") * 0.25
    codes, packed_scales = ops.fp8_qpn8_prepare_sm70(weight, scales)
    layer = SimpleNamespace(
        sm70_fp8_turbomind=True,
        sm70_fp8_qpn8=True,
        weight=codes,
        weight_scale_inv=packed_scales,
        output_size_per_partition=5120,
        sm70_fp8_qpn8_split_k=12,
        sm70_fp8_qpn8_nacc=2,
        sm70_fp8_qpn8_prefetch=False,
    )
    scheme = object.__new__(CompressedTensorsW8A16Fp8)
    x = torch.randn(8, 1536, device="cuda", dtype=torch.float16)
    backing = torch.full((10, 5120), float("nan"), device="cuda", dtype=x.dtype)
    output = backing[1:9]
    reference = scheme.apply_weights(layer, x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        returned = scheme.apply_weights(layer, x, output=output)
    assert returned.data_ptr() == output.data_ptr()
    for amplitude in (0.0, 0.125, -0.25, 1.0):
        x.copy_(torch.randn_like(x) * amplitude)
        reference = scheme.apply_weights(layer, x)
        graph.replay()
        torch.accelerator.synchronize()
        assert torch.equal(reference.view(torch.int16), output.view(torch.int16))
        assert torch.isnan(backing[0]).all() and torch.isnan(backing[-1]).all()
    with pytest.raises(ValueError, match="output buffer"):
        scheme.apply_weights(layer, x, output=backing[:8, ::2])
