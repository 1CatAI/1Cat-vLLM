# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm import _sm70_ops as ops
from vllm.model_executor.layers.quantization.sm70_dflash2_nvfp4_head import (
    SM70DFlash2NVFP4Head,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("rows", [7, 8])
@torch.inference_mode()
def test_private_candidate_head_preserves_target_and_graph_fallback(rows):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 is required")
    torch.manual_seed(123)
    weight = (torch.randn(1056, 5120, device="cuda") * 0.01).to(torch.float8_e4m3fn)
    # Include zeros and a partial quantization chunk.
    weight[:32].zero_()
    scales = torch.ones(1056, 1, device="cuda") * 0.1
    packed, packed_scales, _ = ops.fp8_sm70_prepare(weight, scales, 128, False)
    saved_weight = packed.clone()
    saved_scales = packed_scales.clone()
    head = SimpleNamespace(
        embedding_dim=5120,
        num_embeddings_per_partition=1056,
        weight=packed,
        weight_scale_inv=packed_scales,
    )
    candidate = SM70DFlash2NVFP4Head.from_turbomind_head(head)
    assert torch.equal(packed, saved_weight)
    assert torch.equal(packed_scales, saved_scales)
    assert candidate.state_dict() == {}
    x = torch.randn(rows, 5120, device="cuda", dtype=torch.float16) * 0.1
    eager = candidate(x)
    assert eager is not None
    assert torch.isfinite(eager).all()
    assert torch.count_nonzero(eager[:, :32]) == 0
    for unsupported_rows in (1, 6, 14, 16, 28, 32):
        assert candidate(x[:1].repeat(unsupported_rows, 1)) is None
    assert candidate(x.T.contiguous().T) is None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = candidate(x)
    for amplitude in (0.0, -0.1, 0.2):
        x.copy_(torch.randn_like(x) * amplitude)
        graph.replay()
        assert torch.equal(actual, candidate(x))
    assert torch.equal(packed, saved_weight)
    assert torch.equal(packed_scales, saved_scales)
