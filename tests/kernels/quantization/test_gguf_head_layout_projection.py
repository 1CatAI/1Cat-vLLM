# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from vllm.model_executor.layers.quantization.gguf_layout import GGUFHeadTilingLayout
from vllm.model_executor.layers.quantization.gguf_transcode import transcode_affine
from vllm.model_executor.layers.quantization.gguf_turbomind import (
    apply_prepared_gguf_projections,
    prepare_gguf_projections,
)
from vllm.transformers_utils.gguf_tensor_reader import dequantize, quant_size

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def source(n=96, k=1536, weight_type=14):
    block, size = quant_size(weight_type)
    raw = np.random.default_rng(20261005).integers(
        0, 256, (n * k // block, size), dtype=np.uint8
    )
    offset = -2 if weight_type == 14 else 0
    raw[:, offset : None if offset == -2 else 2] = np.frombuffer(
        np.float16(1 / 2048).tobytes(), np.uint8
    )
    if weight_type in (12, 13):
        raw[:, 2:4] = np.frombuffer(np.float16(1 / 4096).tobytes(), np.uint8)
    return raw.reshape(n, -1)


@pytest.mark.parametrize("m", [1, 5, 20, 512])
@pytest.mark.parametrize("heads_per_group", [2, 3])
@pytest.mark.parametrize("weight_type", [12, 13, 14])
def test_head_layout_projection_matches_official_dequant_and_graph(
    m, heads_per_group, weight_type
):
    raw = source(weight_type=weight_type)
    layout = GGUFHeadTilingLayout(heads_per_group, 128)
    weight = torch.from_numpy(raw).cuda()
    original = prepare_gguf_projections(
        [(weight, weight_type)], torch.float16, True, 256
    )
    restored = prepare_gguf_projections(
        [(weight, weight_type)], torch.float16, True, 256, input_layout=layout
    )
    assert restored[0].input_layout_restored
    reference = layout.weight_to_vllm(
        torch.from_numpy(dequantize(raw, weight_type)), dim=1
    )
    canonical = transcode_affine(raw, weight_type)
    # This fixture has exactly representable group scales. Head restoration
    # changes only the ordering, including every group coefficient.
    np.testing.assert_array_equal(canonical.dequantize(), dequantize(raw, weight_type))
    torch.manual_seed(20261005 + m)
    x = (torch.randn(m, 1536, device="cuda") * 0.125).half()
    expected = x.float() @ reference.cuda().half().float().T
    actual = apply_prepared_gguf_projections(x, restored)
    previous = apply_prepared_gguf_projections(layout.input_to_gguf(x), original)
    torch.testing.assert_close(actual.float(), expected, rtol=0.003, atol=0.0002)
    torch.testing.assert_close(actual, previous, rtol=0.003, atol=0.0002)
    if m in (5, 20):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = apply_prepared_gguf_projections(x, restored)
        x.copy_((torch.randn_like(x) * 0.125).half())
        graph.replay()
        torch.testing.assert_close(
            captured, apply_prepared_gguf_projections(x, restored), rtol=0, atol=0
        )


def test_mixed_unrestorable_projection_keeps_all_shards_in_gguf_order():
    weight = torch.from_numpy(source()).cuda()
    # A dense shard has no canonical affine representation. The other shard
    # must keep its old order so one input transform still serves both.
    projections = prepare_gguf_projections(
        [(weight, 14), (torch.ones(32, 1536, device="cuda").half(), 1)],
        torch.float16,
        True,
        256,
        input_layout=GGUFHeadTilingLayout(2, 128),
    )
    assert not any(p.input_layout_restored for p in projections)
    x = torch.randn(5, 1536, device="cuda").half()
    expected = prepare_gguf_projections(
        [(weight, 14), (torch.ones(32, 1536, device="cuda").half(), 1)],
        torch.float16,
        True,
        256,
    )
    torch.testing.assert_close(
        apply_prepared_gguf_projections(x, projections),
        apply_prepared_gguf_projections(x, expected),
        rtol=0,
        atol=0,
    )


def test_head_boundary_inside_canonical_group_preserves_input_transform():
    projections = prepare_gguf_projections(
        [(torch.from_numpy(source()).cuda(), 14)],
        torch.float16,
        True,
        256,
        input_layout=GGUFHeadTilingLayout(2, 8),
    )
    assert not projections[0].input_layout_restored
