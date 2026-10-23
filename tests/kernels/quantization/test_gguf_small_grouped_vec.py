# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from vllm.model_executor.layers.quantization.gguf_lut_transcode import transcode_lut4
from vllm.model_executor.layers.quantization.gguf_transcode import transcode_affine
from vllm.transformers_utils.gguf_tensor_reader import dequantize, quant_size


@pytest.mark.parametrize("kind", [20, 42])
@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("m", [1, 5, 20])
def test_small_grouped_vectors_tp4_official_reference_and_graph(kind, rank, m):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 CUDA test")
    experts, n, full_k, top_k = 4, 64, 640, 2
    k = full_k // 4
    block, size = quant_size(kind)
    rng = np.random.default_rng(541 + kind)
    data = rng.integers(0, 256, (experts, n, full_k // block, size), dtype=np.uint8)
    data[:, :, :, :2] = np.full(data.shape[:3], 0.01, dtype="<f2")[..., None].view(
        np.uint8
    )
    sources = data.reshape(experts, n, -1)
    prepared, decoded = [], []
    for source in sources:
        projection = (transcode_lut4 if kind == 20 else transcode_affine)(source, kind)
        local = projection.tp_slice(rank, 4, axis=1)
        codes, scales = (
            torch.from_numpy(local.codes).cuda(),
            torch.from_numpy(local.scales).cuda(),
        )
        if kind == 20:
            p = torch.ops._C.gguf_lut4_sm70_prepare(codes, scales, local.lut_id, 32)
        else:
            p = torch.ops._C.gguf_affine_sm70_prepare(
                codes, scales, torch.from_numpy(local.mins).cuda(), local.bits, 32
            )
        prepared.append(p)
        decoded.append(dequantize(source, kind)[:, rank * k : (rank + 1) * k])
    weights, stats = (
        torch.stack([p[0] for p in prepared]),
        torch.stack([p[1] for p in prepared]),
    )
    wp, sp = torch.ops._C.awq_moe_build_strided_ptrs(
        weights, stats, *prepared[0][2].tolist(), experts
    )
    reference_weights = torch.from_numpy(np.stack(decoded)).cuda()
    torch.manual_seed(960 + m)
    # Every token visits expert zero and one other expert. Empty experts and
    # large intervals are exercised alongside singleton intervals.
    ids = torch.stack(
        (torch.zeros(m, dtype=torch.int64), torch.arange(m) % 2 + 1), 1
    ).cuda()
    sorted_ids, order = ids.flatten().sort(stable=True)
    offsets = torch.searchsorted(
        sorted_ids, torch.arange(experts + 1, device="cuda")
    ).int()
    x = (torch.randn(m, k, device="cuda") * 0.1).half()
    routed = x[order // top_k].contiguous()
    output = torch.empty((m * top_k, n), dtype=torch.float16, device="cuda")

    def run():
        torch.ops._C.gguf_small_grouped_vec_sm70_out(
            output, routed, offsets, wp, sp, kind, experts, 32
        )

    def check():
        expected = torch.einsum(
            "rk,rnk->rn", routed.float(), reference_weights[sorted_ids]
        )
        torch.testing.assert_close(output.float(), expected, rtol=0.003, atol=0.002)
        assert torch.isfinite(output).all()

    for _ in range(3):
        run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    routed.mul_(0.5)
    graph.replay()
    check()
    previous = output.clone()
    graph.replay()
    torch.testing.assert_close(output, previous, rtol=0, atol=0)
