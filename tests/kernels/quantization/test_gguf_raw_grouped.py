# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gguf
import numpy as np
import pytest
import torch
import vllm._C  # noqa: F401

from vllm.model_executor.layers.quantization.gguf_raw import RawGGUFProjection
from vllm.transformers_utils.gguf_tensor_reader import quant_size


@pytest.mark.parametrize("kind", [21, 22])
@pytest.mark.parametrize("m", [1, 5, 20, 32])
def test_raw_grouped_gate_up_official_reference_and_graph(kind, m):
    torch.manual_seed(4321 + m)
    # Every token visits expert zero and one other expert. This covers both
    # singleton intervals and the maximum possible tokens at one expert.
    experts, n, k, top_k = m + 2, 7, 768, 2
    block, size = quant_size(kind)
    rng = np.random.default_rng(kind + m)
    weights, references = [], []
    for _ in range(2):
        data = rng.integers(0, 256, (experts * n, k // block, size), dtype=np.uint8)
        d = rng.uniform(0.005, 0.02, data.shape[:2]).astype("<f2")
        data[:, :, :2] = d[..., None].view(np.uint8)
        data = data.reshape(experts * n, -1)
        raw = RawGGUFProjection.from_rows(data, kind)
        weights.append(torch.from_numpy(raw.data.reshape(experts, n, -1)).cuda())
        references.append(
            torch.from_numpy(
                gguf.quants.dequantize(data, gguf.GGMLQuantizationType(kind))
            )
            .reshape(experts, n, k)
            .cuda()
        )
    ids = torch.stack((torch.zeros(m, dtype=torch.int64), torch.arange(1, m + 1)), 1)
    sorted_ids, order = ids.flatten().cuda().sort(stable=True)
    offsets = torch.searchsorted(
        sorted_ids, torch.arange(experts + 1, device="cuda")
    ).int()
    x = torch.randn(m, k, device="cuda", dtype=torch.float16)
    routed = x[order // top_k].contiguous()
    gate = torch.empty((m * top_k, n), device="cuda", dtype=torch.float16)
    up = torch.empty_like(gate)

    def run():
        torch.ops._C.gguf_lattice_raw_grouped_gate_up_sm70_out(
            gate, up, routed, *weights, offsets, sorted_ids, kind, top_k
        )

    def check():
        for out, reference in zip((gate, up), references):
            expected = torch.einsum("rk,rnk->rn", routed.float(), reference[sorted_ids])
            torch.testing.assert_close(out.float(), expected, rtol=0.002, atol=0.003)
            # The original single-row raw decoder provides an independent
            # dispatch check with the same FP32 FMA and warp reduction order.
            scratch = torch.empty((1, n), device="cuda", dtype=torch.float32)
            single = torch.empty((1, n), device="cuda", dtype=torch.float16)
            which = 0 if out is gate else 1
            for row, expert in enumerate(sorted_ids.tolist()):
                torch.ops._C.gguf_lattice_raw_vec_sm70_out(
                    single,
                    routed[row : row + 1],
                    weights[which][expert],
                    kind,
                    scratch,
                    1,
                    False,
                )
                torch.testing.assert_close(out[row], single[0], rtol=0, atol=0)

    for _ in range(3):
        run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    routed.copy_(torch.randn_like(routed))
    graph.replay()
    check()
    previous = gate.clone(), up.clone()
    graph.replay()
    torch.testing.assert_close(gate, previous[0], rtol=0, atol=0)
    torch.testing.assert_close(up, previous[1], rtol=0, atol=0)
