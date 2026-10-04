# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gguf
import numpy as np
import pytest
import torch

from vllm.model_executor.layers.quantization.gguf_raw import RawGGUFProjection
from vllm.model_executor.layers.quantization.gguf_turbomind_moe import GGUFExpertBank
from vllm.transformers_utils.gguf_tensor_reader import quant_size


@pytest.mark.parametrize("kind", [21, 22])
@pytest.mark.parametrize("m", [5, 20])
def test_opaque_gate_up_dispatch_and_changed_input_graph(kind, m):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 CUDA test")
    experts, n, k, top_k = 4, 160, 2560, 2
    block, size = quant_size(kind)
    rng = np.random.default_rng(kind + m)
    banks, raw, decoded = [], [], []
    for _ in range(2):
        data = rng.integers(0, 256, (experts, n, k // block, size), dtype=np.uint8)
        scales = rng.uniform(0.002, 0.01, data.shape[:3]).astype("<f2")
        data[..., :2] = scales[..., None].view(np.uint8)
        rows = data.reshape(experts, n, -1)
        bank = GGUFExpertBank(kind, experts, "cuda", torch.float16)
        for e in range(experts):
            bank.add(e, torch.from_numpy(rows[e]), 0, 1, axis=0)
        bank.finalize()
        banks.append(bank)
        raw.append(
            torch.from_numpy(
                RawGGUFProjection.from_rows(
                    rows.reshape(experts * n, -1), kind
                ).data.reshape(experts, n, -1)
            ).cuda()
        )
        decoded.append(
            torch.from_numpy(
                gguf.quants.dequantize(rows, gguf.GGMLQuantizationType(kind))
            ).cuda()
        )
    ids = torch.stack(
        (torch.zeros(m, dtype=torch.int64), torch.arange(m) % 2 + 1), 1
    ).cuda()
    sorted_ids, order = ids.flatten().sort(stable=True)
    offsets = torch.searchsorted(
        sorted_ids, torch.arange(experts + 1, device="cuda")
    ).int()
    x = (torch.randn(m, k, device="cuda") * 0.1).half()
    routed = x[order // top_k].contiguous()

    def run():
        return torch.ops.vllm.gguf_expert_gate_up(
            routed,
            offsets,
            sorted_ids,
            *raw,
            banks[0].weight_ptrs,
            banks[0].stat_ptrs,
            banks[1].weight_ptrs,
            banks[1].stat_ptrs,
            kind,
            experts,
            banks[0].group,
            n,
            top_k,
            [1, 5, 20] if kind == 21 else [1, 5],
            [],
        )

    def check(outputs):
        for output, weights in zip(outputs, decoded):
            reference = torch.einsum("rk,rnk->rn", routed.float(), weights[sorted_ids])
            torch.testing.assert_close(
                output.float(), reference, rtol=0.003, atol=0.003
            )
            assert torch.isfinite(output).all()

    for _ in range(3):
        outputs = run()
    check(outputs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = run()
    routed.mul_(0.5)
    graph.replay()
    check(outputs)
    previous = tuple(output.clone() for output in outputs)
    graph.replay()
    for actual, expected in zip(outputs, previous):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
