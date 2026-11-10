# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MMQ uses the weight's expert count, including IDs beyond 255."""

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize(
    "experts,quant_type", [(128, 8), (256, 8), (257, 8), (512, 8), (257, 2), (512, 2)]
)
@torch.inference_mode()
def test_mmq_expert_bound_and_changed_graph(experts, quant_type):
    import gguf

    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size

    rng = np.random.default_rng(9683 + experts + quant_type)
    values = rng.standard_normal((experts, 32, 256), dtype=np.float32) * 0.02
    fmt = gguf.GGMLQuantizationType(quant_type)
    encoded = gguf.quants.quantize(values, fmt)
    weight = torch.from_numpy(encoded).cuda()
    decoded = torch.from_numpy(gguf.quants.dequantize(encoded, fmt)).cuda().float()
    x = torch.randn(65, 256, dtype=torch.half, device="cuda") * 0.2
    boundary = torch.tensor(
        [0, min(255, experts - 1), min(256, experts - 1), experts - 1],
        device="cuda",
        dtype=torch.int32,
    )
    ids = boundary.repeat(17)[:65, None].contiguous()

    def run():
        sorted_ids, expert_ids, padded = moe_align_block_size(
            ids, ops.ggml_moe_get_block_size(quant_type), experts
        )
        return ops.ggml_moe_a8(
            x, weight, sorted_ids, expert_ids, padded, quant_type, 32, 1, 65
        )

    def check(actual):
        expected = torch.bmm(
            decoded[ids[:, 0].long()], x.float().unsqueeze(-1)
        ).squeeze(-1)
        assert torch.isfinite(actual).all()
        # Same FP32 dequantized oracle and 2% norm allowance used by the
        # Q8 MTP expert gate; MMQ additionally quantizes activations to Q8_1.
        error = (actual.float() - expected).norm() / expected.norm()
        assert error < 2e-2, error

    check(run())
    if experts == 512:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
        x.mul_(-0.5)
        ids.copy_(boundary.flip(0).repeat(17)[:65, None])
        graph.replay()
        check(captured)
