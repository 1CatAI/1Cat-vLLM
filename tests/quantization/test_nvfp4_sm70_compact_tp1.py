# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real TP1 GLM projection shapes: compact groups and CUDA graph replay."""

import pytest
import torch

from vllm import _sm70_ops as ops
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.sm70.fp4_codec import Fp4MoECodec
from vllm.model_executor.layers.quantization.sm70_turbomind import unpack_mxfp4_weight


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("k", [2048, 4096])
@pytest.mark.parametrize("groups", [32, 40, 80])
def test_compact_tp1_projection_matches_reference_and_replays(k, groups):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    torch.manual_seed(47)
    n, experts = 4096, 2
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device="cuda")
    weights, scales, references = [], [], []
    with set_current_vllm_config(VllmConfig()):
        for _ in range(experts):
            raw = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
            scale = torch.full((k // 16, n), 0.01, dtype=torch.float16, device="cuda")
            packed, prepared_scale, meta = Fp4MoECodec.prepare_weights(
                "nvfp4", unpack_mxfp4_weight(raw), scale, 16
            )
            codes = torch.stack((raw & 15, raw >> 4), dim=-1).reshape(n, k).long()
            reference = lut[codes & 7] * torch.where(codes & 8 != 0, -1.0, 1.0)
            references.append(reference * scale[0, 0].float())
            weights.append(packed)
            scales.append(prepared_scale)
        bankw, banks = torch.stack(weights), torch.stack(scales)
        ptrs = ops.awq_moe_build_strided_ptrs(
            bankw, banks, int(meta[0]), int(meta[1]), experts
        )
        ids = (torch.arange(groups, device="cuda") % experts).int()
        offsets = torch.arange(groups + 1, dtype=torch.int32, device="cuda")
        x = torch.randn(groups, k, device="cuda", dtype=torch.float16)
        output = torch.empty(groups, n, device="cuda", dtype=torch.float16)

        def run():
            ops.nvfp4_moe_dense_stage_sm70_out(
                output, x, offsets, ids, *ptrs, groups, k, n, 16
            )

        run()
        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for factor in [1.0, 0.5]:
            x.mul_(factor)
            graph.replay()
            expected = torch.empty_like(output, dtype=torch.float32)
            for e in range(experts):
                mask = ids == e
                expected[mask] = x[mask].float() @ references[e].T
            torch.accelerator.synchronize()
            assert output.isfinite().all()
            error = (output.float() - expected).norm() / expected.norm()
            assert error < 0.005, error.item()
