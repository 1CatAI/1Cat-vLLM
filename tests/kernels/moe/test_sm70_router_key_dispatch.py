# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU route guards for the lossless FP16 E512/K10 sort-key specialization."""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.router import fused_topk_router as mod


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows", [1, 2, 5, 16])
def test_packed_half_key_is_m1_half_only(monkeypatch, dtype, rows):
    calls = []

    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append((grid, kwargs))

            return launch

    monkeypatch.setattr(mod, "_sm70_qwen38_router_topk_kernel", Kernel())
    x = torch.empty(rows, 512, dtype=dtype)
    weights = torch.empty(rows, 10, dtype=torch.float32)
    ids = torch.empty(rows, 10, dtype=torch.int32)
    mod._sm70_qwen38_router_topk(weights, ids, torch.empty_like(ids), x)
    assert len(calls) == 1
    grid, kwargs = calls[0]
    assert grid == (rows,)
    assert kwargs["PACKED_HALF_KEY"] == (rows == 1 and dtype == torch.float16)
    assert kwargs["num_warps"] == 8  # Keep the FP32 normalization reduction.
