# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm import envs
from vllm.model_executor.layers.quantization.nvfp4_sm70_moe import (
    _use_qwen38_w2_only_mtp5,
)


def test_w2_only_mtp5_gate_excludes_changed_layouts(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 GPU required")

    monkeypatch.setattr(envs, "VLLM_SM70_NVFP4_QWEN38_MOE_W2_ONLY_MTP5", True)
    layer = SimpleNamespace(
        expert_map=None,
        global_num_experts=512,
        local_num_experts=512,
        sm70_nvfp4_qwen38_raw_scale=False,
        sm70_nvfp4_intermediate_size=160,
        moe_config=SimpleNamespace(tp_size=4),
    )
    x = torch.empty((5, 2560), device="cuda", dtype=torch.float16)
    ids = torch.empty((5, 10), device="cuda", dtype=torch.int32)
    assert _use_qwen38_w2_only_mtp5(layer, x, ids)

    layer.expert_map = ids
    assert not _use_qwen38_w2_only_mtp5(layer, x, ids)
    layer.expert_map = None
    layer.sm70_nvfp4_qwen38_raw_scale = True
    assert not _use_qwen38_w2_only_mtp5(layer, x, ids)
    layer.sm70_nvfp4_qwen38_raw_scale = False
    assert not _use_qwen38_w2_only_mtp5(layer, x[:4], ids[:4])
    assert not _use_qwen38_w2_only_mtp5(layer, x, ids.to(torch.int64))
