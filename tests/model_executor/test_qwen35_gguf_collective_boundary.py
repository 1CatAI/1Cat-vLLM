# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models import qwen3_5 as model


@pytest.mark.parametrize(
    "quantization,tp,bias,sm70,expected",
    [
        ("gguf", 4, None, True, True),
        ("compressed-tensors", 4, None, True, True),
        ("gguf", 2, None, True, False),
        ("gguf", 4, True, True, False),
        ("gguf", 4, None, False, False),
        ("awq", 4, None, True, False),
    ],
)
def test_gdn_collective_boundary(monkeypatch, quantization, tp, bias, sm70, expected):
    class Gdn(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.out_proj = SimpleNamespace(bias=bias, reduce_results=True)

    monkeypatch.setattr(model, "Qwen3_5GatedDeltaNet", Gdn)
    monkeypatch.setattr(model, "Qwen3NextMLP", lambda **kwargs: torch.nn.Identity())
    monkeypatch.setattr(model, "Qwen3_5RMSNorm", lambda *a, **kw: torch.nn.Identity())
    monkeypatch.setattr(model, "_is_dflash2_spec_config", lambda _: True)
    monkeypatch.setattr(model.current_platform, "is_device_capability", lambda _: sm70)
    config = SimpleNamespace(
        hidden_size=5120,
        intermediate_size=17408,
        hidden_act="silu",
        rms_norm_eps=9.999999974752427e-7,
        model_type="qwen3_5_text",
    )
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=config, dtype=torch.float16, quantization=quantization
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp),
        cache_config=None,
        quant_config=None,
    )
    layer = model.Qwen3_5DecoderLayer(cfg, "linear_attention", "model.layers.0")
    assert layer.sm70_gdn_outer_allreduce == expected
    assert layer.linear_attn.out_proj.reduce_results == (not expected)
