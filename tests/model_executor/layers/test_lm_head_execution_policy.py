# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Excluded ModelOpt LM heads retain the owning engine's linear policy."""

import torch
import torch.nn.functional as F

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor import parameter
from vllm.model_executor.layers import vocab_parallel_embedding as embedding
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config


def test_modelopt_excluded_head_loads_and_applies(monkeypatch):
    monkeypatch.setattr(embedding, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(embedding, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 1)
    config = VllmConfig()
    quant = ModelOptNvFp4Config(
        is_checkpoint_nvfp4_serialized=True, exclude_modules=["lm_head"]
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with set_current_vllm_config(config), torch.device(device):
        head = embedding.ParallelLMHead(
            128, 64, params_dtype=torch.float16, quant_config=quant, prefix="lm_head"
        )
        assert isinstance(head.quant_method, UnquantizedLinearMethod)
        assert head._layer_execution_policy is config.kernel_config.layer_execution
        with torch.no_grad():
            head.weight.normal_(std=0.1)
        head.quant_method.process_weights_after_loading(head)
        x = torch.randn(3, 64, dtype=torch.float16)
        expected = F.linear(x, head.weight)
        actual = head.quant_method.apply(head, x)
        torch.testing.assert_close(actual, expected)
