# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
from torch import nn

from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.models.glm5next.nvidia.kda import Glm5NextLinearAttention
from vllm.models.glm5next.nvidia.model import (
    Glm5NextForConditionalGeneration,
    Glm5NextModel,
)


def test_glm53_pp_intermediate_state_shapes():
    model = Glm5NextModel.__new__(Glm5NextModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        hidden_size=4096, mhc=True, mhc_num_residual_streams=4
    )

    tensors = model.make_empty_intermediate_tensors(
        batch_size=7, dtype=torch.float16, device=torch.device("cpu")
    ).tensors
    assert tuple(tensors) == ("hidden_states", "residual", "post", "comb")
    assert tensors["hidden_states"].shape == (7, 4096)
    assert tensors["residual"].shape == (7, 4, 4096)
    assert tensors["post"].shape == (7, 4, 1)
    assert tensors["comb"].shape == (7, 4, 4)
    assert tensors["hidden_states"].dtype == torch.float16
    assert tensors["residual"].dtype == torch.float16
    assert tensors["post"].dtype == torch.float32
    assert tensors["comb"].dtype == torch.float32


def test_glm53_merged_kda_state_contract():
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=4),
        speculative_config=None,
        model_config=SimpleNamespace(
            dtype=torch.float16,
            hf_config=SimpleNamespace(
                linear_num_heads=64,
                linear_head_dim=128,
                linear_conv_kernel_dim=4,
            ),
        ),
        cache_config=SimpleNamespace(mamba_cache_dtype="auto"),
    )
    model_shapes = Glm5NextForConditionalGeneration.get_mamba_state_shape_from_config(
        vllm_config
    )
    model_dtypes = Glm5NextForConditionalGeneration.get_mamba_state_dtype_from_config(
        vllm_config
    )
    copy_funcs = Glm5NextForConditionalGeneration.get_mamba_state_copy_func()

    layer = Glm5NextLinearAttention.__new__(Glm5NextLinearAttention)
    nn.Module.__init__(layer)
    layer.tp_size = 4
    layer.num_heads = 64
    layer.head_dim = 128
    layer.conv_size = 4
    layer.num_spec = 0
    layer.model_config = vllm_config.model_config
    layer.cache_config = vllm_config.cache_config

    assert layer.get_state_shape() == model_shapes
    assert layer.get_state_dtype() == model_dtypes
    assert model_dtypes == (torch.float16, torch.float32)
    assert len(copy_funcs) == 2
    assert model_shapes[1] == (16, 128, 128)
    expected_conv = (6144, 3) if is_conv_state_dim_first() else (3, 6144)
    assert model_shapes[0] == expected_conv
