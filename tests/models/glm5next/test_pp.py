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
    _dflash_aux_hidden_state_key,
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


def test_glm53_dflash_mhc_capture_materializes_completed_layer_output():
    model = Glm5NextModel.__new__(Glm5NextModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(mhc=True, mhc_num_residual_streams=4)

    hidden_states = torch.randn(3, 8)
    residual = torch.randn(3, 4, 8)
    post = torch.randn(3, 4, 1)
    comb = torch.randn(3, 4, 4)
    completed_streams = torch.randn(3, 4, 8)

    class FakeLayer:
        def hc_post(self, hidden, incoming_residual, incoming_post, incoming_comb):
            assert hidden is hidden_states
            assert incoming_residual is residual
            assert incoming_post is post
            assert incoming_comb is comb
            return completed_streams

    captured = model._materialize_aux_hidden_state(
        FakeLayer(), hidden_states, residual, post, comb
    )

    torch.testing.assert_close(captured, completed_streams.mean(dim=1))


class _FakeGlmBoundaryLayer(nn.Module):
    def __init__(self, layer_idx: int, increment: float):
        super().__init__()
        self.layer_idx = layer_idx
        self.increment = increment

    def forward(self, positions, hidden_states, residual, post, comb):
        del positions, residual, post, comb
        output = hidden_states + self.increment
        return output, output, None, None


def test_glm53_dflash_aux_hidden_states_cross_pp_boundary(monkeypatch):
    group = SimpleNamespace(is_first_rank=True, is_last_rank=False)
    monkeypatch.setattr(
        "vllm.models.glm5next.nvidia.model.get_pp_group", lambda: group
    )

    stage0 = Glm5NextModel.__new__(Glm5NextModel)
    nn.Module.__init__(stage0)
    stage0.config = SimpleNamespace(hidden_size=8, mhc=False)
    stage0.start_layer = 0
    stage0.end_layer = 24
    stage0.is_sequence_parallel = False
    stage0._active_layers = nn.ModuleList(
        [_FakeGlmBoundaryLayer(5, 1.0), _FakeGlmBoundaryLayer(14, 2.0)]
    )
    stage0._set_aux_hidden_state_layers((6, 15, 25, 34, 43))

    positions = torch.arange(2)
    stage0_output = stage0(
        input_ids=None,
        positions=positions,
        intermediate_tensors=None,
        inputs_embeds=torch.zeros(2, 8),
    )
    assert tuple(stage0_output.tensors) == (
        "hidden_states",
        "residual",
        _dflash_aux_hidden_state_key(6),
        _dflash_aux_hidden_state_key(15),
    )

    group.is_first_rank = False
    group.is_last_rank = True
    stage1 = Glm5NextModel.__new__(Glm5NextModel)
    nn.Module.__init__(stage1)
    stage1.config = SimpleNamespace(hidden_size=8, mhc=False)
    stage1.start_layer = 24
    stage1.end_layer = 45
    stage1.is_sequence_parallel = False
    stage1.norm = nn.Identity()
    stage1._active_layers = nn.ModuleList(
        [
            _FakeGlmBoundaryLayer(24, 3.0),
            _FakeGlmBoundaryLayer(33, 4.0),
            _FakeGlmBoundaryLayer(42, 5.0),
        ]
    )
    stage1._set_aux_hidden_state_layers((6, 15, 25, 34, 43))

    input_schema = stage1.make_empty_intermediate_tensors(
        batch_size=2, dtype=torch.float32, device=torch.device("cpu")
    )
    assert tuple(input_schema.tensors)[-2:] == (
        _dflash_aux_hidden_state_key(6),
        _dflash_aux_hidden_state_key(15),
    )

    hidden_states, aux_hidden_states = stage1(
        input_ids=None,
        positions=positions,
        intermediate_tensors=stage0_output,
    )
    torch.testing.assert_close(hidden_states, torch.full((2, 8), 15.0))
    for captured, expected in zip(aux_hidden_states, (1, 3, 6, 10, 15)):
        torch.testing.assert_close(captured, torch.full((2, 8), float(expected)))


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
