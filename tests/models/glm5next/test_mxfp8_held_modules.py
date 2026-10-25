# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from torch import nn

from vllm.model_executor.layers.quantization.modelopt import ModelOptMxFp8LinearMethod
from vllm.models.glm5next.nvidia import model as glm_model


class _Linear(nn.Module):
    def __init__(self, quant_method=None):
        super().__init__()
        self.quant_method = quant_method


def _mxfp8_method() -> ModelOptMxFp8LinearMethod:
    # The kernel selection needs a GPU platform; the predicate only looks at
    # the method's type.
    return object.__new__(ModelOptMxFp8LinearMethod)


def _language_model() -> nn.Module:
    model = nn.Module()
    model.language_model = nn.Module()
    model.language_model.model = nn.Module()
    layers = nn.ModuleList()
    for _ in range(2):
        layer = nn.Module()
        layer.mlp = nn.Module()
        layer.mlp.gate_up_proj = _Linear(_mxfp8_method())
        layer.self_attn = nn.Module()
        layer.self_attn.o_proj = _Linear(_mxfp8_method())
        layer.self_attn.fused_qkv_a_proj = _Linear(object())
        layers.append(layer)
    model.language_model.model.layers = layers
    model.language_model.lm_head = _Linear(object())
    model.visual = nn.Module()
    model.visual.blocks = nn.ModuleList([nn.Module()])
    model.visual.blocks[0].qkv = _Linear()
    return model


def test_modules_with_an_mxfp8_method_are_held_under_their_checkpoint_names():
    quant_config = SimpleNamespace(
        packed_modules_mapping={"gate_up_proj": ["gate_proj", "up_proj"]}
    )
    held = glm_model._mxfp8_held_modules(_language_model(), quant_config)

    prefix = "model.language_model.layers.1"
    assert held(f"{prefix}.self_attn.o_proj")
    assert held(f"{prefix}.mlp.gate_proj")
    assert held(f"{prefix}.mlp.up_proj")
    assert not held(f"{prefix}.self_attn.q_a_proj")
    assert not held("lm_head")
    assert not held("model.visual.blocks.0.qkv")


def test_an_lm_head_with_an_mxfp8_method_is_held_under_its_checkpoint_name():
    quant_config = SimpleNamespace(packed_modules_mapping={})
    model = _language_model()
    model.language_model.lm_head = _Linear(_mxfp8_method())

    held = glm_model._mxfp8_held_modules(model, quant_config)
    text_held = glm_model._mxfp8_held_modules(model.language_model, quant_config)

    assert held("lm_head")
    assert text_held("lm_head")
    assert not held("model.language_model.layers.0.self_attn.fused_qkv_a_proj")


def test_the_language_model_alone_gives_the_same_answer():
    quant_config = SimpleNamespace(
        packed_modules_mapping={"gate_up_proj": ["gate_proj", "up_proj"]}
    )
    text_model = _language_model().language_model
    held = glm_model._mxfp8_held_modules(text_model, quant_config)

    assert held("model.layers.0.mlp.gate_proj")
    assert held("layers.0.self_attn.o_proj")


def test_an_unquantized_model_holds_nothing():
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Module()])
    model.layers[0].o_proj = _Linear()
    held = glm_model._mxfp8_held_modules(model, None)

    assert not held("model.language_model.layers.0.o_proj")
