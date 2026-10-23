# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The draft model accepts the layer-name mapping emitted by its proposer."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model
from vllm.model_executor.models.qwen3_dflash2 import DFlash2Qwen3Model

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("model_type", [DFlashQwen3Model, DFlash2Qwen3Model])
@pytest.mark.parametrize("kind", ["dict", "list", "tuple", "tensor", "none"])
def test_context_slots_select_each_attention_layer(model_type, kind):
    calls = []

    def update(attn, key, value, cache, slots):
        assert isinstance(slots, torch.Tensor)
        calls.append((attn.layer_name, key, value, cache, slots))

    model = object.__new__(model_type)
    torch.nn.Module.__init__(model)
    model._num_attn_layers = 2
    model._attn_layers = [
        SimpleNamespace(
            layer_name=name,
            kv_cache=torch.empty(0),
            impl=SimpleNamespace(do_kv_cache_update=update),
        )
        for name in ("draft.layers.3.attn", "draft.layers.8.attn")
    ]
    slots = [torch.tensor([9, 7]), torch.tensor([2, 5])]
    mapping = {
        # Deliberately reverse insertion order: lookup must use the layer name.
        model._attn_layers[1].layer_name: slots[1],
        model._attn_layers[0].layer_name: slots[0],
    }
    inputs = {
        "dict": mapping,
        "list": [slots[0], None],
        "tuple": (slots[0], None),
        "tensor": slots[0],
        "none": None,
    }
    keys = torch.arange(8).reshape(2, 2, 1, 2)
    values = keys + 10
    model.store_context_kv(keys, values, inputs[kind])
    expected = 0 if kind == "none" else 1 if kind in ("list", "tuple") else 2
    assert len(calls) == expected
    for i, (name, key, value, cache, selected) in enumerate(calls):
        assert name == model._attn_layers[i].layer_name
        assert torch.equal(key, keys[i]) and torch.equal(value, values[i])
        assert cache is model._attn_layers[i].kv_cache
        assert selected is slots[i if kind == "dict" else 0]


def test_missing_layer_mapping_fails_before_native_publication():
    model = object.__new__(DFlashQwen3Model)
    torch.nn.Module.__init__(model)
    model._num_attn_layers = 1
    model._attn_layers = [SimpleNamespace(layer_name="draft.required")]
    with pytest.raises(KeyError, match="draft.required"):
        model.store_context_kv(torch.empty(1), torch.empty(1), {})
