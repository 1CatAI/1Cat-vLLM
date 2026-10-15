# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from vllm.model_executor.model_loader.gguf_loader import GGUFModelLoader
from vllm.model_executor.model_loader.gguf_multimodal import (
    GGUFMultiModalWeights,
    native_gguf_language_model,
    qualify_native_gguf_names,
)
from vllm.model_executor.models.utils import WeightsMapper


class Wrapped(torch.nn.Module):
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"model.visual.": "visual."})
    _tower_model_names = ["visual"]

    def __init__(self):
        super().__init__()
        self.language_model = torch.nn.Linear(2, 2, bias=False)
        self.visual = torch.nn.Linear(3, 2)

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        loaded = set()
        for name, value in self.hf_to_vllm_mapper.apply(weights):
            params[name].data.copy_(value)
            loaded.add(name)
        return loaded


def checkpoint(tmp_path, **changes):
    weights = {
        "model.visual.weight": torch.arange(6.0).reshape(2, 3),
        "model.visual.bias": torch.tensor([2.0, -1.0]),
    }
    for name, value in changes.items():
        if value is None:
            weights.pop(name)
        else:
            weights[name] = value
    path = tmp_path / "visual.safetensors"
    save_file(weights, path)
    return GGUFMultiModalWeights(str(path)), weights


def test_complete_tower_load_preserves_language_and_independent_models(tmp_path):
    source, weights = checkpoint(tmp_path)
    a, b = Wrapped(), Wrapped()
    before = a.language_model.weight.detach().clone()
    b_before = b.visual.weight.detach().clone()
    assert source.unquantized_modules(a.hf_to_vllm_mapper) == ["visual"]
    assert source.load(a) == {"visual.weight", "visual.bias"}
    assert torch.equal(a.visual.weight, weights["model.visual.weight"])
    assert torch.equal(a.language_model.weight, before)
    assert torch.equal(b.visual.weight, b_before)
    x = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.equal(a.visual(x), torch.tensor([[10.0, 25.0]]))


@pytest.mark.parametrize(
    "change",
    [
        {"model.visual.bias": None},
        {"language_model.weight": torch.ones(2, 2)},
        {"model.visual.unknown": torch.ones(1)},
    ],
)
def test_missing_or_non_tower_weights_reject_before_any_copy(tmp_path, change):
    source, _ = checkpoint(tmp_path, **change)
    model = Wrapped()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with pytest.raises(ValueError):
        source.load(model)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_disabled_tower_rejects(tmp_path):
    source, _ = checkpoint(tmp_path)
    model = Wrapped()
    model._tower_model_names = []
    with pytest.raises(ValueError, match="active model towers"):
        source.load(model)


def test_native_scope_and_quantization_layout_ownership():
    model = Wrapped()
    names = {"model.layers.0.self_attn.q_proj": object()}
    wrapped = SimpleNamespace(language_model_prefix="language_model")
    assert native_gguf_language_model(model, wrapped) is model.language_model
    assert native_gguf_language_model(model, None) is model
    qualified = qualify_native_gguf_names(names, wrapped)
    assert set(qualified) == {"language_model.model.layers.0.self_attn.q_proj"}
    assert next(iter(qualified.values())) is next(iter(names.values()))
    assert qualify_native_gguf_names(names, None) == names


def test_qwen4exp_scope_follows_the_full_model_config():
    import json
    from pathlib import Path

    from vllm.model_executor.model_loader.gguf_adapters import get_gguf_adapter
    from vllm.models.qwen4_exp.config import Qwen4ExpConfig
    from vllm.transformers_utils.gguf_config import gguf_config_from_metadata

    fixture = json.loads(
        Path(__file__).with_name("flashnext_metadata.json").read_text()
    )
    metadata = fixture["metadata"]
    metadata["tokenizer.ggml.tokens"] = range(fixture["vocab_size"])
    text = gguf_config_from_metadata(metadata)
    full = Qwen4ExpConfig(text_config=text)
    assert get_gguf_adapter(text, tp_size=4).language_model_prefix == ""
    assert get_gguf_adapter(full, tp_size=4).language_model_prefix == "language_model"


@pytest.mark.parametrize("path", [None, "", "/missing.safetensors", 5])
def test_invalid_asset_rejects(path):
    with pytest.raises(ValueError, match="local safetensors"):
        GGUFMultiModalWeights(path)


def test_unknown_loader_options_still_reject():
    from vllm.config import LoadConfig

    with pytest.raises(ValueError, match="not supported"):
        GGUFModelLoader(
            LoadConfig(load_format="gguf", model_loader_extra_config={"bad": 1})
        )


@pytest.mark.parametrize(
    "native,companion", [(True, True), (True, False), (False, True)]
)
def test_weight_types_do_not_require_legacy_mmproj(monkeypatch, native, companion):
    from unittest.mock import Mock

    from vllm.config import LoadConfig
    from vllm.model_executor.model_loader import gguf_loader

    loader = GGUFModelLoader(LoadConfig(load_format="gguf"))
    loader._native_adapter = object() if native else None
    loader.multimodal_weights = Mock() if companion else None
    monkeypatch.setattr(loader, "_get_all_gguf_files", lambda _: ["text.gguf"])
    monkeypatch.setattr(
        gguf_loader, "get_gguf_weight_type_map", lambda *_: {"x": "Q8_0"}
    )
    detect = Mock(side_effect=AssertionError("must not search for legacy mmproj"))
    monkeypatch.setattr(gguf_loader, "detect_gguf_multimodal", detect)
    config = SimpleNamespace(hf_config=SimpleNamespace(vision_config=object()))
    assert loader._get_gguf_weight_type(config, "text.gguf", {}) == {"x": "Q8_0"}
    detect.assert_not_called()


def test_legacy_multimodal_still_requires_mmproj(monkeypatch):
    from vllm.config import LoadConfig
    from vllm.model_executor.model_loader import gguf_loader

    loader = GGUFModelLoader(LoadConfig(load_format="gguf"))
    monkeypatch.setattr(loader, "_get_all_gguf_files", lambda _: ["text.gguf"])
    monkeypatch.setattr(gguf_loader, "get_gguf_weight_type_map", lambda *_: {})
    monkeypatch.setattr(gguf_loader, "detect_gguf_multimodal", lambda _: None)
    config = SimpleNamespace(hf_config=SimpleNamespace(vision_config=object()))
    with pytest.raises(AssertionError, match="mm_proj"):
        loader._get_gguf_weight_type(config, "text.gguf", {})


@pytest.mark.parametrize("wrapped", [False, True])
def test_native_stream_names_follow_the_receiving_model(monkeypatch, wrapped):
    from vllm.config import LoadConfig
    from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration
    from vllm.model_executor.models.utils import AutoWeightsLoader

    loader = GGUFModelLoader(LoadConfig(load_format="gguf"))
    loader._native_adapter = SimpleNamespace(language_model_prefix="language_model")
    root = torch.nn.Module()
    root.language_model = torch.nn.Module()
    root.language_model.model = torch.nn.Module()
    root.language_model.model.ple = torch.nn.Linear(2, 2, bias=False)
    receiving = root if wrapped else root.language_model
    names = {"ple": "model.ple.weight", "unrelated": "model.other.weight"}
    monkeypatch.setattr(loader, "_prepare_weights", lambda _: "text.gguf")
    monkeypatch.setattr(loader, "_get_gguf_weights_map", lambda _: names)
    touched = []

    def iterator(config, path, filtered):
        for raw, name in filtered.items():
            assert raw == "ple", "non-PLE payload must not be touched"
            touched.append(raw)
            yield name, torch.ones(2, 2)

    monkeypatch.setattr(loader, "_get_weights_iterator", iterator)
    prefix = "language_model." if wrapped else ""
    mapper = Qwen3VLForConditionalGeneration.hf_to_vllm_mapper if wrapped else None
    seen = []

    def skip(name):
        seen.append(name)
        return name != prefix + "model.ple.weight"

    weights = list(
        loader.get_all_weights(SimpleNamespace(), receiving, skip_weight=skip)
    )
    assert touched == ["ple"]
    assert seen == [prefix + n for n in names.values()]
    assert weights[0][0] == prefix + "model.ple.weight"
    loaded = AutoWeightsLoader(receiving).load_weights(weights, mapper=mapper)
    assert loaded == {prefix + "model.ple.weight"}
    torch.testing.assert_close(root.language_model.model.ple.weight, torch.ones(2, 2))
