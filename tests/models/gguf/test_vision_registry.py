# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import gguf_adapters, gguf_loader
from vllm.model_executor.model_loader.gguf_adapters.vision import VisionAdapterSpec
from vllm.model_executor.model_loader.gguf_loader import GGUFModelLoader
from vllm.transformers_utils import gguf_files


@pytest.mark.parametrize("language_only", [True, False])
def test_glm_native_wrapper_prefix_and_vision_dispatch(monkeypatch, language_only):
    text = SimpleNamespace(model_type="glm5_next_text")
    vision_config = object()
    config = SimpleNamespace(get_text_config=lambda: text, vision_config=vision_config)
    model_config = SimpleNamespace(
        hf_config=config,
        multimodal_config=SimpleNamespace(language_model_only=language_only),
    )
    adapter = SimpleNamespace(
        build_name_map=lambda tensors: {"output.weight": "lm_head.weight"}
    )
    monkeypatch.setattr(gguf_adapters, "get_gguf_adapter", lambda *a, **kw: adapter)
    monkeypatch.setattr(gguf_files, "gguf_shard_paths", lambda p: [Path(p)])
    seen: list[str] = []

    def tensor_index(paths):
        seen.extend(str(p) for p in paths)
        return {"sentinel": object()}

    monkeypatch.setattr(gguf_files, "gguf_tensor_index", tensor_index)
    selected = []
    vision_adapter = object()

    def create(spec, config, tensors):
        selected.append(spec.class_name)
        assert config is vision_config
        return vision_adapter

    monkeypatch.setattr(VisionAdapterSpec, "create", create)

    def detect(path):
        assert not language_only, "Text-only loading opened vision weights"
        return "mmproj.gguf"

    monkeypatch.setattr(gguf_loader, "detect_gguf_multimodal", detect)
    loader = GGUFModelLoader(LoadConfig(load_format="gguf"))
    monkeypatch.setattr(loader, "_prepare_weights", lambda c: "model.gguf")
    assert loader._get_gguf_weights_map(model_config) == {
        "output.weight": "language_model.lm_head.weight"
    }
    if language_only:
        assert selected == [] and seen == ["model.gguf"]
        assert loader._native_vision_adapter is None
    else:
        assert selected == ["Glm5NextVisionGGUF"]
        assert seen == ["model.gguf", "mmproj.gguf"]
        assert loader._native_vision_adapter is vision_adapter


def test_loader_invokes_opt_in_parameter_coverage_before_return(monkeypatch):
    loader = GGUFModelLoader(LoadConfig(load_format="gguf"))
    loaded = {"language_model.model.norm.weight"}
    observed = []
    model = SimpleNamespace(load_weights=lambda weights: loaded)
    loader._native_adapter = SimpleNamespace(
        validate_loaded_parameters=lambda actual_model, names: observed.append(
            (actual_model, names)
        )
    )
    monkeypatch.setattr(loader, "get_all_weights", lambda *args: iter(()))
    loader.load_weights(model, SimpleNamespace())
    assert observed == [(model, loaded)]


def test_text_draft_with_inherited_vision_config_skips_wrapper(monkeypatch):
    text = SimpleNamespace(model_type="glm5_next_text")
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(get_text_config=lambda: text, vision_config=object()),
        is_multimodal_model=False,
        multimodal_config=None,
    )
    adapter = SimpleNamespace(
        build_name_map=lambda tensors: {
            "output.weight": "model.layers.45.shared_head.head.weight"
        }
    )
    monkeypatch.setattr(gguf_adapters, "get_gguf_adapter", lambda *a, **kw: adapter)
    monkeypatch.setattr(gguf_files, "gguf_shard_paths", lambda p: [Path(p)])
    monkeypatch.setattr(gguf_files, "gguf_tensor_index", lambda paths: {})
    monkeypatch.setattr(
        gguf_loader,
        "detect_gguf_multimodal",
        lambda *a: pytest.fail("Text draft must not open the target vision weights"),
    )
    loader = GGUFModelLoader(LoadConfig(load_format="gguf"))
    monkeypatch.setattr(loader, "_prepare_weights", lambda c: "model.gguf")
    assert not loader._requires_vision_weights(model_config)
    assert loader._get_gguf_weights_map(model_config) == {
        "output.weight": "model.layers.45.shared_head.head.weight"
    }
    assert loader._native_vision_adapter is None
