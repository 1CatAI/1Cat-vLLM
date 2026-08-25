# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Generator
from types import SimpleNamespace

import torch.nn as nn

from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.offloader import (
    BaseOffloader,
    PrefetchOffloader,
    get_offloader,
    set_offloader,
)
from vllm.model_executor.offloader.uva import UVAOffloader


class _RecordingOffloader(BaseOffloader):
    def __init__(self) -> None:
        self.component_calls: list[tuple[nn.Module, str]] = []

    def wrap_modules(
        self, modules_generator: Generator[nn.Module, None, None]
    ) -> list[nn.Module]:
        raise AssertionError("tower registration must not reuse the layer wrapper")

    def wrap_module(
        self, module: nn.Module, *, parameter_prefix: str = ""
    ) -> nn.Module:
        self.component_calls.append((module, parameter_prefix))
        return module


class _MultiModalModel(nn.Module):
    _mark_tower_model = SupportsMultiModal._mark_tower_model


def _selector_only_offloader(selectors: set[str]) -> UVAOffloader:
    offloader = UVAOffloader.__new__(UVAOffloader)
    offloader.cpu_offload_params = selectors
    return offloader


def test_uva_tower_selector_uses_qualified_parameter_path() -> None:
    offloader = _selector_only_offloader({"visual"})

    assert offloader._is_parameter_selected("patch_embed.proj.weight", "visual")
    assert not offloader._is_parameter_selected("patch_embed.proj.weight", "audio")
    assert not offloader._is_parameter_selected("patch_embed.proj.weight", "visualizer")
    assert _selector_only_offloader(set())._is_parameter_selected(
        "patch_embed.proj.weight", "visual"
    )


def test_uva_multi_segment_selector_remains_segment_exact() -> None:
    offloader = _selector_only_offloader({"visual.patch_embed"})

    assert offloader._is_parameter_selected("patch_embed.proj.weight", "visual")
    assert not offloader._is_parameter_selected("patch_embedding.proj.weight", "visual")


def test_tower_registration_uses_single_component_wrapper() -> None:
    model = _MultiModalModel()
    offloader = _RecordingOffloader()
    previous_offloader = get_offloader()
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            multimodal_config=SimpleNamespace(get_limit_per_prompt=lambda _modality: 1)
        )
    )

    set_offloader(offloader)
    try:
        with model._mark_tower_model(config, "image"):
            model.visual = nn.Linear(4, 4)
    finally:
        set_offloader(previous_offloader)

    assert model._tower_model_names == ["visual"]
    assert offloader.component_calls == [(model.visual, "visual")]


def test_prefetch_inherits_noop_component_wrapper() -> None:
    tower = nn.Linear(4, 4)
    offloader = PrefetchOffloader.__new__(PrefetchOffloader)

    assert PrefetchOffloader.wrap_module is BaseOffloader.wrap_module
    assert offloader.wrap_module(tower, parameter_prefix="visual") is tower
