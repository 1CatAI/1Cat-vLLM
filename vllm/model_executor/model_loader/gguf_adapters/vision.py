# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lazy vision registrations shared by native GGUF architecture adapters.

Each vision adapter exposes ``name_map``, ``unquantized_modules`` and
``weights(dtype, filtered_map=None)``. Architecture-specific tensor geometry
and conversions belong in that adapter, never in the generic loader.
"""

from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class VisionAdapterSpec:
    module: str
    class_name: str
    language_prefix: str = "language_model."

    def create(self, config, tensors):
        module = import_module(f"{__package__}.{self.module}")
        return getattr(module, self.class_name)(config, tensors)


_VISION_ADAPTERS = {
    "qwen4_exp_text": VisionAdapterSpec("qwen3_vision", "Qwen3VisionGGUF"),
    "glm5_next_text": VisionAdapterSpec("glm5next_vision", "Glm5NextVisionGGUF"),
}


def get_vision_adapter_spec(text_model_type: str) -> VisionAdapterSpec | None:
    return _VISION_ADAPTERS.get(text_model_type)
