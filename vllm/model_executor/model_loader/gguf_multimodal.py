# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unquantized tower weights accompanying a native GGUF language checkpoint."""

from pathlib import Path

import torch
from safetensors import safe_open


class GGUFMultiModalWeights:
    """Keep tower weights separate from the native language adapter's layout."""

    def __init__(self, path: str):
        if not isinstance(path, str) or not path:
            raise ValueError("multimodal_safetensors must be a local safetensors file")
        self.path = Path(path)
        if self.path.suffix != ".safetensors" or not self.path.is_file():
            raise ValueError("multimodal_safetensors must be a local safetensors file")
        with safe_open(self.path, framework="pt", device="cpu") as reader:
            self.names = list(reader.keys())
        if not self.names:
            raise ValueError("The multimodal checkpoint contains no weights")

    def mapped_names(self, mapper):
        return mapper.apply_list(self.names) if mapper is not None else self.names

    def unquantized_modules(self, mapper) -> list[str]:
        return [
            name.removesuffix(".weight")
            for name in self.mapped_names(mapper)
            if name.endswith(".weight")
        ]

    def load(self, model: torch.nn.Module) -> set[str]:
        mapper = getattr(model, "hf_to_vllm_mapper", None)
        mapped = self.mapped_names(mapper)
        towers = tuple(
            name + "." for name in getattr(model, "_tower_model_names", ()) if name
        )
        if not towers or any(not name.startswith(towers) for name in mapped):
            raise ValueError(
                "Multimodal weights must belong only to active model towers"
            )
        expected = {
            name for name, _ in model.named_parameters() if name.startswith(towers)
        }
        if len(set(mapped)) != len(mapped) or set(mapped) != expected:
            raise ValueError(
                "Incomplete or unexpected multimodal weights: "
                f"missing={sorted(expected - set(mapped))}, "
                f"unexpected={sorted(set(mapped) - expected)}"
            )
        with safe_open(self.path, framework="pt", device="cpu") as reader:
            return model.load_weights(
                (name, reader.get_tensor(name)) for name in self.names
            )


def native_gguf_language_model(model, adapter):
    """Resolve the model-declared scope without changing adapter tensor names."""
    prefix = getattr(adapter, "language_model_prefix", "")
    return model.get_submodule(prefix) if prefix else model


def qualify_native_gguf_names(names, adapter):
    prefix = getattr(adapter, "language_model_prefix", "")
    return {
        f"{prefix}.{name}" if prefix else name: value for name, value in names.items()
    }
