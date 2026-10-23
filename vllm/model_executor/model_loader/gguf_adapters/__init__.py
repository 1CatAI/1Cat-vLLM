# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Architecture adapters, independent of Transformers model construction."""

from importlib import import_module

# Values remain lazy so adding an architecture does not import its kernels or
# model implementation while another model's loader is being configured.
_TEXT_ADAPTERS = {
    "qwen4_exp_text": ("qwen4exp", "Qwen4ExpAdapter"),
    "qwen3_5_moe_text": ("qwen35_moe", "Qwen35MoeAdapter"),
    "qwen3_5_text": ("qwen35", "Qwen35Adapter"),
    "glm5_next_text": ("glm5next", "Glm5NextAdapter"),
}

# Drafts can share the target's config model_type while owning a different
# module tree and checkpoint subset. Resolve their explicit architecture first.
_DRAFT_ADAPTERS = {
    "Glm5NextMTPModel": ("glm5next", "Glm5NextMTPAdapter"),
}


def get_gguf_adapter(config, tp_size=1):
    text = config.get_text_config()
    if getattr(text, "gguf_architecture", None) == "dflash" or "DFlash2DraftModel" in (
        getattr(text, "architectures", None) or []
    ):
        from .dflash import DFlashAdapter

        return DFlashAdapter(text, tp_size=tp_size)
    entry = next(
        (
            _DRAFT_ADAPTERS[arch]
            for arch in (getattr(config, "architectures", None) or [])
            if arch in _DRAFT_ADAPTERS
        ),
        None,
    )
    is_draft = entry is not None
    if entry is None:
        entry = _TEXT_ADAPTERS.get(text.model_type)
    if entry is None:
        return None
    module, name = entry
    adapter = getattr(import_module(f"{__name__}.{module}"), name)
    instance = adapter(text, tp_size=tp_size)
    from .vision import get_vision_adapter_spec

    vision = get_vision_adapter_spec(text.model_type)
    instance.language_model_prefix = ""
    if (
        not is_draft
        and vision is not None
        and getattr(config, "vision_config", None) is not None
    ):
        instance.language_model_prefix = vision.language_prefix.removesuffix(".")
    return instance
