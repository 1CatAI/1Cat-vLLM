# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-family dynamic thinking / non-thinking default sampling presets."""

from __future__ import annotations

from typing import Any

QWEN38_THINKING_SAMPLING_DEFAULTS: dict[str, Any] = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
}

QWEN38_NON_THINKING_SAMPLING_DEFAULTS: dict[str, Any] = {
    "temperature": 0.7,
    "top_p": 0.80,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
}

_QWEN38_DYNAMIC_SAMPLING_MODEL_TYPES: frozenset[str] = frozenset(
    {
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
        "qwen4_exp",
        "qwen4_exp_text",
    }
)


def supports_dynamic_thinking_sampling(model_config: Any) -> bool:
    """Return whether ``model_config`` uses Qwen3.8/3.5 dynamic sampling presets."""
    model_id = str(getattr(model_config, "model", "") or "").lower()
    served = " ".join(
        str(x).lower() for x in (getattr(model_config, "served_model_name", None) or ())
    )
    if any(tag in model_id for tag in ("dflash", "-mtp", "_mtp")):
        return False
    if "qwen3.8" in model_id or "qwen3.8" in served:
        return True
    hf_cfg = getattr(model_config, "hf_config", None)
    model_type = getattr(hf_cfg, "model_type", None)
    text_cfg = getattr(model_config, "hf_text_config", None)
    text_model_type = getattr(text_cfg, "model_type", None)
    return (
        model_type in _QWEN38_DYNAMIC_SAMPLING_MODEL_TYPES
        or text_model_type in _QWEN38_DYNAMIC_SAMPLING_MODEL_TYPES
    )


def get_dynamic_thinking_sampling_preset(
    model_config: Any, *, thinking_enabled: bool = True
) -> dict[str, Any] | None:
    """Return the official thinking or non-thinking sampling preset if supported."""
    if not supports_dynamic_thinking_sampling(model_config):
        return None
    return dict(
        QWEN38_THINKING_SAMPLING_DEFAULTS
        if thinking_enabled
        else QWEN38_NON_THINKING_SAMPLING_DEFAULTS
    )
