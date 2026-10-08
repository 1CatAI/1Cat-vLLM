# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Declared speculative attention contracts and process-shared observations."""

from __future__ import annotations

from typing import Any

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backends.flash_v100 import state as _state
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata

logger = init_logger("vllm.v1.attention.backends.flash_attn_v100")


def validate_contract(
    layer: torch.nn.Module,
    attn_metadata: TritonAttentionMetadata,
    window_size: Any,
) -> None:
    if not getattr(layer, "is_dflash_draft_attn", False):
        return

    actual_causal = bool(getattr(attn_metadata, "causal", True))
    expected_causal = getattr(layer, "dflash_expected_causal", None)
    if expected_causal is None:
        raise RuntimeError(
            "FLASH_ATTN_V100 DFlash attention is missing its declared "
            "causality contract."
        )
    expected_causal = bool(expected_causal)
    if actual_causal != expected_causal:
        raise RuntimeError(
            "FLASH_ATTN_V100 DFlash causality mismatch: "
            f"model={expected_causal} metadata={actual_causal}."
        )

    declared_window = getattr(layer, "dflash_expected_sliding_window", None)
    expected_window = (
        (-1, -1)
        if declared_window is None
        else (
            int(declared_window) - 1,
            0 if expected_causal else int(declared_window) - 1,
        )
    )
    actual_window = window_size(actual_causal)
    if actual_window != expected_window:
        raise RuntimeError(
            "FLASH_ATTN_V100 DFlash sliding-window mismatch: "
            f"model={expected_window} backend={actual_window}."
        )

    signature = (
        getattr(layer, "layer_name", None),
        actual_causal,
        actual_window,
        getattr(layer, "dflash_rope_is_neox_style", None),
    )
    if signature not in _state._logged_dflash_attention_contracts:
        _state._logged_dflash_attention_contracts.add(signature)
        logger.info(
            "FLASH_ATTN_V100 DFlash attention contract: layer=%s "
            "causal=%s window=%s rope_neox=%s.",
            signature[0],
            actual_causal,
            actual_window,
            signature[3],
        )
