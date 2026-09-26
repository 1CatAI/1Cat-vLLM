# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Mapping

import torch

from vllm.config import VllmConfig


def uses_dflash_selector_engine(vllm_config: object) -> bool:
    """Return whether the configured DFlash algorithm exposes selector inputs.

    Dispatch is based on the inference algorithm's configuration contract, not
    a model, architecture, or checkpoint identity.
    """
    speculative_config = getattr(vllm_config, "speculative_config", None)
    if (
        speculative_config is None
        or getattr(speculative_config, "method", None) != "dflash"
    ):
        return False
    draft_model_config = getattr(speculative_config, "draft_model_config", None)
    hf_config = getattr(draft_model_config, "hf_config", None)
    dflash_config = getattr(hf_config, "dflash_config", None) or {}
    if not isinstance(dflash_config, Mapping):
        return False
    try:
        return int(dflash_config.get("selector_top_k", 0) or 0) > 0
    except (TypeError, ValueError):
        return False


def uses_native_mtp(vllm_config: object) -> bool:
    """Return whether speculation uses the model's native MTP layers."""
    speculative_config = getattr(vllm_config, "speculative_config", None)
    return getattr(speculative_config, "method", None) == "mtp"


def init_speculator(vllm_config: VllmConfig, device: torch.device):
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    if speculative_config.method == "dflash":
        draft_config = speculative_config.draft_model_config
        if draft_config is None:
            raise ValueError("method='dflash' requires a draft model config")
        if uses_dflash_selector_engine(vllm_config):
            from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import (
                DFlash2Speculator,
            )

            return DFlash2Speculator(vllm_config, device)
        from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

        return DFlashSpeculator(vllm_config, device)
    if speculative_config.use_eagle():
        from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator

        return EagleSpeculator(vllm_config, device)
    raise NotImplementedError(f"{speculative_config.method} is not supported yet.")
