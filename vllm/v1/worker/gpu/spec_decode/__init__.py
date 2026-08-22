# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.config import VllmConfig


def init_speculator(vllm_config: VllmConfig, device: torch.device):
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    if speculative_config.method == "dflash":
        draft_config = speculative_config.draft_model_config
        if draft_config is None:
            raise ValueError("method='dflash' requires a draft model config")
        if "DFlash2DraftModel" in (draft_config.architectures or []):
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
