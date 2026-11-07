# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache


def init_model_state(
    vllm_config: VllmConfig,
    model: nn.Module,
    encoder_cache: EncoderCache | None,
    device: torch.device,
):
    get_model_state_cls = getattr(model, "get_model_state_cls", None)
    if callable(get_model_state_cls):
        model_state_cls = get_model_state_cls()
        return model_state_cls(vllm_config, model, encoder_cache, device)

    if "WhisperForConditionalGeneration" in vllm_config.model_config.architectures:
        from vllm.v1.worker.gpu.model_states.whisper import WhisperModelState

        return WhisperModelState(vllm_config, model, encoder_cache, device)

    hf_text_config = getattr(vllm_config.model_config, "hf_text_config", None)
    if getattr(hf_text_config, "model_type", None) == "deepseek_v41_text":
        from vllm.v1.worker.gpu.model_states.deepseek_v41 import (
            DeepseekV41ModelState,
        )

        return DeepseekV41ModelState(vllm_config, model, encoder_cache, device)

    if vllm_config.model_config.is_hybrid:
        from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

        return MambaHybridModelState(vllm_config, model, encoder_cache, device)

    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    return DefaultModelState(vllm_config, model, encoder_cache, device)
