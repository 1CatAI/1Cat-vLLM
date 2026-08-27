# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
from torch import nn

from vllm.v1.worker.gpu.model_states import init_model_state


def test_model_can_select_custom_model_state() -> None:
    captured: dict[str, object] = {}

    class CustomModelState:
        def __init__(self, vllm_config, model, encoder_cache, device) -> None:
            captured.update(
                vllm_config=vllm_config,
                model=model,
                encoder_cache=encoder_cache,
                device=device,
            )

    class CustomModel(nn.Module):
        @staticmethod
        def get_model_state_cls():
            return CustomModelState

    model = CustomModel()
    vllm_config = SimpleNamespace()
    device = torch.device("cpu")

    model_state = init_model_state(vllm_config, model, None, device)

    assert isinstance(model_state, CustomModelState)
    assert captured == {
        "vllm_config": vllm_config,
        "model": model,
        "encoder_cache": None,
        "device": device,
    }
