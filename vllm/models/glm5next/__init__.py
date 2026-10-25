# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM model implementations are loaded only when requested by the registry."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .nvidia.model import Glm5NextForCausalLM, Glm5NextForConditionalGeneration
    from .nvidia.mtp import Glm5NextMTP

__all__ = [
    "Glm5NextForCausalLM",
    "Glm5NextForConditionalGeneration",
    "Glm5NextMTP",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    from vllm.platforms import current_platform

    if current_platform.is_rocm() or current_platform.is_xpu():
        raise NotImplementedError("GLM-5.3-Flash currently supports NVIDIA CUDA only.")
    from .nvidia.model import Glm5NextForCausalLM, Glm5NextForConditionalGeneration
    from .nvidia.mtp import Glm5NextMTP

    return {
        "Glm5NextForCausalLM": Glm5NextForCausalLM,
        "Glm5NextForConditionalGeneration": Glm5NextForConditionalGeneration,
        "Glm5NextMTP": Glm5NextMTP,
    }[name]
