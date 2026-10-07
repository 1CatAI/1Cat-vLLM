# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1-Flash -- SM70 (Volta) port.

Layout and lane ownership: docs/models/deepseek-v4.1-flash/PORT_DESIGN.md §2.2 (v100-research).
Importing this package must stay cheap and side-effect free: every lane imports
``vllm.models.deepseek_v41.common.contracts``, which imports this ``__init__`` first.
The model class is therefore resolved lazily (PEP 562 ``__getattr__``) and a missing
implementation raises a clear error instead of failing at package import time.
"""

from __future__ import annotations

from typing import Any

__all__ = ["DeepseekV41ForCausalLM"]

_NOT_READY = (
    "DeepseekV41ForCausalLM is not available: {reason}. The DeepSeek-V4.1 port runs only on "
    "CUDA compute capability 7.x (SM70, V100) through vllm.models.deepseek_v41.sm70.model "
    "(PORT_DESIGN §2.2)."
)


def _load_model_class() -> Any:
    from vllm.platforms import current_platform

    if not current_platform.is_cuda():
        raise ImportError(_NOT_READY.format(reason="the current platform is not CUDA"))
    capability = current_platform.get_device_capability()
    if capability is None or capability.major != 7:
        raise ImportError(_NOT_READY.format(reason=f"device capability is {capability}, not 7.x"))
    try:
        from .sm70 import model as sm70_model
    except ModuleNotFoundError as exc:
        if exc.name == f"{__name__}.sm70.model":
            raise ImportError(_NOT_READY.format(
                reason="vllm/models/deepseek_v41/sm70/model.py does not exist yet (lane L-CORE P2)")) from exc
        raise
    return sm70_model.DeepseekV41ForCausalLM


def __getattr__(name: str) -> Any:
    if name == "DeepseekV41ForCausalLM":
        return _load_model_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
