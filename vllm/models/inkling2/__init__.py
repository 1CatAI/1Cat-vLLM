# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 (Thinking Machines). This fork runs it on SM70 (V100):
implementations live under `sm70/`; configs and quantization metadata are
shared with upstream."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .sm70.loop import Inkling2LoopForConditionalGeneration
    from .sm70.model import Inkling2ForConditionalGeneration

__all__ = [
    "Inkling2ForConditionalGeneration",
    "Inkling2LoopForConditionalGeneration",
]


def __getattr__(name: str):
    if name == "Inkling2LoopForConditionalGeneration":
        from .sm70 import loop

        return loop.Inkling2LoopForConditionalGeneration
    if name == "Inkling2ForConditionalGeneration":
        from .sm70 import model

        return model.Inkling2ForConditionalGeneration
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
