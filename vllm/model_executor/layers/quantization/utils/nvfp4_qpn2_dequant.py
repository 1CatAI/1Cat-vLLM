# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility alias for the QPN implementation."""

import sys
from typing import Any

from vllm.model_executor.kernels.linear.qpn import nvfp4_dequant as _implementation


def __getattr__(name: str) -> Any:
    return getattr(_implementation, name)


sys.modules[__name__] = _implementation
