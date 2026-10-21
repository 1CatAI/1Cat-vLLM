# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qualified FA2 library discovery for heterogeneous SM70/RTX deployments.

These optional builds share the default FA2 operator namespace. Selection must
follow the worker's device before loading any extension; the caller retains the
ordinary single-library fallback when no qualified library is installed.
"""

import os

_VOLTA_FILENAME = "_vllm_fa2_C_sm70.abi3.so"
_AMPERE_FILENAME = "_vllm_fa2_C_sm80.abi3.so"


def has_qualified_fa2_library(directory: str) -> bool:
    """Discover dedicated builds without loading one or initializing CUDA."""
    return any(
        os.path.exists(os.path.join(directory, filename))
        for filename in (_VOLTA_FILENAME, _AMPERE_FILENAME)
    )


def qualified_fa2_library_path(
    capability: tuple[int, int], directory: str
) -> str | None:
    """Prefer the qualified Volta or Ampere/Ada binary for this worker."""
    if capability == (7, 0):
        filename = _VOLTA_FILENAME
    elif capability[0] >= 8:
        filename = _AMPERE_FILENAME
    else:
        return None
    path = os.path.join(directory, filename)
    return path if os.path.exists(path) else None
