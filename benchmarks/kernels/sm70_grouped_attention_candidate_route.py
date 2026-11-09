# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bind a private q8 experiment to the Flash-V100 interface's native module."""

from collections.abc import Callable
from functools import wraps
from typing import Any

import torch


def install_grouped_attention_candidate(candidate: Callable[..., Any]) -> None:
    """Install explicitly; importing this module does not change serving.

    The package-relative and top-level extension imports can load the same
    DSO into distinct Python module objects. Patch the object actually used
    by the public interface, rather than independently importing its name.
    The caller owns the candidate's build/quality gates and experiment flag.
    """
    from flash_attn_v100 import flash_attn_interface

    native = flash_attn_interface.flash_attn_v100_cuda
    original = native.grouped_e4m3_fp32_paged_fwd

    @wraps(original)
    def run(q: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        if q.shape == (8, 6, 256) and torch.cuda.is_current_stream_capturing():
            return candidate(q, *args, **kwargs)
        return original(q, *args, **kwargs)

    native.grouped_e4m3_fp32_paged_fwd = run
