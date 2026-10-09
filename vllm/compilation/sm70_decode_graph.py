# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile-phase selection for the SM70 Qwen3.8 decode graph.

The Qwen3.8 V100 serving lane uses two compiled backbones that share the same
module parameters: a dynamic prefill graph and a small-shape decode graph.
This context is active only while tracing/capturing the latter.
"""

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar

import torch

from vllm.config.execution_policy import graph_policy

_sm70_decode_graph_compilation = ContextVar(
    "sm70_decode_graph_compilation", default=False
)


@contextmanager
def sm70_decode_graph_compilation(enabled: bool = True) -> Generator[None, None, None]:
    token = _sm70_decode_graph_compilation.set(enabled)
    try:
        yield
    finally:
        _sm70_decode_graph_compilation.reset(token)


@torch.compiler.assume_constant_result
def is_sm70_decode_graph_compiling() -> bool:
    """Return a trace-time constant for the selected SM70 compilation phase."""
    return _sm70_decode_graph_compilation.get()


def use_sm70_decode_graph_semantics(policy=None) -> bool:
    """Preserve legacy behavior unless the dual-compile lane is active."""
    policy = policy if policy is not None else graph_policy()
    return not policy.dual_compile or is_sm70_decode_graph_compiling()


__all__ = [
    "is_sm70_decode_graph_compiling",
    "sm70_decode_graph_compilation",
    "use_sm70_decode_graph_semantics",
]
