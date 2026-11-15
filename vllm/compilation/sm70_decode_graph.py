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


@contextmanager
def cudagraph_capture_stream_workspace_scope(
    device: torch.device,
    get_capture_context,
) -> Generator[None, None, None]:
    long_workspaces = None
    default_stream_id = None
    capture_stream_id = None
    if device.type == "cuda":
        from vllm.v1.attention.ops import sm70_grouped_long

        long_workspaces = sm70_grouped_long._WORKSPACES
        default_stream_id = torch.cuda.default_stream(device).cuda_stream
        capture_stream_id = get_capture_context().stream.cuda_stream
        if default_stream_id != capture_stream_id:
            for key, bank in list(long_workspaces.items()):
                if len(key) == 5 and key[3] == device and key[4] == default_stream_id:
                    long_workspaces.setdefault((*key[:4], capture_stream_id), bank)
    try:
        yield
    finally:
        if (
            long_workspaces is not None
            and default_stream_id is not None
            and capture_stream_id is not None
            and default_stream_id != capture_stream_id
        ):
            for key, bank in list(long_workspaces.items()):
                if len(key) == 5 and key[3] == device and key[4] == capture_stream_id:
                    long_workspaces.setdefault((*key[:4], default_stream_id), bank)


__all__ = [
    "cudagraph_capture_stream_workspace_scope",
    "is_sm70_decode_graph_compiling",
    "sm70_decode_graph_compilation",
    "use_sm70_decode_graph_semantics",
]
