# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-owned attention storage, independent of backend selection."""

from dataclasses import dataclass, field
from typing import TypeVar, cast

import torch

from vllm.runtime_resources import current_runtime_resources


class WorkspaceCache(dict):
    """Retain buffers borrowed by captured graphs across later eager growth."""

    def __init__(self):
        super().__init__()
        self.captured: dict[int, object] = {}

    def clear(self):
        super().clear()
        self.captured.clear()


def retain_for_capture(cache: dict, value: object, tensor: torch.Tensor) -> None:
    if (
        isinstance(cache, WorkspaceCache)
        and tensor.is_cuda
        and torch.cuda.is_current_stream_capturing()
    ):
        cache.captured[id(value)] = value


@dataclass
class AttentionWorkspaces:
    caches: dict[str, dict | set] = field(default_factory=dict)

    def close(self) -> None:
        # Graph owners must release their captures before this checkpoint.
        for cache in self.caches.values():
            cache.clear()
        self.caches.clear()


Cache = TypeVar("Cache", dict, set)


def workspace_cache(name: str, standalone: Cache) -> Cache:
    resources = current_runtime_resources()
    if resources is None:
        return standalone
    owner = resources.get("attention_workspaces")
    if owner is None:
        owner = resources["attention_workspaces"] = AttentionWorkspaces()
    if name not in owner.caches:
        owner.caches[name] = set() if isinstance(standalone, set) else WorkspaceCache()
    return cast(Cache, owner.caches[name])
