# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synchronous diagnostic subscriptions; execution modules emit input events."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, NamedTuple, TypeVar

import torch

E = TypeVar("E")


class EventStream(Generic[E]):
    def __init__(self) -> None:
        self.subscribers: list[Callable[[E], None]] = []

    def subscribe(self, subscriber: Callable[[E], None]) -> None:
        self.subscribers.append(subscriber)

    def emit(self, event: E) -> None:
        # Observation order and exceptions retain the original synchronous path.
        for subscriber in self.subscribers:
            subscriber(event)


class PrefillReference(NamedTuple):
    key: torch.Tensor
    value: torch.Tensor
    output: torch.Tensor
    difference: torch.Tensor
    nan_count: int


@dataclass
class PrefillDebugEvent:
    layer: torch.nn.Module
    query: torch.Tensor
    key: torch.Tensor | None
    value: torch.Tensor | None
    kv_cache: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    attn_metadata: Any
    out_seq: torch.Tensor
    i: int
    start: int
    end: int
    seq_len: int
    num_kv_heads: int
    head_dim: int
    block_size: int
    causal: bool
    window_size: tuple[int, int]
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    debug_compare: bool
    dump_enabled: bool
    kv_cache_dtype: str
    scale: float
    dense: Any
    torch_reference: Any
    layer_info: Any
    reference: PrefillReference | None = None


prefill_debug = EventStream[PrefillDebugEvent]()


class DiagnosticMessage(NamedTuple):
    message: str
    args: tuple[object, ...]


diagnostic_messages = EventStream[DiagnosticMessage]()
