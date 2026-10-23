# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Proposer method registration for attention feature providers.

Backend-owned implementations register here without importing proposer engines.
Selection consumes the same speculative method as the proposer, including when
attention metadata is constructed before the proposer engine itself.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any, Protocol


class SpecFeature(Protocol):
    def prepare(
        self,
        state: Any,
        attn_metadata: Any,
        common_attn_metadata: Any,
        tree_verify: bool,
        prepared: Any,
    ) -> None: ...


class SpecFeatureRegistry:
    """Immutable registrations and independent per-builder feature instances."""

    def __init__(
        self,
        providers: Mapping[str, Callable[[], SpecFeature]],
        *,
        fallback: Callable[[], SpecFeature],
    ):
        self.providers = MappingProxyType(dict(providers))
        self.fallback = fallback

    def for_method(self, method: str | None) -> SpecFeature:
        return (
            self.providers.get(method, self.fallback)() if method else self.fallback()
        )
