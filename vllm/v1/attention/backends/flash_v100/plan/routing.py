# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ordered admission/attempt/decline with immediate route observations."""

from collections.abc import Callable, Iterable
from typing import Protocol, TypeVar

from vllm.v1.attention.backends.flash_v100 import routing as accounting

RecordRoute = Callable[[str], None]
C = TypeVar("C")
R = TypeVar("R")
C_contra = TypeVar("C_contra", contravariant=True)
R_co = TypeVar("R_co", covariant=True)


class Candidate(Protocol[C_contra, R_co]):
    def admit(self, context: C_contra) -> bool:
        """Pure predicate; admission alone performs no preparation."""
        ...

    def run(self, context: C_contra, record: RecordRoute) -> R_co | None:
        """May prepare/execute and decline; retain all prior side effects."""
        ...


def execute(context: C, candidates: Iterable[Candidate[C, R]]) -> R:
    def record(name: str) -> None:
        # Keep dynamic observations at their original pre/post-op positions.
        accounting._record_route(name)

    for candidate in candidates:
        if candidate.admit(context):
            result = candidate.run(context, record)
            if result is not None:
                return result
    raise RuntimeError("No attention candidate completed the admitted request")


class _LegacyObservation:
    def admit(self, name: str) -> bool:
        return True

    def run(self, name: str, record: RecordRoute) -> bool:
        record(name)
        return True


def record_legacy(name: str) -> None:
    """Temporary adapter for private decode calls outside the forward loop."""
    execute(name, (_LegacyObservation(),))
