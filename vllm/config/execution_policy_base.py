# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Common provenance and calculation hashing for component-owned policies."""

from collections.abc import Callable
from typing import ClassVar

from pydantic import Field

from vllm.config.utils import config, hash_factors, resolve_legacy_fields


@config
class ExecutionPolicy:
    """Common provenance; computation never consumes legacy names."""

    aliases: ClassVar[dict[str, str]] = {}
    sources: dict[str, str] = Field(default_factory=dict, init=False)
    """Initialization source for each field; excluded from calculation hashes."""
    active: bool = Field(default=True, init=False)
    """Whether the owning feature can affect this engine's computation."""

    hash_fields: tuple[str, ...] | None = Field(default=None, init=False)
    """Effective computation fields; resource and unused feature choices are omitted."""

    legacy_reader: ClassVar[Callable[[str], object] | None] = None
    """Optional initialization parser; None uses the registered variable parser."""

    def resolve(self) -> None:
        pending = {
            field: alias
            for field, alias in self.aliases.items()
            if field not in self.sources
        }
        resolve_legacy_fields(self, pending, reader=type(self).legacy_reader)

    def compute_hash(self) -> str:
        return hash_factors(
            {
                field: getattr(self, field)
                for field in (
                    self.hash_fields if self.hash_fields is not None else self.aliases
                )
            }
            if self.active
            else {}
        )


@config
class DeferredExecutionPolicy(ExecutionPolicy):
    """Capture all inputs once while retaining the consumer's error admission."""

    errors: dict[str, str] = Field(default_factory=dict, init=False)
    """Initialization parse failures raised only when the field is consumed."""

    def resolve(self):
        resolve_legacy_fields(
            self,
            {
                field: alias
                for field, alias in self.aliases.items()
                if field not in self.sources
            },
            reader=type(self).legacy_reader,
            deferred_errors=self.errors,
        )

    def value(self, field):
        if field in self.errors:
            raise ValueError(self.errors[field])
        return getattr(self, field)
