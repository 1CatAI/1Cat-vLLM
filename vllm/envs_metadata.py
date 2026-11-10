# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Documentation attached to an environment getter, without changing parsing."""

import os
import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace
from threading import Lock
from typing import Any, Literal

EnvCategory = Literal["configuration", "tuning", "experimental", "debug", "deprecated"]
DeprecationKind = Literal["alias", "experiment", "historical"]

_warned_names: set[str] = set()
_warning_lock = Lock()


def warn_deprecated_once(name: str, message: str) -> None:
    """Deduplicate explicit legacy inputs independently of warning filters."""
    if name not in os.environ:
        return
    with _warning_lock:
        if name in _warned_names:
            return
        _warned_names.add(name)
    warnings.warn(message, FutureWarning, stacklevel=3)


@dataclass(frozen=True)
class EnvVarMetadata:
    description: str
    category: EnvCategory
    declared_default: str
    effective_default: str
    automatic_conditions: tuple[str, ...]
    acceleration_paths: tuple[str, ...]
    user_visible: bool
    deprecated: bool = False
    deprecation_kind: DeprecationKind | None = None
    deprecation_reason: str | None = None
    deprecation_evidence: tuple[str, ...] = ()
    replacement: str | None = None


@dataclass(frozen=True)
class EnvVar:
    getter: Callable[[], Any]
    metadata: EnvVarMetadata
    name: str = ""

    def warn_if_deprecated(self) -> None:
        if self.metadata.deprecated and self.name:
            metadata = self.metadata
            replacement = (
                f" Use {metadata.replacement}." if metadata.replacement else ""
            )
            evidence = " ".join(metadata.deprecation_evidence)
            warn_deprecated_once(
                self.name,
                f"{self.name} is deprecated ({metadata.deprecation_kind}): "
                f"{metadata.deprecation_reason}.{replacement} Evidence: {evidence}. "
                "The compatibility input remains supported.",
            )

    def __call__(self) -> Any:
        self.warn_if_deprecated()
        return self.getter()


def bind_env_names(variables: dict[str, Callable[[], Any]]) -> None:
    """Attach registration names without reading inputs or invoking getters."""
    for name, variable in variables.items():
        if isinstance(variable, EnvVar):
            variables[name] = replace(variable, name=name)


def env_var(
    getter: Callable[[], Any],
    *,
    description: str,
    category: EnvCategory,
    declared_default: str,
    effective_default: str,
    automatic_conditions: tuple[str, ...],
    acceleration_paths: tuple[str, ...],
    user_visible: bool,
    deprecated: bool = False,
    deprecation_kind: DeprecationKind | None = None,
    deprecation_reason: str | None = None,
    deprecation_evidence: tuple[str, ...] = (),
    replacement: str | None = None,
) -> EnvVar:
    """Keep parsing and metadata at the same registration site in envs.py."""
    return EnvVar(
        getter,
        EnvVarMetadata(
            description,
            category,
            declared_default,
            effective_default,
            automatic_conditions,
            acceleration_paths,
            user_visible,
            deprecated,
            deprecation_kind,
            deprecation_reason,
            deprecation_evidence,
            replacement,
        ),
    )
