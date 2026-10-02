# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Documentation attached to an environment getter, without changing parsing."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

EnvCategory = Literal["configuration", "tuning", "experimental", "debug", "deprecated"]


@dataclass(frozen=True)
class EnvVarMetadata:
    description: str
    category: EnvCategory
    declared_default: str
    effective_default: str
    automatic_conditions: tuple[str, ...]
    acceleration_paths: tuple[str, ...]
    user_visible: bool


@dataclass(frozen=True)
class EnvVar:
    getter: Callable[[], Any]
    metadata: EnvVarMetadata

    def __call__(self) -> Any:
        return self.getter()


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
        ),
    )
