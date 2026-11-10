# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serializable compatibility inputs for providers initialized in workers."""

import os
from collections.abc import Iterable
from typing import Any

from pydantic import Field

from vllm.config.utils import config


@config
class LegacyInputs:
    values: dict[str, Any] = Field(default_factory=dict, init=False)
    """Parsed initialization inputs, including registered alias precedence."""
    errors: dict[str, tuple[str, str]] = Field(default_factory=dict, init=False)
    """Deferred parse failures; unused formats do not raise worker errors."""
    present: tuple[str, ...] = Field(default=(), init=False)
    """Explicit legacy names, retained separately from effective defaults."""
    captured: bool = Field(default=False, init=False)
    """An empty snapshot is still a completed initialization."""

    def capture(self, names: Iterable[str]) -> None:
        if self.captured:
            return
        from vllm import envs

        names = tuple(dict.fromkeys(names))
        self.present = tuple(name for name in names if name in os.environ)
        for name in names:
            try:
                self.values[name] = envs.environment_variables[name]()
            except (ValueError, TypeError) as error:
                self.errors[name] = (type(error).__name__, str(error))
        self.captured = True

    def value(self, name: str):
        if name in self.errors:
            kind, message = self.errors[name]
            raise (TypeError if kind == "TypeError" else ValueError)(message)
        return self.values[name]

    def is_set(self, name: str) -> bool:
        return name in self.present
