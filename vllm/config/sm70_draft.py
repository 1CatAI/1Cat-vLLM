# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-engine drafter policy and its legacy environment adapter."""

import os
from typing import Literal

from pydantic import Field

from vllm.config.utils import config

DraftUnit = Literal["d2a", "d2b", "d1a"]
UNITS: tuple[DraftUnit, ...] = ("d2a", "d2b", "d1a")


def parse_legacy_units(raw: str) -> tuple[DraftUnit, ...]:
    raw = raw.strip().lower()
    if raw in ("1", "all", "on", "true", "yes"):
        return UNITS
    selected = {item.strip() for item in raw.split(",")}
    return tuple(unit for unit in UNITS if unit in selected)


def _legacy_units() -> tuple[DraftUnit, ...]:
    return parse_legacy_units(os.getenv("ONECAT_DRAFT47", ""))


@config
class Sm70DraftConfig:
    """Exact drafter shortcuts resolved once before layer construction."""

    units: tuple[DraftUnit, ...] = Field(default_factory=_legacy_units)
    """Enabled shortcuts; an explicit empty tuple overrides the legacy switch."""
