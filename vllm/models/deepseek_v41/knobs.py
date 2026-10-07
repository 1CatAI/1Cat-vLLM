# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Environment-knob helpers for the DeepSeek-V4.1 port (PORT_DESIGN §2.1 rule 4, ``vllm/envs.py`` row).

Each lane declares its own ``VLLM_DS41_<LANE>_*`` flags in its own module through these helpers;
this module holds no flags of other lanes. Parsing is strict: a malformed value raises ``ValueError``
naming the variable -- never a silent fallback to the default.
"""

from __future__ import annotations

import os

PREFIX = "VLLM_DS41_"

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


def _check_name(name: str) -> None:
    if not name.startswith(PREFIX) or len(name) == len(PREFIX):
        raise ValueError(f"DeepSeek-V4.1 knob {name!r} must be named {PREFIX}<LANE>_<FLAG>")


def _raw(name: str) -> str | None:
    _check_name(name)
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value if value != "" else None


def env_bool(name: str, default: bool) -> bool:
    """1/true/yes/on -> True, 0/false/no/off -> False (case-insensitive); unset or empty -> default."""
    value = _raw(name)
    if value is None:
        return default
    lowered = value.lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise ValueError(f"{name}={value!r} is not a boolean (expected one of {sorted(_TRUE | _FALSE)})")


def env_int(name: str, default: int, *, minimum: int | None = None, maximum: int | None = None) -> int:
    """Base-10 integer; unset or empty -> default. Bounds are inclusive and checked on the default too."""
    value = _raw(name)
    if value is None:
        result = default
    else:
        try:
            result = int(value, 10)
        except ValueError as exc:
            raise ValueError(f"{name}={value!r} is not a base-10 integer") from exc
    if minimum is not None and result < minimum:
        raise ValueError(f"{name}={result} is below the minimum {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name}={result} is above the maximum {maximum}")
    return result


def env_str(name: str, default: str, *, choices: tuple[str, ...] | None = None) -> str:
    """Raw string; unset or empty -> default. With ``choices`` the value (and the default) must be one of them."""
    value = _raw(name)
    result = default if value is None else value
    if choices is not None and result not in choices:
        raise ValueError(f"{name}={result!r} is not one of {choices}")
    return result
