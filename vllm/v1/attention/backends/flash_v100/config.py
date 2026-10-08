# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Environment boundary for Flash-V100 policy.

Reads retain their original call sites and short-circuit order. Constructor
assignments capture their values; all other call sites remain dynamic, including
the registered environment module's existing cache semantics. No additional
cache is introduced here.
"""

from __future__ import annotations

import os
from typing import Any, overload

import vllm.envs as envs


def registered(name: str) -> Any:
    return getattr(envs, name)


@overload
def raw(name: str, default: str) -> str: ...


@overload
def raw(name: str, default: None = None) -> str | None: ...


def raw(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)


def env_is_set(name: str) -> bool:
    return name in os.environ
