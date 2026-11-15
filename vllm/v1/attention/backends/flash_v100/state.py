# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared one-shot implementation state (one owner per binding)."""

from __future__ import annotations

import sys
import types

from vllm.logger import log_once_seen, set_log_once_state
from vllm.v1.attention.backends.flash_v100.spec.compatibility import LOG_KEYS


class _LogStateModule(types.ModuleType):
    """Legacy flags are live views of the logger's process-wide event keys."""

    def __getattr__(self, name):
        if name in LOG_KEYS:
            return log_once_seen(LOG_KEYS[name])
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in LOG_KEYS:
            set_log_once_state(LOG_KEYS[name], value)
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in LOG_KEYS:
            set_log_once_state(LOG_KEYS[name], False)
        else:
            super().__delattr__(name)


def _make_log_property(log_key: str) -> property:
    def _get(_self: object) -> bool:
        return log_once_seen(log_key)

    def _set(_self: object, val: bool) -> None:
        set_log_once_state(log_key, val)

    return property(_get, _set)


for _log_attr, _log_key in LOG_KEYS.items():
    setattr(_LogStateModule, _log_attr, _make_log_property(_log_key))


sys.modules[__name__].__class__ = _LogStateModule
