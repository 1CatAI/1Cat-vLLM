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


sys.modules[__name__].__class__ = _LogStateModule
