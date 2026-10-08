# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility module for ``vllm.v1.attention.backends.flash_v100``.

The backend now lives in the ``flash_v100`` package. Attribute reads and
writes on this module are forwarded to the module that owns the name, so
existing imports and ``monkeypatch.setattr(flash_attn_v100, ...)`` keep
working. New code should import from the package.
"""

import sys
import types

from vllm.v1.attention.backends import flash_v100 as _package

__all__ = sorted(
    {
        name
        for module in _package.SUBMODULES
        for name in vars(module)
        if not name.startswith("_")
    }
)


def _owners(name: str) -> list[types.ModuleType]:
    return [module for module in _package.SUBMODULES if name in vars(module)]


class _ForwardingModule(types.ModuleType):
    def __getattr__(self, name: str):
        owners = _owners(name)
        if not owners:
            raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")
        return getattr(owners[0], name)

    def __setattr__(self, name: str, value) -> None:
        owners = _owners(name)
        if not owners or name.startswith("__"):
            super().__setattr__(name, value)
            return
        # A shared import (torch, envs, ...) is patched wherever it is the
        # same object; a backend name has exactly one owner.
        current = getattr(owners[0], name)
        for module in owners:
            if getattr(module, name) is current:
                setattr(module, name, value)

    def __delattr__(self, name: str) -> None:
        owners = _owners(name)
        if not owners:
            super().__delattr__(name)
            return
        for module in owners:
            delattr(module, name)

    def __dir__(self):
        names = set(super().__dir__())
        for module in _package.SUBMODULES:
            names.update(vars(module))
        return sorted(names)


sys.modules[__name__].__class__ = _ForwardingModule
