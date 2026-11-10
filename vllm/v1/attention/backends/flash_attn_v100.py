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
import warnings

from vllm.v1.attention.backends import flash_v100 as _package

warnings.warn(
    "flash_attn_v100 is deprecated; import owners from "
    "vllm.v1.attention.backends.flash_v100 instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = sorted(
    {
        name
        for module in _package.SUBMODULES
        for name in vars(module)
        if not name.startswith("_")
    }
)


def _owners(name: str) -> list[types.ModuleType]:
    return [module for module, _ in _package._compatibility_bindings(name)]


def _canonical_name(name: str) -> str:
    bindings = _package._compatibility_bindings(name)
    return bindings[0][1] if bindings else name


_MISSING = object()


class _ForwardingModule(types.ModuleType):
    def __getattr__(self, name: str):
        bindings = _package._compatibility_bindings(name)
        if not bindings:
            raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")
        module, target = bindings[0]
        return getattr(module, target)

    def __setattr__(self, name: str, value) -> None:
        bindings = _package._compatibility_bindings(name)
        if not bindings or name.startswith("__"):
            super().__setattr__(name, value)
            return
        module, target = bindings[0]
        current = getattr(module, target, _MISSING)
        for module, target in bindings:
            if getattr(module, target, _MISSING) is current:
                setattr(module, target, value)

    def __delattr__(self, name: str) -> None:
        bindings = _package._compatibility_bindings(name)
        if not bindings:
            super().__delattr__(name)
            return
        for module, target in bindings:
            delattr(module, target)

    def __dir__(self):
        names = set(super().__dir__()) | set(_package.COMPATIBILITY_ALIASES)
        for module in _package.SUBMODULES:
            names.update(vars(module))
        return sorted(names)


sys.modules[__name__].__class__ = _ForwardingModule
