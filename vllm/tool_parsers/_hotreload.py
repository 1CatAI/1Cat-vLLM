# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare parser implementations privately before publishing a generation.

Only trusted, already imported implementation modules are re-executed. Stable
parser base classes and serving classes are never reloaded or patched. Old
requests keep their classes and module globals, including after a failed edit.
"""

from __future__ import annotations

import builtins
import importlib.util
import sys
import threading
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.tool_parsers import ToolParserManager

_VERSION = 0
reload_lock = threading.RLock()
_PACKAGE = "vllm.tool_parsers"
_BASE = _PACKAGE + ".abstract_tool_parser"


def current_version() -> int:
    return _VERSION


def _implementation_modules():
    from vllm.tool_parsers import ToolParser

    return {
        name: module
        for name, module in dict(sys.modules).items()
        if name.startswith(_PACKAGE + ".")
        and name not in (_BASE, __name__)
        and module is not None
        and not hasattr(module, "__path__")
        and not any(
            isinstance(value, type)
            and value.__module__ == name
            and issubclass(value, ToolParser)
            and not value.supports_hot_reload
            for value in vars(module).values()
        )
    }


@dataclass
class ParserReload:
    manager: type[ToolParserManager]
    modules: dict[str, ModuleType]
    version: int

    def replacement(self, cls):
        if cls is None:
            return None
        module = self.modules.get(cls.__module__)
        if module is None:
            return cls
        from vllm.tool_parsers import ToolParser

        replacement = getattr(module, cls.__name__)
        if not isinstance(replacement, type) or not issubclass(replacement, ToolParser):
            raise TypeError(f"Invalid reloaded parser {cls.__module__}.{cls.__name__}")
        return replacement


def prepare_reload() -> ParserReload:
    """Called under reload_lock; never mutate a live module or registry."""
    from vllm.tool_parsers import ToolParserManager

    class Manager(ToolParserManager):
        tool_parsers = dict(ToolParserManager.tool_parsers)
        lazy_parsers = dict(ToolParserManager.lazy_parsers)

    loaded = dict(sys.modules)
    originals = _implementation_modules()
    modules: dict[str, ModuleType] = {}
    proxies: dict[str, ModuleType] = {}
    for name in ("vllm", _PACKAGE, _BASE):
        proxy = ModuleType(name)
        proxy.__dict__.update(vars(loaded[name]))
        proxies[name] = proxy
    proxies["vllm"].__dict__["tool_parsers"] = proxies[_PACKAGE]
    proxies[_PACKAGE].__dict__["abstract_tool_parser"] = proxies[_BASE]
    proxies[_PACKAGE].__dict__["ToolParserManager"] = Manager
    proxies[_BASE].__dict__["ToolParserManager"] = Manager

    def isolated_import(name, globals=None, locals=None, fromlist=(), level=0):
        absolute = (
            importlib.util.resolve_name("." * level + name, globals["__package__"])
            if level
            else name
        )
        if absolute in originals:
            module = stage(absolute)
            return module if fromlist else proxies["vllm"]
        if absolute in proxies:
            module = proxies[absolute]
            for item in fromlist or ():
                child = absolute + "." + item
                if child in originals:
                    setattr(module, item, stage(child))
            return module if fromlist else proxies["vllm"]
        if absolute.startswith(_PACKAGE + "."):
            if absolute in loaded:
                # Identity-bound implementations remain canonical dependencies.
                return builtins.__import__(name, globals, locals, fromlist, level)
            raise ValueError(f"New parser dependency requires a restart: {absolute}")
        return builtins.__import__(name, globals, locals, fromlist, level)

    def stage(name):
        if name in modules:
            return modules[name]
        old = originals[name]
        spec = old.__spec__
        source = spec.loader.get_source(name) if spec and spec.loader else None
        if source is None:
            raise ValueError(f"No reloadable Python source for {name}")
        module = ModuleType(name)
        module.__dict__.update(
            {
                "__file__": old.__file__,
                "__spec__": spec,
                "__package__": old.__package__,
                "__loader__": old.__loader__,
                "__builtins__": dict(vars(builtins), __import__=isolated_import),
            }
        )
        modules[name] = module
        setattr(proxies[_PACKAGE], name.rsplit(".", 1)[1], module)
        exec(compile(source, old.__file__, "exec"), module.__dict__)
        return module

    for name in sorted(originals):
        stage(name)
    for name, (module_name, class_name) in Manager.lazy_parsers.items():
        if module_name in modules:
            Manager.tool_parsers[name] = getattr(modules[module_name], class_name)
    prepared = ParserReload(Manager, modules, _VERSION)
    Manager.tool_parsers = {
        name: prepared.replacement(cls) for name, cls in Manager.tool_parsers.items()
    }
    return prepared


def publish_reload(prepared: ParserReload) -> dict:
    """Publish only a complete generation, while holding reload_lock."""
    global _VERSION
    from vllm.tool_parsers import ToolParserManager

    if prepared.version != _VERSION:
        raise RuntimeError("Parser generation changed during reload")
    ToolParserManager.tool_parsers = prepared.manager.tool_parsers
    ToolParserManager.lazy_parsers = prepared.manager.lazy_parsers
    if prepared.modules:
        _VERSION += 1
    return {"reloaded": sorted(prepared.modules), "version": _VERSION}


def reload_now(force: bool = True) -> dict:
    """Compatibility entry; failed preparation publishes nothing."""
    with reload_lock:
        return publish_reload(prepare_reload())
