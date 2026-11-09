# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hot-reload tool parser implementations without restarting the engine.

Iterating on a tool parser normally means restarting the engine, because
loading the model takes minutes. This module makes the edit -> test loop fast
for deployments that serve from a mutable source tree.

Design:

- The HTTP endpoint ``POST /v1/internal/reload`` (see
  ``entrypoints/serve/toolparser/api_router.py``) calls :func:`reload_now`,
  which reloads the already-imported ``vllm.tool_parsers.*`` modules, refreshes
  the ``ToolParserManager`` registry, and bumps a version counter.
- The hot path costs one integer comparison: :func:`current_version` against a
  cached value. Modules are never re-stat-ed and nothing is reloaded per
  request.

Limitations: only modules that are already imported get reloaded, and requests
already in flight finish on the old classes.
"""

from __future__ import annotations

import importlib
import logging
import sys
import threading

logger = logging.getLogger(__name__)

# Reloaded before the parsers themselves: shared helpers that parsers import at
# module scope, so reloading them first means the parsers below pick up the new
# objects.
_PREREQUISITE_MODULES = (
    "vllm.tool_parsers.utils",
    "vllm.tool_parsers.qwen_xml_tool_grammar",
)

_PARSER_PACKAGE = "vllm.tool_parsers."

_VERSION = 0
_LOCK = threading.Lock()


def current_version() -> int:
    """Current reload version. Compare against a cached value on the hot path."""
    return _VERSION


def _loaded_parser_modules() -> list[str]:
    """Every already-imported ``vllm.tool_parsers.*`` module, prerequisites first."""
    names = [
        name
        for name in sys.modules
        if name.startswith(_PARSER_PACKAGE)
        and name != _PARSER_PACKAGE
        and not name.endswith("_hotreload")
    ]
    prereq = [n for n in _PREREQUISITE_MODULES if n in names]
    rest = sorted(n for n in names if n not in prereq)
    return prereq + rest


def _refresh_registry() -> None:
    """Point the lazy registry at the freshly reloaded class objects."""
    try:
        from vllm.tool_parsers import ToolParserManager

        for name, entry in list(ToolParserManager.lazy_parsers.items()):
            module_path, class_name = entry
            module = sys.modules.get(module_path)
            cls = getattr(module, class_name, None) if module is not None else None
            if cls is not None:
                ToolParserManager.tool_parsers[name] = cls
    except Exception:
        logger.exception("tool parser hot-reload registry refresh failed")


def reload_now(force: bool = True) -> dict:
    """Reload the tool parser modules and refresh the registry.

    Args:
        force: reload unconditionally. Kept as a parameter so callers can later
            add an mtime-based skip without changing the signature.

    Returns:
        ``{"reloaded": [...], "version": <new version>}``
    """
    global _VERSION
    reloaded: list[str] = []
    with _LOCK:
        for name in _loaded_parser_modules():
            mod = sys.modules.get(name)
            if mod is None:
                continue
            try:
                importlib.reload(mod)
                reloaded.append(name)
            except Exception:
                # Keep the old module rather than take the server down over a
                # typo in the file being edited.
                logger.exception("tool parser hot-reload failed for %s", name)
        if reloaded:
            _refresh_registry()
            _VERSION += 1
            logger.warning(
                "tool parser hot-reloaded (v%d): %s", _VERSION, reloaded
            )
    return {"reloaded": reloaded, "version": _VERSION}
