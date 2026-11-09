# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hot-reload endpoint for tool parsers and the chat template.

``POST /v1/internal/reload`` applies, in one request:

1. **Tool parsers** -- reloads the tool_parsers modules, refreshes the registry,
   then pushes the new classes onto the live serving objects.
2. **Chat template** -- re-reads the template source and pushes it onto the
   live serving objects.

Nothing extra happens on the request path: no mtime stat, no version compare.
Everything takes effect here, once. Editing a parser or template then costs a
single curl instead of an engine restart (minutes of model load).

Intended for local development and verification. It is safe to leave the route
registered in production -- with no source edits it is a no-op.
"""

from __future__ import annotations

import importlib
import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# Serving objects that cache a parser or a template.
_HOLDERS = (
    "openai_serving_chat",
    "openai_serving_chat_batch",
    "openai_serving_completion",
    "openai_serving_responses",
    "openai_serving_render",
    "openai_serving_tokenization",
)

# Serving classes whose methods are patched onto the live instances.
_SERVING_SOURCES = (
    ("vllm.entrypoints.openai.chat_completion.serving", "OpenAIServingChat"),
)


def _refresh_tool_parser_holders(app) -> list[str]:
    """Push freshly reloaded parser classes onto the live serving objects."""
    from vllm.parser.parser_manager import ParserManager

    done: list[str] = []
    for name in _HOLDERS:
        holder = getattr(app.state, name, None)
        if holder is None:
            continue
        tool_parser_name = getattr(holder, "_tool_parser_name", None)
        if tool_parser_name is None:
            continue  # this object does not hold a tool parser
        try:
            holder.tool_parser = ParserManager.get_tool_parser(
                tool_parser_name=tool_parser_name,
                enable_auto_tools=getattr(holder, "enable_auto_tools", False),
                model_name=getattr(holder.model_config, "model", None),
            )
            holder.parser_cls = ParserManager.get_parser(
                tool_parser_name=tool_parser_name,
                reasoning_parser_name=getattr(holder, "_reasoning_parser_name", None),
                enable_auto_tools=getattr(holder, "enable_auto_tools", False),
                model_name=getattr(holder.model_config, "model", None),
            )
            done.append(name)
        except Exception:
            logger.exception("reload: refreshing tool parser on %s failed", name)
    return done


def _refresh_serving_classes(app) -> list[str]:
    """Copy reloaded methods onto the classes the live instances still use.

    Serving objects are long-lived (``app.state``), and ``importlib.reload``
    only produces **new** classes -- the instances stay bound to the old ones,
    so editing a method has no effect. Rebuilding the instances is not an
    option (they own the tokenizer, engine client and renderer), so copy the
    new class's own attributes across. Method lookup goes through the class, so
    the change takes effect immediately, and instance state is untouched.

    Only the new class's *own* attributes are copied, never inherited base
    behaviour.
    """
    by_name: dict[str, type] = {}
    for module_name, class_name in _SERVING_SOURCES:
        try:
            module = importlib.reload(importlib.import_module(module_name))
        except Exception:
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue
        cls = getattr(module, class_name, None)
        if cls is not None:
            by_name[class_name] = cls

    done: list[str] = []
    for name in _HOLDERS:
        holder = getattr(app.state, name, None)
        if holder is None:
            continue
        old_cls = type(holder)
        new_cls = by_name.get(old_cls.__name__)
        if new_cls is None or old_cls is new_cls:
            continue
        patched = 0
        for key, value in vars(new_cls).items():
            if key.startswith("__"):
                continue
            try:
                setattr(old_cls, key, value)
                patched += 1
            except Exception:
                pass
        if patched:
            done.append(f"{name}:{old_cls.__name__}:{patched}")
    return done


def _refresh_chat_template_holders(app) -> int:
    """Re-read the chat template and push it onto the live serving objects."""
    try:
        from vllm.entrypoints.chat_utils import force_reload_chat_template
    except ImportError:
        return 0

    template = force_reload_chat_template()
    if template is None:
        return 0
    count = 0
    for name in _HOLDERS:
        holder = getattr(app.state, name, None)
        if holder is not None and hasattr(holder, "chat_template"):
            holder.chat_template = template
            count += 1
    return count


@router.post("/v1/internal/reload")
async def reload_all(request: Request):
    """Reload tool parsers, serving classes and the chat template."""
    result: dict = {}

    try:
        from vllm.tool_parsers._hotreload import reload_now

        result["tool_parser"] = reload_now(force=True)
        result["tool_parser_holders"] = _refresh_tool_parser_holders(request.app)
        result["serving_classes"] = _refresh_serving_classes(request.app)
    except Exception as exc:
        result["tool_parser_error"] = str(exc)

    try:
        result["chat_template_holders"] = _refresh_chat_template_holders(request.app)
    except Exception as exc:
        result["chat_template_error"] = str(exc)

    return JSONResponse(result)


def attach_router(app):
    app.include_router(router)
