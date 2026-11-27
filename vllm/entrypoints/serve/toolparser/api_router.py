# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Development-only parser/template reload with prepare-before-publish."""

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from vllm.tool_parsers._hotreload import (
    current_version,
    prepare_reload,
    publish_reload,
    reload_lock,
)

router = APIRouter()
_HOLDERS = (
    "openai_serving_chat",
    "openai_serving_chat_batch",
    "openai_serving_completion",
    "openai_serving_responses",
    "openai_serving_render",
    "openai_serving_tokenization",
)


def _prepare_holders(app, prepared):
    from vllm.parser.abstract_parser import _WrappedParser

    updates = []
    for name in _HOLDERS:
        holder = getattr(app.state, name, None)
        if holder is None:
            continue
        values = vars(holder)
        changes = {}
        if values.get("tool_parser") is not None:
            if not values["tool_parser"].supports_hot_reload:
                raise ValueError(
                    f"{values['tool_parser'].__name__} requires a restart; "
                    "its canonical parser class is retained"
                )
            changes["tool_parser"] = prepared.replacement(values["tool_parser"])
        parser = values.get("parser_cls")
        if isinstance(parser, type) and issubclass(parser, _WrappedParser):
            # A new wrapper preserves in-flight instances and __class__/super
            # closures. Never patch an existing serving or parser class.
            changes["parser_cls"] = type(
                parser.__name__,
                (_WrappedParser,),
                {
                    "__module__": parser.__module__,
                    "reasoning_parser_cls": parser.reasoning_parser_cls,
                    "tool_parser_cls": prepared.replacement(parser.tool_parser_cls),
                },
            )
        if changes:
            updates.append((name, values, changes))
    return updates


def _prepare_template(app):
    from transformers.utils.chat_template_utils import _compile_jinja_template

    args = getattr(app.state, "args", None)
    source = getattr(args, "chat_template", None)
    if source is None or (isinstance(source, str) and any(c in source for c in "{}\n")):
        return None, []
    path = Path(source)
    if not path.is_file():
        from vllm.transformers_utils.chat_templates.registry import CHAT_TEMPLATES_DIR

        path = CHAT_TEMPLATES_DIR / path
    content = path.read_text(encoding="utf-8")
    _compile_jinja_template(content)
    holders = []
    for name in _HOLDERS:
        holder = getattr(app.state, name, None)
        if holder is not None and "chat_template" in vars(holder):
            holders.append(vars(holder))
    return content, holders


@router.post("/v1/internal/kv_tuning")
async def kv_tuning(request: Request):
    """[本地 2026-11-27] 热应用 KV 缓存调参，避免推进 KV 机制时反复重启引擎。

    Body（全部可选，只改传入的键）::

        {"max_blocks": 160, "ttl_seconds": 600,
         "warm_soft_cap": 46, "warm_hard_cap": 77}

    走 EngineCore 的 utility RPC 通道，在引擎线程内两步之间生效。
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        return JSONResponse({"error": "body must be a JSON object"}, status_code=400)
    client = getattr(request.app.state, "engine_client", None)
    core = getattr(client, "engine_core", None)
    invoke = getattr(core, "call_utility_async", None)
    if invoke is None:
        return JSONResponse(
            {"error": "engine core utility channel unavailable"}, status_code=501
        )
    try:
        applied = await invoke("reload_kv_tuning", body)
    except Exception as error:
        return JSONResponse({"error": str(error)}, status_code=500)
    return JSONResponse(applied)


@router.post("/v1/internal/reload")
async def reload_all(request: Request):
    with reload_lock:
        try:
            prepared = prepare_reload()
            updates = _prepare_holders(request.app, prepared)
            template, holders = _prepare_template(request.app)
        except Exception as error:
            return JSONResponse(
                {
                    "error": str(error),
                    "reloaded": [],
                    "version": current_version(),
                },
                status_code=409,
            )

        # No await or external call between publishing and holder updates.
        # Every target is an ordinary instance dictionary prepared above.
        result = publish_reload(prepared)
        for _, values, changes in updates:
            values.update(changes)
        for values in holders:
            values["chat_template"] = template
        if holders:
            from vllm.entrypoints.chat_utils import _cached_load_chat_template

            _cached_load_chat_template.cache_clear()
        return JSONResponse(
            {
                "tool_parser": result,
                "tool_parser_holders": [name for name, _, _ in updates],
                "chat_template_holders": len(holders),
            }
        )


def attach_router(app):
    app.include_router(router)
