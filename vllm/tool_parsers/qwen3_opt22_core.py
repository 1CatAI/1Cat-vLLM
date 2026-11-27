# SPDX-License-Identifier: Apache-2.0
# [opt22 去重 2026-10-14] qwen 双 parser 专用判据核心（自 tool_parsers/utils.py 抽出）。
# 抽出目的：该三判据是工具调用优化（opt22d/e）的核心逻辑，发版 wheel 中随双 parser
# 一并编译为二进制（见 patches/release-gated/）；utils.py 的其余公共件被 10+ parser
# 共用，保持明文。本模块只被 qwen3coder/qwen3xml 两 parser import。
"""Qwen dual-parser request-side predicates (opt22d / opt22e core logic)."""

import json
from typing import Any, TypeAlias

from openai.types.responses.tool import Tool as ResponsesTool

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionToolsParam
from vllm.tool_parsers.qwen3_120_utils import (
    _extract_tool_info,
    find_tool_required,
    request_forbids_tool_calls,
)

Tool: TypeAlias = ChatCompletionToolsParam | ResponsesTool


def registered_tool_names(tools: list[Tool] | None) -> set:
    """已注册工具名集合。``tools`` 未提供时为空集。"""
    names = set()
    for tool in (tools or []):
        fn = getattr(tool, "function", None)
        nm = getattr(fn, "name", None) if fn is not None else None
        if not nm:
            nm = getattr(tool, "name", None)
        if nm:
            names.add(nm)
    return names


def missing_required_args(
    tools: list[Tool] | None, tc
) -> bool:
    """opt22e B：该调用的参数是否**缺少 schema 的必填字段**。

    参数无法解析（非法 JSON / 空）时也视为缺失 —— 真实调用的参数总能解析。
    ``tools`` 未提供或该工具无 ``required`` 声明时返回 False（**宽松兜底、不误伤**）。
    """
    try:
        fn = tc.function
    except Exception:
        return False
    if fn is None:
        return False
    req = find_tool_required(tools, fn.name)
    if not req:
        return False
    args = fn.arguments
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except Exception:
            return True          # 参数非法 ⇒ 必填必然缺失
    if not isinstance(args, dict):
        return True
    return any(k not in args or args[k] in (None, "") for k in req)


def opt22d_should_force_all(request, parser_tools: list[Tool] | None) -> bool:
    """opt22d A/B/C：依"请求侧信号"判断是否整轮降级为文本。

    A ``tool_choice == "none"``：客户端显式禁止调用 → 形似调用必为示例
    B 请求无工具且解析器无工具：无对象可调 → 形似调用必为示例
    （C ``request_forbids_tool_calls`` 在本函数内部复用。）
    """
    if request is None:
        return False
    if getattr(request, "tool_choice", None) == "none":
        return True
    if request_forbids_tool_calls(request):
        return True
    if not getattr(request, "tools", None) and not parser_tools:
        return True
    return False
