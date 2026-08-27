# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run fixed ToolACE and NexusRaven function-calling samples via an API.

These are deterministic dataset subsets for matched route comparisons, not
full official leaderboard submissions. ToolACE function names are sanitized
to the OpenAI-compatible namespace while retaining their original names in
the descriptions and ground truth.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import regex as re
import requests

TOOLACE_MARKER = "Here is a list of functions in JSON format that you can invoke:"
SAFE_TOOL_NAME = re.compile(r"[^A-Za-z0-9_]")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _post_chat(base_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    response = requests.post(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        json=payload,
        timeout=600,
    )
    if response.status_code != 200:
        return {
            "ok": False,
            "status_code": response.status_code,
            "body": response.text,
            "elapsed_seconds": time.perf_counter() - started,
        }
    try:
        body = response.json()
    except requests.JSONDecodeError as exc:
        return {
            "ok": False,
            "status_code": response.status_code,
            "body": response.text,
            "error": str(exc),
            "elapsed_seconds": time.perf_counter() - started,
        }
    choice = body.get("choices", [{}])[0]
    message = choice.get("message") or {}
    return {
        "ok": True,
        "status_code": response.status_code,
        "elapsed_seconds": time.perf_counter() - started,
        "finish_reason": choice.get("finish_reason"),
        "content": message.get("content") or "",
        "reasoning_content": message.get("reasoning_content") or "",
        "prompt_token_ids": body.get("prompt_token_ids") or [],
        "output_token_ids": choice.get("token_ids") or [],
        "tool_calls": message.get("tool_calls") or [],
        "raw_response": body,
    }


def _canonical_calls(result: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    calls: list[dict[str, Any]] = []
    errors: list[str] = []
    for call in result.get("tool_calls") or []:
        function = call.get("function") or {}
        try:
            arguments = json.loads(function.get("arguments") or "")
        except (TypeError, json.JSONDecodeError) as exc:
            errors.append(f"invalid arguments JSON: {exc}")
            continue
        calls.append({"name": function.get("name") or "", "arguments": arguments})
    return calls, errors


def _normalize_schema(value: Any) -> Any:
    if isinstance(value, list):
        return [_normalize_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    normalized = {key: _normalize_schema(item) for key, item in value.items()}
    type_map = {
        "dict": "object",
        "float": "number",
        "tuple": "array",
        "any": None,
        "None": None,
    }
    schema_type = normalized.get("type")
    if isinstance(schema_type, str) and schema_type in type_map:
        mapped = type_map[schema_type]
        if mapped is None:
            normalized.pop("type")
        else:
            normalized["type"] = mapped
    return normalized


def _stratified(items: list[Any], count: int, *, key) -> list[Any]:
    items = sorted(items, key=key)
    if len(items) <= count:
        return items
    if count == 1:
        return [items[len(items) // 2]]
    return [
        items[round(index * (len(items) - 1) / (count - 1))] for index in range(count)
    ]


def _split_top_level(text: str) -> list[str]:
    pieces: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    escaped = False
    for index, character in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
        elif character in {'"', "'"}:
            quote = character
        elif character in "([{":
            depth += 1
        elif character in ")]}" and depth:
            depth -= 1
        elif character == "," and depth == 0:
            pieces.append(text[start:index].strip())
            start = index + 1
    pieces.append(text[start:].strip())
    return [piece for piece in pieces if piece]


def _toolace_functions(record: dict[str, Any]) -> list[dict[str, Any]] | None:
    system = record.get("system") or ""
    marker = system.find(TOOLACE_MARKER)
    if marker == -1:
        return None
    start = system.find("[", marker + len(TOOLACE_MARKER))
    if start == -1:
        return None
    try:
        functions = json.JSONDecoder().raw_decode(system[start:])[0]
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(functions, list) or not all(
        isinstance(function, dict)
        and isinstance(function.get("name"), str)
        and isinstance(function.get("parameters"), dict)
        for function in functions
    ):
        return None
    return functions


def _parse_toolace_calls(
    answer: str, function_names: list[str]
) -> list[dict[str, Any]] | None:
    answer = answer.strip()
    if not answer.startswith("[") or not answer.endswith("]"):
        return None
    body = answer[1:-1].strip()
    if not body:
        return []
    calls: list[dict[str, Any]] = []
    for piece in _split_top_level(body):
        names = [
            name
            for name in function_names
            if piece.startswith(f"{name}(") and piece.endswith(")")
        ]
        if not names:
            return None
        name = max(names, key=len)
        argument_text = piece[len(name) + 1 : -1]
        try:
            call = ast.parse(f"f({argument_text})", mode="eval").body
            if not isinstance(call, ast.Call) or call.args:
                return None
            arguments = {
                keyword.arg: ast.literal_eval(keyword.value)
                for keyword in call.keywords
                if keyword.arg is not None
            }
        except (SyntaxError, ValueError):
            return None
        calls.append({"name": name, "arguments": arguments})
    return calls


def _clear_no_tool_answer(answer: str, functions: list[dict[str, Any]]) -> bool:
    if not functions:
        return True
    lowered = answer.lower()
    signatures = (
        "no available function",
        "no available functions",
        "lacks the required",
        "lacks the parameter",
        "please provide",
        "provide further",
        "additional information",
        "not enough information",
    )
    return any(signature in lowered for signature in signatures)


def _safe_tool_names(
    functions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    tools: list[dict[str, Any]] = []
    name_map: dict[str, str] = {}
    used: set[str] = set()
    for index, function in enumerate(functions):
        slug = SAFE_TOOL_NAME.sub("_", function["name"]).strip("_") or "function"
        safe_name = f"tool_{index}_{slug}"[:64]
        while safe_name in used:
            safe_name = f"tool_{index}_{slug}_{len(used)}"[:64]
        used.add(safe_name)
        name_map[function["name"]] = safe_name
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": safe_name,
                    "description": (
                        f"Original tool name: {function['name']}. "
                        f"{function.get('description') or ''}"
                    ),
                    "parameters": _normalize_schema(function["parameters"]),
                },
            }
        )
    return tools, name_map


def _run_toolace(
    base_url: str,
    model: str,
    seed: int,
    path: Path,
    per_bucket: int,
) -> dict[str, Any]:
    records = json.loads(path.read_text())
    buckets: dict[str, list[dict[str, Any]]] = {
        "single": [],
        "parallel": [],
        "no_tool": [],
    }
    for index, record in enumerate(records):
        conversations = record.get("conversations") or []
        if len(conversations) != 2 or conversations[0].get("from") != "user":
            continue
        functions = _toolace_functions(record)
        if functions is None:
            continue
        answer = conversations[1].get("value") or ""
        calls = _parse_toolace_calls(answer, [item["name"] for item in functions])
        if calls is not None and calls:
            bucket = "parallel" if len(calls) > 1 else "single"
        elif _clear_no_tool_answer(answer, functions):
            bucket = "no_tool"
            calls = []
        else:
            continue
        buckets[bucket].append(
            {
                "index": index,
                "prompt": conversations[0]["value"],
                "functions": functions,
                "expected": calls,
                "system_length": len(record.get("system") or ""),
            }
        )

    results_by_bucket: dict[str, Any] = {}
    for bucket, candidates in buckets.items():
        selected = _stratified(
            candidates,
            per_bucket,
            key=lambda item: (item["system_length"], item["index"]),
        )
        results: list[dict[str, Any]] = []
        for case in selected:
            tools, name_map = _safe_tool_names(case["functions"])
            expected = [
                {"name": name_map[call["name"]], "arguments": call["arguments"]}
                for call in case["expected"]
            ]
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": case["prompt"]}],
                "temperature": 0.0,
                "seed": seed,
                "max_tokens": 1024,
                "return_token_ids": True,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            if tools:
                payload.update(
                    {
                        "tools": tools,
                        "tool_choice": "auto",
                        "parallel_tool_calls": True,
                    }
                )
            result = _post_chat(base_url, payload)
            actual, errors = _canonical_calls(result)
            if result.get("ok") and actual != expected:
                errors.append(f"calls {actual!r} != {expected!r}")
            elif not result.get("ok"):
                errors.append("request failed")
            result.update(
                {
                    "index": case["index"],
                    "expected": expected,
                    "actual": actual,
                    "errors": errors,
                }
            )
            results.append(result)
        results_by_bucket[bucket] = {
            "score": sum(not result["errors"] for result in results),
            "total": len(results),
            "cases": results,
        }
    score = sum(item["score"] for item in results_by_bucket.values())
    total = sum(item["total"] for item in results_by_bucket.values())
    return {
        "score": score,
        "total": total,
        "protocol_passed": all(
            case.get("ok")
            for bucket in results_by_bucket.values()
            for case in bucket["cases"]
        ),
        "per_bucket": results_by_bucket,
        "note": "Fixed ToolACE subset with OpenAI-safe function-name mapping.",
    }


def _nexus_tool(function: dict[str, Any]) -> dict[str, Any]:
    json_types = {
        "str": "string",
        "int": "integer",
        "list": "array",
        "dict": "object",
    }
    properties: dict[str, Any] = {}
    required: list[str] = []
    for argument in function["args_dicts"]:
        properties[argument["name"]] = {
            "description": argument.get("description") or ""
        }
        if json_type := json_types.get(str(argument.get("type"))):
            properties[argument["name"]]["type"] = json_type
        if argument.get("required"):
            required.append(argument["name"])
    return {
        "type": "function",
        "function": {
            "name": function["name"],
            "description": function.get("description") or "",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def _run_nexus(
    base_url: str,
    model: str,
    seed: int,
    queries_path: Path,
    api_path: Path,
    per_domain: int,
) -> dict[str, Any]:
    queries = pd.read_parquet(queries_path)
    functions = pd.read_parquet(api_path)
    function_map = {
        (row["dataset"], row["name"]): row for row in functions.to_dict("records")
    }
    domain_results: dict[str, Any] = {}
    for domain in sorted(queries["dataset"].unique()):
        records = queries[queries["dataset"] == domain].to_dict("records")
        selected = _stratified(
            list(enumerate(records)),
            per_domain,
            key=lambda item: (len(item[1]["prompt"]), item[0]),
        )
        results: list[dict[str, Any]] = []
        for dataset_index, record in selected:
            tools = [
                _nexus_tool(function_map[(domain, name)])
                for name in record["context_functions"]
            ]
            expected = [
                {
                    "name": record["python_function_name"],
                    "arguments": json.loads(record["python_args_dict"]),
                }
            ]
            result = _post_chat(
                base_url,
                {
                    "model": model,
                    "messages": [{"role": "user", "content": record["prompt"]}],
                    "tools": tools,
                    "tool_choice": "auto",
                    "parallel_tool_calls": False,
                    "temperature": 0.0,
                    "seed": seed,
                    "max_tokens": 1024,
                    "return_token_ids": True,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            actual, errors = _canonical_calls(result)
            if result.get("ok") and actual != expected:
                errors.append(f"calls {actual!r} != {expected!r}")
            elif not result.get("ok"):
                errors.append("request failed")
            result.update(
                {
                    "dataset_index": dataset_index,
                    "prompt": record["prompt"],
                    "expected": expected,
                    "actual": actual,
                    "errors": errors,
                }
            )
            results.append(result)
        domain_results[domain] = {
            "score": sum(not result["errors"] for result in results),
            "total": len(results),
            "cases": results,
        }
    score = sum(item["score"] for item in domain_results.values())
    total = sum(item["total"] for item in domain_results.values())
    return {
        "score": score,
        "total": total,
        "protocol_passed": all(
            case.get("ok")
            for domain in domain_results.values()
            for case in domain["cases"]
        ),
        "per_domain": domain_results,
        "note": "Fixed NexusRaven API evaluation subset; exact call/argument match.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="quality-audit-dflash2")
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--toolace-data", type=Path, required=True)
    parser.add_argument("--toolace-per-bucket", type=int, default=4)
    parser.add_argument("--nexus-queries", type=Path, required=True)
    parser.add_argument("--nexus-api-list", type=Path, required=True)
    parser.add_argument("--nexus-per-domain", type=int, default=4)
    args = parser.parse_args()

    artifact = {
        "base_url": args.base_url,
        "model": args.model,
        "seed": args.seed,
        "sources": {
            "toolace": {
                "path": str(args.toolace_data),
                "sha256": _sha256(args.toolace_data),
            },
            "nexus_queries": {
                "path": str(args.nexus_queries),
                "sha256": _sha256(args.nexus_queries),
            },
            "nexus_api_list": {
                "path": str(args.nexus_api_list),
                "sha256": _sha256(args.nexus_api_list),
            },
        },
        "toolace": _run_toolace(
            args.base_url,
            args.model,
            args.seed,
            args.toolace_data,
            args.toolace_per_bucket,
        ),
        "nexus_raven": _run_nexus(
            args.base_url,
            args.model,
            args.seed,
            args.nexus_queries,
            args.nexus_api_list,
            args.nexus_per_domain,
        ),
    }
    artifact["protocol_passed"] = bool(
        artifact["toolace"]["protocol_passed"]
        and artifact["nexus_raven"]["protocol_passed"]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "protocol_passed": artifact["protocol_passed"],
                "toolace": [artifact["toolace"]["score"], artifact["toolace"]["total"]],
                "nexus_raven": [
                    artifact["nexus_raven"]["score"],
                    artifact["nexus_raven"]["total"],
                ],
                "out": str(args.out),
            }
        )
    )
    if not artifact["protocol_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
