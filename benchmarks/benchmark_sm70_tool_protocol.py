# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise long, streamed tool calls and structured output through an API.

This is a deterministic protocol gate, not a substitute for BFCL task scores.
It catches malformed streamed arguments, invalid tool names, history replay
failures, and JSON-schema violations while retaining the raw SSE events needed
to localize a first divergence between two server configurations.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import time
from pathlib import Path
from typing import Any

import jsonschema
import requests

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read a UTF-8 file.",
            "parameters": {
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Grep",
            "description": "Search files for a literal pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                    "output_mode": {
                        "type": "string",
                        "enum": ["content", "files_with_matches", "count"],
                    },
                },
                "required": ["pattern", "path", "output_mode"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Glob",
            "description": "Find files matching a glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["pattern", "path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Run one shell command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Edit",
            "description": "Replace an exact string in a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                    "replace_all": {"type": "boolean"},
                },
                "required": [
                    "file_path",
                    "old_string",
                    "new_string",
                    "replace_all",
                ],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Write",
            "description": "Write complete UTF-8 content to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["file_path", "content"],
                "additionalProperties": False,
            },
        },
    },
]


STEPS: list[dict[str, Any]] = [
    {
        "name": "Read",
        "prompt": "Call Read for /tmp/project/src/app.py. Do not call another tool.",
        "arguments": {"file_path": "/tmp/project/src/app.py"},
        "result": "def run():\n    return 1\n",
    },
    {
        "name": "Grep",
        "prompt": (
            "Call Grep for literal pattern run in /tmp/project/src with "
            "output_mode=count. Do not call another tool."
        ),
        "arguments": {
            "pattern": "run",
            "path": "/tmp/project/src",
            "output_mode": "count",
        },
        "result": "/tmp/project/src/app.py:1",
    },
    {
        "name": "Glob",
        "prompt": (
            "Call Glob with pattern **/*.py under /tmp/project. "
            "Do not call another tool."
        ),
        "arguments": {"pattern": "**/*.py", "path": "/tmp/project"},
        "result": "/tmp/project/src/app.py",
    },
    {
        "name": "Bash",
        "prompt": (
            "Call Bash with exactly: python -m pytest -q. Do not call another tool."
        ),
        "arguments": {"command": "python -m pytest -q"},
        "result": "8 passed in 1.20s",
    },
    {
        "name": "Edit",
        "prompt": (
            "Call Edit on /tmp/project/src/app.py. Replace exactly the string "
            "`    return 1\\n` with `    return 2\\n`, and set replace_all=false. "
            "Preserve all indentation and newlines."
        ),
        "arguments": {
            "file_path": "/tmp/project/src/app.py",
            "old_string": "    return 1\n",
            "new_string": "    return 2\n",
            "replace_all": False,
        },
        "result": "updated /tmp/project/src/app.py",
    },
    {
        "name": "Write",
        "prompt": (
            "Call Write for /tmp/project/result.json with exactly this content: "
            '{"ok":true,"lines":["a\\nb","<parameter=x>literal"]}.'
        ),
        "arguments": {
            "file_path": "/tmp/project/result.json",
            "content": '{"ok":true,"lines":["a\\nb","<parameter=x>literal"]}',
        },
        "result": "wrote 55 bytes",
    },
]


STRUCTURED_CASES: list[dict[str, Any]] = [
    {
        "name": "nested",
        "prompt": (
            "Return the build record for project alpha, successful, with stages "
            "compile=12 and test=8."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "project": {"type": "string", "const": "alpha"},
                "success": {"type": "boolean", "const": True},
                "stages": {
                    "type": "array",
                    "prefixItems": [
                        {
                            "type": "object",
                            "properties": {
                                "name": {"const": "compile"},
                                "seconds": {"type": "integer", "const": 12},
                            },
                            "required": ["name", "seconds"],
                            "additionalProperties": False,
                        },
                        {
                            "type": "object",
                            "properties": {
                                "name": {"const": "test"},
                                "seconds": {"type": "integer", "const": 8},
                            },
                            "required": ["name", "seconds"],
                            "additionalProperties": False,
                        },
                    ],
                    "minItems": 2,
                    "maxItems": 2,
                },
            },
            "required": ["project", "success", "stages"],
            "additionalProperties": False,
        },
    },
    {
        "name": "escaped_text",
        "prompt": "Return path /tmp/a, the two-line text `a\\nb`, and count 2.",
        "schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "const": "/tmp/a"},
                "text": {"type": "string", "const": "a\nb"},
                "count": {"type": "integer", "const": 2},
            },
            "required": ["path", "text", "count"],
            "additionalProperties": False,
        },
    },
]


STREAM_PARITY_CASES: list[dict[str, str]] = [
    {
        "name": "quoted_ascii",
        "expected": 'The "implied" volatility is a "forward-looking" measure.',
    },
    {
        "name": "json_utf8",
        "expected": '{"path":"/tmp/中.txt","text":"a\\nb","emoji":"🌶️"}',
    },
]


def _schema_by_tool_name() -> dict[str, dict[str, Any]]:
    return {item["function"]["name"]: item["function"]["parameters"] for item in TOOLS}


def _stream_chat(base_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    response = requests.post(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=600,
    )
    if response.status_code != 200:
        return {
            "ok": False,
            "status_code": response.status_code,
            "body": response.text,
            "elapsed_seconds": time.perf_counter() - started,
        }

    raw_events: list[dict[str, Any]] = []
    prompt_token_ids: list[int] = []
    output_token_ids: list[int] = []
    output_token_chunks: list[list[int]] = []
    content = ""
    reasoning = ""
    finish_reason = None
    calls: dict[int, dict[str, Any]] = {}
    malformed_sse: list[str] = []
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line or not raw_line.startswith("data:"):
            continue
        data = raw_line[5:].strip()
        if data == "[DONE]":
            break
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            malformed_sse.append(data)
            continue
        raw_events.append(event)
        choice = event.get("choices", [{}])[0]
        if event.get("prompt_token_ids"):
            prompt_token_ids.extend(event["prompt_token_ids"])
        token_ids = choice.get("token_ids") or []
        if token_ids:
            output_token_chunks.append(token_ids)
            output_token_ids.extend(token_ids)
        finish_reason = choice.get("finish_reason") or finish_reason
        delta = choice.get("delta") or {}
        content += delta.get("content") or ""
        reasoning += delta.get("reasoning_content") or ""
        for call_delta in delta.get("tool_calls") or []:
            index = int(call_delta.get("index", 0))
            call = calls.setdefault(
                index,
                {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                },
            )
            call["id"] += call_delta.get("id") or ""
            function = call_delta.get("function") or {}
            call["function"]["name"] += function.get("name") or ""
            call["function"]["arguments"] += function.get("arguments") or ""

    return {
        "ok": not malformed_sse,
        "status_code": response.status_code,
        "elapsed_seconds": time.perf_counter() - started,
        "finish_reason": finish_reason,
        "content": content,
        "reasoning_content": reasoning,
        "prompt_token_ids": prompt_token_ids,
        "output_token_ids": output_token_ids,
        "output_token_chunks": output_token_chunks,
        "tool_calls": [calls[index] for index in sorted(calls)],
        "malformed_sse": malformed_sse,
        "raw_events": raw_events,
    }


def _nonstream_chat(base_url: str, payload: dict[str, Any]) -> dict[str, Any]:
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


def _responses_tools() -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for tool in TOOLS:
        function = tool["function"]
        tools.append(
            {
                "type": "function",
                "name": function["name"],
                "description": function["description"],
                "parameters": function["parameters"],
                "strict": True,
            }
        )
    return tools


def _stream_responses(base_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Collect Responses SSE without relying on a particular SDK version."""
    started = time.perf_counter()
    response = requests.post(
        f"{base_url.rstrip('/')}/v1/responses",
        json=payload,
        stream=True,
        timeout=600,
    )
    if response.status_code != 200:
        return {
            "ok": False,
            "status_code": response.status_code,
            "body": response.text,
            "elapsed_seconds": time.perf_counter() - started,
        }

    raw_events: list[dict[str, Any]] = []
    malformed_sse: list[str] = []
    response_id = ""
    response_status = ""
    output_text_chunks: list[str] = []
    protocol_errors: list[str] = []
    calls: dict[int, dict[str, Any]] = {}

    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line or not raw_line.startswith("data:"):
            continue
        data = raw_line[5:].strip()
        if data == "[DONE]":
            break
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            malformed_sse.append(data)
            continue
        raw_events.append(event)
        event_type = event.get("type") or ""
        event_response = event.get("response") or {}
        response_id = event_response.get("id") or response_id
        response_status = event_response.get("status") or response_status

        if event_type == "response.output_text.delta":
            output_text_chunks.append(event.get("delta") or "")
        elif event_type == "response.output_item.added":
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                output_index = int(event.get("output_index", 0))
                calls[output_index] = {
                    "output_index": output_index,
                    "added_item": item,
                    "argument_chunks": [],
                    "done_arguments": None,
                    "done_item": None,
                }
        elif event_type == "response.function_call_arguments.delta":
            output_index = int(event.get("output_index", 0))
            call = calls.get(output_index)
            if call is None:
                protocol_errors.append(
                    f"arguments delta before item at output {output_index}"
                )
            else:
                call["argument_chunks"].append(event.get("delta") or "")
        elif event_type == "response.function_call_arguments.done":
            output_index = int(event.get("output_index", 0))
            call = calls.get(output_index)
            if call is None:
                protocol_errors.append(
                    f"arguments done before item at output {output_index}"
                )
            else:
                call["done_arguments"] = event.get("arguments") or ""
        elif event_type == "response.output_item.done":
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                output_index = int(event.get("output_index", 0))
                call = calls.get(output_index)
                if call is None:
                    protocol_errors.append(
                        f"item done before item added at output {output_index}"
                    )
                else:
                    call["done_item"] = item
        elif event_type in {"response.failed", "error"}:
            protocol_errors.append(json.dumps(event, ensure_ascii=False))

    function_calls: list[dict[str, Any]] = []
    for output_index in sorted(calls):
        record = calls[output_index]
        added = record["added_item"]
        done = record["done_item"] or {}
        delta_arguments = "".join(record["argument_chunks"])
        done_arguments = record["done_arguments"]
        final_arguments = done.get("arguments")
        if final_arguments is None:
            final_arguments = done_arguments
        if final_arguments is None:
            final_arguments = delta_arguments or added.get("arguments") or ""

        if done_arguments is not None and done_arguments != delta_arguments:
            protocol_errors.append(
                f"arguments delta/done mismatch at output {output_index}"
            )
        if done and done.get("arguments") != final_arguments:
            protocol_errors.append(
                f"arguments done/item mismatch at output {output_index}"
            )
        if done and done.get("name") != added.get("name"):
            protocol_errors.append(f"tool name changed at output {output_index}")

        function_calls.append(
            {
                "output_index": output_index,
                "id": done.get("id") or added.get("id") or "",
                "call_id": done.get("call_id") or added.get("call_id") or "",
                "name": done.get("name") or added.get("name") or "",
                "arguments": final_arguments,
                "argument_chunks": record["argument_chunks"],
                "done_arguments": done_arguments,
                "added_item": added,
                "done_item": record["done_item"],
            }
        )

    completed = any(event.get("type") == "response.completed" for event in raw_events)
    if not completed:
        protocol_errors.append("response.completed event missing")
    return {
        "ok": not malformed_sse and not protocol_errors,
        "status_code": response.status_code,
        "elapsed_seconds": time.perf_counter() - started,
        "response_id": response_id,
        "response_status": response_status,
        "output_text": "".join(output_text_chunks),
        "function_calls": function_calls,
        "malformed_sse": malformed_sse,
        "protocol_errors": protocol_errors,
        "raw_events": raw_events,
    }


def _validate_responses_tool_step(
    result: dict[str, Any], expected: dict[str, Any]
) -> list[str]:
    if not result.get("ok"):
        return ["request failed"]
    calls = result.get("function_calls") or []
    if len(calls) != 1:
        return [f"expected one function call, got {len(calls)}"]
    call = calls[0]
    errors: list[str] = []
    if call["name"] != expected["name"]:
        errors.append(f"tool name {call['name']!r} != {expected['name']!r}")
    if not call.get("call_id"):
        errors.append("call_id missing")
    try:
        arguments = json.loads(call["arguments"])
    except json.JSONDecodeError as exc:
        return errors + [f"invalid arguments JSON: {exc}"]
    schema = _schema_by_tool_name().get(call["name"])
    if schema is None:
        errors.append(f"unknown tool {call['name']!r}")
    else:
        try:
            jsonschema.validate(arguments, schema)
        except jsonschema.ValidationError as exc:
            errors.append(f"schema violation: {exc.message}")
    if arguments != expected["arguments"]:
        errors.append(f"arguments {arguments!r} != {expected['arguments']!r}")
    return errors


def _responses_payload(
    model: str,
    seed: int,
    input_items: str | list[dict[str, Any]],
    tool_name: str,
    *,
    previous_response_id: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "input": input_items,
        "instructions": (
            "Follow the user literally, call exactly the requested listed tool, "
            "and preserve every string byte including whitespace and XML-like text."
        ),
        "tools": _responses_tools(),
        "tool_choice": {"type": "function", "name": tool_name},
        "parallel_tool_calls": False,
        "stream": True,
        "store": True,
        "temperature": 0.0,
        "seed": seed,
        "max_output_tokens": 2048,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if previous_response_id is not None:
        payload["previous_response_id"] = previous_response_id
    return payload


def _run_responses_chain(base_url: str, model: str, seed: int) -> dict[str, Any]:
    """Cover stateful and explicit-history Responses tool protocols."""
    stateful_steps = [*STEPS, STEPS[0]]
    stateful_results: list[dict[str, Any]] = []
    previous_response_id: str | None = None
    previous_call_id = ""
    previous_result = ""
    for step_index, step in enumerate(stateful_steps):
        if previous_response_id is None:
            input_items: str | list[dict[str, Any]] = step["prompt"]
        else:
            input_items = [
                {
                    "type": "function_call_output",
                    "call_id": previous_call_id,
                    "output": previous_result,
                },
                {"role": "user", "content": step["prompt"]},
            ]
        payload = _responses_payload(
            model,
            seed,
            input_items,
            step["name"],
            previous_response_id=previous_response_id,
        )
        result = _stream_responses(base_url, payload)
        errors = _validate_responses_tool_step(result, step)
        result["step"] = step_index
        result["expected_name"] = step["name"]
        result["errors"] = errors
        stateful_results.append(result)
        if errors:
            break
        previous_response_id = result["response_id"]
        previous_call_id = result["function_calls"][0]["call_id"]
        previous_result = step["result"]

    explicit_input: list[dict[str, Any]] = []
    for step_index, step in enumerate(STEPS):
        call_id = f"call_explicit_history_{step_index}"
        explicit_input.extend(
            [
                {"role": "user", "content": step["prompt"]},
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": step["name"],
                    "arguments": json.dumps(
                        step["arguments"], ensure_ascii=False, separators=(",", ":")
                    ),
                },
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": step["result"],
                },
            ]
        )
    replay_step = STEPS[0]
    explicit_input.append({"role": "user", "content": replay_step["prompt"]})
    explicit_result = _stream_responses(
        base_url,
        _responses_payload(
            model,
            seed,
            explicit_input,
            replay_step["name"],
        ),
    )
    explicit_errors = _validate_responses_tool_step(explicit_result, replay_step)
    explicit_result["errors"] = explicit_errors

    return {
        "passed": (
            len(stateful_results) == len(stateful_steps)
            and all(not item["errors"] for item in stateful_results)
            and not explicit_errors
        ),
        "stateful_steps": stateful_results,
        "explicit_history_replay": explicit_result,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _normalize_bfcl_schema(value: Any) -> Any:
    if isinstance(value, list):
        return [_normalize_bfcl_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    normalized = {key: _normalize_bfcl_schema(item) for key, item in value.items()}
    type_map = {"dict": "object", "float": "number", "tuple": "array"}
    if isinstance(normalized.get("type"), str):
        normalized["type"] = type_map.get(normalized["type"], normalized["type"])
    return normalized


def _bfcl_tools(functions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": function["name"],
                "description": function.get("description") or "",
                "parameters": _normalize_bfcl_schema(function["parameters"]),
            },
        }
        for function in functions
    ]


def _bfcl_normalized_value(value: Any) -> Any:
    if isinstance(value, str):
        return re.sub(r"[ ,./\-_*^]", "", value).lower().replace("'", '"')
    if isinstance(value, list):
        return [_bfcl_normalized_value(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _bfcl_normalized_value(item) for key, item in sorted(value.items())
        }
    return value


def _bfcl_call_matches(
    actual: dict[str, Any],
    expected: dict[str, Any],
    functions: dict[str, dict[str, Any]],
) -> tuple[bool, str]:
    expected_name, expected_arguments = next(iter(expected.items()))
    if actual.get("name") != expected_name:
        return False, f"tool name {actual.get('name')!r} != {expected_name!r}"
    arguments = actual.get("arguments")
    if not isinstance(arguments, dict):
        return False, "arguments are not a JSON object"
    function = functions.get(expected_name)
    if function is None:
        return False, f"function description missing for {expected_name!r}"
    properties = function["parameters"].get("properties") or {}
    required = function["parameters"].get("required") or []
    for name in required:
        if name not in arguments:
            return False, f"required argument {name!r} missing"
    for name, value in arguments.items():
        if name not in properties or name not in expected_arguments:
            return False, f"unexpected argument {name!r}"
        allowed = expected_arguments[name]
        if not any(
            _bfcl_normalized_value(value) == _bfcl_normalized_value(candidate)
            for candidate in allowed
        ):
            return False, f"argument {name!r} value {value!r} not in {allowed!r}"
    for name, allowed in expected_arguments.items():
        if name not in arguments and "" not in allowed:
            return False, f"expected argument {name!r} missing"
    return True, ""


def _validate_bfcl_calls(
    result: dict[str, Any],
    entry: dict[str, Any],
    ground_truth: list[dict[str, Any]] | None,
    *,
    irrelevance: bool,
) -> list[str]:
    if not result.get("ok"):
        return ["request failed"]
    canonical = _canonical_tool_calls(result.get("tool_calls") or [])
    if irrelevance:
        return [] if not canonical else [f"irrelevant request called {canonical!r}"]
    assert ground_truth is not None
    if len(canonical) != len(ground_truth):
        return [
            f"wrong number of calls: got {len(canonical)}, expected {len(ground_truth)}"
        ]
    functions = {function["name"]: function for function in entry["function"]}
    unmatched = set(range(len(canonical)))
    errors: list[str] = []
    for expected in ground_truth:
        candidate_errors: list[str] = []
        for index in list(unmatched):
            matched, error = _bfcl_call_matches(canonical[index], expected, functions)
            if matched:
                unmatched.remove(index)
                break
            candidate_errors.append(error)
        else:
            errors.append(f"no call matched {expected!r}: {candidate_errors!r}")
    return errors


def _run_bfcl_sample(
    base_url: str,
    model: str,
    seed: int,
    data_dir: Path,
    per_category: int,
) -> dict[str, Any]:
    categories = {
        "simple_python": False,
        "parallel": False,
        "multiple": False,
        "irrelevance": True,
    }
    category_results: dict[str, Any] = {}
    for category, irrelevance in categories.items():
        filename = f"BFCL_v4_{category}.json"
        entries = _read_jsonl(data_dir / filename)[:per_category]
        ground_truth_by_id: dict[str, list[dict[str, Any]]] = {}
        if not irrelevance:
            ground_truth_by_id = {
                item["id"]: item["ground_truth"]
                for item in _read_jsonl(data_dir / "possible_answer" / filename)
            }
        results: list[dict[str, Any]] = []
        for entry in entries:
            payload = {
                "model": model,
                "messages": entry["question"][0],
                "tools": _bfcl_tools(entry["function"]),
                "tool_choice": "auto",
                "parallel_tool_calls": True,
                "stream": True,
                "return_token_ids": True,
                "temperature": 0.0,
                "seed": seed,
                "max_tokens": 1024,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            result = _stream_chat(base_url, payload)
            errors = _validate_bfcl_calls(
                result,
                entry,
                ground_truth_by_id.get(entry["id"]),
                irrelevance=irrelevance,
            )
            result["id"] = entry["id"]
            result["errors"] = errors
            results.append(result)
        category_results[category] = {
            "passed": sum(not item["errors"] for item in results),
            "total": len(results),
            "cases": results,
        }
    passed = sum(item["passed"] for item in category_results.values())
    total = sum(item["total"] for item in category_results.values())
    return {
        "passed": passed == total,
        "score": passed,
        "total": total,
        "per_category": category_results,
        "note": "Fixed BFCL v4 subset; not a full leaderboard submission.",
    }


def _adapt_openai_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    adapted = copy.deepcopy(schema)
    if "type" not in adapted:
        adapted["type"] = "object"

    def visit(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if not isinstance(value, dict):
            return
        properties = value.get("properties")
        if isinstance(properties, dict):
            if value.get("additionalProperties", True):
                value["additionalProperties"] = False
            value["required"] = list(properties)
        for item in value.values():
            visit(item)

    visit(adapted)
    return adapted


def _run_json_schema_bench_sample(
    base_url: str,
    model: str,
    seed: int,
    schema_dir: Path,
    limit: int,
    max_schema_bytes: int,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    candidates = sorted(
        (
            path
            for path in schema_dir.glob("*.json")
            if path.stat().st_size <= max_schema_bytes
        ),
        key=lambda path: (path.stat().st_size, path.name),
    )
    if len(candidates) > limit:
        if limit == 1:
            candidates = [candidates[len(candidates) // 2]]
        else:
            candidates = [
                candidates[round(index * (len(candidates) - 1) / (limit - 1))]
                for index in range(limit)
            ]
    for path in candidates:
        schema = _adapt_openai_json_schema(json.loads(path.read_text()))
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You need to generate a JSON object that matches the "
                        "schema below."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(schema, ensure_ascii=False),
                },
            ],
            "stream": True,
            "return_token_ids": True,
            "temperature": 0.0,
            "seed": seed,
            "max_tokens": 2048,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": path.stem,
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        result = _stream_chat(base_url, payload)
        errors: list[str] = []
        if not result.get("ok"):
            errors.append("request failed")
        else:
            try:
                value = json.loads(result["content"])
                jsonschema.validate(value, schema)
            except (json.JSONDecodeError, jsonschema.ValidationError) as exc:
                errors.append(str(exc))
        result["schema"] = path.name
        result["errors"] = errors
        results.append(result)
    passed = sum(not item["errors"] for item in results)
    return {
        "passed": passed == len(results),
        "score": passed,
        "total": len(results),
        "cases": results,
        "note": (
            "Deterministic zero-shot WashingtonPost JSONSchemaBench subset; "
            f"schemas are size-stratified up to {max_schema_bytes} bytes; not a "
            "full benchmark score."
        ),
    }


def _canonical_tool_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    canonical: list[dict[str, Any]] = []
    for call in calls:
        function = call.get("function") or {}
        arguments = function.get("arguments") or ""
        try:
            parsed_arguments: Any = json.loads(arguments)
        except (TypeError, json.JSONDecodeError):
            parsed_arguments = arguments
        canonical.append(
            {
                "name": function.get("name") or "",
                "arguments": parsed_arguments,
            }
        )
    return canonical


def _compare_stream_pair(
    nonstream: dict[str, Any],
    stream: dict[str, Any],
    *,
    compare_content: bool,
    compare_tools: bool,
) -> list[str]:
    errors: list[str] = []
    if not nonstream.get("ok") or not stream.get("ok"):
        return ["stream or non-stream request failed"]
    if compare_content and nonstream["content"] != stream["content"]:
        errors.append("stream/non-stream content mismatch")
    if nonstream["reasoning_content"] != stream["reasoning_content"]:
        errors.append("stream/non-stream reasoning mismatch")
    if compare_tools and _canonical_tool_calls(
        nonstream["tool_calls"]
    ) != _canonical_tool_calls(stream["tool_calls"]):
        errors.append("stream/non-stream tool-call mismatch")
    nonstream_ids = nonstream.get("output_token_ids") or []
    stream_ids = stream.get("output_token_ids") or []
    if not nonstream_ids or not stream_ids:
        errors.append("output token IDs unavailable for parity check")
    elif nonstream_ids != stream_ids:
        errors.append("stream/non-stream output token IDs differ")
    return errors


def _run_stream_parity(base_url: str, model: str, seed: int) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for case in STREAM_PARITY_CASES:
        expected = case["expected"]
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Reply with exactly the text after TEXT, without quotes, "
                        f"commentary, or a code fence.\nTEXT: {expected}"
                    ),
                }
            ],
            "return_token_ids": True,
            "temperature": 0.0,
            "seed": seed,
            "max_tokens": 256,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        nonstream = _nonstream_chat(base_url, {**payload, "stream": False})
        stream = _stream_chat(base_url, {**payload, "stream": True})
        errors = _compare_stream_pair(
            nonstream,
            stream,
            compare_content=True,
            compare_tools=False,
        )
        if nonstream.get("ok") and nonstream["content"] != expected:
            errors.append("non-stream exact-copy output mismatch")
        if stream.get("ok") and stream["content"] != expected:
            errors.append("stream exact-copy output mismatch")
        results.append(
            {
                "case": case["name"],
                "expected": expected,
                "nonstream": nonstream,
                "stream": stream,
                "errors": errors,
            }
        )

    step = STEPS[0]
    tool_payload = {
        "model": model,
        "messages": [{"role": "user", "content": step["prompt"]}],
        "tools": TOOLS,
        "tool_choice": "auto",
        "return_token_ids": True,
        "temperature": 0.0,
        "seed": seed,
        "max_tokens": 512,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    nonstream = _nonstream_chat(base_url, {**tool_payload, "stream": False})
    stream = _stream_chat(base_url, {**tool_payload, "stream": True})
    errors = _compare_stream_pair(
        nonstream,
        stream,
        compare_content=True,
        compare_tools=True,
    )
    if nonstream.get("ok"):
        errors.extend(_validate_tool_step(nonstream, step))
    if stream.get("ok"):
        errors.extend(_validate_tool_step(stream, step))
    results.append(
        {
            "case": "tool_read",
            "nonstream": nonstream,
            "stream": stream,
            "errors": errors,
        }
    )
    return {"passed": all(not item["errors"] for item in results), "cases": results}


def _validate_tool_step(result: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    calls = result.get("tool_calls") or []
    if len(calls) != 1:
        return [f"expected one tool call, got {len(calls)}"]
    call = calls[0]
    name = call["function"]["name"]
    if name != expected["name"]:
        errors.append(f"tool name {name!r} != {expected['name']!r}")
    try:
        arguments = json.loads(call["function"]["arguments"])
    except json.JSONDecodeError as exc:
        return errors + [f"invalid arguments JSON: {exc}"]
    schema = _schema_by_tool_name().get(name)
    if schema is None:
        errors.append(f"unknown tool {name!r}")
    else:
        try:
            jsonschema.validate(arguments, schema)
        except jsonschema.ValidationError as exc:
            errors.append(f"schema violation: {exc.message}")
    if arguments != expected["arguments"]:
        errors.append(f"arguments {arguments!r} != {expected['arguments']!r}")
    return errors


def _run_tool_chain(base_url: str, model: str, seed: int) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a tool protocol validation agent. Follow each user request "
                "literally, call only a listed tool, and preserve string whitespace."
            ),
        }
    ]
    results: list[dict[str, Any]] = []
    for step_index, step in enumerate(STEPS):
        messages.append({"role": "user", "content": step["prompt"]})
        payload = {
            "model": model,
            "messages": messages,
            "tools": TOOLS,
            "tool_choice": "auto",
            "stream": True,
            "return_token_ids": True,
            "temperature": 0.0,
            "seed": seed,
            "max_tokens": 2048,
        }
        result = _stream_chat(base_url, payload)
        errors = (
            _validate_tool_step(result, step)
            if result.get("ok")
            else ["request failed"]
        )
        result["step"] = step_index
        result["expected_name"] = step["name"]
        result["errors"] = errors
        results.append(result)
        if errors:
            break

        tool_call = result["tool_calls"][0]
        history_tool_call = copy.deepcopy(tool_call)
        history_tool_call["id"] = f"call_protocol_step_{step_index}"
        messages.append(
            {
                "role": "assistant",
                "content": result.get("content") or None,
                "tool_calls": [history_tool_call],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": history_tool_call["id"],
                "content": step["result"],
            }
        )
    return {
        "passed": len(results) == len(STEPS)
        and all(not item["errors"] for item in results),
        "steps": results,
    }


def _run_structured(base_url: str, model: str, seed: int) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for case in STRUCTURED_CASES:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": case["prompt"]}],
            "stream": True,
            "return_token_ids": True,
            "temperature": 0.0,
            "seed": seed,
            "max_tokens": 512,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": case["name"],
                    "strict": True,
                    "schema": case["schema"],
                },
            },
        }
        result = _stream_chat(base_url, payload)
        errors: list[str] = []
        if not result.get("ok"):
            errors.append("request failed")
        else:
            try:
                value = json.loads(result["content"])
                jsonschema.validate(value, case["schema"])
            except (json.JSONDecodeError, jsonschema.ValidationError) as exc:
                errors.append(str(exc))
        result["case"] = case["name"]
        result["errors"] = errors
        results.append(result)
    return {"passed": all(not item["errors"] for item in results), "cases": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="quality-audit-dflash2")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--bfcl-data-dir", type=Path)
    parser.add_argument("--bfcl-per-category", type=int, default=0)
    parser.add_argument("--json-schema-dir", type=Path)
    parser.add_argument("--json-schema-limit", type=int, default=0)
    parser.add_argument("--json-schema-max-bytes", type=int, default=16384)
    parser.add_argument(
        "--only-datasets",
        action="store_true",
        help="Skip synthetic protocol and Responses gates for matched dataset arms.",
    )
    args = parser.parse_args()

    artifact: dict[str, Any] = {
        "base_url": args.base_url,
        "model": args.model,
        "seed": args.seed,
        "only_datasets": args.only_datasets,
    }
    if not args.only_datasets:
        artifact.update(
            {
                "tool_chain": _run_tool_chain(args.base_url, args.model, args.seed),
                "structured": _run_structured(args.base_url, args.model, args.seed),
                "stream_parity": _run_stream_parity(
                    args.base_url, args.model, args.seed
                ),
                "responses": _run_responses_chain(args.base_url, args.model, args.seed),
            }
        )
    if args.bfcl_data_dir is not None and args.bfcl_per_category > 0:
        artifact["bfcl_fixed_sample"] = _run_bfcl_sample(
            args.base_url,
            args.model,
            args.seed,
            args.bfcl_data_dir,
            args.bfcl_per_category,
        )
    if args.json_schema_dir is not None and args.json_schema_limit > 0:
        artifact["json_schema_bench_fixed_sample"] = _run_json_schema_bench_sample(
            args.base_url,
            args.model,
            args.seed,
            args.json_schema_dir,
            args.json_schema_limit,
            args.json_schema_max_bytes,
        )
    result_keys = {
        "tool_chain",
        "structured",
        "stream_parity",
        "responses",
        "bfcl_fixed_sample",
        "json_schema_bench_fixed_sample",
    }
    required_results = [value for key, value in artifact.items() if key in result_keys]
    artifact["passed"] = bool(
        required_results and all(value["passed"] for value in required_results)
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": artifact["passed"], "out": str(args.out)}))
    if not artifact["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
