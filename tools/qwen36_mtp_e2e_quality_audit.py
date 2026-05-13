#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""End-to-end Qwen3.6 MTP quality and API audit.

This harness intentionally exercises the OpenAI-compatible serving path.  It
catches issues that direct LLM.generate() tests miss: chat template kwargs,
tool-call parser behavior, streaming, speculative decoding metrics, and output
anomalies under official sampling.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


DEFAULT_COMPILATION_CONFIG = {
    "cudagraph_mode": "full_and_piecewise",
    "cudagraph_capture_sizes": [1, 2, 4, 8, 9, 18],
}


@dataclass(frozen=True)
class Scenario:
    name: str
    speculative_config: dict[str, Any] | None


@dataclass(frozen=True)
class Case:
    case_id: str
    category: str
    messages: list[dict[str, Any]]
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] = "none"
    chat_template_kwargs: dict[str, Any] | None = None
    stream: bool = False
    expected_contains: list[str] | None = None
    banned_contains: list[str] | None = None
    expected_tool: str | None = None
    expected_tool_args: dict[str, Any] | None = None
    compare_with_baseline: bool = False


def _default_python_executable() -> str:
    preferred = Path("/home/ymzx/miniconda3/envs/1cat-vllm-0.0.3/bin/python")
    if preferred.exists():
        return str(preferred)
    return sys.executable


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _default_output_dir() -> Path:
    return Path("bench_results/qwen36_27b_mtp_e2e_quality") / time.strftime(
        "%Y%m%d-%H%M%S"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/ymzx/models/Qwen3.6-27B-AWQ")
    parser.add_argument("--python-executable", default=_default_python_executable())
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, default=8490)
    parser.add_argument("--gpu-ids", default="1,2,3,4")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--served-model-name", default="qwen36-27b-awq-audit")
    parser.add_argument("--attention-backend", default="FLASH_ATTN_V100")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--quantization", default="awq")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--kv-cache-auto-trim-ratio", type=float, default=0.0)
    parser.add_argument("--swap-space", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=262144)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--mtp-num-speculative-tokens", type=int, default=6)
    parser.add_argument("--target-lengths", default="512,4096,32768,65536,131072,245760")
    parser.add_argument("--long-output-tokens", type=int, default=96)
    parser.add_argument("--quality-output-tokens", type=int, default=512)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--request-timeout", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-mtp", action="store_true")
    parser.add_argument("--skip-long-context", action="store_true")
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--compilation-config-json",
        default=json.dumps(DEFAULT_COMPILATION_CONFIG),
    )
    return parser.parse_args()


def _scenarios(args: argparse.Namespace) -> list[Scenario]:
    scenarios: list[Scenario] = []
    if not args.skip_baseline:
        scenarios.append(Scenario("baseline", None))
    if not args.skip_mtp:
        scenarios.append(
            Scenario(
                f"mtp_n{args.mtp_num_speculative_tokens}",
                {
                    "method": "mtp",
                    "num_speculative_tokens": args.mtp_num_speculative_tokens,
                },
            )
        )
    return scenarios


def _base_env(args: argparse.Namespace, scenario_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    repo_root = str(Path(__file__).resolve().parents[1])
    flash_root = str(Path(repo_root) / "flash-attention-v100")
    env.setdefault("CUDA_HOME", "/usr/local/cuda-12.8")
    env["PATH"] = f"/usr/local/cuda-12.8/bin:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = (
        f"/usr/local/cuda-12.8/lib64:{env.get('LD_LIBRARY_PATH', '')}"
    )
    env["PYTHONPATH"] = (
        f"{repo_root}:{flash_root}:{env.get('PYTHONPATH', '')}"
    ).rstrip(":")
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["VLLM_USE_V1"] = "1"
    env["VLLM_ATTENTION_BACKEND"] = args.attention_backend
    env["VLLM_SM70_ENABLE_LM_HEAD_FASTPATH"] = "1"
    env["VLLM_SPEC_DEBUG_CORRUPTION"] = "1"
    env["VLLM_CACHE_ROOT"] = str(scenario_dir / "cache")
    return env


def _server_command(args: argparse.Namespace, scenario: Scenario, port: int) -> list[str]:
    command = [
        args.python_executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--served-model-name",
        f"{args.served_model_name}-{scenario.name}",
        "--trust-remote-code",
        "--quantization",
        args.quantization,
        "--dtype",
        args.dtype,
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--kv-cache-auto-trim-ratio",
        str(args.kv_cache_auto_trim_ratio),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--skip-mm-profiling",
        "--mm-processor-cache-gb",
        "0",
        "--limit-mm-per-prompt",
        '{"image":0,"video":0}',
        "--generation-config",
        "vllm",
        "--attention-backend",
        args.attention_backend,
        "--swap-space",
        str(args.swap_space),
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_coder",
        "--host",
        args.host,
        "--port",
        str(port),
        "--compilation-config",
        args.compilation_config_json,
    ]
    if args.disable_custom_all_reduce:
        command.append("--disable-custom-all-reduce")
    if scenario.speculative_config is not None:
        command.extend(["--speculative-config", json.dumps(scenario.speculative_config)])
    return command


def _request_json(
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    if payload is None:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _wait_for_server(host: str, port: int, timeout_s: int, proc: subprocess.Popen[Any]) -> None:
    deadline = time.time() + timeout_s
    url = f"http://{host}:{port}/v1/models"
    last_error = ""
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with return code {proc.returncode}")
        try:
            _request_json(url, timeout=5)
            return
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(2)
    raise TimeoutError(f"timed out waiting for server at {url}; last_error={last_error}")


def _terminate_process(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=45)
        return
    except subprocess.TimeoutExpired:
        pass
    os.killpg(proc.pid, signal.SIGKILL)
    proc.wait(timeout=45)


def _fetch_metrics(base_url: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(f"{base_url}/metrics", timeout=30) as response:
            text = response.read().decode("utf-8", errors="replace")
    except Exception:
        return {"raw_available": False}

    result: dict[str, Any] = {
        "raw_available": True,
        "spec_num_drafts": 0,
        "spec_num_draft_tokens": 0,
        "spec_num_accepted_tokens": 0,
        "spec_accepted_per_pos": {},
        "corrupted_requests": 0,
    }
    accepted_per_pos: dict[int, int] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            value = int(float(parts[-1]))
        except ValueError:
            continue
        if line.startswith("vllm:spec_decode"):
            if "num_drafts" in line:
                result["spec_num_drafts"] += value
            elif "num_draft_tokens" in line:
                result["spec_num_draft_tokens"] += value
            elif "num_accepted_tokens_per_pos" in line:
                match = re.search(r'position="(\d+)"', line)
                if match:
                    pos = int(match.group(1))
                    accepted_per_pos[pos] = accepted_per_pos.get(pos, 0) + value
            elif "num_accepted_tokens" in line:
                result["spec_num_accepted_tokens"] += value
        elif line.startswith("vllm:corrupted_requests"):
            result["corrupted_requests"] += value
    result["spec_accepted_per_pos"] = accepted_per_pos
    return result


def _metric_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    delta = {}
    for key in (
        "spec_num_drafts",
        "spec_num_draft_tokens",
        "spec_num_accepted_tokens",
        "corrupted_requests",
    ):
        delta[key] = int(after.get(key, 0)) - int(before.get(key, 0))
    before_pos = before.get("spec_accepted_per_pos") or {}
    after_pos = after.get("spec_accepted_per_pos") or {}
    per_pos = {}
    for pos in sorted(set(before_pos) | set(after_pos)):
        per_pos[int(pos)] = int(after_pos.get(pos, 0)) - int(before_pos.get(pos, 0))
    delta["spec_accepted_per_pos"] = per_pos
    drafts = delta["spec_num_drafts"]
    draft_tokens = delta["spec_num_draft_tokens"]
    accepted = delta["spec_num_accepted_tokens"]
    delta["spec_acceptance_rate"] = accepted / draft_tokens if draft_tokens else None
    delta["spec_acceptance_length"] = 1.0 + accepted / drafts if drafts else None
    return delta


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def max_run(values: list[Any]) -> int:
    best = 0
    current = object()
    run = 0
    for value in values:
        if value == current:
            run += 1
        else:
            current = value
            run = 1
        best = max(best, run)
    return best


def max_ngram_count(tokens: list[int], n: int) -> int:
    if len(tokens) < n:
        return 0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return max(collections.Counter(grams).values(), default=0)


def anomaly_metrics(text: str, token_ids: list[int]) -> dict[str, Any]:
    chars = [ch for ch in text if not ch.isspace()]
    exclaims = sum(1 for ch in chars if ch in {"!", "！"})
    unique_ratio = len(set(token_ids)) / len(token_ids) if token_ids else 1.0
    metrics = {
        "char_count": len(chars),
        "output_tokens": len(token_ids),
        "exclamation_count": exclaims,
        "exclamation_ratio": exclaims / max(1, len(chars)),
        "max_char_run": max_run(chars),
        "max_token_run": max_run(token_ids),
        "max_4gram_count": max_ngram_count(token_ids, 4),
        "unique_token_ratio": unique_ratio,
    }
    metrics["flagged"] = bool(
        metrics["exclamation_ratio"] >= 0.25
        or metrics["max_char_run"] >= 20
        or metrics["max_token_run"] >= 20
        or metrics["max_4gram_count"] >= 24
        or (len(token_ids) >= 128 and unique_ratio <= 0.08)
    )
    return metrics


def _base_noise_unit(index: int) -> str:
    return (
        f"Segment {index:06d}: 1Cat-vLLM Qwen3.6 V100 audit text. "
        f"Numbers {index * 17 % 9973}, {index * 31 % 7919}, "
        f"{index * 43 % 6151}. This is distractor context only.\n"
    )


def _apply_chat(tokenizer: Any, prompt: str, enable_thinking: bool = False) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def _fit_long_user_prompt(tokenizer: Any, target_tokens: int, code: str) -> str:
    prompt_head = (
        "Read the long context below. It contains exactly one audit code. "
        "Use only the context. Reply with only the code.\n\n<context>\n"
    )
    needle = (
        f"\nAUDIT NEEDLE: The exact audit code is {code}. "
        f"When asked for the audit code, answer exactly {code}.\n\n"
    )
    prompt_tail = "</context>\n\nQuestion: What is the exact audit code?"
    def make_prompt(unit_count: int) -> str:
        units = [_base_noise_unit(i) for i in range(unit_count)]
        insert_at = len(units) // 2
        return (
            prompt_head
            + "".join(units[:insert_at])
            + needle
            + "".join(units[insert_at:])
            + prompt_tail
        )

    def count_prompt(prompt: str) -> int:
        return len(
            tokenizer.encode(
                _apply_chat(tokenizer, prompt),
                add_special_tokens=False,
            )
        )

    unit_tokens = max(
        1,
        len(tokenizer.encode(_base_noise_unit(0), add_special_tokens=False)),
    )
    hi = max(1, int(target_tokens / unit_tokens * 1.30) + 256)
    while count_prompt(make_prompt(hi)) <= target_tokens:
        hi *= 2
    lo = 0
    best = make_prompt(0)
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = make_prompt(mid)
        count = count_prompt(candidate)
        if count <= target_tokens:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _weather_tool() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["city"],
                },
            },
        }
    ]


def _calculator_tool() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Evaluate a basic arithmetic expression.",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            },
        }
    ]


def build_cases(tokenizer: Any, args: argparse.Namespace) -> list[Case]:
    cases = [
        Case(
            case_id="greedy_exact_ok",
            category="greedy_equivalence",
            messages=[{"role": "user", "content": "只输出 OK，不要输出其它字符。"}],
            max_tokens=8,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            chat_template_kwargs={"enable_thinking": False},
            expected_contains=["OK"],
            banned_contains=["<think>", "！", "!"],
            compare_with_baseline=True,
        ),
        Case(
            case_id="greedy_json_contract",
            category="greedy_equivalence",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Return JSON only, no markdown: "
                        '{"audit":"qwen36-mtp","status":"ok","count":3}'
                    ),
                }
            ],
            max_tokens=96,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            chat_template_kwargs={"enable_thinking": False},
            expected_contains=['"audit"', '"status"', '"count"'],
            banned_contains=["```", "<think>", "</think>"],
            compare_with_baseline=True,
        ),
        Case(
            case_id="official_zh_reasoning",
            category="official_sampling",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "请用克制、技术化的中文解释：为什么投机解码不应该改变"
                        "最终输出分布？不要使用感叹号，也不要重复同一句话。"
                    ),
                }
            ],
            max_tokens=args.quality_output_tokens,
            temperature=1.0,
            top_p=0.95,
            top_k=20,
            chat_template_kwargs={"enable_thinking": False},
            banned_contains=["<|im_start|>", "<tool_call>"],
        ),
        Case(
            case_id="official_code_debug",
            category="official_sampling",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "写一个 Python 函数 detect_repetition(text)，返回"
                        "最长重复字符长度、最长重复 token 长度和是否异常。"
                        "代码后用两句话解释边界情况。"
                    ),
                }
            ],
            max_tokens=args.quality_output_tokens,
            temperature=1.0,
            top_p=0.95,
            top_k=20,
            chat_template_kwargs={"enable_thinking": False},
            banned_contains=["<|im_start|>", "<tool_call>"],
        ),
        Case(
            case_id="tool_weather_nonstream",
            category="tool_call",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Call the available weather function for city Beijing "
                        "with unit celsius. Do not answer in natural language."
                    ),
                }
            ],
            max_tokens=192,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            tools=_weather_tool(),
            tool_choice="auto",
            chat_template_kwargs={"enable_thinking": False},
            expected_tool="get_weather",
            expected_tool_args={"city": "Beijing", "unit": "celsius"},
        ),
        Case(
            case_id="tool_calculator_stream",
            category="tool_call_stream",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Use the calculator tool for expression 137*29+418. "
                        "Do not solve it in natural language."
                    ),
                }
            ],
            max_tokens=192,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            tools=_calculator_tool(),
            tool_choice="auto",
            chat_template_kwargs={"enable_thinking": False},
            stream=True,
            expected_tool="calculator",
            expected_tool_args={"expression": "137*29+418"},
        ),
        Case(
            case_id="tool_none_with_tools_available",
            category="tool_call_negative",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "No tool is needed. Answer in one short sentence: "
                        "what is the purpose of a regression test?"
                    ),
                }
            ],
            max_tokens=96,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            tools=_weather_tool(),
            tool_choice="auto",
            chat_template_kwargs={"enable_thinking": False},
            banned_contains=["<tool_call>"],
        ),
    ]
    if not args.skip_long_context:
        for target in _parse_int_list(args.target_lengths):
            code = f"Q36-{target}-MTP-AUDIT"
            prompt = _fit_long_user_prompt(tokenizer, target, code)
            cases.append(
                Case(
                    case_id=f"long_needle_{target}",
                    category="long_context",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=args.long_output_tokens,
                    temperature=0.0,
                    top_p=1.0,
                    top_k=0,
                    chat_template_kwargs={"enable_thinking": False},
                    expected_contains=[code],
                    banned_contains=["<think>", "</think>", "<tool_call>"],
                    compare_with_baseline=target <= 65536,
                )
            )
    return cases


def _extract_nonstream_response(obj: dict[str, Any]) -> dict[str, Any]:
    choice = obj["choices"][0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    token_ids = choice.get("token_ids") or []
    return {
        "content": content,
        "tool_calls": message.get("tool_calls") or [],
        "finish_reason": choice.get("finish_reason"),
        "stop_reason": choice.get("stop_reason"),
        "usage": obj.get("usage") or {},
        "token_ids": token_ids,
    }


def _stream_chat_completion(
    base_url: str,
    payload: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    content_parts: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    usage: dict[str, Any] = {}
    finish_reason = None
    token_ids: list[int] = []
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or line.startswith(":") or not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage") is not None:
                usage = obj["usage"]
            for choice in obj.get("choices") or []:
                finish_reason = choice.get("finish_reason") or finish_reason
                token_ids.extend(choice.get("token_ids") or [])
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    content_parts.append(delta["content"])
                for tc in delta.get("tool_calls") or []:
                    index = int(tc.get("index", 0))
                    acc = tool_calls.setdefault(
                        index,
                        {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                    )
                    if tc.get("id"):
                        acc["id"] += tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        acc["function"]["name"] += fn["name"]
                    if fn.get("arguments"):
                        acc["function"]["arguments"] += fn["arguments"]
    return {
        "content": "".join(content_parts),
        "tool_calls": [tool_calls[idx] for idx in sorted(tool_calls)],
        "finish_reason": finish_reason,
        "stop_reason": None,
        "usage": usage,
        "token_ids": token_ids,
    }


def _run_case(
    base_url: str,
    model_name: str,
    case: Case,
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": case.messages,
        "max_tokens": case.max_tokens,
        "temperature": case.temperature,
        "top_p": case.top_p,
        "top_k": case.top_k,
        "seed": args.seed,
        "stream": case.stream,
        "stream_options": {"include_usage": True} if case.stream else None,
        "tool_choice": case.tool_choice,
        "chat_template_kwargs": case.chat_template_kwargs,
    }
    if case.tools is not None:
        payload["tools"] = case.tools
    payload = {key: value for key, value in payload.items() if value is not None}

    before = _fetch_metrics(base_url)
    start = time.perf_counter()
    if case.stream:
        response = _stream_chat_completion(base_url, payload, args.request_timeout)
    else:
        obj = _request_json(
            f"{base_url}/v1/chat/completions",
            payload=payload,
            timeout=args.request_timeout,
        )
        response = _extract_nonstream_response(obj)
    elapsed = time.perf_counter() - start
    after = _fetch_metrics(base_url)

    text = response["content"]
    token_ids = [int(token_id) for token_id in response.get("token_ids") or []]
    token_metric_source = "response_token_ids"
    if not token_ids:
        # The OpenAI Chat API usually omits vLLM's optional token_ids.  Do not
        # use a repeated placeholder here; that creates false repetition alarms.
        token_ids = [ord(ch) for ch in text if not ch.isspace()]
        token_metric_source = "content_chars"
    grade = _grade_case(case, text, response.get("tool_calls") or [])
    return {
        "case_id": case.case_id,
        "category": case.category,
        "stream": case.stream,
        "sampling": {
            "temperature": case.temperature,
            "top_p": case.top_p,
            "top_k": case.top_k,
            "max_tokens": case.max_tokens,
        },
        "usage": response.get("usage") or {},
        "finish_reason": response.get("finish_reason"),
        "stop_reason": response.get("stop_reason"),
        "latency_sec": elapsed,
        "output_tps": (
            (response.get("usage") or {}).get("completion_tokens", 0) / elapsed
            if elapsed > 0
            else None
        ),
        "content": text,
        "content_prefix": normalize(text)[:800],
        "tool_calls": response.get("tool_calls") or [],
        "token_metric_source": token_metric_source,
        "metrics": anomaly_metrics(text, token_ids),
        "grade": grade,
        "metric_delta": _metric_delta(before, after),
        "compare_with_baseline": case.compare_with_baseline,
    }


def _parse_tool_args(tool_call: dict[str, Any]) -> Any:
    fn = tool_call.get("function") or {}
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            return json.loads(args)
        except json.JSONDecodeError:
            return args
    return args


def _arg_matches(expected: Any, observed: Any) -> bool:
    if isinstance(expected, dict) and isinstance(observed, dict):
        for key, value in expected.items():
            if key not in observed:
                return False
            if str(value).replace(" ", "") != str(observed[key]).replace(" ", ""):
                return False
        return True
    return str(expected).replace(" ", "") in str(observed).replace(" ", "")


def _grade_case(
    case: Case,
    text: str,
    tool_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    failures: list[str] = []
    normalized = normalize(text)
    for expected in case.expected_contains or []:
        if expected not in text and expected not in normalized:
            failures.append(f"missing expected substring: {expected}")
    for banned in case.banned_contains or []:
        if banned in text:
            failures.append(f"banned substring present: {banned}")
    if case.expected_tool is not None:
        if not tool_calls:
            failures.append("missing tool_calls")
        else:
            first = tool_calls[0]
            fn = first.get("function") or {}
            name = fn.get("name")
            if name != case.expected_tool:
                failures.append(f"tool name mismatch: expected {case.expected_tool}, got {name}")
            if case.expected_tool_args:
                observed_args = _parse_tool_args(first)
                if not _arg_matches(case.expected_tool_args, observed_args):
                    failures.append(
                        f"tool args mismatch: expected {case.expected_tool_args}, got {observed_args}"
                    )
    elif case.category == "tool_call_negative" and tool_calls:
        failures.append(f"unexpected tool_calls: {tool_calls}")
    return {"passed": not failures, "failures": failures}


def _compare_against_baseline(
    baseline_rows: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    for row in rows:
        if not row.get("compare_with_baseline"):
            continue
        baseline = baseline_rows.get(row["case_id"])
        if baseline is None:
            continue
        if row["category"] == "greedy_equivalence":
            base_text = normalize(baseline["content"])
            cur_text = normalize(row["content"])
            if cur_text != base_text:
                row["grade"]["passed"] = False
                row["grade"]["failures"].append(
                    "greedy output differs from baseline"
                )
                row["baseline_content_prefix"] = baseline["content_prefix"]
        elif row["category"] == "long_context":
            base_passed = bool(baseline.get("grade", {}).get("passed"))
            if base_passed and not row["grade"]["passed"]:
                row["grade"]["failures"].append(
                    "long-context case passed baseline but failed current scenario"
                )


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [
        row["case_id"]
        for row in rows
        if not row["grade"]["passed"] or row["metrics"]["flagged"]
        or row["metric_delta"].get("corrupted_requests", 0) > 0
    ]
    output_tps = [
        float(row["output_tps"])
        for row in rows
        if row.get("output_tps") is not None and row.get("output_tps") > 0
    ]
    acceptance_lengths = [
        float(row["metric_delta"]["spec_acceptance_length"])
        for row in rows
        if row["metric_delta"].get("spec_acceptance_length") is not None
    ]
    return {
        "passed": len(failed) == 0,
        "total_cases": len(rows),
        "failed_or_flagged_cases": failed,
        "median_output_tps": statistics.median(output_tps) if output_tps else None,
        "median_spec_acceptance_length": (
            statistics.median(acceptance_lengths) if acceptance_lengths else None
        ),
    }


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
    )
    cases = build_cases(tokenizer, args)
    (output_dir / "cases.json").write_text(
        json.dumps(
            [
                {
                    "case_id": case.case_id,
                    "category": case.category,
                    "max_tokens": case.max_tokens,
                    "temperature": case.temperature,
                    "top_p": case.top_p,
                    "top_k": case.top_k,
                    "stream": case.stream,
                    "has_tools": case.tools is not None,
                    "compare_with_baseline": case.compare_with_baseline,
                }
                for case in cases
            ],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    baseline_rows: dict[str, dict[str, Any]] = {}
    scenario_summaries: dict[str, Any] = {}
    for index, scenario in enumerate(_scenarios(args)):
        port = args.base_port + index
        scenario_dir = output_dir / scenario.name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        env = _base_env(args, scenario_dir)
        command = _server_command(args, scenario, port)
        log_path = scenario_dir / "server.log"
        with log_path.open("w", encoding="utf-8") as server_log:
            server_log.write(" ".join(command) + "\n")
            server_log.flush()
            proc = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )
            try:
                _wait_for_server(args.host, port, args.startup_timeout, proc)
                base_url = f"http://{args.host}:{port}"
                model_name = f"{args.served_model_name}-{scenario.name}"
                rows: list[dict[str, Any]] = []
                for case in cases:
                    try:
                        row = _run_case(base_url, model_name, case, args)
                    except urllib.error.HTTPError as exc:
                        body = exc.read().decode("utf-8", errors="replace")
                        row = {
                            "case_id": case.case_id,
                            "category": case.category,
                            "grade": {
                                "passed": False,
                                "failures": [
                                    f"HTTPError {exc.code}: {body[:1000]}"
                                ],
                            },
                            "metrics": {"flagged": True},
                            "metric_delta": {},
                            "content": "",
                            "content_prefix": "",
                        }
                    except Exception as exc:
                        row = {
                            "case_id": case.case_id,
                            "category": case.category,
                            "grade": {
                                "passed": False,
                                "failures": [f"{type(exc).__name__}: {exc}"],
                            },
                            "metrics": {"flagged": True},
                            "metric_delta": {},
                            "content": "",
                            "content_prefix": "",
                        }
                    rows.append(row)
                    print(
                        json.dumps(
                            {
                                "scenario": scenario.name,
                                "case": row["case_id"],
                                "passed": row["grade"]["passed"],
                                "flagged": row["metrics"].get("flagged"),
                                "corrupted_delta": row.get("metric_delta", {}).get(
                                    "corrupted_requests"
                                ),
                                "spec_acc_len": row.get("metric_delta", {}).get(
                                    "spec_acceptance_length"
                                ),
                                "output": row.get("content_prefix", "")[:180],
                                "failures": row["grade"].get("failures", []),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                if scenario.speculative_config is None:
                    baseline_rows = {row["case_id"]: row for row in rows}
                else:
                    _compare_against_baseline(baseline_rows, rows)

                payload = {
                    "scenario": scenario.name,
                    "speculative_config": scenario.speculative_config,
                    "command": command,
                    "rows": rows,
                    "summary": _summarize(rows),
                }
                scenario_summaries[scenario.name] = payload["summary"]
                (scenario_dir / "results.json").write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            finally:
                _terminate_process(proc)

    summary = {
        "model": args.model,
        "attention_backend": args.attention_backend,
        "max_model_len": args.max_model_len,
        "target_lengths": _parse_int_list(args.target_lengths),
        "scenario_summaries": scenario_summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    failed = [
        name for name, item in scenario_summaries.items() if not item.get("passed")
    ]
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
