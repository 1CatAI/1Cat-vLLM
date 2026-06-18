# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the SM70 Qwen3.6 quality/speed matrix in isolated subprocesses.

The matrix is intentionally narrow and reproducible:
- 27B/35B, AWQ/FP8 checkpoints.
- KV cache off (`auto`) and on (`fp8_e5m2`).
- TP2 on V100, Flash-V100 attention, optional MTP4.
- Speed: 4K input / 1K output, with prefill and steady decode separated.
- Quality: chat-template prompts using each model's generation_config sampling.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclasses.dataclass(frozen=True)
class ModelCase:
    name: str
    model: str
    quantization: str
    dtype: str = "half"


MODELS = (
    ModelCase(
        "27b-awq",
        "/home/ymzx/models/Qwen3.6-27B-AWQ",
        "awq",
    ),
    ModelCase(
        "35b-awq",
        "/home/ymzx/models/Qwen3.6-35B-A3B-AWQ",
        "awq",
    ),
    ModelCase(
        "27b-fp8",
        "/home/ymzx/models/Qwen3.6-27B-FP8",
        "fp8",
    ),
    ModelCase(
        "35b-fp8",
        "/home/ymzx/models/Qwen3.6-35B-A3B-FP8",
        "fp8",
    ),
)

KV_CACHE_CASES = ("auto", "fp8_e5m2")

QUALITY_PROMPTS = (
    {
        "id": "code_macos",
        "content": (
            "帮我用 HTML、CSS、JavaScript 做一个 macOS 桌面模拟器，"
            "功能尽可能完整：菜单栏、Dock、可拖动窗口、Finder、设置、"
            "终端、文件管理、窗口最小化/关闭、时钟和主题切换。"
            "请直接给出可以运行的完整代码，不要重复段落。"
        ),
    },
    {
        "id": "code_snake",
        "content": (
            "请用 Python pygame 写一个完整可玩的贪吃蛇游戏，包含开始界面、"
            "暂停、计分、速度递增、碰撞检测、重新开始和代码注释。"
            "输出完整代码，避免重复代码块。"
        ),
    },
    {
        "id": "long_story_review",
        "content": (
            "写一篇约 1000 字中文现实主义短篇小说，主题是雨夜、错过和重逢。"
            "小说后用四点评价优点和不足。不要把同一段话重复输出。"
        ),
    },
    {
        "id": "structured_design",
        "content": (
            "设计一个任务管理 SaaS 的数据库和接口方案，要求包含用户、项目、"
            "任务、评论、附件、权限、审计日志、索引和失败重试策略。"
            "用清晰的小标题输出，不要输出无意义数字串。"
        ),
    },
)

DETERMINISM_PROMPT = (
    "用三点解释 CUDA graph 为什么要求 replay 时输入 tensor 地址稳定，"
    "并说明这对模型输出质量验证有什么影响。"
)


def _sha256_ids(token_ids: list[int]) -> str:
    raw = ",".join(str(i) for i in token_ids).encode()
    return hashlib.sha256(raw).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _load_generation_config(model: str) -> dict[str, Any]:
    path = Path(model) / "generation_config.json"
    if not path.exists():
        return {"temperature": 1.0, "top_p": 1.0, "top_k": -1}
    data = json.loads(path.read_text())
    return {
        "temperature": float(data.get("temperature", 1.0)),
        "top_p": float(data.get("top_p", 1.0)),
        "top_k": int(data.get("top_k", -1)),
    }


def _make_exact_prompt_token_ids(tokenizer: Any, target_len: int) -> list[int]:
    base = (
        "SM70 V100 benchmark prompt. The model should continue with stable, "
        "coherent technical prose about CUDA kernels, attention, KV cache, "
        "and long-context decoding. "
    )
    chunk = tokenizer.encode(base, add_special_tokens=False)
    if not chunk:
        raise RuntimeError("benchmark prompt encoded to no tokens")
    token_ids: list[int] = []
    while len(token_ids) < target_len:
        token_ids.extend(chunk)
    return token_ids[:target_len]


def _make_chat_prompt(tokenizer: Any, content: str) -> str:
    messages = [{"role": "user", "content": content}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _safe_delta(end: float, start: float) -> float | None:
    if end <= 0.0 or start <= 0.0:
        return None
    return end - start


def _request_metrics(metrics: Any, prompt_tokens: int,
                     output_tokens: int) -> dict[str, Any] | None:
    if metrics is None:
        return None
    prefill_time = _safe_delta(metrics.first_token_ts, metrics.scheduled_ts)
    decode_time = _safe_delta(metrics.last_token_ts, metrics.first_token_ts)
    steady_tokens = max(output_tokens - 1, 0)
    return {
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "prefill_time_s": prefill_time,
        "decode_time_s": decode_time,
        "prefill_tps": (
            prompt_tokens / prefill_time if prefill_time else None
        ),
        "steady_decode_tokens": steady_tokens,
        "steady_decode_tps": (
            steady_tokens / decode_time
            if decode_time and steady_tokens > 0
            else None
        ),
        "first_token_latency_s": metrics.first_token_latency,
        "finish_num_generation_tokens": metrics.num_generation_tokens,
        "is_corrupted": metrics.is_corrupted,
    }


def _longest_char_run(text: str, predicate) -> int:
    best = 0
    current = 0
    for ch in text:
        if predicate(ch):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _longest_same_token_run(token_ids: list[int]) -> int:
    best = 0
    last = object()
    current = 0
    for token_id in token_ids:
        if token_id == last:
            current += 1
        else:
            last = token_id
            current = 1
        best = max(best, current)
    return best


def _longest_same_char_run(text: str) -> int:
    best = 0
    last = None
    current = 0
    for ch in text:
        if ch == last:
            current += 1
        else:
            last = ch
            current = 1
        best = max(best, current)
    return best


def _max_repeated_window(text: str, width: int) -> int:
    normalized = re.sub(r"\s+", " ", text)
    if len(normalized) < width:
        return 0
    counts: dict[str, int] = {}
    for idx in range(0, len(normalized) - width + 1):
        window = normalized[idx:idx + width]
        if len(window.strip()) < width // 2:
            continue
        counts[window] = counts.get(window, 0) + 1
    return max(counts.values(), default=0)


def _max_same_line_run(text: str) -> int:
    best = 0
    last = None
    current = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == last:
            current += 1
        else:
            last = line
            current = 1
        best = max(best, current)
    return best


def _quality_metrics(text: str, token_ids: list[int]) -> dict[str, Any]:
    bad_markers = [
        "rgba(rgba",
        "UTF-UTF",
        "propertycorrectly",
        "255555555",
        "00000000000000000000",
        "55555555555555555555",
        "\ufffd",
    ]
    marker_hits = {m: text.count(m) for m in bad_markers if m in text}
    metrics = {
        "chars": len(text),
        "tokens": len(token_ids),
        "text_hash": _sha256_text(text),
        "token_hash": _sha256_ids(token_ids),
        "max_same_token_run": _longest_same_token_run(token_ids),
        "max_digit_run": _longest_char_run(text, str.isdigit),
        "max_same_char_run": _longest_same_char_run(text),
        "repeat20": _max_repeated_window(text, 20),
        "repeat50": _max_repeated_window(text, 50),
        "repeat100": _max_repeated_window(text, 100),
        "max_same_line_run": _max_same_line_run(text),
        "replacement_char_count": text.count("\ufffd"),
        "bad_marker_hits": marker_hits,
    }
    failures = []
    if metrics["tokens"] < 64:
        failures.append("too_few_output_tokens")
    if metrics["max_same_token_run"] > 48:
        failures.append("same_token_run")
    if metrics["max_digit_run"] > 80:
        failures.append("digit_run")
    if metrics["max_same_char_run"] > 120:
        failures.append("same_char_run")
    if metrics["repeat20"] > 80:
        failures.append("repeat20")
    if metrics["repeat50"] > 40:
        failures.append("repeat50")
    if metrics["repeat100"] > 20:
        failures.append("repeat100")
    if metrics["max_same_line_run"] > 12:
        failures.append("same_line_run")
    if marker_hits:
        failures.append("bad_marker")
    metrics["failures"] = failures
    metrics["passed"] = not failures
    return metrics


def _output_record(request_output: Any) -> tuple[str, list[int], str, Any]:
    output = request_output.outputs[0]
    token_ids = list(output.token_ids)
    text = output.text
    return text, token_ids, output.finish_reason, output.stop_reason


def _run_worker(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )
    generation_config = _load_generation_config(args.model)
    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "trust_remote_code": True,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "quantization": args.quantization,
        "kv_cache_dtype": args.kv_cache_dtype,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": 1,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "attention_backend": "FLASH_ATTN_V100",
        "seed": args.seed,
        "disable_log_stats": False,
    }
    if args.speculative_tokens > 0:
        llm_kwargs["speculative_config"] = {
            "method": "mtp",
            "num_speculative_tokens": args.speculative_tokens,
        }
    if args.kv_cache_dtype == "auto":
        llm_kwargs.pop("kv_cache_dtype")
    recorded_engine_kwargs = json.loads(json.dumps(llm_kwargs, default=str))

    start = time.perf_counter()
    llm = LLM(**llm_kwargs)
    load_seconds = time.perf_counter() - start

    warmup_prompt = [{"prompt_token_ids": _make_exact_prompt_token_ids(tokenizer, 512)}]
    llm.generate(
        warmup_prompt,
        SamplingParams(
            max_tokens=32,
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            skip_special_tokens=False,
        ),
    )

    speed_prompt_ids = _make_exact_prompt_token_ids(tokenizer, args.speed_input_len)
    speed_sampling = SamplingParams(
        max_tokens=args.speed_output_len,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        skip_special_tokens=False,
    )
    start = time.perf_counter()
    speed_outputs = llm.generate(
        [{"prompt_token_ids": speed_prompt_ids}],
        speed_sampling,
    )
    speed_wall_s = time.perf_counter() - start
    speed_text, speed_token_ids, finish_reason, stop_reason = _output_record(
        speed_outputs[0])
    speed_metrics = _request_metrics(
        speed_outputs[0].metrics,
        len(speed_prompt_ids),
        len(speed_token_ids),
    )

    official_sampling = SamplingParams(
        max_tokens=args.quality_max_tokens,
        temperature=generation_config["temperature"],
        top_p=generation_config["top_p"],
        top_k=generation_config["top_k"],
        seed=args.sampling_seed,
        skip_special_tokens=False,
    )
    quality_prompts = list(QUALITY_PROMPTS)
    if args.quality_prompt_id:
        selected_quality_prompts = set(args.quality_prompt_id)
        quality_prompts = [
            prompt
            for prompt in quality_prompts
            if prompt["id"] in selected_quality_prompts
        ]
        missing = selected_quality_prompts - {
            prompt["id"] for prompt in quality_prompts
        }
        if missing:
            raise ValueError(f"Unknown quality prompt ids: {sorted(missing)}")

    quality_records = []
    for prompt in quality_prompts:
        chat_prompt = _make_chat_prompt(tokenizer, prompt["content"])
        for repeat_idx in range(args.quality_repeat):
            request = llm.generate([chat_prompt], official_sampling)[0]
            text, token_ids, q_finish_reason, q_stop_reason = _output_record(
                request
            )
            quality_records.append({
                "id": prompt["id"],
                "repeat": repeat_idx + 1,
                "finish_reason": q_finish_reason,
                "stop_reason": q_stop_reason,
                "prompt_tokens": len(request.prompt_token_ids or []),
                "metrics": _quality_metrics(text, token_ids),
                "preview": text[:1000],
                "tail": text[-1000:],
            })

    det_prompt = _make_chat_prompt(tokenizer, DETERMINISM_PROMPT)
    det_sampling = SamplingParams(
        max_tokens=256,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        skip_special_tokens=False,
    )
    det_outputs = [llm.generate([det_prompt], det_sampling)[0] for _ in range(2)]
    det_records = []
    for output in det_outputs:
        text, token_ids, det_finish_reason, det_stop_reason = _output_record(output)
        det_records.append({
            "finish_reason": det_finish_reason,
            "stop_reason": det_stop_reason,
            "metrics": _quality_metrics(text, token_ids),
            "preview": text[:800],
        })
    deterministic_exact = (
        det_records[0]["metrics"]["token_hash"]
        == det_records[1]["metrics"]["token_hash"]
    )

    case_quality_passed = (
        all(record["metrics"]["passed"] for record in quality_records)
        and deterministic_exact
    )
    payload = {
        "case": {
            "name": args.case_name,
            "model": args.model,
            "quantization": args.quantization,
            "dtype": args.dtype,
            "kv_cache_dtype": args.kv_cache_dtype,
            "tensor_parallel_size": args.tensor_parallel_size,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "speculative_tokens": args.speculative_tokens,
        },
        "engine_kwargs": recorded_engine_kwargs,
        "env": {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith((
                "CUDA_",
                "VLLM_",
                "TORCHINDUCTOR_",
                "TRITON_",
                "PYTHONPATH",
            ))
        },
        "generation_config": generation_config,
        "load_seconds": load_seconds,
        "speed": {
            "input_len": args.speed_input_len,
            "max_tokens": args.speed_output_len,
            "wall_seconds": speed_wall_s,
            "finish_reason": finish_reason,
            "stop_reason": stop_reason,
            "output_tokens": len(speed_token_ids),
            "text_hash": _sha256_text(speed_text),
            "token_hash": _sha256_ids(speed_token_ids),
            "request_metrics": speed_metrics,
        },
        "quality": {
            "sampling": {
                **generation_config,
                "seed": args.sampling_seed,
                "max_tokens": args.quality_max_tokens,
                "ignore_eos": False,
            },
            "records": quality_records,
            "determinism": {
                "exact_token_match": deterministic_exact,
                "records": det_records,
            },
            "passed": case_quality_passed,
        },
    }
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps({
        "case": args.case_name,
        "quality_passed": case_quality_passed,
        "prefill_tps": (
            speed_metrics or {}
        ).get("prefill_tps"),
        "steady_decode_tps": (
            speed_metrics or {}
        ).get("steady_decode_tps"),
        "out": str(args.out),
    }, ensure_ascii=False))
    return 0 if case_quality_passed and speed_metrics else 2


def _parse_kv_capacity(log_text: str) -> dict[str, Any]:
    patterns = {
        "gpu_kv_cache_tokens": r"GPU KV cache size:\s*([0-9,]+)\s*tokens",
        "maximum_concurrency": r"Maximum concurrency.*?:\s*([0-9.]+)x",
        "num_gpu_blocks": r"# GPU blocks:\s*([0-9,]+)",
    }
    found: dict[str, Any] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, log_text)
        if not match:
            continue
        raw = match.group(1).replace(",", "")
        found[key] = float(raw) if "." in raw else int(raw)
    return found


def _spec_tokens_from_case_name(case_name: str) -> int:
    match = re.search(r"-mtp(\d+)$", case_name)
    return int(match.group(1)) if match else 0


def _summarize_case(path: Path, log_path: Path,
                    returncode: int) -> dict[str, Any]:
    if not path.exists():
        return {
            "case": path.stem,
            "speculative_tokens": _spec_tokens_from_case_name(path.stem),
            "status": "missing_json",
            "returncode": returncode,
            "log": str(log_path),
        }
    data = json.loads(path.read_text())
    log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
    speed_metrics = (data.get("speed") or {}).get("request_metrics") or {}
    quality = data.get("quality") or {}
    failed_prompts = [
        {
            "id": record["id"],
            "failures": record["metrics"]["failures"],
        }
        for record in quality.get("records", [])
        if not record["metrics"]["passed"]
    ]
    return {
        "case": data["case"]["name"],
        "model": Path(data["case"]["model"]).name,
        "quantization": data["case"]["quantization"],
        "kv_cache_dtype": data["case"]["kv_cache_dtype"],
        "speculative_tokens": data["case"].get("speculative_tokens", 0),
        "returncode": returncode,
        "load_seconds": data.get("load_seconds"),
        "prefill_tps": speed_metrics.get("prefill_tps"),
        "prefill_time_s": speed_metrics.get("prefill_time_s"),
        "decode_tps": speed_metrics.get("steady_decode_tps"),
        "decode_time_s": speed_metrics.get("decode_time_s"),
        "speed_output_tokens": (data.get("speed") or {}).get("output_tokens"),
        "quality_passed": quality.get("passed"),
        "deterministic_exact": (
            quality.get("determinism") or {}
        ).get("exact_token_match"),
        "failed_prompts": failed_prompts,
        "kv_capacity": _parse_kv_capacity(log_text),
        "json": str(path),
        "log": str(log_path),
    }


def _write_summary(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    summary_json = out_dir / "summary.json"
    summary_json.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# SM70 Qwen3.6 Quality/Speed Matrix",
        "",
        "| Case | Spec | Quality | Deterministic | Prefill tok/s | "
        "Decode tok/s | Output tok | KV capacity | Artifact |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        kv_capacity = row.get("kv_capacity") or {}
        kv_text = ", ".join(f"{k}={v}" for k, v in kv_capacity.items()) or "-"
        lines.append(
            (
                "| {case} | {spec} | {quality} | {det} | {prefill} | "
                "{decode} | {out_tok} | {kv} | {artifact} |"
            ).format(
                case=row.get("case"),
                spec=row.get("speculative_tokens", 0),
                quality="PASS" if row.get("quality_passed") else "FAIL",
                det="PASS" if row.get("deterministic_exact") else "FAIL",
                prefill=(
                    f"{row['prefill_tps']:.1f}"
                    if isinstance(row.get("prefill_tps"), (int, float))
                    else "-"
                ),
                decode=(
                    f"{row['decode_tps']:.1f}"
                    if isinstance(row.get("decode_tps"), (int, float))
                    else "-"
                ),
                out_tok=row.get("speed_output_tokens") or "-",
                kv=kv_text,
                artifact=Path(row.get("json", "")).name,
            )
        )
    lines.extend([
        "",
        "Quality gates: official generation_config sampling, no ignore_eos; "
        "fatal gates are repeated windows, long digit/same-token runs, bad "
        "markers, and greedy repeat determinism mismatch.",
    ])
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _default_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env["PYTHONPATH"] = (
        f"{REPO_ROOT}:{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else str(REPO_ROOT)
    )
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor_ymzx")
    env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    env.setdefault("TRITON_CACHE_AUTOTUNING", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
    env.setdefault("VLLM_USE_AOT_COMPILE", "1")
    env.setdefault("VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH", "1")
    env.setdefault("VLLM_SM70_LM_HEAD_TOP1", "0")
    return env


def _run_matrix(args: argparse.Namespace) -> int:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or REPO_ROOT / "bench_results" / (
        f"sm70_quality_speed_matrix_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    env = _default_env(args)

    selected = set(args.only or [])
    for model_case in MODELS:
        for kv_cache_dtype in KV_CACHE_CASES:
            case_name = f"{model_case.name}-kv-{kv_cache_dtype}"
            if args.speculative_tokens > 0:
                case_name = f"{case_name}-mtp{args.speculative_tokens}"
            if selected and case_name not in selected:
                continue
            out_path = out_dir / f"{case_name}.json"
            log_path = out_dir / f"{case_name}.log"
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--case-name",
                case_name,
                "--model",
                model_case.model,
                "--quantization",
                model_case.quantization,
                "--dtype",
                model_case.dtype,
                "--kv-cache-dtype",
                kv_cache_dtype,
                "--tensor-parallel-size",
                str(args.tensor_parallel_size),
                "--max-model-len",
                str(args.max_model_len),
                "--max-num-batched-tokens",
                str(args.max_num_batched_tokens),
                "--gpu-memory-utilization",
                str(args.gpu_memory_utilization),
                "--speed-input-len",
                str(args.speed_input_len),
                "--speed-output-len",
                str(args.speed_output_len),
                "--quality-max-tokens",
                str(args.quality_max_tokens),
                "--quality-repeat",
                str(args.quality_repeat),
                "--sampling-seed",
                str(args.sampling_seed),
                "--seed",
                str(args.seed),
                "--speculative-tokens",
                str(args.speculative_tokens),
                "--out",
                str(out_path),
            ]
            for prompt_id in args.quality_prompt_id or []:
                cmd.extend(["--quality-prompt-id", prompt_id])
            print(f"[matrix] running {case_name}", flush=True)
            with log_path.open("w", encoding="utf-8") as log_file:
                proc = subprocess.run(
                    cmd,
                    cwd=REPO_ROOT,
                    env=env,
                    text=True,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            row = _summarize_case(out_path, log_path, proc.returncode)
            rows.append(row)
            _write_summary(out_dir, rows)
            print(
                "[matrix] done {case}: rc={rc}, quality={quality}, "
                "prefill={prefill}, decode={decode}".format(
                    case=case_name,
                    rc=proc.returncode,
                    quality=row.get("quality_passed"),
                    prefill=row.get("prefill_tps"),
                    decode=row.get("decode_tps"),
                ),
                flush=True,
            )
    _write_summary(out_dir, rows)
    print(f"[matrix] summary: {out_dir / 'summary.md'}", flush=True)
    return 0 if rows and all(r.get("quality_passed") for r in rows) else 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--only", action="append")
    parser.add_argument("--cuda-visible-devices", default="2,3")
    parser.add_argument("--case-name", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--quantization", default="")
    parser.add_argument("--dtype", default="half")
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=262144)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--speed-input-len", type=int, default=4096)
    parser.add_argument("--speed-output-len", type=int, default=1024)
    parser.add_argument("--quality-max-tokens", type=int, default=2048)
    parser.add_argument("--quality-prompt-id", action="append")
    parser.add_argument("--quality-repeat", type=int, default=1)
    parser.add_argument("--sampling-seed", type=int, default=20260617)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speculative-tokens", type=int, default=0)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.worker:
        if args.out is None:
            raise ValueError("--out is required in worker mode")
        return _run_worker(args)
    return _run_matrix(args)


if __name__ == "__main__":
    raise SystemExit(main())
