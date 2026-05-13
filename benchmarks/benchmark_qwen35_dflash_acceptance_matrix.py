#!/usr/bin/env python3
"""Acceptance and speed matrix for Qwen3.5-9B DFlash on natural tasks.

This runner launches comparable servers across environments, sends streamed
chat/completions requests, and records:

- TTFT / TPOT
- prefill_tps / decode_tps / e2e_tps
- speculative acceptance metrics from /metrics deltas

It supports two prompt sources:

- a local Chinese natural-task JSONL suite
- SpecBench category samples
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import statistics
import subprocess
import time
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


CURRENT_REPO = Path("/home/ymzx/桌面/dflash/vllm-sm70-dflash/1Cat-vllm-0.0.4/vllm")
ORACLE_REPO = Path("/home/ymzx/桌面/dflash/vllm")
DEFAULT_MODEL = Path("/home/ymzx/models/Qwen3.5-9B-AWQ")
DEFAULT_DRAFT_MODEL = Path("/home/ymzx/models/Qwen3.5-9B-DFlash")
DEFAULT_NATURAL_SUITE = CURRENT_REPO / "benchmarks/data/qwen35_natural_zh_acceptance.jsonl"
DEFAULT_SPEC_BENCH_CACHE = Path("/tmp/spec_bench_question.jsonl")
SPEC_BENCH_URL = (
    "https://raw.githubusercontent.com/hemingkx/Spec-Bench/refs/heads/main/"
    "data/spec_bench/question.jsonl"
)


@dataclass(frozen=True)
class ServerPreset:
    name: str
    repo_root: Path
    python_executable: str
    gpu_ids: str
    tensor_parallel_size: int
    attention_backend: str | None
    port: int
    extra_env: dict[str, str]
    supports_swap_space: bool = True


@dataclass(frozen=True)
class Scenario:
    name: str
    speculative_config: dict[str, Any] | None


PRESETS: dict[str, ServerPreset] = {
    "v100_tp4_current": ServerPreset(
        name="v100_tp4_current",
        repo_root=CURRENT_REPO,
        python_executable="/home/ymzx/miniconda3/envs/vllm-sm70/bin/python",
        gpu_ids="1,2,3,4",
        tensor_parallel_size=4,
        attention_backend="FLASH_ATTN_V100",
        port=8310,
        extra_env={"VLLM_FLASH_V100_ENABLE_PAGED_PREFILL": "1"},
    ),
    "v100_tp1_current": ServerPreset(
        name="v100_tp1_current",
        repo_root=CURRENT_REPO,
        python_executable="/home/ymzx/miniconda3/envs/vllm-sm70/bin/python",
        gpu_ids="1",
        tensor_parallel_size=1,
        attention_backend="FLASH_ATTN_V100",
        port=8311,
        extra_env={"VLLM_FLASH_V100_ENABLE_PAGED_PREFILL": "1"},
    ),
    "rtx3090_tp1_oracle": ServerPreset(
        name="rtx3090_tp1_oracle",
        repo_root=ORACLE_REPO,
        python_executable="/home/ymzx/miniconda3/envs/vllm-dflash-9b-wheel/bin/python",
        gpu_ids="0",
        tensor_parallel_size=1,
        attention_backend=None,
        port=8312,
        extra_env={},
        supports_swap_space=False,
    ),
}

def _build_scenarios(
    draft_model: Path,
    num_speculative_tokens: int,
) -> dict[str, Scenario]:
    return {
        "baseline": Scenario(name="baseline", speculative_config=None),
        "dflash": Scenario(
            name="dflash",
            speculative_config={
                "method": "dflash",
                "model": str(draft_model),
                "num_speculative_tokens": num_speculative_tokens,
            },
        ),
    }


def _median(values: list[float | int]) -> float | None:
    cleaned = [float(v) for v in values if v is not None]
    if not cleaned:
        return None
    return float(statistics.median(cleaned))


def _download_spec_bench_if_missing(path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(SPEC_BENCH_URL, path)
    return path


def _load_natural_suite(path: Path, categories: set[str] | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if categories and row["category"] not in categories:
                continue
            rows.append(
                {
                    "id": row["id"],
                    "category": row["category"],
                    "prompt": row["prompt"],
                }
            )
    return rows


def _load_spec_bench_suite(
    path: Path,
    categories: set[str] | None,
    max_prompts_per_category: int,
    seed: int,
) -> list[dict[str, str]]:
    path = _download_spec_bench_if_missing(path)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            category = row.get("category", "unknown")
            if categories and category not in categories:
                continue
            turns = row.get("turns") or []
            if not turns:
                continue
            grouped[category].append(
                {
                    "id": f"spec_bench::{category}::{len(grouped[category]):04d}",
                    "category": category,
                    "prompt": turns[0],
                }
            )

    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    for category in sorted(grouped):
        samples = grouped[category]
        rng.shuffle(samples)
        rows.extend(samples[:max_prompts_per_category])
    return rows


def _fetch_spec_decode_metrics(base_url: str) -> tuple[int, int, int, dict[int, int]]:
    txt = requests.get(f"{base_url}/metrics", timeout=30).text
    drafts = 0
    draft_tokens = 0
    accepted_tokens = 0
    accepted_per_pos: dict[int, int] = {}
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or not line.startswith("vllm:spec_decode"):
            continue
        parts = line.split()
        val = int(float(parts[-1]))
        if "num_drafts" in line:
            drafts += val
        elif "num_draft_tokens" in line:
            draft_tokens += val
        elif "num_accepted_tokens_per_pos" in line:
            pos_label = 'position="'
            if pos_label in line:
                start = line.index(pos_label) + len(pos_label)
                end = line.index('"', start)
                pos = int(line[start:end])
                accepted_per_pos[pos] = accepted_per_pos.get(pos, 0) + val
        elif "num_accepted_tokens" in line:
            accepted_tokens += val
    return drafts, draft_tokens, accepted_tokens, accepted_per_pos


def _format_per_pos_rate(
    before: dict[int, int],
    after: dict[int, int],
    delta_drafts: int,
) -> list[float]:
    if delta_drafts <= 0:
        return []
    positions = sorted(set(before) | set(after))
    out: list[float] = []
    for pos in positions:
        out.append((after.get(pos, 0) - before.get(pos, 0)) / delta_drafts)
    return out


def _wait_ready(base_url: str, timeout_s: int, proc: subprocess.Popen[Any]) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        try:
            r = requests.get(f"{base_url}/v1/models", timeout=5)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"timed out waiting for {base_url}")


def _stop_server(proc: subprocess.Popen[Any] | None) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=30)
            return
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                return
            proc.wait(timeout=30)


def _start_server(
    *,
    preset: ServerPreset,
    scenario: Scenario,
    model: Path,
    draft_model: Path,
    dtype: str,
    quantization: str,
    max_model_len: int,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    gpu_memory_utilization: float,
    timeout_s: int,
    cache_root: Path,
) -> tuple[subprocess.Popen[Any], str, Path]:
    base_url = f"http://127.0.0.1:{preset.port}"
    log_path = cache_root / f"{preset.name}_{scenario.name}_server.log"
    cache_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = preset.gpu_ids
    env["CUDA_HOME"] = "/usr/local/cuda-12.8"
    env["PATH"] = f"/usr/local/cuda-12.8/bin:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = (
        f"/usr/local/cuda-12.8/lib64:{env.get('LD_LIBRARY_PATH', '')}"
    )
    env["PYTHONPATH"] = f"{preset.repo_root}:{env.get('PYTHONPATH', '')}".rstrip(":")
    env["VLLM_CACHE_ROOT"] = str(cache_root / "cache")
    env.update(preset.extra_env)

    cmd = [
        preset.python_executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model),
        "--served-model-name",
        f"{preset.name}-{scenario.name}",
        "--tensor-parallel-size",
        str(preset.tensor_parallel_size),
        "--dtype",
        dtype,
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-model-len",
        str(max_model_len),
        "--max-num-batched-tokens",
        str(max_num_batched_tokens),
        "--max-num-seqs",
        str(max_num_seqs),
        "--skip-mm-profiling",
        "--host",
        "127.0.0.1",
        "--port",
        str(preset.port),
        "--compilation-config",
        '{"cudagraph_mode":"full_and_piecewise","cudagraph_capture_sizes":[1,2,4,8,16,32]}',
        "--default-chat-template-kwargs",
        '{"enable_thinking": false}',
        "--trust-remote-code",
    ]
    if quantization != "none":
        cmd.extend(["--quantization", quantization])
    if preset.supports_swap_space:
        cmd.extend(["--swap-space", "4"])
    if preset.attention_backend:
        cmd.extend(["--attention-backend", preset.attention_backend])
    if scenario.speculative_config is not None:
        speculative_config = dict(scenario.speculative_config)
        speculative_config["model"] = str(draft_model)
        cmd.extend(["--speculative-config", json.dumps(speculative_config)])

    logf = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=preset.repo_root,
        env=env,
        stdout=logf,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )
    _wait_ready(base_url, timeout_s, proc)
    return proc, base_url, log_path


def _stream_request(
    *,
    base_url: str,
    model_name: str,
    prompt: str,
    max_completion_tokens: int,
    enable_thinking: bool,
) -> dict[str, Any]:
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_completion_tokens": max_completion_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    st = time.perf_counter()
    first = None
    last = st
    chunks = 0
    usage = None
    content: list[str] = []
    with requests.post(
        f"{base_url}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload,
        stream=True,
        timeout=600,
    ) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or raw.startswith(":") or not raw.startswith("data: "):
                continue
            data = raw[6:]
            if data == "[DONE]":
                break
            obj = json.loads(data)
            now = time.perf_counter()
            choices = obj.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                piece = delta.get("content", "") or delta.get("reasoning_content", "")
                if piece:
                    chunks += 1
                    if first is None:
                        first = now
                    last = now
                    content.append(piece)
            if obj.get("usage") is not None:
                usage = obj["usage"]
    end = time.perf_counter()
    prompt_tokens = None if usage is None else usage.get("prompt_tokens")
    completion_tokens = None if usage is None else usage.get("completion_tokens")
    wall_s = end - st
    if first is None:
        elapsed_s = wall_s
        ttft_s = None
        tpot_ms = None
        decode_tps = None
    else:
        elapsed_s = last - st
        ttft_s = first - st
        if completion_tokens and completion_tokens > 1:
            tpot_ms = (last - first) / (completion_tokens - 1) * 1000.0
            decode_tps = 1000.0 / tpot_ms if tpot_ms > 0 else None
        else:
            tpot_ms = None
            decode_tps = None
    prefill_tps = (
        prompt_tokens / ttft_s
        if prompt_tokens and ttft_s and ttft_s > 0
        else None
    )
    e2e_tps = (
        completion_tokens / elapsed_s
        if completion_tokens and elapsed_s > 0
        else None
    )
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_ms": ttft_s * 1000.0 if ttft_s is not None else None,
        "tpot_ms": tpot_ms,
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
        "e2e_tps": e2e_tps,
        "wall_s": wall_s,
        "chunks": chunks,
        "text_head": "".join(content)[:200],
    }


def _run_prompt(
    *,
    base_url: str,
    model_name: str,
    prompt_row: dict[str, str],
    max_completion_tokens: int,
    enable_thinking: bool,
    scenario_name: str,
) -> dict[str, Any]:
    before = _fetch_spec_decode_metrics(base_url)
    result = _stream_request(
        base_url=base_url,
        model_name=model_name,
        prompt=prompt_row["prompt"],
        max_completion_tokens=max_completion_tokens,
        enable_thinking=enable_thinking,
    )
    after = _fetch_spec_decode_metrics(base_url)

    delta_drafts = after[0] - before[0]
    delta_draft_tokens = after[1] - before[1]
    delta_accepted = after[2] - before[2]
    if scenario_name == "dflash" and delta_draft_tokens > 0:
        acceptance_rate = delta_accepted / delta_draft_tokens * 100.0
        acceptance_length = 1.0 + (delta_accepted / delta_drafts if delta_drafts > 0 else 0.0)
        per_pos = _format_per_pos_rate(before[3], after[3], delta_drafts)
    else:
        acceptance_rate = None
        acceptance_length = None
        per_pos = []

    return {
        "prompt_id": prompt_row["id"],
        "category": prompt_row["category"],
        "prompt": prompt_row["prompt"],
        **result,
        "num_drafts": delta_drafts,
        "draft_tokens": delta_draft_tokens,
        "accepted_tokens": delta_accepted,
        "acceptance_rate": acceptance_rate,
        "acceptance_length": acceptance_length,
        "per_position_acceptance_rates": per_pos,
    }


def _build_basic_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    total_drafts = sum(int(r.get("num_drafts", 0) or 0) for r in results)
    total_draft_tokens = sum(int(r.get("draft_tokens", 0) or 0) for r in results)
    total_accepted_tokens = sum(int(r.get("accepted_tokens", 0) or 0) for r in results)
    return {
        "num_prompts": len(results),
        "median_prompt_tokens": _median([r.get("prompt_tokens") for r in results]),
        "median_completion_tokens": _median([r.get("completion_tokens") for r in results]),
        "median_ttft_ms": _median([r.get("ttft_ms") for r in results]),
        "median_tpot_ms": _median([r.get("tpot_ms") for r in results]),
        "median_prefill_tps": _median([r.get("prefill_tps") for r in results]),
        "median_decode_tps": _median([r.get("decode_tps") for r in results]),
        "median_e2e_tps": _median([r.get("e2e_tps") for r in results]),
        "num_drafts": total_drafts,
        "draft_tokens": total_draft_tokens,
        "accepted_tokens": total_accepted_tokens,
        "acceptance_rate": (
            total_accepted_tokens / total_draft_tokens * 100.0
            if total_draft_tokens > 0
            else None
        ),
        "acceptance_length": (
            1.0 + total_accepted_tokens / total_drafts
            if total_drafts > 0
            else None
        ),
    }


def _summarize_prompt_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary = _build_basic_summary(results)
    categories = sorted({r["category"] for r in results})
    by_category: dict[str, Any] = {}
    for category in categories:
        by_category[category] = _build_basic_summary(
            [r for r in results if r["category"] == category]
        )
    summary["by_category"] = by_category
    return summary


def _diagnose_categories(
    dflash_by_preset: dict[str, list[dict[str, Any]]],
    threshold_rate_pp: float,
    threshold_len: float,
) -> dict[str, Any]:
    if not {
        "v100_tp4_current",
        "v100_tp1_current",
        "rtx3090_tp1_oracle",
    }.issubset(dflash_by_preset):
        return {}

    tp4 = _summarize_prompt_results(dflash_by_preset["v100_tp4_current"])["by_category"]
    tp1 = _summarize_prompt_results(dflash_by_preset["v100_tp1_current"])["by_category"]
    rtx = _summarize_prompt_results(dflash_by_preset["rtx3090_tp1_oracle"])["by_category"]

    out: dict[str, Any] = {}
    for category in sorted(tp4):
        tp4_rate = tp4[category].get("acceptance_rate")
        tp1_rate = tp1.get(category, {}).get("acceptance_rate")
        rtx_rate = rtx.get(category, {}).get("acceptance_rate")
        tp4_len = tp4[category].get("acceptance_length")
        tp1_len = tp1.get(category, {}).get("acceptance_length")
        rtx_len = rtx.get(category, {}).get("acceptance_length")
        diagnosis = "task_intrinsic"
        if (
            rtx_rate is not None
            and tp4_rate is not None
            and (rtx_rate - tp4_rate >= threshold_rate_pp)
        ) or (
            rtx_len is not None
            and tp4_len is not None
            and (rtx_len - tp4_len >= threshold_len)
        ):
            diagnosis = "v100_only"
        elif (
            tp1_rate is not None
            and tp4_rate is not None
            and (tp1_rate - tp4_rate >= threshold_rate_pp)
        ) or (
            tp1_len is not None
            and tp4_len is not None
            and (tp1_len - tp4_len >= threshold_len)
        ):
            diagnosis = "tp4_only"

        out[category] = {
            "v100_tp4_acceptance_rate": tp4_rate,
            "v100_tp4_acceptance_length": tp4_len,
            "v100_tp1_acceptance_rate": tp1_rate,
            "v100_tp1_acceptance_length": tp1_len,
            "rtx3090_tp1_acceptance_rate": rtx_rate,
            "rtx3090_tp1_acceptance_length": rtx_len,
            "diagnosis": diagnosis,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=("natural_zh", "spec_bench"),
        default="natural_zh",
    )
    parser.add_argument(
        "--presets",
        default="v100_tp4_current,v100_tp1_current,rtx3090_tp1_oracle",
    )
    parser.add_argument("--scenarios", default="dflash")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--draft-model", default=str(DEFAULT_DRAFT_MODEL))
    parser.add_argument("--num-speculative-tokens", type=int, default=16)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument(
        "--quantization",
        choices=("awq", "none"),
        default="awq",
    )
    parser.add_argument("--natural-suite-path", default=str(DEFAULT_NATURAL_SUITE))
    parser.add_argument("--spec-bench-path", default=str(DEFAULT_SPEC_BENCH_CACHE))
    parser.add_argument("--categories", default="")
    parser.add_argument("--max-prompts-per-category", type=int, default=8)
    parser.add_argument("--max-completion-tokens", type=int, default=512)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=4608)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4608)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--ready-timeout", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        default="bench_results/qwen35_9b_dflash_acceptance_matrix.json",
    )
    args = parser.parse_args()

    selected_presets = [x for x in args.presets.split(",") if x.strip()]
    selected_scenarios = [x for x in args.scenarios.split(",") if x.strip()]
    categories = {x for x in args.categories.split(",") if x.strip()} or None
    scenarios = _build_scenarios(
        draft_model=Path(args.draft_model),
        num_speculative_tokens=args.num_speculative_tokens,
    )

    missing_presets = [x for x in selected_presets if x not in PRESETS]
    missing_scenarios = [x for x in selected_scenarios if x not in scenarios]
    if missing_presets:
        raise ValueError(f"unknown presets: {missing_presets}")
    if missing_scenarios:
        raise ValueError(f"unknown scenarios: {missing_scenarios}")

    if args.suite == "natural_zh":
        prompts = _load_natural_suite(Path(args.natural_suite_path), categories)
    else:
        prompts = _load_spec_bench_suite(
            Path(args.spec_bench_path),
            categories,
            args.max_prompts_per_category,
            args.seed,
        )
    if not prompts:
        raise ValueError("no prompts selected")

    out_root = Path(args.out)
    if out_root.suffix:
        out_root.parent.mkdir(parents=True, exist_ok=True)
        out_path = out_root
    else:
        out_root.mkdir(parents=True, exist_ok=True)
        out_path = out_root / "result.json"

    cache_dir = out_path.parent / "server_cache"
    dflash_by_preset: dict[str, list[dict[str, Any]]] = {}
    results: list[dict[str, Any]] = []

    for preset_name in selected_presets:
        preset = PRESETS[preset_name]
        for scenario_name in selected_scenarios:
            scenario = scenarios[scenario_name]
            proc = None
            base_url = ""
            log_path = None
            try:
                proc, base_url, log_path = _start_server(
                    preset=preset,
                    scenario=scenario,
                    model=Path(args.model),
                    draft_model=Path(args.draft_model),
                    dtype=args.dtype,
                    quantization=args.quantization,
                    max_model_len=args.max_model_len,
                    max_num_batched_tokens=args.max_num_batched_tokens,
                    max_num_seqs=args.max_num_seqs,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    timeout_s=args.ready_timeout,
                    cache_root=cache_dir / f"{preset.name}_{scenario.name}",
                )

                model_name = f"{preset.name}-{scenario.name}"
                prompt_results = []

                # Warmup with the first prompt using a short decode.
                _run_prompt(
                    base_url=base_url,
                    model_name=model_name,
                    prompt_row=prompts[0],
                    max_completion_tokens=min(32, args.max_completion_tokens),
                    enable_thinking=args.enable_thinking,
                    scenario_name=scenario.name,
                )

                for prompt_row in prompts:
                    prompt_results.append(
                        _run_prompt(
                            base_url=base_url,
                            model_name=model_name,
                            prompt_row=prompt_row,
                            max_completion_tokens=args.max_completion_tokens,
                            enable_thinking=args.enable_thinking,
                            scenario_name=scenario.name,
                        )
                    )

                summary = _summarize_prompt_results(prompt_results)
                payload = {
                    "preset": preset.name,
                    "scenario": scenario.name,
                    "suite": args.suite,
                    "enable_thinking": args.enable_thinking,
                    "repo_root": str(preset.repo_root),
                    "python_executable": preset.python_executable,
                    "gpu_ids": preset.gpu_ids,
                    "tensor_parallel_size": preset.tensor_parallel_size,
                    "base_url": base_url,
                    "log_path": str(log_path),
                    "summary": summary,
                    "prompts": prompt_results,
                }
                results.append(payload)
                if scenario.name == "dflash":
                    dflash_by_preset[preset.name] = prompt_results
            except Exception as exc:  # noqa: BLE001
                results.append(
                    {
                        "preset": preset.name,
                        "scenario": scenario.name,
                        "suite": args.suite,
                        "enable_thinking": args.enable_thinking,
                        "repo_root": str(preset.repo_root),
                        "python_executable": preset.python_executable,
                        "gpu_ids": preset.gpu_ids,
                        "tensor_parallel_size": preset.tensor_parallel_size,
                        "base_url": base_url,
                        "log_path": str(log_path) if log_path is not None else None,
                        "error": str(exc),
                    }
                )
            finally:
                _stop_server(proc)
                time.sleep(3)

    final_payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "suite": args.suite,
        "enable_thinking": args.enable_thinking,
        "model": args.model,
        "draft_model": args.draft_model,
        "dtype": args.dtype,
        "quantization": args.quantization,
        "max_completion_tokens": args.max_completion_tokens,
        "results": results,
        "cross_env_diagnosis": _diagnose_categories(
            dflash_by_preset,
            threshold_rate_pp=10.0,
            threshold_len=2.0,
        ),
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(final_payload, f, ensure_ascii=False, indent=2)

    print(json.dumps({"saved_to": str(out_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
