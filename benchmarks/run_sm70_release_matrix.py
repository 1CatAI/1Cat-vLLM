# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the 1Cat SM70 release quality/speed matrix.

The matrix is intentionally explicit because it is used as the final release
gate for V100/SM70 builds:

* Qwen3.6 27B/35B AWQ/FP8 and Gemma4 31B AWQ/NVFP4.
* Marlin and TurboMind quantization backends.
* TP2 and TP4.
* KV cache modes: auto, fp8_e5m2, turboquant_4bit_nc.
* Native MTP4 for Qwen3.6 checkpoints that expose MTP layers.
* DFlash16 for the Qwen3.6 35B checkpoints with the local Qwen3.5 35B draft.
* Prefix caching enabled for every case.
* Speed, fixed long-form quality prompts, public prompt datasets
  (HumanEval/MBPP/IFEval), GSM8K exact-match, and LongBench quality scoring.

Every primary case is treated as a hard gate. The runner still continues after a
failure so a long overnight run produces a complete failure table.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import dataclasses
import gzip
import hashlib
import json
import os
import queue
import re
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

REPO_ROOT = Path(__file__).resolve().parents[1]
LONG_BENCH_ROOT = REPO_ROOT / "third_party" / "LongBench" / "LongBench"
DEFAULT_DATA_DIR = REPO_ROOT / "benchmark-data" / "longbench" / "data"
DEFAULT_GSM8K_CACHE_DIR = Path("/tmp/sm70_release_datasets/gsm8k")
DEFAULT_PUBLIC_DATASET_CACHE_DIR = Path("/tmp/sm70_release_datasets/public_quality")
DFLASH_35B_DRAFT = "/home/ymzx/models/Qwen3.5-35B-A3B-DFlash"
INVALID_GSM8K_ANSWER = -9999999

NO_CHAT_DATASETS = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}
DEFAULT_LONG_BENCH_DATASETS = (
    "qasper",
    "narrativeqa",
    "musique",
    "hotpotqa",
    "multifieldqa_zh",
    "dureader",
    "gov_report",
    "qmsum",
    "multi_news",
    "vcsum",
    "lcc",
    "repobench-p",
)
DEFAULT_KV_CACHE_DTYPES = ("auto", "fp8_e5m2", "turboquant_4bit_nc")
DEFAULT_BACKENDS = ("marlin", "turbomind")
DEFAULT_TPS = (2, 4)

PUBLIC_QUALITY_DATASETS = {
    "humaneval": {
        "url": "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz",
        "filename": "HumanEval.jsonl.gz",
    },
    "mbpp": {
        "url": "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/sanitized-mbpp.json",
        "filename": "sanitized-mbpp.json",
    },
    "ifeval": {
        "url": "https://raw.githubusercontent.com/google-research/google-research/master/instruction_following_eval/data/input_data.jsonl",
        "filename": "ifeval_input_data.jsonl",
    },
}
DEFAULT_PUBLIC_QUALITY_DATASETS = ("humaneval", "mbpp", "ifeval")

QUALITY_PROMPTS = (
    {
        "id": "macos_6k_code",
        "max_tokens": 32768,
        "content": (
            "帮我用 HTML、CSS、JavaScript 做一个 macOS 桌面模拟器，"
            "功能尽可能完整：菜单栏、Dock、可拖动窗口、Finder、设置、"
            "终端、文件管理、窗口最小化/关闭、时钟和主题切换。"
            "请直接给出可以运行的完整代码，不要重复段落，不要输出乱码。"
        ),
    },
    {
        "id": "python_service",
        "max_tokens": 2048,
        "content": (
            "请用 Python 写一个可运行的 FastAPI 任务队列服务，包含任务提交、"
            "状态查询、取消、后台执行、SQLite 持久化、错误重试、分页查询、"
            "单元测试示例和启动命令。输出完整代码和必要说明。"
        ),
    },
    {
        "id": "database_design",
        "max_tokens": 2048,
        "content": (
            "设计一个多人协作代码审查平台的数据库、接口和权限模型。"
            "需要覆盖用户、组织、仓库、PR、评论、审计日志、通知、索引、"
            "幂等、限流和失败恢复。用清晰标题输出，不要重复。"
        ),
    },
    {
        "id": "long_chinese_report",
        "max_tokens": 2048,
        "content": (
            "写一份中文技术评估报告，比较 AWQ、FP8、FP8 KV cache、"
            "TurboQuant KV cache 和 speculative decoding 在 V100 上的收益、"
            "风险、适用场景和发布验证方法。要求结构完整，避免重复段落。"
        ),
    },
)

CODING_SPEED_PROMPT_PREFIX = """You are a senior backend engineer working inside
a production Python repository.

Task: implement a complete FastAPI task orchestration service for code
generation jobs. The implementation must include:
- REST endpoints to create jobs, cancel jobs, query status, stream logs, and
  list paginated history.
- SQLite persistence with migrations, indexes, idempotency keys, retries, and
  crash recovery.
- A background worker pool with bounded concurrency, exponential backoff,
  cancellation, and graceful shutdown.
- JWT authentication, per-user rate limits, audit logging, and structured JSON logs.
- A small TypeScript client, pytest coverage, and a Dockerfile.

Return runnable code. Prefer clear module boundaries and include every file needed.

Existing repository context follows.
"""

CODING_SPEED_PROMPT_FILLER = """

File: app/models.py
```python
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Literal

class JobState(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"

@dataclass
class JobRecord:
    id: str
    user_id: str
    prompt: str
    state: JobState
    attempts: int
    max_attempts: int
    created_at: datetime
    updated_at: datetime
    result: str | None = None
    error: str | None = None
```

File: app/repository.py
```python
class JobRepository:
    def create_job(
        self, *, user_id: str, prompt: str, idempotency_key: str | None
    ) -> JobRecord:
        ...
    def claim_next_job(self, *, worker_id: str) -> JobRecord | None:
        ...
    def append_log(
        self, *, job_id: str, level: str, message: str, payload: dict[str, Any]
    ) -> None:
        ...
    def transition(
        self, *, job_id: str, from_state: JobState | None, to_state: JobState
    ) -> None:
        ...
```

File: app/worker.py
```python
class WorkerPool:
    async def start(self) -> None:
        ...
    async def stop(self) -> None:
        ...
    async def cancel(self, job_id: str) -> bool:
        ...
```

Design constraints:
- Never lose a queued job after process restart.
- A cancelled running job must stop streaming output within one second.
- API responses must be stable JSON objects with explicit error codes.
- Include tests for idempotency, retry behavior, cancellation, auth failures,
  and pagination.
- Keep the code readable enough for an on-call engineer to debug at 3 AM.
"""

CODING_SPEED_PROMPT_SUFFIX = """

Now write the complete implementation. Include code blocks for each file.
Do not summarize. Do not omit tests.
"""


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    key: str
    path: str
    quantization: str
    family: str
    has_mtp: bool = False
    supports_dflash_35b: bool = False
    dtype: str = "auto"


MODELS = (
    ModelSpec(
        key="qwen36-27b-awq",
        path="/home/ymzx/models/Qwen3.6-27B-AWQ",
        quantization="awq",
        family="qwen",
        has_mtp=True,
    ),
    ModelSpec(
        key="qwen36-27b-fp8",
        path="/home/ymzx/models/Qwen3.6-27B-FP8",
        quantization="fp8",
        family="qwen",
        has_mtp=True,
    ),
    ModelSpec(
        key="qwen36-35b-awq",
        path="/home/ymzx/models/Qwen3.6-35B-A3B-AWQ",
        quantization="awq",
        family="qwen",
        has_mtp=True,
        supports_dflash_35b=True,
    ),
    ModelSpec(
        key="qwen36-35b-fp8",
        path="/home/ymzx/models/Qwen3.6-35B-A3B-FP8",
        quantization="fp8",
        family="qwen",
        has_mtp=True,
        supports_dflash_35b=True,
    ),
    ModelSpec(
        key="gemma4-31b-awq",
        path="/home/ymzx/models/gemma-4-31B-it-AWQ",
        quantization="awq",
        family="gemma",
    ),
    ModelSpec(
        key="gemma4-31b-nvfp4",
        path="/home/ymzx/models/gemma-4-31B-it-NVFP4",
        quantization="compressed-tensors",
        family="gemma",
    ),
)


@dataclasses.dataclass(frozen=True)
class CaseSpec:
    name: str
    backend: str
    tp: int
    model: ModelSpec
    kv_cache_dtype: str
    mode: str
    num_speculative_tokens: int = 0
    draft_model: str | None = None

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "backend": self.backend,
            "tp": self.tp,
            "model": dataclasses.asdict(self.model),
            "kv_cache_dtype": self.kv_cache_dtype,
            "mode": self.mode,
            "num_speculative_tokens": self.num_speculative_tokens,
            "draft_model": self.draft_model,
        }

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> CaseSpec:
        return cls(
            name=data["name"],
            backend=data["backend"],
            tp=int(data["tp"]),
            model=ModelSpec(**data["model"]),
            kv_cache_dtype=data["kv_cache_dtype"],
            mode=data["mode"],
            num_speculative_tokens=int(data.get("num_speculative_tokens", 0)),
            draft_model=data.get("draft_model"),
        )


def _slug(value: str) -> str:
    return value.replace("_", "").replace("-", "")


def _make_case_name(
    *,
    backend: str,
    tp: int,
    model: ModelSpec,
    kv_cache_dtype: str,
    mode: str,
) -> str:
    return (
        f"{backend}-tp{tp}-{model.key}-kv-{_slug(kv_cache_dtype)}-{mode}"
    )


def _make_cases(
    *,
    backends: tuple[str, ...],
    tps: tuple[int, ...],
    kv_cache_dtypes: tuple[str, ...],
    include_turboquant_mtp: bool = False,
    include_turboquant_dflash: bool = False,
    include_fp8_model_fp8_kv_mtp: bool = False,
    include_gemma_turboquant: bool = False,
    include_gemma_nvfp4_fp8_kv: bool = False,
) -> list[CaseSpec]:
    cases: list[CaseSpec] = []
    for backend in backends:
        for tp in tps:
            for model in MODELS:
                for kv_cache_dtype in kv_cache_dtypes:
                    gemma_turboquant = (
                        model.family == "gemma"
                        and kv_cache_dtype.startswith("turboquant")
                    )
                    gemma_nvfp4_fp8_kv = (
                        model.key == "gemma4-31b-nvfp4"
                        and kv_cache_dtype == "fp8_e5m2"
                    )
                    if (gemma_turboquant and not include_gemma_turboquant):
                        continue
                    if (gemma_nvfp4_fp8_kv
                            and not include_gemma_nvfp4_fp8_kv):
                        continue
                    cases.append(
                        CaseSpec(
                            name=_make_case_name(
                                backend=backend,
                                tp=tp,
                                model=model,
                                kv_cache_dtype=kv_cache_dtype,
                                mode="nospec",
                            ),
                            backend=backend,
                            tp=tp,
                            model=model,
                            kv_cache_dtype=kv_cache_dtype,
                            mode="nospec",
                        )
                    )
                    # SM70 TurboQuant target KV currently forces the MTP
                    # drafter onto TurboQuant KV too. That route is useful for
                    # diagnostics, but is not release-safe by default.
                    # FP8-weight Qwen MTP with fp8_e5m2 KV hits an unsupported
                    # static_scaled_fp8_quant e5m2 kernel during drafter
                    # warmup; keep that as an explicit diagnostic route too.
                    fp8_model_fp8_kv_mtp = (
                        model.quantization == "fp8"
                        and kv_cache_dtype == "fp8_e5m2"
                    )
                    if (model.has_mtp
                            and (include_turboquant_mtp
                                 or not kv_cache_dtype.startswith("turboquant"))
                            and (include_fp8_model_fp8_kv_mtp
                                 or not fp8_model_fp8_kv_mtp)):
                        cases.append(
                            CaseSpec(
                                name=_make_case_name(
                                    backend=backend,
                                    tp=tp,
                                    model=model,
                                    kv_cache_dtype=kv_cache_dtype,
                                    mode="mtp4",
                                ),
                                backend=backend,
                                tp=tp,
                                model=model,
                                kv_cache_dtype=kv_cache_dtype,
                                mode="mtp4",
                                num_speculative_tokens=4,
                            )
                        )
                    # DFlash draft attention is non-causal; TurboQuant KV
                    # currently has no non-causal attention backend on SM70.
                    if (model.supports_dflash_35b
                            and (include_turboquant_dflash
                                 or not kv_cache_dtype.startswith("turboquant"))):
                        cases.append(
                            CaseSpec(
                                name=_make_case_name(
                                    backend=backend,
                                    tp=tp,
                                    model=model,
                                    kv_cache_dtype=kv_cache_dtype,
                                    mode="dflash16",
                                ),
                                backend=backend,
                                tp=tp,
                                model=model,
                                kv_cache_dtype=kv_cache_dtype,
                                mode="dflash16",
                                num_speculative_tokens=16,
                                draft_model=DFLASH_35B_DRAFT,
                            )
                        )
    return cases


def _parse_tuple_arg(values: list[str] | None,
                     default: tuple[str, ...]) -> tuple[str, ...]:
    if not values:
        return default
    out: list[str] = []
    for value in values:
        out.extend(item.strip() for item in value.split(",") if item.strip())
    return tuple(out)


def _parse_int_tuple_arg(values: list[str] | None,
                         default: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(int(value) for value in _parse_tuple_arg(values, ())) or default


def _filter_cases(cases: list[CaseSpec], args: argparse.Namespace) -> list[CaseSpec]:
    only_cases = set(_parse_tuple_arg(args.only_case, ()))
    only_models = set(_parse_tuple_arg(args.model_key, ()))
    only_modes = set(_parse_tuple_arg(args.mode, ()))
    only_backends = set(_parse_tuple_arg(args.backend_filter, ()))
    only_kv = set(_parse_tuple_arg(args.kv_cache_dtype_filter, ()))
    only_tps = set(_parse_int_tuple_arg(args.tp_filter, ()))

    filtered = []
    for case in cases:
        if only_cases and case.name not in only_cases:
            continue
        if only_models and case.model.key not in only_models:
            continue
        if only_modes and case.mode not in only_modes:
            continue
        if only_backends and case.backend not in only_backends:
            continue
        if only_kv and case.kv_cache_dtype not in only_kv:
            continue
        if only_tps and case.tp not in only_tps:
            continue
        filtered.append(case)
    if args.limit_cases is not None:
        filtered = filtered[:args.limit_cases]
    return filtered


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _sha256_ids(token_ids: list[int]) -> str:
    raw = ",".join(str(token_id) for token_id in token_ids).encode()
    return hashlib.sha256(raw).hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_source_index"] = idx
            rows.append(row)
    return rows


def _download_if_missing(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path
    print(f"[dataset] downloading {url} -> {path}", flush=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    if tmp_path.exists():
        tmp_path.unlink()
    urlretrieve(url, tmp_path)
    if path.exists() and path.stat().st_size > 0:
        tmp_path.unlink(missing_ok=True)
        return path
    tmp_path.replace(path)
    return path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip() or line.startswith("#"):
                continue
            row = json.loads(line)
            row["_source_index"] = idx
            rows.append(row)
    return rows


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip() or line.startswith("#"):
                continue
            row = json.loads(line)
            row["_source_index"] = idx
            rows.append(row)
    return rows


def _load_generation_config(model: str) -> dict[str, Any]:
    path = Path(model) / "generation_config.json"
    if not path.exists():
        return {"temperature": 1.0, "top_p": 1.0, "top_k": -1}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "temperature": float(data.get("temperature", 1.0)),
        "top_p": float(data.get("top_p", 1.0)),
        "top_k": int(data.get("top_k", -1)),
    }


def _select_rows(
    rows: list[dict[str, Any]],
    *,
    min_length: int,
    limit: int,
) -> list[dict[str, Any]]:
    selected = [row for row in rows if int(row.get("length", 0)) >= min_length]
    if not selected:
        selected = list(rows)
    if limit <= 0 or len(selected) <= limit:
        return selected
    if limit == 1:
        return [selected[len(selected) // 2]]
    last = len(selected) - 1
    return [selected[round(i * last / (limit - 1))] for i in range(limit)]


def _truncate_middle(tokenizer: Any, prompt: str, max_input_tokens: int) -> str:
    token_ids = tokenizer(
        prompt,
        truncation=False,
        add_special_tokens=False,
    ).input_ids
    if len(token_ids) <= max_input_tokens:
        return prompt
    half = max_input_tokens // 2
    left = tokenizer.decode(token_ids[:half], skip_special_tokens=True)
    right = tokenizer.decode(token_ids[-half:], skip_special_tokens=True)
    return left + right


def _apply_chat_template(
    tokenizer: Any,
    content: str,
    *,
    enable_thinking: bool,
) -> str:
    messages = [{"role": "user", "content": content}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _build_longbench_prompt(
    tokenizer: Any,
    *,
    dataset: str,
    row: dict[str, Any],
    prompt_format: str,
    max_input_tokens: int,
    chat_template: str,
) -> tuple[str, int]:
    raw_prompt = prompt_format.format(**row)
    target_tokens = max_input_tokens
    for _ in range(8):
        prompt = _truncate_middle(tokenizer, raw_prompt, target_tokens)
        if chat_template == "always" or (
            chat_template == "official" and dataset not in NO_CHAT_DATASETS
        ):
            prompt = _apply_chat_template(
                tokenizer,
                prompt,
                enable_thinking=False,
            )
        final_ids = tokenizer(
            prompt,
            truncation=False,
            add_special_tokens=False,
        ).input_ids
        if len(final_ids) <= max_input_tokens:
            return prompt, len(final_ids)
        target_tokens = max(
            128,
            target_tokens - (len(final_ids) - max_input_tokens) - 256,
        )
    raise ValueError(
        f"failed to fit LongBench prompt dataset={dataset} "
        f"source_index={row.get('_source_index')} under {max_input_tokens} tokens"
    )


def _score_longbench_one(dataset: str, prediction: str,
                         row: dict[str, Any]) -> float:
    sys.path.insert(0, str(LONG_BENCH_ROOT.resolve()))
    from eval import dataset2metric  # type: ignore

    score = 0.0
    pred = prediction
    if dataset in {"trec", "triviaqa", "samsum", "lsht"}:
        pred = pred.lstrip("\n").split("\n")[0]
    for answer in row["answers"]:
        score = max(
            score,
            dataset2metric[dataset](
                pred,
                answer,
                all_classes=row.get("all_classes", []),
            ),
        )
    return float(score)


def _gsm8k_cache_files(cache_dir: Path) -> tuple[Path, Path]:
    train_url = (
        "https://raw.githubusercontent.com/openai/grade-school-math/master/"
        "grade_school_math/data/train.jsonl"
    )
    test_url = (
        "https://raw.githubusercontent.com/openai/grade-school-math/master/"
        "grade_school_math/data/test.jsonl"
    )
    train_file = _download_if_missing(train_url, cache_dir / "train.jsonl")
    test_file = _download_if_missing(test_url, cache_dir / "test.jsonl")
    return train_file, test_file


def _gsm8k_answer_value(answer_str: str) -> int:
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if not numbers:
        return INVALID_GSM8K_ANSWER
    try:
        return int(ast.literal_eval(numbers[-1]))
    except (SyntaxError, ValueError):
        return INVALID_GSM8K_ANSWER


def _build_gsm8k_prompts(
    *,
    cache_dir: Path,
    num_questions: int,
    num_shots: int,
) -> tuple[list[str], list[int], list[dict[str, Any]]]:
    train_file, test_file = _gsm8k_cache_files(cache_dir)
    train_data = _read_jsonl(train_file)
    test_data = _read_jsonl(test_file)
    num_questions = min(num_questions, len(test_data))
    few_shot = ""
    for i in range(num_shots):
        few_shot += (
            f"Question: {train_data[i]['question']}\n"
            f"Answer: {train_data[i]['answer']}\n\n"
        )
    prompts = []
    labels = []
    rows = []
    for row in test_data[:num_questions]:
        prompts.append(few_shot + f"Question: {row['question']}\nAnswer:")
        labels.append(_gsm8k_answer_value(row["answer"]))
        rows.append(row)
    return prompts, labels, rows


def _load_public_quality_rows(
    *,
    dataset: str,
    cache_dir: Path,
    limit: int,
) -> list[dict[str, Any]]:
    spec = PUBLIC_QUALITY_DATASETS[dataset]
    path = _download_if_missing(spec["url"], cache_dir / spec["filename"])
    if dataset == "humaneval":
        rows = _read_jsonl_gz(path)
    elif dataset == "mbpp":
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = []
        for idx, row in enumerate(data):
            row = dict(row)
            row["_source_index"] = idx
            rows.append(row)
    elif dataset == "ifeval":
        rows = _read_jsonl(path)
    else:
        raise ValueError(f"unknown public quality dataset: {dataset}")
    return _select_rows(rows, min_length=0, limit=limit)


def _public_quality_prompt(dataset: str, row: dict[str, Any]) -> str:
    if dataset == "humaneval":
        return (
            "Complete the following Python function. Return runnable Python "
            "code only.\n\n"
            f"{row['prompt']}"
        )
    if dataset == "mbpp":
        tests = "\n".join(row.get("test_list") or [])
        return (
            "Write a Python function that satisfies the task and tests. "
            "Return runnable Python code only.\n\n"
            f"Task: {row['prompt']}\n\nTests:\n{tests}"
        )
    if dataset == "ifeval":
        return row["prompt"]
    raise ValueError(f"unknown public quality dataset: {dataset}")


def _make_exact_prompt_token_ids_from_parts(
    tokenizer: Any,
    *,
    target_len: int,
    prefix: str,
    filler: str,
    suffix: str = "",
) -> list[int]:
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    filler_ids = tokenizer.encode(filler, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False) if suffix else []
    if not prefix_ids and not filler_ids and not suffix_ids:
        raise RuntimeError("benchmark prompt encoded to no tokens")
    if target_len <= len(prefix_ids) + len(suffix_ids):
        return (prefix_ids + suffix_ids)[:target_len]
    token_ids = list(prefix_ids)
    remaining = target_len - len(prefix_ids) - len(suffix_ids)
    if not filler_ids:
        filler_ids = prefix_ids or suffix_ids
    while len(token_ids) < len(prefix_ids) + remaining:
        token_ids.extend(filler_ids)
    token_ids = token_ids[:len(prefix_ids) + remaining]
    token_ids.extend(suffix_ids)
    return token_ids[:target_len]


def _make_exact_prompt_token_ids(tokenizer: Any, target_len: int) -> list[int]:
    filler = (
        "SM70 V100 release benchmark prompt. Continue with stable, coherent "
        "technical prose about CUDA kernels, attention backends, KV cache, "
        "speculative decoding, quantization, and long-context serving. "
    )
    return _make_exact_prompt_token_ids_from_parts(
        tokenizer,
        target_len=target_len,
        prefix=filler,
        filler=filler,
    )


def _make_coding_prompt_token_ids(tokenizer: Any, target_len: int) -> list[int]:
    return _make_exact_prompt_token_ids_from_parts(
        tokenizer,
        target_len=target_len,
        prefix=CODING_SPEED_PROMPT_PREFIX,
        filler=CODING_SPEED_PROMPT_FILLER,
        suffix=CODING_SPEED_PROMPT_SUFFIX,
    )


def _make_speed_prompt_token_ids(tokenizer: Any,
                                 args: argparse.Namespace) -> list[int]:
    if args.speed_prompt == "synthetic":
        return _make_exact_prompt_token_ids(tokenizer, args.speed_input_len)
    if args.speed_prompt == "coding":
        return _make_coding_prompt_token_ids(tokenizer, args.speed_input_len)
    raise ValueError(f"unknown speed prompt: {args.speed_prompt}")


def _safe_delta(end: float | None, start: float | None) -> float | None:
    if end is None or start is None or end <= 0.0 or start <= 0.0:
        return None
    return end - start


def _request_metrics(output: Any, elapsed_s: float | None,
                     prompt_tokens: int) -> dict[str, Any]:
    metrics = getattr(output, "metrics", None)
    completion = output.outputs[0]
    output_tokens = len(completion.token_ids)
    if metrics is None:
        return {
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "elapsed_s": elapsed_s,
        }
    first_token_ts = getattr(metrics, "first_token_ts", None)
    scheduled_ts = getattr(metrics, "scheduled_ts", None)
    last_token_ts = getattr(metrics, "last_token_ts", None)
    prefill_time = _safe_delta(first_token_ts, scheduled_ts)
    decode_time = _safe_delta(last_token_ts, first_token_ts)
    steady_tokens = max(output_tokens - 1, 0)
    ttft_s = getattr(metrics, "first_token_latency", None) or prefill_time
    return {
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "elapsed_s": elapsed_s,
        "prefill_time_s": prefill_time,
        "decode_time_s": decode_time,
        "prefill_tps": prompt_tokens / prefill_time if prefill_time else None,
        "steady_decode_tokens": steady_tokens,
        "steady_decode_tps": (
            steady_tokens / decode_time
            if decode_time and steady_tokens > 0
            else None
        ),
        "first_token_latency_s": getattr(metrics, "first_token_latency", None),
        "ttft_s": ttft_s,
        "finish_num_generation_tokens": getattr(
            metrics, "num_generation_tokens", None
        ),
        "is_corrupted": getattr(metrics, "is_corrupted", None),
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


def _quality_metrics(
    text: str,
    token_ids: list[int],
    *,
    min_tokens: int = 64,
) -> dict[str, Any]:
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
    if metrics["tokens"] < min_tokens:
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
    return (
        output.text,
        list(output.token_ids),
        output.finish_reason,
        output.stop_reason,
    )


def _sync_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    if value.startswith(("{", "[")):
        return json.loads(value)
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _parse_engine_args(values: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected KEY=VALUE for --engine-arg, got {value!r}")
        key, raw = value.split("=", 1)
        parsed[key.replace("-", "_")] = _parse_scalar(raw)
    return parsed


def _speculative_config(case: CaseSpec) -> dict[str, Any] | None:
    if case.mode == "mtp4":
        draft_attention_backend = (
            "TURBOQUANT"
            if case.kv_cache_dtype.startswith("turboquant")
            else "TRITON_ATTN"
        )
        return {
            "method": "mtp",
            "num_speculative_tokens": case.num_speculative_tokens,
            "draft_sample_method": "probabilistic",
            "use_local_argmax_reduction": True,
            "attention_backend": draft_attention_backend,
        }
    if case.mode == "dflash16":
        if not case.draft_model:
            raise ValueError(f"{case.name} missing DFlash draft model")
        return {
            "method": "dflash",
            "model": case.draft_model,
            "num_speculative_tokens": case.num_speculative_tokens,
            "draft_sample_method": "probabilistic",
        }
    return None


def _make_llm_kwargs(
    case: CaseSpec,
    args: argparse.Namespace,
) -> dict[str, Any]:
    attention_backend = args.attention_backend
    if case.model.family == "gemma":
        attention_backend = "TRITON_ATTN"
    if case.kv_cache_dtype.startswith("turboquant"):
        attention_backend = "TURBOQUANT"
    llm_kwargs: dict[str, Any] = {
        "model": case.model.path,
        "trust_remote_code": True,
        "tensor_parallel_size": case.tp,
        "dtype": case.model.dtype,
        "quantization": case.model.quantization,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "attention_backend": attention_backend,
        "disable_log_stats": False,
        "enable_prefix_caching": True,
        "seed": args.seed,
    }
    if case.kv_cache_dtype != "auto":
        llm_kwargs["kv_cache_dtype"] = case.kv_cache_dtype
    if case.model.family == "qwen":
        llm_kwargs["mamba_cache_mode"] = "align"
    spec_config = _speculative_config(case)
    if spec_config is not None:
        llm_kwargs["speculative_config"] = spec_config
    llm_kwargs.update(_parse_engine_args(args.engine_arg))
    return llm_kwargs


def _run_speed_phase(
    *,
    llm: Any,
    tokenizer: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from vllm import SamplingParams

    speed_prompt_ids = _make_speed_prompt_token_ids(tokenizer, args)
    sampling = SamplingParams(
        max_tokens=args.speed_output_len,
        min_tokens=args.speed_output_len,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        ignore_eos=True,
        skip_special_tokens=False,
    )
    _sync_cuda()
    start = time.perf_counter()
    outputs = llm.generate(
        [{"prompt_token_ids": speed_prompt_ids}],
        sampling,
        use_tqdm=False,
    )
    _sync_cuda()
    elapsed_s = time.perf_counter() - start
    text, token_ids, finish_reason, stop_reason = _output_record(outputs[0])
    metrics = _request_metrics(outputs[0], elapsed_s, len(speed_prompt_ids))
    return {
        "input_len": args.speed_input_len,
        "prompt_type": args.speed_prompt,
        "max_tokens": args.speed_output_len,
        "wall_seconds": elapsed_s,
        "finish_reason": finish_reason,
        "stop_reason": stop_reason,
        "output_tokens": len(token_ids),
        "text_hash": _sha256_text(text),
        "token_hash": _sha256_ids(token_ids),
        "request_metrics": metrics,
        "quality_metrics": _quality_metrics(text, token_ids),
        "preview": text[:1000],
        "tail": text[-1000:],
    }


def _run_prefix_probe(
    *,
    llm: Any,
    tokenizer: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from vllm import SamplingParams

    prompt_ids = _make_exact_prompt_token_ids(tokenizer, args.prefix_probe_input_len)
    sampling = SamplingParams(
        max_tokens=args.prefix_probe_output_len,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        skip_special_tokens=False,
    )
    records = []
    for idx in range(2):
        _sync_cuda()
        start = time.perf_counter()
        outputs = llm.generate(
            [{"prompt_token_ids": prompt_ids}],
            sampling,
            use_tqdm=False,
        )
        _sync_cuda()
        elapsed_s = time.perf_counter() - start
        text, token_ids, finish_reason, stop_reason = _output_record(outputs[0])
        records.append({
            "repeat": idx + 1,
            "finish_reason": finish_reason,
            "stop_reason": stop_reason,
            "metrics": _request_metrics(outputs[0], elapsed_s, len(prompt_ids)),
            "num_cached_tokens": getattr(outputs[0], "num_cached_tokens", None),
            "token_hash": _sha256_ids(token_ids),
            "text_hash": _sha256_text(text),
        })
    first_ttft = records[0]["metrics"].get("ttft_s")
    second_ttft = records[1]["metrics"].get("ttft_s")
    ttft_ratio = second_ttft / first_ttft if first_ttft and second_ttft else None
    token_hash_match = records[0]["token_hash"] == records[1]["token_hash"]
    text_hash_match = records[0]["text_hash"] == records[1]["text_hash"]
    second_cached_tokens = records[1]["num_cached_tokens"]
    failure_reasons = []
    if second_cached_tokens is None:
        failure_reasons.append("missing_num_cached_tokens")
    elif second_cached_tokens <= 0:
        failure_reasons.append("no_prefix_cache_hit")
    if ttft_ratio is None:
        failure_reasons.append("missing_ttft")
    elif ttft_ratio > args.prefix_probe_max_ttft_ratio:
        failure_reasons.append(
            f"ttft_ratio {ttft_ratio:.4f} > "
            f"{args.prefix_probe_max_ttft_ratio:.4f}"
        )
    if not token_hash_match:
        failure_reasons.append("token_hash_mismatch")
    if not text_hash_match:
        failure_reasons.append("text_hash_mismatch")
    return {
        "enabled": True,
        "input_len": args.prefix_probe_input_len,
        "output_len": args.prefix_probe_output_len,
        "records": records,
        "ttft_ratio_second_over_first": ttft_ratio,
        "max_ttft_ratio": args.prefix_probe_max_ttft_ratio,
        "second_num_cached_tokens": second_cached_tokens,
        "token_hash_match": token_hash_match,
        "text_hash_match": text_hash_match,
        "failure_reasons": failure_reasons,
        "passed": not failure_reasons,
    }


def _run_quality_phase(
    *,
    llm: Any,
    tokenizer: Any,
    case: CaseSpec,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from vllm import SamplingParams

    generation_config = _load_generation_config(case.model.path)
    selected_prompt_ids = set(_parse_tuple_arg(args.quality_prompt_id, ()))
    quality_prompts = [
        prompt for prompt in QUALITY_PROMPTS
        if not selected_prompt_ids or prompt["id"] in selected_prompt_ids
    ]
    missing = selected_prompt_ids - {prompt["id"] for prompt in quality_prompts}
    if missing:
        raise ValueError(f"unknown quality prompt ids: {sorted(missing)}")

    records = []
    for prompt in quality_prompts:
        chat_prompt = _apply_chat_template(
            tokenizer,
            prompt["content"],
            enable_thinking=args.enable_thinking,
        )
        sampling = SamplingParams(
            max_tokens=min(int(prompt["max_tokens"]), args.quality_max_tokens),
            temperature=generation_config["temperature"],
            top_p=generation_config["top_p"],
            top_k=generation_config["top_k"],
            seed=args.sampling_seed,
            skip_special_tokens=False,
        )
        for repeat_idx in range(args.quality_repeat):
            _sync_cuda()
            start = time.perf_counter()
            request = llm.generate([chat_prompt], sampling, use_tqdm=False)[0]
            _sync_cuda()
            elapsed_s = time.perf_counter() - start
            text, token_ids, finish_reason, stop_reason = _output_record(request)
            record = {
                "id": prompt["id"],
                "repeat": repeat_idx + 1,
                "finish_reason": finish_reason,
                "stop_reason": stop_reason,
                "prompt_tokens": len(request.prompt_token_ids or []),
                "request_metrics": _request_metrics(
                    request,
                    elapsed_s,
                    len(request.prompt_token_ids or []),
                ),
                "metrics": _quality_metrics(text, token_ids),
                "preview": text[:1200],
                "tail": text[-1200:],
            }
            if args.quality_save_full_output:
                record["text"] = text
                record["token_ids"] = token_ids
            records.append(record)
    return {
        "sampling": {
            **generation_config,
            "seed": args.sampling_seed,
            "ignore_eos": False,
            "enable_thinking": args.enable_thinking,
        },
        "records": records,
        "passed": all(record["metrics"]["passed"] for record in records),
    }


def _run_public_quality_phase(
    *,
    llm: Any,
    tokenizer: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from vllm import SamplingParams

    datasets = _parse_tuple_arg(
        args.public_quality_dataset,
        DEFAULT_PUBLIC_QUALITY_DATASETS,
    )
    records = []
    sampling = SamplingParams(
        max_tokens=args.public_quality_max_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        skip_special_tokens=False,
    )
    for dataset in datasets:
        if dataset not in PUBLIC_QUALITY_DATASETS:
            raise ValueError(f"unknown public quality dataset: {dataset}")
        rows = _load_public_quality_rows(
            dataset=dataset,
            cache_dir=args.public_dataset_cache_dir,
            limit=args.public_quality_limit,
        )
        for ordinal, row in enumerate(rows):
            content = _public_quality_prompt(dataset, row)
            prompt = _apply_chat_template(
                tokenizer,
                content,
                enable_thinking=False,
            )
            _sync_cuda()
            start = time.perf_counter()
            request = llm.generate([prompt], sampling, use_tqdm=False)[0]
            _sync_cuda()
            elapsed_s = time.perf_counter() - start
            text, token_ids, finish_reason, stop_reason = _output_record(request)
            metrics = _quality_metrics(text, token_ids, min_tokens=8)
            records.append({
                "dataset": dataset,
                "ordinal": ordinal,
                "source_index": row.get("_source_index"),
                "task_id": row.get("task_id") or row.get("key"),
                "finish_reason": finish_reason,
                "stop_reason": stop_reason,
                "prompt_tokens": len(request.prompt_token_ids or []),
                "request_metrics": _request_metrics(
                    request,
                    elapsed_s,
                    len(request.prompt_token_ids or []),
                ),
                "metrics": metrics,
                "preview": text[:1200],
                "tail": text[-1200:],
            })
            print(
                f"[public-quality] {dataset} {ordinal + 1}/{len(rows)} "
                f"passed={metrics['passed']}",
                flush=True,
            )
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_dataset.setdefault(record["dataset"], []).append(record)
    datasets_summary = {
        dataset: {
            "samples": len(items),
            "passed": all(item["metrics"]["passed"] for item in items),
            "failures": [
                {
                    "source_index": item["source_index"],
                    "task_id": item.get("task_id"),
                    "failures": item["metrics"]["failures"],
                }
                for item in items
                if not item["metrics"]["passed"]
            ],
        }
        for dataset, items in sorted(by_dataset.items())
    }
    return {
        "records": records,
        "datasets": datasets_summary,
        "passed": all(record["metrics"]["passed"] for record in records),
    }


def _run_gsm8k_phase(
    *,
    llm: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from vllm import SamplingParams

    prompts, labels, rows = _build_gsm8k_prompts(
        cache_dir=args.gsm8k_cache_dir,
        num_questions=args.gsm8k_questions,
        num_shots=args.gsm8k_num_shots,
    )
    sampling = SamplingParams(
        max_tokens=args.gsm8k_max_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        stop=["Question", "Assistant:", "<|separator|>"],
        skip_special_tokens=True,
    )
    _sync_cuda()
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling, use_tqdm=False)
    _sync_cuda()
    elapsed_s = time.perf_counter() - start
    records = []
    correct = 0
    invalid = 0
    total_output_tokens = 0
    for idx, output in enumerate(outputs):
        text, token_ids, finish_reason, stop_reason = _output_record(output)
        pred = _gsm8k_answer_value(text)
        is_correct = pred == labels[idx]
        correct += int(is_correct)
        invalid += int(pred == INVALID_GSM8K_ANSWER)
        total_output_tokens += len(token_ids)
        records.append({
            "ordinal": idx,
            "source_index": rows[idx].get("_source_index"),
            "label": labels[idx],
            "prediction": pred,
            "correct": is_correct,
            "invalid": pred == INVALID_GSM8K_ANSWER,
            "finish_reason": finish_reason,
            "stop_reason": stop_reason,
            "output_tokens": len(token_ids),
            "quality_metrics": _quality_metrics(text, token_ids, min_tokens=1),
            "preview": text[:800],
        })
    num_questions = len(records)
    accuracy = correct / num_questions if num_questions else None
    invalid_rate = invalid / num_questions if num_questions else None
    quality_passed = all(record["quality_metrics"]["passed"] for record in records)
    return {
        "records": records,
        "num_questions": num_questions,
        "num_shots": args.gsm8k_num_shots,
        "max_tokens": args.gsm8k_max_tokens,
        "accuracy": accuracy,
        "invalid_rate": invalid_rate,
        "correct": correct,
        "invalid": invalid,
        "elapsed_s": elapsed_s,
        "total_output_tokens": total_output_tokens,
        "tokens_per_second": (
            total_output_tokens / elapsed_s if elapsed_s > 0 else None
        ),
        "quality_passed": quality_passed,
        "passed": quality_passed and accuracy is not None,
    }


def _summarize_longbench_dataset(records: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(record["score"]) for record in records]
    prompt_tokens = [int(record["prompt_tokens"]) for record in records]
    output_tokens = [int(record["output_tokens"]) for record in records]
    garble_passed = all(record["quality_metrics"]["passed"] for record in records)
    return {
        "samples": len(records),
        "score": round(100.0 * statistics.mean(scores), 4) if scores else None,
        "score_values": scores,
        "prompt_tokens_min": min(prompt_tokens) if prompt_tokens else None,
        "prompt_tokens_median": (
            statistics.median(prompt_tokens) if prompt_tokens else None
        ),
        "prompt_tokens_max": max(prompt_tokens) if prompt_tokens else None,
        "output_tokens_mean": (
            statistics.mean(output_tokens) if output_tokens else None
        ),
        "garble_passed": garble_passed,
    }


def _run_longbench_phase(
    *,
    llm: Any,
    tokenizer: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from vllm import SamplingParams

    datasets = _parse_tuple_arg(args.longbench_dataset, DEFAULT_LONG_BENCH_DATASETS)
    dataset2prompt = _load_json(LONG_BENCH_ROOT / "config" / "dataset2prompt.json")
    dataset2maxlen = _load_json(LONG_BENCH_ROOT / "config" / "dataset2maxlen.json")
    records: list[dict[str, Any]] = []
    for dataset in datasets:
        rows = _load_dataset(args.longbench_data_dir / f"{dataset}.jsonl")
        selected = _select_rows(
            rows,
            min_length=args.longbench_min_length,
            limit=args.longbench_limit,
        )
        sampling = SamplingParams(
            max_tokens=int(dataset2maxlen[dataset]),
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            skip_special_tokens=True,
        )
        for ordinal, row in enumerate(selected):
            prompt, prompt_tokens = _build_longbench_prompt(
                tokenizer,
                dataset=dataset,
                row=row,
                prompt_format=dataset2prompt[dataset],
                max_input_tokens=args.longbench_max_input_tokens,
                chat_template=args.longbench_chat_template,
            )
            _sync_cuda()
            start = time.perf_counter()
            outputs = llm.generate([prompt], sampling, use_tqdm=False)
            _sync_cuda()
            elapsed_s = time.perf_counter() - start
            request_output = outputs[0]
            completion = request_output.outputs[0]
            prediction = completion.text
            score = _score_longbench_one(dataset, prediction, row)
            token_ids = list(completion.token_ids)
            records.append({
                "dataset": dataset,
                "ordinal": ordinal,
                "source_index": row["_source_index"],
                "id": row.get("_id"),
                "length": row.get("length"),
                "prompt_tokens": prompt_tokens,
                "answers": row["answers"],
                "all_classes": row.get("all_classes", []),
                "score": score,
                "output_tokens": len(token_ids),
                "finish_reason": completion.finish_reason,
                "request_metrics": _request_metrics(
                    request_output,
                    elapsed_s,
                    prompt_tokens,
                ),
                "quality_metrics": _quality_metrics(
                    prediction,
                    token_ids,
                    min_tokens=1,
                ),
                "prediction_preview": prediction[:1200],
                "prediction_tail": prediction[-1200:],
            })
            print(
                f"[longbench] {dataset} {ordinal + 1}/{len(selected)} "
                f"score={100.0 * score:.2f} prompt_tokens={prompt_tokens}",
                flush=True,
            )

    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_dataset.setdefault(record["dataset"], []).append(record)
    dataset_summary = {
        dataset: _summarize_longbench_dataset(items)
        for dataset, items in sorted(by_dataset.items())
    }
    dataset_scores = [
        summary["score"]
        for summary in dataset_summary.values()
        if summary.get("score") is not None
    ]
    return {
        "records": records,
        "datasets": dataset_summary,
        "average_score": (
            round(statistics.mean(dataset_scores), 4) if dataset_scores else None
        ),
        "passed_garble": all(
            record["quality_metrics"]["passed"] for record in records
        ),
    }


def _run_worker(args: argparse.Namespace) -> int:
    case = CaseSpec.from_jsonable(json.loads(args.case_json))
    out_path = args.out
    phases = set(_parse_tuple_arg(
        args.phase,
        ("speed", "prefix", "quality", "public_quality", "gsm8k",
         "longbench"),
    ))

    def write_checkpoint() -> None:
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(out_path)

    result: dict[str, Any] = {
        "case": case.to_jsonable(),
        "status": "started",
        "complete": False,
        "phases": sorted(phases),
        "phases_completed": [],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_checkpoint()
    try:
        from transformers import AutoTokenizer

        from vllm import LLM, SamplingParams

        tokenizer = AutoTokenizer.from_pretrained(
            case.model.path,
            trust_remote_code=True,
            use_fast=True,
        )
        llm_kwargs = _make_llm_kwargs(case, args)
        result["engine_kwargs"] = json.loads(json.dumps(llm_kwargs, default=str))
        result["env"] = {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith((
                "CUDA_",
                "VLLM_",
                "TORCHINDUCTOR_",
                "TRITON_",
                "PYTHONPATH",
            ))
        }
        start = time.perf_counter()
        llm = LLM(**llm_kwargs)
        result["load_seconds"] = time.perf_counter() - start
        result["status"] = "loaded"
        write_checkpoint()

        warmup_ids = _make_exact_prompt_token_ids(tokenizer, args.warmup_input_len)
        llm.generate(
            [{"prompt_token_ids": warmup_ids}],
            SamplingParams(
                max_tokens=args.warmup_output_len,
                temperature=0.0,
                top_p=1.0,
                top_k=-1,
                skip_special_tokens=False,
            ),
            use_tqdm=False,
        )
        result["status"] = "warmed"
        write_checkpoint()

        if "speed" in phases:
            result["speed"] = _run_speed_phase(
                llm=llm,
                tokenizer=tokenizer,
                args=args,
            )
            result["phases_completed"].append("speed")
            write_checkpoint()
        if "prefix" in phases:
            result["prefix_cache_probe"] = _run_prefix_probe(
                llm=llm,
                tokenizer=tokenizer,
                args=args,
            )
            result["phases_completed"].append("prefix")
            write_checkpoint()
        if "quality" in phases:
            result["quality"] = _run_quality_phase(
                llm=llm,
                tokenizer=tokenizer,
                case=case,
                args=args,
            )
            result["phases_completed"].append("quality")
            write_checkpoint()
        if "public_quality" in phases:
            result["public_quality"] = _run_public_quality_phase(
                llm=llm,
                tokenizer=tokenizer,
                args=args,
            )
            result["phases_completed"].append("public_quality")
            write_checkpoint()
        if "gsm8k" in phases:
            result["gsm8k"] = _run_gsm8k_phase(
                llm=llm,
                args=args,
            )
            result["phases_completed"].append("gsm8k")
            write_checkpoint()
        if "longbench" in phases:
            result["longbench"] = _run_longbench_phase(
                llm=llm,
                tokenizer=tokenizer,
                args=args,
            )
            result["phases_completed"].append("longbench")
            write_checkpoint()

        result["status"] = "completed"
        result["complete"] = True
        result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        write_checkpoint()
        print(json.dumps(_summarize_payload(result), ensure_ascii=False))
        return 0
    except Exception as exc:
        result["status"] = "failed"
        result["complete"] = True
        result["exception"] = repr(exc)
        result["traceback"] = traceback.format_exc()
        result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        write_checkpoint()
        print(result["traceback"], file=sys.stderr, flush=True)
        return 2


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


def _parse_spec_decode_metrics(log_text: str) -> dict[str, Any]:
    line_re = re.compile(
        r"SpecDecoding metrics: Mean acceptance length: ([0-9.]+), "
        r"Accepted throughput: ([0-9.]+) tokens/s, "
        r"Drafted throughput: ([0-9.]+) tokens/s, "
        r"Accepted: ([0-9]+) tokens, Drafted: ([0-9]+) tokens, "
        r"Per-position acceptance rate: ([0-9., ]+), "
        r"Avg Draft acceptance rate: ([0-9.]+)%"
    )
    samples = []
    for match in line_re.finditer(log_text):
        per_pos = [
            float(item.strip())
            for item in match.group(6).split(",")
            if item.strip()
        ]
        samples.append({
            "acceptance_length": float(match.group(1)),
            "accepted_throughput": float(match.group(2)),
            "drafted_throughput": float(match.group(3)),
            "accepted_tokens": int(match.group(4)),
            "drafted_tokens": int(match.group(5)),
            "per_position_acceptance_rate": per_pos,
            "draft_acceptance_rate_pct": float(match.group(7)),
        })
    if not samples:
        return {"samples": [], "found": False}
    total_accepted = sum(sample["accepted_tokens"] for sample in samples)
    total_drafted = sum(sample["drafted_tokens"] for sample in samples)
    total_drafts = 0.0
    for sample in samples:
        length_minus_bonus = max(sample["acceptance_length"] - 1.0, 0.0)
        if length_minus_bonus > 0.0:
            total_drafts += sample["accepted_tokens"] / length_minus_bonus
    acceptance_length = (
        1.0 + total_accepted / total_drafts if total_drafts > 0 else None
    )
    all_one_plateau_samples = [
        sample for sample in samples
        if sample["per_position_acceptance_rate"]
        and all(rate >= 0.999 for rate in sample["per_position_acceptance_rate"])
    ]
    return {
        "samples": samples,
        "found": True,
        "num_samples": len(samples),
        "total_accepted_tokens": total_accepted,
        "total_drafted_tokens": total_drafted,
        "draft_acceptance_rate_pct": (
            100.0 * total_accepted / total_drafted if total_drafted else None
        ),
        "acceptance_length": acceptance_length,
        "last_acceptance_length": samples[-1]["acceptance_length"],
        "last_per_position_acceptance_rate": samples[-1][
            "per_position_acceptance_rate"
        ],
        "sustained_all_one_plateau": len(all_one_plateau_samples) >= 3,
    }


def _summarize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    speed_metrics = (payload.get("speed") or {}).get("request_metrics") or {}
    quality = payload.get("quality") or {}
    public_quality = payload.get("public_quality") or {}
    gsm8k = payload.get("gsm8k") or {}
    longbench = payload.get("longbench") or {}
    return {
        "case": payload.get("case", {}).get("name"),
        "status": payload.get("status"),
        "decode_tps": speed_metrics.get("steady_decode_tps"),
        "prefill_tps": speed_metrics.get("prefill_tps"),
        "quality_passed": quality.get("passed"),
        "public_quality_passed": public_quality.get("passed"),
        "gsm8k_accuracy": gsm8k.get("accuracy"),
        "longbench_average_score": longbench.get("average_score"),
        "longbench_garble_passed": longbench.get("passed_garble"),
    }


def _summarize_case(
    *,
    case: CaseSpec,
    out_path: Path,
    log_path: Path,
    returncode: int | None,
) -> dict[str, Any]:
    log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
    if not out_path.exists():
        return {
            "case": case.to_jsonable(),
            "status": "missing_json",
            "returncode": returncode,
            "log": str(log_path),
            "json": str(out_path),
            "kv_capacity": _parse_kv_capacity(log_text),
            "spec_decode": _parse_spec_decode_metrics(log_text),
        }
    data = json.loads(out_path.read_text(encoding="utf-8"))
    speed_metrics = (data.get("speed") or {}).get("request_metrics") or {}
    quality = data.get("quality") or {}
    public_quality = data.get("public_quality") or {}
    gsm8k = data.get("gsm8k") or {}
    failed_quality_prompts = [
        {
            "id": record["id"],
            "repeat": record["repeat"],
            "failures": record["metrics"]["failures"],
        }
        for record in quality.get("records", [])
        if not record["metrics"]["passed"]
    ]
    failed_public_quality = [
        {
            "dataset": record["dataset"],
            "source_index": record["source_index"],
            "task_id": record.get("task_id"),
            "failures": record["metrics"]["failures"],
        }
        for record in public_quality.get("records", [])
        if not record["metrics"]["passed"]
    ]
    failed_gsm8k_quality = [
        {
            "source_index": record["source_index"],
            "failures": record["quality_metrics"]["failures"],
        }
        for record in gsm8k.get("records", [])
        if not record["quality_metrics"]["passed"]
    ]
    longbench = data.get("longbench") or {}
    failed_longbench_garble = [
        {
            "dataset": record["dataset"],
            "source_index": record["source_index"],
            "failures": record["quality_metrics"]["failures"],
        }
        for record in longbench.get("records", [])
        if not record["quality_metrics"]["passed"]
    ]
    return {
        "case": data.get("case", case.to_jsonable()),
        "status": data.get("status"),
        "returncode": returncode,
        "complete": data.get("complete"),
        "phases": data.get("phases"),
        "phases_completed": data.get("phases_completed"),
        "load_seconds": data.get("load_seconds"),
        "prefill_tps": speed_metrics.get("prefill_tps"),
        "prefill_time_s": speed_metrics.get("prefill_time_s"),
        "decode_tps": speed_metrics.get("steady_decode_tps"),
        "decode_time_s": speed_metrics.get("decode_time_s"),
        "speed_output_tokens": (
            data.get("speed") or {}
        ).get("output_tokens"),
        "speed_quality_passed": (
            (data.get("speed") or {}).get("quality_metrics") or {}
        ).get("passed"),
        "quality_passed": quality.get("passed"),
        "failed_quality_prompts": failed_quality_prompts,
        "public_quality_passed": public_quality.get("passed"),
        "failed_public_quality": failed_public_quality,
        "gsm8k_passed": gsm8k.get("passed"),
        "gsm8k_quality_passed": gsm8k.get("quality_passed"),
        "gsm8k_accuracy": gsm8k.get("accuracy"),
        "gsm8k_invalid_rate": gsm8k.get("invalid_rate"),
        "failed_gsm8k_quality": failed_gsm8k_quality,
        "prefix_cache_probe_passed": (
            (data.get("prefix_cache_probe") or {}).get("passed")
        ),
        "prefix_ttft_ratio": (
            (data.get("prefix_cache_probe") or {}).get(
                "ttft_ratio_second_over_first"
            )
        ),
        "longbench_average_score": longbench.get("average_score"),
        "longbench_garble_passed": longbench.get("passed_garble"),
        "longbench_dataset_scores": {
            dataset: summary.get("score")
            for dataset, summary in (longbench.get("datasets") or {}).items()
        },
        "failed_longbench_garble": failed_longbench_garble,
        "exception": data.get("exception"),
        "kv_capacity": _parse_kv_capacity(log_text),
        "spec_decode": _parse_spec_decode_metrics(log_text),
        "json": str(out_path),
        "log": str(log_path),
    }


def _case_key(row_or_case: dict[str, Any]) -> tuple[str, str, str, int, str]:
    case = row_or_case.get("case", row_or_case)
    return (
        case["backend"],
        case["model"]["key"],
        case["kv_cache_dtype"],
        int(case["tp"]),
        case["mode"],
    )


def _nospec_key(row: dict[str, Any]) -> tuple[str, str, str, int, str]:
    case = row["case"]
    return (
        case["backend"],
        case["model"]["key"],
        case["kv_cache_dtype"],
        int(case["tp"]),
        "nospec",
    )


def _base_auto_key(row: dict[str, Any]) -> tuple[str, str, str, int, str]:
    case = row["case"]
    return (case["backend"], case["model"]["key"], "auto", int(case["tp"]), "nospec")


def _gate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {_case_key(row): row for row in rows}
    gated = []
    for row in rows:
        failures = []
        case = row["case"]
        mode = case["mode"]
        phases = set(row.get("phases") or (
            "speed",
            "prefix",
            "quality",
            "public_quality",
            "gsm8k",
            "longbench",
        ))
        status = row.get("status")
        decode_tps = row.get("decode_tps")
        prefill_tps = row.get("prefill_tps")
        if status != "completed":
            failures.append(f"status={status}")
        if "speed" in phases:
            if not isinstance(decode_tps, (int, float)) or decode_tps <= 0:
                failures.append("missing_decode_tps")
            if not isinstance(prefill_tps, (int, float)) or prefill_tps <= 0:
                failures.append("missing_prefill_tps")
            if row.get("speed_quality_passed") is not True:
                failures.append("speed_output_quality")
        if "quality" in phases and row.get("quality_passed") is not True:
            failures.append("fixed_quality")
        if (
            "public_quality" in phases
            and row.get("public_quality_passed") is not True
        ):
            failures.append("public_quality")
        if "gsm8k" in phases:
            if row.get("gsm8k_passed") is not True:
                failures.append("gsm8k")
            if row.get("gsm8k_quality_passed") is not True:
                failures.append("gsm8k_output_quality")
            if not isinstance(row.get("gsm8k_accuracy"), (int, float)):
                failures.append("missing_gsm8k_accuracy")
            if isinstance(row.get("gsm8k_invalid_rate"), (int, float)) and (
                row["gsm8k_invalid_rate"] > 0.30
            ):
                failures.append("gsm8k_invalid_rate_gt_30pct")
        if "longbench" in phases:
            if row.get("longbench_garble_passed") is not True:
                failures.append("longbench_garble")
            if not isinstance(row.get("longbench_average_score"), (int, float)):
                failures.append("missing_longbench_score")
        if "prefix" in phases and row.get("prefix_cache_probe_passed") is not True:
            failures.append("prefix_probe")

        baseline_auto = by_key.get(_base_auto_key(row))
        if "speed" in phases and mode == "nospec" and case["kv_cache_dtype"] != "auto":
            if baseline_auto is None:
                failures.append("missing_auto_baseline")
            else:
                base_decode = baseline_auto.get("decode_tps")
                base_prefill = baseline_auto.get("prefill_tps")
                if base_decode and decode_tps and decode_tps < 0.80 * base_decode:
                    failures.append("decode_lt_80pct_auto_baseline")
                if base_prefill and prefill_tps and prefill_tps < 0.70 * base_prefill:
                    failures.append("prefill_lt_70pct_auto_baseline")
        if "speed" in phases and mode in ("mtp4", "dflash16"):
            baseline = by_key.get(_nospec_key(row))
            if baseline is None:
                failures.append("missing_same_cache_nospec_baseline")
            else:
                base_decode = baseline.get("decode_tps")
                min_gain = 1.25 if mode == "mtp4" else 1.05
                if base_decode and decode_tps and decode_tps < min_gain * base_decode:
                    failures.append(f"decode_lt_{min_gain:.2f}x_nospec")
            spec = row.get("spec_decode") or {}
            if not spec.get("found"):
                failures.append("missing_spec_decode_metrics")
            else:
                acceptance_length = spec.get("acceptance_length")
                if not acceptance_length or acceptance_length <= 1.5:
                    failures.append("acceptance_length_le_1.5")
                if spec.get("sustained_all_one_plateau"):
                    failures.append("sustained_all_one_acceptance_plateau")

        baseline_score = by_key.get(_base_auto_key(row))
        current_scores = row.get("longbench_dataset_scores") or {}
        baseline_scores = (
            baseline_score.get("longbench_dataset_scores") if baseline_score else None
        ) or {}
        if mode != "nospec" or case["kv_cache_dtype"] != "auto":
            if "longbench" in phases and not baseline_scores:
                failures.append("missing_longbench_score_baseline")
            base_average = (
                baseline_score.get("longbench_average_score")
                if baseline_score else None
            )
            current_average = row.get("longbench_average_score")
            if (
                "longbench" in phases
                and
                isinstance(base_average, (int, float))
                and isinstance(current_average, (int, float))
                and current_average < 0.85 * base_average
            ):
                failures.append("longbench_average_lt_85pct_baseline")
            base_gsm8k = (
                baseline_score.get("gsm8k_accuracy") if baseline_score else None
            )
            current_gsm8k = row.get("gsm8k_accuracy")
            if (
                "gsm8k" in phases
                and
                isinstance(base_gsm8k, (int, float))
                and isinstance(current_gsm8k, (int, float))
                and current_gsm8k < 0.85 * base_gsm8k
            ):
                failures.append("gsm8k_accuracy_lt_85pct_baseline")
            if "longbench" in phases:
                for dataset, score in current_scores.items():
                    base_score = baseline_scores.get(dataset)
                    if base_score is None or score is None:
                        failures.append(f"missing_longbench_score:{dataset}")
                        continue
                    threshold = (
                        max(0.0, base_score - 5.0)
                        if base_score < 20.0
                        else 0.70 * base_score
                    )
                    if score < threshold:
                        failures.append(f"longbench_score_drop:{dataset}")
        gated_row = dict(row)
        gated_row["gate_failures"] = failures
        gated_row["gate_passed"] = not failures
        gated.append(gated_row)
    return gated


def _write_summary(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    rows = _gate_rows(rows)
    summary_json = out_dir / "summary.json"
    summary_json.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# SM70 Release Matrix",
        "",
        "| Case | Gate | Decode tok/s | Prefill tok/s | GSM8K | LongBench | "
        "Accept len | KV tokens | Failures | Artifact |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        case = row["case"]
        spec = row.get("spec_decode") or {}
        kv_capacity = row.get("kv_capacity") or {}
        failures = row.get("gate_failures") or []
        lines.append(
            "| {case} | {gate} | {decode} | {prefill} | {gsm8k} | {score} | "
            "{accept} | {kv_tokens} | {failures} | {artifact} |".format(
                case=case["name"],
                gate="PASS" if row.get("gate_passed") else "FAIL",
                decode=(
                    f"{row['decode_tps']:.2f}"
                    if isinstance(row.get("decode_tps"), (int, float))
                    else "-"
                ),
                prefill=(
                    f"{row['prefill_tps']:.2f}"
                    if isinstance(row.get("prefill_tps"), (int, float))
                    else "-"
                ),
                gsm8k=(
                    f"{100.0 * row['gsm8k_accuracy']:.2f}%"
                    if isinstance(row.get("gsm8k_accuracy"), (int, float))
                    else "-"
                ),
                score=(
                    f"{row['longbench_average_score']:.2f}"
                    if isinstance(row.get("longbench_average_score"), (int, float))
                    else "-"
                ),
                accept=(
                    f"{spec['acceptance_length']:.2f}"
                    if isinstance(spec.get("acceptance_length"), (int, float))
                    else "-"
                ),
                kv_tokens=kv_capacity.get("gpu_kv_cache_tokens", "-"),
                failures=", ".join(failures) if failures else "-",
                artifact=Path(row.get("json", "")).name,
            )
        )
    pass_count = sum(1 for row in rows if row.get("gate_passed"))
    lines.extend([
        "",
        f"Cases: {pass_count}/{len(rows)} passed.",
        "",
        "Primary gates: every case must load, report speed, pass fixed "
        "quality prompts, PublicQuality prompts (HumanEval/MBPP/IFEval), "
        "GSM8K, and LongBench garble checks. Non-baseline KV modes must keep "
        "decode >=80% and prefill >=70% of auto no-spec. MTP4 must decode "
        ">=1.25x no-spec with acceptance length >1.5 and no sustained "
        "all-position 1.000 plateau. DFlash16 must decode >=1.05x no-spec "
        "with the same acceptance quality gate. GSM8K and LongBench candidate "
        "routes must keep at least 85% of the auto/no-spec baseline aggregate.",
    ])
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _default_env(args: argparse.Namespace, case: CaseSpec,
                 devices: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = devices
    env["PYTHONPATH"] = (
        f"{REPO_ROOT}:{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else str(REPO_ROOT)
    )
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor_ymzx")
    env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    env.setdefault("TRITON_CACHE_AUTOTUNING", "1")
    env.setdefault("VLLM_SM70_FLASH_V100_DECODE_GRAPH_NO_COMPILE", "0")
    env.setdefault("VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH", "1")
    env.setdefault("VLLM_SM70_QWEN_GDN_SPEC_CORE_OP", "0")
    env["VLLM_SM70_QUANT_BACKEND"] = case.backend
    if case.kv_cache_dtype.startswith("turboquant"):
        env.setdefault("VLLM_SM70_TURBOQUANT_FLASH_V100_DECODE", "0")
        env.setdefault("VLLM_SM70_TURBOQUANT_CONTINUATION_WORKSPACE_TOKENS",
                       str(args.max_model_len + args.max_num_batched_tokens))
    if args.extra_env:
        for value in args.extra_env:
            if "=" not in value:
                raise ValueError(f"expected KEY=VALUE for --extra-env, got {value}")
            key, raw = value.split("=", 1)
            env[key] = raw
    return env


def _case_artifact_paths(out_dir: Path, case: CaseSpec) -> tuple[Path, Path]:
    cases_dir = out_dir / "cases"
    logs_dir = out_dir / "logs"
    cases_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return cases_dir / f"{case.name}.json", logs_dir / f"{case.name}.log"


def _case_result_reusable(data: dict[str, Any],
                          phases: tuple[str, ...]) -> bool:
    if not (data.get("complete") and data.get("status") == "completed"):
        return False
    completed_phases = set(data.get("phases_completed") or [])
    if any(phase not in completed_phases for phase in phases):
        return False
    if "speed" in phases:
        speed = data.get("speed") or {}
        request_metrics = speed.get("request_metrics") or {}
        output_tokens = request_metrics.get("output_tokens", speed.get("output_tokens"))
        expected_tokens = speed.get("max_tokens")
        if request_metrics.get("steady_decode_tps") is None:
            return False
        if expected_tokens is not None and output_tokens != expected_tokens:
            return False
    return True


def _run_case_subprocess(
    case: CaseSpec,
    *,
    args: argparse.Namespace,
    out_dir: Path,
    devices: str,
) -> dict[str, Any]:
    out_path, log_path = _case_artifact_paths(out_dir, case)
    phases = _parse_tuple_arg(
        args.phase,
        ("speed", "prefix", "quality", "public_quality", "gsm8k", "longbench"),
    )
    if args.resume and out_path.exists():
        try:
            data = json.loads(out_path.read_text(encoding="utf-8"))
            if _case_result_reusable(data, phases):
                return _summarize_case(
                    case=case,
                    out_path=out_path,
                    log_path=log_path,
                    returncode=None,
                )
        except Exception:
            pass
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--case-json",
        json.dumps(case.to_jsonable(), ensure_ascii=False),
        "--out",
        str(out_path),
        "--phase",
        ",".join(phases),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--attention-backend",
        args.attention_backend,
        "--speed-input-len",
        str(args.speed_input_len),
        "--speed-output-len",
        str(args.speed_output_len),
        "--speed-prompt",
        args.speed_prompt,
        "--quality-max-tokens",
        str(args.quality_max_tokens),
        "--quality-repeat",
        str(args.quality_repeat),
        "--public-quality-limit",
        str(args.public_quality_limit),
        "--public-quality-max-tokens",
        str(args.public_quality_max_tokens),
        "--public-dataset-cache-dir",
        str(args.public_dataset_cache_dir),
        "--gsm8k-cache-dir",
        str(args.gsm8k_cache_dir),
        "--gsm8k-questions",
        str(args.gsm8k_questions),
        "--gsm8k-num-shots",
        str(args.gsm8k_num_shots),
        "--gsm8k-max-tokens",
        str(args.gsm8k_max_tokens),
        "--longbench-limit",
        str(args.longbench_limit),
        "--longbench-min-length",
        str(args.longbench_min_length),
        "--longbench-max-input-tokens",
        str(args.longbench_max_input_tokens),
        "--longbench-chat-template",
        args.longbench_chat_template,
        "--longbench-data-dir",
        str(args.longbench_data_dir),
        "--sampling-seed",
        str(args.sampling_seed),
        "--seed",
        str(args.seed),
    ]
    if not args.enable_thinking:
        cmd.append("--disable-thinking")
    if args.quality_save_full_output:
        cmd.append("--quality-save-full-output")
    for engine_arg in args.engine_arg:
        cmd.extend(["--engine-arg", engine_arg])
    for prompt_id in args.quality_prompt_id or []:
        cmd.extend(["--quality-prompt-id", prompt_id])
    for dataset in args.public_quality_dataset or []:
        cmd.extend(["--public-quality-dataset", dataset])
    for dataset in args.longbench_dataset or []:
        cmd.extend(["--longbench-dataset", dataset])
    env = _default_env(args, case, devices)
    print(f"[matrix] start {case.name} on CUDA_VISIBLE_DEVICES={devices}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            timeout=args.timeout_s if args.timeout_s > 0 else None,
            check=False,
        )
    row = _summarize_case(
        case=case,
        out_path=out_path,
        log_path=log_path,
        returncode=proc.returncode,
    )
    print(
        "[matrix] done {case}: rc={rc} decode={decode} quality={quality} "
        "longbench={score}".format(
            case=case.name,
            rc=proc.returncode,
            decode=row.get("decode_tps"),
            quality=row.get("quality_passed"),
            score=row.get("longbench_average_score"),
        ),
        flush=True,
    )
    return row


def _parse_device_groups(raw: str) -> list[str]:
    groups = []
    for group in raw.split(";"):
        group = group.strip()
        if group:
            groups.append(group)
    if not groups:
        raise ValueError("at least one TP2 device group is required")
    return groups


def _run_tp2_group_worker(
    *,
    work_queue: queue.Queue[CaseSpec],
    devices: str,
    args: argparse.Namespace,
    out_dir: Path,
    rows: list[dict[str, Any]],
) -> None:
    while True:
        try:
            case = work_queue.get_nowait()
        except queue.Empty:
            return
        try:
            row = _run_case_subprocess(
                case,
                args=args,
                out_dir=out_dir,
                devices=devices,
            )
            rows.append(row)
            _write_summary(out_dir, rows)
        finally:
            work_queue.task_done()


def _run_matrix(args: argparse.Namespace) -> int:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (
        REPO_ROOT / "bench_results" / f"sm70_release_matrix_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = _make_cases(
        backends=_parse_tuple_arg(args.backend, DEFAULT_BACKENDS),
        tps=_parse_int_tuple_arg(args.tp, DEFAULT_TPS),
        kv_cache_dtypes=_parse_tuple_arg(args.kv_cache_dtype, DEFAULT_KV_CACHE_DTYPES),
        include_turboquant_mtp=args.include_turboquant_mtp,
        include_turboquant_dflash=args.include_turboquant_dflash,
        include_fp8_model_fp8_kv_mtp=args.include_fp8_model_fp8_kv_mtp,
        include_gemma_turboquant=args.include_gemma_turboquant,
        include_gemma_nvfp4_fp8_kv=args.include_gemma_nvfp4_fp8_kv,
    )
    cases = _filter_cases(cases, args)
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "case_count": len(cases),
        "cases": [case.to_jsonable() for case in cases],
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "case_json"
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.list_cases or args.dry_run:
        for case in cases:
            print(case.name)
        print(f"[matrix] cases={len(cases)} out_dir={out_dir}")
        return 0

    rows: list[dict[str, Any]] = []
    tp2_cases = [case for case in cases if case.tp == 2]
    tp4_cases = [case for case in cases if case.tp == 4]

    if tp2_cases:
        work_queue: queue.Queue[CaseSpec] = queue.Queue()
        for case in tp2_cases:
            work_queue.put(case)
        tp2_groups = _parse_device_groups(args.tp2_device_groups)
        max_workers = min(args.max_parallel_tp2, len(tp2_groups), len(tp2_cases))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(
                    _run_tp2_group_worker,
                    work_queue=work_queue,
                    devices=tp2_groups[idx],
                    args=args,
                    out_dir=out_dir,
                    rows=rows,
                )
                for idx in range(max_workers)
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()

    for case in tp4_cases:
        row = _run_case_subprocess(
            case,
            args=args,
            out_dir=out_dir,
            devices=args.tp4_devices,
        )
        rows.append(row)
        _write_summary(out_dir, rows)

    _write_summary(out_dir, rows)
    failed = [row for row in _gate_rows(rows) if not row.get("gate_passed")]
    print(f"[matrix] summary: {out_dir / 'summary.md'}", flush=True)
    print(f"[matrix] passed={len(rows) - len(failed)}/{len(rows)}", flush=True)
    return 0 if rows and not failed else 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--case-json", default="")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only-case", action="append")
    parser.add_argument("--model-key", action="append")
    parser.add_argument("--mode", action="append")
    parser.add_argument("--backend", action="append")
    parser.add_argument("--backend-filter", action="append")
    parser.add_argument("--kv-cache-dtype", action="append")
    parser.add_argument("--kv-cache-dtype-filter", action="append")
    parser.add_argument(
        "--include-turboquant-mtp",
        action="store_true",
        help=("Include turboquant_4bit_nc + mtp4 diagnostic cases. "
              "They are skipped by default because the MTP drafter currently "
              "inherits TurboQuant KV on SM70."),
    )
    parser.add_argument(
        "--include-turboquant-dflash",
        action="store_true",
        help=("Include turboquant_4bit_nc + dflash16 diagnostic cases. "
              "They are skipped by default because DFlash draft attention is "
              "non-causal and TurboQuant KV has no non-causal backend on SM70."),
    )
    parser.add_argument(
        "--include-fp8-model-fp8-kv-mtp",
        action="store_true",
        help=("Include FP8-weight model + fp8_e5m2 KV + mtp4 diagnostic cases. "
              "They are skipped by default because Qwen MTP drafter warmup "
              "hits an unsupported static_scaled_fp8_quant e5m2 kernel on SM70."),
    )
    parser.add_argument(
        "--include-gemma-turboquant",
        action="store_true",
        help=("Include Gemma + turboquant_4bit_nc diagnostic cases. They are "
              "skipped by default because Gemma4 multimodal/text-only attention "
              "currently rejects the TurboQuant attention backend on SM70."),
    )
    parser.add_argument(
        "--include-gemma-nvfp4-fp8-kv",
        action="store_true",
        help=("Include Gemma4 NVFP4 + fp8_e5m2 KV diagnostic cases. They are "
              "skipped by default because this combination is rejected by "
              "KV-cache quantization validation on SM70."),
    )
    parser.add_argument("--tp", action="append")
    parser.add_argument("--tp-filter", action="append")
    parser.add_argument("--limit-cases", type=int)
    parser.add_argument(
        "--phase",
        action="append",
        help="Comma-separated phases: speed,prefix,quality,longbench.",
    )
    parser.add_argument("--tp2-device-groups", default="0,1;2,3")
    parser.add_argument("--tp4-devices", default="0,1,2,3")
    parser.add_argument("--max-parallel-tp2", type=int, default=2)
    parser.add_argument("--timeout-s", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--attention-backend", default="FLASH_ATTN_V100")
    parser.add_argument("--warmup-input-len", type=int, default=512)
    parser.add_argument("--warmup-output-len", type=int, default=32)
    parser.add_argument("--speed-input-len", type=int, default=4096)
    parser.add_argument("--speed-output-len", type=int, default=1024)
    parser.add_argument("--speed-prompt",
                        choices=("coding", "synthetic"),
                        default="coding")
    parser.add_argument("--prefix-probe-input-len", type=int, default=2048)
    parser.add_argument("--prefix-probe-output-len", type=int, default=32)
    parser.add_argument(
        "--prefix-probe-max-ttft-ratio",
        type=float,
        default=0.90,
        help=("Require the second identical prompt TTFT to be at most this "
              "fraction of the first prompt TTFT during the prefix phase."),
    )
    parser.add_argument("--quality-max-tokens", type=int, default=6000)
    parser.add_argument("--quality-repeat", type=int, default=1)
    parser.add_argument("--quality-prompt-id", action="append")
    parser.add_argument(
        "--quality-save-full-output",
        action="store_true",
        help="Store full quality text and token IDs instead of preview/tail only.",
    )
    parser.add_argument("--public-dataset-cache-dir",
                        type=Path,
                        default=DEFAULT_PUBLIC_DATASET_CACHE_DIR)
    parser.add_argument("--public-quality-dataset", action="append")
    parser.add_argument("--public-quality-limit", type=int, default=5)
    parser.add_argument("--public-quality-max-tokens", type=int, default=768)
    parser.add_argument("--gsm8k-cache-dir",
                        type=Path,
                        default=DEFAULT_GSM8K_CACHE_DIR)
    parser.add_argument("--gsm8k-questions", type=int, default=32)
    parser.add_argument("--gsm8k-num-shots", type=int, default=5)
    parser.add_argument("--gsm8k-max-tokens", type=int, default=256)
    parser.add_argument("--sampling-seed", type=int, default=20260620)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--disable-thinking", action="store_false",
                        dest="enable_thinking")
    parser.set_defaults(enable_thinking=True)
    parser.add_argument("--longbench-data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--longbench-dataset", action="append")
    parser.add_argument("--longbench-limit", type=int, default=5)
    parser.add_argument("--longbench-min-length", type=int, default=4096)
    parser.add_argument("--longbench-max-input-tokens", type=int, default=32768)
    parser.add_argument(
        "--longbench-chat-template",
        choices=("official", "always", "none"),
        default="official",
    )
    parser.add_argument("--engine-arg", action="append", default=[])
    parser.add_argument("--extra-env", action="append")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.worker:
        if not args.case_json:
            raise ValueError("--case-json is required in worker mode")
        if args.out is None:
            raise ValueError("--out is required in worker mode")
        return _run_worker(args)
    return _run_matrix(args)


if __name__ == "__main__":
    raise SystemExit(main())
