# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run paired HumanEval and LongBench-subset gates against a vLLM API."""

from __future__ import annotations

import argparse
import ctypes
import functools
import hashlib
import http.client
import json
import os
import resource
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import regex as re
from transformers import AutoTokenizer

NO_CHAT_DATASETS = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _post_json(
    *,
    host: str,
    port: int,
    path: str,
    payload: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        connection.request(
            "POST",
            path,
            body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        raw = response.read()
    finally:
        connection.close()
    if response.status != 200:
        raise RuntimeError(
            f"{path} returned HTTP {response.status}: "
            f"{raw.decode('utf-8', errors='replace')}"
        )
    return json.loads(raw)


def _chat(
    *,
    host: str,
    port: int,
    model: str,
    content: str,
    max_tokens: int,
    timeout: int,
) -> tuple[str, dict[str, Any]]:
    result = _post_json(
        host=host,
        port=port,
        path="/v1/chat/completions",
        timeout=timeout,
        payload={
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 20260824,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    message = result["choices"][0]["message"]
    return message.get("content") or "", result


def _completion(
    *,
    host: str,
    port: int,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: int,
) -> tuple[str, dict[str, Any]]:
    result = _post_json(
        host=host,
        port=port,
        path="/v1/completions",
        timeout=timeout,
        payload={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 20260824,
        },
    )
    return result["choices"][0].get("text") or "", result


def _extract_python(text: str, entry_point: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL)
    for block in blocks:
        if re.search(rf"\bdef\s+{re.escape(entry_point)}\s*\(", block):
            return block.strip()
    if blocks:
        return max(blocks, key=len).strip()
    return text.strip()


def _sandbox_preexec(sandbox_dir: str) -> None:
    """Deny filesystem/network access before executing generated code."""
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    libc.prctl.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    libc.prctl.restype = ctypes.c_int
    sys_create_ruleset = 444
    sys_add_rule = 445
    sys_restrict_self = 446
    abi = libc.syscall(sys_create_ruleset, 0, 0, 1)
    if abi < 1:
        raise OSError(ctypes.get_errno(), "Landlock is unavailable")

    class RulesetAttr(ctypes.Structure):
        _fields_ = [
            ("handled_access_fs", ctypes.c_uint64),
            ("handled_access_net", ctypes.c_uint64),
        ]

    class PathBeneathAttr(ctypes.Structure):
        _pack_ = 1
        _fields_ = [
            ("allowed_access", ctypes.c_uint64),
            ("parent_fd", ctypes.c_int32),
        ]

    handled_fs = (1 << 13) - 1
    if abi >= 2:
        handled_fs |= 1 << 13
    if abi >= 3:
        handled_fs |= 1 << 14
    ruleset_attr = RulesetAttr(handled_fs, 3 if abi >= 4 else 0)
    ruleset_fd = libc.syscall(
        sys_create_ruleset,
        ctypes.byref(ruleset_attr),
        ctypes.sizeof(ruleset_attr),
        0,
    )
    if ruleset_fd < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset")

    read_only = (1 << 0) | (1 << 2) | (1 << 3)
    allowed_paths = (
        ("/usr", read_only),
        ("/lib", read_only),
        ("/lib64", read_only),
        ("/dev/null", (1 << 1) | (1 << 2)),
        ("/dev/urandom", 1 << 2),
        (sandbox_dir, handled_fs),
    )
    for path, allowed_access in allowed_paths:
        parent_fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
        path_attr = PathBeneathAttr(allowed_access, parent_fd)
        if (
            libc.syscall(
                sys_add_rule,
                ruleset_fd,
                1,
                ctypes.byref(path_attr),
                0,
            )
            < 0
        ):
            raise OSError(ctypes.get_errno(), f"landlock_add_rule({path})")
        os.close(parent_fd)

    if libc.prctl(38, 1, 0, 0, 0) < 0:
        raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS")
    if libc.syscall(sys_restrict_self, ruleset_fd, 0) < 0:
        raise OSError(ctypes.get_errno(), "landlock_restrict_self")
    os.close(ruleset_fd)

    seccomp = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    seccomp.seccomp_init.argtypes = [ctypes.c_uint32]
    seccomp.seccomp_init.restype = ctypes.c_void_p
    seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    seccomp.seccomp_syscall_resolve_name.restype = ctypes.c_int
    seccomp.seccomp_rule_add.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    seccomp.seccomp_rule_add.restype = ctypes.c_int
    seccomp.seccomp_load.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_load.restype = ctypes.c_int
    seccomp.seccomp_release.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_release.restype = None
    context = seccomp.seccomp_init(0x7FFF0000)
    if not context:
        raise RuntimeError("seccomp_init failed")
    deny_with_eperm = 0x00050000 | 1
    for name in (
        b"socket",
        b"socketpair",
        b"connect",
        b"bind",
        b"listen",
        b"accept",
        b"accept4",
        b"sendto",
        b"sendmsg",
        b"recvfrom",
        b"recvmsg",
        b"kill",
        b"tkill",
        b"tgkill",
        b"pidfd_open",
        b"pidfd_getfd",
        b"pidfd_send_signal",
        b"ptrace",
        b"process_vm_readv",
        b"process_vm_writev",
        b"kcmp",
        b"fork",
        b"vfork",
        b"clone",
        b"clone3",
        b"unshare",
        b"setns",
        b"mount",
        b"umount2",
        b"pivot_root",
        b"bpf",
        b"perf_event_open",
        b"userfaultfd",
        b"keyctl",
        b"add_key",
        b"request_key",
        b"open_by_handle_at",
        b"name_to_handle_at",
    ):
        syscall_number = seccomp.seccomp_syscall_resolve_name(name)
        if (
            syscall_number >= 0
            and seccomp.seccomp_rule_add(context, deny_with_eperm, syscall_number, 0)
            != 0
        ):
            raise RuntimeError(f"seccomp_rule_add failed for {name!r}")
    if seccomp.seccomp_load(context) != 0:
        raise RuntimeError("seccomp_load failed")
    seccomp.seccomp_release(context)

    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_AS, (256 << 20, 256 << 20))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
    resource.setrlimit(resource.RLIMIT_NPROC, (16, 16))


def _execute_humaneval(program: str, timeout: int = 8) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="dsv4-humaneval-") as sandbox_dir:
        output_path = Path(sandbox_dir) / "output.log"
        try:
            with output_path.open("wb") as output_file:
                completed = subprocess.run(
                    ["/usr/bin/python3", "-I", "-S", "-"],
                    input=program.encode("utf-8"),
                    stdout=output_file,
                    stderr=subprocess.STDOUT,
                    cwd=sandbox_dir,
                    env={"PATH": "/usr/bin", "HOME": sandbox_dir},
                    preexec_fn=functools.partial(_sandbox_preexec, sandbox_dir),
                    timeout=timeout,
                )
            output = output_path.read_bytes()[-2000:].decode("utf-8", errors="replace")
            return {
                "passed": completed.returncode == 0,
                "returncode": completed.returncode,
                "output": output,
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "passed": False,
                "returncode": None,
                "output": f"timeout: {exc}",
            }


def _run_humaneval(
    *,
    host: str,
    port: int,
    model: str,
    tasks_path: Path,
    limit: int,
    timeout: int,
) -> dict[str, Any]:
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))[:limit]
    records: list[dict[str, Any]] = []
    for task in tasks:
        prompt = (
            "Complete the Python function below. Return runnable Python code "
            "only, without Markdown or explanation.\n\n" + task["prompt"]
        )
        response, raw = _chat(
            host=host,
            port=port,
            model=model,
            content=prompt,
            max_tokens=768,
            timeout=timeout,
        )
        code = _extract_python(response, task["entry_point"])
        if re.search(rf"\bdef\s+{re.escape(task['entry_point'])}\s*\(", code):
            candidate = code
        else:
            candidate = task["prompt"] + code
        program = candidate + "\n" + task["test"] + f"\ncheck({task['entry_point']})\n"
        execution = _execute_humaneval(program)
        records.append(
            {
                "task_id": task["task_id"],
                "passed": execution["passed"],
                "execution": execution,
                "response": response,
                "usage": raw.get("usage"),
            }
        )
    passed = sum(int(record["passed"]) for record in records)
    return {
        "samples": len(records),
        "passed": passed,
        "pass_at_1": passed / len(records) if records else None,
        "records": records,
    }


def _truncate_middle(tokenizer: Any, prompt: str, max_input_tokens: int) -> str:
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(token_ids) <= max_input_tokens:
        return prompt
    half = max_input_tokens // 2
    return tokenizer.decode(
        token_ids[:half], skip_special_tokens=True
    ) + tokenizer.decode(token_ids[-half:], skip_special_tokens=True)


def _select_longbench_rows(
    rows: list[dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    rows = [row for row in rows if int(row.get("length", 0)) >= 8000] or rows
    if len(rows) <= limit:
        return rows
    if limit == 1:
        return [rows[len(rows) // 2]]
    last = len(rows) - 1
    return [rows[round(index * last / (limit - 1))] for index in range(limit)]


def _run_longbench(
    *,
    tokenizer: Any,
    host: str,
    port: int,
    model: str,
    data_dir: Path,
    longbench_root: Path,
    datasets: list[str],
    limit: int,
    max_input_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    config_dir = longbench_root / "config"
    prompt_formats = json.loads(
        (config_dir / "dataset2prompt.json").read_text(encoding="utf-8")
    )
    max_output_tokens = json.loads(
        (config_dir / "dataset2maxlen.json").read_text(encoding="utf-8")
    )
    sys.path.insert(0, str(longbench_root.resolve()))
    try:
        from eval import dataset2metric  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "LongBench metric dependencies are missing; install rouge, jieba, "
            "and fuzzywuzzy into PYTHONPATH"
        ) from exc

    by_dataset: dict[str, Any] = {}
    for dataset in datasets:
        rows = []
        for source_index, line in enumerate(
            (data_dir / f"{dataset}.jsonl").read_text(encoding="utf-8").splitlines()
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_source_index"] = source_index
            rows.append(row)
        selected = _select_longbench_rows(rows, limit)
        records: list[dict[str, Any]] = []
        for row in selected:
            prompt = prompt_formats[dataset].format(**row)
            prompt = _truncate_middle(tokenizer, prompt, max_input_tokens)
            if dataset in NO_CHAT_DATASETS:
                prediction, raw = _completion(
                    host=host,
                    port=port,
                    model=model,
                    prompt=prompt,
                    max_tokens=int(max_output_tokens[dataset]),
                    timeout=timeout,
                )
            else:
                prediction, raw = _chat(
                    host=host,
                    port=port,
                    model=model,
                    content=prompt,
                    max_tokens=int(max_output_tokens[dataset]),
                    timeout=timeout,
                )
            score = max(
                float(
                    dataset2metric[dataset](
                        prediction,
                        answer,
                        all_classes=row.get("all_classes", []),
                    )
                )
                for answer in row["answers"]
            )
            records.append(
                {
                    "source_index": row["_source_index"],
                    "length": row.get("length"),
                    "score": score,
                    "prediction": prediction,
                    "answers": row["answers"],
                    "usage": raw.get("usage"),
                }
            )
        scores = [float(record["score"]) for record in records]
        by_dataset[dataset] = {
            "samples": len(records),
            "score": 100.0 * statistics.mean(scores) if scores else None,
            "records": records,
        }
    dataset_scores = [
        float(result["score"])
        for result in by_dataset.values()
        if result["score"] is not None
    ]
    return {
        "datasets": by_dataset,
        "average_score": statistics.mean(dataset_scores) if dataset_scores else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--humaneval-tasks",
        type=Path,
        default=Path(
            "bench_results/ddtree_humaneval_sweep_20260625/humaneval_tasks.json"
        ),
    )
    parser.add_argument("--humaneval-limit", type=int, default=32)
    parser.add_argument(
        "--longbench-data-dir",
        type=Path,
        default=Path("benchmark-data/longbench/data"),
    )
    parser.add_argument(
        "--longbench-root",
        type=Path,
        default=Path("third_party/LongBench/LongBench"),
    )
    parser.add_argument(
        "--longbench-datasets",
        default="hotpotqa,multifieldqa_zh,gov_report,lcc",
    )
    parser.add_argument("--longbench-limit", type=int, default=4)
    parser.add_argument("--longbench-max-input-tokens", type=int, default=32768)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--min-humaneval-passes", type=int, default=0)
    parser.add_argument("--min-longbench-average", type=float, default=0.0)
    args = parser.parse_args()

    parsed = urlparse(args.base_url)
    if parsed.scheme != "http" or not parsed.hostname:
        raise ValueError("--base-url must be an http URL")
    host = parsed.hostname
    port = parsed.port or 80
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path),
        local_files_only=True,
        trust_remote_code=True,
    )

    datasets = [
        value.strip() for value in args.longbench_datasets.split(",") if value.strip()
    ]
    input_manifest = {
        "humaneval_tasks": {
            "path": str(args.humaneval_tasks),
            "sha256": _sha256(args.humaneval_tasks),
        },
        "longbench_prompt_config": {
            "path": str(args.longbench_root / "config/dataset2prompt.json"),
            "sha256": _sha256(args.longbench_root / "config/dataset2prompt.json"),
        },
        "longbench_maxlen_config": {
            "path": str(args.longbench_root / "config/dataset2maxlen.json"),
            "sha256": _sha256(args.longbench_root / "config/dataset2maxlen.json"),
        },
        "longbench_datasets": {
            dataset: {
                "path": str(args.longbench_data_dir / f"{dataset}.jsonl"),
                "sha256": _sha256(args.longbench_data_dir / f"{dataset}.jsonl"),
            }
            for dataset in datasets
        },
    }

    human_eval = _run_humaneval(
        host=host,
        port=port,
        model=args.model,
        tasks_path=args.humaneval_tasks,
        limit=args.humaneval_limit,
        timeout=args.timeout,
    )
    longbench = _run_longbench(
        tokenizer=tokenizer,
        host=host,
        port=port,
        model=args.model,
        data_dir=args.longbench_data_dir,
        longbench_root=args.longbench_root,
        datasets=datasets,
        limit=args.longbench_limit,
        max_input_tokens=args.longbench_max_input_tokens,
        timeout=args.timeout,
    )
    passed = bool(
        human_eval["passed"] >= args.min_humaneval_passes
        and longbench["average_score"] is not None
        and longbench["average_score"] >= args.min_longbench_average
    )
    result = {
        "contract": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "evaluation_contract": {
            "version": "dsv4_api_quality_v1",
            "strictly_sequential": True,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 20260824,
            "human_eval": {
                "selection": "first_n",
                "limit": args.humaneval_limit,
                "endpoint": "chat_completions",
                "enable_thinking": False,
                "max_tokens": 768,
                "execution": "landlock_seccomp_python_isolated_v1",
            },
            "longbench": {
                "datasets": datasets,
                "selection": "length_ge_8000_evenly_spaced_v1",
                "limit_per_dataset": args.longbench_limit,
                "max_input_tokens": args.longbench_max_input_tokens,
                "truncation": "middle_first_half_last_half_v1",
                "endpoint_policy": "official_no_chat_dataset_set_v1",
                "max_output_tokens": "dataset2maxlen_sha256_manifest",
            },
        },
        "input_manifest": input_manifest,
        "human_eval": human_eval,
        "longbench": longbench,
        "passed": passed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "human_eval_passes": human_eval["passed"],
                "human_eval_samples": human_eval["samples"],
                "longbench_average": longbench["average_score"],
                "passed": passed,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
