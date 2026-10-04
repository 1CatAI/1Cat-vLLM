# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run a frozen no-MTP FP16 TP4 decode timing and natural-EOS quality gate.

Pass the retained cases JSON (MBPP/GSM8K task selections, Chinese questions,
needle lengths, and data hashes). Output includes full text/token IDs and
per-case scores; this is a small regression gate, not a complete benchmark.
The benchmark clears inherited route overrides and uses disk-backed ngrams.
"""

import argparse
import ast
import fcntl
import hashlib
import json
import os
import resource
import shutil
import statistics
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

import regex as re

REQUEST_METRICS_ENABLED = True


def final_text(text):
    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1].strip()
    return text.strip()


def check(case, text):
    body = final_text(text)
    category = case["category"]
    if category == "mbpp":
        fenced = re.findall(r"```(?:python|py)?\s*\n(.*?)```", body, re.S)
        code = max(fenced, key=len) if fenced else body
        try:
            ast.parse(code)
        except SyntaxError:
            return {"passed": False, "reason": "syntax"}
        program = case.get("setup", "") + "\n" + code + "\n" + "\n".join(case["tests"])

        def limits():
            resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
            resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
            resource.setrlimit(resource.RLIMIT_FSIZE, (1024**2, 1024**2))

        with tempfile.TemporaryDirectory(prefix="nomtp-mbpp-") as d:
            source = Path(d) / "program.py"
            source.write_text(program)
            try:
                r = subprocess.run(
                    [sys.executable, "-I", "-S", str(source)],
                    cwd=d,
                    env={"PATH": "/usr/bin:/bin"},
                    capture_output=True,
                    text=True,
                    timeout=8,
                    preexec_fn=limits,
                )
            except subprocess.TimeoutExpired:
                return {"passed": False, "reason": "timeout"}
        return {
            "passed": r.returncode == 0,
            "reason": r.stderr[-1200:] if r.returncode else None,
        }
    if category == "gsm8k":
        matches = re.findall(r"####\s*([-+]?\d[\d,.]*)", body)
        if not matches:
            matches = re.findall(r"[-+]?\d[\d,.]*", body)
        answer = matches[-1].replace(",", "").rstrip(".") if matches else None
        return {
            "passed": answer == case["answer"],
            "answer": answer,
            "expected": case["answer"],
        }
    if category == "chinese":
        return {"passed": any(s in body for s in case["any_expected"])}
    return {"passed": case["answer"] in body}


def metrics(o):
    m = o.metrics
    r = o.outputs[0]
    n = len(r.token_ids)
    return {
        "output_tokens": n,
        "ttft_s": m.first_token_latency,
        "prefill_s": m.first_token_ts - m.scheduled_ts,
        "queued_s": m.scheduled_ts - m.queued_ts,
        "decode_s": m.last_token_ts - m.first_token_ts,
        "tpot_ms": 1000 * (m.last_token_ts - m.first_token_ts) / (n - 1)
        if n > 1
        else None,
        "text": r.text,
        "token_ids": list(r.token_ids),
        "finish_reason": r.finish_reason,
    }


def health_failures(records):
    """Screen natural-EOS outputs separately from task scores.

    Three occurrences of a long identical final-answer line require review;
    a passing code test cannot override this or a token-limit termination.
    """
    failures = []
    for record in records:
        health = record["health"]
        reasons = []
        if not health["natural_eos"]:
            reasons.append("not_natural_eos")
        if not health["nonempty_final"]:
            reasons.append("empty_final_answer")
        if health["replacement_characters"]:
            reasons.append("replacement_characters")
        if health["line_repetition"] >= 3:
            reasons.append("repeated_final_answer_line")
        if reasons:
            failures.append({"id": record["id"], "reasons": reasons})
    return failures


def prompt_token_ids(case, tok):
    prompt = case.get("prompt")
    if case["category"] == "needle":
        filler = tok.encode(
            "This is an irrelevant archived note. No key is recorded in this line.\n",
            add_special_tokens=False,
        )
        body = (filler * ((case["target_tokens"] + len(filler) - 1) // len(filler)))[
            : case["target_tokens"] - 200
        ]
        at = int(len(body) * case["depth"])
        needle = tok.encode(
            "\n唯一的密钥是：" + case["answer"] + "。\n",
            add_special_tokens=False,
        )
        body = body[:at] + needle + body[at:]
        prompt = (
            "请从以下档案找出唯一密钥。\n" + tok.decode(body) + "\n请原样输出密钥。"
        )
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
        return_dict=False,
    )


def run(args):
    out = args.output.parent
    out.mkdir(parents=True, exist_ok=True)
    model = str(args.model)
    suite = (
        json.loads(args.cases.read_text())
        if args.cases is not None
        else {"sampling": {}, "cases": []}
    )
    import torch
    from transformers import AutoTokenizer

    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    import vllm
    from vllm import LLM, SamplingParams

    report = {
        "complete": False,
        "runtime": vllm.__version__,
        "runtime_path": vllm.__file__,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "source_native": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(vllm.__file__).parent.glob("*.so")
        },
        "suite_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest()
        if args.cases is not None
        else None,
        "quality_evaluated": not args.timing_only,
        "quality_seed_base": args.quality_seed,
        "quality_case_ids": args.case_id,
        "sampling": suite["sampling"],
        "contract": {
            "model": model,
            "tp": 4,
            "dtype": "float16",
            "kv": "float16",
            "max_len": 262144,
            "max_num_seqs": 1,
            "budget": 8192,
            "mtp": False,
            "prefix": False,
            "ngram": "disk mmap, no pinned whole-table allocation",
        },
        "timing": [],
        "quality": [],
    }

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    save()
    tok = AutoTokenizer.from_pretrained(model)
    llm = LLM(
        model=model,
        tensor_parallel_size=4,
        dtype="half",
        kv_cache_dtype="float16",
        mamba_ssm_cache_dtype="float32",
        max_model_len=262144,
        max_num_batched_tokens=8192,
        max_num_seqs=1,
        gpu_memory_utilization=0.94,
        enable_prefix_caching=False,
        language_model_only=True,
        speculative_config=None,
        disable_log_stats=not REQUEST_METRICS_ENABLED,
        kernel_config={
            "ple_result_transport": args.ple_result_transport,
            "ple_disk_row_gather": not args.disable_ple_row_gather,
        },
    )
    try:
        cfg = llm.llm_engine.vllm_config
        report["resolved"] = {
            "graph": str(cfg.compilation_config.cudagraph_mode),
            "speculative": str(cfg.speculative_config),
            "acceleration": getattr(cfg, "sm70_acceleration_report", None),
        }
        report["worker_routes"] = llm.collective_rpc(
            "get_sm70_acceleration_report", timeout=30
        )
        save()
        if not args.quality_only:
            chunk = tok.encode(
                "This fixed benchmark prompt is used to create a deterministic "
                "tokenized input for single-request decode measurement. ",
                add_special_tokens=False,
            )
            ids = (chunk * ((8192 + len(chunk) - 1) // len(chunk)))[:8192]
            llm.generate(
                [{"prompt_token_ids": ids}],
                SamplingParams(temperature=0, max_tokens=32, ignore_eos=True),
                use_tqdm=False,
            )
            for i in range(6):
                o = llm.generate(
                    [{"prompt_token_ids": ids}],
                    SamplingParams(
                        temperature=0,
                        top_p=1,
                        top_k=-1,
                        seed=0,
                        max_tokens=513,
                        ignore_eos=True,
                    ),
                    use_tqdm=False,
                )[0]
                report["timing"].append(metrics(o))
                save()
                print("TIMING", i, report["timing"][-1]["tpot_ms"], flush=True)
        for i, case in enumerate([] if args.timing_only else suite["cases"]):
            if args.case_id and case["id"] not in args.case_id:
                continue
            prompt_ids = prompt_token_ids(case, tok)
            o = llm.generate(
                [{"prompt_token_ids": prompt_ids}],
                SamplingParams(
                    temperature=1,
                    top_p=0.95,
                    top_k=20,
                    seed=args.quality_seed + i,
                    max_tokens=4096,
                ),
                use_tqdm=False,
            )[0]
            record = {
                "id": case["id"],
                "category": case["category"],
                "seed": args.quality_seed + i,
                "input_tokens": len(prompt_ids),
                "prompt_token_sha256": hashlib.sha256(
                    json.dumps(prompt_ids).encode()
                ).hexdigest(),
                **metrics(o),
            }
            text = final_text(record["text"])
            lines = [s.strip() for s in text.splitlines() if len(s.strip()) > 24]
            record["health"] = {
                "natural_eos": record["finish_reason"] == "stop",
                "nonempty_final": bool(text),
                "replacement_characters": text.count("\ufffd"),
                "line_repetition": max((lines.count(s) for s in set(lines)), default=0),
            }
            record["score"] = check(case, record["text"])
            report["quality"].append(record)
            save()
            print("QUALITY", case["id"], record["score"], record["health"], flush=True)
        report["summary"] = {
            category: {
                "passed": sum(
                    r["score"]["passed"]
                    for r in report["quality"]
                    if r["category"] == category
                ),
                "total": sum(r["category"] == category for r in report["quality"]),
            }
            for category in ("mbpp", "gsm8k", "chinese", "needle")
        }
        report["median_tpot_ms"] = (
            statistics.median(r["tpot_ms"] for r in report["timing"])
            if report["timing"]
            else None
        )
        report["health_failures"] = health_failures(report["quality"])
        report["health_passed"] = (
            not report["health_failures"] if not args.timing_only else None
        )
        report["complete"] = True
        report["health_triage_required"] = bool(report["health_failures"])
        save()
        if report["health_passed"] is False:
            raise SystemExit(
                "Output anomaly requires three candidate and three baseline seeds; "
                "see report"
            )
    finally:
        llm.llm_engine.engine_core.shutdown()


def compare_timing(reference, candidate, expected_saving_ms):
    """Check the matched contract before calibrating an endpoint estimate."""
    for key in ("runtime", "torch", "cuda", "source_native", "contract"):
        if reference[key] != candidate[key]:
            raise ValueError(f"Timing arms differ in {key}")
    if not reference["complete"] or not candidate["complete"]:
        raise ValueError("Timing arms must be complete")
    baseline = reference["median_tpot_ms"]
    observed = baseline - candidate["median_tpot_ms"]
    error = abs(observed - expected_saving_ms) / expected_saving_ms
    return {
        "baseline_ms": baseline,
        "candidate_ms": candidate["median_tpot_ms"],
        "expected_saving_ms": expected_saving_ms,
        "observed_saving_ms": observed,
        "estimate_relative_error": error,
        "calibration_required": error > 0.15,
    }


def preflight(args):
    """Reject incomplete launch contracts before importing/loading the model."""
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("The frozen runtime requires Python 3.12")
    header = Path(sysconfig.get_path("include")) / "Python.h"
    if not header.is_file():
        raise RuntimeError(f"Python development header is missing: {header}")
    if not REQUEST_METRICS_ENABLED:
        raise RuntimeError("Request metrics must be enabled")
    tree = ast.parse(Path(__file__).read_text())
    guard = ast.parse('if __name__ == "__main__": main()').body[0].test
    if not any(
        isinstance(node, ast.If) and ast.dump(node.test) == ast.dump(guard)
        for node in tree.body
    ):
        raise RuntimeError("The endpoint entry requires a spawn main guard")
    if not (args.model / "config.json").is_file():
        raise RuntimeError("Model config is missing")
    index = args.model / "model.safetensors.index.json"
    if not index.is_file():
        raise RuntimeError("The sharded checkpoint index is missing")
    shards = set(json.loads(index.read_text())["weight_map"].values())
    missing = [name for name in sorted(shards) if not (args.model / name).is_file()]
    if missing:
        raise RuntimeError(f"Checkpoint shards missing: {missing[:3]}")
    if not args.timing_only:
        if args.cases is None or not args.cases.is_file():
            raise RuntimeError("Quality mode requires the frozen cases JSON")
        suite = json.loads(args.cases.read_text())
        known = {case["id"] for case in suite["cases"]}
        if set(args.case_id or []) - known:
            raise RuntimeError("Unknown quality case IDs")
    if args.output.exists():
        raise RuntimeError("Refusing to overwrite a retained result")
    free = shutil.disk_usage(args.output.parent).free
    if free < args.min_free_gib * 1024**3:
        raise RuntimeError(f"Insufficient disk: {free / 1024**3:.2f} GiB free")
    consumers = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            "0,1,2,3",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    if consumers:
        raise RuntimeError(f"GPU 0-3 occupied by PIDs: {consumers}")
    return {
        "python": sys.version,
        "header": str(header),
        "free_bytes": free,
        "gpu_idle": True,
        "request_metrics": True,
        "spawn_guard": True,
        "gpu_lock": "/tmp/gpu0-3.lock",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cases", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--timing-only", action="store_true")
    modes.add_argument("--quality-only", action="store_true")
    parser.add_argument("--quality-seed", type=int, default=4201)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-free-gib", type=float, default=8)
    parser.add_argument("--reference-timing", type=Path)
    parser.add_argument("--expected-saving-ms", type=float)
    parser.add_argument("--disable-ple-row-gather", action="store_true")
    parser.add_argument(
        "--ple-result-transport", choices=("auto", "cuda", "mapped"), default="auto"
    )
    args = parser.parse_args()
    if args.min_free_gib < 8:
        parser.error("--min-free-gib must be at least 8")
    if args.reference_timing is not None:
        if args.quality_only:
            parser.error("--reference-timing requires timing requests")
        if args.expected_saving_ms is None or args.expected_saving_ms <= 0:
            parser.error("--reference-timing requires positive --expected-saving-ms")
        reference = json.loads(args.reference_timing.read_text())
    else:
        reference = None
    args.model = args.model.resolve()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Resolve defaults in a fresh process, before importing the runtime.
    for key in list(os.environ):
        if key.startswith(
            ("VLLM_", "TRITON_", "TORCHINDUCTOR_", "FLASH_QLA_", "ONECAT_", "SM70_")
        ) or key in ("PYTHONPATH", "LD_PRELOAD", "LD_LIBRARY_PATH"):
            del os.environ[key]
    cache = args.output.parent / "runtime"
    os.environ.update(
        CUDA_VISIBLE_DEVICES="0,1,2,3",
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        OMP_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
        VLLM_SM70_QWEN38_HYBRID_PLE="0",
        VLLM_PLE_CPU_OFFLOAD="1",
        VLLM_PLE_DISK_OFFLOAD="1",
        VLLM_CACHE_ROOT=str(cache / "vllm"),
        TRITON_CACHE_DIR=str(cache / "triton"),
        TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
        TORCH_EXTENSIONS_DIR=str(cache / "extensions"),
    )
    summary = {"complete": False, "output": str(args.output)}
    summary_path = args.output.with_suffix(".summary.json")
    try:
        with open("/tmp/gpu0-3.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            summary["preflight"] = preflight(args)
            run(args)
            report = json.loads(args.output.read_text())
            if reference is not None:
                summary["estimate_comparison"] = compare_timing(
                    reference, report, args.expected_saving_ms
                )
            summary.update(
                complete=report["complete"],
                median_tpot_ms=report["median_tpot_ms"],
                quality_evaluated=report["quality_evaluated"],
                health_triage_required=report["health_triage_required"],
                worker_routes=report["worker_routes"],
            )
    except BaseException as error:
        summary["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        )
        print("COMPLETION_SUMMARY", summary_path, flush=True)


if __name__ == "__main__":
    main()
