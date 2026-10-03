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
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import regex as re


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
    suite = json.loads(args.cases.read_text())
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
        "suite_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
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
        disable_log_stats=False,
        kernel_config={
            "ple_result_transport": args.ple_result_transport,
            **(
                {"ple_sample_prefetch": args.ple_sample_prefetch}
                if args.ple_sample_prefetch is not None
                else {}
            ),
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
        for i, case in enumerate(suite["cases"]):
            prompt_ids = prompt_token_ids(case, tok)
            o = llm.generate(
                [{"prompt_token_ids": prompt_ids}],
                SamplingParams(
                    temperature=1, top_p=0.95, top_k=20, seed=4201 + i, max_tokens=4096
                ),
                use_tqdm=False,
            )[0]
            record = {
                "id": case["id"],
                "category": case["category"],
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
        report["median_tpot_ms"] = statistics.median(
            r["tpot_ms"] for r in report["timing"]
        )
        report["complete"] = True
        save()
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--ple-result-transport", choices=("auto", "cuda", "mapped"), default="auto"
    )
    parser.add_argument(
        "--ple-sample-prefetch", action=argparse.BooleanOptionalAction, default=None
    )
    args = parser.parse_args()
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
    with open("/tmp/gpu0-3.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        import subprocess

        if subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                "0,1,2,3",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip():
            raise SystemExit("GPU 0-3 have other consumers; gate not launched")
        run(args)
