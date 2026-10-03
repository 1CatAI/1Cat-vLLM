# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Freeze teacher-forcing windows, capture raw logits, or compare saved arms."""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from benchmarks.benchmark_sm70_qwen38_quality import prompt_token_ids
from benchmarks.qwen38_distribution_probe import (
    distribution_metrics,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    suite = json.loads(args.cases.read_text())
    quality = json.loads(args.quality_reference.read_text())
    reference = {r["id"]: r for r in quality["quality"]}
    probes = []
    for case in suite["cases"]:
        ids = prompt_token_ids(case, tokenizer)
        record = reference[case["id"]]
        if (
            hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            != record["prompt_token_sha256"]
        ):
            raise ValueError(
                f"Prompt differs from frozen quality reference: {case['id']}"
            )
        continuation = record["token_ids"]
        for start in sorted({0, max(0, len(continuation) - 16)}):
            probes.append(
                {
                    "id": f"{case['id']}-{start}",
                    "category": case["category"],
                    "prompt_token_ids": ids + continuation[:start],
                    "continuation": continuation[start : start + 16],
                    "reference_kind": "recorded_default_fp16",
                }
            )
    english = [
        "Explain why a checksum helps detect a corrupted file.",
        "Describe the difference between observation and inference.",
        "Give a clear introduction to sorting a list of numbers.",
        "Explain how to organize a short technical report.",
    ]
    authored = tokenizer.encode(
        "A useful approach is to break the problem into small steps, "
        "check assumptions, and test the result.",
        add_special_tokens=False,
    )[:16]
    for i, text in enumerate(english):
        probes.append(
            {
                "id": f"english-{i}",
                "category": "english",
                "prompt_token_ids": tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=True,
                    return_dict=False,
                ),
                "continuation": authored,
                "reference_kind": "authored_teacher_forcing",
            }
        )
    model_config = json.loads((args.model / "config.json").read_text())
    text_config = model_config.get("text_config", model_config)
    manifest = {
        "vocabulary": text_config["vocab_size"],
        "suite_sha256": digest(args.cases),
        "quality_reference_sha256": digest(args.quality_reference),
        "probes": probes,
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False) + "\n")
    print("FROZEN", len(probes), digest(args.manifest))


def capture(args):
    if args.disable_kernel:
        os.environ["VLLM_DISABLED_KERNELS"] = ",".join(args.disable_kernel)
    import torch

    import vllm
    from vllm import LLM, SamplingParams

    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    manifest = json.loads(args.manifest.read_text())
    widths = [int(w) for w in args.widths.split(",")]
    args.output.mkdir(parents=True, exist_ok=False)
    engine_config = dict(
        model=str(args.model),
        tensor_parallel_size=4,
        dtype="half",
        mamba_ssm_cache_dtype="float32",
        kv_cache_dtype="float16",
        max_model_len=262144,
        max_num_batched_tokens=args.prefill_budget,
        max_num_seqs=args.max_num_seqs or max(widths),
        gpu_memory_utilization=0.94,
        enable_prefix_caching=False,
        language_model_only=True,
        speculative_config=None,
        kernel_config={"ple_result_transport": args.transport},
        worker_extension_cls=(
            "benchmarks.qwen38_distribution_probe.DistributionProbeWorkerExtension"
        ),
        disable_log_stats=False,
    )
    if args.worker_cls:
        engine_config["worker_cls"] = args.worker_cls
    if args.kv_cache_memory_bytes is not None:
        engine_config["kv_cache_memory_bytes"] = args.kv_cache_memory_bytes
    llm = LLM(**engine_config)
    import vllm._C as native

    report = {
        "complete": False,
        "diagnostic_only": True,
        "speed_acceptance": False,
        "manifest_sha256": digest(args.manifest),
        "runtime": vllm.__version__,
        "disabled_kernels": args.disable_kernel,
        "runtime_path": vllm.__file__,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "transport": args.transport,
        "widths": widths,
        "repeats": args.repeats,
        "contract": "TP4 FP16 dense/KV, FP32 accum/state, disk mmap, no MTP",
        "engine": {k: v for k, v in engine_config.items() if k != "kernel_config"},
        "native_sha256": digest(Path(native.__file__)),
        "captures": [],
    }

    def save():
        (args.output / "capture.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    filler = next(p for p in manifest["probes"] if p["category"] == "english")
    try:
        for width in widths:
            for repeat in range(args.repeats):
                for case in manifest["probes"]:
                    count = len(case["continuation"])
                    batch = [case] + [filler] * (width - 1)
                    destinations = [args.output / str(width) / str(repeat) / case["id"]]
                    params, specs = [], []
                    for i, probe in enumerate(batch):
                        length = count if i == 0 else max(count, 64)
                        continuation = (probe["continuation"] * (length + 1))[:length]
                        params.append(
                            SamplingParams(
                                temperature=0,
                                ignore_eos=True,
                                max_tokens=length,
                                logprobs=1,
                            )
                        )
                        specs.append(
                            {
                                "continuation": continuation,
                                "prompt_token_ids": probe["prompt_token_ids"],
                                "vocabulary": manifest["vocabulary"],
                                "capture": i == 0,
                                "output": str(destinations[0]),
                            }
                        )
                    core = llm.llm_engine.engine_core
                    core.call_utility("pause_scheduler", "keep", False)
                    try:
                        request_ids = llm.enqueue(
                            [
                                {"prompt_token_ids": p["prompt_token_ids"]}
                                for p in batch
                            ],
                            params,
                            use_tqdm=False,
                        )
                        llm.collective_rpc(
                            "configure_distribution_probe",
                            timeout=30,
                            kwargs={
                                "requests": dict(zip(request_ids, specs, strict=True))
                            },
                        )
                    finally:
                        core.call_utility("resume_scheduler")
                    outputs = llm.wait_for_completion(use_tqdm=False)
                    if len(outputs) != width or any(
                        len(o.outputs[0].token_ids) != params[i].max_tokens
                        for i, o in enumerate(outputs)
                    ):
                        raise RuntimeError("Incomplete teacher-forcing cohort")
                    if len(list(destinations[0].glob("*.npy"))) != count:
                        raise RuntimeError("Missing teacher-forcing logit row")
                    if any(
                        json.loads(p.read_text())["active_width"] != width
                        for p in destinations[0].glob("*.json")
                    ):
                        raise RuntimeError(
                            "Teacher-forcing cohort lost its active width"
                        )
                    report["captures"].append(
                        {
                            "width": width,
                            "repeat": repeat,
                            "id": case["id"],
                            "category": case["category"],
                            "rows": count,
                        }
                    )
                    save()
                    print("CAPTURE", width, repeat, case["id"], flush=True)
        report["complete"] = True
        save()
    finally:
        llm.llm_engine.engine_core.shutdown()


def summarize_groups(rows, widths):
    groups = {}
    for width in widths:
        for category in ["all"] + sorted({r["category"] for r in rows}):
            selected = [
                r
                for r in rows
                if r["width"] == width
                and (category == "all" or r["category"] == category)
            ]
            kl = [r["kl"] for r in selected]
            summary = {
                "rows": len(selected),
                "mean_kl": float(np.mean(kl)),
                "p99_kl": float(np.quantile(kl, 0.99)),
                "max_kl": max(kl),
                "top1_agreement": float(
                    np.mean([r["top1_agreement"] for r in selected])
                ),
                "max_logit_error": max(r["max_logit_error"] for r in selected),
                "median_logit_error": float(
                    np.median([r["max_logit_error"] for r in selected])
                ),
                "p95_logit_error": float(
                    np.quantile([r["max_logit_error"] for r in selected], 0.95)
                ),
                "p99_logit_error": float(
                    np.quantile([r["max_logit_error"] for r in selected], 0.99)
                ),
                "centered_max_logit_error": max(
                    r["centered_max_logit_error"] for r in selected
                ),
                "top1_disagreements": sum(not r["top1_agreement"] for r in selected),
                "mean_reverse_kl": float(np.mean([r["reverse_kl"] for r in selected])),
            }
            summary["passed"] = (
                summary["mean_kl"] <= 0.001
                and summary["p99_kl"] <= 0.01
                and summary["max_kl"] <= 0.05
                and summary["top1_agreement"] >= 0.99
                and summary["max_logit_error"] <= 0.5
            )
            groups[f"C{width}/{category}"] = summary
    return groups


def compare(args):
    ref = json.loads((args.reference / "capture.json").read_text())
    cand = json.loads((args.output / "capture.json").read_text())
    if not ref["complete"] or not cand["complete"]:
        raise ValueError("Both capture arms must be complete")
    if ref["manifest_sha256"] != cand["manifest_sha256"]:
        raise ValueError("Teacher-forcing manifests differ")
    for field in ("captures", "widths", "repeats", "contract", "engine"):
        if ref[field] != cand[field]:
            raise ValueError(f"Capture contract differs: {field}")
    rows = []
    for cohort in cand["captures"]:
        relative = Path(str(cohort["width"])) / str(cohort["repeat"]) / cohort["id"]
        for step in range(cohort["rows"]):
            filename = f"{step:04d}"
            metadata = json.loads(
                (args.output / relative / (filename + ".json")).read_text()
            )
            expected = json.loads(
                (args.reference / relative / (filename + ".json")).read_text()
            )
            if metadata != expected:
                raise ValueError("Probe prefix, vocabulary or next-token IDs differ")
            if metadata["active_width"] != cohort["width"]:
                raise ValueError(
                    "Teacher-forcing probe did not sustain its active width"
                )
            metrics = distribution_metrics(
                np.load(args.reference / relative / (filename + ".npy")),
                np.load(args.output / relative / (filename + ".npy")),
            )
            rows.append({**cohort, "step": step, **metrics})
    groups = summarize_groups(rows, cand["widths"])
    noise_rows = []
    for cohort in ref["captures"]:
        if cohort["repeat"] == 0:
            continue
        first = Path(str(cohort["width"])) / "0" / cohort["id"]
        other = Path(str(cohort["width"])) / str(cohort["repeat"]) / cohort["id"]
        for step in range(cohort["rows"]):
            filename = f"{step:04d}"
            left = args.reference / first / filename
            right = args.reference / other / filename
            if json.loads(left.with_suffix(".json").read_text()) != json.loads(
                right.with_suffix(".json").read_text()
            ):
                raise ValueError("Default repeats have different probe prefixes")
            noise_rows.append(
                {
                    **cohort,
                    "step": step,
                    **distribution_metrics(
                        np.load(left.with_suffix(".npy")),
                        np.load(right.with_suffix(".npy")),
                    ),
                }
            )
    noise = summarize_groups(noise_rows, ref["widths"]) if noise_rows else {}
    result = {
        "distribution_passed": all(g["passed"] for g in groups.values()),
        "default_noise_passed": (
            all(g["passed"] for g in noise.values()) if noise else None
        ),
        "default_noise": noise,
        "quality_acceptance": "requires separate fixed task suite",
        "groups": groups,
        "rows": rows,
    }
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(groups, indent=2))
    if not result["distribution_passed"] or result["default_noise_passed"] is False:
        raise SystemExit("Distribution thresholds exceeded")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "capture", "compare"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--quality-reference", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument(
        "--transport", choices=("auto", "cuda", "mapped"), default="auto"
    )
    parser.add_argument("--widths", default="1")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--prefill-budget", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--kv-cache-memory-bytes", type=int)
    parser.add_argument(
        "--disable-kernel",
        action="append",
        default=[],
        help="Use the existing disabled-kernel control for matched A/B",
    )
    parser.add_argument("--worker-cls", help="Qualified diagnostic worker class")
    args = parser.parse_args()
    if args.action == "capture":
        while True:
            with open("/tmp/gpu0-3.lock", "a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
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
                if not consumers:
                    capture(args)
                    break
            print("GPU 0-3 occupied; released lock and waiting", flush=True)
            time.sleep(10)
    else:
        {"prepare": prepare, "compare": compare}[args.action](args)
