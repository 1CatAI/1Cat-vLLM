# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reproduce the quality-repaired no-MTP TP4 baseline with freshly built overlays.

Obtain exclusive GPU ownership before running. Normal output checks use the
checkpoint's sampling configuration; forced-length greedy timing is separate.
All model workers shut down after the bounded test. No API service is started.
"""

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

import regex as re

# Support both direct script execution and `python -m benchmarks...`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.sm70_qwen38_baseline import (  # noqa: E402
    FIXED_PROMPT,
    ROOT,
    configure_environment,
    engine_args,
    validate_bundle,
)


def run(args):
    from transformers import AutoTokenizer

    from benchmarks.benchmark_sm70_decode import _hash_ids, _run_once, _summarize
    from vllm import LLM, SamplingParams

    bundle = validate_bundle(args.runtime_dir)
    use_defaults = getattr(args, "use_defaults", False)
    model_args = engine_args(args.model, use_defaults=use_defaults)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    config = json.loads((Path(args.model) / "generation_config.json").read_text())
    natural = SamplingParams(
        max_tokens=513,
        temperature=config["temperature"],
        top_p=config["top_p"],
        top_k=config["top_k"],
        seed=0,
        ignore_eos=False,
        skip_special_tokens=False,
    )
    fixed = SamplingParams(
        max_tokens=513,
        temperature=0,
        seed=0,
        ignore_eos=True,
        skip_special_tokens=False,
    )
    piece = tokenizer.encode(FIXED_PROMPT, add_special_tokens=False)
    lengths = [8192, 261631] if args.long_context else [8192]
    reference = {}
    if args.reference_json:
        previous = json.loads(args.reference_json.read_text())
        reference = {c["name"]: c["runs"][0]["token_ids"] for c in previous["cases"]}
        if not {f"fixed_{n}" for n in lengths}.issubset(reference):
            raise ValueError("Reference is missing a requested fixed-prompt case")
    report = {
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "bundle": bundle,
        "contract": model_args,
        "use_defaults": use_defaults,
        "launch_vllm_env": {
            k: v for k, v in os.environ.items() if k.startswith("VLLM_")
        },
        "gpu_group": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "started_unix": time.time(),
        "health": [],
        "cases": [],
        "complete": False,
        "scope": (
            "Warm single-request fixed-prompt benchmark; forced post-EOS timing "
            "is not quality evidence. Natural health/boundary checks are separate."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix(".tmp.json")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(args.output)

    def health(llm, name, ids, pattern):
        result = _run_once(llm, {"prompt_token_ids": ids}, natural)
        passed = result["finish_reason"] == "stop" and bool(
            re.search(pattern, result["text"].rsplit("</think>", 1)[-1])
        )
        report["health"].append(
            {"name": name, "input_tokens": len(ids), "passed": passed, "result": result}
        )
        save()
        print(json.dumps({"health": name, "passed": passed}), flush=True)
        if not passed:
            raise RuntimeError(f"Natural output health failed: {name}")

    llm = None
    try:
        save()
        llm = LLM(**model_args)
        for name, prompt, pattern in (
            (
                "arithmetic",
                "请计算 19 × 23。请在最后一行写 RESULT=计算结果。",
                r"RESULT\s*=\s*437\b",
            ),
            (
                "record_copy",
                "项目代号 CEDAR-47，编号 8261。最后一行准确输出 CEDAR-47|8261。",
                r"CEDAR-47\s*\|\s*8261",
            ),
        ):
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            health(llm, name, ids, pattern)
        for length in lengths:
            ids = (piece * ((length + len(piece) - 1) // len(piece)))[:length]
            case = {
                "name": f"fixed_{length}",
                "input_tokens": length,
                "prompt_hash": _hash_ids(ids),
                "runs": [],
            }
            report["cases"].append(case)
            case["warmup"] = _run_once(llm, {"prompt_token_ids": ids}, fixed)
            if "workers" not in report:
                workers = llm.collective_rpc("baseline_manifest", timeout=90)
                report["workers"] = workers
                if len(workers) != 4 or sorted(w["rank"] for w in workers) != list(
                    range(4)
                ):
                    raise RuntimeError("Missing or duplicate TP worker manifests")
                for worker in workers:
                    if (
                        worker["ssm_dtype"] != "float32"
                        or worker["mtp"]
                        or worker["prefix_cache"]
                    ):
                        raise RuntimeError(
                            f"Worker precision/state contract mismatch: {worker}"
                        )
                    if (
                        worker["max_model_len"] != 262144
                        or "FULL" not in worker["graph_mode"]
                        or worker["qsa_specialization_version"] != 1
                        or set(worker["binaries"])
                        != {"hc", "qpn", "qsa", "flashqla", "flashv100"}
                        or not worker["native_dependencies"]
                    ):
                        raise RuntimeError(
                            "Worker route or dependency contract mismatch"
                        )
                    if Path(worker["vllm_file"]).resolve() != ROOT / "vllm/__init__.py":
                        raise RuntimeError("Worker imported another source checkout")
                    for component, binary in worker["binaries"].items():
                        if (
                            not binary["mapped"]
                            or binary["sha256"]
                            != bundle["libraries"][component]["sha256"]
                        ):
                            raise RuntimeError(
                                f"Worker did not map the freshly built {component}"
                            )
            for _ in range(args.repeats):
                result = _run_once(llm, {"prompt_token_ids": ids}, fixed)
                metrics = result["request_metrics"]
                if (
                    result["output_tokens"] != 513
                    or not metrics
                    or metrics["raw"]["is_corrupted"]
                ):
                    raise RuntimeError("Missing/corrupted separated request metrics")
                case["runs"].append(result)
                save()
                if result["token_ids"] != case["warmup"]["token_ids"]:
                    raise RuntimeError(f"Token repeatability failed: {case['name']}")
                if (
                    case["name"] in reference
                    and result["token_ids"] != reference[case["name"]]
                ):
                    raise RuntimeError(
                        f"Output differs from frozen reference: {case['name']}"
                    )
            case["summary"] = _summarize(case["runs"])
            case["prefill_tps_mean"] = statistics.mean(
                length / r["request_metrics"]["prefill_time"] for r in case["runs"]
            )
            case["same_token_hash"] = True
            case["matches_reference"] = True if case["name"] in reference else None
            print(
                json.dumps(
                    {
                        "case": case["name"],
                        "summary": case["summary"],
                        "prefill_tps": case["prefill_tps_mean"],
                    }
                ),
                flush=True,
            )
            save()
        if args.long_context:
            marker = "ONECAT_LONG_ARCHIVE_MARKER"
            rendered = tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": marker
                        + "\nFind the archive code and finish with RESULT=<code>.",
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            before, after = rendered.split(marker)
            lead = tokenizer.encode(before, add_special_tokens=False)
            tail = tokenizer.encode(after, add_special_tokens=False)
            filler = tokenizer.encode(
                "A routine archive entry, without task instructions.\n",
                add_special_tokens=False,
            )
            record = tokenizer.encode(
                "\nArchive code: MAPLE-8261.\n", add_special_tokens=False
            )
            count = 261631 - len(lead) - len(tail) - len(record)
            padding = (filler * ((count + len(filler) - 1) // len(filler)))[:count]
            ids = lead + padding[: count // 2] + record + padding[count // 2 :] + tail
            health(llm, "long_middle_record", ids, r"RESULT\s*=\s*MAPLE-8261")
            ids = (piece * ((262143 + len(piece) - 1) // len(piece)))[:262143]
            report["exact_context_boundary"] = _run_once(
                llm,
                {"prompt_token_ids": ids},
                SamplingParams(max_tokens=1, temperature=0, seed=0, ignore_eos=True),
            )
            if report["exact_context_boundary"]["output_tokens"] != 1:
                raise RuntimeError("262143+1 context boundary failed")
        report["complete"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown(timeout=30.0)
        report["ended_unix"] = time.time()
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--native-extension-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-json", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--long-context", action="store_true")
    parser.add_argument(
        "--use-defaults",
        action="store_true",
        help="Use model-aware defaults, without baseline optimization/graph overrides.",
    )
    parser.add_argument("--configured", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("At least two repeats are required")
    if not args.configured:
        configure_environment(
            args.runtime_dir,
            args.output.parent / "cache",
            args.native_extension_dir,
            use_defaults=args.use_defaults,
        )
        # The fresh interpreter installs the same import path in spawned workers.
        os.execv(
            sys.executable,
            [
                sys.executable,
                str(Path(__file__).resolve()),
                *sys.argv[1:],
                "--configured",
            ],
        )

    def stop(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    run(args)


if __name__ == "__main__":
    main()
