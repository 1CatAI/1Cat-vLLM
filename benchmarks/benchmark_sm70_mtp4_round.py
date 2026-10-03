# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unprofiled Flash-Next MTP4 complete-round regression at fixed 8K input.

Run in a fresh process with user-supplied VLLM variables absent. Output includes
actual token sequences and speculative counters; latency uses the endpoint
decode interval and completed rounds, not verifier-only or emitted-token time.
"""

import argparse
import copy
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--startup-diagnostics", action="store_true")
    parser.add_argument("--teacher-forcing-manifest", type=Path)
    parser.add_argument("--quality-manifest", type=Path)
    args = parser.parse_args()
    # Importing vLLM applies platform defaults. Audit first to distinguish
    # those framework defaults from user-supplied performance variables.
    supplied = {
        key: value for key, value in os.environ.items() if key.startswith("VLLM_")
    }
    if supplied:
        raise ValueError(
            f"Default admission requires no supplied VLLM variables: {supplied}"
        )
    if args.repeats < 3:
        raise ValueError("At least three measured repetitions are required")
    # AOT compilation uses this root in addition to CompilationConfig.cache_dir.
    os.environ["VLLM_CACHE_ROOT"] = str(args.out.parent.resolve() / "vllm_cache")
    from benchmarks.benchmark_sm70_model_tokens import (
        _metric_snapshot,
        _request_metrics_dict,
        _spec_decoding_delta,
    )
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    piece = tokenizer.encode(
        "This fixed benchmark prompt is used to create a deterministic "
        "tokenized input for single-request decode measurement. ",
        add_special_tokens=False,
    )
    fixed = (piece * (8192 // len(piece) + 1))[:8192]
    fixtures = [{"id": "fixed8k", "prompt_token_ids": fixed, "natural": False}]
    for index, question in enumerate(
        (
            "一辆车先以每小时60公里行驶2小时，再以每小时80公里行驶1.5小时。"
            "总路程和全程平均速度分别是多少？说明计算过程。",
            "Write a Python function computing the longest increasing subsequence "
            "length in O(n log n). Include three edge-case tests.",
            "Explain how to prove the sum of the first n odd positive integers "
            "equals n squared. Give two different arguments.",
            "请解释为什么整数除法处理负数时需要特别小心，给出 Python 和 C "
            "结果不同的例子，并说明余数的约束。",
        )
    ):
        # Filler stays inside the user message and the actual question is last.
        prefix = tokenizer.encode(
            "Context document. The following repeated reference text contains "
            "no instructions. Use the question at the end.\n",
            add_special_tokens=False,
        )
        suffix = tokenizer.apply_chat_template(
            [{"role": "user", "content": "\nQuestion:\n" + question}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if hasattr(suffix, "input_ids"):
            suffix = suffix.input_ids
        # Preserve the chat delimiters: insert document after the role header.
        header_len = 3
        count = 8192 - len(suffix) - len(prefix)
        filler = (piece * (count // len(piece) + 1))[:count]
        tokens = list(suffix[:header_len]) + prefix + filler + list(suffix[header_len:])
        assert len(tokens) == 8192
        fixtures.append(
            {"id": f"natural8k/{index}", "prompt_token_ids": tokens, "natural": True}
        )
    engine = {
        "model": args.model,
        "tensor_parallel_size": 4,
        "dtype": "half",
        "kv_cache_dtype": "auto",
        "mamba_cache_dtype": "float16",
        "mamba_ssm_cache_dtype": "float32",
        "mamba_cache_mode": "align",
        "max_model_len": 262144,
        "max_num_seqs": 1,
        "max_num_batched_tokens": 2048,
        "kv_cache_memory_bytes": 4 * 1024**3,
        "enable_prefix_caching": True,
        "language_model_only": True,
        "disable_log_stats": False,
        "compilation_config": {
            "cache_dir": str(args.out.parent.resolve() / "compile_cache"),
        },
        "speculative_config": {
            "method": "mtp",
            "num_speculative_tokens": 4,
            "draft_sample_method": "greedy",
        },
    }
    if args.startup_diagnostics:
        engine["worker_cls"] = "benchmarks.sm70_startup_worker.StartupStackWorker"
    report = {
        "source": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "supplied_vllm_env": supplied,
        "engine": copy.deepcopy(engine),
        "fixtures": fixtures,
        "cases": [],
        "warmup_outputs": [],
        "complete": False,
        "startup_diagnostics": args.startup_diagnostics,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        temp = args.out.with_suffix(".tmp")
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        temp.replace(args.out)

    save()
    llm = None
    try:
        started = time.monotonic()
        llm = LLM(**copy.deepcopy(engine))
        report["load_seconds"] = time.monotonic() - started
        report["resolved_engine"] = str(llm.llm_engine.vllm_config)
        for fixture in fixtures:
            sampling = (
                {
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "top_k": 20,
                    "max_tokens": 512,
                    "seed": 20261003,
                    "ignore_eos": False,
                    "skip_special_tokens": False,
                }
                if fixture["natural"]
                else {
                    "temperature": 0,
                    "max_tokens": 513,
                    "seed": 0,
                    "ignore_eos": True,
                    "skip_special_tokens": False,
                }
            )
            # Warm this exact prompt/shape before measuring its requests.
            for repeat in range(-1, args.repeats):
                llm.reset_prefix_cache()
                before = _metric_snapshot(llm)
                result = llm.generate(
                    [{"prompt_token_ids": fixture["prompt_token_ids"]}],
                    SamplingParams(**sampling),
                    use_tqdm=False,
                )[0]
                after = _metric_snapshot(llm)
                if repeat < 0:
                    report["warmup_outputs"].append(
                        {
                            "id": fixture["id"],
                            "token_ids": list(result.outputs[0].token_ids),
                            "text": result.outputs[0].text,
                            "finish_reason": result.outputs[0].finish_reason,
                        }
                    )
                    save()
                    continue
                output = result.outputs[0]
                metrics = _request_metrics_dict(result.metrics, len(output.token_ids))
                spec = _spec_decoding_delta(before, after)
                if not metrics or not spec or metrics["raw"]["is_corrupted"]:
                    raise RuntimeError("Missing or corrupted endpoint/round counters")
                row = {
                    "id": fixture["id"],
                    "repeat": repeat,
                    "prompt_sha256": hashlib.sha256(
                        json.dumps(fixture["prompt_token_ids"]).encode()
                    ).hexdigest(),
                    "sampling": sampling,
                    "token_ids": list(output.token_ids),
                    "text": output.text,
                    "finish_reason": output.finish_reason,
                    "metrics": metrics,
                    "spec_decoding": spec,
                    "complete_round_ms": 1000
                    * metrics["decode_time"]
                    / spec["num_drafts"],
                }
                report["cases"].append(row)
                save()
                print(
                    json.dumps(
                        {
                            key: row[key]
                            for key in (
                                "id",
                                "repeat",
                                "complete_round_ms",
                                "metrics",
                                "spec_decoding",
                            )
                        }
                    ),
                    flush=True,
                )
        report["speed_complete"] = True
        report["latency_passed"] = all(
            row["complete_round_ms"] <= 15 for row in report["cases"]
        )
        save()
        # Diagnostics follow the completed unprofiled speed report and use no
        # timing or acceptance counters from their forced requests.
        if args.teacher_forcing_manifest:
            from benchmarks.sm70_mtp_teacher_forcing import flush, install

            tapes = json.loads(args.teacher_forcing_manifest.read_text())
            report["teacher_forcing"] = []
            for tape in tapes:
                llm.reset_prefix_cache()
                folder = args.out.parent / "teacher_forcing" / tape["id"]
                llm.collective_rpc(
                    install,
                    args=(
                        tape["token_ids"],
                        tape["prompt_length"],
                        tape["prompt_sha256"],
                        str(folder),
                    ),
                )
                succeeded = False
                try:
                    output = llm.generate(
                        [
                            {
                                "prompt_token_ids": tape["token_ids"][
                                    : tape["prompt_length"]
                                ]
                            }
                        ],
                        SamplingParams(
                            temperature=0,
                            ignore_eos=True,
                            max_tokens=tape["output_length"],
                        ),
                        use_tqdm=False,
                    )[0]
                    expected = tape["token_ids"][
                        tape["prompt_length"] : tape["prompt_length"]
                        + tape["output_length"]
                    ]
                    if list(output.outputs[0].token_ids) != expected:
                        raise RuntimeError(
                            "Teacher-forcing output differs from frozen tape"
                        )
                    succeeded = True
                finally:
                    result = llm.collective_rpc(
                        flush, kwargs={"discard": not succeeded}
                    )
                report["teacher_forcing"].append({"id": tape["id"], "workers": result})
                save()
        if args.quality_manifest:
            report["quality"] = []
            for case in json.loads(args.quality_manifest.read_text()):
                llm.reset_prefix_cache()
                before = _metric_snapshot(llm)
                output = llm.generate(
                    [{"prompt_token_ids": case["prompt_token_ids"]}],
                    SamplingParams(**case["sampling"]),
                    use_tqdm=False,
                )[0]
                report["quality"].append(
                    {
                        "id": case["id"],
                        "token_ids": list(output.outputs[0].token_ids),
                        "text": output.outputs[0].text,
                        "finish_reason": output.outputs[0].finish_reason,
                        "spec_decoding": _spec_decoding_delta(
                            before, _metric_snapshot(llm)
                        ),
                    }
                )
                save()
        report["complete"] = True
        save()
    except BaseException:
        import traceback

        report["error"] = traceback.format_exc()
        save()
        raise
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
