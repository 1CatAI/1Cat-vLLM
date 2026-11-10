# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Matched single-request timing, with separate opt-in stage instrumentation.

Run twice, with and without --draft, in separate engine processes. Both runs
use the same 256 MiB KV allocation, chunk 512, TP4 and 30% shared offload.
Raw timings have no finite-logit or stage hooks. Profiled timings are labeled
separately; nested expert/Engram stream intervals are not additive to target.
"""

import argparse
import json
import time
from pathlib import Path

from benchmarks.deepseek_v41 import trace
from benchmarks.deepseek_v41.acceptance import request, validate_commits
from benchmarks.deepseek_v41.main_smoke import (
    _memory_report,
    create_llm,
    trim_allocator,
)
from benchmarks.deepseek_v41.resources import ResourceMonitor


def make_prompt(tokenizer, length):
    marker = "__V41_BENCH_FILL__"
    text = tokenizer.apply_chat_template(
        [
            dict(
                role="user",
                content=marker + "\n忽略前面的背景资料。只输出从0到1000的整数JSON数组，"
                "不要省略任何数字，不要解释。",
            )
        ],
        tokenize=False,
        enable_thinking=False,
    )
    before, after = [
        tokenizer.encode(s, add_special_tokens=False) for s in text.split(marker)
    ]
    fill = tokenizer.encode(
        "背景资料：春夏秋冬，日月星辰。" * 4096, add_special_tokens=False
    )
    size = length - len(before) - len(after)
    if not 0 <= size <= len(fill):
        raise ValueError("Prompt length cannot hold the benchmark instruction")
    return before + fill[:size] + after


def run(args):
    resources = ResourceMonitor()
    llm = None
    report = dict(
        status="running",
        dspark=args.draft is not None,
        scope="single request; raw timings separate from instrumented profiles",
        kv_cache_bytes=256 * 1024**2,
        chunk=512,
        offload_fraction=0.30,
        target_graph=args.decode_graph,
        draft_graph=args.draft_graph,
        custom_allreduce=args.custom_ar,
        cases=[],
    )

    def save():
        report["resources"] = resources.report()
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(args.output)

    try:
        start = time.monotonic()
        llm = create_llm(
            args.model_root,
            512,
            args.draft,
            kv_cache_mib=256,
            decode_graph=args.decode_graph,
            draft_graph=args.draft_graph,
            custom_allreduce=args.custom_ar,
        )
        report["load_seconds"] = time.monotonic() - start
        llm.collective_rpc(trim_allocator)
        tokenizer = llm.get_tokenizer()
        prompts = {n: make_prompt(tokenizer, n) for n in args.lengths}
        for length, prompt in prompts.items():
            # Warm each shape and resident set before recording raw latency.
            for _ in range(2):
                request(llm, prompt, args.output_tokens)
            before = llm.collective_rpc(_memory_report)
            runs = [
                request(llm, prompt, args.output_tokens) for _ in range(args.repeats)
            ]
            if any(len(r["token_ids"]) != args.output_tokens for r in runs):
                raise AssertionError("Timing workload ended before its output budget")
            if any(r["token_ids"] != runs[0]["token_ids"] for r in runs[1:]):
                raise AssertionError("Repeated greedy completions differ")
            report["cases"].append(
                dict(
                    prompt_tokens=length,
                    instrumentation=False,
                    runs=runs,
                    memory_before=before,
                    memory_after=llm.collective_rpc(_memory_report),
                )
            )
            save()
            print(
                json.dumps(dict(raw_prompt_tokens=length, completed=True)), flush=True
            )

        # The same prompts now qualify stage and acceptance reporting; these
        # wall times are excluded from the raw performance comparison.
        llm.collective_rpc(trace.install)
        for length, prompt in prompts.items():
            result = request(llm, prompt, args.output_tokens)
            result["trace_by_rank"] = llm.collective_rpc(trace.drain)
            if args.decode_graph:
                for rank in result["trace_by_rank"]:
                    small = [row for row in rank["target_forwards"] if row["rows"] <= 6]
                    assert all(row["mode"] == "FULL" for row in small), (length, rank)
            if args.custom_ar:
                assert all(
                    rank["custom_allreduce_enabled"] for rank in result["trace_by_rank"]
                )
            validate_commits(
                result["trace_by_rank"], len(prompt) + len(result["token_ids"]) - 1
            )
            report["cases"].append(
                dict(
                    prompt_tokens=length,
                    instrumentation=True,
                    result=result,
                    memory_after=llm.collective_rpc(_memory_report),
                )
            )
            save()
        if resources.report()["host_minimum_available_bytes"] < 8 * 1024**3:
            raise MemoryError("Performance qualification breached the host reserve")
        report["status"] = "completed"
        save()
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        save()
        raise
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()
        resources.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--draft", type=Path)
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 2048])
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--decode-graph", action="store_true")
    parser.add_argument("--draft-graph", action="store_true")
    parser.add_argument("--custom-ar", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.repeats < 1
        or args.output_tokens < 2
        or min(args.lengths) < 1
        or max(args.lengths) + args.output_tokens > 16384
    ):
        parser.error("Invalid request count or context budget")
    run(args)


if __name__ == "__main__":
    main()
