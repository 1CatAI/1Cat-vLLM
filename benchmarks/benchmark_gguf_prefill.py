# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Matched GGUF prefill comparisons with fixed token inputs."""

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

import vllm
from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--draft", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-tokens", type=int, default=32768)
    parser.add_argument("--prefill-chunk", type=int, default=16384)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=1610612736)
    parser.add_argument("--kernel-config", type=json.loads, default={})
    parser.add_argument(
        "--compare",
        choices=("routing", "host-prefill", "prefill-norm"),
        default="routing",
    )
    parser.add_argument("--decode-check", action="store_true")
    parser.add_argument("--completion-check", action="store_true")
    parser.add_argument("--profile-once", action="store_true")
    parser.add_argument(
        "--candidate-only",
        action="store_true",
        help="Measure the candidate after a separately recorded matched comparison",
    )
    args = parser.parse_args()
    if args.input_tokens <= 0 or args.prefill_chunk <= 0:
        parser.error("input and chunk token counts must be positive")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    config = dict(
        model=args.model,
        quantization="gguf",
        dtype="half",
        tensor_parallel_size=args.tp,
        kv_cache_dtype="float16",
        mamba_ssm_cache_dtype="float32",
        max_model_len=args.input_tokens + 1024,
        max_num_batched_tokens=args.prefill_chunk,
        max_num_seqs=4,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        gpu_memory_utilization=0.95,
        enable_prefix_caching=False,
        disable_log_stats=False,
        language_model_only=True,
        compilation_config={"mode": 3, "cudagraph_mode": "FULL"},
        worker_extension_cls="vllm.sm70_graph_observer.GraphParityWorkerExtension",
        kernel_config=args.kernel_config,
        speculative_config={
            "method": "mtp",
            "model": args.draft,
            "num_speculative_tokens": 4,
            "draft_load_config": {"load_format": "safetensors"},
            "draft_sample_method": "greedy",
        },
    )
    trace_directory = args.output.parent / "prefill-trace"
    if args.profile_once:
        config["profiler_config"] = dict(
            profiler="torch",
            torch_profiler_dir=str(trace_directory.resolve()),
            torch_profiler_with_stack=False,
            torch_profiler_with_memory=False,
            torch_profiler_record_shapes=False,
            torch_profiler_dump_cuda_time_total=False,
            torch_profiler_use_gzip=True,
        )
    report = dict(
        config=copy.deepcopy(config),
        version=vllm.__version__,
        origin=vllm.__file__,
        comparison=args.compare,
        candidate_only=args.candidate_only,
        rows=[],
        decode_checks=[],
        completions=[],
        complete=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    save()
    llm = None
    try:
        llm = LLM(**config)
        report["initialized_memory"] = llm.collective_rpc(
            "read_prefill_memory", args=(True,)
        )
        save()
        report["initialized_storages"] = llm.collective_rpc("read_prefill_storages")
        save()
        tokenizer = llm.get_tokenizer()
        unit = tokenizer.encode(
            "请阅读以下技术文档，并概括其中的核心观点。模型推理的性能受计算效率、"
            "内存访问和通信影响，测试应保持输入与运行配置一致。\n",
            add_special_tokens=False,
        )
        ids = (unit * (args.input_tokens // len(unit) + 1))[: args.input_tokens]
        report["input_sha256"] = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        report["input_kind"] = "fixed repeated technical text; not a corpus benchmark"
        params = SamplingParams(temperature=0, max_tokens=1)
        method = {
            "routing": "set_gguf_prefill_routing_policy",
            "host-prefill": "set_qsa_host_prefill_policy",
            "prefill-norm": "set_prefill_rmsnorm_gated_policy",
        }[args.compare]
        report["warmups"] = []
        policies = (True,) if args.candidate_only else (False, True)
        for enabled in policies:
            policy = llm.collective_rpc(method, args=(enabled,))
            warmup = llm.generate({"prompt_token_ids": ids}, params, use_tqdm=False)[0]
            report["warmups"].append(
                dict(
                    enabled=enabled,
                    policy=policy,
                    output_ids=warmup.outputs[0].token_ids,
                    memory=llm.collective_rpc("read_prefill_memory", args=(True,)),
                )
            )
            save()
        measured_policies = (
            (True, True) if args.candidate_only else (False, True, True, False)
        )
        for enabled in measured_policies:
            policy = llm.collective_rpc(method, args=(enabled,))
            before_memory = llm.collective_rpc("read_prefill_memory", args=(True,))
            started = time.perf_counter()
            result = llm.generate({"prompt_token_ids": ids}, params, use_tqdm=False)[0]
            wall_seconds = time.perf_counter() - started
            metrics = result.metrics
            after_memory = llm.collective_rpc("read_prefill_memory")
            row = dict(
                enabled=enabled,
                policy=policy,
                wall_seconds=wall_seconds,
                before_memory=before_memory,
                after_memory=after_memory,
                scheduled_to_first_token_seconds=None,
                output_ids=result.outputs[0].token_ids,
            )
            report["rows"].append(row)
            save()
            if metrics is None or metrics.scheduled_ts <= 0:
                raise RuntimeError(
                    "Prefill timestamps missing; request statistics must be enabled"
                )
            row["scheduled_to_first_token_seconds"] = (
                metrics.first_token_ts - metrics.scheduled_ts
            )
            save()
            print(json.dumps(report["rows"][-1]), flush=True)
        if args.decode_check:
            sys.path.append(str(Path(__file__).resolve().parents[1]))
            from benchmarks.benchmark_flashnext_acceptance import observed_cohort
            from benchmarks.benchmark_sm70_qwen38_concurrency import summarize

            for enabled in policies:
                llm.collective_rpc(method, args=(enabled,))
                for width in (1, 4):
                    steps, outputs = observed_cohort(
                        llm,
                        ids[:8192],
                        SamplingParams(temperature=0, max_tokens=600, ignore_eos=True),
                        width,
                    )
                    check = dict(
                        enabled=enabled,
                        width=width,
                        input_tokens=8192,
                        steps=steps,
                        output_ids=[list(row.outputs[0].token_ids) for row in outputs],
                    )
                    report["decode_checks"].append(check)
                    save()
                    try:
                        check["summary"] = summarize(steps, width)
                    except RuntimeError as error:
                        # Preserve admission failures without losing the later
                        # diagnostic trace. A missing interval is not a pass.
                        check["error"] = str(error)
                    save()
            report["decode_checks_valid"] = all(
                "summary" in row for row in report["decode_checks"]
            )
            save()
        if args.completion_check:
            prompts = (
                "只给出计算结果：17×5 等于多少？",
                "用两句话解释为什么推理测试要分别测 prefill 和 decode。",
                "写一个 Python 函数，返回整数列表中所有偶数的和，并举一个调用例子。",
            )
            for enabled in policies:
                llm.collective_rpc(method, args=(enabled,))
                for prompt in prompts:
                    rendered = tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    output = llm.generate(
                        rendered,
                        SamplingParams(temperature=0, max_tokens=512),
                        use_tqdm=False,
                    )[0].outputs[0]
                    report["completions"].append(
                        dict(
                            enabled=enabled,
                            prompt=prompt,
                            output_ids=list(output.token_ids),
                            text=output.text,
                            finish_reason=output.finish_reason,
                        )
                    )
                    save()
        report["matching_output_ids"] = (
            len(
                {tuple(row["output_ids"]) for row in report["warmups"] + report["rows"]}
            )
            == 1
        )
        save()
        if args.profile_once:
            llm.collective_rpc(method, args=(True,))
            # Use named RPCs and the built-in profiler; callable RPC transport
            # requires unsafe serialization in the multiprocess engine.
            llm.collective_rpc("read_prefill_memory", args=(False, True))
            llm.start_profile("prefill32k")
            warmup = llm.generate({"prompt_token_ids": ids}, params, use_tqdm=False)[0]
            llm.stop_profile()
            warm_files = set(trace_directory.glob("*.pt.trace.json.gz"))
            llm.start_profile("prefill32k")
            started = time.perf_counter()
            profiled = llm.generate({"prompt_token_ids": ids}, params, use_tqdm=False)[
                0
            ]
            profile_wall_seconds = time.perf_counter() - started
            llm.stop_profile()
            files = sorted(set(trace_directory.glob("*.pt.trace.json.gz")) - warm_files)
            if len(files) != args.tp:
                raise RuntimeError(
                    f"Expected {args.tp} worker traces, got {len(files)}"
                )
            report["profile"] = dict(
                scope="profiled candidate; excluded from unprofiled throughput",
                wall_seconds=profile_wall_seconds,
                scheduled_to_first_token_seconds=(
                    profiled.metrics.first_token_ts - profiled.metrics.scheduled_ts
                    if profiled.metrics is not None
                    else None
                ),
                warmup_output_ids=warmup.outputs[0].token_ids,
                output_ids=profiled.outputs[0].token_ids,
                files=[
                    dict(path=str(path), bytes=path.stat().st_size) for path in files
                ],
            )
        report["complete"] = True
        save()
    except Exception as error:
        report["error"] = repr(error)
        save()
        raise
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
