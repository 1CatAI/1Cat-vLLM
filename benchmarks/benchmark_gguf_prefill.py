# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Matched GGUF prefill comparisons with fixed token inputs."""

import argparse
import copy
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

import vllm
from vllm import LLM, SamplingParams


def reset_cold_prefix_cache(llm, enabled: bool) -> bool:
    if enabled and not llm.reset_prefix_cache():
        raise RuntimeError("Prefix cache reset failed before cold prefill")
    return enabled


def cold_prefill_evidence(result, enabled: bool) -> dict:
    cached = result.num_cached_tokens
    if cached not in (None, 0) or (enabled and cached is None):
        raise RuntimeError(f"Cold prefill requires zero cached tokens, got {cached}")
    prompt_tokens = len(result.prompt_token_ids)
    return dict(
        num_cached_tokens=cached,
        computed_prompt_tokens=prompt_tokens - (cached or 0),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--draft", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-tokens", type=int, default=32768)
    parser.add_argument("--prefill-chunk", type=int, default=16384)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--kv-cache-memory-bytes", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--kernel-config", type=json.loads, default={})
    parser.add_argument(
        "--compare",
        choices=("routing", "host-prefill", "host-prefill-grouped", "prefill-norm"),
        default="routing",
    )
    parser.add_argument("--decode-check", action="store_true")
    parser.add_argument("--completion-check", action="store_true")
    parser.add_argument(
        "--completion-context",
        action="store_true",
        help="Run completion/acceptance checks after the full fixed text context",
    )
    parser.add_argument("--acceptance-check", action="store_true")
    parser.add_argument("--profile-once", action="store_true")
    parser.add_argument("--profile-kind", choices=("torch", "cuda"), default="torch")
    fixed_policy = parser.add_mutually_exclusive_group()
    fixed_policy.add_argument(
        "--candidate-only",
        action="store_true",
        help="Measure the candidate after a separately recorded matched comparison",
    )
    fixed_policy.add_argument(
        "--control-only",
        action="store_true",
        help="Measure the control for a separate-process memory admission comparison",
    )
    args = parser.parse_args()
    if args.input_tokens <= 0 or args.prefill_chunk <= 0:
        parser.error("input and chunk token counts must be positive")
    if args.repeats < 2:
        parser.error("at least two measured repeats are required")
    if args.max_model_len is not None and args.max_model_len < args.input_tokens + 1:
        parser.error("context must hold the input and at least one output token")
    if args.completion_context and not args.completion_check:
        parser.error("--completion-context requires --completion-check")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    config = dict(
        model=args.model,
        quantization="gguf",
        dtype="half",
        tensor_parallel_size=args.tp,
        kv_cache_dtype="float16",
        mamba_ssm_cache_dtype="float32",
        max_model_len=args.max_model_len or args.input_tokens + 1024,
        max_num_batched_tokens=args.prefill_chunk,
        max_num_seqs=4,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=args.enable_prefix_caching,
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
        config["profiler_config"] = dict(profiler=args.profile_kind)
        if args.profile_kind == "torch":
            config["profiler_config"].update(
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
        control_only=args.control_only,
        runtime_settings={
            "fused_moe_activation_chunking": (
                vllm.envs.VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING
            ),
            "fused_moe_chunk_rows": vllm.envs.VLLM_FUSED_MOE_CHUNK_SIZE,
            "pytorch_alloc_conf": os.environ.get(
                "PYTORCH_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
            ),
        },
        rows=[],
        decode_checks=[],
        acceptance_checks=[],
        completions=[],
        complete=False,
        prefill_contract="cold prefix cache; warmed compiler and filesystem caches",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    save()
    llm = None
    try:
        llm = LLM(**config)
        effective_cache = llm.llm_engine.vllm_config.cache_config
        report["effective_prefix_caching"] = effective_cache.enable_prefix_caching
        report["effective_mamba_cache_mode"] = effective_cache.mamba_cache_mode
        report["effective_max_model_len"] = (
            llm.llm_engine.vllm_config.model_config.max_model_len
        )
        if effective_cache.enable_prefix_caching != args.enable_prefix_caching:
            raise RuntimeError("Effective prefix-cache policy differs from request")
        if report["effective_max_model_len"] != config["max_model_len"]:
            raise RuntimeError("Effective context limit differs from request")
        report["initialized_memory"] = llm.collective_rpc(
            "read_prefill_memory", args=(True, False, True)
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
            "host-prefill-grouped": "set_qsa_host_prefill_grouped_policy",
            "prefill-norm": "set_prefill_rmsnorm_gated_policy",
        }[args.compare]
        report["warmups"] = []
        policies = (
            (False,)
            if args.control_only
            else (True,)
            if args.candidate_only
            else (False, True)
        )
        for enabled in policies:
            policy = llm.collective_rpc(method, args=(enabled,))
            reset = reset_cold_prefix_cache(llm, args.enable_prefix_caching)
            warmup = llm.generate({"prompt_token_ids": ids}, params, use_tqdm=False)[0]
            report["warmups"].append(
                dict(
                    enabled=enabled,
                    policy=policy,
                    output_ids=warmup.outputs[0].token_ids,
                    memory=llm.collective_rpc("read_prefill_memory", args=(True,)),
                    prefix_cache_reset=reset,
                    **cold_prefill_evidence(warmup, args.enable_prefix_caching),
                )
            )
            save()
        measured_policies = (
            (False,) * args.repeats
            if args.control_only
            else (True,) * args.repeats
            if args.candidate_only
            else (False, True, True, False) * args.repeats
        )
        for enabled in measured_policies:
            policy = llm.collective_rpc(method, args=(enabled,))
            reset = reset_cold_prefix_cache(llm, args.enable_prefix_caching)
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
                prefix_cache_reset=reset,
            )
            report["rows"].append(row)
            save()
            row.update(cold_prefill_evidence(result, args.enable_prefix_caching))
            if metrics is None or metrics.scheduled_ts <= 0:
                raise RuntimeError(
                    "Prefill timestamps missing; request statistics must be enabled"
                )
            row["scheduled_to_first_token_seconds"] = (
                metrics.first_token_ts - metrics.scheduled_ts
            )
            save()
            print(json.dumps(report["rows"][-1]), flush=True)
        report["prefill_summary"] = []
        for enabled in policies:
            times = [
                row["scheduled_to_first_token_seconds"]
                for row in report["rows"]
                if row["enabled"] == enabled
            ]
            mean = statistics.mean(times)
            cv = statistics.pstdev(times) / mean
            report["prefill_summary"].append(
                dict(
                    enabled=enabled,
                    repeats=len(times),
                    mean_seconds=mean,
                    median_seconds=statistics.median(times),
                    min_seconds=min(times),
                    max_seconds=max(times),
                    coefficient_of_variation=cv,
                    median_input_tokens_per_second=args.input_tokens
                    / statistics.median(times),
                    stable=cv <= 0.05,
                )
            )
        save()
        if args.decode_check:
            sys.path.append(str(Path(__file__).resolve().parents[1]))
            from benchmarks.benchmark_flashnext_acceptance import observed_cohort
            from benchmarks.benchmark_sm70_qwen38_concurrency import summarize

            for enabled in policies:
                llm.collective_rpc(method, args=(enabled,))
                for width in (1, 4):
                    reset_cold_prefix_cache(llm, args.enable_prefix_caching)
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
                        cold_prefills=[
                            cold_prefill_evidence(row, args.enable_prefix_caching)
                            for row in outputs
                        ],
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
                    content = prompt
                    if args.completion_context:
                        content = (
                            "以下技术资料仅作背景，请回答最后的问题。\n"
                            + tokenizer.decode(ids)
                            + "\n问题："
                            + prompt
                        )
                    rendered = tokenizer.apply_chat_template(
                        [{"role": "user", "content": content}],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
                    reset_cold_prefix_cache(llm, args.enable_prefix_caching)
                    request = llm.generate(
                        {"prompt_token_ids": prompt_ids},
                        SamplingParams(temperature=0, max_tokens=512),
                        use_tqdm=False,
                    )[0]
                    evidence = cold_prefill_evidence(
                        request, args.enable_prefix_caching
                    )
                    output = request.outputs[0]
                    report["completions"].append(
                        dict(
                            enabled=enabled,
                            prompt=prompt,
                            input_tokens=len(prompt_ids),
                            full_context=args.completion_context,
                            output_ids=list(output.token_ids),
                            text=output.text,
                            finish_reason=output.finish_reason,
                            **evidence,
                        )
                    )
                    save()
        if args.acceptance_check:
            sys.path.append(str(Path(__file__).resolve().parents[1]))
            from benchmarks.benchmark_flashnext_acceptance import natural_row

            prompts = json.loads(
                Path(__file__)
                .with_name("flashnext_acceptance_prompts.json")
                .read_text()
            )
            if len(prompts) != 8 or len({p["id"] for p in prompts}) != 8:
                raise RuntimeError("Acceptance requires eight distinct prompts")
            sampling = SamplingParams(
                temperature=0, max_tokens=600, ignore_eos=False, seed=20261005
            )
            for enabled in policies:
                llm.collective_rpc(method, args=(enabled,))
                for prompt in prompts:
                    content = prompt["prompt"]
                    if args.completion_context:
                        content = (
                            "以下技术资料仅作背景，请回答最后的问题。\n"
                            + tokenizer.decode(ids)
                            + "\n问题："
                            + content
                        )
                    rendered = tokenizer.apply_chat_template(
                        [{"role": "user", "content": content}],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
                    reset_cold_prefix_cache(llm, args.enable_prefix_caching)
                    row = natural_row(llm, prompt, prompt_ids, sampling)
                    if args.enable_prefix_caching and row["num_cached_tokens"] != 0:
                        raise RuntimeError(
                            "Natural acceptance request hit prefix cache"
                        )
                    report["acceptance_checks"].append(dict(enabled=enabled, **row))
                    save()
        report["matching_output_ids"] = (
            len(
                {tuple(row["output_ids"]) for row in report["warmups"] + report["rows"]}
            )
            == 1
        )
        report["baseline_valid"] = (
            report["matching_output_ids"]
            and all(row["stable"] for row in report["prefill_summary"])
            and report.get("decode_checks_valid", True)
            and all(
                row["output_ids"] and row["finish_reason"] == "stop"
                for row in report["completions"]
            )
        )
        save()
        if args.profile_once:
            llm.collective_rpc(method, args=(not args.control_only,))
            # Use named RPCs and the built-in profiler; callable RPC transport
            # requires unsafe serialization in the multiprocess engine.
            llm.collective_rpc("read_prefill_memory", args=(False, True))
            warm_files = set()
            if args.profile_kind == "torch":
                reset_cold_prefix_cache(llm, args.enable_prefix_caching)
                llm.start_profile("prefill32k")
                warm_profile = llm.generate(
                    {"prompt_token_ids": ids}, params, use_tqdm=False
                )[0]
                llm.stop_profile()
                cold_prefill_evidence(warm_profile, args.enable_prefix_caching)
                warm_files = set(trace_directory.glob("*.pt.trace.json.gz"))
            reset_cold_prefix_cache(llm, args.enable_prefix_caching)
            llm.start_profile("prefill32k")
            started = time.perf_counter()
            profiled = llm.generate({"prompt_token_ids": ids}, params, use_tqdm=False)[
                0
            ]
            profile_wall_seconds = time.perf_counter() - started
            llm.stop_profile()
            files = sorted(set(trace_directory.glob("*.pt.trace.json.gz")) - warm_files)
            if args.profile_kind == "torch" and len(files) != args.tp:
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
                profiler=args.profile_kind,
                output_ids=profiled.outputs[0].token_ids,
                **cold_prefill_evidence(profiled, args.enable_prefix_caching),
                files=[
                    dict(path=str(path), bytes=path.stat().st_size) for path in files
                ],
            )
            report["profile_memory"] = llm.collective_rpc(
                "read_prefill_memory", args=(False, False, True)
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
