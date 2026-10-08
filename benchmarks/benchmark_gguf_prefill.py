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
from typing import Any

import torch

import vllm
from vllm import LLM, SamplingParams


def _start_prefill_trace(worker: Any) -> None:
    # Tracing starts after the unprofiled measurements. Release unused cached
    # allocations to admit CUPTI buffers, then warm again before recording.
    torch.accelerator.empty_cache()
    worker._gguf_prefill_trace = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    )
    worker._gguf_prefill_trace.start()


def _activate_prefill_trace(worker: Any) -> None:
    worker._gguf_prefill_trace.step()


def _finish_prefill_trace(worker: Any, directory: str) -> dict:
    profile = worker._gguf_prefill_trace
    profile.stop()
    output = Path(directory) / f"rank{worker.rank}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    profile.export_chrome_trace(str(output))
    worker._gguf_prefill_trace = None
    return dict(rank=worker.rank, path=str(output), bytes=output.stat().st_size)


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
        "--compare", choices=("routing", "host-prefill"), default="routing"
    )
    parser.add_argument("--decode-check", action="store_true")
    parser.add_argument("--profile-once", action="store_true")
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
    report = dict(
        config=copy.deepcopy(config),
        version=vllm.__version__,
        origin=vllm.__file__,
        comparison=args.compare,
        rows=[],
        decode_checks=[],
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
        method = (
            "set_qsa_host_prefill_policy"
            if args.compare == "host-prefill"
            else "set_gguf_prefill_routing_policy"
        )
        report["warmups"] = []
        for enabled in (False, True):
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
        for enabled in (False, True, True, False):
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

            for enabled in (False, True):
                llm.collective_rpc(method, args=(enabled,))
                for width in (1, 4):
                    steps, _ = observed_cohort(
                        llm,
                        ids[:8192],
                        SamplingParams(temperature=0, max_tokens=600, ignore_eos=True),
                        width,
                    )
                    report["decode_checks"].append(
                        dict(
                            enabled=enabled,
                            width=width,
                            summary=summarize(steps, width),
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
            llm.collective_rpc(_start_prefill_trace)
            warmup = llm.generate({"prompt_token_ids": ids}, params, use_tqdm=False)[0]
            llm.collective_rpc(_activate_prefill_trace)
            profiled = llm.generate({"prompt_token_ids": ids}, params, use_tqdm=False)[
                0
            ]
            files = llm.collective_rpc(
                _finish_prefill_trace, args=(str(args.output.parent / "prefill-trace"),)
            )
            report["profile"] = dict(
                scope="profiled candidate; excluded from unprofiled throughput",
                warmup_output_ids=warmup.outputs[0].token_ids,
                output_ids=profiled.outputs[0].token_ids,
                files=files,
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
