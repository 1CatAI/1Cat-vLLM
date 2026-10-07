# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Matched GGUF prefill routing comparisons with fixed token inputs."""

import argparse
import hashlib
import json
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
    report = dict(config=config, version=vllm.__version__, rows=[], complete=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    save()
    llm = LLM(**config)
    try:
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
        llm.generate({"prompt_token_ids": ids}, params, use_tqdm=False)
        for enabled in (False, True, True, False):
            llm.collective_rpc("set_gguf_prefill_routing_policy", args=(enabled,))
            started = time.perf_counter()
            result = llm.generate({"prompt_token_ids": ids}, params, use_tqdm=False)[0]
            metrics = result.metrics
            report["rows"].append(
                dict(
                    routing=enabled,
                    wall_seconds=time.perf_counter() - started,
                    scheduled_to_first_token_seconds=(
                        metrics.first_token_ts - metrics.scheduled_ts
                    ),
                    output_ids=result.outputs[0].token_ids,
                )
            )
            save()
            print(json.dumps(report["rows"][-1]), flush=True)
        report["matching_output_ids"] = (
            len({tuple(row["output_ids"]) for row in report["rows"]}) == 1
        )
        report["complete"] = True
        save()
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
