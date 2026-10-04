# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare GGUF and HF quantized text models under the same TP4 contract.

Run while holding the shared GPU lock. Natural greedy checks retain EOS;
fixed-length synthetic decode timing is reported separately. Prefill uses one
output token. The model/kernel origin and core fingerprint are recorded.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import vllm._C as core

import vllm
from vllm import LLM, SamplingParams

# The shared helpers add source paths for their standalone entrypoints. Restore
# the caller's search path so spawned workers use the same installed packages.
_original_path = sys.path.copy()
try:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from benchmarks.benchmark_sm70_model_tokens import (
        _metric_snapshot,
        _request_metrics_dict,
        _spec_decoding_delta,
    )
    from benchmarks.benchmark_sm70_qwen38_concurrency import (
        generate_cohort,
        summarize,
    )
finally:
    sys.path[:] = _original_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompts-json", type=Path, required=True)
    parser.add_argument("--require-installed", action="store_true")
    parser.add_argument("--cuda-profiler-capture", action="store_true")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--ring", choices=("auto", "disabled"), default="auto")
    parser.add_argument("--mtp-draft", type=Path)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--concurrent-input-len", type=int)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--concurrent-output-len", type=int)
    parser.add_argument("--widths", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--prefill", type=int, nargs="*", default=[8192, 32768])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-batch", type=int, default=8192)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--max-seqs", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--ssm-state-dtype")
    parser.add_argument("--record-first-logprobs", action="store_true")
    args = parser.parse_args()
    if args.require_installed and "site-packages" not in vllm.__file__:
        raise RuntimeError("Benchmark requires an ordinary installed wheel")
    if (
        not args.widths
        or min(args.widths) < 1
        or args.output_len < 64
        or args.input_len < 1
        or (args.concurrent_input_len is not None and args.concurrent_input_len < 1)
        or (args.concurrent_output_len is not None and args.concurrent_output_len < 64)
        or args.repeats < 1
    ):
        raise ValueError("Invalid fixed-width timing workload")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    maximum_input = max(
        [args.input_len, args.concurrent_input_len or args.input_len, *args.prefill]
    )
    maximum_output = max(args.output_len, args.concurrent_output_len or args.output_len)
    model_len = args.max_model_len or maximum_input + maximum_output + 128
    max_seqs = args.max_seqs or max(args.widths)
    required_length = max(
        [
            (args.concurrent_input_len or args.input_len)
            + (args.concurrent_output_len or args.output_len)
            if width > 1
            else args.input_len + args.output_len
            for width in args.widths
        ]
        + [length + 1 for length in args.prefill]
    )
    if model_len < required_length or max_seqs < max(args.widths):
        raise ValueError("Model limits cannot contain the requested cohort")
    config = dict(
        model=str(args.model),
        tensor_parallel_size=4,
        dtype="half",
        kv_cache_dtype=args.kv_cache_dtype,
        max_model_len=model_len,
        max_num_batched_tokens=args.max_batch,
        max_num_seqs=max_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        disable_log_stats=False,
        language_model_only=True,
        enforce_eager=args.eager,
        compilation_config={"mode": 3, "cudagraph_mode": "FULL"},
    )
    if args.eager:
        config.pop("compilation_config")
    if args.model.suffix.lower() == ".gguf":
        config["quantization"] = "gguf"
    if args.mtp_draft is not None:
        config["speculative_config"] = {
            "method": "mtp",
            "model": str(args.mtp_draft),
            "num_speculative_tokens": 4,
            "draft_load_config": {"load_format": "safetensors"},
            "draft_sample_method": "greedy",
        }
    if args.ring == "disabled":
        config["kernel_config"] = {"sm70_ring": {"enabled": False}}
    if args.ssm_state_dtype:
        config["mamba_ssm_cache_dtype"] = args.ssm_state_dtype
    if args.record_first_logprobs:
        config.update(max_logprobs=-1, logprobs_mode="raw_logprobs")
    report = {
        "vllm_version": vllm.__version__,
        "vllm_origin": vllm.__file__,
        "loaded_core_sha256": hashlib.sha256(
            Path(core.__file__).read_bytes()
        ).hexdigest(),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        # Engine initialization mutates nested speculative config in place.
        # Retain the JSON input contract before it contains ModelConfig objects.
        "config": json.loads(json.dumps(config)),
        "decode_contract": {
            "input_len": args.input_len,
            "concurrent_input_len": args.concurrent_input_len or args.input_len,
            "output_len": args.output_len,
            "concurrent_output_len": args.concurrent_output_len or args.output_len,
            "synthetic": True,
            "ignore_eos": True,
            "temperature": args.temperature,
            "atomic_cohort": True,
            "no_mtp": args.mtp_draft is None,
        },
        "natural_greedy": [],
        "first_logprobs": [],
        "decode": [],
        "prefill": [],
        "complete": False,
        "cuda_profiler_capture": args.cuda_profiler_capture,
        "ring_policy": args.ring,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n"
        )

    save()
    llm = LLM(**config)
    try:
        report["collectives"] = [
            {
                "rank": row["rank"],
                "selections": row.get("collective_kernel_selections", {}),
            }
            for row in llm.collective_rpc("get_sm70_acceleration_report")
        ]
        resolved = llm.llm_engine.vllm_config.compilation_config
        report["resolved_compilation"] = {
            "mode": resolved.mode.name,
            "cudagraph_mode": resolved.cudagraph_mode.name,
            "decode_cudagraph_mode": resolved.cudagraph_mode.decode_mode().name,
            "mixed_cudagraph_mode": resolved.cudagraph_mode.mixed_mode().name,
        }
        save()
        if not args.eager and resolved.cudagraph_mode.decode_mode().name != "FULL":
            raise RuntimeError(
                "FULL decode CUDA graph is unavailable with "
                f"{resolved.cudagraph_mode.name}; no timing results recorded"
            )
        report["worker_routes"] = llm.collective_rpc(
            "get_sm70_acceleration_report", timeout=30
        )
        save()
        for worker in report["worker_routes"]:
            if not args.eager and worker["decode_cudagraph_mode"] != "FULL":
                raise RuntimeError(
                    f"Rank {worker['rank']} cannot run FULL decode CUDA graph: "
                    f"{worker['cudagraph_mode']}; no timing results recorded"
                )
        tokenizer = llm.get_tokenizer()
        rows = json.loads(args.prompts_json.read_text())
        if args.record_first_logprobs:
            first = llm.generate(
                [{"prompt_token_ids": r["prompt_token_ids"]} for r in rows],
                SamplingParams(temperature=0, max_tokens=1, logprobs=-1),
                use_tqdm=False,
            )
            for index, output in enumerate(first):
                entries = output.outputs[0].logprobs[0]
                values = np.full(len(tokenizer), np.nan, dtype=np.float32)
                for token_id, entry in entries.items():
                    values[token_id] = entry.logprob
                if not np.isfinite(values).all():
                    raise RuntimeError("Incomplete or nonfinite first-logprob vector")
                path = args.output.with_name(
                    f"{args.output.stem}.prompt-{index}.logprobs.npy"
                )
                np.save(path, values)
                top = np.argsort(values)[-10:][::-1]
                report["first_logprobs"].append(
                    {
                        "index": index,
                        "path": str(path),
                        "vocabulary": len(values),
                        "representation": "log_softmax(raw_logits)",
                        "top10_ids": top.tolist(),
                        "top10_logprobs": values[top].tolist(),
                    }
                )
            save()
        natural = llm.generate(
            [{"prompt_token_ids": r["prompt_token_ids"]} for r in rows],
            SamplingParams(temperature=0, max_tokens=64),
            use_tqdm=False,
        )
        for row, output in zip(rows, natural, strict=True):
            completion = output.outputs[0]
            report["natural_greedy"].append(
                {
                    "prompt": row["prompt"],
                    "ids": list(completion.token_ids),
                    "text": completion.text,
                    "finish_reason": completion.finish_reason,
                }
            )
        save()

        def fixed_prompt(length, index=0):
            piece = tokenizer.encode(
                f"Independent stream {index}: numerical methods and reliable systems. ",
                add_special_tokens=False,
            )
            return {"prompt_token_ids": (piece * (length // len(piece) + 1))[:length]}

        for width in args.widths:
            length = (
                args.concurrent_input_len
                if width > 1 and args.concurrent_input_len
                else args.input_len
            )
            output_length = (
                args.concurrent_output_len
                if width > 1 and args.concurrent_output_len
                else args.output_len
            )
            sampling = SamplingParams(
                temperature=args.temperature,
                seed=4201,
                max_tokens=output_length,
                ignore_eos=True,
            )
            cohort = [fixed_prompt(length, i) for i in range(width)]
            generate_cohort(llm, cohort, sampling, atomic=True)
            for repeat in range(args.repeats):
                records = []
                client = llm.llm_engine.engine_core
                original = client.get_output

                def observed(original=original, records=records):
                    output = original()
                    scheduler = output.scheduler_stats
                    items = sorted(output.outputs, key=lambda x: x.request_id)
                    records.append(
                        {
                            "timestamp": output.timestamp,
                            "running": scheduler.num_running_reqs
                            if scheduler
                            else None,
                            "waiting": scheduler.num_waiting_reqs
                            if scheduler
                            else None,
                            "counts": [len(x.new_token_ids) for x in items],
                            "request_ids": [x.request_id for x in items],
                            "prefill": any(x.prefill_stats is not None for x in items),
                            "finished": any(x.finished for x in items),
                        }
                    )
                    return output

                client.get_output = observed
                before = _metric_snapshot(llm)
                capture = (
                    args.cuda_profiler_capture
                    and width == args.widths[0]
                    and repeat == 0
                )
                try:
                    if capture:
                        torch.accelerator.synchronize()
                        torch.cuda.cudart().cudaProfilerStart()
                    outputs = generate_cohort(llm, cohort, sampling, atomic=True)
                finally:
                    if capture:
                        torch.accelerator.synchronize()
                        torch.cuda.cudart().cudaProfilerStop()
                    client.get_output = original
                if any(len(o.outputs[0].token_ids) != output_length for o in outputs):
                    raise RuntimeError("Incomplete synthetic timing request")
                summary = summarize(records, width)
                spec_decoding = _spec_decoding_delta(before, _metric_snapshot(llm))
                report["decode"].append(
                    {
                        "repeat": repeat,
                        "input_len": length,
                        "output_len": output_length,
                        **summary,
                        "raw_steps": records,
                        "spec_decoding": spec_decoding,
                        "output_token_ids": [
                            list(o.outputs[0].token_ids) for o in outputs
                        ],
                        "requests": [
                            _request_metrics_dict(
                                o.metrics, len(o.outputs[0].token_ids)
                            )
                            for o in outputs
                        ],
                    }
                )
                save()
                print(json.dumps({"repeat": repeat, **summary}), flush=True)
        for length in args.prefill:
            prompt = fixed_prompt(length)
            params = SamplingParams(temperature=0, max_tokens=1)
            llm.generate(prompt, params, use_tqdm=False)
            for repeat in range(args.repeats):
                start = time.perf_counter()
                output = llm.generate(prompt, params, use_tqdm=False)[0]
                elapsed = time.perf_counter() - start
                metrics = _request_metrics_dict(
                    output.metrics, len(output.outputs[0].token_ids)
                )
                row = {
                    "input_len": length,
                    "repeat": repeat,
                    "wall_seconds": elapsed,
                    "metrics": metrics,
                }
                report["prefill"].append(row)
                save()
                print(json.dumps(row), flush=True)
        report["complete"] = True
        save()
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
