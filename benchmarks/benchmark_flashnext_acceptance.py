# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Matched natural-prompt MTP4 acceptance and TP graph CPU-stage observations."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import vllm._C as core

import vllm
from vllm import LLM, SamplingParams

sys.path.append(str(Path(__file__).resolve().parents[1]))
from benchmarks.benchmark_sm70_model_tokens import (  # noqa: E402
    _metric_snapshot,
    _request_metrics_dict,
    _spec_decoding_delta,
)
from benchmarks.benchmark_sm70_qwen38_concurrency import (  # noqa: E402
    generate_cohort,
    summarize,
)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()


def observed_cohort(llm, fixed_ids, params, width=1):
    records = []
    client = llm.llm_engine.engine_core
    original = client.get_output

    def observed():
        output = original()
        scheduler = output.scheduler_stats
        items = sorted(output.outputs, key=lambda row: row.request_id)
        records.append(
            {
                "timestamp": output.timestamp,
                "running": scheduler.num_running_reqs if scheduler else None,
                "waiting": scheduler.num_waiting_reqs if scheduler else None,
                "counts": [len(row.new_token_ids) for row in items],
                "request_ids": [row.request_id for row in items],
                "prefill": any(row.prefill_stats is not None for row in items),
                "finished": any(row.finished for row in items),
            }
        )
        return output

    client.get_output = observed
    try:
        outputs = generate_cohort(
            llm,
            [{"prompt_token_ids": fixed_ids} for _ in range(width)],
            params,
            atomic=True,
        )
    finally:
        client.get_output = original
    return records, outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--node-trace", action="store_true")
    parser.add_argument("--trace-only", action="store_true")
    parser.add_argument("--input-phase-ab", action="store_true")
    parser.add_argument("--require-installed", action="store_true")
    parser.add_argument("--diagnose-attention-transfers", action="store_true")
    parser.add_argument("--kernel-config", type=json.loads, default={})
    parser.add_argument("--completion-prompts", type=Path)
    parser.add_argument("--teacher-forcing", action="store_true")
    parser.add_argument("--teacher-reference", type=Path)
    parser.add_argument("--teacher-positions", type=int, default=8)
    args = parser.parse_args()
    if args.trace_only:
        args.probe = args.node_trace = True
    if args.require_installed and "site-packages" not in vllm.__file__:
        raise RuntimeError("Use a normal installed source-containing wheel")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    prompts = json.loads(args.prompts.read_text())
    if len(prompts) != 8 or len({r["id"] for r in prompts}) != 8:
        raise ValueError("Acceptance comparison requires eight distinct prompts")
    config = dict(
        model=str(args.model),
        tensor_parallel_size=4,
        dtype="half",
        kv_cache_dtype="float16",
        mamba_ssm_cache_dtype="float32",
        max_model_len=9216,
        max_num_batched_tokens=512,
        max_num_seqs=4,
        gpu_memory_utilization=args.memory,
        enable_prefix_caching=False,
        disable_log_stats=False,
        language_model_only=True,
        compilation_config={"mode": 3, "cudagraph_mode": "FULL"},
        worker_extension_cls=(
            "vllm.sm70_gguf_quality.GGUFTeacherWorkerExtension"
            if args.teacher_forcing
            else "vllm.sm70_graph_observer.GraphParityWorkerExtension"
        ),
        kernel_config=args.kernel_config,
        speculative_config={
            "method": "mtp",
            "model": str(args.draft),
            "num_speculative_tokens": 4,
            "draft_load_config": {"load_format": "safetensors"},
            "draft_sample_method": "greedy",
        },
    )
    if args.model.suffix.lower() == ".gguf":
        config["quantization"] = "gguf"
    report = dict(
        complete=False,
        config=json.loads(json.dumps(config)),
        version=vllm.__version__,
        origin=vllm.__file__,
        core_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        sampling={
            "temperature": 0,
            "top_p": 1,
            "top_k": -1,
            "ignore_eos": False,
            "max_tokens": args.max_tokens,
        },
        prompt_text_sha256=digest(prompts),
        rows=[],
        probes=[],
        completions=[],
        trace_only=args.trace_only,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n"
        )

    save()
    llm = LLM(**config)
    try:
        report["worker_routes"] = llm.collective_rpc(
            "get_sm70_acceleration_report", timeout=30
        )
        if any(r["decode_cudagraph_mode"] != "FULL" for r in report["worker_routes"]):
            raise RuntimeError("Actual FULL target decode graph is required")
        if not _metric_snapshot(llm):
            raise RuntimeError("Acceptance counters are unavailable before requests")
        tokenizer = llm.get_tokenizer()
        report["tokenizer"] = dict(
            type=type(tokenizer).__name__,
            vocab=len(tokenizer),
            chat_template=tokenizer.chat_template,
        )
        tokenized = []
        for prompt in prompts:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            ids = tokenizer.encode(rendered, add_special_tokens=False)
            tokenized.append(list(ids))
        report["prompt_tokens_sha256"] = digest(tokenized)
        params = SamplingParams(
            temperature=0,
            top_p=1,
            top_k=-1,
            max_tokens=args.max_tokens,
            ignore_eos=False,
            seed=20261005,
        )
        llm.generate(
            {"prompt_token_ids": tokenized[0]},
            SamplingParams(temperature=0, max_tokens=16),
            use_tqdm=False,
        )
        for prompt, ids in () if args.trace_only else zip(prompts, tokenized):
            before = _metric_snapshot(llm)
            started = time.perf_counter()
            result = llm.generate({"prompt_token_ids": ids}, params, use_tqdm=False)[0]
            elapsed = time.perf_counter() - started
            acceptance = _spec_decoding_delta(before, _metric_snapshot(llm))
            if acceptance is None:
                raise RuntimeError("No request-level speculative counters")
            output = result.outputs[0]
            row = dict(
                id=prompt["id"],
                prompt=prompt["prompt"],
                prompt_token_ids=ids,
                output_token_ids=list(output.token_ids),
                text=output.text,
                output_tokens=len(output.token_ids),
                finish_reason=output.finish_reason,
                wall_seconds=elapsed,
                acceptance=acceptance,
                metrics=_request_metrics_dict(result.metrics, len(output.token_ids)),
            )
            report["rows"].append(row)
            save()
            print(
                json.dumps(
                    {
                        k: row[k]
                        for k in ("id", "output_tokens", "finish_reason", "acceptance")
                    }
                ),
                flush=True,
            )
        if args.completion_prompts is not None:
            for prompt in json.loads(args.completion_prompts.read_text()):
                rendered = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                output = llm.generate(rendered, params, use_tqdm=False)[0].outputs[0]
                report["completions"].append(
                    dict(
                        prompt=prompt,
                        text=output.text,
                        output_token_ids=list(output.token_ids),
                        finish_reason=output.finish_reason,
                    )
                )
                save()
        if args.probe:
            reference = tokenized[1]
            fixed_ids = (reference * (8192 // len(reference) + 1))[:8192]
            probe_params = SamplingParams(
                temperature=0, top_p=1, top_k=-1, max_tokens=256, ignore_eos=True
            )
            llm.generate({"prompt_token_ids": fixed_ids}, probe_params, use_tqdm=False)
            if args.diagnose_attention_transfers:
                llm.collective_rpc("start_attention_transfer_diagnosis", timeout=30)
                llm.generate(
                    {"prompt_token_ids": fixed_ids}, probe_params, use_tqdm=False
                )
                report["attention_transfers"] = llm.collective_rpc(
                    "read_attention_transfer_diagnosis", timeout=30
                )
                save()
            arms = (
                ("late_observed", "early_observed", "early_off", "late_off")
                if args.input_phase_ab
                else ("off_before", "cpu_observed", "off_after")
            )
            for arm in () if args.trace_only else arms:
                if args.input_phase_ab:
                    llm.collective_rpc(
                        "set_graph_input_preparation",
                        args=(arm.startswith("early"),),
                        timeout=30,
                    )
                observing = arm == "cpu_observed" or arm.endswith("_observed")
                if observing:
                    llm.collective_rpc(
                        "start_graph_parity_observer", args=(False,), timeout=30
                    )
                steps, outputs = observed_cohort(llm, fixed_ids, probe_params)
                probe = dict(
                    arm=arm,
                    summary=summarize(steps, 1),
                    output_token_ids=[list(o.outputs[0].token_ids) for o in outputs],
                )
                if observing:
                    probe["workers"] = llm.collective_rpc(
                        "read_graph_parity_observer", timeout=30
                    )
                report["probes"].append(probe)
                save()
                print(json.dumps(dict(arm=arm, summary=probe["summary"])), flush=True)
            if args.input_phase_ab:
                llm.collective_rpc(
                    "set_graph_input_preparation", args=(True,), timeout=30
                )
            if not args.trace_only:
                c4_ids = fixed_ids[:128]
                c4_params = SamplingParams(
                    temperature=0, max_tokens=600, ignore_eos=True
                )
                c4_phases = (False, True) if args.input_phase_ab else (True,)
                report["c4_probes"] = []
                for early in c4_phases:
                    if args.input_phase_ab:
                        llm.collective_rpc(
                            "set_graph_input_preparation", args=(early,), timeout=30
                        )
                    steps, outputs = observed_cohort(llm, c4_ids, c4_params, width=4)
                    cohort = dict(
                        early=early,
                        summary=summarize(steps, 4),
                        output_token_ids=[
                            list(o.outputs[0].token_ids) for o in outputs
                        ],
                    )
                    report["c4_probes"].append(cohort)
                    report["c4_probe"] = cohort
                    save()
            if args.node_trace:
                llm.collective_rpc(
                    "start_graph_parity_observer", args=(True,), timeout=30
                )
                llm.collective_rpc("start_graph_parity_capture", timeout=30)
                try:
                    steps, outputs = observed_cohort(
                        llm,
                        fixed_ids,
                        SamplingParams(temperature=0, max_tokens=256, ignore_eos=True),
                    )
                    report["node_trace"] = dict(
                        output_token_ids=[
                            list(o.outputs[0].token_ids) for o in outputs
                        ],
                        generation_complete=True,
                        workers=llm.collective_rpc(
                            "read_graph_parity_observer", timeout=30
                        ),
                    )
                    # Preserve completed generation and CPU records before
                    # profiler shutdown or optional interval statistics.
                    save()
                    report["node_trace"]["summary"] = summarize(steps, 1)
                    save()
                finally:
                    llm.collective_rpc("stop_graph_parity_capture", timeout=30)
        if args.teacher_forcing:
            reference = (
                json.loads(args.teacher_reference.read_text())
                if args.teacher_reference is not None
                else report
            )
            if reference["prompt_tokens_sha256"] != report["prompt_tokens_sha256"]:
                raise RuntimeError(
                    "Teacher prompts/tokenizer differ from the reference"
                )
            root = args.output.with_suffix("").with_name(args.output.stem + "-teacher")
            root.mkdir(parents=True, exist_ok=True)
            report["teacher_forcing"] = dict(directory=str(root), rows=[])
            for row in reference["rows"]:
                if len(row["output_token_ids"]) <= args.teacher_positions:
                    raise RuntimeError("Teacher continuation is too short")
                for position in range(args.teacher_positions):
                    prefix = (
                        row["prompt_token_ids"] + row["output_token_ids"][:position]
                    )
                    forced = row["output_token_ids"][position]
                    key = f"{row['id']}-{position:03d}"
                    llm.collective_rpc(
                        "start_teacher_capture", args=(str(root), key), timeout=30
                    )
                    try:
                        llm.generate(
                            {"prompt_token_ids": prefix},
                            SamplingParams(
                                temperature=0, max_tokens=6, allowed_token_ids=[forced]
                            ),
                            use_tqdm=False,
                        )
                    finally:
                        workers = llm.collective_rpc("stop_teacher_capture", timeout=30)
                    if any(w["captured"] != 1 for w in workers):
                        raise RuntimeError(
                            f"M5 teacher target was not captured: {workers}"
                        )
                    captured = torch.load(root / f"{key}.pt", weights_only=True)
                    if (
                        captured["position"].item() != len(prefix)
                        or captured["input_ids"].item() != forced
                    ):
                        raise RuntimeError(
                            "Captured distribution has different teacher conditioning"
                        )
                    report["teacher_forcing"]["rows"].append(
                        dict(
                            key=key,
                            prefix_sha256=digest(prefix),
                            forced=forced,
                            position=len(prefix),
                            workers=workers,
                        )
                    )
                    save()
        report["complete"] = True
        save()
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
