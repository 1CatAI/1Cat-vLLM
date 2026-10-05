# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Matched Flash-Next no-MTP endpoint with explicit timing/health separation.

Hold the complete GPU lease externally. Keep all CLI arguments and environment
identical when changing source revisions. Raw reports retain the resolved
runtime, native hashes, worker routes, prompt hash and every generated token.
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--input-len", type=int, default=8192)
    parser.add_argument("--output-len", type=int, default=513)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--health-only", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.94)
    args = parser.parse_args()
    import os

    import torch
    from transformers import AutoTokenizer

    import vllm
    from vllm import LLM, SamplingParams

    contract = dict(
        model=str(args.model),
        tensor_parallel_size=4,
        dtype="half",
        kv_cache_dtype="float16",
        mamba_ssm_cache_dtype="float32",
        max_model_len=262144,
        max_num_batched_tokens=8192,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        disable_log_stats=False,
        enable_prefix_caching=False,
        language_model_only=True,
        speculative_config=None,
    )
    report = dict(
        complete=False,
        source_sha=args.source_sha,
        engine=contract,
        runtime=vllm.__file__,
        version=vllm.__version__,
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        env={
            k: v
            for k, v in os.environ.items()
            if k.startswith(("VLLM_", "CUDA_", "TORCH_", "TRITON_"))
        },
        health=[],
        timing=[],
        started_unix=time.time(),
        native_build_provenance=(
            json.loads(Path(".artifacts/native-provenance.json").read_text())
            if Path(".artifacts/native-provenance.json").exists()
            else None
        ),
        native_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(vllm.__file__).parent.glob("*.so")
        },
    )

    def save():
        temporary = args.output.with_suffix(".tmp.json")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(args.output)

    def metrics(result):
        output = result.outputs[0]
        n = len(output.token_ids)
        m = result.metrics
        pure = m.last_token_ts - m.first_token_ts
        return dict(
            tokens=n,
            token_ids=list(output.token_ids),
            text=output.text,
            finish_reason=output.finish_reason,
            decode_seconds=pure,
            tpot_ms=1000 * pure / (n - 1) if n > 1 else None,
            ttft_seconds=m.first_token_latency,
            prefill_seconds=m.first_token_ts - m.scheduled_ts,
        )

    save()
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    gen = json.loads((args.model / "generation_config.json").read_text())
    llm = LLM(**contract)
    try:
        report["workers"] = llm.collective_rpc("get_sm70_acceleration_report")
        for prompt in (
            "计算19乘以23，在最后一行写RESULT=结果。",
            "原样抄写：CEDAR-47|8261，不要添加其他内容。",
        ):
            ids = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            output = llm.generate(
                [{"prompt_token_ids": ids}],
                SamplingParams(
                    temperature=gen["temperature"],
                    top_p=gen["top_p"],
                    top_k=gen["top_k"],
                    seed=4201,
                    max_tokens=1024,
                ),
                use_tqdm=False,
            )[0]
            report["health"].append(dict(prompt=prompt, **metrics(output)))
            save()
        if args.health_only:
            report["speed_acceptance"] = False
            report["complete"] = True
            save()
            return
        piece = tok.encode(
            "This fixed benchmark prompt is used to create a deterministic "
            "tokenized input for single-request decode measurement. ",
            add_special_tokens=False,
        )
        ids = (piece * ((args.input_len + len(piece) - 1) // len(piece)))[
            : args.input_len
        ]
        report["prompt_sha256"] = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        llm.generate(
            [{"prompt_token_ids": ids}],
            SamplingParams(temperature=0, max_tokens=32, ignore_eos=True),
            use_tqdm=False,
        )
        for i in range(args.repeats):
            result = llm.generate(
                [{"prompt_token_ids": ids}],
                SamplingParams(
                    temperature=0,
                    top_p=1,
                    top_k=-1,
                    seed=0,
                    max_tokens=args.output_len,
                    ignore_eos=True,
                ),
                use_tqdm=False,
            )[0]
            report["timing"].append(metrics(result))
            save()
            print("PURE_DECODE", i, report["timing"][-1]["tpot_ms"], flush=True)
        times = [row["tpot_ms"] for row in report["timing"]]
        report["median_tpot_ms"] = statistics.median(times)
        report["median_decode_tps"] = 1000 / report["median_tpot_ms"]
        report["complete"] = True
        save()
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
