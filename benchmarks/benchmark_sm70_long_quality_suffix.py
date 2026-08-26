# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Long-context suffix-following quality check for SM70 decode work.

This script keeps the prompt shape deterministic while making the final suffix
semantically meaningful. It is intended to catch obvious long-context decode
quality failures such as repeated single-token collapse, replacement characters,
or losing the final instruction.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any


def _hash_ids(token_ids: list[int]) -> str:
    encoded = ",".join(str(token_id) for token_id in token_ids).encode()
    return hashlib.sha256(encoded).hexdigest()


def _build_prompt_ids(
    tokenizer: Any,
    target_len: int,
    filler: str,
    suffix: str,
) -> list[int]:
    filler_ids = tokenizer.encode(filler, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    if not filler_ids:
        raise ValueError("--filler produced no tokens")
    if not suffix_ids:
        raise ValueError("--suffix produced no tokens")

    prefix_len = target_len - len(suffix_ids)
    if prefix_len <= 0:
        raise ValueError(
            f"target length {target_len} is shorter than suffix length "
            f"{len(suffix_ids)}"
        )

    repeats = (prefix_len + len(filler_ids) - 1) // len(filler_ids)
    return (filler_ids * repeats)[:prefix_len] + suffix_ids


def _request_metrics_dict(metrics: Any, output_tokens: int) -> dict[str, Any] | None:
    if metrics is None:
        return None

    decode_time = (
        metrics.last_token_ts - metrics.first_token_ts
        if metrics.last_token_ts and metrics.first_token_ts
        else None
    )
    prefill_time = (
        metrics.first_token_ts - metrics.scheduled_ts
        if metrics.first_token_ts and metrics.scheduled_ts
        else None
    )
    steady_decode_tokens = max(output_tokens - 1, 0)
    return {
        "first_token_latency": metrics.first_token_latency,
        "prefill_time": prefill_time,
        "decode_time": decode_time,
        "steady_decode_tokens": steady_decode_tokens,
        "steady_decode_tps": (
            steady_decode_tokens / decode_time
            if decode_time and steady_decode_tokens > 0
            else None
        ),
        "raw": {
            "is_corrupted": metrics.is_corrupted,
            "arrival_time": metrics.arrival_time,
            "queued_ts": metrics.queued_ts,
            "scheduled_ts": metrics.scheduled_ts,
            "first_token_ts": metrics.first_token_ts,
            "last_token_ts": metrics.last_token_ts,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--input-lens",
        type=int,
        nargs="+",
        default=[16384, 32768, 65536],
    )
    parser.add_argument("--output-len", type=int, default=512)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--quantization", default="awq")
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--filler",
        default=(
            "Long-context quality check material. This paragraph is repeated "
            "only to fill the context window. The important instruction is at "
            "the very end of the prompt. Do not answer until the final question "
            "appears. "
        ),
    )
    parser.add_argument(
        "--suffix",
        default=(
            "\n\n### Final question\n"
            "Ignore the repeated filler above and answer only this final "
            "question. Output exactly four short lines:\n"
            "Long context quality check\n"
            "No garbled text should appear at 16K, 32K, or 64K\n"
            "I can follow the final instruction\n"
            "Done\n"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from transformers import AutoTokenizer

    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model),
        trust_remote_code=args.trust_remote_code,
    )
    llm = LLM(
        model=str(args.model),
        tensor_parallel_size=args.tensor_parallel_size,
        quantization=args.quantization,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
        seed=args.seed,
    )
    sampling_params = SamplingParams(
        max_tokens=args.output_len,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        ignore_eos=args.ignore_eos,
        skip_special_tokens=False,
    )

    results = []
    for input_len in args.input_lens:
        prompt_ids = _build_prompt_ids(
            tokenizer,
            input_len,
            args.filler,
            args.suffix,
        )
        start = time.perf_counter()
        outputs = llm.generate(
            [{"prompt_token_ids": prompt_ids}],
            sampling_params,
            use_tqdm=False,
        )
        elapsed = time.perf_counter() - start
        sequence = outputs[0].outputs[0]
        output_ids = list(sequence.token_ids)
        text = sequence.text
        top_token = (
            max((output_ids.count(token_id), token_id) for token_id in set(output_ids))
            if output_ids
            else (0, None)
        )
        result = {
            "name": f"i{input_len}_o{args.output_len}_quality_suffix",
            "prompt": {
                "input_len": len(prompt_ids),
                "token_hash": _hash_ids(prompt_ids),
            },
            "sampling_params": {
                "max_tokens": args.output_len,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "ignore_eos": args.ignore_eos,
                "skip_special_tokens": False,
            },
            "elapsed_seconds": elapsed,
            "output_tokens": len(output_ids),
            "output_tps": len(output_ids) / elapsed if elapsed > 0 else None,
            "request_metrics": _request_metrics_dict(
                outputs[0].metrics,
                len(output_ids),
            ),
            "finish_reason": sequence.finish_reason,
            "token_hash": _hash_ids(output_ids),
            "unique_tokens": len(set(output_ids)),
            "top_token_count_id": top_token,
            "replacement_chars": text.count("\ufffd"),
            "im_end_count": text.count("<|im_end|>"),
            "text": text,
            "prefix": text[:500],
            "suffix": text[-500:],
        }
        results.append(result)
        print(
            json.dumps(
                {
                    "input_len": len(prompt_ids),
                    "output_tokens": len(output_ids),
                    "finish_reason": sequence.finish_reason,
                    "replacement_chars": result["replacement_chars"],
                    "im_end_count": result["im_end_count"],
                    "unique_tokens": result["unique_tokens"],
                    "top_token_count_id": result["top_token_count_id"],
                    "steady_decode_tps": (
                        result["request_metrics"] or {}
                    ).get("steady_decode_tps"),
                    "prefix": text[:160],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    output = {
        "model": str(args.model),
        "input_lens": args.input_lens,
        "filler": args.filler,
        "suffix": args.suffix,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"WROTE {args.out}", flush=True)


if __name__ == "__main__":
    main()
