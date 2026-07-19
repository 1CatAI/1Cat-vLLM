# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small SM70 prefix-cache end-to-end probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from typing import Any

import torch

from vllm import LLM, SamplingParams

BASE = (
    "请阅读下面的技术说明并保持上下文一致。"
    "1Cat-vLLM 在 Tesla V100 上验证前缀缓存、FlashAttention、"
    "KV cache 和 OpenAI API 服务。"
    "本段用于构造可重复的长前缀；请不要改变事实，只在回答里给出简短总结。\n"
)
TAIL = "\n问题：请用三句话总结上文。"


def _sha_ids(ids: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()


def _make_prompt_ids(tokenizer: Any, target_len: int) -> list[int]:
    base_ids = tokenizer.encode(BASE, add_special_tokens=False)
    tail_ids = tokenizer.encode(TAIL, add_special_tokens=False)
    ids: list[int] = []
    while len(ids) + len(base_ids) + len(tail_ids) < target_len:
        ids.extend(base_ids)
    return ids[: max(0, target_len - len(tail_ids))] + tail_ids


def _request_ttft(output: Any) -> float | None:
    metrics = getattr(output, "metrics", None)
    if metrics is None:
        return None
    ttft = getattr(metrics, "first_token_latency", None)
    if ttft is not None:
        return ttft
    first = getattr(metrics, "first_token_ts", None)
    scheduled = getattr(metrics, "scheduled_ts", None)
    if first and scheduled:
        return first - scheduled
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/ymzx/models/Qwen3.6-27B-AWQ")
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--prompt-len", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()

    kwargs: dict[str, Any] = {
        "model": args.model,
        "trust_remote_code": True,
        "tensor_parallel_size": 4,
        "dtype": "half",
        "quantization": "awq",
        "max_model_len": 8192,
        "max_num_batched_tokens": 4096,
        "max_num_seqs": 1,
        "gpu_memory_utilization": 0.88,
        "attention_backend": "FLASH_ATTN_V100",
        "disable_log_stats": False,
        "enable_prefix_caching": True,
        "mamba_cache_mode": "align",
        "seed": 123,
    }
    if args.kv_cache_dtype != "auto":
        kwargs["kv_cache_dtype"] = args.kv_cache_dtype

    print(
        f"=== prefix-cache probe kv_cache_dtype={args.kv_cache_dtype} ===",
        flush=True,
    )
    llm = LLM(**kwargs)
    tokenizer = llm.get_tokenizer()
    prompt_ids = _make_prompt_ids(tokenizer, args.prompt_len)
    sampling = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        skip_special_tokens=False,
    )

    records: list[dict[str, Any]] = []
    for repeat in (1, 2):
        torch.accelerator.synchronize()
        start = time.perf_counter()
        outputs = llm.generate(
            [{"prompt_token_ids": prompt_ids}],
            sampling,
            use_tqdm=False,
        )
        torch.accelerator.synchronize()
        elapsed = time.perf_counter() - start
        output = outputs[0]
        completion = output.outputs[0]
        record = {
            "repeat": repeat,
            "elapsed_s": elapsed,
            "ttft_s": _request_ttft(output),
            "num_cached_tokens": getattr(output, "num_cached_tokens", None),
            "output_tokens": len(completion.token_ids),
            "token_hash": _sha_ids(list(completion.token_ids)),
            "text_hash": hashlib.sha256(completion.text.encode()).hexdigest(),
            "preview": completion.text[:160],
        }
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    ratio = None
    if records[0]["ttft_s"] and records[1]["ttft_s"]:
        ratio = records[1]["ttft_s"] / records[0]["ttft_s"]
    result = {
        "kv_cache_dtype": args.kv_cache_dtype,
        "input_tokens": len(prompt_ids),
        "second_cached_tokens": records[1]["num_cached_tokens"],
        "ttft_ratio_second_over_first": ratio,
        "token_hash_match": records[0]["token_hash"] == records[1]["token_hash"],
        "text_hash_match": records[0]["text_hash"] == records[1]["text_hash"],
        "passed": (records[1]["num_cached_tokens"] or 0) > 0
        and records[0]["token_hash"] == records[1]["token_hash"],
    }
    print("RESULT " + json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
