# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cold-cache client TTFT and decode measurement for Q8000 prefill.

Deterministic quality comparison; EOS is respected and thinking is disabled.
Engine construction and a short warmup are excluded. Prefix caching and
speculative decoding are disabled.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path


def route_snapshot(worker):
    import torch

    from vllm.v1.attention.backends import flash_attn_v100 as backend

    return {
        "rank": worker.rank,
        "counts": dict(backend._route_counts),
        "fa2_libraries": sorted(
            p for p in torch.ops.loaded_libraries if "_vllm_fa2_C" in p
        ),
    }


def main():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[128000, 256000])
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kv-cache-dtype", default="fp8_e4m3")
    args = parser.parse_args()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=4,
        dtype="half",
        quantization="fp8",
        kv_cache_dtype=args.kv_cache_dtype,
        max_model_len=262144,
        max_num_batched_tokens=8000,
        max_num_seqs=1,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        attention_backend="FLASH_ATTN_V100",
        seed=20260825,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        mamba_cache_dtype="float16",
        mamba_ssm_cache_dtype="float16",
    )
    tokenizer = llm.get_tokenizer()
    # Repeated natural-language records keep lengths exact and the task readable.
    marker = "OBSERVATION_RECORDS_PLACEHOLDER"
    rendered = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "下面是供参考的观测记录。\n"
                + marker
                + "\n请忽略重复记录，只用中文简短回答：太阳系最大的行星是哪一颗？",
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prefix, suffix = rendered.split(marker)
    head = tokenizer.encode(prefix, add_special_tokens=False)
    tail = tokenizer.encode(suffix, add_special_tokens=False)
    record = tokenizer.encode(
        "观测站记录：今天天气晴朗，设备正常运行，数据已经保存。\n",
        add_special_tokens=False,
    )
    llm.generate(
        "请用中文说你好。", SamplingParams(temperature=0, max_tokens=8), use_tqdm=False
    )
    reports = []
    try:
        for length in args.lengths:
            n = length - len(head) - len(tail)
            if n < 0:
                raise ValueError("Prompt length is too short for the task")
            ids = head + (record * ((n + len(record) - 1) // len(record)))[:n] + tail
            params = SamplingParams(
                temperature=0,
                top_p=1,
                top_k=-1,
                max_tokens=args.output_len,
                output_kind=RequestOutputKind.DELTA,
            )
            engine = llm.llm_engine
            before = engine.collective_rpc(route_snapshot)
            start = time.perf_counter()
            engine.add_request(f"cold-{length}", {"prompt_token_ids": ids}, params)
            first = None
            last = None
            tokens = []
            text = ""
            cached = 0
            while engine.has_unfinished_requests():
                for result in engine.step():
                    now = time.perf_counter()
                    cached = max(cached, getattr(result, "num_cached_tokens", 0) or 0)
                    for output in result.outputs:
                        if output.token_ids:
                            if first is None:
                                first = now
                            last = now
                            tokens.extend(output.token_ids)
                            text += output.text
            end = time.perf_counter()
            after = engine.collective_rpc(route_snapshot)
            route_deltas = []
            for old, new in zip(before, after, strict=True):
                route_deltas.append(
                    {
                        "rank": new["rank"],
                        "fa2_libraries": new["fa2_libraries"],
                        "counts": {
                            key: value - old["counts"].get(key, 0)
                            for key, value in new["counts"].items()
                        },
                    }
                )
            if first is None:
                raise RuntimeError("Request returned no token")
            if cached:
                raise RuntimeError(f"Cold request unexpectedly reused {cached} tokens")
            row = dict(
                routes=route_deltas,
                prompt_tokens=len(ids),
                cached_tokens=cached,
                prompt_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                ttft_seconds=first - start,
                wall_seconds=end - start,
                prompt_tokens_per_ttft_s=len(ids) / (first - start),
                decode_seconds=last - first,
                decode_tokens_per_s=(len(tokens) - 1) / (last - first)
                if last > first
                else None,
                output_token_ids=tokens,
                output_text=text,
            )
            reports.append(row)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(reports, ensure_ascii=False, indent=2))
            print("COLD_RESULT " + json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
