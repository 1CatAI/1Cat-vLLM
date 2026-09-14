# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cold-cache CUDA-graph TTFT and decode measurement for Q8192 prefill.

Deterministic quality comparison; EOS is respected unless a fixed-length speed
run explicitly requests ``--ignore-eos``, and thinking is disabled. Engine
construction and a short warmup are excluded. Prefix caching, speculative
decoding, and eager execution are disabled.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path


def route_snapshot(worker):
    import sys

    import torch

    from vllm.v1.attention.backends import flash_attn_v100 as backend

    return {
        "rank": worker.rank,
        "counts": dict(backend._route_counts),
        "fa2_module": getattr(
            sys.modules.get("vllm.vllm_flash_attn._vllm_fa2_C"), "__file__", None
        ),
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
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--min-decode-intervals", type=int, default=63)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kv-cache-dtype", default="fp8_e4m3")
    parser.add_argument(
        "--quantization",
        default=None,
        help="Optional weight-quantization override; default to checkpoint metadata.",
    )
    parser.add_argument(
        "--concurrent-requests",
        type=int,
        nargs="+",
        default=[1],
        help="One or more concurrency levels to run on the same graph engine.",
    )
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--long-prefill-token-threshold", type=int, default=8192)
    args = parser.parse_args()
    if any(concurrency < 1 for concurrency in args.concurrent_requests):
        parser.error("--concurrent-requests must be positive")
    if args.min_decode_intervals < 1:
        parser.error("--min-decode-intervals must be positive")
    if max(args.concurrent_requests) > args.max_num_seqs:
        parser.error("--max-num-seqs must cover every concurrent request")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=4,
        dtype="half",
        quantization=args.quantization,
        kv_cache_dtype=args.kv_cache_dtype,
        max_model_len=262144,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        long_prefill_token_threshold=args.long_prefill_token_threshold,
        gpu_memory_utilization=0.85,
        enforce_eager=False,
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
                "content": "本次观测任务的唯一校验词是「海蓝石榴」。下面是观测记录。\n"
                + marker
                + "\n请简答：校验词是什么，太阳系最大的行星是哪颗？",
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
    warmup = llm.generate(
        "请用中文说你好。", SamplingParams(temperature=0, max_tokens=8), use_tqdm=False
    )
    print(
        "WARMUP " + repr([(x.outputs[0].token_ids, x.outputs[0].text) for x in warmup]),
        flush=True,
    )
    reports = []
    try:
        for length in args.lengths:
            n = length - len(head) - len(tail)
            if n < 0:
                raise ValueError("Prompt length is too short for the task")
            ids = head + (record * ((n + len(record) - 1) // len(record)))[:n] + tail
            for concurrent_requests in args.concurrent_requests:
                params = SamplingParams(
                    temperature=0,
                    top_p=1,
                    top_k=-1,
                    max_tokens=args.output_len,
                    ignore_eos=args.ignore_eos,
                    output_kind=RequestOutputKind.DELTA,
                )
                engine = llm.llm_engine
                before = engine.collective_rpc(route_snapshot)
                start = time.perf_counter()
                request_ids = [
                    f"cold-{length}-b{concurrent_requests}-{index}"
                    for index in range(concurrent_requests)
                ]
                state = {
                    request_id: {
                        "first": None,
                        "last": None,
                        "tokens": [],
                        "text": "",
                        "cached": 0,
                    }
                    for request_id in request_ids
                }
                for request_id in request_ids:
                    engine.add_request(request_id, {"prompt_token_ids": ids}, params)
                while engine.has_unfinished_requests():
                    for result in engine.step():
                        now = time.perf_counter()
                        request_state = state[result.request_id]
                        request_state["cached"] = max(
                            request_state["cached"],
                            getattr(result, "num_cached_tokens", 0) or 0,
                        )
                        for output in result.outputs:
                            if output.token_ids:
                                if request_state["first"] is None:
                                    request_state["first"] = now
                                request_state["last"] = now
                                request_state["tokens"].extend(output.token_ids)
                                request_state["text"] += output.text
                end = time.perf_counter()
                after = engine.collective_rpc(route_snapshot)
                route_deltas = []
                for old, new in zip(before, after, strict=True):
                    route_deltas.append(
                        {
                            "rank": new["rank"],
                            "fa2_libraries": new["fa2_libraries"],
                            "fa2_module": new["fa2_module"],
                            "counts": {
                                key: value - old["counts"].get(key, 0)
                                for key, value in new["counts"].items()
                            },
                        }
                    )
                request_rows = []
                for request_id in request_ids:
                    request_state = state[request_id]
                    first = request_state["first"]
                    last = request_state["last"]
                    tokens = request_state["tokens"]
                    text = request_state["text"]
                    cached = request_state["cached"]
                    if first is None or last is None:
                        raise RuntimeError(f"Request {request_id} returned no token")
                    if cached:
                        raise RuntimeError(
                            f"Cold request {request_id} unexpectedly reused "
                            f"{cached} tokens"
                        )
                    decode_intervals = max(len(tokens) - 1, 0)
                    decode_seconds = last - first
                    observed_decode_tps = (
                        decode_intervals / decode_seconds
                        if decode_seconds > 0 and decode_intervals > 0
                        else None
                    )
                    observed_decode_tpot = (
                        decode_seconds / decode_intervals
                        if decode_seconds > 0 and decode_intervals > 0
                        else None
                    )
                    decode_qualified = decode_intervals >= args.min_decode_intervals
                    request_rows.append(
                        dict(
                            request_id=request_id,
                            prompt_tokens=len(ids),
                            cached_tokens=cached,
                            prompt_sha256=hashlib.sha256(
                                json.dumps(ids).encode()
                            ).hexdigest(),
                            ttft_seconds=first - start,
                            prompt_tokens_per_ttft_s=len(ids) / (first - start),
                            decode_seconds=decode_seconds,
                            decode_intervals=decode_intervals,
                            observed_decode_tokens_per_s=observed_decode_tps,
                            observed_decode_tpot_seconds=observed_decode_tpot,
                            decode_measurement_qualified=decode_qualified,
                            decode_tokens_per_s=(
                                observed_decode_tps if decode_qualified else None
                            ),
                            decode_tpot_seconds=(
                                observed_decode_tpot if decode_qualified else None
                            ),
                            output_token_ids=tokens,
                            output_text=text,
                            retrieval_pass="海蓝石榴" in text,
                            knowledge_pass="木星" in text,
                        )
                    )
                batch_ttft = max(row["ttft_seconds"] for row in request_rows)
                decode_values = [
                    row["decode_tokens_per_s"]
                    for row in request_rows
                    if row["decode_tokens_per_s"] is not None
                ]
                batch_decode_seconds = max(
                    state[request_id]["last"] for request_id in request_ids
                ) - min(state[request_id]["first"] for request_id in request_ids)
                aggregate_decode_intervals = sum(
                    row["decode_intervals"] for row in request_rows
                )
                row = dict(
                    routes=route_deltas,
                    graph=True,
                    weight_quantization_override=args.quantization,
                    kv_cache_dtype=args.kv_cache_dtype,
                    ignore_eos=args.ignore_eos,
                    min_decode_intervals=args.min_decode_intervals,
                    concurrent_requests=concurrent_requests,
                    aggregate_prompt_tokens=concurrent_requests * len(ids),
                    batch_ttft_seconds=batch_ttft,
                    wall_seconds=end - start,
                    aggregate_prompt_tokens_per_batch_ttft_s=(
                        concurrent_requests * len(ids) / batch_ttft
                    ),
                    qualified_decode_requests=len(decode_values),
                    decode_tokens_per_s_min=(
                        min(decode_values) if decode_values else None
                    ),
                    decode_tokens_per_s_mean=(
                        sum(decode_values) / len(decode_values)
                        if decode_values
                        else None
                    ),
                    decode_tokens_per_s_max=(
                        max(decode_values) if decode_values else None
                    ),
                    aggregate_decode_intervals=aggregate_decode_intervals,
                    batch_decode_seconds=batch_decode_seconds,
                    aggregate_decode_tokens_per_s=(
                        aggregate_decode_intervals / batch_decode_seconds
                        if batch_decode_seconds > 0
                        else None
                    ),
                    all_output_token_ids_equal=(
                        len(
                            {
                                tuple(request["output_token_ids"])
                                for request in request_rows
                            }
                        )
                        == 1
                    ),
                )
                if concurrent_requests == 1:
                    row.update(request_rows[0])
                else:
                    row["requests"] = request_rows
                reports.append(row)
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps(reports, ensure_ascii=False, indent=2))
                print("COLD_RESULT " + json.dumps(row, ensure_ascii=False), flush=True)
                if len(decode_values) != concurrent_requests:
                    raise RuntimeError("Sustained decode gate failed")
                if not all(
                    request["retrieval_pass"] and request["knowledge_pass"]
                    for request in request_rows
                ):
                    raise RuntimeError("Quality gate failed; skip remaining requests")
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
