# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online mixed load: two resident decoders and two incoming long prompts.

Input is a JSON object with a ``prompts`` list of token-ID lists. Resident
requests ignore EOS only to hold the load steady; run natural-output quality
checks separately. Counts come from returned token IDs, never SSE chunk counts.
"""

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path

import httpx


async def measure(args, prompts):
    records = [
        {"events": [], "token_ids": [], "text": "", "usage": None} for _ in range(4)
    ]
    ready = [asyncio.Event() for _ in range(4)]
    residents_ready = [asyncio.Event() for _ in range(2)]
    salt = f"mixed-prefill-{time.time_ns()}"

    async with httpx.AsyncClient(timeout=args.timeout, trust_env=False) as client:

        async def request(index):
            resident = index < 2
            record = records[index]
            body = {
                "model": args.model,
                "prompt": prompts[index],
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "seed": 20260923 + index,
                "max_tokens": args.resident_output if resident else 128,
                "ignore_eos": resident,
                "stream": True,
                "stream_options": {"include_usage": True},
                "return_token_ids": True,
                "cache_salt": f"{salt}-{index}",
            }
            record["start_s"] = time.monotonic()
            async with client.stream(
                "POST", args.url + "/v1/completions", json=body
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    if line[6:] == "[DONE]":
                        break
                    event = json.loads(line[6:])
                    if event.get("error"):
                        raise RuntimeError(event["error"])
                    if event.get("usage"):
                        record["usage"] = event["usage"]
                    for choice in event.get("choices", []):
                        ids = choice.get("token_ids") or []
                        if ids:
                            record["events"].append((time.monotonic(), len(ids)))
                            record["token_ids"].extend(ids)
                            ready[index].set()
                            if resident and len(record["token_ids"]) >= 64:
                                residents_ready[index].set()
                        record["text"] += choice.get("text") or ""
                        if choice.get("finish_reason"):
                            record["finish_reason"] = choice["finish_reason"]

        tasks = [asyncio.create_task(request(i)) for i in range(2)]
        try:
            await asyncio.wait_for(
                asyncio.gather(*(r.wait() for r in residents_ready)), args.timeout
            )
            # Both requests are now in steady decode. No pause/barrier hides
            # interaction between their decode and the arriving prefill work.
            injected = time.monotonic()
            long_tasks = [asyncio.create_task(request(i)) for i in range(2, 4)]
            tasks.extend(long_tasks)
            await asyncio.wait_for(
                asyncio.gather(*(r.wait() for r in ready[2:])), args.timeout
            )
            phase_end = max(records[i]["events"][0][0] for i in range(2, 4))
            if any(tasks[i].done() for i in range(2)):
                raise RuntimeError("A resident ended before both prefills completed")
            await asyncio.gather(*long_tasks)
            await asyncio.sleep(1)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    phase_tokens = sum(
        count
        for record in records[:2]
        for timestamp, count in record["events"]
        if injected <= timestamp <= phase_end
    )
    gaps = [
        right[0] - left[0]
        for record in records[:2]
        for left, right in zip(record["events"], record["events"][1:])
        if left[0] < phase_end and right[0] > injected
    ]
    for record in records[2:]:
        assert record["usage"]["completion_tokens"] == len(record["token_ids"])
        assert record["usage"]["prompt_tokens"] == args.input_tokens
    return {
        "resident_count": 2,
        "incoming_count": 2,
        "input_tokens": args.input_tokens,
        "resident_sampling": {"ignore_eos": True, "output_cap": args.resident_output},
        "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20},
        "injected_s": injected,
        "prefill_complete_s": phase_end,
        "prefill_window_s": phase_end - injected,
        "resident_tokens_during_prefill": phase_tokens,
        "resident_tps_during_prefill": phase_tokens / (phase_end - injected),
        "longest_resident_gap_s": max(gaps),
        "incoming_ttft_s": [r["events"][0][0] - r["start_s"] for r in records[2:]],
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-tokens", type=int, default=32768)
    parser.add_argument("--resident-output", type=int, default=32768)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    raw = args.prompts.read_bytes()
    prompts = json.loads(raw)["prompts"][:4]
    if len(prompts) != 4 or any(len(p) != args.input_tokens for p in prompts):
        raise ValueError("Four prompts of the requested token length are required")
    result = asyncio.run(measure(args, prompts))
    result["dataset_sha256"] = hashlib.sha256(raw).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
