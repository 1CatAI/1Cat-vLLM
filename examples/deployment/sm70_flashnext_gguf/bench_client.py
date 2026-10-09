# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single/multi-stream OpenAI-API latency probe for the Flash-Next service.

For each prompt length: TTFT, prefill tokens/s, decode tokens/s and MTP
acceptance (from /metrics deltas). Prompts are built from repeated text and
measured with the server's own usage counts.
"""

import argparse
import concurrent.futures
import json
import re
import time
import urllib.request

FILLER = (
    "The history of computing spans mechanical calculators, vacuum tubes, "
    "transistors and integrated circuits. Each generation traded generality, "
    "cost and speed in different ways, and the software that ran on them "
    "evolved from hand-written machine code to high-level languages. "
)


def post(url, body, stream=False, timeout=3600):
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=timeout)


def metrics(base):
    text = urllib.request.urlopen(base + "/metrics", timeout=30).read().decode()
    out = {}
    for name in (
        "vllm:spec_decode_num_accepted_tokens_total",
        "vllm:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_drafts_total",
    ):
        vals = re.findall(rf"^{re.escape(name)}{{[^}}]*}} ([0-9.e+]+)$", text, re.M)
        out[name] = sum(float(v) for v in vals)
    return out


def one(base, model, prompt, max_tokens):
    body = dict(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0,
        ignore_eos=True,
        stream=True,
        stream_options={"include_usage": True},
    )
    start = time.perf_counter()
    first = None
    stamps = []
    usage = None
    text = []
    with post(base + "/v1/completions", body, stream=True) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                if choice.get("text"):
                    now = time.perf_counter()
                    first = first or now
                    stamps.append(now)
                    text.append(choice["text"])
    end = time.perf_counter()
    prompt_tokens = usage["prompt_tokens"] if usage else None
    completion = usage["completion_tokens"] if usage else None
    ttft = (first - start) if first else None
    decode_s = (end - first) if first else None
    return dict(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion,
        ttft_s=ttft,
        prefill_tps=(prompt_tokens / ttft) if ttft and prompt_tokens else None,
        decode_s=decode_s,
        decode_tps=((completion - 1) / decode_s) if decode_s and completion else None,
        text_head="".join(text)[:120],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8000")
    p.add_argument("--model", required=True)
    p.add_argument("--lengths", default="1000,8000,30000")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    results = []
    for length in [int(x) for x in a.lengths.split(",")]:
        # ~45 tokens per filler sentence block for this tokenizer; trimmed by words.
        words = (FILLER * (length // 40 + 2)).split()
        prompt_words = words[: int(length * 0.74)]
        for rep in range(a.repeats):
            prompts = [
                f"[{rep}-{i}] " + " ".join(prompt_words) + "\nSummarize the above."
                for i in range(a.concurrency)
            ]
            before = metrics(a.base)
            t0 = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(a.concurrency) as pool:
                rows = list(
                    pool.map(lambda pr: one(a.base, a.model, pr, a.max_tokens), prompts)
                )
            wall = time.perf_counter() - t0
            after = metrics(a.base)
            acc = (
                after["vllm:spec_decode_num_accepted_tokens_total"]
                - before["vllm:spec_decode_num_accepted_tokens_total"]
            )
            drafts = (
                after["vllm:spec_decode_num_drafts_total"]
                - before["vllm:spec_decode_num_drafts_total"]
            )
            dtok = (
                after["vllm:spec_decode_num_draft_tokens_total"]
                - before["vllm:spec_decode_num_draft_tokens_total"]
            )
            rec = dict(
                target_length=length,
                repeat=rep,
                concurrency=a.concurrency,
                wall_s=wall,
                rows=rows,
                acceptance_rate=(acc / dtok) if dtok else None,
                tokens_per_draft_round=(1 + acc / drafts) if drafts else None,
            )
            results.append(rec)
            r0 = rows[0]
            print(
                json.dumps(
                    dict(
                        len=r0["prompt_tokens"],
                        c=a.concurrency,
                        ttft=round(r0["ttft_s"] or 0, 3),
                        prefill_tps=round(r0["prefill_tps"] or 0, 1),
                        decode_tps=round(r0["decode_tps"] or 0, 2),
                        acc=round(rec["acceptance_rate"] or 0, 4),
                        tok_per_round=round(rec["tokens_per_draft_round"] or 0, 3),
                    )
                ),
                flush=True,
            )
            with open(a.output, "w") as stream:
                json.dump(results, stream, indent=1)


if __name__ == "__main__":
    main()
