# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Concurrency benchmark and correctness check for the V4.1 PP4 server.

For each N: N streaming greedy chat requests start together (distinct prompts).
Reports per-request decode tok/s, aggregate tok/s over the window where all N
are decoding, rounds (speculative steps) from /metrics and tokens per round.
With --reference, each prompt first runs alone and the concurrent outputs are
compared token by token (common prefix length).
Usage: conc_bench.py [--n 1,2,4,8] [--tokens 256] [--reference]
"""

import argparse
import json
import threading
import time
import urllib.request

import regex as re

URL = "http://127.0.0.1:18000"
TOPICS = [
    "用 Python 写一个线程安全的 LRU 缓存类，带单元测试，并解释时间复杂度。",
    "Explain how a B+ tree handles insertion and node splits, with an example.",
    "写一篇 600 字左右的短文，介绍长江三峡的地理、历史和生态保护。",
    "Write SQL and pandas code computing monthly revenue per user from orders.",
    "解释 TCP 三次握手和四次挥手，以及 TIME_WAIT 状态存在的原因。",
    "Describe the CAP theorem and give one real system for each trade-off.",
    "用 Rust 实现一个简单的命令行 todo 应用，支持增删改查和持久化。",
    "Summarize the causes and consequences of the 2008 financial crisis.",
    "介绍 Transformer 的注意力机制，并推导缩放点积注意力的公式。",
    "Write a bash script that rotates log files older than seven days.",
    "比较快速排序、归并排序和堆排序的稳定性、复杂度和适用场景。",
    "Explain how garbage collection works in Go and in Java.",
    "写一个 React 组件实现带搜索和分页的表格，并说明状态管理思路。",
    "What are the main differences between HTTP/1.1, HTTP/2 and HTTP/3?",
    "介绍光合作用的光反应和暗反应过程。",
    "Design a rate limiter for an API gateway; compare token and leaky bucket.",
]


def metrics():
    text = urllib.request.urlopen(URL + "/metrics", timeout=30).read().decode()
    out = {}
    for name in (
        "spec_decode_num_drafts_total",
        "spec_decode_num_accepted_tokens_total",
        "generation_tokens_total",
    ):
        found = re.findall(rf"^vllm:{name}{{[^}}]*}} ([0-9.e+]+)", text, re.M)
        out[name] = sum(float(x) for x in found)
    return out


def stream(prompt, tokens, record):
    body = dict(
        model="deepseek-v4.1-flash",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=tokens,
        temperature=0,
        stream=True,
        stream_options={"include_usage": True},
        chat_template_kwargs={"thinking": False},
    )
    request = urllib.request.Request(
        URL + "/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    stamps, text, usage = [], [], None
    with urllib.request.urlopen(request, timeout=1800) as response:
        for line in response:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                piece = choice.get("delta", {}).get("content")
                if piece:
                    stamps.append(time.perf_counter())
                    text.append(piece)
    record.update(start=start, stamps=stamps, text="".join(text), usage=usage)


def run(n, tokens):
    prompts = [
        TOPICS[i % len(TOPICS)] + ("" if i < len(TOPICS) else f" ({i})")
        for i in range(n)
    ]
    records = [dict() for _ in range(n)]
    before = metrics()
    threads = [
        threading.Thread(target=stream, args=(p, tokens, r))
        for p, r in zip(prompts, records)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    after = metrics()
    rounds = (
        after["spec_decode_num_drafts_total"] - before["spec_decode_num_drafts_total"]
    )
    generated = after["generation_tokens_total"] - before["generation_tokens_total"]
    accepted = (
        after["spec_decode_num_accepted_tokens_total"]
        - before["spec_decode_num_accepted_tokens_total"]
    )
    # A streamed chunk is one engine step of that request (its accepted
    # drafts + bonus token arrive together), so chunk gaps are rounds.
    live = [r for r in records if len(r["stamps"]) > 1]
    ms_round = [
        1000 * (r["stamps"][-1] - r["stamps"][0]) / (len(r["stamps"]) - 1) for r in live
    ]
    tokens_of = [
        (r["usage"] or {}).get("completion_tokens", len(r["stamps"])) for r in live
    ]
    per_req = [t / (r["stamps"][-1] - r["start"]) for t, r in zip(tokens_of, live)]
    ttft = [r["stamps"][0] - r["start"] for r in records if r["stamps"]]
    span = max(r["stamps"][-1] for r in live) - min(r["stamps"][0] for r in live)
    agg = sum(tokens_of) / span if span > 0 else float("nan")
    # Accepted drafts + the bonus token per verification round.
    tok_round = 1 + accepted / rounds if rounds else float("nan")
    del generated
    wall = max(r["stamps"][-1] for r in records) - min(r["start"] for r in records)
    print(
        f"N={n:3d} ms/round {min(ms_round):6.1f}..{max(ms_round):6.1f} "
        f"(mean {sum(ms_round) / len(ms_round):6.1f}) | per-request "
        f"{sum(per_req) / len(per_req):6.1f} tok/s | aggregate {agg:7.1f} tok/s | "
        f"tok/round {tok_round:4.2f} | TTFT {min(ttft):5.2f}..{max(ttft):5.2f} s | "
        f"wall {wall:5.1f} s",
        flush=True,
    )
    return prompts, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", default="1,2,4,8")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--reference", action="store_true")
    args = parser.parse_args()
    sizes = [int(x) for x in args.n.split(",")]
    reference = {}
    if args.reference:
        for prompt in TOPICS[: max(sizes)]:
            record = {}
            stream(prompt, args.tokens, record)
            reference[prompt] = record["text"]
    for n in sizes:
        prompts, records = run(n, args.tokens)
        if reference:
            same = []
            for p, r in zip(prompts, records):
                want = reference.get(p)
                if want is None:
                    continue
                common = 0
                for a, b in zip(want, r["text"]):
                    if a != b:
                        break
                    common += 1
                same.append(common / max(1, len(want)))
            exact = sum(1 for x in same if x == 1.0)
            print(
                f"      vs single: identical {exact}/{len(same)}, mean common "
                f"prefix {100 * sum(same) / max(1, len(same)):5.1f}% of text",
                flush=True,
            )


if __name__ == "__main__":
    main()
