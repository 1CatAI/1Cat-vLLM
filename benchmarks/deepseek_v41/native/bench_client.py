# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-request streaming benchmark against a local V4.1 server.

Reports TTFT, decode tok/s, and from /metrics the speculative rounds, so
ms/round = decode wall / rounds and tokens/round = emitted / rounds.
Usage: bench_client.py [--url URL] [--tokens N] [--temperature T] [--prompts FILE|default]
"""

import argparse
import json
import time
import urllib.request

import regex as re

PROMPTS = [
    "用 Python 写一个线程安全的 LRU 缓存类，带 get/put/容量淘汰和单元测试，并解释时间复杂度。",
    "Explain how a B+ tree handles insertion and node splits, with a small worked example.",
    "写一篇 600 字左右的短文，介绍长江三峡的地理、历史和生态保护。",
    "Given a JSON array of orders with fields id, user, amount, date, write SQL and pandas code to compute monthly revenue per user and the top 3 users per month.",
]


def metrics(url):
    text = urllib.request.urlopen(url + "/metrics", timeout=30).read().decode()
    out = {}
    for name in (
        "vllm:spec_decode_num_drafts_total",
        "vllm:spec_decode_num_accepted_tokens_total",
        "vllm:spec_decode_num_draft_tokens_total",
        "vllm:generation_tokens_total",
    ):
        m = re.findall(rf"^{re.escape(name)}{{[^}}]*}} ([0-9.e+]+)", text, re.M)
        out[name.split(":")[1]] = sum(float(x) for x in m)
    return out


def run(url, prompt, tokens, temperature):
    body = dict(
        model="deepseek-v4.1-flash",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=tokens,
        temperature=temperature,
        top_p=1.0,
        stream=True,
        chat_template_kwargs={"thinking": False},
        ignore_eos=False,
    )
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    before = metrics(url)
    start = time.time()
    first = None
    chunks = 0
    text = []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            delta = json.loads(line[6:])["choices"][0].get("delta", {})
            piece = delta.get("content") or delta.get("reasoning_content") or ""
            if piece:
                if first is None:
                    first = time.time()
                chunks += 1
                text.append(piece)
    end = time.time()
    after = metrics(url)
    d = {k: after[k] - before[k] for k in after}
    gen = d["generation_tokens_total"]
    rounds = d["spec_decode_num_drafts_total"]
    decode = end - (first or end)
    return dict(
        ttft=round((first or end) - start, 3),
        gen=int(gen),
        decode_s=round(decode, 3),
        tok_s=round((gen - 1) / decode, 2) if decode > 0 else None,
        rounds=int(rounds),
        ms_round=round(decode * 1000 / max(rounds, 1), 2),
        tok_round=round(gen / max(rounds, 1), 3),
        accept=round(
            d["spec_decode_num_accepted_tokens_total"]
            / max(d["spec_decode_num_draft_tokens_total"], 1),
            3,
        ),
        text="".join(text)[:160].replace("\n", " "),
    )


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--url", default="http://127.0.0.1:18000")
    a.add_argument("--tokens", type=int, default=512)
    a.add_argument("--temperature", type=float, default=0.0)
    a.add_argument("--repeat", type=int, default=1)
    args = a.parse_args()
    rows = []
    for _ in range(args.repeat):
        for p in PROMPTS:
            r = run(args.url, p, args.tokens, args.temperature)
            rows.append(r)
            print(json.dumps(r, ensure_ascii=False), flush=True)
    keys = ("ttft", "tok_s", "ms_round", "tok_round", "accept")
    print(
        "MEAN",
        json.dumps(
            {
                k: round(sum(r[k] for r in rows if r[k] is not None) / len(rows), 3)
                for k in keys
            }
        ),
    )
