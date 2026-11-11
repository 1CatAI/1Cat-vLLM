# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill throughput: prompts of ~N tokens, max_tokens=1, TTFT -> prompt tok/s."""

import json
import random
import time
import urllib.request

U = "http://127.0.0.1:18000"
random.seed(0)
words = [
    "system",
    "data",
    "model",
    "value",
    "process",
    "network",
    "memory",
    "layer",
    "token",
    "kernel",
    "graph",
    "stream",
    "the",
    "of",
    "and",
    "to",
    "in",
    "is",
    "for",
    "on",
    "with",
    "as",
    "by",
    "at",
    "from",
    "that",
    "this",
    "which",
    "be",
    "are",
    "was",
]


def post(body):
    r = urllib.request.Request(
        U + "/v1/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(r, timeout=1800).read())


for n in (512, 2048, 4096, 8192, 15000):
    text = " ".join(random.choice(words) for _ in range(int(n * 0.95)))
    for rep in range(2):
        body = dict(
            model="deepseek-v4.1-flash",
            prompt=text + f" [{rep}]",
            max_tokens=1,
            temperature=0,
        )
        t = time.time()
        r = post(body)
        dt = time.time() - t
        pt = r["usage"]["prompt_tokens"]
    print(
        f"prompt {pt:6d} tokens: TTFT {dt:6.2f} s  prefill {pt / dt:7.0f} tok/s",
        flush=True,
    )
