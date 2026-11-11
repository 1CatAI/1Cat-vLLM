# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up, then one nsys capture: a 200-token decode, a 1.5 s pause, one ~15K prefill."""

import json
import random
import time
import urllib.request

U = "http://127.0.0.1:18000"


def post(path, body):
    r = urllib.request.Request(
        U + path,
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(r, timeout=1800).read()


msg = [
    {
        "role": "user",
        "content": "用 Python 写一个线程安全的 LRU 缓存类，带 get/put/容量淘汰和单元测试，并解释时间复杂度。",
    }
]
body = dict(
    model="deepseek-v4.1-flash",
    messages=msg,
    max_tokens=200,
    temperature=0,
    chat_template_kwargs={"thinking": False},
)
random.seed(1)
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
text = " ".join(random.choice(words) for _ in range(14500))
post("/v1/chat/completions", body)
post(
    "/v1/completions",
    dict(model="deepseek-v4.1-flash", prompt="hello " * 50, max_tokens=4),
)
post("/start_profile", {})
t = time.time()
r = json.loads(post("/v1/chat/completions", body))
print("decode", round(time.time() - t, 2), "s", r["usage"], flush=True)
time.sleep(1.5)
t = time.time()
r = json.loads(
    post(
        "/v1/completions",
        dict(model="deepseek-v4.1-flash", prompt=text, max_tokens=1, temperature=0),
    )
)
print(
    "prefill",
    r["usage"]["prompt_tokens"],
    "tokens",
    round(time.time() - t, 2),
    "s",
    flush=True,
)
post("/stop_profile", {})
