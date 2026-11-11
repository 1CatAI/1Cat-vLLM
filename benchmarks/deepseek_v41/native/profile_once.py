# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up, then capture one decode request under nsys (/start_profile)."""

import json
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
    return urllib.request.urlopen(r, timeout=900).read()


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
post("/v1/chat/completions", body)
post("/start_profile", {})
t = time.time()
r = json.loads(post("/v1/chat/completions", body))
print("profiled request", round(time.time() - t, 2), "s", r["usage"])
post("/stop_profile", {})
