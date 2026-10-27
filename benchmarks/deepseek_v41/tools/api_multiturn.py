# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-turn chat over the OpenAI API (streaming): TTFT and decode tok/s per turn."""

import json
import sys
import time
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18000/v1/chat/completions"
DOC = open("/home/ymzx/ds41-dev/src/README.md", encoding="utf-8").read()[:12000]


def chat(messages, max_tokens=200):
    body = json.dumps(
        dict(
            model="deepseek-v4.1-flash",
            messages=messages,
            max_tokens=max_tokens,
            temperature=0,
            stream=True,
            stream_options={"include_usage": True},
        )
    ).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    text = []
    usage = {}
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices", []):
                piece = c.get("delta", {}).get("content")
                if piece:
                    if first is None:
                        first = time.perf_counter() - t0
                    text.append(piece)
    total = time.perf_counter() - t0
    n = usage.get("completion_tokens", 0)
    return (
        "".join(text),
        first,
        (n - 1) / (total - first) if first and n > 1 else 0,
        usage,
    )


for name, opener in (
    ("short", "你好，请用三句话介绍一下长城。"),
    ("long", "请阅读下面的文档：\n\n" + DOC + "\n\n这份文档主要讲了什么？"),
):
    messages = [{"role": "user", "content": opener}]
    for turn, follow in enumerate(
        ("请再简要总结成一句话。", "再列出三个关键词。", None)
    ):
        text, ttft, tps, usage = chat(messages)
        print(
            "%s turn%d prompt=%s ttft=%.2fs decode=%.1f tok/s out=%r"
            % (name, turn + 1, usage.get("prompt_tokens"), ttft or -1, tps, text[:60]),
            flush=True,
        )
        if follow is None:
            break
        messages += [
            {"role": "assistant", "content": text},
            {"role": "user", "content": follow},
        ]
