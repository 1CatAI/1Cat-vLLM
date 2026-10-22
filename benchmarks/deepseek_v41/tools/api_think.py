# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import sys
import time
import urllib.request

base, key = sys.argv[1], sys.argv[2]


def post(body):
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json", "Authorization": "Bearer " + key},
    )
    t = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    return d, time.time() - t


M = "deepseek-v4.1-flash"
d, dt = post(
    {
        "model": M,
        "messages": [
            {
                "role": "user",
                "content": "一个笼子里有鸡和兔共35只，脚共94只，鸡兔各几只？",
            }
        ],
        "max_tokens": 2048,
    }
)
m = d["choices"][0]["message"]
print("[thinking default] %.1fs usage=%s" % (dt, d["usage"]))
print(
    "  reasoning_content:",
    (m.get("reasoning_content") or m.get("reasoning") or "")[:200].replace("\n", " "),
)
print("  content:", (m.get("content") or "")[:200].replace("\n", " "))
d, dt = post(
    {
        "model": M,
        "messages": [
            {
                "role": "user",
                "content": "一个笼子里有鸡和兔共35只，脚共94只，鸡兔各几只？",
            }
        ],
        "max_tokens": 512,
        "chat_template_kwargs": {"thinking": False},
    }
)
m = d["choices"][0]["message"]
print(
    "[thinking off] %.1fs usage=%s reasoning=%r"
    % (dt, d["usage"], (m.get("reasoning_content") or "")[:40])
)
print("  content:", (m.get("content") or "")[:160].replace("\n", " "))
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询城市天气",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]
d, dt = post(
    {
        "model": M,
        "messages": [{"role": "user", "content": "上海明天适合跑步吗？先查一下天气。"}],
        "tools": tools,
        "max_tokens": 1024,
    }
)
m = d["choices"][0]["message"]
print(
    "[thinking + tools] %.1fs finish=%s tool_calls=%s"
    % (
        dt,
        d["choices"][0]["finish_reason"],
        json.dumps(m.get("tool_calls"), ensure_ascii=False)[:200],
    )
)
print("  reasoning:", (m.get("reasoning_content") or "")[:120].replace("\n", " "))
