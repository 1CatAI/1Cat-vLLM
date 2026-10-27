# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import sys
import urllib.request

base = sys.argv[1]
key = open("/home/ymzx/ds41-dev/state/api_key.txt").read().strip()


def post(path, body, auth=True):
    req = urllib.request.Request(
        base + path,
        json.dumps(body).encode(),
        {
            "Content-Type": "application/json",
            **({"Authorization": "Bearer " + key} if auth else {}),
        },
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


d = post(
    "/v1/chat/completions",
    {
        "model": "deepseek-v4.1-flash",
        "messages": [{"role": "user", "content": "用一句话介绍你自己"}],
        "max_tokens": 64,
    },
)
print("chat:", d["choices"][0]["message"]["content"], d["usage"])
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
d = post(
    "/v1/chat/completions",
    {
        "model": "deepseek-v4.1-flash",
        "messages": [{"role": "user", "content": "北京今天天气怎么样？"}],
        "tools": tools,
        "tool_choice": "auto",
        "max_tokens": 128,
    },
)
m = d["choices"][0]["message"]
print(
    "tool_calls:",
    json.dumps(m.get("tool_calls"), ensure_ascii=False),
    "finish:",
    d["choices"][0]["finish_reason"],
)
if m.get("tool_calls"):
    call = m["tool_calls"][0]
    msgs = [
        {"role": "user", "content": "北京今天天气怎么样？"},
        {
            "role": "assistant",
            "content": m.get("content") or "",
            "tool_calls": m["tool_calls"],
        },
        {
            "role": "tool",
            "tool_call_id": call["id"],
            "content": json.dumps(
                {"city": "北京", "weather": "晴", "temp_c": 18}, ensure_ascii=False
            ),
        },
    ]
    d = post(
        "/v1/chat/completions",
        {
            "model": "deepseek-v4.1-flash",
            "messages": msgs,
            "tools": tools,
            "max_tokens": 96,
        },
    )
    print("after tool:", d["choices"][0]["message"]["content"])
try:
    post(
        "/v1/chat/completions",
        {
            "model": "deepseek-v4.1-flash",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
        },
        auth=False,
    )
    print("no-key: ACCEPTED (bad)")
except urllib.error.HTTPError as e:
    print("no-key rejected:", e.code)
