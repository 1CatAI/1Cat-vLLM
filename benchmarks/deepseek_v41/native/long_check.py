# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Greedy 64-token completion of a ~8K-token real-text prompt; prints TTFT and
the text, saves it to ~/v41/long_check_<label>.txt for cross-config diffs."""

import glob
import json
import os
import sys
import time
import urllib.request

U = "http://127.0.0.1:18000"
files = sorted(glob.glob(os.path.expanduser("~/v41/src/vllm/v1/core/sched/*.py")))
text = ""
for f in files:
    text += open(f).read()
    if len(text) > 30000:
        break
text = text[:30000]
msg = [
    {
        "role": "user",
        "content": "下面是一段 Python 源码：\n"
        + text
        + "\n\n请用中文概括这段代码中调度器的主要职责（三句话）。",
    }
]
body = dict(
    model="deepseek-v4.1-flash",
    messages=msg,
    max_tokens=64,
    temperature=0,
    chat_template_kwargs={"thinking": False},
)
r = urllib.request.Request(
    U + "/v1/chat/completions",
    json.dumps(body).encode(),
    {"Content-Type": "application/json"},
)
t = time.time()
out = json.loads(urllib.request.urlopen(r, timeout=600).read())
dt = time.time() - t
content = out["choices"][0]["message"]["content"]
print("prompt tokens", out["usage"]["prompt_tokens"], "time", round(dt, 2), "s")
print(content)
open(os.path.expanduser(f"~/v41/long_check_{sys.argv[1]}.txt"), "w").write(content)
