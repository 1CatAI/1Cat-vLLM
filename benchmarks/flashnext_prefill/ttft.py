# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import random
import sys
import time
import urllib.request

n = int(sys.argv[1])
words = [
    "alpha",
    "river",
    "stone",
    "cloud",
    "engine",
    "paper",
    "signal",
    "forest",
    "copper",
    "lantern",
    "orbit",
    "meadow",
]
r = random.Random(n * 7 + int(time.time()))
text = " ".join(r.choice(words) for _ in range(n))
body = dict(
    model="flash-next",
    max_tokens=1,
    temperature=0,
    prompt=f"[{r.randint(0, 10**9)}] " + text,
)
t = time.time()
req = urllib.request.Request(
    "http://127.0.0.1:8001/v1/completions",
    json.dumps(body).encode(),
    {"Content-Type": "application/json"},
)
u = json.loads(urllib.request.urlopen(req, timeout=1800).read())["usage"]
dt = time.time() - t
tokens = u["prompt_tokens"]
print(
    f"prompt {tokens:6d} tokens  ttft {dt:7.2f} s  {tokens / dt:7.0f} tok/s", flush=True
)
