# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Send one completion of N filler words (for nsys start/stop_profile windows).

Usage: python profile_request.py 32000
"""

import json
import random
import sys
import urllib.request

n = int(sys.argv[1])
w = " ".join(
    random.Random(n).choice(
        ["alpha", "river", "stone", "cloud", "engine", "paper", "signal", "forest"]
    )
    for _ in range(n)
)
b = dict(
    model="flash-next",
    max_tokens=1,
    temperature=0,
    prompt=f"[m{random.randint(0, 10**9)}] " + w,
)
r = urllib.request.urlopen(
    urllib.request.Request(
        "http://127.0.0.1:8001/v1/completions",
        json.dumps(b).encode(),
        {"Content-Type": "application/json"},
    ),
    timeout=900,
)
print(n, json.loads(r.read())["usage"]["prompt_tokens"])
