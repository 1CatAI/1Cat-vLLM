# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Long-context retrieval check (~30K tokens): facts planted early and in
the middle of filler; greedy answers must contain them."""

import json
import sys
import time
import urllib.request

URL = "http://127.0.0.1:%s/v1/chat/completions" % (
    sys.argv[1] if len(sys.argv) > 1 else "8001"
)
FILLER = (
    "The history of computing spans mechanical calculators, vacuum tubes, "
    "transistors and integrated circuits. Each generation traded generality, "
    "cost and speed in different ways, and the software that ran on them "
    "evolved from hand-written machine code to high-level languages. "
)


def ask(prompt, question):
    body = {
        "model": "flash-next",
        "messages": [{"role": "user", "content": prompt + "\n\n" + question}],
        "max_tokens": 40,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        URL, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    t = time.time()
    out = json.load(urllib.request.urlopen(req, timeout=300))
    return (
        out["choices"][0]["message"].get("content") or "",
        out["usage"]["prompt_tokens"],
        time.time() - t,
    )


block = FILLER * 40  # ~2K tokens
facts = {
    "early": (
        "The vault code is 48213.",
        "What is the vault code? Answer with the number only.",
        "48213",
    ),
    "middle": (
        "The ship's captain is named Orlanda Vexley.",
        "What is the ship captain's name? Answer with the name only.",
        "Orlanda",
    ),
}
doc = block + facts["early"][0] + " " + block * 7 + facts["middle"][0] + " " + block * 7
ok = 0
for name, (_, q, want) in facts.items():
    ans, n, dt = ask(doc, q)
    hit = want in ans
    ok += hit
    print(f"{name:7s} prompt={n} {dt:5.1f}s hit={hit} answer={ans.strip()[:80]!r}")
print(f"needle {ok}/{len(facts)}")
