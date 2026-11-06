#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Long-prompt greedy A/B for VLLM_V41_PREFILL_SKIP_ROWS.

The 16-prompt greedy set is short (< 11K tokens), so no step there ever
skips. This sends cold (fresh cache_salt), greedy, thinking-off requests at
the given lengths on a needle haystack (``needle.Builder`` from
--mc-dir, not part of this repository) and records the text and
the top-2 logprobs of every generated token; ``compare`` applies the near-tie
gate between two arms.

    python3 -I prefill_skip_greedy.py run --url http://127.0.0.1:18000 \\
        --mc-dir MC_DIR --data-dir DIR \\
        --n 32768 150000 --out ARM.jsonl --tag ARM
    python3 -I prefill_skip_greedy.py compare OFF.jsonl ON.jsonl

Gate: per prompt byte-identical, or the first differing token is
the baseline's #2 with both top-2 gaps <= 0.10 nats. Exit 0 PASS, 1 otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

GAP = 0.10


def ask(url, model, prompt, max_tokens, salt):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
        "priority": 10,
        "cache_salt": salt,
        "logprobs": True,
        "top_logprobs": 2,
        "chat_template_kwargs": {"thinking": False},
    }
    request = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.monotonic()
    with urllib.request.urlopen(request, timeout=7200) as response:
        result = json.load(response)
    elapsed = time.monotonic() - start
    choice = result["choices"][0]
    steps = [
        {
            "token": item["token"],
            "top": [[t["token"], t["logprob"]] for t in item.get("top_logprobs", [])],
        }
        for item in (choice.get("logprobs") or {}).get("content") or []
    ]
    text = choice["message"].get("content") or ""
    return {
        "text": text,
        "sha": hashlib.sha256(text.encode()).hexdigest(),
        "steps": steps,
        "elapsed_s": round(elapsed, 2),
        "prompt_tokens": result["usage"]["prompt_tokens"],
    }


def run(args):
    sys.path.insert(0, str(args.mc_dir))
    from needle import Builder  # the haystack builder in --mc-dir

    builder = Builder(args.url, args.model, args.data_dir)
    for n in args.n:
        prompt, _ = builder.build(n, 6, args.seed + n)
        row = ask(
            args.url, args.model, prompt, args.max_tokens, f"psg-{n}-{time.time_ns()}"
        )
        row.update(tag=args.tag, n=n)
        with args.out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(
            json.dumps(
                {k: row[k] for k in ("tag", "n", "prompt_tokens", "sha", "elapsed_s")}
            ),
            flush=True,
        )


def gap(step):
    top = step["top"]
    return top[0][1] - top[1][1] if len(top) >= 2 else float("inf")


def verdict(base, other):
    """IDENTICAL, NEAR (the near-tie gate holds) or DIFF."""
    if base["text"] == other["text"]:
        return "IDENTICAL"
    for a, b in zip(base["steps"], other["steps"]):
        if a["token"] == b["token"]:
            continue
        second = a["top"][1][0] if len(a["top"]) >= 2 else None
        ok = b["token"] == second and gap(a) <= GAP and gap(b) <= GAP
        return "NEAR" if ok else "DIFF"
    return "DIFF"  # same tokens, different text (length or decoding)


def compare(args):
    def load(path):
        return {
            row["n"]: row
            for row in map(json.loads, Path(path).read_text().splitlines())
        }

    base, other = load(args.base), load(args.other)
    bad = 0
    for n in sorted(base):
        if n not in other:
            print(f"n={n}: missing in {args.other}")
            bad += 1
            continue
        result = verdict(base[n], other[n])
        bad += result == "DIFF"
        print(f"n={n}: {result} sha {base[n]['sha'][:12]} / {other[n]['sha'][:12]}")
    print("PREFILL-SKIP GREEDY " + ("PASS" if not bad else f"FAIL {bad}"))
    return 1 if bad else 0


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", required=True)
    r.add_argument("--model", default="deepseek-v4.1-flash")
    r.add_argument("--mc-dir", type=Path, required=True)
    r.add_argument("--data-dir", type=Path, required=True)
    r.add_argument("--n", type=int, nargs="+", required=True)
    r.add_argument("--max-tokens", type=int, default=128)
    r.add_argument("--seed", type=int, default=2210)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--tag", default="")
    c = sub.add_parser("compare")
    c.add_argument("base")
    c.add_argument("other")
    args = parser.parse_args()
    if args.command == "run":
        run(args)
        return 0
    return compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
