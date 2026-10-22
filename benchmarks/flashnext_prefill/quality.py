# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired retrieval/arithmetic and EOS checks at specified prompt-token budgets.

Uses production sampling and thinking. Reset only an exclusively owned server.
Record the same prompt hash, seed and actual token length for each artifact.
"""

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import regex as re
from measure_prefill import request

FILLER = (
    "Computing records describe mechanical calculators, vacuum tubes, transistors, "
    "integrated circuits, storage systems and communication networks. "
    "These background records do not contain any vault or captain information.\n"
)
EXPECTED = {"vault": "48213", "captain": "Orlanda Vexley", "packages": 45}


def prompt(repeats):
    return (
        "Read this archive and answer the question at the end.\n"
        + FILLER * (repeats // 8)
        + "The vault code is 48213.\n"
        + FILLER * (repeats // 2)
        + "The ship captain is Orlanda Vexley.\n"
        + FILLER * (repeats - repeats // 8 - repeats // 2)
        + "A warehouse has 17 boxes with 24 items each. After selling 93 items, "
        "all remaining items are packed into groups of 7. How many full packages "
        "are made? Return only one JSON object containing the vault code as a "
        'string, captain name, and number of packages: {"vault": ..., '
        '"captain": ..., "packages": ...}. Think briefly.\n'
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--budgets", type=int, nargs="+", default=[32768])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.budgets) < 512:
        parser.error("prompt budgets must be at least 512 tokens")
    results = []
    for budget in args.budgets:

        def count(repeats):
            messages = [{"role": "user", "content": prompt(repeats)}]
            with request(
                args.base_url,
                "/tokenize",
                {
                    "model": "flash-next",
                    "messages": messages,
                    "chat_template_kwargs": {"enable_thinking": True},
                },
            ) as response:
                return json.load(response)["count"], messages

        fixed, _ = count(0)
        ten, _ = count(10)
        repeats = max(0, int((budget - fixed) * 10 / (ten - fixed)))
        tokens, messages = count(repeats)
        while tokens > budget:
            repeats -= 1
            tokens, messages = count(repeats)
        body = {
            "model": "flash-next",
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "seed": 9683,
            "chat_template_kwargs": {"enable_thinking": True},
        }
        with request(args.base_url, "/reset_prefix_cache", {}) as response:
            response.read()
        start = time.perf_counter()
        with request(args.base_url, "/v1/chat/completions", body) as response:
            result = json.load(response)
        choice = result["choices"][0]
        message = choice["message"]
        content = message.get("content") or ""
        match = re.search(r"\{[^{}]*\}", content)
        try:
            answer = json.loads(match.group()) if match else None
        except json.JSONDecodeError:
            answer = None
        text = " ".join(
            str(message.get(k) or "")
            for k in ("content", "reasoning", "reasoning_content")
        )
        words = text.split()
        ngrams = Counter(tuple(words[i : i + 10]) for i in range(len(words) - 9))
        max_repeat = max(ngrams.values(), default=0)
        row = {
            "budget_tokens": budget,
            "tokenize_count": tokens,
            "prompt_sha256": hashlib.sha256(
                messages[0]["content"].encode()
            ).hexdigest(),
            "seconds": time.perf_counter() - start,
            "usage": result["usage"],
            "choice": choice,
            "max_ten_word_repeat": max_repeat,
            "passed": answer == EXPECTED
            and choice["finish_reason"] == "stop"
            and max_repeat <= 5,
        }
        results.append(row)
        args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)
    raise SystemExit(0 if all(row["passed"] for row in results) else 1)


if __name__ == "__main__":
    main()
