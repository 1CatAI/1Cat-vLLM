# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture complete real prefill routes through the ordinary serving API.

The exclusively owned server must enable_return_routed_experts. These requests
are diagnostic data collection, not performance evidence. No worker hooks or
private kernel/provider factories are used.
"""

import argparse
import hashlib
import io
import json
import random
from pathlib import Path

import numpy as np
import pybase64 as base64
from measure_prefill import WORDS, request
from quality import EXPECTED
from quality import prompt as quality_prompt


def save_routes(result, label, directory):
    choice = result["choices"][0]
    encoded = choice.pop("routed_experts", None)
    if encoded is None:
        raise RuntimeError("The normal serving response has no routed_experts")
    routes = np.load(io.BytesIO(base64.b64decode(encoded)), allow_pickle=False)
    expected_rows = (
        result["usage"]["prompt_tokens"] + result["usage"]["completion_tokens"] - 1
    )
    if routes.shape != (expected_rows, 48, 10):
        raise RuntimeError(
            f"Incomplete route tensor {routes.shape}, rows={expected_rows}"
        )
    if routes.min() < 0 or routes.max() >= 512:
        raise RuntimeError("Unexpected logical expert id")
    path = directory / (label + ".npz")
    np.savez_compressed(path, routed_experts=routes)
    return {
        "file": str(path.resolve()),
        "shape": list(routes.shape),
        "dtype": str(routes.dtype),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    report = {"diagnostic_only": True, "cases": []}
    for index in (0, 4):
        rng = random.Random(9683 + index)
        text = f"[{rng.randrange(10**9)}] " + " ".join(
            rng.choice(WORDS.split()) for _ in range(32000)
        )
        body = {
            "model": "flash-next",
            "prompt": text,
            "temperature": 0,
            "max_tokens": 1,
            "stream": False,
        }
        with request(args.base_url, "/reset_prefix_cache", {}) as response:
            response.read()
        with request(args.base_url, "/v1/completions", body) as response:
            result = json.load(response)
        row = {
            "label": f"speed-{index}",
            "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "usage": result["usage"],
            **save_routes(result, f"speed-{index}", args.directory),
        }
        report["cases"].append(row)
        print(json.dumps(row), flush=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def count(repeats):
        messages = [{"role": "user", "content": quality_prompt(repeats)}]
        body = {
            "model": "flash-next",
            "messages": messages,
            "chat_template_kwargs": {"enable_thinking": True},
        }
        with request(args.base_url, "/tokenize", body) as response:
            return json.load(response)["count"], messages

    fixed, _ = count(0)
    ten, _ = count(10)
    repeats = max(0, int((32768 - fixed) * 10 / (ten - fixed)))
    tokens, messages = count(repeats)
    while tokens > 32768:
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
    with request(args.base_url, "/v1/chat/completions", body) as response:
        result = json.load(response)
    route_info = save_routes(result, "quality32", args.directory)
    message = result["choices"][0]["message"]
    content = message.get("content") or ""
    start, end = content.find("{"), content.rfind("}")
    answer = json.loads(content[start : end + 1]) if start >= 0 else None
    row = {
        "label": "quality32",
        "tokenize_count": tokens,
        "prompt_sha256": hashlib.sha256(messages[0]["content"].encode()).hexdigest(),
        "usage": result["usage"],
        "choice": result["choices"][0],
        "passed": answer == EXPECTED
        and result["choices"][0]["finish_reason"] == "stop",
        **route_info,
    }
    report["cases"].append(row)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in row.items() if k != "choice"}), flush=True)
    if not row["passed"]:
        raise SystemExit("Diagnostic capture failed the existing task check")


if __name__ == "__main__":
    main()
