# SPDX-License-Identifier: Apache-2.0

"""Issue one reproducible OpenAI chat request for DSpark validation."""

import argparse
import json
import time
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18082")
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt")
    prompt.add_argument("--prompt-file", type=Path)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt = (
        args.prompt_file.read_text() if args.prompt_file is not None else args.prompt
    )
    payload = {
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 1.0,
        "top_p": 1.0,
        "seed": args.seed,
    }
    request = urllib.request.Request(
        f"{args.url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=1800) as response:
        result = json.load(response)
    wall_s = time.perf_counter() - start
    artifact = {"request": payload, "wall_s": wall_s, "response": result}
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2))

    completion_tokens = result["usage"]["completion_tokens"]
    choice = result["choices"][0]
    print(
        json.dumps(
            {
                "wall_s": wall_s,
                "prompt_tokens": result["usage"]["prompt_tokens"],
                "completion_tokens": completion_tokens,
                "end_to_end_output_tokens_per_s": completion_tokens / wall_s,
                "finish_reason": choice["finish_reason"],
                "content": choice["message"].get("content") or "",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
