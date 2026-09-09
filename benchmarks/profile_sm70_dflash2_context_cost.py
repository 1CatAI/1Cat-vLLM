# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-prefix DFlash2 cost windows, separate from natural-output scoring."""

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

from prometheus_client.parser import text_string_to_metric_families


def build_prompt(corpus: dict, length: int) -> list[int]:
    remaining = length - len(corpus["prefix"]) - len(corpus["suffix"])
    if remaining < 0:
        raise ValueError("Prefix length is smaller than the task template")
    repeats, tail = divmod(remaining, len(corpus["filler"]))
    return (
        corpus["prefix"]
        + corpus["filler"] * repeats
        + corpus["filler"][:tail]
        + corpus["suffix"]
    )


def metrics(base: str) -> dict:
    with urllib.request.urlopen(base + "/metrics", timeout=30) as response:
        families = text_string_to_metric_families(response.read().decode())
        result = {}
        for family in families:
            for sample in family.samples:
                if "spec_decode" in sample.name or sample.name.endswith(
                    ("_time_seconds_sum", "_latency_seconds_sum")
                ):
                    key = sample.name
                    if "position" in sample.labels:
                        key += ":position=" + sample.labels["position"]
                    result[key] = result.get(key, 0.0) + sample.value
        return result


def request(base: str, prompt: list[int], output_limit: int, seed: int) -> dict:
    if len(prompt) + output_limit > 262144:
        raise ValueError("The complete request exceeds the 256K capacity")
    payload = {
        "model": "quasar-baseline",
        "prompt": prompt,
        "max_tokens": output_limit,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
    }
    before = metrics(base)
    start = time.perf_counter()
    chunks, token_ids, text = [], [], []
    usage, finish = {}, None
    req = urllib.request.Request(
        base + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=3600) as response:
        for raw in response:
            now = time.perf_counter()
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            if line[6:] == "[DONE]":
                break
            event = json.loads(line[6:])
            if event.get("error"):
                raise RuntimeError(event["error"])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                ids = choice.get("token_ids") or []
                content = choice.get("text") or ""
                if ids or content:
                    chunks.append({"at_s": now - start, "tokens": len(ids)})
                token_ids.extend(ids)
                text.append(content)
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    wall = time.perf_counter() - start
    after = metrics(base)
    delta = {key: after.get(key, 0) - before.get(key, 0) for key in before | after}
    assert usage["prompt_tokens"] == len(prompt)
    assert usage["completion_tokens"] == len(token_ids)
    assert chunks and token_ids
    rounds = delta.get("vllm:spec_decode_num_drafts_total", 0)
    accepted = delta.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    proposed = delta.get("vllm:spec_decode_num_draft_tokens_total", 0)
    decode = delta["vllm:request_decode_time_seconds_sum"]
    return {
        "prompt_tokens": len(prompt),
        "prompt_sha256": hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
        "output_limit": output_limit,
        "output_tokens": len(token_ids),
        "token_ids": token_ids,
        "text": "".join(text),
        "finish_reason": finish,
        "wall_s": wall,
        "ttft_s": chunks[0]["at_s"],
        "engine_decode_s": decode,
        "complete_round_ms": decode * 1000 / rounds if rounds else None,
        "pure_decode_tps": (len(token_ids) - 1) / decode if decode else None,
        "rounds": rounds,
        "accepted_drafts_per_round": accepted / rounds if rounds else None,
        "emitted_tokens_per_round": len(token_ids) / rounds if rounds else None,
        "accepted_over_proposed": accepted / proposed if proposed else None,
        "chunks": chunks,
        "stream_intervals_match_round_count": len(chunks) - 1 == rounds,
        "metric_deltas": delta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base", default="http://127.0.0.1:18215")
    parser.add_argument("--lengths", type=int, nargs="+", required=True)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--trace-dir", type=Path)
    args = parser.parse_args()
    assert not args.output.exists(), args.output
    corpus_bytes = args.corpus.read_bytes()
    corpus = json.loads(corpus_bytes)
    for _ in range(900):
        try:
            with urllib.request.urlopen(args.base + "/health", timeout=2) as response:
                if response.status == 200:
                    break
        except (OSError, TimeoutError):
            pass
        time.sleep(2)
    else:
        raise TimeoutError("Context cost server did not become healthy")
    report = {
        "scope": "Bounded latency diagnostic, not natural-output quality scoring",
        "context_capacity": 262144,
        "profiler": args.trace_dir is not None,
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "seed": 0},
        "corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
        "cases": [],
        "complete": False,
    }

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    for length in args.lengths:
        prompt = build_prompt(corpus, length)
        for repeat in range(-1, args.repeats):
            if args.trace_dir and repeat == 0:
                (args.trace_dir / "arm").touch()
            row = request(args.base, prompt, args.output_tokens, 0)
            row.update(warmup=repeat < 0, repeat=repeat)
            report["cases"].append(row)
            save()
            print(
                json.dumps(
                    {
                        key: value
                        for key, value in row.items()
                        if key not in ("text", "token_ids", "chunks", "metric_deltas")
                    }
                ),
                flush=True,
            )
    if args.trace_dir:
        for rank in range(4):
            path = args.trace_dir / f"rank{rank}-rounds.json"
            observations = json.loads(path.read_text())
            assert [row["step"] for row in observations] == list(range(21))
            assert all(row["scheduled"] == 8 for row in observations)
    report["complete"] = True
    save()


if __name__ == "__main__":
    main()
