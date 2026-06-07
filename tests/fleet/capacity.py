#!/usr/bin/env python3
"""Parallel-request capacity probe.

Ramps concurrency against one or more OpenAI-compatible vLLM endpoints using a
long prompt (the same corpus as perf_t2/t3) and reports the MAX concurrency that
still sustains a per-request decode rate >= --target tok/s.

Per-request rate is the streamed DECODE rate (excludes prefill / TTFT):

    rate = (n_tokens - 1) / (t_last_token - t_first_token)

so it reflects the generation speed a user actually perceives under load, not the
one-shot prefill cost. Aggregate throughput across all in-flight requests is also
reported. With multiple --ports the load is round-robined across replicas, so the
reported concurrency is TOTAL across the fleet.

Examples:
    python -m tests.fleet.capacity                       # single replica :8000
    python -m tests.fleet.capacity --ports 8000,8001     # 2-replica fleet
    python -m tests.fleet.capacity --repeat 4 --target 50 --max-tokens 256
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def build_prompt(repeat: int) -> str:
    corpus = pathlib.Path(__file__).parent / "corpus" / "long_prompt.txt"
    base = (
        corpus.read_text()
        if corpus.exists()
        else "The Pacific Ocean is the largest and deepest ocean.\n" * 30
    )
    return (base + "\n\n") * repeat + (
        "\n\nGiven the above text, summarize the main themes in detail covering "
        "geography, biology, physics, history, and technology as separate "
        "sections with examples from the text."
    )


def stream_one(base_url: str, model: str, prompt: str, max_tokens: int) -> dict:
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,
        "ignore_eos": True,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        base_url + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t_req = time.perf_counter()
    t_first = t_last = None
    n = 0
    head = ""
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except json.JSONDecodeError:
                continue
            ch = d.get("choices") or []
            if ch and ch[0].get("text"):
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                t_last = now
                n += 1
                if len(head) < 80:
                    head += ch[0]["text"]
    decode_rate = (
        (n - 1) / (t_last - t_first)
        if (t_first and t_last and t_last > t_first and n > 1)
        else 0.0
    )
    ttft = (t_first - t_req) if t_first else 0.0
    return {"n": n, "rate": decode_rate, "ttft": ttft, "head": head}


def run_level(urls, model, prompt, max_tokens, conc) -> dict:
    def task(i):
        return stream_one(urls[i % len(urls)], model, prompt, max_tokens)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        res = list(ex.map(task, range(conc)))
    wall = time.perf_counter() - t0
    rates = sorted(r["rate"] for r in res)
    ttfts = sorted(r["ttft"] for r in res)
    return {
        "conc": conc,
        "median_rate": statistics.median(rates),
        "min_rate": rates[0],
        "p50_ttft": statistics.median(ttfts),
        "agg": sum(r["n"] for r in res) / wall,
        "sample": res[0]["head"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ports", default="8000", help="comma-separated replica ports")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", default="Qwen3.6-35B-A3B")
    ap.add_argument("--repeat", type=int, default=4, help="corpus repeats -> prompt length")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--target", type=float, default=50.0, help="per-request tok/s SLA")
    ap.add_argument("--ramp", default="1,2,4,8,12,16,24,32,48,64")
    args = ap.parse_args()

    urls = [f"http://{args.host}:{p.strip()}" for p in args.ports.split(",")]
    prompt = build_prompt(args.repeat)
    print(
        f"endpoints={urls}  target>={args.target} tok/s/req  decode={args.max_tokens} "
        f"tok  prompt~{len(prompt.split())} words ({len(urls)} replica(s))"
    )
    print("-" * 72)
    best = None
    for conc in [int(x) for x in args.ramp.split(",")]:
        try:
            r = run_level(urls, args.model, prompt, args.max_tokens, conc)
        except Exception as e:  # noqa: BLE001
            print(f"conc={conc:3d}  ERROR {e!r}")
            break
        ok = r["median_rate"] >= args.target
        print(
            f"conc={conc:3d}  median={r['median_rate']:6.1f} tok/s/req  "
            f"min={r['min_rate']:6.1f}  p50_ttft={r['p50_ttft']:5.2f}s  "
            f"agg={r['agg']:7.1f} tok/s  [{'OK' if ok else 'BELOW TARGET'}]"
        )
        if ok:
            best = r
        else:
            break
    print("-" * 72)
    if best is None:
        print(f"Even conc=1 is below {args.target} tok/s/req.")
    else:
        print(
            f"MAX sustained concurrency >= {args.target} tok/s/req: {best['conc']} "
            f"(aggregate {best['agg']:.0f} tok/s across {len(urls)} replica(s))"
        )
        print(f"sample decode head: {best['sample']!r}")


if __name__ == "__main__":
    main()
