# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""API soak: streaming chat requests of mixed lengths against the OpenAI server.

Per request: TTFT, decode tokens/s, output tokens, finish reason, errors.
Mix: mixed-240 questions, long documents (2K/8K/15K prompt tokens), growing
multi-turn chats, and sampled (temperature 0.7) requests.
"""

import argparse
import json
import random
import time
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:18000"
MODEL = "deepseek-v4.1-flash"


def stream(messages, max_tokens, temperature, timeout=1800):
    body = dict(
        model=MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
        stream_options={"include_usage": True},
        chat_template_kwargs={"enable_thinking": False},
    )
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.monotonic()
    first = None
    text, usage, finish, chunks = [], None, None, 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            for choice in obj.get("choices", []):
                delta = choice.get("delta", {}).get("content")
                if delta:
                    if first is None:
                        first = time.monotonic()
                    text.append(delta)
                    chunks += 1
                finish = choice.get("finish_reason") or finish
    end = time.monotonic()
    out_tokens = usage["completion_tokens"] if usage else None
    decode = (end - first) if first else None
    return dict(
        ttft=round((first or end) - start, 3),
        total=round(end - start, 3),
        prompt_tokens=usage["prompt_tokens"] if usage else None,
        out_tokens=out_tokens,
        decode_tok_s=round((out_tokens - 1) / decode, 1)
        if decode and out_tokens and out_tokens > 1
        else None,
        finish=finish,
        chunks=chunks,
        text="".join(text),
    )


def long_doc(src_root, approx_tokens):
    files = sorted(Path(src_root).rglob("*.py"))
    random.shuffle(files)
    parts, size = [], 0
    for f in files:
        try:
            t = f.read_text()
        except Exception:
            continue
        parts.append(t)
        size += len(t) // 3  # ~3 characters per token for code
        if size >= approx_tokens:
            break
    doc = "".join(parts)[: approx_tokens * 3]
    return doc + "\n\nSummarize what the code above does in five bullet points."


def main(args):
    random.seed(7)
    sys_path = Path(__file__).resolve().parent
    import sys

    sys.path.insert(0, str(sys_path))
    from qgrade import prompt_of

    items = json.load(open(args.items))
    plan = []
    for i, item in enumerate(items):
        plan.append(
            (
                "q",
                dict(
                    messages=[{"role": "user", "content": prompt_of(item)}],
                    max_tokens=512,
                    temperature=0.0,
                ),
            )
        )
        if i % 10 == 5:
            plan.append(
                (
                    "sample",
                    dict(
                        messages=[{"role": "user", "content": prompt_of(item)}],
                        max_tokens=256,
                        temperature=0.7,
                    ),
                )
            )
        if i % 40 == 20:
            for n in (2048, 8192, 15000):
                plan.append(
                    (
                        f"long{n}",
                        dict(
                            messages=[
                                {"role": "user", "content": long_doc(args.src_root, n)}
                            ],
                            max_tokens=256,
                            temperature=0.0,
                        ),
                    )
                )
        if i % 60 == 30:
            plan.append(("multiturn", None))
    if args.sample_stress:
        plan = [
            (
                "sample",
                dict(
                    messages=[
                        {"role": "user", "content": prompt_of(items[i % len(items)])}
                    ],
                    max_tokens=256,
                    temperature=0.8,
                ),
            )
            for i in range(args.sample_stress)
        ]
    out = open(args.output, "w")
    for k, (kind, req) in enumerate(plan[: args.limit]):
        rec = dict(k=k, kind=kind)
        try:
            if kind == "multiturn":
                msgs, turns = [], []
                for turn in (
                    "给我讲讲长城的历史。",
                    "那它在明朝有什么变化？",
                    "再总结成三点。",
                    "用英文翻译你的总结。",
                ):
                    msgs.append({"role": "user", "content": turn})
                    r = stream(msgs, 384, 0.0)
                    msgs.append({"role": "assistant", "content": r["text"]})
                    r.pop("text")
                    turns.append(r)
                rec.update(
                    turns=turns,
                    ok=all(t["finish"] in ("stop", "length") for t in turns),
                )
            else:
                r = stream(**req)
                rec.update(
                    {k2: v for k2, v in r.items() if k2 != "text"},
                    head=r["text"][:120],
                    ok=r["finish"] in ("stop", "length") and r["out_tokens"],
                )
        except Exception as error:
            rec.update(ok=False, error=repr(error)[:500])
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        print(
            json.dumps(
                {
                    x: rec.get(x)
                    for x in (
                        "k",
                        "kind",
                        "ok",
                        "ttft",
                        "decode_tok_s",
                        "prompt_tokens",
                        "out_tokens",
                        "error",
                    )
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--items", default="/home/ymzx/ds41-dev/evaldata/mixed240.json")
    p.add_argument("--src-root", default="/home/ymzx/ds41-dev/src/vllm")
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, default=10000)
    p.add_argument("--sample-stress", type=int, default=0)
    main(p.parse_args())
