#!/usr/bin/env python3
"""Paris smoke + short coherent gen. Exit 0=ok, 2=salad/fail."""
import argparse, json, re, sys, urllib.request

def post(url, payload, timeout=180):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=timeout))

def text_of(r):
    m = r["choices"][0]["message"]
    return ((m.get("content") or "") + (m.get("reasoning_content") or "")).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8003")
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    base = args.url.rstrip("/")
    api = base + "/v1/chat/completions"

    # 1) Paris
    r = post(api, {
        "model": args.model,
        "messages": [{"role": "user", "content": "What is the capital of France? Answer in one word."}],
        "max_tokens": 2048,
        "temperature": 0,
    })
    t = text_of(r)
    paris = "paris" in t.lower()
    print(f"[{args.label}] paris={'OK' if paris else 'FAIL'} sample={t[:200]!r}")

    # 2) short math (catches scale salad / !!!! )
    r2 = post(api, {
        "model": args.model,
        "messages": [{"role": "user", "content": "What is 17*19? Reply with just the number."}],
        "max_tokens": 1024,
        "temperature": 0,
    })
    t2 = text_of(r2)
    has_323 = bool(re.search(r"\b323\b", t2))
    salad = bool(re.search(r"!{4,}|[\uFFFD]{2,}|asdfs|lorem", t2, re.I)) or (len(t2) > 20 and not re.search(r"\d", t2))
    print(f"[{args.label}] math323={'OK' if has_323 else 'FAIL'} salad={'YES' if salad else 'no'} sample={t2[:200]!r}")

    ok = paris and has_323 and not salad
    print(f"[{args.label}] RESULT={'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 2)

if __name__ == "__main__":
    main()
