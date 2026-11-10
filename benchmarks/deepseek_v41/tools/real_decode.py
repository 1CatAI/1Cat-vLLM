# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-text decode diagnosis: per-step wall time, tokens/step, expert misses/step; optional trace."""

import argparse
import gc
import json
import time
from pathlib import Path


def stats(worker, reset=False):
    from vllm.model_executor.layers.quantization.gguf_shared_runtime import (
        SharedExpertRank,
    )

    ranks = [o for o in gc.get_objects() if type(o) is SharedExpertRank]
    out = []
    for r in ranks:
        s = r.statistics
        if reset:
            s.zero_()
        else:
            out.append(s.cpu().tolist())
    if not reset and ranks and getattr(ranks[0], "miss_hist", None) is not None:
        print("MISS_HIST", ranks[0].miss_hist.cpu().tolist(), flush=True)
    return out


def main(args):
    import profile_torch
    from v41_e2e import REAL_TEXT as CASES
    from v41_e2e import chat

    from benchmarks.deepseek_v41.main_smoke import create_llm, trim_allocator
    from benchmarks.deepseek_v41.performance import make_prompt

    llm = create_llm(
        Path("/home/ymzx/ds41-models"),
        2048,
        Path("/home/ymzx/ds41-dev/dspark-v41"),
        0.34,
        kv_cache_mib=384,
        decode_graph=True,
        draft_graph=True,
        custom_allreduce=True,
    )
    tok = llm.get_tokenizer()
    engine = llm.llm_engine
    from vllm import SamplingParams

    report = []
    try:
        llm.collective_rpc(trim_allocator)
        if args.corpus == "fresh":
            from fresh_prompts import FRESH, WARMUP

            for i, text in enumerate(WARMUP):
                engine.add_request(
                    f"warm-{i}",
                    {"prompt_token_ids": chat(tok, text)},
                    SamplingParams(max_tokens=64, temperature=0),
                )
                while engine.has_unfinished_requests():
                    engine.step()
            prompts = [(n, chat(tok, t)) for n, t in FRESH[: args.cases]]
            reps = 1
        else:
            prompts = [("count-128", make_prompt(tok, 128))] + [
                (n, chat(tok, t)) for n, t in CASES[: args.cases]
            ]
            reps = 2
        for name, prompt in prompts:
            for rep in range(reps):  # second pass = experts warmed by the same text
                llm.collective_rpc(stats, args=(True,))
                engine.add_request(
                    f"{name}-{rep}-{time.monotonic_ns()}",
                    {"prompt_token_ids": prompt},
                    SamplingParams(
                        max_tokens=args.tokens,
                        temperature=0,
                        logprobs=0 if args.logprobs else None,
                    ),
                )
                steps = []
                count = 0
                ids = []
                last = None
                while engine.has_unfinished_requests():
                    t = time.perf_counter()
                    outs = engine.step()
                    dt = time.perf_counter() - t
                    n = (
                        sum(len(o.outputs[0].token_ids) for o in outs)
                        if outs
                        else count
                    )
                    if outs and outs[0].outputs[0].token_ids:
                        ids = list(outs[0].outputs[0].token_ids)
                        last = outs[0].outputs[0]
                    steps.append((dt, max(0, n - count)))
                    count = max(count, n)
                st = llm.collective_rpc(stats)[0]
                req = sum(sum(l[0] for l in s) for s in st[:1])
                miss = sum(sum(l[1] for l in s) for s in st[:1])
                dec = steps[2:]  # drop prefill steps
                wall = sum(d for d, _ in dec)
                toks = sum(k for _, k in dec)
                entry = dict(
                    name=name,
                    rep=rep,
                    prompt=len(prompt),
                    tokens=count,
                    decode_steps=len(dec),
                    decode_wall_s=round(wall, 3),
                    ms_per_step=round(1000 * wall / max(1, len(dec)), 2),
                    tok_per_s=round(toks / max(wall, 1e-9), 1),
                    tok_per_step=round(toks / max(1, len(dec)), 2),
                    layer_requests=req,
                    layer_misses=miss,
                    misses_per_step=round(miss / max(1, len(dec)), 2),
                    ttft_s=round(sum(d for d, _ in steps[:2]), 3),
                    ids=ids,
                )
                if args.logprobs and last is not None and last.logprobs is not None:
                    entry["lp"] = [
                        round(float(e[t].logprob), 4) if e and t in e else None
                        for t, e in zip(last.token_ids, last.logprobs)
                    ]
                print(json.dumps(entry), flush=True)
                report.append(entry)
        fresh = [e for e in report if e["rep"] == 0]
        steps_total = sum(e["decode_steps"] for e in fresh)
        wall_total = sum(e["decode_wall_s"] for e in fresh)
        tok_total = sum(e["tok_per_step"] * e["decode_steps"] for e in fresh)
        summary = dict(
            summary=True,
            corpus=args.corpus,
            prompts=len(fresh),
            ms_per_round=round(1000 * wall_total / max(1, steps_total), 2),
            tok_per_s=round(tok_total / max(wall_total, 1e-9), 1),
            tok_per_round=round(tok_total / max(1, steps_total), 2),
            misses_per_round=round(
                sum(e["layer_misses"] for e in fresh) / max(1, steps_total), 2
            ),
        )
        print(json.dumps(summary), flush=True)
        report.append(summary)
        if args.trace:
            llm.collective_rpc(profile_torch.start, args=(str(args.trace),))
            if args.corpus == "fresh":
                from fresh_prompts import FRESH

                name, text = FRESH[min(args.cases, len(FRESH) - 1)]
                prompt = chat(tok, text)
            else:
                name, prompt = prompts[1]
            engine.add_request(
                f"trace-{time.monotonic_ns()}",
                {"prompt_token_ids": prompt},
                SamplingParams(max_tokens=64, temperature=0),
            )
            while engine.has_unfinished_requests():
                engine.step()
            llm.collective_rpc(profile_torch.stop)

        def pred(worker):
            from vllm.models.deepseek_v41 import model as m

            return m.prediction_report()

        rep = llm.collective_rpc(pred)[0]
        if rep:
            print(json.dumps(dict(prediction=rep)), flush=True)
        Path(args.output).write_text(json.dumps(report, indent=1))
    finally:
        engine.engine_core.shutdown()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--trace")
    p.add_argument("--cases", type=int, default=4)
    p.add_argument("--corpus", default="cases", choices=["cases", "fresh"])
    p.add_argument("--tokens", type=int, default=200)
    p.add_argument("--logprobs", action="store_true")
    main(p.parse_args())
