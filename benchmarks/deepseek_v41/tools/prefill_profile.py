# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rank-0 torch.profiler trace of one long prefill (max_tokens=1) + wall TTFTs."""

import argparse
import json
import time
from pathlib import Path


def mem_start(worker):
    if worker.local_rank != 0:
        return
    import torch

    torch.cuda.memory._record_memory_history(max_entries=400000, stacks="python")


def mem_dump(worker, path):
    if worker.local_rank != 0:
        return None
    import torch

    torch.cuda.memory._dump_snapshot(path)
    torch.cuda.memory._record_memory_history(enabled=None)
    return "snapshot %s peak allocated %.0f MiB reserved %.0f MiB" % (
        path,
        torch.accelerator.max_memory_allocated() / 2**20,
        torch.accelerator.max_memory_reserved() / 2**20,
    )


def main(args):
    import profile_torch
    from fresh_prompts import WARMUP
    from v41_e2e import chat, long_prompt

    from benchmarks.deepseek_v41.main_smoke import create_llm, trim_allocator
    from vllm import SamplingParams

    llm = create_llm(
        Path("/home/ymzx/ds41-models"),
        args.chunk,
        Path("/home/ymzx/ds41-dev/dspark-v41"),
        args.offload,
        kv_cache_mib=args.kv_mib,
        decode_graph=True,
        draft_graph=True,
        custom_allreduce=True,
    )
    tok = llm.get_tokenizer()
    engine = llm.llm_engine
    llm.collective_rpc(trim_allocator)

    def run(prompt, name):
        engine.add_request(
            name,
            {"prompt_token_ids": prompt},
            SamplingParams(max_tokens=1, temperature=0),
        )
        t = time.perf_counter()
        while engine.has_unfinished_requests():
            engine.step()
        return time.perf_counter() - t

    for i, text in enumerate(WARMUP[:2]):
        run(chat(tok, text), f"w{i}")
    report = {}
    for n in args.lengths:
        if n < 1024:
            from benchmarks.deepseek_v41.performance import make_prompt

            prompt = make_prompt(tok, n)
        else:
            prompt = long_prompt(tok, Path("/home/ymzx/ds41-dev/src"), n)
        report[n] = round(run(prompt, f"warm-{n}"), 2)
        print("untraced TTFT", n, report[n], flush=True)
    if args.memsnap:
        prompt = long_prompt(tok, Path("/home/ymzx/ds41-dev/src"), args.trace_len)
        llm.collective_rpc(mem_start)
        run(prompt, "memsnap")
        print(llm.collective_rpc(mem_dump, args=(args.memsnap,))[0], flush=True)
        return
    if not args.trace:
        Path(args.output).write_text(json.dumps(dict(untraced=report)))
        return
    prompt = long_prompt(tok, Path("/home/ymzx/ds41-dev/src"), args.trace_len)
    llm.collective_rpc(profile_torch.start, args=(args.trace,))
    ttft = run(prompt, "traced")
    llm.collective_rpc(profile_torch.stop)
    print("traced TTFT", args.trace_len, round(ttft, 2), flush=True)
    Path(args.output).write_text(json.dumps(dict(untraced=report, traced=ttft)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--chunk", type=int, default=2048)
    p.add_argument("--lengths", type=int, nargs="*", default=[128, 2048, 6144])
    p.add_argument("--trace-len", type=int, default=6144)
    p.add_argument("--trace", default="")
    p.add_argument("--memsnap", default="")
    p.add_argument("--output", required=True)
    p.add_argument("--kv-mib", type=int, default=384)
    p.add_argument("--offload", type=float, default=0.34)
    main(p.parse_args())
