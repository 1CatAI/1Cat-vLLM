# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rank-0 torch.profiler trace of steady DSpark rounds (no nsys, low overhead)."""

import argparse
import json
import time
from pathlib import Path


def start(worker, path):
    if worker.local_rank != 0:
        return
    import torch

    runner = worker.model_runner
    originals = worker._tp_originals = []

    def wrap(owner, name, label):
        original = getattr(owner, name)
        originals.append((owner, name, original))

        def traced(*a, **k):
            with torch.profiler.record_function(f"V41::{label}"):
                return original(*a, **k)

        setattr(owner, name, traced)

    wrap(runner, "execute_model", "execute_model")
    wrap(runner, "sample_tokens", "sample_tokens") if hasattr(
        runner, "sample_tokens"
    ) else None
    wrap(runner, "_model_forward", "target_forward")
    wrap(runner, "_sample", "target_sampling")
    wrap(runner.model, "compute_logits", "target_head")
    wrap(runner.model, "commit_speculative_state", "commit")
    wrap(runner.drafter, "propose", "draft_propose") if hasattr(
        runner.drafter, "propose"
    ) else None
    wrap(runner.drafter.model, "forward", "draft_forward")
    wrap(runner.drafter, "_sample_draft_tokens", "draft_sampling")
    worker._tp = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    )
    worker._tp_path = path
    worker._tp.__enter__()


def stop(worker):
    if worker.local_rank != 0:
        return None
    worker._tp.__exit__(None, None, None)
    worker._tp.export_chrome_trace(worker._tp_path)
    for owner, name, original in reversed(worker._tp_originals):
        setattr(owner, name, original)
    return worker._tp_path


def main(args):
    from benchmarks.deepseek_v41.acceptance import request
    from benchmarks.deepseek_v41.main_smoke import create_llm, trim_allocator
    from benchmarks.deepseek_v41.performance import make_prompt

    llm = create_llm(
        Path("/home/ymzx/ds41-models"),
        args.chunk,
        Path("/home/ymzx/ds41-dev/dspark-v41"),
        args.offload,
        kv_cache_mib=384,
        decode_graph=True,
        draft_graph=True,
        custom_allreduce=True,
    )
    try:
        llm.collective_rpc(trim_allocator)
        prompt = make_prompt(llm.get_tokenizer(), 128)
        for _ in range(2):
            request(llm, prompt, 128)
        llm.collective_rpc(start, args=(str(args.output),))
        engine = llm.llm_engine
        original = engine.step
        walls = []

        def step():
            t = time.perf_counter_ns()
            out = original()
            walls.append((time.perf_counter_ns() - t) / 1e6)
            return out

        engine.step = step
        result = request(llm, prompt, 64)
        engine.step = original
        llm.collective_rpc(stop)
        Path(str(args.output) + ".steps.json").write_text(
            json.dumps(dict(walls=walls, result=result))
        )
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--chunk", type=int, default=2048)
    p.add_argument("--offload", type=float, default=0.34)
    main(p.parse_args())
