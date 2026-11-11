# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PP2 milestone M1: PP2xTP4 logits are bit-identical to TP4 with a simulated
layer-20 boundary.

record: one engine load (eager, no drafter, synchronous scheduling), the
128-token counting prompt and 65 greedy steps (prefill + 64 decode). The last
pipeline stage's TP rank 0 saves every ``compute_logits`` output as it came
from the model; the boundary tap (``DeepseekV41Model.pp_boundary_tap``, when
the model has one) saves ``hidden``/``pre`` at the stage boundary, real on PP2
and simulated on TP4.

    # reference: TP4 with VLLM_V41_PP_BOUNDARY_SIM=20 (set by this tool)
    pp2_m1.py record --pp 1 --offload 0.30 --output state/m1-ref
    # PP2xTP4, every expert resident
    pp2_m1.py record --pp 2 --offload 0 --output state/m1-pp2
    pp2_m1.py compare state/m1-ref state/m1-pp2

compare exits 0 only if both runs have the same number of steps, the same
tokens and the same bits in every logits row. Boundary tensors are a
diagnostic (they place a failure before or after the hop); they do not decide
the verdict.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

STEPS = 65  # prefill + 64 decode steps; one compute_logits call each


def install(worker, output):
    """Worker side: wrap compute_logits (last stage, TP rank 0) and the tap."""
    import torch

    from vllm.distributed import get_pp_group, get_tp_group

    pp, tp = get_pp_group(), get_tp_group()
    worker._m1 = state = {
        "output": output,
        "pp_rank": pp.rank_in_group,
        "tp_rank": tp.rank_in_group,
        "logits": [],
        "boundary": [],
        "originals": [],
    }
    if tp.rank_in_group != 0:
        return None
    model = worker.model_runner.model
    if pp.is_last_rank:
        original = model.compute_logits

        def compute_logits(*args, **kwargs):
            logits = original(*args, **kwargs)
            value = logits[0] if isinstance(logits, tuple) else logits
            if value is not None:
                state["logits"].append(value.detach().to("cpu", copy=True))
            return logits

        state["originals"].append((model, "compute_logits", original))
        model.compute_logits = compute_logits
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "pp_boundary_tap"):

        def tap(hidden, pre):
            if torch.cuda.is_current_stream_capturing():
                return
            state["boundary"].append(
                (
                    hidden.detach().to("cpu", copy=True),
                    pre.detach().to("cpu", copy=True),
                )
            )

        state["originals"].append((inner, "pp_boundary_tap", inner.pp_boundary_tap))
        inner.pp_boundary_tap = tap
    return {"pp_rank": pp.rank_in_group, "tap": inner is not None}


def drain(worker):
    """Worker side: save what this rank recorded, undo the wraps."""
    import torch

    state = getattr(worker, "_m1", None)
    if state is None:
        return None
    for owner, name, original in reversed(state["originals"]):
        setattr(owner, name, original)
    out = Path(state["output"])
    saved = {"pp_rank": state["pp_rank"], "tp_rank": state["tp_rank"]}
    if state["logits"]:
        torch.save(state["logits"], out / "logits.pt")
        saved["logits_steps"] = len(state["logits"])
    if state["boundary"]:
        torch.save(state["boundary"], out / "boundary.pt")
        saved["boundary_steps"] = len(state["boundary"])
    worker._m1 = None
    return saved


def memory_report(worker):
    import torch

    from vllm.distributed import get_pp_group, get_tp_group

    free, total = torch.cuda.mem_get_info()
    return {
        "pp_rank": get_pp_group().rank_in_group,
        "tp_rank": get_tp_group().rank_in_group,
        "device": torch.accelerator.current_device_index(),
        "used_gib": round((total - free) / 2**30, 3),
        "allocated_gib": round(torch.accelerator.memory_allocated() / 2**30, 3),
        "max_reserved_gib": round(torch.accelerator.max_memory_reserved() / 2**30, 3),
    }


def record(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.pp == 1 and not args.no_sim:
        os.environ["VLLM_V41_PP_BOUNDARY_SIM"] = "20"
    elif args.pp == 2:
        os.environ.pop("VLLM_V41_PP_BOUNDARY_SIM", None)
    os.environ["V41_BENCH_PP"] = str(args.pp)
    os.environ["V41_BENCH_OFFLOAD"] = str(args.offload)
    from benchmarks.deepseek_v41.main_smoke import create_llm
    from benchmarks.deepseek_v41.performance import make_prompt
    from vllm import SamplingParams

    started = time.monotonic()
    llm = create_llm(
        args.model_root,
        args.chunk,
        None,
        args.offload,
        kv_cache_mib=args.kv_mib,
        decode_graph=False,
        draft_graph=False,
        custom_allreduce=not args.nccl,
    )
    load_seconds = time.monotonic() - started
    try:
        prompt = make_prompt(llm.get_tokenizer(), args.prompt_len)
        installed = llm.collective_rpc(install, args=(str(out),))
        begin = time.monotonic()
        result = llm.generate(
            [{"prompt_token_ids": prompt}],
            SamplingParams(temperature=0, max_tokens=STEPS, ignore_eos=True),
            use_tqdm=False,
        )
        seconds = time.monotonic() - begin
        saved = llm.collective_rpc(drain)
        memory = llm.collective_rpc(memory_report)
    finally:
        del llm
    meta = {
        "pp": args.pp,
        "offload": args.offload,
        "custom_allreduce": not args.nccl,
        "prompt_len": len(prompt),
        "prompt_ids": list(prompt),
        "token_ids": list(result[0].outputs[0].token_ids),
        "load_seconds": round(load_seconds, 1),
        "generate_seconds": round(seconds, 2),
        "installed": installed,
        "saved": saved,
        "memory_by_rank": memory,
        "env": {
            k: v
            for k, v in sorted(os.environ.items())
            if k.startswith(("VLLM_V41", "V41_BENCH", "CUDA_VISIBLE"))
        },
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    counts = [s.get("logits_steps") for s in saved if s and "logits_steps" in s]
    print(f"recorded {counts} logits steps, tokens {len(meta['token_ids'])}, {out}")
    return 0 if counts == [STEPS] else 1


def _bits(t):
    import torch

    width = {2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()]
    return t.contiguous().view(width)


def compare(args):
    import torch

    ref, run = args.reference, args.candidate
    meta_ref = json.loads((ref / "meta.json").read_text())
    meta_run = json.loads((run / "meta.json").read_text())
    a = torch.load(ref / "logits.pt")
    b = torch.load(run / "logits.pt")
    report = {
        "steps": [len(a), len(b)],
        "prompt_equal": meta_ref["prompt_ids"] == meta_run["prompt_ids"],
        "tokens_equal": meta_ref["token_ids"] == meta_run["token_ids"],
        "first_bit_diff_step": None,
        "argmax_diff_steps": [],
        "max_abs_diff": 0.0,
    }
    for step, (x, y) in enumerate(zip(a, b)):
        if x.shape != y.shape or x.dtype != y.dtype:
            if report["first_bit_diff_step"] is None:
                report["first_bit_diff_step"] = step
            report["shape_or_dtype"] = [str(x.shape), str(y.shape), str(x.dtype)]
            break
        if not torch.equal(_bits(x), _bits(y)):
            if report["first_bit_diff_step"] is None:
                report["first_bit_diff_step"] = step
            diff = (x.float() - y.float()).abs().max().item()
            report["max_abs_diff"] = max(report["max_abs_diff"], diff)
        if not torch.equal(x.float().argmax(-1), y.float().argmax(-1)):
            report["argmax_diff_steps"].append(step)
    if (ref / "boundary.pt").is_file() and (run / "boundary.pt").is_file():
        ba, bb = torch.load(ref / "boundary.pt"), torch.load(run / "boundary.pt")
        first = None
        for step, ((ha, pa), (hb, pb)) in enumerate(zip(ba, bb)):
            same = (
                ha.shape == hb.shape
                and pa.shape == pb.shape
                and torch.equal(_bits(ha), _bits(hb))
                and torch.equal(_bits(pa), _bits(pb))
            )
            if not same:
                first = step
                break
        report["boundary"] = {"steps": [len(ba), len(bb)], "first_diff_step": first}
    else:
        report["boundary"] = "not recorded (no pp_boundary_tap on one side)"
    passed = (
        len(a) == len(b) == STEPS
        and report["prompt_equal"]
        and report["tokens_equal"]
        and report["first_bit_diff_step"] is None
    )
    report["verdict"] = "PASS" if passed else "FAIL"
    print(json.dumps(report, indent=1))
    return 0 if passed else 1


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--pp", type=int, choices=(1, 2), required=True)
    r.add_argument("--offload", type=float, required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--model-root", type=Path, default=Path("/path/to/ds41-models"))
    r.add_argument("--prompt-len", type=int, default=128)
    r.add_argument("--chunk", type=int, default=512)
    r.add_argument("--kv-mib", type=int, default=384)
    r.add_argument("--nccl", action="store_true", help="disable custom all-reduce")
    r.add_argument(
        "--no-sim", action="store_true", help="TP4 without the simulated boundary"
    )
    c = sub.add_parser("compare")
    c.add_argument("reference", type=Path)
    c.add_argument("candidate", type=Path)
    args = p.parse_args()
    return record(args) if args.cmd == "record" else compare(args)


if __name__ == "__main__":
    sys.exit(main())
