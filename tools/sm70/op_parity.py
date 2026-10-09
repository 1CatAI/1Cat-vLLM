# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-GPU Flash-V100 forward outputs and matched CUDA-event timing.

Outputs must be finite and exactly equal. Performance admission is separate
and opt-in for the Step 3–6 gate: every named case must remain within +/-2%.
Run parent/candidate in fresh processes with identical native binaries and
exclusive GPU ownership; this runner never takes over another GPU owner.
"""

import argparse
import math
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch

from tools.sm70.flash_v100_trace import prepare_spec
from tools.sm70.parity_common import (
    digest,
    provenance,
    read_json,
    require_same_contract,
    runtime,
)


def fixture(case):
    from vllm.v1.attention.backends import flash_attn_v100 as backend

    generator = torch.Generator().manual_seed(1729)
    q, length = case["query"], case["context"]
    head, group, page = case["head"], case["gqa"], case["page"]
    blocks = (length + page - 1) // page

    def rand(shape):
        return (torch.randn(shape, generator=generator) * 0.1).half().cuda()

    query, key, value = rand((q, group, head)), rand((q, 1, head)), rand((q, 1, head))
    cache = rand((blocks, 2, page, 1, head))
    if case["codec"] == "fp8_e4m3":
        cache = cache.to(torch.float8_e4m3fn).view(torch.uint8)
    output = torch.empty_like(query)
    cpu_qsl = torch.tensor([0, q], dtype=torch.int32)
    metadata = SimpleNamespace(
        num_actual_tokens=q,
        max_query_len=q,
        query_start_loc=cpu_qsl.cuda(),
        query_start_loc_cpu=cpu_qsl,
        seq_lens=torch.tensor([length], device="cuda", dtype=torch.int32),
        seq_lens_cpu=torch.tensor([length], dtype=torch.int32),
        max_seq_len=length,
        max_model_len=length,
        block_table=torch.arange(blocks, device="cuda", dtype=torch.int32)[None],
        slot_mapping=torch.arange(length - q, length, device="cuda", dtype=torch.int64),
        causal=True,
        use_cascade=False,
        mm_prefix_range_tensor=None,
    )
    layer = SimpleNamespace(
        _k_scale_float=1.0,
        _v_scale_float=1.0,
        _k_scale=torch.tensor(1.0, device="cuda"),
        _v_scale=torch.tensor(1.0, device="cuda"),
        is_dflash_draft_attn=False,
        layer_name="parity.attn",
    )
    implementation = backend.FlashAttnV100Impl(
        num_heads=group,
        num_kv_heads=1,
        head_size=head,
        scale=head**-0.5,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype=case["codec"],
    )
    spec_case = dict(
        spec=case["spec"],
        mask="none",
        page=page,
        capture=case["graph"],
        draft=case.get("draft", False),
    )
    builder, update = prepare_spec(spec_case, backend, metadata, layer, [q])
    update()
    if case["spec"] in ("dflash2", "mtp") and not any(
        isinstance(v, torch.Tensor) for v in vars(builder).values()
    ):
        raise AssertionError("Speculative case did not create persistent buffers")

    def buffers(refresh=False):
        if refresh and case["spec"] in ("dflash2", "mtp"):
            update()
        return {
            f"{owner}.{name}": value.data_ptr()
            for owner, obj in (("builder", builder), ("metadata", metadata))
            for name, value in vars(obj).items()
            if isinstance(value, torch.Tensor)
        }

    def forward():
        result = implementation.forward(
            layer, query, key, value, cache, metadata, output=output
        )
        if result.data_ptr() != output.data_ptr():
            raise AssertionError("Forward changed the output destination")

    return forward, output, buffers


def measure(forward, output, buffers, *, graph, iterations, repeats):
    for _ in range(5):
        forward()
    torch.accelerator.synchronize()
    pointers = buffers()
    if graph:
        replay = torch.cuda.CUDAGraph()
        with torch.cuda.graph(replay):
            forward()
        call = replay.replay
    else:
        call = forward
    call()
    torch.accelerator.synchronize()
    first = output.detach().cpu().clone()
    if pointers != buffers(refresh=True):
        raise AssertionError(
            "Persistent buffer pointers changed during metadata update"
        )
    call()
    torch.accelerator.synchronize()
    second = output.detach().cpu().clone()
    if not torch.equal(first, second) or not torch.isfinite(first).all():
        raise AssertionError("Baseline forward is nonfinite or not replay-exact")
    if pointers != buffers():
        raise AssertionError("Persistent buffer pointers changed")
    times = []
    for _ in range(repeats):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        for _ in range(iterations):
            call()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / iterations)
    return dict(
        output=first,
        replay_output=second,
        persistent_pointers_stable=True,
        samples_ms=times,
        median_ms=statistics.median(times),
    )


def compare(reference, candidate, performance=False):
    require_same_contract(reference, candidate)
    if not reference["cases"]:
        raise AssertionError("Empty attention evidence cannot establish parity")
    if reference["cases"].keys() != candidate["cases"].keys():
        raise AssertionError("Case sets differ")
    differences = {}
    roles = {
        c["name"]: c.get("performance_role") for c in candidate["contract"]["cases"]
    }
    if performance and not {"decode_xqa", "prefill_75t"} <= set(roles.values()):
        raise AssertionError("Performance gate requires both XQA and 75T workloads")
    for name, original in reference["cases"].items():
        current = candidate["cases"][name]
        for field in ("output", "replay_output"):
            a, b = original[field], current[field]
            if a.shape != b.shape or a.dtype != b.dtype:
                raise AssertionError(f"{name}: tensor contracts differ")
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise AssertionError(f"{name}: nonfinite output")
            error = (a.float() - b.float()).abs().max().item()
            if error != 0 or not torch.equal(a, b):
                raise AssertionError(f"{name}: max_abs={error}; exact parity required")
        if original["routes"] != current["routes"]:
            raise AssertionError(f"{name}: route counters differ")
        if not (
            original["persistent_pointers_stable"]
            and current["persistent_pointers_stable"]
        ):
            raise AssertionError(f"{name}: unstable persistent storage")
        if any(
            not math.isfinite(row["median_ms"]) or row["median_ms"] <= 0
            for row in (original, current)
        ):
            raise AssertionError(f"{name}: invalid timing evidence")
        delta = current["median_ms"] / original["median_ms"] - 1
        differences[name] = dict(max_abs=0.0, timing_delta=delta)
        if performance and roles[name] and abs(delta) > 0.02:
            raise AssertionError(f"{name}: timing delta {delta:.2%} exceeds +/-2%")
    return differences


def record(args):
    import os
    import sys

    os.environ["VLLM_NO_USAGE_STATS"] = "1"
    os.environ["VLLM_FLASH_V100_ROUTE_SUMMARY"] = "1"
    from vllm.v1.attention.backends import flash_attn_v100 as backend

    cases = read_json(args.cases)
    if not cases or len({c["name"] for c in cases}) != len(cases):
        raise ValueError("Require a nonempty matrix with unique case names")
    if args.iterations < 1 or args.repeats < 1:
        raise ValueError("Timing iterations and repeats must be positive")
    for case in cases:
        print(f"Running {case['name']}", flush=True)
        if not 0 < case["query"] <= case["context"] or not case["required_routes"]:
            raise ValueError("Every case needs valid lengths and an observed route")
    report = dict(
        contract=dict(
            runtime=runtime(),
            cases=cases,
            iterations=args.iterations,
            repeats=args.repeats,
        ),
        provenance=provenance(args.source_sha),
        cases={},
    )
    for case in cases:
        before = dict(backend._route_counts)
        forward, output, buffers = fixture(case)
        result = measure(
            forward,
            output,
            buffers,
            graph=case["graph"],
            iterations=args.iterations,
            repeats=args.repeats,
        )
        result["routes"] = {
            k: v - before.get(k, 0)
            for k, v in backend._route_counts.items()
            if v != before.get(k, 0)
        }
        if any(result["routes"].get(r, 0) == 0 for r in case["required_routes"]):
            raise AssertionError(
                f"{case['name']}: required route missing: {result['routes']}"
            )
        report["cases"][case["name"]] = result
    report["contract"]["native_sha256"] = {
        name: digest(module.__file__)
        for name, module in tuple(sys.modules.items())
        if (name.startswith("vllm.") or "flash_attn_v100_cuda" in name)
        and str(getattr(module, "__file__", "")).endswith(".so")
    }
    torch.save(report, args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("record")
    run.add_argument("--cases", type=Path, required=True)
    run.add_argument("--source-sha", required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--iterations", type=int, default=100)
    run.add_argument("--repeats", type=int, default=9)
    check = commands.add_parser("compare")
    check.add_argument("reference", type=Path)
    check.add_argument("candidate", type=Path)
    check.add_argument("--performance", action="store_true")
    args = parser.parse_args()
    if args.command == "record":
        record(args)
    else:
        print(
            compare(
                torch.load(args.reference, weights_only=True),
                torch.load(args.candidate, weights_only=True),
                args.performance,
            )
        )


if __name__ == "__main__":
    main()
