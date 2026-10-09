# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare installed Q4_K/Q6_K output projection plus HCX schedules.

The trusted weights file contains eight dictionaries: down[320,10240],
inj[4,10240], up[10240,320], nw[10240], fmt (0=Q4_K, 2=Q6_K), name,
and projection[rank]=(codes,high,scale) from the production GGUF plane packer.
K/rank=1536, N=2560. Inputs are synthetic and already normalized. This is a
complete-chain microbenchmark, not a model numerical or latency gate. Acquire
the campaign's four-GPU locks before launching with torch.distributed.run.
"""

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist
import vllm._C as core

from vllm.models.qwen4_exp.nvidia.sm70_hcx import Sm70HcxRuntime, pack_down, pack_up


def assert_bits(actual, expected):
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(
            value.view(torch.int16), reference.view(torch.int16), rtol=0, atol=0
        )


@torch.inference_mode()
def benchmark_rows(args, runtime, weights, rows, rank):
    x = torch.empty(rows, 1536, device="cuda", dtype=torch.float16)
    partial = torch.empty(rows, 2560, device="cuda", dtype=torch.float16)
    hidden = torch.empty(rows, 10240, device="cuda", dtype=torch.float16)
    injection = torch.empty(rows, 4, device="cuda", dtype=torch.float16)
    outputs = tuple(torch.empty_like(value) for value in (hidden, partial, injection))
    ws = torch.empty(80 * 8 * 256, device="cuda", dtype=torch.float32)
    counters = torch.zeros(80, device="cuda", dtype=torch.int32)
    gated = torch.empty_like(x)
    gate_weight = torch.ones(128, device="cuda", dtype=torch.float16)
    gate_scratch = torch.empty_like(x)
    modes = ("separate", "fused_reference", "fused_selected", "producer_only")

    def run(weight, mode, norm=None, with_gate=False, full=True):
        codes, high, scale = weight["planes"]
        if mode in ("separate", "producer_only"):
            torch.ops._C.gguf_dense_segments_sm70_out(
                x,
                [codes],
                [high],
                [scale],
                [partial],
                [weight["fmt"]],
                [2560],
                1536,
                1,
                8,
                ws,
                counters,
                None,
            )
            if mode == "producer_only":
                return
        fused = mode.startswith("fused_")
        torch.ops._C.sm70_hcx_out(
            partial,
            None,
            hidden,
            injection,
            weight["norm"] if norm is None else norm,
            1e-6,
            weight["down"],
            weight["up"],
            *outputs,
            runtime.xn,
            runtime.sq,
            runtime.dpart,
            runtime.bar,
            runtime.seq,
            runtime.ar,
            runtime.lora,
            runtime.hb,
            runtime.logical_rank,
            None,
            int(full),
            x if fused else None,
            codes if fused else None,
            high if fused else None,
            scale if fused else None,
            weight["fmt"] if fused else -1,
            gated if with_gate else None,
            gate_weight if with_gate else None,
            1e-6,
            gate_scratch if with_gate else None,
            mode != "fused_reference",
        )

    def change_input(seed, zero=False):
        torch.manual_seed(seed)
        hidden.normal_()
        injection.normal_()
        torch.manual_seed(seed + rank)
        x.normal_(0, 0.5)
        gated.normal_()
        if zero:
            x.zero_()
            hidden.zero_()
            injection.zero_()

    checks = {mode: 0 for mode in modes if mode != "producer_only"}
    for seed in (20261010, 20261011, 20261012, 20261013):
        change_input(seed, zero=seed == 20261013)
        for weight in weights:
            norm = weight["norm"][:2560].contiguous() if seed == 20261012 else None
            run(weight, "separate", norm)
            reference = tuple(value.clone() for value in outputs)
            # A fused producer must compute its own partial instead of reading
            # the result left by the separate reference call.
            partial.fill_(float("nan"))
            for mode in ("fused_reference", "fused_selected"):
                run(weight, mode, norm)
                assert_bits(outputs, reference)
                checks[mode] += 1

    wrap_checks = 0
    for counter in (2147483552, -96):
        runtime.bar[2:].fill_(counter)
        change_input(20261100 + wrap_checks)
        for weight in (weights[0], weights[3]):
            run(weight, "fused_reference")
            reference = tuple(value.clone() for value in outputs)
            for mode in ("fused_selected", "separate", "fused_selected"):
                run(weight, mode)
                assert_bits(outputs, reference)
                wrap_checks += 1

    fallback_checks = 0
    for weight in (weights[0], weights[3]):
        for with_gate, full in ((True, True), (False, False)):
            run(weight, "fused_reference", with_gate=with_gate, full=full)
            reference = tuple(value.clone() for value in outputs)
            run(weight, "fused_selected", with_gate=with_gate, full=full)
            assert_bits(outputs, reference)
            fallback_checks += 1

    graphs = {}
    for mode in modes:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(args.repeats):
                for weight in weights:
                    run(weight, mode)
        graphs[mode] = graph
        graph.replay()
    torch.cuda.synchronize()
    for seed in (20261200, 20261201, 20261202):
        change_input(seed)
        graphs["separate"].replay()
        reference = tuple(value.clone() for value in outputs)
        partial.fill_(float("nan"))
        for mode in ("fused_reference", "fused_selected"):
            graphs[mode].replay()
            assert_bits(outputs, reference)
            checks[mode] += 1
    for _ in range(4):
        for graph in graphs.values():
            graph.replay()
    torch.cuda.synchronize()
    times = {mode: [] for mode in modes}
    for sample in range(args.samples):
        offset = sample % len(modes)
        cycle = modes[offset:] + modes[:offset]
        for mode in cycle + cycle[::-1]:
            dist.barrier()
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            graphs[mode].replay()
            end.record()
            end.synchronize()
            times[mode].append(
                start.elapsed_time(end) * 1000 / args.repeats / len(weights)
            )
    gathered = [None] * 4
    dist.all_gather_object(
        gathered,
        dict(
            rank=rank,
            samples_us=times,
            exact_checks=checks,
            counter_checks=wrap_checks,
            fallback_checks=fallback_checks,
        ),
    )
    critical = {
        mode: [
            max(record["samples_us"][mode][i] for record in gathered)
            for i in range(len(times[mode]))
        ]
        for mode in modes
    }
    del graphs
    torch.cuda.synchronize()
    return dict(
        rows=rows,
        critical_median_us={mode: statistics.median(v) for mode, v in critical.items()},
        critical_samples_us=critical,
        rank_records=gathered,
    )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--rows", type=int, nargs="+", default=[1, 5, 8], choices=range(1, 9)
    )
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=64)
    args = parser.parse_args()
    if args.samples < 1 or args.repeats < 1:
        parser.error("samples and repeats must be positive")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    runtime = Sm70HcxRuntime(dist.group.WORLD, torch.device("cuda", rank))
    if dist.get_world_size() != 4 or not runtime.enabled or not runtime.full:
        raise RuntimeError("Full-mesh TP4 HCX is required")
    records = torch.load(args.weights, map_location="cpu", weights_only=True)
    if len(records) != 8 or {r["fmt"] for r in records} != {0, 2}:
        raise ValueError(
            "Expected eight real projection/HC pairs including Q4_K and Q6_K"
        )
    weights = []
    for record in records:
        down = torch.cat(
            (record["down"], record["inj"], torch.zeros(12, 10240, dtype=torch.float16))
        ).cuda()
        weights.append(
            dict(
                down=pack_down(down, runtime.logical_rank),
                up=pack_up(record["up"].cuda(), runtime.logical_rank),
                norm=record["nw"].cuda(),
                fmt=record["fmt"],
                name=record["name"],
                planes=[v.cuda() for v in record["projection"][runtime.logical_rank]],
            )
        )
    results = [benchmark_rows(args, runtime, weights, rows, rank) for rows in args.rows]
    if rank == 0:
        result = dict(
            scope=(
                "Installed complete projection plus HC chain; "
                "synthetic normalized inputs; no model claim"
            ),
            pairs=len(weights),
            boundaries_per_replay=args.repeats * len(weights),
            producer_config=dict(
                K_per_rank=1536, N=2560, warps=8, split=1, tiles_per_cta=1
            ),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            weights_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),
            core_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
            results=results,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps({r["rows"]: r["critical_median_us"] for r in results}),
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
