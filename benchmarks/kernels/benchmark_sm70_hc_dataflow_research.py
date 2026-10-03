# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Research-only HC dataflow oracle; does not alter model dispatch."""

import argparse
import fcntl
import importlib.util
import json
import signal
import statistics
from pathlib import Path

import torch


def synchronize(tp):
    for rank in range(tp):
        torch.cuda.synchronize(rank)


def oracle(residual, block, injection, norm, down, up, hc, eps):
    """Independent FP64 reference with the model's FP16 boundaries."""
    m, n = residual.shape
    hidden = n // hc
    combined = (
        residual.double().reshape(m, hc, hidden)
        + block.double()[:, None, :]
        * (2 * torch.sigmoid(injection.double() / hc))[:, :, None]
    ).half()
    f = combined.double()
    normalized = (
        (
            f
            * torch.rsqrt(f.square().mean(-1, keepdim=True) + eps)
            * (1 + norm.double().reshape(1, hc, hidden))
        )
        .half()
        .reshape(m, n)
    )
    projected = (normalized.double() @ down.double().T).half()
    low_rank = up.shape[1]
    scaled = projected[:, :low_rank].double() / hc
    lora = (scaled * torch.sigmoid(scaled)).half()
    gates = (lora.double() @ up.double().T).half().double().reshape(m, hc, hidden)
    output = (
        (torch.sigmoid(gates) * normalized.double().reshape(m, hc, hidden)).sum(1) / hc
    ).half()
    return combined.reshape(m, n), normalized, lora, projected[:, low_rank:], output


def fixture(module, tp, hidden, hc, low_rank, max_m, trace=False):
    g = torch.Generator().manual_seed(8103)
    cpu = {
        "residual": torch.randn(max_m, hc * hidden, generator=g).half(),
        "block": torch.randn(max_m, hidden, generator=g).half(),
        "injection": torch.randn(max_m, hc, generator=g).half(),
        "norm": (torch.randn(hc * hidden, generator=g) * 0.05).half(),
        "down_weight": (
            torch.randn(low_rank + hc, hc * hidden, generator=g) * 0.03
        ).half(),
        "up_weight": (torch.randn(hc * hidden, low_rank, generator=g) * 0.03).half(),
    }
    dc = (low_rank // tp + hc // tp + 1) // 2
    uc = hidden // tp // 8
    local = []
    for rank in range(tp):
        device = torch.device("cuda", rank)
        data = {k: v.to(device) for k, v in cpu.items()}
        data.update(
            combined=torch.empty(
                max_m, hc * hidden, dtype=torch.float16, device=device
            ),
            normalized=torch.empty(
                max_m, hc * hidden, dtype=torch.float16, device=device
            ),
            down=torch.empty(max_m, low_rank + hc, dtype=torch.float16, device=device),
            output=torch.empty(max_m, hidden, dtype=torch.float16, device=device),
            norm_flags=torch.zeros(max_m * hc, dtype=torch.int32, device=device),
            down_flags=torch.zeros(max_m * tp * dc, dtype=torch.int32, device=device),
            up_flags=torch.zeros(max_m * tp * uc, dtype=torch.int32, device=device),
            epochs=torch.zeros(hc + dc + uc, dtype=torch.int32, device=device),
            trace=(
                torch.zeros(max_m, hc + dc + uc, 5, dtype=torch.int64, device=device)
                if trace
                else None
            ),
        )
        local.append(data)
    module.prepare_peers([d["output"] for d in local])
    synchronize(tp)
    return cpu, local


def launch(module, local, rank, width, eps):
    d = local[rank]
    module.run(
        d["residual"][:width],
        d["block"][:width],
        d["injection"][:width],
        d["norm"],
        d["down_weight"],
        d["up_weight"],
        d["combined"][:width],
        d["normalized"][:width],
        d["norm_flags"],
        d["epochs"],
        [v["down"] for v in local],
        [v["down_flags"] for v in local],
        [v["output"] for v in local],
        [v["up_flags"] for v in local],
        rank,
        eps,
        d["trace"],
    )


def capture(module, local, width, eps, steps=1):
    tp = len(local)
    for rank in range(tp):
        with torch.cuda.device(rank):
            launch(module, local, rank, width, eps)
    synchronize(tp)
    graphs = []
    for rank in range(tp):
        with torch.cuda.device(rank):
            graph = torch.cuda.CUDAGraph()
            stream = torch.cuda.Stream(device=rank)
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(steps):
                    launch(module, local, rank, width, eps)
            graphs.append(graph)
    return graphs


def replay(graphs):
    for rank, graph in enumerate(graphs):
        with torch.cuda.device(rank):
            graph.replay()


def run_case(module, tp, hidden, hc, low_rank, trace=False):
    widths = (1, 5, 7, 5, 1, 7, 1, 5)
    eps = 1e-6
    cpu, local = fixture(module, tp, hidden, hc, low_rank, max(widths), trace)
    graphs = {m: capture(module, local, m, eps) for m in sorted(set(widths))}
    # All widths share the same generation counters and reusable buffers.
    expected_generation = len(graphs)
    records = []
    for i, width in enumerate(widths):
        scale = (0.25, 0.5, 1.0, 2.0)[i % 4]
        residual = (cpu["residual"] * scale).half()
        block = (cpu["block"] * scale).half()
        injection = (cpu["injection"] + i * 0.01).half()
        for rank, data in enumerate(local):
            with torch.cuda.device(rank):
                data["residual"].copy_(residual)
                data["block"].copy_(block)
                data["injection"].copy_(injection)
                for name in ("combined", "normalized", "down", "output"):
                    data[name].fill_(float("nan"))
        synchronize(tp)
        expected = oracle(
            residual[:width],
            block[:width],
            injection[:width],
            cpu["norm"],
            cpu["down_weight"],
            cpu["up_weight"],
            hc,
            eps,
        )
        replay(graphs[width])
        synchronize(tp)
        expected_generation += 1
        errors = {}
        for rank, data in enumerate(local):
            observed = (
                data["combined"][:width].cpu(),
                data["normalized"][:width].cpu(),
                data["down"][:width, :low_rank].cpu(),
                data["down"][:width, low_rank:].cpu(),
                data["output"][:width].cpu(),
            )
            for name, actual, reference in zip(
                ("combined", "normalized", "lora", "injection", "output"),
                observed,
                expected,
                strict=True,
            ):
                assert torch.isfinite(actual).all(), (rank, name)
                errors[f"rank{rank}/{name}"] = float(
                    (actual.float() - reference.float()).abs().max()
                )
                torch.testing.assert_close(actual, reference, atol=0.002, rtol=0.002)
            assert torch.isnan(data["output"][width:]).all()
            assert (data["epochs"].cpu() == expected_generation).all()
        record = {"width": width, "scale": scale, "max_errors": errors}
        if trace and width == 1:
            record["cta_globaltimer_ns"] = [d["trace"][0].cpu().tolist() for d in local]
        records.append(record)
    # This is candidate-only timing, not speedup over the production path.
    for data in local:
        data["trace"] = None
    timing_graphs = capture(module, local, 1, eps, steps=100)
    times = []
    for _ in range(5):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        with torch.cuda.device(0):
            start.record()
        replay(timing_graphs)
        with torch.cuda.device(0):
            end.record()
        synchronize(tp)
        times.append(start.elapsed_time(end) * 10)
    return {
        "tp": tp,
        "hidden": hidden,
        "hc": hc,
        "low_rank": low_rank,
        "correctness": records,
        "candidate_c1_us": times,
        "candidate_c1_median_us": statistics.median(times),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--small-only", action="store_true")
    parser.add_argument("--real-only", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "sm70_hc_dataflow_research", args.module
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    shapes = [(1, 64, 4, 32), (2, 64, 4, 32), (4, 64, 4, 32)]
    if not args.small_only:
        shapes.append((4, 2560, 4, 320))
    if args.real_only:
        shapes = [(4, 2560, 4, 320)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with Path("/tmp/gpu0-3.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # Waiting for shared GPUs does not consume the kernel watchdog.
        signal.alarm(args.timeout_seconds)
        result = {"research_only": True, "complete": False, "cases": []}
        for shape in shapes:
            result["cases"].append(run_case(module, *shape, trace=args.trace))
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(
                "CASE", shape, result["cases"][-1]["candidate_c1_median_us"], flush=True
            )
        result["complete"] = True
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        signal.alarm(0)


if __name__ == "__main__":
    main()
