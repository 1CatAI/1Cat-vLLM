# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check compensated q8 schedules and time a sixteen-layer KV working set."""

import argparse
import hashlib
import importlib.util
import json
import statistics
from pathlib import Path

import torch


def load_operator(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    path = Path(manifest["library"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["library_sha256"]
    spec = importlib.util.spec_from_file_location(path.name.split(".")[0], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path.resolve():
        raise RuntimeError(
            f"Extension module alias: requested {path}, loaded {module.__file__}. "
            "Build candidates with distinct native module names."
        )
    return module.run, manifest


def make_case(rows, page, length, num_arms, stride_padding=0):
    pages = (length + page - 1) // page
    raw = torch.randn((pages, 2, page, 1, 256), device="cuda", dtype=torch.float16)
    raw[:, 1].add_(torch.linspace(-4, 4, 256, device="cuda"))
    encoded = raw.to(torch.float8_e4m3fn).view(torch.uint8)
    backing = torch.empty(
        (pages, 2, page, 1, 256 + stride_padding), device="cuda", dtype=torch.uint8
    )
    backing[..., :256].copy_(encoded)
    k, v = backing[..., :256].unbind(1)
    table = torch.randperm(pages, device="cuda").int()[None].contiguous()
    q = torch.randn((rows, 6, 256), device="cuda", dtype=torch.float16) * 0.5
    lengths = torch.arange(
        length - rows + 1, length + 1, device="cuda", dtype=torch.int32
    )
    arms = []
    for _ in range(num_arms):
        guard = torch.full(
            (rows + 2, 6, 256), -777.0, device="cuda", dtype=torch.float16
        )
        arms.append(
            {
                "guard": guard,
                "out": guard[1:-1],
                "partial": torch.full(
                    (80, 8, 6, 256), -777.0, device="cuda", dtype=torch.float32
                ),
                "lse": torch.full(
                    (80, 8, 6, 2), -777.0, device="cuda", dtype=torch.float32
                ),
            }
        )
    return {
        "q": q,
        "k": k,
        "v": v,
        "table": table,
        "lengths": lengths,
        "initial_lengths": lengths.clone(),
        "arms": arms,
    }


def call(case, arm, operator):
    buffers = case["arms"][arm]
    operator(
        case["q"],
        case["k"],
        case["v"],
        buffers["out"],
        case["table"],
        case["lengths"],
        buffers["partial"],
        buffers["lse"],
        0.0625,
        0.5,
        1.25,
    )


def byte_equal(a, b):
    return torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", action="append", type=Path, default=[])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--short", action="store_true")
    parser.add_argument("--sanitizer", action="store_true")
    parser.add_argument("--profile-one", action="store_true")
    parser.add_argument("--performance-page", type=int, default=1648)
    args = parser.parse_args()
    if not args.candidate and not args.profile_one:
        parser.error("At least one --candidate is required for a comparison")
    if args.performance_page <= 0:
        parser.error("--performance-page must be positive")
    assert not args.output.exists(), args.output
    assert torch.cuda.get_device_capability() == (7, 0)
    torch.manual_seed(20260909)
    loaded = [load_operator(p) for p in [args.baseline, *args.candidate]]
    operators = [item[0] for item in loaded]
    if len({id(operator) for operator in operators}) != len(operators):
        raise RuntimeError("Candidate operators share a Python function binding")
    report = {
        "manifests": [item[1] for item in loaded],
        "device": torch.cuda.get_device_name(),
        "scope": "Operator screen, not model or complete-round admission",
        "checks": [],
        "performance": [],
        "complete": False,
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    if args.profile_one:
        case = make_case(8, 1648, 131072, len(operators))
        call(case, 0, operators[0])
        torch.cuda.synchronize()
        report["profile_input"] = {"rows": 8, "page": 1648, "length": 131072}
        report["complete"] = True
        save()
        return

    cases = [
        (8, 3296, 63, 0),
        (8, 3296, 65, 0),
        (8, 3296, 129, 0),
        (8, 1648, 1649, 0),
        (8, 3296, 3297, 0),
        (2, 848, 8197, 8),
        (8, 1648, 3297, 8),
    ]
    if not args.short:
        cases += [(5, 1648, 32768, 0), (8, 1648, 65536, 0), (8, 3296, 131072, 0)]
    if args.sanitizer:
        cases = [(8, 1648, 1649, 0), (8, 1648, 3297, 8)]
    for rows, page, length, padding in cases:
        case = make_case(rows, page, length, len(operators), padding)
        graphs = []
        for arm, operator in enumerate(operators):
            call(case, arm, operator)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                call(case, arm, operator)
            graphs.append(graph)
        for state in ("live", "all_zero", "tail_zero", "first_zero", "restore"):
            case["lengths"].copy_(case["initial_lengths"])
            if state == "all_zero":
                case["lengths"].zero_()
            elif state == "tail_zero":
                case["lengths"][-1] = 0
            elif state == "first_zero":
                case["lengths"][0] = 0
            for buffers, graph in zip(case["arms"], graphs):
                buffers["partial"].fill_(-777.0)
                buffers["lse"].fill_(-777.0)
                graph.replay()
            for arm in range(1, len(operators)):
                checks = {
                    key: byte_equal(case["arms"][0][key], case["arms"][arm][key])
                    for key in ("out", "partial", "lse")
                }
                guards = all(
                    bool((buffers["guard"][[0, -1]] == -777).all())
                    for buffers in case["arms"]
                )
                row = {
                    "rows": rows,
                    "page": page,
                    "length": length,
                    "stride_padding": padding,
                    "state": state,
                    "arm": arm,
                    "byte_exact": checks,
                    "guards": guards,
                }
                report["checks"].append(row)
                if not all(checks.values()) or not guards:
                    save()
                    raise AssertionError(row)
        print(json.dumps({"exact_shape": [rows, page, length, padding]}), flush=True)
        del case, graphs
    save()

    if not args.correctness_only:
        for length in (1024, 32768, 65536, 131072):
            # Sixteen distinct KV allocations represent the target's real
            # layer working set and prevent single-layer hot-cache claims.
            cases = [
                make_case(8, args.performance_page, length, len(operators))
                for _ in range(16)
            ]
            graphs = []
            for arm, operator in enumerate(operators):
                for case in cases:
                    call(case, arm, operator)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for case in cases:
                        call(case, arm, operator)
                graphs.append(graph)
            for _ in range(5):
                for graph in graphs:
                    graph.replay()
            samples = [[] for _ in operators]
            for trial in range(5):
                order = list(range(len(operators)))
                if trial % 2:
                    order.reverse()
                for arm in order:
                    start, end = [
                        torch.cuda.Event(enable_timing=True) for _ in range(2)
                    ]
                    start.record()
                    for _ in range(8):
                        graphs[arm].replay()
                    end.record()
                    end.synchronize()
                    samples[arm].append(start.elapsed_time(end) / 8)
            row = {
                "context": length,
                "page": args.performance_page,
                "layers": 16,
                "samples_ms": samples,
                "median_ms": [statistics.median(values) for values in samples],
                "complete_round_performance": False,
            }
            report["performance"].append(row)
            save()
            print(json.dumps(row), flush=True)
            del cases, graphs
    report["complete"] = True
    save()


if __name__ == "__main__":
    main()
