# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Research-only expert W13/SiLU/W2/weighted-reduce dataflow screen.

No router, shared expert or TP communication is included yet. This screen
establishes local correctness/progress before any production integration.
"""

import argparse
import fcntl
import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-12.8")
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0"
os.environ["MAX_JOBS"] = "2"

import torch
from torch.utils.cpp_extension import load


def oracle(x, ids, weights, w13, w2):
    x, w13, w2 = (t.cpu().double() for t in (x, w13, w2))
    ids, weights = ids.cpu(), weights.cpu().double()
    result = torch.zeros_like(x)
    for m in range(x.shape[0]):
        for slot in range(ids.shape[1]):
            e = ids[m, slot].item()
            if e < 0 or e >= w13.shape[0]:
                continue
            up_gate = (w13[e] @ x[m]).half().double()
            gate, up = up_gate[0::2], up_gate[1::2]
            inter = (
                (torch.nn.functional.silu(gate).half().double() * up).half().double()
            )
            down = (w2[e] @ inter).half().double()
            result[m] += down * weights[m, slot]
    return result.half()


def nvfp4_weight(experts, n, k):
    # Independent CPU layout/decoder, not the production preparation operator.
    codes = torch.randint(0, 16, (experts, n, k), dtype=torch.int64)
    tm_codes = codes.reshape(experts, n, k // 8, 8)[..., [0, 2, 4, 6, 1, 3, 5, 7]]
    packed = (tm_codes << (torch.arange(8) * 4)).sum(dim=-1)
    packed = (
        packed.reshape(experts, n // 32, 32, k // 8)
        .permute(0, 1, 3, 2)
        .contiguous()
        .int()
        .cuda()
    )
    scales = torch.full((experts, k // 16, n), 1 / 16, dtype=torch.float16).cuda()
    table = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float64)
    decoded = table[codes & 7] * torch.where(codes & 8 == 0, 1, -1) / 16
    return packed, scales, decoded.half()


def check(ops, quant="fp16"):
    rows = []
    # Generic shapes, selected-expert duplicates, invalid routes, M=5,
    # replay changes and poisoned workspaces test the compute and task flags.
    for m, h, i, experts, k in [
        (1, 32, 16, 3, 2),
        (5, 96, 48, 7, 5),
        (7, 64, 32, 3, 4),
        (1, 2560, 160, 11, 10),
    ]:
        torch.manual_seed(9703 + m + h)
        x = (torch.randint(-2, 3, (m, h), device="cuda") / 16).half()
        w13 = (torch.randint(-2, 3, (experts, 2 * i, h), device="cuda") / 16).half()
        w2 = (torch.randint(-2, 3, (experts, h, i), device="cuda") / 16).half()
        if quant == "nvfp4":
            packed13, scales13, w13 = nvfp4_weight(experts, 2 * i, h)
            packed2, scales2, w2 = nvfp4_weight(experts, h, i)
        ids = torch.randint(0, experts, (m, k), device="cuda", dtype=torch.int32)
        weights = torch.full((m, k), 1 / k, device="cuda", dtype=torch.float32)
        if m == 7:
            ids[0, 0] = -1
            ids[-1, -1] = experts
        partial = torch.empty(
            (m * k * (i // 16), h), device="cuda", dtype=torch.float32
        )
        out = torch.empty_like(x)
        blocks = torch.cuda.get_device_properties(0).multi_processor_count
        control = torch.zeros(
            5 + 32 * blocks + m * k * (i // 16), device="cuda", dtype=torch.int32
        )
        stream = torch.cuda.Stream()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            if quant == "fp16":
                ops.fp16(x, ids, weights, w13, w2, partial, out, control)
            else:
                ops.nvfp4(
                    x,
                    ids,
                    weights,
                    packed13,
                    scales13,
                    packed2,
                    scales2,
                    partial,
                    out,
                    control,
                    i,
                    experts,
                )
        max_error = 0
        for scale in (0, 0.5, 1, 2, 0.25, 1):
            x.copy_((torch.randint(-2, 3, (m, h), device="cuda") / 16 * scale).half())
            out.fill_(float("nan"))
            partial.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            expected = oracle(x, ids, weights, w13, w2)
            actual = out.cpu()
            assert torch.isfinite(actual).all()
            delta = (actual.float() - expected.float()).abs().max().item()
            max_error = max(max_error, delta)
            torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-3)
            assert control[0].item() == control[4].item()
        assert control[4].item() == 6
        row = dict(
            reader=quant,
            M=m,
            hidden=h,
            intermediate=i,
            experts=experts,
            topk=k,
            replays=6,
            fp64_oracle_max_abs=max_error,
            generation=6,
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def check_width_changes(ops):
    rows = []
    for quant in ("fp16", "nvfp4"):
        h, i, experts, k, maximum = 96, 48, 7, 5, 7
        x = torch.empty((maximum, h), device="cuda", dtype=torch.float16)
        ids = torch.empty((maximum, k), device="cuda", dtype=torch.int32)
        weights = torch.full((maximum, k), 1 / k, device="cuda")
        if quant == "fp16":
            w13 = (torch.randint(-2, 3, (experts, 2 * i, h), device="cuda") / 16).half()
            w2 = (torch.randint(-2, 3, (experts, h, i), device="cuda") / 16).half()
        else:
            p13, s13, w13 = nvfp4_weight(experts, 2 * i, h)
            p2, s2, w2 = nvfp4_weight(experts, h, i)
        output = torch.empty_like(x)
        scratch = torch.empty(
            (maximum * k * (i // 16), h), device="cuda", dtype=torch.float32
        )
        blocks = torch.cuda.get_device_properties(0).multi_processor_count
        control = torch.zeros(
            5 + 32 * blocks + maximum * k * (i // 16), device="cuda", dtype=torch.int32
        )
        stream = torch.cuda.Stream()
        graphs = {}
        torch.cuda.synchronize()
        for m in (1, 5, 7):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                if quant == "fp16":
                    ops.fp16(
                        x[:m],
                        ids[:m],
                        weights[:m],
                        w13,
                        w2,
                        scratch,
                        output[:m],
                        control,
                    )
                else:
                    ops.nvfp4(
                        x[:m],
                        ids[:m],
                        weights[:m],
                        p13,
                        s13,
                        p2,
                        s2,
                        scratch,
                        output[:m],
                        control,
                        i,
                        experts,
                    )
            graphs[m] = graph
        widths = (1, 5, 7, 5, 1, 7, 1, 5)
        for replay, m in enumerate(widths, 1):
            x.copy_((torch.randint(-2, 3, x.shape, device="cuda") / 16).half())
            ids.copy_(
                torch.randint(0, experts, ids.shape, device="cuda", dtype=torch.int32)
            )
            output.fill_(float("nan"))
            scratch.fill_(float("nan"))
            graphs[m].replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                output[:m].cpu(),
                oracle(x[:m], ids[:m], weights[:m], w13, w2),
                atol=2e-4,
                rtol=2e-3,
            )
            assert torch.isnan(output[m:]).all()
            assert control[4].item() == replay
        rows.append(
            dict(
                reader=quant,
                widths=list(widths),
                shared_workspace=True,
                changed_inputs_and_routes=True,
                generations=8,
            )
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    args.build.mkdir(parents=True, exist_ok=True)
    ops = load(
        name="sm70_moe_dataflow_research",
        sources=[str(Path(__file__).with_name("sm70_moe_dataflow_research.cu"))],
        build_directory=str(args.build),
        extra_cuda_cflags=["-O3", "-lineinfo", "--ptxas-options=-v"],
        verbose=True,
    )
    if args.build_only:
        return
    with open("/tmp/gpu0-3.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        live = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                "0,1,2,3",
                "--query-compute-apps=pid",
                "--format=csv,noheader",
            ],
            text=True,
        )
        if live.strip():
            raise SystemExit("GPU busy; no research screen started")
        assert torch.cuda.get_device_capability() == (7, 0)
        report = dict(
            research_only=True,
            includes_router=False,
            includes_shared=False,
            includes_communication=False,
            fp16_oracles=check(ops),
            nvfp4_oracles=check(ops, "nvfp4"),
            shared_workspace_replays=check_width_changes(ops),
        )
        if args.out:
            args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
