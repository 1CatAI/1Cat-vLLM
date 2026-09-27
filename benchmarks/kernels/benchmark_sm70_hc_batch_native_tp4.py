# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Research TP4 HC down/up/mix + both gathers; NOT model throughput.

Run with torchrun --standalone --nproc-per-node=4 on exclusively owned cards.
Each process builds/loads only this tree's research extensions. IPC channels
are dedicated to this screen and are not shared with any runtime communicator.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from benchmark_sm70_hc_batch_reuse import (
    ROOT,
    build,
    capture,
    mix_reference,
    pack_down_weight,
    pack_weight,
    silu_reference,
    time_graph,
)
from safetensors import safe_open
from torch.utils.cpp_extension import load


def build_gather():
    return load(
        name="sm70_hc_batch_gather_screen",
        sources=[str(ROOT / "benchmarks/csrc/benchmark_sm70_hc_batch_gather.cu")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo", "--ptxas-options=-v"],
        verbose=True,
    )


def weights(model, pairs, rank):
    index = json.loads((model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    suffix = ".input_mix_weight_down.weight"
    names = sorted(
        n
        for n in index
        if n.startswith("model.language_model.layers.") and n.endswith(suffix)
    )[:pairs]
    if len(names) != pairs:
        raise ValueError(f"Expected {pairs} HC pairs, got {len(names)}")
    result = []
    for name in names:
        tensors = []
        prefix = name[: -len(suffix)]
        for part in (
            "input_mix_weight_down",
            "block_inject_weight",
            "input_mix_weight_up",
        ):
            key = f"{prefix}.{part}.weight"
            with safe_open(model / index[key], framework="pt") as f:
                tensors.append(f.get_tensor(key).half().cuda())
        d, injection, u = tensors
        d = torch.cat((d, injection, d.new_zeros(12, 10240)))
        down = pack_down_weight(d[rank * 80 : rank * 80 + 88].contiguous())
        up = pack_weight(
            u.view(4, 2560, 320)[:, rank * 640 : (rank + 1) * 640].contiguous()
        )
        result.append((d, u, down, up))
    return names, result


def reference(state, weight, rows):
    for (x, y, lora, gate, out, *_), (d, u, _, _) in zip(state, weight):
        torch.mm(x, d.t(), out=y)
        silu_reference[(rows,)](y, lora, num_warps=4)
        torch.mm(lora, u.t(), out=gate)
        mix_reference[(rows, 5)](x, gate, out, 2560, 0, num_warps=4)


def candidate(ext, gather, peers, state, weight, rank):
    for (x, _, _, _, _, scratch, lora, local, output, injection), (_, _, d, u) in zip(
        state, weight
    ):
        ext.run_down_shard(x, d, scratch)
        gather.run(peers[0], rank, scratch, lora, injection, True)
        ext.run(lora, u, x, local, rank * 640, False, 1, 4, True)
        gather.run(peers[1], rank, local, output, injection, False)


def check(state):
    errors = [0, 0, 0]
    for _, y, lora, _, out, _, actual_lora, _, actual_out, injection in state:
        for i, (a, b) in enumerate(
            ((actual_lora, lora), (actual_out, out), (injection, y[:, 320:324]))
        ):
            errors[i] += int((a.view(torch.int16) != b.view(torch.int16)).sum())
    return errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pairs", type=int, default=8)
    p.add_argument("--rows", default="2,4,8,16")
    p.add_argument("--build-only", action="store_true")
    a = p.parse_args()
    if not 1 <= a.pairs <= 96 or (not a.build_only and a.model is None):
        p.error("Use 1..96 HC pairs and specify --model")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    ext, gather = build(), build_gather()
    if a.build_only:
        assert not torch.cuda.is_initialized()
        print("Built both research extensions without a CUDA context", flush=True)
        return
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    if torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("Requires SM70")
    dist.init_process_group("gloo")
    if dist.get_world_size() != 4:
        raise RuntimeError("Requires exactly TP4")
    owned = [None] * 4
    dist.all_gather_object(owned, os.getpid())
    if rank == 0:
        actual = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                os.environ["CUDA_VISIBLE_DEVICES"],
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        unexpected = sorted({int(v) for v in actual.split()} - set(owned))
    else:
        unexpected = None
    status = [unexpected]
    dist.broadcast_object_list(status, src=0)
    if status[0]:
        raise RuntimeError(f"Cards not exclusive: {status[0]}")
    peers = []
    # Separate down/output channels also isolate epoch state across payload
    # sizes. There is no alias with an engine's auxiliary-stream collectives.
    for _ in range(2):
        pointer, handle = gather.allocate()
        handles = [None] * 4
        dist.all_gather_object(handles, handle)
        peers.append(
            [pointer if i == rank else gather.open(h) for i, h in enumerate(handles)]
        )
    names, weight = weights(a.model, a.pairs, rank)
    result = {
        "contract": (
            "TP4 down/Silu/up/mix/both gathers; excludes combine/norm and model"
        ),
        "source": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                ROOT / "benchmarks/csrc/benchmark_sm70_hc_batch_reuse.cu",
                ROOT / "benchmarks/csrc/benchmark_sm70_hc_batch_gather.cu",
            )
        },
        "extensions_sha256": [
            hashlib.sha256(Path(e.__file__).read_bytes()).hexdigest()
            for e in (ext, gather)
        ],
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "physical_gpus": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "weights": names,
        "reduced_precision_reduction": False,
        "fp16_accumulation": False,
        "complete": False,
        "cases": [],
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for rows in map(int, a.rows.split(",")):
        torch.manual_seed(20260927 + rows)
        state = []
        for _ in weight:
            x = torch.randn(rows, 10240, device="cuda", dtype=torch.half)
            state.append(
                (
                    x,
                    x.new_empty(rows, 336),
                    x.new_empty(rows, 320),
                    x.new_empty(rows, 10240),
                    x.new_empty(rows, 2560),
                    torch.empty(20, rows, 96, device="cuda", dtype=torch.float32),
                    x.new_empty(rows, 320),
                    x.new_empty(rows, 640),
                    x.new_empty(rows, 2560),
                    x.new_empty(rows, 4),
                )
            )
        bg = capture(partial(reference, state, weight, rows))
        cg = capture(partial(candidate, ext, gather, peers, state, weight, rank))
        checks = []
        for scale in (0.0, 0.001, 0.03, 0.1, 1.0, 3.0):
            for tensors in state:
                tensors[0].normal_(0, scale)
                for output in tensors[5:]:
                    output.fill_(float("nan"))
            bg.replay()
            cg.replay()
            errors = [None] * 4
            dist.all_gather_object(errors, check(state))
            checks.append(
                {"scale": scale, "rank_lora_output_injection_mismatches": errors}
            )
        if any(
            any(any(e) for e in c["rank_lora_output_injection_mismatches"])
            for c in checks
        ):
            record = {"rows": rows, "checks": checks, "samples_us": None}
        else:
            samples = []
            for trial in range(6):
                pair = [None, None]
                for arm in (0, 1) if trial % 2 == 0 else (1, 0):
                    dist.barrier()
                    latency = time_graph((bg, cg)[arm], len(weight))
                    times = [None] * 4
                    dist.all_gather_object(times, latency)
                    pair[arm] = times
                samples.append(pair)
            record = {
                "rows": rows,
                "checks": checks,
                "samples_us": samples,
                "rank_max_baseline_candidate_us": [
                    statistics.median(max(s[arm]) for s in samples) for arm in (0, 1)
                ],
            }
        result["cases"].append(record)
        if rank == 0:
            a.out.write_text(json.dumps(result, indent=2) + "\n")
            print(
                json.dumps({k: v for k, v in record.items() if k != "samples_us"}),
                flush=True,
            )
    torch.cuda.synchronize()
    dist.barrier()
    for channel in peers:
        for i, pointer in enumerate(channel):
            if i != rank:
                gather.close(pointer)
    dist.barrier()
    for channel in peers:
        gather.free(channel[rank])
    dist.destroy_process_group()
    result["complete"] = True
    if rank == 0:
        a.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
