# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP4 HC down/up/mix + both gathers; NOT model throughput.

Run with torchrun --standalone --nproc-per-node=4 on exclusively owned cards.
By default build this tree's research extensions with dedicated IPC channels.
With --runtime, use only the ordinary source-built _C extension, runtime
communicator and loader packing; no research extension is built or loaded.
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


def candidate(
    ext, gather, peers, state, weight, rank, communicator=None, fused_chain=False
):
    for (x, _, _, _, _, scratch, lora, local, output, injection), (_, _, d, u) in zip(
        state, weight
    ):
        if communicator is not None:
            communicator.sm70_qwen38_hc_batch(
                x,
                d,
                u,
                scratch,
                lora,
                local,
                output,
                injection,
                fused_chain=fused_chain,
            )
            continue
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


def save_first_failure(state, names, path):
    for name, tensors in zip(names, state):
        x, y, lora, gate, out, partials, actual_lora, local, actual, injection = tensors
        if not any(check([tensors])):
            continue
        torch.save(
            {
                "weight_name": name,
                "tensors": {
                    key: value.detach().cpu()
                    for key, value in zip(
                        (
                            "input",
                            "down_ref",
                            "lora_ref",
                            "gate_ref",
                            "output_ref",
                            "partials",
                            "lora_actual",
                            "local",
                            "output_actual",
                            "injection_actual",
                        ),
                        (
                            x,
                            y,
                            lora,
                            gate,
                            out,
                            partials,
                            actual_lora,
                            local,
                            actual,
                            injection,
                        ),
                    )
                },
            },
            path,
        )
        break


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pairs", type=int, default=8)
    p.add_argument("--rows", default="2,4,8,16")
    p.add_argument("--build-only", action="store_true")
    p.add_argument("--runtime", action="store_true")
    p.add_argument("--fused-chain", action="store_true")
    p.add_argument(
        "--trace", action="store_true", help="Capture four graph replays per arm/width"
    )
    a = p.parse_args()
    if not 1 <= a.pairs <= 96 or (not a.build_only and a.model is None):
        p.error("Use 1..96 HC pairs and specify --model")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    if a.runtime and a.build_only:
        p.error("Build --runtime with the ordinary source build, not this JIT helper")
    if a.fused_chain and not a.runtime:
        p.error("--fused-chain requires the ordinary native --runtime")
    ext, gather = (None, None) if a.runtime else (build(), build_gather())
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
    communicator = None
    # Separate down/output channels also isolate epoch state across payload
    # sizes. There is no alias with an engine's auxiliary-stream collectives.
    for _ in range(0 if a.runtime else 2):
        pointer, handle = gather.allocate()
        handles = [None] * 4
        dist.all_gather_object(handles, handle)
        peers.append(
            [pointer if i == rank else gather.open(h) for i, h in enumerate(handles)]
        )
    names, weight = weights(a.model, a.pairs, rank)
    native_files = []
    if a.runtime:
        import vllm._C

        from vllm import _custom_ops as ops
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )
        from vllm.models.qwen4_exp.nvidia.sm70_fp16_hc import _pack_hc_batch_weight

        if not ops.supports_sm70_qwen38_hc_batch():
            raise RuntimeError("Rebuild this worktree's native batched HC extension")
        native_files = [Path(vllm._C.__file__)]
        if not native_files[0].resolve().is_relative_to(ROOT.resolve()):
            raise RuntimeError("Runtime benchmark must use this worktree's _C")
        weight = [
            (
                d,
                u,
                _pack_hc_batch_weight(d, "down", rank),
                _pack_hc_batch_weight(u, "up", rank),
            )
            for d, u, _, _ in weight
        ]
        communicator = CustomAllreduce(dist.group.WORLD, device=rank)
        probe = torch.empty((2, 10240), device="cuda", dtype=torch.float16)
        if not communicator.can_sm70_qwen38_hc_batch(probe):
            raise RuntimeError("Native TP4 batched HC was not admitted")
        torch.cuda.synchronize()
        dist.barrier()  # All local packet clears must finish before peer sends.
    result = {
        "contract": (
            "TP4 down/Silu/up/mix/both gathers; excludes combine/norm and model"
        ),
        "source": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "runtime": a.runtime,
        "fused_chain": a.fused_chain,
        "timing_arms": (
            ["native-four-kernel", "native-coalesced-fused-three-kernel"]
            if a.fused_chain
            else ["torch-reference", "candidate"]
        ),
        "sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                [
                    ROOT / "csrc/sm70_qwen38_hc_batch.cuh",
                    ROOT / "csrc/custom_all_reduce.cuh",
                    ROOT / "csrc/custom_all_reduce.cu",
                ]
                if a.runtime
                else [
                    ROOT / "benchmarks/csrc/benchmark_sm70_hc_batch_reuse.cu",
                    ROOT / "benchmarks/csrc/benchmark_sm70_hc_batch_gather.cu",
                ]
            )
        },
        "extensions_sha256": [
            hashlib.sha256(Path(e.__file__).read_bytes()).hexdigest()
            for e in (ext, gather)
            if e is not None
        ],
        "native_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in native_files
        },
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
    graphs = {}
    split_graphs = {}
    for rows in map(int, a.rows.split(",")):
        if not 2 <= rows <= 16:
            p.error("HC batch supports M2..16")
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
        cg = capture(
            partial(
                candidate,
                ext,
                gather,
                peers,
                state,
                weight,
                rank,
                communicator,
                a.fused_chain,
            )
        )
        graphs[rows] = (bg, cg, state)
        if a.fused_chain:
            split_graphs[rows] = capture(
                partial(
                    candidate,
                    ext,
                    gather,
                    peers,
                    state,
                    weight,
                    rank,
                    communicator,
                    False,
                )
            )
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
            if any(errors[rank]):
                save_first_failure(
                    state,
                    names,
                    a.out.with_suffix(f".m{rows}.scale{scale}.rank{rank}.failure.pt"),
                )
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
                    latency = time_graph(
                        (split_graphs.get(rows, bg), cg)[arm], len(weight)
                    )
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
    # Reuse the SAME captured graphs across shrinking/growing payloads, with
    # odd replay counts. Fresh captures or only-even epochs can mask stale tags.
    transitions = []
    for rows in (2, 16, 3, 2, 8, 2, 16, 4, 2):
        if rows not in graphs:
            continue
        bg, cg, state = graphs[rows]
        for tensors in state:
            tensors[0].normal_(0, 0.1)
            for output in tensors[5:]:
                output.fill_(float("nan"))
        bg.replay()
        paths = [("candidate", cg)]
        if rows in split_graphs:
            paths += [("native-split", split_graphs[rows]), ("candidate", cg)]
        for label, graph in paths:
            for output in state:
                for tensor in output[5:]:
                    tensor.fill_(float("nan"))
            graph.replay()
            # A graph containing 96 HC pairs advances each epoch an EVEN
            # number of times even when replayed once. Execute one extra
            # pair to really flip active counters before the next width/path.
            candidate(
                ext,
                gather,
                peers,
                state[:1],
                weight[:1],
                rank,
                communicator,
                a.fused_chain and label == "candidate",
            )
            errors = [None] * 4
            dist.all_gather_object(errors, check(state))
            transitions.append(
                {
                    "rows": rows,
                    "path": label,
                    "extra_pair": True,
                    "rank_mismatches": errors,
                }
            )
    result["graph_transitions"] = transitions
    if a.trace:
        # Capture only warmed graphs. Timings above exclude these captured
        # replays; neither is an extrapolated model-token measurement.
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 0:
            torch.cuda.cudart().cudaProfilerStart()
        for rows, (bg, cg, _) in graphs.items():
            for label, graph in (
                ("control", split_graphs.get(rows, bg)),
                ("candidate", cg),
            ):
                dist.barrier()
                with torch.cuda.nvtx.range(f"hc-m{rows}-{label}-rank{rank}"):
                    for _ in range(4):
                        graph.replay()
                torch.cuda.synchronize()
        dist.barrier()
        if rank == 0:
            torch.cuda.cudart().cudaProfilerStop()
    torch.cuda.synchronize()
    dist.barrier()
    for channel in peers:
        for i, pointer in enumerate(channel):
            if i != rank:
                gather.close(pointer)
    dist.barrier()
    for channel in peers:
        gather.free(channel[rank])
    if communicator is not None:
        communicator.close()
    dist.destroy_process_group()
    failed = any(any(e) for t in transitions for e in t["rank_mismatches"]) or any(
        any(any(e) for e in c["rank_lora_output_injection_mismatches"])
        for case in result["cases"]
        for c in case["checks"]
    )
    result["measurements_complete"] = True
    result["complete"] = result["bitwise_passed"] = not failed
    if rank == 0:
        a.out.write_text(json.dumps(result, indent=2) + "\n")
    if failed:
        raise SystemExit("HC numerical gate failed; no runtime promotion")


if __name__ == "__main__":
    main()
