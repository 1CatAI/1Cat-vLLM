# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load real original experts through TP4 shared storage, without a model forward.

This measures the actual cold object, raw GPU carriers and canonical workspace.
It checks selected original rows before and after M6 swaps. It does not load
dense/Engram/DSpark weights or establish a complete-model memory budget.
"""

import argparse
import datetime
import json
import multiprocessing as mp
import os
import socket
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import gguf
import numpy as np
import torch
import torch.distributed as dist

from .shared_host_probe import cgroup_memory, memory_available


class _Group:
    world_size = 4

    def __init__(self, rank):
        self.rank_in_group, self.cpu_group = rank, dist.group.WORLD

    def broadcast_object(self, obj, src):
        values = [obj]
        dist.broadcast_object_list(values, src=src)
        return values[0]


def _pss():
    selected, total = False, 0
    for line in Path("/proc/self/smaps").read_text().splitlines():
        if line and line[0] in "0123456789abcdef" and "-" in line.split()[0]:
            selected = "memfd:vllm-shared-host" in line
        elif selected and line.startswith("Pss:"):
            total += int(line.split()[1]) * 1024
    return total


def _run(rank, root, layers, offload):
    from vllm import _custom_ops  # noqa: F401
    from vllm.config.kernel import Sm70GgufConfig
    from vllm.model_executor.layers.quantization.gguf_shared_runtime import SHARDS
    from vllm.model_executor.layers.quantization.gguf_shared_storage import (
        SharedOriginalExpertCache,
    )

    tensors = {}
    for path in sorted((root / "gguf").glob("*.gguf")):
        reader = gguf.GGUFReader(path, mode="r")
        for tensor in reader.tensors:
            if "_exps.weight" in tensor.name:
                tensors[tensor.name] = tensor
    names = dict(w1="gate", w3="up", w2="down")
    sources = {
        (i, shard): tensors[f"blk.{i}.ffn_{names[shard]}_exps.weight"]
        for i in range(layers)
        for shard in SHARDS
    }
    method = SimpleNamespace(
        is_v41=True, hidden_size=5120, intermediate_size=576, num_experts=384
    )
    layer = SimpleNamespace(tp_size=4, tp_rank=rank)
    policy = Sm70GgufConfig(
        expert_cache_host_storage="shared_original",
        expert_cache_offload_fraction=offload,
        expert_cache_host_slack=0,
        expert_storage="original",
        expert_arena=True,
        expert_load_prefetch_banks=1,
        expert_cache_prefill_tokens=512,
    )
    entries = [
        (i, layer, method, {s: int(sources[i, s].tensor_type) for s in SHARDS})
        for i in range(layers)
    ]
    started = time.monotonic()
    cache = SharedOriginalExpertCache(
        entries, policy, torch.device("cuda", rank), _Group(rank)
    )
    try:

        def check(i, shard, expert, slot):
            geometry = cache.specs[i].layouts(rank)[SHARDS.index(shard)]
            rows = np.array([0, geometry.device_rows // 2, geometry.device_rows - 1])
            byte = geometry.first_block * geometry.block_bytes
            expected = sources[i, shard].data[
                expert,
                rows + geometry.source_row_start,
                byte : byte + geometry.device_row_bytes,
            ]
            actual = (
                cache.bank(i, shard)[slot, torch.from_numpy(rows).to(rank)]
                .cpu()
                .numpy()
            )
            np.testing.assert_array_equal(actual, expected)

        for i in range(layers):
            for shard in SHARDS:
                cache.store_original_bank(i, shard, sources[i, shard])
                for expert in (0, 134, 268):
                    check(i, shard, expert, expert)
            if rank == 0:
                print(
                    json.dumps(
                        dict(loaded_layers=i + 1, seconds=time.monotonic() - started)
                    ),
                    flush=True,
                )
        loaded_seconds = time.monotonic() - started
        before = torch.accelerator.memory_allocated(rank)
        for i in range(layers):
            first = cache.budget.hot_experts
            requested = torch.arange(first, first + 36, dtype=torch.int32, device=rank)
            slots = cache.resolve(i, requested)
            torch.accelerator.synchronize(rank)
            for expert, slot in zip(requested.cpu().tolist(), slots.cpu().tolist()):
                for shard in SHARDS:
                    check(i, shard, expert, slot)
        dist.barrier()
        report = dict(
            rank=rank,
            passed=True,
            pid=os.getpid(),
            identity=vars(cache.data.identity),
            cold_bytes=cache.budget.cold_bytes,
            offload_fraction=offload,
            cold_experts=cache.budget.cold_experts,
            gpu_slots=cache.budget.gpu_slots,
            raw_gpu_bytes=cache.arena.numel(),
            canonical_workspace_bytes=cache.tm.storage_bytes(),
            torch_allocated_bytes=before,
            torch_peak_bytes=torch.accelerator.max_memory_allocated(rank),
            shared_object_pss_bytes=_pss(),
            memory_available_bytes=memory_available(),
            cgroup=cgroup_memory(),
            loaded_seconds=loaded_seconds,
            checked_rows_per_sample=3,
            loaded_sample_experts_per_layer=3,
            swapped_experts_per_layer=36,
        )
        reports = [None] * 4
        dist.all_gather_object(reports, report)
        if (
            len({(r["identity"]["device"], r["identity"]["inode"]) for r in reports})
            != 1
        ):
            raise AssertionError("TP workers did not use one shared object")
        if min(r["memory_available_bytes"] for r in reports) < 8 * 1024**3:
            raise MemoryError("Measured available host memory fell below 8 GiB")
        report["group_pss_bytes"] = sum(r["shared_object_pss_bytes"] for r in reports)
        return report
    finally:
        torch.accelerator.synchronize(rank)
        dist.barrier()
        cache.close()


def _consumer(rank, connection, root, directory, layers, offload):
    try:
        torch.set_num_threads(1)
        torch.accelerator.set_device_index(rank)
        dist.init_process_group(
            "gloo",
            init_method=f"file://{directory}/rendezvous",
            rank=rank,
            world_size=4,
            timeout=datetime.timedelta(seconds=90),
        )
        connection.send(json.dumps(_run(rank, root, layers, offload)).encode())
    except BaseException:
        connection.send(
            json.dumps(dict(rank=rank, error=traceback.format_exc())).encode()
        )
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        connection.close()


def probe(root, layers, offload=0.30):
    if torch.accelerator.device_count() != 4 or not 1 <= layers <= 40:
        raise ValueError("Use four GPUs and 1..40 layers")
    context = mp.get_context("spawn")
    processes, connections = [], []
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="v41-weights-") as directory:
        try:
            for rank in range(4):
                parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
                parent.settimeout(1800)
                process = context.Process(
                    target=_consumer,
                    args=(rank, child, root, directory, layers, offload),
                )
                process.start()
                child.close()
                connections.append(parent)
                processes.append(process)
            reports = [json.loads(c.recv(65536)) for c in connections]
            if not all(r.get("passed") for r in reports):
                raise RuntimeError(reports)
            for process in processes:
                process.join(20)
                if process.exitcode != 0:
                    raise RuntimeError((process.pid, process.exitcode))
            return dict(
                status="passed",
                layers=layers,
                workers=reports,
                wall_seconds=time.monotonic() - started,
                memory_available_after_release=memory_available(),
                scope=(
                    "Real expert storage and sampled bytes; no model forward "
                    "or dense/draft/Engram weights"
                ),
            )
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join(5)
                if process.is_alive():
                    process.kill()
                    process.join(5)
            for connection in connections:
                connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument(
        "--offload", type=float, choices=(0.20, 0.25, 0.30), default=0.30
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = probe(args.model_root, args.layers, args.offload)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
