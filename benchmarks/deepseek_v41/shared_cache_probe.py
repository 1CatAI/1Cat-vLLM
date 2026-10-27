# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synthetic TP4 M1/M6 expert-cache protocol, not a model speed benchmark."""

import argparse
import json
import multiprocessing as mp
import socket
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from vllm.model_executor.layers.quantization.gguf_shared_runtime import (
    SHARDS,
    SharedExpertLayerSpec,
    SharedExpertOwner,
    SharedExpertRank,
)
from vllm.utils.shared_host import SharedHostArena


def _run_rank(rank, data, control, specs, iterations, barrier):
    banks, storage = {}, []
    for layer, spec in enumerate(specs):
        for shard, layout in zip(SHARDS, spec.layouts(rank)):
            size = (spec.resident + 8) * layout.device_rows * layout.device_row_bytes
            backing = torch.zeros(
                size + layout.guard_bytes, dtype=torch.uint8, device=rank
            )
            bank = backing[:size].view(
                spec.resident + 8, layout.device_rows, layout.device_row_bytes
            )
            for expert in range(spec.resident):
                bank[expert].view(torch.int16).fill_(expert + layer * 1024)
            banks[layer, shard] = bank
            storage.append((backing, layout))
    cache = SharedExpertRank(data, control, rank, specs, banks)
    try:
        cache.warmup()
        graphs = {}
        for layer in range(len(specs)):
            for count in (6, 36):
                ids = torch.zeros(count, dtype=torch.int32, device=rank)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = cache.resolve(layer, ids)
                graphs[layer, count] = graph, ids, output
        barrier.wait(180)
        misses, milliseconds = [], []
        for i in range(iterations):
            count = 6 if i % 2 == 0 else 36
            for layer, spec in enumerate(specs):
                graph, ids, output = graphs[layer, count]
                requested = (
                    torch.arange(count, device=rank, dtype=torch.int32)
                    + spec.resident
                    + i * 36
                ) % spec.experts
                ids.copy_(requested)
                before, after = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                before.record()
                graph.replay()
                after.record()
                after.synchronize()
                milliseconds.append(before.elapsed_time(after))
                misses.append(int(cache.control_host[3]))
                for shard in SHARDS:
                    actual = banks[layer, shard][output.long()].view(torch.int16)
                    if not bool(
                        (actual == requested[:, None, None] + layer * 1024).all()
                    ):
                        raise AssertionError(
                            (rank, i, layer, shard, "incorrect expert bytes")
                        )
        for backing, layout in storage:
            if layout.guard_bytes and bool(backing[-layout.guard_bytes :].any()):
                raise AssertionError("Bank guard was overwritten")
        return {
            "rank": rank,
            "passed": True,
            "misses": misses,
            "protocol_milliseconds": milliseconds,
        }
    finally:
        cache.close()


def _consumer(rank, connection, spec, iterations, barrier):
    data = control = None
    try:
        torch.set_num_threads(1)
        torch.accelerator.set_device_index(rank)
        data = SharedHostArena.receive(connection)
        control = SharedHostArena.receive(connection)
        report = _run_rank(rank, data, control, spec, iterations, barrier)
        connection.send(json.dumps(report).encode())
        barrier.wait(180)
    except BaseException:
        connection.send(
            json.dumps({"rank": rank, "error": traceback.format_exc()}).encode()
        )
        barrier.abort()
        raise
    finally:
        if control is not None:
            control.close()
        if data is not None:
            data.close()
        connection.close()


def probe(iterations=8, layers=1):
    if iterations < 2 or torch.accelerator.device_count() != 4 or not 1 <= layers <= 6:
        raise ValueError("Use at least two iterations and exactly four GPUs")
    kinds = (
        (29, 16, 17),
        (29, 29, 29),
        (16, 16, 16),
        (17, 17, 17),
        (29, 29, 16),
        (17, 17, 29),
    )
    specs = tuple(
        SharedExpertLayerSpec(5120, 2304, 384, 269, kinds[i]) for i in range(layers)
    )
    context = mp.get_context("spawn")
    barrier = context.Barrier(5)
    connections, processes = [], []
    started = time.monotonic()
    with SharedExpertOwner(specs, timeout_ns=60_000_000_000) as owner:
        for layer, spec in enumerate(specs):
            for expert in range(spec.resident, spec.experts):
                offset = owner.coordinator.cold_offset(layer, expert)
                row = np.ndarray(
                    (spec.expert_bytes // 2,),
                    dtype="<u2",
                    buffer=owner.data.buffer,
                    offset=offset,
                )
                row.fill(expert + layer * 1024)
        del row
        try:
            for rank in range(4):
                parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
                parent.settimeout(240)
                process = context.Process(
                    target=_consumer, args=(rank, child, specs, iterations, barrier)
                )
                process.start()
                child.close()
                connections.append(parent)
                processes.append(process)
                owner.send(parent)
            barrier.wait(180)
            reports = [json.loads(connection.recv(65536)) for connection in connections]
            if not all(report.get("passed") for report in reports):
                raise RuntimeError(reports)
            if owner.coordinator.error():
                raise RuntimeError(owner.coordinator.error())
            if len({tuple(report["misses"]) for report in reports}) != 1:
                raise AssertionError("Ranks observed different copy plans")
            cold = 0
            for layer, spec in enumerate(specs):
                for expert in range(spec.experts):
                    if owner.coordinator.slot(layer, expert) >= 0:
                        continue
                    cold += 1
                    offset = owner.coordinator.cold_offset(layer, expert)
                    row = np.ndarray(
                        (spec.expert_bytes // 2,),
                        dtype="<u2",
                        buffer=owner.data.buffer,
                        offset=offset,
                    )
                    if not bool((row == expert + layer * 1024).all()):
                        raise AssertionError(
                            (layer, expert, "canonical cold bytes corrupted")
                        )
            del row
            if cold != 115 * layers:
                raise AssertionError("Actual cold expert count changed")
            barrier.wait(180)
            for process in processes:
                process.join(20)
                if process.exitcode != 0:
                    raise RuntimeError((process.pid, process.exitcode))
            result = {
                "status": "passed",
                "iterations": iterations,
                "geometry": {
                    "hidden": 5120,
                    "intermediate": 2304,
                    "experts": 384,
                    "cold": cold,
                    "tp": 4,
                    "layers": layers,
                },
                "arena": vars(owner.data.identity),
                "workers": reports,
                "all_canonical_cold_bytes_match": True,
                "wall_seconds_including_startup": time.monotonic() - started,
            }
        finally:
            barrier.abort()
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join(5)
                if process.is_alive():
                    process.kill()
                    process.join(5)
            for connection in connections:
                connection.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = probe(args.iterations, args.layers)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
