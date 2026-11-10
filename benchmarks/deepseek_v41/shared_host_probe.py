# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Four-process CUDA access to one host allocation, including PSS accounting.

Run inside the isolated GPU container. Large capacities are registered in full
but GPU kernels access bounded samples, including offsets past four GiB.
"""

import argparse
import json
import mmap
import multiprocessing as mp
import os
import socket
import sys
import time
import traceback
from pathlib import Path

import torch

from vllm.utils.shared_host import SharedHostArena

_NAME = "vllm-ds41-shared-host-probe"
_SAMPLE = 1024**2


def shared_pss(pid: int) -> dict[str, int]:
    totals = dict.fromkeys(("Size", "Rss", "Pss", "Shared_Dirty", "Private_Dirty"), 0)
    selected = False
    for line in Path(f"/proc/{pid}/smaps").read_text().splitlines():
        if line and line[0] in "0123456789abcdef" and "-" in line.split()[0]:
            selected = f"memfd:{_NAME}" in line
        elif selected and ":" in line:
            key, value = line.split(":", 1)
            if key in totals:
                totals[key] += int(value.split()[0]) * 1024
    return totals


def memory_available() -> int:
    line = next(
        line
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    return int(line.split()[1]) * 1024


def cgroup_memory() -> dict[str, int | str]:
    """Record cgroup usage separately from this object's proportional size."""
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        hierarchy, controllers, relative = line.split(":", 2)
        if hierarchy == "0" and not controllers:
            root = Path("/sys/fs/cgroup") / relative.lstrip("/")
            if not (root / "memory.current").exists():
                # A cgroup namespace can present the process's group at root.
                root = Path("/sys/fs/cgroup")
            if (root / "memory.current").exists():
                return {
                    "path": str(root),
                    "current": int((root / "memory.current").read_text()),
                    "limit": (root / "memory.max").read_text().strip(),
                }
        if "memory" in controllers.split(","):
            root = Path("/sys/fs/cgroup/memory") / relative.lstrip("/")
            if not (root / "memory.usage_in_bytes").exists():
                root = Path("/sys/fs/cgroup/memory")
            return {
                "path": str(root),
                "current": int((root / "memory.usage_in_bytes").read_text()),
                "limit": (root / "memory.limit_in_bytes").read_text().strip(),
            }
    raise RuntimeError("Memory cgroup accounting is required for this probe")


def _consumer(rank, connection, barrier):
    host = mapped = view = arena = native = marker = peer_data = None
    registered = False
    try:
        torch.set_num_threads(1)
        torch.accelerator.set_device_index(rank)
        # The prebuilt helper registers an existing region and wraps its device
        # alias; it must not use a tensor helper that silently makes a copy.
        from vllm.model_executor.layers.quantization.gguf_expert_cache import _native

        native = _native()
        arena = SharedHostArena.receive(connection)
        view = memoryview(arena.buffer)
        host = torch.frombuffer(view, dtype=torch.uint8)
        native.host_register(host.data_ptr(), arena.size)
        registered = True
        mapped = native.mapped_view(host, rank)
        sample = min(_SAMPLE, arena.size)
        offsets = {0, arena.size - sample}
        if arena.size > 4 * 1024**3 + sample:
            offsets.add(4 * 1024**3 + mmap.PAGESIZE)
        for offset in sorted(offsets):
            checksum = int(mapped[offset : offset + sample].sum().item())
            if checksum != 37 * sample:
                raise AssertionError((rank, offset, checksum))
        report = {
            "rank": rank,
            "pid": os.getpid(),
            "identity": vars(arena.identity),
            "cpu_pointer": host.data_ptr(),
            "device_pointer": mapped.data_ptr(),
            "sample_offsets": sorted(offsets),
            "samples_match": True,
            "shared_mapping": shared_pss(os.getpid()),
        }
        connection.send(json.dumps(report).encode())
        barrier.wait(180)
        # A GPU kernel, rather than merely a DMA copy, writes shared host pages.
        start = rank * mmap.PAGESIZE
        mapped[start : start + mmap.PAGESIZE].fill_(rank + 1)
        torch.accelerator.synchronize()
        barrier.wait(180)
        barrier.wait(180)
        marker = mapped[4 * mmap.PAGESIZE : 5 * mmap.PAGESIZE]
        if int(marker.sum().item()) != 99 * mmap.PAGESIZE:
            raise AssertionError("CPU updates are not visible to the GPU")
        peer = (rank + 1) % 4
        peer_data = mapped[peer * mmap.PAGESIZE : (peer + 1) * mmap.PAGESIZE]
        if int(peer_data.sum().item()) != (peer + 1) * mmap.PAGESIZE:
            raise AssertionError("Other GPU updates are not visible")
        connection.send(json.dumps({"rank": rank, "bidirectional_ok": True}).encode())
        barrier.wait(180)
    except BaseException:
        connection.send(
            json.dumps({"rank": rank, "error": traceback.format_exc()}).encode()
        )
        barrier.abort()
        raise
    finally:
        marker = peer_data = mapped = None
        if registered:
            torch.accelerator.synchronize()
            native.host_unregister(host.data_ptr())
        mapped = host = None
        if view is not None:
            view.release()
        if arena is not None:
            arena.close()
        connection.close()


def probe(size: int) -> dict:
    if sys.platform != "linux" or not hasattr(os, "memfd_create"):
        raise RuntimeError("The CUDA shared-host probe requires Linux")
    if not torch.cuda.is_available() or torch.accelerator.device_count() != 4:
        raise RuntimeError("Expose exactly the four test GPUs to the probe")
    if size < 8 * mmap.PAGESIZE:
        raise ValueError("Probe needs at least eight pages")
    if memory_available() < size + 12 * 1024**3:
        raise RuntimeError(
            "Insufficient RAM for arena, worker overhead and 8 GiB reserve"
        )
    context = mp.get_context("spawn")
    barrier = context.Barrier(5)
    processes, connections = [], []
    started = time.monotonic()
    baseline_memory = memory_available()
    baseline_cgroup = cgroup_memory()
    with SharedHostArena.create(size, _NAME) as arena:
        view = memoryview(arena.buffer)
        host = torch.frombuffer(view, dtype=torch.uint8)
        host.fill_(37)
        host = None
        view.release()
        try:
            for rank in range(4):
                parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
                parent.settimeout(180)
                process = context.Process(target=_consumer, args=(rank, child, barrier))
                process.start()
                child.close()
                processes.append(process)
                connections.append(parent)
                arena.send(parent)
            reports = [json.loads(conn.recv(65536)) for conn in connections]
            if any("error" in report for report in reports):
                raise RuntimeError(reports)
            if any(report["identity"] != vars(arena.identity) for report in reports):
                raise AssertionError("Workers mapped different memory objects")
            # Read accounting only after every worker has mapped the object.
            for report in reports:
                report["shared_mapping"] = shared_pss(report["pid"])
            owner_pss = shared_pss(os.getpid())
            physical_pss = owner_pss["Pss"] + sum(
                report["shared_mapping"]["Pss"] for report in reports
            )
            # Attribute only this memfd, not CUDA contexts or repeated RSS.
            if not 0.9 * arena.size <= physical_pss <= 1.1 * arena.size:
                raise AssertionError(
                    ("Expected one physical arena", physical_pss, size)
                )
            mapped_cgroup = cgroup_memory()
            mapped_memory = memory_available()
            if mapped_memory < 8 * 1024**3:
                raise RuntimeError("Shared mapping left less than 8 GiB host reserve")
            barrier.wait(180)
            barrier.wait(180)
            for rank in range(4):
                data = arena.buffer[rank * mmap.PAGESIZE : (rank + 1) * mmap.PAGESIZE]
                if data != bytes([rank + 1]) * mmap.PAGESIZE:
                    raise AssertionError("GPU writes are not visible to CPU")
            arena.buffer[4 * mmap.PAGESIZE : 5 * mmap.PAGESIZE] = (
                b"\x63" * mmap.PAGESIZE
            )
            barrier.wait(180)
            updates = [json.loads(conn.recv(65536)) for conn in connections]
            if not all(report.get("bidirectional_ok") for report in updates):
                raise RuntimeError(updates)
            barrier.wait(180)
            for process in processes:
                process.join(30)
                if process.exitcode != 0:
                    raise RuntimeError(f"Worker exited with {process.exitcode}")
            result = {
                "status": "passed",
                "arena": vars(arena.identity),
                "shared_pss_sum_bytes": physical_pss,
                "owner_mapping": owner_pss,
                "workers": reports,
                "bidirectional_checks": updates,
                "host_memory_available_before": baseline_memory,
                "host_memory_available_mapped": mapped_memory,
                "cgroup_before": baseline_cgroup,
                "cgroup_mapped": mapped_cgroup,
                "cgroup_mapped_increment_bytes": (
                    mapped_cgroup["current"] - baseline_cgroup["current"]
                ),
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
    if shared_pss(os.getpid())["Size"]:
        raise AssertionError("Owner retained the arena after cleanup")
    result["owner_mapping_released"] = True
    result["host_memory_available_after_release"] = memory_available()
    result["cgroup_after_release"] = cgroup_memory()
    result["wall_seconds"] = time.monotonic() - started
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bytes", type=int, default=256 * 1024**2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = probe(args.bytes)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
