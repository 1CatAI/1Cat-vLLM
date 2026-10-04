# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Research-only boundary probes; never used as a model speed claim.

Run in the owned installed-artifact venv and hold /tmp/gpu0-3.lock.
The graph interval includes timestamp instrumentation and the producer's
final store. It is not a pure scheduler gap. NVLink time uses system-scope
release/acquire and amortizes two separately launched kernels over a loop.
"""

import argparse
import fcntl
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).parent / "csrc/sm70_decode_boundary_baselines.cu"
    with open("/tmp/gpu0-3.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        users = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                "0,1,2,3",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        if users.strip():
            raise RuntimeError("GPU 0-3 are occupied; retry after they are released")
        assert torch.cuda.get_device_capability(0) == (7, 0)
        assert torch.cuda.can_device_access_peer(0, 1)
        module = load(
            name="qwen38_boundary_research",
            sources=[str(source)],
            extra_cuda_cflags=["-O3", "-lineinfo"],
            verbose=True,
        )
        report = {
            "research_only": True,
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "devices": [torch.cuda.get_device_name(i) for i in range(2)],
        }
        with torch.cuda.device(0):
            a = torch.zeros(1, device="cuda", dtype=torch.int32)
            b = torch.zeros_like(a)
            times = torch.zeros(128, device="cuda", dtype=torch.int64)
            module.boundary(a, b, times, 0)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for i in range(64):
                    module.boundary(a, b, times, 2 * i)
                    module.boundary(b, a, times, 2 * i + 1)
            samples = []
            for _ in range(9):
                graph.replay()
                torch.cuda.synchronize()
                t = times.cpu().tolist()
                samples.extend((t[2 * i + 1] - t[2 * i]) / 1000 for i in range(1, 63))
            report["graph_instrumented_boundary_us"] = {
                "median": statistics.median(samples),
                "mean": statistics.mean(samples),
                "min": min(samples),
                "max": max(samples),
                "samples": samples,
                "note": (
                    "Timestamp-to-timestamp; includes final store "
                    "and instrumentation. Global timer resolution is coarse."
                ),
            }
        flags = [
            torch.zeros(1, device=f"cuda:{i}", dtype=torch.int32) for i in range(2)
        ]
        status = [torch.zeros_like(x) for x in flags]
        streams = [torch.cuda.Stream(i) for i in range(2)]
        for i in range(2):
            module.enable_peer(i, 1 - i)
            torch.cuda.synchronize(i)
        samples = []
        count = 4096
        for r in range(10):
            with torch.cuda.device(0), torch.cuda.stream(streams[0]):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                module.roundtrip(
                    flags[0], flags[1], status[0], 1 + r * count, count, True
                )
                end.record()
            with torch.cuda.device(1), torch.cuda.stream(streams[1]):
                module.roundtrip(
                    flags[1], flags[0], status[1], 1 + r * count, count, False
                )
            for i in range(2):
                torch.cuda.synchronize(i)
            assert all(x.item() == 0 for x in status), "Flag wait timed out"
            assert all(x.item() == (r + 1) * count for x in flags), "Wrong generation"
            if r:
                samples.append(start.elapsed_time(end) * 1000 / count)
        report["nvlink_roundtrip_us"] = {
            "median": statistics.median(samples),
            "samples": samples,
            "iterations_per_sample": count,
            "note": (
                "System release/acquire, two GPUs; includes amortized launch skew."
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    k: {a: b for a, b in v.items() if a != "samples"}
                    for k, v in report.items()
                    if k.endswith("us")
                },
                indent=2,
            )
        )
        fcntl.flock(lock, fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
