# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen queued model work for interference from independent H2D DMA.

Run in fresh processes with CUDA_DEVICE_MAX_CONNECTIONS=8 and 32. These
synthetic streams expose possible queue aliasing; they do not establish a
model speedup. A main-stream delay keeps subsequent work queued before it
executes, separating Python launch latency from device scheduling.
"""

import argparse
import json
import os
import statistics
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--streams", type=int, default=12)
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()
    if args.streams < 1 or args.samples < 1:
        parser.error("streams and samples must be positive")
    torch.accelerator.set_device_index(0)
    main_stream = torch.cuda.current_stream()
    sides = [
        (priority, i, torch.cuda.Stream(priority=priority))
        for priority in (0, -1)
        for i in range(args.streams)
    ]
    # Each copy repeats known bytes into a distinct buffer from foreground.
    host = torch.full((16 << 20,), 37, dtype=torch.uint8, pin_memory=True)
    device = torch.empty_like(host, device="cuda")
    foreground = torch.zeros(4 << 20, dtype=torch.float32, device="cuda")
    events = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
    begin, fg_begin, fg_end, copy_begin, copy_end = events

    def sample(side):
        torch.accelerator.synchronize()
        foreground.zero_()
        begin.record(main_stream)
        # One CTA, approximately 15 ms on V100: enough to submit the whole
        # foreground sequence while leaving the other SMs and DMA available.
        torch.cuda._sleep(20_000_000)
        if side is not None:
            side.wait_event(begin)
            with torch.cuda.stream(side):
                copy_begin.record()
                for _ in range(16):
                    device.copy_(host, non_blocking=True)
                copy_end.record()
        fg_begin.record(main_stream)
        for _ in range(32):
            foreground.add_(1)
        fg_end.record(main_stream)
        fg_end.synchronize()
        fg_ms = fg_begin.elapsed_time(fg_end)
        total_ms = begin.elapsed_time(fg_end)
        if side is not None:
            copy_end.synchronize()
            copy_ms = copy_begin.elapsed_time(copy_end)
        else:
            copy_ms = 0.0
        assert torch.equal(foreground, torch.full_like(foreground, 32))
        if side is not None:
            assert torch.equal(device.cpu(), host)
        return {"foreground_ms": fg_ms, "main_total_ms": total_ms, "copy_ms": copy_ms}

    # Warm allocations, event objects, and both priority pools.
    sample(None)
    for _, _, side in sides:
        sample(side)
    cases = []
    for priority, index, side in [(None, None, None), *sides, (None, None, None)]:
        samples = [sample(side) for _ in range(args.samples)]
        row = {
            "priority": priority,
            "stream_index": index,
            "samples": samples,
            **{key: statistics.median(x[key] for x in samples) for key in samples[0]},
        }
        cases.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "samples"}), flush=True)
    args.output.write_text(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "connections": os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", "unset"),
                "cases": cases,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
