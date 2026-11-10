# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import statistics
from pathlib import Path

import torch

import vllm
from vllm.models.qwen4_exp.nvidia.sm70_fp16_gemv import _split_gdn_projection_tails

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
results = []
for m in (5, 20):
    banks = [
        (
            torch.randn(m, 4096, device="cuda", dtype=torch.float16),
            torch.randn(m, 24, device="cuda", dtype=torch.float16),
            torch.empty(m, 12, 128, device="cuda", dtype=torch.float16),
        )
        for _ in range(36)
    ]

    def run(candidate, banks=banks, m=m):
        for q, ba, z in banks:
            if candidate:
                _split_gdn_projection_tails(q, ba, z)
            else:
                _ = ba[:, :12].contiguous()
                _ = ba[:, 12:].contiguous()
                z.copy_(q[:, 2560:].contiguous().view(m, 12, 128))

    timings = []
    for candidate in (False, True):
        for _ in range(10):
            run(candidate)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(candidate)
        for _ in range(10):
            graph.replay()
        samples = []
        for _ in range(5):
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            for _ in range(100):
                graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 10)
        timings.append(statistics.median(samples))
    results.append(
        {
            "m": m,
            "layers": 36,
            "old_us": timings[0],
            "new_us": timings[1],
            "saved_us": timings[0] - timings[1],
        }
    )
report = {
    "version": vllm.__version__,
    "cases": results,
    "scope": (
        "projection tail copy only; QKV stride and recurrent arithmetic unchanged; "
        "not model-round time"
    ),
}
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
