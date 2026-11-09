# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare identity-gather and direct decode for ordered IQ4_NL PLE rows."""

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch

from vllm.model_executor.layers.quantization.gguf import (
    apply_gguf_embedding,
    dequantize_gguf_rows,
)
from vllm.transformers_utils.gguf_tensor_reader import dequantize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    result = {"torch": torch.__version__, "cuda": torch.version.cuda, "rows": []}
    for tokens in (5, 20):
        # Flash-Next has two n-gram lengths, eight heads each, width 640.
        rows, width = tokens * 16, 640
        data = np.random.default_rng(967).integers(
            0, 256, (rows, width // 32, 18), dtype=np.uint8
        )
        data[:, :, :2] = np.array([1 / 1024], dtype=np.float16).view(np.uint8)
        data = data.reshape(rows, -1)
        packet = torch.from_numpy(data).cuda()
        ids = torch.arange(rows, device="cuda")
        expected = torch.from_numpy(dequantize(data, 20)).to("cuda", torch.float16)
        operations = {
            "identity_gather": lambda ids=ids, packet=packet, width=width: (
                apply_gguf_embedding(ids, packet, 20, width, torch.float16)
            ),
            "direct_decode": lambda packet=packet, width=width: dequantize_gguf_rows(
                packet, 20, width, torch.float16
            ),
        }
        graphs = {}
        for name, operation in operations.items():
            for _ in range(3):
                actual = operation()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(100):
                    actual = operation()
            graphs[name] = graph
            graph.replay()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        samples = {name: [] for name in graphs}
        for epoch in range(8):
            for name in list(graphs)[:: 1 if epoch % 2 == 0 else -1]:
                graph = graphs[name]
                for _ in range(20):
                    graph.replay()
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                for _ in range(10):
                    graph.replay()
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end))
        result["rows"].append(
            {
                "tokens": tokens,
                "rows": rows,
                "width": width,
                "format": "IQ4_NL",
                "epoch_mean_us": samples,
                "median_us": {k: statistics.median(v) for k, v in samples.items()},
                "bitwise_official": True,
            }
        )
        if args.profile:
            torch.cuda.cudart().cudaProfilerStart()
            for name in ("identity_gather", "direct_decode"):
                for _ in range(100):
                    graphs[name].replay()
                torch.accelerator.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            break
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
