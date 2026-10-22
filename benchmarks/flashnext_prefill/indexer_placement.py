# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Same native MTP indexer, same values, mapped host versus device storage."""

import argparse
import json
import math
import statistics
from functools import partial
from pathlib import Path

import torch

from vllm.models.qwen4_exp.nvidia.ops.qsa_shared_key import maybe_shared_key_scores
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor


def score(q, history, table, requests, positions, lengths, columns):
    result = maybe_shared_key_scores(
        q, history, table, requests, positions, lengths, 4, columns, math.sqrt(128)
    )
    assert result is not None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20261015)
    rows = []
    for context in (32768, 65536, 131072, 261000):
        page_size = 204
        pages = math.ceil((context + 5) / (page_size * 4))
        host = torch.randn(pages, page_size, 1, 128, dtype=torch.float16).pin_memory()
        cache = host.cuda()
        mapped = get_accelerator_view_from_cpu_tensor(host)
        q = torch.randn(5, 4, 128, device="cuda", dtype=torch.float16)
        table = torch.arange(pages, device="cuda", dtype=torch.int32)[None, :]
        requests = torch.zeros(5, device="cuda", dtype=torch.int32)
        positions = torch.arange(context, context + 5, device="cuda", dtype=torch.int64)
        lengths = torch.tensor([context + 5], device="cuda", dtype=torch.int32)
        outputs = {}
        timings = {}
        for name, history in (("host", mapped), ("device", cache)):
            run = partial(
                score,
                q,
                history,
                table,
                requests,
                positions,
                lengths,
                pages * page_size,
            )
            for _ in range(5):
                run()
            torch.accelerator.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                scores, visible = run()
            samples = []
            for _ in range(9):
                start = torch.Event(device="cuda", enable_timing=True)
                end = torch.Event(device="cuda", enable_timing=True)
                start.record()
                for _ in range(20):
                    graph.replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) / 20)
            timings[name] = statistics.median(samples)
            outputs[name] = scores.clone(), visible.clone()
            torch.accelerator.synchronize()
        a, visible = outputs["host"]
        b, other_visible = outputs["device"]
        assert torch.equal(visible, other_visible)
        mask = torch.arange(a.shape[1], device="cuda")[None, :] < visible[:, None]
        assert torch.equal(a[mask], b[mask])
        row = dict(
            context_tokens=context,
            rows=5,
            heads=4,
            dimension=128,
            host_ms=timings["host"],
            device_ms=timings["device"],
            speedup=timings["host"] / timings["device"],
            bitwise_equal=True,
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
        args.out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
