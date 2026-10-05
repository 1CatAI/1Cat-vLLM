# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare full pinned IQ4_NL PLE tables on one and two NUMA nodes."""

import argparse
import concurrent.futures
import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import gguf
import numpy as np
import torch

from vllm.models.qwen4_exp.common.ple import (
    available_host_bytes,
    ple_host_reserve_bytes,
    total_host_bytes,
)
from vllm.models.qwen4_exp.nvidia.gguf_embedding import pinned_iq4nl_rows
from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpNGramEmbedding
from vllm.platforms import current_platform
from vllm.transformers_utils.gguf_config import load_gguf_config
from vllm.utils.cpu_resource_utils import parse_id_list
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed-gguf", type=Path, required=True)
    parser.add_argument("--config-gguf", type=Path, required=True)
    parser.add_argument("--acceptance-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source = gguf.GGUFReader(str(args.packed_gguf))
    table = next(t for t in source.tensors if t.name == "per_layer_token_embd.weight")
    if int(table.tensor_type) != 20 or int(table.shape[0]) != 160:
        raise ValueError("Requires packed IQ4_NL rows with K=160")
    raw = table.data.reshape(-1, 90)
    available, total = available_host_bytes(), total_host_bytes()
    if available is None or total is None:
        raise ValueError("Cannot establish available host capacity")
    if available - ple_host_reserve_bytes(total) < 2 * raw.nbytes:
        raise ValueError("Two complete pinned table copies do not fit host capacity")
    rows = json.loads(args.acceptance_file.read_text())["rows"]

    config = load_gguf_config(args.config_gguf).get_text_config()
    ref = SimpleNamespace(
        eos_token_id=config.gguf_ple_eos_token_id,
        **{k: torch.tensor(v) for k, v in config.gguf_ple_constants.items()},
    )
    torch.set_num_threads(1)
    hosts = []
    original = os.sched_getaffinity(0)

    nodes = [current_platform.get_device_numa_node(rank) for rank in range(4)]
    if nodes[0] != nodes[1] or nodes[2] != nodes[3] or nodes[0] == nodes[2]:
        raise ValueError("Requires two GPUs on each of two CPU NUMA nodes")
    for device in [0, 2]:
        node = nodes[device]
        cores = set(
            parse_id_list(
                Path(f"/sys/devices/system/node/node{node}/cpulist").read_text()
            )
        )
        os.sched_setaffinity(0, cores & original)
        torch.accelerator.set_device_index(device)
        start = time.monotonic()
        host = torch.empty(raw.shape, dtype=torch.uint8, pin_memory=True)
        for begin in range(0, raw.shape[0], 1048576):
            end = min(begin + 1048576, raw.shape[0])
            host[begin:end].copy_(torch.from_numpy(raw[begin:end]))
        hosts.append(host)
        print(
            json.dumps(
                dict(
                    event="allocated",
                    node=node,
                    bytes=host.numel(),
                    seconds=time.monotonic() - start,
                )
            ),
            flush=True,
        )
    os.sched_setaffinity(0, original)
    # Record physical page placement, not merely requested affinity.
    placements = []
    for host in hosts:
        address = host.data_ptr()
        mapping = None
        for line in Path("/proc/self/maps").read_text().splitlines():
            span = line.split()[0]
            lo, hi = [int(x, 16) for x in span.split("-")]
            if lo <= address < hi:
                mapping = span.split("-")[0]
                break
        matches = [
            line
            for line in Path("/proc/self/numa_maps").read_text().splitlines()
            if mapping and line.split()[0] == mapping
        ]
        placements.append(matches)
    results = []
    bank = torch.cat(
        [
            Qwen4ExpNGramEmbedding._compute_ngram_ids_cpu_small(
                ref,
                torch.tensor(row["output_token_ids"], dtype=torch.int32),
                torch.tensor([0, len(row["output_token_ids"])], dtype=torch.int32),
                torch.tensor([row["prompt_token_ids"][-2:]], dtype=torch.int32),
            ).reshape(-1)
            for row in rows
        ]
    )
    barrier = threading.Barrier(4, timeout=60)

    def prepare_rank(rank, m):
        torch.accelerator.set_device_index(rank)
        selected = rows[:1] if m < 20 else rows[:4]
        step = 1 if m == 1 else 5
        tokens = torch.tensor(
            [t for row in selected for t in row["output_token_ids"][:step]],
            dtype=torch.int32,
        )
        starts = torch.arange(0, m + 1, step, dtype=torch.int32)
        context = torch.tensor(
            [row["prompt_token_ids"][-2:] for row in selected], dtype=torch.int32
        )
        indices = Qwen4ExpNGramEmbedding._compute_ngram_ids_cpu_small(
            ref, tokens, starts, context
        ).reshape(-1)
        count = {1: 4096, 5: 768, 20: 192}[m]
        bank_ids = bank[: count * m * 16].reshape(count, m * 16).cuda()
        ids = bank_ids[0]
        indices = ids.cpu()
        book = torch.tensor(
            gguf.quants.IQ4_NL.kvalues, dtype=torch.float32, device="cuda"
        )
        output = torch.empty((indices.numel(), 160), dtype=torch.float16, device="cuda")
        expected = torch.from_numpy(
            gguf.quants.dequantize(
                np.array(raw[indices.numpy()], copy=True),
                gguf.GGMLQuantizationType.IQ4_NL,
            ).astype(np.float16)
        ).cuda()
        graphs = []
        for host in hosts:
            mapped = get_accelerator_view_from_cpu_tensor(host)

            def run(mapped=mapped):
                pinned_iq4nl_rows(mapped.data_ptr(), ids, book, output, 160)

            for _ in range(8):
                run()
            torch.accelerator.synchronize()
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=torch.cuda.Stream(device=rank)):
                for slot in range(count):
                    pinned_iq4nl_rows(
                        mapped.data_ptr(), bank_ids[slot], book, output, 160
                    )
            graphs.append(graph)
        return graphs, ids, book, output, indices, bank_ids, count

    def run_rank(rank, m, state):
        torch.accelerator.set_device_index(rank)
        graphs, ids, book, output, indices, bank_ids, count = state
        timing = []
        for arm in ["single", "local", "local", "single"]:
            index = 0 if arm == "single" or rank < 2 else 1
            graph = graphs[index]
            for _ in range(4):
                graph.replay()
            torch.accelerator.synchronize()
            barrier.wait()
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(8):
                graph.replay()
            stop.record()
            stop.synchronize()
            timing.append(
                dict(arm=arm, us=start.elapsed_time(stop) * 1000 / (8 * count))
            )
            barrier.wait()
        return dict(
            rank=rank,
            m=m,
            rows=indices.numel(),
            bytes=indices.numel() * 90,
            exact=True,
            timings=timing,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for m in [1, 5, 20]:
            states = [prepare_rank(rank, m) for rank in range(4)]
            batch = list(
                pool.map(
                    lambda rank, m=m, states=states: run_rank(rank, m, states[rank]),
                    range(4),
                )
            )
            results.extend(batch)
            print(json.dumps(batch), flush=True)
            args.output.write_text(
                json.dumps(
                    dict(
                        placements=placements,
                        results=results,
                        scope="changing_real_ngram_ids",
                        torch=torch.__version__,
                        cuda=torch.version.cuda,
                    ),
                    indent=2,
                )
            )
    print("complete", flush=True)


if __name__ == "__main__":
    main()
