# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare paged q8 attention on request-major C1/C4 KV working sets.

Timings cover attention plus its split reduction, not the complete verifier.
The sixteen-layer working set exceeds L2 without injecting an eviction kernel.
"""

import argparse
import hashlib
import importlib.util
import json
import statistics
import subprocess
from pathlib import Path

import torch


def load_operator(path: Path):
    manifest = json.loads(path.read_text())
    library = Path(manifest["library"])
    if hashlib.sha256(library.read_bytes()).hexdigest() != manifest["library_sha256"]:
        raise ValueError("Library digest differs from its build manifest")
    spec = importlib.util.spec_from_file_location(manifest["module_name"], library)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the benchmark library")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    decoded = module.decoder_check()
    manifest["all_e4m3_codes_bitwise_decode"] = torch.equal(decoded[0], decoded[1])
    if not manifest["all_e4m3_codes_bitwise_decode"]:
        raise AssertionError("E4M3 direct decoding differs from the lookup oracle")
    return module.run, manifest


def make_layer(batch: int, length: int, page: int):
    pages_per_request = (length + page - 1) // page
    pages = batch * pages_per_request
    # Matches the backend's [pages,2,page,kv_heads,head_dim] physical layout.
    raw = torch.randn(pages, 2, page, 1, 256, device="cuda", dtype=torch.float16)
    cache = raw.to(torch.float8_e4m3fn).view(torch.uint8)
    key, value = cache.unbind(1)
    table = torch.randperm(pages, device="cuda").to(torch.int32)
    table = table.view(batch, pages_per_request).contiguous()
    query = torch.randn(batch * 8, 6, 256, device="cuda", dtype=torch.float16) * 0.5
    row_lengths = torch.arange(
        length - 7, length + 1, device="cuda", dtype=torch.int32
    ).repeat(batch)
    return query, key, value, table, row_lengths


def call(operator, layer, buffers):
    query, key, value, table, lengths = layer
    out, partial, lse = buffers
    operator(query, key, value, out, table, lengths, partial, lse, 0.0625, 0.5, 1.25)


def dense_reference(layer):
    """Independent FP64 QK, softmax and PV, one request at a time."""
    query, key, value, table, lengths = layer
    batch = table.shape[0]
    outputs = []
    for request in range(batch):
        keys = key[table[request].long()].reshape(-1, 256)
        values = value[table[request].long()].reshape(-1, 256)
        keys = keys.view(torch.float8_e4m3fn).double() * 0.5
        values = values.view(torch.float8_e4m3fn).double() * 1.25
        q = query[request * 8 : (request + 1) * 8].double()
        scores = torch.einsum("qhd,kd->qhk", q, keys) * 0.0625
        positions = torch.arange(keys.shape[0], device=query.device)
        visible = positions[None, :] < lengths[request * 8 : (request + 1) * 8, None]
        scores.masked_fill_(~visible[:, None, :], -torch.inf)
        probabilities = scores.softmax(-1)
        outputs.append(torch.einsum("qhk,kd->qhd", probabilities, values))
    return torch.cat(outputs)


def clocks():
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,power.limit,power.draw,clocks.sm,clocks.mem,temperature.gpu",
            "--format=csv,noheader",
        ],
        text=True,
    ).splitlines()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4])
    parser.add_argument(
        "--contexts",
        type=int,
        nargs="+",
        default=[1024, 8192, 32768, 65536, 131072, 262144],
    )
    parser.add_argument("--page-size", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; retain prior experiments under distinct paths")
    if args.page_size <= 0 or args.page_size % 16 or args.layers < 1:
        parser.error("Require a positive aligned page size and layer count")
    if args.repeats < 1 or any(n < 8 or n > 262144 for n in args.contexts):
        parser.error("Require positive repeats and contexts between 8 and 262144")
    if any(b not in (1, 4) for b in args.batch_sizes):
        parser.error("This workload compares C1 and C4")
    assert torch.cuda.get_device_capability() == (7, 0)
    torch.manual_seed(123)
    loaded = [load_operator(path) for path in args.manifest]
    report = {
        "scope": "Independent attention operator; not end-to-end admission",
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "manifests": [manifest for _, manifest in loaded],
        "page_size": args.page_size,
        "layers": args.layers,
        "kv_values": "Seeded synthetic E4M3 values in the model's physical layout",
        "bandwidth_scope": "Logical single-pass KV bytes, not measured DRAM bytes",
        "measurements": [],
    }
    for batch in args.batch_sizes:
        for length in args.contexts:
            if batch == 4 and length > 131072:
                continue
            layers = [
                make_layer(batch, length, args.page_size) for _ in range(args.layers)
            ]
            buffers = []
            for _, manifest in loaded:
                splits = manifest["splits"]
                shape = (splits, 8, 6) if batch == 1 else (batch, splits, 8, 6)
                buffers.append(
                    (
                        torch.empty_like(layers[0][0]),
                        torch.empty(*shape, 256, device="cuda", dtype=torch.float32),
                        torch.empty(*shape, 2, device="cuda", dtype=torch.float32),
                    )
                )
            reference = dense_reference(layers[0])
            checks = []
            snapshots = []
            for (operator, manifest), workspace in zip(loaded, buffers):
                call(operator, layers[0], workspace)
                snapshots.append(workspace[0].clone())
                delta = workspace[0].double() - reference
                checks.append(
                    {
                        "variant": manifest["variant"],
                        "output_max_abs_diff": delta.abs().max().item(),
                        "output_rmse": delta.square().mean().sqrt().item(),
                        "finite": bool(torch.isfinite(workspace[0]).all()),
                        "bitwise_to_first": torch.equal(workspace[0], snapshots[0]),
                    }
                )
            del reference
            graphs = []
            for (operator, _), workspace in zip(loaded, buffers):
                begin = torch.cuda.Event(enable_timing=True, external=True)
                end = torch.cuda.Event(enable_timing=True, external=True)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    begin.record()
                    for layer in layers:
                        call(operator, layer, workspace)
                    end.record()
                graphs.append((graph, begin, end))
            before = clocks()
            samples = [[] for _ in graphs]
            for repeat in range(args.repeats + 5):
                # Rotate reference/candidate order to reduce clock/order bias.
                for index in [(repeat + j) % len(graphs) for j in range(len(graphs))]:
                    graph, begin, end = graphs[index]
                    graph.replay()
                    end.synchronize()
                    if repeat >= 5:
                        samples[index].append(begin.elapsed_time(end) * 1000)
            kv_bytes = batch * length * 256 * 2
            report["measurements"].append(
                {
                    "batch": batch,
                    "context": length,
                    "checks": checks,
                    "clocks_before": before,
                    "clocks_after": clocks(),
                    "cache_allocation_bytes": sum(
                        layer[1].numel() * 2 for layer in layers
                    ),
                    "per_arm_scratch_bytes": sum(t.nbytes for t in buffers[0][1:]),
                    "scratch_bytes_by_variant": {
                        manifest["variant"]: sum(t.nbytes for t in buffer[1:])
                        for (_, manifest), buffer in zip(loaded, buffers)
                    },
                    "one_pass_logical_kv_bytes_per_layer": kv_bytes,
                    "ideal_kv_us_at_800_GB_s": kv_bytes / 800e3,
                    "timings": [
                        {
                            "variant": manifest["variant"],
                            "layer_chain_us": statistics.mean(values),
                            "mean_layer_us": statistics.mean(values) / args.layers,
                            "effective_kv_GB_s": kv_bytes
                            * args.layers
                            / statistics.mean(values)
                            / 1000,
                            "samples_us": values,
                        }
                        for (_, manifest), values in zip(loaded, samples)
                    ],
                }
            )
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            for graph, _, _ in graphs:
                graph.reset()
            del graph, layer, workspace, graphs, layers, buffers, snapshots, delta


if __name__ == "__main__":
    main()
