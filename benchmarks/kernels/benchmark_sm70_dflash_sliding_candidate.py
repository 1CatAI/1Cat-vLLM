# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare draft sliding-window split attention with the shipped paged path.

Eight distinct KV working sets keep the 1K scan larger than V100 L2. Model
admission still requires teacher logits and natural and long-context tasks.
"""

import argparse
import hashlib
import importlib.util
import json
import statistics
from pathlib import Path

import torch
from flash_attn_v100 import flash_attn_prefill_paged


def make_layer(batch, length, page):
    per_request = (length + page - 1) // page
    pages = batch * per_request
    cache = torch.randn(pages, 2, page, 2, 128, device="cuda", dtype=torch.float16)
    key, value = cache.unbind(1)
    query = torch.randn(batch * 8, 8, 128, device="cuda", dtype=torch.float16) * 0.5
    table = torch.randperm(pages, device="cuda").int().reshape(batch, per_request)
    lengths = torch.tensor(
        [length - (r % 3) * 17 for r in range(batch)],
        device="cuda",
        dtype=torch.int32,
    )
    return query, key, value, table, lengths


def dense_reference(layer):
    query, key, value, table, lengths = layer
    outputs = []
    for request, length in enumerate(lengths.tolist()):
        positions = torch.arange(max(0, length - 8 - 2047), length, device="cuda")
        physical = table[request, positions // key.shape[1]].long()
        offsets = positions % key.shape[1]
        k, v = key[physical, offsets].double(), value[physical, offsets].double()
        q = query[request * 8 : (request + 1) * 8].double().reshape(8, 2, 4, 128)
        scores = torch.einsum("qhgd,khd->qhgk", q, k) * (128**-0.5)
        query_positions = torch.arange(length - 8, length, device="cuda")
        visible = (positions[None] >= query_positions[:, None] - 2047) & (
            positions[None] <= query_positions[:, None] + 2047
        )
        scores.masked_fill_(~visible[:, None, None], -torch.inf)
        outputs.append(
            torch.einsum("qhgk,khd->qhgd", scores.softmax(-1), v).reshape(8, 8, 128)
        )
    return torch.cat(outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--contexts", type=int, nargs="+", default=[1024, 8192, 32768, 131072, 262144]
    )
    parser.add_argument("--page-size", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists")
    manifest = json.loads(args.manifest.read_text())
    library = Path(manifest["library"])
    assert (
        hashlib.sha256(library.read_bytes()).hexdigest() == manifest["library_sha256"]
    )
    spec = importlib.util.spec_from_file_location(manifest["module_name"], library)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert torch.cuda.get_device_capability() == (7, 0)
    torch.manual_seed(123)
    report = {
        "manifest": manifest,
        "page_size": args.page_size,
        "layers": args.layers,
        "torch": torch.__version__,
        "scope": "FP16 KV operator comparison, not model admission",
        "measurements": [],
    }
    report["boundary_checks"] = []
    for length in (8, 127, 128, 129, 1023, 1024, 1025, 2047, 2048, 2049, 2055, 4097):
        layer = make_layer(1, length, args.page_size)
        out = torch.empty_like(layer[0])
        partial = torch.empty(1, 40, 8, 8, 128, device="cuda", dtype=torch.float32)
        lse = torch.empty(1, 40, 8, 8, device="cuda", dtype=torch.float32)
        q, k, v, blocks, lengths = layer
        module.run(q, k, v, out, blocks, lengths, partial, lse, 128**-0.5)
        oracle = dense_reference(layer)
        torch.testing.assert_close(out.double(), oracle, atol=1e-3, rtol=1e-3)
        report["boundary_checks"].append(
            {"context": length, "max_abs": (out.double() - oracle).abs().max().item()}
        )
    layer = make_layer(4, 2055, args.page_size)
    layer[-1][-1] = 0
    out = torch.full_like(layer[0], float("nan"))
    partial = torch.full(
        (4, 40, 8, 8, 128), float("nan"), device="cuda", dtype=torch.float32
    )
    lse = torch.full((4, 40, 8, 8), float("nan"), device="cuda", dtype=torch.float32)
    q, k, v, blocks, lengths = layer
    module.run(q, k, v, out, blocks, lengths, partial, lse, 128**-0.5)
    assert torch.isfinite(out).all() and torch.count_nonzero(out[-8:]) == 0
    torch.testing.assert_close(
        out.double(), dense_reference(layer), atol=1e-3, rtol=1e-3
    )
    report["zero_length_padding_pass"] = True
    del layer, out, partial, lse, q, k, v, blocks, lengths, oracle
    for batch in (1, 4):
        for length in args.contexts:
            if batch == 4 and length > 131072:
                continue
            layers = [
                make_layer(batch, length, args.page_size) for _ in range(args.layers)
            ]
            outputs = [torch.empty_like(layers[0][0]) for _ in range(2)]
            partial = torch.empty(
                batch, 40, 8, 8, 128, device="cuda", dtype=torch.float32
            )
            lse = torch.empty(batch, 40, 8, 8, device="cuda", dtype=torch.float32)

            def reference(layer, batch=batch, out=outputs[0]):
                q, k, v, blocks, lengths = layer
                flash_attn_prefill_paged(
                    q.reshape(batch, 8, 8, 128),
                    k,
                    v,
                    blocks,
                    lengths,
                    out=out.reshape(batch, 8, 8, 128),
                    causal=False,
                    window_size=(2047, 2047),
                )

            def candidate(layer, out=outputs[1], partial=partial, lse=lse):
                q, k, v, blocks, lengths = layer
                module.run(q, k, v, out, blocks, lengths, partial, lse, 128**-0.5)

            oracle = dense_reference(layers[0])
            checks = []
            for fn, output in zip((reference, candidate), outputs):
                fn(layers[0])
                diff = output.double() - oracle
                checks.append(
                    {
                        "finite": bool(torch.isfinite(output).all()),
                        "max_abs": diff.abs().max().item(),
                        "rmse": diff.square().mean().sqrt().item(),
                    }
                )
            graphs = []
            for fn in (reference, candidate):
                graph = torch.cuda.CUDAGraph()
                begin = torch.cuda.Event(enable_timing=True, external=True)
                end = torch.cuda.Event(enable_timing=True, external=True)
                with torch.cuda.graph(graph):
                    begin.record()
                    for layer in layers:
                        fn(layer)
                    end.record()
                graphs.append((graph, begin, end))
            samples = [[], []]
            for repeat in range(args.repeats + 5):
                for index in (repeat % 2, 1 - repeat % 2):
                    graph, begin, end = graphs[index]
                    graph.replay()
                    end.synchronize()
                    if repeat >= 5:
                        samples[index].append(begin.elapsed_time(end) * 1000)
            scan_tokens = min(length, 2055)
            kv_bytes = batch * scan_tokens * 2 * 128 * 2 * 2
            report["measurements"].append(
                {
                    "batch": batch,
                    "context": length,
                    "checks": checks,
                    "candidate_scratch_bytes": partial.nbytes + lse.nbytes,
                    "logical_window_kv_bytes_per_layer": kv_bytes,
                    "timings": [
                        {
                            "variant": name,
                            "layer_us": statistics.mean(values) / args.layers,
                            "samples_chain_us": values,
                            "logical_kv_GB_s": kv_bytes
                            * args.layers
                            / statistics.mean(values)
                            / 1000,
                        }
                        for name, values in zip(("reference", "candidate"), samples)
                    ],
                }
            )
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            for graph, _, _ in graphs:
                graph.reset()
            del (
                graph,
                graphs,
                layer,
                layers,
                output,
                outputs,
                partial,
                lse,
                oracle,
                diff,
            )


if __name__ == "__main__":
    main()
