# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native SM70 dense microbenchmark on all Flash-Next TP4 weight shapes.

Run control and candidate with their complete source runtimes and identical
arguments under the GPU lock. Model loading, attention and MTP are not run.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
import torch
import vllm._C

from vllm.model_executor.layers.quantization.gguf_dense_hmma_formats import decode, pack
from vllm.transformers_utils.gguf_tensor_reader import GGUFReader, dequantize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--samples", type=int, default=12)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    rng = np.random.default_rng(20261010)
    readers = [GGUFReader(f) for f in sorted(args.model.glob("*.gguf"))]
    tensors = {t.name: t for reader in readers for t in reader.tensors}
    groups = []
    for layer in range(48):
        prefix = f"blk.{layer}."
        if prefix + "attn_qkv.weight" in tensors:
            members = [
                ("gdn_in", [prefix + "attn_qkv.weight", prefix + "attn_gate.weight"]),
                ("gdn_out", [prefix + "ssm_out.weight"]),
            ]
        else:
            members = [
                ("attn_in", [prefix + f"attn_{v}.weight" for v in ("q", "k", "v")]),
                ("attn_out", [prefix + "attn_output.weight"]),
            ]
        for kind, names in members:
            codes, high, stats, fmts, ns, refs, types = [], [], [], [], [], [], []
            x = None
            for name in names:
                tensor = tensors[name]
                raw = tensor.data
                if kind.endswith("_out"):
                    raw = np.ascontiguousarray(raw[:, : raw.shape[1] // 4])
                else:
                    # Two 256-dimensional KV heads are replicated over TP4.
                    n = (
                        256
                        if name.endswith(("attn_k.weight", "attn_v.weight"))
                        else raw.shape[0] // 4
                    )
                    raw = np.ascontiguousarray(raw[:n])
                fmt, q, s, minimum, group = decode(raw, int(tensor.tensor_type))
                k = q.shape[1]
                if x is None:
                    x = torch.from_numpy(
                        rng.normal(0, 0.5, (20, k)).astype(np.float16)
                    ).cuda()
                c, h, sc = [
                    torch.from_numpy(p).cuda() for p in pack(fmt, q, s, minimum, group)
                ]
                codes.append(c)
                high.append(h)
                stats.append(sc)
                fmts.append(fmt)
                ns.append(q.shape[0])
                types.append(int(tensor.tensor_type))
                w = torch.from_numpy(
                    dequantize(raw, int(tensor.tensor_type)).astype(np.float32)
                ).cuda()
                refs.append(x.float() @ w.T)
            tiles = sum((n + 31) // 32 for n in ns)
            groups.append(
                dict(
                    kind=kind,
                    layer=layer,
                    x=x,
                    k=k,
                    codes=codes,
                    high=high,
                    stats=stats,
                    fmts=fmts,
                    ns=ns,
                    types=types,
                    out=torch.empty(20, sum(ns), device="cuda", dtype=torch.float16),
                    ref=torch.cat(refs, dim=1),
                    ws=torch.empty(tiles * 3 * 256, device="cuda"),
                    cnt=torch.zeros(tiles * 3, device="cuda", dtype=torch.int32),
                )
            )

    def run(g, m):
        torch.ops._C.gguf_dense_segments_sm70_out(
            g["x"][:m],
            g["codes"],
            g["high"],
            g["stats"],
            list(g["out"][:m].split(g["ns"], dim=1)),
            g["fmts"],
            g["ns"],
            g["k"],
            1,
            8 if g["k"] == 1536 else 4,
            g["ws"],
            g["cnt"],
            None,
        )

    report = dict(
        scope="Native operator microbenchmark; no end-to-end claim",
        source=args.source_sha,
        core=str(vllm._C.__file__),
        core_sha256=hashlib.sha256(Path(vllm._C.__file__).read_bytes()).hexdigest(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        measurements=[],
        checks=[],
    )
    for m in (5, 20):
        for g in groups:
            run(g, m)
            torch.accelerator.synchronize()
            y, ref = g["out"][:m].float(), g["ref"][:m]
            error = float((y - ref).norm() / ref.norm())
            assert error < 0.003
            assert not bool(g["cnt"].any())
            report["checks"].append(
                dict(
                    M=m,
                    kind=g["kind"],
                    layer=g["layer"],
                    official_rel_l2=error,
                    output_sha256=hashlib.sha256(
                        g["out"][:m].cpu().numpy().tobytes()
                    ).hexdigest(),
                )
            )
        for kind in ("gdn_in", "gdn_out", "attn_in", "attn_out"):
            subset = [g for g in groups if g["kind"] == kind]
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                for g in subset:
                    run(g, m)
            torch.accelerator.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for g in subset:
                    run(g, m)
            observations = []
            for _ in range(args.samples):
                for _ in range(3):
                    graph.replay()
                start, stop = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                for _ in range(8):
                    graph.replay()
                stop.record()
                stop.synchronize()
                observations.append(start.elapsed_time(stop) * 1000 / 8)
            record = dict(
                M=m,
                kind=kind,
                layers=len(subset),
                K=subset[0]["k"],
                N=subset[0]["ns"],
                types=sorted({tuple(g["types"]) for g in subset}),
                weight_bytes=sum(
                    t.numel()
                    for g in subset
                    for name in ("codes", "high", "stats")
                    for t in g[name]
                ),
                median_us=statistics.median(observations),
                samples_us=observations,
            )
            report["measurements"].append(record)
            print(json.dumps(record), flush=True)
            args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
