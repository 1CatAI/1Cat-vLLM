# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Research-only TP4 HC-to-dense chain screen; not model latency.

Build the candidate DSO from
benchmarks/csrc/sm70_hcx_consumer_pipeline_micro.cu.
Real rank0 TP4 projection shards are mirrored on all ranks for this isolated
screen. HC weights use each rank's actual TP4 partition. Numerical gates run
before graph timing; both arms share canonical packed dense storage.
"""

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import vllm._C as core

from vllm.model_executor.layers.quantization import gguf_dmv13_dense as dense
from vllm.models.qwen4_exp.nvidia.sm70_hcx import Sm70HcxRuntime, pack_down, pack_up
from vllm.transformers_utils.gguf_tensor_reader import GGUFReader, dequantize


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hc-weights", type=Path, required=True)
    parser.add_argument("--projection-shards", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, nargs="+", required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--rows", type=int, default=5, choices=range(1, 9))
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(rank)
    torch.set_num_threads(1)
    torch.ops.load_library(str(args.library))
    dist.init_process_group("gloo")
    assert dist.get_world_size() == 4
    rt = Sm70HcxRuntime(dist.group.WORLD, torch.device("cuda", rank))
    assert rt.enabled, rt.reason
    m = args.rows
    with np.load(args.projection_shards, allow_pickle=False) as archive:
        records = json.loads(str(archive["metadata"]))
        for record in records:
            for seg in record["segs"]:
                seg["raw"] = archive[seg["array"]].copy()
    records = [r for r in records if r["kind"] in ("gdn_in", "attn_in")][: args.pairs]
    hc = torch.load(args.hc_weights, map_location="cpu", weights_only=True)
    tensors = {t.name: t for file in args.gguf for t in GGUFReader(file).tensors}
    pairs = []
    for record, h in zip(records, hc):
        down = (
            torch.cat((h["down"].float(), h["inj"].float(), torch.zeros(12, 10240)))
            .half()
            .cuda()
        )
        up = h["up"].half().cuda()
        group = {
            "wd": pack_down(down, rt.logical_rank),
            "wu": pack_up(up, rt.logical_rank),
            "nw": h["nw"].half().cuda(),
            "kind": record["kind"],
            "layer": record["layer"],
            "codes": [],
            "high": [],
            "scale": [],
            "fmts": [],
            "ns": [],
            "oracle": [],
            "official": [],
        }
        for seg in record["segs"]:
            # The supplied extraction divides every column-parallel tensor
            # by four. Attention has two KV heads replicated under TP4, so
            # K/V must retain a full 256-row head, not a 128-row quarter.
            raw = seg["raw"]
            if record["kind"] == "attn_in" and seg["name"].endswith(
                ("attn_k.weight", "attn_v.weight")
            ):
                source = tensors[seg["name"]]
                raw = np.ascontiguousarray(source.data[:256])
            fmt, q, sc, minimum, gs = dense.decode(raw, seg["gtype"])
            assert q.shape[1] == 2560 and fmt in (
                dense.Q4K,
                dense.Q5K,
                dense.Q6K,
                dense.LUT4,
            )
            planes = list(dense.pack(fmt, q, sc, minimum, gs))
            for name, plane in zip(("codes", "high", "scale"), planes):
                group[name].append(torch.from_numpy(np.ascontiguousarray(plane)).cuda())
            group["fmts"].append(fmt)
            group["ns"].append(q.shape[0])
            group["oracle"].append(
                torch.from_numpy(dense.reconstruct(fmt, q, sc, minimum, gs))
                .cuda()
                .float()
            )
            group["official"].append(
                torch.from_numpy(
                    dequantize(raw, seg["gtype"]).astype(np.float32)
                ).cuda()
            )
        n = sum(group["ns"])
        out = torch.empty(m, n, device="cuda", dtype=torch.float16)
        group["outs"] = list(out.split(group["ns"], dim=1))
        group["out"] = out
        tiles = n // 32
        group["ws"] = torch.zeros(tiles * 256, device="cuda")
        group["cnt"] = torch.zeros(tiles, dtype=torch.int32, device="cuda")
        group["part"] = torch.empty(20, m, n, device="cuda")
        group["ready"] = torch.zeros(80, dtype=torch.int32, device="cuda")
        pairs.append(group)
    del records, hc
    assert len(pairs) == args.pairs
    torch.manual_seed(20261008)
    residual = torch.randn(m, 10240, device="cuda").half()
    inj = torch.randn(m, 4, device="cuda").half()
    torch.manual_seed(20261008 + rank)
    partial = (torch.randn(m, 2560, device="cuda") * 0.5).half()
    resout = torch.empty_like(residual)
    block = torch.empty_like(partial)
    injout = torch.empty_like(inj)

    def run(g, arm):
        launch = (
            torch.ops._C.sm70_hcx_out
            if arm == "packaged"
            else torch.ops.round14_hcx_consumer.run
        )
        common = (
            partial,
            None,
            residual,
            inj,
            g["nw"],
            1e-6,
            g["wd"],
            g["wu"],
            resout,
            block,
            injout,
            rt.xn,
            rt.sq,
            rt.dpart,
            rt.bar,
            rt.seq,
            rt.ar,
            rt.lora,
            rt.hb,
            rt.logical_rank,
            None,
            int(rt.full),
            None,
            None,
            None,
            None,
            -1,
            None,
            None,
            1e-6,
            None,
        )
        if arm == "packaged":
            launch(*common)
        else:
            launch(
                *common,
                g["codes"],
                g["high"],
                g["scale"],
                g["outs"],
                g["fmts"],
                g["ns"],
                g["part"],
                g["ready"],
                arm == "pipeline",
            )
        if arm != "pipeline":
            torch.ops._C.gguf_dense_segments_sm70_out(
                block,
                g["codes"],
                g["high"],
                g["scale"],
                g["outs"],
                g["fmts"],
                g["ns"],
                2560,
                1,
                4,
                g["ws"],
                g["cnt"],
                None,
            )

    errors = []
    for g in pairs:
        run(g, "packaged")
        torch.cuda.synchronize()
        expected = [t.clone() for t in (resout, block, injout)]
        expected_proj = g["out"].clone()
        canonical = torch.cat([block.float() @ w.T for w in g["oracle"]], dim=1)
        official = torch.cat([block.float() @ w.T for w in g["official"]], dim=1)
        run(g, "pipeline")
        torch.cuda.synchronize()
        for actual, target in zip((resout, block, injout), expected):
            torch.testing.assert_close(actual, target, rtol=0, atol=0)
        scale = canonical.abs().max().clamp_min(1e-6)
        err = float((g["out"].float() - canonical).abs().max() / scale)
        assert err < 0.002, (g["kind"], err)
        errors.append(
            {
                "kind": g["kind"],
                "layer": g["layer"],
                "formats": g["fmts"],
                "pipeline_vs_canonical_rel_max": err,
                "packaged_vs_canonical_rel_max": float(
                    (expected_proj.float() - canonical).abs().max() / scale
                ),
                "pipeline_vs_packaged_rel_max": float(
                    (g["out"].float() - expected_proj.float()).abs().max() / scale
                ),
                "pipeline_vs_official_rel_max": float(
                    (g["out"].float() - official).abs().max()
                    / official.abs().max().clamp_min(1e-6)
                ),
            }
        )
    # Free FP32 numerical oracles before timings; both arms retain one set of
    # canonical projection codes and coefficients.
    for g in pairs:
        del g["oracle"], g["official"]
    graphs = {}
    for arm in ("packaged", "copy_control", "pipeline"):
        for g in pairs:
            run(g, arm)
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(args.repeats):
                for g in pairs:
                    run(g, arm)
        graphs[arm] = graph
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
    for _ in range(2):
        residual.mul_(0.75)
        inj.add_(0.0625)
        partial.add_((rank + 1) * 0.015625)
        graphs["packaged"].replay()
        torch.cuda.synchronize()
        expected = [t.clone() for t in (resout, block, injout)]
        expected_proj = pairs[-1]["out"].clone()
        for arm in ("copy_control", "pipeline"):
            graphs[arm].replay()
            torch.cuda.synchronize()
            for actual, target in zip((resout, block, injout), expected):
                torch.testing.assert_close(actual, target, rtol=0, atol=0)
            diff = float(
                (pairs[-1]["out"].float() - expected_proj.float()).abs().max()
                / expected_proj.abs().max().clamp_min(1e-6)
            )
            assert diff < 0.002, diff
    times = {arm: [] for arm in graphs}
    for arm in (
        "packaged",
        "pipeline",
        "pipeline",
        "packaged",
        "copy_control",
        "copy_control",
    ):
        for _ in range(args.samples):
            dist.barrier()
            begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            begin.record()
            graphs[arm].replay()
            end.record()
            end.synchronize()
            times[arm].append(
                begin.elapsed_time(end) * 1000 / args.repeats / len(pairs)
            )
    own = {"rank": rank, "samples_us": times, "errors": errors}
    results = [None] * 4
    dist.all_gather_object(results, own)
    if rank == 0:
        result = {
            "scope": "research-only HC-to-dense chain; private DSO; "
            "rank0 projection shards mirrored; not endpoint",
            "M": m,
            "pairs": len(pairs),
            "core_sha256": hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
            "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
            "max_rank_median_us": {
                arm: max(statistics.median(r["samples_us"][arm]) for r in results)
                for arm in graphs
            },
            "records": results,
            "hc_bytes_exact": True,
            "changed_input_graph_checks": 2,
        }
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result["max_rank_median_us"]), flush=True)
    del graphs
    torch.cuda.synchronize()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
