# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check same-graph q8 norm/packed-column switches against frozen libraries.

Inputs are four consecutive layers of real prepared weights on each of four
ranks. No model route is installed. --timing measures the column/norm working
set only, excluding row projections, communication and the complete round.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from benchmarks.kernels.sm70_qpn2_graph_layout import LayoutTemplates
from benchmarks.kernels.sm70_qpn2_graph_nodes import nodes
from vllm.model_executor.layers import layernorm as norm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--column-library", type=Path, required=True)
    p.add_argument("--packed-library", type=Path, required=True)
    p.add_argument("--dual-norm", type=Path, required=True)
    p.add_argument("--timing", action="store_true")
    p.add_argument("--cycles", type=int, default=9)
    args = p.parse_args()
    if args.cycles < 1:
        p.error("--cycles must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank)
    assert torch.cuda.get_device_capability() == (7, 0)
    libraries = {"column": args.column_library, "packed": args.packed_library}
    for path in libraries.values():
        torch.ops.load_library(str(path))
    spec = importlib.util.spec_from_file_location("benchmark_dual_norm", args.dual_norm)
    dual = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dual)
    dist.init_process_group("gloo")
    assert dist.get_world_size() == 4
    generator = torch.Generator(device="cuda").manual_seed(20260909 + rank)
    items = []
    provenance = []
    for path in sorted((args.root / f"rank{rank}").glob("*.pt")):
        item = torch.load(path, map_location="cpu", weights_only=True)
        if item["k"] != 5120:
            continue
        provenance.append(
            dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        )
        item["codes"] = item["codes"].cuda()
        item["scales"] = item["scales"].cuda()
        items.append(item)
    assert len(items) == 8
    templates = LayoutTemplates(items, dual)
    report = dict(
        rank=rank,
        templates=templates.describe(),
        provenance=provenance,
        libraries={
            k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in libraries.items()
        },
        cases=[],
    )
    print(json.dumps(dict(rank=rank, templates=templates.describe())), flush=True)

    def make(rows, kind):
        records = []
        for item in items:
            x = torch.randn(
                (rows, 5120), device="cuda", dtype=torch.float16, generator=generator
            )
            weight = torch.randn(
                (5120,), device="cuda", dtype=torch.float16, generator=generator
            )
            residual = (
                None
                if kind == "none"
                else torch.randn(
                    (rows, 5120),
                    device="cuda",
                    dtype=torch.float16 if kind == "half" else torch.float32,
                    generator=generator,
                )
            )
            out = torch.empty_like(x)
            resout = (
                None if residual is None else torch.empty_like(x, dtype=torch.float32)
            )
            result = torch.empty(
                (rows, item["n"] // (2 if item["gated"] else 1)),
                device="cuda",
                dtype=torch.float16,
            )
            records.append((item, x, weight, residual, out, resout, result))

        def run():
            for item, x, weight, residual, out, resout, result in records:
                if kind == "float":
                    norm._sm70_dflash2_gemma_fused_add_rms_kernel[(rows,)](
                        x,
                        residual,
                        weight,
                        out,
                        resout,
                        hidden_size=5120,
                        BLOCK_SIZE=8192,
                        epsilon=1e-6,
                        num_warps=8,
                        num_stages=1,
                    )
                else:
                    norm._sm70_dflash2_fixed_gemma_rms_kernel[(rows,)](
                        x,
                        residual,
                        weight,
                        out,
                        resout,
                        HAS_RESIDUAL=residual is not None,
                        epsilon=1e-6,
                        num_warps=16,
                        num_stages=1,
                        enable_fp_fusion=True,
                    )
                if rows > 8:
                    continue  # Frozen raw QPN2 rejects larger rows; norm-only fallback.
                namespace = torch.ops._qpn2_capped if rows == 8 else torch.ops._C
                if rows == 8:
                    op = namespace.gated if item["gated"] else namespace.gemm
                else:
                    op = (
                        namespace.nvfp4_qpn2_gated_sm70_out
                        if item["gated"]
                        else namespace.nvfp4_qpn2_gemm_sm70_out
                    )
                op(
                    result,
                    out,
                    item["codes"],
                    item["scales"],
                    item["global_scale"],
                    item["split_k"],
                    item["nacc"],
                )

        run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            run()
        graph.instantiate()
        return graph, records

    def fingerprint(graph):
        kernels, parents = nodes(graph)
        data = [
            (
                key,
                n.name,
                n.params.func,
                n.grid,
                [a.hex() for a in n.args],
                parents[key],
            )
            for key, n in sorted(kernels.items())
        ]
        return hashlib.sha256(json.dumps(data).encode()).hexdigest()

    count = 0
    for kind in ("none", "half", "float"):
        graph, records = make(8, kind)
        edit = templates.patch(graph)
        assert len(edit.edits) == 16 and len(edit.norms) == 8
        raw_sha = fingerprint(graph)
        for cycle in range(args.cycles):
            scale = (0.01, 1.0, 8.0)[cycle % 3]
            for item, x, weight, residual, out, resout, result in records:
                x.copy_(
                    torch.randn(
                        x.shape, device="cuda", dtype=x.dtype, generator=generator
                    )
                    * scale
                )
                if residual is not None:
                    residual.copy_(
                        torch.randn(
                            residual.shape,
                            device="cuda",
                            dtype=residual.dtype,
                            generator=generator,
                        )
                        * scale
                    )
            saved = []
            for mode in ("control", "candidate", "control"):
                edit.switch(mode)
                graph.replay()
                torch.cuda.synchronize()
                outputs = [t for rec in records for t in rec[4:] if t is not None]
                if not saved:
                    saved = [t.clone() for t in outputs]
                else:
                    for index, (reference, actual) in enumerate(zip(saved, outputs)):
                        assert torch.equal(
                            reference.view(torch.uint8), actual.view(torch.uint8)
                        ), (
                            rank,
                            kind,
                            cycle,
                            mode,
                            index,
                            int(
                                (
                                    reference.view(torch.uint8)
                                    != actual.view(torch.uint8)
                                ).sum()
                            ),
                        )
                assert all(bool(torch.isfinite(t).all()) for t in outputs)
                edit.check_canaries()
                assert fingerprint(graph) == raw_sha, (
                    "Executable edit changed raw graph"
                )
                if mode == "candidate":
                    by_ptr = {r[4].data_ptr(): r[4] for r in records}
                    for original, _ in edit.edits:
                        if original.handle not in edit.norms:
                            continue
                        key = (original.name, tuple(map(len, original.args)))
                        idx = templates.norms[key][3]
                        logical = by_ptr[original.integer(idx)]
                        packed = (
                            edit.norms[original.handle][0]
                            .view(320, 8, 16)
                            .permute(1, 0, 2)
                            .contiguous()
                            .view_as(logical)
                        )
                        assert torch.equal(
                            logical.view(torch.uint8), packed.view(torch.uint8)
                        ), "Packed norm mismatch"
            count += len(records)
            report["cases"].append(
                dict(
                    kind=kind,
                    cycle=cycle,
                    scale=scale,
                    outputs_state_bits_equal=True,
                    restore_equal=True,
                    canaries=True,
                )
            )
    timing = None
    if args.timing:
        samples = {"control": [], "candidate": []}
        for trial in range(7):
            for mode in (
                ("control", "candidate") if trial % 2 == 0 else ("candidate", "control")
            ):
                edit.switch(mode)
                for _ in range(8):
                    graph.replay()
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                for _ in range(64):
                    graph.replay()
                end.record()
                end.synchronize()
                samples[mode].append(start.elapsed_time(end) / 64)
        timing = dict(
            samples_ms=samples,
            scope=(
                "eight real column weights plus FP32-residual norms; "
                "no row projections, communication or model timing"
            ),
        )
    fallbacks = []
    for rows in (1, 7, 9, 32):
        graph, records = make(rows, "float")
        edit = templates.patch(graph)
        assert len(edit.edits) == 0
        fallbacks.append(
            dict(
                rows=rows,
                scope="norm+raw QPN2"
                if rows <= 8
                else "norm only; raw QPN2 correctly rejects M>8",
            )
        )
    matrix_a = torch.randn(
        (8, 128), device="cuda", dtype=torch.float16, generator=generator
    )
    matrix_b = torch.randn(
        (128, 256), device="cuda", dtype=torch.float16, generator=generator
    )
    matrix_out = torch.empty((8, 256), device="cuda", dtype=torch.float16)
    torch.mm(matrix_a, matrix_b, out=matrix_out)
    torch.cuda.synchronize()
    unrelated = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(unrelated):
        torch.mm(matrix_a, matrix_b, out=matrix_out)
    unrelated.instantiate()
    unrelated.replay()
    torch.cuda.synchronize()
    expected_matrix = matrix_out.clone()
    untouched = templates.patch(unrelated)
    assert len(untouched.edits) == 0
    untouched.switch("candidate")
    unrelated.replay()
    torch.cuda.synchronize()
    assert torch.equal(matrix_out.view(torch.uint8), expected_matrix.view(torch.uint8))
    report["unrelated_cublas_graph_unmodified"] = True
    report.update(
        timing=timing,
        source_hashes={
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                Path(__file__).with_name("sm70_qpn2_graph_layout.py"),
                Path(__file__).with_name("sm70_qpn2_graph_nodes.py"),
                args.dual_norm,
            )
        },
        passed=True,
        projection_cases=count,
        switch_replays=len(report["cases"]) * 3,
        unmodified_rows=fallbacks,
    )
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, report)
    if rank == 0:
        args.output.write_text(
            json.dumps(
                dict(
                    passed=True,
                    ranks=gathered,
                    scope="same captured operator graph; not model or timing evidence",
                ),
                indent=2,
            )
            + "\n"
        )
    dist.destroy_process_group()
    print(json.dumps(dict(rank=rank, passed=True, projection_cases=count)), flush=True)


if __name__ == "__main__":
    main()
