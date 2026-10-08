# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Research-only projection/TP/norm chain, real shards and packet-order checks."""

import argparse
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

import gguf
import numpy as np
import torch
from benchmark_gguf_persistent_mlp import pack, source
from torch.utils.cpp_extension import load

FLAGS = ["-O3", "-lineinfo", "-gencode=arch=compute_70,code=sm_70"]


def norm_error(actual, expected):
    def ordered(tensor):
        bits = tensor.view(torch.int16).int()
        return torch.where(bits < 0, -32768 - bits, bits)

    relative = float(
        (actual.float() - expected.float()).norm() / expected.float().norm()
    )
    ulp = int((ordered(actual) - ordered(expected)).abs().max())
    assert torch.isfinite(actual).all() and ulp == 0 and relative == 0, (
        relative,
        ulp,
    )
    return relative, ulp


def extension(args, generated):
    return load(
        name="gguf_collective_pipeline_research",
        sources=[str(generated)],
        build_directory=str(args.output / "build"),
        extra_cuda_cflags=FLAGS,
        extra_include_paths=[str(args.root / "csrc")],
        verbose=True,
    )


def worker(rank, args, generated):
    torch.accelerator.set_device_index(rank)
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{args.output / 'rendezvous'}",
        rank=rank,
        world_size=4,
    )
    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce

    mod = extension(args, generated)
    sizes = mod.sizes()
    peer_buffers = [CustomAllreduce.create_shared_buffer(sizes[0])]
    for i, pointers in enumerate(peer_buffers):
        mod.init(pointers[rank], sizes[i], i == 0)
    torch.accelerator.synchronize()
    torch.distributed.barrier()
    reader = gguf.GGUFReader(str(args.model))
    tensors = {t.name: t for t in reader.tensors}
    cases = []
    torch.manual_seed(123)
    residual = torch.randn(8, 5120, device=rank, dtype=torch.float32)
    weight = (
        torch.randn(5120, device=rank, dtype=getattr(torch, args.norm_dtype)) * 0.01
    )
    for layer in (6, 24, 51):
        tensor = tensors[f"blk.{layer}.ffn_down.weight"]
        k, n = map(int, tensor.shape)
        block, size = gguf.GGML_QUANT_SIZES[tensor.tensor_type]
        raw = tensor.data.reshape(n, k // block, size)[
            :, rank * (k // block // 4) : (rank + 1) * (k // block // 4)
        ]
        raw = np.ascontiguousarray(raw).reshape(n, -1)
        fmt, codes, scales, table = pack(args.root, raw, int(tensor.tensor_type))
        codes, scales, table = [
            torch.from_numpy(a).cuda() for a in (codes, scales, table)
        ]
        x = torch.randn(8, 4352, device=rank, dtype=torch.float16) * 0.5
        partial = torch.empty(8, 5120, device=rank, dtype=torch.float16)
        normalized = torch.empty_like(partial)
        res_out = torch.empty_like(residual)

        def call(
            fused,
            x=x,
            codes=codes,
            scales=scales,
            table=table,
            partial=partial,
            normalized=normalized,
            res_out=res_out,
            fmt=fmt,
        ):
            mod.run(
                x,
                codes,
                scales,
                table,
                partial,
                peer_buffers[0],
                rank,
                residual,
                weight,
                normalized,
                res_out,
                fmt,
                fused,
            )

        call(False)
        torch.accelerator.synchronize()
        expected_partial, expected_norm, expected_res = (
            partial.clone(),
            normalized.clone(),
            res_out.clone(),
        )
        call(True)
        torch.accelerator.synchronize()
        assert torch.equal(partial, expected_partial), (rank, layer, "partial")
        assert torch.equal(res_out, expected_res), (rank, layer, "residual")
        norm_relative, norm_ulp = norm_error(normalized, expected_norm)
        # Distinct banks overrun L2; every arm includes projection + real TP4
        # allreduce + add/RMSNorm. The control uses the unchanged current kernels.
        banks = [
            (
                x.clone(),
                codes.clone(),
                scales.clone(),
                torch.empty_like(partial),
                torch.empty_like(normalized),
                torch.empty_like(res_out),
            )
            for _ in range(8)
        ]
        graphs = []
        for fused in (False, True):

            def chain(fused=fused, banks=banks, table=table, fmt=fmt):
                for bx, bc, bs, bp, bn, br in banks:
                    mod.run(
                        bx,
                        bc,
                        bs,
                        table,
                        bp,
                        peer_buffers[0],
                        rank,
                        residual,
                        weight,
                        bn,
                        br,
                        fmt,
                        fused,
                    )

            chain()
            torch.accelerator.synchronize()
            torch.distributed.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                chain()
            graphs.append(graph)
        samples = {False: [], True: []}
        for arm in (False, True, True, False) * 5:
            torch.distributed.barrier()
            graph = graphs[int(arm)]
            for _ in range(10):
                graph.replay()
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            for _ in range(50):
                graph.replay()
            end.record()
            end.synchronize()
            samples[arm].append(start.elapsed_time(end) * 1000 / 400)
        replay_relative = []
        replay_ulp = []
        for _ in range(20):
            for bx, *_ in banks:
                bx.normal_().mul_(0.5)
            graphs[0].replay()
            expected = [
                (bp.clone(), bn.clone(), br.clone()) for _, _, _, bp, bn, br in banks
            ]
            graphs[1].replay()
            torch.accelerator.synchronize()
            for (_, _, _, bp, bn, br), (ep, en, er) in zip(banks, expected):
                assert torch.equal(bp, ep) and torch.equal(br, er)
                error, ulp = norm_error(bn, en)
                replay_relative.append(error)
                replay_ulp.append(ulp)
        clocks = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={rank}",
                "--query-gpu=clocks.sm,clocks.mem",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        cases.append(
            dict(
                rank=rank,
                layer=layer,
                format=fmt,
                norm_dtype=args.norm_dtype,
                partial_bitwise=True,
                residual_bitwise=True,
                norm_relative_l2=norm_relative,
                norm_max_ulp=norm_ulp,
                graph_replays=20,
                max_graph_norm_relative_l2=max(replay_relative),
                max_graph_norm_ulp=max(replay_ulp),
                control_us=statistics.mean(samples[False]),
                pipeline_us=statistics.mean(samples[True]),
                samples_us={str(k): v for k, v in samples.items()},
                clocks=clocks,
            )
        )
        print(json.dumps(cases[-1]), flush=True)
        del graphs, banks
    torch.distributed.barrier()
    for pointers in peer_buffers:
        CustomAllreduce.free_shared_buffer(pointers, rank=rank)
    (args.output / f"rank{rank}.json").write_text(
        json.dumps(
            dict(
                research_only=True,
                serving_runtime=False,
                shared_production_packet_buffer=True,
                cases=cases,
                torch=torch.__version__,
                cuda_sha256=hashlib.sha256(generated.read_bytes()).hexdigest(),
            ),
            indent=2,
        )
        + "\n"
    )
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--norm-dtype", choices=("float16", "float32"), default="float16"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "build").mkdir(exist_ok=True)
    generated = source(
        args.root, args.output, "gguf_projection_collective_pipeline_sm70.cuh"
    )
    extension(args, generated)
    torch.multiprocessing.spawn(worker, args=(args, generated), nprocs=4)
