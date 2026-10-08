# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Research-only tile-ready MLP chain screen; never a serving extension.

Run under the shared GPU locks. The generated translation unit mechanically
reuses the current DMV device body; it does not introduce another GGUF reader.
Only a winning whole-chain result warrants adding a production operator.
"""

import argparse
import hashlib
import importlib.util
import json
import statistics
import subprocess
import uuid
from pathlib import Path

import gguf
import numpy as np
import torch
from torch.utils.cpp_extension import load


def source(root, directory, task_threads, protocol):
    header_name = (
        "gguf_tile_ready_mlp_compact_sm70.cuh"
        if task_threads == 256
        else "gguf_tile_ready_mlp_sm70.cuh"
    )
    original = (root / "csrc/sm70_turbomind/ops/gguf_dmv_sm70.cu").read_text()
    device = original[: original.index("}  // namespace")]
    marker = (
        "template <int KW, int TN, int FA, int FB, int FC = -1, bool Clocked = false>"
    )
    start = device.index(marker + "\n__global__")
    end = device.index(marker + "\nvoid launch", start)
    body = device[start:end]
    body = body.replace(
        "__global__ void __launch_bounds__(32 * KW * TN)\n    dense_mv",
        "__device__ __forceinline__ void dense_tile",
    )
    body = body.replace(
        "uint64_t* timestamps) {", "uint64_t* timestamps, int tg, int sp) {"
    )
    body = body.replace("  WarpTimer<Clocked> timer(timestamps, W);\n", "")
    body = body.replace("blockIdx.x", "tg").replace("blockIdx.y", "sp")
    body = body.replace("dense_tile(Segs segs,", "dense_tile(const Segs& segs,")
    body = body.replace("  const int tg = tg, sp = sp;\n", "")
    # The persistent block has 16 warps; down retains its original eight
    # compute warps and original four-way FP32 reduction. Extra warps attend
    # block barriers but perform no matrix work or activation accesses.
    body = body.replace("tin, g0, g1,", "tin, g0, warp < W ? g1 : 0,")
    # Keep the control kernel byte-for-byte from the production source.
    # Factoring its body into a device call can change register allocation;
    # that would manufacture a different, slower baseline.
    if "__device__ __forceinline__ void dense_tile" not in device:
        device = device[:start] + body + device[start:]
    begin = device.index("template <int FMT, int KW, int TN, bool LegacyIq2 = false>")
    finish = device.index("__device__ __forceinline__ void write_out", begin)
    streamed = device[begin:finish].replace("void body6(", "void ready_body6(")
    streamed = streamed.replace(
        "bool gdn_heads, bool pair) {",
        "bool gdn_heads, bool pair, int* ready, int generation) {",
    )
    streamed = streamed.replace(
        "  auto xload = [&](int gg) {",
        """  auto xload = [&](int gg) {
    // A K128 activation chunk is produced by two paired N64 tiles.
    // Immutable weights have already been loaded before entering this wait.
    if (lane == 0) {
      while (atomicAdd(ready + gg * 2, 0) != generation) {}
      while (atomicAdd(ready + gg * 2 + 1, 0) != generation) {}
    }
    __syncwarp();""",
    )
    # Read producer-written activations through coherent L2, not __ldg's
    # read-only cache. All other arithmetic and reduction orders are retained.
    streamed = streamed.replace("X[jj] = __ldg(", "X[jj] = __ldcg(")
    if task_threads == 256:
        streamed = streamed.replace(
            "while (atomicAdd(ready + gg * 2, 0) != generation) {}\n"
            "      while (atomicAdd(ready + gg * 2 + 1, 0) != generation) {}",
            "for (int producer = gg * 4; producer < gg * 4 + 4; ++producer) {\n"
            "        while (atomicAdd(ready + producer, 0) != generation) {}\n"
            "      }",
        )
    if protocol == "acquire":
        streamed = streamed.replace(
            "atomicAdd(ready + gg * 2, 0)", "read_ready(ready + gg * 2)"
        )
        streamed = streamed.replace(
            "atomicAdd(ready + gg * 2 + 1, 0)", "read_ready(ready + gg * 2 + 1)"
        )
        streamed = streamed.replace(
            "atomicAdd(ready + producer, 0)", "read_ready(ready + producer)"
        )
        device += """
__device__ __forceinline__ int read_ready(const int* flag) {
  int value;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
               : "=r"(value) : "l"(flag) : "memory");
  return value;
}
__device__ __forceinline__ void publish_ready(int* flag, int generation) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;"
               :: "l"(flag), "r"(generation) : "memory");
}
"""
    ready_tile = body.replace("void dense_tile", "void ready_tile")
    ready_tile = ready_tile.replace(
        "int tg, int sp) {", "int tg, int sp, int* ready, int generation) {"
    )
    ready_tile = ready_tile.replace("body6<", "ready_body6<")
    # This down-only task has no output gate. Statically remove the optional
    # gate load from its inlined epilogue rather than carrying a nullable input.
    ready_tile = ready_tile.replace("segs.sgate", "nullptr")
    ready_tile = ready_tile.replace(
        "                                  pair);",
        "                                  pair, ready, generation);",
    )
    ready_tile = ready_tile.replace(
        " M,\n"
        "                               xs, lut, acc, segs.tab, segs.gdn_heads, pair);",
        " M,\n"
        "                               xs, lut, acc, segs.tab, segs.gdn_heads, "
        "pair, ready, generation);",
    )
    device += streamed + ready_tile
    device += "}  // namespace\n"
    header = (root / "benchmarks/csrc" / header_name).read_text()
    if protocol == "acquire":
        header = header.replace(
            "atomicExch(ready + t, generation)", "publish_ready(ready + t, generation)"
        )
    target = directory / "persistent_mlp.cu"
    content = device + "\n" + header
    if not target.exists() or target.read_text() != content:
        target.write_text(content)
    return target


def pack(root, raw, kind):
    directory = root / "vllm/model_executor/layers/quantization"
    modules = []
    for name in ("gguf_dmv_formats", "gguf_dense_hmma_formats"):
        spec = importlib.util.spec_from_file_location(name, directory / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    iq, affine = modules
    if kind == 21:
        fmt, codes, scales = iq.pack(raw, kind)
    else:
        fmt, q, scale, minimum, group = affine.decode(raw, kind)
        codes, _, scales = affine.pack(fmt, q, scale, minimum, group)
        if fmt == 3:
            scales = iq.compact_lut4_scale(scales)
    return fmt, codes, scales, iq.tables()


def worker(rank, args, generated):
    torch.accelerator.set_device_index(rank)
    extension = load(
        name="gguf_tile_ready_mlp_research",
        sources=[str(generated)],
        build_directory=str(args.output / "build"),
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
            "--ptxas-options=-v",
            "-gencode=arch=compute_70,code=sm_70",
        ],
        verbose=False,
    )
    reader = gguf.GGUFReader(str(args.model))
    tensors = {t.name: t for t in reader.tensors}
    cases = []
    for layer in (6, 24, 51):
        planes, references, kinds = [], [], []
        for role in ("ffn_gate", "ffn_up", "ffn_down"):
            tensor = tensors[f"blk.{layer}.{role}.weight"]
            kind = int(tensor.tensor_type)
            k, n = map(int, tensor.shape)
            block, size = gguf.GGML_QUANT_SIZES[tensor.tensor_type]
            raw = tensor.data.reshape(n, k // block, size)
            if role == "ffn_down":
                raw = raw[:, rank * (k // block // 4) : (rank + 1) * (k // block // 4)]
            else:
                raw = raw[rank * (n // 4) : (rank + 1) * (n // 4)]
            raw = np.ascontiguousarray(raw).reshape(raw.shape[0], -1)
            fmt, codes, scales, table = pack(args.root, raw, kind)
            planes.append(
                (fmt, torch.from_numpy(codes).cuda(), torch.from_numpy(scales).cuda())
            )
            references.append(
                torch.from_numpy(gguf.quants.dequantize(raw, tensor.tensor_type)).cuda()
            )
            kinds.append(kind)
        tab = torch.from_numpy(table).cuda()
        codes, scales = [p[1] for p in planes], [p[2] for p in planes]
        x = torch.randn(8, 5120, device=rank, dtype=torch.float16)
        h = torch.empty(8, 4352, device=rank, dtype=torch.float16)
        out = torch.empty_like(x)
        epochs = torch.zeros(
            160 if args.task_threads == 256 else 80, device=rank, dtype=torch.int32
        )
        ready = torch.zeros(
            136 if args.task_threads == 256 else 68, device=rank, dtype=torch.int32
        )
        errors = []
        for amplitude in (0.125, 0.5, 1.0):
            x.normal_().mul_(amplitude)
            extension.run(
                x, codes, scales, tab, h, out, planes[-1][0], epochs, ready, False
            )
            torch.accelerator.synchronize()
            expected_h, expected_out = h.clone(), out.clone()
            extension.run(
                x, codes, scales, tab, h, out, planes[-1][0], epochs, ready, True
            )
            torch.accelerator.synchronize()
            assert torch.equal(h, expected_h), (rank, layer, "gate")
            assert torch.equal(out, expected_out), (rank, layer, "down")
            gate = (x.float() @ references[0].float().T).half().float()
            up = (x.float() @ references[1].float().T).half().float()
            oracle_h = (torch.nn.functional.silu(gate) * up).half()
            oracle = (oracle_h.float() @ references[2].float().T).half()
            relative = float(
                (out.float() - oracle.float()).norm() / oracle.float().norm()
            )
            assert torch.isfinite(out).all() and relative < 0.01, relative
            errors.append(relative)
        if args.numeric_only:
            cases.append(
                dict(
                    layer=layer,
                    rank=rank,
                    bitwise_equal=True,
                    official_relative_l2=errors,
                )
            )
            continue
        # Rotating distinct banks exceeds Volta L2. This is a whole-chain
        # operator test, not a model round or communication measurement.
        banks = [
            (
                x.clone(),
                [c.clone() for c in codes],
                [s.clone() for s in scales],
                torch.empty_like(h),
                torch.empty_like(out),
            )
            for _ in range(8)
        ]
        graphs = []
        for persistent in (False, True):
            graph = torch.cuda.CUDAGraph()

            def call(
                banks=banks,
                tab=tab,
                fmt=planes[-1][0],
                persistent=persistent,
                epochs=epochs,
                ready=ready,
            ):
                for bx, bc, bs, bh, bo in banks:
                    extension.run(
                        bx, bc, bs, tab, bh, bo, fmt, epochs, ready, persistent
                    )

            call()
            torch.accelerator.synchronize()
            with torch.cuda.graph(graph):
                call()
            graphs.append(graph)
        torch.distributed.barrier()
        samples = {False: [], True: []}
        for arm in (False, True, True, False) * 5:
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
        # Captured cooperative replays must remain exact after changed inputs.
        for _ in range(20):
            for bx, *_ in banks:
                bx.normal_().mul_(0.5)
            graphs[0].replay()
            expected = [(bh.clone(), bo.clone()) for _, _, _, bh, bo in banks]
            graphs[1].replay()
            torch.accelerator.synchronize()
            for (_, _, _, bh, bo), (eh, eo) in zip(banks, expected):
                assert torch.equal(bh, eh) and torch.equal(bo, eo)
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
                layer=layer,
                source_types=kinds,
                rank=rank,
                bitwise_equal=True,
                changed_input_graph_replays=20,
                official_relative_l2=errors,
                separate_us=statistics.mean(samples[False]),
                persistent_us=statistics.mean(samples[True]),
                samples_us={str(k): v for k, v in samples.items()},
                clocks=clocks,
            )
        )
        del graphs, banks, references, codes, scales
        torch.accelerator.memory.empty_cache()
    (args.output / f"rank{rank}.json").write_text(
        json.dumps(
            dict(
                research_only=True,
                task_threads=args.task_threads,
                task_protocol=args.task_protocol,
                serving_runtime=False,
                torch=torch.__version__,
                source_sha=args.source_sha,
                generated_cuda_sha256=hashlib.sha256(
                    generated.read_bytes()
                ).hexdigest(),
                cases=cases,
            ),
            indent=2,
        )
        + "\n"
    )
    torch.distributed.destroy_process_group()


def launch_worker(rank, args, generated):
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{args.rendezvous}",
        rank=rank,
        world_size=args.ranks,
    )
    worker(rank, args, generated)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-protocol", choices=("atomic", "acquire"), default="atomic"
    )
    parser.add_argument("--task-threads", type=int, choices=(256, 512), default=512)
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--numeric-only", action="store_true")
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.rendezvous = args.output / f"rendezvous-{uuid.uuid4().hex}"
    (args.output / "build").mkdir(exist_ok=True)
    generated = source(args.root, args.output, args.task_threads, args.task_protocol)
    load(
        name="gguf_tile_ready_mlp_research",
        sources=[str(generated)],
        build_directory=str(args.output / "build"),
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
            "--ptxas-options=-v",
            "-gencode=arch=compute_70,code=sm_70",
        ],
        verbose=True,
    )
    torch.multiprocessing.spawn(
        launch_worker, args=(args, generated), nprocs=args.ranks
    )
