# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Research-only AR+norm overlap with the next M8 gate/up weight reads."""

import argparse
import hashlib
import importlib.util
import json
import statistics
import time
import uuid
from pathlib import Path

import gguf
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load

from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce


def pack(root, raw):
    path = root / "vllm/model_executor/layers/quantization/gguf_dmv_formats.py"
    spec = importlib.util.spec_from_file_location("gguf_norm_weight_formats", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fmt, codes, scales = module.pack(raw, 21)
    assert fmt == 5
    return codes, scales, module.tables()


def source(root, output, unconditional=False):
    original = (root / "csrc/sm70_turbomind/ops/gguf_dmv_sm70.cu").read_text()
    device = original[: original.index("}  // namespace")]
    begin = device.index("template <int FMT, int KW, int TN, bool LegacyIq2 = false>")
    end = device.index("__device__ __forceinline__ void write_out", begin)
    body = device[begin:end].replace("void body6(", "void ready_body6(")
    body = body.replace(
        "bool gdn_heads, bool pair) {", "bool gdn_heads, bool pair, const int* ready) {"
    )
    # Preserve load/MMA order, but fetch the first two packets before input is ready.
    body = body.replace("  Ld<FMT> A;", "  Ld<FMT> A, B;")
    body = body.replace(
        "__ldg(reinterpret_cast<const uint4*>(",
        "__ldcg(reinterpret_cast<const uint4*>(",
    )
    body = body.replace(
        "    xload(g);",
        "    if (g + stride < end && on) load<FMT>(B, sg, t, g + stride, S, G, lane);",
    )
    needle = "  __syncthreads();\n  if (g >= end) return;"
    body = body.replace(
        needle,
        """  __syncthreads();
  if (threadIdx.x == 0) {
    int value;
    do {
      asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(value)
                   : "l"(ready) : "memory");
    } while (value == 0);
  }
  __syncthreads();
  if (g < end) xload(g);
  if (g >= end) return;""",
    )
    body = body.replace("    Ld<FMT> B;\n", "")
    body = body.replace("      if (on) load<FMT>(B, sg, t, gn, S, G, lane);\n", "")
    body = body.replace(
        "    g = gn;",
        "    g = gn;\n"
        "    if (g + stride < end && on) load<FMT>(B, sg, t, g + stride, S, G, lane);",
    )
    marker = (
        "template <int KW, int TN, int FA, int FB, int FC = -1, bool Clocked = false>"
    )
    begin = device.index(marker + "\n__global__")
    end = device.index(marker + "\nvoid launch", begin)
    kernel = device[begin:end].replace("dense_mv(Segs", "ready_dense_mv(Segs")
    kernel = kernel.replace(
        "uint64_t* timestamps) {", "uint64_t* timestamps, const int* ready) {"
    )
    kernel = kernel.replace("body6<", "ready_body6<")
    kernel = kernel.replace(
        "                                  pair);",
        "                                  pair, ready);",
    )
    kernel = kernel.replace(
        "                               xs, lut, acc, segs.tab, segs.gdn_heads, pair);",
        "                               xs, lut, acc, segs.tab, "
        "segs.gdn_heads, pair, ready);",
    )
    # Cache the complete M8 activation tile once. Padding a row by one
    # uint4 keeps distinct MMA rows in distinct shared-memory banks.
    cached_body = r"""
template <int FMT, int KW, int TN, bool LegacyIq2 = false>
__device__ __forceinline__ void cached_body6(
    const Seg& sg, int t, bool on, int kslot, int pid, int g0, int g1, int S,
    int G, const half* __restrict__ x, int ldx, int M, uint4* xs, half2* lut,
    float (&acc)[8], const uint4* __restrict__ tab, bool gdn_heads, bool pair) {
  constexpr int NT = 32 * KW * TN;
  const int lane = threadIdx.x % 32;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  constexpr int VectorsPerRow = 640, SharedStride = 641;
  int g = g0 + kslot;
  Ld<FMT> A;
  if (g < g1 && on) load<FMT>(A, sg, t, g, S, G, lane);
  for (int i = threadIdx.x; i < 8 * VectorsPerRow; i += NT) {
    const int row = i / VectorsPerRow, col = i % VectorsPerRow;
    xs[row * SharedStride + col] = __ldcg(
        reinterpret_cast<const uint4*>(x + row * ldx) + col);
  }
  if (tab != nullptr) {
    uint4* t4 = reinterpret_cast<uint4*>(lut);
    for (int i = threadIdx.x; i < TAB_VECS; i += NT) t4[i] = __ldg(tab + i);
  }
  __syncthreads();
  if (g >= g1) return;
  while (true) {
    Ld<FMT> B;
    const int gn = g + KW;
    const bool more = gn < g1;
    if (more && on) load<FMT>(B, sg, t, gn, S, G, lane);
    if (on) {
#pragma unroll
      for (int st = 0; st < 4; ++st) {
        const uint4* xp = xs + r * SharedStride + g * 16 + st * 4;
        uint4 xa[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) xa[j] = xp[j];
        uint32_t hw[16];
        decode<FMT, LegacyIq2>(A, st, hw, lut);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          mma(acc, xa[j].x, xa[j].y, hw[4 * j], hw[4 * j + 1]);
          mma(acc, xa[j].z, xa[j].w, hw[4 * j + 2], hw[4 * j + 3]);
        }
      }
    }
    if (!more) break;
    A = B;
    g = gn;
  }
}
"""
    if unconditional:
        cached_body = (
            cached_body.replace(
                "if (g < g1 && on) load<FMT>(A, sg, t, g, S, G, lane);",
                "load<FMT>(A, sg, t, g, S, G, lane);",
            )
            .replace(
                "if (more && on) load<FMT>(B, sg, t, gn, S, G, lane);",
                "load<FMT>(B, sg, t, more ? gn : g, S, G, lane);",
            )
            .replace("    if (on) {", "    {")
        )
    cached_kernel = (
        device[begin:end]
        .replace("dense_mv(Segs", "cached_dense_mv(Segs")
        .replace("body6<", "cached_body6<")
        .replace(
            "reinterpret_cast<half2*>(smem + KW * 256)",
            "reinterpret_cast<half2*>(smem + 8 * 641)",
        )
    )
    header = (root / "benchmarks/csrc/gguf_norm_next_projection_sm70.cuh").read_text()
    generated = output / "norm_next_projection.cu"
    content = (
        device
        + body
        + kernel
        + cached_body
        + cached_kernel
        + "}  // namespace\n"
        + header
    )
    if not generated.exists() or generated.read_text() != content:
        generated.write_text(content)
    return generated


def worker(rank, args, generated):
    torch.accelerator.set_device_index(rank)
    dist.init_process_group(
        "gloo", init_method=f"file://{args.rendezvous}", rank=rank, world_size=4
    )
    extension = load(
        name="gguf_norm_next_projection_research",
        sources=[str(generated)],
        build_directory=str(args.output / "build"),
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
            "--ptxas-options=-v",
            "-gencode=arch=compute_70,code=sm_70",
        ],
    )
    ca = CustomAllreduce(group=dist.group.WORLD, device=rank)
    assert not ca.disabled and ca.fully_connected
    reader = gguf.GGUFReader(str(args.model))
    tensors = {t.name: t for t in reader.tensors}
    chain = []
    for layer in (6, 23, 24, 51):
        codes, scales = [], []
        for role in ("ffn_gate", "ffn_up"):
            tensor = tensors[f"blk.{layer}.{role}.weight"]
            k, n = map(int, tensor.shape)
            assert int(tensor.tensor_type) == 21 and k == 5120
            raw = np.ascontiguousarray(
                tensor.data.reshape(n, k // 256, 110)[
                    rank * (n // 4) : (rank + 1) * (n // 4)
                ]
            ).reshape(n // 4, -1)
            code, scale, table = pack(args.root, raw)
            codes.append(torch.from_numpy(code).cuda())
            scales.append(torch.from_numpy(scale).cuda())
        table = torch.from_numpy(table).cuda()
        inp = torch.randn(8, 5120, device=rank, dtype=torch.float16) * 0.03
        residual = torch.randn(8, 5120, device=rank, dtype=torch.float32) * 0.03
        norm_tensor = tensors[f"blk.{layer}.post_attention_norm.weight"]
        norm_weight = (
            torch.from_numpy(
                gguf.quants.dequantize(norm_tensor.data, norm_tensor.tensor_type).copy()
            )
            .cuda()
            .half()
            .reshape(-1)
        )
        norm_weight = norm_weight - 1
        out = torch.empty(8, 4352, device=rank, dtype=torch.float16)
        ready = torch.zeros(1, device=rank, dtype=torch.int32)
        chain.append(
            dict(
                inp=inp,
                residual=residual,
                weight=norm_weight,
                codes=codes,
                scales=scales,
                table=table,
                out=out,
                ready=ready,
            )
        )
    auxiliary = torch.cuda.Stream()
    main = torch.cuda.current_stream()
    auxiliary.wait_stream(main)
    main.wait_stream(auxiliary)

    arms = (0, 3) if args.activation_cache else (0, 1, 2)

    def call(arm):
        for case in chain:
            if arm == 1:
                extension.publish(case["ready"], False)
                auxiliary.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(auxiliary):
                    norm, res = ca.sm70_tp4_all_reduce_gemma_rms_norm(
                        case["inp"], case["residual"], case["weight"], 1e-6
                    )
                    extension.publish(case["ready"], True)
                extension.run(
                    norm,
                    case["codes"],
                    case["scales"],
                    case["table"],
                    case["out"],
                    case["ready"],
                    True,
                )
                torch.cuda.current_stream().wait_stream(auxiliary)
            else:
                norm, res = ca.sm70_tp4_all_reduce_gemma_rms_norm(
                    case["inp"], case["residual"], case["weight"], 1e-6
                )
                if arm == 2:
                    extension.publish(case["ready"], True)
                extension.run(
                    norm,
                    case["codes"],
                    case["scales"],
                    case["table"],
                    case["out"],
                    case["ready"],
                    2 if arm == 3 else int(arm == 2),
                )
            case["norm"] = norm
            case["res_out"] = res

    def snapshots():
        return [
            [case[key].clone() for key in ("norm", "res_out", "out")] for case in chain
        ]

    def check(expected):
        for case, baseline in zip(chain, expected):
            for key, oracle in zip(("norm", "res_out", "out"), baseline):
                assert torch.equal(
                    case[key].view(torch.uint8), oracle.view(torch.uint8)
                ), (rank, key)

    for amp in (0.01, 0.03, 0.125):
        for case in chain:
            case["inp"].normal_(std=amp)
            case["residual"].normal_(std=amp)
        call(0)
        torch.accelerator.synchronize()
        expected = snapshots()
        for arm in arms[1:]:
            call(arm)
            torch.accelerator.synchronize()
            check(expected)
    graph_outputs = {}
    graphs = {}
    for arm in arms:
        dist.barrier()
        call(arm)
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with ca.capture(), torch.cuda.graph(graph):
            call(arm)
        graphs[arm] = graph
        graph_outputs[arm] = [
            [case[key] for key in ("norm", "res_out", "out")] for case in chain
        ]
        graph.replay()
        torch.accelerator.synchronize()
    for _ in range(20):
        for case in chain:
            case["inp"].normal_(std=0.03)
            case["residual"].normal_(std=0.03)
        graphs[0].replay()
        torch.accelerator.synchronize()
        expected_outputs = [
            [tensor.clone() for tensor in row] for row in graph_outputs[0]
        ]
        for arm in arms[1:]:
            graphs[arm].replay()
            torch.accelerator.synchronize()
            for candidate, expected in zip(graph_outputs[arm], expected_outputs):
                assert all(
                    torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                    for a, b in zip(candidate, expected)
                ), rank
    samples = {arm: [] for arm in arms}
    intervals = []
    if args.trace_only:
        dist.barrier()
        torch.cuda.cudart().cudaProfilerStart()
        for arm in arms:
            torch.cuda.nvtx.range_push(f"norm_gate_arm{arm}")
            for _ in range(4):
                graphs[arm].replay()
            torch.accelerator.synchronize()
            torch.cuda.nvtx.range_pop()
            dist.barrier()
        torch.cuda.cudart().cudaProfilerStop()
    if not args.numeric_only and not args.trace_only:
        warmup_until = time.monotonic() + 5.0
        while time.monotonic() < warmup_until:
            for iteration in range(512):
                graphs[arms[iteration % len(arms)]].replay()
            torch.accelerator.synchronize()
        for arm in (arms + arms[::-1]) * 5:
            dist.barrier()
            for _ in range(10):
                graphs[arm].replay()
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            before = time.time()
            start.record()
            for _ in range(100):
                graphs[arm].replay()
            end.record()
            end.synchronize()
            us = start.elapsed_time(end) * 1000 / 100
            samples[arm].append(us)
            intervals.append(
                dict(arm=arm, us=us, wall_start=before, wall_end=time.time())
            )
    result = dict(
        research_only=True,
        unconditional_prefetch=args.cache_unconditional,
        serving_runtime=False,
        rank=rank,
        source_sha=args.source_sha,
        generated_cuda_sha256=hashlib.sha256(generated.read_bytes()).hexdigest(),
        eager_norm_residual_projection_bitwise=True,
        graph_norm_residual_projection_bitwise=True,
        changed_input_graph_replays=20,
        samples_us=samples,
        timing_intervals=intervals,
        activation_cache_chain_us=statistics.mean(samples[3])
        if samples.get(3)
        else None,
        original_chain_us=statistics.mean(samples[0]) if samples[0] else None,
        overlap_chain_us=statistics.mean(samples.get(1, []))
        if samples.get(1)
        else None,
        serial_candidate_chain_us=statistics.mean(samples.get(2, []))
        if samples.get(2)
        else None,
    )
    (args.output / f"rank{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
    dist.barrier()
    ca.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--numeric-only", action="store_true")
    parser.add_argument("--trace-only", action="store_true")
    parser.add_argument("--activation-cache", action="store_true")
    parser.add_argument("--cache-unconditional", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "build").mkdir(exist_ok=True)
    args.rendezvous = args.output / ("gloo-" + uuid.uuid4().hex)
    generated = source(args.root, args.output, args.cache_unconditional)
    torch.multiprocessing.spawn(worker, args=(args, generated), nprocs=4, join=True)
