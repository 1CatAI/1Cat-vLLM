# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Research-only mixed-format common MMA loop screen.

Run under shared GPU locks. The original control kernel is unchanged.
Only decode/load format dispatch is moved inside a common MMA loop; existing
readers, tile geometry, accumulator order and epilogue stay unchanged.
"""

import argparse
import hashlib
import importlib.util
import json
import statistics
import subprocess
import time
import uuid
from pathlib import Path

import gguf
import numpy as np
import torch
from torch.utils.cpp_extension import load


def source(root, directory):
    original = (root / "csrc/sm70_turbomind/ops/gguf_dmv_sm70.cu").read_text()
    device = original[: original.index("}  // namespace")]
    begin = device.index("template <int FMT, int KW, int TN, bool LegacyIq2 = false>")
    finish = device.index("__device__ __forceinline__ void write_out", begin)
    body = device[begin:finish]
    body = body.replace(
        "template <int FMT, int KW, int TN, bool LegacyIq2 = false>",
        "template <int KW, int TN, int FA, int FB>",
    )
    body = body.replace("void body6(", "void common_body6(")
    body = body.replace("  Ld<FMT> A;", "  Ld<Q4K> A;")
    body = body.replace("    Ld<FMT> B;", "    Ld<Q4K> B;")
    left = body.index("  if constexpr (LegacyIq2)")
    right = body.index("  uint4* slot", left)
    body = body[:left] + body[right:]
    body = body.replace(
        "load<FMT>(A, sg, t, g, S, G, lane)",
        "common_load<FA, FB>(A, sg, t, g, S, G, lane)",
    )
    body = body.replace(
        "load<FMT>(B, sg, t, gn, S, G, lane)",
        "common_load<FA, FB>(B, sg, t, gn, S, G, lane)",
    )
    body = body.replace(
        "decode<FMT, LegacyIq2>(A, st, hw, lut)",
        "common_decode<FA, FB>(A, st, hw, lut, sg.fmt)",
    )
    marker = (
        "template <int KW, int TN, int FA, int FB, int FC = -1, bool Clocked = false>"
    )
    start = device.index(marker + "\n__global__")
    end = device.index(marker + "\nvoid launch", start)
    kernel = device[start:end].replace("dense_mv(Segs", "common_dense_mv(Segs")
    start_branch = kernel.index("  if (FA == FB || sg.fmt == FA)")
    end_branch = kernel.index("  __syncthreads();", start_branch)
    kernel = (
        kernel[:start_branch]
        + """  common_body6<KW, TN, FA, FB>(
      sg, t, on, kslot, tin, g0, g1, S, G, x, ldx, M, xs, lut, acc,
      segs.tab, segs.gdn_heads, pair);
"""
        + kernel[end_branch:]
    )
    d0 = device.index("template <int FMT, bool ExactLattice = false>")
    d1 = device.index("__device__ __constant__ int8_t kIQ4", d0)
    decoder = device[d0:d1].replace("void decode(", "void family_decode(")
    decoder = decoder.replace(
        "const half2* lut) {", "const half2* lut, float coefficient) {"
    )
    decoder = decoder.replace("(FMT == IQ3X ? 0.25f : 1.0f)", "coefficient")
    family = device[begin:finish]
    family = family.replace(
        "template <int FMT, int KW, int TN, bool LegacyIq2 = false>",
        "template <int KW, int TN>",
    )
    family = family.replace("void body6(", "void family_body6(")
    left = family.index("  if constexpr (LegacyIq2)")
    right = family.index("  uint4* slot", left)
    family = family[:left] + family[right:]
    family = family.replace("Ld<FMT>", "Ld<IQ3S>").replace("load<FMT>", "load<IQ3S>")
    # The source pointer and coefficient are invariant across the entire loop.
    family = family.replace(
        "  const int lane = threadIdx.x % 32;",
        "  const int lane = threadIdx.x % 32;\n"
        "  const half2* family_lut = lut + (sg.fmt == IQ3X ? 512 : 0);\n"
        "  const float coefficient = sg.fmt == IQ3X ? 0.25f : 1.0f;",
    )
    family = family.replace(
        "decode<FMT, LegacyIq2>(A, st, hw, lut)",
        "family_decode<IQ3S>(A, st, hw, family_lut, coefficient)",
    )
    family_kernel = kernel.replace("common_dense_mv", "family_dense_mv")
    family_kernel = family_kernel.replace(
        "common_body6<KW, TN, FA, FB>", "family_body6<KW, TN>"
    )
    compact = family.replace("void family_body6(", "void compact_family_body6(")
    compact = compact.replace(
        "    if (on) {\n#pragma unroll\n      for (int st = 0; st < 4; ++st) {",
        "    if (on) {\n      Ld<IQ3S> step_weights = A;\n"
        "#pragma unroll 1\n      for (int st = 0; st < 4; ++st) {",
    )
    compact = compact.replace(
        "family_decode<IQ3S>(A, st, hw, family_lut, coefficient)",
        "family_decode<IQ3S>(step_weights, 0, hw, family_lut, coefficient)",
    )
    tail = """          mma(acc, xa[j].z, xa[j].w, hw[4 * j + 2], hw[4 * j + 3]);
        }
"""
    compact = compact.replace(
        tail,
        tail
        + """        // Advance a fixed decoder packet in registers. This avoids
        // dynamically indexing an array when sharing the four K32 steps.
        step_weights.c[0][0].x = step_weights.c[0][0].z;
        step_weights.c[0][0].y = step_weights.c[0][0].w;
        step_weights.c[0][0].z = step_weights.c[1][0].x;
        step_weights.c[0][0].w = step_weights.c[1][0].y;
        step_weights.c[1][0].x = step_weights.c[1][0].z;
        step_weights.c[1][0].y = step_weights.c[1][0].w;
        step_weights.c[2][0].x = step_weights.c[2][0].y;
        step_weights.c[2][0].y = step_weights.c[2][0].z;
        step_weights.c[2][0].z = step_weights.c[2][0].w;
        step_weights.sc.x >>= 8;
        step_weights.sc.y = (step_weights.sc.y & 0xffffu) |
                            ((step_weights.sc.y >> 4) & 0xffff0000u);
""",
    )
    compact_kernel = family_kernel.replace("family_dense_mv", "compact_family_dense_mv")
    compact_kernel = compact_kernel.replace(
        "family_body6<KW, TN>", "compact_family_body6<KW, TN>"
    )
    helpers = """
template <int FA, int FB>
__device__ __forceinline__ void common_load(
    Ld<Q4K>& storage, const Seg& sg, int t, int g, int S, int G, int lane) {
  static_assert(sizeof(Ld<FA>) == sizeof(storage));
  static_assert(sizeof(Ld<FB>) == sizeof(storage));
  if (sg.fmt == FA)
    load<FA>(reinterpret_cast<Ld<FA>&>(storage), sg, t, g, S, G, lane);
  else
    load<FB>(reinterpret_cast<Ld<FB>&>(storage), sg, t, g, S, G, lane);
}
template <int FA, int FB>
__device__ __forceinline__ void common_decode(
    const Ld<Q4K>& storage, int st, uint32_t (&hw)[16], const half2* lut,
    int fmt) {
  if (fmt == FA)
    decode<FA>(reinterpret_cast<const Ld<FA>&>(storage), st, hw, lut);
  else
    decode<FB>(reinterpret_cast<const Ld<FB>&>(storage), st, hw, lut);
}
"""
    header = (root / "benchmarks/csrc/gguf_shared_loop_sm70.cuh").read_text()
    target = directory / "shared_loop.cu"
    content = (
        device
        + helpers
        + body
        + kernel
        + decoder
        + family
        + family_kernel
        + compact
        + compact_kernel
        + "}  // namespace\n"
        + header
    )
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
    if kind in (18, 21):
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
        name="gguf_shared_loop_research",
        sources=[str(generated)],
        build_directory=str(args.output / "build"),
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
            "--ptxas-options=-v",
            "-gencode=arch=compute_70,code=sm_70",
        ],
    )
    reader = gguf.GGUFReader(str(args.model))
    tensors = {t.name: t for t in reader.tensors}
    chain, cases = [], []
    for layer in (22, 38) if args.family else (22, 38, 39, 42, 36, 47):
        planes, oracles, kinds = [], [], []
        for role in ("ffn_gate", "ffn_up"):
            t = tensors[f"blk.{layer}.{role}.weight"]
            k, n = map(int, t.shape)
            block, size = gguf.GGML_QUANT_SIZES[t.tensor_type]
            raw = t.data.reshape(n, k // block, size)
            raw = np.ascontiguousarray(raw[rank * (n // 4) : (rank + 1) * (n // 4)])
            raw = raw.reshape(n // 4, -1)
            fmt, codes, scales, table = pack(args.root, raw, int(t.tensor_type))
            planes.append(
                (fmt, torch.from_numpy(codes).cuda(), torch.from_numpy(scales).cuda())
            )
            oracles.append(
                torch.from_numpy(gguf.quants.dequantize(raw, t.tensor_type)).cuda()
            )
            kinds.append(int(t.tensor_type))
        tab = torch.from_numpy(table).cuda()
        x = torch.randn(8, 5120, device=rank, dtype=torch.float16)
        out = torch.empty(8, 4352, device=rank, dtype=torch.float16)
        c, sc = [p[1] for p in planes], [p[2] for p in planes]
        fa, fb = [p[0] for p in planes]
        candidate_sc = list(sc)
        if args.family:
            for i, fmt in enumerate((fa, fb)):
                if fmt == 6:
                    words = sc[i].view(torch.int32)
                    expanded = torch.zeros(
                        words.shape + (2,), device=rank, dtype=torch.int32
                    )
                    expanded[..., 1] = words
                    candidate_sc[i] = expanded.view(torch.uint8)
        variant = 3 if args.compact else (2 if args.family else 1)
        errors = []
        for amp in (0.125, 0.5, 1.0):
            x.normal_(std=amp)
            extension.run(x, c, sc, tab, out, fa, fb, False)
            ref = out.clone()
            extension.run(x, c, candidate_sc, tab, out, fa, fb, variant)
            torch.accelerator.synchronize()
            assert torch.equal(out, ref), (rank, layer, "pair bits")
            gate = (x.float() @ oracles[0].float().T).half().float()
            up = (x.float() @ oracles[1].float().T).half().float()
            oracle = (torch.nn.functional.silu(gate) * up).half()
            error = float((out.float() - oracle.float()).norm() / oracle.float().norm())
            assert torch.isfinite(out).all() and error < 0.01
            errors.append(error)
        cases.append(
            dict(
                layer=layer,
                types=kinds,
                formats=[fa, fb],
                bitwise_equal=True,
                official_relative_l2=errors,
            )
        )
        chain.append((x, c, sc, candidate_sc, tab, out, fa, fb))
        del oracles
    if not args.numeric_only:
        # Six independent mixed-format layers exceed L2 and vary the code stream.
        graphs = []
        for common in (False, True):

            def call(common=common):
                for x, c, sc, candidate_sc, tab, out, fa, fb in chain:
                    extension.run(
                        x,
                        c,
                        candidate_sc if common else sc,
                        tab,
                        out,
                        fa,
                        fb,
                        variant if common else 0,
                    )

            call()
            torch.accelerator.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                call()
            graphs.append(graph)
        samples = {False: [], True: []}
        timeline = []
        for arm in (False, True, True, False) * 5:
            torch.distributed.barrier()
            graph = graphs[int(arm)]
            for _ in range(10):
                graph.replay()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            wall_start = time.time()
            start.record()
            for _ in range(50):
                graph.replay()
            end.record()
            end.synchronize()
            us = start.elapsed_time(end) * 1000 / 50
            samples[arm].append(us)
            timeline.append(
                dict(
                    common=arm, chain_us=us, wall_start=wall_start, wall_end=time.time()
                )
            )
        for _ in range(20):
            for x, *_ in chain:
                x.normal_(std=0.5)
            graphs[0].replay()
            expected = [out.clone() for _, _, _, _, _, out, _, _ in chain]
            graphs[1].replay()
            torch.accelerator.synchronize()
            for (_, _, _, _, _, out, _, _), ref in zip(chain, expected):
                assert torch.equal(out, ref)
        performance = dict(
            original_chain_us=statistics.mean(samples[False]),
            common_chain_us=statistics.mean(samples[True]),
            samples_us={str(k): v for k, v in samples.items()},
            timing_intervals=timeline,
            changed_input_graph_replays=20,
        )
    else:
        performance = None
    clocks = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={rank}",
            "--query-gpu=clocks.sm,clocks.mem",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    record = dict(
        research_only=True,
        serving_runtime=False,
        family_variant=args.family,
        compact_variant=args.compact,
        rank=rank,
        source_sha=args.source_sha,
        generated_cuda_sha256=hashlib.sha256(generated.read_bytes()).hexdigest(),
        cases=cases,
        performance=performance,
        clocks_after_timing=clocks,
    )
    (args.output / f"rank{rank}.json").write_text(json.dumps(record, indent=2) + "\n")
    torch.distributed.destroy_process_group()


def distributed_worker(rank, args, generated):
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
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--numeric-only", action="store_true")
    parser.add_argument("--family", action="store_true")
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--ranks", type=int, choices=(1, 4), default=4)
    args = parser.parse_args()
    if args.compact:
        args.family = True
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "build").mkdir(exist_ok=True)
    args.rendezvous = args.output / f"rendezvous-{uuid.uuid4().hex}"
    generated = source(args.root, args.output)
    load(
        name="gguf_shared_loop_research",
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
        distributed_worker, args=(args, generated), nprocs=args.ranks
    )
