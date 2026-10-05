# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen M8 operand dependencies without changing production dispatch.

The two ablations deliberately compute different results: they must never be
used in serving. A source-extracted control and a fixed-shape specialization
retain the production HMMA and reduction order. All variants record compiler
resources; ablation timing alone does not establish a deployable speedup.
"""

import argparse
import hashlib
import json
import os
import runpy
import statistics
from functools import partial
from pathlib import Path

VARIANTS = ["control", "constant_activation", "constant_weight", "fixed_shape"]


def generate(root: Path) -> str:
    source = (root / "csrc/sm70_turbomind/ops/nvfp4_qpn2_sm70.cu").read_text()
    helpers = source[
        source.index(
            "__device__ __forceinline__ half2 fp8e4m3_to_half2"
        ) : source.index("// Batch kernels share")
    ]
    macro = source[
        source.index("#define VLLM_SM70_QPN2_MMA") : source.index(
            "// Four row tiles reuse"
        )
    ]
    start = source.index("template <int SplitK, int NAcc, int RowTiles = 1")
    end = source.index("\nvoid launch_qpn2", start)
    end = source.rfind("\ntemplate <int SplitK", start, end)
    kernels = source[start:end]
    assert kernels.count("__global__ void") == 2
    weight_load = """    const uint2 packed = reader.load(group);
    const half2 scale = nvfp4_effective_scale(
        __ldg(scale_ptr + static_cast<size_t>(group) * 32), global_scale);
    half2 weights[8];
    dequant_e2m1x8(packed.x, scale, weights);
    dequant_e2m1x8(packed.y, scale, weights + 4);"""
    activation_load = (
        "        input01 = *reinterpret_cast<const uint4*>(input_row + group * 16);\n"
        "        input23 = *reinterpret_cast<const uint4*>(input_row + group * 16 + 8);"
    )
    assert kernels.count(weight_load) == 2
    assert kernels.count(activation_load) == 2
    copies = []
    for variant in VARIANTS:
        text = kernels
        if variant == "constant_activation":
            text = text.replace(
                activation_load,
                "        input01 = make_uint4(0x30003000, 0x30003000, "
                "0x30003000, 0x30003000);\n        input23 = input01;",
            )
        elif variant == "constant_weight":
            text = text.replace(
                weight_load,
                "    half2 weights[8];\n#pragma unroll\n"
                "    for (int i = 0; i < 8; ++i) "
                "weights[i] = __float2half2_rn(0.001f);",
            )
        elif variant == "fixed_shape":
            boundary = "  static_assert(RowTiles == 1 || RowTiles == 2,"
            down, gate = text.split(
                "template <int SplitK, int NAcc, int RowTiles = 1, "
                "bool TurboMindLayout = false>\n"
            )
            down = down.replace(boundary, "  m = 8; n = 5120; k = 4352;\n" + boundary)
            gate = gate.replace(
                boundary, "  m = 8; hidden = 4352; k = 5120;\n" + boundary
            )
            text = (
                down
                + (
                    "template <int SplitK, int NAcc, int RowTiles = 1, "
                    "bool TurboMindLayout = false>\n"
                )
                + gate
            )
        text = text.replace("nvfp4_qpn2_sm70_kernel", f"operand_{variant}_down")
        text = text.replace("nvfp4_qpn2_gated_sm70_kernel", f"operand_{variant}_gate")
        copies.append(text)
    launches = []
    for index, variant in enumerate(VARIANTS):
        launches.append(f"""    case {index}:
      if (gated) operand_{variant}_gate<8, 1><<<136, 512, 0, stream>>>(
          w, s, x, y, n, k, m, scale);
      else operand_{variant}_down<16, 2><<<160, 512, 0, stream>>>(
          w, s, x, y, n, k, m, scale);
      break;""")
    wrapper = Path(__file__).with_suffix(".cuh").read_text()
    return (
        "#include <torch/all.h>\n#include <torch/library.h>\n#include <cuda_fp16.h>\n"
        "#include <ATen/cuda/CUDAContext.h>\n"
        "#include <c10/cuda/CUDAGuard.h>\n"
        "#include <c10/cuda/CUDAException.h>\n"
        '#include "nvfp4_qpn2_layout.cuh"\n'
        "namespace {\nconstexpr int kQpn2RowsPerCta = 8;\n"
        + helpers
        + macro
        + "\n".join(copies)
        + wrapper.replace("// GENERATED_LAUNCHES", "\n".join(launches))
    )


def build(args):
    from torch.utils.cpp_extension import load

    args.output.mkdir(parents=True, exist_ok=True)
    generated = args.output / "operands.cu"
    generated.write_text(generate(args.source_root))
    os.environ.setdefault("MAX_JOBS", "1")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0"
    load(
        name="qpn_operands700",
        sources=[str(generated)],
        extra_include_paths=[str(args.source_root / "csrc/sm70_turbomind/ops")],
        extra_cuda_cflags=["-O3", "--ptxas-options=-v"],
        extra_ldflags=["-Wl,-Bsymbolic"],
        build_directory=str(args.output),
        is_python_module=False,
        verbose=True,
    )
    return generated


def run(args, generated):
    import torch

    from vllm import _sm70_ops as ops

    torch.set_grad_enabled(False)
    torch.manual_seed(20261005)
    load_weights = runpy.run_path(
        str(args.source_root / "benchmarks/kernels/benchmark_sm70_nvfp4_qpn2.py")
    )["_load_projection_shards"]
    extension = torch.ops._qpn_operands700
    # Read eviction avoids writeback from a preceding fill in the timed range.
    eviction = torch.ones(128 * 1024 * 1024 // 4, device="cuda", dtype=torch.int32)
    sink = torch.empty(256, device="cuda", dtype=torch.int64)
    records = []
    for layer in args.layers:
        for projection in load_weights(args.model, layer, 0, 4):
            codes, scales = ops.nvfp4_qpn2_prepare_sm70(
                projection.packed.cuda(), projection.scales.cuda()
            )
            n, packed_k = projection.packed.shape
            gated = projection.gated_silu
            x = torch.randn(8, packed_k * 2, device="cuda", dtype=torch.float16) * 0.1
            out = x.new_empty((8, n // 2 if gated else n))
            arguments = (out, x, codes, scales, projection.inverse_global_scale, gated)
            prod = (
                ops.nvfp4_qpn2_gated_sm70_out if gated else ops.nvfp4_qpn2_gemm_sm70_out
            )
            checks = []
            for amplitude in [0.0, 0.125, -0.125, 0.25]:
                x.copy_(torch.randn_like(x) * amplitude)
                prod(*arguments[:5], 8 if gated else 16, 1 if gated else 2)
                reference = out.clone()
                for variant in [0, 3]:
                    extension.run(*arguments, variant)
                    exact = torch.equal(
                        out.view(torch.int16), reference.view(torch.int16)
                    )
                    checks.append(
                        {
                            "variant": VARIANTS[variant],
                            "amplitude": amplitude,
                            "bitwise": exact,
                        }
                    )
                    if not exact:
                        raise AssertionError((layer, gated, checks[-1]))
            x.copy_(torch.randn_like(x) * 0.1)
            functions = [
                partial(prod, *arguments[:5], 8 if gated else 16, 1 if gated else 2)
            ]
            functions += [partial(extension.run, *arguments, v) for v in range(4)]
            graphs = []
            for fn in functions:
                for _ in range(5):
                    fn()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                begin = torch.cuda.Event(enable_timing=True, external=True)
                end = torch.cuda.Event(enable_timing=True, external=True)
                with torch.cuda.graph(graph):
                    extension.evict(eviction, sink)
                    begin.record()
                    fn()
                    end.record()
                graphs.append((graph, begin, end))
            samples = {name: [] for name in ["production"] + VARIANTS}
            if args.profile:
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStart()
                for graph, _, _ in graphs[1:]:
                    graph.replay()
                    torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()
            for repeat in range(args.repeats + 5):
                for offset in range(len(graphs)):
                    index = (repeat + offset) % len(graphs)
                    graph, begin, end = graphs[index]
                    graph.replay()
                    end.synchronize()
                    if repeat >= 5:
                        samples[list(samples)[index]].append(
                            begin.elapsed_time(end) * 1000
                        )
            records.append(
                {
                    "layer": layer,
                    "gated": gated,
                    "shape": [8, n, packed_k * 2],
                    "checks": checks,
                    "weight_scale_bytes": codes.numel() + scales.numel(),
                    "timing": {
                        name: {
                            "median_us": statistics.median(values),
                            "mean_us": statistics.mean(values),
                            "samples_us": values,
                        }
                        for name, values in samples.items()
                    },
                }
            )
            print(json.dumps(records[-1]), flush=True)
            for graph, _, _ in graphs:
                graph.reset()
    result = {
        "research_only": True,
        "not_end_to_end": True,
        "invalid_model_outputs": ["constant_activation", "constant_weight"],
        "cold_l2": "128MiB read eviction outside timing",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "source_sha256": hashlib.sha256(generated.read_bytes()).hexdigest(),
        "library_sha256": hashlib.sha256(
            (args.output / "qpn_operands700.so").read_bytes()
        ).hexdigest(),
        "records": records,
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 16, 32, 55])
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.generate_only:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "operands.cu").write_text(generate(args.source_root))
        return
    if not args.build_only and args.model is None:
        parser.error("--model is required unless --build-only is set")
    generated = build(args)
    if not args.build_only:
        run(args, generated)


if __name__ == "__main__":
    main()
