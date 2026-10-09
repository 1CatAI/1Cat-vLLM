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

VARIANTS = [
    "control",
    "constant_activation",
    "constant_weight",
    "fixed_shape",
    "packed_input",
    "packed_scaled",
    "packed_chain",
    "packed_scaled_chain",
    "n16_chain",
    "packed_scaled_swizzle_chain",
    "warp_stage_chain",
    "cachecg",
    "cacheca",
    "packed_scaled_cachecg_chain",
    "packed_scaled_cacheca_chain",
    "loop1",
    "loop2",
    "phase8",
    "phase16",
    "gather_a",
    "stream_a",
]


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
    fast_decode = """
__device__ __forceinline__ void range_decode(unsigned packed, half2 scale,
                                            half2 output[4]) {
  constexpr unsigned sign = 0x80008000u, magnitude = 0x0e000e00u;
  unsigned values[4];
  values[0] = ((packed << 12) & sign) | ((packed << 9) & magnitude);
  values[1] = ((packed << 8) & sign) | ((packed << 5) & magnitude);
  values[2] = ((packed << 4) & sign) | ((packed << 1) & magnitude);
  values[3] = (packed & sign) | ((packed >> 3) & magnitude);
#pragma unroll
  for (int i = 0; i < 4; ++i)
    output[i] = __hmul2(*reinterpret_cast<half2*>(&values[i]), scale);
}
"""
    for variant in VARIANTS:
        if variant in ["n16_chain", "warp_stage_chain"]:
            continue
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
        elif variant.startswith("packed_"):
            # Reuse the K16 activation layout already proposed in #822, while
            # holding the native weight layout and M8 arithmetic fixed.
            old = (
                "const half* input_row = input + static_cast<size_t>(row) * k;\n"
                + activation_load
            )
            assert text.count(old) == 2
            text = text.replace(
                old,
                "const half* input_row = input + "
                "static_cast<size_t>(group) * 128 + row * 16;\n"
                "        input01 = *reinterpret_cast<const uint4*>(input_row);\n"
                "        input23 = *reinterpret_cast<const uint4*>(input_row + 8);",
            )
            if "scaled" in variant:
                # Exact only when the already-rounded effective half scale,
                # multiplied by 2^14, is finite. The screen checks this bound
                # before calling the candidate; this is not a serving gate.
                text = text.replace(
                    "    half2 weights[8];",
                    "    const half2 expanded_scale = __hmul2(\n"
                    "        scale, __float2half2_rn(16384.0f));\n"
                    "    half2 weights[8];",
                )
                text = text.replace(
                    "dequant_e2m1x8(packed.x, scale,",
                    "range_decode(packed.x, expanded_scale,",
                )
                text = text.replace(
                    "dequant_e2m1x8(packed.y, scale,",
                    "range_decode(packed.y, expanded_scale,",
                )
            if "chain" in variant:
                old_store = (
                    "output[static_cast<size_t>(output_row) * hidden + "
                    "output_tile * 32 +\n"
                    "             output_col]"
                )
                assert text.count(old_store) == 1
                text = text.replace(
                    old_store,
                    "output[static_cast<size_t>(output_tile * 2 + "
                    "output_col / 16) * 128 +\n"
                    "             output_row * 16 + output_col % 16]",
                )
            if "swizzle" in variant:
                # Preserve each logical K partition and final sum order, but
                # decorrelate which K segment each scheduler issues first
                # across neighboring CTAs. This changes no stored layout.
                old_down = "  const int warp = threadIdx.x >> 5;"
                old_gate = "  const int warp = warp_in_block - projection * SplitK;"
                assert text.count(old_down) == text.count(old_gate) == 1
                text = text.replace(
                    old_down,
                    "  const int warp = ((threadIdx.x >> 5) + blockIdx.x) "
                    "& (SplitK - 1);",
                )
                text = text.replace(
                    old_gate,
                    "  const int warp = (warp_in_block - projection * SplitK "
                    "+ blockIdx.x) & (SplitK - 1);",
                )
        if variant in ["gather_a", "stream_a"]:
            start_a = kernels.index("      uint4 input01 = make_uint4")
            end_a = kernels.index("\n    }", start_a)
            original_a = kernels[start_a:end_a]
            assert text.count(original_a) == 2
            new_a = """      const int read_row = row_base + row_tile * 8 + (lane >> 2);
      uint2 piece = make_uint2(0, 0);
      if (read_row < m) {
        piece = *reinterpret_cast<const uint2*>(input +
            static_cast<size_t>(read_row) * k + group * 16 + (lane & 3) * 4);
      }
      const int owner = local_row * 4;
"""
            if variant == "gather_a":
                new_a += """      unsigned a[8];
#pragma unroll
      for (int quarter = 0; quarter < 4; ++quarter) {
        a[quarter * 2] = __shfl_sync(0xffffffff, piece.x, owner + quarter);
        a[quarter * 2 + 1] = __shfl_sync(0xffffffff, piece.y, owner + quarter);
      }
#pragma unroll
      for (int quarter = 0; quarter < 4; ++quarter) {
        VLLM_SM70_QPN2_MMA(accum[row_tile][quarter % NAcc],
            a[quarter * 2], a[quarter * 2 + 1], b[quarter * 2], b[quarter * 2 + 1]);
      }"""
            else:
                new_a += """#pragma unroll
      for (int quarter = 0; quarter < 4; ++quarter) {
        const unsigned a0 = __shfl_sync(0xffffffff, piece.x, owner + quarter);
        const unsigned a1 = __shfl_sync(0xffffffff, piece.y, owner + quarter);
        VLLM_SM70_QPN2_MMA(accum[row_tile][quarter % NAcc],
            a0, a1, b[quarter * 2], b[quarter * 2 + 1]);
      }"""
            text = text.replace(original_a, new_a)
        if variant.startswith("loop"):
            marker = "#pragma unroll 4\n  for (int group"
            assert text.count(marker) == 2
            text = text.replace(
                marker, f"#pragma unroll {variant[-1]}\n  for (int group"
            )
        if variant.startswith("phase"):
            phases = int(variant.removeprefix("phase"))
            for reader in [
                "Nvfp4Qpn2CodeReader<TurboMindLayout, CacheCodes>",
                "Nvfp4Qpn2CodeReader<TurboMindLayout>",
            ]:
                assert text.count(reader) == 1
                text = text.replace(reader, f"QpnPhaseReader<{phases}>")
            scale_index = "scale_ptr + static_cast<size_t>(group) * 32"
            assert text.count(scale_index) == 2
            text = text.replace(
                scale_index, "scale_ptr + reader.physical_group(group) * 32"
            )
        if "cachecg" in variant or "cacheca" in variant:
            policy = 0 if "cachecg" in variant else 1
            for reader in [
                "Nvfp4Qpn2CodeReader<TurboMindLayout, CacheCodes>",
                "Nvfp4Qpn2CodeReader<TurboMindLayout>",
            ]:
                assert text.count(reader) == 1
                text = text.replace(reader, f"QpnCacheReader<{policy}>")
        text = text.replace("nvfp4_qpn2_sm70_kernel", f"operand_{variant}_down")
        text = text.replace("nvfp4_qpn2_gated_sm70_kernel", f"operand_{variant}_gate")
        copies.append(text)
    launches = []
    for index, variant in enumerate(VARIANTS):
        if variant in ["n16_chain", "warp_stage_chain"]:
            continue
        launches.append(f"""    case {index}:
      if (gated) operand_{variant}_gate<8, 1><<<136, 512, 0, stream>>>(
          w, s, x, y, n, k, m, scale);
      else operand_{variant}_down<16, 2><<<160, 512, 0, stream>>>(
          w, s, x, y, n, k, m, scale);
      break;""")
    cache_reader = """
template <int Policy> struct QpnCacheReader {
  const uint2* base;
  __device__ QpnCacheReader(const uint8_t* codes, int tile, int groups, int lane)
      : base(reinterpret_cast<const uint2*>(codes) + tile * groups * 32 + lane) {}
  __device__ __forceinline__ uint2 load(int group) const {
    if constexpr (Policy == 0) return __ldcg(base + group * 32);
    else return __ldca(base + group * 32);
  }
};
template <int Phases> struct QpnPhaseReader {
  const uint2* base;
  int groups, phase;
  __device__ QpnPhaseReader(const uint8_t* codes, int tile, int count, int lane)
      : base(reinterpret_cast<const uint2*>(codes) + tile * count * 32 + lane),
        groups(count), phase(tile & (Phases - 1)) {}
  __device__ __forceinline__ int physical_group(int group) const {
    const int shifted = group + phase;
    return shifted < groups ? shifted : shifted - groups;
  }
  __device__ __forceinline__ uint2 load(int group) const {
    return __ldcs(base + physical_group(group) * 32);
  }
};
"""
    wrapper = (
        Path(__file__).with_name("benchmark_sm70_qpn_n16.cuh").read_text()
        + Path(__file__).with_name("benchmark_sm70_qpn_warp_stage.cuh").read_text()
        + Path(__file__).with_suffix(".cuh").read_text()
    )
    return (
        "#include <torch/all.h>\n#include <torch/library.h>\n#include <cuda_fp16.h>\n"
        "#include <ATen/cuda/CUDAContext.h>\n"
        "#include <c10/cuda/CUDAGuard.h>\n"
        "#include <c10/cuda/CUDAException.h>\n"
        '#include "nvfp4_qpn2_layout.cuh"\n'
        '#include "activation_pack_sm70.cuh"\n'
        "namespace {\nconstexpr int kQpn2RowsPerCta = 8;\n"
        + helpers
        + fast_decode
        + cache_reader
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
    selected = [VARIANTS.index(name) for name in (args.variants or VARIANTS)]

    def packed_variant(index):
        name = VARIANTS[index]
        return name.startswith("packed_") or name in ["n16_chain", "warp_stage_chain"]

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
            effective = (
                scales.view(torch.float8_e4m3fn).float()
                * projection.inverse_global_scale
            ).half()
            expanded = (effective.float() * 16384).half()
            if not torch.isfinite(expanded).all().item():
                raise ValueError("Scale exceeds exact range-decode capability")
            packed_x = torch.empty_like(x)
            packed_arguments = (out, packed_x, *arguments[2:])
            phase_arguments = {}
            for variant in selected:
                name = VARIANTS[variant]
                if not name.startswith("phase"):
                    continue
                phases = int(name.removeprefix("phase"))
                tiles, groups = n // 32, packed_k // 8
                tile = torch.arange(tiles, device=x.device)[:, None]
                physical = torch.arange(groups, device=x.device)[None, :]
                logical = (physical - tile % phases) % groups
                rotated_w = codes.view(tiles, groups, 256)[tile, logical].contiguous()
                rotated_s = scales.view(tiles, groups, 32)[tile, logical].contiguous()
                phase_arguments[variant] = (
                    out,
                    x,
                    rotated_w,
                    rotated_s,
                    projection.inverse_global_scale,
                    gated,
                )

            def arguments_for(
                variant,
                phase_arguments=phase_arguments,
                packed_arguments=packed_arguments,
                arguments=arguments,
            ):
                return phase_arguments.get(
                    variant, packed_arguments if packed_variant(variant) else arguments
                )

            prod = (
                ops.nvfp4_qpn2_gated_sm70_out if gated else ops.nvfp4_qpn2_gemm_sm70_out
            )
            checks = []
            for amplitude in [0.0, 0.125, -0.125, 0.25]:
                x.copy_(torch.randn_like(x) * amplitude)
                prod(*arguments[:5], 8 if gated else 16, 1 if gated else 2)
                reference = out.clone()
                extension.pack(x, packed_x)
                for variant in selected:
                    if variant in [1, 2]:
                        continue
                    extension.run(*arguments_for(variant), variant)
                    compared = out
                    if gated and "chain" in VARIANTS[variant]:
                        compared = out.view(-1, 8, 16).permute(1, 0, 2).reshape_as(out)
                    exact = torch.equal(
                        compared.view(torch.int16), reference.view(torch.int16)
                    )
                    checks.append(
                        {
                            "variant": VARIANTS[variant],
                            "amplitude": amplitude,
                            "bitwise": exact,
                        }
                    )
                    if not exact and variant != 8:
                        raise AssertionError((layer, gated, checks[-1]))
                    if not torch.isfinite(compared).all().item():
                        raise AssertionError((layer, gated, "nonfinite", variant))
            x.copy_(torch.randn_like(x) * 0.1)
            extension.pack(x, packed_x)
            functions = [
                partial(prod, *arguments[:5], 8 if gated else 16, 1 if gated else 2)
            ]
            functions += [
                partial(extension.run, *arguments_for(v), v) for v in selected
            ]
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
            samples = {
                name: [] for name in ["production"] + [VARIANTS[v] for v in selected]
            }
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
                    for _ in range(args.replays_per_sample):
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
    parser.add_argument("--variants", choices=VARIANTS, nargs="+")
    parser.add_argument("--replays-per-sample", type=int, default=10)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--load-existing", action="store_true")
    args = parser.parse_args()
    if args.generate_only:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "operands.cu").write_text(generate(args.source_root))
        return
    if not args.build_only and args.model is None:
        parser.error("--model is required unless --build-only is set")
    if args.load_existing:
        import torch

        generated = args.output / "operands.cu"
        if generated.read_text() != generate(args.source_root):
            raise ValueError("Existing library source does not match this screen")
        torch.ops.load_library(str(args.output / "qpn_operands700.so"))
    else:
        generated = build(args)
    if not args.build_only:
        run(args, generated)


if __name__ == "__main__":
    main()
