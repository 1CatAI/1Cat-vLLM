# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen exact effective-scale operands in cold real-weight M8 MLP graphs.

The control is extracted from the owned source and checked against the shipped
operator. Candidates either replace per-group scale conversion with a 256-entry
FP16 table or store effective FP16 scales next to the unchanged packed codes.
All packing is outside timing. Extensions are research-only, not serving routes.
"""

import argparse
import hashlib
import json
import random
import runpy
import statistics
from functools import partial
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from vllm import _sm70_ops as ops


def extract_kernel(source, name):
    anchor = source.index("__global__ void " + name)
    begin = source.rfind("template <", 0, anchor)
    body = source.index("{", anchor)
    depth = 1
    end = body + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[begin:end]


def generate(source):
    text = source.read_text()
    helpers = text[
        text.index("__device__ __forceinline__ half2 fp8e4m3") : text.index(
            "// Batch kernels share"
        )
    ]
    macro = text[
        text.index("#define VLLM_SM70_QPN2_MMA") : text.index("// Four row tiles reuse")
    ]
    kernels, calls = [], []
    for gated in (False, True):
        name = "nvfp4_qpn2_" + ("gated_" if gated else "") + "sm70_kernel"
        original = extract_kernel(text, name)
        scale_begin = original.index("    const half2 scale =")
        scale_end = original.index("    half2 weights[8];", scale_begin)
        scale_expression = original[scale_begin:scale_end]
        for mode in range(3):
            variant_name = name + f"_scale{mode}"
            variant = original.replace(name, variant_name)
            extra = ""
            if mode == 1:
                variant = variant.replace(
                    "int m, float global_scale) {",
                    "int m, float global_scale, const half* scale_table) {",
                )
                variant = variant.replace(
                    scale_expression,
                    "    const half h = __ldg(scale_table + __ldg(\n"
                    "        scale_ptr + static_cast<size_t>(group) * 288));\n"
                    "    const half2 scale = __halves2half2(h, h);\n",
                )
                extra = ", reinterpret_cast<const half*>(table.data_ptr<at::Half>())"
            elif mode == 2:
                reader_begin = variant.index("  const Nvfp4Qpn2CodeReader<")
                reader_end = variant.index("\n\n  float accum[", reader_begin)
                variant = (
                    variant[:reader_begin]
                    + "  const EffectiveCodeReader reader(\n"
                    + "      codes, tile, groups_k16, lane);\n"
                    + "  const half* scale_ptr = reinterpret_cast<const half*>(\n"
                    + "      codes + static_cast<size_t>(tile) * groups_k16 * 320\n"
                    + "      + 256) + lane;"
                    + variant[reader_end:]
                )
                variant = variant.replace(
                    scale_expression,
                    "    const half h = __ldg(\n"
                    "        scale_ptr + static_cast<size_t>(group) * 160);\n"
                    "    const half2 scale = __halves2half2(h, h);\n",
                )
            assert variant.count("__global__ void " + variant_name) == 1
            kernels.append(variant)
            template = (
                "8, 1, 1, false, true" if gated else "16, 2, 1, false, false, true"
            )
            calls.append(
                f"if (mode == {mode} && gated == {str(gated).lower()}) "
                f"{variant_name}<{template}><<<width / 32, 512, 0, stream>>>"
                f"(c, s, x, y, width, k, 8, global_scale{extra});"
            )
    return (
        r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include "nvfp4_qpn2_layout.cuh"
namespace {
constexpr int kQpn2RowsPerCta = 8;
struct EffectiveCodeReader {
  const uint8_t* base;
  __device__ EffectiveCodeReader(const uint8_t* codes, int tile, int groups, int lane)
      : base(codes + static_cast<size_t>(tile) * groups * 320 + lane * 8) {}
  __device__ __forceinline__ uint2 load(int group) const {
    return __ldcs(reinterpret_cast<const uint2*>(base + group * 320));
  }
};
"""
        + helpers
        + macro
        + "\n".join(kernels)
        + r"""
__global__ void prepare_effective(uint8_t* dest, const uint8_t* codes,
                                 const uint8_t* scales, int groups, float global) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < groups * 256)
    dest[(i / 256) * 320 + i % 256] = codes[i];
  if (i < groups * 32) {
    half h = __low2half(nvfp4_effective_scale(scales[i], global));
    auto* out = reinterpret_cast<half*>(dest + (i / 32) * 320 + 256);
    out[i % 32] = h;
  }
}
__global__ void prepare_table(half* table, float global) {
  int i = threadIdx.x;
  table[i] = __low2half(nvfp4_effective_scale(static_cast<uint8_t>(i), global));
}
}
void prepare(torch::Tensor dest, torch::Tensor table, torch::Tensor codes,
             torch::Tensor scales, double global) {
  auto stream = at::cuda::getCurrentCUDAStream();
  int groups = codes.numel() / 256;
  TORCH_CHECK(codes.is_contiguous() && scales.is_contiguous() &&
              dest.numel() == groups * 320 && table.numel() == 256,
              "native contiguous code/scale groups required");
  prepare_effective<<<(groups * 256 + 255) / 256, 256, 0, stream>>>(
      dest.data_ptr<uint8_t>(), codes.data_ptr<uint8_t>(),
      scales.data_ptr<uint8_t>(), groups, global);
  prepare_table<<<1, 256, 0, stream>>>(
      reinterpret_cast<half*>(table.data_ptr<at::Half>()), global);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void launch(torch::Tensor output, torch::Tensor input, torch::Tensor codes,
            torch::Tensor scales, torch::Tensor table, double global_scale,
            bool gated, int mode) {
  TORCH_CHECK(input.size(0) == 8 && input.is_contiguous() && mode >= 0 && mode <= 2,
              "M8-only research screen");
  auto stream = at::cuda::getCurrentCUDAStream();
  auto* c = codes.data_ptr<uint8_t>();
  auto* s = scales.data_ptr<uint8_t>();
  auto* x = reinterpret_cast<const half*>(input.data_ptr<at::Half>());
  auto* y = reinterpret_cast<half*>(output.data_ptr<at::Half>());
  int width = output.size(1), k = input.size(1);
"""
        + "\n".join(calls)
        + r"""
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepare", &prepare);
  m.def("launch", &launch);
}
"""
    )


def graph_pair(control, candidate, eviction, iterations):
    graphs = []
    for call in (control, candidate):
        graph = torch.cuda.CUDAGraph()
        begin = torch.cuda.Event(enable_timing=True, external=True)
        end = torch.cuda.Event(enable_timing=True, external=True)
        with torch.cuda.graph(graph):
            eviction.fill_(1)
            begin.record()
            call()
            end.record()
        for _ in range(10):
            graph.replay()
        graphs.append((graph, begin, end))
    samples = [[], []]
    rng = random.Random(123)
    for _ in range(iterations):
        order = [0, 1]
        rng.shuffle(order)
        for index in order:
            graph, begin, end = graphs[index]
            graph.replay()
            end.synchronize()
            samples[index].append(begin.elapsed_time(end) * 1000)
    differences = [a - b for a, b in zip(*samples)]
    boot = sorted(
        statistics.mean(rng.choices(differences, k=len(differences)))
        for _ in range(2000)
    )
    return {
        "control_us": statistics.mean(samples[0]),
        "candidate_us": statistics.mean(samples[1]),
        "saving_us": statistics.mean(differences),
        "saving_ci95_us": [boot[50], boot[1949]],
        "samples_us": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.source_root / "csrc/sm70_turbomind/ops/nvfp4_qpn2_sm70.cu"
    text = generate(source)
    generated = args.out / "effective_scale.cu"
    if not generated.exists() or generated.read_text() != text:
        generated.write_text(text)
    extension = load(
        name="qpn2_effective_scale_screen",
        sources=[str(generated)],
        extra_include_paths=[str(source.parent)],
        extra_cuda_cflags=["-O3", "-lineinfo", "--ptxas-options=-v"],
        verbose=True,
    )
    if args.compile_only:
        return
    helpers = runpy.run_path(
        str(Path(__file__).with_name("benchmark_sm70_nvfp4_qpn2.py"))
    )
    projections = helpers["_load_projection_shards"](args.model, 0, 0, 4)
    torch.manual_seed(123)
    torch.set_grad_enabled(False)
    assert torch.cuda.get_device_capability() == (7, 0)
    eviction = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    prepared, results = [], []
    for projection in projections:
        codes, scales = ops.nvfp4_qpn2_prepare_sm70(
            projection.packed.cuda(), projection.scales.cuda()
        )
        n, k = projection.packed.shape[0], projection.packed.shape[1] * 2
        width = n // 2 if projection.gated_silu else n
        bundle = torch.cat(
            (codes.view(-1, 256), scales.view(-1, 32)), dim=1
        ).contiguous()
        effective = torch.empty(bundle.size(0), 320, device="cuda", dtype=torch.uint8)
        table = torch.empty(256, device="cuda", dtype=torch.float16)
        extension.prepare(
            effective, table, codes, scales, projection.inverse_global_scale
        )
        x = torch.randn(8, k, device="cuda", dtype=torch.float16)
        base_input = x.clone()
        outputs = [
            torch.empty(8, width, device="cuda", dtype=x.dtype) for _ in range(4)
        ]
        calls = [
            partial(
                extension.launch,
                outputs[mode],
                x,
                effective if mode == 2 else bundle,
                scales,
                table,
                projection.inverse_global_scale,
                projection.gated_silu,
                mode,
            )
            for mode in range(3)
        ]
        production = (
            ops.nvfp4_qpn2_gated_sm70_out
            if projection.gated_silu
            else ops.nvfp4_qpn2_gemm_sm70_out
        )
        for amplitude in (0.01, 0.1, 1.0, 4.0):
            x.copy_(base_input * amplitude)
            for call in calls:
                call()
            production(
                outputs[3],
                x,
                codes,
                scales,
                projection.inverse_global_scale,
                8 if projection.gated_silu else 16,
                1 if projection.gated_silu else 2,
            )
            assert all(
                torch.equal(y.view(torch.int16), outputs[3].view(torch.int16))
                for y in outputs[:3]
            )
        x.copy_(base_input * 0.1)
        for mode in (1, 2):
            result = graph_pair(calls[0], calls[mode], eviction, args.iters)
            result.update(
                projection="gate_up" if projection.gated_silu else "down",
                mode=mode,
                bitwise=True,
                operand_bytes=effective.numel() if mode == 2 else bundle.numel(),
                table_bytes=table.numel() * table.element_size(),
            )
            results.append(result)
            print(
                json.dumps({k: v for k, v in result.items() if k != "samples_us"}),
                flush=True,
            )
        prepared.append((projection, bundle, scales, effective, table))
    hidden = torch.randn(8, 5120, device="cuda", dtype=torch.float16) * 0.1
    middle = [torch.empty(8, 4352, device="cuda", dtype=hidden.dtype) for _ in range(3)]
    outputs = [
        torch.empty(8, 5120, device="cuda", dtype=hidden.dtype) for _ in range(3)
    ]

    def mlp(mode):
        for i, (projection, bundle, scales, effective, table) in enumerate(prepared):
            extension.launch(
                middle[mode] if i == 0 else outputs[mode],
                hidden if i == 0 else middle[mode],
                effective if mode == 2 else bundle,
                scales,
                table,
                projection.inverse_global_scale,
                projection.gated_silu,
                mode,
            )

    for mode in range(3):
        mlp(mode)
    assert all(
        torch.equal(y.view(torch.int16), outputs[0].view(torch.int16))
        for y in outputs[1:]
    )
    for mode in (1, 2):
        result = graph_pair(partial(mlp, 0), partial(mlp, mode), eviction, args.iters)
        result.update(
            projection="mlp",
            mode=mode,
            bitwise=True,
            optimistic_56_layer_saving_ms=result["saving_us"] * 56 / 1000,
        )
        results.append(result)
        print(
            json.dumps({k: v for k, v in result.items() if k != "samples_us"}),
            flush=True,
        )
    (args.out / "result.json").write_text(
        json.dumps(
            {
                "results": results,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "generated_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "research_only": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
