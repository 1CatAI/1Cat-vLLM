# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Research-only cooperative M8 MLP, preserving both projection algorithms.

An 80-CTA cooperative launch loops over the original gate/up and down tiles,
using a grid barrier between projections. Reuse the 16 KiB partial scratch.
No weight prefetch, layout change, split tuning, or decode change is involved.
Cooperative residency is checked before launch; there is no unsafe spin barrier.
"""

import argparse
import hashlib
import json
import runpy
from pathlib import Path

import torch
from benchmark_sm70_gdn_norm_operand import census_pair, time_pair
from torch.utils.cpp_extension import load

from vllm import _sm70_ops as ops


def generate(source, blocks=80):
    original = source.read_text()
    scale_begin = original.index("__device__ __forceinline__ half2 fp8e4m3")
    scale_end = original.index("// QPN2 stores", scale_begin)
    decode_begin = original.index("__device__ __forceinline__ void dequant_e2m1x8")
    decode_end = original.index("// Batch kernels share", decode_begin)
    macro = original[
        original.index("#define VLLM_SM70_QPN2_MMA") : original.index(
            "// Four row tiles reuse"
        )
    ]
    begin = original.rfind(
        "template <", 0, original.index("__global__ void nvfp4_qpn2_sm70_kernel")
    )
    end = original.rfind("template <", begin, original.index("\nvoid launch_qpn2"))
    body = original[begin:end]
    body = body.replace("__global__ void", "__device__ __forceinline__ void")
    body = body.replace(
        "float global_scale) {",
        "float global_scale, int assigned_tile, float* scratch) {",
    )
    body = body.replace(
        "const int tile = TurboMindLayout ? blockIdx.y : blockIdx.x;",
        "const int tile = assigned_tile;",
    ).replace(
        "const int output_tile = TurboMindLayout ? blockIdx.y : blockIdx.x;",
        "const int output_tile = assigned_tile;",
    )
    body = body.replace(
        "  __shared__ float partials[SplitK][RowTiles * 256];",
        "  auto partials = reinterpret_cast<float (*)[RowTiles * 256]>(scratch);",
    ).replace(
        "  __shared__ float partials[2][SplitK][RowTiles * 256];",
        "  auto partials = "
        "reinterpret_cast<float (*)[SplitK][RowTiles * 256]>(scratch);",
    )
    generated = (
        """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include "nvfp4_qpn2_layout.cuh"
namespace {
constexpr int kQpn2RowsPerCta = 8;
"""
        + original[scale_begin:scale_end]
        + original[decode_begin:decode_end]
        + macro
        + body
        + """
__global__ void cooperative_mlp_kernel(
    const uint8_t* gate_w, const uint8_t* gate_s, const uint8_t* down_w,
    const uint8_t* down_s, const half* input, half* intermediate, half* output,
    float gate_scale, float down_scale) {
  __shared__ float scratch[16 * 256];
  for (int tile = blockIdx.x; tile < 136; tile += gridDim.x) {
    nvfp4_qpn2_gated_sm70_kernel<8, 1>(
        gate_w, gate_s, input, intermediate, 4352, 5120, 8, gate_scale,
        tile, scratch);
    __syncthreads();
  }
  cooperative_groups::this_grid().sync();
  for (int tile = blockIdx.x; tile < 160; tile += gridDim.x) {
    nvfp4_qpn2_sm70_kernel<16, 2>(
        down_w, down_s, intermediate, output, 5120, 4352, 8, down_scale,
        tile, scratch);
    __syncthreads();
  }
}

void launch(torch::Tensor output, torch::Tensor input, torch::Tensor intermediate,
            torch::Tensor gate_codes, torch::Tensor gate_scales,
            torch::Tensor down_codes, torch::Tensor down_scales,
            double gate_global, double down_global) {
  TORCH_CHECK(input.sizes() == torch::IntArrayRef({8, 5120}));
  TORCH_CHECK(output.sizes() == input.sizes());
  TORCH_CHECK(intermediate.sizes() == torch::IntArrayRef({8, 4352}));
  int device = 0, supported = 0, active = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &supported, cudaDevAttrCooperativeLaunch, device));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, cooperative_mlp_kernel, 512, 0));
  const auto* props = at::cuda::getDeviceProperties(device);
  TORCH_CHECK(supported && active * props->multiProcessorCount >= 80,
              "The cooperative grid must fit concurrently on this device");
  const uint8_t* gw = gate_codes.data_ptr<uint8_t>();
  const uint8_t* gs = gate_scales.data_ptr<uint8_t>();
  const uint8_t* dw = down_codes.data_ptr<uint8_t>();
  const uint8_t* ds = down_scales.data_ptr<uint8_t>();
  const half* x = reinterpret_cast<const half*>(input.data_ptr<at::Half>());
  half* up = reinterpret_cast<half*>(intermediate.data_ptr<at::Half>());
  half* y = reinterpret_cast<half*>(output.data_ptr<at::Half>());
  float gc = gate_global, dc = down_global;
  void* arguments[] = {&gw, &gs, &dw, &ds, &x, &up, &y, &gc, &dc};
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
      reinterpret_cast<void*>(cooperative_mlp_kernel), dim3(80), dim3(512),
      arguments, 0, at::cuda::getCurrentCUDAStream()));
}
std::vector<int64_t> resources() {
  int active = 0;
  cudaFuncAttributes attributes;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, cooperative_mlp_kernel));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, cooperative_mlp_kernel, 512, 0));
  return {active, attributes.numRegs,
          static_cast<int64_t>(attributes.sharedSizeBytes),
          static_cast<int64_t>(attributes.localSizeBytes)};
}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launch", &launch); m.def("resources", &resources);
}
"""
    )
    if blocks == 160:
        generated = generated.replace(
            "__global__ void cooperative_mlp_kernel(",
            "__global__ __launch_bounds__(512, 2) void cooperative_mlp_kernel(",
        ).replace("multiProcessorCount >= 80", "multiProcessorCount >= 160")
        generated = generated.replace("dim3(80), dim3(512)", "dim3(160), dim3(512)")
    return generated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--blocks", type=int, choices=(80, 160), default=80)
    parser.add_argument("--census-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.manual_seed(123)
    assert torch.cuda.get_device_capability() == (7, 0)
    original = args.source_root / "csrc/sm70_turbomind/ops/nvfp4_qpn2_sm70.cu"
    generated = args.out / "cooperative_mlp.cu"
    text = generate(original, args.blocks)
    if not generated.exists() or generated.read_text() != text:
        generated.write_text(text)
    module = load(
        name=f"round12_qpn2_cooperative_mlp_{args.blocks}",
        sources=[str(generated)],
        extra_include_paths=[str(original.parent)],
        extra_cuda_cflags=["-O3", "-lineinfo", "--ptxas-options=-v"],
        verbose=True,
    )
    helpers = runpy.run_path(
        str(Path(__file__).with_name("benchmark_sm70_nvfp4_qpn2.py"))
    )
    projections = []
    for projection in helpers["_load_projection_shards"](args.model, 0, 0, 4):
        weights, scales = ops.nvfp4_qpn2_prepare_sm70(
            projection.packed.cuda(), projection.scales.cuda()
        )
        projections.append((weights, scales, projection.inverse_global_scale))
    (gw, gs, gc), (dw, ds, dc) = projections
    x = torch.randn(8, 5120, device="cuda", dtype=torch.float16)
    up, out = x.new_empty(8, 4352), x.new_empty(8, 5120)

    def baseline():
        ops.nvfp4_qpn2_gated_sm70_out(up, x, gw, gs, gc, 8, 1)
        ops.nvfp4_qpn2_gemm_sm70_out(out, up, dw, ds, dc, 16, 2)

    def candidate():
        module.launch(out, x, up, gw, gs, dw, ds, gc, dc)

    for amplitude in (0.01, 0.125, 1.0, 4.0):
        x.normal_().mul_(amplitude)
        baseline()
        expected_up, expected_out = up.clone(), out.clone()
        candidate()
        assert torch.equal(up.view(torch.int16), expected_up.view(torch.int16))
        assert torch.equal(out.view(torch.int16), expected_out.view(torch.int16))
    result = (
        census_pair((baseline, candidate))
        if args.census_only
        else time_pair((baseline, candidate), args.iters)
    )
    result.update(
        source_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
        generated_sha256=hashlib.sha256(generated.read_bytes()).hexdigest(),
        production_dispatch_changed=False,
        intermediate_and_output_bitwise=True,
        cold_l2_bytes=32 * 1024 * 1024,
        expected_compute_kernel_nodes=[2, 1],
        grid_blocks=args.blocks,
        cooperative_residency_checked=True,
        kernel_resources_order=["resident_ctas_per_sm", "registers", "shared", "local"],
        kernel_resources=module.resources(),
    )
    filename = "census.json" if args.census_only else "result.json"
    (args.out / filename).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "samples_us"}))


if __name__ == "__main__":
    main()
