# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# CUDA source anchors retain their original spelling.
# ruff: noqa: E501
"""Distribute the existing eight paired K slices across resident four-warp CTAs.

This keeps each slice and the final ordered FP32 reduction unchanged. It adds
an explicit inter-CTA partial buffer and release flags, whose complete MLP cost
is included. Cooperative residency is required; there is no grid barrier.
"""

import argparse
import hashlib
import importlib.util
import json
import runpy
from functools import partial
from pathlib import Path

import torch
from benchmark_sm70_qpn2_effective_scale import graph_pair
from benchmark_sm70_qpn2_paired_gate import generate as paired_source
from benchmark_sm70_qpn2_product_lookup import function_end
from torch.utils.cpp_extension import load

from vllm import _sm70_ops as ops


def generate(root):
    text = paired_source(root / "csrc/sm70_turbomind/ops/nvfp4_qpn2_sm70.cu")
    text = text[: text.index("PYBIND11_MODULE(")]
    begin = text.index("template <bool Interleave>")
    gate = text[begin : function_end(text, begin)]
    gate = gate.replace("paired_gate_kernel", "resident_slice_gate")
    gate = gate.replace("__launch_bounds__(256, 3)", "__launch_bounds__(128, 4)")
    gate = gate.replace(
        "float global) {", "float global, float* scratch, uint32_t* signals) {"
    )
    gate = gate.replace(
        "  __shared__ float partials[2][Split][256];",
        r"""
  const int tile=blockIdx.x/2,part=blockIdx.x%2;
  volatile float (*partials)[Split][256]=
      reinterpret_cast<volatile float (*)[Split][256]>(scratch+tile*2*Split*256);
  volatile uint32_t* ready=signals+tile*2;
  const uint32_t generation=ready[part]+1;
""",
    )
    gate = gate.replace(
        "const int warp = threadIdx.x >> 5;",
        "const int warp = (threadIdx.x >> 5)+part*4;",
    )
    gate = gate.replace("codes, scales, blockIdx.x,", "codes, scales, tile,")
    gate = gate.replace(
        "codes, scales, blockIdx.x + hidden / 32,", "codes, scales, tile + hidden / 32,"
    )
    begin = gate.index("  __syncthreads();\n  const int e =")
    gate = (
        gate[:begin]
        + r"""
  __threadfence();
  __syncthreads();
  if(part==0) {
    if(threadIdx.x==0) ready[0]=generation;
    return;
  }
  if(threadIdx.x==0) while(ready[0]!=generation) {}
  __syncthreads();
#pragma unroll
  for(int pass=0;pass<2;++pass) {
    const int e=threadIdx.x+pass*128;
    float gate=0.f,up=0.f;
#pragma unroll
    for(int w=0;w<Split;++w) {
      gate+=partials[0][w][e];up+=partials[1][w][e];
    }
    const half gh=__float2half(gate),uh=__float2half(up);
    const float gf=__half2float(gh);
    const half silu=__float2half(gf/(1.f+expf(-gf)));
    output[static_cast<size_t>(e/32)*hidden+tile*32+e%32]=__hmul(silu,uh);
  }
  __syncthreads();
  if(threadIdx.x==0) ready[1]=generation;
}
"""
    )
    wrapper = r"""
void slices(torch::Tensor output,torch::Tensor input,torch::Tensor codes,
            torch::Tensor scales,double global,torch::Tensor scratch,
            torch::Tensor signals) {
  TORCH_CHECK(input.sizes()==torch::IntArrayRef({8,5120}) && input.is_contiguous());
  TORCH_CHECK(output.sizes()==torch::IntArrayRef({8,4352}) && output.is_contiguous());
  TORCH_CHECK(scratch.numel()==136*2*8*256 && scratch.scalar_type()==torch::kFloat32);
  TORCH_CHECK(signals.numel()==272 && signals.scalar_type()==torch::kInt32);
  int device,resident;
  C10_CUDA_CHECK(cudaGetDevice(&device));cudaDeviceProp prop;
  C10_CUDA_CHECK(cudaGetDeviceProperties(&prop,device));
  auto kernel=resident_slice_gate<false>;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel,128,0));
  TORCH_CHECK(prop.cooperativeLaunch && resident*prop.multiProcessorCount>=272,
              "All slice CTAs must be admitted resident");
  const auto* c=codes.data_ptr<uint8_t>();const auto* s=scales.data_ptr<uint8_t>();
  const auto* x=reinterpret_cast<const half*>(input.data_ptr<at::Half>());
  auto* y=reinterpret_cast<half*>(output.data_ptr<at::Half>());
  auto* p=scratch.data_ptr<float>();auto* r=reinterpret_cast<uint32_t*>(signals.data_ptr<int32_t>());
  int hidden=4352,k=5120;float g=global;
  void* args[]={&c,&s,&x,&y,&hidden,&k,&g,&p,&r};
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(kernel),dim3(272),
                 dim3(128),args,0,at::cuda::getCurrentCUDAStream()));
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("launch",&launch);m.def("pair",&launch_pair);m.def("slices",&slices);
}
"""
    return text + "\nnamespace {\n" + gate + "\n}\n" + wrapper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--extension", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.out / "resident_slices.cu"
    source.write_text(generate(args.source_root))
    if args.extension:
        spec = importlib.util.spec_from_file_location(
            args.extension.name.split(".")[0], args.extension
        )
        assert spec and spec.loader
        extension = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(extension)
    else:
        extension = load(
            name="sm70_qpn2_resident_slices_screen",
            sources=[str(source)],
            extra_include_paths=[str(args.source_root / "csrc/sm70_turbomind/ops")],
            extra_cuda_cflags=["-O3", "-lineinfo", "--ptxas-options=-v"],
            verbose=True,
        )
    if args.compile_only:
        return
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    torch.manual_seed(123)
    loader = runpy.run_path(
        str(Path(__file__).with_name("benchmark_sm70_nvfp4_qpn2.py"))
    )["_load_projection_shards"]
    operands = []
    for projection in loader(args.model, 0, 0, 4):
        codes, scales = ops.nvfp4_qpn2_prepare_sm70(
            projection.packed.cuda(), projection.scales.cuda()
        )
        bundle = torch.cat((codes.view(-1, 256), scales.view(-1, 32)), 1).contiguous()
        operands.append((bundle, scales, projection.inverse_global_scale))
    x = torch.empty(8, 5120, device="cuda", dtype=torch.float16)
    up = torch.empty(8, 4352, device="cuda", dtype=torch.float16)
    down = torch.empty_like(x)
    scratch = torch.empty(136 * 2 * 8 * 256, device="cuda", dtype=torch.float32)
    signals = torch.zeros(272, device="cuda", dtype=torch.int32)
    empty = torch.empty(0, dtype=torch.float16)

    def run(candidate):
        bundle, scales, scale = operands[0]
        if candidate:
            extension.slices(up, x, bundle, scales, scale, scratch, signals)
        else:
            extension.pair(up, x, bundle, scales, scale, 1)
        bundle, scales, scale = operands[1]
        extension.launch(down, up, bundle, scales, empty, scale, False, 0)

    for amplitude in (0.01, 0.125, 1.0, 4.0):
        x.normal_(0, amplitude)
        run(False)
        golden = [up.clone(), down.clone()]
        run(True)
        for actual, expected in zip((up, down), golden):
            assert torch.isfinite(actual).all()
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    x.normal_(0, 1.0)
    eviction = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    result = graph_pair(partial(run, False), partial(run, True), eviction, args.iters)
    result.update(
        bitwise=True,
        scratch_bytes=scratch.numel() * scratch.element_size(),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        library_sha256=hashlib.sha256(
            Path(extension.__file__).read_bytes()
        ).hexdigest(),
    )
    args.out.joinpath("result.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps({k: v for k, v in result.items() if k != "samples_us"}), flush=True
    )


if __name__ == "__main__":
    main()
