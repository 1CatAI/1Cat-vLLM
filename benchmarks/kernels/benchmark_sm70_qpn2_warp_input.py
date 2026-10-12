# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Coalesce M8 activation fragments in the existing grouped-M32 input layout.

One warp reads a contiguous K16/M8 activation block, then exchanges registers.
Gate/up produces that same grouped layout for down, so only the initial pack
is an extra kernel. Both its cost and the complete MLP are measured; this does
not change a production tensor's layout or install a hidden dispatch.
"""

import argparse
import hashlib
import importlib.util
import json
import runpy
from functools import partial
from pathlib import Path

import torch
from benchmark_sm70_qpn2_effective_scale import extract_kernel, graph_pair
from benchmark_sm70_qpn2_paired_gate import generate as paired_source
from benchmark_sm70_qpn2_product_lookup import function_end
from torch.utils.cpp_extension import load

from vllm import _sm70_ops as ops

WARP_INPUT = r"""
    unsigned a0[4],a1[4];
    if constexpr(WarpLoad) {
    const int owner=((lane&3)<<2)|(lane&16);
    const uint2 data=*reinterpret_cast<const uint2*>(
        input+static_cast<size_t>(group)*128+lane*4);
    a0[0]=__shfl_sync(0xffffffff,data.x,owner);
    a0[1]=__shfl_sync(0xffffffff,data.y,owner);
    a0[2]=__shfl_sync(0xffffffff,data.x,owner+1);
    a0[3]=__shfl_sync(0xffffffff,data.y,owner+1);
    a1[0]=__shfl_sync(0xffffffff,data.x,owner+2);
    a1[1]=__shfl_sync(0xffffffff,data.y,owner+2);
    a1[2]=__shfl_sync(0xffffffff,data.x,owner+3);
    a1[3]=__shfl_sync(0xffffffff,data.y,owner+3);
    } else {
      const int input_row=(lane&3)+((lane&16) ? 4 : 0);
      const half* a=input+static_cast<size_t>(group)*128+input_row*16;
      const uint4 low=*reinterpret_cast<const uint4*>(a);
      const uint4 high=*reinterpret_cast<const uint4*>(a+8);
      const unsigned* l=reinterpret_cast<const unsigned*>(&low);
      const unsigned* h=reinterpret_cast<const unsigned*>(&high);
#pragma unroll
      for(int i=0;i<4;++i) {a0[i]=l[i];a1[i]=h[i];}
    }
"""


def generate(root):
    text = paired_source(root / "csrc/sm70_turbomind/ops/nvfp4_qpn2_sm70.cu")
    text = text[: text.index("PYBIND11_MODULE(")]
    begin = text.index("template <bool Interleave>")
    gate = text[begin : function_end(text, begin)].replace(
        "paired_gate_kernel", "warp_input_gate"
    )
    gate = gate.replace(
        "template <bool Interleave>", "template <bool Interleave,bool WarpLoad>"
    )
    begin = gate.index("    const half* a = input +")
    end = gate.index("    if constexpr (Interleave)", begin)
    gate = gate[:begin] + WARP_INPUT + gate[end:]
    old = "output[static_cast<size_t>(e / 32) * hidden + blockIdx.x * 32 + e % 32]"
    assert gate.count(old) == 1
    gate = gate.replace(
        old,
        "output[static_cast<size_t>(blockIdx.x)*256+((e%32)/16)*128+(e/32)*16+e%16]",
    )
    down = extract_kernel(text, "nvfp4_qpn2_sm70_kernel_scale0")
    down = down.replace("nvfp4_qpn2_sm70_kernel_scale0", "warp_input_down")
    down = down.replace(
        "bool BundledScales = false>",
        "bool BundledScales = false,bool WarpLoad = true>",
    )
    begin = down.index("      uint4 input01 =")
    end = down.index("      VLLM_SM70_QPN2_MMA", begin)
    down = down[:begin] + WARP_INPUT + down[end:]
    additions = r"""
__global__ void pack_m8_input(half* output,const half* input,int k) {
  const int e=blockIdx.x*blockDim.x+threadIdx.x;
  if(e<k*8) {
    const int row=e/k,col=e%k;
    output[static_cast<size_t>(col/16)*128+row*16+col%16]=input[e];
  }
}
"""
    wrapper = r"""
void pack(torch::Tensor output,torch::Tensor input) {
  TORCH_CHECK(input.size(0)==8 && output.sizes()==input.sizes());
  pack_m8_input<<<(input.numel()+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(input.data_ptr<at::Half>()),input.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void warp_input(torch::Tensor output,torch::Tensor input,torch::Tensor codes,
                torch::Tensor scales,double global,bool gated,bool warp_load) {
  TORCH_CHECK(input.size(0)==8 && input.is_contiguous() && codes.is_contiguous());
  const auto* c=codes.data_ptr<uint8_t>();const auto* s=scales.data_ptr<uint8_t>();
  const auto* x=reinterpret_cast<const half*>(input.data_ptr<at::Half>());
  auto* y=reinterpret_cast<half*>(output.data_ptr<at::Half>());
  auto stream=at::cuda::getCurrentCUDAStream();
  if(gated) {
    auto kernel=warp_load ? warp_input_gate<false,true> : warp_input_gate<false,false>;
    kernel<<<136,256,0,stream>>>(c,s,x,y,4352,5120,global);
  } else {
    auto kernel=warp_load ? warp_input_down<16,2,1,false,false,true,true>
                         : warp_input_down<16,2,1,false,false,true,false>;
    kernel<<<160,512,0,stream>>>(c,s,x,y,5120,4352,8,global);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("launch",&launch);m.def("pair",&launch_pair);
  m.def("pack",&pack);m.def("warp_input",&warp_input);
}
"""
    return text + "\nnamespace {\n" + additions + gate + down + "\n}\n" + wrapper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--entry-already-packed", action="store_true")
    parser.add_argument("--plain-packed-input", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.out / "warp_input.cu"
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
            name="sm70_qpn2_packed_input_screen",
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
    packed = torch.empty_like(x)
    up = torch.empty(8, 4352, device="cuda", dtype=torch.float16)
    down = torch.empty_like(x)
    empty = torch.empty(0, dtype=torch.float16)

    def run(candidate):
        bundle, scales, scale = operands[0]
        if candidate:
            if not args.entry_already_packed:
                extension.pack(packed, x)
            extension.warp_input(
                up, packed, bundle, scales, scale, True, not args.plain_packed_input
            )
        else:
            extension.pair(up, x, bundle, scales, scale, 1)
        bundle, scales, scale = operands[1]
        if candidate:
            extension.warp_input(
                down, up, bundle, scales, scale, False, not args.plain_packed_input
            )
        else:
            extension.launch(down, up, bundle, scales, empty, scale, False, 0)

    for amplitude in (0.01, 0.125, 1.0, 4.0):
        x.normal_(0, amplitude)
        extension.pack(packed, x)
        run(False)
        golden = [up.clone(), down.clone()]
        run(True)
        unpacked = up.view(4352 // 16, 8, 16).permute(1, 0, 2).reshape(8, 4352)
        for actual, expected in zip((unpacked, down), golden):
            assert torch.isfinite(actual).all()
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    x.normal_(0, 1.0)
    extension.pack(packed, x)
    eviction = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    result = graph_pair(partial(run, False), partial(run, True), eviction, args.iters)
    result.update(
        bitwise=True,
        entry_pack_included=not args.entry_already_packed,
        cooperative_warp_load=not args.plain_packed_input,
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
