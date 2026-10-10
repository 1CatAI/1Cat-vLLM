# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen exact scale/code half2 lookup in cold real-weight M8 MLP graphs.

Interleave the two nibbles consumed by each HMMA half2 in the existing 288-byte
native bundle. A 256-KiB table contains the exact original FP16 products of
every scale/code pair. This removes bias/scale arithmetic without widening the
weight stream. Tables and candidate layout are prepared outside timing; they
are research operands, not a serving layout or additional precision change.
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
from torch.utils.cpp_extension import load

from vllm import _sm70_ops as ops


def function_end(text, start):
    cursor = text.index("{", start) + 1
    depth = 1
    while depth:
        depth += (text[cursor] == "{") - (text[cursor] == "}")
        cursor += 1
    return cursor


def generate(root):
    text = paired_source(root / "csrc/sm70_turbomind/ops/nvfp4_qpn2_sm70.cu")
    text = text[: text.index("PYBIND11_MODULE(")]
    start = text.index("template <bool Interleave>")
    gate = text[start : function_end(text, start)]
    gate = gate.replace("paired_gate_kernel", "product_gate_kernel")
    gate = gate.replace("float global) {", "float global, const uint32_t* lookup) {")
    gate = gate.replace("Nvfp4PairReader<false, true>", "ProductReader")
    gate = gate.replace("lane, global)", "lane, global, lookup)")
    assert gate.count("global, lookup)") == 2
    down = extract_kernel(text, "nvfp4_qpn2_sm70_kernel_scale0")
    down = down.replace("nvfp4_qpn2_sm70_kernel_scale0", "product_down_kernel")
    down = down.replace(
        "int m, float global_scale) {",
        "int m, float global_scale, const uint32_t* lookup) {",
    )
    begin = down.index("    const half2 scale =")
    end = down.index("    const unsigned* b =", begin)
    down = (
        down[:begin]
        + r"""
    const uint8_t scale_code=__ldg(scale_ptr+static_cast<size_t>(group)*288);
    half2 weights[8];
    product_decode(packed,scale_code,lookup,weights);
"""
        + down[end:]
    )
    additions = r"""
namespace {
__device__ __forceinline__ void product_decode(uint2 code,uint8_t scale,
                                               const uint32_t* table,half2* out) {
  const uint32_t* row=table+static_cast<int>(scale)*256;
#pragma unroll
  for(int j=0;j<4;++j) {
    const uint32_t a=__ldg(row+((code.x>>(8*j))&255));
    const uint32_t b=__ldg(row+((code.y>>(8*j))&255));
    out[j]=*reinterpret_cast<const half2*>(&a);
    out[j+4]=*reinterpret_cast<const half2*>(&b);
  }
}
struct ProductReader {
  Nvfp4Qpn2CodeReader<false,false,true> codes;
  const uint8_t* scales; const uint32_t* lookup;
  __device__ ProductReader(const uint8_t* w,const void*,int tile,int groups,
                          int lane,float,const uint32_t* table)
      :codes(w,tile,groups,lane),
       scales(w+static_cast<size_t>(tile)*groups*288+256+lane),lookup(table) {}
  __device__ __forceinline__ void load(int group,half2* weights) const {
    product_decode(codes.load(group),__ldg(scales+group*288),lookup,weights);
  }
};
__device__ __forceinline__ uint32_t interleave_pairs(uint32_t word) {
  uint32_t result=0;
#pragma unroll
  for(int j=0;j<4;++j) {
    uint32_t byte=((word>>(4*j))&15)|((word>>(12+4*j))&240);
    result|=byte<<(8*j);
  }
  return result;
}
__global__ void prepare_product_codes(uint8_t* dst,const uint8_t* src,int groups) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<groups*32) {
    const int group=i/32,lane=i%32;
    const uint2 raw=reinterpret_cast<const uint2*>(src+group*288)[lane];
    reinterpret_cast<uint2*>(dst+group*288)[lane]=
      make_uint2(interleave_pairs(raw.x),interleave_pairs(raw.y));
    dst[group*288+256+lane]=src[group*288+256+lane];
  }
}
__global__ void prepare_product_table(uint32_t* table,float global) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  const int byte=i&255,scale=i>>8;
  const uint32_t word=(byte&15)|((byte&240)<<12);
  half2 weights[4];
  dequant_e2m1x8(word,nvfp4_effective_scale(scale,global),weights);
  table[i]=*reinterpret_cast<uint32_t*>(&weights[0]);
}
"""
    wrapper = r"""
void prepare_lookup(torch::Tensor dst,torch::Tensor table,
                    torch::Tensor src,double global) {
  TORCH_CHECK(src.is_contiguous() && dst.sizes()==src.sizes() && table.numel()==65536);
  auto stream=at::cuda::getCurrentCUDAStream();const int groups=src.numel()/288;
  prepare_product_codes<<<(groups*32+255)/256,256,0,stream>>>(
    dst.data_ptr<uint8_t>(),src.data_ptr<uint8_t>(),groups);
  prepare_product_table<<<256,256,0,stream>>>(
    reinterpret_cast<uint32_t*>(table.data_ptr<int32_t>()),global);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void lookup(torch::Tensor output,torch::Tensor input,torch::Tensor codes,
            torch::Tensor scales,torch::Tensor table,double global,bool gated) {
  TORCH_CHECK(input.size(0)==8 && input.is_contiguous() && codes.is_contiguous());
  auto* x=reinterpret_cast<const half*>(input.data_ptr<at::Half>());
  auto* y=reinterpret_cast<half*>(output.data_ptr<at::Half>());
  auto* c=codes.data_ptr<uint8_t>();auto* s=scales.data_ptr<uint8_t>();
  auto* lut=reinterpret_cast<const uint32_t*>(table.data_ptr<int32_t>());
  auto stream=at::cuda::getCurrentCUDAStream();
  if(gated) product_gate_kernel<false><<<136,256,0,stream>>>(
    c,s,x,y,4352,5120,global,lut);
  else product_down_kernel<16,2,1,false,false,true><<<160,512,0,stream>>>(
    c,s,x,y,5120,4352,8,global,lut);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("launch",&launch);m.def("pair",&launch_pair);
  m.def("lookup",&lookup);m.def("prepare_lookup",&prepare_lookup);
}
"""
    return text + additions + gate + "\n" + down + "\n}\n" + wrapper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--ncu-direct", action="store_true")
    parser.add_argument("--ncu-reference", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.out / "product_lookup.cu"
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
            name="sm70_qpn2_product_lookup_screen",
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
        converted = torch.empty_like(bundle)
        table = torch.empty(65536, dtype=torch.int32, device="cuda")
        extension.prepare_lookup(
            converted, table, bundle, projection.inverse_global_scale
        )
        operands.append(
            (bundle, converted, scales, table, projection.inverse_global_scale)
        )
    x = torch.empty(8, 5120, device="cuda", dtype=torch.float16)
    up = torch.empty(8, 4352, device="cuda", dtype=torch.float16)
    down = torch.empty_like(x)
    empty = torch.empty(0, dtype=torch.float16)

    def run(candidate):
        for gated, (result, input_value), item in zip(
            (True, False), ((up, x), (down, up)), operands
        ):
            bundle, converted, scales, table, scale = item
            if candidate:
                extension.lookup(
                    result, input_value, converted, scales, table, scale, gated
                )
            elif gated:
                extension.pair(result, input_value, bundle, scales, scale, 1)
            else:
                extension.launch(
                    result, input_value, bundle, scales, empty, scale, False, 0
                )

    if args.ncu_direct:
        x.normal_(0, 1.0)
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda").fill_(1)
        torch.cuda.synchronize()
        run(not args.ncu_reference)
        torch.cuda.synchronize()
        return
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
    results = graph_pair(partial(run, False), partial(run, True), eviction, args.iters)
    results.update(
        bitwise=True,
        timed_activation_std=1.0,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        library_sha256=hashlib.sha256(
            Path(extension.__file__).read_bytes()
        ).hexdigest(),
    )
    args.out.joinpath("result.json").write_text(json.dumps(results, indent=2))
    print(
        json.dumps({k: v for k, v in results.items() if k != "samples_us"}), flush=True
    )


if __name__ == "__main__":
    main()
