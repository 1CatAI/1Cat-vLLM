# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen magnitude-only product tables with per-layer shared-memory staging.

The original products are rounded before sign reconstruction. Only the finite
real-weight case is screened. This is not a production format or dispatch.
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
from benchmark_sm70_qpn2_product_lookup import generate as full_product_source
from torch.utils.cpp_extension import load

from vllm import _sm70_ops as ops


def generate(root):
    text = full_product_source(root)
    begin = text.index("__device__ __forceinline__ void product_decode")
    end = text.index("struct ProductReader", begin)
    text = (
        text[:begin]
        + r"""
template<int Byte>
__device__ __forceinline__ half2 magnitude_product(uint32_t word,
                                                 const uint32_t* row) {
  uint32_t signs;
  constexpr uint32_t selector=((8+Byte)<<12)|((12+Byte)<<4);
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(signs)
      : "r"(word), "r"(word<<1), "n"(selector));
  uint32_t bits=row[(word>>(8*Byte))&63]^(signs&0x80008000);
  return *reinterpret_cast<const half2*>(&bits);
}
__device__ __forceinline__ void product_decode(uint2 code,uint8_t scale,
                             const uint32_t* table,int minimum,half2* out) {
  const uint32_t* row=table+(static_cast<int>(scale)-minimum)*65;
  out[0]=magnitude_product<0>(code.x,row);
  out[1]=magnitude_product<1>(code.x,row);
  out[2]=magnitude_product<2>(code.x,row);
  out[3]=magnitude_product<3>(code.x,row);
  out[4]=magnitude_product<0>(code.y,row);
  out[5]=magnitude_product<1>(code.y,row);
  out[6]=magnitude_product<2>(code.y,row);
  out[7]=magnitude_product<3>(code.y,row);
}
"""
        + text[end:]
    )
    text = (
        text.replace(
            "const uint8_t* scales; const uint32_t* lookup;",
            "const uint8_t* scales; const uint32_t* lookup; int minimum;",
        )
        .replace(
            "int lane,float,const uint32_t* table)",
            "int lane,float,const uint32_t* table,int first)",
        )
        .replace("lane),lookup(table) {}", "lane),lookup(table),minimum(first) {}")
        .replace(
            "product_decode(codes.load(group),__ldg(scales+group*288),lookup,weights);",
            "product_decode(codes.load(group),__ldg(scales+group*288),lookup,minimum,weights);",
        )
    )
    text = text.replace(
        "uint32_t byte=((word>>(4*j))&15)|((word>>(12+4*j))&240);",
        "uint32_t u=word>>(4*j),v=word>>(16+4*j);\n"
        "    uint32_t byte=(u&7)|((v&7)<<3)|((u&8)<<3)|((v&8)<<4);",
    ).replace(
        "const int byte=i&255,scale=i>>8;\n"
        "  const uint32_t word=(byte&15)|((byte&240)<<12);",
        "const int byte=i%65,scale=i/65;\n"
        "  const uint32_t word=(byte&7)|((byte&56)<<13);",
    )
    text = text.replace("table.numel()==65536", "table.numel()==16640")
    text = text.replace(
        "prepare_product_table<<<256,256,0,stream>>>",
        "prepare_product_table<<<65,256,0,stream>>>",
    )
    # The two candidate kernels share the same table initialization protocol.
    for name in ("product_gate_kernel", "product_down_kernel"):
        begin = text.index(name)
        body = text.index("{", begin)
        signature = text[begin:body].replace(
            "const uint32_t* lookup)",
            "const uint32_t* lookup,int minimum,int count)",
        )
        assert "int count)" in signature
        text = text[:begin] + signature + text[body:]
        body = begin + len(signature)
        text = (
            text[: body + 1]
            + r"""
  extern __shared__ uint32_t shared_products[];
  if(count) {
    for(int i=threadIdx.x;i<count*65;i+=blockDim.x)
      shared_products[i]=__ldg(lookup+minimum*65+i);
    __syncthreads();
    lookup=shared_products;
  }
"""
            + text[body + 1 :]
        )
    text = text.replace("lane, global, lookup)", "lane, global, lookup, minimum)")
    text = text.replace(
        "product_decode(packed,scale_code,lookup,weights);",
        "product_decode(packed,scale_code,lookup,minimum,weights);",
    )
    text = text.replace(
        "double global,bool gated) {",
        "double global,bool gated,int minimum,int count) {",
    )
    text = text.replace(
        "<<<136,256,0,stream>>>(\n    c,s,x,y,4352,5120,global,lut)",
        "<<<136,256,count*65*4,stream>>>(\n"
        "    c,s,x,y,4352,5120,global,lut,minimum,count)",
    )
    text = text.replace(
        "<<<160,512,0,stream>>>(\n    c,s,x,y,5120,4352,8,global,lut)",
        "<<<160,512,count*65*4,stream>>>(\n"
        "    c,s,x,y,5120,4352,8,global,lut,minimum,count)",
    )
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--global-table", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.out / "shared_product.cu"
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
            name="sm70_qpn2_shared_product_screen",
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
        table = torch.empty(16640, dtype=torch.int32, device="cuda")
        extension.prepare_lookup(
            converted, table, bundle, projection.inverse_global_scale
        )
        scale_codes = projection.scales.view(torch.uint8)
        minimum = int(scale_codes.min())
        count = int(scale_codes.max()) - minimum + 1
        if args.global_table:
            minimum, count = 0, 0
        operands.append(
            (
                bundle,
                converted,
                scales,
                table,
                projection.inverse_global_scale,
                minimum,
                count,
            )
        )
    x = torch.empty(8, 5120, device="cuda", dtype=torch.float16)
    up = torch.empty(8, 4352, device="cuda", dtype=torch.float16)
    down = torch.empty_like(x)
    empty = torch.empty(0, dtype=torch.float16)

    def run(candidate):
        for gated, (result, value), item in zip(
            (True, False), ((up, x), (down, up)), operands
        ):
            bundle, converted, scales, table, scale, minimum, count = item
            if candidate:
                extension.lookup(
                    result,
                    value,
                    converted,
                    scales,
                    table,
                    scale,
                    gated,
                    minimum,
                    count,
                )
            elif gated:
                extension.pair(result, value, bundle, scales, scale, 1)
            else:
                extension.launch(result, value, bundle, scales, empty, scale, False, 0)

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
        global_table=args.global_table,
        scale_ranges=[list(item[-2:]) for item in operands],
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
