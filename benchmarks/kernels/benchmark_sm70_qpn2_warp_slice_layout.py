# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen contiguous warp slices without changing any K accumulation order.

Transpose bundled K groups outside timing so the CTA's warps read adjacent
slices on each loop iteration. Pair read/checksum skeletons and complete MLPs;
the temporary dual-layout microbench is not a serving-memory implementation.
"""

import argparse
import hashlib
import json
import runpy
from functools import partial
from pathlib import Path

import torch
from benchmark_sm70_qpn2_effective_scale import extract_kernel, generate, graph_pair
from torch.utils.cpp_extension import load

from vllm import _sm70_ops as ops


def source_text(source):
    text = generate(source)
    for gated in (False, True):
        name = "nvfp4_qpn2_" + ("gated_" if gated else "") + "sm70_kernel"
        old = extract_kernel(text, name + "_scale1")
        new = extract_kernel(text, name + "_scale0")
        new = new.replace(name + "_scale0", name + "_scale1")
        new = new.replace(
            "reader.load(group)", "reader.load((group-group_begin)*SplitK+warp)"
        )
        new = new.replace(
            "static_cast<size_t>(group) *",
            "static_cast<size_t>((group-group_begin)*SplitK+warp) *",
        )
        text = text.replace(old, new)
    text = text.replace(
        ", reinterpret_cast<const half*>(table.data_ptr<at::Half>())", ""
    )
    skeleton = r"""
template <bool Striped, bool Gated>
__global__ void read_weight_slices(const uint8_t* bundle, uint32_t* sink,
                                  int hidden, int k) {
  constexpr int Split=Gated?8:16;
  const int lane=threadIdx.x%32, warp_id=threadIdx.x/32;
  const int projection=Gated?warp_id/Split:0, warp=warp_id%Split;
  const int groups=k/16, span=groups/Split, start=warp*span;
  const int tile=blockIdx.x+projection*(hidden/32);
  const uint8_t* base=bundle+static_cast<size_t>(tile)*groups*288;
  uint32_t value=0;
#pragma unroll 4
  for (int t=0;t<span;++t) {
    const int physical=Striped?t*Split+warp:start+t;
    const auto packed=__ldcs(reinterpret_cast<const uint2*>(base+physical*288)+lane);
    const uint8_t scale=__ldg(base+physical*288+256+lane);
    value^=packed.x^packed.y^static_cast<uint32_t>(scale);
  }
  sink[blockIdx.x*512+threadIdx.x]=value;
}
void launch_read_skeleton(torch::Tensor sink,torch::Tensor bundle,int hidden,int k,
          bool gated,int mode) {
  auto stream=at::cuda::getCurrentCUDAStream();
  const auto* code=bundle.data_ptr<uint8_t>(); auto* y=sink.data_ptr<uint32_t>();
  if(gated && mode==0) read_weight_slices<false,true><<<hidden/32,512,0,stream>>>(
      code,y,hidden,k);
  if(gated && mode==1) read_weight_slices<true,true><<<hidden/32,512,0,stream>>>(
      code,y,hidden,k);
  if(!gated && mode==0) read_weight_slices<false,false><<<hidden/32,512,0,stream>>>(
      code,y,hidden,k);
  if(!gated && mode==1) read_weight_slices<true,false><<<hidden/32,512,0,stream>>>(
      code,y,hidden,k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""
    text = text.replace("PYBIND11_MODULE(", skeleton + "\nPYBIND11_MODULE(")
    return text.replace(
        'm.def("prepare", &prepare);', 'm.def("read", &launch_read_skeleton);'
    )


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
    generated = args.out / "warp_slice_layout.cu"
    text = source_text(source)
    if not generated.exists() or generated.read_text() != text:
        generated.write_text(text)
    extension = load(
        name="qpn2_warp_slice_layout_screen",
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
    torch.manual_seed(123)
    torch.set_grad_enabled(False)
    empty = torch.empty(0, device="cuda", dtype=torch.float16)
    operands = []
    for p in helpers["_load_projection_shards"](args.model, 0, 0, 4):
        codes, scales = ops.nvfp4_qpn2_prepare_sm70(p.packed.cuda(), p.scales.cuda())
        bundle = torch.cat((codes.view(-1, 256), scales.view(-1, 32)), 1).contiguous()
        split = 8 if p.gated_silu else 16
        groups = p.packed.shape[1] * 2 // 16
        striped = (
            bundle.view(-1, split, groups // split, 288).transpose(1, 2).contiguous()
        )
        striped = striped.view(-1, 288)
        width = p.packed.shape[0] // (2 if p.gated_silu else 1)
        sink = torch.empty(width // 32 * 512, device="cuda", dtype=torch.uint32)
        operands.append((p, codes, scales, bundle, striped, sink))
    x = torch.randn(8, 5120, device="cuda", dtype=torch.float16)
    original = x.clone()
    middle = [x.new_empty(8, 4352) for _ in range(2)]
    output = [torch.empty_like(x) for _ in range(2)]

    def projection(index, mode):
        p, _, scales, bundle, striped, _ = operands[index]
        extension.launch(
            middle[mode] if index == 0 else output[mode],
            x if index == 0 else middle[mode],
            striped if mode else bundle,
            scales,
            empty,
            p.inverse_global_scale,
            index == 0,
            mode,
        )

    def mlp(mode):
        projection(0, mode)
        projection(1, mode)

    for amplitude in (0.01, 0.125, 1.0, 4.0):
        x.copy_(original * amplitude)
        mlp(0)
        mlp(1)
        assert all(
            torch.equal(a.view(torch.int16), b.view(torch.int16))
            for a, b in ((middle[0], middle[1]), (output[0], output[1]))
        )
        p, codes, scales, _, _, _ = operands[0]
        ref = torch.empty_like(middle[0])
        ops.nvfp4_qpn2_gated_sm70_out(
            ref, x, codes, scales, p.inverse_global_scale, 8, 1
        )
        assert torch.equal(ref.view(torch.int16), middle[0].view(torch.int16))
    x.copy_(original * 0.125)
    eviction = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    results = []
    for p, _, _, bundle, striped, sink in operands:
        width = p.packed.shape[0] // (2 if p.gated_silu else 1)
        k = p.packed.shape[1] * 2

        def read(
            mode,
            sink=sink,
            striped=striped,
            bundle=bundle,
            width=width,
            k=k,
            gated=p.gated_silu,
        ):
            extension.read(sink, striped if mode else bundle, width, k, gated, mode)

        read(0)
        checksum = sink.clone()
        read(1)
        assert torch.equal(sink, checksum)
        result = graph_pair(partial(read, 0), partial(read, 1), eviction, args.iters)
        results.append(
            {
                "name": "read_gate" if p.gated_silu else "read_down",
                "logical_payload_bytes": bundle.numel(),
                **result,
            }
        )
    for name, call in (
        ("gate", partial(projection, 0)),
        ("down", partial(projection, 1)),
        ("mlp", mlp),
    ):
        result = graph_pair(partial(call, 0), partial(call, 1), eviction, args.iters)
        results.append({"name": name, **result})
    result = {
        "scope": "Real layer0 TP4 shard, M8 FP16, cold L2 graph",
        "bitwise": True,
        "read_checksum_bitwise": True,
        "results": results,
        "cuda_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }
    (args.out / "result.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            [{k: v for k, v in r.items() if k != "samples_us"} for r in results]
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
