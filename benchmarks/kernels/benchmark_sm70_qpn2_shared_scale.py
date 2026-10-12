# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cache a projection's exact effective-scale table once per CTA.

This differs from rebuilding scaled FP4 value tables per group. Keep original
FP4 decoding, HMMA order and bundled weights. One 512B/1KiB shared lookup replaces
scale conversion and multiplication throughout K, with no added weight layout.
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


def source_for_screen(source):
    body = generate(source)
    body = body[: body.index("PYBIND11_MODULE(")]
    variants, calls = [], []
    for gated in (False, True):
        name = "nvfp4_qpn2_" + ("gated_" if gated else "") + "sm70_kernel_scale1"
        kernel = extract_kernel(body, name)
        old = """    const half h = __ldg(scale_table + __ldg(
        scale_ptr + static_cast<size_t>(group) * 288));
    const half2 scale = __halves2half2(h, h);
"""
        assert kernel.count(old) == 1
        for mode in (3, 4):
            variant = kernel.replace(name, name + f"_shared{mode}")
            declaration = (
                "  __shared__ half cached_scale[256];\n"
                "  if (threadIdx.x < 256) cached_scale[threadIdx.x] = "
                "scale_table[threadIdx.x];\n"
                if mode == 3
                else "  __shared__ half2 cached_scale[256];\n"
                "  if (threadIdx.x < 256) {\n"
                "    const half h = scale_table[threadIdx.x];\n"
                "    cached_scale[threadIdx.x] = __halves2half2(h, h);\n"
                "  }\n"
            )
            begin = variant.index("{", variant.index("__global__ void"))
            variant = (
                variant[: begin + 1]
                + "\n"
                + declaration
                + "  __syncthreads();\n"
                + variant[begin + 1 :]
            )
            replacement = (
                "    const half h = cached_scale[__ldg(\n"
                "        scale_ptr + static_cast<size_t>(group) * 288)];\n"
                "    const half2 scale = __halves2half2(h, h);\n"
                if mode == 3
                else "    const half2 scale = cached_scale[__ldg(\n"
                "        scale_ptr + static_cast<size_t>(group) * 288)];\n"
            )
            variants.append(variant.replace(old, replacement))
            template = "8,1,1,false,true" if gated else "16,2,1,false,false,true"
            calls.append(
                f"if (mode=={mode} && gated=={str(gated).lower()}) "
                f"{name}_shared{mode}<{template}><<<width/32,512,0,stream>>>"
                "(c,s,x,y,width,k,8,global,tab);"
            )
    return (
        body
        + "\n".join(variants)
        + r"""
void table_only(torch::Tensor table, double global) {
  TORCH_CHECK(table.numel()==256 && table.scalar_type()==torch::kFloat16);
  prepare_table<<<1,256,0,at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<half*>(table.data_ptr<at::Half>()),global);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void shared(torch::Tensor output, torch::Tensor input, torch::Tensor codes,
            torch::Tensor scales, torch::Tensor table, double global,
            bool gated, int mode) {
  TORCH_CHECK(input.size(0)==8 && input.is_contiguous() &&
              output.is_contiguous() && (mode==3 || mode==4));
  auto stream=at::cuda::getCurrentCUDAStream();
  auto* c=codes.data_ptr<uint8_t>(); auto* s=scales.data_ptr<uint8_t>();
  auto* x=reinterpret_cast<const half*>(input.data_ptr<at::Half>());
  auto* y=reinterpret_cast<half*>(output.data_ptr<at::Half>());
  auto* tab=reinterpret_cast<const half*>(table.data_ptr<at::Half>());
  int width=output.size(1),k=input.size(1);
"""
        + "\n".join(calls)
        + r"""
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("launch",&launch);m.def("shared",&shared);m.def("table",&table_only);
}
"""
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.source_root / "csrc/sm70_turbomind/ops/nvfp4_qpn2_sm70.cu"
    text = source_for_screen(source)
    generated = args.out / "shared_scale.cu"
    if not generated.exists() or generated.read_text() != text:
        generated.write_text(text)
    extension = load(
        name="qpn2_shared_scale_screen",
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
    operands = []
    for p in helpers["_load_projection_shards"](args.model, 0, 0, 4):
        codes, scales = ops.nvfp4_qpn2_prepare_sm70(p.packed.cuda(), p.scales.cuda())
        bundle = torch.cat(
            (codes.view(-1, 256), scales.view(-1, 32)), dim=1
        ).contiguous()
        table = torch.empty(256, device="cuda", dtype=torch.float16)
        extension.table(table, p.inverse_global_scale)
        operands.append((p, codes, scales, bundle, table))
    x = torch.randn(8, 5120, device="cuda", dtype=torch.float16)
    original = x.clone()
    middle = [torch.empty(8, 4352, device="cuda", dtype=x.dtype) for _ in range(5)]
    outputs = [torch.empty_like(x) for _ in range(5)]

    def projection(item, operand, output, mode):
        p, _, scales, bundle, table = item
        if mode == 0:
            extension.launch(
                output,
                operand,
                bundle,
                scales,
                table,
                p.inverse_global_scale,
                p.gated_silu,
                0,
            )
        else:
            extension.shared(
                output,
                operand,
                bundle,
                scales,
                table,
                p.inverse_global_scale,
                p.gated_silu,
                mode,
            )

    def gate(mode):
        projection(operands[0], x, middle[mode], mode)

    def down(mode):
        projection(operands[1], middle[0], outputs[mode], mode)

    def mlp(mode):
        gate(mode)
        projection(operands[1], middle[mode], outputs[mode], mode)

    for amplitude in (0.01, 0.125, 1.0, 4.0):
        x.copy_(original * amplitude)
        for mode in (0, 3, 4):
            mlp(mode)
        reference = torch.empty_like(middle[0])
        p, codes, scales, _, _ = operands[0]
        ops.nvfp4_qpn2_gated_sm70_out(
            reference, x, codes, scales, p.inverse_global_scale, 8, 1
        )
        assert torch.equal(reference.view(torch.int16), middle[0].view(torch.int16))
        for mode in (3, 4):
            assert torch.equal(
                middle[0].view(torch.int16), middle[mode].view(torch.int16)
            )
            assert torch.equal(
                outputs[0].view(torch.int16), outputs[mode].view(torch.int16)
            )
    x.copy_(original * 0.125)
    gate(0)
    eviction = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    results = []
    for name, call in (("gate", gate), ("down", down), ("mlp", mlp)):
        for mode in (3, 4):
            result = graph_pair(
                partial(call, 0), partial(call, mode), eviction, args.iters
            )
            result.update(
                component=name,
                mode=mode,
                bitwise=True,
                shared_table_bytes=512 if mode == 3 else 1024,
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                generated_sha256=hashlib.sha256(text.encode()).hexdigest(),
                research_only=True,
            )
            results.append(result)
            print(
                json.dumps({k: v for k, v in result.items() if k != "samples_us"}),
                flush=True,
            )
    (args.out / "result.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
