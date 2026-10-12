# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen two logical FP8 K slices per warp with unchanged accumulation order.

Keep native weight layout, original split/nacc and shared partials. Half as
many physical warps produce the same ordered logical partials in two phases.
This changes resident CTA geometry rather than padding or adding K splits.
"""

import argparse
import hashlib
import json
from functools import partial
from pathlib import Path

import torch
from benchmark_sm70_qpn2_effective_scale import extract_kernel, graph_pair
from safetensors import safe_open
from torch.utils.cpp_extension import load

from vllm import _sm70_ops as ops


def generate(source):
    text = source.read_text()
    helpers = text[
        text.index("__device__ __forceinline__ void fp8x8") : text.index(
            "__global__ void fp8_qpn8_prepack"
        )
    ]
    macro = text[
        text.index("#define VLLM_SM70_MMA_8N8K4") : text.index(
            "__global__ void fp8_qpn8_ba_split"
        )
    ]
    original = extract_kernel(text, "fp8_qpn8_sm70_kernel")
    candidate = original.replace("fp8_qpn8_sm70_kernel", "two_phase_qpn8")
    candidate = candidate.replace(
        "const int warp = threadIdx.x >> 5;",
        "const int physical_warp = threadIdx.x >> 5;",
    )
    begin = candidate.index("  const int quadpair =")
    end = candidate.index("  __syncthreads();", begin)
    candidate = (
        candidate[:begin]
        + "  for (int phase = 0; phase < 2; ++phase) {\n"
        + "    const int warp = physical_warp + phase * (SplitK / 2);\n"
        + candidate[begin:end]
        + "  }\n"
        + candidate[end:]
    )
    return (
        r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
namespace {
"""
        + helpers
        + macro
        + original
        + candidate
        + r"""
}
void launch(torch::Tensor output, torch::Tensor input, torch::Tensor codes,
            torch::Tensor scales, int split, bool phased) {
  TORCH_CHECK(input.size(0)==8 && input.is_contiguous() &&
              output.is_contiguous() && (split==12 || split==16));
  const int n=output.size(1), k=input.size(1);
  TORCH_CHECK(n%32==0 && (k/16)%split==0);
  auto stream=at::cuda::getCurrentCUDAStream();
  auto* c=codes.data_ptr<uint8_t>();
  auto* s=reinterpret_cast<const half*>(scales.data_ptr<at::Half>());
  auto* x=reinterpret_cast<const half*>(input.data_ptr<at::Half>());
  auto* y=reinterpret_cast<half*>(output.data_ptr<at::Half>());
  if (phased && split==16)
    two_phase_qpn8<16,2,true,false><<<n/32,256,0,stream>>>(
        c,s,x,y,nullptr,nullptr,nullptr,nullptr,nullptr,0,0,n,k,8,true);
  else if (phased && split==12)
    two_phase_qpn8<12,2,true,false><<<n/32,192,0,stream>>>(
        c,s,x,y,nullptr,nullptr,nullptr,nullptr,nullptr,0,0,n,k,8,true);
  else if (split==16)
    fp8_qpn8_sm70_kernel<16,2,true,false><<<n/32,512,0,stream>>>(
        c,s,x,y,nullptr,nullptr,nullptr,nullptr,nullptr,0,0,n,k,8,true);
  else
    fp8_qpn8_sm70_kernel<12,2,true,false><<<n/32,384,0,stream>>>(
        c,s,x,y,nullptr,nullptr,nullptr,nullptr,nullptr,0,0,n,k,8,true);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("launch",&launch); }
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
    src = args.source_root / "csrc/sm70_turbomind/ops/fp8_qpn8_sm70.cu"
    text = generate(src)
    generated = args.out / "two_phase.cu"
    if not generated.exists() or generated.read_text() != text:
        generated.write_text(text)
    extension = load(
        name="qpn8_two_phase_screen",
        sources=[str(generated)],
        extra_cuda_cflags=["-O3", "-lineinfo", "--ptxas-options=-v"],
        verbose=True,
    )
    if args.compile_only:
        return
    torch.set_grad_enabled(False)
    torch.manual_seed(123)
    with safe_open(args.model / "model.safetensors", framework="pt") as f:

        def read(name):
            return f.get_tensor("model.language_model.layers.0." + name)

        rows = list(range(512)) + list(range(2048, 2560)) + list(range(4096, 5632))
        qkvz = torch.cat(
            (
                read("linear_attn.in_proj_qkv.weight").view(torch.uint8)[rows],
                read("linear_attn.in_proj_z.weight").view(torch.uint8)[:1536],
            )
        )
        qscales = torch.cat(
            (
                read("linear_attn.in_proj_qkv.weight_scale")[rows],
                read("linear_attn.in_proj_z.weight_scale")[:1536],
            )
        )
        out = read("linear_attn.out_proj.weight").view(torch.uint8)[:, :1536]
        outscales = read("linear_attn.out_proj.weight_scale")
    eviction = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    results = []
    for name, w, s, split in (
        ("qkvz", qkvz, qscales, 16),
        ("gdn_out", out, outscales, 12),
    ):
        n, k = w.shape
        codes, scales = ops.fp8_qpn8_prepare_sm70(
            w.contiguous().view(torch.float8_e4m3fn).cuda(), s.float().cuda()
        )
        x = torch.randn(8, k, device="cuda", dtype=torch.float16)
        original = x.clone()
        outputs = [
            torch.empty(8, n, device="cuda", dtype=torch.float16) for _ in range(2)
        ]

        def call(arm, outputs=outputs, x=x, codes=codes, scales=scales, split=split):
            extension.launch(outputs[arm], x, codes, scales, split, bool(arm))

        for amplitude in (0.01, 0.125, 1.0, 4.0):
            x.copy_(original * amplitude)
            call(0)
            call(1)
            reference = torch.empty_like(outputs[0])
            ops.fp8_qpn8_gemm_sm70_out(
                reference, x, codes, scales, split, 2, True, False
            )
            assert torch.equal(
                reference.view(torch.int16), outputs[0].view(torch.int16)
            )
            assert torch.equal(
                outputs[0].view(torch.int16), outputs[1].view(torch.int16)
            )
        x.copy_(original * 0.125)
        result = graph_pair(partial(call, 0), partial(call, 1), eviction, args.iters)
        result.update(
            component=name,
            bitwise=True,
            split=split,
            nacc=2,
            source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
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
