# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen direct FP32 construction of E4M3 scale bits, without table loads.

Keep the global FP32 multiply, FP16 rounding and original FP4 decoder. This
isolates conversion instructions from the rejected scale-table memory paths.
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
    fast = r"""
__device__ __forceinline__ half2 scale_from_bits(uint8_t value, float global) {
  const unsigned magnitude = value & 127u;
  const unsigned sign = (static_cast<unsigned>(value) & 128u) << 24;
  // Normal E4M3 values have FP32 exponent bias 127-7=120. The original
  // decoder maps all bit patterns arithmetically, including signed zero.
  const float normal = __uint_as_float(sign | (magnitude << 20) | 0u);
  const float raw = magnitude >= 8
      ? __uint_as_float(sign | ((magnitude << 20) + 0x3c000000u))
      : copysignf(static_cast<float>(magnitude) * 0.001953125f, normal);
  const half scaled = __float2half_rn(__fmul_rn(raw, global));
  return __halves2half2(scaled, scaled);
}
"""
    text = text.replace(
        "#define VLLM_SM70_QPN2_MMA", fast + "\n#define VLLM_SM70_QPN2_MMA"
    )
    for gated in (False, True):
        name = "nvfp4_qpn2_" + ("gated_" if gated else "") + "sm70_kernel"
        old = extract_kernel(text, name + "_scale1")
        new = extract_kernel(text, name + "_scale0")
        new = new.replace(name + "_scale0", name + "_scale1")
        new = new.replace("nvfp4_effective_scale(", "scale_from_bits(")
        text = text.replace(old, new)
    text = text.replace(
        ", reinterpret_cast<const half*>(table.data_ptr<at::Half>())", ""
    )
    extra = r"""
__global__ void scale_check(half* original, half* candidate, float global) {
  const int i=threadIdx.x;
  original[i]=__low2half(nvfp4_effective_scale(i,global));
  candidate[i]=__low2half(scale_from_bits(i,global));
}
void check(torch::Tensor a,torch::Tensor b,double global) {
  scale_check<<<1,256,0,at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<half*>(a.data_ptr<at::Half>()),
      reinterpret_cast<half*>(b.data_ptr<at::Half>()),global);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""
    text = text.replace("PYBIND11_MODULE(", extra + "\nPYBIND11_MODULE(")
    return text.replace('m.def("prepare", &prepare);', 'm.def("check", &check);')


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
    generated = args.out / "scale_bits.cu"
    text = source_text(source)
    if not generated.exists() or generated.read_text() != text:
        generated.write_text(text)
    extension = load(
        name="qpn2_scale_bits_screen",
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
        operands.append((p, codes, scales, bundle))
        a = torch.empty(256, device="cuda", dtype=torch.float16)
        b = torch.empty_like(a)
        for factor in (p.inverse_global_scale, 0.001, 0.123, 1.0, 10.0):
            extension.check(a, b, factor)
            assert torch.equal(a.view(torch.int16), b.view(torch.int16))
    x = torch.randn(8, 5120, device="cuda", dtype=torch.float16)
    original = x.clone()
    middle = [x.new_empty(8, 4352) for _ in range(2)]
    output = [torch.empty_like(x) for _ in range(2)]

    def projection(index, mode):
        p, _, scales, bundle = operands[index]
        extension.launch(
            middle[mode] if index == 0 else output[mode],
            x if index == 0 else middle[mode],
            bundle,
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
        p, codes, scales, _ = operands[0]
        ref = torch.empty_like(middle[0])
        ops.nvfp4_qpn2_gated_sm70_out(
            ref, x, codes, scales, p.inverse_global_scale, 8, 1
        )
        assert torch.equal(ref.view(torch.int16), middle[0].view(torch.int16))
    x.copy_(original * 0.125)
    eviction = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    results = []
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
        "all_256_scale_codes": True,
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
