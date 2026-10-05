# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QPN8 read/decode/full skeletons with exact production M8 lane addresses.

Checksum instrumentation and resource differences must be considered when
interpreting stages. CUDA graph timing and NCU direct-launch counters are
separate experiments. Nothing in this benchmark changes production dispatch.
"""

import argparse
import hashlib
import json
import statistics
from functools import partial
from pathlib import Path

import torch
from safetensors import safe_open
from torch.utils.cpp_extension import load

from vllm import _sm70_ops as ops


def generate(source: Path) -> str:
    original = source.read_text()
    helpers = original[
        original.index("__device__ __forceinline__ void fp8x8") : original.index(
            "__global__ void fp8_qpn8_prepack"
        )
    ]
    macro = original[
        original.index("#define VLLM_SM70_MMA") : original.index(
            "__global__ void fp8_qpn8_ba_split_copy"
        )
    ]
    begin = original.rfind(
        "template <", 0, original.index("__global__ void fp8_qpn8_sm70_kernel")
    )
    end = original.rfind(
        "template <", begin, original.index("void launch_fp8_qpn8_sm70")
    )
    full = original[begin:end]
    variants = []
    for stage in ("a_read", "b_decode"):
        prefix = full[: full.index("  float accum[")]
        prefix = prefix.replace("fp8_qpn8_sm70_kernel", "fp8_qpn8_sm70_kernel_" + stage)
        prefix = prefix.replace(
            "bool channel_scales) {", "bool channel_scales, unsigned* sink) {"
        )
        prefix += """
  unsigned check[4] = {};
  const half scale = __ldg(group_scales + tile*32 + qpn8_col_from_lane(lane));
  const half2 scale2 = __halves2half2(scale, scale);
#pragma unroll 4
  for (int group = group_begin; group < group_begin + groups_per_warp; ++group) {
    const uint4 packed = __ldcs(code_ptr + static_cast<size_t>(group)*32);
"""
        if stage == "a_read":
            prefix += """
    check[0] ^= packed.x; check[1] ^= packed.y;
    check[2] ^= packed.z; check[3] ^= packed.w;
"""
        else:
            prefix += """
    half2 weights[8];
    fp8x8_to_half2x4_fast(make_uint2(packed.x, packed.y), weights);
    fp8x8_to_half2x4_fast(make_uint2(packed.z, packed.w), weights+4);
#pragma unroll
    for (int j=0; j<8; ++j) weights[j] = __hmul2(weights[j], scale2);
    const unsigned* bits = reinterpret_cast<const unsigned*>(weights);
#pragma unroll
    for (int j=0; j<4; ++j) check[j] ^= bits[j] ^ bits[j+4];
"""
        prefix += """
  }
  sink[blockIdx.x*blockDim.x+threadIdx.x] =
      check[0]^check[1]^check[2]^check[3]^__half_as_ushort(scale);
}
"""
        variants.append(prefix)
    calls = []
    for stage, suffix in enumerate(("_a_read", "_b_decode", "")):
        sink = (
            ", reinterpret_cast<unsigned*>(scratch.data_ptr<int32_t>())"
            if suffix
            else ""
        )
        calls.append(f"""
  if (stage == {stage}) {{
    if (nacc == 2) {{
      fp8_qpn8_sm70_kernel{suffix}<16,2,true,false><<<n/32,512,0,stream>>>(
        c,s,x,y,nullptr,nullptr,nullptr,nullptr,nullptr,0,n,n,k,8,true{sink});
    }} else {{
      fp8_qpn8_sm70_kernel{suffix}<16,1,true,false><<<n/32,512,0,stream>>>(
        c,s,x,y,nullptr,nullptr,nullptr,nullptr,nullptr,0,n,n,k,8,true{sink});
    }}
  }}
""")
    return (
        """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
namespace {
"""
        + helpers
        + macro
        + full
        + "\n".join(variants)
        + """
void launch(torch::Tensor out, torch::Tensor input, torch::Tensor codes,
            torch::Tensor scales, int nacc, int stage, torch::Tensor scratch) {
  const int n=out.size(1), k=input.size(1);
  const auto stream=at::cuda::getCurrentCUDAStream();
  const auto* c=codes.data_ptr<uint8_t>();
  const auto* s=reinterpret_cast<const half*>(scales.data_ptr<at::Half>());
  const auto* x=reinterpret_cast<const half*>(input.data_ptr<at::Half>());
  auto* y=reinterpret_cast<half*>(out.data_ptr<at::Half>());
"""
        + "\n".join(calls)
        + """
}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {m.def("launch", &launch);}
"""
    )


def load_weights(model):
    with safe_open(model / "model.safetensors", framework="pt") as f:
        base = "model.language_model.layers.0.linear_attn."
        qkv = f.get_tensor(base + "in_proj_qkv.weight")
        z = f.get_tensor(base + "in_proj_z.weight")
        qkv_s = f.get_tensor(base + "in_proj_qkv.weight_scale")
        z_s = f.get_tensor(base + "in_proj_z.weight_scale")
        indices = torch.cat(
            [torch.arange(512), torch.arange(2048, 2560), torch.arange(4096, 5632)]
        )
        qkvz = torch.cat([qkv[indices], z[:1536]])
        scales = torch.cat([qkv_s[indices], z_s[:1536]])
        out = f.get_tensor(base + "out_proj.weight")[:, :1536].contiguous()
        out_s = f.get_tensor(base + "out_proj.weight_scale")
    return [("qkvz", qkvz, scales, 2), ("gdn_out", out, out_s, 1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--source-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--profile-launches-only", action="store_true")
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.source_root / "csrc/sm70_turbomind/ops/fp8_qpn8_sm70.cu"
    generated = args.out / "qpn8-stages.cu"
    generated.write_text(generate(source))
    extension = load(
        name="round12_qpn8_stages",
        sources=[str(generated)],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        verbose=True,
    )
    if args.compile_only:
        return
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    torch.manual_seed(123)
    assert torch.cuda.get_device_capability() == (7, 0)
    eviction = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    rows, runs = [], []
    for name, weight, scales, nacc in load_weights(args.model):
        codes, packed_scales = ops.fp8_qpn8_prepare_sm70(
            weight.cuda(), scales.float().cuda()
        )
        n, k = weight.shape
        x = torch.randn(8, k, device="cuda", dtype=torch.float16) * 0.1
        out = torch.empty(8, n, device="cuda", dtype=torch.float16)
        truth = torch.empty_like(out)
        scratch = torch.empty(n // 32, 512, device="cuda", dtype=torch.int32)
        ops.fp8_qpn8_gemm_sm70_out(
            truth, x, codes, packed_scales, 16, nacc, True, False
        )
        extension.launch(out, x, codes, packed_scales, nacc, 2, scratch)
        assert torch.equal(out, truth)
        for stage in range(3):
            run = partial(
                extension.launch, out, x, codes, packed_scales, nacc, stage, scratch
            )
            run()
            torch.cuda.synchronize()
            if args.profile_launches_only:
                runs.append(run)
                continue
            graph = torch.cuda.CUDAGraph()
            begin = torch.cuda.Event(enable_timing=True, external=True)
            end = torch.cuda.Event(enable_timing=True, external=True)
            with torch.cuda.graph(graph):
                eviction.fill_(1)
                begin.record()
                run()
                end.record()
            for _ in range(10):
                graph.replay()
            timings = []
            for _ in range(args.iters):
                graph.replay()
                end.synchronize()
                timings.append(begin.elapsed_time(end) * 1000)
            row = {
                "projection": name,
                "stage": stage,
                "graph_us": statistics.median(timings),
                "full_control_bitwise": True,
                "grid": n // 32,
                "threads": 512,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    if args.profile_launches_only:
        torch.cuda.cudart().cudaProfilerStart()
        for run in runs:
            run()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    report = {
        "cases": rows,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "research_only": True,
        "profile_launches_only": args.profile_launches_only,
    }
    (args.out / "result.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
