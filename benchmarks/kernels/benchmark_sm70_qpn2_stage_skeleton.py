# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Separate QPN2 native M8 weight loads, decoding and HMMA.

Generate research-only read/decode skeletons from the production CUDA source.
All three stages retain its CTA grid, lane addresses, split and loop order.
Removal of computation changes register pressure: report NCU occupancy too.
Do not interpret logical bytes/time as measured DRAM bandwidth.
"""

import argparse
import hashlib
import importlib.util
import json
import runpy
import statistics
from functools import partial
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from vllm import _sm70_ops as ops


def source_for_screen(source: Path, decoder: str = "production") -> str:
    original = source.read_text()
    # Keep the exact production dequantization functions and MMA macro.
    helpers = original[
        original.index("__device__ __forceinline__ half2 fp8e4m3") : original.index(
            "// Batch kernels share"
        )
    ]
    if decoder == "prmt":
        begin = helpers.index("__device__ __forceinline__ void dequant_e2m1x8")
        helpers = (
            helpers[:begin]
            + """
__device__ __forceinline__ void dequant_e2m1x8(unsigned packed, half2 scale,
                                             half2 output[4]) {
  // Exact high bytes of half(0, .5, 1, 1.5, 2, 3, 4, 6). Low bytes are zero.
  constexpr unsigned lo = 0x3e3c3800u;
  constexpr unsigned hi = 0x46444240u;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const unsigned selector = ((packed >> (4*i)) & 7u) << 4 |
                              ((packed >> (16+4*i)) & 7u) << 12;
    const unsigned sign = (packed << (12-4*i)) & 0x80008000u;
    unsigned bits = __byte_perm(lo, hi, selector) | sign;
    output[i] = __hmul2(*reinterpret_cast<half2*>(&bits), scale);
  }
}
"""
        )
    shared_decode = ""
    if decoder == "shared":
        helpers += """
__device__ __forceinline__ void shared_e2m1x16(uint2 packed, half2 scale,
                                             volatile float* lut, int lane,
                                             half2 output[8]) {
  // Each lane owns its eight entries. Entry-major storage selects a distinct
  // shared-memory bank for each lane; no inter-lane synchronization is needed.
  const unsigned codes[4] = {0x38000000u, 0x3e003c00u,
                             0x42004000u, 0x46004400u};
#pragma unroll
  for(int i=0; i<4; ++i) {
    const half2 v=__hmul2(*reinterpret_cast<const half2*>(&codes[i]), scale);
    const unsigned b=*reinterpret_cast<const unsigned*>(&v);
    lut[(i*2)*32+lane]=__uint_as_float(b&0xffffu);
    lut[(i*2+1)*32+lane]=__uint_as_float(b>>16);
  }
  const unsigned words[2]={packed.x,packed.y};
#pragma unroll
  for(int w=0;w<2;++w) {
#pragma unroll
    for(int i=0;i<4;++i) {
      const unsigned word=words[w];
      const unsigned lo=__float_as_uint(lut[((word>>(4*i))&7u)*32+lane]);
      const unsigned hi=__float_as_uint(lut[((word>>(16+4*i))&7u)*32+lane]);
      const unsigned sign=(word<<(12-4*i))&0x80008000u;
      const unsigned bits=(lo|(hi<<16))^sign;
      output[w*4+i]=*reinterpret_cast<const half2*>(&bits);
    }
  }
}
"""
        shared_decode = """
    static_assert(RowTiles==1, "The research LUT reuses one row tile's scratch");
    half2 weights[8];
    volatile float* lut = reinterpret_cast<float*>(partials) +
                           (threadIdx.x>>5)*256;
    shared_e2m1x16(packed,scale,lut,lane,weights);
"""
    macro = original[
        original.index("#define VLLM_SM70_QPN2_MMA") : original.index(
            "// Four row tiles reuse"
        )
    ]
    index = original.index("__global__ void nvfp4_qpn2_sm70_kernel")
    begin = original.rfind("template <", 0, index)
    # Identify the launch template directly rather than depending on lines.
    end = original.rfind("template <", begin, original.index("\nvoid launch_qpn2"))
    complete = original[begin:end]
    if decoder == "shared":
        old_decode = """    half2 weights[8];
    dequant_e2m1x8(packed.x, scale, weights);
    dequant_e2m1x8(packed.y, scale, weights + 4);
"""
        assert complete.count(old_decode) == 2
        # Every warp owns its original 256-float partial tile during the loop,
        # then overwrites only that same tile before the unchanged CTA barrier.
        complete = complete.replace(old_decode, shared_decode)
    gated_template = (
        "template <int SplitK, int NAcc, int RowTiles = 1, "
        "bool TurboMindLayout = false>"
    )
    kernel_parts = complete.split(gated_template)
    assert len(kernel_parts) == 2
    down, gated_body = kernel_parts
    gated = gated_template + gated_body
    variants = []
    for body, name in (
        (down, "nvfp4_qpn2_sm70_kernel"),
        (gated, "nvfp4_qpn2_gated_sm70_kernel"),
    ):
        prefix = body[: body.index("  float accum[")]
        for stage in ("a_read", "b_decode"):
            variant = prefix.replace(name, name + "_" + stage)
            variant = variant.replace(
                "float global_scale) {", "float global_scale, unsigned* sink) {"
            )
            variant += """
  unsigned checks[4] = {};
#pragma unroll 4
  for (int group = group_begin; group < group_begin + groups_per_warp; ++group) {
    const uint2 packed = reader.load(group);
    const uint8_t raw_scale = __ldg(scale_ptr + static_cast<size_t>(group) * 32);
"""
            if stage == "a_read":
                # Keep observable results so ptxas cannot discard the reads.
                variant += """
    checks[0] ^= packed.x;
    checks[1] ^= packed.y;
    checks[2] ^= raw_scale;
"""
            else:
                decode_body = """
    const half2 scale = nvfp4_effective_scale(raw_scale, global_scale);
    half2 weights[8];
    dequant_e2m1x8(packed.x, scale, weights);
    dequant_e2m1x8(packed.y, scale, weights + 4);
    const unsigned* b = reinterpret_cast<const unsigned*>(weights);
#pragma unroll
    for (int j = 0; j < 4; ++j) checks[j] ^= b[j] ^ b[j + 4];
"""
                if decoder == "shared":
                    decode_body = decode_body.replace(old_decode, shared_decode)
                variant += decode_body
            variants.append(
                variant
                + """
  }
  sink[blockIdx.x * blockDim.x + threadIdx.x] =
      checks[0] ^ checks[1] ^ checks[2] ^ checks[3];
}
"""
            )
    calls = []
    for stage_index, suffix in enumerate(("_a_read", "_b_decode", "")):
        sink_arg = (
            ", reinterpret_cast<unsigned*>(sink.data_ptr<int32_t>())" if suffix else ""
        )
        calls.append(f"""
  if (stage == {stage_index}) {{
    if (gated) {{
      nvfp4_qpn2_gated_sm70_kernel{suffix}<8, 1><<<dim3(width/32), 512, 0, stream>>>(
          c, s, x, y, width, k, 8, scale{sink_arg});
    }} else {{
      nvfp4_qpn2_sm70_kernel{suffix}<16, 2><<<dim3(width/32), 512, 0, stream>>>(
          c, s, x, y, width, k, 8, scale{sink_arg});
    }}
  }}
""")
    return (
        """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include "nvfp4_qpn2_layout.cuh"
namespace {
constexpr int kQpn2RowsPerCta = 8;
"""
        + helpers
        + macro
        + complete
        + "\n".join(variants)
        + """
}
void launch_stage(torch::Tensor output, torch::Tensor input,
                  torch::Tensor codes, torch::Tensor scales,
                  double global_scale, bool gated, int stage, torch::Tensor sink) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto* c = codes.data_ptr<uint8_t>();
  const auto* s = scales.data_ptr<uint8_t>();
  const auto* x = reinterpret_cast<const half*>(input.data_ptr<at::Half>());
  auto* y = reinterpret_cast<half*>(output.data_ptr<at::Half>());
  const int width = output.size(1), k = input.size(1);
  const float scale = global_scale;
"""
        + "\n".join(calls)
        + """
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("launch", &launch_stage); }
"""
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--source-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--decoder", choices=("production", "prmt", "shared"), default="production"
    )
    parser.add_argument(
        "--profile-launches-only",
        action="store_true",
        help="Collect hardware counters from direct launches without event graphs",
    )
    parser.add_argument(
        "--extension",
        type=Path,
        help="Precompiled research extension with matching Torch ABI",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cuda_source = args.source_root / "csrc/sm70_turbomind/ops/nvfp4_qpn2_sm70.cu"
    generated = args.out / "qpn2-stage-skeleton.cu"
    generated.write_text(source_for_screen(cuda_source, args.decoder))
    extension_name = "round12_qpn2_stage_skeleton"
    if args.decoder != "production":
        # Pybind caches modules by exported name. Give candidate DSOs distinct
        # names so a paired process cannot accidentally reuse the control.
        extension_name += "_" + args.decoder
    if args.extension is None:
        extension = load(
            name=extension_name,
            sources=[str(generated)],
            extra_include_paths=[str(cuda_source.parent)],
            extra_cuda_cflags=["-O3", "-lineinfo"],
            verbose=True,
        )
    else:
        spec = importlib.util.spec_from_file_location(extension_name, args.extension)
        assert spec is not None and spec.loader is not None
        extension = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(extension)
    if args.compile_only:
        print("Compiled research stages; no GPU execution.", flush=True)
        return
    torch.set_grad_enabled(False)
    torch.manual_seed(123)
    assert torch.cuda.get_device_capability() == (7, 0)
    helpers = runpy.run_path(
        str(args.source_root / "benchmarks/kernels/benchmark_sm70_nvfp4_qpn2.py")
    )
    projections = helpers["_load_projection_shards"](args.model, 0, 0, 4)
    eviction = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    records, graphs = [], []
    for projection in projections:
        codes, scales = ops.nvfp4_qpn2_prepare_sm70(
            projection.packed.cuda(), projection.scales.cuda()
        )
        n, k = projection.packed.shape[0], projection.packed.shape[1] * 2
        gated = projection.gated_silu
        x = torch.randn(8, k, device="cuda", dtype=torch.float16) * 0.1
        y = torch.empty(8, n // 2 if gated else n, device="cuda", dtype=x.dtype)
        reference = torch.empty_like(y)
        sink = torch.empty(y.shape[1] // 32, 512, device="cuda", dtype=torch.int32)
        original = (
            ops.nvfp4_qpn2_gated_sm70_out if gated else ops.nvfp4_qpn2_gemm_sm70_out
        )
        original(
            reference,
            x,
            codes,
            scales,
            projection.inverse_global_scale,
            8 if gated else 16,
            1 if gated else 2,
        )
        extension.launch(
            y, x, codes, scales, projection.inverse_global_scale, gated, 2, sink
        )
        torch.testing.assert_close(y, reference, rtol=0, atol=0)
        for stage in range(3):
            run = partial(
                extension.launch,
                y,
                x,
                codes,
                scales,
                projection.inverse_global_scale,
                gated,
                stage,
                sink,
            )

            run()
            torch.cuda.synchronize()
            if args.profile_launches_only:
                # Some profiler versions fail while capturing external-event
                # graphs. Keep performance timing in the normal cold graph run;
                # these direct launches provide counters for the same kernels.
                graphs.append((None, run))
                continue
            graph = torch.cuda.CUDAGraph()
            start = torch.cuda.Event(enable_timing=True, external=True)
            end = torch.cuda.Event(enable_timing=True, external=True)
            with torch.cuda.graph(graph):
                eviction.fill_(1)
                start.record()
                run()
                end.record()
            for _ in range(10):
                graph.replay()
            samples = []
            for _ in range(args.iters):
                graph.replay()
                end.synchronize()
                samples.append(start.elapsed_time(end) * 1000)
            records.append(
                dict(
                    projection="gate_up" if gated else "down",
                    stage=("read", "read_decode", "full")[stage],
                    cold_l2_graph_us=statistics.fmean(samples),
                    logical_weight_bytes=codes.numel() + scales.numel(),
                    grid=y.shape[1] // 32,
                    threads=512,
                    full_output_bitwise=True,
                )
            )
            # Retain tensors referenced by the graph's raw pointers.
            graphs.append((graph, run))
    if args.profile or args.profile_launches_only:
        torch.cuda.cudart().cudaProfilerStart()
        for graph, run in graphs:
            if graph is None:
                run()
            else:
                graph.replay()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    record = dict(
        cases=records,
        source_sha256=hashlib.sha256(cuda_source.read_bytes()).hexdigest(),
        source_root=str(args.source_root),
        research_only=True,
        measured_dram_bandwidth="Use NCU counters, not bytes/time",
        profiler_launches_only=args.profile_launches_only,
        decoder=args.decoder,
    )
    (args.out / "result.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
