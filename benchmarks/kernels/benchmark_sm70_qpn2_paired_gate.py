# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen gate/up sharing activation registers without shared-memory staging.

Each warp computes both projections with the original split and FP32 order.
Sequential and interleaved schedules use the existing bundled weight layout.
This is distinct from copying activations to shared memory or adding prefetch.
"""

import argparse
import hashlib
import json
import runpy
from functools import partial
from pathlib import Path

import torch
from benchmark_sm70_qpn2_effective_scale import generate as scale_source
from benchmark_sm70_qpn2_effective_scale import graph_pair
from torch.utils.cpp_extension import load

from vllm import _sm70_ops as ops


def generate(source):
    text = source.read_text()
    reader = text[
        text.index(
            "template <bool TurboMindLayout, bool BundledScales = false>"
        ) : text.index("#define VLLM_SM70_QPN2_MMA")
    ]
    body = scale_source(source)
    body = body[: body.index("PYBIND11_MODULE(")]
    return (
        body
        + "\nnamespace {\n"
        + reader
        + r"""
template <bool Interleave>
__global__ __launch_bounds__(256, 3) void paired_gate_kernel(
    const uint8_t* codes, const uint8_t* scales, const half* input,
    half* output, int hidden, int k, float global) {
  constexpr int Split = 8;
  __shared__ float partials[2][Split][256];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int qp = (lane >> 2) & 3;
  const int row = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int groups = k / 16;
  const int per_warp = groups / Split;
  Nvfp4PairReader<false, true> readers[2] = {
      Nvfp4PairReader<false, true>(codes, scales, blockIdx.x, groups, lane, global),
      Nvfp4PairReader<false, true>(codes, scales, blockIdx.x + hidden / 32,
                                 groups, lane, global)};
  float accum[2][8] = {};
#pragma unroll 4
  for (int group = warp * per_warp; group < (warp + 1) * per_warp; ++group) {
    const half* a = input + static_cast<size_t>(row) * k + group * 16;
    const uint4 a01 = *reinterpret_cast<const uint4*>(a);
    const uint4 a23 = *reinterpret_cast<const uint4*>(a + 8);
    const unsigned* a0 = reinterpret_cast<const unsigned*>(&a01);
    const unsigned* a1 = reinterpret_cast<const unsigned*>(&a23);
    if constexpr (Interleave) {
      half2 weights[2][8];
#pragma unroll
      for (int p = 0; p < 2; ++p) readers[p].load(group, weights[p]);
      const unsigned* b0 = reinterpret_cast<const unsigned*>(weights[0]);
      const unsigned* b1 = reinterpret_cast<const unsigned*>(weights[1]);
      VLLM_SM70_QPN2_MMA(accum[0], a0[0], a0[1], b0[0], b0[1]);
      VLLM_SM70_QPN2_MMA(accum[1], a0[0], a0[1], b1[0], b1[1]);
      VLLM_SM70_QPN2_MMA(accum[0], a0[2], a0[3], b0[2], b0[3]);
      VLLM_SM70_QPN2_MMA(accum[1], a0[2], a0[3], b1[2], b1[3]);
      VLLM_SM70_QPN2_MMA(accum[0], a1[0], a1[1], b0[4], b0[5]);
      VLLM_SM70_QPN2_MMA(accum[1], a1[0], a1[1], b1[4], b1[5]);
      VLLM_SM70_QPN2_MMA(accum[0], a1[2], a1[3], b0[6], b0[7]);
      VLLM_SM70_QPN2_MMA(accum[1], a1[2], a1[3], b1[6], b1[7]);
    } else {
#pragma unroll
      for (int p = 0; p < 2; ++p) {
        half2 weights[8];
        readers[p].load(group, weights);
        const unsigned* b = reinterpret_cast<const unsigned*>(weights);
        VLLM_SM70_QPN2_MMA(accum[p], a0[0], a0[1], b[0], b[1]);
        VLLM_SM70_QPN2_MMA(accum[p], a0[2], a0[3], b[2], b[3]);
        VLLM_SM70_QPN2_MMA(accum[p], a1[0], a1[1], b[4], b[5]);
        VLLM_SM70_QPN2_MMA(accum[p], a1[2], a1[3], b[6], b[7]);
      }
    }
  }
#pragma unroll
  for (int p = 0; p < 2; ++p) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int r = (i & 2) + ((lane & 16) ? 4 : 0) + (lane & 1);
      const int c = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
      partials[p][warp][r * 32 + qp * 8 + c] = accum[p][i];
    }
  }
  __syncthreads();
  const int e = threadIdx.x;
  float gate = 0.0f, up = 0.0f;
#pragma unroll
  for (int w = 0; w < Split; ++w) {
    gate += partials[0][w][e];
    up += partials[1][w][e];
  }
  const half gh = __float2half(gate);
  const half uh = __float2half(up);
  const float gf = __half2float(gh);
  const half silu = __float2half(gf / (1.0f + expf(-gf)));
  output[static_cast<size_t>(e / 32) * hidden + blockIdx.x * 32 + e % 32] =
      __hmul(silu, uh);
}
}
void launch_pair(torch::Tensor output, torch::Tensor input, torch::Tensor codes,
                 torch::Tensor scales, double global, int mode) {
  TORCH_CHECK(input.size(0) == 8 && input.size(1) == 5120 && output.size(1) == 4352,
              "qualified M8 TP4 gate/up shape only");
  auto stream = at::cuda::getCurrentCUDAStream();
  auto* x = reinterpret_cast<const half*>(input.data_ptr<at::Half>());
  auto* y = reinterpret_cast<half*>(output.data_ptr<at::Half>());
  if (mode == 1)
    paired_gate_kernel<false><<<136, 256, 0, stream>>>(
        codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),
        x, y, 4352, 5120, global);
  else if (mode == 2)
    paired_gate_kernel<true><<<136, 256, 0, stream>>>(
        codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),
        x, y, 4352, 5120, global);
  else TORCH_CHECK(false, "candidate mode must be1 or2");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launch", &launch);
  m.def("pair", &launch_pair);
}
"""
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
    generated = args.out / "paired_gate.cu"
    text = generate(source)
    if not generated.exists() or generated.read_text() != text:
        generated.write_text(text)
    extension = load(
        name="qpn2_paired_gate_screen",
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
    table = torch.empty(0, dtype=torch.float16)
    operands = []
    for p in helpers["_load_projection_shards"](args.model, 0, 0, 4):
        codes, scales = ops.nvfp4_qpn2_prepare_sm70(p.packed.cuda(), p.scales.cuda())
        bundle = torch.cat(
            (codes.view(-1, 256), scales.view(-1, 32)), dim=1
        ).contiguous()
        operands.append((p, codes, scales, bundle))
    gate, down = operands
    x = torch.randn(8, 5120, device="cuda", dtype=torch.float16)
    original = x.clone()
    middle = [torch.empty(8, 4352, device="cuda", dtype=x.dtype) for _ in range(3)]
    outputs = [torch.empty(8, 5120, device="cuda", dtype=x.dtype) for _ in range(3)]

    def call_gate(mode):
        p, _, scales, bundle = gate
        if mode:
            extension.pair(
                middle[mode], x, bundle, scales, p.inverse_global_scale, mode
            )
        else:
            extension.launch(
                middle[0], x, bundle, scales, table, p.inverse_global_scale, True, 0
            )

    def mlp(mode):
        call_gate(mode)
        p, _, scales, bundle = down
        extension.launch(
            outputs[mode],
            middle[mode],
            bundle,
            scales,
            table,
            p.inverse_global_scale,
            False,
            0,
        )

    for amplitude in (0.01, 0.1, 1.0, 4.0):
        x.copy_(original * amplitude)
        for mode in range(3):
            mlp(mode)
        ref = torch.empty_like(middle[0])
        p, codes, scales, _ = gate
        ops.nvfp4_qpn2_gated_sm70_out(
            ref, x, codes, scales, p.inverse_global_scale, 8, 1
        )
        assert torch.equal(ref.view(torch.int16), middle[0].view(torch.int16))
        assert all(
            torch.equal(y.view(torch.int16), middle[0].view(torch.int16))
            for y in middle[1:]
        )
        assert all(
            torch.equal(y.view(torch.int16), outputs[0].view(torch.int16))
            for y in outputs[1:]
        )
    x.copy_(original * 0.1)
    eviction = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    results = []
    for name, call in (("gate", call_gate), ("mlp", mlp)):
        for mode in (1, 2):
            result = graph_pair(
                partial(call, 0), partial(call, mode), eviction, args.iters
            )
            result.update(
                component=name,
                mode=mode,
                bitwise=True,
                weight_layout="bundled288",
                no_extra_weight_storage=True,
                optimistic_56_layer_saving_ms=result["saving_us"] * 56 / 1000,
            )
            results.append(result)
            print(
                json.dumps({k: v for k, v in result.items() if k != "samples_us"}),
                flush=True,
            )
    (args.out / "result.json").write_text(
        json.dumps(
            {
                "results": results,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "generated_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "research_only": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
