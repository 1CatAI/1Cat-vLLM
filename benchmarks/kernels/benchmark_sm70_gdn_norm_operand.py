# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Research-only GDN norm inside the out-projection's head-local warp.

The accepted SplitK12 out projection assigns exactly one 128-column head to
each warp. Normalize that head in registers and exchange the existing MMA
operands within each four-lane token group. Delta geometry, weight addresses,
and the projection reduction remain unchanged. The norm reduction order does
change; this screen does not establish numerical or model admission.
"""

import argparse
import ctypes
import hashlib
import json
import statistics
from pathlib import Path

import torch
from safetensors import safe_open
from torch.utils.cpp_extension import load

from vllm import _sm70_ops as ops
from vllm.model_executor.layers.fla.ops.layernorm_guard import rmsnorm_fn


def generate(source, static_operands=False):
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
    kernel = original[begin:end].replace(
        "fp8_qpn8_sm70_kernel", "gdn_norm_operand_kernel"
    )
    kernel = kernel.replace(
        "bool channel_scales) {",
        "bool channel_scales, const half* norm_weight, const half* gate) {",
        1,
    )
    kernel = kernel.replace(
        "  const int groups_per_warp = groups_k16 / SplitK;",
        "  constexpr int groups_per_warp = 8;",
        1,
    )
    marker = "  float accum[RowTiles][NAcc][8];"
    assert kernel.count(marker) == 1
    kernel = kernel.replace(
        marker,
        """
  // Eight token rows, with four lanes holding disjoint 32-column chunks.
  const int head = warp;
  const int chunk = quadpair * 32;
  const int base = row * 1536 + head * 128 + chunk;
  float values[32];
  float squares[32];
#pragma unroll
  for (int j = 0; j < 32; ++j) {
    values[j] = __half2float(input[base + j]);
    squares[j] = values[j] * values[j];
  }
#pragma unroll
  for (int distance = 16; distance > 0; distance >>= 1) {
#pragma unroll
    for (int j = 0; j < distance; ++j) {
      squares[j] += squares[j + distance];
    }
  }
  float sum = squares[0];
  sum += __shfl_xor_sync(0xffffffff, sum, 4);
  sum += __shfl_xor_sync(0xffffffff, sum, 8);
  const float inv = rsqrtf(sum * (1.0f / 128.0f) + 1e-6f);
  unsigned normalized[16];
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const int c0 = chunk + 2 * j;
    const float z0 = __half2float(gate[base + 2 * j]);
    const float z1 = __half2float(gate[base + 2 * j + 1]);
    const float y0 = (values[2*j] * inv) * __half2float(norm_weight[c0]);
    const float y1 = (values[2*j+1] * inv) * __half2float(norm_weight[c0+1]);
    const half2 pair = __floats2half2_rn(
        y0 * (z0 / (1.0f + __expf(-z0))),
        y1 * (z1 / (1.0f + __expf(-z1))));
    normalized[j] = *reinterpret_cast<const unsigned*>(&pair);
  }
"""
        + marker,
        1,
    )
    old = """      if (input_row_idx < m) {
        const half* input_row = input + static_cast<size_t>(input_row_idx) * k;
        input01 = *reinterpret_cast<const uint4*>(input_row + group * 16);
        input23 = *reinterpret_cast<const uint4*>(input_row + group * 16 + 8);
      }"""
    assert kernel.count(old) == 1
    kernel = kernel.replace(
        old,
        """
      const int local_group = group - group_begin;
      const int source_lane = (lane & 19) | ((local_group >> 1) << 2);
      const int word = (local_group & 1) * 8;
      input01 = make_uint4(
          __shfl_sync(0xffffffff, normalized[word], source_lane),
          __shfl_sync(0xffffffff, normalized[word+1], source_lane),
          __shfl_sync(0xffffffff, normalized[word+2], source_lane),
          __shfl_sync(0xffffffff, normalized[word+3], source_lane));
      input23 = make_uint4(
          __shfl_sync(0xffffffff, normalized[word+4], source_lane),
          __shfl_sync(0xffffffff, normalized[word+5], source_lane),
          __shfl_sync(0xffffffff, normalized[word+6], source_lane),
          __shfl_sync(0xffffffff, normalized[word+7], source_lane));
""",
        1,
    )
    kernel = kernel.replace("#pragma unroll 4\n", "#pragma unroll 8\n", 1)
    if static_operands:
        reduction = """#pragma unroll
  for (int distance = 16; distance > 0; distance >>= 1) {
#pragma unroll
    for (int j = 0; j < distance; ++j) {
      squares[j] += squares[j + distance];
    }
  }"""
        assert kernel.count(reduction) == 1
        kernel = kernel.replace(
            reduction,
            "\n".join(
                f"#pragma unroll\n  for (int j = 0; j < {distance}; ++j) "
                f"{{ squares[j] += squares[j + {distance}]; }}"
                for distance in (16, 8, 4, 2, 1)
            ),
            1,
        )
        old_loop = (
            "  for (int group = group_begin; group < group_begin + groups_per_warp;\n"
            "       ++group) {"
        )
        assert kernel.count(old_loop) == 1
        kernel = kernel.replace(
            old_loop,
            "  for (int local_group = 0; local_group < 8; ++local_group) {\n"
            "    const int group = group_begin + local_group;",
            1,
        ).replace("      const int local_group = group - group_begin;\n", "", 1)
    return (
        """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <cuda_fp16.h>
namespace {
"""
        + helpers
        + macro
        + kernel
        + """
void launch(torch::Tensor output, torch::Tensor input, torch::Tensor gate,
            torch::Tensor weight, torch::Tensor codes, torch::Tensor scales) {
  TORCH_CHECK(input.sizes() == torch::IntArrayRef({8, 1536}));
  TORCH_CHECK(gate.sizes() == input.sizes() && weight.numel() == 128);
  TORCH_CHECK(output.sizes() == torch::IntArrayRef({8, 5120}));
  const auto stream = at::cuda::getCurrentCUDAStream();
  gdn_norm_operand_kernel<12, 2, true, false><<<160, 384, 0, stream>>>(
      codes.data_ptr<uint8_t>(),
      reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      nullptr, nullptr, nullptr, nullptr, nullptr, 0, 5120, 5120, 1536, 8, true,
      reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(gate.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("launch", &launch); }
"""
    )


def census_pair(functions):
    """Read actual graph nodes separately from the event-time experiment."""
    driver = ctypes.CDLL("libcuda.so.1")
    records = []
    for function in functions:
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            function()
        handle = ctypes.c_void_p(graph.raw_cuda_graph())
        count = ctypes.c_size_t()
        assert driver.cuGraphGetNodes(handle, None, ctypes.byref(count)) == 0
        nodes = (ctypes.c_void_p * count.value)()
        assert driver.cuGraphGetNodes(handle, nodes, ctypes.byref(count)) == 0
        kinds = {}
        for node in nodes:
            kind = ctypes.c_int()
            assert (
                driver.cuGraphNodeGetType(ctypes.c_void_p(node), ctypes.byref(kind))
                == 0
            )
            kinds[str(kind.value)] = kinds.get(str(kind.value), 0) + 1
        assert "4" not in kinds and "13" not in kinds, "Do not omit nested bodies"
        records.append({"direct_nodes": count.value, "node_types": kinds})
    kernels = [r["node_types"].get("0", 0) for r in records]
    assert kernels == [2, 1], f"Expected native route was not captured: {records}"
    return {"kernel_nodes": kernels, "graphs": records, "census_only": True}


def time_pair(functions, iterations):
    eviction = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    graphs, samples = [], [[], []]
    for function in functions:
        graph = torch.cuda.CUDAGraph()
        start = torch.cuda.Event(enable_timing=True, external=True)
        end = torch.cuda.Event(enable_timing=True, external=True)
        with torch.cuda.graph(graph):
            eviction.fill_(1)
            start.record()
            function()
            end.record()
        graphs.append((graph, start, end))
    for iteration in range(iterations + 10):
        for index in (iteration % 2, 1 - iteration % 2):
            graph, start, end = graphs[index]
            graph.replay()
            end.synchronize()
            if iteration >= 10:
                samples[index].append(start.elapsed_time(end) * 1000)
    return {
        "baseline_us": statistics.fmean(samples[0]),
        "candidate_us": statistics.fmean(samples[1]),
        "samples_us": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--static-operands", action="store_true")
    parser.add_argument("--census-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.manual_seed(123)
    assert torch.cuda.get_device_capability() == (7, 0)
    original = args.source_root / "csrc/sm70_turbomind/ops/fp8_qpn8_sm70.cu"
    generated = args.out / "gdn_norm_operand.cu"
    text = generate(original, args.static_operands)
    if not generated.exists() or generated.read_text() != text:
        generated.write_text(text)
    module = load(
        name="round12_gdn_norm_operand" + ("_static" if args.static_operands else ""),
        sources=[str(generated)],
        extra_cuda_cflags=["-O3", "-lineinfo", "--ptxas-options=-v"],
        verbose=True,
    )
    with safe_open(args.model / "model.safetensors", framework="pt") as f:
        base = "model.language_model.layers.0.linear_attn."
        codes, scales = ops.fp8_qpn8_prepare_sm70(
            f.get_tensor(base + "out_proj.weight")[:, :1536].contiguous().cuda(),
            f.get_tensor(base + "out_proj.weight_scale").float().cuda(),
        )
        weight = f.get_tensor(base + "norm.weight").half().cuda()
    core = torch.randn(8, 1536, device="cuda", dtype=torch.float16)
    gate = torch.randn_like(core)
    output = core.new_empty(8, 5120)

    def baseline():
        normalized = rmsnorm_fn(
            core.view(96, 128),
            weight,
            None,
            z=gate.view(96, 128),
            eps=1e-6,
            norm_before_gate=True,
        ).view(8, 1536)
        ops.fp8_qpn8_gemm_sm70_out(
            output, normalized, codes, scales, 12, 2, True, False
        )

    def candidate():
        module.launch(output, core, gate, weight, codes, scales)

    errors = []
    for amplitude in (0.01, 0.125, 1.0, 4.0):
        core.normal_().mul_(amplitude)
        baseline()
        expected = output.clone()
        candidate()
        error = (output.float() - expected.float()).abs()
        torch.testing.assert_close(output, expected, rtol=0.025, atol=0.04)
        errors.append({"amplitude": amplitude, "max_error": error.max().item()})
    result = (
        census_pair((baseline, candidate))
        if args.census_only
        else time_pair((baseline, candidate), args.iters)
    )
    result.update(
        errors=errors,
        source_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
        generated_sha256=hashlib.sha256(generated.read_bytes()).hexdigest(),
        production_dispatch_changed=False,
        model_quality_admitted=False,
        cold_l2_bytes=32 * 1024 * 1024,
        expected_compute_kernel_nodes=[2, 1],
        static_operands=args.static_operands,
    )
    filename = "census.json" if args.census_only else "result.json"
    (args.out / filename).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "samples_us"}))


if __name__ == "__main__":
    main()
