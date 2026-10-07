# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen forty-CTA direct pull against push with identical norm arithmetic.

Both arms use independent CUDA IPC allocations, not native class ABI overlays.
Pull retains five CUB128 partials per row and the original ordered FP32 sum.
Start and end peer handshakes prevent premature input reads and input reuse.
This extension is research-only; no serving implementation is registered.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open
from torch.utils.cpp_extension import load

from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
from vllm.platforms import current_platform


def generate(root):
    text = (root / "csrc/custom_all_reduce.cuh").read_text()
    begin = text.index("template <typename WeightT, bool Reference = false>")
    end = text.index("\nclass CustomAllreduce", begin)
    local = text[begin:end].replace(
        "sm70_push_allreduce_gemma_rms_norm", "push_local_norm"
    )
    marker = """      vllm::sm70_push_store_volatile_16b(
          value,"""
    assert local.count(marker) == 1
    local = local.replace(marker, "      if (peer != rank) " + marker.strip())
    load_begin = local.index("        vllm::sm70_push_load_volatile_16b(")
    load_end = local.index("#pragma unroll", load_begin)
    local = (
        local[:load_begin]
        + "        if (peer == rank) peers[peer] = value;\n        else {\n"
        + local[load_begin:load_end]
        + "        }\n"
        + local[load_end:]
    )
    marker = """      vllm::sm70_push_store_volatile_16b(
          empty,"""
    assert local.count(marker) == 1
    local = local.replace(marker, "      if (peer != rank) " + marker.strip())
    norm = text[begin:end].replace(
        "sm70_push_allreduce_gemma_rms_norm", "direct_pull_norm"
    )
    norm = norm.replace(
        "vllm::RankData buffers, const half* input,",
        "vllm::RankData buffers, vllm::RankData inputs, const half* input,",
    )
    marker = "  float values[P::size] = {};"
    norm = norm.replace(
        marker,
        """
  auto* own_signals = reinterpret_cast<PullSignals*>(
      const_cast<void*>(buffers.ptrs[rank])) + 0;
  own_signals = reinterpret_cast<PullSignals*>(
      reinterpret_cast<char*>(own_signals) + kSm70Tp4PushAllreduceBufferBytes);
  const uint32_t epoch = own_signals->epoch[blockIdx.x] + 1;
  if (tid < 4) {
    auto* peer = reinterpret_cast<PullSignals*>(
        const_cast<char*>(reinterpret_cast<const char*>(buffers.ptrs[tid])) +
        kSm70Tp4PushAllreduceBufferBytes);
    vllm::st_flag_sys_visible(&peer->start[blockIdx.x][rank], epoch);
    while (vllm::ld_flag_volatile(&own_signals->start[blockIdx.x][tid]) != epoch) {}
    vllm::membar_sys();
  }
  __syncthreads();
"""
        + marker,
    )
    begin = norm.index("    P value = reinterpret_cast<const P*>(input)[pack];")
    end = norm.index("    const P sum =", begin)
    norm = (
        norm[:begin]
        + """
    P peers[4];
#pragma unroll
    for (int peer = 0; peer < 4; ++peer) {
      vllm::sm70_push_load_volatile_16b(peers[peer], inputs.ptrs[peer], pack);
#pragma unroll
      for (int i = 0; i < P::size; ++i)
        vllm::sm70_push_escape_sentinel(peers[peer].data[i]);
    }
"""
        + norm[end:]
    )
    begin = norm.index("    P empty;")
    end = norm.index("\n  }\n  using Reduce", begin)
    norm = norm[:begin] + norm[end:]
    end = norm.rfind("\n}")
    norm = (
        norm[:end]
        + """
  __syncthreads();
  if (tid < 4) {
    auto* peer = reinterpret_cast<PullSignals*>(
        const_cast<char*>(reinterpret_cast<const char*>(buffers.ptrs[tid])) +
        kSm70Tp4PushAllreduceBufferBytes);
    vllm::st_flag_release(&peer->end[blockIdx.x][rank], epoch);
    while (vllm::ld_flag_acquire(&own_signals->end[blockIdx.x][tid]) != epoch) {}
  }
  __syncthreads();
  if (tid == 0) own_signals->epoch[blockIdx.x] = epoch;
"""
        + norm[end:]
    )
    return (
        r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include "custom_all_reduce.cuh"
using namespace vllm;
struct PullSignals {
  uint32_t epoch[40];
  uint32_t start[40][4];
  uint32_t end[40][4];
};
"""
        + norm
        + local
        + r"""
size_t buffer_bytes() {
  return kSm70Tp4PushAllreduceBufferBytes + sizeof(PullSignals);
}
RankData peers(const std::vector<int64_t>& pointers) {
  TORCH_CHECK(pointers.size() == 4);
  RankData data{};
  for (int i=0; i<4; ++i) data.ptrs[i] = reinterpret_cast<void*>(pointers[i]);
  return data;
}
torch::Tensor alias(int64_t pointer, int device) {
  return torch::from_blob(reinterpret_cast<void*>(pointer), {8,5120},
      [](void*){}, torch::TensorOptions().device(torch::kCUDA, device)
      .dtype(torch::kFloat16));
}
__global__ void initialize_partials(char* buffer) {
  auto* meta = reinterpret_cast<Sm70PushNormMeta*>(buffer+kSm70PushNormOffset);
  if (threadIdx.x<80) reinterpret_cast<float*>(meta->partial)[threadIdx.x]=-1.f;
}
void initialize(const std::vector<int64_t>& pointers, int rank) {
  auto* base = reinterpret_cast<char*>(pointers.at(rank));
  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(base,0,buffer_bytes(),stream));
  C10_CUDA_CHECK(cudaMemsetAsync(
      base+kSm70PushNormOffset+kSm70PushNormMetaBytes, 0x7f,
      kSm70Tp4PushAllreduceBufferBytes-kSm70PushNormOffset-
      kSm70PushNormMetaBytes,stream));
  initialize_partials<<<1,128,0,stream>>>(base);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void launch(torch::Tensor output, torch::Tensor rout, torch::Tensor input,
            torch::Tensor residual, torch::Tensor weight,
            const std::vector<int64_t>& buffers,
            const std::vector<int64_t>& inputs, int rank, int mode) {
  TORCH_CHECK(input.sizes()==torch::IntArrayRef({8,5120}) &&
              input.scalar_type()==torch::kFloat16 && input.is_contiguous());
  TORCH_CHECK(weight.scalar_type()==torch::kFloat32 &&
              residual.scalar_type()==torch::kFloat32);
  TORCH_CHECK(input.data_ptr()==reinterpret_cast<void*>(inputs.at(rank)));
  auto stream = at::cuda::getCurrentCUDAStream();
  auto* x = reinterpret_cast<const half*>(input.data_ptr<at::Half>());
  auto* y = reinterpret_cast<half*>(output.data_ptr<at::Half>());
  auto* r = residual.data_ptr<float>(); auto* w = weight.data_ptr<float>();
  auto* ro = rout.data_ptr<float>();
  if (mode == 1) direct_pull_norm<float><<<40,128,0,stream>>>(
      peers(buffers),peers(inputs),x,r,w,y,ro,rank,1e-6f);
  else if (mode == 2) push_local_norm<float><<<40,128,0,stream>>>(
      peers(buffers),x,r,w,y,ro,rank,1e-6f);
  else sm70_push_allreduce_gemma_rms_norm<float><<<40,128,0,stream>>>(
      peers(buffers),x,r,w,y,ro,rank,1e-6f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("alias",&alias); m.def("launch",&launch);
  m.def("initialize",&initialize); m.def("buffer_bytes",&buffer_bytes);
}
"""
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--burst", type=int, default=1)
    parser.add_argument("--norm-partial-packets", action="store_true")
    parser.add_argument("--versioned-triples", action="store_true")
    parser.add_argument("--local-buffer-pointer", action="store_true")
    parser.add_argument("--norm-packet-parts", type=int, choices=(5, 10, 20), default=5)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.out / "pull_norm.cu"
    if args.norm_partial_packets:
        from sm70_tp4_norm_partial_packets import generate as packet_source

        if args.local_buffer_pointer:
            from sm70_tp4_local_buffer_pointer import generate as local_source

            assert not args.versioned_triples
            text = local_source(args.source_root, args.norm_packet_parts)
        elif args.versioned_triples:
            from sm70_tp4_versioned_triples import generate as wire_source

            assert args.norm_packet_parts == 5
            text = wire_source(args.source_root)
        else:
            text = packet_source(args.source_root, args.norm_packet_parts)
    else:
        text = generate(args.source_root)
    if not source.exists() or source.read_text() != text:
        source.write_text(text)
    name = (
        f"tp4_norm_local_pointer{args.norm_packet_parts}_screen"
        if args.local_buffer_pointer
        else "tp4_norm_versioned_triples_screen"
        if args.versioned_triples
        else f"tp4_norm_partial_packets{args.norm_packet_parts}_screen"
        if args.norm_partial_packets and args.norm_packet_parts != 5
        else "tp4_norm_partial_packets_screen"
        if args.norm_partial_packets
        else "tp4_pull_norm_screen"
    )
    if args.extension:
        spec = importlib.util.spec_from_file_location(name, args.extension)
        assert spec and spec.loader
        extension = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(extension)
    else:
        extension = load(
            name=name,
            sources=[str(source)],
            extra_include_paths=[str(args.source_root / "csrc")],
            extra_cuda_cflags=["-O3", "-lineinfo", "--ptxas-options=-v"],
            verbose=True,
        )
    if args.compile_only:
        return
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    dist.init_process_group(
        "nccl", device_id=torch.device("cuda", rank), timeout=timedelta(seconds=120)
    )
    group = dist.new_group(backend="gloo")
    assert dist.get_world_size() == 4
    assert torch.cuda.get_device_capability() == (7, 0)
    assert current_platform.is_fully_connected([0, 1, 2, 3])
    buffers = CustomAllreduce.create_shared_buffer(extension.buffer_bytes(), group)
    inputs = CustomAllreduce.create_shared_buffer(8 * 5120 * 2, group)
    extension.initialize(buffers, rank)
    x = extension.alias(inputs[rank], rank)
    torch.cuda.synchronize()
    dist.barrier()
    torch.manual_seed(123 + rank)
    original = torch.randn_like(x)
    torch.manual_seed(345)
    residual = torch.randn(8, 5120, device="cuda") * 0.125
    with safe_open(args.model / "model.safetensors", framework="pt") as f:
        weight = (
            f.get_tensor("model.language_model.layers.1.input_layernorm.weight")
            .float()
            .cuda()
        )
    outputs = [(torch.empty_like(x), torch.empty_like(residual)) for _ in range(3)]

    def launch(arm):
        y, rout = outputs[arm]
        extension.launch(y, rout, x, residual, weight, buffers, inputs, rank, arm)

    differences = []
    for amplitude in (0.01, 0.125, 1.0, 4.0):
        x.copy_(original * amplitude)
        torch.cuda.synchronize()
        dist.barrier()
        for arm in range(3):
            launch(arm)
            torch.cuda.synchronize()
        for arm, candidate in enumerate(outputs[1:], start=1):
            for component, (a, b) in enumerate(zip(outputs[0], candidate)):
                bits = torch.int16 if a.element_size() == 2 else torch.int32
                exact = torch.equal(a.view(bits), b.view(bits))
                differences.append(
                    {
                        "amplitude": amplitude,
                        "arm": arm,
                        "component": "normalized" if component == 0 else "residual",
                        "bitwise": exact,
                        "max_abs": (a.float() - b.float()).abs().max().item(),
                    }
                )
                assert torch.isfinite(b).all()
                if not (
                    args.norm_partial_packets
                    and args.norm_packet_parts != 5
                    and arm == 1
                    and component == 0
                ):
                    assert exact
    x.copy_(original * 0.125)
    eviction = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    graphs = []
    for arm in range(3):
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        start = torch.cuda.Event(enable_timing=True, external=True)
        end = torch.cuda.Event(enable_timing=True, external=True)
        with torch.cuda.graph(graph):
            eviction.fill_(1)
            start.record()
            for _ in range(args.burst):
                launch(arm)
            end.record()
        graphs.append((graph, start, end))
    samples = [[], [], []]
    for i in range(args.iters + 20):
        for arm in ((i + j) % 3 for j in range(3)):
            graph, start, end = graphs[arm]
            graph.replay()
            end.synchronize()
            if i >= 20:
                samples[arm].append(start.elapsed_time(end) * 1000 / args.burst)
    result = {
        "rank": rank,
        "burst": args.burst,
        "push_mean_us": statistics.mean(samples[0]),
        "candidate_mean_us": statistics.mean(samples[1]),
        "candidate_route": "local_buffer_pointer"
        if args.local_buffer_pointer
        else "versioned_triples"
        if args.versioned_triples
        else "norm_partial_packets"
        if args.norm_partial_packets
        else "direct_pull",
        "local_push_mean_us": statistics.mean(samples[2]),
        "samples_us": samples,
        "output_and_residual_bitwise": all(d["bitwise"] for d in differences),
        "differences": differences,
        "norm_packet_parts": args.norm_packet_parts,
        "kernel_nodes_in_timed_range": args.burst,
        "generated_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "research_only": True,
    }
    (args.out / f"rank{rank}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "samples_us"}))
    torch.cuda.synchronize()
    dist.barrier()
    CustomAllreduce.free_shared_buffer(inputs, rank=rank)
    CustomAllreduce.free_shared_buffer(buffers, rank=rank)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
