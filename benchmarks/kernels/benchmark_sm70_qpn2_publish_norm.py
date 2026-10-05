# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Research-only real down projection with TP4 publication in its epilogue.

Keep the production FP32 reduction and norm order. Both arms contain two
compute nodes: this screens overlap and communication, not a node-count gain.
Use only a leased, fully NVLink-connected SM70 TP4 group.
"""

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
import runpy
import statistics
from pathlib import Path

import torch
import torch.distributed as dist
from benchmark_sm70_qpn2_stage_skeleton import source_for_screen
from safetensors import safe_open
from torch.utils.cpp_extension import load

from vllm import _sm70_ops as ops
from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
from vllm.platforms import current_platform


def row_norm_source(original):
    """Screen serial CUB128 partials without inter-CTA norm readiness flags.

    Keep all five original partial reductions and their final ordered sum. The
    normalized epilogue rereads the just-written FP32 residual; this is a
    bandwidth tradeoff to screen, not a presumed improvement.
    """
    begin = original.index("template <typename WeightT, bool Reference = false>")
    end = original.index("\nclass CustomAllreduce", begin)
    norm = original[begin:end]
    body_begin = norm.index("  const int pack = row *")
    body_end = norm.index("  using Reduce = cub::BlockReduce<float, Threads>;")
    epilogue_begin = norm.index("  if (tid < PacksPerPart) {", body_end)
    epilogue_end = norm.rfind("\n}")
    prefix = norm[:body_begin].replace(
        "sm70_push_allreduce_gemma_rms_norm", "round12_row_gemma_norm"
    )
    prefix = prefix.replace(
        "const int row = blockIdx.x / Parts, part = blockIdx.x % Parts;",
        "const int row = blockIdx.x;",
    )
    epilogue = norm[epilogue_begin:epilogue_end]
    return (
        prefix
        + """
  using Reduce = cub::BlockReduce<float, Threads>;
  __shared__ typename Reduce::TempStorage storage;
  __shared__ float partials[Parts];
  __shared__ float inverse;
  for (int part = 0; part < Parts; ++part) {
"""
        + norm[body_begin:body_end]
        + """
    variance = Reduce(storage).Reduce(variance, CubAddOp{}, Threads);
    if (tid == 0) partials[part] = variance;
    __syncthreads();
  }
  if (tid == 0) {
    float total = 0;
#pragma unroll
    for (int part = 0; part < Parts; ++part) total += partials[part];
    inverse = rsqrtf(total / Width + epsilon);
    meta->generation[row] = generation;
  }
  __syncthreads();
  for (int part = 0; part < Parts; ++part) {
    const int pack = row * (Width / P::size) + part * PacksPerPart + tid;
    float values[P::size];
    const float4 a = reinterpret_cast<const float4*>(residual_out)[pack * 2];
    const float4 b = reinterpret_cast<const float4*>(residual_out)[pack * 2 + 1];
    values[0]=a.x; values[1]=a.y; values[2]=a.z; values[3]=a.w;
    values[4]=b.x; values[5]=b.y; values[6]=b.z; values[7]=b.w;
"""
        + epilogue
        + "\n  }\n}\n"
    )


def generate(root, row_norm=False):
    original = (root / "csrc/custom_all_reduce.cuh").read_text()
    begin = original.index("template <typename WeightT, bool Reference = false>")
    end = original.index("\nclass CustomAllreduce", begin)
    norm = original[begin:end].replace(
        "sm70_push_allreduce_gemma_rms_norm", "round12_consume_gemma_norm"
    )
    start_push = norm.index("    P value = reinterpret_cast<const P*>(input)[pack];")
    end_push = norm.index("    P peers[4];", start_push)
    norm = norm[:start_push] + norm[end_push:]
    skeleton = source_for_screen(root / "csrc/sm70_turbomind/ops/nvfp4_qpn2_sm70.cu")
    skeleton = skeleton[: skeleton.index("void launch_stage(")]
    begin = skeleton.index("template <int SplitK, int NAcc, int RowTiles = 1,")
    end = skeleton.index(
        "template <int SplitK, int NAcc, int RowTiles = 1, bool", begin + 10
    )
    producer = skeleton[begin:end].replace(
        "nvfp4_qpn2_sm70_kernel", "round12_qpn2_publish_down"
    )
    producer = producer.replace(
        "int m, float global_scale) {",
        "int m, float global_scale, vllm::RankData buffers, int rank) {\n"
        "  __shared__ __align__(16) half published[256];",
    )
    store = (
        "      output[static_cast<size_t>(output_row) * n + tile * 32"
        " + output_col] =\n          __float2half(value);"
    )
    assert store in producer
    producer = producer.replace(
        store, store + "\n      published[element] = __float2half(value);"
    )
    closing = producer.rfind("}")
    producer = (
        producer[:closing]
        + """
  __syncthreads();
  if (threadIdx.x < 32) {
    using P = typename vllm::packed_t<half>::P;
    const int row = threadIdx.x / 4;
    const int pack = row*(5120/8) + tile*4 + threadIdx.x%4;
    const char* local = reinterpret_cast<const char*>(buffers.ptrs[rank]) +
                        kSm70PushNormOffset;
    const auto* meta = reinterpret_cast<const volatile Sm70PushNormMeta*>(local);
    const uint32_t generation = meta->generation[row] + 1;
    const int epoch_offset = (generation&1)*4*(8*5120/8);
    P value = reinterpret_cast<const P*>(published)[threadIdx.x];
#pragma unroll
    for (int i=0; i<8; ++i) vllm::sm70_push_escape_sentinel(value.data[i]);
#pragma unroll
    for (int peer=0; peer<4; ++peer) {
      char* destination = const_cast<char*>(reinterpret_cast<const char*>(
          buffers.ptrs[peer])) + kSm70PushNormOffset + kSm70PushNormMetaBytes +
          (epoch_offset + rank*(8*5120/8))*sizeof(P);
      vllm::sm70_push_store_volatile_16b(value, destination, pack);
    }
  }
}
"""
    )
    return (
        (
            '#include "custom_all_reduce.cuh"\n'
            "using vllm::kSm70PushNormOffset; using vllm::kSm70PushNormMetaBytes;\n"
            "using vllm::kSm70Tp4PushAllreduceBufferBytes; "
            "using vllm::Sm70PushNormMeta;\n"
            "using vllm::Sm70PushNormReferenceMeta; using vllm::kSm70PushNormParts;\n"
            "using vllm::kSm70GemmaRmsNormHiddenSize; using vllm::kSm70PushNormRows;\n"
            + skeleton
            + producer
            + norm
            + (row_norm_source(original) if row_norm else "")
            + """
size_t buffer_bytes() { return kSm70Tp4PushAllreduceBufferBytes; }
vllm::RankData peers(const std::vector<int64_t>& pointers) {
  TORCH_CHECK(pointers.size()==4);
  vllm::RankData result{};
  for(int i=0;i<4;++i) result.ptrs[i]=reinterpret_cast<void*>(pointers[i]);
  return result;
}
__global__ void initialize_norm_packets(char* buffer) {
  auto* meta = reinterpret_cast<Sm70PushNormMeta*>(buffer+kSm70PushNormOffset);
  if(threadIdx.x<80) reinterpret_cast<float*>(meta->partial)[threadIdx.x]=-1.0f;
}
void initialize(const std::vector<int64_t>& pointers,int rank) {
  char* buffer = reinterpret_cast<char*>(pointers.at(rank));
  auto stream = at::cuda::getCurrentCUDAStream();
  cudaMemsetAsync(buffer+kSm70PushNormOffset,0,kSm70PushNormMetaBytes,stream);
  cudaMemsetAsync(buffer+kSm70PushNormOffset+kSm70PushNormMetaBytes,0x7f,
      kSm70Tp4PushAllreduceBufferBytes-kSm70PushNormOffset-kSm70PushNormMetaBytes,stream);
  initialize_norm_packets<<<1,128,0,stream>>>(buffer);
}
void down(torch::Tensor y,torch::Tensor x,torch::Tensor c,torch::Tensor s,
          double scale,const std::vector<int64_t>& pointers,int rank,bool publish) {
  TORCH_CHECK(x.size(0)==8 && y.size(1)==5120);
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto* codes=c.data_ptr<uint8_t>(); const auto* scales=s.data_ptr<uint8_t>();
  const auto* input=reinterpret_cast<const half*>(x.data_ptr<at::Half>());
  auto* output=reinterpret_cast<half*>(y.data_ptr<at::Half>());
  if(publish) round12_qpn2_publish_down<16,2><<<160,512,0,stream>>>(
      codes,scales,input,output,5120,x.size(1),8,scale,peers(pointers),rank);
  else nvfp4_qpn2_sm70_kernel<16,2><<<160,512,0,stream>>>(
      codes,scales,input,output,5120,x.size(1),8,scale);
}
void launch_norm(torch::Tensor y,torch::Tensor residual_out,torch::Tensor input,
          torch::Tensor residual,torch::Tensor weight,
          const std::vector<int64_t>& pointers,int rank,bool consume) {
  const auto buffers=peers(pointers); auto stream=at::cuda::getCurrentCUDAStream();
  const auto* x=reinterpret_cast<const half*>(input.data_ptr<at::Half>());
  const auto* r=residual.data_ptr<float>(); const auto* w=weight.data_ptr<float>();
  auto* out=reinterpret_cast<half*>(y.data_ptr<at::Half>());
  auto* rout=residual_out.data_ptr<float>();
  if(consume) ROUND12_NORM_VARIANT<float><<<ROUND12_NORM_GRID,128,0,stream>>>(
      buffers,x,r,w,out,rout,rank,1e-6f);
  else vllm::sm70_push_allreduce_gemma_rms_norm<float><<<40,128,0,stream>>>(
      buffers,x,r,w,out,rout,rank,1e-6f);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("down",&down);m.def("norm",&launch_norm);m.def("initialize",&initialize);
  m.def("buffer_bytes",&buffer_bytes);
}
"""
        )
        .replace(
            "ROUND12_NORM_VARIANT",
            "round12_row_gemma_norm" if row_norm else "round12_consume_gemma_norm",
        )
        .replace("ROUND12_NORM_GRID", "8" if row_norm else "40")
    )


def graph_kernel_nodes(graph):
    driver = ctypes.CDLL("libcuda.so.1")
    handle = ctypes.c_void_p(graph.raw_cuda_graph())
    size = ctypes.c_size_t()
    assert driver.cuGraphGetNodes(handle, None, ctypes.byref(size)) == 0
    nodes = (ctypes.c_void_p * size.value)()
    assert driver.cuGraphGetNodes(handle, nodes, ctypes.byref(size)) == 0
    kernels = 0
    for node in nodes:
        kind = ctypes.c_int()
        assert driver.cuGraphNodeGetType(ctypes.c_void_p(node), ctypes.byref(kind)) == 0
        kernels += kind.value == 0
    return kernels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--row-norm-screen", action="store_true")
    parser.add_argument("--norm-only-screen", action="store_true")
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()
    if args.norm_only_screen and not args.row_norm_screen:
        parser.error("--norm-only-screen requires --row-norm-screen")
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.out / "publish-down.cu"
    source.write_text(generate(args.source_root, args.row_norm_screen))
    name = "round12_qpn2_publish_norm"
    if args.row_norm_screen:
        name += "_row"
    if args.extension:
        spec = importlib.util.spec_from_file_location(name, args.extension)
        extension = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(extension)
    else:
        extension = load(
            name=name,
            sources=[str(source)],
            extra_include_paths=[
                str(args.source_root / "csrc"),
                str(args.source_root / "csrc/sm70_turbomind/ops"),
            ],
            extra_cuda_cflags=["-O3", "-lineinfo"],
            verbose=True,
        )
    if args.compile_only:
        return
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    dist.init_process_group("nccl")
    group = dist.new_group(backend="gloo")
    assert dist.get_world_size() == 4
    assert torch.cuda.get_device_capability() == (7, 0)
    assert current_platform.is_fully_connected([0, 1, 2, 3])
    pointers = CustomAllreduce.create_shared_buffer(extension.buffer_bytes(), group)
    extension.initialize(pointers, rank)
    torch.cuda.synchronize()
    dist.barrier()
    helper = runpy.run_path(
        str(args.source_root / "benchmarks/kernels/benchmark_sm70_nvfp4_qpn2.py")
    )
    projection = next(
        p
        for p in helper["_load_projection_shards"](args.model, 0, rank, 4)
        if not p.gated_silu
    )
    codes, scales = ops.nvfp4_qpn2_prepare_sm70(
        projection.packed.cuda(), projection.scales.cuda()
    )
    torch.manual_seed(123 + rank)
    x = (
        torch.randn(
            8, projection.packed.shape[1] * 2, device="cuda", dtype=torch.float16
        )
        * 0.1
    )
    with safe_open(args.model / "model.safetensors", framework="pt") as weights:
        weight = (
            weights.get_tensor("model.language_model.layers.1.input_layernorm.weight")
            .float()
            .cuda()
        )
    torch.manual_seed(345)
    residual = torch.randn(8, 5120, device="cuda", dtype=torch.float32) * 0.1
    outputs = [
        (
            torch.empty(8, 5120, device="cuda", dtype=torch.float16),
            torch.empty(8, 5120, device="cuda", dtype=torch.float16),
            torch.empty_like(residual),
        )
        for _ in range(2)
    ]
    eviction = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)

    def launch(arm):
        partial, normalized, rout = outputs[arm]
        if not args.norm_only_screen:
            extension.down(
                partial,
                x,
                codes,
                scales,
                projection.inverse_global_scale,
                pointers,
                rank,
                bool(arm) and not args.row_norm_screen,
            )
        extension.norm(
            normalized, rout, partial, residual, weight, pointers, rank, bool(arm)
        )

    graphs = []
    for arm in range(2):
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        start = torch.cuda.Event(enable_timing=True, external=True)
        end = torch.cuda.Event(enable_timing=True, external=True)
        dist.barrier()
        with torch.cuda.graph(graph):
            eviction.fill_(1)
            start.record()
            launch(arm)
            end.record()
        graphs.append((graph, start, end))
    for amplitude in (0.0, -0.1, 0.1, 1.0):
        x.normal_().mul_(amplitude)
        if args.norm_only_screen:
            for partial, _, _ in outputs:
                extension.down(
                    partial,
                    x,
                    codes,
                    scales,
                    projection.inverse_global_scale,
                    pointers,
                    rank,
                    False,
                )
            torch.cuda.synchronize()
        dist.barrier()
        for graph, _, _ in graphs:
            graph.replay()
        torch.cuda.synchronize()
        for control, candidate in zip(outputs[0], outputs[1]):
            assert torch.equal(control.view(torch.int16), candidate.view(torch.int16))
    samples = [[], []]
    for iteration in range(args.iters + 20):
        for arm in (0, 1) if iteration % 2 == 0 else (1, 0):
            dist.barrier()
            graph, start, end = graphs[arm]
            graph.replay()
            end.synchronize()
            if iteration >= 20:
                samples[arm].append(start.elapsed_time(end) * 1000)
    row = dict(
        rank=rank,
        control_us=statistics.median(samples[0]),
        candidate_us=statistics.median(samples[1]),
        candidate="row_norm" if args.row_norm_screen else "epilogue_publication",
        samples_us=samples,
        bitwise_four_amplitudes=True,
        projection_included=not args.norm_only_screen,
        compute_nodes_per_arm=1 if args.norm_only_screen else 2,
        measured_graph_kernel_nodes=[graph_kernel_nodes(g) for g, _, _ in graphs],
        eviction_included_in_graph_count=True,
        torch=torch.__version__,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    (args.out / f"rank{rank}.json").write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row), flush=True)
    dist.barrier()
    CustomAllreduce.free_shared_buffer(pointers, rank=rank)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
