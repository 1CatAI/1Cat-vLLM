// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include <cstdint>
#include <cstring>
#include <set>
#include <optional>
#include <vector>

#if CUDART_VERSION >= 12080
namespace {

cudaError_t validate_sampler_child(cudaGraph_t graph) {
  size_t count = 0;
  auto status = cudaGraphGetNodes(graph, nullptr, &count);
  if (status != cudaSuccess) return status;
  std::vector<cudaGraphNode_t> nodes(count);
  status = cudaGraphGetNodes(graph, nodes.data(), &count);
  if (status != cudaSuccess) return status;
  for (auto node : nodes) {
    cudaGraphNodeType type;
    status = cudaGraphNodeGetType(node, &type);
    if (status != cudaSuccess) return status;
    switch (type) {
      case cudaGraphNodeTypeKernel:
      case cudaGraphNodeTypeMemcpy:
      case cudaGraphNodeTypeMemset:
      case cudaGraphNodeTypeEmpty:
        break;
      case cudaGraphNodeTypeGraph: {
        cudaGraph_t child;
        status = cudaGraphChildGraphNodeGetGraph(node, &child);
        if (status != cudaSuccess) return status;
        status = validate_sampler_child(child);
        if (status != cudaSuccess) return status;
        break;
      }
      default:
        // Conditional bodies cannot contain event or host nodes. Reject
        // captured collectives with those nodes before modifying the parent.
        return cudaErrorNotSupported;
    }
  }
  return cudaSuccess;
}

__global__ void set_sampler_branch(cudaGraphConditionalHandle handle,
                                   const bool* flags, int count,
                                   int64_t* counters) {
  if (threadIdx.x == 0) {
    bool use_reference = false;
    for (int i = 0; i < count; ++i) {
      use_reference |= flags[i];
    }
    cudaGraphSetConditional(handle, use_reference);
    if (counters) {
      atomicAdd(reinterpret_cast<unsigned long long*>(counters) +
                    static_cast<int>(use_reference),
                1ULL);
    }
  }
}

}  // namespace
#endif

// Remove only a captured NCCL all-gather stream bridge. Explicit
// predecessor/successor edges retain the collective's place in this
// caller-owned child graph; the parent conditional node orders the complete
// child against prefix work.
int64_t sm70_sampler_graph_prepare_collective(torch::Tensor flags,
                                              int64_t graph_handle) {
  TORCH_CHECK(flags.is_cuda() && graph_handle);
#if CUDART_VERSION < 12080
  return static_cast<int64_t>(cudaErrorNotSupported);
#else
  const at::cuda::OptionalCUDAGuard guard(device_of(flags));
  const auto graph = reinterpret_cast<cudaGraph_t>(graph_handle);
  size_t count = 0;
  auto status = cudaGraphGetNodes(graph, nullptr, &count);
  if (status != cudaSuccess) return static_cast<int64_t>(status);
  std::vector<cudaGraphNode_t> nodes(count);
  status = cudaGraphGetNodes(graph, nodes.data(), &count);
  if (status != cudaSuccess) return static_cast<int64_t>(status);
  struct EventBridge {
    cudaGraphNode_t node;
    bool wait;
    std::vector<cudaGraphNode_t> before, after;
  };
  std::vector<EventBridge> bridges;
  auto adjacent = [](cudaGraphNode_t node, bool predecessors,
                     std::vector<cudaGraphNode_t>& result) {
    size_t size = 0;
    auto get = predecessors ? cudaGraphNodeGetDependencies
                            : cudaGraphNodeGetDependentNodes;
    auto error = get(node, nullptr, &size);
    if (error != cudaSuccess) return error;
    result.resize(size);
    return size ? get(node, result.data(), &size) : cudaSuccess;
  };
  auto is_allgather = [](cudaGraphNode_t node) {
    cudaGraphNodeType type;
    if (cudaGraphNodeGetType(node, &type) != cudaSuccess ||
        type != cudaGraphNodeTypeKernel)
      return false;
    CUDA_KERNEL_NODE_PARAMS params{};
    const char* name = nullptr;
    return cuGraphKernelNodeGetParams(reinterpret_cast<CUgraphNode>(node),
                                      &params) == CUDA_SUCCESS &&
           cuFuncGetName(&name, params.func) == CUDA_SUCCESS && name &&
           // Driver names may include the C++ mangling prefix.
           std::strstr(name, "ncclDevKernel_AllGather_");
  };
  // NCCL puts one wait at its captured stream's start, then a record after
  // each collective. Ordinary producer/consumer kernels can also be direct
  // dependencies of the collective; they must not be rejected or removed.
  for (auto node : nodes) {
    cudaGraphNodeType type;
    status = cudaGraphNodeGetType(node, &type);
    if (status != cudaSuccess) return static_cast<int64_t>(status);
    if (type == cudaGraphNodeTypeWaitEvent ||
        type == cudaGraphNodeTypeEventRecord) {
      EventBridge bridge{node, type == cudaGraphNodeTypeWaitEvent, {}, {}};
      status = adjacent(node, true, bridge.before);
      if (status != cudaSuccess) return static_cast<int64_t>(status);
      status = adjacent(node, false, bridge.after);
      if (status != cudaSuccess) return static_cast<int64_t>(status);
      bridges.push_back(std::move(bridge));
    } else if (type == cudaGraphNodeTypeGraph) {
      cudaGraph_t child;
      status = cudaGraphChildGraphNodeGetGraph(node, &child);
      if (status != cudaSuccess) return static_cast<int64_t>(status);
      status = validate_sampler_child(child);
      if (status != cudaSuccess) return static_cast<int64_t>(status);
    } else if (type != cudaGraphNodeTypeKernel &&
               type != cudaGraphNodeTypeMemcpy &&
               type != cudaGraphNodeTypeMemset &&
               type != cudaGraphNodeTypeEmpty) {
      return static_cast<int64_t>(cudaErrorNotSupported);
    }
  }
  auto find_event = [&bridges](cudaGraphNode_t node) -> const EventBridge* {
    for (const auto& bridge : bridges) {
      if (bridge.node == node) return &bridge;
    }
    return nullptr;
  };
  // Validate all events before mutation. A root wait must feed an identified
  // all-gather. Every record must come from one; only another all-gather or
  // its validated stream-start wait can consume that record.
  for (const auto& bridge : bridges) {
    if (bridge.wait) {
      if (bridge.after.size() != 1 || !is_allgather(bridge.after[0]))
        return static_cast<int64_t>(cudaErrorNotSupported);
      for (auto node : bridge.before) {
        auto event = find_event(node);
        if (!event || event->wait)
          return static_cast<int64_t>(cudaErrorNotSupported);
      }
    } else {
      if (bridge.before.size() != 1 || !is_allgather(bridge.before[0]))
        return static_cast<int64_t>(cudaErrorNotSupported);
      for (auto node : bridge.after) {
        auto event = find_event(node);
        if (!is_allgather(node) && (!event || !event->wait))
          return static_cast<int64_t>(cudaErrorNotSupported);
      }
    }
  }
  std::set<std::pair<cudaGraphNode_t, cudaGraphNode_t> > edges;
  for (const auto& bridge : bridges) {
    for (auto before : bridge.before) {
      if (auto event = find_event(before)) before = event->before[0];
      for (auto after : bridge.after) {
        if (auto event = find_event(after)) after = event->after[0];
        if (before == after) return static_cast<int64_t>(cudaErrorNotSupported);
        edges.emplace(before, after);
      }
    }
  }
  for (const auto& edge : edges) {
    std::vector<cudaGraphNode_t> existing;
    status = adjacent(edge.first, false, existing);
    if (status != cudaSuccess) return static_cast<int64_t>(status);
    bool present = false;
    for (auto node : existing) present |= node == edge.second;
    if (present) continue;
    status = cudaGraphAddDependencies(graph, &edge.first, &edge.second, 1);
    if (status != cudaSuccess) return static_cast<int64_t>(status);
  }
  for (const auto& bridge : bridges) {
    status = cudaGraphDestroyNode(bridge.node);
    if (status != cudaSuccess) return static_cast<int64_t>(status);
  }
  return static_cast<int64_t>(validate_sampler_child(graph));
#endif
}

// Mutate only an uninstantiated, caller-owned parent graph. A nonzero status
// means the caller must discard this graph and retain its existing sampler.
// The flags and both captured child graphs must outlive parent graph replay.
int64_t sm70_sampler_graph_attach_branch(
    torch::Tensor flags, int64_t parent_handle, int64_t reference_handle,
    int64_t compact_handle, std::optional<torch::Tensor> counters) {
  TORCH_CHECK(flags.is_cuda() && flags.scalar_type() == torch::kBool &&
                  flags.is_contiguous() && flags.numel() > 0 &&
                  flags.numel() <= 64,
              "sampler graph flags must be 1..64 contiguous CUDA booleans");
  TORCH_CHECK(parent_handle && reference_handle && compact_handle,
              "sampler graph handles must be nonzero");
  if (counters) {
    TORCH_CHECK(counters->device() == flags.device() &&
                    counters->scalar_type() == torch::kInt64 &&
                    counters->is_contiguous() && counters->numel() == 2,
                "sampler graph counters must be two device int64 values");
  }
#if CUDART_VERSION < 12080
  return static_cast<int64_t>(cudaErrorNotSupported);
#else
  const at::cuda::OptionalCUDAGuard guard(device_of(flags));
  cudaStreamCaptureStatus capture_status;
  auto status =
      cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream(), &capture_status);
  if (status != cudaSuccess) return static_cast<int64_t>(status);
  TORCH_CHECK(capture_status == cudaStreamCaptureStatusNone,
              "attach sampler branches after ending stream capture");
  const auto parent = reinterpret_cast<cudaGraph_t>(parent_handle);
  const auto reference = reinterpret_cast<cudaGraph_t>(reference_handle);
  const auto compact = reinterpret_cast<cudaGraph_t>(compact_handle);
  status = validate_sampler_child(reference);
  if (status != cudaSuccess) return static_cast<int64_t>(status);
  status = validate_sampler_child(compact);
  if (status != cudaSuccess) return static_cast<int64_t>(status);
  size_t node_count = 0;
  status = cudaGraphGetNodes(parent, nullptr, &node_count);
  if (status != cudaSuccess) return static_cast<int64_t>(status);
  std::vector<cudaGraphNode_t> nodes(node_count), leaves;
  status = cudaGraphGetNodes(parent, nodes.data(), &node_count);
  if (status != cudaSuccess) return static_cast<int64_t>(status);
  for (auto node : nodes) {
    size_t dependents = 0;
    status = cudaGraphNodeGetDependentNodes(node, nullptr, &dependents);
    if (status != cudaSuccess) return static_cast<int64_t>(status);
    if (dependents == 0) leaves.push_back(node);
  }

  cudaGraphConditionalHandle condition;
  status = cudaGraphConditionalHandleCreate(&condition, parent, 0,
                                            cudaGraphCondAssignDefault);
  if (status != cudaSuccess) return static_cast<int64_t>(status);
  const bool* data = flags.data_ptr<bool>();
  int count = static_cast<int>(flags.numel());
  int64_t* counter_data = counters ? counters->data_ptr<int64_t>() : nullptr;
  void* arguments[] = {&condition, &data, &count, &counter_data};
  cudaGraphNodeParams setter_params{};
  setter_params.type = cudaGraphNodeTypeKernel;
  setter_params.kernel.func = reinterpret_cast<void*>(set_sampler_branch);
  setter_params.kernel.gridDim = dim3(1);
  setter_params.kernel.blockDim = dim3(32);
  setter_params.kernel.kernelParams = arguments;
  cudaGraphNode_t setter;
  status = cudaGraphAddNode(&setter, parent, leaves.data(), leaves.size(),
                            &setter_params);
  if (status != cudaSuccess) return static_cast<int64_t>(status);

  cudaGraphNodeParams branch_params{};
  branch_params.type = cudaGraphNodeTypeConditional;
  branch_params.conditional.handle = condition;
  branch_params.conditional.type = cudaGraphCondTypeIf;
  branch_params.conditional.size = 2;
  cudaGraphNode_t branch;
  status = cudaGraphAddNode(&branch, parent, &setter, 1, &branch_params);
  if (status != cudaSuccess) return static_cast<int64_t>(status);
  cudaGraphNode_t child;
  status = cudaGraphAddChildGraphNode(
      &child, branch_params.conditional.phGraph_out[0], nullptr, 0, reference);
  if (status != cudaSuccess) return static_cast<int64_t>(status);
  status = cudaGraphAddChildGraphNode(
      &child, branch_params.conditional.phGraph_out[1], nullptr, 0, compact);
  return static_cast<int64_t>(status);
#endif
}
