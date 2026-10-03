// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// nvcc -O3 -arch=sm_70 sm70_peer_store_latency.cu -o peer_store
// Run with CUDA_VISIBLE_DEVICES set to the GPUs under test.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t status = (call);                                           \
    if (status != cudaSuccess) {                                           \
      std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(status)); \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

// This measures SM stores plus a system fence, never a copy engine.
__global__ void peer_store(uint4* destination, const uint4* source, int count) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += blockDim.x * gridDim.x) {
    uint4 value = source[i];
    asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};"
                 :
                 : "l"(destination + i), "r"(value.x), "r"(value.y),
                   "r"(value.z), "r"(value.w)
                 : "memory");
  }
  __threadfence_system();
}

int main() {
  constexpr int bytes = 8 * 5120 * 2;
  constexpr int nodes = 100;
  constexpr int repeats = 20;
  int devices = 0;
  CUDA_CHECK(cudaGetDeviceCount(&devices));
  // Percentiles describe per-node means of graph replays, not individual
  // stores.
  std::printf(
      "source,destination,bytes,blocks,mean_us,graph_mean_p50_us,"
      "graph_mean_p99_us,valid\n");
  for (int source = 0; source < devices; ++source) {
    for (int destination = 0; destination < devices; ++destination) {
      if (source == destination) continue;
      int accessible = 0;
      CUDA_CHECK(cudaDeviceCanAccessPeer(&accessible, source, destination));
      if (!accessible) {
        std::fprintf(stderr, "No peer access: %d -> %d\n", source, destination);
        continue;
      }
      uint4 *input = nullptr, *output = nullptr;
      CUDA_CHECK(cudaSetDevice(destination));
      CUDA_CHECK(cudaMalloc(&output, bytes));
      CUDA_CHECK(cudaMemset(output, 0, bytes));
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaSetDevice(source));
      CUDA_CHECK(cudaDeviceEnablePeerAccess(destination, 0));
      CUDA_CHECK(cudaMalloc(&input, bytes));
      CUDA_CHECK(cudaMemset(input, 0x5a, bytes));
      cudaStream_t stream;
      cudaEvent_t start, end;
      CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      CUDA_CHECK(cudaEventCreate(&start));
      CUDA_CHECK(cudaEventCreate(&end));
      CUDA_CHECK(cudaDeviceSynchronize());
      for (int blocks : {8, 20, 40}) {
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        for (int i = 0; i < nodes; ++i)
          peer_store<<<blocks, 256, 0, stream>>>(output, input, bytes / 16);
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(
            cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        for (int i = 0; i < 3; ++i)
          CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<float> timings;
        float total = 0;
        for (int i = 0; i < repeats; ++i) {
          CUDA_CHECK(cudaEventRecord(start, stream));
          CUDA_CHECK(cudaGraphLaunch(executable, stream));
          CUDA_CHECK(cudaEventRecord(end, stream));
          CUDA_CHECK(cudaEventSynchronize(end));
          float ms = 0;
          CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
          timings.push_back(ms * 1000 / nodes);
          total += timings.back();
        }
        CUDA_CHECK(cudaSetDevice(destination));
        std::vector<unsigned char> host(bytes);
        CUDA_CHECK(
            cudaMemcpy(host.data(), output, bytes, cudaMemcpyDeviceToHost));
        bool valid = std::all_of(host.begin(), host.end(),
                                 [](unsigned char x) { return x == 0x5a; });
        CUDA_CHECK(cudaSetDevice(source));
        std::sort(timings.begin(), timings.end());
        std::printf("%d,%d,%d,%d,%.3f,%.3f,%.3f,%d\n", source, destination,
                    bytes, blocks, total / repeats, timings[repeats / 2],
                    timings.back(), valid);
        std::fflush(stdout);
        if (!valid) return 2;
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(graph));
      }
      CUDA_CHECK(cudaEventDestroy(start));
      CUDA_CHECK(cudaEventDestroy(end));
      CUDA_CHECK(cudaStreamDestroy(stream));
      CUDA_CHECK(cudaFree(input));
      CUDA_CHECK(cudaDeviceDisablePeerAccess(destination));
      CUDA_CHECK(cudaSetDevice(destination));
      CUDA_CHECK(cudaFree(output));
    }
  }
}
