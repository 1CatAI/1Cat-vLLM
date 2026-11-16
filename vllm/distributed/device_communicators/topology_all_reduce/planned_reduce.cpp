// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

void planned_raw(const uint4*, uint4*, const std::vector<int64_t>&, int*, int,
                 int, int, int, int, cudaStream_t);
struct Scope {
  int old;
  explicit Scope(int d) {
    C10_CUDA_CHECK(cudaGetDevice(&old));
    C10_CUDA_CHECK(cudaSetDevice(d));
  }
  ~Scope() { cudaSetDevice(old); }
};
std::tuple<int64_t, pybind11::bytes> allocate(int device, int64_t bytes) {
  TORCH_CHECK(bytes > 0 && bytes <= 1024LL * 1024 * 1024);
  Scope scope(device);
  void* ptr;
  C10_CUDA_CHECK(cudaMalloc(&ptr, bytes));
  C10_CUDA_CHECK(cudaMemset(ptr, 0xff, bytes));
  cudaIpcMemHandle_t handle;
  C10_CUDA_CHECK(cudaIpcGetMemHandle(&handle, ptr));
  return {(int64_t)ptr, pybind11::bytes((char*)&handle, sizeof(handle))};
}
int64_t open_handle(int device, pybind11::bytes value) {
  Scope scope(device);
  std::string raw = value;
  TORCH_CHECK(raw.size() == sizeof(cudaIpcMemHandle_t));
  cudaIpcMemHandle_t handle;
  memcpy(&handle, raw.data(), sizeof(handle));
  void* ptr;
  C10_CUDA_CHECK(
      cudaIpcOpenMemHandle(&ptr, handle, cudaIpcMemLazyEnablePeerAccess));
  return (int64_t)ptr;
}
void release(int device, int64_t ptr, bool imported) {
  Scope scope(device);
  if (imported)
    C10_CUDA_CHECK(cudaIpcCloseMemHandle((void*)ptr));
  else
    C10_CUDA_CHECK(cudaFree((void*)ptr));
}
void enable_peer(int device, int target) {
  Scope scope(device);
  auto status = cudaDeviceEnablePeerAccess(target, 0);
  if (status == cudaErrorPeerAccessAlreadyEnabled) {
    cudaGetLastError();
    return;
  }
  C10_CUDA_CHECK(status);
}
void reset(int device, int64_t ptr, int64_t bytes) {
  Scope scope(device);
  C10_CUDA_CHECK(cudaMemset((void*)ptr, 0xff, bytes));
}

class Engine {
  int device, candidate, threads, blocks, capacity;
  std::vector<int> owned, devices;
  std::vector<std::vector<int64_t>> pointers;
  std::vector<torch::Tensor> flags;
  std::vector<cudaStream_t> streams;
  std::vector<cudaEvent_t> done;
  cudaEvent_t ready = nullptr;

 public:
  Engine(int device_, std::vector<int> owned_, std::vector<int> devices_,
         std::vector<std::vector<int64_t>> pointers_,
         std::vector<torch::Tensor> flags_, int candidate_, int threads_,
         int blocks_, int capacity_)
      : device(device_),
        candidate(candidate_),
        threads(threads_),
        blocks(blocks_),
        capacity(capacity_),
        owned(owned_),
        devices(devices_),
        pointers(pointers_),
        flags(flags_) {
    TORCH_CHECK(!owned.empty() && owned.size() == pointers.size() &&
                owned.size() == flags.size());
    TORCH_CHECK((threads == 64 || threads == 128 || threads == 256) &&
                blocks >= 1 && blocks <= 32 && capacity > 0 &&
                capacity % 16 == 0);
    Scope scope(device);
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    for (size_t i = 0; i < owned.size(); ++i) {
      int node = owned[i];
      TORCH_CHECK(node >= 0 && node < (int)devices.size() &&
                  pointers[i].size() == devices.size());
      TORCH_CHECK(flags[i].is_cuda() &&
                  flags[i].get_device() == devices[node] &&
                  flags[i].scalar_type() == at::kInt &&
                  flags[i].is_contiguous() && flags[i].numel() >= blocks);
      Scope other(devices[node]);
      cudaStream_t stream = nullptr;
      cudaEvent_t event = nullptr;
      if (devices[node] != device) {
        C10_CUDA_CHECK(
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        C10_CUDA_CHECK(
            cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
      }
      streams.push_back(stream);
      done.push_back(event);
    }
  }
  void run(torch::Tensor x, torch::Tensor out) {
    TORCH_CHECK(x.is_cuda() && x.get_device() == device &&
                out.device() == x.device() && x.scalar_type() == at::kHalf &&
                out.scalar_type() == at::kHalf);
    TORCH_CHECK(x.is_contiguous() && out.is_contiguous() &&
                x.nbytes() == out.nbytes() && x.nbytes() > 0 &&
                x.nbytes() <= capacity && x.nbytes() % 16 == 0);
    TORCH_CHECK((uintptr_t)x.data_ptr() % 16 == 0 &&
                (uintptr_t)out.data_ptr() % 16 == 0);
    Scope scope(device);
    auto main = c10::cuda::getCurrentCUDAStream(device).stream();
    bool helpers = owned.size() > 1;
    if (helpers) C10_CUDA_CHECK(cudaEventRecord(ready, main));
    for (size_t i = 0; i < owned.size(); ++i) {
      bool helper = devices[owned[i]] != device;
      Scope other(devices[owned[i]]);
      auto stream = helper ? streams[i] : main;
      if (helper) C10_CUDA_CHECK(cudaStreamWaitEvent(stream, ready, 0));
      planned_raw(helper ? nullptr : (const uint4*)x.data_ptr(),
                  helper ? nullptr : (uint4*)out.data_ptr(), pointers[i],
                  (int*)flags[i].data_ptr(), candidate, owned[i], threads,
                  blocks, x.nbytes() / 16, stream);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      if (helper) C10_CUDA_CHECK(cudaEventRecord(done[i], stream));
    }
    if (helpers)
      for (auto event : done)
        if (event) C10_CUDA_CHECK(cudaStreamWaitEvent(main, event, 0));
  }
  void close() {
    if (!ready) return;
    Scope scope(device);
    C10_CUDA_CHECK(cudaEventDestroy(ready));
    ready = nullptr;
    for (size_t i = 0; i < owned.size(); ++i)
      if (streams[i]) {
        Scope other(devices[owned[i]]);
        C10_CUDA_CHECK(cudaStreamDestroy(streams[i]));
        C10_CUDA_CHECK(cudaEventDestroy(done[i]));
        streams[i] = nullptr;
      }
  }
  ~Engine() {
    try {
      close();
    } catch (...) {
    }
  }
};
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("allocate", &allocate);
  m.def("open_handle", &open_handle);
  m.def("release", &release);
  m.def("enable_peer", &enable_peer);
  m.def("reset", &reset);
  pybind11::class_<Engine>(m, "Engine", pybind11::module_local())
      .def(pybind11::init<int, std::vector<int>, std::vector<int>,
                          std::vector<std::vector<int64_t>>,
                          std::vector<torch::Tensor>, int, int, int, int>())
      .def("run", &Engine::run)
      .def("close", &Engine::close);
}
