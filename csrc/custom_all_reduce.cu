#include <atomic>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <torch/all.h>

#include "custom_all_reduce.cuh"

// Fake pointer type, must match fptr_t type in ops.h.
// We use this type alias to indicate when pointers are passed in as int64_t.
using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

bool sm70_profile_trace_enabled() {
  const char* value = std::getenv("VLLM_SM70_PROFILE_TRACE");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

const char* scalar_type_name(at::ScalarType scalar_type) {
  switch (scalar_type) {
    case at::ScalarType::Float:
      return "float32";
    case at::ScalarType::Half:
      return "float16";
    case at::ScalarType::BFloat16:
      return "bfloat16";
    default:
      return "other";
  }
}

const char* capture_status_name(cudaStreamCaptureStatus status) {
  switch (status) {
    case cudaStreamCaptureStatusNone:
      return "none";
    case cudaStreamCaptureStatusActive:
      return "active";
    case cudaStreamCaptureStatusInvalidated:
      return "invalidated";
    default:
      return "unknown";
  }
}

fptr_t init_custom_ar(const std::vector<fptr_t>& fake_ipc_ptrs,
                      torch::Tensor& rank_data, int64_t rank,
                      bool fully_connected) {
  int world_size = fake_ipc_ptrs.size();
  if (world_size > 8)
    throw std::invalid_argument("world size > 8 is not supported");
  if (world_size % 2 != 0)
    throw std::invalid_argument("Odd num gpus is not supported for now");
  if (rank < 0 || rank >= world_size)
    throw std::invalid_argument("invalid rank passed in");

  vllm::Signal* ipc_ptrs[8];
  for (int i = 0; i < world_size; i++) {
    ipc_ptrs[i] = reinterpret_cast<vllm::Signal*>(fake_ipc_ptrs[i]);
  }
  return (fptr_t) new vllm::CustomAllreduce(ipc_ptrs, rank_data.data_ptr(),
                                            rank_data.numel(), rank, world_size,
                                            fully_connected);
}

/**
 * Make sure tensor t's data lies completely within ((char)t.data_ptr()) +
 * t.numel() * t.element_size(). This is slightly weaker than t.is_contiguous()
 * because it allows transpose of contiguous slice (i.e. slicing the first
 * dimension). Currently, we require this because stride information is not
 * passed into the kernels and we treat input tensors as flat.
 *
 * Examples
 * A = torch.zeros(3, 3, 3)
 * 1. A: OK
 * 2. A[1:]: OK
 * 3. A.permute(2, 0, 1): OK
 * 4. A[1:].permute(2, 0, 1): OK
 * 5. A[None].expand(2, -1, -1, -1): Not OK
 * 6. A[:, 1:, 1:]: Not OK
 */
bool _is_weak_contiguous(torch::Tensor& t) {
  return t.is_contiguous() ||
         (t.storage().nbytes() - t.storage_offset() * t.element_size() ==
          t.numel() * t.element_size());
}

/**
 * Performs an out-of-place allreduce and stores result in out.
 *
 * If _reg_buffer is null, assumes inp.data_ptr() is already IPC-registered.
 * Otherwise, _reg_buffer is assumed to be IPC-registered and inp is first
 * copied into _reg_buffer.
 */
void all_reduce(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out,
                fptr_t _reg_buffer, int64_t reg_buffer_sz_bytes) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(inp));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(inp.numel(), out.numel());
  TORCH_CHECK(_is_weak_contiguous(out));
  TORCH_CHECK(_is_weak_contiguous(inp));
  auto input_size = inp.numel() * inp.element_size();
  auto reg_buffer = reinterpret_cast<void*>(_reg_buffer);
  if (reg_buffer) {
    TORCH_CHECK_LE(input_size, reg_buffer_sz_bytes);
    AT_CUDA_CHECK(cudaMemcpyAsync(reg_buffer, inp.data_ptr(), input_size,
                                  cudaMemcpyDeviceToDevice, stream));
  } else {
    reg_buffer = inp.data_ptr();
  }

  switch (out.scalar_type()) {
    case at::ScalarType::Float: {
      fa->allreduce<float>(stream, reinterpret_cast<float*>(reg_buffer),
                           reinterpret_cast<float*>(out.data_ptr()),
                           out.numel());
      break;
    }
    case at::ScalarType::Half: {
      fa->allreduce<half>(stream, reinterpret_cast<half*>(reg_buffer),
                          reinterpret_cast<half*>(out.data_ptr()), out.numel());
      break;
    }
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case at::ScalarType::BFloat16: {
      fa->allreduce<nv_bfloat16>(
          stream, reinterpret_cast<nv_bfloat16*>(reg_buffer),
          reinterpret_cast<nv_bfloat16*>(out.data_ptr()), out.numel());
      break;
    }
#endif
    default:
      throw std::runtime_error(
          "custom allreduce only supports float32, float16 and bfloat16");
  }
}

namespace {

template <int kWorldSize>
void sm70_all_reduce_gemma_rms_norm_impl(
    fptr_t _fa, torch::Tensor& inp, torch::Tensor& residual,
    torch::Tensor& weight, torch::Tensor& normalized_out,
    torch::Tensor& residual_out, fptr_t _reg_buffer,
    int64_t reg_buffer_sz_bytes, double epsilon) {
  constexpr int64_t kHiddenSize = vllm::kSm70GemmaRmsNormHiddenSize;
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(inp));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK_EQ(inp.scalar_type(), at::ScalarType::Half);
  TORCH_CHECK_EQ(normalized_out.scalar_type(), at::ScalarType::Half);
  TORCH_CHECK_EQ(residual_out.scalar_type(), at::ScalarType::Float);
  if constexpr (kWorldSize == 4) {
    TORCH_CHECK(residual.scalar_type() == at::ScalarType::Float,
                "SM70 TP4 Gemma RMSNorm prototype residual must be float32.");
  } else {
    TORCH_CHECK(residual.scalar_type() == at::ScalarType::Half ||
                residual.scalar_type() == at::ScalarType::Float,
                "SM70 Gemma RMSNorm prototype residual must be float16 or "
                "float32.");
  }
  TORCH_CHECK(weight.scalar_type() == at::ScalarType::Half ||
              weight.scalar_type() == at::ScalarType::Float,
              "SM70 Gemma RMSNorm prototype weight must be float16 or "
              "float32.");
  TORCH_CHECK_EQ(inp.dim(), 2);
  TORCH_CHECK_EQ(inp.size(1), kHiddenSize);
  TORCH_CHECK(residual.sizes() == inp.sizes(),
              "SM70 Gemma RMSNorm prototype residual shape must match inp.");
  TORCH_CHECK(normalized_out.sizes() == inp.sizes(),
              "SM70 Gemma RMSNorm prototype normalized_out shape must match "
              "inp.");
  TORCH_CHECK(residual_out.sizes() == inp.sizes(),
              "SM70 Gemma RMSNorm prototype residual_out shape must match inp.");
  TORCH_CHECK_EQ(weight.dim(), 1);
  TORCH_CHECK_EQ(weight.numel(), kHiddenSize);
  TORCH_CHECK_EQ(residual.get_device(), inp.get_device());
  TORCH_CHECK_EQ(weight.get_device(), inp.get_device());
  TORCH_CHECK_EQ(normalized_out.get_device(), inp.get_device());
  TORCH_CHECK_EQ(residual_out.get_device(), inp.get_device());
  TORCH_CHECK(inp.is_contiguous());
  TORCH_CHECK(residual.is_contiguous());
  TORCH_CHECK(weight.is_contiguous());
  TORCH_CHECK(normalized_out.is_contiguous());
  TORCH_CHECK(residual_out.is_contiguous());

  const auto input_size = inp.numel() * inp.element_size();
  auto reg_buffer = reinterpret_cast<void*>(_reg_buffer);
  if (reg_buffer) {
    TORCH_CHECK_LE(input_size, reg_buffer_sz_bytes);
    AT_CUDA_CHECK(cudaMemcpyAsync(reg_buffer, inp.data_ptr(), input_size,
                                  cudaMemcpyDeviceToDevice, stream));
  } else {
    reg_buffer = inp.data_ptr();
  }

  const int num_tokens = static_cast<int>(inp.size(0));
  const int hidden_size = static_cast<int>(inp.size(1));
  const float epsilon_f = static_cast<float>(epsilon);

  if constexpr (kWorldSize == 4) {
    static const bool trace_enabled = sm70_profile_trace_enabled();
    static std::atomic<bool> logged_route{false};
    if (trace_enabled) {
      cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
      AT_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
      bool expected = false;
      if (capture_status == cudaStreamCaptureStatusActive &&
          logged_route.compare_exchange_strong(expected, true)) {
        std::cerr << "SM70 TP4 all_reduce_gemma_rms_norm op reached"
                  << " rank=" << fa->rank_ << " num_tokens=" << num_tokens
                  << " residual=" << scalar_type_name(residual.scalar_type())
                  << " capture=" << capture_status_name(capture_status)
                  << std::endl;
      }
    }
  }

  auto input_ptr = reinterpret_cast<half*>(reg_buffer);
  auto normalized_out_ptr = reinterpret_cast<half*>(normalized_out.data_ptr());
  auto residual_out_ptr = reinterpret_cast<float*>(residual_out.data_ptr());

  if (residual.scalar_type() == at::ScalarType::Float) {
    auto residual_ptr = reinterpret_cast<const float*>(residual.data_ptr());
    if (weight.scalar_type() == at::ScalarType::Float) {
      fa->sm70_allreduce_gemma_rms_norm<kWorldSize, float, float>(
          stream, input_ptr, residual_ptr,
          reinterpret_cast<const float*>(weight.data_ptr()), normalized_out_ptr,
          residual_out_ptr, num_tokens, hidden_size, epsilon_f);
    } else {
      fa->sm70_allreduce_gemma_rms_norm<kWorldSize, float, half>(
          stream, input_ptr, residual_ptr,
          reinterpret_cast<const half*>(weight.data_ptr()), normalized_out_ptr,
          residual_out_ptr, num_tokens, hidden_size, epsilon_f);
    }
  } else if constexpr (kWorldSize == 2) {
    auto residual_ptr = reinterpret_cast<const half*>(residual.data_ptr());
    if (weight.scalar_type() == at::ScalarType::Float) {
      fa->sm70_allreduce_gemma_rms_norm<kWorldSize, half, float>(
          stream, input_ptr, residual_ptr,
          reinterpret_cast<const float*>(weight.data_ptr()), normalized_out_ptr,
          residual_out_ptr, num_tokens, hidden_size, epsilon_f);
    } else {
      fa->sm70_allreduce_gemma_rms_norm<kWorldSize, half, half>(
          stream, input_ptr, residual_ptr,
          reinterpret_cast<const half*>(weight.data_ptr()), normalized_out_ptr,
          residual_out_ptr, num_tokens, hidden_size, epsilon_f);
    }
  }
}

}  // namespace

void sm70_tp2_all_reduce_gemma_rms_norm(
    fptr_t _fa, torch::Tensor& inp, torch::Tensor& residual,
    torch::Tensor& weight, torch::Tensor& normalized_out,
    torch::Tensor& residual_out, fptr_t _reg_buffer,
    int64_t reg_buffer_sz_bytes, double epsilon) {
  sm70_all_reduce_gemma_rms_norm_impl<2>(
      _fa, inp, residual, weight, normalized_out, residual_out, _reg_buffer,
      reg_buffer_sz_bytes, epsilon);
}

void sm70_tp4_all_reduce_gemma_rms_norm(
    fptr_t _fa, torch::Tensor& inp, torch::Tensor& residual,
    torch::Tensor& weight, torch::Tensor& normalized_out,
    torch::Tensor& residual_out, fptr_t _reg_buffer,
    int64_t reg_buffer_sz_bytes, double epsilon) {
  sm70_all_reduce_gemma_rms_norm_impl<4>(
      _fa, inp, residual, weight, normalized_out, residual_out, _reg_buffer,
      reg_buffer_sz_bytes, epsilon);
}

void all_reduce_sum2(fptr_t _fa, torch::Tensor& inp_a, torch::Tensor& inp_b,
                     torch::Tensor& out) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(inp_a));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  AT_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));

  TORCH_CHECK_EQ(inp_a.scalar_type(), inp_b.scalar_type());
  TORCH_CHECK_EQ(inp_a.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(inp_a.numel(), inp_b.numel());
  TORCH_CHECK_EQ(inp_a.numel(), out.numel());
  TORCH_CHECK(_is_weak_contiguous(inp_a));
  TORCH_CHECK(_is_weak_contiguous(inp_b));
  TORCH_CHECK(_is_weak_contiguous(out));

  static std::atomic<bool> logged_sum2_route{false};
  bool expected = false;
  if (sm70_profile_trace_enabled() &&
      logged_sum2_route.compare_exchange_strong(expected, true)) {
    std::cerr << "SM70 custom all_reduce_sum2 op reached"
              << " rank=" << fa->rank_ << " world_size=" << fa->world_size_
              << " numel=" << out.numel()
              << " dtype=" << scalar_type_name(out.scalar_type())
              << " capture=" << capture_status_name(capture_status)
              << std::endl;
  }

  switch (out.scalar_type()) {
    case at::ScalarType::Float: {
      fa->allreduce_sum2<float>(stream, reinterpret_cast<float*>(inp_a.data_ptr()),
                                reinterpret_cast<float*>(inp_b.data_ptr()),
                                reinterpret_cast<float*>(out.data_ptr()),
                                out.numel());
      break;
    }
    case at::ScalarType::Half: {
      fa->allreduce_sum2<half>(stream, reinterpret_cast<half*>(inp_a.data_ptr()),
                               reinterpret_cast<half*>(inp_b.data_ptr()),
                               reinterpret_cast<half*>(out.data_ptr()),
                               out.numel());
      break;
    }
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case at::ScalarType::BFloat16: {
      fa->allreduce_sum2<nv_bfloat16>(
          stream, reinterpret_cast<nv_bfloat16*>(inp_a.data_ptr()),
          reinterpret_cast<nv_bfloat16*>(inp_b.data_ptr()),
          reinterpret_cast<nv_bfloat16*>(out.data_ptr()), out.numel());
      break;
    }
#endif
    default:
      throw std::runtime_error(
          "custom allreduce sum2 only supports float32, float16 and bfloat16");
  }
}

void top1_argmax(fptr_t _fa, torch::Tensor& input_pair, torch::Tensor& output,
                 fptr_t _reg_buffer, int64_t reg_buffer_sz_bytes) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input_pair));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK(input_pair.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(output.scalar_type() == at::ScalarType::Long);
  TORCH_CHECK(input_pair.numel() == 2);
  TORCH_CHECK(output.numel() == 1);
  TORCH_CHECK(_is_weak_contiguous(input_pair));
  TORCH_CHECK(_is_weak_contiguous(output));

  auto input_size = input_pair.numel() * input_pair.element_size();
  auto reg_buffer = reinterpret_cast<void*>(_reg_buffer);
  if (reg_buffer) {
    TORCH_CHECK_LE(input_size, reg_buffer_sz_bytes);
    AT_CUDA_CHECK(cudaMemcpyAsync(reg_buffer, input_pair.data_ptr(), input_size,
                                  cudaMemcpyDeviceToDevice, stream));
  } else {
    reg_buffer = input_pair.data_ptr();
  }

  fa->top1_argmax(stream, reinterpret_cast<float*>(reg_buffer),
                  reinterpret_cast<int64_t*>(output.data_ptr()));
}

void tile_runtime_all_reduce(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out,
                             fptr_t _reg_buffer,
                             int64_t reg_buffer_sz_bytes,
                             int64_t tile_numel, int64_t engine_blocks,
                             int64_t compute_iters) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(inp));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(inp.numel(), out.numel());
  TORCH_CHECK(_is_weak_contiguous(inp));
  TORCH_CHECK(_is_weak_contiguous(out));
  TORCH_CHECK(tile_numel > 0);
  TORCH_CHECK(engine_blocks >= 0);
  TORCH_CHECK(compute_iters >= 0);

  auto input_size = inp.numel() * inp.element_size();
  auto reg_buffer = reinterpret_cast<void*>(_reg_buffer);
  TORCH_CHECK(reg_buffer != nullptr,
              "SM70 tile runtime prototype requires a registered staging "
              "buffer.");
  TORCH_CHECK_LE(input_size, reg_buffer_sz_bytes);

  switch (out.scalar_type()) {
    case at::ScalarType::Float: {
      fa->tile_runtime_allreduce<float>(
          stream, reinterpret_cast<const float*>(inp.data_ptr()),
          reinterpret_cast<float*>(reg_buffer),
          reinterpret_cast<float*>(out.data_ptr()), out.numel(), tile_numel,
          engine_blocks, compute_iters);
      break;
    }
    case at::ScalarType::Half: {
      fa->tile_runtime_allreduce<half>(
          stream, reinterpret_cast<const half*>(inp.data_ptr()),
          reinterpret_cast<half*>(reg_buffer),
          reinterpret_cast<half*>(out.data_ptr()), out.numel(), tile_numel,
          engine_blocks, compute_iters);
      break;
    }
    default:
      throw std::runtime_error(
          "SM70 tile runtime prototype supports float32 and float16 only");
  }
}

void tile_runtime_all_reduce_engine(fptr_t _fa, torch::Tensor& inp,
                                    torch::Tensor& out, fptr_t _reg_buffer,
                                    int64_t reg_buffer_sz_bytes,
                                    int64_t tile_numel,
                                    int64_t producer_blocks,
                                    int64_t reducer_blocks,
                                    int64_t compute_iters) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(inp));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(inp.numel(), out.numel());
  TORCH_CHECK(_is_weak_contiguous(inp));
  TORCH_CHECK(_is_weak_contiguous(out));
  TORCH_CHECK(tile_numel > 0);
  TORCH_CHECK(producer_blocks >= 0);
  TORCH_CHECK(reducer_blocks >= 0);
  TORCH_CHECK(compute_iters >= 0);

  auto input_size = inp.numel() * inp.element_size();
  auto reg_buffer = reinterpret_cast<void*>(_reg_buffer);
  TORCH_CHECK(reg_buffer != nullptr,
              "SM70 tile runtime engine requires a registered staging buffer.");
  TORCH_CHECK_LE(input_size, reg_buffer_sz_bytes);

  switch (out.scalar_type()) {
    case at::ScalarType::Float: {
      fa->tile_runtime_allreduce_engine<float>(
          stream, reinterpret_cast<const float*>(inp.data_ptr()),
          reinterpret_cast<float*>(reg_buffer),
          reinterpret_cast<float*>(out.data_ptr()), out.numel(), tile_numel,
          producer_blocks, reducer_blocks, compute_iters);
      break;
    }
    case at::ScalarType::Half: {
      fa->tile_runtime_allreduce_engine<half>(
          stream, reinterpret_cast<const half*>(inp.data_ptr()),
          reinterpret_cast<half*>(reg_buffer),
          reinterpret_cast<half*>(out.data_ptr()), out.numel(), tile_numel,
          producer_blocks, reducer_blocks, compute_iters);
      break;
    }
    default:
      throw std::runtime_error(
          "SM70 tile runtime engine supports float32 and float16 only");
  }
}

void tile_runtime_wait_reduce(fptr_t _fa, torch::Tensor& staging,
                              torch::Tensor& out, int64_t tile_numel,
                              int64_t reducer_blocks) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(staging));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK_EQ(staging.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(staging.numel(), out.numel());
  TORCH_CHECK(_is_weak_contiguous(staging));
  TORCH_CHECK(_is_weak_contiguous(out));
  TORCH_CHECK(tile_numel > 0);
  TORCH_CHECK(reducer_blocks >= 0);

  switch (out.scalar_type()) {
    case at::ScalarType::Float: {
      fa->tile_runtime_wait_reduce<float>(
          stream, reinterpret_cast<float*>(staging.data_ptr()),
          reinterpret_cast<float*>(out.data_ptr()), out.numel(), tile_numel,
          reducer_blocks);
      break;
    }
    case at::ScalarType::Half: {
      fa->tile_runtime_wait_reduce<half>(
          stream, reinterpret_cast<half*>(staging.data_ptr()),
          reinterpret_cast<half*>(out.data_ptr()), out.numel(), tile_numel,
          reducer_blocks);
      break;
    }
    default:
      throw std::runtime_error(
          "SM70 tile runtime wait-reduce supports float32 and float16 only");
  }
}

void dispose(fptr_t _fa) {
  delete reinterpret_cast<vllm::CustomAllreduce*>(_fa);
}

int64_t meta_size() { return sizeof(vllm::Signal); }

void register_buffer(fptr_t _fa, const std::vector<fptr_t>& fake_ipc_ptrs) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  TORCH_CHECK(fake_ipc_ptrs.size() == fa->world_size_);
  void* ipc_ptrs[8];
  for (int i = 0; i < fake_ipc_ptrs.size(); i++) {
    ipc_ptrs[i] = reinterpret_cast<void*>(fake_ipc_ptrs[i]);
  }
  fa->register_buffer(ipc_ptrs);
}

// Use vector<int64_t> to represent byte data for python binding compatibility.
std::tuple<std::vector<int64_t>, std::vector<int64_t>>
get_graph_buffer_ipc_meta(fptr_t _fa) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  auto [handle, offsets] = fa->get_graph_buffer_ipc_meta();
  std::vector<int64_t> bytes(handle.begin(), handle.end());
  return std::make_tuple(bytes, offsets);
}

// Use vector<int64_t> to represent byte data for python binding compatibility.
void register_graph_buffers(fptr_t _fa,
                            const std::vector<std::vector<int64_t>>& handles,
                            const std::vector<std::vector<int64_t>>& offsets) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  std::vector<std::string> bytes;
  bytes.reserve(handles.size());
  for (int i = 0; i < handles.size(); i++) {
    bytes.emplace_back(handles[i].begin(), handles[i].end());
  }
  bytes.reserve(handles.size());
  fa->register_graph_buffers(bytes, offsets);
}

std::tuple<fptr_t, torch::Tensor> allocate_shared_buffer_and_handle(
    int64_t size) {
  auto device_index = c10::cuda::current_device();
  at::DeviceGuard device_guard(at::Device(at::DeviceType::CUDA, device_index));
  void* buffer;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  AT_CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));

  // Allocate buffer
#if defined(USE_ROCM)
  // data buffers need to be "uncached" for signal on MI200
  AT_CUDA_CHECK(
      hipExtMallocWithFlags((void**)&buffer, size, hipDeviceMallocUncached));
#else
  AT_CUDA_CHECK(cudaMalloc((void**)&buffer, size));
#endif
  AT_CUDA_CHECK(cudaMemsetAsync(buffer, 0, size, stream));
  AT_CUDA_CHECK(cudaStreamSynchronize(stream));
  AT_CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));

  // Create IPC memhandle for the allocated buffer.
  // Will use it in open_mem_handle.
  auto options =
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
  auto handle =
      torch::empty({static_cast<int64_t>(sizeof(cudaIpcMemHandle_t))}, options);
  AT_CUDA_CHECK(
      cudaIpcGetMemHandle((cudaIpcMemHandle_t*)handle.data_ptr(), buffer));

  return std::make_tuple(reinterpret_cast<fptr_t>(buffer), handle);
}

fptr_t open_mem_handle(torch::Tensor& mem_handle) {
  void* ipc_ptr;
  AT_CUDA_CHECK(cudaIpcOpenMemHandle(
      (void**)&ipc_ptr, *((const cudaIpcMemHandle_t*)mem_handle.data_ptr()),
      cudaIpcMemLazyEnablePeerAccess));
  return reinterpret_cast<fptr_t>(ipc_ptr);
}

void free_shared_buffer(fptr_t buffer) {
  AT_CUDA_CHECK(cudaFree(reinterpret_cast<void*>(buffer)));
}
