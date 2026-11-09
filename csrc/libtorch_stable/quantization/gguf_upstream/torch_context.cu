// SPDX-License-Identifier: Apache-2.0
#include "torch_context.cuh"
#include "ggml_dypes.cuh"
#include "mmq.cuh"

#include <algorithm>
#include <memory>
#include <optional>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/c/shim.h>

// The upstream context's stream() helper lazily creates a private
// cudaStreamNonBlocking stream whenever it observes a null slot. The bridge
// must never let that happen: all bridge-owned work (dtype casts, quantize,
// zeroing) runs on the Torch current stream, so upstream kernels launched
// through ctx.stream() have to observe the same underlying stream or reads
// and writes race with no ordering dependency.
//
// Torch's default stream reports a null handle. Upstream treats null as
// "not created", so a raw assignment would silently keep the private-stream
// behavior. Normalize instead to CUDA's special non-null handles, which map
// to the same underlying default stream the null handle denotes.
// setup.py does not compile with --default-stream per-thread, so the legacy
// default-stream semantics apply for this build; keep both mappings in one
// place in case that flag ever changes.
static cudaStream_t normalize_borrowed_stream(cudaStream_t stream) {
  return stream != nullptr ? stream : cudaStreamLegacy;
}

cudaStream_t current_stream(int32_t device_index) {
  void* raw_stream = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(device_index, &raw_stream));
  return normalize_borrowed_stream(static_cast<cudaStream_t>(raw_stream));
}

void check_launch(const char* op_name) {
  const cudaError_t error = cudaGetLastError();
  STD_TORCH_CHECK(error == cudaSuccess, op_name,
                  ": CUDA launch failed: ", cudaGetErrorString(error));
}

bool has_weight_padding(const Tensor& W, int64_t k, int64_t type,
                        const char* op_name) {
  int64_t storage_size = 0;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_storage_size(W.get(), &storage_size));
  const size_t element_size = W.element_size();
  const size_t offset_bytes =
      static_cast<size_t>(W.storage_offset()) * element_size;
  const size_t logical_bytes = static_cast<size_t>(W.numel()) * element_size;
  if (storage_size < 0 || static_cast<uint64_t>(storage_size) < offset_bytes) {
    return false;
  }
  const size_t available_bytes =
      static_cast<size_t>(storage_size) - offset_bytes;
  return available_bytes >=
         logical_bytes + storage_padding_bytes(k, type, op_name);
}

void cast_contiguous_async(void* dst, ScalarType dst_type, const void* src,
                           ScalarType src_type, int64_t count,
                           cudaStream_t stream) {
  STD_TORCH_CHECK(dst_type != src_type,
                  "CUDA cast requires different source and target dtypes");
  if (dst_type == ScalarType::Float) {
    if (src_type == ScalarType::Half) {
      gguf_cast_async<float, half>(dst, src, count, stream);
    } else {
      STD_TORCH_CHECK(src_type == ScalarType::BFloat16,
                      "unsupported CUDA cast source dtype");
      gguf_cast_async<float, __nv_bfloat16>(dst, src, count, stream);
    }
  } else if (dst_type == ScalarType::Half) {
    if (src_type == ScalarType::Float) {
      gguf_cast_async<half, float>(dst, src, count, stream);
    } else {
      STD_TORCH_CHECK(src_type == ScalarType::BFloat16,
                      "unsupported CUDA cast source dtype");
      gguf_cast_async<half, __nv_bfloat16>(dst, src, count, stream);
    }
  } else {
    STD_TORCH_CHECK(dst_type == ScalarType::BFloat16,
                    "unsupported CUDA cast target dtype");
    if (src_type == ScalarType::Float) {
      gguf_cast_async<__nv_bfloat16, float>(dst, src, count, stream);
    } else {
      STD_TORCH_CHECK(src_type == ScalarType::Half,
                      "unsupported CUDA cast source dtype");
      gguf_cast_async<__nv_bfloat16, half>(dst, src, count, stream);
    }
  }
}

// Extra bytes handed out with every pool allocation. The upstream MMQ kernels
// assume the ggml_cuda_pool hands back chunks carved from larger aligned
// blocks, so tile loads can read a few bytes past the requested size (the
// caller-side J_max guard undercounts when src1 is a broadcast view with
// ne11 == 1, which is exactly how the bridge builds the MoE activation
// tensor). Returning exactly `size` bytes made those reads out-of-bounds,
// surfacing as NaN outputs or illegal memory accesses depending on where the
// trailing tile landed. The tail must be zeroed, not merely allocated:
// mul_mat_q reads full J-row tiles whose trailing rows lie past the logical
// data (write-back masks them out by j_max), and garbage scale bytes in that
// tail occasionally poisoned results non-deterministically.
//
// gguf_constants::kMmqTileColumnsMax mirrors mul_mat_q_switch_J's upper bound
// (its tile-column upper bound), so a tail of one row of the largest selected
// tile covers any J_best guard. This is deliberately NOT derived from
// MATRIX_ROW_PADDING (that guards the K dimension, not J). sizeof stays a
// run-time sizeof(block_q8_1_mmq): upstream static-asserts block_fp4_mmq has
// the same size, so one tail formula covers the Q8 and native-FP4 layouts.
constexpr size_t kPoolGuardTailBytes =
    static_cast<size_t>(gguf_constants::kMmqTileColumnsMax) *
    sizeof(block_q8_1_mmq);

// The pool exposes two entry points with different guard policies:
//
//  - alloc() (the ggml_cuda_pool virtual override): every upstream
//    ggml_cuda_pool_alloc goes through here, and the interface carries only a
//    byte count - no buffer purpose. Upstream callers (mmvq.cu / mmq.cu
//    quantized activations, ids_src1/ids_dst/expert_bounds, NVFP4 src1_scale,
//    stream-k tmp_fixup) have heterogeneous tile-read patterns and their own
//    J_max guard undercounts for broadcast views (ne11 == 1), so the
//    conservative zeroed tail stays. Compatibility-layer policy for the
//    pinned upstream (002a12ad); do not shrink without per-buffer proofs.
//
//  - alloc_workspace(): bridge-owned workspace allocations only (dense
//    buffers). The caller knows the exact layout and already reserves every
//    needed guard explicitly (the Q8 region includes the j_max tile guard in
//    q8_bytes), so no extra tail is appended - protection lives at the
//    correct interior offset instead of being duplicated at the block end.
//    The dense workspace must never contain ids/scale/fixup data; those are
//    allocated by upstream itself via the virtual entry, where the tail
//    still applies.
TorchScratchPool::TorchScratchPool(const Tensor& prototype)
    : prototype_(prototype) {}

void* TorchScratchPool::alloc(size_t size, size_t* actual_size) {
  return alloc_impl(std::max<size_t>(size, 1) + kPoolGuardTailBytes,
                    actual_size);
}

void* TorchScratchPool::alloc_workspace(size_t bytes, size_t* actual_size) {
  return alloc_impl(bytes, actual_size);
}

void TorchScratchPool::free(void* /*ptr*/, size_t /*size*/) {}

void* TorchScratchPool::alloc_impl(size_t bytes, size_t* actual_size) {
  const int64_t int_count = static_cast<int64_t>((bytes + 3) / 4);
  owners_.push_back(torch::stable::new_zeros(
      prototype_, {int_count}, std::optional<ScalarType>(ScalarType::Int)));
  *actual_size = static_cast<size_t>(int_count) * 4;
  return owners_.back().data_ptr();
}

// Own the device guard and GGML context for one upstream launch. The context
// owns the Torch-backed pool; its raw pointer is only a non-owning workspace
// handle while this call is alive.
UpstreamCall::UpstreamCall(const Tensor& prototype)
    : device_guard(prototype.get_device_index()),
      stream(current_stream(prototype.get_device_index())),
      context(prototype.get_device_index()) {
  const int32_t device_index = prototype.get_device_index();
  // Borrow Torch's current stream; runtime.cu does not destroy it.
  context.streams[device_index][0] = stream;
  context.curr_stream_no = 0;
  auto owner = std::make_unique<TorchScratchPool>(prototype);
  scratch_pool = owner.get();
  // GGML context owns the pool for the full lifetime of this call.
  context.pools[device_index][0] = std::move(owner);
}

ProjectionBuffers projection_buffers(const Tensor& X, int64_t output_rows,
                                     int64_t row, size_t q8_bytes,
                                     TorchScratchPool& scratch_pool,
                                     cudaStream_t stream) {
  Tensor output =
      torch::stable::new_empty(X, {output_rows, row}, X.scalar_type());
  const bool cast = X.scalar_type() != ScalarType::Float;
  const auto align = [](size_t bytes) { return (bytes + 255) / 256 * 256; };
  const size_t input_offset = align(q8_bytes);
  const size_t input_bytes = cast ? X.numel() * sizeof(float) : 0;
  const size_t result_offset = input_offset + align(input_bytes);
  const size_t result_bytes = cast ? output_rows * row * sizeof(float) : 0;
  size_t actual_size = 0;
  // fp32 input with no bridge-owned quantized region needs no scratch at all:
  // input comes straight from X and the result goes straight into the output
  // tensor. Skip the allocation instead of handing back a pointer that is
  // only used for pointer arithmetic that never dereferences.
  const bool need_scratch = cast || q8_bytes > 0;
  // All temporary regions have one per-call Torch owner. The workspace entry
  // adds no tail: every guard the layout needs is already part of q8_bytes.
  // 256-byte alignment preserves the upstream quantizer's vector loads and
  // graph/stream safety.
  char* scratch = nullptr;
  if (need_scratch) {
    scratch = static_cast<char*>(scratch_pool.alloc_workspace(
        result_offset + result_bytes, &actual_size));
  }
  const float* input = cast ? reinterpret_cast<float*>(scratch + input_offset)
                            : static_cast<const float*>(X.data_ptr());
  float* result = cast ? reinterpret_cast<float*>(scratch + result_offset)
                       : static_cast<float*>(output.data_ptr());
  if (cast) {
    cast_contiguous_async(scratch + input_offset, ScalarType::Float,
                          X.data_ptr(), X.scalar_type(), X.numel(), stream);
  }
  return {output, scratch, input, result};
}

Tensor finish_output(const ProjectionBuffers& buffers, cudaStream_t stream) {
  const auto dtype = buffers.output.scalar_type();
  if (dtype != ScalarType::Float) {
    cast_contiguous_async(buffers.output.data_ptr(), dtype, buffers.result,
                          ScalarType::Float, buffers.output.numel(), stream);
  }
  check_launch("upstream projection output");
  return buffers.output;
}
