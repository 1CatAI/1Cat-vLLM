// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/c/shim.h>
#include "ggml_constants.cuh"

using torch::headeronly::ScalarType;
using torch::stable::Tensor;

// 1. Packed-weight/storage contract. GGML constants and the type catalog
// live in ggml_constants.cuh, which is also used by runtime.cu.
namespace {
bool upstream_mmq_type_supported(int64_t type) {
  switch (type) {
#define GGUF_TYPE_CASE(value) case value:
#define GGUF_MMQ_TYPE_CASE(type, block, cpp_type, name, quantized, mmq) \
  GGUF_IF_MMQ(mmq, GGUF_TYPE_CASE, type)
    GGUF_GGML_TYPE_TRAITS(GGUF_MMQ_TYPE_CASE)
#undef GGUF_MMQ_TYPE_CASE
#undef GGUF_TYPE_CASE
    return true;
    default:
      return false;
  }
}

bool upstream_mmvq_type_supported(int64_t type) {
  return upstream_mmq_type_supported(type) || type == GGML_TYPE_IQ1_M;
}

bool is_upstream_weight_type(int64_t type) {
  // Ternary storage has dequantization/BLAS support, even though it has no
  // upstream MMQ template instance. Admission must follow the operation.
  return upstream_mmvq_type_supported(type) || type == GGML_TYPE_TQ1_0 ||
         type == GGML_TYPE_TQ2_0;
}

bool is_upstream_float_type(int64_t type) {
  return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 ||
         type == GGML_TYPE_BF16;
}

bool float_type_matches(const Tensor& W, int64_t type) {
  return (type == GGML_TYPE_F32 && W.scalar_type() == ScalarType::Float) ||
         (type == GGML_TYPE_F16 && W.scalar_type() == ScalarType::Half) ||
         (type == GGML_TYPE_BF16 && W.scalar_type() == ScalarType::BFloat16);
}

int64_t block_size_for_type(int64_t type, const char* op_name) {
  STD_TORCH_CHECK(is_upstream_weight_type(type), op_name,
                  ": unsupported upstream quantization type: ", type);
  return ggml_blck_size(static_cast<ggml_type>(type));
}

size_t type_size_for_type(int64_t type, const char* op_name) {
  STD_TORCH_CHECK(is_upstream_weight_type(type), op_name,
                  ": unsupported upstream quantization type: ", type);
  return ggml_type_size(static_cast<ggml_type>(type));
}

size_t storage_padding_bytes(int64_t k, int64_t type, const char* op_name) {
  const int64_t remainder = k % gguf_constants::kMatrixRowPadding;
  if (remainder == 0) {
    return 0;
  }
  const int64_t missing = gguf_constants::kMatrixRowPadding - remainder;
  const int64_t block_size = block_size_for_type(type, op_name);
  const size_t type_size = type_size_for_type(type, op_name);
  STD_TORCH_CHECK(missing % block_size == 0, op_name,
                  ": upstream row padding is not aligned to a quantization "
                  "block");
  return static_cast<size_t>(missing / block_size) * type_size;
}

// Shared body of the packed-row -> logical-k derivation. packed_row_bytes is
// the per-row packed byte count (W.size(1) for dense, W.size(2) for MoE); the
// wrappers keep their operation-specific error text.
int64_t logical_k_from_packed_row_bytes(int64_t packed_row_bytes, int64_t type,
                                        const char* op_name,
                                        const char* row_desc) {
  const size_t type_size = type_size_for_type(type, op_name);
  const int64_t block_size = block_size_for_type(type, op_name);
  STD_TORCH_CHECK(packed_row_bytes > 0 && packed_row_bytes % type_size == 0,
                  op_name, ": packed ", row_desc,
                  " size is not a multiple of the quantization type size");
  return packed_row_bytes / static_cast<int64_t>(type_size) * block_size;
}

int64_t logical_k_from_weight(const Tensor& W, int64_t type,
                              const char* op_name) {
  return logical_k_from_packed_row_bytes(W.size(1), type, op_name, "row");
}

int64_t padded_k(int64_t k) {
  return (k + gguf_constants::kMatrixRowPadding - 1) /
         gguf_constants::kMatrixRowPadding * gguf_constants::kMatrixRowPadding;
}

}  // namespace

// 2. CUDA dtype conversion on the caller's stream.
// Dense bridge tensors are contiguous and the stream/device are already set.
// Avoid the general aten::to dispatcher for these simple linear conversions.
// Quantization and matrix multiplication stay in the unmodified upstream TUs.
template <typename Dst, typename Src>
static __global__ void gguf_cast(Dst* dst, const Src* src, int64_t count) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    dst[i] = static_cast<Dst>(static_cast<float>(src[i]));
  }
}

template <typename Dst, typename Src>
static void gguf_cast_async(void* dst, const void* src, int64_t count,
                            cudaStream_t stream) {
  if (count > 0) {
    gguf_cast<<<(count + 255) / 256, 256, 0, stream>>>(
        static_cast<Dst*>(dst), static_cast<const Src*>(src), count);
  }
}

// 3. Non-owning GGML descriptors over Torch and bridge buffers.
namespace {
ggml_tensor make_quant_tensor(const Tensor& W, int64_t type, int64_t k,
                              int64_t row) {
  ggml_tensor tensor{};
  tensor.type = static_cast<ggml_type>(type);
  tensor.ne[0] = k;
  tensor.ne[1] = row;
  tensor.ne[2] = 1;
  tensor.ne[3] = 1;
  tensor.nb[0] = type_size_for_type(type, "upstream MMVQ");
  tensor.nb[1] = static_cast<size_t>(W.size(1));
  tensor.nb[2] = tensor.nb[1] * static_cast<size_t>(row);
  tensor.nb[3] = tensor.nb[2];
  tensor.data = W.data_ptr();
  return tensor;
}

ggml_tensor make_float_weight_tensor(const Tensor& W, int64_t type) {
  ggml_tensor tensor{};
  tensor.type = static_cast<ggml_type>(type);
  tensor.ne[0] = W.size(1);
  tensor.ne[1] = W.size(0);
  tensor.ne[2] = 1;
  tensor.ne[3] = 1;
  tensor.nb[0] = static_cast<size_t>(W.stride(1)) * W.element_size();
  tensor.nb[1] = static_cast<size_t>(W.stride(0)) * W.element_size();
  tensor.nb[2] = tensor.nb[1] * static_cast<size_t>(W.size(0));
  tensor.nb[3] = tensor.nb[2];
  tensor.data = W.data_ptr();
  return tensor;
}

// Contiguous two-dimensional F32 descriptor shared by the activation (ne[0]
// = row length) and output (ne[0] = row count) constructions; the thin
// wrappers keep their call-site semantics readable.
ggml_tensor make_f32_tensor_2d(const void* data, int64_t ne0, int64_t ne1) {
  ggml_tensor tensor{};
  tensor.type = GGML_TYPE_F32;
  tensor.ne[0] = ne0;
  tensor.ne[1] = ne1;
  tensor.ne[2] = 1;
  tensor.ne[3] = 1;
  tensor.nb[0] = sizeof(float);
  tensor.nb[1] = static_cast<size_t>(ne0) * sizeof(float);
  tensor.nb[2] = tensor.nb[1] * static_cast<size_t>(ne1);
  tensor.nb[3] = tensor.nb[2];
  tensor.data = const_cast<void*>(data);
  return tensor;
}

ggml_tensor make_moe_weight_tensor(const Tensor& W, int64_t type, int64_t k,
                                   int64_t row) {
  ggml_tensor tensor{};
  tensor.type = static_cast<ggml_type>(type);
  tensor.ne[0] = k;
  tensor.ne[1] = row;
  tensor.ne[2] = W.size(0);
  tensor.ne[3] = 1;
  tensor.nb[0] = type_size_for_type(type, "upstream MoE");
  tensor.nb[1] = static_cast<size_t>(W.size(2));
  tensor.nb[2] = tensor.nb[1] * static_cast<size_t>(W.size(1));
  tensor.nb[3] = tensor.nb[2] * static_cast<size_t>(W.size(0));
  tensor.data = W.data_ptr();
  return tensor;
}

ggml_tensor make_moe_float_weight_tensor(const Tensor& W, int64_t type) {
  ggml_tensor tensor{};
  tensor.type = static_cast<ggml_type>(type);
  tensor.ne[0] = W.size(2);
  tensor.ne[1] = W.size(1);
  tensor.ne[2] = W.size(0);
  tensor.ne[3] = 1;
  tensor.nb[0] = static_cast<size_t>(W.stride(2)) * W.element_size();
  tensor.nb[1] = static_cast<size_t>(W.stride(1)) * W.element_size();
  tensor.nb[2] = static_cast<size_t>(W.stride(0)) * W.element_size();
  tensor.nb[3] = tensor.nb[2] * static_cast<size_t>(W.size(0));
  tensor.data = W.data_ptr();
  return tensor;
}

ggml_tensor make_moe_input_tensor(const float* data, int64_t k,
                                  int64_t tokens) {
  ggml_tensor tensor{};
  tensor.type = GGML_TYPE_F32;
  tensor.ne[0] = k;
  tensor.ne[1] = 1;
  tensor.ne[2] = tokens;
  tensor.ne[3] = 1;
  tensor.nb[0] = sizeof(float);
  tensor.nb[1] = static_cast<size_t>(k) * sizeof(float);
  tensor.nb[2] = tensor.nb[1];
  tensor.nb[3] = tensor.nb[2] * static_cast<size_t>(tokens);
  tensor.data = const_cast<float*>(data);
  return tensor;
}

ggml_tensor make_moe_ids_tensor(const int32_t* data, int64_t top_k,
                                int64_t tokens) {
  ggml_tensor tensor{};
  tensor.type = GGML_TYPE_I32;
  tensor.ne[0] = top_k;
  tensor.ne[1] = tokens;
  tensor.ne[2] = 1;
  tensor.ne[3] = 1;
  tensor.nb[0] = sizeof(int32_t);
  tensor.nb[1] = static_cast<size_t>(top_k) * sizeof(int32_t);
  tensor.nb[2] = tensor.nb[1] * static_cast<size_t>(tokens);
  tensor.nb[3] = tensor.nb[2];
  tensor.data = const_cast<int32_t*>(data);
  return tensor;
}

ggml_tensor make_moe_output_tensor(float* data, int64_t row, int64_t top_k,
                                   int64_t tokens) {
  ggml_tensor tensor{};
  tensor.type = GGML_TYPE_F32;
  tensor.ne[0] = row;
  tensor.ne[1] = top_k;
  tensor.ne[2] = tokens;
  tensor.ne[3] = 1;
  tensor.nb[0] = sizeof(float);
  tensor.nb[1] = static_cast<size_t>(row) * sizeof(float);
  tensor.nb[2] = tensor.nb[1] * static_cast<size_t>(top_k);
  tensor.nb[3] = tensor.nb[2] * static_cast<size_t>(tokens);
  tensor.data = data;
  return tensor;
}

}  // namespace
