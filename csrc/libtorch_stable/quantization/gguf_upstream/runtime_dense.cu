// SPDX-License-Identifier: Apache-2.0
#include "ggml_dypes.cuh"
#include "torch_context.cuh"
#include "mmq.cuh"
#include "mmvq.cuh"
#include "mmvf.cuh"
#include "mmf.cuh"
#include "quantize.cuh"

#include <climits>
#include <limits>
#include <torch/csrc/stable/ops.h>

using torch::headeronly::ScalarType;

Tensor run_upstream_blas(const Tensor& W, const Tensor& X, int64_t type,
                         int64_t row, int64_t k);

namespace {
constexpr int64_t kDenseMmvf = 1;
constexpr int64_t kDenseMmf = 2;
constexpr int64_t kDenseMmvq = 4;
constexpr int64_t kDenseMmq = 8;
constexpr int64_t kDenseBlas = 16;

void check_dense_inputs(const Tensor& W, const Tensor& X, int64_t row,
                        const char* op_name) {
  STD_TORCH_CHECK(W.is_cuda() && X.is_cuda(), op_name,
                  ": W and X must be CUDA tensors");
  STD_TORCH_CHECK(W.get_device_index() == X.get_device_index(), op_name,
                  ": W and X must be on the same CUDA device");
  STD_TORCH_CHECK(W.dim() == 2 && X.dim() == 2, op_name,
                  ": W and X must be rank-2 tensors");
  STD_TORCH_CHECK(W.is_contiguous() && X.is_contiguous(), op_name,
                  ": W and X must be contiguous");
  STD_TORCH_CHECK(X.scalar_type() == ScalarType::Float ||
                      X.scalar_type() == ScalarType::Half ||
                      X.scalar_type() == ScalarType::BFloat16,
                  op_name, ": X must have dtype fp32, fp16, or bf16");
  STD_TORCH_CHECK(row > 0 && row <= W.size(0), op_name,
                  ": row must be in (0, W.size(0)]");
}

// Dense-only dispatch into the upstream MMQ template instances.
// Explicit instances are emitted by the upstream template-instances sources;
// this wrapper does not copy or specialize the upstream kernel body.
void ggml_upstream_mul_mat_q(ggml_backend_cuda_context& context,
                             const mmq_args& args, cudaStream_t stream) {
  switch (args.type_x) {
#define GGUF_MMQ_INSTANTIATE(type)               \
  case type:                                     \
    mul_mat_q_case<type>(context, args, stream); \
    break;
#define GGUF_MMQ_CASE(type, block, cpp_type, name, quantized, mmq) \
  GGUF_IF_MMQ(mmq, GGUF_MMQ_INSTANTIATE, type)
    GGUF_GGML_TYPE_TRAITS(GGUF_MMQ_CASE)
#undef GGUF_MMQ_CASE
#undef GGUF_MMQ_INSTANTIATE
    default:
      GGML_ABORT("unsupported upstream MMQ type");
  }
}

bool mmq_has_launch_config(ggml_type type, int cc, int64_t batch, int64_t row,
                           size_t smpbo) {
  if (ggml_cuda_mmq_get_J_max(type, row % 128 != 0, cc, batch) <= 0) {
    return false;
  }
  for (int j = gguf_constants::kMmqTileStep;
       j <= gguf_constants::kMmqTileColumnsMax;
       j += gguf_constants::kMmqTileStep) {
    const auto config = ggml_cuda_mmq_get_config(type, j, row % 128 != 0, cc);
    if (config.type != GGML_TYPE_COUNT &&
        mmq_get_nbytes_shared(config, cc) <= smpbo) {
      return true;
    }
  }
  return false;
}

Tensor run_upstream_float(const Tensor& W, const Tensor& X, int64_t type,
                          int64_t row, int64_t route) {
  UpstreamCall call(X);
  const ProjectionBuffers buffers =
      projection_buffers(X, X.size(0), row, 0, *call.scratch_pool, call.stream);
  ggml_tensor src0 = make_float_weight_tensor(W, type);
  ggml_tensor src1 = make_f32_tensor_2d(buffers.input, X.size(1), X.size(0));
  ggml_tensor dst = make_f32_tensor_2d(buffers.result, row, X.size(0));
  if (route == kDenseMmvf) {
    ggml_cuda_mul_mat_vec_f(call.context, &src0, &src1, nullptr, &dst);
  } else {
    ggml_cuda_mul_mat_f(call.context, &src0, &src1, nullptr, &dst);
  }
  check_launch(route == kDenseMmvf ? "upstream MMVF" : "upstream MMF");
  return finish_output(buffers, call.stream);
}

Tensor run_upstream_mmvq(const Tensor& W, const Tensor& X, int64_t type,
                         int64_t row, int64_t k) {
  UpstreamCall call(X);
  const cudaStream_t stream = call.stream;
  const int64_t batch = X.size(0);
  const int64_t k_padded = padded_k(k);

  const size_t q8_bytes = static_cast<size_t>(batch) *
                          static_cast<size_t>(k_padded) * sizeof(block_q8_1) /
                          QK8_1;
  const ProjectionBuffers buffers = projection_buffers(
      X, X.size(0), row, q8_bytes, *call.scratch_pool, stream);
  void* q8_data = buffers.q8;

  quantize_row_q8_1_cuda(buffers.input, nullptr, q8_data,
                         static_cast<ggml_type>(type), k, k, 0, 0, k_padded,
                         batch, 1, 1, stream);

  ggml_tensor src0 = make_quant_tensor(W, type, k, row);
  ggml_tensor src1 = make_f32_tensor_2d(buffers.input, k_padded, batch);
  ggml_tensor dst = make_f32_tensor_2d(buffers.result, row, batch);
  ggml_cuda_op_mul_mat_vec_q(call.context, &src0, &src1, &dst,
                             static_cast<const char*>(W.data_ptr()),
                             buffers.input, static_cast<const char*>(q8_data),
                             buffers.result, 0, row, batch, k_padded, stream);
  check_launch("upstream MMVQ");
  return finish_output(buffers, stream);
}

Tensor run_upstream_mmq(const Tensor& W, const Tensor& X, int64_t type,
                        int64_t row, int64_t k) {
  const int32_t device_index = X.get_device_index();
  const int64_t batch = X.size(0);
  const int64_t k_padded = padded_k(k);
  const int cc = ggml_cuda_info().devices[device_index].cc;
  const bool fallback = row % 128 != 0;
  const int j_max = ggml_cuda_mmq_get_J_max(static_cast<ggml_type>(type),
                                            fallback, cc, batch);
  STD_TORCH_CHECK(j_max > 0, "ggml_dense_mmq: no upstream MMQ configuration");
  STD_TORCH_CHECK(
      mmq_has_launch_config(static_cast<ggml_type>(type), cc, batch, row,
                            ggml_cuda_info().devices[device_index].smpbo),
      "ggml_dense_mmq: no launchable upstream MMQ tile");
  UpstreamCall call(X);
  const cudaStream_t stream = call.stream;

  // Native FP4 (Blackwell MMA path): upstream swaps the Q8_1_MMQ activation
  // format for block_fp4_mmq and needs a separate per-column scale buffer for
  // NVFP4, with different block sizes, strides and kernel-side ne_block. The
  // bridge's merged Q8 workspace cannot express that layout, so for these
  // types hand the whole operator to the upstream wrapper: build plain F32
  // descriptors for src1/dst and let ggml_cuda_mul_mat_q quantize, allocate
  // (via the installed TorchScratchPool), scale and launch on ctx.stream()
  // itself. Describing src1 with the logical k (never k_padded, which would
  // claim padding we did not allocate) keeps upstream's own
  // MATRIX_ROW_PADDING handling authoritative.
  const bool native_fp4 = blackwell_mma_available(cc) &&
                          (type == GGML_TYPE_MXFP4 || type == GGML_TYPE_NVFP4);
  if (native_fp4) {
    // No bridge quantized workspace is needed; output conversion still runs
    // through the shared dense-buffers conversion path (zero q8 region).
    const ProjectionBuffers buffers =
        projection_buffers(X, X.size(0), row, 0, *call.scratch_pool, stream);
    ggml_tensor src0 = make_quant_tensor(W, type, k, row);
    ggml_tensor src1 = make_f32_tensor_2d(buffers.input, k, batch);
    ggml_tensor dst = make_f32_tensor_2d(buffers.result, row, batch);
    ggml_cuda_mul_mat_q(call.context, &src0, &src1, /*ids=*/nullptr, &dst);
    check_launch("upstream MMQ (native FP4)");
    return finish_output(buffers, stream);
  }

  const size_t q8_bytes = static_cast<size_t>(batch) *
                              static_cast<size_t>(k_padded) *
                              sizeof(block_q8_1_mmq) / QK8_1_MMQ +
                          static_cast<size_t>(j_max) * sizeof(block_q8_1_mmq);
  const ProjectionBuffers buffers = projection_buffers(
      X, X.size(0), row, q8_bytes, *call.scratch_pool, stream);
  void* q8_data = buffers.q8;

  quantize_mmq_q8_1_cuda(buffers.input, nullptr, q8_data,
                         static_cast<ggml_type>(type), k, k, 0, 0, k_padded,
                         batch, 1, 1, stream);

  // Named-field initialization (C++17: no designated initializers) so an
  // upstream mmq_args field addition fails to compile here instead of
  // silently shifting all subsequent positional values. Every field is
  // annotated with its source; stride_channel/sample entries use 1 because
  // the bridge builds a single-channel, single-sample dense descriptor.
  mmq_args args{};
  args.x = static_cast<const char*>(W.data_ptr());
  args.type_x = static_cast<ggml_type>(type);
  args.y = static_cast<const int*>(q8_data);
  args.ids_dst = nullptr;        // dense path: no expert routing
  args.expert_bounds = nullptr;  // dense path: no expert bounds
  args.dst = buffers.result;
  args.y_scale = nullptr;  // Q8 activation path: no NVFP4 scale
  args.ncols_x = k;
  args.nrows_x = row;
  args.ncols_dst = batch;
  args.stride_row_x = static_cast<int64_t>(
      W.size(1) / type_size_for_type(type, "upstream MMQ"));
  args.ncols_y = batch;
  args.nrows_dst = row;
  args.nchannels_x = 1;
  args.nchannels_y = 1;
  args.stride_channel_x = 1;
  args.stride_channel_y = 1;
  args.stride_channel_dst = 1;
  args.nsamples_x = 1;
  args.nsamples_y = 1;
  args.stride_sample_x = 1;
  args.stride_sample_y = 1;
  args.stride_sample_dst = 1;
  args.ncols_max = batch;
  args.ncols_opt = batch;
  ggml_upstream_mul_mat_q(call.context, args, stream);
  check_launch("upstream MMQ");
  return finish_output(buffers, stream);
}

int64_t upstream_dense_capabilities(const Tensor& W, const Tensor& X,
                                    int64_t type, int64_t row) {
  if (!W.is_cuda() || !X.is_cuda() ||
      W.get_device_index() != X.get_device_index() || W.dim() != 2 ||
      X.dim() != 2 || !W.is_contiguous() || !X.is_contiguous() || row <= 0 ||
      row > W.size(0) || X.size(0) <= 0 || X.size(0) > INT_MAX ||
      (X.scalar_type() != ScalarType::Float &&
       X.scalar_type() != ScalarType::Half &&
       X.scalar_type() != ScalarType::BFloat16)) {
    return 0;
  }
  const DeviceGuard device_guard(X.get_device_index());
  const auto& device = ggml_cuda_info().devices[X.get_device_index()];
  const auto quant_type = static_cast<ggml_type>(type);
  const int64_t batch = X.size(0);
  if (is_upstream_float_type(type)) {
    if (!float_type_matches(W, type) || row != W.size(0) ||
        W.size(1) != X.size(1) ||
        reinterpret_cast<uintptr_t>(W.data_ptr()) % (2 * W.element_size()) !=
            0) {
      return 0;
    }
    const ggml_tensor weight = make_float_weight_tensor(W, type);
    int64_t caps = 0;
    if (ggml_cuda_should_use_mmvf(quant_type, device.cc, weight.ne, weight.nb,
                                  batch)) {
      caps |= kDenseMmvf;
    }
    if (ggml_cuda_should_use_mmf(quant_type, device.cc, device.warp_size,
                                 weight.ne, weight.nb, batch, false)) {
      caps |= kDenseMmf;
    }
    return caps;
  }
  if (!is_upstream_weight_type(type) || W.element_size() != 1) {
    return 0;
  }
  const int64_t k = logical_k_from_weight(W, type, "dense capabilities");
  if (X.size(1) != k) {
    return 0;
  }
  int64_t caps = row <= INT_MAX && k <= INT_MAX &&
                         (type != GGML_TYPE_MXFP4 || k % 256 == 0)
                     ? kDenseBlas
                     : 0;
  const bool padded = has_weight_padding(W, k, type, "dense capabilities");
  if (upstream_mmvq_type_supported(type) && padded &&
      batch <= MMVQ_MAX_BATCH_SIZE &&
      ggml_cuda_should_use_mmvq(quant_type, device.cc, batch)) {
    caps |= kDenseMmvq;
  }
  if (padded && upstream_mmq_type_supported(type) &&
      ggml_cuda_should_use_mmq(quant_type, device.cc, batch, 0) &&
      mmq_has_launch_config(quant_type, device.cc, batch, row, device.smpbo)) {
    caps |= kDenseMmq;
  }
  return caps;
}

Tensor run_selected_dense(const Tensor& W, const Tensor& X, int64_t type,
                          int64_t row, int64_t route, const char* op_name) {
  if (route == kDenseMmvf || route == kDenseMmf) {
    STD_TORCH_CHECK(float_type_matches(W, type) && row == W.size(0) &&
                        W.size(1) == X.size(1) &&
                        reinterpret_cast<uintptr_t>(W.data_ptr()) %
                                (2 * W.element_size()) ==
                            0,
                    op_name, ": W type, row, K, or alignment mismatch");
    return run_upstream_float(W, X, type, row, route);
  }
  STD_TORCH_CHECK(
      route == kDenseMmvq || route == kDenseMmq || route == kDenseBlas, op_name,
      ": invalid kernel route ", route);
  STD_TORCH_CHECK(is_upstream_weight_type(type) && W.element_size() == 1,
                  op_name, ": expected a supported packed weight: ", type);
  const int64_t k = logical_k_from_weight(W, type, op_name);
  STD_TORCH_CHECK(X.size(1) == k, op_name, ": X K dimension mismatch");
  if (route == kDenseMmvq) {
    STD_TORCH_CHECK(upstream_mmvq_type_supported(type) &&
                        X.size(0) <= MMVQ_MAX_BATCH_SIZE &&
                        has_weight_padding(W, k, type, op_name),
                    op_name, ": batch or weight padding invalid");
    return run_upstream_mmvq(W, X, type, row, k);
  }
  if (route == kDenseMmq) {
    STD_TORCH_CHECK(upstream_mmq_type_supported(type) &&
                        has_weight_padding(W, k, type, op_name),
                    op_name, ": type or weight padding invalid");
    return run_upstream_mmq(W, X, type, row, k);
  }
  return run_upstream_blas(W, X, type, row, k);
}

Tensor run_explicit_dense(Tensor W, Tensor X, int64_t type, int64_t row,
                          int64_t route, const char* name) {
  check_dense_inputs(W, X, row, name);
  if (X.size(0) == 0) {
    return torch::stable::new_empty(X, {0, row}, X.scalar_type());
  }
  return run_selected_dense(W, X, type, row, route, name);
}
}  // namespace

bool ggml_should_use_mmvq(int64_t type, int64_t cc, int64_t batch) {
  // No device lookup here: Python supplies the tensor device's capability,
  // and policy tests can exercise every architecture without that hardware.
  return upstream_mmvq_type_supported(type) && batch > 0 &&
         batch <= MMVQ_MAX_BATCH_SIZE && cc > 0 &&
         cc <= std::numeric_limits<int>::max() &&
         ggml_cuda_should_use_mmvq(static_cast<ggml_type>(type),
                                   static_cast<int>(cc), batch);
}

int64_t ggml_dense_upstream_capabilities(Tensor W, Tensor X, int64_t type,
                                         int64_t row) {
  return upstream_dense_capabilities(W, X, type, row);
}

Tensor ggml_dense_mmvq(Tensor W, Tensor X, int64_t type, int64_t row) {
  return run_explicit_dense(W, X, type, row, kDenseMmvq, "ggml_dense_mmvq");
}
Tensor ggml_dense_mmq(Tensor W, Tensor X, int64_t type, int64_t row) {
  return run_explicit_dense(W, X, type, row, kDenseMmq, "ggml_dense_mmq");
}
Tensor ggml_dense_mmvf(Tensor W, Tensor X, int64_t type, int64_t row) {
  return run_explicit_dense(W, X, type, row, kDenseMmvf, "ggml_dense_mmvf");
}
Tensor ggml_dense_mmf(Tensor W, Tensor X, int64_t type, int64_t row) {
  return run_explicit_dense(W, X, type, row, kDenseMmf, "ggml_dense_mmf");
}
Tensor ggml_dense_blas(Tensor W, Tensor X, int64_t type, int64_t row) {
  return run_explicit_dense(W, X, type, row, kDenseBlas, "ggml_dense_blas");
}
