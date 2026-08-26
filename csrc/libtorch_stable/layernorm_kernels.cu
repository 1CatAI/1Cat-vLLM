#include <numeric>

#ifndef USE_ROCM
  #include <atomic>
  #include <cuda_bf16.h>
  #include <iostream>
#endif

#include "torch_utils.h"

#include "../cub_helpers.h"
#include "../core/batch_invariant.hpp"
#include "../type_convert.cuh"
#include "dispatch_utils.h"
#include "quantization/vectorization_utils.cuh"

namespace vllm {

// TODO(woosuk): Further optimize this kernel.
template <typename scalar_t, int VEC_SIZE, int NUM_DIMS>
__global__ void rms_norm_kernel(
    scalar_t* __restrict__ out,           // [..., hidden_size]
    const scalar_t* __restrict__ input,   // [..., hidden_size]
    const int64_t input_stride_d2,        // input.stride(-2)
    const int64_t input_stride_d3,        // input.stride(-3)
    const int64_t input_stride_d4,        // input.stride(-4)
    const int64_t input_shape_d2,         // input.size(-2)
    const int64_t input_shape_d3,         // input.size(-3)
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;
  const scalar_t* input_row;
  if constexpr (NUM_DIMS == 2) {
    // 2D for layernorm normal case [batch_size, hidden]
    input_row = input + blockIdx.x * input_stride_d2;
  } else if constexpr (NUM_DIMS == 3) {
    // 3D for q/k norm [batch_size, num_heads, head_size]
    int batch_idx = blockIdx.x / input_shape_d2;
    int head_idx = blockIdx.x % input_shape_d2;
    input_row =
        input + batch_idx * input_stride_d3 + head_idx * input_stride_d2;
  } else if constexpr (NUM_DIMS == 4) {
    // 4D for transformers model_impl qk norm [batch, seq, head, head_dim]
    int batch_idx = blockIdx.x / (input_shape_d3 * input_shape_d2);
    int remaining = blockIdx.x % (input_shape_d3 * input_shape_d2);
    int seq_idx = remaining / input_shape_d2;
    int head_idx = remaining % input_shape_d2;
    input_row = input + batch_idx * input_stride_d4 +
                seq_idx * input_stride_d3 + head_idx * input_stride_d2;
  }

  auto vec_op = [&variance](const vec_n_t<scalar_t, VEC_SIZE>& vec) {
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      float x = static_cast<float>(vec.val[i]);
      variance += x * x;
    }
  };
  auto scalar_op = [&variance](const scalar_t& val) {
    float x = static_cast<float>(val);
    variance += x * x;
  };
  vllm::vectorize_read_with_alignment<VEC_SIZE>(
      input_row, hidden_size, threadIdx.x, blockDim.x, vec_op, scalar_op);

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  scalar_t* out_row = out + blockIdx.x * hidden_size;
  auto* v_in = reinterpret_cast<const vec_n_t<scalar_t, VEC_SIZE>*>(input_row);
  auto* v_w = reinterpret_cast<const vec_n_t<scalar_t, VEC_SIZE>*>(weight);
  auto* v_out = reinterpret_cast<vec_n_t<scalar_t, VEC_SIZE>*>(out_row);
  for (int i = threadIdx.x; i < hidden_size / VEC_SIZE; i += blockDim.x) {
    vec_n_t<scalar_t, VEC_SIZE> dst;
    vec_n_t<scalar_t, VEC_SIZE> src1 = v_in[i];
    vec_n_t<scalar_t, VEC_SIZE> src2 = v_w[i];
#pragma unroll
    for (int j = 0; j < VEC_SIZE; j++) {
      float x = static_cast<float>(src1.val[j]);
      float w = static_cast<float>(src2.val[j]);
      dst.val[j] = static_cast<scalar_t>(x * s_variance * w);
    }
    v_out[i] = dst;
  }
}

/* Function specialization in the case of FP16/BF16 tensors.
   Additional optimizations we can make in this case are
   packed and vectorized operations, which help with the
   memory latency bottleneck. */
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width > 0) && _typeConvert<scalar_t>::exists>
fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,  // [..., hidden_size]
    const int64_t input_stride,
    scalar_t* __restrict__ residual,      // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size) {
  // Sanity checks on our vector struct and type-punned pointer arithmetic
  static_assert(std::is_pod_v<_f16Vec<scalar_t, width>>);
  static_assert(sizeof(_f16Vec<scalar_t, width>) == sizeof(scalar_t) * width);

  const int vec_hidden_size = hidden_size / width;
  const int64_t vec_input_stride = input_stride / width;
  __shared__ float s_variance;
  float variance = 0.0f;
  /* These and the argument pointers are all declared `restrict` as they are
     not aliased in practice. Argument pointers should not be dereferenced
     in this kernel as that would be undefined behavior */
  auto* __restrict__ input_v =
      reinterpret_cast<_f16Vec<scalar_t, width>*>(input);
  auto* __restrict__ residual_v =
      reinterpret_cast<_f16Vec<scalar_t, width>*>(residual);
  auto* __restrict__ weight_v =
      reinterpret_cast<const _f16Vec<scalar_t, width>*>(weight);

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    int64_t strided_id = blockIdx.x * vec_input_stride + idx;
    _f16Vec<scalar_t, width> temp = input_v[strided_id];
    temp += residual_v[id];
    variance += temp.sum_squares();
    residual_v[id] = temp;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    int64_t strided_id = blockIdx.x * vec_input_stride + idx;
    _f16Vec<scalar_t, width> res = residual_v[id];
    _f16Vec<scalar_t, width> w = weight_v[idx];
    _f16Vec<scalar_t, width> out;
    using Converter = _typeConvert<scalar_t>;
#pragma unroll
    for (int j = 0; j < width; ++j) {
      float x = Converter::convert(res.data[j]);
      float wf = Converter::convert(w.data[j]);
      out.data[j] = Converter::convert(x * s_variance * wf);
    }
    input_v[strided_id] = out;
  }
}

/* Generic fused_add_rms_norm_kernel
   The width field is not used here but necessary for other specializations.
 */
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width == 0) || !_typeConvert<scalar_t>::exists>
fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,  // [..., hidden_size]
    const int64_t input_stride,
    scalar_t* __restrict__ residual,      // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    scalar_t z = input[blockIdx.x * input_stride + idx];
    z += residual[blockIdx.x * hidden_size + idx];
    float x = (float)z;
    variance += x * x;
    residual[blockIdx.x * hidden_size + idx] = z;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)residual[blockIdx.x * hidden_size + idx];
    float w = (float)weight[idx];
    input[blockIdx.x * input_stride + idx] = (scalar_t)(x * s_variance * w);
  }
}

#ifndef USE_ROCM
constexpr int kSm70GemmaHiddenSize = 5120;
constexpr int kSm70GemmaLongPrefillThreads = 256;
constexpr int kSm70GemmaVectorWidth = 4;

struct alignas(8) Sm70Half4 {
  half values[kSm70GemmaVectorWidth];
};

template <typename WeightT>
__device__ __forceinline__ float sm70_gemma_weight_to_float(WeightT value);

template <>
__device__ __forceinline__ float sm70_gemma_weight_to_float(float value) {
  return value;
}

template <>
__device__ __forceinline__ float sm70_gemma_weight_to_float(half value) {
  return __half2float(value);
}

template <>
__device__ __forceinline__ float sm70_gemma_weight_to_float(
    __nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename WeightT>
__global__ void __launch_bounds__(kSm70GemmaLongPrefillThreads, 2)
    sm70_gemma_long_prefill_fused_add_rms_norm_kernel(
        const half* __restrict__ input, const float* __restrict__ residual,
        const WeightT* __restrict__ weight, half* __restrict__ normalized_out,
        float* __restrict__ residual_out, float epsilon) {
  constexpr int kVectorsPerRow = kSm70GemmaHiddenSize / kSm70GemmaVectorWidth;
  constexpr int kVectorsPerThread =
      kVectorsPerRow / kSm70GemmaLongPrefillThreads;
  static_assert(kVectorsPerRow % kSm70GemmaLongPrefillThreads == 0);
  const int vector_row_offset = blockIdx.x * kVectorsPerRow;
  const auto* input4 = reinterpret_cast<const Sm70Half4*>(input);
  const auto* residual4 = reinterpret_cast<const float4*>(residual);
  auto* residual_out4 = reinterpret_cast<float4*>(residual_out);

  float4 row_values[kVectorsPerThread];
  float variance = 0.0f;
  #pragma unroll
  for (int iter = 0; iter < kVectorsPerThread; ++iter) {
    const int vector_idx = threadIdx.x + iter * blockDim.x;
    const Sm70Half4 x = input4[vector_row_offset + vector_idx];
    const float4 r = residual4[vector_row_offset + vector_idx];
    float4 value;
    value.x = __half2float(x.values[0]) + r.x;
    value.y = __half2float(x.values[1]) + r.y;
    value.z = __half2float(x.values[2]) + r.z;
    value.w = __half2float(x.values[3]) + r.w;
    row_values[iter] = value;
    residual_out4[vector_row_offset + vector_idx] = value;
    variance += value.x * value.x;
    variance += value.y * value.y;
    variance += value.z * value.z;
    variance += value.w * value.w;
  }

  // Keep the exact rms_norm_kernel<float, 4, 2> reduction topology.
  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduce_store;
  __shared__ float inverse_rms;
  variance = BlockReduce(reduce_store).Reduce(variance, CubAddOp{}, blockDim.x);
  if (threadIdx.x == 0) {
    inverse_rms = rsqrtf(variance / kSm70GemmaHiddenSize + epsilon);
  }
  __syncthreads();

  auto* normalized4 = reinterpret_cast<Sm70Half4*>(normalized_out);
  #pragma unroll
  for (int iter = 0; iter < kVectorsPerThread; ++iter) {
    const int vector_idx = threadIdx.x + iter * blockDim.x;
    const float4 value = row_values[iter];
    const int column = vector_idx * kSm70GemmaVectorWidth;
    const float w0 = sm70_gemma_weight_to_float(weight[column]) + 1.0f;
    const float w1 = sm70_gemma_weight_to_float(weight[column + 1]) + 1.0f;
    const float w2 = sm70_gemma_weight_to_float(weight[column + 2]) + 1.0f;
    const float w3 = sm70_gemma_weight_to_float(weight[column + 3]) + 1.0f;
    Sm70Half4 out;
    out.values[0] = __float2half_rn(value.x * inverse_rms * w0);
    out.values[1] = __float2half_rn(value.y * inverse_rms * w1);
    out.values[2] = __float2half_rn(value.z * inverse_rms * w2);
    out.values[3] = __float2half_rn(value.w * inverse_rms * w3);
    normalized4[vector_row_offset + vector_idx] = out;
  }
}
#endif

}  // namespace vllm

void rms_norm(torch::stable::Tensor& out,     // [..., hidden_size]
              torch::stable::Tensor& input,   // [..., hidden_size]
              torch::stable::Tensor& weight,  // [hidden_size]
              double epsilon) {
  STD_TORCH_CHECK(out.is_contiguous());
  if (input.stride(-1) != 1) {
    input = torch::stable::contiguous(input);
  }
  STD_TORCH_CHECK(input.stride(-1) == 1);
  STD_TORCH_CHECK(weight.is_contiguous());

  int hidden_size = input.size(-1);

  int num_tokens = input.numel() / hidden_size;
  int num_dims = input.dim();
  int64_t input_stride_d2 = input.stride(-2);
  int64_t input_stride_d3 = (num_dims >= 3) ? input.stride(-3) : 0;
  int64_t input_stride_d4 = (num_dims >= 4) ? input.stride(-4) : 0;
  int64_t input_shape_d2 = (num_dims >= 3) ? input.size(-2) : 0;
  int64_t input_shape_d3 = (num_dims >= 4) ? input.size(-3) : 0;

  // For large num_tokens, use smaller blocks to increase SM concurrency.
  const int max_block_size = (num_tokens < 256) ? 1024 : 256;
  dim3 grid(num_tokens);
  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();
  VLLM_STABLE_DISPATCH_RANK234(num_dims, [&] {
    VLLM_STABLE_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "rms_norm_kernel", [&] {
          const int calculated_vec_size =
              std::gcd(16 / sizeof(scalar_t), hidden_size);
          const int block_size =
              std::min(hidden_size / calculated_vec_size, max_block_size);
          dim3 block(block_size);
          VLLM_STABLE_DISPATCH_VEC_SIZE(calculated_vec_size, [&] {
            vllm::rms_norm_kernel<scalar_t, vec_size, tensor_rank>
                <<<grid, block, 0, stream>>>(
                    out.mutable_data_ptr<scalar_t>(),
                    input.const_data_ptr<scalar_t>(), input_stride_d2,
                    input_stride_d3, input_stride_d4, input_shape_d2,
                    input_shape_d3, weight.const_data_ptr<scalar_t>(), epsilon,
                    num_tokens, hidden_size);
          });
        });
  });
}

#define LAUNCH_FUSED_ADD_RMS_NORM(width)                                \
  VLLM_STABLE_DISPATCH_FLOATING_TYPES(                                  \
      input.scalar_type(), "fused_add_rms_norm_kernel", [&] {           \
        vllm::fused_add_rms_norm_kernel<scalar_t, width>                \
            <<<grid, block, 0, stream>>>(                               \
                input.mutable_data_ptr<scalar_t>(), input_stride,       \
                residual.mutable_data_ptr<scalar_t>(),                  \
                weight.const_data_ptr<scalar_t>(), epsilon, num_tokens, \
                hidden_size);                                           \
      });

void fused_add_rms_norm(torch::stable::Tensor& input,     // [..., hidden_size]
                        torch::stable::Tensor& residual,  // [..., hidden_size]
                        torch::stable::Tensor& weight,    // [hidden_size]
                        double epsilon) {
  STD_TORCH_CHECK(weight.scalar_type() == input.scalar_type());
  STD_TORCH_CHECK(input.scalar_type() == residual.scalar_type());
  STD_TORCH_CHECK(residual.is_contiguous());
  STD_TORCH_CHECK(weight.is_contiguous());
  int hidden_size = input.size(-1);
  int64_t input_stride = input.stride(-2);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  /* This kernel is memory-latency bound in many scenarios.
     When num_tokens is large, a smaller block size allows
     for increased block occupancy on CUs and better latency
     hiding on global mem ops. */
  const int max_block_size = (num_tokens < 256) ? 1024 : 256;
  dim3 block(std::min(hidden_size, max_block_size));
  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();
  /*If the tensor types are FP16/BF16, try to use the optimized kernel
    with packed + vectorized ops.
    Max optimization is achieved with a width-8 vector of FP16/BF16s
    since we can load at most 128 bits at once in a global memory op.
    However, this requires each tensor's data to be aligned to 16
    bytes.
   */
  auto inp_ptr = reinterpret_cast<std::uintptr_t>(input.data_ptr());
  auto res_ptr = reinterpret_cast<std::uintptr_t>(residual.data_ptr());
  auto wt_ptr = reinterpret_cast<std::uintptr_t>(weight.data_ptr());
  constexpr int vector_width = 8;
  constexpr int req_alignment_bytes =
      vector_width * 2;  // vector_width * sizeof(bfloat16 or float16) (float32
                         // falls back to non-vectorized version anyway)
  bool ptrs_are_aligned = inp_ptr % req_alignment_bytes == 0 &&
                          res_ptr % req_alignment_bytes == 0 &&
                          wt_ptr % req_alignment_bytes == 0;
  bool offsets_are_multiple_of_vector_width =
      hidden_size % vector_width == 0 && input_stride % vector_width == 0;
  bool batch_invariant_launch = vllm::vllm_is_batch_invariant();
  if (ptrs_are_aligned && offsets_are_multiple_of_vector_width &&
      !batch_invariant_launch) {
    LAUNCH_FUSED_ADD_RMS_NORM(8);
  } else {
    LAUNCH_FUSED_ADD_RMS_NORM(0);
  }
}

#ifndef USE_ROCM
void sm70_gemma_long_prefill_fused_add_rms_norm(
    torch::stable::Tensor& normalized_out, torch::stable::Tensor& residual_out,
    torch::stable::Tensor& input, torch::stable::Tensor& residual,
    torch::stable::Tensor& weight, double epsilon) {
  using torch::headeronly::ScalarType;
  STD_TORCH_CHECK(input.scalar_type() == ScalarType::Half);
  STD_TORCH_CHECK(normalized_out.scalar_type() == ScalarType::Half);
  STD_TORCH_CHECK(residual.scalar_type() == ScalarType::Float);
  STD_TORCH_CHECK(residual_out.scalar_type() == ScalarType::Float);
  STD_TORCH_CHECK(weight.scalar_type() == ScalarType::Half ||
                  weight.scalar_type() == ScalarType::BFloat16 ||
                  weight.scalar_type() == ScalarType::Float);
  STD_TORCH_CHECK(input.dim() == 2);
  STD_TORCH_CHECK(input.size(1) == vllm::kSm70GemmaHiddenSize);
  STD_TORCH_CHECK(input.size(0) >= 256);
  STD_TORCH_CHECK(residual.dim() == 2);
  STD_TORCH_CHECK(residual.size(0) == input.size(0));
  STD_TORCH_CHECK(residual.size(1) == input.size(1));
  STD_TORCH_CHECK(normalized_out.dim() == 2);
  STD_TORCH_CHECK(normalized_out.size(0) == input.size(0));
  STD_TORCH_CHECK(normalized_out.size(1) == input.size(1));
  STD_TORCH_CHECK(residual_out.dim() == 2);
  STD_TORCH_CHECK(residual_out.size(0) == input.size(0));
  STD_TORCH_CHECK(residual_out.size(1) == input.size(1));
  STD_TORCH_CHECK(weight.dim() == 1);
  STD_TORCH_CHECK(weight.numel() == vllm::kSm70GemmaHiddenSize);
  STD_TORCH_CHECK(input.is_contiguous());
  STD_TORCH_CHECK(residual.is_contiguous());
  STD_TORCH_CHECK(weight.is_contiguous());
  STD_TORCH_CHECK(normalized_out.is_contiguous());
  STD_TORCH_CHECK(residual_out.is_contiguous());
  STD_TORCH_CHECK(input.get_device_index() == residual.get_device_index());
  STD_TORCH_CHECK(input.get_device_index() == weight.get_device_index());
  STD_TORCH_CHECK(input.get_device_index() ==
                  normalized_out.get_device_index());
  STD_TORCH_CHECK(input.get_device_index() == residual_out.get_device_index());

  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();
  static std::atomic<bool> logged_route{false};
  bool expected = false;
  if (logged_route.compare_exchange_strong(expected, true)) {
    std::cerr << "SM70 exact mixed-dtype Gemma RMSNorm long-prefill op reached"
              << " tokens=" << input.size(0) << std::endl;
  }
  const auto launch = [&]<typename WeightT>(const WeightT* weight_ptr) {
    vllm::sm70_gemma_long_prefill_fused_add_rms_norm_kernel<<<
        input.size(0), vllm::kSm70GemmaLongPrefillThreads, 0, stream>>>(
        reinterpret_cast<const half*>(input.const_data_ptr<c10::Half>()),
        residual.const_data_ptr<float>(), weight_ptr,
        reinterpret_cast<half*>(normalized_out.mutable_data_ptr<c10::Half>()),
        residual_out.mutable_data_ptr<float>(), static_cast<float>(epsilon));
  };
  if (weight.scalar_type() == ScalarType::Half) {
    launch(reinterpret_cast<const half*>(weight.const_data_ptr<c10::Half>()));
  } else if (weight.scalar_type() == ScalarType::BFloat16) {
    launch(reinterpret_cast<const __nv_bfloat16*>(
        weight.const_data_ptr<c10::BFloat16>()));
  } else {
    launch(weight.const_data_ptr<float>());
  }
}
#endif
