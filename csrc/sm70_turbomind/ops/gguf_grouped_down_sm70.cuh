// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
namespace {
template <int Type>
__global__ void raw_down_grouped_routes(half* projected, const Q8_1* input,
                                        const int* ids, const int* order,
                                        const uint8_t* weights, int routes,
                                        int n, int groups, int stride,
                                        int left_groups, int experts) {
  __shared__ int active, count, expert, original[4];
  extern __shared__ Q8_1 shared_x[];
  const int route = blockIdx.y;
  if (threadIdx.x == 0) {
    expert = ids[route];
    int first = route;
    while (first > 0 && ids[first - 1] == expert) --first;
    active = ((route - first) % 4 == 0) && expert >= 0 && expert < experts;
    count = 1;
    while (count < 4 && route + count < routes && ids[route + count] == expert)
      ++count;
    for (int q = 0; q < count; ++q) original[q] = order[route + q];
  }
  __syncthreads();
  if (!active) return;
  for (int i = threadIdx.x; i < count * groups * 9; i += blockDim.x) {
    const int src = original[i / (groups * 9)];
    reinterpret_cast<uint32_t*>(shared_x)[i] =
        reinterpret_cast<const uint32_t*>(
            input)[int64_t(src) * groups * 9 + i % (groups * 9)];
  }
  __syncthreads();
  const int col = blockIdx.x * 128 + threadIdx.x;
  if (col >= n) return;
  const uint8_t* row = weights + (int64_t(expert) * n + col) * stride;
  float acc[4] = {};
  for (int group = 0; group < groups; ++group) {
    const auto decoded = vllm::sm70_gguf::OriginalIntegerDot<Type>::decode(
        row, group + left_groups);
#pragma unroll
    for (int q = 0; q < 4; ++q)
      if (q < count)
        acc[q] += vllm::sm70_gguf::OriginalIntegerDot<Type>::dot_decoded(
            decoded, shared_x[q * groups + group]);
  }
#pragma unroll
  for (int q = 0; q < 4; ++q)
    if (q < count)
      projected[int64_t(original[q]) * n + col] = __float2half_rn(acc[q]);
}

__global__ void raw_down_grouped_unroute(half* output, const half* projected,
                                         const float* weights, int n,
                                         int top_k) {
  const int token = blockIdx.y;
  const int col = blockIdx.x * 128 + threadIdx.x;
  if (col >= n) return;
  // Retain the four warp-owned subtotals and their original addition order.
  float sum = 0.f;
#pragma unroll
  for (int owner = 0; owner < 4; ++owner) {
    float partial = 0.f;
    for (int route = owner; route < top_k; route += 4) {
      const int slot = token * top_k + route;
      partial +=
          __half2float(projected[int64_t(slot) * n + col]) * weights[slot];
    }
    sum += partial;
  }
  output[int64_t(token) * n + col] = __float2half_rn(sum);
}

template <int Type>
void launch_grouped_raw_down(torch::Tensor out, torch::Tensor projected,
                             torch::Tensor input, torch::Tensor ids,
                             torch::Tensor order, torch::Tensor probabilities,
                             torch::Tensor weights, int left) {
  const int n = out.size(1), top_k = ids.size(1), groups = input.size(2);
  const auto stream = at::cuda::getCurrentCUDAStream();
  raw_down_grouped_routes<Type><<<dim3((n + 127) / 128, ids.numel()), 128,
                                  4 * groups * sizeof(Q8_1), stream>>>(
      reinterpret_cast<half*>(projected.data_ptr()),
      reinterpret_cast<const Q8_1*>(input.data_ptr()), ids.data_ptr<int>(),
      order.data_ptr<int>(), weights.data_ptr<uint8_t>(), ids.numel(), n,
      groups, weights.size(2), left / 32, weights.size(0));
  raw_down_grouped_unroute<<<dim3((n + 127) / 128, out.size(0)), 128, 0,
                             stream>>>(
      reinterpret_cast<half*>(out.data_ptr()),
      reinterpret_cast<const half*>(projected.data_ptr()),
      probabilities.data_ptr<float>(), n, top_k);
}
}  // namespace

void gguf_dp4a_raw_down_grouped_sm70_out(
    torch::Tensor out, torch::Tensor projected, torch::Tensor input,
    torch::Tensor ids, torch::Tensor order, torch::Tensor probabilities,
    torch::Tensor weights, int64_t source_type, int64_t left) {
  const c10::cuda::CUDAGuard guard(input.device());
  require_sm70();
  TORCH_CHECK(
      (source_type == 20 || source_type == 42) && left >= 0 && left % 32 == 0,
      "Grouped down requires original IQ4_NL or Q2_0 rows");
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kByte &&
                  input.is_contiguous() && input.dim() == 4 &&
                  input.size(0) > 0 && input.size(0) <= 40 &&
                  input.size(1) > 0 && input.size(1) <= 16 &&
                  input.size(2) > 0 && input.size(3) == sizeof(Q8_1),
              "Invalid grouped down Q8_1 inputs");
  const int m = input.size(0), top_k = input.size(1), k = input.size(2) * 32;
  TORCH_CHECK(ids.device() == input.device() && ids.scalar_type() == at::kInt &&
                  ids.is_contiguous() && ids.dim() == 2 && ids.size(0) == m &&
                  ids.size(1) == top_k && order.device() == input.device() &&
                  order.scalar_type() == at::kInt && order.is_contiguous() &&
                  order.sizes() == ids.sizes() &&
                  probabilities.device() == input.device() &&
                  probabilities.scalar_type() == at::kFloat &&
                  probabilities.is_contiguous() &&
                  probabilities.sizes() == ids.sizes(),
              "Invalid grouped down route mapping");
  TORCH_CHECK(
      out.device() == input.device() && out.scalar_type() == at::kHalf &&
          out.is_contiguous() && out.dim() == 2 && out.size(0) == m &&
          out.size(1) > 0 && weights.device() == input.device() &&
          weights.scalar_type() == at::kByte && weights.is_contiguous() &&
          weights.dim() == 3 && weights.size(0) > 0 &&
          weights.size(1) == out.size(1) &&
          weights.size(2) >= ((k + left + (source_type == 20 ? 31 : 63)) /
                              (source_type == 20 ? 32 : 64)) *
                                 18 &&
          projected.device() == input.device() &&
          projected.scalar_type() == at::kHalf && projected.is_contiguous() &&
          projected.numel() == int64_t(m) * top_k * out.size(1),
      "Invalid grouped down banks or output");
  if (source_type == 20)
    launch_grouped_raw_down<20>(out, projected, input, ids, order,
                                probabilities, weights, left);
  else
    launch_grouped_raw_down<42>(out, projected, input, ids, order,
                                probabilities, weights, left);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
