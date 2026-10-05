// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Lossless FP16-source compact target selection; no precision conversion.
#include <torch/all.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <climits>
#include <cub/block/block_radix_sort.cuh>

template <int Items, bool Merge>
__global__ void select_compact_half_keys(const half* x, uint32_t* partial,
                                         float* values, int64_t* ids, int n,
                                         int stride, int k, int width) {
  constexpr int Threads = 256, Tile = Threads * Items;
  using Sort = cub::BlockRadixSort<uint32_t, Threads, Items>;
  __shared__ typename Sort::TempStorage storage;
  const int row = blockIdx.x;
  uint32_t keys[Items];
#pragma unroll
  for (int i = 0; i < Items; ++i) {
    const int col = threadIdx.x * Items + i;
    if constexpr (Merge) {
      keys[i] = col < width ? partial[row * width + col] : 0;
    } else {
      const int index = blockIdx.y * Tile + col;
      uint16_t bits = index < n ? __half_as_ushort(x[row * stride + index]) : 0;
      if ((bits & 0x7fff) == 0) bits = 0;
      if ((bits & 0x7fff) > 0x7c00) bits = 0x7fff;
      const uint16_t order = (bits & 0x8000) ? uint16_t(~bits) : bits ^ 0x8000;
      keys[i] = index < n ? (uint32_t(order) << 16) | (0xffff - index) : 0;
    }
  }
  Sort(storage).SortDescending(keys);
#pragma unroll
  for (int i = 0; i < Items; ++i) {
    const int col = threadIdx.x * Items + i;
    if (col < k) {
      if constexpr (Merge) {
        const int index = 0xffff - (keys[i] & 0xffff);
        ids[row * k + col] = index;
        values[row * k + col] = __half2float(x[row * stride + index]);
      } else {
        partial[row * width + blockIdx.y * k + col] = keys[i];
      }
    }
  }
}
void sm70_compact_half_topk_out(const at::Tensor& x, at::Tensor partial,
                                at::Tensor values, at::Tensor ids, int64_t k) {
  c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.dim() == 2 &&
              x.stride(1) == 1);
  TORCH_CHECK(x.size(1) <= 65535 && x.size(1) >= k && k == 64);
  int tiles = (x.size(1) + 1023) / 1024, width = tiles * k;
  TORCH_CHECK(width <= 4096 && partial.numel() == x.size(0) * width);
  TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= 32);
  TORCH_CHECK(partial.is_cuda() && values.is_cuda() && ids.is_cuda());
  TORCH_CHECK(partial.device() == x.device() && values.device() == x.device() &&
              ids.device() == x.device());
  TORCH_CHECK(partial.scalar_type() == at::kUInt32 &&
              values.scalar_type() == at::kFloat &&
              ids.scalar_type() == at::kLong);
  TORCH_CHECK(partial.is_contiguous() && values.is_contiguous() &&
              ids.is_contiguous());
  TORCH_CHECK(values.sizes() == at::IntArrayRef({x.size(0), k}) &&
              ids.sizes() == values.sizes());
  TORCH_CHECK(x.stride(0) <= INT_MAX / x.size(0));
  auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props->major == 7 && props->minor == 0);
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto* data = reinterpret_cast<const half*>(x.data_ptr());
  select_compact_half_keys<4, false>
      <<<dim3(x.size(0), tiles), 256, 0, stream>>>(
          data, partial.data_ptr<uint32_t>(), nullptr, nullptr, x.size(1),
          x.stride(0), k, width);
  if (width <= 1024)
    select_compact_half_keys<4, true><<<x.size(0), 256, 0, stream>>>(
        data, partial.data_ptr<uint32_t>(), values.data_ptr<float>(),
        ids.data_ptr<int64_t>(), x.size(1), x.stride(0), k, width);
  else
    select_compact_half_keys<16, true><<<x.size(0), 256, 0, stream>>>(
        data, partial.data_ptr<uint32_t>(), values.data_ptr<float>(),
        ids.data_ptr<int64_t>(), x.size(1), x.stride(0), k, width);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "sm70_compact_half_topk_out(Tensor x, Tensor(a!) partial, Tensor(b!) "
      "values, Tensor(c!) ids, int k) -> ()");
}
TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("sm70_compact_half_topk_out", &sm70_compact_half_topk_out);
}
