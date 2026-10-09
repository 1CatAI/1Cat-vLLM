// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Lossless FP16/FP32-source compact target selection; no precision conversion.
#include <torch/all.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <climits>
#include <type_traits>
#include <cub/block/block_radix_sort.cuh>

template <typename Input, typename Key, int Items, bool Merge,
          bool Intermediate = false>
__global__ void select_compact_keys(const Input* x, Key* partial, float* values,
                                    int64_t* ids, int n, int stride, int k,
                                    int width, Key* extra) {
  constexpr int Threads = 256, Tile = Threads * Items;
  using Sort = cub::BlockRadixSort<Key, Threads, Items>;
  __shared__ typename Sort::TempStorage storage;
  const int row = blockIdx.x;
  Key keys[Items];
#pragma unroll
  for (int i = 0; i < Items; ++i) {
    const int col = threadIdx.x * Items + i;
    if constexpr (Merge) {
      const int index = col + (Intermediate ? blockIdx.y * Tile : 0);
      keys[i] = index < width ? partial[row * width + index] : 0;
    } else {
      const int index = blockIdx.y * Tile + col;
      Key order;
      if constexpr (std::is_same_v<Input, half>) {
        uint16_t bits =
            index < n ? __half_as_ushort(x[row * stride + index]) : 0;
        if ((bits & 0x7fff) == 0) bits = 0;
        if ((bits & 0x7fff) > 0x7c00) bits = 0x7fff;
        order = (bits & 0x8000) ? uint16_t(~bits) : bits ^ 0x8000;
      } else {
        uint32_t bits =
            index < n ? __float_as_uint(x[row * stride + index]) : 0;
        if ((bits & 0x7fffffff) == 0) bits = 0;
        if ((bits & 0x7fffffff) > 0x7f800000) bits = 0x7fffffff;
        order = (bits & 0x80000000) ? ~bits : bits ^ 0x80000000;
      }
      keys[i] = index < n ? (order << 16) | (0xffff - index) : 0;
    }
  }
  Sort(storage).SortDescending(keys, 0, sizeof(Key) == 8 ? 48 : 32);
#pragma unroll
  for (int i = 0; i < Items; ++i) {
    const int col = threadIdx.x * Items + i;
    if (col < k) {
      if constexpr (Merge) {
        if constexpr (Intermediate) {
          extra[(row * gridDim.y + blockIdx.y) * k + col] = keys[i];
        } else {
          const int index = 0xffff - (keys[i] & 0xffff);
          ids[row * k + col] = index;
          if constexpr (std::is_same_v<Input, half>)
            values[row * k + col] = __half2float(x[row * stride + index]);
          else
            values[row * k + col] = x[row * stride + index];
        }
      } else {
        partial[row * width + blockIdx.y * k + col] = keys[i];
      }
    }
  }
}

// Stable radix ranks preserve ascending vocabulary IDs within a tied leaf.
// Only the primary float key is sorted here; merges retain the full ID key.
template <int Items>
__global__ void select_fp32_leaf(const float* x, uint64_t* partial, int n,
                                 int stride, int width) {
  constexpr int Threads = 256, Tile = Threads * Items;
  using Sort = cub::BlockRadixSort<uint32_t, Threads, Items, uint16_t>;
  __shared__ typename Sort::TempStorage storage;
  uint32_t keys[Items];
  uint16_t indices[Items];
  const int row = blockIdx.x;
#pragma unroll
  for (int i = 0; i < Items; ++i) {
    const int index = blockIdx.y * Tile + threadIdx.x * Items + i;
    uint32_t bits = index < n ? __float_as_uint(x[row * stride + index]) : 0;
    if ((bits & 0x7fffffff) == 0) bits = 0;
    if ((bits & 0x7fffffff) > 0x7f800000) bits = 0x7fffffff;
    keys[i] = index < n ? ((bits & 0x80000000) ? ~bits : bits ^ 0x80000000) : 0;
    indices[i] = index < n ? index : 0xffff;
  }
  Sort(storage).SortDescending(keys, indices);
#pragma unroll
  for (int i = 0; i < Items; ++i) {
    const int col = threadIdx.x * Items + i;
    if (col < 64)
      partial[row * width + blockIdx.y * 64 + col] =
          (uint64_t(keys[i]) << 16) | (0xffff - indices[i]);
  }
}

template <typename Input, typename Key>
void launch_compact_topk(const at::Tensor& x, at::Tensor partial,
                         at::Tensor values, at::Tensor ids, int k, int tiles,
                         int width, cudaStream_t stream) {
  const auto* data = reinterpret_cast<const Input*>(x.data_ptr());
  if constexpr (std::is_same_v<Input, float>) {
    select_fp32_leaf<4><<<dim3(x.size(0), tiles), 256, 0, stream>>>(
        data, partial.data_ptr<uint64_t>(), x.size(1), x.stride(0), width);
  } else {
    select_compact_keys<Input, Key, 4, false>
        <<<dim3(x.size(0), tiles), 256, 0, stream>>>(
            data, partial.data_ptr<Key>(), nullptr, nullptr, x.size(1),
            x.stride(0), k, width, nullptr);
  }
  if constexpr (std::is_same_v<Input, float>) {
    if (width > 1024) {
      const int groups = (width + 511) / 512;
      auto* temp = partial.data_ptr<Key>() + x.size(0) * width;
      select_compact_keys<Input, Key, 2, true, true>
          <<<dim3(x.size(0), groups), 256, 0, stream>>>(
              data, partial.data_ptr<Key>(), nullptr, nullptr, x.size(1),
              x.stride(0), k, width, temp);
      select_compact_keys<Input, Key, 2, true><<<x.size(0), 256, 0, stream>>>(
          data, temp, values.data_ptr<float>(), ids.data_ptr<int64_t>(),
          x.size(1), x.stride(0), k, groups * k, nullptr);
      return;
    }
  }
  if (width <= 1024)
    select_compact_keys<Input, Key, 4, true><<<x.size(0), 256, 0, stream>>>(
        data, partial.data_ptr<Key>(), values.data_ptr<float>(),
        ids.data_ptr<int64_t>(), x.size(1), x.stride(0), k, width, nullptr);
  else
    select_compact_keys<Input, Key, 16, true><<<x.size(0), 256, 0, stream>>>(
        data, partial.data_ptr<Key>(), values.data_ptr<float>(),
        ids.data_ptr<int64_t>(), x.size(1), x.stride(0), k, width, nullptr);
}

void sm70_compact_half_topk_out(const at::Tensor& x, at::Tensor partial,
                                at::Tensor values, at::Tensor ids, int64_t k) {
  c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(x.is_cuda() &&
              (x.scalar_type() == at::kHalf || x.scalar_type() == at::kFloat) &&
              x.dim() == 2 && x.stride(1) == 1);
  TORCH_CHECK(x.size(1) <= 65535 && x.size(1) >= k && k == 64);
  int tiles = (x.size(1) + 1023) / 1024, width = tiles * k;
  TORCH_CHECK(width <= 4096 &&
              partial.numel() ==
                  x.size(0) *
                      (width + (x.scalar_type() == at::kFloat && width > 1024
                                    ? ((width + 511) / 512) * k
                                    : 0)));
  TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= 32);
  TORCH_CHECK(partial.is_cuda() && values.is_cuda() && ids.is_cuda());
  TORCH_CHECK(partial.device() == x.device() && values.device() == x.device() &&
              ids.device() == x.device());
  TORCH_CHECK(partial.scalar_type() ==
                  (x.scalar_type() == at::kHalf ? at::kUInt32 : at::kUInt64) &&
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
  if (x.scalar_type() == at::kHalf)
    launch_compact_topk<half, uint32_t>(x, partial, values, ids, k, tiles,
                                        width, stream);
  else
    launch_compact_topk<float, uint64_t>(x, partial, values, ids, k, tiles,
                                         width, stream);
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
