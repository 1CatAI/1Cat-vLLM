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

template <int Items, bool Merge, typename IdT = int64_t>
__global__ void select_compact_half_keys(const half* x, uint32_t* partial,
                                         float* values, IdT* ids, int n,
                                         int stride, int k, int width) {
  constexpr int Threads = 256, Tile = Threads * Items;
  using Sort = cub::BlockRadixSort<uint32_t, Threads, Items>;
  __shared__ typename Sort::TempStorage storage;
  const int64_t row = blockIdx.x;
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
        ids[row * k + col] = static_cast<IdT>(index);
        if (values != nullptr) {
          values[row * k + col] = __half2float(x[row * stride + index]);
        }
      } else {
        partial[row * width + blockIdx.y * k + col] = keys[i];
      }
    }
  }
}

template <int Items, int Stage, typename IdT = int64_t>
__global__ void select_compact_half_wide_keys(
    const half* x, const uint64_t* src_keys, uint64_t* dst_keys, float* values,
    IdT* ids, int n, int stride, int k, int in_width, int out_width) {
  constexpr int Threads = 256, Tile = Threads * Items;
  using Sort = cub::BlockRadixSort<uint64_t, Threads, Items>;
  __shared__ typename Sort::TempStorage storage;
  const int64_t row = blockIdx.x;
  uint64_t keys[Items];
#pragma unroll
  for (int i = 0; i < Items; ++i) {
    const int col = threadIdx.x * Items + i;
    if constexpr (Stage == 0) {
      const int index = blockIdx.y * Tile + col;
      if (index < n) {
        uint16_t bits = __half_as_ushort(x[row * stride + index]);
        if ((bits & 0x7fff) == 0) bits = 0;
        if ((bits & 0x7fff) > 0x7c00) bits = 0x7fff;
        const uint16_t order =
            (bits & 0x8000) ? uint16_t(~bits) : bits ^ 0x8000;
        keys[i] = (uint64_t(order) << 32) |
                  uint64_t(0xffffffffu - static_cast<uint32_t>(index));
      } else {
        keys[i] = 0ULL;
      }
    } else {
      const int index = (Stage == 1 ? blockIdx.y * Tile : 0) + col;
      keys[i] = index < in_width ? src_keys[row * in_width + index] : 0ULL;
    }
  }
  Sort(storage).SortDescending(keys, 0, 48);
#pragma unroll
  for (int i = 0; i < Items; ++i) {
    const int col = threadIdx.x * Items + i;
    if (col < k) {
      if constexpr (Stage == 2) {
        const uint32_t index =
            0xffffffffu - static_cast<uint32_t>(keys[i] & 0xffffffffULL);
        ids[row * k + col] = static_cast<IdT>(index);
        if (values != nullptr) {
          values[row * k + col] = __half2float(x[row * stride + index]);
        }
      } else {
        dst_keys[row * out_width + blockIdx.y * k + col] = keys[i];
      }
    }
  }
}

int64_t sm70_compact_half_topk_wide_revision() { return 1; }

void sm70_compact_half_topk_out(const at::Tensor& x, at::Tensor partial,
                                at::Tensor values, at::Tensor ids, int64_t k) {
  c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.dim() == 2 &&
              x.stride(1) == 1);
  TORCH_CHECK(x.size(1) <= 524288 && x.size(1) >= k && k == 64);
  TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= 64);
  TORCH_CHECK(partial.is_cuda() && values.is_cuda() && ids.is_cuda());
  TORCH_CHECK(partial.device() == x.device() && values.device() == x.device() &&
              ids.device() == x.device());
  TORCH_CHECK(values.scalar_type() == at::kFloat &&
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
  if (x.size(1) <= 65535) {
    int tiles = (x.size(1) + 1023) / 1024, width = tiles * k;
    TORCH_CHECK(width <= 4096 && partial.scalar_type() == at::kUInt32 &&
                partial.numel() == x.size(0) * width);
    select_compact_half_keys<4, false, int64_t>
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
  } else {
    int tiles1 = (x.size(1) + 1023) / 1024, w1 = tiles1 * k;
    int tiles2 = (w1 + 1023) / 1024, w2 = tiles2 * k;
    TORCH_CHECK(w2 <= 2048 && partial.scalar_type() == at::kLong &&
                partial.numel() == x.size(0) * (w1 + w2));
    auto* stage1 = reinterpret_cast<uint64_t*>(partial.data_ptr<int64_t>());
    auto* stage2 = stage1 + x.size(0) * w1;
    select_compact_half_wide_keys<4, 0, int64_t>
        <<<dim3(x.size(0), tiles1), 256, 0, stream>>>(
            data, nullptr, stage1, nullptr, nullptr, x.size(1), x.stride(0), k,
            0, w1);
    select_compact_half_wide_keys<4, 1, int64_t>
        <<<dim3(x.size(0), tiles2), 256, 0, stream>>>(
            nullptr, stage1, stage2, nullptr, nullptr, x.size(1), x.stride(0),
            k, w1, w2);
    if (w2 <= 1024) {
      select_compact_half_wide_keys<4, 2><<<x.size(0), 256, 0, stream>>>(
          data, stage2, nullptr, values.data_ptr<float>(),
          ids.data_ptr<int64_t>(), x.size(1), x.stride(0), k, w2, 0);
    } else {
      select_compact_half_wide_keys<8, 2><<<x.size(0), 256, 0, stream>>>(
          data, stage2, nullptr, values.data_ptr<float>(),
          ids.data_ptr<int64_t>(), x.size(1), x.stride(0), k, w2, 0);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sm70_compact_half_topk_ids_out(const at::Tensor& x, at::Tensor partial,
                                    at::Tensor ids, int64_t k) {
  c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.dim() == 2 &&
              x.stride(1) == 1);
  TORCH_CHECK(x.size(1) <= 524288 && x.size(1) >= k && (k == 32 || k == 64));
  TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= 64);
  TORCH_CHECK(partial.is_cuda() && ids.is_cuda());
  TORCH_CHECK(partial.device() == x.device() && ids.device() == x.device());
  TORCH_CHECK(ids.scalar_type() == at::kInt);
  TORCH_CHECK(partial.is_contiguous() && ids.is_contiguous());
  TORCH_CHECK(ids.sizes() == at::IntArrayRef({x.size(0), k}));
  TORCH_CHECK(x.stride(0) <= INT_MAX / x.size(0));
  auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props->major == 7 && props->minor == 0);
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto* data = reinterpret_cast<const half*>(x.data_ptr());
  if (x.size(1) <= 65535) {
    int tiles = (x.size(1) + 1023) / 1024, width = tiles * k;
    TORCH_CHECK(width <= 4096 && partial.scalar_type() == at::kUInt32 &&
                partial.numel() == x.size(0) * width);
    select_compact_half_keys<4, false, int32_t>
        <<<dim3(x.size(0), tiles), 256, 0, stream>>>(
            data, partial.data_ptr<uint32_t>(), nullptr, nullptr, x.size(1),
            x.stride(0), k, width);
    if (width <= 1024) {
      select_compact_half_keys<4, true, int32_t><<<x.size(0), 256, 0, stream>>>(
          nullptr, partial.data_ptr<uint32_t>(), nullptr,
          ids.data_ptr<int32_t>(), x.size(1), x.stride(0), k, width);
    } else if (width <= 2048) {
      select_compact_half_keys<8, true, int32_t><<<x.size(0), 256, 0, stream>>>(
          nullptr, partial.data_ptr<uint32_t>(), nullptr,
          ids.data_ptr<int32_t>(), x.size(1), x.stride(0), k, width);
    } else {
      select_compact_half_keys<16, true, int32_t>
          <<<x.size(0), 256, 0, stream>>>(nullptr, partial.data_ptr<uint32_t>(),
                                          nullptr, ids.data_ptr<int32_t>(),
                                          x.size(1), x.stride(0), k, width);
    }
  } else if (x.size(1) <= 131072 && k == 32) {
    int tiles1 = (x.size(1) + 2047) / 2048, w1 = tiles1 * k;
    TORCH_CHECK(w1 <= 2048 && partial.scalar_type() == at::kLong &&
                partial.numel() == x.size(0) * w1);
    auto* stage1 = reinterpret_cast<uint64_t*>(partial.data_ptr<int64_t>());
    select_compact_half_wide_keys<8, 0, int32_t>
        <<<dim3(x.size(0), tiles1), 256, 0, stream>>>(
            data, nullptr, stage1, nullptr, nullptr, x.size(1), x.stride(0), k,
            0, w1);
    if (w1 <= 1024) {
      select_compact_half_wide_keys<4, 2, int32_t>
          <<<x.size(0), 256, 0, stream>>>(nullptr, stage1, nullptr, nullptr,
                                          ids.data_ptr<int32_t>(), x.size(1),
                                          x.stride(0), k, w1, 0);
    } else {
      select_compact_half_wide_keys<8, 2, int32_t>
          <<<x.size(0), 256, 0, stream>>>(nullptr, stage1, nullptr, nullptr,
                                          ids.data_ptr<int32_t>(), x.size(1),
                                          x.stride(0), k, w1, 0);
    }
  } else if (x.size(1) <= 262144) {
    int tiles1 = (x.size(1) + 2047) / 2048, w1 = tiles1 * k;
    int tiles2 = (w1 + 2047) / 2048, w2 = tiles2 * k;
    TORCH_CHECK(w2 <= 2048 && partial.scalar_type() == at::kLong &&
                partial.numel() == x.size(0) * (w1 + w2));
    auto* stage1 = reinterpret_cast<uint64_t*>(partial.data_ptr<int64_t>());
    auto* stage2 = stage1 + x.size(0) * w1;
    select_compact_half_wide_keys<8, 0, int32_t>
        <<<dim3(x.size(0), tiles1), 256, 0, stream>>>(
            data, nullptr, stage1, nullptr, nullptr, x.size(1), x.stride(0), k,
            0, w1);
    select_compact_half_wide_keys<8, 1, int32_t>
        <<<dim3(x.size(0), tiles2), 256, 0, stream>>>(
            nullptr, stage1, stage2, nullptr, nullptr, x.size(1), x.stride(0),
            k, w1, w2);
    if (w2 <= 1024) {
      select_compact_half_wide_keys<4, 2, int32_t>
          <<<x.size(0), 256, 0, stream>>>(nullptr, stage2, nullptr, nullptr,
                                          ids.data_ptr<int32_t>(), x.size(1),
                                          x.stride(0), k, w2, 0);
    } else {
      select_compact_half_wide_keys<8, 2, int32_t>
          <<<x.size(0), 256, 0, stream>>>(nullptr, stage2, nullptr, nullptr,
                                          ids.data_ptr<int32_t>(), x.size(1),
                                          x.stride(0), k, w2, 0);
    }
  } else {
    int tiles1 = (x.size(1) + 1023) / 1024, w1 = tiles1 * k;
    int tiles2 = (w1 + 1023) / 1024, w2 = tiles2 * k;
    TORCH_CHECK(w2 <= 2048 && partial.scalar_type() == at::kLong &&
                partial.numel() == x.size(0) * (w1 + w2));
    auto* stage1 = reinterpret_cast<uint64_t*>(partial.data_ptr<int64_t>());
    auto* stage2 = stage1 + x.size(0) * w1;
    select_compact_half_wide_keys<4, 0, int32_t>
        <<<dim3(x.size(0), tiles1), 256, 0, stream>>>(
            data, nullptr, stage1, nullptr, nullptr, x.size(1), x.stride(0), k,
            0, w1);
    select_compact_half_wide_keys<4, 1, int32_t>
        <<<dim3(x.size(0), tiles2), 256, 0, stream>>>(
            nullptr, stage1, stage2, nullptr, nullptr, x.size(1), x.stride(0),
            k, w1, w2);
    if (w2 <= 1024) {
      select_compact_half_wide_keys<4, 2, int32_t>
          <<<x.size(0), 256, 0, stream>>>(nullptr, stage2, nullptr, nullptr,
                                          ids.data_ptr<int32_t>(), x.size(1),
                                          x.stride(0), k, w2, 0);
    } else {
      select_compact_half_wide_keys<8, 2, int32_t>
          <<<x.size(0), 256, 0, stream>>>(nullptr, stage2, nullptr, nullptr,
                                          ids.data_ptr<int32_t>(), x.size(1),
                                          x.stride(0), k, w2, 0);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "sm70_compact_half_topk_out(Tensor x, Tensor(a!) partial, Tensor(b!) "
      "values, Tensor(c!) ids, int k) -> ()");
  m.def(
      "sm70_compact_half_topk_ids_out(Tensor x, Tensor(a!) partial, "
      "Tensor(b!) ids, int k) -> ()");
  m.def("sm70_compact_half_topk_wide_revision() -> int");
}
TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("sm70_compact_half_topk_out", &sm70_compact_half_topk_out);
  m.impl("sm70_compact_half_topk_ids_out", &sm70_compact_half_topk_ids_out);
}
TORCH_LIBRARY_IMPL(_C, CompositeExplicitAutograd, m) {
  m.impl("sm70_compact_half_topk_wide_revision",
         &sm70_compact_half_topk_wide_revision);
}
