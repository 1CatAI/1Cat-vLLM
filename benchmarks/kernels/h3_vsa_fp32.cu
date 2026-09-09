// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Acceptance-only H3 sparse FP32 arithmetic; no runtime backend registration.
// Requires CUTLASS 4.4.2 and SM70. Preserve sequential FMA and global softmax.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <algorithm>
#include <cmath>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/kernel/default_gemm_universal.h>
#include <cutlass/gemm/device/gemm_universal_base.h>
#include <cutlass/epilogue/thread/linear_combination.h>

struct Geometry {
  const __half *q, *k, *v;
  __half* out;
  const int *indices, *sizes;
  int rows, heads, blocks, prefix, video_keep, group_indices;
  int first, count, keep;
  float scale;
};
__device__ int64_t map_offset(const Geometry& g, int group, int query) {
  return int64_t(group) * g.group_indices +
         (query < g.prefix
              ? query * g.blocks
              : g.prefix * g.blocks + (query - g.prefix) * g.video_keep);
}
__device__ int64_t input_offset(const Geometry& g, int group, int row,
                                int channel) {
  return (int64_t(group / g.heads) * g.rows + row) * g.heads * 128 +
         (group % g.heads) * 128 + channel;
}
__global__ void compact(const bool* mask, int* indices, int blocks, int prefix,
                        int video_keep, int group_indices) {
  int row = blockIdx.x, lane = threadIdx.x, query = row % blocks,
      group = row / blocks;
  int capacity = query < prefix ? blocks : video_keep;
  int64_t offset =
      int64_t(group) * group_indices +
      (query < prefix ? query * blocks
                      : prefix * blocks + (query - prefix) * video_keep);
  int count = 0;
  for (int start = 0; start < blocks; start += 32) {
    int key = start + lane;
    bool selected = key < blocks && mask[int64_t(row) * blocks + key];
    unsigned ballot = __ballot_sync(0xffffffff, selected);
    int pos = count + __popc(ballot & ((1u << lane) - 1));
    if (selected && pos < capacity) indices[offset + pos] = key;
    count += __popc(ballot);
  }
}

template <bool PV>
using BaseKernel = typename cutlass::gemm::kernel::DefaultGemmUniversal<
    float, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 1,
    float,
    typename std::conditional<PV, cutlass::layout::RowMajor,
                              cutlass::layout::ColumnMajor>::type,
    cutlass::ComplexTransform::kNone, 1, float, cutlass::layout::RowMajor,
    float, cutlass::arch::OpClassSimt, cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<64, 128, 8>, cutlass::gemm::GemmShape<32, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2,
    cutlass::arch::OpMultiplyAdd, cutlass::gemm::SharedMemoryClearOption::kNone,
    false, true, false>::GemmKernel;

template <bool PV>
struct H3IndexedCutlassKernel : BaseKernel<PV> {
  using Parent = BaseKernel<PV>;
  using Params = typename Parent::Params;
  using SharedStorage = typename Parent::SharedStorage;
  CUTLASS_DEVICE void operator()(Params const& input, SharedStorage& storage) {
    Params p = input;
    p.ptr_gather_B_indices +=
        int64_t(blockIdx.z) * (PV ? p.problem_size.k() : p.problem_size.n());
    Parent::operator()(p, storage);
  }
  CUTLASS_DEVICE static void invoke(Params const& p, SharedStorage& storage) {
    H3IndexedCutlassKernel op;
    op(p, storage);
  }
};
using QK =
    cutlass::gemm::device::GemmUniversalBase<H3IndexedCutlassKernel<false>>;
using PV =
    cutlass::gemm::device::GemmUniversalBase<H3IndexedCutlassKernel<true>>;

__device__ void convert_vector(uint4 value, float4* dest) {
  const __half2* h = reinterpret_cast<const __half2*>(&value);
  float2 a = __half22float2(h[0]), b = __half22float2(h[1]);
  float2 c = __half22float2(h[2]), d = __half22float2(h[3]);
  dest[0] = make_float4(a.x, a.y, b.x, b.y);
  dest[1] = make_float4(c.x, c.y, d.x, d.y);
}
__global__ void convert_qkv(Geometry g, float* converted, int64_t elements) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= elements / 8) return;
  int row = (i * 8 / (g.heads * 128)) % g.rows;
  bool valid = row % 64 < g.sizes[row / 64];
  uint4 zero = make_uint4(0, 0, 0, 0);
  convert_vector(valid ? reinterpret_cast<const uint4*>(g.q)[i] : zero,
                 reinterpret_cast<float4*>(converted + i * 8));
  convert_vector(valid ? reinterpret_cast<const uint4*>(g.k)[i] : zero,
                 reinterpret_cast<float4*>(converted + elements + i * 8));
  convert_vector(valid ? reinterpret_cast<const uint4*>(g.v)[i] : zero,
                 reinterpret_cast<float4*>(converted + 2 * elements + i * 8));
}
__global__ void prepare_batch(Geometry g, const float* data, int64_t elements,
                              float* scores, float* probability, float* result,
                              int64_t* pointers, int* tokens) {
  int batch = blockIdx.x, group = batch / g.count,
      query = g.first + batch % g.count;
  int batches = gridDim.x, n = g.keep * 64;
  int64_t offset = map_offset(g, group, query);
  for (int i = threadIdx.x; i < n; i += blockDim.x)
    tokens[int64_t(batch) * n + i] = g.indices[offset + i / 64] * 64 + i % 64;
  if (threadIdx.x == 0) {
    pointers[batch] =
        reinterpret_cast<int64_t>(data + input_offset(g, group, query * 64, 0));
    pointers[batches + batch] = reinterpret_cast<int64_t>(
        data + elements + input_offset(g, group, 0, 0));
    pointers[2 * batches + batch] =
        reinterpret_cast<int64_t>(scores + int64_t(batch) * 64 * n);
    pointers[3 * batches + batch] =
        reinterpret_cast<int64_t>(probability + int64_t(batch) * 64 * n);
    pointers[4 * batches + batch] = reinterpret_cast<int64_t>(
        data + 2 * elements + input_offset(g, group, 0, 0));
    pointers[5 * batches + batch] =
        reinterpret_cast<int64_t>(result + int64_t(batch) * 64 * 128);
  }
}
__global__ void mask_scores(Geometry g, const int* tokens, float* scores) {
  int batch = blockIdx.y, key = blockIdx.x, col = threadIdx.x;
  int n = g.keep * 64, selected = tokens[int64_t(batch) * n + key * 64] / 64;
  if (col < g.sizes[selected]) return;
  for (int q = 0; q < 64; ++q)
    scores[(int64_t(batch) * 64 + q) * n + key * 64 + col] = -CUDART_INF_F;
}
__global__ void scatter_result(Geometry g, const float* result, int batches) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= batches * 64 * 16) return;
  int batch = i / (64 * 16), row = i / 16 % 64, d = i % 16 * 8;
  int group = batch / g.count, query = g.first + batch % g.count;
  if (row >= g.sizes[query]) return;
  float4 a = reinterpret_cast<const float4*>(result + i * 8)[0];
  float4 b = reinterpret_cast<const float4*>(result + i * 8)[1];
  uint4 value;
  __half2* h = reinterpret_cast<__half2*>(&value);
  h[0] = __floats2half2_rn(a.x, a.y);
  h[1] = __floats2half2_rn(a.z, a.w);
  h[2] = __floats2half2_rn(b.x, b.y);
  h[3] = __floats2half2_rn(b.z, b.w);
  *reinterpret_cast<uint4*>(
      g.out + input_offset(g, group, query * 64 + row, d)) = value;
}

torch::Tensor forward_impl(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                           torch::Tensor mask, torch::Tensor sizes,
                           double scale, int64_t prefix, int64_t topk,
                           bool checked) {
  TORCH_CHECK(q.is_cuda() && q.dim() == 4 && q.scalar_type() == at::kHalf &&
                  q.is_contiguous() && q.size(0) > 0 && q.size(1) > 0 &&
                  q.size(1) % 64 == 0 && q.size(2) > 0 && q.size(3) == 128,
              "indexed FP32 H3 requires contiguous FP16 BSHD128 tiles");
  for (auto& x : {k, v})
    TORCH_CHECK(x.device() == q.device() && x.sizes() == q.sizes() &&
                    x.scalar_type() == at::kHalf && x.is_contiguous() &&
                    !x.requires_grad(),
                "matching inference QKV required");
  TORCH_CHECK(!q.requires_grad(), "inference only");
  int64_t blocks = q.size(1) / 64, groups = q.size(0) * q.size(2);
  TORCH_CHECK(
      q.size(1) <= INT_MAX && groups * blocks <= INT_MAX && groups * 8 <= 65535,
      "indexed H3 exceeds index limits");
  TORCH_CHECK(prefix >= 0 && prefix < blocks && topk > 0,
              "invalid H3 prefix/topk");
  TORCH_CHECK(std::isfinite(scale) && scale > 0,
              "positive finite scale required");
  TORCH_CHECK(mask.device() == q.device() && mask.dim() == 4 &&
                  mask.scalar_type() == at::kBool && mask.is_contiguous() &&
                  mask.size(0) == q.size(0) && mask.size(1) == q.size(2) &&
                  mask.size(2) == blocks && mask.size(3) == blocks,
              "H3 map shape mismatch");
  TORCH_CHECK(sizes.device() == q.device() && sizes.dim() == 1 &&
                  sizes.numel() == blocks && sizes.is_contiguous() &&
                  sizes.scalar_type() == at::kInt,
              "H3 sizes shape mismatch");
  c10::cuda::CUDAGuard guard(q.device());
  auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props->major == 7 && props->minor == 0, "SM70 required");
  int keep = prefix + std::min(topk, blocks - prefix);
  if (checked) {
    TORCH_CHECK(sizes.min().item<int>() > 0 && sizes.max().item<int>() <= 64,
                "sizes must lie in [1,64]");
    auto counts = mask.sum(-1);
    TORCH_CHECK((prefix == 0 ||
                 counts.slice(-1, 0, prefix).eq(blocks).all().item<bool>()) &&
                    counts.slice(-1, prefix).eq(keep).all().item<bool>(),
                "map must follow H3 prefix and topk counts");
    TORCH_CHECK(prefix == 0 || mask.slice(-1, 0, prefix).all().item<bool>(),
                "all prefix keys must be selected");
  }
  int64_t per_group = prefix * blocks + (blocks - prefix) * keep;
  TORCH_CHECK(per_group <= INT_MAX, "compact map exceeds index limits");

  TORCH_CHECK(((uintptr_t(q.data_ptr()) | uintptr_t(k.data_ptr()) |
                uintptr_t(v.data_ptr())) &
               15) == 0,
              "16-byte aligned sources required");
  auto indices = torch::zeros({groups, per_group}, sizes.options());
  auto output = torch::zeros_like(q);
  auto stream = at::cuda::getCurrentCUDAStream();
  compact<<<groups * blocks, 32, 0, stream>>>(mask.data_ptr<bool>(),
                                              indices.data_ptr<int>(), blocks,
                                              prefix, keep, per_group);
  Geometry g{reinterpret_cast<const __half*>(q.data_ptr()),
             reinterpret_cast<const __half*>(k.data_ptr()),
             reinterpret_cast<const __half*>(v.data_ptr()),
             reinterpret_cast<__half*>(output.data_ptr()),
             indices.data_ptr<int>(),
             sizes.data_ptr<int>(),
             int(q.size(1)),
             int(q.size(2)),
             int(blocks),
             int(prefix),
             keep,
             int(per_group),
             0,
             0,
             0,
             float(scale)};
  auto data = torch::empty({3, q.numel()}, q.options().dtype(at::kFloat));
  convert_qkv<<<(q.numel() / 8 + 255) / 256, 256, 0, stream>>>(
      g, data.data_ptr<float>(), q.numel());
  QK qk;
  PV pv;
  for (int section = 0; section < 2; ++section) {
    int begin = section == 0 ? 0 : prefix, end = section == 0 ? prefix : blocks;
    g.keep = section == 0 ? blocks : keep;
    int chunk = std::max<int64_t>(
        1, std::min<int64_t>(
               8, 256 * 1024 * 1024 / (groups * g.keep * 64 * 128 * 4)));
    for (int first = begin; first < end; first += chunk) {
      g.first = first;
      g.count = std::min(chunk, end - first);
      int batches = groups * g.count, n = g.keep * 64;
      auto scores =
          torch::empty({batches, 64, n}, q.options().dtype(at::kFloat));
      auto probability = torch::empty_like(scores);
      auto result =
          torch::empty({batches, 64, 128}, q.options().dtype(at::kFloat));
      auto pointers =
          torch::empty({6, batches}, sizes.options().dtype(at::kLong));
      auto tokens = torch::empty({batches, n}, sizes.options());
      auto* ptr = pointers.data_ptr<int64_t>();
      prepare_batch<<<batches, 128, 0, stream>>>(
          g, data.data_ptr<float>(), q.numel(), scores.data_ptr<float>(),
          probability.data_ptr<float>(), result.data_ptr<float>(), ptr,
          tokens.data_ptr<int>());
      QK::Arguments qa(cutlass::gemm::GemmUniversalMode::kArray, {64, n, 128},
                       batches, {float(scale), 0.0f}, ptr, ptr + batches,
                       ptr + 2 * batches, ptr + 2 * batches, 0, 0, 0, 0,
                       int(q.size(2)) * 128, int(q.size(2)) * 128, n, n,
                       nullptr, tokens.data_ptr<int>(), nullptr);
      TORCH_CHECK(qk(qa, nullptr, stream) == cutlass::Status::kSuccess,
                  "CUTLASS QK failed");
      mask_scores<<<dim3(g.keep, batches), 64, 0, stream>>>(
          g, tokens.data_ptr<int>(), scores.data_ptr<float>());
      torch::softmax_out(probability, scores, -1);
      PV::Arguments pa(cutlass::gemm::GemmUniversalMode::kArray, {64, 128, n},
                       batches, {1.0f, 0.0f}, ptr + 3 * batches,
                       ptr + 4 * batches, ptr + 5 * batches, ptr + 5 * batches,
                       0, 0, 0, 0, n, int(q.size(2)) * 128, 128, 128, nullptr,
                       tokens.data_ptr<int>(), nullptr);
      TORCH_CHECK(pv(pa, nullptr, stream) == cutlass::Status::kSuccess,
                  "CUTLASS PV failed");
      scatter_result<<<(batches * 64 * 16 + 255) / 256, 256, 0, stream>>>(
          g, result.data_ptr<float>(), batches);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }
  return output;
}

torch::Tensor forward(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                      torch::Tensor mask, torch::Tensor sizes, double scale,
                      int64_t prefix, int64_t topk) {
  return forward_impl(q, k, v, mask, sizes, scale, prefix, topk, true);
}
torch::Tensor prevalidated(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                           torch::Tensor mask, torch::Tensor sizes,
                           double scale, int64_t prefix, int64_t topk) {
  return forward_impl(q, k, v, mask, sizes, scale, prefix, topk, false);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward);
  m.def("_forward_prevalidated", &prevalidated);
}
