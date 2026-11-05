// SPDX-License-Identifier: Apache-2.0
// Compact scheduling for the existing OpenMMLab/1Cat SM70 canonical MMA.
// Arithmetic, operand decoders and epilogue are reused with original notices.
#include <torch/types.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cassert>
#include <climits>
#include <vector>
#include "src/turbomind/kernels/gemm/arch/config_sm70_s884.h"

namespace prefmoe {
namespace tm = turbomind;
namespace tg = turbomind::gemm;
using tm::Array;

struct MemberTile { int expert, tile, rows, reserved; };

// One stream owns this plan until the GEMM completes. No ticket or stale
// generation can survive: every launch overwrites count and every live entry.
template<int BM>
__global__ void plan_members(const int* offsets, const int* indices,
                             const tg::StridedPtr* weights,
                             const tg::StridedPtr* stats,
                             int experts, int routes, int input_rows,
                             int* count, MemberTile* tasks, int capacity) {
  __shared__ int prefix[33];
  const int e = threadIdx.x;
  if (e < experts) {
    const int lo = offsets[e], hi = offsets[e + 1];
    assert(lo >= 0 && hi >= lo && hi <= routes);
    if (hi > lo) {
      assert(weights[e].ptr != nullptr && stats[e].ptr != nullptr);
      assert(weights[e].stride > 0 && stats[e].stride > 0);
    }
    prefix[e + 1] = (hi - lo + BM - 1) / BM;
  }
  if (e == 0) prefix[0] = 0;
  __syncthreads();
  if (e == 0) {
    for (int j = 0; j < experts; ++j) prefix[j + 1] += prefix[j];
    assert(prefix[experts] <= capacity);
    *count = prefix[experts];
  }
  __syncthreads();
  if (e < experts) {
    for (int j = prefix[e]; j < prefix[e + 1]; ++j)
      tasks[j] = {e, j - prefix[e], offsets[e + 1] - offsets[e], 0};
  }
  if (indices) {
    for (int r = offsets[0] + e; r < offsets[experts]; r += blockDim.x)
      assert(indices[r] >= 0 && indices[r] < input_rows);
  }
}

template<int BM, int BN, int BK>
struct CompactScheduler {
  static constexpr int group_axis = 0;
  using Tile = typename tg::SchedulerSm70<tg::kColMajor, BM, BN, BK, BK, false, 0>::Tile;
  struct SharedStorage {};
  Array<int, 3> tiles_;
  MemberTile member;
  int n_tile, n, k;
  template<class Reinit>
  __device__ bool init(Tile& t, SharedStorage&, Reinit) {
    t.tile_id = {member.tile, n_tile, 0};
    t.shape = {member.rows, n, k};
    t.k_iters = {0, k / BK};
    t.group_id = member.expert;
    t.linear_tile_id = 0;  // no partial-buffer indexing without split K
    return true;
  }
};

template<int Type, int BM, int BN>
struct Configuration {
  static constexpr int group = Type == 16 ? 32 : 16;
  using C = tg::sm70_s884::Config_GgufLattice<Type, group, tg::kColMajor, 0>;
  using T = typename C::template Type<BM, BN, 32, (BM == 128 && BN == 128 ? 4 : 2), 2, 1,
      tm::cache_policy::Default, tm::cache_policy::Default, 2, false, 1, group>;
  using Scheduler = CompactScheduler<BM, BN, 32>;
  using Kernel = tg::GemmUniversal<tg::Sm70, typename T::Mainloop,
                                  typename T::Epilogue, Scheduler>;
};

template<int Type, int BM, int BN>
__global__ void compact_gemm(tg::GemmParam p, tg::EpilogueParam ep,
                             const int* count, const MemberTile* tasks,
                             int n, int k, bool persistent) {
  using C = Configuration<Type, BM, BN>;
  using Kernel = typename C::Kernel;
  extern __shared__ __align__(16) char scratch[];
  const int nt = (n + BN - 1) / BN;
  const int total = *count * nt;
  // N fastest: weight reuse across nearby member tiles without global tickets.
  for (int task = blockIdx.x; task < total; task += gridDim.x) {
    const auto member = tasks[task / nt];
    typename C::Scheduler sched{{0, nt, 1}, member, task % nt, n, k};
    Kernel kernel;
    kernel(p, ep, sched, scratch);
    // Epilogue/mainloop storage is a union and will be reused by the next tile.
    __syncthreads();
    if (!persistent) break;
  }
}

template<int Type, int BM, int BN>
void launch(torch::Tensor out, torch::Tensor input, torch::Tensor offsets,
            torch::Tensor wp, torch::Tensor sp, torch::Tensor plan,
            const int* indices, int experts, int routes, int grid,
            cudaStream_t stream) {
  using C = Configuration<Type, BM, BN>;
  using Kernel = typename C::Kernel;
  const int n = out.size(1), k = input.size(1);
  const int capacity = (plan.numel() - 1) / 4;
  int* count = plan.data_ptr<int>();
  auto* tasks = reinterpret_cast<MemberTile*>(count + 1);
  plan_members<BM><<<1, 32, 0, stream>>>(offsets.data_ptr<int>(), indices,
      reinterpret_cast<tg::StridedPtr*>(wp.data_ptr()),
      reinterpret_cast<tg::StridedPtr*>(sp.data_ptr()), experts, routes,
      input.size(0), count, tasks, capacity);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  tg::MatrixParam a{input.data_ptr(), int(input.stride(0)), offsets.data_ptr<int>(),
                    const_cast<int*>(indices), nullptr};
  tg::MatrixParam b{wp.data_ptr(), 0, nullptr, nullptr, nullptr};
  tg::MatrixParam v{sp.data_ptr(), 0, nullptr, nullptr, nullptr};
  tg::MatrixParam d{out.data_ptr(), int(out.stride(0)), offsets.data_ptr<int>(), nullptr, nullptr};
  tg::GemmParam p{a, b, {}, v};
  tg::EpilogueParam ep{};
  ep.c = d;
  ep.combine_mat = {d, 1.f, 0.f};
  constexpr int shared = sizeof(typename Kernel::SharedStorage);
  auto fn = compact_gemm<Type, BM, BN>;
  if constexpr (shared > 48 * 1024)
    C10_CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
  const bool persistent = grid != 0;
  const int blocks = persistent ? grid : capacity * ((n + BN - 1) / BN);
  fn<<<blocks, Kernel::WARP_CNT * 32, shared, stream>>>(p, ep, count, tasks, n, k, persistent);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int Type>
void dispatch(torch::Tensor out, torch::Tensor input, torch::Tensor offsets,
              torch::Tensor wp, torch::Tensor sp, torch::Tensor plan,
              const int* indices, int experts, int routes, int tile, int grid,
              cudaStream_t stream) {
#define CELL(ID, M, N) case ID: launch<Type, M, N>(out, input, offsets, wp, sp, plan, indices, experts, routes, grid, stream); return
  switch (tile) {
    CELL(0, 32, 64); CELL(1, 64, 64); CELL(2, 128, 64);
    CELL(3, 32, 128); CELL(4, 64, 128); CELL(5, 128, 128);
  }
#undef CELL
  TORCH_CHECK(false, "prefill IQ tile must be 0..5");
}

void gemm(torch::Tensor out, torch::Tensor input, torch::Tensor offsets,
          torch::Tensor wp, torch::Tensor sp, torch::Tensor plan,
          int64_t codec, int64_t token_count, int64_t tile, int64_t grid,
          std::optional<torch::Tensor> indices) {
  TORCH_CHECK(token_count > 6, "prefill IQ forbids decode/verify M1..6");
  TORCH_CHECK(codec == 29 || codec == 16 || codec == 17, "prefill IQ codec must be IQ1_M/IQ2_XXS/IQ2_XS");
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kFloat16 &&
              input.dim() == 2 && input.stride(1) == 1,
              "prefill IQ input must be CUDA FP16 rows");
  for (const auto& t : {out, offsets, wp, sp, plan})
    TORCH_CHECK(t.device() == input.device(), "prefill IQ device mismatch");
  TORCH_CHECK(out.scalar_type() == torch::kFloat16 && out.dim() == 2 &&
              out.stride(1) == 1 && out.stride(0) >= out.size(1), "prefill IQ output rows");
  TORCH_CHECK(offsets.scalar_type() == torch::kInt32 && offsets.dim() == 1 &&
              offsets.is_contiguous() && offsets.numel() >= 2 && offsets.numel() <= 33,
              "prefill IQ offsets require 1..32 experts");
  const int64_t experts = offsets.numel() - 1;
  for (const auto& t : {wp, sp})
    TORCH_CHECK(t.scalar_type() == torch::kUInt8 && t.is_contiguous() &&
                t.numel() == experts * 16, "prefill IQ prepared pointer table size/type");
  TORCH_CHECK(tile >= 0 && tile < 6 && (grid == 0 || grid == 80 || grid == 160),
              "prefill IQ tile/grid unsupported");
  const int64_t routes = out.size(0);
  const int64_t n = out.size(1), k = input.size(1);
  TORCH_CHECK((n == 1152 && k == 5120) || (n == 1280 && k == 2304),
              "prefill IQ only accepts V4.1 TP4 gate/up or row-split down");
  TORCH_CHECK(routes <= 6 * 16384 && input.size(0) <= 6 * 16384 &&
              input.stride(0) >= k && input.stride(0) <= INT_MAX &&
              out.stride(0) <= INT_MAX, "prefill IQ capacity/stride invalid");
  const int bm = 32 << (tile % 3);
  const int64_t capacity = routes / bm + experts;
  TORCH_CHECK(plan.scalar_type() == torch::kInt32 && plan.dim() == 1 &&
              plan.is_contiguous() && plan.numel() >= 1 + capacity * 4,
              "prefill IQ member plan capacity/type invalid");
  TORCH_CHECK(!out.is_alias_of(input), "prefill IQ input/output storage alias");
  for (const auto& t : {out, input, offsets, wp, sp})
    TORCH_CHECK(!plan.is_alias_of(t), "prefill IQ plan storage alias");
  const int* idx = nullptr;
  if (indices) {
    TORCH_CHECK(n == 1152 && input.size(0) == token_count &&
                indices->device() == input.device() &&
                indices->scalar_type() == torch::kInt32 && indices->dim() == 1 &&
                indices->is_contiguous() && indices->numel() == routes,
                "prefill IQ gate/up requires matching CUDA int32 route indices");
    idx = indices->data_ptr<int>();
    TORCH_CHECK(!plan.is_alias_of(*indices), "prefill IQ plan/index storage alias");
  } else {
    TORCH_CHECK(n == 1280 && input.size(0) == routes,
                "prefill IQ down requires contiguous route rows");
  }
  const c10::cuda::CUDAGuard guard(input.device());
  TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major == 7 &&
              at::cuda::getCurrentDeviceProperties()->minor == 0, "prefill IQ requires SM70");
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (routes == 0) { plan.zero_(); return; }
  switch (codec) {
#define CODEC(T) case T: dispatch<T>(out,input,offsets,wp,sp,plan,idx,experts,routes,tile,grid,stream); return
    CODEC(29); CODEC(16); CODEC(17);
#undef CODEC
  }
  TORCH_CHECK(false, "prefill IQ codec must be IQ1_M/IQ2_XXS/IQ2_XS");
}

template<int Type, int BM, int BN>
void resource_cell(std::vector<int64_t>& values, int tile) {
  using Kernel = typename Configuration<Type, BM, BN>::Kernel;
  constexpr int shared = sizeof(typename Kernel::SharedStorage);
  constexpr int threads = Kernel::WARP_CNT * 32;
  auto fn = compact_gemm<Type, BM, BN>;
  if constexpr (shared > 48 * 1024)
    C10_CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
  cudaFuncAttributes attrs{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attrs, fn));
  int blocks = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, fn, threads, shared));
  values.insert(values.end(), {Type, tile, threads, shared, attrs.numRegs, blocks});
}

std::vector<int64_t> resources(torch::Tensor device_anchor) {
  TORCH_CHECK(device_anchor.is_cuda(), "resource query requires actual CUDA device");
  const c10::cuda::CUDAGuard guard(device_anchor.device());
  TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major == 7 &&
              at::cuda::getCurrentDeviceProperties()->minor == 0, "resource query requires SM70");
  std::vector<int64_t> values;
#define RES(T) resource_cell<T,32,64>(values,0); resource_cell<T,64,64>(values,1); resource_cell<T,128,64>(values,2); resource_cell<T,32,128>(values,3); resource_cell<T,64,128>(values,4); resource_cell<T,128,128>(values,5)
  RES(29); RES(16); RES(17);
#undef RES
  return values;
}
}  // namespace prefmoe

TORCH_LIBRARY(v41prefmoe, m) {
  m.def("gemm(Tensor(a!) out, Tensor input, Tensor offsets, Tensor weights, Tensor stats, Tensor(b!) plan, int codec, int token_count, int tile, int grid, Tensor? indices) -> ()", &prefmoe::gemm);
  m.def("resources(Tensor device_anchor) -> int[]", &prefmoe::resources);
}
