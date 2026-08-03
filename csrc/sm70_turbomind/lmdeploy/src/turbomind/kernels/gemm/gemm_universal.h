// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <climits>

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/data_type.h"
#include "src/turbomind/kernels/core/layout.h"
#include "src/turbomind/kernels/core/math.h"

#include "src/turbomind/kernels/gemm/cta_map.h"
#include "src/turbomind/kernels/gemm/desc.h"
#include "src/turbomind/kernels/gemm/epilogue.h"
#include "src/turbomind/kernels/gemm/thread_map.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"

namespace turbomind::gemm {

struct GemmParam {
  MatrixParam a;
  MatrixParam b;
  MatrixParam u;
  MatrixParam v;
};

template <class Op>
__inline__ __device__ MatrixData resolve_op(const MatrixParam& param,
                                            int gemm_id) {
  return resolve<typename Op::Dtype, Op::GmemIter::kMode>(param, gemm_id);
}

template <class Arch_, class Mainloop, class Epilogue_, class Scheduler_>
struct GemmUniversal {
  // using Impl = typename Mainloop::Impl;
  using Impl = Mainloop;

  using Ta = typename Impl::Ta;
  using Tb = typename Impl::Tb;
  using Tu = typename Impl::Tu;
  using Tv = typename Impl::Tv;

  using Arch = Arch_;
  using Scheduler = Scheduler_;
  using Epilogue = Epilogue_;

  using Tc = typename Epilogue::Tc;

  // col major == M-major (A)
  // row major == N-major (B)
  static constexpr Order kOrderC = Epilogue::kOrder;

  static constexpr int CTA_M = Impl::CTA_M;
  static constexpr int CTA_N = Impl::CTA_N;
  static constexpr int CTA_K = Impl::CTA_K;

  static constexpr bool kDynamicSched = Scheduler::group_axis >= 0;
  static constexpr bool kSplitK = Epilogue::SplitK;

  using FragC = typename Impl::FragC;

  static constexpr int WARP_CNT = Impl::WARPS;

  using OperandA = typename Mainloop::OperandA;
  using OperandB = typename Mainloop::OperandB;
  using OperandU = typename Mainloop::OperandU;
  using OperandV = typename Mainloop::OperandV;

  static constexpr int kChunkSizeK =
      std::max(CTA_K, std::max(OperandU::kGroupSize, OperandV::kGroupSize));

  static constexpr int kGSizeU = OperandU::kGroupSize;
  static constexpr int kGSizeV = OperandV::kGroupSize;

  struct SharedStorage {
    union {
      typename Mainloop::SharedStorage mainloop;
      typename Epilogue::SharedStorage epilogue;
    };
    typename Scheduler::SharedStorage sched;
  };

  static constexpr Order kOrderA = OperandA::kOrder;
  static constexpr Order kOrderB = OperandB::kOrder;
  static constexpr Order kOrderU = OperandU::kOrder;
  static constexpr Order kOrderV = OperandV::kOrder;

  static constexpr Pack kPackA = OperandA::kPack;
  static constexpr Pack kPackB = OperandB::kPack;

  using Param = GemmParam;

  __device__ void operator()(const Param& param, const EpilogueParam& epi_param,
                             Scheduler& sched, char* smem_buf) {
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    typename Scheduler::Tile tile;

    if (!sched.init(tile, storage.sched, std::false_type{})) {
      return;
    }

    const auto& [M, N, K] = tile.shape.__a;

    const auto tile_id = tile.tile_id;

    const int offset_m = tile_id[0] * CTA_M;
    const int offset_n = tile_id[1] * CTA_N;

    const int offset_k = tile.k_iters[0] * CTA_K;

    if (offset_m >= M || offset_n >= N || offset_k >= K) {  // empty tile
      return;
    }

    const int extent_m = min(CTA_M, M - offset_m);
    const int extent_n = min(CTA_N, N - offset_n);

    // Is 8 enough?
    __align__(8) FragC frag_C{};

    int tile_iter = tile.k_iters[1];

    const int g = tile.group_id;

    const auto mat_A = resolve_op<OperandA>(param.a, g);
    const auto mat_B = resolve_op<OperandB>(param.b, g);
    const auto mat_U = resolve_op<OperandU>(param.u, g);
    const auto mat_V = resolve_op<OperandV>(param.v, g);

    typename OperandA::GmemIter gmem_A{
        mat_A, {offset_m, offset_k}, {extent_m, CTA_K}};
    typename OperandB::GmemIter gmem_B{
        mat_B, {offset_n, offset_k}, {extent_n, CTA_K}};

    const int2 offset_U{offset_m, cdiv(offset_k, kGSizeU)},
        extent_U{extent_m, cdiv(CTA_K, kGSizeU)};
    typename OperandU::GmemIter gmem_U{mat_U, offset_U, extent_U};

    const int2 offset_V{offset_n, cdiv(offset_k, kGSizeV)},
        extent_V{extent_n, cdiv(CTA_K, kGSizeV)};
    typename OperandV::GmemIter gmem_V{mat_V, offset_V, extent_V};

    Mainloop mainloop{};
    mainloop(gmem_A, gmem_B, gmem_U, gmem_V, frag_C, tile_iter,
             storage.mainloop);

    {
      sched.init(tile, storage.sched, std::true_type{});

      const auto [M, N, K] = tile.shape.__a;

      int4 tile_offset{tile.tile_id[0], tile.tile_id[1], tile.tile_id[2],
                       tile.group_id};

      const int2 extents = {min(CTA_M, M - tile_offset.x * CTA_M),
                            min(CTA_N, N - tile_offset.y * CTA_N)};

      const bool is_last = (tile.k_iters[0] + tile.k_iters[1]) * CTA_K == K;

      Epilogue epilogue{};
      epilogue(frag_C,  //
               tile_offset, extents, sched.tiles_[2], tile.linear_tile_id,
               is_last, epi_param, storage.epilogue);
    }
  }
};

extern __shared__ char smem_buf[];

__device__ __forceinline__ bool TryRunTileAllReduceReducerCta(
    const EpilogueParam& epi_param) {
  if (!epi_param.tile_allreduce) {
    return false;
  }

  const auto& tile_ar = epi_param.tile_allreduce_param;
  if (tile_ar.kernel_reducer_blocks <= 0) {
    return false;
  }
  if ((int)blockIdx.z != tile_ar.producer_grid_z) {
    return false;
  }

  if (tile_ar.world_size != 2 || tile_ar.rank_data == nullptr ||
      tile_ar.output == nullptr || tile_ar.self_signal == nullptr ||
      tile_ar.tile_numel <= 0 || tile_ar.output_numel <= 0 ||
      (tile_ar.tile_numel & 1) || (tile_ar.output_numel & 1) ||
      tile_ar.producer_grid_x <= 0 || tile_ar.producer_grid_y <= 0) {
    return true;
  }

  const int reducer_id =
      (int)blockIdx.y * tile_ar.producer_grid_x + (int)blockIdx.x;
  if (reducer_id >= tile_ar.kernel_reducer_blocks) {
    return true;
  }

  const int tile_count =
      (tile_ar.output_numel + tile_ar.tile_numel - 1) / tile_ar.tile_numel;
  if (tile_count <= 0 || tile_count > vllm::sm70_tile_runtime::kMaxBlocks) {
    return true;
  }

  const int tid = threadIdx.x;
  auto* self_signal =
      reinterpret_cast<vllm::sm70_tile_runtime::Signal*>(tile_ar.self_signal);
  const auto rank_data =
      *reinterpret_cast<const vllm::sm70_tile_runtime::RankData*>(
          tile_ar.rank_data);
  const auto* rank0 = reinterpret_cast<const half2*>(rank_data.ptrs[0]);
  const auto* rank1 = reinterpret_cast<const half2*>(rank_data.ptrs[1]);
  auto* output = reinterpret_cast<half2*>(tile_ar.output);

  for (int tile_id = reducer_id; tile_id < tile_count;
       tile_id += tile_ar.kernel_reducer_blocks) {
    const int elem_begin = tile_id * tile_ar.tile_numel;
    const int elem_end =
        min(elem_begin + tile_ar.tile_numel, tile_ar.output_numel);
    const int h2_begin = elem_begin / 2;
    const int h2_end = elem_end / 2;
    const auto flag = self_signal->_flag[tile_id] + 1;

    if (tid < tile_ar.world_size) {
      auto* self_flag = &self_signal->start[tile_id][tid];
      while (vllm::sm70_tile_runtime::load_flag_sys_visible(self_flag) !=
             flag) {
      }
    }

    __syncthreads();

    for (int idx = h2_begin + tid; idx < h2_end; idx += blockDim.x) {
      output[idx] = __hadd2(rank0[idx], rank1[idx]);
    }

    __syncthreads();
    if (tid == 0) {
      self_signal->_flag[tile_id] = flag;
    }
  }

  return true;
}

template <class Kernel, class Param, class EpilogueParam, class Scheduler>
__global__ void gemm_kernel(Param param, EpilogueParam epi_param,
                            Scheduler sched) {
#if __CUDA_ARCH__
  if constexpr (Kernel::Arch::is_compatible(__CUDA_ARCH__)) {
    if (TryRunTileAllReduceReducerCta(epi_param)) {
      return;
    }
    Kernel kernel;
    kernel(param, epi_param, sched, smem_buf);
  }
#endif
}

}  // namespace turbomind::gemm
