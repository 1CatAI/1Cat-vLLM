#undef FLASH_NAMESPACE
#define FLASH_NAMESPACE onecat_v37_tail
/******************************************************************************
 * Copyright (c) 2026, 1CatAI.
 ******************************************************************************/

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/nn/functional.h>

#include <limits>
#include <type_traits>

#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>

#include "namespace_config.h"
#include "kernel_traits.h"
#include "utils.h"
#include "softmax.h"
#include "mask.h"

namespace FLASH_NAMESPACE {

using namespace cute;

struct Sm70D256SplitDTraits {
  using Element = cutlass::half_t;
  using MmaAtom = MMA_Atom<SM70_8x8x4_F32F16F16F32_TN>;
  using PvMmaAtom = MMA_Atom<SM70_8x8x4_F32F16F16F32_TT>;

  static constexpr int kHeadDim = 256;
  static constexpr int kBlockM = 64;
  static constexpr int kBlockN = 32;
  static constexpr int kDChunk = 64;
  static constexpr int kDChunks = kHeadDim / kDChunk;
  static constexpr int kOwnedDChunks = kDChunks / 2;
  static constexpr int kNThreads = 256;
  static constexpr int kMmaThreads = 32;
  static constexpr int kWarpsPerGroup = 2;
  static constexpr int kMmaGroups = kNThreads / (kWarpsPerGroup * kMmaThreads);
  static constexpr int kGroupRows = kBlockM / kMmaGroups;
  static constexpr int kQkWarpRows = kGroupRows / kWarpsPerGroup;
  static constexpr int kQkRowsPerThread = kQkWarpRows / 4;
  static constexpr int kOutputRowsPerThread = kGroupRows / 4;

  // Each warp in a pair owns eight distinct Q rows for QK, then the pair
  // shares the resulting P tile and each warp owns D/2 for PV. This keeps
  // the standard FA2 N32 online-softmax order without duplicating QK work.
  using QkTiledMma = TiledMMA<MmaAtom, Layout<Shape<_1, _4, _1>>,
                              Tile<Int<kQkWarpRows>, Int<kBlockN / 4>, _4>>;
  using PvTiledMma = TiledMMA<PvMmaAtom, Layout<Shape<_1, _4, _1>>,
                              Tile<Int<kGroupRows>, Int<kDChunk / 4>, _4>>;
  static_assert(decltype(size(QkTiledMma{}))::value == kMmaThreads);
  static_assert(decltype(size(PvTiledMma{}))::value == kMmaThreads);

  using SmemLayoutAtom = decltype(composition(
      Swizzle<3, 3, 3>{}, Layout<Shape<_8, _64>, Stride<_64, _1>>{}));
  using SmemLayoutQ = decltype(tile_to_shape(
      SmemLayoutAtom{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));
  using SmemLayoutKV = decltype(tile_to_shape(
      SmemLayoutAtom{}, Shape<Int<kBlockN>, Int<kDChunk>>{}));
  // Volta HMMA.884 assigns one warp to four 8-thread quadpairs and services
  // the 64-bit operand loads as two half-warps. Pitch 68 advances each row
  // by one bank pair; the extra 16-half phase every 16 rows folds row bit 4
  // into bank-pair bit 2 so every half-warp covers all 16 bank pairs once.
  using SmemLayoutK = Layout<
      Shape<Shape<Int<16>, Int<2>>, Int<kDChunk>>,
      Stride<Stride<Int<kDChunk + 4>, Int<16 * (kDChunk + 4) + 16>>, _1>>;
  // TT PV consumes V as KxD. Row bit 1 is folded into D-address bit 2;
  // the producer applies the inverse 64-bit-half swap before STS.128.
  using SmemLayoutV =
      Layout<Shape<Shape<_2, Int<kBlockN / 2>>, Shape<_32, Int<kDChunk / 32>>>,
             Stride<Stride<_32, Int<4 * kBlockN>>, Stride<_1, _64>>>;
  using SmemLayoutP =
      Layout<Shape<Int<kBlockM>, Int<kBlockN>>, Stride<Int<kBlockN>, _1>>;

  using SmemCopyAtom =
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>;
  using SmemCopyAtomTransposed = SmemCopyAtom;

  static constexpr int kGmemElemsPerLoad = 8;
  static constexpr int kGmemThreadsPerRow = kDChunk / kGmemElemsPerLoad;
  using GmemLayoutAtom = Layout<
      Shape<Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
      Stride<Int<kGmemThreadsPerRow>, _1>>;
  using GmemTiledCopy =
      decltype(make_tiled_copy(Copy_Atom<SM70_LDG_GLOBAL_CG_128b, Element>{},
                               GmemLayoutAtom{}, Layout<Shape<_1, _8>>{}));
  static constexpr int kGmemRowsPerThread =
      kBlockN / (kNThreads / kGmemThreadsPerRow);
  using GmemTiledCopyPaged = decltype(make_tiled_copy(
      Copy_Atom<SM70_LDG_GLOBAL_CG_128b, Element>{}, GmemLayoutAtom{},
      Layout<Shape<Int<kGmemRowsPerThread>, _8>, Stride<_8, _1>>{}));

  static constexpr int kGmemKElemsPerLoad = 4;
  static constexpr int kGmemKThreadsPerRow = kDChunk / kGmemKElemsPerLoad;
  using GmemKLayoutAtom = Layout<
      Shape<Int<kNThreads / kGmemKThreadsPerRow>, Int<kGmemKThreadsPerRow>>,
      Stride<Int<kGmemKThreadsPerRow>, _1>>;
  using GmemKTiledCopy =
      decltype(make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{},
                               GmemKLayoutAtom{}, Layout<Shape<_1, _4>>{}));
  static constexpr int kGmemKRowsPerThread =
      kBlockN / (kNThreads / kGmemKThreadsPerRow);
  using GmemKTiledCopyPaged = decltype(make_tiled_copy(
      Copy_Atom<UniversalCopy<uint64_t>, Element>{}, GmemKLayoutAtom{},
      Layout<Shape<Int<kGmemKRowsPerThread>, _4>, Stride<_4, _1>>{}));

  static constexpr int kQElements = size(SmemLayoutQ{});
  static constexpr int kKVElements = size(SmemLayoutKV{});
  static_assert(size(SmemLayoutV{}) == kKVElements);
  // The second K stage is live only before P is materialized, so it may
  // alias the beginning of the later P region. Keep the K stages disjoint
  // and on the same 128-byte bank phase without moving the V/P regions.
  static constexpr int kKStageElements = 2240;
  static_assert(kKStageElements >= cosize(SmemLayoutK{}));
  static_assert(kKStageElements + cosize(SmemLayoutK{}) <=
                2 * kKVElements + size(SmemLayoutP{}));
  static constexpr int kPElements = size(SmemLayoutP{});
  static constexpr int kExchangeRows = kMmaGroups * kGroupRows;
  static constexpr int kTensorSmemBytes =
      (kQElements + 2 * kKVElements + kPElements) * sizeof(Element);
  static constexpr int kExchangeBytes = 2 * kExchangeRows * sizeof(float);
  static constexpr int kSmemBytes = kTensorSmemBytes + kExchangeBytes;
  static_assert(kSmemBytes == 45568);
};

template <typename TiledCopy, typename SrcTensor, typename DstTensor>
__device__ __forceinline__ void copy_even_tile(TiledCopy tiled_copy,
                                               const SrcTensor& src,
                                               DstTensor& dst) {
  static_assert(decltype(rank(src))::value == 3);
  static_assert(decltype(rank(dst))::value == 3);
#pragma unroll
  for (int m = 0; m < size<1>(src); ++m) {
#pragma unroll
    for (int k = 0; k < size<2>(src); ++k) {
      cute::copy(tiled_copy, src(_, m, k), dst(_, m, k));
    }
  }
}

template <typename RegTensor, typename SmemTensor, typename CoordTensor>
__device__ __forceinline__ void store_v_fragment_128_swizzled(
    const RegTensor& source, SmemTensor& destination,
    const CoordTensor& coordinates) {
  static_assert(decltype(size<0>(source))::value == 8);
  static_assert(decltype(size<0>(destination))::value == 8);
  static_assert(decltype(size<1>(source))::value ==
                decltype(size<1>(destination))::value);
  static_assert(decltype(size<2>(source))::value ==
                decltype(size<2>(destination))::value);
#pragma unroll
  for (int k = 0; k < size<2>(source); ++k) {
#pragma unroll
    for (int m = 0; m < size<1>(source); ++m) {
      auto words = recast<uint32_t const>(source(_, m, k));
      const uint32_t address = static_cast<uint32_t>(
          __cvta_generic_to_shared(&destination(0, m, k)));
      const int row = get<0>(coordinates(0, m, k));
      if (row & 2) {
        asm volatile(
            "st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n" ::"r"(address),
            "r"(words(2)), "r"(words(3)), "r"(words(0)), "r"(words(1)));
      } else {
        asm volatile(
            "st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n" ::"r"(address),
            "r"(words(0)), "r"(words(1)), "r"(words(2)), "r"(words(3)));
      }
    }
  }
}

template <typename SmemTensor, typename TensorB>
__device__ __forceinline__ void load_v_fragment_tt(const SmemTensor& sV,
                                                   TensorB& b_words, int phase,
                                                   int lane) {
  auto* v = sV.data().get();
  const int d_lane = ((lane & 0x0c) << 1) | ((lane & 0x10) >> 2);
  const int k = phase * 4 + (lane & 0x03);
  const int offset =
      (d_lane ^ ((k & 0x02) << 1)) | ((k & 0x01) << 5) | ((k & 0x1e) << 6);
  const uint32_t address =
      static_cast<uint32_t>(__cvta_generic_to_shared(v + offset));
  uint32_t word0;
  uint32_t word1;
  uint32_t word2;
  uint32_t word3;
  asm volatile(
      "ld.shared.v2.u32 {%0, %1}, [%4];\n"
      "ld.shared.v2.u32 {%2, %3}, [%4+128];\n"
      : "=r"(word0), "=r"(word1), "=r"(word2), "=r"(word3)
      : "r"(address));
  b_words(0, 0) = word0;
  b_words(1, 0) = word1;
  b_words(0, 1) = word2;
  b_words(1, 1) = word3;
}

template <int kPhase, typename TensorO, typename TensorP, typename SmemTensor,
          typename TiledMma, typename TensorB, typename TensorBWords,
          typename TensorBNext, typename TensorBNextWords>
__device__ __forceinline__ void splitd_pv_gemm_tt_phase(
    TensorO& acc_o, const TensorP& tPrP, const SmemTensor& sV,
    TensorB& current_b, TensorBWords& current_b_words, TensorBNext& next_b,
    TensorBNextWords& next_b_words, TiledMma tiled_mma, int lane) {
  constexpr int kPhases = decltype(size<2>(tPrP))::value;
  static_assert(kPhase < kPhases);
  if constexpr (kPhase + 1 < kPhases) {
    load_v_fragment_tt(sV, next_b_words, kPhase + 1, lane);
  }
  cute::gemm(tiled_mma, tPrP(_, _, kPhase), current_b, acc_o);
  if constexpr (kPhase + 1 < kPhases) {
    splitd_pv_gemm_tt_phase<kPhase + 1>(acc_o, tPrP, sV, next_b, next_b_words,
                                        current_b, current_b_words, tiled_mma,
                                        lane);
  }
}

template <typename TensorO, typename TensorP, typename SmemTensor,
          typename TiledMma>
__device__ __forceinline__ void splitd_pv_gemm_tt(TensorO& acc_o,
                                                  const TensorP& tPrP,
                                                  const SmemTensor& sV,
                                                  TiledMma tiled_mma,
                                                  int lane) {
  using Element = typename Sm70D256SplitDTraits::Element;
  using BLayout = Layout<Shape<_4, _2>, Stride<_1, _4>>;
  auto b0 = make_tensor<Element>(BLayout{});
  auto b1 = make_tensor<Element>(BLayout{});
  auto b0_words = recast<uint32_t>(b0);
  auto b1_words = recast<uint32_t>(b1);
  static_assert(decltype(size<2>(tPrP))::value ==
                Sm70D256SplitDTraits::kBlockN / 4);
  static_assert(decltype(size(b0))::value == 8);
  static_assert(decltype(size<0>(b0_words))::value == 2);
  static_assert(decltype(size<1>(b0_words))::value == 2);
  load_v_fragment_tt(sV, b0_words, 0, lane);
  splitd_pv_gemm_tt_phase<0>(acc_o, tPrP, sV, b0, b0_words, b1, b1_words,
                             tiled_mma, lane);
}

template <bool PagedKV, typename Tensor>
__device__ __forceinline__ auto reshape_kv_thread_tensor(Tensor tensor) {
  if constexpr (PagedKV) {
    return make_tensor(tensor.data(), reshape_thread_tile(tensor.layout()));
  } else {
    return tensor;
  }
}

template <typename TensorScores, typename OLayout, int kOElements>
__device__ __forceinline__ void splitd_n32_online_softmax(
    TensorScores& acc_s,
    float (&o_storage)[Sm70D256SplitDTraits::kOwnedDChunks][kOElements],
    OLayout o_layout, float (&row_max)[Sm70D256SplitDTraits::kQkRowsPerThread],
    float (&row_sum)[Sm70D256SplitDTraits::kQkRowsPerThread],
    float* row_scale_exchange, int mma_group, int n_warp, int lane,
    float softmax_scale_log2, bool first_tile) {
  using Traits = Sm70D256SplitDTraits;
  auto scores = make_tensor(
      acc_s.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_s.layout()));
  static_assert(decltype(size<0>(scores))::value == Traits::kQkRowsPerThread);

  float work[Traits::kQkRowsPerThread];
  auto work_tensor = make_tensor(make_rmem_ptr(&work[0]),
                                 Shape<Int<Traits::kQkRowsPerThread>>{});
  if (first_tile) {
    FLASH_NAMESPACE::sm70_reduce_max<true>(scores, work_tensor);
  } else {
#pragma unroll
    for (int slot = 0; slot < Traits::kQkRowsPerThread; ++slot) {
      work[slot] = row_max[slot];
    }
    FLASH_NAMESPACE::sm70_reduce_max<false>(scores, work_tensor);
  }

  float scores_max[Traits::kQkRowsPerThread];
#pragma unroll
  for (int slot = 0; slot < Traits::kQkRowsPerThread; ++slot) {
    const float next_max = work[slot];
    const float safe_max = next_max == -INFINITY ? 0.0f : next_max;
    work[slot] = first_tile
                     ? 1.0f
                     : exp2f((row_max[slot] - safe_max) * softmax_scale_log2);
    row_max[slot] = next_max;
    scores_max[slot] = safe_max;
    if (!first_tile) {
      row_sum[slot] *= work[slot];
    }
  }

  if ((lane & 0x0e) == 0) {
#pragma unroll
    for (int slot = 0; slot < Traits::kQkRowsPerThread; ++slot) {
      const int row =
          FLASH_NAMESPACE::sm70_row_slot<Traits::kQkWarpRows>(slot, lane);
      row_scale_exchange[mma_group * Traits::kGroupRows +
                         n_warp * Traits::kQkWarpRows + row] = work[slot];
    }
  }
  __syncthreads();

  if (!first_tile) {
#pragma unroll
    for (int d = 0; d < Traits::kOwnedDChunks; ++d) {
      auto acc_o = make_tensor(make_rmem_ptr(&o_storage[d][0]), o_layout);
      auto acc_o_rc = make_tensor(
          acc_o.data(),
          FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_o.layout()));
#pragma unroll
      for (int row = 0; row < Traits::kOutputRowsPerThread; ++row) {
        const int logical_row =
            FLASH_NAMESPACE::sm70_row_slot<Traits::kGroupRows>(row, lane);
        const float row_scale =
            row_scale_exchange[mma_group * Traits::kGroupRows + logical_row];
#pragma unroll
        for (int col = 0; col < size<1>(acc_o_rc); ++col) {
          acc_o_rc(row, col) *= row_scale;
        }
      }
    }
  }

  auto scores_max_tensor = make_tensor(make_rmem_ptr(&scores_max[0]),
                                       Shape<Int<Traits::kQkRowsPerThread>>{});
  FLASH_NAMESPACE::sm70_scale_apply_exp2(scores, scores_max_tensor,
                                         softmax_scale_log2);

  FLASH_NAMESPACE::sm70_reduce_sum<true>(scores, work_tensor);
#pragma unroll
  for (int slot = 0; slot < Traits::kQkRowsPerThread; ++slot) {
    row_sum[slot] += work[slot];
  }
}

template <int kThreadsPerRow, int kRowsPerThread, int kElemsPerLoad>
__device__ __forceinline__ int64_t
paged_kv_thread_offset(int tid, int n_block, int d_chunk, int page_size,
                       const int* __restrict__ block_table, int64_t page_stride,
                       int64_t row_stride) {
  const int row_in_tile = (tid / kThreadsPerRow) * kRowsPerThread;
  const int logical_row = n_block * Sm70D256SplitDTraits::kBlockN + row_in_tile;
  const int physical_page = block_table[logical_row / page_size];
  return static_cast<int64_t>(physical_page) * page_stride +
         static_cast<int64_t>(logical_row % page_size) * row_stride +
         d_chunk * Sm70D256SplitDTraits::kDChunk +
         (tid % kThreadsPerRow) * kElemsPerLoad;
}

template <typename Element, bool PagedKV, bool SplitKV3 = false,
          bool StoreState = false, bool FloatOutput = false,
          bool Unnormalized = false, bool RecoverScoreTiles = false>
__global__ __launch_bounds__(
    Sm70D256SplitDTraits::kNThreads,
    1) void sm70_d256_splitd_dense_kernel(const Element* __restrict__ q,
                                          const Element* __restrict__ k,
                                          const Element* __restrict__ v,
                                          std::conditional_t<
                                              FloatOutput, float,
                                              Element>* __restrict__ out,
                                          int q_batch_stride, int q_row_stride,
                                          int q_head_stride, int k_outer_stride,
                                          int k_row_stride, int k_head_stride,
                                          int v_outer_stride, int v_row_stride,
                                          int v_head_stride, int query_len,
                                          int kv_len, int heads_q, int heads_kv,
                                          float softmax_scale_log2,
                                          const int* __restrict__ block_table,
                                          int page_size,
                                          int block_table_batch_stride,
                                          float* __restrict__ partial_out,
                                          float* __restrict__ partial_max,
                                          float* __restrict__ partial_sum,
                                          const int* recovery_tiles = nullptr) {
  using Traits = Sm70D256SplitDTraits;
  constexpr int kBlockM = Traits::kBlockM;
  constexpr int kBlockN = Traits::kBlockN;
  constexpr int kDChunk = Traits::kDChunk;
  if constexpr (RecoverScoreTiles) {
    if (!recovery_tiles[blockIdx.x]) return;
  }

  const int tid = threadIdx.x;
  const int warp = tid / Traits::kMmaThreads;
  const int mma_group = warp / Traits::kWarpsPerGroup;
  const int lane = tid % Traits::kMmaThreads;
  const int m_block = blockIdx.x;
  const int split = SplitKV3 ? blockIdx.y % 3 : 0;
  const int batch = SplitKV3 ? blockIdx.y / 3 : blockIdx.y;
  const int head_q = blockIdx.z;
  const int head_kv = head_q / (heads_q / heads_kv);
  const int query_row_base = m_block * kBlockM;
  const int kv_offset = kv_len - query_len;
  const int* sequence_block_table =
      PagedKV ? block_table + batch * block_table_batch_stride : nullptr;

  extern __shared__ __align__(128) Element smem[];
  Element* q_smem_ptr = smem;
  Element* kv_smem_ptr = q_smem_ptr + Traits::kQElements;
  auto sQ =
      make_tensor(make_smem_ptr(q_smem_ptr), typename Traits::SmemLayoutQ{});

  typename Traits::GmemTiledCopy gmem_copy;
  auto gmem_thread = gmem_copy.get_thread_slice(tid);
  using GmemVCopy =
      std::conditional_t<PagedKV, typename Traits::GmemTiledCopyPaged,
                         typename Traits::GmemTiledCopy>;
  GmemVCopy gmem_v_copy;
  auto gmem_v_thread = gmem_v_copy.get_thread_slice(tid);
  using GmemKCopy =
      std::conditional_t<PagedKV, typename Traits::GmemKTiledCopyPaged,
                         typename Traits::GmemKTiledCopy>;
  GmemKCopy gmem_k_copy;
  auto gmem_k_thread = gmem_k_copy.get_thread_slice(tid);
  {
    const int64_t q_batch_offset = static_cast<int64_t>(batch) * q_batch_stride;
    auto mQ =
        make_tensor(make_gmem_ptr(q + q_batch_offset + head_q * q_head_stride),
                    make_shape(query_len, Int<Traits::kHeadDim>{}),
                    make_stride(q_row_stride, _1{}));
    auto gQ = local_tile(mQ, Shape<Int<kBlockM>, Int<Traits::kHeadDim>>{},
                         make_coord(m_block, 0));
    auto tQgQ = gmem_thread.partition_S(gQ);
    auto tQsQ = gmem_thread.partition_D(sQ);
    copy_even_tile(gmem_copy, tQgQ, tQsQ);
  }
  __syncthreads();

  typename Traits::QkTiledMma qk_tiled_mma;
  auto qk_mma_thread = qk_tiled_mma.get_thread_slice(lane);
  typename Traits::PvTiledMma pv_tiled_mma;
  auto pv_mma_thread = pv_tiled_mma.get_thread_slice(lane);

  using OFragment = decltype(partition_fragment_C(
      pv_tiled_mma, Shape<Int<Traits::kGroupRows>, Int<kDChunk>>{}));
  constexpr int kOElements = decltype(size(OFragment{}))::value;
  using OLayout = typename OFragment::layout_type;
  float o_storage[Traits::kOwnedDChunks][kOElements];
#pragma unroll
  for (int d = 0; d < Traits::kOwnedDChunks; ++d) {
#pragma unroll
    for (int i = 0; i < kOElements; ++i) {
      o_storage[d][i] = 0.0f;
    }
  }

  float row_max[Traits::kQkRowsPerThread];
  float row_sum[Traits::kQkRowsPerThread];
#pragma unroll
  for (int row = 0; row < Traits::kQkRowsPerThread; ++row) {
    row_max[row] = -INFINITY;
    row_sum[row] = 0.0f;
  }

  const int max_kv_for_tile = query_row_base + kBlockM + kv_offset;
  const int n_block_limit = max_kv_for_tile < kv_len ? max_kv_for_tile : kv_len;
  const int visible_n_blocks = cute::ceil_div(n_block_limit, kBlockN);
  int n_block_min = 0;
  int n_block_max = visible_n_blocks - 1;
  if constexpr (SplitKV3) {
    n_block_min = visible_n_blocks * split / 3;
    n_block_max = visible_n_blocks * (split + 1) / 3 - 1;
  }

  const int64_t k_batch_offset =
      PagedKV ? 0 : static_cast<int64_t>(batch) * k_outer_stride;
  auto mK =
      make_tensor(make_gmem_ptr(k + k_batch_offset + head_kv * k_head_stride),
                  make_shape(kv_len, Int<Traits::kHeadDim>{}),
                  make_stride(k_row_stride, _1{}));
  auto sKFirst =
      make_tensor(make_smem_ptr(kv_smem_ptr), typename Traits::SmemLayoutK{});
  auto tKsKFirstRaw = gmem_k_thread.partition_D(sKFirst);
  auto tKsKFirst = reshape_kv_thread_tensor<PagedKV>(tKsKFirstRaw);
  auto tKrKNext = make_fragment_like(tKsKFirst);
  auto gKFirst = local_tile(mK, Shape<Int<kBlockN>, Int<kDChunk>>{},
                            make_coord(n_block_max, 0));
  auto tKgKFirstRaw = gmem_k_thread.partition_S(gKFirst);
  auto tKgKFirst = reshape_kv_thread_tensor<PagedKV>(tKgKFirstRaw);
  int64_t k_thread_tile_base = 0;
  if constexpr (PagedKV) {
    k_thread_tile_base = paged_kv_thread_offset<Traits::kGmemKThreadsPerRow,
                                                Traits::kGmemKRowsPerThread,
                                                Traits::kGmemKElemsPerLoad>(
        tid, n_block_max, 0, page_size, sequence_block_table, k_outer_stride,
        k_row_stride);
    tKgKFirst.data() = mK.data() + k_thread_tile_base;
  }
  copy_even_tile(gmem_k_copy, tKgKFirst, tKsKFirst);
  __syncthreads();

  for (int n_block = n_block_max; n_block >= n_block_min; --n_block) {
    const int n_warp = warp & 1;
    const int group_row_base = mma_group * Traits::kGroupRows;
    const int qk_row_base = group_row_base + n_warp * Traits::kQkWarpRows;
    auto acc_s = partition_fragment_C(
        qk_tiled_mma, Shape<Int<Traits::kQkWarpRows>, Int<kBlockN>>{});
    clear(acc_s);

#pragma unroll
    for (int d_chunk = 0; d_chunk < Traits::kDChunks; ++d_chunk) {
      auto sK = make_tensor(
          make_smem_ptr(kv_smem_ptr + (d_chunk & 1) * Traits::kKStageElements),
          typename Traits::SmemLayoutK{});
      if (d_chunk + 1 < Traits::kDChunks) {
        auto gKNext = local_tile(mK, Shape<Int<kBlockN>, Int<kDChunk>>{},
                                 make_coord(n_block, d_chunk + 1));
        auto tKgKNextRaw = gmem_k_thread.partition_S(gKNext);
        auto tKgKNext = reshape_kv_thread_tensor<PagedKV>(tKgKNextRaw);
        if constexpr (PagedKV) {
          tKgKNext.data() =
              mK.data() + k_thread_tile_base + (d_chunk + 1) * kDChunk;
        }
        copy_even_tile(gmem_k_copy, tKgKNext, tKrKNext);
      }

      auto sQChunk = local_tile(
          sQ, Shape<Int<Traits::kQkWarpRows>, Int<kDChunk>>{},
          make_coord(mma_group * Traits::kWarpsPerGroup + n_warp, d_chunk));
      auto tSrQ = qk_mma_thread.partition_fragment_A(sQChunk);
      auto tSrK = qk_mma_thread.partition_fragment_B(sK);
      auto tOsQ = qk_mma_thread.partition_A(sQChunk);
      auto tOsK = qk_mma_thread.partition_B(sK);
      auto smem_copy_q =
          make_tiled_copy_A(typename Traits::SmemCopyAtom{}, qk_tiled_mma);
      auto smem_copy_k =
          make_tiled_copy_B(typename Traits::SmemCopyAtom{}, qk_tiled_mma);
      auto smem_thread_q = smem_copy_q.get_thread_slice(lane);
      auto smem_thread_k = smem_copy_k.get_thread_slice(lane);
      auto tSsQ = smem_thread_q.retile_S(tOsQ);
      auto tSsK = smem_thread_k.retile_S(tOsK);
      FLASH_NAMESPACE::gemm<false, false>(
          acc_s, tSrQ, tSrK, tSsQ, tSsK, qk_tiled_mma, smem_copy_q, smem_copy_k,
          smem_thread_q, smem_thread_k);
      if (d_chunk + 1 < Traits::kDChunks) {
        auto sKNext = make_tensor(
            make_smem_ptr(kv_smem_ptr +
                          ((d_chunk + 1) & 1) * Traits::kKStageElements),
            typename Traits::SmemLayoutK{});
        auto tKsKNextRaw = gmem_k_thread.partition_D(sKNext);
        auto tKsKNext = reshape_kv_thread_tensor<PagedKV>(tKsKNextRaw);
        cute::copy(tKrKNext, tKsKNext);
        __syncthreads();
      }
    }

    const int64_t v_batch_offset =
        PagedKV ? 0 : static_cast<int64_t>(batch) * v_outer_stride;
    auto mV =
        make_tensor(make_gmem_ptr(v + v_batch_offset + head_kv * v_head_stride),
                    make_shape(kv_len, Int<Traits::kHeadDim>{}),
                    make_stride(v_row_stride, _1{}));
    auto sV0 =
        make_tensor(make_smem_ptr(kv_smem_ptr), typename Traits::SmemLayoutV{});
    auto sV1 = make_tensor(make_smem_ptr(kv_smem_ptr + Traits::kKVElements),
                           typename Traits::SmemLayoutV{});
    auto tVsV0Raw = gmem_v_thread.partition_D(sV0);
    auto tVsV1Raw = gmem_v_thread.partition_D(sV1);
    auto tVsV0 = reshape_kv_thread_tensor<PagedKV>(tVsV0Raw);
    auto tVsV1 = reshape_kv_thread_tensor<PagedKV>(tVsV1Raw);
    auto tVrV0 = make_fragment_like(tVsV0);
    auto tVrV1 = make_fragment_like(tVsV1);
    auto cV = make_identity_tensor(Shape<Int<kBlockN>, Int<kDChunk>>{});
    auto tVcVRaw = gmem_v_thread.partition_S(cV);
    auto tVcV = reshape_kv_thread_tensor<PagedKV>(tVcVRaw);
    auto gV0 = local_tile(mV, Shape<Int<kBlockN>, Int<kDChunk>>{},
                          make_coord(n_block, 0));
    auto gV2 = local_tile(mV, Shape<Int<kBlockN>, Int<kDChunk>>{},
                          make_coord(n_block, Int<Traits::kOwnedDChunks>{}));
    auto tVgV0Raw = gmem_v_thread.partition_S(gV0);
    auto tVgV2Raw = gmem_v_thread.partition_S(gV2);
    auto tVgV0 = reshape_kv_thread_tensor<PagedKV>(tVgV0Raw);
    auto tVgV2 = reshape_kv_thread_tensor<PagedKV>(tVgV2Raw);
    int64_t v_thread_tile_base = 0;
    if constexpr (PagedKV) {
      v_thread_tile_base = paged_kv_thread_offset<Traits::kGmemThreadsPerRow,
                                                  Traits::kGmemRowsPerThread,
                                                  Traits::kGmemElemsPerLoad>(
          tid, n_block, 0, page_size, sequence_block_table, v_outer_stride,
          v_row_stride);
      tVgV0.data() = mV.data() + v_thread_tile_base;
      tVgV2.data() =
          mV.data() + v_thread_tile_base + Traits::kOwnedDChunks * kDChunk;
    }
    copy_even_tile(gmem_v_copy, tVgV0, tVrV0);
    copy_even_tile(gmem_v_copy, tVgV2, tVrV1);

    FLASH_NAMESPACE::Mask<true, false, false> mask(kv_len, query_len, -1, 0,
                                                   0.0f);
    mask.template apply_mask<true, true>(acc_s, n_block * kBlockN,
                                         query_row_base + qk_row_base, 0);
    Element* p_smem_ptr = kv_smem_ptr + 2 * Traits::kKVElements;
    float* row_scale_exchange =
        reinterpret_cast<float*>(p_smem_ptr + Traits::kPElements);
    splitd_n32_online_softmax(acc_s, o_storage, OLayout{}, row_max, row_sum,
                              row_scale_exchange, mma_group, n_warp, lane,
                              softmax_scale_log2, n_block == n_block_max);

    store_v_fragment_128_swizzled(tVrV0, tVsV0, tVcV);
    store_v_fragment_128_swizzled(tVrV1, tVsV1, tVcV);

    auto sP =
        make_tensor(make_smem_ptr(p_smem_ptr), typename Traits::SmemLayoutP{});
    auto cS =
        make_identity_tensor(Shape<Int<Traits::kQkWarpRows>, Int<kBlockN>>{});
    auto tScS = qk_mma_thread.partition_C(cS);
#pragma unroll
    for (int i = 0; i < size(acc_s); ++i) {
      const int row = get<0>(tScS(i));
      const int col = get<1>(tScS(i));
      sP(qk_row_base + row, col) = Element(acc_s(i));
    }
    __syncthreads();

    auto gV1 = local_tile(mV, Shape<Int<kBlockN>, Int<kDChunk>>{},
                          make_coord(n_block, 1));
    auto gV3 =
        local_tile(mV, Shape<Int<kBlockN>, Int<kDChunk>>{},
                   make_coord(n_block, Int<Traits::kOwnedDChunks + 1>{}));
    auto tVgV1Raw = gmem_v_thread.partition_S(gV1);
    auto tVgV3Raw = gmem_v_thread.partition_S(gV3);
    auto tVgV1 = reshape_kv_thread_tensor<PagedKV>(tVgV1Raw);
    auto tVgV3 = reshape_kv_thread_tensor<PagedKV>(tVgV3Raw);
    if constexpr (PagedKV) {
      tVgV1.data() = mV.data() + v_thread_tile_base + kDChunk;
      tVgV3.data() = mV.data() + v_thread_tile_base +
                     (Traits::kOwnedDChunks + 1) * kDChunk;
    }
    copy_even_tile(gmem_v_copy, tVgV1, tVrV0);
    copy_even_tile(gmem_v_copy, tVgV3, tVrV1);

    auto sPGroup =
        local_tile(sP, Shape<Int<Traits::kGroupRows>, Int<kBlockN>>{},
                   make_coord(mma_group, 0));
    auto tPrP = pv_mma_thread.partition_fragment_A(sPGroup);
    auto tOsP = pv_mma_thread.partition_A(sPGroup);
    auto smem_copy_p =
        make_tiled_copy_A(typename Traits::SmemCopyAtom{}, pv_tiled_mma);
    auto smem_thread_p = smem_copy_p.get_thread_slice(lane);
    auto tPsP = smem_thread_p.retile_S(tOsP);
    auto tPrPView = smem_thread_p.retile_D(tPrP);
#pragma unroll
    for (int k_tile = 0; k_tile < size<2>(tPrP); ++k_tile) {
      cute::copy(smem_copy_p, tPsP(_, _, k_tile), tPrPView(_, _, k_tile));
    }

#pragma unroll
    for (int d_local = 0; d_local < Traits::kOwnedDChunks; ++d_local) {
      auto acc_o =
          make_tensor(make_rmem_ptr(&o_storage[d_local][0]), OLayout{});
      auto sV =
          make_tensor(make_smem_ptr(kv_smem_ptr + n_warp * Traits::kKVElements),
                      typename Traits::SmemLayoutV{});
      splitd_pv_gemm_tt(acc_o, tPrP, sV, pv_tiled_mma, lane);
      __syncthreads();
      if (d_local + 1 < Traits::kOwnedDChunks) {
        store_v_fragment_128_swizzled(tVrV0, tVsV0, tVcV);
        store_v_fragment_128_swizzled(tVrV1, tVsV1, tVcV);
        __syncthreads();
        if (n_block > n_block_min) {
          auto gKNextBlock = local_tile(mK, Shape<Int<kBlockN>, Int<kDChunk>>{},
                                        make_coord(n_block - 1, 0));
          auto tKgKNextBlockRaw = gmem_k_thread.partition_S(gKNextBlock);
          auto tKgKNextBlock =
              reshape_kv_thread_tensor<PagedKV>(tKgKNextBlockRaw);
          if constexpr (PagedKV) {
            k_thread_tile_base =
                paged_kv_thread_offset<Traits::kGmemKThreadsPerRow,
                                       Traits::kGmemKRowsPerThread,
                                       Traits::kGmemKElemsPerLoad>(
                    tid, n_block - 1, 0, page_size, sequence_block_table,
                    k_outer_stride, k_row_stride);
            tKgKNextBlock.data() = mK.data() + k_thread_tile_base;
          }
          copy_even_tile(gmem_k_copy, tKgKNextBlock, tKrKNext);
        }
      }
    }
    if (n_block > n_block_min) {
      auto sKNextBlock = make_tensor(make_smem_ptr(kv_smem_ptr),
                                     typename Traits::SmemLayoutK{});
      auto tKsKNextBlockRaw = gmem_k_thread.partition_D(sKNextBlock);
      auto tKsKNextBlock = reshape_kv_thread_tensor<PagedKV>(tKsKNextBlockRaw);
      cute::copy(tKrKNext, tKsKNextBlock);
      __syncthreads();
    }
  }

  const int n_warp = warp & 1;
  const int group_row_base = mma_group * Traits::kGroupRows;
  Element* p_smem_ptr = kv_smem_ptr + 2 * Traits::kKVElements;
  float* row_sum_exchange =
      reinterpret_cast<float*>(p_smem_ptr + Traits::kPElements) +
      Traits::kExchangeRows;
  SumOp<float> sum_op;
#pragma unroll
  for (int slot = 0; slot < Traits::kQkRowsPerThread; ++slot) {
    row_sum[slot] =
        FLASH_NAMESPACE::sm70_row_allreduce_8(row_sum[slot], sum_op);
  }
  if constexpr (SplitKV3) {
    const int64_t split_row_stride =
        static_cast<int64_t>(gridDim.y / 3) * query_len * heads_q;
    if ((lane & 0x0e) == 0) {
#pragma unroll
      for (int slot = 0; slot < Traits::kQkRowsPerThread; ++slot) {
        const int row =
            FLASH_NAMESPACE::sm70_row_slot<Traits::kQkWarpRows>(slot, lane);
        const int query_row = query_row_base + group_row_base +
                              n_warp * Traits::kQkWarpRows + row;
        const int64_t row_offset =
            (static_cast<int64_t>(batch) * query_len + query_row) * heads_q +
            head_q;
        const int64_t partial_row = split * split_row_stride + row_offset;
        partial_max[partial_row] = row_max[slot];
        partial_sum[partial_row] = row_sum[slot];
      }
    }

#pragma unroll
    for (int d_local = 0; d_local < Traits::kOwnedDChunks; ++d_local) {
      auto acc_o =
          make_tensor(make_rmem_ptr(&o_storage[d_local][0]), OLayout{});
      auto cO =
          make_identity_tensor(Shape<Int<Traits::kGroupRows>, Int<kDChunk>>{});
      auto tOcO = pv_mma_thread.partition_C(cO);
#pragma unroll
      for (int i = 0; i < size(acc_o); ++i) {
        const int row = get<0>(tOcO(i));
        const int col = get<1>(tOcO(i));
        const int query_row = query_row_base + group_row_base + row;
        const int64_t row_offset =
            (static_cast<int64_t>(batch) * query_len + query_row) * heads_q +
            head_q;
        const int64_t partial_row = split * split_row_stride + row_offset;
        const int d =
            (n_warp * Traits::kOwnedDChunks + d_local) * kDChunk + col;
        partial_out[partial_row * Traits::kHeadDim + d] = acc_o(i);
      }
    }
  } else {
    if constexpr (StoreState) {
      if ((lane & 0x0e) == 0) {
#pragma unroll
        for (int slot = 0; slot < Traits::kQkRowsPerThread; ++slot) {
          const int row =
              FLASH_NAMESPACE::sm70_row_slot<Traits::kQkWarpRows>(slot, lane);
          const int query_row = query_row_base + group_row_base +
                                n_warp * Traits::kQkWarpRows + row;
          const int64_t state_row =
              (static_cast<int64_t>(batch) * query_len + query_row) * heads_q +
              head_q;
          // The standalone prefix path stores maxima after applying
          // softmax_scale.  Export the dense-tail state in the same
          // natural-log coordinate so the two states can be merged
          // directly with exp(prefix_max - global_max).
          partial_max[state_row] =
              row_max[slot] * (softmax_scale_log2 * float(M_LN2));
          partial_sum[state_row] = row_sum[slot];
        }
      }
    }
    if ((lane & 0x0e) == 0) {
#pragma unroll
      for (int slot = 0; slot < Traits::kQkRowsPerThread; ++slot) {
        const int row =
            FLASH_NAMESPACE::sm70_row_slot<Traits::kQkWarpRows>(slot, lane);
        row_sum_exchange[mma_group * Traits::kGroupRows +
                         n_warp * Traits::kQkWarpRows + row] = row_sum[slot];
      }
    }
    __syncthreads();

    const int64_t out_batch_offset =
        static_cast<int64_t>(batch) * query_len * heads_q * Traits::kHeadDim;
#pragma unroll
    for (int d_local = 0; d_local < Traits::kOwnedDChunks; ++d_local) {
      auto acc_o =
          make_tensor(make_rmem_ptr(&o_storage[d_local][0]), OLayout{});
      auto acc_o_rc = make_tensor(
          acc_o.data(),
          FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_o.layout()));
#pragma unroll
      for (int row = 0; row < Traits::kOutputRowsPerThread; ++row) {
        const int logical_row =
            FLASH_NAMESPACE::sm70_row_slot<Traits::kGroupRows>(row, lane);
        const float inv_sum =
            1.0f /
            row_sum_exchange[mma_group * Traits::kGroupRows + logical_row];
#pragma unroll
        for (int col = 0; col < size<1>(acc_o_rc); ++col) {
          if constexpr (!Unnormalized) acc_o_rc(row, col) *= inv_sum;
        }
      }

      auto cO =
          make_identity_tensor(Shape<Int<Traits::kGroupRows>, Int<kDChunk>>{});
      auto tOcO = pv_mma_thread.partition_C(cO);
#pragma unroll
      for (int i = 0; i < size(acc_o); ++i) {
        const int row = get<0>(tOcO(i));
        const int col = get<1>(tOcO(i));
        const int query_row = query_row_base + group_row_base + row;
        const int64_t offset =
            out_batch_offset +
            static_cast<int64_t>(query_row) * heads_q * Traits::kHeadDim +
            head_q * Traits::kHeadDim +
            (n_warp * Traits::kOwnedDChunks + d_local) * kDChunk + col;
        out[offset] = std::conditional_t<FloatOutput, float, Element>(acc_o(i));
      }
    }
  }
}

extern "C" cudaError_t onecat_v37_dense_state_float_raw(
    const void* q, const void* k, const void* v, float* state_max,
    float* state_sum, void* out, int query_len, int kv_len, int heads_q,
    int heads_kv, float softmax_scale, int unnormalized, cudaStream_t stream) {
  if (q == nullptr || k == nullptr || v == nullptr || state_max == nullptr ||
      state_sum == nullptr || out == nullptr || query_len <= 0 ||
      kv_len < query_len || heads_q <= 0 || heads_kv <= 0 ||
      heads_q % heads_kv != 0 ||
      query_len % Sm70D256SplitDTraits::kBlockM != 0 ||
      kv_len % Sm70D256SplitDTraits::kBlockN != 0) {
    return cudaErrorInvalidValue;
  }
  const dim3 block(Sm70D256SplitDTraits::kNThreads);
  const dim3 grid(query_len / Sm70D256SplitDTraits::kBlockM, 1, heads_q);
  auto kernel = unnormalized
                    ? sm70_d256_splitd_dense_kernel<cutlass::half_t, false,
                                                    false, true, true, true>
                    : sm70_d256_splitd_dense_kernel<cutlass::half_t, false,
                                                    false, true, true, false>;
  cudaError_t result =
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           Sm70D256SplitDTraits::kSmemBytes);
  if (result != cudaSuccess) {
    return result;
  }
  kernel<<<grid, block, Sm70D256SplitDTraits::kSmemBytes, stream>>>(
      static_cast<const cutlass::half_t*>(q),
      static_cast<const cutlass::half_t*>(k),
      static_cast<const cutlass::half_t*>(v), static_cast<float*>(out),
      query_len * heads_q * Sm70D256SplitDTraits::kHeadDim,
      heads_q * Sm70D256SplitDTraits::kHeadDim, Sm70D256SplitDTraits::kHeadDim,
      kv_len * heads_kv * Sm70D256SplitDTraits::kHeadDim,
      heads_kv * Sm70D256SplitDTraits::kHeadDim, Sm70D256SplitDTraits::kHeadDim,
      kv_len * heads_kv * Sm70D256SplitDTraits::kHeadDim,
      heads_kv * Sm70D256SplitDTraits::kHeadDim, Sm70D256SplitDTraits::kHeadDim,
      query_len, kv_len, heads_q, heads_kv, softmax_scale * float(M_LOG2E),
      nullptr, 0, 0, nullptr, state_max, state_sum, nullptr);
  return cudaPeekAtLastError();
}

// The compact-score operator calls this after joining its prefix and tail.
// Flags cover 64 query tokens (all six heads); only flagged tiles overwrite
// normalized FP32 residuals. Logits, statistics and PV accumulation stay FP32;
// the caller supplies centered/scaled V and restores its output afterwards.
extern "C" cudaError_t onecat_v37_recover_score_tiles_raw(
    const void* q, const void* k, const void* v, void* out,
    const int* recovery_tiles, int query_len, int kv_len, cudaStream_t stream) {
  if (q == nullptr || k == nullptr || v == nullptr || out == nullptr ||
      recovery_tiles == nullptr || query_len <= 0 || query_len % 64 != 0 ||
      kv_len < query_len || kv_len % 32 != 0) {
    return cudaErrorInvalidValue;
  }
  using Traits = Sm70D256SplitDTraits;
  auto kernel = sm70_d256_splitd_dense_kernel<cutlass::half_t, false, false,
                                              false, true, false, true>;
  cudaError_t result = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Traits::kSmemBytes);
  if (result != cudaSuccess) return result;
  kernel<<<dim3(query_len / 64, 1, 6), Traits::kNThreads, Traits::kSmemBytes,
           stream>>>(static_cast<const cutlass::half_t*>(q),
                     static_cast<const cutlass::half_t*>(k),
                     static_cast<const cutlass::half_t*>(v),
                     static_cast<float*>(out), query_len * 6 * 256, 6 * 256,
                     256, kv_len * 256, 256, 256, kv_len * 256, 256, 256,
                     query_len, kv_len, 6, 1, 0.0625f * float(M_LOG2E), nullptr,
                     0, 0, nullptr, nullptr, nullptr, recovery_tiles);
  return cudaPeekAtLastError();
}
}  // namespace FLASH_NAMESPACE
