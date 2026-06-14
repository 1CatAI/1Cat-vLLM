#pragma once

#include <cuda_fp16.h>

#include <cstdint>

#include "cutlass/cutlass.h"
#include "quantization/marlin/sm70_marlin_common.cuh"

namespace marlin::sm70 {

template <int ContiguousWords>
struct QwordVector;

template <>
struct QwordVector<1> {
  using Type = uint32_t;
};

template <>
struct QwordVector<2> {
  using Type = uint2;
};

template <>
struct QwordVector<4> {
  using Type = uint4;
};

template <int ContiguousWords>
CUTLASS_DEVICE typename QwordVector<ContiguousWords>::Type load_qword_vector(
    uint32_t const* ptr) {
  static_assert(ContiguousWords == 1 || ContiguousWords == 2 ||
                    ContiguousWords == 4,
                "SM70 qword vector load supports 1, 2, or 4 words.");
  if constexpr (ContiguousWords == 1) {
    return *ptr;
  } else {
    return *reinterpret_cast<typename QwordVector<ContiguousWords>::Type const*>(
        ptr);
  }
}

template <int ContiguousWords>
CUTLASS_DEVICE uint32_t qword_from_vector(
    typename QwordVector<ContiguousWords>::Type const& words, int c) {
  static_assert(ContiguousWords == 1 || ContiguousWords == 2 ||
                    ContiguousWords == 4,
                "SM70 qword vector access supports 1, 2, or 4 words.");
  if constexpr (ContiguousWords == 1) {
    return words;
  } else {
    uint32_t const* words_ptr = reinterpret_cast<uint32_t const*>(&words);
    return words_ptr[c];
  }
}

CUTLASS_DEVICE uint32_t qword_from_vector(uint4 const& words, int c) {
  uint32_t const* words_ptr = reinterpret_cast<uint32_t const*>(&words);
  return words_ptr[c];
}

CUTLASS_DEVICE uint32_t qword_from_vector(uint2 const& words, int c) {
  uint32_t const* words_ptr = reinterpret_cast<uint32_t const*>(&words);
  return words_ptr[c];
}

template <int CtaN>
CUTLASS_DEVICE int u4_cta_n_qweight_offset_from_logical(int size_n,
                                                        int logical_k,
                                                        int logical_n) {
  static_assert(CtaN == 64 || CtaN == 128 || CtaN == 256,
                "SM70 Marlin CTA_N must be 64, 128, or 256.");
  constexpr int kGroupTiles = CtaN / kQuantTileN;
  int const k_tile = logical_k / kQuantTileK;
  int const local_k = logical_k - k_tile * kQuantTileK;
  int const n_tile = logical_n / kQuantTileN;
  int const group_n_tile = n_tile / kGroupTiles;
  int const group_first_n_tile = group_n_tile * kGroupTiles;
  int const subtile = n_tile - group_first_n_tile;
  int const local_n_vec = (logical_n - n_tile * kQuantTileN) / 8;
  int const local_word = local_k * (kQuantTileN / 8) + local_n_vec;

  return k_tile * (size_n * 2) +
         group_n_tile * kGroupTiles * (kQuantTileK * kQuantTileN / 8) +
         local_word * kGroupTiles + subtile;
}

template <int CtaN>
CUTLASS_DEVICE int u8_cta_n_qweight_offset_from_logical(int size_n,
                                                        int logical_k,
                                                        int logical_n) {
  static_assert(CtaN == 64 || CtaN == 128 || CtaN == 256,
                "SM70 Marlin CTA_N must be 64, 128, or 256.");
  constexpr int kGroupTiles = CtaN / kQuantTileN;
  int const k_tile = logical_k / kQuantTileK;
  int const local_k = logical_k - k_tile * kQuantTileK;
  int const n_tile = logical_n / kQuantTileN;
  int const group_n_tile = n_tile / kGroupTiles;
  int const group_first_n_tile = group_n_tile * kGroupTiles;
  int const subtile = n_tile - group_first_n_tile;
  int const local_n_word = (logical_n - n_tile * kQuantTileN) / 4;
  int const local_word = local_k * (kQuantTileN / 4) + local_n_word;

  return k_tile * (size_n * 4) +
         group_n_tile * kGroupTiles * (kQuantTileK * kQuantTileN / 4) +
         local_word * kGroupTiles + subtile;
}

template <int CtaN>
CUTLASS_DEVICE int u8_cta_n_qweight_word_stride_from_logical() {
  static_assert(CtaN == 64 || CtaN == 128 || CtaN == 256,
                "SM70 Marlin CTA_N must be 64, 128, or 256.");
  return CtaN / kQuantTileN;
}

}  // namespace marlin::sm70
