#pragma once

#include <c10/cuda/CUDAException.h>

#include <cuda_runtime_api.h>
#include <torch/library.h>

#include <cstdint>

#include "cutlass/gemm/gemm.h"

namespace marlin::sm70 {

constexpr int kCtaK = 32;
constexpr int kQuantTileK = 16;
constexpr int kQuantTileN = 64;

template <typename SharedStorage, typename Kernel>
inline size_t configure_sm70_dynamic_smem(Kernel kernel) {
  size_t smem_bytes = sizeof(SharedStorage);
  if (smem_bytes >= (48u << 10)) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)));
  }
  return smem_bytes;
}

template <int CtaM, int CtaN, int Warps>
struct Sm70WarpShape;

template <>
struct Sm70WarpShape<32, 128, 4> {
  using Type = cutlass::gemm::GemmShape<32, 32, 32>;
};

template <>
struct Sm70WarpShape<32, 256, 4> {
  using Type = cutlass::gemm::GemmShape<32, 64, 32>;
};

template <>
struct Sm70WarpShape<64, 64, 4> {
  using Type = cutlass::gemm::GemmShape<32, 32, 32>;
};

template <>
struct Sm70WarpShape<64, 128, 4> {
  using Type = cutlass::gemm::GemmShape<32, 64, 32>;
};

template <>
struct Sm70WarpShape<64, 128, 8> {
  using Type = cutlass::gemm::GemmShape<32, 32, 32>;
};

template <>
struct Sm70WarpShape<64, 256, 4> {
  using Type = cutlass::gemm::GemmShape<64, 64, 32>;
};

template <>
struct Sm70WarpShape<64, 256, 8> {
  using Type = cutlass::gemm::GemmShape<32, 64, 32>;
};

template <>
struct Sm70WarpShape<128, 64, 4> {
  using Type = cutlass::gemm::GemmShape<64, 32, 32>;
};

template <>
struct Sm70WarpShape<128, 64, 8> {
  using Type = cutlass::gemm::GemmShape<32, 32, 32>;
};

template <>
struct Sm70WarpShape<128, 128, 4> {
  using Type = cutlass::gemm::GemmShape<64, 64, 32>;
};

template <>
struct Sm70WarpShape<128, 128, 8> {
  using Type = cutlass::gemm::GemmShape<64, 32, 32>;
};

template <>
struct Sm70WarpShape<128, 256, 8> {
  using Type = cutlass::gemm::GemmShape<64, 64, 32>;
};

template <>
struct Sm70WarpShape<256, 64, 4> {
  using Type = cutlass::gemm::GemmShape<64, 64, 32>;
};

template <>
struct Sm70WarpShape<256, 64, 8> {
  using Type = cutlass::gemm::GemmShape<64, 32, 32>;
};

template <>
struct Sm70WarpShape<256, 128, 8> {
  using Type = cutlass::gemm::GemmShape<64, 64, 32>;
};

struct Sm70CtaGeometry {
  int cta_m;
  int cta_n;
  int warps;
};

inline constexpr char const* kSupportedSm70MarlinCtaGeometries =
    "32x128x4, 32x256x4, 64x64x4, 64x128x4, 64x128x8, "
    "64x256x4, 64x256x8, 128x64x4, 128x64x8, 128x128x4, "
    "128x128x8, 128x256x8, 256x64x4, 256x64x8, and 256x128x8";

inline bool sm70_marlin_cta_geometry_is_supported(
    Sm70CtaGeometry geometry) {
  int const cta_m = geometry.cta_m;
  int const cta_n = geometry.cta_n;
  int const warps = geometry.warps;
  return (cta_m == 32 && cta_n == 128 && warps == 4) ||
         (cta_m == 32 && cta_n == 256 && warps == 4) ||
         (cta_m == 64 && cta_n == 64 && warps == 4) ||
         (cta_m == 64 && cta_n == 128 && warps == 4) ||
         (cta_m == 64 && cta_n == 128 && warps == 8) ||
         (cta_m == 64 && cta_n == 256 && warps == 4) ||
         (cta_m == 64 && cta_n == 256 && warps == 8) ||
         (cta_m == 128 && cta_n == 64 && warps == 4) ||
         (cta_m == 128 && cta_n == 64 && warps == 8) ||
         (cta_m == 128 && cta_n == 128 && warps == 4) ||
         (cta_m == 128 && cta_n == 128 && warps == 8) ||
         (cta_m == 128 && cta_n == 256 && warps == 8) ||
         (cta_m == 256 && cta_n == 64 && warps == 4) ||
         (cta_m == 256 && cta_n == 64 && warps == 8) ||
         (cta_m == 256 && cta_n == 128 && warps == 8);
}

inline int sm70_marlin_dense_auto_cta_n(int64_t size_n) {
  if (size_n % 256 == 0) {
    return 256;
  }
  if (size_n % 128 == 0) {
    return 128;
  }
  if (size_n % 64 == 0) {
    return 64;
  }
  TORCH_CHECK(false, "SM70 Marlin requires size_n divisible "
                     "by 64. Got size_n = ", size_n, ".");
  return 0;
}

inline int sm70_marlin_dense_auto_cta_m(int64_t size_m, int auto_cta_n) {
  if (auto_cta_n == 64) {
    if (size_m >= 256) {
      return 256;
    }
    if (size_m >= 128) {
      return 128;
    }
    return 64;
  }
  if (auto_cta_n == 128) {
    if (size_m >= 256) {
      return 256;
    }
    if (size_m >= 128) {
      return 128;
    }
    if (size_m >= 64) {
      return 64;
    }
    return 32;
  }
  if (auto_cta_n == 256) {
    if (size_m >= 128) {
      return 128;
    }
    if (size_m >= 64) {
      return 64;
    }
    return 32;
  }
  TORCH_CHECK(false, "SM70 Marlin auto CTA_M requires CTA_N in {64, 128, 256}. "
                     "Got CTA_N = ", auto_cta_n, ".");
  return 0;
}

inline int sm70_marlin_dense_auto_warps_for_cta(int auto_cta_m, int auto_cta_n) {
  if (auto_cta_n == 64) {
    if (auto_cta_m == 64) {
      return 4;
    }
    if (auto_cta_m == 128 || auto_cta_m == 256) {
      return 8;
    }
  } else if (auto_cta_n == 128) {
    if (auto_cta_m == 32) {
      return 4;
    }
    if (auto_cta_m == 64 || auto_cta_m == 128 || auto_cta_m == 256) {
      return 8;
    }
  } else if (auto_cta_n == 256) {
    if (auto_cta_m == 32) {
      return 4;
    }
    if (auto_cta_m == 64 || auto_cta_m == 128) {
      return 8;
    }
  }
  TORCH_CHECK(false, "SM70 Marlin auto CTA default warps has no supported "
                     "geometry for CTA_M=", auto_cta_m,
                     ", CTA_N=", auto_cta_n, ".");
  return 0;
}

inline Sm70CtaGeometry sm70_marlin_dense_auto_cta_geometry(int64_t size_m,
                                                        int64_t size_n) {
  int const auto_cta_n = sm70_marlin_dense_auto_cta_n(size_n);
  int const auto_cta_m = sm70_marlin_dense_auto_cta_m(size_m, auto_cta_n);
  return {auto_cta_m, auto_cta_n,
          sm70_marlin_dense_auto_warps_for_cta(auto_cta_m, auto_cta_n)};
}

inline void validate_sm70_marlin_dense_cta_geometry_supported(char const* op_name,
                                                              Sm70CtaGeometry geometry) {
  TORCH_CHECK(sm70_marlin_cta_geometry_is_supported(geometry),
              "Unsupported SM70 Marlin CTA geometry for ", op_name, ": ",
              geometry.cta_m, "x", geometry.cta_n, "x", geometry.warps,
              ". Supported geometries are ",
              kSupportedSm70MarlinCtaGeometries, ".");
}

inline void validate_sm70_marlin_dense_cta_n_alignment(char const* op_name,
                                                       Sm70CtaGeometry geometry,
                                                       int64_t size_n) {
  TORCH_CHECK(
      size_n % geometry.cta_n == 0 && size_n % kQuantTileN == 0,
      "SM70 Marlin requires size_n divisible by both CTA_N and 64 for ",
      op_name, " with CTA geometry ", geometry.cta_m, "x", geometry.cta_n,
      "x", geometry.warps, ". Got size_n = ", size_n, ".");
}

}  // namespace marlin::sm70
