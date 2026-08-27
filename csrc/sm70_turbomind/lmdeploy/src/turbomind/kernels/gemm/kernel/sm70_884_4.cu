// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/arch/config_sm70_s884.h"
#include "src/turbomind/kernels/gemm/registry.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

using namespace sm70_s884;
using namespace cache_policy;
using S = cache_policy::Stream;
using D = cache_policy::Default;

namespace {

// Keep shape-specific small-N tactics out of prefill and unrelated CUDA graph
// shapes.
template <class Gemm, int ExactM>
class ExactMKernelImpl final : public KernelImpl<Gemm> {
 public:
  bool is_feasible(const GemmDesc& desc) const noexcept override {
    return desc.m == ExactM && KernelImpl<Gemm>::is_feasible(desc);
  }
};

template <class Gemm, int ExactM, int ExactN, int ExactK>
class ExactMnkKernelImpl final : public KernelImpl<Gemm> {
 public:
  bool is_feasible(const GemmDesc& desc) const noexcept override {
    return desc.m == ExactM && desc.n == ExactN && desc.k == ExactK &&
           KernelImpl<Gemm>::is_feasible(desc);
  }
};

template <class Gemm>
class Dsv4ExponentFoldedM6KernelImpl final : public KernelImpl<Gemm> {
 public:
  Dsv4ExponentFoldedM6KernelImpl() {
    this->info_.name += "_sm70_mxfp4_efold_dsv4_m6";
  }

  bool is_feasible(const GemmDesc& desc) const noexcept override {
    const bool accepted_shape =
        (desc.n == 1024 && desc.k == 4096) ||
        (desc.n == 4096 && desc.k == 512);
    return desc.m == 6 && desc.num == 1 && accepted_shape &&
           KernelImpl<Gemm>::is_feasible(desc);
  }
};

}  // namespace

void Registry::sm70_884_4() {
  if constexpr (1) {
    // clang-format off
        using C = Config_U4_d<kColMajor>;
        Add<C::Type<128, 256, 16, 2, 4, 1, D, D, 2, true, 1, 128, 128, 128>>();
        Add<C::Type<128, 128, 16, 2, 2, 1, D, D, 2, true, 1, 128, 64, 128>>();
        Add<C::Type<128, 128, 16, 2, 2, 1, D, S, 2, true, 1, 128, 64, 128>>();
        Add<C::Type< 96, 128, 32, 2, 2, 1, D, S, 2, true, 1, 128, 48, 128>>();
        Add<C::Type< 64, 128, 32, 2, 2, 1, D, D, 2, true, 1, 128, 32, 128>>();
        Add<C::Type< 64, 128, 32, 2, 2, 1, D, S, 2, true, 1, 128, 32, 128>>();
        Add<C::Type< 64, 128, 16, 1, 4, 1, D, S, 2, true, 1, 128, 32, 128>>();
        Add<C::Type< 64, 256, 16, 1, 4, 1, D, S, 2, true, 1, 128, 64, 128>>();
        Add<C::Type< 32, 128, 32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type< 32, 256, 32, 1, 4, 1, D, S, 2, true, 1, 128, 32, 128>>();
        Add<C::Type< 16, 128, 32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type< 16, 256, 32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type<  8, 128, 64, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type<  8, 128, 32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type<  8, 256, 64, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type<  8, 256, 64, 1, 4, 2, D, S, 2, true, 1, 128>>();

        using CS = Config_U4_d_A8x64Swizzle<kColMajor>;
        using C32 = CS::Type< 8, 32, 64, 1, 1, 1, D, S, 2, true, 1, 128>;
        using C64 = CS::Type< 8, 64, 64, 1, 2, 1, D, S, 2, true, 1, 128, -1, -1, 2>;
        Add(std::make_unique<ExactMKernelImpl<typename C32::Kernel, 5>>());
        Add(std::make_unique<ExactMKernelImpl<typename C64::Kernel, 5>>());
        Add(std::make_unique<ExactMnkKernelImpl<typename C64::Kernel, 1, 8704, 5120>>());
        Add(std::make_unique<ExactMnkKernelImpl<typename C64::Kernel, 1, 4096, 5120>>());

    // clang-format on
  }

  if constexpr (1) {
    // clang-format off
        // GroupSizeV=128
        using C = Config_U4_g<kColMajor>;
        Add<C::Type<128, 256,  16, 2, 4, 1, D, D, 2,   0 , 1, 128, 128, 128>>();
        Add<C::Type<128, 128,  16, 2, 2, 1, D, D, 2, true, 1, 128,  64, 128>>();
        Add<C::Type< 64, 128,  32, 1, 4, 1, D, S, 2, true, 1, 128,  32, 128>>();
        Add<C::Type< 64, 256,  16, 1, 4, 1, D, S, 2, true, 1, 128,  64, 128>>();
        Add<C::Type< 32, 128,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type< 32, 256,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type< 16, 256,  64, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type< 16, 256,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type< 16, 128,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type< 16, 256,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type<  8, 128,  64, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type<  8, 256,  64, 1, 4, 1, D, S, 2, true, 1, 128>>();
        // GroupSizeV=64
        Add<C::Type<128, 128,  16, 2, 2, 1, D, D, 2, true, 1, 64,  64, 128>>();
        Add<C::Type< 64, 128,  32, 1, 4, 1, D, S, 2, true, 1, 64,  32, 128>>();
        Add<C::Type< 32, 128,  32, 1, 4, 1, D, S, 2, true, 1, 64>>();
        Add<C::Type< 16, 128,  32, 1, 4, 1, D, S, 2, true, 1, 64>>();
        Add<C::Type<  8, 128,  64, 1, 4, 1, D, S, 2, true, 1, 64>>();
        Add<C::Type<  8, 256,  64, 1, 4, 1, D, S, 2, true, 1, 64>>();
        // GroupSizeV=32
        Add<C::Type<128, 128,  16, 2, 2, 1, D, D, 2, true, 1, 32,  64, 128>>();
        Add<C::Type< 64, 128,  32, 1, 4, 1, D, S, 2, true, 1, 32,  32, 128>>();
        Add<C::Type< 32, 128,  32, 1, 4, 1, D, S, 2, true, 1, 32>>();
        Add<C::Type< 16, 128,  32, 1, 4, 1, D, S, 2, true, 1, 32>>();
        Add<C::Type<  8, 128,  64, 1, 4, 1, D, S, 2, true, 1, 32>>();
        Add<C::Type<  8, 256,  64, 1, 4, 1, D, S, 2, true, 1, 32>>();
    // clang-format on
  }

  if constexpr (1) {
    // clang-format off
        using C = Config_MXF4<kColMajor, 0>;
        Add<C::Type<128, 128,  16, 2, 2, 1, D, D, 2, true, 1, 32,  64, 128>>();
        Add<C::Type< 64, 128,  32, 1, 4, 1, D, S, 2, true, 1, 32,  32, 128>>();
        Add<C::Type< 32, 128,  32, 1, 4, 1, D, S, 2, true, 1, 32>>();
        Add<C::Type< 16, 128,  32, 1, 4, 1, D, S, 2, true, 1, 32>>();
        Add<C::Type<  8, 128,  64, 1, 4, 1, D, S, 2, true, 1, 32>>();
    // clang-format on
  }

  if constexpr (1) {
    // Preserve the complete M=6 tactic search space. Dispatch maps the
    // measured ordinary tactic to its exact folded counterpart and stores it
    // separately, so this route cannot poison ordinary MXFP4 selection.
    // clang-format off
        using C = Config_MXF4_ExponentFolded<kColMajor, 0>;
        using C128 = C::Type<128, 128, 16, 2, 2, 1, D, D, 2, true, 1, 32, 64, 128>;
        using C64  = C::Type< 64, 128, 32, 1, 4, 1, D, S, 2, true, 1, 32, 32, 128>;
        using C32  = C::Type< 32, 128, 32, 1, 4, 1, D, S, 2, true, 1, 32>;
        using C16  = C::Type< 16, 128, 32, 1, 4, 1, D, S, 2, true, 1, 32>;
        using C8   = C::Type<  8, 128, 64, 1, 4, 1, D, S, 2, true, 1, 32>;
        Add(std::make_unique<Dsv4ExponentFoldedM6KernelImpl<typename C128::Kernel>>());
        Add(std::make_unique<Dsv4ExponentFoldedM6KernelImpl<typename C64::Kernel>>());
        Add(std::make_unique<Dsv4ExponentFoldedM6KernelImpl<typename C32::Kernel>>());
        Add(std::make_unique<Dsv4ExponentFoldedM6KernelImpl<typename C16::Kernel>>());
        Add(std::make_unique<Dsv4ExponentFoldedM6KernelImpl<typename C8::Kernel>>());
    // clang-format on
  }

  if constexpr (1) {
    // clang-format off
        using C = Config_NVF4<kColMajor, 0>;
        Add<C::Type<128, 128,  16, 2, 2, 1, D, D, 2, true, 1, 16,  64, 128>>();
        Add<C::Type< 64, 128,  32, 1, 4, 1, D, S, 2, true, 1, 16,  32, 128>>();
        Add<C::Type< 32, 128,  32, 1, 4, 1, D, S, 2, true, 1, 16>>();
        Add<C::Type< 16, 128,  32, 1, 4, 1, D, S, 2, true, 1, 16>>();
        Add<C::Type<  8, 128,  64, 1, 4, 1, D, S, 2, true, 1, 16>>();
        using C32K64L1 = C::Type<8, 32, 64, 1, 1, 1, D, S, 2, true, 1, 16>;
        using C32K64L2 = C::Type<8, 32, 64, 1, 1, 1, D, S, 2, true, 1, 16, -1, -1, 2>;
        using C32K128L1 = C::Type<8, 32, 128, 1, 1, 1, D, S, 2, true, 1, 16>;
        using C32K128L2 = C::Type<8, 32, 128, 1, 1, 1, D, S, 2, true, 1, 16, -1, -1, 2>;
        Add(std::make_unique<ExactMnkKernelImpl<typename C32K64L1::Kernel, 1, 8704, 5120>>());
        Add(std::make_unique<ExactMnkKernelImpl<typename C32K64L2::Kernel, 1, 5120, 4352>>());
        Add(std::make_unique<ExactMnkKernelImpl<typename C32K128L1::Kernel, 1, 8704, 5120>>());
        Add(std::make_unique<ExactMnkKernelImpl<typename C32K128L2::Kernel, 1, 5120, 4352>>());
    // clang-format on
  }
}

}  // namespace turbomind::gemm
